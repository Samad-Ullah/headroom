"""Wire-format-specific system-prompt injection for prompt levers.

A :class:`~.contract.PromptLever` returns plain text — the contract will not
even let it see a request body (:class:`~.contract.ConversationFeatures`
carries no body reference at all). Something still has to take that text and
land it in the one place each wire format keeps its system prompt, and three
providers keep it in three different shapes: a top-level ``system`` field
that can be ``None``, a string, or a content-block list (Anthropic); a
``role: "system"``/``"developer"`` message buried inside ``messages`` (OpenAI
chat); and a flat ``instructions`` string (OpenAI Responses). This module is
that landing gear, ported from ``headroom/proxy/output_steering.py`` and
``headroom/proxy/output_verbosity_policy.py``.

Two properties matter more than the mechanics:

* **Cache-prefix preservation.** All three injectors append after whatever
  the client already sent rather than rewriting it, so an earlier
  ``cache_control`` breakpoint (Anthropic) or the bulk of an existing prompt
  (chat, Responses) is left untouched and the provider's cached prefix stays
  hot. This is the same law :mod:`.contract` names outright: touching the
  system prompt mid-conversation is the 60x mistake, so a prompt lever's text
  is decided once and replayed byte-for-byte — the appended block itself must
  therefore also be byte-stable, which is what the sentinel wrapping below
  buys.
* **Idempotent replacement.** The sentinel lets a repeated call recognise and
  replace its own previous block instead of stacking a new one underneath it
  every turn, which would grow the prompt without bound and drift the cache
  key on every single request instead of once.

Every defensive guard from the original code is preserved, including the ones
that look unnecessary until the client sends the one malformed body that
proves they aren't (see the block-text guard in
:func:`_apply_anthropic_system_suffix`).
"""

from __future__ import annotations

from typing import Any

from .contract import WireFormat

__all__ = [
    "STEERING_SENTINEL",
    "STEERING_SUFFIX",
    "apply_system_suffix",
    "replace_or_append_steering_block",
]

# Sentinel prefix marks the appended block so application is idempotent and
# the block is recognizable in logs/diffs.
STEERING_SENTINEL = "<headroom_output_shaping>"
STEERING_SUFFIX = "</headroom_output_shaping>"


def replace_or_append_steering_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing steering block in text, or append one at the tail.

    Verbatim port of
    ``headroom.proxy.output_verbosity_policy.replace_or_append_steering_block``.
    Shared by the OpenAI chat and Responses injectors below, both of which
    hold their system prompt as a single string that this function can search
    and splice directly.
    """
    start = existing.find(STEERING_SENTINEL)
    if start >= 0:
        end = existing.find(STEERING_SUFFIX, start)
        end = len(existing) if end < 0 else end + len(STEERING_SUFFIX)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip("\n")
        parts = [part for part in (prefix, block, suffix) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing


def _apply_anthropic_system_suffix(body: dict[str, Any], block: str) -> bool:
    """Append after the last system block of an Anthropic ``system`` field.

    Appending after the last block keeps any ``cache_control`` breakpoint on
    an earlier block intact: the cached prefix is unchanged and only the
    small, byte-stable steering block is reprocessed.

    Verbatim port of ``headroom.proxy.output_steering.apply_verbosity_steering``.
    """
    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": block}]
        return True
    if isinstance(system, str):
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": block},
        ]
        return True
    if isinstance(system, list):
        for entry in system:
            # Guard the text is a string before ``startswith``: a malformed
            # client block (``{"type": "text", "text": null}``) would otherwise
            # raise ``AttributeError`` here and 500 the request. The OpenAI chat
            # sibling below guards this exact case too.
            entry_text = entry.get("text") if isinstance(entry, dict) else None
            if isinstance(entry_text, str) and entry_text.startswith(STEERING_SENTINEL):
                if entry_text == block:
                    return False
                entry["text"] = block
                return True
        system.append({"type": "text", "text": block})
        return True
    return False


def _apply_openai_chat_system_suffix(body: dict[str, Any], block: str) -> bool:
    """Append or replace the steering block in an OpenAI chat/completions body.

    OpenAI ``/v1/chat/completions`` carries the system prompt as a
    ``role: "system"`` (or ``"developer"``) message inside ``messages`` rather
    than a top-level field, so it needs its own injector (the Anthropic
    ``system`` and Responses ``instructions`` variants do not reach it — the
    root cause of GitHub Copilot CLI seeing zero output savings, #2302).

    The block is appended to the tail of the LAST system/developer message so
    a treatment conversation's steering stays byte-stable across turns (and
    re-applies idempotently via the sentinel). When the request carries no
    system message at all, one is inserted at the front. Returns True only
    when the body actually changed.

    Verbatim port of
    ``headroom.proxy.output_steering.apply_openai_chat_verbosity_steering``.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False

    target: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            target = message
    if target is None:
        # No system prompt to append to — insert one carrying just the block.
        messages.insert(0, {"role": "system", "content": block})
        return True

    content = target.get("content")
    if content is None:
        target["content"] = block
        return True
    if isinstance(content, str):
        updated, changed = replace_or_append_steering_block(content, block)
        if changed:
            target["content"] = updated
        return changed
    if isinstance(content, list):
        # OpenAI also accepts a content-part list ([{"type": "text", ...}]).
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and part["text"].startswith(STEERING_SENTINEL)
            ):
                if part["text"] == block:
                    return False
                part["text"] = block
                return True
        content.append({"type": "text", "text": block})
        return True
    return False


def _apply_openai_responses_system_suffix(body: dict[str, Any], block: str) -> bool:
    """Append or replace the steering block in OpenAI Responses ``instructions``.

    ``instructions`` is the Responses cache hot zone: the appended block is
    byte-stable, so within a conversation every shaped turn sends identical
    instructions bytes and the provider prefix cache stays hot after the
    first shaped turn (the same contract as the Anthropic system-tail
    append).

    Verbatim port of
    ``headroom.proxy.output_steering.apply_openai_responses_verbosity_steering``.
    """
    instructions = body.get("instructions")
    if instructions is None:
        body["instructions"] = block
        return True
    if not isinstance(instructions, str):
        return False

    updated, changed = replace_or_append_steering_block(instructions, block)
    if changed:
        body["instructions"] = updated
    return changed


def apply_system_suffix(body: dict[str, Any], text: str, wire: WireFormat) -> bool:
    """Append ``text`` to ``body``'s system prompt, dispatching on ``wire``.

    ``text`` is the raw prompt text a lever decided
    (:attr:`contract.PromptDecision.system_suffix`) — this function wraps it
    in the sentinel and suffix so every injector below can find and replace
    its own previous block on a later turn instead of stacking a duplicate
    underneath it. Returns True only when the body actually changed, so a
    caller can tell an idempotent no-op (the block was already there,
    byte-identical) from a real mutation.
    """
    block = f"{STEERING_SENTINEL}\n{text}\n{STEERING_SUFFIX}"

    if wire is WireFormat.ANTHROPIC:
        return _apply_anthropic_system_suffix(body, block)
    if wire is WireFormat.OPENAI_CHAT:
        return _apply_openai_chat_system_suffix(body, block)
    if wire is WireFormat.OPENAI_RESPONSES:
        return _apply_openai_responses_system_suffix(body, block)
    return False
