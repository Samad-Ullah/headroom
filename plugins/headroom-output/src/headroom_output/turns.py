"""Structural turn classification for the output-shaping plugin.

Every param lever in this plugin decides from :class:`~.contract.TurnFeatures`,
and the single most consequential field on that type is ``turn_kind``: it is
what tells a lever "the human is not reading this" (a mechanical tool-result
continuation, safe to tighten) from "the human is about to read this"
(a new ask, or an error the model needs room to reason about). Getting that
classification wrong in either direction is expensive — under-classifying a
new ask as mechanical clips a real answer, over-classifying a continuation as
a new ask forfeits savings — so it has to be exact and it has to be cheap
enough to run on every single turn.

The core (``headroom/proxy/output_turn_policy.py`` and the Responses-specific
half of ``headroom/proxy/output_shaper.py``) already solved this problem once,
by classifying purely from structure: block types, roles, and JSON-shaped
error markers, never prose content. Content heuristics would need a model in
the loop (defeating the point of a cheap request-side lever) and would be
non-deterministic across paraphrases of the same tool failure. This module
ports that structural classifier so a third-party lever gets the same
guarantee without importing from ``headroom.proxy`` directly — the whole point
of the plugin boundary in ``contract.py``.

This module also owns the two questions that have to be answered before
classification can even start: which wire format is this request (three
providers reuse the word "messages" for three different shapes), and which
shapeable fields did the client actually send. The latter is the concrete
mechanism behind the contract's never-inject rule — a lever can only ask to
lower ``effort`` if ``present_fields`` says the client already sent one — so
it is deliberately conservative: any shape it cannot positively confirm is
omitted rather than guessed.
"""

from __future__ import annotations

import json
from typing import Any

from .contract import TurnKind, WireFormat

__all__ = [
    "classify",
    "detect_wire_format",
    "present_fields",
    "turn_index",
]

# ---------------------------------------------------------------------------
# response.create envelope
# ---------------------------------------------------------------------------


# Codex's WebSocket transport wraps a Responses payload in a
# ``{"type": "response.create", "response": {...}}`` envelope. Every function
# below reads request-shaped fields (``input``, ``instructions``, ``messages``,
# ...) straight off the body, so unwrapping once up front means the rest of
# this module never has to know the envelope exists.
def _unwrap_response_create(body: dict[str, Any]) -> dict[str, Any]:
    response = body.get("response")
    if body.get("type") == "response.create" and isinstance(response, dict):
        return response
    return body


# ---------------------------------------------------------------------------
# Wire-format detection
# ---------------------------------------------------------------------------

# Fields that only ever appear on an OpenAI chat/completions body. Presence of
# any one, alongside a `messages` list, backs up the role-based signal below
# for a system message stripped down to a bare {"role", "content"} pair with
# nothing else distinguishing it.
_OPENAI_CHAT_SHAPE_FIELDS = frozenset(
    {
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "response_format",
        "logit_bias",
        "seed",
        "parallel_tool_calls",
        "modalities",
        "audio",
        "prediction",
        "stream_options",
        "max_completion_tokens",
        "n",
    }
)


def _has_role_based_system_message(messages: list[Any]) -> bool:
    """True when a ``system``/``developer`` role message sits in ``messages``.

    This is the decisive structural signal for OpenAI chat: Anthropic never
    carries system instructions as a message, only as the top-level ``system``
    field, so a role-based system message inside ``messages`` cannot be an
    Anthropic-format body.
    """
    return any(
        isinstance(message, dict) and message.get("role") in ("system", "developer")
        for message in messages
    )


def detect_wire_format(body: dict[str, Any]) -> WireFormat:
    """Identify which of the three request shapes ``body`` is.

    Order matters: the Responses check runs first because ``input`` is the
    one field no Anthropic or OpenAI chat body ever sends, so its presence
    settles the question outright regardless of anything else in the body.
    Only once Responses is ruled out does distinguishing Anthropic from
    OpenAI chat — both of which use a ``messages`` list — become necessary.
    """
    body = _unwrap_response_create(body)
    if not isinstance(body, dict):
        return WireFormat.ANTHROPIC

    if "input" in body or isinstance(body.get("instructions"), str):
        return WireFormat.OPENAI_RESPONSES

    messages = body.get("messages")
    if isinstance(messages, list):
        has_system_message = _has_role_based_system_message(messages)
        has_chat_fields = any(field in body for field in _OPENAI_CHAT_SHAPE_FIELDS)
        if has_system_message or (messages and has_chat_fields):
            return WireFormat.OPENAI_CHAT

    # Anthropic is the default for anything with (or without) a `messages`
    # list that didn't positively match one of the OpenAI shapes above — the
    # safe direction, since Anthropic-format levers are the most conservative
    # about what they touch.
    return WireFormat.ANTHROPIC


# ---------------------------------------------------------------------------
# Anthropic turn classification
# ---------------------------------------------------------------------------


def _classify_anthropic(messages: Any) -> TurnKind:
    """Classify the latest Anthropic-style turn from message structure only.

    Verbatim port of ``headroom.proxy.output_turn_policy.classify_turn``.
    """
    if not isinstance(messages, list) or not messages:
        return TurnKind.UNKNOWN
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return TurnKind.UNKNOWN

    content = last.get("content")
    if isinstance(content, str):
        return TurnKind.NEW_USER_ASK if content.strip() else TurnKind.UNKNOWN
    if not isinstance(content, list) or not content:
        return TurnKind.UNKNOWN

    saw_tool_result = False
    saw_error = False
    for block in content:
        if not isinstance(block, dict):
            return TurnKind.UNKNOWN
        btype = block.get("type")
        if btype == "tool_result":
            saw_tool_result = True
            if block.get("is_error") is True:
                saw_error = True
        elif btype == "text":
            return TurnKind.NEW_USER_ASK
        elif btype in ("image", "document"):
            return TurnKind.NEW_USER_ASK

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_result:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


# ---------------------------------------------------------------------------
# OpenAI Responses turn classification
# ---------------------------------------------------------------------------

# Trailing `input` item types that represent tool output coming back to the
# model — the Responses counterpart of an Anthropic `tool_result` block.
_RESPONSES_TOOL_OUTPUT_TYPES = frozenset(
    {
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)


def _structural_error_sniff(output: Any) -> bool:
    """Structural error sniff on a tool-output payload.

    Neither the Responses format nor OpenAI chat's ``tool`` messages carry an
    ``is_error`` flag, but agent harnesses encode failure structurally in the
    output payload: a JSON object with a nonzero ``exit_code``,
    ``success: false``, or a truthy ``error`` field. Only those JSON fields are
    inspected — never prose content. Ported from
    ``headroom.proxy.output_shaper._responses_tool_output_is_error`` and
    reused for OpenAI chat's ``tool``/``function`` messages, which encode
    failure the same way.
    """
    data: Any = output
    if isinstance(output, str):
        stripped = output.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            return False
    if not isinstance(data, dict):
        return False
    # Direct fields, plus the common {"output": ..., "metadata": {...}} nesting.
    scopes: list[dict[str, Any]] = [data]
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        scopes.append(metadata)
    for scope in scopes:
        exit_code = scope.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
            return True
        if scope.get("success") is False:
            return True
        if scope.get("error"):
            return True
    return False


def _classify_responses(input_data: Any) -> TurnKind:
    """Classify a Responses request's turn from its ``input`` field.

    Mirrors :func:`_classify_anthropic` semantics on the Responses item list:
    the trailing run of tool-output items decides the turn. A trailing user
    message is a new ask; tool outputs are mechanical unless any carries a
    structural error marker. Purely structural — item types and JSON fields,
    no content regexes.

    Verbatim port of ``headroom.proxy.output_shaper.classify_responses_turn``
    (the error-sniffing variant — deliberately not the older, error-blind
    ``classify_openai_responses_input`` in ``output_turn_policy.py``, which
    predates the structural error sniff and is superseded by this one).
    """
    if isinstance(input_data, str):
        return TurnKind.NEW_USER_ASK if input_data.strip() else TurnKind.UNKNOWN
    if not isinstance(input_data, list) or not input_data:
        return TurnKind.UNKNOWN

    saw_tool_output = False
    saw_error = False
    for item in reversed(input_data):
        if not isinstance(item, dict):
            return TurnKind.UNKNOWN
        itype = item.get("type")
        if itype in _RESPONSES_TOOL_OUTPUT_TYPES:
            saw_tool_output = True
            if _structural_error_sniff(item.get("output")):
                saw_error = True
            continue
        # First non-tool-output item ends the trailing run.
        if saw_tool_output:
            break
        if itype == "message" or (itype is None and "role" in item):
            role = item.get("role")
            if role == "user":
                return TurnKind.NEW_USER_ASK
            return TurnKind.UNKNOWN
        return TurnKind.UNKNOWN

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_output:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


# ---------------------------------------------------------------------------
# OpenAI chat turn classification
# ---------------------------------------------------------------------------

# `tool` is the current chat/completions role for a tool result; `function`
# is the deprecated predecessor some older clients still send.
_CHAT_TOOL_ROLES = frozenset({"tool", "function"})

# Content-part types that mean a human supplied fresh material this turn.
_CHAT_USER_PART_TYPES = frozenset({"text", "image_url", "input_audio", "file"})


def _classify_openai_chat(messages: Any) -> TurnKind:
    """Classify the latest OpenAI chat/completions turn from message shape.

    The core has no chat-specific turn classifier today — ``shape_openai_chat_
    request`` only applies verbosity steering, never effort routing, so no
    classifier was ever needed there. This is a structural analogue of
    :func:`_classify_anthropic`, adapted to chat's shape: a trailing ``user``
    message with text/media content is a new ask; a trailing ``tool`` (or
    legacy ``function``) message is the chat counterpart of a clean
    ``tool_result``, promoted to an error continuation by the same structural
    sniff used for Responses. Still purely structural — no prose content is
    inspected.
    """
    if not isinstance(messages, list) or not messages:
        return TurnKind.UNKNOWN
    last = messages[-1]
    if not isinstance(last, dict):
        return TurnKind.UNKNOWN

    role = last.get("role")
    if role == "user":
        content = last.get("content")
        if isinstance(content, str):
            return TurnKind.NEW_USER_ASK if content.strip() else TurnKind.UNKNOWN
        if isinstance(content, list) and content:
            for part in content:
                if not isinstance(part, dict):
                    return TurnKind.UNKNOWN
                if part.get("type") in _CHAT_USER_PART_TYPES:
                    return TurnKind.NEW_USER_ASK
            return TurnKind.UNKNOWN
        return TurnKind.UNKNOWN

    if role in _CHAT_TOOL_ROLES:
        if _structural_error_sniff(last.get("content")):
            return TurnKind.ERROR_CONTINUATION
        return TurnKind.MECHANICAL_CONTINUATION

    return TurnKind.UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(body: dict[str, Any], wire: WireFormat) -> TurnKind:
    """Classify the latest turn of ``body``, dispatching on ``wire``.

    Returns :class:`contract.TurnKind` — never the core's own
    ``output_turn_policy.TurnKind`` — so a lever author needs only this
    plugin's types and never reaches into ``headroom.proxy``.
    """
    body = _unwrap_response_create(body)
    if wire is WireFormat.ANTHROPIC:
        return _classify_anthropic(body.get("messages"))
    if wire is WireFormat.OPENAI_RESPONSES:
        return _classify_responses(body.get("input"))
    if wire is WireFormat.OPENAI_CHAT:
        return _classify_openai_chat(body.get("messages"))
    return TurnKind.UNKNOWN


def present_fields(body: dict[str, Any], wire: WireFormat) -> frozenset[str]:
    """Which shapeable fields the client actually sent.

    This is the concrete mechanism behind the contract's never-inject rule
    (see :attr:`contract.LeverDescriptor.requires_present`): a lever can only
    be handed a slot to lower ``effort`` if this function first confirms the
    client already sent one, on a wire format where that field is meaningful.
    Deliberately conservative — any shape this function cannot positively
    confirm is omitted, never guessed, because omission just forfeits a
    saving while a wrong guess can hand a lever a field the client never
    sent.
    """
    body = _unwrap_response_create(body)
    fields: set[str] = set()

    if wire is WireFormat.OPENAI_RESPONSES:
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            fields.add("effort")
    elif wire is WireFormat.ANTHROPIC:
        output_config = body.get("output_config")
        if isinstance(output_config, dict) and isinstance(output_config.get("effort"), str):
            fields.add("effort")
    # OPENAI_CHAT has no effort-shaped field in the core today, so it is
    # never added here regardless of what a chat body happens to contain.

    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            fields.add("thinking_budget")

    text_config = body.get("text")
    if isinstance(text_config, dict) and isinstance(text_config.get("verbosity"), str):
        fields.add("text_verbosity")

    max_key = "max_output_tokens" if wire is WireFormat.OPENAI_RESPONSES else "max_tokens"
    max_value = body.get(max_key)
    if isinstance(max_value, int) and not isinstance(max_value, bool) and max_value > 0:
        fields.add("max_tokens")

    return frozenset(fields)


def turn_index(body: dict[str, Any], wire: WireFormat) -> int:
    """Count assistant turns already in the history — 0 on the first turn.

    Enables the position-aware shaping described on
    :attr:`contract.TurnFeatures.turn_index`: an output token emitted early is
    re-sent as input on every remaining turn, so cutting it early is worth
    substantially more than cutting it late.
    """
    body = _unwrap_response_create(body)

    if wire is WireFormat.OPENAI_RESPONSES:
        input_data = body.get("input")
        if not isinstance(input_data, list):
            return 0
        return sum(
            1
            for item in input_data
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        )

    # Anthropic and OpenAI chat both carry history as a `messages` list keyed
    # by `role`, so the same count applies to either.
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
