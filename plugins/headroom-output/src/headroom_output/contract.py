"""The output-shaping contract: pure data in, pure decisions out.

This module is the load-bearing part of the plugin. Everything else is an
implementation of one of the protocols defined here, and the *shape* of these
types is what keeps the plugin safe to hand to third parties.

The cache-key law
-----------------
A provider's prefix cache is keyed on the model, the system prompt, the tool
definitions, and the message history. Changing any of those mid-conversation
invalidates the cached prefix. On a 100k-token context that costs roughly
$0.27 (the prefix re-reads at $3.00/1M instead of $0.30/1M) against the ~$0.005
of output a terser instruction might have saved — about 60x worse than doing
nothing at all.

Everything *outside* the cache key — ``effort``, ``thinking.budget_tokens``,
``text.verbosity``, ``max_tokens``, ``tool_choice``, sampling params — is free
to change on every single turn.

That partition is expressed here as two decision types, not as documentation:

* :class:`PromptDecision` is produced **once per conversation** and replayed
  byte-for-byte on every later turn. It is the only place prompt text can come
  from.
* :class:`ParamDecision` is produced **per turn** and has no field capable of
  carrying prompt text. A lever author physically cannot bust the prefix cache
  through it.

The expensive mistake is unrepresentable rather than merely discouraged. That
is the whole reason for two types where one would have been simpler.

Capability, not lowest common denominator
-----------------------------------------
Output-shaping levers are not portable. ``output_config.effort`` is Anthropic;
``reasoning.effort`` is OpenAI Responses and has a ``minimal`` floor Anthropic
lacks; ``text.verbosity`` exists on gpt-5 and nowhere else. A previous attempt
at a portable design left GitHub Copilot with zero output savings because the
one lever that generalised was also the weakest one.

So a lever declares the wire formats and models it applies to
(:class:`LeverDescriptor`), and the registry asks "which levers apply to this
request?" rather than assuming a common denominator. Coverage gaps then show up
as an empty lever list that the caller can report, instead of as silence.

Purity
------
Levers never touch a request body. They receive plain features and return plain
decisions; the executor owns every mutation, and enforces the safety rules
(lowered-only, never-inject) that stop a lever from causing a provider 400.
This mirrors ``headroom.transforms.compressor_registry``: pure data in, pure
data out, no Python-only objects across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ConversationFeatures",
    "LeverDescriptor",
    "LeverMode",
    "OutcomeReading",
    "ParamDecision",
    "PromptDecision",
    "ParamLever",
    "PromptLever",
    "TurnFeatures",
    "TurnKind",
    "WireFormat",
]


class WireFormat(str, Enum):
    """The request shape being shaped.

    Not the same axis as the provider: Anthropic-format requests reach Bedrock,
    Vertex and Foundry, and the OpenAI chat format is spoken by a dozen
    non-OpenAI upstreams. Levers care about the wire shape, because that is
    what determines which fields exist.
    """

    ANTHROPIC = "anthropic"
    """Anthropic Messages: top-level ``system``, ``output_config.effort``."""

    OPENAI_CHAT = "openai_chat"
    """OpenAI ``/v1/chat/completions``: system prompt is a ``messages`` entry."""

    OPENAI_RESPONSES = "openai_responses"
    """OpenAI Responses: ``instructions`` string, ``reasoning.effort``."""


class TurnKind(str, Enum):
    """Structural classification of the latest turn.

    Determined from block types, roles and error flags only — never from
    content heuristics — so classification is deterministic and testable
    without a model in the loop.
    """

    NEW_USER_ASK = "new_user_ask"
    """A human is waiting on this response and will read it."""

    MECHANICAL_CONTINUATION = "mechanical_continuation"
    """A clean tool result came back; the next act is almost always another
    tool call. Nobody reads the prose here, which makes it the cheapest place
    in the whole conversation to spend fewer tokens."""

    ERROR_CONTINUATION = "error_continuation"
    """A tool returned an error. Deliberately excluded from tightening: this is
    where the model most needs room to reason."""

    UNKNOWN = "unknown"
    """Could not classify. Treated exactly like ``NEW_USER_ASK`` — the safe
    direction, because under-shaping costs money and over-shaping costs trust."""


class LeverMode(str, Enum):
    """How a lever's decision is used.

    ``SHADOW`` is a first-class mode, not a rollout step. Every lever computes
    its decision and records what it *would* have done long before it is
    allowed to change a live request. The point is to answer "how often would
    this have truncated someone?" from real traffic, at zero user risk, rather
    than from an argument.
    """

    OFF = "off"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class LeverDescriptor:
    """Static declaration of what a lever is and where it applies.

    Attributes:
        name: Unique, stable identifier. Appears in attribution labels, so
            renaming one orphans its history.
        summary: One line, shown by ``headroom output levers``.
        wire_formats: Formats this lever can act on. A lever that only knows
            how to write ``output_config`` declares ``ANTHROPIC`` alone.
        model_prefixes: Lowercased model-id prefixes this lever applies to.
            Empty means "any model on the declared wire formats".
        turn_kinds: Turn kinds this lever acts on. Empty means all.
        mutates_cache_key: True only for levers that touch the model, system
            prompt, tools or history. Such a lever MUST be conversation-scoped;
            the registry refuses to run it per-turn.
        requires_present: Field the client must already have sent for this
            lever to act. Enforces the never-inject rule at declaration time:
            a lever that lowers ``effort`` declares it, so a model that does
            not accept the parameter never receives one.
    """

    name: str
    summary: str
    wire_formats: tuple[WireFormat, ...]
    model_prefixes: tuple[str, ...] = ()
    turn_kinds: tuple[TurnKind, ...] = ()
    mutates_cache_key: bool = False
    requires_present: str | None = None

    def applies_to(self, wire: WireFormat, model: str, kind: TurnKind) -> bool:
        """Whether this lever is a candidate for one request.

        Candidacy is not permission — the executor still enforces
        ``requires_present`` and the lowered-only rules against the real body.
        This is the cheap declarative filter that runs first.
        """
        if wire not in self.wire_formats:
            return False
        if self.model_prefixes:
            m = (model or "").lower()
            if not any(m.startswith(p) for p in self.model_prefixes):
                return False
        if self.turn_kinds and kind not in self.turn_kinds:
            return False
        return True


@dataclass(frozen=True)
class ConversationFeatures:
    """What a prompt lever may decide from. Stable across the conversation.

    Deliberately excludes anything that changes turn to turn. A prompt lever
    that could see the turn kind would be tempted to vary its output by it,
    which is the 60x mistake. It cannot see what it must not react to.
    """

    conversation_key: str
    wire_format: WireFormat
    model: str
    harness: str = "unknown"
    """Which agent is wrapped ("claude", "codex", "copilot", ...). Carried so
    coverage can be measured per harness — the gap that let Copilot see zero
    savings go unnoticed until someone filed an issue."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnFeatures:
    """What a param lever may decide from. Recomputed every turn."""

    conversation_key: str
    wire_format: WireFormat
    model: str
    turn_kind: TurnKind
    input_tokens: int
    has_tools: bool
    turn_index: int = 0
    """0-based position in the conversation. Enables position-aware shaping:
    an output token emitted early is re-sent as input on every remaining turn,
    so cutting it early is worth substantially more than cutting it late."""

    harness: str = "unknown"
    present_fields: frozenset[str] = frozenset()
    """Which shapeable fields the client actually sent. The executor derives
    this from the body so levers never have to inspect one."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptDecision:
    """A prompt lever's output. Computed once per conversation, then frozen.

    The registry caches this against ``conversation_key`` and replays the same
    bytes on every later turn. A lever returning different text for the same
    key is a bug the registry will detect and report rather than obey.
    """

    system_suffix: str | None = None
    """Text appended after the last system block, i.e. after any
    ``cache_control`` breakpoint. ``None`` means this lever declines."""

    label: str | None = None
    """Attribution label, e.g. ``verbosity:L3``. Appears in the stratum so the
    ledger can tell which decision produced which outcome."""


@dataclass(frozen=True)
class ParamDecision:
    """A param lever's output. Every field is OUTSIDE the provider cache key.

    There is deliberately no field here that can carry prompt text, a model id,
    or a tool definition. That absence is the safety property: a third-party
    lever cannot invalidate a prefix cache through this type, however it is
    written.

    All numeric and ranked fields are *requests to lower*. The executor ignores
    any value that would raise the client's own setting, and ignores every
    field whose target the client did not send. A lever cannot cause a 400 by
    introducing a parameter the model does not support.
    """

    effort: str | None = None
    """Target for ``output_config.effort`` / ``reasoning.effort``."""

    thinking_budget: int | None = None
    """Target for ``thinking.budget_tokens``."""

    text_verbosity: str | None = None
    """Target for OpenAI ``text.verbosity``."""

    max_tokens: int | None = None
    """Target ceiling for ``max_tokens`` / ``max_output_tokens``."""

    labels: tuple[str, ...] = ()
    """Attribution labels for whatever this decision changed."""

    def is_empty(self) -> bool:
        """True when the lever declined to act."""
        return (
            self.effort is None
            and self.thinking_budget is None
            and self.text_verbosity is None
            and self.max_tokens is None
        )


@dataclass(frozen=True)
class OutcomeReading:
    """What actually happened. Read-only, supplied by the core's meter.

    Levers consume this to adapt (a truncation means the cap was too tight),
    but can never write it. Plugins declare what they did; the core records
    what happened; attribution is the core's arithmetic over both. A lever able
    to write its own control group could report any number it liked.
    """

    conversation_key: str
    output_tokens: int
    stop_reason: str | None = None
    thinking_tokens: int | None = None
    """Separate from ``output_tokens`` where the provider reports it. Effort
    routing cuts thinking; steering cuts visible text. Pooled into one counter,
    neither lever can be attributed — which is the state of the core today."""

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turn_index: int = 0
    labels: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        """Whether the response was cut off by a token ceiling.

        The unambiguous, provider-supplied feedback signal — the thing that
        makes a closed control loop possible for ``max_tokens`` where the
        verbosity signals (interrupt, fast-skip) needed transcript inference
        and were never wired up.
        """
        return self.stop_reason in ("max_tokens", "length")

    @property
    def visible_tokens(self) -> int:
        """Output tokens excluding thinking, when the split is available."""
        if self.thinking_tokens is None:
            return self.output_tokens
        return max(0, self.output_tokens - self.thinking_tokens)


@runtime_checkable
class PromptLever(Protocol):
    """A lever that contributes prompt text, once per conversation."""

    descriptor: LeverDescriptor

    def decide_conversation(self, features: ConversationFeatures) -> PromptDecision:
        """Return the text to append, or an empty decision to decline."""
        ...


@runtime_checkable
class ParamLever(Protocol):
    """A lever that adjusts request parameters, every turn."""

    descriptor: LeverDescriptor

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        """Return the parameter targets, or an empty decision to decline."""
        ...

    def observe(self, outcome: OutcomeReading) -> None:
        """Feed back what happened. Default implementations may ignore it.

        This is the seam that closes the control loop: a lever that sets a
        ceiling learns from truncations, without the registry needing to know
        anything about how it adapts.
        """
        ...
