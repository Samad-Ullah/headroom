"""Output-token shaping for Headroom.

Headroom's core compresses what goes *into* a model. This plugin reduces what
comes *out* of one, which is where the money is: across the provider price
table an output token costs 5x a fresh input token and 50x a cached one, and
in an agentic loop with a warm prefix cache the comparison that matters is the
second one.

The governing constraint is the provider cache key. Model, system prompt, tool
definitions and message history are all part of it, so changing any of them
mid-conversation invalidates the cached prefix — roughly $0.27 on a 100k
context, against the ~$0.005 of output a terser instruction might save. Every
other knob (``effort``, ``thinking.budget_tokens``, ``text.verbosity``,
``max_tokens``) sits outside the cache key and is free to change every turn.

That partition is enforced by the types in :mod:`headroom_output.contract`:
prompt text can only come from a decision made once per conversation, and the
per-turn decision type has no field capable of carrying it. The expensive
mistake is unrepresentable rather than merely documented.

Quick start::

    from headroom_output import build_default_shaper

    shaper = build_default_shaper(verbosity_level=2, holdout_fraction=0.05)
    outcome = shaper.shape(body, conversation_key=session_id, harness="claude")
    # ... send the request, then:
    shaper.observe(
        conversation_key=session_id,
        output_tokens=usage.output_tokens,
        stop_reason=response.stop_reason,
    )
"""

from __future__ import annotations

from .contract import (
    ConversationFeatures,
    LeverDescriptor,
    LeverMode,
    OutcomeReading,
    ParamDecision,
    ParamLever,
    PromptDecision,
    PromptLever,
    TurnFeatures,
    TurnKind,
    WireFormat,
)
from .registry import LeverRegistry, ShapePlan, apply_param_decision, merge_param_decisions
from .shaper import OutputShaper, ShapeOutcome, build_default_shaper

__version__ = "0.1.0"

__all__ = [
    "ConversationFeatures",
    "LeverDescriptor",
    "LeverMode",
    "LeverRegistry",
    "OutcomeReading",
    "OutputShaper",
    "ParamDecision",
    "ParamLever",
    "PromptDecision",
    "PromptLever",
    "ShapeOutcome",
    "ShapePlan",
    "TurnFeatures",
    "TurnKind",
    "WireFormat",
    "__version__",
    "apply_param_decision",
    "build_default_shaper",
    "merge_param_decisions",
]
