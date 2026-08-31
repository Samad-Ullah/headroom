"""Effort routing: lower reasoning effort where nobody is reading the prose.

A mechanical continuation — a clean tool result comes back, the next act is
almost always another tool call — is the cheapest place in a conversation to
spend fewer output tokens, because thinking bills as output and harnesses
routinely pin effort high for every turn regardless of turn kind.

ERROR_CONTINUATION is deliberately not in ``turn_kinds`` below. A tool error
is exactly where the model most needs room to reason about what went wrong;
tightening effort there would save a handful of tokens on the turn most
likely to need a considered retry, which is a bad trade every time. See
``TurnKind.ERROR_CONTINUATION``'s own docstring in contract.py — this lever
just honors it.

The registry has already confirmed, before ``decide_turn`` runs, that (a) the
turn is a mechanical continuation and (b) the client sent an ``effort``-like
field for this wire format (``requires_present``). This lever does not
re-check either: it only returns a target, and the executor clamps it to
lower-only against whatever the client actually sent.
"""

from __future__ import annotations

from ..contract import (
    LeverDescriptor,
    OutcomeReading,
    ParamDecision,
    TurnFeatures,
    TurnKind,
    WireFormat,
)


class EffortLever:
    """Requests a lower ``effort`` / ``reasoning.effort`` on mechanical turns."""

    def __init__(self, target: str = "low") -> None:
        self.target = target
        self.descriptor = LeverDescriptor(
            name="effort",
            summary=f"Lower effort to {target!r} on mechanical tool-result continuations.",
            wire_formats=(
                WireFormat.ANTHROPIC,
                WireFormat.OPENAI_CHAT,
                WireFormat.OPENAI_RESPONSES,
            ),
            turn_kinds=(TurnKind.MECHANICAL_CONTINUATION,),
            requires_present="effort",
        )

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        """Always request ``target`` — candidacy was already decided upstream."""
        return ParamDecision(effort=self.target, labels=(f"effort:{self.target}",))

    def observe(self, outcome: OutcomeReading) -> None:
        """Stateless: a fixed target has nothing to learn from an outcome."""
        return None
