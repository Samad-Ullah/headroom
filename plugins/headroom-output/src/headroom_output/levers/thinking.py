"""Legacy thinking-budget clamp for models still sending ``budget_tokens``.

Modern effort routing (see ``effort.py``) covers ``output_config.effort``, but
some clients still speak the older ``thinking: {type: "enabled", budget_tokens}``
form. That budget is a hard token allowance for thinking, billed as output, so
clamping it to the API floor on a mechanical continuation is the direct
equivalent of lowering effort for a client that has no effort field to lower.

The ``type`` field itself is never touched — disabling thinking mid-conversation
while prior turns carry thinking blocks in history 400s on some models, and a
type toggle would bust the messages cache tier regardless. Only the numeric
budget moves, and only downward (the executor enforces that; this lever just
names the floor).
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

#: Anthropic's minimum accepted ``thinking.budget_tokens``. Below this the API
#: rejects the request outright, so it is the safest possible clamp target.
API_FLOOR = 1024


class ThinkingBudgetLever:
    """Clamps ``thinking.budget_tokens`` to the API floor on mechanical turns."""

    def __init__(self, floor: int = API_FLOOR) -> None:
        self.floor = floor
        self.descriptor = LeverDescriptor(
            name="thinking_budget",
            summary=f"Clamp thinking.budget_tokens to the {floor}-token API floor.",
            wire_formats=(WireFormat.ANTHROPIC,),
            turn_kinds=(TurnKind.MECHANICAL_CONTINUATION,),
            requires_present="thinking_budget",
        )

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        """Always request the floor — the executor only applies it if lower."""
        return ParamDecision(thinking_budget=self.floor, labels=(f"thinking_budget:{self.floor}",))

    def observe(self, outcome: OutcomeReading) -> None:
        """Stateless: a fixed floor has nothing to learn from an outcome."""
        return None
