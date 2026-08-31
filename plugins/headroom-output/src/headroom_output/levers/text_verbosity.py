"""OpenAI ``text.verbosity`` lever — the gpt-5-only counterpart of steering.

``text.verbosity`` exists on gpt-5 and nowhere else; it is not a portable
concept the way the prompt-based steering block is, so it lives in its own
param lever gated by ``model_prefixes=("gpt-5",)`` rather than being folded
into verbosity steering. This is the "capability, not lowest common
denominator" principle from contract.py: a lever that only some models
support declares that narrowly instead of being watered down to fit everyone.
"""

from __future__ import annotations

from ..contract import LeverDescriptor, OutcomeReading, ParamDecision, TurnFeatures, WireFormat


class TextVerbosityLever:
    """Lowers OpenAI ``text.verbosity`` to a fixed target on gpt-5 requests."""

    def __init__(self, target: str = "low") -> None:
        self.target = target
        self.descriptor = LeverDescriptor(
            name="text_verbosity",
            summary=f"Lower OpenAI text.verbosity to {target!r} on gpt-5 requests.",
            wire_formats=(WireFormat.OPENAI_CHAT, WireFormat.OPENAI_RESPONSES),
            model_prefixes=("gpt-5",),
            requires_present="text_verbosity",
        )

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        """Always request ``target`` — the executor clamps to lower-only."""
        return ParamDecision(text_verbosity=self.target, labels=(f"text_verbosity:{self.target}",))

    def observe(self, outcome: OutcomeReading) -> None:
        """Stateless: a fixed target has nothing to learn from an outcome."""
        return None
