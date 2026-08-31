"""The five output-shaping levers and their default-configuration factory.

Each class here satisfies either :class:`~headroom_output.contract.PromptLever`
or :class:`~headroom_output.contract.ParamLever` structurally (no base class
required — the registry checks via ``hasattr``/``runtime_checkable``). See
each module's docstring for what it does and why.
"""

from __future__ import annotations

from .effort import EffortLever
from .max_tokens import AdaptiveMaxTokensLever
from .steering import VerbosityLever
from .text_verbosity import TextVerbosityLever
from .thinking import ThinkingBudgetLever

__all__ = [
    "AdaptiveMaxTokensLever",
    "EffortLever",
    "TextVerbosityLever",
    "ThinkingBudgetLever",
    "VerbosityLever",
    "build_default_levers",
]


def build_default_levers(
    *,
    verbosity_level: int = 2,
) -> tuple[
    VerbosityLever,
    EffortLever,
    ThinkingBudgetLever,
    TextVerbosityLever,
    AdaptiveMaxTokensLever,
]:
    """Instantiate all five levers with their default configuration.

    Order is stable and matches the module list above: the one prompt lever
    first, then the three fixed-target param levers, then the one stateful
    param lever. Callers that register in this order get deterministic
    ``registry.names()`` output and deterministic log ordering.
    """
    return (
        VerbosityLever(level=verbosity_level),
        EffortLever(),
        ThinkingBudgetLever(),
        TextVerbosityLever(),
        AdaptiveMaxTokensLever(),
    )
