"""Lever registry and executor — the only code here that touches a request body.

Levers are pure: features in, :class:`ParamDecision` / :class:`PromptDecision`
out. Everything that could break a live request lives in this module instead,
written once and tested once:

* **Never-inject.** A parameter the client did not send is never created. This
  is not politeness — models that do not accept ``output_config.effort`` return
  400 when it appears, so injection turns a savings feature into an outage.
  Enforced from :attr:`LeverDescriptor.requires_present` plus
  :attr:`TurnFeatures.present_fields`.
* **Lowered-only.** Every numeric and ranked target may move a value down and
  never up. A lever asking for ``effort="max"`` is ignored, not obeyed.
* **Cache-key discipline.** A lever declaring ``mutates_cache_key`` is refused
  a per-turn slot outright, and prompt decisions are memoised per conversation
  so the same bytes are replayed on every later turn.
* **Isolation.** A lever that raises is disabled for the process and the
  request proceeds without it. One bad lever must not cost a user their turn.

Merging
-------
Several levers may target the same field. The merge rule is always "the safest
lower value wins": the minimum for token counts, the lowest rank for ordered
enums. Two levers can therefore never fight — the outcome does not depend on
registration order, which keeps the executor deterministic and the ledger's
attribution meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from .contract import (
    ConversationFeatures,
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

log = logging.getLogger(__name__)

__all__ = ["LeverRegistry", "ShapePlan", "merge_param_decisions"]

#: Ordered efforts, lowest first. The Responses format adds ``minimal`` below
#: Anthropic's floor, so one table spans both and the executor clamps to what
#: the client actually sent rather than to a format-specific floor.
_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_TEXT_VERBOSITY_ORDER = ("low", "medium", "high")


def _rank(order: tuple[str, ...], value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return order.index(value)
    except ValueError:
        return None


def _lower_of(order: tuple[str, ...], a: str | None, b: str | None) -> str | None:
    """The lower-ranked of two ordered values; either may be ``None``."""
    ra, rb = _rank(order, a), _rank(order, b)
    if ra is None:
        return b if rb is not None else None
    if rb is None:
        return a
    return a if ra <= rb else b


def merge_param_decisions(decisions: list[ParamDecision]) -> ParamDecision:
    """Combine decisions so the safest lower value wins, order-independently."""
    effort: str | None = None
    text_verbosity: str | None = None
    thinking_budget: int | None = None
    max_tokens: int | None = None
    labels: list[str] = []

    for d in decisions:
        effort = _lower_of(_EFFORT_ORDER, effort, d.effort)
        text_verbosity = _lower_of(_TEXT_VERBOSITY_ORDER, text_verbosity, d.text_verbosity)
        if d.thinking_budget is not None:
            thinking_budget = (
                d.thinking_budget
                if thinking_budget is None
                else min(thinking_budget, d.thinking_budget)
            )
        if d.max_tokens is not None:
            max_tokens = d.max_tokens if max_tokens is None else min(max_tokens, d.max_tokens)
        labels.extend(d.labels)

    return ParamDecision(
        effort=effort,
        thinking_budget=thinking_budget,
        text_verbosity=text_verbosity,
        max_tokens=max_tokens,
        labels=tuple(labels),
    )


@dataclass
class ShapePlan:
    """What the registry decided, before anything is applied.

    Returned by :meth:`LeverRegistry.plan` so a caller can inspect, log or
    discard it. Shadow mode is exactly "produce the plan, skip the apply", so
    the shadow path and the live path compute identically and cannot drift.
    """

    params: ParamDecision = field(default_factory=ParamDecision)
    prompt: PromptDecision = field(default_factory=PromptDecision)
    considered: tuple[str, ...] = ()
    """Levers whose descriptors matched this request."""

    skipped: tuple[str, ...] = ()
    """Levers that matched but declined, or were held back by mode/guards."""

    shadow_only: tuple[str, ...] = ()
    """Levers in SHADOW mode: their decision is recorded, never applied."""

    @property
    def empty(self) -> bool:
        return self.params.is_empty() and self.prompt.system_suffix is None


class LeverRegistry:
    """Holds levers, plans a request, and applies the plan safely."""

    def __init__(self) -> None:
        self._prompt: dict[str, tuple[PromptLever, LeverMode]] = {}
        self._param: dict[str, tuple[ParamLever, LeverMode]] = {}
        self._prompt_cache: dict[str, PromptDecision] = {}
        self._broken: set[str] = set()

    # -- registration -----------------------------------------------------

    def register_prompt(self, lever: PromptLever, mode: LeverMode = LeverMode.LIVE) -> None:
        self._prompt[lever.descriptor.name] = (lever, mode)

    def register_param(self, lever: ParamLever, mode: LeverMode = LeverMode.LIVE) -> None:
        """Register a per-turn lever.

        A lever declaring ``mutates_cache_key`` is refused here. Per-turn
        cache-key changes are the single most expensive mistake in this
        subsystem, so the registry will not host one however it is declared.
        """
        if lever.descriptor.mutates_cache_key:
            raise ValueError(
                f"lever {lever.descriptor.name!r} declares mutates_cache_key=True and "
                "cannot be a per-turn lever: changing the model, system prompt, tools "
                "or history mid-conversation invalidates the provider prefix cache, "
                "which costs far more than any output saving. Register it as a prompt "
                "lever (decided once per conversation) instead."
            )
        self._param[lever.descriptor.name] = (lever, mode)

    def set_mode(self, name: str, mode: LeverMode) -> bool:
        for table in (self._prompt, self._param):
            if name in table:
                lever, _ = table[name]
                table[name] = (lever, mode)
                return True
        return False

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({*self._prompt, *self._param}))

    # -- planning ---------------------------------------------------------

    def plan(
        self,
        conversation: ConversationFeatures,
        turn: TurnFeatures,
    ) -> ShapePlan:
        """Decide what should happen to this request. Mutates nothing."""
        considered: list[str] = []
        skipped: list[str] = []
        shadow: list[str] = []
        live_params: list[ParamDecision] = []
        shadow_params: list[ParamDecision] = []

        prompt = self._plan_prompt(conversation, considered, skipped, shadow)

        for name, (lever, mode) in sorted(self._param.items()):
            d = lever.descriptor
            if not d.applies_to(turn.wire_format, turn.model, turn.turn_kind):
                continue
            considered.append(name)
            if mode is LeverMode.OFF or name in self._broken:
                skipped.append(name)
                continue
            # Never-inject: the client must already have sent the field.
            if d.requires_present and d.requires_present not in turn.present_fields:
                skipped.append(name)
                continue
            try:
                decision = lever.decide_turn(turn)
            except Exception as exc:  # noqa: BLE001 — one bad lever must not cost a turn
                log.warning("output lever %r failed and was disabled: %s", name, exc, exc_info=True)
                self._broken.add(name)
                skipped.append(name)
                continue
            if decision.is_empty():
                skipped.append(name)
                continue
            if mode is LeverMode.SHADOW:
                shadow.append(name)
                shadow_params.append(decision)
            else:
                live_params.append(decision)

        merged = merge_param_decisions(live_params)
        if shadow_params:
            # Shadow decisions are carried as labels only, so a caller can
            # record what would have happened without any chance of applying it.
            ghost = merge_param_decisions(shadow_params)
            merged = replace(
                merged,
                labels=merged.labels + tuple(f"shadow:{lb}" for lb in ghost.labels),
            )

        return ShapePlan(
            params=merged,
            prompt=prompt,
            considered=tuple(considered),
            skipped=tuple(skipped),
            shadow_only=tuple(shadow),
        )

    def _plan_prompt(
        self,
        conversation: ConversationFeatures,
        considered: list[str],
        skipped: list[str],
        shadow: list[str],
    ) -> PromptDecision:
        """Resolve the conversation-scoped prompt decision, memoised.

        The memo is the mechanism that enforces byte-stability: the first turn
        of a conversation decides, and every later turn replays. A lever that
        would have returned something different is not consulted again, so
        drift cannot reach the wire.
        """
        cached = self._prompt_cache.get(conversation.conversation_key)
        if cached is not None:
            return cached

        chosen = PromptDecision()
        for name, (lever, mode) in sorted(self._prompt.items()):
            d = lever.descriptor
            # Prompt levers are conversation-scoped, so turn kind is not a
            # filter here — deliberately: reacting to it is the 60x mistake.
            if not d.applies_to(conversation.wire_format, conversation.model, TurnKind.UNKNOWN):
                continue
            considered.append(name)
            if mode is LeverMode.OFF or name in self._broken:
                skipped.append(name)
                continue
            try:
                decision = lever.decide_conversation(conversation)
            except Exception as exc:  # noqa: BLE001
                log.warning("output lever %r failed and was disabled: %s", name, exc, exc_info=True)
                self._broken.add(name)
                skipped.append(name)
                continue
            if decision.system_suffix is None:
                skipped.append(name)
                continue
            if mode is LeverMode.SHADOW:
                shadow.append(name)
                continue
            chosen = decision
            break  # one prompt suffix per conversation; first match wins

        self._prompt_cache[conversation.conversation_key] = chosen
        return chosen

    # -- feedback ---------------------------------------------------------

    def observe(self, outcome: OutcomeReading) -> None:
        """Hand an outcome to every param lever that wants it."""
        for name, (lever, _mode) in self._param.items():
            if name in self._broken:
                continue
            observe = getattr(lever, "observe", None)
            if observe is None:
                continue
            try:
                observe(outcome)
            except Exception as exc:  # noqa: BLE001 — feedback must never break a response
                log.warning("output lever %r observe() failed: %s", name, exc)

    def forget_conversation(self, conversation_key: str) -> None:
        """Drop a memoised prompt decision (bounded-memory housekeeping)."""
        self._prompt_cache.pop(conversation_key, None)


def apply_param_decision(
    body: dict[str, Any],
    decision: ParamDecision,
    wire: WireFormat,
) -> list[str]:
    """Apply a merged decision to a request body. Returns applied labels.

    Every write here is guarded twice: the field must already exist (so a
    parameter is never introduced), and the new value must be strictly lower
    than the client's (so shaping can only ever reduce spend). A decision that
    passes neither guard is a no-op, which is why a misbehaving lever degrades
    to "no savings" rather than to a broken request.
    """
    applied: list[str] = []

    if decision.effort is not None:
        container_key = "reasoning" if wire is WireFormat.OPENAI_RESPONSES else "output_config"
        container = body.get(container_key)
        if isinstance(container, dict):
            current = container.get("effort")
            cr, nr = _rank(_EFFORT_ORDER, current), _rank(_EFFORT_ORDER, decision.effort)
            if cr is not None and nr is not None and nr < cr:
                container["effort"] = decision.effort
                applied.append(f"effort:{current}->{decision.effort}")

    if decision.thinking_budget is not None:
        thinking = body.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            current = thinking.get("budget_tokens")
            if isinstance(current, int) and decision.thinking_budget < current:
                thinking["budget_tokens"] = decision.thinking_budget
                applied.append(f"thinking_budget:{current}->{decision.thinking_budget}")

    if decision.text_verbosity is not None:
        text_cfg = body.get("text")
        if isinstance(text_cfg, dict):
            current = text_cfg.get("verbosity")
            cr = _rank(_TEXT_VERBOSITY_ORDER, current)
            nr = _rank(_TEXT_VERBOSITY_ORDER, decision.text_verbosity)
            if cr is not None and nr is not None and nr < cr:
                text_cfg["verbosity"] = decision.text_verbosity
                applied.append(f"text_verbosity:{current}->{decision.text_verbosity}")

    if decision.max_tokens is not None:
        key = "max_output_tokens" if wire is WireFormat.OPENAI_RESPONSES else "max_tokens"
        current = body.get(key)
        if isinstance(current, int) and 0 < decision.max_tokens < current:
            body[key] = decision.max_tokens
            applied.append(f"max_tokens:{current}->{decision.max_tokens}")

    return applied
