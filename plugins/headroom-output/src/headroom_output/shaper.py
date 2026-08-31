"""The façade the host calls: one request body in, one shaped body out.

This is the whole plugin from the outside. Everything below it — wire-format
detection, turn classification, the lever registry, the safety clamps — is an
implementation detail the host never has to know about.

Where this plugs in
-------------------
Today the host must call :meth:`OutputShaper.shape` explicitly, because the
seam this plugin actually wants does not exist yet: ``PipelineEvent`` carries
``messages``, ``tools`` and ``headers``, and every field the shaper writes
(``system``, ``output_config.effort``, ``thinking``, ``text``, ``max_tokens``)
is on the request body, which the event does not expose. So the integration is
a direct call for now and becomes an event handler the moment the pipeline
grows a ``PRE_SEND_PARAMS`` stage carrying the body.

Both halves of that future contract are already the shape of this class:
:meth:`shape` is the request side, :meth:`observe` is the response side. When
the seam lands, wiring is renaming two call sites rather than a rewrite.

Ordering
--------
:meth:`shape` must run **after** every other body mutation the host performs.
The turn classifier reads the final message list, and a compression pass that
rewrites tool results after classification would leave the shaper deciding on a
request that no longer exists. The core makes the same ordering choice for the
same reason.

Holdout
-------
A conversation is assigned once to ``treatment`` or ``control`` and stays there.
Two reasons that happen to align: mixing shaped and unshaped turns inside one
conversation pollutes the comparison, and flipping the system-prompt tail
mid-conversation invalidates the provider's prefix cache — which costs far more
than the experiment is worth. Conversation-stable assignment is not a nicety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    ConversationFeatures,
    LeverMode,
    OutcomeReading,
    TurnFeatures,
)
from .registry import LeverRegistry, ShapePlan, apply_param_decision

log = logging.getLogger(__name__)

__all__ = ["OutputShaper", "ShapeOutcome", "build_default_shaper"]


@dataclass
class ShapeOutcome:
    """What :meth:`OutputShaper.shape` did to one request."""

    changed: bool = False
    labels: list[str] = field(default_factory=list)
    arm: str = "treatment"
    stratum: str = ""
    plan: ShapePlan | None = None
    considered: tuple[str, ...] = ()
    """Levers whose declarations matched this request.

    Empty means no lever in the registry knows how to help this wire format and
    model — the coverage gap that let a whole client see zero savings without
    anything in the data saying so. Hosts should surface this, not swallow it.
    """


class OutputShaper:
    """Applies registered levers to provider request bodies."""

    def __init__(
        self,
        registry: LeverRegistry,
        *,
        holdout_fraction: float = 0.0,
        shadow_ledger: Any | None = None,
    ) -> None:
        self._registry = registry
        self._holdout = max(0.0, min(1.0, holdout_fraction))
        self._shadow = shadow_ledger

    # -- request side -----------------------------------------------------

    def shape(
        self,
        body: dict[str, Any],
        *,
        conversation_key: str | None = None,
        harness: str = "unknown",
        input_tokens: int = 0,
    ) -> ShapeOutcome:
        """Shape ``body`` in place. Returns what happened.

        Never raises: a failure anywhere in shaping degrades to "this request
        was not shaped", because the alternative is a savings feature costing a
        user their turn.
        """
        try:
            return self._shape_inner(
                body,
                conversation_key=conversation_key,
                harness=harness,
                input_tokens=input_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — shaping must never break a request
            log.warning("output shaping failed; request forwarded unshaped: %s", exc, exc_info=True)
            return ShapeOutcome(changed=False)

    def _shape_inner(
        self,
        body: dict[str, Any],
        *,
        conversation_key: str | None,
        harness: str,
        input_tokens: int,
    ) -> ShapeOutcome:
        from . import turns, wire
        from .stats import assign_arm, stratum_key

        payload = _unwrap_envelope(body)
        fmt = turns.detect_wire_format(payload)
        kind = turns.classify(payload, fmt)
        model = str(payload.get("model") or "")

        key = conversation_key or _fallback_conversation_key(payload, model)
        arm = assign_arm(key, self._holdout)
        stratum = stratum_key(
            model=model,
            turn_kind=kind.value,
            input_tokens=input_tokens,
            has_tools=bool(payload.get("tools")),
            harness=harness,
        )

        conversation = ConversationFeatures(
            conversation_key=key,
            wire_format=fmt,
            model=model,
            harness=harness,
        )
        turn = TurnFeatures(
            conversation_key=key,
            wire_format=fmt,
            model=model,
            turn_kind=kind,
            input_tokens=input_tokens,
            has_tools=bool(payload.get("tools")),
            turn_index=turns.turn_index(payload, fmt),
            harness=harness,
            present_fields=turns.present_fields(payload, fmt),
            metadata={"stratum": stratum},
        )

        plan = self._registry.plan(conversation, turn)
        outcome = ShapeOutcome(
            arm=arm,
            stratum=stratum,
            plan=plan,
            considered=plan.considered,
        )

        # The control arm is planned but never applied. Planning it anyway is
        # deliberate: the shadow ledger then sees what *would* have happened to
        # control conversations too, which is the only way to sanity-check that
        # the two arms are actually comparable.
        if arm != "treatment":
            outcome.labels.append(f"control:{stratum}")
            return outcome

        applied = apply_param_decision(payload, plan.params, fmt)
        if applied:
            outcome.changed = True
            outcome.labels.extend(applied)

        suffix = plan.prompt.system_suffix
        if suffix and wire.apply_system_suffix(payload, suffix, fmt):
            outcome.changed = True
            if plan.prompt.label:
                outcome.labels.append(plan.prompt.label)

        # Shadow decisions never reach the body; they ride the label channel so
        # a host can record them beside the real ones.
        outcome.labels.extend(lb for lb in plan.params.labels if lb.startswith("shadow:"))
        outcome.labels.append(f"stratum:{stratum}")
        return outcome

    # -- response side ----------------------------------------------------

    def observe(
        self,
        *,
        conversation_key: str,
        output_tokens: int,
        stop_reason: str | None = None,
        thinking_tokens: int | None = None,
        turn_index: int = 0,
        labels: tuple[str, ...] = (),
    ) -> None:
        """Feed one response back to the levers that adapt.

        This is what closes the control loop. ``stop_reason`` is the signal
        that matters: it is provider-supplied and unambiguous, unlike the
        interrupt and fast-skip signals the core's verbosity controller was
        designed around — which is precisely why that controller, though
        complete and tested, was never wired to anything.
        """
        reading = OutcomeReading(
            conversation_key=conversation_key,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            thinking_tokens=thinking_tokens,
            turn_index=turn_index,
            labels=labels,
        )
        self._registry.observe(reading)

    @property
    def registry(self) -> LeverRegistry:
        return self._registry


def _unwrap_envelope(body: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the Responses ``response.create`` envelope, if present.

    The WebSocket Responses transport nests the real request under
    ``{"type": "response.create", "response": {...}}``. Shaping the envelope
    instead of the request is a silent no-op, so unwrap once at the boundary.
    """
    inner = body.get("response")
    if body.get("type") == "response.create" and isinstance(inner, dict):
        return inner
    return body


def _fallback_conversation_key(payload: dict[str, Any], model: str) -> str:
    """Derive a conversation-stable key when the host supplies none.

    Prefers an identifier the harness already carries; falls back to hashing
    the first user message, which is stable for the life of a conversation
    because history is append-only. A host that knows its own session id should
    pass it instead — this is the last resort, not the intended path.
    """
    import hashlib

    for key in ("conversation_id", "session_id", "thread_id"):
        value = payload.get(key)
        if isinstance(value, str) and value and value.lower() != "auto":
            return f"{key}:{value}"

    seed = model
    for msg in payload.get("messages") or ():
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                seed += "\x00" + content[:512]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        seed += "\x00" + str(block.get("text", ""))[:512]
                        break
            break
    return hashlib.sha256(seed.encode("utf-8", "ignore")).hexdigest()


def build_default_shaper(
    *,
    verbosity_level: int = 2,
    holdout_fraction: float = 0.0,
    max_tokens_mode: LeverMode = LeverMode.SHADOW,
) -> OutputShaper:
    """Assemble the stock lever set.

    ``max_tokens`` defaults to :attr:`LeverMode.SHADOW` on purpose. It is the
    only lever here that can truncate a real answer, so it ships computing its
    decision and applying nothing until a bind rate from real traffic says it
    is safe. Shipping a new lever straight to LIVE is how you find out from a
    customer instead of from your own ledger.
    """
    from .levers import build_default_levers

    registry = LeverRegistry()
    for lever in build_default_levers(verbosity_level=verbosity_level):
        if hasattr(lever, "decide_conversation"):
            registry.register_prompt(lever)
        else:
            # Keyed on capability, not on a lever's name. Any lever that can
            # cap a token ceiling can truncate a real answer, so it inherits
            # the cautious default — including a third-party one this factory
            # has never heard of. Matching on the name would have silently
            # promoted a renamed lever to LIVE, which is the exact failure
            # this default exists to prevent.
            can_truncate = lever.descriptor.requires_present == "max_tokens"
            mode = max_tokens_mode if can_truncate else LeverMode.LIVE
            registry.register_param(lever, mode)
    return OutputShaper(registry, holdout_fraction=holdout_fraction)
