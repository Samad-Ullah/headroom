"""Adaptive ``max_tokens`` ceiling, learned online from output-token history.

Why this lever could be built when the core's verbosity autotuner could not
------------------------------------------------------------------------------
A closed control loop needs an unambiguous failure signal. The core's
verbosity controller (``headroom/proxy/output_shaper.py`` and friends) wanted
to tighten steering automatically but needed to infer "was this response too
short to be useful?" from things like a user interrupt or an immediate
fast-skip reply — signals that require transcript inference, are ambiguous
(a user might interrupt for reasons having nothing to do with verbosity), and
were consequently never wired up as a live loop.

``max_tokens`` has no such problem. The provider tells you directly, in
``stop_reason``, whether the response was cut off (``"max_tokens"`` /
``"length"``). That single field is sufficient, unambiguous, and free — no
transcript inference required. That is the entire reason this lever gets a
real closed loop and the verbosity lever does not.

The statistics
--------------
Each stratum (see :func:`_default_stratum_key`) keeps an online accumulator of
observed ``output_tokens`` — count, sum, sum-of-squares, matching the
``_Accum`` formula in ``headroom/proxy/output_savings.py`` so the two stay
comparable. From that we derive a cap:

    cap = (mean + k * sigma) * multiplier

``mean + k*sigma`` (``k`` defaults to 3.0) covers the overwhelming majority of
a roughly-normal output-length distribution for that stratum, so truncating a
genuinely long-but-legitimate response is rare by construction, not by luck.
``multiplier`` starts at 1.0 and is the AIMD state described below. A stratum
never proposes a cap below ``floor`` (default 256) or before it has seen
``min_samples`` (default 50) observations — three data points is not a
distribution, and capping off one would truncate real users on nothing more
than noise.

The control loop: AIMD
-----------------------
``observe()`` is where the loop closes:

* Every observation feeds the stratum's accumulator, whether or not this
  lever's cap was even applied that turn — the accumulator's job is to learn
  the true shape of the distribution, not just the shape conditioned on
  already being capped.
* A truncation (``outcome.truncated``) means the cap was too tight *right
  now*, for real, on a real request. That is the expensive event, so the
  response is immediate and large: multiply the multiplier by ``widen_factor``
  (default 1.5) and enter a cooldown of ``cooldown_observations`` (default 20)
  further observations during which the multiplier is not tightened. This is
  the "back off fast" half of AIMD.
* Once cooldown has elapsed, each subsequent non-truncating observation decays
  the multiplier back toward 1.0 by ``probe_decay`` (default 0.9), never
  undershooting 1.0. This is the "probe back slowly" half: a single widen
  event takes on the order of tens of clean observations to fully unwind,
  which is deliberate — a cap that flaps back to tight the moment traffic
  looks calm again would just truncate the next similar request.

``truncation_rate`` is the published safety metric for the lever as a whole:
if it drifts up, the statistics (or ``k``) need revisiting before this lever
should be trusted further.

Purity
------
No file I/O happens here. ``to_dict()`` / ``from_dict()`` make the lever's
state round-trippable so a separate persistence layer (owned elsewhere) can
save and restore it across process restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contract import (
    LeverDescriptor,
    OutcomeReading,
    ParamDecision,
    TurnFeatures,
    WireFormat,
)

#: Coarse input-token buckets, matching
#: ``headroom/proxy/output_savings_policy.input_bucket`` exactly so strata
#: computed here and in the core's offline baseline stay comparable.
_INPUT_BUCKETS = (2_000, 8_000, 32_000, 128_000)


def _input_bucket(input_tokens: int) -> str:
    """Map an input-token count to the same coarse label the core uses."""
    if input_tokens < _INPUT_BUCKETS[0]:
        return "xs"
    if input_tokens < _INPUT_BUCKETS[1]:
        return "s"
    if input_tokens < _INPUT_BUCKETS[2]:
        return "m"
    if input_tokens < _INPUT_BUCKETS[3]:
        return "l"
    return "xl"


def _stratum_from_labels(labels: tuple[str, ...]) -> str | None:
    """Recover the stratum from an outcome's labels.

    The shaper tags every request with ``stratum:<key>``, so the outcome
    carries enough to attribute itself even when no decide_turn is on record.
    """
    for label in labels or ():
        if label.startswith("stratum:"):
            key = label[len("stratum:") :]
            if key:
                return key
    return None


def _default_stratum_key(features: TurnFeatures) -> str:
    """Derive a stratum key when the caller did not supply one.

    Order is most-to-least specific, mirroring the core's
    ``output_savings_policy.stratum_key`` so keys read the same way in logs.
    """
    return "|".join(
        (
            (features.model or "unknown").lower(),
            features.turn_kind.value,
            _input_bucket(features.input_tokens),
            "tools" if features.has_tools else "notools",
        )
    )


@dataclass
class _Accum:
    """Running count / sum / sum-of-squares for online mean & variance.

    Matches ``headroom.proxy.output_savings._Accum`` field-for-field so a
    dumped accumulator means the same thing in both places.
    """

    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        """Sample variance (unbiased). 0 when fewer than 2 observations."""
        if self.n < 2:
            return 0.0
        return max(0.0, (self.sumsq - self.sum * self.sum / self.n) / (self.n - 1))

    def to_dict(self) -> dict[str, float]:
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Accum:
        acc = cls()
        acc.n = int(d.get("n", 0))
        acc.sum = float(d.get("sum", 0.0))
        acc.sumsq = float(d.get("sumsq", 0.0))
        return acc


@dataclass
class _StratumState:
    """Everything the lever knows about one stratum."""

    accum: _Accum = field(default_factory=_Accum)
    multiplier: float = 1.0
    """AIMD state: >=1.0. Widened on truncation, decayed back on quiet turns."""

    cooldown_remaining: int = 0
    """Observations left before tightening resumes after the last widen."""

    observations: int = 0
    truncations: int = 0

    def cap(self, k: float, floor: int) -> float:
        raw = self.accum.mean + k * (self.accum.var**0.5)
        return max(float(floor), raw * self.multiplier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accum": self.accum.to_dict(),
            "multiplier": self.multiplier,
            "cooldown_remaining": self.cooldown_remaining,
            "observations": self.observations,
            "truncations": self.truncations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _StratumState:
        return cls(
            accum=_Accum.from_dict(d.get("accum") or {}),
            multiplier=float(d.get("multiplier", 1.0)),
            cooldown_remaining=int(d.get("cooldown_remaining", 0)),
            observations=int(d.get("observations", 0)),
            truncations=int(d.get("truncations", 0)),
        )


class AdaptiveMaxTokensLever:
    """Caps ``max_tokens`` per stratum from observed output-token history.

    See the module docstring for the statistics and the AIMD control loop.
    Defaults:

    * ``k=3.0`` — mean + 3 sigma, generous enough that a legitimate long
      response is rarely the one that gets capped.
    * ``min_samples=50`` — below this a mean/variance estimate is noise, not
      a distribution; the lever declines rather than guess.
    * ``floor=256`` — an absolute backstop so a pathological stratum (e.g. a
      handful of near-zero-output samples) can never produce an unusably
      small cap.
    * ``widen_factor=1.5`` — a single truncation raises the cap 50% on the
      spot; large enough to clear most transient under-estimates in one step.
    * ``cooldown_observations=20`` — roughly a few user turns' worth of
      mechanical continuations in a typical agentic loop; long enough that one
      truncation does not get immediately half-undone by the very next quiet
      observation.
    * ``probe_decay=0.9`` — per quiet observation after cooldown, the
      multiplier's excess over 1.0 shrinks by 10%. Deliberately slow relative
      to the instant widen: AIMD's whole point is backing off fast and
      recovering slowly, so a cap does not flap back to tight the moment
      traffic looks calm and immediately truncate the next similar request.
    """

    def __init__(
        self,
        *,
        k: float = 3.0,
        min_samples: int = 50,
        floor: int = 256,
        widen_factor: float = 1.5,
        cooldown_observations: int = 20,
        probe_decay: float = 0.9,
    ) -> None:
        self.k = k
        self.min_samples = min_samples
        self.floor = floor
        self.widen_factor = widen_factor
        self.cooldown_observations = cooldown_observations
        self.probe_decay = probe_decay
        self._strata: dict[str, _StratumState] = {}
        # conversation_key -> stratum key decided on the most recent decide_turn
        # call for that conversation. OutcomeReading carries no model/turn-kind
        # data of its own, so this is how observe() learns which accumulator an
        # outcome belongs to without the contract needing a wider payload.
        self._pending: dict[str, str] = {}
        # Hard bound on the pending map. An entry is written on every
        # decide_turn and normally removed by the matching observe(), but a
        # request that never produces an outcome (upstream error, cancelled
        # stream, client disconnect) leaves one behind forever. Unbounded, that
        # is a slow memory leak in a proxy that runs for weeks.
        self._max_pending = 4096
        self.descriptor = LeverDescriptor(
            name="adaptive_max_tokens",
            summary="Cap max_tokens per stratum from observed output-token history.",
            wire_formats=(
                WireFormat.ANTHROPIC,
                WireFormat.OPENAI_CHAT,
                WireFormat.OPENAI_RESPONSES,
            ),
            requires_present="max_tokens",
        )

    def _stratum_key(self, features: TurnFeatures) -> str:
        supplied = features.metadata.get("stratum")
        if isinstance(supplied, str) and supplied:
            return supplied
        return _default_stratum_key(features)

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        """Propose a cap once the stratum has enough history, else decline.

        Recording the key in ``_pending`` happens unconditionally — even a
        cold-start decline needs to remember which stratum this turn belongs
        to, so the eventual ``observe()`` call can feed the right accumulator
        and the stratum can warm up in the first place.
        """
        key = self._stratum_key(features)
        if len(self._pending) >= self._max_pending:
            # Drop the oldest half rather than clearing: keeps the most recent
            # conversations attributable, and dicts preserve insertion order.
            for stale in list(self._pending)[: self._max_pending // 2]:
                del self._pending[stale]
        self._pending[features.conversation_key] = key

        state = self._strata.get(key)
        if state is None or state.accum.n < self.min_samples:
            return ParamDecision()

        cap = int(round(state.cap(self.k, self.floor)))
        return ParamDecision(max_tokens=cap, labels=(f"max_tokens_stratum:{key}",))

    def observe(self, outcome: OutcomeReading) -> None:
        """Feed the outcome into its stratum and run the AIMD update.

        Resolving the stratum has two paths, and the fallback matters. The
        primary one is the ``_pending`` entry written by ``decide_turn``. But a
        proxy's request and response paths are decoupled, so an outcome can
        arrive with no matching decide_turn — a different worker, a restart, or
        simply a host that reports outcomes it never asked us to shape. Dropping
        those silently would make a learning loop that quietly learns nothing,
        which is the worst possible failure mode for this lever: it would look
        healthy and never warm up.

        So fall back to the ``stratum:<key>`` label the shaper already puts on
        every request. That also lets the lever learn from **control-arm**
        turns, which is strictly better data than it would otherwise have:
        control turns are unshaped by definition, so they are the cleanest
        estimate of what this stratum costs without us.
        """
        key = self._pending.pop(outcome.conversation_key, None)
        if key is None:
            key = _stratum_from_labels(outcome.labels)
        if key is None:
            return

        state = self._strata.setdefault(key, _StratumState())
        state.accum.add(float(outcome.output_tokens))
        state.observations += 1

        if outcome.truncated:
            # The expensive event: the cap was too tight for a real request.
            # Back off immediately and hold off tightening for a while so the
            # very next quiet observation doesn't undo the correction.
            state.truncations += 1
            state.multiplier *= self.widen_factor
            state.cooldown_remaining = self.cooldown_observations
        elif state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
        else:
            # Slow probe back toward the plain statistical cap.
            state.multiplier = max(1.0, 1.0 + (state.multiplier - 1.0) * self.probe_decay)

    @property
    def truncation_rate(self) -> float:
        """Fraction of observed turns that were truncated, across all strata.

        The published safety metric for this lever: a rising rate means the
        statistics (or ``k``) need revisiting before trusting it further.
        """
        total_obs = sum(s.observations for s in self._strata.values())
        if not total_obs:
            return 0.0
        total_trunc = sum(s.truncations for s in self._strata.values())
        return total_trunc / total_obs

    def stats(self) -> dict[str, dict[str, Any]]:
        """Plain-dict per-stratum snapshot, for the CLI to print."""
        out: dict[str, dict[str, Any]] = {}
        for key, state in self._strata.items():
            ready = state.accum.n >= self.min_samples
            out[key] = {
                "n": state.accum.n,
                "mean": state.accum.mean,
                "var": state.accum.var,
                "multiplier": state.multiplier,
                "cooldown_remaining": state.cooldown_remaining,
                "observations": state.observations,
                "truncations": state.truncations,
                "truncation_rate": (
                    state.truncations / state.observations if state.observations else 0.0
                ),
                "cap": state.cap(self.k, self.floor) if ready else None,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise tunables and learned state. No I/O — the caller writes it."""
        return {
            "k": self.k,
            "min_samples": self.min_samples,
            "floor": self.floor,
            "widen_factor": self.widen_factor,
            "cooldown_observations": self.cooldown_observations,
            "probe_decay": self.probe_decay,
            "strata": {key: state.to_dict() for key, state in self._strata.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdaptiveMaxTokensLever:
        """Inverse of :meth:`to_dict`. Pending in-flight state is not carried:
        it is transient per-process bookkeeping, not durable learned state."""
        lever = cls(
            k=float(d.get("k", 3.0)),
            min_samples=int(d.get("min_samples", 50)),
            floor=int(d.get("floor", 256)),
            widen_factor=float(d.get("widen_factor", 1.5)),
            cooldown_observations=int(d.get("cooldown_observations", 20)),
            probe_decay=float(d.get("probe_decay", 0.9)),
        )
        lever._strata = {
            key: _StratumState.from_dict(v) for key, v in (d.get("strata") or {}).items()
        }
        return lever
