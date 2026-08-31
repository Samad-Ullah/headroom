"""Stratification, holdout assignment and online statistics for output-shaping.

Ported from ``headroom.proxy.output_savings_policy`` /
``headroom.proxy.output_savings`` (the core's counterfactual-savings machinery)
and extended with one field the core's stratum key lacks: ``harness``.

Why the extension matters
--------------------------
The core's stratum is ``model_family|turn_kind|input_bucket|tools`` — nothing
in it names *which client* sent the request. GitHub Copilot silently received
zero output savings for a stretch because a wiring bug meant the shaper never
saw Copilot traffic at all, and there was no row in any report that could show
a whole harness getting nothing: every stratum any harness could hit looked
the same regardless of who actually produced the data. Coverage gaps were
invisible by construction.

Adding ``harness`` to the key means a systemic per-client bug now shows up as
its own row of zeros instead of being averaged away into a healthy-looking
aggregate.

This module is intentionally pure (stdlib only, no I/O) so it can be tested
and reasoned about independent of the executor and registry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Coarse input-token buckets, in tokens. Coarse on purpose: too many strata
# make per-stratum baselines sparse and noisy. Boundaries match the core
# exactly so a baseline learned by one side is directly comparable to the
# other.
_INPUT_BUCKETS = (2_000, 8_000, 32_000, 128_000)

#: Stratum key field order, most-specific-last.
#:
#: The core relies on trimming *trailing* fields to back off from a specific
#: stratum to a more general one when a lookup misses (see
#: ``BaselineModel.lookup``): dropping "tools" first, then "input_bucket",
#: then "turn_kind" widens the match while staying as specific as possible at
#: each step. That property must survive the extension.
#:
#: ``harness`` is placed *first* — the least specific position, trimmed last —
#: for the opposite reason: it is the one field whose whole point is to stay
#: visible even after every other field has been generalised away. A lookup
#: backed all the way off still lands on a `harness|model_family` bucket
#: rather than losing the harness dimension entirely, so "this client sees a
#: healthy mean across everything else" and "this client sees nothing" remain
#: distinguishable at every level of back-off.
_STRATUM_FIELDS = ("harness", "model_family", "turn_kind", "input_bucket", "tools")


def input_bucket(input_tokens: int) -> str:
    """Map an input-token count to a coarse bucket label.

    Matches ``headroom.proxy.output_savings_policy.input_bucket`` exactly.
    """
    if input_tokens < _INPUT_BUCKETS[0]:
        return "xs"
    if input_tokens < _INPUT_BUCKETS[1]:
        return "s"
    if input_tokens < _INPUT_BUCKETS[2]:
        return "m"
    if input_tokens < _INPUT_BUCKETS[3]:
        return "l"
    return "xl"


def model_family(model: str) -> str:
    """Collapse a model id to a coarse family for stratification.

    Matches ``headroom.proxy.output_savings_policy.model_family`` exactly.
    Token-spend behaviour clusters by family far more than by point release,
    so every ``claude-opus-*`` (etc.) is bucketed together.
    """
    m = model.lower()
    for fam in ("opus", "sonnet", "haiku", "fable", "mythos", "gpt", "gemini"):
        if fam in m:
            return fam
    return "other"


def stratum_key(
    *,
    model: str,
    turn_kind: str,
    input_tokens: int,
    has_tools: bool,
    harness: str = "unknown",
) -> str:
    """Build a stratum key from request features observable before the response.

    Field order is ``harness|model_family|turn_kind|input_bucket|tools`` — see
    ``_STRATUM_FIELDS`` for the back-off rationale. Baseline/ledger lookups may
    back off by trimming trailing fields; this ordering is a load-bearing part
    of that contract, not cosmetic.
    """
    return "|".join(
        (
            harness,
            model_family(model),
            turn_kind,
            input_bucket(input_tokens),
            "tools" if has_tools else "notools",
        )
    )


def parse_stratum(key: str) -> dict[str, str]:
    """Invert :func:`stratum_key` for reporting.

    Raises ``ValueError`` if *key* does not have the expected number of
    ``|``-separated fields — a malformed key is a bug worth surfacing loudly
    rather than silently mis-attributing a report row.
    """
    parts = key.split("|")
    if len(parts) != len(_STRATUM_FIELDS):
        raise ValueError(
            f"stratum key {key!r} has {len(parts)} fields, expected {len(_STRATUM_FIELDS)}"
        )
    return dict(zip(_STRATUM_FIELDS, parts, strict=True))


def assign_arm(conversation_key: str, holdout_fraction: float) -> str:
    """Deterministically assign a conversation to ``treatment`` or ``control``.

    Ported verbatim from
    ``headroom.proxy.output_savings_policy.assign_arm``: a sha256 digest of the
    conversation key decides the arm, so the same conversation always lands in
    the same arm without any shared mutable state.

    Conversation-stable assignment matters for two reasons that happen to
    align: it is what makes the treatment/control comparison an unbiased A/B
    test (mixing shaped and unshaped turns within one conversation would
    pollute the comparison), and it keeps the provider's prefix cache stable,
    because flipping a conversation's system-prompt tail mid-stream would bust
    the cached prefix regardless of any experiment design concern.
    """
    if holdout_fraction <= 0.0:
        return "treatment"
    if holdout_fraction >= 1.0:
        return "control"
    digest = hashlib.sha256(("arm:" + conversation_key).encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "control" if frac < holdout_fraction else "treatment"


@dataclass
class Accum:
    """Online count / sum / sum-of-squares for streaming mean & variance.

    Ported from the core's ``_Accum`` (``headroom.proxy.output_savings``). The
    variance formula is the textbook "sum of squares minus square of sums"
    computation, which loses precision when ``sum*sum`` is large relative to
    the spread of the data (catastrophic cancellation). It is kept exactly as
    written in the core rather than "simplified" to a two-pass or Welford
    formulation, so behaviour between the core and this port stays identical
    bit-for-bit on the same inputs.
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
        """Sample variance (unbiased, ddof=1). 0 when fewer than 2 observations."""
        if self.n < 2:
            return 0.0
        return max(0.0, (self.sumsq - self.sum * self.sum / self.n) / (self.n - 1))

    @property
    def stddev(self) -> float:
        return self.var**0.5

    def merge(self, other: Accum) -> None:
        """Fold another accumulator's observations into this one.

        n / sum / sumsq are additive, so merging is element-wise addition and
        is exactly equivalent to having ``add``-ed both observation streams in
        either order.
        """
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq

    def to_dict(self) -> dict[str, float]:
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Accum:
        a = cls()
        a.n = int(d.get("n", 0))
        a.sum = float(d.get("sum", 0.0))
        a.sumsq = float(d.get("sumsq", 0.0))
        return a
