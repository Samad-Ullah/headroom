"""Shadow-mode recording: measure a lever before it is allowed to act.

Shadow mode is a first-class capability of every param lever, not a debug
flag bolted on for the rollout. A lever computes what it *would* do — cap
``max_tokens`` at X, drop ``effort`` to "low", whatever its ``ParamDecision``
says — and that intent is recorded here against what actually happened,
*before* the lever is ever given a ``LeverMode.LIVE`` slot that lets it change
a real request.

The question this module answers is narrow and load-bearing: "how often would
this decision have constrained a real response?" For a ``max_tokens`` cap that
question is literally "what would our truncation rate have been" — the single
number that decides whether a lever is safe to flip to live traffic. Answering
it from real production traffic, at zero risk to any actual request, is the
entire point of recording shadow decisions in the first place rather than
reasoning about safety from first principles or a synthetic benchmark.

This module is kept deliberately pure — no imports from ``levers/`` or
``registry.py`` — so it is a leaf: it can be constructed, fed records, and
asserted against in a test with nothing else from the plugin in scope. The
only I/O is the explicit ``save``/``load`` pair.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Default cap on the number of distinct strata a ledger will track before
#: routing overflow into a single "__other__" bucket. A shadow ledger lives
#: for the life of a long-running proxy process; without a cap, a lever fed a
#: pathological or adversarial stream of never-repeating stratum keys (a bug
#: upstream, or a hostile client) would grow the ledger without bound. A
#: telemetry structure must not be a memory-exhaustion vector.
_DEFAULT_MAX_STRATA = 512

#: Overflow bucket name. Chosen to be unrepresentable as a real stratum key
#: (real keys never contain literal "__" wrapping) so it can never collide.
_OVERFLOW_STRATUM = "__other__"

#: How many observed values to retain per (lever, field, stratum) bucket for
#: percentile estimation. Bounded independently of the strata cap so a single
#: hot stratum on a long-running proxy cannot grow one bucket without limit.
#: Newest samples are kept (a ``deque`` with ``maxlen`` drops the oldest),
#: which biases percentiles toward recent behaviour — the right trade-off for
#: a live safety signal that should track current traffic, not all history.
_MAX_SAMPLES_PER_BUCKET = 4096


@dataclass(frozen=True)
class ShadowRecord:
    """One lever's shadow decision for one turn, plus what really happened.

    Attributes:
        stratum: The stratum key (see ``stats.stratum_key``) this turn falls
            into.
        lever: The lever's ``LeverDescriptor.name``.
        field: Which decision field this record is about, e.g. "max_tokens"
            or "effort". A single turn's decision can produce one record per
            field it set.
        intended: What the lever would have set the field to, had it been
            live.
        observed: What actually happened — e.g. the real ``output_tokens`` the
            provider returned, unconstrained by this lever.
        would_have_bound: Whether ``intended`` would have constrained
            ``observed`` had the lever been live (e.g. ``observed >
            intended`` for a ``max_tokens`` cap, meaning the response would
            have been truncated).
        turn_index: 0-based position in the conversation, for position-aware
            analysis (an early cut is worth more than a late one — see
            ``TurnFeatures.turn_index`` in the contract).
    """

    stratum: str
    lever: str
    field: str
    intended: int | str
    observed: int
    would_have_bound: bool
    turn_index: int = 0


@dataclass
class _Bucket:
    """Accumulated shadow statistics for one (lever, field, stratum) triple."""

    count: int = 0
    bind_count: int = 0
    headroom_sum: float = 0.0
    headroom_n: int = 0
    observed_samples: deque[int] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_BUCKET)
    )

    def add(self, rec: ShadowRecord) -> None:
        self.count += 1
        if rec.would_have_bound:
            self.bind_count += 1
        if isinstance(rec.intended, (int, float)) and not isinstance(rec.intended, bool):
            self.headroom_sum += rec.intended - rec.observed
            self.headroom_n += 1
        self.observed_samples.append(rec.observed)

    def merge(self, other: _Bucket) -> None:
        self.count += other.count
        self.bind_count += other.bind_count
        self.headroom_sum += other.headroom_sum
        self.headroom_n += other.headroom_n
        # Keep the most recent samples across both sides: concatenate then let
        # the maxlen deque drop the oldest.
        merged = deque(self.observed_samples, maxlen=_MAX_SAMPLES_PER_BUCKET)
        merged.extend(other.observed_samples)
        self.observed_samples = merged

    def bind_rate(self) -> float:
        return self.bind_count / self.count if self.count else 0.0

    def mean_headroom(self) -> float | None:
        return self.headroom_sum / self.headroom_n if self.headroom_n else None

    def percentiles(self) -> tuple[float | None, float | None]:
        """Nearest-rank p50/p95 of observed values, or (None, None) if empty."""
        if not self.observed_samples:
            return None, None
        values = sorted(self.observed_samples)
        return _percentile(values, 50), _percentile(values, 95)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "bind_count": self.bind_count,
            "headroom_sum": self.headroom_sum,
            "headroom_n": self.headroom_n,
            "observed_samples": list(self.observed_samples),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Bucket:
        b = cls()
        b.count = int(d.get("count", 0))
        b.bind_count = int(d.get("bind_count", 0))
        b.headroom_sum = float(d.get("headroom_sum", 0.0))
        b.headroom_n = int(d.get("headroom_n", 0))
        samples = d.get("observed_samples") or []
        b.observed_samples = deque(
            (int(x) for x in samples[-_MAX_SAMPLES_PER_BUCKET:]),
            maxlen=_MAX_SAMPLES_PER_BUCKET,
        )
        return b


def _percentile(sorted_values: list[int], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, round(pct / 100.0 * (len(sorted_values) - 1))))
    return float(sorted_values[k])


class ShadowLedger:
    """Accumulates shadow-mode decisions, keyed by (lever, field, stratum).

    Storage is a nested ``lever -> field -> stratum -> _Bucket`` dict rather
    than a flat dict keyed by a joined tuple, so it serialises to plain JSON
    without inventing a key-encoding scheme.
    """

    def __init__(self, max_strata: int = _DEFAULT_MAX_STRATA) -> None:
        self._max_strata = max_strata
        self._known_strata: set[str] = set()
        self._data: dict[str, dict[str, dict[str, _Bucket]]] = {}

    # ---- recording ---------------------------------------------------

    def record(self, rec: ShadowRecord) -> None:
        stratum = self._admit(rec.stratum)
        field_map = self._data.setdefault(rec.lever, {})
        stratum_map = field_map.setdefault(rec.field, {})
        bucket = stratum_map.setdefault(stratum, _Bucket())
        bucket.add(rec)

    def _admit(self, stratum: str) -> str:
        """Map an incoming stratum key to the one it should be recorded under.

        Once ``max_strata`` distinct strata have been seen, any further new
        stratum is folded into the overflow bucket instead of growing the
        tracked set — see ``_DEFAULT_MAX_STRATA`` for why this bound exists.
        """
        if stratum in self._known_strata:
            return stratum
        if len(self._known_strata) < self._max_strata:
            self._known_strata.add(stratum)
            return stratum
        return _OVERFLOW_STRATUM

    # ---- reporting -----------------------------------------------------

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per (lever, field, stratum): count, bind_rate, mean headroom, percentiles.

        Flat dict keyed by ``"{lever}|{field}|{stratum}"`` (rather than nested,
        so a caller — e.g. ``headroom output shadow``, which sorts and filters
        this by a ``--lever`` prefix — can treat it as a plain table without
        walking three levels of dict first). Each row is
        ``{count, bind_rate, mean_headroom, p50, p95}``. ``mean_headroom`` is
        ``None`` when the field never carried a numeric ``intended`` value
        (e.g. an ``effort`` string). ``p50``/``p95`` are the nearest-rank
        percentiles of ``observed`` and are ``None`` only if the bucket
        somehow has no samples.
        """
        out: dict[str, dict[str, Any]] = {}
        for lever, field_map in self._data.items():
            for field_name, stratum_map in field_map.items():
                for stratum, bucket in stratum_map.items():
                    p50, p95 = bucket.percentiles()
                    out[f"{lever}|{field_name}|{stratum}"] = {
                        "count": bucket.count,
                        "bind_rate": bucket.bind_rate(),
                        "mean_headroom": bucket.mean_headroom(),
                        "p50": p50,
                        "p95": p95,
                    }
        return out

    def bind_rate(self, lever: str, field: str | None = None) -> float:
        """The headline safety number: fraction of turns that would have bound.

        For a ``max_tokens`` cap this IS the would-be truncation rate — the
        number that decides whether the lever is safe to promote to
        ``LeverMode.LIVE``. Aggregated across every field of ``lever`` when
        ``field`` is ``None``, otherwise scoped to that one field.
        """
        field_map = self._data.get(lever, {})
        count = 0
        bind_count = 0
        fields = field_map.values() if field is None else [field_map.get(field, {})]
        for stratum_map in fields:
            for bucket in stratum_map.values():
                count += bucket.count
                bind_count += bucket.bind_count
        return bind_count / count if count else 0.0

    # ---- merge -----------------------------------------------------------

    def merge(self, other: ShadowLedger) -> None:
        """Fold another ledger's observations into this one.

        Strata already known to either side are merged bucket-wise. Newly
        admitted strata from ``other`` go through the same ``max_strata``
        admission check as a live ``record`` would, so the combined ledger
        never exceeds the bound regardless of how large the two inputs were.
        """
        for lever, field_map in other._data.items():
            for field_name, stratum_map in field_map.items():
                for stratum, bucket in stratum_map.items():
                    target_stratum = self._admit(stratum)
                    dest = self._data.setdefault(lever, {}).setdefault(field_name, {})
                    existing = dest.get(target_stratum)
                    if existing is None:
                        # Copy rather than alias so mutating either ledger
                        # afterwards can't leak into the other.
                        merged = _Bucket()
                        merged.merge(bucket)
                        dest[target_stratum] = merged
                    else:
                        existing.merge(bucket)

    # ---- persistence -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_strata": self._max_strata,
            "known_strata": sorted(self._known_strata),
            "data": {
                lever: {
                    field_name: {
                        stratum: bucket.to_dict() for stratum, bucket in stratum_map.items()
                    }
                    for field_name, stratum_map in field_map.items()
                }
                for lever, field_map in self._data.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShadowLedger:
        ledger = cls(max_strata=int(d.get("max_strata", _DEFAULT_MAX_STRATA)))
        ledger._known_strata = set(d.get("known_strata") or ())
        for lever, field_map in (d.get("data") or {}).items():
            out_field_map: dict[str, dict[str, _Bucket]] = {}
            for field_name, stratum_map in field_map.items():
                out_field_map[field_name] = {
                    stratum: _Bucket.from_dict(b) for stratum, b in stratum_map.items()
                }
            ledger._data[lever] = out_field_map
        return ledger

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the ledger as JSON, atomically.

        Writes to a temp file in the same directory as *path* then
        ``os.replace``s it into place, so a crash mid-write leaves either the
        old file or the new one intact — never a truncated, unparsable ledger.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, p)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ShadowLedger:
        """Load a ledger previously written by :meth:`save`.

        Returns an empty ledger if the file is missing or unparsable — a
        shadow ledger is a telemetry aid, and failing open (rather than
        raising and taking the proxy down with it) is the correct default for
        a non-load-bearing side channel.
        """
        p = Path(path)
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            return cls()
        try:
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            return cls()
