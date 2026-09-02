"""Paired analysis of the evaluation records.

The design is paired: every arm runs every task, so each task contributes one
matched observation per arm. Differences are therefore analysed per task, which
removes between-task variance (some bugs simply produce longer logs).

Confidence intervals are bootstrap percentile intervals over tasks (10,000
resamples), which makes no normality assumption — appropriate here because the
per-task savings are visibly bimodal.
"""
from __future__ import annotations

import json
import random
import statistics as st
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

random.seed(20260902)


def boot_ci(xs, fn=st.mean, n=10000, alpha=0.05):
    if not xs:
        return (float("nan"), float("nan"))
    reps = []
    k = len(xs)
    for _ in range(n):
        reps.append(fn([xs[random.randrange(k)] for _ in range(k)]))
    reps.sort()
    return reps[int(alpha / 2 * n)], reps[int((1 - alpha / 2) * n) - 1]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                str(pathlib.Path(__file__).resolve().parents[1] / "results/eval/records-main.json"))
    records = json.loads(path.read_text())
    by_arm = defaultdict(dict)
    for r in records:
        by_arm[r["arm"]][r["task_id"]] = r

    arms = list(by_arm)
    control = "none"
    tasks = sorted(by_arm[control])

    print(f"tasks={len(tasks)}  arms={arms}\n")
    print(f"{'arm':22s} {'mean ptok':>11s} {'median':>9s} {'solved':>8s} "
          f"{'turns':>6s} {'unserviced':>11s}")
    print("-" * 74)
    for arm in arms:
        rs = [by_arm[arm][t] for t in tasks if t in by_arm[arm]]
        pt = [r["prompt_tokens"] for r in rs]
        print(f"{arm:22s} {st.mean(pt):11,.0f} {st.median(pt):9,.0f} "
              f"{sum(r['solved'] for r in rs):>4d}/{len(rs):<3d} "
              f"{st.mean([r['turns'] for r in rs]):6.1f} "
              f"{sum(r['unserviced_tool_calls'] for r in rs):>11d}")

    print(f"\nPaired reduction vs '{control}' (per task, then bootstrapped)")
    print("-" * 74)
    print(f"{'arm':22s} {'mean saved':>11s} {'95% CI':>20s} {'tasks improved':>16s}")
    for arm in arms:
        if arm == control:
            continue
        pcts, improved = [], 0
        for t in tasks:
            if t not in by_arm[arm]:
                continue
            base = by_arm[control][t]["prompt_tokens"]
            got = by_arm[arm][t]["prompt_tokens"]
            if base:
                pct = 100.0 * (base - got) / base
                pcts.append(pct)
                improved += pct > 1.0
        lo, hi = boot_ci(pcts)
        print(f"{arm:22s} {st.mean(pcts):10.1f}% {f'[{lo:.1f}, {hi:.1f}]':>20s} "
              f"{improved:>10d}/{len(pcts):<5d}")

    # Per-task detail for the treatment arm, to expose any bimodality.
    treat = "headroom-extension"
    if treat in by_arm:
        print(f"\nPer-task detail — {treat}")
        print("-" * 74)
        rows = []
        for t in tasks:
            if t not in by_arm[treat]:
                continue
            base = by_arm[control][t]["prompt_tokens"]
            got = by_arm[treat][t]["prompt_tokens"]
            rows.append((100.0 * (base - got) / base, t, base, got))
        for pct, t, base, got in sorted(rows):
            flag = "   <-- NOT COMPRESSED" if pct < 50 else ""
            print(f"  {t:22s} {base:>7,} -> {got:>7,}  {pct:6.1f}%{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
