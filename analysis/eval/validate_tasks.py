"""Sanity-check the benchmark before it is used to measure anything.

For every task: the suite must FAIL as generated, and must PASS once the seeded
bug is reverted. A task that fails either check is unusable — it would either
be trivially solved or impossible.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import BUGS, build_all  # noqa: E402

PY = sys.executable


def run_pytest(d: Path) -> tuple[int, str]:
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"],
                       cwd=d, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval-tasks")
    build_all(root)
    ok = True
    print(f"{'task':22s} {'broken':>10s} {'fixed':>10s} {'log lines':>10s}  verdict")
    print("-" * 70)
    for bug in BUGS:
        d = root / bug.key
        rc_broken, out_broken = run_pytest(d)
        n_lines = len(out_broken.splitlines())

        src = (d / "reportlib.py").read_text()
        (d / "reportlib.py").write_text(src.replace(bug.broken, bug.correct, 1))
        rc_fixed, out_fixed = run_pytest(d)
        (d / "reportlib.py").write_text(src)  # restore the bug

        good = rc_broken != 0 and rc_fixed == 0
        ok &= good
        print(f"{bug.key:22s} {'FAIL' if rc_broken else 'pass':>10s} "
              f"{'pass' if rc_fixed == 0 else 'FAIL':>10s} {n_lines:>10d}  "
              f"{'ok' if good else 'UNUSABLE'}")
    print("-" * 70)
    print("all tasks usable" if ok else "SOME TASKS UNUSABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
