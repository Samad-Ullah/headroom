#!/usr/bin/env bash
# toggle_patch.sh on|off
#
# Swaps the patched content_detector into the INSTALLED headroom package so
# baseline and treatment can be measured in the same interpreter.
#
#   ./toggle_patch.sh on          # patched detector (this fork's change)
#   ./toggle_patch.sh off         # upstream detector
#   PYTHON=/path/to/python ./toggle_patch.sh on
#
# The install location is resolved by ASKING the interpreter, not by guessing a
# venv path: the venv frequently lives outside the repo, and a guessed path fails
# silently, which makes the toggle a no-op and every subsequent measurement a lie.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-}"

case "$MODE" in on|off) ;; *) echo "usage: toggle_patch.sh on|off" >&2; exit 2 ;; esac

command -v "$PYTHON" >/dev/null || { echo "no such interpreter: $PYTHON (set PYTHON=)" >&2; exit 1; }

TARGET="$("$PYTHON" - <<'PY'
import os, pathlib, sys

# Drop the cwd from sys.path: run from the repo root, "import headroom" would
# otherwise resolve to the checkout (a namespace package with no compiled Rust
# core) instead of the installed wheel we mean to patch.
sys.path = [p for p in sys.path if p not in ("", os.getcwd())]
try:
    import headroom
except ImportError:
    sys.exit("headroom is not importable by this interpreter; set PYTHON=")
p = pathlib.Path(headroom.__file__).parent / "transforms" / "content_detector.py"
if "site-packages" not in str(p):
    sys.exit(f"refusing to patch a non-installed copy: {p}")
print(p)
PY
)"

ORIG="$REPO/analysis/results/content_detector.BASELINE.py"
PATCHED="$REPO/headroom/transforms/content_detector.py"
[ -f "$ORIG" ]    || { echo "missing baseline copy: $ORIG" >&2; exit 1; }
[ -f "$PATCHED" ] || { echo "missing patched copy: $PATCHED" >&2; exit 1; }

[ "$MODE" = on ] && cp "$PATCHED" "$TARGET" || cp "$ORIG" "$TARGET"
find "$(dirname "$TARGET")" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Verify the swap actually took, so a failure can never look like a success.
"$PYTHON" - "$MODE" <<'PY'
import os, sys, pathlib
sys.path = [p for p in sys.path if p not in ("", os.getcwd())]
import headroom
src = (pathlib.Path(headroom.__file__).parent / "transforms" / "content_detector.py").read_text()
has = "_TEST_RUNNER_PATTERNS" in src
want = sys.argv[1] == "on"
if has != want:
    sys.exit(f"TOGGLE FAILED: wanted {sys.argv[1]}, file says patched={has}")
print(f"{'PATCHED' if has else 'BASELINE'} detector active")
PY
