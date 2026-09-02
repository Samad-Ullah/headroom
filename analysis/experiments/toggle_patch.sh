#!/usr/bin/env bash
# toggle_patch.sh on|off -- swap the patched content_detector into the installed
# wheel so baseline and treatment can be measured in the same interpreter.
#   VENV=/path/to/venv  ./toggle_patch.sh on
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO/.venv313}"
SP="$(ls -d "$VENV"/lib/python3.*/site-packages/headroom/transforms/content_detector.py)"
ORIG="$REPO/analysis/results/content_detector.BASELINE.py"
PATCHED="$REPO/headroom/transforms/content_detector.py"
[ -f "$ORIG" ] || { echo "missing baseline copy: $ORIG" >&2; exit 1; }
case "${1:-}" in
  on)  cp "$PATCHED" "$SP"; echo "PATCHED detector active" ;;
  off) cp "$ORIG"    "$SP"; echo "BASELINE detector active" ;;
  *)   echo "usage: toggle_patch.sh on|off" >&2; exit 1 ;;
esac
find "$VENV" -name '__pycache__' -path '*transforms*' -exec rm -rf {} + 2>/dev/null || true
