#!/usr/bin/env bash
# Provision the evaluation environment.
#
# Python 3.11+ matters: on 3.10 pip resolves onnxruntime <1.24 and headroom
# silently falls back to pure-Python content detection.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO/.venv313}"

uv python install 3.13
uv venv --python 3.13 "$VENV"
PY="$VENV/bin/python"

# CPU-only torch: the CUDA wheels are multi-GB and unnecessary here.
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" "headroom-ai[proxy,code,evals,ml]" pytest

"$PY" - <<'PYX'
from huggingface_hub import snapshot_download
print(snapshot_download("chopratejas/kompress-v2-base",
                        revision="b1563631b35bfdcee37587ad530147497d820d4c"))
PYX
"$PY" -c "import importlib.metadata as m; print('headroom-ai', m.version('headroom-ai'))"
