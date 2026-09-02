"""Reproducible compression matrix. Run from a NEUTRAL cwd (not the repo root).

    cd /tmp && python analysis/experiments/matrix.py
"""
import pathlib
import sys, platform, importlib.metadata as md
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import headroom
# Guard: the repo root shadows the wheel as a namespace package and
# has no compiled Rust _core. Fail loudly rather than measure the wrong thing.
assert "site-packages" in headroom.__file__, f"SHADOWED IMPORT: {headroom.__file__}"

from fixtures import pytest_log, github_json, source_file
from headroom import compress
from headroom.transforms.content_detector import detect_content_type
from headroom.transforms import kompress_compressor as kc
from headroom.transforms.content_router import ContentRouter

def v(p):
    try: return md.version(p)
    except Exception: return "MISSING"

kc.warm_kompress_model()   # the startup "not ready" banner is stale; force the load
print(f"python {platform.python_version()}  headroom {v('headroom-ai')}  "
      f"onnxruntime {v('onnxruntime')}  magika {v('magika')}  "
      f"kompress_ready={kc.is_kompress_available()}")
print(f"import: {headroom.__file__}")

CASES = [("pytest log", pytest_log()), ("py source", source_file()), ("github json", github_json())]
ARMS  = [("default", {}), ("unprotected", {"protect_recent": 0, "compress_user_messages": True,
                                           "protect_analysis_context": False})]

router = ContentRouter()
print("\n-- detection -> strategy --")
for n, b in CASES:
    d = detect_content_type(b)
    print(f"   {n:12s} {d.content_type.value:12s} conf={d.confidence:.2f} -> "
          f"{router._strategy_from_detection(d)}")

def build(blob, pad=8):
    m = [{"role":"user","content":"Why is the billing test failing?"},
         {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"run"}}]},
         {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":blob}]}]
    for i in range(pad):
        m.append({"role":"assistant","content":f"Checking hypothesis {i}."})
        m.append({"role":"user","content":"Keep going."})
    return m

print("\n-- compress() pipeline --")
print(f"   {'case':12s} {'arm':12s} {'before':>8s} {'after':>8s} {'saved':>7s}  transforms")
for n, b in CASES:
    for lbl, kw in ARMS:
        r = compress(build(b), model="gpt-4o", **kw)
        pct = 100*(r.tokens_before-r.tokens_after)/r.tokens_before if r.tokens_before else 0
        print(f"   {n:12s} {lbl:12s} {r.tokens_before:8,d} {r.tokens_after:8,d} {pct:6.1f}%  "
              f"{sorted(set(r.transforms_applied))}")

print("\n-- same payloads, compressors called DIRECTLY --")
from headroom.transforms.log_compressor import LogCompressor
from headroom.transforms.code_compressor import compress_code
lg = pytest_log(); res = LogCompressor().compress(lg)
print(f"   LogCompressor   {len(lg):8,d} {len(res.compressed):8,d} "
      f"{100*(1-len(res.compressed)/len(lg)):6.1f}%  (chars)")
sc = source_file(); cc = compress_code(sc, language="python")
txt = cc.compressed if hasattr(cc, "compressed") else str(cc)
print(f"   CodeCompressor  {len(sc):8,d} {len(txt):8,d} "
      f"{100*(1-len(txt)/len(sc)):6.1f}%  (chars)")
