import pathlib
import sys, os, traceback; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
WARM = os.environ.get("WARM") == "1"
from headroom.transforms import kompress_compressor as kc
if WARM: kc.warm_kompress_model()
from fixtures import pytest_log
from headroom.transforms.content_router import ContentRouter
from headroom import compress

log = []
def wrap(cls, name):
    orig = getattr(cls, name)
    def spy(self, *a, **kw):
        big = any(isinstance(x, str) and len(x) > 5000 for x in a)
        try:
            out = orig(self, *a, **kw)
        except Exception as e:
            if big: log.append((name, "RAISED", type(e).__name__, str(e)[:80]))
            raise
        if big:
            def sz(v):
                if isinstance(v, str): return len(v)
                if isinstance(v, tuple): return tuple(sz(x) for x in v)
                return v
            log.append((name, "in", len(a[0]), "out", sz(out)))
        return out
    setattr(cls, name, spy)

for m in ("_lossless_first", "_try_ml_compressor", "_relevance_split_compress",
          "_get_log_compressor", "_try_external_compressor"):
    if hasattr(ContentRouter, m): wrap(ContentRouter, m)

blob = pytest_log()
msgs = [{"role":"user","content":"Why is the billing test failing?"},
        {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"run"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":blob}]}]
for i in range(8):
    msgs.append({"role":"assistant","content":f"Checking {i}."})
    msgs.append({"role":"user","content":"Keep going."})
r = compress(msgs, model="gpt-4o", protect_recent=0, compress_user_messages=True,
             protect_analysis_context=False)
print(f"warm={int(WARM)}  result {r.tokens_before}->{r.tokens_after}  "
      f"({100*(r.tokens_before-r.tokens_after)/r.tokens_before:.1f}%)")
for e in log: print("   ", e)
