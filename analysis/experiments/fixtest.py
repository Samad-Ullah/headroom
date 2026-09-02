import pathlib
import sys, os; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
WARM = os.environ.get("WARM") == "1"
NOSPLIT = os.environ.get("NOSPLIT") == "1"
from headroom.transforms import kompress_compressor as kc
if WARM: kc.warm_kompress_model()
from fixtures import pytest_log
from headroom.transforms.content_router import ContentRouter
if NOSPLIT:
    ContentRouter._relevance_split_compress = lambda self, *a, **kw: None
from headroom import compress

blob = pytest_log()
msgs = [{"role":"user","content":"Why is the billing test failing?"},
        {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"run"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":blob}]}]
for i in range(8):
    msgs.append({"role":"assistant","content":f"Checking {i}."})
    msgs.append({"role":"user","content":"Keep going."})
r = compress(msgs, model="gpt-4o", protect_recent=0, compress_user_messages=True,
             protect_analysis_context=False)
blk = r.messages[2]["content"]; out = blk[0].get("content") if isinstance(blk,list) else blk
print(f"  warm={int(WARM)} relevance_split_disabled={int(NOSPLIT)}  "
      f"{r.tokens_before:,}->{r.tokens_after:,} "
      f"({100*(r.tokens_before-r.tokens_after)/r.tokens_before:5.1f}%)  block={len(str(out)):,}ch")
