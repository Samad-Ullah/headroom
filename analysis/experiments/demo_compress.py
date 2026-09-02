"""Config matrix demo. Run: .venv/bin/python experiments/demo_compress.py"""
from headroom import compress
from fixtures import pytest_log, github_json, source_file

CASES = [
    ("pytest log (400 pass, 1 fail)", pytest_log()),
    ("GitHub issues JSON (120 items)", github_json()),
    ("Python source (30 functions)", source_file()),
]

# Each arm is a (label, kwargs) pair passed straight to compress().
ARMS = [
    ("default", {}),
    ("protect_recent=0", {"protect_recent": 0}),
    ("+compress_user_msgs", {"protect_recent": 0, "compress_user_messages": True}),
    ("+no analysis guard", {"protect_recent": 0, "compress_user_messages": True,
                            "protect_analysis_context": False}),
]

PAD = 8  # filler turns so the payload falls outside protect_recent


def build(blob: str, pad: int) -> list:
    msgs = [{"role": "user", "content": "Why is the billing test failing?"}]
    msgs.append({"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "run"}}]})
    msgs.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": blob}]})
    for i in range(pad):
        msgs.append({"role": "assistant", "content": f"Checking hypothesis {i}."})
        msgs.append({"role": "user", "content": "Keep going."})
    return msgs


if __name__ != "__main__":
    import sys as _s; _s.exit(0)
hdr = f"{'case':32s}" + "".join(f"{a[0]:>21s}" for a in ARMS)
print(hdr)
print("-" * len(hdr))

for name, blob in CASES:
    row = f"{name:32s}"
    for _label, kw in ARMS:
        r = compress(build(blob, PAD), model="gpt-4o", **kw)
        pct = 100.0 * (r.tokens_before - r.tokens_after) / r.tokens_before if r.tokens_before else 0.0
        row += f"{r.tokens_before:8,d}->{r.tokens_after:<7,d}{pct:4.0f}%"
    print(row)

print()
r = compress(build(pytest_log(), PAD), model="gpt-4o",
             protect_recent=0, compress_user_messages=True,
             protect_analysis_context=False)
print("transforms applied (pytest log, most aggressive arm):", r.transforms_applied)

blk = r.messages[2]["content"]
txt = blk[0].get("content") if isinstance(blk, list) else blk
txt = str(txt)
print(f"\n=== compressed pytest log ({len(txt)} chars) ===\n")
print(txt[:1500])
