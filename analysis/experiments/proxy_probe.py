"""Send fixture payloads through headroom's proxy to a recording mock upstream,
then measure what the upstream actually received.

    python proxy_probe.py <proxy_port> <label>
"""
import json, os, pathlib, sys, time
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fixtures import pytest_log, github_json, source_file

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
LABEL = sys.argv[2] if len(sys.argv) > 2 else "proxy"
REC = pathlib.Path("/tmp/mock-records")

import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")
def ntok(s): return len(ENC.encode(s))

def build(blob):
    """Canonical OpenAI tool-call shape: assistant tool_calls + role=tool result."""
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Why is the billing test failing?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"cmd":"pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": blob},
    ] + [m for i in range(8) for m in (
        {"role": "assistant", "content": f"Checking hypothesis {i}."},
        {"role": "user", "content": "Keep going."})]

CASES = [("pytest log", pytest_log()), ("py source", source_file()), ("github json", github_json())]

print(f"{'case':14s} {'sent':>9s} {'forwarded':>10s} {'saved':>8s}")
print("-" * 46)
rows = []
for name, blob in CASES:
    msgs = build(blob)
    sent = sum(ntok(m["content"]) for m in msgs if isinstance(m.get("content"), str))
    before = len(list(REC.glob("*.json")))
    r = httpx.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        headers={"x-headroom-base-url": "http://127.0.0.1:9999",
                 "authorization": "Bearer sk-mock-000",
                 "content-type": "application/json"},
        json={"model": "gpt-4o", "messages": msgs, "max_tokens": 16},
        timeout=180.0,
    )
    r.raise_for_status()
    time.sleep(0.4)
    recs = sorted(REC.glob("*.json"))
    if len(recs) <= before:
        print(f"{name:14s}  NO UPSTREAM RECORD (status {r.status_code})"); continue
    body = json.loads(recs[-1].read_bytes())
    fwd = sum(ntok(m["content"]) for m in body.get("messages", []) if isinstance(m.get("content"), str))
    pct = 100.0 * (sent - fwd) / sent if sent else 0.0
    rows.append((name, sent, fwd, pct))
    print(f"{name:14s} {sent:9,d} {fwd:10,d} {pct:7.1f}%")

if rows:
    ts, tf = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print("-" * 46)
    print(f"{'TOTAL':14s} {ts:9,d} {tf:10,d} {100*(ts-tf)/ts:7.1f}%")

try:
    s = httpx.get(f"http://127.0.0.1:{PORT}/stats", timeout=10).json()
    c = s.get("compression") or s.get("session") or {}
    print("\nproxy /stats (compression):",
          json.dumps({k: c[k] for k in list(c)[:10]}, indent=2)[:700])
except Exception as e:
    print("stats unavailable:", e)
