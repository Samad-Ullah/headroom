"""A scripted stand-in for a model provider.

It plays a fixed, competent agent: run the tests, read the module, write the
fixed module, stop. Because it counts ``prompt_tokens`` with tiktoken over the
messages it *actually receives*, every arm's token figure is a true end-to-end
measurement of what compression achieved on the wire.

What it CANNOT measure: whether a real model would still solve the task from
compressed input. The script already knows the answer, so it is blind to
information loss by construction. Task-success numbers from this provider are
meaningless and the runner labels them as such.
"""

from __future__ import annotations

import json
import os
import re
import pathlib
import uuid

import tiktoken
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ENC = tiktoken.get_encoding("cl100k_base")
TASK_ROOT = pathlib.Path(os.environ.get("TASK_ROOT", "/tmp/eval-tasks"))
DUMP = os.environ.get("MOCK_DUMP")
app = FastAPI()


def count(messages) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(ENC.encode(c))
        elif isinstance(c, list):
            n += len(ENC.encode(json.dumps(c)))
        for call in (m.get("tool_calls") or []):
            n += len(ENC.encode(json.dumps(call)))
    return n


def tool_call(name: str, args: dict) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def reply(msg: dict, prompt_tokens: int, completion_tokens: int, model: str) -> dict:
    return {"id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
            "model": model, "choices": [{"index": 0, "message": msg,
                                         "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


@app.get("/health")
async def health():
    return {"ok": True}


@app.api_route("/{full_path:path}", methods=["POST"])
async def chat(full_path: str, request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "mock")
    ptok = count(messages)
    if DUMP:
        d = pathlib.Path(DUMP); d.mkdir(parents=True, exist_ok=True)
        tid = request.headers.get("x-eval-task", "?")
        n = len(list(d.glob(f"{tid}-*.json")))
        (d / f"{tid}-{n:02d}.json").write_text(json.dumps(
            [{"role": m.get("role"),
              "chars": len(m["content"]) if isinstance(m.get("content"), str) else 0,
              "head": (m.get("content") or "")[:90] if isinstance(m.get("content"), str) else ""}
             for m in messages], indent=1))

    # Which task is this? The mock is told via the request header the runner sets.
    task_id = request.headers.get("x-eval-task", "")
    names = [m.get("name") for m in messages if m.get("role") == "tool"]
    n_tool = len([m for m in messages if m.get("role") == "tool"])

    # Script: run tests -> read module -> write fix -> done.
    if n_tool == 0:
        msg = {"role": "assistant", "content": None, "tool_calls": [tool_call("run_tests", {})]}
    elif n_tool == 1:
        msg = {"role": "assistant", "content": None,
               "tool_calls": [tool_call("read_file", {"path": "reportlib.py"})]}
    elif n_tool == 2:
        # Repair from ground truth: this stand-in does not diagnose, it applies
        # the known fix, so token accounting stays valid while success does not.
        manifest = json.loads((TASK_ROOT / "manifest.json").read_text())
        entry = next((t for t in manifest if t["task_id"] == task_id), None)
        src = (TASK_ROOT / task_id / "reportlib.py").read_text() if entry else ""
        fixed = src.replace(entry["broken_line"], entry["correct_line"], 1) if entry else src
        msg = {"role": "assistant", "content": None,
               "tool_calls": [tool_call("write_file", {"path": "reportlib.py", "content": fixed})]}
    else:
        msg = {"role": "assistant", "content": "DONE"}

    ctok = len(ENC.encode(json.dumps(msg)))
    return JSONResponse(reply(msg, ptok, ctok, model))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9998, log_level="warning")
