"""A minimal tool-using coding agent, deliberately small and auditable.

It speaks the OpenAI chat-completions API, so the same agent can be pointed
either straight at a provider (uncompressed control arm) or at a Headroom proxy
(treatment arms) by changing one base URL. Nothing else differs between arms.

Design notes that matter for the measurement:

* Token accounting comes from the provider's own ``usage.prompt_tokens`` on every
  response — the quantity actually billed — not from a local estimate.
* Temperature is 0 and the tool set is fixed, so run-to-run variation comes only
  from the provider.
* Headroom's proxy may inject a ``headroom_retrieve`` tool that this agent does
  not implement. Rather than pretend otherwise, unknown tool calls are answered
  honestly and counted in ``unserviced_tool_calls``; a non-zero count in the
  results would mean the compressed arms were handicapped and the experiment
  needs rerunning with inline CCR resolution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

TOOLS = [
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the project's pytest suite and return its full output.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the project directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite a file with new contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]

SYSTEM = (
    "You are a software engineer fixing a bug in a small Python project. "
    "Run the test suite, read the failure, find the single incorrect line in "
    "reportlib.py, and fix it with write_file. Do not edit the tests. "
    "When the suite passes, reply with DONE and nothing else."
)


@dataclass
class RunResult:
    task_id: str
    arm: str
    solved: bool = False
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    unserviced_tool_calls: int = 0
    error: str | None = None
    tool_calls: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__}


class Agent:
    def __init__(self, base_url: str, api_key: str, model: str,
                 workdir: Path, max_turns: int = 10, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.workdir = workdir
        self.max_turns = max_turns
        self.timeout = timeout

    # ---- tools -------------------------------------------------------------
    def _run_tests(self) -> str:
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "--no-header"],
            cwd=self.workdir, capture_output=True, text=True, timeout=300)
        return p.stdout + p.stderr

    def _read_file(self, path: str) -> str:
        f = self.workdir / path
        if not f.is_file():
            return f"error: no such file: {path}"
        return f.read_text()

    def _write_file(self, path: str, content: str) -> str:
        f = self.workdir / path
        if not f.is_file():
            return f"error: refusing to create a new file: {path}"
        f.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def _dispatch(self, name: str, args: dict, res: RunResult) -> str:
        res.tool_calls.append(name)
        if name == "run_tests":
            return self._run_tests()
        if name == "read_file":
            return self._read_file(args.get("path", ""))
        if name == "write_file":
            return self._write_file(args.get("path", ""), args.get("content", ""))
        res.unserviced_tool_calls += 1
        return (f"error: tool {name!r} is not available in this environment; "
                f"proceed using run_tests, read_file and write_file only.")

    # ---- loop --------------------------------------------------------------
    def solve(self, task_id: str, arm: str) -> RunResult:
        res = RunResult(task_id=task_id, arm=arm)
        started = time.time()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                     "The suite is failing. Find and fix the bug in reportlib.py."}]
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for _ in range(self.max_turns):
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"authorization": f"Bearer {self.api_key}",
                                 "content-type": "application/json"},
                        json={"model": self.model, "messages": messages,
                              "tools": TOOLS, "temperature": 0},
                    )
                    r.raise_for_status()
                    body = r.json()
                    res.turns += 1
                    usage = body.get("usage") or {}
                    res.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    res.completion_tokens += usage.get("completion_tokens", 0) or 0

                    msg = body["choices"][0]["message"]
                    messages.append(msg)
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        break
                    for call in calls:
                        fn = call.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        out = self._dispatch(fn.get("name", ""), args, res)
                        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                         "content": out})
        except Exception as exc:  # noqa: BLE001
            res.error = f"{type(exc).__name__}: {str(exc)[:200]}"

        # Ground truth: the suite decides, not the model's own claim.
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=self.workdir, capture_output=True, text=True, timeout=300)
            res.solved = p.returncode == 0
        except Exception as exc:  # noqa: BLE001
            res.error = res.error or f"verify failed: {exc}"
        res.wall_seconds = round(time.time() - started, 2)
        return res
