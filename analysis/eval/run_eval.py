"""Run the coding-agent evaluation across arms and write per-run records.

Arms differ in exactly one thing each — the path the prompt takes to the model:

  none                  agent -> provider                     (uncompressed control)
  headroom-baseline     agent -> headroom proxy -> provider   (upstream detector)
  headroom-extension    agent -> headroom proxy -> provider   (patched detector)

Every arm runs every task on a pristine copy of that task, in a fixed order,
with temperature 0. Token counts come from the provider's own usage field.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import pathlib
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from agent import Agent  # noqa: E402
from tasks import build_all  # noqa: E402

VENV = Path(os.environ.get("EVAL_VENV_BIN", str(Path(sys.executable).parent)))
TOGGLE = str(Path(__file__).resolve().parents[1] / "experiments" / "toggle_patch.sh")


def wait_for(url: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=3).status_code < 500:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


def start(cmd: list[str], log: Path, env: dict | None = None) -> subprocess.Popen:
    fh = open(log, "w")
    e = {**os.environ, **(env or {})}
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            env=e, start_new_session=True)


def stop(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def set_detector(patched: bool) -> None:
    subprocess.run(["bash", TOGGLE, "on" if patched else "off"],
                   check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider-url", default="http://127.0.0.1:9998/v1",
                    help="OpenAI-compatible endpoint the proxy forwards to")
    ap.add_argument("--provider-host", default="http://127.0.0.1:9998",
                    help="base host handed to headroom via x-headroom-base-url")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "sk-mock-000"))
    ap.add_argument("--tasks-root", default="/tmp/eval-tasks")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "eval-run"))
    ap.add_argument("--arms", default="none,headroom-baseline,headroom-extension")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--mock", action="store_true", help="also launch the scripted provider")
    ap.add_argument("--order", choices=["forward", "reverse"], default="forward",
                    help="task order; reverse tests whether outliers track position not task")
    args = ap.parse_args()

    root = Path(args.tasks_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)

    manifest = build_all(root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    task_ids = [m["task_id"] for m in manifest]
    if args.order == "reverse":
        task_ids = list(reversed(task_ids))
    print(f"{len(task_ids)} tasks in {root}")

    mock_proc = None
    if args.mock:
        mock_proc = start([str(VENV / "python"), str(Path(__file__).resolve().parent / "mock_provider.py")],
                          out / "mock.log", {"TASK_ROOT": str(root)})
        if not wait_for("http://127.0.0.1:9998/health"):
            print("mock provider failed to start", file=sys.stderr)
            stop(mock_proc)
            return 1
        print("scripted provider up on 9998")

    records: list[dict] = []
    try:
        for arm in args.arms.split(","):
            arm = arm.strip()
            proxy = None
            port = 8801
            if arm == "none":
                base_url = args.provider_url
                extra_headers = {}
            else:
                set_detector(patched=(arm == "headroom-extension"))
                proxy = start(
                    [str(VENV / "headroom"), "proxy", "--port", str(port),
                     "--mode", "token", "--code-aware", "--no-rate-limit", "--no-cache"],
                    out / f"proxy-{arm}.log",
                    {"HEADROOM_ALLOWED_BASE_URLS": f"{args.provider_host},127.0.0.1",
                     "OPENAI_API_KEY": args.api_key, "HEADROOM_UPDATE_CHECK": "off"},
                )
                if not wait_for(f"http://127.0.0.1:{port}/health", 90):
                    print(f"proxy for {arm} failed to start", file=sys.stderr)
                    stop(proxy)
                    return 1
                base_url = f"http://127.0.0.1:{port}/v1"
                extra_headers = {}
                print(f"proxy up for arm {arm} on {port}")

            for tid in task_ids:
                work = Path(f"/tmp/eval-run/{arm}/{tid}")
                if work.exists():
                    shutil.rmtree(work)
                work.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(root / tid, work)

                agent = Agent(base_url=base_url, api_key=args.api_key,
                              model=args.model, workdir=work, max_turns=args.max_turns)
                # headroom needs to be told where upstream lives; the control arm
                # talks to the provider directly and ignores the header.
                agent_headers = {"x-headroom-base-url": args.provider_host,
                                 "x-eval-task": tid, **extra_headers}
                orig = agent.solve
                res = _solve_with_headers(agent, tid, arm, agent_headers)
                records.append(res)
                print(f"  [{arm:20s}] {tid:20s} solved={str(res['solved']):5s} "
                      f"ptok={res['prompt_tokens']:>7,} turns={res['turns']} "
                      f"unserviced={res['unserviced_tool_calls']}")
            stop(proxy)
    finally:
        stop(mock_proc)

    (runs_dir / "records.json").write_text(json.dumps(records, indent=2))
    print(f"\n{len(records)} records -> {runs_dir / 'records.json'}")
    return 0


def _solve_with_headers(agent: Agent, tid: str, arm: str, headers: dict) -> dict:
    """Run the agent with extra request headers (task id + upstream override)."""
    import httpx as _hx
    real_post = _hx.Client.post

    def patched_post(self, url, **kw):
        kw.setdefault("headers", {})
        kw["headers"] = {**headers, **kw["headers"]}
        return real_post(self, url, **kw)

    _hx.Client.post = patched_post
    try:
        return agent.solve(tid, arm).as_dict()
    finally:
        _hx.Client.post = real_post


if __name__ == "__main__":
    raise SystemExit(main())
