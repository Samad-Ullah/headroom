"""OpenAI-compatible recording mock. Accepts ANY path; logs the path it received."""
import json, os, pathlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

RECORD_DIR = pathlib.Path(os.environ.get("MOCK_RECORD_DIR", "/tmp/mock-records"))
RECORD_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI()
_n = {"i": 0}

@app.get("/health")
async def health(): return {"ok": True}

@app.api_route("/{full_path:path}", methods=["POST", "GET", "PUT"])
async def catch_all(full_path: str, request: Request):
    body = await request.body()
    _n["i"] += 1
    (RECORD_DIR / f"{_n['i']:03d}.json").write_bytes(body or b"{}")
    (RECORD_DIR / f"{_n['i']:03d}.path").write_text(f"{request.method} /{full_path}")
    try: model = json.loads(body).get("model", "gpt-4o")
    except Exception: model = "gpt-4o"
    return JSONResponse({
        "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999, log_level="warning")
