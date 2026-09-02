# Day 1 findings — first contact with headroom

Repo pinned at **1390d897155e69f8b4554eed5641c2e523860d0f** (2026-09-01), headroom-ai **0.37.0**.
Two environments, both CPU-only torch:

| env | Python | onnxruntime | native detect | Kompress |
|---|---|---|---|---|
| `.venv`    | 3.10.12 | 1.23.2 | **off** (needs ≥1.24) | absent |
| `.venv313` | 3.13.15 | 1.29.0 | on | present + warmed |

Repro: `experiments/fixtures.py`, `experiments/matrix.py`.

## 1. The compressors themselves are good

Called directly, bypassing the pipeline:

| Compressor | Before | After | Saved |
|---|---:|---:|---:|
| `LogCompressor`, pytest log (400 pass / 1 fail) | 22,264 ch | 909 ch | **95.9%** |
| `CodeCompressor`, Python source (30 fns) | 10,775 ch | 5,444 ch | **49.5%** |

`LogCompressor` keeps the header, the tail of the passing run, the `FAILED` line, the full
traceback and the summary, then appends `[401 lines omitted: 2 FAIL]` and a CCR handle
`[418 lines compressed to 17. Retrieve more: hash=024eb58b…]`. Exactly as designed.

## 2. The pipeline never reaches them

`compress(messages, …)` on the same payloads, `compress_user_messages=True` (all
protections lifted):

| Case | before → after | saved | transforms |
|---|---|---:|---|
| pytest log | 7,976 → 7,976 | **0.0%** | `router:noop` |
| Python source | 2,691 → 2,691 | **0.0%** | `router:noop` |
| GitHub JSON | 23,319 → 12,445 | 46.6% | `router:tool_result:mixed` |

**Strategy selection is correct** — the router picks the right compressor and the result is
still discarded:

```
pytest log    text         conf=0.50 -> CompressionStrategy.TEXT
py source     source_code  conf=0.94 -> CompressionStrategy.CODE_AWARE
github json   json_array   conf=1.00 -> CompressionStrategy.SMART_CRUSHER
```

So the loss is *downstream of routing*. `content_router.py:6482` defines a
`ratio_too_high` rejection bucket — the pipeline appears to compress and then discard
against a savings gate. **This is the thread to pull next.**

## 3. Hypotheses tested and REFUTED

- ~~"0% is an artifact of Python 3.10 / onnxruntime 1.23 disabling native detection."~~
  **Wrong.** 3.13 + onnxruntime 1.29 gives byte-identical results. Ruled out.
- ~~"Kompress model is missing, so the TEXT route is dead."~~ **Wrong.**
  `is_kompress_available()` returns `True`; the startup banner *"Kompress model not ready;
  requests will not be compressed"* fires before the background load and is **stale and
  misleading**. It is emitted on every run regardless.
- ~~"`protect_recent` / `compress_user_messages` explain it."~~ **Wrong.** Lifting every
  protection changes `router:protected:user_message` into `router:noop` — same 0%.

## 4. A real detection gap (independent of the above)

A pytest log is classified `PLAIN_TEXT` at confidence **0.50** — never `BUILD_OUTPUT`.
Per `_strategy_from_detection`, only `BUILD_OUTPUT` maps to `CompressionStrategy.LOG`.
So the 95.9%-effective `LogCompressor` is unreachable for the single most common tool
output a coding agent produces, and the content is sent to generic text compression instead.
**Strong extension candidate.**

## 5. Environment hazards to control for

- **The repo checkout shadows the installed wheel.** Running Python with cwd
  `/root/monperus` makes `headroom/` resolve as a namespace package, and the checkout has
  no compiled Rust `_core` → `ModuleNotFoundError: No module named 'headroom._core'`.
  Always run from a neutral cwd and assert `headroom.__file__` points into site-packages.
- Compressor results are dataclasses (`LogCompressionResult`), not strings.
  `str(result)` measures the repr and yields **negative** savings — this produced a bogus
  "log compression inflates input by 7%" in the first pass. Use `.compressed`.
- `protect_recent=4` silently protects short conversations end-to-end; a 3-message
  benchmark measures nothing at all.
- Degradation notices go to stderr and are otherwise silent; a harness that ignores stderr
  reports honest-looking 0% for entirely the wrong reason.
