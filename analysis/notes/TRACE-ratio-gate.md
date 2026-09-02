# Trace: why `compress()` returns 0% on logs and source code

Env: `.venv313` (Python 3.13.15, onnxruntime 1.29.0, Kompress warmed), headroom-ai 0.37.0,
repo @ `1390d897155e69f8b4554eed5641c2e523860d0f`.
Raw trace: [`results/router-debug-trace-code.txt`](../results/router-debug-trace-code.txt).
Matrix: [`results/matrix.txt`](../results/matrix.txt).

## The `ratio_too_high` hypothesis was WRONG

`content_router.py:1616`

```python
min_ratio_relaxed: float = 1.0     # accept any shrink (no savings floor)
min_ratio_aggressive: float = 1.0  # same under pressure; net-cost is the guard
```

Both default to 1.0 — the gate accepts *any* shrink. It never fired. `ratio_too_high` is
not the cause. Ruled out.

## What the debug trace actually shows (source-code case)

```
event=content_router_strategy_result
  requested_strategy "code_aware"  ->  actual_strategy "kompress"
  reason "code_aware_unavailable_fallback_kompress"

Kompress backend=onnx words=1297 chunks=4 inference_ms=2474 ratio=0.804
event=content_router_output  total_original_tokens=3079  total_compressed_tokens=1043
  tokens_saved=2036  savings_percentage=66.1  compression_ratio=0.339

[router] route_counts={'small':17, 'content_blocks':2, 'cache_miss':1,
                       'lossy_unrecoverable_skipped':1}  compressed=0
```

**66.1% compression was computed, then discarded.** Two independent causes, each on its
own sufficient to produce the observed 0%.

## Cause 1 — AST code compression is off by default (deliberate)

`content_router.py:1503`

```python
enable_code_aware: bool = False  # Disabled: use code graph MCP tools instead
```

Verified: the compressor is installed and works. `_get_code_compressor()` returns
`CodeAwareCompressor`; `_registry_compress("code_aware", …)` returns `compressed=True`,
10,775 → 5,444 chars (**49.5%**); tree-sitter 0.26.0 + language-pack 0.13.0 present;
`is_tree_sitter_available = True`.

So the router skips a working compressor because of a config default, then labels the
outcome `code_aware_unavailable_fallback_kompress` (`content_router.py:3335`). The label is
**misleading** — the strategy is *disabled*, not unavailable, and the same string is also
emitted when the adapter ran but declined. The proxy can enable it
(`proxy/server.py:844`, `enable_code_aware=config.code_aware_enabled`), so this default is
library-mode-specific.

## Cause 2 — the reversibility gate discards unmarked lossy output

`content_router.py:5577`

```python
if accept_ratio < min_ratio:
    # tool ground truth must stay reversible — a lossy summarizer
    # (kompress/text/code) that emitted no CCR retrieve marker is
    # unrecoverable, so the agent would act on a fabricated summary (#1307).
    # Keep the original verbatim instead.
    if (enforce_rev and self.config.ccr_inject_marker
            and result.strategy_used in self.LOSSY_UNMARKED_STRATEGIES
            and not CCR_RETRIEVAL_MARKER_RE.search(result.compressed)):
        self._cache.mark_skip(content_key)
        result_slots[slot_idx] = message          # ← original kept
        route_counts["lossy_unrecoverable_skipped"] += 1
        continue
```

Kompress emits no CCR retrieval marker, and in library mode there is no CCR store behind
it, so every lossy result is dropped. This is a **deliberate safety design** (issue #1307):
better to send the original than a summary the agent cannot un-summarize.

Contrast: `LogCompressor` *does* emit a marker —
`[418 lines compressed to 17. Retrieve more: hash=024eb58b…]` — so its output would survive
this gate. It is simply never reached (see below).

## Why the pytest log never reaches `LogCompressor`

Detection classifies it `PLAIN_TEXT` at confidence **0.50**, never `BUILD_OUTPUT`.
`_strategy_from_detection` maps only `BUILD_OUTPUT → CompressionStrategy.LOG`, so the log
goes to generic `TEXT`/Kompress instead — which is then dropped by Cause 2. The
95.9%-effective log compressor is unreachable for the most common tool output a coding
agent produces.

## Net consequence (stated conservatively)

In **library mode** (`from headroom import compress`), with defaults, on these payloads:
lossy text and code compression is effectively disabled end-to-end. Only structural/JSON
compression (SmartCrusher, 46.6%) survives. Headroom's README advertises "15–20% for coding
tasks"; on this harness the coding-shaped payloads yield **0%**.

**Not yet established** — and the decisive next experiment: whether running the same
payloads through `headroom proxy` (CCR active, `code_aware_enabled` settable) recovers the
66% that library mode discards. The README states "Without the proxy, the eval runner falls
back to local compression only (no CCR)", which predicts it should. Until that is measured,
the correct claim is about **library-mode defaults**, not about headroom as a whole.
