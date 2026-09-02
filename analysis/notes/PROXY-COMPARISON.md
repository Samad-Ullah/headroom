# Proxy vs library mode — does the proxy recover the discarded savings?

repo @ `1390d897155e69f8b4554eed5641c2e523860d0f` · headroom-ai 0.37.0 · Python 3.13.15
Raw: [`results/proxy-arms.txt`](../results/proxy-arms.txt) ·
harness: [`experiments/proxy_probe.py`](../experiments/proxy_probe.py),
[`experiments/mock_upstream.py`](../experiments/mock_upstream.py)

## Method

`headroom proxy` is pointed at a **local recording mock** via the `x-headroom-base-url`
header (allowlisted with `HEADROOM_ALLOWED_BASE_URLS`, which the SSRF guard requires for
loopback). The mock stores the exact request body it receives and returns a canned
completion. "Forwarded" is therefore **what the upstream actually received**, not a
self-reported statistic — and the whole comparison costs $0 in API spend.

Payloads and message shapes are identical across arms; the tool output uses the canonical
OpenAI shape (assistant `tool_calls` + `role: "tool"` result). Semantic caching is disabled
(`--no-cache`) — with it on, repeated probes are served from cache and never reach upstream,
which silently produced "0% saved" in an earlier run.

## Result

| payload | ARM 1 library | ARM 2 proxy default (`--mode cache`) | ARM 3 proxy `--mode token --code-aware` |
|---|---:|---:|---:|
| pytest log | 0.0% | 0.0% | **0.0%** |
| Python source | 0.0% | 0.0% | **44.1%** |
| GitHub JSON | 46.6% | 0.0% | 46.8% |
| **total** | 32.0% | **0.0%** | **35.6%** |

## 1. Hypothesis CONFIRMED — for code

The 66% that library mode computed and discarded (`lossy_unrecoverable_skipped`) is real and
recoverable: through the proxy in token mode, source code compresses **44.1%** end-to-end.
CCR is active there, so the reversibility gate at `content_router.py:5577` is satisfied and
the result survives. Note the proxy enables `code_aware` on its own — its banner prints
`Code-Aware: ENABLED (AST-based)` — while the library default is
`enable_code_aware: bool = False`.

## 2. NEW — the proxy's DEFAULT mode forwarded 0% on every payload

`--mode cache` is the documented default, and in this experiment it forwarded all three
payloads **byte-unchanged**, including the JSON that library mode compresses 46.6%.

**Caveat, stated plainly:** cache mode exists to freeze prior turns so the provider's KV
prefix cache keeps hitting, and this probe is **single-shot**. A real agent session
compresses on first sighting and freezes thereafter, so a one-request probe may understate
cache mode substantially. This number should not be quoted as "the default proxy saves
nothing" until it is re-measured over a genuine multi-turn trajectory. **That is the single
biggest open question in this analysis so far.**

## 3. ROBUST — the pytest log compresses 0% in every arm

Mode-independent, config-independent, environment-independent. Across library mode, proxy
cache mode and proxy token mode, a 400-line pytest log is forwarded **completely unchanged**.

Yet `LogCompressor` compresses that exact input **95.9%** (22,264 → 909 chars) when called
directly, and emits a proper CCR marker, so it would pass the reversibility gate that blocks
Kompress.

Cause (from [`docs/TRACE-ratio-gate.md`](TRACE-ratio-gate.md)): detection classifies the log
as `PLAIN_TEXT` at confidence **0.50**, never `BUILD_OUTPUT`, and `_strategy_from_detection`
routes only `BUILD_OUTPUT → CompressionStrategy.LOG`. The log compressor is therefore
unreachable for the most common tool output a coding agent produces.

## Extension target

Finding 3 is the strongest candidate: it survives every arm, has a measured ceiling
(95.9% on this fixture), a clear mechanism (a detection gap, not a design tradeoff), and the
downstream machinery already works — `LogCompressor` emits CCR markers, so nothing else
needs changing. The fix is content detection, and the evaluation is a coding agent on
test-failure-shaped tasks, which is exactly the shape the recruitment task asks for.

## Threats to validity

- Single-shot probes; no multi-turn trajectory yet (see caveat in §2).
- Three synthetic fixtures, not sampled from real agent sessions. Headroom ships
  `headroom evals probes` for scoring *recorded* sessions — that should be used next.
- Token counts use `cl100k_base` on message `content` strings only; tool-call argument
  JSON and message framing overhead are excluded, so absolute totals are approximate.
  Relative comparisons across arms are unaffected (identical accounting throughout).
- The upstream is a mock; no claim is made here about answer quality, only about tokens.
