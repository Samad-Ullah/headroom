# The cold/warm anomaly: `relevance_split` pre-empts the strategy's own compressor

Data: [`results/coldwarm-2x2.txt`](../results/coldwarm-2x2.txt) ·
harness: [`experiments/fixtest.py`](../experiments/fixtest.py),
[`experiments/instr2.py`](../experiments/instr2.py)
repo @ `1390d89` + `feat/test-output-detection` (`566b0bc`), headroom-ai 0.37.0.

## Symptom

The same 400-pass/1-fail pytest log, same config, same transform label
(`router:tool_result:log`), compresses **95.1%** when the Kompress ML model is not
loaded and **9.7%** when it is. Deterministic: 6/6 warm runs and 4/4 cold runs identical.

## Root cause

`content_router.py:3241`, in the block dispatch, **before** the built-in compressor
if/elif chain:

```python
if self.config.relevance_split and strategy in (
    CompressionStrategy.LOG,
    CompressionStrategy.SEARCH,
):
    kind = "log" if strategy is CompressionStrategy.LOG else "search"
    split = self._relevance_split_compress(content, kind, context)
    if split is not None:
        return split, _estimate_tokens(split), [kind, "relevance_split"]   # ← early return
```

`relevance_split` keeps high-relevance records verbatim and sends the low-value tail to
Kompress. Per its own comment it "self-gates on beating the whole-block fold" — the
**STAGE-0 lossless fold**. It never compares itself against **the dedicated compressor for
the strategy it is pre-empting**, and it `return`s, so `LogCompressor` (dispatched ~160
lines later) never runs.

Instrumented call trace makes the mechanism explicit:

| | `_try_ml_compressor` | `_relevance_split_compress` | reaches LogCompressor? |
|---|---|---|---|
| cold | 22,264 → 22,264 (Kompress absent, no-op) | **None** | yes → **95.2%** |
| warm | 22,264 → 20,570 (7.6%) | **20,570** | **no** → 9.7% |

Cold only "works" by accident: with Kompress unavailable the split cannot beat the floor,
returns `None`, and execution falls through to the compressor that should have run anyway.

## Causal isolation (2×2)

| Kompress | `relevance_split` | tokens | saved |
|---|---|---:|---:|
| cold | on | 7,968 → 380 | 95.2% |
| cold | off | 7,968 → 380 | 95.2% |
| **warm** | **on** | 7,968 → 7,197 | **9.7%** |
| warm | off | 7,968 → 380 | 95.2% |

Neutralising `_relevance_split_compress` alone restores full compression while warm. The
interaction is exact: degradation requires **both** a loaded model and `relevance_split`.

## Why this matters in practice

`relevance_split: bool = True` (`content_router.py:1639`) — on by default — and the proxy
loads Kompress. **The warm+split cell is the normal production configuration.** So on this
payload the deployed default yields 9.7% where 95.2% is available: a **12× shortfall**, on
the single most common tool output a coding agent produces.

It also explains the earlier baseline observation in
[`EXTENSION.md`](EXTENSION.md): a syslog the *unpatched* detector already classified as
`build` compressed 32.9% cold vs 12.4% warm. Same mechanism, no dependency on this
extension — **this is an upstream defect, not something the extension introduced.**

## Suggested fix (not yet implemented)

Gate the split against the strategy's own compressor rather than only the lossless fold:
compute the built-in result first (or make the split's self-gate consult it) and adopt
`relevance_split` only when it is actually smaller. Equivalent minimal change: do not
`return` early — carry the split as a *candidate* into the existing selection logic that
already picks the best of several results.

## Method note — two false conclusions on the way here

Both recorded because they nearly produced wrong findings:

1. **A silent no-op toggle.** Disabling the feature via
   `ContentRouterConfig.relevance_split = False` changed nothing, which looked like
   exoneration. Setting a class attribute does **not** change a dataclass `__init__`
   default captured at class creation, so the flag was never actually off. The correct
   test replaces the method.
2. **A false negative from bad instrumentation.** A spy that called
   `_try_ml_compressor(self, content, context, question)` positionally raised on the real
   signature; `content_router.py:3268` catches `Exception` and sets `_komp = None`, which
   silently produces the *good* result. The anomaly appeared to vanish under observation.
   Signature-safe `*args, **kwargs` spies fixed it.

## Threats to validity

- One payload shape (pytest log) plus one syslog. The magnitude of the shortfall on other
  log formats is unmeasured.
- Measured in library mode. The proxy path was not re-measured after this diagnosis; the
  earlier proxy arms ran before the detector patch existed, so the production-configuration
  shortfall is **inferred** from defaults (`relevance_split=True`, proxy loads Kompress),
  not directly observed end-to-end. That measurement is the obvious next step.
