# Extension: test-runner output detection

Branch `feat/test-output-detection`, commit `566b0bc` on top of upstream `1390d89`.
Diff: `headroom/transforms/content_detector.py` (+95/-7), plus 8 new tests.

## The gap

`_LOG_PATTERNS` anchors test status to the **start** of a line (`^\s*PASSED|^\s*FAILED`).
No mainstream runner emits it there. Measured against the pattern:

| runner | line | matched? |
|---|---|---|
| pytest | `tests/test_billing.py::test_proration FAILED   [100%]` | **no** |
| go test | `--- PASS: TestFoo (0.00s)` | no |
| cargo | `test result: FAILED. 12 passed; 1 failed` | no |
| jest | `  ✓ renders correctly (12 ms)` | no |

A 400-line pytest log therefore scored a pattern density of **0.005 against a 0.10
threshold — off by 20×** — and fell through to `PLAIN_TEXT`. Whole-log density was 0.0097,
so the hard-coded `[:200]` window was not even the binding constraint.

Consequence: the blob routed to generic text compression. `LogCompressor` compresses that
exact input **95.9%** and emits a CCR retrieval marker, so it would also clear the
reversibility gate (`content_router.py:5577`) that silently discards unmarked lossy output.
The capable compressor was simply unreachable.

## The change

1. **`_TEST_RUNNER_PATTERNS`** — pytest/go/cargo/jest/JUnit status lines, matched
   **case-sensitively** so English prose ("the test passed") cannot trigger them. Tally
   lines require corroboration (a duration, a second count, or mocha's line-leading form)
   rather than a bare `<n> passed`.
2. **Head+tail sampling** replaces `[:200]`. Test logs are back-loaded — failures,
   traceback and summary sit at the end, so a head-only window scores exactly the part
   carrying no signal.
3. **Acceptance**: 5% test-line density with an absolute floor of 3 matches. The generic
   10% path is unchanged.

## Result — library mode, 400-pass/1-fail pytest log

| | baseline | patched |
|---|---|---|
| detection | `text` conf 0.50 | `build` conf 0.98 |
| strategy | `TEXT` | `LOG` |
| transform | `router:noop` | `router:tool_result:log` |
| tokens | 7,976 → 7,976 (**0.0%**) | 7,976 → 388 (**95.1%**) |
| independent tiktoken | 7,893 → 7,893 | 7,893 → 305 (**96.1%**) |

The compressed block preserves the header, the `FAILED` line, the traceback, the summary
tally, and appends `[401 lines omitted: 2 FAIL]` plus a CCR handle.

Cross-runner detection (all `build` after the patch): pytest 0.979, go test 1.000,
cargo 0.966, jest 0.969.

## Tests

`tests/test_content_detector_test_output.py` — 8 tests: 4 runner formats, back-loaded
failure tail, and three false-positive guards (prose containing "3 passed", source code,
single stray status line).

- **patched: 23 passed** (new + existing detection suite)
- **baseline: 5 failed** — the 4 substantive new tests plus the one existing
  exact-metadata assertion updated for the added `test_matches` key.

`go test` passes on baseline too: `--- PASS:` incidentally matches the pre-existing
`^-{3,}` separator pattern. Only pytest/cargo/jest were genuinely broken.

## Honest scope — two things this does NOT fix

1. **A pre-existing false positive.** A changelog containing "5 failed uploads" is
   classified `build` at confidence 1.000 — in **baseline too** (`test_matches: 0`, so none
   of my patterns fired). Cause is `_LOG_PATTERNS[0]` matching `FAILED` case-insensitively.
   Out of scope; documented, not fixed.

2. **Warming the Kompress model makes log compression dramatically worse.** Same input,
   same config, same transform label — the only variable is whether the ML model is loaded:

   | | Kompress cold | Kompress warm |
   |---|---|---|
   | pytest log (patched) | **95.1%**, 909 ch | 9.7%, 20,570 ch |
   | syslog (**baseline detector, patch off**) | 32.9% (`lossless_search`) | 12.4% (`log`) |

   The second row is measured with this extension **disabled**, on a blob the baseline
   detector already classified as `build` — so this is a **pre-existing pipeline defect,
   independent of the extension**, and model readiness also changes which transform is
   selected. It means the headline 95.1% above holds only in the cold-model configuration.
   **This is the most important open item and needs its own investigation.**

## Threats to validity

- Synthetic fixtures, not sampled from recorded agent sessions. `headroom evals probes`
  scores real recordings and should be used next.
- Token savings ≠ task success. No coding agent has been run against this yet; that is the
  evaluation the recruitment task actually asks for and it has not been done.
- Detection improvements are measured on 4 runner formats; no large corpus of real logs.
- The changelog false positive suggests the generic `_LOG_PATTERNS` precision is weak
  overall; a proper evaluation should measure detector precision/recall on a labelled set,
  not just these fixtures.
