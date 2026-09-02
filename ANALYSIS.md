# ANALYSIS

Study and extension of [headroom](https://github.com/chopratejas/headroom), a local-first
context-compression layer for LLM agents.

**Base commit:** `1390d897155e69f8b4554eed5641c2e523860d0f` (2026-09-01) · **package:**
`headroom-ai` 0.37.0 · **platform:** Python 3.13.15, Linux, CPU-only.
**My change:** [`headroom/transforms/content_detector.py`](headroom/transforms/content_detector.py)
(+95/−7) and [`tests/test_content_detector_test_output.py`](tests/test_content_detector_test_output.py).
All harnesses, raw records and supporting notes are under [`analysis/`](analysis/).

**In one paragraph.** Headroom's detector fails to recognise the output of every mainstream
test runner, so a coding agent's most common bulky tool output — a test log — is never
routed to the log compressor that shrinks it 96%. I fixed the detector and measured the
effect with a 20-task coding-agent benchmark: prompt tokens fall **78.4%** (median
**86.6%**) against an uncompressed control, where unmodified headroom saves **0.9%**.
While measuring, I isolated a separate upstream defect in which a pre-emptive optimisation
returns early and skips the strategy's own compressor, costing 12× on the same input. I did
**not** establish that compression preserves task success; that requires model credentials I
did not have, and I make no claim about it.

---

## 1. Features exercised

I ran headroom in **library mode** (`compress(messages)`), as a **proxy**
(`headroom proxy`), and inspected the **MCP/CCR** machinery. Reproduction script:
[`analysis/experiments/matrix.py`](analysis/experiments/matrix.py); raw output
[`analysis/results/matrix.txt`](analysis/results/matrix.txt).

The most interesting components, and what I could confirm about each:

**`LogCompressor` is genuinely well designed.** On a 400-pass/1-fail pytest log it keeps the
session header, the tail of the passing run, the `FAILED` line, the full traceback and the
summary tally, then appends `[401 lines omitted: 2 FAIL]` and a retrieval handle
`[418 lines compressed to 17. Retrieve more: hash=024eb58b…]` — **32,068 → 1,209 chars,
96.2%**, on real `pytest -v` output.

**CCR (reversible compression) is the load-bearing idea.** Originals stay local; the model
receives a marker it can redeem. This is what makes lossy compression defensible, and it has
a consequence most users will never see: at
[`content_router.py:5577`](headroom/transforms/content_router.py#L5577) the router
*discards* any lossy result whose compressor emitted no retrieval marker, keeping the
original verbatim instead (citing issue #1307). In library mode I watched it compute a
**66% reduction and throw it away** for exactly this reason.

**Content-type routing** is the system's hinge. `ContentRouter` maps a detected type to a
strategy; `SmartCrusher` (JSON) worked well and unattended — **46.6%** on a 120-record
GitHub response.

**Config defaults are load-bearing and under-documented.** `enable_code_aware` is
[`False` by default](headroom/transforms/content_router.py#L1503) ("use code graph MCP tools
instead"), so AST code compression never runs in library mode even though it delivers 49.5%
when enabled — the proxy turns it on, the library does not.

**The evaluation framework** (`python -m headroom.evals`) and the offline **session probes**
are the most credible thing in the repo: probes score what compression removed from *your*
recorded sessions with no API key, bucketing facts as retained / recoverable / lost.

### Methodology traps worth recording

Three near-misses, each of which would have produced a wrong number:

1. `protect_recent=4` silently protects short conversations end to end. My first harness had
   three messages and measured nothing.
2. Compressor results are dataclasses. `str(result)` measures the *repr*: my first pass
   reported the log compressor **inflating** input by 7%; it actually saves 95.9%.
3. Running Python from the checkout makes the repo shadow the installed wheel as a namespace
   package, and the checkout has no compiled Rust `_core`. The harness now asserts
   `headroom.__file__` resolves into `site-packages`.

---

## 2. The gap

Measuring three realistic tool outputs in library mode gave a stark split
([`analysis/results/matrix.txt`](analysis/results/matrix.txt)):

| payload | pipeline | compressor called directly |
|---|---:|---:|
| pytest log | **0.0%** (`router:noop`) | 95.9% |
| Python source | **0.0%** (`router:noop`) | 49.5% |
| GitHub JSON | 46.6% | — |

The compressors work; the parts never reach them. I tested and **refuted** three
explanations — Python/onnxruntime version (rebuilt on 3.13: byte-identical), a missing
Kompress model (`is_kompress_available()` returns `True`; the "not ready" banner is stale and
printed on every run), and protection settings (lifting all of them changes
`router:protected:user_message` into `router:noop`, same 0%). Detail in
[`analysis/notes/TRACE-ratio-gate.md`](analysis/notes/TRACE-ratio-gate.md).

The surviving cause is a **detection** failure. `_LOG_PATTERNS` anchors test status to the
*start* of a line (`^\s*PASSED|^\s*FAILED`). No mainstream runner emits it there:

| runner | line | matched |
|---|---|---|
| pytest | `tests/test_billing.py::test_proration FAILED   [100%]` | **no** |
| go test | `--- PASS: TestFoo (0.00s)` | no |
| cargo | `test result: FAILED. 12 passed; 1 failed` | no |
| jest | `  ✓ renders correctly (12 ms)` | no |

A 400-line pytest log scored a pattern density of **0.005 against a 0.10 threshold — off by
20×** — and fell through to `PLAIN_TEXT`. Whole-log density was 0.0097, so the hard-coded
200-line detection window was not even the binding constraint. Only `BUILD_OUTPUT` maps to
the log strategy, so the 96%-effective compressor was unreachable for the single most common
output a coding agent produces.


### Library mode and proxy mode are not the same product

Measured end to end by pointing the proxy at a local recording upstream
([`mock_upstream.py`](analysis/experiments/mock_upstream.py),
[`proxy_probe.py`](analysis/experiments/proxy_probe.py)), so "forwarded" is what the
provider actually received, at zero API cost
([`proxy-arms.txt`](analysis/results/proxy-arms.txt)):

| payload | library | proxy default (`--mode cache`) | proxy `--mode token --code-aware` |
|---|---:|---:|---:|
| pytest log | 0.0% | 0.0% | 0.0% |
| Python source | 0.0% | 0.0% | **44.1%** |
| GitHub JSON | 46.6% | 0.0% | 46.8% |

Two things follow. The 66% that library mode computes and discards **is** recoverable — with
CCR active through the proxy, source code compresses 44.1%. And the proxy's *default* mode
forwarded all three payloads unchanged. I flag that second number as **weakly supported**:
`--mode cache` freezes prior turns to protect the provider's prefix cache, and this probe is
single-shot, so it may understate multi-turn behaviour badly. It should not be quoted as
"the default proxy saves nothing" without a multi-turn measurement.

---

## 3. The extension

Commit `566b0bc`, branch `feat/test-output-detection`. Three changes to
[`content_detector.py`](headroom/transforms/content_detector.py):

1. **`_TEST_RUNNER_PATTERNS`** — pytest/go/cargo/jest/JUnit status lines, matched
   **case-sensitively**, because runners shout `PASSED` and English prose says "passed".
   Tally lines require corroboration (a duration, a second count, or mocha's line-leading
   form) rather than a bare `<n> passed`.
2. **Head + tail sampling** replaces `[:200]`. Test logs are back-loaded — failure,
   traceback and summary sit at the bottom — so a head-only window scores precisely the
   region carrying no signal.
3. **Acceptance** on a 5% test-line density with an absolute floor of 3 matches; the generic
   10% path is untouched.

I built the false-positive guards *after* my first version misclassified a paragraph
containing "3 passed" as build output. [Eight
tests](tests/test_content_detector_test_output.py) cover four runner formats, a
back-loaded failure tail, and three negative cases (prose, source code, a lone status line).
Against the patched detector the detection suite is **23 passed**; against the baseline, **5
fail** — the four substantive new tests plus one existing exact-metadata assertion I updated
for an added `test_matches` key. `go test` passes on baseline too: `--- PASS:` incidentally
matches a pre-existing separator pattern, so only pytest/cargo/jest were genuinely broken.

**Scope I deliberately did not extend.** A changelog containing "5 failed uploads" is
misclassified as build output at confidence 1.000 — **in the baseline as well**
(`test_matches: 0`, so none of my patterns fired). It comes from a pre-existing
case-insensitive `FAILED` pattern. Documented, not fixed.

---

## 4. Evaluation

Harness: [`analysis/eval/`](analysis/eval/). Records:
[`analysis/results/eval/`](analysis/results/eval/). Design in
[`analysis/notes/EVALUATION.md`](analysis/notes/EVALUATION.md).

**Benchmark.** 20 bug-fix tasks ([`tasks.py`](analysis/eval/tasks.py)), each a small Python
package with a seeded single-token bug and a 400-test suite in which one test fails. The
agent must run the suite, read the failure, locate the line, fix it. Success is decided by
**re-running the suite**, not by the model's claim. Every task was validated before use —
must fail as generated, must pass once the bug is reverted
([`validate_tasks.py`](analysis/eval/validate_tasks.py)): 20/20 usable.

The suite runs `pytest -v` (~414 lines, 32 KB, ~8,000 tokens). This matters: my first version
used `pytest -q`, which emits 18 lines and would have made the experiment meaningless.

**Agent.** A minimal three-tool agent ([`agent.py`](analysis/eval/agent.py)) —
`run_tests`, `read_file`, `write_file` — OpenAI chat-completions, temperature 0, 10-turn cap.
Token counts come from the provider's own `usage.prompt_tokens`, i.e. what is billed.

**Arms**, identical but for the path the prompt takes; paired, every arm runs every task on a
pristine copy in fixed order.

### Results

| arm | mean/task | median | total (20) | saved vs control |
|---|---:|---:|---:|---:|
| no compression | 25,938 | 25,919 | 518,767 | — |
| headroom, upstream detector | 25,696 | 25,910 | 513,911 | **0.9%** |
| headroom + extension | 5,600 | 3,486 | 112,008 | **78.4%** |

Paired per-task reduction, bootstrap 95% CI over tasks (10,000 resamples):
upstream **0.9% [−0.0, 2.3]** (2/20 tasks improved); with the extension **78.4% [65.9, 86.7]**
(19/20 improved).

The distribution is **bimodal**, so the median is the honest headline: **18/20 tasks compress
at a median 86.6%**, two do not compress at all. Reporting the mean alone would misdescribe
it in both directions.

**Reproducibility.** A full replicate produced **0/20 classification flips**
([`records-replicate.json`](analysis/results/eval/records-replicate.json)). Reversing task
order moved the two failures from positions 6–7 to 14–15 and they **still failed** with
near-identical counts ([`records-reversed-order.json`](analysis/results/eval/records-reversed-order.json)),
refuting the natural hypothesis that they reflect background model-load timing.

**One design worry closed empirically.** Headroom's proxy can inject a `headroom_retrieve`
tool my agent does not implement, which would handicap the compressed arms. The agent counts
such calls: `unserviced_tool_calls` is **0 across all 60 runs**.

---

## 5. A second, upstream defect

While validating the extension I found the measured saving depended on whether the Kompress
ML model was loaded — 95.1% cold, 9.7% warm, deterministic across repeats. Root cause at
[`content_router.py:3241`](headroom/transforms/content_router.py#L3241): for `LOG`/`SEARCH`,
a `relevance_split` path runs **before** the built-in dispatch and returns early whenever it
produces any result. Its self-gate compares against the lossless fold floor — never against
**the dedicated compressor it is pre-empting**.

A 2×2 isolates it ([`coldwarm-2x2.txt`](analysis/results/coldwarm-2x2.txt)): 95.2% in three
cells, **9.7%** only when the model is loaded *and* the split is enabled — which is the
shipped default. The good case works **by accident**: with no model the split fails, returns
`None`, and execution falls through to the compressor that should have run anyway. The same
signature appears on a syslog with my extension **disabled** (32.9% cold vs 12.4% warm), so
this is upstream, not something I introduced. I diagnosed but did **not** fix it: the right
change is to carry the split as a candidate into the existing best-of selection rather than
return early, and that deserves its own evaluation.

---

## 6. What remains uncertain

**Task success is not measured.** No API credentials were available, so the model was a
deterministic scripted stand-in ([`mock_provider.py`](analysis/eval/mock_provider.py)). It
counts tokens over the messages it *actually receives* after compression, so **the token
figures are a true end-to-end measurement**; but it already knows the fix, so it is blind to
information loss by construction. The `solved` column reads 20/20 in every arm and **means
nothing**. Whether an agent still solves these tasks from compressed context is the question
the task centres on and it is open. The harness needs only `--provider-url`, `--model` and a
key; ~1.6M prompt tokens across three arms is roughly $0.25 on gpt-4o-mini.

**Two of twenty tasks are unexplained.** `comparison_flip` and `off_by_one_range` forward
essentially uncompressed. I eliminated, by direct test on those exact logs: detection (all
`build` @1.00), LogCompressor capability (95–96% standalone), the lossless-fold early return
(`None` for all three), `compact_lossless`, error-indicator protection (`True` for all
alike), ordering, and nondeterminism. One task's forwarded text has lines *merged*, so a
weaker transform ran — plausibly the same family as §5 — but the proxy exposes no router
decisions even at `HEADROOM_LOG_LEVEL=DEBUG` and **I never observed the branch**. It is an
open question, not a diagnosis.

**External validity.** The benchmark is synthetic, single-module, single-log-format
(pytest), one agent scaffold, and uniform bug difficulty — every bug is reachable from the
assertion, so the log always suffices. A standard benchmark (SWE-bench Lite, Aider polyglot)
would carry more weight; I chose this one because it reliably produces the bulky test logs
the extension acts on, which most benchmark tasks do not. The go/cargo/jest patterns are
covered by unit tests but not end to end.

**Not claimed.** Nothing here measures answer quality, latency, or cost after provider
prefix-cache effects — compression that busts a KV-cache prefix can raise billed cost while
lowering token count, and I did not measure that.

---

## 7. Conclusion

Headroom's compressors are strong and its reversibility design is sound. Its weakness on the
workload it advertises — coding agents — is **routing, not compression**: as shipped, it
saves 0.9% on a realistic agent trajectory because a detector rule never matches real test
output. A 95-line detector change lifts that to a median 86.6%. A second, independent defect
means even the corrected path silently under-delivers by 12× in the default configuration.
Both are cheap to fix and neither required understanding the ML model — they required
measuring what the tool actually does rather than what its README says.

---

## Appendix — reproducing this

```bash
# environment (Python 3.11+ required: on 3.10 pip resolves onnxruntime <1.24
# and headroom silently disables native content detection)
uv venv --python 3.13 .venv && uv pip install "headroom-ai[proxy,code,evals,ml]"

# 1. compression matrix, library mode, patched vs baseline detector
python analysis/experiments/matrix.py > analysis/results/matrix.txt

# 2. the cold/warm defect, 2x2 isolation
WARM=1 NOSPLIT=0 python analysis/experiments/fixtest.py   # and the other 3 cells

# 3. detector unit tests
pytest tests/test_content_detector_test_output.py  # 8 passed

# 4. the coding-agent evaluation (scripted provider, no API key needed)
python analysis/eval/validate_tasks.py             # 20/20 tasks usable
python analysis/eval/run_eval.py --mock            # writes runs/records.json
python analysis/eval/analyze.py                    # paired stats + bootstrap CIs

# 4b. the same evaluation against a real model
python analysis/eval/run_eval.py \
    --provider-url https://api.openai.com/v1 \
    --provider-host https://api.openai.com \
    --model gpt-4o-mini --api-key "$OPENAI_API_KEY"
```

Supporting notes, in the order the work happened:
[DAY1-FINDINGS](analysis/notes/DAY1-FINDINGS.md) ·
[TRACE-ratio-gate](analysis/notes/TRACE-ratio-gate.md) ·
[PROXY-COMPARISON](analysis/notes/PROXY-COMPARISON.md) ·
[EXTENSION](analysis/notes/EXTENSION.md) ·
[COLDWARM](analysis/notes/COLDWARM.md) ·
[EVALUATION](analysis/notes/EVALUATION.md)
