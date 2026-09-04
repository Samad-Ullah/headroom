# ANALYSIS

Study and extension of [headroom](https://github.com/chopratejas/headroom), a local-first
context-compression layer for LLM agents.

Base commit `1390d897155e69f8b4554eed5641c2e523860d0f` (2026-09-01), package `headroom-ai`
0.37.0, on Python 3.13.15 / Linux / CPU-only. My change is confined to
[`headroom/transforms/content_detector.py`](headroom/transforms/content_detector.py)
(+95/−7) plus [`tests/test_content_detector_test_output.py`](tests/test_content_detector_test_output.py).
Harnesses, raw records and working notes are under [`analysis/`](analysis/).

Headroom's content detector does not recognise the output of any mainstream test runner, so
the bulkiest thing a coding agent produces all day — a test log — never reaches the log
compressor that would shrink it by 96%. I fixed the detector and measured the effect on a
20-task coding-agent benchmark. Prompt tokens fall 78.4% (median 86.6%) against an
uncompressed control; unmodified headroom saves 0.9% on the same workload. Along the way I
isolated a second, unrelated defect in which a pre-emptive optimisation returns early and
skips the compressor it displaced, costing 12× on the same input. I did not establish that
compression preserves task success, and I say so at length in §6.

---

## 1. Features exercised

I ran headroom three ways: as a library (`compress(messages)`), as a proxy
(`headroom proxy`), and through its MCP/CCR machinery. The reproduction script is
[`analysis/experiments/matrix.py`](analysis/experiments/matrix.py) and its output is
[`analysis/results/matrix.txt`](analysis/results/matrix.txt).

The log compressor deserves the most credit. Given a 400-pass/1-fail pytest log it keeps the
session header, the tail of the passing run, the `FAILED` line, the whole traceback and the
summary tally, then appends `[401 lines omitted: 2 FAIL]` and a retrieval handle
`[418 lines compressed to 17. Retrieve more: hash=024eb58b…]`. On real `pytest -v` output
that is 32,068 → 1,209 characters, a 96.2% reduction, with the diagnostic content intact.
It is a genuinely good piece of engineering and almost nothing in this report reaches it.

CCR — reversible compression — is the idea the whole design rests on. Originals stay on
disk and the model receives a marker it can redeem. That is what makes lossy compression
defensible rather than reckless, and it has a consequence users will never see: at
[`content_router.py:5577`](headroom/transforms/content_router.py#L5577) the router discards
any lossy result whose compressor emitted no retrieval marker, keeping the original verbatim
instead, citing issue #1307. In library mode I watched it compute a 66% reduction and throw
it away for exactly that reason. Conservative, and defensible, but expensive.

Content-type routing is the system's hinge, and `SmartCrusher` (JSON) worked unattended,
giving 46.6% on a 120-record GitHub response. Configuration defaults turn out to carry more
weight than the documentation suggests: `enable_code_aware` is
[`False` by default](headroom/transforms/content_router.py#L1503), commented "use code graph
MCP tools instead", so AST code compression never runs in library mode even though it
delivers 49.5% once enabled. The proxy switches it on; the library does not. Nothing warns
you.

The evaluation framework (`python -m headroom.evals`) and the offline session probes are the
most credible thing in the repository. Probes score what compression removed from your own
recorded sessions without an API key, bucketing facts as retained, recoverable or lost.

### Three ways I nearly measured the wrong thing

Worth recording, because each produced a plausible number before I caught it.

`protect_recent=4` silently protects short conversations end to end, so my first harness —
three messages long — measured nothing at all and reported it as 0% compression.

Compressor results are dataclasses, not strings. Calling `str(result)` measures the repr.
My first pass reported the log compressor *inflating* its input by 7%; it actually saves
95.9%. That one is embarrassing but instructive: the number was precise, reproducible and
entirely wrong.

Running Python from the checkout makes the repo shadow the installed wheel as a namespace
package, and the checkout has no compiled Rust `_core`. The harness now asserts that
`headroom.__file__` resolves inside `site-packages` and fails loudly otherwise.

---

## 2. The gap

Measuring three realistic tool outputs in library mode split cleanly
([`analysis/results/matrix.txt`](analysis/results/matrix.txt)):

| payload | through the pipeline | compressor called directly |
|---|---:|---:|
| pytest log | 0.0% (`router:noop`) | 95.9% |
| Python source | 0.0% (`router:noop`) | 49.5% |
| GitHub JSON | 46.6% | — |

The compressors work. The content never reaches them. I proposed three explanations and all
three were wrong: the Python/onnxruntime version (I rebuilt the environment on 3.13 and got
byte-identical results), a missing Kompress model (`is_kompress_available()` returns `True`;
the "not ready" banner is stale and prints on every run), and protection settings (lifting
every one of them turns `router:protected:user_message` into `router:noop` and changes
nothing). The detail is in
[`analysis/notes/TRACE-ratio-gate.md`](analysis/notes/TRACE-ratio-gate.md). Being wrong
three times was the most useful part of the week — each failed explanation removed a way of
dismissing the result, so what survived was in the tool rather than on my machine.

What survives is a detection failure. `_LOG_PATTERNS` anchors test status to the start of a
line (`^\s*PASSED|^\s*FAILED`), and no mainstream runner puts it there:

| runner | line | matched |
|---|---|---|
| pytest | `tests/test_billing.py::test_proration FAILED   [100%]` | no |
| go test | `--- PASS: TestFoo (0.00s)` | no |
| cargo | `test result: FAILED. 12 passed; 1 failed` | no |
| jest | `  ✓ renders correctly (12 ms)` | no |

A 400-line pytest log scored a pattern density of 0.005 against a 0.10 threshold, off by a
factor of twenty, and fell through to `PLAIN_TEXT`. Density over the whole log was 0.0097,
so the hard-coded 200-line detection window was not even the binding constraint — the
patterns themselves were. Since only `BUILD_OUTPUT` maps to the log strategy, a
96%-effective compressor sat unreachable behind a regular expression.

### Library mode and proxy mode are not the same product

I measured this end to end by pointing the proxy at a local recording upstream
([`mock_upstream.py`](analysis/experiments/mock_upstream.py),
[`proxy_probe.py`](analysis/experiments/proxy_probe.py)), so "forwarded" means what the
provider actually received, at zero API cost
([`proxy-arms.txt`](analysis/results/proxy-arms.txt)):

| payload | library | proxy default (`--mode cache`) | proxy `--mode token --code-aware` |
|---|---:|---:|---:|
| pytest log | 0.0% | 0.0% | 0.0% |
| Python source | 0.0% | 0.0% | 44.1% |
| GitHub JSON | 46.6% | 0.0% | 46.8% |

Two things follow. The 66% that library mode computes and discards is genuinely
recoverable: with CCR active through the proxy, source code compresses 44.1%. And the
proxy's default mode forwarded all three payloads unchanged. I flag that second figure as
weakly supported — `--mode cache` freezes prior turns to protect the provider's prefix
cache, and my probe is single-shot, so it may understate multi-turn behaviour badly. Nobody
should quote it as "the default proxy saves nothing" without a multi-turn measurement, and
that is a measurement I did not have time to build.

---

## 3. The extension

Commit `034bb000` on branch `feat/test-output-detection`, three changes to
[`content_detector.py`](headroom/transforms/content_detector.py).

First, a `_TEST_RUNNER_PATTERNS` table covering pytest, go, cargo, jest and JUnit status
lines, matched case-sensitively. That last detail matters more than it looks: runners shout
`PASSED` while English prose says "passed", and case is what separates a log from a
paragraph about a log. Tally lines need corroboration — a duration, a second count, or
mocha's line-leading form — rather than a bare `<n> passed`.

Second, head-and-tail sampling replaces `[:200]`. Test logs are back-loaded. The failure,
the traceback and the summary all sit at the bottom, so scoring only the head judges the log
on precisely the region carrying no signal.

Third, acceptance on a 5% test-line density with an absolute floor of three matches. The
generic 10% path is untouched.

I built the false-positive guards only after my first version confidently classified a
paragraph containing "3 passed" as build output, which is how the case-sensitivity rule came
about. [Eight tests](tests/test_content_detector_test_output.py) cover four runner formats,
a back-loaded failure tail, and three negative cases: prose, source code, and a lone status
line. Against the patched detector the detection suite reports 23 passed; against the
baseline, 5 fail — the four substantive new tests plus one existing exact-metadata assertion
I updated for an added `test_matches` key. Worth noting that `go test` passes on the
baseline too, because `--- PASS:` happens to match a pre-existing separator pattern. Only
pytest, cargo and jest were genuinely broken.

One thing I deliberately left alone. A changelog containing "5 failed uploads" is
misclassified as build output at confidence 1.000 — in the baseline as well, with
`test_matches: 0` proving none of my patterns fired. It comes from a pre-existing
case-insensitive `FAILED` pattern. It is a real problem and it is not mine to fix here.

---

## 4. Evaluation

Harness in [`analysis/eval/`](analysis/eval/), records in
[`analysis/results/eval/`](analysis/results/eval/), design notes in
[`analysis/notes/EVALUATION.md`](analysis/notes/EVALUATION.md).

The benchmark is 20 bug-fix tasks ([`tasks.py`](analysis/eval/tasks.py)), each a small
Python package carrying a seeded single-token bug and a 400-test suite in which one test
fails. The agent must run the suite, read the failure, find the line and fix it. Success is
decided by re-running the suite afterwards, never by the model's claim to have fixed it.
Every task was validated before use — it must fail as generated and pass once the bug is
reverted ([`validate_tasks.py`](analysis/eval/validate_tasks.py)) — and all 20 qualified.

The suite runs under `pytest -v`, producing roughly 414 lines, 32 KB, about 8,000 tokens.
That detail nearly sank the experiment: my first version used `pytest -q`, which emits 18
lines, and would have measured compression on a payload with nothing to compress.

The agent ([`agent.py`](analysis/eval/agent.py)) has three tools — `run_tests`,
`read_file`, `write_file` — speaks the OpenAI chat-completions API at temperature 0, and
stops after ten turns. Token counts come from the provider's own `usage.prompt_tokens`,
which is what gets billed. The three arms differ in exactly one respect, the path the prompt
takes to the model, and the design is paired: every arm runs every task on a pristine copy,
in a fixed order.

### Results

| arm | mean/task | median | total (20) | saved vs control |
|---|---:|---:|---:|---:|
| no compression | 25,938 | 25,919 | 518,767 | — |
| headroom, upstream detector | 25,696 | 25,910 | 513,911 | 0.9% |
| headroom + extension | 5,600 | 3,486 | 112,008 | 78.4% |

Paired per-task reduction with bootstrap 95% confidence intervals over tasks (10,000
resamples): upstream 0.9% [−0.0, 2.3], improving 2 tasks of 20; with the extension 78.4%
[65.9, 86.7], improving 19 of 20.

The distribution is bimodal, so the median is the honest headline rather than the mean: 18
tasks of 20 compress at a median 86.6%, and two do not compress at all. Quoting the mean
alone would misdescribe the result in both directions.

On reproducibility, a full replicate produced zero classification flips across all 20 tasks
([`records-replicate.json`](analysis/results/eval/records-replicate.json)). Reversing the
task order moved the two failures from positions 6–7 to positions 14–15 and they still
failed, with near-identical counts
([`records-reversed-order.json`](analysis/results/eval/records-reversed-order.json)). That
refuted my own hypothesis that they reflected background model-load timing, which is the
explanation I had expected to confirm.

One design worry closed itself empirically. Headroom's proxy can inject a
`headroom_retrieve` tool my agent does not implement, which would quietly handicap the
compressed arms. The agent counts such calls, and `unserviced_tool_calls` is 0 across all
60 runs.

---

## 5. A second, upstream defect

While validating the extension I noticed the measured saving depended on whether the
Kompress ML model happened to be loaded: 95.1% cold, 9.7% warm, deterministic across
repeats. The cause is at
[`content_router.py:3241`](headroom/transforms/content_router.py#L3241). For `LOG` and
`SEARCH` strategies a `relevance_split` path runs before the built-in dispatch and returns
early whenever it produces any result at all. Its self-gate compares against the lossless
fold floor, never against the dedicated compressor it is pre-empting.

A 2×2 isolates it ([`coldwarm-2x2.txt`](analysis/results/coldwarm-2x2.txt)): 95.2% in three
cells and 9.7% in the fourth, the one where the model is loaded and the split is enabled,
which is the shipped default. The good case works by accident. With no model available the
split fails, returns `None`, and execution falls through to the compressor that should have
run in the first place. The same signature appears on a syslog with my extension disabled
(32.9% cold against 12.4% warm), so this is upstream and not something I introduced.

I diagnosed it but did not fix it. The right change is to carry the split as a candidate
into the existing best-of selection instead of returning early, and that deserves its own
evaluation rather than being bolted onto this one.

---

## 6. What remains uncertain

Task success is not measured, and this is the largest hole in the work. No API credentials
were available on the evaluation machine, so the model was a deterministic scripted stand-in
([`mock_provider.py`](analysis/eval/mock_provider.py)). It counts tokens over the messages
it actually receives after compression, which is why the token figures above are a true
end-to-end measurement. But it already knows the fix, so it is blind to information loss by
construction. The `solved` column reads 20/20 in every arm and means nothing whatsoever.
Whether an agent still solves these tasks from compressed context is the question this task
centres on and I have not answered it. The harness needs only `--provider-url`, `--model`
and a key; roughly 1.6M prompt tokens across three arms is about $0.25 on gpt-4o-mini.

Two of the twenty tasks are unexplained. `comparison_flip` and `off_by_one_range` forward
essentially uncompressed. Tested directly on those exact logs I eliminated detection (all
three detect `build` at confidence 1.00), LogCompressor capability (95–96% standalone), the
lossless-fold early return (`None` for all three), `compact_lossless`, error-indicator
protection (`True` for all alike), ordering, and nondeterminism. One task's forwarded text
has lines merged, so some weaker transform ran, plausibly the same family as §5. But the
proxy exposes no router decisions even at `HEADROOM_LOG_LEVEL=DEBUG` and I never observed
the branch. It is an open question, not a diagnosis, and I would rather leave it that way
than dress up a guess.

External validity is limited. The benchmark is synthetic, single-module, single-log-format,
one agent scaffold, and every bug is a single-token edit reachable from the assertion, so
the log always suffices to solve the task. A standard benchmark such as SWE-bench Lite or
the Aider polyglot set would carry more weight. I chose this one because it reliably
produces the bulky test logs the extension acts on, which most benchmark tasks do not — but
that choice trades generality for relevance and a reader should discount accordingly. The
go, cargo and jest patterns are covered by unit tests but never exercised end to end.

Finally, what is not claimed at all: nothing here measures answer quality, latency, or cost
once provider prefix-cache effects are included. Compression that busts a KV-cache prefix
can raise the bill while lowering the token count, and I did not measure that.

---

## 7. Conclusion

Headroom's compressors are strong and its reversibility design is sound. Its weakness on the
workload it advertises for — coding agents — is routing, not compression. As shipped it
saves 0.9% on a realistic agent trajectory, because a detector rule never matches real test
output. Ninety-five lines of detector change lift that to a median 86.6%. A second,
independent defect means even the corrected path under-delivers by 12× in the default
configuration.

Neither problem required understanding the machine-learning model, and neither is
expensive to fix. What they required was measuring what the tool does rather than reading
what it says it does.

---

## Appendix — reproducing this

```bash
# Python 3.11+ matters: on 3.10 pip resolves onnxruntime <1.24 and headroom
# silently falls back to pure-Python content detection.
uv venv --python 3.13 .venv && uv pip install "headroom-ai[proxy,code,evals,ml]"
export PYTHON="$PWD/.venv/bin/python"        # toggle_patch.sh asks this interpreter

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

Working notes, in the order the investigation happened:
[DAY1-FINDINGS](analysis/notes/DAY1-FINDINGS.md) ·
[TRACE-ratio-gate](analysis/notes/TRACE-ratio-gate.md) ·
[PROXY-COMPARISON](analysis/notes/PROXY-COMPARISON.md) ·
[EXTENSION](analysis/notes/EXTENSION.md) ·
[COLDWARM](analysis/notes/COLDWARM.md) ·
[EVALUATION](analysis/notes/EVALUATION.md)
