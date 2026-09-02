# Coding-agent evaluation

Harness: [`eval/`](../eval) · records + summary: [`results/eval/`](../results/eval)
repo @ `1390d89` + `feat/test-output-detection` (`566b0bc`), headroom-ai 0.37.0, Python 3.13.15.

## Design

**Benchmark.** 20 bug-fix tasks. Each is a small Python package with a seeded
single-token bug (operator flip, off-by-one, inverted guard, wrong default, …) and a
400-test suite in which one test fails. The agent must run the suite, read the failure,
locate the line and fix it. Success is decided by **running the suite afterwards**, not by
the model's claim.

Every task was validated before use: the suite must fail as generated and pass once the
seeded bug is reverted ([`eval/validate_tasks.py`](../eval/validate_tasks.py)). 20/20 usable.

The suite is deliberately verbose (`pytest -v`, ~414 lines / 32 KB / ~8,000 tokens) because
the object of study is what happens to a **bulky test log** on the way to the model. This
is real pytest output, not the synthetic fixture used earlier — and the extension's
behaviour on it was confirmed independently (baseline detects `text` @0.50, patched detects
`build` @1.00).

**Agent.** A minimal tool-using agent ([`eval/agent.py`](../eval/agent.py)) with three
tools — `run_tests`, `read_file`, `write_file` — speaking the OpenAI chat-completions API,
temperature 0, 10-turn cap. Token counts come from the provider's own `usage.prompt_tokens`,
i.e. what is actually billed.

**Arms.** Identical in every respect except the path the prompt takes:

| arm | path |
|---|---|
| `none` | agent → provider (uncompressed control) |
| `headroom-baseline` | agent → headroom proxy → provider, upstream detector |
| `headroom-extension` | agent → headroom proxy → provider, patched detector |

Paired design: every arm runs every task on a pristine copy, in fixed order.

## Result — prompt tokens

| arm | mean/task | median | total (20 tasks) | saved vs control |
|---|---:|---:|---:|---:|
| `none` | 25,938 | 25,919 | 518,767 | — |
| `headroom-baseline` | 25,696 | 25,910 | 513,911 | **0.9%** |
| `headroom-extension` | 5,600 | 3,486 | 112,008 | **78.4%** |

Paired per-task reduction, bootstrap 95% CI over tasks (10,000 resamples):

- `headroom-baseline`: **0.9%** [−0.0, 2.3] — 2/20 tasks improved
- `headroom-extension`: **78.4%** [65.9, 86.7] — 19/20 tasks improved

The result is **bimodal**, so the median matters more than the mean: **18/20 tasks compress
at a median 86.6%**, and 2 do not compress at all. Reporting the mean alone would
misdescribe the distribution in both directions.

## Reproducibility

- **Replicate run**: identical arm re-run end to end — **0/20 classification flips**, per-task
  figures identical to within ±30 tokens.
- **Order reversal**: with task order reversed, the two failing tasks moved from positions
  6–7 to positions 14–15 and **still failed**, with near-identical counts (23,555 / 26,071 vs
  23,555 / 26,072). The anomaly tracks the *task*, not the position — which refutes the
  natural hypothesis that it reflects background model-load timing.

## The unexplained 2/20

`comparison_flip` and `off_by_one_range` are forwarded essentially uncompressed
(24,262 and 32,896 chars) while comparable tasks compress to ~1,210. Eliminated as causes,
each tested directly on those exact logs:

| candidate | result |
|---|---|
| content detection | all three detect `build` @ confidence 1.00 |
| `LogCompressor` capability | 95.2–96.3% on all three, standalone |
| lossless fold early-return | `_lossless_first` returns `None` for all three |
| `compact_lossless` | returns input unchanged for all three |
| error-indicator protection | `strong_error=True` for all three alike |
| run ordering / model-load timing | refuted by the reversal experiment |
| run-to-run nondeterminism | refuted by the replicate (0 flips) |

`comparison_flip`'s forwarded text has had lines **merged** (`===== 400 items
test_reportlib.py::…` where the original has `collected 401 items` on its own line), so
*some* transform ran and produced a much weaker result than LogCompressor would have — the
same family as the `relevance_split` defect in [`COLDWARM.md`](COLDWARM.md), where the
router commits to the first path yielding output without comparing it against the
strategy's dedicated compressor. **This is a hypothesis, not a confirmed diagnosis**: the
proxy does not expose router decisions even at `HEADROOM_LOG_LEVEL=DEBUG`, and the branch
was never directly observed. It is recorded as an open question.

## What this evaluation does NOT establish

**It does not measure task success.** No API credentials were available on the evaluation
machine, so the model was a deterministic scripted stand-in
([`eval/mock_provider.py`](../eval/mock_provider.py)) that plays a fixed competent agent:
run tests → read module → write the known fix → stop.

- **What that makes valid**: the token figures. The stand-in counts `prompt_tokens` with
  tiktoken over the messages it *actually receives*, after the proxy has compressed them, so
  every number above is a true end-to-end measurement of compression on the wire.
- **What that makes invalid**: the `solved` column. It is 20/20 in every arm because the
  script already knows the answer; it is blind to information loss **by construction**. No
  claim about accuracy, task success, or agent behaviour under compression can be drawn from
  this run, and none is made.

The task asks whether compression preserves the agent's ability to do the job. **That
question is still open.** Running these same 20 tasks against a real model would answer it;
the harness needs only `--provider-url`, `--model` and a key, and at ~26k prompt tokens ×
20 tasks × 3 arms ≈ **1.6M prompt tokens**, that is roughly **$0.25 on gpt-4o-mini** or a
few dollars on a frontier model.

One useful negative already available: `unserviced_tool_calls` is **0 across all 60 runs**,
so the compressed arms were never handicapped by the agent's inability to service an
injected `headroom_retrieve` call. That design worry is empirically dead.

## Other threats to validity

- **Synthetic benchmark.** Tasks are generated, not drawn from real repositories, and all
  share one module shape. A standard benchmark (SWE-bench Lite, Aider polyglot) would carry
  more external validity; this one was chosen because it reliably produces the bulky test
  logs the extension acts on, which most benchmark tasks do not.
- **Single log format.** All 20 tasks produce pytest output. The go/cargo/jest patterns are
  covered by unit tests but not by this end-to-end evaluation.
- **One agent, one scaffold.** A different agent (different prompt, different turn budget,
  a summarisation step) could change the token profile substantially.
- **Uniform bug difficulty.** Every bug is a single-token edit reachable from the assertion,
  so the log is always sufficient to solve the task. Harder tasks needing more exploration
  would exercise compression differently.
