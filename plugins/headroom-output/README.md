# headroom-output

Output-token shaping for Headroom.

Headroom's core compresses what goes **into** a model. This plugin reduces what comes **out** of one.

## Why output tokens

From Headroom's own price table (`headroom/pricing/anthropic_prices.py`, Sonnet), normalised to the cached-input rate:

| | $ / 1M | relative |
|---|---:|---:|
| Cached input | 0.30 | 1× |
| Fresh input | 3.00 | 10× |
| **Output** | **15.00** | **50×** |

The ratio holds across the table — Haiku 0.80/4.00, Opus 15.00/75.00. Thinking bills as output, and agent harnesses commonly pin reasoning effort at its maximum on *every* turn, including the mechanical ones where the model is only deciding which file to read next.

There is a second cost nobody counts. An assistant turn does not disappear — it stays in the message history and is re-sent as input on every subsequent turn. So a wasted output token is billed once at the output rate and again on each remaining turn:

```
true_cost = output_price + (turns_remaining × input_rate_at_position)
```

200 tokens of ceremony at turn 1 of a 50-turn Sonnet session costs $0.0030 as output plus $0.0029 as cached input — **about 2× the sticker price**, and roughly 11× if the cache is cold.

## The cache-key law

This is the constraint the whole design turns on.

| In the cache key — **freeze per conversation** | Outside it — **free every turn** |
|---|---|
| `model` | `output_config.effort` / `reasoning.effort` |
| `system` (steering text lives here) | `thinking.budget_tokens` |
| `tools` definitions | `text.verbosity` |
| message history | `max_tokens` / `max_output_tokens` |

Change something on the left mid-conversation and the provider's cached prefix is invalidated. On a 100k context that is about **+$0.27**, against the ~$0.005 of output a terser instruction might have saved — roughly **60× worse than doing nothing**.

The obvious idea — "be terse on mechanical turns, verbose when a human is reading" — is therefore one of the most expensive things you could implement. It looks like a free win and it is a silent, compounding loss.

So the contract expresses the partition in its **types**:

- `PromptDecision` is computed **once per conversation**, memoised against the conversation key, and replayed byte-for-byte afterwards. It is the only place prompt text can come from.
- `ParamDecision` is computed **per turn** and has no field capable of carrying prompt text, a model id, or a tool definition.

A third-party lever cannot bust a prefix cache through `ParamDecision`, however it is written. The expensive mistake is unrepresentable, not merely discouraged.

## Levers

Levers are **not portable**, and pretending otherwise has a cost: a previous design that generalised across formats left GitHub Copilot with zero output savings, because the one lever that generalised was also the weakest. So each lever declares the wire formats and models it applies to, and the registry asks *which levers apply here* rather than assuming a common denominator.

| Lever | Scope | Formats | Requires client to have sent |
|---|---|---|---|
| `verbosity` | conversation | all three | — |
| `effort` | turn | all three | `effort` |
| `thinking_budget` | turn | Anthropic | `thinking.budget_tokens` |
| `text_verbosity` | turn | OpenAI chat + Responses, gpt-5 | `text.verbosity` |
| `max_tokens` | turn | all three | `max_tokens` |

Two safety rules are enforced by the executor, never by the lever:

- **Never-inject.** A parameter the client did not send is never created. Models that do not accept `output_config.effort` return 400 when it appears, so injection turns a savings feature into an outage.
- **Lowered-only.** Every target may move a value down and never up. A lever asking for `effort="max"` is ignored, not obeyed.

A lever that raises is disabled for the process and the request proceeds without it.

## The adaptive ceiling

`max_tokens` is the one genuinely new lever, and the argument for it is that the data to drive it is already being collected.

Clients send a blanket ceiling — often 32,000 — unrelated to what any particular turn needs. Meanwhile the stratified accumulator tracks `n`, `sum` and `sumsq` per stratum: mean *and* variance. That is exactly the distribution needed to set `cap = mean + kσ`. A mechanical continuation whose stratum averages 180 output tokens with σ=90 does not need a 32,000-token allowance.

It closes a control loop that has been open. The feedback signal is `stop_reason == "max_tokens"` — **a field the provider returns**. That matters because the core's `verbosity_controller.py` is a complete, tested AIMD state machine that nothing ever calls, precisely because its signals (user interrupted, user replied too fast to have read the answer) need transcript inference. The ceiling lever has no such problem: truncation means widen immediately and cool down; sustained slack means tighten one step.

## Shadow mode

Shadow is a **first-class mode**, not a rollout step. A lever in shadow computes its decision, records what it would have done, and applies nothing.

The number it produces is the **bind rate**: how often the intended value would actually have constrained the real response. For the ceiling lever that is the would-be truncation rate, and it is the go/no-go number.

`max_tokens` therefore ships in shadow by default. It is the only lever here that can truncate a real answer, so it stays inert until a bind rate from real traffic says otherwise.

```
headroom output shadow ~/.headroom/output_shadow.json
```

## Install

```bash
pip install -e plugins/headroom-output
```

This registers the `headroom output` command group through `headroom.cli_extension` — a seam that works today with no core change.

```bash
headroom output levers                          # what is registered, and where it applies
headroom output plan captured_request.json      # dry-run; sends nothing, writes nothing
headroom output shadow output_shadow.json       # bind rates
```

`headroom output plan` is the artifact to hand a reviewer before enabling a lever: it prints the exact decision for a captured request without touching a provider.

## Using it directly

```python
from headroom_output import build_default_shaper

shaper = build_default_shaper(verbosity_level=2, holdout_fraction=0.05)

outcome = shaper.shape(body, conversation_key=session_id, harness="claude")
# ... send the request ...
shaper.observe(
    conversation_key=session_id,
    output_tokens=usage.output_tokens,
    stop_reason=response.stop_reason,
)
```

`shape()` must run **after** every other body mutation. The turn classifier reads the final message list, and a compression pass that rewrites tool results afterwards would leave the shaper having decided on a request that no longer exists.

`shape()` never raises. Any failure degrades to "this request was not shaped."

## Status, and the seam this still needs

The plugin is complete and testable standalone, but the host must currently call `shape()` and `observe()` explicitly. The seam it actually wants does not exist yet: `PipelineEvent` carries `messages`, `tools` and `headers`, and **every field this plugin writes lives on the request body**, which the event does not expose.

Landing it properly needs, in core:

1. **`thinking_tokens` and `turn_index` on the outcome.** Today `output_tokens` is a single number. Effort routing cuts thinking; steering cuts visible text. Pooled into one counter, neither lever can be attributed — including the ones already shipped. This is the prerequisite for everything else.
2. **A `PRE_SEND_PARAMS` stage carrying the body**, emitted after all message mutation.
3. **An `OUTCOME_OBSERVED` stage** carrying output tokens, stop reason and cache counts.
4. **An opt-in gate on `pipeline_extension`.** `discover_pipeline_extensions()` currently invokes every installed package that declares the group. That was tolerable when the contract could only touch `messages`; a contract that can rewrite `max_tokens` and `effort` on live traffic must be opt-in the way `proxy/extensions.py` already is.

`shape()` and `observe()` are deliberately shaped like that future contract, so wiring becomes renaming two call sites rather than a rewrite.

## Extending it

The plugin defines its own seam. Add a lever without forking:

```toml
[project.entry-points."headroom.output_lever"]
house_style = "my_pkg.levers:HouseStyleLever"
```

A lever implements `PromptLever` or `ParamLever` and declares a `LeverDescriptor`. The registry refuses to register a `mutates_cache_key=True` lever as a per-turn lever — that raise is the cache-key law being enforced at registration rather than discovered on a bill.

## License

Apache-2.0.
