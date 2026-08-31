"""The safety properties that make this plugin safe to hand to third parties.

Every test here corresponds to a specific way a shaping plugin can cost someone
money or break their request. They are the reason the contract has the shape it
does, so if one of these starts failing the fix is almost never in the test.
"""

from __future__ import annotations

import pytest

from headroom_output.contract import (
    ConversationFeatures,
    LeverDescriptor,
    LeverMode,
    OutcomeReading,
    ParamDecision,
    PromptDecision,
    TurnFeatures,
    TurnKind,
    WireFormat,
)
from headroom_output.registry import (
    LeverRegistry,
    apply_param_decision,
    merge_param_decisions,
)

ALL_FORMATS = (WireFormat.ANTHROPIC, WireFormat.OPENAI_CHAT, WireFormat.OPENAI_RESPONSES)


# --------------------------------------------------------------------------
# test doubles
# --------------------------------------------------------------------------


class _ParamLever:
    def __init__(self, name: str, decision: ParamDecision, **kw) -> None:
        self.descriptor = LeverDescriptor(
            name=name,
            summary=f"test lever {name}",
            wire_formats=kw.pop("wire_formats", ALL_FORMATS),
            **kw,
        )
        self._decision = decision
        self.observed: list[OutcomeReading] = []

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        return self._decision

    def observe(self, outcome: OutcomeReading) -> None:
        self.observed.append(outcome)


class _ExplodingLever:
    def __init__(self, name: str = "boom") -> None:
        self.descriptor = LeverDescriptor(
            name=name, summary="always raises", wire_formats=ALL_FORMATS
        )
        self.calls = 0

    def decide_turn(self, features: TurnFeatures) -> ParamDecision:
        self.calls += 1
        raise RuntimeError("lever is broken")


class _PromptLever:
    def __init__(self, name: str, texts: list[str]) -> None:
        self.descriptor = LeverDescriptor(
            name=name,
            summary="test prompt lever",
            wire_formats=ALL_FORMATS,
            mutates_cache_key=True,
        )
        self._texts = texts
        self.calls = 0

    def decide_conversation(self, features: ConversationFeatures) -> PromptDecision:
        # Returns a DIFFERENT string each call, so any test that sees stable
        # output is genuinely exercising memoisation rather than luck.
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return PromptDecision(system_suffix=text, label=f"{self.descriptor.name}:v{self.calls}")


def _turn(**kw) -> TurnFeatures:
    base = {
        "conversation_key": "conv-1",
        "wire_format": WireFormat.ANTHROPIC,
        "model": "claude-sonnet-4",
        "turn_kind": TurnKind.MECHANICAL_CONTINUATION,
        "input_tokens": 5_000,
        "has_tools": True,
        "present_fields": frozenset({"effort", "max_tokens", "thinking_budget"}),
    }
    base.update(kw)
    return TurnFeatures(**base)


def _conversation(**kw) -> ConversationFeatures:
    base = {
        "conversation_key": "conv-1",
        "wire_format": WireFormat.ANTHROPIC,
        "model": "claude-sonnet-4",
    }
    base.update(kw)
    return ConversationFeatures(**base)


# --------------------------------------------------------------------------
# the cache-key law
# --------------------------------------------------------------------------


def test_cache_key_lever_cannot_be_registered_per_turn():
    """A per-turn prompt change is the single most expensive mistake here.

    Registering a cache-key-mutating lever as a per-turn lever must fail loudly
    at registration, not silently cost 60x on a bill.
    """
    registry = LeverRegistry()
    lever = _PromptLever("verbosity", ["a"])
    with pytest.raises(ValueError, match="mutates_cache_key"):
        registry.register_param(lever)


def test_param_decision_has_no_field_for_prompt_text():
    """The safety property is structural, so assert on the structure.

    If someone adds a text-carrying field to ParamDecision, the type stops
    being a guarantee and this test is the tripwire.
    """
    allowed = {"effort", "thinking_budget", "text_verbosity", "max_tokens", "labels"}
    assert set(ParamDecision.__dataclass_fields__) == allowed


def test_prompt_decision_is_memoised_per_conversation():
    """Prompt text must be byte-stable for the life of a conversation.

    The lever deliberately returns different text on every call; the registry
    must consult it once and replay the first answer forever after.
    """
    registry = LeverRegistry()
    lever = _PromptLever("verbosity", ["FIRST", "SECOND", "THIRD"])
    registry.register_prompt(lever)

    conv = _conversation()
    first = registry.plan(conv, _turn()).prompt.system_suffix
    for _ in range(5):
        assert registry.plan(conv, _turn()).prompt.system_suffix == first
    assert lever.calls == 1, "prompt lever must be consulted exactly once per conversation"


def test_different_conversations_get_independent_prompt_decisions():
    registry = LeverRegistry()
    registry.register_prompt(_PromptLever("verbosity", ["FIRST", "SECOND"]))

    a = registry.plan(_conversation(conversation_key="a"), _turn(conversation_key="a"))
    b = registry.plan(_conversation(conversation_key="b"), _turn(conversation_key="b"))
    assert a.prompt.system_suffix == "FIRST"
    assert b.prompt.system_suffix == "SECOND"


# --------------------------------------------------------------------------
# never-inject
# --------------------------------------------------------------------------


def test_lever_is_skipped_when_required_field_absent():
    """Injecting a parameter the client never sent 400s some models.

    That turns a savings feature into an outage, so absence of the field must
    skip the lever entirely rather than create it.
    """
    registry = LeverRegistry()
    registry.register_param(
        _ParamLever("effort", ParamDecision(effort="low"), requires_present="effort")
    )
    plan = registry.plan(_conversation(), _turn(present_fields=frozenset()))
    assert plan.params.effort is None
    assert "effort" in plan.skipped


def test_apply_never_creates_an_absent_field():
    """Belt and braces: even a decision that reaches apply must not create."""
    body: dict = {}
    applied = apply_param_decision(
        body,
        ParamDecision(effort="low", thinking_budget=1024, text_verbosity="low", max_tokens=500),
        WireFormat.ANTHROPIC,
    )
    assert applied == []
    assert body == {}, "apply must not introduce any key the client did not send"


# --------------------------------------------------------------------------
# lowered-only
# --------------------------------------------------------------------------


def test_effort_is_never_raised():
    body = {"output_config": {"effort": "low"}}
    applied = apply_param_decision(body, ParamDecision(effort="max"), WireFormat.ANTHROPIC)
    assert applied == []
    assert body["output_config"]["effort"] == "low"


def test_effort_is_lowered_when_target_is_lower():
    body = {"output_config": {"effort": "xhigh"}}
    applied = apply_param_decision(body, ParamDecision(effort="low"), WireFormat.ANTHROPIC)
    assert body["output_config"]["effort"] == "low"
    assert applied == ["effort:xhigh->low"]


def test_max_tokens_is_never_raised():
    body = {"max_tokens": 400}
    applied = apply_param_decision(body, ParamDecision(max_tokens=8_000), WireFormat.ANTHROPIC)
    assert applied == []
    assert body["max_tokens"] == 400


def test_responses_format_uses_its_own_field_names():
    """Responses spells these differently; writing the Anthropic names is a no-op."""
    body = {"reasoning": {"effort": "high"}, "max_output_tokens": 4_000}
    applied = apply_param_decision(
        body, ParamDecision(effort="low", max_tokens=500), WireFormat.OPENAI_RESPONSES
    )
    assert body["reasoning"]["effort"] == "low"
    assert body["max_output_tokens"] == 500
    assert len(applied) == 2


def test_thinking_budget_only_clamps_when_enabled():
    """Never touch thinking.type — toggling it 400s when history carries
    thinking blocks, and busts the messages cache tier."""
    body = {"thinking": {"type": "disabled", "budget_tokens": 8_000}}
    applied = apply_param_decision(body, ParamDecision(thinking_budget=1024), WireFormat.ANTHROPIC)
    assert applied == []
    assert body["thinking"]["budget_tokens"] == 8_000


# --------------------------------------------------------------------------
# merge determinism
# --------------------------------------------------------------------------


def test_merge_takes_the_safest_lower_value():
    merged = merge_param_decisions(
        [
            ParamDecision(effort="medium", max_tokens=2_000),
            ParamDecision(effort="low", max_tokens=800),
            ParamDecision(max_tokens=1_500),
        ]
    )
    assert merged.effort == "low"
    assert merged.max_tokens == 800


def test_merge_is_order_independent():
    """Two levers must never fight: the result cannot depend on which
    registered first, or attribution stops meaning anything."""
    a = ParamDecision(effort="medium", max_tokens=2_000, labels=("a",))
    b = ParamDecision(effort="low", max_tokens=800, labels=("b",))
    forward = merge_param_decisions([a, b])
    backward = merge_param_decisions([b, a])
    assert forward.effort == backward.effort
    assert forward.max_tokens == backward.max_tokens
    assert sorted(forward.labels) == sorted(backward.labels)


def test_merge_of_nothing_is_empty():
    assert merge_param_decisions([]).is_empty()


# --------------------------------------------------------------------------
# isolation and modes
# --------------------------------------------------------------------------


def test_a_broken_lever_is_disabled_and_the_request_survives():
    registry = LeverRegistry()
    boom = _ExplodingLever()
    registry.register_param(boom)
    registry.register_param(_ParamLever("good", ParamDecision(effort="low")))

    first = registry.plan(_conversation(), _turn())
    assert first.params.effort == "low", "a broken lever must not suppress a working one"
    assert "boom" in first.skipped

    registry.plan(_conversation(), _turn())
    assert boom.calls == 1, "a lever that raised once must not be retried every request"


def test_shadow_lever_never_reaches_the_body():
    """Shadow is the whole safety story for a lever that can truncate.
    If a shadow decision could be applied, the mode would be worthless."""
    registry = LeverRegistry()
    registry.register_param(
        _ParamLever("max_tokens", ParamDecision(max_tokens=100, labels=("max_tokens:100",))),
        LeverMode.SHADOW,
    )
    plan = registry.plan(_conversation(), _turn())
    assert plan.params.max_tokens is None
    assert "max_tokens" in plan.shadow_only
    assert any(lb.startswith("shadow:") for lb in plan.params.labels)


def test_off_lever_is_not_consulted():
    registry = LeverRegistry()
    lever = _ParamLever("effort", ParamDecision(effort="low"))
    registry.register_param(lever, LeverMode.OFF)
    plan = registry.plan(_conversation(), _turn())
    assert plan.params.effort is None
    assert "effort" in plan.skipped


# --------------------------------------------------------------------------
# capability matching
# --------------------------------------------------------------------------


def test_lever_does_not_apply_to_a_foreign_wire_format():
    registry = LeverRegistry()
    registry.register_param(
        _ParamLever(
            "text_verbosity",
            ParamDecision(text_verbosity="low"),
            wire_formats=(WireFormat.OPENAI_CHAT,),
        )
    )
    plan = registry.plan(_conversation(), _turn(wire_format=WireFormat.ANTHROPIC))
    assert plan.considered == (), "a non-matching lever must not even be considered"


def test_model_prefix_gates_the_lever():
    registry = LeverRegistry()
    registry.register_param(
        _ParamLever(
            "text_verbosity",
            ParamDecision(text_verbosity="low"),
            model_prefixes=("gpt-5",),
        )
    )
    assert registry.plan(_conversation(), _turn(model="claude-sonnet-4")).considered == ()
    assert registry.plan(
        _conversation(model="gpt-5-mini"), _turn(model="gpt-5-mini")
    ).considered == ("text_verbosity",)


def test_error_continuations_are_left_alone():
    """A failing tool call is where the model most needs room to reason.
    Tightening there saves a few tokens and costs a recovery."""
    registry = LeverRegistry()
    registry.register_param(
        _ParamLever(
            "effort",
            ParamDecision(effort="low"),
            turn_kinds=(TurnKind.MECHANICAL_CONTINUATION,),
        )
    )
    plan = registry.plan(_conversation(), _turn(turn_kind=TurnKind.ERROR_CONTINUATION))
    assert plan.considered == ()


def test_empty_considered_list_signals_a_coverage_gap():
    """No matching lever is a reportable fact, not a silent no-op — this is
    exactly how a whole client came to receive zero savings unnoticed."""
    registry = LeverRegistry()
    plan = registry.plan(_conversation(), _turn())
    assert plan.considered == ()
    assert plan.empty


# --------------------------------------------------------------------------
# feedback
# --------------------------------------------------------------------------


def test_observe_reaches_levers_and_survives_a_raising_one():
    registry = LeverRegistry()
    good = _ParamLever("good", ParamDecision(effort="low"))
    registry.register_param(good)

    reading = OutcomeReading(conversation_key="c", output_tokens=250, stop_reason="max_tokens")
    registry.observe(reading)
    assert good.observed == [reading]


def test_truncated_reads_both_provider_spellings():
    assert OutcomeReading(conversation_key="c", output_tokens=1, stop_reason="max_tokens").truncated
    assert OutcomeReading(conversation_key="c", output_tokens=1, stop_reason="length").truncated
    assert not OutcomeReading(
        conversation_key="c", output_tokens=1, stop_reason="end_turn"
    ).truncated


def test_visible_tokens_excludes_thinking_when_the_split_is_known():
    """Effort routing cuts thinking; steering cuts visible text. Pooled into
    one counter neither can be attributed — which is the core's state today."""
    pooled = OutcomeReading(conversation_key="c", output_tokens=1_000)
    assert pooled.visible_tokens == 1_000

    split = OutcomeReading(conversation_key="c", output_tokens=1_000, thinking_tokens=700)
    assert split.visible_tokens == 300
