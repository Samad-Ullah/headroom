"""The pipeline seam that lets an extension shape what a model writes.

Two stages and one field carry the whole capability:

* ``PRE_SEND_PARAMS`` — the last point at which the outbound request can be
  changed, emitted after every message mutation.
* ``OUTCOME_OBSERVED`` — what actually happened, so an adaptive extension can
  close a control loop.
* ``PipelineEvent.body`` — because the fields that control what a model
  *writes* live on the request body and on no other field of the event.

The discovery gate is tested here too. It matters more than it looks: this
contract can now rewrite ``max_tokens`` and ``effort`` on live traffic, so an
unaudited package installed as a transitive dependency must not be able to
switch itself on.
"""

from __future__ import annotations

import dataclasses

import pytest

from headroom.pipeline import (
    CANONICAL_PIPELINE_STAGES,
    OutcomeSnapshot,
    PipelineEvent,
    PipelineExtensionManager,
    PipelineStage,
    discover_pipeline_extensions,
)


class _Recorder:
    """Minimal extension: records every event and can mutate the body."""

    def __init__(self, mutate: dict | None = None) -> None:
        self.seen: list[PipelineEvent] = []
        self._mutate = mutate or {}

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        self.seen.append(event)
        if event.stage is PipelineStage.PRE_SEND_PARAMS and event.body is not None:
            event.body.update(self._mutate)

    def stages(self) -> list[PipelineStage]:
        return [e.stage for e in self.seen]


class TestStages:
    def test_new_stages_are_canonical(self):
        assert PipelineStage.PRE_SEND_PARAMS in CANONICAL_PIPELINE_STAGES
        assert PipelineStage.OUTCOME_OBSERVED in CANONICAL_PIPELINE_STAGES

    def test_params_stage_runs_after_pre_send(self):
        """Ordering is the contract: a params extension must see the request as
        it will actually go out, after every message mutation."""
        order = list(CANONICAL_PIPELINE_STAGES)
        assert order.index(PipelineStage.PRE_SEND_PARAMS) > order.index(PipelineStage.PRE_SEND)

    def test_outcome_stage_runs_last(self):
        order = list(CANONICAL_PIPELINE_STAGES)
        assert order.index(PipelineStage.OUTCOME_OBSERVED) == len(order) - 1


class TestBodyMutation:
    def test_extension_can_lower_max_tokens_through_the_body(self):
        """The capability that did not exist before: reaching an output-side
        control. None of messages/tools/headers can express this."""
        rec = _Recorder(mutate={"max_tokens": 500})
        mgr = PipelineExtensionManager(extensions=[rec], discover=False)
        body = {"model": "claude-sonnet-4", "max_tokens": 32_000, "messages": []}

        mgr.emit(PipelineStage.PRE_SEND_PARAMS, operation="proxy.request", body=body)

        assert body["max_tokens"] == 500, "body must be mutated in place"

    def test_body_is_absent_on_stages_that_do_not_pass_it(self):
        rec = _Recorder(mutate={"max_tokens": 1})
        mgr = PipelineExtensionManager(extensions=[rec], discover=False)
        mgr.emit(PipelineStage.INPUT_RECEIVED, operation="proxy.request", messages=[])
        assert rec.seen[0].body is None

    def test_a_raising_extension_does_not_break_the_emit(self):
        class _Boom:
            def on_pipeline_event(self, event):
                raise RuntimeError("bad extension")

        mgr = PipelineExtensionManager(extensions=[_Boom(), _Recorder()], discover=False)
        body = {"max_tokens": 100}
        event = mgr.emit(PipelineStage.PRE_SEND_PARAMS, operation="proxy.request", body=body)
        assert event.body is body


class TestOutcomeSnapshot:
    def test_truncated_reads_both_provider_spellings(self):
        assert OutcomeSnapshot(stop_reason="max_tokens").truncated
        assert OutcomeSnapshot(stop_reason="length").truncated
        assert not OutcomeSnapshot(stop_reason="end_turn").truncated
        assert not OutcomeSnapshot().truncated

    def test_visible_split(self):
        assert OutcomeSnapshot(output_tokens=900, thinking_tokens=700).visible_output_tokens == 200

    def test_unknown_thinking_yields_unknown_visible_not_the_total(self):
        """Defaulting to output_tokens would credit visible-text levers with
        reductions that reasoning-effort levers produced."""
        assert OutcomeSnapshot(output_tokens=900).visible_output_tokens is None

    def test_snapshot_is_frozen(self):
        """Extensions learn from the measurement; they must not rewrite it."""
        snap = OutcomeSnapshot(output_tokens=100)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.output_tokens = 5  # type: ignore[misc]

    def test_snapshot_reaches_the_extension(self):
        rec = _Recorder()
        mgr = PipelineExtensionManager(extensions=[rec], discover=False)
        snap = OutcomeSnapshot(output_tokens=250, stop_reason="max_tokens")
        mgr.emit(PipelineStage.OUTCOME_OBSERVED, operation="proxy.outcome", outcome=snap)
        assert rec.seen[0].outcome is snap
        assert rec.seen[0].outcome.truncated


class TestDiscoveryGate:
    """Installing a package must not silently start rewriting live requests."""

    def test_discovery_is_off_without_an_explicit_enable(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_PIPELINE_EXTENSIONS", raising=False)
        assert discover_pipeline_extensions() == []

    def test_empty_env_does_not_enable(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_PIPELINE_EXTENSIONS", "   ,  ,")
        assert discover_pipeline_extensions() == []

    def test_explicit_argument_beats_the_env(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_PIPELINE_EXTENSIONS", "something")
        # No entry point by this name exists, so the result is empty either way;
        # what is asserted is that resolution does not raise and stays opt-in.
        assert discover_pipeline_extensions(["definitely-not-installed"]) == []

    def test_directly_passed_extensions_are_unaffected_by_the_gate(self, monkeypatch):
        """Constructing an extension and handing it over is already explicit
        consent — the gate covers entry-point discovery only."""
        monkeypatch.delenv("HEADROOM_PIPELINE_EXTENSIONS", raising=False)
        rec = _Recorder()
        mgr = PipelineExtensionManager(extensions=[rec], discover=True)
        assert mgr.enabled
        mgr.emit(PipelineStage.PRE_SEND_PARAMS, operation="proxy.request", body={})
        assert rec.stages() == [PipelineStage.PRE_SEND_PARAMS]
