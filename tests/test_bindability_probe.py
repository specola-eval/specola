"""Sim-free tests for the static-slot bindability probe."""

from __future__ import annotations

import json
from typing import FrozenSet, Optional

import numpy as np

from specola.adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from specola.adapters.fake import FakeAdapter
from specola.conformance import (
    BindabilityArtifact,
    bindability_evidence,
    probe_static_slot_bindability,
)
from specola.entities import EntityState, TrackState


class _OneEntityAdapter(SimulatorAdapter):
    """Single ``widget`` entity with two keys in two metadata groups."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    @property
    def dt(self) -> float:
        return 0.05

    @property
    def spatial_dims(self) -> int:
        return 3

    def property_metadata(self, entity_id: int):
        return {
            "visible": PropertyMeta(group="pose"),
            "vanishing": PropertyMeta(group="hidden"),
        }

    def _reset_impl(self) -> MonitorObservation:
        self._step = 0
        return self._observation(hide=False)

    def _get_observation_impl(self) -> MonitorObservation:
        self._step += 1
        return self._observation(hide=self._hides_key())

    def _hides_key(self) -> bool:
        return False

    def _observation(self, hide: bool) -> MonitorObservation:
        properties = {"visible": 1.0}
        if not hide:
            properties["vanishing"] = 2.0
        return MonitorObservation(
            timestep=self._step,
            time=self._step * self.dt,
            perceived_entities={
                7: EntityState(
                    entity_id=7,
                    entity_class="widget",
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    track_age=self._step,
                    track_state=TrackState.CONFIRMED,
                    properties=properties,
                )
            },
        )


class _ClosedHidingAdapter(_OneEntityAdapter):
    """Closed class vocabulary that drops one key once a scope is installed."""

    @property
    def supported_classes(self) -> Optional[FrozenSet[str]]:
        return frozenset({"widget"})

    def _hides_key(self) -> bool:
        return self._extraction_scope is not None


class _OpenAdapter(_OneEntityAdapter):
    """Open class vocabulary: no static class exists to probe."""

    @property
    def supported_classes(self) -> Optional[FrozenSet[str]]:
        return None


def test_mock_adapter_binds_every_static_class_group():
    artifact = probe_static_slot_bindability(FakeAdapter)

    assert artifact.outcome == "pass"
    assert artifact.class_mode == "closed"
    assert set(artifact.static_classes) == {
        "robot", "object", "region", "gripper",
    }
    assert artifact.runtime_populated_classes == ()
    assert artifact.cases
    assert len(artifact.results) == len(artifact.cases)
    assert all(item.status == "pass" for item in artifact.results)
    assert {case.entity_class for case in artifact.cases} == set(
        artifact.static_classes
    )


def test_scoped_out_key_fails_the_runtime_leg_naming_the_key():
    artifact = probe_static_slot_bindability(_ClosedHidingAdapter)

    assert artifact.outcome == "fail"
    assert {case.component for case in artifact.cases} == {
        "visible", "vanishing",
    }
    failed = [item for item in artifact.results if item.status == "fail"]
    assert [item.case.component for item in failed] == ["vanishing"]
    assert "vanishing" in failed[0].reason
    assert "widget" in failed[0].reason
    assert "7" in failed[0].reason
    passed = [item for item in artifact.results if item.status == "pass"]
    assert [item.case.component for item in passed] == ["visible"]


def test_open_class_vocabulary_is_not_applicable():
    artifact = probe_static_slot_bindability(_OpenAdapter)

    assert artifact.outcome == "not_applicable"
    assert artifact.class_mode == "open"
    assert artifact.static_classes == ()
    assert artifact.cases == ()
    assert artifact.results == ()


def test_setup_failure_keeps_the_exception_text():
    def factory():
        raise RuntimeError("no display available")

    artifact = probe_static_slot_bindability(factory)

    assert artifact.outcome == "setup_unavailable"
    assert "no display available" in artifact.detail
    assert artifact.cases == ()


def test_artifact_round_trips_through_json():
    artifact = probe_static_slot_bindability(FakeAdapter)

    text = json.dumps(artifact.to_json_dict(), sort_keys=True)
    restored = json.loads(text)

    assert restored["outcome"] == "pass"
    assert restored["adapter_type"].endswith("_MockAdapter")
    assert len(restored["cases"]) == len(artifact.cases)
    assert len(restored["results"]) == len(artifact.results)
    assert restored["results"][0]["case"] == restored["cases"][0]
    assert type(artifact) is BindabilityArtifact


def test_evidence_digest_is_the_digest_a_row_records_as_provenance():
    """Row drivers publish the digest; the ledger publishes the reference."""
    from specola.conformance.ledger import _bindability_reference

    artifact = probe_static_slot_bindability(FakeAdapter)
    evidence = bindability_evidence(artifact)

    assert evidence["artifact"] == artifact.to_json_dict()
    assert evidence["sha256"] == _bindability_reference(artifact).content.sha256


def test_compile_leg_break_is_undecided_rather_than_a_failed_slot(monkeypatch):
    """Only a refusal decides a slot; a broken compile leg decides nothing."""
    from specola.conformance import bindability as probe_module

    def explode(*unused, **ignored):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(probe_module, "compile_specs", explode)
    artifact = probe_static_slot_bindability(_ClosedHidingAdapter)

    assert artifact.outcome == "setup_unavailable"
    assert {item.status for item in artifact.results} == {"unexecuted"}
    assert "RuntimeError: compiler unavailable" in artifact.detail
    assert all(
        "compiler unavailable" in item.reason for item in artifact.results
    )


def test_both_probe_legs_close_the_adapter_they_built():
    """Two adapters are built, so two are released."""

    class _CountingAdapter(_ClosedHidingAdapter):
        closed = 0

        def close(self) -> None:
            type(self).closed += 1

    artifact = probe_static_slot_bindability(_CountingAdapter)

    assert artifact.outcome == "fail"  # the probe still ran both legs
    assert _CountingAdapter.closed == 2


def test_a_factory_that_rebuilds_its_scene_leaves_every_case_undecided():
    """A verdict over two different scenes is about the factory, not the slot."""

    class _NewSceneEachCall(_ClosedHidingAdapter):
        instances = 0

        def __init__(self) -> None:
            super().__init__()
            type(self).instances += 1
            self._suffix = type(self).instances

        def property_metadata(self, entity_id: int):
            return {"visible_{}".format(self._suffix): PropertyMeta(group="pose")}

        def _observation(self, hide: bool) -> MonitorObservation:
            observation = super()._observation(hide)
            entity = observation.perceived_entities[7]
            entity.properties = {"visible_{}".format(self._suffix): 1.0}
            return observation

    artifact = probe_static_slot_bindability(_NewSceneEachCall)

    assert artifact.outcome == "setup_unavailable"
    assert {item.status for item in artifact.results} == {"unexecuted"}
    assert artifact.detail.startswith("UnboundStaticSlotError:")
    assert all(
        item.reason.startswith("runtime leg unavailable: ")
        for item in artifact.results
    )
