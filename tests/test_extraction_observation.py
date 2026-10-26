"""Structured transport and lifecycle checks, not producer fidelity evidence."""

from dataclasses import replace
import gzip
import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.capabilities import reset_adapter
from specola.discovery_capture import discovery_payload
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_restore import DiscoveryRestoreError, discovery_model_from_document
from specola.discovery_serialize import discovery_document
from specola.entities import EntityState, TrackState
from specola.extraction_observation import (
    ExtractionObservation, ExtractionObservationError, StructuredMetadata,
    StructuredSample, extraction_document, extraction_from_document,
)
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration


def _metadata():
    return StructuredMetadata("fixture.state", "Declared fixture state", "float64[2]",
                              "unitless", "not_applicable", "identity", ("fixture.state",), "a" * 64)


class _StructuredAdapter(SimulatorAdapter):
    """A shared-boundary fixture without a claim about engine semantics."""

    def __init__(self):
        super().__init__()
        self.index = 0
        self.reads = 0
        self.raw = np.array([1.0, np.nan])
        self.transform = lambda value: value

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    def _reset_impl(self):
        self.index = 0
        return self._observe()

    def _get_observation_impl(self):
        self.index += 1
        return self._observe()

    def _observe(self):
        self.reads += 1
        entity = EntityState(entity_id=7, entity_class="object", position=np.zeros(3),
                             track_state=TrackState.CONFIRMED, track_age=self.index,
                             properties={"state_0": float(self.raw[0]), "other": 2.0})
        envelope = ExtractionObservation(
            timestep=self.index, time=self.index * self.dt,
            episode_epoch=self.extraction_episode_epoch,
            configuration_epoch=self.query_configuration_epoch,
            producer_surface_sha256="b" * 64, manifest_sha256="c" * 64,
            queries=self.query_configuration, metadata=(_metadata(),),
            samples=(StructuredSample("fixture.state", "object", (7,), NativeValue.capture(self.raw),
                                      "present", entity_id=7, scalar_keys=("state_0",)),),
        )
        return MonitorObservation(self.index, self.index * self.dt, {7: entity},
                                  extraction=self.transform(envelope))


def _envelope():
    envelope = _StructuredAdapter().reset().extraction
    assert envelope is not None
    return envelope


def test_typed_round_trip_retains_nan_and_detaches_native_buffer():
    adapter = _StructuredAdapter()
    first = adapter.reset().extraction
    assert first is not None
    document = extraction_document(first)
    encoded = json.dumps(document, allow_nan=False)
    restored = extraction_from_document(json.loads(encoded))
    assert restored == first
    adapter.raw[:] = 8.0
    adapter.get_observation()
    assert first.samples[0].value.restore().tolist()[0] == 1.0
    assert np.isnan(restored.samples[0].value.restore()[1])
    assert extraction_document(first) == document


@pytest.mark.parametrize("mutation", (
    lambda value: replace(value, timestep=True),
    lambda value: replace(value, time=float("nan")),
    lambda value: replace(value, producer_surface_sha256="not a digest"),
    lambda value: replace(value, metadata=()),
    lambda value: replace(value, metadata=value.metadata * 2),
    lambda value: replace(value, samples=value.samples * 2),
    lambda value: replace(value, samples=value.samples + (replace(value.samples[0], owner_index=(8,)),)),
    lambda value: replace(value, verification_only=1),
))
def test_invalid_envelope_identities_and_duplicate_output_addresses_fail(mutation):
    with pytest.raises(ExtractionObservationError):
        mutation(_envelope())


@pytest.mark.parametrize("mutation", (
    lambda value: value.update(schema="unknown"),
    lambda value: value.update(extra="unknown"),
    lambda value: value["metadata"][0].update(extra="unknown"),
    lambda value: value["samples"][0].update(extra="unknown"),
    lambda value: value["samples"][0].update(owner_index="7"),
    lambda value: value["samples"][0].update(value=object()),
    lambda value: value["samples"][0].update(scalar_keys=("state_0",)),
    lambda value: value["samples"].append(value),
))
def test_malformed_wire_documents_fail_without_coercion(mutation):
    document = extraction_document(_envelope())
    mutation(document)
    with pytest.raises(ExtractionObservationError):
        extraction_from_document(document)


def test_scalar_scope_projects_links_without_losing_structured_values():
    adapter = _StructuredAdapter()
    full = adapter.reset().extraction
    adapter.set_extraction_scope({"object": frozenset({"other"})})
    scoped = adapter.get_observation()
    assert full is not None and scoped.extraction is not None
    assert set(scoped.perceived_entities[7].properties) == {"other"}
    assert scoped.extraction.samples[0].scalar_keys == ()
    assert scoped.extraction.samples[0].value == full.samples[0].value


@pytest.mark.parametrize("scoped", (False, True))
def test_projection_cannot_hide_a_broken_scalar_link(scoped):
    adapter = _StructuredAdapter()
    first = adapter.reset()
    if scoped:
        adapter.set_extraction_scope({"object": frozenset()})
    adapter.transform = lambda value: replace(value, samples=(replace(value.samples[0], scalar_keys=("missing",)),))
    with pytest.raises(ExtractionObservationError, match="absent property"):
        adapter.get_observation()
    assert adapter.last_observation is first


@pytest.mark.parametrize("mutation", (
    lambda value: replace(value, time=value.time + 1),
    lambda value: replace(value, samples=(replace(value.samples[0], entity_id=100),)),
    lambda value: replace(value, producer_surface_sha256="d" * 64),
    lambda value: replace(value, configuration_epoch=value.configuration_epoch + 1),
    lambda value: replace(value, episode_epoch=value.episode_epoch + 1),
    lambda value: None,
))
def test_rejected_observation_never_replaces_the_accepted_snapshot(mutation):
    adapter = _StructuredAdapter()
    first = adapter.reset()
    adapter.transform = mutation
    with pytest.raises(ExtractionObservationError):
        adapter.get_observation()
    assert adapter.last_observation is first
    assert adapter._structured_metadata_by_quantity == {"fixture.state": _metadata()}


def test_metadata_meaning_cannot_change_after_a_temporary_absence():
    adapter = _StructuredAdapter()
    adapter.reset()
    adapter.transform = lambda value: replace(value, metadata=(), samples=())
    absent = adapter.get_observation()
    adapter.transform = lambda value: replace(value, metadata=(replace(_metadata(), unit="changed"),))
    with pytest.raises(ExtractionObservationError, match="meaning changed"):
        adapter.get_observation()
    assert adapter.last_observation is absent


def _queries():
    return QueryConfiguration.from_document({"schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": "fixture.query", "arguments": {"cutoff": 2.0}}]})


def test_query_change_requires_reset_before_any_native_read():
    adapter = _StructuredAdapter()
    first = adapter.reset().extraction
    adapter.set_query_configuration(_queries())
    with pytest.raises(RuntimeError, match="Query configuration changed"):
        adapter.get_observation()
    assert adapter.reads == 1
    second = adapter.reset().extraction
    assert first is not None and second is not None
    assert second.episode_epoch == first.episode_epoch + 1
    assert second.configuration_epoch == first.configuration_epoch + 1
    assert second.queries == _queries()
    adapter.set_query_configuration(_queries())
    adapter.get_observation()


def test_monitor_installs_queries_before_reset_and_compilation(tmp_path):
    from specola.session import Monitor

    path = tmp_path / "specs.yaml"
    path.write_text('specs:\n- name: state\n  formula: "object_*[state_0] >= 0"\n')
    adapter = _StructuredAdapter()
    monitor = Monitor(str(path), results_dir=None)
    session = monitor.watch(adapter, queries=_queries())
    assert adapter.last_observation is not None and adapter.last_observation.extraction is not None
    assert adapter.last_observation.extraction.queries == _queries()
    session.step()
    session.end()


def test_discovery_capture_retains_the_given_reset_not_the_latest_read():
    adapter = _StructuredAdapter()
    reset = reset_adapter(adapter)
    adapter.raw[0] = 3.0
    adapter.get_observation()
    document = json.loads(gzip.decompress(discovery_payload(adapter, reset, simulator="fixture", task=None, effective_dt=adapter.dt)))
    restored = extraction_from_document(document["extraction"])
    assert restored == reset.initial_observation.extraction
    assert restored.samples[0].value.restore()[0] == 1.0
    assert adapter.reads == 2
    model = discovery_model_from_document(document)
    assert model.extraction == restored
    assert discovery_document(filter_discovery(model, DiscoveryQuery(include_raw=True)))["extraction"] == document["extraction"]


def test_discovery_restore_rejects_malformed_nested_envelope():
    adapter = _StructuredAdapter()
    reset = reset_adapter(adapter)
    document = json.loads(gzip.decompress(discovery_payload(adapter, reset, simulator="fixture", task=None, effective_dt=adapter.dt)))
    document["extraction"]["samples"][0]["value"] = {"kind": "arbitrary_object"}
    with pytest.raises(DiscoveryRestoreError):
        discovery_model_from_document(document)


def test_verification_only_envelope_cannot_enter_ordinary_discovery():
    adapter = _StructuredAdapter()
    observation = adapter.reset()
    assert observation.extraction is not None
    envelope = replace(observation.extraction, verification_only=True)
    with pytest.raises(ExtractionObservationError, match="non-verification"):
        build_discovery_model(adapter, observation.perceived_entities, simulator="fixture", task=None,
                              effective_dt=adapter.dt, extraction=envelope)


def test_restored_discovery_rejects_nested_candidate_identity():
    adapter = _StructuredAdapter()
    reset = reset_adapter(adapter)
    document = json.loads(gzip.decompress(discovery_payload(
        adapter, reset, simulator="fixture", task=None, effective_dt=adapter.dt)))
    document["extraction"]["verification_only"] = True
    with pytest.raises(DiscoveryRestoreError, match="candidate.*ordinary"):
        discovery_model_from_document(document)


def test_recording_round_trip_keeps_native_instant_independent_of_loop_step(tmp_path):
    from _dashboard_fixtures import MANIFEST, make_snapshots
    from specola.dashboard.recording import RecordingReader, RecordingWriter, snapshot_to_line

    snapshot = make_snapshots(steps=1)[0]
    envelope = replace(_envelope(), timestep=42)
    line = snapshot_to_line(snapshot, 0.0, set(), extraction=envelope)
    writer = RecordingWriter(tmp_path / "recording", MANIFEST)
    writer.append_step(line)
    writer.finalize(status="completed")
    restored = RecordingReader(tmp_path / "recording").read_step(snapshot.step)
    assert restored["step"] != restored["extraction"]["timestep"]
    assert extraction_from_document(restored["extraction"]) == envelope


@pytest.mark.parametrize("mutation", (
    lambda value: replace(value, verification_only=True),
    lambda value: replace(value, time=1.0),
))
def test_recording_rejects_invalid_envelope_before_metadata_commit(mutation):
    from _dashboard_fixtures import make_snapshots
    from specola.dashboard.recording import snapshot_to_line

    seen = set()
    with pytest.raises(ExtractionObservationError):
        snapshot_to_line(make_snapshots(steps=1)[0], 0.0, seen, extraction=mutation(_envelope()))
    assert seen == set()


def test_recorder_rejects_verification_input_before_capture_or_enqueue():
    from specola.dashboard.recorder import WebRecorderSink

    envelope = replace(_envelope(), verification_only=True)
    adapter = SimpleNamespace(last_observation=SimpleNamespace(extraction=envelope))
    sink = SimpleNamespace(active=True)
    with pytest.raises(ExtractionObservationError, match="verification-only"):
        WebRecorderSink.on_step(cast(WebRecorderSink, sink), adapter, object())
