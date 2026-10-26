"""The persisted scene must be the scene the episode monitored.

A caller can reset its adapter before ``Monitor.watch()`` (``specola
run`` did, in ``load_sim()``, until 2026-10-12), and ``watch()`` resets
again; the second is what ``MonitorEngine.start_episode()`` compiles against. For a simulator that
randomizes per reset those are different scenes — five consecutive
RoboCasa resets of one task produced five different object sets and five
different instructions.

So a sidecar written from the first reset would name objects the episode
never spawned, and an offline check compiled against it would resolve
keys the run never had. The adapter here makes that failure detectable:
its two resets expose different entity ids and different keys, so a
capture from the wrong one is a test failure rather than a subtle
mismatch nobody notices until a spec silently binds nothing.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from specola.adapter import SimulatorAdapter
from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.discovery_capture import discovery_payload
from specola.entities import EntityState, TrackState


class _ShiftingAdapter(SimulatorAdapter):
    """Exposes a different scene on every reset, like a randomized sim.

    RoboCasa is the real instance of this: it draws a fresh kitchen per
    reset, so the two resets a run performs describe different scenes.
    Encoding that here is what makes capturing the wrong reset a test
    failure instead of an invisible mismatch.
    """

    supported_classes = frozenset({"object"})

    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    @property
    def dt(self) -> float:
        return 0.1

    @property
    def spatial_dims(self) -> int:
        return 3

    def _reset_impl(self):
        return self._scene()

    def _get_observation_impl(self):
        return self._scene(advance=False)

    def _scene(self, advance=True):
        from specola.adapter import MonitorObservation

        if advance:
            self.reset_count += 1
        index = max(1, self.reset_count)
        entity = EntityState(
            entity_id=100 + index,
            entity_class="object",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            position=np.zeros(3),
            properties={"xpos_scene{}_2".format(index): 0.5 * index},
        )
        return MonitorObservation(
            timestep=0,
            time=0.0,
            perceived_entities={entity.entity_id: entity},
        )

    @property
    def capabilities(self):
        return AdapterCapabilities(
            spatial_dims=3,
            class_mode=VocabularyMode.CLOSED,
            supported_classes=frozenset({"object"}),
            runtime_populated_classes=frozenset({"contact"}),
        )


def _document(payload):
    return json.loads(gzip.decompress(payload))


def test_payload_describes_the_reset_it_was_handed():
    """Not a fresh look at the adapter, which would move the scene on."""
    from specola.capabilities import reset_adapter

    adapter = _ShiftingAdapter()
    reset_adapter(adapter)                       # reset #1, discarded
    second = reset_adapter(adapter)              # reset #2, the episode's

    document = _document(discovery_payload(
        adapter, second, simulator="shifting", task=None, effective_dt=0.1,
    ))

    ids = [entity["id"] for entity in document["entities"]]
    keys = [
        prop["key"]
        for entity in document["entities"]
        for prop in entity["properties"]
    ]
    assert ids == [102], ids
    assert keys == ["xpos_scene2_2"], keys
    # Building the payload must not have reset the adapter again: a
    # third reset would describe a scene nothing is monitoring.
    assert adapter.reset_count == 2


def test_watch_hands_the_hook_the_reset_it_compiles_against(tmp_path):
    """The end-to-end ordering claim, through the real Monitor.

    The caller resets before ``watch`` does; the hook must see the
    later one. Asserting on the entity id makes an off-by-one-reset
    visible, which reading the code cannot.
    """
    from specola.session import Monitor

    spec_file = tmp_path / "specs.yaml"
    spec_file.write_text(
        "specs:\n"
        "- name: scene_present\n"
        '  formula: "object_*[xpos_*_2] >= -1"\n'
    )

    adapter = _ShiftingAdapter()
    adapter.reset()                              # the caller's earlier reset
    seen = {}

    monitor = Monitor(str(spec_file), results_dir=str(tmp_path / "out"))
    monitor.watch(
        adapter,
        _reset_hook=lambda a, reset: seen.update(
            entities=dict(reset.initial_observation.perceived_entities),
            capabilities=reset.capabilities,
            metadata=reset.property_metadata,
        ),
    )

    assert list(seen["entities"]) == [102], list(seen["entities"])
    assert seen["capabilities"].class_mode is VocabularyMode.CLOSED
    assert "contact" in seen["capabilities"].runtime_populated_classes


def test_document_is_unfiltered_whatever_the_terminal_asked_for():
    """A filter is a question someone asked once.

    The recording has to answer questions nobody has asked yet, so it
    persists the raw scene rather than whatever the CLI was narrowed to.
    """
    from specola.capabilities import reset_adapter

    adapter = _ShiftingAdapter()
    document = _document(discovery_payload(
        adapter, reset_adapter(adapter), simulator="shifting",
        task=None, effective_dt=0.1,
    ))
    assert document["filters"]["include_raw"] is True
    assert document["filters"]["class"] == []
    assert document["filters"]["signal"] == []


def test_all_nine_capability_fields_survive_the_round_trip():
    """Three of nine is enough to browse and not enough to compile.

    Without ``class_mode`` and ``supported_classes`` a rebuilt check
    falls back to OPEN vocabulary and defers references a CLOSED adapter
    rejects — accepting specifications the live check refuses.
    """
    from dataclasses import fields

    from specola.capabilities import reset_adapter
    from specola.discovery_serialize import capabilities_document

    adapter = _ShiftingAdapter()
    document = _document(discovery_payload(
        adapter, reset_adapter(adapter), simulator="shifting",
        task=None, effective_dt=0.1,
    ))
    encoded = document["capabilities"]
    assert set(encoded) == {field.name for field in fields(AdapterCapabilities)}
    assert encoded["class_mode"] == "closed"
    assert encoded["supported_classes"] == ["object"]
    assert encoded["runtime_populated_classes"] == ["contact"]
    assert encoded["entity_id_mode"] == "open"

    # Deterministic: the same capabilities must encode to the same bytes
    # or the sidecar's digest means nothing.
    twice = capabilities_document(adapter.capabilities)
    assert json.dumps(twice, sort_keys=True) == json.dumps(
        capabilities_document(adapter.capabilities), sort_keys=True
    )


def test_sidecar_round_trips_through_the_recording(tmp_path):
    pytest.importorskip("flask_socketio")
    from specola.capabilities import reset_adapter
    from specola.dashboard.recording import (
        RecordingWriter,
        read_discovery_document,
    )

    adapter = _ShiftingAdapter()
    payload = discovery_payload(
        adapter, reset_adapter(adapter), simulator="shifting",
        task=None, effective_dt=0.1,
    )
    writer = RecordingWriter(tmp_path, {"run_id": "r", "status": None})
    entry = writer.write_discovery(payload)

    assert entry["encoding"] == "gzip+json"
    assert entry["compressed_bytes"] == len(payload)
    assert writer.manifest["discovery"] == entry
    assert json.loads(
        (tmp_path / "manifest.json").read_text()
    )["discovery"] == entry

    restored = read_discovery_document(tmp_path)
    assert restored["entities"][0]["id"] == 101


def test_a_recording_without_a_sidecar_still_reads(tmp_path):
    """Every recording made before this existed must keep replaying."""
    from specola.dashboard.recording import (
        read_discovery_bytes,
        read_discovery_document,
    )

    assert read_discovery_bytes(tmp_path) is None
    assert read_discovery_document(tmp_path) is None


def test_a_corrupt_sidecar_reads_as_an_absent_one(tmp_path):
    """Monitor data does not depend on it, so it must not be held hostage."""
    from specola.dashboard.recording import (
        DISCOVERY_FILE,
        read_discovery_document,
    )

    (tmp_path / DISCOVERY_FILE).write_bytes(b"not gzip at all")
    assert read_discovery_document(tmp_path) is None
