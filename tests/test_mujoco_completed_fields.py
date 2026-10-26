"""Exercise typed capture and declared layouts without invoking a simulator.

Allocation and numerical fidelity use the separate independent native C
fixture. The storage stand-in here only tests capture joins and serialization.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from specola.adapters import mujoco_completed_fields as completed
from specola.adapters.mujoco_fields import NativeFieldValue
from specola.adapters.mujoco_logging import NativeLog
from specola.conformance.query_bindings import QueryBindingError
from specola.native_values import NativeValue


VERSIONS = tuple(path.stem.removeprefix("mujoco-projection-") for path in
                 sorted((Path(completed.__file__).parent.parent / "data").glob("mujoco-projection-*.json")))
CONFIGURATION = {"identity": "owned pure storage fixture", "sha256": "1" * 64}


def fixture(monkeypatch, version, damage=""):
    subject = completed.MujocoCompletedFields(object(), SimpleNamespace(_version=version, check=lambda: None))
    counts = {"nbody": 3, "njnt": 2, "ngeom": 4, "nsite": 0, "ncam": 1, "nlight": 0}
    plans = {row["field_id"]: row["plans"][0] for row in subject._document["storage"]["fields"] if row["plans"]}
    arrays, values = {}, []
    for identity, contract in subject._contracts.items():
        shape = tuple(int(value) if str(value).isdecimal() else counts[value]
                      for value in contract["binding_shape"])
        array = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        if contract["field"] == "energy":
            array[:] = [-0.0, np.nan]
        arrays[identity] = array
        plan = plans[identity]
        values.append(NativeFieldValue(identity, "mjData_", tuple(plan["native_path"]),
                                       shape, NativeValue.capture(array)))
    if damage == "reader":
        values = [value for value in values if not value.field_id.endswith("@energy")]
    elif damage == "owner":
        values = [replace(value, value=NativeValue.capture(np.zeros((2, 3))))
                  if value.field_id.endswith("@xpos") else value for value in values]
    calls = []

    class Storage:
        def __init__(self, *args, **kwargs):
            calls.append("open")

        def _root_integer(self, root, field):
            assert root == "mjModel_"
            calls.append("count")
            return counts[field]

        def read(self):
            calls.append("read")
            return tuple(values)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(completed, "MujocoNativeFields", Storage)
    stages = tuple(dict.fromkeys(identity.removeprefix("c:@F@") for row in subject._contracts.values()
                                 for identity in row["completion"]["function_header_ids"]))
    log = NativeLog(2, 1, b"subject", None, b"", b"fixture.c", 9, False)
    snapshot = SimpleNamespace(data=object(), completed=stages,
                               completion_warnings=(SimpleNamespace(message=b"owned warning"),),
                               completion_logs=(log,))
    snapshot.data = SimpleNamespace(energy=np.array([-0.0, 41.5], dtype=np.float64))
    snapshot.warnings = snapshot.completion_warnings
    snapshot.logs = snapshot.completion_logs
    snapshot.stage_results = {}
    for identity, value in subject.capture_stage("mj_energyPos", snapshot,
            stages[:stages.index("mj_energyPos") + 1], 0, 0):
        snapshot.stage_results[identity] = {"mj_energyPos": value}
    return subject, snapshot, arrays, calls


@pytest.mark.parametrize("version", VERSIONS)
def test_capture_preserves_declared_payload_and_configuration_order(monkeypatch, version):
    subject, snapshot, arrays, calls = fixture(monkeypatch, version)
    captured = subject.capture(snapshot, dict(reversed(tuple(CONFIGURATION.items()))))
    quantities = {row.quantity_id: row for row in subject.quantities}
    metadata = {row.quantity_id: row for row in captured.metadata}
    for sample in captured.samples:
        quantity = quantities[sample.quantity_id]
        quantity.outcomes[0].native_schema.validate(sample.value.document(), sample.quantity_id)
        payload = sample.value.restore()
        field = sample.quantity_id.removeprefix("mujoco.completed.mjData.")
        expected = arrays["c:@S@mjData_@FI@" + field]
        assert payload["value"].dtype == expected.dtype
        assert payload["value"].shape == expected.shape
        assert payload["value"].tobytes() == expected.tobytes()
        assert payload["configuration_identity"] == CONFIGURATION
        assert sample.owner_kind == "mjData" and sample.owner_index == ()
        assert sample.scalar_keys == () and sample.entity_id is None
        assert metadata[sample.quantity_id].source_members
        before = sample.value.document()
        expected.fill(31)
        assert sample.value.document() == before
    assert calls[0] == "open" and calls[-1] == "close"
    assert calls.index("read") > calls.index("count")


@pytest.mark.parametrize("damage", ["reader", "owner"])
def test_missing_field_or_owner_refuses_capture(monkeypatch, damage):
    subject, snapshot, _, calls = fixture(monkeypatch, VERSIONS[0], damage)
    with pytest.raises(QueryBindingError, match="population differs|omitted a physical owner"):
        subject.capture(snapshot, CONFIGURATION)
    assert calls[-1] == "close"


def test_missing_completion_refuses_before_storage(monkeypatch):
    subject, snapshot, _, calls = fixture(monkeypatch, VERSIONS[0])
    snapshot.completed = ()
    with pytest.raises(QueryBindingError, match="stage was not called"):
        subject.capture(snapshot, CONFIGURATION)
    assert calls == []


@pytest.mark.parametrize("damage", ["energy_component", "log_component", "dtype", "frame",
                                    "stage_missing", "stage_component", "stage_prefix", "stage_dtype"])
def test_closed_result_schema_rejects_component_and_type_loss(monkeypatch, damage):
    subject, snapshot, _, _ = fixture(monkeypatch, VERSIONS[0])
    sample = next(item for item in subject.capture(snapshot, CONFIGURATION).samples
                  if item.quantity_id.endswith(".energy"))
    quantity = next(item for item in subject.quantities if item.quantity_id == sample.quantity_id)
    payload = sample.value.restore()
    if damage == "energy_component":
        payload["value"] = payload["value"][:1]
    elif damage == "log_component":
        del payload["completion_logs"][0]["topic"]
    elif damage == "dtype":
        payload["value"] = payload["value"].astype(np.float32)
    elif damage == "stage_missing":
        del payload["stage_results"]
    elif damage == "stage_component":
        payload["stage_results"]["mj_energyPos"]["value"] = np.zeros(1, dtype=np.float64)
    elif damage == "stage_prefix":
        payload["stage_results"]["mj_energyPos"]["completion"] = snapshot.completed
    elif damage == "stage_dtype":
        payload["stage_results"]["mj_energyPos"]["value"] = np.zeros(2, dtype=np.float32)
    else:
        # A new frame result cannot silently extend the declared payload.
        payload["frame"] = "wrong frame"
    with pytest.raises(QueryBindingError):
        quantity.outcomes[0].native_schema.validate(NativeValue.capture(payload).document(), sample.quantity_id)


@pytest.mark.parametrize("version", VERSIONS)
def test_interpretation_contracts_survive_quantity_metadata(monkeypatch, version):
    subject, _, _, _ = fixture(monkeypatch, version)
    metadata = {item.quantity_id: item for item in subject.metadata}
    for contract in subject._contracts.values():
        schema = json.loads(metadata[contract["projection_id"]].native_schema)
        semantic = schema["outcomes"]["completed_from_stored_state"]["x-specola-native-semantics"]
        for key in ("stored_input_contract", "native_energy_branches", "stage_result"):
            if key in contract:
                assert semantic[key] == contract[key]


@pytest.mark.parametrize("version", VERSIONS)
def test_intermediate_energy_keeps_its_own_bytes_prefix_and_diagnostics(monkeypatch, version):
    subject, snapshot, _, _ = fixture(monkeypatch, version)
    source = snapshot.data.energy.copy()
    snapshot.data.energy[:] = [93, 94]
    sample = next(item for item in subject.capture(snapshot, CONFIGURATION).samples
                  if item.quantity_id.endswith(".energy"))
    payload = sample.value.restore()
    intermediate = payload["stage_results"]["mj_energyPos"]
    assert intermediate["value"].tobytes() == source.tobytes()
    assert intermediate["value"].tobytes() != payload["value"].tobytes()
    assert intermediate["completion"] == snapshot.completed[:snapshot.completed.index("mj_energyPos") + 1]
    assert intermediate["warnings"] == (b"owned warning",)
    assert intermediate["logs"][0]["subject"] == b"subject"


def test_missing_intermediate_reader_is_rejected_before_final_storage(monkeypatch):
    subject, snapshot, _, calls = fixture(monkeypatch, VERSIONS[0])
    snapshot.stage_results.clear()
    with pytest.raises(QueryBindingError, match="intermediate field population"):
        subject.capture(snapshot, CONFIGURATION)
    assert calls == []


def test_actual_projection_loop_captures_before_velocity_overwrites_energy(monkeypatch):
    from contextlib import contextmanager, nullcontext
    from specola.adapters.mujoco_projection import MujocoProjection
    from specola.adapters.mujoco_scratch import ScratchObservation

    subject, _, _, _ = fixture(monkeypatch, VERSIONS[0])
    source = SimpleNamespace(energy=np.array([9.0, 13.0]))
    scratch = SimpleNamespace(energy=source.energy.copy())

    def position(model, data):
        data.energy[0] = 17.0

    def velocity(model, data):
        data.energy[1] = 23.0

    def unchanged(model, data):
        pass

    functions = {"mj_comVel": unchanged, "mj_subtreeVel": unchanged,
                 "mj_rnePostConstraint": unchanged, "mj_energyPos": position, "mj_energyVel": velocity}

    @contextmanager
    def observe(data, *, completion):
        yield ScratchObservation(data, scratch, completion, [])

    projection = MujocoProjection.__new__(MujocoProjection)
    projection._closed = False
    projection._model = object()
    projection._functions = functions
    projection._energy_stages = lambda: ()
    projection._completed_fields = subject
    projection._guard = SimpleNamespace(function=functions.__getitem__)
    monkeypatch.setattr(projection, "_timer", SimpleNamespace(preserve=nullcontext), raising=False)
    projection._session = SimpleNamespace(_validate_subject=lambda data: None, observe=observe)
    with projection.observe(source) as snapshot:
        retained = snapshot.stage_results["c:@S@mjData_@FI@energy"]["mj_energyPos"].restore()
        assert retained["value"].tolist() == [17.0, 13.0]
        assert snapshot.data.energy.tolist() == [17.0, 23.0]
        assert retained["completion"][-1] == "mj_energyPos"
        assert "mj_energyVel" not in retained["completion"]
    assert source.energy.tolist() == [9.0, 13.0]
