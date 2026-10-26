"""Sim-free tests of the shared Isaac Lab state-tier helpers (adapters/isaac_state.py).

The tensor tests need torch and the layer tests need pxr; each is skipped
where its package is absent (torch is in the ``libero`` and ``isaacsim``
environments, pxr in ``isaacsim``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from specola.adapters import isaac_state


def _state(torch, num_envs=2):
    """A get_state dictionary: two articulations (one without joints) and a gripper."""
    rng = np.random.default_rng(0)

    def leaf(*shape):
        return torch.as_tensor(rng.standard_normal((num_envs, *shape)).astype(np.float32))

    return {
        "articulation": {
            "robot": {"root_pose": leaf(7), "root_velocity": leaf(6),
                      "joint_position": leaf(3), "joint_velocity": leaf(3)},
            "peg": {"root_pose": leaf(7), "root_velocity": leaf(6),
                    "joint_position": leaf(0), "joint_velocity": leaf(0)},
        },
        "deformable_object": {},
        "rigid_object": {},
        "gripper": {"suction": torch.as_tensor(np.array([-1, 1][:num_envs], dtype=np.int32))},
    }


def test_flatten_and_unflatten_round_trip_one_environment():
    torch = pytest.importorskip("torch")
    state = _state(torch)
    rows = isaac_state.layout(state)
    assert rows[0] == ["robot", "articulation", "root_pose", [7], "float32"]
    assert ["peg", "articulation", "joint_position", [0], "float32"] in rows
    assert rows[-1] == ["suction", "gripper", "", [], "int32"]

    flat = isaac_state.flatten(state, 1)
    assert flat.dtype == np.float64
    assert flat.shape == (isaac_state.state_length(rows),) == (7 + 6 + 3 + 3 + 7 + 6 + 1,)

    rebuilt = isaac_state.unflatten(rows, flat)
    for asset, kind, key, _, _ in rows:
        original = state[kind][asset][key] if key else state[kind][asset]
        leaf = rebuilt[kind][asset][key] if key else rebuilt[kind][asset]
        # The recorded dtype and a leading dimension of 1: the articulation
        # writers assert float32 and index by env_ids.
        assert leaf.dtype == original.dtype
        assert tuple(leaf.shape) == (1, *original.shape[1:])
        assert torch.equal(leaf[0], original[1])

    with pytest.raises(ValueError, match="length 33"):
        isaac_state.unflatten(rows, flat[:-1])


def test_surface_grippers_refuse_the_tier_beyond_one_environment():
    scene = SimpleNamespace(surface_grippers={"suction": object()})
    assert isaac_state.gripper_refusal(scene, 1) is None
    reason = isaac_state.gripper_refusal(scene, 2)
    assert "surface grippers" in reason and "2 environments" in reason
    assert isaac_state.gripper_refusal(SimpleNamespace(surface_grippers={}), 2) is None


class _Holder:
    def method(self):
        return None


def test_normalisation_is_stable_and_serialisable(tmp_path):
    asset = tmp_path / "robot.usd"
    asset.write_bytes(b"#usda 1.0\n")
    function = lambda x: x  # noqa: E731
    config = {
        "func": function,
        "method": _Holder().method,
        "usd_path": str(asset),
        "remote": "https://example.com/robot.usd",
        "gains": (1.0, 2.5),
        "limit": float("inf"),
        "count": 3,
        "flag": True,
        "nothing": None,
        "ids": slice(None),
        "handle": object(),
        "nested": {7: [np.float32(0.5)]},
    }
    out = isaac_state.normalise(config)
    assert out["func"] == f"{function.__module__}.{function.__qualname__}"
    assert out["method"] == f"{_Holder.__module__}._Holder.method"
    assert out["usd_path"] == hashlib.sha256(b"#usda 1.0\n").hexdigest()
    assert out["remote"] == "https://example.com/robot.usd"
    assert out["gains"] == ["1.0", "2.5"]
    assert out["limit"] == "inf"
    assert (out["count"], out["flag"], out["nothing"]) == (3, True, None)
    assert out["ids"] == "slice(None, None, None)"
    assert "0x" not in out["handle"]
    assert out["nested"] == {"7": ["0.5"]}
    json.dumps(out)
    assert isaac_state.config_sha256(config) == isaac_state.config_sha256(dict(config))


def test_one_field_of_the_configuration_changes_the_digest():
    config = {"sim": {"dt": 1 / 120, "gravity": (0.0, 0.0, -9.81)}, "seed": 7}
    mutated = copy.deepcopy(config)
    mutated["sim"]["gravity"] = (0.0, 0.0, -9.80)
    assert isaac_state.config_sha256(config) == isaac_state.config_sha256(copy.deepcopy(config))
    assert isaac_state.config_sha256(config) != isaac_state.config_sha256(mutated)


def test_physics_digest_reads_the_monitored_environment_only():
    def scene(mass):
        peg = SimpleNamespace(
            data=SimpleNamespace(body_mass=mass, body_inertia=np.ones((2, 1, 9), np.float32)),
            root_view=SimpleNamespace(get_material_properties=lambda: np.full((2, 1, 3), 0.75, np.float32)),
        )
        return SimpleNamespace(articulations={"peg": peg})

    rows = [["peg", "articulation", "root_pose", [7], "float32"]]
    base = np.array([[0.019], [0.019]], np.float32)
    heavier = base.copy()
    heavier[1, 0] = 0.021
    assert isaac_state.physics_sha256(scene(base), rows, 0) == isaac_state.physics_sha256(scene(heavier), rows, 0)
    assert isaac_state.physics_sha256(scene(base), rows, 1) != isaac_state.physics_sha256(scene(heavier), rows, 1)


def test_layer_records_identify_content_not_identifiers(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    def stage(size):
        built = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(built, "/World/Box").GetSizeAttr().Set(size)
        return built

    first, second, changed = stage(1.0), stage(1.0), stage(2.0)
    identifiers = [layer.identifier for layer in first.GetUsedLayers()]
    assert identifiers != [layer.identifier for layer in second.GetUsedLayers()]
    records = isaac_state.layer_records(first)
    assert records == isaac_state.layer_records(second)
    assert records != isaac_state.layer_records(changed)
    assert all(record[0] == "anon" for record in records)
    assert not any(identifier in json.dumps(records) for identifier in identifiers)

    on_disk = tmp_path / "box.usda"
    first.GetRootLayer().Export(str(on_disk))   # the stage must outlive its layer handle
    opened = Usd.Stage.Open(str(on_disk))
    digest = hashlib.sha256(on_disk.read_bytes()).hexdigest()
    assert ["file", str(on_disk), digest] in isaac_state.layer_records(opened)
