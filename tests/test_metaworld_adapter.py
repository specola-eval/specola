"""Sim-free tests for Meta-World's declaration-only adapter policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import specola.adapters.metaworld as metaworld_adapter
from specola.adapters.metaworld import MetaworldAdapter
from specola.adapters.metaworld_declarations import (
    MetaworldDeclarationError,
    evaluate_accessor,
    normalize_orientation,
    project_success_artifact,
    task_declaration,
    task_names,
)
from specola.entities import EntityState, TrackState


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_ARTIFACT = ROOT / "benchmarks" / "metaworld_success_criteria.json"
PACKAGED_MAP = (
    ROOT / "src" / "specola" / "data" / "metaworld_v3_source_map.json"
)
ADAPTER_SOURCE = ROOT / "src" / "specola" / "adapters" / "metaworld.py"


class _NamedData:
    def __init__(self, *, bodies=None, geoms=None, joints=None, sites=None):
        self.bodies = bodies or {}
        self.geoms = geoms or {}
        self.joints = joints or {}
        self.sites = sites or {}
        self.body_calls = []
        self.geom_calls = []
        self.joint_calls = []
        self.site_calls = []

    def body(self, name):
        self.body_calls.append(name)
        return self.bodies[name]

    def geom(self, name):
        self.geom_calls.append(name)
        return self.geoms[name]

    def joint(self, name):
        self.joint_calls.append(name)
        return self.joints[name]

    def site(self, name):
        self.site_calls.append(name)
        return self.sites[name]


def _state_only_adapter(task, environment):
    adapter = object.__new__(MetaworldAdapter)
    adapter._task_declaration = task_declaration(task)
    adapter._metaworld_env = environment
    adapter._timestep = 0
    adapter._spatial_source = lambda item: None
    adapter._source_linear_velocity = lambda item: None
    return adapter


def _registrations(task):
    declaration = task_declaration(task)
    return {
        "target": 10,
        **{
            "object:{}".format(item.index): 20 + item.index
            for item in declaration.objects
        },
    }


def test_packaged_map_is_exact_projection_of_canonical_task2_artifact():
    source = json.loads(SUCCESS_ARTIFACT.read_text())
    packaged = json.loads(PACKAGED_MAP.read_text())

    assert project_success_artifact(source) == packaged
    assert packaged["task_count"] == 50
    assert sum(len(row["objects"]) for row in packaged["tasks"]) == 53
    assert sum(len(row["orientations"]) for row in packaged["tasks"]) == 53
    assert all("criterion" not in row for row in packaged["tasks"])
    assert all("runtime" not in row for row in packaged["tasks"])


def test_packaged_map_preserves_source_kind_and_identity_regressions():
    assembly = task_declaration("assembly-v3").objects
    door_lock = task_declaration("door-lock-v3").objects
    hammer = task_declaration("hammer-v3").objects
    sweep = task_declaration("sweep-v3").objects

    assert [(item.kind, item.declared_name) for item in assembly] == [
        ("site", "RoundNut-8")
    ]
    assert [(item.kind, item.declared_name) for item in door_lock] == [
        ("site_helper", "lockStartLock")
    ]
    assert [(item.kind, item.declared_name) for item in hammer] == [
        ("body_com", "hammer"),
        ("body_com", "nail_link"),
    ]
    assert [(item.kind, item.declared_name) for item in sweep] == [
        ("body_frame", "obj")
    ]


def test_exact_v3_denominator_is_50_tasks_53_objects():
    names = task_names()

    assert len(names) == 50
    assert len(set(names)) == 50
    assert sum(len(task_declaration(name).objects) for name in names) == 53


def test_module_has_no_criterion_or_region_policy():
    tree = ast.parse(ADAPTER_SOURCE.read_text())
    forbidden_calls = {
        "compute_reward",
        "evaluate_state",
        "get_success",
        "is_success",
        "success",
    }
    observed_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    observed_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    assert "RegionSpec" not in observed_names
    assert "_discover_extra_regions" not in MetaworldAdapter.__dict__
    assert observed_calls.isdisjoint(forbidden_calls)


def test_default_body_policy_has_no_object_name_heuristic():
    adapter = object.__new__(MetaworldAdapter)
    rules = adapter._default_body_rules()

    assert len(rules) == 1
    assert rules[0].matches("base")
    assert not rules[0].matches("obj")
    assert not rules[0].matches("objA")
    assert not rules[0].matches("hammer")


def test_target_and_site_declared_object_materialize_with_exact_provenance():
    site_position = np.array([0.1, 0.2, 0.3])
    quaternion = np.array([0.9, 0.1, 0.2, 0.3])
    target = np.array([0.4, 0.5, 0.6])
    data = _NamedData(
        bodies={"RoundNut": SimpleNamespace(xquat=quaternion)},
        sites={
            "RoundNut-8": SimpleNamespace(xpos=site_position),
            "goal": SimpleNamespace(xpos=np.zeros(3)),
        },
    )
    environment = SimpleNamespace(
        data=data,
        _get_pos_goal=lambda: target.copy(),
        _get_pos_objects=lambda: site_position.copy(),
        _get_quat_objects=lambda: quaternion.copy(),
    )
    adapter = _state_only_adapter("assembly-v3", environment)

    specs = adapter._declare_task_entities()
    states = adapter._declared_entity_states(
        _registrations("assembly-v3")
    )

    assert [(spec.name, spec.source_kind) for spec in specs] == [
        ("target", "task_accessor"),
        ("RoundNut-8", "site"),
    ]
    assert "envs/sawyer_assembly_peg_v3.py:105" in specs[1].source_provenance
    np.testing.assert_array_equal(states["target"].position, target)
    np.testing.assert_array_equal(states["object:0"].position, site_position)
    assert states["target"].properties == {
        "target_pos_0": 0.4,
        "target_pos_1": 0.5,
        "target_pos_2": 0.6,
    }
    assert states["object:0"].properties["object_quat_w"] == 0.9
    assert "goal" not in data.site_calls
    metadata = adapter._declared_entity_property_metadata("object:0")
    assert metadata["object_pos_0"].tags == (
        "metaworld:declared",
        "position_source_kind:site",
        "role:target",
    )
    assert metadata["object_quat_w"].tags == (
        "metaworld:declared",
        "role:target",
        "orientation_source_kind:body_quaternion",
    )


def test_discovery_warns_when_raw_goal_site_disagrees_with_declared_target():
    adapter = object.__new__(MetaworldAdapter)
    adapter.name_to_id = {
        "site:goal": 4,
        "declared:target": 9,
    }
    entities = {
        4: EntityState(
            entity_id=4,
            entity_class="site",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            position=np.array([0.0, 0.0, 0.0]),
        ),
        9: EntityState(
            entity_id=9,
            entity_class="target",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            position=np.array([0.0, 0.0, 0.25]),
        ),
    }

    warnings = adapter.discovery_warnings(entities)

    assert len(warnings) == 1
    assert "differs from declared target 'target' by 0.25 m" in warnings[0]
    assert "tolerance 1e-06 m" in warnings[0]
    assert "_get_pos_goal() is authoritative" in warnings[0]


def test_discovery_does_not_warn_when_raw_goal_site_agrees_with_target():
    adapter = object.__new__(MetaworldAdapter)
    adapter.name_to_id = {
        "site:goal": 4,
        "declared:target": 9,
    }
    entities = {
        4: EntityState(
            entity_id=4,
            entity_class="site",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            position=np.array([0.0, 0.0, 0.0]),
        ),
        9: EntityState(
            entity_id=9,
            entity_class="target",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            position=np.array([0.0, 0.0, 5e-7]),
        ),
    }

    assert adapter.discovery_warnings(entities) == ()


def test_zero_placeholder_is_reconciled_but_not_exposed_as_orientation():
    body_position = np.array([0.3, 0.4, 0.5])
    declared_position = body_position + np.array([0.0, -0.16, 0.05])
    environment = SimpleNamespace(
        data=_NamedData(),
        get_body_com=lambda name: body_position.copy(),
        _get_pos_goal=lambda: np.array([0.0, 0.0, 0.0]),
        _get_pos_objects=lambda: declared_position.copy(),
        _get_quat_objects=lambda: np.zeros(4),
    )
    adapter = _state_only_adapter("drawer-close-v3", environment)

    state = adapter._declared_entity_states(
        _registrations("drawer-close-v3")
    )["object:0"]
    metadata = adapter._declared_entity_property_metadata("object:0")

    assert set(state.properties) == {
        "object_pos_0",
        "object_pos_1",
        "object_pos_2",
    }
    assert set(metadata) == set(state.properties)


def test_scipy_orientation_is_reordered_to_common_wxyz_representation():
    result = normalize_orientation(
        np.array([0.1, 0.2, 0.3, 0.9]),
        "scipy_xyzw",
    )

    np.testing.assert_array_equal(result, [0.9, 0.1, 0.2, 0.3])


def test_runtime_disagreement_fails_without_nearest_assignment():
    source_position = np.array([0.1, 0.2, 0.3])
    declared_position = np.array([0.1, 0.2, 0.300001])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    environment = SimpleNamespace(
        data=_NamedData(
            bodies={"RoundNut": SimpleNamespace(xquat=quaternion)},
            sites={"RoundNut-8": SimpleNamespace(xpos=source_position)},
        ),
        _get_pos_goal=lambda: np.zeros(3),
        _get_pos_objects=lambda: declared_position.copy(),
        _get_quat_objects=lambda: quaternion.copy(),
    )
    adapter = _state_only_adapter("assembly-v3", environment)

    with pytest.raises(
        MetaworldDeclarationError,
        match="disagrees with its source accessor",
    ):
        adapter._declared_entity_states(_registrations("assembly-v3"))


def test_coincident_unreferenced_body_cannot_change_hammer_identity():
    positions = {
        "hammer": np.array([0.1, 0.2, 0.3]),
        "nail_link": np.array([0.4, 0.5, 0.6]),
        "coincident_rogue": np.array([0.1, 0.2, 0.3]),
    }
    quaternions = {
        "hammer": np.array([1.0, 0.0, 0.0, 0.0]),
        "nail_link": np.array([0.9, 0.1, 0.2, 0.3]),
        "coincident_rogue": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    data = _NamedData(
        bodies={
            name: SimpleNamespace(xquat=quaternion)
            for name, quaternion in quaternions.items()
        }
    )
    body_com_calls = []

    def get_body_com(name):
        body_com_calls.append(name)
        return positions[name].copy()

    environment = SimpleNamespace(
        data=data,
        get_body_com=get_body_com,
        _get_pos_goal=lambda: np.zeros(3),
        _get_pos_objects=lambda: np.concatenate(
            (positions["hammer"], positions["nail_link"])
        ),
        _get_quat_objects=lambda: np.concatenate(
            (quaternions["hammer"], quaternions["nail_link"])
        ),
    )
    adapter = _state_only_adapter("hammer-v3", environment)

    states = adapter._declared_entity_states(_registrations("hammer-v3"))

    assert [spec.name for spec in adapter._declare_task_entities()[1:]] == [
        "hammer",
        "nail_link",
    ]
    assert [
        spec.tags[-1]
        for spec in adapter._declare_task_entities()[1:]
    ] == ["role:target", "role:distractor"]
    assert set(states) == {"target", "object:0", "object:1"}
    assert body_com_calls == ["hammer", "nail_link"]
    assert data.body_calls == ["hammer", "nail_link"]
    assert "coincident_rogue" not in body_com_calls + data.body_calls


def test_accessor_interpreter_rejects_criterion_and_unreviewed_calls():
    environment = SimpleNamespace()

    for expression in (
        "self.evaluate_state()",
        "self.compute_reward()",
        "__import__('os').getcwd()",
        "self.data.body(name).xpos",
    ):
        with pytest.raises(MetaworldDeclarationError):
            evaluate_accessor(expression, environment)


def test_dial_accessor_interpreter_preserves_authored_offset():
    body_position = np.array([0.2, 0.3, 0.4])
    qpos = np.array([np.pi / 2.0])
    environment = SimpleNamespace(
        data=_NamedData(
            joints={"knob_Joint_1": SimpleNamespace(qpos=qpos)}
        ),
        get_body_com=lambda name: body_position.copy(),
    )
    expression = task_declaration("dial-turn-v3").objects[0].accessor

    evaluated = evaluate_accessor(expression, environment)

    expected = body_position + np.array([0.05, 0.0, 0.0])
    np.testing.assert_allclose(evaluated, expected, atol=1e-15)


def test_registry_factory_draws_the_task_variant_from_the_seed(monkeypatch):
    import sys

    benchmarks = []

    class FakeEnv:
        def set_task(self, task):
            self.task = task

        def reset(self):
            pass

    class FakeMT1:
        def __init__(self, env_name, seed=None):
            benchmarks.append(seed)
            self.train_classes = {env_name: FakeEnv}
            self.train_tasks = ["variant drawn from {}".format(seed)]

    monkeypatch.setitem(sys.modules, "metaworld", SimpleNamespace(MT1=FakeMT1))
    monkeypatch.setattr(metaworld_adapter, "MetaworldAdapter", lambda env, **kwargs: env)
    env = metaworld_adapter._make_metaworld("pick-place-v3", None, False, seed=7)
    assert benchmarks == [7] and env.task == "variant drawn from 7"
    metaworld_adapter._make_metaworld("pick-place-v3", None, False)
    assert benchmarks == [7, None]
