"""Sim-free regressions for LIBERO-specific declaration policy."""

from __future__ import annotations

import ast
import inspect
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import sentinel

import numpy as np
import pytest

import specola.adapters.libero as libero_adapter
from specola.adapters.libero import LIBEROAdapter
from specola.adapters.mujoco import MujocoAdapter


def _fake_libero_modules(monkeypatch):
    task = SimpleNamespace(
        problem_folder="suite",
        bddl_file="task.bddl",
    )

    class _TaskSuite:
        def get_task(self, task_id):
            assert task_id == 0
            return task

    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {
            "libero_object": _TaskSuite,
        }
    )
    created = []

    class _BaseFakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.seed_value = None
            self.reset_called = False
            created.append(self)

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            self.reset_called = True

    class FakeControlEnv(_BaseFakeEnv):
        pass

    class FakeOffScreenRenderEnv(_BaseFakeEnv):
        pass

    libero_package = ModuleType("libero")
    libero_module = ModuleType("libero.libero")
    libero_module.benchmark = benchmark
    libero_module.get_libero_path = lambda _kind: "/declarations"
    envs_module = ModuleType("libero.libero.envs")
    envs_module.OffScreenRenderEnv = FakeOffScreenRenderEnv
    wrapper_module = ModuleType("libero.libero.envs.env_wrapper")
    wrapper_module.ControlEnv = FakeControlEnv
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)
    monkeypatch.setitem(
        sys.modules,
        "libero.libero.envs.env_wrapper",
        wrapper_module,
    )
    monkeypatch.setattr(libero_adapter.os.path, "isfile", lambda _path: True)
    return created, FakeControlEnv, FakeOffScreenRenderEnv


def test_make_env_defaults_to_non_rendering_control_env(monkeypatch):
    created, control_class, _offscreen_class = _fake_libero_modules(
        monkeypatch
    )

    env, _task = libero_adapter.make_libero_env()

    assert isinstance(env, control_class)
    assert env.kwargs["has_offscreen_renderer"] is False
    assert env.kwargs["use_camera_obs"] is False
    assert env.reset_called is True
    assert len(created) == 1


def test_make_env_retains_explicit_offscreen_option(monkeypatch):
    created, _control_class, offscreen_class = _fake_libero_modules(
        monkeypatch
    )

    env, _task = libero_adapter.make_libero_env(
        offscreen_renderer=True
    )

    assert isinstance(env, offscreen_class)
    assert env.kwargs["has_offscreen_renderer"] is True
    assert env.kwargs["use_camera_obs"] is False
    assert len(created) == 1


@pytest.mark.parametrize("options,offscreen", (({}, True), ({"capture_frames": False}, False)))
def test_registry_factory_builds_the_renderer_only_when_frames_may_be_captured(monkeypatch, options, offscreen):
    created, control_class, offscreen_class = _fake_libero_modules(monkeypatch)
    monkeypatch.setattr(libero_adapter, "LIBEROAdapter", lambda env, **kwargs: env)

    env = libero_adapter._make_libero("libero_object:0", None, True, **options)

    assert isinstance(env, offscreen_class if offscreen else control_class)
    assert env.kwargs["has_offscreen_renderer"] is offscreen
    assert len(created) == 1


def _declaration(
    name,
    *,
    root_body=None,
    site_type=None,
    size=None,
    parent_name=None,
    joints=(),
    contact_geoms=None,
):
    if contact_geoms is None:
        contact_geoms = () if root_body is None else (f"{name}_geom",)
    return SimpleNamespace(
        name=name,
        root_body=root_body,
        site_type=site_type,
        size=size,
        parent_name=parent_name,
        joints=list(joints),
        contact_geoms=list(contact_geoms),
    )


def _snapshot_adapter(
    *,
    objects=None,
    fixtures=None,
    sites=None,
    parsed_objects=None,
    parsed_fixtures=None,
    parsed_regions=None,
    goal_state=None,
    obj_of_interest=None,
    workspace_name="main_table",
    body_names=None,
    model_site_names=None,
    model_site_sizes=None,
    model_site_types=None,
    geom_names=None,
):
    objects = objects or {
        "cup_1": _declaration(
            "cup_1",
            root_body="cup_1_main",
            joints=("cup_1_joint0",),
        )
    }
    fixtures = fixtures or {
        "cabinet_1": _declaration(
            "cabinet_1",
            root_body="cabinet_1_main",
            joints=("cabinet_1_drawer",),
        )
    }
    sites = sites or {
        "cabinet_1_drawer_region": _declaration(
            "cabinet_1_drawer_region",
            site_type="box",
            size=(0.1, 0.2, 0.3),
            parent_name="cabinet_1",
            joints=("cabinet_1_drawer",),
        )
    }
    parsed_objects = parsed_objects or {
        "cup": list(objects),
    }
    parsed_fixtures = parsed_fixtures or {
        "table": [workspace_name],
        "cabinet": list(fixtures),
    }
    parsed_regions = parsed_regions or {
        **{name: {} for name in sites},
        "unused_fixture_1_region": {},
    }
    goal_state = (
        goal_state
        if goal_state is not None
        else [["On", next(iter(objects)), "cabinet_1_drawer_region"]]
    )
    obj_of_interest = (
        obj_of_interest
        if obj_of_interest is not None
        else [next(iter(objects))]
    )
    body_names = body_names or (
        "world",
        "robot0_base",
        "table",
        "cup_1_main",
        "cabinet_1_main",
    )
    model_site_names = model_site_names or tuple(sites)
    model_site_sizes = (
        model_site_sizes
        if model_site_sizes is not None
        else [sites[name].size for name in model_site_names]
    )
    model_site_types = (
        model_site_types
        if model_site_types is not None
        else ["box" for _name in model_site_names]
    )
    geom_names = geom_names or tuple(
        geom_name
        for declaration in (*objects.values(), *fixtures.values())
        for geom_name in declaration.contact_geoms
    )

    inner = SimpleNamespace(
        parsed_problem={
            "objects": parsed_objects,
            "fixtures": parsed_fixtures,
            "regions": parsed_regions,
            "goal_state": goal_state,
        },
        objects_dict=objects,
        object_states_dict={},
        object_sites_dict=sites,
        fixtures_dict=fixtures,
        obj_of_interest=obj_of_interest,
        workspace_name=workspace_name,
        _arena_type="table",
    )
    wrapper = SimpleNamespace(env=SimpleNamespace(env=inner))

    adapter = object.__new__(LIBEROAdapter)
    adapter.env = wrapper
    adapter._declaration_snapshot_value = None
    adapter._model = SimpleNamespace(
        site_size=np.asarray(model_site_sizes, dtype=float)
    )
    names = {
        "body": tuple(body_names),
        "site": tuple(model_site_names),
        "geom": tuple(geom_names),
    }
    adapter._names = lambda kind: names[kind]
    adapter._name2id = lambda kind, name: names[kind].index(name)
    adapter._id2name = lambda kind, object_id: names[kind][object_id]
    adapter._geom_type_name = (
        lambda object_id, site=False: model_site_types[object_id]
    )
    return adapter


def _relation_state_adapter(*, with_parent=True):
    sites = {
        "cabinet_1_drawer_region": _declaration(
            "cabinet_1_drawer_region",
            site_type="box",
            size=(0.1, 0.2, 0.3),
            parent_name="cabinet_1" if with_parent else None,
            joints=("cabinet_1_drawer",) if with_parent else (),
        )
    }
    adapter = _snapshot_adapter(
        objects={
            "cup_1": _declaration(
                "cup_1",
                root_body="cup_1_main",
                contact_geoms=("cup_collision",),
            ),
            "plate_1": _declaration(
                "plate_1",
                root_body="plate_1_main",
                contact_geoms=("plate_collision",),
            ),
        },
        fixtures={
            "cabinet_1": _declaration(
                "cabinet_1",
                root_body="cabinet_1_main",
                contact_geoms=("cabinet_collision",),
            )
        },
        sites=sites,
        body_names=(
            "world",
            "robot0_base",
            "table",
            "cup_1_main",
            "plate_1_main",
            "cabinet_1_main",
        ),
        geom_names=(
            "cup_collision",
            "plate_collision",
            "cabinet_collision",
        ),
        obj_of_interest=("cup_1", "plate_1"),
    )
    adapter._data = SimpleNamespace(
        body_xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.5, 0.8],
                [0.1, 0.1, 0.6],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        site_xpos=np.asarray([[0.2, 0.3, 0.4]], dtype=float),
        site_xmat=np.asarray(
            [
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=float,
        ),
        ncon=0,
        contact=(),
    )
    adapter._timestep = 0
    return adapter


def test_constructor_initializes_empty_declaration_snapshot(monkeypatch):
    def fake_mujoco_init(self, env, **kwargs):
        assert self._declaration_snapshot_value is None

    monkeypatch.setattr(MujocoAdapter, "__init__", fake_mujoco_init)

    LIBEROAdapter(object())


def test_reset_invalidates_declaration_snapshot(monkeypatch):
    adapter = object.__new__(LIBEROAdapter)
    adapter._declaration_snapshot_value = sentinel.old_snapshot

    monkeypatch.setattr(
        MujocoAdapter,
        "_reset_impl",
        lambda self: sentinel.observation,
    )

    assert adapter._reset_impl() is sentinel.observation
    assert adapter._declaration_snapshot_value is None


def test_inner_environment_is_reached_through_nested_wrappers():
    declarations = {
        name: object()
        for name in LIBEROAdapter._INNER_DECLARATION_ATTRIBUTES
    }
    inner = type("_Inner", (), declarations)()
    wrapper = type(
        "_Wrapper",
        (),
        {"env": type("_Middle", (), {"env": inner})()},
    )()
    adapter = object.__new__(LIBEROAdapter)
    adapter.env = wrapper

    assert adapter._libero_inner_env() is inner


def test_inner_environment_missing_declarations_fails_loudly():
    adapter = object.__new__(LIBEROAdapter)
    adapter.env = type("_Wrapper", (), {"env": object()})()

    with pytest.raises(RuntimeError, match="objects_dict"):
        adapter._libero_inner_env()


def test_snapshot_reconciles_declared_roots_and_instantiated_regions():
    adapter = _snapshot_adapter()

    snapshot = adapter._declaration_snapshot()

    assert snapshot.parsed_object_names == ("cup_1",)
    assert snapshot.parsed_fixture_names == ("cabinet_1", "main_table")
    assert snapshot.objects == (("cup_1", "cup_1_main"),)
    assert snapshot.object_roles == (
        (
            "cup_1",
            "target",
            "structural_derivation",
            "obj_of_interest member; goal operands cross-checked",
        ),
    )
    assert snapshot.fixtures == (("cabinet_1", "cabinet_1_main"),)
    assert [
        (
            record.name,
            record.category,
            record.root_body,
            record.disposition,
        )
        for record in snapshot.arena_fixtures
    ] == [
        ("main_table", "table", "table", "arena_owned_compiled_body"),
    ]
    assert snapshot.object_roots == frozenset({"cup_1_main"})
    assert snapshot.fixture_roots == frozenset(
        {"cabinet_1_main", "table"}
    )
    assert len(snapshot.regions) == 1
    assert snapshot.regions[0].name == "cabinet_1_drawer_region"
    assert snapshot.regions[0].source_name == "cabinet_1_drawer_region"
    assert snapshot.regions[0].half_extents == (0.1, 0.2, 0.3)
    assert snapshot.uninstantiated_region_names == (
        "unused_fixture_1_region",
    )
    assert adapter.uninstantiated_declared_regions == (
        "unused_fixture_1_region",
    )


def test_relation_population_is_static_ordered_and_declaration_backed():
    adapter = _relation_state_adapter()

    specs = adapter._declare_task_entities()

    assert [spec.declaration_key for spec in specs] == [
        "object_relation:cup_1__to__plate_1",
        "object_relation:plate_1__to__cup_1",
        "object_region_relation:cup_1__to__cabinet_1_drawer_region",
        "object_region_relation:plate_1__to__cabinet_1_drawer_region",
    ]
    assert [spec.entity_class for spec in specs] == [
        "object_relation",
        "object_relation",
        "object_region_relation",
        "object_region_relation",
    ]
    assert specs[0].tags == (
        "source_declared_name:cup_1",
        "destination_declared_name:plate_1",
        "declared_pair:cup_1__to__plate_1",
    )


def test_relation_population_is_independent_of_goal_atoms():
    first = _relation_state_adapter()
    second = _relation_state_adapter()
    second._libero_inner_env().parsed_problem["goal_state"] = [
        ["On", "plate_1", "cup_1"]
    ]

    assert [
        spec.declaration_key for spec in first._declare_task_entities()
    ] == [
        spec.declaration_key for spec in second._declare_task_entities()
    ]


def test_object_relation_state_preserves_false_contact():
    adapter = _relation_state_adapter()
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    states = adapter._declared_entity_states(registrations)
    state = states["object_relation:cup_1__to__plate_1"]

    assert state.entity_id == 20
    assert state.entity_class == "object_relation"
    assert state.properties == {
        "horizontal_center_distance": pytest.approx(0.5),
        "source_minus_destination_z": pytest.approx(0.2),
        "declared_geom_contact": 0.0,
    }


def test_declared_contact_matches_all_native_rows_without_solver_filter():
    adapter = _relation_state_adapter()
    adapter._data.ncon = 1
    adapter._data.contact = (
        SimpleNamespace(
            geom1=0,
            geom2=1,
            exclude=1,
            efc_address=-1,
        ),
    )
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    states = adapter._declared_entity_states(registrations)

    assert states[
        "object_relation:cup_1__to__plate_1"
    ].properties["declared_geom_contact"] == 1.0
    assert states[
        "object_relation:plate_1__to__cup_1"
    ].properties["declared_geom_contact"] == 1.0
    assert states[
        "object_region_relation:cup_1__to__cabinet_1_drawer_region"
    ].properties["source_parent_declared_geom_contact"] == 0.0


def test_declared_contact_reads_buffer_as_presented_at_observation_entry():
    # The observation pass snapshots the contact buffer before the
    # kinematic forward; a mid-observation change to the rows (which is
    # what forward's re-detection does) must not alter the reported
    # contact state.
    adapter = _relation_state_adapter()
    adapter._data.ncon = 1
    adapter._data.contact = (SimpleNamespace(geom1=0, geom2=1),)
    adapter._presented_contact_pairs = adapter._collect_native_contact_pairs()
    adapter._data.ncon = 0
    adapter._data.contact = ()
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    states = adapter._declared_entity_states(registrations)

    assert states[
        "object_relation:cup_1__to__plate_1"
    ].properties["declared_geom_contact"] == 1.0
    adapter._presented_contact_pairs = None
    states = adapter._declared_entity_states(registrations)
    assert states[
        "object_relation:cup_1__to__plate_1"
    ].properties["declared_geom_contact"] == 0.0


def test_observation_snapshot_precedes_base_observation_pass():
    # The override must snapshot the buffer BEFORE the base observation
    # pass (whose kinematic forward re-detects contacts). Simulate the
    # re-detection by zeroing the buffer inside the base call and assert
    # the declared read still reflects the entry-time buffer.
    adapter = _relation_state_adapter()
    adapter._data.ncon = 1
    adapter._data.contact = (SimpleNamespace(geom1=0, geom2=1),)
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }
    captured = {}

    def fake_base(self):
        self._data.ncon = 0
        self._data.contact = ()
        captured["states"] = self._declared_entity_states(registrations)
        return None

    from specola.adapters import mujoco as mujoco_module
    original = mujoco_module.MujocoAdapter._get_observation_impl
    mujoco_module.MujocoAdapter._get_observation_impl = fake_base
    try:
        type(adapter)._get_observation_impl(adapter)
    finally:
        mujoco_module.MujocoAdapter._get_observation_impl = original

    assert captured["states"][
        "object_relation:cup_1__to__plate_1"
    ].properties["declared_geom_contact"] == 1.0
    assert getattr(adapter, "_presented_contact_pairs", None) is None


def test_declared_contact_ignores_irrelevant_undeclared_geom():
    adapter = _relation_state_adapter()
    adapter._data.ncon = 1
    adapter._data.contact = (
        SimpleNamespace(geom1=0, geom2=3),
    )
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    states = adapter._declared_entity_states(registrations)

    assert states[
        "object_relation:cup_1__to__plate_1"
    ].properties["declared_geom_contact"] == 0.0


def test_object_region_relation_matches_installed_matrix_conventions():
    adapter = _relation_state_adapter()
    adapter._data.ncon = 1
    adapter._data.contact = (
        SimpleNamespace(geom1=0, geom2=2),
    )
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    properties = adapter._declared_entity_states(registrations)[
        "object_region_relation:cup_1__to__cabinet_1_drawer_region"
    ].properties

    assert properties == {
        "world_delta_x": pytest.approx(0.2),
        "world_delta_y": pytest.approx(0.2),
        "world_delta_z": pytest.approx(0.4),
        "installed_matrix_delta_x": pytest.approx(-0.2),
        "installed_matrix_delta_y": pytest.approx(0.2),
        "installed_matrix_delta_z": pytest.approx(0.4),
        "region_half_extent_x": pytest.approx(0.1),
        "region_half_extent_y": pytest.approx(0.2),
        "region_half_extent_z": pytest.approx(0.3),
        "installed_projected_half_extent_x": pytest.approx(0.2),
        "installed_projected_half_extent_y": pytest.approx(0.1),
        "installed_projected_half_extent_z": pytest.approx(0.3),
        "parent_present": 1.0,
        "source_parent_declared_geom_contact": 1.0,
    }


def test_projected_extent_uses_literal_matrix_times_size_not_abs_matrix():
    adapter = _relation_state_adapter()
    cosine = np.sqrt(0.5)
    installed_matrix = np.asarray(
        [
            [cosine, -cosine, 0.0],
            [cosine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    adapter._data.site_xmat[0] = installed_matrix
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    properties = adapter._declared_entity_states(registrations)[
        "object_region_relation:cup_1__to__cabinet_1_drawer_region"
    ].properties

    expected = np.abs(installed_matrix @ np.asarray([0.1, 0.2, 0.3]))
    conventional_aabb = np.abs(installed_matrix) @ np.asarray(
        [0.1, 0.2, 0.3]
    )
    assert properties["installed_projected_half_extent_x"] == pytest.approx(
        expected[0]
    )
    assert expected[0] != pytest.approx(conventional_aabb[0])


def test_parentless_region_exposes_absence_without_vacuous_contact_truth():
    adapter = _relation_state_adapter(with_parent=False)
    registrations = {
        spec.declaration_key: index
        for index, spec in enumerate(adapter._declare_task_entities(), 20)
    }

    properties = adapter._declared_entity_states(registrations)[
        "object_region_relation:cup_1__to__cabinet_1_drawer_region"
    ].properties

    assert properties["parent_present"] == 0.0
    assert properties["source_parent_declared_geom_contact"] == 0.0


def test_relation_metadata_has_stable_operand_and_pair_tags():
    adapter = _relation_state_adapter()

    metadata = adapter._declared_entity_property_metadata(
        "object_relation:cup_1__to__plate_1"
    )

    assert metadata["horizontal_center_distance"].note == "m"
    assert metadata["declared_geom_contact"].note == "1"
    expected_tags = {
        "source_declared_name:cup_1",
        "destination_declared_name:plate_1",
        "declared_pair:cup_1__to__plate_1",
    }
    assert expected_tags <= set(
        metadata["horizontal_center_distance"].tags
    )
    assert {
        "source_contact_geom:cup_collision",
        "destination_contact_geom:plate_collision",
    } <= set(metadata["horizontal_center_distance"].tags)


def test_contact_geom_provenance_uses_model_id_not_name_list_position():
    adapter = object.__new__(LIBEROAdapter)
    adapter._id2name = lambda kind, object_id: {
        ("geom", 3): "cup_collision",
        ("geom", 7): "plate_collision",
    }[(kind, object_id)]
    adapter._names = lambda kind: (
        "cup_collision",
        "plate_collision",
    )

    assert adapter._relation_contact_provenance_tags((3,), (7,)) == (
        "source_contact_geom:cup_collision",
        "destination_contact_geom:plate_collision",
    )


def test_snapshot_roles_are_interest_members_and_declared_complement():
    adapter = _snapshot_adapter(
        objects={
            "cup_1": _declaration("cup_1", root_body="cup_1_main"),
            "plate_1": _declaration("plate_1", root_body="plate_1_main"),
            "can_1": _declaration("can_1", root_body="can_1_main"),
        },
        body_names=(
            "world",
            "robot0_base",
            "table",
            "cup_1_main",
            "plate_1_main",
            "can_1_main",
            "cabinet_1_main",
        ),
        goal_state=[["On", "cup_1", "plate_1"]],
        obj_of_interest=["cup_1", "plate_1"],
    )

    assert adapter._declaration_snapshot().role_by_object_name == {
        "can_1": (
            "distractor",
            "structural_derivation",
            'parsed_problem["objects"] minus obj_of_interest',
        ),
        "cup_1": (
            "target",
            "structural_derivation",
            "obj_of_interest member; goal operands cross-checked",
        ),
        "plate_1": (
            "target",
            "structural_derivation",
            "obj_of_interest member; goal operands cross-checked",
        ),
    }


def test_snapshot_rejects_goal_operand_absent_from_obj_of_interest():
    adapter = _snapshot_adapter(
        goal_state=[["On", "cup_1", "cabinet_1_drawer_region"]],
        obj_of_interest=[],
    )

    with pytest.raises(RuntimeError, match="goal operand.*absent"):
        adapter._declaration_snapshot()


def test_snapshot_rejects_undeclared_object_of_interest():
    adapter = _snapshot_adapter(
        obj_of_interest=["cup_1", "missing_symbol"],
    )

    with pytest.raises(
        RuntimeError, match="obj_of_interest contains undeclared symbols"
    ):
        adapter._declaration_snapshot()


def test_snapshot_accounts_parsed_populations_with_source_mapped_arena_root():
    adapter = _snapshot_adapter()

    snapshot = adapter._declaration_snapshot()

    assert set(snapshot.parsed_object_names) == {
        name for name, _root in snapshot.objects
    }
    assert set(snapshot.parsed_fixture_names) == (
        {name for name, _root in snapshot.fixtures}
        | {record.name for record in snapshot.arena_fixtures}
    )
    assert "version-matched arena implementation" in (
        snapshot.arena_fixtures[0].reason
    )


def test_snapshot_rejects_unaccounted_parsed_objects_and_fixtures():
    adapter = _snapshot_adapter(
        parsed_objects={
            "cup": ["cup_1", "missing_object_1"],
        },
    )
    with pytest.raises(RuntimeError, match="object declaration mismatch"):
        adapter._declaration_snapshot()

    adapter = _snapshot_adapter(
        parsed_fixtures={
            "table": ["main_table"],
            "cabinet": ["cabinet_1", "missing_fixture_1"],
        },
    )
    with pytest.raises(RuntimeError, match="fixture declaration mismatch"):
        adapter._declaration_snapshot()


def test_body_rules_use_exact_declared_roots_without_name_fallback():
    adapter = _snapshot_adapter(
        body_names=(
            "world",
            "robot0_base",
            "table",
            "cup_1_main",
            "cabinet_1_main",
            "rogue_default_main",
        ),
        model_site_names=(
            "cabinet_1_drawer_region",
            "rogue_default_site",
        ),
        model_site_sizes=((0.1, 0.2, 0.3), (0.01, 0.01, 0.01)),
    )

    rules = adapter._default_body_rules()

    assert rules[1].matches("cabinet_1_main")
    assert rules[1].joint_scope == "subtree"
    assert rules[2].matches("cup_1_main")
    assert rules[2].joint_scope == "body"
    assert not any(rule.matches("rogue_default_main") for rule in rules)


def test_body_labels_use_declaration_names_not_compiled_root_names(
    monkeypatch,
):
    adapter = _snapshot_adapter()
    snapshot = adapter._declaration_snapshot()
    adapter._bodies = (
        SimpleNamespace(
            entity_id=1, body_name="cup_1_main", body_id=2
        ),
        SimpleNamespace(
            entity_id=2, body_name="robot0_base", body_id=1
        ),
    )
    adapter._model.body_parentid = np.asarray([0, 0, 0])
    adapter.entity_labels = {}

    def fake_base_labels(self):
        self.entity_labels.update(
            {
                1: "cup_1_main",
                2: "robot0_base",
            }
        )

    monkeypatch.setattr(
        MujocoAdapter,
        "_build_labels",
        fake_base_labels,
    )

    adapter._build_labels()

    assert snapshot.declared_name_by_root["cup_1_main"] == "cup_1"
    assert adapter.entity_labels[1] == "cup_1  (cup)"
    assert adapter.entity_labels[2] == "robot0_base"


def test_declared_root_missing_from_model_fails_loudly():
    adapter = _snapshot_adapter(
        body_names=("world", "robot0_base", "table", "cabinet_1_main"),
    )

    with pytest.raises(RuntimeError, match="cup_1.*resolved 0 times"):
        adapter._declaration_snapshot()


def test_two_declarations_cannot_share_one_root_body():
    adapter = _snapshot_adapter(
        objects={
            "cup_1": _declaration("cup_1", root_body="shared_main"),
        },
        fixtures={
            "cabinet_1": _declaration(
                "cabinet_1",
                root_body="shared_main",
            ),
        },
        body_names=("world", "robot0_base", "table", "shared_main"),
    )

    with pytest.raises(RuntimeError, match="duplicate roots"):
        adapter._declaration_snapshot()


def test_instantiated_region_must_be_declared_and_model_backed():
    adapter = _snapshot_adapter(
        parsed_regions={"some_other_region": {}},
    )

    with pytest.raises(RuntimeError, match="absent from parsed_problem"):
        adapter._declaration_snapshot()

    adapter = _snapshot_adapter(
        model_site_names=("some_other_site",),
        model_site_sizes=((0.1, 0.2, 0.3),),
    )
    with pytest.raises(RuntimeError, match="resolved 0 times"):
        adapter._declaration_snapshot()


def test_only_declared_box_sites_become_regions():
    sites = {
        "marker": _declaration(
            "marker",
            site_type="sphere",
            size=(0.1, 0.1, 0.1),
        )
    }
    adapter = _snapshot_adapter(
        sites=sites,
        parsed_regions={"marker": {}},
        model_site_names=("marker",),
        model_site_sizes=((0.1, 0.1, 0.1),),
    )

    with pytest.raises(RuntimeError, match="only box regions"):
        adapter._declaration_snapshot()


def test_region_preserves_zero_dimension_without_inventing_default():
    sites = {
        "flat_region": _declaration(
            "flat_region",
            site_type="box",
            size=(0.0, 0.2, 0.3),
        )
    }
    adapter = _snapshot_adapter(
        sites=sites,
        parsed_regions={"flat_region": {}},
        model_site_names=("flat_region",),
        model_site_sizes=((0.0, 0.2, 0.3),),
    )

    snapshot = adapter._declaration_snapshot()

    assert snapshot.regions[0].half_extents == (0.0, 0.2, 0.3)


def test_region_declaration_and_compiled_size_must_agree():
    adapter = _snapshot_adapter(
        model_site_sizes=((0.1, 0.2, 0.4),),
    )

    with pytest.raises(RuntimeError, match="declaration/model size mismatch"):
        adapter._declaration_snapshot()


def test_region_declaration_and_compiled_site_type_must_agree():
    adapter = _snapshot_adapter(
        model_site_types=("sphere",),
    )

    with pytest.raises(RuntimeError, match="compiled type='sphere'"):
        adapter._declaration_snapshot()


def test_libero_policy_has_no_criterion_or_name_heuristic_calls():
    tree = ast.parse(inspect.getsource(libero_adapter))
    banned_calls = {
        "_check_success",
        "check_success",
        "_eval_predicate",
        "check_ontop",
        "check_contain",
        "check_contact",
        "is_open",
        "is_close",
        "turn_on",
        "turn_off",
    }
    called = {
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert called.isdisjoint(banned_calls)
    assert not hasattr(libero_adapter, "_DEFAULT_REGION_HALF_EXTENTS")
    assert not hasattr(LIBEROAdapter, "_site_stem_index")


def test_declared_qpos_mismatch_raises_without_replacing_model_value(
    monkeypatch,
):
    adapter = object.__new__(LIBEROAdapter)
    monkeypatch.setattr(
        LIBEROAdapter,
        "_declared_joint_values",
        lambda self, entity_name: (
            ("cabinet_drawer", np.asarray([2.0])),
        ),
    )
    properties = {"qpos_cabinet_drawer": 1.0}

    with pytest.raises(RuntimeError, match="disagrees with MjData"):
        adapter._reconcile_declared_qpos(properties, "cabinet")

    assert properties["qpos_cabinet_drawer"] == 1.0


def test_declared_qpos_skips_only_what_the_extraction_scope_removed(
    monkeypatch,
):
    """A scoped-out joint has no extracted value to reconcile against.

    Found by running a real LIBERO episode under the X2 spec set: the
    task declares more fixture joints than any spec reads, and treating
    a scoped-out key as a materialization failure crashed every run.
    """
    adapter = object.__new__(LIBEROAdapter)
    monkeypatch.setattr(
        LIBEROAdapter,
        "_declared_joint_values",
        lambda self, entity_name: (
            ("cabinet_drawer", np.asarray([2.0])),
        ),
    )

    # Out of scope: never extracted, so there is nothing to assert.
    adapter._extraction_scope = {"region": frozenset({"qpos_elsewhere"})}
    adapter._reconcile_declared_qpos({}, "cabinet", "region")

    # In scope and still absent: exactly the failure this check exists
    # to catch, and a scope must not soften it.
    adapter._extraction_scope = {
        "region": frozenset({"qpos_cabinet_drawer"})
    }
    with pytest.raises(RuntimeError, match="did not materialize"):
        adapter._reconcile_declared_qpos({}, "cabinet", "region")

    # No class supplied: the strict reading stands, so a caller that
    # forgets the argument fails loudly instead of skipping silently.
    adapter._extraction_scope = {"region": frozenset()}
    with pytest.raises(RuntimeError, match="did not materialize"):
        adapter._reconcile_declared_qpos({}, "cabinet")


def test_declared_joint_identity_does_not_read_live_state():
    adapter = _snapshot_adapter()
    inner = adapter._libero_inner_env()
    inner.object_states_dict["cabinet_1"] = SimpleNamespace(
        get_joint_state=lambda: pytest.fail(
            "identity lookup must not read live state"
        )
    )
    original_name2id = adapter._name2id
    adapter._name2id = lambda kind, name: (
        7
        if kind == "joint" and name == "cabinet_1_drawer"
        else original_name2id(kind, name)
    )

    assert adapter._declared_joint_names(
        "cabinet_1_drawer_region"
    ) == ("cabinet_1_drawer",)
    assert adapter._declared_joint_ids(
        "cabinet_1_drawer_region"
    ) == (7,)


def test_object_owned_region_without_joints_is_not_an_articulation_owner():
    adapter = _snapshot_adapter(
        sites={
            "cup_1_contain_region": _declaration(
                "cup_1_contain_region",
                site_type="box",
                size=(0.1, 0.2, 0.3),
                parent_name="cup_1",
            )
        },
    )

    assert adapter._declared_joint_names(
        "cup_1_contain_region"
    ) == ()


def test_fixture_qpos_reconciles_after_shared_eager_materialization(
    monkeypatch,
):
    adapter = object.__new__(LIBEROAdapter)
    adapter._declaration_snapshot_value = SimpleNamespace(
        fixture_name_by_root={"cabinet_1_main": "cabinet_1"}
    )
    adapter._bodies = (
        SimpleNamespace(
            entity_id=4,
            body_name="cabinet_1_main",
            entity_class="region",
        ),
    )
    adapter._regions = ()
    perceived = {
        4: SimpleNamespace(properties={}),
    }

    def fake_shared_materialization(self, entities):
        entities[4].properties["qpos_cabinet_1_drawer"] = 0.25

    reconciled = []

    def fake_reconcile(self, properties, fixture_name, entity_class):
        reconciled.append(
            (
                fixture_name,
                properties["qpos_cabinet_1_drawer"],
                entity_class,
            )
        )

    monkeypatch.setattr(
        MujocoAdapter,
        "_materialize_curated_fields",
        fake_shared_materialization,
    )
    monkeypatch.setattr(
        LIBEROAdapter,
        "_reconcile_declared_qpos",
        fake_reconcile,
    )

    adapter._materialize_curated_fields(perceived)

    assert reconciled == [("cabinet_1", 0.25, "region")]


def test_perception_filtered_fixture_skips_reconciliation(monkeypatch):
    adapter = object.__new__(LIBEROAdapter)
    adapter._declaration_snapshot_value = SimpleNamespace(
        fixture_name_by_root={"cabinet_1_main": "cabinet_1"}
    )
    adapter._bodies = (
        SimpleNamespace(
            entity_id=4,
            body_name="cabinet_1_main",
        ),
    )
    adapter._regions = ()

    monkeypatch.setattr(
        MujocoAdapter,
        "_materialize_curated_fields",
        lambda self, entities: None,
    )
    reconciled = []
    monkeypatch.setattr(
        LIBEROAdapter,
        "_reconcile_declared_qpos",
        lambda self, properties, name: reconciled.append(name),
    )

    adapter._materialize_curated_fields({})

    assert reconciled == []
