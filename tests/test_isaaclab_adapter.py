"""Sim-free tests for the Isaac Lab adapter.

Isaac Sim is a 27 GB install, so these build stand-in assets shaped like
``Articulation`` rather than importing it -- the same trick the MuJoCo
adapter tests use with tiny XML models. What is pinned here is the adapter's
own logic: property/metadata parity, the vectorized-index selection, key
naming, and the group/unit conventions a selector depends on.

A live-simulator check lives in ``scripts/isaaclab_adapter_probe.py``; it
cannot run here because importing Isaac Lab requires a running simulation
app.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from specola.adapters.isaaclab import (
    IsaacLabAdapter,
    _bake_transform,
    _fan_triangulate,
    _token,
)
from specola.adapters.isaaclab_extraction import ScalarPlanExtractor
from specola.entities import TrackState


class _FakeData:
    """Stand-in for ``Articulation.data`` with ``(num_envs, ...)`` arrays."""

    def __init__(self, num_envs, n_joints, n_bodies):
        rng = np.random.default_rng(0)
        self.joint_pos = rng.standard_normal((num_envs, n_joints))
        self.joint_vel = rng.standard_normal((num_envs, n_joints))
        self.body_pos_w = rng.standard_normal((num_envs, n_bodies, 3))
        self.root_pos_w = rng.standard_normal((num_envs, 3))
        self.root_lin_vel_w = rng.standard_normal((num_envs, 3))


class _FakeRootView:
    def __init__(self, wrench):
        self._wrench = wrench

    def get_link_incoming_joint_force(self):
        return self._wrench


class _FakeArticulation:
    def __init__(self, num_envs, joint_names, body_names, with_wrench=True):
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.data = _FakeData(num_envs, len(joint_names), len(body_names))
        self.root_view = (
            _FakeRootView(
                np.random.default_rng(1).standard_normal(
                    (num_envs, len(body_names), 6)
                )
            )
            if with_wrench
            else None
        )


class _FakeScene:
    def __init__(self, assets):
        self._assets = assets
        self.articulations = {
            key: asset for key, asset in assets.items() if key not in {"terrain", "ground"}
        }
        self.rigid_objects = {}
        self.deformable_objects = {}
        self.rigid_object_collections = {}
        self.sensors = {}
        self.surface_grippers = {}
        self.extras = {}

    def keys(self):
        return list(self._assets)

    def __getitem__(self, key):
        return self._assets[key]


class _FakeEnv:
    def __init__(self, num_envs=2, with_wrench=True):
        self.num_envs = num_envs
        self.physics_dt = 1.0 / 120.0
        self.step_dt = 1.0 / 15.0
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int64)
        self.scene = _FakeScene(
            {
                "terrain": None,
                "robot": _FakeArticulation(
                    num_envs,
                    ["panda_joint1", "panda_joint2"],
                    ["panda_link0", "force_sensor"],
                    with_wrench=with_wrench,
                ),
                "held_asset": _FakeArticulation(
                    num_envs, [], ["forge_round_peg_8mm"], with_wrench=with_wrench
                ),
                "fixed_asset": _FakeArticulation(
                    num_envs, [], ["forge_hole_8mm"], with_wrench=with_wrench
                ),
            }
        )


def test_metadata_parity_across_every_entity():
    """Every exposed key must have metadata, and vice versa, per entity.

    This is the invariant that makes ``specola discover`` usable as the
    vocabulary: a key present in one and missing from the other leaves a
    spec author unable to see what they can bind.
    """
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env, env_index=0)
    adapter.reset()
    env.episode_length_buf[0] += 1
    observation = adapter.get_observation()

    assert observation.perceived_entities, "no entities were produced"
    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(
            entity.properties
        ), f"parity broken for entity {entity_id}"


def test_no_group_mixes_units():
    """A selector aggregates a whole group, so one group is one unit.

    Force and torque arrive in the same six-vector; keeping them in one
    group would let a min/max compare newtons against newton-metres.
    """
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env, env_index=0)
    adapter.reset()
    env.episode_length_buf[0] += 1
    observation = adapter.get_observation()

    units_by_group = {}
    for entity_id in observation.perceived_entities:
        for key, meta in adapter.property_metadata(entity_id).items():
            units_by_group.setdefault(meta.group, set()).add(meta.note)

    for group, units in units_by_group.items():
        assert len(units) == 1, f"group {group!r} mixes units {units}"

    assert units_by_group.get("link forces") == {"N"}
    assert units_by_group.get("link torques") == {"N\u00b7m"}


def test_env_index_selects_that_environment():
    """The adapter reports its own environment, not env 0 for everyone.

    Task randomisation in Isaac Lab is per environment, so an adapter that
    silently read env 0 would report one arbitrary sample of the population
    while appearing to work.
    """
    env = _FakeEnv(num_envs=2)
    first = IsaacLabAdapter(env, env_index=0).reset()
    second = IsaacLabAdapter(env, env_index=1).reset()

    robot_first = first.perceived_entities[0].properties
    robot_second = second.perceived_entities[0].properties

    assert set(robot_first) == set(robot_second)
    assert robot_first["joint_pos_panda_joint1"] != pytest.approx(
        robot_second["joint_pos_panda_joint1"]
    )

    raw = env.scene["robot"].data.joint_pos
    assert robot_second["joint_pos_panda_joint1"] == pytest.approx(raw[1][0])


def test_scalar_reference_maps_every_environment_row_to_native_values():
    """One scalar adapter per row must reproduce the row's native values.

    This is the pre-vectorized reference contract.  It uses the native fake
    arrays as the oracle rather than comparing two extraction paths that could
    share the same indexing bug.  A future batched strategy must match this
    row mapping exactly before any performance comparison is meaningful.
    """
    env = _FakeEnv(num_envs=4)
    robot = env.scene["robot"]
    data = robot.data

    for env_index in range(env.num_envs):
        observation = IsaacLabAdapter(env, env_index=env_index).reset()
        entity = observation.perceived_entities[0]

        np.testing.assert_allclose(entity.position, data.root_pos_w[env_index])
        np.testing.assert_allclose(
            entity.velocity, data.root_lin_vel_w[env_index]
        )
        for slot, name in enumerate(robot.joint_names):
            assert entity.properties["joint_pos_{0}".format(_token(name))] == pytest.approx(
                data.joint_pos[env_index, slot]
            )
            assert entity.properties["joint_vel_{0}".format(_token(name))] == pytest.approx(
                data.joint_vel[env_index, slot]
            )
        for slot, name in enumerate(robot.body_names):
            for component in range(3):
                key = "body_pos_w_{0}_{1}".format(_token(name), component)
                assert entity.properties[key] == pytest.approx(
                    data.body_pos_w[env_index, slot, component]
                )


def test_scalar_reference_refreshes_selected_row_after_partial_reset():
    """A per-environment reset must invalidate observation values, not plans.

    Isaac's static names/layout can stay cached across an episode, but native
    state rows change after a partial reset.  This test changes only the
    selected row, resets that adapter, and checks both fresh values and the
    reset timestep.  A future incremental cache must preserve this behavior.
    """
    env = _FakeEnv(num_envs=3)
    adapter = IsaacLabAdapter(env, env_index=1)
    first = adapter.reset()
    first_value = first.perceived_entities[0].properties["joint_pos_panda_joint1"]
    untouched_before = np.asarray(
        env.scene["robot"].data.joint_pos[[0, 2]]
    ).copy()

    env.scene["robot"].data.joint_pos[1, 0] = first_value + 100.0
    env.scene["robot"].data.root_pos_w[1] = np.array([41.0, 42.0, 43.0])
    # Rows 0 and 2 remain untouched: this models a partial vector-env reset.
    refreshed = adapter.reset()
    entity = refreshed.perceived_entities[0]

    assert refreshed.timestep == 0
    assert entity.properties["joint_pos_panda_joint1"] == pytest.approx(
        first_value + 100.0
    )
    np.testing.assert_allclose(entity.position, [41.0, 42.0, 43.0])
    assert entity.properties["joint_pos_panda_joint1"] != pytest.approx(first_value)
    np.testing.assert_allclose(
        env.scene["robot"].data.joint_pos[[0, 2]], untouched_before
    )


def test_out_of_range_env_index_is_rejected():
    with pytest.raises(ValueError, match="outside range"):
        IsaacLabAdapter(_FakeEnv(num_envs=2), env_index=2)


def test_missing_wrench_omits_only_wrench_keys():
    """An asset without a wrench source loses those keys, not all of them.

    ``get_link_incoming_joint_force`` is not available on every asset or
    backend. Absence must narrow the vocabulary rather than fabricate zeros,
    which would be indistinguishable from a genuine zero contact force.
    """
    adapter = IsaacLabAdapter(_FakeEnv(with_wrench=False), env_index=0)
    observation = adapter.reset()

    robot = observation.perceived_entities[0].properties
    assert not [k for k in robot if k.startswith("link_wrench_")]
    assert "joint_pos_panda_joint1" in robot
    assert "body_pos_w_panda_link0_0" in robot
    assert all(not entity.unresolved_semantic_candidates for entity in adapter._extraction_plan.entities)

    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(entity.properties)


def test_dt_is_the_environment_step_not_the_physics_step():
    """Observations arrive at the env rate; Isaac decimates physics inside it."""
    adapter = IsaacLabAdapter(_FakeEnv(), env_index=0)
    assert adapter.dt == pytest.approx(1.0 / 15.0)
    assert adapter.dt != pytest.approx(1.0 / 120.0)


def test_classes_and_ids_are_stable_across_steps():
    adapter = IsaacLabAdapter(_FakeEnv(), env_index=0)
    first = adapter.reset()
    second = adapter.get_observation()

    assert set(first.perceived_entities) == set(second.perceived_entities)
    for entity_id, entity in first.perceived_entities.items():
        assert (
            second.perceived_entities[entity_id].entity_class
            == entity.entity_class
        )
    assert adapter.supported_classes == frozenset({"robot", "object"})
    assert first.perceived_entities[0].entity_class == "robot"
    assert first.perceived_entities[0].track_state is TrackState.CONFIRMED


def test_non_articulated_scene_entries_are_skipped():
    """A terrain has no state to report; inventing one would be a fiction."""
    adapter = IsaacLabAdapter(_FakeEnv(), env_index=0)
    assert "terrain" not in adapter.entity_labels.values()
    assert set(adapter.entity_labels.values()) == {
        "robot",
        "held_asset",
        "fixed_asset",
    }


def test_sensor_entries_are_not_ingested_as_physical_assets():
    env = _FakeEnv()
    camera = _FakeCamera(env.num_envs)
    env.scene._assets["monitor_cam"] = camera
    env.scene.articulations.pop("monitor_cam", None)
    env.scene.sensors["monitor_cam"] = camera

    adapter = IsaacLabAdapter(env)
    assert "monitor_cam" not in adapter.entity_labels.values()
    assert adapter.camera_names() == ("monitor_cam",)


def test_documented_extra_entries_are_excluded_even_if_they_expose_data():
    env = _FakeEnv()
    extra = _FakeArticulation(env.num_envs, (), ("extra_body",))
    env.scene._assets["task_extra"] = extra
    env.scene.extras["task_extra"] = extra
    adapter = IsaacLabAdapter(env)
    assert "task_extra" not in adapter.entity_labels.values()
    assert ("task_extra", "extra_scene_entry") in adapter._extraction_plan.excluded_scene_keys


@pytest.mark.parametrize("family", ["rigid_object_collections", "surface_grippers"])
def test_unsupported_documented_scene_families_fail_loudly(family):
    env = _FakeEnv()
    env.scene._assets["unsupported"] = _FakeArticulation(
        env.num_envs, (), ("body",)
    )
    getattr(env.scene, family)["unsupported"] = env.scene._assets["unsupported"]
    env.scene.articulations.pop("unsupported", None)
    with pytest.raises(ValueError, match="unsupported"):
        IsaacLabAdapter(env)


def test_unclassified_scene_key_fails_loudly_instead_of_using_data_shape():
    env = _FakeEnv()
    env.scene._assets["mystery"] = _FakeArticulation(env.num_envs, (), ("body",))
    env.scene.articulations.pop("mystery", None)
    with pytest.raises(ValueError, match="unclassified"):
        IsaacLabAdapter(env)


def test_static_names_and_metadata_are_compiled_once_not_recomputed_per_step():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    initial_metadata = adapter.property_metadata(0)
    env.scene["robot"].joint_names[:] = ["changed_name", "changed_other"]
    env.scene["robot"].body_names[:] = ["changed_body", "changed_sensor"]

    observation = adapter.reset()
    assert set(adapter.property_metadata(0)) == set(initial_metadata)
    assert "joint_pos_panda_joint1" in observation.perceived_entities[0].properties
    assert "joint_pos_changed_name" not in observation.perceived_entities[0].properties


def test_declared_names_beyond_initial_array_shape_are_omitted():
    env = _FakeEnv()
    env.scene["robot"].joint_names.append("third_joint")
    env.scene["robot"].body_names.append("third_body")
    adapter = IsaacLabAdapter(env)
    observation = adapter.reset()
    properties = observation.perceived_entities[0].properties
    assert "joint_pos_third_joint" not in properties
    assert "body_pos_w_third_body_0" not in properties
    assert set(adapter.property_metadata(0)) == set(properties)


def test_runtime_array_shrink_and_wrench_failure_preserve_metadata_parity():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    robot = env.scene["robot"]
    robot.data.joint_pos = robot.data.joint_pos[:, :1]
    robot.data.joint_vel = robot.data.joint_vel[:, :1]
    robot.data.body_pos_w = robot.data.body_pos_w[:, :1, :]

    class _FailingRootView:
        def get_link_incoming_joint_force(self):
            raise RuntimeError("backend wrench unavailable")

    robot.root_view = _FailingRootView()
    observation = adapter.reset()
    properties = observation.perceived_entities[0].properties
    assert "joint_pos_panda_joint2" not in properties
    assert "body_pos_w_force_sensor_0" not in properties
    assert not [key for key in properties if key.startswith("link_wrench_")]
    assert set(adapter.property_metadata(0)) == set(properties)


def test_invalid_wrench_shape_omits_wrench_surface_and_metadata():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    env.scene["robot"].root_view = _FakeRootView(np.zeros((env.num_envs, 6)))
    properties = adapter.reset().perceived_entities[0].properties
    assert not [key for key in properties if key.startswith("link_wrench_")]
    assert set(adapter.property_metadata(0)) == set(properties)


def test_family_map_entries_not_present_in_scene_keys_fail_loudly():
    env = _FakeEnv()
    env.scene.rigid_objects["reverse_only"] = _FakeArticulation(env.num_envs, (), ("body",))
    with pytest.raises(ValueError, match="absent from scene.keys"):
        IsaacLabAdapter(env)


def test_documented_physical_family_precedes_static_spelling_exclusion():
    env = _FakeEnv()
    asset = _FakeArticulation(env.num_envs, (), ("ground_body",))
    env.scene._assets["ground"] = asset
    env.scene.articulations["ground"] = asset
    adapter = IsaacLabAdapter(env)
    assert "ground" in adapter._entity_id_of
    assert ("ground", "static_scene_entry") not in adapter._extraction_plan.excluded_scene_keys


def test_sensor_exclusion_ledger_and_semantic_records_are_explicit():
    env = _FakeEnv()
    camera = _FakeCamera(env.num_envs)
    env.scene._assets["monitor_cam"] = camera
    env.scene.sensors["monitor_cam"] = camera
    adapter = IsaacLabAdapter(env)
    assert ("monitor_cam", "unresolved_sensor_kind_not_materialized") in adapter._extraction_plan.unresolved_scene_keys
    entity = adapter._extraction_plan.entities[0]
    assert entity.position.owner == "asset_root"
    assert entity.position.unit == "m"
    assert entity.velocity.exactness == "legacy_unvalidated"
    assert entity.unresolved_semantic_candidates == ()
    wrench = next(
        field for field in entity.fields
        if field.source == "get_link_incoming_joint_force"
    )
    assert wrench.frame == "body1 at body1 origin"
    assert wrench.owner.startswith("incoming_joint/body:")
    assert "wrench-reset-freshness-probe" in wrench.provenance
    assert wrench.exactness == "exact"
    assert wrench.applicability == "applicable_when_view_getter_present"


def test_injected_scalar_strategy_is_used_without_changing_legacy_output():
    calls = {"compile": 0, "extract": 0}

    class RecordingStrategy:
        def __init__(self):
            self._delegate = ScalarPlanExtractor(
                lambda value: np.asarray(value) if value is not None else None,
                _token,
            )

        def compile_plan(self, scene, robot_class, object_class):
            calls["compile"] += 1
            return self._delegate.compile_plan(scene, robot_class, object_class)

        def extract(self, plan, env_index, timestep, time_seconds):
            calls["extract"] += 1
            return self._delegate.extract(plan, env_index, timestep, time_seconds)

        def metadata(self, plan, env_index, entity_id):
            return self._delegate.metadata(plan, env_index, entity_id)

    adapter = IsaacLabAdapter(_FakeEnv(), extractor_strategy=RecordingStrategy())
    full = adapter.reset()
    adapter.get_observation()
    assert calls == {"compile": 1, "extract": 2}

    requested = next(iter(full.perceived_entities[0].properties))
    adapter.set_extraction_scope({"robot": frozenset((requested,))})
    scoped = adapter.get_observation()
    assert set(scoped.perceived_entities[0].properties) == {requested}
    assert set(adapter.property_metadata(0)) == {requested}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("panda_joint1", "panda_joint1"),
        ("Panda-Link 0", "panda_link_0"),
        ("forge/round::peg", "forge_round_peg"),
        ("3rd_link", "n_3rd_link"),
    ],
)
def test_token_sanitisation(raw, expected):
    assert _token(raw) == expected


def test_empty_token_is_rejected():
    with pytest.raises(ValueError, match="empty token"):
        _token("///")


def test_scene_without_articulations_is_rejected():
    """Failing loudly beats reporting an empty scene as a healthy one."""

    class _Empty(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.scene = _FakeScene({"terrain": None, "ground": None})

    with pytest.raises(ValueError, match="no articulated assets"):
        IsaacLabAdapter(_Empty(), env_index=0)


def test_registry_factory_seeds_the_environment_config(monkeypatch):
    from specola.adapters import isaaclab as isaaclab_adapter

    made = []
    utils = ModuleType("isaaclab_tasks.utils")
    utils.parse_env_cfg = lambda task, num_envs: SimpleNamespace(seed=None, num_envs=num_envs)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", utils)
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(
        make=lambda task, cfg: made.append(cfg) or SimpleNamespace(reset=lambda: None)))
    monkeypatch.setattr(isaaclab_adapter, "_kit_app_running", lambda: True)
    monkeypatch.setattr(isaaclab_adapter, "IsaacLabAdapter", lambda env, env_index: env)

    isaaclab_adapter._make_isaaclab("Isaac-Forge-PegInsert-Direct-v0", None, False, seed=7)
    isaaclab_adapter._make_isaaclab("Isaac-Forge-PegInsert-Direct-v0", None, False)
    assert [cfg.seed for cfg in made] == [7, None]


torch = pytest.importorskip("torch")


class _DeviceProxyArray:
    """Models ``isaaclab.utils.warp.proxy_array.ProxyArray``.

    Two behaviours matter, both learned from the real object after a live run
    failed twice against guesses at its API:

    * the tensor is published as ``.torch``; there is no ``to_torch()``;
    * unknown attribute access **forwards to the tensor**, so ``.numpy``
      resolves to ``Tensor.numpy``, which raises on CUDA memory.

    The forwarding is what makes a duck-typed converter dangerous: probing
    for ``numpy`` finds a method that exists and then fails. A CPU tensor
    stands in for the device one -- what is being pinned is which accessor
    the adapter reaches for, not where the memory lives.
    """

    def __init__(self, array, device="cpu"):
        self._tensor = torch.as_tensor(np.asarray(array)).to(device)
        self.numpy_calls = 0

    @property
    def torch(self):
        return self._tensor

    def __getattr__(self, name):
        if name == "numpy":

            def _raise():
                self.numpy_calls += 1
                raise TypeError(
                    "can't convert cuda:0 device type tensor to numpy. "
                    "Use Tensor.cpu() to copy the tensor to host memory first."
                )

            return _raise
        raise AttributeError(name)


def test_device_backed_arrays_do_not_go_through_numpy():
    """Device arrays must be materialised via to_torch, not .numpy().

    Regression for the failure the first live run hit. Pins the ordering
    inside ``_to_numpy`` rather than merely that a value comes back, so the
    test fails if the two branches are ever swapped again.
    """
    from specola.adapters.isaaclab import _to_numpy

    proxy = _DeviceProxyArray([[1.0, 2.0], [3.0, 4.0]])
    result = _to_numpy(proxy)

    assert result is not None
    np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])
    assert proxy.numpy_calls == 0, (
        "_to_numpy called .numpy() on a device-backed array; that raises for "
        "real ProxyArrays and only works here because the stand-in is numpy"
    )


def test_adapter_reads_device_backed_scene_end_to_end():
    """The whole observation path survives device-backed arrays.

    Guards against a future extraction site reaching into ``.numpy()``
    directly and reintroducing the same failure one field at a time.
    """
    env = _FakeEnv()
    robot = env.scene["robot"]
    data = robot.data
    for field in ("joint_pos", "joint_vel", "body_pos_w", "root_pos_w",
                  "root_lin_vel_w"):
        setattr(data, field, _DeviceProxyArray(getattr(data, field)))
    robot.root_view = _FakeRootView(
        _DeviceProxyArray(np.zeros((env.num_envs, len(robot.body_names), 6)))
    )

    adapter = IsaacLabAdapter(env, env_index=1)
    adapter.reset()
    env.episode_length_buf[1] += 1
    observation = adapter.get_observation()

    props = observation.perceived_entities[0].properties
    assert "joint_pos_panda_joint1" in props
    assert "link_wrench_force_sensor_0" in props
    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(entity.properties)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable for live tensor test"
)
def test_adapter_reads_cuda_backed_scene_without_crossing_per_row():
    """CUDA input remains correctly attributed to the selected environment.

    The CPU proxy test above checks the published ProxyArray accessor.  This
    optional test exercises the actual host transfer path when CUDA exists;
    skipped result is evidence that CUDA is unavailable to this test process,
    not a pass for a GPU implementation.
    """
    env = _FakeEnv(num_envs=3)
    robot = env.scene["robot"]
    data = robot.data
    expected_joint = np.asarray(data.joint_pos[2], dtype=float)
    expected_root = np.asarray(data.root_pos_w[2], dtype=float)
    for field in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "root_pos_w",
        "root_lin_vel_w",
    ):
        setattr(data, field, _DeviceProxyArray(getattr(data, field), device="cuda"))

    adapter = IsaacLabAdapter(env, env_index=2)
    observation = adapter.reset()
    entity = observation.perceived_entities[0]
    assert entity.properties["joint_pos_panda_joint1"] == pytest.approx(
        expected_joint[0]
    )
    np.testing.assert_allclose(entity.position, expected_root)


class _FakeCamera:
    """A per-environment Camera: leading axis is the environment."""

    def __init__(self, num_envs, height=4, width=6, dtype=np.uint8):
        base = np.arange(num_envs * height * width * 3, dtype=np.float64)
        arr = base.reshape(num_envs, height, width, 3)
        if dtype == np.uint8:
            arr = (arr % 256).astype(np.uint8)
        else:
            arr = (arr / arr.max()).astype(dtype)
        self.data = type("D", (), {"output": {"rgb": arr}})()


def _env_with_camera(num_envs=4, **kw):
    env = _FakeEnv(num_envs=num_envs)
    camera = _FakeCamera(num_envs, **kw)
    env.scene._assets["monitor_cam"] = camera
    env.scene.sensors = {"monitor_cam": camera}
    return env


def test_scene_camera_is_attributed_per_environment():
    """A per-env Camera is what makes vectorized frames honest.

    Isaac Lab renders one camera per environment in a single vectorized
    pass and indexes the output by environment, so slicing env_index gives
    this environment's pixels rather than the whole grid the viewport shows.
    Pinning that the slice matches the right row is the point: an off-by-one
    here would mislabel a neighbour's scene, which is exactly the bug the
    viewport restriction exists to avoid.
    """
    env = _env_with_camera(num_envs=4)
    raw = env.scene.sensors["monitor_cam"].data.output["rgb"]

    for index in range(4):
        adapter = IsaacLabAdapter(env, env_index=index)
        assert adapter.camera_names() == ("monitor_cam",)
        frame = adapter.capture_frame()
        np.testing.assert_array_equal(frame, raw[index][:, :, :3])

    first = IsaacLabAdapter(env, env_index=0).capture_frame()
    second = IsaacLabAdapter(env, env_index=1).capture_frame()
    assert not np.array_equal(first, second), (
        "two environments returned the same frame; the slice is not per-env"
    )


def test_scene_camera_works_where_the_viewport_is_refused():
    """The camera lifts the num_envs > 1 restriction, the viewport does not."""
    env = _env_with_camera(num_envs=8)
    env.render_mode = "rgb_array"

    adapter = IsaacLabAdapter(env, env_index=5)
    assert "monitor_cam" in adapter.camera_names()
    assert "viewport" not in adapter.camera_names()
    assert adapter.capture_frame() is not None


def test_float_camera_output_is_scaled_to_uint8():
    env = _env_with_camera(num_envs=2, dtype=np.float32)
    frame = IsaacLabAdapter(env, env_index=1).capture_frame()
    assert frame.dtype == np.uint8
    assert 0 <= frame.min() and frame.max() <= 255


def test_camera_without_an_environment_axis_is_refused():
    """A 3-D output means the camera is not per-env; indexing it would lie."""
    env = _FakeEnv(num_envs=4)
    env.scene.sensors = {
        "bad_cam": type(
            "C", (), {"data": type("D", (), {"output": {"rgb": np.zeros((4, 6, 3))}})()}
        )()
    }
    env.scene._assets["bad_cam"] = env.scene.sensors["bad_cam"]
    adapter = IsaacLabAdapter(env, env_index=2)
    assert adapter.capture_frame() is None


def test_no_camera_is_offered_for_a_vectorized_scene():
    """A shared viewport must not be attributed to one environment.

    Isaac Lab lays parallel environments out in a grid behind a single
    viewport. Reporting a camera under num_envs > 1 would hand every monitor
    the same whole-grid image, which the dashboard would then file against
    each environment's timeline as that environment's camera. Mislabelled
    pixels are worse than no pixels.
    """
    env = _FakeEnv(num_envs=2)
    env.render_mode = "rgb_array"

    adapter = IsaacLabAdapter(env, env_index=1)
    assert adapter.camera_names() == ()
    assert adapter.capture_frame() is None


def test_camera_is_offered_for_a_single_environment():
    env = _FakeEnv(num_envs=1)
    env.render_mode = "rgb_array"
    env.render = lambda: np.full((4, 6, 3), 7, dtype=np.uint8)

    adapter = IsaacLabAdapter(env, env_index=0)
    assert adapter.camera_names() == ("viewport",)
    frame = adapter.capture_frame()
    assert frame is not None and frame.shape == (4, 6, 3)
    assert frame.dtype == np.uint8


def test_no_camera_without_rgb_array_render_mode():
    """An adapter cannot enable rendering, so it must not promise frames."""
    env = _FakeEnv(num_envs=1)  # render_mode unset
    adapter = IsaacLabAdapter(env, env_index=0)
    assert adapter.camera_names() == ()
    assert adapter.capture_frame() is None


def test_black_warmup_frames_are_pumped_not_returned():
    """Omniverse renders black until its pipeline warms up.

    Measured on the live scene: a frame taken after one step is uniformly
    zero while one after five is fully populated. Returning the zero frame
    would give the dashboard a black video indistinguishable from a dead
    camera, so the first capture pumps the renderer instead.
    """
    env = _FakeEnv(num_envs=1)
    env.render_mode = "rgb_array"

    calls = {"n": 0}

    def _render():
        calls["n"] += 1
        # Black for the first three calls, as the real pipeline is.
        if calls["n"] <= 3:
            return np.zeros((4, 6, 3), dtype=np.uint8)
        return np.full((4, 6, 3), 42, dtype=np.uint8)

    env.render = _render
    adapter = IsaacLabAdapter(env, env_index=0)

    frame = adapter.capture_frame()
    assert frame is not None and frame.any(), (
        "a black warm-up frame was returned instead of being pumped through"
    )
    assert calls["n"] >= 4

    # The retry runs once; a genuinely black scene is then reported honestly
    # rather than being retried forever.
    env.render = lambda: np.zeros((4, 6, 3), dtype=np.uint8)
    after = adapter.capture_frame()
    assert after is not None and not after.any()


# ---------------------------------------------------------------------------
# 3D pose panel: the sim-free pieces
# ---------------------------------------------------------------------------


def test_fan_triangulate_passes_triangles_through():
    tris = _fan_triangulate([3, 3], [0, 1, 2, 4, 5, 6])
    assert tris.dtype == np.uint32
    np.testing.assert_array_equal(tris, [[0, 1, 2], [4, 5, 6]])


def test_fan_triangulate_fans_polygons():
    tris = _fan_triangulate([5], [7, 8, 9, 10, 11])
    np.testing.assert_array_equal(tris, [[7, 8, 9], [7, 9, 10], [7, 10, 11]])


def test_fan_triangulate_drops_degenerate_faces_without_shifting_offsets():
    # The two-vertex face contributes no triangle, but its indices still
    # occupy the flat list: the quad after it must read from the right
    # offset.
    tris = _fan_triangulate([3, 2, 4], [0, 1, 2, 10, 11, 20, 21, 22, 23])
    np.testing.assert_array_equal(tris, [[0, 1, 2], [20, 21, 22], [20, 22, 23]])


def test_fan_triangulate_handles_empty_input():
    assert _fan_triangulate([], []).shape == (0, 3)


def test_bake_transform_uses_row_vector_convention():
    # Gf matrices keep translation in row 3 and treat points as row
    # vectors: p' = p @ M[:3, :3] + M[3, :3].
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    matrix[3, :3] = (5.0, 6.0, 7.0)
    baked, mirrored = _bake_transform(np.array([[1.0, 0.0, 0.0]]), matrix)
    assert baked.dtype == np.float32
    np.testing.assert_allclose(baked, [[5.0, 7.0, 7.0]])
    assert mirrored is False


def test_bake_transform_flags_mirroring():
    _, mirrored = _bake_transform(
        np.zeros((1, 3)), np.diag([-1.0, 1.0, 1.0, 1.0])
    )
    assert mirrored is True


def test_scene_hooks_are_none_without_a_stage():
    adapter = IsaacLabAdapter(_FakeEnv())
    for _ in range(2):
        assert adapter.scene_geometry() is None
        assert adapter.scene_poses() is None


def test_scene_hooks_stay_none_for_the_adapter_lifetime():
    # Viability is decided once. A stage appearing later must not flip
    # the hooks from None to arrays mid-episode -- the recording joins
    # pose records to data lines by ordinal.
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    assert adapter.scene_poses() is None
    env.scene.stage = object()
    assert adapter.scene_poses() is None
    assert adapter.scene_geometry() is None


def test_scene_hooks_are_none_without_quaternions():
    # _FakeData carries body_pos_w but no body_quat_w: the plan must
    # refuse on the pose fields alone, before touching any USD API (no
    # pxr is installed in this environment).
    env = _FakeEnv()
    env.scene.stage = object()
    adapter = IsaacLabAdapter(env)
    assert adapter.scene_poses() is None
    assert adapter.scene_geometry() is None


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class _SteppingEnv(_FakeEnv):
    """``_FakeEnv`` with Isaac Lab's step contract: a torch action tensor
    of shape ``(num_envs, action_dim)``, gymnasium's five-tuple back."""

    def __init__(self, num_envs=2, low=(-1.0, -1.0), high=(1.0, 1.0),
                 terminate_at=None):
        super().__init__(num_envs=num_envs)
        self.single_action_space = _Box(low, high)
        self.device = "cpu"
        self.actions = []
        self._terminate_at = terminate_at

    def step(self, action):
        torch = pytest.importorskip("torch")
        assert isinstance(action, torch.Tensor), type(action)
        self.actions.append(action.clone())
        n = len(self.actions)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._terminate_at is not None and n >= self._terminate_at:
            terminated[:] = True
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return None, None, terminated, truncated, {}


def test_hold_advancer_steps_the_env_with_zero_actions():
    pytest.importorskip("torch")
    env = _SteppingEnv()
    advance = IsaacLabAdapter(env).make_advancer("hold", None, 0)
    assert advance(0) and advance(1)
    assert len(env.actions) == 2
    assert tuple(env.actions[0].shape) == (2, 2)
    assert not env.actions[0].any()


def test_random_advancer_samples_the_single_action_space():
    pytest.importorskip("torch")
    env = _SteppingEnv(low=(0.5, -2.0), high=(1.0, -1.0))
    advance = IsaacLabAdapter(env).make_advancer("random", None, 3)
    advance(0)
    a = env.actions[0].numpy()
    assert a.shape == (2, 2)
    assert (a >= [0.5, -2.0]).all() and (a <= [1.0, -1.0]).all()
    assert not np.array_equal(a[0], a[1])


def test_advancer_stops_when_the_monitored_env_terminates():
    pytest.importorskip("torch")
    env = _SteppingEnv(terminate_at=2)
    advance = IsaacLabAdapter(env).make_advancer("hold", None, 0)
    assert advance(0) and advance(1)
    assert not advance(2)
    assert len(env.actions) == 2


def test_random_advancer_refuses_an_unbounded_action_space():
    pytest.importorskip("torch")
    env = _SteppingEnv(low=(-np.inf, -1.0), high=(np.inf, 1.0))
    with pytest.raises(SystemExit, match="bounded action space"):
        IsaacLabAdapter(env).make_advancer("random", None, 0)


@pytest.mark.parametrize("mode, demo", [("replay", None), ("hold", "x.hdf5")])
def test_replay_and_demo_are_refused(mode, demo):
    with pytest.raises(SystemExit, match="replay"):
        IsaacLabAdapter(_SteppingEnv()).make_advancer(mode, demo, 0)


# ---------------------------------------------------------------------------
# State tier (replay)
# ---------------------------------------------------------------------------


class _StateScene(_FakeScene):
    """``_FakeScene`` with Isaac Lab's ``get_state``/``reset_to`` over nested tensors."""

    def __init__(self, assets, num_envs, calls):
        super().__init__(assets)
        self.num_envs = num_envs
        self.calls = calls
        rng = np.random.default_rng(2)

        def leaf(width):
            return torch.as_tensor(rng.standard_normal((num_envs, width)).astype(np.float32))

        self.state = {
            "articulation": {
                key: {"root_pose": leaf(7), "root_velocity": leaf(6),
                      "joint_position": leaf(len(asset.joint_names)),
                      "joint_velocity": leaf(len(asset.joint_names))}
                for key, asset in self.articulations.items() if asset is not None
            },
            "deformable_object": {},
            "rigid_object": {},
            "gripper": {},
        }

    def get_state(self, is_relative=False):
        assert is_relative is False
        return {kind: {asset: ({key: value.clone() for key, value in leaves.items()}
                               if isinstance(leaves, dict) else leaves.clone())
                       for asset, leaves in assets.items()}
                for kind, assets in self.state.items()}

    def reset_to(self, state, env_ids=None, is_relative=False):
        self.calls.append(("reset_to", list(env_ids), is_relative))
        for kind, assets in state.items():
            for asset, leaves in assets.items():
                items = leaves.items() if isinstance(leaves, dict) else [("", leaves)]
                for key, value in items:
                    target = self.state[kind][asset][key] if key else self.state[kind][asset]
                    assert value.dtype == target.dtype
                    assert value.shape == (len(env_ids), *target.shape[1:])
                    target[env_ids] = value

    def update(self, dt):
        self.calls.append(("update", dt))


class _StateEnv(_FakeEnv):
    def __init__(self, num_envs):
        super().__init__(num_envs=num_envs)
        self.calls = []
        self.scene = _StateScene(self.scene._assets, num_envs, self.calls)
        self.cfg = SimpleNamespace(to_dict=lambda: {"seed": 7, "sim": {"dt": 1 / 120}})
        self.sim = SimpleNamespace(forward=lambda: self.calls.append("forward"))
        self.spec = SimpleNamespace(id="Isaac-Fake-v0")


def test_state_round_trips_through_reset_to_for_the_monitored_environment():
    env = _StateEnv(num_envs=3)
    adapter = IsaacLabAdapter(env, env_index=1)
    signature = adapter.state_signature()
    captured = adapter.capture_state()
    assert signature["engine"] == "isaaclab" and signature["task"] == "Isaac-Fake-v0"
    assert signature["length"] == captured.shape[0] == 3 * (7 + 6) + 2 * 2
    assert signature["layout"][:2] == [
        ["robot", "articulation", "root_pose", [7], "float32"],
        ["robot", "articulation", "root_velocity", [6], "float32"],
    ]
    before = adapter._env.scene.get_state()

    for leaves in env.scene.state["articulation"].values():
        for value in leaves.values():
            value[1] += 1.0
    assert not np.array_equal(adapter.capture_state(), captured)
    adapter.restore_state(captured)

    # One reset_to for the monitored environment, then a forward pass and a
    # buffer update; no physics step exists on the fake to be called.
    assert env.calls == [("reset_to", [1], False), "forward", ("update", 0.0)]
    np.testing.assert_array_equal(adapter.capture_state(), captured)
    for asset, leaves in before["articulation"].items():
        for key, value in leaves.items():
            assert torch.equal(env.scene.state["articulation"][asset][key], value)
    with pytest.raises(ValueError, match="length"):
        adapter.restore_state(captured[:-1])


def test_surface_grippers_refuse_the_state_tier_beyond_one_environment():
    for num_envs in (1, 2):
        env = _StateEnv(num_envs=num_envs)
        adapter = IsaacLabAdapter(env, env_index=0)
        # The extraction plan refuses surface grippers at construction
        # (isaaclab_extraction.py), so one is added afterwards to reach the
        # state tier's own rule, which the shared helper also gives robodojo.
        env.scene.surface_grippers = {"suction": object()}
        env.scene.state["gripper"]["suction"] = torch.full((num_envs,), -1, dtype=torch.int32)
        if num_envs == 1:
            signature = adapter.state_signature()
            assert ["suction", "gripper", "", [], "int32"] in signature["layout"]
            adapter.restore_state(adapter.capture_state())
            assert env.calls[0] == ("reset_to", [0], False)
            continue
        with pytest.warns(RuntimeWarning, match="surface grippers and 2 environments"):
            assert adapter.state_signature() is None
        assert adapter.capture_state() is None
        with pytest.raises(NotImplementedError, match="surface grippers"):
            adapter.restore_state(np.zeros(1))
        assert env.calls == []
