"""Pre-refactor characterization of Isaac Lab's current monitor projection.

This is a legacy parity gate, not a statement of the desired deformable
contract. Future deformable properties may be asserted separately, but may not
weaken the frozen legacy entity/property/metadata denominator below.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from specola.adapters.isaaclab import IsaacLabAdapter, _token
from specola.adapters.isaaclab_extraction import ScalarPlanExtractor
from specola.session import Monitor


_LEGACY_ENTITY_IDS = frozenset((0, 1, 2))
_DEFORMABLE_ENTITY_ID = 1
_LEGACY_ENTITY_LABELS = {0: "robot", 1: "cloth", 2: "crate"}
_LEGACY_PROPERTY_KEYS = {
    0: frozenset((
        "body_pos_w_arm_link_0", "body_pos_w_arm_link_1", "body_pos_w_arm_link_2",
        "body_pos_w_tool_0", "body_pos_w_tool_1", "body_pos_w_tool_2",
        "joint_pos_elbow_joint", "joint_pos_shoulder_joint",
        "joint_vel_elbow_joint", "joint_vel_shoulder_joint",
        "link_wrench_arm_link_0", "link_wrench_arm_link_1", "link_wrench_arm_link_2",
        "link_wrench_arm_link_3", "link_wrench_arm_link_4", "link_wrench_arm_link_5",
        "link_wrench_tool_0", "link_wrench_tool_1", "link_wrench_tool_2",
        "link_wrench_tool_3", "link_wrench_tool_4", "link_wrench_tool_5",
    )),
    1: frozenset(),
    2: frozenset((
        "body_pos_w_crate_body_0", "body_pos_w_crate_body_1",
        "body_pos_w_crate_body_2",
    )),
}


class _FakeRootView:
    def __init__(self, wrench):
        self._wrench = np.asarray(wrench, dtype=np.float32)

    def get_link_incoming_joint_force(self):
        return self._wrench


class _FakeAsset:
    def __init__(self, data, joint_names=(), body_names=(), wrench=None):
        self.data = data
        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        if wrench is not None:
            self.root_view = _FakeRootView(wrench)


class _FakeScene:
    def __init__(self, assets, articulations=(), rigid_objects=(), deformable_objects=()):
        self._assets = dict(assets)
        self.articulations = {
            key: self._assets[key] for key in articulations
        }
        self.rigid_objects = {
            key: self._assets[key] for key in rigid_objects
        }
        self.deformable_objects = {
            key: self._assets[key] for key in deformable_objects
        }
        self.rigid_object_collections = {}
        self.sensors = {}
        self.surface_grippers = {}
        self.extras = {}

    def keys(self):
        return tuple(self._assets)

    def __getitem__(self, key):
        return self._assets[key]


class _FakeEnv:
    num_envs = 1
    step_dt = 0.25
    physics_dt = 0.01

    def __init__(self):
        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int64)
        articulation = _FakeAsset(
            SimpleNamespace(
                root_pos_w=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                root_lin_vel_w=np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
                joint_pos=np.array([[1.25, -2.5]], dtype=np.float32),
                joint_vel=np.array([[0.5, -1.0]], dtype=np.float32),
                body_pos_w=np.array(
                    [[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]],
                    dtype=np.float32,
                ),
            ),
            joint_names=("shoulder-joint", "elbow joint"),
            body_names=("arm/link", "tool"),
            wrench=np.array(
                [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]],
                dtype=np.float32,
            ),
        )
        nodal_positions = np.array(
            [[[28.0, 30.0, 32.0], [32.0, 34.0, 36.0]]], dtype=np.float32
        )
        nodal_velocities = np.array(
            [[[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]]], dtype=np.float32
        )
        deformable = _FakeAsset(
            SimpleNamespace(
                # These root fields are exactly the nodal means. root_vel_w
                # remains invisible because current extraction asks only for
                # root_lin_vel_w.
                root_pos_w=np.array([[30.0, 32.0, 34.0]], dtype=np.float32),
                root_vel_w=np.array([[3.0, 5.0, 7.0]], dtype=np.float32),
                nodal_pos_w=nodal_positions,
                nodal_vel_w=nodal_velocities,
                # Synthetic topology-like candidate: structurally accepted
                # extra `.data` is ignored; this is not a claimed PhysX API.
                representative_topology_candidate=np.array([[0, 1, 1, 0]]),
            ),
        )
        rigid = _FakeAsset(
            SimpleNamespace(
                root_pos_w=np.array([[20.0, 21.0, 22.0]], dtype=np.float32),
                root_lin_vel_w=np.array([[2.0, 2.5, 3.0]], dtype=np.float32),
                body_pos_w=np.array([[[23.0, 24.0, 25.0]]], dtype=np.float32),
            ),
            body_names=("crate body",),
        )
        # InteractiveScene-style order: terrain is skipped, then robot,
        # deformable, then rigid. The resulting monitor IDs are 0, 1, 2.
        self.scene = _FakeScene(
            {"terrain": None, "robot": articulation, "cloth": deformable, "crate": rigid},
            articulations=("robot",), rigid_objects=("crate",),
            deformable_objects=("cloth",),
        )


def _metadata_record(value):
    return {
        "group": value.group,
        "note": value.note,
        "tags": tuple(value.tags),
        "unit": value.unit,
        "family": value.family,
        "component": value.component,
        "frame": value.frame,
    }


def _legacy_projection(adapter, observation):
    """Normalize exactly the legacy surface, ignoring new deformable fields.

    Unknown entities and any new articulation/rigid fields remain failures.
    Only entity 1's post-refactor deformable properties, metadata, and tags
    and the raw-wrench fields (included since the 2026-08-20 freshness
    probe; filtered here to keep the frozen legacy surface comparable)
    are filtered.
    """

    def vector(value):
        return None if value is None else tuple(float(item) for item in value)

    if set(observation.perceived_entities) != _LEGACY_ENTITY_IDS:
        raise AssertionError("legacy entity IDs changed")
    labels = {int(key): str(value) for key, value in adapter.entity_labels.items()}
    if labels != _LEGACY_ENTITY_LABELS:
        raise AssertionError("legacy entity labels changed")

    entities = {}
    metadata = {}
    component_tags = {}
    for entity_id in sorted(_LEGACY_ENTITY_IDS):
        entity = observation.perceived_entities[entity_id]
        properties = {
            key: float(value) for key, value in sorted(entity.properties.items())
        }
        property_metadata = {
            key: _metadata_record(value)
            for key, value in sorted(adapter.property_metadata(entity_id).items())
        }
        tags = dict(sorted(observation.component_tags.get(entity_id, {}).items()))
        if entity_id == _DEFORMABLE_ENTITY_ID:
            properties = {
                key: value
                for key, value in properties.items()
                if key in _LEGACY_PROPERTY_KEYS[entity_id]
            }
            property_metadata = {
                key: value
                for key, value in property_metadata.items()
                if key in _LEGACY_PROPERTY_KEYS[entity_id]
            }
            tags = {
                key: value
                for key, value in tags.items()
                if key in _LEGACY_PROPERTY_KEYS[entity_id]
            }
        properties = {key: value for key, value in properties.items() if not key.startswith("link_wrench_")}
        property_metadata = {key: value for key, value in property_metadata.items() if not key.startswith("link_wrench_")}
        entities[entity_id] = {
            "entity_class": entity.entity_class,
            "position": vector(entity.position),
            "velocity": vector(entity.velocity),
            "track_age": entity.track_age,
            "track_state": entity.track_state.value,
            "properties": properties,
        }
        metadata[entity_id] = property_metadata
        if tags:
            component_tags[entity_id] = tags
    return {
        "timestep": observation.timestep,
        "time": float(observation.time),
        "entity_labels": labels,
        "entities": entities,
        "component_tags": component_tags,
        "property_metadata": metadata,
    }


def _meta(group, note):
    return {
        "group": group,
        "note": note,
        "tags": (),
        "unit": None,
        "family": None,
        "component": None,
        "frame": None,
    }


def _expected_legacy_projection(timestep):
    """Explicit golden values; derives only the known fixed-rate clock/age."""

    return {
        "timestep": timestep,
        "time": 0.25 * timestep,
        "entity_labels": _LEGACY_ENTITY_LABELS,
        "component_tags": {},
        "entities": {
            0: {
                "entity_class": "robot",
                "position": (1.0, 2.0, 3.0),
                "velocity": (0.5, 1.0, 1.5),
                "track_age": timestep,
                "track_state": "confirmed",
                "properties": {
                    "body_pos_w_arm_link_0": 10.0,
                    "body_pos_w_arm_link_1": 11.0,
                    "body_pos_w_arm_link_2": 12.0,
                    "body_pos_w_tool_0": 13.0,
                    "body_pos_w_tool_1": 14.0,
                    "body_pos_w_tool_2": 15.0,
                    "joint_pos_elbow_joint": -2.5,
                    "joint_pos_shoulder_joint": 1.25,
                    "joint_vel_elbow_joint": -1.0,
                    "joint_vel_shoulder_joint": 0.5,
                    "link_wrench_arm_link_0": 1.0,
                    "link_wrench_arm_link_1": 2.0,
                    "link_wrench_arm_link_2": 3.0,
                    "link_wrench_arm_link_3": 4.0,
                    "link_wrench_arm_link_4": 5.0,
                    "link_wrench_arm_link_5": 6.0,
                    "link_wrench_tool_0": 7.0,
                    "link_wrench_tool_1": 8.0,
                    "link_wrench_tool_2": 9.0,
                    "link_wrench_tool_3": 10.0,
                    "link_wrench_tool_4": 11.0,
                    "link_wrench_tool_5": 12.0,
                },
            },
            1: {
                "entity_class": "object",
                "position": (30.0, 32.0, 34.0),
                "velocity": None,
                "track_age": timestep,
                "track_state": "confirmed",
                "properties": {},
            },
            2: {
                "entity_class": "object",
                "position": (20.0, 21.0, 22.0),
                "velocity": (2.0, 2.5, 3.0),
                "track_age": timestep,
                "track_state": "confirmed",
                "properties": {
                    "body_pos_w_crate_body_0": 23.0,
                    "body_pos_w_crate_body_1": 24.0,
                    "body_pos_w_crate_body_2": 25.0,
                },
            },
        },
        "property_metadata": {
            0: {
                "body_pos_w_arm_link_0": _meta("body position", "m"),
                "body_pos_w_arm_link_1": _meta("body position", "m"),
                "body_pos_w_arm_link_2": _meta("body position", "m"),
                "body_pos_w_tool_0": _meta("body position", "m"),
                "body_pos_w_tool_1": _meta("body position", "m"),
                "body_pos_w_tool_2": _meta("body position", "m"),
                "joint_pos_elbow_joint": _meta("joint position", "rad or m"),
                "joint_pos_shoulder_joint": _meta("joint position", "rad or m"),
                "joint_vel_elbow_joint": _meta("joint velocity", "rad/s or m/s"),
                "joint_vel_shoulder_joint": _meta("joint velocity", "rad/s or m/s"),
                "link_wrench_arm_link_0": _meta("link forces", "N"),
                "link_wrench_arm_link_1": _meta("link forces", "N"),
                "link_wrench_arm_link_2": _meta("link forces", "N"),
                "link_wrench_arm_link_3": _meta("link torques", "N·m"),
                "link_wrench_arm_link_4": _meta("link torques", "N·m"),
                "link_wrench_arm_link_5": _meta("link torques", "N·m"),
                "link_wrench_tool_0": _meta("link forces", "N"),
                "link_wrench_tool_1": _meta("link forces", "N"),
                "link_wrench_tool_2": _meta("link forces", "N"),
                "link_wrench_tool_3": _meta("link torques", "N·m"),
                "link_wrench_tool_4": _meta("link torques", "N·m"),
                "link_wrench_tool_5": _meta("link torques", "N·m"),
            },
            1: {},
            2: {
                "body_pos_w_crate_body_0": _meta("body position", "m"),
                "body_pos_w_crate_body_1": _meta("body position", "m"),
                "body_pos_w_crate_body_2": _meta("body position", "m"),
            },
        },
    }


def test_legacy_projection_and_metadata_are_frozen_before_deformable_refactor():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    deformable_plan = next(
        entity for entity in adapter._extraction_plan.entities if entity.key == "cloth"
    )
    assert deformable_plan.profile == "deformable_legacy_incomplete"
    assert deformable_plan.fields == ()
    assert deformable_plan.position.source == "root_pos_w"

    raw_reset = adapter.reset()
    _legacy_projection(adapter, raw_reset)
    # The raw-wrench surface is included since the 2026-08-20
    # reset-freshness probe resolved issue #105 (the projection helper
    # filters it to keep the frozen legacy comparison surface intact).
    raw_properties = raw_reset.perceived_entities[0].properties
    assert raw_properties["link_wrench_arm_link_0"] == 1.0
    assert raw_properties["link_wrench_tool_5"] == 12.0
    assert any(key.startswith("link_wrench_") for key in adapter.property_metadata(0))
    robot_plan = next(entity for entity in adapter._extraction_plan.entities if entity.key == "robot")
    assert robot_plan.unresolved_semantic_candidates == ()
    wrench_fields = [field for field in robot_plan.fields if field.source == "get_link_incoming_joint_force"]
    assert wrench_fields
    assert all(field.exactness == "exact" for field in wrench_fields)
    assert all("wrench-reset-freshness-probe" in field.provenance for field in wrench_fields)
    env.episode_length_buf[0] += 1
    _legacy_projection(adapter, adapter.get_observation())


def test_scope_is_native_pruning_projection_and_keeps_structural_vectors():
    """Native pruning remains an exact projection of the full observation."""

    def vectorized_env():
        env = _FakeEnv()
        env.num_envs = 4
        env.episode_length_buf = np.zeros(env.num_envs, dtype=np.int64)
        for entity in ("robot", "cloth", "crate"):
            data = env.scene[entity].data
            for source, value in tuple(vars(data).items()):
                if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
                    setattr(data, source, np.repeat(value, env.num_envs, axis=0))
        return env

    accesses = []
    transfers = []

    class _Selection:
        def __init__(self, label, index, value):
            self.label = label
            self.index = index
            self.value = np.asarray(value)

    class _IndexedStorage:
        def __init__(self, label, value):
            self.label = label
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def __getitem__(self, index):
            accesses.append((self.label, repr(index)))
            return _Selection(self.label, index, self.value[index])

    class _IndexedSource:
        def __init__(self, label, value):
            self.torch = _IndexedStorage(label, value)
            # Match ProxyArray vector dtypes: logical Warp shape omits the
            # scalar component axis exposed by the zero-copy torch view.
            self.shape = (
                self.torch.shape[:-1]
                if label.rsplit(".", 1)[-1]
                in ("root_pos_w", "root_lin_vel_w", "root_vel_w", "body_pos_w")
                else self.torch.shape
            )

    def recording_to_numpy(value):
        if value is None:
            return None
        assert isinstance(value, _Selection)
        transfers.append((value.label, value.value.shape))
        return value.value

    def instrumented_adapter():
        env = vectorized_env()
        for entity in ("robot", "cloth", "crate"):
            data = env.scene[entity].data
            for source, value in tuple(vars(data).items()):
                if isinstance(value, np.ndarray):
                    label = "{0}.{1}".format(entity, source)
                    setattr(data, source, _IndexedSource(label, value))
        return IsaacLabAdapter(
            env,
            env_index=2,
            extractor_strategy=ScalarPlanExtractor(recording_to_numpy, _token),
        )

    full = instrumented_adapter()
    accesses.clear()
    transfers.clear()
    full_observation = full.reset()
    full_accesses = list(accesses)
    full_transfers = list(transfers)
    assert (len(full_accesses), len(full_transfers)) == (9, 9)
    assert sum(int(np.prod(shape)) for _, shape in full_transfers) == 28

    scoped = instrumented_adapter()
    scoped.set_extraction_scope(
        {
            "robot": frozenset({"joint_pos_shoulder_joint"}),
            "object": frozenset(),
        }
    )
    accesses.clear()
    transfers.clear()
    scoped_observation = scoped.reset()

    # robot: root position/velocity + joint_pos; cloth: root position; crate:
    # root position/velocity. No body or joint-velocity source is converted.
    expected_transfers = [
        ("robot.root_pos_w", (3,)),
        ("robot.root_lin_vel_w", (3,)),
        ("robot.joint_pos", (1,)),
        ("cloth.root_pos_w", (3,)),
        ("crate.root_pos_w", (3,)),
        ("crate.root_lin_vel_w", (3,)),
    ]
    assert transfers == expected_transfers
    assert (len(accesses), len(transfers)) == (6, 6)
    assert sum(int(np.prod(shape)) for _, shape in transfers) == 16
    assert [label for label, _ in accesses] == [label for label, _ in transfers]
    assert accesses[2] == ("robot.joint_pos", "(2, [0])")
    assert all(index.startswith("(2,") for _, index in accesses)
    assert (scoped_observation.timestep, scoped_observation.time) == (
        full_observation.timestep, full_observation.time
    )
    assert tuple(scoped_observation.perceived_entities) == tuple(
        full_observation.perceived_entities
    )
    assert scoped_observation.component_tags == full_observation.component_tags == {}
    for entity_id, full_entity in full_observation.perceived_entities.items():
        scoped_entity = scoped_observation.perceived_entities[entity_id]
        assert (
            scoped_entity.entity_id, scoped_entity.entity_class,
            scoped_entity.track_age, scoped_entity.track_state,
        ) == (
            full_entity.entity_id, full_entity.entity_class,
            full_entity.track_age, full_entity.track_state,
        )
        np.testing.assert_array_equal(scoped_entity.position, full_entity.position)
        if full_entity.velocity is None:
            assert scoped_entity.velocity is None
        else:
            np.testing.assert_array_equal(scoped_entity.velocity, full_entity.velocity)
        requested = {"joint_pos_shoulder_joint"} if full_entity.entity_class == "robot" else set()
        assert set(scoped_entity.properties) == requested
        scoped_metadata = scoped.property_metadata(entity_id)
        full_metadata = full.property_metadata(entity_id)
        assert set(scoped_metadata) == requested
        for key in requested:
            assert scoped_entity.properties[key] == pytest.approx(
                full_entity.properties[key]
            )
            assert scoped_metadata[key] == full_metadata[key]

    scoped.set_extraction_scope({
        "robot": frozenset({"joint_vel_elbow_joint", "body_pos_w_tool_2"}),
        "object": frozenset(),
    })
    accesses.clear()
    transfers.clear()
    component_observation = scoped.get_observation()
    assert tuple(component_observation.perceived_entities[0].properties) == (
        "joint_vel_elbow_joint",
        "body_pos_w_tool_2",
    )
    property_sources = {"robot.joint_vel", "robot.body_pos_w"}
    assert [item for item in accesses if item[0] in property_sources] == [
        ("robot.joint_vel", "(2, [1])"),
        ("robot.body_pos_w", "(2, [1], [2])"),
    ]
    assert [item for item in transfers if item[0] in property_sources] == [
        ("robot.joint_vel", (1,)),
        ("robot.body_pos_w", (1,)),
    ]


def test_scope_none_restores_full_output_and_runtime_shrink_updates_metadata():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    initial_keys = {
        "joint_pos_shoulder_joint", "joint_pos_elbow_joint", "body_pos_w_arm_link_0"
    }
    adapter.set_extraction_scope({"robot": frozenset(initial_keys), "object": frozenset()})
    first = adapter.reset()
    robot = first.perceived_entities[0]
    assert set(robot.properties) == initial_keys
    assert set(adapter.property_metadata(0)) == set(robot.properties)

    env.scene["robot"].data.joint_pos = env.scene["robot"].data.joint_pos[:, :1]
    env.scene["robot"].data.body_pos_w = env.scene["robot"].data.body_pos_w[:, :0, :]
    shrunk = adapter.get_observation().perceived_entities[0]
    assert set(shrunk.properties) == {"joint_pos_shoulder_joint"}
    assert set(adapter.property_metadata(0)) == set(shrunk.properties)

    adapter.set_extraction_scope(None)
    restored = adapter.get_observation().perceived_entities[0]
    assert set(restored.properties) == {
        "joint_pos_shoulder_joint",
        "joint_vel_elbow_joint",
        "joint_vel_shoulder_joint",
        "link_wrench_arm_link_0", "link_wrench_arm_link_1",
        "link_wrench_arm_link_2", "link_wrench_arm_link_3",
        "link_wrench_arm_link_4", "link_wrench_arm_link_5",
        "link_wrench_tool_0", "link_wrench_tool_1",
        "link_wrench_tool_2", "link_wrench_tool_3",
        "link_wrench_tool_4", "link_wrench_tool_5",
    }
    assert set(adapter.property_metadata(0)) == set(restored.properties)


def test_full_and_scoped_property_metadata_order_remains_plan_order():
    adapter = IsaacLabAdapter(_FakeEnv())
    full = adapter.reset().perceived_entities[0]
    full_keys = tuple(full.properties)
    assert full_keys[:4] == (
        "joint_pos_shoulder_joint",
        "joint_vel_shoulder_joint",
        "joint_pos_elbow_joint",
        "joint_vel_elbow_joint",
    )
    assert tuple(adapter.property_metadata(0)) == full_keys

    requested = frozenset(("joint_vel_shoulder_joint", "joint_pos_elbow_joint"))
    adapter.set_extraction_scope({"robot": requested})
    scoped = adapter.get_observation().perceived_entities[0]
    assert tuple(scoped.properties) == ("joint_vel_shoulder_joint", "joint_pos_elbow_joint")
    assert tuple(adapter.property_metadata(0)) == tuple(scoped.properties)


def test_failed_observation_cannot_change_metadata_for_last_accepted_state():
    env = _FakeEnv()
    adapter = IsaacLabAdapter(env)
    adapter.set_extraction_scope({"robot": frozenset(("joint_pos_shoulder_joint",))})
    accepted = adapter.reset()
    accepted_metadata = adapter.property_metadata(0)

    adapter.set_extraction_scope({"robot": frozenset(("joint_pos_elbow_joint",))})
    env.scene["robot"].data.joint_pos = env.scene["robot"].data.joint_pos[:, :1]
    env.scene["robot"].data.root_pos_w = np.zeros((1, 2), dtype=float)

    with pytest.raises(ValueError, match="position"):
        adapter.get_observation()

    assert adapter.last_observation is accepted
    assert adapter.property_metadata(0) == accepted_metadata


def test_one_scalar_strategy_keeps_adapter_scopes_plan_local():
    strategy = ScalarPlanExtractor(
        lambda value: np.asarray(value) if value is not None else None,
        _token,
    )
    shoulder = IsaacLabAdapter(_FakeEnv(), extractor_strategy=strategy)
    elbow = IsaacLabAdapter(_FakeEnv(), extractor_strategy=strategy)
    shoulder.set_extraction_scope({"robot": frozenset(("joint_pos_shoulder_joint",))})
    elbow.set_extraction_scope({"robot": frozenset(("joint_vel_elbow_joint",))})

    shoulder_properties = shoulder.reset().perceived_entities[0].properties
    elbow_properties = elbow.reset().perceived_entities[0].properties

    assert set(shoulder_properties) == {"joint_pos_shoulder_joint"}
    assert set(elbow_properties) == {"joint_vel_elbow_joint"}
    assert set(shoulder.property_metadata(0)) == set(shoulder_properties)
    assert set(elbow.property_metadata(0)) == set(elbow_properties)

    shoulder.set_extraction_scope({"robot": frozenset(("joint_pos_elbow_joint",))})
    assert set(shoulder.get_observation().perceived_entities[0].properties) == {
        "joint_pos_elbow_joint"
    }
    assert set(elbow.get_observation().perceived_entities[0].properties) == {
        "joint_vel_elbow_joint"
    }


def test_empty_scope_survives_reset_reuse_for_static_isaac_scene_plan():
    """Isaac's accepted scene-plan domain is static; reset reuses that plan."""
    adapter = IsaacLabAdapter(_FakeEnv())
    adapter.set_extraction_scope({})

    first = adapter.reset()
    second = adapter.reset()

    assert tuple(first.perceived_entities) == tuple(second.perceived_entities)
    for observation in (first, second):
        assert all(not entity.properties for entity in observation.perceived_entities.values())
        assert all(adapter.property_metadata(key) == {} for key in observation.perceived_entities)


def test_monitor_watch_clears_for_full_discovery_then_reinstalls_scope_twice(tmp_path):
    specs = tmp_path / "isaac-scope.yaml"
    specs.write_text("specs:\n- name: shoulder_floor\n  formula: \"G[0,1](robot_*[joint_pos_shoulder_joint] >= -100.0)\"\n", encoding="utf-8")
    adapter = IsaacLabAdapter(_FakeEnv())
    calls = []
    original = adapter.set_extraction_scope

    def record(scope):
        calls.append(scope)
        original(scope)

    adapter.set_extraction_scope = record
    monitor = Monitor(str(specs), results_dir=None)
    expected = {"robot": frozenset(("joint_pos_shoulder_joint",))}

    first = monitor.watch(adapter)
    assert calls[-2:] == [None, expected]
    assert len(adapter.last_observation.perceived_entities[0].properties) == 22
    first.step()
    assert set(adapter.last_observation.perceived_entities[0].properties) == {
        "joint_pos_shoulder_joint"
    }

    monitor.watch(adapter)
    assert calls[-2:] == [None, expected]
    assert len(adapter.last_observation.perceived_entities[0].properties) == 22


def test_scoped_extraction_never_invokes_unresolved_wrench_getter():
    env = _FakeEnv()
    calls = {"wrench": 0}

    class _NeverCallRootView:
        def get_link_incoming_joint_force(self):
            calls["wrench"] += 1
            raise AssertionError("unresolved wrench getter was invoked")

    for key in ("robot", "cloth", "crate"):
        env.scene[key].root_view = _NeverCallRootView()

    adapter = IsaacLabAdapter(env)
    adapter.set_extraction_scope({"robot": frozenset({"joint_pos_shoulder_joint"}), "object": frozenset()})
    observation = adapter.reset()

    assert calls["wrench"] == 0
    assert all(
        not key.startswith("link_wrench_")
        for entity in observation.perceived_entities.values()
        for key in entity.properties
    )
