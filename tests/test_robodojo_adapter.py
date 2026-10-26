"""Sim-free tests for the RoboDojo adapter.

RoboDojo needs a 27 GB Isaac Sim install plus its downloaded asset tree, so
these build stand-ins shaped like its managers rather than importing it --
the same trick the Isaac Lab and MuJoCo adapter tests use. What is pinned
here is the adapter's own logic: property/metadata parity, key naming and
group units, per-environment selection, the coupled-robot and
rigid/dynamic-alias de-duplication, dynamic scene population, and the
camera surface.

Anything that needs a live scene -- whether a captured frame looks right,
whether the 3D panel would be viable -- is not testable here and is called
out in the adapter's module docstring instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from specola.adapters.robodojo import RoboDojoAdapter, _handles, _token
from specola.entities import TrackState


class _DofType:
    """Stand-in for ``omni.physics.tensors.DofType``."""

    def __init__(self, name):
        self.name = name


class _SharedMetatype:
    def __init__(self, dof_types):
        self.dof_types = dof_types


class _RootPhysxView:
    def __init__(self, dof_types):
        self.shared_metatype = _SharedMetatype(dof_types)


class _ArticulationData:
    """Stand-in for Isaac Lab's ``Articulation.data``."""

    def __init__(self, num_envs, n_joints, n_bodies, seed=0, wrench=True):
        rng = np.random.default_rng(seed)
        self.joint_pos = rng.standard_normal((num_envs, n_joints))
        self.joint_vel = rng.standard_normal((num_envs, n_joints))
        self.body_pos_w = rng.standard_normal((num_envs, n_bodies, 3))
        self.body_quat_w = rng.standard_normal((num_envs, n_bodies, 4))
        self.root_pos_w = rng.standard_normal((num_envs, 3))
        self.root_lin_vel_w = rng.standard_normal((num_envs, 3))
        self.body_link_vel_w = rng.standard_normal((num_envs, n_bodies, 6))
        if wrench:
            self.body_incoming_joint_wrench_b = rng.standard_normal(
                (num_envs, n_bodies, 6)
            )


class _RobotAsset:
    """Stand-in for the Isaac Lab ``Articulation`` behind a RoboDojo robot."""

    def __init__(
        self,
        num_envs,
        joint_names,
        body_names,
        dof_types=None,
        wrench=True,
        seed=0,
    ):
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.data = _ArticulationData(
            num_envs, len(joint_names), len(body_names), seed=seed, wrench=wrench
        )
        if dof_types is not None:
            self.root_physx_view = _RootPhysxView(
                [_DofType(kind) for kind in dof_types]
            )


class _Robot:
    """Stand-in for a RobotManager robot descriptor."""

    def __init__(
        self,
        arm_name,
        arm_joints_name,
        gripper_joints_name,
        ee_link_name,
        base_link,
    ):
        self.arm_name = arm_name
        self.gripper_name = arm_name.replace("arm", "ee")
        self.arm_joints_name = list(arm_joints_name)
        self.gripper_joints_name = list(gripper_joints_name)
        self.ee_link_name = ee_link_name
        self.base_link = base_link


class _RobotManager:
    def __init__(self, robot_list, robot_key):
        self.robot_list = list(robot_list)
        self.robot_key = list(robot_key)


class _RigidObject:
    def __init__(self, position, orientation, linear, angular):
        self._pose = (np.asarray(position), np.asarray(orientation))
        self._linear = np.asarray(linear)
        self._angular = np.asarray(angular)

    def get_world_pose(self):
        return self._pose

    def get_linear_velocity(self):
        return self._linear

    def get_angular_velocity(self):
        return self._angular


class _ArticulationObject(_RigidObject):
    def __init__(self, position, orientation, dof_names, joint_pos, joint_vel):
        super().__init__(position, orientation, np.zeros(3), np.zeros(3))
        self.dof_names = list(dof_names)
        self._joint_pos = np.asarray(joint_pos)
        self._joint_vel = np.asarray(joint_vel)

    def get_joint_positions(self):
        return self._joint_pos

    def get_joint_velocities(self):
        return self._joint_vel


class _PoseOnlyObject:
    """Garment / geometry prims: a root pose, no velocity."""

    def __init__(self, root_pose):
        self._root_pose = np.asarray(root_pose, dtype=float)

    def get_state(self, is_relative=False):
        assert is_relative is False, "world-frame poses only"
        return {"root_pose": self._root_pose}


class _FluidObject:
    def __init__(self, particles):
        self._particles = np.asarray(particles, dtype=float)

    def get_particle_positions(self):
        return self._particles, None, None


class _LayoutManager:
    def __init__(self, records, instance_types=None, metadata=None):
        self._records = records
        # RoboDojo's own declared-type table: instance name -> layout type.
        self.instance_type_by_env = [dict(instance_types or {}), {}]
        # Per-instance asset metadata (oriented bbox and the like), as
        # RoboDojo's get_instance_metadata answers it; None when absent.
        self._metadata = dict(metadata or {})

    def get_layout_records(self, env_idx, object_type):
        return self._records.get(object_type, [])

    def get_instance_metadata(self, env_idx, inst_name=None, label=None):
        return self._metadata.get(inst_name)


class _SceneManager:
    def __init__(self, collections, layout_manager: "_LayoutManager | None" = None):
        # {collection: {instance name: object}}
        self.collections = collections
        self.layout_manager = layout_manager

    def get_objects(self, env_ids=None, object_name=None, object_type=None):
        env_id = (env_ids or [0])[0]
        # Mirrors SceneManager: "dynamic" and "rigid" answer with the SAME
        # dictionary, which is exactly the double-report trap the adapter
        # has to avoid.
        if object_type == "dynamic":
            object_type = "rigid"
        objects = self.collections.get(object_type, {})
        return {
            f"env{env_id}_{object_type}_{name}": obj
            for name, obj in objects.items()
        }


class _CaptureManager:
    def __init__(self, annotator_type, frames):
        self.annotator_type = annotator_type
        self._frames = frames
        self.calls = []

    def step(self, env_ids=None, cam_ids=None):
        self.calls.append((tuple(env_ids or ()), tuple(cam_ids or ())))
        cam_id = (cam_ids or [0])[0]
        env_id = (env_ids or [0])[0]
        frame = self._frames[cam_id][env_id]
        return [{"rgb": [{"data": frame, "info": None}]}]


class _CameraManager:
    def __init__(self, camera_names):
        self.camera_names = camera_names


class _Sim:
    def __init__(self, step_dt):
        self.step_dt = step_dt
        self.physics_dt = 0.004


class _Env:
    """Stand-in for a RoboDojo ``TaskEnv`` after its first reset."""

    def __init__(
        self,
        num_envs=2,
        robot_manager=None,
        scene_manager=None,
        camera_manager=None,
        capture_manager=None,
    ):
        self.num_envs = num_envs
        self.sim = _Sim(0.012)
        self.dt = 0.004
        self.renders = 0
        self.robot_manager = robot_manager or _default_robot_manager(num_envs)
        self.scene_manager = scene_manager or _default_scene_manager()
        self.camera_manager = camera_manager
        self.capture_manager = capture_manager
        # Present so a test can replace them with a surface that explodes
        # when touched; the adapter must never read either.
        self.reward_manager: object = None
        self.obs_manager: object = None

        self.sim_steps = 0

    def render(self):
        self.renders += 1

    def sim_step(self, render=True):
        assert render is False, "monitoring must not drive the renderer"
        self.sim_steps += 1


def _default_robot_manager(num_envs=2, coupled=False):
    asset = _RobotAsset(
        num_envs,
        ["joint1", "joint2", "finger_joint1"],
        ["base_link", "link1", "gripper_link"],
        dof_types=["Rotation", "Rotation", "Translation"],
    )
    left = _Robot(
        "left_arm", ["joint1", "joint2"], ["finger_joint1"], "gripper_link", "base_link"
    )
    if coupled:
        right = _Robot(
            "right_arm",
            ["joint1", "joint2"],
            ["finger_joint1"],
            "gripper_link",
            "base_link",
        )
        # A coupled pair is ONE Isaac Lab articulation behind two entries.
        return _RobotManager([left, right], [asset, asset])
    return _RobotManager([left], [asset])


def _default_scene_manager():
    return _SceneManager(
        {
            "rigid": {
                "bowl_0": _RigidObject(
                    [0.1, 0.2, 0.8], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.0, 0.0, 0.2]
                )
            },
            "articulation": {
                "drawer_0": _ArticulationObject(
                    [0.3, 0.0, 0.7],
                    [1.0, 0.0, 0.0, 0.0],
                    ["slide_joint"],
                    [0.05],
                    [0.01],
                )
            },
            "geometry": {
                "marker_0": _PoseOnlyObject([0.0, 0.1, 0.75, 1.0, 0.0, 0.0, 0.0])
            },
            "garment": {
                "cloth_0": _PoseOnlyObject([0.2, 0.1, 0.76, 1.0, 0.0, 0.0, 0.0])
            },
            "fluid": {
                "water_0": _FluidObject([[0.0, 0.0, 0.8], [0.2, 0.0, 0.8]])
            },
        },
        layout_manager=_LayoutManager(
            {
                "Rigid": [{"inst_name": "bowl_0", "label": "bowl0"}],
                "Articulation": [{"inst_name": "drawer_0", "label": None}],
            }
        ),
    )


def _declare_type(scene, name, layout_type):
    """Declare one instance's layout type, the way a task YAML would."""
    layout = scene.layout_manager
    assert layout is not None
    layout.instance_type_by_env[0][name] = layout_type


def _entity_of(adapter, observation, label):
    for entity_id, entity in observation.perceived_entities.items():
        if adapter.entity_labels.get(entity_id) == label:
            return entity_id, entity
    raise AssertionError(f"no entity labelled {label!r}")


def test_metadata_parity_across_every_entity():
    """Every exposed key has metadata and vice versa, for every entity class.

    ``specola discover`` output is the whole vocabulary a spec author works
    from, so a key present in one and missing from the other leaves them
    unable to see what they can bind.
    """
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()

    classes = {
        entity.entity_class for entity in observation.perceived_entities.values()
    }
    assert classes == {"robot", "rigid", "articulation", "geometry", "garment", "fluid"}

    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(
            entity.properties
        ), f"parity broken for entity {entity_id}"


def test_bounding_box_follows_the_pose_and_the_asset_metadata():
    """A posed object with RoboDojo bbox metadata reports its world-axis
    box at the current pose; an object without metadata reports no box.

    RoboDojo's container goals (``is_A_in_B`` and friends) are box tests
    against per-asset sizes, which is why a pose alone cannot restate them.
    """
    scene = _default_scene_manager()
    # Yaw of 90 degrees about z: the box's x extent lands on world y.
    scene.collections["rigid"]["bowl_0"] = _RigidObject(
        [0.1, 0.2, 0.8], [0.70710678, 0.0, 0.0, 0.70710678], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    half = (0.1, 0.05, 0.02)
    corners = [
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    scene.layout_manager._metadata["bowl_0"] = {"geometry": {"oriented_bbox": {"vertices": corners}}}
    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    observation = adapter.reset()

    _, bowl = _entity_of(adapter, observation, "bowl0")
    assert bowl.properties["aabb_min_0"] == pytest.approx(0.1 - 0.05)
    assert bowl.properties["aabb_max_0"] == pytest.approx(0.1 + 0.05)
    assert bowl.properties["aabb_min_1"] == pytest.approx(0.2 - 0.1)
    assert bowl.properties["aabb_max_1"] == pytest.approx(0.2 + 0.1)
    assert bowl.properties["aabb_min_2"] == pytest.approx(0.8 - 0.02)
    assert bowl.properties["aabb_max_2"] == pytest.approx(0.8 + 0.02)
    meta = adapter.property_metadata(_entity_of(adapter, observation, "bowl0")[0])
    assert meta["aabb_min_0"].group == "bounding box"

    # The geometry marker has a pose but no metadata: no box keys at all.
    _, marker = _entity_of(adapter, observation, "marker_0")
    assert not [k for k in marker.properties if k.startswith("aabb_")]


def test_every_reported_class_is_declared_supported():
    """A closed supported_classes must cover everything a reset can emit."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()

    for entity in observation.perceived_entities.values():
        assert entity.entity_class in adapter.supported_classes


def test_no_group_mixes_units():
    """A selector aggregates a whole group, so one group is one unit."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()

    units_by_group = {}
    for entity_id in observation.perceived_entities:
        for meta in adapter.property_metadata(entity_id).values():
            units_by_group.setdefault(meta.group, set()).add(meta.note)

    mixed = {
        group: units for group, units in units_by_group.items() if len(units) > 1
    }
    # Body orientation carries a component label per axis rather than a
    # unit; a quaternion is dimensionless, so this is not a unit mix.
    assert set(mixed) <= {"body orientation"}
    assert units_by_group["link forces"] == {"N"}
    assert units_by_group["link torques"] == {"N·m"}
    assert units_by_group["joint angle"] == {"rad"}
    assert units_by_group["joint displacement"] == {"m"}


def test_joint_groups_follow_the_engine_dof_types():
    """Revolute joints are angles, prismatic joints are displacements.

    The unit is read from PhysX's DOF type rather than guessed from the
    joint's name, so a gripper finger that slides is not filed as radians.
    """
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()
    robot_id, _ = _entity_of(adapter, observation, "left_arm")
    meta = adapter.property_metadata(robot_id)

    assert meta["joint_pos_joint1"].group == "joint angle"
    assert meta["joint_pos_joint1"].note == "rad"
    assert meta["joint_pos_finger_joint1"].group == "joint displacement"
    assert meta["joint_pos_finger_joint1"].note == "m"
    assert meta["joint_vel_finger_joint1"].group == "joint linear velocity"


def test_unknown_dof_types_fall_back_to_one_undeclared_unit():
    """Without engine DOF types the unit is reported unknown, not assumed."""
    asset = _RobotAsset(1, ["joint1"], ["base_link"])  # no root_physx_view
    robot = _Robot("arm", ["joint1"], [], "base_link", "base_link")
    env = _Env(num_envs=1, robot_manager=_RobotManager([robot], [asset]))

    adapter = RoboDojoAdapter(env, env_index=0)
    observation = adapter.reset()
    robot_id, _ = _entity_of(adapter, observation, "arm")
    meta = adapter.property_metadata(robot_id)

    assert meta["joint_pos_joint1"].group == "joint position"
    assert meta["joint_pos_joint1"].note == "rad or m"
    assert meta["joint_pos_joint1"].unit is None


def test_declared_roles_become_structural_tags():
    """Arm/gripper/end-effector tags come from the robot's own declaration."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()
    robot_id, _ = _entity_of(adapter, observation, "left_arm")
    meta = adapter.property_metadata(robot_id)

    assert meta["joint_pos_joint1"].tags == ("arm", "left")
    assert meta["joint_pos_finger_joint1"].tags == ("gripper", "left")
    assert meta["body_pos_w_gripper_link_0"].tags == ("ee", "left")
    assert meta["body_pos_w_base_link_0"].tags == ("base", "left")


def test_env_index_selects_that_environment():
    """The adapter reports its own environment, not env 0 for everyone.

    RoboDojo seeds layouts per environment, so an adapter that silently read
    env 0 would describe one arbitrary sample while appearing to work.
    """
    env = _Env(num_envs=2)
    first = RoboDojoAdapter(env, env_index=0).reset()
    second = RoboDojoAdapter(env, env_index=1).reset()

    raw = env.robot_manager.robot_key[0].data.joint_pos
    assert first.perceived_entities[0].properties[
        "joint_pos_joint1"
    ] == pytest.approx(raw[0][0])
    assert second.perceived_entities[0].properties[
        "joint_pos_joint1"
    ] == pytest.approx(raw[1][0])


def test_out_of_range_env_index_is_rejected():
    with pytest.raises(ValueError, match="outside range"):
        RoboDojoAdapter(_Env(num_envs=2), env_index=2)


def test_uninitialised_env_is_rejected():
    """An env whose scene has not been built must not look like an empty one."""
    env = _Env(robot_manager=_RobotManager([], []))
    with pytest.raises(ValueError, match="no articulation handles"):
        RoboDojoAdapter(env, env_index=0)


def test_coupled_robots_are_one_articulation_not_two_entities():
    """A coupled pair is one Isaac Lab asset, so it is one entity.

    Reporting it twice would republish every joint of the first entity under
    a second id, and a spec binding over robots would match the same
    articulation twice.
    """
    env = _Env(num_envs=1, robot_manager=_default_robot_manager(1, coupled=True))
    adapter = RoboDojoAdapter(env, env_index=0)
    observation = adapter.reset()

    robots = [
        entity
        for entity in observation.perceived_entities.values()
        if entity.entity_class == "robot"
    ]
    assert len(robots) == 1
    assert adapter.entity_labels[robots[0].entity_id] == "left_arm+right_arm"
    # Both arms' declarations still supply tags for the shared articulation.
    meta = adapter.property_metadata(robots[0].entity_id)
    assert meta["joint_pos_joint1"].tags in (("arm", "left"), ("arm", "right"))


def test_rigid_objects_are_not_reported_twice_through_the_dynamic_alias():
    """SceneManager answers to both "rigid" and "dynamic" with one dict."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()

    rigid = [
        entity
        for entity in observation.perceived_entities.values()
        if entity.entity_class == "rigid"
    ]
    assert len(rigid) == 1


def test_task_labels_name_entities_and_fall_back_to_instance_names():
    """``select_mode.label`` is the name a spec author has already seen."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    adapter.reset()

    labels = set(adapter.entity_labels.values())
    assert "bowl0" in labels  # labelled in the task YAML
    assert "drawer_0" in labels  # unlabelled: keeps its instance name


def test_scene_population_may_change_between_observations():
    """Objects respawn per episode; ids stay stable for survivors."""
    scene = _default_scene_manager()
    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    first = adapter.reset()
    bowl_id, _ = _entity_of(adapter, first, "bowl0")

    scene.collections["rigid"]["bowl_1"] = _RigidObject(
        [0.4, 0.2, 0.8], [1.0, 0.0, 0.0, 0.0], np.zeros(3), np.zeros(3)
    )
    second = adapter.get_observation()
    assert bowl_id in second.perceived_entities
    assert len(second.perceived_entities) == len(first.perceived_entities) + 1

    del scene.collections["rigid"]["bowl_0"]
    third = adapter.get_observation()
    assert bowl_id not in third.perceived_entities


def test_entity_class_may_not_change_within_an_episode():
    """An instance name is an identity, and an identity keeps its class.

    Keying entities by ``<collection>:<name>`` instead would turn the same
    declared object into two entities without anyone noticing.
    """
    scene = _default_scene_manager()
    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    adapter.reset()

    moved = scene.collections["rigid"].pop("bowl_0")
    scene.collections["garment"]["bowl_0"] = moved
    with pytest.raises(ValueError, match="reported as both"):
        adapter.get_observation()


def test_robot_values_are_read_on_every_observation():
    """The robot key layout is worked out once; the values it gathers are
    read fresh on every observation, from every array."""
    env = _Env()
    adapter = RoboDojoAdapter(env, env_index=1)
    robot_id, _ = _entity_of(adapter, adapter.reset(), "left_arm")
    data = env.robot_manager.robot_key[0].data
    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                 "body_link_vel_w", "body_incoming_joint_wrench_b"):
        setattr(data, name, getattr(data, name) + 1.0)
    robot = adapter.get_observation().perceived_entities[robot_id]
    assert robot.properties["joint_pos_joint2"] == data.joint_pos[1][1]
    assert robot.properties["joint_vel_finger_joint1"] == data.joint_vel[1][2]
    assert robot.properties["body_pos_w_link1_2"] == data.body_pos_w[1][1][2]
    assert robot.properties["body_quat_w_gripper_link_z"] == data.body_quat_w[1][2][3]
    assert robot.properties["body_ang_vel_w_link1_1"] == data.body_link_vel_w[1][1][4]
    assert robot.properties["link_wrench_gripper_link_5"] == data.body_incoming_joint_wrench_b[1][2][5]


def test_colliding_joint_names_raise_rather_than_drop_one():
    """Two names that sanitise to one token must not silently merge."""
    asset = _RobotAsset(
        1, ["finger.1", "finger_1"], ["base_link"], dof_types=["Rotation", "Rotation"]
    )
    robot = _Robot("arm", ["finger.1", "finger_1"], [], "base_link", "base_link")
    env = _Env(num_envs=1, robot_manager=_RobotManager([robot], [asset]))

    assert _token("finger.1") == _token("finger_1")
    adapter = RoboDojoAdapter(env, env_index=0)
    with pytest.raises(ValueError, match="claimed by both"):
        adapter.reset()


def test_fluid_reports_particles_without_inventing_a_position():
    """A particle set has no declared per-particle identity or pose."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()
    fluid_id, fluid = _entity_of(adapter, observation, "water_0")

    assert fluid.entity_class == "fluid"
    assert fluid.position is None
    assert fluid.velocity is None
    assert fluid.properties["particle_count"] == 2
    assert fluid.properties["particle_centroid_0"] == pytest.approx(0.1)
    note = adapter.property_metadata(fluid_id)["particle_centroid_0"].note
    assert note is not None and "approximation" in note


def test_pose_only_objects_report_no_velocity():
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()
    _, cloth = _entity_of(adapter, observation, "cloth_0")

    assert cloth.velocity is None
    assert cloth.position == pytest.approx(np.array([0.2, 0.1, 0.76]))
    assert "root_lin_vel_w_0" not in cloth.properties
    assert cloth.properties["root_quat_w_w"] == pytest.approx(1.0)


def test_articulated_fixtures_report_their_joints():
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    observation = adapter.reset()
    _, drawer = _entity_of(adapter, observation, "drawer_0")

    assert drawer.properties["joint_pos_slide_joint"] == pytest.approx(0.05)
    assert drawer.properties["joint_vel_slide_joint"] == pytest.approx(0.01)
    assert drawer.properties["root_pos_w_0"] == pytest.approx(0.3)


def test_dt_defaults_to_the_simulator_step_and_can_be_declared():
    """Temporal bounds are in these seconds, so the cadence must be honest."""
    env = _Env()
    assert RoboDojoAdapter(env, env_index=0).dt == pytest.approx(0.012)
    assert RoboDojoAdapter(env, env_index=0, dt=0.04).dt == pytest.approx(0.04)


def test_track_age_counts_from_when_an_entity_appeared():
    """An object that respawns mid-episode is new, not as old as the run."""
    scene = _default_scene_manager()
    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    adapter.reset()
    second = adapter.get_observation()

    for entity in second.perceived_entities.values():
        assert entity.track_state is TrackState.CONFIRMED
        assert entity.track_age == 1

    scene.collections["rigid"]["bowl_1"] = _RigidObject(
        [0.4, 0.2, 0.8], [1.0, 0.0, 0.0, 0.0], np.zeros(3), np.zeros(3)
    )
    third = adapter.get_observation()
    newcomer_id, newcomer = _entity_of(adapter, third, "bowl_1")
    assert newcomer.track_age == 0
    assert third.perceived_entities[0].track_age == 2

    fourth = adapter.get_observation()
    assert fourth.perceived_entities[newcomer_id].track_age == 1


def test_reset_restarts_ages_without_renaming_survivors():
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    adapter.reset()
    adapter.get_observation()
    after = adapter.reset()

    for entity in after.perceived_entities.values():
        assert entity.track_age == 0


def test_cameras_without_rgb_are_not_advertised():
    """Promising frames that never arrive is worse than declaring none."""
    capture = _CaptureManager(
        annotator_type=[["rgb"], ["distance_to_camera"]],
        frames=[
            [np.full((4, 6, 4), 7, dtype=np.uint8)] * 2,
            [np.zeros((4, 6, 4), dtype=np.uint8)] * 2,
        ],
    )
    env = _Env(
        camera_manager=_CameraManager([["head", "wrist_depth"], ["head", "wrist_depth"]]),
        capture_manager=capture,
    )
    adapter = RoboDojoAdapter(env, env_index=1)

    assert adapter.camera_names() == ("head",)


def test_capture_frame_returns_rgb_for_its_own_environment():
    """RoboDojo's tiled capture is indexed per environment, so a frame is
    attributable to the monitored scene."""
    frames = [
        [
            np.full((4, 6, 4), 3, dtype=np.uint8),
            np.full((4, 6, 4), 9, dtype=np.uint8),
        ]
    ]
    capture = _CaptureManager(annotator_type=[["rgb"]], frames=frames)
    env = _Env(
        camera_manager=_CameraManager([["head"], ["head"]]),
        capture_manager=capture,
    )
    adapter = RoboDojoAdapter(env, env_index=1)

    frame = adapter.capture_frame()
    assert frame is not None
    assert frame.shape == (4, 6, 3)
    assert frame.dtype == np.uint8
    assert frame.any(), "a frame of zeros would be a dead camera, not a scene"
    assert int(frame[0, 0, 0]) == 9  # env 1's frame, not env 0's
    assert capture.calls == [((1,), (0,))]
    # A frame with content ends the warm-up on its first pass; only a
    # black one costs more.
    assert env.renders == 1, "the frame must be fenced before it is read"


def test_capture_frame_is_none_without_cameras():
    adapter = RoboDojoAdapter(_Env(), env_index=0)
    assert adapter.camera_names() == ()
    assert adapter.capture_frame() is None


def test_unknown_camera_name_is_rejected():
    capture = _CaptureManager(
        annotator_type=[["rgb"]], frames=[[np.zeros((2, 2, 4), dtype=np.uint8)] * 2]
    )
    env = _Env(
        camera_manager=_CameraManager([["head"], ["head"]]),
        capture_manager=capture,
    )
    adapter = RoboDojoAdapter(env, env_index=0)

    with pytest.raises(ValueError, match="Unknown RoboDojo camera"):
        adapter.capture_frame("nose")


def test_the_adapter_never_reads_a_reward_or_verdict_api():
    """A monitor fed the benchmark's own verdict can only restate it.

    The reset-pose tables of ``func_parser`` are the one permitted read
    (see test_reward_state_quantities_come_from_robodojos_own_accessors);
    an env without a func_parser must not be probed any further than
    this.
    """

    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"adapter touched reward surface {name!r}")

    env = _Env()
    env.reward_manager = _Exploding()
    env.obs_manager = _Exploding()

    adapter = RoboDojoAdapter(env, env_index=0)
    adapter.reset()
    adapter.get_observation()


def test_hold_mode_advances_physics_under_no_new_command():
    """A scene-level spec can be exercised without a policy in the loop."""
    env = _Env()
    adapter = RoboDojoAdapter(env, env_index=0)
    advance = adapter.make_advancer("hold", None, 0)

    assert advance(0) is True
    assert advance(1) is True
    assert env.sim_steps == 2


def test_unsupported_drive_modes_are_refused_not_approximated():
    """RoboDojo defines no action box to sample and no state replay."""
    adapter = RoboDojoAdapter(_Env(), env_index=0)

    with pytest.raises(SystemExit, match="mode random is not supported"):
        adapter.make_advancer("random", None, 0)
    with pytest.raises(SystemExit, match="no --demo replay"):
        adapter.make_advancer("hold", "episode.hdf5", 0)
    # No state tier: get_state misses the scene objects
    # (docs/evidence/state-replay-robodojo-2026-10-14.md).
    assert adapter.state_signature() is None and adapter.capture_state() is None


def test_a_black_first_frame_is_retried_until_the_renderer_warms_up():
    """Omniverse renders a just-built scene black; that is not a camera."""
    black = np.zeros((4, 6, 4), dtype=np.uint8)
    lit = np.full((4, 6, 4), 5, dtype=np.uint8)

    class _WarmingCapture(_CaptureManager):
        """Black for the first two reads, then a real frame."""

        def step(self, env_ids=None, cam_ids=None):
            self.calls.append((tuple(env_ids or ()), tuple(cam_ids or ())))
            frame = black if len(self.calls) <= 2 else lit
            return [{"rgb": [{"data": frame, "info": None}]}]

    capture = _WarmingCapture(annotator_type=[["rgb"]], frames=[[black, black]])
    env = _Env(
        camera_manager=_CameraManager([["head"], ["head"]]),
        capture_manager=capture,
    )
    adapter = RoboDojoAdapter(env, env_index=0)

    frame = adapter.capture_frame()
    assert frame is not None and frame.any()
    assert env.renders == 3, "one fenced pass per attempt, no more"

    # Warmed: the next frame costs one pass, black or not.
    adapter.capture_frame()
    assert env.renders == 4


def test_a_scene_that_stays_black_is_reported_not_hidden():
    """After the retries, the renderer's own output is what is returned."""
    black = np.zeros((4, 6, 4), dtype=np.uint8)
    capture = _CaptureManager(annotator_type=[["rgb"]], frames=[[black, black]])
    env = _Env(
        camera_manager=_CameraManager([["head"], ["head"]]),
        capture_manager=capture,
    )
    adapter = RoboDojoAdapter(env, env_index=0)

    frame = adapter.capture_frame()
    assert frame is not None and not frame.any()
    assert env.renders == RoboDojoAdapter._WARMUP_RENDERS


class _DynamicObject:
    """RoboDojo's DynamicObject: a USD xform with a pose and NO velocity.

    It is stored in the same collection as the PhysX rigid bodies, which
    is exactly why the adapter must not assume that collection is
    type-homogeneous.
    """

    def __init__(self, position, orientation):
        self._pose = (np.asarray(position), np.asarray(orientation))

    def get_world_pose(self):
        return self._pose


def test_a_dynamic_object_is_not_asked_for_a_velocity_it_lacks():
    """The conveyor tasks put DynamicObject in the rigid collection.

    Asking every member of that dictionary for get_linear_velocity()
    raises AttributeError, which would take down reset() on those tasks.
    """
    scene = _default_scene_manager()
    scene.collections["rigid"]["box_0"] = _DynamicObject(
        [0.0, 0.5, 0.8], [1.0, 0.0, 0.0, 0.0]
    )
    _declare_type(scene, "box_0", "dynamic")

    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    observation = adapter.reset()
    box_id, box = _entity_of(adapter, observation, "box_0")

    assert box.entity_class == "dynamic"
    assert box.entity_class in adapter.supported_classes
    assert box.position == pytest.approx(np.array([0.0, 0.5, 0.8]))
    assert box.velocity is None
    # No invented zeros: a fabricated velocity is indistinguishable from a
    # genuinely stationary body.
    assert not [k for k in box.properties if "vel" in k]
    assert box.properties["root_quat_w_w"] == pytest.approx(1.0)
    assert set(adapter.property_metadata(box_id)) == set(box.properties)


def test_the_declared_layout_type_beats_the_collection_it_is_stored_in():
    """A rigid-collection member declared "dynamic" is reported dynamic."""
    scene = _default_scene_manager()
    _declare_type(scene, "bowl_0", "dynamic")

    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    observation = adapter.reset()
    _, bowl = _entity_of(adapter, observation, "bowl0")

    assert bowl.entity_class == "dynamic"


def test_an_unknown_declared_type_falls_back_to_the_collection():
    """An unfamiliar layout type must not slip past supported_classes."""
    scene = _default_scene_manager()
    _declare_type(scene, "bowl_0", "trampoline")

    adapter = RoboDojoAdapter(_Env(scene_manager=scene), env_index=0)
    observation = adapter.reset()
    _, bowl = _entity_of(adapter, observation, "bowl0")

    assert bowl.entity_class == "rigid"
    for entity in observation.perceived_entities.values():
        assert entity.entity_class in adapter.supported_classes


def test_handles_recognises_the_env_an_eval_run_actually_holds():
    """EvalEnv is defined in src.eval_client.eval_env, not in a task module.

    Wrapping the caller's own env is the main way a policy run gets
    monitored, so detection that only knew the task modules would miss it.
    """

    class _EvalEnvLike:
        pass

    _EvalEnvLike.__module__ = "src.eval_client.eval_env"
    assert _handles(_EvalEnvLike())

    class _TaskEnvLike:
        pass

    _TaskEnvLike.__module__ = "task.RoboDojo.tasks.stack_bowls"
    assert _handles(_TaskEnvLike())

    # A fork's own subclass is recognised by its managers instead.
    assert _handles(_Env())

    class _SomeOtherSimEnv:
        pass

    _SomeOtherSimEnv.__module__ = "robosuite.environments.manipulation"
    assert not _handles(_SomeOtherSimEnv())


def test_link_velocities_come_from_the_link_frame():
    """Link velocities are read from body_link_vel_w (the frame body_pos_w
    is in), split into m/s and rad/s groups, one key per link and axis,
    and absent when the articulation does not provide them."""
    asset = _RobotAsset(2, ["joint1"], ["base_link", "gripper_link"], seed=3)
    robot = _Robot("left_arm", ["joint1"], [], "gripper_link", "base_link")
    env = _Env(num_envs=2, robot_manager=_RobotManager([robot], [asset]),
               scene_manager=_SceneManager({}))
    adapter = RoboDojoAdapter(env, env_index=1)
    obs = adapter.reset()
    arm = next(e for e in obs.perceived_entities.values() if e.entity_class == "robot")
    vel = asset.data.body_link_vel_w[1]
    for slot, link in enumerate(["base_link", "gripper_link"]):
        for axis in range(3):
            assert arm.properties[f"body_lin_vel_w_{link}_{axis}"] == pytest.approx(vel[slot][axis])
            assert arm.properties[f"body_ang_vel_w_{link}_{axis}"] == pytest.approx(vel[slot][3 + axis])
    meta = adapter.property_metadata(arm.entity_id)
    assert meta["body_lin_vel_w_gripper_link_0"].unit == "m/s"
    assert meta["body_ang_vel_w_gripper_link_0"].unit == "rad/s"
    assert meta["body_lin_vel_w_gripper_link_0"].tags == meta["body_pos_w_gripper_link_0"].tags
    del asset.data.body_link_vel_w
    obs = adapter.get_observation()
    arm = next(e for e in obs.perceived_entities.values() if e.entity_class == "robot")
    assert not any(k.startswith("body_lin_vel_w") for k in arm.properties)


def _pair_queries(arguments, instances=()):
    from specola.query_config import QueryConfiguration

    return QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "robodojo.mesh_pair", "arguments": arguments}],
        "instances": list(instances),
    })


def test_pair_scope_matches_compiled_slot_selectors():
    """Issue #136: under a session's entity scope a mesh pair is read when a
    compiled slot can bind it (pin absent or naming it, every @ label among
    its tags); reset, no scope, population: all and named pairs read all."""
    robot = ("a:left_arm", "b:bowl0", "robot", "method:vertex_sample")
    environment = ("a:bowl0", "b:cup0", "environment", "method:vertex_sample")
    adapter = RoboDojoAdapter(_Env(), distmax=1.0)
    adapter._entity_id_of.update({"pair:robot": 40, "pair:environment": 41})

    def read(selectors):
        adapter.set_extraction_scope({"mesh_pair": frozenset({"mesh_distance_sampled"})})
        adapter.set_extraction_entity_scope(selectors)
        return {key for key, tags in (("pair:robot", robot), ("pair:environment", environment))
                if adapter._pair_in_scope(key, tags)}

    everything = {"pair:robot", "pair:environment"}
    filtered = {"mesh_pair": frozenset({(None, (("mesh_distance_sampled", ("robot",)),))})}
    # Before the episode starts, which is where reset reads, every pair.
    assert read(filtered) == everything
    adapter.reset()
    assert read(filtered) == {"pair:robot"}
    assert read({"mesh_pair": frozenset({(41, ())})}) == {"pair:environment"}
    assert read({"mesh_pair": frozenset({(40, (("mesh_distance_sampled", ("environment",)),))})}) == set()
    assert read({"mesh_pair": frozenset({(None, ())})}) == everything
    # No specification names a mesh pair: none is read.
    assert read({"robot": frozenset({(None, ())})}) == set()
    adapter.set_extraction_scope(None)
    assert {key for key, tags in (("pair:robot", robot), ("pair:environment", environment))
            if adapter._pair_in_scope(key, tags)} == everything

    adapter.set_query_configuration(_pair_queries({"distmax": 0.05, "population": "all"}))
    assert read(filtered) == everything
    adapter.set_query_configuration(_pair_queries({"distmax": 0.05}, [{
        "binding": "robodojo.mesh_pair", "id": "arm_bowl", "arguments": {},
        "operands": {"a": {"kind": "entity", "name": "left_arm"}, "b": {"kind": "entity", "name": "bowl0"}},
    }]))
    assert read({"robot": frozenset({(None, ())})}) == everything


def _robots_handed_to_the_task(monkeypatch, tmp_path):
    """Run the factory against stand-ins for the RoboDojo modules it imports
    and return the robot entries of the config the task constructor gets.
    The constructor stand-in stops the build there, so no Kit is needed."""
    import sys
    import types

    from specola.adapters import robodojo

    class Built(Exception):
        pass

    handed = []

    def task_class(cfg, app):
        handed.append(cfg)
        raise Built

    # What process_config returns: the robot entries as RoboDojo's
    # env_cfg/robot YAML spells them, one with the flag already set.
    built = types.SimpleNamespace(
        sim=types.SimpleNamespace(),
        robot=types.SimpleNamespace(robots=[{"robot_name": "x5"}, {"robot_name": "x5", "need_planner": True}]),
    )
    stand_ins = {
        "omegaconf": {"OmegaConf": types.SimpleNamespace(create=lambda cfg: cfg, update=lambda *a, **k: None)},
        "env.global_configs": {"BENCHMARK": "Stub", "ENV_CONFIG_PATH": str(tmp_path), "ROOT_DIR": str(tmp_path)},
        "utils.load_file": {"load_yaml": lambda path: {"config": dict.fromkeys(("sim", "scene", "camera", "robot"), "x")}},
        "utils.pipeline_utils": {"process_randomization": lambda cfg: cfg,
                                 "process_config": lambda cfg, task_name: (built, 1)},
        "task.Stub.task_registry": {"load_task_class": lambda task: (task, task_class),
                                    "task_config_path": lambda root, name: root},
    }
    for name, attributes in stand_ins.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(robodojo, "_robodojo_root", lambda: str(tmp_path))
    monkeypatch.setattr(robodojo, "_ensure_simulation_app", lambda **kwargs: object())
    with pytest.raises(Built):
        robodojo._make_robodojo("stub_task", None, True)
    return handed[0].robot.robots


def test_the_factory_skips_the_planner_unless_asked_to_keep_it(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBODOJO_PLANNER", raising=False)
    robots = _robots_handed_to_the_task(monkeypatch, tmp_path)
    assert [robot.get("need_planner") for robot in robots] == [False, False]

    monkeypatch.setenv("ROBODOJO_PLANNER", "1")
    robots = _robots_handed_to_the_task(monkeypatch, tmp_path)
    assert [robot.get("need_planner") for robot in robots] == [None, True]


@pytest.mark.robodojo
def test_a_live_session_reads_only_the_pairs_its_specs_bind(tmp_path, monkeypatch):
    """stack_bowls layout 0: reset forms every pair, and after it a session
    whose one specification selects the pairs tagged b:bowl0 reads those."""
    pytest.importorskip("isaaclab")
    import gc
    import sys

    # Kit appends sys.argv to its own command line; under `pytest -m
    # robodojo` it segfaulted in SimulationApp._start_app (Isaac Sim 5.1.0,
    # 2026-10-08), while `-k` selection started.
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    monkeypatch.delenv("ROBODOJO_PLANNER", raising=False)
    from specola import registry
    from specola.adapters.robodojo import _robodojo_root
    from specola.session import Monitor

    try:
        _robodojo_root()
    except ValueError as exc:
        pytest.skip(str(exc))
    adapter = registry.resolve("robodojo").factory("stack_bowls", None, True)
    assert isinstance(adapter, RoboDojoAdapter)
    # The factory freezes the heap only while the task env is built.
    assert gc.get_freeze_count() == 0
    # RoboDojo honoured need_planner: no cuRobo planner was built.
    assert adapter._env.robot_manager.planner == {}
    adapter.set_query_configuration(_pair_queries({"distmax": 0.05, "robot_links": ["link6", "link7", "link8"]}))
    specs = tmp_path / "specs.yaml"
    specs.write_text('specs:\n- name: s\n  formula: "mesh_pair_*[mesh_distance_sampled @ b:bowl0] >= 0.005"\n')
    session = Monitor(str(specs), results_dir=None).watch(adapter)

    def pairs():
        observation = adapter.last_observation
        assert observation is not None
        return {adapter.entity_labels[eid] for eid, entity in observation.perceived_entities.items()
                if entity.entity_class == "mesh_pair"}

    formed = pairs()
    adapter._env.sim_step(render=False)
    session.step()
    read = pairs()
    session.end()
    assert read == {label for label in formed if label.endswith("|bowl0")}
    assert read and read < formed


# -- RoboDojo reward-state quantities (native_*, reset_*, ee_*, footprints) --


def _box(half):
    return [
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]


class _NativeLayoutManager(_LayoutManager):
    """Adds the pose and bbox accessors RoboDojo's reward helpers use."""

    def __init__(self, records, poses, boxes, model_ids, instance_types=None):
        metadata = {
            name: {"model_id": model_ids[name], "geometry": {"oriented_bbox": {"vertices": box}}}
            for name, box in boxes.items()
        }
        super().__init__(records, instance_types=instance_types, metadata=metadata)
        self.poses = poses  # name -> (pos, quat); None models a pose-less type

    def get_instance_pose(self, env_idx, label=None, inst_name=None, relative=True):
        if inst_name not in self.poses:
            return (None, None)
        pose = self.poses[inst_name]
        if pose is None:
            return None  # RoboDojo falls through for "dynamic" instances
        return (np.asarray(pose[0], dtype=np.float32), np.asarray(pose[1], dtype=np.float32))

    def get_instance_bbox_vertices(self, inst_name, env_idx):
        return np.asarray(self._metadata[inst_name]["geometry"]["oriented_bbox"]["vertices"])


class _NativeRobotManager(_RobotManager):
    def __init__(self, robot_list, robot_key, endposes):
        super().__init__(robot_list, robot_key)
        self.endposes = endposes  # arm name -> 7-vector

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        return {0: np.asarray(self.endposes[robot.arm_name], dtype=np.float32), 1: None}


class _FuncParser:
    def __init__(self, pre_state, homes):
        self.pre_state = [pre_state, {}]
        self.robot_origin_endpose = [homes, {}]


class _RewardManager:
    """Only func_parser's two reset tables may be read."""

    def __init__(self, func_parser):
        self.func_parser = func_parser

    def __getattr__(self, name):
        raise AssertionError(f"adapter touched reward surface {name!r}")


def _calc_polygon_module():
    """RoboDojo's utils.transformer.calc_polygon, for the import the
    adapter makes inside a RoboDojo process (copied from checkout
    726e9aa: transforms3d's quat2mat normalises the quaternion)."""
    import types

    from shapely.geometry import MultiPoint

    def quat2mat(q):
        w, x, y, z = q
        n = w * w + x * x + y * y + z * z
        s = 2.0 / n
        return np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ])

    def calc_polygon(origin_pose, origin_bbox_points, margin=0.0):
        pose = np.asarray(origin_pose, dtype=float).reshape(7)
        points = np.asarray(origin_bbox_points, dtype=float).reshape(-1, 3)
        world = points @ quat2mat(pose[3:]).T + pose[:3]
        polygon = MultiPoint(world[:, :2]).convex_hull.buffer(margin)
        return polygon, world[:, 2].min(), world[:, 2].max()

    module = types.ModuleType("utils.transformer")
    module.calc_polygon = calc_polygon
    package = types.ModuleType("utils")
    package.transformer = module
    return package, module


def _native_env(target_pose, basket_pose=((0.0, 0.0, 0.8), (1.0, 0.0, 0.0, 0.0))):
    robots = _default_robot_manager(num_envs=2)
    robot_manager = _NativeRobotManager(
        robots.robot_list, robots.robot_key,
        {"left_arm": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]},
    )
    poses = {"basket_0": basket_pose, "target_0": target_pose, "belt_0": None}
    layout = _NativeLayoutManager(
        {"Rigid": [
            {"inst_name": "basket_0", "label": "basket"},
            {"inst_name": "target_0", "label": "target"},
        ], "Dynamic": [{"inst_name": "belt_0", "label": "conveyor"}]},
        poses,
        {"basket_0": _box((0.18, 0.12, 0.11)), "target_0": _box((0.03, 0.03, 0.03))},
        {"basket_0": 2, "target_0": 7},
    )
    rigid = {
        name: _RigidObject(*(pose or ((1.0, 0.0, 0.7), (1.0, 0.0, 0.0, 0.0))), np.zeros(3), np.zeros(3))
        for name, pose in poses.items()
    }
    env = _Env(robot_manager=robot_manager, scene_manager=_SceneManager({"rigid": rigid}, layout))
    env.reward_manager = _RewardManager(_FuncParser(
        {"target_0": {"pose": np.asarray([0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)}},
        {"left_arm": np.asarray([0.1, 0.2, 0.35, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)},
    ))
    return env


def test_reward_state_quantities_come_from_robodojos_own_accessors(monkeypatch):
    """native_*, reset_*, model_id and ee_* are what RoboDojo's predicates
    read, value for value; a pose-less dynamic instance reports none."""
    pytest.importorskip("shapely")
    package, module = _calc_polygon_module()
    monkeypatch.setitem(__import__("sys").modules, "utils", package)
    monkeypatch.setitem(__import__("sys").modules, "utils.transformer", module)

    adapter = RoboDojoAdapter(_native_env(((0.05, 0.0, 0.79), (1.0, 0.0, 0.0, 0.0))), env_index=0)
    observation = adapter.reset()

    _, target = _entity_of(adapter, observation, "target")
    assert [target.properties[f"native_pos_{a}"] for a in "012"] == pytest.approx([0.05, 0.0, 0.79])
    assert target.properties["native_quat_w"] == 1.0
    assert target.properties["reset_pos_2"] == pytest.approx(0.7)
    assert target.properties["model_id"] == 7.0
    assert target.properties["footprint_zmin"] == pytest.approx(0.76)
    assert target.properties["footprint_zmax"] == pytest.approx(0.82)
    # 0.18 - (0.05 + 0.03) = 0.10 to the basket's +x edge; y gives 0.09.
    assert target.properties["footprint_in_basket"] == pytest.approx(0.09)
    _, basket = _entity_of(adapter, observation, "basket")
    assert "reset_pos_2" not in basket.properties  # no capture, no key
    _, belt = _entity_of(adapter, observation, "conveyor")
    assert not [k for k in belt.properties if k.startswith(("native_", "footprint_"))]

    robot = next(e for e in observation.perceived_entities.values() if e.entity_class == "robot")
    assert robot.properties["ee_pos_2"] == pytest.approx(0.3)
    assert robot.properties["ee_home_pos_2"] == pytest.approx(0.35)
    assert robot.properties["ee_home_quat_w"] == 1.0

    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(entity.properties)


def test_footprint_margin_sign_is_robodojos_containment(monkeypatch):
    """The margin is non-negative exactly when the basket footprint
    contains the target footprint (shapely contains, as is_A_cover_B
    tests), over tilted, yawed and displaced targets."""
    pytest.importorskip("shapely")
    package, module = _calc_polygon_module()
    rng = np.random.default_rng(3)
    boxes = (np.asarray(_box((0.18, 0.12, 0.11))), np.asarray(_box((0.03, 0.05, 0.02))))
    agree = 0
    for _ in range(400):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-0.6, 0.6)
        quat = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
        centre = np.r_[rng.uniform(-0.25, 0.25, size=2), 0.8]
        outer = module.calc_polygon(np.r_[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], boxes[0])[0]
        inner = module.calc_polygon(np.r_[centre, quat], boxes[1])[0]
        margin = RoboDojoAdapter._containment_margin(outer, inner)
        assert (margin >= 0) == bool(outer.contains(inner))
        agree += 1
    assert agree == 400
    # On the boundary: contained, margin exactly zero (inclusive).
    from shapely.geometry import box

    assert RoboDojoAdapter._containment_margin(box(0, 0, 1, 1), box(0, 0, 0.5, 0.5)) == 0.0
    assert box(0, 0, 1, 1).contains(box(0, 0, 0.5, 0.5))
