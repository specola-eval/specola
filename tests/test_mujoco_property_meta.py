"""MuJoCo adapter property metadata is derived from jnt_type, not names."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import BodyRule, MujocoAdapter, RegionSpec

_XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0.4 0.2 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
      <body name="lid_body" pos="0 0 0.05">
        <joint name="lid_hinge" type="hinge" axis="0 1 0"/>
        <geom name="lid_geom" type="box" size="0.04 0.04 0.01" mass="0.02"/>
      </body>
    </body>
    <body name="arm" pos="0 0 0.2">
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="arm_hinge" type="hinge" axis="0 0 1"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.3"/>
        <site name="link1_site" pos="0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_hinge" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
  <sensor>
    <touch name="link1_touch" site="link1_site"/>
    <accelerometer name="link1_accel" site="link1_site"/>
  </sensor>
</mujoco>
"""


class _TinyEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


def _make_adapter():
    adapter = MujocoAdapter(
        _TinyEnv(),
        body_rules=[
            BodyRule(match="obj", cls="object"),
            BodyRule(match=lambda name: name == "arm", cls="robot",
                     joint_scope="subtree"),
        ],
    )
    observation = adapter.reset()
    entities = observation.perceived_entities
    by_class = {e.entity_class: e for e in entities.values()}
    return adapter, by_class


def test_free_joint_position_metadata():
    adapter, by_class = _make_adapter()
    obj = by_class["object"]
    meta = adapter.property_metadata(obj.entity_id)
    assert meta["qpos_obj_joint_0"].group == "joint position"
    assert meta["qpos_obj_joint_3"].group == "joint orientation"
    assert meta["qvel_obj_joint_0"].group == "joint linear velocity"
    assert meta["qvel_obj_joint_3"].group == "joint angular velocity"
    assert meta["qfrc_applied_obj_joint_0"].group == "applied forces"
    # every exposed property key has metadata
    assert set(meta) == set(obj.properties)


def test_hinge_joint_never_claims_position():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    meta = adapter.property_metadata(robot.entity_id)
    assert meta["qpos_arm_hinge"].group == "joint angle"
    assert meta["qvel_arm_hinge"].group == "joint angular velocity"


def test_metadata_carries_units_from_joint_type():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    obj = by_class["object"]
    robot_meta = adapter.property_metadata(robot.entity_id)
    obj_meta = adapter.property_metadata(obj.entity_id)
    assert robot_meta["qpos_arm_hinge"].note == "rad"
    assert robot_meta["qvel_arm_hinge"].note == "rad/s"
    assert obj_meta["qpos_obj_joint_0"].note == "m"
    assert obj_meta["qvel_obj_joint_0"].note == "m/s"
    assert obj_meta["qvel_obj_joint_3"].note == "rad/s"
    assert obj_meta["qpos_obj_joint_3"].note == "quat w"


def test_groups_split_by_physical_quantity():
    adapter, by_class = _make_adapter()
    robot_meta = adapter.property_metadata(by_class["robot"].entity_id)
    obj_meta = adapter.property_metadata(by_class["object"].entity_id)
    # hinge: torque applied about an axis
    assert robot_meta["qfrc_applied_arm_hinge"].group == "applied torques"
    assert robot_meta["qfrc_applied_arm_hinge"].note == "N·m"
    # free joint: first three DOF are forces, last three torques
    assert obj_meta["qfrc_applied_obj_joint_0"].group == "applied forces"
    assert obj_meta["qfrc_applied_obj_joint_0"].note == "N"
    assert obj_meta["qfrc_applied_obj_joint_3"].group == "applied torques"
    assert obj_meta["qfrc_applied_obj_joint_3"].note == "N·m"


def _planar_meta():
    class _PlanarEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(_PLANAR_XML)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _PlanarEnv(),
        body_rules=[BodyRule(match="torso", cls="robot",
                             joint_scope="subtree")],
    )
    observation = adapter.reset()
    entity_id = next(iter(observation.perceived_entities))
    return adapter.property_metadata(entity_id)


def test_slide_joints_group_as_displacements():
    meta = _planar_meta()
    # articulation on a descendant body keeps the quantity groups
    assert meta["qpos_piston_slide"].group == "joint displacement"
    assert meta["qpos_piston_slide"].note == "m"
    assert meta["qvel_piston_slide"].group == "joint linear velocity"
    assert meta["qvel_piston_slide"].note == "m/s"
    assert meta["qfrc_applied_rootx"].group == "applied forces"
    assert meta["qfrc_applied_rootx"].note == "N"


def test_root_joints_group_as_base_pose():
    meta = _planar_meta()
    # slide/hinge on the matched body itself are pose coordinates of the
    # body relative to its spawn frame, not articulation -- and they split
    # by unit the same way articulation does, so no group mixes m with rad
    assert meta["qpos_rootx"].group == "base displacement (rel. spawn)"
    assert meta["qpos_rootx"].note == "m"
    assert meta["qpos_rooty"].group == "base angle (rel. spawn)"
    assert meta["qpos_rooty"].note == "rad"
    assert meta["qvel_rootx"].group == "base linear velocity"
    assert meta["qvel_rootx"].note == "m/s"
    assert meta["qvel_rooty"].group == "base angular velocity"
    assert meta["qvel_rooty"].note == "rad/s"


def test_no_group_mixes_units():
    """
    A `@ group` selector aggregates every key in the group, so a group
    carrying two units silently compares e.g. metres against radians.
    `specola check` warns about it, but the warning should never be
    reachable through a group an adapter itself produces.

    "body orientation" is the one intended exception: its notes are the
    quaternion component labels ("quat w", "quat x", ...), not units --
    the components share one dimensionless unit. That `note` doubles as a
    component label there is why this test allowlists rather than skips it.
    """
    meta = _planar_meta()

    notes_by_group = {}
    for key, entry in meta.items():
        if entry is None or not entry.group:
            continue
        notes_by_group.setdefault(entry.group, set()).add(entry.note)

    mixed = {
        group
        for group, notes in notes_by_group.items()
        if len(notes) > 1
    }

    assert mixed == {"body orientation"}, (
        f"groups carrying more than one unit: {sorted(mixed)}"
    )


def _mixed_unit_groups(meta):
    notes_by_group = {}
    for entry in meta.values():
        if entry is None or not entry.group:
            continue
        notes_by_group.setdefault(entry.group, set()).add(entry.note)
    return {
        group
        for group, notes in notes_by_group.items()
        if len(notes) > 1
    }


def test_no_group_mixes_units_on_an_actuator_bearing_model():
    """
    ``test_no_group_mixes_units`` above runs on ``_planar_meta()``, whose
    model declares no actuator and no sensor — so it has never been able
    to see the ``actuators`` group at all. This runs the same check on a
    model that has both, which makes one known exception visible instead
    of leaving it hidden by the fixture's shape.

    Every actuator group is unit-homogeneous, so nothing new appears in
    the allowlist: ``body orientation`` and ``joint orientation`` are the
    same quaternion-component-label case the sibling test documents.

    The command groups carry the control range as their note while their
    group name carries the units the command is expressed in, so they are
    unit-homogeneous like every other actuator family.
    ``test_actuator_command_groups_are_unit_homogeneous`` below covers
    that with a slide actuator beside a hinge one.

    The point of pinning the allowlist is that any *new* mixed group
    fails this test rather than slipping through.
    """
    adapter, by_class = _make_adapter()

    mixed = set()
    for entity in (by_class["robot"], by_class["object"]):
        mixed |= _mixed_unit_groups(
            adapter.property_metadata(entity.entity_id)
        )

    assert mixed == {
        "body orientation",
        "joint orientation",
    }, f"groups carrying more than one unit: {sorted(mixed)}"


def test_kinematic_root_rules_no_body_class():
    from specola.adapters.dmcontrol import kinematic_root_rules

    model = mujoco.MjModel.from_xml_string(_PLANAR_XML)
    rules = kinematic_root_rules(model)
    assert [rule.cls for rule in rules] == ["robot"]
    assert rules[0].joint_scope == "subtree"


def test_unknown_entity_returns_empty():
    adapter, _ = _make_adapter()
    assert adapter.property_metadata(9999) == {}


def test_touch_sensor_attached_to_owning_entity():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    assert "sensordata_link1_touch" in robot.properties
    meta = adapter.property_metadata(robot.entity_id)
    assert meta["sensordata_link1_touch"].group == "sensors: touch"
    assert meta["sensordata_link1_touch"].note == "N"
    assert set(meta) == set(robot.properties)


def test_dim3_sensor_produces_indexed_keys():
    # accelerometer has dim=3, unlike the dim=1 touch sensor above; each
    # axis gets its own suffixed key and matching metadata.
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    keys = [f"sensordata_link1_accel_{axis}" for axis in range(3)]
    for key in keys:
        assert key in robot.properties
    meta = adapter.property_metadata(robot.entity_id)
    for key in keys:
        assert meta[key].group == "sensors: accelerometer"
        assert meta[key].note == "m/s²"
    assert set(meta) == set(robot.properties)


def test_orphan_sensors_get_synthetic_entity():
    # only the object is matched -> the touch sensor's body (link1)
    # belongs to no entity, so a synthetic "sensors" entity carries it
    adapter = MujocoAdapter(
        _TinyEnv(),
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    observation = adapter.reset()
    by_class = {e.entity_class: e for e in
                observation.perceived_entities.values()}
    assert "sensors" in by_class
    sensors = by_class["sensors"]
    assert sensors.position is None
    assert "sensordata_link1_touch" in sensors.properties
    meta = adapter.property_metadata(sensors.entity_id)
    assert set(meta) == set(sensors.properties)


def test_actuator_keys_attached_via_joint_transmission():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    assert "ctrl_arm_motor" in robot.properties
    assert "actuator_force_arm_motor" in robot.properties
    meta = adapter.property_metadata(robot.entity_id)
    assert meta["ctrl_arm_motor"].group == "actuator torque command"
    assert "[-1" in (meta["ctrl_arm_motor"].note or "")
    assert meta["actuator_force_arm_motor"].group == "actuator output torque"
    assert meta["actuator_force_arm_motor"].note == "N·m"
    assert set(meta) == set(robot.properties)


def test_orphan_actuators_get_synthetic_entity():
    # only the object is matched -> the motor's joint body (link1)
    # belongs to no entity, so a synthetic "actuators" entity carries it
    adapter = MujocoAdapter(
        _TinyEnv(),
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    observation = adapter.reset()
    by_class = {e.entity_class: e for e in
                observation.perceived_entities.values()}
    assert "actuators" in by_class
    actuators = by_class["actuators"]
    assert actuators.position is None
    assert "ctrl_arm_motor" in actuators.properties
    assert "actuator_force_arm_motor" in actuators.properties
    meta = adapter.property_metadata(actuators.entity_id)
    assert set(meta) == set(actuators.properties)


_PLANAR_XML = """
<mujoco>
  <worldbody>
    <body name="torso" pos="0 0 0.6">
      <joint name="rootx" type="slide" axis="1 0 0"/>
      <joint name="rootz" type="slide" axis="0 0 1"/>
      <joint name="rooty" type="hinge" axis="0 1 0"/>
      <geom name="torso_geom" type="capsule" fromto="-0.2 0 0 0.2 0 0" size="0.05" mass="5"/>
      <body name="piston" pos="0 0 -0.1">
        <joint name="piston_slide" type="slide" axis="0 0 1"/>
        <geom name="piston_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_every_entity_exposes_own_pose_keys():
    adapter, by_class = _make_adapter()
    for name, entity in (("obj", by_class["object"]),
                         ("arm", by_class["robot"])):
        for key in (f"xpos_{name}_0", f"xpos_{name}_1", f"xpos_{name}_2",
                    f"xquat_{name}_w", f"xquat_{name}_x",
                    f"xquat_{name}_y", f"xquat_{name}_z"):
            assert key in entity.properties, (entity.entity_class, key)
        assert np.allclose(
            [entity.properties[f"xpos_{name}_{i}"] for i in range(3)],
            entity.position,
        )


def test_own_pose_keys_have_metadata():
    adapter, by_class = _make_adapter()
    meta = adapter.property_metadata(by_class["object"].entity_id)
    assert meta["xpos_obj_2"].group == "body position"
    assert meta["xpos_obj_2"].note == "m"
    assert meta["xquat_obj_w"].group == "body orientation"
    assert meta["xquat_obj_w"].note == "quat w"
    # parity rule still holds
    assert set(meta) == set(by_class["object"].properties)


def test_subtree_bodies_contribute_pose_properties():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    # link1 is a descendant of arm (subtree scope)
    for key in ("xpos_link1_0", "xpos_link1_2", "xquat_link1_w",
                "cvel_link1_0", "cvel_link1_5"):
        assert key in robot.properties, key
    meta = adapter.property_metadata(robot.entity_id)
    assert meta["xpos_link1_2"].group == "body position"
    assert meta["cvel_link1_0"].group == "body angular velocity"
    assert meta["cvel_link1_0"].note == "rad/s"
    assert meta["cvel_link1_5"].group == "body com-frame linear velocity"
    assert meta["cvel_link1_5"].note == "m/s"
    assert set(meta) == set(robot.properties)


def test_non_subtree_entities_only_own_pose():
    adapter, by_class = _make_adapter()
    obj = by_class["object"]     # BodyRule without subtree scope
    # obj has a child body (lid_body) so a subtree leak would be observable
    pose_keys = [
        key for key in obj.properties
        if key.startswith(("xpos_", "xquat_", "cvel_"))
    ]
    assert pose_keys
    # every pose key names the object's own body, none another body
    assert all(
        key.startswith(("xpos_obj_", "xquat_obj_", "cvel_obj_"))
        for key in pose_keys
    )
    assert "cvel_obj_3" in obj.properties
    # the child body's pose/velocity must not leak in under "body" scope
    assert "xpos_lid_body_0" not in obj.properties
    assert "cvel_lid_body_0" not in obj.properties


def test_all_default_signals_extracted():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    for signal in ("qpos", "qvel", "qacc", "qfrc_actuator",
                   "qfrc_constraint", "qfrc_bias", "qfrc_applied",
                   "qfrc_smooth", "qfrc_passive", "qfrc_spring",
                   "qfrc_damper", "qfrc_fluid", "qfrc_gravcomp"):
        assert f"{signal}_arm_hinge" in robot.properties, signal


def test_new_signal_metadata_groups_and_units():
    adapter, by_class = _make_adapter()
    robot_meta = adapter.property_metadata(by_class["robot"].entity_id)
    obj_meta = adapter.property_metadata(by_class["object"].entity_id)
    assert robot_meta["qacc_arm_hinge"].group == "joint angular acceleration"
    assert robot_meta["qacc_arm_hinge"].note == "rad/s²"
    assert robot_meta["qfrc_actuator_arm_hinge"].group == "actuator torques"
    assert robot_meta["qfrc_actuator_arm_hinge"].note == "N·m"
    assert robot_meta["qfrc_constraint_arm_hinge"].group == "constraint torques"
    assert robot_meta["qfrc_bias_arm_hinge"].group == "bias torques"
    # free joint: translational vs rotational split per index
    assert obj_meta["qfrc_actuator_obj_joint_0"].group == "actuator forces"
    assert obj_meta["qfrc_actuator_obj_joint_0"].note == "N"
    assert obj_meta["qfrc_actuator_obj_joint_3"].group == "actuator torques"
    assert obj_meta["qacc_obj_joint_0"].group == "joint linear acceleration"
    assert obj_meta["qacc_obj_joint_0"].note == "m/s²"
    assert obj_meta["qacc_obj_joint_3"].group == "joint angular acceleration"
    assert obj_meta["qacc_obj_joint_3"].note == "rad/s²"
    # parity holds with the wider signal set
    assert set(obj_meta) == set(by_class["object"].properties)


def test_passive_signal_metadata_groups_and_units():
    meta = _planar_meta()
    # hinge on the matched body: rotational, so torques
    assert meta["qfrc_spring_rooty"].group == "spring torques"
    assert meta["qfrc_spring_rooty"].note == "N·m"
    # slide joints take forces
    assert meta["qfrc_passive_rootx"].group == "passive forces"
    assert meta["qfrc_passive_rootx"].note == "N"
    assert meta["qfrc_damper_piston_slide"].group == "damper forces"
    assert meta["qfrc_fluid_piston_slide"].group == "fluid forces"
    assert meta["qfrc_gravcomp_piston_slide"].group == "gravcomp forces"
    assert meta["qfrc_gravcomp_piston_slide"].note == "N"


def test_body_force_and_acceleration_metadata_splits_by_unit():
    meta = _planar_meta()
    for family, torque_group, force_group in (
        ("cfrc_ext", "body external torque", "body external force"),
        ("cfrc_int", "body internal torque", "body internal force"),
    ):
        for index in range(3):
            assert meta[f"{family}_torso_{index}"].group == torque_group
            assert meta[f"{family}_torso_{index}"].note == "N·m"
        for index in range(3, 6):
            assert meta[f"{family}_torso_{index}"].group == force_group
            assert meta[f"{family}_torso_{index}"].note == "N"

    for index in range(3):
        assert meta[f"cacc_torso_{index}"].group == "body angular acceleration"
        assert meta[f"cacc_torso_{index}"].note == "rad/s²"
    for index in range(3, 6):
        assert (
            meta[f"cacc_torso_{index}"].group
            == "body com-frame linear acceleration"
        )
        assert meta[f"cacc_torso_{index}"].note == "m/s²"


def test_body_force_families_follow_the_cvel_body_scope():
    # cfrc_ext/cfrc_int/cacc are scoped to the same pose_body_ids as
    # cvel, so their key sets line up body for body.
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    cvel_tokens = {
        key[len("cvel_"):].rsplit("_", 1)[0]
        for key in robot.properties
        if key.startswith("cvel_")
    }
    for family in ("cfrc_ext", "cfrc_int", "cacc"):
        tokens = {
            key[len(family) + 1:].rsplit("_", 1)[0]
            for key in robot.properties
            if key.startswith(f"{family}_")
        }
        assert tokens == cvel_tokens, family

    obj = by_class["object"]     # BodyRule without subtree scope
    assert "cfrc_ext_lid_body_0" not in obj.properties
    assert "cacc_lid_body_0" not in obj.properties


_FREE_JOINT_ACTUATOR_XML = """
<mujoco>
  <worldbody>
    <body name="floater" pos="0 0 1">
      <freejoint name="fj"/>
      <geom name="fg" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="push_x" joint="fj" gear="1 0 0 0 0 0"/>
  </actuator>
</mujoco>
"""


_TWO_ACTUATOR_XML = """
<mujoco>
  <worldbody>
    <body name="rig" pos="0 0 1">
      <joint name="lift" type="slide" axis="0 0 1"/>
      <geom name="rg" type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="wrist" pos="0.1 0 0">
        <joint name="twist" type="hinge" axis="0 0 1"/>
        <geom name="wg" type="box" size="0.02 0.02 0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="lift_motor" joint="lift" ctrlrange="-5 5" ctrllimited="true"/>
    <motor name="twist_motor" joint="twist" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

_ACTUATOR_SHORTCUT_XML = """
<mujoco>
  <worldbody>
    <body name="rig">
      <body name="slide_link">
        <joint name="slide" type="slide"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
      </body>
      <body name="hinge_link">
        <joint name="hinge" type="hinge"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_position" joint="slide" kp="10" kv="2"/>
    <position name="hinge_position" joint="hinge" kp="10" kv="2"/>
    <position name="slide_position_zero" joint="slide" kp="0"/>
    <velocity name="slide_velocity" joint="slide" kv="3"/>
    <velocity name="hinge_velocity" joint="hinge" kv="3"/>
    <velocity name="slide_velocity_small" joint="slide" kv="1e-9"/>
    <velocity name="hinge_velocity_zero" joint="hinge" kv="0"/>
    <intvelocity name="slide_intvelocity" joint="slide"
                 kp="10" kv="2" actrange="-1 1"/>
    <intvelocity name="hinge_intvelocity" joint="hinge"
                 kp="10" kv="2" actrange="-1 1"/>
    <damper name="slide_damper" joint="slide" kv="4"
            ctrlrange="0 1"/>
    <damper name="hinge_damper" joint="hinge" kv="4"
            ctrlrange="0 1"/>
    <damper name="slide_damper_small" joint="slide" kv="1e-9"
            ctrlrange="0 1"/>
    <general name="unknown_slide" joint="slide"
             gaintype="fixed" gainprm="1"
             biastype="affine" biasprm="0 -2 0"/>
    <general name="unknown_hinge" joint="hinge"
             gaintype="fixed" gainprm="1"
             biastype="affine" biasprm="0 -2 0"/>
  </actuator>
</mujoco>
"""


def _two_actuator_meta():
    class _RigEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(_TWO_ACTUATOR_XML)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _RigEnv(),
        body_rules=[
            BodyRule(match="rig", cls="robot", joint_scope="subtree"),
        ],
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    return adapter.property_metadata(entity.entity_id), entity


def test_actuator_groups_split_by_unit_like_every_other_family():
    """
    One actuator drives a slide joint and one a hinge, so each actuator
    family spans two units on this single model. They must land in
    different groups, or `@ <group>` would aggregate newtons with
    newton-metres.

    The names carry the ``actuator`` provenance without colliding with
    the per-DOF ``actuator forces``/``actuator torques`` groups, which
    are the generalized force *at each DOF* rather than the actuator's
    own output.
    """
    meta, _ = _two_actuator_meta()

    assert meta["actuator_force_lift_motor"].group == "actuator output force"
    assert meta["actuator_force_lift_motor"].note == "N"
    assert meta["actuator_force_twist_motor"].group == "actuator output torque"
    assert meta["actuator_force_twist_motor"].note == "N·m"

    assert meta["actuator_length_lift_motor"].group == "actuator displacement"
    assert meta["actuator_length_lift_motor"].note == "m"
    assert meta["actuator_length_twist_motor"].group == "actuator angle"
    assert meta["actuator_length_twist_motor"].note == "rad"

    assert meta["actuator_velocity_lift_motor"].group == (
        "actuator linear velocity"
    )
    assert meta["actuator_velocity_lift_motor"].note == "m/s"
    assert meta["actuator_velocity_twist_motor"].group == (
        "actuator angular velocity"
    )
    assert meta["actuator_velocity_twist_motor"].note == "rad/s"


def test_actuator_command_groups_are_unit_homogeneous():
    # Regression for issue #84. A single `actuator command` group held a
    # torque command in N·m beside a position command in m, so
    # `@ actuator command` aggregated incomparable quantities -- the same
    # defect as #70, and the one the split of the old `actuators` group
    # was meant to remove.
    #
    # The command's units follow the actuator's kind, not its transmitted
    # joint alone, so a motor on a slide joint commands N while a motor on
    # a hinge commands N·m. The control range stays in the note, which is
    # the authoring answer to "what threshold do I write here".
    meta, _ = _two_actuator_meta()

    assert meta["ctrl_lift_motor"].group == "actuator force command"
    assert meta["ctrl_twist_motor"].group == "actuator torque command"
    assert meta["ctrl_lift_motor"].note != meta["ctrl_twist_motor"].note

    # No group carrying the actuator provenance may hold two units -- the
    # four output families and both command families alike.
    actuator_groups = {
        group
        for group in _mixed_unit_groups(meta)
        if group.startswith("actuator")
    }
    assert actuator_groups == set()


def test_actuator_shortcuts_have_command_semantics_not_bias_lookalikes():
    class _ShortcutEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(
                _ACTUATOR_SHORTCUT_XML
            )
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _ShortcutEnv(),
        body_rules=[
            BodyRule(match=lambda name: name == "rig", cls="robot",
                     joint_scope="subtree"),
        ],
    )
    observation = adapter.reset()
    robot = next(iter(observation.perceived_entities.values()))
    meta = adapter.property_metadata(robot.entity_id)

    expected_groups = {
        "ctrl_slide_position": "actuator position command",
        "ctrl_hinge_position": "actuator angle command",
        "ctrl_slide_velocity": "actuator linear velocity command",
        "ctrl_hinge_velocity": "actuator angular velocity command",
        "ctrl_slide_velocity_small": "actuator linear velocity command",
        "ctrl_slide_intvelocity": "actuator linear velocity command",
        "ctrl_hinge_intvelocity": "actuator angular velocity command",
        "ctrl_slide_damper": "actuator damping-scale command",
        "ctrl_hinge_damper": "actuator damping-scale command",
        "ctrl_slide_damper_small": "actuator damping-scale command",
    }
    opaque = {
        "ctrl_slide_position_zero",
        "ctrl_hinge_velocity_zero",
        "ctrl_unknown_slide",
        "ctrl_unknown_hinge",
    }

    assert {
        key: meta[key].group
        for key in expected_groups
    } == expected_groups
    for key in opaque:
        assert meta[key].note is None
        assert {"semantics:opaque", "unit:unresolved"} <= set(meta[key].tags)
    assert opaque <= set(robot.properties)
    assert adapter.field_materialization["ctrl"].scene_applicable
    assert not adapter.field_materialization["ctrl"].owner_omissions
    assert set(meta) == set(robot.properties)


def test_ctrl_unit_note_does_not_swallow_unexpected_model_errors():
    class _BrokenModel:
        @property
        def actuator_dyntype(self):
            raise RuntimeError("sentinel model failure")

    adapter = object.__new__(MujocoAdapter)
    adapter._model = _BrokenModel()

    with pytest.raises(RuntimeError, match="sentinel model failure"):
        adapter._ctrl_unit_note(0, "N", "m", "m/s")


def test_free_joint_actuator_quantities_are_exposed_as_opaque():
    """
    A free joint is not a scalar coordinate: MuJoCo projects it through
    the actuator's six-component gear vector. Measured on this model,
    ``actuator_length`` stays 0.0 whatever the body's position, and a
    translation-only gear gives a velocity in m/s where the joint type
    alone would claim rad/s. The live model does not expose one homogeneous
    scalar transmission unit. The values are nevertheless exposed by MuJoCo
    as actuator state, so they remain authoring keys with explicit opaque
    semantics and unresolved units rather than being silently omitted.
    """
    class _FreeEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(
                _FREE_JOINT_ACTUATOR_XML
            )
            self.data = mujoco.MjData(self.model)
            self.data.qpos[0] = 0.37
            self.data.qvel[0] = 0.25
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _FreeEnv(),
        body_rules=[BodyRule(match="floater", cls="object")],
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    meta = adapter.property_metadata(entity.entity_id)

    opaque_keys = {
        "ctrl_push_x",
        "actuator_force_push_x",
        "actuator_length_push_x",
        "actuator_velocity_push_x",
    }
    for key in opaque_keys:
        assert meta[key].note is None
        assert {"semantics:opaque", "unit:unresolved"} <= set(meta[key].tags)
    assert opaque_keys <= set(entity.properties)
    for field_name in (
        "ctrl",
        "actuator_force",
        "actuator_length",
        "actuator_velocity",
    ):
        status = adapter.field_materialization[field_name]
        assert status.scene_applicable
        assert status.reason is None
        assert not status.owner_omissions

    assert set(meta) == set(entity.properties)


def test_actuator_length_and_velocity_metadata():
    adapter, by_class = _make_adapter()
    robot = by_class["robot"]
    meta = adapter.property_metadata(robot.entity_id)
    # arm_motor drives a hinge, so its transmission coordinate is an angle
    assert meta["actuator_length_arm_motor"].group == "actuator angle"
    assert meta["actuator_length_arm_motor"].note == "rad"
    assert meta["actuator_velocity_arm_motor"].group == "actuator angular velocity"
    assert meta["actuator_velocity_arm_motor"].note == "rad/s"
    assert set(meta) == set(robot.properties)


def test_planar_base_joints_get_base_acceleration_group():
    meta = _planar_meta()
    assert meta["qacc_rootx"].group == "base linear acceleration"
    assert meta["qacc_rootx"].note == "m/s²"
    assert meta["qacc_rooty"].group == "base angular acceleration"
    assert meta["qacc_rooty"].note == "rad/s²"
    assert meta["qacc_piston_slide"].group == "joint linear acceleration"


def test_joint_signal_selection_is_rejected_as_non_global_availability():
    with pytest.raises(TypeError, match="availability is global"):
        MujocoAdapter(
            _TinyEnv(),
            body_rules=[BodyRule(match="obj", cls="object")],
            joint_signals=("qpos", "qvel"),
        )


# ---------------------------------------------------------------------------
# Pose/cvel key-collision guard (pose_body_ids built at reset)
# ---------------------------------------------------------------------------

_POSE_COLLISION_XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 0.2">
      <freejoint name="root_joint"/>
      <geom name="root_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
      <body name="part.1" pos="0.1 0 0">
        <joint name="part1_hinge" type="hinge" axis="0 0 1"/>
        <geom name="part1_geom" type="box" size="0.02 0.02 0.02" mass="0.05"/>
      </body>
      <body name="part_1" pos="-0.1 0 0">
        <joint name="part2_hinge" type="hinge" axis="0 0 1"/>
        <geom name="part2_geom" type="box" size="0.02 0.02 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _PoseCollisionEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_POSE_COLLISION_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


def test_colliding_pose_body_names_receive_indexed_tokens():
    adapter = MujocoAdapter(
        _PoseCollisionEnv(),
        body_rules=[
            BodyRule(
                match="root",
                cls="robot",
                joint_scope="subtree",
            ),
        ],
    )

    observation = adapter.reset()
    properties = observation.perceived_entities[
        adapter.name_to_id["root"]
    ].properties
    assert "xpos_part_1_model_index_2_0" in properties
    assert "xpos_part_1_model_index_3_0" in properties


# ---------------------------------------------------------------------------
# Observation-wide property_metadata parity
# ---------------------------------------------------------------------------

_PARITY_XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0.4 0.2 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
      <body name="lid_body" pos="0 0 0.05">
        <joint name="lid_hinge" type="hinge" axis="0 1 0"/>
        <geom name="lid_geom" type="box" size="0.04 0.04 0.01" mass="0.02"/>
      </body>
    </body>
    <body name="arm" pos="0 0 0.2">
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="arm_hinge" type="hinge" axis="0 0 1"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.3"/>
        <site name="link1_site" pos="0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_hinge" ctrlrange="-1 1" ctrllimited="true"/>
    <motor name="lid_motor" joint="lid_hinge" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
  <sensor>
    <touch name="link1_touch" site="link1_site"/>
    <accelerometer name="link1_accel" site="link1_site"/>
    <jointpos name="lid_jointpos" joint="lid_hinge"/>
  </sensor>
</mujoco>
"""


class _ParityEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_PARITY_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


def test_observation_wide_metadata_parity():
    """
    property_metadata parity (every exposed property key has metadata,
    and vice versa) is exercised per entity class elsewhere in this
    file. This test asserts it holds simultaneously, in one
    observation, for every entity class the MuJoCo adapter can produce:
        a subtree-scoped robot, a body-scoped object, a gripper site, a
        region, raw geometry, global simulation state, and the synthetic
        orphan-sensor / orphan-actuator entities.

    ``lid_body`` (a child of ``obj``, outside both the object's
    body-only scope and the robot's arm/link1 subtree) is what makes
    ``lid_jointpos``/``lid_motor`` orphans; ``link1_site`` doubles as
    the gripper site, so ``link1_touch``/``link1_accel`` -- defined
    exactly on it -- attach to the gripper instead of the robot (see
    the gripper-site sensor-ownership precedence in
    ``_sensor_owner_entity_id``).
    """
    adapter = MujocoAdapter(
        _ParityEnv(),
        body_rules=[
            BodyRule(match="obj", cls="object"),
            BodyRule(
                match=lambda name: name == "arm",
                cls="robot",
                joint_scope="subtree",
            ),
        ],
        regions=[
            RegionSpec(
                name="goal_zone",
                source="fixed",
                center=(0.3, 0.0, 0.4),
                half_extents=(0.1, 0.1, 0.1),
            ),
        ],
        gripper_site="link1_site",
    )
    observation = adapter.reset()
    entities = observation.perceived_entities
    by_class = {e.entity_class: e for e in entities.values()}

    # The fixture scene rests bodies in contact, so the dynamic
    # contact-pair class appears alongside the static classes and its
    # metadata parity is validated by the same loop below.
    assert set(by_class) == {
        "object",
        "robot",
        "gripper",
        "region",
        "geom",
        "simulation",
        "sensors",
        "actuators",
        "contact",
        "body",
        "geom_pair",
    }

    for entity in entities.values():
        meta = adapter.property_metadata(entity.entity_id)
        assert set(meta) == set(entity.properties), entity.entity_class


# ---------------------------------------------------------------------------
# Subsystem tags: base adapter has none; the hook injects them per joint
# ---------------------------------------------------------------------------

def test_base_adapter_joint_keys_carry_no_tags():
    adapter, by_class = _make_adapter()
    meta = adapter.property_metadata(by_class["robot"].entity_id)
    assert meta["qvel_arm_hinge"].tags == (
        "source_kind:joint",
        "model_index:2",
        "identity:model_name",
    )
    assert adapter._joint_subsystem_tags(0) == ()


def test_joint_subsystem_tags_hook_injects_tags_on_every_signal():
    class _TaggedAdapter(MujocoAdapter):
        def _joint_subsystem_tags(self, joint_id):
            return ("arm", "right")

    adapter = _TaggedAdapter(
        _TinyEnv(),
        body_rules=[
            BodyRule(match=lambda name: name == "arm", cls="robot",
                     joint_scope="subtree"),
        ],
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    meta = adapter.property_metadata(entity.entity_id)
    assert meta["qvel_arm_hinge"].tags[:2] == ("arm", "right")
    assert "source_kind:joint" in meta["qvel_arm_hinge"].tags
    assert meta["qfrc_actuator_arm_hinge"].tags[:2] == (
        "arm",
        "right",
    )
    assert "source_kind:joint" in (
        meta["qfrc_actuator_arm_hinge"].tags
    )
    # groups and units untouched by tag injection
    assert meta["qvel_arm_hinge"].group == "joint angular velocity"
    assert meta["qvel_arm_hinge"].note == "rad/s"
