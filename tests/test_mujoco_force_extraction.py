"""
Physical-value tests for the MuJoCo force/acceleration families.

These assert that each signal is *non-zero when physics says it should
be*, not merely that its key exists. ``cfrc_ext``, ``cfrc_int`` and
``cacc`` are only populated by an explicit ``mj_rnePostConstraint``
call, which ``mj_step`` never makes; a model that declares a
force/torque sensor gets that call as a side effect, so every model
here declares no sensor except where the test is specifically about the
sensor case. A key-exists assertion would pass in exactly the
silent-zeros failure mode this file guards against (issue #76).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import BodyRule, MujocoAdapter

# 1 kg box resting on a plane, no sensors of any kind: after the scene
# settles the plane must push back with the box's weight, 1 kg * 9.81
# m/s^2, and that shows up at cfrc_ext index 5 (the z force).
_RESTING_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="block" pos="0 0 0.05">
      <freejoint name="block_joint"/>
      <geom name="block_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# The same model plus a declared force sensor. MuJoCo runs
# rnePostConstraint itself to compute that sensor, so cfrc_ext is
# populated here without any help from the adapter -- which is what
# makes this the control for the unconditional call being harmless.
_RESTING_SENSOR_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="block" pos="0 0 0.05">
      <freejoint name="block_joint"/>
      <geom name="block_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
      <site name="block_site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <sensor>
    <force name="block_force" site="block_site"/>
  </sensor>
</mujoco>
"""

# A sprung, damped, partly gravity-compensated hinge in a fluid, driven
# by a motor: every member of the passive joint-force family is
# configured, so each reads non-zero rather than "zero because the model
# configures none of it". Contacts are off (contype/conaffinity=0) so
# cfrc_int carries the parent-child interaction alone.
_PENDULUM_XML = """
<mujoco>
  <option gravity="0 0 -9.81" density="1000" viscosity="0.5"/>
  <worldbody>
    <body name="mount" pos="0 0 1">
      <geom name="mount_geom" type="box" size="0.02 0.02 0.02" mass="0.01"
            contype="0" conaffinity="0"/>
      <body name="link" pos="0 0 0" gravcomp="0.5">
        <joint name="hinge" type="hinge" axis="0 1 0" stiffness="5"
               damping="1" springref="0.5"/>
        <geom name="link_geom" type="capsule" fromto="0 0 0 0.3 0 0"
              size="0.02" mass="1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge"/>
  </actuator>
</mujoco>
"""


class _SteppedEnv:
    """Env whose scene has already been advanced by ``steps`` mj_step calls."""

    def __init__(self, xml: str, steps: int) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def step(self, count: int) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)


def _resting_block(xml: str = _RESTING_XML, steps: int = 300):
    env = _SteppedEnv(xml, steps)
    adapter = MujocoAdapter(
        env,
        body_rules=[BodyRule(match="block", cls="object")],
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    return adapter, env, entity


def _pendulum(steps: int):
    env = _SteppedEnv(_PENDULUM_XML, steps)
    adapter = MujocoAdapter(
        env,
        body_rules=[
            BodyRule(
                match="mount",
                cls="robot",
                joint_scope="subtree",
            ),
        ],
    )
    observation = adapter.reset()
    entity = next(iter(observation.perceived_entities.values()))
    return adapter, env, entity


# ---------------------------------------------------------------------------
# cfrc_ext: the load-bearing regression test for the whole feature
# ---------------------------------------------------------------------------

def test_resting_body_reports_its_weight_without_a_declared_sensor():
    """
    The regression test for issue #76: on a model with no sensor to
    trigger rnePostConstraint as a side effect, a settled 1 kg body must
    report 9.81 N of support at cfrc_ext index 5. This fails -- reading
    exactly 0.0 -- whenever the adapter stops making the call itself.
    """
    _, _, block = _resting_block()

    assert block.properties["cfrc_ext_block_5"] == pytest.approx(
        9.81,
        rel=1e-3,
    )


def test_declared_force_sensor_agrees_with_the_unconditional_call():
    """
    MuJoCo already ran rnePostConstraint for the declared sensor, so the
    adapter's own call is redundant here. Redundant must mean identical,
    not merely non-zero: rnePostConstraint is idempotent on an unchanged
    state, and this is what says so.
    """
    _, _, without_sensor = _resting_block(_RESTING_XML)
    _, _, with_sensor = _resting_block(_RESTING_SENSOR_XML)

    assert with_sensor.properties["cfrc_ext_block_5"] == pytest.approx(
        without_sensor.properties["cfrc_ext_block_5"],
        rel=1e-6,
    )


def test_free_body_carries_no_internal_interaction_force():
    # cfrc_int is the interaction with the *parent* body; the block's
    # parent is the world, which pushes on it through no joint at all.
    # Not a silent-zeros risk -- the same observation's cfrc_ext is
    # non-zero, so the call demonstrably ran.
    _, _, block = _resting_block()

    assert block.properties["cfrc_ext_block_5"] > 9.0
    assert block.properties["cfrc_int_block_5"] == pytest.approx(0.0, abs=1e-6)


def test_resting_body_reports_proper_acceleration_not_zero():
    """
    ``cacc`` is proper (accelerometer-style) acceleration: MuJoCo seeds
    the world body with -gravity, so a body at rest on the ground reads
    +9.81 m/s^2 in z rather than 0. Asserting the +9.81 is what makes
    this a non-zero test.
    """
    _, _, block = _resting_block()

    assert block.properties["cacc_block_5"] == pytest.approx(9.81, rel=1e-3)


def test_observation_reflects_the_state_the_caller_last_stepped_to():
    # The call is made at observation time, not at reset time: dropping
    # the block from rest and re-observing must change cfrc_ext, which it
    # cannot do if the value were cached from the reset observation.
    adapter, env, block = _resting_block()

    supported = block.properties["cfrc_ext_block_5"]
    assert supported > 9.0

    # lift the block clear of the floor and let it fall freely
    env.data.qpos[2] = 3.0
    env.data.qvel[:] = 0.0
    env.step(10)

    falling = adapter.get_observation()
    entity = next(iter(falling.perceived_entities.values()))

    assert entity.properties["cfrc_ext_block_5"] == pytest.approx(0.0, abs=1e-6)
    # in free fall the proper acceleration is zero, unlike at rest
    assert entity.properties["cacc_block_5"] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# cfrc_int on an articulated body
# ---------------------------------------------------------------------------

def test_parent_body_carries_the_child_weight_through_cfrc_int():
    # Settled pendulum: the mount holds the 1 kg link up, so the
    # interaction force with the parent is the link's weight.
    _, _, robot = _pendulum(steps=4000)

    assert robot.properties["cfrc_int_link_5"] == pytest.approx(
        9.81,
        rel=1e-3,
    )
    # no contacts in this model, so nothing external touches the link
    assert robot.properties["cfrc_ext_link_5"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Passive joint-force family
# ---------------------------------------------------------------------------

def test_every_passive_signal_reads_non_zero_when_configured():
    # 30 steps in: still swinging, so the damper and fluid terms have not
    # decayed to zero the way they do at rest.
    _, _, robot = _pendulum(steps=30)

    for signal in (
        "qfrc_passive",
        "qfrc_spring",
        "qfrc_damper",
        "qfrc_fluid",
        "qfrc_gravcomp",
    ):
        key = f"{signal}_hinge"
        assert key in robot.properties, key
        assert abs(robot.properties[key]) > 1e-3, (key, robot.properties[key])


def test_passive_components_sum_to_qfrc_passive():
    # qfrc_passive is the total; spring, damper, fluid and gravcomp are
    # its components. Asserting the sum catches a mis-wired address entry
    # that a per-signal non-zero check would not.
    _, _, robot = _pendulum(steps=30)

    total = (
        robot.properties["qfrc_spring_hinge"]
        + robot.properties["qfrc_damper_hinge"]
        + robot.properties["qfrc_fluid_hinge"]
        + robot.properties["qfrc_gravcomp_hinge"]
    )

    assert robot.properties["qfrc_passive_hinge"] == pytest.approx(
        total,
        rel=1e-6,
    )


# ---------------------------------------------------------------------------
# Actuator length / velocity
# ---------------------------------------------------------------------------

def test_actuator_length_and_velocity_track_the_transmitted_joint():
    """
    A unit-gear joint transmission transmits the joint coordinate
    directly, so these must *equal* the joint's own qpos/qvel -- a
    stronger statement than "non-zero", and one that fails if the
    actuator id and joint id are ever crossed.

    The comparison needs a self-consistent state. Like ``xpos`` and
    ``cvel``, ``actuator_length``/``actuator_velocity`` are written
    during a forward pass and are not recomputed by the integration at
    the end of ``mj_step``, so straight after a step they describe the
    configuration from before it (measured: 0.5855 against a stepped-to
    ``qvel`` of 0.5889). ``mj_forward`` brings the whole of MjData onto
    one configuration.
    """
    adapter, env, _ = _pendulum(steps=30)

    env.forward()
    observation = adapter.get_observation()
    robot = next(iter(observation.perceived_entities.values()))

    assert robot.properties["actuator_length_motor"] != 0.0
    assert robot.properties["actuator_velocity_motor"] != 0.0

    assert robot.properties["actuator_velocity_motor"] == pytest.approx(
        robot.properties["qvel_hinge"],
        rel=1e-9,
    )
    assert robot.properties["actuator_length_motor"] == pytest.approx(
        robot.properties["qpos_hinge"],
        rel=1e-9,
    )


# ---------------------------------------------------------------------------
# cvel's reference point, which README.md documents as a trap
# ---------------------------------------------------------------------------

# Two links on hinges, the first pinned to the world through its own
# origin: that pin makes the body-origin velocity exactly zero, which is
# what separates the three candidate readings of cvel's linear half.
_TWO_LINK_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
      <body name="tip" pos="0.3 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_cvel_linear_half_is_referenced_to_the_tree_root_subtree_com():
    """
    Pins the ``cvel`` trap README.md documents.

    ``cvel_<body>_3..5`` is the velocity of the body-fixed point
    currently coincident with ``subtree_com[body_rootid[body]]`` -- the
    *kinematic tree root's* subtree centre of mass, one shared point for
    every body in the tree. It is neither the body origin's velocity nor
    the body's own subtree-COM velocity, and the README said the latter
    until this test was written to settle it.

    Asserting the reference point rather than a magnitude is what makes
    this catch a future rewording: both wrong readings are numerically
    distinct here, and ``tip`` is the discriminating case, since its own
    subtree COM is not its tree root's.
    """
    model = mujoco.MjModel.from_xml_string(_TWO_LINK_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [0.4, -0.25]
    data.qvel[:] = [1.3, -0.7]
    mujoco.mj_forward(model, data)

    for name in ("base", "tip"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        reference = np.asarray(
            data.subtree_com[int(model.body_rootid[body])]
        )

        origin = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, body, origin, 0
        )
        angular, origin_linear = origin[:3], origin[3:]

        predicted = origin_linear + np.cross(
            angular, reference - np.asarray(data.xpos[body])
        )

        assert np.allclose(
            predicted, np.asarray(data.cvel[body])[3:], atol=1e-9
        ), name

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")

    # base's origin is pinned by its hinge, so the origin velocity is
    # exactly zero while cvel's linear half is not: cvel is not it.
    origin = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, base, origin, 0
    )
    assert np.allclose(origin[3:], 0.0, atol=1e-12)
    assert not np.allclose(np.asarray(data.cvel[base])[3:], 0.0, atol=1e-6)

    # and tip's own subtree COM is its inertial COM, whose velocity
    # differs from cvel's -- so cvel is not "the body's subtree COM"
    # either. mj_subtreeVel fills subtree_linvel with exactly that.
    mujoco.mj_subtreeVel(model, data)
    assert not np.allclose(
        np.asarray(data.subtree_linvel[tip]),
        np.asarray(data.cvel[tip])[3:],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Parity: every new key carries metadata
# ---------------------------------------------------------------------------

def test_metadata_parity_holds_over_the_widened_signal_set():
    adapter, _, robot = _pendulum(steps=30)
    meta = adapter.property_metadata(robot.entity_id)
    assert set(meta) == set(robot.properties)

    adapter, _, block = _resting_block()
    meta = adapter.property_metadata(block.entity_id)
    assert set(meta) == set(block.properties)
