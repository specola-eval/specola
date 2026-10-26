"""
Tests for the PyBullet adapter (issue #33) — the first non-MuJoCo engine
integration, implemented directly against SimulatorAdapter.

All tests run headless (DIRECT client) against tiny inline URDF fixtures,
so no display and no asset downloads are needed. The suite is skipped
loudly when pybullet is not installed (run inside the `pybullet` conda
env).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

pytest.importorskip(
    "pybullet",
    reason="PyBullet adapter suite needs pybullet; pip install pybullet",
)
import pybullet as p

import specola.adapters.pybullet as pybullet_module
from specola.adapters.pybullet import PyBulletAdapter
from specola.registry import resolve
from specola.spec import parser as _spec_parser
from specola.spec.selectors import resolve_selector

pytestmark = pytest.mark.pybullet


ROBOT_URDF = """
<robot name="testbot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="arm_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="slide_link">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="tool_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="shoulder_pivot" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="rail_slide" type="prismatic">
    <parent link="arm_link"/>
    <child link="slide_link"/>
    <origin xyz="0.1 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" effort="10" velocity="1"/>
  </joint>
  <joint name="tool_mount" type="fixed">
    <parent link="slide_link"/>
    <child link="tool_link"/>
    <origin xyz="0.05 0 0"/>
  </joint>
</robot>
"""

BOX_URDF = """
<robot name="loose-box">
  <link name="box_body">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
</robot>
"""


@pytest.fixture()
def client():
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(physicsClientId=cid)


@pytest.fixture()
def robot_urdf(tmp_path):
    path = tmp_path / "testbot.urdf"
    path.write_text(ROBOT_URDF)
    return str(path)


@pytest.fixture()
def box_urdf(tmp_path):
    path = tmp_path / "box.urdf"
    path.write_text(BOX_URDF)
    return str(path)


def _scene(client, robot_urdf, box_urdf):
    robot_uid = p.loadURDF(
        robot_urdf, useFixedBase=True, physicsClientId=client
    )
    box_uid = p.loadURDF(
        box_urdf, basePosition=[1.0, 0.0, 0.5], physicsClientId=client
    )
    return robot_uid, box_uid


def _spy_native_reads(adapter):
    """Count snapshot-native reads without changing the PyBullet client."""

    counts = Counter()
    delegate = adapter._p
    names = {
        "getBasePositionAndOrientation",
        "getBaseVelocity",
        "getBodyInfo",
        "getContactPoints",
        "getJointState",
        "getLinkState",
    }

    class Spy:
        def __getattr__(self, name):
            value = getattr(delegate, name)
            if name not in names:
                return value

            def wrapped(*args, **kwargs):
                counts[name] += 1
                return value(*args, **kwargs)

            return wrapped

    adapter._p = Spy()
    return counts


def _filtered_properties(observation, scope):
    return {
        entity_id: {
            key: value
            for key, value in entity.properties.items()
            if key in scope.get(entity.entity_class, frozenset())
        }
        for entity_id, entity in observation.perceived_entities.items()
    }


# ---------------------------------------------------------------------------
# Discovery and classification
# ---------------------------------------------------------------------------

def test_reset_discovers_bodies_with_default_classes(
    client, robot_urdf, box_urdf
):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()

    assert set(obs.perceived_entities) == {robot_uid, box_uid}
    # Default rule: articulated bodies are robots, jointless bodies objects.
    assert obs.perceived_entities[robot_uid].entity_class == "robot"
    assert obs.perceived_entities[box_uid].entity_class == "object"
    assert adapter.entity_labels[robot_uid] == "testbot"
    assert adapter.entity_labels[box_uid] == "loose-box"


def test_body_rules_override_default_classification(
    client, robot_urdf, box_urdf
):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(
        client, body_rules=[("loose-*", "cargo"), ("test*", "manipulator")]
    )
    obs = adapter.reset()

    assert obs.perceived_entities[robot_uid].entity_class == "manipulator"
    assert obs.perceived_entities[box_uid].entity_class == "cargo"


def test_body_rules_match_sim_name_not_override_label(client, robot_urdf):
    first = p.loadURDF(
        robot_urdf, useFixedBase=True, physicsClientId=client
    )
    second = p.loadURDF(
        robot_urdf,
        basePosition=[2.0, 0.0, 0.0],
        useFixedBase=True,
        physicsClientId=client,
    )

    # Disambiguating a duplicate name must not change classification:
    # both copies match the rule via their sim-reported name "testbot".
    adapter = PyBulletAdapter(
        client,
        body_rules=[("testbot", "manipulator")],
        body_labels={second: "testbot_b"},
    )
    obs = adapter.reset()

    assert obs.perceived_entities[first].entity_class == "manipulator"
    assert obs.perceived_entities[second].entity_class == "manipulator"


def test_supported_classes_closed_and_includes_rule_classes(
    client, robot_urdf, box_urdf
):
    _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client, body_rules=[("loose-*", "cargo")])
    assert adapter.supported_classes == frozenset(
        {"cargo", "robot", "object"}
    )


# ---------------------------------------------------------------------------
# Sim-native property vocabulary
# ---------------------------------------------------------------------------

def test_joint_keys_follow_sim_native_naming(client, robot_urdf, box_urdf):
    robot_uid, _ = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()
    props = obs.perceived_entities[robot_uid].properties

    for prefix in ("jointpos", "jointvel", "jointtorque"):
        assert f"{prefix}_shoulder_pivot" in props
        assert f"{prefix}_rail_slide" in props
        # Fixed joints carry no state.
        assert f"{prefix}_tool_mount" not in props


def test_base_link_and_contact_keys(client, robot_urdf, box_urdf):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()

    robot_props = obs.perceived_entities[robot_uid].properties
    box_props = obs.perceived_entities[box_uid].properties
    # Per-entity quantities are entity-generic: every body exposes the
    # same base-tier vocabulary regardless of its name.
    for props in (robot_props, box_props):
        for axis in range(3):
            assert f"basepos_{axis}" in props
            assert f"basevel_{axis}" in props
            assert f"baseangvel_{axis}" in props
        for axis in range(4):
            assert f"basequat_{axis}" in props
        assert "contactforce" in props

    # Links named in the URDF, including fixed-joint children.
    for axis in range(3):
        assert f"linkpos_arm_link_{axis}" in robot_props
        assert f"linkpos_slide_link_{axis}" in robot_props
        assert f"linkpos_tool_link_{axis}" in robot_props
    assert not any(key.startswith("jointpos_") for key in box_props)


def test_base_position_matches_pybullet_state(client, robot_urdf, box_urdf):
    _, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()
    entity = obs.perceived_entities[box_uid]

    raw_pos, _ = p.getBasePositionAndOrientation(
        box_uid, physicsClientId=client
    )
    assert np.allclose(entity.position, raw_pos)
    assert np.allclose(
        [entity.properties[f"basepos_{i}"] for i in range(3)],
        raw_pos,
    )


def test_scoped_snapshot_matches_full_projection_and_prunes_native_reads(
    client, robot_urdf, box_urdf
):
    """Mixed classes retain canonical vectors while reading only kept keys."""

    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)
    full = PyBulletAdapter(client)
    scoped = PyBulletAdapter(client)
    full.reset()
    scoped.reset()
    scope = {
        "robot": frozenset({
            "basepos_0",
            "jointpos_shoulder_pivot",
            "linkpos_arm_link_2",
        }),
        "object": frozenset({"basevel_1"}),
    }
    scoped.set_extraction_scope(scope)

    expected = full.get_observation()
    reads = _spy_native_reads(scoped)
    actual = scoped.get_observation()

    assert {
        entity_id: entity.properties
        for entity_id, entity in actual.perceived_entities.items()
    } == _filtered_properties(expected, scope)
    for entity_id in (robot_uid, box_uid):
        assert np.array_equal(
            actual.perceived_entities[entity_id].position,
            expected.perceived_entities[entity_id].position,
        )
        assert np.array_equal(
            actual.perceived_entities[entity_id].velocity,
            expected.perceived_entities[entity_id].velocity,
        )
        assert set(scoped.property_metadata(entity_id)) == set(
            actual.perceived_entities[entity_id].properties
        )

    assert reads["getBodyInfo"] == 2
    assert reads["getBasePositionAndOrientation"] == 2
    assert reads["getBaseVelocity"] == 2
    assert reads["getContactPoints"] == 0
    assert reads["getJointState"] == 1
    assert reads["getLinkState"] == 1


def test_contact_read_occurs_only_for_a_live_class_requesting_it(
    client, robot_urdf, box_urdf
):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)
    adapter = PyBulletAdapter(client)
    adapter.reset()
    adapter.set_extraction_scope({
        "robot": frozenset({"contactforce"}),
        "object": frozenset(),
    })

    reads = _spy_native_reads(adapter)
    observation = adapter.get_observation()

    assert set(observation.perceived_entities[robot_uid].properties) == {
        "contactforce"
    }
    assert observation.perceived_entities[box_uid].properties == {}
    assert reads["getContactPoints"] == 1
    assert reads["getJointState"] == 0
    assert reads["getLinkState"] == 0


def test_none_scope_keeps_full_projection_and_native_read_denominator(
    client, robot_urdf, box_urdf
):
    _scene(client, robot_urdf, box_urdf)
    untouched = PyBulletAdapter(client)
    explicit_none = PyBulletAdapter(client)
    untouched.reset()
    explicit_none.reset()
    explicit_none.set_extraction_scope(None)

    expected = untouched.get_observation()
    reads = _spy_native_reads(explicit_none)
    actual = explicit_none.get_observation()

    assert {
        entity_id: entity.properties
        for entity_id, entity in actual.perceived_entities.items()
    } == {
        entity_id: entity.properties
        for entity_id, entity in expected.perceived_entities.items()
    }
    for entity_id, entity in actual.perceived_entities.items():
        reference = expected.perceived_entities[entity_id]
        assert np.array_equal(entity.position, reference.position)
        assert np.array_equal(entity.velocity, reference.velocity)
        assert explicit_none.property_metadata(entity_id) == (
            untouched.property_metadata(entity_id)
        )
    assert reads["getContactPoints"] == 1
    assert reads["getJointState"] == 2
    assert reads["getLinkState"] == 3
    assert reads["getBasePositionAndOrientation"] == 2
    assert reads["getBaseVelocity"] == 2


# ---------------------------------------------------------------------------
# property_metadata parity and units
# ---------------------------------------------------------------------------

def test_observation_wide_metadata_parity(client, robot_urdf, box_urdf):
    _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()

    for entity_id, entity in obs.perceived_entities.items():
        meta = adapter.property_metadata(entity_id)
        assert set(meta) == set(entity.properties), (
            f"metadata parity broken for entity {entity_id}"
        )


def test_exact_fixed_scene_consistent_si_metadata_semantics(
    client, robot_urdf, box_urdf
):
    _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    observation = adapter.reset()
    by_class = {
        entity.entity_class: adapter.property_metadata(entity_id)
        for entity_id, entity in observation.perceived_entities.items()
    }
    actual = {
        f"{entity_class}.{key}": (
            meta.unit, meta.frame, meta.group, meta.note
        )
        for entity_class, metadata in by_class.items()
        for key, meta in metadata.items()
    }
    classes = ("robot", "object")
    world = "world Cartesian"
    joint = "parent-local joint axis"
    links = ("arm_link", "slide_link", "tool_link")
    basepos = ("m", world, "body position", "m")
    basevel = ("m/s", world, "body linear velocity", "m/s")
    baseangvel = ("rad/s", world, "body angular velocity", "rad/s")
    quaternion = tuple(
        ("1", "active base-local-to-world (x, y, z, w)",
         "body orientation", f"quat {axis}") for axis in "xyzw"
    )
    contact = ("N", "frame-free scalar per-body aggregate",
               "contact normal force", "N")
    linkpos = ("m", world, "link position", "m (world COM)")
    joints = {
        "jointpos_shoulder_pivot": ("rad", joint, "joint angle", "rad"),
        "jointvel_shoulder_pivot": ("rad/s", joint, "joint angular velocity", "rad/s"),
        "jointtorque_shoulder_pivot": ("N·m", joint, "applied torques", "N·m"),
        "jointpos_rail_slide": ("m", joint, "joint displacement", "m"),
        "jointvel_rail_slide": ("m/s", joint, "joint linear velocity", "m/s"),
        "jointtorque_rail_slide": ("N", joint, "applied forces", "N"),
    }
    expected = {
        **{f"{c}.basepos_{a}": basepos for c in classes for a in range(3)},
        **{f"{c}.basevel_{a}": basevel for c in classes for a in range(3)},
        **{f"{c}.baseangvel_{a}": baseangvel for c in classes for a in range(3)},
        **{f"{c}.basequat_{a}": quaternion[a] for c in classes for a in range(4)},
        **{f"{c}.contactforce": contact for c in classes},
        **{f"robot.linkpos_{x}_{a}": linkpos for x in links for a in range(3)},
        **{f"robot.{key}": value for key, value in joints.items()},
    }
    assert len(actual) == len(expected) == 43
    assert actual == expected

    robot = by_class["robot"]
    selector = getattr(
        _spec_parser.parse("robot_*[* @ link position] <= 0"), "references"
    )[0].selector
    groups = {key: meta.group for key, meta in robot.items()}
    assert len(resolve_selector(selector, frozenset(robot), groups)) == 9
    module_doc = pybullet_module.__doc__ or ""
    assert "PyBullet does not globally mandate SI" in module_doc
    assert "Specola's explicit consistent-SI declaration" in module_doc


# ---------------------------------------------------------------------------
# Multi-instance scenes and display labels
# ---------------------------------------------------------------------------

def test_same_urdf_twice_exposes_identical_vocabulary(client, robot_urdf):
    first = p.loadURDF(
        robot_urdf, useFixedBase=True, physicsClientId=client
    )
    second = p.loadURDF(
        robot_urdf,
        basePosition=[2.0, 0.0, 0.0],
        useFixedBase=True,
        physicsClientId=client,
    )

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()

    # Two copies of one model are two entities with the SAME property
    # vocabulary — a wildcard spec binds both without configuration.
    first_props = obs.perceived_entities[first].properties
    second_props = obs.perceived_entities[second].properties
    assert set(first_props) == set(second_props)
    assert obs.perceived_entities[first].entity_class == "robot"
    assert obs.perceived_entities[second].entity_class == "robot"

    # Display labels are disambiguated so bindings stay tellable apart;
    # property keys are unaffected.
    assert adapter.entity_labels[first] == "testbot"
    assert adapter.entity_labels[second] == f"testbot#{second}"


def test_body_labels_are_display_only(client, robot_urdf, box_urdf):
    robot_uid, _ = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(
        client, body_labels={robot_uid: "left arm"}
    )
    obs = adapter.reset()

    assert adapter.entity_labels[robot_uid] == "left arm"
    # The override changes no property key.
    props = obs.perceived_entities[robot_uid].properties
    assert "basepos_0" in props
    assert not any("left" in key for key in props)


# ---------------------------------------------------------------------------
# Timebase and stepping ownership
# ---------------------------------------------------------------------------

def test_dt_read_from_engine_fixed_timestep(client, robot_urdf, box_urdf):
    _scene(client, robot_urdf, box_urdf)
    p.setTimeStep(0.01, physicsClientId=client)

    adapter = PyBulletAdapter(client)
    assert adapter.dt == pytest.approx(0.01)


def test_observations_advance_timestep_without_stepping_sim(
    client, robot_urdf, box_urdf
):
    _, box_uid = _scene(client, robot_urdf, box_urdf)
    # Gravity makes the free-floating box non-stationary the moment
    # anything steps the engine, so a regression that steps from inside
    # the adapter moves the box and fails the frozen-state assertion.
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)

    adapter = PyBulletAdapter(client)
    first = adapter.reset()
    second = adapter.get_observation()
    third = adapter.get_observation()

    assert first.timestep < second.timestep < third.timestep
    assert first.time <= second.time <= third.time

    # The adapter never steps the simulation: without an external
    # stepSimulation() the scene state is frozen.
    assert np.allclose(
        second.perceived_entities[box_uid].position,
        third.perceived_entities[box_uid].position,
    )

    # And an external step does move the box, proving the assertion
    # above is not vacuously true.
    p.stepSimulation(physicsClientId=client)
    after = adapter.get_observation()
    assert not np.allclose(
        third.perceived_entities[box_uid].position,
        after.perceived_entities[box_uid].position,
    )


def test_contact_force_aggregates_after_step(client, box_urdf, tmp_path):
    # Two overlapping boxes produce a contact once the engine steps.
    other = tmp_path / "box2.urdf"
    other.write_text(BOX_URDF.replace("loose-box", "other-box"))

    a = p.loadURDF(
        box_urdf, basePosition=[0.0, 0.0, 0.0], physicsClientId=client
    )
    b = p.loadURDF(
        str(other), basePosition=[0.0, 0.0, 0.08], physicsClientId=client
    )

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()
    assert obs.perceived_entities[a].properties["contactforce"] \
        == pytest.approx(0.0)

    p.stepSimulation(physicsClientId=client)
    obs = adapter.get_observation()

    assert obs.perceived_entities[a].properties["contactforce"] > 0
    assert obs.perceived_entities[b].properties["contactforce"] > 0


def test_removed_body_disappears_instead_of_crashing(
    client, robot_urdf, box_urdf
):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    obs = adapter.reset()
    assert box_uid in obs.perceived_entities

    p.removeBody(box_uid, physicsClientId=client)
    obs = adapter.get_observation()

    assert box_uid not in obs.perceived_entities
    assert robot_uid in obs.perceived_entities


def test_reused_uid_with_different_name_stays_absent(
    client, robot_urdf, box_urdf, tmp_path
):
    _, box_uid = _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    adapter.reset()

    # PyBullet reuses the freed uid for the next load; the replacement
    # body has a different name and must not surface under the old
    # entity's identity.
    p.removeBody(box_uid, physicsClientId=client)
    other = tmp_path / "impostor.urdf"
    other.write_text(BOX_URDF.replace("loose-box", "impostor-box"))
    reused = p.loadURDF(
        str(other), basePosition=[9.0, 9.0, 9.0], physicsClientId=client
    )

    obs = adapter.get_observation()
    if reused == box_uid:
        assert box_uid not in obs.perceived_entities
    else:  # engine did not reuse the uid; absence still required
        assert box_uid not in obs.perceived_entities


def test_make_advancer_refuses_replay_random_and_demo(client, robot_urdf, box_urdf):
    # Issue #132: a mode or flag the adapter does not implement is refused,
    # never accepted and ignored.
    _scene(client, robot_urdf, box_urdf)

    adapter = PyBulletAdapter(client)
    for mode in ("replay", "random"):
        with pytest.raises(SystemExit, match=f"--mode {mode} is not supported by the pybullet adapter"):
            adapter.make_advancer(mode, None, None)
    for mode in ("replay", "random", "hold"):
        with pytest.raises(SystemExit, match="The pybullet adapter has no --demo replay"):
            adapter.make_advancer(mode, "demo.hdf5", None)


def test_hold_steps_physics_and_applies_no_control(client):
    adapter = PyBulletAdapter(client)
    stepped = []

    class _OnlyStepping:
        # Any other engine call (setJointMotorControl*, applyExternalForce)
        # would raise AttributeError here.
        def stepSimulation(self, physicsClientId):
            stepped.append(physicsClientId)

    adapter._p = _OnlyStepping()
    advance = adapter.make_advancer("hold", None, None)

    assert advance(0) is True and advance(1) is True
    assert stepped == [client, client]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_registered_factory_requires_task():
    entry = resolve("pybullet")
    with pytest.raises(ValueError, match="--task"):
        entry.factory(None, None, True)


def test_factory_loads_urdf_scene_and_owns_stepping(robot_urdf):
    entry = resolve("pybullet")
    adapter = entry.factory(robot_urdf, None, True)
    try:
        obs = adapter.reset()
        assert len(obs.perceived_entities) == 1

        advance = adapter.make_advancer("hold", None, None)
        assert advance(0) is True
        after = adapter.get_observation()
        assert after.timestep == obs.timestep + 1
    finally:
        adapter.close()


def test_egl_stays_off_without_optin(client, monkeypatch):
    # The EGL plugin can exit() the whole process on a broken EGL stack,
    # so hardware rendering must never engage without PYBULLET_EGL set.
    from specola.adapters.pybullet import _maybe_load_egl

    calls = []
    fake_pb = type(
        "P", (), {"loadPlugin": lambda *a, **k: calls.append(a)})()
    monkeypatch.delenv("PYBULLET_EGL", raising=False)
    assert _maybe_load_egl(fake_pb, client) == -1
    monkeypatch.setenv("PYBULLET_EGL", "0")
    assert _maybe_load_egl(fake_pb, client) == -1
    monkeypatch.setenv("PYBULLET_EGL", "false")
    assert _maybe_load_egl(fake_pb, client) == -1
    assert calls == []


def test_factory_loads_egl_before_bodies(monkeypatch, tmp_path):
    # Regression: the EGL renderer only sees bodies loaded after the
    # plugin, so the factory must load it before any loadURDF — plugin
    # loaded afterwards rendered blank frames (2026-07-31).
    import specola.adapters.pybullet as mod

    order = []
    monkeypatch.setenv("PYBULLET_EGL", "1")
    monkeypatch.setattr(
        mod, "_maybe_load_egl",
        lambda pb, client: order.append("egl") or 7)
    real_load = p.loadURDF
    monkeypatch.setattr(
        p, "loadURDF",
        lambda *a, **k: order.append("urdf") or real_load(*a, **k))
    urdf = tmp_path / "box.urdf"
    urdf.write_text(BOX_URDF)
    adapter = mod._make_pybullet(str(urdf), None, False)
    try:
        assert order[0] == "egl"
        assert adapter._egl_plugin == 7
    finally:
        p.disconnect(physicsClientId=adapter._client)


# ---------------------------------------------------------------------------
# State tier (replay)
# ---------------------------------------------------------------------------

_INERTIAL = (
    '<inertial><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" '
    'iyy="0.01" iyz="0" izz="0.01"/></inertial>'
)

SPHERICAL_URDF = f"""
<robot name="ball-arm">
  <link name="mount">{_INERTIAL}</link>
  <link name="bob">{_INERTIAL}</link>
  <joint name="ball" type="spherical">
    <parent link="mount"/><child link="bob"/><origin xyz="0 0 -0.3"/>
  </joint>
</robot>
"""

PLANAR_URDF = f"""
<robot name="slider">
  <link name="mount">{_INERTIAL}</link>
  <link name="slab">{_INERTIAL}</link>
  <joint name="slab_joint" type="planar">
    <parent link="mount"/><child link="slab"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def _factory_signature(task):
    adapter = pybullet_module._make_pybullet(task, None, False)
    try:
        return adapter.state_signature()
    finally:
        adapter.close()


def test_same_urdf_twice_gives_one_state_signature(robot_urdf, box_urdf):
    import importlib.metadata

    task = f"{robot_urdf}@fixed,{box_urdf}"
    first = _factory_signature(task)
    assert _factory_signature(task) == first
    assert first["package"] == importlib.metadata.version("pybullet")
    robot, box = first["bodies"]
    assert robot["fixed_base"] is True and box["fixed_base"] is False
    assert robot["urdf_sha256"] != box["urdf_sha256"]
    # 13 base values per body, position and velocity of the two movable joints.
    assert first["length"] == 2 * 13 + 2 * 2


def test_a_changed_collision_box_changes_the_state_signature(box_urdf, tmp_path):
    changed = tmp_path / "box_changed.urdf"
    changed.write_text(BOX_URDF.replace('size="0.1 0.1 0.1"', 'size="0.1 0.1 0.2"'))
    before, after = _factory_signature(box_urdf), _factory_signature(str(changed))
    assert before != after
    # The engine's own collision read differs, not only the file digest.
    assert before["bodies"][0]["links"] != after["bodies"][0]["links"]


def test_a_changed_link_transform_changes_the_state_signature(robot_urdf, tmp_path):
    changed = tmp_path / "testbot_changed.urdf"
    changed.write_text(ROBOT_URDF.replace('<origin xyz="0.1 0 0"/>', '<origin xyz="0.12 0 0"/>'))
    before = _factory_signature(f"{robot_urdf}@fixed")
    after = _factory_signature(f"{changed}@fixed")
    assert before != after
    assert before["bodies"][0]["joints"] != after["bodies"][0]["joints"]


def test_restore_brings_base_and_joint_state_back(client, robot_urdf, box_urdf, tmp_path):
    robot_uid, box_uid = _scene(client, robot_urdf, box_urdf)
    ball_path = tmp_path / "ball.urdf"
    ball_path.write_text(SPHERICAL_URDF)
    ball_uid = p.loadURDF(str(ball_path), basePosition=[-1.0, 0.0, 1.0],
                          useFixedBase=True, physicsClientId=client)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
    p.resetBaseVelocity(box_uid, [0.2, 0.0, 0.0], [3.0, -2.0, 7.0], physicsClientId=client)
    p.resetJointState(robot_uid, 0, 0.3, 0.5, physicsClientId=client)
    p.resetJointState(robot_uid, 1, 0.1, -0.2, physicsClientId=client)
    p.resetJointStateMultiDof(ball_uid, 0, [0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.2],
                              physicsClientId=client)

    adapter = PyBulletAdapter(client)
    signature = adapter.state_signature()
    assert [joint["state"] for joint in signature["bodies"][2]["joints"]] == [[4, 3]]
    for _ in range(20):
        p.stepSimulation(physicsClientId=client)
    saved = adapter.capture_state()
    assert saved.shape == (signature["length"],)
    for _ in range(20):
        p.stepSimulation(physicsClientId=client)
    assert not np.array_equal(adapter.capture_state(), saved)

    adapter.restore_state(saved)
    restored = adapter.capture_state()
    # PyBullet 3.2.7 reports a base quaternion through a rotation matrix,
    # so a restored one reads back rounded (the bound is the replay
    # harness's declared tolerance). Every other value comes back bit for
    # bit.
    quaternions, offset = [], 0
    for body in signature["bodies"]:
        quaternions.extend(range(offset + 3, offset + 7))
        offset += 13 + sum(sum(joint["state"]) for joint in body["joints"])
    others = np.setdiff1d(np.arange(saved.size), quaternions)
    assert np.array_equal(restored[others], saved[others])
    assert np.max(np.abs(restored[quaternions] - saved[quaternions])) <= 4 * np.spacing(1.0)

    with pytest.raises(ValueError, match=f"length {saved.size}"):
        adapter.restore_state(saved[:-1])


def test_contacts_after_restore_describe_the_restored_configuration(client, box_urdf):
    ground = p.createMultiBody(
        0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client),
        physicsClientId=client)
    box = p.loadURDF(box_urdf, basePosition=[0.3, -0.2, 0.5], physicsClientId=client)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
    adapter = PyBulletAdapter(client)
    in_air = adapter.capture_state()
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
    resting = adapter.capture_state()
    assert p.getContactPoints(box, ground, physicsClientId=client)

    adapter.restore_state(in_air)
    assert not p.getContactPoints(box, ground, physicsClientId=client)

    adapter.restore_state(resting)
    points = p.getContactPoints(box, ground, physicsClientId=client)
    assert points
    x, y = resting[13:15]          # the box is the second body
    for point in points:
        on_box, on_ground, distance = point[5], point[6], point[8]
        assert abs(on_box[0] - x) <= 0.051 and abs(on_box[1] - y) <= 0.051
        assert abs(on_ground[2]) < 1e-3 and abs(distance) < 1e-3


def test_a_planar_joint_has_no_state_tier(client, tmp_path):
    # PyBullet 3.2.7 reports two of a planar joint's three positions and
    # velocities, and a restore from them leaves the link elsewhere.
    path = tmp_path / "planar.urdf"
    path.write_text(PLANAR_URDF)
    p.loadURDF(str(path), useFixedBase=True, physicsClientId=client)
    adapter = PyBulletAdapter(client)
    assert adapter.state_signature() is None
    assert adapter.capture_state() is None
    with pytest.raises(NotImplementedError, match="planar"):
        adapter.restore_state(np.zeros(13))
