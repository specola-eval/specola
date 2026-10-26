"""Live parity checks for RoboCasa fixture identity and native state."""

from __future__ import annotations

import numpy as np
import pytest

from specola.adapters.mujoco import _property_token
from specola.adapters.robocasa import RoboCasaAdapter

# Same pin as benchmarks/extraction_profile.py's default (--seed 0
# --layout 1 --style 1): layout_ids/style_ids alone still leave
# env.rng free to redraw which fixture instance a task gets (e.g. the
# stove class asserted below), so the rng is reseeded again right
# before the reset -- see that module's docstring for the measured
# evidence.
SEED = 0
LAYOUT = 1
STYLE = 1


def _make_env(task_name):
    pytest.importorskip("robocasa")
    import robosuite as suite

    env = suite.make(
        env_name=task_name,
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        seed=SEED,
        layout_ids=LAYOUT,
        style_ids=STYLE,
    )
    env.rng = np.random.default_rng(SEED)
    env.reset()
    adapter = RoboCasaAdapter(env)
    adapter.reset()
    return env, adapter


@pytest.fixture(scope="module")
def drawer():
    env, adapter = _make_env("OpenDrawer")
    yield env, adapter
    env.close()


@pytest.fixture(scope="module")
def stove():
    env, adapter = _make_env("TurnOnStove")
    yield env, adapter
    env.close()


@pytest.fixture(scope="module")
def pick_place():
    env, adapter = _make_env("PickPlaceCounterToCabinet")
    yield env, adapter
    env.close()


def _body_record(adapter, root):
    return next(
        body for body in adapter._bodies
        if body.body_name == root
        and body.entity_class == "fixture"
    )


@pytest.mark.robocasa
def test_drawer_joint_qpos_matches_native_mjdata(drawer):
    env, adapter = drawer
    record = _body_record(adapter, env.drawer.root_body)
    assert record.joint_ids

    observation = adapter.get_observation()
    properties = observation.perceived_entities[
        record.entity_id
    ].properties
    for joint_id in record.joint_ids:
        joint_name = adapter._id2name("joint", joint_id)
        start = int(adapter._model.jnt_qposadr[joint_id])
        key = "qpos_{}".format(_property_token(joint_name))
        assert properties[key] == pytest.approx(
            float(adapter._data.qpos[start])
        )


@pytest.mark.robocasa
def test_drawer_region_follows_live_compiled_geom(drawer):
    env, adapter = drawer
    geom_name = env.drawer._regions["int"]["elem"].get("name")
    region = next(
        record for record in adapter._regions
        if record.spec.source_name == geom_name
    )
    fixture = _body_record(adapter, env.drawer.root_body)
    joint_id = fixture.joint_ids[0]
    qpos_address = int(adapter._model.jnt_qposadr[joint_id])
    original = float(adapter._data.qpos[qpos_address])

    before = adapter.get_observation().perceived_entities[
        region.entity_id
    ].position.copy()
    try:
        adapter._data.qpos[qpos_address] = original + 0.05
        env.sim.forward()
        after = adapter.get_observation().perceived_entities[
            region.entity_id
        ].position
        geom_id = adapter._name2id("geom", geom_name)
        assert after == pytest.approx(adapter._data.geom_xpos[geom_id])
        assert np.linalg.norm(after - before) == pytest.approx(0.05)
    finally:
        adapter._data.qpos[qpos_address] = original
        env.sim.forward()


@pytest.mark.robocasa
def test_unregistered_task_stove_still_has_native_joint_state(stove):
    env, adapter = stove
    # TurnOnStove uses get_fixture() directly, so this proves physical
    # scene declarations do not depend on fixture_refs completeness.
    assert not env.fixture_refs
    record = _body_record(adapter, env.stove.root_body)
    selected_joint_name = env.stove.knob_joints[env.knob].get("name")
    selected_joint_id = adapter._name2id("joint", selected_joint_name)
    assert selected_joint_id in record.joint_ids

    properties = adapter.get_observation().perceived_entities[
        record.entity_id
    ].properties
    qpos_address = int(adapter._model.jnt_qposadr[selected_joint_id])
    key = "qpos_{}".format(_property_token(selected_joint_name))
    assert properties[key] == pytest.approx(
        float(adapter._data.qpos[qpos_address])
    )
    assert adapter.entity_fixture_types[record.entity_id] == "Stove"


@pytest.mark.robocasa
def test_pick_place_gripper_uses_live_end_effector_site(pick_place):
    _env, adapter = pick_place
    assert adapter._gripper is not None
    assert adapter._gripper.site_name == "gripper0_right_grip_site"
    properties = adapter.get_observation().perceived_entities[
        adapter._gripper.entity_id
    ].properties
    assert "gripper_qpos_abs_mean" in properties
    assert {
        "site_xpos_gripper0_right_grip_site_0",
        "site_xpos_gripper0_right_grip_site_1",
        "site_xpos_gripper0_right_grip_site_2",
    } <= set(properties)
