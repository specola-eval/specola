"""Detached completion preserves the caller's native state and evolution."""

import copy

import numpy as np
import pytest

from specola.adapters.mujoco_scratch import MujocoScratchSession
from specola.conformance.query_bindings import QueryBindingError

pytestmark = pytest.mark.mujoco


@pytest.fixture
def subject():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body pos="0 0 2"><freejoint/>
      <geom type="sphere" size="0.1"/><site pos="0.2 0 0"/>
      <camera pos="0 0 1"/>
    </body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_completion_updates_a_detached_copy_and_preserves_stored_fields(subject):
    _, model, data = subject
    before = data.geom_xpos.copy()
    data.qpos[0] = 0.75
    with MujocoScratchSession(model, data) as session:
        with session.observe(data, completion=("mj_kinematics", "mj_comPos")) as snapshot:
            assert snapshot.source is data
            assert snapshot.data is not data
            assert snapshot.data.model is model
            assert snapshot.completed == ("mj_kinematics", "mj_comPos")
            np.testing.assert_array_equal(snapshot.data.qpos, data.qpos)
            np.testing.assert_array_equal(snapshot.data.geom_xpos[:, 0], [0.75])
            assert snapshot.warnings == ()
        with pytest.raises(QueryBindingError, match="borrow has expired"):
            snapshot.data
    np.testing.assert_array_equal(data.geom_xpos, before)
    assert data.qpos[0] == 0.75


def test_copy_and_completion_do_not_change_subsequent_evolution(subject):
    mujoco, model, data = subject
    data.qvel[:] = np.linspace(-0.4, 0.4, model.nv)
    control = copy.copy(data)
    with MujocoScratchSession(model, data) as session:
        with session.observe(data, completion=("mj_kinematics", "mj_comPos", "mj_comVel", "mj_camlight")):
            pass
    mujoco.mj_step(model, data)
    mujoco.mj_step(model, control)
    np.testing.assert_array_equal(data.qpos, control.qpos)
    np.testing.assert_array_equal(data.qvel, control.qvel)
    np.testing.assert_array_equal(data.qacc_warmstart, control.qacc_warmstart)
    assert data.time == control.time


@pytest.mark.parametrize("completion", (("mj_forward",), ("mj_comVel",), ("mj_kinematics", "mj_kinematics")))
def test_completion_requires_a_supported_dependency_order_before_copy(subject, completion):
    _, model, data = subject
    with MujocoScratchSession(model, data) as session:
        with pytest.raises(QueryBindingError, match="completion"):
            with session.observe(data, completion=completion):
                pytest.fail("invalid completion executed")


def test_sessions_reject_same_thread_reentry(subject):
    _, model, data = subject
    with MujocoScratchSession(model, data) as first, MujocoScratchSession(model, data) as second:
        with first.observe(data):
            with pytest.raises(QueryBindingError, match="reentrant"):
                with second.observe(data):
                    pytest.fail("reentrant native interception executed")


def test_a_different_model_data_pair_cannot_share_a_session(subject):
    mujoco, model, data = subject
    foreign_model = mujoco.MjModel.from_xml_string("<mujoco/>")
    foreign = mujoco.MjData(foreign_model)
    with MujocoScratchSession(model, data) as session:
        with pytest.raises(QueryBindingError, match="model identity"):
            with session.observe(foreign):
                pytest.fail("foreign data was copied")
