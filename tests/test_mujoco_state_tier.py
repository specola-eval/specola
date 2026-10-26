"""MuJoCo family state tier: capture, restore and signature, and the live
kinematics refresh, sim-free."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import MujocoAdapter  # noqa: E402

_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
    <body name="box" pos="0.2 0 0.3">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


def _stepped(adapter, model, data, steps=10):
    """Step with varying controls; per step the state, the live fields and fresh kinematics."""
    rng = np.random.default_rng(0)
    states, live, xpos = [], [], []
    for _ in range(steps):
        data.ctrl[:] = rng.uniform(-1, 1, model.nu)
        mujoco.mj_step(model, data)
        states.append(adapter.capture_state())
        live.append({name: np.copy(getattr(data, name)) for name in ("time", "qpos", "qvel", "ctrl")})
        fresh = mujoco.MjData(model)
        fresh.time, fresh.qpos[:], fresh.qvel[:] = data.time, data.qpos, data.qvel
        mujoco.mj_forward(model, fresh)
        xpos.append(fresh.xpos.copy())
    return states, live, xpos


def _assert_restored(data, live):
    for name, value in live.items():
        assert np.array_equal(getattr(data, name), value), name


def test_restore_reproduces_the_captured_state_and_its_kinematics():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=data))
    states, live, xpos = _stepped(adapter, model, data)
    assert all(s.dtype == np.float64 and s.ndim == 1 for s in states)
    assert not np.array_equal(states[4], states[9])

    # Each state owns its buffer; the data goes back from step 10 to step 5.
    adapter.restore_state(states[4])
    _assert_restored(data, live[4])
    assert not np.array_equal(data.qpos, live[9]["qpos"])
    # Kinematics describe the restored positions, as a fresh forward pass does
    # (mj_step leaves them at the positions before its integration).
    assert np.array_equal(data.xpos, xpos[4])


def test_restore_unwraps_robosuite_style_bindings_and_forwards_once():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    class _Binding:
        """robosuite.utils.binding_utils style: the raw struct at ``_model``/``_data``."""

        def __init__(self, attr, raw):
            object.__setattr__(self, attr, raw)
            object.__setattr__(self, "_raw", raw)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_raw"), name)

    class _Sim:
        def __init__(self):
            self.model = _Binding("_model", model)
            self.data = _Binding("_data", data)
            self.forwards = 0

        def forward(self):
            self.forwards += 1
            mujoco.mj_forward(model, data)

    sim = _Sim()
    adapter = MujocoAdapter(SimpleNamespace(sim=sim))
    states, live, xpos = _stepped(adapter, model, data, steps=6)
    adapter.restore_state(states[2])
    adapter.restore_state(states[3])
    assert sim.forwards == 2
    _assert_restored(data, live[3])
    assert np.array_equal(data.xpos, xpos[3])
    assert adapter.state_signature()["length"] == states[0].shape[0]


def test_signature_names_the_compiled_model():
    def signature(xml):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return MujocoAdapter(SimpleNamespace(model=model, data=data)).state_signature()

    first, again = signature(_SCENE), signature(_SCENE)
    assert first == again
    assert first["engine"] == "mujoco" and first["version"] == mujoco.__version__
    assert first["spec"] == "mjSTATE_INTEGRATION"
    # One geometry size changed: the same state length, another model.
    mutated = signature(_SCENE.replace('size="0.05 0.05 0.05"', 'size="0.05 0.05 0.06"'))
    assert mutated["length"] == first["length"]
    assert mutated["model_sha256"] != first["model_sha256"]

    model = mujoco.MjModel.from_xml_string(_SCENE)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=mujoco.MjData(model)))
    assert first["length"] == adapter.capture_state().shape[0]


def test_signature_leaves_out_appearance_only():
    # RoboCasa colours its counter tops from the global NumPy RNG, so one
    # seed builds kitchens that differ only in geom_rgba (2026-10-14).
    model = mujoco.MjModel.from_xml_string(_SCENE)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=mujoco.MjData(model)))
    before = adapter.state_signature()
    model.geom_rgba[1] = (0.9, 0.1, 0.2, 0.5)
    assert adapter.state_signature() == before
    # The digest works on a copy: the live colour is untouched.
    assert np.allclose(model.geom_rgba[1], (0.9, 0.1, 0.2, 0.5))
    model.geom_size[1, 0] += 0.01
    assert adapter.state_signature()["model_sha256"] != before["model_sha256"]


def test_restore_refuses_a_state_of_another_length():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=mujoco.MjData(model)))
    with pytest.raises(ValueError, match="length"):
        adapter.restore_state(np.zeros(3))


# A sprung, damped hinge driven through a joint and a tendon actuator, and a
# box resting on the floor: passive forces, a bias force and contact rows
# (efc_vel, efc_aref) are all nonzero after a step.
_LOADED = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.5" stiffness="2"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <site name="tip" pos="0.3 0 0"/>
    </body>
    <body name="box" pos="0.2 0 0.049">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <camera name="eye" pos="0 -1 0.5" xyaxes="1 0 0 0 0 1"/>
  </worldbody>
  <tendon>
    <fixed name="spool"><joint joint="hinge" coef="1"/></fixed>
  </tendon>
  <actuator>
    <motor name="drive" joint="hinge" gear="2"/>
    <motor name="pull" tendon="spool"/>
  </actuator>
</mujoco>
"""
_STEP_OUTPUT_PREFIXES = ("qfrc_", "qacc", "efc_", "cacc", "cfrc_", "actuator_force", "sensordata")


def test_a_live_observation_describes_one_instant_and_keeps_the_step_outputs():
    # mj_step leaves poses, velocities, the mass matrix and the transmission
    # at the positions it integrated from; the observation's refresh brings
    # them to qpos and qvel without touching anything the step computed.
    model = mujoco.MjModel.from_xml_string(_LOADED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=data))
    adapter.reset()
    for _ in range(5):
        data.ctrl[:] = (0.7, -0.3)
        mujoco.mj_step(model, data)
    arrays = [name for name in dir(data)
              if not name.startswith("_") and isinstance(getattr(data, name), np.ndarray)]
    # Arrays the step did not allocate (sparse solver and island rows) are
    # views of free arena memory, which any call with stack scratch
    # overwrites; mj_rne alone writes only its result and that scratch.
    before = {name: np.copy(getattr(data, name)) for name in arrays}
    mujoco.mj_rne(model, data, 0, np.zeros(model.nv))
    arrays = [name for name in arrays if np.array_equal(getattr(data, name), before[name])]
    before = {name: np.copy(getattr(data, name)) for name in arrays}
    state, ncon = adapter.capture_state(), data.ncon
    contact_dist = np.copy(data.contact.dist)
    fresh = copy.copy(data)
    mujoco.mj_forward(model, fresh)
    assert fresh.efc_vel.size and np.any(fresh.qfrc_passive) and np.any(fresh.qfrc_bias)

    adapter.get_observation()

    for name in ("xpos", "xmat", "site_xpos", "cam_xpos", "cvel", "cdof_dot", "qM",
                 "ten_length", "ten_velocity", "actuator_length", "actuator_velocity"):
        assert np.array_equal(getattr(data, name), getattr(fresh, name)), name
    for name in ("xpos", "cvel", "actuator_velocity", "ten_velocity"):
        assert not np.array_equal(before[name], getattr(fresh, name)), name   # the lag existed
    for name in arrays:
        now = getattr(data, name)
        if name.startswith(_STEP_OUTPUT_PREFIXES):
            assert np.array_equal(now, before[name]), name
        else:
            assert np.array_equal(now, before[name]) or np.array_equal(now, getattr(fresh, name)), name
    assert np.array_equal(adapter.capture_state(), state)
    assert data.ncon == ncon and np.array_equal(data.contact.dist, contact_dist)
