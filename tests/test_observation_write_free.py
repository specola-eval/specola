"""The observation path must not perturb simulation state.

Pins the 2026-08-10 maintainer decision (X15 gate 3): the
self-consistency forward inside ``get_observation()`` must not change
where the next ``mj_step`` starts from. The defect it guards against:
``mj_forward`` rewrites ``qacc_warmstart`` (persistent solver state),
so a monitored trajectory diverged from an unmonitored one at ~1e-4
under identical actions.

Known limitation, measured 2026-08-10: on this environment's MuJoCo
and these small scenes, a post-step ``mj_forward`` reproduces the
warmstart bit-exactly (probed directly, including capped-iteration and
noslip variants), so these tests did NOT fail on the pre-fix code here
— the defect manifests on the Diffusion stack's mujoco 3.3.1 robosuite
scenes. The discriminating regression is scenario F of
``benchmarks/x15/dp_step_accounting_check.py`` (real RoboCasa scene,
scripted actions, monitored-vs-unmonitored state chains), which failed
pre-fix and passes post-fix. These tests still pin the invariant's
shape where they run.
"""

import builtins

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from types import SimpleNamespace

from specola.adapters.mujoco import BodyRule, MujocoAdapter
from specola.adapters.dmcontrol import DMControlAdapter
from specola.adapters.libero import LIBEROAdapter
from specola.adapters.mamujoco import MaMujocoAdapter
from specola.adapters.metaworld import MetaworldAdapter
from specola.adapters.robocasa import RoboCasaAdapter

pytestmark = pytest.mark.mujoco

_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="2"/>
    </body>
    <body name="probe" pos="0.02 0.02 0.5">
      <freejoint name="probe_free"/>
      <geom name="probe_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _StructWrapper:
    def __init__(self, raw, attribute):
        self._raw = raw
        setattr(self, attribute, raw)

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _env():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def _adapter(env):
    return MujocoAdapter(
        env,
        body_rules=(BodyRule(match="box", cls="object",
                             joint_scope="body"),),
    )


def test_get_observation_preserves_qacc_warmstart():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    for _ in range(5):
        mujoco.mj_step(env.model, env.data)
    before = np.array(env.data.qacc_warmstart, copy=True)
    adapter.get_observation()
    np.testing.assert_array_equal(env.data.qacc_warmstart, before)


_FORWARD_CONTRACT = pytest.mark.xfail(
    strict=True,
    reason=(
        "Extraction design 2026-09-04: the adapter no longer calls mj_forward "
        "on live data; kinematics completes on an owned scratch copy under the "
        "native guard, which rejects a patched producer namespace. The "
        "equivalent injection point is the projection stage; recorded as open "
        "in docs/superpowers/plans/2026-09-07-geometry-distance-prefilter-and-provider.md."
    ),
)

@_FORWARD_CONTRACT
@pytest.mark.parametrize("forward_path", ("raw", "wrapped"))
def test_failed_required_forward_is_loud_and_preserves_last_observation(
    monkeypatch, forward_path,
):
    env = _env()
    adapter = _adapter(env)
    accepted = adapter.reset()
    adapter_data_before = np.array(env.data.qacc_warmstart, copy=True)
    forward_data = (
        env.data if forward_path == "raw" else mujoco.MjData(env.model)
    )
    forward_data_before = np.array(forward_data.qacc_warmstart, copy=True)

    def fail_forward(*_args):
        forward_data.qacc_warmstart[:] = forward_data_before + 1.0
        raise TypeError("incompatible forward binding")

    if forward_path == "raw":
        monkeypatch.setattr(mujoco, "mj_forward", fail_forward)
    else:
        env.sim = SimpleNamespace(
            data=forward_data,
            forward=fail_forward,
        )

    with pytest.raises(
        RuntimeError,
        match="required forward recomputation",
    ) as excinfo:
        adapter.get_observation()

    assert isinstance(excinfo.value.__cause__, TypeError)
    assert adapter.last_observation is accepted
    np.testing.assert_array_equal(forward_data.qacc_warmstart, forward_data_before)
    np.testing.assert_array_equal(env.data.qacc_warmstart, adapter_data_before)


@_FORWARD_CONTRACT
def test_raw_forward_unwraps_supported_binding_structs(monkeypatch):
    raw = _env()
    env = SimpleNamespace(
        model=_StructWrapper(raw.model, "_model"),
        data=_StructWrapper(raw.data, "_data"),
    )
    adapter = _adapter(env)
    adapter.reset()
    raw.data.qacc_warmstart[:] = 0.125
    warmstart_before = np.array(raw.data.qacc_warmstart, copy=True)
    original = mujoco.mj_forward
    calls = []

    def require_raw(model, data):
        assert model is raw.model
        assert data is raw.data
        calls.append((model, data))
        original(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", require_raw)

    adapter.get_observation()

    assert calls == [(raw.model, raw.data)]
    np.testing.assert_array_equal(raw.data.qacc_warmstart, warmstart_before)


def test_observed_trajectory_equals_unobserved():
    # Two identical rollouts under zero actions through contact; one is
    # observed every step. qpos histories must match bit for bit — this
    # catches ANY state-perturbing write in the observation path, not
    # just the warmstart one.
    histories = []
    for observe in (False, True):
        env = _env()
        adapter = _adapter(env)
        adapter.reset()
        qpos_hist = []
        for _ in range(200):
            mujoco.mj_step(env.model, env.data)
            if observe:
                adapter.get_observation()
            qpos_hist.append(np.array(env.data.qpos, copy=True))
        histories.append(np.stack(qpos_hist))
    np.testing.assert_array_equal(histories[0], histories[1])


@_FORWARD_CONTRACT
def test_raw_forward_type_error_refuses_stale_derived_state(monkeypatch):
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    before = np.array(env.data.qacc_warmstart, copy=True)

    def fail_forward(model, data):
        del model, data
        raise TypeError("injected raw-forward failure")

    monkeypatch.setattr(mujoco, "mj_forward", fail_forward)

    with pytest.raises(
        RuntimeError,
        match="refusing stale derived-state reads",
    ) as error:
        adapter.get_observation()

    assert isinstance(error.value.__cause__, TypeError)
    assert str(error.value.__cause__) == "injected raw-forward failure"
    np.testing.assert_array_equal(env.data.qacc_warmstart, before)


@_FORWARD_CONTRACT
def test_wrapped_forward_type_error_refuses_stale_derived_state():
    raw_env = _env()

    class FailingSim:
        model = raw_env.model
        data = raw_env.data

        @staticmethod
        def forward():
            raise TypeError("injected wrapped-forward failure")

    env = SimpleNamespace(sim=FailingSim())
    adapter = _adapter(env)
    adapter.reset()
    before = np.array(raw_env.data.qacc_warmstart, copy=True)

    with pytest.raises(
        RuntimeError,
        match="refusing stale derived-state reads",
    ) as error:
        adapter.get_observation()

    assert isinstance(error.value.__cause__, TypeError)
    assert str(error.value.__cause__) == "injected wrapped-forward failure"
    np.testing.assert_array_equal(raw_env.data.qacc_warmstart, before)


@_FORWARD_CONTRACT
def test_raw_forward_import_error_refuses_stale_derived_state(monkeypatch):
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    before = np.array(env.data.qacc_warmstart, copy=True)
    import_original = builtins.__import__

    def fail_mujoco_import(name, *args, **kwargs):
        if name == "mujoco":
            raise ImportError("injected missing mujoco")
        return import_original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_mujoco_import)

    with pytest.raises(
        RuntimeError,
        match="refusing stale derived-state reads",
    ) as error:
        adapter.get_observation()

    assert isinstance(error.value.__cause__, ImportError)
    assert str(error.value.__cause__) == "injected missing mujoco"
    np.testing.assert_array_equal(env.data.qacc_warmstart, before)


def test_all_five_mujoco_family_wrappers_inherit_shared_blocker_fixes():
    wrappers = (
        RoboCasaAdapter,
        LIBEROAdapter,
        MetaworldAdapter,
        DMControlAdapter,
        MaMujocoAdapter,
    )

    assert len(wrappers) == 5
    assert len(set(wrappers)) == 5
    assert all(issubclass(adapter, MujocoAdapter) for adapter in wrappers)
    assert all(
        adapter._kinematic_forward is MujocoAdapter._kinematic_forward
        for adapter in wrappers
    )
    assert all(
        adapter._xaxis_owner_triples is MujocoAdapter._xaxis_owner_triples
        for adapter in wrappers
    )
    assert all(
        adapter._materialize_joint_geometry
        is MujocoAdapter._materialize_joint_geometry
        for adapter in wrappers
    )
