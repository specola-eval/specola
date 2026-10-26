"""Paired-run check that observing does not change a RoboCasa rollout.

MuJoCo 3.3.1, which RoboCasa pins, keeps its convex solver's scratch in
thread-local arrays that the query reader's scratch data copy does not
isolate. The reader admits that build on measured evidence (custody note
docs/superpowers/specs/2026-10-02-mujoco-query-custody-3.3.md, option B).
This test is that evidence for the audited task: the same control
sequence is stepped from the same integration state twice, once with a
full adapter observation after every step and once with none, and the
integration state must match bit for bit at every step. The step count is
small because one observation of this scene takes about 67 s.

It shows no effect on these trajectories. It does not prove isolation.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.robocasa

REPO = Path(__file__).resolve().parents[1]
TASK = "PickPlaceCounterToCabinet"
STEPS = 12  # one observation takes about 67 s on this scene (measured 2026-10-02)


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robocasa")
    from specola import registry
    from specola.query_config import load_query_configuration

    adapter = registry.resolve("robocasa").factory(TASK, None, True)
    adapter.set_query_configuration(
        load_query_configuration(REPO / "examples" / "geometry_queries.yaml")
    )
    adapter.reset()
    env = adapter.env
    while not hasattr(env, "sim"):
        env = env.env
    return mujoco, adapter, env.sim.model._model, env.sim.data._data


def _state(mujoco, model, data) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    out = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, out, spec)
    return out


def _rollout(mujoco, model, data, start, controls, observe):
    mujoco.mj_setState(model, data, start, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    states, contacts = [], []
    for control in controls:
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        if observe is not None:
            observe()
        states.append(_state(mujoco, model, data))
        contacts.append(int(data.ncon))
    return np.array(states), contacts


@pytest.mark.parametrize("seed", [0, 1])
def test_observation_does_not_change_the_rollout(scene, seed):
    mujoco, adapter, model, data = scene
    assert mujoco.__version__ == "3.3.1", "this check is about the 3.3.1 build"
    start = _state(mujoco, model, data)
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    controls = np.random.default_rng(seed).uniform(low, high, (STEPS, model.nu))

    # Unobserved first: the adapter's observation clock only moves forward.
    plain, plain_contacts = _rollout(mujoco, model, data, start, controls, None)
    observed, contacts = _rollout(
        mujoco, model, data, start, controls, adapter.get_observation
    )

    assert max(contacts) > 0, "the rollout never had a contact to query"
    assert contacts == plain_contacts
    # The queries must have run, not all been pruned by the bound.
    executed = [s for s in adapter._geometry_samples
                if s.availability.startswith("native_")]
    assert executed, "no geometry pair was executed in the last observation"
    # Compare bytes per step, so a NaN on both sides counts as equal.
    differing = [k for k in range(STEPS)
                 if plain[k].tobytes() != observed[k].tobytes()]
    assert not differing, (
        f"integration state differs from step {differing[0]} "
        f"({len(differing)} of {STEPS} steps)"
    )
