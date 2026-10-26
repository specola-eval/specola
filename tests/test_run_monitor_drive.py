"""
Drive-mode helpers must work with wrapped simulator environments.

LIBERO's ControlEnv/OffScreenRenderEnv hold the underlying robosuite
environment at ``.env`` and forward ``step()`` and ``sim`` but not
``action_spec`` or ``action_dim``. The advancer must step the outermost
wrapper (so wrapper bookkeeping runs) while reading action metadata from
whichever level of the wrapper chain exposes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from specola.run_dashboard import DashboardResult
from specola.run_monitor import _make_advancer, _typed_eval_loop, prime_replay_env

_MOCK_SPECS = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"


class _InnerRobosuiteEnv:
    """Robosuite-style env: owns action metadata."""

    action_dim = 4

    @property
    def action_spec(self):
        return -np.ones(self.action_dim), np.ones(self.action_dim)

    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action))


class _LiberoStyleWrapper:
    """Forwards step() but not action_spec/action_dim, like ControlEnv."""

    def __init__(self, inner):
        self.env = inner
        self.stepped = []

    def step(self, action):
        self.stepped.append(np.asarray(action))
        self.env.step(action)


class _FakeAdapter:
    def __init__(self, env):
        self.env = env


def test_random_mode_drives_wrapped_env():
    inner = _InnerRobosuiteEnv()
    wrapper = _LiberoStyleWrapper(inner)
    adapter = _FakeAdapter(wrapper)

    advance = _make_advancer(adapter, "random", demo=None, seed=7)

    assert advance(0) is True
    # The outermost wrapper was stepped, so wrapper bookkeeping ran.
    assert len(wrapper.stepped) == 1
    action = wrapper.stepped[0]
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_hold_mode_drives_wrapped_env_with_zero_action():
    inner = _InnerRobosuiteEnv()
    wrapper = _LiberoStyleWrapper(inner)
    adapter = _FakeAdapter(wrapper)

    advance = _make_advancer(adapter, "hold", demo=None, seed=0)

    assert advance(0) is True
    assert len(wrapper.stepped) == 1
    assert np.array_equal(wrapper.stepped[0], np.zeros(4))


def test_unwrapped_env_still_works():
    inner = _InnerRobosuiteEnv()
    adapter = _FakeAdapter(inner)

    advance = _make_advancer(adapter, "random", demo=None, seed=7)

    assert advance(0) is True
    assert len(inner.actions) == 1


class _Box:
    def __init__(self, dim):
        self.low = -np.ones(dim)
        self.high = np.ones(dim)
        self.shape = (dim,)


class _GymTruncatingEnv:
    """Gymnasium-style env that raises if stepped again after truncation,
    like Meta-World's SawyerXYZEnv once curr_path_length >= max_path_length."""

    action_dim = 4

    def __init__(self, max_steps):
        self.action_space = _Box(4)
        self.max_steps = max_steps
        self.count = 0

    def step(self, action):
        if self.count >= self.max_steps:
            raise ValueError("You must reset the env manually once truncate==True")
        self.count += 1
        truncated = self.count >= self.max_steps
        return np.zeros(4), 0.0, False, truncated, {}


def test_random_mode_stops_at_truncation_instead_of_crashing():
    env = _GymTruncatingEnv(max_steps=2)
    adapter = _FakeAdapter(env)

    advance = _make_advancer(adapter, "random", demo=None, seed=7)

    assert advance(0) is True   # count 0 -> 1, truncated=False
    assert advance(1) is True   # count 1 -> 2, truncated=True (last valid frame)
    assert advance(2) is False  # must NOT call env.step() again
    assert env.count == 2


def test_hold_mode_stops_at_truncation_instead_of_crashing():
    env = _GymTruncatingEnv(max_steps=2)
    adapter = _FakeAdapter(env)

    advance = _make_advancer(adapter, "hold", demo=None, seed=0)

    assert advance(0) is True
    assert advance(1) is True
    assert advance(2) is False
    assert env.count == 2


def test_adapter_supplied_advancer_wins_over_duck_typing():
    """An adapter may own its drive behavior (make_advancer hook);
    simulators with non-robosuite/gymnasium control APIs (e.g. CARLA)
    then need no shared-CLI changes."""
    calls = []

    class _CarlaLikeAdapter:
        env = None  # no action_spec/action_space anywhere

        def make_advancer(self, mode, demo, seed):
            assert mode == "random" and seed == 7
            def advance(step):
                calls.append(step)   # e.g. world.tick() + autopilot
                return step < 2
            return advance

    advance = _make_advancer(_CarlaLikeAdapter(), "random", demo=None, seed=7)

    assert advance(0) is True
    assert advance(2) is False
    assert calls == [0, 2]


class _SimEnv:
    """Robosuite-style env exposing ``.sim``, so replay reaches the demo file."""

    sim = object()


@pytest.mark.parametrize("fragment", ["", "#12"])
def test_replay_rejects_missing_demo_before_compilation(tmp_path, fragment):
    """Issue #135: prime_replay_env runs before the run directory exists,
    so a missing --demo path must stop there with a one-line message
    naming the path, not an h5py traceback later in the run."""
    missing = tmp_path / "does-not-exist.hdf5"
    adapter = _FakeAdapter(_LiberoStyleWrapper(_SimEnv()))

    with pytest.raises(SystemExit) as rejected:
        prime_replay_env(adapter, str(missing) + fragment, simulator="libero", task=None)

    assert str(rejected.value) == f"--demo path does not exist: {missing}"


def test_adapter_owned_replay_does_not_check_demo_path(tmp_path):
    """Adapters with make_advancer own --demo; the runner must not open
    or validate a path whose format it does not know."""
    class _OwnsReplay:
        env = _SimEnv()

        def make_advancer(self, mode, demo, seed):
            raise AssertionError("not reached")

    prime_replay_env(_OwnsReplay(), str(tmp_path / "not-a-file"), simulator="dmcontrol", task=None)


@pytest.mark.parametrize("mode", ["random", "hold"])
def test_fallback_refuses_demo_outside_replay(mode):
    """Issue #132: the duck-typed fallback (metaworld, robocasa, libero,
    mock) reads --demo only in replay; any other mode must refuse it
    rather than run and ignore the file."""
    adapter = _FakeAdapter(_InnerRobosuiteEnv())

    with pytest.raises(SystemExit, match=f"--demo is only read by --mode replay, not --mode {mode}"):
        _make_advancer(adapter, mode, demo="demo.hdf5", seed=0)


# -- adapter-owned drive: dmcontrol and mamujoco (issue #132) ---------------
# Both adapters used to discard --demo and run random actions for any mode
# other than hold, so --mode replay monitored random actions with no error.


class _DMControlEnv:
    """dm_env-style stub: action_spec() bounds and a step() that records."""

    class _Spec:
        shape = (2,)
        minimum = np.full(2, -1.0)
        maximum = np.full(2, 1.0)

    class _TimeStep:
        @staticmethod
        def last():
            return False

    def __init__(self):
        self.actions = []

    def action_spec(self):
        return self._Spec()

    def step(self, action):
        self.actions.append(np.asarray(action))
        return self._TimeStep()


class _ParallelEnv:
    """PettingZoo parallel-API stub with one bounded Box per agent."""

    class _Space:
        low = np.full(2, -1.0, dtype=np.float32)
        high = np.full(2, 1.0, dtype=np.float32)
        dtype = np.float32

        def sample(self):
            raise AssertionError("unseeded space sampler was used")

    agents = ["agent_0", "agent_1"]

    def __init__(self):
        self.actions = []

    def action_space(self, agent):
        return self._Space()

    def step(self, actions):
        self.actions.append(actions)


def _dmcontrol(env):
    from specola.adapters.dmcontrol import DMControlAdapter
    adapter = object.__new__(DMControlAdapter)
    adapter._dm_env = env
    return adapter


def _mamujoco(env):
    from specola.adapters.mamujoco import MaMujocoAdapter
    adapter = object.__new__(MaMujocoAdapter)
    adapter._parallel_env = env
    return adapter


@pytest.mark.parametrize("build, env_type, name", [
    (_dmcontrol, _DMControlEnv, "dmcontrol"),
    (_mamujoco, _ParallelEnv, "mamujoco"),
])
def test_adapter_refuses_replay_and_demo(build, env_type, name):
    adapter = build(env_type())

    with pytest.raises(SystemExit, match=f"not supported by the {name} adapter \\(no replay path\\)"):
        adapter.make_advancer("replay", None, 0)
    for mode in ("replay", "random", "hold"):
        with pytest.raises(SystemExit, match=f"The {name} adapter has no --demo replay"):
            adapter.make_advancer(mode, "does-not-exist.hdf5", 0)


def test_dmcontrol_hold_is_zero_and_random_is_seeded():
    def actions(mode, seed):
        env = _DMControlEnv()
        advance = _dmcontrol(env).make_advancer(mode, None, seed)
        assert advance(0) and advance(1)
        return np.stack(env.actions)

    np.testing.assert_array_equal(actions("hold", 7), np.zeros((2, 2)))
    random = actions("random", 7)
    assert np.all(random != 0.0) and np.all(np.abs(random) <= 1.0)
    np.testing.assert_array_equal(random, actions("random", 7))
    assert not np.array_equal(random, actions("random", 8))


def test_mamujoco_hold_is_zero_and_random_is_seeded():
    def actions(mode, seed):
        env = _ParallelEnv()
        advance = _mamujoco(env).make_advancer(mode, None, seed)
        assert advance(0)
        step = env.actions[0]
        assert list(step) == _ParallelEnv.agents
        assert all(a.dtype == np.float32 for a in step.values())
        return np.stack([step[agent] for agent in _ParallelEnv.agents])

    np.testing.assert_array_equal(actions("hold", 7), np.zeros((2, 2)))
    random = actions("random", 7)
    assert np.all(random != 0.0) and np.all(np.abs(random) <= 1.0)
    np.testing.assert_array_equal(random, actions("random", 7))
    assert not np.array_equal(random, actions("random", 8))


def test_cli_dt_override_reaches_spec_window():
    """Regression: --dt must change the compiled spec window, not just
    the discovery header's printed spec_dt.

    examples/mock_specs.yaml specs use G/F[0,5s]; the mock adapter's own
    dt is 0.05s, so the default window is 5s/0.05s = 100 steps. Passing
    --dt 0.1 must halve that to 50 steps (5s/0.1s) -- run_monitor.main()
    used to compute this override via load_sim() but then routed the
    episode through Monitor.watch(), which used to hardcode adapter.dt
    and silently drop it.
    """
    def run(extra_args):
        proc = subprocess.run(
            [sys.executable, "-m", "specola.run_monitor",
             "--sim", "mock", "--specs", str(_MOCK_SPECS),
             "--steps", "5", *extra_args],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout

    default_out = run([])
    assert "(5/100 steps)" in default_out

    override_out = run(["--dt", "0.1"])
    assert "(5/50 steps)" in override_out
    assert "(5/100 steps)" not in override_out


def test_cli_surfaces_skipped_specs(tmp_path):
    """Finding I4: a spec referencing an unknown entity class is skipped by
    the facade (skip_unsatisfiable=True), but the CLI must SAY so — an amber
    skip line naming the spec and a NOT APPLICABLE marker in the episode
    summary — never drop it silently, while still exiting success.
    """
    specs = tmp_path / "specs.yaml"
    specs.write_text(
        "specs:\n"
        "  - name: real_cap\n"
        '    formula: "G[0,1s](object_*[mass] <= 1.0)"\n'
        "  - name: bogus_cap\n"
        '    formula: "G[0,1s](unicorn_*[mass] <= 1.0)"\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(specs), "--steps", "5"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # Amber skip line: names the spec and says it was skipped.
    assert "bogus_cap" in out and "skipped" in out
    # Episode summary marks the skipped spec NOT APPLICABLE rather than
    # omitting it.
    assert "NOT APPLICABLE" in out


def test_plain_mode_avoids_periodic_snapshot_blocks():
    """The plain fallback logs transitions and a final summary; per-step
    values remain in the durable result trace instead of flooding stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(_MOCK_SPECS), "--steps", "5"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "snapshot every" not in proc.stdout
    assert "Episode summary" in proc.stdout
    assert "4 initial monitor(s)" in proc.stdout


def test_textual_stop_finalizes_episode_as_truncated(monkeypatch):
    import specola.run_dashboard as dashboard_module
    import specola.run_monitor as monitor_module

    statuses = []

    class Engine:
        managers = {}
        spec_names = ()

        @staticmethod
        def specification_metadata():
            return ()

        @staticmethod
        def manager_specs():
            return {}

        @staticmethod
        def skipped_specs():
            return {}

        @staticmethod
        def unmatched_slots():
            return {}

    class Session:
        engine = Engine()
        adapter = object()

        @staticmethod
        def end(*, status):
            statuses.append(status)
            return object()

    monkeypatch.setattr(
        dashboard_module,
        "run_textual_dashboard",
        lambda *args, **kwargs: DashboardResult(None, True),
    )
    monkeypatch.setattr(
        monitor_module, "_print_typed_summary", lambda *args: None
    )

    _typed_eval_loop(
        Session(),
        advance=lambda step: True,
        mode="hold",
        max_steps=1,
        entity_labels={},
        render=False,
        ui="textual",
        refresh_rate=10.0,
        summary_only=True,
    )

    assert statuses == ["truncated"]


def test_worker_abandoned_skips_result_finalization(monkeypatch):
    import pytest

    import specola.run_dashboard as dashboard_module
    import specola.run_monitor as monitor_module
    from specola.run_dashboard import WorkerAbandoned

    statuses = []

    class Engine:
        managers = {}
        spec_names = ()

        @staticmethod
        def specification_metadata():
            return ()

        @staticmethod
        def manager_specs():
            return {}

        @staticmethod
        def skipped_specs():
            return {}

        @staticmethod
        def unmatched_slots():
            return {}

    class Session:
        engine = Engine()
        adapter = object()

        @staticmethod
        def end(*, status):
            statuses.append(status)
            return object()

    def abandoned(*args, **kwargs):
        raise WorkerAbandoned("test abandon")

    monkeypatch.setattr(
        dashboard_module, "run_textual_dashboard", abandoned
    )
    monkeypatch.setattr(
        monitor_module, "_print_typed_summary", lambda *args: None
    )

    with pytest.raises(WorkerAbandoned):
        _typed_eval_loop(
            Session(),
            advance=lambda step: True,
            mode="hold",
            max_steps=1,
            entity_labels={},
            render=False,
            ui="textual",
            refresh_rate=10.0,
            summary_only=True,
        )

    assert statuses == []
