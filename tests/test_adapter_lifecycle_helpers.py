"""The shared drive and viewer helpers adapters build on (specola.drive,
specola.viewer). Drive tests use real gymnasium wrappers over a tiny
in-memory environment; they skip where gymnasium is not installed."""

import pytest

from specola.viewer import SharedProcessState, ViewerOwnership


# -- drive -----------------------------------------------------------------

def _gym():
    return pytest.importorskip("gymnasium")


def _toy_env(end_after=None):
    gym = _gym()
    import numpy as np

    class Toy(gym.Env):
        """Counts steps; records every action that reaches it."""

        metadata = {"render_modes": []}

        def __init__(self):
            self.action_space = gym.spaces.Box(0.0, 10.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.actions = []
            self.t = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=float).copy())
            self.t += 1
            terminated = end_after is not None and self.t >= end_after
            return np.array([self.t], dtype=np.float32), 0.0, terminated, False, {}

    return Toy()


def _counting(env):
    gym = _gym()

    class Counting(gym.Wrapper):
        calls = 0

        def step(self, action):
            type(self).calls += 1
            return self.env.step(action)

    return Counting(env)


def _autoreset_class():
    import gymnasium.wrappers as w

    return getattr(w, "Autoreset", None) or getattr(w, "AutoResetWrapper")


def test_hold_steps_the_outermost_wrapper_with_zeros():
    from specola.drive import gym_advancer

    toy = _toy_env()
    env = _counting(toy)
    env.reset(seed=0)
    advance = gym_advancer(env, "hold", seed=0)
    assert advance(0) and advance(1)
    assert type(env).calls == 2  # a drive that bypasses the wrapper fails here
    assert [a.tolist() for a in toy.actions] == [[0.0, 0.0], [0.0, 0.0]]


def test_random_is_seeded_and_within_bounds():
    from specola.drive import gym_advancer

    runs = []
    for _ in range(2):
        toy = _toy_env()
        toy.reset(seed=0)
        advance = gym_advancer(toy, "random", seed=7)
        for step in range(5):
            advance(step)
        runs.append([a.tolist() for a in toy.actions])
    assert runs[0] == runs[1]
    assert all(0.0 <= x <= 10.0 for action in runs[0] for x in action)
    assert len({tuple(a) for a in runs[0]}) == 5


def test_the_terminating_step_is_read_and_the_next_call_stops():
    from specola.drive import gym_advancer

    toy = _toy_env(end_after=2)
    toy.reset(seed=0)
    advance = gym_advancer(toy, "hold", seed=0)
    assert advance(0) is True
    assert advance(1) is True   # this step terminates; its observation is read
    assert advance(2) is False  # and nothing steps after it
    assert len(toy.actions) == 2


def test_an_episode_limit_wrapper_still_applies():
    gym = _gym()
    from specola.drive import gym_advancer

    toy = _toy_env()
    env = gym.wrappers.TimeLimit(toy, max_episode_steps=3)
    env.reset(seed=0)
    advance = gym_advancer(env, "hold", seed=0)
    assert [advance(step) for step in range(5)] == [True, True, True, False, False]
    assert len(toy.actions) == 3


def test_an_action_rescaling_wrapper_still_applies():
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    toy = _toy_env()
    env = gym.wrappers.RescaleAction(toy, np.float32(-1.0), np.float32(1.0))
    env.reset(seed=0)
    gym_advancer(env, "hold", seed=0)(0)
    # zero in the rescaled [-1, 1] space is the middle of the native [0, 10]
    assert toy.actions[0].tolist() == pytest.approx([5.0, 5.0])


def test_an_automatic_reset_wrapper_subclass_is_refused():
    from specola.drive import gym_advancer

    class MyReset(_autoreset_class()):
        pass

    env = _counting(MyReset(_toy_env(end_after=1)))
    with pytest.raises(SystemExit, match="MyReset"):
        gym_advancer(env, "hold", seed=0)


def test_replay_and_demo_are_refused():
    from specola.drive import gym_advancer

    toy = _toy_env()
    with pytest.raises(SystemExit, match="replay"):
        gym_advancer(toy, "replay", seed=0)
    with pytest.raises(SystemExit, match="--demo"):
        gym_advancer(toy, "hold", seed=0, demo="episode.hdf5")


def test_an_env_without_an_action_space_is_refused():
    from specola.drive import gym_advancer

    with pytest.raises(SystemExit, match="action_space"):
        gym_advancer(object(), "hold", seed=0)


def test_the_run_loop_fallback_refuses_an_automatic_reset_chain():
    from types import SimpleNamespace
    from specola.run_monitor import _make_advancer

    class MyReset(_autoreset_class()):
        pass

    adapter = SimpleNamespace(env=MyReset(_toy_env(end_after=1)))
    with pytest.raises(SystemExit, match="MyReset"):
        _make_advancer(adapter, "hold", None, 0)


def test_the_run_loop_fallback_still_drives_a_plain_gymnasium_env():
    from types import SimpleNamespace
    from specola.run_monitor import _make_advancer

    toy = _toy_env(end_after=2)
    toy.reset(seed=0)
    advance = _make_advancer(SimpleNamespace(env=toy), "hold", None, 0)
    assert [advance(step) for step in range(3)] == [True, True, False]


def _space_env(space, end_after=None):
    gym = _gym()
    import numpy as np

    class SpaceEnv(gym.Env):
        """Records every action; its action space is whatever the test gives."""

        metadata = {"render_modes": []}

        def __init__(self):
            self.action_space = space
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.actions = []

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(action)
            done = end_after is not None and len(self.actions) >= end_after
            return np.zeros(1, dtype=np.float32), 0.0, done, False, {}

    return SpaceEnv()


def test_discrete_hold_needs_the_simulators_no_op():
    gym = _gym()
    from specola.drive import gym_advancer

    env = _space_env(gym.spaces.Discrete(5))
    env.reset(seed=0)
    with pytest.raises(SystemExit, match="hold_action"):
        gym_advancer(env, "hold", seed=0)  # index 0 is an active action in many envs
    advance = gym_advancer(env, "hold", seed=0, hold_action=1)
    advance(0)
    advance(1)
    assert env.actions == [1, 1]
    assert all(type(a) is int for a in env.actions)


def test_a_hold_action_outside_the_space_is_refused():
    gym = _gym()
    from specola.drive import gym_advancer

    with pytest.raises(SystemExit, match="not in the action space"):
        gym_advancer(_space_env(gym.spaces.Discrete(5)), "hold", seed=0, hold_action=7)


def test_discrete_random_is_seeded_and_respects_start():
    gym = _gym()
    from specola.drive import gym_advancer

    runs = []
    for _ in range(2):
        env = _space_env(gym.spaces.Discrete(4, start=2))
        env.reset(seed=0)
        advance = gym_advancer(env, "random", seed=3)
        for step in range(20):
            advance(step)
        runs.append(list(env.actions))
    assert runs[0] == runs[1]
    assert set(runs[0]) <= {2, 3, 4, 5} and len(set(runs[0])) > 1
    assert all(type(a) is int for a in runs[0])


def test_multi_discrete_random_draws_each_component_in_range():
    gym = _gym()
    from specola.drive import gym_advancer

    env = _space_env(gym.spaces.MultiDiscrete([3, 2]))
    env.reset(seed=0)
    advance = gym_advancer(env, "random", seed=0)
    for step in range(30):
        advance(step)
    assert all(env.action_space.contains(a) for a in env.actions)
    with pytest.raises(SystemExit, match="hold_action"):
        gym_advancer(env, "hold", seed=0)


def test_box_hold_action_overrides_zeros():
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    env = _space_env(gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32))
    env.reset(seed=0)
    gym_advancer(env, "hold", seed=0, hold_action=np.array([0.5, -0.5], dtype=np.float32))(0)
    assert np.asarray(env.actions[0]).tolist() == [0.5, -0.5]


def test_a_float64_hold_action_is_accepted_for_a_float32_box():
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    env = _space_env(gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32))
    env.reset(seed=0)
    gym_advancer(env, "hold", seed=0, hold_action=np.zeros(2))(0)  # float64
    assert np.asarray(env.actions[0]).dtype == np.float32


def test_box_default_hold_and_random_actions_are_members_of_the_space():
    # External review: default zeros were float64 (not in a float32 Box),
    # random was always float32 (not in an integer Box).
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    for space in (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                  gym.spaces.Box(-3, 3, shape=(2,), dtype=np.int64)):
        for mode in ("hold", "random"):
            env = _space_env(space)
            env.reset(seed=0)
            advance = gym_advancer(env, mode, seed=0)
            for step in range(10):
                advance(step)
            assert all(space.contains(a) for a in env.actions), (space, mode)


def test_hold_survives_a_wrapper_that_converts_the_action_in_place():
    # External review: one shared hold array drifted 1, 2, 3 through a
    # wrapper that adds its offset in place.
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    native = _space_env(gym.spaces.Box(0.0, 2.0, shape=(1,), dtype=np.float64))

    class Offset(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)

        def action(self, action):
            action += 1.0
            return action

    env = Offset(native)
    env.reset(seed=0)
    advance = gym_advancer(env, "hold", seed=0)
    for step in range(3):
        advance(step)
    assert [a.tolist() for a in native.actions] == [[1.0], [1.0], [1.0]]


def test_box_random_covers_scalar_unsigned_and_bool_spaces():
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    for space in (gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
                  gym.spaces.Box(0, 2**64 - 1, shape=(2,), dtype=np.uint64),
                  gym.spaces.Box(0, 1, shape=(2,), dtype=bool)):
        env = _space_env(space)
        env.reset(seed=0)
        advance = gym_advancer(env, "random", seed=0)
        for step in range(50):
            advance(step)
        assert all(space.contains(a) for a in env.actions), space
        if space.dtype == bool:
            assert len({tuple(a) for a in env.actions}) > 1  # not always True


def test_box_refuses_zeros_outside_the_space_and_unbounded_random():
    gym = _gym()
    import numpy as np
    from specola.drive import gym_advancer

    with pytest.raises(SystemExit, match="zeros fall outside"):
        gym_advancer(_space_env(gym.spaces.Box(1.0, 2.0, shape=(2,), dtype=np.float32)), "hold", seed=0)
    for dtype in (np.float32, np.int64):  # gymnasium clamps an integer inf to the dtype's limits
        with pytest.raises(SystemExit, match="unbounded"):
            gym_advancer(_space_env(gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=dtype)), "random", seed=0)


def test_an_unsupported_action_space_is_refused():
    gym = _gym()
    from specola.drive import gym_advancer

    space = gym.spaces.Dict({"a": gym.spaces.Discrete(2)})
    with pytest.raises(SystemExit, match="Dict"):
        gym_advancer(_space_env(space), "random", seed=0)


# -- viewer ----------------------------------------------------------------

class _Sim:
    """A simulator holding at most one viewer reference, as native ones do."""

    def __init__(self, viewer=None):
        self.viewer = viewer
        self.closed = []

    def draw(self):
        if self.viewer is None:
            self.viewer = object()

    def close_viewer(self, viewer):
        self.closed.append(viewer)


def _ownership(sim):
    return ViewerOwnership(lambda: sim.viewer, sim.close_viewer)


def test_a_viewer_that_existed_first_stays_the_callers():
    callers = object()
    sim = _Sim(viewer=callers)
    owner = _ownership(sim)
    owner.render(sim.draw)
    owner.render(sim.draw)
    owner.release()
    assert sim.closed == []


def test_a_viewer_the_adapter_created_is_released_after_a_native_reset():
    sim = _Sim()
    owner = _ownership(sim)
    owner.render(sim.draw)
    created = sim.viewer
    sim.viewer = None  # a native reset drops its reference
    owner.render(sim.draw)
    second = sim.viewer
    owner.release()
    assert sim.closed == [created, second]
    owner.release()
    assert sim.closed == [created, second]  # released once


def test_a_viewer_created_after_the_callers_is_the_adapters():
    callers = object()
    sim = _Sim(viewer=callers)
    owner = _ownership(sim)
    owner.render(sim.draw)
    sim.viewer = None
    owner.render(sim.draw)
    replacement = sim.viewer
    owner.release()
    assert sim.closed == [replacement]


def test_a_viewer_the_caller_opens_between_renders_stays_the_callers():
    sim = _Sim()
    owner = _ownership(sim)
    owner.render(sim.draw)
    created = sim.viewer
    owner.release()
    callers = object()
    sim.viewer = callers  # the caller opens its own viewer afterwards
    owner.render(sim.draw)
    owner.release()
    assert sim.closed == [created]


def test_a_draw_that_raises_after_opening_a_viewer_still_owns_it():
    sim = _Sim()
    owner = _ownership(sim)

    def failing_draw():
        sim.draw()
        raise RuntimeError("draw failed after the window opened")

    with pytest.raises(RuntimeError):
        owner.render(failing_draw)
    created = sim.viewer
    owner.release()
    assert sim.closed == [created]


def test_release_attempts_every_viewer_when_one_close_raises():
    sim = _Sim()
    owner = _ownership(sim)
    owner.render(sim.draw)
    first = sim.viewer
    sim.viewer = None
    owner.render(sim.draw)
    second = sim.viewer
    attempts = []

    def close(viewer):
        attempts.append(viewer)
        if viewer is first:
            raise RuntimeError("first close failed")

    owner._close = close
    with pytest.raises(RuntimeError, match="first close failed"):
        owner.release()
    assert attempts == [first, second]


def test_process_state_is_restored_by_the_last_release_in_any_order():
    value = {"on": False}
    state = SharedProcessState(lambda: value["on"], lambda v: value.update(on=v))
    first, second = object(), object()
    state.acquire(first)
    value["on"] = True          # the simulator turns it on while rendering
    state.acquire(second)       # sees it on, but the original is already saved
    state.release(first)
    assert value["on"] is True  # the second holder still renders
    state.release(second)
    assert value["on"] is False
    state.release(second)       # a repeated release is a no-op
    assert value["on"] is False


def test_matplotlib_interactive_mode_is_one_shared_state():
    pytest.importorskip("matplotlib")
    import matplotlib
    from specola.viewer import matplotlib_interactive

    before = matplotlib.is_interactive()
    holder = object()
    matplotlib_interactive().acquire(holder)
    matplotlib.interactive(not before)
    matplotlib_interactive().release(holder)
    assert matplotlib.is_interactive() is before
    assert matplotlib_interactive() is matplotlib_interactive()


def test_pygame_is_quit_only_if_it_was_not_running_before():
    pygame = pytest.importorskip("pygame")
    from specola.viewer import pygame_session

    pygame.quit()
    holder = object()
    pygame_session().acquire(holder)
    pygame.init()                      # the simulator starts pygame to render
    pygame_session().release(holder)
    assert not pygame.get_init()       # it was not running before: quit

    pygame.init()                      # the caller runs pygame already
    other = object()
    pygame_session().acquire(other)
    pygame_session().release(other)
    assert pygame.get_init()           # left running for its owner
    pygame.quit()
    assert pygame_session() is pygame_session()


def test_a_caller_display_started_without_pygame_init_survives():
    pygame = pytest.importorskip("pygame")
    from specola.viewer import pygame_session

    pygame.quit()
    pygame.display.init()              # the caller starts only the display
    holder = object()
    pygame_session().acquire(holder)
    pygame_session().release(holder)
    assert pygame.display.get_init()
    pygame.quit()
