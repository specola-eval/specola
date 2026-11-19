"""
Drive a gymnasium-style environment for ``specola run``'s ``--mode``.

``gym_advancer`` is the one implementation of the hold and random drives
over an environment with an ``action_space``: the run loop's generic
fallback uses it, and an adapter's ``make_advancer`` can return it. It
steps the environment it is given, the outermost wrapper of the caller's
chain, so every wrapper (an episode limit, an action rescaling) still
applies. It refuses a chain holding an automatic-reset wrapper, which
would replace the terminal state before the adapter reads it, and it ends
the episode the way the run loop needs: the step that terminates or
truncates returns ``True`` so its observation is read, and the next call
returns ``False`` without stepping.

Action spaces: ``Box`` holds zeros in its dtype (or ``hold_action``, which
it needs when zeros fall outside the space) and draws uniformly within its
finite bounds; ``Discrete`` and ``MultiDiscrete`` draw seeded indices,
and hold needs ``hold_action``, the simulator's own no-op, because no index
is neutral in general (index 0 is an active action in many environments).
Any other space is refused.

Converting actions the native environment cannot take (a rescaling
wrapper hands an array to a native step that needs a scalar) is not done
here: the factory puts a thin wrapper beneath every action-transforming
wrapper, as ``docs/adapter-hooks.md`` (the ``make_advancer`` row)
describes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

import numpy as np

_DRIVE_MODES = ("hold", "random")


def step_signals_episode_end(result: Any) -> bool:
    """True if a gym or gymnasium ``step()`` return ends the episode.

    Envs like Meta-World raise if ``step()`` is called again once their
    internal step budget is exhausted, instead of returning stale obs, so
    the advancer stops itself at truncation rather than relying on the
    caller to notice next time around.
    """
    if not isinstance(result, tuple):
        return False
    if len(result) == 5:      # gymnasium: obs, reward, terminated, truncated, info
        _, _, terminated, truncated, _ = result
        return bool(terminated or truncated)
    if len(result) == 4:      # legacy gym: obs, reward, done, info
        _, _, done, _ = result
        return bool(done)
    return False


def wrapper_chain(env: Any) -> Iterator[Any]:
    """Yield ``env`` and each object behind it along the ``.env`` chain."""
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = getattr(current, "env", None)


def _automatic_reset_classes() -> tuple:
    try:
        import gymnasium.wrappers as wrappers
    except Exception:
        return ()
    # ``AutoResetWrapper`` before gymnasium 1.0, ``Autoreset`` from 1.0.
    return tuple(
        cls for cls in (getattr(wrappers, "AutoResetWrapper", None), getattr(wrappers, "Autoreset", None))
        if isinstance(cls, type)
    )


def refuse_automatic_reset(env: Any) -> None:
    """Raise ``SystemExit`` if the chain holds an automatic-reset wrapper.

    Matched by type, so a subclass or a re-export is caught too. Stepping
    through such a wrapper replaces the terminal state with the next
    episode's reset state before the adapter reads it.
    """
    classes = _automatic_reset_classes()
    if not classes:
        return
    for layer in wrapper_chain(env):
        if isinstance(layer, classes):
            raise SystemExit(
                f"the environment's wrapper chain holds {type(layer).__name__}, an "
                "automatic-reset wrapper: it replaces the terminal state before the "
                "adapter reads it. Drive the environment without it."
            )


def _space_kind(space: Any) -> str:
    """``Box``, ``Discrete``, ``MultiDiscrete``, or the space's type name.

    gymnasium spaces are matched by type; anything else (a legacy ``gym``
    space, a duck-typed stand-in) by the attributes the drive uses.
    """
    try:
        from gymnasium import spaces
    except Exception:
        spaces = None
    if spaces is not None:
        for kind in ("Box", "Discrete", "MultiDiscrete"):
            if isinstance(space, getattr(spaces, kind)):
                return kind
        if isinstance(space, spaces.Space):
            return type(space).__name__
    name = type(space).__name__
    if hasattr(space, "low") and hasattr(space, "high"):
        return "Box"
    if hasattr(space, "nvec"):
        return "MultiDiscrete"
    if hasattr(space, "n") and "Binary" not in name:
        return "Discrete"
    return name


def gym_advancer(env: Any, mode: str, seed: int, demo: Optional[str] = None,
                 hold_action: Any = None) -> Callable[[int], bool]:
    """Return ``advance(step) -> bool`` for ``--mode hold`` or ``--mode random``.

    ``hold`` sends ``hold_action`` every step (for a ``Box`` space, zeros of
    its shape when none is given; a ``Discrete`` or ``MultiDiscrete`` space
    requires it). ``random`` draws uniformly from the space with a generator
    seeded with ``seed``. The caller resets ``env`` before the first
    ``advance``. ``replay`` and ``--demo`` are refused: a gymnasium
    environment carries no replay source of its own.
    """
    if mode == "replay":
        raise SystemExit(
            "--mode replay needs a recorded source this environment does not "
            "have; use --mode hold or --mode random"
        )
    if demo is not None:
        raise SystemExit(f"--demo is only read by --mode replay, not --mode {mode}")
    if mode not in _DRIVE_MODES:
        raise SystemExit(f"unknown drive mode {mode!r}; expected one of {_DRIVE_MODES}")
    space_owner = next((layer for layer in wrapper_chain(env) if hasattr(layer, "action_space")), None)
    if space_owner is None:
        raise SystemExit(f"--mode {mode} needs an environment exposing action_space")
    refuse_automatic_reset(env)
    space = space_owner.action_space
    kind = _space_kind(space)
    if kind not in ("Box", "Discrete", "MultiDiscrete"):
        raise SystemExit(
            f"--mode {mode} cannot drive a {kind} action space; the adapter's "
            "make_advancer drives it instead"
        )
    dtype = getattr(space, "dtype", None)
    default_hold = mode == "hold" and hold_action is None and kind == "Box"
    if kind == "Box":
        # gymnasium clamps an infinite integer bound to the dtype's limits,
        # so its own boundedness flags decide; isfinite for anything else.
        bounded = (space.is_bounded("both") if hasattr(space, "is_bounded")
                   else bool(np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))))
        if mode == "random" and not bounded:
            raise SystemExit(
                f"--mode random cannot draw uniformly from the unbounded action space {space}; "
                "the adapter's make_advancer drives it instead"
            )
        if default_hold:
            hold_action = np.zeros(space.shape)
        if hold_action is not None:
            # Box.contains refuses a dtype it cannot cast to (float64 into float32).
            hold_action = np.asarray(hold_action, dtype=dtype)
    if hold_action is not None and hasattr(space, "contains") and not space.contains(hold_action):
        raise SystemExit(
            f"hold_action {hold_action!r} is not in the action space {space}"
            + ("; zeros fall outside it, so the adapter's make_advancer passes the "
               "simulator's own hold_action" if default_hold else "")
        )
    if mode == "hold" and hold_action is None and kind != "Box":
        raise SystemExit(
            f"--mode hold on a {kind} action space needs hold_action, the "
            "simulator's no-op: no index is neutral in general"
        )
    rng = np.random.default_rng(seed)
    done = False

    def draw() -> Any:
        if kind == "Discrete":
            return int(getattr(space, "start", 0)) + int(rng.integers(int(space.n)))
        if kind == "MultiDiscrete":
            start = np.asarray(getattr(space, "start", 0))
            return (start + rng.integers(np.asarray(space.nvec))).astype(space.dtype)
        if np.issubdtype(dtype, np.bool_):
            return rng.integers(space.low, space.high, endpoint=True).astype(dtype)
        if np.issubdtype(dtype, np.integer):
            return rng.integers(space.low, space.high, endpoint=True, dtype=dtype)
        # asarray: a scalar Box draws a Python float, which has no astype.
        return np.asarray(rng.uniform(space.low, space.high)).astype(dtype)

    def advance(step: int) -> bool:
        nonlocal done
        if done:
            return False
        if mode == "hold":
            # A fresh copy each step: a wrapper may convert the action in place.
            action = hold_action.copy() if isinstance(hold_action, np.ndarray) else hold_action
        else:
            action = draw()
        done = step_signals_episode_end(env.step(action))
        return True

    return advance
