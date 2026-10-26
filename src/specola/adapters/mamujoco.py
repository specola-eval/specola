"""
mamujoco.py — Multi-Agent MuJoCo (MaMuJoCo) SimulatorAdapter
============================================================
Farama's MaMuJoCo factorizes single MuJoCo robots (HalfCheetah, Ant,
...) into multiple agents controlling disjoint joint groups, behind a
PettingZoo parallel API. The physics scene is still one robot: the
adapter reads the underlying single-agent env's model/data, so the
monitored entities are the robot's bodies — the multi-agent structure
lives entirely in the drive behavior, where ``make_advancer`` samples a
per-agent action dict for ``parallel_env.step``.

Body classification mirrors the dm_control adapter: kinematic roots →
``robot`` (subtree joints). Other named bodies are not separate
entities; each contributes pose/velocity properties to the owning
robot.

    specola discover    --sim mamujoco --task HalfCheetah:2x3
    specola run --sim mamujoco --task Ant:4x2 \\
                    --specs my_specs.yaml --steps 300
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..registry import register as _register
from .dmcontrol import kinematic_root_rules
from .mujoco import MujocoAdapter


class _MaMujocoProxy:
    """Resolve MaMuJoCo's replaceable single-agent backend at reset."""

    def __init__(self, parallel_env) -> None:
        self._parallel_env = parallel_env

    @property
    def _inner(self):
        return self._parallel_env.unwrapped.single_agent_env.unwrapped

    @property
    def model(self):
        return self._inner.model

    @property
    def data(self):
        return self._inner.data

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MaMujocoAdapter(MujocoAdapter):
    """Adapter for gymnasium_robotics.mamujoco_v1 parallel environments."""

    def __init__(self, parallel_env, **kwargs) -> None:
        self._parallel_env = parallel_env
        super().__init__(_MaMujocoProxy(parallel_env), **kwargs)

    def _default_body_rules(self):
        return kinematic_root_rules(self._model)

    def make_advancer(self, mode, demo, seed):
        """
        Drive one episode for ``specola run``.

        ``hold`` gives every agent a zero action and ``random`` samples
        each agent's action box uniformly from a generator seeded with
        ``seed`` (the spaces' own ``sample()`` is not seeded by it).
        ``replay`` and ``--demo`` are refused: the adapter reads no
        demonstration format, so a demonstration file would otherwise be
        ignored while random actions ran in its place (#132).
        """
        if demo is not None:
            raise SystemExit(
                "The mamujoco adapter has no --demo replay: it reads no "
                "demonstration format."
            )
        if mode not in ("random", "hold"):
            raise SystemExit(
                f"--mode {mode} is not supported by the mamujoco adapter "
                "(no replay path); use --mode random to sample each agent's "
                "action space or --mode hold to apply zero actions."
            )
        env = self._parallel_env
        rng = np.random.default_rng(seed)

        def advance(step: int) -> bool:
            if not env.agents:
                return False
            if mode == "hold":
                actions = {
                    agent: env.action_space(agent).low * 0.0
                    for agent in env.agents
                }
            else:
                actions = {}
                for agent in env.agents:
                    space = env.action_space(agent)
                    actions[agent] = rng.uniform(
                        space.low, space.high
                    ).astype(space.dtype)
            env.step(actions)
            return bool(env.agents)

        return advance


def _make_mamujoco(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> MaMujocoAdapter:
    """Registry factory. ``task`` is ``scenario:agent_conf``, e.g.
    ``HalfCheetah:2x3`` or ``Ant:4x2``."""
    del robot, target_only

    try:
        from gymnasium_robotics import mamujoco_v1
    except ImportError as exc:
        raise ImportError(
            "gymnasium-robotics[mamujoco] is required for the "
            "'mamujoco' adapter."
        ) from exc

    scenario, _, agent_conf = (task or "HalfCheetah:2x3").partition(":")
    env = mamujoco_v1.parallel_env(scenario, agent_conf or None)
    env.reset(seed=0)
    return MaMujocoAdapter(env)


_register(
    name="mamujoco",
    factory=_make_mamujoco,
    description="Multi-Agent MuJoCo (gymnasium-robotics mamujoco_v1)",
    install_hint='pip install "gymnasium-robotics[mamujoco]"',
    requires=("gymnasium_robotics",),
)


# Bounded migration provider (maintainer decision 2026-09-07): the wrapper
# row inherits the engine provider and adds its own source custody.
from .mujoco_provider import install_bounded_provider as _install_bounded_provider  # noqa: E402

_install_bounded_provider(MaMujocoAdapter)
