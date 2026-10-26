"""
dmcontrol.py — DeepMind Control Suite SimulatorAdapter
======================================================
dm_control wraps MuJoCo in a ``physics`` object; the raw model/data live
at ``physics.model.ptr`` / ``physics.data.ptr``, which the generic
MujocoAdapter consumes directly. Control runs at
``env.control_timestep()`` (not the physics timestep), and actions
follow the dm_env API (``action_spec()`` returning a BoundedArray), so
drive behavior is adapter-owned via ``make_advancer``.

Body classification is computed from the kinematic tree: each root body
(parent == world) becomes class ``robot`` with subtree joint state.
Non-root bodies are not separate entities; each contributes its own
pose/velocity properties (``xpos_<body>_*``, ``xquat_<body>_*``,
``cvel_<body>_*``) to the owning robot. Suite domains vary widely
(walker, cartpole, quadruped, ...), so inspect with:

    specola discover    --sim dmcontrol --task walker:walk
    specola run --sim dmcontrol --task walker:walk \\
                    --specs my_specs.yaml --steps 300
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..registry import register as _register
from .mujoco import BodyRule, MujocoAdapter


class _PhysicsProxy:
    """Resolve the dm_control physics object at each adapter reset."""

    def __init__(self, dm_env) -> None:
        self._dm_env = dm_env

    @property
    def model(self):
        return self._dm_env.physics.model.ptr

    @property
    def data(self):
        return self._dm_env.physics.data.ptr

    @property
    def dt(self) -> float:
        return float(self._dm_env.control_timestep())


def kinematic_root_rules(model) -> Sequence[BodyRule]:
    """Classify kinematic-root bodies as ``robot`` (subtree joints).

    Non-root bodies are not separate entities: subtree bodies are
    property families on the owning ``robot`` (``xpos_<body>_*``,
    ``xquat_<body>_*``, ``cvel_<body>_*``), not entities of their own.
    """
    import mujoco

    roots = set()
    for body_id in range(1, model.nbody):
        if model.body_parentid[body_id] == 0:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                roots.add(name)

    return [
        BodyRule(
            match=lambda name, roots=frozenset(roots): name in roots,
            cls="robot",
            priority=0,
            joint_scope="subtree",
        ),
    ]


class DMControlAdapter(MujocoAdapter):
    """Adapter for dm_control suite environments."""

    def __init__(self, dm_env, **kwargs) -> None:
        self._dm_env = dm_env
        super().__init__(_PhysicsProxy(dm_env), **kwargs)

    def _default_body_rules(self) -> Sequence[BodyRule]:
        return kinematic_root_rules(self._model)

    def make_advancer(self, mode, demo, seed):
        """
        Drive one episode for ``specola run``.

        ``hold`` applies a zero action every control step and ``random``
        samples one uniformly from ``action_spec()`` with ``seed``.
        ``replay`` and ``--demo`` are refused: the adapter reads no
        demonstration format, so a demonstration file would otherwise be
        ignored while random actions ran in its place (#132).
        """
        if demo is not None:
            raise SystemExit(
                "The dmcontrol adapter has no --demo replay: it reads no "
                "demonstration format."
            )
        if mode not in ("random", "hold"):
            raise SystemExit(
                f"--mode {mode} is not supported by the dmcontrol adapter "
                "(no replay path); use --mode random to sample actions from "
                "the action spec or --mode hold to apply zero actions."
            )
        spec = self._dm_env.action_spec()
        rng = np.random.default_rng(seed)

        def advance(step: int) -> bool:
            if mode == "hold":
                action = np.zeros(spec.shape)
            else:
                action = rng.uniform(spec.minimum, spec.maximum)
            timestep = self._dm_env.step(action)
            return not timestep.last()

        return advance



def _make_dmcontrol(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> DMControlAdapter:
    """Registry factory. ``task`` is ``domain:task``, e.g. ``walker:walk``."""
    del robot, target_only

    try:
        from dm_control import suite
    except ImportError as exc:
        raise ImportError(
            "dm_control is required for the 'dmcontrol' adapter."
        ) from exc

    domain, _, task_name = (task or "walker:walk").partition(":")
    env = suite.load(domain, task_name or "walk")
    env.reset()
    return DMControlAdapter(env)


def _handles(obj) -> bool:
    """Cheap module-path sniff — never imports dm_control itself."""
    return (type(obj).__module__ or "").startswith("dm_control")


def _wrap(env, **kwargs) -> DMControlAdapter:
    """Wrap an existing dm_control ``Environment`` (adapter kwargs pass
    through, e.g. ``body_rules=`` to override the kinematic-root default)."""
    return DMControlAdapter(env, **kwargs)


_register(
    name="dmcontrol",
    factory=_make_dmcontrol,
    description="DeepMind Control Suite (dm_control, MuJoCo)",
    install_hint="pip install dm_control",
    handles=_handles,
    wrap=_wrap,
    requires=("dm_control",),
)


# Bounded migration provider (maintainer decision 2026-09-07): the wrapper
# row inherits the engine provider and adds its own source custody.
from .mujoco_provider import install_bounded_provider as _install_bounded_provider  # noqa: E402

_install_bounded_provider(DMControlAdapter)
