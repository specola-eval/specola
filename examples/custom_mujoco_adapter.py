"""
Template: Specola adapter for a new MuJoCo-based simulator.

Anything built on MuJoCo (plain mujoco, robosuite, or a robosuite-based
suite like RoboCasa/LIBERO) subclasses ``MujocoAdapter``, which already
handles observation extraction, entity-ID stability, joint/property
discovery, region materialization, and contract validation. You supply
only the naming conventions of *your* simulator, then register the
adapter — no specola file changes are needed.

Try it against any sim (here shown with a robosuite-style env):

    specola write-specs  --adapter-module custom_mujoco_adapter --sim mysim
    specola run  --adapter-module custom_mujoco_adapter --sim mysim \\
                     --specs my_specs.yaml --steps 200

or make it permanently discoverable from your own package by declaring
an entry point (no --adapter-module needed afterwards):

    # your package's pyproject.toml
    [project.entry-points."specola.adapters"]
    mysim = "my_package.my_adapter"
"""

from __future__ import annotations

from typing import Optional

from specola.adapters.mujoco import BodyRule, MujocoAdapter, RegionSpec
from specola.registry import register


class MySimAdapter(MujocoAdapter):
    """Adapter for MySim: only the naming conventions live here."""

    # ------------------------------------------------------------------
    # 1. Body classification — map MuJoCo body names to entity classes.
    #    First matching rule wins. `match` is a substring or a callable
    #    body_name -> bool. `joint_scope` controls which joint state is
    #    exposed as entity properties ("none" | "body" | "subtree").
    # ------------------------------------------------------------------
    def _default_body_rules(self):
        return [
            BodyRule(
                match=lambda name: name == "my_robot_base",
                cls="robot",
                priority=0,
                joint_scope="subtree",   # whole articulated robot
            ),
            BodyRule(
                match=lambda name: name.startswith("movable_"),
                cls="object",
            ),
        ]

    # ------------------------------------------------------------------
    # 2. Gripper site (optional) — MuJoCo site exposed as its own
    #    entity of class ``gripper``. Names are searched in order.
    # ------------------------------------------------------------------
    def _default_gripper_sites(self):
        return ["my_gripper_site", "grip_site"]

    # ------------------------------------------------------------------
    # 3. Regions (optional) — static boxes exposed as ``region``
    #    entities with hw/hh/hd half-extents and x0..z1 world bounds,
    #    ready for containment formulas. Discover them from your task
    #    model, or pass explicit RegionSpec objects to __init__.
    # ------------------------------------------------------------------
    def _discover_extra_regions(self):
        return [
            RegionSpec(
                name="target_zone",
                source="fixed",              # explicit world-frame box;
                center=(1.0, 0.0, 0.9),      # other sources: "site",
                half_extents=(0.25, 0.20, 0.15),  # "geom", "int_sites"
            ),
        ]

    # ------------------------------------------------------------------
    # 4. Drive behavior (optional) — only for specola run's demo
    #    loop. Adapters for simulators whose control API is neither
    #    robosuite-style (action_spec) nor gymnasium-style
    #    (action_space) — e.g. CARLA — own their stepping here; the
    #    monitoring core never needs this (it only consumes
    #    observations). Return a callable step -> bool (False ends the
    #    episode).
    #
    #    def make_advancer(self, mode, demo, seed):
    #        rng = np.random.default_rng(seed)
    #        def advance(step):
    #            self.env.step(rng.uniform(-1, 1, self.env.action_dim))
    #            return True
    #        return advance
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5. Labels (optional) — human-readable names for CLI output,
    #    e.g. "movable_cup_3" -> "cup".
    # ------------------------------------------------------------------
    def _object_category(self, body_name: str) -> Optional[str]:
        if body_name.startswith("movable_"):
            return body_name.split("_")[1]
        return None


# ----------------------------------------------------------------------
# Registration — importing this module makes `--sim mysim` work. The
# factory builds the env itself so the simulator package is imported
# only when the adapter is actually used (keep module import light).
# ----------------------------------------------------------------------
def _make_mysim(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> MySimAdapter:
    del target_only

    import mysim  # your simulator package; imported lazily on purpose

    env = mysim.make(task or "DefaultTask", robot=robot or "DefaultBot")
    env.reset()  # the sim must have live state before adapter discovery
    return MySimAdapter(env)


register(
    name="mysim",
    factory=_make_mysim,
    description="MySim (MuJoCo-based) adapter template",
    install_hint="pip install mysim",
)
