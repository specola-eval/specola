"""
fake.py — dependency-free simulator adapter for tests and demos.

The mock scene includes one robot, two heterogeneous objects, one
region with half-extents and materialized bounds, and one gripper —
exercising per-slot binding eligibility without any simulator.
"""

from __future__ import annotations

from typing import FrozenSet, Optional

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..entities import EntityState, TrackState
from ..registry import register as _register


# ---------------------------------------------------------------------------

class _MockAdapter(SimulatorAdapter):
    """
    Dependency-free adapter used for interface and monitoring tests.

    The mock scene intentionally includes:

    - one robot with joint and force properties;
    - two objects with different property sets;
    - one region with half-extents;
    - one gripper with actuation and force properties.

    The different object property sets exercise the architecture's support for
    heterogeneous entities within one class.
    """

    DT = 0.05

    entity_labels = {
        0: "mock_robot_base",
        1: "mock_target_object",
        2: "mock_distractor_object",
        3: "mock_cabinet_region",
        4: "mock_gripper_site",
    }

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    @property
    def dt(self) -> float:
        return self.DT

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def supported_classes(
        self,
    ) -> FrozenSet[str]:
        return frozenset(
            {
                "robot",
                "object",
                "region",
                "gripper",
            }
        )

    def property_metadata(
        self,
        entity_id: int,
    ):
        if entity_id != 0:
            return {}

        return {
            "qpos_joint1": PropertyMeta(
                group="joint angle",
                note="rad",
            ),
            "qvel_joint1": PropertyMeta(
                group="joint angular velocity",
                note="rad/s",
            ),
            "qfrc_applied_joint1": PropertyMeta(
                group="applied torques",
                note="N·m",
            ),
        }

    def _reset_impl(
        self,
    ) -> MonitorObservation:
        self._step = 0
        return self._observation()

    def _get_observation_impl(
        self,
    ) -> MonitorObservation:
        self._step += 1
        return self._observation()

    def _observation(
        self,
    ) -> MonitorObservation:
        time = self._step * self.DT

        entities = {
            0: EntityState(
                entity_id=0,
                entity_class="robot",
                position=np.asarray(
                    [0.0, 0.0, 0.8],
                    dtype=float,
                ),
                velocity=np.zeros(
                    3,
                    dtype=float,
                ),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties={
                    "qpos_joint1": 0.1,
                    "qvel_joint1": 0.0,
                    "qfrc_applied_joint1": 0.0,
                },
            ),
            1: EntityState(
                entity_id=1,
                entity_class="object",
                position=np.asarray(
                    [
                        0.5 + 0.01 * time,
                        0.0,
                        0.9,
                    ],
                    dtype=float,
                ),
                velocity=np.asarray(
                    [0.01, 0.0, 0.0],
                    dtype=float,
                ),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties={
                    "mass": 0.25,
                    "temperature": 22.0,
                },
            ),
            2: EntityState(
                entity_id=2,
                entity_class="object",
                position=np.asarray(
                    [
                        0.7,
                        0.2 + 0.005 * time,
                        0.9,
                    ],
                    dtype=float,
                ),
                velocity=np.asarray(
                    [0.0, 0.005, 0.0],
                    dtype=float,
                ),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties={
                    "mass": 0.40,
                    "hinge_angle": 0.15,
                },
            ),
            3: EntityState(
                entity_id=3,
                entity_class="region",
                position=np.asarray(
                    [1.0, 0.0, 1.2],
                    dtype=float,
                ),
                velocity=np.zeros(
                    3,
                    dtype=float,
                ),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties={
                    "hw": 0.25,
                    "hh": 0.20,
                    "hd": 0.15,
                    # Materialized world-frame bounds (center ± half-extent),
                    # matching the MujocoAdapter region contract.
                    "x0": 1.0 - 0.25, "x1": 1.0 + 0.25,
                    "y0": 0.0 - 0.20, "y1": 0.0 + 0.20,
                    "z0": 1.2 - 0.15, "z1": 1.2 + 0.15,
                },
            ),
            4: EntityState(
                entity_id=4,
                entity_class="gripper",
                position=np.asarray(
                    [0.3, 0.0, 1.0],
                    dtype=float,
                ),
                velocity=np.zeros(
                    3,
                    dtype=float,
                ),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties={
                    "grip": 0.02,
                    "gripper_qpos_0": 0.02,
                    "gripper_qpos_1": -0.02,
                    "gripper_qpos_mean": 0.0,
                    "gripper_qpos_abs_mean": 0.02,
                    "recent_ee_force_x": 1.0,
                    "recent_ee_force_y": 0.0,
                    "recent_ee_force_z": 2.0,
                    "recent_ee_force_norm": float(
                        np.sqrt(5.0)
                    ),
                },
            ),
        }

        return MonitorObservation(
            timestep=self._step,
            time=time,
            perceived_entities=entities,
        )




def _make_mock(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> _MockAdapter:
    del task, robot, target_only
    return _MockAdapter()


_register(
    name="mock",
    factory=_make_mock,
    description=(
        "Built-in mock simulator "
        "(no external dependencies)"
    ),
)

FakeAdapter = _MockAdapter
