"""
entities.py — Simulator-Neutral Entity Ontology
================================================

Canonical runtime representation of entities exposed to the STL monitor.

The adapter is responsible for discovering entities and producing one
EntityState per perceived entity at each timestep.

Design invariants
-----------------
- entity_id is stable for the duration of an episode.
- entity_class is an adapter-defined string.
- position and velocity are optional NumPy vectors; the core resolves specs
  from properties only, so an entity with no spatial notion may omit both.
- properties contains arbitrary scalar simulator state.
- Property keys may appear or disappear between observations.
- TrackState supports both ground-truth simulation and perception-based inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np


class SpecConfigError(ValueError):
    """
    Raised when a specification requests state that the connected adapter
    cannot provide.

    Examples include:
      - referencing a property that no eligible entity currently provides;
      - using an entity class outside a closed adapter vocabulary;
      - requesting a component that is not among an entity's properties.

    Configuration errors must fail loudly rather than being treated as
    transient monitor-evaluation failures.
    """


class UnknownClassError(SpecConfigError):
    """A formula references an entity class outside the episode's
    class vocabulary. The one spec error a caller may choose to treat
    as "not applicable to this task" rather than fatal
    (MonitorEngine.start_episode(skip_unsatisfiable=True))."""


class UnboundStaticSlotError(SpecConfigError):
    """A wildcard slot on a static class has no candidate entity.

    A class is static when the adapter's class vocabulary is CLOSED and
    the class is not listed in
    ``AdapterCapabilities.runtime_populated_classes``: its instance
    population is fixed at episode start, so no entity exposing the
    slot's required components can appear later. Unlike
    ``UnknownClassError`` this is never skipped by
    ``MonitorEngine.start_episode(skip_unsatisfiable=True)``, so
    ``Monitor.watch()`` raises rather than compiling a monitor that can
    never bind."""


class TrackState(str, Enum):
    """
    Perception lifecycle state for an entity track.

    Ground-truth simulator adapters will normally emit CONFIRMED entities.
    Perception-based adapters may use TENTATIVE and LOST to prevent unstable
    detections from immediately creating, suspending, or pruning monitors.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class EntityState:
    """
    State s_e(t) of one entity at one timestep.

    Parameters
    ----------
    entity_id:
        Integer identifier that remains stable for the duration of an episode.

    entity_class:
        Adapter-defined category such as ``"robot"``, ``"object"``,
        ``"gripper"``, or ``"region"``.

    position:
        Spatial position vector of length ``adapter.spatial_dims``, or
        ``None`` for an entity with no spatial notion. The core imposes no
        spatial requirement; spec resolution only reads ``properties``.

    velocity:
        Spatial velocity vector of length ``adapter.spatial_dims``, or
        ``None`` for an entity with no spatial notion.

    track_age:
        Number of timesteps for which the entity has been tracked.

    track_state:
        Current perception lifecycle state.

    properties:
        Arbitrary scalar state exposed by the adapter. Examples include
        joint positions, joint velocities, applied forces, gripper actuation,
        region half-extents, and task-specific physical quantities.

        Property values and keys may change at every timestep. The binding
        lifecycle handles temporary ineligibility when a required key is
        absent; the base adapter does not impose a fixed property-key schema.

        A dynamic contact can therefore be represented as a transient entity
        or property. This data model does not by itself establish stable
        contact identity or make a per-contact representation semantically
        useful; an adapter needs an explicit design for those decisions.
    """

    entity_id: int
    entity_class: str
    track_age: int
    track_state: TrackState
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    properties: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=float)

        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float)

    @property
    def is_confirmed(self) -> bool:
        """Whether the entity is eligible for confirmed-only bindings."""
        return self.track_state == TrackState.CONFIRMED
