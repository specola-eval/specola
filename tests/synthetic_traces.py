"""
synthetic_traces.py — safe-then-violating step traces for detection-
latency evaluation (§1.3 of the monitoring-evaluation-notes spec).

Reusable in-memory MonitorObservation-sequence generator: a single
scalar-property entity whose value is safe (+margin) up to some step
``t0`` and violating (-margin) from ``t0`` onward, against the
predicate ``<class>_<id>[<key>] >= 0``. A second generator injects a
suspend/resume gap (the entity vanishes from ``perceived_entities`` for
a run of steps) right after the violation, to exercise the
RESET_AND_REWARM policy (design doc §5.9).

Not simulator-specific: every observation is a plain ``SimpleNamespace``
duck-typing ``MonitorObservation`` (timestep, time, perceived_entities),
matching the construction pattern in tests/test_engine.py and
tests/test_lifecycle_rewarm.py. ``EntityState`` carries properties
only; position/velocity are omitted (optional per the current entity
model — core imposes no spatial requirement).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

from specola.entities import EntityState, TrackState


def make_entity(
    entity_id: int,
    value: float,
    entity_class: str = "sig",
    key: str = "a",
) -> EntityState:
    """One entity exposing a single scalar property."""
    return EntityState(
        entity_id=entity_id,
        entity_class=entity_class,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={key: float(value)},
    )


def make_observation(
    t: int,
    dt: float,
    values: Dict[int, float],
    entity_class: str = "sig",
    key: str = "a",
) -> SimpleNamespace:
    """One MonitorObservation-shaped snapshot.

    ``values`` maps entity_id -> scalar property value; entity IDs
    absent from ``values`` are entirely absent from
    ``perceived_entities`` (used to synthesize suspend/resume gaps —
    the entity is not present with a neutral value, it is not there).
    """
    entities = {
        eid: make_entity(eid, value, entity_class, key)
        for eid, value in values.items()
    }
    return SimpleNamespace(
        timestep=t,
        time=float(t) * dt,
        perceived_entities=entities,
    )


def step_trace(
    t0: int,
    n_steps: int,
    dt: float = 1.0,
    margin: float = 1.0,
    entity_id: int = 1,
    entity_class: str = "sig",
    key: str = "a",
) -> List[SimpleNamespace]:
    """Safe-then-violating step trace, entity present at every step.

    value(t) = +margin for t < t0 (safe: predicate ``>= 0`` holds with
    margin), -margin for t >= t0 (violating, same margin — a clean,
    injected sign flip with no ramp).
    """
    return [
        make_observation(
            t, dt,
            {entity_id: margin if t < t0 else -margin},
            entity_class, key,
        )
        for t in range(n_steps)
    ]


def step_trace_with_gap(
    t0: int,
    gap_start: int,
    gap_len: int,
    n_steps: int,
    dt: float = 1.0,
    margin: float = 1.0,
    entity_id: int = 1,
    entity_class: str = "sig",
    key: str = "a",
    recover_after_gap: bool = False,
) -> List[SimpleNamespace]:
    """``step_trace`` with the entity ABSENT for one contiguous gap.

    The entity is fully missing from ``perceived_entities`` (not fed a
    neutral value) for t in [gap_start, gap_start + gap_len) — this is
    what drives the monitor instance through SUSPENDED and back,
    exercising the RESET_AND_REWARM policy (design §5.9) rather than a
    "neutral sample" fallback.

    recover_after_gap=False (default): the underlying t0-relative
    schedule continues after the gap (still violating, when
    gap_start > t0) — exercises that detection-latency guarantees
    still hold relative to the *new* contiguous segment after a
    rewarm, even though the condition never actually recovered.

    recover_after_gap=True: values return to +margin (safe) once the
    gap ends, regardless of t0 — exercises that the pre-gap violation
    is discarded rather than carried across the gap.
    """
    gap_end = gap_start + gap_len
    observations = []
    for t in range(n_steps):
        if gap_start <= t < gap_end:
            values: Dict[int, float] = {}
        elif recover_after_gap and t >= gap_end:
            values = {entity_id: margin}
        else:
            values = {entity_id: margin if t < t0 else -margin}
        observations.append(
            make_observation(t, dt, values, entity_class, key)
        )
    return observations
