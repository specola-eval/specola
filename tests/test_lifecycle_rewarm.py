"""
Reset-and-rewarm suspension semantics (design stage 7, decision 6).

While suspended: lifecycle counters advance, the temporal backend is
never called, nothing is fabricated. On resumption within the retention
horizon: same instance, backend reset, current observation is sample
zero of the new contiguous segment, verdict_final False until the
window refills. verdict_final is per-instance (contiguous samples since
last segment start), not a global episode step count.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from specola.entities import EntityState, TrackState
from specola.monitor import IncrementalBindingManager
from specola.stl_parser import parse_stl


def obs(t, value=None):
    entities = {}
    if value is not None:
        entities[1] = EntityState(
            entity_id=1, entity_class="sig",
            position=np.zeros(3), velocity=np.zeros(3),
            track_age=t, track_state=TrackState.CONFIRMED,
            properties={"a": float(value)},
        )
    return SimpleNamespace(timestep=t, time=float(t), perceived_entities=entities)


def run(manager, sequence):
    out = []
    for t, value in enumerate(sequence):
        results = manager.update(obs(t, value))
        out.append(results)
    return out


def only_instance(manager):
    return next(iter(manager._instances.values()))


def test_suspension_never_calls_the_backend():
    f = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0, valid_classes=["sig"])
    m = IncrementalBindingManager(f)
    m.update(obs(0, 1.0))
    inst = only_instance(m)

    calls = []
    original = inst._backend.update
    inst._backend.update = lambda t, s: calls.append(t) or original(t, s)

    m.update(obs(1, None))  # suspended
    assert calls == []
    assert inst.suspended and inst.steps_suspended == 1


def test_resume_rewarms_from_fresh_segment():
    f = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0, valid_classes=["sig"])
    m = IncrementalBindingManager(f)

    # active with violating history, then a gap, then clean values
    m.update(obs(0, -5.0))
    inst = only_instance(m)
    m.update(obs(1, None))            # suspend (within retention=3)
    r2 = m.update(obs(2, 2.0))        # resume: sample zero of new segment
    assert only_instance(m) is inst   # identity retained
    assert next(iter(r2.values())) == pytest.approx(2.0)  # -5 history gone
    assert not inst.verdict_final     # rewarming
    m.update(obs(3, 2.0))
    r4 = m.update(obs(4, 2.0))        # third contiguous sample: window full
    assert inst.verdict_final
    assert next(iter(r4.values())) == pytest.approx(2.0)


def test_eventually_absence_is_not_violation():
    f = parse_stl("F[0,4](sig_1[a] >= 0)", dt=1.0, valid_classes=["sig"])
    m = IncrementalBindingManager(f)
    m.update(obs(0, 3.0))
    m.update(obs(1, None))            # old behavior fed -inf here
    r = m.update(obs(2, -1.0))
    # fresh segment: value is -1 (provisional), not contaminated by -inf
    assert next(iter(r.values())) == pytest.approx(-1.0)


def test_nested_formula_survives_gap_within_retention():
    f = parse_stl(
        "G[0,4](F[0,2](sig_1[a] >= 0))", dt=1.0, valid_classes=["sig"]
    )
    m = IncrementalBindingManager(f)
    m.update(obs(0, 1.0))
    inst = only_instance(m)
    m.update(obs(1, None))
    m.update(obs(2, 1.0))
    assert only_instance(m) is inst   # retained, no immediate prune
    assert m.n_pruned == 0


def test_retention_horizon_prunes():
    f = parse_stl("G[0,2](sig_1[a] >= 0)", dt=1.0, valid_classes=["sig"])
    m = IncrementalBindingManager(f)
    m.update(obs(0, 1.0))
    for t in (1, 2, 3):
        m.update(obs(t, None))
    assert m.n_pruned == 1


def test_verdict_final_is_per_instance_not_global_step():
    f = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0, valid_classes=["sig"])
    m = IncrementalBindingManager(f)
    m.update(obs(0, None))            # entity not present yet
    m.update(obs(1, None))
    m.update(obs(2, None))
    m.update(obs(3, None))
    m.update(obs(4, 1.0))             # binding created at global step 4
    inst = only_instance(m)
    assert not inst.verdict_final     # only 1 sample despite step >= window
    m.update(obs(5, 1.0))
    m.update(obs(6, 1.0))
    assert inst.verdict_final         # exactly W=3 contiguous samples
