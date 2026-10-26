"""
No exception may be converted into an apparently valid robustness value.

Design-spec rules: temporal-backend availability is resolved when a
specification is compiled (manager construction), never during an
arbitrary monitor update; runtime evaluation errors never silently
return stale robustness.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import specola.monitor as monitor
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.monitor import IncrementalBindingManager
from specola.stl_parser import parse_stl


def _observation(t, value):
    entity = EntityState(
        entity_id=1,
        entity_class="sig",
        position=np.zeros(3),
        velocity=np.zeros(3),
        track_age=t,
        track_state=TrackState.CONFIRMED,
        properties={"a": value},
    )
    return SimpleNamespace(
        timestep=t, time=float(t), perceived_entities={1: entity}
    )


def test_missing_rtamt_fails_at_manager_construction_for_legacy_formula(
    monkeypatch,
):
    """Legacy hand-constructed nested-RTAMT formulas (no backend factory)
    fail at manager construction when RTAMT is missing. Compiled formulas
    resolve their backend at parse_stl time instead (see
    tests/test_backend_selection.py)."""
    monkeypatch.setattr(monitor, "_rtamt_available", lambda: False)

    def mu(binding, resolver):
        return 1.0

    legacy = monitor.SpecTemplate(
        params=[],
        predicate_fn=mu,
        predicate_fns={"mu_0": mu},
        rtamt_spec="historically[0,2](mu_0)",
    )

    with pytest.raises(RuntimeError, match="RTAMT"):
        IncrementalBindingManager(legacy)


def test_backend_error_enters_terminal_error_state_never_stale_rho():
    """Design §6: an unrecoverable backend error marks the instance
    ERROR; robustness is None from then on — never the stale last_rho —
    and the instance is never updated again."""
    parsed = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0)
    manager = IncrementalBindingManager(parsed)

    first = manager.update(_observation(0, 1.0))
    assert next(iter(first.values())) is not None

    instance = next(iter(manager._instances.values()))
    calls = []

    def broken_update(step, values):
        calls.append(step)
        raise RuntimeError("backend exploded")

    instance._backend.update = broken_update

    second = manager.update(_observation(1, 1.0))
    assert next(iter(second.values())) is None      # not stale last_rho
    assert instance.errored
    assert "backend exploded" in instance.error_message

    third = manager.update(_observation(2, 1.0))
    assert next(iter(third.values())) is None
    assert len(calls) == 1                          # never updated again


def test_engine_reports_error_status():
    from types import SimpleNamespace as _SN

    from specola.results import EvaluationStatus

    parsed = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0)
    manager = IncrementalBindingManager(parsed)

    from specola.engine import MonitorEngine

    engine = MonitorEngine.__new__(MonitorEngine)
    engine._specs = []
    engine._managers = {"spec": manager}
    engine._prev_bindings = {"spec": set()}

    engine.update(_observation(0, 1.0))
    instance = next(iter(manager._instances.values()))
    instance._backend.update = lambda t, s: (_ for _ in ()).throw(
        RuntimeError("boom")
    )

    result = engine.update(_observation(1, 1.0))
    assert result.evaluations[0].status is EvaluationStatus.ERROR
    assert result.evaluations[0].robustness is None


def test_spec_config_error_still_propagates():
    parsed = parse_stl("G[0,3](sig_1[a] >= 0)", dt=1.0)
    manager = IncrementalBindingManager(parsed)
    manager.update(_observation(0, 1.0))

    with pytest.raises(SpecConfigError):
        manager.update(_observation(1, float("nan")))
