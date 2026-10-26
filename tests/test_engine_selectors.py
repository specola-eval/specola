"""
test_engine_selectors.py — engine-level selector fan-out (stage 3, task 4).

MonitorEngine.start_episode compiles each spec's formula against the
episode's catalog with semantic.compile_formulas (plural): a spec whose
formula carries a component selector or a "**" binding may compile to
more than one monitor. Each compiled monitor gets its own manager, keyed
``f"{spec.name}{suffix}"`` where ``suffix`` is the compiled formula's
name with the authored formula text stripped off the front (empty for a
trivial, non-fanning formula — the regression case that must reproduce
today's manager key exactly).

There is deliberately no fan-out cap (explicit user decision, stage-3
plan): a spec that resolves to N monitors gets N monitors. Visibility
into that fan-out is a ``monitor_counts()`` accessor plus one summary
line printed by ``run_monitor`` at episode start — never a limit and
never a warning.

Fixtures echo Task 3's robot entity (see test_spec_selectors.py) with
one property nudged above threshold so a bare-selector fan-out produces
one passing key and one violated key, exercising per-key robustness and
a violated key naming itself in the engine's report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from specola.adapter import PropertyMeta
from specola.components import ComponentCatalog
from specola.engine import MonitorEngine
from specola.entities import EntityState, TrackState
from specola.run_monitor import _monitor_count_line


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entity(entity_id, entity_class, properties):
    return EntityState(
        entity_id=entity_id,
        entity_class=entity_class,
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def _entities():
    # qacc_a passes robot_*[qacc_*] <= 50 (rho=+20); qacc_b violates it
    # (rho=-10) — one passing key, one violated key, same wildcard slot.
    return {1: _entity(1, "robot", {"qacc_a": 30.0, "qacc_b": 60.0})}


def _metadata():
    joint = PropertyMeta(group="joint angular acceleration")
    return {1: {"qacc_a": joint, "qacc_b": joint}}


def _catalog():
    return ComponentCatalog.from_entities(_entities(), metadata=_metadata())


def _obs(t):
    return SimpleNamespace(
        timestep=t, time=float(t), perceived_entities=_entities()
    )


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "specs.yaml"
    path.write_text(
        "specs:\n"
        "- name: qacc\n"
        "  formula: \"robot_*[qacc_*] <= 50\"\n"
        "- name: plain\n"
        "  formula: \"robot_*[qacc_a] <= 100\"\n"
    )
    return str(path)


# ---------------------------------------------------------------------------
# Fan-out produces distinguishable managers
# ---------------------------------------------------------------------------

def test_bare_fanout_yields_distinguishable_managers(spec_file):
    engine = MonitorEngine.load(spec_file)
    engine.start_episode(dt=1.0, catalog=_catalog())

    assert set(engine.managers) == {
        "qacc [qacc_a@1]",
        "qacc [qacc_b@1]",
        "plain",
    }


def test_trivial_spec_keeps_old_manager_key(spec_file):
    """Regression: a non-fanning spec must produce exactly today's key."""
    engine = MonitorEngine.load(spec_file)
    engine.start_episode(dt=1.0, catalog=_catalog())

    assert "plain" in engine.managers
    assert engine.managers["plain"].formula.name == "robot_*[qacc_a] <= 100"


# ---------------------------------------------------------------------------
# Per-key robustness and violation reporting
# ---------------------------------------------------------------------------

def test_per_key_robustness_and_violation_report(spec_file):
    engine = MonitorEngine.load(spec_file)
    engine.start_episode(dt=1.0, catalog=_catalog())

    result = engine.update(_obs(0))
    by_name = {ev.spec_name: ev for ev in result.evaluations}

    assert set(by_name) == {
        "qacc [qacc_a@1]",
        "qacc [qacc_b@1]",
        "plain",
    }

    a = by_name["qacc [qacc_a@1]"]
    b = by_name["qacc [qacc_b@1]"]

    assert a.verdict_final
    assert b.verdict_final
    assert a.robustness == pytest.approx(20.0)
    assert b.robustness == pytest.approx(-10.0)
    assert a.robustness >= 0
    assert b.robustness < 0   # the violated key names itself: "qacc [qacc_b@1]"


# ---------------------------------------------------------------------------
# Monitor-count visibility — no cap, pure visibility
# ---------------------------------------------------------------------------

def test_monitor_counts(spec_file):
    engine = MonitorEngine.load(spec_file)
    engine.start_episode(dt=1.0, catalog=_catalog())

    assert engine.monitor_counts() == {"qacc": 2, "plain": 1}


def test_monitor_counts_recomputed_per_episode(spec_file):
    """monitor_counts() reflects the most recent start_episode() call."""
    engine = MonitorEngine.load(spec_file)
    assert engine.monitor_counts() == {}

    engine.start_episode(dt=1.0, catalog=_catalog())
    assert engine.monitor_counts() == {"qacc": 2, "plain": 1}


def test_monitor_count_line_totals_and_only_flags_fanned_specs():
    assert _monitor_count_line({"qacc": 2, "plain": 1}) == (
        "monitors: 3  (qacc: 2)"
    )
    # No spec fanned: no parenthetical, no cap/warning tone.
    assert _monitor_count_line({"plain": 1}) == "monitors: 1"


def test_monitor_count_line_lists_every_fanned_spec():
    line = _monitor_count_line({"a": 3, "b": 1, "c": 2})
    assert line.startswith("monitors: 6  (")
    assert "a: 3" in line
    assert "c: 2" in line
    assert "b: 1" not in line
