"""Shared sim-free fixtures for the dashboard test modules.

Snapshots are produced by driving RunStateTracker like the run pipeline
does, never by hand-building RunSnapshot objects.
"""

from __future__ import annotations

from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_state import RunStateTracker

MANIFEST = {
    "format_version": 1,
    "run_id": "r1",
    "simulator": "mock",
    "task": "t",
    "mode": "live",
    "cameras": ["cam"],
    "capture_size": [640, 480],
    "stride": 1,
    "timebase": "step+sim_time",
    "created_wall_time": 1.0,
    "status": None,
}


def _evaluation(step, robustness, status=EvaluationStatus.ACTIVE):
    return BindingEvaluation(
        spec_name="no drops",
        binding=(2,),
        robustness=robustness,
        status=status,
        # A non-final ACTIVE verdict stays PENDING (run_state._state_for);
        # final ACTIVE verdicts produce SATISFIED/VIOLATED transitions.
        verdict_final=status is EvaluationStatus.ACTIVE,
        timestep=step,
        time=step * 0.05,
        segment_samples=4,
        window_steps=20,
    )


def make_snapshots(steps=3):
    """pending, then alternating satisfied/violated, over ``steps`` steps."""
    tracker = RunStateTracker({"no drops": "no drops"}, labels={2: "onion"})
    snapshots = []
    for index in range(steps):
        if index == 0:
            evaluation = _evaluation(0, 0.4, EvaluationStatus.WARMING_UP)
        elif index % 2:
            evaluation = _evaluation(index, 0.5)
        else:
            evaluation = _evaluation(index, -0.2)
        snapshots.append(tracker.update(EngineResult(
            timestep=index,
            time=index * 0.05,
            evaluations=(evaluation,),
        )))
    return snapshots
