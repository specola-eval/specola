from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_log import render_plain_transitions
from specola.run_state import RunStateTracker


def test_plain_log_emits_literal_first_transition_without_snapshot_dump():
    tracker = RunStateTracker({"manager": "authored"}, labels={2: "onion"})
    snapshot = tracker.update(EngineResult(
        timestep=0,
        time=0.0,
        evaluations=(BindingEvaluation(
            spec_name="manager",
            binding=(2,),
            robustness=-0.1,
            status=EvaluationStatus.ACTIVE,
            verdict_final=True,
            timestep=0,
            time=0.0,
        ),),
    ))

    lines = render_plain_transitions(snapshot)

    assert lines == ["step 0     authored · onion · — → violated"]
    assert all("Specifications" not in line for line in lines)


def test_summary_only_suppresses_transition_lines():
    snapshot = RunStateTracker({"manager": "authored"}).snapshot(0, 0.0)
    assert render_plain_transitions(snapshot, summary_only=True) == []
