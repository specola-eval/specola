from pathlib import Path
from types import SimpleNamespace

from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_state import RunStateTracker
from specola.run_summary import (
    build_run_summary,
    render_run_summary,
    undetermined_reason,
)


def test_completed_summary_uses_final_and_worst_and_absolute_result(tmp_path):
    tracker = RunStateTracker({"manager": "authored"}, labels={2: "onion"})
    tracker.update(EngineResult(
        timestep=0,
        time=0.0,
        evaluations=(BindingEvaluation(
            spec_name="manager", binding=(2,), robustness=-0.2,
            status=EvaluationStatus.ACTIVE, verdict_final=True,
            timestep=0, time=0.0,
        ),),
    ))
    snapshot = tracker.update(EngineResult(
        timestep=1,
        time=0.1,
        evaluations=(BindingEvaluation(
            spec_name="manager", binding=(2,), robustness=0.1,
            status=EvaluationStatus.ACTIVE, verdict_final=True,
            timestep=1, time=0.1,
        ),),
    ))
    result = tmp_path / "episode.parquet"
    record = SimpleNamespace(
        status="completed", steps=2, episode_id="run-1", path=result,
    )
    engine = SimpleNamespace(
        spec_provenance={"spec_digest": "abc123"},
    )

    summary = build_run_summary(record, snapshot, engine)
    text = render_run_summary(summary)

    assert summary.completion == "complete"
    assert summary.result_path == str(Path(result).resolve())
    assert "FINAL +0.100" in text
    assert "WORST -0.200" in text
    assert "1 recovery" in text


def test_undetermined_reason_uses_binding_segment_progress_not_global_steps(
    tmp_path,
):
    snapshot = RunStateTracker({"manager": "authored"}).update(EngineResult(
        timestep=100,
        time=10.0,
        evaluations=(BindingEvaluation(
            spec_name="manager",
            binding=(2,),
            robustness=0.1,
            status=EvaluationStatus.WARMING_UP,
            verdict_final=False,
            timestep=100,
            time=10.0,
            segment_samples=3,
            window_steps=20,
        ),),
    ))
    record = SimpleNamespace(
        status="completed",
        steps=101,
        episode_id="run-1",
        path=tmp_path / "episode.csv",
    )
    engine = SimpleNamespace(spec_provenance={"spec_digest": "abc123"})

    summary = build_run_summary(record, snapshot, engine)

    assert undetermined_reason(summary, "authored") == " (3/20 steps)"
