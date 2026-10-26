from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_present import format_binding, render_run_snapshot
from specola.run_state import RunStateTracker


def _snapshot(final=False):
    tracker = RunStateTracker({"manager": "authored"}, labels={2: "onion"})
    return tracker.update(EngineResult(
        timestep=3,
        time=0.3,
        evaluations=(BindingEvaluation(
            spec_name="manager",
            binding=(2,),
            robustness=0.0,
            status=(
                EvaluationStatus.ACTIVE
                if final else EvaluationStatus.WARMING_UP
            ),
            verdict_final=final,
            timestep=3,
            time=0.3,
            segment_samples=4,
            window_steps=20,
        ),),
    ))


def test_collapsed_specification_has_counts_but_no_aggregate_robustness():
    lines = render_run_snapshot(_snapshot(final=True))
    spec_line = next(line for line in lines if "authored" in line)
    assert "1 satisfied" in spec_line
    assert "worst" not in spec_line.casefold()


def test_binding_format_retains_signed_zero_and_structural_trace():
    binding = _snapshot(final=True).specifications[0].bindings[0]
    text = format_binding(binding)
    assert "NOW=+0.000" in text
    assert "WORST=+0.000" in text
    assert "▴" not in text
