import json
from dataclasses import replace

from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_inspect import binding_json
from specola.run_state import BindingState, RunStateTracker, RunTransition


def test_selected_binding_json_uses_typed_snapshot():
    evaluation = BindingEvaluation(
        spec_name="manager", binding=(4,), robustness=-0.25,
        status=EvaluationStatus.ACTIVE, verdict_final=True,
        timestep=3, time=0.3,
    )
    tracker = RunStateTracker({"manager": "authored"})
    snapshot = tracker.update(EngineResult(
        timestep=3, time=0.3, evaluations=(evaluation,),
    ))
    binding = snapshot.specifications[0].bindings[0]

    document = json.loads(binding_json(binding))

    assert document["schema_version"] == 1
    assert document["specification"] == "authored"
    assert document["state"]["status"] == "violated"
    assert document["binding"]["entity_ids"] == [4]
    assert document["binding"]["binding_key"] == "manager::entity:4"
    assert document["binding"]["display"] == "e4"
    assert document["robustness"]["current"] == -0.25
    assert document["robustness"]["final"] == -0.25
    assert document["robustness"]["current_verdict_final"] is True


def test_selected_binding_json_encodes_nonfinite_history_as_strings():
    evaluation = BindingEvaluation(
        spec_name="manager", binding=(4,), robustness=-0.25,
        status=EvaluationStatus.ACTIVE, verdict_final=True,
        timestep=3, time=0.3,
    )
    binding = RunStateTracker({"manager": "authored"}).update(
        EngineResult(
            timestep=3,
            time=0.3,
            evaluations=(evaluation,),
        )
    ).specifications[0].bindings[0]
    binding = replace(
        binding,
        first_violation_time=float("nan"),
        transitions=(
            RunTransition(
                step=3,
                simulator_time=float("inf"),
                from_state=None,
                to_state=BindingState.VIOLATED,
            ),
        ),
    )

    document = json.loads(binding_json(binding))

    assert document["history"]["first_violation_time"] == "NaN"
    assert document["history"]["transitions"][0][
        "simulator_time"
    ] == "Infinity"
