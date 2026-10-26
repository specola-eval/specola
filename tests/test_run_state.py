from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_state import (
    BindingState,
    MonitorDefinition,
    RunSnapshot,
    RunStateTracker,
    SpecificationSnapshot,
    SpecificationState,
    recovery_transitions,
)


BINDING = (2,)


def _result(
    step, rho, status, final=True, resumed=(),
    segment_samples=None, window_steps=None, error_message=None,
    component_values=(),
):
    evaluation = BindingEvaluation(
        spec_name="manager",
        binding=BINDING,
        robustness=rho,
        status=status,
        verdict_final=final,
        timestep=step,
        time=step * 0.1,
        segment_samples=segment_samples,
        window_steps=window_steps,
        error_message=error_message,
        component_values=component_values,
    )
    return EngineResult(
        timestep=step,
        time=step * 0.1,
        evaluations=(evaluation,),
        resumed=tuple(resumed),
    )


def test_literal_transitions_and_recovery_are_distinct():
    tracker = RunStateTracker({"manager": "authored"})
    tracker.update(_result(0, 0.4, EvaluationStatus.ACTIVE))
    tracker.update(_result(1, -0.2, EvaluationStatus.ACTIVE))
    snapshot = tracker.update(_result(2, 0.1, EvaluationStatus.ACTIVE))

    binding = snapshot.specifications[0].bindings[0]
    assert binding.state is BindingState.SATISFIED
    assert [(item.from_state, item.to_state) for item in binding.transitions] == [
        (None, BindingState.SATISFIED),
        (BindingState.SATISFIED, BindingState.VIOLATED),
        (BindingState.VIOLATED, BindingState.SATISFIED),
    ]
    assert len(recovery_transitions(binding.transitions)) == 1
    assert binding.worst_robustness == -0.2


def test_resume_enters_rewarming_until_final():
    tracker = RunStateTracker({"manager": "authored"})
    tracker.update(_result(0, None, EvaluationStatus.SUSPENDED, final=False))
    snapshot = tracker.update(_result(
        1, 0.2, EvaluationStatus.WARMING_UP, final=False,
        resumed=(("manager", BINDING),),
    ))
    assert snapshot.specifications[0].bindings[0].state is BindingState.REWARMING


def test_missing_data_is_visible_not_not_applicable():
    tracker = RunStateTracker({"manager": "authored"})
    snapshot = tracker.update(_result(
        0, None, EvaluationStatus.UNKNOWN_MISSING_DATA, final=False,
    ))
    assert snapshot.specifications[0].bindings[0].state is BindingState.MISSING_DATA


def test_temporal_readiness_is_carried_not_inferred():
    tracker = RunStateTracker({"manager": "authored"})
    snapshot = tracker.update(_result(
        100, 0.2, EvaluationStatus.WARMING_UP, final=False,
        segment_samples=3, window_steps=20,
    ))
    binding = snapshot.specifications[0].bindings[0]

    assert binding.segment_samples == 3
    assert binding.window_steps == 20
    assert binding.state is BindingState.PENDING


def test_inactive_history_is_waiting_and_revival_reuses_identity():
    tracker = RunStateTracker({"manager": "authored"})
    first = tracker.update(_result(
        0, 0.0, EvaluationStatus.ACTIVE,
    )).specifications[0].bindings[0]
    absent = tracker.update(EngineResult(
        timestep=1, time=0.1, evaluations=(),
    ))
    specification = absent.specifications[0]

    assert specification.state is SpecificationState.WAITING
    assert specification.active_count == 0
    assert specification.seen_count == 1
    assert specification.bindings[0].active is False
    assert specification.bindings[0].state is BindingState.SUSPENDED

    revived = tracker.update(_result(
        2, 0.3, EvaluationStatus.ACTIVE,
    )).specifications[0].bindings[0]
    assert revived.binding_key == first.binding_key
    assert len(revived.transitions) == 3


def test_definition_metadata_reaches_binding_snapshot():
    definition = MonitorDefinition(
        manager_name="manager",
        specification="authored",
        authored_formula="G[0,1s](object_*[mass] >= 0)",
        resolved_formula="G[0,1s](object_2[mass] >= 0)",
        backend="native",
        window_steps=20,
        entity_classes=("object",),
        components_by_slot=(("mass",),),
    )
    tracker = RunStateTracker(
        {"manager": "authored"},
        definitions={"manager": definition},
        labels={2: "onion"},
    )

    binding = tracker.update(_result(
        0, -0.1, EvaluationStatus.ACTIVE,
        component_values=((2, "mass", 0.4),),
    )).specifications[0].bindings[0]

    assert binding.authored_formula.startswith("G[0,1s]")
    assert binding.resolved_formula == "G[0,1s](object_2[mass] >= 0)"
    assert binding.backend == "native"
    assert binding.entities[0].entity_class == "object"
    assert binding.components[0].key == "mass"
    assert binding.components[0].value == 0.4
    assert binding.binding_key == "manager::object:2::mass"


def test_component_note_is_not_misreported_as_a_robustness_unit():
    definition = MonitorDefinition(
        manager_name="manager",
        specification="authored",
        entity_classes=("object",),
        components_by_slot=(("object_quat_w",),),
    )
    tracker = RunStateTracker(
        {"manager": "authored"},
        definitions={"manager": definition},
        component_metadata={
            2: {
                "object_quat_w": SimpleNamespace(
                    note="quat w",
                    unit=None,
                    tags=(),
                ),
            },
        },
    )

    binding = tracker.update(_result(
        0,
        0.95,
        EvaluationStatus.ACTIVE,
        component_values=((2, "object_quat_w", 0.95),),
    )).specifications[0].bindings[0]

    assert binding.components[0].unit is None
    assert binding.unit is None


def test_zero_robustness_is_satisfied_and_retained_as_worst():
    tracker = RunStateTracker({"manager": "authored"})
    binding = tracker.update(_result(
        0, 0.0, EvaluationStatus.ACTIVE,
    )).specifications[0].bindings[0]
    assert binding.state is BindingState.SATISFIED
    assert binding.current_robustness == 0.0
    assert binding.final_robustness == 0.0
    assert binding.worst_robustness == 0.0


def test_last_final_survives_suspension_and_rewarming():
    tracker = RunStateTracker({"manager": "authored"})
    tracker.update(_result(
        0, -0.2, EvaluationStatus.ACTIVE, final=True,
    ))
    suspended = tracker.update(_result(
        1, None, EvaluationStatus.SUSPENDED, final=False,
    )).specifications[0].bindings[0]
    binding = tracker.update(_result(
        2, 0.1, EvaluationStatus.WARMING_UP, final=False,
        resumed=(("manager", BINDING),),
    )).specifications[0].bindings[0]

    assert suspended.current_robustness is None
    assert binding.current_robustness == 0.1
    assert binding.final is False
    assert binding.final_robustness == -0.2
    assert binding.final_step == 0


def _legacy_snapshot(tracker, step, simulator_time, transitions):
    """Reproduce the pre-cache snapshot construction path."""
    specifications = []
    for name in tracker._spec_order:
        if name in tracker._skipped:
            specifications.append(SpecificationSnapshot(
                name=name,
                state=SpecificationState.NOT_APPLICABLE,
                bindings=(),
                state_counts=(),
                skip_reason=tracker._skipped[name],
                authored_formula="",
            ))
            continue
        # Rebuild the mutable-history tuples independently of the cached
        # snapshot() path so the equality below actually compares two
        # constructions rather than the cache with itself.
        bindings = tuple(
            replace(
                binding.snapshot(),
                transitions=tuple(binding.transitions),
                trace=tuple(binding.trace),
            )
            for binding in tracker._bindings.values()
            if binding.specification == name
        )
        active = tuple(binding for binding in bindings if binding.active)
        counts = tuple(
            (state, sum(
                1 for binding in active if binding.state is state
            ))
            for state in BindingState
            if any(binding.state is state for binding in active)
        )
        if not active:
            spec_state = SpecificationState.WAITING
        elif not any(binding.final for binding in active):
            spec_state = SpecificationState.UNDETERMINED
        else:
            spec_state = SpecificationState.ACTIVE
        specifications.append(SpecificationSnapshot(
            name=name,
            state=spec_state,
            bindings=bindings,
            state_counts=counts,
            active_count=len(active),
            seen_count=len(bindings),
            authored_formula=next(
                (
                    binding.authored_formula for binding in bindings
                    if binding.authored_formula
                ),
                "",
            ),
        ))
    return RunSnapshot(
        step=step,
        simulator_time=simulator_time,
        specifications=tuple(specifications),
        transitions=tuple(transitions),
    )


def test_cached_snapshot_matches_legacy_path_through_suspension_and_revival():
    tracker = RunStateTracker(
        {"manager": "authored"},
        skipped_specs={"unavailable": "UnknownClass: absent"},
    )
    episode = (
        _result(0, None, EvaluationStatus.WARMING_UP, final=False),
        _result(1, -0.2, EvaluationStatus.ACTIVE),
        EngineResult(timestep=2, time=0.2, evaluations=()),
        _result(
            3,
            0.1,
            EvaluationStatus.WARMING_UP,
            final=False,
            resumed=(("manager", BINDING),),
        ),
        _result(4, 0.3, EvaluationStatus.ACTIVE),
        # Same state again: no transition appends, so the cached
        # transitions tuple must be reused by identity.
        _result(5, 0.4, EvaluationStatus.ACTIVE),
    )

    previous_transitions = None
    for result in episode:
        snapshot = tracker.update(result)
        assert snapshot == _legacy_snapshot(
            tracker,
            result.timestep,
            result.time,
            snapshot.transitions,
        )
        transitions = snapshot.specifications[0].bindings[0].transitions
        if (
            previous_transitions is not None
            and len(transitions) == len(previous_transitions)
        ):
            assert transitions is previous_transitions
        previous_transitions = transitions
