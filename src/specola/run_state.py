"""Authoritative typed presentation state for ``specola run``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .results import EngineResult, EvaluationStatus


class BindingState(str, Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    SUSPENDED = "suspended"
    REWARMING = "rewarming"
    MISSING_DATA = "missing_data"
    ERROR = "error"


class SpecificationState(str, Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    UNDETERMINED = "undetermined"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RunTransition:
    step: int
    simulator_time: float
    from_state: Optional[BindingState]
    to_state: BindingState


@dataclass(frozen=True)
class MonitorDefinition:
    manager_name: str
    specification: str
    authored_formula: str = ""
    resolved_formula: str = ""
    backend: Optional[str] = None
    window_steps: Optional[int] = None
    entity_classes: Tuple[str, ...] = ()
    components_by_slot: Tuple[Tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class BindingEntity:
    entity_id: int
    entity_class: str
    label: Optional[str] = None
    role: Optional[str] = None
    role_basis: Optional[str] = None
    role_provenance: Optional[str] = None
    declared_name: Optional[str] = None


@dataclass(frozen=True)
class BindingComponent:
    entity_id: int
    key: str
    value: Optional[float] = None
    unit: Optional[str] = None
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BindingSnapshot:
    manager_name: str
    specification: str
    entity_ids: Tuple[int, ...]
    binding_key: str
    label: str
    state: BindingState
    active: bool
    current_robustness: Optional[float]
    final: bool
    final_robustness: Optional[float]
    final_step: Optional[int]
    final_time: Optional[float]
    worst_robustness: Optional[float]
    worst_step: Optional[int]
    worst_time: Optional[float]
    first_violation_step: Optional[int]
    first_violation_time: Optional[float]
    trace: Tuple[Optional[float], ...]
    transitions: Tuple[RunTransition, ...]
    step: Optional[int] = None
    simulator_time: Optional[float] = None
    authored_formula: str = ""
    resolved_formula: str = ""
    backend: Optional[str] = None
    window_steps: Optional[int] = None
    segment_samples: Optional[int] = None
    error_message: Optional[str] = None
    entities: Tuple[BindingEntity, ...] = ()
    components: Tuple[BindingComponent, ...] = ()
    unit: Optional[str] = None


@dataclass(frozen=True)
class SpecificationSnapshot:
    name: str
    state: SpecificationState
    bindings: Tuple[BindingSnapshot, ...]
    state_counts: Tuple[Tuple[BindingState, int], ...]
    skip_reason: Optional[str] = None
    active_count: int = 0
    seen_count: int = 0
    authored_formula: str = ""


@dataclass(frozen=True)
class RunSnapshot:
    step: int
    simulator_time: float
    specifications: Tuple[SpecificationSnapshot, ...]
    transitions: Tuple[Tuple[str, str, RunTransition], ...]
    wall_time: Optional[float] = None
    completion: Optional[str] = None
    result_path: Optional[str] = None


@dataclass
class _MutableBinding:
    manager_name: str
    specification: str
    entity_ids: Tuple[int, ...]
    binding_key: str
    label: str
    state: BindingState
    active: bool
    current_robustness: Optional[float]
    final: bool
    final_robustness: Optional[float]
    final_step: Optional[int]
    final_time: Optional[float]
    worst_robustness: Optional[float]
    worst_step: Optional[int]
    worst_time: Optional[float]
    first_violation_step: Optional[int]
    first_violation_time: Optional[float]
    trace: list
    transitions: list
    transitions_snapshot: Optional[Tuple[RunTransition, ...]]
    step: Optional[int]
    simulator_time: Optional[float]
    authored_formula: str
    resolved_formula: str
    backend: Optional[str]
    window_steps: Optional[int]
    segment_samples: Optional[int]
    error_message: Optional[str]
    entities: Tuple[BindingEntity, ...]
    components: Tuple[BindingComponent, ...]
    unit: Optional[str]

    def append_transition(self, transition: RunTransition) -> None:
        self.transitions.append(transition)
        self.transitions_snapshot = None

    def snapshot(self) -> BindingSnapshot:
        if self.transitions_snapshot is None:
            self.transitions_snapshot = tuple(self.transitions)
        return BindingSnapshot(
            manager_name=self.manager_name,
            specification=self.specification,
            entity_ids=self.entity_ids,
            binding_key=self.binding_key,
            label=self.label,
            state=self.state,
            active=self.active,
            current_robustness=self.current_robustness,
            final=self.final,
            final_robustness=self.final_robustness,
            final_step=self.final_step,
            final_time=self.final_time,
            worst_robustness=self.worst_robustness,
            worst_step=self.worst_step,
            worst_time=self.worst_time,
            first_violation_step=self.first_violation_step,
            first_violation_time=self.first_violation_time,
            trace=tuple(self.trace),
            transitions=self.transitions_snapshot,
            step=self.step,
            simulator_time=self.simulator_time,
            authored_formula=self.authored_formula,
            resolved_formula=self.resolved_formula,
            backend=self.backend,
            window_steps=self.window_steps,
            segment_samples=self.segment_samples,
            error_message=self.error_message,
            entities=self.entities,
            components=self.components,
            unit=self.unit,
        )


def recovery_transitions(
    transitions: Tuple[RunTransition, ...],
) -> Tuple[RunTransition, ...]:
    return tuple(
        transition for transition in transitions
        if transition.from_state is BindingState.VIOLATED
        and transition.to_state is BindingState.SATISFIED
    )


def _state_for(evaluation, rewarming: bool) -> BindingState:
    if evaluation.status is EvaluationStatus.ERROR:
        return BindingState.ERROR
    if evaluation.status is EvaluationStatus.UNKNOWN_MISSING_DATA:
        return BindingState.MISSING_DATA
    if evaluation.status is EvaluationStatus.SUSPENDED:
        return BindingState.SUSPENDED
    if (
        evaluation.status is EvaluationStatus.WARMING_UP
        or not evaluation.verdict_final
    ):
        return BindingState.REWARMING if rewarming else BindingState.PENDING
    if evaluation.robustness is None:
        return BindingState.MISSING_DATA
    return (
        BindingState.VIOLATED
        if evaluation.robustness < 0
        else BindingState.SATISFIED
    )


class RunStateTracker:
    """Accumulate immutable presentation snapshots from EngineResult."""

    def __init__(
        self,
        manager_specs: Mapping[str, str],
        *,
        skipped_specs: Optional[Mapping[str, str]] = None,
        labels: Optional[Mapping[int, str]] = None,
        definitions: Optional[Mapping[str, MonitorDefinition]] = None,
        entity_details: Optional[Mapping[int, BindingEntity]] = None,
        component_metadata: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
        trace_limit: int = 80,
    ) -> None:
        self._manager_specs = dict(manager_specs)
        self._skipped = dict(skipped_specs or {})
        # Kept by reference deliberately: the run CLI passes a live
        # label view that follows the adapter's population, so entities
        # born mid-episode display their real labels; a copy would
        # freeze the reset-time snapshot.
        self._labels = labels if labels is not None else {}
        self._definitions = dict(definitions or {})
        self._entity_details = dict(entity_details or {})
        self._component_metadata = dict(component_metadata or {})
        self._trace_limit = trace_limit
        self._bindings: Dict[Tuple[str, Tuple[int, ...]], _MutableBinding] = {}
        self._bindings_by_spec: Dict[str, List[_MutableBinding]] = {}
        self._spec_order = []
        for specification in self._manager_specs.values():
            if specification not in self._spec_order:
                self._spec_order.append(specification)
            self._bindings_by_spec.setdefault(specification, [])
        for specification in self._skipped:
            if specification not in self._spec_order:
                self._spec_order.append(specification)
            self._bindings_by_spec.setdefault(specification, [])

    def _label(self, entity_ids: Tuple[int, ...]) -> str:
        return " ↔ ".join(
            self._labels.get(entity_id, "e{}".format(entity_id))
            for entity_id in entity_ids
        ) or "global"

    def _definition(self, manager_name: str) -> MonitorDefinition:
        return self._definitions.get(
            manager_name,
            MonitorDefinition(
                manager_name=manager_name,
                specification=self._manager_specs.get(
                    manager_name, manager_name
                ),
            ),
        )

    def _binding_details(
        self,
        definition: MonitorDefinition,
        binding: Tuple[int, ...],
        component_values=(),
    ) -> Tuple[
        Tuple[BindingEntity, ...],
        Tuple[BindingComponent, ...],
        Optional[str],
    ]:
        entities = []
        components = []
        values = {
            (entity_id, key): value
            for entity_id, key, value in component_values
        }
        for index, entity_id in enumerate(binding):
            entity_class = (
                definition.entity_classes[index]
                if index < len(definition.entity_classes)
                else "entity"
            )
            detail = self._entity_details.get(entity_id)
            if detail is None:
                detail = BindingEntity(
                    entity_id=entity_id,
                    entity_class=entity_class,
                    label=self._labels.get(entity_id),
                )
            elif not detail.entity_class and entity_class:
                detail = BindingEntity(
                    entity_id=detail.entity_id,
                    entity_class=entity_class,
                    label=detail.label,
                    role=detail.role,
                    role_basis=detail.role_basis,
                    role_provenance=detail.role_provenance,
                    declared_name=detail.declared_name,
                )
            entities.append(detail)
            keys = (
                definition.components_by_slot[index]
                if index < len(definition.components_by_slot)
                else ()
            )
            metadata = self._component_metadata.get(entity_id, {})
            for key in keys:
                meta = metadata.get(key)
                declared_unit = getattr(meta, "unit", None)
                components.append(BindingComponent(
                    entity_id=entity_id,
                    key=key,
                    value=values.get((entity_id, key)),
                    unit=(
                        str(declared_unit)
                        if declared_unit is not None else None
                    ),
                    tags=tuple(getattr(meta, "tags", ()) or ()),
                ))
        units = {
            component.unit for component in components
            if component.unit is not None
        }
        unit = next(iter(units)) if len(units) == 1 else None
        return tuple(entities), tuple(components), unit

    def update(self, result: EngineResult) -> RunSnapshot:
        resumed = {(name, tuple(binding)) for name, binding in result.resumed}
        seen = set()
        emitted = []
        for evaluation in result.evaluations:
            binding = tuple(evaluation.binding)
            key = (evaluation.spec_name, binding)
            seen.add(key)
            specification = self._manager_specs.get(
                evaluation.spec_name, evaluation.spec_name
            )
            definition = self._definition(evaluation.spec_name)
            previous = self._bindings.get(key)
            is_rewarming = key in resumed or (
                previous is not None
                and previous.state in (
                    BindingState.SUSPENDED,
                    BindingState.REWARMING,
                )
            )
            state = _state_for(evaluation, is_rewarming)
            if previous is None:
                entity_records, components, unit = self._binding_details(
                    definition, binding, evaluation.component_values
                )
                identity_parts = [definition.manager_name]
                identity_parts.extend(
                    "{}:{}".format(entity.entity_class, entity.entity_id)
                    for entity in entity_records
                )
                identity_parts.extend(
                    component.key for component in components
                )
                previous = _MutableBinding(
                    manager_name=evaluation.spec_name,
                    specification=specification,
                    entity_ids=binding,
                    binding_key="::".join(identity_parts),
                    label=self._label(binding),
                    state=state,
                    active=True,
                    current_robustness=evaluation.robustness,
                    final=evaluation.verdict_final,
                    final_robustness=None,
                    final_step=None,
                    final_time=None,
                    worst_robustness=None,
                    worst_step=None,
                    worst_time=None,
                    first_violation_step=None,
                    first_violation_time=None,
                    trace=[],
                    transitions=[],
                    transitions_snapshot=None,
                    step=result.timestep,
                    simulator_time=result.time,
                    authored_formula=definition.authored_formula,
                    resolved_formula=(
                        definition.resolved_formula
                        or definition.authored_formula
                    ),
                    backend=definition.backend,
                    window_steps=(
                        evaluation.window_steps
                        if evaluation.window_steps is not None
                        else definition.window_steps
                    ),
                    segment_samples=evaluation.segment_samples,
                    error_message=evaluation.error_message,
                    entities=entity_records,
                    components=components,
                    unit=unit,
                )
                self._bindings[key] = previous
                self._bindings_by_spec.setdefault(
                    specification, []
                ).append(previous)

            if not previous.transitions or previous.state is not state:
                transition = RunTransition(
                    step=result.timestep,
                    simulator_time=result.time,
                    from_state=(
                        previous.state if previous.transitions else None
                    ),
                    to_state=state,
                )
                previous.append_transition(transition)
                emitted.append((
                    specification, previous.binding_key, transition
                ))

            previous.state = state
            previous.active = True
            previous.current_robustness = evaluation.robustness
            previous.final = evaluation.verdict_final
            previous.step = result.timestep
            previous.simulator_time = result.time
            previous.segment_samples = evaluation.segment_samples
            if evaluation.window_steps is not None:
                previous.window_steps = evaluation.window_steps
            previous.error_message = evaluation.error_message
            values = {
                (entity_id, key): value
                for entity_id, key, value in evaluation.component_values
            }
            if values:
                previous.components = tuple(
                    BindingComponent(
                        entity_id=component.entity_id,
                        key=component.key,
                        value=values.get(
                            (component.entity_id, component.key),
                            component.value,
                        ),
                        unit=component.unit,
                        tags=component.tags,
                    )
                    for component in previous.components
                )
            rho = evaluation.robustness
            if evaluation.verdict_final and rho is not None:
                previous.final_robustness = rho
                previous.final_step = result.timestep
                previous.final_time = result.time
            previous.trace.append(rho)
            del previous.trace[:-self._trace_limit]
            if rho is not None and math.isfinite(rho):
                if (
                    previous.worst_robustness is None
                    or rho < previous.worst_robustness
                ):
                    previous.worst_robustness = rho
                    previous.worst_step = result.timestep
                    previous.worst_time = result.time
            if (
                state is BindingState.VIOLATED
                and previous.first_violation_step is None
            ):
                previous.first_violation_step = result.timestep
                previous.first_violation_time = result.time

        for key, binding in self._bindings.items():
            if key in seen or not binding.active:
                continue
            transition = RunTransition(
                step=result.timestep,
                simulator_time=result.time,
                from_state=binding.state,
                to_state=BindingState.SUSPENDED,
            )
            binding.state = BindingState.SUSPENDED
            binding.active = False
            binding.current_robustness = None
            binding.final = False
            binding.step = result.timestep
            binding.simulator_time = result.time
            binding.append_transition(transition)
            emitted.append((binding.specification, binding.binding_key, transition))

        return self.snapshot(result.timestep, result.time, tuple(emitted))

    def snapshot(
        self,
        step: int,
        simulator_time: float,
        transitions=(),
    ) -> RunSnapshot:
        specifications = []
        for name in self._spec_order:
            if name in self._skipped:
                specifications.append(SpecificationSnapshot(
                    name=name,
                    state=SpecificationState.NOT_APPLICABLE,
                    bindings=(),
                    state_counts=(),
                    skip_reason=self._skipped[name],
                    authored_formula="",
                ))
                continue
            bindings = tuple(
                binding.snapshot()
                for binding in self._bindings_by_spec.get(name, ())
            )
            active = tuple(binding for binding in bindings if binding.active)
            count_by_state = {
                state: 0 for state in BindingState
            }
            for binding in active:
                count_by_state[binding.state] += 1
            counts = tuple(
                (state, count_by_state[state])
                for state in BindingState
                if count_by_state[state]
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
