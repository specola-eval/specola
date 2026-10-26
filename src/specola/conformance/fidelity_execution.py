"""Internal paired raw execution facts for declared fidelity-case variants.

The collector records a TraceCase public-call trace; hooks interleave around
the post-reset GET. Hook payloads are raw frozen artifacts whose schemas and
simulator semantics remain owned by backend harnesses. This custody check does
not make artifacts tamper-resistant: ``artifact_tamper_resistant`` is false,
including for a hook that modifies, uses, and restores itself in one call.
"""

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from .core_runner import (
    CallExpectation, CallKind, CaseExecution, ConformanceValidationError, DriverBundle, ExceptionFact,
    FidelityCase, FrozenJsonArtifact, TraceCase, _callable_source,
    _callable_target, _exception, _manual, _python_marshal_code_sha256,
    _run_case, _source_data, _source_for_class, validate_fidelity_authority,
)


_HOOK_NAMES = ("advance", "native_snapshot", "materialization_trace", "interaction_counts")


@dataclass(frozen=True)
class FidelityHookFact:
    """One raw harness result or a captured harness failure."""

    name: str
    artifact: Optional[FrozenJsonArtifact]
    exception: Optional[ExceptionFact]

    def __post_init__(self) -> None:
        if self.name not in _HOOK_NAMES or (self.artifact is None) == (self.exception is None):
            raise ConformanceValidationError("hook fact needs one named artifact or exception")
        if self.artifact is not None and not isinstance(self.artifact, FrozenJsonArtifact):
            raise ConformanceValidationError("hook fact artifact must be FrozenJsonArtifact")
        if self.exception is not None and not isinstance(self.exception, ExceptionFact):
            raise ConformanceValidationError("hook fact exception must be ExceptionFact")


@dataclass(frozen=True)
class FidelityVariantExecution:
    """Raw facts for one exact configuration/case/variant denominator member."""

    configuration_id: str
    case_id: str
    variant_id: str
    profile_denominator: Tuple[str, ...]
    candidate_denominator: Tuple[str, ...]
    trace: CaseExecution
    hooks: Tuple[FidelityHookFact, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) and item for item in (self.configuration_id, self.case_id, self.variant_id)):
            raise ConformanceValidationError("variant execution IDs must be text")
        if not isinstance(self.profile_denominator, tuple) or not isinstance(self.candidate_denominator, tuple):
            raise ConformanceValidationError("variant execution denominators must be tuples")
        if not isinstance(self.trace, CaseExecution) or self.trace.case_id != self.case_id:
            raise ConformanceValidationError("variant execution trace must bind case")
        if tuple((item.kind, item.episode_index, item.observation_index, item.expectation) for item in self.trace.plan) != ((CallKind.RESET, 0, None, CallExpectation.RETURN_OBSERVATION), (CallKind.GET_OBSERVATION, 0, 0, CallExpectation.RETURN_OBSERVATION)):
            raise ConformanceValidationError("variant execution trace needs one reset and one post-reset observation")
        if not isinstance(self.hooks, tuple) or any(not isinstance(item, FidelityHookFact) for item in self.hooks):
            raise ConformanceValidationError("variant execution hook facts have wrong type")
        if tuple(item.name for item in self.hooks) != _HOOK_NAMES:
            raise ConformanceValidationError("variant execution needs exact ordered hook facts")


@dataclass(frozen=True)
class FidelityExecutionArtifact:
    """Raw paired execution; deliberately contains no outcome or obligation field."""

    authority: FrozenJsonArtifact
    configuration_denominator: Tuple[str, ...]
    case_denominator: Tuple[str, ...]
    executions: Tuple[FidelityVariantExecution, ...]

    def __post_init__(self) -> None:
        records = _authority_records(self.authority)
        if not isinstance(self.configuration_denominator, tuple) or not isinstance(self.case_denominator, tuple) or not isinstance(self.executions, tuple):
            raise ConformanceValidationError("execution artifact denominators and facts must be tuples")
        if any(not isinstance(item, FidelityVariantExecution) for item in self.executions):
            raise ConformanceValidationError("execution artifact facts have wrong type")
        pairs = tuple((item["case_id"], item["variant_id"]) for item in records)
        if tuple((item.case_id, item.variant_id) for item in self.executions) != pairs:
            raise ConformanceValidationError("execution denominator must exactly match authority variants")
        if tuple(dict.fromkeys(item["configuration_id"] for item in records)) != self.configuration_denominator or tuple(dict.fromkeys(item["case_id"] for item in records)) != self.case_denominator:
            raise ConformanceValidationError("execution configuration/case denominator disagrees with authority")
        for execution, record in zip(self.executions, records):
            if (execution.configuration_id != record["configuration_id"] or list(execution.profile_denominator) != record["profile_denominator"] or list(execution.candidate_denominator) != record["candidate_denominator"]):
                raise ConformanceValidationError("execution facts disagree with authority record")


def run_fidelity_execution(authority, cases, traces):
    """Collect one fresh paired raw trace and four hook facts per declared variant."""

    records = _authority_records(authority)
    if not isinstance(cases, tuple) or not isinstance(traces, tuple) or not cases or not traces:
        raise ConformanceValidationError("fidelity cases and traces must be non-empty tuples")
    if any(not isinstance(item, FidelityCase) for item in cases) or any(not isinstance(item, TraceCase) for item in traces):
        raise ConformanceValidationError("fidelity cases and traces have wrong type")
    if tuple(item.case_id for item in cases) != tuple(item.case_id for item in traces):
        raise ConformanceValidationError("fidelity traces must exactly match case denominator")
    if any(item.episode_count != 1 or item.observations_per_episode != 1 or item.probe_observe_before_reset for item in traces):
        raise ConformanceValidationError("paired execution needs one reset and one post-reset observation")
    expected = tuple((case.case_id, variant) for case in cases for variant in case.variant_denominator)
    if tuple((item["case_id"], item["variant_id"]) for item in records) != expected:
        raise ConformanceValidationError("authority variants must exactly match declared cases")
    by_pair = {(item["case_id"], item["variant_id"]): item for item in records}
    seen_adapters, seen_producers, retained, executions = [], [], [], []
    factory_state, hook_state = [], []
    for case, trace in zip(cases, traces):
        for variant in case.variant_denominator:
            record = by_pair[(case.case_id, variant)]
            _validate_factory(case, record, factory_state)
            try:
                try:
                    bundle = case.bundle_factory(variant)
                except Exception as error:
                    failed = replace(trace, driver_factory=lambda: (_ for _ in ()).throw(error))
                    raw_trace, _ = _run_case(failed, set())
                    failure_facts = tuple(_unavailable(name, "bundle factory failed") for name in _HOOK_NAMES)
                    executions.append(FidelityVariantExecution(case.configuration_id, case.case_id, variant, case.profile_denominator, case.candidate_denominator, raw_trace, failure_facts))
                else:
                    hook_bindings = _validate_bundle(case, variant, bundle, record, seen_adapters, seen_producers, hook_state)
                    adapter, producer = bundle.adapter, bundle.producer
                    retained.append(bundle)
                    facts = []  # type: List[FidelityHookFact]
                    callback_errors = []
                    def capture(driver, names, bindings):
                        if driver is not adapter:
                            callback_errors.append(ConformanceValidationError("callback driver does not reconcile with retained adapter"))
                            return
                        for name, hook, target, code in zip(names, *bindings):
                            try:
                                _validate_hook(bundle, adapter, producer, record, name, hook, target, code)
                            except ConformanceValidationError as error:
                                callback_errors.append(error)
                                return
                            try:
                                facts.append(_capture_hook(name, hook))
                            finally:
                                try:
                                    _validate_hook(bundle, adapter, producer, record, name, hook, target, code)
                                except ConformanceValidationError as error:
                                    callback_errors.append(error)
                                    return
                    def before_observation(driver, call):
                        capture(driver, _HOOK_NAMES[:2], tuple(item[:2] for item in hook_bindings))
                    def after_observation(driver, call):
                        capture(driver, _HOOK_NAMES[2:], tuple(item[2:] for item in hook_bindings))
                    raw_trace, _ = _run_case(replace(trace, driver_factory=lambda adapter=adapter: adapter), {id(item.adapter) for item in retained[:-1]}, before_observation, after_observation)
                    if callback_errors:
                        raise callback_errors[0]
                    while len(facts) < len(_HOOK_NAMES):
                        facts.append(_unavailable(_HOOK_NAMES[len(facts)], "call was not reached"))
                    _validate_bundle(case, variant, bundle, record, seen_adapters, seen_producers, hook_state, hook_bindings, adapter, producer, recheck=True)
                    executions.append(FidelityVariantExecution(case.configuration_id, case.case_id, variant, case.profile_denominator, case.candidate_denominator, raw_trace, tuple(facts)))
            finally:
                _validate_factory(case, record, factory_state)
    return FidelityExecutionArtifact(authority, tuple(dict.fromkeys(case.configuration_id for case in cases)), tuple(case.case_id for case in cases), tuple(executions))


def _authority_records(authority):
    validate_fidelity_authority(authority)
    import json
    return json.loads(authority.canonical_json)["payload"]["cases"]


def _validate_factory(case, record, state):
    factory = case.bundle_factory
    target = _callable_target(factory, "bundle factory")
    prior = next((item for item in state if item[0] is target), None)
    digest = _python_marshal_code_sha256(target)
    if (record["bundle_factory_source"] != _source_data(_callable_source(factory, "bundle factory"))
            or (prior is None and record["python_marshal_code_sha256"] != digest)
            or (prior is not None and prior[1] is not target.__code__)):
        raise ConformanceValidationError("bundle factory does not reconcile with captured authority")
    if prior is None:
        state.append((target, target.__code__))


def _validate_bundle(case, variant, bundle, record, adapters, producers, hook_state, retained=None, adapter=None, producer_object=None, recheck=False):
    if not isinstance(bundle, DriverBundle) or bundle.variant_id != variant or bundle.producer_measurement.producer is not bundle.producer:
        raise ConformanceValidationError("bundle factory returned wrong variant or producer")
    if not recheck and (any(bundle.adapter is item for item in adapters) or any(bundle.producer is item for item in producers)):
        raise ConformanceValidationError("paired execution reused adapter or producer")
    if not recheck:
        adapters.append(bundle.adapter); producers.append(bundle.producer)
    hooks = (bundle.advance, bundle.native_snapshot, bundle.materialization_trace, bundle.interaction_counts)
    targets = tuple(_callable_target(hook, "bundle hook") for hook in hooks)
    codes = tuple(target.__code__ for target in targets)
    digests = tuple(_python_marshal_code_sha256(target) for target in targets)
    measurement = bundle.producer_measurement
    prior_codes = [next((item[1] for item in hook_state if item[0] is target), None) for target in targets]
    record_mismatch = (
        recheck and (retained is None or bundle.adapter is not adapter or bundle.producer is not producer_object or any(item is not saved for item, saved in zip(hooks, retained[0])) or any(item is not saved for item, saved in zip(targets, retained[1])) or any(item is not saved for item, saved in zip(codes, retained[2])))
        or record["configuration_id"] != case.configuration_id
        or record["profile_denominator"] != list(case.profile_denominator)
        or record["candidate_denominator"] != list(case.candidate_denominator)
        or record["adapter_source"] != _source_data(_source_for_class(type(bundle.adapter)))
        or record["producer"] != {"name": measurement.producer_name, "configuration_id": measurement.configuration_id, "type_source": _source_data(_source_for_class(type(bundle.producer))), "version": measurement.version, "content_sha256": measurement.content_sha256}
        or [item["source"] for item in record["hooks"]] != [_source_data(_callable_source(hook, "bundle hook")) for hook in hooks]
        or any(prior is None and item["python_marshal_code_sha256"] != digest or prior is not None and prior is not target.__code__ for item, target, prior, digest in zip(record["hooks"], targets, prior_codes, digests))
    )
    if record_mismatch:
        raise ConformanceValidationError("bundle does not reconcile with captured authority")
    for target, prior in zip(targets, prior_codes):
        if prior is None:
            hook_state.append((target, target.__code__))
    return hooks, targets, codes


def _validate_hook(bundle, adapter, producer, record, name, hook, target, code):
    """Reconcile one retained callable immediately around its invocation."""

    current = getattr(bundle, name)
    authority = record["hooks"][_HOOK_NAMES.index(name)]
    current_target = _callable_target(current, "bundle hook")
    if (bundle.adapter is not adapter or bundle.producer is not producer
            or current is not hook or current_target is not target or current_target.__code__ is not code
            or _source_data(_callable_source(current, "bundle hook")) != authority["source"]
            or _python_marshal_code_sha256(current_target) != authority["python_marshal_code_sha256"]):
        raise ConformanceValidationError("retained hook does not reconcile with captured authority")


def _unavailable(name, message):
    return FidelityHookFact(name, None, _manual(None, "fidelity_hook", "Unavailable", message, False))

def _capture_hook(name, hook):
    try:
        artifact = hook()
    except Exception as error:
        return FidelityHookFact(name, None, _exception(None, "fidelity_hook", error, False))
    if not isinstance(artifact, FrozenJsonArtifact):
        return FidelityHookFact(name, None, _manual(None, "fidelity_hook", "TypeError", "hook did not return FrozenJsonArtifact", False))
    return FidelityHookFact(name, artifact, None)
