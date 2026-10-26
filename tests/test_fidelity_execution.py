"""Adversarial raw paired-fidelity execution facts."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.conformance import (
    ConformanceValidationError, DriverBundle, EvidenceIdentity, FidelityCase,
    TraceCase, capture_fidelity_authority, freeze_json_artifact,
    measure_producer, run_fidelity_execution,
)
from specola.entities import EntityState, TrackState


_REUSE = _WRONG_VARIANT = _BAD_HOOK = _FAIL_HOOK = _THROW_FACTORY = _DRIFT_PRODUCER = False
_FAIL_RESET = _FAIL_GET = _SWAP_ADAPTER = _TRANSIENT_NATIVE = False
_RETAINED = _CURRENT = _ORIGINAL_SNAPSHOT = None
_EVENTS = []
_VARIANTS = ("unscoped", "scope:proper", "scope:empty", "criterion-poison")


class _Adapter(SimulatorAdapter):
    def __init__(self): super().__init__(); self.index = 0
    @property
    def dt(self): return 0.1
    @property
    def spatial_dims(self): return 3
    def _reset_impl(self):
        if _FAIL_RESET: raise RuntimeError("reset failure")
        self.index = 0; _EVENTS.append(("reset", id(self)))
        if _SWAP_ADAPTER: object.__setattr__(_CURRENT, "adapter", _Adapter())
        return self._observation()
    def _get_observation_impl(self):
        if _FAIL_GET: raise RuntimeError("get failure")
        self.index += 1; _EVENTS.append(("get_observation", id(self))); return self._observation()
    def _observation(self):
        return MonitorObservation(self.index, self.index * .1, {1: EntityState(1, "object", self.index, TrackState.CONFIRMED, np.zeros(3), np.zeros(3), {"mass": 1.0})})


class _Producer:
    pass


def _digest(value): return hashlib.sha256(value.encode("utf-8")).hexdigest()
def _hook(name):
    assert _CURRENT is not None
    _EVENTS.append((name, id(_CURRENT.adapter)))
    if _FAIL_HOOK: raise RuntimeError("hook failure")
    if _DRIFT_PRODUCER and name == "interaction_counts":
        object.__setattr__(_CURRENT.producer_measurement, "producer_name", "forged")
        object.__setattr__(_CURRENT.producer_measurement, "configuration_id", "forged-configuration")
        object.__setattr__(_CURRENT.producer_measurement, "version", "2")
        object.__setattr__(_CURRENT.producer_measurement, "content_sha256", _digest("forged"))
    return freeze_json_artifact("hook", "test-hook-v1", {"value": 1})
def _advance():
    global _ORIGINAL_SNAPSHOT
    assert _CURRENT is not None
    if _TRANSIENT_NATIVE:
        _ORIGINAL_SNAPSHOT = _snapshot.__code__
        _snapshot.__code__ = _snapshot_replacement.__code__
    return _hook("advance")
def _snapshot():
    return _hook("native_snapshot")
def _snapshot_replacement():
    result = _hook("native_snapshot")
    if _TRANSIENT_NATIVE:
        assert _ORIGINAL_SNAPSHOT is not None
        _snapshot.__code__ = _ORIGINAL_SNAPSHOT
    return result
def _materialization(): return _hook("materialization_trace")
def _interactions():
    return _hook("interaction_counts")
def _other_hook(): return freeze_json_artifact("other", "test-hook-v1", {"value": 1})
def _bundle(variant):
    global _RETAINED, _CURRENT
    if _REUSE and _RETAINED is not None:
        object.__setattr__(_RETAINED, "variant_id", variant)
        return _RETAINED
    producer = _Producer()
    hook = _other_hook if _BAD_HOOK else _advance
    bundle = DriverBundle(_Adapter(), producer, measure_producer(producer, "producer", "configuration", version="1"), "wrong" if _WRONG_VARIANT else variant, hook, _snapshot, _materialization, _interactions)
    _RETAINED = _CURRENT = bundle
    return bundle
def _factory(variant):
    return _bundle(variant)
def _factory_replacement(variant): return _bundle(variant)
def _retained_hook():
    return freeze_json_artifact("retained-hook-literal-identity", "test-hook-v1", {"value": 1})
def _retained_hook_factory(variant):
    producer = _Producer()
    return DriverBundle(_Adapter(), producer, measure_producer(producer, "producer", "configuration", version="1"), variant, _retained_hook, _other_hook, _other_hook, _other_hook)
def _retained_factory_literal(variant):
    producer = _Producer()
    return DriverBundle(_Adapter(), producer, measure_producer(producer, "retained-factory-literal-identity", "configuration", version="1"), variant, _other_hook, _other_hook, _other_hook, _other_hook)
def _throwing_factory(variant):
    if _THROW_FACTORY: raise RuntimeError("factory failure")
    return _bundle(variant)
def _factory_throw(variant): raise RuntimeError("factory failure")
def _last_factory_mutation(variant):
    if _THROW_FACTORY and variant == _VARIANTS[-1]:
        _last_factory_mutation.__code__ = _factory_throw.__code__
        raise RuntimeError("factory failure")
    return _bundle(variant)
def _case(): return FidelityCase("configuration", "case", _VARIANTS, ("profile",), ("candidate",), _factory)
def _trace(): return TraceCase("case", EvidenceIdentity("trace", _digest("trace")), 1, _Adapter)
def _authority(case=None):
    global _REUSE, _WRONG_VARIANT, _BAD_HOOK, _FAIL_HOOK, _THROW_FACTORY, _DRIFT_PRODUCER, _FAIL_RESET, _FAIL_GET, _SWAP_ADAPTER, _TRANSIENT_NATIVE, _RETAINED, _CURRENT, _ORIGINAL_SNAPSHOT, _EVENTS
    _REUSE = _WRONG_VARIANT = _BAD_HOOK = _FAIL_HOOK = _THROW_FACTORY = _DRIFT_PRODUCER = _FAIL_RESET = _FAIL_GET = _SWAP_ADAPTER = _TRANSIENT_NATIVE = False; _RETAINED = _CURRENT = _ORIGINAL_SNAPSHOT = None; _EVENTS = []
    return capture_fidelity_authority((case or _case(),))


def test_paired_execution_has_exact_required_variants_calls_and_hook_facts():
    authority = _authority()
    result = run_fidelity_execution(authority, (_case(),), (_trace(),))
    assert tuple(item.variant_id for item in result.executions) == _VARIANTS
    assert all(len(item.trace.plan) == 2 and all(fact.artifact is not None for fact in item.hooks) for item in result.executions)
    assert [name for name, _ in _EVENTS] == list(("reset", "advance", "native_snapshot", "get_observation", "materialization_trace", "interaction_counts") * len(_VARIANTS))
    assert all(len({identity for _, identity in _EVENTS[index:index + 6]}) == 1 for index in range(0, len(_EVENTS), 6))


def test_paired_execution_digest_ignores_retained_hook_literal_reference():
    case = FidelityCase("configuration", "case", _VARIANTS, ("profile",), ("candidate",), _retained_hook_factory)
    authority = _authority(case)
    result = run_fidelity_execution(authority, (case,), (_trace(),))
    assert all(item.hooks[0].artifact is not None for item in result.executions)


def test_paired_execution_digest_ignores_retained_factory_literal_reference():
    case = FidelityCase("configuration", "case", _VARIANTS, ("profile",), ("candidate",), _retained_factory_literal)
    authority = _authority(case)
    result = run_fidelity_execution(authority, (case,), (_trace(),))
    assert len(result.executions) == len(_VARIANTS)


def test_paired_execution_rejects_reuse_wrong_variant_and_hook_authority_mismatch():
    global _REUSE, _WRONG_VARIANT, _BAD_HOOK, _RETAINED
    authority = _authority(); _REUSE = True
    with pytest.raises(ConformanceValidationError, match="reused"):
        run_fidelity_execution(authority, (_case(),), (_trace(),))
    _authority(); _WRONG_VARIANT = True
    with pytest.raises(ConformanceValidationError, match="wrong variant"):
        run_fidelity_execution(authority, (_case(),), (_trace(),))
    _authority(); _BAD_HOOK = True
    with pytest.raises(ConformanceValidationError, match="reconcile"):
        run_fidelity_execution(authority, (_case(),), (_trace(),))
    _REUSE = _WRONG_VARIANT = _BAD_HOOK = False; _RETAINED = None


def test_paired_execution_validators_reject_missing_extra_and_malformed_raw_facts():
    result = run_fidelity_execution(_authority(), (_case(),), (_trace(),))
    execution = result.executions[0]
    with pytest.raises(ConformanceValidationError, match="exact ordered"):
        replace(execution, hooks=execution.hooks[:-1])
    with pytest.raises(ConformanceValidationError, match="exact ordered"):
        replace(execution, hooks=execution.hooks + (execution.hooks[-1],))
    with pytest.raises(ConformanceValidationError, match="must be tuples"):
        replace(execution, candidate_denominator=["candidate"])
    with pytest.raises(ConformanceValidationError, match="plan prefix"):
        replace(execution, trace=replace(execution.trace, plan=execution.trace.plan[:1]))
    with pytest.raises(ConformanceValidationError, match="one reset"):
        run_fidelity_execution(_authority(), (_case(),), (TraceCase("case", EvidenceIdentity("trace", _digest("trace")), 2, _Adapter),))
    with pytest.raises(ConformanceValidationError, match="one reset"):
        run_fidelity_execution(_authority(), (_case(),), (TraceCase("case", EvidenceIdentity("trace", _digest("trace")), 1, _Adapter, probe_observe_before_reset=True),))


def test_paired_execution_captures_hook_failure_without_status():
    global _FAIL_HOOK
    authority = _authority(); _FAIL_HOOK = True
    result = run_fidelity_execution(authority, (_case(),), (_trace(),))
    assert not result.executions[0].trace.exceptions
    assert all(fact.exception is not None for fact in result.executions[0].hooks)
    _FAIL_HOOK = False


def test_paired_execution_keeps_four_raw_hook_facts_after_trace_failures():
    global _FAIL_RESET, _FAIL_GET
    authority = _authority(); _FAIL_RESET = True
    result = run_fidelity_execution(authority, (_case(),), (_trace(),))
    assert result.executions[0].trace.exceptions and all(item.exception is not None for item in result.executions[0].hooks)
    _authority(); _FAIL_GET = True
    result = run_fidelity_execution(authority, (_case(),), (_trace(),))
    assert result.executions[0].trace.exceptions and tuple(item.name for item in result.executions[0].hooks) == ("advance", "native_snapshot", "materialization_trace", "interaction_counts")
    assert all(item.artifact is not None for item in result.executions[0].hooks[:2])
    assert all(item.exception is not None for item in result.executions[0].hooks[2:])
    _FAIL_RESET = _FAIL_GET = False

def test_paired_execution_rechecks_live_producer_hook_and_factory_drift():
    global _DRIFT_PRODUCER, _THROW_FACTORY, _SWAP_ADAPTER, _TRANSIENT_NATIVE
    authority = _authority(); _DRIFT_PRODUCER = True
    with pytest.raises(ConformanceValidationError, match="reconcile"):
        run_fidelity_execution(authority, (_case(),), (_trace(),))
    _DRIFT_PRODUCER = False
    authority = _authority(); original = _snapshot.__code__; _TRANSIENT_NATIVE = True
    try:
        with pytest.raises(ConformanceValidationError, match="retained hook"):
            run_fidelity_execution(authority, (_case(),), (_trace(),))
        assert "native_snapshot" not in [item[0] for item in _EVENTS]
    finally:
        _TRANSIENT_NATIVE = False; _snapshot.__code__ = original
    authority = _authority(); _SWAP_ADAPTER = True
    with pytest.raises(ConformanceValidationError, match="retained hook"):
        run_fidelity_execution(authority, (_case(),), (_trace(),))
    assert not {"advance", "native_snapshot", "materialization_trace", "interaction_counts"} & {item[0] for item in _EVENTS}
    _SWAP_ADAPTER = False
    case = FidelityCase("configuration", "case", _VARIANTS, ("profile",), ("candidate",), _throwing_factory)
    authority = capture_fidelity_authority((case,)); _THROW_FACTORY = True
    result = run_fidelity_execution(authority, (case,), (_trace(),))
    assert all(item.trace.exceptions and all(fact.exception is not None for fact in item.hooks) for item in result.executions)
    _THROW_FACTORY = False
    case = FidelityCase("configuration", "case", _VARIANTS, ("profile",), ("candidate",), _last_factory_mutation)
    authority = capture_fidelity_authority((case,)); original = _last_factory_mutation.__code__; _THROW_FACTORY = True
    try:
        with pytest.raises(ConformanceValidationError, match="factory"):
            run_fidelity_execution(authority, (case,), (_trace(),))
    finally:
        _THROW_FACTORY = False; _last_factory_mutation.__code__ = original
    authority = _authority(); original = _advance.__code__
    try:
        _advance.__code__ = _snapshot.__code__
        with pytest.raises(ConformanceValidationError, match="reconcile"):
            run_fidelity_execution(authority, (_case(),), (_trace(),))
    finally:
        _advance.__code__ = original
    authority = _authority(); original = _factory.__code__
    try:
        _factory.__code__ = _factory_replacement.__code__
        with pytest.raises(ConformanceValidationError, match="factory"):
            run_fidelity_execution(authority, (_case(),), (_trace(),))
    finally:
        _factory.__code__ = original
