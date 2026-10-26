"""Adversarial tests for raw conformance facts and pure assessment."""

from dataclasses import replace
import hashlib
import json
import marshal
from typing import cast

import numpy as np
import pytest

from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.conformance import (
    CertificationSubject, ConformanceValidationError,
    DriverBundle, EvidenceIdentity, ExceptionFact, ExecutionOutcome,
    FidelityCase, FrozenJsonArtifact, FrozenObservation, PredicateStatus,
    TraceCase, capture_fidelity_authority, derive_assessment,
    freeze_json_artifact, measure_adapter_provenance, measure_producer,
    run_core_conformance, validate_fidelity_authority,
)
from specola.conformance.core_runner import MetadataProbeFact
from specola.entities import EntityState, TrackState
from specola.extraction_observation import (
    ExtractionObservation, StructuredMetadata, StructuredSample, extraction_from_document,
)
from specola.native_values import NativeValue


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(value):
    return EvidenceIdentity(value, _digest(value))


class _Adapter(SimulatorAdapter):
    def __init__(self):
        super().__init__()
        self.index = 0

    @property
    def dt(self): return 0.1

    @property
    def spatial_dims(self): return 3

    def _reset_impl(self):
        self.index = 0
        return self._obs()

    def _get_observation_impl(self):
        self.index += 1
        return self._obs()

    def _obs(self):
        entity = EntityState(1, "object", self.index, TrackState.CONFIRMED, np.zeros(3), np.zeros(3), {"mass": 1.0})
        return MonitorObservation(self.index, self.index * .1, {1: entity})


class _TypedAdapter(_Adapter):
    def _obs(self):
        observation = super()._obs()
        envelope = ExtractionObservation(
            timestep=self.index, time=self.index * self.dt,
            episode_epoch=self.extraction_episode_epoch,
            configuration_epoch=self.query_configuration_epoch,
            producer_surface_sha256="a" * 64, manifest_sha256="b" * 64,
            queries=self.query_configuration,
            metadata=(StructuredMetadata(
                "fixture.state", "Fixture native state", "record", "unitless",
                "not_applicable", "identity", ("fixture.state",), "c" * 64,
            ),),
            samples=(StructuredSample(
                "fixture.state", "body", (3,),
                NativeValue.capture({"mass": 1.0, "rows": np.array([[self.index, np.nan]], dtype=">f8")}),
                "present", entity_id=1, scalar_keys=("mass",),
            ),),
            verification_only=False,
        )
        return replace(observation, extraction=envelope)


class _Producer:
    pass


class _UnmeasurableHook:
    def __call__(self):
        return None


def _harness_hook():
    return None


def _other_harness_hook():
    return None


def _fidelity_bundle(variant, configuration_id="configuration"):
    producer = _Producer()
    return DriverBundle(_Adapter(), producer, measure_producer(producer, "test-producer", configuration_id, version="1"), variant, _harness_hook, _harness_hook, _harness_hook, _harness_hook)


def _fidelity_case(factory, configuration_id="configuration", case_id="case", candidates=("candidate",)):
    return FidelityCase(configuration_id, case_id, ("variant",), ("profile",), candidates, factory)


class _LatePatch(_Adapter):
    def _reset_impl(self):
        observation = super()._reset_impl()
        self.get_observation = lambda: self._obs()
        return observation


class _Override(_Adapter):
    def reset(self): return self._reset_impl()
    def get_observation(self):
        self.index += 1
        return self._obs()


class _Bomb:
    def __repr__(self): raise RuntimeError("canonicalization bomb")


class _CaptureBomb(_Override):
    def _obs(self):
        observation = super()._obs()
        observation.perceived_entities[1].properties["mass"] = _Bomb()
        return observation


class _BadTags(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.component_tags = "not-a-dict"
        return observation


class _BadTagReference(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.component_tags = {1: {"absent_property": ("arm",)}}
        return observation


class _LookupBomb(_Adapter):
    def __getattribute__(self, name):
        if name == "get_observation":
            raise RuntimeError("lookup bomb")
        return super().__getattribute__(name)


class _CrossCallBad(_Override):
    def _obs(self):
        entity_class = "object" if self.index == 0 else "different_object"
        entity = EntityState(1, entity_class, self.index, TrackState.CONFIRMED,
                             np.zeros(3), np.zeros(3), {"mass": 1.0})
        return MonitorObservation(0, 0.0, {1: entity})


class _NumpyScalar(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.perceived_entities[1].properties["mass"] = np.float32(1.0)
        return observation


class _EmptyEntities(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.perceived_entities = {}
        return observation


class _OptionalVectors(_Adapter):
    def _obs(self):
        observation = super()._obs()
        entity = observation.perceived_entities[1]
        entity.position = None
        entity.velocity = None
        return observation


class _EmptyTagLabels(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.component_tags = {1: {"mass": ()}}
        return observation


class _ValidTags(_Adapter):
    def _obs(self):
        observation = super()._obs()
        observation.component_tags = {1: {"mass": ("arm",)}}
        return observation


class _MetadataFailure(_Adapter):
    def property_metadata(self, entity_id):
        raise ValueError("metadata root cause")


class _TwoEntityMetadataFailure(_MetadataFailure):
    def _obs(self):
        observation = super()._obs()
        observation.perceived_entities[2] = EntityState(
            2, "object", self.index, TrackState.CONFIRMED,
            np.zeros(3), np.zeros(3), {"mass": 1.0},
        )
        return observation


class _NoneMetadata(_Adapter):
    def property_metadata(self, entity_id):
        return None


class _FalseyMetadata(_Adapter):
    class _Mapping(dict):
        def __bool__(self):
            return False

    def property_metadata(self, entity_id):
        return self._Mapping({"mass": object()})


class _ForgedCallable:
    def __init__(self, driver):
        self.__func__ = SimulatorAdapter.reset
        self.__self__ = driver
        self._driver = driver

    def __call__(self):
        return self._driver._reset_impl()


class _ForgedReset(_Adapter):
    def __init__(self):
        super().__init__()
        self.reset = _ForgedCallable(self)


def _subject(cls):
    source = measure_adapter_provenance(cls()).ordered_mro_sources[0]
    return CertificationSubject("mock", cls.__module__, cls.__qualname__, EvidenceIdentity(source.path, source.sha256), "synthetic", "finite synthetic domain", _identity("supported"))


def _case(name, factory, probe=False):
    return TraceCase(name, _identity("trace-" + name), 2, factory, probe_observe_before_reset=probe)


def _assert_component_tag_boundary_rejection(adapter_class):
    artifact = run_core_conformance(
        _subject(adapter_class),
        (_case(adapter_class.__name__, adapter_class),),
    )
    facts = artifact.executions[0]
    assessment = derive_assessment(artifact)
    tags = [
        item
        for item in assessment.predicate_results
        if item.predicate_id == "TAG.RECONCILIATION.V1"
    ]

    assert facts.attempted == (facts.plan[0],)
    assert facts.successful == ()
    assert facts.observations == ()
    assert len(facts.exceptions) == 1
    assert facts.exceptions[0].expected is False
    assert assessment.outcome is ExecutionOutcome.VIOLATION
    assert len(tags) == len(facts.plan)
    assert all(item.status is PredicateStatus.UNEXECUTED for item in tags)


def test_raw_facts_have_exact_plan_prefixes_frozen_observations_and_post_mro():
    artifact = run_core_conformance(_subject(_Adapter), (_case("good", _Adapter),))
    facts = artifact.executions[0]
    assert not hasattr(facts, "outcome")
    assert facts.attempted == facts.plan == facts.successful
    assert tuple(item.call for item in facts.observations) == facts.successful
    assert facts.provenance_before.ordered_mro_sources == facts.provenance_after.ordered_mro_sources
    assessment = derive_assessment(artifact)
    assert assessment.outcome is ExecutionOutcome.COMPLETE
    assert all(item.status is not PredicateStatus.UNEXECUTED for item in assessment.predicate_results)
    encoded = json.loads(artifact.to_json())
    assert encoded["derived_assessment"]["outcome"] == "complete"
    assert {claim["status"] for claim in encoded["derived_assessment"]["claims"]} == {"unverified"}



def test_raw_runner_retains_structured_results_and_verification_identity():
    artifact = run_core_conformance(_subject(_TypedAdapter), (_case("typed", _TypedAdapter),))
    facts = artifact.executions[0]
    assert not facts.exceptions
    assert len(facts.observations) == 3
    for index, frozen in enumerate(facts.observations):
        payload = json.loads(frozen.canonical_json)
        assert "extraction" in payload, "raw conformance capture discarded native quantity evidence"
        envelope = extraction_from_document(payload["extraction"])
        assert envelope.timestep == index
        assert envelope.verification_only is False
        assert envelope.producer_surface_sha256 == "a" * 64
        assert envelope.manifest_sha256 == "b" * 64
        assert envelope.episode_epoch == 0
        assert envelope.configuration_epoch == 0
        assert envelope.samples[0].address == ("fixture.state", "body", (3,), "", "")
        value = envelope.samples[0].value.restore()
        assert value["mass"] == 1.0
        assert value["rows"].dtype == np.dtype(">f8")
        assert value["rows"].shape == (1, 2)
        assert value["rows"][0, 0] == index
        assert np.isnan(value["rows"][0, 1])
        assert replace(frozen) == frozen
    assert derive_assessment(artifact).outcome is ExecutionOutcome.COMPLETE


def test_raw_runner_rejects_candidate_envelopes_without_verification_scope():
    class UnscopedCandidate(_TypedAdapter):
        def _obs(self):
            observation = super()._obs()
            assert observation.extraction is not None
            return replace(observation, extraction=replace(observation.extraction, verification_only=True))

    artifact = run_core_conformance(_subject(UnscopedCandidate), (_case("unscoped", UnscopedCandidate),))
    facts = artifact.executions[0]
    assert facts.observations == ()
    assert len(facts.exceptions) == 1
    assert facts.exceptions[0].message == "structured observation has the wrong verification mode"


@pytest.mark.parametrize("mutation", ["schema", "missing_value", "dtype", "unknown_field"])
def test_frozen_evidence_rejects_malformed_structured_payload(mutation):
    artifact = run_core_conformance(_subject(_TypedAdapter), (_case("typed", _TypedAdapter),))
    frozen = artifact.executions[0].observations[0]
    payload = json.loads(frozen.canonical_json)
    assert "extraction" in payload, "raw conformance capture discarded native quantity evidence"
    envelope = payload["extraction"]
    if mutation == "schema":
        envelope["schema"] = "unknown"
    elif mutation == "missing_value":
        del envelope["samples"][0]["value"]
    elif mutation == "dtype":
        envelope["samples"][0]["value"] = {"type": "array", "dtype": "unrecognized"}
    else:
        envelope["silently_discarded"] = True
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with pytest.raises(ConformanceValidationError, match="structured"):
        replace(frozen, canonical_json=text, sha256=_digest(text))


def test_legacy_raw_capture_keeps_its_exact_payload_fields():
    artifact = run_core_conformance(_subject(_Adapter), (_case("legacy", _Adapter),))
    payload = json.loads(artifact.executions[0].observations[0].canonical_json)
    assert set(payload) == {"type", "timestep", "time", "perceived_entities", "component_tags"}


def test_expected_pre_reset_rejection_is_a_satisfied_lifecycle_fact_and_run_continues():
    artifact = run_core_conformance(_subject(_Adapter), (_case("probe", _Adapter, probe=True),))
    facts = artifact.executions[0]
    assert facts.attempted == facts.successful == facts.plan
    assert facts.exceptions[0].expected is True
    assert facts.exceptions[0].call == facts.plan[0]
    assessment = derive_assessment(artifact)
    pre = [item for item in assessment.predicate_results if item.predicate_id == "LIFECYCLE.PRE_RESET.V1"]
    assert pre[0].status is PredicateStatus.SATISFIED_ON_TRACE
    assert assessment.outcome is ExecutionOutcome.COMPLETE


def test_late_monkeypatch_is_bound_and_derived_as_violation():
    artifact = run_core_conformance(_subject(_LatePatch), (_case("patch", _LatePatch),))
    facts = artifact.executions[0]
    assert facts.bound_callables[1].owner_qualname.endswith("<lambda>")
    assert derive_assessment(artifact).outcome is ExecutionOutcome.VIOLATION


def test_capture_failure_is_raw_exception_and_not_successful_observation():
    artifact = run_core_conformance(_subject(_CaptureBomb), (_case("bomb", _CaptureBomb),))
    facts = artifact.executions[0]
    assert facts.exceptions[0].phase == "capture"
    assert not facts.successful and not facts.observations
    assessment = derive_assessment(artifact)
    assert assessment.outcome is ExecutionOutcome.VIOLATION
    assert any(item.status is PredicateStatus.UNEXECUTED for item in assessment.predicate_results)


def test_replace_cannot_store_or_forge_derived_complete_outcome():
    artifact = run_core_conformance(_subject(_Adapter), (_case("replace", _Adapter),))
    assert not hasattr(artifact.executions[0], "predicate_results")
    with pytest.raises(ConformanceValidationError, match="observations must match"):
        replace(artifact.executions[0], observations=())


def test_public_override_and_ordinary_object_do_not_assess_complete():
    override = run_core_conformance(_subject(_Override), (_case("override", _Override),))
    ordinary = run_core_conformance(_subject(_Adapter), (_case("ordinary", object),))
    assert derive_assessment(override).outcome is ExecutionOutcome.VIOLATION
    assert derive_assessment(ordinary).outcome is ExecutionOutcome.VIOLATION


def test_forged_func_and_self_attributes_do_not_pass_bound_method_identity():
    artifact = run_core_conformance(_subject(_ForgedReset), (_case("forged-method", _ForgedReset),))
    result = next(
        item for item in derive_assessment(artifact).predicate_results
        if item.predicate_id == "CALL.PUBLIC_CALLABLE.V1"
    )
    assert result.status is PredicateStatus.VIOLATED


def test_base_template_rejects_nonmapping_component_tags_before_freezing():
    _assert_component_tag_boundary_rejection(_BadTags)


def test_base_template_rejects_component_tags_for_missing_property():
    _assert_component_tag_boundary_rejection(_BadTagReference)


def test_base_template_rejects_empty_component_tag_labels():
    _assert_component_tag_boundary_rejection(_EmptyTagLabels)


def test_component_tag_reconciliation_has_valid_and_zero_tag_denominators():
    valid = run_core_conformance(
        _subject(_ValidTags),
        (_case("valid-tags", _ValidTags),),
    )
    valid_results = derive_assessment(valid).predicate_results
    valid_tags = [
        item
        for item in valid_results
        if item.predicate_id == "TAG.RECONCILIATION.V1"
    ]

    assert len(valid_tags) == len(valid.executions[0].observations)
    assert all(
        item.status is PredicateStatus.SATISFIED_ON_TRACE
        for item in valid_tags
    )
    assert all("|tag:" in item.witness for item in valid_tags)

    zero = run_core_conformance(
        _subject(_Adapter),
        (_case("zero-tags", _Adapter),),
    )
    zero_results = derive_assessment(zero).predicate_results
    zero_tags = [
        item
        for item in zero_results
        if item.predicate_id == "TAG.RECONCILIATION.V1"
    ]

    assert len(zero_tags) == len(zero.executions[0].observations)
    assert all(item.status is PredicateStatus.UNEXERCISED for item in zero_tags)
    assert all(item.witness.endswith("|none") for item in zero_tags)


def test_zero_entity_trace_has_exactly_one_zero_witness_per_relevant_predicate():
    artifact = run_core_conformance(_subject(_EmptyEntities), (_case("empty", _EmptyEntities),))
    results = derive_assessment(artifact).predicate_results
    for predicate_id in (
        "ENTITY.STRUCTURE.V1", "VECTOR.STRUCTURE.V1", "PROPERTY.SCALAR.V1",
        "TAG.RECONCILIATION.V1", "ENTITY.IDENTITY_CLASS_STABILITY.V1",
        "DISCOVERY.METADATA.V1",
    ):
        zero = [item for item in results if item.predicate_id == predicate_id and item.witness.endswith("|none")]
        assert len(zero) == len(artifact.executions[0].observations)


def test_absent_optional_vectors_satisfy_structure_predicate():
    artifact = run_core_conformance(
        _subject(_OptionalVectors),
        (_case("optional-vectors", _OptionalVectors),),
    )
    vector_results = [
        item
        for item in derive_assessment(artifact).predicate_results
        if item.predicate_id == "VECTOR.STRUCTURE.V1"
    ]

    assert len(vector_results) == 2 * len(artifact.executions[0].observations)
    assert all(
        item.status is PredicateStatus.SATISFIED_ON_TRACE
        for item in vector_results
    )
    assert derive_assessment(artifact).outcome is ExecutionOutcome.COMPLETE


@pytest.mark.parametrize(
    "malformed",
    (
        None,
        {"type": "NoneType"},
        {"value": "None"},
        {"type": "NoneType", "value": "None", "extra": True},
        {"type": "str", "value": "None"},
        {"type": "noneType", "value": "None"},
        {"type": "NoneType", "value": None},
        {"type": "NoneType", "value": True},
        {"type": "NoneType", "value": 1},
        {"type": "NoneType", "value": "not-none"},
    ),
)
def test_malformed_canonical_none_vector_is_rejected(malformed):
    artifact = run_core_conformance(
        _subject(_OptionalVectors),
        (_case("malformed-optional-vector", _OptionalVectors),),
    )
    original = artifact.executions[0].observations[0]
    parsed = json.loads(original.canonical_json)
    parsed["perceived_entities"]["items"][0][1]["position"] = malformed
    altered_text = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    altered = FrozenObservation(original.call, altered_text, _digest(altered_text))
    execution = replace(
        artifact.executions[0],
        observations=(altered,) + artifact.executions[0].observations[1:],
    )
    altered_artifact = replace(artifact, executions=(execution,))
    vector_results = [
        item
        for item in derive_assessment(altered_artifact).predicate_results
        if item.predicate_id == "VECTOR.STRUCTURE.V1"
        and item.call == original.call
        and item.witness.endswith(":position")
    ]

    assert len(vector_results) == 1
    assert vector_results[0].status is PredicateStatus.VIOLATED
    assert derive_assessment(altered_artifact).outcome is ExecutionOutcome.VIOLATION


def test_metadata_failure_keeps_adapter_entity_context_and_cause():
    artifact = run_core_conformance(
        _subject(_MetadataFailure), (_case("metadata-failure", _MetadataFailure),),
    )
    facts = artifact.executions[0]
    metadata_errors = [item for item in facts.exceptions if item.phase == "metadata"]
    assert metadata_errors
    assert all(item.entity_id == 1 for item in metadata_errors)
    assert all("_MetadataFailure entity 1" in item.message for item in metadata_errors)
    assert all("ValueError: metadata root cause" in item.traceback_text for item in metadata_errors)
    results = [
        item for item in derive_assessment(artifact).predicate_results
        if item.predicate_id == "DISCOVERY.METADATA.V1"
    ]
    assert results and all(item.status is PredicateStatus.VIOLATED for item in results)
    assert derive_assessment(artifact).outcome is ExecutionOutcome.VIOLATION


@pytest.mark.parametrize("adapter_class", (_Adapter, _NoneMetadata, _FalseyMetadata))
def test_metadata_empty_none_and_falsey_mappings_are_accepted(adapter_class):
    artifact = run_core_conformance(
        _subject(adapter_class), (_case("metadata-accepted", adapter_class),),
    )
    facts = artifact.executions[0]
    assert not [item for item in facts.exceptions if item.phase == "metadata"]
    results = [
        item for item in derive_assessment(artifact).predicate_results
        if item.predicate_id == "DISCOVERY.METADATA.V1"
    ]
    assert results and all(item.status is PredicateStatus.SATISFIED_ON_TRACE for item in results)
    assert derive_assessment(artifact).outcome is ExecutionOutcome.COMPLETE


def test_two_metadata_failures_on_one_call_are_independent_raw_violations():
    artifact = run_core_conformance(
        _subject(_TwoEntityMetadataFailure),
        (_case("metadata-two", _TwoEntityMetadataFailure),),
    )
    facts = artifact.executions[0]
    for call in facts.observations:
        errors = [
            item for item in facts.exceptions
            if item.phase == "metadata" and item.call == call.call
        ]
        assert {item.entity_id for item in errors} == {1, 2}
    results = [
        item for item in derive_assessment(artifact).predicate_results
        if item.predicate_id == "DISCOVERY.METADATA.V1"
    ]
    assert len(results) == len(facts.metadata_probes)
    assert all(item.status is PredicateStatus.VIOLATED for item in results)
    assert derive_assessment(artifact).outcome is ExecutionOutcome.VIOLATION
    json.loads(artifact.to_json())


def _metadata_probe_facts():
    artifact = run_core_conformance(_subject(_Adapter), (_case("probe-shape", _Adapter),))
    return artifact, artifact.executions[0]


@pytest.mark.parametrize("mutation", ("missing", "extra", "reverse", "absent"))
def test_metadata_probes_must_match_exact_canonical_denominator(mutation):
    artifact, facts = _metadata_probe_facts()
    probes = list(facts.metadata_probes)
    if mutation == "missing":
        probes = probes[1:]
    elif mutation == "extra":
        probes.append(MetadataProbeFact(facts.observations[0].call, 99))
    elif mutation == "reverse":
        probes.reverse()
    else:
        probes[0] = MetadataProbeFact(probes[0].call, 99)
    with pytest.raises(ConformanceValidationError, match="canonical observed-entity denominator"):
        replace(facts, metadata_probes=tuple(probes))


def test_duplicate_metadata_probe_is_rejected():
    _, facts = _metadata_probe_facts()
    with pytest.raises(ConformanceValidationError, match="unique per call/entity"):
        replace(facts, metadata_probes=facts.metadata_probes + (facts.metadata_probes[0],))


def test_forged_metadata_exception_must_identify_an_exact_probe():
    artifact, facts = _metadata_probe_facts()
    forged = ExceptionFact(
        facts.observations[0].call, "metadata", "RuntimeError", "forged",
        "trace", False, 99,
    )
    with pytest.raises(ConformanceValidationError, match="identify an observed probe"):
        replace(facts, exceptions=facts.exceptions + (forged,))


def test_getattr_failure_is_a_raw_violation_artifact_not_a_runner_escape():
    artifact = run_core_conformance(_subject(_LookupBomb), (_case("lookup", _LookupBomb),))
    facts = artifact.executions[0]
    assert facts.exceptions[0].phase == "lookup"
    assert facts.exceptions[0].call == facts.attempted[1]
    assert derive_assessment(artifact).outcome is ExecutionOutcome.VIOLATION


def test_cross_call_time_and_entity_class_predicates_are_independent_witnesses():
    artifact = run_core_conformance(_subject(_CrossCallBad), (_case("cross", _CrossCallBad),))
    results = derive_assessment(artifact).predicate_results
    assert any(item.predicate_id == "OBSERVATION.CROSS_CALL_TIME_INDEX.V1" and item.status is PredicateStatus.VIOLATED for item in results)
    assert any(item.predicate_id == "ENTITY.IDENTITY_CLASS_STABILITY.V1" and item.status is PredicateStatus.VIOLATED for item in results)


def test_numpy_scalars_are_canonicalized_as_python_scalars_before_assessment():
    artifact = run_core_conformance(_subject(_NumpyScalar), (_case("numpy", _NumpyScalar),))
    fact = artifact.executions[0].observations[0]
    assert "float32" not in fact.canonical_json
    assert derive_assessment(artifact).outcome is ExecutionOutcome.COMPLETE


def test_missing_canonical_track_state_violates_entity_structure():
    artifact = run_core_conformance(_subject(_Adapter), (_case("track-state", _Adapter),))
    facts = artifact.executions[0]
    original = facts.observations[0]
    payload = json.loads(original.canonical_json)
    del payload["perceived_entities"]["items"][0][1]["track_state"]
    altered_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    altered = FrozenObservation(original.call, altered_text, _digest(altered_text))
    altered_facts = replace(facts, observations=(altered,) + facts.observations[1:])
    altered_artifact = replace(artifact, executions=(altered_facts,))
    assert any(
        item.predicate_id == "ENTITY.STRUCTURE.V1" and item.status is PredicateStatus.VIOLATED
        for item in derive_assessment(altered_artifact).predicate_results
    )


def test_frozen_observation_rejects_missing_required_payload_field():
    artifact = run_core_conformance(_subject(_Adapter), (_case("payload", _Adapter),))
    call = artifact.executions[0].observations[0].call
    text = json.dumps({"type": "MonitorObservation"}, separators=(",", ":"))
    with pytest.raises(ConformanceValidationError, match="payload is incomplete"):
        FrozenObservation(call, text, _digest(text))


def test_case_execution_rejects_observation_and_unexpected_exception_for_same_call():
    artifact = run_core_conformance(_subject(_Adapter), (_case("conflict", _Adapter),))
    facts = artifact.executions[0]
    conflict = ExceptionFact(facts.observations[0].call, "test", "RuntimeError", "failure", "trace", False)
    with pytest.raises(ConformanceValidationError, match="observation and unexpected exception"):
        replace(facts, exceptions=(conflict,))


def test_case_denominator_disambiguates_identical_call_witnesses_across_cases():
    artifact = run_core_conformance(
        _subject(_Adapter),
        (_case("first", _Adapter), _case("second", _Adapter)),
    )
    assessment = derive_assessment(artifact)
    assert assessment.outcome is ExecutionOutcome.COMPLETE
    assert any(item.witness.startswith("case:first|") for item in assessment.predicate_results)
    assert any(item.witness.startswith("case:second|") for item in assessment.predicate_results)


def test_each_episode_uses_the_public_reset_call_before_its_observations():
    case = TraceCase("two-episodes", _identity("trace-two"), 1, _Adapter, episode_count=2)
    artifact = run_core_conformance(_subject(_Adapter), (case,))
    resets = [call for call in artifact.executions[0].plan if call.kind.value == "reset"]
    assert [call.episode_index for call in resets] == [0, 1]
    assert derive_assessment(artifact).outcome is ExecutionOutcome.COMPLETE


def test_invalid_case_identifier_and_step_denominators_are_rejected_early():
    with pytest.raises(ConformanceValidationError, match="case ID"):
        TraceCase("", _identity("trace-invalid"), 1, _Adapter)
    with pytest.raises(ConformanceValidationError, match="observations per episode"):
        TraceCase("invalid-steps", _identity("trace-invalid-steps"), 0, _Adapter)


def test_factory_unavailability_is_distinct_from_an_adapter_violation():
    def unavailable():
        raise OSError()

    artifact = run_core_conformance(_subject(_Adapter), (_case("unavailable", unavailable),))
    assert artifact.executions[0].exceptions[0].phase == "factory"
    assert artifact.executions[0].exceptions[0].message == "OSError"
    assert derive_assessment(artifact).outcome is ExecutionOutcome.SETUP_UNAVAILABLE


def test_replaced_source_declaration_cannot_match_measured_direct_source():
    subject = _subject(_Adapter)
    forged = replace(subject, expected_adapter_source=_identity("forged"))
    artifact = run_core_conformance(forged, (_case("source", _Adapter),))
    assert derive_assessment(artifact).outcome is ExecutionOutcome.VIOLATION


def test_producer_measurement_requires_version_or_content_digest():
    with pytest.raises(ConformanceValidationError, match="version or content digest"):
        measure_producer(_Producer(), "test-producer", "configuration")
    measured, substitute = _Producer(), _Producer()
    with pytest.raises(ConformanceValidationError, match="producer disagrees"):
        DriverBundle(_Adapter(), substitute, measure_producer(measured, "test-producer", "configuration", version="1"), "variant", _harness_hook, _harness_hook, _harness_hook, _harness_hook)


def test_authority_capture_rejects_producer_configuration_mismatch():
    bundle = _fidelity_bundle("variant", "other-configuration")
    with pytest.raises(ConformanceValidationError, match="configuration"):
        capture_fidelity_authority((_fidelity_case(lambda variant: bundle),))
    with pytest.raises(ConformanceValidationError, match="wrong variant"):
        capture_fidelity_authority((_fidelity_case(lambda variant: _fidelity_bundle("ignored")),))


def test_authority_capture_rejects_reused_adapter_or_producer_identity():
    bundle = _fidelity_bundle("variant")
    def reused_factory(variant):
        object.__setattr__(bundle, "variant_id", variant)
        return bundle
    cases = (FidelityCase("configuration", "case", ("variant-a", "variant-b"), ("profile",), ("candidate",), reused_factory),)
    with pytest.raises(ConformanceValidationError, match="reused adapter or producer"):
        capture_fidelity_authority(cases)
    bundle = _fidelity_bundle("variant")
    cases = (_fidelity_case(lambda variant: bundle), _fidelity_case(lambda variant: bundle, "configuration-two", "case-two"))
    with pytest.raises(ConformanceValidationError, match="reused adapter or producer"):
        capture_fidelity_authority(cases)


def test_driver_bundle_rejects_non_adapter_and_authority_rechecks_adapter_source():
    producer = _Producer()
    with pytest.raises(ConformanceValidationError, match="SimulatorAdapter"):
        DriverBundle(cast(SimulatorAdapter, object()), producer, measure_producer(producer, "test-producer", "configuration", version="1"), "variant", _harness_hook, _harness_hook, _harness_hook, _harness_hook)
    bundle = _fidelity_bundle("variant")
    object.__setattr__(bundle, "adapter_source", replace(bundle.adapter_source, sha256=_digest("forged-source")))
    with pytest.raises(ConformanceValidationError, match="constructed objects"):
        capture_fidelity_authority((_fidelity_case(lambda variant: bundle),))


def test_driver_bundle_rejects_unmeasurable_and_drifting_hook_sources():
    producer = _Producer()
    with pytest.raises(ConformanceValidationError, match="source-measurable"):
        DriverBundle(_Adapter(), producer, measure_producer(producer, "test-producer", "configuration", version="1"), "variant", _UnmeasurableHook(), _harness_hook, _harness_hook, _harness_hook)
    bundle = _fidelity_bundle("variant")
    object.__setattr__(bundle, "advance", _other_harness_hook)
    with pytest.raises(ConformanceValidationError, match="hook source drift"):
        capture_fidelity_authority((_fidelity_case(lambda variant: bundle),))
    with pytest.raises(ConformanceValidationError, match="source-measurable"):
        capture_fidelity_authority((_fidelity_case(_UnmeasurableHook()),))
    case = None
    def drifting_factory(variant):
        object.__setattr__(case, "bundle_factory", _fidelity_bundle)
        return _fidelity_bundle(variant)
    case = _fidelity_case(drifting_factory)
    with pytest.raises(ConformanceValidationError, match="declaration drift"):
        capture_fidelity_authority((case,))
    first_bundle = _fidelity_bundle("variant")
    def mutate_first_hook(variant):
        object.__setattr__(first_bundle, "advance", _other_harness_hook)
        return _fidelity_bundle(variant)
    with pytest.raises(ConformanceValidationError, match="retained bundle source drift"):
        capture_fidelity_authority((_fidelity_case(lambda variant: first_bundle), _fidelity_case(mutate_first_hook, case_id="case-two")))
    first_bundle = _fidelity_bundle("variant")
    hook_code = _harness_hook.__code__
    def mutate_hook_code(variant):
        _harness_hook.__code__ = _other_harness_hook.__code__
        return _fidelity_bundle(variant)
    try:
        with pytest.raises(ConformanceValidationError, match="retained bundle source drift"):
            capture_fidelity_authority((_fidelity_case(lambda variant: first_bundle), _fidelity_case(mutate_hook_code, case_id="case-two")))
    finally:
        _harness_hook.__code__ = hook_code
    first_bundle = _fidelity_bundle("variant")
    def first_factory(variant):
        return first_bundle
    def replacement_factory(variant):
        return first_bundle
    factory_code = first_factory.__code__
    def mutate_factory_code(variant):
        first_factory.__code__ = replacement_factory.__code__
        return _fidelity_bundle(variant)
    try:
        with pytest.raises(ConformanceValidationError, match="bundle factory source or code drift"):
            capture_fidelity_authority((_fidelity_case(first_factory), _fidelity_case(mutate_factory_code, case_id="case-two")))
    finally:
        first_factory.__code__ = factory_code
    hook_code = _harness_hook.__code__
    original_digest = hashlib.sha256(marshal.dumps(hook_code, 2)).hexdigest()
    try:
        _harness_hook.__code__ = _other_harness_hook.__code__
        artifact = capture_fidelity_authority((_fidelity_case(_fidelity_bundle),))
        record = json.loads(artifact.canonical_json)["payload"]["cases"][0]
        replacement_digest = hashlib.sha256(marshal.dumps(_other_harness_hook.__code__, 2)).hexdigest()
        assert record["python_marshal_code_sha256"] == hashlib.sha256(marshal.dumps(_fidelity_bundle.__code__, 2)).hexdigest()
        assert all(hook["python_marshal_code_sha256"] == replacement_digest for hook in record["hooks"])
        assert replacement_digest != original_digest
    finally:
        _harness_hook.__code__ = hook_code


def test_fidelity_case_rejects_missing_or_duplicate_denominators():
    with pytest.raises(ConformanceValidationError, match="candidate denominator"):
        _fidelity_case(_fidelity_bundle, candidates=())
    with pytest.raises(ConformanceValidationError, match="variant denominator"):
        FidelityCase("configuration", "case", ("variant", "variant"), ("profile",), ("candidate",), _fidelity_bundle)
    artifact = capture_fidelity_authority((_fidelity_case(_fidelity_bundle), _fidelity_case(_fidelity_bundle, case_id="case-two")))
    assert json.loads(artifact.canonical_json)["payload"]["configuration_denominator"] == ["configuration"]
    first, second = None, None
    def mutate_second(variant):
        object.__setattr__(second, "candidate_denominator", ("changed",))
        return _fidelity_bundle(variant)
    first, second = _fidelity_case(mutate_second), _fidelity_case(_fidelity_bundle, case_id="case-two")
    with pytest.raises(ConformanceValidationError, match="declaration drift"):
        capture_fidelity_authority((first, second))


def test_frozen_authority_artifact_rejects_forged_json_or_digest():
    artifact = capture_fidelity_authority((_fidelity_case(_fidelity_bundle),))
    payload = json.loads(artifact.canonical_json)
    payload["identity"] = "forged"
    forged_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ConformanceValidationError, match="digest mismatch"):
        FrozenJsonArtifact(artifact.identity, artifact.schema, forged_json, artifact.sha256)
    with pytest.raises(ConformanceValidationError, match="disagrees"):
        FrozenJsonArtifact(artifact.identity, artifact.schema, forged_json, _digest(forged_json))
    with pytest.raises(ConformanceValidationError, match="payload JSON invalid"):
        freeze_json_artifact("artifact", "schema", {"not-json": set()})
    signed = json.loads(artifact.canonical_json)
    del signed["payload"]["cases"][0]["hooks"]
    signed_json = json.dumps(signed, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ConformanceValidationError, match="case record has exact keys"):
        validate_fidelity_authority(FrozenJsonArtifact(artifact.identity, artifact.schema, signed_json, _digest(signed_json)))
    signed = json.loads(artifact.canonical_json)
    signed["payload"]["cases"][0]["hooks"][0]["python_marshal_code_sha256"] = "forged"
    signed_json = json.dumps(signed, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ConformanceValidationError, match="hook marshal code SHA-256"):
        validate_fidelity_authority(FrozenJsonArtifact(artifact.identity, artifact.schema, signed_json, _digest(signed_json)))
