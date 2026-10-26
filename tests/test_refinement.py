"""Adversarial tests for pure fidelity refinement-fact validation."""

from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace
import hashlib
import json
import sys

import numpy as np
import pytest

import specola.conformance.ledger as ledger_module
from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.conformance import (
    AdapterCertification,
    ApproximateWitness,
    AssuranceLabel,
    BindabilityArtifact,
    BindabilityCase,
    BindabilityResult,
    CandidateDeclaration,
    CandidateDisposition,
    CandidateRecord,
    CaseRefinementFacts,
    CertificationSubject,
    ConformanceValidationError,
    CoreConformanceArtifact,
    CriterionInteractionFact,
    Denominator,
    DerivedObligationFact,
    DerivedEvidenceReference,
    DiscoveryMaterializationFacts,
    DriverBundle,
    EvidenceIdentity,
    ExceptionFact,
    ExactWitness,
    ExactnessKind,
    ExpectedResetRejection,
    FieldAddress,
    FidelityCase,
    FidelityInventory,
    InventoryValidationError,
    NativeKindMode,
    ObligationResult,
    ProjectionEntry,
    ProducerProvenance,
    PredicateStatus,
    RefinementEvidence,
    RejectionKey,
    RuntimeDenominator,
    SemanticFieldRecord,
    SemanticManifest,
    SPECOLA_V1_CATALOGUE,
    STRUCTURAL_PREDICATE_SCHEMA,
    validate_bindability_artifact,
    SurfaceCallFact,
    SupportedDomainFacts,
    TraceCase,
    VariantRefinementFacts,
    VariantRole,
    WitnessKey,
    WitnessProvenance,
    assemble_core_publication_rows,
    capture_fidelity_authority,
    derive_core_obligation_facts,
    derive_assessment,
    freeze_json_artifact,
    measure_producer,
    run_core_conformance,
    run_fidelity_execution,
    validate_refinement_evidence,
    validate_specola_certification,
)
from specola.conformance.ledger import _content_reference, _field_coordinate, _plain_fact, _predicate_references
from specola.entities import EntityState, TrackState
from specola.extraction_observation import (
    ExtractionObservation, StructuredMetadata, StructuredSample, extraction_document,
)
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration


_VARIANTS = ("full", "small", "empty", "poison")
_CURRENT_VARIANT = "full"
_AUTHORITY_RECORD = None


@dataclass
class _MutableCandidate:
    candidate: str
    provenance: object
    approximation_relation: object


class _Adapter(SimulatorAdapter):
    def __init__(self, variant="full"):
        super().__init__()
        self.index = 0
        self.variant = variant

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    def _reset_impl(self):
        self.index = 0
        return self._observation()

    def _get_observation_impl(self):
        self.index += 1
        return self._observation()

    def _observation(self):
        properties = {"a": 1.0, "b": 2.0}
        second_properties = {"a": 1.0}
        if self.variant in ("small", "poison"):
            properties = {"b": 2.0}
            second_properties = {}
        elif self.variant == "empty":
            properties = {}
            second_properties = {}
        entity = EntityState(
            1, "object", self.index, TrackState.CONFIRMED,
            np.zeros(3), np.zeros(3), properties,
        )
        second = EntityState(
            2, "object", self.index, TrackState.CONFIRMED,
            np.ones(3), np.zeros(3), second_properties,
        )
        tags = {
            1: {key: ("tag-" + key,) for key in properties},
            2: {key: ("tag-" + key,) for key in second_properties},
        }
        return MonitorObservation(self.index, self.index * 0.1, {1: entity, 2: second}, tags)


class _Producer:
    pass


class _UnsupportedAdapter(_Adapter):
    def _reset_impl(self):
        raise TypeError("unsupported native kind: soft_body")


def _rejection_factory_failure():
    raise RuntimeError("factory failed before public reset")


def _advance():
    return freeze_json_artifact("advance", "test-hook-v1", {"advanced": True})


def _native_snapshot():
    return freeze_json_artifact("native", "test-hook-v1", {"state": [1, 2]})


def _materialization():
    return _semantic_hook("materialization")


def _material_payload():
    assert _AUTHORITY_RECORD is not None
    shared = _shared(_AUTHORITY_RECORD)
    keys = shared[4]
    scope = (None, (keys[1].field,), (), (keys[1].field,))[_VARIANTS.index(_CURRENT_VARIANT)]
    output = _outputs(keys, scope)
    accesses = keys if scope is None else tuple(item for item in keys if item.field in scope)
    payload = {"expected_reads": _jsonable(accesses), "expected_transfers": _jsonable(accesses), "output": _jsonable(output)}
    if _CURRENT_VARIANT == "full":
        case = _shared_case(shared)
        names = ("configuration_id", "case_id", "producer", "domain", "candidates", "denominator", "witnesses", "surface")
        payload["case"] = {name: _jsonable(getattr(case, name)) for name in names}
    return payload


def _interactions():
    return _semantic_hook("interactions")


def _semantic_hook(name):
    if name == "materialization":
        return freeze_json_artifact(name, "specola-refinement-materialization-v1", _material_payload())
    return freeze_json_artifact(name, "specola-refinement-interactions-v1", _interaction_payload())


def _interaction_payload():
    assert _AUTHORITY_RECORD is not None
    keys = _shared(_AUTHORITY_RECORD)[4]
    scope = (None, (keys[1].field,), (), (keys[1].field,))[_VARIANTS.index(_CURRENT_VARIANT)]
    accesses = keys if scope is None else tuple(item for item in keys if item.field in scope)
    payload = {"reads": _jsonable(accesses), "reward": 0, "success": 0, "transfers": _jsonable(accesses), "verdict": 0}
    return payload


def _bundle(variant):
    global _CURRENT_VARIANT
    _CURRENT_VARIANT = variant
    producer = _Producer()
    measured = measure_producer(producer, "producer", "configuration", version="1")
    return DriverBundle(
        _Adapter(variant), producer, measured, variant,
        _advance, _native_snapshot, _materialization, _interactions,
    )


def _factory(variant):
    return _bundle(variant)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _execution():
    global _AUTHORITY_RECORD
    case = FidelityCase(
        "configuration", "case", _VARIANTS, ("core",), ("a", "b"), _factory,
    )
    trace = TraceCase("case", EvidenceIdentity("trace", _digest("trace")), 1, _Adapter)
    authority = capture_fidelity_authority((case,))
    _AUTHORITY_RECORD = json.loads(authority.canonical_json)["payload"]["cases"][0]
    for _ in range(20):
        _materialization(); _interactions()
    authority = capture_fidelity_authority((case,))
    _AUTHORITY_RECORD = json.loads(authority.canonical_json)["payload"]["cases"][0]
    return run_fidelity_execution(authority, (case,), (trace,))

def _capture_rejection(key, factory):
    identity = "refinement-reset-rejection:{}:{}:{}".format(
        key.configuration_id, key.case_id, key.native_kind,
    )
    declaration = EvidenceIdentity(identity, _digest(identity))
    subject = CertificationSubject(
        "test.entry", __name__, _UnsupportedAdapter.__qualname__, declaration,
        "test-producer", key.configuration_id, declaration,
    )
    trace = TraceCase(key.case_id, declaration, 1, factory)
    return run_core_conformance(subject, (trace,)).executions[0]


def _rejection():
    key = RejectionKey("configuration", "unsupported-soft-body", "soft_body")
    execution = _capture_rejection(key, _UnsupportedAdapter)
    return ExpectedResetRejection(
        key, "TypeError", "unsupported native kind", execution,
    )


def _shared(record):
    producer = ProducerProvenance(
        "producer", "configuration", "1", None,
        record["producer"]["type_source"]["sha256"],
    )
    a = FieldAddress("a", "entity:1/native:shared", "entity:1")
    b = FieldAddress("b", "entity:1/native:shared")
    a2 = FieldAddress("a", "entity:2/native:shared", "entity:2")
    keys = tuple(
        WitnessKey(field, call)
        for call in ("reset", "get_observation") for field in (a, b, a2)
    )
    denominator = RuntimeDenominator(("a", "b"), (a, b, a2), ("reset", "get_observation"), keys)
    pa = WitnessProvenance("native.a[address]", producer)
    pb = WitnessProvenance("native.b[address]", producer)
    declarations = (
        _candidate("a", pa, None, "object entity population"),
        _candidate("b", pb, "absolute-error", "entity:1"),
    )
    witnesses = (
        ExactWitness(keys[0], '{"type":"float","value":"1.0"}', '{"type":"float","value":"1.0"}', pa),
        ApproximateWitness(keys[1], '{"type":"float","value":"2.0"}', "absolute-error", 0.1, 0.2, pb),
        ExactWitness(keys[2], '{"type":"float","value":"1.0"}', '{"type":"float","value":"1.0"}', pa),
        ExactWitness(keys[3], '{"type":"float","value":"1.0"}', '{"type":"float","value":"1.0"}', pa),
        ApproximateWitness(keys[4], '{"type":"float","value":"2.0"}', "absolute-error", 0.1, 0.2, pb),
        ExactWitness(keys[5], '{"type":"float","value":"1.0"}', '{"type":"float","value":"1.0"}', pa),
    )
    surface_calls = (
        SurfaceCallFact("reset", (a, b, a2), (a, b, a2), (a, b, a2)),
        SurfaceCallFact("get_observation", (a, b, a2), (a, b, a2), (a, b, a2)),
    )
    surface = DiscoveryMaterializationFacts(
        (a, a2), (b,), (a, b, a2), (a, b, a2), surface_calls,
    )
    source = record["adapter_source"]
    domain = SupportedDomainFacts(
        EvidenceIdentity(source["path"], source["sha256"]),
        NativeKindMode.EXTERNALLY_SUPPLIED,
        ("configuration",), ("rigid_body", "soft_body"),
        ("rigid_body",), "rigid_body",
    )
    return producer, domain, declarations, denominator, keys, witnesses, surface, _rejection()


def _candidate(name, provenance, relation, owner):
    return CandidateDeclaration(
        name, provenance, relation, owner, "declared addresses",
        "accepted lifecycle", "identity", name, "closed at reset" if name == "a" else "runtime schema",
        "test", "m", ("test",), "m", "test-family", name, "world",
    )


def _outputs(keys, scope=None):
    selected = keys if scope is None else tuple(item for item in keys if item.field in scope)
    fields = tuple(dict.fromkeys(item.field for item in selected))
    return tuple(
        ProjectionEntry(
            key,
            "/perceived_entities/items/{}/1/properties/items/{}/1".format(
                1 if "entity:2" in key.field.address else 0,
                tuple(item for item in fields if ("entity:2" in item.address) == ("entity:2" in key.field.address)).index(key.field),
            ),
            key.field.candidate,
            '{"type":"float","value":"1.0"}' if key.field.candidate == "a" else '{"type":"float","value":"2.0"}',
            key.field.runtime_owner or "entity:1",
            _metadata_json(key.field.candidate),
            "object:{}".format(2 if "entity:2" in key.field.address else 1),
        )
        for key in selected
    )


def _metadata_json(candidate):
    return json.dumps({
        "component": candidate, "family": "test-family", "frame": "world",
        "group": "test", "note": "m", "tags": ["test"], "unit": "m",
    }, sort_keys=True, separators=(",", ":"))


def _shared_case(shared):
    producer, domain, declarations, denominator, _, witnesses, surface, rejection = shared
    return CaseRefinementFacts(
        "configuration", "case", producer, domain, declarations, denominator, witnesses,
        surface, (), (rejection.key,), (rejection,),
    )


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return json.loads(json.dumps(value))


def _valid_evidence(execution):
    record = json.loads(execution.authority.canonical_json)["payload"]["cases"][0]
    shared = _shared(record)
    _, _, _, denominator, keys, _, _, _ = shared
    entries = _outputs(keys)
    b = denominator.address_denominator[1]
    roles = (
        VariantRole.UNSCOPED,
        VariantRole.PROPER_SCOPE,
        VariantRole.EMPTY_SCOPE,
        VariantRole.CRITERION_POISON,
    )
    requested = (None, (b,), (), (b,))
    variants = []
    for execution_fact, role, scope in zip(execution.executions, roles, requested):
        hook_digests = tuple(item.artifact.sha256 for item in execution_fact.hooks)
        selected = _outputs(keys, scope)
        accesses = keys if scope is None else tuple(item for item in keys if item.field in scope)
        variants.append(VariantRefinementFacts(
            execution_fact.variant_id, role, scope, hook_digests,
            hook_digests[1], selected, accesses, accesses, accesses, accesses,
            CriterionInteractionFact(0, 0, 0),
        ))
    case = replace(_shared_case(shared), variants=tuple(variants))
    return RefinementEvidence(execution.authority.sha256, (case,))


def test_valid_exact_and_bounded_refinement_facts_have_no_result_object():
    execution = _execution()
    evidence = _valid_evidence(execution)

    assert validate_refinement_evidence(execution, evidence) is None


def _derivation_inputs():
    execution = _execution()
    evidence = _valid_evidence(execution)
    case = evidence.cases[0]
    trace = execution.executions[0].trace
    source = trace.provenance_before
    assert source is not None
    direct = source.ordered_mro_sources[0]
    assert direct.path is not None and direct.sha256 is not None
    declaration = EvidenceIdentity(direct.path, direct.sha256)
    subject = CertificationSubject(
        "test.entry", source.direct_module, source.direct_qualname, declaration,
        case.producer.name, case.configuration_id, declaration,
    )
    core = CoreConformanceArtifact(
        subject, STRUCTURAL_PREDICATE_SCHEMA, (trace.case_id,),
        (trace.trace_source_declaration,), (trace,),
    )
    provenance = _producer_reference(case.producer)
    inventory = FidelityInventory(
        Denominator("test-surface", ("a", "b"), "test enumeration", provenance),
        tuple(CandidateRecord(name, CandidateDisposition.INCLUDED, "included", "test") for name in ("a", "b")),
    )
    fields = tuple(_semantic_field(item, provenance) for item in case.candidates)
    manifest = SemanticManifest("test-manifest", "1", ("a", "b"), fields, "test manifest")
    return core, inventory, manifest, execution, evidence


def _producer_reference(producer):
    return "{}:{}:{}:{}:{}".format(
        producer.name, producer.configuration_id, producer.version or "",
        producer.content_sha256 or "", producer.type_source_sha256 or "",
    )


def _semantic_field(candidate, provenance):
    relation = candidate.approximation_relation
    return SemanticFieldRecord(
        candidate.candidate, candidate.provenance.source_accessor, candidate.owner,
        "declared addresses", "accepted lifecycle", "identity", "m", "world",
        ExactnessKind.APPROXIMATE if relation else ExactnessKind.EXACT,
        relation, candidate.candidate, "test", ("test",),
        "closed at reset" if candidate.candidate == "a" else "runtime schema",
        provenance, note="m", family="test-family", component=candidate.candidate,
    )


def test_derives_exact_immutable_fourteen_core_fact_denominator_purely():
    inputs = _derivation_inputs()

    first = derive_core_obligation_facts(*inputs)
    second = derive_core_obligation_facts(*inputs)

    assert first == second
    assert tuple(item.obligation_id for item in first) == SPECOLA_V1_CATALOGUE.definition_denominator[:14]
    assert all(set(item.__dataclass_fields__) == {"obligation_id", "evidence_references"} for item in first)
    assert tuple(item.coordinates[0] for item in first[-1].evidence_references) == tuple(
        "obligation=" + item.obligation_id for item in first[:-1]
    )
    assert all(type(reference) is DerivedEvidenceReference for item in first for reference in item.evidence_references)
    with pytest.raises(FrozenInstanceError):
        setattr(first[0], "obligation_id", "changed")


def test_assembles_exact_twenty_six_publication_rows():
    inputs = _derivation_inputs()
    facts = derive_core_obligation_facts(*inputs)
    statuses = {item.predicate_id: item.status for item in derive_assessment(inputs[0]).predicate_results}
    assert statuses["LIFECYCLE.PRE_RESET.V1"] is PredicateStatus.UNEXERCISED
    rows = assemble_core_publication_rows(*inputs)
    assert tuple(item.obligation_id for item in rows) == SPECOLA_V1_CATALOGUE.definition_denominator
    assert all(item.result is ObligationResult.PASS and item.applicable for item in rows[:14])
    assert all(item.result is ObligationResult.UNEXECUTED and item.applicable for item in rows[14:])
    assert all(item.assurance_labels == (AssuranceLabel.UNVERIFIED,) for item in rows[14:])
    rationale = "applicable=True means absence is not proven; capability presence is not claimed"
    assert rows[14].obligation_id == "CORE.STATIC_SLOT_BINDABILITY"
    assert rows[14].rationale == "static slot bindability probe not executed"
    assert all(item.rationale == rationale for item in rows[15:])
    for row, fact in zip(rows, facts):
        expected = tuple(
            json.dumps(_plain_fact(reference), sort_keys=True, separators=(",", ":"))
            for reference in fact.evidence_references
        )
        assert row.evidence_provenance == expected
    validate_specola_certification(AdapterCertification(
        inputs[1], inputs[2], SPECOLA_V1_CATALOGUE, rows,
    ))


def _probe_artifact():
    """One artifact the real probe emitted on the mock adapter."""
    from specola.conformance import probe_for_registry_name

    artifact = probe_for_registry_name("mock")
    assert artifact.outcome == "pass"
    return artifact


def _consistent_artifact(outcome):
    """A hand-built artifact whose results support ``outcome``."""
    case = BindabilityCase("robot", "joint angle", "qpos_joint1", 0)
    other = BindabilityCase("robot", "pose", "xpos_base", 0)
    statuses = {
        "pass": ("pass", "pass"),
        "fail": ("pass", "fail"),
        "setup_unavailable": ("unexecuted", "unexecuted"),
        "not_applicable": (),
    }[outcome]
    cases = (case, other)[:len(statuses)]
    return BindabilityArtifact(
        "specola.adapters.fake._MockAdapter", "closed", ("robot",), (),
        cases,
        tuple(
            BindabilityResult(item, status, status)
            for item, status in zip(cases, statuses)
        ),
        outcome,
        "detail" if outcome != "pass" else "",
    )


def test_bindability_artifact_decides_the_static_slot_row():
    inputs = _derivation_inputs()
    artifact = _probe_artifact()

    rows = assemble_core_publication_rows(*inputs, artifact)
    assert rows[14].obligation_id == "CORE.STATIC_SLOT_BINDABILITY"
    assert rows[14].result is ObligationResult.PASS
    assert rows[14].assurance_labels == (
        AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )
    assert tuple(item.obligation_id for item in rows) == SPECOLA_V1_CATALOGUE.definition_denominator
    validate_specola_certification(AdapterCertification(
        inputs[1], inputs[2], SPECOLA_V1_CATALOGUE, rows,
    ))

    for outcome, result in (
        ("fail", ObligationResult.FAIL),
        ("not_applicable", ObligationResult.UNEXECUTED),
        ("setup_unavailable", ObligationResult.UNEXECUTED),
    ):
        rows = assemble_core_publication_rows(
            *inputs, _consistent_artifact(outcome)
        )
        assert len(rows) == 26
        assert rows[14].obligation_id == "CORE.STATIC_SLOT_BINDABILITY"
        assert rows[14].result is result
        assert rows[14].applicable
        validate_specola_certification(AdapterCertification(
            inputs[1], inputs[2], SPECOLA_V1_CATALOGUE, rows,
        ))


_CASE = BindabilityCase("robot", "joint angle", "qpos_joint1", 0)
_OTHER = BindabilityCase("robot", "pose", "xpos_base", 0)


def _artifact(cases, results, outcome, detail=""):
    return BindabilityArtifact(
        "specola.adapters.fake._MockAdapter", "closed", ("robot",), (),
        cases, results, outcome, detail,
    )


@pytest.mark.parametrize("artifact", (
    pytest.param(_artifact((), (), "pass"), id="empty-pass"),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "fail", "no monitor"),), "pass",
        ),
        id="pass-with-a-failed-result",
    ),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "pass", "bound"),), "fail",
        ),
        id="fail-with-all-passing-results",
    ),
    pytest.param(
        _artifact(
            (_CASE, _OTHER),
            (
                BindabilityResult(_OTHER, "pass", "bound"),
                BindabilityResult(_CASE, "pass", "bound"),
            ),
            "pass",
        ),
        id="mismatched-case-order",
    ),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "bound", "bound"),), "pass",
        ),
        id="unknown-result-status",
    ),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "pass", "bound"),), "passed",
        ),
        id="unknown-outcome",
    ),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "unexecuted", "no leg"),),
            "setup_unavailable",
        ),
        id="setup-unavailable-without-detail",
    ),
    pytest.param(
        _artifact(
            (_CASE,), (BindabilityResult(_CASE, "pass", "bound"),),
            "not_applicable",
        ),
        id="not-applicable-with-a-case",
    ),
))
def test_inconsistent_bindability_artifact_is_refused(artifact):
    """An outcome the probe cannot emit never decides an obligation."""
    inputs = _derivation_inputs()

    with pytest.raises(ConformanceValidationError):
        validate_bindability_artifact(artifact)
    with pytest.raises(ConformanceValidationError):
        derive_core_obligation_facts(*inputs, artifact)
    with pytest.raises(ConformanceValidationError):
        assemble_core_publication_rows(*inputs, artifact)


def test_probe_emitted_artifacts_pass_validation():
    for artifact in (
        _probe_artifact(),
        _consistent_artifact("fail"),
        _consistent_artifact("not_applicable"),
        _consistent_artifact("setup_unavailable"),
    ):
        validate_bindability_artifact(artifact)


def test_bindability_evidence_passes_the_exact_frozen_fact_gate():
    """A look-alike probe record is refused like any other raw input."""
    inputs = _derivation_inputs()

    @dataclass
    class _LooseArtifact:
        outcome: str = "pass"
        adapter_type: str = "impostor"
        cases: tuple = ()

    for impostor in (_LooseArtifact(), _LooseArtifact(cases=([],))):
        with pytest.raises(ConformanceValidationError, match="exact"):
            derive_core_obligation_facts(*inputs, impostor)
        with pytest.raises(ConformanceValidationError, match="exact"):
            assemble_core_publication_rows(*inputs, impostor)


def test_core_evidence_modes_satisfy_every_catalogue_rule():
    rows = assemble_core_publication_rows(*_derivation_inputs())
    for row, definition in zip(rows[:14], SPECOLA_V1_CATALOGUE.definitions[:14]):
        labels = set(row.assurance_labels)
        assert set(definition.pass_evidence_rule.all_of) <= labels
        assert not definition.pass_evidence_rule.any_of or labels & set(definition.pass_evidence_rule.any_of)


@pytest.mark.parametrize("attack", ("missing-obligation-id", "invented-obligation-id"))
def test_assembly_rejects_inexact_internal_denominator(monkeypatch, attack):
    inputs = _derivation_inputs()
    facts = derive_core_obligation_facts(*inputs)
    changed = facts[:-1] if attack.startswith("missing") else facts + (replace(facts[-1], obligation_id="invented"),)
    monkeypatch.setattr(ledger_module, "derive_core_obligation_facts", lambda *unused: changed)
    with pytest.raises(ConformanceValidationError, match="exact core obligation denominator"):
        assemble_core_publication_rows(*inputs)


@pytest.mark.parametrize(
    "attack,keywords", (
        ("caller-supplied-result", ("result",)),
        ("caller-supplied-assurance", ("assurance",)),
        ("standalone-no-silent-or-certification-boolean", ("no_silent_acceptance", "certified")),
    ),
)
def test_assembly_rejects_caller_outcome_fields(attack, keywords):
    for keyword in keywords:
        with pytest.raises(TypeError):
            assemble_core_publication_rows(*_derivation_inputs(), **{keyword: True})


def test_assembly_rejects_caller_applicability():
    with pytest.raises(TypeError):
        assemble_core_publication_rows(*_derivation_inputs(), **{"applicable": True})


@pytest.mark.parametrize(
    "status", (PredicateStatus.UNEXECUTED, PredicateStatus.UNEXERCISED),
)
def test_assembly_emits_no_rows_for_non_satisfied_structural_state(monkeypatch, status):
    inputs = _derivation_inputs()
    assessment = derive_assessment(inputs[0])
    target = next(
        item for item in assessment.predicate_results if item.predicate_id == "PROPERTY.SCALAR.V1"
    )
    changed = replace(assessment, predicate_results=tuple(
        replace(item, status=status) if item is target else item for item in assessment.predicate_results
    ))
    monkeypatch.setattr(ledger_module, "derive_assessment", lambda unused: changed)
    with pytest.raises(ConformanceValidationError, match="exactly satisfied"):
        assemble_core_publication_rows(*inputs)


def test_assembly_rejects_real_unmapped_source_violation():
    core, *inputs = _derivation_inputs()
    subject = replace(core.subject_declarations, expected_adapter_qualname="wrong")
    core = replace(core, subject_declarations=subject)
    assessment = derive_assessment(core)
    assert any(item.status is PredicateStatus.VIOLATED for item in assessment.predicate_results)
    assert len(derive_core_obligation_facts(core, *inputs)) == 14
    with pytest.raises(ConformanceValidationError, match="violated or unexecuted"):
        assemble_core_publication_rows(core, *inputs)


def test_shared_address_rows_keep_candidate_qualified_coordinates():
    rows = assemble_core_publication_rows(*_derivation_inputs())
    provenance = "".join(item for row in rows[:14] for item in row.evidence_provenance)
    assert json.dumps(_field_coordinate(FieldAddress("a", "entity:1/native:shared"), "entity:1"))[1:-1] in provenance
    assert json.dumps(_field_coordinate(FieldAddress("b", "entity:1/native:shared"), "entity:1"))[1:-1] in provenance
    assert provenance.count(json.dumps(_field_coordinate(None))[1:-1]) == 1
    assert _field_coordinate(FieldAddress("a@x", "y")) != _field_coordinate(FieldAddress("a", "x@y"))
    assert _field_coordinate(FieldAddress("a", "x", "entity:1")) != _field_coordinate(FieldAddress("a", "x", "entity:2"))
    assert _field_coordinate(FieldAddress("a", "x", "entity:1")) == _field_coordinate(FieldAddress("a", "x"), "entity:1")


def test_assembly_rejects_coordinated_runtime_field_collapse(monkeypatch):
    inputs = _derivation_inputs()
    runtime_ids = {
        "CORE.DISCOVERY_MATERIALIZATION", "CORE.CRITERION_INDEPENDENCE",
        "CORE.SCOPE_PROJECTION", "CORE.APPROXIMATION_DISCLOSURE",
    }
    facts = []
    for fact in derive_core_obligation_facts(*inputs):
        if fact.obligation_id in runtime_ids:
            references = tuple(replace(
                reference,
                coordinates=tuple(
                    _field_coordinate(None) if item.startswith("field=") else item
                    for item in reference.coordinates
                ),
            ) for reference in fact.evidence_references)
            fact = replace(fact, evidence_references=references)
        facts.append(fact)
    monkeypatch.setattr(ledger_module, "derive_core_obligation_facts", lambda *unused: tuple(facts))
    with pytest.raises(ConformanceValidationError, match="runtime field coordinates"):
        assemble_core_publication_rows(*inputs)


_DERIVATION_ATTACKS = (
    ("supported-domain-aggregate-omission", "rejection denominator"),
    ("producer-provenance-aggregate-misbinding", "producer provenance"),
    ("manifest-runtime-witness-omission", "field denominator"),
    ("reset-discovery-derivation-omission", "reset discovery"),
    ("runtime-schema-materialization-omission", "materialization"),
    ("finite-trace-union-omission", "complete trace union"),
    ("proper-scope-variant-omission", "exact ordered roles"),
    ("empty-scope-variant-omission", "exact ordered roles"),
    ("criterion-poison-variant-omission", "exact ordered roles"),
)


@pytest.mark.parametrize("attack,reason", _DERIVATION_ATTACKS, ids=[item[0] for item in _DERIVATION_ATTACKS])
def test_nine_obligation_derivation_attacks_fail_for_intended_reason(attack, reason):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    surface = case.surface
    if attack == "supported-domain-aggregate-omission":
        case = replace(case, rejection_denominator=(), rejections=())
    elif attack == "producer-provenance-aggregate-misbinding":
        denominator = replace(inventory.denominator, producer_provenance="wrong")
        inventory = replace(inventory, denominator=denominator)
    elif attack == "manifest-runtime-witness-omission":
        manifest = replace(manifest, field_denominator=("a",), fields=manifest.fields[:1])
    elif attack == "reset-discovery-derivation-omission":
        case = replace(case, surface=replace(surface, reset_discovery=surface.reset_discovery[:1]))
    elif attack == "runtime-schema-materialization-omission":
        calls = (surface.calls[0], replace(surface.calls[1], materialized=surface.calls[1].materialized[:1]))
        case = replace(case, surface=replace(surface, calls=calls))
    elif attack == "finite-trace-union-omission":
        case = replace(case, surface=replace(surface, complete_trace_union=surface.complete_trace_union[:1]))
    else:
        index = {"proper-scope-variant-omission": 1, "empty-scope-variant-omission": 2, "criterion-poison-variant-omission": 3}[attack]
        case = replace(case, variants=case.variants[:index] + case.variants[index + 1:])
    evidence = replace(evidence, cases=(case,))

    with pytest.raises((ConformanceValidationError, InventoryValidationError), match=reason):
        derive_core_obligation_facts(core, inventory, manifest, execution, evidence)


def test_derivation_rejects_untyped_inputs_and_exposes_no_ledger_result_types():
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    untyped = replace(evidence, cases=list(evidence.cases))
    with pytest.raises(ConformanceValidationError, match="exact raw evidence types"):
        derive_core_obligation_facts(core, inventory, manifest, execution, untyped)
    rejection = evidence.cases[0].rejections[0].execution
    unrelated = replace(core, case_denominator=(rejection.case_id,), source_denominator=(rejection.trace_source_declaration,), executions=(rejection,))
    with pytest.raises(ConformanceValidationError, match="paired unscoped execution"):
        derive_core_obligation_facts(unrelated, inventory, manifest, execution, evidence)
    forbidden = {"result", "assurance", "applicable", "certified"}
    assert not forbidden & set(DerivedObligationFact.__dataclass_fields__)


def test_structural_unexercised_is_not_satisfied():
    core, _, _, _, _ = _derivation_inputs()
    assessment = derive_assessment(core)
    target = next(item for item in assessment.predicate_results if item.predicate_id == "PROPERTY.SCALAR.V1")
    changed = replace(assessment, predicate_results=(replace(target, status=PredicateStatus.UNEXERCISED),))
    content = EvidenceIdentity("core", _digest(core.to_json()))
    with pytest.raises(ConformanceValidationError, match="exactly satisfied"):
        _predicate_references(changed, ("PROPERTY.SCALAR.V1",), content)


_SEMANTIC_FIELDS = (
    "owner", "address_population", "population_lifecycle", "value_transform",
    "unit", "frame", "author_key", "group", "tags", "discovery_behavior",
)


@pytest.mark.parametrize("field_name", _SEMANTIC_FIELDS, ids=["semantic-{}-custody".format(item.replace("_", "-")) for item in _SEMANTIC_FIELDS])
def test_manifest_semantics_reconcile_to_independent_raw_declarations(field_name):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    value = ("forged",) if field_name == "tags" else "forged-" + field_name
    changed = replace(manifest.fields[0], **{field_name: value})
    with pytest.raises(ConformanceValidationError, match="semantic manifest"):
        derive_core_obligation_facts(core, inventory, replace(manifest, fields=(changed,) + manifest.fields[1:]), execution, evidence)


def test_raw_semantic_declaration_is_hook_custodied():
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    changed = replace(case.candidates[0], address_population="forged")
    evidence = replace(evidence, cases=(replace(case, candidates=(changed,) + case.candidates[1:]),))
    with pytest.raises(ConformanceValidationError, match="materialization facts disagree"):
        derive_core_obligation_facts(core, inventory, manifest, execution, evidence)


@pytest.mark.parametrize("field_name", ("owner", "author_key", "group", "note", "tags", "unit", "family", "component", "frame"))
def test_runtime_owner_author_and_property_metadata_reconcile(field_name):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    value = ("forged",) if field_name == "tags" else "forged-" + field_name
    index = 1 if field_name == "owner" else 0
    changed = replace(case.candidates[index], **{field_name: value})
    candidates = case.candidates[:index] + (changed,) + case.candidates[index + 1:]
    case = replace(case, candidates=candidates)
    execution, evidence = _coordinate(execution, replace(evidence, cases=(case,)), "exact-value")
    with pytest.raises(ConformanceValidationError, match="runtime owner|runtime property metadata"):
        derive_core_obligation_facts(core, inventory, manifest, execution, evidence)


_DOMAIN_ATTACKS = (
    ("domain-source-provenance-custody", "source"),
    ("domain-configuration-omission", "configuration"),
    ("domain-configuration-invention", "configuration"),
    ("domain-kind-omission", "native kind audit"),
    ("domain-kind-silent-acceptance", "accepted kind witness"),
)


@pytest.mark.parametrize("attack,reason", _DOMAIN_ATTACKS, ids=[item[0] for item in _DOMAIN_ATTACKS])
def test_closed_supported_domain_attacks_are_rejected(attack, reason):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    domain = case.domain
    if attack == "domain-source-provenance-custody":
        domain = replace(domain, source=EvidenceIdentity("wrong", _digest("wrong")))
    elif attack == "domain-configuration-omission":
        domain = replace(domain, configuration_denominator=())
    elif attack == "domain-configuration-invention":
        domain = replace(domain, configuration_denominator=("configuration", "invented"))
    elif attack == "domain-kind-omission":
        domain = replace(domain, native_kind_denominator=("rigid_body",))
    else:
        domain = replace(domain, accepted_native_kinds=("rigid_body", "soft_body"))
        case = replace(case, rejection_denominator=(), rejections=())
    case = replace(case, domain=domain)
    execution, evidence = _coordinate(execution, replace(evidence, cases=(case,)), "exact-value")
    with pytest.raises(ConformanceValidationError, match=reason):
        derive_core_obligation_facts(core, inventory, manifest, execution, evidence)


def test_factory_closed_domain_needs_no_fake_rejection_case():
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    domain = replace(
        case.domain, native_kind_mode=NativeKindMode.FACTORY_CLOSED,
        native_kind_denominator=("rigid_body",), accepted_native_kinds=("rigid_body",),
    )
    case = replace(case, domain=domain, rejection_denominator=(), rejections=())
    execution, evidence = _coordinate(execution, replace(evidence, cases=(case,)), "exact-value")
    assert len(derive_core_obligation_facts(core, inventory, manifest, execution, evidence)) == 14


def test_multiowner_candidate_requires_complete_runtime_owner_custody():
    _, _, _, execution, evidence = _derivation_inputs()
    case = evidence.cases[0]
    addresses = case.denominator.address_denominator
    assert addresses[0].candidate == addresses[2].candidate == "a"
    assert {addresses[0].runtime_owner, addresses[2].runtime_owner} == {"entity:1", "entity:2"}
    validate_refinement_evidence(execution, evidence)
    output = case.variants[0].output
    owner_output = tuple(replace(item, owner="entity:1") if item.key.field == addresses[2] else item for item in output)
    owner_case = replace(case, variants=(replace(case.variants[0], output=owner_output),) + case.variants[1:])
    with pytest.raises(ConformanceValidationError, match="runtime owner"):
        validate_refinement_evidence(execution, replace(evidence, cases=(owner_case,)))
    changed = replace(addresses[2], runtime_owner=None)
    denominator = replace(case.denominator, address_denominator=addresses[:2] + (changed,))
    evidence = replace(evidence, cases=(replace(case, denominator=denominator),))
    with pytest.raises(ConformanceValidationError, match="all explicit"):
        validate_refinement_evidence(execution, evidence)
    changed = replace(addresses[2], runtime_owner="")
    denominator = replace(case.denominator, address_denominator=addresses[:2] + (changed,))
    with pytest.raises(ConformanceValidationError, match="runtime address owner"):
        validate_refinement_evidence(execution, replace(evidence, cases=(replace(case, denominator=denominator),)))


@pytest.mark.parametrize("attack", ("call-address-coordinate-omission", "reference-content-tamper"))
def test_evidence_references_digest_content_and_name_exact_coordinates(attack):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    facts = derive_core_obligation_facts(core, inventory, manifest, execution, evidence)
    runtime_ids = {"CORE.DISCOVERY_MATERIALIZATION", "CORE.CRITERION_INDEPENDENCE",
                   "CORE.SCOPE_PROJECTION", "CORE.APPROXIMATION_DISCLOSURE"}
    for fact in facts:
        assert all(len(reference.content.sha256) == 64 for reference in fact.evidence_references)
        if fact.obligation_id in runtime_ids:
            assert all(any(item.startswith("call=") for item in reference.coordinates) for reference in fact.evidence_references)
            assert all(any(item == _field_coordinate(None) or item.startswith("field=[") for item in reference.coordinates) for reference in fact.evidence_references)
    if attack == "reference-content-tamper":
        case = evidence.cases[0]
        artifact = freeze_json_artifact("native", "test-hook-v1", {"state": [9]})
        variants = []
        for index, variant in enumerate(case.variants):
            execution = _replace_hook(execution, index, 1, artifact)
            digests = list(variant.hook_artifact_sha256s)
            digests[1] = artifact.sha256
            variants.append(replace(variant, hook_artifact_sha256s=tuple(digests),
                                    native_snapshot_sha256=artifact.sha256))
        evidence = replace(evidence, cases=(replace(case, variants=tuple(variants)),))
        changed = derive_core_obligation_facts(core, inventory, manifest, execution, evidence)
        assert all(facts[index].evidence_references != changed[index].evidence_references
                   for index in (9, 10, 13))


@pytest.mark.parametrize(
    "part", ("enumeration", "record", "manifest"),
    ids=("inventory-enumeration-provenance-digest", "inventory-record-provenance-digest", "manifest-content-digest"),
)
def test_inventory_and_manifest_provenance_change_content_references(part):
    values = _derivation_inputs()
    before = derive_core_obligation_facts(*values)
    core, inventory, manifest, execution, evidence = values
    if part == "enumeration":
        inventory = replace(inventory, denominator=replace(inventory.denominator, enumeration_provenance="changed"))
    elif part == "record":
        inventory = replace(inventory, records=(replace(inventory.records[0], provenance="changed"),) + inventory.records[1:])
    else:
        manifest = replace(manifest, provenance="changed")
    after = derive_core_obligation_facts(core, inventory, manifest, execution, evidence)
    index = 7 if part != "manifest" else 6
    assert before[index].evidence_references != after[index].evidence_references


def test_no_silent_acceptance_digests_prior_fact_contents():
    facts = derive_core_obligation_facts(*_derivation_inputs())
    changed = replace(facts[0], evidence_references=(replace(facts[0].evidence_references[0], coordinates=("changed",)),))
    assert _content_reference("derived:" + facts[0].obligation_id, changed, "obligation=" + facts[0].obligation_id) != facts[-1].evidence_references[0]


class _ForgedType(type):
    def __hash__(cls):
        return hash(cls.__mro__[1])
    def __eq__(cls, other):
        return other is cls or other is cls.__mro__[1]


def _frozen_lookalike(value, forged):
    extra = {"CaseExecution": "result", "CandidateRecord": "assurance",
             "SemanticFieldRecord": "applicable", "FidelityVariantExecution": "certificate"}.get(
                 type(value).__name__, "result")
    module = sys.modules[type(value).__module__]
    metaclass = _ForgedType if forged else type
    cls = dataclass(frozen=True)(metaclass(
        "Caller" + type(value).__name__, (type(value),),
        {"__module__": module.__name__, "__annotations__": {extra: str}, extra: "forged"},
    ))
    setattr(module, cls.__name__, cls)
    return cls(**{item.name: getattr(value, item.name) for item in fields(type(value))})


@pytest.mark.parametrize(
    "tree", ("core", "inventory", "manifest", "execution", "refinement"),
    ids=("nested-core-lookalike", "nested-inventory-lookalike", "nested-manifest-lookalike",
         "nested-execution-lookalike", "nested-refinement-lookalike"),
)
@pytest.mark.parametrize("forged", (False, True), ids=("plain", "metaclass"))
def test_nested_frozen_lookalikes_are_rejected(tree, forged):
    core, inventory, manifest, execution, evidence = _derivation_inputs()
    if tree == "core":
        fake = _frozen_lookalike(core.executions[0], forged)
        core = replace(core, executions=(fake,))
        execution = replace(execution, executions=(replace(execution.executions[0], trace=fake),) + execution.executions[1:])
    elif tree == "inventory":
        inventory = replace(inventory, records=(_frozen_lookalike(inventory.records[0], forged),) + inventory.records[1:])
    elif tree == "manifest":
        manifest = replace(manifest, fields=(_frozen_lookalike(manifest.fields[0], forged),) + manifest.fields[1:])
    elif tree == "execution":
        execution = replace(execution, executions=(_frozen_lookalike(execution.executions[0], forged),) + execution.executions[1:])
    else:
        case = evidence.cases[0]
        evidence = replace(evidence, cases=(replace(case, candidates=(_frozen_lookalike(case.candidates[0], forged),) + case.candidates[1:]),))
        execution, evidence = _coordinate(execution, evidence, "exact-value")
    derived_validator = assemble_core_publication_rows if tree in ("core", "manifest") else derive_core_obligation_facts
    with pytest.raises(ConformanceValidationError, match="exact frozen"):
        if tree == "refinement":
            validate_refinement_evidence(execution, evidence)
        else:
            derived_validator(core, inventory, manifest, execution, evidence)


def test_direct_derived_fact_is_not_a_recomputation_authority():
    forged = DerivedObligationFact("CORE.NO_SILENT_ACCEPTANCE", (
        DerivedEvidenceReference(EvidenceIdentity("forged", _digest("forged")), ("forged",)),
    ))
    with pytest.raises(ConformanceValidationError, match="exact raw evidence types"):
        derive_core_obligation_facts(forged, forged, forged, forged, forged)
    with pytest.raises(ConformanceValidationError, match="exact raw evidence types"):
        assemble_core_publication_rows(forged, forged, forged, forged, forged)


@pytest.mark.parametrize(
    "mutation,reason",
    (
        ("unexecuted", "one executed failing public reset"),
        ("factory", "one executed failing public reset"),
        ("hook", "reset rejection exception type mismatch"),
        ("wrong-kind", "captured authority"),
        ("wrong-configuration", "configuration disagrees"),
    ),
)
def test_rejection_requires_captured_public_reset_authority(mutation, reason):
    execution = _execution()
    evidence = _valid_evidence(execution)
    case = evidence.cases[0]
    rejection = case.rejections[0]
    if mutation == "unexecuted":
        raw = execution.executions[0].trace
        raw = replace(
            raw,
            case_id=rejection.key.case_id,
            trace_source_declaration=rejection.execution.trace_source_declaration,
        )
        rejection = replace(rejection, execution=raw)
    elif mutation == "factory":
        raw = _capture_rejection(rejection.key, _rejection_factory_failure)
        rejection = replace(rejection, execution=raw)
    elif mutation == "hook":
        error = replace(rejection.execution.exceptions[0], phase="before_observation")
        raw = replace(rejection.execution, exceptions=(error,))
        rejection = replace(rejection, execution=raw)
    elif mutation == "wrong-kind":
        rejection = replace(
            rejection,
            key=replace(rejection.key, native_kind="made_up"),
        )
    elif mutation == "wrong-configuration":
        rejection = replace(
            rejection,
            key=replace(rejection.key, configuration_id="wrong"),
        )
    case = replace(
        case,
        rejection_denominator=(rejection.key,),
        rejections=(rejection,),
    )
    evidence = replace(evidence, cases=(case,))
    with pytest.raises(ConformanceValidationError, match=reason):
        validate_refinement_evidence(execution, evidence)

def test_scope_preserves_surviving_component_tags():
    execution = _execution()
    evidence = _valid_evidence(execution)
    variant = execution.executions[1]
    observation = variant.trace.observations[-1]
    value = json.loads(observation.canonical_json)
    value["component_tags"]["items"][0][1]["items"].append([
        {"type": "str", "value": "'a'"},
        {"items": [{"type": "str", "value": "'tag-a'"}], "type": "tuple"},
    ])
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    observations = variant.trace.observations[:-1] + (
        replace(observation, canonical_json=text, sha256=_digest(text)),
    )
    changed = replace(variant, trace=replace(variant.trace, observations=observations))
    execution = replace(
        execution,
        executions=(execution.executions[0], changed) + execution.executions[2:],
    )
    with pytest.raises(ConformanceValidationError, match="component tags disagree"):
        validate_refinement_evidence(execution, evidence)



def _typed_execution(execution, mutation=None):
    variants = []
    for index, variant in enumerate(execution.executions):
        observations = []
        links = (("a", "b"), ("b",), (), ("b",))[index]
        for observation in variant.trace.observations:
            payload = json.loads(observation.canonical_json)
            timestep = int(payload["timestep"]["value"])
            envelope = ExtractionObservation(
                timestep, timestep * 0.1, 0, 0, "a" * 64, "b" * 64,
                QueryConfiguration.empty(),
                (StructuredMetadata("fixture.state", "Native fixture array", "float64[2]",
                                    "unitless", "world", "identity", ("fixture.state",), "c" * 64),),
                (StructuredSample("fixture.state", "body", (7,),
                                  NativeValue.capture(np.array([1., 2.])), "present", 1, links),),
                verification_only=True,
            )
            payload["extraction"] = extraction_document(envelope)
            if index == 1 and mutation is not None:
                mutation(payload["extraction"])
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            observations.append(replace(observation, canonical_json=text, sha256=_digest(text)))
        variants.append(replace(variant, trace=replace(variant.trace, observations=tuple(observations))))
    return replace(execution, executions=tuple(variants))


def test_scope_preserves_native_values_while_projecting_structured_scalar_links():
    execution = _typed_execution(_execution())
    validate_refinement_evidence(execution, _valid_evidence(execution))


@pytest.mark.parametrize("mutation", ["value", "owner", "frame", "configuration", "sample", "missing_link", "extra_link"])
def test_scope_cannot_change_structured_native_evidence(mutation):
    def corrupt(envelope):
        if mutation == "value":
            envelope["samples"][0]["value"] = NativeValue.capture(np.array([1., 9.])).document()
        elif mutation == "owner":
            envelope["samples"][0]["owner_index"] = [8]
        elif mutation == "frame":
            envelope["metadata"][0]["frame"] = "body"
        elif mutation == "configuration":
            envelope["configuration_epoch"] += 1
        elif mutation == "sample":
            envelope["samples"] = []
        elif mutation == "missing_link":
            envelope["samples"][0]["scalar_keys"] = []
        else:
            envelope["samples"][0]["scalar_keys"] = ["a", "b"]

    execution = _typed_execution(_execution(), corrupt)
    with pytest.raises(ConformanceValidationError, match="observation context|structured links"):
        validate_refinement_evidence(execution, _valid_evidence(execution))

def test_nested_refinement_facts_require_exact_frozen_types():
    execution = _execution()
    evidence = _valid_evidence(execution)
    case = evidence.cases[0]
    original = case.candidates[0]
    fake = _MutableCandidate(original.candidate, original.provenance, None)
    evidence = replace(evidence, cases=(replace(case, candidates=(fake,) + case.candidates[1:]),))
    with pytest.raises(ConformanceValidationError, match="exact frozen fact types"):
        validate_refinement_evidence(execution, evidence)


def _rename_candidate(case):
    old = case.denominator.address_denominator[1]
    new = replace(old, candidate="c")
    field = lambda value: new if value == old else value
    key = lambda value: replace(value, field=field(value.field))
    denominator = replace(case.denominator, candidate_denominator=("a", "c"), address_denominator=(case.denominator.address_denominator[0], new), witness_denominator=tuple(key(item) for item in case.denominator.witness_denominator))
    candidates = (case.candidates[0], replace(case.candidates[1], candidate="c"))
    witnesses = tuple(replace(item, key=key(item.key)) for item in case.witnesses)
    calls = tuple(replace(item, discovered=tuple(field(value) for value in item.discovered), metadata=tuple(field(value) for value in item.metadata), materialized=tuple(field(value) for value in item.materialized)) for item in case.surface.calls)
    surface = replace(case.surface, runtime_schema_fields=(new,), reset_discovery=tuple(field(value) for value in case.surface.reset_discovery), complete_trace_union=tuple(field(value) for value in case.surface.complete_trace_union), calls=calls)
    variants = tuple(replace(item, requested_fields=None if item.requested_fields is None else tuple(field(value) for value in item.requested_fields), output=tuple(replace(value, key=key(value.key)) for value in item.output), expected_reads=tuple(key(value) for value in item.expected_reads), reads=tuple(key(value) for value in item.reads), expected_transfers=tuple(key(value) for value in item.expected_transfers), transfers=tuple(key(value) for value in item.transfers)) for item in case.variants)
    return replace(case, candidates=candidates, denominator=denominator, witnesses=witnesses, surface=surface, variants=variants)

def _drop_witness_pairs(case):
    keep = (case.denominator.witness_denominator[0], case.denominator.witness_denominator[3])
    witnesses = tuple(item for item in case.witnesses if item.key in keep)
    calls = tuple(
        replace(
            item,
            discovered=tuple(key.field for key in keep if key.call_id == item.call_id),
            metadata=tuple(key.field for key in keep if key.call_id == item.call_id),
            materialized=tuple(key.field for key in keep if key.call_id == item.call_id),
        )
        for item in case.surface.calls
    )
    surface = replace(
        case.surface,
        reset_discovery=calls[0].discovered,
        complete_trace_union=tuple(dict.fromkeys(key.field for key in keep)),
        calls=calls,
    )
    variants = tuple(
        replace(
            item,
            output=tuple(entry for entry in item.output if entry.key in keep),
            expected_reads=tuple(key for key in item.expected_reads if key in keep),
            reads=tuple(key for key in item.reads if key in keep),
            expected_transfers=tuple(key for key in item.expected_transfers if key in keep),
            transfers=tuple(key for key in item.transfers if key in keep),
        )
        for item in case.variants
    )
    denominator = replace(case.denominator, witness_denominator=keep)
    return replace(
        case, denominator=denominator, witnesses=witnesses,
        surface=surface, variants=variants,
    )


def _mutate(evidence, name):
    case = evidence.cases[0]
    witnesses = case.witnesses
    variants = case.variants
    surface = case.surface
    rejection = case.rejections[0]
    if name == "exact-value":
        changed = '{"type":"float","value":"9.0"}'
        witnesses = tuple(replace(item, expected_json=changed, emitted_json=changed) if isinstance(item, ExactWitness) else item for item in witnesses)
        variants = tuple(replace(item, output=tuple(replace(value, value_json=changed) if value.key.field.candidate == "a" else value for value in item.output)) for item in variants)
    elif name == "wrong-shape":
        witnesses = (ApproximateWitness(witnesses[0].key, "1.0", "absolute-error", 0.0, 0.0, witnesses[0].provenance),) + witnesses[1:]
    elif name == "nonfinite-bound":
        witnesses = (witnesses[0], replace(witnesses[1], bound=float("inf"))) + witnesses[2:]
    elif name == "bound-exceeded":
        witnesses = (witnesses[0], replace(witnesses[1], residual=0.3)) + witnesses[2:]
    elif name == "candidate-denominator":
        return replace(evidence, cases=(_rename_candidate(case),))
    elif name == "witness-cross-product":
        return replace(evidence, cases=(_drop_witness_pairs(case),))
    elif name == "call-denominator":
        case = replace(case, denominator=replace(case.denominator, call_denominator=("reset",)))
    elif name == "provenance":
        wrong = WitnessProvenance("wrong", witnesses[0].provenance.producer)
        witnesses = (replace(witnesses[0], provenance=wrong),) + witnesses[1:]
    elif name == "discovery":
        calls = (replace(surface.calls[0], discovered=(surface.calls[0].discovered[0],)),) + surface.calls[1:]
        surface = replace(surface, calls=calls)
    elif name == "metadata":
        calls = (replace(surface.calls[0], metadata=(surface.calls[0].metadata[0],)),) + surface.calls[1:]
        surface = replace(surface, calls=calls)
    elif name == "materialization":
        calls = (replace(surface.calls[0], materialized=(surface.calls[0].materialized[0],)),) + surface.calls[1:]
        surface = replace(surface, calls=calls)
    elif name == "trace-union":
        surface = replace(surface, complete_trace_union=(surface.complete_trace_union[0],))
    elif name == "variant-roles":
        variants = (variants[0], replace(variants[1], role="proper_scope", requested_fields=case.denominator.address_denominator), replace(variants[2], role="empty_scope", requested_fields=(case.denominator.address_denominator[0],)), variants[3])
    elif name == "projection":
        variants = tuple(
            replace(
                item,
                output=tuple(
                    replace(
                        entry,
                        observation_pointer=entry.observation_pointer.replace(
                            "/perceived_entities/items/0/",
                            "/perceived_entities/items/2/",
                        ),
                    )
                    for entry in item.output
                ),
            )
            for item in variants
        )
    elif name == "unrequested-read":
        variants = (variants[0], replace(variants[1], reads=case.denominator.witness_denominator)) + variants[2:]
    elif name == "scope-over-emission":
        pass
    elif name == "criterion-native":
        variants = variants[:-1] + (replace(variants[-1], native_snapshot_sha256=_digest("wrong")),)
    elif name == "criterion-output":
        variants = variants[:-1] + (replace(variants[-1], output=()),)
    elif name == "criterion-interaction":
        pass
    elif name == "rejection-denominator":
        missing = RejectionKey("configuration", "missing", "rigid_body")
        case = replace(case, rejection_denominator=case.rejection_denominator + (missing,))
    elif name == "rejection-type":
        rejection = replace(rejection, expected_exception_type="ValueError")
    elif name == "rejection-message":
        rejection = replace(rejection, expected_message_fragment="different")
    elif name == "authority-custody":
        return replace(evidence, execution_authority_sha256=_digest("wrong"))
    elif name == "hook-custody":
        pass
    else:
        raise AssertionError(name)
    case = replace(case, witnesses=witnesses, surface=surface, variants=variants, rejections=(rejection,))
    return replace(evidence, cases=(case,))


def _replace_hook(execution, variant_index, hook_index, artifact, exception=None):
    variant = execution.executions[variant_index]
    hook = replace(variant.hooks[hook_index], artifact=artifact, exception=exception)
    hooks = variant.hooks[:hook_index] + (hook,) + variant.hooks[hook_index + 1:]
    variants = execution.executions[:variant_index] + (replace(variant, hooks=hooks),) + execution.executions[variant_index + 1:]
    return replace(execution, executions=variants)


def _coordinate(execution, evidence, name):
    case = evidence.cases[0]
    if name in ("exact-value", "witness-cross-product", "projection"):
        variants = list(case.variants)
        for index, variant in enumerate(variants):
            material = {"expected_reads": _jsonable(variant.expected_reads), "expected_transfers": _jsonable(variant.expected_transfers), "output": _jsonable(variant.output)}
            if index == 0:
                names = ("configuration_id", "case_id", "producer", "domain", "candidates", "denominator", "witnesses", "surface")
                material["case"] = {key: _jsonable(getattr(case, key)) for key in names}
            artifact = freeze_json_artifact("materialization", "specola-refinement-materialization-v1", material)
            execution = _replace_hook(execution, index, 2, artifact)
            digests = list(variant.hook_artifact_sha256s); digests[2] = artifact.sha256
            variants[index] = replace(variant, hook_artifact_sha256s=tuple(digests))
        evidence = replace(evidence, cases=(replace(case, variants=tuple(variants)),))
        if name == "projection":
            executions = []
            for execution_fact in execution.executions:
                observations = []
                for observation in execution_fact.trace.observations:
                    value = json.loads(observation.canonical_json)
                    entity = json.loads(json.dumps(value["perceived_entities"]["items"][0]))
                    entity[0]["value"] = "3"
                    entity[1]["entity_id"]["value"] = "3"
                    entity[1]["entity_class"]["value"] = "'robot'"
                    value["perceived_entities"]["items"].append(entity)
                    tags = json.loads(json.dumps(value["component_tags"]["items"][0]))
                    tags[0]["value"] = "3"
                    value["component_tags"]["items"].append(tags)
                    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
                    observations.append(replace(
                        observation, canonical_json=text, sha256=_digest(text),
                    ))
                executions.append(replace(
                    execution_fact,
                    trace=replace(
                        execution_fact.trace,
                        observations=tuple(observations),
                        metadata_probes=tuple(
                            probe
                            for observation in execution_fact.trace.observations
                            for probe in tuple(
                                item for item in execution_fact.trace.metadata_probes
                                if item.call == observation.call
                            ) + (replace(next(
                                item for item in execution_fact.trace.metadata_probes
                                if item.call == observation.call
                            ), entity_id=3),)
                        ),
                    ),
                ))
            execution = replace(execution, executions=tuple(executions))
    elif name == "criterion-interaction":
        variant = case.variants[-1]
        payload = {"reads": _jsonable(variant.reads), "reward": 7, "success": 3, "transfers": _jsonable(variant.transfers), "verdict": 2}
        artifact = freeze_json_artifact("interactions", "specola-refinement-interactions-v1", payload)
        execution = _replace_hook(execution, 3, 3, artifact)
        digests = list(variant.hook_artifact_sha256s); digests[3] = artifact.sha256
        variants = case.variants[:-1] + (replace(variant, hook_artifact_sha256s=tuple(digests)),)
        evidence = replace(evidence, cases=(replace(case, variants=variants),))
    elif name == "scope-over-emission":
        variant = execution.executions[1]
        observations = []
        for observation in variant.trace.observations:
            value = json.loads(observation.canonical_json)
            entity = value["perceived_entities"]["items"][0][1]
            entity["properties"]["items"].append([
                {"type": "str", "value": "'z'"},
                {"type": "float", "value": "99.0"},
            ])
            tags = value["component_tags"]["items"][0][1]["items"]
            tags.append([
                {"type": "str", "value": "'z'"},
                {"items": [{"type": "str", "value": "'tag-z'"}], "type": "tuple"},
            ])
            text = json.dumps(value, sort_keys=True, separators=(",", ":"))
            observations.append(replace(
                observation, canonical_json=text, sha256=_digest(text),
            ))
        changed = replace(
            variant, trace=replace(variant.trace, observations=tuple(observations)),
        )
        execution = replace(
            execution,
            executions=(execution.executions[0], changed) + execution.executions[2:],
        )
    elif name == "hook-custody":
        error = ExceptionFact(None, "fidelity_hook", "RuntimeError", "hook failed", "captured hook traceback", False)
        execution = _replace_hook(execution, 0, 2, None, error)
    return execution, evidence


_MUTATIONS = (
    ("exact-value", "refinement output disagrees with frozen observation"),
    ("wrong-shape", "witness shape disagrees with candidate declaration"),
    ("nonfinite-bound", "approximation residual and bound must be finite"),
    ("bound-exceeded", "approximation residual exceeds bound"),
    ("candidate-denominator", "candidate denominator disagrees with execution authority"),
    ("witness-cross-product", "witness denominator must be exact ordered address-call product"),
    ("call-denominator", "call denominator disagrees with paired execution"),
    ("provenance", "witness provenance disagrees with candidate declaration"),
    ("discovery", "discovery must match witness denominator for call"),
    ("metadata", "metadata must match discovery for call"),
    ("materialization", "materialization must match discovery for call"),
    ("trace-union", "complete trace union disagrees with materialized calls"),
    ("variant-roles", "variants need exact ordered roles"),
    ("projection", "projection owner or identity disagrees with frozen entity"),
    ("unrequested-read", "scoped reads include unrequested field"),
    ("scope-over-emission", "scoped observation properties disagree with declared output"),
    ("criterion-native", "criterion variants need equal native snapshot"),
    ("criterion-output", "criterion variants need equal output"),
    ("criterion-interaction", "interaction facts disagree with captured hook"),
    ("rejection-denominator", "rejections must exactly match denominator"),
    ("rejection-type", "reset rejection exception type mismatch"),
    ("rejection-message", "reset rejection message mismatch"),
    ("authority-custody", "refinement authority digest disagrees with execution"),
    ("hook-custody", "refinement needs successful trace and required hooks"),
)


@pytest.mark.parametrize("mutation,reason", _MUTATIONS, ids=[item[0] for item in _MUTATIONS])
def test_twenty_four_adversarial_mutations_fail_for_intended_reason(mutation, reason):
    execution = _execution()
    evidence = _mutate(_valid_evidence(execution), mutation)
    execution, evidence = _coordinate(execution, evidence, mutation)

    with pytest.raises(ConformanceValidationError, match=reason):
        validate_refinement_evidence(execution, evidence)
