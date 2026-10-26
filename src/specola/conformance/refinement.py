"""Immutable semantic facts and pure backend-calculated refinement checks."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional, Tuple, Union

from .core_runner import CallExpectation, CallKind, CaseExecution, ConformanceValidationError, EvidenceIdentity
from .fidelity_execution import FidelityExecutionArtifact

class VariantRole(str, Enum):
    UNSCOPED = "unscoped"
    PROPER_SCOPE = "proper_scope"
    EMPTY_SCOPE = "empty_scope"
    CRITERION_POISON = "criterion_poison"

class NativeKindMode(str, Enum):
    EXTERNALLY_SUPPLIED = "externally_supplied"
    FACTORY_CLOSED = "factory_closed"
@dataclass(frozen=True)
class ProducerProvenance:
    name: str
    configuration_id: str
    version: Optional[str]
    content_sha256: Optional[str]
    type_source_sha256: Optional[str]
@dataclass(frozen=True)
class FieldAddress:
    candidate: str
    address: str
    runtime_owner: Optional[str] = None
@dataclass(frozen=True)
class WitnessKey:
    field: FieldAddress
    call_id: str
@dataclass(frozen=True)
class WitnessProvenance:
    source_accessor: str
    producer: ProducerProvenance
@dataclass(frozen=True)
class CandidateDeclaration:
    candidate: str
    provenance: WitnessProvenance
    approximation_relation: Optional[str]
    owner: str
    address_population: str
    population_lifecycle: str
    value_transform: str
    author_key: str
    discovery_behavior: str
    group: str
    note: Optional[str]
    tags: Tuple[str, ...]
    unit: Optional[str]
    family: Optional[str]
    component: Optional[str]
    frame: Optional[str]

@dataclass(frozen=True)
class SupportedDomainFacts:
    source: EvidenceIdentity
    native_kind_mode: NativeKindMode
    configuration_denominator: Tuple[str, ...]
    native_kind_denominator: Tuple[str, ...]
    accepted_native_kinds: Tuple[str, ...]
    case_native_kind: str
@dataclass(frozen=True)
class RuntimeDenominator:
    candidate_denominator: Tuple[str, ...]
    address_denominator: Tuple[FieldAddress, ...]
    call_denominator: Tuple[str, ...]
    witness_denominator: Tuple[WitnessKey, ...]
@dataclass(frozen=True)
class ExactWitness:
    key: WitnessKey
    expected_json: str
    emitted_json: str
    provenance: WitnessProvenance
@dataclass(frozen=True)
class ApproximateWitness:
    key: WitnessKey
    emitted_json: str
    relation_id: str
    residual: float
    bound: float
    provenance: WitnessProvenance

RefinementWitness = Union[ExactWitness, ApproximateWitness]
@dataclass(frozen=True)
class SurfaceCallFact:
    call_id: str
    discovered: Tuple[FieldAddress, ...]
    metadata: Tuple[FieldAddress, ...]
    materialized: Tuple[FieldAddress, ...]
@dataclass(frozen=True)
class DiscoveryMaterializationFacts:
    manifest_fields: Tuple[FieldAddress, ...]
    runtime_schema_fields: Tuple[FieldAddress, ...]
    reset_discovery: Tuple[FieldAddress, ...]
    complete_trace_union: Tuple[FieldAddress, ...]
    calls: Tuple[SurfaceCallFact, ...]
@dataclass(frozen=True)
class ProjectionEntry:
    key: WitnessKey
    observation_pointer: str
    author_key: str
    value_json: str
    owner: str
    metadata_json: str
    identity: str
@dataclass(frozen=True)
class CriterionInteractionFact:
    reward: int
    success: int
    verdict: int
@dataclass(frozen=True)
class VariantRefinementFacts:
    variant_id: str
    role: VariantRole
    requested_fields: Optional[Tuple[FieldAddress, ...]]
    hook_artifact_sha256s: Tuple[str, ...]
    native_snapshot_sha256: str
    output: Tuple[ProjectionEntry, ...]
    expected_reads: Tuple[WitnessKey, ...]
    reads: Tuple[WitnessKey, ...]
    expected_transfers: Tuple[WitnessKey, ...]
    transfers: Tuple[WitnessKey, ...]
    interactions: CriterionInteractionFact

@dataclass(frozen=True)
class RejectionKey:
    configuration_id: str
    case_id: str
    native_kind: str

@dataclass(frozen=True)
class ExpectedResetRejection:
    key: RejectionKey
    expected_exception_type: str
    expected_message_fragment: str
    execution: CaseExecution

@dataclass(frozen=True)
class CaseRefinementFacts:
    configuration_id: str
    case_id: str
    producer: ProducerProvenance
    domain: SupportedDomainFacts
    candidates: Tuple[CandidateDeclaration, ...]
    denominator: RuntimeDenominator
    witnesses: Tuple[RefinementWitness, ...]
    surface: DiscoveryMaterializationFacts
    variants: Tuple[VariantRefinementFacts, ...]
    rejection_denominator: Tuple[RejectionKey, ...]
    rejections: Tuple[ExpectedResetRejection, ...]

@dataclass(frozen=True)
class RefinementEvidence:
    execution_authority_sha256: str
    cases: Tuple[CaseRefinementFacts, ...]

_EXACT_FACT_TYPES = frozenset(
    value
    for module_name in (__name__, CaseExecution.__module__)
    for value in vars(sys.modules[module_name]).values()
    if isinstance(value, type) and is_dataclass(value) and value.__module__ == module_name
)

def validate_refinement_evidence(execution, evidence):
    """Validate raw facts without deriving obligations or assurance labels."""
    if type(execution) is not FidelityExecutionArtifact or type(evidence) is not RefinementEvidence:
        raise ConformanceValidationError("refinement validation needs typed execution and evidence")
    _immutable(evidence)
    if evidence.execution_authority_sha256 != execution.authority.sha256:
        raise ConformanceValidationError("refinement authority digest disagrees with execution")
    if tuple(item.case_id for item in evidence.cases) != execution.case_denominator:
        raise ConformanceValidationError("refinement cases must exactly match execution denominator")
    records = json.loads(execution.authority.canonical_json)["payload"]["cases"]
    for case in evidence.cases:
        paired = tuple(item for item in execution.executions if item.case_id == case.case_id)
        authority = tuple(item for item in records if item["case_id"] == case.case_id)
        _validate_case(case, paired, authority, execution.configuration_denominator)
    domains = tuple(case.domain for case in evidence.cases)
    headers = tuple((item.source, item.native_kind_mode, item.configuration_denominator,
                     item.native_kind_denominator, item.accepted_native_kinds) for item in domains)
    if any(item != headers[0] for item in headers):
        raise ConformanceValidationError("refinement cases disagree on supported domain")
    expected = tuple((configuration, kind) for configuration in domains[0].configuration_denominator
                     for kind in domains[0].accepted_native_kinds)
    actual = tuple(dict.fromkeys((case.configuration_id, case.domain.case_native_kind) for case in evidence.cases))
    if actual != expected:
        raise ConformanceValidationError("accepted kind witness denominator is not exact")

def _validate_case(case, paired, authority, configurations):
    _texts((case.configuration_id, case.case_id), "case identity")
    if not paired or any(item.configuration_id != case.configuration_id for item in paired):
        raise ConformanceValidationError("refinement case configuration disagrees with execution")
    _validate_producer(case.producer, authority)
    declarations = _unique(case.candidates, lambda item: item.candidate, "candidate declarations")
    _validate_declarations(declarations)
    denominator = case.denominator
    _unique_text(denominator.candidate_denominator, "candidate denominator")
    if denominator.candidate_denominator != tuple(item.candidate for item in declarations):
        raise ConformanceValidationError("candidate denominator disagrees with declarations")
    if any(item.candidate_denominator != denominator.candidate_denominator for item in paired):
        raise ConformanceValidationError("candidate denominator disagrees with execution authority")
    if any(tuple(item["candidate_denominator"]) != denominator.candidate_denominator for item in authority):
        raise ConformanceValidationError("candidate denominator disagrees with execution authority")
    addresses = _unique(
        denominator.address_denominator,
        lambda item: (item.candidate, item.address),
        "address denominator",
    )
    if tuple(dict.fromkeys(item.candidate for item in addresses)) != denominator.candidate_denominator:
        raise ConformanceValidationError("address denominator disagrees with candidate denominator")
    for candidate in denominator.candidate_denominator:
        owners = tuple(item.runtime_owner for item in addresses if item.candidate == candidate)
        if any(item is None for item in owners) and any(item is not None for item in owners):
            raise ConformanceValidationError("candidate address owners must be all explicit or all declaration-owned")
        for owner in owners:
            if owner is not None:
                _text(owner, "runtime address owner")
    _unique_text(denominator.call_denominator, "call denominator")
    actual_calls = tuple(item.call.kind.value for item in paired[0].trace.observations)
    paired_calls = tuple(tuple(fact.call.kind.value for fact in item.trace.observations) for item in paired)
    if denominator.call_denominator != actual_calls or any(item != actual_calls for item in paired_calls):
        raise ConformanceValidationError("call denominator disagrees with paired execution")
    witness_keys = _unique(
        denominator.witness_denominator,
        lambda item: (item.field, item.call_id),
        "witness denominator",
    )
    expected_keys = tuple(
        WitnessKey(address, call_id)
        for call_id in denominator.call_denominator
        for address in addresses
    )
    if witness_keys != expected_keys:
        raise ConformanceValidationError("witness denominator must be exact ordered address-call product")
    witnesses = _unique(case.witnesses, lambda item: item.key, "runtime witnesses")
    if tuple(item.key for item in witnesses) != witness_keys:
        raise ConformanceValidationError("runtime witnesses must exactly match witness denominator")
    by_candidate = {item.candidate: item for item in declarations}
    for witness in witnesses:
        declaration = by_candidate[witness.key.field.candidate]
        if declaration.approximation_relation is not None:
            _text(declaration.approximation_relation, "approximation relation")
        if witness.provenance != declaration.provenance:
            raise ConformanceValidationError("witness provenance disagrees with candidate declaration")
        if witness.provenance.producer != case.producer:
            raise ConformanceValidationError("candidate provenance disagrees with measured producer")
        _text(witness.provenance.source_accessor, "source accessor")
        if declaration.approximation_relation is None:
            if not isinstance(witness, ExactWitness):
                raise ConformanceValidationError("witness shape disagrees with candidate declaration")
            _canonical_json(witness.expected_json)
            _canonical_json(witness.emitted_json)
            if witness.expected_json != witness.emitted_json:
                raise ConformanceValidationError("exact witness values differ")
        else:
            if not isinstance(witness, ApproximateWitness) or witness.relation_id != declaration.approximation_relation:
                raise ConformanceValidationError("witness shape disagrees with candidate declaration")
            _canonical_json(witness.emitted_json)
            if not math.isfinite(witness.residual) or not math.isfinite(witness.bound):
                raise ConformanceValidationError("approximation residual and bound must be finite")
            if witness.residual < 0 or witness.bound < 0 or witness.residual > witness.bound:
                raise ConformanceValidationError("approximation residual exceeds bound")
    _validate_surface(case.surface, denominator)
    _validate_rejections(case, paired)
    _validate_domain(case, paired, configurations)
    _validate_variants(case, paired)

def _validate_domain(case, paired, configurations):
    domain = case.domain
    _unique_text(domain.configuration_denominator, "supported configuration denominator")
    _unique_text(domain.native_kind_denominator, "native kind denominator")
    _unique_text(domain.accepted_native_kinds, "accepted native kinds")
    _text(domain.case_native_kind, "case native kind")
    if domain.configuration_denominator != configurations:
        raise ConformanceValidationError("supported configuration denominator disagrees with execution")
    if domain.case_native_kind not in domain.accepted_native_kinds:
        raise ConformanceValidationError("case native kind is not in accepted domain")
    if any(item not in domain.native_kind_denominator for item in domain.accepted_native_kinds):
        raise ConformanceValidationError("accepted native kinds must belong to native kind denominator")
    rejected = tuple(item for item in domain.native_kind_denominator if item not in domain.accepted_native_kinds)
    if domain.native_kind_mode is NativeKindMode.FACTORY_CLOSED:
        if rejected:
            raise ConformanceValidationError("factory-closed native kind denominator must equal accepted kinds")
        expected = ()
    elif domain.native_kind_mode is NativeKindMode.EXTERNALLY_SUPPLIED:
        expected = tuple((case.configuration_id, kind) for kind in rejected)
    else:
        raise ConformanceValidationError("supported domain needs typed native kind mode")
    actual = tuple((item.configuration_id, item.native_kind) for item in case.rejection_denominator)
    if actual != expected or any(item.configuration_id != case.configuration_id for item in paired):
        raise ConformanceValidationError("supported native kind audit is not closed")

def _validate_declarations(declarations):
    for declaration in declarations:
        _texts((
            declaration.owner, declaration.address_population,
            declaration.population_lifecycle, declaration.value_transform,
            declaration.author_key, declaration.discovery_behavior,
            declaration.group,
        ), "semantic declaration")
        for value in (declaration.note, declaration.unit, declaration.family,
                      declaration.component, declaration.frame):
            if value is not None:
                _text(value, "optional semantic declaration")
        if type(declaration.tags) is not tuple or any(type(item) is not str or not item for item in declaration.tags):
            raise ConformanceValidationError("semantic declaration tags must be text tuple")

def _validate_producer(producer, authority):
    _texts((producer.name, producer.configuration_id), "producer provenance")
    expected = (producer.name, producer.configuration_id, producer.version,
                producer.content_sha256, producer.type_source_sha256)
    observed = []
    for item in authority:
        source = item["producer"]
        observed.append((source["name"], source["configuration_id"],
                         source["version"], source["content_sha256"],
                         source["type_source"]["sha256"]))
    if not observed or any(item != expected for item in observed):
        raise ConformanceValidationError("producer provenance disagrees with execution authority")
    if producer.version is None and producer.content_sha256 is None:
        raise ConformanceValidationError("producer provenance needs version or content digest")

def _validate_surface(surface, denominator):
    manifest = _unique(surface.manifest_fields, lambda item: item, "manifest fields")
    runtime = _unique(surface.runtime_schema_fields, lambda item: item, "runtime schema fields")
    if set(manifest) & set(runtime) or set(manifest) | set(runtime) != set(denominator.address_denominator):
        raise ConformanceValidationError("manifest and runtime schema must partition address denominator")
    if tuple(item.call_id for item in surface.calls) != denominator.call_denominator:
        raise ConformanceValidationError("surface calls must exactly match call denominator")
    materialized = []
    for call in surface.calls:
        expected = tuple(item.field for item in denominator.witness_denominator if item.call_id == call.call_id)
        if call.discovered != expected:
            raise ConformanceValidationError("discovery must match witness denominator for call")
        if call.metadata != call.discovered:
            raise ConformanceValidationError("metadata must match discovery for call")
        if call.materialized != call.discovered:
            raise ConformanceValidationError("materialization must match discovery for call")
        materialized.extend(call.materialized)
    if surface.reset_discovery != surface.calls[0].discovered:
        raise ConformanceValidationError("reset discovery disagrees with reset call")
    if surface.complete_trace_union != tuple(dict.fromkeys(materialized)):
        raise ConformanceValidationError("complete trace union disagrees with materialized calls")

def _validate_variants(case, paired):
    variants = case.variants
    denominator = case.denominator
    roles = (VariantRole.UNSCOPED, VariantRole.PROPER_SCOPE, VariantRole.EMPTY_SCOPE, VariantRole.CRITERION_POISON)
    if any(type(item.role) is not VariantRole for item in variants) or tuple(item.role for item in variants) != roles:
        raise ConformanceValidationError("variants need exact ordered roles")
    if tuple(item.variant_id for item in variants) != tuple(item.variant_id for item in paired):
        raise ConformanceValidationError("variants need exact ordered roles")
    full = variants[0]
    declarations = {item.candidate: item for item in case.candidates}
    if full.requested_fields is not None or tuple(item.key for item in full.output) != denominator.witness_denominator:
        raise ConformanceValidationError("unscoped output must cover exact witness denominator")
    for item in full.output:
        _canonical_json(item.value_json)
        _canonical_json(item.metadata_json)
        _texts((item.owner, item.identity), "projection ownership and identity")
        declaration = declarations[item.key.field.candidate]
        runtime_owner = item.key.field.runtime_owner
        expected_owner = declaration.owner if runtime_owner is None else runtime_owner
        if item.owner != expected_owner or item.author_key != declaration.author_key:
            raise ConformanceValidationError("runtime owner or author key disagrees with semantic declaration")
        if item.metadata_json != _declaration_metadata(declaration):
            raise ConformanceValidationError("runtime property metadata disagrees with semantic declaration")
    if tuple(item.value_json for item in full.output) != tuple(item.emitted_json for item in case.witnesses):
        raise ConformanceValidationError("unscoped output values disagree with refinement witnesses")
    poison = variants[3]
    proper = variants[1]
    if _projection_semantics(poison.output) != _projection_semantics(proper.output):
        raise ConformanceValidationError("criterion variants need equal output")
    all_fields = denominator.address_denominator
    full_observations = paired[0].trace.observations
    for variant, execution in zip(variants, paired):
        _validate_interactions(variant.interactions)
        variant_keys = tuple(entry.key for entry in variant.output)
        selected = tuple(item for item in full.output if item.key in variant_keys)
        actual = _variant_contexts(
            denominator.call_denominator, execution.trace.observations,
            variant.output, True,
        )
        expected = _variant_contexts(
            denominator.call_denominator, full_observations,
            selected, False,
        )
        if actual != expected:
            raise ConformanceValidationError("scope and criterion variants changed observation context")
    _validate_accesses(full, all_fields, denominator)
    if len({item.native_snapshot_sha256 for item in variants[:3]}) != 1:
        raise ConformanceValidationError("scope variants need equal native snapshot")
    for variant in variants[1:3]:
        scope = variant.requested_fields
        if scope is None or any(item not in all_fields for item in scope):
            raise ConformanceValidationError("scoped variant needs declared fields")
        if variant.role is VariantRole.PROPER_SCOPE and (not scope or len(scope) >= len(all_fields)):
            raise ConformanceValidationError("proper scope must be nonempty strict subset")
        if variant.role is VariantRole.EMPTY_SCOPE and scope:
            raise ConformanceValidationError("empty scope must request no fields")
        expected_output = tuple(item for item in full.output if item.key.field in scope)
        if _projection_semantics(variant.output) != _projection_semantics(expected_output):
            raise ConformanceValidationError("scoped output is not exact projection")
        _validate_accesses(variant, scope, denominator)
    if poison.requested_fields != proper.requested_fields:
        raise ConformanceValidationError("criterion variants need equal scope")
    _validate_accesses(poison, poison.requested_fields or (), denominator)
    if poison.native_snapshot_sha256 != proper.native_snapshot_sha256:
        raise ConformanceValidationError("criterion variants need equal native snapshot")
    for variant, execution in zip(variants, paired):
        _validate_custody(case, variant, execution)

def _variant_contexts(call_ids, observations, output, exact_properties):
    return tuple(
        _observation_context(
            observation,
            tuple(item for item in output if item.key.call_id == call_id),
            exact_properties,
        )
        for call_id, observation in zip(call_ids, observations)
    )

def _projection_semantics(output):
    return tuple(
        (item.key, item.author_key, item.value_json, item.owner,
         item.metadata_json, item.identity) for item in output
    )

def _validate_custody(case, variant, execution):
    if execution.trace.exceptions or any(item.artifact is None for item in execution.hooks):
        raise ConformanceValidationError("refinement needs successful trace and required hooks")
    digests = tuple(item.artifact.sha256 for item in execution.hooks)
    if variant.hook_artifact_sha256s != digests:
        raise ConformanceValidationError("refinement hook custody disagrees with execution")
    if variant.native_snapshot_sha256 != digests[1]:
        raise ConformanceValidationError("native snapshot digest disagrees with captured hook")
    expected = {
        "reads": _plain(variant.reads), "reward": variant.interactions.reward,
        "success": variant.interactions.success, "transfers": _plain(variant.transfers),
        "verdict": variant.interactions.verdict,
    }
    if _artifact_payload(execution.hooks[3].artifact, "specola-refinement-interactions-v1") != expected:
        raise ConformanceValidationError("interaction facts disagree with captured hook")
    material = {
        "expected_reads": _plain(variant.expected_reads),
        "expected_transfers": _plain(variant.expected_transfers), "output": _plain(variant.output),
    }
    if variant.role is VariantRole.UNSCOPED:
        names = (
            "configuration_id", "case_id", "producer", "domain", "candidates",
            "denominator", "witnesses", "surface",
        )
        material["case"] = {name: _plain(getattr(case, name)) for name in names}
    if _artifact_payload(execution.hooks[2].artifact, "specola-refinement-materialization-v1") != material:
        raise ConformanceValidationError("materialization facts disagree with captured hook")
    observations = {
        call_id: json.loads(observation.canonical_json)
        for call_id, observation in zip(
            case.denominator.call_denominator, execution.trace.observations,
        )
    }
    for entry in variant.output:
        _, observed = _projection_location(observations[entry.key.call_id], entry)
        if observed != json.loads(entry.value_json):
            raise ConformanceValidationError("refinement output disagrees with frozen observation")

def _validate_accesses(variant, scope, denominator):
    groups = (
        (variant.expected_reads, "expected reads"),
        (variant.reads, "native reads"),
        (variant.expected_transfers, "expected transfers"),
        (variant.transfers, "native transfers"),
    )
    for values, label in groups:
        _unique(values, lambda item: item, label)
        if any(item not in denominator.witness_denominator for item in values):
            raise ConformanceValidationError("native accesses must belong to witness denominator")
    if any(item.field not in scope for item in variant.reads):
        raise ConformanceValidationError("scoped reads include unrequested field")
    if variant.reads != variant.expected_reads:
        raise ConformanceValidationError("native reads disagree with expected denominator")
    if any(item.field not in scope for item in variant.transfers):
        raise ConformanceValidationError("scoped transfers include unrequested field")
    if variant.transfers != variant.expected_transfers:
        raise ConformanceValidationError("native transfers disagree with expected denominator")

def _validate_interactions(interactions):
    if not isinstance(interactions, CriterionInteractionFact):
        raise ConformanceValidationError("criterion interactions need nonnegative integer counts")
    counts = (interactions.reward, interactions.success, interactions.verdict)
    if any(type(item) is not int or item < 0 for item in counts):
        raise ConformanceValidationError("criterion interactions need nonnegative integer counts")
    if counts != (0, 0, 0):
        raise ConformanceValidationError("criterion interactions must be zero")

def _validate_rejections(case, paired):
    denominator = case.rejection_denominator
    needs_rejection = set(case.domain.native_kind_denominator) != set(case.domain.accepted_native_kinds)
    if (
        not isinstance(denominator, tuple) or (not denominator and needs_rejection)
        or any(type(item) is not RejectionKey for item in denominator)
    ):
        raise ConformanceValidationError("rejection denominator needs typed expected cases")
    _unique(denominator, lambda item: (item.configuration_id, item.case_id, item.native_kind), "rejection denominator")
    if tuple(item.key for item in case.rejections) != denominator:
        raise ConformanceValidationError("rejections must exactly match denominator")
    authority_configurations = {item.configuration_id for item in paired}
    for item in case.rejections:
        key = item.key
        _texts(
            (key.configuration_id, key.case_id, key.native_kind,
             item.expected_exception_type, item.expected_message_fragment),
            "reset rejection declaration",
        )
        if key.configuration_id != case.configuration_id or authority_configurations != {key.configuration_id}:
            raise ConformanceValidationError("reset rejection configuration disagrees with execution authority")
        execution = item.execution
        identity = "refinement-reset-rejection:{}:{}:{}".format(key.configuration_id, key.case_id, key.native_kind)
        if (
            type(execution) is not CaseExecution
            or execution.case_id != key.case_id
            or execution.trace_source_declaration.identity != identity
        ):
            raise ConformanceValidationError("reset rejection case disagrees with captured authority")
        if not execution.plan:
            raise ConformanceValidationError("reset rejection public call was not planned")
        call = execution.plan[0]
        if call.kind is not CallKind.RESET or call.expectation is not CallExpectation.RETURN_OBSERVATION:
            raise ConformanceValidationError("reset rejection needs one executed failing public reset")
        if execution.attempted != (call,) or execution.successful or execution.observations:
            raise ConformanceValidationError("reset rejection needs one executed failing public reset")
        if len(execution.bound_callables) != 1:
            raise ConformanceValidationError("reset rejection needs captured public reset binding")
        bound = execution.bound_callables[0]
        if (
            bound.call != call or bound.method_name != "reset"
            or getattr(bound.callable, "__self__", None) is not bound.driver
        ):
            raise ConformanceValidationError("reset rejection did not invoke the public reset binding")
        before = execution.provenance_before
        after = execution.provenance_after
        if before is None or after is None or before != after:
            raise ConformanceValidationError("reset rejection adapter authority is incomplete")
        if (
            type(bound.driver).__module__ != before.direct_module
            or type(bound.driver).__qualname__ != before.direct_qualname
        ):
            raise ConformanceValidationError("reset rejection adapter authority is incomplete")
        if len(execution.exceptions) != 1:
            raise ConformanceValidationError("reset rejection needs one captured exception")
        actual = execution.exceptions[0]
        if (
            actual.call != call or actual.phase != "call" or actual.expected
            or actual.exception_type != item.expected_exception_type
        ):
            raise ConformanceValidationError("reset rejection exception type mismatch")
        if item.expected_message_fragment not in actual.message or key.native_kind not in actual.message:
            raise ConformanceValidationError("reset rejection message mismatch")

def _canonical_json(value):
    try:
        parsed = json.loads(value)
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConformanceValidationError("witness value must be canonical finite JSON: {}".format(error))
    if canonical != value:
        raise ConformanceValidationError("witness value must be canonical finite JSON")

def _artifact_payload(artifact, schema):
    if artifact.schema != schema:
        raise ConformanceValidationError("refinement hook schema mismatch")
    return json.loads(artifact.canonical_json)["payload"]

def _json_pointer(value, pointer):
    _text(pointer, "observation JSON pointer")
    try:
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ConformanceValidationError("observation JSON pointer does not resolve")
    return value

def _projection_location(value, entry):
    tokens = entry.observation_pointer.strip("/").split("/")
    if (
        len(tokens) != 8
        or tokens[:2] != ["perceived_entities", "items"]
        or tokens[3:6] != ["1", "properties", "items"]
        or tokens[7] != "1"
    ):
        raise ConformanceValidationError("projection pointer must address one entity property")
    entity_pointer = "/" + "/".join(tokens[:3])
    entity_key = _json_pointer(value, entity_pointer + "/0")
    entity = _json_pointer(value, entity_pointer + "/1")
    author_key = _json_pointer(value, entry.observation_pointer.rsplit("/", 1)[0] + "/0")
    if not isinstance(author_key, dict) or author_key.get("value") != repr(entry.author_key):
        raise ConformanceValidationError("projection author key disagrees with frozen observation")
    try:
        identity_parts = entry.identity.rsplit(":", 1)
        correct_owner = entry.owner == "entity:" + str(entity_key["value"])
        correct_identity = (
            len(identity_parts) == 2
            and entity["entity_class"]["value"] == repr(identity_parts[0])
            and str(entity["entity_id"]["value"]) == identity_parts[1]
            and entity_key == entity["entity_id"]
        )
    except (KeyError, TypeError):
        correct_owner = False
        correct_identity = False
    if not correct_owner or not correct_identity:
        raise ConformanceValidationError("projection owner or identity disagrees with frozen entity")
    encoded_entity = json.dumps(entity_key, sort_keys=True, separators=(",", ":"))
    return encoded_entity, _json_pointer(value, entry.observation_pointer)

def _observation_context(observation, output, exact_properties):
    value = json.loads(observation.canonical_json)
    allowed = set()
    for item in output:
        entity_key, _ = _projection_location(value, item)
        allowed.add((entity_key, repr(item.author_key)))
    emitted = set()
    for pair in value["perceived_entities"]["items"]:
        entity_key = json.dumps(pair[0], sort_keys=True, separators=(",", ":"))
        for property_pair in pair[1]["properties"]["items"]:
            emitted.add((entity_key, property_pair[0].get("value")))
    if exact_properties and emitted != allowed:
        raise ConformanceValidationError("scoped observation properties disagree with declared output")
    filtered_tags = []
    emitted_tags = set()
    for pair in value["component_tags"]["items"]:
        entity_key = json.dumps(pair[0], sort_keys=True, separators=(",", ":"))
        emitted_tags.update((entity_key, tag[0].get("value")) for tag in pair[1]["items"])
        kept = [tag for tag in pair[1]["items"] if (entity_key, tag[0].get("value")) in allowed]
        if kept:
            pair[1]["items"] = kept
            filtered_tags.append(pair)
    if exact_properties and not emitted_tags.issubset(allowed):
        raise ConformanceValidationError("scoped component tags disagree with declared output")
    value["component_tags"]["items"] = filtered_tags
    if "extraction" in value:
        # The typed sink retains the configured native subject under scalar
        # scope. Project only its scalar links; keep values and semantic
        # identity in the comparison so a scoped reader cannot change them.
        for sample in value["extraction"]["samples"]:
            entity_key = json.dumps({"type": "int", "value": repr(sample["entity_id"])},
                                    sort_keys=True, separators=(",", ":"))
            projected = [key for key in sample["scalar_keys"] if (entity_key, repr(key)) in allowed]
            if exact_properties and projected != sample["scalar_keys"]:
                raise ConformanceValidationError("scoped structured links disagree with declared output")
            sample["scalar_keys"] = projected
    for pair in value["perceived_entities"]["items"]:
        pair[1].pop("properties")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _declaration_metadata(declaration):
    return json.dumps({
        "component": declaration.component, "family": declaration.family,
        "frame": declaration.frame, "group": declaration.group,
        "note": declaration.note, "tags": list(declaration.tags),
        "unit": declaration.unit,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _plain(values):
    if is_dataclass(values) and not isinstance(values, type):
        return {key: _plain(value) for key, value in asdict(values).items()}
    if isinstance(values, Enum):
        return values.value
    if isinstance(values, dict):
        return {key: _plain(value) for key, value in values.items()}
    if isinstance(values, tuple):
        return [_plain(item) for item in values]
    return values

def _immutable(value):
    if isinstance(value, (list, dict, set)):
        raise ConformanceValidationError("refinement facts must use immutable containers")
    if is_dataclass(value):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen or not any(type(value) is item for item in _EXACT_FACT_TYPES):
            raise ConformanceValidationError("refinement facts must use exact frozen fact types")
        children = (getattr(value, item.name) for item in fields(value) if item.compare)
    elif isinstance(value, tuple):
        children = iter(value)
    else:
        return
    for item in children:
        _immutable(item)

def _unique(values, key, label):
    if not isinstance(values, tuple):
        raise ConformanceValidationError("{} must be unique tuple".format(label))
    keys = tuple(key(item) for item in values)
    if len(set(keys)) != len(keys):
        raise ConformanceValidationError("{} must be unique tuple".format(label))
    return values

def _unique_text(values, label):
    if (
        not isinstance(values, tuple) or not values
        or any(type(item) is not str or not item.strip() for item in values)
        or len(set(values)) != len(values)
    ):
        raise ConformanceValidationError("{} must be nonempty unique text tuple".format(label))

def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ConformanceValidationError("{} must be nonempty text".format(label))

def _texts(values, label):
    for value in values:
        _text(value, label)
