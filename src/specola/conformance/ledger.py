"""Immutable, engine-neutral adapter-fidelity inventory records.

The ledger accounts for declared finite populations and evidence.  It does not
define simulator physics, adapter extraction, or a universal semantic model;
backend manifests remain responsible for those meanings.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import sys
from typing import Dict, Iterable, Optional, Tuple

from .bindability import BindabilityArtifact, validate_bindability_artifact
from .core_runner import (
    ConformanceValidationError, CoreConformanceArtifact, EvidenceIdentity,
    PredicateStatus, derive_assessment,
)
from .fidelity_execution import FidelityExecutionArtifact
from .refinement import ApproximateWitness, RefinementEvidence, VariantRole, validate_refinement_evidence


class CandidateDisposition(str, Enum):
    """One mutually exclusive disposition for a denominator candidate."""

    INCLUDED = "included"
    EXCLUDED = "excluded"
    UNRESOLVED = "unresolved"


class ExclusionCategory(str, Enum):
    """Audited bases for excluding a candidate from a declared surface."""

    NON_STATE_OPERATIONAL_CACHE = "non_state_operational_cache"
    DUPLICATE_ALIAS = "duplicate_alias"
    OUTSIDE_DECLARED_SUPPORTED_DOMAIN = "outside_declared_supported_domain"
    UNPOPULATED_IN_DECLARED_LIFECYCLE = "unpopulated_in_declared_lifecycle"


class ExactnessKind(str, Enum):
    """Whether a manifest field is exact or uses a declared approximation."""

    EXACT = "exact"
    APPROXIMATE = "approximate"


class AssuranceLabel(str, Enum):
    """Evidence vocabulary retained from the formal-conformance design."""

    ENFORCED = "enforced"
    EXHAUSTIVELY_CLASSIFIED = "exhaustively_classified"
    PROPERTY_TESTED = "property_tested"
    DIFFERENTIALLY_CHECKED = "differentially_checked"
    SOURCE_AUDITED = "source_audited"
    UNVERIFIED = "unverified"


ACCEPTED_ASSURANCE_LABELS = frozenset(
    label for label in AssuranceLabel if label is not AssuranceLabel.UNVERIFIED
)


class ApplicabilityKind(str, Enum):
    """Whether a catalogue definition is mandatory or source-domain dependent."""

    CORE = "core"
    CONDITIONAL = "conditional"


class ObligationResult(str, Enum):
    """Per-obligation result without collapsing incomplete work to failure."""

    PASS = "pass"
    FAIL = "fail"
    UNEXECUTED = "unexecuted"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AssuranceRule:
    """Minimum PASS evidence: every ``all_of`` and one ``any_of`` label."""

    all_of: Tuple[AssuranceLabel, ...]
    any_of: Tuple[AssuranceLabel, ...]


@dataclass(frozen=True)
class ObligationDefinition:
    """One stable obligation in a versioned finite catalogue."""

    obligation_id: str
    applicability: ApplicabilityKind
    pass_evidence_rule: AssuranceRule


@dataclass(frozen=True)
class ObligationCatalogue:
    """Versioned authoritative denominator for certification obligations."""

    identity: str
    version: str
    definition_denominator: Tuple[str, ...]
    definitions: Tuple[ObligationDefinition, ...]
    provenance: str


@dataclass(frozen=True)
class Denominator:
    """A named finite candidate population and the evidence for its boundary."""

    identity: str
    candidates: Tuple[str, ...]
    enumeration_provenance: str
    producer_provenance: str


@dataclass(frozen=True)
class CandidateRecord:
    """Classification and evidence for exactly one denominator candidate."""

    name: str
    disposition: CandidateDisposition
    rationale: str
    provenance: str
    exclusion_category: Optional[ExclusionCategory] = None
    exclusion_detail: Optional[str] = None


@dataclass(frozen=True)
class FidelityInventory:
    """An immutable investigation inventory and its declared partition."""

    denominator: Denominator
    records: Tuple[CandidateRecord, ...]


@dataclass(frozen=True)
class SemanticFieldRecord:
    """Manifest semantics for one included inventory candidate."""

    candidate_name: str
    source_accessor: str
    owner: str
    address_population: str
    population_lifecycle: str
    value_transform: str
    unit: str
    frame: str
    exactness: ExactnessKind
    approximation_relation: Optional[str]
    author_key: str
    group: str
    tags: Tuple[str, ...]
    discovery_behavior: str
    producer_provenance: str
    note: Optional[str] = None
    family: Optional[str] = None
    component: Optional[str] = None


@dataclass(frozen=True)
class SemanticManifest:
    """Versioned semantics for exactly the included inventory population."""

    identity: str
    version: str
    field_denominator: Tuple[str, ...]
    fields: Tuple[SemanticFieldRecord, ...]
    provenance: str


@dataclass(frozen=True)
class ObligationRecord:
    """Evidence-backed outcome for one catalogue obligation."""

    obligation_id: str
    result: ObligationResult
    applicable: bool
    rationale: Optional[str]
    assurance_labels: Tuple[AssuranceLabel, ...]
    evidence_provenance: Tuple[str, ...]


@dataclass(frozen=True)
class AdapterCertification:
    """A closed inventory and exact per-obligation catalogue coverage."""

    inventory: FidelityInventory
    manifest: SemanticManifest
    catalogue: ObligationCatalogue
    obligations: Tuple[ObligationRecord, ...]


class InventoryValidationError(ValueError):
    """Raised when an inventory or certification record is inconsistent."""


def _runtime_rule() -> AssuranceRule:
    return AssuranceRule(
        all_of=(AssuranceLabel.SOURCE_AUDITED,),
        any_of=(
            AssuranceLabel.ENFORCED,
            AssuranceLabel.PROPERTY_TESTED,
            AssuranceLabel.DIFFERENTIALLY_CHECKED,
        ),
    )


def _closure_rule(*, source_audited: bool) -> AssuranceRule:
    all_of = [AssuranceLabel.EXHAUSTIVELY_CLASSIFIED]
    if source_audited:
        all_of.append(AssuranceLabel.SOURCE_AUDITED)
    return AssuranceRule(all_of=tuple(all_of), any_of=())


def _source_rule() -> AssuranceRule:
    return AssuranceRule(all_of=(AssuranceLabel.SOURCE_AUDITED,), any_of=())


def _definition(
    obligation_id: str,
    applicability: ApplicabilityKind,
    rule: AssuranceRule,
) -> ObligationDefinition:
    return ObligationDefinition(
        obligation_id=obligation_id,
        applicability=applicability,
        pass_evidence_rule=rule,
    )


def _build_specola_v1_catalogue() -> ObligationCatalogue:
    runtime = _runtime_rule()
    source = _source_rule()
    source_closure = _closure_rule(source_audited=True)
    closure = _closure_rule(source_audited=False)
    definitions = (
        _definition("CORE.LIFECYCLE", ApplicabilityKind.CORE, runtime),
        _definition("CORE.OBSERVATION_STRUCTURE", ApplicabilityKind.CORE, runtime),
        _definition("CORE.TIME_INDEX", ApplicabilityKind.CORE, runtime),
        _definition("CORE.IDENTITY_CLASS", ApplicabilityKind.CORE, runtime),
        _definition("CORE.VALUE_DOMAIN", ApplicabilityKind.CORE, runtime),
        _definition("CORE.SUPPORTED_DOMAIN", ApplicabilityKind.CORE, source_closure),
        _definition("CORE.SEMANTIC_MANIFEST", ApplicabilityKind.CORE, runtime),
        _definition("CORE.INVENTORY_CLOSURE", ApplicabilityKind.CORE, closure),
        _definition("CORE.DISCOVERY_MATERIALIZATION", ApplicabilityKind.CORE, runtime),
        _definition("CORE.CRITERION_INDEPENDENCE", ApplicabilityKind.CORE, runtime),
        _definition("CORE.SCOPE_PROJECTION", ApplicabilityKind.CORE, runtime),
        _definition("CORE.PRODUCER_PROVENANCE", ApplicabilityKind.CORE, source),
        _definition("CORE.APPROXIMATION_DISCLOSURE", ApplicabilityKind.CORE, runtime),
        _definition("CORE.NO_SILENT_ACCEPTANCE", ApplicabilityKind.CORE, runtime),
        _definition("CORE.STATIC_SLOT_BINDABILITY", ApplicabilityKind.CORE, runtime),
        _definition("CAP.RIGID_BODY", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("CAP.ARTICULATION", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("CAP.DEFORMABLE_STATE", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("CAP.DEFORMABLE_TOPOLOGY", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("CAP.CONTACT", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("CAP.PHYSICAL_SENSOR", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("FAC.CAMERA_CAPTURE", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("FAC.RENDERING", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("FAC.SCENE_GEOMETRY", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("FAC.SCENE_POSES", ApplicabilityKind.CONDITIONAL, runtime),
        _definition("FAC.AUTO_WRAP", ApplicabilityKind.CONDITIONAL, runtime),
    )
    return ObligationCatalogue(
        identity="specola-adapter-fidelity",
        version="1.1",
        definition_denominator=tuple(
            definition.obligation_id for definition in definitions
        ),
        definitions=definitions,
        provenance=(
            "approved adapter-fidelity certification semantics 2026-08-13; "
            "CORE.STATIC_SLOT_BINDABILITY added 2026-09-02 by "
            "docs/superpowers/plans/2026-09-02-never-bound-static-slot.md"
        ),
    )


SPECOLA_V1_CATALOGUE = _build_specola_v1_catalogue()


@dataclass(frozen=True)
class DerivedEvidenceReference:
    """Content identity plus exact coordinates into one validated input."""
    content: EvidenceIdentity
    coordinates: Tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.content) is not EvidenceIdentity:
            raise ConformanceValidationError("derived evidence needs exact content identity")
        if type(self.coordinates) is not tuple or not self.coordinates:
            raise ConformanceValidationError("derived evidence needs exact coordinates")
        if any(type(item) is not str or not item for item in self.coordinates):
            raise ConformanceValidationError("derived evidence coordinates must be text")


@dataclass(frozen=True)
class DerivedObligationFact:
    """One internally derived core-obligation evidence fact, never a result."""
    obligation_id: str
    evidence_references: Tuple[DerivedEvidenceReference, ...]

    def __post_init__(self) -> None:
        if type(self.obligation_id) is not str or not self.obligation_id.strip():
            raise ConformanceValidationError("derived obligation ID must be text")
        if (
            type(self.evidence_references) is not tuple or not self.evidence_references
            or any(type(item) is not DerivedEvidenceReference for item in self.evidence_references)):
            raise ConformanceValidationError("derived obligation evidence references have wrong type")


_STRUCTURAL_OBLIGATIONS = (
    ("CORE.LIFECYCLE", ("CALL.PUBLIC_CALLABLE.V1",)),
    ("CORE.OBSERVATION_STRUCTURE", ("OBSERVATION.TYPE.V1", "ENTITY.STRUCTURE.V1")),
    ("CORE.TIME_INDEX", ("OBSERVATION.TIME_INDEX.V1", "OBSERVATION.CROSS_CALL_TIME_INDEX.V1")),
    ("CORE.IDENTITY_CLASS", ("ENTITY.IDENTITY_CLASS_STABILITY.V1",)),
    ("CORE.VALUE_DOMAIN", ("VECTOR.STRUCTURE.V1", "PROPERTY.SCALAR.V1")),
)


def _field_coordinate(field, fallback_owner=None):
    if field is None:
        return "field=none"
    owner = field.runtime_owner if field.runtime_owner is not None else fallback_owner
    return "field=" + json.dumps((field.candidate, field.address, owner), separators=(",", ":"))


def _runtime_owner(case, field):
    if field.runtime_owner is not None:
        return field.runtime_owner
    return next(item.owner for item in case.candidates if item.candidate == field.candidate)


def _case_field_coordinate(case, field):
    return _field_coordinate(field, None if field is None else _runtime_owner(case, field))


def derive_core_obligation_facts(core, inventory, manifest, execution, evidence, bindability=None):
    """Purely derive the exact core evidence facts from raw inputs.

    ``bindability`` is the optional static-slot bindability artifact
    (``specola.conformance.bindability``). A ``pass`` outcome adds the
    fifteenth core fact; every other outcome, and its absence, leaves the
    derivation at the fourteen facts the five raw inputs support, because
    a non-PASS obligation has no PASS evidence to derive.
    """
    expected_types = (CoreConformanceArtifact, FidelityInventory, SemanticManifest,
                      FidelityExecutionArtifact, RefinementEvidence)
    actual_types = tuple(type(item) for item in (core, inventory, manifest, execution, evidence))
    if actual_types != expected_types or type(evidence.cases) is not tuple:
        raise ConformanceValidationError("derivation needs exact raw evidence types")
    if bindability is not None and type(bindability) is not BindabilityArtifact:
        raise ConformanceValidationError("derivation needs exact raw evidence types")
    for value in (core, inventory, manifest, execution, evidence, bindability):
        _exact_frozen_tree(value)  # a None probe is not a fact tree
    validate_semantic_manifest(inventory, manifest)
    validate_refinement_evidence(execution, evidence)
    unscoped = tuple(
        next(item.trace for item in execution.executions if (item.case_id, item.variant_id)
             == (case.case_id, case.variants[0].variant_id))
        for case in evidence.cases
    )
    if core.case_denominator != execution.case_denominator or core.executions != unscoped:
        raise ConformanceValidationError("core evidence disagrees with paired unscoped execution")
    assessment = derive_assessment(core)
    core_content = _content_reference("core-conformance", core, "denominator=core").content
    facts = [
        DerivedObligationFact(obligation_id, _predicate_references(assessment, predicates, core_content))
        for obligation_id, predicates in _STRUCTURAL_OBLIGATIONS
    ]
    cases = evidence.cases
    included = manifest.field_denominator
    producer_references = tuple(_producer_reference(case.producer) for case in cases)
    if any(case.denominator.candidate_denominator != included for case in cases):
        raise ConformanceValidationError("runtime witnesses disagree with manifest field denominator")
    if inventory.denominator.producer_provenance not in producer_references:
        raise ConformanceValidationError("inventory producer provenance disagrees with measured producer")
    if any(case.domain.source != core.subject_declarations.supported_domain_source_declaration for case in cases):
        raise ConformanceValidationError("supported domain source disagrees with core authority")
    fields = {field.candidate_name: field for field in manifest.fields}
    for case in cases:
        for declaration in case.candidates:
            field = fields[declaration.candidate]
            if _manifest_semantics(field) != _declaration_semantics(declaration, case.producer):
                raise ConformanceValidationError("semantic manifest disagrees with runtime witness declaration")
    inventory_reference = _content_reference("inventory", inventory, "denominator=inventory")
    manifest_reference = _content_reference("manifest", manifest, "denominator=manifest")
    paired_by_key = {(item.case_id, item.variant_id): item for item in execution.executions}
    def hook_reference(case, variant, hook, *coordinates):
        artifact = paired_by_key[(case.case_id, variant.variant_id)].hooks[hook].artifact
        return DerivedEvidenceReference(EvidenceIdentity(artifact.identity, artifact.sha256), tuple(coordinates))
    def variant_references(roles):
        return tuple(
            _content_reference("variant-execution", paired_by_key[(case.case_id, variant.variant_id)],
                               "case=" + case.case_id, "variant=" + variant.variant_id,
                               "call=" + (entry.key.call_id if entry else "none"),
                               _case_field_coordinate(case, entry.key.field if entry else None))
            for case in cases for variant in case.variants if variant.role in roles
            for entry in variant.output or (None,)
        )
    facts.extend((
        DerivedObligationFact("CORE.SUPPORTED_DOMAIN", tuple(
            hook_reference(case, case.variants[0], 2, "case=" + case.case_id,
                           "configuration=" + case.configuration_id, "domain=" + kind)
            for case in cases for kind in case.domain.native_kind_denominator
        )),
        DerivedObligationFact("CORE.SEMANTIC_MANIFEST", (manifest_reference,) + tuple(
            hook_reference(case, case.variants[0], 2, "case=" + case.case_id,
                           "candidate=" + key.field.candidate, "call=" + key.call_id,
                           "address=" + key.field.address,
                           "owner=" + _runtime_owner(case, key.field))
            for case in cases for key in case.denominator.witness_denominator
        )),
        DerivedObligationFact("CORE.INVENTORY_CLOSURE", (inventory_reference,)),
        DerivedObligationFact("CORE.DISCOVERY_MATERIALIZATION", tuple(
            hook_reference(case, case.variants[0], 2, "case=" + case.case_id,
                           "call=" + key.call_id, _case_field_coordinate(case, key.field))
            for case in cases for key in case.denominator.witness_denominator
        )),
        DerivedObligationFact("CORE.CRITERION_INDEPENDENCE", variant_references(
            (VariantRole.PROPER_SCOPE, VariantRole.CRITERION_POISON))),
        DerivedObligationFact("CORE.SCOPE_PROJECTION", variant_references(
            (VariantRole.UNSCOPED, VariantRole.PROPER_SCOPE, VariantRole.EMPTY_SCOPE))),
        DerivedObligationFact("CORE.PRODUCER_PROVENANCE", tuple(
            DerivedEvidenceReference(EvidenceIdentity(execution.authority.identity, execution.authority.sha256),
                                     ("case=" + case.case_id, "configuration=" + case.configuration_id,
                                      "producer=" + reference))
            for case, reference in zip(cases, producer_references)
        )),
        DerivedObligationFact("CORE.APPROXIMATION_DISCLOSURE", tuple(
            hook_reference(case, case.variants[0], 2, "case=" + case.case_id,
                           "call=" + witness.key.call_id, _case_field_coordinate(case, witness.key.field),
                           "relation=" + (witness.relation_id if isinstance(witness, ApproximateWitness) else "exact"))
            for case in cases for witness in case.witnesses
        )),
    ))
    derived_ids = SPECOLA_V1_CATALOGUE.definition_denominator[:14]
    if tuple(item.obligation_id for item in facts) != derived_ids[:-1]:
        raise ConformanceValidationError("derived prior-core denominator is not exact")
    facts.append(DerivedObligationFact("CORE.NO_SILENT_ACCEPTANCE", tuple(
        _content_reference("derived:" + item.obligation_id, item, "obligation=" + item.obligation_id)
        for item in facts
    )))
    if _bindability_passes(bindability):
        facts.append(DerivedObligationFact(
            "CORE.STATIC_SLOT_BINDABILITY", (_bindability_reference(bindability),)
        ))
    return tuple(facts)


_BINDABILITY_UNEXECUTED_RATIONALE = "static slot bindability probe not executed"
_BINDABILITY_FAIL_RATIONALE = "static slot bindability probe reported a never-bindable slot"


def bindability_obligation_status(bindability):
    """The result and rationale one probe outcome supports, as a pair.

    ``pass`` and ``fail`` are the only outcomes carrying runtime evidence
    about bindability. An absent, not-applicable, or setup-unavailable
    probe stays UNEXECUTED, because a probe that did not decide proves
    nothing either way. Row drivers that assemble their own obligation
    shape derive it here rather than restating the mapping.

    The outcome is validated against the results it summarizes before it
    is read, so an artifact whose summary the probe could not have
    produced raises rather than deciding a row.
    """
    if bindability is None:
        return ObligationResult.UNEXECUTED, _BINDABILITY_UNEXECUTED_RATIONALE
    validate_bindability_artifact(bindability)
    if bindability.outcome == "pass":
        return ObligationResult.PASS, None
    if bindability.outcome == "fail":
        return ObligationResult.FAIL, _BINDABILITY_FAIL_RATIONALE
    return (
        ObligationResult.UNEXECUTED,
        "static slot bindability probe outcome: " + bindability.outcome,
    )


def _bindability_passes(bindability):
    # Through the status mapping, so a PASS fact is never derived from an
    # outcome the probe's own results do not support.
    return bindability_obligation_status(bindability)[0] is ObligationResult.PASS


def _bindability_reference(bindability):
    return _content_reference(
        "bindability", bindability,
        "adapter=" + bindability.adapter_type,
        "outcome=" + bindability.outcome,
        "cases=" + str(len(bindability.cases)),
    )


def _bindability_row(bindability):
    """The conservative record for a static-slot bindability artifact.

    Absent or unexecuted evidence stays UNEXECUTED: a probe that could
    not run proves nothing about bindability. A ``fail`` outcome is
    recorded as FAIL with the probe as its evidence.
    """
    result, rationale = bindability_obligation_status(bindability)
    if bindability is None:
        return ObligationRecord(
            "CORE.STATIC_SLOT_BINDABILITY", result, True,
            rationale, (AssuranceLabel.UNVERIFIED,),
            (rationale or "static slot bindability probe not executed",),
        )
    provenance = (json.dumps(
        _plain_fact(_bindability_reference(bindability)),
        sort_keys=True, separators=(",", ":"),
    ),)
    if result is ObligationResult.FAIL:
        return ObligationRecord(
            "CORE.STATIC_SLOT_BINDABILITY", result, True, rationale,
            (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.DIFFERENTIALLY_CHECKED),
            provenance,
        )
    return ObligationRecord(
        "CORE.STATIC_SLOT_BINDABILITY", result, True, rationale,
        (AssuranceLabel.UNVERIFIED,), provenance,
    )


_CORE_EVIDENCE_MODES = (
    ("CORE.LIFECYCLE", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.OBSERVATION_STRUCTURE", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.TIME_INDEX", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.IDENTITY_CLASS", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.VALUE_DOMAIN", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.SUPPORTED_DOMAIN", (
        AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,
        AssuranceLabel.SOURCE_AUDITED,
    )),
    ("CORE.SEMANTIC_MANIFEST", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
    ("CORE.INVENTORY_CLOSURE", (AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,)),
    ("CORE.DISCOVERY_MATERIALIZATION", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
    ("CORE.CRITERION_INDEPENDENCE", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
    ("CORE.SCOPE_PROJECTION", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
    ("CORE.PRODUCER_PROVENANCE", (AssuranceLabel.SOURCE_AUDITED,)),
    ("CORE.APPROXIMATION_DISCLOSURE", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
    ("CORE.NO_SILENT_ACCEPTANCE", (AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED)),
    ("CORE.STATIC_SLOT_BINDABILITY", (
        AssuranceLabel.SOURCE_AUDITED,
        AssuranceLabel.DIFFERENTIALLY_CHECKED,
    )),
)


def assemble_core_publication_rows(core, inventory, manifest, execution, evidence, bindability=None):
    """Recompute and validate the exact conservative publication rows.

    ``bindability`` is the optional static-slot bindability artifact; see
    ``derive_core_obligation_facts``. Without one the static-slot
    obligation is recorded UNEXECUTED rather than assumed.
    """
    facts = derive_core_obligation_facts(core, inventory, manifest, execution, evidence, bindability)
    core_ids = SPECOLA_V1_CATALOGUE.definition_denominator[:15]
    derived_ids = core_ids if _bindability_passes(bindability) else core_ids[:14]
    if tuple(item.obligation_id for item in facts) != derived_ids:
        raise ConformanceValidationError("publication rows require exact core obligation denominator")
    modes = dict(_CORE_EVIDENCE_MODES)
    if tuple(modes) != core_ids:
        raise ConformanceValidationError("publication evidence-mode mapping is not exact")
    expected_fields = {
        "CORE.DISCOVERY_MATERIALIZATION": tuple(
            _case_field_coordinate(case, key.field)
            for case in evidence.cases
            for key in case.denominator.witness_denominator
        ),
        "CORE.CRITERION_INDEPENDENCE": tuple(
            _case_field_coordinate(case, entry.key.field if entry else None)
            for case in evidence.cases
            for variant in case.variants
            if variant.role in (VariantRole.PROPER_SCOPE, VariantRole.CRITERION_POISON)
            for entry in variant.output or (None,)
        ),
        "CORE.SCOPE_PROJECTION": tuple(
            _case_field_coordinate(case, entry.key.field if entry else None)
            for case in evidence.cases
            for variant in case.variants
            if variant.role in (VariantRole.UNSCOPED, VariantRole.PROPER_SCOPE, VariantRole.EMPTY_SCOPE)
            for entry in variant.output or (None,)
        ),
        "CORE.APPROXIMATION_DISCLOSURE": tuple(
            _case_field_coordinate(case, witness.key.field)
            for case in evidence.cases
            for witness in case.witnesses
        ),
    }
    facts_by_id = {item.obligation_id: item for item in facts}
    for obligation_id, expected in expected_fields.items():
        actual = tuple(
            coordinate
            for reference in facts_by_id[obligation_id].evidence_references
            for coordinate in reference.coordinates
            if coordinate.startswith("field=")
        )
        if actual != expected:
            raise ConformanceValidationError("publication runtime field coordinates are not exact")

    statuses = {item.status for item in derive_assessment(core).predicate_results}
    if statuses & {PredicateStatus.VIOLATED, PredicateStatus.UNEXECUTED}:
        raise ConformanceValidationError("publication predicates are violated or unexecuted")

    rows = tuple(
        ObligationRecord(
            fact.obligation_id,
            ObligationResult.PASS,
            True,
            None,
            modes[fact.obligation_id],
            tuple(
                json.dumps(_plain_fact(reference), sort_keys=True, separators=(",", ":"))
                for reference in fact.evidence_references
            ),
        )
        for fact in facts
    )
    if not _bindability_passes(bindability):
        rows += (_bindability_row(bindability),)
    conditional_provenance = (
        "conditional capability evidence not assembled from the five core raw inputs",
    )
    rows += tuple(
        ObligationRecord(
            obligation_id,
            ObligationResult.UNEXECUTED,
            True,
            "applicable=True means absence is not proven; capability presence is not claimed",
            (AssuranceLabel.UNVERIFIED,),
            conditional_provenance,
        )
        for obligation_id in SPECOLA_V1_CATALOGUE.definition_denominator[15:]
    )
    validate_specola_certification(AdapterCertification(
        inventory,
        manifest,
        SPECOLA_V1_CATALOGUE,
        rows,
    ))
    return rows


def _predicate_references(assessment, predicate_ids, content):
    selected = tuple(item for item in assessment.predicate_results if item.predicate_id in predicate_ids)
    if set(item.predicate_id for item in selected) != set(predicate_ids):
        raise ConformanceValidationError("structural obligation lacks its exact predicate denominator")
    if any(item.status is not PredicateStatus.SATISFIED_ON_TRACE for item in selected):
        raise ConformanceValidationError("structural obligation requires exactly satisfied evidence")
    return tuple(DerivedEvidenceReference(content, (
        "predicate=" + item.predicate_id, "call=" + item.call.kind.value,
        "episode=" + str(item.call.episode_index), "witness=" + item.witness,
    )) for item in selected)


def _producer_reference(producer):
    return "{}:{}:{}:{}:{}".format(
        producer.name, producer.configuration_id, producer.version or "",
        producer.content_sha256 or "", producer.type_source_sha256 or "",
    )


def _manifest_semantics(field):
    return (
        field.candidate_name, field.source_accessor, field.owner,
        field.address_population, field.population_lifecycle, field.value_transform,
        field.author_key, field.discovery_behavior, field.group, field.note,
        field.tags, field.unit, field.family, field.component, field.frame,
        field.approximation_relation, field.producer_provenance,
    )


def _declaration_semantics(declaration, producer):
    return (
        declaration.candidate, declaration.provenance.source_accessor, declaration.owner,
        declaration.address_population, declaration.population_lifecycle, declaration.value_transform,
        declaration.author_key, declaration.discovery_behavior, declaration.group, declaration.note,
        declaration.tags, declaration.unit, declaration.family, declaration.component, declaration.frame,
        declaration.approximation_relation, _producer_reference(producer),
    )


def _content_reference(identity, value, *coordinates):
    canonical = json.dumps(_plain_fact(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return DerivedEvidenceReference(EvidenceIdentity(identity, _sha256(canonical)), tuple(coordinates))


def _plain_fact(value):
    if is_dataclass(value):
        return {item.name: _plain_fact(getattr(value, item.name)) for item in fields(value) if item.compare}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain_fact(item) for item in value]
    return value


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_EXACT_FACT_TYPES = frozenset(
    candidate for name in (CoreConformanceArtifact.__module__, FidelityExecutionArtifact.__module__,
                           RefinementEvidence.__module__, BindabilityArtifact.__module__, __name__)
    for candidate in vars(sys.modules[name]).values() if (
        isinstance(candidate, type) and candidate.__module__ == name and is_dataclass(candidate))
)


def _exact_frozen_tree(value):
    if isinstance(value, (list, dict, set)):
        raise ConformanceValidationError("derivation inputs require immutable containers")
    if is_dataclass(value):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen or not any(type(value) is item for item in _EXACT_FACT_TYPES):
            raise ConformanceValidationError("derivation inputs require exact frozen fact types")
        children = (getattr(value, item.name) for item in fields(value) if item.compare)
    elif isinstance(value, tuple):
        children = iter(value)
    else:
        return
    for child in children:
        _exact_frozen_tree(child)


def validate_investigation(inventory: FidelityInventory) -> None:
    """Validate a total inventory partition, permitting unresolved members."""

    _validate_inventory(inventory)


def validate_inventory_closure(inventory: FidelityInventory) -> None:
    """Validate a total inventory whose candidates are all resolved."""

    unresolved_names = _validate_inventory(inventory)
    if unresolved_names:
        raise InventoryValidationError(
            "inventory has unresolved candidates: {}".format(
                _format_names(unresolved_names)
            )
        )


def validate_semantic_manifest(
    inventory: FidelityInventory,
    manifest: SemanticManifest,
) -> None:
    """Bind one semantic record to every included inventory candidate."""

    validate_inventory_closure(inventory)
    if not isinstance(manifest, SemanticManifest):
        raise InventoryValidationError(
            "certification manifest must be a SemanticManifest"
        )
    _require_non_empty_text(manifest.identity, "semantic manifest identity")
    _require_non_empty_text(manifest.version, "semantic manifest version")
    _require_non_empty_text(manifest.provenance, "semantic manifest provenance")
    _require_tuple(manifest.field_denominator, "semantic manifest field denominator")
    _require_tuple(manifest.fields, "semantic manifest fields")

    dispositions = {record.name: record.disposition for record in inventory.records}
    included = tuple(
        candidate
        for candidate in inventory.denominator.candidates
        if dispositions[candidate] is CandidateDisposition.INCLUDED
    )
    if not included:
        raise InventoryValidationError(
            "closed certification inventory must include at least one candidate"
        )
    if manifest.field_denominator != included:
        raise InventoryValidationError(
            "semantic manifest field denominator must exactly match included "
            "inventory candidates in denominator order"
        )
    for name in manifest.field_denominator:
        _require_non_empty_text(name, "semantic manifest candidate name")
    duplicates = _duplicates(manifest.field_denominator)
    if duplicates:
        raise InventoryValidationError(
            "semantic manifest field denominator has duplicates: {}".format(
                _format_names(duplicates)
            )
        )

    field_names = []
    for field_record in manifest.fields:
        _validate_semantic_field_record(field_record)
        field_names.append(field_record.candidate_name)
    if tuple(field_names) != manifest.field_denominator:
        raise InventoryValidationError(
            "semantic manifest fields must exactly cover the field denominator "
            "in order"
        )


def _validate_semantic_field_record(field_record: SemanticFieldRecord) -> None:
    if not isinstance(field_record, SemanticFieldRecord):
        raise InventoryValidationError(
            "semantic manifest fields must be SemanticFieldRecord instances"
        )
    for value, label in (
        (field_record.candidate_name, "semantic field candidate name"),
        (field_record.source_accessor, "semantic field source accessor"),
        (field_record.owner, "semantic field owner"),
        (field_record.address_population, "semantic field address population"),
        (field_record.population_lifecycle, "semantic field population lifecycle"),
        (field_record.value_transform, "semantic field value transform"),
        (field_record.unit, "semantic field unit"),
        (field_record.frame, "semantic field frame"),
        (field_record.author_key, "semantic field author key"),
        (field_record.group, "semantic field group"),
        (field_record.discovery_behavior, "semantic field discovery behavior"),
        (field_record.producer_provenance, "semantic field producer provenance"),
    ):
        _require_non_empty_text(value, label)
    if not isinstance(field_record.exactness, ExactnessKind):
        raise InventoryValidationError(
            "semantic field exactness must be an ExactnessKind"
        )
    _validate_optional_text(
        field_record.approximation_relation,
        "semantic field approximation relation",
    )
    for value, label in (
        (field_record.note, "semantic field note"),
        (field_record.family, "semantic field family"),
        (field_record.component, "semantic field component"),
    ):
        _validate_optional_text(value, label)
    if field_record.exactness is ExactnessKind.EXACT:
        if field_record.approximation_relation is not None:
            raise InventoryValidationError(
                "exact semantic field cannot declare an approximation relation"
            )
    elif field_record.approximation_relation is None:
        raise InventoryValidationError(
            "approximate semantic field requires an approximation relation"
        )
    _require_tuple(field_record.tags, "semantic field tags")
    for tag in field_record.tags:
        _require_non_empty_text(tag, "semantic field tag")
    duplicate_tags = _duplicates(field_record.tags)
    if duplicate_tags:
        raise InventoryValidationError(
            "semantic field tags must not contain duplicates: {}".format(
                _format_names(duplicate_tags)
            )
        )


def validate_catalogue(catalogue: ObligationCatalogue) -> None:
    """Validate a nonempty, finite, versioned obligation definition set."""

    if not isinstance(catalogue, ObligationCatalogue):
        raise InventoryValidationError(
            "catalogue must be an ObligationCatalogue instance"
        )
    _require_non_empty_text(catalogue.identity, "catalogue identity")
    _require_non_empty_text(catalogue.version, "catalogue version")
    _require_non_empty_text(catalogue.provenance, "catalogue provenance")
    _require_tuple(catalogue.definition_denominator, "definition denominator")
    _require_tuple(catalogue.definitions, "catalogue definitions")
    if not catalogue.definition_denominator:
        raise InventoryValidationError("definition denominator must be non-empty")
    if not catalogue.definitions:
        raise InventoryValidationError("catalogue definitions must be non-empty")

    for obligation_id in catalogue.definition_denominator:
        _require_non_empty_text(obligation_id, "definition denominator ID")
    duplicates = _duplicates(catalogue.definition_denominator)
    if duplicates:
        raise InventoryValidationError(
            "definition denominator has duplicate IDs: {}".format(
                _format_names(duplicates)
            )
        )

    definition_ids = []
    for definition in catalogue.definitions:
        _validate_definition(definition)
        definition_ids.append(definition.obligation_id)
    duplicates = _duplicates(definition_ids)
    if duplicates:
        raise InventoryValidationError(
            "catalogue definitions have duplicate IDs: {}".format(
                _format_names(duplicates)
            )
        )

    denominator_ids = set(catalogue.definition_denominator)
    definition_id_set = set(definition_ids)
    if denominator_ids != definition_id_set:
        missing = denominator_ids - definition_id_set
        invented = definition_id_set - denominator_ids
        raise InventoryValidationError(
            "definition denominator must exactly match definitions; missing: {}; "
            "invented: {}".format(_format_names(missing), _format_names(invented))
        )


def validate_certification(certification: AdapterCertification) -> None:
    """Validate a closed inventory and exact declared catalogue coverage."""

    if not isinstance(certification, AdapterCertification):
        raise InventoryValidationError(
            "certification must be an AdapterCertification instance"
        )
    validate_semantic_manifest(certification.inventory, certification.manifest)
    validate_catalogue(certification.catalogue)
    _require_tuple(certification.obligations, "certification obligations")
    if not certification.obligations:
        raise InventoryValidationError("certification obligations must be non-empty")

    records_by_id = _records_by_id(certification.obligations)
    definitions_by_id = {
        definition.obligation_id: definition
        for definition in certification.catalogue.definitions
    }
    record_ids = set(records_by_id)
    definition_ids = set(definitions_by_id)
    invented = record_ids - definition_ids
    if invented:
        raise InventoryValidationError(
            "invented obligation IDs: {}".format(_format_names(invented))
        )
    missing = definition_ids - record_ids
    if missing:
        raise InventoryValidationError(
            "missing obligation IDs: {}".format(_format_names(missing))
        )
    for obligation_id in certification.catalogue.definition_denominator:
        _validate_obligation(
            records_by_id[obligation_id], definitions_by_id[obligation_id]
        )


def validate_specola_certification(certification: AdapterCertification) -> None:
    """Validate the publication-gate record against the approved Specola v1 set."""

    if not isinstance(certification, AdapterCertification):
        raise InventoryValidationError(
            "certification must be an AdapterCertification instance"
        )
    if certification.catalogue != SPECOLA_V1_CATALOGUE:
        raise InventoryValidationError(
            "Specola v1 catalogue is required for publication certification"
        )
    validate_certification(certification)


def is_certified(certification: AdapterCertification) -> bool:
    """Return whether the Specola v1 publication record has no unmet obligation."""

    validate_specola_certification(certification)
    definitions = {
        definition.obligation_id: definition
        for definition in certification.catalogue.definitions
    }
    records = {
        record.obligation_id: record for record in certification.obligations
    }
    for obligation_id, definition in definitions.items():
        record = records[obligation_id]
        if definition.applicability is ApplicabilityKind.CORE:
            if record.result is not ObligationResult.PASS:
                return False
        elif record.applicable and record.result is not ObligationResult.PASS:
            return False
    return True


def _validate_definition(definition: ObligationDefinition) -> None:
    if not isinstance(definition, ObligationDefinition):
        raise InventoryValidationError(
            "catalogue definitions must be ObligationDefinition instances"
        )
    _require_non_empty_text(definition.obligation_id, "obligation definition ID")
    if not isinstance(definition.applicability, ApplicabilityKind):
        raise InventoryValidationError(
            "obligation definition applicability must be an ApplicabilityKind"
        )
    _validate_assurance_rule(definition.pass_evidence_rule)


def _validate_assurance_rule(rule: AssuranceRule) -> None:
    if not isinstance(rule, AssuranceRule):
        raise InventoryValidationError(
            "pass evidence rule must be an AssuranceRule instance"
        )
    _require_tuple(rule.all_of, "pass evidence rule all_of")
    _require_tuple(rule.any_of, "pass evidence rule any_of")
    if not rule.all_of and not rule.any_of:
        raise InventoryValidationError("pass evidence rule must not be empty")
    _validate_accepted_rule_labels(rule.all_of, "pass evidence rule all_of")
    _validate_accepted_rule_labels(rule.any_of, "pass evidence rule any_of")
    overlap = set(rule.all_of) & set(rule.any_of)
    if overlap:
        raise InventoryValidationError(
            "pass evidence rule all_of and any_of must not overlap: {}".format(
                _format_names(label.value for label in overlap)
            )
        )


def _validate_accepted_rule_labels(
    labels: Tuple[AssuranceLabel, ...], label: str
) -> None:
    for assurance in labels:
        if not isinstance(assurance, AssuranceLabel):
            raise InventoryValidationError("{} has an invalid assurance label".format(label))
        if assurance not in ACCEPTED_ASSURANCE_LABELS:
            raise InventoryValidationError("{} cannot require UNVERIFIED".format(label))
    duplicates = _duplicates(labels)
    if duplicates:
        raise InventoryValidationError(
            "{} has duplicate assurance labels: {}".format(
                label,
                _format_names(
                    item.value
                    for item in duplicates
                    if isinstance(item, AssuranceLabel)
                ),
            )
        )


def _records_by_id(
    obligations: Tuple[ObligationRecord, ...]
) -> Dict[str, ObligationRecord]:
    records = {}
    for obligation in obligations:
        if not isinstance(obligation, ObligationRecord):
            raise InventoryValidationError(
                "certification obligations must be ObligationRecord instances"
            )
        _require_non_empty_text(obligation.obligation_id, "obligation ID")
        if obligation.obligation_id in records:
            raise InventoryValidationError(
                "duplicate obligation IDs: {}".format(obligation.obligation_id)
            )
        records[obligation.obligation_id] = obligation
    return records


def _validate_obligation(
    obligation: ObligationRecord,
    definition: ObligationDefinition,
) -> None:
    if not isinstance(obligation.result, ObligationResult):
        raise InventoryValidationError("obligation result must be an ObligationResult")
    if not isinstance(obligation.applicable, bool):
        raise InventoryValidationError("obligation applicability must be a bool")
    _validate_optional_text(obligation.rationale, "obligation rationale")
    _validate_assurance_labels(obligation.assurance_labels)
    _require_tuple(obligation.evidence_provenance, "obligation evidence provenance")
    if not obligation.evidence_provenance:
        raise InventoryValidationError(
            "obligation evidence provenance must be non-empty"
        )
    for provenance in obligation.evidence_provenance:
        _require_non_empty_text(provenance, "obligation evidence provenance")

    if obligation.result is ObligationResult.PASS:
        _validate_pass_assurance(obligation, definition.pass_evidence_rule)
    elif not obligation.assurance_labels:
        raise InventoryValidationError(
            "non-PASS obligation requires non-empty assurance labels"
        )
    elif obligation.result is ObligationResult.UNEXECUTED:
        if obligation.assurance_labels != (AssuranceLabel.UNVERIFIED,):
            raise InventoryValidationError(
                "UNEXECUTED obligation requires exactly UNVERIFIED assurance"
            )

    if obligation.result is ObligationResult.NOT_APPLICABLE:
        if definition.applicability is ApplicabilityKind.CORE:
            raise InventoryValidationError("CORE obligation can never be NOT_APPLICABLE")
        if obligation.applicable:
            raise InventoryValidationError(
                "not_applicable obligation must declare applicable=False"
            )
        _require_non_empty_text(
            obligation.rationale,
            "not_applicable obligation rationale",
        )
        required = {
            AssuranceLabel.SOURCE_AUDITED,
            AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,
        }
        missing = required - set(obligation.assurance_labels)
        if missing:
            raise InventoryValidationError(
                "NOT_APPLICABLE requires SOURCE_AUDITED and "
                "EXHAUSTIVELY_CLASSIFIED; missing: {}".format(
                    _format_names(label.value for label in missing)
                )
            )
        return

    if not obligation.applicable:
        raise InventoryValidationError(
            "applicable=False requires a not_applicable obligation result"
        )


def _validate_pass_assurance(
    obligation: ObligationRecord,
    rule: AssuranceRule,
) -> None:
    labels = set(obligation.assurance_labels)
    if not labels:
        raise InventoryValidationError(
            "claimed PASS requires non-empty accepted assurance labels"
        )
    if AssuranceLabel.UNVERIFIED in labels:
        raise InventoryValidationError(
            "claimed PASS must not use UNVERIFIED assurance"
        )
    if not labels <= ACCEPTED_ASSURANCE_LABELS:
        raise InventoryValidationError("claimed PASS requires accepted assurance labels")
    missing = set(rule.all_of) - labels
    if missing:
        raise InventoryValidationError(
            "claimed PASS violates evidence rule all_of; missing: {}".format(
                _format_names(label.value for label in missing)
            )
        )
    if rule.any_of and not (labels & set(rule.any_of)):
        raise InventoryValidationError(
            "claimed PASS violates evidence rule any_of: {}".format(
                _format_names(label.value for label in rule.any_of)
            )
        )


def _validate_inventory(inventory: FidelityInventory) -> Tuple[str, ...]:
    if not isinstance(inventory, FidelityInventory):
        raise InventoryValidationError(
            "inventory must be a FidelityInventory instance"
        )
    if not isinstance(inventory.denominator, Denominator):
        raise InventoryValidationError("inventory denominator must be a Denominator")

    denominator = inventory.denominator
    _require_non_empty_text(denominator.identity, "denominator identity")
    _require_tuple(denominator.candidates, "denominator candidates")
    if not denominator.candidates:
        raise InventoryValidationError("denominator candidates must be non-empty")
    _require_non_empty_text(
        denominator.enumeration_provenance,
        "denominator enumeration provenance",
    )
    _require_non_empty_text(
        denominator.producer_provenance,
        "denominator producer provenance",
    )

    for candidate in denominator.candidates:
        _require_non_empty_text(candidate, "denominator candidate name")
    duplicate_denominator_names = _duplicates(denominator.candidates)
    if duplicate_denominator_names:
        raise InventoryValidationError(
            "denominator contains duplicate candidate names: {}".format(
                _format_names(duplicate_denominator_names)
            )
        )

    _require_tuple(inventory.records, "inventory records")
    record_names = []
    unresolved_names = []
    for record in inventory.records:
        _validate_candidate_record(record)
        record_names.append(record.name)
        if record.disposition is CandidateDisposition.UNRESOLVED:
            unresolved_names.append(record.name)

    duplicate_record_names = _duplicates(record_names)
    if duplicate_record_names:
        raise InventoryValidationError(
            "duplicate candidate record names create an overlapping partition: {}"
            .format(_format_names(duplicate_record_names))
        )

    denominator_names = set(denominator.candidates)
    record_name_set = set(record_names)
    outside_names = record_name_set - denominator_names
    if outside_names:
        raise InventoryValidationError(
            "candidate records outside denominator: {}".format(
                _format_names(outside_names)
            )
        )
    omitted_names = denominator_names - record_name_set
    if omitted_names:
        raise InventoryValidationError(
            "inventory omits denominator candidates: {}".format(
                _format_names(omitted_names)
            )
        )
    return tuple(unresolved_names)


def _validate_candidate_record(record: CandidateRecord) -> None:
    if not isinstance(record, CandidateRecord):
        raise InventoryValidationError(
            "inventory records must be CandidateRecord instances"
        )
    _require_non_empty_text(record.name, "candidate record name")
    _require_non_empty_text(record.rationale, "candidate record rationale")
    _require_non_empty_text(record.provenance, "candidate record provenance")
    if not isinstance(record.disposition, CandidateDisposition):
        raise InventoryValidationError(
            "candidate disposition must be a CandidateDisposition"
        )

    if record.disposition is CandidateDisposition.EXCLUDED:
        if not isinstance(record.exclusion_category, ExclusionCategory):
            raise InventoryValidationError(
                "excluded candidate requires an exclusion category"
            )
        _require_non_empty_text(record.exclusion_detail, "exclusion detail")
        return

    if record.exclusion_category is not None:
        raise InventoryValidationError(
            "only excluded candidates may declare an exclusion category"
        )
    if record.exclusion_detail is not None:
        raise InventoryValidationError(
            "only excluded candidates may declare exclusion detail"
        )


def _validate_assurance_labels(labels: Tuple[AssuranceLabel, ...]) -> None:
    _require_tuple(labels, "assurance labels")
    for label in labels:
        if not isinstance(label, AssuranceLabel):
            allowed = ", ".join(item.value for item in AssuranceLabel)
            raise InventoryValidationError(
                "assurance label must be one of: {}".format(allowed)
            )
    duplicates = _duplicates(labels)
    if duplicates:
        raise InventoryValidationError(
            "assurance labels must not contain duplicates: {}".format(
                _format_names(
                    item.value
                    for item in duplicates
                    if isinstance(item, AssuranceLabel)
                )
            )
        )
    if AssuranceLabel.UNVERIFIED in labels and len(labels) != 1:
        raise InventoryValidationError("UNVERIFIED assurance label must be exclusive")


def _require_tuple(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise InventoryValidationError("{} must be an immutable tuple".format(label))


def _require_non_empty_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InventoryValidationError("{} must be non-empty text".format(label))


def _validate_optional_text(value: Optional[str], label: str) -> None:
    if value is not None:
        _require_non_empty_text(value, label)


def _duplicates(values: Iterable[object]) -> Tuple[object, ...]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen:
            if value not in duplicates:
                duplicates.append(value)
        else:
            seen.add(value)
    return tuple(duplicates)


def _format_names(names: Iterable[object]) -> str:
    return ", ".join(sorted(str(name) for name in names))
