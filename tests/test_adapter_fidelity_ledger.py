"""Tests for the engine-neutral adapter-fidelity inventory ledger."""

import pytest

from specola.conformance.ledger import (
    AdapterCertification,
    ApplicabilityKind,
    AssuranceLabel,
    AssuranceRule,
    CandidateDisposition,
    CandidateRecord,
    Denominator,
    ExactnessKind,
    ExclusionCategory,
    FidelityInventory,
    InventoryValidationError,
    ObligationCatalogue,
    ObligationDefinition,
    ObligationRecord,
    ObligationResult,
    SemanticFieldRecord,
    SemanticManifest,
    SPECOLA_V1_CATALOGUE,
    is_certified,
    validate_certification,
    validate_investigation,
    validate_inventory_closure,
    validate_semantic_manifest,
    validate_specola_certification,
)


def _denominator(candidates=("body.xpos", "body.xquat")):
    return Denominator(
        identity="public-body-state-v1",
        candidates=candidates,
        enumeration_provenance="producer API audit 2026-08-13",
        producer_provenance="ExampleEngine 1.2",
    )


def _record(
    name,
    disposition,
    rationale="categorical decision",
    exclusion_category=None,
    exclusion_detail=None,
):
    return CandidateRecord(
        name=name,
        disposition=disposition,
        rationale=rationale,
        provenance="producer documentation section 4",
        exclusion_category=exclusion_category,
        exclusion_detail=exclusion_detail,
    )


def _inventory(records, **overrides):
    values = {"denominator": _denominator(), "records": tuple(records)}
    values.update(overrides)
    return FidelityInventory(**values)


def _closed_inventory():
    return _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record(
            "body.xquat",
            CandidateDisposition.EXCLUDED,
            exclusion_category=ExclusionCategory.DUPLICATE_ALIAS,
            exclusion_detail="aliases body.orientation with identical provenance",
        ),
    ))


def _field(candidate_name="body.xpos", **overrides):
    values = {
        "candidate_name": candidate_name,
        "source_accessor": "engine.body_xpos[address]",
        "owner": "body at manifest address",
        "address_population": "all declared bodies in the supported scene",
        "population_lifecycle": "body exists in the accepted scene",
        "value_transform": "identity float conversion",
        "unit": "m",
        "frame": "world Cartesian frame",
        "exactness": ExactnessKind.EXACT,
        "approximation_relation": None,
        "author_key": "bodypos_{axis}",
        "group": "body position",
        "tags": ("body",),
        "discovery_behavior": "closed field declared at reset",
        "producer_provenance": "ExampleEngine 1.2 documentation section 4",
    }
    values.update(overrides)
    return SemanticFieldRecord(**values)


def _manifest(fields=(_field(),), **overrides):
    values = {
        "identity": "example-body-state-manifest",
        "version": "1",
        "field_denominator": tuple(field.candidate_name for field in fields),
        "fields": tuple(fields),
        "provenance": "manifest fixture in test_adapter_fidelity_ledger.py",
    }
    values.update(overrides)
    return SemanticManifest(**values)


def _rule():
    return AssuranceRule(
        all_of=(AssuranceLabel.SOURCE_AUDITED,),
        any_of=(AssuranceLabel.ENFORCED, AssuranceLabel.PROPERTY_TESTED),
    )


def _definition(
    obligation_id="A-STRUCT-001",
    applicability=ApplicabilityKind.CORE,
    pass_evidence_rule=None,
):
    return ObligationDefinition(
        obligation_id=obligation_id,
        applicability=applicability,
        pass_evidence_rule=pass_evidence_rule or _rule(),
    )


def _catalogue(definitions=(_definition(),), **overrides):
    values = {
        "identity": "adapter-fidelity-core",
        "version": "1.0.0",
        "definition_denominator": tuple(
            definition.obligation_id for definition in definitions
        ),
        "definitions": tuple(definitions),
        "provenance": "approved certification design 2026-08-13",
    }
    values.update(overrides)
    return ObligationCatalogue(**values)


def _obligation(
    obligation_id="A-STRUCT-001",
    result=ObligationResult.PASS,
    applicable=True,
    rationale=None,
    labels=(AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.ENFORCED),
    evidence=("tests/test_adapter_contract.py",),
):
    return ObligationRecord(
        obligation_id=obligation_id,
        result=result,
        applicable=applicable,
        rationale=rationale,
        assurance_labels=labels,
        evidence_provenance=evidence,
    )


def _certification(
    inventory,
    catalogue=None,
    obligations=(_obligation(),),
    manifest=None,
):
    return AdapterCertification(
        inventory=inventory,
        manifest=manifest or _manifest(),
        catalogue=catalogue or _catalogue(),
        obligations=tuple(obligations),
    )


def _specola_v1_certification(
    failed_id=None,
    conditional_id=None,
    conditional_result=ObligationResult.PASS,
):
    obligations = []
    for definition in SPECOLA_V1_CATALOGUE.definitions:
        labels = list(definition.pass_evidence_rule.all_of)
        if definition.pass_evidence_rule.any_of:
            labels.append(definition.pass_evidence_rule.any_of[0])
        result = ObligationResult.PASS
        applicable = True
        rationale = None
        if definition.obligation_id == conditional_id:
            result = conditional_result
            if result is ObligationResult.NOT_APPLICABLE:
                applicable = False
                rationale = "audited source domain does not implement this facility"
                labels = [
                    AssuranceLabel.SOURCE_AUDITED,
                    AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,
                ]
            elif result is ObligationResult.UNEXECUTED:
                labels = [AssuranceLabel.UNVERIFIED]
        if definition.obligation_id == failed_id:
            result = ObligationResult.FAIL
            labels = [AssuranceLabel.UNVERIFIED]
        obligations.append(_obligation(
            definition.obligation_id,
            result=result,
            applicable=applicable,
            rationale=rationale,
            labels=tuple(labels),
        ))
    return AdapterCertification(
        inventory=_closed_inventory(),
        manifest=_manifest(),
        catalogue=SPECOLA_V1_CATALOGUE,
        obligations=tuple(obligations),
    )


def test_investigation_accepts_total_partition_with_unresolved_candidate():
    inventory = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record("body.xquat", CandidateDisposition.UNRESOLVED),
    ))

    validate_investigation(inventory)


def test_inventory_closure_rejects_unresolved_candidate():
    inventory = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record("body.xquat", CandidateDisposition.UNRESOLVED),
    ))

    with pytest.raises(InventoryValidationError, match="unresolved"):
        validate_inventory_closure(inventory)


def test_semantic_manifest_exactly_covers_included_inventory_candidates():
    inventory = _closed_inventory()
    manifest = _manifest()

    validate_semantic_manifest(inventory, manifest)
    validate_certification(_certification(inventory, manifest=manifest))


def test_semantic_manifest_uses_authoritative_denominator_order():
    denominator = _denominator(candidates=("field.a", "field.b"))
    inventory = _inventory(
        (
            _record("field.b", CandidateDisposition.INCLUDED),
            _record("field.a", CandidateDisposition.INCLUDED),
        ),
        denominator=denominator,
    )
    canonical = _manifest(fields=(_field("field.a"), _field("field.b")))
    reordered = _manifest(fields=(_field("field.b"), _field("field.a")))

    validate_semantic_manifest(inventory, canonical)
    with pytest.raises(InventoryValidationError, match="exactly match included"):
        validate_semantic_manifest(inventory, reordered)


@pytest.mark.parametrize(
    "manifest, message",
    [
        (_manifest(fields=()), "exactly match included"),
        (
            _manifest(
                fields=(_field("body.xquat"),),
                field_denominator=("body.xquat",),
            ),
            "exactly match included",
        ),
        (
            _manifest(
                fields=(_field(), _field()),
                field_denominator=("body.xpos", "body.xpos"),
            ),
            "exactly match included",
        ),
        (
            _manifest(
                fields=(_field(),),
                field_denominator=("body.xpos", "body.xquat"),
            ),
            "exactly match included",
        ),
    ],
)
def test_semantic_manifest_rejects_missing_extra_duplicate_or_excluded_fields(
    manifest, message
):
    with pytest.raises(InventoryValidationError, match=message):
        validate_semantic_manifest(_closed_inventory(), manifest)


def test_semantic_manifest_requires_complete_machine_readable_field_semantics():
    required_text_fields = (
        "source_accessor",
        "owner",
        "address_population",
        "population_lifecycle",
        "value_transform",
        "unit",
        "frame",
        "author_key",
        "group",
        "discovery_behavior",
        "producer_provenance",
    )
    for name in required_text_fields:
        manifest = _manifest(fields=(_field(**{name: ""}),))
        with pytest.raises(InventoryValidationError, match="non-empty text"):
            validate_semantic_manifest(_closed_inventory(), manifest)


def test_semantic_manifest_approximation_and_tag_rules_are_fail_loud():
    exact_with_relation = _manifest(fields=(_field(
        approximation_relation="within 0.01 m",
    ),))
    approximate_without_relation = _manifest(fields=(_field(
        exactness=ExactnessKind.APPROXIMATE,
    ),))
    duplicate_tags = _manifest(fields=(_field(tags=("body", "body")),))

    with pytest.raises(InventoryValidationError, match="exact.*approximation"):
        validate_semantic_manifest(_closed_inventory(), exact_with_relation)
    with pytest.raises(InventoryValidationError, match="requires an approximation"):
        validate_semantic_manifest(_closed_inventory(), approximate_without_relation)
    with pytest.raises(InventoryValidationError, match="duplicate"):
        validate_semantic_manifest(_closed_inventory(), duplicate_tags)


def test_semantic_manifest_accepts_declared_approximation():
    manifest = _manifest(fields=(_field(
        exactness=ExactnessKind.APPROXIMATE,
        approximation_relation="absolute error at most 0.01 m",
    ),))

    validate_semantic_manifest(_closed_inventory(), manifest)


def test_certification_requires_exact_nonempty_catalogue_coverage():
    catalogue = _catalogue((_definition("A-STRUCT-001"), _definition("A-SEM-001")))
    missing = _certification(
        _closed_inventory(),
        catalogue=catalogue,
        obligations=(_obligation("A-STRUCT-001"),),
    )
    invented = _certification(
        _closed_inventory(),
        catalogue=_catalogue(),
        obligations=(_obligation("A-INVENTED-001"),),
    )

    with pytest.raises(InventoryValidationError, match="missing obligation"):
        validate_certification(missing)
    with pytest.raises(InventoryValidationError, match="invented obligation"):
        validate_certification(invented)


def test_catalogue_rejects_empty_and_mismatched_definition_denominators():
    empty = _catalogue((), definition_denominator=())
    mismatch = _catalogue(definition_denominator=("A-OTHER-001",))

    for catalogue in (empty, mismatch):
        certification = _certification(_closed_inventory(), catalogue=catalogue)
        with pytest.raises(InventoryValidationError, match="definition denominator"):
            validate_certification(certification)


def test_specola_publication_validation_rejects_caller_invented_catalogue():
    certification = _certification(_closed_inventory())

    validate_certification(certification)
    with pytest.raises(InventoryValidationError, match="Specola v1 catalogue"):
        validate_specola_certification(certification)


def test_authoritative_v1_catalogue_covers_approved_core_and_conditional_families():
    identifiers = set(SPECOLA_V1_CATALOGUE.definition_denominator)

    assert {
        "CORE.LIFECYCLE",
        "CORE.OBSERVATION_STRUCTURE",
        "CORE.TIME_INDEX",
        "CORE.IDENTITY_CLASS",
        "CORE.VALUE_DOMAIN",
        "CORE.SUPPORTED_DOMAIN",
        "CORE.SEMANTIC_MANIFEST",
        "CORE.INVENTORY_CLOSURE",
        "CORE.DISCOVERY_MATERIALIZATION",
        "CORE.CRITERION_INDEPENDENCE",
        "CORE.SCOPE_PROJECTION",
        "CORE.PRODUCER_PROVENANCE",
        "CORE.APPROXIMATION_DISCLOSURE",
        "CORE.NO_SILENT_ACCEPTANCE",
        "CORE.STATIC_SLOT_BINDABILITY",
        "CAP.RIGID_BODY",
        "CAP.ARTICULATION",
        "CAP.DEFORMABLE_STATE",
        "CAP.DEFORMABLE_TOPOLOGY",
        "CAP.CONTACT",
        "CAP.PHYSICAL_SENSOR",
        "FAC.CAMERA_CAPTURE",
        "FAC.RENDERING",
        "FAC.SCENE_GEOMETRY",
        "FAC.SCENE_POSES",
        "FAC.AUTO_WRAP",
    } == identifiers


def test_authoritative_v1_catalogue_has_exact_expected_evidence_rules():
    runtime = (
        (AssuranceLabel.SOURCE_AUDITED,),
        (
            AssuranceLabel.ENFORCED,
            AssuranceLabel.PROPERTY_TESTED,
            AssuranceLabel.DIFFERENTIALLY_CHECKED,
        ),
    )
    expected = {
        "CORE.LIFECYCLE": (ApplicabilityKind.CORE, runtime),
        "CORE.OBSERVATION_STRUCTURE": (ApplicabilityKind.CORE, runtime),
        "CORE.TIME_INDEX": (ApplicabilityKind.CORE, runtime),
        "CORE.IDENTITY_CLASS": (ApplicabilityKind.CORE, runtime),
        "CORE.VALUE_DOMAIN": (ApplicabilityKind.CORE, runtime),
        "CORE.SUPPORTED_DOMAIN": (
            ApplicabilityKind.CORE,
            ((AssuranceLabel.EXHAUSTIVELY_CLASSIFIED, AssuranceLabel.SOURCE_AUDITED), ()),
        ),
        "CORE.SEMANTIC_MANIFEST": (ApplicabilityKind.CORE, runtime),
        "CORE.INVENTORY_CLOSURE": (
            ApplicabilityKind.CORE,
            ((AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,), ()),
        ),
        "CORE.DISCOVERY_MATERIALIZATION": (ApplicabilityKind.CORE, runtime),
        "CORE.CRITERION_INDEPENDENCE": (ApplicabilityKind.CORE, runtime),
        "CORE.SCOPE_PROJECTION": (ApplicabilityKind.CORE, runtime),
        "CORE.PRODUCER_PROVENANCE": (
            ApplicabilityKind.CORE,
            ((AssuranceLabel.SOURCE_AUDITED,), ()),
        ),
        "CORE.APPROXIMATION_DISCLOSURE": (ApplicabilityKind.CORE, runtime),
        "CORE.NO_SILENT_ACCEPTANCE": (ApplicabilityKind.CORE, runtime),
        "CORE.STATIC_SLOT_BINDABILITY": (ApplicabilityKind.CORE, runtime),
        "CAP.RIGID_BODY": (ApplicabilityKind.CONDITIONAL, runtime),
        "CAP.ARTICULATION": (ApplicabilityKind.CONDITIONAL, runtime),
        "CAP.DEFORMABLE_STATE": (ApplicabilityKind.CONDITIONAL, runtime),
        "CAP.DEFORMABLE_TOPOLOGY": (ApplicabilityKind.CONDITIONAL, runtime),
        "CAP.CONTACT": (ApplicabilityKind.CONDITIONAL, runtime),
        "CAP.PHYSICAL_SENSOR": (ApplicabilityKind.CONDITIONAL, runtime),
        "FAC.CAMERA_CAPTURE": (ApplicabilityKind.CONDITIONAL, runtime),
        "FAC.RENDERING": (ApplicabilityKind.CONDITIONAL, runtime),
        "FAC.SCENE_GEOMETRY": (ApplicabilityKind.CONDITIONAL, runtime),
        "FAC.SCENE_POSES": (ApplicabilityKind.CONDITIONAL, runtime),
        "FAC.AUTO_WRAP": (ApplicabilityKind.CONDITIONAL, runtime),
    }

    observed = {
        definition.obligation_id: (
            definition.applicability,
            (
                definition.pass_evidence_rule.all_of,
                definition.pass_evidence_rule.any_of,
            ),
        )
        for definition in SPECOLA_V1_CATALOGUE.definitions
    }
    assert observed == expected

def test_source_audited_alone_cannot_pass_runtime_obligation():
    certification = _certification(
        _closed_inventory(),
        obligations=(_obligation(labels=(AssuranceLabel.SOURCE_AUDITED,)),),
    )

    with pytest.raises(InventoryValidationError, match="evidence rule"):
        validate_certification(certification)


def test_pass_evidence_rule_accepts_declared_all_of_and_any_of_evidence():
    certification = _certification(
        _closed_inventory(),
        obligations=(_obligation(labels=(
            AssuranceLabel.SOURCE_AUDITED,
            AssuranceLabel.PROPERTY_TESTED,
        )),),
    )

    validate_certification(certification)


def test_not_applicable_is_conditional_only_and_requires_audited_classification():
    conditional = _definition(
        "A-CONTACT-001",
        applicability=ApplicabilityKind.CONDITIONAL,
    )
    catalogue = _catalogue((conditional,))
    valid = _certification(
        _closed_inventory(),
        catalogue=catalogue,
        obligations=(_obligation(
            "A-CONTACT-001",
            result=ObligationResult.NOT_APPLICABLE,
            applicable=False,
            rationale="accepted source domain has no contact representation",
            labels=(
                AssuranceLabel.SOURCE_AUDITED,
                AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,
            ),
        ),),
    )
    core_na = _certification(_closed_inventory(), obligations=(_obligation(
        result=ObligationResult.NOT_APPLICABLE,
        applicable=False,
        rationale="not relevant",
        labels=(
            AssuranceLabel.SOURCE_AUDITED,
            AssuranceLabel.EXHAUSTIVELY_CLASSIFIED,
        ),
    ),))
    weak_na = _certification(
        _closed_inventory(),
        catalogue=catalogue,
        obligations=(_obligation(
            "A-CONTACT-001",
            result=ObligationResult.NOT_APPLICABLE,
            applicable=False,
            rationale="not present",
            labels=(AssuranceLabel.SOURCE_AUDITED,),
        ),),
    )

    validate_certification(valid)
    with pytest.raises(InventoryValidationError, match="CORE"):
        validate_certification(core_na)
    with pytest.raises(InventoryValidationError, match="EXHAUSTIVELY_CLASSIFIED"):
        validate_certification(weak_na)


def test_is_certified_requires_passing_core_and_applicable_conditional_rows():
    passing = _specola_v1_certification()
    failing = _specola_v1_certification(failed_id="CORE.LIFECYCLE")

    assert is_certified(passing) is True
    assert is_certified(failing) is False


@pytest.mark.parametrize(
    "result, expected",
    [
        (ObligationResult.PASS, True),
        (ObligationResult.FAIL, False),
        (ObligationResult.UNEXECUTED, False),
        (ObligationResult.NOT_APPLICABLE, True),
    ],
)
def test_is_certified_covers_applicable_and_not_applicable_conditional_branches(
    result, expected
):
    certification = _specola_v1_certification(
        conditional_id="CAP.CONTACT",
        conditional_result=result,
    )

    assert is_certified(certification) is expected


@pytest.mark.parametrize(
    "records, message",
    [
        (
            (
                _record("body.xpos", CandidateDisposition.INCLUDED),
                _record(
                    "body.xpos",
                    CandidateDisposition.EXCLUDED,
                    exclusion_category=ExclusionCategory.DUPLICATE_ALIAS,
                    exclusion_detail="duplicate of a declared candidate",
                ),
                _record("body.xquat", CandidateDisposition.INCLUDED),
            ),
            "duplicate candidate record",
        ),
        (
            (_record("body.xpos", CandidateDisposition.INCLUDED),),
            "omits denominator candidates",
        ),
        (
            (
                _record("body.xpos", CandidateDisposition.INCLUDED),
                _record("body.xquat", CandidateDisposition.INCLUDED),
                _record(
                    "joint.qpos",
                    CandidateDisposition.EXCLUDED,
                    exclusion_category=ExclusionCategory.DUPLICATE_ALIAS,
                    exclusion_detail="duplicate alias",
                ),
            ),
            "outside denominator",
        ),
    ],
)
def test_partition_rejects_duplicate_omitted_and_outside_candidates(records, message):
    with pytest.raises(InventoryValidationError, match=message):
        validate_investigation(_inventory(records))


def test_validation_rejects_empty_and_duplicate_denominators_for_all_modes():
    duplicate = _inventory((), denominator=_denominator(("body.xpos", "body.xpos")))
    empty = _inventory((), denominator=_denominator(()))

    for inventory in (duplicate, empty):
        for validator in (validate_investigation, validate_inventory_closure):
            with pytest.raises(InventoryValidationError):
                validator(inventory)


def test_body_xquat_cannot_be_waived_by_removed_broad_categories():
    inventory = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record(
            "body.xquat",
            CandidateDisposition.EXCLUDED,
            exclusion_category="engine_internal",
            exclusion_detail="broad prose would make it waivable",
        ),
    ))

    with pytest.raises(InventoryValidationError, match="exclusion category"):
        validate_investigation(inventory)


@pytest.mark.parametrize("category", ("not_currently_needed", "unvalidated_version"))
def test_arbitrary_and_unvalidated_version_exclusions_remain_unresolved(category):
    inventory = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record(
            "body.xquat",
            CandidateDisposition.EXCLUDED,
            exclusion_category=category,
            exclusion_detail="prose cannot create an exclusion category",
        ),
    ))

    with pytest.raises(InventoryValidationError, match="exclusion category"):
        validate_investigation(inventory)


def test_unvalidated_version_is_recorded_as_unresolved_not_excluded():
    inventory = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record(
            "body.xquat",
            CandidateDisposition.UNRESOLVED,
            rationale="producer version has not been audited",
        ),
    ))

    validate_investigation(inventory)
    with pytest.raises(InventoryValidationError, match="unresolved"):
        validate_inventory_closure(inventory)


def test_every_exclusion_requires_concrete_detail_and_provenance():
    missing_detail = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        _record(
            "body.xquat",
            CandidateDisposition.EXCLUDED,
            exclusion_category=ExclusionCategory.DUPLICATE_ALIAS,
        ),
    ))
    missing_provenance = _inventory((
        _record("body.xpos", CandidateDisposition.INCLUDED),
        CandidateRecord(
            name="body.xquat",
            disposition=CandidateDisposition.EXCLUDED,
            rationale="duplicate alias",
            provenance="",
            exclusion_category=ExclusionCategory.DUPLICATE_ALIAS,
            exclusion_detail="same producer accessor as body.orientation",
        ),
    ))

    with pytest.raises(InventoryValidationError, match="exclusion detail"):
        validate_investigation(missing_detail)
    with pytest.raises(InventoryValidationError, match="provenance"):
        validate_investigation(missing_provenance)


def test_unverified_is_exclusive_and_pass_rejects_it_or_empty_assurance():
    contradictory = _certification(_closed_inventory(), obligations=(_obligation(
        labels=(AssuranceLabel.UNVERIFIED, AssuranceLabel.SOURCE_AUDITED),
    ),))
    unverified_pass = _certification(_closed_inventory(), obligations=(_obligation(
        labels=(AssuranceLabel.UNVERIFIED,),
    ),))
    empty_pass = _certification(_closed_inventory(), obligations=(_obligation(labels=()),))

    with pytest.raises(InventoryValidationError, match="exclusive"):
        validate_certification(contradictory)
    with pytest.raises(InventoryValidationError, match="must not use UNVERIFIED"):
        validate_certification(unverified_pass)
    with pytest.raises(InventoryValidationError, match="non-empty accepted assurance"):
        validate_certification(empty_pass)


@pytest.mark.parametrize(
    "result, labels, message",
    [
        (ObligationResult.UNEXECUTED, (), "non-empty assurance"),
        (ObligationResult.UNEXECUTED, (AssuranceLabel.SOURCE_AUDITED,), "UNEXECUTED"),
        (ObligationResult.FAIL, (), "non-empty assurance"),
        (
            ObligationResult.FAIL,
            (AssuranceLabel.UNVERIFIED, AssuranceLabel.SOURCE_AUDITED),
            "exclusive",
        ),
    ],
)
def test_nonpassing_results_cannot_use_empty_or_contradictory_assurance(
    result, labels, message
):
    certification = _certification(
        _closed_inventory(),
        obligations=(_obligation(result=result, labels=labels),),
    )

    with pytest.raises(InventoryValidationError, match=message):
        validate_certification(certification)


def test_fail_accepts_concrete_or_exclusively_unverified_assurance():
    concrete = _certification(
        _closed_inventory(),
        obligations=(_obligation(
            result=ObligationResult.FAIL,
            labels=(AssuranceLabel.SOURCE_AUDITED,),
        ),),
    )
    unverified = _certification(
        _closed_inventory(),
        obligations=(_obligation(
            result=ObligationResult.FAIL,
            labels=(AssuranceLabel.UNVERIFIED,),
        ),),
    )

    validate_certification(concrete)
    validate_certification(unverified)


def test_duplicate_obligation_ids_and_missing_evidence_are_rejected():
    definition = _definition()
    duplicate = _certification(
        _closed_inventory(),
        catalogue=_catalogue((definition,)),
        obligations=(_obligation(), _obligation()),
    )
    missing_evidence = _certification(
        _closed_inventory(),
        obligations=(_obligation(evidence=()),),
    )

    with pytest.raises(InventoryValidationError, match="duplicate obligation"):
        validate_certification(duplicate)
    with pytest.raises(InventoryValidationError, match="evidence provenance"):
        validate_certification(missing_evidence)
