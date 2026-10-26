"""Extraction obligations do not reinterpret the historical certification."""

from specola.conformance.extraction_catalogue import SPECOLA_EXTRACTION_V1_CATALOGUE
from specola.conformance.ledger import ApplicabilityKind, ObligationCatalogue, SPECOLA_V1_CATALOGUE


def test_extraction_reuses_catalogue_types_without_changing_legacy_ids():
    catalogue = SPECOLA_EXTRACTION_V1_CATALOGUE
    assert type(catalogue) is ObligationCatalogue
    assert catalogue.identity != SPECOLA_V1_CATALOGUE.identity
    assert SPECOLA_V1_CATALOGUE.version == "1.1"
    assert all(item.applicability is ApplicabilityKind.CORE for item in catalogue.definitions)
    assert not set(catalogue.definition_denominator) & set(SPECOLA_V1_CATALOGUE.definition_denominator)
    assert catalogue.definition_denominator == tuple(item.obligation_id for item in catalogue.definitions)
    assert len(set(catalogue.definition_denominator)) == len(catalogue.definitions)


def test_accepted_extraction_obligations_are_not_omitted():
    # Specola's normative test obligations are intentional constants, unlike
    # producer member names and task populations, which must be generated.
    assert {name[len("EXTRACTION."):] for name in SPECOLA_EXTRACTION_V1_CATALOGUE.definition_denominator} == {
        "REGISTRY_COVERAGE", "PRODUCER_SURFACE", "CLASSIFICATION_CLOSURE", "SEMANTIC_RECORDS",
        "MATERIALIZATION_COVERAGE", "QUERY_CONTRACT", "NATIVE_FIDELITY", "LIFECYCLE_FRESHNESS",
        "OBSERVATION_EFFECTS", "IDENTITY_AND_METADATA", "DISCOVERY_BINDABILITY", "SCOPE_PROJECTION",
        "CRITERION_INDEPENDENCE", "STATIC_AND_REPLAY", "VERSION_AND_EVIDENCE",
    }
