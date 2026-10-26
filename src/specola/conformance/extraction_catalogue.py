"""Required extraction obligations using the existing conformance vocabulary.

This catalogue does not modify historical catalogue 1.1 or grant a result from
the presence of test files. Each enrolled row needs executed, identity-bound
evidence under the extraction programme before it can pass its release gate.
"""

from .ledger import (
    ApplicabilityKind, AssuranceLabel, AssuranceRule, ObligationCatalogue,
    ObligationDefinition,
)


def _catalogue() -> ObligationCatalogue:
    structural = AssuranceRule(
        all_of=(AssuranceLabel.ENFORCED, AssuranceLabel.SOURCE_AUDITED),
        any_of=(),
    )
    runtime = AssuranceRule(
        all_of=(AssuranceLabel.ENFORCED,),
        any_of=(AssuranceLabel.PROPERTY_TESTED, AssuranceLabel.DIFFERENTIALLY_CHECKED),
    )
    fidelity = AssuranceRule(
        all_of=(AssuranceLabel.SOURCE_AUDITED, AssuranceLabel.DIFFERENTIALLY_CHECKED),
        any_of=(),
    )
    closure = AssuranceRule(
        all_of=(AssuranceLabel.ENFORCED, AssuranceLabel.SOURCE_AUDITED,
                AssuranceLabel.EXHAUSTIVELY_CLASSIFIED),
        any_of=(),
    )
    definitions = tuple(
        ObligationDefinition("EXTRACTION." + name, ApplicabilityKind.CORE, rule)
        for name, rule in (
            ("REGISTRY_COVERAGE", structural),
            ("PRODUCER_SURFACE", structural),
            ("CLASSIFICATION_CLOSURE", closure),
            ("SEMANTIC_RECORDS", structural),
            ("MATERIALIZATION_COVERAGE", runtime),
            ("QUERY_CONTRACT", runtime),
            ("NATIVE_FIDELITY", fidelity),
            ("LIFECYCLE_FRESHNESS", runtime),
            ("OBSERVATION_EFFECTS", runtime),
            ("IDENTITY_AND_METADATA", runtime),
            ("DISCOVERY_BINDABILITY", runtime),
            ("SCOPE_PROJECTION", runtime),
            ("CRITERION_INDEPENDENCE", runtime),
            ("STATIC_AND_REPLAY", runtime),
            ("VERSION_AND_EVIDENCE", structural),
        )
    )
    return ObligationCatalogue(
        identity="specola-adapter-extraction", version="1.0",
        definition_denominator=tuple(definition.obligation_id for definition in definitions),
        definitions=definitions,
        provenance="accepted read-surface completeness design 2026-09-04; eight live adapter programme",
    )


SPECOLA_EXTRACTION_V1_CATALOGUE = _catalogue()
