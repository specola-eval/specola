"""API/result coverage is a separate join from quantity semantics."""

from dataclasses import replace

import pytest

from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.extraction import (
    CoverageLink, ExtractionManifest, ExtractionValidationError,
    NonObservationReason, QuantityOutcome, QuantityRecord, ResultDeclaration, ResultSurface,
    SourceAddress, SurfaceClassification, SurfaceDisposition,
    validate_extraction_manifest,
)
from specola.conformance.ledger import ExactnessKind
from specola.conformance.query_bindings import ArgumentSchema
from specola.conformance.surface import (
    ApiMember, MemberKind, ProducerIdentity, ProducerSurface, SurfaceRoot,
    SurfaceValidationError, surface_digest,
)


def _evidence():
    return (EvidenceIdentity("native source", "a" * 64),)


def _fixture():
    evidence = _evidence()[0]
    surface = ProducerSurface(
        ProducerIdentity("producer", "1", "native", evidence), evidence, evidence,
        (SurfaceRoot("native", evidence, "independent native reference index"),),
        (ApiMember("native", "query", MemberKind.CALLABLE, "query(pair) -> distance, witness"),),
    )
    root = SourceAddress("native.query")
    distance = SourceAddress("native.query", "return.distance")
    witness = SourceAddress("native.query", "return.witness")
    results = ResultSurface(surface_digest(surface), (
        ResultDeclaration(distance, "float64 scalar", "supported collider pair", _evidence()),
        ResultDeclaration(witness, "float64[2,3]", "supported collider pair", _evidence()),
    ), _evidence())
    quantity = QuantityRecord(
        "native_distance", (distance,), "native signed collider distance", "native_pair_ids",
        "pair domain", "m", "world", ExactnessKind.EXACT, "supported collider pair",
        "current native state", "identity", "properties", "distance_{pair}",
        "distance_metadata", "distance_query_reader", _evidence(), "geom_pair",
        (QuantityOutcome("supported", "native collider support", ArgumentSchema.from_document({
            "type": "object", "properties": {"kind": {"const": "float64"}, "data": {"type": "string"}},
            "required": ["kind", "data"], "additionalProperties": False,
        }), _evidence()),),
    )
    witness_quantity = replace(quantity, quantity_id="native_witness", sources=(witness,),
                               native_meaning="native closest points", property_pattern="witness_{pair}_{component}")
    classifications = tuple(SurfaceClassification(address, disposition, "native declaration mapping", _evidence()) for address, disposition in (
        (root, SurfaceDisposition.REPRESENTED_ELSEWHERE),
        (distance, SurfaceDisposition.EXPOSED),
        (witness, SurfaceDisposition.EXPOSED),
    ))
    manifest = ExtractionManifest("native extraction", surface_digest(surface), classifications,
                                  (quantity, witness_quantity), (
        CoverageLink(root, "query returns both declared results", _evidence(), target_source=distance),
        CoverageLink(root, "query returns both declared results", _evidence(), target_source=witness),
        CoverageLink(distance, "identity", _evidence(), target_quantity="native_distance"),
        CoverageLink(witness, "preserve both xyz triples", _evidence(), target_quantity="native_witness"),
    ))
    return surface, results, manifest


def test_unparsed_producer_root_blocks_closure_without_erasing_its_population():
    import json
    from specola.conformance.surface import surface_document, surface_from_document

    surface, results, manifest = _fixture()
    root = SurfaceRoot("unparsed_header", _evidence()[0], "installed producer header declaration",
                       unresolved_reason="compiler diagnostics prevent a complete declaration capture")
    surface = replace(surface, roots=surface.roots + (root,))
    digest = surface_digest(surface)
    results = replace(results, producer_surface_digest=digest)
    manifest = replace(manifest, producer_surface_digest=digest)
    restored = surface_from_document(json.loads(json.dumps(surface_document(surface))))
    assert restored == surface
    validate_extraction_manifest(restored, results, manifest, require_closed=False)
    with pytest.raises(ExtractionValidationError, match="unresolved producer root: unparsed_header"):
        validate_extraction_manifest(restored, results, manifest)


@pytest.mark.parametrize("empty, populated", ((True, False), (False, True)))
def test_unresolved_root_cannot_claim_empty_or_parsed_members(empty, populated):
    surface, _, _ = _fixture()
    root = SurfaceRoot("unparsed_header", _evidence()[0], "installed producer header declaration",
                       empty_reason="no declarations" if empty else "",
                       unresolved_reason="native header could not be parsed")
    members = surface.members + ((ApiMember("unparsed_header", "field", MemberKind.FIELD, "int field"),) if populated else ())
    with pytest.raises(SurfaceValidationError, match="unresolved root"):
        surface_digest(replace(surface, roots=surface.roots + (root,), members=members))


def test_stored_relations_restore_exact_semantics_before_validation():
    import json
    from specola.conformance import extraction

    surface, results, manifest = _fixture()
    result_document = json.loads(json.dumps(extraction.result_surface_document(results)))
    manifest_document = json.loads(json.dumps(extraction.manifest_document(manifest)))
    restored_results = extraction.result_surface_from_document(surface, result_document)
    restored_manifest = extraction.manifest_from_document(surface, restored_results, manifest_document)
    assert restored_results == results
    assert restored_manifest == manifest
    assert extraction.manifest_digest(restored_manifest) == extraction.manifest_digest(manifest)


def test_stored_unresolved_relation_requires_explicit_draft_restoration():
    import json
    from specola.conformance import extraction

    surface, results, manifest = _fixture()
    draft = replace(manifest, quantities=(), coverage=(), classifications=tuple(
        replace(row, disposition=SurfaceDisposition.UNRESOLVED)
        for row in manifest.classifications
    ))
    document = json.loads(json.dumps(extraction.manifest_document(draft)))
    with pytest.raises(ExtractionValidationError, match="unresolved"):
        extraction.manifest_from_document(surface, results, document)
    assert extraction.manifest_from_document(surface, results, document, require_closed=False) == draft


def test_stored_manifest_rejects_each_dropped_or_added_record_field():
    import copy
    import json
    from specola.conformance import extraction

    surface, results, manifest = _fixture()
    document = json.loads(json.dumps(extraction.manifest_document(manifest)))

    def paths(value, prefix=()):
        if isinstance(value, dict):
            yield prefix
            for key, item in value.items():
                yield from paths(item, prefix + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from paths(item, prefix + (index,))

    def at(value, path):
        for component in path:
            value = value[component]
        return value

    for path in paths(document):
        for key in at(document, path):
            changed = copy.deepcopy(document)
            del at(changed, path)[key]
            with pytest.raises(ExtractionValidationError):
                extraction.manifest_from_document(surface, results, changed)
        changed = copy.deepcopy(document)
        at(changed, path)["unused_new_field"] = "must not be silently discarded"
        with pytest.raises(ExtractionValidationError):
            extraction.manifest_from_document(surface, results, changed)


def test_stored_result_surface_retains_omitted_component_detection():
    import json
    from specola.conformance import extraction

    surface, results, manifest = _fixture()
    document = json.loads(json.dumps(extraction.result_surface_document(results)))
    document["results"].pop()
    restored = extraction.result_surface_from_document(surface, document)
    with pytest.raises(ExtractionValidationError, match="classification population"):
        validate_extraction_manifest(surface, restored, manifest)


def test_many_results_and_quantities_close_without_equating_names():
    validate_extraction_manifest(*_fixture())


def test_missing_result_component_cannot_be_hidden_by_one_exposed_output():
    surface, results, manifest = _fixture()
    with pytest.raises(ExtractionValidationError, match="classification population"):
        validate_extraction_manifest(surface, results, replace(manifest, classifications=manifest.classifications[:-1]))


def test_missing_unused_api_is_detected_even_when_surface_pins_are_updated():
    surface, results, manifest = _fixture()
    surface = replace(surface, members=surface.members + (ApiMember("native", "unused_query", MemberKind.CALLABLE, "new_query()"),))
    results = replace(results, producer_surface_digest=surface_digest(surface))
    manifest = replace(manifest, producer_surface_digest=surface_digest(surface))
    with pytest.raises(ExtractionValidationError, match="classification population"):
        validate_extraction_manifest(surface, results, manifest)


def test_unresolved_is_a_draft_not_a_closed_manifest():
    surface, results, manifest = _fixture()
    classifications = tuple(replace(item, disposition=SurfaceDisposition.UNRESOLVED) for item in manifest.classifications)
    draft = replace(manifest, classifications=classifications, quantities=(), coverage=())
    validate_extraction_manifest(surface, results, draft, require_closed=False)
    with pytest.raises(ExtractionValidationError, match="unresolved"):
        validate_extraction_manifest(surface, results, draft)


def test_non_observation_reasons_are_closed_and_do_not_hide_results():
    surface, results, manifest = _fixture()
    operation = replace(manifest.classifications[0], disposition=SurfaceDisposition.NOT_OBSERVATION,
                        non_observation_reason=NonObservationReason.LIFECYCLE_OR_CONTROL_OPERATION)
    # The operation itself can be non-observational while independently
    # readable state-bearing results still require their own coverage.
    updated = replace(manifest, classifications=(operation,) + manifest.classifications[1:], coverage=manifest.coverage[2:])
    validate_extraction_manifest(surface, results, updated)
    with pytest.raises(ExtractionValidationError, match="accepted reason"):
        validate_extraction_manifest(surface, results, replace(updated, classifications=(replace(operation, non_observation_reason="too expensive"),) + updated.classifications[1:]))
    with pytest.raises(ExtractionValidationError, match="classification population"):
        validate_extraction_manifest(surface, results, replace(updated, classifications=(operation,)))


@pytest.mark.parametrize("field", ("owner_kind", "owner_resolver_id", "index_domain", "unit", "frame", "native_meaning", "availability", "freshness", "output_sink", "property_pattern", "metadata_resolver_id", "reader_id"))
def test_each_quantity_semantic_and_execution_address_is_required(field):
    surface, results, manifest = _fixture()
    quantity = replace(manifest.quantities[0], **{field: ""})
    with pytest.raises(ExtractionValidationError):
        validate_extraction_manifest(surface, results, replace(manifest, quantities=(quantity,) + manifest.quantities[1:]))


def test_approximation_is_an_addition_not_a_native_coverage_replacement():
    surface, results, manifest = _fixture()
    approximate = replace(manifest.quantities[0], quantity_id="approximation", exactness=ExactnessKind.APPROXIMATE,
                          approximation_relation="declared approximation relation")
    validate_extraction_manifest(surface, results, replace(manifest, quantities=manifest.quantities + (approximate,)))
    links = tuple(replace(link, target_quantity="approximation") if link.target_quantity == "native_distance" else link for link in manifest.coverage)
    with pytest.raises(ExtractionValidationError, match="approximation cannot discharge"):
        validate_extraction_manifest(surface, results, replace(manifest, quantities=(approximate,) + manifest.quantities[1:], coverage=links))


@pytest.mark.parametrize("change, message", (
    (lambda m: replace(m, classifications=m.classifications + m.classifications[:1]), "duplicates"),
    (lambda m: replace(m, coverage=m.coverage + m.coverage[:1]), "duplicate coverage"),
    (lambda m: replace(m, quantities=m.quantities + m.quantities[:1]), "duplicate quantity"),
    (lambda m: replace(m, coverage=m.coverage[:-1]), "no coverage mapping"),
    (lambda m: replace(m, coverage=(replace(m.coverage[0], target_source=SourceAddress("missing")),) + m.coverage[1:]), "dangling"),
    (lambda m: replace(m, coverage=(replace(m.coverage[0], target_quantity="native_distance"),) + m.coverage[1:]), "exactly one target"),
))
def test_inconsistent_coverage_joins_fail(change, message):
    surface, results, manifest = _fixture()
    with pytest.raises(ExtractionValidationError, match=message):
        validate_extraction_manifest(surface, results, change(manifest))


def test_cycle_is_rejected_even_with_an_extra_path_to_a_quantity():
    surface, results, manifest = _fixture()
    cycle = CoverageLink(manifest.classifications[1].source, "cyclic alias", _evidence(), target_source=manifest.classifications[0].source)
    with pytest.raises(ExtractionValidationError, match="cycle"):
        validate_extraction_manifest(surface, results, replace(manifest, coverage=manifest.coverage + (cycle,)))


def test_result_census_and_manifest_must_match_current_producer():
    surface, results, manifest = _fixture()
    with pytest.raises(ExtractionValidationError, match="result census producer"):
        validate_extraction_manifest(surface, replace(results, producer_surface_digest="stale"), manifest)
    with pytest.raises(ExtractionValidationError, match="manifest producer"):
        validate_extraction_manifest(surface, results, replace(manifest, producer_surface_digest="stale"))


def test_unknown_result_and_missing_provenance_fail():
    surface, results, manifest = _fixture()
    wrong = replace(results.results[0], source=SourceAddress("native.absent", "return"))
    with pytest.raises(ExtractionValidationError, match="known member"):
        validate_extraction_manifest(surface, replace(results, results=(wrong,) + results.results[1:]), manifest)
    with pytest.raises(ExtractionValidationError, match="nonempty evidence"):
        validate_extraction_manifest(surface, replace(results, census_provenance=()), manifest)


def test_quantity_cannot_claim_a_source_mapped_only_to_another_output():
    surface, results, manifest = _fixture()
    distance, witness = manifest.quantities
    distance = replace(distance, sources=distance.sources + witness.sources)
    with pytest.raises(ExtractionValidationError, match="no coverage path"):
        validate_extraction_manifest(surface, results, replace(manifest, quantities=(distance, witness)))


def test_quantity_source_may_have_a_transitive_lossless_mapping():
    surface, results, manifest = _fixture()
    distance, witness = manifest.quantities
    distance = replace(distance, sources=distance.sources + (manifest.classifications[0].source,))
    validate_extraction_manifest(surface, results, replace(manifest, quantities=(distance, witness)))


def test_invalid_source_field_does_not_escape_as_an_unhashable_type_error():
    surface, results, manifest = _fixture()
    distance = replace(manifest.quantities[0], sources=(SourceAddress([], "return"),))  # pyright: ignore[reportArgumentType] - malformed-input negative control
    with pytest.raises(ExtractionValidationError, match="source member ID"):
        validate_extraction_manifest(surface, results, replace(manifest, quantities=(distance,) + manifest.quantities[1:]))


def test_alias_depth_has_no_recursive_python_stack_limit():
    surface, results, manifest = _fixture()
    addresses = tuple(SourceAddress("native.query", "alias." + str(index).zfill(5)) for index in range(1100))
    results = replace(results, results=tuple(sorted(results.results + tuple(
        ResultDeclaration(address, "float64", "fixture", _evidence()) for address in addresses
    ), key=lambda record: record.source)))
    classifications = tuple(SurfaceClassification(address, SurfaceDisposition.REPRESENTED_ELSEWHERE,
                                                   "lossless alias", _evidence()) for address in addresses)
    links = tuple(CoverageLink(address, "identity alias", _evidence(), target_source=target)
                  for address, target in zip(addresses, addresses[1:] + (manifest.quantities[0].sources[0],)))
    validate_extraction_manifest(surface, results, replace(manifest,
        classifications=manifest.classifications + classifications, coverage=manifest.coverage + links))


def test_exposed_source_needs_its_own_direct_output_mapping():
    surface, results, manifest = _fixture()
    root = replace(manifest.classifications[0], disposition=SurfaceDisposition.EXPOSED)
    with pytest.raises(ExtractionValidationError, match="direct quantity mapping"):
        validate_extraction_manifest(surface, results, replace(manifest, classifications=(root,) + manifest.classifications[1:]))


def test_represented_source_cannot_be_a_dead_end():
    surface, results, manifest = _fixture()
    with pytest.raises(ExtractionValidationError, match="no path to an exact"):
        validate_extraction_manifest(surface, results, replace(manifest, coverage=manifest.coverage[2:]))


@pytest.mark.parametrize("disposition", (SurfaceDisposition.NOT_OBSERVATION, SurfaceDisposition.UNRESOLVED))
def test_quantity_cannot_read_a_non_observation_or_unresolved_source(disposition):
    surface, results, manifest = _fixture()
    item = replace(manifest.classifications[1], disposition=disposition,
                   non_observation_reason=NonObservationReason.SOFTWARE_PLUMBING_OR_UTILITY
                   if disposition is SurfaceDisposition.NOT_OBSERVATION else None)
    with pytest.raises(ExtractionValidationError, match="quantity source"):
        validate_extraction_manifest(surface, results, replace(manifest,
            classifications=(manifest.classifications[0], item) + manifest.classifications[2:]), require_closed=False)


def test_coverage_cannot_discharge_a_non_observation():
    surface, results, manifest = _fixture()
    root = replace(manifest.classifications[0], disposition=SurfaceDisposition.NOT_OBSERVATION,
                   non_observation_reason=NonObservationReason.SOFTWARE_PLUMBING_OR_UTILITY)
    updated = replace(manifest, classifications=(root,) + manifest.classifications[1:])
    with pytest.raises(ExtractionValidationError, match="coverage cannot discharge"):
        validate_extraction_manifest(surface, results, updated)
    edge = CoverageLink(manifest.quantities[0].sources[0], "invalid target", _evidence(), target_source=root.source)
    with pytest.raises(ExtractionValidationError, match="non-observation target"):
        validate_extraction_manifest(surface, results, replace(updated, coverage=manifest.coverage[2:] + (edge,)))


def test_direct_mapping_requires_a_matching_quantity_source():
    surface, results, manifest = _fixture()
    wrong = replace(manifest.coverage[-1], target_quantity="native_distance")
    with pytest.raises(ExtractionValidationError, match="quantity/source join"):
        validate_extraction_manifest(surface, results, replace(manifest, coverage=manifest.coverage[:-1] + (wrong,)))


def test_observation_cannot_carry_a_non_observation_reason():
    surface, results, manifest = _fixture()
    wrong = replace(manifest.classifications[0], non_observation_reason=NonObservationReason.SOFTWARE_PLUMBING_OR_UTILITY)
    with pytest.raises(ExtractionValidationError, match="observation carries"):
        validate_extraction_manifest(surface, results, replace(manifest, classifications=(wrong,) + manifest.classifications[1:]))


def test_result_census_order_must_be_canonical():
    surface, results, manifest = _fixture()
    with pytest.raises(ExtractionValidationError, match="sorted and unique"):
        validate_extraction_manifest(surface, replace(results, results=tuple(reversed(results.results))), manifest)
