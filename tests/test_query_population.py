"""Configured bindings retain separate, independently declared owner domains."""

from dataclasses import replace

import pytest

from specola.adapter import MonitorObservation
from specola.conformance import instances
from specola.conformance.extraction import manifest_digest, quantity_metadata
from specola.conformance.provider import ExtractionGuardError, PreparedExtraction
from specola.conformance.query_bindings import QueryBindingError, QueryConfigurationRequired
from specola.extraction_observation import (
    ExtractionObservation, StructuredSample, extraction_document, extraction_from_document,
)
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration
from test_query_contract import _fixture as contract_fixture


def _fixture(*, explicit=False, overlapping=False):
    surface, results, manifest, first, declaration = contract_fixture()
    manifest = replace(manifest, quantities=tuple(replace(q, output_sink="structured", property_pattern="structured_only")
                                                 for q in manifest.quantities))
    second = replace(first, binding_id="second.query")
    bindings = (first, second)
    queries = QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": b.binding_id, "arguments": {"cutoff": 2.0}} for b in bindings],
        "instances": ([{"binding": first.binding_id, "id": "inspection", "arguments": {"cutoff": 1.0},
                        "operands": {"first": {"kind": "geom", "id": 0},
                                     "second": {"kind": "geom", "id": 1}}}] if explicit else []),
    })
    # Synthetic producer declarations enumerate owners separately for the two
    # bindings. No observation or production owner resolver supplies this table.
    owners = ((first.binding_id, "", (0, 1)),
              (second.binding_id, "", (0, 1) if overlapping else (2, 3)))
    if explicit:
        owners += ((first.binding_id, "inspection", (0, 1)),)
    expected = tuple(instances.ExpectedInstance(
        q.quantity_id, "geom_pair", owner, token, None,
        (instances.ExpectedOutcome("supported", q.outcomes[0].native_schema),), binding_id=binding,
    ) for binding, token, owner in owners for q in manifest.quantities)
    domains = tuple(instances.QuantityDomain(q.quantity_id, "fixture native owner table", q.provenance)
                    for q in manifest.quantities)
    query_domains = tuple(instances.QueryPopulationDomain(
        b.binding_id, q.quantity_id, "fixture binding owner table", q.provenance,
    ) for b in bindings for q in manifest.quantities)
    population = instances.RuntimePopulation(
        manifest.producer_surface_digest, manifest_digest(manifest), queries, 0, 0, domains, expected,
        query_domains=query_domains,
    )
    prepared = PreparedExtraction(surface, results, manifest, population, lambda: None,
                                  bindings=bindings, declarations=(declaration,))
    return prepared


def _samples(prepared):
    # These literal fixture outputs intentionally do not enumerate expected
    # instances: an omission on either side must remain detectable.
    return tuple(StructuredSample(q, "geom_pair", owner, NativeValue.capture(0.25), "supported",
                                  binding_id=binding)
                 for binding, owner in (("native.query", (0, 1)), ("second.query", (2, 3)))
                 for q in ("native_distance", "native_witness"))


def _observation(prepared, samples):
    population = prepared.population
    return MonitorObservation(0, 0.0, {}, extraction=ExtractionObservation(
        0, 0.0, 0, 0, population.producer_surface_sha256, population.manifest_sha256,
        population.queries, tuple(quantity_metadata(q) for q in prepared.manifest.quantities), samples,
    ))


def test_preparation_joins_query_contract_arguments_and_runtime_domains():
    prepared = _fixture(explicit=True)
    prepared.validate_before_read()
    with pytest.raises(QueryConfigurationRequired):
        replace(prepared, population=replace(prepared.population, queries=QueryConfiguration.empty())).validate_before_read()
    with pytest.raises(QueryBindingError, match="outputs"):
        replace(prepared, bindings=(replace(prepared.bindings[0], quantity_ids=("native_distance",)),)
                + prepared.bindings[1:]).validate_before_read()
    with pytest.raises(ExtractionGuardError, match="unbound_callable_quantities"):
        replace(prepared, bindings=()).validate_before_read()


def test_each_binding_has_its_own_population_even_when_other_owners_survive():
    prepared = _fixture()
    for binding in prepared.bindings:
        reduced = tuple(row for row in prepared.population.instances if row.binding_id != binding.binding_id)
        with pytest.raises(instances.InstanceValidationError, match="query population.*reason"):
            replace(prepared, population=replace(prepared.population, instances=reduced)).validate_before_read()
    for domain in prepared.population.query_domains:
        reduced = tuple(row for row in prepared.population.query_domains if row != domain)
        with pytest.raises(instances.InstanceValidationError, match="query population domain"):
            replace(prepared, population=replace(prepared.population, query_domains=reduced)).validate_before_read()


def test_explicit_result_components_are_required_per_token_and_binding():
    prepared = _fixture(explicit=True)
    for q in ("native_distance", "native_witness"):
        reduced = tuple(row for row in prepared.population.instances
                        if not (row.instance_token == "inspection" and row.quantity_id == q))
        with pytest.raises(instances.InstanceValidationError, match="explicit query.*quantity"):
            replace(prepared, population=replace(prepared.population, instances=reduced)).validate_before_read()
    named = next(row for row in prepared.population.instances if row.instance_token)
    bad = replace(named, binding_id="second.query")
    with pytest.raises(instances.InstanceValidationError, match="token.*binding"):
        replace(prepared, population=replace(prepared.population, instances=(bad,))).validate_before_read()


@pytest.mark.parametrize("changes", ({"binding_id": "unknown"}, {"binding_id": ""}, {"instance_token": "unknown"}))
def test_unknown_query_addresses_are_rejected(changes):
    prepared = _fixture()
    rows = (replace(prepared.population.instances[0], **changes),) + prepared.population.instances[1:]
    with pytest.raises(instances.InstanceValidationError, match="binding|token"):
        replace(prepared, population=replace(prepared.population, instances=rows)).validate_before_read()


def test_empty_reason_is_local_to_one_binding_population():
    prepared = _fixture(explicit=True)
    population = prepared.population
    population = replace(population, instances=tuple(row for row in population.instances
                         if row.binding_id != "native.query" or row.instance_token),
                         query_domains=tuple(replace(row, empty_reason="fixture declares no population owners")
                         if row.binding_id == "native.query" else row for row in population.query_domains))
    replace(prepared, population=population).validate_before_read()
    bad = replace(population, query_domains=population.query_domains + (population.query_domains[0],))
    with pytest.raises(instances.InstanceValidationError, match="duplicate query population"):
        replace(prepared, population=bad).validate_before_read()


def test_structured_addresses_distinguish_bindings_and_materialization_checks_them():
    prepared = _fixture()
    prepared.validate_before_read()
    observation = _observation(prepared, _samples(prepared))
    prepared.validate_after_read(observation)
    assert observation.extraction is not None
    assert len(observation.extraction.samples[0].address) == 5
    bad = replace(observation.extraction.samples[0], binding_id="other.query")
    with pytest.raises(instances.InstanceValidationError, match="instance population"):
        prepared.validate_after_read(_observation(prepared, (bad,) + observation.extraction.samples[1:]))
    overlapping = _fixture(overlapping=True)
    overlapping.validate_before_read()
    samples = tuple(replace(sample, owner_index=(0, 1)) for sample in _samples(prepared))
    overlapping.validate_after_read(_observation(overlapping, samples))


def test_binding_identity_round_trip_and_legacy_sample_restoration():
    prepared = _fixture()
    observation = _observation(prepared, _samples(prepared)).extraction
    assert observation is not None
    assert extraction_from_document(extraction_document(observation)) == observation
    document = extraction_document(observation)
    for sample in document["samples"]:
        del sample["binding_id"]
    original = [dict(sample) for sample in document["samples"]]
    restored = extraction_from_document(document)
    assert all(sample.binding_id == "" for sample in restored.samples)
    assert document["samples"] == original


def test_field_rows_reject_query_tokens_and_binding_identity():
    from test_extraction_instances import _provider_fixture

    surface, results, manifest, population, _ = _provider_fixture()
    for change in ({"binding_id": "native.query"}, {"instance_token": "inspection"}):
        bad = replace(population, instances=(replace(population.instances[0], **change),) + population.instances[1:])
        with pytest.raises(instances.InstanceValidationError, match="field.*query"):
            PreparedExtraction(surface, results, manifest, bad, lambda: None).validate_before_read()


def test_population_rows_need_a_resolver_and_domains_cannot_be_invented():
    prepared = _fixture(explicit=True)
    bindings = tuple(replace(b, population_resolver_id=None, population_arguments=None) for b in prepared.bindings)
    configuration = prepared.population.queries.document()
    configuration["populations"] = []
    population = replace(prepared.population, queries=QueryConfiguration.from_document(configuration), query_domains=())
    with pytest.raises(instances.InstanceValidationError, match="no configured population"):
        replace(prepared, bindings=bindings, population=population).validate_before_read()
    population = replace(population, instances=tuple(row for row in population.instances if row.instance_token))
    replace(prepared, bindings=bindings, population=population).validate_before_read()
    domain = prepared.population.query_domains[0]
    with pytest.raises(instances.InstanceValidationError, match="domain set"):
        replace(prepared, bindings=bindings, population=replace(population, query_domains=(
            replace(domain, empty_reason="no population resolver exists"),))).validate_before_read()


def test_distinct_output_owner_domains_do_not_require_equal_owner_sets():
    prepared = _fixture()
    population = replace(prepared.population, instances=tuple(
        replace(row, owner_index=(7, 8)) if row.quantity_id == "native_witness" else row
        for row in prepared.population.instances))
    replace(prepared, population=population).validate_before_read()


def test_overlapping_scalar_projections_remain_an_explicit_configuration_error():
    prepared = _fixture(overlapping=True)
    population = replace(prepared.population, instances=tuple(replace(row, entity_id=3, outcomes=(
        replace(row.outcomes[0], projections=(instances.ScalarProjection(row.quantity_id),)),))
        for row in prepared.population.instances))
    with pytest.raises(instances.InstanceValidationError, match="scalar address.*multiple"):
        replace(prepared, population=population).validate_before_read()


def test_combined_quantity_requires_one_owner_producing_binding():
    from specola.conformance.extraction import CoverageLink, SourceAddress, SurfaceClassification, SurfaceDisposition
    from specola.conformance.query_bindings import ArgumentSchema, CallableParameterSurface
    from specola.conformance.surface import ApiMember, MemberKind, surface_digest

    prepared = _fixture()
    source = SourceAddress("native.additional_query")
    surface = replace(prepared.surface, members=tuple(sorted(prepared.surface.members + (
        ApiMember("native", "additional_query", MemberKind.CALLABLE, "additional_query() -> native float64"),
    ), key=lambda row: row.member_id)))
    quantity = prepared.manifest.quantities[0]
    manifest = replace(prepared.manifest, producer_surface_digest=surface_digest(surface),
        quantities=(replace(quantity, sources=quantity.sources + (source,)),) + prepared.manifest.quantities[1:],
        classifications=prepared.manifest.classifications + (
            SurfaceClassification(source, SurfaceDisposition.EXPOSED, "combined fixture computation", quantity.provenance),),
        coverage=prepared.manifest.coverage + (
            CoverageLink(source, "fixture combined computation", quantity.provenance, target_quantity=quantity.quantity_id),))
    schema = ArgumentSchema.from_document({"type": "object", "properties": {}, "additionalProperties": False})
    additional = replace(prepared.bindings[0], binding_id=source.member_id, callable_id=source.member_id,
                         parameters=(), arguments=schema, quantity_ids=(quantity.quantity_id,),
                         population_resolver_id=None, population_arguments=None)
    population = replace(prepared.population, producer_surface_sha256=surface_digest(surface),
                         manifest_sha256=manifest_digest(manifest),
                         queries=QueryConfiguration.from_document({"schema": "specola.queries.v1", "instances": [],
                             "populations": [{"binding": "native.query", "arguments": {"cutoff": 2.0}}]}),
                         instances=tuple(row for row in prepared.population.instances if row.binding_id == "native.query"),
                         query_domains=tuple(row for row in prepared.population.query_domains if row.binding_id == "native.query"))
    prepared = replace(prepared, surface=surface, manifest=manifest, population=population,
                       results=replace(prepared.results, producer_surface_digest=surface_digest(surface)),
                       bindings=(prepared.bindings[0], additional), declarations=prepared.declarations + (
                           CallableParameterSurface(source.member_id, (), quantity.provenance),))
    prepared.validate_before_read()
    with pytest.raises(instances.InstanceValidationError, match="more than one owner-producing"):
        replace(prepared, bindings=(prepared.bindings[0], replace(additional,
            population_resolver_id="second population", population_arguments=schema))).validate_before_read()


@pytest.mark.parametrize("sample", (None, [], "not a sample"))
def test_legacy_normalization_rejects_non_object_samples(sample):
    from specola.extraction_observation import ExtractionObservationError

    prepared = _fixture()
    observation = _observation(prepared, _samples(prepared)).extraction
    assert observation is not None
    document = extraction_document(observation)
    document["samples"] = [sample]
    with pytest.raises(ExtractionObservationError):
        extraction_from_document(document)


def test_prepared_query_configuration_must_match_adapter_before_any_reader():
    from test_extraction_instances import _ProviderAdapter

    adapter = _ProviderAdapter()
    adapter.set_query_configuration(_fixture().population.queries)
    with pytest.raises(ExtractionGuardError, match="prepared_subject_mismatch"):
        adapter.reset()
    assert "read" not in adapter.events
    assert adapter.last_observation is None
    assert not adapter.episode_started


@pytest.mark.parametrize("field", ("episode_epoch", "configuration_epoch"))
def test_prepared_epochs_must_match_adapter_before_any_reader(field):
    from test_extraction_instances import _ProviderAdapter, _provider_fixture

    adapter = _ProviderAdapter()
    surface, results, manifest, population, _ = _provider_fixture()
    prepared = PreparedExtraction(surface, results, manifest, replace(population, **{field: 1}), lambda: None)
    with pytest.raises(ExtractionGuardError, match="prepared_subject_mismatch"):
        prepared.validate_before_read(adapter)
