"""Independent runtime declarations detect omitted native outputs."""

from dataclasses import replace
import json

import numpy as np
import pytest

from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.extraction import (
    CoverageLink, ExtractionManifest, QuantityOutcome, QuantityRecord, ResultDeclaration, ResultSurface, SourceAddress,
    SurfaceClassification, SurfaceDisposition,
    manifest_digest, quantity_metadata,
)
from specola.conformance.instances import (
    ExpectedInstance, ExpectedOutcome, InstanceValidationError, NativePathStep, QuantityDomain,
    RuntimePopulation, ScalarProjection, ValueConstraint, validate_materialization, validate_runtime_population,
)
from specola.conformance.ledger import ExactnessKind
from specola.conformance.query_bindings import ArgumentSchema
from specola.conformance.provider import ExtractionProvider, PreparedExtraction
from specola.conformance.surface import ApiMember, MemberKind, ProducerIdentity, ProducerSurface, SurfaceRoot, surface_digest
from specola.entities import EntityState, TrackState
from specola.extraction_observation import ExtractionObservation, StructuredSample
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration


def _array_schema(shape=None):
    properties = {
        "kind": {"const": "ndarray"}, "dtype": {"const": "<f8"},
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "data": {"type": "string"},
    }
    if shape is not None:
        properties["shape"] = {"const": list(shape)}
    return ArgumentSchema.from_document({
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    })


def _fixture(extra=0):
    # Bodies 2 and 5 own entities 11 and 12; ``extra`` appends more owners.
    rows = ((2, 11, (1.0, -2.0)), (5, 12, (3.0, 4.0))) + tuple(
        (20 + index, 30 + index, (float(index), 0.5)) for index in range(extra))
    evidence = (EvidenceIdentity("native declaration fixture", "a" * 64),)
    outcome = QuantityOutcome("present", "declared body exists", _array_schema(), evidence)
    quantity = QuantityRecord(
        quantity_id="body.velocity", sources=(SourceAddress("native.body", "velocity"),),
        native_meaning="body velocity", owner_resolver_id="native body table",
        index_domain="body IDs", unit="m/s", frame="world", exactness=ExactnessKind.EXACT,
        availability="declared body exists", freshness="current native state", transformation="identity",
        output_sink="structured_and_properties", property_pattern="velocity_{axis}",
        metadata_resolver_id="body velocity metadata", reader_id="body velocity reader",
        provenance=evidence, owner_kind="body", outcomes=(outcome,),
    )
    manifest = ExtractionManifest("fixture", "b" * 64, (), (quantity,), ())
    queries = QueryConfiguration.empty()
    domain = QuantityDomain("body.velocity", "native body declarations", evidence)
    # These native owners and layouts are fixture declarations, not collected
    # from reader outputs or from production owner/address resolvers.
    instances = tuple(ExpectedInstance(
        "body.velocity", "body", (body,), "", entity,
        (ExpectedOutcome("present", _array_schema((2,)), (
            ScalarProjection("velocity_x", (NativePathStep("element", (0,)),)),
            ScalarProjection("velocity_y", (NativePathStep("element", (1,)),)),
        )),),
    ) for body, entity, _ in rows)
    population = RuntimePopulation(
        manifest.producer_surface_digest, manifest_digest(manifest), queries, 0, 0,
        (domain,), instances,
    )
    samples = tuple(StructuredSample(
        "body.velocity", "body", (body,), NativeValue.capture(np.array(values, dtype="<f8")),
        "present", entity, ("velocity_x", "velocity_y"),
    ) for body, entity, values in rows)
    extraction = ExtractionObservation(
        0, 0.0, 0, 0, manifest.producer_surface_digest, manifest_digest(manifest), queries,
        (quantity_metadata(quantity),), samples,
    )
    entities = {
        entity: EntityState(entity, "body", 0, TrackState.CONFIRMED,
                            position=np.zeros(3), velocity=np.zeros(3),
                            properties=dict(zip(("velocity_x", "velocity_y"), values)))
        for _, entity, values in rows
    }
    observation = MonitorObservation(0, 0.0, entities, extraction=extraction)
    return manifest, population, observation


def _provider_fixture():
    manifest, population, observation = _fixture()
    evidence = EvidenceIdentity("native declaration fixture", "a" * 64)
    surface = ProducerSurface(
        ProducerIdentity("native fixture", "1", "synthetic", evidence), evidence, evidence,
        (SurfaceRoot("native", evidence, "fixture producer declaration"),),
        (ApiMember("native", "body", MemberKind.TYPE, "body.velocity: float64[2]"),),
    )
    source, result = SourceAddress("native.body"), SourceAddress("native.body", "velocity")
    results = ResultSurface(surface_digest(surface), (ResultDeclaration(result, "float64[2]", "declared body exists", (evidence,)),), (evidence,))
    manifest = replace(manifest, producer_surface_digest=surface_digest(surface), classifications=(
        SurfaceClassification(source, SurfaceDisposition.REPRESENTED_ELSEWHERE, "body result fields", (evidence,)),
        SurfaceClassification(result, SurfaceDisposition.EXPOSED, "native velocity", (evidence,)),
    ), coverage=(
        CoverageLink(source, "body field projection", (evidence,), target_source=result),
        CoverageLink(result, "identity", (evidence,), target_quantity="body.velocity"),
    ))
    population = replace(population, producer_surface_sha256=surface_digest(surface), manifest_sha256=manifest_digest(manifest))
    assert observation.extraction is not None
    observation = replace(observation, extraction=replace(observation.extraction, producer_surface_sha256=surface_digest(surface), manifest_sha256=manifest_digest(manifest)))
    return surface, results, manifest, population, observation


class _Provider(ExtractionProvider):
    def guard_source(self, adapter):
        assert isinstance(adapter, _ProviderAdapter)
        adapter.events.append("source guard")

    def guard_runtime(self, adapter):
        adapter.events.append("runtime guard")

    def prepare(self, adapter, *, reset):
        assert isinstance(adapter, _ProviderAdapter)
        adapter.events.append("prepare")
        surface, results, manifest, population, _ = _provider_fixture()
        model_generation = adapter.model_generation

        def guard_after_read():
            adapter.events.append("post-read guard")
            if adapter.model_generation != model_generation:
                raise RuntimeError("native model changed during observation")

        return PreparedExtraction(surface, results, manifest, population, guard_after_read)


class _ProviderAdapter(SimulatorAdapter):
    _extraction_provider = _Provider()

    def __init__(self, damage=""):
        super().__init__()
        self.index = -1
        self.damage = damage
        self.events = []
        self.model_generation = 0

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    def _bind_extraction_subject(self, *, reset):
        self.events.append("bind")

    def _validate_prepared_extraction(self, prepared):
        assert self._prepared_extraction is prepared
        assert self._extraction_scope is None
        self.events.append("layout")
        if self.damage == "layout":
            raise RuntimeError("native layout differs before read")

    def _reset_impl(self):
        self.index = -1
        return self._get_observation_impl()

    def _get_observation_impl(self):
        self.events.append("read")
        assert type(self._prepared_extraction) is PreparedExtraction
        if self.damage == "nested_read":
            prepared = self._prepared_extraction
            before = list(self.events)
            for nested in (self.reset, self.get_observation,
                           lambda: self._read_extraction_observation(reset=True),
                           lambda: self._read_extraction_observation(reset=True, provider=self._extraction_provider)):
                with pytest.raises(RuntimeError, match="already active"):
                    nested()
            assert self.events == before
            assert self._prepared_extraction is prepared
        self.index += 1
        _, _, _, _, observation = _provider_fixture()
        assert observation.extraction is not None
        observation = replace(observation, timestep=self.index, time=self.index * 0.1,
                              extraction=replace(observation.extraction, timestep=self.index, time=self.index * 0.1))
        if self.damage == "reader":
            assert observation.extraction is not None
            observation = replace(observation, extraction=replace(observation.extraction, samples=()))
        if self.damage == "model":
            self.model_generation += 1
        # Existing adapters may prune scalar readers according to scope.
        # A provider-enrolled raw read must still receive the full subject.
        for entity in observation.perceived_entities.values():
            entity.properties = {key: value for key, value in entity.properties.items()
                                 if self._property_is_requested("body", key)}
        return observation


def test_public_adapter_path_prepares_before_read_and_projects_after_join():
    adapter = _ProviderAdapter()
    adapter.set_extraction_scope({"body": frozenset({"velocity_x"})})
    initial = adapter.reset()
    following = adapter.get_observation()
    assert adapter._prepared_extraction is None
    assert adapter.events == [
        "source guard", "bind", "prepare", "layout", "read", "post-read guard",
        "runtime guard", "bind", "prepare", "layout", "read", "post-read guard",
    ]
    for observation in (initial, following):
        assert {tuple(entity.properties) for entity in observation.perceived_entities.values()} == {("velocity_x",)}
        assert observation.extraction is not None
        assert [sample.owner_index for sample in observation.extraction.samples] == [(2,), (5,)]
        assert [sample.scalar_keys for sample in observation.extraction.samples] == [("velocity_x",)] * 2
        assert observation.extraction.samples[0].value.restore().tolist() == [1.0, -2.0]


@pytest.mark.parametrize("damage", ("reader", "model", "layout"))
def test_public_adapter_path_rejects_damage_before_recording(damage):
    adapter = _ProviderAdapter(damage)
    with pytest.raises((InstanceValidationError, RuntimeError)):
        adapter.reset()
    assert not adapter.episode_started
    assert adapter.last_observation is None
    assert adapter._prepared_extraction is None
    if damage == "layout":
        assert "read" not in adapter.events


def test_nested_read_refuses_before_provider_and_preserves_outer_subject():
    adapter = _ProviderAdapter("nested_read")
    initial = adapter.reset()
    following = adapter.get_observation()
    assert initial.timestep == 0 and following.timestep == 1
    assert adapter.episode_started and adapter.last_observation is following
    assert adapter._prepared_extraction is None


def test_provider_removal_cannot_turn_an_active_episode_into_legacy(monkeypatch):
    from specola.conformance.provider import ExtractionGuardError

    adapter = _ProviderAdapter()
    initial = adapter.reset()
    events = list(adapter.events)
    monkeypatch.setattr(_ProviderAdapter, "_extraction_provider", None)
    with pytest.raises(ExtractionGuardError, match="provider_binding_changed"):
        adapter.get_observation()
    assert adapter.events == events
    assert adapter.last_observation is initial


def test_scope_is_restored_after_a_provider_read_fails():
    adapter = _ProviderAdapter("reader")
    adapter.set_extraction_scope({"body": frozenset({"velocity_x"})})
    with pytest.raises(InstanceValidationError):
        adapter.reset()
    adapter.damage = ""
    observation = adapter.reset()
    assert {tuple(entity.properties) for entity in observation.perceived_entities.values()} == {("velocity_x",)}


def test_declared_population_and_scalar_projection_are_materialized():
    validate_materialization(*_fixture())


@pytest.mark.parametrize("change", (
    "reader", "owner", "extra_owner", "address", "kind", "outcome", "shape", "dtype",
    "frame", "meaning", "provenance", "configuration", "epoch", "manifest",
    "missing_link", "extra_link", "scalar_value", "scalar_sign", "extra_scalar",
    "domain", "duplicate_expected", "empty_population", "empty_outcomes",
))
def test_omissions_and_information_changes_fail(change):
    manifest, population, observation = _fixture()
    extraction = observation.extraction
    assert extraction is not None
    first = extraction.samples[0]
    if change == "reader":
        extraction = replace(extraction, samples=())
    elif change == "owner":
        extraction = replace(extraction, samples=extraction.samples[:1])
    elif change == "extra_owner":
        extraction = replace(extraction, samples=extraction.samples + (replace(first, owner_index=(8,), scalar_keys=()),))
    elif change in ("address", "kind", "outcome", "shape", "dtype", "missing_link", "extra_link"):
        replacements = {
            "address": {"owner_index": (7,)}, "kind": {"owner_kind": "geom"},
            "outcome": {"availability": "unresolved"},
            "shape": {"value": NativeValue.capture(np.array([1.0], dtype="<f8"))},
            "dtype": {"value": NativeValue.capture(np.array([1.0, -2.0], dtype="<f4"))},
            "missing_link": {"scalar_keys": ("velocity_x",)},
            "extra_link": {"scalar_keys": ("velocity_x", "velocity_y", "unused")},
        }
        extraction = replace(extraction, samples=(replace(first, **replacements[change]),) + extraction.samples[1:])
    elif change in ("frame", "meaning", "provenance"):
        field = {"frame": "frame", "meaning": "native_meaning", "provenance": "provenance_sha256"}[change]
        extraction = replace(extraction, metadata=(replace(extraction.metadata[0], **{field: "c" * 64}),))
    elif change == "configuration":
        extraction = replace(extraction, producer_surface_sha256="c" * 64)
    elif change == "epoch":
        extraction = replace(extraction, episode_epoch=1)
    elif change == "manifest":
        extraction = replace(extraction, manifest_sha256="c" * 64)
    elif change in ("scalar_value", "scalar_sign", "extra_scalar"):
        observation.perceived_entities[11].properties["velocity_x" if change != "extra_scalar" else "unused"] = (
            -1.0 if change == "scalar_sign" else 99.0
        )
    elif change == "domain":
        population = replace(population, domains=())
    elif change == "duplicate_expected":
        population = replace(population, instances=population.instances + population.instances[:1])
    elif change == "empty_population":
        population = replace(population, instances=())
    elif change == "empty_outcomes":
        population = replace(population, instances=(replace(population.instances[0], outcomes=()),) + population.instances[1:])
    observation = replace(observation, extraction=extraction)
    with pytest.raises(InstanceValidationError):
        validate_materialization(manifest, population, observation)


def test_empty_native_domain_requires_an_explicit_reason():
    manifest, population, observation = _fixture()
    assert observation.extraction is not None
    population = replace(population, instances=(), domains=(replace(population.domains[0], empty_reason="native body table is empty"),))
    observation = replace(observation, perceived_entities={}, extraction=replace(observation.extraction, samples=()))
    validate_materialization(manifest, population, observation)


def test_structured_only_result_does_not_require_a_scalar_owner():
    manifest, population, observation = _fixture()
    assert observation.extraction is not None
    instances = tuple(replace(item, entity_id=None, outcomes=tuple(replace(outcome, projections=()) for outcome in item.outcomes)) for item in population.instances)
    samples = tuple(replace(item, entity_id=None, scalar_keys=()) for item in observation.extraction.samples)
    population = replace(population, instances=instances)
    observation = replace(observation, perceived_entities={}, extraction=replace(observation.extraction, samples=samples))
    validate_materialization(manifest, population, observation)


def test_independently_declared_legacy_scalar_slots_remain_available():
    manifest, population, observation = _fixture()
    observation.perceived_entities[11].properties["mass"] = 2.0
    population = replace(population, legacy_scalar_slots=((11, "mass"),), legacy_scalar_provenance=(EvidenceIdentity("legacy body mass declaration", "d" * 64),))
    validate_materialization(manifest, population, observation)
    observation.perceived_entities[11].properties.pop("mass")
    with pytest.raises(InstanceValidationError, match="scalar population"):
        validate_materialization(manifest, population, observation)


@pytest.mark.parametrize("position", (0, -1))
@pytest.mark.parametrize("damage", ("missing_field", "renamed_field", "field_kind"))
def test_nested_result_components_are_checked_by_declared_schema(damage, position):
    manifest, population, observation = _fixture()
    assert observation.extraction is not None
    properties = {"kind": {"const": "record"}, "fields": {
        "type": "array", "prefixItems": [
            {"type": "array", "prefixItems": [{"const": name}, _array_schema((2,)).document()], "minItems": 2, "maxItems": 2}
            for name in ("velocity", "witness")
        ], "minItems": 2, "maxItems": 2,
    }}
    schema = ArgumentSchema.from_document({"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False})
    quantity = replace(manifest.quantities[0], outcomes=(replace(manifest.quantities[0].outcomes[0], native_schema=schema),))
    manifest = replace(manifest, quantities=(quantity,))
    instances = tuple(replace(item, entity_id=None, outcomes=(ExpectedOutcome("present", schema),)) for item in population.instances)
    population = replace(population, manifest_sha256=manifest_digest(manifest), instances=instances)
    samples = tuple(replace(item, entity_id=None, scalar_keys=(), value=NativeValue.capture({
        "velocity": np.array([1.0, 2.0], dtype="<f8"), "witness": np.array([3.0, 4.0], dtype="<f8"),
    })) for item in observation.extraction.samples)
    extraction = replace(observation.extraction, manifest_sha256=manifest_digest(manifest), metadata=(quantity_metadata(quantity),), samples=samples)
    observation = replace(observation, perceived_entities={}, extraction=extraction)
    validate_materialization(manifest, population, observation)
    # The last position follows a sample whose layout was already accepted.
    velocity = np.array([1.0, 2.0], dtype="<f8")
    value = {"missing_field": {"velocity": velocity},
             "renamed_field": {"velocity": velocity, "witnesses": np.array([3.0, 4.0], dtype="<f8")},
             "field_kind": {"velocity": velocity, "witness": [3.0, 4.0]}}[damage]
    damaged = list(samples)
    damaged[position] = replace(samples[position], value=NativeValue.capture(value))
    observation = replace(observation, extraction=replace(extraction, samples=tuple(damaged)))
    with pytest.raises(InstanceValidationError, match="native result layout differs"):
        validate_materialization(manifest, population, observation)


def test_metadata_retains_result_paths_and_outcome_schemas():
    manifest, _, _ = _fixture()
    metadata = quantity_metadata(manifest.quantities[0])
    assert json.loads(metadata.source_members[0]) == ["native.body", "velocity"]
    assert json.loads(metadata.native_schema) == {"outcomes": {"present": _array_schema().document()}}


@pytest.mark.parametrize("operation,argument,expected", (
    ("identity", None, (1.0, 3.0)),
    ("min_with_argument", 2.0, (1.0, 2.0)),
    ("ge_indicator", 2.0, (0.0, 1.0)),
))
def test_scalar_operations_preserve_native_values_and_check_the_projection(operation, argument, expected):
    manifest, population, observation = _fixture()
    instances = tuple(replace(instance, outcomes=(replace(
        instance.outcomes[0], projections=(replace(instance.outcomes[0].projections[0], operation=operation, argument=argument), instance.outcomes[0].projections[1]),
    ),)) for instance in population.instances)
    population = replace(population, instances=instances)
    for entity_id, value in zip((11, 12), expected):
        observation.perceived_entities[entity_id].properties["velocity_x"] = value
    validate_materialization(manifest, population, observation)
    assert observation.extraction is not None
    assert [sample.value.restore()[0] for sample in observation.extraction.samples] == [1.0, 3.0]
    observation.perceived_entities[11].properties["velocity_x"] = expected[0] + 1.0
    with pytest.raises(InstanceValidationError, match="projection"):
        validate_materialization(manifest, population, observation)


@pytest.mark.parametrize("relation,argument,valid,invalid", (
    ("finite", None, 1.0, float("nan")), ("lt", 2.0, 1.0, 2.0),
    ("le", 2.0, 2.0, 3.0), ("ge", 2.0, 2.0, 1.0), ("eq", 2.0, 2.0, 1.0),
))
def test_outcome_predicates_reject_a_native_value_from_another_outcome(relation, argument, valid, invalid):
    manifest, population, observation = _fixture()
    constraint = ValueConstraint((NativePathStep("element", (0,)),), relation, argument)
    instances = tuple(replace(instance, entity_id=None, outcomes=(replace(instance.outcomes[0], projections=(), constraints=(constraint,)),)) for instance in population.instances)
    population = replace(population, instances=instances)
    assert observation.extraction is not None
    samples = tuple(replace(sample, entity_id=None, scalar_keys=(), value=NativeValue.capture(np.array([valid, 0.0], dtype="<f8"))) for sample in observation.extraction.samples)
    observation = replace(observation, perceived_entities={}, extraction=replace(observation.extraction, samples=samples))
    validate_materialization(manifest, population, observation)
    damaged = replace(samples[0], value=NativeValue.capture(np.array([invalid, 0.0], dtype="<f8")))
    assert observation.extraction is not None
    observation = replace(observation, extraction=replace(observation.extraction, samples=(damaged,) + samples[1:]))
    with pytest.raises(InstanceValidationError, match="outcome policy"):
        validate_materialization(manifest, population, observation)


def test_boolean_native_value_needs_an_explicit_numeric_encoding():
    manifest, population, observation = _fixture()
    schema = ArgumentSchema.from_document({
        "type": "object", "properties": {"kind": {"const": "bool"}, "value": {"type": "boolean"}},
        "required": ["kind", "value"], "additionalProperties": False,
    })
    quantity = replace(manifest.quantities[0], outcomes=(replace(manifest.quantities[0].outcomes[0], native_schema=schema),))
    manifest = replace(manifest, quantities=(quantity,))
    population = replace(population, manifest_sha256=manifest_digest(manifest), instances=tuple(
        replace(instance, outcomes=(ExpectedOutcome("present", schema, (ScalarProjection("velocity_x"),)),))
        for instance in population.instances
    ))
    assert observation.extraction is not None
    samples = tuple(replace(sample, value=NativeValue.capture(True), scalar_keys=("velocity_x",)) for sample in observation.extraction.samples)
    observation = replace(observation, extraction=replace(observation.extraction, manifest_sha256=manifest_digest(manifest), metadata=(quantity_metadata(quantity),), samples=samples))
    for entity in observation.perceived_entities.values():
        entity.properties = {"velocity_x": 1.0}
    with pytest.raises(InstanceValidationError, match="real scalar"):
        validate_materialization(manifest, population, observation)


def _with_extra_legacy_scalar(observation):
    """Add a legacy owner (no typed projection) carrying one adapter scalar."""
    entities = dict(observation.perceived_entities)
    entities[13] = EntityState(13, "region", 0, TrackState.CONFIRMED, position=None, velocity=None,
                               properties={"legacy_key": 7.0})
    return replace(observation, perceived_entities=entities)


def test_exact_census_rejects_an_undeclared_scalar_without_a_rule():
    manifest, population, observation = _fixture()
    with pytest.raises(InstanceValidationError, match="scalar population differs"):
        validate_materialization(manifest, population, _with_extra_legacy_scalar(observation))


def test_legacy_rule_accepts_reviewed_adapter_scalars_and_keeps_projection_checks():
    manifest, population, observation = _fixture()
    ruled = replace(population, legacy_scalar_rule="scalars emitted by the reviewed adapter source",
                    legacy_scalar_provenance=(EvidenceIdentity("specola/adapters/fixture.py", "c" * 64),))
    validate_materialization(manifest, ruled, _with_extra_legacy_scalar(observation))
    entities = dict(observation.perceived_entities)
    entities[11] = replace(entities[11], properties={"velocity_x": 1.0})
    with pytest.raises(InstanceValidationError, match="declared scalar projection is missing"):
        validate_materialization(manifest, ruled, replace(observation, perceived_entities=entities))
    entities[11] = replace(observation.perceived_entities[11], properties={"velocity_x": 9.0, "velocity_y": -2.0})
    with pytest.raises(InstanceValidationError, match="differs from its declared native projection"):
        validate_materialization(manifest, ruled, replace(observation, perceived_entities=entities))


def test_legacy_rule_keeps_the_exact_census_on_typed_owners():
    manifest, population, observation = _fixture()
    ruled = replace(population, legacy_scalar_rule="scalars emitted by the reviewed adapter source",
                    legacy_scalar_provenance=(EvidenceIdentity("specola/adapters/fixture.py", "c" * 64),))
    entities = dict(observation.perceived_entities)
    entities[11] = replace(entities[11], properties=dict(entities[11].properties, stray_key=0.5))
    with pytest.raises(InstanceValidationError, match="typed owner scalar population differs"):
        validate_materialization(manifest, ruled, replace(observation, perceived_entities=entities))


def test_legacy_rule_needs_provenance_and_excludes_enumerated_slots():
    manifest, population, observation = _fixture()
    provenance = (EvidenceIdentity("specola/adapters/fixture.py", "c" * 64),)
    with pytest.raises(InstanceValidationError, match="legacy scalar population"):
        validate_materialization(manifest, replace(population, legacy_scalar_rule="reviewed adapter source"), observation)
    with pytest.raises(InstanceValidationError, match="exclusive"):
        validate_materialization(manifest, replace(
            population, legacy_scalar_rule="reviewed adapter source", legacy_scalar_provenance=provenance,
            legacy_scalar_slots=((11, "legacy_key"),)), observation)
    with pytest.raises(InstanceValidationError, match="nonempty text"):
        validate_materialization(manifest, replace(population, legacy_scalar_rule="  ", legacy_scalar_provenance=provenance), observation)


def _count_path_checks(monkeypatch):
    """Record each native-path check, a step taken once per declared projection."""
    from specola.conformance import instances

    calls = []
    original = instances._path

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(instances, "_path", counted)
    return calls


def test_an_accepted_population_is_not_walked_again_per_read(monkeypatch):
    # Issue #136: a provider builds a new RuntimePopulation record per read
    # around the same immutable declaration tuples, and both the pre-read and
    # post-read validation used to walk every declared instance.
    manifest, population, observation = _fixture()
    calls = _count_path_checks(monkeypatch)
    validate_runtime_population(manifest, population)
    assert len(calls) == 4
    validate_materialization(manifest, population, observation)
    following = replace(population, episode_epoch=1)
    assert observation.extraction is not None
    validate_runtime_population(manifest, following)
    validate_materialization(manifest, following, replace(observation, extraction=replace(observation.extraction, episode_epoch=1)))
    assert len(calls) == 4


@pytest.mark.parametrize("renew", ("manifest", "domains", "instances"))
def test_an_equal_declaration_with_a_new_identity_is_validated_again(monkeypatch, renew):
    manifest, population, _ = _fixture()
    validate_runtime_population(manifest, population)
    calls = _count_path_checks(monkeypatch)
    if renew == "manifest":
        manifest = replace(manifest)
    else:
        population = replace(population, **{renew: tuple(list(getattr(population, renew)))})
    validate_runtime_population(manifest, population)
    assert len(calls) == 4


def _count_descriptions(monkeypatch):
    """Record each manifest digest and quantity metadata computation."""
    from specola.conformance import instances

    calls = []
    digest, metadata = instances.manifest_digest, instances.quantity_metadata

    def counted_digest(manifest):
        calls.append("digest")
        return digest(manifest)

    def counted_metadata(quantity):
        calls.append("metadata")
        return metadata(quantity)

    monkeypatch.setattr(instances, "manifest_digest", counted_digest)
    monkeypatch.setattr(instances, "quantity_metadata", counted_metadata)
    return calls


def test_an_identical_manifest_is_described_once_across_reads(monkeypatch):
    # Issue #136, maintainer decision 2026-10-12: providers hand every read
    # the same frozen manifest, whose digest and replay metadata cost about
    # 9 ms per LIBERO observation.
    manifest, population, observation = _fixture()
    calls = _count_descriptions(monkeypatch)
    for _ in range(2):
        validate_runtime_population(manifest, replace(population))
        validate_materialization(manifest, replace(population), observation)
    assert calls == ["digest", "metadata"]


def test_a_new_manifest_object_is_described_again(monkeypatch):
    manifest, population, observation = _fixture()
    validate_materialization(manifest, population, observation)
    calls = _count_descriptions(monkeypatch)
    validate_materialization(replace(manifest), population, observation)
    assert calls == ["digest", "metadata"]


def test_a_changed_manifest_is_still_rejected_after_an_accepted_one():
    manifest, population, observation = _fixture()
    validate_materialization(manifest, population, observation)
    with pytest.raises(InstanceValidationError, match="identity differs"):
        validate_materialization(replace(manifest, identity="changed"), population, observation)


@pytest.mark.parametrize("field,value,match", (
    ("episode_epoch", -1, "epochs"),
    ("configuration_epoch", True, "epochs"),
    ("manifest_sha256", "c" * 64, "identity differs"),
    ("producer_surface_sha256", "c" * 64, "identity differs"),
    ("queries", None, "query configuration"),
    ("legacy_scalar_rule", "reviewed adapter source", "legacy scalar population"),
))
def test_per_read_fields_are_checked_on_an_accepted_population(field, value, match):
    manifest, population, observation = _fixture()
    validate_materialization(manifest, population, observation)
    changed = replace(population, **{field: value})
    with pytest.raises(InstanceValidationError, match=match):
        validate_runtime_population(manifest, changed)
    with pytest.raises(InstanceValidationError, match=match):
        validate_materialization(manifest, changed, observation)


def test_a_rejected_population_is_rejected_on_every_use():
    manifest, population, observation = _fixture()
    validate_materialization(manifest, population, observation)
    duplicated = replace(population, instances=population.instances + population.instances[:1])
    for validate in (lambda: validate_runtime_population(manifest, duplicated),
                     lambda: validate_runtime_population(manifest, duplicated),
                     lambda: validate_materialization(manifest, duplicated, observation)):
        with pytest.raises(InstanceValidationError, match="duplicate expected native instance"):
            validate()


def _count_schema_runs(monkeypatch):
    """Record the schema text of each jsonschema layout validation."""
    calls = []
    original = ArgumentSchema.validate

    def counted(self, *args, **kwargs):
        calls.append(self.canonical_json)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArgumentSchema, "validate", counted)
    return calls


def _last_sample_replaced(observation, value):
    samples = observation.extraction.samples
    last = replace(samples[-1], value=NativeValue.capture(value))
    return replace(observation, extraction=replace(observation.extraction, samples=samples[:-1] + (last,)))


def test_layout_schemas_run_once_per_encoded_layout_in_an_episode(monkeypatch):
    # Issue #136: jsonschema ran on every sample's document, 669,622 runs per
    # RoboCasa observation. Documents that differ only in their base64
    # payloads share one verdict (2026-10-07), kept for the episode's
    # observations of one declared population (2026-10-12).
    manifest, population, observation = _fixture(extra=8)
    assert observation.extraction is not None and len(observation.extraction.samples) == 10
    calls = _count_schema_runs(monkeypatch)
    validate_materialization(manifest, population, observation)
    assert sorted(calls) == sorted((_array_schema().canonical_json, _array_schema((2,)).canonical_json))
    calls.clear()
    # Providers build a new population record per read around the same
    # declarations.
    validate_materialization(manifest, replace(population), observation)
    assert calls == []


@pytest.mark.parametrize("renew", ("instances", "episode_epoch", "configuration_epoch"))
def test_new_declarations_or_a_new_episode_validate_layouts_again(monkeypatch, renew):
    manifest, population, observation = _fixture(extra=8)
    validate_materialization(manifest, population, observation)
    calls = _count_schema_runs(monkeypatch)
    if renew == "instances":
        population = replace(population, instances=tuple(list(population.instances)))
    else:
        population = replace(population, **{renew: 1})
        assert observation.extraction is not None
        observation = replace(observation, extraction=replace(observation.extraction, **{renew: 1}))
    validate_materialization(manifest, population, observation)
    assert sorted(calls) == sorted((_array_schema().canonical_json, _array_schema((2,)).canonical_json))


def test_a_layout_first_seen_in_a_later_observation_is_validated(monkeypatch):
    manifest, population, observation = _fixture(extra=8)
    population = replace(population, instances=tuple(
        replace(instance, outcomes=(replace(instance.outcomes[0], layout=_array_schema()),))
        for instance in population.instances))
    validate_materialization(manifest, population, observation)
    calls = _count_schema_runs(monkeypatch)
    later = _last_sample_replaced(observation, np.array([7.0, 0.5, 0.0]))
    validate_materialization(manifest, population, later)
    assert calls == [_array_schema().canonical_json]
    validate_materialization(manifest, population, later)
    assert calls == [_array_schema().canonical_json]
    # A layout the schema rejects is rejected in a later observation too.
    with pytest.raises(InstanceValidationError, match="native result layout differs"):
        validate_materialization(manifest, population, _last_sample_replaced(observation, [7.0, 0.5]))


def test_each_distinct_encoded_layout_is_validated_and_can_be_accepted(monkeypatch):
    manifest, population, observation = _fixture(extra=8)
    population = replace(population, instances=tuple(
        replace(instance, outcomes=(replace(instance.outcomes[0], layout=_array_schema()),))
        for instance in population.instances))
    observation = _last_sample_replaced(observation, np.array([7.0, 0.5, 0.0]))
    calls = _count_schema_runs(monkeypatch)
    validate_materialization(manifest, population, observation)
    assert calls == [_array_schema().canonical_json] * 2


@pytest.mark.parametrize("value", (
    np.array([7.0, 0.5, 0.0]), np.array([7.0, 0.5], dtype="<f4"), [7.0, 0.5],
), ids=("shape", "dtype", "kind"))
def test_a_later_sample_with_another_layout_is_still_rejected(value):
    # The damaged last sample keeps its projected values, so only the layout
    # check can reject it.
    manifest, population, observation = _fixture(extra=8)
    with pytest.raises(InstanceValidationError, match="native result layout differs"):
        validate_materialization(manifest, population, _last_sample_replaced(observation, value))


@pytest.mark.parametrize("check", ("projection", "constraint"))
def test_scalar_checks_still_run_on_every_sample(check):
    manifest, population, observation = _fixture(extra=8)
    if check == "projection":
        observation.perceived_entities[37].properties["velocity_y"] = 0.25
        match = "differs from its declared native projection"
    else:
        constraint = ValueConstraint((NativePathStep("element", (0,)),), "lt", 7.0)
        population = replace(population, instances=tuple(
            replace(instance, outcomes=(replace(instance.outcomes[0], constraints=(constraint,)),))
            for instance in population.instances))
        match = "outcome policy"
    with pytest.raises(InstanceValidationError, match=match):
        validate_materialization(manifest, population, observation)
