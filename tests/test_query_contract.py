"""Callable output closure follows materialization, including alias provenance."""

from dataclasses import replace

import pytest

from specola.conformance import query_bindings
from specola.conformance.audit import contract_surface_audit
from specola.conformance.extraction import (
    CoverageLink, ExactnessKind, ExtractionValidationError, SourceAddress,
    SurfaceClassification, SurfaceDisposition, validate_extraction_manifest,
)
from specola.conformance.surface import ApiMember, MemberKind, surface_digest
from test_extraction_manifest import _evidence, _fixture as manifest_fixture
from test_query_bindings import _fixture as binding_fixture


def _fixture():
    surface, results, manifest = manifest_fixture()
    binding, declaration = binding_fixture()
    binding = replace(binding, quantity_ids=("native_distance", "native_witness"),
                      reader_id="distance_query_reader")
    return surface, results, manifest, binding, declaration


def _validate(surface, results, manifest, bindings, declarations):
    return query_bindings.validate_query_contract(
        surface, results, manifest, bindings=bindings, declarations=declarations)


def _stored_aliases():
    surface, results, manifest, binding, declaration = _fixture()
    fields = (ApiMember("native", "distance_storage", MemberKind.FIELD, "float64"),
              ApiMember("native", "witness_storage", MemberKind.FIELD, "float64[2,3]"))
    surface = replace(surface, members=tuple(sorted(surface.members + fields, key=lambda item: item.member_id)))
    results = replace(results, producer_surface_digest=surface_digest(surface))
    addresses = (SourceAddress("native.distance_storage"), SourceAddress("native.witness_storage"))
    classes = tuple(replace(item, disposition=SurfaceDisposition.REPRESENTED_ELSEWHERE)
                    for item in manifest.classifications)
    classes += tuple(SurfaceClassification(address, SurfaceDisposition.EXPOSED,
                                          "native stored values", _evidence()) for address in addresses)
    quantities = tuple(replace(quantity, sources=quantity.sources + (address,), reader_id="native_field_reader")
                       for quantity, address in zip(manifest.quantities, addresses))
    links = tuple(link for link in manifest.coverage if link.target_source is not None)
    for quantity, address in zip(quantities, addresses):
        links += (CoverageLink(quantity.sources[0], "lossless native storage", _evidence(), target_source=address),
                  CoverageLink(address, "identity native storage", _evidence(), target_quantity=quantity.quantity_id))
    manifest = replace(manifest, producer_surface_digest=surface_digest(surface),
                       classifications=classes, quantities=quantities, coverage=links)
    return surface, results, manifest, binding, declaration


def test_complete_query_contract_preserves_distinct_operand_and_output_owner_resolvers():
    surface, results, manifest, binding, declaration = _fixture()
    assert binding.owner_resolver_id != manifest.quantities[0].owner_resolver_id
    assert _validate(surface, results, manifest, (binding,), (declaration,)) == ()


@pytest.mark.parametrize("omitted", ("native_distance", "native_witness"))
def test_each_omitted_query_output_fails_the_contract_and_audit(omitted):
    surface, results, manifest, binding, declaration = _fixture()
    binding = replace(binding, quantity_ids=tuple(item for item in binding.quantity_ids if item != omitted))
    for validate in (_validate, lambda s, r, m, b, d: contract_surface_audit(s, r, m, bindings=b, declarations=d)):
        with pytest.raises(query_bindings.QueryBindingError, match="query binding outputs"):
            validate(surface, results, manifest, (binding,), (declaration,))


def test_reader_substitution_does_not_pass_a_known_quantity_id_check():
    surface, results, manifest, binding, declaration = _fixture()
    with pytest.raises(query_bindings.QueryBindingError, match="reader"):
        _validate(surface, results, manifest, (replace(binding, reader_id="unrelated_reader"),), (declaration,))


def test_missing_binding_is_an_explicit_audit_obligation():
    surface, results, manifest, _, declaration = _fixture()
    assert _validate(surface, results, manifest, (), (declaration,)) == ("native.query",)
    document = contract_surface_audit(surface, results, manifest, declarations=(declaration,))
    assert document["unbound_callable_quantities"] == ["native.query"]
    assert document["contract_status"] == "declarations_only"


def test_missing_and_duplicate_parameter_declarations_fail():
    surface, results, manifest, binding, declaration = _fixture()
    for declarations in ((), (declaration, declaration)):
        with pytest.raises(query_bindings.QueryBindingError, match="declaration"):
            _validate(surface, results, manifest, (binding,), declarations)


def test_alias_provenance_does_not_authorize_or_require_query_execution():
    surface, results, manifest, binding, declaration = _stored_aliases()
    assert _validate(surface, results, manifest, (), (declaration,)) == ()
    with pytest.raises(query_bindings.QueryBindingError, match="no manifest quantity"):
        _validate(surface, results, manifest, (binding,), (declaration,))


def test_field_member_cannot_become_a_query_binding():
    surface, results, manifest, binding, declaration = _stored_aliases()
    binding = replace(binding, callable_id="native.distance_storage")
    declaration = replace(declaration, callable_id=binding.callable_id)
    with pytest.raises(query_bindings.QueryBindingError, match="producer callable"):
        _validate(surface, results, manifest, (binding,), (declaration,))


def test_same_callable_bindings_must_materialize_the_same_complete_outputs():
    surface, results, manifest, binding, declaration = _fixture()
    second = replace(binding, binding_id="another_instance_contract")
    assert _validate(surface, results, manifest, (binding, second), (declaration,)) == ()
    with pytest.raises(query_bindings.QueryBindingError, match="duplicate"):
        _validate(surface, results, manifest, (binding, binding), (declaration,))
    with pytest.raises(query_bindings.QueryBindingError, match="query binding outputs"):
        _validate(surface, results, manifest, (binding, replace(second, quantity_ids=("native_distance",))), (declaration,))


def test_approximate_extra_output_is_materialized_without_discharging_native_coverage():
    surface, results, manifest, binding, declaration = _fixture()
    approximate = replace(manifest.quantities[0], quantity_id="approximate_distance", exactness=ExactnessKind.APPROXIMATE,
                          approximation_relation="declared approximation relation")
    manifest = replace(manifest, quantities=manifest.quantities + (approximate,))
    with pytest.raises(query_bindings.QueryBindingError, match="query binding outputs"):
        _validate(surface, results, manifest, (binding,), (declaration,))
    binding = replace(binding, quantity_ids=binding.quantity_ids + (approximate.quantity_id,))
    assert _validate(surface, results, manifest, (binding,), (declaration,)) == ()


def test_approximation_cannot_hide_its_reader_behind_alias_only_provenance():
    surface, results, manifest, _, _ = _stored_aliases()
    approximate = replace(manifest.quantities[0], quantity_id="approximate_distance",
                          sources=(SourceAddress("native.query", "return.distance"),),
                          exactness=ExactnessKind.APPROXIMATE,
                          approximation_relation="declared approximation relation")
    with pytest.raises(ExtractionValidationError, match="no materialized source"):
        validate_extraction_manifest(surface, results, replace(manifest, quantities=manifest.quantities + (approximate,)))
