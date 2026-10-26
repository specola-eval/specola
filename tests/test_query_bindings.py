"""Query subjects come from native declarations and author configuration."""

from dataclasses import replace
import json
from typing import Any, cast

import pytest

from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.query_bindings import (
    ArgumentSchema, CallableParameterSurface, ConfiguredQuery, ParameterRole, QueryBindingError,
    QueryBindingRecord, QueryConfigurationRequired, QueryParameter,
    ResolvedOperand, bind_query_configuration, validate_query_binding,
)
from specola.query_config import QueryConfiguration


def _fixture():
    evidence = (EvidenceIdentity("independent fixture declaration", "a" * 64),)
    schema = ArgumentSchema.from_document({
        "type": "object", "additionalProperties": False,
        "properties": {"cutoff": {"type": "number", "exclusiveMinimum": 0}},
        "required": ["cutoff"],
    })
    binding = QueryBindingRecord(
        "native.query", "native.query",
        (QueryParameter("state", ParameterRole.CONTEXT, "native state"),
         QueryParameter("first", ParameterRole.OWNER, "native ID", owner_kind="geom"),
         QueryParameter("second", ParameterRole.OWNER, "native ID", owner_kind="geom"),
         QueryParameter("cutoff", ParameterRole.ARGUMENT, "float64"),
         QueryParameter("witness", ParameterRole.OUTPUT_BUFFER, "float64[6]")),
        schema, ("native.return", "native.witness"), "native_geom_ids", "native_query",
        "same state", "declared scratch effects", "retain return and witness",
        "declared cutoff validity", evidence, population_resolver_id="ordered_native_pairs",
        population_arguments=schema,
    )
    # This census stands for independent producer declarations, not extraction.
    declaration = CallableParameterSurface("native.query", ("state", "first", "second", "cutoff", "witness"), evidence)
    return binding, declaration


def _document():
    return {"schema": "specola.queries.v1",
            "populations": [{"binding": "native.query", "arguments": {"cutoff": 2.0}}],
            "instances": [{"binding": "native.query", "id": "inspection",
                           "arguments": {"cutoff": 0.5},
                           "operands": {"first": {"kind": "geom", "name": "first"},
                                        "second": {"kind": "geom", "id": 3}}}]}


def _bind(document, resolver=None, binding=None, declaration=None):
    original, original_declaration = _fixture()
    return bind_query_configuration(
        QueryConfiguration.from_document(document),
        (binding or original,), (declaration or original_declaration,),
        ("native.return", "native.witness"),
        resolver or (lambda kind, operand: (operand.get("id", 2),)),
    )


def test_native_roles_cover_parameters_without_calling_the_query():
    binding, declaration = _fixture()
    validate_query_binding(binding, declaration, binding.quantity_ids)
    population, instance = _bind(_document())
    assert population.instance_token is None and population.operands == ()
    assert population.arguments() == {"cutoff": 2.0}
    assert instance.instance_token == "inspection"
    assert [(item.parameter, item.kind, item.native_index) for item in instance.operands] == [
        ("first", "geom", (2,)), ("second", "geom", (3,)),
    ]
    detached = instance.arguments()
    detached["cutoff"] = 10
    assert instance.arguments() == {"cutoff": 0.5}


def test_unused_new_native_parameter_is_not_hidden_by_binding_records():
    binding, declaration = _fixture()
    declaration = replace(declaration, parameter_names=declaration.parameter_names + ("new_parameter",))
    with pytest.raises(QueryBindingError, match="parameter population"):
        validate_query_binding(binding, declaration, binding.quantity_ids)


@pytest.mark.parametrize("mutation", (
    lambda item: replace(item, parameters=tuple(reversed(item.parameters))),
    lambda item: replace(item, parameters=item.parameters + (item.parameters[0],)),
    lambda item: replace(item, parameters=(replace(item.parameters[0], role="read"),) + item.parameters[1:]),
    lambda item: replace(item, parameters=(replace(item.parameters[0], owner_kind="geom"),) + item.parameters[1:]),
    lambda item: replace(item, parameters=(replace(item.parameters[0], default_json="1"),) + item.parameters[1:]),
    lambda item: replace(item, parameters=(item.parameters[0], replace(item.parameters[1], owner_kind=None)) + item.parameters[2:]),
    lambda item: replace(item, quantity_ids=("missing",)),
    lambda item: replace(item, population_arguments=None),
    lambda item: replace(item, population_resolver_id=None),
    lambda item: replace(item, effect_policy=""),
    lambda item: replace(item, freshness=""),
    lambda item: replace(item, result_mapping=""),
    lambda item: replace(item, provenance=()),
))
def test_incomplete_or_ambiguous_bindings_fail(mutation):
    binding, declaration = _fixture()
    with pytest.raises(QueryBindingError):
        validate_query_binding(mutation(binding), declaration, binding.quantity_ids)


@pytest.mark.parametrize("mutation", (
    lambda item: item["populations"][0].update(binding="unknown"),
    lambda item: item["instances"][0].update(binding="unknown"),
    lambda item: item["populations"][0]["arguments"].update(cutoff=0),
    lambda item: item["instances"][0]["arguments"].update(cutoff=False),
    lambda item: item["instances"][0]["arguments"].update(extra=2),
    lambda item: item["instances"][0]["operands"]["first"].update(kind="body"),
    lambda item: item["instances"][0]["operands"].pop("first"),
    lambda item: item["instances"][0]["operands"].update(extra={"kind": "geom", "id": 0}),
))
def test_invalid_arguments_fail_before_any_native_resolution(mutation):
    document = _document()
    mutation(document)
    calls = []
    with pytest.raises(QueryBindingError):
        _bind(document, resolver=lambda *args: calls.append(args))
    assert calls == []


def test_missing_finite_population_arguments_do_not_invent_a_default():
    document = _document()
    document["populations"] = []
    calls = []
    with pytest.raises(QueryConfigurationRequired) as caught:
        _bind(document, resolver=lambda *args: calls.append(args))
    assert caught.value.code == "query_configuration_required"
    assert caught.value.missing == (("native.query", ("cutoff",)),)
    assert calls == []


def test_open_argument_domain_has_no_eager_population():
    binding, _ = _fixture()
    binding = replace(binding, population_resolver_id=None, population_arguments=None)
    document = _document()
    with pytest.raises(QueryBindingError, match="no finite population"):
        _bind(document, binding=binding)
    document["populations"] = []
    assert len(_bind(document, binding=binding)) == 1
    document["instances"] = []
    assert _bind(document, binding=binding) == ()


def test_only_declared_native_defaults_are_applied():
    binding, declaration = _fixture()
    schema = binding.arguments.document()
    schema["required"] = []
    schema = ArgumentSchema.from_document(schema)
    binding = replace(binding, arguments=schema, population_arguments=schema,
                      parameters=tuple(replace(item, default_json="1.0") if item.name == "cutoff" else item
                                       for item in binding.parameters))
    document = _document()
    document["populations"] = []
    document["instances"][0]["arguments"] = {}
    population, instance = _bind(document, binding=binding, declaration=declaration)
    assert population.arguments() == instance.arguments() == {"cutoff": 1.0}


@pytest.mark.parametrize("result", ([1], (), (True,), ("1",), 1))
def test_native_resolver_must_return_a_typed_owner_address(result):
    with pytest.raises(QueryBindingError, match="invalid owner address"):
        _bind(_document(), resolver=lambda *args: result)


def test_native_name_or_id_resolution_failure_is_not_an_absent_value():
    def reject(kind, operand):
        raise QueryBindingError("geom owner is absent: " + json.dumps(operand))

    with pytest.raises(QueryBindingError, match="geom owner is absent"):
        _bind(_document(), resolver=reject)


def test_schema_requires_closed_argument_names_and_canonical_json():
    binding, _ = _fixture()
    # Accepted texts are cached per process; a rejected text raises every time.
    for _ in range(2):
        with pytest.raises(QueryBindingError, match="closed properties"):
            ArgumentSchema.from_document({"type": "object", "properties": {}})
        with pytest.raises(QueryBindingError, match="not canonical"):
            ArgumentSchema(json.dumps(binding.arguments.document()))
        with pytest.raises(QueryBindingError):
            ArgumentSchema.from_document({"type": "object", "properties": {}, "additionalProperties": False,
                                          "minProperties": -1})
    assert ArgumentSchema(binding.arguments.canonical_json) == binding.arguments


def test_population_required_default_is_present_after_default_application():
    binding, _ = _fixture()
    explicit_schema = binding.arguments.document()
    explicit_schema["required"] = []
    binding = replace(binding, arguments=ArgumentSchema.from_document(explicit_schema),
                      parameters=tuple(replace(item, default_json="1.0") if item.name == "cutoff" else item
                                       for item in binding.parameters))
    # JSON Schema required applies to the bound object, including a declared
    # native default. It is not a requirement to spell that argument in YAML.
    document = _document()
    document["populations"] = []
    document["instances"] = []
    population, = _bind(document, binding=binding)
    assert population.arguments() == {"cutoff": 1.0}


@pytest.mark.parametrize("default", ("-1", '"wrong type"', "not JSON"))
def test_invalid_defaults_are_record_errors_before_binding(default):
    binding, declaration = _fixture()
    schema = binding.arguments.document()
    schema["required"] = []
    binding = replace(binding, arguments=ArgumentSchema.from_document(schema),
                      parameters=tuple(replace(item, default_json=default) if item.name == "cutoff" else item
                                       for item in binding.parameters))
    with pytest.raises(QueryBindingError):
        validate_query_binding(binding, declaration, binding.quantity_ids)


def test_default_subschema_local_reference_uses_the_root_schema():
    binding, declaration = _fixture()
    schema = binding.arguments.document()
    schema["$defs"] = {"cutoff": schema["properties"]["cutoff"]}
    schema["properties"]["cutoff"] = {"$ref": "#/$defs/cutoff"}
    schema["required"] = []
    binding = replace(binding, arguments=ArgumentSchema.from_document(schema),
                      parameters=tuple(replace(item, default_json="-1") if item.name == "cutoff" else item
                                       for item in binding.parameters))
    with pytest.raises(QueryBindingError, match="invalid declared default"):
        validate_query_binding(binding, declaration, binding.quantity_ids)


@pytest.mark.parametrize("construct", (
    lambda: ConfiguredQuery("native.query", '{"cutoff": 1}'),
    lambda: ConfiguredQuery("native.query", '{"cutoff":NaN}'),
    lambda: ConfiguredQuery("native.query", '[]'),
    lambda: ConfiguredQuery("native.query", '{}', instance_token="bad\n"),
    lambda: ConfiguredQuery("native.query", '{}', operands=(ResolvedOperand("geom", "geom", (1,)),)),
    lambda: ConfiguredQuery("native.query", '{}', operands=cast(Any, ("not typed",))),
    lambda: ResolvedOperand("geom", "geom", (True,)),
    lambda: ResolvedOperand("geom", "geom", cast(Any, [])),
))
def test_direct_resolved_subject_construction_does_not_bypass_validation(construct):
    with pytest.raises(QueryBindingError):
        construct()


def test_author_mapping_order_does_not_change_native_operand_order():
    document = _document()
    document["instances"][0]["operands"] = dict(reversed(tuple(document["instances"][0]["operands"].items())))
    assert _bind(document) == _bind(_document())
