"""Native argument rejection precedes the first owner lookup."""

from dataclasses import replace

import pytest

from specola.adapters.mujoco_queries import validate_mujoco_arguments
from specola.conformance.query_bindings import (
    ArgumentSchema, QueryBindingError, bind_query_configuration, validate_query_arguments,
)
from specola.query_config import QueryConfiguration
from test_query_bindings import _fixture, _document


@pytest.mark.parametrize("native_type,bad", (("integer", 1.0), ("integer", 2**31),
                                           ("integer", -(2**31)-1), ("number", 10**400)))
def test_last_native_argument_failure_precedes_first_name_resolution(native_type, bad):
    binding, declaration = _fixture()
    schema = ArgumentSchema.from_document({
        "type": "object", "additionalProperties": False,
        "properties": {"cutoff": {"type": native_type}}, "required": ["cutoff"],
    })
    binding = replace(binding, arguments=schema, population_arguments=schema)
    document = _document()
    document["populations"][0]["arguments"]["cutoff"] = 1
    document["instances"][0]["arguments"]["cutoff"] = 1
    document["instances"].append(dict(document["instances"][0], id="last",
                                     arguments={"cutoff": bad}))
    configuration = QueryConfiguration.from_document(document)
    calls = []
    with pytest.raises(QueryBindingError, match="not representable"):
        bind_query_configuration(configuration, (binding,), (declaration,), binding.quantity_ids,
                                 lambda *args: calls.append(args),
                                 validate_native_arguments=validate_mujoco_arguments)
    assert calls == []
    with pytest.raises(QueryBindingError, match="not representable"):
        validate_query_arguments(configuration, (binding,), (declaration,), binding.quantity_ids,
                                 validate_native_arguments=validate_mujoco_arguments)


def test_native_checker_receives_defaults_without_mutating_prepared_arguments():
    binding, declaration = _fixture()
    binding = replace(binding, parameters=tuple(
        replace(p, default_json="2") if p.name == "cutoff" else p for p in binding.parameters))
    schema = binding.arguments.document()
    schema["required"] = []
    binding = replace(binding, arguments=ArgumentSchema.from_document(schema))
    configuration = QueryConfiguration.empty()
    seen = []

    def inspect(record, arguments):
        seen.append((record.binding_id, dict(arguments)))
        arguments["cutoff"] = 99

    configured, = bind_query_configuration(configuration, (binding,), (declaration,), binding.quantity_ids,
                                           lambda *args: pytest.fail("unexpected resolver"),
                                           validate_native_arguments=inspect)
    assert seen == [(binding.binding_id, {"cutoff": 2})]
    assert configured.arguments() == {"cutoff": 2}


def test_native_checker_cannot_change_nested_configuration_arguments():
    binding, declaration = _fixture()
    schema = ArgumentSchema.from_document({
        "type": "object", "additionalProperties": False,
        "properties": {"cutoff": {"type": "array", "items": {"type": "number"}}},
        "required": ["cutoff"],
    })
    binding = replace(binding, arguments=schema, population_arguments=schema)
    document = _document()
    document["populations"][0]["arguments"]["cutoff"] = [1, 2]
    document["instances"] = []
    configuration = QueryConfiguration.from_document(document)

    def mutate(record, arguments):
        arguments["cutoff"][0] = 99

    configured, = bind_query_configuration(configuration, (binding,), (declaration,), binding.quantity_ids,
                                           lambda *args: pytest.fail("unexpected resolver"),
                                           validate_native_arguments=mutate)
    assert configured.arguments() == {"cutoff": [1, 2]}
    assert configuration.document() == document


@pytest.mark.parametrize("cutoff", (0.0, -0.0, -0.025, -0.1))
def test_clearance_population_refuses_nonpositive_cutoff_before_owner_lookup(cutoff):
    from specola.adapters.mujoco_queries import _binding_record
    from specola.conformance.core_runner import EvidenceIdentity
    from specola.conformance.query_bindings import CallableParameterSurface

    evidence = (EvidenceIdentity("owned-query-contract-fixture", "1" * 64),)
    binding = _binding_record(evidence)
    declaration = CallableParameterSurface(binding.callable_id,
        ("m", "d", "geom1", "geom2", "distmax", "fromto"), evidence)
    configuration = QueryConfiguration.from_document({
        "schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": cutoff}}],
    })
    with pytest.raises(QueryBindingError, match="less than or equal to the minimum of 0"):
        bind_query_configuration(configuration, (binding,), (declaration,), binding.quantity_ids,
            lambda *args: pytest.fail("invalid population reached native owner lookup"),
            validate_native_arguments=validate_mujoco_arguments)


@pytest.mark.parametrize("cutoff", (0.0, -0.0, -0.025, -0.1))
def test_explicit_raw_cutoff_remains_distinct_from_clearance_policy(cutoff):
    from specola.adapters.mujoco_queries import _binding_record
    from specola.conformance.core_runner import EvidenceIdentity
    from specola.conformance.query_bindings import CallableParameterSurface

    evidence = (EvidenceIdentity("owned-query-contract-fixture", "1" * 64),)
    binding = _binding_record(evidence)
    declaration = CallableParameterSurface(binding.callable_id,
        ("m", "d", "geom1", "geom2", "distmax", "fromto"), evidence)
    configuration = QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": 2.0}}],
        "instances": [{"id": "raw_cutoff", "binding": "mujoco.mj_geomDistance",
            "arguments": {"distmax": cutoff},
            "operands": {"geom1": {"kind": "geom", "id": 0},
                         "geom2": {"kind": "geom", "id": 1}}}],
    })
    validate_query_arguments(configuration, (binding,), (declaration,), binding.quantity_ids,
                             validate_native_arguments=validate_mujoco_arguments)
    import struct
    retained = configuration.document()["instances"][0]["arguments"]["distmax"]
    assert struct.pack("d", retained) == struct.pack("d", cutoff)
