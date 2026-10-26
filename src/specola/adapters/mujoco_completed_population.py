"""Resolve completed-field instances before capturing native array elements.

This supplies the quantity-to-instance portion of a prepared provider subject.
It does not classify the producer API, select callable coverage edges, or
establish native fidelity. Native tests use an independent owner/address oracle.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from .mujoco_fields import MujocoNativeFields
from ..conformance.instances import ExpectedInstance, ExpectedOutcome, QuantityDomain, QueryPopulationDomain
from ..conformance.query_bindings import ArgumentSchema, QueryBindingError
from ..native_values import NativeValue


@dataclass(frozen=True)
class CompletedPopulation:
    """Existing conformance records for one declared completed-field subset."""

    domains: tuple
    instances: tuple
    query_domains: tuple


def _field_layout_pair(schema, name):
    pairs = schema["properties"]["fields"]["prefixItems"]
    found = [pair for pair in pairs if pair["prefixItems"][0] == {"const": name}]
    if len(found) != 1:
        raise QueryBindingError("completed result has no unique declared component: " + name)
    return found[0]["prefixItems"]


def _replace_field_layout(schema, name, layout):
    _field_layout_pair(schema, name)[1] = layout


def completed_population(model, data, guard, quantities, configuration_identity, quantity_bindings):
    """Build exact owner/shape constraints without reading quantity payloads.

    The authenticated provider supplies the actual configuration and the
    independently declared producing binding for each quantity. No binding
    is inferred from a result sample or selected by a specification scope.
    The provider retains its own runtime post-read custody check.
    """
    path = Path(__file__).parent.parent / "data" / ("mujoco-projection-" + guard._version + ".json")
    document = json.loads(path.read_text())
    if document["version"] != guard._version:
        raise QueryBindingError("completed population producer version differs")
    contracts = {row["projection_id"]: row for row in document["contracts"]}
    declared = {quantity.quantity_id: quantity for quantity in quantities}
    if (len(contracts) != len(document["contracts"]) or len(declared) != len(quantities)
            or declared.keys() != contracts.keys() or set(quantity_bindings) != contracts.keys()):
        raise QueryBindingError("completed population quantity or binding denominator differs")
    if any(type(binding) is not str or not binding for binding in quantity_bindings.values()):
        raise QueryBindingError("completed population requires its declared producing bindings")
    if (type(configuration_identity) is not dict
            or set(configuration_identity) != {"identity", "sha256"}):
        raise QueryBindingError("completed population requires its prepared configuration identity")
    from ..conformance.extraction import _evidence
    from ..conformance.core_runner import EvidenceIdentity

    _evidence(EvidenceIdentity(**configuration_identity), "completed population configuration")
    configuration = {"identity": configuration_identity["identity"], "sha256": configuration_identity["sha256"]}
    reader = MujocoNativeFields(model, data, guard, _profile=document["storage"])
    try:
        # These are root count and address reads, not calls to reader.read().
        # Exact native allocation and pointer bounds are checked before the
        # later capture can dereference any completed numeric field.
        regions = reader._field_regions()
        if regions != reader._bindings:
            raise QueryBindingError("completed owner layout changed during population resolution")
        layouts = {}
        for (field_id, root, native_path), (_, _, native_shape, binding_shape, dtype) in regions.items():
            if root != "mjData_" or field_id in layouts:
                raise QueryBindingError("completed population has an ambiguous native root field")
            layouts[field_id] = (native_path, native_shape, binding_shape, dtype)
        populations = {}
        for quantity, contract in contracts.items():
            count_field = contract["count_field"]
            count = reader._root_integer("mjModel_", count_field) if count_field else 1
            if count < 0:
                raise QueryBindingError("completed population owner extent is negative")
            populations[quantity] = tuple(range(count))
        if (reader._buffer_domains() != reader._buffers or reader._field_regions() != regions):
            raise QueryBindingError("completed allocation or owner domain changed during resolution")
        reader._validate_subject()
    finally:
        reader.close()
    if layouts.keys() != {contract["native_field_id"] for contract in contracts.values()}:
        raise QueryBindingError("completed population omitted a native field layout")
    domains, instances, query_domains = [], [], []
    for identity, contract in contracts.items():
        quantity = declared[identity]
        native_path, native_shape, binding_shape, dtype = layouts[contract["native_field_id"]]
        if contract["count_field"] and native_shape[0] != len(populations[identity]):
            raise QueryBindingError("completed array and physical owner extents differ")
        outcome, = quantity.outcomes
        if outcome.outcome_id != "completed_from_stored_state" or quantity.owner_kind != "mjData":
            raise QueryBindingError("completed population result meaning differs")
        schema = outcome.native_schema.document()
        value_layout = {
            "type": "object", "properties": {
                "kind": {"const": "ndarray"}, "dtype": {"const": dtype},
                "shape": {"const": list(binding_shape)}, "data": {"type": "string"}},
            "required": ["kind", "dtype", "shape", "data"], "additionalProperties": False,
        }
        _replace_field_layout(schema, "value", value_layout)
        if "stage_result" in contract:
            stages_layout = _field_layout_pair(schema, "stage_results")[1]
            stage_layout = _field_layout_pair(stages_layout, contract["stage_result"]["immediate_stage_function"])[1]
            _replace_field_layout(stage_layout, "value", value_layout)
        for name, value in (
            ("native_shape", native_shape), ("native_path", native_path),
            ("owner_indices", populations[identity]),
            ("physical_owner_kind", contract["physical_owner_kind"]),
            ("configuration_identity", configuration),
        ):
            _replace_field_layout(schema, name, {"const": NativeValue.capture(value).document()})
        rule = json.dumps({"root": "mjData", "native_field_id": contract["native_field_id"],
                           "physical_owner": contract["physical_owner_kind"],
                           "owner_extent_header_id": contract["owner_extent_header_id"],
                           "native_shape": native_shape, "binding_shape": binding_shape}, sort_keys=True)
        binding = quantity_bindings[identity]
        domains.append(QuantityDomain(identity, rule, quantity.provenance))
        instances.append(ExpectedInstance(identity, "mjData", (), "", None,
                         (ExpectedOutcome(outcome.outcome_id, ArgumentSchema.from_document(schema)),),
                         binding_id=binding))
        # The root field remains populated when its contained owner axis is
        # empty; its typed zero-extent value is still a required observation.
        query_domains.append(QueryPopulationDomain(binding, identity, rule, quantity.provenance))
    return CompletedPopulation(tuple(domains), tuple(instances), tuple(query_domains))
