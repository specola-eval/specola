"""Preserve completed scratch arrays with their native owner and stage meaning.

The source contracts describe a selected projection population. They do not
replace the producer inventory or establish full extraction readiness. Stored
native fields remain separate quantities at the original observation stage.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from .mujoco_fields import MujocoNativeFields
from ..conformance.core_runner import EvidenceIdentity
from ..conformance.extraction import QuantityOutcome, QuantityRecord, SourceAddress, quantity_metadata
from ..conformance.ledger import ExactnessKind
from ..conformance.query_bindings import ArgumentSchema, QueryBindingError
from ..extraction_observation import StructuredSample
from ..native_values import NativeValue


COMPLETION_ROOT = "mujoco.c.functions"


def callable_member_id(usr: str) -> str:
    """Map a clang function USR to its bounded-surface member ID."""
    if not usr.startswith("c:@F@") or "." in usr:
        raise QueryBindingError("completion callable is not a plain C function USR: " + usr)
    return COMPLETION_ROOT + "." + usr.removeprefix("c:@F@")


@dataclass(frozen=True)
class CompletedFieldValues:
    """Detached native arrays and the semantics required to interpret them."""

    metadata: tuple
    samples: tuple


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _record(fields):
    return _object({"kind": {"const": "record"}, "fields": {
        "type": "array", "minItems": len(fields), "maxItems": len(fields),
        "prefixItems": [{"type": "array", "minItems": 2, "maxItems": 2,
                         "prefixItems": [{"const": name}, layout]}
                        for name, layout in fields.items()]}})


def _sequence(item):
    return _object({"kind": {"const": "tuple"}, "items": {"type": "array", "items": item}})


def _scalar(kind, value_schema):
    return _object({"kind": {"const": kind}, "value": value_schema})


def _output_schema(contract, semantic_schema):
    """Declare the encoded record layout independently of captured values."""
    import numpy as np

    scalar = contract["native_type"].get("pointee") or contract["native_type"]["element"]
    dtype = np.dtype("f" + str(scalar["size"])).str
    dimensions = contract["binding_shape"]
    shape = {"type": "array", "minItems": len(dimensions), "maxItems": len(dimensions),
             "prefixItems": [{"const": int(value)} if str(value).isdecimal()
                             else {"type": "integer", "minimum": 0} for value in dimensions]}
    encoded_bytes = _object({"kind": {"const": "bytes"}, "data": {"type": "string"}})
    integer = _scalar("int", {"type": "string", "pattern": "^-?(0|[1-9][0-9]*)$"})
    text = _scalar("string", {"type": "string"})
    nullable_bytes = {"oneOf": [encoded_bytes, _object({"kind": {"const": "null"}})]}
    # These are the reviewed NativeLog result mappings. The independent native
    # fixture joins all eight members to C declarations and mutation controls.
    log = _record({"level": integer, "topic": integer, "subject": encoded_bytes,
                   "body": nullable_bytes, "func": nullable_bytes, "file": nullable_bytes,
                   "line": integer, "timestamp": _scalar("bool", {"type": "boolean"})})
    array = _object({"kind": {"const": "ndarray"}, "dtype": {"const": dtype},
                     "shape": shape, "data": {"type": "string"}})
    payload = {
        "value": array,
        "native_shape": _sequence(integer),
        "physical_owner_kind": {"const": NativeValue.capture(contract["physical_owner_kind"]).document()},
        "owner_indices": _sequence(integer), "native_path": _sequence(text),
        "completion": _sequence(text), "completion_warnings": _sequence(encoded_bytes),
        "completion_logs": _sequence(log),
        "configuration_identity": _record({"identity": text, "sha256": text}),
        "source_stage": {"const": NativeValue.capture("copied stored qacc and constraints; no new dynamics solution").document()},
    }
    if "stage_result" in contract:
        stage = contract["stage_result"]["immediate_stage_function"]
        stage_completion = _sequence(text)
        stage_completion["properties"]["items"].update({
            "contains": {"const": NativeValue.capture(stage).document()}, "minContains": 1, "maxContains": 1,
        })
        subsequent = [item["callable_id"].removeprefix("c:@F@")
                      for item in contract["completion"]["field_output_crosswalk"]
                      if item["materialized_path"] == "value"]
        stage_completion["properties"]["items"]["allOf"] = [
            {"not": {"contains": {"const": NativeValue.capture(name).document()}}} for name in subsequent]
        payload["stage_results"] = _record({stage: _record({
            "value": array, "completion": stage_completion,
            "warnings": _sequence(encoded_bytes), "logs": _sequence(log),
        })})
    schema = _record(payload)
    schema["x-specola-native-semantics"] = semantic_schema
    return ArgumentSchema.from_document(schema)


class MujocoCompletedFields:
    """Capture declared logical fields inside an active projection interval."""

    def __init__(self, model, guard):
        self._model, self._guard = model, guard
        path = Path(__file__).parent.parent / "data" / ("mujoco-projection-" + guard._version + ".json")
        self._document = json.loads(path.read_text())
        if self._document["version"] != guard._version:
            raise QueryBindingError("completed field producer version differs")
        self._contracts = {row["native_field_id"]: row for row in self._document["contracts"]}
        if len(self._contracts) != len(self._document["contracts"]):
            raise QueryBindingError("duplicate completed native field declaration")
        identities = [row["projection_id"] for row in self._contracts.values()]
        if len(identities) != len(set(identities)):
            raise QueryBindingError("duplicate completed quantity identity")
        self.quantities = self._quantities()
        self.metadata = tuple(quantity_metadata(quantity) for quantity in self.quantities)

    def _quantities(self):
        provenance = tuple(EvidenceIdentity(**item) for item in self._document["sources"])
        provenance += tuple(EvidenceIdentity(item["name"], item["sha256"]) for item in (
            self._document["source_artifact"], self._document["storage"]["source_artifact"]))
        result = []
        for contract in self._contracts.values():
            # The crosswalk names callables by clang USR (``c:@F@name``). The
            # bounded producer surface addresses the same callables under the
            # ``mujoco.c.functions`` root, so sources use that member spelling.
            sources = tuple(SourceAddress(callable_member_id(row["callable_id"]), row["result_path"])
                            for row in contract["completion"]["field_output_crosswalk"])
            schema = {
                "native_field_id": contract["native_field_id"],
                "value": {"native_type": contract["native_type"],
                          "binding_shape": contract["binding_shape"],
                          "native_shape": contract["native_shape"]},
                "physical_owner_kind": contract["physical_owner_kind"],
                "owner_indices": "native model index order on the first axis; one simulation owner for embedded energy",
                "owner_extent_header_id": contract["owner_extent_header_id"],
                "owner_enum": contract["owner_enum"], "components": contract["components"],
                "completion": contract["completion"], "freshness": contract["freshness"],
                "availability": contract["availability"],
                "completion_warnings": "owned native warning C-string bytes in callback order",
                "completion_logs": "owned native log records in callback order, with each declared result component",
                "configuration_identity": "provider-authenticated model/configuration identity; physical units retain producer-selected L/M/T",
                "source_stage": "copied qacc, contacts and constraint results; completion does not solve new dynamics",
                "scalar_projection": "structured-only; legacy scalar projections have separate declarations",
            }
            for key in ("stored_input_contract", "native_energy_branches", "stage_result"):
                if key in contract:
                    schema[key] = contract[key]
            result.append(QuantityRecord(
                quantity_id=contract["projection_id"], sources=sources,
                native_meaning=contract["native_meaning"],
                owner_resolver_id="mujoco_completed_native_root_field",
                index_domain=json.dumps({"root": "mjData", "physical_owner": contract["physical_owner_kind"],
                                         "extent_field": contract["owner_extent_header_id"],
                                         "native_shape": contract["native_shape"]}, sort_keys=True),
                unit=json.dumps({"system": contract["unit_system"], "components": contract["components"]},
                                sort_keys=True, separators=(",", ":")),
                frame=json.dumps({"axes": contract["frame"], "reference": contract["frame_reference"]},
                                 sort_keys=True, separators=(",", ":")),
                exactness=ExactnessKind.EXACT,
                availability=json.dumps(contract["availability"], sort_keys=True), freshness=contract["freshness"],
                transformation="Copy the complete native array after the declared scratch completion stages; retain native owner order, dtype and bits.",
                output_sink="structured", property_pattern="structured-only",
                metadata_resolver_id="mujoco_completed_field_source_contract",
                reader_id="specola.adapters.mujoco_completed_fields.MujocoCompletedFields.capture",
                provenance=provenance, owner_kind="mjData",
                outcomes=(QuantityOutcome("completed_from_stored_state",
                          "The declared scratch stages returned under their source and effect guards; copied solver state is not freshly solved.",
                          _output_schema(contract, schema), provenance),),
            ))
        return tuple(result)

    def capture_stage(self, stage, snapshot, completed, warning_start, log_start):
        """Own declared intermediate output before the next native call."""
        if not completed or completed[-1] != stage:
            raise QueryBindingError("intermediate result has the wrong completion boundary")
        results = []
        for identity, contract in self._contracts.items():
            declared = contract.get("stage_result")
            if declared is None or declared["immediate_stage_function"] != stage:
                continue
            if declared["native_field_id"] != identity or declared["output_field_name"] != contract["field"]:
                raise QueryBindingError("intermediate result native address differs")
            results.append((identity, NativeValue.capture({
                "value": getattr(snapshot.data, contract["field"]),
                "completion": completed,
                "warnings": tuple(item.message for item in snapshot.warnings[warning_start:]),
                "logs": tuple(asdict(item) for item in snapshot.logs[log_start:]),
            })))
        return tuple(results)

    def capture(self, snapshot, configuration_identity):
        """Copy logical array elements before the borrowed scratch data expires.

        The surrounding projection authenticates completion, allocation
        initialization and effects. Layout checks do not independently prove
        these conditions or the physical consistency of copied solver inputs.
        """
        if (type(configuration_identity) is not dict
                or set(configuration_identity) != {"identity", "sha256"}
                or type(configuration_identity["identity"]) is not str
                or not configuration_identity["identity"].strip()
                or type(configuration_identity["sha256"]) is not str
                or re.fullmatch("[0-9a-f]{64}", configuration_identity["sha256"]) is None):
            raise QueryBindingError("completed fields require a declared model configuration identity")
        completed = tuple(snapshot.completed)
        expected_stage_fields = {identity for identity, contract in self._contracts.items() if "stage_result" in contract}
        stage_results = getattr(snapshot, "stage_results", {})
        if set(stage_results) != expected_stage_fields:
            raise QueryBindingError("completed intermediate field population differs")
        for contract in self._contracts.values():
            required = tuple(identity.removeprefix("c:@F@")
                             for identity in contract["completion"]["function_header_ids"])
            if any(stage not in completed for stage in required):
                raise QueryBindingError("completed field result stage was not called")
        reader = MujocoNativeFields(self._model, snapshot.data, self._guard,
                                   _profile=self._document["storage"])
        try:
            # Resolve owner domains from declared count fields before reading
            # elements. The native fixture uses a separate C owner oracle.
            populations = {}
            for identity, contract in self._contracts.items():
                count_field = contract["count_field"]
                count = reader._root_integer("mjModel_", count_field) if count_field else 1
                if count < 0:
                    raise QueryBindingError("completed field owner extent is negative")
                populations[identity] = tuple(range(count))
            values = reader.read()
        finally:
            reader.close()
        if {item.field_id for item in values} != self._contracts.keys() or len(values) != len(self._contracts):
            raise QueryBindingError("completed native result field population differs")
        warnings = tuple(item.message for item in snapshot.completion_warnings)
        logs = tuple(asdict(item) for item in snapshot.completion_logs)
        samples = []
        for item in values:
            contract = self._contracts[item.field_id]
            value = item.value.restore()
            owner_indices = populations[item.field_id]
            if contract["count_field"] and value.shape[0] != len(owner_indices):
                raise QueryBindingError("completed native array omitted a physical owner")
            payload = {
                "value": value, "native_shape": item.native_shape,
                "physical_owner_kind": contract["physical_owner_kind"],
                "owner_indices": owner_indices, "native_path": item.native_path,
                "completion": completed, "completion_warnings": warnings,
                "completion_logs": logs,
                "configuration_identity": {"identity": configuration_identity["identity"],
                                           "sha256": configuration_identity["sha256"]},
                "source_stage": "copied stored qacc and constraints; no new dynamics solution",
            }
            if "stage_result" in contract:
                stage = contract["stage_result"]["immediate_stage_function"]
                stage_values = stage_results[item.field_id]
                if set(stage_values) != {stage}:
                    raise QueryBindingError("completed intermediate stage population differs")
                stage_payload = stage_values[stage].restore()
                prefix = completed[:completed.index(stage) + 1]
                stage_value = stage_payload["value"]
                if (stage_payload["completion"] != prefix
                        or stage_value.dtype != value.dtype or stage_value.shape != value.shape):
                    raise QueryBindingError("completed intermediate stage or native layout differs")
                payload["stage_results"] = {stage: stage_payload}
            samples.append(StructuredSample(contract["projection_id"], "mjData", (),
                                             NativeValue.capture(payload), "completed_from_stored_state"))
        self._guard.check()
        return CompletedFieldValues(self.metadata, tuple(samples))
