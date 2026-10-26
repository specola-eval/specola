"""Read source-declared MuJoCo spatial queries into typed native values.

These eight families are part of the producer programme, not its denominator.
The caller supplies callable identities from the independent producer surface.
Object-type branches have separate bindings so each native owner population
receives an explicit frame argument without inventing a producer default.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from graphlib import TopologicalSorter
import hashlib
import json
from pathlib import Path

import numpy as np

from .mujoco_native import MujocoNativeGuard
from .mujoco_scratch import MujocoScratchSession
from .mujoco_queries import validate_mujoco_arguments
from .mujoco_query_metadata import MujocoQueryMetadata
from ..conformance.core_runner import EvidenceIdentity
from ..conformance.native_images import PinnedNativeFile
from ..conformance.query_bindings import (
    ArgumentSchema, ParameterRole, QueryBindingRecord, QueryBindingError,
    QueryParameter, bind_query_configuration, validate_query_binding,
    validate_query_arguments,
)
from ..extraction_observation import StructuredMetadata, StructuredSample
from ..native_values import NativeValue


@dataclass(frozen=True)
class SpatialQueryPlan:
    """A reviewed semantic mapping joined to an actual callable declaration."""

    binding: QueryBindingRecord
    contract: dict
    owner_kind: str
    count_field: str
    completion: tuple
    native_object_type: object = None


def spatial_bindings(version, declarations):
    """Join pinned parameter/result semantics to independent surface identities."""
    path = Path(__file__).parent.parent / "data" / ("mujoco-spatial-queries-" + version + ".json")
    if not path.is_file():
        raise QueryBindingError("unresolved native spatial-query source version")
    document = json.loads(path.read_text())
    plans = []
    for contract in document["contracts"]:
        function = contract["function"]
        if function not in declarations:
            raise QueryBindingError("native spatial query has no independent source declaration: " + function)
        declaration = declarations[function]
        quantity_ids = tuple(q["quantity_id"] for q in contract["quantities"])
        output_names = tuple(o["native_parameter_name"] for o in contract["outputs"])
        if (len(output_names) != len(set(output_names))
                or tuple(q.rsplit(".", 1)[1] for q in quantity_ids) != output_names):
            raise QueryBindingError("native spatial result-to-quantity mapping differs")
        provenance = tuple(EvidenceIdentity(**source) for source in contract["source"])
        population = contract["population"]
        branches = population["object_types"] or {None: population}
        for branch_name, branch in branches.items():
            binding_id = "mujoco.query." + function + ("." + branch_name if branch_name else "")
            owner_kind = branch_name or branch["owner_kind"]
            count_field = (branch["population"] if branch_name else branch)["count_field"]
            schema = json.loads(json.dumps(contract["argument_schema"]))
            roles = []
            for parameter in contract["parameters"]:
                role = ParameterRole(parameter["role"])
                if branch_name and parameter["name"] == "objtype":
                    role = ParameterRole.CONTEXT
                    del schema["properties"]["objtype"]
                    schema["required"].remove("objtype")
                roles.append(QueryParameter(parameter["name"], role, parameter["native_type"],
                                            owner_kind=owner_kind if role is ParameterRole.OWNER else None))
            argument_schema = ArgumentSchema.from_document(schema)
            completion = list(contract["completion_dependencies"])
            for dependency in contract["conditional_completion_dependencies"]:
                if dependency["when"] == "objtype==" + str(branch_name):
                    completion.insert(completion.index(dependency["after"]) + 1, dependency["call"])
            binding = QueryBindingRecord(
                binding_id, declaration.callable_id, tuple(roles), argument_schema, quantity_ids,
                "native_mujoco_" + owner_kind + "_name_or_id", "native_mujoco_query_" + function,
                "Detached evaluation from copied stored state after the declared stages; qacc is not solved.",
                "Synchronized detached data, guarded native code, captured TLS warnings and preserved fenv.",
                json.dumps(contract["outputs"], sort_keys=True),
                "Preserve complete native arrays, nonfinite payloads and typed empty DOF domains.",
                provenance, tuple(completion), "native_mujoco_" + owner_kind + "_population", argument_schema)
            validate_query_binding(binding, declaration, quantity_ids)
            plans.append(SpatialQueryPlan(binding, contract, owner_kind, count_field,
                                          tuple(completion), branch.get("enum_value")))
    return tuple(plans)


class MujocoSpatialQueryReader:
    """Execute configured native populations with owned results and metadata."""

    def __init__(self, model, data, configuration, declarations, *,
                 _guard=None, _session=None, _defer_configuration=False):
        self._held = ExitStack()
        self._closed = False
        try:
            self._mujoco = import_module("mujoco")
            self._model = model
            self.plans = spatial_bindings(self._mujoco.__version__, declarations)
            self._plans = {plan.binding.binding_id: plan for plan in self.plans}
            self.bindings = tuple(plan.binding for plan in self.plans)
            quantity_ids = tuple(dict.fromkeys(q for binding in self.bindings for q in binding.quantity_ids))
            declared = tuple(declarations[name] for name in dict.fromkeys(plan.contract["function"] for plan in self.plans))
            if not _defer_configuration:
                validate_query_arguments(configuration, self.bindings, declared, quantity_ids,
                                         validate_native_arguments=validate_mujoco_arguments)
            self._guard = _guard if _guard is not None else self._held.enter_context(MujocoNativeGuard(self._mujoco))
            self._session = _session if _session is not None else self._held.enter_context(MujocoScratchSession(model, data, self._guard))
            self._functions = {plan.contract["function"]: self._guard.function(plan.contract["function"])
                               for plan in self.plans}
            distribution = metadata.distribution("mujoco")
            headers = {(plan.contract["header"]["relative_path"], plan.contract["header"]["sha256"])
                       for plan in self.plans}
            self._headers = tuple(self._held.enter_context(PinnedNativeFile(distribution.locate_file(path), sha))
                                  for path, sha in headers)
            self._native_metadata = MujocoQueryMetadata(model, self._mujoco)
            self._dimensions = {plan.count_field: self._dimension(plan.count_field) for plan in self.plans}
            self._dimensions["mjModel.nv"] = self._dimension("mjModel.nv")
            # Native routines use signed int indices and 3*nv arithmetic.
            if self._dimensions["mjModel.nv"] > np.iinfo(np.int32).max // 3:
                raise QueryBindingError("native spatial query integer extent is unresolved")
            for plan in self.plans:
                if plan.native_object_type is not None:
                    if int(getattr(self._mujoco.mjtObj, plan.owner_kind)) != plan.native_object_type:
                        raise QueryBindingError("native object enumeration and source declaration differ")
            self.configured = None
            if not _defer_configuration:
                with self._session.observe(data):
                    configured = bind_query_configuration(configuration, self.bindings,
                        declared, quantity_ids, self._resolve_owner,
                        validate_native_arguments=validate_mujoco_arguments)
                self._install_configuration(configuration, configured)
            stages = TopologicalSorter()
            for plan in self.plans:
                for index, stage in enumerate(plan.completion):
                    stages.add(stage, *plan.completion[:index])
            self._completion = tuple(stages.static_order())
            self.metadata = self._metadata()
        except BaseException:
            self.close()
            raise

    def _install_configuration(self, configuration, configured):
        if any(item.binding_id not in self._plans for item in configured):
            raise QueryBindingError("spatial reader received another binding's configured query")
        self.configuration = configuration
        self.configured = tuple(configured)

    def _dimension(self, field):
        if not field.startswith("mjModel."):
            raise QueryBindingError("spatial query dimension has no native model declaration")
        value = int(getattr(self._model, field.split(".", 1)[1]))
        if not 0 <= value <= np.iinfo(np.int32).max:
            raise QueryBindingError("native spatial query dimension is outside signed integer range")
        return value

    def _resolve_owner(self, kind, operand):
        plan = next((plan for plan in self.plans if plan.owner_kind == kind), None)
        if plan is None:
            raise QueryBindingError("native spatial query owner kind is unresolved")
        if "name" in operand:
            # XBODY has the same native name/address domain as BODY; the query
            # operand and structured address retain the distinct XBODY type.
            name_type = "mjOBJ_BODY" if kind == "mjOBJ_XBODY" else kind
            if not name_type.startswith("mjOBJ_"):
                name_type = "mjOBJ_" + name_type.upper()
            index = int(self._guard.function("mj_name2id")(
                self._model, getattr(self._mujoco.mjtObj, name_type), operand["name"]))
        else:
            index = operand["id"]
        if type(index) is not int or not 0 <= index < self._dimensions[plan.count_field]:
            raise QueryBindingError("native spatial query owner is absent or out of bounds")
        return (index,)

    def _metadata(self):
        records = {}
        for plan in self.plans:
            for quantity, output in zip(plan.contract["quantities"], plan.contract["outputs"]):
                records[quantity["quantity_id"]] = StructuredMetadata(
                    quantity["quantity_id"], quantity["native_meaning"],
                    json.dumps({"value": {"native_shape": output["schema_shape"], "dtype": "mjtNum"},
                                "arguments": "exact native arguments", "warnings": "native byte strings",
                                "completion_warnings": "native byte strings", "completion": "ordered native stages",
                                "logs": "owned mjLogMessage fields when declared by the producer",
                                "completion_logs": "owned mjLogMessage fields when declared by the producer",
                                "native_model_metadata": self._native_metadata.value.document(),
                                "native_model_metadata_sha256": self._native_metadata.value.sha256,
                                "state_context": "owned stored qpos/qvel/qacc/time and separately completed xaxis/xmat",
                                "configuration_sha256": "complete immutable query configuration identity",
                                "owner_name": "native bytes or explicit absence; identity remains native indices"}),
                    quantity["unit"], quantity["frame"], quantity["transformation"],
                    (plan.binding.callable_id,), hashlib.sha256(json.dumps(
                        plan.contract, sort_keys=True).encode()).hexdigest())
        return tuple(records.values())

    def read(self, data):
        """Return structured-only results for each configured native owner."""
        if self._closed:
            raise QueryBindingError("native spatial query reader is closed")
        self._validate_current_subject()
        with self._session.observe(data, completion=self._completion) as snapshot:
            return self._read_snapshot(snapshot)

    def _validate_current_subject(self):
        self._native_metadata.validate()
        for header in self._headers:
            header.validate()
        if any(self._dimension(field) != size for field, size in self._dimensions.items()):
            raise QueryBindingError("native spatial query owner or DOF domain changed; reset required")
        for name, function in self._functions.items():
            if function is not self._guard.function(name):
                raise QueryBindingError("native spatial query reader identity changed")

    def _read_snapshot(self, snapshot):
        """Capture owned results within the caller's shared scratch interval."""
        data = snapshot.data
        if self._closed or self.configured is None:
            raise QueryBindingError("native spatial query reader is closed or unconfigured")
        if data is not self._session._scratch or not set(self._completion).issubset(snapshot.completed):
            raise QueryBindingError("spatial query requires this session's completed scratch")
        self._validate_current_subject()
        samples = []
        expected_addresses = set()
        completion_warnings = tuple(warning.message for warning in snapshot.completion_warnings)
        completion_logs = tuple(asdict(log) for log in snapshot.completion_logs)
        # Keep the stored inputs and the completed coordinate bases separate.
        # These are core state inputs, not a claim of complete MjData capture.
        state_context = {
            "stored": {name: getattr(snapshot.source, name) for name in ("qpos", "qvel", "qacc", "time")},
            "completed": {name: getattr(snapshot.data, name) for name in ("xaxis", "xmat")},
            "acceleration_evaluation": "copied stored qacc; no new dynamics solution",
        }
        for configured in self.configured:
            plan = self._plans[configured.binding_id]
            owners = (range(self._dimensions[plan.count_field]) if configured.instance_token is None
                      else tuple(operand.native_index[0] for operand in configured.operands))
            arguments = configured.arguments()
            if plan.native_object_type is not None:
                arguments["objtype"] = plan.native_object_type
            for owner in owners:
                owner_index = ((plan.native_object_type, owner) if plan.native_object_type is not None else (owner,))
                owner_kind = "native_object" if plan.native_object_type is not None else plan.owner_kind
                for quantity in plan.binding.quantity_ids:
                    expected_addresses.add((quantity, owner_kind, owner_index,
                                            configured.instance_token or "", configured.binding_id))
                outputs = {o["native_parameter_name"]: np.empty(tuple(
                    self._dimensions[axis] if type(axis) is str else axis for axis in o["schema_shape"]),
                    dtype=data.qpos.dtype) for o in plan.contract["outputs"]}
                native_args = []
                for parameter in plan.binding.parameters:
                    if parameter.role is ParameterRole.OWNER:
                        native_args.append(owner)
                    elif parameter.role is ParameterRole.OUTPUT_BUFFER:
                        native_args.append(outputs[parameter.name])
                    elif parameter.name == "m":
                        native_args.append(self._model)
                    elif parameter.name == "d":
                        native_args.append(snapshot.data)
                    else:
                        native_args.append(arguments[parameter.name])
                start = len(snapshot.warnings)
                log_start = len(snapshot.logs)
                self._functions[plan.contract["function"]](*native_args)
                warnings = tuple(warning.message for warning in snapshot.warnings[start:])
                logs = tuple(asdict(log) for log in snapshot.logs[log_start:])
                for quantity, output in zip(plan.binding.quantity_ids, plan.contract["outputs"]):
                    value = outputs[output["native_parameter_name"]]
                    availability = ("native_nonfinite" if not np.isfinite(value).all() else
                                    "empty_native_dof_domain" if not value.size else "native_value")
                    samples.append(StructuredSample(quantity, owner_kind, owner_index,
                        NativeValue.capture({"value": value, "arguments": arguments, "warnings": warnings,
                                             "completion_warnings": completion_warnings,
                                             "logs": logs, "completion_logs": completion_logs,
                                             "completion": snapshot.completed,
                                             "state_context": state_context,
                                             "configuration_sha256": self.configuration.sha256,
                                             "native_model_metadata_sha256": self._native_metadata.value.sha256,
                                             "owner_name": self._native_metadata.owner_name(plan.owner_kind, owner)}), availability,
                        instance_token=configured.instance_token or "", binding_id=configured.binding_id))
        addresses = [sample.address for sample in samples]
        if len(addresses) != len(set(addresses)) or set(addresses) != expected_addresses:
            raise QueryBindingError("native spatial query materialization population differs")
        for header in self._headers:
            header.validate()
        return tuple(samples)

    def close(self):
        """Retire owned scratch and native custody resources."""
        if not self._closed:
            try:
                self._held.close()
            finally:
                self._closed = True
