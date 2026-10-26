"""Declared query arguments and resolved subjects, separate from execution.

The binder validates author configuration before resolving native operands.
It never calls a query or derives its population from specification scope.
Declaration validity does not establish producer semantics or completeness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache, wraps
import json
from typing import Callable, Dict, Optional, Tuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable

from ..query_config import QueryConfiguration, QueryConfigurationError, _json_tree, _validator
from .core_runner import EvidenceIdentity
from .extraction import (
    _partition, _provenance, _record_values, _restore_evidence, _restore_items, _text, _tuple,
    validate_extraction_manifest,
)
from .ledger import ExactnessKind
from .surface import MemberKind, _keys


class QueryBindingError(QueryConfigurationError):
    """A binding or configured native query subject is inconsistent."""


class QueryConfigurationRequired(QueryBindingError):
    """Finite-population arguments must be supplied before observation."""

    code = "query_configuration_required"

    def __init__(self, missing: Tuple[Tuple[str, Tuple[str, ...]], ...]):
        self.missing = missing
        super().__init__(self.code + ": " + "; ".join(binding + " requires " + ", ".join(names)
                                                      for binding, names in missing))


def _binding_errors(function):
    """Keep declaration and syntax failures in the query error domain."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except QueryBindingError:
            raise
        except (ValueError, TypeError, RecursionError, SchemaError, Unresolvable) as error:
            raise QueryBindingError(str(error)) from error
    return checked


class ParameterRole(str, Enum):
    CONTEXT = "context"
    OWNER = "owner"
    ARGUMENT = "argument"
    OUTPUT_BUFFER = "output_buffer"


def _canonical(document) -> str:
    _json_tree(document)
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


@lru_cache()
def _compiled_argument_schema(canonical_json: str):
    """Reuse compiled schemas with the standard library's bounded cache."""
    return Draft202012Validator(json.loads(canonical_json), registry=Registry())


@lru_cache(maxsize=4096)
def _check_schema_text(canonical_json: str) -> None:
    """Accept each distinct canonical schema text once per process.

    Providers rebuild identical schema records on every observation; the
    metaschema check dominated that cost (58 calls, 1.86 s of a 3.93 s
    profiled fixture read; session capture 2026-09-07, profile-provider.log),
    and after it the canonical round trip (23 records, about 11 ms of a
    104 ms LIBERO observation, 2026-10-08). Every check reads only the
    immutable text, so a cached acceptance is exact; a rejected text is not
    cached and raises again.
    """
    document = json.loads(canonical_json)
    if _canonical(document) != canonical_json:
        raise QueryBindingError("argument schema is not canonical JSON")
    Draft202012Validator.check_schema(document)
    if (type(document) is not dict or document.get("type") != "object"
            or document.get("additionalProperties") is not False
            or type(document.get("properties")) is not dict):
        raise QueryBindingError("query argument schema must declare an object with closed properties")


@dataclass(frozen=True)
class ArgumentSchema:
    """A detached JSON Schema with no external reference retrieval."""

    canonical_json: str

    @_binding_errors
    def __post_init__(self):
        if type(self.canonical_json) is not str:
            raise QueryBindingError("argument schema must be canonical JSON text")
        _check_schema_text(self.canonical_json)

    @classmethod
    @_binding_errors
    def from_document(cls, document: dict) -> ArgumentSchema:
        return cls(_canonical(document))

    def document(self) -> dict:
        return json.loads(self.canonical_json)

    @_binding_errors
    def validate(self, arguments: dict, binding_id: str, *, allow_missing_arguments: bool = False) -> None:
        _json_tree(arguments)
        validator = _compiled_argument_schema(self.canonical_json)
        for error in validator.iter_errors(arguments):
            if allow_missing_arguments and error.validator == "required" and not error.absolute_path:
                continue
            path = ".".join(map(str, error.absolute_path)) or "arguments"
            raise QueryBindingError(binding_id + ": " + path + ": " + error.message)


@dataclass(frozen=True)
class QueryParameter:
    """One native parameter's explicit role in a query binding."""

    name: str
    role: ParameterRole
    native_schema: str
    owner_kind: Optional[str] = None
    default_json: Optional[str] = None


@dataclass(frozen=True)
class CallableParameterSurface:
    """Producer-derived parameter order, independent of the binding record."""

    callable_id: str
    parameter_names: Tuple[str, ...]
    provenance: Tuple[EvidenceIdentity, ...]


@dataclass(frozen=True)
class QueryBindingRecord:
    """A callable-backed result contract, not an eager MjData field record."""

    binding_id: str
    callable_id: str
    parameters: Tuple[QueryParameter, ...]
    arguments: ArgumentSchema
    quantity_ids: Tuple[str, ...]
    owner_resolver_id: str
    reader_id: str
    freshness: str
    effect_policy: str
    result_mapping: str
    outcome_policy: str
    provenance: Tuple[EvidenceIdentity, ...]
    completion_dependencies: Tuple[str, ...] = ()
    population_resolver_id: Optional[str] = None
    population_arguments: Optional[ArgumentSchema] = None


def query_binding_document(binding: QueryBindingRecord, declaration: CallableParameterSurface) -> dict:
    """Keep a query contract and its independent native parameter census."""
    return {"schema": "specola.query-binding.v1", "binding": asdict(binding),
            "declaration": asdict(declaration)}


@_binding_errors
def query_binding_from_document(document, quantity_ids: Tuple[str, ...]):
    """Restore declarations without resolving an owner or invoking a query.

    The supplied quantity population belongs to the manifest. This validates
    the existing parameter/output references; the complete API/result relation
    and runtime population still require their own validation.
    """
    _keys(document, {"schema", "binding", "declaration"}, "query binding document")
    if document["schema"] != "specola.query-binding.v1":
        raise QueryBindingError("unsupported query binding schema")

    def parameter(record):
        values = _record_values(QueryParameter, record)
        values["role"] = ParameterRole(values["role"])
        return QueryParameter(**values)

    def schema(record):
        return ArgumentSchema(**_record_values(ArgumentSchema, record))

    values = _record_values(QueryBindingRecord, document["binding"])
    values["parameters"] = _restore_items(values["parameters"], parameter)
    values["arguments"] = schema(values["arguments"])
    values["quantity_ids"] = _restore_items(values["quantity_ids"], lambda value: value)
    values["provenance"] = _restore_evidence(values["provenance"])
    values["completion_dependencies"] = _restore_items(values["completion_dependencies"], lambda value: value)
    if values["population_arguments"] is not None:
        values["population_arguments"] = schema(values["population_arguments"])
    binding = QueryBindingRecord(**values)
    values = _record_values(CallableParameterSurface, document["declaration"])
    values["parameter_names"] = _restore_items(values["parameter_names"], lambda value: value)
    values["provenance"] = _restore_evidence(values["provenance"])
    declaration = CallableParameterSurface(**values)
    validate_query_binding(binding, declaration, quantity_ids)
    return binding, declaration


def materialized_query_outputs(surface, manifest):
    """Derive callable outputs from a validated manifest's direct read edges.

    Quantity sources also retain transitive alias provenance. Those aliases do
    not require executing a callable that is represented through other storage.
    Approximate outputs use directly materialized sources but cannot discharge
    the separate exact native coverage relation.
    """
    by_quantity = {}
    for link in manifest.coverage:
        if link.target_quantity is not None:
            by_quantity.setdefault(link.target_quantity, set()).add(link.source)
    direct = {source for sources in by_quantity.values() for source in sources}
    outputs = {member.member_id: set() for member in surface.members if member.kind is MemberKind.CALLABLE}
    for quantity in manifest.quantities:
        frontier = (by_quantity.get(quantity.quantity_id, set()) if quantity.exactness is ExactnessKind.EXACT
                    else set(quantity.sources) & direct)
        for source in frontier:
            if source.member_id in outputs:
                outputs[source.member_id].add(quantity.quantity_id)
    return {name: frozenset(quantities) for name, quantities in outputs.items()}


@_binding_errors
def validate_query_contract(surface, results, manifest, *, bindings=(), declarations=(), require_closed=True):
    """Check the whole callable relation and return explicitly unbound members.

    Ordinary readers reject a nonempty return. Declaration-only audit reports
    retain it as an outstanding obligation, without acquiring runtime readiness.
    The legacy per-record validator cannot establish this cross-record relation.
    """
    validate_extraction_manifest(surface, results, manifest, require_closed=require_closed)
    _tuple(bindings, QueryBindingRecord, "query bindings")
    _tuple(declarations, CallableParameterSurface, "callable parameter declarations")
    materialized = materialized_query_outputs(surface, manifest)
    declared = {}
    for declaration in declarations:
        _text(declaration.callable_id, "callable parameter declaration ID")
        _provenance(declaration.provenance, "callable parameter declaration")
        if declaration.callable_id in declared:
            raise QueryBindingError("duplicate callable parameter declaration")
        if declaration.callable_id not in materialized:
            raise QueryBindingError("parameter declaration is not a producer callable member")
        if type(declaration.parameter_names) is not tuple:
            raise QueryBindingError("producer parameter names must be a tuple")
        for name in declaration.parameter_names:
            _text(name, "producer parameter name")
        if len(set(declaration.parameter_names)) != len(declaration.parameter_names):
            raise QueryBindingError("producer parameter declaration contains duplicates")
        declared[declaration.callable_id] = declaration
    quantities = {quantity.quantity_id: quantity for quantity in manifest.quantities}
    seen, bound = set(), set()
    for binding in bindings:
        if binding.binding_id in seen:
            raise QueryBindingError("duplicate query binding ID")
        seen.add(binding.binding_id)
        if binding.callable_id not in materialized:
            raise QueryBindingError("binding callable is not a producer callable member")
        declaration = declared.get(binding.callable_id)
        if declaration is None:
            raise QueryBindingError("query binding has no independent parameter declaration")
        validate_query_binding(binding, declaration, tuple(quantities))
        expected = materialized[binding.callable_id]
        if not expected:
            raise QueryBindingError("binding materializes no manifest quantity of its callable")
        _partition(binding.quantity_ids, tuple(sorted(expected)), "query binding outputs for " + binding.callable_id)
        if {quantities[identity].reader_id for identity in expected} != {binding.reader_id}:
            raise QueryBindingError("query binding reader differs from quantity reader custody")
        bound.add(binding.callable_id)
    return tuple(sorted(name for name, outputs in materialized.items() if outputs and name not in bound))


@_binding_errors
def validate_query_binding(binding: QueryBindingRecord, declaration: CallableParameterSurface,
                           quantity_ids: Tuple[str, ...]) -> None:
    """Check local roles and known IDs; whole-contract output closure is separate."""
    if type(binding) is not QueryBindingRecord or type(declaration) is not CallableParameterSurface:
        raise QueryBindingError("query binding and callable declaration must be typed records")
    _provenance(declaration.provenance, "callable parameter declaration")
    _provenance(binding.provenance, "query binding")
    for field in ("binding_id", "callable_id", "owner_resolver_id", "reader_id", "freshness",
                  "effect_policy", "result_mapping", "outcome_policy"):
        _text(getattr(binding, field), field)
    if binding.callable_id != declaration.callable_id:
        raise QueryBindingError("binding callable differs from producer parameter declaration")
    if type(binding.parameters) is not tuple or any(type(item) is not QueryParameter for item in binding.parameters):
        raise QueryBindingError("query parameters must be typed records")
    if type(declaration.parameter_names) is not tuple:
        raise QueryBindingError("producer parameter names must be a tuple")
    for name in declaration.parameter_names:
        _text(name, "producer parameter name")
    names = tuple(item.name for item in binding.parameters)
    for name in names:
        _text(name, "binding parameter name")
    _partition(names, declaration.parameter_names, "query parameter population")
    if names != declaration.parameter_names or len(set(declaration.parameter_names)) != len(declaration.parameter_names):
        raise QueryBindingError("native query parameter order differs or contains duplicates")
    arguments, required = set(), set()
    for parameter in binding.parameters:
        if type(parameter.role) is not ParameterRole:
            raise QueryBindingError("query parameter needs a declared role")
        _text(parameter.native_schema, "native parameter schema")
        if parameter.role is ParameterRole.OWNER:
            _text(parameter.owner_kind, "native operand kind")
        elif parameter.owner_kind is not None:
            raise QueryBindingError("non-owner parameter declares an owner kind")
        if parameter.role is ParameterRole.ARGUMENT:
            arguments.add(parameter.name)
            if parameter.default_json is None:
                required.add(parameter.name)
            elif type(parameter.default_json) is not str or _canonical(json.loads(parameter.default_json)) != parameter.default_json:
                raise QueryBindingError("native default must be canonical JSON")
        elif parameter.default_json is not None:
            raise QueryBindingError("only configured arguments can have authoring defaults")
    if type(binding.arguments) is not ArgumentSchema:
        raise QueryBindingError("query binding needs a typed argument schema")
    schema = binding.arguments.document()
    if set(schema["properties"]) != arguments or set(schema.get("required", ())) != required:
        raise QueryBindingError("argument schema does not cover the declared native arguments and required values")
    validator = Draft202012Validator(schema, registry=Registry())
    for parameter in binding.parameters:
        if parameter.default_json is not None:
            for error in validator.descend(json.loads(parameter.default_json), schema["properties"][parameter.name]):
                raise QueryBindingError(binding.binding_id + ": invalid declared default for " + parameter.name + ": " + error.message)
    if type(quantity_ids) is not tuple:
        raise QueryBindingError("known quantity IDs must be a tuple")
    for quantity_id in quantity_ids:
        _text(quantity_id, "known quantity ID")
    if len(set(quantity_ids)) != len(quantity_ids):
        raise QueryBindingError("known quantity IDs contain duplicates")
    for values, label in ((binding.quantity_ids, "binding quantities"), (binding.completion_dependencies, "completion dependencies")):
        if type(values) is not tuple:
            raise QueryBindingError(label + " must be a tuple")
        for value in values:
            _text(value, label)
        if len(set(values)) != len(values):
            raise QueryBindingError(label + " contains duplicates")
    if not binding.quantity_ids or not set(binding.quantity_ids) <= set(quantity_ids):
        raise QueryBindingError("binding outputs need known quantity IDs")
    if binding.population_resolver_id is not None:
        _text(binding.population_resolver_id, "finite population resolver")
        if type(binding.population_arguments) is not ArgumentSchema:
            raise QueryBindingError("finite population needs an argument schema")
        # A population schema may add population-only arguments that select
        # the population (``population: all``, issue #136). They never reach
        # the native callable, so none may reuse a native parameter name.
        population = set(binding.population_arguments.document()["properties"])
        if not arguments <= population or (population - arguments) & set(names):
            raise QueryBindingError("population argument schema changes native parameter names")
    elif binding.population_arguments is not None:
        raise QueryBindingError("population schema has no finite population resolver")


@dataclass(frozen=True)
class ResolvedOperand:
    parameter: str
    kind: str
    native_index: Tuple[int, ...]

    @_binding_errors
    def __post_init__(self):
        _text(self.parameter, "operand parameter")
        _text(self.kind, "operand kind")
        if type(self.native_index) is not tuple or not self.native_index or any(type(item) is not int for item in self.native_index):
            raise QueryBindingError("native resolver returned an invalid owner address")


@dataclass(frozen=True)
class ConfiguredQuery:
    """One explicit query or a finite-population argument declaration."""

    binding_id: str
    arguments_json: str
    operands: Tuple[ResolvedOperand, ...] = ()
    instance_token: Optional[str] = None

    @_binding_errors
    def __post_init__(self):
        _text(self.binding_id, "configured binding ID")
        if type(self.arguments_json) is not str:
            raise QueryBindingError("configured arguments must be canonical JSON text")
        arguments = json.loads(self.arguments_json)
        if type(arguments) is not dict or _canonical(arguments) != self.arguments_json:
            raise QueryBindingError("configured arguments must be a canonical JSON object")
        if type(self.operands) is not tuple or any(type(item) is not ResolvedOperand for item in self.operands):
            raise QueryBindingError("configured operands must be typed records")
        names = [item.parameter for item in self.operands]
        if len(set(names)) != len(names):
            raise QueryBindingError("configured operands contain duplicate parameters")
        if self.instance_token is None:
            if self.operands:
                raise QueryBindingError("population declarations cannot carry explicit operands")
        else:
            import re
            pattern = _validator().schema["properties"]["instances"]["items"]["properties"]["id"]["pattern"]
            if type(self.instance_token) is not str or re.fullmatch(pattern, self.instance_token) is None:
                raise QueryBindingError("configured instance token is invalid")

    def arguments(self) -> dict:
        return json.loads(self.arguments_json)


@_binding_errors
def _prepare_query_configuration(
    configuration: QueryConfiguration,
    bindings: Tuple[QueryBindingRecord, ...],
    declarations: Tuple[CallableParameterSurface, ...],
    quantity_ids: Tuple[str, ...],
    validate_native_arguments=None,
):
    """Prepare declared arguments and operands without native resolution."""
    if type(configuration) is not QueryConfiguration:
        raise QueryBindingError("query configuration must be immutable")
    if type(bindings) is not tuple or type(declarations) is not tuple:
        raise QueryBindingError("query bindings and declarations must be tuples")
    if any(type(item) is not CallableParameterSurface for item in declarations):
        raise QueryBindingError("callable declarations must be typed records")
    declared = {item.callable_id: item for item in declarations}
    if len(declared) != len(declarations):
        raise QueryBindingError("duplicate callable parameter declaration")
    known = {}
    for binding in bindings:
        if type(binding) is not QueryBindingRecord or binding.callable_id not in declared:
            raise QueryBindingError("query binding has no producer parameter declaration")
        validate_query_binding(binding, declared[binding.callable_id], quantity_ids)
        if binding.binding_id in known:
            raise QueryBindingError("duplicate query binding ID")
        known[binding.binding_id] = binding
    document = configuration.document()
    for entry in document["populations"] + document["instances"]:
        if entry["binding"] not in known:
            raise QueryBindingError("unknown query binding: " + entry["binding"])
    populations = {entry["binding"]: entry["arguments"] for entry in document["populations"]}
    for binding_id in populations:
        if known[binding_id].population_resolver_id is None:
            raise QueryBindingError("query has no finite population: " + binding_id)
    pending, missing = [], []
    for binding in bindings:
        if binding.population_resolver_id is not None:
            arguments = dict(populations.get(binding.binding_id, {}))
            defaults = {item.name: json.loads(item.default_json) for item in binding.parameters if item.default_json is not None}
            arguments = dict(defaults, **arguments)
            assert binding.population_arguments is not None
            required = (set(binding.arguments.document().get("required", ()))
                        | set(binding.population_arguments.document().get("required", ())))
            absent = tuple(sorted(required - set(arguments)))
            native = _native_arguments(binding, arguments)
            if absent:
                binding.arguments.validate(native, binding.binding_id, allow_missing_arguments=True)
                binding.population_arguments.validate(arguments, binding.binding_id, allow_missing_arguments=True)
                missing.append((binding.binding_id, absent))
                continue
            binding.arguments.validate(native, binding.binding_id)
            binding.population_arguments.validate(arguments, binding.binding_id)
            pending.append((binding, arguments, {}, None))
    for instance in document["instances"]:
        binding = known[instance["binding"]]
        defaults = {item.name: json.loads(item.default_json) for item in binding.parameters if item.default_json is not None}
        arguments = dict(defaults, **instance["arguments"])
        binding.arguments.validate(arguments, binding.binding_id)
        expected = {item.name: item.owner_kind for item in binding.parameters if item.role is ParameterRole.OWNER}
        operands = instance["operands"]
        if set(operands) != set(expected) or any(operands[name]["kind"] != kind for name, kind in expected.items()):
            raise QueryBindingError(binding.binding_id + ": native operand names or kinds differ")
        pending.append((binding, arguments, operands, instance["id"]))
    if missing:
        raise QueryConfigurationRequired(tuple(missing))
    if validate_native_arguments is not None:
        for binding, arguments, _, _ in pending:
            validate_native_arguments(binding, json.loads(_canonical(_native_arguments(binding, arguments))))
    return tuple(pending)


def _native_arguments(binding: QueryBindingRecord, arguments: dict) -> dict:
    """The arguments the native callable takes; population-only ones stay out."""
    native = binding.arguments.document()["properties"]
    return {name: value for name, value in arguments.items() if name in native}


@_binding_errors
def validate_query_arguments(
    configuration: QueryConfiguration,
    bindings: Tuple[QueryBindingRecord, ...],
    declarations: Tuple[CallableParameterSurface, ...],
    quantity_ids: Tuple[str, ...],
    *,
    validate_native_arguments=None,
) -> None:
    """Validate argument contracts without looking up a native owner.

    This checks owner declaration shapes and kinds. Native existence, bounds,
    population, effects and readiness remain separate runtime obligations.
    """
    _prepare_query_configuration(configuration, bindings, declarations, quantity_ids,
                                 validate_native_arguments)


@_binding_errors
def bind_query_configuration(
    configuration: QueryConfiguration,
    bindings: Tuple[QueryBindingRecord, ...],
    declarations: Tuple[CallableParameterSurface, ...],
    quantity_ids: Tuple[str, ...],
    resolve_operand: Callable[[str, Dict], Tuple[int, ...]],
    *,
    validate_native_arguments=None,
) -> Tuple[ConfiguredQuery, ...]:
    """Resolve native owners after the shared argument phase has passed."""
    pending = _prepare_query_configuration(configuration, bindings, declarations, quantity_ids,
                                           validate_native_arguments)
    resolved = []
    for binding, arguments, operands, token in pending:
        owners = []
        for parameter in binding.parameters:
            if parameter.name not in operands:
                continue
            operand = operands[parameter.name]
            index = resolve_operand(operand["kind"], dict(operand))
            if type(index) is not tuple or not index or any(type(value) is not int for value in index):
                raise QueryBindingError(binding.binding_id + ": native resolver returned an invalid owner address")
            owners.append(ResolvedOperand(parameter.name, operand["kind"], index))
        resolved.append(ConfiguredQuery(binding.binding_id, _canonical(arguments), tuple(owners), token))
    return tuple(resolved)
