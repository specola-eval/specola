"""Independent producer API snapshots, without extraction classifications.

A snapshot records a finite producer declaration population. Equality detects
drift; it does not establish physical semantics or completeness of its roots.
Root-census evidence and semantic classification are separate obligations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import inspect
import json
from types import (
    BuiltinFunctionType, ClassMethodDescriptorType, FunctionType,
    GetSetDescriptorType, MemberDescriptorType, MethodDescriptorType,
    ModuleType, WrapperDescriptorType,
)
from typing import Any, Dict, Iterable, Mapping, Tuple, cast

from .core_runner import ConformanceValidationError, EvidenceIdentity

_TYPE_METADATA = vars(type)
_MODULE_DICTIONARY = vars(ModuleType)["__dict__"]


class SurfaceValidationError(ValueError):
    """A producer snapshot is malformed or differs from its reviewed pin."""


class MemberKind(str, Enum):
    """Syntactic member kinds, not decisions about observation semantics."""

    FIELD = "field"
    PROPERTY = "property"
    CALLABLE = "callable"
    TYPE = "type"
    MODULE = "module"


@dataclass(frozen=True)
class ProducerIdentity:
    """Version, backend and content identity of a measured producer."""

    name: str
    version: str
    backend: str
    content: EvidenceIdentity


@dataclass(frozen=True)
class SurfaceRoot:
    """A named declaration root and evidence for its membership boundary."""

    name: str
    source: EvidenceIdentity
    population_rule: str
    empty_reason: str = ""
    unresolved_reason: str = ""


@dataclass(frozen=True)
class ApiMember:
    """One public member and captured native declaration data."""

    root: str
    name: str
    kind: MemberKind
    declaration: str

    @property
    def member_id(self) -> str:
        return self.root + "." + self.name


@dataclass(frozen=True)
class ProducerSurface:
    """A producer-derived candidate set that contains no semantic verdicts."""

    producer: ProducerIdentity
    enumerator: EvidenceIdentity
    root_census: EvidenceIdentity
    roots: Tuple[SurfaceRoot, ...]
    members: Tuple[ApiMember, ...]


def _text(value: Any, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise SurfaceValidationError(label + " must be nonempty text")


def _evidence(value: Any, label: str) -> None:
    if type(value) is not EvidenceIdentity:
        raise SurfaceValidationError(label + " must be an EvidenceIdentity")
    # Revalidate even when the caller has bypassed a frozen record's constructor.
    try:
        EvidenceIdentity(value.identity, value.sha256)
    except ConformanceValidationError as error:
        raise SurfaceValidationError(label + ": " + str(error)) from error


def _ordered_unique(values: Tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(set(values))):
        raise SurfaceValidationError(label + " must be sorted and unique")


def validate_surface(surface: ProducerSurface) -> None:
    """Validate a snapshot without claiming that its root census is complete."""
    if type(surface) is not ProducerSurface:
        raise SurfaceValidationError("snapshot must be a ProducerSurface")
    if type(surface.producer) is not ProducerIdentity:
        raise SurfaceValidationError("snapshot needs a ProducerIdentity")
    for name in ("name", "version", "backend"):
        _text(getattr(surface.producer, name), "producer " + name)
    _evidence(surface.producer.content, "producer content")
    _evidence(surface.enumerator, "enumerator")
    _evidence(surface.root_census, "root census")
    if type(surface.roots) is not tuple or not surface.roots:
        raise SurfaceValidationError("snapshot needs a nonempty tuple of roots")
    if type(surface.members) is not tuple:
        raise SurfaceValidationError("snapshot members must be a tuple")
    for root in surface.roots:
        if type(root) is not SurfaceRoot:
            raise SurfaceValidationError("snapshot roots must be SurfaceRoot records")
        _text(root.name, "root name")
        _text(root.population_rule, "root population rule")
        _evidence(root.source, "root source")
        if type(root.empty_reason) is not str:
            raise SurfaceValidationError("root empty reason must be text")
        if type(root.unresolved_reason) is not str:
            raise SurfaceValidationError("root unresolved reason must be text")
    root_names = tuple(root.name for root in surface.roots)
    _ordered_unique(root_names, "root names")
    populated = set()
    for member in surface.members:
        if type(member) is not ApiMember:
            raise SurfaceValidationError("snapshot members must be ApiMember records")
        if member.root not in root_names:
            raise SurfaceValidationError("member has an unknown root: " + str(member.root))
        _text(member.name, "member name")
        if "." in member.name:
            raise SurfaceValidationError("member name is local to its root")
        if type(member.kind) is not MemberKind:
            raise SurfaceValidationError("member kind must be a MemberKind")
        _text(member.declaration, "member declaration")
        populated.add(member.root)
    _ordered_unique(tuple(m.member_id for m in surface.members), "member IDs")
    for root in surface.roots:
        if root.unresolved_reason:
            if not root.unresolved_reason.strip() or root.empty_reason or root.name in populated:
                raise SurfaceValidationError("unresolved root cannot claim empty or captured declarations: " + root.name)
            continue
        if root.name in populated and root.empty_reason:
            raise SurfaceValidationError("populated root has an empty reason: " + root.name)
        if root.name not in populated and not root.empty_reason.strip():
            raise SurfaceValidationError("empty root needs a reason: " + root.name)


def surface_document(surface: ProducerSurface) -> Dict[str, Any]:
    """Return the versioned, deterministic JSON representation of a snapshot."""
    validate_surface(surface)
    return {"schema": "specola.producer-surface.v1", **asdict(surface)}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def surface_digest(surface: ProducerSurface) -> str:
    """Hash content, root evidence and declarations, without a stored pass flag."""
    return hashlib.sha256(_canonical(surface_document(surface)).encode("utf-8")).hexdigest()


def _keys(value: Any, names: Iterable[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(names):
        raise SurfaceValidationError(label + " has missing or unexpected fields")
    return value


def surface_from_document(document: Mapping[str, Any]) -> ProducerSurface:
    """Restore exact schema v1; reject unknown fields rather than dropping them."""
    raw = _keys(document, ("schema", "producer", "enumerator", "root_census", "roots", "members"), "snapshot")
    if raw["schema"] != "specola.producer-surface.v1":
        raise SurfaceValidationError("unknown producer surface schema")

    def evidence(item: Any) -> EvidenceIdentity:
        item = _keys(item, ("identity", "sha256"), "evidence")
        try:
            return EvidenceIdentity(**item)
        except ConformanceValidationError as error:
            raise SurfaceValidationError("invalid surface evidence: " + str(error)) from error

    producer = _keys(raw["producer"], ("name", "version", "backend", "content"), "producer")
    if type(raw["roots"]) is not list or type(raw["members"]) is not list:
        raise SurfaceValidationError("serialized roots and members must be lists")
    roots = []
    for item in raw["roots"]:
        item = _keys(item, ("name", "source", "population_rule", "empty_reason", "unresolved_reason"), "root")
        roots.append(SurfaceRoot(item["name"], evidence(item["source"]), item["population_rule"],
                                 item["empty_reason"], item["unresolved_reason"]))
    members = []
    for item in raw["members"]:
        item = _keys(item, ("root", "name", "kind", "declaration"), "member")
        try:
            kind = MemberKind(item["kind"])
        except (ValueError, TypeError) as error:
            raise SurfaceValidationError("unknown member kind") from error
        members.append(ApiMember(item["root"], item["name"], kind, item["declaration"]))
    result = ProducerSurface(
        ProducerIdentity(producer["name"], producer["version"], producer["backend"], evidence(producer["content"])),
        evidence(raw["enumerator"]), evidence(raw["root_census"]), tuple(roots), tuple(members),
    )
    validate_surface(result)
    return result


def validate_known_surface(expected: ProducerSurface, observed: ProducerSurface) -> None:
    """Reject producer, root, signature or field drift against an independent pin."""
    validate_surface(expected)
    validate_surface(observed)
    if expected.producer != observed.producer:
        raise SurfaceValidationError("producer identity differs from reviewed surface")
    if expected.enumerator != observed.enumerator:
        raise SurfaceValidationError("surface enumerator differs from reviewed pin")
    if expected.root_census != observed.root_census or expected.roots != observed.roots:
        raise SurfaceValidationError("producer root census differs from reviewed surface")
    prior = {member.member_id: member for member in expected.members}
    live = {member.member_id: member for member in observed.members}
    added = sorted(set(live) - set(prior))
    removed = sorted(set(prior) - set(live))
    changed = sorted(name for name in set(prior) & set(live) if prior[name] != live[name])
    if added or removed or changed:
        raise SurfaceValidationError(
            "producer surface drift: added={}; removed={}; changed={}".format(added, removed, changed)
        )


def _class_dict(cls: type) -> Mapping[str, Any]:
    namespace = _TYPE_METADATA["__dict__"].__get__(cls)
    if any(type(name) is not str for name in namespace):
        raise SurfaceValidationError("type namespace keys must be exact strings")
    return namespace


def _type_members(cls: type) -> Dict[str, Any]:
    result = {}
    for base in reversed(_TYPE_METADATA["__mro__"].__get__(cls)):
        result.update(_class_dict(base))
    return result


def _type_name(cls: type) -> str:
    # CPython's type descriptor distinguishes a static type's native name
    # from a heap type's module entry without invoking metaclass hooks.
    # A static function type also declares an instance __module__ descriptor;
    # that raw dictionary entry is not the module of the function type.
    try:
        module = _TYPE_METADATA["__module__"].__get__(cls)
    except AttributeError as error:
        raise SurfaceValidationError("type has no declared module identity") from error
    name = _TYPE_METADATA["__qualname__"].__get__(cls)
    if type(module) is not str or not module or type(name) is not str or not name:
        raise SurfaceValidationError("type identity requires declared module and qualified name strings")
    return module + "." + name


def _literal(value: Any) -> Dict[str, Any]:
    """Describe declaration literals without calling producer repr methods."""
    value_type = type(value)
    result: Dict[str, Any] = {"python_type": _type_name(value_type)}
    if value is None or value_type in (str, bool, int):
        result["value"] = value
    elif value_type is float:
        # Hex preserves signed zero and non-finite declaration constants while
        # retaining strict JSON in the outer evidence record.
        result["hex"] = float.hex(value)
    elif value_type is bytes:
        result["hex"] = bytes.hex(value)
    elif value_type is tuple:
        result["items"] = [_literal(item) for item in value]
    elif issubclass(value_type, type):
        result["target"] = _type_name(value)
    elif issubclass(value_type, ModuleType):
        result["target"] = _MODULE_DICTIONARY.__get__(value).get("__name__")
    else:
        result["unresolved_value"] = "requires an upstream typed declaration"
    return result


def _python_signature(function: Any) -> Dict[str, Any]:
    """Read Python function metadata without rendering user-defined defaults."""
    code = function.__code__
    defaults = function.__defaults__ or ()
    keyword_defaults = function.__kwdefaults__ or {}
    annotations = function.__annotations__
    parameters = []
    positional_count = code.co_argcount
    for index, name in enumerate(code.co_varnames[:positional_count + code.co_kwonlyargcount]):
        kind = "positional_only" if index < code.co_posonlyargcount else "positional_or_keyword"
        item: Dict[str, Any] = {"name": name, "kind": kind}
        if index >= positional_count:
            item["kind"] = "keyword_only"
            if name in keyword_defaults:
                item["default"] = _literal(keyword_defaults[name])
        elif index >= positional_count - len(defaults):
            item["default"] = _literal(defaults[index - positional_count + len(defaults)])
        if name in annotations:
            item["annotation"] = _literal(annotations[name])
        parameters.append(item)
    offset = positional_count + code.co_kwonlyargcount
    for flag, kind in ((inspect.CO_VARARGS, "var_positional"), (inspect.CO_VARKEYWORDS, "var_keyword")):
        if code.co_flags & flag:
            name = code.co_varnames[offset]
            item = {"name": name, "kind": kind}
            if name in annotations:
                item["annotation"] = _literal(annotations[name])
            parameters.append(item)
            offset += 1
    result: Dict[str, Any] = {"parameters": parameters}
    if "return" in annotations:
        result["return"] = _literal(annotations["return"])
    return result


def static_namespace_members(root: str, owner: Any) -> Tuple[ApiMember, ...]:
    """Enumerate public and special names without invoking descriptors.

    This captures one namespace, not a transitive API census. Producer-specific
    collectors add upstream typed declarations, runtime instance populations
    and explicit roots for nested types. Dynamic exports remain a separate
    obligation; this function never calls ``dir``, producer getters or repr.
    Unknown default objects retain their type and an unresolved-value marker.
    Collector root closure must resolve dynamic namespace hooks separately.
    Special names retain constructors, indexed queries and representation
    declarations; their presence does not authorize execution or classify
    them as observations.
    """
    _text(root, "root name")
    instance_names = set()
    defining_owners = {}
    if issubclass(type(owner), ModuleType):
        namespace = dict(_MODULE_DICTIONARY.__get__(owner))
        if any(type(name) is not str for name in namespace):
            raise SurfaceValidationError("module namespace keys must be exact strings")
        defining_owners = {name: namespace.get("__name__", root) for name in namespace}
    else:
        is_class = issubclass(type(owner), type)
        cls = owner if is_class else type(owner)
        namespace = {}
        for base in reversed(_TYPE_METADATA["__mro__"].__get__(cls)):
            base_namespace = _class_dict(base)
            namespace.update(base_namespace)
            defining_owners.update({name: _type_name(base) for name in base_namespace})
        if not is_class:
            dictionary_descriptor = namespace.get("__dict__")
            if dictionary_descriptor is not None and type(dictionary_descriptor) is not GetSetDescriptorType:
                raise SurfaceValidationError("instance __dict__ is shadowed by a producer descriptor")
            if dictionary_descriptor is not None:
                instance_dict = dictionary_descriptor.__get__(owner)
                if type(instance_dict) is not dict:
                    raise SurfaceValidationError("instance namespace is not a dictionary")
                if any(type(name) is not str for name in instance_dict):
                    raise SurfaceValidationError("instance namespace keys must be exact strings")
                for name, value in instance_dict.items():
                    prior = namespace.get(name)
                    prior_type_members = _type_members(type(prior))
                    if "__set__" not in prior_type_members and "__delete__" not in prior_type_members:
                        namespace[name] = value
                        defining_owners[name] = _type_name(cls)
                        instance_names.add(name)
    result = []
    for name, value in sorted(namespace.items()):
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue
        binding_kind = None
        if type(value) in (staticmethod, classmethod):
            binding_kind = type(value).__name__
            value = value.__func__
        value_type = type(value)
        type_members = _type_members(value_type)
        if issubclass(value_type, ModuleType):
            kind = MemberKind.MODULE
        elif issubclass(value_type, type):
            kind = MemberKind.TYPE
        elif "__get__" in type_members and ("__set__" in type_members or "__delete__" in type_members):
            kind = MemberKind.PROPERTY
        elif callable(value):
            kind = MemberKind.CALLABLE
        elif "__get__" in type_members:
            kind = MemberKind.PROPERTY
        else:
            kind = MemberKind.FIELD
        declaration: Dict[str, Any] = {"python_type": _type_name(value_type), "defining_owner": defining_owners[name]}
        if binding_kind is not None:
            declaration["binding_kind"] = binding_kind
        native_metadata_types = (BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType,
                                 ClassMethodDescriptorType, GetSetDescriptorType, MemberDescriptorType)
        if kind is MemberKind.TYPE:
            declaration["target"] = _type_name(cast(type, value))
            doc = _class_dict(cast(type, value)).get("__doc__")
        elif kind is MemberKind.MODULE:
            declaration["target"] = _MODULE_DICTIONARY.__get__(value).get("__name__")
            doc = None
        elif value_type in (property, FunctionType) + native_metadata_types:
            doc = value.__doc__
        else:
            doc = None
        if kind in (MemberKind.CALLABLE, MemberKind.PROPERTY, MemberKind.TYPE):
            if type(doc) is str:
                declaration["doc"] = doc
        if kind is MemberKind.FIELD and name not in instance_names:
            declaration["literal"] = _literal(value)
        if kind is MemberKind.CALLABLE:
            declaration["signature"] = None
            if value_type is FunctionType:
                declaration["signature"] = _python_signature(value)
                declaration["wrapped"] = "__wrapped__" in vars(value)
            elif value_type in native_metadata_types:
                declaration["text_signature"] = getattr(value, "__text_signature__", None)
            else:
                declaration["unresolved_signature"] = "requires an upstream typed declaration"
        result.append(ApiMember(root, name, kind, _canonical(declaration)))
    return tuple(result)
