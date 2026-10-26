"""Independent API surface pins detect members no extraction code mentions."""

from dataclasses import replace
import json
from types import ModuleType

import pytest

from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.surface import (
    ApiMember, MemberKind, ProducerIdentity, ProducerSurface, SurfaceRoot,
    SurfaceValidationError, static_namespace_members, surface_digest,
    surface_document, surface_from_document, validate_known_surface,
    validate_surface,
)


def _evidence(name="source"):
    return EvidenceIdentity(name, "a" * 64)


def _surface():
    return ProducerSurface(
        ProducerIdentity("native", "1.0", "cpu", _evidence()),
        _evidence("enumerator"), _evidence("independent-root-census"),
        (SurfaceRoot("native.State", _evidence(), "upstream struct declarations"),),
        (ApiMember("native.State", "position", MemberKind.FIELD, "double[3]"),),
    )


def _member(root, owner, name):
    return next(member for member in static_namespace_members(root, owner) if member.name == name)


def test_snapshot_round_trip_and_content_digest():
    original = _surface()
    restored = surface_from_document(json.loads(json.dumps(surface_document(original))))
    assert restored == original
    assert surface_digest(restored) == surface_digest(original)
    assert "disposition" not in json.dumps(surface_document(original))
    validate_known_surface(original, restored)


@pytest.mark.parametrize("kind", tuple(MemberKind))
def test_unused_producer_addition_fails_without_an_adapter_change(kind):
    expected = _surface()
    added = ApiMember("native.State", "unused", kind, "new upstream declaration")
    with pytest.raises(SurfaceValidationError, match="added=.*unused"):
        validate_known_surface(expected, replace(expected, members=expected.members + (added,)))


@pytest.mark.parametrize("declaration", ("double[4]", "query(a, b, cutoff) -> (distance, normal)", "float[3]"))
def test_signature_result_or_type_drift_fails(declaration):
    expected = _surface()
    with pytest.raises(SurfaceValidationError, match="changed=.*position"):
        validate_known_surface(expected, replace(expected, members=(replace(expected.members[0], declaration=declaration),)))


def test_removed_member_is_not_silently_accepted():
    expected = _surface()
    extra = ApiMember("native.State", "velocity", MemberKind.FIELD, "double[3]")
    with pytest.raises(SurfaceValidationError, match="removed=.*velocity"):
        validate_known_surface(replace(expected, members=expected.members + (extra,)), expected)


@pytest.mark.parametrize("field,value", (("version", "1.1"), ("backend", "gpu"), ("content", EvidenceIdentity("source", "b" * 64))))
def test_version_backend_and_edited_package_fail(field, value):
    expected = _surface()
    observed = replace(expected, producer=replace(expected.producer, **{field: value}))
    with pytest.raises(SurfaceValidationError, match="producer identity"):
        validate_known_surface(expected, observed)


def test_unused_root_and_census_changes_fail():
    expected = _surface()
    root = SurfaceRoot("native.Unknown", _evidence(), "upstream public class", "no public members")
    with pytest.raises(SurfaceValidationError, match="root census"):
        validate_known_surface(expected, replace(expected, roots=expected.roots + (root,)))
    with pytest.raises(SurfaceValidationError, match="root census"):
        validate_known_surface(expected, replace(expected, root_census=_evidence("different census")))


def test_empty_child_surface_requires_explicit_evidence():
    snapshot = _surface()
    with pytest.raises(SurfaceValidationError, match="empty root needs a reason"):
        validate_surface(replace(snapshot, members=()))
    empty = replace(snapshot.roots[0], empty_reason="no methods in this declared type")
    validate_surface(replace(snapshot, roots=(empty,), members=()))
    with pytest.raises(SurfaceValidationError, match="populated root"):
        validate_surface(replace(snapshot, roots=(empty,)))


@pytest.mark.parametrize("change", (
    lambda s: replace(s, roots=()),
    lambda s: replace(s, members=s.members + s.members),
    lambda s: replace(s, roots=s.roots + s.roots),
    lambda s: replace(s, members=(replace(s.members[0], root="unknown"),)),
    lambda s: replace(s, members=(replace(s.members[0], declaration=""),)),
    lambda s: replace(s, members=(replace(s.members[0], kind="field"),)),
))
def test_invalid_snapshot_structure_fails(change):
    with pytest.raises(SurfaceValidationError):
        validate_surface(change(_surface()))


def test_reader_rejects_unknown_fields_and_schemas():
    document = json.loads(json.dumps(surface_document(_surface())))
    document["pass"] = True
    with pytest.raises(SurfaceValidationError, match="unexpected fields"):
        surface_from_document(document)
    document.pop("pass")
    document["schema"] = "future"
    with pytest.raises(SurfaceValidationError, match="unknown producer surface schema"):
        surface_from_document(document)


def test_static_enumeration_never_calls_properties_or_custom_dir():
    class State:
        @property
        def position(self):
            raise AssertionError("inventory executed a property")

        def __dir__(self):
            raise AssertionError("inventory executed __dir__")

        def query(self, target, cutoff=1.0):
            raise AssertionError("inventory executed a query")

    actual = static_namespace_members("native.State", State())
    assert [(m.name, m.kind) for m in actual if not m.name.startswith("_")] == [
        ("position", MemberKind.PROPERTY), ("query", MemberKind.CALLABLE),
    ]
    parameters = json.loads(next(m for m in actual if m.name == "query").declaration)["signature"]["parameters"]
    assert [p["name"] for p in parameters] == ["self", "target", "cutoff"]
    assert parameters[-1]["default"]["hex"] == float.hex(1.0)


def test_static_enumeration_includes_inheritance_and_new_module_exports():
    class Base:
        field = 0

    class Child(Base):
        extra = 1

    assert {m.name for m in static_namespace_members("native.Child", Child)} == set(vars(object)) | set(vars(Base)) | set(vars(Child))
    module = ModuleType("native")
    vars(module).update(Child=Child, helper=lambda arg: arg, counter=0)
    actual = static_namespace_members("native", module)
    assert {m.name for m in actual} == set(vars(module))
    assert [(m.name, m.kind) for m in actual if not m.name.startswith("_")] == [
        ("Child", MemberKind.TYPE), ("counter", MemberKind.FIELD), ("helper", MemberKind.CALLABLE),
    ]


def test_enumerator_drift_and_evidence_changes_affect_identity():
    expected = _surface()
    observed = replace(expected, enumerator=_evidence("new enumerator"))
    assert surface_digest(observed) != surface_digest(expected)
    with pytest.raises(SurfaceValidationError, match="enumerator"):
        validate_known_surface(expected, observed)


def test_native_callable_and_property_documentation_is_retained():
    module = ModuleType("native")
    vars(module)["native_query"] = len

    class State:
        @property
        def position(self):
            """Native xyz, metres in world coordinates."""
            raise AssertionError("must not read property")

    assert json.loads(_member("native", module, "native_query").declaration)["doc"] == len.__doc__
    assert "metres in world" in _member("native.State", State, "position").declaration


def test_callable_instances_are_not_misclassified_as_data_fields():
    class Query:
        def __call__(self):
            raise AssertionError("must not execute query")

    module = ModuleType("native")
    vars(module)["query"] = Query()
    assert _member("native", module, "query").kind is MemberKind.CALLABLE


def test_instance_fields_are_captured_without_reading_descriptors():
    class State:
        def __init__(self):
            self.qpos = 1.0
            self.__dict__["position"] = "must not replace data descriptor"

        @property
        def position(self):
            raise AssertionError("must not read property")

    state = State()
    members = {m.name: m for m in static_namespace_members("native.State", state)}
    assert set(members) == set(vars(object)) | set(vars(State)) | {"qpos"}
    assert members["position"].kind is MemberKind.PROPERTY
    assert "literal" not in json.loads(members["qpos"].declaration)
    state.qpos = 2.0
    assert members["qpos"] == {m.name: m for m in static_namespace_members("native.State", state)}["qpos"]


def test_shadowed_dictionary_is_rejected_without_invoking_it():
    class State:
        @property
        def __dict__(self):  # pyright: ignore[reportIncompatibleVariableOverride]
            raise AssertionError("unsafe __dict__ read")

    with pytest.raises(SurfaceValidationError, match="shadowed"):
        static_namespace_members("native.State", State())


def test_instance_and_value_custom_attribute_hooks_are_not_executed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("unsafe attribute lookup: " + name)

    class State(Poison):
        field = Poison()

    assert _member("native.State", State(), "field").kind is MemberKind.FIELD


def test_function_default_repr_is_never_called_and_captures_are_stable():
    class Default:
        def __repr__(self):
            raise AssertionError("unsafe repr")

    def make_query():
        def query(target=Default()):
            raise AssertionError("unsafe query")
        return query

    module = ModuleType("native")
    vars(module)["query"] = make_query()
    first = static_namespace_members("native", module)
    vars(module)["query"] = make_query()
    assert static_namespace_members("native", module) == first
    declaration = next(m.declaration for m in first if m.name == "query")
    assert "unresolved_value" in declaration
    assert "0x" not in declaration


def test_immutable_constants_and_alias_targets_are_declarations():
    module = ModuleType("native")
    vars(module).update(constant=2, State=int, child=ModuleType("first"))
    before = static_namespace_members("native", module)
    vars(module).update(constant=3, State=str, child=ModuleType("second"))
    after = static_namespace_members("native", module)
    assert {left.name for left, right in zip(before, after) if left != right} == {"constant", "State", "child"}


def test_malformed_evidence_has_surface_exception_type():
    document = json.loads(json.dumps(surface_document(_surface())))
    document["producer"]["content"]["sha256"] = "A" * 64
    with pytest.raises(SurfaceValidationError, match="invalid surface evidence"):
        surface_from_document(document)
    evidence = _evidence()
    object.__setattr__(evidence, "sha256", "invalid")
    with pytest.raises(SurfaceValidationError):
        validate_surface(replace(_surface(), enumerator=evidence))


def test_member_classification_contamination_and_kind_only_drift_fail():
    document = json.loads(json.dumps(surface_document(_surface())))
    document["members"][0]["disposition"] = "exposed"
    with pytest.raises(SurfaceValidationError, match="unexpected fields"):
        surface_from_document(document)
    before = _surface()
    after = replace(before, members=(replace(before.members[0], kind=MemberKind.PROPERTY),))
    with pytest.raises(SurfaceValidationError, match="changed"):
        validate_known_surface(before, after)


@pytest.mark.parametrize("name", ("producer", "enumerator", "root_census"))
def test_wrong_envelope_types_are_rejected(name):
    with pytest.raises(SurfaceValidationError):
        validate_surface(replace(_surface(), **{name: "not a record"}))


def test_evaluated_annotations_and_type_defaults_preserve_targets():
    module = ModuleType("native")

    def query(target: int, dtype=float) -> str:
        raise AssertionError("unsafe query")

    vars(module)["query"] = query
    original = static_namespace_members("native", module)
    declaration = json.loads(next(m.declaration for m in original if m.name == "query"))["signature"]
    assert declaration["parameters"][0]["annotation"]["target"] == "builtins.int"
    assert declaration["parameters"][1]["default"]["target"] == "builtins.float"
    assert declaration["return"]["target"] == "builtins.str"
    query.__annotations__["target"] = float
    assert static_namespace_members("native", module) != original


def test_python_signature_preserves_parameter_kinds_and_defaults():
    module = ModuleType("native")

    def query(a, /, b=1, *args, c, d=2, **kw):
        raise AssertionError("unsafe query")

    vars(module)["query"] = query
    parameters = json.loads(_member("native", module, "query").declaration)["signature"]["parameters"]
    assert [(p["name"], p["kind"]) for p in parameters] == [
        ("a", "positional_only"), ("b", "positional_or_keyword"),
        ("c", "keyword_only"), ("d", "keyword_only"),
        ("args", "var_positional"), ("kw", "var_keyword"),
    ]
    assert parameters[1]["default"]["value"] == 1
    assert parameters[3]["default"]["value"] == 2


def test_non_data_descriptors_are_properties_without_being_read():
    class Descriptor:
        def __get__(self, owner, owner_type=None):
            raise AssertionError("unsafe descriptor")

    class State:
        lazy_state = Descriptor()

    assert _member("native.State", State, "lazy_state").kind is MemberKind.PROPERTY


def test_method_binding_kind_wrapping_and_native_text_signatures():
    class State:
        @staticmethod
        def static_query(arg): ...

        @classmethod
        def class_query(cls, arg): ...

    members = {m.name: json.loads(m.declaration) for m in static_namespace_members("native.State", State)}
    assert members["static_query"]["binding_kind"] == "staticmethod"
    assert members["class_query"]["binding_kind"] == "classmethod"
    descriptor = next(m for m in static_namespace_members("builtins.list", list) if m.name == "append")
    assert json.loads(descriptor.declaration)["text_signature"] == getattr(list.append, "__text_signature__")


def test_literal_tuple_bytes_and_negative_zero_are_lossless():
    module = ModuleType("native")
    vars(module)["constant"] = (b"\x00\xff", -0.0, None)
    literal = json.loads(_member("native", module, "constant").declaration)["literal"]
    assert literal["items"][0]["hex"] == "00ff"
    assert literal["items"][1]["hex"] == float.hex(-0.0)
    assert literal["items"][2]["value"] is None


def test_namespace_protocol_constructors_and_inherited_queries_are_not_executed():
    class Base:
        def __getitem__(self, address):
            raise AssertionError("inventory executed an indexed query")

    class State(Base):
        def __init__(self, configuration):
            raise AssertionError("inventory constructed native state")

        def __call__(self, owner, *, frame):
            raise AssertionError("inventory executed a callable query")

        @property
        def __array_interface__(self):
            raise AssertionError("inventory read native result metadata")

    members = {m.name: m for m in static_namespace_members("native.State", State)}
    assert {"__init__", "__call__", "__getitem__", "__array_interface__"} <= set(members)
    assert members["__array_interface__"].kind is MemberKind.PROPERTY
    assert json.loads(members["__getitem__"].declaration)["defining_owner"].endswith(".Base")
    signature = json.loads(members["__call__"].declaration)["signature"]
    assert [(p["name"], p["kind"]) for p in signature["parameters"]] == [
        ("self", "positional_or_keyword"), ("owner", "positional_or_keyword"),
        ("frame", "keyword_only"),
    ]


def test_namespace_protocol_addition_fails_independent_drift_check():
    class State:
        position = 1

    before = replace(_surface(), members=static_namespace_members("native.State", State))

    def indexed_query(self, address):
        raise AssertionError("inventory executed an unused query")

    setattr(State, "__getitem__", indexed_query)
    after = replace(before, members=static_namespace_members("native.State", State))
    with pytest.raises(SurfaceValidationError, match="added=.*__getitem__"):
        validate_known_surface(before, after)


def test_namespace_protocol_data_and_type_declarations_are_retained():
    class Result:
        value = 1

    module = ModuleType("native")
    vars(module).update(__result_type__=Result, __result_layout__=("value",), _private=1)
    members = {m.name: m for m in static_namespace_members("native", module)}
    assert members["__result_type__"].kind is MemberKind.TYPE
    assert json.loads(members["__result_layout__"].declaration)["literal"]["items"][0]["value"] == "value"
    assert "_private" not in members


def test_native_function_type_keeps_builtin_module_identity():
    module = ModuleType("native")
    vars(module)["query"] = len
    declaration = json.loads(_member("native", module, "query").declaration)
    assert declaration["python_type"] == "builtins.builtin_function_or_method"


@pytest.mark.parametrize("attribute", ("__module__", "__qualname__"))
def test_unresolved_type_identity_never_executes_string_subclass_hooks(attribute):
    class Poison(str):
        def __add__(self, other):
            raise AssertionError("inventory executed a string addition hook")

        def __radd__(self, other):
            raise AssertionError("inventory executed a reflected string hook")

    class Result:
        value = 1

    setattr(Result, attribute, Poison("unresolved"))
    module = ModuleType("native")
    vars(module)["Result"] = Result
    with pytest.raises(SurfaceValidationError, match="type.*identity"):
        static_namespace_members("native", module)


def test_non_string_type_module_is_not_replaced_with_an_unqualified_name():
    class Result:
        value = 1

    setattr(Result, "__module__", None)
    module = ModuleType("native")
    vars(module)["Result"] = Result
    with pytest.raises(SurfaceValidationError, match="type.*identity"):
        static_namespace_members("native", module)


@pytest.mark.parametrize("owner_kind", ("class", "module", "instance"))
def test_namespace_keys_cannot_execute_equality_hooks(owner_kind):
    comparisons = []

    class ModuleKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            comparisons.append(other)
            return str.__eq__(self, other)

    if owner_kind == "class":
        owner = type("State", (), {ModuleKey("__module__"): "native", "position": 1})
    elif owner_kind == "module":
        owner = ModuleType("native")
        del vars(owner)["__name__"]
        vars(owner)[ModuleKey("__name__")] = "native"
    else:
        owner = type("State", (), {})()
        vars(owner)[ModuleKey("position")] = 1
    comparisons.clear()
    with pytest.raises(SurfaceValidationError, match="namespace.*keys"):
        static_namespace_members("native", owner)
    assert comparisons == []
