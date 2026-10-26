"""Persist producer query declarations without resolving or executing owners."""

from copy import deepcopy
import json

import pytest

from specola.conformance.query_bindings import QueryBindingError
from test_query_bindings import _fixture


def test_query_binding_round_trip_retains_native_roles_defaults_and_results():
    from specola.conformance.query_bindings import query_binding_document, query_binding_from_document

    binding, declaration = _fixture()
    document = json.loads(json.dumps(query_binding_document(binding, declaration)))
    actual_binding, actual_declaration = query_binding_from_document(document, binding.quantity_ids)
    assert actual_binding == binding
    assert actual_declaration == declaration
    assert actual_binding.quantity_ids == ("native.return", "native.witness")
    assert actual_binding.parameters[3].default_json is None


def test_query_binding_document_rejects_missing_or_extra_record_fields():
    from specola.conformance.query_bindings import query_binding_document, query_binding_from_document

    binding, declaration = _fixture()
    original = json.loads(json.dumps(query_binding_document(binding, declaration)))
    # JSON Schema itself permits arbitrary property names; these are the
    # surrounding fixed record dictionaries, derived from the actual document.
    paths = [(), ("binding",), ("declaration",), ("binding", "arguments"),
             ("binding", "population_arguments")]
    paths += [("binding", "parameters", index) for index in range(len(original["binding"]["parameters"]))]
    paths += [("binding", "provenance", index) for index in range(len(original["binding"]["provenance"]))]
    paths += [("declaration", "provenance", index)
              for index in range(len(original["declaration"]["provenance"]))]
    for path in paths:
        selected = original
        for component in path:
            selected = selected[component]
        for key in tuple(selected) + ("unused_new_record_field",):
            changed = deepcopy(original)
            target = changed
            for component in path:
                target = target[component]
            if key in target:
                del target[key]
            else:
                target[key] = True
            with pytest.raises(QueryBindingError):
                query_binding_from_document(changed, binding.quantity_ids)


def test_restoration_does_not_hide_an_omitted_native_output_parameter():
    from specola.conformance.query_bindings import query_binding_document, query_binding_from_document

    binding, declaration = _fixture()
    document = json.loads(json.dumps(query_binding_document(binding, declaration)))
    document["binding"]["parameters"] = [parameter for parameter in document["binding"]["parameters"]
                                            if parameter["name"] != "witness"]
    with pytest.raises(QueryBindingError):
        query_binding_from_document(document, binding.quantity_ids)


@pytest.mark.parametrize("field", ["parameters", "quantity_ids", "provenance", "completion_dependencies"])
def test_restoration_requires_native_record_sequence_types(field):
    from specola.conformance.query_bindings import query_binding_document, query_binding_from_document

    binding, declaration = _fixture()
    document = json.loads(json.dumps(query_binding_document(binding, declaration)))
    document["binding"][field] = "not a sequence of records"
    with pytest.raises(QueryBindingError):
        query_binding_from_document(document, binding.quantity_ids)


def test_argument_only_validation_leaves_native_owners_unresolved():
    from specola.conformance.query_bindings import validate_query_arguments
    from specola.query_config import QueryConfiguration
    from test_query_bindings import _document

    binding, declaration = _fixture()
    document = _document()
    # Existence is a native owner question, not a JSON argument question.
    document["instances"][0]["operands"]["first"]["name"] = "not_in_the_native_scene"
    configuration = QueryConfiguration.from_document(document)
    assert validate_query_arguments(configuration, (binding,), (declaration,), binding.quantity_ids) is None
    assert configuration.document() == document


@pytest.mark.parametrize("mutation", ["missing", "invalid", "owner_kind", "unknown_query"])
def test_argument_only_and_runtime_binding_share_rejections(mutation):
    from specola.conformance.query_bindings import bind_query_configuration, validate_query_arguments
    from specola.query_config import QueryConfiguration
    from test_query_bindings import _document

    binding, declaration = _fixture()
    document = _document()
    if mutation == "missing":
        document["populations"] = []
    elif mutation == "invalid":
        document["instances"][0]["arguments"]["cutoff"] = -1
    elif mutation == "owner_kind":
        document["instances"][0]["operands"]["first"]["kind"] = "body"
    else:
        document["instances"][0]["binding"] = "unused_new_query"
    configuration = QueryConfiguration.from_document(document)

    def forbidden_owner_resolution(*args):
        pytest.fail("invalid arguments must fail before native owner resolution")

    with pytest.raises(QueryBindingError) as pure:
        validate_query_arguments(configuration, (binding,), (declaration,), binding.quantity_ids)
    with pytest.raises(QueryBindingError) as runtime:
        bind_query_configuration(configuration, (binding,), (declaration,), binding.quantity_ids,
                                 forbidden_owner_resolution)
    assert type(pure.value) is type(runtime.value)
    assert str(pure.value) == str(runtime.value)
