"""Schema authoring never invokes native observation preparation or readers."""

from dataclasses import replace
import json
import sys

import pytest

from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.provider import ExtractionGuardError, ExtractionProvider
from specola.conformance.query_bindings import QueryBindingError
from specola.query_config import QueryConfiguration
from test_extraction_instances import _ProviderAdapter
from test_query_contract import _fixture


def _prepared(adapter):
    from specola.conformance.authoring import AuthoringOwnerDomain, PreparedAuthoring

    surface, results, manifest, binding, declaration = _fixture()
    evidence = (EvidenceIdentity("fixture native name table", "b" * 64),)
    return PreparedAuthoring(
        surface, results, manifest, (binding,), (declaration,),
        adapter.query_configuration, adapter.query_configuration_epoch,
        evidence[0],
        (AuthoringOwnerDomain("native_pair_ids", "geom_pair", "ordered native pairs",
                              (((0, 1), None), ((1, 0), None)), evidence),
         AuthoringOwnerDomain("native_geom_ids", "geom", "native geometry table",
                              (((0,), "first"), ((1,), "second")), evidence)),
        lambda: adapter.events.append("authoring post guard"),
    )


class _AuthoringProvider(ExtractionProvider):
    def guard_source(self, adapter):
        assert isinstance(adapter, _AuthoringAdapter)
        adapter.events.append("source guard")
        if adapter.damage == "source":
            raise ExtractionGuardError("producer_identity_mismatch", ("fixture drift",))

    def guard_runtime(self, adapter):
        assert isinstance(adapter, _AuthoringAdapter)
        adapter.events.append("runtime guard")
        if adapter.damage == "runtime":
            raise ExtractionGuardError("producer_identity_mismatch", ("fixture runtime drift",))

    def prepare_authoring(self, adapter):
        assert isinstance(adapter, _AuthoringAdapter)
        adapter.events.append("authoring")
        prepared = _prepared(adapter)
        if adapter.damage == "configuration":
            adapter.set_query_configuration(_configured())
        if adapter.damage == "unresolved":
            manifest = replace(prepared.manifest, quantities=(), coverage=(), classifications=tuple(
                replace(item, disposition=type(item.disposition).UNRESOLVED)
                for item in prepared.manifest.classifications))
            return replace(prepared, manifest=manifest)
        if adapter.damage == "result":
            return replace(prepared, results=replace(prepared.results, results=prepared.results.results[:-1]))
        if adapter.damage == "binding":
            return replace(prepared, bindings=())
        if adapter.damage == "domain":
            return replace(prepared, owner_domains=prepared.owner_domains[:-1])
        if adapter.damage == "duplicate":
            first = prepared.owner_domains[0]
            return replace(prepared, owner_domains=(replace(first, owners=first.owners + first.owners[:1]),)
                           + prepared.owner_domains[1:])
        if adapter.damage == "empty":
            return replace(prepared, owner_domains=(replace(prepared.owner_domains[0], owners=()),)
                           + prepared.owner_domains[1:])
        return prepared

    def prepare(self, adapter, *, reset):
        raise AssertionError("schema authoring invoked observation preparation")


class _AuthoringAdapter(_ProviderAdapter):
    _extraction_provider = _AuthoringProvider()

    def _reset_impl(self):
        raise AssertionError("schema authoring invoked an observation reader")


def _configured():
    return QueryConfiguration.from_document({"schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": "native.query", "arguments": {"cutoff": 2.0}}]})


def test_authoring_keeps_the_full_closed_schema_and_native_names_without_values():
    from specola.conformance.authoring import prepare_schema_authoring

    adapter = _AuthoringAdapter()
    document = prepare_schema_authoring(adapter)
    assert document is not None
    assert document["schema_only"] is True
    assert document["query_configuration_required"] is True
    assert document["status"] == "query_configuration_required"
    assert document["schema"] == "specola.extraction-authoring.v1"
    assert document["missing_arguments"] == [{"binding_id": "native.query", "arguments": ["cutoff"]}]
    assert {quantity["quantity_id"] for quantity in document["contract"]["quantities"]} == {
        "native_distance", "native_witness"}
    assert document["owner_domains"][1]["owners"] == [[[0], "first"], [[1], "second"]]
    assert "entities" not in document and "observations" not in document
    assert adapter.events == ["source guard", "bind", "authoring", "authoring post guard", "runtime guard"]
    assert adapter.last_observation is None and not adapter.episode_started
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize("damage", ("source", "runtime", "configuration", "unresolved", "result",
                                     "binding", "domain", "duplicate", "empty"))
def test_missing_arguments_cannot_bypass_a_broken_contract_or_native_subject(damage):
    from specola.conformance.authoring import prepare_schema_authoring

    adapter = _AuthoringAdapter(damage)
    with pytest.raises((ExtractionGuardError, QueryBindingError, ValueError)):
        prepare_schema_authoring(adapter)
    assert adapter.last_observation is None and not adapter.episode_started
    assert "read" not in adapter.events and "prepare" not in adapter.events


def test_configured_authoring_preflight_does_not_prepare_or_read_observations():
    from specola.conformance.authoring import prepare_schema_authoring

    adapter = _AuthoringAdapter()
    adapter.set_query_configuration(_configured())
    assert prepare_schema_authoring(adapter) is None
    assert adapter.last_observation is None
    assert "authoring post guard" in adapter.events


def test_legacy_adapter_and_optional_provider_interface_remain_ordinary():
    from specola.adapters.fake import _MockAdapter
    from specola.conformance.authoring import prepare_schema_authoring

    mock = _MockAdapter()
    assert prepare_schema_authoring(mock) is None
    assert mock.last_observation is None
    assert prepare_schema_authoring(_ProviderAdapter()) is None


@pytest.mark.parametrize("output_format", ("json", "table"))
def test_discover_uses_the_real_factory_and_emits_only_the_schema_document(monkeypatch, capsys, output_format):
    from specola import cli_common, discover

    adapter = _AuthoringAdapter()
    requests = []

    def factory(*args, **kwargs):
        requests.append((args, kwargs))
        return adapter

    monkeypatch.setattr(cli_common, "_load_sim", factory)
    monkeypatch.setattr(sys, "argv", ["specola discover", "--sim", "native_fixture", "--task", "task",
        "--robot", "robot", "--all-regions", "--format", output_format, "--no-pager"])
    discover.main()
    output = capsys.readouterr().out
    if output_format == "json":
        document = json.loads(output)
        assert document["schema_only"] is True
        assert document["subject"]["requested_adapter"] == "native_fixture"
    else:
        assert "schema_only: true" in output
        assert "query_configuration_required" in output
    assert requests == [(("native_fixture",), {"task": "task", "robot": "robot", "target_only": False})]
    assert adapter.last_observation is None


@pytest.mark.parametrize("extra", (("--web", "0"), ("--ui", "plain"), ("--class", "*"),
                                    ("--raw",), ("--field-contract",)))
def test_schema_only_discovery_rejects_ordinary_consumers_before_any_output(monkeypatch, tmp_path, extra):
    from specola import cli_common, discover

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: _AuthoringAdapter())
    monkeypatch.setattr(sys, "argv", ["specola discover", "--output", str(tmp_path / "output"), *extra])
    with pytest.raises(SystemExit, match="schema-only"):
        discover.main()
    assert list(tmp_path.iterdir()) == []


def test_bad_source_cannot_create_a_requested_schema_file(monkeypatch, tmp_path):
    from specola import cli_common, discover

    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: _AuthoringAdapter("source"))
    monkeypatch.setattr(sys, "argv", ["specola discover", "--output", str(tmp_path / "schema.json")])
    with pytest.raises(SystemExit):
        discover.main()
    assert list(tmp_path.iterdir()) == []


def test_query_missing_arguments_do_not_hide_invalid_supplied_arguments():
    from specola.conformance.query_bindings import validate_query_arguments, QueryConfigurationRequired

    _, _, manifest, binding, declaration = _fixture()
    configuration = QueryConfiguration.from_document({"schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": binding.binding_id, "arguments": {"undeclared": 2}}]})
    with pytest.raises(QueryBindingError) as caught:
        validate_query_arguments(configuration, (binding,), (declaration,),
                                 tuple(item.quantity_id for item in manifest.quantities))
    assert not isinstance(caught.value, QueryConfigurationRequired)


def test_authoring_output_is_detached_and_cannot_become_an_ordinary_discovery_model():
    from specola.conformance.authoring import prepare_schema_authoring
    from specola.discovery_restore import DiscoveryRestoreError, discovery_model_from_document

    adapter = _AuthoringAdapter()
    document = prepare_schema_authoring(adapter)
    assert document is not None
    with pytest.raises(DiscoveryRestoreError, match="ordinary"):
        discovery_model_from_document(document)
    document["owner_domains"][1]["owners"][0][1] = "invented name"
    second = prepare_schema_authoring(adapter)
    assert second is not None
    assert second["owner_domains"][1]["owners"][0][1] == "first"


def test_empty_native_name_population_requires_an_explicit_reason():
    adapter = _AuthoringAdapter()
    prepared = _prepared(adapter)
    empty = replace(prepared.owner_domains[0], owners=(), empty_reason="fixture declaration has no pair owners")
    replace(prepared, owner_domains=(empty,) + prepared.owner_domains[1:]).validate(adapter)
    with pytest.raises(ValueError, match="empty reason"):
        replace(prepared, owner_domains=(replace(empty, owners=prepared.owner_domains[0].owners),)
                + prepared.owner_domains[1:]).validate(adapter)


def test_unknown_bundled_producer_is_not_a_schema_authoring_fallback():
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance.authoring import prepare_schema_authoring

    # Construction is avoided: the class-bound guard must fail before handles,
    # properties, observation preparation or a simulator can be consulted.
    # The bounded provider's source guard refuses an adapter built without
    # the MuJoCo initializer before any native handle exists.
    adapter = object.__new__(MujocoAdapter)
    with pytest.raises(ExtractionGuardError, match="native_subject_unavailable"):
        prepare_schema_authoring(adapter)


def test_query_missing_root_argument_does_not_hide_nested_required_components():
    from specola.conformance.query_bindings import ArgumentSchema, QueryParameter, ParameterRole, validate_query_arguments

    _, _, manifest, binding, declaration = _fixture()
    schema = binding.arguments.document()
    schema["properties"]["options"] = {"type": "object", "properties": {"nested": {"type": "integer"}},
                                         "required": ["nested"], "additionalProperties": False}
    schema["required"].append("options")
    schema = ArgumentSchema.from_document(schema)
    binding = replace(binding, parameters=binding.parameters + (QueryParameter("options", ParameterRole.ARGUMENT, "native options"),),
                      arguments=schema, population_arguments=schema)
    declaration = replace(declaration, parameter_names=declaration.parameter_names + ("options",))
    configuration = QueryConfiguration.from_document({"schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": binding.binding_id, "arguments": {"options": {}}}]})
    with pytest.raises(QueryBindingError, match="nested"):
        validate_query_arguments(configuration, (binding,), (declaration,),
                                 tuple(item.quantity_id for item in manifest.quantities))


def test_discover_can_write_a_labeled_schema_file_after_guards(monkeypatch, tmp_path):
    from specola import cli_common, discover

    adapter = _AuthoringAdapter()
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: adapter)
    destination = tmp_path / "schema.json"
    monkeypatch.setattr(sys, "argv", ["specola discover", "--format", "json", "--output", str(destination)])
    discover.main()
    document = json.loads(destination.read_text())
    assert document["schema_only"] is True
    assert list(tmp_path.iterdir()) == [destination]
    assert adapter.last_observation is None


def test_capture_drift_is_rejected_before_schema_publication(monkeypatch):
    from specola.conformance.authoring import prepare_schema_authoring

    adapter = _AuthoringAdapter()
    provider = adapter._extraction_provider
    original = provider.prepare_authoring
    independent_generation = adapter.model_generation

    def prepare_with_drift(subject):
        prepared = original(subject)
        subject.model_generation += 1

        def reject_drift():
            if subject.model_generation != independent_generation:
                raise ExtractionGuardError("synthetic_capture_drift", ("model changed during capture",))

        return replace(prepared, guard_after_authoring=reject_drift)

    monkeypatch.setattr(provider, "prepare_authoring", prepare_with_drift)
    with pytest.raises(ExtractionGuardError, match="synthetic_capture_drift"):
        prepare_schema_authoring(adapter)
    assert adapter.last_observation is None and not adapter.episode_started
