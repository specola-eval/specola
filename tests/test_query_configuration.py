"""A configured native query subject is separate from formulas and scope."""

import json
from dataclasses import FrozenInstanceError

import pytest

from specola.query_config import QueryConfiguration, QueryConfigurationError, load_query_configuration


def _document():
    return {"schema": "specola.queries.v1", "populations": [
        {"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": 2.5}},
    ], "instances": [{
        "binding": "mujoco.mj_geomDistance", "id": "inspection",
        "operands": {"geom1": {"kind": "geom", "id": 0}, "geom2": {"kind": "geom", "name": "obstacle"}},
        "arguments": {"distmax": 0.5},
    }]}


def test_immutable_configuration_detaches_input_and_accessor_results():
    document = _document()
    config = QueryConfiguration.from_document(document)
    original = config.sha256
    document["populations"][0]["arguments"]["distmax"] = 7.0
    returned = config.population_arguments("mujoco.mj_geomDistance")
    assert returned is not None
    returned["distmax"] = 9.0
    config.instances()[0]["id"] = "changed"
    assert config.document() == _document()
    assert config.sha256 == original
    with pytest.raises(FrozenInstanceError):
        config.canonical_json = "changed"  # pyright: ignore[reportAttributeAccessIssue] - mutation negative control


def test_equivalent_mapping_order_has_same_identity_but_changed_arguments_do_not():
    document = _document()
    reordered = dict(reversed(tuple(document.items())))
    before = QueryConfiguration.from_document(document)
    assert before == QueryConfiguration.from_document(reordered)
    document["populations"][0]["arguments"]["distmax"] = 1.0
    assert before.sha256 != QueryConfiguration.from_document(document).sha256


def test_empty_document_has_no_fabricated_cutoff_or_instance():
    config = QueryConfiguration.empty()
    assert config.population_arguments("mujoco.mj_geomDistance") is None
    assert config.instances() == ()


@pytest.mark.parametrize("mutation", (
    lambda d: d.update(schema="unknown"),
    lambda d: d.update(specifications=[]),
    lambda d: d.pop("instances"),
    lambda d: d["populations"].append(d["populations"][0]),
    lambda d: d["instances"].append(d["instances"][0]),
    lambda d: d["instances"][0].update(id="bad token"),
    lambda d: d["instances"][0].update(id="inspection\n"),
    lambda d: d["instances"][0]["operands"]["geom1"].update(name="both"),
    lambda d: d["instances"][0]["operands"]["geom1"].update(id=True),
    lambda d: d["instances"][0]["operands"]["geom1"].update(id=2.0),
    lambda d: d["instances"][0]["operands"]["geom1"].update(kind=" "),
    lambda d: d["populations"][0].update(binding=" "),
    lambda d: d["populations"][0].update(binding="mujoco.mj_geomDistance "),
    lambda d: d["populations"][0]["arguments"].update(population="scoped"),
    lambda d: d["populations"][0]["arguments"].update(population=True),
))
def test_unknown_missing_duplicate_and_ambiguous_declarations_fail(mutation):
    document = _document()
    mutation(document)
    with pytest.raises(QueryConfigurationError):
        QueryConfiguration.from_document(document)


def test_population_all_is_the_one_population_directive():
    """``population: all`` asks for the full pair universe under a scope (issue #136)."""
    document = _document()
    document["populations"][0]["arguments"]["population"] = "all"
    config = QueryConfiguration.from_document(document)
    assert config.population_arguments("mujoco.mj_geomDistance") == {"distmax": 2.5, "population": "all"}


@pytest.mark.parametrize("value", (float("inf"), float("-inf"), float("nan"), object(), {1: "not a string key"}, (1, 2)))
def test_nonfinite_and_non_json_arguments_fail(value):
    document = _document()
    document["populations"][0]["arguments"]["distmax"] = value
    with pytest.raises(QueryConfigurationError):
        QueryConfiguration.from_document(document)


def test_cyclic_configuration_is_not_an_executable_or_recursive_object_graph():
    document = _document()
    document["instances"].append(document)
    with pytest.raises(QueryConfigurationError, match="cycle"):
        QueryConfiguration.from_document(document)


@pytest.mark.parametrize("suffix, source", (
    (".json", '{"schema":"specola.queries.v1","schema":"specola.queries.v1","instances":[],"populations":[]}'),
    (".yaml", "schema: specola.queries.v1\nschema: specola.queries.v1\ninstances: []\npopulations: []\n"),
    (".yaml", "!!python/object/apply:os.system ['must not execute']"),
    (".yaml", "schema: specola.queries.v1\ninstances: []\npopulations: []\n1: value\n"),
))
def test_duplicate_and_executable_document_syntax_is_rejected(tmp_path, suffix, source):
    path = tmp_path / ("queries" + suffix)
    path.write_text(source)
    with pytest.raises(QueryConfigurationError):
        load_query_configuration(path)


def test_json_exponents_and_yaml_round_trip_to_the_same_configuration(tmp_path):
    path = tmp_path / "queries.json"
    source = json.dumps(_document()).replace("2.5", "25e-1")
    path.write_text(source)
    assert load_query_configuration(path) == QueryConfiguration.from_document(_document())
    yaml_path = tmp_path / "queries.yaml"
    yaml_path.write_text("schema: specola.queries.v1\npopulations: []\ninstances: []\n")
    assert load_query_configuration(yaml_path) == QueryConfiguration.empty()


def test_direct_constructor_cannot_bypass_canonical_validation():
    with pytest.raises(QueryConfigurationError, match="not canonical"):
        QueryConfiguration(json.dumps(_document()))
    with pytest.raises(QueryConfigurationError):
        QueryConfiguration('{"schema":"unknown"}')


def test_schema_registry_never_retrieves_external_references(monkeypatch):
    import specola.query_config as module

    def cannot_retrieve(*args, **kwargs):
        pytest.fail("query validation attempted network retrieval")

    monkeypatch.setattr("urllib.request.urlopen", cannot_retrieve)
    module._validator.cache_clear()
    try:
        assert QueryConfiguration.empty().instances() == ()
        monkeypatch.setattr(module, "_schema_document", lambda: {"$ref": "https://invalid.example/schema"})
        module._validator.cache_clear()
        with pytest.raises(Exception, match="Unresolvable"):
            QueryConfiguration.empty()
    finally:
        module._validator.cache_clear()
