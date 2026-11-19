"""Default cutoffs and unread discovery use the same eligible pair identities."""

import gzip
import json
import sys

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapter import PropertyMeta
from specola.adapters.mujoco import MujocoAdapter
from specola.adapters.mujoco_geom_pairs import CAPPED_KEY, CLIPPED_KEY
from specola.capabilities import reset_adapter, reset_catalog
from specola.discovery_capture import discovery_payload
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_render import render_discovery
from specola.discovery_restore import discovery_model_from_document, catalog_from_model, entity_states_from_model
from specola.query_config import QueryConfiguration
from test_mujoco_bounded_provider import _Env, _pairs, _queries, _spec_file
from test_mujoco_queries import A, B, _move_body


@pytest.fixture
def native_calls():
    calls = []
    original_profile = sys.getprofile()
    native = mujoco.mj_geomDistance
    def profile(frame, event, arg):
        if event == "c_call" and arg is native:
            calls.append(1)
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(original_profile)

@pytest.fixture
def default_adapter():
    adapter = MujocoAdapter(_Env())
    yield adapter
    adapter.close_native_extraction()


def test_adapter_starts_with_default_and_explicit_override_wins(default_adapter):
    assert default_adapter.query_configuration == QueryConfiguration.mujoco_default()
    assert default_adapter.query_configuration_is_default
    default_adapter.set_query_configuration(_queries(0.02))
    assert default_adapter.query_configuration == _queries(0.02)
    assert not default_adapter.query_configuration_is_default


def test_invalid_explicit_configuration_does_not_restore_default(default_adapter):
    from specola.conformance.query_bindings import QueryConfigurationRequired
    default_adapter.set_query_configuration(QueryConfiguration.empty())
    with pytest.raises(QueryConfigurationRequired):
        default_adapter.reset_catalog_only()
    assert default_adapter.query_configuration == QueryConfiguration.empty()
    assert not default_adapter.query_configuration_is_default
    assert not default_adapter._catalog_only_reset_active


@pytest.mark.parametrize("initial_read", [False, True])
def test_retained_reset_catalog_is_independent_of_later_pair_scope(default_adapter, initial_read):
    adapter = default_adapter
    adapter.set_extraction_entity_scope(None if initial_read else {})
    reset = reset_adapter(adapter)
    expected = {record.entity_id for record in adapter._geom_pair_records.values()}
    pair_id = adapter._geom_pair_records[(A, B)].entity_id
    adapter.set_extraction_entity_scope({} if initial_read else {
        "geom_pair": frozenset({(pair_id, ())}),
    })
    adapter.get_observation()
    # A subsequent reset replaces adapter-owned registries too.
    adapter.reset()
    document = json.loads(gzip.decompress(discovery_payload(
        adapter, reset, simulator="mujoco", task=None, effective_dt=adapter.dt,
    )))
    pairs = {item["id"]: item for item in document["entities"] if item["class"] == "geom_pair"}
    assert set(pairs) == expected
    if initial_read:
        for entity_id, entity in _pairs(reset.initial_observation).items():
            assert {prop["key"]: prop["value"] for prop in pairs[entity_id]["properties"]} == entity.properties
    else:
        assert all(prop["value"] is None for entity in pairs.values() for prop in entity["properties"])
    for entity_id, item in pairs.items():
        for prop in item["properties"]:
            original = reset.property_metadata[entity_id][prop["key"]]
            assert isinstance(original, PropertyMeta)
            assert prop["tags"] == list(original.tags)
            assert prop["group"] == original.group
            assert prop["unit"] == original.note


@pytest.mark.parametrize("cutoff", [0.05, 0.02])
def test_unread_catalog_round_trip_and_authoring(default_adapter, native_calls, tmp_path, cutoff):
    adapter = default_adapter
    adapter.set_query_configuration(_queries(cutoff))
    calls = native_calls
    adapter.set_extraction_entity_scope({})
    reset = reset_adapter(adapter)
    assert not _pairs(reset.initial_observation)
    document = json.loads(gzip.decompress(discovery_payload(
        adapter, reset, simulator="mujoco", task=None, effective_dt=adapter.dt,
    )))
    pairs = [item for item in document["entities"] if item["class"] == "geom_pair"]
    expected = {record.entity_id for record in adapter._geom_pair_records.values()}
    assert {item["id"] for item in pairs} == expected
    assert len(expected) == 15
    assert all(prop["value"] is None for item in pairs for prop in item["properties"])
    assert calls == []
    restored = discovery_model_from_document(document)
    catalog = catalog_from_model(restored)
    assert expected <= set(catalog.entities)
    assert not expected.intersection(entity_states_from_model(restored))
    from specola.check_specs import _format_value
    assert _format_value(None) == "not measured"
    assert _format_value(float("nan")) == "+nan"
    view = filter_discovery(restored, DiscoveryQuery(classes=("geom_pair",), include_raw=True))
    assert "not measured" in render_discovery(view, detailed=True, verbose=True)
    pair = adapter._geom_pair_records[(A, B)]
    authored = next(item for item in pairs if item["id"] == pair.entity_id)
    assert CAPPED_KEY in {prop["key"] for prop in authored["properties"]}
    from specola.session import Monitor
    session = Monitor(_spec_file(tmp_path, f'geom_pair_{authored["id"]}[{CAPPED_KEY}] > -1 and geom_pair_{authored["id"]}[{CLIPPED_KEY}] >= 0'), results_dir=None).watch(adapter)
    assert calls == []
    _move_body(adapter.env.model, adapter.env.data, "b", (0.21, 0.0, 0.3))
    session.step()
    observed = _pairs(adapter.last_observation)
    assert set(observed) == {pair.entity_id}
    assert observed[pair.entity_id].properties[CAPPED_KEY] == pytest.approx(0.01)
    assert observed[pair.entity_id].properties[CLIPPED_KEY] == 0.0
    assert len(calls) == 1
    _move_body(adapter.env.model, adapter.env.data, "b", (2.0, 0.0, 0.3))
    session.step()
    assert _pairs(adapter.last_observation)[pair.entity_id].properties[CAPPED_KEY] == cutoff
    assert _pairs(adapter.last_observation)[pair.entity_id].properties[CLIPPED_KEY] == 1.0
    from specola.discovery_model import property_metadata_for
    entities = adapter.last_observation.perceived_entities
    catalog, metadata = reset_catalog(adapter, entities, property_metadata_for(adapter, entities))
    model = build_discovery_model(adapter, entities, catalog=catalog, property_metadata=metadata,
        simulator="mujoco", task=None, effective_dt=adapter.dt)
    measured = next(item for item in model.entities if item.entity_id == pair.entity_id)
    assert {prop.key: prop.value for prop in measured.properties}[CAPPED_KEY] == cutoff
    assert {item.entity_id for item in model.entities if item.entity_class == "geom_pair"} == expected


@pytest.mark.parametrize("population_all", [False, True])
def test_standalone_discover_resets_catalog_only(monkeypatch, capsys, native_calls, population_all):
    from specola import discover
    from specola import cli_common
    made = []
    def construct(*args, **kwargs):
        adapter = MujocoAdapter(_Env())
        if population_all:
            document = _queries(0.02).document()
            document["populations"][0]["arguments"]["population"] = "all"
            adapter.set_query_configuration(QueryConfiguration.from_document(document))
        made.append(adapter)
        return adapter
    monkeypatch.setattr(cli_common, "construct_sim", construct)
    monkeypatch.setattr("sys.argv", ["specola discover", "--sim", "mujoco", "--format", "json"])
    try:
        discover.main()
        document = json.loads(capsys.readouterr().out)
        assert not _pairs(made[0].last_observation)
        assert len([item for item in document["entities"] if item["class"] == "geom_pair"]) == 15
        assert native_calls == []
        if population_all:
            assert made[0].query_configuration.population_arguments("mujoco.mj_geomDistance") == {"distmax": 0.02, "population": "all"}
            made[0].reset()
            assert len(_pairs(made[0].last_observation)) == 15
            assert len(native_calls) > 0
    finally:
        for adapter in made:
            adapter.close_native_extraction()
