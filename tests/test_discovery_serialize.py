from __future__ import annotations

import json

from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_serialize import discovery_json


def test_json_round_trip_accounts_for_complete_mock_catalogue():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )

    document = json.loads(discovery_json(
        filter_discovery(model, DiscoveryQuery()),
    ))

    assert document["schema_version"] == 1
    assert len(document["entities"]) == len(entities)
    assert sum(len(entity["properties"]) for entity in document["entities"]) == sum(
        len(entity.properties) for entity in entities.values()
    )
    assert document["counts"]["before"]["entities"] == len(entities)


def test_document_top_level_keys_are_pinned():
    """The pin discovery_serialize's docstring says makes additions deliberate.

    It described this test before this test existed, so a key could be
    added or removed with nothing objecting — which is precisely the
    drift the comment claims is prevented. A consumer written against an
    older schema keeps parsing when a nullable key appears, so growth is
    fine; it just has to be chosen. Changing this set means changing the
    document every downstream reader sees.
    """
    from specola.cli_common import _MockAdapter
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_model import build_discovery_model
    from specola.discovery_serialize import discovery_document

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery())

    assert set(discovery_document(view)) == {
        "adapter_type",
        "capabilities",
        "classes",
        "counts",
        "declared_classes",
        "entities",
        "filters",
        "schema_version",
        "selector_palette",
        "simulator",
        "spatial_dims",
        "task",
        "task_description",
        "timing",
        "warnings",
    }
    assert set(
        discovery_document(view, include_field_contract=True)
    ) == set(discovery_document(view)) | {"field_contract"}


def test_field_contract_record_carries_every_key_the_adapter_reports():
    """The exported materialization record, pinned key by key.

    The field contract is a machine surface: a diagnostic the adapter
    fills and the document drops is invisible to everything downstream
    (issue #98). Extending this set is meant to be a deliberate act.
    """
    from specola.adapters.mujoco import MujocoFieldMaterialization
    from specola.adapters.mujoco_contract import FieldSurfaceReport

    refusal = "no gather plan: values came from 2 source arrays (xpos, qpos)"
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.field_surface_report = FieldSurfaceReport(
        included=("xpos",),
        excluded=(),
        unclassified=(),
        overlap=(),
    )
    adapter.field_materialization = {
        "xpos": MujocoFieldMaterialization(
            field_name="xpos",
            binding_present=True,
            scene_applicable=True,
            emitted_keys=("entity[0].properties.xpos_alpha_0",),
            reason=None,
            plan_refusal=refusal,
        ),
    }
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )

    document = json.loads(discovery_json(
        filter_discovery(model, DiscoveryQuery()),
        include_field_contract=True,
    ))

    record, = document["field_contract"]["materialization"]
    assert set(record) == {
        "field_name",
        "binding_present",
        "scene_applicable",
        "emitted_keys",
        "reason",
        "plan_refusal",
    }
    assert record["plan_refusal"] == refusal
