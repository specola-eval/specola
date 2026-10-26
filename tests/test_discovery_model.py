from __future__ import annotations

import pytest

from specola.cli_common import _MockAdapter
from specola.adapter import PropertyMeta
from specola.discovery_model import build_discovery_model, property_metadata_for


def test_model_preserves_every_mock_entity_and_property():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities

    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
    )

    assert len(model.entities) == len(entities)
    assert sum(len(entity.properties) for entity in model.entities) == sum(
        len(entity.properties) for entity in entities.values()
    )
    robot = next(entity for entity in model.entities if entity.entity_id == 0)
    qpos = next(prop for prop in robot.properties if prop.key == "qpos_joint1")
    assert (qpos.group, qpos.unit, qpos.tags) == (
        "joint angle", "rad", (),
    )


def test_model_keeps_declared_dynamic_class_without_live_instance():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.capabilities = type("Capabilities", (), {
        "components_by_class": {"contact": frozenset({"force"})},
        "runtime_populated_classes": frozenset({"contact"}),
        "component_tags_by_class": {"contact": frozenset({"robot"})},
    })()

    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )

    declared = next(item for item in model.declared_classes
                    if item.entity_class == "contact")
    assert declared.live_count == 0
    assert declared.components == ("force",)
    assert declared.tags == ("robot",)


def test_model_transports_declaration_and_property_metadata_losslessly():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.entity_roles = {1: "target"}
    adapter.entity_role_basis = {1: "benchmark_declaration"}
    adapter.entity_role_provenance = {1: "task.objects[0]"}
    adapter.entity_declared_names = {1: "soup"}
    adapter.entity_declared_categories = {1: "food"}
    adapter.entity_declaration_sources = {1: "task.objects"}
    original = adapter.property_metadata

    def metadata(entity_id):
        if entity_id == 1:
            return {
                "mass": PropertyMeta(
                    group="mass",
                    note="legacy annotation",
                    tags=("role:target",),
                    unit="kg",
                    family="object_mass",
                    component="scalar",
                    frame="body",
                )
            }
        return original(entity_id)

    adapter.property_metadata = metadata
    model = build_discovery_model(
        adapter, entities, simulator="mock", task="task:0",
        effective_dt=adapter.dt,
    )

    entity = next(item for item in model.entities if item.entity_id == 1)
    prop = next(item for item in entity.properties if item.key == "mass")
    assert (
        entity.role,
        entity.role_basis,
        entity.role_provenance,
        entity.declared_name,
        entity.declared_category,
        entity.declaration_source,
    ) == (
        "target",
        "benchmark_declaration",
        "task.objects[0]",
        "soup",
        "food",
        "task.objects",
    )
    assert (
        prop.group,
        prop.unit,
        prop.family,
        prop.component,
        prop.frame,
        prop.note,
        prop.tags,
    ) == (
        "mass",
        "kg",
        "object_mass",
        "scalar",
        "body",
        "legacy annotation",
        ("role:target",),
    )


@pytest.mark.parametrize(
    "original_cause",
    [ValueError("fixture metadata failure"), KeyError("fixture key")],
)
def test_property_metadata_failure_identifies_adapter_and_entity(
    original_cause,
):
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities

    def metadata(entity_id):
        if entity_id == 2:
            raise original_cause
        return {}

    adapter.property_metadata = metadata

    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter _MockAdapter entity 2",
    ) as exc_info:
        build_discovery_model(
            adapter,
            entities,
            simulator="mock",
            task=None,
            effective_dt=adapter.dt,
        )

    assert exc_info.value.__cause__ is original_cause


def test_property_metadata_does_not_wrap_base_exceptions():
    adapter = _MockAdapter()

    def metadata(entity_id):
        raise KeyboardInterrupt()

    adapter.property_metadata = metadata

    with pytest.raises(KeyboardInterrupt):
        property_metadata_for(adapter, [0])


def test_property_metadata_explicit_empty_mapping_is_valid():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.property_metadata = lambda entity_id: {}

    metadata = property_metadata_for(adapter, entities)

    assert metadata == {entity_id: {} for entity_id in entities}


def test_property_metadata_none_compatibility_is_explicit():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.property_metadata = lambda entity_id: None

    metadata = property_metadata_for(adapter, entities)

    assert metadata == {entity_id: {} for entity_id in entities}


def test_property_metadata_mapping_truthiness_is_not_evaluated():
    class TruthinessForbiddenMapping(dict):
        def __bool__(self):
            raise AssertionError("metadata mapping truthiness was evaluated")

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    declared = TruthinessForbiddenMapping()
    adapter.property_metadata = lambda entity_id: declared

    metadata = property_metadata_for(adapter, entities)

    assert all(value is declared for value in metadata.values())


def test_nonempty_falsey_property_metadata_survives_model_build():
    class FalseyMapping(dict):
        def __bool__(self):
            return False

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    declared = FalseyMapping({
        "qpos_joint1": PropertyMeta(group="joint angle", note="rad"),
    })

    def metadata(entity_id):
        return declared if entity_id == 0 else {}

    adapter.property_metadata = metadata
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
    )

    robot = next(entity for entity in model.entities if entity.entity_id == 0)
    qpos = next(prop for prop in robot.properties if prop.key == "qpos_joint1")
    assert (qpos.group, qpos.unit) == ("joint angle", "rad")


def test_dynamic_entity_metadata_failure_preserves_dynamic_entity_id():
    adapter = _MockAdapter()
    original_cause = LookupError("late entity metadata missing")

    def metadata(entity_id):
        raise original_cause

    adapter.property_metadata = metadata

    with pytest.raises(RuntimeError, match=r"adapter _MockAdapter entity 99") as exc_info:
        property_metadata_for(adapter, [99])

    assert exc_info.value.__cause__ is original_cause
