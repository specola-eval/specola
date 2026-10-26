"""Core resolves adapter-extracted keys only; no built-in vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from specola.adapter import PropertyMeta
from specola.adapters.fake import FakeAdapter
from specola.components import (
    ComponentCatalog,
    ComponentResolver,
    discovered_components_for,
    validate_component_name,
)
from specola.entities import EntityState, TrackState


def _entity(properties):
    return EntityState(
        entity_id=1,
        entity_class="object",
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.zeros(3),
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def test_components_are_property_keys_only():
    entity = _entity({"xpos_2": 3.0, "qpos_hinge": 0.1})
    assert discovered_components_for(entity) == frozenset(
        {"xpos_2", "qpos_hinge"}
    )


def test_px_is_not_special_and_may_be_an_adapter_key():
    # the reserved-name check is gone: an adapter may name a key "px"
    entity = _entity({"px": 1.0})
    assert "px" in discovered_components_for(entity)


def test_resolver_rejects_removed_vocabulary():
    resolver = ComponentResolver({1: _entity({"xpos_2": 3.0})})
    with pytest.raises(Exception) as exc_info:
        resolver.value(1, "speed")
    assert "xpos_2" in str(exc_info.value)


def test_entity_state_allows_no_spatial_fields():
    entity = EntityState(
        entity_id=2,
        entity_class="signal",
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={"battery": 0.9},
    )
    assert discovered_components_for(entity) == frozenset({"battery"})
    resolver = ComponentResolver({2: entity})
    assert resolver.value(2, "battery") == 0.9


def test_adapter_validates_spatial_vector_none_passes_through():
    # Entities with no spatial fields (EntityState.position/velocity may be
    # None) must not be rejected by the adapter's observation validation.
    adapter = FakeAdapter()

    adapter._validate_spatial_vector(
        None,
        entity_id=1,
        vector_name="position",
    )
    adapter._validate_spatial_vector(
        None,
        entity_id=1,
        vector_name="velocity",
    )

    # Non-None vectors are still validated as before.
    with pytest.raises(TypeError):
        adapter._validate_spatial_vector(
            [1.0, 2.0, 3.0],
            entity_id=1,
            vector_name="position",
        )


def test_entity_state_defaults_position_and_velocity_to_none():
    entity = EntityState(
        entity_id=3,
        entity_class="signal",
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={"battery": 0.5},
    )
    assert entity.position is None
    assert entity.velocity is None


def test_entity_state_coerces_list_position_to_float_ndarray():
    entity = EntityState(
        entity_id=4,
        entity_class="object",
        track_age=0,
        track_state=TrackState.CONFIRMED,
        position=[1, 2, 3],
        properties={},
    )
    assert isinstance(entity.position, np.ndarray)
    assert entity.position.dtype == np.float64
    assert np.array_equal(entity.position, np.array([1.0, 2.0, 3.0]))


def test_catalog_carries_component_groups():
    entity = _entity({"qacc_j": 0.0, "xpos_b_2": 1.0})
    metadata = {1: {
        "qacc_j": PropertyMeta(group="joint angular acceleration",
                               note="rad/s²"),
        "xpos_b_2": PropertyMeta(group="body position", note="m"),
    }}
    catalog = ComponentCatalog.from_entities([entity], metadata=metadata)
    groups = catalog.component_groups_for_entity(1)
    assert groups["qacc_j"] == "joint angular acceleration"
    assert groups["xpos_b_2"] == "body position"


def test_catalog_groups_default_empty():
    catalog = ComponentCatalog.from_entities([_entity({"k": 1.0})])
    assert catalog.component_groups_for_entity(1) == {}


def test_supports_and_validate_reject_non_str_identically():
    """The two entry points must agree on the non-string rejection.

    ``ComponentResolver.supports`` consults its cache before validating, so
    it carries its own inlined ``isinstance`` guard rather than paying a call
    into the shared one on a hot path. That duplication is deliberate and
    measured, but nothing in the code stops the two messages from drifting;
    this pins them. Without the guard the argument would reach the cache-key
    tuple hash and surface as an unhashable-type error instead.
    """
    resolver = ComponentResolver({1: _entity({"mass": 1.0})})

    for bad in (123, [1, 2], None, 4.5):
        with pytest.raises(TypeError) as from_validate:
            validate_component_name(bad)
        with pytest.raises(TypeError) as from_supports:
            resolver.supports(1, bad)
        assert str(from_supports.value) == str(from_validate.value)


def test_supports_caches_a_negative_answer():
    """A cached ``False`` must survive the lookup, not be recomputed.

    ``supports`` reads its cache with a sentinel default precisely because
    the stored values are booleans: a plain ``dict.get(key)`` cannot tell a
    cached ``False`` from an absent key, so every negative answer would be
    recomputed and the reordering would silently lose its benefit on exactly
    the entities that fail eligibility.
    """
    entity = _entity({"mass": 1.0})
    resolver = ComponentResolver({1: entity})

    assert resolver.supports(1, "absent_key") is False
    assert resolver._support_cache[(1, "absent_key")] is False

    # Make the underlying answer differ, so a cached False and a recomputed
    # one are distinguishable: recomputation would now return True.
    entity.properties["absent_key"] = 2.0
    assert resolver.supports(1, "absent_key") is False, (
        "a cached False was recomputed; the sentinel default is not in effect"
    )

    # A pair never asked about still resolves against the live entity.
    assert resolver.supports(1, "mass") is True
