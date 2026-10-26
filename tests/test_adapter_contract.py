"""Normative simulator-adapter boundary and its executable invariants."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from specola.adapter import MonitorObservation, SimulatorAdapter
from specola.discovery_model import property_metadata_for
from specola.entities import EntityState, SpecConfigError, TrackState


REPO = Path(__file__).resolve().parents[1]


class _SequenceAdapter(SimulatorAdapter):
    """Small direct adapter whose frames exercise only the base contract."""

    def __init__(self, frames):
        super().__init__()
        self._frames = list(frames)
        self._index = 0

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    def _reset_impl(self):
        self._index = 0
        return self._observation()

    def _get_observation_impl(self):
        self._index += 1
        return self._observation()

    def _observation(self):
        entity_class, properties = self._frames[self._index]
        entity = EntityState(
            entity_id=7,
            entity_class=entity_class,
            position=np.zeros(3),
            velocity=np.zeros(3),
            track_age=self._index,
            track_state=TrackState.CONFIRMED,
            properties=dict(properties),
        )
        return MonitorObservation(
            timestep=self._index,
            time=self._index * self.dt,
            perceived_entities={entity.entity_id: entity},
        )


class _ObservationSequenceAdapter(SimulatorAdapter):
    """Adapter for adversarial observations not representable by normal dicts."""

    def __init__(self, frames):
        super().__init__()
        self._frames = tuple(frames)
        self._index = 0

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    def _reset_impl(self):
        self._index = 0
        return self._frames[self._index]

    def _get_observation_impl(self):
        self._index += 1
        return self._frames[self._index]


def _entity(entity_id, entity_class="object", properties=None, track_age=0):
    return EntityState(
        entity_id=entity_id,
        entity_class=entity_class,
        position=np.zeros(3),
        velocity=np.zeros(3),
        track_age=track_age,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties or {"mass": 1.0}),
    )


class _StringAlias:
    """Non-string key that deliberately aliases a validated string key."""

    def __hash__(self):
        return hash("mass")

    def __eq__(self, other):
        return other == "mass"


class _ClassAlias:
    """Non-string class that deliberately aliases the validated class."""

    def __eq__(self, other):
        return other == "object"


class _UnhashableKeyMapping(Mapping):
    """Adversarial Mapping whose items view yields a non-hashable key."""

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 1

    def items(self):
        return [([], 1.0)]


def test_property_keys_may_appear_and_disappear_between_observations():
    adapter = _SequenceAdapter(
        [
            ("object", {"mass": 1.0}),
            ("object", {"temperature": 20.0}),
            ("object", {"mass": 1.0, "charge": -0.2}),
        ]
    )

    assert set(adapter.reset().perceived_entities[7].properties) == {"mass"}
    assert adapter.active_properties_by_entity() == {7: frozenset({"mass"})}
    assert set(adapter.get_observation().perceived_entities[7].properties) == {
        "temperature"
    }
    assert adapter.active_properties_by_entity() == {
        7: frozenset({"temperature"})
    }
    assert set(adapter.get_observation().perceived_entities[7].properties) == {
        "mass",
        "charge",
    }
    assert adapter.active_properties_by_entity() == {
        7: frozenset({"mass", "charge"})
    }


@pytest.mark.parametrize(
    ("invalid_name", "error_type", "message"),
    [
        ("", ValueError, "empty property name"),
        (1, TypeError, "property names must be strings"),
        (_StringAlias(), TypeError, "property names must be strings"),
    ],
)
def test_property_names_are_validated_when_they_first_appear_after_reset(
    invalid_name, error_type, message
):
    adapter = _SequenceAdapter(
        [
            ("object", {"mass": 1.0}),
            ("object", {invalid_name: 2.0}),
        ]
    )
    adapter.reset()

    with pytest.raises(error_type, match=message):
        adapter.get_observation()


def test_rejected_later_entity_rolls_back_earlier_entity_name_staging():
    adapter = _ObservationSequenceAdapter(
        (
            MonitorObservation(
                timestep=0,
                time=0.0,
                perceived_entities={1: _entity(1), 2: _entity(2)},
            ),
            MonitorObservation(
                timestep=1,
                time=0.1,
                perceived_entities={
                    1: _entity(1, properties={"new": 2.0}, track_age=1),
                    2: _entity(2, properties={"bad": np.nan}, track_age=1),
                },
            ),
        )
    )
    adapter.reset()

    with pytest.raises(ValueError, match="non-finite"):
        adapter.get_observation()

    assert adapter._validated_property_names_by_entity == {
        1: {"mass"},
        2: {"mass"},
    }


def test_unhashable_non_string_property_name_gets_the_contract_error():
    entity = EntityState(
        entity_id=7,
        entity_class="object",
        position=np.zeros(3),
        velocity=np.zeros(3),
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=_UnhashableKeyMapping(),
    )

    with pytest.raises(TypeError, match="property names must be strings"):
        SimulatorAdapter._validate_property_values(entity, set())


def test_entity_class_mutation_under_one_id_still_fails():
    adapter = _SequenceAdapter(
        [("object", {"mass": 1.0}), ("robot", {"mass": 1.0})]
    )
    adapter.reset()

    with pytest.raises(SpecConfigError, match="changed class"):
        adapter.get_observation()


def test_boolean_dictionary_id_cannot_alias_an_existing_integer_id():
    adapter = _ObservationSequenceAdapter(
        (
            MonitorObservation(0, 0.0, {1: _entity(1)}),
            MonitorObservation(1, 0.1, {True: _entity(1, track_age=1)}),
        )
    )
    adapter.reset()

    with pytest.raises(TypeError, match="dictionary keys must be integers"):
        adapter.get_observation()


@pytest.mark.parametrize("invalid_id", [True, 1.0])
def test_repeated_payload_entity_id_is_still_type_checked(invalid_id):
    adapter = _ObservationSequenceAdapter(
        (
            MonitorObservation(0, 0.0, {1: _entity(1)}),
            MonitorObservation(
                1, 0.1, {1: _entity(invalid_id, track_age=1)}
            ),
        )
    )
    adapter.reset()

    with pytest.raises(TypeError, match="entity_id must be an integer"):
        adapter.get_observation()


def test_repeated_payload_entity_class_is_still_type_checked():
    adapter = _ObservationSequenceAdapter(
        (
            MonitorObservation(0, 0.0, {1: _entity(1)}),
            MonitorObservation(
                1, 0.1, {1: _entity(1, _ClassAlias(), track_age=1)}
            ),
        )
    )
    adapter.reset()

    with pytest.raises(TypeError, match="entity_class must be str"):
        adapter.get_observation()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_dynamic_properties_still_fail(value):
    adapter = _SequenceAdapter([("object", {"signal": value})])

    with pytest.raises(ValueError, match="non-finite"):
        adapter.reset()


def test_generic_interface_keeps_exactly_four_abstract_members():
    assert SimulatorAdapter.__abstractmethods__ == frozenset(
        {"dt", "spatial_dims", "_reset_impl", "_get_observation_impl"}
    )


def test_simulator_adapter_owns_the_normative_semantic_boundary():
    doc = " ".join((SimulatorAdapter.__doc__ or "").split())

    for required in (
        "Field inclusion is a semantic commitment",
        "Field omission is a semantic commitment",
        "eagerly materialised",
        # Compiled specs decide what is read, never what it means: the
        # scope is a projection of the full surface, and only a monitor
        # session installs one (issue #96, Task 5).
        "Compiled specifications never decide what a field *means*",
        "The scope is a projection, never a redefinition",
        "success",
        "verdict",
        "sufficient generic operands and state",
        "independent of compiled specifications, goals, and criteria",
        "false or absent relation is semantically meaningful",
        "static or dynamic, ordered or unordered",
        "never emits the criterion's answer",
        "Engine introspection",
        "dynamic contacts",
    ):
        assert required in doc


def test_entity_documentation_does_not_claim_a_fixed_property_schema():
    source = (REPO / "src" / "specola" / "entities.py").read_text()

    assert "fixed property schema" not in source
    assert "keys must remain fixed" not in source
    assert "locked schema" not in source
    assert "Property keys may appear or disappear" in source
    assert "Property values and keys may change at every timestep" in source


class _FalseyMapping(Mapping):
    """Mapping whose false truth value must not suppress validation."""

    def __init__(self, pairs):
        self._pairs = tuple(pairs)

    def __getitem__(self, key):
        for candidate, value in self._pairs:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._pairs)

    def __len__(self):
        return len(self._pairs)

    def __bool__(self):
        return False


class _TupleSubclass(tuple):
    """Tuple subclass rejected by the exact runtime tag representation."""


@pytest.mark.parametrize(
    ("component_tags", "error_type", "message"),
    [
        ("not-a-mapping", TypeError, "component_tags must be a mapping"),
        (
            {1: "not-a-mapping"},
            TypeError,
            "component_tags for entity 1 must be a mapping",
        ),
        (
            {True: {"mass": ("arm",)}},
            TypeError,
            "component_tags entity IDs must be integers",
        ),
        (
            {-1: {"mass": ("arm",)}},
            ValueError,
            "component_tags entity IDs must be nonnegative",
        ),
        (
            {1.0: {"mass": ("arm",)}},
            TypeError,
            "component_tags entity IDs must be integers",
        ),
        ({2: {"mass": ("arm",)}}, ValueError, "does not name a present entity"),
        ({1: {"": ("arm",)}}, ValueError, "empty component name"),
        ({1: {1: ("arm",)}}, TypeError, "component names must be strings"),
        (
            {1: {_StringAlias(): ("arm",)}},
            TypeError,
            "component names must be strings",
        ),
        ({1: {"missing": ("arm",)}}, ValueError, "does not name a property"),
        ({1: {"mass": ["arm"]}}, TypeError, "tags must be a tuple"),
        (
            {1: {"mass": _TupleSubclass(("arm",))}},
            TypeError,
            "tags must be a tuple",
        ),
        ({1: {"mass": ()}}, ValueError, "tags must be nonempty"),
        ({1: {"mass": ("",)}}, ValueError, "tags must be nonempty strings"),
        ({1: {"mass": (1,)}}, TypeError, "tags must be strings"),
    ],
)
def test_component_tags_reject_malformed_structure_without_state_commit(
    component_tags, error_type, message
):
    initial = MonitorObservation(0, 0.0, {1: _entity(1)})
    rejected = MonitorObservation(
        1, 0.1, {1: _entity(1, track_age=1)}, component_tags
    )
    adapter = _ObservationSequenceAdapter((initial, rejected))
    adapter.reset()

    with pytest.raises(error_type, match=message):
        adapter.get_observation()

    assert adapter.last_observation is initial
    assert adapter._validated_entity_ids == {1}
    assert adapter._validated_property_names_by_entity == {1: {"mass"}}


def test_component_tags_accept_falsey_outer_and_inner_mappings():
    tags = _FalseyMapping(
        ((1, _FalseyMapping((("mass", ("arm", "left")),))),)
    )
    adapter = _ObservationSequenceAdapter(
        (MonitorObservation(0, 0.0, {1: _entity(1)}, tags),)
    )

    observation = adapter.reset()

    assert observation.component_tags is tags


def test_component_tags_reject_unhashable_non_string_component_name():
    tags = _FalseyMapping(((1, _FalseyMapping((([], ("arm",)),))),))
    adapter = _ObservationSequenceAdapter(
        (MonitorObservation(0, 0.0, {1: _entity(1)}, tags),)
    )

    with pytest.raises(TypeError, match="component names must be strings"):
        adapter.reset()


def test_component_tags_reject_unhashable_entity_id_before_reconciliation():
    tags = _FalseyMapping((([], _FalseyMapping((("mass", ("arm",)),))),))
    adapter = _ObservationSequenceAdapter(
        (MonitorObservation(0, 0.0, {1: _entity(1)}, tags),)
    )

    with pytest.raises(
        TypeError, match="component_tags entity IDs must be integers"
    ):
        adapter.reset()


def test_component_tags_may_appear_disappear_and_reappear_with_properties():
    adapter = _ObservationSequenceAdapter(
        (
            MonitorObservation(
                0, 0.0, {1: _entity(1, properties={"mass": 1.0})},
                {1: {"mass": ("body",)}},
            ),
            MonitorObservation(
                1,
                0.1,
                {
                    1: _entity(
                        1, properties={"temperature": 2.0}, track_age=1
                    )
                },
            ),
            MonitorObservation(
                2, 0.2, {1: _entity(1, properties={"mass": 3.0}, track_age=2)},
                {1: {"mass": ("body",)}},
            ),
        )
    )

    adapter.reset()
    adapter.get_observation()
    assert adapter.get_observation().component_tags == {
        1: {"mass": ("body",)}
    }


def test_component_tags_failure_rolls_back_new_entity_validation():
    initial = MonitorObservation(0, 0.0, {1: _entity(1)})
    malformed = MonitorObservation(
        1,
        0.1,
        {
            1: _entity(1, track_age=1),
            2: _entity(2, track_age=0),
        },
        {2: {"missing": ("arm",)}},
    )
    recovered = MonitorObservation(
        2,
        0.2,
        {
            1: _entity(1, track_age=2),
            2: _entity(2, track_age=1),
        },
        {2: {"mass": ("arm",)}},
    )
    adapter = _ObservationSequenceAdapter((initial, malformed, recovered))
    adapter.reset()

    with pytest.raises(ValueError, match="does not name a property"):
        adapter.get_observation()

    assert adapter.last_observation is initial
    assert adapter._validated_entity_ids == {1}

    assert adapter.get_observation() is recovered
    assert adapter._validated_entity_ids == {1, 2}


def test_base_scope_is_exact_property_and_tag_projection():
    robot = _entity(
        7,
        entity_class="robot",
        properties={"keep": 1.0, "drop": 2.0},
    )
    object_entity = _entity(
        8,
        entity_class="object",
        properties={"mass": 3.0},
    )
    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities={7: robot, 8: object_entity},
        component_tags={
            7: {"keep": ("selected",), "drop": ("unselected",)},
            8: {"mass": ("physical",)},
        },
    )
    full_adapter = _ObservationSequenceAdapter((observation,))
    scoped_adapter = _ObservationSequenceAdapter((observation,))
    scoped_adapter.set_extraction_scope({"robot": frozenset({"keep"})})

    full = full_adapter.reset()
    scoped = scoped_adapter.reset()

    assert scoped.timestep == full.timestep
    assert scoped.time == full.time
    assert set(scoped.perceived_entities) == set(full.perceived_entities)
    assert scoped.perceived_entities[7].entity_class == "robot"
    assert scoped.perceived_entities[7].position is robot.position
    assert scoped.perceived_entities[7].velocity is robot.velocity
    assert scoped.perceived_entities[7].properties == {"keep": 1.0}
    assert scoped.perceived_entities[8].properties == {}
    assert scoped.component_tags == {7: {"keep": ("selected",)}, 8: {}}
    assert full.perceived_entities[7].properties == {
        "keep": 1.0,
        "drop": 2.0,
    }


def test_scope_projects_metadata_only_after_scoped_observation_is_accepted():
    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities={
            7: _entity(
                7,
                entity_class="robot",
                properties={"keep": 1.0, "drop": 2.0},
            )
        },
    )
    later = MonitorObservation(
        timestep=1,
        time=0.1,
        perceived_entities={
            7: _entity(
                7,
                entity_class="robot",
                track_age=1,
                properties={"keep": 1.0, "drop": 2.0},
            )
        },
    )
    adapter = _ObservationSequenceAdapter((observation, later))
    adapter.reset()
    setattr(
        adapter,
        "property_metadata",
        lambda entity_id: {"keep": object(), "drop": object()},
    )

    assert set(property_metadata_for(adapter, (7,))[7]) == {"keep", "drop"}

    adapter.set_extraction_scope({"robot": frozenset({"keep"})})

    assert set(property_metadata_for(adapter, (7,))[7]) == {"keep", "drop"}

    adapter.get_observation()

    assert set(property_metadata_for(adapter, (7,))[7]) == {"keep"}


def test_scope_does_not_hide_invalid_tag_references():
    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities={
            7: _entity(7, entity_class="robot", properties={"keep": 1.0})
        },
        component_tags={7: {"ghost": ("invalid",)}},
    )
    adapter = _ObservationSequenceAdapter((observation,))
    adapter.set_extraction_scope({"robot": frozenset()})

    with pytest.raises(ValueError, match="does not name a property"):
        adapter.reset()


@pytest.mark.parametrize("labels", [(), ("",), (1,)])
def test_scope_does_not_hide_malformed_labels_for_scoped_out_property(labels):
    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities={
            7: _entity(7, entity_class="robot", properties={"drop": 1.0})
        },
        component_tags={7: {"drop": labels}},
    )
    adapter = _ObservationSequenceAdapter((observation,))
    adapter.set_extraction_scope({"robot": frozenset()})

    with pytest.raises((TypeError, ValueError)):
        adapter.reset()


def test_scope_does_not_normalize_invalid_entity_mapping_shape():
    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities=cast(dict, MappingProxyType({})),
    )
    adapter = _ObservationSequenceAdapter((observation,))
    adapter.set_extraction_scope({})

    with pytest.raises(
        TypeError,
        match="MonitorObservation.perceived_entities must be a dict",
    ):
        adapter.reset()


def test_scoped_out_nonfinite_property_is_removed_before_common_validation():
    """Values outside compiled scope are not part of monitored output."""

    observation = MonitorObservation(
        timestep=0,
        time=0.0,
        perceived_entities={
            7: _entity(
                7,
                entity_class="robot",
                properties={"keep": 1.0, "drop": float("nan")},
            )
        },
    )
    adapter = _ObservationSequenceAdapter((observation,))
    adapter.set_extraction_scope({"robot": frozenset({"keep"})})

    scoped = adapter.reset()

    assert scoped.perceived_entities[7].properties == {"keep": 1.0}
