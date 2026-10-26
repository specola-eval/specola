"""Runtime tag eligibility for declared-schema dynamic classes (D20 Part 2).

Static entity classes resolve ``@`` selector filters at compile time by
desugaring against the reset catalog. Dynamic classes (contact pairs)
are born mid-episode, so a compile-time expansion can never reach the
pair the spec is about. The engine extension under test:

- ``AdapterCapabilities.component_tags_by_class`` declares the possible
  structural tags a declared-CLOSED-schema class may carry at runtime;
- an ``@``-filtered exact-key selector on such a class compiles into a
  wildcard slot that RETAINS the filter (``ParameterSpec.
  component_tag_filters``) instead of desugaring — a label outside the
  declared tags, a key outside the declared schema, or an unsupported
  selector form fails compilation loudly;
- per-step binding eligibility evaluates the retained filter against
  ``MonitorObservation.component_tags``, so a matching pair binds at the
  step it is born and non-matching pairs never bind.

Scope guard: only declared-schema dynamic classes route this way;
static-class desugaring is untouched.
"""

from __future__ import annotations

import pytest

from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.components import ComponentCatalog
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.adapter import MonitorObservation
from specola.monitor import IncrementalBindingManager, enumerate_bindings
from specola.stl_parser import compile_specs


_CONTACT_COMPONENTS = frozenset(
    {
        "contact_point_force_max",
        "contact_normal_force_sum",
    }
)

_DECLARED_TAGS = frozenset(
    {"role:target", "role:distractor", "role:robot"}
)

_FILTERED = (
    "G[0,1s](robot_contact_*"
    "[contact_point_force_max @ role:distractor] <= 50)"
)


def _catalog(with_contact: bool = False) -> ComponentCatalog:
    entities = {
        0: EntityState(
            entity_id=0,
            entity_class="object",
            position=None,
            velocity=None,
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties={"height": 1.0},
        )
    }
    if with_contact:
        entities[5] = _pair(5)
    return ComponentCatalog.from_entities(entities)


def _capabilities(declared_tags=_DECLARED_TAGS) -> AdapterCapabilities:
    tags_by_class = (
        {"robot_contact": frozenset(declared_tags)}
        if declared_tags
        else {}
    )
    return AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"object", "robot_contact"}),
        components_by_class={
            "robot_contact": _CONTACT_COMPONENTS,
        },
        component_mode_by_class={
            "robot_contact": VocabularyMode.CLOSED,
        },
        component_tags_by_class=tags_by_class,
    )


def _compile(formula: str, **kwargs):
    kwargs.setdefault("catalog", _catalog())
    kwargs.setdefault("capabilities", _capabilities())
    return compile_specs(formula, dt=0.1, **kwargs)


def _pair(entity_id: int) -> EntityState:
    return EntityState(
        entity_id=entity_id,
        entity_class="robot_contact",
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={
            "contact_point_force_max": 1.0,
            "contact_normal_force_sum": 1.0,
        },
    )


def _obs(timestep, entities, component_tags=None) -> MonitorObservation:
    return MonitorObservation(
        timestep=timestep,
        time=float(timestep) * 0.1,
        perceived_entities=entities,
        component_tags=component_tags or {},
    )


def _tags(*labels):
    return {
        key: tuple(labels)
        for key in _CONTACT_COMPONENTS
    }


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class TestCompile:
    def test_zero_instances_compile_and_retain_filter(self):
        compiled = _compile(_FILTERED)
        assert len(compiled) == 1
        params = compiled[0].params
        assert len(params) == 1
        assert params[0].is_wildcard
        assert params[0].component_tag_filters == {
            "contact_point_force_max": ("role:distractor",),
        }

    def test_live_instance_at_reset_still_routes_runtime(self):
        # Routing is by class declaration, not by instance count: a pair
        # that happens to exist at reset must not flip the spec into
        # compile-time desugaring (which would pin reset pairs only).
        compiled = _compile(_FILTERED, catalog=_catalog(with_contact=True))
        assert len(compiled) == 1
        assert compiled[0].params[0].component_tag_filters

    def test_label_outside_declared_tags_rejected(self):
        formula = _FILTERED.replace("role:distractor", "role:banana")
        with pytest.raises(SpecConfigError) as excinfo:
            _compile(formula)
        message = str(excinfo.value)
        assert "role:banana" in message
        assert "role:distractor" in message

    def test_key_outside_declared_schema_rejected(self):
        formula = _FILTERED.replace(
            "contact_point_force_max", "contact_force_typo"
        )
        with pytest.raises(SpecConfigError) as excinfo:
            _compile(formula)
        assert "contact_force_typo" in str(excinfo.value)

    def test_no_declared_tags_rejects_filter(self):
        with pytest.raises(SpecConfigError):
            _compile(
                _FILTERED,
                capabilities=_capabilities(declared_tags=frozenset()),
            )

    def test_glob_atom_with_filter_rejected(self):
        formula = _FILTERED.replace(
            "contact_point_force_max @", "contact_* @"
        )
        with pytest.raises(SpecConfigError):
            _compile(formula)

    def test_quantifier_with_filter_rejected(self):
        formula = _FILTERED.replace(
            "[contact_point_force_max @",
            "[any: contact_point_force_max @",
        )
        with pytest.raises(SpecConfigError):
            _compile(formula)

    def test_exclude_term_with_filter_rejected(self):
        formula = _FILTERED.replace(
            " @ role:distractor]",
            " @ role:distractor \\ contact_normal_force_sum]",
        )
        with pytest.raises(SpecConfigError):
            _compile(formula)

    def test_aggregator_composition_rejected(self):
        formula = (
            "G[0,1s](max(robot_contact_*"
            "[contact_point_force_max @ role:distractor], 1) <= 50)"
        )
        with pytest.raises(SpecConfigError):
            _compile(formula)

    def test_unfiltered_declared_schema_path_unchanged(self):
        compiled = _compile(
            "G[0,1s](robot_contact_*[contact_point_force_max] <= 50)"
        )
        assert len(compiled) == 1
        assert compiled[0].params[0].component_tag_filters == {}

    def test_static_class_filter_still_desugars(self):
        # An object entity whose key carries a tag through catalog
        # metadata keeps compile-time desugaring: one pinned formula per
        # matching entity, and no retained runtime filter.
        class _Meta:
            def __init__(self, group, tags):
                self.group = group
                self.tags = tags

        entity = EntityState(
            entity_id=3,
            entity_class="object",
            position=None,
            velocity=None,
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties={"height": 2.0},
        )
        catalog = ComponentCatalog.from_entities(
            {3: entity},
            metadata={3: {"height": _Meta("size", ("role:distractor",))}},
        )
        compiled = _compile(
            "G[0,1s](object_*[height @ role:distractor] <= 2)",
            catalog=catalog,
        )
        assert len(compiled) == 1
        assert not compiled[0].params[0].is_wildcard
        assert compiled[0].params[0].component_tag_filters == {}


# ---------------------------------------------------------------------------
# Runtime eligibility
# ---------------------------------------------------------------------------


class TestRuntimeBinding:
    def _manager(self) -> IncrementalBindingManager:
        return IncrementalBindingManager(_compile(_FILTERED)[0])

    def test_no_pairs_no_bindings(self):
        manager = self._manager()
        results = manager.update(_obs(0, {}))
        assert results == {}

    def test_pair_born_mid_episode_binds_that_step(self):
        manager = self._manager()
        manager.update(_obs(0, {}))

        entities = {7: _pair(7)}
        results = manager.update(
            _obs(
                1,
                entities,
                component_tags={
                    7: _tags("role:distractor", "role:robot")
                },
            )
        )
        # Bindings are slot-aligned entity-ID tuples; one slot here.
        assert set(results) == {(7,)}

    def test_non_matching_pair_never_binds(self):
        manager = self._manager()
        entities = {
            7: _pair(7),
            8: _pair(8),
        }
        results = manager.update(
            _obs(
                0,
                entities,
                component_tags={
                    7: _tags("role:distractor", "role:robot"),
                    8: _tags("role:target", "role:robot"),
                },
            )
        )
        assert set(results) == {(7,)}

    def test_pair_without_tag_metadata_never_binds(self):
        manager = self._manager()
        results = manager.update(_obs(0, {7: _pair(7)}))
        assert results == {}

    def test_enumerate_bindings_honors_filter(self):
        formula = _compile(_FILTERED)[0]
        observation = _obs(
            0,
            {7: _pair(7), 8: _pair(8)},
            component_tags={
                7: _tags("role:distractor", "role:robot"),
                8: _tags("role:target", "role:robot"),
            },
        )
        bindings = enumerate_bindings(formula, observation)
        assert bindings == {(7,)}
