"""Selector templates on declared-schema classes follow runtime arrivals.

Before this extension, any non-trivial selector (``all:`` keyword, bare
glob, exclude list) desugared per entity against the reset catalog, so a
spec like ``vehicle_*[all: velocity_*]`` monitored only the entities that
happened to exist at reset — observed live on the CARLA adapter, where
churned-in vehicles never gained monitors while single-key specs bound
them immediately.

The extension under test: when a class declares a CLOSED component
schema (``AdapterCapabilities.components_by_class``) and the slot's
references carry no ``@`` filter, selector keys resolve once against the
DECLARATION and the rewritten references stay unpinned wildcard slots.
Which entity satisfies the slot is then per-observation binding
eligibility, so members born mid-episode bind the step they appear —
identical to how trivial single-key references already behaved.

Scope guards: ``@``-filtered selectors keep the stricter runtime-filter
route (test_runtime_tag_eligibility.py); undeclared classes keep
compile-time desugaring; population-dependent aggregation (``**``,
min/max/sum calls) stays a loud compile error on declared classes.
"""

from __future__ import annotations

import pytest

from specola.adapter import MonitorObservation
from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.components import ComponentCatalog
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.monitor import IncrementalBindingManager
from specola.stl_parser import compile_specs

_CONTACT_COMPONENTS = frozenset(
    {
        "contact_point_force_max",
        "contact_normal_force_sum",
    }
)


def _catalog() -> ComponentCatalog:
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
    return ComponentCatalog.from_entities(entities)


def _capabilities(
    runtime: bool = True,
    groups: bool = False,
) -> AdapterCapabilities:
    return AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"object", "robot_contact"}),
        components_by_class={"robot_contact": _CONTACT_COMPONENTS},
        component_mode_by_class={"robot_contact": VocabularyMode.CLOSED},
        runtime_populated_classes=(
            frozenset({"robot_contact"}) if runtime else frozenset()
        ),
        component_groups_by_class=(
            {
                "robot_contact": {
                    key: "contact force" for key in _CONTACT_COMPONENTS
                }
            }
            if groups
            else {}
        ),
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


def _obs(timestep, entities) -> MonitorObservation:
    return MonitorObservation(
        timestep=timestep,
        time=float(timestep) * 0.1,
        perceived_entities=entities,
    )


class TestCompilation:
    def test_all_keyword_compiles_with_zero_instances(self):
        # The old path raised "No entity of class 'robot_contact'".
        compiled = _compile("G[0,1s](robot_contact_*[all: contact_*] <= 50)")
        assert len(compiled) == 1
        parameter = compiled[0].params[0]
        assert parameter.is_wildcard
        assert set(parameter.required_components) == _CONTACT_COMPONENTS

    def test_bare_glob_fans_per_key_but_stays_unpinned(self):
        compiled = _compile("G[0,1s](robot_contact_*[contact_*] <= 50)")
        assert len(compiled) == 2
        assert all(spec.params[0].is_wildcard for spec in compiled)

    def test_exclude_resolves_against_declaration(self):
        compiled = _compile(
            "G[0,1s](robot_contact_*"
            "[all: contact_* \\ contact_normal_force_sum] <= 50)"
        )
        assert len(compiled) == 1
        assert set(compiled[0].params[0].required_components) == {
            "contact_point_force_max"
        }

    def test_glob_outside_declaration_rejected(self):
        with pytest.raises(SpecConfigError):
            _compile("G[0,1s](robot_contact_*[all: nope_*] <= 50)")

    def test_population_aggregation_rejected(self):
        with pytest.raises(SpecConfigError) as excinfo:
            _compile("G[0,1s](max(robot_contact_*[contact_*], 1) <= 50)")
        assert "declared component schema" in str(excinfo.value)

    def test_star_star_rejected_even_with_reset_population(self):
        # '**' never enters slot routing, so this guard must fire from
        # the reference walk itself. With members present at reset the
        # old behavior silently compiled pinned to them — the exact bug
        # class the declared route exists to eliminate.
        catalog = ComponentCatalog.from_entities(
            {5: _pair(5), 6: _pair(6)}
        )
        for formula in (
            "G[0,1s](robot_contact_**[all: contact_*] <= 50)",
            "G[0,1s](min(robot_contact_**[contact_point_force_max]) <= 50)",
        ):
            with pytest.raises(SpecConfigError) as excinfo:
                _compile(formula, catalog=catalog)
            assert "declared component schema" in str(excinfo.value)

    def test_exclude_with_filter_rejected_by_name(self):
        # No groups declared: the '@' label in the exclude term has no
        # static meaning, and the error names the (empty) vocabulary.
        with pytest.raises(SpecConfigError) as excinfo:
            _compile(
                "G[0,1s](robot_contact_*"
                "[all: contact_* \\ contact_normal_force_sum @ force]"
                " <= 50)"
            )
        assert "Declared groups" in str(excinfo.value)

    def test_undeclared_class_keeps_compile_time_desugaring(self):
        compiled = _compile("G[0,1s](object_*[all: h*] <= 5)")
        assert len(compiled) == 1
        assert not compiled[0].params[0].is_wildcard


class TestRuntimeArrivals:
    def test_entity_born_mid_episode_binds_that_step(self):
        compiled = _compile(
            "G[0,1s](robot_contact_*[all: contact_*] <= 50)"
        )
        manager = IncrementalBindingManager(compiled[0])
        assert manager.update(_obs(0, {})) == {}
        results = manager.update(_obs(1, {7: _pair(7)}))
        assert set(results) == {(7,)}

    def test_second_arrival_gains_its_own_binding(self):
        compiled = _compile(
            "G[0,1s](robot_contact_*[all: contact_*] <= 50)"
        )
        manager = IncrementalBindingManager(compiled[0])
        manager.update(_obs(0, {7: _pair(7)}))
        results = manager.update(_obs(1, {7: _pair(7), 9: _pair(9)}))
        assert set(results) == {(7,), (9,)}


class TestExternalReviewRegressions:
    """Pins for the three demonstrated defects the external (Codex)
    review found in the first version of this route."""

    def test_legacy_labels_match_explicit_labels(self):
        # Legacy `class_A_*` and explicit `class{A}_*` are one label
        # namespace; the first rewrite dropped legacy labels, silently
        # collapsing two slots into one parameter.
        catalog = ComponentCatalog.from_entities(
            {1: _pair(1), 2: _pair(2)}
        )
        legacy = _compile(
            "G[0,1s]((robot_contact_A_*[all: contact_*] <= 50)"
            " and (robot_contact_B_*[all: contact_*] <= 60))",
            catalog=catalog,
        )
        explicit = _compile(
            "G[0,1s]((robot_contact{A}_*[all: contact_*] <= 50)"
            " and (robot_contact{B}_*[all: contact_*] <= 60))",
            catalog=catalog,
        )
        assert len(legacy) == len(explicit) == 1
        assert (
            [p.param_key for p in legacy[0].params]
            == [p.param_key for p in explicit[0].params]
        )
        assert len(legacy[0].params) == 2

        manager = IncrementalBindingManager(legacy[0])
        results = manager.update(_obs(0, {1: _pair(1), 2: _pair(2)}))
        assert set(results) == {(1, 2), (2, 1)}

    def test_legacy_labelled_and_trivial_reference_share_one_slot(self):
        compiled = _compile(
            "G[0,1s]((robot_contact_A_*[all: contact_*] <= 50)"
            " and (robot_contact_A_*[contact_point_force_max] <= 60))"
        )
        assert len(compiled[0].params) == 1

    def test_declared_group_filter_compiles_and_follows_arrivals(self):
        # Published metadata groups must stay usable in @ filters on a
        # declared runtime class; the first version rejected them as
        # undeclared runtime tags.
        capabilities = _capabilities(groups=True)
        for formula in (
            "G[0,1s](robot_contact_*"
            "[contact_point_force_max @ contact force] <= 50)",
            "G[0,1s](robot_contact_*[all: contact_* @ contact force] <= 50)",
            "G[0,1s](robot_contact_*"
            "[all: contact_* \\ contact_normal_force_sum @ contact force]"
            " <= 50)",
        ):
            compiled = _compile(formula, capabilities=capabilities)
            assert len(compiled) == 1
            assert compiled[0].params[0].is_wildcard

        compiled = _compile(
            "G[0,1s](robot_contact_*[all: contact_* @ contact force] <= 50)",
            capabilities=capabilities,
        )
        manager = IncrementalBindingManager(compiled[0])
        assert manager.update(_obs(0, {})) == {}
        assert set(manager.update(_obs(1, {7: _pair(7)}))) == {(7,)}

    def test_unknown_group_label_rejected_with_vocabulary(self):
        with pytest.raises(SpecConfigError) as excinfo:
            _compile(
                "G[0,1s](robot_contact_*[all: contact_* @ nope] <= 50)",
                capabilities=_capabilities(groups=True),
            )
        assert "contact force" in str(excinfo.value)

    def test_static_declared_class_keeps_catalog_semantics(self):
        # CLOSED component vocabulary and runtime population are
        # independent: a static declared class keeps reset-catalog
        # desugaring, including '**' aggregation over its population.
        catalog = ComponentCatalog.from_entities(
            {5: _pair(5), 6: _pair(6)}
        )
        static_caps = _capabilities(runtime=False)
        compiled = _compile(
            "G[0,1s](max(robot_contact_**[contact_point_force_max]) <= 50)",
            catalog=catalog,
            capabilities=static_caps,
        )
        assert len(compiled) == 1
        pinned = _compile(
            "G[0,1s](robot_contact_*[all: contact_*] <= 50)",
            catalog=catalog,
            capabilities=static_caps,
        )
        # Static class: per-entity desugaring, one pinned spec per member.
        assert all(not spec.params[0].is_wildcard for spec in pinned)
