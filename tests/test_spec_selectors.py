"""
test_spec_selectors.py — stage-3 desugaring pass equivalence matrix.

Every sugared spec must compile to predicates numerically identical to its
hand-written longhand (compiled through today's ``parse_stl``). Robustness is
evaluated on synthetic ``EntityState``s through a ``ComponentResolver`` — the
same objects a live monitor step feeds a compiled predicate.

Fixtures (from the Task-3 brief):

    robot entity 1 {qacc_a: 30, qacc_b: 45, touch_f1: 0.0, touch_f2: 0.9}
    object entity 3 {xpos_cup_2: 1.2}
    object entity 5 {xpos_bowl_2: 0.4}
    groups: qacc_* -> "joint angular acceleration",
            touch_* -> "sensors: touch",
            xpos_*  -> "body position"
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from specola.adapter import PropertyMeta
from specola.components import (
    ComponentCatalog,
    ComponentResolver,
)
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.spec.selectors import resolve_selector
from specola.spec import parser as _spec_parser
from specola.spec import semantic as _semantic
from specola.stl_parser import compile_specs, parse_stl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entity(entity_id, entity_class, properties):
    return EntityState(
        entity_id=entity_id,
        entity_class=entity_class,
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def _entities():
    return {
        1: _entity(
            1,
            "robot",
            {"qacc_a": 30.0, "qacc_b": 45.0, "touch_f1": 0.0, "touch_f2": 0.9},
        ),
        3: _entity(3, "object", {"xpos_cup_2": 1.2}),
        5: _entity(5, "object", {"xpos_bowl_2": 0.4}),
    }


def _metadata():
    joint = PropertyMeta(group="joint angular acceleration")
    touch = PropertyMeta(group="sensors: touch")
    body = PropertyMeta(group="body position")
    return {
        1: {
            "qacc_a": joint,
            "qacc_b": joint,
            "touch_f1": touch,
            "touch_f2": touch,
        },
        3: {"xpos_cup_2": body},
        5: {"xpos_bowl_2": body},
    }


@pytest.fixture
def catalog():
    return ComponentCatalog.from_entities(
        _entities(),
        metadata=_metadata(),
    )


@pytest.fixture
def resolver():
    return ComponentResolver(_entities())


# ---------------------------------------------------------------------------
# Robustness harness
# ---------------------------------------------------------------------------

def _binding(formula, resolver):
    """Bind each parameter slot to a concrete entity id.

    Pinned slots use their pinned id; wildcard slots pick the (single)
    fixture entity of the right class that exposes the required components.
    """
    binding: Dict = {}
    for parameter in formula.params:
        if parameter.pinned_entity_id is not None:
            binding[parameter.param_key] = parameter.pinned_entity_id
            continue
        chosen = None
        for entity_id, entity in resolver.entities.items():
            if entity.entity_class != parameter.entity_class:
                continue
            if all(
                component in entity.properties
                for component in parameter.required_components
            ):
                chosen = entity_id
                break
        assert chosen is not None, (
            f"no fixture entity for {parameter.param_key}"
        )
        binding[parameter.param_key] = chosen
    return binding


def _rho(formula, resolver):
    return formula.predicate_fn(_binding(formula, resolver), resolver)


def _rho_longhand(text, resolver, catalog):
    formula = parse_stl(text, catalog=catalog)
    return _rho(formula, resolver)


# ---------------------------------------------------------------------------
# resolve_selector unit coverage
# ---------------------------------------------------------------------------

def _selector(text):
    return _spec_parser.parse(f"robot_*[{text}] <= 0").references[0].selector


def test_resolve_glob(catalog):
    components = catalog.components_for_entity(1)
    groups = catalog.component_groups_for_entity(1)
    assert resolve_selector(_selector("qacc_*"), components, groups) == (
        "qacc_a",
        "qacc_b",
    )


def test_resolve_difference(catalog):
    components = catalog.components_for_entity(1)
    groups = catalog.component_groups_for_entity(1)
    assert resolve_selector(
        _selector("qacc_* \\ {qacc_b}"), components, groups
    ) == ("qacc_a",)


def test_resolve_group_with_spaces_and_colon(catalog):
    components = catalog.components_for_entity(1)
    groups = catalog.component_groups_for_entity(1)
    assert resolve_selector(
        _selector("* @ sensors: touch"), components, groups
    ) == ("touch_f1", "touch_f2")


def test_resolve_empty_lists_real_keys(catalog):
    components = catalog.components_for_entity(1)
    groups = catalog.component_groups_for_entity(1)
    with pytest.raises(SpecConfigError) as exc:
        resolve_selector(
            _selector("qacc_* @ sensors: touch"), components, groups
        )
    message = str(exc.value)
    assert "qacc_a" in message and "touch_f1" in message


# ---------------------------------------------------------------------------
# Case 1: aggregator over a wildcard slot (one formula)
# ---------------------------------------------------------------------------

def test_case1_max_aggregator(catalog, resolver):
    formulas = compile_specs("max(robot_*[qacc_*]) <= 50", catalog=catalog)
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(robot_*[qacc_a], robot_*[qacc_b]) <= 50", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Case 2: bare selector fans out to one formula per key, pinned
# ---------------------------------------------------------------------------

def test_case2_bare_fanout(catalog, resolver):
    formulas = compile_specs("robot_*[qacc_*] <= 50", catalog=catalog)
    assert len(formulas) == 2
    by_suffix = {f.name.split(" ")[-1]: f for f in formulas}
    assert set(by_suffix) == {"[qacc_a@1]", "[qacc_b@1]"}

    # Reviewer Minor 3: fanned-out formulas are PINNED, never left wildcard.
    for formula in formulas:
        assert formula.params, "expected at least one parameter"
        for parameter in formula.params:
            assert parameter.is_wildcard is False
            assert parameter.pinned_entity_id == 1

    got_a = _rho(by_suffix["[qacc_a@1]"], resolver)
    got_b = _rho(by_suffix["[qacc_b@1]"], resolver)
    assert got_a == pytest.approx(
        _rho_longhand("robot_*[qacc_a] <= 50", resolver, catalog)
    )
    assert got_b == pytest.approx(
        _rho_longhand("robot_*[qacc_b] <= 50", resolver, catalog)
    )
    assert got_a == pytest.approx(20.0)
    assert got_b == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Case 3: sum(...) elementwise textual expansion
# ---------------------------------------------------------------------------

def test_case3_sum_elementwise(catalog, resolver):
    formulas = compile_specs(
        "sum(abs(robot_*[qacc_*])) <= 100", catalog=catalog
    )
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "(abs(robot_*[qacc_a]) + abs(robot_*[qacc_b])) <= 100",
        resolver,
        catalog,
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Case 4: any/all keyword direction table
# ---------------------------------------------------------------------------

def test_case4_any_ge_is_max(catalog, resolver):
    # Canonical spelling: the quantifier keyword lives INSIDE the brackets.
    formulas = compile_specs(
        "robot_*[any: touch_*] >= 0.5", catalog=catalog
    )
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(robot_*[touch_f1], robot_*[touch_f2]) >= 0.5", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(0.4)


def test_case4_all_ge_is_min(catalog, resolver):
    formulas = compile_specs(
        "robot_*[all: touch_*] >= 0.5", catalog=catalog
    )
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "min(robot_*[touch_f1], robot_*[touch_f2]) >= 0.5", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(-0.5)


def test_case4_all_le_is_max(catalog, resolver):
    formulas = compile_specs("robot_*[all: qacc_*] <= 50", catalog=catalog)
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(robot_*[qacc_a], robot_*[qacc_b]) <= 50", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(5.0)


def test_case4_outside_keyword_prefix_rejected(catalog):
    # Decision 1: the outside-prefix spelling is rejected with a pointer at
    # the canonical in-bracket form.
    with pytest.raises(SpecConfigError) as exc:
        compile_specs("any: robot_*[touch_*] >= 0.5", catalog=catalog)
    message = str(exc.value)
    assert "inside" in message
    assert "robot_*[any: touch_*]" in message


def test_case4_keyword_with_arithmetic_rejected(catalog):
    # Fix 1: a quantified selector must be the entire comparison side, or the
    # surrounding arithmetic would be silently discarded.
    with pytest.raises(SpecConfigError) as exc:
        compile_specs("robot_*[any: qacc_*] + 3 <= 50", catalog=catalog)
    assert "entire comparison side" in str(exc.value)


def test_case4_keyword_alone_compiles(catalog, resolver):
    # The un-decorated quantified form still compiles to min(...)/max(...).
    formulas = compile_specs("robot_*[any: qacc_*] <= 50", catalog=catalog)
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "min(robot_*[qacc_a], robot_*[qacc_b]) <= 50", resolver, catalog
    )
    assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# Case 5: ** entity expansion with heterogeneous keys inside one formula
# ---------------------------------------------------------------------------

def test_case5_double_star_heterogeneous(catalog, resolver):
    formulas = compile_specs(
        "max(object_**[xpos_*_2]) >= 1.1", catalog=catalog
    )
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(object_3[xpos_cup_2], object_5[xpos_bowl_2]) >= 1.1",
        resolver,
        catalog,
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Case 6: difference, group filter, and empty-match error
# ---------------------------------------------------------------------------

def test_case6_difference(catalog, resolver):
    formulas = compile_specs(
        "robot_*[qacc_* \\ {qacc_b}] <= 50", catalog=catalog
    )
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand("robot_*[qacc_a] <= 50", resolver, catalog)
    assert got == pytest.approx(want)
    assert got == pytest.approx(20.0)


def test_case6_group_filter(catalog, resolver):
    formulas = compile_specs(
        "robot_*[any: * @ sensors: touch] >= 0.5", catalog=catalog
    )
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(robot_*[touch_f1], robot_*[touch_f2]) >= 0.5", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(0.4)


def test_case6_disjoint_glob_and_group_raises(catalog):
    with pytest.raises(SpecConfigError) as exc:
        compile_specs("robot_*[qacc_* @ sensors: touch] <= 50", catalog=catalog)
    assert "qacc_a" in str(exc.value)


# ---------------------------------------------------------------------------
# Case 7: error conditions
# ---------------------------------------------------------------------------

def test_case7_bare_double_star_rejected(catalog):
    with pytest.raises(SpecConfigError):
        compile_specs("object_**[xpos_cup_2] >= 1.0", catalog=catalog)


def test_case7_keyword_equality_rejected(catalog):
    with pytest.raises(SpecConfigError):
        compile_specs("robot_*[any: touch_*] == 0", catalog=catalog)


def test_case7_two_selectors_one_aggregator_arg_rejected(catalog):
    with pytest.raises(SpecConfigError):
        compile_specs(
            "sum(robot_*[qacc_*] + robot_*[touch_*]) <= 100", catalog=catalog
        )


def test_case7_non_trivial_selector_without_catalog_rejected():
    with pytest.raises(SpecConfigError):
        compile_specs("max(robot_*[qacc_*]) <= 50")


def test_case7_parse_stl_on_fanout_raises(catalog):
    with pytest.raises(SpecConfigError, match="compile_specs"):
        parse_stl("robot_*[qacc_*] <= 50", catalog=catalog)


# ---------------------------------------------------------------------------
# Case 8: trivial regression — parse_stl path is untouched
# ---------------------------------------------------------------------------

def test_case8_trivial_regression(catalog, resolver):
    formula = parse_stl("G[0,1s](robot_*[qacc_a] <= 50)", catalog=catalog)
    assert len(formula.params) == 1
    parameter = formula.params[0]
    assert parameter.entity_class == "robot"
    assert parameter.is_wildcard
    assert parameter.required_components == ["qacc_a"]
    # trivial selector still yields exactly one formula from compile_specs
    formulas = compile_specs("G[0,1s](robot_*[qacc_a] <= 50)", catalog=catalog)
    assert len(formulas) == 1


# ---------------------------------------------------------------------------
# Two-robot fixtures: slot-level pinning must keep one authored slot on ONE
# entity per output formula (Fix 2). With two robots of the same class, a
# wildcard slot that fans over entities must pin EVERY reference sharing that
# slot — trivial siblings included — so injectivity can never split them.
# ---------------------------------------------------------------------------

def _two_robot_entities():
    return {
        1: _entity(1, "robot", {"qacc_a": 30.0, "qacc_b": 45.0}),
        2: _entity(2, "robot", {"qacc_a": 10.0, "qacc_b": 70.0}),
    }


def _two_robot_metadata():
    joint = PropertyMeta(group="joint angular acceleration")
    keys = {"qacc_a": joint, "qacc_b": joint}
    return {1: dict(keys), 2: dict(keys)}


@pytest.fixture
def two_robot_catalog():
    return ComponentCatalog.from_entities(
        _two_robot_entities(),
        metadata=_two_robot_metadata(),
    )


@pytest.fixture
def two_robot_resolver():
    return ComponentResolver(_two_robot_entities())


def test_two_robot_slot_pinning_keeps_entity_together(
    two_robot_catalog, two_robot_resolver
):
    # robot_*[qacc_a] is a trivial sibling of max(robot_*[qacc_*]); both share
    # the (robot, "", "*") slot and must land on the SAME robot per formula.
    formulas = compile_specs(
        "robot_*[qacc_a] + max(robot_*[qacc_*]) <= 100",
        catalog=two_robot_catalog,
    )
    assert len(formulas) == 2

    by_suffix = {f.name.split(" ")[-1]: f for f in formulas}
    assert set(by_suffix) == {"[@1]", "[@2]"}

    for entity_id, formula in ((1, by_suffix["[@1]"]), (2, by_suffix["[@2]"])):
        # Every parameter is pinned to the one robot this formula belongs to.
        assert formula.params
        for parameter in formula.params:
            assert parameter.is_wildcard is False
            assert parameter.pinned_entity_id == entity_id

        got = _rho(formula, two_robot_resolver)
        entity = two_robot_resolver.entities[entity_id]
        a = entity.properties["qacc_a"]
        b = entity.properties["qacc_b"]
        # Longhand: qacc_a + max(qacc_a, qacc_b) <= 100.
        want = 100.0 - (a + max(a, b))
        assert got == pytest.approx(want)


def test_two_robot_max_aggregator_pins_each(
    two_robot_catalog, two_robot_resolver
):
    # Case 1 with two robots: two pinned formulas, each equal to that robot's
    # hand-computed longhand.
    formulas = compile_specs(
        "max(robot_*[qacc_*]) <= 50", catalog=two_robot_catalog
    )
    assert len(formulas) == 2

    by_suffix = {f.name.split(" ")[-1]: f for f in formulas}
    assert set(by_suffix) == {"[@1]", "[@2]"}

    for entity_id, formula in ((1, by_suffix["[@1]"]), (2, by_suffix["[@2]"])):
        for parameter in formula.params:
            assert parameter.is_wildcard is False
            assert parameter.pinned_entity_id == entity_id

        got = _rho(formula, two_robot_resolver)
        entity = two_robot_resolver.entities[entity_id]
        want = 50.0 - max(
            entity.properties["qacc_a"], entity.properties["qacc_b"]
        )
        assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# Final-review regression shapes (stage-3, spec-selectors branch).
#
# Each asserts numeric equality with hand-written longhand (via parse_stl)
# AND the slot/pinning invariant the restructure guarantees: entity fan-out
# is enumerated once per formula, injectively across same-class slots, and
# key expansion happens per reference within each entity assignment.
# ---------------------------------------------------------------------------

def _rho_multiset(formula, resolver):
    """Sorted robustness of EVERY predicate in a (possibly multi-atom) formula.

    ``formula.predicate_fn`` is only the first atom; a boolean-combined
    formula needs every ``predicate_fns`` entry to compare against longhand.
    """
    binding = _binding(formula, resolver)
    return sorted(
        fn(binding, resolver) for fn in formula.predicate_fns.values()
    )


# C1: one authored '*' slot reused across two comparisons must pin to ONE
# entity per output formula — never split across two by injectivity.

def test_c1_slot_reused_across_two_comparisons(
    two_robot_catalog, two_robot_resolver
):
    formulas = compile_specs(
        "(robot_*[all: qacc_*] <= 50) and (robot_*[qacc_a] >= 0)",
        catalog=two_robot_catalog,
    )
    assert len(formulas) == 2

    seen_entities = set()
    for formula in formulas:
        pinned = {p.pinned_entity_id for p in formula.params}
        # The whole formula lives on exactly ONE robot: no wildcard leftover,
        # no second entity dragged in by injectivity.
        assert None not in pinned
        assert len(pinned) == 1
        entity_id = pinned.pop()
        seen_entities.add(entity_id)

        longhand = (
            f"(max(robot_{entity_id}[qacc_a], robot_{entity_id}[qacc_b]) "
            f"<= 50) and (robot_{entity_id}[qacc_a] >= 0)"
        )
        want = _rho_multiset(
            parse_stl(longhand, catalog=two_robot_catalog),
            two_robot_resolver,
        )
        assert _rho_multiset(formula, two_robot_resolver) == pytest.approx(
            want
        )

    assert seen_entities == {1, 2}


# C2: expansion binds the INNERMOST aggregator; sibling args stay untouched.

def test_c2_nested_aggregator_binds_innermost(catalog, resolver):
    formulas = compile_specs(
        "max(0, min(robot_*[qacc_*])) <= 40", catalog=catalog
    )
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(0, min(robot_*[qacc_a], robot_*[qacc_b])) <= 40",
        resolver,
        catalog,
    )
    assert got == pytest.approx(want)
    # max(0, min(30, 45)) = 30 -> 40 - 30 = 10 (the bug gave -5).
    assert got == pytest.approx(10.0)


# I5: 'Min('/'Max(' are aggregators too (same operator as min/max).

def test_i5_capitalized_min_is_aggregator(catalog, resolver):
    formulas = compile_specs("Min(robot_*[qacc_*]) <= 50", catalog=catalog)
    assert len(formulas) == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "min(robot_*[qacc_a], robot_*[qacc_b]) <= 50", resolver, catalog
    )
    assert got == pytest.approx(want)
    assert got == pytest.approx(20.0)


# C3: labelled same-class slots fan injectively -> P(2,2)=2, no self-pairs.

def test_c3_labelled_two_slot_fanout_no_self_pairs(
    two_robot_catalog, two_robot_resolver
):
    formulas = compile_specs(
        "max(robot{A}_*[qacc_*]) - max(robot{B}_*[qacc_*]) >= 1",
        catalog=two_robot_catalog,
    )
    assert len(formulas) == 2  # P(2,2), not 2*2=4

    rhos = []
    for formula in formulas:
        pinned = {p.pinned_entity_id for p in formula.params}
        # Two DISTINCT robots per formula: never a self-comparison.
        assert None not in pinned
        assert len(pinned) == 2
        rhos.append(_rho(formula, two_robot_resolver))

    # Longhands for the two injective assignments (A,B) in {(1,2),(2,1)}.
    want = sorted(
        _rho_longhand(
            f"max(robot_{a}[qacc_a], robot_{a}[qacc_b]) - "
            f"max(robot_{b}[qacc_a], robot_{b}[qacc_b]) >= 1",
            two_robot_resolver,
            two_robot_catalog,
        )
        for a, b in ((1, 2), (2, 1))
    )
    assert sorted(rhos) == pytest.approx(want)
    # A genuine self-pair would evaluate to max(x)-max(x)-1 = -1.
    assert all(r != pytest.approx(-1.0) for r in rhos)


# I4: heterogeneous class — eligible members monitored, non-matching skipped;
# all-non-matching raises naming real keys.

def _hetero_entities():
    return {
        3: _entity(3, "object", {"xpos_cup_2": 1.2}),
        5: _entity(5, "object", {"mass": 0.4}),  # exposes no xpos_*_2 key
    }


def _hetero_catalog():
    body = PropertyMeta(group="body position")
    metadata = {3: {"xpos_cup_2": body}, 5: {"mass": PropertyMeta(group="")}}
    return ComponentCatalog.from_entities(_hetero_entities(), metadata=metadata)


def test_i4_heterogeneous_skips_non_matching_member():
    catalog = _hetero_catalog()
    resolver = ComponentResolver(_hetero_entities())
    formulas = compile_specs("object_*[xpos_*_2] <= 2.0", catalog=catalog)
    # Only object 3 exposes an xpos_*_2 key; object 5 is skipped, not fatal.
    assert len(formulas) == 1
    assert formulas[0].name.endswith("[xpos_cup_2@3]")
    for parameter in formulas[0].params:
        assert parameter.pinned_entity_id == 3
    got = _rho(formulas[0], resolver)
    want = _rho_longhand("object_3[xpos_cup_2] <= 2.0", resolver, catalog)
    assert got == pytest.approx(want)
    assert got == pytest.approx(0.8)


def test_i4_all_non_matching_raises_naming_keys():
    catalog = _hetero_catalog()
    with pytest.raises(SpecConfigError) as exc:
        compile_specs("object_*[qvel_*] <= 2.0", catalog=catalog)
    message = str(exc.value)
    # Names real keys of the class's entities.
    assert "xpos_cup_2" in message and "mass" in message


# I6: a repeated identical selector token rewrites identically (no cartesian).

def test_i6_repeated_identical_token_two_monitors(catalog, resolver):
    formulas = compile_specs(
        "robot_*[qacc_*] + robot_*[qacc_*] <= 100", catalog=catalog
    )
    assert len(formulas) == 2  # per key, NOT 2*2 mixed-key monitors
    by_suffix = {f.name.split(" ")[-1]: f for f in formulas}
    assert set(by_suffix) == {"[qacc_a@1]", "[qacc_b@1]"}

    got_a = _rho(by_suffix["[qacc_a@1]"], resolver)
    got_b = _rho(by_suffix["[qacc_b@1]"], resolver)
    want_a = _rho_longhand(
        "robot_*[qacc_a] + robot_*[qacc_a] <= 100", resolver, catalog
    )
    want_b = _rho_longhand(
        "robot_*[qacc_b] + robot_*[qacc_b] <= 100", resolver, catalog
    )
    assert got_a == pytest.approx(want_a)
    assert got_b == pytest.approx(want_b)
    assert got_a == pytest.approx(40.0)   # 100 - (30 + 30)
    assert got_b == pytest.approx(10.0)   # 100 - (45 + 45)


# I7: a trivial sibling inside an aggregator argument is pinned same-entity,
# not rejected as an overlap.

def test_i7_aggregator_arg_with_trivial_sibling(catalog, resolver):
    formulas = compile_specs(
        "max(robot_*[qacc_*] - robot_*[qacc_a]) <= 0", catalog=catalog
    )
    assert len(formulas) == 1
    for parameter in formulas[0].params:
        assert parameter.pinned_entity_id == 1
    got = _rho(formulas[0], resolver)
    want = _rho_longhand(
        "max(robot_*[qacc_a] - robot_*[qacc_a], "
        "robot_*[qacc_b] - robot_*[qacc_a]) <= 0",
        resolver,
        catalog,
    )
    assert got == pytest.approx(want)
    # -max(30-30, 45-30) = -15.
    assert got == pytest.approx(-15.0)


# Zero-monitor expansion: injectivity over-constraint must raise loudly,
# never compile to a silent no-op ("never vacuous, never silent").

def test_zero_monitor_expansion_raises(catalog):
    # Two labelled same-class injective slots, one eligible robot:
    # P(1, 2) = 0 assignments -> loud error, not zero monitors.
    with pytest.raises(SpecConfigError) as exc_info:
        compile_specs(
            "max(robot{A}_*[qacc_*]) - max(robot{B}_*[qacc_*]) >= 1",
            catalog=catalog,
        )
    message = str(exc_info.value)
    assert "expands to zero monitors" in message
    assert "'robot'" in message
    assert "1 eligible entity" in message
    assert "2 injective slots" in message
