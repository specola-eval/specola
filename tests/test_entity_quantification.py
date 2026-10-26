"""Safe handling of scene-level ``**`` reductions.

``**`` expands one selector across every eligible entity in a class.  The
current textual expansion can preserve a flat reduction, but it cannot
preserve per-entity groups under a different enclosing reduction.  These
tests pin the safe boundary until formula-level quantification is available.
"""

from __future__ import annotations

from typing import Dict

import pytest

from specola.components import ComponentCatalog, ComponentResolver
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.stl_parser import compile_specs, parse_stl


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
        1: _entity(1, "object", {"px": 0.5, "py": 0.5}),
        2: _entity(2, "object", {"px": 5.0, "py": 0.5}),
        4: _entity(
            4,
            "region",
            {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0},
        ),
    }


@pytest.fixture
def catalog():
    return ComponentCatalog.from_entities(_entities())


@pytest.fixture
def resolver():
    return ComponentResolver(_entities())


def _binding(formula, resolver):
    binding: Dict = {}
    for parameter in formula.params:
        if parameter.pinned_entity_id is not None:
            binding[parameter.param_key] = parameter.pinned_entity_id
            continue
        matches = [
            entity_id
            for entity_id, entity in resolver.entities.items()
            if entity.entity_class == parameter.entity_class
            and all(
                component in entity.properties
                for component in parameter.required_components
            )
        ]
        assert len(matches) == 1
        binding[parameter.param_key] = matches[0]
    return binding


def _rho(formula, resolver):
    return formula.predicate_fn(_binding(formula, resolver), resolver)


def _rho_longhand(text, resolver, catalog):
    formula = parse_stl(text, catalog=catalog, backend="native")
    return _rho(formula, resolver)


_BOX_MARGINS = (
    "object_**[px] - region_4[x0], "
    "region_4[x1] - object_**[px], "
    "object_**[py] - region_4[y0], "
    "region_4[y1] - object_**[py]"
)


def test_mixed_max_min_over_two_entities_is_rejected(catalog):
    with pytest.raises(SpecConfigError) as exc_info:
        compile_specs(
            f"Max(Min({_BOX_MARGINS})) >= 0",
            catalog=catalog,
            backend="native",
        )

    message = str(exc_info.value)
    assert "2 object entities" in message
    assert "min" in message.lower()
    assert "max" in message.lower()
    assert (
        "flatten per-entity groups into one reduction and change its meaning"
        in message
    )
    assert "grouped scene-level form is not expressible in this release" in message
    assert "Write one explicit formula per entity" in message
    assert "any object_*:" not in message
    assert "all object_*:" not in message


def test_reverse_mixed_min_max_over_two_entities_is_rejected(catalog):
    with pytest.raises(SpecConfigError, match="flatten per-entity groups"):
        compile_specs(
            "Min(Max(object_**[p*])) >= 1",
            catalog=catalog,
            backend="native",
        )


def test_deeper_equal_reducer_does_not_hide_mixed_outer_reducer(catalog):
    with pytest.raises(SpecConfigError, match="flatten per-entity groups"):
        compile_specs(
            "Max(Min(Min(object_**[p*]))) >= 0",
            catalog=catalog,
            backend="native",
        )


def test_mixed_sum_and_max_over_two_entities_is_rejected(catalog):
    with pytest.raises(SpecConfigError, match="flatten per-entity groups"):
        compile_specs(
            "Max(sum(object_**[px])) >= 0",
            catalog=catalog,
            backend="native",
        )


@pytest.mark.parametrize(
    ("formula_text", "longhand"),
    [
        (
            "Max(Max(object_**[p*])) >= 1",
            "Max("
            "Max(object_1[px], object_1[py]), "
            "Max(object_2[px], object_2[py])"
            ") >= 1",
        ),
        (
            "Min(Min(object_**[p*])) >= 1",
            "Min("
            "Min(object_1[px], object_1[py]), "
            "Min(object_2[px], object_2[py])"
            ") >= 1",
        ),
    ],
)
def test_equal_reductions_match_grouped_longhand_numerically(
    formula_text, longhand, catalog, resolver
):
    formula = compile_specs(
        formula_text, catalog=catalog, backend="native"
    )[0]

    assert _rho(formula, resolver) == pytest.approx(
        _rho_longhand(longhand, resolver, catalog)
    )


def test_one_entity_mixed_reduction_matches_longhand():
    entities = {
        entity_id: entity
        for entity_id, entity in _entities().items()
        if entity_id != 2
    }
    catalog = ComponentCatalog.from_entities(entities)
    resolver = ComponentResolver(entities)
    formula = compile_specs(
        f"Max(Min({_BOX_MARGINS})) >= 0",
        catalog=catalog,
        backend="native",
    )[0]
    longhand = (
        "Max(Min("
        "object_1[px] - region_4[x0], "
        "region_4[x1] - object_1[px], "
        "object_1[py] - region_4[y0], "
        "region_4[y1] - object_1[py]"
        ")) >= 0"
    )

    assert _rho(formula, resolver) == pytest.approx(
        _rho_longhand(longhand, resolver, catalog)
    )


def test_plain_star_component_glob_keeps_per_entity_monitors(catalog, resolver):
    formulas = compile_specs(
        "Max(Min(object_*[p*])) >= 0",
        catalog=catalog,
        backend="native",
    )

    assert len(formulas) == 2
    got = sorted(_rho(formula, resolver) for formula in formulas)
    want = sorted(
        _rho_longhand(
            f"Max(Min(object_{entity_id}[px], object_{entity_id}[py])) >= 0",
            resolver,
            catalog,
        )
        for entity_id in (1, 2)
    )
    assert got == pytest.approx(want)


def test_double_star_any_keyword_retains_flat_meaning(catalog, resolver):
    formula = compile_specs(
        "object_**[any: p*] >= 4",
        catalog=catalog,
        backend="native",
    )[0]
    longhand = (
        "Max("
        "object_1[px], object_1[py], "
        "object_2[px], object_2[py]"
        ") >= 4"
    )

    assert _rho(formula, resolver) == pytest.approx(
        _rho_longhand(longhand, resolver, catalog)
    )


def test_direct_unenclosed_reduction_retains_universal_meaning(
    catalog, resolver
):
    formula = compile_specs(
        "Min("
        "object_**[px] - region_4[x0], "
        "region_4[x1] - object_**[px]"
        ") >= 0",
        catalog=catalog,
        backend="native",
    )[0]
    longhand = (
        "Min("
        "object_1[px] - region_4[x0], "
        "object_2[px] - region_4[x0], "
        "region_4[x1] - object_1[px], "
        "region_4[x1] - object_2[px]"
        ") >= 0"
    )

    assert _rho(formula, resolver) == pytest.approx(
        _rho_longhand(longhand, resolver, catalog)
    )
    assert _rho(formula, resolver) == pytest.approx(-4.0)


def test_bare_double_star_keeps_existing_error(catalog):
    with pytest.raises(SpecConfigError, match="only valid inside an aggregator"):
        compile_specs(
            "object_**[px] >= 0",
            catalog=catalog,
            backend="native",
        )


def test_two_distinct_selectors_in_one_argument_keeps_existing_error(catalog):
    with pytest.raises(
        SpecConfigError,
        match="contains more than one distinct selector reference",
    ):
        compile_specs(
            "Max(object_**[px] - object_**[py]) >= 0",
            catalog=catalog,
            backend="native",
        )
