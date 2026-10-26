"""
test_spec_render.py — AST -> text renderer round-trip, and desugared-text
attachment on compiled formulas (stage-4 Task 1).

Round-trip bar: ``formula_text(parse(text))``, when re-parsed, yields an
AST that renders to the *same string* again (a fixed point), for every
node kind the grammar supports. Exact-string parity with the originally
authored text is a bonus where the grammar permits it, not the bar.
"""

from __future__ import annotations

import pytest

from specola.adapter import PropertyMeta
from specola.components import ComponentCatalog
from specola.entities import EntityState, TrackState
from specola.spec import parser as _spec_parser
from specola.spec import render
from specola.stl_parser import compile_specs, parse_stl


# ---------------------------------------------------------------------------
# Round-trip idempotence: render(parse(text)) -> parse -> render is fixed.
# ---------------------------------------------------------------------------

ROUND_TRIP_FORMULAS = [
    "sig_1[a] >= 0",
    "sig_1[a] >= 0 and sig_1[b] <= 1",
    "sig_1[a] >= 0 or sig_1[b] <= 1",
    "not sig_1[a] >= 0",
    "sig_1[a] >= 0 and sig_1[b] <= 1 and not sig_1[c] == 2",
    "sig_1[a] >= 0 or not sig_1[b] <= 1",
    "G[0,5s](sig_1[a] >= 0)",
    "F[0,100](sig_1[a] >= 0)",
    "historically(sig_1[a] >= 0)",
    "once(sig_1[a] >= 0)",
    "prev(sig_1[a] >= 0)",
    "rise(sig_1[a] >= 0)",
    "fall(sig_1[a] >= 0)",
    "sig_1[a] >= 0 since sig_1[b] <= 1",
    "sig_1[a] >= 0 since[1,3] sig_1[b] <= 1",
    "sig_1[a] >= 0 -> sig_1[b] <= 1",
    "sig_1[a] >= 0 <-> sig_1[b] <= 1",
    "sig_1[a] >= 0 xor sig_1[b] <= 1",
    "G[0,5s](sig_1[a] >= 0 and sig_1[b] <= 1)",
    "(sig_1[a] >= 0 -> sig_1[b] <= 1) and sig_1[c] == 2",
]


@pytest.mark.parametrize("text", ROUND_TRIP_FORMULAS)
def test_round_trip_is_a_fixed_point(text):
    first_ast = _spec_parser.parse(text)
    first_text = render.formula_text(first_ast)

    second_ast = _spec_parser.parse(first_text)
    second_text = render.formula_text(second_ast)

    assert second_text == first_text
    assert second_ast == first_ast


# ---------------------------------------------------------------------------
# Bounds render exactly as authored: steps stay bare, seconds keep 's'.
# ---------------------------------------------------------------------------

def test_seconds_bound_keeps_s_suffix():
    node = _spec_parser.parse("G[0,5s](sig_1[a] >= 0)")
    assert (
        render.formula_text(node)
        == "historically[0,5s](sig_1[a] >= 0)"
    )


def test_step_bound_stays_bare():
    node = _spec_parser.parse("F[0,100](sig_1[a] >= 0)")
    assert render.formula_text(node) == "once[0,100](sig_1[a] >= 0)"


def test_since_bounds_and_operands_render_verbatim():
    node = _spec_parser.parse("sig_1[a] >= 0 since[1,3] sig_1[b] <= 1")
    assert (
        render.formula_text(node)
        == "sig_1[a] >= 0 since[1,3] sig_1[b] <= 1"
    )


def test_implies_renders_canonical_arrow():
    node = _spec_parser.parse("sig_1[a] >= 0 -> sig_1[b] <= 1")
    assert render.formula_text(node) == "sig_1[a] >= 0 -> sig_1[b] <= 1"


# ---------------------------------------------------------------------------
# desugared_text on compiled formulas
# ---------------------------------------------------------------------------

def _catalog():
    entity = EntityState(
        entity_id=1,
        entity_class="robot",
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={"qacc_a": 30.0, "qacc_b": 45.0},
    )
    joint = PropertyMeta(group="joint angular acceleration")
    metadata = {1: {"qacc_a": joint, "qacc_b": joint}}
    return ComponentCatalog.from_entities({1: entity}, metadata=metadata)


def test_desugared_text_contains_expanded_keys_and_reparses_cleanly():
    formulas = compile_specs(
        "G[0,1s](max(robot_*[qacc_*]) <= 50)", catalog=_catalog()
    )

    assert len(formulas) == 1
    text = formulas[0].desugared_text
    assert text is not None
    assert "qacc_a" in text
    assert "qacc_b" in text

    reparsed = _spec_parser.parse(text)
    assert render.formula_text(reparsed) == text


@pytest.mark.parametrize("operator", ["prev", "rise", "fall"])
def test_causal_unary_operators_preserve_selector_expansion(operator):
    formulas = compile_specs(
        f"{operator}(max(robot_*[qacc_*]) <= 50)", catalog=_catalog()
    )

    assert len(formulas) == 1
    text = formulas[0].desugared_text
    assert text is not None
    assert "qacc_a" in text and "qacc_b" in text
    assert render.formula_text(_spec_parser.parse(text)) == text


def test_trivial_formula_has_no_desugared_text():
    formula = parse_stl("sig_1[a] >= 0", dt=1.0)
    assert formula.desugared_text is None
