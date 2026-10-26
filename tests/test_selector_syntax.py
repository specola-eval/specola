"""
Component-selector grammar: text inside [...] -> ComponentSelector.

Pure syntax layer (stage 3, task 2): no catalog, no simulator, no entity
resolution, zero behavior change for existing (trivial) specs.

- ``specola.spec.selectors.parse_selector`` turns bracket text into an
  immutable ``ComponentSelector``.
- ``specola.spec.grammar.PARAM_RE`` widens to accept selector text and
  the ``**`` binding token, but still matches every old reference form
  identically.
- The old semantic-compile path (``parse_stl`` / ``semantic.compile_formula``)
  must refuse ``**`` bindings and non-trivial selectors loudly; Task 3 lifts
  the restriction.
"""

from __future__ import annotations

import pytest

from specola.entities import SpecConfigError
from specola.spec import grammar
from specola.spec import parser as spec_parser
from specola.spec.ast import ComponentSelector, SelectorTerm
from specola.spec.selectors import parse_selector
from specola.stl_parser import parse_stl


# ---------------------------------------------------------------------------
# parse_selector: well-formed selectors
# ---------------------------------------------------------------------------

def test_trivial_name_round_trip():
    selector = parse_selector("qpos_j")

    assert selector == ComponentSelector(
        quantifier=None,
        include=SelectorTerm(
            atom="qpos_j",
            names=("qpos_j",),
            is_glob=False,
            group=None,
        ),
        exclude=(),
        raw="qpos_j",
    )
    assert selector.is_trivial is True


def test_glob_is_not_trivial():
    selector = parse_selector("qacc_*")

    assert selector.include.atom == "qacc_*"
    assert selector.include.names == ("qacc_*",)
    assert selector.include.is_glob is True
    assert selector.is_trivial is False


def test_list_literal_whitespace_tolerated():
    selector = parse_selector("{a, b}")

    assert selector.include.names == ("a", "b")
    assert selector.include.is_glob is False
    assert selector.is_trivial is False


def test_list_literal_single_member_is_still_not_trivial():
    selector = parse_selector("{qacc_rootx}")

    assert selector.include.names == ("qacc_rootx",)
    assert selector.is_trivial is False


def test_group_filter_with_spaces():
    selector = parse_selector("qacc_* @ joint angular acceleration")

    assert selector.include.atom == "qacc_*"
    assert selector.include.group == "joint angular acceleration"
    assert selector.is_trivial is False


def test_group_and_exclude_combined():
    selector = parse_selector(
        "qacc_* @ joint angular acceleration \\ {qacc_rootx}"
    )

    assert selector.quantifier is None
    assert selector.include == SelectorTerm(
        atom="qacc_*",
        names=("qacc_*",),
        is_glob=True,
        group="joint angular acceleration",
    )
    assert selector.exclude == (
        SelectorTerm(
            atom="{qacc_rootx}",
            names=("qacc_rootx",),
            is_glob=False,
            group=None,
        ),
    )


def test_quantifier_any():
    selector = parse_selector("any: touch_*")

    assert selector.quantifier == "any"
    assert selector.include.atom == "touch_*"
    assert selector.include.is_glob is True


def test_quantifier_all():
    selector = parse_selector("all: qacc_*")

    assert selector.quantifier == "all"
    assert selector.include.atom == "qacc_*"


def test_raw_preserves_original_text():
    text = "qacc_* @ joint angular acceleration \\ {qacc_rootx}"
    selector = parse_selector(text)

    assert selector.raw == text


# ---------------------------------------------------------------------------
# parse_selector: malformed input
# ---------------------------------------------------------------------------

def test_empty_selector_rejected():
    with pytest.raises(ValueError):
        parse_selector("")


def test_list_literal_trailing_comma_rejected():
    with pytest.raises(ValueError):
        parse_selector("{a,}")


def test_quantifier_missing_colon_parses_as_atom_with_space():
    # No ':' after 'all' -> the whole text is a bare atom, and a bare atom
    # may not contain whitespace.
    with pytest.raises(ValueError):
        parse_selector("all qacc_*")


def test_nested_braces_rejected():
    with pytest.raises(ValueError):
        parse_selector("{{a}}")


def test_stray_at_with_no_atom_rejected():
    with pytest.raises(ValueError):
        parse_selector("@ joint angular acceleration")


def test_stray_at_with_no_group_rejected():
    with pytest.raises(ValueError):
        parse_selector("qacc_* @")


def test_double_at_rejected():
    with pytest.raises(ValueError):
        parse_selector("qacc_* @ a @ b")


def test_unterminated_list_literal_rejected():
    with pytest.raises(ValueError):
        parse_selector("{a, b")


# ---------------------------------------------------------------------------
# PARAM_RE: old reference forms still match identically
# ---------------------------------------------------------------------------

def test_param_re_matches_old_wildcard_form():
    match = grammar.PARAM_RE.search("robot_*[qpos_j]")

    assert match is not None
    assert match.group("head") == "robot"
    assert match.group("explicit_label") is None
    assert match.group("binding") == "*"
    assert match.group("component") == "qpos_j"


def test_param_re_matches_old_explicit_label_form():
    match = grammar.PARAM_RE.search("robot{A}_*[k]")

    assert match is not None
    assert match.group("head") == "robot"
    assert match.group("explicit_label") == "A"
    assert match.group("binding") == "*"
    assert match.group("component") == "k"


def test_param_re_matches_old_pinned_form():
    match = grammar.PARAM_RE.search("object_3[k]")

    assert match is not None
    assert match.group("head") == "object"
    assert match.group("explicit_label") is None
    assert match.group("binding") == "3"
    assert match.group("component") == "k"


def test_param_re_matches_new_double_star_binding():
    match = grammar.PARAM_RE.search("object_**[xpos_*_2]")

    assert match is not None
    assert match.group("head") == "object"
    assert match.group("binding") == "**"
    assert match.group("component") == "xpos_*_2"


def test_param_re_matches_new_selector_text_with_special_characters():
    match = grammar.PARAM_RE.search(
        "robot_*[qacc_* \\ {qacc_a}]"
    )

    assert match is not None
    assert match.group("component") == "qacc_* \\ {qacc_a}"


# ---------------------------------------------------------------------------
# EntityReference: selector attached, component stays raw text
# ---------------------------------------------------------------------------

def test_entity_reference_carries_raw_component_and_parsed_selector():
    node = spec_parser.parse("robot_*[qacc_* \\ {qacc_a}] >= 0")
    reference = node.references[0]

    assert reference.component == "qacc_* \\ {qacc_a}"
    assert reference.selector.include.atom == "qacc_*"
    assert reference.selector.exclude[0].names == ("qacc_a",)


def test_entity_reference_double_star_binding():
    node = spec_parser.parse("object_**[xpos_*_2] >= 0")
    reference = node.references[0]

    assert reference.binding == "**"
    assert reference.selector.include.atom == "xpos_*_2"


# ---------------------------------------------------------------------------
# Grammar splitters treat [...] as opaque (no accidental split on \, space,
# comma, @ hidden inside a selector).
# ---------------------------------------------------------------------------

def test_bracket_contents_do_not_leak_into_top_level_split():
    from specola.spec import ast

    node = spec_parser.parse(
        "robot_*[qacc_* \\ {qacc_a}] <= 5 and robot_*[x] >= 0"
    )

    assert isinstance(node, ast.And)
    assert isinstance(node.left, ast.Comparison)
    assert isinstance(node.right, ast.Comparison)
    assert node.left.operator == "<="
    assert node.right.operator == ">="
    assert node.left.left == "robot_*[qacc_* \\ {qacc_a}]"
    assert node.right.left == "robot_*[x]"


# ---------------------------------------------------------------------------
# Task 3: selectors compile against a catalog. Without one, parse_stl refuses
# them loudly (a catalog is required to resolve any selector or "**").
# ---------------------------------------------------------------------------

def test_double_star_binding_rejected_without_catalog():
    with pytest.raises(SpecConfigError, match="catalog"):
        parse_stl("object_**[xpos_*_2] >= 0", dt=1.0)


def test_glob_selector_rejected_without_catalog():
    with pytest.raises(SpecConfigError, match="catalog"):
        parse_stl("robot_*[qacc_*] >= 0", dt=1.0)


def test_selector_with_exclude_rejected_without_catalog():
    with pytest.raises(SpecConfigError, match="catalog"):
        parse_stl(
            "robot_*[qacc_* \\ {qacc_a}] <= 5", dt=1.0
        )


def test_trivial_selector_still_compiles_unchanged():
    formula = parse_stl("sig_1[a] >= 0", dt=1.0)

    assert formula.params[0].required_components == ["a"]
    assert formula.params[0].pinned_entity_id == 1
