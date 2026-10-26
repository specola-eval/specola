"""
Syntax/semantics separation: the spec package contract.

- ``specola.spec.parser.parse`` is syntax-only: no dt, no catalog, no
  class vocabulary, no RTAMT. It returns an immutable AST.
- ``specola.spec.semantic`` merges entity references into parameter
  slots ordered by first source occurrence and enforces same-class
  injectivity for pinned slots.
- ``parse_stl`` remains the compatibility entry point; its behavior is
  pinned by tests/test_parser_semantics.py.
"""

from __future__ import annotations

import pytest

from specola.entities import SpecConfigError
from specola.spec import ast
from specola.spec.parser import parse
from specola.stl_parser import parse_stl


# ---------------------------------------------------------------------------
# Syntax-only parsing to an immutable AST
# ---------------------------------------------------------------------------

def test_parse_takes_only_formula_text():
    import inspect

    parameters = inspect.signature(parse).parameters
    assert list(parameters) == ["text"]


def test_nested_formula_structure():
    node = parse("G[0,4](F[0,3](sig_1[a] >= 0))")

    assert isinstance(node, ast.Always)
    assert isinstance(node.child, ast.Eventually)
    assert isinstance(node.child.child, ast.Comparison)
    assert node.child.child.operator == ">="


def test_authored_bounds_are_not_dt_converted():
    node = parse("G[0,5s](sig_1[a] >= 0)")

    assert node.bounds.upper.value == 5.0
    assert node.bounds.upper.in_seconds is True
    assert node.bounds.lower.value == 0.0


def test_boolean_structure_and_precedence():
    node = parse("sig_1[a] >= 0 or sig_1[a] >= 1 and sig_1[b] >= 0")

    assert isinstance(node, ast.Or)
    assert isinstance(node.left, ast.Comparison)
    assert isinstance(node.right, ast.And)


def test_since_structure():
    node = parse("(sig_1[a] >= 0) since[1,4] (sig_1[b] >= 0)")

    assert isinstance(node, ast.Since)
    assert node.bounds.lower.value == 1.0
    assert node.bounds.upper.value == 4.0


def test_entity_references_extracted_per_comparison():
    node = parse("sig_1[a] >= obj_*[m]")

    references = node.references
    assert [(r.head, r.binding, r.component) for r in references] == [
        ("sig", "1", "a"),
        ("obj", "*", "m"),
    ]


def test_ast_nodes_are_immutable():
    node = parse("sig_1[a] >= 0")

    with pytest.raises(Exception):
        node.operator = "<="


def test_syntax_errors_raised_without_any_context():
    with pytest.raises(ValueError):
        parse("G[0,5](sig_1[a] >= 0")
    with pytest.raises(ValueError):
        parse("0 <= sig_1[a] <= 1")


# ---------------------------------------------------------------------------
# Semantic compilation: slots and injectivity
# ---------------------------------------------------------------------------

def test_slots_ordered_by_first_source_occurrence():
    parsed = parse_stl(
        "obj_*[m] >= 0 and sig_1[a] >= 0 and obj_*[m] <= 5", dt=1.0
    )

    assert [p.param_key[0] for p in parsed.params] == ["obj", "sig"]


def test_slot_order_follows_source_not_robustness_rewrite():
    # '<=' rewrites robustness to (right - left); slot order must still
    # follow the authored source: sig before obj.
    parsed = parse_stl("sig_1[a] <= obj_*[m]", dt=1.0)

    assert [p.param_key[0] for p in parsed.params] == ["sig", "obj"]


def test_duplicate_pinned_same_class_rejected():
    with pytest.raises(SpecConfigError, match="pinned"):
        parse_stl("sig{A}_1[a] >= sig{B}_1[b]", dt=1.0)


def test_repeated_pinned_reference_is_one_slot():
    parsed = parse_stl("sig_1[a] >= 0 and sig_1[b] >= 0", dt=1.0)

    assert len(parsed.params) == 1
    assert parsed.params[0].required_components == ["a", "b"]
