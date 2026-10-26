"""
parser.py — syntax-only recursive-descent parser for STL spec templates.

``parse(text)`` turns formula text into an immutable AST. It knows the
grammar and operator precedence and nothing else: no dt, no constants,
no class vocabulary, no component catalog, no temporal backend. Every
context-dependent question is answered later by semantic compilation.

Precedence, loosest to tightest:

    <->   →   ->   →   or / xor   →   and   →   since   →   not
    →   prefix past-time operators   →   atomic comparison
"""

from __future__ import annotations

from . import grammar
from .ast import (
    And,
    Comparison,
    EntityReference,
    Eventually,
    Always,
    Fall,
    FormulaNode,
    Iff,
    Implies,
    Not,
    Or,
    Previous,
    Rise,
    Since,
    Xor,
)
from .selectors import parse_selector


def parse(text: str) -> FormulaNode:
    """Parse one past-time STL spec template into an AST."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("formula text must be a non-empty string")

    stripped = text.strip()
    grammar.validate_delimiters(stripped)
    return _parse(stripped)


def _parse(text: str) -> FormulaNode:
    text = grammar.strip_outer_parentheses(text)

    if not text:
        raise ValueError("Empty STL subformula")

    # Equivalence has the lowest precedence.
    equivalences = grammar.top_level_symbol_operators(text, ("<->",))
    if equivalences:
        index, operator = equivalences[-1]
        return Iff(
            left=_parse(text[:index]),
            right=_parse(text[index + len(operator):]),
        )

    # Implication is right-associative: split at the leftmost occurrence.
    implications = grammar.top_level_symbol_operators(text, ("->",))
    if implications:
        index, operator = implications[0]
        return Implies(
            left=_parse(text[:index]),
            right=_parse(text[index + len(operator):]),
        )

    selected = grammar.rightmost(
        grammar.top_level_word_operators(text, ("or", "xor")),
        grammar.top_level_symbol_operators(text, ("||", "∨")),
    )
    if selected is not None:
        index, operator = selected
        left = _parse(text[:index])
        right = _parse(text[index + len(operator):])
        if operator.lower() in {"or", "||", "∨"}:
            return Or(left=left, right=right)
        return Xor(left=left, right=right)

    selected = grammar.rightmost(
        grammar.top_level_word_operators(text, ("and",)),
        grammar.top_level_symbol_operators(text, ("&&", "∧")),
    )
    if selected is not None:
        index, operator = selected
        return And(
            left=_parse(text[:index]),
            right=_parse(text[index + len(operator):]),
        )

    since_parts = grammar.split_top_level_since(text)
    if since_parts is not None:
        left_text, bounds_text, right_text = since_parts
        return Since(
            left=_parse(left_text),
            bounds=(
                grammar.parse_interval(bounds_text)
                if bounds_text is not None
                else None
            ),
            right=_parse(right_text),
        )

    if text.lower().startswith("not "):
        return Not(child=_parse(text[4:].strip()))

    if text.startswith(("!", "¬")):
        return Not(child=_parse(text[1:].strip()))

    temporal = grammar.match_prefix_temporal(text)
    if temporal is not None:
        operator, bounds_text, body = temporal
        bounds = (
            grammar.parse_interval(bounds_text)
            if bounds_text is not None
            else None
        )
        child = _parse(body)

        if operator.lower() in {"g", "h", "historically", "always"}:
            return Always(bounds=bounds, child=child)
        return Eventually(bounds=bounds, child=child)

    event = grammar.match_prefix_event(text)
    if event is not None:
        operator, body = event
        child = _parse(body)
        return {
            "prev": Previous,
            "rise": Rise,
            "fall": Fall,
        }[operator.lower()](child=child)

    return _parse_comparison(text)


def _parse_comparison(text: str) -> Comparison:
    comparisons = grammar.top_level_symbol_operators(
        text, grammar.COMPARISON_OPERATORS
    )

    if not comparisons:
        raise ValueError(
            f"Expected an STL formula or atomic comparison, got {text!r}"
        )

    if len(comparisons) != 1:
        raise ValueError(
            f"Chained comparisons are unsupported: {text!r}. Write each "
            f"comparison separately and combine them with 'and' or 'or'."
        )

    index, operator = comparisons[0]

    if operator == "!=":
        raise ValueError(
            "The '!=' operator has no exact continuous quantitative "
            "robustness semantics. Use abs(lhs - rhs) >= epsilon instead."
        )

    left = text[:index].strip()
    right = text[index + len(operator):].strip()

    if not left or not right:
        raise ValueError(
            f"Comparison {operator!r} requires two operands: {text!r}"
        )

    return Comparison(
        left=left,
        operator=operator,
        right=right,
        references=_extract_references(left) + _extract_references(right),
    )


def _extract_references(expression: str):
    return tuple(
        EntityReference(
            head=match.group("head"),
            explicit_label=match.group("explicit_label"),
            binding=match.group("binding"),
            component=match.group("component"),
            selector=parse_selector(match.group("component")),
        )
        for match in grammar.PARAM_RE.finditer(expression)
    )
