"""
render.py — AST → text renderer for past-time STL spec templates.

``formula_text(node)`` is the syntactic mirror of ``spec.parser.parse``:
given an AST produced by the parser (or by semantic desugaring / selector
expansion), it re-emits parenthesized, round-trippable formula text.

- ``Comparison`` re-emits its stored left/operator/right text verbatim
  (comparisons carry authored expression text, not a further AST).
- ``Not``/``And``/``Or``/``Implies``/``Iff``/``Xor``/``Since`` spell the
  parser's canonical operator tokens (``not``, ``and``, ``or``, ``->``,
  ``<->``, ``xor``, ``since``).
- bounded ``Always``/``Eventually`` re-render their ``Interval`` exactly as
  authored: a step bound stays a bare integer, a seconds bound keeps its
  ``s`` suffix (``Bound.in_seconds``) — see ``grammar.parse_bound``.

Parenthesization: ``And``/``Or``/``Xor``/``Implies``/``Iff``/``Since`` are
the only node types whose rendering is not self-delimiting — each spells
a top-level keyword or symbol that ``parser._parse`` would rediscover if
it leaked, unparenthesized, into a surrounding formula. Every such node is
therefore wrapped in parentheses whenever it appears as a child of any
other node (including another node of the same kind); ``Comparison``,
``Not`` and prefix temporal nodes are already self-delimiting
(atomic text, a prefix keyword, and a mandatory ``OP[bounds](...)`` shell
respectively) and are never given extra parentheses. This wrapping is
conservative rather than precedence-minimal, but it is what makes
``formula_text`` a fixed point under repeated parse/render round trips:
render is a pure function of AST shape, and an unambiguous rendering
always re-parses to a structurally identical AST.

Nothing here knows about dt, catalogs, or simulators.
"""

from __future__ import annotations

from . import ast


# Node types whose own rendering is not self-delimiting: each spells a
# top-level operator token that a surrounding ``_parse`` scan could
# rediscover if the rendered text were embedded unparenthesized.
_COMPOSITE = (ast.And, ast.Or, ast.Xor, ast.Implies, ast.Iff, ast.Since)


def formula_text(node: ast.FormulaNode) -> str:
    """Render an AST node back into parseable, round-trippable text."""
    return _render(node)


def _render(node: ast.FormulaNode) -> str:
    if isinstance(node, ast.Comparison):
        return f"{node.left} {node.operator} {node.right}"

    if isinstance(node, ast.Not):
        return f"not {_operand(node.child)}"

    if isinstance(node, ast.And):
        return f"{_operand(node.left)} and {_operand(node.right)}"

    if isinstance(node, ast.Or):
        return f"{_operand(node.left)} or {_operand(node.right)}"

    if isinstance(node, ast.Xor):
        return f"{_operand(node.left)} xor {_operand(node.right)}"

    if isinstance(node, ast.Implies):
        return f"{_operand(node.left)} -> {_operand(node.right)}"

    if isinstance(node, ast.Iff):
        return f"{_operand(node.left)} <-> {_operand(node.right)}"

    if isinstance(node, ast.Since):
        interval = (
            f"[{_interval(node.bounds)}]"
            if node.bounds is not None
            else ""
        )
        return f"{_operand(node.left)} since{interval} {_operand(node.right)}"

    if isinstance(node, ast.Always):
        interval = (
            f"[{_interval(node.bounds)}]"
            if node.bounds is not None
            else ""
        )
        return f"historically{interval}({_render(node.child)})"

    if isinstance(node, ast.Eventually):
        interval = (
            f"[{_interval(node.bounds)}]"
            if node.bounds is not None
            else ""
        )
        return f"once{interval}({_render(node.child)})"

    if isinstance(node, ast.Previous):
        return f"prev({_render(node.child)})"

    if isinstance(node, ast.Rise):
        return f"rise({_render(node.child)})"

    if isinstance(node, ast.Fall):
        return f"fall({_render(node.child)})"

    raise TypeError(f"Unknown formula node type: {type(node)!r}")


def _operand(node: ast.FormulaNode) -> str:
    """Render a child node, parenthesizing it iff it is not self-delimiting."""
    text = _render(node)
    if isinstance(node, _COMPOSITE):
        return f"({text})"
    return text


def _interval(interval: ast.Interval) -> str:
    return f"{_bound(interval.lower)},{_bound(interval.upper)}"


def _bound(bound: ast.Bound) -> str:
    number = _number(bound.value)
    return f"{number}s" if bound.in_seconds else number


def _number(value: float) -> str:
    """Render a bound's authored value with no gratuitous decimal point."""
    if value == int(value):
        return str(int(value))
    return repr(value)
