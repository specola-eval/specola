"""
selection.py — compile-time temporal backend selection.

Modes: ``native`` (the complete causal past-time language), ``rtamt``
(the same temporal operators, except iff/xor over a temporal subformula,
and failing at compile when RTAMT is unavailable), and ``auto`` (native).
Resolution happens when a specification is compiled, never during a
monitor update.
"""

from __future__ import annotations

from ..entities import SpecConfigError
from ..spec import ast
from . import rtamt as rtamt_backend


BACKEND_MODES = ("auto", "native", "rtamt")


def supports_native(node: ast.FormulaNode) -> bool:
    """Whether the formula lies in the native backend's fragment."""
    if isinstance(node, ast.Comparison):
        return True

    if isinstance(node, ast.Not):
        return supports_native(node.child)

    if isinstance(node, (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor)):
        return supports_native(node.left) and supports_native(node.right)

    if isinstance(node, (ast.Always, ast.Eventually)):
        return supports_native(node.child)

    if isinstance(node, ast.Since):
        return supports_native(node.left) and supports_native(node.right)

    if isinstance(node, (ast.Previous, ast.Rise, ast.Fall)):
        return supports_native(node.child)

    return False


def _has_temporal(node: ast.FormulaNode) -> bool:
    if isinstance(
        node,
        (
            ast.Always,
            ast.Eventually,
            ast.Since,
            ast.Previous,
            ast.Rise,
            ast.Fall,
        ),
    ):
        return True
    if isinstance(node, ast.Not):
        return _has_temporal(node.child)
    if isinstance(node, (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor)):
        return _has_temporal(node.left) or _has_temporal(node.right)
    return False


def _iff_xor_over_temporal(node: ast.FormulaNode) -> bool:
    """Iff/Xor whose derived and/or/not encoding would duplicate a
    temporal subformula. Both RTAMT routes fail here, so such formulas
    are only evaluable natively:

    - RTAMT does provide primitive '<->'/'xor' operators, but under a
      value-distance robustness (-|left - right| / |left - right|) that
      is not sign-consistent with the De Morgan min/max semantics this
      pipeline pins (both operands clearly satisfied can report a
      violation), so the primitives cannot be used at all.
    - The De Morgan text encoding this pipeline emits instead
      (rtamt_text.equivalence/exclusive_or) duplicates each operand;
      RTAMT evaluates repeated identical stateful subformulas
      incorrectly (empirically: values diverge from textbook
      robustness, while single-use encodings like '->' agree).

    Both failure routes are reproduced by
    benchmarks/rtamt_iff_semantics.py. For memoryless operands the
    De Morgan encoding is exact (backend-parity tests)."""
    if isinstance(node, (ast.Iff, ast.Xor)):
        if _has_temporal(node.left) or _has_temporal(node.right):
            return True
    if isinstance(node, ast.Not):
        return _iff_xor_over_temporal(node.child)
    if isinstance(node, (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor)):
        return _iff_xor_over_temporal(node.left) or _iff_xor_over_temporal(
            node.right
        )
    if isinstance(node, (ast.Always, ast.Eventually)):
        return _iff_xor_over_temporal(node.child)
    if isinstance(node, (ast.Previous, ast.Rise, ast.Fall)):
        return _iff_xor_over_temporal(node.child)
    if isinstance(node, ast.Since):
        return _iff_xor_over_temporal(node.left) or _iff_xor_over_temporal(
            node.right
        )
    return False


def choose_backend(requested: str, node: ast.FormulaNode) -> str:
    """Resolve the backend for one formula; raise on impossible requests."""
    if requested not in BACKEND_MODES:
        raise ValueError(
            f"Unknown backend {requested!r}; expected one of "
            f"{BACKEND_MODES}"
        )

    in_fragment = supports_native(node)

    if requested == "native":
        if not in_fragment:
            raise SpecConfigError(
                "This formula is outside the native backend's supported "
                "causal past-time fragment; use backend 'rtamt' or 'auto'."
            )
        return "native"

    if requested == "rtamt":
        chosen = "rtamt"
    else:
        chosen = "native" if in_fragment else "rtamt"

    if chosen == "rtamt" and _iff_xor_over_temporal(node):
        raise SpecConfigError(
            "This formula applies <->/xor to a temporal subformula, "
            "which cannot be encoded correctly for RTAMT (the derived "
            "Boolean form duplicates a stateful subformula, and RTAMT "
            "evaluates such duplicates incorrectly). Use the native "
            "backend, or restructure the formula."
        )

    if chosen == "rtamt" and not rtamt_backend.rtamt_available():
        raise SpecConfigError(
            "This specification requires the RTAMT temporal backend, "
            "but RTAMT is not installed or its C++ backend is not "
            "functional. Install it with: pip install 'specola[rtamt]'"
        )

    return chosen
