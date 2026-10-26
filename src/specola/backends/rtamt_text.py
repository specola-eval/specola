"""
rtamt_text.py — RTAMT formula-text fragments.

The only place that knows RTAMT's concrete syntax. Semantic compilation
walks the specification AST and assembles the monitored formula from
these fragments.

Two RTAMT operators are deliberately never emitted:

- ``<->`` computes -|l - r| (numeric closeness of robustness values, not
  logical equivalence: it is negative even when both operands are false);
- ``xor`` computes |l - r| (positive even when both operands are true).

Equivalence and exclusive-or are therefore compiled through their
and/or/not derived forms.
"""

from __future__ import annotations


def interval(lower: int, upper: int) -> str:
    return f"[{lower},{upper}]"


def negation(child: str) -> str:
    return f"(not ({child}))"


def conjunction(left: str, right: str) -> str:
    return f"(({left}) and ({right}))"


def disjunction(left: str, right: str) -> str:
    return f"(({left}) or ({right}))"


def implication(left: str, right: str) -> str:
    return disjunction(negation(left), right)


def equivalence(left: str, right: str) -> str:
    return conjunction(
        disjunction(negation(left), right),
        disjunction(negation(right), left),
    )


def exclusive_or(left: str, right: str) -> str:
    return disjunction(
        conjunction(left, negation(right)),
        conjunction(negation(left), right),
    )


def historically(interval_text: str, child: str) -> str:
    return f"historically{interval_text} ({child})"


def historically_unbounded(child: str) -> str:
    return f"historically ({child})"


def once(interval_text: str, child: str) -> str:
    return f"once{interval_text} ({child})"


def once_unbounded(child: str) -> str:
    return f"once ({child})"


def since(left: str, interval_text: str, right: str) -> str:
    return f"(({left}) since{interval_text} ({right}))"


def since_unbounded(left: str, right: str) -> str:
    return f"(({left}) since ({right}))"


def previous(child: str) -> str:
    """Strong previous via an exact one-sample past window.

    RTAMT 0.3.5's built-in ``prev`` returns positive infinity at the
    trace origin. ``once[1,1]`` returns negative infinity there and the
    exact previous child robustness thereafter, matching Specola's
    registered strong-previous contract.
    """
    return once(interval(1, 1), child)


def rise(child: str) -> str:
    return f"rise ({child})"


def fall(child: str) -> str:
    return f"fall ({child})"
