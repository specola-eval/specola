"""
grammar.py — lexical layer of the STL specification language.

Token patterns and text-scanning helpers used by the syntax parser.
Everything here is context-free: no dt, no catalogs, no simulators.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from .ast import Bound, Interval


# Capture the complete class or legacy-label head first; the head is
# resolved against a class vocabulary during semantic compilation.
#
# binding: "**" (fan-out group) must be tried before "*" (wildcard), since
# regex alternation is ordered and "*" alone would otherwise swallow the
# first "*" of a "**" token.
#
# component: selector syntax (stage 3) may hold arbitrary text — spaces,
# "\", "{", "}", ",", ":", "@" — so the charset only excludes "]", which
# terminates the bracket and can never itself appear inside a selector.
# The component text is captured raw here; specola.spec.selectors
# parses it into a ComponentSelector.
PARAM_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<head>[A-Za-z][A-Za-z0-9_]*)"
    r"(?:\{(?P<explicit_label>[A-Z])\})?"
    r"_(?P<binding>\*\*|\*|\d+)"
    r"\[(?P<component>[^\]]+)\]"
)

COMPARISON_OPERATORS = (
    ">=",
    "<=",
    "==",
    "!=",
    ">",
    "<",
)

CONSTANT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Delimiters and parenthesis handling
# ---------------------------------------------------------------------------

def validate_delimiters(text: str) -> None:
    parentheses = 0
    brackets = 0

    for index, character in enumerate(text):
        if character == "(":
            parentheses += 1
        elif character == ")":
            parentheses -= 1
            if parentheses < 0:
                raise ValueError(f"Unmatched ')' at position {index}")
        elif character == "[":
            brackets += 1
        elif character == "]":
            brackets -= 1
            if brackets < 0:
                raise ValueError(f"Unmatched ']' at position {index}")

    if parentheses:
        raise ValueError("Unmatched '(' in formula")
    if brackets:
        raise ValueError("Unmatched '[' in formula")


def outer_parentheses_match(text: str) -> bool:
    depth = 0

    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
            if depth == 0 and index != len(text) - 1:
                return False

    return depth == 0


def strip_outer_parentheses(text: str) -> str:
    text = text.strip()

    while (
        text.startswith("(")
        and text.endswith(")")
        and outer_parentheses_match(text)
    ):
        text = text[1:-1].strip()

    return text


def find_matching_parenthesis(text: str, opening_index: int) -> int:
    depth = 0

    for index in range(opening_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index

    raise ValueError(f"Unmatched '(' in formula {text!r}")


# ---------------------------------------------------------------------------
# Top-level operator scanning
# ---------------------------------------------------------------------------

def _is_word_boundary(text: str, start: int, end: int) -> bool:
    left_valid = start == 0 or not (
        text[start - 1].isalnum() or text[start - 1] == "_"
    )
    right_valid = end == len(text) or not (
        text[end].isalnum() or text[end] == "_"
    )
    return left_valid and right_valid


def top_level_word_operators(
    text: str,
    words: Sequence[str],
) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    parentheses = 0
    brackets = 0

    for index, character in enumerate(text):
        if character == "(":
            parentheses += 1
        elif character == ")":
            parentheses -= 1
        elif character == "[":
            brackets += 1
        elif character == "]":
            brackets -= 1

        if parentheses == 0 and brackets == 0:
            for word in words:
                end = index + len(word)
                if (
                    text[index:end].lower() == word.lower()
                    and _is_word_boundary(text, index, end)
                ):
                    found.append((index, text[index:end]))

    return found


def top_level_symbol_operators(
    text: str,
    operators: Sequence[str],
) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    parentheses = 0
    brackets = 0
    index = 0

    ordered = sorted(operators, key=len, reverse=True)

    while index < len(text):
        character = text[index]

        if character == "(":
            parentheses += 1
            index += 1
            continue
        if character == ")":
            parentheses -= 1
            index += 1
            continue
        if character == "[":
            brackets += 1
            index += 1
            continue
        if character == "]":
            brackets -= 1
            index += 1
            continue

        if parentheses == 0 and brackets == 0:
            matched = next(
                (
                    operator
                    for operator in ordered
                    if text.startswith(operator, index)
                ),
                None,
            )
            if matched is not None:
                found.append((index, matched))
                index += len(matched)
                continue

        index += 1

    return found


def rightmost(
    *groups: Sequence[Tuple[int, str]],
) -> Optional[Tuple[int, str]]:
    values = [item for group in groups for item in group]
    if not values:
        return None
    return max(values, key=lambda item: item[0])


def split_top_level_since(
    text: str,
) -> Optional[Tuple[str, Optional[str], str]]:
    """Return one top-level since split, rejecting ungrouped chains."""
    parentheses = 0
    brackets = 0
    matches = []

    for index, character in enumerate(text):
        if character == "(":
            parentheses += 1
        elif character == ")":
            parentheses -= 1
        elif character == "[":
            brackets += 1
        elif character == "]":
            brackets -= 1

        at_whitespace_start = (
            character.isspace()
            and (index == 0 or not text[index - 1].isspace())
        )
        if parentheses == 0 and brackets == 0 and at_whitespace_start:
            match = re.match(
                r"\s+(?:S|since)(?:\s*\[([^\]]+)\])?\s+",
                text[index:],
                re.IGNORECASE,
            )
            if match is not None:
                matches.append((index, match))

    if len(matches) > 1:
        raise ValueError(
            "Since is non-associative; parenthesize each since formula: "
            f"{text!r}"
        )

    if matches:
        index, match = matches[0]
        left = text[:index].strip()
        right = text[index + match.end():].strip()
        if not left or not right:
            raise ValueError(f"Since requires two operands: {text!r}")
        return left, match.group(1), right

    return None


def match_prefix_temporal(
    text: str,
) -> Optional[Tuple[str, Optional[str], str]]:
    """Return (operator, optional bounds, body) for a past prefix op."""
    match = re.match(
        r"^(historically|once)(?:\s*\[([^\]]+)\])?\s*\(",
        text,
        re.IGNORECASE,
    )
    if match is None:
        match = re.match(
            r"^(G|F|H|P|always|eventually)\s*\[([^\]]+)\]\s*\(",
            text,
            re.IGNORECASE,
        )
    if match is None:
        return None

    opening_index = match.end() - 1
    closing_index = find_matching_parenthesis(text, opening_index)

    if closing_index != len(text) - 1:
        return None

    body = text[opening_index + 1 : closing_index].strip()
    if not body:
        raise ValueError(
            f"Temporal operator {match.group(1)!r} has an empty body"
        )

    return match.group(1), match.group(2), body


def match_prefix_event(text: str) -> Optional[Tuple[str, str]]:
    """Return (operator, body) for prev/rise/fall over the whole string."""
    match = re.match(r"^(prev|rise|fall)\s*\(", text, re.IGNORECASE)
    if match is None:
        return None

    opening_index = match.end() - 1
    closing_index = find_matching_parenthesis(text, opening_index)
    if closing_index != len(text) - 1:
        return None

    body = text[opening_index + 1 : closing_index].strip()
    if not body:
        raise ValueError(
            f"Temporal operator {match.group(1)!r} has an empty body"
        )
    return match.group(1), body


# ---------------------------------------------------------------------------
# Authored temporal bounds (syntactic validation only; no dt)
# ---------------------------------------------------------------------------

def parse_bound(text: str) -> Bound:
    text = text.strip()
    in_seconds = text.lower().endswith("s")
    number_text = text[:-1] if in_seconds else text

    try:
        value = float(number_text)
    except ValueError as exc:
        raise ValueError(f"Invalid temporal bound {text!r}") from exc

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Temporal bound must be finite and nonnegative: {text!r}"
        )

    if not in_seconds:
        rounded = math.floor(value + 0.5)
        if not math.isclose(value, rounded, abs_tol=1.0e-9):
            raise ValueError(
                f"Step bound must be an integer; append 's' for "
                f"seconds: {text!r}"
            )
        value = float(rounded)

    return Bound(value=value, in_seconds=in_seconds)


def parse_interval(bounds_text: str) -> Interval:
    parts = [part.strip() for part in bounds_text.split(",")]

    if len(parts) != 2:
        raise ValueError(
            f"Expected interval [lo,hi], got [{bounds_text}]"
        )

    return Interval(
        lower=parse_bound(parts[0]),
        upper=parse_bound(parts[1]),
    )
