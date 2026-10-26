"""
selectors.py — component-selector grammar: bracket text -> ComponentSelector.

Mirrors grammar.py's ``parse_bound`` / ``parse_interval``: text in, an
immutable value out, nothing else. No catalog, no simulator, no entity
resolution — matching a selector against discovered components is a
Task-3 concern (see ``ComponentSelector.is_trivial`` in ``ast.py`` for the
boundary the old semantic-compile path still understands).

Grammar (informal), applied to the text found inside one component
bracket ``[...]``:

    selector   := [ quantifier ":" ] term ( "\\" term )*
    quantifier := "any" | "all"
    term       := atom [ "@" group ]
    atom       := list | glob-or-name
    list       := "{" name ( "," name )* "}"     (whitespace tolerated)
    glob-or-name := [a-z_][a-z0-9_*]*             (no internal whitespace)
    group      := free text, trimmed

A bare atom (no quantifier, glob, list, group, or exclude clause) is a
"trivial" selector — an exact component name, identical to every
pre-selector reference form.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import FrozenSet, List, Mapping, Optional, Set, Tuple

from ..entities import SpecConfigError
from .ast import ComponentSelector, SelectorTerm


_QUANTIFIER_RE = re.compile(r"^(any|all)\s*:\s*", re.IGNORECASE)

# An atom is an exact key, or a glob. Globs use "*" only and may begin with
# one (a bare "*" is "every key"), so the leading character set admits "*".
_ATOM_RE = re.compile(r"^[a-z_*][a-z0-9_*]*$")

_LIST_ITEM_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@lru_cache(maxsize=4096)
def parse_selector(text: str) -> ComponentSelector:
    """Parse one component-bracket's text into a ``ComponentSelector``.

    Raises ``ValueError`` on any malformed selector: an empty atom, an
    unbalanced or nested ``{...}``, a stray ``@``, or a bare atom that
    contains whitespace (e.g. a quantifier keyword used without its
    trailing ``:``).

    Memoized, which is sound because this is a pure function of its text
    returning frozen dataclasses, and it is the only producer of them.
    Worth doing because desugaring re-parses one bracket once per
    expansion: profiling a `fixture_*[*]` palette on RoboCasa showed
    168,268 calls for a single selector, with ``_top_level_indices``
    scanning the same string 336,536 times and 7.5 million ``isspace``
    calls underneath it. Malformed text raises rather than returns, and
    ``lru_cache`` does not cache exceptions, so a bad selector is
    re-parsed each time -- which costs nothing, since it fails fast.
    """
    raw = text
    work = text.strip()

    if not work:
        raise ValueError(f"Empty component selector {raw!r}")

    quantifier_match = _QUANTIFIER_RE.match(work)

    if quantifier_match is not None:
        quantifier = quantifier_match.group(1).lower()
        remainder = work[quantifier_match.end():]
    else:
        quantifier = None
        remainder = work

    term_texts = _split_top_level(remainder, "\\")

    if not term_texts[0].strip():
        raise ValueError(f"Empty selector atom in {raw!r}")

    include = _parse_term(term_texts[0])
    exclude = tuple(
        _parse_term(term_text) for term_text in term_texts[1:]
    )

    return ComponentSelector(
        quantifier=quantifier,
        include=include,
        exclude=exclude,
        raw=raw,
    )


def _parse_term(term_text: str) -> SelectorTerm:
    term_text = term_text.strip()

    if not term_text:
        raise ValueError("Empty selector term")

    at_positions = _top_level_indices(term_text, "@")

    if len(at_positions) > 1:
        raise ValueError(
            f"Selector term has more than one '@': {term_text!r}"
        )

    if at_positions:
        split_index = at_positions[0]
        atom_text = term_text[:split_index].strip()
        group = term_text[split_index + 1:].strip()
        if not atom_text or not group:
            raise ValueError(
                f"Stray '@' in selector term {term_text!r}"
            )
    else:
        atom_text = term_text
        group = None

    if atom_text.startswith("{"):
        atom, names = _parse_list_literal(atom_text)
        is_glob = False
    else:
        if any(character.isspace() for character in atom_text):
            raise ValueError(
                f"Selector atom must not contain spaces: {atom_text!r}"
            )
        if not _ATOM_RE.fullmatch(atom_text):
            raise ValueError(f"Invalid selector atom {atom_text!r}")
        atom = atom_text
        names = (atom_text,)
        is_glob = "*" in atom_text

    return SelectorTerm(
        atom=atom,
        names=names,
        is_glob=is_glob,
        group=group,
    )


def _parse_list_literal(atom_text: str) -> Tuple[str, Tuple[str, ...]]:
    if not atom_text.endswith("}"):
        raise ValueError(f"Unterminated list literal {atom_text!r}")

    inner = atom_text[1:-1]

    if "{" in inner or "}" in inner:
        raise ValueError(f"Nested braces are not allowed: {atom_text!r}")

    names: List[str] = []

    for item in inner.split(","):
        name = item.strip()
        if not name:
            raise ValueError(
                f"Empty entry in list literal {atom_text!r}"
            )
        if not _LIST_ITEM_RE.fullmatch(name):
            raise ValueError(
                f"Invalid list entry {name!r} in {atom_text!r}"
            )
        names.append(name)

    return atom_text, tuple(names)


def _top_level_indices(text: str, character: str) -> List[int]:
    """Indices of ``character`` outside any ``{...}`` span.

    Also validates brace balance/nesting for ``text`` as a side effect,
    since every caller needs that guarantee before splitting on it.
    """
    depth = 0
    indices: List[int] = []

    for index, current in enumerate(text):
        if current == "{":
            depth += 1
            if depth > 1:
                raise ValueError(
                    f"Nested braces are not allowed in {text!r}"
                )
        elif current == "}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unmatched '}}' in {text!r}")
        elif depth == 0 and current == character:
            indices.append(index)

    if depth != 0:
        raise ValueError(f"Unmatched '{{' in {text!r}")

    return indices


def _split_top_level(text: str, character: str) -> List[str]:
    indices = _top_level_indices(text, character)
    parts: List[str] = []
    start = 0

    for index in indices:
        parts.append(text[start:index])
        start = index + 1

    parts.append(text[start:])
    return parts


# ---------------------------------------------------------------------------
# Resolution against a discovered component set (Task 3)
# ---------------------------------------------------------------------------

def resolve_selector(
    selector: ComponentSelector,
    components: FrozenSet[str],
    groups: Mapping[str, str],
    tags: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> Tuple[str, ...]:
    """Resolve one selector against ONE entity's component set.

    ``components`` is the entity's component keys (any collection); ``groups``
    maps a component key to its metadata group label; ``tags`` maps a
    component key to a tuple of structural tag labels. The result is the
    sorted tuple of matched keys.

    Resolution is include-first: the include term's atom (exact / glob /
    list) matches, its optional ``@ group`` filter intersects (matching a
    key when the label equals its group OR is one of its tags), and each
    ``\\``-separated exclude term is then subtracted. An empty final set —
    or an exclude term that removes nothing (a misspelled difference) —
    raises ``SpecConfigError`` naming the entity's real keys, never a
    vacuous truth.
    """
    component_set = frozenset(components)
    tag_map: Mapping[str, Tuple[str, ...]] = tags or {}

    included = _resolve_term(selector.include, component_set, groups, tag_map)

    for term in selector.exclude:
        removed = _resolve_term(term, component_set, groups, tag_map)
        if not removed:
            raise SpecConfigError(
                f"Selector exclusion {term.atom!r} in {selector.raw!r} "
                f"matched no component. Available components: "
                f"{sorted(component_set)}"
            )
        included = included - removed

    if not included:
        message = (
            f"Component selector {selector.raw!r} matched no component. "
            f"Available components: {sorted(component_set)}"
        )
        # A group filter that matches nothing is almost always a mistyped
        # group name; naming the entity's real groups points straight at it.
        uses_group = selector.include.group is not None or any(
            term.group is not None for term in selector.exclude
        )
        if uses_group:
            available_groups = sorted(
                {group for group in groups.values() if group}
            )
            message += f". Available groups: {available_groups}"
            available_tags = sorted(
                {
                    tag
                    for key_tags in tag_map.values()
                    for tag in key_tags
                }
            )
            if available_tags:
                message += f". Available tags: {available_tags}"
        raise SpecConfigError(message)

    return tuple(sorted(included))


def _resolve_term(
    term: SelectorTerm,
    components: FrozenSet[str],
    groups: Mapping[str, str],
    tags: Mapping[str, Tuple[str, ...]],
) -> Set[str]:
    if term.is_glob:
        pattern = _glob_pattern(term.atom)
        matched = {
            component
            for component in components
            if pattern.match(component)
        }
    else:
        # Exact name or {list}: each authored member matches only if the
        # entity actually exposes it.
        matched = {
            name for name in term.names if name in components
        }

    if term.group is not None:
        wanted = term.group.strip()
        matched = {
            component
            for component in matched
            if groups.get(component, "").strip() == wanted
            or wanted in tags.get(component, ())
        }

    return matched


def _glob_pattern(atom: str) -> "re.Pattern":
    # Only "*" is a wildcard; every other character is literal.
    escaped = re.escape(atom).replace(r"\*", ".*")
    return re.compile(f"^{escaped}$")
