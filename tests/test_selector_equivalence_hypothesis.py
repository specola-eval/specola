"""
Property-based selector longhand-equivalence (eval-experiments plan, Task 2).

``test_spec_selectors.py`` proves the desugaring right on the cases the
author thought of. This module randomizes the same property: a generated
catalog and a generated selector expression, desugared through
``compile_specs``, must agree numerically with a longhand expansion built
independently from the catalog.

"Independently" is the whole point. The expected key set here comes from a
matcher written against the grammar documented in ``spec/selectors.py``,
never from ``resolve_selector`` — reusing the implementation would test the
code against itself and pass no matter what either side did.

Scope. The generator emits only legal specs; illegal-input rejection is
already pinned by ``test_spec_selectors.py`` and is not re-tested here. It
covers exact/glob/list atoms, the optional ``@ group`` filter, ``\\``
exclusions, the ``any``/``all`` direction table, ``min``/``max``/``sum``
aggregators, wildcard entity slots, and ``**`` entity expansion. Left out
deliberately: structural ``tags`` as a second ``@`` target (every generated
key carries the default empty tags, so only group labels are exercised),
capitalized ``Min``/``Max``, and arithmetic inside aggregator arguments.

A counterexample is a real desugaring bug: shrink it, reproduce it
standalone, report it. Do not weaken the property.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

import pytest

# Skip visibly (not a collection error) in an env without hypothesis, so
# `pytest tests/` stays runnable everywhere while the skip is loud in the
# summary. All simulator envs have hypothesis installed (issue #32).
pytest.importorskip(
    "hypothesis",
    reason="selector-equivalence suite is property-based; pip install hypothesis",
)
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from specola.adapter import PropertyMeta
from specola.components import ComponentCatalog, ComponentResolver
from specola.entities import EntityState, TrackState
from specola.stl_parser import compile_specs, parse_stl


# ---------------------------------------------------------------------------
# Vocabulary
#
# Groups are assigned per key independently of the key's prefix, so a group
# filter selects a subset no glob can express and the intersection of the
# two is exercised rather than one masking the other.
# ---------------------------------------------------------------------------

CLASSES = ("robot", "object")
ENTITY_IDS = (1, 2, 3)
KEYS = ("ka_a", "ka_b", "ka_c", "tb_1", "tb_2", "px_x", "px_y")
GROUPS = ("accel", "sensors: touch", "body position")
GLOBS = ("*", "ka_*", "tb_*", "px_*", "*_a", "*_1", "k*_*")
OPERATORS = ("<=", "<", ">=", ">")
THRESHOLDS = (-1.0, 0.0, 0.5, 2.0, 50.0)


class Case(NamedTuple):
    """One generated example: the sugared spec and its expected expansion."""

    sugared: str
    # (name suffix or None, longhand text) per expected compiled formula.
    expected: Tuple[Tuple[str, str], ...]
    entities: Dict[int, EntityState]
    catalog: ComponentCatalog
    resolver: ComponentResolver


# ---------------------------------------------------------------------------
# Independent oracle — written from the grammar in spec/selectors.py, which
# states: "Only '*' is a wildcard; every other character is literal."
# ---------------------------------------------------------------------------

def _glob_matches(pattern: str, name: str) -> bool:
    """True when ``name`` matches ``pattern`` under that one-wildcard rule."""
    literals = pattern.split("*")

    if len(literals) == 1:
        return name == pattern

    head, tail = literals[0], literals[-1]

    if not name.startswith(head) or not name.endswith(tail):
        return False

    cursor = len(head)
    for literal in literals[1:-1]:
        found = name.find(literal, cursor)
        if found < 0:
            return False
        cursor = found + len(literal)

    # The head and tail must not overlap in the same characters.
    return cursor <= len(name) - len(tail)


def _term_matches(
    atom: str,
    names: Sequence[str],
    is_glob: bool,
    group: str,
    keys: Sequence[str],
    groups: Dict[str, str],
) -> set:
    """One include or exclude term, resolved against one entity's keys."""
    if is_glob:
        matched = {key for key in keys if _glob_matches(atom, key)}
    else:
        matched = {name for name in names if name in keys}

    if group:
        matched = {key for key in matched if groups[key] == group}

    return matched


class Selector(NamedTuple):
    """A selector as authored, kept separate from its resolution."""

    atom: str
    names: Tuple[str, ...]
    is_glob: bool
    group: str
    excludes: Tuple[Tuple[str, Tuple[str, ...], bool, str], ...]
    quantifier: str

    def is_trivial(self) -> bool:
        """Whether this is an old-style exact name, per ``ast.py``.

        A trivial selector is not desugared at all: it stays one formula
        with the reference's own slot, wildcard included. The fan-out
        shapes below therefore have nothing to compare against and skip it.
        """
        return (
            not self.quantifier
            and not self.excludes
            and not self.is_glob
            and not self.group
            and len(self.names) == 1
            and self.names[0] == self.atom
        )

    def text(self) -> str:
        parts = [_term_text(self.atom, self.group)]
        for atom, _names, _is_glob, group in self.excludes:
            parts.append(_term_text(atom, group))
        body = " \\ ".join(parts)
        if self.quantifier:
            return f"{self.quantifier}: {body}"
        return body

    def resolve(
        self, keys: Sequence[str], groups: Dict[str, str]
    ) -> Tuple[str, ...]:
        """The oracle: which of ``keys`` this selector names, sorted."""
        included = _term_matches(
            self.atom, self.names, self.is_glob, self.group, keys, groups
        )
        for atom, names, is_glob, group in self.excludes:
            included -= _term_matches(
                atom, names, is_glob, group, keys, groups
            )
        return tuple(sorted(included))

    def excludes_all_bite(
        self, keys: Sequence[str], groups: Dict[str, str]
    ) -> bool:
        """Every exclusion must remove something, or compilation errors.

        Exclusions resolve against the entity's whole key set, not against
        what the include term matched.
        """
        return all(
            _term_matches(atom, names, is_glob, group, keys, groups)
            for atom, names, is_glob, group in self.excludes
        )


def _term_text(atom: str, group: str) -> str:
    return f"{atom} @ {group}" if group else atom


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

@st.composite
def _worlds(draw, min_keys=1):
    """A catalog: entities of one or two classes, each with its own keys.

    ``min_keys`` raises the floor on per-entity key count. Properties that
    are about multi-key expansion set it rather than filtering for the
    multi-key case afterwards, which starves the generator.
    """
    n_entities = draw(st.integers(min_value=1, max_value=3))
    chosen_ids = sorted(
        draw(
            st.sets(
                st.sampled_from(ENTITY_IDS),
                min_size=n_entities,
                max_size=n_entities,
            )
        )
    )

    group_of = {key: draw(st.sampled_from(GROUPS)) for key in KEYS}

    entities = {}
    metadata = {}
    for entity_id in chosen_ids:
        keys = sorted(
            draw(
                st.sets(
                    st.sampled_from(KEYS),
                    min_size=min_keys,
                    max_size=len(KEYS),
                )
            )
        )
        values = {
            key: draw(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                )
            )
            for key in keys
        }
        entity_class = draw(st.sampled_from(CLASSES))
        entities[entity_id] = EntityState(
            entity_id=entity_id,
            entity_class=entity_class,
            position=None,
            velocity=None,
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties=values,
        )
        metadata[entity_id] = {
            key: PropertyMeta(group=group_of[key]) for key in keys
        }

    return entities, metadata, group_of


@st.composite
def _selectors(
    draw,
    group_of,
    with_excludes=True,
    quantified=False,
    kinds=("exact", "glob", "list"),
):
    kind = draw(st.sampled_from(kinds))

    if kind == "glob":
        atom = draw(st.sampled_from(GLOBS))
        names: Tuple[str, ...] = ()
        is_glob = True
    elif kind == "list":
        members = sorted(
            draw(st.sets(st.sampled_from(KEYS), min_size=1, max_size=3))
        )
        atom = "{" + ", ".join(members) + "}"
        names = tuple(members)
        is_glob = False
    else:
        name = draw(st.sampled_from(KEYS))
        atom = name
        names = (name,)
        is_glob = False

    group = draw(st.sampled_from(("",) + GROUPS))

    excludes: List[Tuple[str, Tuple[str, ...], bool, str]] = []
    if with_excludes:
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            if draw(st.booleans()):
                excludes.append(
                    (draw(st.sampled_from(GLOBS)), (), True, "")
                )
            else:
                name = draw(st.sampled_from(KEYS))
                excludes.append((name, (name,), False, ""))

    quantifier = draw(st.sampled_from(("any", "all"))) if quantified else ""

    # Reported by --hypothesis-show-statistics, so "did this suite actually
    # exercise groups and exclusions" is answered by the log, not by argument.
    event(f"atom kind: {kind}")
    event(f"group filter: {bool(group)}")
    event(f"exclude terms: {len(excludes)}")
    if quantifier:
        event(f"quantifier: {quantifier}")

    return Selector(
        atom=atom,
        names=names,
        is_glob=is_glob,
        group=group,
        excludes=tuple(excludes),
        quantifier=quantifier,
    )


def _build(entities, metadata):
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    return catalog, ComponentResolver(entities)


def _quantifier_aggregator(quantifier: str, operator: str) -> str:
    """The value aggregator a quantified selector must compile to.

    ``all`` takes the worst case for the comparison's direction, ``any``
    the best. For an upward comparison the worst case is the minimum.
    """
    upward = operator in (">=", ">")
    if quantifier == "any":
        return "max" if upward else "min"
    return "min" if upward else "max"


# ---------------------------------------------------------------------------
# Robustness harness
# ---------------------------------------------------------------------------

def _rho(formula, resolver):
    binding = {
        parameter.param_key: parameter.pinned_entity_id
        for parameter in formula.params
    }
    assert all(
        entity_id is not None for entity_id in binding.values()
    ), f"expected every slot pinned, got {formula.params}"
    return formula.predicate_fn(binding, resolver)


def _suffix(formula) -> str:
    """The expansion suffix compile_specs appends to a fanned-out name."""
    tail = formula.name.rsplit(" ", 1)[-1]
    return tail if tail.startswith("[") and tail.endswith("]") else ""


def _assert_matches(case: Case):
    formulas = compile_specs(case.sugared, catalog=case.catalog)

    assert len(formulas) == len(case.expected), (
        f"{case.sugared!r} expanded to {len(formulas)} formulas, "
        f"expected {len(case.expected)}: "
        f"got {[f.name for f in formulas]}, "
        f"want {[suffix for suffix, _ in case.expected]}"
    )

    # A single-formula expansion is matched positionally. Which suffix a
    # one-formula expansion carries is Property 4's subject, not this
    # property's, and pairing on it here would conflate the two.
    if len(formulas) == 1:
        by_suffix = {case.expected[0][0]: formulas[0]}
    else:
        by_suffix = {_suffix(formula): formula for formula in formulas}

    for suffix, longhand in case.expected:
        assert suffix in by_suffix, (
            f"{case.sugared!r} produced no formula with suffix {suffix!r}; "
            f"got {sorted(by_suffix)}"
        )
        got = _rho(by_suffix[suffix], case.resolver)
        want = _rho(
            parse_stl(longhand, catalog=case.catalog), case.resolver
        )
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (
            f"{case.sugared!r} {suffix} = {got!r}, "
            f"longhand {longhand!r} = {want!r}"
        )


# ---------------------------------------------------------------------------
# Property 1: pinned-entity shapes (the key dimension alone)
# ---------------------------------------------------------------------------

@st.composite
def _pinned_cases(draw):
    entities, metadata, group_of = draw(_worlds())
    entity_id = draw(st.sampled_from(sorted(entities)))
    entity = entities[entity_id]
    keys = sorted(entity.properties)

    shape = draw(st.sampled_from(("bare", "min", "max", "sum", "quantified")))
    selector = draw(
        _selectors(group_of, quantified=(shape == "quantified"))
    )

    resolved = selector.resolve(keys, group_of)
    assume(resolved)
    assume(selector.excludes_all_bite(keys, group_of))

    # Issue #78: a trivial selector skips desugaring entirely, so the literal
    # `sum(` survives into the numeric expression and sympy raises. Excluded
    # here rather than fixed, and pinned below by test_sum_over_trivial_...
    assume(not (shape == "sum" and selector.is_trivial()))

    event(f"pinned shape: {shape}")
    event(f"resolved keys: {min(len(resolved), 5)}")

    operator = draw(st.sampled_from(OPERATORS))
    threshold = draw(st.sampled_from(THRESHOLDS))
    head = f"{entity.entity_class}_{entity_id}"
    reference = f"{head}[{selector.text()}]"

    def longhand_args():
        return ", ".join(f"{head}[{key}]" for key in resolved)

    if shape == "bare":
        sugared = f"{reference} {operator} {threshold}"
        expected = tuple(
            (
                f"[{key}@{entity_id}]" if len(resolved) > 1 else "",
                f"{head}[{key}] {operator} {threshold}",
            )
            for key in resolved
        )
    elif shape == "sum":
        sugared = f"sum({reference}) {operator} {threshold}"
        summed = " + ".join(f"{head}[{key}]" for key in resolved)
        expected = (("", f"({summed}) {operator} {threshold}"),)
    elif shape == "quantified":
        aggregator = _quantifier_aggregator(selector.quantifier, operator)
        sugared = f"{reference} {operator} {threshold}"
        expected = (
            (
                "",
                f"{aggregator}({longhand_args()}) {operator} {threshold}",
            ),
        )
    else:
        sugared = f"{shape}({reference}) {operator} {threshold}"
        expected = (
            ("", f"{shape}({longhand_args()}) {operator} {threshold}"),
        )

    catalog, resolver = _build(entities, metadata)
    return Case(sugared, expected, entities, catalog, resolver)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(case=_pinned_cases())
def test_pinned_selector_matches_longhand(case):
    _assert_matches(case)


# ---------------------------------------------------------------------------
# Property 2: wildcard entity slot — entities resolving empty are skipped,
# never an error and never a vacuous monitor
# ---------------------------------------------------------------------------

@st.composite
def _wildcard_cases(draw):
    entities, metadata, group_of = draw(_worlds())
    entity_class = draw(st.sampled_from(CLASSES))
    members = sorted(
        entity_id
        for entity_id, entity in entities.items()
        if entity.entity_class == entity_class
    )
    assume(members)

    shape = draw(st.sampled_from(("bare", "min", "max")))
    # Exclusions resolve per entity, so a term that bites on one member and
    # not another would be testing legality, not equivalence. Include-only
    # keeps every generated wildcard spec legal by construction.
    selector = draw(_selectors(group_of, with_excludes=False))
    assume(not selector.is_trivial())

    per_entity = {
        entity_id: selector.resolve(
            sorted(entities[entity_id].properties), group_of
        )
        for entity_id in members
    }
    matching = [
        entity_id for entity_id in members if per_entity[entity_id]
    ]
    assume(matching)

    event(f"wildcard shape: {shape}")
    event(f"matching entities: {len(matching)}")

    operator = draw(st.sampled_from(OPERATORS))
    threshold = draw(st.sampled_from(THRESHOLDS))
    reference = f"{entity_class}_*[{selector.text()}]"

    if shape == "bare":
        sugared = f"{reference} {operator} {threshold}"
        fanned = [
            (entity_id, key)
            for entity_id in matching
            for key in per_entity[entity_id]
        ]
        expected = tuple(
            (
                f"[{key}@{entity_id}]" if len(fanned) > 1 else "",
                f"{entity_class}_{entity_id}[{key}] {operator} {threshold}",
            )
            for entity_id, key in fanned
        )
    else:
        sugared = f"{shape}({reference}) {operator} {threshold}"
        expected = tuple(
            (
                f"[@{entity_id}]" if len(matching) > 1 else "",
                "{}({}) {} {}".format(
                    shape,
                    ", ".join(
                        f"{entity_class}_{entity_id}[{key}]"
                        for key in per_entity[entity_id]
                    ),
                    operator,
                    threshold,
                ),
            )
            for entity_id in matching
        )

    catalog, resolver = _build(entities, metadata)
    return Case(sugared, expected, entities, catalog, resolver)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(case=_wildcard_cases())
def test_wildcard_entity_selector_matches_longhand(case):
    _assert_matches(case)


# ---------------------------------------------------------------------------
# Property 3: ** collapses every matching entity into ONE formula
# ---------------------------------------------------------------------------

@st.composite
def _double_star_cases(draw):
    entities, metadata, group_of = draw(_worlds())
    entity_class = draw(st.sampled_from(CLASSES))
    members = sorted(
        entity_id
        for entity_id, entity in entities.items()
        if entity.entity_class == entity_class
    )
    assume(members)

    selector = draw(
        _selectors(
            group_of,
            with_excludes=False,
            quantified=draw(st.booleans()),
        )
    )
    assume(not selector.is_trivial())

    per_entity = {
        entity_id: selector.resolve(
            sorted(entities[entity_id].properties), group_of
        )
        for entity_id in members
    }
    matching = [
        entity_id for entity_id in members if per_entity[entity_id]
    ]
    assume(matching)

    event(f"** entities: {len(matching)}")
    event(f"** total args: {min(sum(len(per_entity[e]) for e in matching), 6)}")

    operator = draw(st.sampled_from(OPERATORS))
    threshold = draw(st.sampled_from(THRESHOLDS))
    reference = f"{entity_class}_**[{selector.text()}]"

    tokens = [
        f"{entity_class}_{entity_id}[{key}]"
        for entity_id in matching
        for key in per_entity[entity_id]
    ]
    arguments = ", ".join(tokens)

    if selector.quantifier:
        aggregator = _quantifier_aggregator(selector.quantifier, operator)
        sugared = f"{reference} {operator} {threshold}"
    else:
        aggregator = draw(st.sampled_from(("min", "max", "sum")))
        sugared = f"{aggregator}({reference}) {operator} {threshold}"

    if aggregator == "sum":
        longhand = f"({' + '.join(tokens)}) {operator} {threshold}"
    else:
        longhand = f"{aggregator}({arguments}) {operator} {threshold}"

    expected = (("", longhand),)

    catalog, resolver = _build(entities, metadata)
    return Case(sugared, expected, entities, catalog, resolver)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(case=_double_star_cases())
def test_double_star_selector_matches_longhand(case):
    _assert_matches(case)


# ---------------------------------------------------------------------------
# Property 4: the fan-out suffix names exactly the resolved keys
#
# Results are keyed on these names downstream (report aggregation groups by
# the spec column), so a suffix naming the wrong key would misattribute a
# verdict without changing any robustness number.
# ---------------------------------------------------------------------------

@st.composite
def _fanout_cases(draw):
    entities, metadata, group_of = draw(_worlds(min_keys=4))
    entity_id = draw(st.sampled_from(sorted(entities)))
    entity = entities[entity_id]
    keys = sorted(entity.properties)

    # Globs and lists are what produce a multi-key expansion; an exact atom
    # resolves to at most one key and has no fan-out to name.
    selector = draw(_selectors(group_of, kinds=("glob", "list")))
    resolved = selector.resolve(keys, group_of)
    assume(len(resolved) > 1)
    assume(selector.excludes_all_bite(keys, group_of))

    event(f"fanout width: {min(len(resolved), 5)}")

    head = f"{entity.entity_class}_{entity_id}"
    sugared = f"{head}[{selector.text()}] <= 1.0"

    catalog, resolver = _build(entities, metadata)
    return sugared, resolved, entity_id, catalog


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(case=_fanout_cases())
def test_fanout_suffixes_name_the_resolved_keys(case):
    sugared, resolved, entity_id, catalog = case
    formulas = compile_specs(sugared, catalog=catalog)

    assert {_suffix(formula) for formula in formulas} == {
        f"[{key}@{entity_id}]" for key in resolved
    }, f"{sugared!r} -> {[formula.name for formula in formulas]}"

    for formula in formulas:
        for parameter in formula.params:
            assert parameter.is_wildcard is False
            assert parameter.pinned_entity_id == entity_id


# ---------------------------------------------------------------------------
# Known defect, pinned so a fix flips it visibly (specola #78)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="specola #78: sum() over a trivial reference skips desugaring, so "
    "the literal 'sum(' reaches sympy and raises \"'Symbol' object is not "
    "iterable\". Delete this xfail when #78 is fixed.",
)
def test_sum_over_trivial_reference_should_not_reach_sympy():
    entities = {
        1: EntityState(
            entity_id=1,
            entity_class="robot",
            position=None,
            velocity=None,
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties={"ka_a": 3.0},
        )
    }
    metadata = {1: {"ka_a": PropertyMeta(group="accel")}}
    catalog, resolver = _build(entities, metadata)

    formulas = compile_specs("sum(robot_1[ka_a]) <= 50", catalog=catalog)

    assert _rho(formulas[0], resolver) == pytest.approx(47.0)
