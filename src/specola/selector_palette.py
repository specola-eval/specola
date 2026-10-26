"""Every selector form available for one discovery selection.

``discovery_selector.selector_for_signal`` answers a narrower question:
what is *the* reference for this signal. That is the right answer when a
UI needs one selector to insert, and the wrong one when a person is
learning what the language can express — the entity binding token and
the component bracket vary independently, and the choice between them
changes how many monitors a specification creates.

This module enumerates the distinct selector *shapes* for a selection
and verifies each one against the reset snapshot. Shapes, not
instantiations: a class with seventy-eight entities offers the same
handful of forms as a class with two, so the row count is bounded by the
grammar rather than by the scene. Each binding token carries the
component shapes that teach something about it — the wildcard token
carries the full bracket vocabulary because that is where a reader meets
it; ``**`` carries only the quantified forms, because a bare ``**``
bracket does not compile.

Verification runs the production parser and resolver
(``discovery_selector._resolved_pairs``), never a reimplementation, so a
form's bracket resolves against this scene exactly as the compiler will
resolve it. That is bracket-level verification and no more: it does not
parse the entity head, apply adapter capabilities, or know where in a
formula the reference will be placed. Where a caller can supply a
``ComponentCatalog``, ``verify_forms`` closes the gap by putting each
form's probe through the semantic layer -- routed exactly as
``compile_formulas`` routes it, so the check answers the question
production asks rather than a narrower one.

Measured on RoboCasa's ``results/20260816_004207_robocasa_notask``
recording (98 entities, 30,259 signals), whole palettes: ``contact``
0.030 s, ``object`` 0.018 s, ``robot`` 0.073 s, ``fixture`` 1.94 s. The
gap is worth closing: ten of ``contact``'s sixteen forms resolve
cleanly and are illegal. An earlier version of this paragraph said
"2.4 ms per form" without naming a scene; that number was from the mock
adapter's five entities and was wrong about RoboCasa by four orders of
magnitude. Performance claims here name their scene.

Placement is not a property of the reference and cannot be verified by
either layer: a quantified selector must be the whole comparison side,
so ``Max(class_*[all: k]) >= 0`` is rejected while
``class_*[all: k] >= 0`` compiles. Those forms carry that constraint in
their note, because it is invisible in the reference text itself.

A shape that cannot be expressed for a selection is returned with its
reason rather than dropped: the absence of a form is itself information
a spec author needs.

Nothing here decides presentation. Consumers render the fields they
want: the Textual explorer, the Rich overview, the plain renderer, the
JSON document, and the browser all read the same records.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Dict, List, Optional, Sequence, Tuple

from .discovery_filter import DiscoveryView
from .discovery_model import DiscoveryEntity, DiscoveryProperty
from .discovery_selector import (
    DiscoveryScope,
    ROLE_ABSENT,
    SignalSelection,
    _glob_for_keys,
    _resolved_pairs,
    entities_for_scope,
    scope_for_view,
    signal_selections,
)


# How many names a generated ``{a, b, c}`` literal or ``\`` exclusion may
# carry before the shape is withheld. A literal exists to name an exact
# set a reader can check at a glance; past this many names it stops doing
# that, and the glob or group form says the same thing better.
LIST_CAP = 8

# How many ``@ label`` shapes to instantiate. Adapters that tag keys with
# per-entity identity (RoboCasa emits ``declared_name:<name>``,
# ``model_index:<n>``, ``fixture_ref:<name>``) produce one label per
# entity, so an uncapped enumeration returns hundreds of rows that all
# teach the same shape -- measured 531 forms and 9.16 s on the
# 84-entity fixture class of
# results/20260816_004207_robocasa_notask (label_cap=10**6, 2026-08-16). Labels are ranked by how many keys they cover
# so the ones that select a family come before the ones that select a
# single key, and the number withheld is reported on the last kept row:
# a cap that hides its own existence would read as "these are all the
# labels there are".
LABEL_CAP = 8


# Shape identifiers. Stable strings: the JSON document and the browser
# both key off them, so they are API rather than display text.
PIN = "pin"
WILDCARD = "wildcard"
FANOUT = "fanout"
SLOTTED = "slotted"

EXACT = "exact"
GLOB = "glob"
EVERY = "every"
LIST = "list"
GROUP = "group"
TAG = "tag"
ROLE = "role"
DIFFERENCE = "difference"
QUANTIFIED = "quantified"


@dataclass(frozen=True)
class SelectorForm:
    """One complete, pasteable entity reference and what it resolves to.

    ``reference`` is the whole thing — ``object_*[xpos_*_2]`` — because a
    bare ``object_*`` is not a reference and cannot be checked, copied,
    or inserted into a formula.

    ``binding_kind`` names the entity token's shape (pin, wildcard,
    fanout, slotted) and ``component_kind`` the bracket's (exact, glob,
    every, list, group, tag, role, difference, quantified). The pair
    identifies the shape; ``reference`` is one instantiation of it.

    ``bindings`` is the verified expansion as ``(entity_id, exact_key)``
    pairs. It is one relation, not two lists: an entity may resolve
    several keys and a key may appear on several entities, so parallel
    ``entity_ids``/``keys`` tuples could not say which key belongs to
    which entity -- and that pairing is exactly what shows why
    ``object_*[xpos_obj_main_2]`` binds one object out of three.
    ``entity_ids`` and ``keys`` remain available as derived views.

    ``monitors`` is how many monitor instances the form creates, which is
    not ``len(bindings)``: a quantified or fanned-out form collapses its
    expansion into one verdict. It is None for a two-slot form, whose
    fan-out the semantic layer decides.

    ``unavailable_reason`` is non-None exactly when the shape cannot be
    expressed for this selection, and then ``reference`` still carries
    the text that was attempted so a reader can see what was tried.
    """

    reference: str
    binding_kind: str
    component_kind: str
    entity_form: str
    component_form: str
    bindings: Tuple[Tuple[int, str], ...]
    monitors: Optional[int]
    quantifier: Optional[str] = None
    note: Optional[str] = None
    unavailable_reason: Optional[str] = None
    # Whether this form's legality was decided by the semantic layer, as
    # opposed to by bracket resolution alone. A row that was never
    # checked must not be indistinguishable from one that passed: the
    # whole reason to check is that the two disagree -- on RoboCasa's
    # ``contact`` class, 10 of 16 bracket-available forms are illegal.
    verified: bool = False
    # Whether this form reproduces the role the scope selected. Carried
    # rather than inferred from ``component_kind``, because the row that
    # expresses a role can be one an earlier shape already produced --
    # a tag form and a role form reach identical text -- and only one of
    # them is kept.
    reproduces_role: bool = False
    # Concrete formulas this form desugars to at compile time. Not the
    # same as ``len(bindings)``, which is how many monitors those
    # formulas later produce: a trivial reference such as
    # ``object_*[mass]`` is one formula whose wildcard survives to
    # runtime and binds two entities. Worth showing either way --
    # ``fixture{A}_*[*]`` is 648,802,096 formulas.
    expansions: Optional[int] = None
    # The second reference of a two-slot form; None for every other
    # shape. Slotted forms are the only ones that are not a single
    # reference, and flattening them into ``reference`` would produce a
    # string that does not parse.
    partner_reference: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def entity_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({entity_id for entity_id, _key in self.bindings}))

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted({key for _entity, key in self.bindings}))


@dataclass(frozen=True)
class _Candidate:
    """One bracket text awaiting verification."""

    component_form: str
    component_kind: str
    quantifier: Optional[str] = None
    note: Optional[str] = None
    check_mapping: bool = False


def _properties_by_key(
    entities: Sequence[DiscoveryEntity],
) -> Dict[str, List[DiscoveryProperty]]:
    by_key: Dict[str, List[DiscoveryProperty]] = {}
    for entity in entities:
        for prop in entity.properties:
            by_key.setdefault(prop.key, []).append(prop)
    return by_key


def _labels_for_keys(
    entities: Sequence[DiscoveryEntity],
    keys: Sequence[str],
    cap: int,
) -> Tuple[Tuple[Tuple[str, str], ...], int]:
    """The ``@`` labels worth offering for ``keys``, and how many were not.

    A selector's ``@ label`` matches a key when the label equals its
    group *or* is one of its tags, so both vocabularies produce the same
    syntax and are enumerated together. The kind is kept because the two
    answer different questions — a group says what kind of quantity a key
    is and carries its unit, a tag says where on the machine it sits and
    may span units.

    Groups come first so the unit-safe choice is the one a reader meets
    first, and within each kind the labels covering the most keys come
    first: a label that selects one key is an instantiation, while a
    label that selects a family is the shape worth learning. A label
    serving as both group and tag is reported once, as a group — that is
    the stronger claim, and the selector cannot distinguish them anyway
    (``ComponentCatalog.ambiguous_filter_labels`` reports the collision
    separately).

    The second return value is how many labels were left out, which the
    caller must surface rather than swallow.
    """
    wanted = set(keys)
    coverage: Dict[Tuple[str, str], int] = {}
    groups = set()
    for entity in entities:
        for prop in entity.properties:
            if prop.key not in wanted:
                continue
            if prop.group:
                groups.add(prop.group)
    for entity in entities:
        for prop in entity.properties:
            if prop.key not in wanted:
                continue
            if prop.group:
                key = (prop.group, GROUP)
                coverage[key] = coverage.get(key, 0) + 1
            for tag in prop.tags:
                if not tag or tag in groups:
                    continue
                key = (tag, TAG)
                coverage[key] = coverage.get(key, 0) + 1
    ordered = sorted(
        coverage,
        key=lambda item: (
            0 if item[1] == GROUP else 1,
            -coverage[item],
            item[0],
        ),
    )
    return tuple(ordered[:cap]), max(0, len(ordered) - cap)


def _monitor_count(
    pairs: Sequence[Tuple[int, str]],
    *,
    binding_kind: str,
    quantifier: Optional[str],
) -> Optional[int]:
    """Monitors a form creates, given its verified (entity, key) pairs.

    A bare selector creates one monitor per matched key per bound entity.
    A quantifier collapses one entity's keys into a single verdict, and
    ``**`` collapses the whole class into one. This mirrors the table in
    docs/writing-specs.md rather than restating its reasoning.
    """
    if not pairs:
        return 0
    if binding_kind == FANOUT:
        return 1
    if binding_kind == SLOTTED:
        # Not computed. A two-slot fan-out is the ordered pairs of
        # distinct entities, but each pair's monitor count depends on
        # what the bracket resolves to *for that entity* -- entities of
        # one class need not carry the same keys -- and on the author's
        # ``symmetric:`` assertion, which halves the pairs and which
        # nothing here can know. A plausible product of the three
        # numbers would be wrong whenever the class is heterogeneous,
        # which is the case this palette exists to make visible.
        return None
    if quantifier is not None:
        return len({entity_id for entity_id, _key in pairs})
    return len(pairs)


def _verified(
    *,
    population: Sequence[DiscoveryEntity],
    entity_form: str,
    candidate: _Candidate,
    binding_kind: str,
    expected: Optional[Sequence[Tuple[int, str]]],
    scoped_ids: frozenset,
    partner: Optional[str],
) -> SelectorForm:
    """Resolve one candidate through the production parser and resolver.

    ``expected`` is the (entity, key) mapping the caller believes the
    form should produce. A mismatch makes the form unavailable rather
    than silently offering a reference that monitors something other than
    what was selected; it is checked only for candidates that claim to
    reproduce the selection exactly -- a widening shape such as ``*`` or
    ``@ label`` deliberately resolves to more, and comparing it would
    report the widening as a defect.

    The comparison is taken over ``scoped_ids`` only. A class-wide token
    resolves against every entity of the class, which for a role- or
    entity-narrowed scope is a superset of what the selection covered;
    comparing the raw pair set would fail every shape on such a scope for
    a reason that has nothing to do with the shape.
    """
    bracket = candidate.component_form
    if candidate.quantifier is not None:
        bracket = "{}: {}".format(candidate.quantifier, bracket)
    reference = "{}[{}]".format(entity_form, bracket)
    pairs = _resolved_pairs(tuple(population), bracket)
    reason: Optional[str] = None
    bound_entities = len({entity_id for entity_id, _ in pairs})
    if not pairs:
        reason = "Resolves to no reset-time binding."
    elif binding_kind == SLOTTED and bound_entities < 2:
        reason = (
            "Two slots need two distinct entities carrying this "
            "component; {} carries it.".format(bound_entities)
        )
    elif (
        candidate.check_mapping
        and expected is not None
        and tuple(sorted(
            pair for pair in pairs if pair[0] in scoped_ids
        )) != tuple(sorted(expected))
    ):
        reason = (
            "Resolves to a different entity/key mapping than the "
            "selection."
        )
    return SelectorForm(
        reference=reference,
        binding_kind=binding_kind,
        component_kind=candidate.component_kind,
        entity_form=entity_form,
        component_form=bracket,
        bindings=tuple(sorted(pairs)),
        monitors=_monitor_count(
            pairs,
            binding_kind=binding_kind,
            quantifier=candidate.quantifier,
        ),
        quantifier=candidate.quantifier,
        note=candidate.note,
        unavailable_reason=reason,
        partner_reference=(
            "{}[{}]".format(partner, bracket)
            if partner is not None else None
        ),
    )


def _difference_candidate(
    population: Sequence[DiscoveryEntity],
    selected_keys: Sequence[str],
    glob: str,
    list_cap: int,
) -> Optional[_Candidate]:
    """A ``glob \\ {names}`` candidate reaching ``selected_keys``.

    What the glob matches is asked of the resolver rather than recomputed
    here, so the subtraction is taken against the same key set the
    compiler will see. Returns None when the shape teaches nothing —
    the glob already lands on exactly the wanted keys, or the exclusion
    needs more names than a reader can check. An exclusion that removes
    nothing is a compile error in the resolver, so it is never generated.
    """
    if "*" not in glob:
        return None
    matched = {key for _entity, key in _resolved_pairs(
        tuple(population), glob
    )}
    wanted = set(selected_keys)
    if not wanted or not wanted <= matched:
        return None
    excluded = sorted(matched - wanted)
    if not excluded or len(excluded) > list_cap:
        return None
    return _Candidate(
        component_form="{} \\ {{{}}}".format(glob, ", ".join(excluded)),
        component_kind=DIFFERENCE,
        note=(
            "subtract the whole set you mean; single-quote the YAML or "
            "it eats the backslash"
        ),
        check_mapping=True,
    )


def _entity_tokens(
    view: DiscoveryView,
    scope: DiscoveryScope,
    entity_class: str,
) -> Tuple[Tuple[str, str, Optional[str]], ...]:
    """Entity tokens offered for a scope, as (form, kind, note).

    A pinned scope leads with its own id and still offers the wildcard,
    because widening is the common next question. A class or role scope
    has no id to pin, so the wildcard leads. The two-slot token appears
    only when the class actually holds two entities to distinguish.
    """
    population = sum(
        entity.entity_class == entity_class for entity in view.entities
    )
    tokens: List[Tuple[str, str, Optional[str]]] = []
    if scope.entity_id is not None:
        tokens.append((
            "{}_{}".format(entity_class, scope.entity_id),
            PIN,
            "this entity only; ids are per-scene",
        ))
    tokens.append((
        "{}_*".format(entity_class),
        WILDCARD,
        "one monitor per entity carrying the component",
    ))
    tokens.append((
        "{}_**".format(entity_class),
        FANOUT,
        "whole class in one verdict; legal only under any:, all:, or an "
        "aggregator",
    ))
    if population >= 2:
        tokens.append((
            "{}{{A}}_*".format(entity_class),
            SLOTTED,
            "two distinct entities of the class; add symmetric: true "
            "when the expression is order-independent",
        ))
    return tuple(tokens)


def _candidates_for_token(
    *,
    binding_kind: str,
    entity_note: Optional[str],
    scoped: Sequence[DiscoveryEntity],
    population: Sequence[DiscoveryEntity],
    selected_keys: Sequence[str],
    scope_keys: Sequence[str],
    list_cap: int,
    label_cap: int,
) -> List[_Candidate]:
    """Bracket shapes worth showing under one entity token.

    Each shape is demonstrated once, under the token where it teaches the
    most. The wildcard carries the full bracket vocabulary because that
    is the token a reader meets it on; a pinned entity carries only the
    two forms that differ from it; ``**`` carries only the quantified
    forms, since a bare ``**`` bracket does not compile; the two-slot
    token carries one form, because its lesson is the slots.
    """
    glob = _glob_for_keys(selected_keys)
    exact = _Candidate(
        component_form=selected_keys[0],
        component_kind=EXACT,
        note=(
            entity_note
            if len(selected_keys) == 1
            else "exact key; {} more in this selection".format(
                len(selected_keys) - 1
            )
        ),
        check_mapping=len(selected_keys) == 1,
    )
    # A key set with no shared prefix or suffix condenses to a bare
    # ``*``, which is the every-component shape rather than a glob over
    # a naming convention. Naming it GLOB would offer the same text
    # twice under two different lessons.
    globbed = _Candidate(
        component_form=glob,
        component_kind=EVERY if glob == "*" else GLOB,
        note=(
            "every component the entity exposes"
            if glob == "*"
            else "* is the only wildcard and it crosses underscores"
        ),
        check_mapping=glob != "*",
    )

    if binding_kind == PIN:
        return [exact] + ([globbed] if glob != selected_keys[0] else [])

    if binding_kind == FANOUT:
        base = globbed if glob != selected_keys[0] else exact
        return [
            _Candidate(
                component_form=base.component_form,
                component_kind=QUANTIFIED,
                quantifier=quantifier,
                note=(
                    (
                        "one verdict over every entity and key"
                        if quantifier == "any"
                        else "one verdict; every entity and key must hold"
                    )
                    + "; must be the whole comparison side, not inside "
                    "Min/Max/sum"
                ),
            )
            for quantifier in ("any", "all")
        ]

    if binding_kind == SLOTTED:
        return [
            _Candidate(
                component_form=base.component_form,
                component_kind=base.component_kind,
                note=entity_note,
            )
            for base in (globbed if glob != selected_keys[0] else exact,)
        ]

    # WILDCARD carries the full vocabulary.
    candidates = [exact]
    if glob != selected_keys[0]:
        candidates.append(globbed)
    if sorted(scope_keys) != sorted(selected_keys):
        candidates.append(_Candidate(
            component_form="*",
            component_kind=EVERY,
            note="every component the entity exposes",
        ))
    if 2 <= len(selected_keys) <= list_cap:
        candidates.append(_Candidate(
            component_form="{{{}}}".format(", ".join(selected_keys)),
            component_kind=LIST,
            note="fails loudly if one of these is renamed away",
            check_mapping=True,
        ))
    labels, withheld = _labels_for_keys(scoped, selected_keys, label_cap)
    for index, (label, kind) in enumerate(labels):
        note = (
            "one unit per group, so this aggregates safely"
            if kind == GROUP
            else "structural; a tag may span units"
        )
        if withheld and index == len(labels) - 1:
            note += "; {} further label{} not listed".format(
                withheld, "" if withheld == 1 else "s"
            )
        candidates.append(_Candidate(
            component_form="{} @ {}".format(glob, label),
            component_kind=kind,
            note=note,
        ))
    difference = _difference_candidate(
        population, selected_keys, glob, list_cap
    )
    if difference is not None:
        candidates.append(difference)
    for quantifier in ("all", "any"):
        candidates.append(_Candidate(
            component_form=glob,
            component_kind=QUANTIFIED,
            quantifier=quantifier,
            note=(
                "one verdict per entity; the quantifier picks the "
                "aggregation for the comparison, and must be the whole "
                "comparison side, not inside Min/Max/sum"
            ),
        ))
    return _deduped(candidates)


def _deduped(candidates: Sequence[_Candidate]) -> List[_Candidate]:
    """Drop repeats of the same bracket text under one entity token.

    Several shapes converge on the same text on small or heterogeneous
    key sets — a one-key selection's glob is its exact key, and a key set
    with nothing in common condenses to ``*``. The first spelling wins,
    because the candidate list is ordered by what each shape teaches.
    """
    seen = set()
    unique: List[_Candidate] = []
    for candidate in candidates:
        signature = (candidate.component_form, candidate.quantifier)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    return unique


def selector_palette(
    view: DiscoveryView,
    scope: DiscoveryScope,
    *,
    signal: Optional[SignalSelection] = None,
    quantity: Optional[str] = None,
    list_cap: int = LIST_CAP,
    label_cap: int = LABEL_CAP,
) -> Tuple[SelectorForm, ...]:
    """Every distinct selector shape available for one selection.

    ``scope`` narrows the entities; ``signal`` narrows to one aggregated
    signal within it, and ``quantity`` to one quantity family. Passing
    neither enumerates the shapes over every component in scope, which is
    what selecting a bare class means.

    The returned order is the order a person reads them in: most specific
    binding token first, and within a token the exact component before
    the ways of widening it.
    """
    scoped = entities_for_scope(view, scope)
    if not scoped:
        return ()
    classes = {entity.entity_class for entity in scoped}
    if len(classes) != 1:
        # A selector reference names exactly one class. A cross-class
        # scope has no single reference, and inventing one would offer a
        # form the compiler rejects.
        return ()
    entity_class = next(iter(classes))

    # Class-wide references resolve against every entity of the class,
    # not the scoped subset: a role-narrowed scope whose class has other
    # members would otherwise hide the over-binding the reference causes.
    class_entities = tuple(
        entity for entity in view.entities
        if entity.entity_class == entity_class
    )

    scope_keys = sorted(_properties_by_key(scoped))
    scoped_ids = frozenset(entity.entity_id for entity in scoped)
    expected: Optional[Tuple[Tuple[int, str], ...]] = None
    if signal is not None:
        selected_keys = sorted({
            mapping.exact_key for mapping in signal.mappings
        })
        expected = tuple(sorted(
            (mapping.entity_id, mapping.exact_key)
            for mapping in signal.mappings
        ))
    elif quantity is not None:
        selected_keys = sorted({
            prop.key
            for entity in scoped
            for prop in entity.properties
            if (prop.group or "ungrouped") == quantity
        })
    else:
        selected_keys = scope_keys
    if not selected_keys:
        return ()

    forms: List[SelectorForm] = []
    for entity_form, binding_kind, entity_note in _entity_tokens(
        view, scope, entity_class
    ):
        population = (
            scoped if binding_kind == PIN else class_entities
        )
        partner = (
            "{}{{B}}_*".format(entity_class)
            if binding_kind == SLOTTED else None
        )
        # Which shapes claim to reproduce the selection exactly is the
        # candidate's own ``check_mapping``, not the token's: a glob
        # condensed from the selection's keys, an explicit list of them,
        # and a difference built to reach them all make that claim on
        # every token. The comparison is confined to the scoped entities
        # inside ``_verified``, so a class-wide token resolving against
        # the whole class does not fail for entities the selection never
        # covered.
        for candidate in _candidates_for_token(
            binding_kind=binding_kind,
            entity_note=entity_note,
            scoped=scoped,
            population=population,
            selected_keys=selected_keys,
            scope_keys=scope_keys,
            list_cap=list_cap,
            label_cap=label_cap,
        ):
            forms.append(_verified(
                population=population,
                entity_form=entity_form,
                candidate=candidate,
                binding_kind=binding_kind,
                expected=expected,
                scoped_ids=scoped_ids,
                partner=partner,
            ))
        if binding_kind == WILDCARD:
            role_form = _role_candidate(
                population=population,
                scoped=scoped,
                scope=scope,
                entity_form=entity_form,
                binding_kind=binding_kind,
                glob=_glob_for_keys(selected_keys),
                selected_keys=selected_keys,
            )
            if role_form is not None:
                forms.append(role_form)
    return _without_duplicate_references(forms)


def _without_duplicate_references(forms: List[SelectorForm]):
    """One row per reference, because the reference is what identifies it.

    The role candidate is built after the per-token dedup and can land on
    text an earlier form already used: on a roleless scope every class
    emitted ``fixture_*[*]`` twice, once available and once refused with
    the note that a role cannot be written in this grammar. Two rows with
    identical text and opposite verdicts are wrong on the screen, and
    they also corrupt any cache keyed by reference -- the terminal
    writer's verdict cache is, so whichever row was focused first
    poisoned the other in whichever direction.

    The earlier row wins and keeps its verdict; the later row's reason
    is preserved as a note, because what it was there to teach -- that
    the selection's role is not expressible -- is still true.
    """
    kept: List[SelectorForm] = []
    at = {}
    for form in forms:
        index = at.get(form.reference)
        if index is None:
            at[form.reference] = len(kept)
            kept.append(form)
            continue
        # Merge rather than discard. The duplicate is usually the role
        # candidate, and what it carries is not redundant even when its
        # text is: either it names the row that reproduces a role
        # selection -- which ``default_form`` must land on, or a role
        # scope silently monitors entities the selection excluded -- or
        # it explains why the role could not be expressed at all.
        existing = kept[index]
        merged = existing
        if form.component_kind == ROLE and form.available:
            merged = _replace(merged, reproduces_role=True)
        # The role row's own words survive, because what it explains --
        # that this reference does or does not reproduce the selected
        # role -- is invisible in the reference text either way.
        carried = form.note or (
            form.unavailable_reason if not form.available else None
        )
        if carried:
            merged = _replace(merged, note=carried)
        kept[index] = merged
    return tuple(kept)


def palette_for_view(
    view: DiscoveryView,
    *,
    list_cap: int = LIST_CAP,
    label_cap: int = LABEL_CAP,
    verify: bool = True,
) -> Tuple[Tuple[SelectorForm, ...], Optional[str]]:
    """The palette a filtered view implies, or why it implies none.

    The non-interactive surfaces — plain, Rich, JSON — express a
    selection through CLI filters rather than a tree click, so they need
    one shared answer to "what did this filter select". Sharing
    ``scope_for_view`` with the single-preview path is what stops the
    same filter meaning different things on different surfaces.

    A filter that narrows to one signal palettes that signal; one that
    leaves several palettes the scope as a whole, which is the same
    thing the explorer shows before a signal row is highlighted.

    Returns ``(forms, None)`` or ``((), reason)``.
    """
    scope, reason = scope_for_view(view)
    if scope is None:
        return (), reason
    signals = signal_selections(view, scope)
    signal = signals[0] if len(signals) == 1 else None
    forms = selector_palette(
        view,
        scope,
        signal=signal,
        list_cap=list_cap,
        label_cap=label_cap,
    )
    if not forms:
        return (), "No selector form is available for this selection."
    if verify:
        # Every surface checks, because every surface that shows a form
        # also offers a starter for it, and bracket availability is not
        # the compiler's answer: ten of RoboCasa's sixteen `contact`
        # forms resolve cleanly and are illegal. The catalog is derived
        # from the view's own model rather than asked for, so a caller
        # cannot forget to pass it -- which is how these three surfaces
        # came to show unchecked rows as though they were checked.
        forms = _verified_against_model(view, forms)
    return forms, None


def _verified_against_model(view: DiscoveryView, forms):
    """Check a palette using the model the view was filtered from.

    Returns the forms unchanged if a catalog cannot be built -- a
    surface showing bracket-level answers is worse than one showing
    compiler-level answers and better than one showing an exception.
    Unchecked forms are distinguishable: ``verified`` stays False.
    """
    model = getattr(view, "model", None)
    if model is None:
        return forms
    try:
        from .discovery_restore import catalog_from_model

        return verify_forms(
            forms,
            catalog=catalog_from_model(model),
            dt=model.effective_dt,
            capabilities=getattr(model, "capabilities", None),
        )
    except Exception:                               # noqa: BLE001
        return forms


STARTER_NAME = "selected_signal_requirement"
STARTER_WINDOW = "historically[0,1s]"


# Above this many expansions a form is offered without a pasteable
# starter. The palette's own verification no longer compiles, so nothing
# here is protecting the verifier -- it protects the person who presses
# Check. `check_draft` compiles for real, holds a process-global lock
# while it does, and has no bound of its own; the widest RoboCasa form,
# `fixture{A}_*[*]`, is 648,802,096 expansions. `object{A}_*[*]` at
# 366,054 takes 3.3 s to expand, and 648M is nearly 1,800x that.
#
# The form stays listed and stays marked available, because it is: the
# count is a fact about what it would build, not a defect in it.
STARTER_EXPANSION_CAP = 100_000


def starter_spec_for_form(
    form: SelectorForm,
    *,
    name: str = STARTER_NAME,
) -> dict:
    """A valid, compilable specification entry built around one form.

    One authority, because a starter is generated on four surfaces and a
    second implementation is a second chance to emit something the
    compiler rejects. The mock this UI was designed from got three
    things wrong that this cannot: it authored a ``window:`` key that is
    not in the schema and is silently ignored, and it called ``dist()``
    and ``norm()``, which are not registered functions and parse into
    undefined ones.

    The window comes from the temporal operator, which is the only place
    a window can come from. A quantified reference is placed as the
    whole comparison side, never inside an aggregator, because the
    compiler rejects the latter.

    A two-slot form generates ``abs(a - b) <= threshold``, and both
    halves of that matter.

    ``abs`` is required by the ``symmetric`` flag. Where the flag bites,
    it makes the enumerator emit one sorted representative per unordered
    pair rather than both orderings (``bindings/enumerate.py``,
    ``itertools.combinations``). Under ``a - b >= threshold`` that is
    wrong: swapping the slots flips the sign, so exactly one arbitrary
    ordering of each pair would be evaluated and the other's violations
    never reported. ``abs`` makes the expression genuinely
    order-independent, which is what the flag asserts.

    Where the flag bites is narrower than it looks, and an earlier
    version of this docstring claimed the flag "keeps the halved monitor
    count", which is false for most of the forms that carry it.
    ``_expand_formula`` pins both entities before ``symmetric`` is ever
    consulted -- it does not take ``symmetric`` as a parameter -- so a
    two-slot form over a *non-trivial* bracket is fully materialized and
    the enumerator has no wildcard slots left to fold, and it requires
    two. Measured on RoboCasa's ``mocap{A}_*[mocap_*]``: symmetric True
    and False both give 98 templates, zero wildcard parameters. The
    halving is real only for the exact-key bracket a signal-scoped
    palette produces. ``abs`` is required either way; the monitor count
    simply is not halved in the first case.

    ``<=`` is required because a starter must be falsifiable. Every
    starter ships ``threshold: 0.0`` as a placeholder the author
    replaces, and ``abs(a - b) >= 0`` is true of every binding -- an
    author who ran it unmodified would get a green monitor that cannot
    fail and no sign that anything was left to do. As a proximity
    requirement it fails until the threshold is set, which is the
    honest default for a template. The one-slot form keeps ``>=``: it
    already discriminates at zero.
    """
    if form.partner_reference:
        body = "abs({} - {}) <= threshold".format(
            form.reference, form.partner_reference
        )
    else:
        body = "{} >= threshold".format(form.reference)
    entry = {
        "name": name,
        "formula": "{}({})".format(STARTER_WINDOW, body),
        "constants": {"threshold": 0.0},
    }
    if form.partner_reference:
        entry["symmetric"] = True
    return entry


def starter_yaml_for_form(
    form: SelectorForm,
    *,
    name: str = STARTER_NAME,
) -> str:
    """The starter as a YAML document, quoted by the YAML writer.

    Never string concatenation. A difference form carries a backslash,
    and YAML treats ``\\`` inside double quotes as an escape — hand-built
    text loses it silently and the selector then matches everything it
    was supposed to subtract from. ``safe_dump`` picks a quoting style
    that survives.
    """
    import yaml

    return yaml.safe_dump(
        {"specs": [starter_spec_for_form(form, name=name)]},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _probe_formula(form: SelectorForm) -> str:
    """The smallest legal formula that exercises one form's shape.

    The probe stays out of aggregators, and that is a measured decision
    rather than a stylistic one. A bare selector is legal whatever its
    expansion size -- it means one monitor per matched key -- while
    ``Max(...)`` over the same selector makes the compiler build one
    symbolic expression with a term per key. On RoboCasa's object class
    the wide ``object_*[*]`` form (741 keys) compiles bare in 0.276 s and
    does not finish inside 25 s wrapped in ``Max``. A quantified selector
    additionally must be the whole comparison side, so an aggregator
    would reject it outright.

    Two references appear only for the two-slot form, subtracted rather
    than aggregated for the same reason.
    """
    if form.partner_reference:
        body = "{} - {} >= 0".format(form.reference, form.partner_reference)
    else:
        body = "{} >= 0".format(form.reference)
    return "historically[0,1s]({})".format(body)


def _slot_pairing(form: SelectorForm) -> Tuple[bool, int]:
    """Whether a two-slot form has a legal assignment, and how many.

    Both slots of a palette two-slot form carry the same class and the
    same bracket, so the arrangement question is only whether two
    *distinct* entities resolve -- and the expansion count is the ordered
    pairs of their keys::

        sum_{a != b} k_a k_b  =  (sum_e k_e)^2 - sum_e k_e^2

    Both come from one pass over the already-resolved ``(entity, key)``
    pairs. Enumerating the cross product to learn the same two facts
    costs 6.9 s and 366,054 rewritten formulas for ``object{A}_*[*]`` on
    RoboCasa.
    """
    per_entity: Dict[int, int] = {}
    for entity_id, _key in form.bindings:
        per_entity[entity_id] = per_entity.get(entity_id, 0) + 1
    total = sum(per_entity.values())
    squares = sum(count * count for count in per_entity.values())
    return len(per_entity) >= 2, total * total - squares


def _validate_expansion(
    form: SelectorForm, *, catalog, dt: float, capabilities,
) -> Tuple[Optional[str], Optional[int]]:
    """Decide one form's legality without building any predicate.

    Returns ``(refusal, expansions)``; ``refusal`` is None when the form
    is legal.

    This is the semantic layer's own answer, reached by the semantic
    layer's own code -- ``_expand_formula`` resolves the entity head
    against the class vocabulary, applies adapter capabilities, enforces
    ``**`` placement and quantifier position, and validates declared
    group and tag labels. Every refusal this UI cares about is raised
    there.

    What it deliberately does not do is compile the expansions into
    evaluable predicates. That step exists to produce something the
    monitor can call, and it answers no legality question: profiling
    ``object_*[all: *]`` attributed 109.7 s of a 109.8 s profiled
    compile to sympy's ``Min``/``Max`` constructor, doing pairwise
    algebraic simplification over one term per component. That is a
    profiled time and inflated: the same compile is 33.3 s unprofiled.
    The attribution is what matters and it survives -- essentially all
    of it is inside sympy -- and expansion alone reaches the same
    verdicts in about a millisecond per form.

    One check the predicate step owns is not reproduced here: an unknown
    free symbol, i.e. a constant an author named and never defined. It
    cannot arise for a probe, which compares against a literal ``0``, so
    this is complete for palette forms and is *not* a general substitute
    for compiling a formula somebody typed. ``check_draft`` still
    compiles.
    """
    from .entities import SpecConfigError
    from .spec.grammar import PARAM_RE
    from .spec.semantic import _expand_formula, _needs_desugaring
    from .stl_parser import _prepare, compile_specs

    # Entity class names are not validated where entity states are built,
    # while reference heads have a stricter lexical grammar. A class name
    # that cannot be written as a reference would otherwise reach the
    # compiler as a parse error rather than a refusal here.
    # PARAM_RE matches a whole reference, bracket included -- it is the
    # production that finds references inside a formula, not a head
    # pattern.
    for reference in filter(None, (form.reference, form.partner_reference)):
        if not PARAM_RE.fullmatch(reference):
            return ("Not a valid reference: {!r} cannot be written in "
                    "the selector grammar".format(reference), None)

    if form.partner_reference:
        # Validate the bracket once, through the single-slot form that
        # shares it, then settle the arrangement arithmetically.
        single = _replace(form, partner_reference=None)
        refusal, _ = _validate_expansion(
            single, catalog=catalog, dt=dt, capabilities=capabilities
        )
        if refusal is not None:
            return refusal, None
        pairable, expansions = _slot_pairing(form)
        if not pairable:
            return ("Needs two distinct entities of this class; only one "
                    "carries the selected component", None)
        return None, expansions

    probe = _probe_formula(form)
    try:
        node, prepared = _prepare(
            probe, dt, {}, False, catalog, None, "auto", capabilities,
        )
        # Route exactly as ``compile_formulas`` routes, because routing
        # differently is how a check ends up answering a question
        # production never asked. A formula with no non-trivial selector
        # never reaches ``_expand_formula`` in production -- it goes
        # straight to ``compile_formula``, which is the only caller of
        # ``_validate_slot_against_catalog`` (semantic.py:846) and so the
        # only place five refusals are raised: a component outside a
        # declared CLOSED schema, no entity jointly exposing the required
        # components, a pinned id absent, a pinned class mismatch, a
        # pinned entity missing components.
        #
        # Sending those formulas to expansion instead checked strictly
        # less than production does, and it did so for the palette's most
        # common shape: `object_*[mass]` is trivial, and `default_form`
        # returns the first available row. Compiling one is 0.5-23 ms
        # measured on RoboCasa, because a trivial formula has no fan-out
        # -- the cost this module exists to avoid is the *expansion*
        # case, and that one still never compiles.
        if not _needs_desugaring(node):
            compile_specs(
                probe, dt=dt, capabilities=capabilities, constants={},
                symmetric=bool(form.partner_reference), catalog=catalog,
                backend="auto",
            )
            return None, 1
        expansions, _filters = _expand_formula(
            node, prepared["catalog"], prepared["class_vocabulary"],
            prepared["capabilities"],
        )
    except (SpecConfigError, ValueError, TypeError) as exc:
        return "Does not compile: {}".format(exc), None
    return None, len(expansions)


def verify_forms(
    forms: Sequence[SelectorForm],
    *,
    catalog,
    dt: float,
    capabilities=None,
) -> Tuple[SelectorForm, ...]:
    """Re-verify a palette against the semantic layer, not the bracket.

    ``selector_palette`` resolves bracket text against the reset
    snapshot, which is what the compiler will do to that bracket but is
    not the whole of what the compiler does: it also parses the entity
    head, enforces the ``**`` placement rule, applies adapter
    capabilities, and rejects slot arrangements the grammar allows but
    the semantics do not. On RoboCasa's ``contact`` class that gap is 10
    of 16 forms -- each one bracket-available, illegal, and carrying a
    starter somebody would paste.

    Every returned form carries ``verified``. A caller must not treat an
    unverified form as checked: this used to return capped forms
    unchanged, which made "not checked" and "checked and fine"
    indistinguishable in both surfaces.

    Forms already unavailable keep their original reason: a
    resolution-level refusal is more specific than a semantic one, and
    this pass exists to catch what resolution could not see, not to
    relabel what it already caught.
    """
    verified: List[SelectorForm] = []
    for form in forms:
        if not form.available:
            verified.append(form)
            continue
        refusal, expansions = _validate_expansion(
            form, catalog=catalog, dt=dt, capabilities=capabilities
        )
        verified.append(_replace(
            form,
            verified=True,
            expansions=expansions,
            unavailable_reason=refusal,
        ))
    return tuple(verified)


def default_form(
    forms: Sequence[SelectorForm],
    scope: DiscoveryScope,
) -> Optional[SelectorForm]:
    """The form a consumer should land on for ``scope``.

    Normally the first available one, which is the narrowest binding
    with the most exact component — the same reference the single-preview
    panel used to produce.

    A scope that declares a role is the exception. Selecting a role means
    selecting those entities, and a plain ``class_*`` reference binds the
    whole class, so landing on it would silently monitor entities the
    selection excluded. Where the role filter is expressible, it is the
    form that reproduces the selection, and it wins.

    Shared rather than reimplemented per renderer: the Textual explorer,
    the plain renderer and the browser must agree on which reference a
    selection means, or the same click produces different specifications
    in different surfaces.
    """
    available = [form for form in forms if form.available]
    if not available:
        return forms[0] if forms else None
    if isinstance(scope.role, str):
        roled = next(
            (
                form for form in available
                if form.reproduces_role or form.component_kind == ROLE
            ),
            None,
        )
        if roled is not None:
            return roled
    return available[0]


def _role_candidate(
    *,
    population: Sequence[DiscoveryEntity],
    scoped: Sequence[DiscoveryEntity],
    scope: DiscoveryScope,
    entity_form: str,
    binding_kind: str,
    glob: str,
    selected_keys: Sequence[str],
) -> Optional[SelectorForm]:
    """The ``@ role:<role>`` form, or the reason there cannot be one.

    A role is declared on the entity, but the selector filters on the
    key's ``role:*`` property tag, and an adapter may declare the first
    without emitting the second. A roleless scope has no expressible
    form at all: the grammar can filter *for* a label, never *against*
    every label.
    """
    if scope.role is ROLE_ABSENT:
        return SelectorForm(
            reference="{}[{}]".format(entity_form, glob),
            binding_kind=binding_kind,
            component_kind=ROLE,
            entity_form=entity_form,
            component_form=glob,
            bindings=(),
            monitors=0,
            unavailable_reason=(
                "The selector grammar cannot express only entities with "
                "no declared role."
            ),
        )
    if not isinstance(scope.role, str):
        return None
    role_tag = "role:" + scope.role
    component_form = "{} @ {}".format(glob, role_tag)
    wanted = set(selected_keys)
    tagged = any(
        role_tag in prop.tags
        for entity in scoped
        for prop in entity.properties
        if prop.key in wanted
    )
    if not tagged:
        return SelectorForm(
            reference="{}[{}]".format(entity_form, component_form),
            binding_kind=binding_kind,
            component_kind=ROLE,
            entity_form=entity_form,
            component_form=component_form,
            bindings=(),
            monitors=0,
            unavailable_reason=(
                "Role is declared but its role:* property tag is absent."
            ),
        )
    return _verified(
        population=population,
        entity_form=entity_form,
        candidate=_Candidate(
            component_form=component_form,
            component_kind=ROLE,
            note="binds only entities declared with this role",
        ),
        binding_kind=binding_kind,
        expected=None,
        scoped_ids=frozenset(
            entity.entity_id for entity in scoped
        ),
        partner=None,
    )
