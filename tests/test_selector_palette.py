"""The palette may only offer references the compiler accepts.

Every assertion here is about that contract rather than about wording:
a form's text must lex as an entity reference, an available form must
resolve to something, and an unavailable one must say why. The mock
adapter supplies a class with two entities and heterogeneous keys, which
is the shape that separates the interesting cases (a glob that
over-binds, a two-slot form with nothing to pair, a difference that
subtracts a real key).
"""

from __future__ import annotations

from dataclasses import replace

import re

import pytest

from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import (
    build_discovery_model,
    property_metadata_for,
)
from specola.discovery_selector import (
    DiscoveryScope,
    ROLE_ABSENT,
    signal_selections,
)
from specola.selector_palette import (
    GLOB,
    GROUP,
    LIST,
    TAG,
    SelectorForm,
    _validate_expansion,
    verify_forms,
    EXACT,
    FANOUT,
    LABEL_CAP,
    PIN,
    QUANTIFIED,
    ROLE,
    SLOTTED,
    WILDCARD,
    selector_palette,
)
from specola.spec.grammar import PARAM_RE
from specola.spec.selectors import parse_selector


@pytest.fixture(scope="module")
def view():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    return filter_discovery(model, DiscoveryQuery(include_raw=False))


def _object_scope(view):
    return DiscoveryScope("object")


def _pinned_object_scope(view):
    entity = next(
        entity for entity in view.entities
        if entity.entity_class == "object"
    )
    return DiscoveryScope("object", entity_id=entity.entity_id)


def _all_forms(view):
    """Every form over every scope the mock exposes, signal by signal."""
    out = []
    for entity_class in sorted({
        entity.entity_class for entity in view.entities
    }):
        scope = DiscoveryScope(entity_class)
        out.extend(selector_palette(view, scope))
        for signal in signal_selections(view, scope):
            out.extend(selector_palette(view, scope, signal=signal))
    return out


def test_every_reference_lexes_as_an_entity_reference(view):
    """A form that PARAM_RE cannot see is not a reference at all."""
    forms = _all_forms(view)
    assert forms
    for form in forms:
        for text in filter(None, (form.reference, form.partner_reference)):
            match = PARAM_RE.search(text)
            assert match is not None, text
            # The whole string must be the reference, not a prefix of it:
            # a bracket the lexer stops short of would silently monitor
            # less than the palette displayed.
            assert match.group(0) == text, text


def test_every_bracket_parses_as_a_selector(view):
    for form in _all_forms(view):
        parse_selector(form.component_form)


def test_available_forms_resolve_and_unavailable_ones_explain(view):
    for form in _all_forms(view):
        if form.available:
            assert form.entity_ids, form.reference
            assert form.keys, form.reference
        else:
            assert form.unavailable_reason, form.reference


def test_monitor_count_is_stated_or_withheld_never_guessed(view):
    """A two-slot fan-out is the semantic layer's to decide, not this one.

    Its size depends on which keys each paired entity actually carries
    and on the author's ``symmetric:`` assertion, so the palette reports
    None rather than a plausible product. Every other shape resolves to
    a count it can stand behind.
    """
    for form in _all_forms(view):
        if form.binding_kind == SLOTTED:
            assert form.monitors is None, form.reference
        elif form.available:
            assert form.monitors is not None, form.reference
            assert form.monitors > 0, form.reference


def test_fanout_is_never_offered_bare(view):
    """``**`` outside a quantifier or aggregator does not compile."""
    fanout = [
        form for form in _all_forms(view)
        if form.binding_kind == FANOUT
    ]
    assert fanout
    for form in fanout:
        assert form.quantifier in ("any", "all"), form.reference
        assert form.component_kind == QUANTIFIED


def test_pinned_scope_leads_with_its_own_id_and_still_widens(view):
    forms = selector_palette(view, _pinned_object_scope(view))
    kinds = [form.binding_kind for form in forms]
    assert kinds[0] == PIN
    assert WILDCARD in kinds
    pinned = next(form for form in forms if form.binding_kind == PIN)
    assert pinned.entity_form.startswith("object_")
    assert "*" not in pinned.entity_form


def test_class_scope_offers_no_pin(view):
    forms = selector_palette(view, _object_scope(view))
    assert all(form.binding_kind != PIN for form in forms)


def test_two_slot_form_needs_two_entities_in_the_class(view):
    """The mock's gripper class holds one entity; object holds two."""
    gripper = selector_palette(view, DiscoveryScope("gripper"))
    assert all(form.binding_kind != SLOTTED for form in gripper)

    objects = selector_palette(view, _object_scope(view))
    slotted = [f for f in objects if f.binding_kind == SLOTTED]
    assert slotted
    for form in slotted:
        assert form.partner_reference
        assert "{A}" in form.reference
        assert "{B}" in form.partner_reference


def test_two_slot_form_over_a_single_entity_signal_is_refused(view):
    """Two slots bind two distinct entities, so one is not enough.

    The mock gives ``hinge_angle`` to one object only, so a two-slot
    form over it can produce no pair at all. That is a refusal with a
    reason, not a monitor count of zero: the count is never claimed for
    this shape.
    """
    scope = _object_scope(view)
    signal = next(
        signal for signal in signal_selections(view, scope)
        if len({m.entity_id for m in signal.mappings}) == 1
    )
    slotted = [
        form for form in selector_palette(view, scope, signal=signal)
        if form.binding_kind == SLOTTED
    ]
    assert slotted
    for form in slotted:
        assert not form.available
        assert "two distinct entities" in form.unavailable_reason


def test_no_duplicate_reference_within_a_palette(view):
    for entity_class in sorted({
        entity.entity_class for entity in view.entities
    }):
        scope = DiscoveryScope(entity_class)
        for signal in (None,) + signal_selections(view, scope):
            forms = selector_palette(view, scope, signal=signal)
            references = [form.reference for form in forms]
            assert len(references) == len(set(references)), entity_class


def test_quantifier_collapses_the_monitor_count(view):
    """A quantified form is one verdict per entity; a bare one is per key."""
    scope = _object_scope(view)
    forms = selector_palette(view, scope)
    bare = {
        form.component_form: form for form in forms
        if form.binding_kind == WILDCARD and form.quantifier is None
    }
    quantified = [
        form for form in forms
        if form.binding_kind == WILDCARD and form.quantifier is not None
    ]
    assert quantified
    for form in quantified:
        twin = bare.get(form.component_form.split(": ", 1)[1])
        if twin is None or len(twin.keys) < 2:
            continue
        assert form.monitors < twin.monitors
        assert form.monitors == len(form.entity_ids)


def _over_binding_view(view):
    """Two entities whose keys share a prefix but not a selection.

    Entity 1 carries ``xpos_0``/``xpos_1``; entity 2 carries ``xpos_2``.
    A selection of entity 1's keys condenses to the glob ``xpos_*``,
    which also matches entity 2 — the exact situation where a generated
    glob monitors more than was selected.
    """
    template = next(
        entity for entity in view.entities
        if entity.entity_class == "object"
    )
    prop = template.properties[0]

    def make(entity_id, keys, group):
        # signal_selections groups by (class, group, unit, component,
        # frame, family-or-key), so the group is what decides which keys
        # aggregate into one selectable signal. Giving entity 1's keys a
        # shared group and entity 2's a different one produces exactly
        # one signal whose condensed glob reaches past its own selection.
        return replace(
            template,
            entity_id=entity_id,
            properties=tuple(
                replace(
                    prop, key=key, family="xpos", group=group, tags=()
                )
                for key in keys
            ),
        )

    return replace(view, entities=(
        make(1, ("xpos_0", "xpos_1"), "planar position"),
        make(2, ("xpos_2",), "height"),
    ))


def test_a_glob_that_over_binds_the_selection_is_refused(view):
    """A generated glob must not silently widen what was selected.

    Without this check the palette offers ``object_*[xpos_*]`` for a
    selection covering only entity 1, and the resulting specification
    monitors an entity the author never chose.
    """
    wide = _over_binding_view(view)
    scope = DiscoveryScope("object")
    signal = next(
        signal for signal in signal_selections(wide, scope)
        if {m.exact_key for m in signal.mappings} == {"xpos_0", "xpos_1"}
    )
    forms = selector_palette(wide, scope, signal=signal)
    globbed = [
        form for form in forms
        if form.binding_kind == WILDCARD
        and form.component_form == "xpos_*"
    ]
    assert globbed, [f.reference for f in forms]
    for form in globbed:
        assert not form.available, form.reference
        assert "different entity/key mapping" in form.unavailable_reason

    # The explicit list names exactly the selection, so it survives the
    # same check the glob failed.
    listed = [
        form for form in forms
        if form.component_kind == LIST and form.binding_kind == WILDCARD
    ]
    assert listed
    for form in listed:
        assert form.available, form.unavailable_reason
        assert set(form.keys) == {"xpos_0", "xpos_1"}


def test_mapping_check_ignores_entities_outside_the_scope(view):
    """A role-narrowed scope resolves class-wide; that is not a mismatch.

    ``object_*`` is checked against every object in the class, so the
    resolved pairs legitimately include entities the scope excluded.
    Comparing the raw pair set would fail every shape for a reason that
    has nothing to do with the shape.
    """
    entities = []
    seen_object = False
    for entity in view.entities:
        if entity.entity_class != "object":
            entities.append(entity)
            continue
        entities.append(replace(
            entity, role="distractor" if seen_object else "target"
        ))
        seen_object = True
    roled = replace(view, entities=tuple(entities))
    scope = DiscoveryScope("object", role="target")
    signals = signal_selections(roled, scope)
    assert signals
    for signal in signals:
        forms = selector_palette(roled, scope, signal=signal)
        wildcard = [
            form for form in forms
            if form.binding_kind == WILDCARD
            and form.component_kind in (EXACT, GLOB)
        ]
        assert wildcard
        assert any(form.available for form in wildcard), [
            form.unavailable_reason for form in wildcard
        ]


def test_cross_class_scope_has_no_reference(view):
    """A reference names one class; a role spanning classes names none."""
    assert selector_palette(view, DiscoveryScope(None, role=None)) == ()


def test_roleless_scope_reports_the_grammar_limit(view):
    """The grammar filters *for* a label, never *against* every label.

    The limit used to be reported on a second row carrying the same
    reference text as the class-wide form -- two rows, identical text,
    opposite verdicts. That is wrong on screen and it corrupted every
    cache keyed by reference. The limit is now a note on the one row,
    which is where a reader looking at that text needs it.
    """
    forms = selector_palette(
        view, DiscoveryScope("object", role=ROLE_ABSENT)
    )
    assert len({f.reference for f in forms}) == len(forms)
    explained = [
        f for f in forms
        if "no declared role" in (f.note or "")
        or "no declared role" in (f.unavailable_reason or "")
    ]
    assert explained


def test_declared_role_without_its_property_tag_is_refused(view):
    """A role on the entity is not the ``role:*`` tag the filter reads."""
    entities = tuple(
        replace(entity, role="target")
        if entity.entity_class == "object" else entity
        for entity in view.entities
    )
    tagged_view = replace(view, entities=entities)
    forms = selector_palette(
        tagged_view, DiscoveryScope("object", role="target")
    )
    role_forms = [f for f in forms if f.component_kind == ROLE]
    assert role_forms
    for form in role_forms:
        assert not form.available
        assert "role:* property tag is absent" in form.unavailable_reason


def test_declared_role_with_its_property_tag_resolves(view):
    entities = []
    for entity in view.entities:
        if entity.entity_class != "object":
            entities.append(entity)
            continue
        properties = tuple(
            replace(prop, tags=prop.tags + ("role:target",))
            for prop in entity.properties
        )
        entities.append(
            replace(entity, role="target", properties=properties)
        )
    tagged_view = replace(view, entities=tuple(entities))
    forms = selector_palette(
        tagged_view, DiscoveryScope("object", role="target")
    )
    # A role that IS expressible resolves as the tag form that expresses
    # it. The palette used to emit a second, identically-referenced row
    # of kind ROLE alongside it; one row per reference now, so the
    # assertion is on the text a person would paste rather than on which
    # internal shape produced it.
    role_forms = [
        f for f in forms
        if "@ role:target" in f.component_form and f.available
    ]
    assert role_forms
    assert len({f.reference for f in forms}) == len(forms)
    for form in role_forms:
        assert form.entity_ids


def test_label_cap_reports_what_it_withheld(view):
    """A cap that hides its own existence reads as a complete list."""
    entities = []
    for entity in view.entities:
        if entity.entity_class != "object":
            entities.append(entity)
            continue
        properties = tuple(
            replace(
                prop,
                tags=prop.tags + tuple(
                    "synthetic_tag_{}".format(index)
                    for index in range(LABEL_CAP + 5)
                ),
            )
            for prop in entity.properties
        )
        entities.append(replace(entity, properties=properties))
    wide = replace(view, entities=tuple(entities))
    forms = selector_palette(wide, DiscoveryScope("object"))
    labelled = [form for form in forms if " @ " in form.component_form]
    # Cap applies per entity token, and only the wildcard token carries
    # the label vocabulary.
    assert len(labelled) <= LABEL_CAP
    notes = " ".join(form.note or "" for form in labelled)
    assert "further label" in notes


def test_list_literal_stays_readable(view):
    """A generated ``{a, b, c}`` names a set a reader can check."""
    forms = _all_forms(view)
    for form in forms:
        if not form.component_form.startswith("{"):
            continue
        names = form.component_form.strip("{}").split(",")
        assert 2 <= len(names) <= 8, form.reference


def test_difference_never_generates_an_empty_exclusion(view):
    """An exclusion that removes nothing is a compile error upstream."""
    for form in _all_forms(view):
        if "\\" not in form.component_form:
            continue
        assert form.available, form.unavailable_reason
        excluded = form.component_form.split("\\", 1)[1]
        assert excluded.strip().strip("{}").strip()


def test_empty_scope_yields_nothing(view):
    assert selector_palette(view, DiscoveryScope("no_such_class")) == ()


def test_bindings_keep_the_entity_to_key_relation(view):
    """Two parallel lists could not say which key sits on which entity.

    That pairing is the whole point: it is what shows a class-wide
    reference resolving on one entity out of three.
    """
    for form in _all_forms(view):
        for entity_id, key in form.bindings:
            assert isinstance(entity_id, int)
            assert isinstance(key, str)
        assert form.entity_ids == tuple(sorted(
            {entity_id for entity_id, _key in form.bindings}
        ))
        assert form.keys == tuple(sorted(
            {key for _entity, key in form.bindings}
        ))


def test_quantified_forms_state_their_placement_constraint(view):
    """`Max(class_*[all: k])` is rejected; `class_*[all: k]` compiles.

    The constraint is invisible in the reference text, so a row that
    does not carry it hands over a string that fails in the one place
    an author is most likely to put it.
    """
    quantified = [
        form for form in _all_forms(view)
        if form.quantifier is not None and form.available
    ]
    assert quantified
    for form in quantified:
        assert "whole comparison side" in (form.note or ""), form.reference


def _catalog_for(view):
    from specola.components import ComponentCatalog

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    metadata = property_metadata_for(adapter, entities)
    return (
        ComponentCatalog.from_entities(entities, metadata=metadata),
        adapter.dt,
        getattr(adapter, "capabilities", None),
    )


def test_every_offered_form_survives_real_compilation(view):
    """Bracket resolution is not the whole of what the compiler checks.

    verify_forms compiles each reference in a minimal formula of its own
    shape. Nothing the palette offers may fail that, or the palette is
    handing out text that does not work.
    """
    catalog, dt, capabilities = _catalog_for(view)
    checked = 0
    for entity_class in sorted({
        entity.entity_class for entity in view.entities
    }):
        scope = DiscoveryScope(entity_class)
        for signal in (None,) + signal_selections(view, scope):
            forms = selector_palette(view, scope, signal=signal)
            verified = verify_forms(
                forms, catalog=catalog, dt=dt, capabilities=capabilities
            )
            assert len(verified) == len(forms)
            for before, after in zip(forms, verified):
                assert after.reference == before.reference
                if before.available:
                    assert after.available, (
                        after.reference, after.unavailable_reason
                    )
                    checked += 1
                else:
                    # A resolution-level refusal is more specific than
                    # "did not compile" and must not be overwritten.
                    assert after.unavailable_reason == (
                        before.unavailable_reason
                    )
    assert checked > 20


def test_probe_never_wraps_a_form_in_an_aggregator():
    """The probe shape is load-bearing and was chosen by measurement.

    A bare selector is legal at any expansion size -- it means one
    monitor per matched key -- while an aggregator over the same
    selector makes the compiler build one symbolic expression with a
    term per key. Measured on RoboCasa's object class, the wide
    ``object_*[*]`` form (741 keys) compiles bare in 0.276 s and does
    not finish inside 25 s wrapped in ``Max``. A quantified selector
    additionally must be the whole comparison side, so an aggregator
    would reject it outright.
    """
    from specola.selector_palette import _probe_formula

    def form(reference, *, quantifier=None, partner=None):
        return SelectorForm(
            reference=reference,
            binding_kind=WILDCARD,
            component_kind=QUANTIFIED if quantifier else EXACT,
            entity_form="object_*",
            component_form="mass",
            bindings=((1, "mass"),),
            monitors=1,
            quantifier=quantifier,
            partner_reference=partner,
        )

    for candidate in (
        form("object_*[mass]"),
        form("object_*[all: mass]", quantifier="all"),
        form("object_*[*]"),
        form("object{A}_*[mass]", partner="object{B}_*[mass]"),
    ):
        probe = _probe_formula(candidate)
        for aggregator in ("Max(", "Min(", "sum(", "max(", "min("):
            assert aggregator not in probe, probe
        assert candidate.reference in probe


def test_probes_of_real_forms_compile(view):
    """The probe must be legal for every shape the palette emits."""
    from specola.selector_palette import _probe_formula
    from specola.stl_parser import compile_specs

    catalog, dt, capabilities = _catalog_for(view)
    for form in _all_forms(view):
        if not form.available:
            continue
        compile_specs(
            _probe_formula(form),
            dt=dt,
            capabilities=capabilities,
            constants={},
            symmetric=bool(form.partner_reference),
            catalog=catalog,
            backend="auto",
        )


def test_a_two_slot_starter_is_actually_order_independent():
    """``symmetric`` is not a hint; it changes which bindings exist.

    ``bindings/enumerate.py`` emits one sorted representative per
    unordered pair when a formula is declared symmetric
    (``itertools.combinations``), instead of both orderings. Under
    ``a - b >= threshold`` that silently discards half the comparisons:
    swapping the slots flips the sign, so exactly one arbitrary ordering
    of each pair is ever evaluated and the other's violations are never
    reported.

    ``abs`` makes the expression genuinely order-independent, which is
    what the flag asserts, and keeps the halved monitor count that is
    the reason to declare it at all.
    """
    from specola.selector_palette import starter_spec_for_form

    two_slot = SelectorForm(
        reference="object{A}_*[mass]",
        binding_kind=WILDCARD,
        component_kind=EXACT,
        entity_form="object{A}_*",
        component_form="mass",
        bindings=((1, "mass"),),
        monitors=1,
        partner_reference="object{B}_*[mass]",
    )
    starter = starter_spec_for_form(two_slot)
    assert starter["symmetric"] is True
    assert "abs(" in starter["formula"]

    # And falsifiable at the placeholder threshold. ``abs(a - b) >= 0``
    # holds for every binding, so an author who ran the starter
    # unmodified would get a monitor that cannot fail and no sign that
    # the threshold still needed setting.
    assert starter["constants"] == {"threshold": 0.0}
    assert _numeric(starter["formula"], a=1.0, b=2.0) > 0.0
    assert "<=" in starter["formula"]

    # The property the flag asserts, checked rather than assumed: the
    # expression's value does not depend on which slot is which.
    body = starter["formula"]
    swapped = (body
               .replace("object{A}", "\x00")
               .replace("object{B}", "object{A}")
               .replace("\x00", "object{B}"))
    assert _numeric(body, a=1.0, b=2.0) == _numeric(swapped, a=1.0, b=2.0)

    one_slot = SelectorForm(
        reference="object_*[mass]",
        binding_kind=WILDCARD,
        component_kind=EXACT,
        entity_form="object_*",
        component_form="mass",
        bindings=((1, "mass"),),
        monitors=1,
    )
    assert "symmetric" not in starter_spec_for_form(one_slot)


def _numeric(formula: str, *, a: float, b: float) -> float:
    """Evaluate the left-hand side of a starter for two slot values."""
    inner = formula[formula.index("(") + 1:formula.rindex(")")]
    expression = re.split(r"<=|>=", inner)[0].strip()
    expression = (expression
                  .replace("object{A}_*[mass]", repr(a))
                  .replace("object{B}_*[mass]", repr(b)))
    return eval(expression, {"abs": abs, "__builtins__": {}})  # noqa: S307


def test_parse_selector_is_memoized_and_still_pure():
    """Desugaring re-parses a bracket once per expansion.

    Profiling a `fixture_*[*]` palette on RoboCasa showed 168,268
    `parse_selector` calls for one selector, with `_top_level_indices`
    rescanning the same string 336,536 times underneath. The function is
    a pure map from text to frozen dataclasses and is their only
    producer, so it is memoized.

    The win is real where texts repeat (`robot` 0.269 s -> 0.073 s) and
    absent where they cannot: `_reference_from_match` substitutes a
    different exact key per expansion, so a wide class's brackets are
    unique by construction and the cache misses every time. That is why
    `fixture` stayed at ~2.2 s and why the remaining fix is to not
    materialize, not to cache harder.
    """
    from specola.spec.selectors import parse_selector

    first = parse_selector("qacc_* @ joint acceleration")
    assert parse_selector("qacc_* @ joint acceleration") is first

    # Distinct text is distinct, and equal text is equal by value too.
    assert parse_selector("mass") is not first
    assert parse_selector("mass").raw == "mass"

    # A malformed selector still raises every time -- lru_cache does not
    # cache exceptions, so nothing is memoized that should not be.
    for _ in range(2):
        with pytest.raises(ValueError):
            parse_selector("")


def test_validation_routes_the_way_production_routes(view):
    """A trivial reference must not be checked by the wrong path.

    ``compile_formulas`` short-circuits when ``_needs_desugaring`` is
    false: such a formula never reaches ``_expand_formula``, it goes
    straight to ``compile_formula`` -- the only caller of
    ``_validate_slot_against_catalog`` and so the only place five
    refusals are raised (a component outside a declared CLOSED schema,
    no entity jointly exposing the required components, a pinned id
    absent, a pinned class mismatch, a pinned entity missing
    components).

    Sending those to expansion instead checked strictly less than
    production does, on the palette's most common shape: ``object_*[k]``
    is trivial and ``default_form`` returns the first available row.
    """
    from specola.selector_palette import _probe_formula
    from specola.spec.parser import parse
    from specola.spec.semantic import _needs_desugaring

    catalog, dt, capabilities = _catalog_for(view)
    trivial = [
        form for form in _all_forms(view)
        if form.available
        and not _needs_desugaring(parse(_probe_formula(form)))
    ]
    assert trivial, "no trivial form in the palette to check the routing on"

    # Every trivial form's verdict must equal what compile_specs says,
    # because that is the path production uses for it.
    from specola.selector_palette import _validate_expansion
    from specola.stl_parser import compile_specs

    for form in trivial:
        refusal, _ = _validate_expansion(
            form, catalog=catalog, dt=dt, capabilities=capabilities
        )
        try:
            compile_specs(
                _probe_formula(form), dt=dt, capabilities=capabilities,
                constants={}, symmetric=bool(form.partner_reference),
                catalog=catalog, backend="auto",
            )
            compiled_ok = True
        except Exception:                           # noqa: BLE001
            compiled_ok = False
        assert (refusal is None) == compiled_ok, form.reference


def test_one_row_per_reference(view):
    """Two rows with the same text cannot both be right.

    The role candidate is built after the per-token dedup and lands on
    text an earlier shape already produced: on a roleless scope every
    class emitted its class-wide reference twice, once available and
    once refused. That is wrong on screen, and it corrupts any cache
    keyed by reference -- the terminal writer's was, so whichever row
    was focused first poisoned the other.
    """
    from specola.discovery_selector import ROLE_ABSENT

    for entity_class in sorted({e.entity_class for e in view.entities}):
        for scope in (
            DiscoveryScope(entity_class),
            DiscoveryScope(entity_class, role=ROLE_ABSENT),
        ):
            forms = selector_palette(view, scope)
            references = [form.reference for form in forms]
            assert len(set(references)) == len(references), (
                entity_class, scope.role,
            )


def test_an_unverified_form_is_distinguishable_from_a_checked_one(view):
    """Otherwise "not checked" and "checked and fine" look identical."""
    forms = selector_palette(view, DiscoveryScope("object"))
    assert all(not form.verified for form in forms)

    catalog, dt, capabilities = _catalog_for(view)
    checked = verify_forms(
        forms, catalog=catalog, dt=dt, capabilities=capabilities
    )
    assert all(form.verified for form in checked if form.available)


def test_the_catalog_covers_every_view_the_palette_is_built_from(view):
    """Everything else here rests on this, so it is checked rather than
    assumed.

    ``form.bindings`` and the probe's component text both come from keys
    resolved against a *view*, while the compiler resolves against a
    *catalog*. If a view could hold a key the catalog does not, the
    palette would quietly build references the compiler has never heard
    of and every count on the form would be wrong with no error
    anywhere.

    It holds by construction: ``catalog_from_model`` is built from the
    whole ``model.entities``, and ``filter_discovery`` only ever drops
    entities and keeps a subset of each one's properties -- it never
    synthesizes. This pins that, so a future filter that invents a
    property fails here instead of silently.
    """
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_restore import catalog_from_model

    model = view.model
    catalog = catalog_from_model(model)
    queries = [
        DiscoveryQuery(include_raw=False),
        DiscoveryQuery(include_raw=True),
        DiscoveryQuery(include_raw=True, classes=("object",)),
        DiscoveryQuery(include_raw=True, signals=("mass",)),
        DiscoveryQuery(include_raw=True, search=("object",)),
    ]
    for query in queries:
        filtered = filter_discovery(model, query)
        for entity in filtered.entities:
            supported = set(
                catalog.components_for_entity(entity.entity_id) or ()
            )
            keys = {prop.key for prop in entity.properties}
            assert keys <= supported, (query, entity.entity_id,
                                       sorted(keys - supported)[:3])
            in_class = set(
                catalog.components_for_class(entity.entity_class) or ()
            )
            assert keys <= in_class, (query, entity.entity_class)


def test_expansion_counts_are_derivable_from_bindings(view):
    """The count is free; the cost is the validation, not the counting.

    Measured on RoboCasa's raw view, per form, ``_expand_formula``'s
    expansion count equals a closed form over ``form.bindings``:

    * single-slot, unquantified, needing desugaring (glob, every, list,
      group, tag): ``len(bindings)`` -- ``geom_*[geom_*]`` 23,775 = 23,775
    * a *trivial* reference: exactly 1, because its wildcard survives to
      runtime -- ``object_*[mass]`` is one formula and two bindings
    * quantified: the number of distinct entities -- ``geom_*[all: *]``
      1,585 against 23,775 bindings
    * ``**`` fan-out: exactly 1
    * two-slot: ``(sum_e k_e)^2 - sum_e k_e^2``, already computed by
      ``_slot_pairing`` without enumerating the cross product

    So materializing expansions to *count* them is unnecessary. It is
    still what validates them, and the per-form cost after that is
    0.16-0.46 s on the widest classes, cached per scope to 5-8 ms in the
    browser. Avoiding the materialization would mean a planning prefix
    inside ``spec/semantic.py``, which is not worth cutting a cached
    one-time cost -- this test exists so that if it ever is, the
    arithmetic is already proven rather than guessed.
    """
    catalog, dt, capabilities = _catalog_for(view)
    for form in _all_forms(view):
        if not form.available or form.partner_reference:
            continue
        _refusal, counted = _validate_expansion(
            form, catalog=catalog, dt=dt, capabilities=capabilities
        )
        if counted is None:
            continue
        from specola.selector_palette import _probe_formula
        from specola.spec.parser import parse
        from specola.spec.semantic import _needs_desugaring

        if not _needs_desugaring(parse(_probe_formula(form))):
            # A trivial reference is one formula whose wildcard survives
            # to runtime: `object_*[mass]` is 1 expansion and 2 bindings.
            # Expansions are compile-time formulas, bindings are the
            # monitors they later produce, and conflating them is how a
            # count ends up describing the wrong thing.
            expected = 1
        elif form.binding_kind == FANOUT:
            expected = 1
        elif form.component_kind == QUANTIFIED:
            expected = len({entity for entity, _key in form.bindings})
        else:
            expected = len(form.bindings)
        assert counted == expected, (form.reference, form.component_kind)


def test_verification_actually_refuses_something(view):
    """A validator that always said "fine" must not pass this file.

    Every other test here runs on the mock, where all 30 forms across
    every class are legal -- so `assert all(form.verified)` and
    `assert form.available` are both satisfied by a `_validate_expansion`
    that returns `(None, 1)` unconditionally. That is the recorded
    mistake ("the mock is not a test scene for anything user-facing") one
    layer up: the test written to close the gap could not detect it.

    A CLOSED component vocabulary is what makes the two answers differ,
    and it is a property of the adapter's capabilities, not of the scene
    size -- so it can be constructed here without a simulator. RoboCasa
    declares `contact` and `robot_contact` CLOSED; this declares `object`
    CLOSED over a component the entities do not carry, which is the same
    shape as the ten `contact` refusals.
    """
    from specola.capabilities import AdapterCapabilities, VocabularyMode

    catalog, dt, capabilities = _catalog_for(view)
    # The mock declares none, which is the point: without a CLOSED
    # vocabulary the two answers never differ, so the mock alone cannot
    # tell a working validator from a broken one.
    assert capabilities is None

    # The shape RoboCasa's ten refusals actually take: a class whose
    # component schema is declared CLOSED, carrying no declared groups,
    # against a form that filters on one. `robot` is the mock class that
    # has group forms. The group filter cannot be
    # resolved from the reset catalog for such a class and cannot be
    # resolved from the declaration either, so it fails compilation --
    # while the bracket itself resolves perfectly well.
    keys = frozenset(
        key for form in selector_palette(view, DiscoveryScope("robot"))
        for _entity, key in form.bindings
    )
    narrowed = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"object", "robot", "gripper", "region"}),
        component_mode_by_class={"robot": VocabularyMode.CLOSED},
        components_by_class={"robot": keys},
    )
    forms = selector_palette(view, DiscoveryScope("robot"))
    grouped = [f for f in forms if f.component_kind in {GROUP, TAG}]
    assert grouped, "no group/tag form on this scene to check"

    permissive = verify_forms(
        grouped, catalog=catalog, dt=dt, capabilities=capabilities
    )
    strict = verify_forms(
        grouped, catalog=catalog, dt=dt, capabilities=narrowed
    )
    assert any(form.available for form in permissive)
    refused = [form for form in strict if not form.available]
    assert refused, "declaring a CLOSED vocabulary refused nothing"
    assert all("compile" in (f.unavailable_reason or "") for f in refused)
