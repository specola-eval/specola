"""
check_specs.py — specola check CLI: explain a spec file against a live sim
========================================================================
The researcher-facing safety net the selector design leans on: given a
specs YAML and a simulator adapter, print exactly what each authored
formula compiles to — selector expansion, parameter slots and their
eligible entities, every referenced component key with its unit and
current value, and a plain-English ALL/ANY reading for each n-ary
``min(``/``max(`` comparison — without running an episode.

Usage
-----
    specola check --sim mock specs.yaml
    specola check --sim robocasa --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml specs.yaml

Design notes
------------
- Spec parsing reuses ``MonitorEngine`` (engine.py) so the YAML format
  never drifts from ``specola run``'s. ``MonitorEngine.load`` compiles
  every entry at once and aborts on the first failure (``start_episode``'s
  all-or-nothing contract), which is wrong for a tool whose job is to
  report every entry's status — so this module calls the public
  ``MonitorEngine.iter_spec_entries`` instead, which parses the same YAML
  shape but returns one ``(name, entry, error)`` triple per entry instead
  of raising on the first bad one. ``compile_specs`` (stl_parser.py) is
  then called per entry, here, catching failures independently — this
  part still cannot go through ``start_episode()``, since that call
  aborts the whole episode on the first spec that fails to compile.

- "wildcard / pinned / expanded" slot origin: ``ParameterSpec`` records
  only ``is_wildcard`` and ``pinned_entity_id`` — it does not tag *how* a
  pinned slot came to be pinned. This module infers it per parameter: a
  wildcard parameter is "wildcard"; a pinned parameter is "pinned" when
  its own ``class_<id>[`` token literally appears in the *authored*
  formula text (the author typed that entity ID); otherwise "expanded" —
  the slot is pinned only because ``compile_specs`` expanded a selector
  or a "**" binding against the discovered catalog and happened to land
  on this one entity. Checking per parameter (not per formula) matters
  for compound specs: a formula can mix an author-pinned slot with a
  selector-expanded one, and only the latter should read "expanded".
  "expanded" is a deliberately honest umbrella — it does not claim
  "**" specifically, since a component-selector (``all:``/``any:``) that
  happens to resolve its wildcard entity binding down to one concrete
  entity produces the exact same author-didn't-name-it shape as a "**"
  fan-out.

- Quantifier readings and mixed-unit warnings both operate on the
  *final* comparison text: ``formula.desugared_text`` when selector
  expansion rewrote it, otherwise the entry's own authored formula text
  (already fully concrete when no selector/`**` was used). That text is
  re-parsed with ``spec.parser.parse`` purely to walk ``ast.Comparison``
  nodes — ``.left``/``.right``/``.operator``/``.references`` are exactly
  what the parser already extracts, so no bespoke text scanning is
  needed beyond spotting a whole-side ``min(``/``max(``/``Min(``/``Max(``
  call.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import textwrap
from ast import literal_eval as _literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

import yaml

from .adapter import PropertyMeta
from .bindings.eligibility import passes_tag_filters
from .capabilities import reset_catalog
from .cli_common import amber, bold, cyan, dim, green, load_sim, red
from .components import ComponentCatalog
from .engine import MonitorEngine
from .entities import EntityState
from .monitor import ParameterSpec, SpecTemplate
from .pager import maybe_page
from .spec import ast as _ast
from .spec import grammar as _grammar
from .spec import parser as _spec_parser
from .spec.selectors import _glob_pattern
from .spec.semantic import (
    _mirror_operator,
    _split_top_level_commas,
    resolve_reference_head,
)
from .stl_parser import compile_specs


# ---------------------------------------------------------------------------
# Quantifier-reading direction map: (aggregator, effective operator) -> who
# must satisfy the per-component bound the aggregator is compared against.
# ---------------------------------------------------------------------------

_DIRECTION = {
    ("max", "<="): "ALL", ("max", "<"): "ALL",
    ("min", ">="): "ALL", ("min", ">"): "ALL",
    ("min", "<="): "ANY", ("min", "<"): "ANY",
    ("max", ">="): "ANY", ("max", ">"): "ANY",
}

_CALL_RE = re.compile(r"^(min|max)\((.*)\)$", re.IGNORECASE)
_LEGACY_GF_RE = re.compile(
    r"(?<![A-Za-z0-9_])([GF])\s*\[",
    re.IGNORECASE,
)


def _uses_legacy_gf(formula_text: str) -> bool:
    """Whether authored text uses Specola's legacy G/F past aliases."""
    return _LEGACY_GF_RE.search(formula_text) is not None


def _iter_comparisons(node: _ast.FormulaNode) -> List[_ast.Comparison]:
    """Every ``Comparison`` leaf under a formula AST, in source order."""
    if isinstance(node, _ast.Comparison):
        return [node]
    if isinstance(
        node,
        (
            _ast.Not,
            _ast.Always,
            _ast.Eventually,
            _ast.Previous,
            _ast.Rise,
            _ast.Fall,
        ),
    ):
        return _iter_comparisons(node.child)
    if isinstance(
        node,
        (_ast.And, _ast.Or, _ast.Implies, _ast.Iff, _ast.Xor, _ast.Since),
    ):
        return _iter_comparisons(node.left) + _iter_comparisons(node.right)
    raise TypeError(f"Unknown formula node type: {type(node)!r}")


def _whole_reduction(text: str) -> Optional[Tuple[str, List[str]]]:
    """Return a whole-side min/max call and its top-level arguments."""
    stripped = text.strip()
    match = _CALL_RE.match(stripped)
    if match is None:
        return None

    open_index = len(match.group(1))
    try:
        close_index = _grammar.find_matching_parenthesis(
            stripped, open_index
        )
    except ValueError:
        return None
    if close_index != len(stripped) - 1:
        return None

    inner = stripped[open_index + 1:close_index]
    args = [
        inner[start:end].strip()
        for start, end in _split_top_level_commas(inner)
    ]
    return match.group(1).lower(), args


def _flatten_equal_reductions(
    func: str,
    args: List[str],
) -> List[str]:
    """Flatten nested calls of the same reduction for presentation."""
    flattened: List[str] = []
    for argument in args:
        nested = _whole_reduction(argument)
        if nested is not None and nested[0] == func:
            flattened.extend(
                _flatten_equal_reductions(func, nested[1])
            )
        else:
            flattened.append(argument)
    return flattened


def _reduction_extent(args: List[str]) -> str:
    """Describe term, concrete-entity, and per-entity component counts."""
    per_entity: Dict[Tuple[str, int], Set[str]] = {}
    for argument in args:
        for match in _grammar.PARAM_RE.finditer(argument):
            binding = match.group("binding")
            if not binding.isdigit():
                continue
            entity = (match.group("head"), int(binding))
            per_entity.setdefault(entity, set()).add(
                match.group("component").strip()
            )

    term_count = len(args)
    term_word = "term" if term_count == 1 else "terms"
    if not per_entity:
        return f"{term_count} {term_word}"

    component_counts = {
        entity: len(components)
        for entity, components in per_entity.items()
    }
    if len(per_entity) == 1:
        component_count = next(iter(component_counts.values()))
        component_word = (
            "component" if component_count == 1 else "components"
        )
        return (
            f"{term_count} {term_word} using {component_count} "
            f"{component_word}"
        )

    classes = {entity_class for entity_class, _entity_id in per_entity}
    counts = set(component_counts.values())
    if len(classes) == 1 and len(counts) == 1:
        entity_class = next(iter(classes))
        component_count = next(iter(counts))
        component_word = (
            "component" if component_count == 1 else "components"
        )
        return (
            f"{term_count} {term_word} spanning {len(per_entity)} "
            f"{entity_class} entities × {component_count} "
            f"{component_word} per entity"
        )

    ordered_entities = sorted(component_counts)
    details = ", ".join(
        f"{entity_class}_{entity_id}="
        f"{component_counts[(entity_class, entity_id)]}"
        for entity_class, entity_id in ordered_entities
    )
    if len(classes) == 1:
        entity_span = (
            f"{len(per_entity)} {next(iter(classes))} entities"
        )
    else:
        entity_span = f"{len(per_entity)} entities"
    return (
        f"{term_count} {term_word} spanning {entity_span}; "
        f"components per entity: {details}"
    )


def _quantifier_reading(comparison: _ast.Comparison) -> Optional[str]:
    """Plain-English ALL/ANY reading when one whole side is min(/max(...).

    Equal nested reductions are flattened for reporting because
    ``max(max(...))`` and ``min(min(...))`` preserve their grouping.  Terms
    remain separate from component counts: arithmetic may repeat a component,
    and concrete references may span several entities.
    """
    for is_left, side_text, other_text in (
        (True, comparison.left, comparison.right),
        (False, comparison.right, comparison.left),
    ):
        reduction = _whole_reduction(side_text)
        if reduction is None:
            continue
        func, args = reduction
        args = _flatten_equal_reductions(func, args)
        if len(args) < 2:
            continue
        effective = (
            comparison.operator if is_left
            else _mirror_operator(comparison.operator)
        )
        reading = _DIRECTION.get((func, effective))
        if reading is None:
            continue
        return (
            f"{func} over {_reduction_extent(args)} {effective} "
            f"{other_text.strip()} — {reading} must satisfy"
        )
    return None


def _reference_entity_id(
    reference: _ast.EntityReference,
    entity_class: str,
    catalog: ComponentCatalog,
    formula: SpecTemplate,
) -> Optional[int]:
    """A concrete entity ID to read this reference's unit/value from.

    Pinned references (a digit binding) resolve directly — the ID is the
    binding text itself, so this always agrees with the matching
    ``ParameterSpec.pinned_entity_id``. A wildcard resolves to the same
    representative entity the slot table already picked for its
    ``ParameterSpec`` (``_representative_entity``): the lowest-ID entity
    that jointly supports *every* component that slot requires, not just
    the one component this particular reference happens to read. Picking
    independently per component (as the naive version of this function
    once did) can name an entity that supports this key but not some
    other key the same slot needs — an entity that could never actually
    be bound to that slot — which is exactly what would make a mixed-unit
    warning cite a unit that could never occur.
    """
    if reference.binding not in ("*", "**"):
        return int(reference.binding)
    for param in formula.params:
        if (
            param.is_wildcard
            and param.entity_class == entity_class
            and reference.component in param.required_components
        ):
            return _representative_entity(param, catalog)
    return None


def _comparison_units(
    comparison: _ast.Comparison,
    catalog: ComponentCatalog,
    metadata: Dict[int, Dict[str, PropertyMeta]],
    formula: SpecTemplate,
) -> List[str]:
    """Distinct non-None units referenced by one comparison, sorted."""
    units = set()
    for reference in comparison.references:
        try:
            entity_class, _label = resolve_reference_head(
                reference, catalog.classes, strict=True
            )
            entity_id = _reference_entity_id(
                reference, entity_class, catalog, formula
            )
        except Exception:
            continue
        if entity_id is None:
            continue
        meta = metadata.get(entity_id, {}).get(reference.component)
        unit = _declared_unit(meta)
        if unit is not None:
            units.add(unit)
    return sorted(units)


def _declared_unit(meta: Optional[PropertyMeta]) -> Optional[str]:
    """Return the dedicated unit, falling back to legacy note-as-unit."""
    if meta is None:
        return None
    unit = getattr(meta, "unit", None)
    return unit if unit is not None else meta.note


# ---------------------------------------------------------------------------
# Slot / key tables
# ---------------------------------------------------------------------------

def _slot_origin(param: ParameterSpec, authored_text: str) -> str:
    """Per-parameter slot origin — see module docstring for the rationale.

    A wildcard param is always "wildcard". A pinned param is "pinned"
    when the author literally typed its ``class_<id>[`` token — plain,
    explicitly labelled (``class{A}_<id>[``), or legacy-labelled
    (``class_A_<id>[``) — somewhere in the *authored* formula text;
    otherwise the pin only exists because desugaring expanded a selector
    or "**" binding, so "expanded". A negative lookbehind before the
    class name keeps a short class name from false-matching inside a
    longer one that happens to share a suffix (``bot_1[`` must not match
    inside ``robot_1[``); the trailing ``[`` immediately after the ID
    keeps this from false-matching a different, longer entity ID
    (``robot_1[`` is not a substring of ``robot_10[``).

    Converse hole (left as-is deliberately — heuristic honesty, not a
    bug to chase): a slot that is pinned *only* because desugaring
    expanded a selector or "**" binding still reports "pinned" if the
    author's authored text happens to contain that same
    ``class_<id>[`` token elsewhere in the formula — e.g. a different,
    hand-pinned parameter of the same class and ID in another clause.
    This function only checks whether the token occurs anywhere in the
    authored text, not whether *this* parameter is the occurrence that
    produced it, so that case is misreported "pinned" instead of
    "expanded".
    """
    if param.is_wildcard:
        return "wildcard"
    cls = param.entity_class
    entity_id = param.pinned_entity_id
    pattern = (
        rf"(?<![A-Za-z0-9_]){re.escape(cls)}"
        rf"(?:\{{[A-Z]\}}|_[A-Z])?_{entity_id}\["
    )
    return "pinned" if re.search(pattern, authored_text) else "expanded"


def _eligible_entities(
    param: ParameterSpec,
    catalog: ComponentCatalog,
) -> List[int]:
    candidate_ids = (
        catalog.entity_ids_for_class(param.entity_class)
        if param.is_wildcard
        else (
            (param.pinned_entity_id,)
            if param.pinned_entity_id is not None
            else ()
        )
    )
    return sorted(
        entity_id
        for entity_id in candidate_ids
        if catalog.has_entity(entity_id)
        and catalog.entity_info(entity_id).supports_all(
            param.required_components
        )
        and passes_tag_filters(
            param,
            entity_id,
            {
                entity_id: catalog.component_tags_for_entity(entity_id),
            },
        )
    )


def _representative_entity(
    param: ParameterSpec,
    catalog: ComponentCatalog,
) -> Optional[int]:
    if not param.is_wildcard:
        return param.pinned_entity_id
    eligible = _eligible_entities(param, catalog)
    return eligible[0] if eligible else None


def _class_slots_have_injective_assignment(
    params,
    candidates,
) -> bool:
    """Whether one entity class can fill all of its slots injectively."""
    ordered = sorted(
        (param.param_key for param in params),
        key=lambda param_key: (
            len(candidates[param_key]),
            param_key[1],
        ),
    )
    slot_by_entity: Dict[int, Tuple[str, int]] = {}

    def augment(
        param_key: Tuple[str, int],
        seen_entities: Set[int],
    ) -> bool:
        for entity_id in sorted(candidates[param_key]):
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            previous = slot_by_entity.get(entity_id)
            if previous is None or augment(previous, seen_entities):
                slot_by_entity[entity_id] = param_key
                return True
        return False

    return all(augment(param_key, set()) for param_key in ordered)


def _entity_label(
    entity_id: int,
    labels: Optional[Dict[int, str]],
) -> str:
    label = labels.get(entity_id) if labels else None
    return f"{label}[{entity_id}]" if label else str(entity_id)


def _print_slot_table(
    formula: SpecTemplate,
    catalog: ComponentCatalog,
    labels: Optional[Dict[int, str]],
    authored_text: str,
) -> None:
    print(dim("      slots:"))
    for param in formula.params:
        origin = _slot_origin(param, authored_text)
        eligible = _eligible_entities(param, catalog)
        entity_text = (
            ", ".join(_entity_label(e, labels) for e in eligible)
            or "(none)"
        )
        slot_name = f"{param.entity_class}_{param.param_index}"
        retained_filters = " ".join(
            "{} @ {}".format(component, " @ ".join(tags))
            for component, tags in sorted(
                param.component_tag_filters.items()
            )
            if tags
        )
        filter_text = (
            "  component filters: {}".format(retained_filters)
            if retained_filters
            else ""
        )
        print(
            f"        {param.entity_class:<12} {slot_name:<14} "
            f"{origin:<8} {entity_text}{filter_text}"
        )


def _format_value(value: object) -> str:
    try:
        return f"{float(value):+.4g}"
    except (TypeError, ValueError):
        return "—"


def _print_key_table(
    formula: SpecTemplate,
    catalog: ComponentCatalog,
    metadata: Dict[int, Dict[str, PropertyMeta]],
    ents: Dict[int, EntityState],
) -> None:
    rows: List[Tuple[str, str, str, str, Optional[float]]] = []
    seen: set = set()

    # A row shows the representative entity's metadata, but hoisting says
    # "all", so the claim has to hold for every entity the slot binds --
    # two entities may spell one key with different units or groups.
    uniform_across_entities = True

    for param in formula.params:
        representative = _representative_entity(param, catalog)
        eligible = (
            _eligible_entities(param, catalog)
            if param.is_wildcard
            else []
        )
        for key in param.required_components:
            slot_key = (param.param_key, key)
            if slot_key in seen:
                continue
            seen.add(slot_key)

            meta = (
                metadata.get(representative, {}).get(key)
                if representative is not None
                else None
            )
            declared_unit = _declared_unit(meta)
            unit = declared_unit if declared_unit is not None else "—"
            group = meta.group if meta is not None else "—"
            meta_tags = getattr(meta, "tags", ()) if meta is not None else ()
            tag_label = f"[{' '.join(meta_tags)}]" if meta_tags else ""

            entity = ents.get(representative) if representative is not None else None
            value = (
                entity.properties.get(key)
                if entity is not None
                else None
            )

            for other in eligible:
                other_meta = metadata.get(other, {}).get(key)

                if other_meta is None:
                    continue

                other_tags = getattr(other_meta, "tags", ()) or ()
                other_tag_label = (
                    f"[{' '.join(other_tags)}]" if other_tags else ""
                )
                other_declared_unit = _declared_unit(other_meta)
                other_unit = (
                    other_declared_unit
                    if other_declared_unit is not None
                    else "—"
                )

                if (
                    other_meta.group != group
                    or other_unit != unit
                    or other_tag_label != tag_label
                ):
                    uniform_across_entities = False

            slot_name = f"{param.entity_class}_{param.param_index}"
            rows.append((slot_name, key, unit, group, tag_label, value))

    # Whenever a selector filtered on a group, every row repeats the same
    # long label and buries the keys. Group and tags hoist independently:
    # a group filter gives one group across many tags, a tag filter the
    # reverse.
    groups = {row[3] for row in rows}
    tag_labels = {row[4] for row in rows}
    one_group = (
        groups.pop()
        if len(groups) == 1 and uniform_across_entities
        else None
    )
    one_tag = (
        tag_labels.pop()
        if len(tag_labels) == 1 and uniform_across_entities
        else None
    )

    hoisted = []
    if one_group is not None and one_group != "—":
        hoisted.append(f"all in group {one_group}")
    if one_tag:
        hoisted.append(f"tagged {one_tag}")

    header = "      keys:"
    if hoisted:
        header += f"  ({', '.join(hoisted)})"
    print(dim(header))

    for slot_name, key, unit, group, tag_label, value in rows:
        shown = []
        # A uniform group is dropped from the rows whether or not it has a
        # name: "—" on every row says only that the adapter supplied no
        # metadata, which the header already conveys by staying silent.
        if one_group is None:
            shown.append(group)
        if one_tag is None and tag_label:
            shown.append(tag_label)
        cell = " ".join(shown)
        cell = f"{cell:<26} " if cell else ""
        print(
            f"        {slot_name:<14} {key:<28} "
            f"{unit:<8} {cell}{_format_value(value)}"
        )


def _covering_glob(
    key: str,
    entity_ids: List[int],
    catalog: ComponentCatalog,
) -> Optional[str]:
    """A component glob matching a key on every one of ``entity_ids``.

    Candidates replace one contiguous run of the key's ``_``-separated
    tokens with ``*``, shortest run first, so the most literal pattern
    that still spans the class wins — ``xpos_obj_main_2`` yields
    ``xpos_*_main_2`` rather than the weaker ``xpos_*_2``. The
    all-tokens candidate is never generated: a bare ``*`` spans every
    class trivially and names no key in particular. Candidates are valid
    selector atoms by construction, a component name being
    ``[a-z_][a-z0-9_]*`` and the atom grammar that same set plus ``*``.

    None means no candidate matched a key on every entity: the class is
    genuinely heterogeneous rather than merely spelling one property per
    entity, so there is no glob to point the author at.
    """
    tokens = key.split("_")
    key_sets = [
        catalog.components_for_entity(entity_id)
        for entity_id in entity_ids
    ]
    for run in range(1, len(tokens)):
        for start in range(len(tokens) - run + 1):
            candidate = "_".join(
                tokens[:start] + ["*"] + tokens[start + run:]
            )
            pattern = _glob_pattern(candidate)
            if all(
                any(pattern.match(name) for name in names)
                for names in key_sets
            ):
                return candidate
    return None


def _partial_binding_warnings(
    formula: SpecTemplate,
    catalog: ComponentCatalog,
) -> List[str]:
    """One warning per wildcard slot binding a strict subset of its class.

    A wildcard slot reads as "every entity of this class" but binds only
    the entities exposing every key it names, so it silently narrows the
    moment one class member lacks one of those keys — the ordinary case
    on adapters that qualify component keys by body name, where each
    entity's keys are spelled after itself and the class shares none.
    Pinned and expanded slots are skipped: both name a single entity on
    purpose, so binding one is exactly what was asked for.
    """
    warnings: List[str] = []
    for param in formula.params:
        if not param.is_wildcard:
            continue
        class_ids = sorted(
            catalog.entity_ids_for_class(param.entity_class)
        )
        bound = _eligible_entities(param, catalog)
        if len(class_ids) < 2 or len(bound) >= len(class_ids):
            continue

        # The slot's own keys are what narrowed it; the first one no
        # class member shares is the key to name and to build a glob
        # from. Sorted so the same slot reports the same key every run.
        missing = next(
            (
                key
                for key in sorted(param.required_components)
                if not all(
                    catalog.entity_info(entity_id).supports(key)
                    for entity_id in class_ids
                )
            ),
            None,
        )
        if missing is None:
            continue

        detail = f"not all expose {missing}"
        glob = _covering_glob(missing, class_ids, catalog)
        if glob is not None:
            detail += (
                f"; a glob such as {glob} covers all {len(class_ids)}"
            )
        warnings.append(
            f"{param.entity_class}_* bound {len(bound)} of "
            f"{len(class_ids)} {param.entity_class} entities ({detail})"
        )
    return warnings


def _print_readings_and_warnings(
    formula: SpecTemplate,
    fallback_text: str,
    catalog: ComponentCatalog,
    metadata: Dict[int, Dict[str, PropertyMeta]],
) -> None:
    # Slot-level, so reported before the per-comparison walk below — and
    # independent of it, since a formula whose desugared text fails to
    # re-parse still has slots worth warning about.
    for warning in _partial_binding_warnings(formula, catalog):
        print(f"      {amber('warning:')} {warning}")

    text = formula.desugared_text or fallback_text
    try:
        node = _spec_parser.parse(text)
    except ValueError:
        return

    for comparison in _iter_comparisons(node):
        try:
            reading = _quantifier_reading(comparison)
        except Exception:
            reading = None
        if reading:
            print(f"      reading: {reading}")

        try:
            units = _comparison_units(comparison, catalog, metadata, formula)
        except Exception:
            units = []
        if len(units) >= 2:
            print(
                f"      {amber('warning:')} mixed units "
                f"({', '.join(units)})"
            )


# ---------------------------------------------------------------------------
# Error reporting: SpecConfigError verbatim, plus "did you mean" for
# unknown-component failures.
# ---------------------------------------------------------------------------

_WILDCARD_MISSING_RE = re.compile(
    r"No discovered entity of class '([^']*)' exposes all required "
    r"components (\[[^\]]*\])\."
)
_PINNED_MISSING_RE = re.compile(
    r"Pinned entity \d+ \(([^)]*)\) does not expose required components "
    r"(\[[^\]]*\])\."
)


def _print_did_you_mean(
    error_message: str,
    catalog: ComponentCatalog,
) -> None:
    match = _WILDCARD_MISSING_RE.search(error_message)
    if match is None:
        match = _PINNED_MISSING_RE.search(error_message)
    if match is None:
        return

    entity_class, comps_text = match.group(1), match.group(2)
    try:
        required = _literal_eval(comps_text)
    except Exception:
        return

    available = sorted(catalog.components_for_class(entity_class))
    for component in required:
        if component in available:
            continue
        matches = difflib.get_close_matches(
            component, available, n=3, cutoff=0.5
        )
        if matches:
            print(
                f"      did you mean: {', '.join(matches)} "
                f"(for {component!r})"
            )


# ---------------------------------------------------------------------------
# Per-entry reporting
# ---------------------------------------------------------------------------

def _check_entry(
    spec,
    dt: float,
    catalog: ComponentCatalog,
    metadata: Dict[int, Dict[str, PropertyMeta]],
    ents: Dict[int, EntityState],
    labels: Optional[Dict[int, str]],
    capabilities=None,
) -> bool:
    """Explain one spec entry; require compilation and a viable binding."""
    print(f"  {bold(spec.name)}")
    print(f"    formula: {cyan(spec.formula_text)}")
    if _uses_legacy_gf(spec.formula_text):
        print(
            f"    {amber('warning:')} G/F are legacy Specola aliases for "
            "past historically/once; prefer historically[a,b](...) or "
            "once[a,b](...) in new specifications"
        )
    if spec.constants:
        constants_text = yaml.safe_dump(
            dict(spec.constants),
            default_flow_style=True,
            sort_keys=True,
        ).strip()
        print(f"    constants: {cyan(constants_text)}")

    try:
        compiled = compile_specs(
            spec.formula_text,
            dt=dt,
            capabilities=capabilities,
            constants=spec.constants,
            symmetric=spec.symmetric,
            catalog=catalog,
            backend=spec.backend,
        )
    except (ValueError, TypeError) as exc:
        print(f"    {red('error:')} {exc}")
        _print_did_you_mean(str(exc), catalog)
        print()
        return False

    if compiled[0].desugared_text is not None:
        print(dim("    expands to:"))
        for formula in compiled:
            suffix = formula.name[len(spec.formula_text):]
            if suffix:
                print(f"      {suffix}: {formula.desugared_text}")
            else:
                print(f"      {formula.desugared_text}")
        print(f"    monitors: {len(compiled)}")

    has_complete_binding = True
    for formula in compiled:
        if len(compiled) > 1:
            suffix = formula.name[len(spec.formula_text):]
            print(f"    monitor{suffix}:")
        _print_slot_table(formula, catalog, labels, spec.formula_text)
        _print_key_table(formula, catalog, metadata, ents)
        _print_readings_and_warnings(
            formula, spec.formula_text, catalog, metadata
        )
        candidates = {
            param.param_key: frozenset(
                _eligible_entities(param, catalog)
            )
            for param in formula.params
        }
        class_assignment_status = {
            entity_class: _class_slots_have_injective_assignment(
                params, candidates
            )
            for entity_class, params in formula.params_by_class.items()
        }
        has_reset_binding = all(class_assignment_status.values())
        print(
            "      reset bindings: {}".format(
                ">=1" if has_reset_binding else "0"
            )
        )
        if not has_reset_binding:
            dynamic_classes = set(
                getattr(
                    capabilities,
                    "runtime_populated_classes",
                    frozenset(),
                )
                or ()
            )
            unsatisfied_classes = [
                entity_class
                for entity_class, satisfied
                in class_assignment_status.items()
                if not satisfied
            ]
            unsatisfied_static_classes = [
                entity_class
                for entity_class in unsatisfied_classes
                if entity_class not in dynamic_classes
            ]
            if unsatisfied_static_classes:
                has_complete_binding = False
                print(
                    "      {}  no complete reset-time binding; static "
                    "class slot(s) cannot form an injective assignment: {}"
                    .format(
                        red("error:"),
                        ", ".join(
                            sorted(unsatisfied_static_classes)
                        ),
                    )
                )
            else:
                print(
                    "      {}  no reset-time binding; only declared "
                    "runtime-populated class slot(s) are unsatisfied "
                    "({}), so the monitor compiles and waits".format(
                        amber("warning:"),
                        ", ".join(sorted(unsatisfied_classes)),
                    )
                )
        for entity_id in _unresolved_pinned_ids(formula, catalog):
            # An open entity-ID vocabulary (dynamic populations) makes
            # an absent pin compile-and-wait rather than an error; a
            # silent wait on a typoed ID would look like a monitor that
            # never fires, so name it here.
            print(
                "    {}  pinned entity ID {} is not in the discovered "
                "catalog; the monitor compiles and waits for it to "
                "appear at runtime".format(amber("⚠"), entity_id)
            )

    print()
    return has_complete_binding


def _unresolved_pinned_ids(formula, catalog: ComponentCatalog):
    """Pinned entity IDs the reset catalog does not contain."""
    unresolved = []
    for param in getattr(formula, "params", ()) or ():
        pinned = getattr(param, "pinned_entity_id", None)
        if pinned is not None and not catalog.has_entity(int(pinned)):
            unresolved.append(int(pinned))
    return unresolved


def _read_shown_entities(adapter, entries, dt, catalog, ents, capabilities) -> None:
    """Read the unread catalog entities whose values the report prints.

    The catalog-only reset leaves the MuJoCo family's geom pairs unread
    (issue #136), but the key table prints the value of each slot's
    representative entity. This reads just those, through one scoped
    observation before any step, and adds them to ``ents``; their
    metadata is already in the catalog's. Entries that fail to compile
    are reported later.
    """
    unread = set(catalog.entities) - set(ents)
    selectors: Dict[str, Set[Tuple[int, tuple]]] = {}
    for _name, spec, error in entries:
        if error is not None or not unread:
            continue
        try:
            compiled = compile_specs(
                spec.formula_text, dt=dt, capabilities=capabilities,
                constants=spec.constants, symmetric=spec.symmetric,
                catalog=catalog, backend=spec.backend,
            )
        except (ValueError, TypeError):
            continue
        for formula in compiled:
            for param in formula.params:
                entity_id = _representative_entity(param, catalog)
                if entity_id is not None and entity_id in unread:
                    selectors.setdefault(param.entity_class, set()).add((entity_id, ()))
    if not selectors:
        return
    adapter.set_extraction_entity_scope(selectors)
    observation = adapter.get_observation()
    for entity_id, state in observation.perceived_entities.items():
        if entity_id in unread:
            ents[entity_id] = state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from .conformance.verification import require_ordinary_execution
    require_ordinary_execution()
    ap = argparse.ArgumentParser(
        prog="specola check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Explain a spec file against a live simulator adapter — "
            "selector expansion, parameter slots, referenced component "
            "keys, and quantifier readings — without running an episode."
        ),
        epilog=textwrap.dedent(
            """
            specola check --sim mock specs.yaml
            specola check --sim robocasa --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml specs.yaml
            """
        ),
    )
    ap.add_argument("specs")
    ap.add_argument("--sim", default="mock")
    ap.add_argument("--queries", type=Path, default=None, metavar="PATH",
                    help="Native query configuration YAML or JSON, separate from specs")
    ap.add_argument("--adapter-module", action="append", default=[],
                    metavar="MODULE",
                    help="Import MODULE so its adapter registration runs "
                         "(repeatable); use with --sim <registered-name>")
    ap.add_argument("--task", default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--all-regions", action="store_true",
                    help="Discover regions from every fixture, not just "
                         "the task's target regions (adapters without "
                         "the distinction ignore this)")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--no-pager", action="store_true",
                    help="Print directly instead of paging long output")
    args = ap.parse_args()

    from .registry import load_adapter_module
    for module_name in args.adapter_module:
        load_adapter_module(module_name)

    if not Path(args.specs).exists():
        print(red(f"  Not found: {args.specs}"))
        sys.exit(1)

    from .query_config import load_query_configuration
    queries = load_query_configuration(args.queries) if args.queries is not None else None
    from .conformance.provider import ExtractionGuardError
    from .run_failures import report_extraction_failure
    try:
        adapter, dt, _dims, ents, _classes = load_sim(
            args.sim, task=args.task, robot=args.robot,
            target_only=not args.all_regions, dt_override=args.dt,
            header="Specola Spec Check", queries=queries, catalog_only=True,
        )
    except ExtractionGuardError as exc:
        report_extraction_failure(exc, command="check", adapter=args.sim, task=args.task)
        raise SystemExit(1) from None

    labels = getattr(adapter, "entity_labels", None)

    meta_fn = getattr(adapter, "property_metadata", None)
    metadata: Dict[int, Dict[str, PropertyMeta]] = {}
    if callable(meta_fn):
        for entity_id in ents:
            try:
                entity_meta = meta_fn(entity_id) or {}
            except Exception:
                entity_meta = {}
            if entity_meta:
                metadata[entity_id] = entity_meta

    # The catalog-only reset read no geom pair; the catalog and the
    # metadata still list every one (issue #136).
    catalog, extended = reset_catalog(adapter, ents, metadata)
    metadata = cast(Dict[int, Dict[str, PropertyMeta]], extended)

    # Reuse MonitorEngine's YAML/formula-syntax loader so the spec format
    # never drifts from specola run's, via the public per-entry API:
    # MonitorEngine.load()'s all-or-nothing compile (start_episode's
    # contract) is wrong for a tool whose whole job is reporting every
    # entry's outcome, so this calls iter_spec_entries() instead — see
    # module docstring. OSError (e.g. `specs` naming a directory) is
    # caught alongside the YAML/shape ValueError so a bad path prints a
    # clean error instead of a raw traceback.
    try:
        entries = MonitorEngine.iter_spec_entries(args.specs)
    except (ValueError, OSError) as exc:
        print(red(f"  Spec load error:\n  {exc}"))
        sys.exit(1)

    capabilities = getattr(adapter, "capabilities", None)
    try:
        _read_shown_entities(adapter, entries, dt, catalog, ents, capabilities)
    except ExtractionGuardError as exc:
        report_extraction_failure(exc, command="check", adapter=args.sim, task=args.task)
        raise SystemExit(1) from None

    all_ok = True

    with maybe_page(disable=args.no_pager):
        print()
        print(bold(
            f"  Checking {len(entries)} spec(s) from {args.specs} "
            f"against --sim {args.sim}"
        ))
        print()

        # `@ label` matches a group OR a tag, so a name serving as both
        # resolves to the union of two axes -- and the two differ in what
        # they promise about units. Adapter-level, so it is reported once.
        ambiguous = catalog.ambiguous_filter_labels()
        if ambiguous:
            print(
                f"  {amber('warning:')} this adapter uses "
                f"{', '.join(repr(name) for name in ambiguous)} as both a "
                f"group and a tag; '@ <name>' matches either, so such a "
                f"filter selects the union"
            )
            print()

        for name, spec, error in entries:
            if error is not None:
                print(f"  {bold(name)}")
                print(f"    {red('error:')} {error}")
                print()
                all_ok = False
                continue
            ok = _check_entry(
                spec,
                dt,
                catalog,
                metadata,
                ents,
                labels,
                capabilities=capabilities,
            )
            all_ok = all_ok and ok

        if all_ok:
            print(f"  {green('✓')} all specs compile and bind or wait\n")
        else:
            print(
                f"  {red('✗')} one or more specs failed compilation "
                "or binding validation\n"
            )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
