"""Scoped signal selections and conservative selector previews."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from os.path import commonprefix
from typing import Dict, Iterable, Optional, Tuple, Union

from .discovery_filter import DiscoveryView
from .discovery_model import DiscoveryEntity, DiscoveryProperty


class DiscoveryRoleScope(Enum):
    ABSENT = "absent"


ROLE_ABSENT = DiscoveryRoleScope.ABSENT


@dataclass(frozen=True)
class DiscoveryScope:
    """One role, class, class-role intersection, or concrete entity."""

    entity_class: Optional[str]
    role: Union[str, DiscoveryRoleScope, None] = None
    entity_id: Optional[int] = None

    @property
    def label(self) -> str:
        if self.entity_id is not None:
            return "{}_{}".format(self.entity_class, self.entity_id)
        role_label = (
            "(no declared role)"
            if self.role is ROLE_ABSENT else self.role
        )
        if self.entity_class is not None and role_label is not None:
            return "{} ∩ {}".format(self.entity_class, role_label)
        if self.entity_class is not None:
            return self.entity_class
        return str(role_label or "all entities")


@dataclass(frozen=True)
class SignalMapping:
    entity_id: int
    entity_class: str
    label: Optional[str]
    exact_key: str
    value: float
    unit: Optional[str]


@dataclass(frozen=True)
class SignalSelection:
    pattern: str
    quantity: Optional[str]
    component: Optional[str]
    unit: Optional[str]
    frame: Optional[str]
    mappings: Tuple[SignalMapping, ...]


@dataclass(frozen=True)
class SelectorPreview:
    selector: Optional[str]
    binding_count: int
    explanation: str
    example: Optional[str] = None


def entities_for_scope(
    view: DiscoveryView,
    scope: DiscoveryScope,
) -> Tuple[DiscoveryEntity, ...]:
    return tuple(
        entity
        for entity in view.entities
        if (
            scope.entity_class is None
            or entity.entity_class == scope.entity_class
        )
        and (
            scope.role is None
            or (
                scope.role is ROLE_ABSENT
                and entity.role is None
            )
            or entity.role == scope.role
        )
        and (scope.entity_id is None or entity.entity_id == scope.entity_id)
    )


def _glob_for_keys(keys: Iterable[str]) -> str:
    unique = tuple(sorted(set(keys)))
    if len(unique) == 1:
        return unique[0]
    prefix = commonprefix(unique)
    suffix = commonprefix(tuple(value[::-1] for value in unique))[::-1]
    while prefix and not prefix.endswith("_"):
        prefix = prefix[:-1]
    while suffix and not suffix.startswith("_"):
        suffix = suffix[1:]
    if len(prefix) + len(suffix) >= min(len(value) for value in unique):
        return "*"
    return "{}*{}".format(prefix, suffix)


def signal_selections(
    view: DiscoveryView,
    scope: DiscoveryScope,
) -> Tuple[SignalSelection, ...]:
    """Aggregate entity-owned exact signals inside the selected scope."""
    scoped = entities_for_scope(view, scope)
    grouped: Dict[Tuple[object, ...], list] = {}
    for entity in scoped:
        for prop in entity.properties:
            signature = (
                entity.entity_class,
                prop.group,
                prop.unit,
                prop.component,
                prop.frame,
                # Undeclared families stay exact. Equal exact keys such as
                # ``mass`` still aggregate across entities without inferring
                # any semantics from their spelling.
                prop.family or prop.key,
            )
            if scope.entity_id is not None:
                signature += (prop.key,)
            grouped.setdefault(signature, []).append((entity, prop))

    selections = []
    for signature, pairs in grouped.items():
        pattern = _glob_for_keys(prop.key for _entity, prop in pairs)
        mappings = tuple(
            SignalMapping(
                entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                label=entity.label,
                exact_key=prop.key,
                value=prop.value,
                unit=prop.unit,
            )
            for entity, prop in sorted(
                pairs, key=lambda pair: (pair[0].entity_id, pair[1].key)
            )
        )
        selections.append(SignalSelection(
            pattern=pattern,
            quantity=signature[1],
            unit=signature[2],
            component=signature[3],
            frame=signature[4],
            mappings=mappings,
        ))
    return tuple(sorted(
        selections,
        key=lambda selection: (
            selection.quantity or "",
            selection.pattern,
        ),
    ))


def _resolved_pairs(
    entities: Tuple[DiscoveryEntity, ...],
    selector_text: str,
) -> Tuple[Tuple[int, str], ...]:
    """Resolve with the production selector parser/resolver."""
    from .entities import SpecConfigError
    from .spec.selectors import parse_selector, resolve_selector

    try:
        selector = parse_selector(selector_text)
    except (SpecConfigError, ValueError):
        return ()
    pairs = []
    for entity in entities:
        components = frozenset(prop.key for prop in entity.properties)
        groups = {
            prop.key: prop.group
            for prop in entity.properties
            if prop.group is not None
        }
        tags = {
            prop.key: prop.tags
            for prop in entity.properties
            if prop.tags
        }
        try:
            pairs.extend(
                (entity.entity_id, key)
                for key in resolve_selector(
                    selector, components, groups, tags
                )
            )
        except SpecConfigError:
            continue
    return tuple(sorted(pairs))


def selector_for_signal(
    view: DiscoveryView,
    scope: DiscoveryScope,
    signal: SignalSelection,
) -> SelectorPreview:
    scoped = entities_for_scope(view, scope)
    if not scoped or not signal.mappings:
        return SelectorPreview(None, 0, "No reset-time binding matches.")
    mapping_classes = {
        mapping.entity_class for mapping in signal.mappings
    }
    if len(mapping_classes) != 1:
        return SelectorPreview(
            None,
            0,
            "Selection spans multiple classes; select one class.",
        )
    entity_class = next(iter(mapping_classes))
    scoped = tuple(
        entity for entity in scoped
        if entity.entity_class == entity_class
    )
    # Class-wide references must resolve against every entity of the
    # class, not only the scoped subset: a roleless scope on a class with
    # roled members would otherwise hide the over-binding it produces.
    class_entities = tuple(
        entity for entity in view.entities
        if entity.entity_class == entity_class
    )

    if scope.entity_id is not None:
        if len(signal.mappings) != 1:
            return SelectorPreview(
                None,
                len(signal.mappings),
                "Selected signal expands to multiple exact keys.",
            )
        bracket = signal.mappings[0].exact_key
        reference = "{}_{}[{}]".format(
            entity_class, scope.entity_id, bracket
        )
    else:
        bracket = signal.pattern
        if isinstance(scope.role, str):
            role_tag = "role:" + scope.role
            if any(
                role_tag not in next(
                    prop.tags
                    for prop in entity.properties
                    if prop.key == mapping.exact_key
                )
                for entity in scoped
                for mapping in signal.mappings
                if mapping.entity_id == entity.entity_id
            ):
                return SelectorPreview(
                    None,
                    0,
                    "Role is declared but its role:* property tag is absent.",
                )
            bracket += " @ " + role_tag
        reference = "{}_*[{}]".format(entity_class, bracket)

    resolution_population = (
        scoped if scope.entity_id is not None else class_entities
    )
    expected_pairs = tuple(sorted(
        (mapping.entity_id, mapping.exact_key)
        for mapping in signal.mappings
    ))
    resolved_pairs = _resolved_pairs(resolution_population, bracket)
    if resolved_pairs != expected_pairs and scope.entity_id is None:
        exact_keys = sorted({
            mapping.exact_key for mapping in signal.mappings
        })
        atom = (
            exact_keys[0]
            if len(exact_keys) == 1
            else "{" + ", ".join(exact_keys) + "}"
        )
        bracket = atom
        if isinstance(scope.role, str):
            bracket += " @ role:" + scope.role
        reference = "{}_*[{}]".format(entity_class, bracket)
        resolved_pairs = _resolved_pairs(resolution_population, bracket)
    binding_count = len(resolved_pairs)
    if resolved_pairs != expected_pairs:
        if scope.role is ROLE_ABSENT:
            return SelectorPreview(
                None,
                binding_count,
                "The selector grammar cannot express only entities with "
                "no declared role.",
            )
        return SelectorPreview(
            None,
            binding_count,
            "No exact selector preserves the selected entity/key mapping.",
        )
    if binding_count == 0:
        return SelectorPreview(
            None,
            0,
            "Generated reference resolves to no catalogue binding.",
        )
    example = (
        "name: selected_signal_requirement\n"
        "formula: \"historically[0,1s]({} >= threshold)\"\n"
        "constants: {{threshold: 0.0}}"
    ).format(reference)
    return SelectorPreview(
        reference,
        binding_count,
        "{} reset-time binding(s).".format(binding_count),
        example,
    )


def first_selector_preview(view: DiscoveryView) -> SelectorPreview:
    """Return the first production-verified concrete authoring example."""
    for entity in view.entities:
        if entity.raw:
            continue
        scope = DiscoveryScope(
            entity_class=entity.entity_class,
            entity_id=entity.entity_id,
        )
        for signal in signal_selections(view, scope):
            preview = selector_for_signal(view, scope, signal)
            if preview.selector is not None:
                return preview
    return SelectorPreview(
        None,
        0,
        "No production-valid reset-time selector is available.",
    )


def scope_for_view(
    view: DiscoveryView,
) -> Tuple[Optional[DiscoveryScope], Optional[str]]:
    """The scope a filtered view implies, or why it implies none.

    A CLI filter is how the non-interactive surfaces say what the tree
    click says in the explorer, so the two must agree on what a
    selection means. Exactly one entity reads as a pinned scope; one
    class reads as that class, narrowed by role when the survivors share
    one; anything spanning classes has no single reference, because a
    reference names exactly one class.

    Returns ``(scope, None)`` or ``(None, reason)`` — never both, and
    the reason is user-facing text.
    """
    if not view.entities:
        return None, "No reset-time binding matches."
    if len(view.entities) == 1:
        entity = view.entities[0]
        return DiscoveryScope(
            entity_class=entity.entity_class,
            entity_id=entity.entity_id,
        ), None
    classes = {entity.entity_class for entity in view.entities}
    if len(classes) != 1:
        return None, (
            "Selection spans multiple classes; narrow it for a selector."
        )
    roles = {entity.role for entity in view.entities}
    # A filter that happens to leave only roleless entities selected the
    # class, not "the ones without a role". Inferring ROLE_ABSENT here
    # would answer a question nobody asked -- and, since the grammar
    # cannot express rolelessness, would attach a refusal to an ordinary
    # ``--class object``. Explicit rolelessness still exists: the
    # explorer's "(no declared role)" node sets it directly, and there
    # the refusal is the honest answer.
    role = (
        next(iter(roles))
        if len(roles) == 1 and None not in roles
        else None
    )
    return DiscoveryScope(next(iter(classes)), role=role), None


def selector_preview(view: DiscoveryView) -> SelectorPreview:
    """Compatibility helper for one filtered CLI detail view."""
    scope, reason = scope_for_view(view)
    if scope is None:
        return SelectorPreview(
            None,
            0 if not view.entities else sum(
                len(entity.properties) for entity in view.entities
            ),
            reason or "No reset-time binding matches.",
        )
    signals = signal_selections(view, scope)
    if len(signals) != 1:
        return SelectorPreview(
            None,
            sum(len(entity.properties) for entity in view.entities),
            "Selection spans multiple signals; narrow it for a selector.",
        )
    return selector_for_signal(view, scope, signals[0])
