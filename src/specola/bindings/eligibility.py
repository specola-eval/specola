"""
eligibility.py — per-slot binding candidate sets.

For every parameter slot, independently:

    candidates[slot] = { e | e.entity_class == slot.entity_class
                           and selector matches (pinned ID, if any)
                           and track policy allows e
                           and e supports slot.required_components }

Two same-class slots may have different candidate sets: an object
exposing only ``hinge_angle`` can fill a slot requiring it while an
object exposing only ``mass`` fills the other. Class-level component
unions are never used for eligibility.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from ..components import ComponentResolver


def passes_tag_filters(
    parameter: Any,
    entity_id: int,
    component_tags: Optional[
        Mapping[int, Mapping[str, Tuple[str, ...]]]
    ],
) -> bool:
    """Whether an entity satisfies a slot's retained ``@`` tag filters.

    Filters exist only on declared-schema dynamic classes, where
    compilation validated every label against the class's declared tag
    vocabulary — so matching here is against observation-time tags
    alone, never group labels. An entity with no tag entry for a
    filtered component is ineligible, not an error: the filter simply
    does not match it.
    """
    filters = getattr(parameter, "component_tag_filters", None)
    if not filters:
        return True
    entity_tags = (component_tags or {}).get(entity_id, {})
    for component, labels in filters.items():
        present = entity_tags.get(component, ())
        if any(label not in present for label in labels):
            return False
    return True


def slot_candidates(
    formula: Any,
    entities: Mapping[int, Any],
    require_confirmed: bool,
    resolver: ComponentResolver,
    component_tags: Optional[
        Mapping[int, Mapping[str, Tuple[str, ...]]]
    ] = None,
) -> Dict[Any, FrozenSet[int]]:
    """Return {param_key: eligible entity IDs} for one observation.

    ``component_tags`` is the observation's per-entity structural tag
    mapping (``MonitorObservation.component_tags``); it is consulted
    only for slots carrying retained ``@`` tag filters.
    """
    candidates: Dict[Any, FrozenSet[int]] = {}

    for parameter in formula.params:
        eligible = set()

        for entity_id, entity in entities.items():
            if entity.entity_class != parameter.entity_class:
                continue
            if require_confirmed and not entity.is_confirmed:
                continue
            if (
                parameter.pinned_entity_id is not None
                and entity_id != parameter.pinned_entity_id
            ):
                continue
            if not resolver.supports_all(
                entity_id, parameter.required_components
            ):
                continue
            if not passes_tag_filters(
                parameter, entity_id, component_tags
            ):
                continue
            eligible.add(entity_id)

        candidates[parameter.param_key] = frozenset(eligible)

    return candidates
