"""
enumerate.py — complete binding enumeration over per-slot candidates.

The reference implementation for binding sets (design §5.7): sound
(every emitted binding satisfies selector, class, per-slot component,
track-policy, and same-class injectivity constraints), complete (every
satisfying assignment is emitted), and unique (each exactly once).

A formula with zero entity parameters yields exactly one empty binding
(decision 4 in the design spec).

Symmetry: when the formula is declared symmetric and every wildcard
slot of a class has identical requirements and identical candidates,
one canonical (sorted) representative is emitted per equivalence class.
When candidates differ, permuting slots is not an invariance, so full
injective enumeration is used.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, FrozenSet, List, Mapping, Set

from ..monitor_types import Binding, ParamKey, make_binding


def enumerate_complete(
    formula: Any,
    candidates: Mapping[ParamKey, FrozenSet[int]],
) -> Set[Binding]:
    """Compute Θ_φ(t): all valid complete bindings for one observation."""
    if not formula.params:
        return {()}  # the empty slot-aligned binding

    class_assignment_lists: List[List[Dict[ParamKey, int]]] = []

    for entity_class, class_params in formula.params_by_class.items():
        assignments = _class_assignments(
            class_params,
            candidates,
            symmetric=formula.symmetric,
        )
        if not assignments:
            return set()
        class_assignment_lists.append(assignments)

    result: Set[Binding] = set()
    for combo in itertools.product(*class_assignment_lists):
        merged: Dict[ParamKey, int] = {}
        for class_assignment in combo:
            merged.update(class_assignment)
        result.add(make_binding(merged, formula.params))

    return result


def _class_assignments(
    class_params: List[Any],
    candidates: Mapping[ParamKey, FrozenSet[int]],
    symmetric: bool,
) -> List[Dict[ParamKey, int]]:
    """All injective assignments for the slots of one entity class."""
    pinned = [p for p in class_params if not p.is_wildcard]
    wilds = [p for p in class_params if p.is_wildcard]

    assignment_base: Dict[ParamKey, int] = {}
    used: Set[int] = set()

    for parameter in pinned:
        slot_candidates = candidates.get(parameter.param_key, frozenset())
        if parameter.pinned_entity_id not in slot_candidates:
            return []
        # Injectivity is enforced here as well as at semantic compilation,
        # so hand-constructed formulas cannot produce unsound bindings.
        if parameter.pinned_entity_id in used:
            return []
        assignment_base[parameter.param_key] = parameter.pinned_entity_id
        used.add(parameter.pinned_entity_id)

    if not wilds:
        return [assignment_base]

    wild_candidates = [
        candidates.get(p.param_key, frozenset()) - used for p in wilds
    ]

    interchangeable = (
        symmetric
        and len(wilds) >= 2
        and all(
            p.required_components == wilds[0].required_components
            for p in wilds[1:]
        )
        and all(c == wild_candidates[0] for c in wild_candidates[1:])
    )

    if interchangeable:
        # Interchangeable slots with identical candidates: emit one
        # canonical sorted representative per unordered choice.
        assignments = []
        for combo in itertools.combinations(
            sorted(wild_candidates[0]), len(wilds)
        ):
            assignment = dict(assignment_base)
            for parameter, entity_id in zip(wilds, combo):
                assignment[parameter.param_key] = entity_id
            assignments.append(assignment)
        return assignments

    assignments: List[Dict[ParamKey, int]] = []

    def backtrack(index: int, assignment: Dict[ParamKey, int]) -> None:
        if index == len(wilds):
            assignments.append(dict(assignment))
            return
        parameter = wilds[index]
        for entity_id in sorted(wild_candidates[index]):
            if entity_id in assignment_used:
                continue
            assignment_used.add(entity_id)
            assignment[parameter.param_key] = entity_id
            backtrack(index + 1, assignment)
            del assignment[parameter.param_key]
            assignment_used.discard(entity_id)

    assignment_used: Set[int] = set()
    backtrack(0, dict(assignment_base))

    return assignments
