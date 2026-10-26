"""
incremental.py — delta-based maintenance of the active binding set.

``IncrementalBindingSet`` maintains, across observations, the set of
bindings that complete enumeration would produce, updating it from
candidate-map deltas instead of recomputing the cross-class product.

Specification
-------------
The structure is specified by the invariant (design §5.7):

    after update(candidates),
    active == enumerate_complete(formula, candidates)

Property-based tests validate the implementation over generated
observation sequences (tests/test_binding_parity.py,
tests/test_incremental_bindings.py); the argument below is why the
invariant holds. Tests validate; they do not prove.

Proof sketch
------------
Write Θ(C) for enumerate_complete(formula, C). Membership of a binding
in Θ(C) is: (i) class consistency and same-class injectivity — properties
of the binding alone — and (ii) every (slot, entity) pair of the binding
lies in C[slot]. Let C_old be the previous map (inductively,
active == Θ(C_old)), C_new the new one, and per slot s:

    added[s]   = C_new[s] − C_old[s]
    removed[s] = C_old[s] − C_new[s]
    stable[s]  = C_old[s] ∩ C_new[s] = C_new[s] − added[s]

Deactivation removes exactly the active bindings using a removed pair:

    active − deactivated
        = { b ∈ Θ(C_old) | every pair of b in C_new }
        = Θ(C_old) ∩ Θ(C_new)            (by (i)+(ii))

Activation adds exactly the bindings of Θ(C_new) using ≥1 added pair.
A binding of Θ(C_new) using no added pair draws every pair from stable
⊆ C_old and hence lies in Θ(C_old); one using an added pair cannot lie
in Θ(C_old). Therefore:

    activated = Θ(C_new) − Θ(C_old)

and the new active set is
(Θ(C_old) ∩ Θ(C_new)) ∪ (Θ(C_new) − Θ(C_old)) = Θ(C_new).  ∎

Activation is enumerated per class without the full cross-class product:
a binding uses an added pair iff some class's assignment does, so we sum
disjointly over the *first* churn class whose assignment uses an added
pair — classes before it contribute stable-only assignments, it
contributes assignments(new) − assignments(stable), classes after it
contribute assignments(new).

Symmetric canonicalization commutes with the delta because
interchangeability is a *static* property of the formula: it requires
identical required_components across the interchangeable slots, and
identical requirements imply identical candidate sets in every
observation (eligibility is a function of class, track state, and
supported components only) — so new and stable assignments canonicalize
consistently.

Full enumeration happens exactly once, at bootstrap; ``n_full`` and
``n_incremental`` counters make the behavior observable.
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from ..monitor_types import Binding, ParamKey, make_binding
from .enumerate import _class_assignments, enumerate_complete


_Pair = Tuple[ParamKey, int]


class IncrementalBindingSet:
    """Active binding set maintained from per-slot candidate deltas."""

    def __init__(self, formula) -> None:
        self.formula = formula
        self._prev: Optional[Dict[ParamKey, FrozenSet[int]]] = None
        self._active: Set[Binding] = set()
        self._index: Dict[_Pair, Set[Binding]] = {}

        self.n_full: int = 0
        self.n_incremental: int = 0

    def update(
        self,
        candidates: Mapping[ParamKey, FrozenSet[int]],
    ) -> Set[Binding]:
        """Bring the active set to Θ(candidates); return a copy of it.

        The map must contain exactly the formula's parameter slots
        (an empty slot is an empty set, never an absent key), and for a
        symmetric formula the interchangeable slots of a class must have
        identical candidate sets — both hold by construction for maps
        produced by ``bindings.eligibility.slot_candidates`` and are
        enforced here so raw callers fail loudly instead of silently
        breaking the invariant.
        """
        snapshot = {
            key: frozenset(value) for key, value in candidates.items()
        }

        self._validate_candidate_map(snapshot)

        if self._prev is not None and snapshot == self._prev:
            return set(self._active)

        if self._prev is None:
            self._rebuild(snapshot)
        else:
            self._apply_delta(snapshot)

        self._prev = snapshot
        return set(self._active)

    # ------------------------------------------------------------------

    def _validate_candidate_map(
        self,
        snapshot: Dict[ParamKey, FrozenSet[int]],
    ) -> None:
        expected_keys = {p.param_key for p in self.formula.params}
        if set(snapshot) != expected_keys:
            missing = sorted(expected_keys - set(snapshot))
            unexpected = sorted(set(snapshot) - expected_keys)
            raise ValueError(
                f"Candidate map must contain exactly the formula's "
                f"parameter slot keys (an empty slot is an empty set, "
                f"never an absent key). Missing: {missing}; "
                f"unexpected: {unexpected}."
            )

        if not self.formula.symmetric:
            return

        for class_params in self.formula.params_by_class.values():
            wilds = [p for p in class_params if p.is_wildcard]
            if len(wilds) < 2:
                continue
            if any(
                p.required_components != wilds[0].required_components
                for p in wilds[1:]
            ):
                continue
            first = snapshot[wilds[0].param_key]
            if any(
                snapshot[p.param_key] != first for p in wilds[1:]
            ):
                raise ValueError(
                    f"Symmetric interchangeable slots of class "
                    f"{wilds[0].entity_class!r} must have identical "
                    f"candidate sets (identical requirements imply this "
                    f"for maps built by slot_candidates)."
                )

    def _rebuild(
        self,
        candidates: Dict[ParamKey, FrozenSet[int]],
    ) -> None:
        self.n_full += 1
        self._active = set(
            enumerate_complete(self.formula, candidates)
        )
        self._index = {}
        for binding in self._active:
            self._register(binding)

    def _pairs(self, binding: Binding):
        for parameter, entity_id in zip(self.formula.params, binding):
            yield (parameter.param_key, entity_id)

    def _register(self, binding: Binding) -> None:
        for pair in self._pairs(binding):
            self._index.setdefault(pair, set()).add(binding)

    def _deregister(self, binding: Binding) -> None:
        for pair in self._pairs(binding):
            entries = self._index.get(pair)
            if entries is None:
                continue
            entries.discard(binding)
            if not entries:
                del self._index[pair]

    def _apply_delta(
        self,
        candidates: Dict[ParamKey, FrozenSet[int]],
    ) -> None:
        self.n_incremental += 1

        added: Dict[ParamKey, FrozenSet[int]] = {}
        removed: Dict[ParamKey, FrozenSet[int]] = {}

        for key, new_ids in candidates.items():
            old_ids = self._prev.get(key, frozenset())
            gained = new_ids - old_ids
            lost = old_ids - new_ids
            if gained:
                added[key] = gained
            if lost:
                removed[key] = lost

        # Deactivate: active bindings using any removed (slot, entity) pair.
        for key, ids in removed.items():
            for entity_id in ids:
                for binding in list(
                    self._index.get((key, entity_id), ())
                ):
                    self._active.discard(binding)
                    self._deregister(binding)

        if not added:
            return

        for binding in self._bindings_with_added(candidates, added):
            if binding not in self._active:
                self._active.add(binding)
                self._register(binding)

    def _bindings_with_added(
        self,
        candidates: Dict[ParamKey, FrozenSet[int]],
        added: Dict[ParamKey, FrozenSet[int]],
    ) -> Set[Binding]:
        """Θ(candidates) members using at least one added pair."""
        stable = {
            key: candidates[key] - added.get(key, frozenset())
            for key in candidates
        }

        classes = list(self.formula.params_by_class)
        churn_classes = [
            cls
            for cls in classes
            if any(
                added.get(p.param_key)
                for p in self.formula.params_by_class[cls]
            )
        ]
        if not churn_classes:
            return set()

        def frozen(assignments) -> Set[FrozenSet]:
            return {frozenset(a.items()) for a in assignments}

        assignments_new: Dict[str, Set[FrozenSet]] = {}
        assignments_stable: Dict[str, Set[FrozenSet]] = {}

        for cls in classes:
            params = self.formula.params_by_class[cls]
            assignments_new[cls] = frozen(
                _class_assignments(
                    params, candidates, self.formula.symmetric
                )
            )
            if cls in churn_classes:
                assignments_stable[cls] = frozen(
                    _class_assignments(
                        params, stable, self.formula.symmetric
                    )
                )

        result: Set[Binding] = set()

        # Disjoint decomposition on the first churn class whose
        # assignment uses an added pair (see module docstring).
        for position, churn_class in enumerate(churn_classes):
            added_assignments = (
                assignments_new[churn_class]
                - assignments_stable[churn_class]
            )
            if not added_assignments:
                continue

            earlier = set(churn_classes[:position])
            parts: List[Set[FrozenSet]] = []
            feasible = True

            for cls in classes:
                if cls == churn_class:
                    part = added_assignments
                elif cls in earlier:
                    part = assignments_stable[cls]
                else:
                    part = assignments_new[cls]
                if not part:
                    feasible = False
                    break
                parts.append(part)

            if not feasible:
                continue

            for combo in itertools.product(*parts):
                merged: Dict[ParamKey, int] = {}
                for class_assignment in combo:
                    merged.update(dict(class_assignment))
                result.add(make_binding(merged, self.formula.params))

        return result
