"""
Delta-based incremental binding maintenance (design stage 5).

The incremental structure must produce exactly the complete-enumeration
result while doing delta work: full enumeration only at bootstrap,
incremental updates otherwise, with observable counters.
"""

from __future__ import annotations

import pytest

from specola.bindings import enumerate_complete
from specola.bindings.incremental import IncrementalBindingSet
from specola.monitor import ParameterSpec, SpecTemplate, make_binding


def formula_two_slots():
    return SpecTemplate(
        params=[
            ParameterSpec(("object", 1), True, ["hinge"]),
            ParameterSpec(("object", 2), True, ["mass"]),
        ],
        predicate_fn=lambda b, r: 0.0,
    )


A = ("object", 1)
B = ("object", 2)


def test_bootstrap_uses_full_enumeration_once():
    bindings = IncrementalBindingSet(formula_two_slots())

    active = bindings.update({A: frozenset({1}), B: frozenset({2})})

    assert active == {(1, 2)}
    assert bindings.n_full == 1
    assert bindings.n_incremental == 0


def test_addition_is_handled_incrementally():
    bindings = IncrementalBindingSet(formula_two_slots())
    bindings.update({A: frozenset({1}), B: frozenset({2})})

    active = bindings.update({A: frozenset({1, 3}), B: frozenset({2, 3})})

    assert active == {(1, 2), (1, 3), (3, 2)}
    assert bindings.n_full == 1
    assert bindings.n_incremental == 1


def test_removal_deactivates_only_affected_bindings():
    bindings = IncrementalBindingSet(formula_two_slots())
    bindings.update({A: frozenset({1, 3}), B: frozenset({2, 3})})

    active = bindings.update({A: frozenset({1}), B: frozenset({2, 3})})

    assert active == {(1, 2), (1, 3)}
    assert bindings.n_incremental == 1


def test_unchanged_candidates_are_a_no_op():
    bindings = IncrementalBindingSet(formula_two_slots())
    candidates = {A: frozenset({1}), B: frozenset({2})}
    first = bindings.update(candidates)

    second = bindings.update(dict(candidates))

    assert second == first
    assert bindings.n_full == 1
    assert bindings.n_incremental == 0


def test_simultaneous_add_and_remove():
    bindings = IncrementalBindingSet(formula_two_slots())
    bindings.update({A: frozenset({1}), B: frozenset({2})})

    active = bindings.update({A: frozenset({3}), B: frozenset({2})})

    assert active == {(3, 2)}


def test_matches_reference_over_random_walk():
    import random

    rng = random.Random(11)
    formula = formula_two_slots()
    bindings = IncrementalBindingSet(formula)

    for _ in range(200):
        candidates = {
            A: frozenset(
                eid for eid in range(5) if rng.random() < 0.5
            ),
            B: frozenset(
                eid for eid in range(5) if rng.random() < 0.5
            ),
        }
        active = bindings.update(candidates)
        assert active == enumerate_complete(formula, candidates)

    assert bindings.n_incremental > 0


def test_manager_exposes_incremental_counters():
    from types import SimpleNamespace

    import numpy as np

    from specola.entities import EntityState, TrackState
    from specola.monitor import IncrementalBindingManager

    manager = IncrementalBindingManager(formula_two_slots())

    def observation(t, ids_props):
        entities = {
            eid: EntityState(
                entity_id=eid,
                entity_class="object",
                position=np.zeros(3),
                velocity=np.zeros(3),
                track_age=1,
                track_state=TrackState.CONFIRMED,
                properties=props,
            )
            for eid, props in ids_props.items()
        }
        return SimpleNamespace(
            timestep=t, time=float(t), perceived_entities=entities
        )

    manager.update(observation(0, {1: {"hinge": 1.0}, 2: {"mass": 1.0}}))
    manager.update(observation(1, {1: {"hinge": 1.0}, 2: {"mass": 1.0}}))
    manager.update(
        observation(
            2,
            {
                1: {"hinge": 1.0},
                2: {"mass": 1.0},
                3: {"hinge": 1.0, "mass": 1.0},
            },
        )
    )

    assert manager.binding_set.n_full == 1
    assert manager.binding_set.n_incremental == 1  # step 1 was a no-op


# ---------------------------------------------------------------------------
# Contract enforcement at the public-API boundary (reviewer findings)
# ---------------------------------------------------------------------------

def test_update_rejects_wrong_slot_key_set():
    """Omitting a now-empty slot must fail loudly, not silently corrupt
    the active set."""
    bindings = IncrementalBindingSet(formula_two_slots())
    bindings.update({A: frozenset({1, 2}), B: frozenset({1, 2})})

    with pytest.raises(ValueError, match="parameter slot"):
        bindings.update({A: frozenset({1, 2})})


def test_update_rejects_unequal_candidates_for_interchangeable_slots():
    """Symmetric interchangeable slots have identical candidate sets by
    construction (identical requirements); raw maps violating that
    precondition must be rejected, not silently mis-canonicalized."""
    formula = SpecTemplate(
        params=[
            ParameterSpec(("object", 1), True, ["m"]),
            ParameterSpec(("object", 2), True, ["m"]),
        ],
        predicate_fn=lambda b, r: 0.0,
        symmetric=True,
    )
    bindings = IncrementalBindingSet(formula)

    with pytest.raises(ValueError, match="interchangeable"):
        bindings.update(
            {
                ("object", 1): frozenset({1, 2, 3}),
                ("object", 2): frozenset({1, 2}),
            }
        )


def test_parameterless_template_through_incremental_set():
    formula = SpecTemplate(
        params=[], predicate_fn=lambda b, r: 0.0
    )
    bindings = IncrementalBindingSet(formula)

    first = bindings.update({})
    second = bindings.update({})

    assert first == {()}
    assert second == first
    assert bindings.n_full == 1
    assert bindings.n_incremental == 0
