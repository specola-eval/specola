"""
Slot-specific complete binding enumeration (design stage 4).

Binding soundness: every active binding satisfies selector, class,
per-slot component, track-policy, and same-class injectivity
constraints. Completeness: every assignment satisfying them appears.
Eligibility is evaluated per parameter slot — never by combining
requirements across an entity class.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from specola.entities import EntityState, TrackState
from specola.monitor import (
    IncrementalBindingManager,
    binding_label,
    binding_to_dict,
    enumerate_bindings,
    make_binding,
)
from specola.stl_parser import parse_stl


def entity(eid, cls="object", props=None, confirmed=True):
    return EntityState(
        entity_id=eid,
        entity_class=cls,
        position=np.zeros(3),
        velocity=np.zeros(3),
        track_age=5,
        track_state=(
            TrackState.CONFIRMED if confirmed else TrackState.TENTATIVE
        ),
        properties=props or {},
    )


def obs(entities, t=0):
    return SimpleNamespace(
        timestep=t,
        time=float(t),
        perceived_entities={e.entity_id: e for e in entities},
    )


# ---------------------------------------------------------------------------
# Per-slot eligibility (design-spec defect 1)
# ---------------------------------------------------------------------------

def test_heterogeneous_same_class_entities_bind_slot_specifically():
    formula = parse_stl(
        "object{A}_*[hinge] >= 0 and object{B}_*[mass] >= 0",
        dt=1.0,
        valid_classes=["object"],
    )
    hinge_only = entity(1, props={"hinge": 0.5})
    mass_only = entity(2, props={"mass": 1.0})

    bindings = enumerate_bindings(formula, obs([hinge_only, mass_only]))

    assert len(bindings) == 1
    assignment = binding_to_dict(next(iter(bindings)), formula.params)
    assert assignment[("object", 1)] == 1  # slot A ← hinge entity
    assert assignment[("object", 2)] == 2  # slot B ← mass entity


def test_slot_candidates_are_not_class_unions():
    formula = parse_stl(
        "object{A}_*[hinge] >= 0 and object{B}_*[mass] >= 0",
        dt=1.0,
        valid_classes=["object"],
    )
    both = entity(1, props={"hinge": 0.1, "mass": 0.1})
    hinge_only = entity(2, props={"hinge": 0.2})

    bindings = enumerate_bindings(formula, obs([both, hinge_only]))

    # A∈{1,2}, B∈{1}, injective ⇒ exactly (A=2, B=1)
    assert len(bindings) == 1
    assignment = binding_to_dict(next(iter(bindings)), formula.params)
    assert assignment[("object", 1)] == 2
    assert assignment[("object", 2)] == 1


def test_wildcard_injectivity_within_class():
    formula = parse_stl(
        "object{A}_*[m] >= object{B}_*[m]",
        dt=1.0,
        valid_classes=["object"],
    )
    entities = [entity(i, props={"m": float(i)}) for i in (1, 2)]

    bindings = enumerate_bindings(formula, obs(entities))

    assert len(bindings) == 2  # (1,2) and (2,1); never (i,i)
    for binding in bindings:
        assignment = binding_to_dict(binding, formula.params)
        assert assignment[("object", 1)] != assignment[("object", 2)]


def test_pinned_entity_excluded_from_wildcard_candidates():
    formula = parse_stl(
        "object_1[m] >= 0 and object{B}_*[m] >= 0",
        dt=1.0,
        valid_classes=["object"],
    )
    entities = [entity(1, props={"m": 1.0}), entity(2, props={"m": 2.0})]

    bindings = enumerate_bindings(formula, obs(entities))

    assert len(bindings) == 1
    assignment = binding_to_dict(next(iter(bindings)), formula.params)
    assert assignment[("object", 1)] == 1
    assert assignment[("object", 2)] == 2


def test_unconfirmed_entities_excluded_when_required():
    formula = parse_stl(
        "object_*[m] >= 0", dt=1.0, valid_classes=["object"]
    )
    entities = [
        entity(1, props={"m": 1.0}, confirmed=False),
        entity(2, props={"m": 2.0}),
    ]

    bindings = enumerate_bindings(
        formula, obs(entities), require_confirmed=True
    )

    assert len(bindings) == 1
    assert binding_to_dict(next(iter(bindings)), formula.params)[("object", 1)] == 2


# ---------------------------------------------------------------------------
# Parameterless templates (design-spec defect 4): one empty binding
# ---------------------------------------------------------------------------

def test_parameterless_template_gets_one_empty_binding():
    formula = parse_stl("1.0 >= 0.5", dt=1.0)

    bindings = enumerate_bindings(formula, obs([]))

    assert bindings == {()}


def test_parameterless_template_evaluates_through_manager():
    formula = parse_stl("G[0,3](1.0 >= 0.5)", dt=1.0)
    manager = IncrementalBindingManager(formula)

    for t in range(6):
        results = manager.update(obs([], t=t))

    assert len(results) == 1
    assert next(iter(results.values())) == 0.5


# ---------------------------------------------------------------------------
# Binding labels with string classes (design-spec defect 3)
# ---------------------------------------------------------------------------

def test_binding_label_works_with_string_classes():
    formula = parse_stl(
        "object_*[m] >= 0", dt=1.0, valid_classes=["object"]
    )
    binding = make_binding({("object", 1): 7}, formula.params)

    label = binding_label(binding, formula.params)

    assert "7" in label


# ---------------------------------------------------------------------------
# Incremental manager equals complete enumeration under churn
# ---------------------------------------------------------------------------

def test_incremental_matches_enumeration_under_slot_churn():
    formula = parse_stl(
        "object{A}_*[hinge] >= 0 and object{B}_*[mass] >= 0",
        dt=1.0,
        valid_classes=["object"],
    )
    manager = IncrementalBindingManager(formula)

    hinge1 = entity(1, props={"hinge": 0.5})
    mass2 = entity(2, props={"mass": 1.0})
    both3 = entity(3, props={"hinge": 0.1, "mass": 0.2})

    scenario = [
        [hinge1, mass2],          # one binding
        [hinge1, mass2, both3],   # entity 3 joins both slots
        [mass2, both3],           # entity 1 departs
        [mass2],                  # entity 3 departs → no bindings
        [hinge1, mass2],          # entity 1 returns
    ]

    for t, entities in enumerate(scenario):
        observation = obs(entities, t=t)
        manager.update(observation)
        assert manager.current_bindings == enumerate_bindings(
            formula, observation
        ), f"divergence at step {t}"


# ---------------------------------------------------------------------------
# Reference-implementation soundness (reviewer findings)
# ---------------------------------------------------------------------------

def test_enumeration_rejects_duplicate_pinned_even_when_hand_constructed():
    """The reference enumeration enforces injectivity itself: two same-class
    slots pinned to one entity yield no bindings, even for formulas built
    without the semantic compiler's compile-time rejection."""
    from specola.monitor import ParameterSpec, SpecTemplate

    formula = SpecTemplate(
        params=[
            ParameterSpec(("object", 1), False, ["m"], pinned_entity_id=2),
            ParameterSpec(("object", 2), False, ["m"], pinned_entity_id=2),
        ],
        predicate_fn=lambda b, r: 0.0,
    )

    bindings = enumerate_bindings(formula, obs([entity(2, props={"m": 1.0})]))

    assert bindings == set()


def test_symmetric_canonicalization_needs_identical_requirements():
    """Symmetric canonicalization applies only when permuting slots is a
    true invariance. Slots with different requirements must be fully
    enumerated even when their candidate sets momentarily coincide."""
    formula = parse_stl(
        "object{A}_*[hinge] >= object{B}_*[mass]",
        dt=1.0,
        valid_classes=["object"],
        symmetric=True,
    )
    entities = [
        entity(1, props={"hinge": 0.1, "mass": 0.2}),
        entity(2, props={"hinge": 0.3, "mass": 0.4}),
    ]

    bindings = enumerate_bindings(formula, obs(entities))

    assert len(bindings) == 2  # (A=1,B=2) and (A=2,B=1); robustness differs


def test_symmetric_identical_requirements_emits_canonical_representative():
    formula = parse_stl(
        "object{A}_*[m] + object{B}_*[m] >= 1",
        dt=1.0,
        valid_classes=["object"],
        symmetric=True,
    )
    entities = [entity(i, props={"m": float(i)}) for i in (1, 2, 3)]

    bindings = enumerate_bindings(formula, obs(entities))

    assert len(bindings) == 3  # C(3,2) canonical representatives


# ---------------------------------------------------------------------------
# Lifecycle identity (reviewer coverage gap)
# ---------------------------------------------------------------------------

def test_reactivation_within_grace_reuses_the_same_instance():
    formula = parse_stl(
        "G[0,10](object_*[m] >= 0)", dt=1.0, valid_classes=["object"]
    )
    manager = IncrementalBindingManager(formula)

    manager.update(obs([entity(1, props={"m": 1.0})], t=0))
    original = next(iter(manager._instances.values()))

    manager.update(obs([], t=1))  # suspend
    manager.update(obs([entity(1, props={"m": 1.0})], t=2))  # reactivate

    assert next(iter(manager._instances.values())) is original
    assert not original.suspended


def test_prune_then_return_creates_a_fresh_instance():
    formula = parse_stl(
        "G[0,2](object_*[m] >= 0)", dt=1.0, valid_classes=["object"]
    )
    manager = IncrementalBindingManager(formula)

    manager.update(obs([entity(1, props={"m": 1.0})], t=0))
    original = next(iter(manager._instances.values()))

    for t in (1, 2, 3):  # exceed the grace period
        manager.update(obs([], t=t))
    assert manager.n_pruned == 1

    manager.update(obs([entity(1, props={"m": 1.0})], t=4))

    replacement = next(iter(manager._instances.values()))
    assert replacement is not original
