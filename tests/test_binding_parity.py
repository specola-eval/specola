"""
Property-based binding parity (design §5.7 / §7.3).

The binding engine's correctness invariant, validated over generated
observation sequences:

    after every observation,
    manager.current_bindings == enumerate_bindings(formula, observation)

Generators cover pinned and wildcard selectors, heterogeneous per-slot
component requirements, symmetric and asymmetric formulas, entity
arrivals/departures/returns, track-state changes, and per-entity
component availability changes. A separate property checks the reference
enumeration itself against a brute-force injective oracle.

Tests validate; they do not prove — the algorithmic argument lives in
specola/bindings/incremental.py.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

# Skip visibly (not a collection error) in an env without hypothesis, so
# `pytest tests/` stays runnable everywhere while the skip is loud in the
# summary. All simulator envs have hypothesis installed (issue #32).
pytest.importorskip(
    "hypothesis",
    reason="binding-parity suite is property-based; pip install hypothesis",
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from specola.components import ComponentResolver
from specola.entities import EntityState, TrackState
from specola.monitor import (
    IncrementalBindingManager,
    ParameterSpec,
    SpecTemplate,
    enumerate_bindings,
    make_binding,
)


CLASSES = ("alpha", "beta")
COMPONENTS = ("p", "q")
ENTITY_IDS = tuple(range(5))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _requirement_sets(draw):
    return sorted(
        draw(st.sets(st.sampled_from(COMPONENTS), max_size=2))
    )


@st.composite
def formulas(draw, symmetric=None):
    if symmetric is None:
        symmetric = draw(st.booleans())

    # min 0: parameterless templates (one empty binding) go through the
    # same properties.
    n_slots = draw(st.integers(min_value=0, max_value=4))

    # Symmetric declarations require identical requirements across the
    # interchangeable wildcard slots of a class (documented contract).
    class_requirements = {
        cls: _requirement_sets(draw) for cls in CLASSES
    }

    params = []
    counters = {}
    for _ in range(n_slots):
        cls = draw(st.sampled_from(CLASSES))
        index = counters.get(cls, 0) + 1
        counters[cls] = index
        wildcard = draw(st.booleans())

        if symmetric and wildcard:
            requirements = list(class_requirements[cls])
        else:
            requirements = _requirement_sets(draw)

        params.append(
            ParameterSpec(
                param_key=(cls, index),
                is_wildcard=wildcard,
                required_components=requirements,
                pinned_entity_id=(
                    None
                    if wildcard
                    else draw(st.sampled_from(ENTITY_IDS))
                ),
            )
        )

    return SpecTemplate(
        params=params,
        predicate_fn=lambda binding, resolver: 0.0,
        symmetric=symmetric,
    )


@st.composite
def episodes(draw):
    """A class-stable entity population with per-step churn."""
    class_of = {
        eid: draw(st.sampled_from(CLASSES)) for eid in ENTITY_IDS
    }

    n_steps = draw(st.integers(min_value=2, max_value=7))
    steps = []

    for _ in range(n_steps):
        present = draw(
            st.sets(st.sampled_from(ENTITY_IDS), max_size=len(ENTITY_IDS))
        )
        entities = {}
        for eid in sorted(present):
            properties = {
                component: 1.0
                for component in draw(
                    st.sets(st.sampled_from(COMPONENTS), max_size=2)
                )
            }
            confirmed = draw(st.booleans())
            entities[eid] = EntityState(
                entity_id=eid,
                entity_class=class_of[eid],
                position=np.zeros(3),
                velocity=np.zeros(3),
                track_age=1,
                track_state=(
                    TrackState.CONFIRMED
                    if confirmed
                    else TrackState.TENTATIVE
                ),
                properties=properties,
            )
        steps.append(entities)

    return steps


def _observation(entities, t):
    return SimpleNamespace(
        timestep=t, time=float(t), perceived_entities=entities
    )


# ---------------------------------------------------------------------------
# Property 1: manager == complete enumeration, after every observation
# ---------------------------------------------------------------------------

@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(formula=formulas(), episode=episodes())
def test_manager_equals_complete_enumeration(formula, episode):
    manager = IncrementalBindingManager(formula)

    for t, entities in enumerate(episode):
        observation = _observation(entities, t)
        manager.update(observation)
        assert manager.current_bindings == enumerate_bindings(
            formula, observation
        ), f"parity broken at step {t}"


# ---------------------------------------------------------------------------
# Property 2: reference enumeration == brute-force injective oracle
# ---------------------------------------------------------------------------

def _brute_force(formula, observation):
    resolver = ComponentResolver(observation.perceived_entities)

    slot_candidates = []
    for parameter in formula.params:
        eligible = []
        for eid, entity in observation.perceived_entities.items():
            if entity.entity_class != parameter.entity_class:
                continue
            if not entity.is_confirmed:
                continue
            if (
                parameter.pinned_entity_id is not None
                and eid != parameter.pinned_entity_id
            ):
                continue
            if not resolver.supports_all(
                eid, parameter.required_components
            ):
                continue
            eligible.append(eid)
        slot_candidates.append(eligible)

    if not formula.params:
        return {()}

    result = set()
    for combo in itertools.product(*slot_candidates):
        used_per_class = {}
        injective = True
        for parameter, eid in zip(formula.params, combo):
            used = used_per_class.setdefault(parameter.entity_class, set())
            if eid in used:
                injective = False
                break
            used.add(eid)
        if injective:
            assignment = {
                parameter.param_key: eid
                for parameter, eid in zip(formula.params, combo)
            }
            result.add(
                make_binding(
                    _canonicalize(formula, assignment), formula.params
                )
            )
    return result


def _canonicalize(formula, assignment):
    """Collapse symmetric orbits to their canonical representative.

    For a symmetric formula, wildcard slots of a class with identical
    requirements are interchangeable; the canonical representative
    assigns the sorted entity IDs across those slots in slot order.
    """
    if not formula.symmetric:
        return assignment

    result = dict(assignment)
    for class_params in formula.params_by_class.values():
        wilds = [p for p in class_params if p.is_wildcard]
        if len(wilds) < 2:
            continue
        if any(
            p.required_components != wilds[0].required_components
            for p in wilds[1:]
        ):
            continue
        entity_ids = sorted(result[p.param_key] for p in wilds)
        for parameter, eid in zip(wilds, entity_ids):
            result[parameter.param_key] = eid
    return result


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(formula=formulas(), episode=episodes())
def test_reference_enumeration_equals_brute_force(formula, episode):
    for t, entities in enumerate(episode):
        observation = _observation(entities, t)
        assert enumerate_bindings(formula, observation) == _brute_force(
            formula, observation
        ), f"oracle divergence at step {t}"
