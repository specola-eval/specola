"""Spec-scoped extraction restricts what is read, never what it means.

The projection is active in monitor modes only (``specola run`` and the
``Monitor``/``Session`` researcher API). Every other entrypoint leaves
the scope ``None``, which must behave exactly as it did before the scope
existed — the first test below is the one that pins that.

Semantics under test (maintainer decisions, 2026-08-02):

1. Scope is class-based: ``{entity_class: frozenset(property keys)}``,
   unioned over every compiled parameter slot. Every entity of a
   referenced class gets the required keys whether or not it is bound,
   which is what keeps binding eligibility (which scans unbound
   entities) working.
2. Monitor modes only.
3. ``property_metadata`` and the field report read within the scope.
4. Off-spec properties are not extracted and therefore not
   finiteness-checked.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapter import MonitorObservation
from specola.adapters.mujoco import BodyRule, MujocoAdapter
from specola.engine import MonitorEngine
from specola.session import Monitor

# The scene, the perception-gate-aware body rules, and the stepping
# helper are the gather-plan suite's; this file tests a restriction of
# the same extraction path, so it must exercise the same scene.
from test_gather_plan import (  # noqa: E402
    _BODY_RULES,
    _Env,
    _MODEL_A,
    _MODEL_TWINS,
    _positions,
    _properties,
    _step,
)

pytestmark = pytest.mark.mujoco


# One key from a row-gather field whose owner also takes a position sink
# (xpos), one from an address-slice field (qpos), and one from a
# property built outside the curated-field path entirely (geom_size).
# Between them they cover every way a property reaches an observation.
_SCOPE: Dict[str, frozenset] = {
    "object": frozenset({"qpos_alpha_slide", "xpos_alpha_2"}),
    "geom": frozenset({"geom_size_alpha_geom_0"}),
}


def _make(
    xml: str = _MODEL_A,
    scope: Optional[Dict[str, frozenset]] = None,
    cls=MujocoAdapter,
    **kwargs,
) -> Tuple[_Env, MujocoAdapter]:
    env = _Env(xml)
    adapter = cls(env, body_rules=_BODY_RULES, **kwargs)
    adapter.reset()
    if scope is not None:
        adapter.set_extraction_scope(scope)
    return env, adapter


def _metadata(
    adapter: MujocoAdapter,
    observation: MonitorObservation,
) -> Dict[int, List[str]]:
    return {
        entity_id: sorted(adapter.property_metadata(entity_id))
        for entity_id in observation.perceived_entities
    }


def _count_materializer_calls(adapter: MujocoAdapter) -> Counter:
    """Count materializer entries per field, by wrapping the dispatch.

    A field that is skipped for the episode and a field that runs from a
    gather plan both enter their materializer exactly once; a field whose
    plan was refused enters it every step. That difference is what
    distinguishes "planned or skipped" from "back on the scalar path",
    which is the regression these tests exist to catch.
    """
    calls: Counter = Counter()
    for method_name in set(adapter._MATERIALIZER_METHODS.values()):
        original = getattr(adapter, method_name)

        def wrapper(field_name, perceived, _original=original):
            calls[field_name] += 1
            return _original(field_name, perceived)

        setattr(adapter, method_name, wrapper)
    return calls


def _run(adapter: MujocoAdapter, env: _Env, steps: int) -> List:
    observations = []
    for _ in range(steps):
        _step(env)
        observations.append(adapter.get_observation())
    return observations


# ---------------------------------------------------------------------
# The default path
# ---------------------------------------------------------------------


def test_scope_none_is_the_untouched_path():
    """An explicit ``None`` scope equals never mentioning a scope at all.

    Full-extraction equality against the pre-scope adapter is what the
    rest of the suite asserts; this pins the one comparison a test can
    make directly — that the new call, on its default argument, is inert
    across properties, positions, metadata and the field report.
    """
    env_untouched, untouched = _make()
    env_explicit, explicit = _make()
    explicit.set_extraction_scope(None)

    for _ in range(6):
        _step(env_untouched)
        _step(env_explicit)
        a = untouched.get_observation()
        b = explicit.get_observation()
        assert _properties(a) == _properties(b)
        assert _positions(a) == _positions(b)
        assert _metadata(untouched, a) == _metadata(explicit, b)

    assert untouched.field_materialization == explicit.field_materialization
    assert untouched.field_surface_report == explicit.field_surface_report
    assert explicit.field_surface_report.scope_active is False


# ---------------------------------------------------------------------
# Scope as a restriction
# ---------------------------------------------------------------------


def test_scoped_properties_are_the_full_ones_filtered():
    env_full, full = _make()
    env_scoped, scoped = _make(scope=_SCOPE)

    for _ in range(6):
        _step(env_full)
        _step(env_scoped)
        observed_full = full.get_observation()
        observed_scoped = scoped.get_observation()

        assert set(observed_full.perceived_entities) == set(
            observed_scoped.perceived_entities
        )
        for entity_id, entity in observed_full.perceived_entities.items():
            allowed = _SCOPE.get(entity.entity_class, frozenset())
            expected = {
                key: value
                for key, value in entity.properties.items()
                if key in allowed
            }
            assert (
                observed_scoped.perceived_entities[entity_id].properties
                == expected
            ), entity.entity_class

    # Non-empty, or the assertion above is vacuous.
    assert observed_scoped.perceived_entities[0].properties
    assert observed_scoped.perceived_entities[11].properties


def test_positions_are_not_scoped():
    env_full, full = _make()
    env_scoped, scoped = _make(scope=_SCOPE)

    for _ in range(6):
        _step(env_full)
        _step(env_scoped)
        assert _positions(full.get_observation()) == _positions(
            scoped.get_observation()
        )


def test_property_metadata_reads_within_the_scope():
    env, scoped = _make(scope=_SCOPE)
    observation = _run(scoped, env, 4)[-1]

    for entity_id, entity in observation.perceived_entities.items():
        assert set(scoped.property_metadata(entity_id)) == set(
            entity.properties
        )

    assert scoped.field_surface_report.scope_active is True


def test_mujoco_metadata_scope_commits_only_with_a_valid_observation():
    env, adapter = _make()
    scope_a = {"object": frozenset({"qpos_alpha_slide"})}
    scope_b = {"object": frozenset({"xpos_alpha_2"})}
    adapter.set_extraction_scope(scope_a)
    _step(env)
    accepted = adapter.get_observation()
    owner_id = next(
        entity_id for entity_id, entity in accepted.perceived_entities.items()
        if "qpos_alpha_slide" in entity.properties
    )
    assert set(adapter.property_metadata(owner_id)) == {"qpos_alpha_slide"}

    adapter.set_extraction_scope(scope_b)
    assert set(adapter.property_metadata(owner_id)) == {"qpos_alpha_slide"}

    canonical_metadata = adapter.property_metadata

    def missing_current_key(entity_id):
        metadata = canonical_metadata(entity_id)
        metadata.pop("xpos_alpha_2", None)
        return metadata

    adapter.property_metadata = missing_current_key
    adapter._parity_validated.clear()
    _step(env)
    with pytest.raises(RuntimeError, match="Property metadata mismatch"):
        adapter.get_observation()

    adapter.property_metadata = canonical_metadata
    assert adapter.last_observation is accepted
    assert set(adapter.property_metadata(owner_id)) == {"qpos_alpha_slide"}

    _step(env)
    committed = adapter.get_observation()
    assert set(committed.perceived_entities[owner_id].properties) == {"xpos_alpha_2"}
    assert set(adapter.property_metadata(owner_id)) == {"xpos_alpha_2"}


def test_scope_survives_reset_and_still_restricts():
    env, scoped = _make(scope=_SCOPE)
    _run(scoped, env, 2)
    scoped.reset()
    observation = _run(scoped, env, 2)[-1]

    for entity_id, entity in observation.perceived_entities.items():
        allowed = _SCOPE.get(entity.entity_class, frozenset())
        assert set(entity.properties) <= allowed


# ---------------------------------------------------------------------
# Gather-plan interaction
# ---------------------------------------------------------------------

# No "geom" entry at all: every geom-owned key is out of scope, which is
# what a realistic spec set does to the raw-geom classes.
_OBJECT_ONLY: Dict[str, frozenset] = {
    "object": frozenset({"qpos_alpha_slide"}),
}


def test_field_with_every_key_scoped_out_is_dropped_for_the_episode():
    env, scoped = _make(scope=_OBJECT_ONLY)
    calls = _count_materializer_calls(scoped)
    _run(scoped, env, 5)

    status = scoped.field_materialization["geom_xmat"]
    assert status.emitted_keys == ()
    assert status.reason is not None and "scope" in status.reason
    # Not planned and not re-entered: the field costs nothing per step.
    assert "geom_xmat" not in scoped._gather_plan
    assert calls["geom_xmat"] == 1


def test_position_only_field_still_plans_when_its_keys_are_scoped_out():
    """``geom_xpos`` loses every property key but keeps its position sink.

    Refusing to plan here would put the single most expensive field in
    the extraction profile back on the scalar path for the whole
    episode, so the plan has to survive with no property targets at all.
    """
    env, scoped = _make(scope=_OBJECT_ONLY)
    calls = _count_materializer_calls(scoped)
    observations = _run(scoped, env, 5)

    plan = scoped._gather_plan["geom_xpos"]
    assert plan.targets == []
    assert plan.position_rows
    assert calls["geom_xpos"] == 1

    # The positions the plan carries still follow the simulator: the
    # alpha geom is on a slide joint driven by the stepping helper.
    early = _positions(observations[0])[11]
    late = _positions(observations[-1])[11]
    assert early is not None and late is not None and early != late


def test_scope_naming_geom_and_site_classes_keeps_the_property_anchor():
    """A scope that keeps geom and site keys leaves that path untouched.

    The owner-record anchor (issue #101) is for rows a scope stripped of
    every property key. Where the keys survive, each position row is
    still placed by its own owner's gathered property indices, and the
    twins model is where that distinction bites: its coincident geoms and
    sites are exactly the rows the value trial cannot separate.
    """
    env_full, full = _make(xml=_MODEL_TWINS)
    reference = _run(full, env_full, 5)[-1]

    # Unioned over every owner of the class, the way a compiled spec set
    # reaches the adapter: one owner's keys alone would leave every other
    # owner of the same class anchorless, which is the other test.
    scope: Dict[str, frozenset] = {"geom": frozenset(), "site": frozenset()}
    for entity in reference.perceived_entities.values():
        if entity.entity_class in scope:
            scope[entity.entity_class] |= {
                key
                for key in entity.properties
                if key.startswith(("geom_xpos_", "site_xpos_"))
            }
    assert all(len(keys) >= 6 for keys in scope.values()), scope

    env, scoped = _make(xml=_MODEL_TWINS, scope=scope)
    observations = _run(scoped, env, 5)

    for field_name in ("geom_xpos", "site_xpos"):
        assert scoped.field_materialization[field_name].plan_refusal is None
        plan = scoped._gather_plan[field_name]
        assert plan.targets and plan.position_rows

    assert _positions(observations[-1]) == _positions(reference)
    for entity in observations[-1].perceived_entities.values():
        allowed = scope.get(entity.entity_class, frozenset())
        assert set(entity.properties) == set(
            reference.perceived_entities[
                entity.entity_id
            ].properties
        ) & allowed


def test_partially_scoped_field_plans_over_the_surviving_targets():
    env, scoped = _make(scope={"object": frozenset({"xpos_alpha_2"})})
    calls = _count_materializer_calls(scoped)
    observations = _run(scoped, env, 5)

    plan = scoped._gather_plan["xpos"]
    assert [key for _entity_id, key in plan.targets] == ["xpos_alpha_2"]
    assert calls["xpos"] == 1

    env_full, full = _make()
    for _ in range(5):
        _step(env_full)
    reference = full.get_observation()
    assert observations[-1].perceived_entities[0].properties == {
        "xpos_alpha_2": reference.perceived_entities[0].properties[
            "xpos_alpha_2"
        ]
    }


def test_scoped_out_key_of_a_gated_owner_does_not_break_planning():
    """A hidden owner's scoped-out key must not reach the plan filter.

    The scope gate resolves a hidden owner's class from the episode's
    identity history and drops the key without recording an omission, so
    the capture can hold a sink whose owner is absent from this
    observation — which the plan filter has to survive.
    """
    blocked: set = set()

    def gate(entity, _env) -> bool:
        return entity.entity_id not in blocked

    env, scoped = _make(perception_gate=gate)
    _run(scoped, env, 1)
    blocked.add(0)
    scoped.set_extraction_scope({"object": frozenset({"xpos_gamma_2"})})

    observation = _run(scoped, env, 3)[-1]
    assert 0 not in observation.perceived_entities
    assert set(observation.perceived_entities[1].properties) == {
        "xpos_gamma_2"
    }


# ---------------------------------------------------------------------
# Mid-episode arrival
# ---------------------------------------------------------------------

# A cube dropped onto a plane: the contact entity does not exist at
# reset and materializes once the cube lands.
_MODEL_DROP = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="alpha" pos="0 0 0.08">
      <freejoint name="alpha_free"/>
      <geom name="alpha_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


# A resting box (a plain `contact` pair from the first step) and a
# sphere that falls onto the floor later (a `robot_contact` pair, since
# its body is declared a robot body). One materializer serves both
# classes and allocates their entities as pairs first touch.
_MODEL_MIXED_CONTACT = """
<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <body name="box" pos="0 0 0.02">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="robot_link" pos="0.4 0 0.25">
      <freejoint/>
      <geom name="robot_geom" type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_scoping_one_contact_class_keeps_the_other_class_arriving():
    """A dynamic-population field must never be retired for the episode.

    ``MjData.contact`` materializes its own entities and decides each
    pair's class when the pair first touches, so a pass that emits
    nothing does not mean the field can never emit anything. Retiring it
    here lost every ``robot_contact`` that arose later — and a safety
    spec bound to that class then simply never evaluated, with no error
    anywhere.
    """
    scope = {"robot_contact": frozenset({"contact_normal_force_sum"})}
    env = _Env(_MODEL_MIXED_CONTACT)
    scoped = MujocoAdapter(
        env,
        body_rules=(BodyRule(match=lambda name: name == "box", cls="object"),),
        robot_bodies=("robot_link",),
    )
    scoped.reset()
    scoped.set_extraction_scope(scope)

    steps = 120
    seen = Counter()
    arrival = {}
    keys_by_class = {"contact": set(), "robot_contact": set()}
    for step in range(steps):
        mujoco.mj_step(env.model, env.data)
        observation = scoped.get_observation()
        for entity in observation.perceived_entities.values():
            if entity.entity_class not in keys_by_class:
                continue
            seen[entity.entity_class] += 1
            keys_by_class[entity.entity_class] |= set(entity.properties)
            arrival.setdefault(entity.entity_class, step)

    # The in-scope class arrives mid-episode and carries exactly its
    # scoped key. Without the exemption both of these are zero.
    assert seen["robot_contact"] > 0
    assert arrival["robot_contact"] > 0, "arrival was not mid-episode"
    assert keys_by_class["robot_contact"] == {"contact_normal_force_sum"}

    # The out-of-scope class is projected, not dropped: its entities are
    # still there, every step, carrying no keys. The bug also made this
    # one appear exactly once and then vanish.
    assert keys_by_class["contact"] == set()
    assert seen["contact"] > 1

    # Nothing was retired, so the field keeps looking for new pairs.
    assert "contact" not in scoped._scope_skipped_fields


def test_midepisode_arrival_gets_exactly_the_scoped_keys():
    scope = {"contact": frozenset({"contact_normal_force_sum"})}
    env, scoped = _make(xml=_MODEL_DROP, scope=scope)

    contacts = {}
    for _ in range(120):
        mujoco.mj_step(env.model, env.data)
        observation = scoped.get_observation()
        contacts = {
            entity_id: entity
            for entity_id, entity in observation.perceived_entities.items()
            if entity.entity_class == "contact"
        }
        if contacts:
            break

    assert contacts, "no contact pair materialized in 120 steps"
    for entity_id, entity in contacts.items():
        assert set(entity.properties) == {"contact_normal_force_sum"}
        assert set(scoped.property_metadata(entity_id)) == {
            "contact_normal_force_sum"
        }


# ---------------------------------------------------------------------
# Scope computation and monitor-mode wiring
# ---------------------------------------------------------------------


_SPECS = """
specs:
- name: alpha_height
  formula: "G[0,10](object_*[xpos_alpha_2] >= 0.0)"
- name: alpha_slide
  formula: "G[0,10](object_*[qpos_alpha_slide] <= 1.0)"
"""


def _write_specs(tmp_path) -> str:
    path = tmp_path / "specs.yaml"
    path.write_text(_SPECS)
    return str(path)


def test_engine_scope_unions_every_slot(tmp_path):
    env, adapter = _make()
    engine = MonitorEngine.load(_write_specs(tmp_path))
    engine.start_episode(dt=adapter.dt, capabilities=adapter.capabilities)

    assert engine.extraction_scope() == {
        "object": frozenset({"xpos_alpha_2", "qpos_alpha_slide"}),
    }


def test_session_scopes_the_adapter_and_keeps_the_rho_series(tmp_path):
    specs = _write_specs(tmp_path)
    env_full, adapter_full = _make()
    env_scoped, adapter_scoped = _make()

    session_full = Monitor(specs, results_dir=None).watch(adapter_full)
    # The control arm: same compilation, scope switched back off.
    adapter_full.set_extraction_scope(None)
    session_scoped = Monitor(specs, results_dir=None).watch(adapter_scoped)

    assert adapter_scoped.field_surface_report.scope_active is True
    assert adapter_full.field_surface_report.scope_active is False

    series_full = []
    series_scoped = []
    for _ in range(30):
        _step(env_full)
        _step(env_scoped)
        for series, session in (
            (series_full, session_full),
            (series_scoped, session_scoped),
        ):
            result = session.step()
            series.append(
                sorted(
                    (
                        evaluation.spec_name,
                        evaluation.binding,
                        evaluation.robustness,
                        evaluation.verdict_final,
                    )
                    for evaluation in result.raw.evaluations
                )
            )

    assert series_full == series_scoped
    # Vacuous unless something actually bound and evaluated.
    assert any(step for step in series_full)


def test_discovery_entrypoints_leave_the_scope_alone():
    """No entrypoint other than the monitor session sets a scope."""
    import specola.discover as discover
    import specola.check_specs as check_specs
    import specola.explore as explore
    import specola.write_specs as write_specs

    for module in (discover, check_specs, explore, write_specs):
        source = open(module.__file__).read()
        assert "set_extraction_scope" not in source, module.__name__
