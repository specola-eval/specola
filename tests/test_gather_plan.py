"""Per-episode gather plans must reproduce the scalar emit path exactly."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapter import MonitorObservation
from specola.adapters.mujoco import BodyRule, MujocoAdapter

# Classified bodies (rather than raw ones) are what carry the
# EntityState.position sink and the perception gate.
_BODY_RULES = (
    BodyRule(
        match=lambda name: name in {"alpha", "gamma", "delta"},
        cls="object",
    ),
)

pytestmark = pytest.mark.mujoco


# Carries every owner kind the curated fields gather from -- sites,
# cameras, lights, mocap bodies, a tendon, a sensor and a stateful
# (integrator) actuator -- because several of those fields have no owner
# on any simulator scene the benchmarks run, so this model is their only
# equality evidence. Two of each owner kind, not one: a single owner
# always sits at row 0, where the row stride cancels and a wrong width
# gathers the right values anyway.
_MODEL_A = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light name="key_light" pos="0 0 2" dir="0 0 -1"/>
    <light name="fill_light" pos="1 1 2" dir="-1 -1 -1"/>
    <camera name="fixed_cam" pos="1 0 0.5" euler="0 1.57 0"/>
    <camera name="side_cam" pos="0 1 0.6" euler="0 0 1.57"/>
    <body name="alpha" pos="0 0 0.4">
      <joint name="alpha_slide" type="slide" axis="0 0 1"/>
      <geom name="alpha_geom" type="box" size="0.02 0.02 0.02"/>
      <site name="alpha_site" pos="0 0 0.03" size="0.005"/>
      <body name="beta" pos="0.1 0 0">
        <joint name="beta_hinge" type="hinge"/>
        <geom name="beta_geom" type="sphere" size="0.02"/>
        <site name="beta_site" pos="0.01 0.02 0.03" size="0.004"/>
      </body>
    </body>
    <body name="gamma" pos="0.3 0 0.2">
      <freejoint name="gamma_free"/>
      <geom name="gamma_geom" type="capsule" size="0.02 0.03"/>
    </body>
    <body name="target" mocap="true" pos="0.5 0 0.3">
      <geom name="target_geom" type="sphere" size="0.01"
            contype="0" conaffinity="0"/>
    </body>
    <body name="target_two" mocap="true" pos="0.7 0.1 0.35">
      <geom name="target_two_geom" type="sphere" size="0.01"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="alpha_tendon">
      <joint joint="alpha_slide" coef="1"/>
    </fixed>
    <fixed name="beta_tendon">
      <joint joint="beta_hinge" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="alpha_motor" joint="alpha_slide"/>
    <general name="beta_integrator" joint="beta_hinge"
             dyntype="integrator" gainprm="1"/>
    <!-- gear, so the two integrators on one joint do not report
         identical actuator_length and actuator_velocity: equal values
         at two owner ids hide a transposed index. -->
    <general name="beta_integrator_two" joint="beta_hinge"
             dyntype="integrator" gainprm="0.5" gear="2"/>
  </actuator>
  <sensor>
    <framepos name="alpha_pos_sensor" objtype="site" objname="alpha_site"/>
    <framequat name="beta_quat_sensor" objtype="site" objname="beta_site"/>
  </sensor>
</mujoco>
"""


# Flexes need their own model: they are the one owner kind that cannot be
# bolted onto _MODEL_A without dedicated vertex bodies. Two of them, so
# the second one's address into the flat vertex/edge arrays is nonzero.
_MODEL_FLEX = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="v0" pos="0 0 1"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
    <body name="v1" pos="0.1 0 1"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
    <body name="v2" pos="0.25 0 1"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
    <body name="w0" pos="0 0 1.5"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
    <body name="w1" pos="0.17 0 1.5"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
    <body name="w2" pos="0.41 0 1.5"><freejoint/>
      <geom type="sphere" size="0.005"/></body>
  </worldbody>
  <deformable>
    <!-- dim=1: edge stiffness is rejected on 2-D flexes, and without it
         MuJoCo never requests the edge Jacobian, which is what makes
         flexedge_velocity applicable at all. The second rope's vertices
         are spaced unevenly and differently from the first, so all four
         edge lengths differ: two identical ropes would gather the right
         values from the wrong addresses. -->
    <flex name="rope" dim="1" body="v0 v1 v2"
          vertex="0 0 0 0 0 0 0 0 0" element="0 1 1 2">
      <edge stiffness="10" damping="1"/>
    </flex>
    <flex name="rope_two" dim="1" body="w0 w1 w2"
          vertex="0 0 0 0 0 0 0 0 0" element="0 1 1 2">
      <edge stiffness="10" damping="1"/>
    </flex>
  </deformable>
</mujoco>
"""

_MODEL_B = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="delta" pos="0 0 0.25">
      <joint name="delta_hinge" type="hinge"/>
      <geom name="delta_geom" type="box" size="0.03 0.03 0.03"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="delta_motor" joint="delta_hinge"/>
  </actuator>
</mujoco>
"""


# Same body count as _MODEL_A, different names and placement: a stale
# plan swapped onto this model keeps every index in range and the right
# length, so nothing but the values gives it away.
_MODEL_C = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="epsilon" pos="0.2 0 0.5">
      <joint name="epsilon_slide" type="slide" axis="0 0 1"/>
      <geom name="epsilon_geom" type="sphere" size="0.03"/>
      <body name="zeta" pos="0 0.15 0">
        <joint name="zeta_hinge" type="hinge"/>
        <geom name="zeta_geom" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
    <body name="eta" pos="0.4 0.1 0.3">
      <freejoint name="eta_free"/>
      <geom name="eta_geom" type="capsule" size="0.03 0.02"/>
    </body>
    <body name="theta" mocap="true" pos="0.6 0.2 0.4">
      <geom name="theta_geom" type="sphere" size="0.01"
            contype="0" conaffinity="0"/>
    </body>
    <body name="iota" mocap="true" pos="0.8 0.3 0.45">
      <geom name="iota_geom" type="sphere" size="0.01"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="epsilon_motor" joint="epsilon_slide"/>
  </actuator>
</mujoco>
"""


# Coincident owners of the ordinary kind: a collision geom and the
# visual twin drawn over it share a frame, and so do two sites on one
# attachment point. Both pairs hold identical position rows all episode,
# which is the shape a scoped monitor run leaves geom_xpos and site_xpos
# in once every geom and site key is dropped (issue #101). The bodies
# are named for _BODY_RULES so the model still has one in-scope key.
_MODEL_TWINS = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="alpha" pos="0 0 0.4">
      <joint name="alpha_slide" type="slide" axis="0 0 1"/>
      <geom name="alpha_geom" type="box" size="0.02 0.02 0.02"/>
      <geom name="alpha_geom_visual" type="box" size="0.021 0.021 0.021"
            contype="0" conaffinity="0"/>
      <site name="alpha_site" pos="0 0 0.03" size="0.005"/>
      <site name="alpha_site_twin" pos="0 0 0.03" size="0.004"/>
    </body>
    <body name="gamma" pos="0.3 0 0.2">
      <freejoint name="gamma_free"/>
      <geom name="gamma_geom" type="capsule" size="0.02 0.03"/>
      <site name="gamma_site" pos="0.01 0 0" size="0.004"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="alpha_motor" joint="alpha_slide"/>
  </actuator>
</mujoco>
"""


class _Env:
    """Minimal MuJoCo env surface: the adapter only needs model/data."""

    def __init__(self, xml: str = _MODEL_A) -> None:
        self.load(xml)

    def load(self, xml: str) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # xfrc_applied and qfrc_applied are user inputs the solver never
        # writes, so they stay all-zero unless something fills them --
        # and a field whose every value is zero cannot tell a right
        # gather index from a wrong one. qvel likewise starts at rest,
        # which leaves every velocity-derived field identically zero.
        # All three are small enough not to disturb the physics.
        for name in ("xfrc_applied", "qfrc_applied", "qvel"):
            array = getattr(self.data, name)
            array[:] = 1e-3 * np.arange(
                array.size, dtype=float
            ).reshape(array.shape)
        mujoco.mj_forward(self.model, self.data)


class _ScalarAdapter(MujocoAdapter):
    """The pre-vectorization path: no field is ever planned."""

    _VECTOR_CANDIDATES = frozenset()


class _NoSinkTableAdapter(MujocoAdapter):
    """The pre-Task-4 path: every step reformats keys and rebuilds meta.

    The two fields MuJoCo cannot gather (``qM``, ``actuator_moment``)
    have no vectorized arm to compare against, so their reference is the
    same adapter with the sink-table shortcut switched off.
    """

    def _replay_value_only_sinks(self, field_name, perceived, values):
        return False


def _step(env: _Env, control: float = 0.05) -> None:
    # Per-actuator control, not one value for all: two stateful
    # actuators driven identically hold identical activations, and then
    # act/act_dot cannot tell their two addresses apart.
    env.data.ctrl[:] = control * np.arange(
        1, env.data.ctrl.size + 1, dtype=float
    )
    mujoco.mj_step(env.model, env.data)


def _run(
    adapter: MujocoAdapter,
    env: _Env,
    steps: int,
) -> List[MonitorObservation]:
    observations = []
    for _ in range(steps):
        _step(env)
        observations.append(adapter.get_observation())
    return observations


def _properties(
    observation: MonitorObservation,
) -> Dict[int, Dict[str, float]]:
    return {
        entity_id: dict(entity.properties)
        for entity_id, entity in observation.perceived_entities.items()
    }


def _vector(value) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    return tuple(np.asarray(value, dtype=float).tolist())


def _positions(
    observation: MonitorObservation,
) -> Dict[int, Optional[Tuple[float, ...]]]:
    return {
        entity_id: _vector(entity.position)
        for entity_id, entity in observation.perceived_entities.items()
    }


def _contract(
    adapter: MujocoAdapter,
    observation: MonitorObservation,
) -> Dict[str, Any]:
    """Everything the adapter exposes for one observation.

    The same surface the issue #101 probe recorded when it compared a
    scope-skipping run against the scalar path: per entity, its class,
    track state and age, position, velocity, properties and property
    metadata; per observation, the component tags and the clock; per
    field, its materialization record. ``plan_refusal`` is the one field
    dropped, because it is the one that must differ -- an adapter that
    never plans never refuses either.
    """
    return {
        "timestep": observation.timestep,
        "time": observation.time,
        "component_tags": {
            entity_id: {key: tuple(tags) for key, tags in keys.items()}
            for entity_id, keys in observation.component_tags.items()
        },
        "entities": {
            entity_id: (
                entity.entity_class,
                entity.track_state,
                entity.track_age,
                _vector(entity.position),
                _vector(entity.velocity),
                dict(entity.properties),
                adapter.property_metadata(entity_id),
            )
            for entity_id, entity in observation.perceived_entities.items()
        },
        "fields": {
            field_name: replace(record, plan_refusal=None)
            for field_name, record in adapter.field_materialization.items()
        },
    }


def _make(
    cls=MujocoAdapter,
    xml: str = _MODEL_A,
    **kwargs,
) -> Tuple[_Env, MujocoAdapter]:
    env = _Env(xml)
    adapter = cls(env, body_rules=_BODY_RULES, **kwargs)
    adapter.reset()
    return env, adapter


# Every field that must reach the gather path on _MODEL_A, grown group
# by group as fields are converted (issue #96, Task 3). Ordering follows
# benchmarks/results/extraction/field-classification.md.
_PLANNED_FIELDS = (
    # Group A -- row gather, flat index row * width + component.
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cinert",
    "crb",
    "cacc",
    "cfrc_ext",
    "cfrc_int",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "cvel",
    "subtree_com",
    "subtree_linvel",
    "subtree_angmom",
    "xfrc_applied",
    "xanchor",
    "xaxis",
    "cam_xpos",
    "cam_xmat",
    "mocap_pos",
    "mocap_quat",
    "light_xpos",
    "light_xdir",
    # Group B -- address-slice gather, flat index start + offset.
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qfrc_actuator",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_damper",
    "qfrc_fluid",
    "qfrc_gravcomp",
    "qfrc_passive",
    "qfrc_smooth",
    "qfrc_spring",
    "qfrc_applied",
    "cdof",
    "cdof_dot",
    "sensordata",
    "act",
    "act_dot",
    # Group C -- scalar per owner, flat index is the owner id.
    "ctrl",
    "actuator_force",
    "actuator_length",
    "actuator_velocity",
    "ten_length",
    "ten_velocity",
    # Group D -- two values, indexed directly.
    "energy",
)

# Flexes live in their own model, so they carry their own expectation.
_FLEX_PLANNED_FIELDS = (
    "flexvert_xpos",
    "flexedge_length",
    "flexedge_velocity",
)

# The five fields that also write EntityState.position.
_POSITION_FIELDS = frozenset(
    {"xpos", "geom_xpos", "site_xpos", "cam_xpos", "mocap_pos"}
)


def test_every_candidate_is_expected_to_plan() -> None:
    # _PLANNED_FIELDS is the intent (these must reach the gather path on
    # the model named beside them) and _VECTOR_CANDIDATES is the
    # implementation; deriving one from the other would make the install
    # test vacuous. Asserting they agree instead catches a candidate added
    # without coverage.
    assert set(_PLANNED_FIELDS) | set(_FLEX_PLANNED_FIELDS) == set(
        MujocoAdapter._VECTOR_CANDIDATES
    )


@pytest.mark.parametrize(
    "xml, field_name",
    [(_MODEL_A, name) for name in _PLANNED_FIELDS]
    + [(_MODEL_FLEX, name) for name in _FLEX_PLANNED_FIELDS],
    ids=list(_PLANNED_FIELDS) + list(_FLEX_PLANNED_FIELDS),
)
def test_gather_plan_installs(xml: str, field_name: str) -> None:
    env, adapter = _make(xml=xml)
    _run(adapter, env, steps=2)

    record = adapter.field_materialization[field_name]
    assert record.scene_applicable, record.reason
    assert record.plan_refusal is None
    plan = adapter._gather_plan[field_name]
    assert plan.attr in (field_name, "body_" + field_name)
    assert plan.flat_indices.dtype == np.intp
    # Nothing else in the plan says the position recorder ran, and the
    # install trial only proves the rows it did record gather correctly.
    assert bool(plan.position_rows) == (field_name in _POSITION_FIELDS)


def test_xaxis_plans_the_exact_hinge_slide_owner_population() -> None:
    # xaxis declares hinge/slide owners, not all joints. The free joint in
    # _MODEL_A is outside that static population, so it neither records an
    # owner omission nor prevents the surviving rows from gathering.
    env, adapter = _make()
    _run(adapter, env, steps=2)

    assert "xanchor" in adapter._gather_plan
    assert "xaxis" in adapter._gather_plan
    record = adapter.field_materialization["xaxis"]
    assert record.owner_omissions == ()
    assert record.plan_refusal is None
    plan = adapter._gather_plan["xaxis"]
    expected_rows = [
        joint_id
        for joint_id, joint_type in enumerate(env.model.jnt_type)
        if int(joint_type) in (2, 3)
    ]
    np.testing.assert_array_equal(
        plan.flat_indices,
        np.array(
            [
                joint_id * 3 + axis
                for joint_id in expected_rows
                for axis in range(3)
            ],
            dtype=np.intp,
        ),
    )


def test_gather_plan_installs_for_body_pose() -> None:
    env, adapter = _make()
    _run(adapter, env, steps=2)

    nbody = int(env.model.nbody)
    plans = adapter._gather_plan
    assert set(plans) >= {"xpos", "xquat", "xmat"}
    widths = {"xpos": 3, "xquat": 4, "xmat": 9}
    for field_name, width in widths.items():
        plan = plans[field_name]
        assert plan.attr in (field_name, "body_" + field_name)
        assert plan.flat_indices.dtype == np.intp
        assert plan.flat_indices.size == nbody * width
        assert len(plan.targets) == nbody * width
    assert plans["xpos"].position_rows
    assert not plans["xquat"].position_rows
    assert not plans["xmat"].position_rows


# Drops every geom-owned key, which leaves geom_xpos with no property
# targets and nothing but its position sink -- the shape of plan the
# value trial cannot see into (issue #98).
_OBJECT_ONLY_SCOPE = {"object": frozenset({"qpos_alpha_slide"})}


def _position_row_adapter(field_name: str, corrupt):
    """Adapter that rewrites one field's recorded position rows.

    A row is a materializer's claim about where an owner's position was
    copied from, and no materializer records a wrong one today, so
    corrupting the record is the only way to exercise the install trial.
    """

    class _Adapter(MujocoAdapter):
        def _install_gather_plan(self, name, *args):
            if name == field_name:
                rows = self._plan_position_rows
                rows[:] = corrupt(rows)
            return super()._install_gather_plan(name, *args)

    return _Adapter


def _swapped(rows):
    """The first two owners recorded against each other's rows.

    Both row ids stay in range and both belong to owners the field really
    writes, so nothing but a comparison against the positions the scalar
    pass emitted can tell the plan is wrong.
    """
    if len(rows) < 2:
        return rows
    return [
        (rows[0][0], rows[1][1]),
        (rows[1][0], rows[0][1]),
    ] + rows[2:]


def _wrong_position_row_adapter(field_name: str):
    return _position_row_adapter(field_name, _swapped)


@pytest.mark.parametrize(
    "field_name, scope",
    [("xpos", None), ("geom_xpos", _OBJECT_ONLY_SCOPE)],
    ids=["with_properties", "position_only"],
)
def test_wrong_position_row_refuses_plan(field_name, scope) -> None:
    if scope is not None:
        # The case the property trial passes trivially: no flat indices
        # left to gather, so the two arrays it compares are both empty.
        env_clean, clean = _make()
        clean.set_extraction_scope(scope)
        _run(clean, env_clean, steps=2)
        reference = clean._gather_plan[field_name]
        assert reference.flat_indices.size == 0
        assert reference.position_rows

    env, adapter = _make(_wrong_position_row_adapter(field_name))
    if scope is not None:
        adapter.set_extraction_scope(scope)
    _run(adapter, env, steps=2)

    assert field_name not in adapter._gather_plan
    refusal = adapter.field_materialization[field_name].plan_refusal
    assert refusal is not None
    assert "did not reproduce the emitted positions" in refusal


@pytest.mark.parametrize(
    "row_id", [10_000, -1], ids=["out_of_range", "negative"]
)
def test_position_row_outside_the_source_array_refuses_plan(
    row_id: int,
) -> None:
    # The negative row is the worse of the two: Python indexes it from
    # the end of the array and gathers some other owner's position, so
    # it has to be refused by range rather than left to the value
    # comparison, which only catches it when the two happen to differ.
    env, adapter = _make(
        _position_row_adapter(
            "xpos", lambda rows: [(rows[0][0], row_id)] + rows[1:]
        )
    )
    _run(adapter, env, steps=2)

    assert "xpos" not in adapter._gather_plan
    refusal = adapter.field_materialization["xpos"].plan_refusal
    assert refusal is not None
    assert "a position row falls outside" in refusal


def _equal_mocap_env() -> _Env:
    """A scene whose two mocap bodies sit at the same point.

    Nothing forbids it: mocap targets are placed by the task, and two of
    them stacked (or both left at the origin) is an ordinary reset state.
    """
    env = _Env()
    env.data.mocap_pos[1] = env.data.mocap_pos[0]
    mujoco.mj_forward(env.model, env.data)
    return env


def test_coincident_owners_with_property_anchors_still_plan() -> None:
    # Coincident owners are ordinary -- unused mocap bodies parked
    # together, a collision geom and its visual twin -- and they must not
    # cost the field its gather. The owner's own property indices say
    # which row it was read from, whatever the values are.
    env = _equal_mocap_env()
    env_scalar = _equal_mocap_env()
    adapter = MujocoAdapter(env, body_rules=_BODY_RULES)
    scalar = _ScalarAdapter(env_scalar, body_rules=_BODY_RULES)
    adapter.reset()
    scalar.reset()
    observation = _run(adapter, env, steps=3)[-1]
    scalar_observation = _run(scalar, env_scalar, steps=3)[-1]

    for field_name in ("mocap_pos", "geom_xpos"):
        assert adapter.field_materialization[field_name].plan_refusal is None
        assert field_name in adapter._gather_plan
    assert _positions(observation) == _positions(scalar_observation)


def test_row_disagreeing_with_its_owners_properties_refuses_plan() -> None:
    # The swap the value comparison cannot see: both owners hold the same
    # vector, so gathering either row reproduces the emitted positions.
    # Their property indices still disagree with the recorded rows.
    env = _equal_mocap_env()
    adapter = _wrong_position_row_adapter("mocap_pos")(
        env, body_rules=_BODY_RULES
    )
    adapter.reset()
    _run(adapter, env, steps=2)

    assert "mocap_pos" not in adapter._gather_plan
    refusal = adapter.field_materialization["mocap_pos"].plan_refusal
    assert refusal is not None
    assert "not a row the same owner's properties gather from" in refusal


def test_coincident_owners_without_property_anchors_still_plan() -> None:
    # The scoped position-only shape: every property key of the field is
    # dropped, so no owner has a property anchor, and two owners sit at
    # the same point. The reset's own owner records still say which row
    # each owner was read from (issue #101), so the field keeps its plan.
    env = _equal_mocap_env()
    env_scalar = _equal_mocap_env()
    adapter = MujocoAdapter(env, body_rules=_BODY_RULES)
    scalar = _ScalarAdapter(env_scalar, body_rules=_BODY_RULES)
    adapter.reset()
    scalar.reset()
    adapter.set_extraction_scope(_OBJECT_ONLY_SCOPE)
    scalar.set_extraction_scope(_OBJECT_ONLY_SCOPE)
    observation = _run(adapter, env, steps=3)[-1]
    scalar_observation = _run(scalar, env_scalar, steps=3)[-1]

    for field_name in ("geom_xpos", "mocap_pos"):
        assert adapter.field_materialization[field_name].plan_refusal is None
        plan = adapter._gather_plan[field_name]
        assert plan.targets == []
        assert plan.position_rows
    assert _positions(observation) == _positions(scalar_observation)


class _NoRecordAnchorAdapter(MujocoAdapter):
    """The ladder with its last structural rung removed."""

    def _position_row_records(self, field_name):
        return {}


def test_coincident_owners_with_no_anchor_at_all_refuse_the_plan() -> None:
    # The final rung, reached only by a position row no property key and
    # no owner record can place: two owners at the same point cannot be
    # told apart by anything, and refusing is the right answer. No
    # shipped materializer records such a row, so blinding the record
    # lookup is what exercises it.
    env = _equal_mocap_env()
    adapter = _NoRecordAnchorAdapter(env, body_rules=_BODY_RULES)
    adapter.reset()
    adapter.set_extraction_scope(_OBJECT_ONLY_SCOPE)
    _run(adapter, env, steps=2)

    assert "geom_xpos" not in adapter._gather_plan
    refusal = adapter.field_materialization["geom_xpos"].plan_refusal
    assert refusal is not None and "hold the same vector" in refusal


def test_position_only_plans_install_over_coincident_twins() -> None:
    """The measured shape from issue #101, end to end.

    Every geom and site key is out of scope, so neither field has a
    property anchor for any owner, and each carries a coincident pair --
    a geom and its visual twin, two sites on one attachment point. Both
    fields have to plan (that is the whole speedup: their materializers
    stop running), and what the scoped adapter reports afterwards has to
    match the scalar path over the full contract, not just positions.
    """
    env, adapter = _make(xml=_MODEL_TWINS)
    env_scalar, scalar = _make(_ScalarAdapter, xml=_MODEL_TWINS)
    adapter.set_extraction_scope(_OBJECT_ONLY_SCOPE)
    scalar.set_extraction_scope(_OBJECT_ONLY_SCOPE)

    for step in range(4):
        _step(env)
        _step(env_scalar)
        assert _contract(adapter, adapter.get_observation()) == _contract(
            scalar, scalar.get_observation()
        ), step

    for field_name, rows in (("geom_xpos", 3), ("site_xpos", 3)):
        assert adapter.field_materialization[field_name].plan_refusal is None
        plan = adapter._gather_plan[field_name]
        assert plan.targets == []
        assert len(plan.position_rows) == rows
    assert not scalar._gather_plan


def test_twin_rows_swapped_against_their_records_refuse_the_plan() -> None:
    # The corruption no other rung can see: the two twins hold the same
    # vector, so the value trial reproduces the emitted positions either
    # way, and the scope left no property key to anchor on. Only the
    # owner records disagree.
    env, adapter = _make(
        _wrong_position_row_adapter("geom_xpos"), xml=_MODEL_TWINS
    )
    adapter.set_extraction_scope(_OBJECT_ONLY_SCOPE)
    _run(adapter, env, steps=2)

    assert "geom_xpos" not in adapter._gather_plan
    refusal = adapter.field_materialization["geom_xpos"].plan_refusal
    assert refusal is not None and "owner record holds" in refusal


def test_separated_position_rows_plan_and_track_their_own_owners() -> None:
    # The same field on the unmodified scene, where the two mocap bodies
    # are metres apart: the plan installs and each owner keeps its own
    # position, which is what the refusal above must not cost in general.
    env, adapter = _make()
    env_scalar, scalar = _make(_ScalarAdapter)
    observation = _run(adapter, env, steps=3)[-1]
    scalar_observation = _run(scalar, env_scalar, steps=3)[-1]

    assert len(adapter._gather_plan["mocap_pos"].position_rows) == 2
    assert _positions(observation) == _positions(scalar_observation)


@pytest.mark.parametrize(
    "xml", [_MODEL_A, _MODEL_FLEX], ids=["mixed", "flex"]
)
def test_vectorized_equals_scalar_observation(xml: str) -> None:
    env_a, adapter_a = _make(xml=xml)
    env_b, adapter_b = _make(_ScalarAdapter, xml=xml)

    obs_a = _run(adapter_a, env_a, steps=3)
    obs_b = _run(adapter_b, env_b, steps=3)

    assert adapter_a._gather_plan
    assert not adapter_b._gather_plan

    for step, (left, right) in enumerate(zip(obs_a, obs_b)):
        assert _properties(left) == _properties(right), step
        assert _positions(left) == _positions(right), step
        for entity in left.perceived_entities.values():
            for value in entity.properties.values():
                assert type(value) is float


def test_cold_vs_warm_equality() -> None:
    env, warm = _make()
    warm_obs = _run(warm, env, steps=3)[-1]

    cold = MujocoAdapter(env, body_rules=_BODY_RULES)
    cold.reset()
    cold_obs = cold.get_observation()

    assert _properties(warm_obs) == _properties(cold_obs)
    assert _positions(warm_obs) == _positions(cold_obs)


def test_position_sink_preserved() -> None:
    env, adapter = _make()
    observation = _run(adapter, env, steps=3)[-1]

    body_entities = [
        (entity_id, entity)
        for entity_id, entity in observation.perceived_entities.items()
        if entity.position is not None
    ]
    assert body_entities

    positions = np.asarray(env.data.xpos, dtype=float)
    matched = 0
    for _entity_id, entity in body_entities:
        current = np.asarray(entity.position, dtype=float)
        if current.shape != (3,):
            continue
        if np.any(np.all(np.isclose(positions, current), axis=1)):
            matched += 1
    assert matched

    # Copy semantics: the sink must not alias the live MjData row.
    before = {
        entity_id: np.asarray(entity.position, dtype=float).copy()
        for entity_id, entity in body_entities
    }
    for attr in (
        "xpos", "geom_xpos", "site_xpos", "cam_xpos", "mocap_pos"
    ):
        getattr(env.data, attr)[:] += 1.0
    for entity_id, entity in body_entities:
        assert np.array_equal(
            np.asarray(entity.position, dtype=float), before[entity_id]
        )


def test_perception_gate_skip() -> None:
    seen: List[int] = []
    blocked: set = set()

    def watching_gate(entity, _env) -> bool:
        seen.append(entity.entity_id)
        return entity.entity_id not in blocked

    def gate(entity, _env) -> bool:
        return entity.entity_id not in blocked

    env, adapter = _make(perception_gate=watching_gate)
    env_scalar, scalar = _make(_ScalarAdapter, perception_gate=gate)
    settled = _run(adapter, env, steps=2)[-1]
    _run(scalar, env_scalar, steps=2)
    assert adapter._gather_plan

    planned = {
        entity_id for entity_id, _key in adapter._gather_plan["xpos"].targets
    }
    gated = [
        entity_id for entity_id in dict.fromkeys(seen)
        if entity_id in planned
    ]
    assert gated
    hidden = gated[0]
    hidden_keys = set(settled.perceived_entities[hidden].properties)

    blocked.add(hidden)
    away = _run(adapter, env, steps=1)[-1]
    away_scalar = _run(scalar, env_scalar, steps=1)[-1]
    assert hidden not in away.perceived_entities
    assert away.perceived_entities
    assert _properties(away) == _properties(away_scalar)

    # An owner the gate hides and then reveals again keeps its full key
    # set: the plan still carries its targets.
    blocked.clear()
    back = _run(adapter, env, steps=1)[-1]
    back_scalar = _run(scalar, env_scalar, steps=1)[-1]
    assert hidden in back.perceived_entities
    assert set(back.perceived_entities[hidden].properties) == hidden_keys
    assert _properties(back) == _properties(back_scalar)
    assert _positions(back) == _positions(back_scalar)


def test_nonfinite_in_hidden_owner_is_ignored() -> None:
    blocked: set = set()

    def gate(entity, _env) -> bool:
        return entity.entity_id not in blocked

    env, adapter = _make(perception_gate=gate)
    env_scalar, scalar = _make(_ScalarAdapter, perception_gate=gate)
    _run(adapter, env, steps=2)
    _run(scalar, env_scalar, steps=2)

    entity_id, body_id = adapter._gather_plan["xpos"].position_rows[0]
    blocked.add(entity_id)
    for target in (env, env_scalar):
        _step(target)
        target.data.xpos[body_id][0] = np.nan

    observation = adapter.get_observation()
    scalar_observation = scalar.get_observation()
    assert entity_id not in observation.perceived_entities
    assert _properties(observation) == _properties(scalar_observation)


def test_nonfinite_raises_same_message() -> None:
    env_a, adapter_a = _make()
    env_b, adapter_b = _make(_ScalarAdapter)

    _run(adapter_a, env_a, steps=2)
    _run(adapter_b, env_b, steps=2)
    assert adapter_a._gather_plan

    messages = []
    for env, adapter in ((env_a, adapter_a), (env_b, adapter_b)):
        _step(env)
        # Poison the state, not a derived array: since the 2026-08-09
        # correctness change the adapter runs a kinematic forward before
        # extraction, so a NaN written into data.xpos would be lawfully
        # recomputed away. A NaN in qpos propagates through forward into
        # every derived position — the divergence the guard exists for.
        env.data.qpos[-1] = np.nan
        with pytest.raises(RuntimeError) as excinfo:
            adapter.get_observation()
        messages.append(str(excinfo.value))

    assert "is non-finite" in messages[0]
    assert messages[0] == messages[1]


@pytest.mark.parametrize(
    "first, second",
    [(_MODEL_A, _MODEL_B), (_MODEL_B, _MODEL_A), (_MODEL_A, _MODEL_C)],
    ids=["shrink", "grow", "same_size"],
)
def test_plan_cleared_on_reset_model_swap(first: str, second: str) -> None:
    # Three shapes of stale plan, verified by deleting the reset clear:
    # shrinking puts indices out of range (IndexError); growing and
    # same_size keep them in range, and the plan writes the previous
    # model's property keys, which the adapter's own metadata parity
    # guard rejects. A same-sized swap cannot produce matching keys with
    # wrong values here -- entity ids and body rows are both assigned in
    # model declaration order, so a reordered model moves the keys too.
    env, adapter = _make(xml=first)
    _run(adapter, env, steps=2)

    env.load(second)
    adapter.reset()
    observation = _run(adapter, env, steps=2)[-1]

    assert adapter._gather_plan["xpos"].flat_indices.size == (
        int(env.model.nbody) * 3
    )

    env_scalar, scalar = _make(_ScalarAdapter, xml=second)
    scalar_observation = _run(scalar, env_scalar, steps=2)[-1]
    assert _properties(observation) == _properties(scalar_observation)
    assert _positions(observation) == _positions(scalar_observation)


_SINK_TABLE_FIELDS = ("qM", "actuator_moment")


def test_sink_table_equals_rebuilt_keys() -> None:
    # qM and actuator_moment stay on the scalar emit path; in steady
    # state they replay a recorded (entity, key) table instead of
    # formatting keys and building PropertyMeta per element. The table
    # carries the emission order, so a mis-ordered flatten pairs a real
    # value with the wrong key -- which only shows up against an adapter
    # that still derives both together.
    env_a, adapter_a = _make()
    env_b, adapter_b = _make(_NoSinkTableAdapter)

    obs_a = _run(adapter_a, env_a, steps=4)
    obs_b = _run(adapter_b, env_b, steps=4)

    assert set(adapter_a._value_only_sinks) >= set(_SINK_TABLE_FIELDS)
    # The table must cover every element of both matrices.
    nv = int(env_a.model.nv)
    assert len(adapter_a._value_only_sinks["qM"]) == nv * nv
    assert len(adapter_a._value_only_sinks["actuator_moment"]) == (
        int(env_a.model.nu) * nv
    )

    for step, (left, right) in enumerate(zip(obs_a, obs_b)):
        assert _properties(left) == _properties(right), step

    # The equality above discriminates actuator_moment's flatten axis
    # (dense is nu x nv, and 3 != 8) but not the value-to-key pairing
    # within it, and cannot discriminate qM's axis at all -- the inertia
    # matrix is symmetric, so a transposed flatten emits the same values.
    # Reversing the table is the direct check that the pairing is what
    # reaches the observation.
    adapter_a._value_only_sinks["actuator_moment"].reverse()
    reversed_obs = _run(adapter_a, env_a, steps=1)[-1]
    reference_obs = _run(adapter_b, env_b, steps=1)[-1]
    assert _properties(reversed_obs) != _properties(reference_obs)

    # And a recorded key no full pass registered must not slip through as
    # a property with no metadata behind it.
    table = adapter_a._value_only_sinks["actuator_moment"]
    table[0] = (table[0][0], "actuator_moment_never_registered")
    with pytest.raises(RuntimeError, match="without metadata"):
        _run(adapter_a, env_a, steps=1)


def test_no_sink_table_while_the_gate_hides_an_actuator_owner() -> None:
    # The table is only safe because a full pass that omitted an owner
    # never records one: its pairs would include sinks with no registered
    # metadata, and the value-only replay passes no metadata at all. An
    # owner hidden at reset and revealed later is the path that proves it.
    blocked: set = set()

    def gate(entity, _env) -> bool:
        return entity.entity_id not in blocked

    probe_env, probe = _make()
    _run(probe, probe_env, steps=1)
    # Every owner in a recorded table was perceived by the pass that
    # recorded it, so the first entry is a gateable actuator owner.
    hidden = probe._value_only_sinks["actuator_moment"][0][0]

    blocked.add(hidden)
    env, adapter = _make(perception_gate=gate)
    env_reference, reference = _make(_NoSinkTableAdapter, perception_gate=gate)
    _run(adapter, env, steps=2)
    _run(reference, env_reference, steps=2)
    assert "actuator_moment" not in adapter._value_only_sinks

    blocked.clear()
    observations = _run(adapter, env, steps=3)
    reference_observations = _run(reference, env_reference, steps=3)
    assert hidden in observations[-1].perceived_entities
    assert "actuator_moment" in adapter._value_only_sinks
    assert _properties(observations[-1]) == _properties(
        reference_observations[-1]
    )


def test_sink_table_rebuilt_on_reset_model_swap() -> None:
    # Pins the rebuild, not the reset clear: the first post-reset pass is
    # a full one and overwrites the table outright, so deleting
    # _value_only_sinks.clear() leaves this green. The clear still earns
    # its line for the one path that misses the overwrite -- a new model
    # whose first pass omits an owner records nothing -- and no test
    # reaches that (Task 4 report, concerns).
    env, adapter = _make(xml=_MODEL_A)
    _run(adapter, env, steps=2)
    assert adapter._value_only_sinks["qM"]

    env.load(_MODEL_B)
    adapter.reset()
    observation = _run(adapter, env, steps=3)[-1]
    assert len(adapter._value_only_sinks["qM"]) == int(env.model.nv) ** 2

    env_reference, reference = _make(_NoSinkTableAdapter, xml=_MODEL_B)
    reference_observation = _run(reference, env_reference, steps=3)[-1]
    assert _properties(observation) == _properties(reference_observation)


def test_no_plan_when_gate_hides_an_owner_during_the_first_pass() -> None:
    # The dangerous direction of the perception gate: an owner hidden
    # before the field's first full pass never enters the capture, so a
    # plan built from it would freeze that owner out of the field's sinks
    # for the whole episode. Correctness rests on the pass's owner
    # omission refusing the plan -- the capture-count check alone cannot
    # see it, because the omitted emission drops out of both counts.
    blocked: set = set()

    def gate(entity, _env) -> bool:
        return entity.entity_id not in blocked

    probe_env, probe = _make()
    _run(probe, probe_env, steps=1)
    hidden = probe._gather_plan["xpos"].position_rows[0][0]

    # Blocked before the adapter exists: reset() runs the first full
    # pass, so gating afterwards would be too late to matter.
    blocked.add(hidden)
    env, adapter = _make(perception_gate=gate)
    # Stepped in lockstep with the gated adapter so the final comparison
    # is between two envs at the same physics state.
    env_scalar, scalar = _make(_ScalarAdapter)
    _run(adapter, env, steps=2)
    _run(scalar, env_scalar, steps=2)
    assert hidden not in adapter.get_observation().perceived_entities
    for field_name in ("xpos", "xquat", "xmat"):
        assert field_name not in adapter._gather_plan
        refusal = adapter.field_materialization[field_name].plan_refusal
        assert refusal is not None and "an owner was omitted" in refusal
        # The refusal is a performance note, not a capability one. The
        # omission itself belongs in reason (it predates this) but the
        # refusal must not leak there -- the field contract renders
        # reason as the field's capability line.
        assert "no gather plan" not in (
            adapter.field_materialization[field_name].reason or ""
        )

    # Unhiding restores the owner, and the field replans on the next
    # clean full pass -- the new sink drops it off the steady path first.
    blocked.clear()
    observations = _run(adapter, env, steps=3)
    scalar_observations = _run(scalar, env_scalar, steps=3)
    assert hidden in observations[-1].perceived_entities
    assert "xpos" in adapter._gather_plan
    assert adapter.field_materialization["xpos"].plan_refusal is None
    assert _properties(observations[-1]) == _properties(
        scalar_observations[-1]
    )
    assert _positions(observations[-1]) == _positions(
        scalar_observations[-1]
    )
