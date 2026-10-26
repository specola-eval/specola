"""Dynamic contact-pair entity tests for the shared MuJoCo adapter.

Contact rows in ``MjData`` have unstable indices: one body pair occupied
row indices 25 through 164 over a single LIBERO episode. The adapter
therefore never exposes rows. It materializes one dynamic entity per
unordered body pair, keyed on ``model.geom_bodyid`` of the contact's
geoms, carrying scalar force aggregations computed via
``mj_contactForce``. These tests pin the identity, aggregation,
lifecycle, and discovery semantics of that representation.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import (
    CONTACT_ENTITY_CLASS,
    CONTACT_PROPERTY_KEYS,
    ROBOT_CONTACT_ENTITY_CLASS,
    BodyRule,
    MujocoAdapter,
    _contact_pair_key,
)
from specola.capabilities import VocabularyMode

pytestmark = pytest.mark.mujoco


_CONTACT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"
            mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

_TWO_BODY_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"
            mass="2"/>
    </body>
    <body name="probe" pos="1 1 0.3">
      <freejoint name="probe_free"/>
      <geom name="probe_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _env(xml: str = _CONTACT_XML) -> SimpleNamespace:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def _adapter(env=None, **kwargs) -> MujocoAdapter:
    if env is None:
        env = _env()
    kwargs.setdefault(
        "body_rules",
        (BodyRule(match="box", cls="object", joint_scope="body"),),
    )
    return MujocoAdapter(env, **kwargs)


def _settle(env, steps: int = 400) -> None:
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)


def _entities_of_class(observation, entity_class):
    return {
        entity_id: state
        for entity_id, state in observation.perceived_entities.items()
        if state.entity_class == entity_class
    }


def _retained_pair_rows(model, data):
    """Reference filter: retained contact rows grouped by body pair."""
    rows = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        if int(contact.exclude) != 0 or int(contact.efc_address) < 0:
            continue
        bodies = tuple(
            sorted(
                int(model.geom_bodyid[int(contact.geom[side])])
                for side in (0, 1)
            )
        )
        rows.setdefault(bodies, []).append(index)
    return rows


# ---------------------------------------------------------------------------
# Discovery and capability schema
# ---------------------------------------------------------------------------


def test_capabilities_declare_contact_schema_before_first_contact():
    adapter = _adapter()
    adapter.reset()

    capabilities = adapter.capabilities
    for entity_class in (
        CONTACT_ENTITY_CLASS,
        ROBOT_CONTACT_ENTITY_CLASS,
    ):
        declared = capabilities.components_by_class[entity_class]
        assert CONTACT_PROPERTY_KEYS <= set(declared)
        assert (
            capabilities.component_mode_by_class[entity_class]
            is VocabularyMode.CLOSED
        )


def test_no_contact_entities_in_separated_reset_snapshot():
    adapter = _adapter()
    observation = adapter.reset()

    assert _entities_of_class(observation, CONTACT_ENTITY_CLASS) == {}
    assert (
        _entities_of_class(observation, ROBOT_CONTACT_ENTITY_CLASS)
        == {}
    )


# ---------------------------------------------------------------------------
# Appearance and aggregation semantics
# ---------------------------------------------------------------------------


def test_settled_box_produces_one_pair_entity_with_exact_keys():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    _settle(env)

    observation = adapter.get_observation()
    contacts = _entities_of_class(observation, CONTACT_ENTITY_CLASS)

    reference = _retained_pair_rows(env.model, env.data)
    assert len(contacts) == len(reference) == 1

    (state,) = contacts.values()
    assert set(state.properties) == set(CONTACT_PROPERTY_KEYS)

    (pair_rows,) = reference.values()
    assert state.properties["contact_point_count"] == len(pair_rows)

    # The floor geom sits on the world body, so the pair label names
    # "world" rather than the geom.
    label = adapter.entity_labels[state.entity_id]
    assert "box" in label and "world" in label


def test_force_aggregations_match_direct_mj_contact_force():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    _settle(env)

    observation = adapter.get_observation()
    (state,) = _entities_of_class(
        observation, CONTACT_ENTITY_CLASS
    ).values()

    (rows,) = _retained_pair_rows(env.model, env.data).values()
    result = np.zeros(6)
    normals = []
    magnitudes = []
    world_vectors = []
    for index in rows:
        mujoco.mj_contactForce(env.model, env.data, index, result)
        contact = env.data.contact[index]
        rotation = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        world = rotation.T @ result[0:3]
        first_body = int(
            env.model.geom_bodyid[int(contact.geom[0])]
        )
        second_body = int(
            env.model.geom_bodyid[int(contact.geom[1])]
        )
        if second_body < first_body:
            world = -world
        normals.append(float(result[0]))
        magnitudes.append(float(np.linalg.norm(result[0:3])))
        world_vectors.append(world)

    properties = state.properties
    assert properties["contact_normal_force_max"] == pytest.approx(
        max(normals)
    )
    assert properties["contact_normal_force_sum"] == pytest.approx(
        sum(normals)
    )
    assert properties["contact_point_force_max"] == pytest.approx(
        max(magnitudes)
    )
    assert properties["contact_resultant_force"] == pytest.approx(
        float(np.linalg.norm(np.sum(world_vectors, axis=0)))
    )

    # A settled 2 kg box rests its weight on the floor: the summed
    # normal force brackets m*g.
    weight = 2.0 * 9.81
    assert 0.5 * weight < properties["contact_normal_force_sum"]
    assert properties["contact_normal_force_sum"] < 2.0 * weight

    assert properties["contact_min_dist"] <= 0.0

    # Independent ground truth for the world-frame resultant: the box's
    # only external load is this contact, so MuJoCo's own cfrc_ext force
    # half (components 3:6) must equal the pair resultant. This check
    # does not share the adapter's frame/sign arithmetic, so a
    # canonicalization error cannot cancel out of both sides.
    mujoco.mj_rnePostConstraint(env.model, env.data)
    box_body = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "box"
    )
    external_force = float(
        np.linalg.norm(env.data.cfrc_ext[box_body][3:6])
    )
    assert properties["contact_resultant_force"] == pytest.approx(
        external_force, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Robot participant classification
# ---------------------------------------------------------------------------


def test_robot_body_policy_selects_robot_contact_class():
    env = _env()
    adapter = _adapter(env, robot_bodies=("box",))
    adapter.reset()
    _settle(env)

    observation = adapter.get_observation()
    robot_contacts = _entities_of_class(
        observation, ROBOT_CONTACT_ENTITY_CLASS
    )
    assert len(robot_contacts) == 1
    assert _entities_of_class(observation, CONTACT_ENTITY_CLASS) == {}


# ---------------------------------------------------------------------------
# Identity across disappearance, reappearance, and reset
# ---------------------------------------------------------------------------


def test_pair_identity_survives_separation_with_exact_zero_force():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    _settle(env)

    observation = adapter.get_observation()
    (first_id,) = _entities_of_class(
        observation, CONTACT_ENTITY_CLASS
    )

    # Lift the box well clear of the floor; the pair separates.
    env.data.qpos[0:3] = (0.0, 0.0, 1.0)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert _retained_pair_rows(env.model, env.data) == {}

    separated = adapter.get_observation()
    contacts = _entities_of_class(separated, CONTACT_ENTITY_CLASS)

    # The pair persists: a separated pair's contact force is exactly
    # zero, which is measured state, not fabrication.
    assert set(contacts) == {first_id}
    state = contacts[first_id]
    assert state.properties["contact_point_count"] == 0
    assert state.properties["contact_normal_force_sum"] == 0.0
    assert state.properties["contact_point_force_max"] == 0.0
    assert "contact_min_dist" not in state.properties
    separated_sinks = adapter.field_materialization["contact"].emitted_keys
    assert "entity[{}].properties.contact_min_dist".format(first_id) not in separated_sinks

    # Drop it back; the same pair key reuses the same entity ID.
    env.data.qpos[0:3] = (0.0, 0.0, 0.06)
    mujoco.mj_forward(env.model, env.data)
    _settle(env, steps=200)

    recontact = adapter.get_observation()
    contacts = _entities_of_class(recontact, CONTACT_ENTITY_CLASS)
    assert set(contacts) == {first_id}
    assert contacts[first_id].properties["contact_point_count"] > 0
    assert "entity[{}].properties.contact_min_dist".format(first_id) in adapter.field_materialization["contact"].emitted_keys


def test_rejected_contact_rolls_back_identity_and_metadata():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    adapter.set_extraction_scope(
        {CONTACT_ENTITY_CLASS: frozenset({"contact_normal_force_sum"})}
    )
    canonical_metadata = adapter.property_metadata
    rejected_id = None
    state_names = (
        "_contact_pairs", "_next_runtime_entity_id", "name_to_id",
        "entity_labels", "_entity_class_by_id",
        "_materialized_property_meta", "_merged_meta_cache",
        "_field_materialization",
    )
    def contract_state():
        return tuple(
            deepcopy(getattr(adapter, name)) for name in state_names
        )
    def reject_new_contact(entity_id):
        nonlocal rejected_id
        metadata = dict(canonical_metadata(entity_id))
        if (
            entity_id not in adapter.last_observation.perceived_entities
            and "contact_normal_force_sum" in metadata
        ):
            rejected_id = entity_id
            metadata.pop("contact_normal_force_sum")
        return metadata
    adapter.property_metadata = reject_new_contact
    for _ in range(400):
        mujoco.mj_step(env.model, env.data)
        prior = adapter.last_observation
        accepted_state = contract_state()
        try:
            adapter.get_observation()
        except RuntimeError as error:
            assert "Property metadata mismatch" in str(error)
            break
    else:
        pytest.fail("no contact rejection in 400 steps")
    adapter.property_metadata = canonical_metadata
    assert rejected_id is not None
    assert adapter.last_observation is prior
    assert contract_state() == accepted_state
    assert adapter.property_metadata(rejected_id) == {}
    env.data.qpos[2] = 0.4
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert rejected_id not in adapter.get_observation().perceived_entities
    env.data.qpos[2] = 0.08
    mujoco.mj_forward(env.model, env.data)
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)
        accepted_contact = adapter.get_observation()
        if rejected_id in accepted_contact.perceived_entities:
            break
    else:
        pytest.fail("no later accepted contact in 200 steps")
    expected_keys = {"contact_normal_force_sum"}
    assert set(accepted_contact.perceived_entities[rejected_id].properties) == expected_keys
    assert set(adapter.property_metadata(rejected_id)) == expected_keys

    adapter.set_extraction_scope({
        CONTACT_ENTITY_CLASS: frozenset(expected_keys | {"contact_min_dist"})
    })
    accepted_state = contract_state()

    def reject_existing_contact(entity_id):
        metadata = dict(canonical_metadata(entity_id))
        if entity_id == rejected_id:
            metadata.pop("contact_normal_force_sum")
        return metadata

    adapter.property_metadata = reject_existing_contact
    with pytest.raises(RuntimeError, match="Property metadata mismatch"):
        adapter.get_observation()
    adapter.property_metadata = canonical_metadata
    assert adapter.last_observation is accepted_contact
    assert contract_state() == accepted_state
    assert set(adapter._materialized_property_meta[rejected_id]) == expected_keys
    adapter.set_extraction_scope({CONTACT_ENTITY_CLASS: frozenset(expected_keys)})

    env.data.qpos[2] = 0.4
    mujoco.mj_forward(env.model, env.data)
    retained = adapter.get_observation()
    assert retained.perceived_entities[rejected_id].properties[
        "contact_normal_force_sum"
    ] == 0.0
    assert set(adapter.property_metadata(rejected_id)) == expected_keys


def test_distinct_pairs_get_distinct_ids_and_reset_clears_them():
    env = _env(_TWO_BODY_XML)
    adapter = _adapter(
        env,
        body_rules=(
            BodyRule(match="box", cls="object", joint_scope="body"),
            BodyRule(match="probe", cls="object", joint_scope="body"),
        ),
    )
    adapter.reset()

    # Land both bodies so two distinct pairs exist.
    env.data.qpos[0:3] = (0.0, 0.0, 0.05)
    env.data.qpos[7:10] = (1.0, 1.0, 0.05)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _settle(env, steps=200)

    observation = adapter.get_observation()
    contacts = _entities_of_class(observation, CONTACT_ENTITY_CLASS)
    assert len(contacts) == 2
    assert len({state.entity_id for state in contacts.values()}) == 2

    labels = {
        adapter.entity_labels[entity_id] for entity_id in contacts
    }
    assert len(labels) == 2

    # Separate both bodies, then reset: the adapter's reset clears the
    # pair registry, and with no live contact in the new episode's
    # first snapshot no contact entity exists.
    env.data.qpos[0:3] = (0.0, 0.0, 1.0)
    env.data.qpos[7:10] = (1.0, 1.0, 1.0)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    reset_observation = adapter.reset()
    assert (
        _entities_of_class(reset_observation, CONTACT_ENTITY_CLASS)
        == {}
    )


# ---------------------------------------------------------------------------
# Flex guard and metadata parity
# ---------------------------------------------------------------------------


def test_flex_participant_fails_loud():
    env = _env()

    class _FlexContact:
        geom = (-1, 0)
        exclude = 0
        efc_address = 0

    with pytest.raises(RuntimeError, match="flex"):
        _contact_pair_key(env.model, _FlexContact())


def test_contact_property_metadata_declares_units():
    env = _env()
    adapter = _adapter(env)
    adapter.reset()
    _settle(env)

    observation = adapter.get_observation()
    (state,) = _entities_of_class(
        observation, CONTACT_ENTITY_CLASS
    ).values()

    metadata = adapter.property_metadata(state.entity_id)
    for key in state.properties:
        assert key in metadata

    force_keys = (
        "contact_normal_force_max",
        "contact_normal_force_sum",
        "contact_point_force_max",
        "contact_resultant_force",
    )
    for key in force_keys:
        assert metadata[key].note == "N"
    assert metadata["contact_min_dist"].note == "m"
