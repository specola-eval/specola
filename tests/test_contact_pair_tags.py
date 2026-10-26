"""Structural tags on dynamic contact-pair entities (D20 Part 1).

The shared MuJoCo adapter asks a narrow subclass hook,
``_contact_pair_tags(body_pair)``, for structural tags at the moment a
contact pair is first materialized. Tags are stored on the pair record,
ride every contact property key through per-entity property metadata,
and never change for the lifetime of the pair. The base adapter has no
simulator policy, so its hook returns no tags and today's metadata is
unchanged.

Role *content* for RoboCasa (body-ownership map, ``role:*`` values) is
covered by the RoboCasa role tests; here the contract of the shared
mechanism is pinned with a synthetic model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import (
    CONTACT_ENTITY_CLASS,
    ROBOT_CONTACT_ENTITY_CLASS,
    BodyRule,
    MujocoAdapter,
)

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


class _TaggingAdapter(MujocoAdapter):
    """Test double: tags every pair and counts hook invocations."""

    def __init__(self, *args, **kwargs):
        self.hook_calls = []
        super().__init__(*args, **kwargs)

    def _contact_pair_tags(
        self, body_pair: Tuple[int, int]
    ) -> Tuple[str, ...]:
        self.hook_calls.append(body_pair)
        return ("role:distractor",)


def _env() -> SimpleNamespace:
    model = mujoco.MjModel.from_xml_string(_CONTACT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def _settle(env, steps: int = 400) -> None:
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)


def _pair_states(observation):
    return {
        entity_id: state
        for entity_id, state in observation.perceived_entities.items()
        if state.entity_class
        in (CONTACT_ENTITY_CLASS, ROBOT_CONTACT_ENTITY_CLASS)
    }


def _make(adapter_cls, env):
    return adapter_cls(
        env,
        body_rules=(
            BodyRule(match="box", cls="object", joint_scope="body"),
        ),
    )


def test_base_adapter_pairs_carry_no_tags():
    env = _env()
    adapter = _make(MujocoAdapter, env)
    adapter.reset()
    _settle(env)
    observation = adapter.get_observation()

    pairs = _pair_states(observation)
    assert pairs, "expected a settled floor contact pair"
    for entity_id, state in pairs.items():
        metadata = adapter.property_metadata(entity_id)
        for key in state.properties:
            assert metadata[key].tags == ()


def test_hook_tags_ride_every_contact_property_key():
    env = _env()
    adapter = _make(_TaggingAdapter, env)
    adapter.reset()
    _settle(env)
    observation = adapter.get_observation()

    pairs = _pair_states(observation)
    assert pairs
    for entity_id, state in pairs.items():
        metadata = adapter.property_metadata(entity_id)
        assert state.properties, "pair must emit contact keys"
        for key in state.properties:
            assert "role:distractor" in metadata[key].tags


def test_hook_called_once_per_pair_and_tags_stay_stable():
    env = _env()
    adapter = _make(_TaggingAdapter, env)
    adapter.reset()
    _settle(env)
    adapter.get_observation()
    first_calls = list(adapter.hook_calls)
    assert len(first_calls) == len(set(first_calls))

    # Later observations reuse the recorded pair; the hook is not
    # re-consulted and the tags do not change.
    for _ in range(3):
        mujoco.mj_step(env.model, env.data)
        observation = adapter.get_observation()
    assert adapter.hook_calls == first_calls

    for entity_id, state in _pair_states(observation).items():
        metadata = adapter.property_metadata(entity_id)
        for key in state.properties:
            assert "role:distractor" in metadata[key].tags


def test_observation_carries_component_tags_for_tagged_pairs():
    env = _env()
    adapter = _make(_TaggingAdapter, env)
    adapter.reset()
    _settle(env)
    observation = adapter.get_observation()

    pairs = _pair_states(observation)
    assert pairs
    for entity_id, state in pairs.items():
        entry = observation.component_tags[entity_id]
        assert set(entry) == set(state.properties)
        assert all(
            "role:distractor" in tags for tags in entry.values()
        )


def test_base_adapter_observation_component_tags_empty():
    env = _env()
    adapter = _make(MujocoAdapter, env)
    adapter.reset()
    _settle(env)
    observation = adapter.get_observation()
    assert _pair_states(observation)
    # Geometry distance pairs carry structural body tags from the shared
    # reader; contact pairs of the base adapter still carry none.
    contact_ids = {entity_id for entity_id, entity in observation.perceived_entities.items()
                   if entity.entity_class != "geom_pair"}
    assert {entity_id for entity_id in observation.component_tags if entity_id in contact_ids} == set()


def test_reset_clears_pair_tag_registry():
    env = _env()
    adapter = _make(_TaggingAdapter, env)
    adapter.reset()
    _settle(env)
    adapter.get_observation()
    assert adapter.hook_calls

    calls_before_reset = len(adapter.hook_calls)
    adapter.reset()
    observation = adapter.get_observation()
    # Fresh episode over still-touching simulator state: the registry
    # was cleared, so the pair is re-materialized and the hook is
    # consulted again for it.
    pairs = _pair_states(observation)
    assert pairs
    assert len(adapter.hook_calls) > calls_before_reset
    for entity_id, state in pairs.items():
        metadata = adapter.property_metadata(entity_id)
        for key in state.properties:
            assert "role:distractor" in metadata[key].tags
