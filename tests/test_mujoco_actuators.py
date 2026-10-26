"""
MuJoCo actuator extraction: name-collision guard.

Companion to the joint-transmission actuator tests in
tests/test_mujoco_property_meta.py
(``test_actuator_keys_attached_via_joint_transmission`` /
``test_orphan_actuators_get_synthetic_entity``), covering the actuator
analogue of the sensors name-collision review finding
(tests/test_mujoco_sensors.py, ``test_colliding_sensor_names_raise_at_reset``):
two actuators whose names sanitize to the same property token must raise
at reset, not silently drop one actuator's data
(``_actuator_properties`` / ``_actuator_property_meta``).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import BodyRule, MujocoAdapter


class _Env:
    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


_COLLISION_XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0.1">
      <joint name="obj_hinge" type="hinge" axis="0 0 1"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="a.b" joint="obj_hinge"/>
    <motor name="a_b" joint="obj_hinge"/>
  </actuator>
</mujoco>
"""


def test_colliding_actuator_names_receive_indexed_tokens():
    adapter = MujocoAdapter(
        _Env(_COLLISION_XML),
        body_rules=[BodyRule(match="obj", cls="object")],
    )

    observation = adapter.reset()
    properties = observation.perceived_entities[
        adapter.name_to_id["obj"]
    ].properties
    assert "ctrl_a_b_model_index_0" in properties
    assert "ctrl_a_b_model_index_1" in properties


def test_spatial_tendon_motor_uses_tendon_coordinate_units():
    env = _Env(
        """
        <mujoco>
          <worldbody>
            <site name="anchor"/>
            <body name="obj" pos="0.5 0 0">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05"/>
              <site name="moving"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="rope">
              <site site="anchor"/>
              <site site="moving"/>
            </spatial>
          </tendon>
          <actuator>
            <motor name="winch" tendon="rope"/>
          </actuator>
        </mujoco>
        """
    )
    env.data.ctrl[0] = 2.0
    mujoco.mj_forward(env.model, env.data)
    adapter = MujocoAdapter(
        env,
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    observation = adapter.reset()

    actuator_id = adapter._orphan_actuator_entity_id
    assert actuator_id is not None
    properties = observation.perceived_entities[actuator_id].properties
    metadata = adapter.property_metadata(actuator_id)
    expected_units = {
        "actuator_force_winch": "N",
        "actuator_length_winch": "m",
        "actuator_velocity_winch": "m/s",
    }
    assert {"ctrl_winch", *expected_units} <= properties.keys()
    assert metadata["ctrl_winch"].group == "actuator force command"
    for key, unit in expected_units.items():
        assert metadata[key].note == unit
        assert "unit:unresolved" not in metadata[key].tags
        assert {
            "actuator_transmission_type:tendon",
            "actuator_transmission_object0_index:0",
        } <= set(metadata[key].tags)
    assert properties["ctrl_winch"] == pytest.approx(2.0)
    assert properties["actuator_length_winch"] == pytest.approx(0.5)
