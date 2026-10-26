"""
MuJoCo sensor extraction: name-collision guard and gripper-site
ownership.

Companion to the touch-sensor tests in tests/test_mujoco_property_meta.py
(``test_touch_sensor_attached_to_owning_entity`` /
``test_orphan_sensors_get_synthetic_entity``), covering two sensors
review findings:

  - two sensors whose names sanitize to the same property token must
    raise at reset, not silently drop one sensor's data
    (``_sensor_properties`` / ``_sensor_property_meta``);
  - a sensor defined on the gripper's own site belongs to the gripper
    entity -- not whatever body/subtree that site's body happens to
    fall under -- and the gripper's property_metadata parity must cover
    it (``_sensor_owner_entity_id`` / ``_gripper_property_meta``).
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


# ---------------------------------------------------------------------------
# Name-collision guard
# ---------------------------------------------------------------------------

_COLLISION_XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0.1">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
      <site name="site_one" pos="0 0 0" size="0.01"/>
      <site name="site_two" pos="0 0.02 0" size="0.01"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="a.b" site="site_one"/>
    <touch name="a_b" site="site_two"/>
  </sensor>
</mujoco>
"""


def test_colliding_sensor_names_receive_indexed_tokens():
    adapter = MujocoAdapter(
        _Env(_COLLISION_XML),
        body_rules=[BodyRule(match="obj", cls="object")],
    )

    observation = adapter.reset()
    properties = observation.perceived_entities[
        adapter.name_to_id["obj"]
    ].properties
    assert "sensordata_a_b_model_index_0" in properties
    assert "sensordata_a_b_model_index_1" in properties


# ---------------------------------------------------------------------------
# Gripper-site sensor ownership
# ---------------------------------------------------------------------------

_GRIPPER_SENSOR_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 0.2">
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.2 0 0"
            size="0.02" mass="0.5"/>
      <site name="grip_site" pos="0.2 0 0" size="0.01"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="grip_touch" site="grip_site"/>
  </sensor>
</mujoco>
"""


def _make_gripper_adapter():
    adapter = MujocoAdapter(
        _Env(_GRIPPER_SENSOR_XML),
        body_rules=[
            BodyRule(
                match=lambda name: name == "arm",
                cls="robot",
                joint_scope="subtree",
            ),
        ],
        gripper_site="grip_site",
    )
    observation = adapter.reset()
    by_class = {
        entity.entity_class: entity
        for entity in observation.perceived_entities.values()
    }
    return adapter, by_class


def test_gripper_site_sensor_attaches_to_gripper_not_robot():
    # grip_site's body ("arm") is inside the robot's subtree-scoped
    # pose_body_ids, so a naive body walk would (mis)attach the sensor
    # to "robot" instead of the gripper site it is literally defined on.
    adapter, by_class = _make_gripper_adapter()
    robot = by_class["robot"]
    gripper = by_class["gripper"]

    assert "sensordata_grip_touch" in gripper.properties
    assert "sensordata_grip_touch" not in robot.properties


def test_gripper_site_sensor_metadata_parity_holds():
    adapter, by_class = _make_gripper_adapter()
    gripper = by_class["gripper"]

    meta = adapter.property_metadata(gripper.entity_id)

    assert meta["sensordata_grip_touch"].group == "sensors: touch"
    assert meta["sensordata_grip_touch"].note == "N"


def test_user_sensor_values_are_exposed_with_opaque_semantics():
    env = _Env(
        """
        <mujoco>
          <worldbody>
            <body name="obj">
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
          <sensor>
            <user name="custom_state" dim="2"/>
          </sensor>
        </mujoco>
        """
    )
    env.data.sensordata[:] = [3.25, -2.5]
    adapter = MujocoAdapter(env, body_rules=())
    observation = adapter.reset()

    sensor_id = adapter._orphan_sensor_entity_id
    assert sensor_id is not None
    properties = observation.perceived_entities[sensor_id].properties
    metadata = adapter.property_metadata(sensor_id)
    for index, expected in enumerate((3.25, -2.5)):
        key = "sensordata_custom_state_{}".format(index)
        assert properties[key] == pytest.approx(expected)
        assert metadata[key].note is None
        assert {"semantics:opaque", "unit:unresolved"} <= set(
            metadata[key].tags
        )
        assert {
            "sensor_object_type:unknown",
            "sensor_object_index:-1",
            "sensor_reference_type:unknown",
            "sensor_reference_index:-1",
        } <= set(metadata[key].tags)
    assert adapter.field_materialization["sensordata"].scene_applicable
