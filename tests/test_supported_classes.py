"""
Closed-vocabulary adapters must declare the synthetic orphan-sensor and
orphan-actuator classes alongside their other entity classes.

MujocoAdapter attaches sensors with no resolvable body/site owner to a
synthetic entity of class "sensors" (adapters/mujoco.py,
``_ORPHAN_SENSOR_ENTITY_CLASS``), and actuators with no resolvable body
owner to a synthetic entity of class "actuators"
(``_ORPHAN_ACTUATOR_ENTITY_CLASS``). LIBEROAdapter and RoboCasaAdapter
both declare a *closed* ``supported_classes`` vocabulary, so any reset
that happens to produce an orphan sensor or orphan actuator is rejected
by ``SimulatorAdapter._validate_entity_entry`` unless "sensors" /
"actuators" are members of that vocabulary.

Both adapters compute ``supported_classes`` purely from instance
attributes (gripper_class / body_rules / regions) rather than by
querying a live simulator, so the fix is pinned here without
constructing an env -- the same bare-adapter pattern used in
tests/test_gripper_property_meta.py.

This module also pins the global MuJoCo availability rule:
``RoboCasaAdapter`` has no field-family selector, and every curated joint
family present on its live ``MjData`` is exposed eagerly.
"""

from __future__ import annotations

import pytest

from specola.adapters.mujoco import (
    _CURATED_FIELD_OWNER_ENTITY_CLASSES,
    _ORPHAN_ACTUATOR_ENTITY_CLASS,
    _ORPHAN_SENSOR_ENTITY_CLASS,
    _SIGNAL_ADDRESSES,
)


def _bare_adapter(cls):
    """A MujocoAdapter subclass instance with __init__ skipped."""
    return object.__new__(cls)


def test_libero_supported_classes_includes_sensors():
    from specola.adapters.libero import LIBEROAdapter

    adapter = _bare_adapter(LIBEROAdapter)
    adapter.gripper_class = "gripper"

    assert _ORPHAN_SENSOR_ENTITY_CLASS in adapter.supported_classes
    assert _ORPHAN_ACTUATOR_ENTITY_CLASS in adapter.supported_classes
    assert adapter.supported_classes == frozenset(
        {
            "robot",
            "object",
            "fixture",
            "region",
            "object_relation",
            "object_region_relation",
            "gripper",
            "sensors",
            "actuators",
        }
        | _CURATED_FIELD_OWNER_ENTITY_CLASSES
    )


def test_robocasa_supported_classes_includes_sensors():
    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = _bare_adapter(RoboCasaAdapter)
    adapter.body_rules = []
    adapter.gripper_class = "gripper"
    adapter.regions = []

    assert _ORPHAN_SENSOR_ENTITY_CLASS in adapter.supported_classes
    assert _ORPHAN_ACTUATOR_ENTITY_CLASS in adapter.supported_classes
    assert adapter.supported_classes == frozenset(
        {
            "gripper",
            "region",
            "sensors",
            "actuators",
        }
        | _CURATED_FIELD_OWNER_ENTITY_CLASSES
    )


def test_robocasa_supported_classes_includes_sensors_with_custom_rules():
    """The fix must survive custom body_rules/regions, not just defaults."""
    from specola.adapters.mujoco import BodyRule, RegionSpec
    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = _bare_adapter(RoboCasaAdapter)
    adapter.body_rules = [BodyRule(match="foo", cls="widget")]
    adapter.gripper_class = "gripper"
    adapter.regions = [
        RegionSpec(
            name="zone",
            source="fixed",
            center=(0.0, 0.0, 0.0),
            half_extents=(0.1, 0.1, 0.1),
            cls="zone",
        )
    ]

    assert _ORPHAN_SENSOR_ENTITY_CLASS in adapter.supported_classes
    assert _ORPHAN_ACTUATOR_ENTITY_CLASS in adapter.supported_classes
    assert adapter.supported_classes == frozenset(
        {
            "widget",
            "gripper",
            "zone",
            "sensors",
            "actuators",
        }
        | _CURATED_FIELD_OWNER_ENTITY_CLASSES
    )


def test_robocasa_joint_signals_match_global_live_field_set():
    """
    A real tiny model pins that RoboCasa exposes the shared live set, with
    no simulator-local subset override.
    """
    mujoco = pytest.importorskip("mujoco")

    xml = """
    <mujoco>
      <worldbody>
        <body name="mobilebase0_base">
          <joint name="mobilebase0_joint_mobile_forward"
                 type="slide" axis="1 0 0"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="mobilebase0_motor"
               joint="mobilebase0_joint_mobile_forward"/>
      </actuator>
    </mujoco>
    """

    class _TinyRoboCasaEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = RoboCasaAdapter(_TinyRoboCasaEnv())

    assert set(adapter.joint_signals) == set(_SIGNAL_ADDRESSES)
