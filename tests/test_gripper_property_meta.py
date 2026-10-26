"""
Gripper property metadata: the metadata/property parity invariant
(metadata keys == property keys) must hold for the gripper entity on
every MuJoCo-derived adapter, including the RoboSuite-flavored extras
that LIBERO and RoboCasa add on top of the base site_xpos_* keys.

These hooks are pure dict code -- no live simulator is required.
"""

from __future__ import annotations

from specola.adapters.mujoco import MujocoAdapter, _SiteRecord


def _bare_adapter(cls):
    """A MujocoAdapter subclass instance with __init__ skipped."""
    return object.__new__(cls)


# ---------------------------------------------------------------------------
# Base MujocoAdapter: site_xpos_* metadata only
# ---------------------------------------------------------------------------

def test_base_gripper_hook_covers_site_xpos_keys():
    adapter = _bare_adapter(MujocoAdapter)
    adapter._gripper = _SiteRecord(
        entity_id=4, site_name="grip_site", site_id=0,
        entity_class="gripper",
    )
    meta = adapter._gripper_property_meta()
    assert set(meta) == {
        "site_xpos_grip_site_0",
        "site_xpos_grip_site_1",
        "site_xpos_grip_site_2",
    }
    assert meta["site_xpos_grip_site_0"].group == "site position"
    assert meta["site_xpos_grip_site_0"].note == "m"


def test_base_gripper_hook_empty_without_a_gripper():
    adapter = _bare_adapter(MujocoAdapter)
    adapter._gripper = None
    assert adapter._gripper_property_meta() == {}


def test_property_metadata_gripper_branch_uses_the_hook():
    adapter = _bare_adapter(MujocoAdapter)
    adapter._bodies = []
    adapter._gripper = _SiteRecord(
        entity_id=4, site_name="grip_site", site_id=0,
        entity_class="gripper",
    )
    assert (
        adapter.property_metadata(4)
        == adapter._gripper_property_meta()
    )


# ---------------------------------------------------------------------------
# LIBEROAdapter: finger-joint extras
# ---------------------------------------------------------------------------

class _FakeGripper:
    def __init__(self, qpos):
        self.qpos = qpos


class _FakeRobot:
    def __init__(self, gripper=None, **extra):
        self.gripper = gripper
        for name, value in extra.items():
            setattr(self, name, value)


class _FakeEnv:
    def __init__(self, robots=None):
        self.robots = robots


def test_libero_gripper_meta_matches_property_keys():
    from specola.adapters.libero import LIBEROAdapter

    adapter = _bare_adapter(LIBEROAdapter)
    adapter._gripper = _SiteRecord(
        entity_id=9, site_name="gripper0_grip_site", site_id=0,
        entity_class="gripper",
    )
    adapter.env = _FakeEnv(
        robots=[_FakeRobot(gripper=_FakeGripper(qpos=[0.02, -0.02]))]
    )

    meta = adapter._gripper_property_meta()

    assert set(meta) == {
        "site_xpos_gripper0_grip_site_0",
        "site_xpos_gripper0_grip_site_1",
        "site_xpos_gripper0_grip_site_2",
        "gripper_qpos_0", "gripper_qpos_1",
        "gripper_qpos_mean", "gripper_qpos_abs_mean", "grip",
    }
    assert meta["gripper_qpos_0"].group == "gripper"
    assert meta["gripper_qpos_0"].note == "m"
    assert meta["gripper_qpos_mean"].group == "gripper"
    assert meta["grip"].group == "gripper"


def test_libero_gripper_meta_falls_back_without_a_robot():
    from specola.adapters.libero import LIBEROAdapter

    adapter = _bare_adapter(LIBEROAdapter)
    adapter._gripper = _SiteRecord(
        entity_id=9, site_name="gripper0_grip_site", site_id=0,
        entity_class="gripper",
    )
    adapter.env = _FakeEnv(robots=[])

    meta = adapter._gripper_property_meta()

    assert set(meta) == {
        "site_xpos_gripper0_grip_site_0",
        "site_xpos_gripper0_grip_site_1",
        "site_xpos_gripper0_grip_site_2",
    }


# ---------------------------------------------------------------------------
# RoboCasaAdapter: finger-joint + end-effector force/torque extras
# ---------------------------------------------------------------------------

def test_robocasa_gripper_meta_matches_property_keys():
    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = _bare_adapter(RoboCasaAdapter)
    adapter._gripper = _SiteRecord(
        entity_id=9, site_name="gripper0_grip_site", site_id=0,
        entity_class="gripper",
    )
    adapter.env = _FakeEnv(
        robots=[
            _FakeRobot(
                gripper=_FakeGripper(qpos=[0.03, -0.03]),
                recent_ee_force=[1.0, 0.0, 2.0],
                recent_ee_torque=[0.1, 0.0, 0.2],
            )
        ]
    )

    meta = adapter._gripper_property_meta()

    assert set(meta) == {
        "site_xpos_gripper0_grip_site_0",
        "site_xpos_gripper0_grip_site_1",
        "site_xpos_gripper0_grip_site_2",
        "gripper_qpos_0", "gripper_qpos_1",
        "gripper_qpos_mean", "gripper_qpos_abs_mean", "grip",
        "recent_ee_force_x", "recent_ee_force_y",
        "recent_ee_force_z", "recent_ee_force_norm",
        "recent_ee_torque_x", "recent_ee_torque_y",
        "recent_ee_torque_z", "recent_ee_torque_norm",
    }
    assert meta["recent_ee_force_x"].group == "gripper"
    assert meta["recent_ee_force_x"].note == "N"
    assert meta["recent_ee_torque_x"].group == "gripper"
    assert meta["recent_ee_torque_x"].note == "N·m"


def test_robocasa_gripper_meta_falls_back_without_a_robot():
    from specola.adapters.robocasa import RoboCasaAdapter

    adapter = _bare_adapter(RoboCasaAdapter)
    adapter._gripper = _SiteRecord(
        entity_id=9, site_name="gripper0_grip_site", site_id=0,
        entity_class="gripper",
    )
    adapter.env = _FakeEnv(robots=[])

    meta = adapter._gripper_property_meta()

    assert set(meta) == {
        "site_xpos_gripper0_grip_site_0",
        "site_xpos_gripper0_grip_site_1",
        "site_xpos_gripper0_grip_site_2",
    }
