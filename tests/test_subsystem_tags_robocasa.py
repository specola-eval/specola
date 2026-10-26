"""RoboCasaAdapter subsystem classification from robosuite part lists.

Pure dict code -- no live simulator required (same convention as
tests/test_gripper_property_meta.py). The fake robot mirrors the shape of
robosuite's ``_ref_joints_indexes_dict`` after ``setup_references()``.
"""

from __future__ import annotations

from specola.adapters.robocasa import RoboCasaAdapter


def _bare_adapter():
    return object.__new__(RoboCasaAdapter)


class _FakeRobot:
    def __init__(self, arms=("right",)):
        self.arms = list(arms)
        self._ref_joints_indexes_dict = {
            "base": [0, 1],
            "torso": [2],
            "head": [9],
        }
        for offset, arm in enumerate(self.arms):
            self._ref_joints_indexes_dict[arm] = [3 + offset]
            self._ref_joints_indexes_dict[f"{arm}_gripper"] = [7 + offset]

    def get_gripper_name(self, arm):
        return f"{arm}_gripper"


class _FakeEnv:
    def __init__(self, robots):
        self.robots = robots


def _adapter(robots):
    adapter = _bare_adapter()
    adapter.env = _FakeEnv(robots)
    return adapter


def test_single_arm_classification():
    adapter = _adapter([_FakeRobot()])
    assert adapter._joint_subsystem_tags(3) == ("arm", "right")
    assert adapter._joint_subsystem_tags(7) == ("gripper", "right")
    assert adapter._joint_subsystem_tags(0) == ("base",)
    assert adapter._joint_subsystem_tags(2) == ("torso",)


def test_head_and_unknown_joints_get_no_tags():
    adapter = _adapter([_FakeRobot()])
    assert adapter._joint_subsystem_tags(9) == ()      # head: unmapped
    assert adapter._joint_subsystem_tags(42) == ()     # not a robot joint


def test_bimanual_sides():
    adapter = _adapter([_FakeRobot(arms=("right", "left"))])
    assert adapter._joint_subsystem_tags(3) == ("arm", "right")
    assert adapter._joint_subsystem_tags(4) == ("arm", "left")
    assert adapter._joint_subsystem_tags(7) == ("gripper", "right")
    assert adapter._joint_subsystem_tags(8) == ("gripper", "left")


def test_no_robots_degrades_to_no_tags():
    adapter = _adapter([])
    assert adapter._joint_subsystem_tags(3) == ()


def test_robot_without_part_dict_degrades_to_no_tags():
    class _BareRobot:
        pass

    adapter = _adapter([_BareRobot()])
    assert adapter._joint_subsystem_tags(3) == ()
