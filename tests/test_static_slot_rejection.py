"""Fixture reproduction of the X15 never-bound gripper slot.

Two co-located sites, one of them the adapter's gripper entity and the
other a raw ``site`` entity. A specification that names the raw site's
key on the ``gripper`` class has no candidate entity at reset and none
can appear later, because ``gripper`` is a static class of a closed
vocabulary. Compilation rejects it instead of waiting forever
(docs/reviews/2026-09-02-x15-never-bound-slot-postmortem.md, Link 3).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import (  # noqa: E402
    _CURATED_FIELD_OWNER_ENTITY_CLASSES,
    MujocoAdapter,
)
from specola.entities import UnboundStaticSlotError  # noqa: E402
from specola.session import Monitor  # noqa: E402

pytestmark = pytest.mark.mujoco


_MODEL = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="hand" pos="0 0 0.5">
      <joint name="hand_slide" type="slide" axis="0 0 1"/>
      <geom name="hand_geom" type="box" size="0.02 0.02 0.02"/>
      <site name="gripper0_right_grip_site" pos="0 0 0.01" size="0.005"/>
      <site name="gripper0_right_grip_site_cylinder" pos="0 0 0.01"
            size="0.004"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Env:
    """Minimal MuJoCo env surface: the adapter only needs model/data."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_MODEL)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


class _ClosedVocabularyAdapter(MujocoAdapter):
    """Declares a class vocabulary, as every bundled sim adapter does."""

    @property
    def supported_classes(self):
        return frozenset({"gripper"}) | _CURATED_FIELD_OWNER_ENTITY_CLASSES


def _adapter():
    return _ClosedVocabularyAdapter(
        _Env(), gripper_site="gripper0_right_grip_site_cylinder"
    )


def _spec_file(tmp_path, component):
    path = tmp_path / "gripper.yaml"
    path.write_text(
        "specs:\n"
        "  - name: gripper_height\n"
        f'    formula: "gripper_*[{component}] >= 0"\n'
    )
    return str(path)


def test_watch_rejects_key_that_belongs_to_another_entity(tmp_path):
    path = _spec_file(tmp_path, "site_xpos_gripper0_right_grip_site_0")
    monitor = Monitor(path, results_dir=None)

    with pytest.raises(UnboundStaticSlotError) as excinfo:
        monitor.watch(_adapter())

    assert "gripper" in str(excinfo.value)


def test_watch_binds_the_gripper_entity_own_key(tmp_path):
    path = _spec_file(
        tmp_path, "site_xpos_gripper0_right_grip_site_cylinder_0"
    )
    monitor = Monitor(path, results_dir=None)

    session = monitor.watch(_adapter())
    result = session.step()
    session.end()

    assert result.rho("gripper_height") is not None
