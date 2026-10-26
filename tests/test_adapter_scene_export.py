"""scene_geometry()/scene_poses(): base default and MuJoCo export shapes."""

import numpy as np
import pytest

from specola.adapter import SimulatorAdapter


def test_base_adapter_has_no_scene_capability():
    assert SimulatorAdapter.scene_geometry(object()) is None      # type: ignore[arg-type]
    assert SimulatorAdapter.scene_poses(object()) is None         # type: ignore[arg-type]


MJCF = """
<mujoco>
  <asset><mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.2 0.3" rgba="1 0 0 1" group="1"/>
      <geom type="mesh" mesh="tet"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.mujoco
def test_mujoco_scene_export_shapes():
    mujoco = pytest.importorskip("mujoco")
    from specola.adapters.mujoco import MujocoAdapter

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter.__new__(MujocoAdapter)
    adapter._model, adapter._data = model, data

    geometry = adapter.scene_geometry()
    assert geometry is not None
    meta = geometry["json"]
    assert meta["nbody"] == model.nbody and meta["ngeom"] == 3
    # No env on this adapter: the plain-MuJoCo viewer convention.
    assert meta["visible_groups"] == [0, 1, 2]
    box = next(g for g in meta["geoms"] if g["type"] == "box")
    assert box["size"] == pytest.approx([0.1, 0.2, 0.3])
    assert box["rgba"] == pytest.approx([1, 0, 0, 1]) and box["group"] == 1
    mesh_geom = next(g for g in meta["geoms"] if g["type"] == "mesh")
    entry = meta["meshes"][mesh_geom["mesh"]]
    verts = np.frombuffer(
        geometry["meshes"], dtype="<f4",
        count=entry["vert_count"] * 3, offset=entry["vert_offset"] * 12)
    assert verts.reshape(-1, 3).shape == (4, 3)
    faces = np.frombuffer(
        geometry["meshes"], dtype="<u4", offset=meta["faces_start"]
        + entry["face_offset"] * 12, count=entry["face_count"] * 3)
    assert faces.max() < entry["vert_count"]

    poses = adapter.scene_poses()
    assert poses is not None
    xpos, xquat = poses
    assert xpos.shape == (model.nbody, 3) and xpos.dtype == np.float32
    assert xquat.shape == (model.nbody, 4) and xquat.dtype == np.float32
    assert xpos.base is None  # fresh copy, not a view into mjData


MATERIAL_MJCF = """
<mujoco>
  <asset>
    <material name="m" rgba="0 1 0 0.5"/>
    <hfield name="h" nrow="2" ncol="2" size="1 1 1 0.1"/>
  </asset>
  <worldbody>
    <geom type="sphere" size="0.1" material="m"/>
    <geom type="sphere" size="0.1" material="m" rgba="1 0 0 1"/>
    <geom type="sphere" size="0.1"/>
    <geom type="hfield" hfield="h"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.mujoco
def test_mujoco_scene_export_colors_and_skipped_types():
    mujoco = pytest.importorskip("mujoco")
    from specola.adapters.mujoco import MujocoAdapter

    model = mujoco.MjModel.from_xml_string(MATERIAL_MJCF)
    adapter = MujocoAdapter.__new__(MujocoAdapter)
    adapter._model, adapter._data = model, mujoco.MjData(model)

    meta = adapter.scene_geometry()["json"]
    assert meta["ngeom"] == 4                    # model total, skips included
    assert meta["skipped"] == {"hfield": 1}
    # Exported geoms keep model order with the skipped ones removed.
    material_only, material_and_rgba, plain = meta["geoms"]
    assert material_only["rgba"] == pytest.approx([0, 1, 0, 0.5])
    assert material_and_rgba["rgba"] == pytest.approx([1, 0, 0, 1])
    assert plain["rgba"] == pytest.approx([0.5, 0.5, 0.5, 1])
    # Resolving colors must not write back into the model.
    assert model.geom_rgba[0].tolist() == pytest.approx([0.5, 0.5, 0.5, 1])


@pytest.mark.mujoco
def test_visible_groups_follow_the_simulators_render_convention():
    mujoco = pytest.importorskip("mujoco")
    from specola.adapters.mujoco import MujocoAdapter

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter.__new__(MujocoAdapter)
    adapter._model, adapter._data = model, data

    class _RobosuiteLike:
        # robosuite defaults: collision hulls hidden, visual meshes shown
        render_collision_mesh = False
        render_visual_mesh = True

    adapter.env = _RobosuiteLike()
    geometry = adapter.scene_geometry()
    assert geometry is not None
    assert geometry["json"]["visible_groups"] == [1, 2]

    adapter.env.render_collision_mesh = True
    assert adapter._visible_geom_groups() == [0, 1, 2]

    class _Wrapper:
        # LIBERO-style wrapper: takes the flags as kwargs, stores
        # neither, holds the real env at .env.
        def __init__(self, inner):
            self.env = inner

    adapter.env = _Wrapper(_RobosuiteLike())
    adapter.env.env.render_collision_mesh = False
    assert adapter._visible_geom_groups() == [1, 2]
