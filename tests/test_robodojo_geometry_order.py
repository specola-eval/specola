"""A first mesh export must preserve operands across plan invalidation."""
from types import SimpleNamespace

import numpy as np
import pytest

import specola.adapters.robodojo as module
from specola.adapters.robodojo import RoboDojoAdapter


@pytest.mark.parametrize("dirty", [False, True])
def test_first_export_preserves_immutable_function_operands(monkeypatch, dirty):
    adapter = object.__new__(RoboDojoAdapter)
    adapter._geometry_listener = None
    adapter._scene_plan = None
    adapter._scene_plan_built = False
    adapter._native_geometry_snapshot = None
    adapter._geometry_revision = 0
    adapter._geometry_dirty = False
    adapter._function_input_requests = {"robodojo.scene": 1}
    monkeypatch.setattr(adapter, "_labels", lambda: {})
    monkeypatch.setattr(adapter, "_body_poses", lambda plan: (
        np.zeros((1, 3)), np.array([[1., 0., 0., 0.]])))
    vertices = []

    def build():
        points = np.full((3, 3), len(vertices), dtype=float)
        vertices.append(points)
        return {"bodies": [{}], "robots": (), "objects": (), "geometry": None,
                "collected": [(0, 0, (1., 1., 1., 1.), points, np.array([[0, 1, 2]]))],
                "skipped": {}}

    monkeypatch.setattr(adapter, "_build_scene_plan", build)
    monkeypatch.setattr(module, "pack_scene", lambda *args: {"json": {"bodies": []}})
    first = adapter._capture_function_inputs()
    assert first is not None
    old = first.values["robodojo.scene"]["geometry"]
    if dirty:
        path = SimpleNamespace(name="points", GetPrimPath=lambda: "/World/mesh")
        notice = SimpleNamespace(GetResyncedPaths=lambda: (), GetChangedInfoOnlyPaths=lambda: (path,))
        adapter._geometry_changed(notice, None)
    adapter.scene_geometry()
    second = adapter._capture_function_inputs()
    assert second is not None
    new = second.values["robodojo.scene"]["geometry"]
    assert (old is new) is (not dirty)
    assert second.revision == int(dirty)
    assert adapter._scene_plan is not None
    assert "collected" not in adapter._scene_plan
    assert new is adapter._native_geometry_snapshot
    vertices[-1][:] = 99
    assert np.all(old["collected"][0][3] == 0)
    assert np.all(new["collected"][0][3] == int(dirty))
    with pytest.raises(ValueError):
        new["collected"][0][3][0, 0] = 12


def test_export_without_functions_releases_collected_rows(monkeypatch):
    adapter = object.__new__(RoboDojoAdapter)
    adapter._native_geometry_snapshot = None
    plan = {"geometry": None, "bodies": [], "collected": [], "skipped": {}}
    monkeypatch.setattr(adapter, "_ensure_scene_plan", lambda: plan)
    monkeypatch.setattr(module, "pack_scene", lambda *args: {})
    assert adapter.scene_geometry() == {}
    assert "collected" not in plan
    assert adapter._native_geometry_snapshot is None
