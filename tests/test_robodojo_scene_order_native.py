"""Native USD ordering acceptance, separate from the frozen Task 9B census."""
import hashlib
from typing import cast

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import UsdGeom
from test_robodojo_scene import _stage, _env, _session, _function_inputs, _CUBE_POINTS
from specola.dashboard.recorder import WebRecorderSink
from specola.dashboard.recording import RecordingReader, read_geometry
from specola.run_state import RunStateTracker
from specola.adapters.robodojo import RoboDojoAdapter


def test_authored_mesh_before_first_recording_export_preserves_inputs(tmp_path):
    stage = _stage()
    env, _bowl = _env(stage)
    session = _session(tmp_path, env)
    adapter = cast(RoboDojoAdapter, session.adapter)
    tracker = RunStateTracker(session.engine.manager_specs())
    sink = WebRecorderSink(tmp_path / "recorded", run_id="native-order", simulator="robodojo",
                           task="synthetic USD ordering fixture", mode="hold", cameras=(), video="jpeg")
    try:
        first_step = session.step()
        first = _function_inputs(session)
        assert first is not None
        old = first.values["robodojo.scene"]["geometry"]
        old_vertices = old["collected"][0][3].copy()
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/envs/env_0/robot0/base_link/collision"))
        mesh.GetPointsAttr().Set([(2 * x, y, z) for x, y, z in _CUBE_POINTS])
        assert adapter._geometry_dirty
        sink.on_step(adapter, tracker.update(first_step.raw))
        assert sink.active
        exported = adapter.scene_geometry()
        second_step = session.step()
        second = _function_inputs(session)
        assert second is not None
        assert second.revision != first.revision
        assert old is not second.values["robodojo.scene"]["geometry"]
        assert np.array_equal(old["collected"][0][3], old_vertices)
        assert not second.values["robodojo.scene"]["geometry"]["collected"][0][3].flags.writeable
        sink.on_step(adapter, tracker.update(second_step.raw))
        session.end()
        sink.finalize_episode("truncated")
        sink.close()
        reader = RecordingReader(sink.recording_dir)
        assert reader.num_steps() == 2
        saved = read_geometry(sink.recording_dir)
        assert saved is not None and exported is not None
        assert saved["json"] == exported["json"]
        assert hashlib.sha256(saved["meshes"]).hexdigest() == hashlib.sha256(exported["meshes"]).hexdigest()
    finally:
        sink.close()
        adapter.close()
