"""Live-view render hook: SimulatorAdapter.render() and --render.

The base hook is a no-op so every adapter accepts render() calls; the
MuJoCo adapter opens one passive viewer lazily and syncs it per call,
unwrapping robosuite's MjModel/MjData wrappers to the raw MuJoCo
objects the viewer requires. The CLI flag threads the per-step call
and refuses adapters that keep the base no-op (issue #131).
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from specola.adapters.fake import _make_mock

_MOCK_SPECS = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"


def test_base_adapter_render_is_noop():
    adapter = _make_mock(None, None, False)
    assert adapter.render() is None
    assert adapter.render() is None


def test_dmcontrol_inherits_working_render_hooks():
    # dmcontrol uses the generic passive-viewer hooks (an earlier refusal
    # was removed after end-to-end verification: full 999-step episodes
    # run clean; the one observed mid-episode reset never reproduced and
    # observation validation guards it loudly if it recurs).
    from specola.adapters.dmcontrol import DMControlAdapter
    from specola.adapters.mujoco import MujocoAdapter

    assert DMControlAdapter.render is MujocoAdapter.render
    assert DMControlAdapter.close_render is MujocoAdapter.close_render


def test_run_monitor_render_flag_refused_on_mock(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(_MOCK_SPECS),
         "--output-dir", str(tmp_path / "results"),
         "--steps", "5", "--render"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1, proc.stderr
    assert (
        "--render is not supported by the mock adapter: it has no live viewer"
        in proc.stderr
    )


# ---------------------------------------------------------------------------
# MujocoAdapter: lazy passive viewer (requires mujoco; viewer monkeypatched)
# ---------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import BodyRule, MujocoAdapter  # noqa: E402

_XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _TinyEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


class _ViewerSpy:
    def __init__(self):
        self.sync_calls = 0
        self.close_calls = 0
        self.closed = False

    def is_running(self):
        return not self.closed

    def sync(self):
        self.sync_calls += 1

    def close(self):
        self.close_calls += 1
        self.closed = True


def _adapter_and_spy(monkeypatch):
    import mujoco.viewer

    adapter = MujocoAdapter(
        _TinyEnv(),
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    adapter.reset()

    spy = _ViewerSpy()
    launches = []

    def fake_launch_passive(model, data):
        launches.append((model, data))
        return spy

    monkeypatch.setattr(
        mujoco.viewer, "launch_passive", fake_launch_passive
    )
    return adapter, spy, launches


def test_mujoco_render_launches_viewer_once_and_syncs(monkeypatch):
    adapter, spy, launches = _adapter_and_spy(monkeypatch)

    adapter.render()
    adapter.render()
    adapter.render()

    assert len(launches) == 1
    assert spy.sync_calls == 3
    # The viewer received the raw MuJoCo objects the adapter monitors.
    assert launches[0][0] is adapter._model
    assert launches[0][1] is adapter._data


def test_mujoco_render_stops_syncing_after_viewer_closed(monkeypatch):
    adapter, spy, launches = _adapter_and_spy(monkeypatch)

    adapter.render()
    spy.closed = True
    adapter.render()
    adapter.render()

    assert len(launches) == 1          # never relaunched
    assert spy.sync_calls == 1         # no syncs after close


def test_base_adapter_close_render_is_noop():
    adapter = _make_mock(None, None, False)
    assert adapter.close_render() is None


def test_mujoco_close_render_closes_viewer_once_and_settles(monkeypatch):
    # The viewer must be closed deterministically before interpreter
    # teardown, and close must be followed by a settle pause: process
    # exit racing the viewer thread's shutdown dumps core on MuJoCo
    # 3.10 even after close() (verified by minimal repro; a 2s pause
    # eliminates it).
    from specola.adapters import mujoco as mujoco_adapter

    sleeps = []
    monkeypatch.setattr(
        mujoco_adapter.time, "sleep", lambda s: sleeps.append(s)
    )

    adapter, spy, launches = _adapter_and_spy(monkeypatch)

    adapter.render()
    adapter.close_render()
    adapter.close_render()          # idempotent

    assert spy.close_calls == 1
    assert sleeps == [mujoco_adapter._VIEWER_TEARDOWN_SETTLE]
    # A render after close relaunches cleanly rather than syncing a
    # dead viewer.
    adapter.render()
    assert len(launches) == 2


def test_mujoco_close_render_without_render_is_noop(monkeypatch):
    adapter, spy, launches = _adapter_and_spy(monkeypatch)
    assert adapter.close_render() is None
    assert launches == []


def test_base_adapter_capture_frame_returns_none():
    adapter = _make_mock(None, None, False)
    assert adapter.capture_frame() is None
    assert adapter.capture_frame(camera="anything", width=64, height=48) is None


_CAMERA_XML = """
<mujoco>
  <worldbody>
    <camera name="cam0" pos="0 -1 1"/>
    <body name="obj" pos="0 0 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _RendererSpy:
    def __init__(self, model, height, width):
        self.model = model
        self.height = height
        self.width = width
        self.scenes = []
        self.close_calls = 0
        self.scene = types.SimpleNamespace(flags=[0] * 32)

    def update_scene(self, data, camera):
        self.scenes.append((data, camera))

    def render(self):
        import numpy as np

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        self.close_calls += 1


def _capture_adapter(monkeypatch, xml=_XML):
    class _Env:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _Env(),
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    adapter.reset()

    spies = []

    def fake_renderer(model, height, width):
        spies.append(_RendererSpy(model, height, width))
        return spies[-1]

    # capture_frame imports mujoco lazily; patching the module attribute
    # reaches that import.
    monkeypatch.setattr(mujoco, "Renderer", fake_renderer)
    return adapter, spies


def test_mujoco_capture_frame_defaults_to_first_named_camera(monkeypatch):
    adapter, spies = _capture_adapter(monkeypatch, xml=_CAMERA_XML)

    frame = adapter.capture_frame()

    assert frame.shape == (240, 320, 3)
    assert frame.dtype.name == "uint8"
    assert len(spies) == 1
    assert spies[0].scenes[0][1] == "cam0"
    # The renderer received the raw model the adapter monitors.
    assert spies[0].model is adapter._model


def test_mujoco_capture_frame_enables_shadows(monkeypatch):
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame()

    flags = spies[0].scene.flags
    assert flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] == 1
    assert flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] == 1


def test_mujoco_capture_frame_raises_offscreen_buffer_cap(monkeypatch):
    # Models author small offwidth/offheight; the cap is visual-only
    # and must not limit capture resolution.
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame(width=1280, height=720)

    assert int(adapter._model.vis.global_.offwidth) >= 1280
    assert int(adapter._model.vis.global_.offheight) >= 720


def test_mujoco_capture_frame_free_camera_when_model_has_none(monkeypatch):
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame()

    assert spies[0].scenes[0][1] == -1


def test_mujoco_capture_frame_unknown_camera_raises(monkeypatch):
    adapter, spies = _capture_adapter(monkeypatch, xml=_CAMERA_XML)

    with pytest.raises(ValueError, match="cam0"):
        adapter.capture_frame(camera="nope")
    assert spies == []


def test_mujoco_capture_frame_caches_renderer_and_rebuilds_on_resize(
    monkeypatch,
):
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame()
    adapter.capture_frame()
    assert len(spies) == 1

    adapter.capture_frame(width=64, height=48)
    assert len(spies) == 2
    assert spies[0].close_calls == 1
    assert (spies[1].height, spies[1].width) == (48, 64)


def test_mujoco_close_render_closes_offscreen_renderer(monkeypatch):
    from specola.adapters import mujoco as mujoco_adapter

    monkeypatch.setattr(
        mujoco_adapter.time, "sleep", lambda s: None
    )
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame()
    adapter.close_render()

    assert spies[0].close_calls == 1
    # A capture after close rebuilds cleanly.
    adapter.capture_frame()
    assert len(spies) == 2


def test_mujoco_capture_frame_real_offscreen_render():
    # End-to-end against the real mujoco.Renderer. Runs in a subprocess:
    # creating a fresh offscreen GL context late in the suite, after many
    # other EGL consumers, can kill the interpreter outright (observed
    # 2026-08-20, reproducible in-suite, absent in isolation), and an
    # uncatchable crash must surface as a test outcome, not take the
    # whole run down. Skipped where no offscreen GL context exists.
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys; sys.path.insert(0, {root!r});"
        "sys.path.insert(0, {tests!r});"
        "from test_render_hook import _TinyEnv;"
        "from specola.adapters.mujoco import MujocoAdapter, BodyRule;"
        "env = _TinyEnv();"
        "adapter = MujocoAdapter(env, body_rules=[BodyRule(match='obj', cls='object')]);"
        "adapter.reset();"
        "frame = adapter.capture_frame(width=64, height=48);"
        "adapter.close_render();"
        "assert frame.shape == (48, 64, 3), frame.shape;"
        "assert frame.dtype.name == 'uint8', frame.dtype.name;"
        "print('RENDER-OK')"
    ).format(
        root=str(Path(__file__).resolve().parents[1] / "src"),
        tests=str(Path(__file__).resolve().parent),
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0 and "RENDER-OK" in completed.stdout:
        return
    # Skip ONLY on recognized GL-context failure signatures; any other
    # nonzero exit is a real regression (import failure, reset failure,
    # wrong frame) and must fail with the captured output.
    gl_signatures = (
        "gladLoadGL",
        "OpenGL context",
        "GL context",
        "EGL",
        "GLFW",
        "osmesa",
        "MUJOCO_GL",
        "Could not initialize",
    )
    crashed_on_gl = completed.returncode < 0 or any(
        signature in completed.stderr for signature in gl_signatures
    )
    if crashed_on_gl and "AssertionError" not in completed.stderr:
        pytest.skip(
            "no working offscreen GL context (rc={}): {}".format(
                completed.returncode, completed.stderr.strip()[-300:]
            )
        )
    raise AssertionError(
        "offscreen render subprocess failed (rc={}):\nstdout:\n{}\n"
        "stderr:\n{}".format(
            completed.returncode, completed.stdout, completed.stderr
        )
    )


_UNNAMED_FIRST_XML = """
<mujoco>
  <worldbody>
    <camera pos="0 -1 1"/>
    <camera name="named_cam" pos="0 1 1"/>
    <body name="obj" pos="0 0 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

_ONLY_UNNAMED_XML = """
<mujoco>
  <worldbody>
    <camera pos="0 -1 1"/>
    <body name="obj" pos="0 0 0.5">
      <freejoint name="obj_joint"/>
      <geom name="obj_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_mujoco_capture_frame_skips_unnamed_default_camera(monkeypatch):
    # Monitoring identities for unnamed cameras are not renderer names;
    # the default must be the first GENUINE name (review finding).
    adapter, spies = _capture_adapter(monkeypatch, xml=_UNNAMED_FIRST_XML)

    adapter.capture_frame()

    assert spies[0].scenes[0][1] == "named_cam"


def test_mujoco_capture_frame_only_unnamed_cameras_uses_free_camera(
    monkeypatch,
):
    adapter, spies = _capture_adapter(monkeypatch, xml=_ONLY_UNNAMED_XML)

    adapter.capture_frame()

    assert spies[0].scenes[0][1] == -1


def test_mujoco_capture_frame_rebuilds_renderer_on_model_change(
    monkeypatch,
):
    # mujoco.Renderer is model-bound; reset() may replace the backend
    # model (DMControl/MA-MuJoCo) — a stale cache renders old imagery.
    adapter, spies = _capture_adapter(monkeypatch)

    adapter.capture_frame()
    assert len(spies) == 1

    replacement = _TinyEnv()
    adapter._model = replacement.model
    adapter._data = replacement.data
    adapter.capture_frame()

    assert len(spies) == 2
    assert spies[0].close_calls == 1
    assert spies[1].model is replacement.model


def test_pybullet_capture_frame_returns_rgb_uint8():
    pybullet = pytest.importorskip("pybullet")
    from specola.adapters.pybullet import PyBulletAdapter

    client = pybullet.connect(pybullet.DIRECT)
    try:
        adapter = PyBulletAdapter(client)
        adapter.reset()
        frame = adapter.capture_frame(width=64, height=48)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype.name == "uint8"
        with pytest.raises(ValueError, match="named cameras"):
            adapter.capture_frame(camera="front")
    finally:
        pybullet.disconnect(client)


def test_mujoco_render_unwraps_robosuite_style_wrappers(monkeypatch):
    import mujoco.viewer

    class _WrappedModel:
        def __init__(self, raw):
            self._model = raw

    class _WrappedData:
        def __init__(self, raw):
            self._data = raw

    env = _TinyEnv()
    adapter = MujocoAdapter(
        env,
        body_rules=[BodyRule(match="obj", cls="object")],
    )
    adapter.reset()
    adapter._model = _WrappedModel(env.model)
    adapter._data = _WrappedData(env.data)

    launches = []
    monkeypatch.setattr(
        mujoco.viewer, "launch_passive",
        lambda model, data: (launches.append((model, data)), _ViewerSpy())[1],
    )

    adapter.render()

    assert launches == [(env.model, env.data)]
