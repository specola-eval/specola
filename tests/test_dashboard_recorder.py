"""WebRecorderSink: threading, backpressure, degradation — sim-free."""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from _dashboard_fixtures import make_snapshots

from specola.dashboard import recorder as recorder_module
from specola.dashboard.recorder import WebRecorderSink
from specola.dashboard.recording import (
    PosesReader, RecordingReader, read_geometry)
from specola.dashboard.video import VideoPipelineError


@pytest.fixture(autouse=True)
def no_encoder_probe(monkeypatch):
    """No recorder test may spawn ffmpeg.

    The default store is "auto", which probes the hardware encoder on
    the first frame — on a box that has one, every test here would flip
    to the video store and start real subprocesses. Tests that want the
    video store stub the probe and the writer themselves.
    """
    monkeypatch.setattr(recorder_module, "probe_encoder",
                        lambda encoder_args: False)


class _FrameAdapter:
    def __init__(self, frame=None, names=("cam0", "cam1")):
        self.frame = frame
        self.names = tuple(names)
        self.calls = []

    def camera_names(self):
        return self.names

    def capture_frame(self, camera=None, *, width=320, height=240):
        self.calls.append((camera, width, height))
        return self.frame


_GEOMETRY = {"json": {"format_version": 1, "nbody": 1, "ngeom": 0,
                      "bodies": [], "geoms": [], "meshes": [],
                      "faces_start": 0, "skipped": {}},
             "meshes": b""}


class _SceneAdapter(_FrameAdapter):
    """Adapter with the scene hooks; ``geometry=None`` means no 3D scene."""

    def __init__(self, geometry=_GEOMETRY, **kwargs):
        super().__init__(**kwargs)
        self.geometry = geometry
        self.geometry_calls = 0
        self.pose_calls = 0

    def scene_geometry(self):
        self.geometry_calls += 1
        return self.geometry

    def scene_poses(self):
        self.pose_calls += 1
        return (np.ones((1, 3), np.float32), np.full((1, 4), 2, np.float32))


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _sink(tmp_path, **kwargs):
    defaults = dict(run_id="r", simulator="mock", task="t", mode="live",
                    cameras=None)
    defaults.update(kwargs)
    return WebRecorderSink(tmp_path, **defaults)


def _drive(sink, adapter, steps):
    for snapshot in make_snapshots(steps=steps):
        sink.on_step(adapter, snapshot)


def test_records_data_every_step_and_default_camera(tmp_path):
    adapter = _FrameAdapter(frame=_frame())
    sink = _sink(tmp_path)
    _drive(sink, adapter, 3)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert reader.num_steps() == 3
    assert reader.manifest["cameras"] == ["cam0"]
    assert reader.manifest["status"] == "completed"
    assert adapter.calls[0][0] is None            # default = camera=None
    assert reader.frame("cam0", 2) is not None
    assert reader.read_step(1)["wall_time"] is not None


def test_stride_thins_frames_not_data(tmp_path):
    adapter = _FrameAdapter(frame=_frame())
    sink = _sink(tmp_path, stride=2)
    _drive(sink, adapter, 4)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert reader.num_steps() == 4                # data every step
    assert len(adapter.calls) == 2                # frames at steps 0, 2
    assert reader.frame("cam0", 3) == (reader.frame("cam0", 2)[0], 2)


def test_all_cameras(tmp_path):
    adapter = _FrameAdapter(frame=_frame())
    sink = _sink(tmp_path, cameras=("all",))
    _drive(sink, adapter, 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert reader.manifest["cameras"] == ["cam0", "cam1"]
    assert reader.cameras() == ["cam0", "cam1"]
    assert {call[0] for call in adapter.calls} == {"cam0", "cam1"}


def test_explicit_cameras_by_name(tmp_path):
    adapter = _FrameAdapter(frame=_frame())
    sink = _sink(tmp_path, cameras=("cam1",))
    _drive(sink, adapter, 1)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert reader.manifest["cameras"] == ["cam1"]
    assert adapter.calls == [("cam1", 640, 480)]


def test_never_raises_and_disables_on_error(tmp_path, capsys):
    class _Boom:
        def camera_names(self):
            return ()

        def capture_frame(self, camera=None, *, width=320, height=240):
            raise RuntimeError("boom")

    sink = _sink(tmp_path)
    _drive(sink, _Boom(), 2)                      # must not raise
    assert sink.active is False
    assert "Web warning" in capsys.readouterr().err
    sink.close()


def test_frames_dropped_before_data(tmp_path):
    adapter = _FrameAdapter(frame=_frame())
    sink = _sink(tmp_path, queue_size=1)          # force backpressure
    release = threading.Event()
    real_encode = sink._encode

    def slow_encode(array):
        release.wait(timeout=10.0)
        return real_encode(array)

    sink._encode = slow_encode
    # Transient stall: released while the data put is still waiting, so
    # the sink must drop frames but keep every data line and stay active.
    threading.Timer(0.5, release.set).start()
    for snapshot in make_snapshots(steps=5):
        sink.on_step(adapter, snapshot)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert sink.active is True
    assert reader.num_steps() == 5                # data never dropped
    assert sink.frames_dropped > 0


def test_close_is_failed_path_backstop(tmp_path):
    sink = _sink(tmp_path)
    _drive(sink, _FrameAdapter(), 1)
    sink.close()
    reader = RecordingReader(sink.recording_dir)
    assert reader.manifest["status"] == "failed"

    sink2 = WebRecorderSink(tmp_path / "second", run_id="r", simulator="mock",
                            task="t", mode="live", cameras=None)
    sink2.on_step(_FrameAdapter(), make_snapshots(steps=1)[0])
    sink2.finalize_episode("truncated")
    sink2.close()                                 # no-op after finalize
    reader2 = RecordingReader(sink2.recording_dir)
    assert reader2.manifest["status"] == "truncated"


def test_none_frame_writes_no_jpeg(tmp_path):
    adapter = _FrameAdapter(frame=None)           # mock adapter behavior
    sink = _sink(tmp_path)
    _drive(sink, adapter, 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert reader.num_steps() == 2
    assert reader.cameras() == []


def test_on_line_callback_from_writer_thread(tmp_path):
    seen = []

    def on_line(line, frames, dropped):
        seen.append((line["step"], sorted(frames), dropped,
                     threading.current_thread().name))

    sink = _sink(tmp_path)
    sink.set_callbacks(on_line, None)
    _drive(sink, _FrameAdapter(frame=_frame()), 2)
    sink.finalize_episode("completed")
    assert [entry[0] for entry in seen] == [0, 1]
    assert seen[0][1] == ["cam0"]
    assert all(entry[3] == "specola-web-writer" for entry in seen)


def test_on_episode_end_callback(tmp_path):
    statuses = []
    sink = _sink(tmp_path)
    sink.set_callbacks(None, statuses.append)
    _drive(sink, _FrameAdapter(), 1)
    sink.finalize_episode("completed")
    assert statuses == ["completed"]


def test_records_geometry_once_and_a_pose_record_per_line(tmp_path):
    pushed = []
    geometry_seen = []
    adapter = _SceneAdapter(frame=_frame())

    def on_line(line, frames, dropped, poses=None):
        pushed.append(poses)
        # The server announces geometry from disk on the first line, so
        # the files must precede every line callback.
        geometry_seen.append(read_geometry(sink.recording_dir) is not None)

    sink = _sink(tmp_path, on_line=on_line)
    _drive(sink, adapter, 3)
    sink.finalize_episode("completed")
    assert adapter.geometry_calls == 1            # probed once per episode
    assert adapter.pose_calls == 3
    assert geometry_seen and geometry_seen[0] is True
    assert read_geometry(sink.recording_dir) == _GEOMETRY
    reader = PosesReader.open(sink.recording_dir)
    assert reader is not None
    assert reader.nbody == 1 and reader.count == 3
    record = reader.record(0)
    assert np.frombuffer(record, dtype="<f4").tolist() == [1.0] * 3 + [2.0] * 4
    # The live path pushes exactly the bytes the sidecar stores.
    assert pushed == [record] * 3


def test_without_scene_capability_writes_no_sidecar(tmp_path):
    seen = []

    def on_line(line, frames, dropped):            # 3-argument callback
        seen.append(line["step"])

    sink = _sink(tmp_path, on_line=on_line)
    _drive(sink, _FrameAdapter(), 2)
    sink.finalize_episode("completed")
    assert sink.active is True
    assert seen == [0, 1]
    assert read_geometry(sink.recording_dir) is None
    assert PosesReader.open(sink.recording_dir) is None


def test_geometry_probe_returning_none_stops_pose_capture(tmp_path):
    adapter = _SceneAdapter(geometry=None)         # hooks exist, no scene
    sink = _sink(tmp_path)
    _drive(sink, adapter, 3)
    sink.finalize_episode("completed")
    assert adapter.geometry_calls == 1 and adapter.pose_calls == 0
    assert PosesReader.open(sink.recording_dir) is None
    assert RecordingReader(sink.recording_dir).num_steps() == 3


def test_poses_survive_frame_backpressure(tmp_path):
    # Frames drop under a full queue; the pose record rides with its line.
    adapter = _SceneAdapter(frame=_frame())
    sink = _sink(tmp_path, queue_size=1, on_line=None)
    release = threading.Event()
    real_encode = sink._encode

    def slow_encode(array):
        release.wait(timeout=10.0)
        return real_encode(array)

    sink._encode = slow_encode
    threading.Timer(0.5, release.set).start()
    _drive(sink, adapter, 5)
    sink.finalize_episode("completed")
    assert sink.active is True and sink.frames_dropped > 0
    reader = PosesReader.open(sink.recording_dir)
    assert reader is not None and reader.count == 5


# -- video frame store ------------------------------------------------------

# One access unit as SegmentWriter hands it over: an SPS NAL (type 7)
# whose profile/constraint/level bytes name the codec, then an IDR.
_UNIT = b"\x00\x00\x00\x01\x67\x42\xc0\x1f\x00\x00\x00\x01\x65idr"
_CODEC = "avc1.42C01F"


def _stub_writers(monkeypatch, *, lag=2, fail_after=None,
                  fail_construct=False, fail_construct_after=None):
    """Install a SegmentWriter stand-in; return the list of instances.

    The stand-in pairs units with fed steps first-in-first-out and lags
    them the way a real encoder's pipeline does, so tests read the
    delivered steps off ``returned`` instead of assuming a lag.
    ``fail_construct_after=N`` builds N writers then raises — the
    mid-list case, where already-built writers must still be closed.

    Installing a writer stub implies a usable pipeline, so the probe is
    stubbed True here (the store gate probes under "nvenc" too); tests
    about the probe itself re-stub it after calling this.
    """
    monkeypatch.setattr(recorder_module, "probe_encoder",
                        lambda encoder_args: True)
    made = []

    class _StubWriter:
        def __init__(self, directory, camera, width, height, **kwargs):
            if fail_construct or (fail_construct_after is not None
                                  and len(made) >= fail_construct_after):
                raise VideoPipelineError("no encoder here")
            self.directory = directory
            self.camera = camera
            self.size = (width, height)
            self.fed = []
            self.returned = []
            self.closes = 0
            made.append(self)

        def feed(self, frame, step):
            self.fed.append(step)
            if fail_after is not None and len(self.fed) > fail_after:
                raise VideoPipelineError("ffmpeg stopped accepting frames")
            ready = self.fed[:-lag] if lag else list(self.fed)
            pairs = [(fed, _UNIT) for fed in ready[len(self.returned):]]
            self.returned.extend(fed for fed, _ in pairs)
            return pairs

        def close(self):
            self.closes += 1

    monkeypatch.setattr(recorder_module, "SegmentWriter", _StubWriter)
    return made


def test_forced_jpeg_store_ignores_the_video_pipeline(tmp_path, monkeypatch):
    made = _stub_writers(monkeypatch)
    entries = []
    sink = _sink(tmp_path, video="jpeg",
                 on_line=lambda line, frames, dropped: entries.append(frames))
    _drive(sink, _FrameAdapter(frame=_frame()), 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert made == []                             # no encoder was started
    assert reader.manifest["frame_store"] == "jpeg"
    assert reader.manifest["format_version"] == 1
    assert reader.frame("cam0", 1) is not None
    assert [entry["step"] for entry in entries[1].values()] == [1]
    assert {entry["codec"] for entry in entries[1].values()} == {"jpeg"}


def test_forced_nvenc_records_video_segments(tmp_path, monkeypatch):
    made = _stub_writers(monkeypatch, lag=1)
    stores = []
    sink = _sink(tmp_path, video="nvenc", on_line=lambda line, frames, d:
                 stores.append(RecordingReader(
                     sink.recording_dir).manifest["frame_store"]))
    _drive(sink, _FrameAdapter(frame=_frame()), 3)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert [writer.camera for writer in made] == ["cam0"]
    assert made[0].size == (64, 48)               # the frame's real size
    assert made[0].fed == [0, 1, 2]
    assert made[0].closes == 1                    # closed on the sentinel
    assert reader.manifest["frame_store"] == "video_segments"
    assert reader.manifest["format_version"] == 2
    # The manifest names the store before the first line that used it.
    assert stores[0] == "video_segments"
    assert reader.frame("cam0", 2) is None        # no JPEGs on disk


def test_nvenc_construction_failure_falls_back_to_jpeg(tmp_path, monkeypatch,
                                                       capsys):
    _stub_writers(monkeypatch, fail_construct=True)
    sink = _sink(tmp_path, video="nvenc")
    _drive(sink, _FrameAdapter(frame=_frame()), 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert sink.active is True
    assert "Web warning" in capsys.readouterr().err
    assert reader.manifest["frame_store"] == "jpeg"
    assert reader.frame("cam0", 1) is not None    # JPEGs, verbatim


def test_forced_nvenc_probe_failure_falls_back_to_jpeg(tmp_path,
                                                       monkeypatch, capsys):
    # ffmpeg starts fine on a box without the hardware encoder and only
    # fails at the first frame — so the store gate probes under "nvenc"
    # too, and a failed probe lands on JPEG instead of stopping frames.
    made = _stub_writers(monkeypatch)
    monkeypatch.setattr(recorder_module, "probe_encoder",
                        lambda encoder_args: False)
    sink = _sink(tmp_path, video="nvenc")
    _drive(sink, _FrameAdapter(frame=_frame()), 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert made == []                             # no writer constructed
    assert "Web warning" in capsys.readouterr().err
    assert reader.manifest["frame_store"] == "jpeg"
    assert reader.frame("cam0", 1) is not None


def test_mid_list_construction_failure_closes_built_writers(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    # Writer for cam0 builds, cam1's raises (a hardware session cap):
    # the built writer must be closed, not leaked, and the sink falls
    # back to JPEG for every camera.
    made = _stub_writers(monkeypatch, fail_construct_after=1)
    sink = _sink(tmp_path, video="nvenc", cameras=("all",))
    _drive(sink, _FrameAdapter(frame=_frame()), 2)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert len(made) == 1 and made[0].closes == 1
    assert "Web warning" in capsys.readouterr().err
    assert reader.manifest["frame_store"] == "jpeg"
    assert reader.frame("cam0", 1) is not None
    assert reader.frame("cam1", 1) is not None


@pytest.mark.parametrize("usable, store, announced",
                         ((True, "video_segments", "video segments"),
                          (False, "jpeg", "JPEG")))
def test_auto_probes_the_encoder_once(tmp_path, monkeypatch, capsys,
                                      usable, store, announced):
    _stub_writers(monkeypatch)
    probes = []

    def probe(encoder_args):
        probes.append(encoder_args)
        return usable

    monkeypatch.setattr(recorder_module, "probe_encoder", probe)
    sink = _sink(tmp_path, video="auto")
    _drive(sink, _FrameAdapter(frame=_frame()), 4)
    sink.finalize_episode("completed")
    assert len(probes) == 1                       # once per episode
    assert RecordingReader(sink.recording_dir).manifest["frame_store"] == store
    assert announced in capsys.readouterr().err


def test_access_units_reach_on_line_with_codec_and_fed_steps(tmp_path,
                                                             monkeypatch):
    made = _stub_writers(monkeypatch, lag=2)
    seen = []

    def on_line(line, frames, dropped):
        seen.append((line["step"], frames))

    sink = _sink(tmp_path, video="nvenc", on_line=on_line)
    _drive(sink, _FrameAdapter(frame=_frame()), 6)
    sink.finalize_episode("completed")
    delivered = [(line_step, frames["cam0"])
                 for line_step, frames in seen if frames]
    # Ground truth is what the writer returned, not an assumed lag.
    assert [entry["step"] for _, entry in delivered] == made[0].returned
    assert made[0].returned                       # the stub did deliver units
    assert all(entry["bytes"] == _UNIT and entry["codec"] == _CODEC
               for _, entry in delivered)
    # Units arrive behind their line: that is the encoder's pipeline.
    assert all(entry["step"] <= line_step for line_step, entry in delivered)
    assert any(entry["step"] < line_step for line_step, entry in delivered)


def test_feed_failure_drops_video_and_keeps_data_flowing(tmp_path, monkeypatch,
                                                         capsys):
    made = _stub_writers(monkeypatch, lag=0, fail_after=3)
    seen = []

    def on_line(line, frames, dropped):
        seen.append((line["step"], frames))

    sink = _sink(tmp_path, video="nvenc", on_line=on_line)
    _drive(sink, _FrameAdapter(frame=_frame()), 6)
    sink.finalize_episode("completed")
    reader = RecordingReader(sink.recording_dir)
    assert sink.active is True                    # the sink survives
    assert capsys.readouterr().err.count("Web warning") == 1
    assert made[0].closes == 1                    # dropped writers are closed
    assert [step for step, _ in seen] == [0, 1, 2, 3, 4, 5]
    assert reader.num_steps() == 6                # data lines keep flowing
    assert [step for step, frames in seen if frames] == [0, 1, 2]


def test_web_extra_ships_a_jpeg_encoder():
    # The recorder degrades to frameless recording when neither cv2 nor
    # PIL imports (warned once, data still recorded). A bare
    # ``pip install specola[web]`` must therefore carry one encoder, or
    # any env whose simulator does not ship cv2 (e.g. pybullet) serves
    # a permanently frameless dashboard.
    import re
    from pathlib import Path

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    web_extra = re.search(r"^web = \[(.*?)\]", pyproject,
                          re.DOTALL | re.MULTILINE)
    assert web_extra is not None
    assert re.search(r"Pillow|opencv", web_extra.group(1))
