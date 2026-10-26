"""DashboardServer over the socketio test client — sim-free.

Pattern: HRC/monitor/tests/test_dashboard_server.py. Recordings are
built with RecordingWriter + tracker-driven snapshots; live behavior is
exercised by calling server.on_line() directly (the recorder writer
thread's entry point), which is deterministic on the test thread.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask_socketio")

from _dashboard_fixtures import MANIFEST, make_snapshots
# One source for the video-fixture vocabulary: the encoder args, the
# sampled step list (first sample at step 1) and the segment length are
# the recording tests' and are shared, never restated.
from test_dashboard_recording import (
    PER_SEGMENT,
    VIDEO_STEPS,
    X264_ARGS,
    _feed_frames,
)

from specola.dashboard.recording import (
    FRAME_STORE_VIDEO,
    PosesWriter,
    RecordingReader,
    RecordingWriter,
    snapshot_to_line,
    write_geometry,
)
from specola.dashboard.server import DashboardServer
from specola.dashboard.video import SegmentWriter

_SCENE = {"format_version": 1, "nbody": 1, "ngeom": 0, "bodies": [],
          "geoms": [], "meshes": [], "faces_start": 0, "skipped": {}}
_AVC = "avc1.42C01F"


def _lines(steps):
    seen = set()
    return [snapshot_to_line(snapshot, wall_time=float(index), seen=seen)
            for index, snapshot in enumerate(make_snapshots(steps=steps))]


def _recording(tmp_path, steps=5, frames_at=(0, 2, 4), finalize="completed"):
    directory = tmp_path / "rec"
    writer = RecordingWriter(directory, dict(MANIFEST))
    for line in _lines(steps):
        writer.append_step(line)
    for step in frames_at:
        writer.write_frame("cam", step, "jpeg{}".format(step).encode())
    if finalize:
        writer.finalize(finalize)
    return directory


def _video_writer(tmp_path, frames):
    """A recording whose camera lives in the video store: a real libx264
    encode over VIDEO_STEPS, with a data line for every step those
    samples fall in. The caller closes the returned segment writer."""
    directory = tmp_path / "rec"
    writer = RecordingWriter(directory, dict(MANIFEST))
    for line in _lines(VIDEO_STEPS[-1] + 1):
        writer.append_step(line)
    writer.set_frame_store(FRAME_STORE_VIDEO)
    segments = SegmentWriter(directory, "cam", 64, 48,
                             frames_per_segment=PER_SEGMENT,
                             encoder_args=X264_ARGS)
    _feed_frames(segments, 0, frames)
    return writer, directory, segments


def _video_recording(tmp_path, frames=len(VIDEO_STEPS)):
    writer, directory, segments = _video_writer(tmp_path, frames)
    segments.close()
    writer.finalize("completed")
    return directory


def _open_video_recording(tmp_path):
    """A video recording ffmpeg is still writing, fed just past the
    close of its first segment: the steps after it are indexed but their
    bytes are not readable yet."""
    writer, directory, segments = _video_writer(tmp_path, PER_SEGMENT + 4)
    index_path = directory / "video" / "cam" / "index.json"
    fed = PER_SEGMENT + 4
    # the encoder's lag decides which feed call sees segment 0 close
    while fed < len(VIDEO_STEPS) and not index_path.exists():
        _feed_frames(segments, fed, fed + 1)
        fed += 1
    assert index_path.exists()
    return directory, segments, fed


def _entry(step, codec=_AVC):
    """One wire frame entry as the recorder pushes it: bytes, the step
    they were encoded for, and the codec they are in."""
    return {"bytes": b"unit-%d" % step, "step": step, "codec": codec}


def _connect(server):
    http = server.app.test_client()
    client = server.socketio.test_client(server.app, flask_test_client=http)
    return http, client


def _received(client, name):
    return [message for message in client.get_received()
            if message["name"] == name]


def _seek(client, step, **payload):
    ack = client.emit("view", dict(payload, seek=step), callback=True)
    assert ack["ok"] is True
    return _received(client, "step_bundle")[-1]["args"][0]


def _write_scene(directory, values, nbody=1):
    """Write the 3D scene files: geometry plus one pose record per value.

    Returns the records in write order, which is the order the server
    must join them to data lines by (record i belongs to line i, whatever
    step numbers the run used).
    """
    write_geometry(directory, {"json": _SCENE, "meshes": b"\x01\x02"})
    writer = PosesWriter(directory, nbody=nbody)
    records = []
    for value in values:
        xpos = np.full((nbody, 3), value, dtype=np.float32)
        xquat = np.full((nbody, 4), value, dtype=np.float32)
        writer.append(xpos, xquat)
        records.append(xpos.tobytes() + xquat.tobytes())
    writer.close()
    return records


def test_page_serves_and_connect_gets_run_meta(tmp_path):
    server = DashboardServer(_recording(tmp_path), mode="view")
    http = server.app.test_client()
    page = http.get("/")
    assert page.status_code == 200
    assert b"cdn." not in page.data
    assert http.get(
        "/static/vendor/socket.io.min.js").status_code == 200
    client = server.socketio.test_client(server.app, flask_test_client=http)
    metas = _received(client, "run_meta")
    assert len(metas) == 1
    payload = metas[0]["args"][0]
    assert payload["protocol_version"] == 1
    assert payload["mode"] == "view"
    assert payload["max_step"] == 4
    assert payload["bindings"] and payload["specs"]
    assert payload["manifest"]["status"] == "completed"
    client.disconnect()


def test_seek_is_point_read_with_epoch_bump(tmp_path):
    server = DashboardServer(_recording(tmp_path), mode="view")
    _, client = _connect(server)
    client.get_received()
    ack = client.emit("view", {"seek": 3}, callback=True)
    assert ack["ok"] is True and ack["epoch"] == 1 and ack["position"] == 3
    bundles = _received(client, "step_bundle")
    assert len(bundles) == 1
    bundle = bundles[0]["args"][0]
    assert bundle["step"] == 3 and bundle["epoch"] == 1 and bundle["seq"] == 0
    # nearest-earlier frame captioning: step 3 shows the step-2 frame
    assert bundle["frames"]["cam"]["step"] == 2
    assert bundle["frames"]["cam"]["bytes"] == b"jpeg2"
    assert bundle["frames"]["cam"]["codec"] == "jpeg"
    client.disconnect()


def test_seek_serves_the_video_store_with_its_codec(tmp_path):
    server = DashboardServer(_video_recording(tmp_path), mode="view")
    _, client = _connect(server)
    client.get_received()
    frame = _seek(client, 6, cameras=["cam"])["frames"]["cam"]
    assert frame["codec"].startswith("avc1.")
    # a self-contained access unit, nearest earlier sampled step
    assert frame["bytes"].startswith(b"\x00\x00\x00\x01")
    assert frame["step"] == 5
    # Before the first sample (step 1) the pane still shows it, with an
    # honest caption — the same clamp the JPEG store applies.
    early = _seek(client, 0, cameras=["cam"])["frames"]["cam"]
    assert early["step"] == 1
    client.disconnect()


def test_view_mode_picks_a_video_camera_without_explicit_cameras(tmp_path):
    # A pure video recording has no frames/*/*.jpg for the default
    # camera pick to find; it must come off the video index instead.
    server = DashboardServer(_video_recording(tmp_path), mode="view")
    _, client = _connect(server)
    client.get_received()
    frame = _seek(client, 10)["frames"]["cam"]
    assert frame["step"] == 9 and frame["codec"].startswith("avc1.")
    client.disconnect()


def test_open_segment_seeks_do_not_shadow_the_closed_frame(tmp_path):
    # Steps inside the segment ffmpeg still has open serve the last
    # readable frame instead. Caching that under the requested step
    # would keep serving it after the segment closes.
    directory, segments, fed = _open_video_recording(tmp_path)
    target = VIDEO_STEPS[fed - 1]
    try:
        server = DashboardServer(directory, mode="view")
        _, client = _connect(server)
        client.get_received()
        first = _seek(client, target, cameras=["cam"])["frames"]["cam"]
        assert first["step"] < target
        assert _seek(client, target, cameras=["cam"])["frames"]["cam"] == first
    finally:
        segments.close()
    served = _seek(client, target, cameras=["cam"])["frames"]["cam"]
    assert served["step"] == target
    assert served["bytes"] != first["bytes"]
    client.disconnect()


def test_offset_start_recording_seeks_clamp_to_first_step(tmp_path):
    # Tracker timesteps need not start at 0 (robocasa starts at 1): a
    # seek below the first recorded step must clamp to it, not vanish.
    directory = tmp_path / "rec"
    writer = RecordingWriter(directory, dict(MANIFEST))
    for line in _lines(3):
        line = dict(line)
        line["step"] += 1                    # steps 1..3, no step 0
        writer.append_step(line)
    writer.finalize("completed")
    server = DashboardServer(directory, mode="view")
    _, client = _connect(server)
    meta = _received(client, "run_meta")[0]["args"][0]
    assert meta["min_step"] == 1 and meta["max_step"] == 3
    ack = client.emit("view", {"seek": 0}, callback=True)
    assert ack["ok"] is True and ack["position"] == 1
    bundle = _received(client, "step_bundle")[-1]["args"][0]
    assert bundle["step"] == 1
    client.disconnect()


def test_connect_gets_geometry_when_the_recording_has_a_scene(tmp_path):
    directory = _recording(tmp_path)
    _write_scene(directory, [0.0] * 5)
    server = DashboardServer(directory, mode="view")
    _, client = _connect(server)
    events = _received(client, "geometry")
    assert len(events) == 1
    payload = events[0]["args"][0]
    assert payload["protocol_version"] == 1 and payload["run_id"] == "r1"
    assert payload["json"] == _SCENE
    assert payload["meshes"] == b"\x01\x02"
    client.disconnect()


def test_seek_reads_the_pose_record_by_ordinal_not_step(tmp_path):
    # Steps start at 1 (robocasa): record i belongs to the i-th data
    # line, so seeking step 2 must serve record 1, not record 2.
    directory = tmp_path / "rec"
    writer = RecordingWriter(directory, dict(MANIFEST))
    for line in _lines(3):
        line = dict(line)
        line["step"] += 1                    # steps 1..3, no step 0
        writer.append_step(line)
    writer.finalize("completed")
    records = _write_scene(directory, [10.0, 20.0, 30.0])
    server = DashboardServer(directory, mode="view")
    _, client = _connect(server)
    client.get_received()
    ack = client.emit("view", {"seek": 2}, callback=True)
    assert ack["ok"] is True
    bundle = _received(client, "step_bundle")[-1]["args"][0]
    assert bundle["step"] == 2
    assert bundle["poses"] == {"bytes": records[1], "step": 2}
    client.disconnect()


def test_recording_without_a_scene_has_no_geometry_or_poses(tmp_path):
    server = DashboardServer(_recording(tmp_path), mode="view")
    _, client = _connect(server)
    assert not _received(client, "geometry")
    ack = client.emit("view", {"seek": 3}, callback=True)
    assert ack["ok"] is True
    bundle = _received(client, "step_bundle")[-1]["args"][0]
    assert "poses" not in bundle
    client.disconnect()


def test_seek_clamps_out_of_range(tmp_path):
    server = DashboardServer(_recording(tmp_path), mode="view")
    _, client = _connect(server)
    client.get_received()
    ack = client.emit("view", {"seek": 99}, callback=True)
    assert ack["ok"] is True and ack["position"] == 4
    ack = client.emit("view", {"seek": -3}, callback=True)
    assert ack["ok"] is True and ack["position"] == 0
    client.disconnect()


def _live_server(tmp_path, prefill=0):
    directory = tmp_path / "rec"
    writer = RecordingWriter(directory, dict(MANIFEST))
    lines = _lines(6)
    for line in lines[:prefill]:
        writer.append_step(line)
    server = DashboardServer(directory, mode="live")
    return server, writer, lines


def test_live_follow_latest_only(tmp_path):
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    for line in lines[:3]:
        writer.append_step(line)
        server.on_line(line, {"cam": _entry(line["step"], codec="jpeg")}, 0)
    # No acks sent: bundle 0 in flight, 1 replaced by 2 in pending.
    bundles = _received(client, "step_bundle")
    with server._clients_lock:
        state = next(iter(server._clients.values()))
    assert state.delivery.dropped == 1
    assert bundles[0]["args"][0]["step"] == 0
    assert bundles[0]["args"][0]["live"] is True
    client.disconnect()


def test_live_bundle_carries_the_pushed_poses(tmp_path):
    server, writer, lines = _live_server(tmp_path)
    records = _write_scene(server.recording_dir, [1.0])
    _, client = _connect(server)
    client.get_received()
    writer.append_step(lines[0])
    server.on_line(lines[0], {}, 0, poses=records[0])
    bundle = _received(client, "step_bundle")[0]["args"][0]
    assert bundle["poses"] == {"bytes": records[0], "step": lines[0]["step"]}
    client.disconnect()


def test_geometry_reaches_a_client_that_connected_before_the_files(tmp_path):
    # Live race: the client connects before step 0, so the scene files
    # do not exist yet; the first line must carry the event to it.
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    assert not _received(client, "geometry")
    records = _write_scene(server.recording_dir, [1.0, 2.0])
    writer.append_step(lines[0])
    server.on_line(lines[0], {}, 0, poses=records[0])
    assert len(_received(client, "geometry")) == 1
    writer.append_step(lines[1])
    server.on_line(lines[1], {}, 0, poses=records[1])
    assert not _received(client, "geometry")          # once, never again
    client.disconnect()


def test_geometry_is_not_repeated_after_a_connect_time_emit(tmp_path):
    server, writer, lines = _live_server(tmp_path)
    records = _write_scene(server.recording_dir, [1.0])
    _, client = _connect(server)
    assert len(_received(client, "geometry")) == 1     # on connect
    writer.append_step(lines[0])
    server.on_line(lines[0], {}, 0, poses=records[0])
    assert not _received(client, "geometry")           # not again
    client.disconnect()


def test_detached_client_gets_run_status_growth(tmp_path):
    server, writer, lines = _live_server(tmp_path, prefill=2)
    _, client = _connect(server)
    client.get_received()
    ack = client.emit("view", {"seek": 0}, callback=True)   # detach
    assert ack["ok"] is True
    client.get_received()
    writer.append_step(lines[2])
    server.on_line(lines[2], {}, 0)
    server._last_status_at = 0.0                     # defeat the throttle
    writer.append_step(lines[3])
    server.on_line(lines[3], {}, 0)
    statuses = _received(client, "run_status")
    assert statuses
    assert statuses[-1]["args"][0]["max_step"] == 3
    assert not _received(client, "step_bundle")      # position undisturbed
    client.disconnect()


def test_new_bindings_reemit_run_meta(tmp_path):
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    writer.append_step(lines[0])                     # carries new_bindings
    server.on_line(lines[0], {}, 0)
    metas = _received(client, "run_meta")
    assert metas and metas[-1]["args"][0]["bindings"]
    client.disconnect()


def test_live_frame_entries_go_out_as_the_recorder_pushed_them(tmp_path):
    # Access units lag the line they arrive with; the entry names the
    # step its bytes were encoded for, and that is what the client and
    # the sampled-step index must both see.
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    for line in lines[2:5]:
        writer.append_step(line)
        server.on_line(line, {"cam": _entry(line["step"] - 2)}, 0)
    bundle = _received(client, "step_bundle")[0]["args"][0]
    assert bundle["step"] == 2
    assert bundle["frames"]["cam"] == _entry(0)      # passed through whole
    assert server.index.sampled["cam"] == [0, 1, 2]
    client.disconnect()


def test_first_frames_reload_the_manifest_and_reemit_run_meta(tmp_path):
    # The writer thread stamps the frame store into the manifest just
    # before the first frames reach this callback; with a capture stride
    # that can be long after the last new_bindings line.
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    writer.append_step(lines[0])
    server.on_line(lines[0], {}, 0)                  # new_bindings run_meta
    meta = _received(client, "run_meta")[-1]["args"][0]
    assert meta["manifest"]["frame_store"] == "jpeg"
    writer.set_frame_store(FRAME_STORE_VIDEO)
    writer.append_step(lines[1])
    server.on_line(lines[1], {"cam": _entry(0)}, 0)
    metas = _received(client, "run_meta")
    assert len(metas) == 1
    assert metas[0]["args"][0]["manifest"]["frame_store"] == "video_segments"
    assert metas[0]["args"][0]["manifest"]["format_version"] == 2
    writer.append_step(lines[2])
    server.on_line(lines[2], {"cam": _entry(1)}, 0)
    assert not _received(client, "run_meta")         # once, never again
    client.disconnect()


def test_seeks_before_the_first_segment_serve_the_last_wire_unit(tmp_path):
    # index.json does not exist until ffmpeg closes the first segment,
    # so for the opening stretch of a live run the unit last delivered
    # is the only frame there is; a seek into it gets that unit under
    # the step it was encoded for, and nothing before it.
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    for index, line in enumerate(lines[:4]):
        writer.append_step(line)
        server.on_line(line, {"cam": _entry(index - 1)} if index else {}, 0)
    assert _seek(client, 3, cameras=["cam"])["frames"]["cam"] == _entry(2)
    assert _seek(client, 0, cameras=["cam"])["frames"] == {}
    client.disconnect()


def test_disk_beats_a_stale_wire_unit(tmp_path):
    # Guards the arbitration's disk arm: a wire unit older than what the
    # store can serve must lose, or a stalled encoder would pin every
    # seek to its last delivery.
    server = DashboardServer(_video_recording(tmp_path), mode="view")
    server._wire_frames["cam"] = {"bytes": b"\x00\x00\x00\x01old",
                                  "step": 1, "codec": "avc1.42C01F"}
    _, client = _connect(server)
    client.get_received()
    frame = _seek(client, 10, cameras=["cam"])["frames"]["cam"]
    assert frame["step"] == 9                    # the disk's newer frame
    client.disconnect()


def test_series_window_decimated(tmp_path):
    server = DashboardServer(_recording(tmp_path, steps=5), mode="view")
    _, client = _connect(server)
    client.get_received()
    binding = list(server.index.bindings)[0]
    ack = client.emit(
        "view", {},
        callback=True)                               # no-op change is fine
    ack = client.emit(
        "series_window",
        {"bindings": [list(binding)], "start": 0, "end": 4, "px": 100},
        callback=True)
    assert ack["ok"] is True
    assert len(ack["series"]) == 1
    spec_name, key, points = ack["series"][0]
    assert (spec_name, key) == binding
    assert [point[0] for point in points] == [0, 1, 2, 3, 4]
    client.disconnect()


def test_transitions_page(tmp_path):
    server = DashboardServer(_recording(tmp_path, steps=5), mode="view")
    _, client = _connect(server)
    ack = client.emit("transitions_page", {"offset": 0, "limit": 2},
                      callback=True)
    assert ack["ok"] is True
    assert ack["total"] == len(server.index.transitions)
    assert len(ack["entries"]) <= 2
    client.disconnect()


def test_stop_run_sets_event_live_only(tmp_path):
    server, writer, lines = _live_server(tmp_path, prefill=1)
    fired = []
    server.on_stop_run(lambda: fired.append(True))
    _, client = _connect(server)
    # The real client sends an empty payload object — keep parity.
    ack = client.emit("stop_run", {}, callback=True)
    assert ack["ok"] is True
    assert server.stop_requested.is_set() and fired == [True]
    client.disconnect()

    view_server = DashboardServer(_recording(tmp_path / "v"), mode="view")
    _, view_client = _connect(view_server)
    ack = view_client.emit("stop_run", {}, callback=True)
    assert ack["ok"] is False
    view_client.disconnect()


def test_episode_end_broadcast_and_reader_swap(tmp_path):
    server, writer, lines = _live_server(tmp_path)
    _, client = _connect(server)
    client.get_received()
    for line in lines[:3]:
        writer.append_step(line)
        server.on_line(line, {}, 0)
    writer.finalize("truncated")                     # gzips + unlinks jsonl
    server.on_episode_end("truncated")
    ended = _received(client, "episode_end")
    assert ended
    payload = ended[-1]["args"][0]
    assert payload["status"] == "truncated"
    assert payload["view_command"].startswith("specola view ")
    # point reads still work against the swapped (gz) reader
    ack = client.emit("view", {"seek": 1}, callback=True)
    assert ack["ok"] is True
    bundle = _received(client, "step_bundle")[-1]["args"][0]
    assert bundle["step"] == 1
    # replay unlocks: play advances position through delivered bundles
    ack = client.emit("view", {"seek": 0, "play": True, "rate": 8.0},
                      callback=True)
    assert ack["ok"] is True
    deadline = 50
    state = next(iter(server._clients.values()))
    while state.playing and deadline:
        server.socketio.sleep(0.05)
        deadline -= 1
    assert state.position == 2
    client.disconnect()


def test_busy_port_raises_before_thread(tmp_path):
    import socket as socket_module

    blocker = socket_module.socket()
    blocker.bind(("127.0.0.1", 0))
    port = blocker.getsockname()[1]
    server = DashboardServer(_recording(tmp_path), mode="view")
    with pytest.raises(OSError):
        server.start(port)
    blocker.close()


def test_start_rebinds_through_time_wait(tmp_path):
    # A just-killed server leaves its socket in TIME_WAIT for ~60s.
    # werkzeug rebinds through that (allow_reuse_address), so the bind
    # probe must too, or every quick restart fails as "port busy".
    import socket as socket_module

    listener = socket_module.socket()
    listener.setsockopt(
        socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen(1)
    client = socket_module.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    conn.close()                      # active close: TIME_WAIT server-side
    client.close()
    listener.close()

    bare = socket_module.socket()
    try:
        bare.bind(("127.0.0.1", port))
    except OSError:
        pass                          # TIME_WAIT in place, as expected
    else:
        bare.close()
        pytest.skip("platform released the port immediately")

    server = DashboardServer(_recording(tmp_path), mode="view")
    url = server.start(port)
    assert url.endswith(str(port))
    server.shutdown()
