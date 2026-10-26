"""Recording format: serialization, writer/reader round-trip, O(1) seek."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
from _dashboard_fixtures import MANIFEST, make_snapshots

from specola.dashboard.recording import (
    FRAME_STORE_VIDEO,
    RecordingReader,
    RecordingWriter,
    VideoFrameReader,
    frame_path,
    snapshot_to_line,
)
from specola.dashboard.video import SegmentWriter

X264_ARGS = ["-c:v", "libx264", "-preset", "ultrafast", "-tune",
             "zerolatency", "-g", "1", "-pix_fmt", "yuv420p"]
VIDEO_STEPS = list(range(1, 40, 2))    # min_step 1, one frame every two steps
PER_SEGMENT = 8


def _write(tmp_path, steps, manifest=MANIFEST):
    writer = RecordingWriter(tmp_path / "rec", manifest)
    seen = set()
    for index, snapshot in enumerate(make_snapshots(steps=steps)):
        writer.append_step(snapshot_to_line(
            snapshot, wall_time=float(index), seen=seen))
    return writer


def _feed_frames(segments, start, stop):
    """Feed ``VIDEO_STEPS[start:stop]`` as solid-color frames."""
    for ordinal in range(start, stop):
        segments.feed(np.full((48, 64, 3), (ordinal * 11) % 200, np.uint8),
                      VIDEO_STEPS[ordinal])


def _annexb(reader, step):
    """``annexb`` asserted present, so the assertions below can index
    the pair without tripping over the Optional."""
    found = reader.annexb(step)
    assert found is not None
    return found


def _video_recording(tmp_path, frames=len(VIDEO_STEPS)):
    """A recording whose camera lives in the video store: a real libx264
    encode, with segments short enough that these few frames span three
    of them. The caller closes the returned writer."""
    directory = tmp_path / "rec"
    writer = _write(tmp_path, steps=4)
    writer.set_frame_store(FRAME_STORE_VIDEO)
    segments = SegmentWriter(directory, "cam", 64, 48,
                             frames_per_segment=PER_SEGMENT,
                             encoder_args=X264_ARGS)
    _feed_frames(segments, 0, frames)
    return directory, segments


def test_metadata_emitted_once_then_deltas():
    snapshots = make_snapshots(steps=3)
    seen = set()
    lines = [snapshot_to_line(snapshot, wall_time=float(index), seen=seen)
             for index, snapshot in enumerate(snapshots)]
    assert "new_bindings" in lines[0] and "new_specs" in lines[0]
    assert "new_bindings" not in lines[1] and "new_specs" not in lines[1]
    assert lines[1]["wall_time"] == 1.0  # recorder stamps wall time
    # transitions are per-step deltas; concatenated they equal the
    # binding's accumulated history
    total = sum(len(line["transitions"]) for line in lines)
    accumulated = snapshots[-1].specifications[0].bindings[0].transitions
    assert total == len(accumulated)


def test_writer_reader_round_trip(tmp_path):
    writer = _write(tmp_path, steps=4)
    writer.write_frame("cam", 0, b"jpeg0")
    writer.write_frame("cam", 2, b"jpeg2")
    reader = RecordingReader(tmp_path / "rec")
    assert reader.num_steps() == 4
    assert reader.read_step(2)["step"] == 2          # O(1) point read
    assert reader.frame("cam", 3) == (b"jpeg2", 2)   # nearest earlier sample
    assert reader.frame("cam", 0) == (b"jpeg0", 0)
    assert reader.frame("cam", -1) is None
    assert reader.cameras() == ["cam"]
    assert frame_path(tmp_path / "rec", "cam", 2).name == "00000002.jpg"


def test_nearest_step_serves_earlier_truth(tmp_path):
    _write(tmp_path, steps=4)                        # steps 0..3
    reader = RecordingReader(tmp_path / "rec")
    assert reader.nearest_step(2) == 2               # exact hit
    assert reader.nearest_step(99) == 3              # past the head
    assert reader.nearest_step(-1) is None           # before the first
    # The live-seek race this exists for: a sought step not yet
    # flushed must resolve to the closest earlier recorded line
    # rather than a silent miss (the client otherwise keeps showing
    # the live head after a seek).
    assert reader.read_step(reader.nearest_step(99))["step"] == 3


def test_reader_discards_trailing_partial_line(tmp_path):
    _write(tmp_path, steps=2)
    with open(tmp_path / "rec" / "snapshots.jsonl", "a") as fh:
        fh.write('{"step": 2, "trunca')      # simulated crash mid-write
    reader = RecordingReader(tmp_path / "rec")
    assert reader.num_steps() == 2


def test_refresh_sees_appended_lines(tmp_path):
    writer = RecordingWriter(tmp_path / "rec", MANIFEST)
    seen = set()
    snapshots = make_snapshots(steps=4)
    writer.append_step(snapshot_to_line(snapshots[0], wall_time=0.0, seen=seen))
    reader = RecordingReader(tmp_path / "rec")
    assert reader.num_steps() == 1
    for index, snapshot in enumerate(snapshots[1:], start=1):
        writer.append_step(snapshot_to_line(
            snapshot, wall_time=float(index), seen=seen))
    assert reader.refresh() == 3
    assert reader.read_step(3)["step"] == 3


def test_finalize_gzips_and_reader_reopens(tmp_path):
    writer = _write(tmp_path, steps=3)
    writer.finalize("completed")
    assert not (tmp_path / "rec" / "snapshots.jsonl").exists()
    assert (tmp_path / "rec" / "snapshots.jsonl.gz").exists()
    reader = RecordingReader(tmp_path / "rec")
    assert reader.manifest["status"] == "completed"
    assert reader.num_steps() == 3
    assert reader.read_step(1)["step"] == 1
    assert [line["step"] for line in reader.all_lines()] == [0, 1, 2]


def test_finalize_is_idempotent(tmp_path):
    writer = _write(tmp_path, steps=1)
    writer.finalize("truncated")
    writer.finalize("failed")            # second call must be a no-op
    reader = RecordingReader(tmp_path / "rec")
    assert reader.manifest["status"] == "truncated"


def test_non_finite_robustness_serializes_as_null():
    snapshot = make_snapshots(steps=1)[0]
    binding = dataclasses.replace(
        snapshot.specifications[0].bindings[0],
        current_robustness=float("inf"))
    spec = dataclasses.replace(
        snapshot.specifications[0], bindings=(binding,))
    snapshot = dataclasses.replace(snapshot, specifications=(spec,))
    line = snapshot_to_line(snapshot, wall_time=0.0, seen=set())
    encoded = json.dumps(line, allow_nan=False)      # must not raise
    assert json.loads(encoded)["specs"][0]["bindings"][0]["rho"] is None


def test_manifest_defaults_to_the_jpeg_store(tmp_path):
    _write(tmp_path, steps=1)
    manifest = json.loads((tmp_path / "rec" / "manifest.json").read_text())
    assert manifest["frame_store"] == "jpeg"
    assert manifest["format_version"] == 1      # additive: readers unchanged


def test_video_store_bumps_the_manifest_version(tmp_path):
    writer = _write(tmp_path, steps=1)
    writer.set_frame_store(FRAME_STORE_VIDEO)
    reader = RecordingReader(tmp_path / "rec")
    assert reader.manifest["frame_store"] == "video_segments"
    assert reader.manifest["format_version"] == 2


def test_recording_without_a_frame_store_key_still_reads(tmp_path):
    # Recordings written before the video store existed: version 1, no
    # frame_store key at all.
    writer = _write(tmp_path, steps=2)
    writer.write_frame("cam", 0, b"jpeg0")
    (tmp_path / "rec" / "manifest.json").write_text(json.dumps(MANIFEST))
    reader = RecordingReader(tmp_path / "rec")
    assert "frame_store" not in reader.manifest
    assert reader.cameras() == ["cam"]
    assert reader.frame_entry("cam", 1) == (b"jpeg0", "jpeg", 0)


def test_video_frame_reader_serves_exact_and_nearest_earlier_steps(tmp_path):
    directory, segments = _video_recording(tmp_path)
    segments.close()
    reader = VideoFrameReader.open(directory, "cam")
    assert reader is not None
    assert reader.codec.startswith("avc1.")
    assert reader.steps() == VIDEO_STEPS

    unit, actual = _annexb(reader, 5)                # sampled step, exact hit
    index = json.loads(
        (directory / "video" / "cam" / "index.json").read_text())
    assert actual == 5
    # self-contained: the parameter sets lead the unit, each behind a
    # start code, and the picture follows
    assert unit.startswith(b"\x00\x00\x00\x01" + bytes.fromhex(index["sps"]))
    assert bytes.fromhex(index["pps"]) in unit
    assert unit.count(b"\x00\x00\x00\x01") >= 3
    assert _annexb(reader, 6)[1] == 5                # nearest earlier sample
    assert _annexb(reader, 999)[1] == VIDEO_STEPS[-1]   # past tail: the head
    assert reader.annexb(0) is None                  # before the first sample


def test_video_frame_reader_open_rejects_absent_and_corrupt_indexes(tmp_path):
    directory, segments = _video_recording(tmp_path, frames=PER_SEGMENT + 2)
    segments.close()
    assert VideoFrameReader.open(directory, "nosuchcam") is None
    (directory / "video" / "cam" / "index.json").write_text("{oops")
    assert VideoFrameReader.open(directory, "cam") is None
    # a version-2 recording whose index is unreadable degrades to
    # no frames; it never raises at the reader
    reader = RecordingReader(directory)
    assert reader.manifest["format_version"] == 2
    assert reader.cameras() == []
    assert reader.frame_entry("cam", 5) is None


def test_video_frame_reader_reloads_the_index_as_segments_close(tmp_path):
    # Live seek: the writer rewrites index.json every time a segment
    # closes, so a reader opened mid-run must see the later segments.
    directory, segments = _video_recording(tmp_path, frames=PER_SEGMENT + 4)
    index_path = directory / "video" / "cam" / "index.json"
    reader = None
    fed = PER_SEGMENT + 4
    try:
        # the encoder's lag decides which feed call sees segment 0 close
        while fed < len(VIDEO_STEPS) and not index_path.exists():
            _feed_frames(segments, fed, fed + 1)
            fed += 1
        reader = VideoFrameReader.open(directory, "cam")
        assert reader is not None
        # Frames of the segment ffmpeg still has open are in the index
        # by step, but their bytes are not readable yet, so a seek into
        # them serves the last frame of the last closed segment.
        indexed = sum(len(segment["packets"]) for segment
                      in json.loads(index_path.read_text())["segments"])
        assert 0 < indexed < fed
        unit, actual = _annexb(reader, VIDEO_STEPS[fed - 1])
        assert actual == VIDEO_STEPS[indexed - 1]
        assert unit.startswith(b"\x00\x00\x00\x01")
        _feed_frames(segments, fed, len(VIDEO_STEPS))
    finally:
        segments.close()
    assert reader is not None
    assert reader.steps() == VIDEO_STEPS            # re-read, never cached
    assert _annexb(reader, VIDEO_STEPS[-1])[1] == VIDEO_STEPS[-1]


def test_frame_entry_serves_the_video_store(tmp_path):
    directory, segments = _video_recording(tmp_path)
    segments.close()
    reader = RecordingReader(directory)
    entry = reader.frame_entry("cam", 6)
    assert entry is not None
    unit, codec, actual = entry
    assert codec.startswith("avc1.") and actual == 5
    assert unit.startswith(b"\x00\x00\x00\x01")
    assert reader.frame_entry("cam", 0) is None
    assert reader.frame_entry("nosuchcam", 5) is None


def test_frame_entry_serves_the_jpeg_store(tmp_path):
    writer = _write(tmp_path, steps=4)
    writer.write_frame("cam", 2, b"jpeg2")
    reader = RecordingReader(tmp_path / "rec")
    assert reader.frame_entry("cam", 3) == (b"jpeg2", "jpeg", 2)
    assert reader.frame("cam", 3) == (b"jpeg2", 2)      # left untouched
    assert reader.frame_entry("cam", 1) is None


def test_cameras_and_frame_steps_cover_the_video_store(tmp_path):
    # RunIndex.from_reader and the view-mode camera pick read these, so
    # a pure video recording would otherwise serve no cameras at all.
    directory, segments = _video_recording(tmp_path)
    segments.close()
    jpeg = frame_path(directory, "jpegcam", 0)
    jpeg.parent.mkdir(parents=True)
    jpeg.write_bytes(b"jpeg0")
    reader = RecordingReader(directory)
    assert reader.cameras() == ["cam", "jpegcam"]
    assert reader.frame_steps("cam") == VIDEO_STEPS
    assert reader.frame_steps("jpegcam") == [0]
    assert reader.frame_steps("nosuchcam") == []


def test_poses_roundtrip_and_seek(tmp_path):
    from specola.dashboard.recording import PosesReader, PosesWriter
    writer = PosesWriter(tmp_path, nbody=2)
    for step in range(3):
        xpos = np.full((2, 3), step, dtype=np.float32)
        xquat = np.full((2, 4), step + 10, dtype=np.float32)
        writer.append(xpos, xquat)
    writer.close()
    reader = PosesReader.open(tmp_path)
    assert reader is not None and reader.nbody == 2 and reader.count == 3
    record = reader.record(1)
    assert record is not None
    values = np.frombuffer(record, dtype="<f4")
    assert values[:6].tolist() == [1.0] * 6 and values[6:].tolist() == [11.0] * 8
    assert reader.record(3) is None


def test_poses_reader_ignores_truncated_tail(tmp_path):
    from specola.dashboard.recording import PosesReader, PosesWriter
    writer = PosesWriter(tmp_path, nbody=2)
    writer.append(np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32))
    writer.close()
    with (tmp_path / "poses.bin").open("ab") as handle:
        handle.write(b"\x00" * 5)          # partial record
    reader = PosesReader.open(tmp_path)
    assert reader is not None
    assert reader.count == 1 and reader.record(1) is None


def test_poses_reader_absent(tmp_path):
    from specola.dashboard.recording import PosesReader
    assert PosesReader.open(tmp_path) is None


def test_poses_reader_rejects_bad_header(tmp_path):
    from specola.dashboard.recording import PosesReader
    (tmp_path / "poses.bin").write_bytes(b"XXXX" + b"\x00" * 12)
    assert PosesReader.open(tmp_path) is None
    (tmp_path / "poses.bin").write_bytes(b"SP")     # short header
    assert PosesReader.open(tmp_path) is None


def test_poses_reader_sees_records_appended_after_open(tmp_path):
    # Live mode: the server's reader is open while the writer appends.
    from specola.dashboard.recording import PosesReader, PosesWriter
    writer = PosesWriter(tmp_path, nbody=1)
    writer.append(np.zeros((1, 3), np.float32), np.zeros((1, 4), np.float32))
    reader = PosesReader.open(tmp_path)
    assert reader is not None and reader.count == 1
    writer.append(np.ones((1, 3), np.float32), np.ones((1, 4), np.float32))
    assert reader.count == 2                       # re-stats, never caches
    assert reader.record(1) is not None
    writer.close()


def test_geometry_roundtrip(tmp_path):
    from specola.dashboard.recording import read_geometry, write_geometry
    payload = {"json": {"format_version": 1, "nbody": 1, "ngeom": 0,
                        "bodies": [], "geoms": [], "meshes": [],
                        "faces_start": 0, "skipped": {}},
               "meshes": b"\x01\x02"}
    write_geometry(tmp_path, payload)
    loaded = read_geometry(tmp_path)
    assert loaded == payload
    assert read_geometry(tmp_path / "missing") is None
