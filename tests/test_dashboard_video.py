"""ffmpeg video pipeline: Annex-B parsers, segment writer, encoder probe.

The integration tests drive a real ffmpeg through libx264 so they run on
any box; the same body runs against NVENC behind the ``nvenc`` marker.
Byte-level assertions stay encoder-agnostic (start codes, pairing, index
shape) — the NAL sequence inside an access unit differs between x264 and
NVENC and is not part of the contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from specola.dashboard.video import (
    NVENC_ARGS,
    SegmentWriter,
    codec_string,
    parse_avcc_hexdump,
    probe_encoder,
    split_annexb,
    to_annexb,
)

X264_ARGS = ["-c:v", "libx264", "-preset", "ultrafast", "-tune",
             "zerolatency", "-g", "1", "-pix_fmt", "yuv420p"]

# Captured verbatim from `ffprobe -show_streams -show_data -of json` on a
# segment written by this pipeline (ffmpeg 4.4.2, libx264, 2026-08-03).
AVCC_HEXDUMP = (
    "\n00000000: 0142 c00a ffe1 0015 6742 c00a dc47 a100  .B......gB...G..\n"
    "00000010: 0003 0001 0000 0300 288f 1227 8001 0004  ........(..'....\n"
    "00000020: 68ce 0fc8                                h...\n"
)

WIDTH, HEIGHT, FRAMES = 64, 48, 170
PER_SEGMENT = 80


def _nal(nal_type: int, payload: bytes) -> bytes:
    return b"\x00\x00\x00\x01" + bytes([nal_type]) + payload


def _access_unit(tag: bytes) -> bytes:
    return _nal(7, b"sps" + tag) + _nal(8, b"pps") + _nal(5, b"idr" + tag)


def _nal_types(annexb: bytes) -> list:
    types, offset = [], 0
    while True:
        found = annexb.find(b"\x00\x00\x01", offset)
        if found < 0 or found + 3 >= len(annexb):
            return types
        types.append(annexb[found + 3] & 0x1f)
        offset = found + 3


def test_split_annexb_returns_complete_units_and_a_remainder():
    stream = _access_unit(b"a") + _access_unit(b"b") + _access_unit(b"c")
    units, remainder = split_annexb(stream)
    assert units == [_access_unit(b"a"), _access_unit(b"b")]
    assert remainder == _access_unit(b"c")   # complete once the next starts


def test_split_annexb_survives_chunk_boundaries():
    stream = b"".join(_access_unit(bytes([65 + i])) for i in range(4))
    collected, buffer = [], b""
    for start in range(0, len(stream), 3):   # splits start codes apart
        units, buffer = split_annexb(buffer + stream[start:start + 3])
        collected.extend(units)
    collected.append(buffer)
    assert collected == [_access_unit(bytes([65 + i])) for i in range(4)]


def test_split_annexb_ignores_non_boundary_nals():
    unit = (_nal(7, b"sps") + _nal(8, b"pps") + _nal(6, b"sei")
            + _nal(5, b"idr"))
    units, remainder = split_annexb(unit + unit)
    assert units == [unit] and remainder == unit


def test_parse_avcc_hexdump_extracts_parameter_sets():
    sps, pps = parse_avcc_hexdump(AVCC_HEXDUMP)
    assert sps[0] & 0x1f == 7 and len(sps) == 21
    assert pps[0] & 0x1f == 8 and len(pps) == 4
    assert sps.hex().startswith("6742c00a")


def test_parse_avcc_hexdump_rejects_garbage():
    from specola.dashboard.video import VideoPipelineError
    with pytest.raises(VideoPipelineError):
        parse_avcc_hexdump("00000000: 9999 9999  ....\n")


def test_codec_string_from_sps():
    assert codec_string(parse_avcc_hexdump(AVCC_HEXDUMP)[0]) == "avc1.42C00A"
    # baseline 720p on this box's NVENC, per the plan's verified facts
    assert codec_string(b"\x67\x42\xc0\x1f\x8c\x8d") == "avc1.42C01F"


def test_probe_encoder_accepts_a_working_encoder():
    assert probe_encoder(X264_ARGS) is True


def test_probe_encoder_rejects_a_missing_encoder():
    assert probe_encoder(["-c:v", "nonsense"]) is False


def _run_pipeline(tmp_path, encoder_args):
    """Feed 170 frames as steps 1..170 and return the wire pairs, the
    final index, and the index and pair count as they looked mid-run."""
    directory = tmp_path / "rec"
    index_path = directory / "video" / "cam" / "index.json"
    writer = SegmentWriter(directory, "cam", WIDTH, HEIGHT,
                           encoder_args=encoder_args)
    pairs, mid_run, mid_pairs = [], None, 0
    for step in range(1, FRAMES + 1):
        frame = np.full((HEIGHT, WIDTH, 3), (step * 3) % 200, dtype=np.uint8)
        pairs.extend(writer.feed(frame, step))
        if step == 120:                 # segment 0 closed, segment 1 open
            mid_run, mid_pairs = json.loads(index_path.read_text()), len(pairs)
    writer.close()
    return pairs, json.loads(index_path.read_text()), mid_run, mid_pairs


def _assert_pipeline(tmp_path, encoder_args):
    pairs, index, mid_run, mid_pairs = _run_pipeline(tmp_path, encoder_args)

    # Units pair with fed steps in order. Both bounds are deliberately
    # slack: the wire lag is the encoder's pipeline depth plus the unit
    # boundary the reader waits for, which is not a constant across
    # encoders, and whatever the encoder still holds at close never
    # reaches the wire at all (the disk store keeps every frame).
    steps = [step for step, _ in pairs]
    assert steps == list(range(1, len(steps) + 1))
    assert len(steps) >= FRAMES - 30
    assert mid_pairs >= 100          # units flow during the run, not in a
    #                                  burst at close
    for _, unit in pairs:
        assert unit.startswith(b"\x00\x00\x00\x01")
        assert 5 in _nal_types(unit)          # all-intra: every unit an IDR

    assert index["format_version"] == 1
    assert index["codec"].startswith("avc1.")
    assert index["frames_per_segment"] == PER_SEGMENT
    assert index["steps"] == list(range(1, FRAMES + 1))
    sps, pps = bytes.fromhex(index["sps"]), bytes.fromhex(index["pps"])
    assert sps[0] & 0x1f == 7 and pps[0] & 0x1f == 8
    assert codec_string(sps) == index["codec"]

    segments = index["segments"]
    assert [len(s["packets"]) for s in segments] == [PER_SEGMENT, PER_SEGMENT,
                                                     FRAMES - 2 * PER_SEGMENT]
    assert [s["file"] for s in segments] == ["seg_%05d.mp4" % i
                                             for i in range(3)]

    # every indexed [pos, size] slice is a real sample that converts
    directory = tmp_path / "rec" / "video" / "cam"
    for segment in segments:
        data = (directory / segment["file"]).read_bytes()
        for pos, size in segment["packets"]:
            unit = to_annexb(data[pos:pos + size], sps, pps)
            assert unit.startswith(b"\x00\x00\x00\x01")
            assert 5 in _nal_types(unit)

    # The index is rewritten as each segment closes, not only at close:
    # 120 frames in, exactly one segment is closed and indexed, which is
    # what a live seek needs. Its steps are the frames fed by the time
    # that segment closed, so they run ahead of its packets but stop
    # short of the frames fed since.
    assert mid_run is not None
    assert len(mid_run["segments"]) == 1
    assert len(mid_run["segments"][0]["packets"]) == PER_SEGMENT
    assert PER_SEGMENT <= len(mid_run["steps"]) < 120
    assert mid_run["steps"] == list(range(1, len(mid_run["steps"]) + 1))


def test_segment_writer_pipeline_libx264(tmp_path):
    _assert_pipeline(tmp_path, X264_ARGS)


@pytest.mark.nvenc
def test_segment_writer_pipeline_nvenc(tmp_path):
    if not probe_encoder(NVENC_ARGS):
        pytest.skip("h264_nvenc unavailable")
    _assert_pipeline(tmp_path, NVENC_ARGS)


def test_feed_rejects_a_wrong_sized_frame(tmp_path):
    from specola.dashboard.video import VideoPipelineError
    writer = SegmentWriter(tmp_path / "rec", "cam", WIDTH, HEIGHT,
                           encoder_args=X264_ARGS)
    try:
        with pytest.raises(VideoPipelineError):
            writer.feed(np.zeros((8, 8, 3), dtype=np.uint8), 1)
    finally:
        writer.close()
