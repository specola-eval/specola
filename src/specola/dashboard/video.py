"""Hardware video encoding for the dashboard's camera frames.

One long-running ffmpeg per camera encodes raw RGB frames once and tees
two outputs: fixed-length MP4 segments on disk (the durable store, with
an index.json mapping fed steps to byte ranges) and self-contained
Annex-B access units on stdout (the live wire, decoded by WebCodecs in
the browser). This module owns that subprocess work plus the pure
parsers needed to read the output back.

Every ffmpeg flag here is load-bearing and was verified against ffmpeg
4.4.2: `-flags:v +global_header` keeps the parameter sets out of the MP4
samples, `dump_extra=freq=keyframe` puts them back on the pipe branch so
each access unit stands alone, and `-pix_fmt yuv420p` is required
because the automatic path from RGB lands on full-range yuvj420p and
breaks the browser's color assumptions. Design:
docs/superpowers/plans/2026-08-03-nvenc-sensor-pipeline.md.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from collections import deque
from io import BufferedReader, BufferedWriter
from pathlib import Path
from queue import Empty, Queue
from typing import Deque, List, Optional, Tuple, cast

import numpy as np


class VideoPipelineError(RuntimeError):
    """Any failure of the ffmpeg pipeline or of the data it produces.

    Callers apply the dashboard's failure contract: fall back to the
    JPEG frame store, never let a video problem reach the sim loop.
    """


NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
              "-g", "1", "-profile:v", "baseline", "-pix_fmt", "yuv420p"]

_SEGMENT_PATTERN = "seg_%05d.mp4"
_SEGMENT_NAME = "seg_{:05d}.mp4"
_INDEX_FILE = "index.json"
_INDEX_VERSION = 1
_DEFAULT_FPS = 20
_FRAMES_PER_SEGMENT = 80
_CLOSE_TIMEOUT = 30.0
_READ_SIZE = 1 << 20

# An access unit starts at a start code introducing a sequence parameter
# set or an access unit delimiter. In the all-intra streams this module
# produces that is one IDR picture per unit, whatever the encoder puts
# in between (x264 emits SPS,PPS,IDR with an SEI banner on the first
# unit; NVENC emits SPS,PPS,SEI,SEI,IDR throughout).
_AU_FIRST_NAL_TYPES = frozenset((7, 9))

_HEXDUMP_LINE = re.compile(r"^[0-9a-fA-F]{4,}:\s*(.*)$")


def split_annexb(buffer: bytes) -> Tuple[List[bytes], bytes]:
    """Split a raw h264 byte stream into complete access units.

    Returns the complete units and the trailing remainder, which the
    caller re-submits with the next chunk. A unit is only complete once
    the following one has begun, so the newest unit always stays in the
    remainder until its successor arrives. Bytes ahead of the first
    start code cannot belong to a complete unit and are dropped.
    """
    starts = []
    index = 0
    while True:
        found = buffer.find(b"\x00\x00\x01", index)
        if found < 0:
            break
        index = found + 3
        if index >= len(buffer):
            break                       # NAL type byte has not arrived yet
        if (buffer[index] & 0x1f) in _AU_FIRST_NAL_TYPES:
            # step back onto the leading zero of a four-byte start code
            starts.append(found - 1 if found and not buffer[found - 1]
                          else found)
    if not starts:
        return [], buffer
    units = [buffer[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    return units, buffer[starts[-1]:]


def to_annexb(sample: bytes, sps: bytes, pps: bytes) -> bytes:
    """One MP4 sample to a self-contained Annex-B access unit.

    A sample carries its NALs behind four-byte length prefixes and keeps
    the parameter sets in the container, so a decoder handed the bare
    bytes finds neither start codes nor an SPS. Both go back in here:
    the prefixes become start codes and the parameter sets from the
    index lead the unit, which is what WebCodecs needs of a key frame.
    Four-byte prefixes are what this pipeline writes (ffprobe reports
    ``nal_length_size`` 4); a sample that is not a run of them is a
    corrupt read and raises.
    """
    units = [b"\x00\x00\x00\x01" + sps, b"\x00\x00\x00\x01" + pps]
    offset = 0
    while offset + 4 <= len(sample):
        length = int.from_bytes(sample[offset:offset + 4], "big")
        offset += 4
        nal = sample[offset:offset + length]
        if len(nal) != length:
            break
        units.append(b"\x00\x00\x00\x01" + nal)
        offset += length
    if offset != len(sample):
        raise VideoPipelineError("sample is not a run of length prefixes")
    return b"".join(units)


def parse_avcc_hexdump(text: str) -> Tuple[bytes, bytes]:
    """ffprobe's ``extradata`` hexdump to the raw SPS and PPS NALs.

    ffprobe prints extradata as offset-prefixed hex with a trailing
    ASCII column ("00000000: 0142 c00a ...  .B......"). The bytes are an
    AVC decoder configuration record; its parameter sets are what a
    WebCodecs decoder needs prepended to every key frame.
    """
    digits = []
    for line in text.splitlines():
        match = _HEXDUMP_LINE.match(line.strip())
        if match:
            digits.append(match.group(1).split("  ")[0].replace(" ", ""))
    try:
        return _parse_avcc(bytes.fromhex("".join(digits)))
    except ValueError as exc:
        raise VideoPipelineError(
            "unreadable avcC extradata: {}".format(exc)) from exc


def _parse_avcc(record: bytes) -> Tuple[bytes, bytes]:
    if len(record) < 7 or record[0] != 1:
        raise ValueError("not an AVC decoder configuration record")
    sps, offset = _read_length_prefixed(record, 6, record[5] & 0x1f)
    if offset >= len(record):
        raise ValueError("record ends before its parameter sets do")
    pps, _ = _read_length_prefixed(record, offset + 1, record[offset])
    if not sps or not pps:
        raise ValueError("record carries no SPS/PPS pair")
    return sps[0], pps[0]


def _read_length_prefixed(record: bytes, offset: int,
                          count: int) -> Tuple[List[bytes], int]:
    nals = []
    for _ in range(count):
        length = int.from_bytes(record[offset:offset + 2], "big")
        offset += 2
        nal = record[offset:offset + length]
        if len(nal) != length or not nal:
            raise ValueError("truncated parameter set")
        nals.append(nal)
        offset += length
    return nals, offset


def codec_string(sps: bytes) -> str:
    """WebCodecs codec string for an h264 SPS: the profile, constraint
    flags and level that follow the NAL header byte."""
    if len(sps) < 4:
        raise VideoPipelineError("SPS is too short for a codec string")
    return "avc1." + sps[1:4].hex().upper()


def probe_encoder(encoder_args: List[str]) -> bool:
    """True when the whole pipeline runs with these encoder arguments.

    Encodes a few gray frames through the real tee command and checks
    that ffmpeg exited cleanly and that parseable access units came back
    on the pipe branch, so a caller can pick the frame store without
    committing to it.
    """
    frames = bytes([128]) * (64 * 64 * 3) * 4
    with tempfile.TemporaryDirectory() as directory:
        command = _ffmpeg_command(Path(directory), 64, 64, _DEFAULT_FPS,
                                  _FRAMES_PER_SEGMENT / _DEFAULT_FPS,
                                  encoder_args, "error")
        try:
            done = subprocess.run(command, input=frames,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  timeout=_CLOSE_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return False
    return done.returncode == 0 and bool(split_annexb(done.stdout)[0])


def _tee_path(path: Path) -> str:
    """Escape a path for a tee output specification.

    The tee muxer reads ``:`` and ``|`` as syntax and ffmpeg reads a
    leading ``name:`` as a protocol, so the path is escaped and forced
    onto the file protocol.
    """
    escaped = (str(path).replace("\\", "\\\\").replace(":", "\\:")
               .replace("|", "\\|").replace("[", "\\[").replace("]", "\\]"))
    return "file\\:" + escaped


def _ffmpeg_command(video_dir: Path, width: int, height: int, fps: int,
                    segment_seconds: float, encoder_args: List[str],
                    loglevel: str) -> List[str]:
    """The fixed command template: raw RGB in, MP4 segments and Annex-B
    access units out. Only the codec block is caller-supplied."""
    tee = "[f=segment:segment_time={:g}:segment_format=mp4:{}]{}|{}".format(
        segment_seconds, "reset_timestamps=1",
        _tee_path(video_dir / _SEGMENT_PATTERN),
        "[f=h264:bsfs/v=dump_extra=freq=keyframe]pipe:1")
    return (["ffmpeg", "-hide_banner", "-loglevel", loglevel, "-nostdin",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "{}x{}".format(width, height), "-r", str(fps),
             "-i", "pipe:0", "-flags:v", "+global_header"]
            + list(encoder_args)
            + ["-f", "tee", "-map", "0:v", tee])


def _ffprobe(args: List[str]) -> dict:
    command = ["ffprobe", "-hide_banner", "-loglevel", "error",
               "-select_streams", "v", "-of", "json"] + args
    try:
        done = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=_CLOSE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoPipelineError("ffprobe failed: {}".format(exc)) from exc
    if done.returncode != 0:
        raise VideoPipelineError("ffprobe failed: {}".format(
            done.stderr.decode("utf-8", "replace").strip()))
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise VideoPipelineError("unreadable ffprobe output") from exc


def _segment_packets(path: Path) -> List[List[int]]:
    """``[pos, size]`` per sample: the absolute file offset of the
    sample's first length prefix and its exact byte length, so a reader
    slices samples straight out of the segment."""
    packets = _ffprobe(["-show_packets", str(path)]).get("packets", [])
    try:
        return [[int(packet["pos"]), int(packet["size"])]
                for packet in packets]
    except (KeyError, ValueError) as exc:
        raise VideoPipelineError(
            "unreadable ffprobe packets for {}: {}".format(path, exc)
        ) from exc


def _segment_parameter_sets(path: Path) -> Tuple[bytes, bytes]:
    streams = _ffprobe(["-show_streams", "-show_data", str(path)]).get(
        "streams", [])
    if not streams or "extradata" not in streams[0]:
        raise VideoPipelineError("segment carries no extradata")
    return parse_avcc_hexdump(streams[0]["extradata"])


def _write_atomic(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


class SegmentWriter:
    """Encodes one camera into MP4 segments plus a live Annex-B stream.

    ffmpeg runs for the whole episode: raw RGB goes in on stdin,
    segments of ``frames_per_segment`` frames land in
    ``directory/video/<camera>/``, and access units come back on stdout,
    where a daemon reader thread queues them. Callers see units only
    through ``feed``, and failures only as VideoPipelineError from
    ``feed`` or ``close`` — never from the reader thread, which a caller
    cannot catch.

    ``index.json`` is rewritten every time a segment closes, so a live
    dashboard can seek into everything already on disk while the run
    continues. ffmpeg's stderr goes to ``ffmpeg.log`` beside the
    segments: draining it in-process would cost another thread, and
    discarding it would make a hardware failure undiagnosable.

    A writer whose ``feed`` raised must still be ``close``d (swallow the
    second error): the reader thread holds a reference to the writer and
    ffmpeg blocks on its pipes, so a dropped-without-close writer leaks
    one process and one thread for the life of the program.
    """

    def __init__(self, directory, camera: str, width: int, height: int,
                 fps: int = _DEFAULT_FPS,
                 frames_per_segment: int = _FRAMES_PER_SEGMENT,
                 encoder_args: Optional[List[str]] = None) -> None:
        self.directory = Path(directory)
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frames_per_segment = int(frames_per_segment)
        self.video_dir = self.directory / "video" / camera
        try:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoPipelineError(
                "could not create {}: {}".format(self.video_dir, exc)
            ) from exc
        self._frame_bytes = self.width * self.height * 3
        self._steps: List[int] = []             # every fed step, in order
        self._pending: Deque[int] = deque()     # fed, awaiting their unit
        self._segments: List[dict] = []
        self._queue: "Queue[bytes]" = Queue()
        self._error: Optional[BaseException] = None
        self._codec: Optional[str] = None
        self._sps = b""
        self._pps = b""
        self._closed = False
        # segment_time and frames_per_segment must agree: raw stdin with
        # -r gives synthetic timestamps, so this identity is what makes
        # segment boundaries land on exact frame counts.
        command = _ffmpeg_command(self.video_dir, self.width, self.height,
                                  self.fps,
                                  self.frames_per_segment / self.fps,
                                  encoder_args or NVENC_ARGS, "warning")
        try:
            self._log = open(self.video_dir / "ffmpeg.log", "wb")
        except OSError as exc:
            raise VideoPipelineError(
                "could not open ffmpeg.log: {}".format(exc)) from exc
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log)
        except OSError as exc:
            self._log.close()
            raise VideoPipelineError(
                "could not start ffmpeg: {}".format(exc)) from exc
        self._stdin = cast(BufferedWriter, self._process.stdin)
        self._reader = threading.Thread(
            target=self._read_units, name="specola-video-" + camera,
            daemon=True)
        self._reader.start()

    def feed(self, frame: np.ndarray, step: int) -> List[Tuple[int, bytes]]:
        """Encode one frame; return the access units now ready for the
        wire, each paired with the step it was fed for.

        Units lag their frames by a few steps — the encoder's pipeline
        depth plus the unit boundary the reader waits for — so a call
        usually returns the unit of an earlier step, or nothing. The
        encoder emits in order, so pairing is first in, first out.
        """
        self._check_reader()
        payload = frame.tobytes()
        if len(payload) != self._frame_bytes:
            raise VideoPipelineError(
                "frame is {} bytes, expected {} for {}x{} RGB".format(
                    len(payload), self._frame_bytes, self.width, self.height))
        try:
            self._stdin.write(payload)
            self._stdin.flush()
        except (OSError, ValueError) as exc:
            # ValueError: write to a closed file — a caller bug (feed
            # after close), reported on the documented exception.
            raise VideoPipelineError(
                "ffmpeg stopped accepting frames ({}); see {}".format(
                    exc, self.video_dir / "ffmpeg.log")) from exc
        self._steps.append(step)
        self._pending.append(step)
        self._index_closed_segments()
        return self._drain_units()

    def close(self) -> None:
        """Flush the encoder, index the segments still unindexed and
        write the final index. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=_CLOSE_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._reader.join(timeout=_CLOSE_TIMEOUT)
        self._drain_units()
        self._log.close()
        # every remaining segment is closed now that ffmpeg has exited
        while (self._segment_path(len(self._segments)).exists()
               and self._index_segment(len(self._segments))):
            pass
        self._write_index()
        self._check_reader()
        if self._process.returncode:
            raise VideoPipelineError(
                "ffmpeg exited with {}; see {}".format(
                    self._process.returncode,
                    self.video_dir / "ffmpeg.log"))

    def _read_units(self) -> None:
        """Reader thread: never raises, records the failure instead."""
        stream = cast(BufferedReader, self._process.stdout)
        buffer = b""
        try:
            while True:
                chunk = stream.read1(_READ_SIZE)
                if not chunk:
                    break
                units, buffer = split_annexb(buffer + chunk)
                for unit in units:
                    self._queue.put(unit)
            if buffer:
                self._queue.put(buffer)     # last unit, it has no successor
        except BaseException as exc:        # noqa: BLE001 - see docstring
            self._error = exc

    def _check_reader(self) -> None:
        if self._error is not None:
            raise VideoPipelineError(
                "video reader thread failed: {}".format(self._error))

    def _drain_units(self) -> List[Tuple[int, bytes]]:
        paired = []
        while self._pending:
            try:
                unit = self._queue.get_nowait()
            except Empty:
                break
            paired.append((self._pending.popleft(), unit))
        return paired

    def _segment_path(self, number: int) -> Path:
        return self.video_dir / _SEGMENT_NAME.format(number)

    def _index_closed_segments(self) -> None:
        """ffmpeg opens the next segment only after closing the current
        one, so the appearance of segment N+1 is the signal that segment
        N is complete and can be probed."""
        while (self._segment_path(len(self._segments) + 1).exists()
               and self._index_segment(len(self._segments))):
            self._write_index()

    def _index_segment(self, number: int) -> bool:
        """Probe one closed segment into the index. False when it holds
        no samples (ffmpeg opens a segment file before it has anything
        to put in it), which also stops the indexing loops."""
        path = self._segment_path(number)
        packets = _segment_packets(path)
        if not packets:
            return False
        if self._codec is None:
            self._sps, self._pps = _segment_parameter_sets(path)
            self._codec = codec_string(self._sps)
        self._segments.append({"file": path.name, "packets": packets})
        return True

    def _write_index(self) -> None:
        _write_atomic(self.video_dir / _INDEX_FILE, json.dumps({
            "format_version": _INDEX_VERSION,
            "codec": self._codec,
            "sps": self._sps.hex(),
            "pps": self._pps.hex(),
            "frames_per_segment": self.frames_per_segment,
            "steps": self._steps,
            "segments": self._segments,
        }).encode("utf-8"))
