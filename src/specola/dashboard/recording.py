"""On-disk recording format: the durable buffer for the web dashboard.

One directory per episode: manifest.json, snapshots.jsonl (gzipped at
finalize), the camera frames, and — for adapters with a 3D scene —
poses.bin plus geometry.json/meshes.bin. Frames live in one of two
stores, named by the manifest's ``frame_store``: JPEG files under
frames/<camera>/<step:08d>.jpg, or h264 segments under video/<camera>/
written by dashboard.video. The recording is the only durable state
store; every server-side index is a rebuildable cache. Format spec:
docs/superpowers/specs/2026-07-30-web-dashboard-design.md.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import os
import json
import math
import shutil
import struct
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from specola.dashboard.video import VideoPipelineError, to_annexb
from specola.extraction_observation import (
    ExtractionObservation, ExtractionObservationError, extraction_document,
)


def _num(value):
    """JSON-safe float: non-finite values become null."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _binding_to_json(binding) -> dict:
    worst = None
    if binding.worst_robustness is not None:
        worst = {"rho": _num(binding.worst_robustness),
                 "step": binding.worst_step,
                 "time": _num(binding.worst_time)}
    final = None
    if binding.final:
        final = {"rho": _num(binding.final_robustness),
                 "step": binding.final_step,
                 "time": _num(binding.final_time)}
    first_violation = None
    if binding.first_violation_step is not None:
        first_violation = {"step": binding.first_violation_step,
                           "time": _num(binding.first_violation_time)}
    return {"key": binding.binding_key, "state": binding.state.value,
            "active": binding.active,
            "rho": _num(binding.current_robustness),
            "worst": worst, "final": final,
            "first_violation": first_violation,
            "error": binding.error_message}


def _binding_meta(spec_name: str, binding) -> dict:
    return {"spec": spec_name, "key": binding.binding_key,
            "label": binding.label,
            "authored_formula": binding.authored_formula,
            "resolved_formula": binding.resolved_formula,
            "backend": binding.backend,
            "window_steps": binding.window_steps,
            "unit": binding.unit,
            "entities": [{"id": entity.entity_id,
                          "class": entity.entity_class,
                          "label": entity.label, "role": entity.role}
                         for entity in binding.entities]}


def snapshot_to_line(snapshot, wall_time: float, seen: set, *, extraction: Optional[ExtractionObservation] = None) -> dict:
    """Serialize one RunSnapshot; ``seen`` carries spec names and
    (spec, key) pairs across calls so metadata is emitted once on first
    appearance."""
    encoded_extraction = None
    if extraction is not None:
        if type(extraction) is not ExtractionObservation or extraction.verification_only:
            raise ExtractionObservationError("ordinary recordings cannot contain verification-only observations")
        if extraction.time != snapshot.simulator_time:
            raise ExtractionObservationError("recording and extraction simulator times differ")
        encoded_extraction = extraction_document(extraction)
    new_specs, new_bindings, specs = [], [], []
    for spec in snapshot.specifications:
        if spec.name not in seen:
            seen.add(spec.name)
            new_specs.append({"name": spec.name,
                              "authored_formula": spec.authored_formula})
        bindings = []
        for binding in spec.bindings:
            if (spec.name, binding.binding_key) not in seen:
                seen.add((spec.name, binding.binding_key))
                new_bindings.append(_binding_meta(spec.name, binding))
            bindings.append(_binding_to_json(binding))
        specs.append({"name": spec.name, "state": spec.state.value,
                      "skip_reason": spec.skip_reason,
                      "bindings": bindings})
    line = {"step": snapshot.step,
            "sim_time": _num(snapshot.simulator_time),
            "wall_time": _num(wall_time),
            "specs": specs,
            "transitions": [
                {"spec": spec_name, "key": key, "step": transition.step,
                 "time": _num(transition.simulator_time),
                 "from": (transition.from_state.value
                          if transition.from_state else None),
                 "to": transition.to_state.value}
                for (spec_name, key, transition) in snapshot.transitions]}
    if new_specs:
        line["new_specs"] = new_specs
    if encoded_extraction is not None:
        line["extraction"] = encoded_extraction
    if new_bindings:
        line["new_bindings"] = new_bindings
    return line


def frame_path(directory: Path, camera: str, step: int) -> Path:
    return Path(directory) / "frames" / camera / "{:08d}.jpg".format(step)


def index_at_or_before(steps: Sequence[int], step: int) -> Optional[int]:
    """Index of the last entry of ascending ``steps`` at or before
    ``step``; None when ``step`` precedes them all."""
    index = bisect.bisect_right(steps, step) - 1
    return index if index >= 0 else None


# Frame stores. JPEG is the default and stays format_version 1 — the
# manifest key is additive, so a reader that ignores it still reads the
# recording. The video store is a real format change (frames are no
# longer files) and carries format_version 2.
DISCOVERY_FILE = "discovery.json.gz"
FRAME_STORE_JPEG = "jpeg"
FRAME_STORE_VIDEO = "video_segments"
_VIDEO_DIR = "video"
_VIDEO_INDEX = "index.json"


def video_index_path(directory: Path, camera: str) -> Path:
    return Path(directory) / _VIDEO_DIR / camera / _VIDEO_INDEX


def _read_video_index(path: Path) -> Optional[dict]:
    """Parse one camera's index.json, or None when it is absent or
    unusable — a broken index must cost that camera its frames and
    nothing else.

    ``indexed`` counts the frames whose bytes are readable: the writer
    records a step as soon as it is fed but can only index a segment
    once ffmpeg closes it, so the step list runs ahead of the segments.
    """
    try:
        raw = json.loads(path.read_bytes())
        if int(raw.get("format_version", 0)) != 1:
            return None
        index = {"codec": str(raw["codec"]),
                 "sps": bytes.fromhex(raw["sps"]),
                 "pps": bytes.fromhex(raw["pps"]),
                 "per_segment": int(raw["frames_per_segment"]),
                 "steps": [int(step) for step in raw["steps"]],
                 "segments": list(raw["segments"])}
        index["indexed"] = sum(len(segment["packets"])
                               for segment in index["segments"])
    except (OSError, TypeError, ValueError, KeyError):
        return None
    if index["per_segment"] < 1 or not index["sps"] or not index["pps"]:
        return None
    return index


class VideoFrameReader:
    """Random access into one camera's h264 segments.

    Frames are samples inside fixed-length MP4 segments; index.json maps
    each sampled step to a byte range and carries the parameter sets, so
    a seek is one file read plus the Annex-B conversion — the same shape
    the JPEG store has, and the same nearest-earlier semantics.

    The index is re-read whenever it changes on disk: the writer
    rewrites it as each segment closes, so a reader opened mid-run must
    not cache it. Segment files are opened per call like PosesReader,
    for the same reason.

    Use ``open()`` rather than the constructor.
    """

    def __init__(self, directory: Path, camera: str, index: dict) -> None:
        self.directory = Path(directory)
        self.camera = camera
        self._path = video_index_path(self.directory, camera)
        self._index = index
        self._stamp = self._mtime()

    @classmethod
    def open(cls, directory: Path, camera: str
             ) -> Optional["VideoFrameReader"]:
        index = _read_video_index(video_index_path(directory, camera))
        return None if index is None else cls(directory, camera, index)

    @property
    def codec(self) -> str:
        """WebCodecs codec string of this camera's stream."""
        return self._current()["codec"]

    def steps(self) -> List[int]:
        """Sampled steps, ascending."""
        return list(self._current()["steps"])

    def annexb(self, step: int) -> Optional[Tuple[bytes, int]]:
        """Nearest-earlier sampled frame as a self-contained access
        unit, paired with the step it was sampled at.

        None before the first sampled step. A step inside the segment
        ffmpeg still has open resolves to the last frame of the last
        closed segment: those steps are indexed but their bytes are not
        readable yet, and the closest earlier truth beats nothing.
        """
        index = self._current()
        ordinal = index_at_or_before(index["steps"], step)
        if ordinal is None:
            return None
        ordinal = min(ordinal, index["indexed"] - 1)
        if ordinal < 0:
            return None
        try:
            number, packet = divmod(ordinal, index["per_segment"])
            segment = index["segments"][number]
            position, size = segment["packets"][packet]
            with open(self.directory / _VIDEO_DIR / self.camera
                      / segment["file"], "rb") as fh:
                fh.seek(position)
                sample = fh.read(size)
            if len(sample) != size:
                return None
            unit = to_annexb(sample, index["sps"], index["pps"])
        except (OSError, IndexError, TypeError, ValueError,
                VideoPipelineError):
            return None
        return unit, index["steps"][ordinal]

    def _mtime(self) -> Optional[int]:
        try:
            return self._path.stat().st_mtime_ns
        except OSError:
            return None

    def _current(self) -> dict:
        """The index as it is on disk now; the last good one stands when
        a re-read fails (a rewrite in flight, a truncated file)."""
        stamp = self._mtime()
        if stamp != self._stamp:
            index = _read_video_index(self._path)
            if index is not None:
                self._index = index
            self._stamp = stamp
        return self._index


def _ordinary_discovery_bytes(payload: bytes) -> bytes:
    """Detach and validate a sidecar before it reaches a writer or queue."""
    from ..conformance.verification import require_ordinary_execution
    from ..discovery_restore import discovery_model_from_document

    require_ordinary_execution()
    detached = bytes(payload)
    discovery_model_from_document(json.loads(gzip.decompress(detached)))
    return detached


class RecordingWriter:
    """Synchronous file IO only; threading lives in the recorder sink."""

    def __init__(self, directory: Path, manifest: dict) -> None:
        from ..conformance.verification import require_ordinary_execution
        require_ordinary_execution(manifest)
        self.directory = Path(directory)
        (self.directory / "frames").mkdir(parents=True, exist_ok=True)
        self.manifest = dict(manifest)
        self.manifest.setdefault("frame_store", FRAME_STORE_JPEG)
        self.rewrite_manifest()
        self._camera_dirs: set = set()
        self._fh = open(self.directory / "snapshots.jsonl", "a",
                        encoding="utf-8")

    def set_frame_store(self, store: str) -> None:
        """Record which store holds this recording's frames, with the
        format version that store implies, and republish the manifest.

        The recorder calls this once, when it learns whether the video
        pipeline came up, which may be several steps into the episode.
        """
        self.manifest["frame_store"] = store
        self.manifest["format_version"] = (
            2 if store == FRAME_STORE_VIDEO else 1)
        self.rewrite_manifest()

    def rewrite_manifest(self) -> None:
        tmp = self.directory / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2))
        tmp.replace(self.directory / "manifest.json")

    def write_discovery(self, payload: bytes) -> dict:
        """Publish the episode's discovery sidecar, gzipped.

        Written once, before the first snapshot line, and never touched
        again: it describes the reset the episode compiled against, and
        RoboCasa randomizes its kitchen per reset, so a document written
        later from a fresh look at the adapter would describe a
        different scene than the one being monitored.

        Measured on RoboCasa PickPlaceCounterToCabinet: the full raw
        document is 55.13 MiB of JSON and 1.66 MiB gzipped, against the
        ~14 MB geometry.json this recording already carries. tmp+replace
        with an fsync, matching the finalize path, so a reader never
        sees a partial file.

        Returns the manifest entry describing what was written; the
        digest lets a consumer verify the bytes it decompresses are the
        bytes this episode produced.
        """
        return self._write_discovery_bytes(_ordinary_discovery_bytes(payload))

    def _write_discovery_bytes(self, payload: bytes) -> dict:
        """Write bytes checked by this writer or the recorder queue item.

        The queue validates on its submitting thread so discovery model
        reconstruction cannot delay frame and snapshot writes on the worker.
        This internal path retains those exact immutable bytes.
        """
        from ..conformance.verification import require_ordinary_execution
        require_ordinary_execution()
        tmp = self.directory / (DISCOVERY_FILE + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.directory / DISCOVERY_FILE)
        entry = {
            "file": DISCOVERY_FILE,
            "schema_version": 1,
            "encoding": "gzip+json",
            "compressed_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.manifest["discovery"] = entry
        self.rewrite_manifest()
        return entry

    def append_step(self, line: dict) -> None:
        self._fh.write(json.dumps(line, allow_nan=False) + "\n")
        self._fh.flush()

    def write_frame(self, camera: str, step: int, jpeg: bytes) -> None:
        path = frame_path(self.directory, camera, step)
        if camera not in self._camera_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._camera_dirs.add(camera)
        path.write_bytes(jpeg)

    def finalize(self, status: str) -> None:
        if self._fh.closed:
            return
        self._fh.close()
        self.manifest["status"] = status
        self.rewrite_manifest()
        source = self.directory / "snapshots.jsonl"
        # tmp+replace: a reader must never see a half-written .gz
        tmp = self.directory / "snapshots.jsonl.gz.tmp"
        with open(source, "rb") as f_in:
            with gzip.open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        tmp.replace(str(source) + ".gz")
        source.unlink()


def discovery_path(directory) -> Path:
    return Path(directory) / DISCOVERY_FILE


def read_discovery_bytes(directory) -> Optional[bytes]:
    """The episode's discovery sidecar as written, or None.

    Absent for every recording made before the sidecar existed, and for
    any run whose discovery capture failed — both of which must stay
    replayable. The caller decides what to disable; this only reports
    whether the bytes are there.
    """
    try:
        return discovery_path(directory).read_bytes()
    except OSError:
        return None


def read_discovery_document(directory) -> Optional[dict]:
    """The sidecar decoded, or None when it is absent or unreadable.

    A corrupt sidecar reads as an absent one on purpose: the recording's
    monitor data is independent of it, and refusing to replay a run
    because its Discover tab cannot load would trade a working
    capability for a broken one.
    """
    payload = read_discovery_bytes(directory)
    if payload is None:
        return None
    try:
        return json.loads(gzip.decompress(payload))
    except (OSError, ValueError, EOFError):
        return None


class RecordingReader:
    """O(1) step access over a live (growing) or finalized recording.

    Live file: byte-offset index, point reads by seek; ``refresh()``
    catches up on appended lines. Finalized (.gz): decompressed into
    memory at open, list index. Trailing partial lines (hard kill) are
    ignored until completed.

    The frame index is scanned once at construction and is NOT refreshed:
    during a live run the server tracks newly sampled frames itself (its
    RunIndex is fed by the recorder callback); ``frame()`` is for
    finalized recordings and for readers opened after the frames exist.
    A reader also does not survive finalize (the JSONL it indexes is
    gzipped and removed) — the server swaps in a fresh reader at episode
    end. Video-store cameras are discovered at construction the same
    way, though each one's own index is re-read as segments close, so a
    camera already found keeps growing under a live reader.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text())
        self._gz_lines: Optional[List[dict]] = None
        self._offsets: List[int] = []
        self._steps: List[int] = []
        self._tail = 0
        self._refresh_lock = threading.Lock()
        gz = self.directory / "snapshots.jsonl.gz"
        if gz.exists():
            self._gz_lines = []
            with gzip.open(gz, "rt", encoding="utf-8") as fh:
                for raw in fh:
                    if raw.endswith("\n"):
                        self._gz_lines.append(json.loads(raw))
            self._steps = [line["step"] for line in self._gz_lines]
        else:
            self.refresh()
        self._frame_index = {}
        frames = self.directory / "frames"
        if frames.is_dir():
            for cam_dir in sorted(frames.iterdir()):
                if cam_dir.is_dir():
                    steps = sorted(int(p.stem) for p in cam_dir.glob("*.jpg"))
                    if steps:
                        self._frame_index[cam_dir.name] = steps
        self._video = {}
        video = self.directory / _VIDEO_DIR
        if video.is_dir():
            for cam_dir in sorted(video.iterdir()):
                reader = VideoFrameReader.open(self.directory, cam_dir.name)
                if reader is not None:
                    self._video[cam_dir.name] = reader

    def reload_manifest(self) -> None:
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text())

    def refresh(self) -> int:
        """Index lines appended since the last call (live mode only).

        Thread-safe: concurrent seeks from several dashboard clients may
        trigger refresh at once.
        """
        if self._gz_lines is not None:
            return 0
        path = self.directory / "snapshots.jsonl"
        if not path.exists():
            return 0
        added = 0
        with self._refresh_lock:
            with open(path, "rb") as fh:
                fh.seek(self._tail)
                while True:
                    offset = fh.tell()
                    raw = fh.readline()
                    if not raw.endswith(b"\n"):
                        break                   # partial trailing line
                    self._offsets.append(offset)
                    self._steps.append(json.loads(raw)["step"])
                    self._tail = fh.tell()
                    added += 1
        return added

    def num_steps(self) -> int:
        return len(self._steps)

    def read_step(self, step: int) -> dict:
        index = bisect.bisect_left(self._steps, step)
        if index >= len(self._steps) or self._steps[index] != step:
            raise KeyError(step)
        if self._gz_lines is not None:
            return self._gz_lines[index]
        with open(self.directory / "snapshots.jsonl", "rb") as fh:
            fh.seek(self._offsets[index])
            return json.loads(fh.readline())

    def nearest_step(self, step: int) -> Optional[int]:
        """Nearest recorded step at or before ``step``.

        ``None`` before the first recorded step. Mirrors the
        nearest-earlier semantics frames already have: a live seek can
        race the writer (the sought line still in the queue), and
        serving the closest earlier truth beats serving nothing.
        """
        index = index_at_or_before(self._steps, step)
        return self._steps[index] if index is not None else None

    def all_lines(self) -> Iterator[dict]:
        """Every complete line in append order (startup indexing path)."""
        if self._gz_lines is not None:
            yield from self._gz_lines
            return
        path = self.directory / "snapshots.jsonl"
        if not path.exists():
            return
        with open(path, "rb") as fh:
            for raw in fh:
                if raw.endswith(b"\n"):
                    yield json.loads(raw)

    def cameras(self) -> List[str]:
        return sorted(set(self._frame_index) | set(self._video))

    def frame_steps(self, camera: str) -> List[int]:
        """Sampled steps for ``camera``, ascending, across both frame
        stores (frame-index caveat in the class docstring applies)."""
        steps = set(self._frame_index.get(camera, ()))
        video = self._video.get(camera)
        if video is not None:
            steps.update(video.steps())
        return sorted(steps)

    def frame(self, camera: str, step: int) -> Optional[Tuple[bytes, int]]:
        """Nearest-earlier sampled JPEG frame, or None before the first
        one. JPEG store only; ``frame_entry`` covers both."""
        steps = self._frame_index.get(camera)
        if not steps:
            return None
        index = index_at_or_before(steps, step)
        if index is None:
            return None
        actual = steps[index]
        return frame_path(self.directory, camera, actual).read_bytes(), actual

    def frame_entry(self, camera: str,
                    step: int) -> Optional[Tuple[bytes, str, int]]:
        """Nearest-earlier frame from whichever store holds this camera:
        its bytes, the codec they are in (``"jpeg"`` or an ``avc1.``
        string) and the step they were sampled at."""
        video = self._video.get(camera)
        if video is not None:
            found = video.annexb(step)
            return None if found is None else (found[0], video.codec,
                                               found[1])
        found = self.frame(camera, step)
        return None if found is None else (found[0], "jpeg", found[1])


# poses.bin: 16-byte header, then fixed-size records of nbody*7
# little-endian float32 (the nbody xpos rows, then the nbody xquat rows).
# Fixed records make step lookup a seek; the header carries nbody so a
# reader needs nothing else.
_POSES_FILE = "poses.bin"
_POSES_MAGIC = b"SPPO"
_POSES_VERSION = 1
_POSES_HEADER = struct.Struct("<4sIII")     # magic, version, nbody, reserved


class PosesWriter:
    """Appends body poses to the pose sidecar, one record per data line.

    Records carry no step number: they are joined to snapshots.jsonl by
    ordinal, so record *i* belongs to the *i*-th data line. Flushed per
    append, matching snapshots.jsonl — a killed run stays readable up to
    the last complete record.
    """

    def __init__(self, directory: Path, nbody: int) -> None:
        self.directory = Path(directory)
        self.nbody = int(nbody)
        self._fh = open(self.directory / _POSES_FILE, "wb")
        self._fh.write(_POSES_HEADER.pack(
            _POSES_MAGIC, _POSES_VERSION, self.nbody, 0))
        self._fh.flush()

    def append(self, xpos, xquat) -> None:
        """Append one record from (nbody, 3) and (nbody, 4) float32
        arrays."""
        self._fh.write(xpos.astype("<f4", copy=False).tobytes())
        self._fh.write(xquat.astype("<f4", copy=False).tobytes())
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class PosesReader:
    """Random access into the pose sidecar of a live or finished episode.

    ``count`` and ``record`` re-stat the file on every call: in live mode
    the writer keeps appending under an already-open reader, and a cached
    size would freeze the panel at the length it had when the first
    client connected. A partially written trailing record is invisible
    until complete.

    Use ``open()`` rather than the constructor — it validates the header
    and returns None for an absent or unreadable sidecar, since a broken
    sidecar must cost the 3D panel and nothing else.
    """

    def __init__(self, directory: Path, nbody: int) -> None:
        self.directory = Path(directory)
        self.nbody = int(nbody)
        self.record_size = self.nbody * 7 * 4
        self._path = self.directory / _POSES_FILE

    @classmethod
    def open(cls, directory: Path) -> Optional["PosesReader"]:
        try:
            with open(Path(directory) / _POSES_FILE, "rb") as fh:
                header = fh.read(_POSES_HEADER.size)
        except OSError:
            return None
        if len(header) != _POSES_HEADER.size:
            return None
        magic, version, nbody, _reserved = _POSES_HEADER.unpack(header)
        if magic != _POSES_MAGIC or version != _POSES_VERSION or nbody < 1:
            return None
        return cls(directory, nbody)

    @property
    def count(self) -> int:
        """Complete records currently on disk."""
        try:
            size = self._path.stat().st_size
        except OSError:
            return 0
        return max(size - _POSES_HEADER.size, 0) // self.record_size

    def record(self, ordinal: int) -> Optional[bytes]:
        """Raw ``nbody * 28`` bytes of one record; None past the tail."""
        if ordinal < 0 or ordinal >= self.count:
            return None
        with open(self._path, "rb") as fh:
            fh.seek(_POSES_HEADER.size + ordinal * self.record_size)
            data = fh.read(self.record_size)
        return data if len(data) == self.record_size else None


def write_geometry(directory: Path, geometry: dict) -> None:
    """Write the static scene description: geometry.json + meshes.bin.

    Both files land atomically, and meshes.bin lands first, so a reader
    that sees geometry.json always finds the buffer it indexes.
    """
    directory = Path(directory)
    _write_atomic(directory / "meshes.bin", geometry["meshes"])
    _write_atomic(directory / "geometry.json",
                  json.dumps(geometry["json"]).encode("utf-8"))


def read_geometry(directory: Path) -> Optional[dict]:
    """``{"json": dict, "meshes": bytes}``, or None when absent or
    unreadable."""
    directory = Path(directory)
    try:
        meta = json.loads((directory / "geometry.json").read_bytes())
        meshes = (directory / "meshes.bin").read_bytes()
    except (OSError, ValueError):
        return None
    return {"json": meta, "meshes": meshes}


def _write_atomic(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
