"""Recorder sink: attaches to _snapshot_stream, one sink among several.

Simulator thread: capture + serialize + enqueue only. Writer thread:
frame encode + all disk writes. Never raises into the run; the first
internal error prints one warning naming the cause and disables the
sink. Verification-only input is a contract violation and raises before
capture or enqueue. Frames drop before data under backpressure; drops are counted
and surfaced downstream.

Frames go to one of two stores, chosen once on the writer thread when
the first frames arrive: JPEG files per step, or hardware-encoded h264
video segments (dashboard/video.py). A video failure is not a sink
failure — it costs the frames and nothing else, so the data lines the
monitoring depends on keep flowing.
"""

from __future__ import annotations

import io
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from specola.extraction_observation import ExtractionObservationError

from .recording import (
    FRAME_STORE_JPEG, FRAME_STORE_VIDEO, PosesWriter, RecordingWriter,
    snapshot_to_line, write_geometry, _ordinary_discovery_bytes)
from .video import (
    NVENC_ARGS, SegmentWriter, VideoPipelineError, codec_string,
    probe_encoder)

_SENTINEL = object()


@dataclass(frozen=True, init=False)
class _DiscoveryItem:
    """An ordinary discovery document checked before entering the FIFO."""

    payload: bytes

    def __init__(self, payload: Union[bytes, bytearray, memoryview]) -> None:
        object.__setattr__(self, "payload", _ordinary_discovery_bytes(bytes(payload)))


def warn(message: str) -> None:
    print("  Web warning: {}".format(message), file=sys.stderr)


def note(message: str) -> None:
    print("  Web: {}".format(message), file=sys.stderr)


def _codec_from_unit(unit: bytes) -> Optional[str]:
    """WebCodecs codec string from the SPS an access unit carries.

    The segment writer only learns the codec once ffmpeg closes the
    first segment, dozens of frames in, but the wire needs it on the
    first unit — and every unit this pipeline emits leads with its
    parameter sets, so the unit itself is the earlier source.
    """
    index = 0
    while True:
        found = unit.find(b"\x00\x00\x01", index)
        if found < 0:
            return None
        index = found + 3
        if index + 4 <= len(unit) and (unit[index] & 0x1f) == 7:
            return codec_string(unit[index:index + 4])


class WebRecorderSink:
    def __init__(
        self,
        run_directory,
        *,
        run_id: str,
        simulator: str,
        task: str,
        mode: str,
        cameras: Optional[Tuple[str, ...]],
        size: Tuple[int, int] = (640, 480),
        stride: int = 1,
        video: str = "auto",
        on_line: Optional[Callable] = None,
        on_episode_end: Optional[Callable] = None,
        queue_size: int = 64,
    ) -> None:
        self.recording_dir = Path(run_directory) / "recording_000"
        self._writer = RecordingWriter(self.recording_dir, {
            "format_version": 1,
            "run_id": run_id,
            "simulator": simulator,
            "task": task,
            "mode": mode,
            "cameras": [],
            "capture_size": list(size),
            "stride": int(stride),
            "timebase": "step+sim_time",
            "created_wall_time": time.time(),
            # Identity, so a listing can say where a run is and offer a
            # way back to it. The pid corroborates a same-host state and
            # is never the sole basis for one.
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "dashboard_url": None,
            "status": None,
        })
        self._cameras_arg = cameras
        # [(recorded name, capture_frame argument)]
        self._resolved: Optional[List[Tuple[str, Optional[str]]]] = None
        self._size = size
        self._stride = int(stride)
        self._on_line = on_line
        self._on_episode_end = on_episode_end
        self._on_discovery = None
        self._video = video
        # Both resolved on the writer thread when the first frames
        # arrive: the store for this recording, and one segment writer
        # per camera while the video store is alive.
        self._store: Optional[str] = None
        self._writers: Dict[str, SegmentWriter] = {}
        self._seen: set = set()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._encoder = None            # resolved lazily on writer thread
        # None until the scene capability is probed on the first step;
        # False for adapters without a 3D scene, which are never asked
        # again.
        self._scene: Optional[bool] = None
        self._poses = None              # PosesWriter, lazily on writer thread
        self.frames_dropped = 0
        self.active = True
        self._finalized = False
        # Guards the check-then-set on active/_finalized: on_step runs on
        # the simulator thread, _drain on the writer thread, and a future
        # stop path may call close() from a server thread.
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain, name="specola-web-writer", daemon=True)
        self._thread.start()

    def write_discovery(self, payload: bytes) -> None:
        """Queue the discovery sidecar ahead of the first snapshot line.

        Through the same FIFO the lines use, so ordering is a property
        of the queue rather than of thread timing: a client that reads a
        data line can rely on the scene description already being on
        disk. The submitting thread validates the document and hands over
        immutable bytes. The writer thread writes those bytes without
        rebuilding the discovery model. Invalid discovery data raises before
        queueing, including candidate-only data nested in the extraction.
        """
        if not self.active:
            return
        try:
            self._queue.put(_DiscoveryItem(payload), timeout=2.0)
        except queue.Full:
            warn("discovery sidecar dropped: writer queue stalled; "
                  "monitoring and recording continue.")

    def set_dashboard_url(self, url: str) -> None:
        """Record where this run is being served, once it is known."""
        self._writer.manifest["dashboard_url"] = url
        self._writer.rewrite_manifest()

    def set_callbacks(self, on_line, on_episode_end,
                      on_discovery=None) -> None:
        """Late wiring: the server is constructed after the sink."""
        self._on_line = on_line
        self._on_episode_end = on_episode_end
        if on_discovery is not None:
            self._on_discovery = on_discovery

    # -- simulator thread -------------------------------------------------

    def on_step(self, adapter, snapshot) -> None:
        if not self.active:
            return
        observation = getattr(adapter, "last_observation", None)
        extraction = observation.extraction if observation is not None else None
        if extraction is not None and extraction.verification_only:
            raise ExtractionObservationError("verification-only observations cannot enter an ordinary recorder")
        try:
            line = snapshot_to_line(snapshot, time.time(), self._seen, extraction=extraction)
            if self._resolved is None:
                self._resolved = self._resolve_cameras(adapter)
            geometry = None
            if self._scene is None:
                probe = getattr(adapter, "scene_geometry", None)
                geometry = probe() if probe is not None else None
                self._scene = geometry is not None
            # Two float32 copies per step; everything else about the
            # poses happens on the writer thread.
            poses = adapter.scene_poses() if self._scene else None
            frames = []
            if self._stride <= 1 or snapshot.step % self._stride == 0:
                width, height = self._size
                for name, argument in self._resolved:
                    array = adapter.capture_frame(
                        camera=argument, width=width, height=height)
                    if array is not None:
                        frames.append((name, array))
            self._enqueue(line, frames, poses, geometry)
        except Exception as exc:
            self._disable(exc)

    def _resolve_cameras(
        self, adapter
    ) -> List[Tuple[str, Optional[str]]]:
        names = tuple(adapter.camera_names())
        if self._cameras_arg is None:
            # Adapter default; record it under its real name when known.
            resolved: List[Tuple[str, Optional[str]]] = [
                (names[0] if names else "default", None)]
        elif self._cameras_arg == ("all",):
            resolved = ([(name, name) for name in names]
                        or [("default", None)])
        else:
            resolved = [(name, name) for name in self._cameras_arg]
        self._writer.manifest["cameras"] = [name for name, _ in resolved]
        self._writer.rewrite_manifest()
        return resolved

    def _enqueue(self, line: dict, frames: list, poses, geometry) -> None:
        try:
            self._queue.put_nowait((line, frames, poses, geometry))
        except queue.Full:
            self.frames_dropped += len(frames)
            try:
                # Data outranks frames; a stall past this timeout is a
                # disk-level failure and hits the disable path. Poses are
                # data-sized and stay with their line: only frames drop.
                self._queue.put((line, [], poses, geometry), timeout=2.0)
            except queue.Full:
                raise RuntimeError("writer queue stalled >2s (disk?)")

    def _disable(self, exc: BaseException) -> None:
        with self._state_lock:
            if not self.active:
                return
            self.active = False
        warn("{}: {}; visualization disabled, monitoring "
              "continues.".format(type(exc).__name__, exc))

    # -- writer thread ----------------------------------------------------

    def _encode(self, array) -> Optional[bytes]:
        """JPEG-encode one HxWx3 uint8 RGB frame; cv2, then PIL."""
        if self._encoder is None:
            try:
                import cv2  # type: ignore[import-not-found]

                def encode_cv2(frame):
                    ok, buffer = cv2.imencode(".jpg", frame[:, :, ::-1])
                    return buffer.tobytes() if ok else None

                self._encoder = encode_cv2
            except ImportError:
                try:
                    from PIL import Image  # type: ignore[import-not-found]

                    def encode_pil(frame):
                        buffer = io.BytesIO()
                        Image.fromarray(frame).save(
                            buffer, "JPEG", quality=85)
                        return buffer.getvalue()

                    self._encoder = encode_pil
                    warn("cv2 unavailable, using PIL JPEG encoder "
                          "(slower).")
                except ImportError:
                    def encode_none(frame):
                        return None

                    self._encoder = encode_none
                    warn("no JPEG encoder (cv2/PIL missing); frames "
                          "not recorded, data still is.")
        return self._encoder(array)

    def _select_store(self, frames: list) -> None:
        """Choose the frame store, once, from the first frames to reach
        this thread, and record the choice in the manifest before the
        line that used it goes out."""
        store = FRAME_STORE_JPEG
        if self._video != FRAME_STORE_JPEG and self._start_video(frames):
            store = FRAME_STORE_VIDEO
        self._store = store
        self._writer.set_frame_store(store)

    def _start_video(self, frames: list) -> bool:
        """Bring up one segment writer per camera. False falls the
        recording back to JPEG: a run must never end because the video
        pipeline could not start."""
        # Probe under "nvenc" too, not just "auto": ffmpeg starts fine on
        # a box without the hardware encoder and only fails once the
        # first frame reaches it — past the store decision, where the
        # failure would stop frames instead of falling back. The probe
        # is a sub-second four-frame encode and makes the promised
        # fallback real.
        usable = probe_encoder(NVENC_ARGS)
        note("hardware h264 encoder {}; recording camera frames as "
             "{}.".format("available" if usable else "unavailable",
                          "video segments" if usable else "JPEG images"))
        if not usable:
            if self._video == "nvenc":
                warn("--web-video nvenc requested but the encoder probe "
                     "failed; recording JPEG frames instead.")
            return False
        try:
            for name, array in frames:
                height, width = array.shape[:2]
                self._writers[name] = SegmentWriter(
                    self.recording_dir, name, width, height)
        except VideoPipelineError as exc:
            warn("could not start the video encoder ({}); recording JPEG "
                 "frames instead.".format(exc))
            self._close_writers(quiet=True)
            return False
        return True

    def _write_jpegs(self, frames: list, step: int) -> dict:
        written = {}
        for name, array in frames:
            jpeg = self._encode(array)
            if jpeg is not None:
                self._writer.write_frame(name, step, jpeg)
                written[name] = {"bytes": jpeg, "step": step,
                                 "codec": FRAME_STORE_JPEG}
        return written

    def _feed_video(self, frames: list, step: int) -> dict:
        """Encode this step's frames; return the access units now ready
        for the wire, each carrying the step it was fed for.

        Units lag their frames by the encoder's pipeline depth, so a
        step often carries an earlier step's unit or none at all. When
        several arrive at once only the last goes out: delivery is
        latest-only and every unit is a key frame, so nothing downstream
        depends on the ones skipped — they are all on disk regardless.
        """
        written = {}
        for name, array in frames:
            writer = self._writers.get(name)
            if writer is None:
                continue
            try:
                units = writer.feed(array, step)
            except VideoPipelineError as exc:
                self._drop_video(exc)
                break
            for fed, unit in units:
                written[name] = {"bytes": unit, "step": fed,
                                 "codec": _codec_from_unit(unit)}
        return written

    def _drop_video(self, exc: BaseException) -> None:
        """A live video failure costs the frames, not the run: the sink
        stays active so data lines, poses and the monitoring they feed
        keep going. Dropping the writers is what stops the frames."""
        warn("video encoding failed ({}); camera frames stop, data "
             "recording continues.".format(exc))
        self._close_writers(quiet=True)

    def _close_writers(self, quiet: bool = False) -> None:
        """Close every segment writer, whatever happens. A writer whose
        feed failed must still be closed or its ffmpeg process and
        reader thread outlive the run; that close raises the same
        failure again, which is what ``quiet`` swallows."""
        for name, writer in self._writers.items():
            try:
                writer.close()
            except Exception as exc:
                if not quiet:
                    warn("could not close the video writer for {}: "
                         "{}".format(name, exc))
        self._writers = {}

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                if self._poses is not None:
                    self._poses.close()
                # Closing here and not in finalize_episode keeps ffmpeg's
                # stdin owned by this thread: the caller thread would
                # race a still-draining feed onto it.
                self._close_writers()
                return
            if not self.active:
                continue                # drain quietly after an error
            if isinstance(item, _DiscoveryItem):
                try:
                    self._writer._write_discovery_bytes(item.payload)
                    if self._on_discovery is not None:
                        # Same bytes to the server as to disk, so a live
                        # browser and the recording can never describe
                        # different scenes.
                        self._on_discovery(item.payload)
                except Exception as exc:
                    # Losing the sidecar costs the Discover tab and the
                    # offline check. It does not cost the recording, the
                    # frames, the parquet results or the run, so this
                    # warns and carries on rather than taking the
                    # recorder down with it.
                    warn("{}: {}; discovery publication failed; "
                          "monitoring and recording continue.".format(
                              type(exc).__name__, exc))
                continue
            line, frames, poses, geometry = item
            try:
                if geometry is not None:
                    # Before any line: a client that sees a data line must
                    # find the scene it belongs to.
                    write_geometry(self.recording_dir, geometry)
                if frames and self._store is None:
                    self._select_store(frames)
                written = (
                    self._feed_video(frames, line["step"])
                    if self._store == FRAME_STORE_VIDEO
                    else self._write_jpegs(frames, line["step"]))
                record = None
                if poses is not None:
                    xpos, xquat = poses
                    if self._poses is None:
                        self._poses = PosesWriter(
                            self.recording_dir, nbody=len(xpos))
                    self._poses.append(xpos, xquat)
                    record = (xpos.astype("<f4", copy=False).tobytes()
                              + xquat.astype("<f4", copy=False).tobytes())
                # Data lands after frames so on_line fires with both on
                # disk.
                self._writer.append_step(line)
                if self._on_line is not None:
                    if record is None:
                        # Keeps 3-argument callbacks working when the
                        # adapter has no scene capability.
                        self._on_line(line, written, self.frames_dropped)
                    else:
                        self._on_line(line, written, self.frames_dropped,
                                      poses=record)
            except Exception as exc:
                self._disable(exc)

    # -- caller thread ----------------------------------------------------

    def finalize_episode(self, status: str) -> None:
        """Finalize beside session.end(); idempotent, never raises."""
        with self._state_lock:
            if self._finalized:
                return
            self._finalized = True
        try:
            try:
                self._queue.put(_SENTINEL, timeout=2.0)
            except queue.Full:
                pass                    # writer stalled; handled below
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Writer still draining (slow disk / frame burst at
                # episode end). Never close the file under it: leave the
                # recording un-gzipped — still fully readable — and only
                # stamp the terminal status into the manifest. poses.bin
                # is left open too; it is flushed per record, so readers
                # see everything written so far. Silence
                # on_line first: nothing may arrive after episode end.
                self._on_line = None
                self._writer.manifest["status"] = status
                self._writer.rewrite_manifest()
                warn("writer still draining at finalize; recording "
                      "left uncompressed at {}.".format(
                          self.recording_dir))
            else:
                self._writer.finalize(status)
            if self._on_episode_end is not None:
                self._on_episode_end(status)
        except Exception as exc:
            self._disable(exc)

    def close(self, status: str = "failed") -> None:
        """Outer-finally backstop; no-op when already finalized."""
        self.finalize_episode(status)
