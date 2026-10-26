"""Flask-SocketIO dashboard server: live push + point-read replay.

Binds 127.0.0.1 only, no authentication (HRC's documented stance;
remote viewing via SSH port forwarding). ``stop_run`` is the
dashboard's first write capability, knowingly accepted for
127.0.0.1 + same-origin; revisit before any LAN binding.

The recording on disk is the only durable state store. RunIndex holds
bounded derived caches (per-binding series, transition log, sampled
frame steps) that are rebuildable from disk — caches, never truth.
"""

from __future__ import annotations

import gc
import io
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from flask import Flask, request
from flask_socketio import SocketIO

from .delivery import LatestOnlyDelivery
from .protocol import (Bundle, decimate, episode_end, geometry, run_meta,
                       run_status, selector_palette_result)
from .recorder import warn as _warn
from .recording import (FRAME_STORE_JPEG, PosesReader, RecordingReader,
                        VideoFrameReader, frame_path, index_at_or_before,
                        read_geometry)

PACKAGE_ROOT = Path(__file__).resolve().parent

_STATUS_INTERVAL_S = 0.25   # HRC's status-throttle cadence


def wait_for_interrupt() -> None:
    """Block the main thread until Ctrl-C, for the serve-until-exit CLIs.

    The wait is a timed sleep in a loop. Any thread may take a
    process-directed SIGINT, while Python runs the handler only on the
    main thread, and an untimed wait wakes only for a signal delivered
    to that thread. With a long-polling Socket.IO client attached, 9 of
    20 interrupts to an untimed wait were lost this way on the mock
    adapter, leaving the dashboard serving.

    ``gc.freeze()`` then exempts the heap from the interpreter's exit-time
    garbage collection. A served recording's parsed lines can run to tens
    of millions of objects (30 M tracked for a five-step RoboCasa
    recording), and collecting them after main returned kept ``view``
    alive 40 s past Ctrl-C. Nothing in that heap needs a finalizer:
    callers wait here only with the recording finalized or opened
    read-only, and the standard streams and atexit handlers still run.
    """
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        gc.freeze()


LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def _pick_signal(candidates, wanted, index):
    """Which aggregated signal a request names, or why none of them.

    Position is the address, because the rendered pattern is not an
    identifier: ``_glob_for_keys`` returns a bare ``*`` whenever the
    shared prefix and suffix cover the shortest key, so two unrelated
    families in one scope can print identically. Matching on the string
    alone would answer about whichever came first.

    A pattern without a position still resolves when it is unambiguous,
    so a client that only knows the text keeps working; an ambiguous one
    is refused rather than guessed at.
    """
    if (isinstance(index, int) and not isinstance(index, bool)
            and 0 <= index < len(candidates)):
        chosen = candidates[index]
        if not wanted or chosen.pattern == wanted:
            return chosen, None
        # A position that no longer names the row the page thought it
        # did: the scope moved under it. Stale, so refused.
        return None, (
            "signal {!r} is no longer at position {}".format(wanted, index)
        )
    if not wanted:
        return None, "no signal at position {!r}".format(index)
    matches = [
        candidate for candidate in candidates
        if candidate.pattern == wanted
    ]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "no signal named {!r} in this scope".format(wanted)
    return None, (
        "{} signals in this scope are named {!r}; select one by "
        "position".format(len(matches), wanted)
    )


def _palette_row(form) -> dict:
    """One palette form, as the wire carries it.

    The starter is withheld above the expansion cap. It is a thing to
    paste into the draft box, and pressing Check on a form that expands
    to hundreds of millions of formulas runs unbounded compilation under
    a process-global lock. The form itself stays listed and stays
    available, because it is.
    """
    from ..selector_palette import starter_spec_for_form

    too_large = form.available and _starter_too_large(form)
    return {
        "reference": form.reference,
        "partner_reference": form.partner_reference,
        "binding_kind": form.binding_kind,
        "component_kind": form.component_kind,
        "quantifier": form.quantifier,
        "note": form.note,
        "available": form.available,
        "unavailable_reason": form.unavailable_reason,
        "verified": form.verified,
        "expansions": form.expansions,
        # On the wire because the page has to land a role selection on
        # the row that reproduces it. Without this the browser cannot
        # implement `default_form` even if it wants to, and a role scope
        # lands on a class-wide reference that monitors entities the
        # selection excluded.
        "reproduces_role": form.reproduces_role,
        "starter": (
            starter_spec_for_form(form)
            if form.available and not too_large else None
        ),
        "starter_withheld_reason": (
            "expands to {:,} formulas; too large to check "
            "interactively".format(form.expansions) if too_large else None
        ),
    }


def _starter_too_large(form) -> bool:
    from ..selector_palette import STARTER_EXPANSION_CAP

    return (form.expansions or 0) > STARTER_EXPANSION_CAP


def _validated_constants(value):
    from ..spec.semantic import validate_constants

    return validate_constants(value)


def host_name(host_header: Optional[str]) -> str:
    """The name part of a Host header, with any port removed.

    Handles the three spellings a client can send: ``localhost:8090``,
    a bracketed IPv6 literal ``[::1]:8090``, and a bare name with no
    port at all (browsers omit it on 80 and 443).
    """
    value = (host_header or "").strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else ""
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value                    # bare name, or unbracketed IPv6


def host_is_allowed(environ) -> bool:
    """Whether a request's Host header names a loopback address.

    The same-origin posture this server documents does not survive DNS
    rebinding, and this is the check that does. engineio derives its
    allowed origin from the request's *own* Host header
    (``engineio/base_server.py``), so a page on an attacker-controlled
    name whose DNS has been rebound to 127.0.0.1 sends
    ``Host: evil.example:8090`` with a matching Origin, and the origin
    check passes. It also skips the check entirely when Origin is absent
    (``engineio/server.py``), which a non-browser client simply omits.

    The name is what matters and the port is not checked. An attacker
    who could make a browser send ``localhost:9999`` could equally make
    it send the real port, so the port defends nothing -- while
    requiring it breaks a documented workflow: ``ssh -L 8091:...:8090``
    makes the browser send the *client-side* port, and browsers omit the
    port entirely on 80 and 443. What an attacker cannot do is make a
    rebound page send a loopback *name*, because that name is not theirs
    to point anywhere.
    """
    return host_name(environ.get("HTTP_HOST")) in LOOPBACK_NAMES


class LoopbackOnly:
    """WSGI wrapper refusing any request whose Host is not loopback.

    Outermost, and that placement is the whole point. SocketIO wraps the
    Flask app and intercepts ``/socket.io/`` ahead of Flask's routing, so
    a ``before_request`` guard never sees transport traffic — and the
    Engine.IO handshake allocates a session id *before* any namespace
    handler runs. Measured against a live server: a forged Host returned
    200 with a sid. Nothing crossed, because joining the namespace is a
    later step, but the session was real.

    Wrapping the composed application is the only single place that sees
    every request: Flask routes, static files, the Engine.IO handshake,
    and the WebSocket upgrade. One guard that misses nothing beats two
    that each miss something.
    """

    def __init__(self, application) -> None:
        self._application = application

    def __call__(self, environ, start_response):
        if not host_is_allowed(environ):
            start_response(
                "403 Forbidden",
                [("Content-Type", "text/plain; charset=utf-8")],
            )
            return [b"Not a loopback host.\n"]
        return self._application(environ, start_response)


class DiscoverySnapshot:
    """One episode's scene, decoded once and then never mutated.

    Holds the gzip bytes the browser downloads, the digest that proves
    which episode they came from, and the Python model every palette
    request is resolved against. The model is the authority: a browser
    sends strings, and strings are resolved here rather than trusted.

    The semantic variant is derived on first request and cached. It is
    much smaller than the raw one — 0.52 MiB against 1.66 on RoboCasa —
    and it is what a person browsing entities actually wants, so making
    the common case cheap is worth one lazy re-encode.
    """

    __slots__ = ("raw_gzip", "digest", "_model", "_error",
                 "_semantic_gzip", "_lock")

    def __init__(self, raw_gzip: bytes, model=None) -> None:
        import hashlib

        self.raw_gzip = raw_gzip
        self.digest = hashlib.sha256(raw_gzip).hexdigest()
        self._model = model
        self._error: Optional[BaseException] = None
        self._semantic_gzip: Optional[bytes] = None
        self._lock = threading.Lock()

    @classmethod
    def from_gzip(cls, payload: bytes) -> "DiscoverySnapshot":
        """Take the bytes now; build the model when one is first needed.

        Publishing runs on the recorder's writer thread, which is the
        thread that drains the snapshot queue. Decompressing and
        rebuilding the model there was measured at 0.504 s on
        results/20260816_004207_robocasa_notask (1.10 MiB gzipped,
        37.73 MiB decoded, 2,053 entities) -- half a second during which
        no snapshot line is written and frames pile up against a bounded
        queue, where they are dropped. Nothing on that thread needs the
        model; the first palette or check request does, and it can pay
        for it.

        The bytes are still validated here, cheaply: an unparseable
        payload must be rejected at publish time so the run warns once
        rather than failing every later request.
        """
        import gzip as _gzip

        if payload[:2] != b"\x1f\x8b":
            raise ValueError("discovery payload is not gzip")
        with _gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
            if handle.read(1) != b"{":
                raise ValueError("discovery payload is not a JSON object")
        return cls(payload)

    @property
    def model(self):
        """The Python model, built on first use and then reused.

        A document that gzips and parses can still fail to restore --
        a missing ``schema_version``, an entity record the restorer
        refuses. That failure is remembered rather than re-raised from
        scratch each time, so it can be *reported* by
        ``_discovery_ready`` and the routes instead of surfacing as a
        500 on every request.
        """
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._model is None:
                import gzip as _gzip
                import json as _json

                from ..discovery_restore import (
                    discovery_model_from_document,
                )

                try:
                    self._model = discovery_model_from_document(
                        _json.loads(_gzip.decompress(self.raw_gzip))
                    )
                except Exception as exc:            # noqa: BLE001
                    self._error = exc
                    raise
            return self._model

    def restore_failure(self) -> Optional[BaseException]:
        """Why this snapshot cannot be used, or None while it can.

        Resolves the model, so callers must not be on the recorder's
        writer thread. ``set_discovery_snapshot`` calls it from a
        throwaway thread for exactly that reason: the run learns within
        a moment whether its own sidecar is usable, and says so once,
        instead of every browser request discovering it again.
        """
        try:
            self.model
        except Exception as exc:                    # noqa: BLE001
            return exc
        return None

    def gzip_for(self, variant: str) -> bytes:
        # Resolve on the requesting thread before returning either variant.
        # Gzip framing alone cannot establish ordinary discovery eligibility.
        model = self.model
        if variant != "semantic":
            return self.raw_gzip
        # ``self.model`` takes the same lock, so it is resolved before
        # entering it rather than inside.
        with self._lock:
            if self._semantic_gzip is None:
                import gzip as _gzip

                from ..discovery_filter import (
                    DiscoveryQuery,
                    filter_discovery,
                )
                from ..discovery_serialize import discovery_json

                view = filter_discovery(
                    model, DiscoveryQuery(include_raw=False)
                )
                self._semantic_gzip = _gzip.compress(
                    discovery_json(view).encode("utf-8"), 6
                )
            return self._semantic_gzip


class RunIndex:
    """Derived caches over the recording; rebuildable from disk.

    Series arrays and the transition log grow with run length (a
    scalar per binding per step) — proportional to the recording, never
    larger, and always reconstructible from it.
    """

    def __init__(self) -> None:
        self.specs: List[dict] = []
        self.bindings: Dict[Tuple[str, str], dict] = {}
        # (spec, key) -> ([steps], [rho-or-None]); append-ordered
        self.series: Dict[Tuple[str, str], Tuple[list, list]] = {}
        # Ingested steps in arrival order (ascending): the pose sidecar
        # joins its records to data lines by ordinal, so this list is
        # what turns a step number into a record index.
        self.steps: List[int] = []
        self.sim_times: Dict[int, Optional[float]] = {}
        self.transitions: List[dict] = []
        self.max_step = -1
        # First recorded step: tracker timesteps need not start at 0
        # (robocasa starts at 1); seeks clamp to [min_step, max_step].
        self.min_step = 0
        self._min_seen = False
        self.sampled: Dict[str, List[int]] = {}

    def ingest_line(self, line: dict) -> None:
        for spec in line.get("new_specs", []):
            self.specs.append(spec)
        for meta in line.get("new_bindings", []):
            self.bindings[(meta["spec"], meta["key"])] = meta
        step = line["step"]
        self.steps.append(step)
        self.sim_times[step] = line.get("sim_time")
        for spec in line["specs"]:
            for binding in spec["bindings"]:
                series = self.series.setdefault(
                    (spec["name"], binding["key"]), ([], []))
                series[0].append(step)
                series[1].append(binding["rho"])
        self.transitions.extend(line["transitions"])
        if step > self.max_step:
            self.max_step = step
        if not self._min_seen or step < self.min_step:
            self.min_step = step
            self._min_seen = True

    def note_frames(self, cameras_step: dict) -> None:
        for camera, step in cameras_step.items():
            self.sampled.setdefault(camera, []).append(step)

    def nearest_frame_step(self, camera: str, step: int) -> Optional[int]:
        """Nearest earlier sampled step; before the first sample, the
        first sample (capture may start a step late — an honest caption
        beats an empty pane)."""
        steps = self.sampled.get(camera)
        if not steps:
            return None
        index = index_at_or_before(steps, step)
        return steps[index] if index is not None else steps[0]

    @classmethod
    def from_reader(cls, reader: RecordingReader) -> "RunIndex":
        index = cls()
        for line in reader.all_lines():
            index.ingest_line(line)
        for camera in reader.cameras():
            index.sampled[camera] = reader.frame_steps(camera)
        return index


class _ClientState:
    def __init__(self, sid: str, sender, *, follow: bool) -> None:
        self.sid = sid
        self.epoch = 0
        self.seq = 0
        self.follow = follow
        self.position: Optional[int] = None
        self.cameras: Optional[List[str]] = None
        self.playing = False
        self.rate = 1.0
        self.play_generation = 0
        self.geometry_sent = False
        self.delivery = LatestOnlyDelivery(sender)
        # Serializes epoch/seq/follow/position between the recorder
        # writer thread (on_line), playback tasks, and socketio handler
        # threads: without it a seek's epoch bump can interleave with a
        # live bundle build and produce a stale bundle wearing the new
        # epoch. Never hold while emitting.
        self.lock = threading.Lock()


class DashboardServer:
    def __init__(self, recording_dir, *, mode: str) -> None:
        if mode not in ("live", "view"):
            raise ValueError("mode must be 'live' or 'view'")
        self.mode = mode
        self.recording_dir = Path(recording_dir)
        self.reader = RecordingReader(self.recording_dir)
        if mode == "view":
            self.index = RunIndex.from_reader(self.reader)
        else:
            self.index = RunIndex()
        self.stop_requested = threading.Event()
        self._stop_callbacks: List[Callable[[], None]] = []
        self._ended = mode == "view"
        self._ended_status: Optional[str] = (
            self.reader.manifest.get("status") if mode == "view" else None)
        self.frames_dropped = 0
        # "Not shut down": gates playback loops and the CLI's
        # serve-until-exit message; start() only adds the URL.
        self.active = True
        self.url: Optional[str] = None
        self._clients: Dict[str, _ClientState] = {}
        self._clients_lock = threading.Lock()
        # Last disk-read JPEG frame per camera: most steps re-serve the
        # same nearest sample (stride > 1, paused clients), so skip the
        # re-read. One entry per camera, bounded.
        self._frame_cache: Dict[str, Tuple[int, bytes]] = {}
        # Last frame entry delivered live per camera, one each. The
        # video store cannot read back the newest frames — a segment is
        # only indexed once ffmpeg closes it, which is dozens of steps
        # in — so for that window this is the only copy that exists.
        self._wire_frames: Dict[str, dict] = {}
        self._frames_seen = False
        # Segment readers, opened on first use: a live run's video index
        # appears mid-episode, after this server was constructed.
        self._video_readers: Dict[str, VideoFrameReader] = {}
        # 3D scene sidecars, both absent for adapters without the
        # capability: the geometry payload is read at most once when
        # present; a missing sidecar retries its open per lookup, since
        # the live recorder may write it after the first client arrives.
        self._geometry: Optional[dict] = None
        self._geometry_probed = False
        self._poses_reader: Optional[PosesReader] = None
        # The episode's scene. Live runs publish it when the adapter's
        # own reset happens; a recording carries it on disk. Absent for
        # recordings written before the sidecar existed, and for runs
        # whose capture failed -- both of which stay fully replayable.
        self._discovery: Optional[DiscoverySnapshot] = None
        self._context = None
        # scope -> verified palette. Dropped whenever the scene changes,
        # because a verdict belongs to the scene it was reached against.
        self._palette_cache: Dict[tuple, tuple] = {}
        if mode == "view":
            self._load_discovery_from_disk()
        self._last_status_at = 0.0
        self._emit_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.app, self.socketio = self._create_app()

    # -- wiring -----------------------------------------------------------

    def _load_discovery_from_disk(self) -> None:
        from .recording import read_discovery_bytes

        payload = read_discovery_bytes(self.recording_dir)
        if payload is None:
            return
        try:
            self._discovery = DiscoverySnapshot.from_gzip(payload)
            # The compiled scene belongs to the old snapshot; a draft
            # checked against it would report on an episode that is no
            # longer the one being served.
            self._context = None
        except Exception as exc:                    # noqa: BLE001
            # A recording whose scene cannot be decoded still replays;
            # only Discover and the offline check go unavailable.
            _warn("{}: {}; discovery unavailable for this recording, "
                  "replay unaffected.".format(type(exc).__name__, exc))

    def set_discovery_snapshot(self, payload: bytes) -> None:
        """Publish the live episode's scene to connected clients.

        Called once, from the recorder's writer thread, with the same
        bytes written to disk -- so the browser and the sidecar can
        never describe different scenes.
        """
        try:
            snapshot = DiscoverySnapshot.from_gzip(payload)
        except Exception as exc:                    # noqa: BLE001
            _warn("{}: {}; discovery unavailable for this run, "
                  "monitoring unaffected.".format(type(exc).__name__, exc))
            return
        self._discovery = snapshot
        self._palette_cache = {}
        # The compiled scene belongs to the old snapshot; a draft checked
        # against it would report on an episode that is no longer the one
        # being served.
        self._context = None
        self._emit("discovery_ready", self._discovery_ready(), sid=None)

        # Restoring costs 0.4-0.6 s on RoboCasa and this is the
        # recorder's writer thread, which must get back to draining the
        # snapshot queue. Proving the document restorable is still worth
        # doing eagerly -- a run should warn once, now, rather than have
        # every later browser request rediscover the same failure -- so
        # it happens on a thread of its own.
        threading.Thread(
            target=self._announce_restore_failure, args=(snapshot,),
            name="specola-discovery-restore", daemon=True,
        ).start()

    def _announce_restore_failure(self, snapshot) -> None:
        failure = snapshot.restore_failure()
        if failure is None or self._discovery is not snapshot:
            return
        _warn("{}: {}; this run's scene cannot be rebuilt, so Discover "
              "and checking are unavailable. Monitoring, recording and "
              "results are unaffected.".format(
                  type(failure).__name__, failure))
        self._emit("discovery_ready", self._discovery_ready(), sid=None)

    def _discovery_ready(self) -> dict:
        from .protocol import discovery_ready

        snapshot = self._discovery
        if snapshot is None:
            return discovery_ready(
                self.reader.manifest.get("run_id"),
                available=False, digest=None,
                reason="This recording carries no discovery snapshot.",
            )
        # A snapshot whose restore already failed is reported unavailable
        # with the reason, so the tab stays disabled and explains itself
        # rather than enabling and then 500-ing on its first fetch. Only
        # a *known* failure counts: asking here would restore the model
        # on whichever thread called this.
        failure = snapshot._error
        return discovery_ready(
            self.reader.manifest.get("run_id"),
            available=failure is None,
            digest=snapshot.digest,
            reason=(
                None if failure is None else
                "This run's scene cannot be rebuilt ({}: {}), so Discover "
                "and checking are unavailable. Monitoring, recording and "
                "results are unaffected.".format(
                    type(failure).__name__, failure)
            ),
        )

    def on_stop_run(self, callback: Callable[[], None]) -> None:
        self._stop_callbacks.append(callback)
        # A stop_run may land before a UI path registers (e.g. the
        # textual driver thread starting up); honor it immediately so
        # the stop is not lost in that window.
        if self.stop_requested.is_set():
            callback()

    def start(self, port: int) -> str:
        """Bind 127.0.0.1 and serve from a daemon thread; returns the URL.

        Raises OSError while still on the caller's thread when the port
        is busy. ``port == 0`` picks a free ephemeral port.
        """
        # ponytail: bind-probe then close has a race window; acceptable
        # for a localhost dev server.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # SO_REUSEADDR to match werkzeug (allow_reuse_address=True):
            # without it a just-killed server's TIME_WAIT socket fails
            # this probe for ~60s even though nothing is listening and
            # werkzeug itself would rebind fine. A live listener still
            # fails the bind either way.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
            port = probe.getsockname()[1]
        finally:
            probe.close()
        def _serve() -> None:
            try:
                self.socketio.run(
                    self.app, host="127.0.0.1", port=port,
                    allow_unsafe_werkzeug=True)
            except Exception as exc:
                # Failure isolation: a server crash warns once and only
                # disables the dashboard; monitoring is untouched.
                _warn("{}: {}; dashboard server stopped, monitoring "
                      "continues.".format(type(exc).__name__, exc))
            finally:
                # A crashed server must not leave active=True or the
                # CLI's serve-until-exit wait would hang on a dead URL.
                self.active = False

        self._thread = threading.Thread(
            target=_serve, name="specola-web-server", daemon=True)
        self._thread.start()
        self.url = "http://127.0.0.1:{}".format(port)
        return self.url

    def shutdown(self) -> None:
        self.active = False

    # -- live feed (recorder writer thread) --------------------------------

    def on_line(self, line: dict, frames: Dict[str, dict],
                frames_dropped: int, poses: Optional[bytes] = None) -> None:
        if self._ended:
            # A still-draining writer must not push past episode end.
            return
        self.index.ingest_line(line)
        first_frames = bool(frames) and not self._frames_seen
        if frames:
            # An entry is sampled at the step its bytes were encoded
            # for, which lags the line it arrives with by the encoder's
            # pipeline; the encoder emits in order, so these stay
            # ascending and the sampled-step bisect holds.
            self.index.note_frames({name: entry["step"]
                                    for name, entry in frames.items()})
            self._wire_frames.update(frames)
            self._frames_seen = True
        self.frames_dropped = frames_dropped
        if line.get("new_specs") or line.get("new_bindings") or first_frames:
            # New bindings while a client is detached update run_meta
            # without disturbing its position. So does the first frame:
            # the writer thread stamps the frame store into the manifest
            # just before that callback, and under a capture stride that
            # can land long after the last new_bindings line.
            self.reader.reload_manifest()
            self._emit("run_meta", self._run_meta(), sid=None)
        if not self._geometry_probed:
            # The recorder writes the scene files before the first line
            # callback, so one read settles the run — and any client that
            # connected before them (live: it connected before step 0)
            # gets the event from the loop below.
            self._geometry_payload()
            self._geometry_probed = True
        now = time.monotonic()
        status_due = now - self._last_status_at >= _STATUS_INTERVAL_S
        if status_due:
            self._last_status_at = now
        with self._clients_lock:
            clients = list(self._clients.values())
        live_poses = (None if poses is None
                      else {"bytes": poses, "step": line["step"]})
        for state in clients:
            self._send_geometry(state)
            state.delivery.expire()
            bundle = None
            with state.lock:
                if state.follow:
                    state.position = line["step"]
                    bundle = self._make_bundle(
                        state, line, line["step"], live=True,
                        live_frames=frames, poses=live_poses)
            if bundle is not None:
                state.delivery.offer(bundle)
            elif status_due:
                self._emit("run_status",
                           run_status(self.reader.manifest.get("run_id"),
                                      max_step=self.index.max_step,
                                      drops=self._drops(state)),
                           sid=state.sid)

    def on_episode_end(self, status: str) -> None:
        """Recorder finalized the episode: swap the reader (the JSONL was
        gzipped; old byte offsets are dead) and unlock replay controls."""
        self._ended = True
        self._ended_status = status
        self.reader = RecordingReader(self.recording_dir)
        with self._clients_lock:
            clients = list(self._clients.values())
        for state in clients:
            with state.lock:
                state.follow = False       # there is no head to follow
        self._emit("episode_end",
                   episode_end(self.reader.manifest.get("run_id"),
                               status=status,
                               max_step=self.index.max_step,
                               view_command=self._view_command()),
                   sid=None)

    # -- internals ----------------------------------------------------------

    @property
    def phase(self) -> str:
        """The tri-state the wire calls ``mode``: ``live`` (head still
        being appended), ``ended`` (live run finished, now replayable),
        or ``view`` (opened from disk)."""
        if self.mode == "view":
            return "view"
        return "ended" if self._ended else "live"

    def _view_command(self) -> str:
        return "specola view {}".format(self.recording_dir)

    def _drops(self, state: _ClientState) -> dict:
        return {"frames": self.frames_dropped,
                "delivery": state.delivery.dropped}

    def _run_meta(self) -> dict:
        return run_meta(
            self.reader.manifest, mode=self.phase,
            specs=list(self.index.specs),
            bindings=list(self.index.bindings.values()),
            min_step=self.index.min_step,
            max_step=self.index.max_step,
            view_command=self._view_command())

    def _geometry_payload(self) -> Optional[dict]:
        """The recording's scene files, read at most once per run.

        A miss is not cached before the first line arrives: a live client
        can connect while the recorder has yet to write the files.
        """
        if self._geometry is None and not self._geometry_probed:
            self._geometry = read_geometry(self.recording_dir)
        return self._geometry

    def _send_geometry(self, state: _ClientState) -> None:
        """Emit the scene to one client, at most once per connection."""
        payload = self._geometry_payload()
        if payload is None:
            # Do not latch the flag on a miss: the live recorder may not
            # have written the files yet, and a latched miss would skip
            # this client forever.
            return
        # The connect handler and the recorder's on_line thread can race
        # here for the same client; the flag flip must be atomic so the
        # MB-class payload goes out once. Emit outside the lock.
        with state.lock:
            if state.geometry_sent:
                return
            state.geometry_sent = True
        self._emit("geometry",
                   geometry(self.reader.manifest.get("run_id"),
                            meta=payload["json"], meshes=payload["meshes"]),
                   sid=state.sid)

    def _poses(self, step: int) -> Optional[dict]:
        """The sidecar's pose record for ``step``, or None when the run
        has no scene.

        Records join snapshots.jsonl by ordinal, never by step number
        (robocasa steps start at 1), so the step is first located in the
        index's ascending arrival order.
        """
        if self._poses_reader is None:
            self._poses_reader = PosesReader.open(self.recording_dir)
            if self._poses_reader is None:
                return None
        ordinal = index_at_or_before(self.index.steps, step)
        if ordinal is None or self.index.steps[ordinal] != step:
            return None
        record = self._poses_reader.record(ordinal)
        return None if record is None else {"bytes": record, "step": step}

    def _emit(self, event: str, payload: dict, sid: Optional[str]) -> None:
        # python-socketio emits can span multiple packets when binary
        # image bytes are attached; serialize all outbound writes to
        # prevent packet interleaving across threads (HRC precedent).
        with self._emit_lock:
            self.socketio.emit(event, payload, to=sid)

    def _select_frames(self, state: _ClientState, step: int,
                       live_frames: Optional[Dict[str, dict]] = None
                       ) -> dict:
        cameras = state.cameras
        if cameras is None:
            if live_frames:
                cameras = [sorted(live_frames)[0]]
            else:
                available = sorted(self.index.sampled) or self.reader.cameras()
                cameras = available[:1]
        frames = {}
        for camera in cameras:
            if live_frames is not None and camera in live_frames:
                # The recorder's entry goes out as it came in: it names
                # the step its bytes were encoded for and the codec they
                # are in, and nothing here knows better than the encoder.
                frames[camera] = live_frames[camera]
                continue
            entry = self._disk_entry(camera, step)
            wire = self._wire_frames.get(camera)
            if (wire is not None and wire["step"] <= step
                    and (entry is None or wire["step"] > entry["step"])):
                # Newer than anything readable: this is the live window
                # the store has not caught up with yet.
                entry = wire
            if entry is not None:
                frames[camera] = entry
        return frames

    def _video_reader(self, camera: str) -> Optional[VideoFrameReader]:
        """This camera's segment reader, or None when it has no video
        index — re-probed per lookup while absent, like the pose
        sidecar: a live run's index appears once its first segment
        closes, which is after this server was constructed."""
        reader = self._video_readers.get(camera)
        if reader is None:
            reader = VideoFrameReader.open(self.recording_dir, camera)
            if reader is not None:
                self._video_readers[camera] = reader
        return reader

    def _disk_entry(self, camera: str, step: int) -> Optional[dict]:
        """Nearest-earlier stored frame for ``camera``, as a wire entry.

        A video camera answers for itself: the requested step goes to
        the segment reader, which resolves it against its complete index
        and reports the step it actually served. The sampled steps this
        server holds are the ones the wire delivered — behind the index
        whenever a burst of units arrived at once — so resolving here
        would name an older frame than the store can serve. Nothing is
        cached on that path either: the served step is only known after
        the read a cache would have saved, and it changes under the
        reader as ffmpeg closes the segment it is inside.

        The JPEG store keeps both: its sampled steps are exactly the
        files on disk, and the reader indexes those files once at
        construction, so a live run's frames are reachable by path and
        no other way.
        """
        video = self._video_reader(camera)
        if video is not None:
            found = video.annexb(step)
            if found is None:
                # Before the first sample: serve it anyway, mirroring
                # the JPEG clamp — an honest caption beats an empty
                # pane, and capture may start a step late.
                steps = video.steps()
                if steps:
                    found = video.annexb(steps[0])
            return (None if found is None
                    else {"bytes": found[0], "step": found[1],
                          "codec": video.codec})
        nearest = self.index.nearest_frame_step(camera, step)
        if nearest is None:
            found = self.reader.frame_entry(camera, step)
            return (None if found is None
                    else {"bytes": found[0], "step": found[2],
                          "codec": found[1]})
        cached = self._frame_cache.get(camera)
        if cached is not None and cached[0] == nearest:
            data = cached[1]
        else:
            try:
                data = frame_path(
                    self.recording_dir, camera, nearest).read_bytes()
            except OSError:
                return None
            self._frame_cache[camera] = (nearest, data)
        return {"bytes": data, "step": nearest, "codec": FRAME_STORE_JPEG}

    def _read_line(self, step: int) -> Optional[dict]:
        try:
            return self.reader.read_step(step)
        except (KeyError, OSError):
            # OSError included: a live seek can race the finalize
            # path that gzips and unlinks snapshots.jsonl before
            # the reader is swapped; the fallback below must still
            # run rather than the handler crashing (review catch).
            # refresh() self-guards on finalized readers (returns 0).
            self.reader.refresh()
        try:
            return self.reader.read_step(step)
        except (KeyError, OSError):
            pass
        # A silent miss made a live seek look ignored: the thumb moved
        # while every datum on screen stayed at the head (the sought
        # line can still be in the writer queue). Serve the nearest
        # earlier recorded step instead; the bundle carries that line's
        # own step, so the client shows the truth it was served.
        nearest = self.reader.nearest_step(step)
        if nearest is None or nearest == step:
            return None
        try:
            return self.reader.read_step(nearest)
        except (KeyError, OSError):
            return None

    def _point_read(self, state: _ClientState, step: int) -> bool:
        """Serve exactly the sought step (never latest-only).

        Caller holds ``state.lock``.
        """
        line = self._read_line(step)
        if line is None:
            return False
        # The nearest-earlier fallback may serve a different step than
        # requested; keep the client's position honest about it.
        position = int(line.get("step", step))
        state.position = position
        bundle = self._make_bundle(state, line, position, live=False)
        self._emit("step_bundle", bundle.to_wire(), sid=state.sid)
        return True

    def _make_bundle(self, state: _ClientState, line: dict, step: int, *,
                     live: bool,
                     live_frames: Optional[Dict[str, dict]] = None,
                     poses: Optional[dict] = None) -> Bundle:
        """Build the next bundle for ``state`` and bump its seq.

        Live bundles carry the poses the recorder pushed with the line;
        seeks and playback read the sidecar for the step being served.
        Caller holds ``state.lock``.
        """
        if poses is None and not live:
            # Key on the line actually being served: the nearest-earlier
            # read can hand back a different step than the one sought,
            # and the poses must describe that line.
            poses = self._poses(int(line.get("step", step)))
        bundle = Bundle(
            run_id=self.reader.manifest.get("run_id", ""),
            epoch=state.epoch, seq=state.seq, line=line,
            frames=self._select_frames(state, step,
                                       live_frames=live_frames),
            max_step=self.index.max_step, live=live,
            drops=self._drops(state), poses=poses)
        state.seq += 1
        return bundle

    def _playback_loop(self, state: _ClientState, generation: int) -> None:
        """Per-client replay advance (ended/view modes only)."""
        # ponytail: per-step sleep accumulates drift; wall-clock anchor
        # like HRC's PlaybackClock if it ever matters.
        while (state.playing and state.play_generation == generation
               and self.active):
            position = state.position if state.position is not None else -1
            next_step = position + 1
            if next_step > self.index.max_step:
                state.playing = False
                break
            current = self.index.sim_times.get(position)
            upcoming = self.index.sim_times.get(next_step)
            delay = 0.0
            if current is not None and upcoming is not None:
                delay = max(0.0, (upcoming - current) / max(state.rate, 1e-6))
            # stub annotates seconds as int; runtime accepts float
            self.socketio.sleep(
                min(delay, 1.0))  # pyright: ignore[reportArgumentType]
            if not (state.playing and state.play_generation == generation):
                break
            line = self._read_line(next_step)
            if line is None:
                state.playing = False
                break
            state.delivery.expire()
            with state.lock:
                if state.play_generation != generation:
                    break
                state.position = next_step
                bundle = self._make_bundle(state, line, next_step,
                                           live=False)
            state.delivery.offer(bundle)

    def _apply_view(self, state: _ClientState,
                    payload: dict) -> Tuple[dict, bool]:
        """Apply one client view change. Caller holds ``state.lock``.

        Every accepted change bumps the epoch and resyncs the client's
        seq guard; delivery state is discarded. Returns the ack payload
        and whether a playback loop must be started (after releasing
        the lock).
        """
        # Validate the whole patch BEFORE mutating: a rejected view
        # change must not leave earlier fields partially applied.
        cameras = None
        if payload.get("cameras") is not None:
            cameras = [str(name) for name in payload["cameras"]]
        rate = None
        if payload.get("rate") is not None:
            rate = float(payload["rate"])
            if not 0.01 <= rate <= 64:
                raise ValueError("rate out of range")
        seek = None
        if payload.get("seek") is not None:
            seek = max(self.index.min_step,
                       min(int(payload["seek"]),
                           max(self.index.max_step, self.index.min_step)))

        state.epoch += 1
        state.seq = 0
        state.delivery.reset()
        if cameras is not None:
            state.cameras = cameras or None
        if rate is not None:
            state.rate = rate
        if seek is not None:
            state.follow = False
            state.position = seek
            self._point_read(state, seek)
        if payload.get("follow") and self.phase == "live":
            state.follow = True
        start_playback = False
        if payload.get("play") is not None:
            wants_play = bool(payload["play"])
            state.play_generation += 1
            if wants_play and self._ended:
                state.playing = True
                start_playback = True
            else:
                state.playing = False
        return ({"ok": True, "epoch": state.epoch,
                 "position": state.position}, start_playback)

    # What a page sends for "entities with no declared role". A string,
    # because JSON has no way to name Python's sentinel; it is mapped to
    # ROLE_ABSENT here and nowhere else. The NUL makes it unmatchable
    # against any real role name.
    ROLELESS_TOKEN = "\u0000roleless"

    def _scope_from(self, payload: dict):
        """The class/role/entity a browser asked about, resolved here.

        A browser sends strings. They name a scope; they never become a
        path, a formula or an adapter access. A role-only scope carries
        no class, exactly as the terminal's role browse mode does, so an
        absent ``entity_class`` is a scope rather than an error whenever
        a role is present.
        """
        from ..discovery_selector import DiscoveryScope, ROLE_ABSENT

        raw_role = payload.get("role")
        role = None
        if raw_role == self.ROLELESS_TOKEN:
            role = ROLE_ABSENT
        elif raw_role:
            role = str(raw_role)
        entity_class = payload.get("entity_class")
        if not entity_class and role is None:
            return None
        entity_id = payload.get("entity_id")
        return DiscoveryScope(
            entity_class=str(entity_class) if entity_class else None,
            role=role,
            entity_id=int(entity_id) if entity_id is not None else None,
        )

    def _view_for(self, payload: dict):
        from ..discovery_filter import DiscoveryQuery, filter_discovery

        return filter_discovery(
            self._discovery.model,
            DiscoveryQuery(include_raw=bool(payload.get("include_raw"))),
        )

    def _resolve_signals(self, request_id, payload: dict) -> dict:
        """Every signal in one scope, aggregated by the shared Python.

        Coverage counts are deliberately absent (maintainer's call). They
        are still what tells an author that ``object_*[hinge_angle]``
        binds one of two objects rather than both, so a partial fan-out
        now shows up at check time -- the checker's report names the
        bound keys -- rather than in this list.
        """
        from ..discovery_selector import signal_selections
        from .protocol import signal_rows_result

        scope = self._scope_from(payload)
        if scope is None:
            return signal_rows_result(
                request_id, signals=[],
                reason="no class or role selected")
        view = self._view_for(payload)
        signals = signal_selections(view, scope)
        return signal_rows_result(
            request_id,
            scope={"entity_class": scope.entity_class,
                   "entity_id": scope.entity_id},
            signals=[
                {"pattern": signal.pattern,
                 "quantity": signal.quantity,
                 "component": signal.component,
                 "unit": signal.unit,
                 "frame": signal.frame,
                 # One example mapping, so a person can see a real key
                 # and a real value without the page holding all of them.
                 "example_key": (
                     signal.mappings[0].exact_key if signal.mappings else None
                 ),
                 "example_value": (
                     signal.mappings[0].value if signal.mappings else None
                 )}
                for signal in signals
            ],
            reason=None if signals else "no signals in this scope",
        )

    def _resolve_palette(self, request_id, payload: dict) -> dict:
        """Turn a requested class/entity/signal into verified forms."""
        scope = self._scope_from(payload)
        if scope is None:
            return selector_palette_result(
                request_id, forms=[],
                reason="no class or role selected")
        view = self._view_for(payload)
        answered = {"entity_class": scope.entity_class,
                    "entity_id": scope.entity_id,
                    "signal": payload.get("signal")}

        signal, refusal = self._signal_for(view, scope, payload)
        if refusal is not None:
            # Refused, never widened. Falling back to the whole scope
            # answers a question nobody asked: a person who selected one
            # signal would be handed the forms for every component in the
            # class and could copy a starter for an unrelated one.
            return selector_palette_result(
                request_id, scope=answered, forms=[], reason=refusal)

        forms = self._cached_palette(view, scope, signal, payload)
        return selector_palette_result(
            request_id, scope=answered,
            forms=[_palette_row(form) for form in forms],
            reason=None if forms else "no selector form for this selection",
        )

    def _signal_for(self, view, scope, payload):
        """Which signal the request names, or why none of them."""
        from ..discovery_selector import signal_selections

        wanted = payload.get("signal")
        index = payload.get("signal_index")
        if not wanted and index is None:
            return None, None
        return _pick_signal(signal_selections(view, scope), wanted, index)

    def _cached_palette(self, view, scope, signal, payload):
        """The verified palette for one selection, built at most once.

        Keyed before enumeration, not after: building the palette is
        itself 249 ms on `fixture`, so a cache covering only the
        verification left most of the repeat cost in place. The scene is
        immutable for a snapshot's lifetime, so nothing under this key
        can go stale while the key is valid, and publishing a new
        snapshot drops the whole cache.
        """
        from ..selector_palette import selector_palette, verify_forms

        # `signal_index` is in the key because `_pick_signal` resolves by
        # position: two unrelated families in one scope can print the
        # same pattern text, which is the whole reason that function
        # takes an index. A key that dropped it would hand the second
        # row's request the first row's palette, defeating the
        # disambiguation one function below.
        key = (scope.entity_class, scope.role, scope.entity_id,
               payload.get("signal"), payload.get("signal_index"),
               bool(payload.get("include_raw")))
        forms = self._palette_cache.get(key)
        if forms is not None:
            return forms

        # Verified as a palette, not one focused row at a time. Bracket
        # availability and the semantic layer disagree about 10 of
        # RoboCasa's 16 `contact` forms, so a list of unverified rows
        # offers ten illegal selectors as though they were fine.
        # Verifying on focus was the right shape while one form cost
        # seconds to compile; expansion-only made the whole contact
        # palette 8 ms and the widest class 2.4 s.
        forms = selector_palette(view, scope, signal=signal)
        context, _error = self._check_context()
        if context is not None:
            forms = verify_forms(
                forms, catalog=context.catalog, dt=context.dt,
                capabilities=context.capabilities,
            )
        self._palette_cache[key] = forms
        return forms

    def _check_context(self):
        """The scene a draft is compiled against, or why there is none.

        Keyed to the snapshot it was built from, not merely cached.
        Building takes a moment, and a publish landing inside that window
        sets ``_context = None`` from the recorder's thread -- a plain
        assignment afterwards would overwrite that reset with a context
        for the *previous* episode, and every later check would compile
        against a scene the browser is no longer being shown.
        """
        snapshot = self._discovery
        if snapshot is None:
            return None, (
                "This run carries no discovery snapshot, so a check "
                "against it cannot reproduce the live one."
            )
        cached = self._context
        if cached is not None and cached[0] is snapshot:
            return cached[1], None

        from ..authoring import CheckContext

        try:
            context = CheckContext.from_discovery_model(snapshot.model)
        except Exception as exc:                    # noqa: BLE001
            return None, str(exc)
        if self._discovery is snapshot:
            self._context = (snapshot, context)
        return context, None

    @property
    def authored_dir(self) -> Path:
        """Where authored specifications go, and the only such place.

        Derived from the recording, never from the browser. There is no
        filename field on the wire, so a traversal attempt, an absolute
        path or a symlink cannot be expressed in the first place.
        """
        return self.recording_dir.parent / "authored"

    def _save_specs(self, context, entries, collision: str) -> dict:
        from ..authoring import (
            available_renamed_path,
            check_draft,
            normalise_name,
            spec_entry,
            write_specs_document,
        )
        from ..spec.semantic import validate_constants

        prepared = []
        seen = set()
        for raw in entries:
            if not isinstance(raw, dict):
                return {"ok": False, "error": "entry must be an object"}
            name = normalise_name(str(raw.get("name") or "").strip())
            if not name:
                return {"ok": False, "error": "every entry needs a name"}
            if name in seen:
                return {"ok": False,
                        "error": "duplicate specification name {!r}".format(name)}
            seen.add(name)
            formula = str(raw.get("formula") or "")
            try:
                constants = validate_constants(raw.get("constants") or {})
            except (TypeError, ValueError) as exc:
                return {"ok": False,
                        "error": "{!r}: {}".format(name, exc)}
            symmetric = bool(raw.get("symmetric"))
            backend = str(raw.get("backend") or "auto")
            # Re-checked here, immediately before writing: a client can
            # accumulate entries over a long session, and nothing
            # guarantees a verdict it recorded earlier still holds.
            verdict = check_draft(
                context, name=name, formula=formula,
                constants=constants, backend=backend, symmetric=symmetric,
            )
            if not verdict.accepted:
                return {"ok": False,
                        "error": "{!r} no longer compiles".format(name),
                        "report": verdict.report}
            prepared.append(spec_entry(
                name=name, formula=formula, constants=constants,
                backend=backend, symmetric=symmetric,
            ))

        directory = self.authored_dir
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "specs.yaml"
        # Same order the terminal writer uses: an existing file is
        # reported back and the choice is the author's, never a default
        # that silently destroys someone's specifications.
        overwrite = collision == "overwrite"
        if destination.exists() and collision == "rename":
            destination = Path(available_renamed_path(str(destination)))
        if destination.exists() and not overwrite:
            return {"ok": False, "collision": True,
                    "path": str(destination),
                    "error": "{} already exists".format(destination.name)}
        try:
            write_specs_document(
                prepared, str(destination), overwrite=overwrite
            )
        except FileExistsError:
            return {"ok": False, "collision": True,
                    "path": str(destination),
                    "error": "{} appeared before the write".format(
                        destination.name)}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(destination),
                "count": len(prepared)}

    # -- app ----------------------------------------------------------------

    def _create_app(self):
        app = Flask(
            __name__,
            static_folder=str(PACKAGE_ROOT / "static"),
        )
        app.config["SECRET_KEY"] = "specola-dashboard"
        # Flask's default static max-age is 12h; the client iterated
        # fast today and a stale cached dashboard.js looks like a broken
        # dashboard. ETag revalidation still gives 304s.
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
        socketio = SocketIO(
            app,
            async_mode="threading",
            # None = same-origin only (HRC's CORS posture); the stub's
            # str|list annotation is narrower than the runtime contract.
            cors_allowed_origins=None,  # pyright: ignore[reportArgumentType]
            manage_session=False,
            # The server shares its process with the simulator loop. A
            # saturated step phase (scene build, capture, encode) can
            # stall packet handling past the 20 s default while an
            # MB-class geometry payload is in flight; the default then
            # drops the client, whose reconnect re-requests the same
            # payload and the cycle repeats. 60 s rides out the stalls.
            ping_timeout=60,
        )

        # After SocketIO has wrapped app.wsgi_app, so this sits outside
        # its /socket.io/ interception rather than behind it.
        app.wsgi_app = LoopbackOnly(app.wsgi_app)

        @app.get("/api/runs")
        def runs():                     # pyright: ignore[reportUnusedFunction]
            """Recordings on disk, and what can honestly be said of them.

            Rooted at the results directory this recording sits in, so
            the listing is scoped to the tree the server was pointed at
            rather than to anywhere a browser might name.
            """
            from .catalogue import scan_results

            root = self.recording_dir.parent.parent
            records = scan_results(root)
            return {"root": str(root),
                    "runs": [record.to_wire() for record in records],
                    "current": str(self.recording_dir)}

        @app.get("/api/discovery")
        def discovery():                # pyright: ignore[reportUnusedFunction]
            """Serve the scene document, still compressed.

            Sent as stored rather than decoded. Measured on
            results/20260816_004207_robocasa_notask: 1.10 MiB gzipped,
            37.73 MiB decoded. Handing the browser the bytes with
            Content-Encoding lets it decompress once, in C, instead of
            this process decoding and re-encoding tens of megabytes per
            request. (Scene sizes vary a lot -- RoboCasa randomizes per
            reset, and discovery_capture.py records 55 MiB for
            PickPlaceCounterToCabinet -- which is why the recording is
            named here rather than the number quoted bare.)
            """
            from flask import Response

            if self._discovery is None:
                return ({"available": False,
                         "reason": "no discovery snapshot for this run"},
                        404)
            variant = request.args.get("variant", "semantic")
            if variant not in ("semantic", "raw"):
                return ({"error": "unknown variant"}, 400)
            try:
                payload = self._discovery.gzip_for(variant)
            except Exception as exc:                # noqa: BLE001
                # A document that gzipped and parsed can still fail to
                # restore. That is this recording's permanent condition,
                # not a server fault, so it is reported once with its
                # reason rather than as a 500 the page retries forever.
                return ({"available": False,
                         "reason": "{}: {}".format(
                             type(exc).__name__, exc)}, 409)
            return Response(
                payload,
                mimetype="application/json",
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(payload)),
                    "X-Specola-Discovery-Digest": self._discovery.digest,
                    "Cache-Control": "no-store",
                },
            )

        @app.get("/")
        def index():                    # pyright: ignore[reportUnusedFunction]
            return app.send_static_file("index.html")

        @socketio.on("connect")
        def connected():                # pyright: ignore[reportUnusedFunction]
            sid = request.sid

            def sender(payload: dict, callback) -> None:
                with self._emit_lock:
                    socketio.emit("step_bundle", payload, to=sid,
                                  callback=callback)

            state = _ClientState(
                sid, sender, follow=(self.phase == "live"))
            with self._clients_lock:
                self._clients[sid] = state
            self._emit("run_meta", self._run_meta(), sid=sid)
            self._emit("discovery_ready", self._discovery_ready(), sid=sid)
            self._send_geometry(state)

        @socketio.on("disconnect")
        def disconnected():             # pyright: ignore[reportUnusedFunction]
            with self._clients_lock:
                state = self._clients.pop(request.sid, None)
            if state is not None:
                state.playing = False
                state.play_generation += 1

        @socketio.on("view")
        def view(payload):              # pyright: ignore[reportUnusedFunction]
            with self._clients_lock:
                state = self._clients.get(request.sid)
            if state is None or not isinstance(payload, dict):
                return {"ok": False, "error": "no client state"}
            try:
                with state.lock:
                    result, start_playback = self._apply_view(state, payload)
            except (KeyError, TypeError, ValueError) as error:
                return {"ok": False, "error": str(error)}
            if start_playback:
                socketio.start_background_task(
                    self._playback_loop, state, state.play_generation)
            return result

        @socketio.on("selector_palette")
        def selector_palette_request(payload):  # pyright: ignore[reportUnusedFunction]
            """Resolve a browser selection against the Python model.

            The browser sends names. Names are looked up in the
            immutable snapshot and handed to the same
            ``selector_palette()`` the terminal calls, so the two
            surfaces cannot offer different selectors for one
            selection -- and nothing the browser sends reaches a file
            path, a formula, or the adapter.
            """
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            request_id = payload.get("request_id")
            if self._discovery is None:
                return selector_palette_result(
                    request_id, forms=[],
                    reason="no discovery snapshot for this run")
            try:
                return self._resolve_palette(request_id, payload)
            except Exception as error:              # noqa: BLE001
                # Includes DiscoveryRestoreError: an unusable snapshot
                # costs Discover its answer, never the connection.
                return selector_palette_result(
                    request_id, forms=[], reason=str(error))

        @socketio.on("signal_rows")
        def signal_rows(payload):       # pyright: ignore[reportUnusedFunction]
            """Aggregate one scope's signals, so only Python ever does.

            The page renders what this returns and sends a pattern from
            it back; it does not derive one. Two implementations of the
            condensation rule is how a page came to ask about
            ``xpos_body_*`` while Python called that same family
            ``xpos_body_*_0``.
            """
            from .protocol import signal_rows_result

            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            request_id = payload.get("request_id")
            if self._discovery is None:
                return signal_rows_result(
                    request_id, signals=[],
                    reason="no discovery snapshot for this run")
            try:
                return self._resolve_signals(request_id, payload)
            except Exception as error:              # noqa: BLE001
                return signal_rows_result(
                    request_id, signals=[], reason=str(error))

        @socketio.on("spec_check")
        def spec_check(payload):        # pyright: ignore[reportUnusedFunction]
            """Compile one draft against this run's own scene.

            Every reply echoes the revision it was asked about. The
            browser edits while a check is in flight, so a reply that
            names an older revision must not be allowed to authorize a
            Keep -- the terminal writer has always enforced that by
            comparing draft and validated revisions, and a second
            authoring surface that forgot to would accept a draft nobody
            validated.
            """
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            revision = payload.get("revision")
            context, error = self._check_context()
            if context is None:
                return {"ok": False, "revision": revision,
                        "accepted": False, "error": error}
            from ..authoring import check_draft, normalise_name, parse_constants

            raw_constants = payload.get("constants") or ""
            try:
                constants = (
                    parse_constants(raw_constants)
                    if isinstance(raw_constants, str)
                    else _validated_constants(raw_constants)
                )
            except (TypeError, ValueError) as exc:
                return {"ok": True, "revision": revision, "accepted": False,
                        "report": str(exc)}
            name = normalise_name(str(payload.get("name") or "").strip())
            if not name:
                return {"ok": True, "revision": revision, "accepted": False,
                        "report": "Specification name is required."}
            result = check_draft(
                context,
                name=name,
                formula=str(payload.get("formula") or ""),
                constants=constants,
                backend=str(payload.get("backend") or "auto"),
                symmetric=bool(payload.get("symmetric")),
            )
            from ..authoring import spec_entry

            return {
                "ok": True, "revision": revision,
                "digest": self._discovery.digest,
                "accepted": result.accepted,
                "report": result.report or (result.error or ""),
                # The accepted entry travels back already normalized, so
                # the page keeps what the server parsed rather than
                # re-deriving constants from the text a second time and
                # sending something the checker never saw.
                "entry": spec_entry(
                    name=name, formula=str(payload.get("formula") or ""),
                    constants=constants,
                    backend=str(payload.get("backend") or "auto"),
                    symmetric=bool(payload.get("symmetric")),
                ) if result.accepted else None,
            }

        @socketio.on("spec_save")
        def spec_save(payload):         # pyright: ignore[reportUnusedFunction]
            """Write kept specifications beside this run.

            The browser sends entries and a collision choice. It does not
            send a path, and there is nowhere for it to put one: the
            destination is derived from the recording this server was
            started on, so a traversal attempt has nothing to traverse.

            Every entry is re-checked here immediately before writing.
            A client can accumulate entries across a long session while
            nothing guarantees its earlier verdicts still hold.
            """
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            entries = payload.get("entries")
            if not isinstance(entries, list) or not entries:
                return {"ok": False, "error": "nothing to save"}
            context, error = self._check_context()
            if context is None:
                return {"ok": False, "error": error}
            return self._save_specs(context, entries,
                                    str(payload.get("collision") or "rename"))

        @socketio.on("stop_run")
        def stop_run(payload=None):     # pyright: ignore[reportUnusedFunction]
            if self.phase != "live":
                return {"ok": False, "error": "no live run to stop"}
            self.stop_requested.set()
            for callback in list(self._stop_callbacks):
                try:
                    callback()
                except Exception:
                    pass
            return {"ok": True}

        @socketio.on("series_window")
        def series_window(payload):     # pyright: ignore[reportUnusedFunction]
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            try:
                start = int(payload.get("start", 0))
                end = int(payload.get("end", self.index.max_step))
                px = max(1, min(int(payload.get("px", 1024)), 4096))
                requested = payload.get("bindings") or []
                series = []
                for entry in requested[:256]:
                    spec_name, key = str(entry[0]), str(entry[1])
                    stored = self.index.series.get((spec_name, key))
                    if stored is None:
                        continue
                    steps, values = stored
                    series.append(
                        [spec_name, key,
                         decimate(steps, values, start, end, px)])
            except (KeyError, TypeError, ValueError, IndexError) as error:
                return {"ok": False, "error": str(error)}
            return {"ok": True, "series": series}

        @socketio.on("transitions_page")
        def transitions_page(payload):  # pyright: ignore[reportUnusedFunction]
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            try:
                offset = max(0, int(payload.get("offset", 0)))
                limit = max(1, min(int(payload.get("limit", 200)), 500))
            except (TypeError, ValueError) as error:
                return {"ok": False, "error": str(error)}
            entries = self.index.transitions[offset:offset + limit]
            return {"ok": True, "total": len(self.index.transitions),
                    "offset": offset, "entries": entries}

        return app, socketio
