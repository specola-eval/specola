"""One episode's simulator states, one row per step, for ``--mode replay``.

The container is a NumPy ``.npz`` with two members: ``header`` (a JSON
string) and ``states`` (float64, shape ``(steps, length)``). Row 0 is
the state of the episode's reset; row k is the state the k-th
observation described. ``.npz`` rather than HDF5 because NumPy is a
core dependency and h5py is not (argued 2026-10-13 in the plan that
introduced this module; confirmation pending there). Rows are spooled
to a raw file while recording and read back once at close, so the
peak memory cost is the recording's own size, 8 bytes per state element
per step. Replay never re-steps physics: it restores row k and
observes, so what a recording can reproduce is exactly what a restored
state determines (positions, velocities, distances), and the per-adapter
evidence pages say which keys that leaves out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

FORMAT_VERSION = 1
SUFFIX = ".npz"
_MISSING = object()


class StateRecordingError(ValueError):
    """A recording that cannot be written, read or replayed as asked."""


class StateWriter:
    def __init__(self, path, *, simulator: str, task: Optional[str], signature: dict, dt: float) -> None:
        # The reader refuses a recording whose signature lacks a positive
        # integer ``length``, so refuse to write one: an adapter author who
        # omits the key learns it here, at startup, not from every replay.
        if not isinstance(signature, dict):
            raise StateRecordingError(f"state signature must be a dict, got {type(signature).__name__}")
        length = signature.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise StateRecordingError("state signature must carry a positive integer 'length'")
        self.path = Path(path)
        self._header = {
            "format_version": FORMAT_VERSION, "simulator": simulator, "task": task,
            "signature": signature, "dt": float(dt), "steps": 0, "status": "recording",
        }
        self._spool_path = self.path.with_suffix(".rows")
        self._spool: Optional[BinaryIO] = None
        self._width: Optional[int] = None
        self._steps = 0
        self._closed: Optional[Path] = None

    def append(self, state: np.ndarray) -> None:
        if self._closed is not None:
            raise StateRecordingError("recording already closed")
        # An owned copy: adapters may hand back the same buffer every step.
        row = np.array(state, dtype=np.float64, copy=True)
        if row.ndim != 1:
            raise StateRecordingError(f"a state is a 1-D array, got shape {row.shape}")
        if not np.isfinite(row).all():
            raise StateRecordingError("a state holds a non-finite value")
        if self._spool is None:
            declared = self._header["signature"].get("length")
            if declared is not None and declared != row.shape[0]:
                raise StateRecordingError(
                    f"state length {row.shape[0]} differs from the signature's {declared}"
                )
            self._width = row.shape[0]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._spool = open(self._spool_path, "wb")
        elif row.shape[0] != self._width:
            raise StateRecordingError(
                f"state length {row.shape[0]} differs from the first row's {self._width}"
            )
        self._spool.write(row.tobytes())
        self._steps += 1

    def close(self, status: str) -> Path:
        if self._closed is not None:
            return self._closed
        self._header["steps"] = self._steps
        self._header["status"] = status
        if self._spool is not None:
            self._spool.close()
            states = np.fromfile(self._spool_path, dtype=np.float64).reshape(self._steps, self._width)
        else:
            states = np.zeros((0, 0))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, header=json.dumps(self._header, sort_keys=True), states=states)
        if self._spool_path.exists():
            self._spool_path.unlink()
        self._closed = self.path
        return self.path


class StateRecording:
    def __init__(self, header: dict, states: np.ndarray) -> None:
        self.header = header
        self.states = states

    @classmethod
    def load(cls, path) -> "StateRecording":
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                header = json.loads(str(archive["header"]))
                states = np.asarray(archive["states"], dtype=np.float64)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            raise StateRecordingError(f"{path} is not a readable state recording: {exc}")
        if not isinstance(header, dict):
            raise StateRecordingError(f"{path}: header is not an object")
        if header.get("format_version") != FORMAT_VERSION or isinstance(header.get("format_version"), bool):
            raise StateRecordingError(
                f"{path}: format_version {header.get('format_version')!r}, this build reads {FORMAT_VERSION}"
            )
        for key, kind in (("simulator", str), ("task", (str, type(None))), ("signature", dict),
                          ("dt", (int, float)), ("steps", int), ("status", str)):
            value = header.get(key, _MISSING)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise StateRecordingError(f"{path}: header field {key} missing or of the wrong type")
        if not (math.isfinite(header["dt"]) and header["dt"] > 0):
            raise StateRecordingError(f"{path}: dt must be a positive finite number")
        length = header["signature"].get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise StateRecordingError(f"{path}: signature length must be a positive integer")
        if not np.isfinite(states).all():
            raise StateRecordingError(f"{path}: states hold a non-finite value")
        if states.ndim != 2 or states.shape[0] != header["steps"]:
            raise StateRecordingError(
                f"{path}: states has shape {states.shape}, header says {header['steps']} rows"
            )
        if states.shape[0] == 0:
            raise StateRecordingError(f"{path}: no rows; a replayable recording holds at least the reset state")
        if length != states.shape[1]:
            raise StateRecordingError(
                f"{path}: rows are {states.shape[1]} wide, the signature says {length}"
            )
        return cls(header, states)


def is_state_recording(path) -> bool:
    """Whether ``path`` is a Specola state recording.

    False for any other suffix (the robomimic reader takes those). A
    ``.npz`` that lacks the two members is an error, not a fall-through:
    the caller asked for a state recording and handed over something else.
    """
    path = Path(path)
    if path.suffix != SUFFIX:
        return False
    if not path.is_file():
        raise StateRecordingError(f"{path} does not exist")
    try:
        with np.load(path, allow_pickle=False) as archive:
            members = set(archive.files)
    except (OSError, ValueError, EOFError) as exc:
        raise StateRecordingError(f"{path} is not a state recording: {exc}")
    if {"header", "states"} - members:
        raise StateRecordingError(f"{path} is not a state recording: members {sorted(members)}")
    return True


def check_compatible(recording: StateRecording, *, simulator: str, task: Optional[str], signature: dict, dt: float) -> None:
    header = recording.header
    # Identity first: a recording from another simulator with another dt
    # must not be answered with advice to change --dt.
    if header.get("simulator") != simulator:
        raise StateRecordingError(f"recorded on simulator {header.get('simulator')!r}, this run is {simulator!r}")
    if header.get("task") != task:
        raise StateRecordingError(f"recorded on task {header.get('task')!r}, this run is {task!r}")
    if header.get("signature") != signature:
        raise StateRecordingError(
            "state signature differs: the recording was made on another model or engine build "
            f"(recorded {json.dumps(header.get('signature'), sort_keys=True)}, live {json.dumps(signature, sort_keys=True)})"
        )
    if not math.isclose(float(header["dt"]), float(dt), rel_tol=1e-9, abs_tol=0.0):
        raise StateRecordingError(
            f"recorded at dt {header['dt']}, this run monitors at {dt}; pass --dt {header['dt']} to match"
        )
