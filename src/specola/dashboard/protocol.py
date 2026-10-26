"""Wire protocol: versioned envelope, step bundles, series decimation.

Epoch and seq are per-connection, not global (design decision: each
client owns its playback position, unlike HRC's shared one). The full
epoch/seq envelope applies to ``step_bundle`` — the only ordered,
position-bearing stream; ``run_meta`` and ``run_status`` are
connection-independent notifications and carry ``protocol_version`` +
``run_id`` only. Frame bytes stay raw ``bytes`` in the wire dict —
python-socketio packs them as binary attachments natively; base64
never appears on this wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

PROTOCOL_VERSION = 1


@dataclass
class Bundle:
    """One step's snapshot delta plus frames for the selected cameras.

    Satisfies LatestOnlyDelivery's contract (epoch, seq, to_wire).
    """

    run_id: str
    epoch: int
    seq: int
    line: dict                 # one snapshots.jsonl line
    # camera -> {"bytes": frame, "step": int, "codec": str}: the step is
    # the one the frame was sampled at, which for a hardware-encoded
    # access unit lags the bundle's own step; the codec is "jpeg" or the
    # stream's "avc1.PPCCLL" string.
    frames: dict
    max_step: int
    live: bool
    drops: dict                # {"frames": int, "delivery": int}
    poses: Optional[dict] = None   # {"bytes": record, "step": int} or None

    def to_wire(self) -> dict:
        wire = {"protocol_version": PROTOCOL_VERSION,
                "run_id": self.run_id,
                "epoch": self.epoch, "seq": self.seq,
                "step": self.line["step"],
                "sim_time": self.line["sim_time"],
                "wall_time": self.line["wall_time"],
                "specs": self.line["specs"],
                "transitions": self.line["transitions"],
                "new_specs": self.line.get("new_specs", []),
                "new_bindings": self.line.get("new_bindings", []),
                "frames": self.frames,
                "max_step": self.max_step,
                "live": self.live,
                "drops": self.drops}
        if self.poses is not None:
            # Absent key, not a null: runs without a 3D scene keep the
            # exact wire shape older clients already parse.
            wire["poses"] = self.poses
        return wire


def run_meta(manifest: dict, *, mode: str, specs: list, bindings: list,
             max_step: int, view_command: str, min_step: int = 0) -> dict:
    return {"protocol_version": PROTOCOL_VERSION,
            "run_id": manifest.get("run_id"),
            "mode": mode,
            "manifest": manifest,
            "specs": specs,
            "bindings": bindings,
            "min_step": min_step,
            "max_step": max_step,
            "view_command": view_command}


def geometry(run_id: Optional[str], *, meta: dict, meshes: bytes) -> dict:
    """The static 3D scene: ``meta`` is geometry.json, ``meshes`` the
    vertex/face buffer it indexes. Sent once per connection, when the
    recording has a scene at all."""
    return {"protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "json": meta,
            "meshes": meshes}


def run_status(run_id: Optional[str], *, max_step: int, drops: dict) -> dict:
    return {"protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "max_step": max_step,
            "drops": drops}


def episode_end(run_id: Optional[str], *, status: str, max_step: int,
                view_command: str) -> dict:
    return {"protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "status": status,
            "max_step": max_step,
            "view_command": view_command}


def decimate(steps: Sequence[int], values: Sequence[Optional[float]],
             start: int, end: int, px: int) -> List[list]:
    """Min/max per pixel column over the [start, end] step window.

    Returns ``[[step, vmin, vmax], ...]`` where ``step`` is the first
    sampled step in the column; ``None`` values are skipped. When the
    window holds at most ``px`` points, the raw points come back as
    ``[step, v, v]``. ``steps`` must be non-decreasing — the recording
    reader and the server's series index are append-ordered, which
    guarantees it.
    """
    if end < start or px < 1:
        return []
    span = end - start + 1
    columns: dict = {}
    raw_count = 0
    for step, value in zip(steps, values):
        if value is None or step < start or step > end:
            continue
        raw_count += 1
        column = (step - start) * px // span
        entry = columns.get(column)
        if entry is None:
            columns[column] = [step, value, value]
        else:
            if value < entry[1]:
                entry[1] = value
            if value > entry[2]:
                entry[2] = value
    if raw_count <= px:
        return [[step, value, value]
                for step, value in zip(steps, values)
                if value is not None and start <= step <= end]
    return [columns[column] for column in sorted(columns)]


def discovery_ready(run_id, *, available: bool, digest=None,
                    reason=None) -> dict:
    """Announce whether this run's scene can be browsed and checked.

    Additive, and protocol_version stays 1: no existing event shape
    changes, and a client that ignores this keeps working exactly as it
    did. Carries the digest rather than the document — the document is
    fetched over HTTP with Content-Encoding, because attaching 1.66 MiB
    to a socket payload would put it through the same path as per-step
    bundles.
    """
    return {"protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "available": bool(available),
            "digest": digest,
            "reason": reason,
            "variants": ["semantic", "raw"]}


def selector_palette_result(request_id, *, forms, reason=None,
                            scope=None) -> dict:
    """One palette, resolved server-side against the immutable model.

    ``request_id`` travels back untouched so a client can drop a reply
    that arrived after it moved on. Rows carry the reference and the
    shape metadata; they carry no binding or monitor count, because the
    palette's rows are forms.
    """
    return {"protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "scope": scope,
            "reason": reason,
            "forms": forms}


def signal_rows_result(request_id, *, signals, reason=None,
                       scope=None) -> dict:
    """The signals in one scope, aggregated by Python and only by Python.

    The browser used to group and condense these itself, which meant two
    implementations of one rule. They disagreed: the JavaScript kept a
    shared prefix while ``_glob_for_keys`` keeps a shared prefix *and* a
    shared suffix, so ``xpos_body_a_0`` and ``xpos_body_b_0`` were named
    ``xpos_body_*`` in the page and ``xpos_body_*_0`` in Python. The
    pattern a page then sent back matched nothing, and a selection of one
    axis could be answered with the palette for the whole scope.

    Rows are aggregated once, here, and the pattern the page sends back
    is one this function produced.
    """
    return {"protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "scope": scope,
            "reason": reason,
            "signals": signals}
