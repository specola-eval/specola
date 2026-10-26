"""What runs exist on disk, and which of them is still going.

There is no registry and no daemon. A run's state is derived from files
it already writes, which is why closing the browser, killing the server
or losing the machine's network changes nothing about what can be
learned later: the recording is the record.

The derivation is deliberately cautious about one thing. A recording
whose heartbeat has gone quiet is reported as **unknown**, never as
dead. Three different situations produce silence and only one of them is
a dead run:

* the adapter is still loading — a RoboCasa scene build takes seconds,
  and the recording directory and server exist before it finishes;
* the recorder failed and disabled itself, which by design leaves
  monitoring, results and the episode running;
* the process really did die.

Calling all three "died" would report a healthy run as lost, so the
distinction is kept and the reason is shown rather than guessed at. A
same-host pid can corroborate, and only corroborate: a pid that is gone
proves the process ended, while a pid that answers proves nothing about
another machine.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# How long a recording may go without a new line before its state stops
# being "running". Generous on purpose: a scene build, a geometry export
# and a slow first step all happen before the first line is written, and
# calling that a death would be wrong far more often than right.
QUIET_AFTER_S = 30.0

RUNNING = "running"
UNKNOWN = "unknown"
FINISHED = "finished"


@dataclass(frozen=True)
class RunRecord:
    """One recording on disk, and what can honestly be said about it."""

    directory: str
    run_id: str
    simulator: Optional[str]
    task: Optional[str]
    mode: Optional[str]
    status: Optional[str]
    state: str
    detail: str
    quiet_for: Optional[float]
    host: Optional[str]
    pid: Optional[int]
    dashboard_url: Optional[str]
    has_discovery: bool
    modified: Optional[float]

    def to_wire(self) -> dict:
        return {
            "directory": self.directory,
            "run_id": self.run_id,
            "simulator": self.simulator,
            "task": self.task,
            "mode": self.mode,
            "status": self.status,
            "state": self.state,
            "detail": self.detail,
            "quiet_for": self.quiet_for,
            "host": self.host,
            "pid": self.pid,
            "dashboard_url": self.dashboard_url,
            "has_discovery": self.has_discovery,
            "modified": self.modified,
        }


def _pid_is_gone(host: Optional[str], pid: Optional[int]) -> bool:
    """Whether a same-host pid is definitely no longer running.

    Only ever used to *strengthen* a conclusion. A missing pid on this
    host proves the process ended; a live pid proves little, since the
    number could have been reused, and a pid on another host cannot be
    inspected at all.
    """
    if pid is None or host is None:
        return False
    try:
        import socket

        if host != socket.gethostname():
            return False
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return False


def describe_recording(directory) -> Optional[RunRecord]:
    """Read one recording directory, or None when it is not one."""
    path = Path(directory)
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None

    growing = path / "snapshots.jsonl"
    finished_lines = path / "snapshots.jsonl.gz"
    status = manifest.get("status")
    now = time.time()

    modified: Optional[float] = None
    quiet_for: Optional[float] = None
    for candidate in (growing, finished_lines, manifest_path):
        try:
            modified = candidate.stat().st_mtime
            break
        except OSError:
            continue
    if modified is not None:
        quiet_for = max(0.0, now - modified)

    host = manifest.get("host")
    pid = manifest.get("pid")

    if status:
        state, detail = FINISHED, str(status)
    elif not growing.exists():
        # No line file yet: the episode has not started writing. That is
        # ordinary during a slow adapter load.
        state, detail = UNKNOWN, "no snapshots written yet"
    elif quiet_for is not None and quiet_for <= QUIET_AFTER_S:
        state, detail = RUNNING, "last line {:.0f}s ago".format(quiet_for)
    elif _pid_is_gone(host, pid):
        state, detail = UNKNOWN, "process {} on this host has exited".format(pid)
    else:
        state = UNKNOWN
        detail = (
            "no line for {:.0f}s — still loading, recorder disabled, "
            "or ended without finalizing".format(quiet_for or 0.0)
        )

    return RunRecord(
        directory=str(path),
        run_id=str(manifest.get("run_id") or path.parent.name),
        simulator=manifest.get("simulator"),
        task=manifest.get("task"),
        mode=manifest.get("mode"),
        status=status,
        state=state,
        detail=detail,
        quiet_for=quiet_for,
        host=host,
        pid=pid,
        dashboard_url=manifest.get("dashboard_url"),
        has_discovery=bool(manifest.get("discovery")),
        modified=modified,
    )


def scan_results(root, *, limit: int = 200) -> List[RunRecord]:
    """Every recording under ``root``, newest first.

    Bounded, and the bound is reported by the caller rather than hidden:
    a results tree accumulates thousands of episodes over a campaign and
    listing all of them is neither fast nor useful.
    """
    base = Path(root)
    records: List[RunRecord] = []
    try:
        candidates = list(base.glob("*/recording_*"))
    except OSError:
        return records
    for candidate in candidates:
        record = describe_recording(candidate)
        if record is not None:
            records.append(record)
    # Ordered by the same timestamp the state is derived from, not by
    # the directory's own mtime: a directory is touched when a file is
    # created in it, not when one is written to, so the two disagree for
    # exactly the recordings that are still being appended.
    records.sort(key=lambda record: record.modified or 0.0, reverse=True)
    return records[:limit]
