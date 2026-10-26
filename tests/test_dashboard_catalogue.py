"""Silence is not death, and the listing must not say it is.

A recording stops being written for three different reasons and only
one of them is a dead run: the adapter may still be loading, the
recorder may have disabled itself while monitoring continues, or the
process may really have ended. Reporting all three as "died" would
declare healthy runs lost, so quiet recordings are reported as unknown
with the reason attached.
"""

from __future__ import annotations

import json
import os
import socket
import time

from specola.dashboard.catalogue import (
    FINISHED,
    QUIET_AFTER_S,
    RUNNING,
    UNKNOWN,
    describe_recording,
    scan_results,
)


def _recording(path, *, status=None, lines=True, gz=False, **manifest):
    path.mkdir(parents=True, exist_ok=True)
    document = {"run_id": path.parent.name, "status": status}
    document.update(manifest)
    (path / "manifest.json").write_text(json.dumps(document))
    if lines:
        (path / "snapshots.jsonl").write_text('{"step": 1}\n')
    if gz:
        (path / "snapshots.jsonl.gz").write_bytes(b"\x1f\x8b")
    return path


def _age(path, seconds):
    stamp = time.time() - seconds
    for name in ("snapshots.jsonl", "snapshots.jsonl.gz", "manifest.json"):
        target = path / name
        if target.exists():
            os.utime(target, (stamp, stamp))


def test_a_freshly_written_recording_is_running(tmp_path):
    record = describe_recording(_recording(tmp_path / "r" / "recording_000"))
    assert record.state == RUNNING
    assert "last line" in record.detail


def test_a_finalized_recording_reports_its_status(tmp_path):
    record = describe_recording(_recording(
        tmp_path / "r" / "recording_000", status="completed", gz=True,
    ))
    assert record.state == FINISHED
    assert record.detail == "completed"


def test_a_quiet_recording_is_unknown_not_dead(tmp_path):
    """The distinction this module exists for."""
    path = _recording(tmp_path / "r" / "recording_000")
    _age(path, QUIET_AFTER_S + 60)
    record = describe_recording(path)
    assert record.state == UNKNOWN
    assert record.state != "died"
    # The reason names all three possibilities rather than choosing one.
    assert "still loading" in record.detail
    assert "recorder disabled" in record.detail


def test_a_recording_with_no_lines_yet_is_unknown(tmp_path):
    """A slow adapter load looks exactly like this and is not a failure."""
    record = describe_recording(_recording(
        tmp_path / "r" / "recording_000", lines=False,
    ))
    assert record.state == UNKNOWN
    assert "not started" in record.detail or "yet" in record.detail


def test_a_dead_same_host_pid_sharpens_the_reason(tmp_path):
    """Corroboration only: it explains a silence, never creates one."""
    path = _recording(
        tmp_path / "r" / "recording_000",
        host=socket.gethostname(), pid=2 ** 22,
    )
    _age(path, QUIET_AFTER_S + 60)
    record = describe_recording(path)
    assert record.state == UNKNOWN
    assert "exited" in record.detail


def test_a_live_pid_does_not_by_itself_make_a_run_running(tmp_path):
    """A pid can be reused, and pids on other hosts cannot be inspected."""
    path = _recording(
        tmp_path / "r" / "recording_000",
        host=socket.gethostname(), pid=os.getpid(),
    )
    _age(path, QUIET_AFTER_S + 60)
    assert describe_recording(path).state == UNKNOWN


def test_a_pid_on_another_host_is_never_inspected(tmp_path):
    path = _recording(
        tmp_path / "r" / "recording_000",
        host="some-other-machine", pid=1,
    )
    _age(path, QUIET_AFTER_S + 60)
    record = describe_recording(path)
    assert record.state == UNKNOWN
    assert "exited" not in record.detail


def test_identity_and_discovery_presence_travel(tmp_path):
    record = describe_recording(_recording(
        tmp_path / "r" / "recording_000",
        simulator="mock", task="t", host="h", pid=5,
        dashboard_url="http://127.0.0.1:8090",
        discovery={"file": "discovery.json.gz"},
    ))
    assert record.simulator == "mock"
    assert record.dashboard_url == "http://127.0.0.1:8090"
    assert record.has_discovery is True
    assert record.to_wire()["state"] == record.state


def test_scanning_lists_recordings_newest_first(tmp_path):
    for index in range(3):
        path = _recording(tmp_path / "run{}".format(index) / "recording_000")
        _age(path, 100 - index * 10)
    records = scan_results(tmp_path)
    assert [record.run_id for record in records] == ["run2", "run1", "run0"]


def test_scanning_is_bounded(tmp_path):
    for index in range(8):
        _recording(tmp_path / "run{}".format(index) / "recording_000")
    assert len(scan_results(tmp_path, limit=3)) == 3


def test_a_directory_that_is_not_a_recording_is_skipped(tmp_path):
    (tmp_path / "junk" / "recording_000").mkdir(parents=True)
    assert scan_results(tmp_path) == []
    assert describe_recording(tmp_path / "junk" / "recording_000") is None
