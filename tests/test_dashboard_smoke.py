"""End-to-end smoke: mock run with --web serves, records, and replays.

The child's stdout+stderr go straight to a log file (no pipe: nothing
to deadlock, nothing buffered by the test); waits poll the file.
Assertions quote only the decisive lines.
"""

from __future__ import annotations

import importlib.util
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from specola.dashboard.recording import RecordingReader

# Gates the tests that spawn a server, not the whole module.
# RecordingReader imports no flask and the URL-parse test is pure string
# work, so a module-level importorskip would silently skip that
# regression test on the documented sim-free install -- the one place
# nobody would notice it had stopped running.
needs_web = pytest.mark.skipif(
    importlib.util.find_spec("flask_socketio") is None,
    reason="dashboard server tests need the web extra: flask_socketio",
)

SPECS = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"


def _wait_for_line(proc, log_path, needle, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(errors="replace")
        for line in text.splitlines():
            if needle in line:
                return line
        if proc.poll() is not None and needle not in text:
            break
        time.sleep(0.2)
    proc.kill()
    raise AssertionError(
        "never saw {!r}; full output in {}".format(needle, log_path))


def _url_from(line):
    """Pull the URL out of an announcement line.

    _spawn merges the child's stderr into the same log, and Werkzeug's
    production warning is written without a leading newline, so it
    splices straight onto the end of "Web dashboard: http://host:port".
    Taking the rest of the line (or up to the next space) then hands
    urlopen a port like "38435Werkzeug".
    """
    match = re.search(r"http://[^\s:]+:\d+", line)
    assert match, "no URL in {!r}".format(line)
    return match.group(0)


def _spawn(tmp_path, log_name, arguments):
    log_path = tmp_path / log_name
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        # No -u: a script reading the address from a pipe or file sees the
        # same buffered stdout, so the announcement lines must flush.
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(SPECS),
         "--output-dir", str(tmp_path / "results")] + arguments,
        stdout=log, stderr=subprocess.STDOUT)
    return proc, log_path, log


def test_url_survives_a_spliced_stderr_warning():
    # The exact line shape that failed once under suite load. Deterministic
    # because the splice itself is not -- the smoke tests below pass on
    # either parse, most of the time.
    spliced = (
        "  Web dashboard: http://127.0.0.1:38435Werkzeug appears to be "
        "used in a production deployment. Consider switching to a "
        "production web server instead."
    )
    assert _url_from(spliced) == "http://127.0.0.1:38435"
    assert _url_from("  Web dashboard: http://127.0.0.1:38435") == (
        "http://127.0.0.1:38435")
    assert _url_from("Serving foo at http://0.0.0.0:5000 now") == (
        "http://0.0.0.0:5000")


@needs_web
def test_mock_run_with_web_serves_and_records(tmp_path):
    proc, log_path, log = _spawn(
        tmp_path, "run.log",
        ["--steps", "30", "--ui", "plain", "--web", "0"])
    try:
        line = _wait_for_line(proc, log_path, "Web dashboard: ", timeout=60)
        url = _url_from(line)
        _wait_for_line(proc, log_path, "still serving", timeout=120)
        page = urllib.request.urlopen(url, timeout=10).read()
        assert b"specola" in page.lower()
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=15)
        log.close()
    assert proc.returncode == 0
    recording = next((tmp_path / "results").glob("*/recording_000"))
    reader = RecordingReader(recording)
    assert reader.manifest["status"] == "completed"
    assert reader.num_steps() == 30


@needs_web
def test_view_serves_finished_recording(tmp_path):
    proc, log_path, log = _spawn(
        tmp_path, "record.log",
        ["--steps", "10", "--ui", "plain", "--web", "0"])
    try:
        _wait_for_line(proc, log_path, "still serving", timeout=120)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=15)
        log.close()
    recording = next((tmp_path / "results").glob("*/recording_000"))

    view_log_path = tmp_path / "view.log"
    view_log = open(view_log_path, "w", encoding="utf-8")
    view = subprocess.Popen(
        [sys.executable, "-m", "specola.cli", "view",
         str(recording), "--port", "0"],
        stdout=view_log, stderr=subprocess.STDOUT)
    try:
        # specola's own announcement is "Serving <recording> at <url>";
        # Flask's " * Serving Flask app" banner can precede it, so key
        # on the fragment only the announcement contains.
        line = _wait_for_line(view, view_log_path, "at http", timeout=60)
        url = _url_from(line)
        page = urllib.request.urlopen(url, timeout=10).read()
        assert b"specola" in page.lower()
    finally:
        if view.poll() is None:
            view.send_signal(signal.SIGINT)
            view.wait(timeout=15)
        view_log.close()
    assert view.returncode == 0


# Runs the CLI behind a cyclic ballast whose finalizer reports. The
# ballast stands in for a served recording's parsed lines (30 M objects on
# RoboCasa): its finalizer running means exit garbage-collected the heap
# alive at Ctrl-C, which held RoboCasa view 40 s past SIGINT (issue #126).
# The mock heap is too small to show that as wall time.
_BALLAST_CLI = """
import os, runpy, sys

class Ballast:
    def __del__(self, write=os.write):
        write(2, b"BALLAST COLLECTED AT EXIT\\n")

ballast = Ballast()
ballast.cycle = ballast
sys.argv = ["specola"] + sys.argv[1:]
runpy.run_module("specola.cli", run_name="__main__")
"""


def _interrupt_with_client(command, log_path, ready):
    """Serve, attach a Socket.IO client, SIGINT the pid alone.

    Returns the exit code and the log; raises if the exit takes over 15 s.
    """
    import socketio

    with open(log_path, "w", encoding="utf-8") as log:
        # SIG_DFL: a `cmd &` from a non-interactive shell starts with SIGINT
        # ignored, and its children inherit that, so Python would never
        # raise KeyboardInterrupt; pin the disposition interactive Ctrl-C
        # has.
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _BALLAST_CLI] + command,
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        client = socketio.SimpleClient()
        try:
            url = _url_from(_wait_for_line(proc, log_path, ready, 120))
            urllib.request.urlopen(url, timeout=10).read()
            # With a long-polling client, an untimed main-thread wait lost
            # the SIGINT to another thread 9 times in 20; the 15 s bound
            # is what catches that.
            client.connect(url, transports=["polling"])
            assert client.receive(timeout=10)[0] == "run_meta"
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=15)
        finally:
            client.disconnect()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return proc.returncode, log_path.read_text(errors="replace")


@needs_web
def test_sigint_ends_served_run_and_view_promptly(tmp_path):
    pytest.importorskip("requests", reason="Socket.IO client polling")
    code, text = _interrupt_with_client(
        ["run", "--sim", "mock", "--specs", str(SPECS), "--steps", "10",
         "--ui", "plain", "--web", "0",
         "--output-dir", str(tmp_path / "results")],
        tmp_path / "run.log", "still serving")
    assert code == 0, text[-2000:]
    assert "Dashboard closed" in text
    assert "BALLAST COLLECTED AT EXIT" not in text
    recording = next((tmp_path / "results").glob("*/recording_000"))
    assert RecordingReader(recording).manifest["status"] == "completed"

    code, text = _interrupt_with_client(
        ["view", str(recording), "--port", "0"],
        tmp_path / "view.log", "at http")
    assert code == 0, text[-2000:]
    assert "BALLAST COLLECTED AT EXIT" not in text


@needs_web
def test_view_rejects_non_recording(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "specola.cli", "view", str(tmp_path)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "not a recording directory" in proc.stderr


@needs_web
def test_web_flags_require_web(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(SPECS),
         "--steps", "1", "--web-stride", "3",
         "--output-dir", str(tmp_path / "results")],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "require --web" in proc.stderr


@needs_web
def test_run_without_web_is_unaffected(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(SPECS),
         "--steps", "3", "--ui", "plain",
         "--output-dir", str(tmp_path / "results")],
        capture_output=True, text=True, timeout=120)
    (tmp_path / "plain.log").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0
    assert "Web" not in proc.stderr
    assert not list((tmp_path / "results").glob("*/recording_000"))
