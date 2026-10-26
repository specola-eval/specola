"""``discover --web`` serves a scene that has no run behind it.

Two contracts meet here and neither may give way. ``--format json``
writes exactly one document to stdout, which is what makes it pipeable;
and a served dashboard announces a URL. The announcement goes to stderr,
and so does everything Flask and werkzeug print from the serving thread
— a banner appended after a complete document turns one valid document
into two things, which is how this was found.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("flask_socketio")

SPECOLA = [sys.executable, "-m", "specola.cli"]


def _run_web(tmp_path, *extra, wait=9.0):
    process = subprocess.Popen(
        SPECOLA + ["discover", "--sim", "mock", "--web", "0", *extra],
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + wait
    err_lines = []
    url = None
    # Read until both announcements have appeared rather than assuming
    # they are adjacent: werkzeug's own warnings are redirected to the
    # same stream and interleave with them.
    while time.monotonic() < deadline:
        line = process.stderr.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        err_lines.append(line)
        if url is None and "http://127.0.0.1:" in line:
            url = next(
                (part for part in line.split() if part.startswith("http")),
                None,
            )
        if url is not None and any(
            text.startswith("Scene:") for text in err_lines
        ):
            break
    return process, url, "".join(err_lines)


def test_the_url_is_announced_on_stderr(tmp_path):
    process, url, stderr = _run_web(tmp_path)
    try:
        assert url, stderr
        assert "Discover:" in stderr
        assert "Scene:" in stderr
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_json_stdout_stays_one_document_while_serving(tmp_path):
    process, url, stderr = _run_web(tmp_path, "--format", "json")
    try:
        assert url, stderr
        # Read what has been written so far; the document is flushed
        # before anything is served, so it is complete by now.
        import os

        os.set_blocking(process.stdout.fileno(), False)
        payload = process.stdout.read() or ""
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert payload, "no document on stdout"
    assert "Serving Flask" not in payload
    document = json.loads(payload)
    assert document["simulator"] == "mock"
    assert document["entities"]


def test_a_scene_directory_is_created_for_authoring(tmp_path):
    process, url, stderr = _run_web(tmp_path)
    try:
        assert url, stderr
        sessions = list((tmp_path / "results").glob("discover_mock_*"))
        assert sessions, "no synthetic discover session directory"
        assert (sessions[0] / "recording_000" / "manifest.json").is_file()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_textual_and_web_are_refused_together(tmp_path):
    result = subprocess.run(
        SPECOLA + ["discover", "--sim", "mock", "--web", "--ui", "textual"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "cannot be combined" in (result.stderr + result.stdout)
