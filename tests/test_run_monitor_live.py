"""Run-monitor display-mode integration (non-TTY fallback and PTY mode)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Integration: non-TTY fallback (subprocess) has no cursor codes
# ---------------------------------------------------------------------------

def run_cli(*extra):
    return subprocess.run(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(REPO / "examples/mock_specs.yaml"),
         "--steps", "30", *extra],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )


def test_non_tty_output_has_no_cursor_codes():
    proc = run_cli()
    assert proc.returncode == 0, proc.stderr
    assert "\x1b[F" not in proc.stdout
    assert "\x1b[2K" not in proc.stdout
    # Plain mode is transition-only by default; the durable trace carries
    # per-step snapshots.
    assert "snapshot every" not in proc.stdout
    assert "Episode summary" in proc.stdout


# ---------------------------------------------------------------------------
# Issue #102: --web implies a summary-only terminal unless --ui/
# --summary-only/--no-live were passed explicitly.
# ---------------------------------------------------------------------------

_TRANSITION_LINE = re.compile(r"step \d")


def _args(**overrides):
    base = dict(web=None, ui=None, summary_only=False, no_live=False,
                render=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    ("case", "kwargs", "compatible_tty", "web_active", "expected"),
    (
        # (a) --web genuinely serving, no --ui: summary-only plain, on
        # a TTY (would otherwise be Textual) and off it (already plain).
        ("web serving, tty", dict(web=8090), True, True,
         ("plain", True, True)),
        ("web serving, non-tty", dict(web=8090), False, True,
         ("plain", True, True)),
        ("web serving, port 0, tty", dict(web=0), True, True,
         ("plain", True, True)),
        # A --web that never came up (bad port, missing extra) must NOT
        # claim a browser viewport it doesn't have: falls back to
        # today's normal default, same as if --web were absent.
        ("web requested but failed to start, tty", dict(web=8090), True,
         False, ("textual", False, False)),
        ("web requested but failed to start, non-tty", dict(web=8090),
         False, False, ("plain", False, False)),
        # (b)/(c) explicit --ui always wins over the --web default, even
        # when the dashboard is genuinely serving.
        ("web serving + explicit textual", dict(web=8090, ui="textual"),
         True, True, ("textual", False, False)),
        ("web serving + explicit rich", dict(web=8090, ui="rich"),
         True, True, ("rich", False, False)),
        ("web serving + explicit plain", dict(web=8090, ui="plain"),
         True, True, ("plain", False, False)),
        # explicit --summary-only/--no-live under --web also win as
        # given, whether or not the dashboard is serving.
        ("web serving + explicit summary-only",
         dict(web=8090, summary_only=True), True, True,
         ("textual", True, False)),
        ("web serving + explicit no-live", dict(web=8090, no_live=True),
         True, True, ("plain", False, False)),
        # (d) --summary-only alone (no --web): unchanged, does not
        # touch UI selection.
        ("summary-only, no web, tty", dict(summary_only=True), True,
         False, ("textual", True, False)),
        ("summary-only, no web, non-tty", dict(summary_only=True), False,
         False, ("plain", True, False)),
        # (e) bare run: today's TTY-conditional default, untouched.
        ("bare, tty", dict(), True, False, ("textual", False, False)),
        ("bare, non-tty", dict(), False, False, ("plain", False, False)),
        ("bare + render, tty", dict(render=True), True, False,
         ("rich", False, False)),
        # --web genuinely serving + --render still forces plain (the
        # browser is the viewport; --render's own Rich preference does
        # not apply) -- but a --web that failed to start lets --render's
        # normal Rich preference resume exactly as if --web were absent.
        ("web serving + render, tty", dict(web=8090, render=True), True,
         True, ("plain", True, True)),
        ("web failed + render, tty", dict(web=8090, render=True), True,
         False, ("rich", False, False)),
    ),
)
def test_resolve_ui_and_summary(
    case, kwargs, compatible_tty, web_active, expected
):
    from specola.run_monitor import _resolve_ui_and_summary

    assert _resolve_ui_and_summary(
        _args(**kwargs), compatible_tty, web_active
    ) == expected, case


needs_web = pytest.mark.skipif(
    importlib.util.find_spec("flask_socketio") is None,
    reason="needs a genuinely serving dashboard: the web extra "
           "(flask_socketio)",
)


def _run_cli_web(*extra, timeout=20):
    """Like run_cli, but safe when --web is passed.

    A --web dashboard that actually comes up keeps `specola run`
    serving after the episode ends until Ctrl-C (run_monitor.main's
    `Event().wait()`), so a plain subprocess.run with a timeout would
    just hang until it kills the process. This waits for a natural
    exit first (what happens whenever the dashboard never came up --
    the common case in this repo's sim-free env, which lacks the web
    extra); if the process is still running once the mock episode must
    be done, it sends SIGINT the same way a user would to end the
    dashboard.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "specola.run_monitor",
         "--sim", "mock", "--specs", str(REPO / "examples/mock_specs.yaml"),
         "--steps", "30", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=timeout)
    return proc, out, err


def test_web_default_falls_back_when_dashboard_unavailable():
    """(a), failure path: a --web that never came up must not claim a
    browser viewport it doesn't have. Occupying the target port forces
    DashboardServer.start() to fail with OSError regardless of whether
    the web extra is installed, so this runs (and is meaningful) in any
    environment -- it is what's actually exercised in this repo's
    sim-free env, which lacks flask_socketio."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        proc = run_cli("--web", str(port))
    finally:
        blocker.close()
    assert proc.returncode == 0, proc.stderr
    assert "Web warning" in proc.stderr
    assert "Terminal UI is off" not in proc.stdout
    assert _TRANSITION_LINE.search(proc.stdout)
    assert "Episode summary" in proc.stdout


@needs_web
def test_web_default_prints_notice_when_dashboard_is_actually_serving():
    """(a), success path: once the browser dashboard genuinely comes
    up, the terminal shows the one-line notice, suppresses per-step
    transitions, and still prints the final summary."""
    proc, out, err = _run_cli_web("--web", "0")
    assert proc.returncode == 0, err
    assert "Web dashboard: " in out
    assert "Terminal UI is off" in out
    assert "--ui textual" in out
    assert not _TRANSITION_LINE.search(out)
    assert "Episode summary" in out


def test_web_with_explicit_ui_keeps_it_and_skips_notice():
    """(c): an explicit --ui under --web keeps exactly that UI -- no
    notice, transitions render normally -- whether or not the
    dashboard actually managed to start (see _run_cli_web)."""
    proc, out, err = _run_cli_web("--web", "0", "--ui", "rich")
    assert proc.returncode == 0, err
    assert "Terminal UI is off" not in out
    assert _TRANSITION_LINE.search(out)
    assert "Episode summary" in out


def test_web_video_rejects_an_unknown_encoder():
    """--web-video is argparse-validated: a typo fails at parse time,
    long before a run starts."""
    proc = run_cli("--web", "0", "--web-video", "vp9")
    assert proc.returncode == 2
    assert "--web-video" in proc.stderr
    assert "invalid choice" in proc.stderr


def test_web_video_requires_web():
    proc = run_cli("--web-video", "jpeg")
    assert proc.returncode != 0
    assert "require --web" in proc.stderr


def test_summary_only_without_web_is_unchanged():
    """(d): --summary-only alone still suppresses transitions and never
    prints the --web notice (it never saw --web)."""
    proc = run_cli("--summary-only")
    assert proc.returncode == 0, proc.stderr
    assert "Terminal UI is off" not in proc.stdout
    assert not _TRANSITION_LINE.search(proc.stdout)
    assert "Episode summary" in proc.stdout


def test_bare_run_is_unaffected():
    """(e): no --web, no --ui -- the highest-stakes case. Bare
    `specola run` keeps printing transitions and never shows the --web
    notice."""
    proc = run_cli()
    assert proc.returncode == 0, proc.stderr
    assert "Terminal UI is off" not in proc.stdout
    assert _TRANSITION_LINE.search(proc.stdout)
    assert "Episode summary" in proc.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="pty is POSIX/Linux")
def test_tty_live_mode_emits_cursor_codes_and_no_live_disables():
    import pty

    def run_in_pty(*extra):
        controller, follower = pty.openpty()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "specola.run_monitor",
                 "--sim", "mock",
                 "--specs", str(REPO / "examples/mock_specs.yaml"),
                 "--steps", "30", *extra],
                stdout=follower, stderr=subprocess.DEVNULL, cwd=REPO,
                env={**os.environ, "TERM": "xterm"},
            )
            os.close(follower)
            chunks = []
            while True:
                try:
                    data = os.read(controller, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            proc.wait(timeout=120)
            return b"".join(chunks).decode("utf-8", "replace")
        finally:
            os.close(controller)

    # stdout alone is not a compatible interactive terminal: stdin is not
    # attached to this PTY, so automatic selection remains plain.
    live = run_in_pty()
    assert "\x1b[F" not in live
    scrolling = run_in_pty("--no-live")
    assert "\x1b[F" not in scrolling


@pytest.mark.skipif(sys.platform != "linux", reason="pty is POSIX/Linux")
def test_render_tty_reports_automatic_rich_fallback(tmp_path):
    import pty

    # --render refuses adapters without a live viewer (issue #131), so the
    # run needs a mock that overrides render().
    (tmp_path / "viewer_adapter.py").write_text(
        "from specola.adapters.fake import _MockAdapter\n"
        "from specola.registry import register\n"
        "class ViewerAdapter(_MockAdapter):\n"
        "    def render(self):\n"
        "        pass\n"
        "register('viewer', lambda task, robot, target_only: ViewerAdapter())\n"
    )
    pythonpath = os.pathsep.join(
        filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")])
    )
    controller, follower = pty.openpty()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "specola.run_monitor",
             "--adapter-module", "viewer_adapter", "--sim", "viewer",
             "--specs", str(REPO / "examples/mock_specs.yaml"),
             "--steps", "1", "--render"],
            stdin=follower, stdout=follower, stderr=subprocess.PIPE,
            text=True, cwd=REPO,
            env={**os.environ, "TERM": "xterm", "PYTHONPATH": pythonpath},
        )
        os.close(follower)
        follower = -1
        while True:
            try:
                if not os.read(controller, 65536):
                    break
            except OSError:
                break
        stderr = proc.stderr.read()
        assert proc.wait(timeout=120) == 0, stderr
    finally:
        if follower >= 0:
            os.close(follower)
        os.close(controller)

    assert (
        "defaulting to the Rich terminal UI because --render keeps the "
        "simulator viewer on the main thread"
    ) in stderr


def test_rich_live_coalesces_updates_and_flushes_latest_on_exit(monkeypatch):
    import rich.live

    import specola.run_live as run_live
    from specola.run_state import RunSnapshot

    instances = []
    clock = [0.0]

    class SpyLive:
        def __init__(self, renderable, **kwargs):
            self.renderable = renderable
            self.auto_refresh = kwargs.get("auto_refresh", True)
            self.redirect_stdout = kwargs.get("redirect_stdout", True)
            self.redirect_stderr = kwargs.get("redirect_stderr", True)
            self.refresh_arguments = []
            self.final_renderable = None
            instances.append(self)

        def __enter__(self):
            return self

        def update(self, renderable, refresh=False):
            self.renderable = renderable
            self.refresh_arguments.append(refresh)

        def __exit__(self, exc_type, exc, traceback):
            self.final_renderable = self.renderable
            return False

    monkeypatch.setattr(rich.live, "Live", SpyLive)
    monkeypatch.setattr(run_live, "monotonic", lambda: clock[0])
    renderer = run_live.RichRunLive(refresh_rate=10.0)

    with renderer:
        for step in range(100):
            if step == 50:
                clock[0] = 0.1
            renderer.update(RunSnapshot(
                step=step,
                simulator_time=step * 0.1,
                specifications=(),
                transitions=(),
            ))

    live = instances[0]
    assert live.auto_refresh is False
    assert live.redirect_stdout is True
    assert live.redirect_stderr is True
    assert [
        index for index, refresh in enumerate(live.refresh_arguments)
        if refresh
    ] == [0, 50]
    assert live.final_renderable is not None
    assert "step=99" in live.final_renderable.renderable


def test_terminal_ui_pty_harness_records_transport_denominators():
    module_spec = importlib.util.spec_from_file_location(
        "specola_terminal_ui_pty",
        REPO / "benchmarks" / "terminal_ui_pty.py",
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    metrics, transcript = module.measure(
        [sys.executable, "-c", 'print("one"); print("two")'],
        cwd=REPO,
        rows=12,
        columns=40,
        timeout=5.0,
        environment=os.environ,
    )

    assert metrics["exit_code"] == 0
    assert metrics["timed_out"] is False
    assert metrics["terminal"] == {"rows": 12, "columns": 40}
    assert metrics["terminal_bytes"] == len(transcript)
    assert metrics["newline_bytes"] == 2
    assert metrics["pty_read_chunks"] >= 1
    assert "not displayed-row" in metrics["measurement_scope"][
        "newline_bytes"
    ]

    descendant_command = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(2)']); "
        "print('parent complete')"
    )
    started = time.monotonic()
    descendant_metrics, descendant_transcript = module.measure(
        [sys.executable, "-c", descendant_command],
        cwd=REPO,
        rows=12,
        columns=40,
        timeout=5.0,
        environment=os.environ,
    )
    assert descendant_metrics["exit_code"] == 0
    assert time.monotonic() - started < 1.5
    assert b"parent complete" in descendant_transcript
