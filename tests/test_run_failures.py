from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import logging
import sys

import pytest

from specola.run_failures import RunFailure, render_run_failure
from specola.run_monitor import _SimulatorOutput


def test_failure_render_preserves_phase_and_verbatim_error():
    failure = RunFailure(
        phase="simulator reset",
        adapter="libero",
        task="libero_object:0",
        error="OSError: [Errno 30] Read-only file system: 'scene.xml'",
        diagnostics_path="/tmp/run-error.log",
        result_path=None,
    )

    text = render_run_failure(failure)

    assert "Phase       simulator reset" in text
    assert "OSError: [Errno 30] Read-only file system: 'scene.xml'" in text
    assert "Results     none" in text


def test_simulator_output_is_mirrored_and_saved(tmp_path):
    stdout = StringIO()
    stderr = StringIO()
    capture = _SimulatorOutput(tmp_path / "simulator.log")
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with capture.capture():
                print("simulator notice")
                print("simulator warning", file=sys.stderr)
    finally:
        capture.close()

    assert stdout.getvalue() == "simulator notice\n"
    assert stderr.getvalue() == "simulator warning\n"
    assert (tmp_path / "simulator.log").read_text() == (
        "simulator notice\nsimulator warning\n"
    )


def test_disabled_simulator_capture_never_replaces_process_streams(tmp_path):
    stdout = StringIO()
    capture = _SimulatorOutput(tmp_path / "simulator.log")
    try:
        with redirect_stdout(stdout):
            process_stream = sys.stdout
            with capture.capture(enabled=False) as log_stream:
                assert sys.stdout is process_stream
                assert log_stream is capture.stream
                print("owned by Textual")
    finally:
        capture.close()

    assert stdout.getvalue() == "owned by Textual\n"


def test_log_promotion_keeps_existing_text_and_retained_streams(tmp_path):
    startup = tmp_path / "startup" / "simulator.log"
    final = tmp_path / "run" / "simulator.log"
    final.parent.mkdir()
    final.write_text("previous run\n")
    output = _SimulatorOutput(startup)
    retained_stream = output.stream
    handler = logging.StreamHandler(retained_stream)
    logger = logging.getLogger("test.startup.promotion")
    logger.addHandler(handler)
    try:
        with output.capture():
            print("startup stdout")
            print("startup stderr", file=sys.stderr)
            logger.warning("startup logger")
        output.promote(final)
        retained_stream.write("retained writer\n")
        logger.warning("run logger")
        output.promote(final)
    finally:
        logger.removeHandler(handler)
        handler.close()
        output.close()
    assert final.read_text() == (
        "previous run\nstartup stdout\nstartup stderr\nstartup logger\n"
        "retained writer\nrun logger\n"
    )
    assert startup.read_text() == "startup stdout\nstartup stderr\nstartup logger\n"
    assert output.closed
    output.flush()
    output.close()


def test_failed_log_promotion_retains_full_startup_stream(tmp_path, monkeypatch):
    from specola import run_monitor

    startup = tmp_path / "startup.log"
    final = tmp_path / "run" / "simulator.log"
    output = _SimulatorOutput(startup)
    retained_stream = output.stream

    def partial_copy(source, destination):
        destination.write(source.read(3))
        raise OSError("copy interrupted")

    monkeypatch.setattr(run_monitor.shutil, "copyfileobj", partial_copy)
    try:
        output.write("complete startup\n")
        with pytest.raises(OSError, match="copy interrupted"):
            output.promote(final)
        retained_stream.write("failure diagnostics\n")
        assert output.path == startup
    finally:
        output.close()
    assert startup.read_text() == "complete startup\nfailure diagnostics\n"
    assert final.read_text() == "com"
