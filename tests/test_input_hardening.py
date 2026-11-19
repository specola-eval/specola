"""Regression tests for the 2026-11-19 user-test review.

Each test names the finding it pins in
docs/reviews/2026-11-19-user-test-bugs.md.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from specola.adapters.fake import _MockAdapter
from specola.session import Monitor
from specola.spec import parser as spec_parser
from specola.spec.semantic import _compile_predicate

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _specs(tmp_path, formula):
    path = tmp_path / "specs.yaml"
    path.write_text("specs:\n- name: a\n  formula: '{}'\n".format(formula))
    return str(path)


# -- B1: formula text must not execute -------------------------------------

@pytest.mark.parametrize("expression", [
    '__import__("os").system("touch {marker}")',
    "().__class__.__mro__",
    "x.real",
    "x[0]",
    "[x for x in (1,)]",
    "(lambda: 1)()",
    '"text"',
    "x if x else 1",
    "x < 1",
    "open('{marker}', 'w')",
    "sqrt(x, evaluate=False)",
    "True",
])
def test_non_arithmetic_expressions_are_refused(tmp_path, expression):
    marker = tmp_path / "marker"
    with pytest.raises(ValueError):
        _compile_predicate(
            expression.format(marker=marker), {"x": ((("object"), 1), "mass")}, {},
        )
    assert not marker.exists()


def test_injection_through_a_spec_file_runs_nothing(tmp_path):
    marker = tmp_path / "marker"
    specs = _specs(
        tmp_path,
        '__import__("os").system("touch {}") <= object_*[mass]'.format(marker),
    )
    monitor = Monitor(specs, results_dir=None)
    with pytest.raises(Exception):
        monitor.watch(_MockAdapter())
    assert not marker.exists()


@pytest.mark.parametrize("expression", [
    "x - 9**9**9**9",
    "9**(9**9 + pi - pi) - x",          # the name does not make it symbolic
    "2**floor(1e10) - x",
    "((9**1000)**1000)**1000 - x",      # each exponent is small
    "9**(999*x/x*999*999) - x",         # cancels to a constant exponent
    "(1/2)**(10**9) - x",               # exact rational, underflows as float
    "9**(sqrt(x - 100)*0 + 9**9) - x",
    # Small in double precision, unbounded in exact arithmetic.
    "x - 9**((10**17 + 1 - 10**17)*10**9)",
    "x - 9**(((10**17 + 1)/10**17 - 1)*10**26)",
    "x - 9**(10**9*floor((10**17 + 1)*sqrt(4)/2 - 10**17))",
    "9**((x*10**17 + 1 - x*10**17)*10**9)",
    "9**(10**9*(floor(sqrt(3)*sqrt(3)) - 2)) - x",
    "(9*x)**(10**9)",                   # distributes to 9**(10**9)
    "sqrt(2)**(10**9) - x",
    "(9**1000*x)**1000",
    "floor(exp(exp(20))) - x",          # rounding evaluates to full precision
    # Text that Python's float rounds away and sympy's Float keeps.
    "x - 9**(10**9 * floor(1e-400*1e200*1e200 + 0.5))",
    "x - 9**(10**9 * ceiling(1.0000000000000000001 - 1))",
    # A base of units only still expands exactly.
    "x - abs((1 + sqrt(-1))**(4*10**9))",
    # n is a declared constant: a symbol while parsing, 1e9 afterwards.
    "x - 9**floor(n)",
    "x - floor(n*exp(exp(20)))",
    "x - 9**floor(((n*10**11 + 1) - n*10**11)*n)",
])
def test_unbounded_powers_are_refused_before_sympy(expression):
    started = time.monotonic()
    with pytest.raises(ValueError):
        _compile_predicate(
            expression, {"x": (("object", 1), "mass")}, {"n": 1e9},
        )
    assert time.monotonic() - started < 5.0


def test_documented_arithmetic_still_compiles():
    symbols = {"x": (("object", 1), "mass"), "y": (("object", 2), "mass")}
    for expression in (
        "sqrt((x - y)**2 + 1e-3) - 0.5",
        "Max(abs(x), abs(y)) - limit",
        "-x**2 + 2**y + atan2(x, y) / pi",
        "x**-0.5",
        "x**(1/2) - 1",
        "sqrt(x - 100)**2 - 1",          # undefined over the reals at the sample
        "1/(1 + exp(-50*(x - 0.3))) - 0.5",
        "exp(1000*x) - 1",               # overflows at the sample, still valid
        "log(x) - log(y, 10)",
        "1/(x - 1.013) - 2",
        "(1000*x)**300 - 1",
        "x**limit + 9**limit",
        "floor(x*1e6) - 3",
    ):
        assert callable(_compile_predicate(expression, symbols, {"limit": 2.0}))


def test_unknown_name_reports_unknown_symbol():
    # 'limit' is a sympy function; undeclared, it used to surface as a
    # TypeError about subtracting a function.
    with pytest.raises(Exception, match="Unknown symbol"):
        _compile_predicate("x - limit", {"x": (("object", 1), "mass")}, {})


# -- B11: parser depth -------------------------------------------------------

def test_deep_formula_is_a_parse_error_not_a_recursion_error():
    with pytest.raises(ValueError, match="deeply nested"):
        spec_parser.parse("not (" * 3000 + "object_*[mass] <= 1" + ")" * 3000)


# -- B9/B10: facade arguments -------------------------------------------------

@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf"), "0.05", True])
def test_watch_rejects_bad_dt(tmp_path, dt):
    monitor = Monitor(_specs(tmp_path, "object_*[mass] <= 1"), results_dir=None)
    with pytest.raises((TypeError, ValueError), match="dt must be"):
        monitor.watch(_MockAdapter(), dt=dt)


@pytest.mark.parametrize("run_id", ["../../escaped", "/abs", "a/b", "..", " ", 3])
def test_run_id_must_be_one_path_component(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        Monitor(_specs(tmp_path, "object_*[mass] <= 1"), run_id=run_id)


# -- B2/B12: specola run ------------------------------------------------------

def _run(*args, **kwargs):
    # Unbuffered: the signal test reads the ready line from the pipe.
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, "-m", "specola.cli", "run", "--sim", "mock", *args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        **kwargs,
    )


def test_negative_steps_is_a_usage_error(tmp_path):
    process = _run("--specs", _specs(tmp_path, "object_*[mass] <= 1"),
                   "--steps", "-5", cwd=tmp_path)
    output, _ = process.communicate(timeout=120)
    assert process.returncode == 2
    assert "--steps" in output


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_terminated_run_writes_truncated_results(tmp_path, signum):
    process = _run(
        "--specs", _specs(tmp_path, "object_*[mass] <= 1"), "--steps", "0",
        "--ui", "plain", "--summary-only", "--run-name", "sig", cwd=tmp_path,
    )
    try:
        # The ready line is printed once the loop is about to start.
        for line in process.stdout:
            if "ready" in line:
                break
        time.sleep(1.0)
        process.send_signal(signum)
        process.communicate(timeout=180)
    finally:
        # --steps 0 never ends by itself; never leave it running.
        if process.poll() is None:
            process.kill()
    results = list((tmp_path / "results" / "sig").glob("*_ep0_*.meta.json"))
    assert results, "no episode result was written after " + signum.name
    assert '"truncated"' in results[0].read_text()


def test_terminal_hangup_still_writes_results(tmp_path):
    """An SSH drop closes the terminal and then sends SIGHUP: the shutdown
    must not die on its first print to the dead terminal."""
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        [sys.executable, "-m", "specola.cli", "run", "--sim", "mock",
         "--specs", _specs(tmp_path, "object_*[mass] <= 1"), "--steps", "0",
         "--ui", "plain", "--summary-only", "--run-name", "hup"],
        stdin=slave, stdout=slave, stderr=slave, env=env, cwd=tmp_path,
        start_new_session=True,
    )
    os.close(slave)
    try:
        seen = b""
        while b"ready" not in seen:
            seen += os.read(master, 4096)
        time.sleep(1.0)
        # The kernel does both at once for a controlling terminal; here
        # the signal goes first so the run cannot print in between.
        process.send_signal(signal.SIGHUP)
        os.close(master)
        process.wait(timeout=180)
    finally:
        if process.poll() is None:
            process.kill()
    results = list((tmp_path / "results" / "hup").glob("*_ep0_*.meta.json"))
    assert results, "no episode result was written after a terminal hangup"


# -- D2: a failed result write can be retried with save() ---------------------

@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_save_retries_a_failed_write(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    monitor = Monitor(
        _specs(tmp_path, "object_*[mass] <= 1"), results_dir=str(locked / "r"),
    )
    session = monitor.watch(_MockAdapter())
    with pytest.raises(RuntimeError, match="end"):
        session.save()
    session.step()
    try:
        with pytest.raises(OSError):
            session.end(success=True)
        assert session.end().path is None       # end() stays idempotent
        with pytest.raises(OSError):
            session.save()                      # cause not fixed yet
    finally:
        locked.chmod(0o755)
    record = session.save()
    assert record.success is True and record.status == "completed"
    assert record.path is not None and Path(record.path).is_file()
    assert session.save().path == record.path   # nothing left to retry
    assert session.end().path == record.path


def test_save_without_result_files_is_a_no_op(tmp_path):
    monitor = Monitor(_specs(tmp_path, "object_*[mass] <= 1"), results_dir=None)
    session = monitor.watch(_MockAdapter())
    session.step()
    record = session.end()
    assert session.save() is record and record.path is None


def test_evaluator_matches_sympify_on_supported_arithmetic():
    """The walk replaced ``sympify(...).subs(constants)``; on arithmetic
    that both accept it must build the same expression."""
    import sympy as sp
    from specola.spec.semantic import _SYMPY_LOCALS, _evaluate_arithmetic

    constants = {"a": 0.1, "b": 3.0}
    for expression in (
        "x**2 + 2*x*a - a/3 + floor(b)*x",
        "a*x*b + (a + b)*x/a - sqrt(a)*x**b",
        "Max(x, a)**2 - 1e-3*b + 1.0000000000000000001*x",
        "x/3 - 1/3 + 2**-3 + x**-0.5 + (-x)**3 - -x",
        "atan2(x, a) + log(x, 10) + exp(-b*x) + abs(x - a)",
        "(1e-400*x + 1e400*y) / 7",
    ):
        names = dict(_SYMPY_LOCALS, a=sp.Symbol("a"), b=sp.Symbol("b"))
        expected = sp.sympify(expression, locals=names).subs(
            {sp.Symbol(name): value for name, value in constants.items()}
        )
        actual = _evaluate_arithmetic(expression, {"x", "y"}, constants)
        assert sp.srepr(actual) == sp.srepr(expected), expression


def test_save_can_retry_an_interrupted_write(tmp_path, monkeypatch):
    from specola import session as session_module

    monitor = Monitor(
        _specs(tmp_path, "object_*[mass] <= 1"), results_dir=str(tmp_path / "r"),
    )
    session = monitor.watch(_MockAdapter())
    session.step()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(session_module, "save_rows", interrupted)
        with pytest.raises(KeyboardInterrupt):
            session.end()
    assert session.end().path is None
    assert Path(session.save().path).is_file()


# -- D6: browser drafts compile in a killable child ---------------------------

def _mock_discovery(tmp_path):
    from specola.discovery_capture import discovery_payload

    captured = []
    monitor = Monitor(_specs(tmp_path, "object_*[mass] <= 1"), results_dir=None)
    monitor.watch(_MockAdapter(), _reset_hook=lambda adapter, reset: captured.append(
        discovery_payload(adapter, reset, simulator="mock", task=None,
                          effective_dt=0.05)))
    return captured[0]


def test_draft_worker_matches_the_in_process_check(tmp_path):
    from specola.authoring import CheckContext, DraftChecker, check_draft
    from specola.dashboard.server import DiscoverySnapshot

    payload = _mock_discovery(tmp_path)
    context = CheckContext.from_discovery_model(
        DiscoverySnapshot.from_gzip(payload).model)
    marker = tmp_path / "marker"
    drafts = [
        dict(name="ok", formula="object_*[mass] >= 0.1"),
        dict(name="constants", formula="object_*[mass] <= c", constants={"c": 2.0}),
        dict(name="unknown", formula="nothing_*[mass] >= 0.1"),
        dict(name="syntax", formula="object_*[mass] >>> 1"),
        dict(name="inject",
             formula='__import__("os").system("touch %s") <= 1' % marker),
    ]
    checker = DraftChecker(payload)
    try:
        for draft in drafts:
            expected = check_draft(context, **draft)
            actual = checker.check(**draft)
            assert (actual.accepted, actual.report, actual.error) == (
                expected.accepted, expected.report, expected.error), draft
        assert checker.check(**drafts[0]).accepted
        assert not marker.exists()
    finally:
        checker.close()


def test_draft_worker_is_killed_at_its_time_limit(tmp_path):
    from specola.authoring import DraftChecker

    checker = DraftChecker(_mock_discovery(tmp_path))
    try:
        assert checker.check(name="a", formula="object_*[mass] >= 0.1").accepted
        child = checker._process
        os.kill(child.pid, signal.SIGSTOP)           # a child that never answers
        late = checker.check(
            name="a", formula="object_*[mass] >= 0.1", timeout=0.5)
        assert not late.accepted and "stopped after" in late.error
        assert child.poll() is not None             # killed, not abandoned
        # The next draft gets a fresh child and a normal answer.
        assert checker.check(name="a", formula="object_*[mass] >= 0.1").accepted
        assert checker._process is not child
    finally:
        checker.close()


def test_spec_check_socket_uses_the_worker(tmp_path):
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer

    payload = _mock_discovery(tmp_path)
    recording = tmp_path / "run" / "recording_000"
    writer = RecordingWriter(recording, {"run_id": "draft", "cameras": []})
    writer.write_discovery(payload)
    writer.finalize("completed")
    server = DashboardServer(recording, mode="view")
    client = server.socketio.test_client(server.app)
    try:
        good = client.emit("spec_check", {
            "name": "ok", "formula": "object_*[mass] >= 0.1", "revision": 1,
        }, callback=True)
        assert good["accepted"], good
        assert server._draft_checker_for is not None
        bad = client.emit("spec_check", {
            "name": "bad", "formula": "9**9**9**9 <= object_*[mass]",
            "revision": 2,
        }, callback=True)
        assert not bad["accepted"] and "power" in bad["report"]
        # Saving re-validates, and that too must not compile here.
        saved = client.emit("spec_save", {"entries": [
            {"name": "bad", "formula": "9**9**9**9 <= object_*[mass]"},
        ]}, callback=True)
        assert not saved["ok"] and "no longer compiles" in saved["error"]
        bomb = "a0: &a0 {x: 1}\n" + "".join(
            "a%d: &a%d {<<: [*a%d, *a%d]}\n" % (i, i, i - 1, i - 1)
            for i in range(1, 26))
        started = time.monotonic()
        refused = client.emit("spec_check", {
            "name": "ok", "formula": "object_*[mass] >= 0.1",
            "constants": bomb, "revision": 3,
        }, callback=True)
        assert not refused["accepted"] and "aliases" in refused["report"]
        assert time.monotonic() - started < 5.0
    finally:
        client.disconnect()
        server.shutdown()
    assert server._draft_checker_for is None
    late = server._check_untrusted_draft(name="a", formula="object_*[mass] >= 0")
    assert not late.accepted and "unavailable" in late.error


def test_socket_drafts_are_refused_when_the_worker_cannot_start(
        tmp_path, monkeypatch):
    """No fallback to compiling browser text in the dashboard process."""
    import subprocess

    from specola import authoring
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer

    recording = tmp_path / "run" / "recording_000"
    writer = RecordingWriter(recording, {"run_id": "draft", "cameras": []})
    writer.write_discovery(_mock_discovery(tmp_path))
    writer.finalize("completed")
    server = DashboardServer(recording, mode="view")
    client = server.socketio.test_client(server.app)

    def no_child(*args, **kwargs):
        raise OSError("cannot fork")

    def no_in_process_check(*args, **kwargs):
        raise AssertionError("browser text compiled in the dashboard process")

    monkeypatch.setattr(subprocess, "Popen", no_child)
    monkeypatch.setattr(authoring, "check_draft", no_in_process_check)
    try:
        checked = client.emit("spec_check", {
            "name": "ok", "formula": "object_*[mass] >= 0.1", "revision": 1,
        }, callback=True)
        assert not checked["accepted"] and "unavailable" in checked["report"]
        saved = client.emit("spec_save", {"entries": [
            {"name": "ok", "formula": "object_*[mass] >= 0.1"},
        ]}, callback=True)
        assert not saved["ok"]
    finally:
        client.disconnect()
        server.shutdown()


def test_closed_draft_checker_stays_closed(tmp_path):
    from specola.authoring import DRAFT_MAX_BYTES, DraftChecker

    checker = DraftChecker(_mock_discovery(tmp_path))
    assert checker.check(name="a", formula="object_*[mass] >= 0.1").accepted
    big = checker.check(name="a", formula="1 <= " + "1" * DRAFT_MAX_BYTES)
    assert not big.accepted and "larger" in big.error
    child = checker._process
    os.kill(child.pid, signal.SIGSTOP)
    started = time.monotonic()
    checker.close()
    assert time.monotonic() - started < 5.0 and child.poll() is not None
    with pytest.raises(OSError):
        checker.check(name="a", formula="object_*[mass] >= 0.1")


def test_save_recovers_from_an_interrupt_while_preparing_rows(
        tmp_path, monkeypatch):
    monitor = Monitor(
        _specs(tmp_path, "object_*[mass] <= 1"), results_dir=str(tmp_path / "r"),
    )
    session = monitor.watch(_MockAdapter())
    session.step()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(session, "_build_full_rows", interrupted)
        with pytest.raises(KeyboardInterrupt):
            session.end()
    assert session.end().path is None
    assert Path(session.save().path).is_file()


def test_long_decimal_literal_is_refused_not_rounded():
    with pytest.raises(ValueError, match="decimal literal"):
        _compile_predicate(
            "x - ceiling(1." + "0" * 39 + "1 - 1)",
            {"x": (("object", 1), "mass")}, {},
        )


def test_constants_comment_with_an_asterisk_is_accepted():
    from specola.authoring import parse_constants

    assert parse_constants("limit: 5  # m/s * 2 & more") == {"limit": 5.0}
    with pytest.raises(ValueError, match="aliases"):
        parse_constants("a: &a {x: 1}\nb: {<<: *a}")
