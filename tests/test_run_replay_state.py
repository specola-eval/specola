import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from specola import run_monitor, state_recording as sr

SPECS = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"


def _mock_class():
    """The mock class the registry builds right now.

    The conformance authority probe re-imports ``specola.adapters.fake``
    (see ``tests/test_run_startup.py``), so the class is resolved at call
    time rather than bound at collection.
    """
    return importlib.import_module("specola.adapters.fake")._MockAdapter


def _with_scalar(observation, scalar):
    """A copy of ``observation`` whose robot reads ``scalar`` as ``qvel_joint1``.

    ``examples/mock_specs.yaml`` monitors ``robot_*[qvel_joint1]``
    (``joint_settles``), so the robustness trace depends on the scalar.
    """
    robot = observation.perceived_entities[0]
    robot = dataclasses.replace(robot, properties={**robot.properties, "qvel_joint1": scalar})
    return dataclasses.replace(
        observation, perceived_entities={**observation.perceived_entities, 0: robot}
    )


def _Stateful():
    """A mock whose observation depends on a restorable scalar."""

    class _Stateful(_mock_class()):
        def __init__(self):
            super().__init__()
            self.scalar = 0.0
            self.calls = []
            self.history = []

        def state_signature(self):
            return {"engine": "stateful-mock", "length": 1}

        def capture_state(self):
            return np.array([self.scalar])

        def restore_state(self, state):
            self.calls.append("restore")
            self.scalar = float(state[0])

        def _reset_impl(self):
            self.calls.append("reset")
            return super()._reset_impl()

        def _observation(self, *args, **kwargs):
            # Both the reset and every step read the scene through here, so
            # the scalar reaches one exported property and the log gets one
            # entry per observation, row 0 included.
            observation = _with_scalar(super()._observation(*args, **kwargs), self.scalar)
            self.history.append(self.scalar)
            return observation

        def make_advancer(self, mode, demo, seed):
            rng = np.random.default_rng(seed)

            def advance(step):
                self.scalar += float(rng.uniform(0, 1))
                return True
            return advance

    return _Stateful()


def _run(monkeypatch, tmp_path, *extra, adapter=None):
    """Run the CLI; a successful main() returns, a rejected startup raises SystemExit(1)."""
    from specola import cli_common
    adapter = adapter or _Stateful()
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: adapter)
    monkeypatch.setattr(sys, "argv", ["specola run", "--sim", "mock", "--specs", str(SPECS),
                                      "--output-dir", str(tmp_path), "--ui", "plain", *extra])
    try:
        run_monitor.main()
    except SystemExit as done:
        return adapter, done.code
    return adapter, 0


def _state_files(tmp_path):
    return sorted(tmp_path.glob("*/*_states.npz"))


def _rho(output_dir):
    """The run's robustness column, in a fixed row order."""
    pytest.importorskip("pyarrow")
    from specola import report
    frame = report.load(str(output_dir)).sort_values(["step", "spec", "binding_key"])
    return frame["rho"].to_numpy()


def test_a_run_writes_one_state_file_per_episode(monkeypatch, tmp_path, capsys):
    _, code = _run(monkeypatch, tmp_path, "--steps", "5", "--run-name", "rec")
    assert code == 0
    (path,) = _state_files(tmp_path)
    rec = sr.StateRecording.load(path)
    assert rec.header["steps"] == 6          # reset row plus five steps
    assert rec.header["status"] == "completed"
    assert rec.header["simulator"] == "mock"
    assert str(path.resolve()) in capsys.readouterr().out


def test_no_state_recording_writes_nothing(monkeypatch, tmp_path):
    _, code = _run(monkeypatch, tmp_path, "--steps", "5", "--no-state-recording")
    assert code == 0 and _state_files(tmp_path) == []


def test_replay_restores_the_recorded_states(monkeypatch, tmp_path):
    adapter, _ = _run(monkeypatch, tmp_path, "--steps", "5", "--run-name", "rec", "--seed", "1")
    (path,) = _state_files(tmp_path)
    recording = sr.StateRecording.load(path)
    recorded = recording.states[:, 0].tolist()
    assert adapter.history == recorded          # row k is what observation k saw
    assert recorded[1:] != [0.0] * 5
    replayed, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(path),
                          "--steps", "5", "--no-state-recording")
    assert code == 0
    assert replayed.history == recorded
    assert replayed.calls.count("restore") == 6 and replayed.calls.count("reset") == 1
    assert replayed.calls.index("restore") < replayed.calls.index("reset")
    assert _state_files(tmp_path / "replay") == []

    # The monitored trace follows the restored states, row for row.
    original = _rho(tmp_path / "rec")
    np.testing.assert_array_equal(_rho(tmp_path / "replay"), original)
    # Negative control: the same rows in another order monitor another trace.
    shuffled = tmp_path / "shuffled_states.npz"
    np.savez(shuffled, header=json.dumps(recording.header), states=recording.states[::-1])
    _, code = _run(monkeypatch, tmp_path / "shuffled", "--mode", "replay", "--demo", str(shuffled),
                   "--steps", "5", "--no-state-recording")
    assert code == 0
    assert not np.array_equal(_rho(tmp_path / "shuffled"), original, equal_nan=True)


@pytest.mark.parametrize("steps, completion, observed", [
    ("10", "partial", 4),      # a positive limit the recording cannot meet
    ("3", "complete", 4),      # exactly the recording
    ("2", "complete", 3),      # the limit ends the run first
    ("0", "complete", 4),      # unbounded: the recording's end completes the run
])
def test_replay_length_against_steps(monkeypatch, tmp_path, capsys, steps, completion, observed):
    _, _ = _run(monkeypatch, tmp_path, "--steps", "3", "--run-name", "rec")
    (path,) = _state_files(tmp_path)
    capsys.readouterr()
    replayed, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(path), "--steps", steps)
    assert code == 0
    out = capsys.readouterr().out
    assert f"completion {completion}" in out
    assert ("recording exhausted after 3 steps" in out) == (completion == "partial")
    assert len(replayed.history) == observed


def test_robomimic_replay_marks_its_end_as_exhausted(monkeypatch):
    """Both replay tiers end a short run as truncated through one flag."""
    class _Sim:
        def set_state_from_flattened(self, state):
            pass

        def forward(self):
            pass

    class _Env:
        sim = _Sim()

    class _Adapter:
        env = _Env()

    monkeypatch.setattr(run_monitor, "_iter_replay_states", lambda demo: iter([np.zeros(2)]))
    advance = run_monitor._make_advancer(_Adapter(), "replay", "demo.hdf5", 0)
    assert advance(0) and not getattr(advance, "exhausted", False)
    assert not advance(1) and getattr(advance, "exhausted", False)


def test_a_state_recording_never_reaches_the_robomimic_reader():
    # A #N fragment keeps a .npz from the state-recording suffix check.
    with pytest.raises(SystemExit, match="is a Specola state recording"):
        run_monitor._resolve_demo("ep0_states.npz#1")


def test_web_replay_restores_before_the_only_reset(monkeypatch, tmp_path):
    pytest.importorskip("flask_socketio")
    from specola.dashboard import server as dashboard_server
    from specola import discovery_capture, engine as engine_module

    _, _ = _run(monkeypatch, tmp_path, "--steps", "3", "--run-name", "rec", "--seed", "2")
    (path,) = _state_files(tmp_path)
    recorded = sr.StateRecording.load(path).states[:, 0].tolist()
    adapter = _Stateful()
    monkeypatch.setattr(dashboard_server, "wait_for_interrupt", lambda: None)
    original_payload = discovery_capture.discovery_payload
    original_start = engine_module.MonitorEngine.start_episode

    def logged_payload(adapter_, reset, **kw):
        adapter.calls.append("discovery")
        adapter.discovery_scalar = adapter.scalar
        return original_payload(adapter_, reset, **kw)

    def logged_start(self, *a, **kw):
        adapter.calls.append("compile")
        return original_start(self, *a, **kw)

    monkeypatch.setattr(discovery_capture, "discovery_payload", logged_payload)
    monkeypatch.setattr(engine_module.MonitorEngine, "start_episode", logged_start)
    replayed, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(path),
                          "--steps", "3", "--web", "0", adapter=adapter)
    assert code == 0
    assert replayed.calls[:4] == ["restore", "reset", "discovery", "compile"]
    assert replayed.discovery_scalar == recorded[0]
    assert replayed.history == recorded
    (recording_dir,) = (tmp_path / "replay").glob("*/recording_000")
    assert (recording_dir / "discovery.json.gz").exists()


def test_a_failing_result_finalisation_closes_the_state_file_as_failed(monkeypatch, tmp_path):
    from specola import session as session_module
    adapter = _Stateful()

    def failing_end(self, status):
        raise OSError("disk full")

    monkeypatch.setattr(session_module.Session, "end", failing_end)
    with pytest.raises((OSError, SystemExit)):
        _run(monkeypatch, tmp_path, "--steps", "2", adapter=adapter)
    (path,) = _state_files(tmp_path)
    assert sr.StateRecording.load(path).header["status"] == "failed"


def test_replay_refuses_a_signature_mismatch_before_any_restore(monkeypatch, tmp_path, capsys):
    _, _ = _run(monkeypatch, tmp_path, "--steps", "2", "--run-name", "rec")
    (path,) = _state_files(tmp_path)
    rec = sr.StateRecording.load(path)
    rec.header["signature"] = {"engine": "other", "length": 1}
    forged = tmp_path / "forged_states.npz"
    np.savez(forged, header=json.dumps(rec.header), states=rec.states)
    adapter, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(forged))
    assert code == 1
    assert "state signature differs" in capsys.readouterr().err
    assert "restore" not in adapter.calls
    assert not (tmp_path / "replay").exists()


def test_replay_refuses_a_wrong_state_length(monkeypatch, tmp_path, capsys):
    _, _ = _run(monkeypatch, tmp_path, "--steps", "2", "--run-name", "rec")
    (path,) = _state_files(tmp_path)
    rec = sr.StateRecording.load(path)
    forged = tmp_path / "forged_states.npz"
    np.savez(forged, header=json.dumps(rec.header), states=np.hstack([rec.states, rec.states]))
    _, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(forged))
    assert code == 1
    assert "2 wide, the signature says 1" in capsys.readouterr().err


def test_replay_refuses_another_dt(monkeypatch, tmp_path, capsys):
    _, _ = _run(monkeypatch, tmp_path, "--steps", "2", "--run-name", "rec")
    (path,) = _state_files(tmp_path)
    adapter, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(path), "--dt", "0.5")
    assert code == 1
    assert "pass --dt" in capsys.readouterr().err
    assert "restore" not in adapter.calls


def test_a_malformed_npz_is_rejected_not_read_as_a_demonstration(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "bad_states.npz"
    np.savez(bad, x=np.zeros(2))
    _, code = _run(monkeypatch, tmp_path / "replay", "--mode", "replay", "--demo", str(bad))
    assert code == 1
    assert "not a state recording" in capsys.readouterr().err


def test_an_interrupted_run_closes_the_state_file_as_truncated(monkeypatch, tmp_path):
    adapter = _Stateful()
    original = adapter.make_advancer

    def make_advancer(mode, demo, seed):
        advance = original(mode, demo, seed)

        def interrupting(step):
            if step == 2:
                raise KeyboardInterrupt
            return advance(step)
        return interrupting

    adapter.make_advancer = make_advancer
    _run(monkeypatch, tmp_path, "--steps", "5", adapter=adapter)
    (path,) = _state_files(tmp_path)
    rec = sr.StateRecording.load(path)
    assert rec.header["status"] == "truncated" and rec.header["steps"] == 3


def test_replay_is_refused_on_an_adapter_without_a_state_tier(monkeypatch, tmp_path, capsys):
    _, _ = _run(monkeypatch, tmp_path, "--steps", "2", "--run-name", "rec")
    (path,) = _state_files(tmp_path)
    from specola import cli_common
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: _mock_class()())
    monkeypatch.setattr(sys, "argv", ["specola run", "--sim", "mock", "--specs", str(SPECS),
                                      "--output-dir", str(tmp_path / "r"), "--ui", "plain",
                                      "--mode", "replay", "--demo", str(path)])
    with pytest.raises(SystemExit) as done:
        run_monitor.main()
    assert done.value.code == 1
    assert "no state tier" in capsys.readouterr().err


def test_a_non_finite_state_stops_recording_but_not_monitoring(monkeypatch, tmp_path, capsys):
    adapter = _Stateful()
    original = adapter.capture_state

    def capture():
        state = original()
        return np.array([float("nan")]) if len(adapter.history) == 3 else state

    adapter.capture_state = capture
    _, code = _run(monkeypatch, tmp_path, "--steps", "5", adapter=adapter)
    assert code == 0
    assert len(adapter.history) == 6              # monitoring ran to the end
    (path,) = _state_files(tmp_path)
    assert sr.StateRecording.load(path).header["status"] == "failed"
    assert "State recording stopped at step" in capsys.readouterr().out
