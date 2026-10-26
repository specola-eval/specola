"""Rejected startup retains diagnostics without publishing a monitoring run."""

import importlib
import logging
import json
from pathlib import Path
import sys

import pytest

from specola import run_monitor
from specola.adapters.fake import _MockAdapter
from specola.engine import MonitorEngine
from specola.query_config import QueryConfiguration


def _registered_mock_class():
    """The mock class the registry builds right now.

    The conformance authority probe (``conformance.authority``) re-imports
    ``specola.adapters.fake`` and replaces the cached module, so after a
    certification-row test in the same session the class bound above at
    collection is no longer the one ``--sim mock`` constructs. A patch on
    the stale class then never fires, and a run that expected a startup
    rejection drives the mock scene with no step bound.
    """
    return importlib.import_module("specola.adapters.fake")._MockAdapter


@pytest.mark.parametrize("failure_stage", ["load", "replay", "watch", "compile"])
@pytest.mark.parametrize("web", [False, True])
def test_startup_rejection_does_not_publish_run(tmp_path, monkeypatch, capsys,
                                             failure_stage, web):
    _MockAdapter = _registered_mock_class()
    output = tmp_path / "results"
    diagnostics = []
    recorder_calls = []
    server_calls = []
    original_failure = run_monitor._write_failure
    original_reset = _MockAdapter._reset_impl
    resets = []

    def reject():
        print("startup stdout " + failure_stage)
        print("startup stderr " + failure_stage, file=sys.stderr)
        logging.getLogger("startup.native").warning("startup logger %s", failure_stage)
        raise ValueError("guard rejected " + failure_stage)

    def reset(adapter):
        resets.append(len(resets) + 1)
        if failure_stage == "watch":
            reject()
        return original_reset(adapter)

    def write_failure(**kwargs):
        diagnostics.append(kwargs)
        original_failure(**kwargs)

    from specola.dashboard import recorder, server
    monkeypatch.setattr(recorder, "WebRecorderSink", lambda *a, **k: recorder_calls.append(a))
    monkeypatch.setattr(server, "DashboardServer", lambda *a, **k: server_calls.append(a))
    monkeypatch.setattr(_MockAdapter, "_reset_impl", reset)
    monkeypatch.setattr(run_monitor, "_write_failure", write_failure)
    if failure_stage == "load":
        from specola import cli_common
        monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: reject())
    if failure_stage == "replay":
        monkeypatch.setattr(run_monitor, "prime_replay_env", lambda *a, **k: reject())
    if failure_stage == "compile":
        monkeypatch.setattr(MonitorEngine, "start_episode", lambda *a, **k: reject())
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    args = ["specola run", "--sim", "mock", "--specs", str(specs),
            "--output-dir", str(output), "--run-name", "rejected", "--ui", "plain"]
    if web:
        args.extend(["--web", "0"])
    if failure_stage == "replay":
        args.extend(["--mode", "replay", "--demo", "unused.hdf5"])
    monkeypatch.setattr(sys, "argv", args)
    handlers_before = tuple(logging.getLogger().handlers)

    with pytest.raises(SystemExit) as failure:
        run_monitor.main()

    assert failure.value.code == 1
    assert not output.exists()
    assert recorder_calls == []
    assert server_calls == []
    assert tuple(logging.getLogger().handlers) == handlers_before
    assert len(diagnostics) == 1
    record = diagnostics[0]
    assert record.get("result_path") is None
    path = record["diagnostics_path"]
    assert output not in path.parents
    assert "ValueError: guard rejected " + failure_stage in path.read_text()
    log = path.with_name("simulator.log").read_text()
    for stream in ("stdout", "stderr", "logger"):
        assert "startup " + stream + " " + failure_stage in log
    rendered = capsys.readouterr()
    assert "Results     none" in rendered.err
    assert "guard rejected " + failure_stage in rendered.err


def test_load_installs_queries_before_reset_and_reads_reset_metadata(monkeypatch):
    from specola import cli_common

    queries = QueryConfiguration.from_document({
        "schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": "native.query", "arguments": {"cutoff": 0.125}}],
    })
    events = []

    class ConfiguredAdapter(_MockAdapter):
        def _reset_impl(self):
            assert self.query_configuration == queries
            events.append("reset")
            self.DT = 0.2
            return super()._reset_impl()

        @property
        def dt(self):
            events.append("dt")
            return super().dt

    adapter = ConfiguredAdapter()
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: adapter)
    result = cli_common.load_sim("configured", queries=queries)
    # Base metadata validation may read dt during reset. The effective CLI dt
    # must come from the completed reset rather than the pre-reset value.
    assert result[1] == 0.2
    assert adapter.query_configuration is queries


@pytest.mark.parametrize("command", ["discover", "check", "run"])
def test_public_cli_installs_query_subject_before_first_reset(tmp_path, monkeypatch, command):
    _MockAdapter = _registered_mock_class()
    from specola import check_specs, discover

    document = {
        "schema": "specola.queries.v1", "instances": [],
        "populations": [{"binding": "fixture.query", "arguments": {"limit": 0.125}}],
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(document))
    expected = QueryConfiguration.from_document(document)
    seen = []
    original_reset = _MockAdapter._reset_impl

    def reset(adapter):
        seen.append(adapter.query_configuration)
        assert adapter.query_configuration == expected
        return original_reset(adapter)

    monkeypatch.setattr(_MockAdapter, "_reset_impl", reset)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    args = ["specola " + command, "--sim", "mock", "--queries", str(path)]
    if command == "discover":
        args.extend(["--format", "json"])
        main = discover.main
    elif command == "check":
        args.extend([str(specs), "--no-pager"])
        main = check_specs.main
    else:
        args.extend(["--specs", str(specs), "--steps", "1", "--ui", "plain",
                     "--output-dir", str(tmp_path / "results")])
        main = run_monitor.main
    monkeypatch.setattr(sys, "argv", args)
    if command == "check":
        with pytest.raises(SystemExit) as finished:
            main()
        assert finished.value.code == 0
    else:
        main()
    assert seen == [expected]


def test_web_startup_publishes_its_single_reset_after_compilation(tmp_path, monkeypatch):
    _MockAdapter = _registered_mock_class()
    from specola import discovery_capture
    from specola.dashboard import recorder, server

    events = []
    captured = []
    queued = []
    original_reset = _MockAdapter._reset_impl
    original_watch = run_monitor.Monitor.watch
    original_payload = discovery_capture.discovery_payload
    original_sink = recorder.WebRecorderSink
    resets = []
    output = tmp_path / "results"

    def reset(adapter):
        resets.append(len(resets) + 1)
        print("native reset " + str(len(resets)))
        return original_reset(adapter)

    def payload(adapter, reset_result, **kwargs):
        assert len(resets) == 1
        assert not output.exists()
        data = original_payload(adapter, reset_result, **kwargs)
        captured.append(data)
        events.append("capture")
        return data

    def watch(monitor, *args, **kwargs):
        session = original_watch(monitor, *args, **kwargs)
        assert not output.exists()
        events.append("compiled")
        return session

    class Sink(original_sink):
        def __init__(self, *args, **kwargs):
            assert events == ["capture", "compiled"]
            events.append("recorder")
            super().__init__(*args, **kwargs)

        def write_discovery(self, payload):
            assert events[-1] == "callbacks"
            queued.append(payload)
            events.append("queued")
            super().write_discovery(payload)

        def set_callbacks(self, on_line, on_episode_end, on_discovery=None):
            events.append("callbacks")
            super().set_callbacks(on_line, on_episode_end, on_discovery)

    class Server:
        active = False

        def __init__(self, *args, **kwargs):
            assert events[-1] == "recorder"

        def start(self, port):
            return "http://unused.invalid"

        def on_line(self, *args):
            pass

        def on_episode_end(self, *args):
            pass

        def set_discovery_snapshot(self, *args):
            pass

    def evaluate(session, **kwargs):
        assert events == ["capture", "compiled", "recorder", "callbacks", "queued"]
        assert queued == captured
        assert len(queued) == 1
        log = (output / "complete" / "simulator.log").read_text()
        assert "native reset 1\n" in log and "native reset 2\n" not in log
        events.append("evaluate")
        session.end()

    monkeypatch.setattr(_MockAdapter, "_reset_impl", reset)
    monkeypatch.setattr(run_monitor.Monitor, "watch", watch)
    monkeypatch.setattr(discovery_capture, "discovery_payload", payload)
    monkeypatch.setattr(recorder, "WebRecorderSink", Sink)
    monkeypatch.setattr(server, "DashboardServer", Server)
    monkeypatch.setattr(run_monitor, "_typed_eval_loop", evaluate)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--web", "0",
        "--ui", "plain", "--output-dir", str(output), "--run-name", "complete",
    ])
    run_monitor.main()
    assert events[-1] == "evaluate"
    assert len(resets) == 1


def test_run_resets_the_adapter_once_and_reports_that_reset(tmp_path, monkeypatch, capsys):
    """``specola run`` has no readiness reset: ``Monitor.watch`` performs the
    episode's only one, and the ready line counts the entities it read."""
    _MockAdapter = _registered_mock_class()
    calls = []
    original_reset = _MockAdapter.reset

    def reset(adapter):
        observation = original_reset(adapter)
        calls.append(len(observation.perceived_entities))
        return observation

    monkeypatch.setattr(_MockAdapter, "reset", reset)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--steps", "2",
        "--output-dir", str(tmp_path / "results"), "--ui", "plain",
    ])
    run_monitor.main()
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert out.count("ready   adapter_dt=") == 1
    assert "entities={}".format(calls[0]) in out


def test_missing_query_arguments_reject_before_metadata_and_artifacts(tmp_path, monkeypatch, capsys):
    _MockAdapter = _registered_mock_class()
    from specola.conformance.provider import ExtractionProvider
    from specola.conformance.query_bindings import QueryConfigurationRequired

    class RequiredQueries(ExtractionProvider):
        def guard_source(self, adapter):
            assert adapter.query_configuration == QueryConfiguration.empty()
            raise QueryConfigurationRequired((("fixture.distance", ("distmax",)),))

    def forbidden_metadata(adapter):
        pytest.fail("native metadata was read before the source/query guard")

    monkeypatch.setattr(_MockAdapter, "_extraction_provider", RequiredQueries(), raising=False)
    monkeypatch.setattr(_MockAdapter, "dt", property(forbidden_metadata))
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs),
        "--output-dir", str(output), "--ui", "plain", "--web", "0",
    ])
    with pytest.raises(SystemExit) as rejected:
        run_monitor.main()
    assert rejected.value.code == 1
    assert not output.exists()
    error = capsys.readouterr().err
    assert "query_configuration_required: fixture.distance requires distmax" in error
    assert "Results     none" in error
    assert "specola-startup-" in error


def test_render_without_live_viewer_is_rejected_before_artifacts(tmp_path, monkeypatch, capsys):
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--steps", "1",
        "--output-dir", str(output), "--ui", "plain", "--render",
    ])
    with pytest.raises(SystemExit) as rejected:
        run_monitor.main()
    assert rejected.value.code == 1
    assert not output.exists()
    error = capsys.readouterr().err
    assert "--render is not supported by the mock adapter: it has no live viewer" in error
    assert "Results     none" in error


@pytest.mark.parametrize("extra, message", [
    (["--mode", "replay"], "--mode replay needs a real sim env (not 'mock')"),
    (["--demo", "x.hdf5", "--mode", "hold"],
     "--demo is only read by --mode replay, not --mode hold"),
])
def test_drive_mode_refusal_is_rejected_before_artifacts(tmp_path, monkeypatch, capsys,
                                                         extra, message):
    """A drive mode the adapter cannot run is a startup rejection, like a
    missing --demo (#135): no run directory, no result file."""
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--steps", "1",
        "--output-dir", str(output), "--ui", "plain", *extra,
    ])
    with pytest.raises(SystemExit) as rejected:
        run_monitor.main()
    assert rejected.value.code == message
    assert not output.exists()
    error = capsys.readouterr().err
    assert "Phase       drive mode setup" in error
    assert "Error       SystemExit: " + message in error
    assert "Results     none" in error
    assert "specola-startup-" in error


def test_drive_advancer_is_built_once_before_the_run_directory(tmp_path, monkeypatch):
    """Building an advancer can act on the simulator (CARLA's random mode
    spawns its fleet), so startup builds it once and the stream uses it."""
    output = tmp_path / "results"
    built = []
    original = run_monitor._make_advancer

    def make_advancer(*args, **kwargs):
        assert not output.exists()
        built.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(run_monitor, "_make_advancer", make_advancer)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--steps", "2",
        "--output-dir", str(output), "--ui", "plain", "--mode", "hold",
    ])
    run_monitor.main()
    assert built == ["hold"]
    assert list(output.glob("*/*.parquet")) or list(output.glob("*/*.csv"))


def test_render_runs_on_an_adapter_with_a_live_viewer(tmp_path, monkeypatch):
    from specola import cli_common

    calls = []

    class ViewerAdapter(_MockAdapter):
        def render(self):
            calls.append("render")

    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: ViewerAdapter())
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    monkeypatch.setattr(sys, "argv", [
        "specola run", "--sim", "mock", "--specs", str(specs), "--steps", "1",
        "--output-dir", str(tmp_path / "results"), "--ui", "plain", "--render",
    ])
    run_monitor.main()
    assert calls == ["render"]


@pytest.mark.parametrize("command", ["discover", "check", "run"])
def test_pending_native_row_is_an_explicit_cli_rejection(tmp_path, monkeypatch, capsys, command):
    from specola import check_specs, cli_common, discover
    from specola import adapter_enrollment as enrollment
    from specola.adapter import SimulatorAdapter
    from specola.adapters.carla import CarlaAdapter

    # A synthetic pending row: carla re-enrolled without an extraction
    # provider, as it stood before the 2026-10-13 scope exclusion.
    monkeypatch.setattr(enrollment, "EXTRACTION_SCOPE_EXCLUSIONS", tuple(
        item for item in enrollment.EXTRACTION_SCOPE_EXCLUSIONS if item[0] != "carla"))
    monkeypatch.setitem(enrollment._REVIEWED_IMPLEMENTATIONS, "specola.adapters.carla",
                        ("CarlaAdapter", "_make_carla"))
    adapter = object.__new__(CarlaAdapter)
    SimulatorAdapter.__init__(adapter)
    monkeypatch.setattr(cli_common, "_load_sim", lambda *a, **k: adapter)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    output = tmp_path / "results"
    args = ["specola " + command, "--sim", "carla"]
    if command == "discover":
        args.extend(["--format", "json", "--output", str(output)])
        main = discover.main
    elif command == "check":
        args.extend([str(specs), "--no-pager"])
        main = check_specs.main
    else:
        args.extend(["--specs", str(specs), "--output-dir", str(output), "--web", "0"])
        main = run_monitor.main
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as rejected:
        main()
    assert rejected.value.code == 1
    assert not output.exists()
    rendered = capsys.readouterr()
    assert "Observation none" in rendered.err
    assert "Extraction  bundled_row_pending" in rendered.err
    assert "Results     none" in rendered.err
    assert "specola-startup-" in rendered.err


def test_run_tells_a_factory_whether_it_captures_frames(tmp_path, monkeypatch):
    """Only ``run`` without ``--web`` or ``--render`` passes
    ``capture_frames=False``, and only to a factory that declares it."""
    from specola import registry
    from specola.dashboard import recorder, server

    received = []

    def frames(task, robot, target_only, capture_frames=True):
        received.append(capture_frames)
        return _MockAdapter()

    def legacy(task, robot, target_only):
        received.append("legacy")
        return _MockAdapter()

    monkeypatch.setitem(registry._REGISTRY, "frames", registry._AdapterEntry("frames", frames, ""))
    monkeypatch.setitem(registry._REGISTRY, "legacy", registry._AdapterEntry("legacy", legacy, ""))
    # --web starts no dashboard here: its construction fails and the run
    # continues without one.
    monkeypatch.setattr(recorder, "WebRecorderSink", lambda *a, **k: None)
    monkeypatch.setattr(server, "DashboardServer", lambda *a, **k: None)
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    runs = (("frames",), ("frames", "--web", "0"), ("frames", "--render"), ("legacy",))
    for index, (sim, *extra) in enumerate(runs):
        monkeypatch.setattr(sys, "argv", [
            "specola run", "--sim", sim, "--specs", str(specs), "--steps", "1", "--ui", "plain",
            "--output-dir", str(tmp_path / "results"), "--run-name", "run" + str(index), *extra,
        ])
        try:
            run_monitor.main()
        except SystemExit:
            pass  # --render is refused after construction: the mock has no viewer
    assert received == [False, True, True, "legacy"]


def test_run_passes_its_seed_to_a_factory_that_declares_one(tmp_path, monkeypatch):
    from specola import registry

    received = []

    def seeded(task, robot, target_only, seed=None):
        received.append(seed)
        return _MockAdapter()

    def legacy(task, robot, target_only):
        received.append("legacy")
        return _MockAdapter()

    monkeypatch.setitem(registry._REGISTRY, "seeded", registry._AdapterEntry("seeded", seeded, ""))
    monkeypatch.setitem(registry._REGISTRY, "legacy", registry._AdapterEntry("legacy", legacy, ""))
    specs = Path(__file__).resolve().parents[1] / "examples" / "mock_specs.yaml"
    for index, (sim, *extra) in enumerate((("seeded", "--seed", "7"), ("seeded",), ("legacy", "--seed", "7"))):
        monkeypatch.setattr(sys, "argv", [
            "specola run", "--sim", sim, "--specs", str(specs), "--steps", "1", "--ui", "plain",
            "--output-dir", str(tmp_path / "results"), "--run-name", "run" + str(index), *extra,
        ])
        run_monitor.main()
    assert received == [7, 0, "legacy"]
