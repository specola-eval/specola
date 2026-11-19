"""Public local loading, CLI authoring and recorded browser trust boundaries."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import yaml

from test_spec_functions import InputsAdapter, specification


@pytest.fixture
def cli_scene(tmp_path):
    path = specification(tmp_path)
    source = tmp_path / "local.py"
    source.write_text(source.read_text().replace(
        '    calls.append(',
        '    from pathlib import Path\n    Path(__file__).with_suffix(".evaluated").touch()\n    calls.append(',
    ))
    (tmp_path / "function_fixture.py").write_text('''
import json
from pathlib import Path
from test_spec_functions import InputsAdapter
from specola.registry import register
class Fixture(InputsAdapter):
    def make_advancer(self, *args, **kwargs):
        return lambda step: True
    def close(self):
        Path(__file__).with_suffix('.captures').write_text(json.dumps(self.captures))
register('function-fixture', lambda *args, **kwargs: Fixture())
''')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=os.pathsep.join(
        str(p) for p in (root / "src", root / "tests", tmp_path)))

    def run(command, *arguments, stdin=None):
        result = subprocess.run(
            [sys.executable, "-m", "specola.cli", command, "--sim", "function-fixture",
             "--adapter-module", "function_fixture", *map(str, arguments)],
            env=env, cwd=tmp_path, input=stdin, capture_output=True, text=True, timeout=30,
        )
        (tmp_path / f"{command}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return result
    return path, run


def test_discover_and_plain_writer_explicitly_load_and_preserve_functions(cli_scene, tmp_path):
    path, run = cli_scene
    document = json.loads(run("discover", "--functions", path, "--format", "json").stdout)
    assert set(document["authored_functions"]["declarations"]) == {"twice"}
    assert document["authored_functions"]["active_functions"] == ["twice"]
    assert "derived" in json.dumps(document["entities"])
    assert not (tmp_path / "local.evaluated").exists()
    saved = tmp_path / "written.yaml"
    run("write-specs", "--functions", path, "--ui", "plain", "--output", saved,
        stdin="written\nobject_0[derived] >= 1\n\ny\n\n\n")
    loaded = yaml.safe_load(saved.read_text())
    assert loaded["functions"] == document["authored_functions"]["declarations"]
    assert loaded["specs"][0]["formula"] == "object_0[derived] >= 1"
    assert not (tmp_path / "local.evaluated").exists()
    run("check", saved, "--no-pager")
    assert not (tmp_path / "local.evaluated").exists()


@pytest.mark.parametrize("active", [True, False])
def test_cli_check_run_preserve_loaded_and_requested_identity(cli_scene, tmp_path, active):
    path, run = cli_scene
    if not active:
        document = yaml.safe_load(path.read_text())
        for entry in document["specs"]:
            entry["formula"] = r"Min(object_0[* \ derived]) >= 0"
        path.write_text(yaml.safe_dump(document))
    run("check", path, "--no-pager")
    assert not (tmp_path / "local.evaluated").exists()
    run("run", "--specs", path, "--steps", "2", "--mode", "hold", "--ui", "plain",
        "--output-dir", tmp_path / "results", "--no-state-recording")
    assert (tmp_path / "local.evaluated").exists() is active
    assert bool(json.loads((tmp_path / "function_fixture.captures").read_text())) is active
    sidecar = next((tmp_path / "results").rglob("*.meta.json"), None)
    if sidecar is not None:
        metadata = json.loads(sidecar.read_text())
    else:
        import pyarrow.parquet as pq
        parquet = next((tmp_path / "results").rglob("*.parquet"))
        metadata = {key.decode(): value.decode() for key, value in pq.read_metadata(parquet).metadata.items()}
    assert set(json.loads(metadata["spec_functions"])) == {"twice"}
    assert json.loads(metadata["spec_functions_active"]) == (["twice"] if active else [])


def test_browser_run_and_restored_scene_never_load_code_from_requests(tmp_path):
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer
    from specola.discovery_capture import discovery_payload
    from specola.session import Monitor
    from specola.engine import MonitorEngine
    path = specification(tmp_path)
    adapter = InputsAdapter()
    captured = []
    monitor = Monitor(str(path), results_dir=None)
    session = monitor.watch(adapter, _reset_hook=lambda a, r: captured.append(
        discovery_payload(a, r, simulator="function-fixture", task=None, effective_dt=0.1)))
    session.step()
    recording = tmp_path / "run" / "recording_000"
    writer = RecordingWriter(recording, {"run_id": "functions", "cameras": []})
    writer.write_discovery(captured[0])
    writer.finalize("completed")
    for restored in (False, True):
        server = DashboardServer(recording, mode="view")
        if not restored:
            server.set_discovery_snapshot(captured[0])
        (tmp_path / "local.py").write_text("raise RuntimeError('browser must not execute source')")
        client = server.socketio.test_client(server.app)
        try:
            checked = client.emit("spec_check", {"name": "browser", "formula": "object_0[derived] >= 1",
                "revision": 2, "functions": {"evil": {"python": str(tmp_path / "local.py")}}}, callback=True)
            assert isinstance(checked, dict)
            assert checked["accepted"], checked
            denied = client.emit("spec_save", {"entries": [dict(checked["entry"], python=str(tmp_path / "local.py"))]}, callback=True)
            assert isinstance(denied, dict)
            assert not denied["ok"]
            saved = client.emit("spec_save", {"entries": [checked["entry"]]}, callback=True)
            assert isinstance(saved, dict)
            assert saved["ok"], saved
            assert set(yaml.safe_load(Path(saved["path"]).read_text())["functions"]) == {"twice"}
            with pytest.raises(ValueError, match="source digest changed"):
                MonitorEngine.load(saved["path"])
        finally:
            client.disconnect()


def test_inactive_output_collision_and_unknown_dynamic_targets(tmp_path):
    from specola.session import Monitor
    from specola.capabilities import reset_adapter
    from specola.spec_functions import FunctionSet, load_functions
    path = specification(tmp_path, formula="object_0[native] >= 0")
    document = yaml.safe_load(path.read_text())
    document["functions"]["twice"]["outputs"] = {"native": {}}
    functions = FunctionSet(load_functions(document["functions"], path), [])
    adapter = InputsAdapter()
    with pytest.raises(ValueError, match="output collision"):
        functions.prepare(adapter, reset_adapter(adapter))
    assert adapter.captures == 0
    document["functions"]["twice"]["entity_class"] = "distance_pair"
    document["functions"]["twice"]["outputs"] = {"distance": {"unit": "m"}}
    path.write_text(yaml.safe_dump(document))
    session = Monitor(str(path), results_dir=None).watch(adapter)
    session.step()
    assert not session.engine.functions._plans
    assert not session.engine.functions.labels
    assert adapter.captures == 0


def test_textual_cli_keeps_loaded_declarations_when_saving(tmp_path, monkeypatch):
    import asyncio
    from textual.app import App
    from textual.widgets import Button, Input, TextArea
    from specola import registry, write_specs
    path = specification(tmp_path)
    saved = tmp_path / "textual.yaml"
    registry.register("function-textual", lambda *args, **kwargs: InputsAdapter(), replace=True)

    def run_app(app):
        async def drive():
            async with app.run_test(size=(140, 55)) as pilot:
                await pilot.pause()
                app.query_one("#spec_name", Input).value = "textual"
                app.query_one("#formula", TextArea).load_text("object_0[derived] >= 1")
                await pilot.pause()
                await pilot.click("#validate")
                await pilot.pause()
                assert not app.query_one("#keep", Button).disabled
                await pilot.click("#keep")
                await pilot.pause()
                await pilot.click("#save")
                await pilot.pause()
                list(app.screen.query(Input))[0].value = str(saved)
                await pilot.press("enter")
                await pilot.pause()
        asyncio.run(drive())

    monkeypatch.setattr(App, "run", run_app)
    monkeypatch.setattr(write_specs, "_is_tty", lambda: True)
    monkeypatch.setattr(sys, "argv", ["specola write-specs", "--sim", "function-textual",
        "--functions", str(path), "--ui", "textual", "--output", str(saved)])
    write_specs.main()
    assert set(yaml.safe_load(saved.read_text())["functions"]) == {"twice"}


@pytest.mark.parametrize("command", ["discover", "run"])
def test_public_web_entrypoint_receives_loaded_functions(tmp_path, monkeypatch, command):
    from specola import registry, discover, run_monitor
    from specola.dashboard import server as dashboard
    path = specification(tmp_path)

    class Fixture(InputsAdapter):
        def make_advancer(self, *args, **kwargs):
            return lambda step: True

    registry.register("function-web", lambda *args, **kwargs: Fixture(), replace=True)
    checked = []

    def serve_until_checked(server):
        client = server.socketio.test_client(server.app)
        try:
            result = client.emit("spec_check", {"name": "web", "formula": "object_0[derived] >= 1"}, callback=True)
            assert result["accepted"], result
            saved = client.emit("spec_save", {"entries": [result["entry"]]}, callback=True)
            assert saved["ok"], saved
            assert set(yaml.safe_load(Path(saved["path"]).read_text())["functions"]) == {"twice"}
            checked.append(server.recording_dir if hasattr(server, "recording_dir") else True)
        finally:
            client.disconnect()

    monkeypatch.setattr(dashboard.DashboardServer, "start", lambda self, port: "http://127.0.0.1:0")
    monkeypatch.setattr(dashboard, "wait_for_interrupt", serve_until_checked)
    monkeypatch.chdir(tmp_path)
    arguments = [f"specola {command}", "--sim", "function-web", "--web", "0"]
    if command == "discover":
        arguments += ["--functions", str(path)]
    else:
        arguments += ["--specs", str(path), "--steps", "2", "--mode", "hold", "--ui", "plain", "--no-state-recording", "--output-dir", str(tmp_path / "results")]
    monkeypatch.setattr(sys, "argv", arguments)
    (discover.main if command == "discover" else run_monitor.main)()
    assert len(checked) == 1
