"""Adapter ownership ends before result publication."""
from types import SimpleNamespace

import pytest

from specola.adapters.fake import FakeAdapter
from specola.run_monitor import _snapshot_stream


def test_base_close_attempts_each_resource_once_after_failure(capsys):
    adapter = FakeAdapter()
    calls = []

    def render():
        calls.append('render')
        raise RuntimeError('injected renderer close failure')

    adapter.close_render = render
    adapter.close_native_extraction = lambda: calls.append('native')
    adapter.close()
    adapter.close()
    assert calls == ['render', 'native']
    assert 'RuntimeError: injected renderer close failure' in capsys.readouterr().err


@pytest.mark.mujoco
@pytest.mark.parametrize('owned', [True, False])
def test_mujoco_close_attempts_env_after_renderer_failure(capsys, owned):
    """A factory-built env is closed even after a renderer failure; an env
    the caller passed in (Monitor.watch, RoboCasaAdapter(env)) stays open."""
    pytest.importorskip('mujoco')
    from specola.adapters.mujoco import MujocoAdapter

    calls = []
    adapter = object.__new__(MujocoAdapter)
    adapter.env = SimpleNamespace(close=lambda: calls.append('env'))
    if owned:
        adapter._owns_environment = adapter.env.close
    adapter.close_native_extraction = lambda: calls.append('native')

    def render():
        calls.append('render')
        raise RuntimeError('injected renderer close failure')

    adapter.close_render = render
    adapter.close()
    adapter.close()
    assert calls == ['render', 'native'] + (['env'] if owned else [])
    assert 'injected renderer close failure' in capsys.readouterr().err


@pytest.mark.parametrize('early_stop', [False, True])
def test_stream_closes_on_owner_thread_without_rendering(early_stop):
    import threading

    calls = []
    adapter = SimpleNamespace(close=lambda: calls.append(threading.get_ident()))
    stop = threading.Event()
    if early_stop:
        stop.set()
    session = SimpleNamespace(adapter=adapter)
    assert list(_snapshot_stream(session, None, advance=lambda _: False,
                                 max_steps=1, render=False, stop_event=stop)) == []
    assert calls == [threading.get_ident()]

def test_stream_preserves_original_error_after_cleanup_failure(capsys):
    adapter = FakeAdapter()

    def cleanup():
        raise RuntimeError("injected renderer close failure")

    def advance(_):
        raise ValueError("original step failure")

    adapter.close_render = cleanup
    with pytest.raises(ValueError, match="original step failure"):
        list(_snapshot_stream(SimpleNamespace(adapter=adapter), None,
                              advance=advance, max_steps=1, render=False))
    assert "injected renderer close failure" in capsys.readouterr().err


def test_cli_closes_adapter_when_startup_rejects_specs(tmp_path, monkeypatch):
    import specola.run_monitor as module
    import sys

    adapter = FakeAdapter()
    closed = []
    adapter.close_render = lambda: closed.append(True)
    monkeypatch.setattr(module, "construct_sim", lambda *a, **kw: adapter)
    monkeypatch.setattr(sys, "argv", [
        "run", "--sim", "mock", "--specs", str(tmp_path / "missing.yaml"),
        "--output-dir", str(tmp_path), "--ui", "plain"])
    with pytest.raises(SystemExit):
        module.main()
    assert closed == [True]

@pytest.mark.parametrize("failure", ["reset", "dt", "schema"])
def test_load_sim_closes_unreturned_adapter(tmp_path, monkeypatch, failure):
    import specola.cli_common as common
    import specola.conformance.authoring as authoring

    adapter = FakeAdapter()
    closed = []
    adapter.close_render = lambda: closed.append(True)
    monkeypatch.setattr(common, "construct_sim", lambda *a, **kw: adapter)
    original = ValueError("injected reset failure")

    def reset():
        raise original

    if failure == "reset":
        monkeypatch.setattr(adapter, "reset", reset)
    if failure == "schema":
        monkeypatch.setattr(authoring, "prepare_schema_authoring",
                            lambda _: {"subject": {}, "missing_arguments": []})
    expected = authoring.SchemaOnlyDiscovery if failure == "schema" else ValueError
    with pytest.raises(expected) as caught:
        common.load_sim("mock", dt_override=-1 if failure == "dt" else None,
                        allow_schema_only=failure == "schema")
    if failure == "reset":
        assert caught.value is original
    assert closed == [True]

@pytest.mark.parametrize("entry", ["construct_sim", "load_sim"])
def test_construct_sim_registers_close_for_interpreter_exit(monkeypatch, entry):
    """Every CLI releases through one place: run builds through
    construct_sim, the other three through load_sim on top of it, and the
    adapter is closed at exit even when the caller never calls close
    (check and write-specs did not, and robosuite's EGL teardown raised)."""
    import atexit
    import specola.cli_common as common

    adapter = FakeAdapter()
    registered = []
    monkeypatch.setattr(common, "_load_sim", lambda *a, **kw: adapter)
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **kw: registered.append(fn))
    result = getattr(common, entry)("mock")
    returned = result if entry == "construct_sim" else result[0]
    assert returned is adapter
    assert [fn.__self__ for fn in registered] == [adapter]
    registered[0]()
    registered[0]()                     # idempotent after the caller's own close
    assert adapter._closed is True


def test_abandoned_worker_keeps_adapter_open_through_interpreter_exit(tmp_path):
    """A run whose simulator worker is abandoned leaves the adapter to that
    worker; the exit-time close construct_sim registered must not release
    it under the running step (external review, reproduced at exit)."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    child = f"""
import sys, threading
import specola.cli_common as common
import specola.run_monitor as run
from specola.adapters.fake import FakeAdapter
from specola.run_dashboard import WorkerAbandoned

worker = None
adapter = FakeAdapter()
adapter.close_native_extraction = lambda: print(
    "CLOSED worker_alive=" + str(worker is not None and worker.is_alive()), flush=True)
common._load_sim = lambda *args, **kwargs: adapter

def abandoned(*args, **kwargs):
    global worker
    worker = threading.Thread(target=threading.Event().wait, daemon=True)
    worker.start()
    raise WorkerAbandoned("injected worker timeout")

run._typed_eval_loop = abandoned
sys.argv = ["run", "--sim", "mock", "--specs", {str(repo / "examples" / "mock_specs.yaml")!r},
            "--steps", "1", "--ui", "plain", "--output-dir", {str(tmp_path / "results")!r}]
try:
    run.main()
except WorkerAbandoned:
    print("ABANDONED", flush=True)
"""
    completed = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120)
    assert "ABANDONED" in completed.stdout, completed.stdout + completed.stderr
    assert "CLOSED" not in completed.stdout, completed.stdout


@pytest.mark.parametrize("schema_only", [False, True])
def test_discover_closes_adapter_before_json_return(tmp_path, monkeypatch, schema_only):
    import sys
    import specola.cli_common as common
    import specola.discover as discover
    import specola.conformance.authoring as authoring

    adapter = FakeAdapter()
    closed = []
    adapter.close_render = lambda: closed.append(True)
    monkeypatch.setattr(common, "construct_sim", lambda *a, **kw: adapter)
    if schema_only:
        monkeypatch.setattr(authoring, "prepare_schema_authoring",
                            lambda _: {"subject": {}, "missing_arguments": []})
    monkeypatch.setattr(sys, "argv", ["discover", "--sim", "mock",
                        "--format", "json", "--output", str(tmp_path / "scene.json")])
    discover.main()
    assert closed == [True]
    assert (tmp_path / "scene.json").is_file()


def test_discover_closes_adapter_when_post_load_metadata_fails(monkeypatch):
    import sys
    import specola.cli_common as common
    import specola.discover as discover

    adapter = FakeAdapter()
    closed = []
    adapter.close_render = lambda: closed.append(True)
    monkeypatch.setattr(common, "construct_sim", lambda *a, **kw: adapter)
    monkeypatch.setattr(common, "load_sim", lambda *a, **kw: (adapter, 0.05, 3, {}, []))
    failure = RuntimeError("injected metadata failure")

    def metadata(_):
        raise failure

    monkeypatch.setattr(common, "_task_description", metadata)
    monkeypatch.setattr(sys, "argv", ["discover", "--sim", "mock", "--format", "json"])
    with pytest.raises(RuntimeError) as caught:
        discover.main()
    assert caught.value is failure
    assert closed == [True]
