"""Sim-free tests for the isaaclab factory's Kit app ownership (issue #133).

Every CLI surface on ``--sim isaaclab`` failed with ``No module named
'carb'`` because nothing launched the Kit app before ``isaaclab_tasks`` was
imported. The factory now launches one when none is live and leaves a
caller's app alone. Kit, Isaac Lab and gymnasium are stubbed here, so this
runs in any env; the live check is ``specola discover --sim isaaclab``.
"""

from __future__ import annotations

import sys
import types

import pytest

from specola.adapters import isaaclab


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.mark.parametrize("live", [False, True])
def test_factory_launches_kit_only_when_no_app_is_live(monkeypatch, live):
    events = []

    class _Env:
        def reset(self):
            events.append("reset")

    parse_env_cfg = lambda task, num_envs: events.append(("cfg", task, num_envs))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", _module("isaaclab_tasks"))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        _module("isaaclab_tasks.utils", parse_env_cfg=parse_env_cfg),
    )
    monkeypatch.setitem(
        sys.modules, "gymnasium", _module("gymnasium", make=lambda task, cfg: _Env())
    )

    monkeypatch.setattr(isaaclab, "_kit_app_running", lambda: live)
    monkeypatch.setattr(isaaclab, "_launch_kit_app", lambda: events.append("launch"))
    monkeypatch.setattr(isaaclab, "IsaacLabAdapter", lambda env, env_index: env_index)

    assert isaaclab._make_isaaclab("Isaac-Forge-PegInsert-Direct-v0", None, True) == 0

    # The launch precedes the task registry: parse_env_cfg needs Kit's modules.
    launched = [] if live else ["launch"]
    assert events == launched + [("cfg", "Isaac-Forge-PegInsert-Direct-v0", 1), "reset"]


def test_launch_hides_cli_argv_from_kit_and_restores_stdout(monkeypatch):
    import atexit
    import signal

    class _App:
        def close(self):
            pass

    seen = {}

    class _AppLauncher:
        def __init__(self, **kwargs):
            seen["argv"], seen["kwargs"] = list(sys.argv), kwargs
            # What Isaac Lab 3.0's AppLauncher does to the host process.
            sys.argv += kwargs["kit_args"].split()
            sys.stdout = sys.__stdout__
            self.app = _App()

    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(
        sys.modules, "isaaclab.app", _module("isaaclab.app", AppLauncher=_AppLauncher)
    )
    exit_hooks, handlers = [], {}
    monkeypatch.setattr(atexit, "register", exit_hooks.append)
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)
    monkeypatch.setattr(sys, "argv", ["specola", "discover", "--sim", "isaaclab"])
    redirected = object()
    monkeypatch.setattr(sys, "stdout", redirected)

    app = isaaclab._launch_kit_app()

    assert seen["argv"] == ["specola"]
    assert seen["kwargs"]["headless"] is True
    assert "--/app/hangDetector/enabled=false" in seen["kwargs"]["kit_args"]
    assert sys.argv == ["specola", "discover", "--sim", "isaaclab"]
    assert sys.stdout is redirected
    assert exit_hooks == [app.close]
    assert handlers == {signal.SIGINT: signal.default_int_handler}
