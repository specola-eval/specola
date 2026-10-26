"""
The CLI keeps a failing exit status when a Kit app (Isaac Sim) is live.

Kit's fast shutdown, reached through the ``SimulationApp.close`` the
Isaac Lab and RoboDojo factories register at exit, ends the process with
status 0. ``specola.cli.main`` ends a nonzero exit itself first. No
simulator is needed: ``omni.kit.app`` is a stub in ``sys.modules`` and
``os._exit`` is recorded instead of called.
"""

from __future__ import annotations

import sys
import types

import pytest

from specola import cli


class _Exited(Exception):
    pass


def _raise(exc):
    def dispatch():
        raise exc

    return dispatch


@pytest.fixture
def exits(monkeypatch):
    codes = []

    def fake_exit(code):
        codes.append(code)
        raise _Exited

    monkeypatch.setattr(cli.os, "_exit", fake_exit)
    return codes


def _kit(monkeypatch, running):
    app = types.SimpleNamespace(is_running=lambda: running)
    module = types.SimpleNamespace(get_app=lambda: app)
    monkeypatch.setitem(sys.modules, "omni.kit.app", module)


@pytest.mark.parametrize(
    "exc, code",
    [
        (SystemExit(1), 1),
        (SystemExit(2), 2),
        (SystemExit("refused: no replay"), 1),
        (RuntimeError("boom"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_live_kit_app_ends_with_the_failing_code(monkeypatch, exits, capsys, exc, code):
    _kit(monkeypatch, running=True)
    monkeypatch.setattr(cli, "_dispatch", _raise(exc))
    with pytest.raises(_Exited):
        cli.main()
    assert exits == [code]
    if isinstance(exc, SystemExit) and isinstance(exc.code, str):
        assert "refused: no replay" in capsys.readouterr().err
    if isinstance(exc, RuntimeError):
        assert "RuntimeError: boom" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [None, SystemExit(0), SystemExit(None)])
def test_live_kit_app_leaves_success_to_kits_close(monkeypatch, exits, exc):
    _kit(monkeypatch, running=True)
    monkeypatch.setattr(cli, "_dispatch", _raise(exc) if exc else lambda: None)
    if exc is None:
        cli.main()
    else:
        with pytest.raises(SystemExit):
            cli.main()
    assert exits == []


@pytest.mark.parametrize("kit", ["absent", "stopped"])
def test_without_a_live_app_the_exit_is_untouched(monkeypatch, exits, kit):
    if kit == "absent":
        monkeypatch.delitem(sys.modules, "omni.kit.app", raising=False)
    else:
        _kit(monkeypatch, running=False)
    monkeypatch.setattr(cli, "_dispatch", _raise(SystemExit(1)))
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    assert exits == []
