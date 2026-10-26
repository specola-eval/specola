"""Pager: page on TTY, stay out of the way everywhere else."""

from __future__ import annotations

import signal
import subprocess
import sys

from specola import pager


def test_should_page_false_when_disabled(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert pager.should_page(disable=True) is False


def test_should_page_false_when_not_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert pager.should_page(disable=False) is False


def test_pager_command_defaults_to_less(monkeypatch):
    monkeypatch.delenv("PAGER", raising=False)
    assert pager.pager_command() == ["less", "-FRXS"]


def test_pager_command_respects_env(monkeypatch):
    monkeypatch.setenv("PAGER", "more -d")
    assert pager.pager_command() == ["more", "-d"]


def test_maybe_page_passthrough_when_not_paging(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    with pager.maybe_page():
        print("hello")
    assert capsys.readouterr().out == "hello\n"


def test_maybe_page_falls_back_when_pager_missing(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("PAGER", "definitely-not-a-real-pager-xyz")
    with pager.maybe_page():
        print("fallback")
    assert "fallback" in capsys.readouterr().out


def test_maybe_page_ignores_sigint_while_pager_runs(monkeypatch):
    # Ctrl-C belongs to the pager (less interrupts searches with it);
    # the parent must not die with a KeyboardInterrupt traceback.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    seen = {}

    def fake_run(command, **kwargs):
        seen["handler"] = signal.getsignal(signal.SIGINT)

    monkeypatch.setattr(pager.subprocess, "run", fake_run)
    before = signal.getsignal(signal.SIGINT)
    with pager.maybe_page():
        print("paged")
    assert seen["handler"] is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) is before  # restored after


def test_discover_cli_accepts_no_pager_flag():
    proc = subprocess.run(
        [sys.executable, "-m", "specola.discover",
         "--sim", "mock", "--no-pager"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "Entities" in proc.stdout
