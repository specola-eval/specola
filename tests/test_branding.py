"""Branding appears only in interactive terminals, and `specola doctor`
reports adapter readiness from declared requirements."""

from __future__ import annotations

import io
import sys

import pytest

from specola import branding, cli, doctor, registry
from specola._brand_art import ICON_8, ICON_10


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("NO_COLOR", "SPECOLA_THEME", "COLORFGBG", "TERM"):
        monkeypatch.delenv(name, raising=False)


def test_enabled_only_for_a_terminal_without_no_color(clean_env, monkeypatch):
    assert branding.enabled(_Tty())
    assert not branding.enabled(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not branding.enabled(_Tty())


@pytest.mark.parametrize("encoding, expected", [
    ("latin-1", False), ("utf-16", False), ("no-such-codec", False),
    ("UTF8", True), ("cp65001", True),
])
def test_enabled_only_for_a_utf8_terminal(clean_env, encoding, expected):
    class _EncodedTty(_Tty):
        pass

    _EncodedTty.encoding = encoding
    assert branding.enabled(_EncodedTty()) is expected


def test_disabled_for_a_dumb_terminal(clean_env, monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert not branding.enabled(_Tty())


def test_unknown_background_uses_the_terminal_foreground(clean_env):
    assert branding.brand_color() == "default"


@pytest.mark.parametrize("env, expected", [
    ({"SPECOLA_THEME": "light"}, "light"),
    ({"SPECOLA_THEME": "dark", "COLORFGBG": "0;15"}, "dark"),
    ({"COLORFGBG": "15;0"}, "dark"),
    ({"COLORFGBG": "0;15"}, "light"),
    ({"COLORFGBG": "0;7"}, "light"),
    ({"COLORFGBG": "15;16"}, "unknown"),
    ({"COLORFGBG": "15;232"}, "unknown"),
    ({"COLORFGBG": "garbage"}, "unknown"),
    ({}, "unknown"),
])
def test_theme(clean_env, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert branding.theme() == expected
    assert branding.brand_color() == branding.COLORS[expected]


def test_icon_art_has_rule_row_and_even_width():
    from rich.cells import cell_len

    for art in (ICON_8, ICON_10):
        widths = {cell_len(line) for line in art}
        assert len(widths) == 1
        assert sum("○" in line for line in art) == 1


def test_version_is_plain_when_piped(clean_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["specola", "--version"])
    cli.main()
    out = capsys.readouterr().out
    assert out.strip() == f"specola {branding.__version__}"


def test_bare_specola_prints_usage_without_banner_when_piped(clean_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["specola"])
    cli.main()
    out = capsys.readouterr().out
    assert out.startswith("usage: specola")
    assert branding.MARK not in out


def test_register_records_requires_and_status_reports_missing(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_iter_adapter_entry_points", lambda: [])
    registry.register("present", factory=lambda *a: None, requires=("json",))
    registry.register("absent", factory=lambda *a: None,
                      requires=("json", "no_such_module_xyz"), install_hint="pip install xyz")
    registry.register("bare", factory=lambda *a: None)
    status = {s.name: s for s in registry.adapter_status()}
    assert status["present"].ready and status["present"].missing == ()
    assert not status["absent"].ready
    assert status["absent"].missing == ("no_such_module_xyz",)
    assert status["bare"].ready and status["bare"].requires == ()


def test_status_reports_an_adapter_that_fails_to_load(monkeypatch):
    class _Broken:
        name = "broken"
        value = "nowhere.broken"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_iter_adapter_entry_points", lambda: [_Broken()])
    (status,) = registry.adapter_status()
    assert status.name == "broken" and not status.ready
    assert "ImportError: boom" in status.load_error


def test_doctor_plain_output_lists_each_adapter(monkeypatch, capsys, clean_env):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_iter_adapter_entry_points", lambda: [])
    registry.register("present", factory=lambda *a: None, requires=("json",))
    registry.register("absent", factory=lambda *a: None, requires=("no_such_module_xyz",))
    monkeypatch.setattr(sys, "argv", ["specola doctor"])
    doctor.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"specola {branding.__version__}")
    assert any(line.split() == ["present", "packages", "found"] for line in out)
    assert any(line.split() == ["absent", "missing", "no_such_module_xyz"] for line in out)


def test_bundled_adapters_declare_requirements():
    names = {s.name: s for s in registry.adapter_status()}
    expected_empty = {"mock", "dataset"}
    for name, status in names.items():
        if name in expected_empty or status.load_error:
            continue
        assert status.requires, f"{name} declares no simulator packages"


def test_lockup_and_panel_render(clean_env):
    from rich.console import Console

    env = branding.Environment(version="9.9.9", python="3.11.0", adapters=[], extras=())
    for renderable in (branding.lockup(env), branding.panel(env), branding.version_line("9.9.9")):
        console = Console(file=io.StringIO(), width=100, color_system=None)
        console.print(renderable)
        text = console.file.getvalue()
        assert "9.9.9" in text or "S p e c o l a" in text


def test_third_party_adapters_are_not_imported_when_asked(monkeypatch):
    class _ThirdParty:
        name = "external"
        value = "someone_else.adapter"

        def load(self):
            raise AssertionError("third-party module must not be imported")

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_iter_adapter_entry_points", lambda: [_ThirdParty()])
    (status,) = registry.adapter_status(load_third_party=False)
    assert status.name == "external" and not status.checked and not status.ready


def test_bare_specola_header_does_not_repeat_the_name(monkeypatch, capsys):
    from specola import cli_common

    monkeypatch.setattr(branding, "enabled", lambda stream=None: True)
    printed = []
    monkeypatch.setattr(branding, "print_branded", lambda r, stream=None: printed.append(r.plain))
    with pytest.raises(Exception):
        cli_common.construct_sim("no-such-adapter", header="Specola")
    assert printed == [f"{branding.MARK} {branding.NAME}"]


@pytest.mark.parametrize("columns, expected", [("100", "full"), ("30", "narrow")])
def test_narrow_terminal_gets_the_fallback(clean_env, monkeypatch, columns, expected):
    from rich.text import Text

    monkeypatch.setenv("COLUMNS", columns)
    env = branding.Environment(version="9.9.9", python="3.11.0", adapters=[], extras=())
    stream = io.StringIO()
    branding.print_branded(branding.lockup(env), stream=stream, narrow=Text("narrow"))
    out = stream.getvalue()
    assert ("narrow" in out) == (expected == "narrow")
