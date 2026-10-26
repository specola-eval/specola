"""
Adapter registration and discovery contract.

Primary requirement (design spec §5.3, decision on 2026-07-14):
adding a new simulator must require only a new adapter module plus
registration — no changes to the parser, monitor, binding logic,
component resolver, runner, or shared CLI code.

Covers:
- duplicate-name rejection unless replacement is explicit;
- discovery through ``specola.adapters`` entry points (3.8-compatible);
- discovery through an explicitly loaded adapter module (--adapter-module);
- construction of a discovered adapter without any hardcoded name table.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from specola import registry
from specola.adapter import SimulatorAdapter


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the process-wide adapter registry around each test."""
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)


def _dummy_factory(task, robot, target_only):
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Duplicate registration
# ---------------------------------------------------------------------------

def test_register_rejects_duplicate_name():
    registry.register(name="dupsim", factory=_dummy_factory)

    with pytest.raises(ValueError, match="dupsim"):
        registry.register(name="dupsim", factory=_dummy_factory)


def test_register_allows_explicit_replacement():
    registry.register(name="dupsim", factory=_dummy_factory)

    def other_factory(task, robot, target_only):
        raise NotImplementedError

    registry.register(
        name="dupsim",
        factory=other_factory,
        replace=True,
    )

    assert registry.get("dupsim").factory is other_factory


# ---------------------------------------------------------------------------
# Explicit adapter-module loading (the --adapter-module development path)
# ---------------------------------------------------------------------------

ADAPTER_MODULE_SOURCE = textwrap.dedent(
    """
    from specola.cli_common import _MockAdapter
    from specola.registry import register


    def _make(task, robot, target_only):
        return _MockAdapter()


    register(
        name="tempsim",
        factory=_make,
        description="temporary test simulator",
    )
    """
)


@pytest.fixture
def temp_adapter_module(tmp_path, monkeypatch):
    module_dir = tmp_path / "temp_adapter_pkg"
    module_dir.mkdir()
    (module_dir / "temp_adapter.py").write_text(ADAPTER_MODULE_SOURCE)
    monkeypatch.syspath_prepend(str(module_dir))
    # Each test writes the module to a fresh tmp_path; drop any cached
    # import so module-level registration runs again.
    sys.modules.pop("temp_adapter", None)
    yield "temp_adapter"
    sys.modules.pop("temp_adapter", None)


def test_load_adapter_module_registers_adapter(temp_adapter_module):
    registry.load_adapter_module(temp_adapter_module)

    entry = registry.get("tempsim")
    assert entry is not None
    assert entry.description == "temporary test simulator"


def test_loaded_module_adapter_constructs_via_shared_cli(temp_adapter_module):
    """A registered test adapter is usable without editing shared CLI code."""
    from specola.cli_common import _load_sim

    registry.load_adapter_module(temp_adapter_module)

    adapter = _load_sim("tempsim")
    assert isinstance(adapter, SimulatorAdapter)
    observation = adapter.reset()
    assert observation.perceived_entities


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------

class _FakeEntryPoint:
    """importlib.metadata.EntryPoint stand-in with the same read surface."""

    def __init__(self, name, module_name):
        self.name = name
        self.value = module_name
        self.group = registry.ENTRY_POINT_GROUP

    def load(self):
        import importlib

        return importlib.import_module(self.value)


def test_resolve_discovers_adapter_through_entry_points(
    temp_adapter_module, monkeypatch
):
    monkeypatch.setattr(
        registry,
        "_iter_adapter_entry_points",
        lambda: [_FakeEntryPoint("tempsim", temp_adapter_module)],
    )

    entry = registry.resolve("tempsim")
    assert entry.name == "tempsim"


def test_available_includes_unloaded_entry_point_names(
    temp_adapter_module, monkeypatch
):
    monkeypatch.setattr(
        registry,
        "_iter_adapter_entry_points",
        lambda: [_FakeEntryPoint("tempsim", temp_adapter_module)],
    )

    assert "tempsim" in registry.available()


def test_resolve_unknown_name_lists_available(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_iter_adapter_entry_points",
        lambda: [],
    )
    registry.register(name="onlysim", factory=_dummy_factory)

    with pytest.raises(ValueError) as excinfo:
        registry.resolve("nosuchsim")

    assert "nosuchsim" in str(excinfo.value)
    assert "onlysim" in str(excinfo.value)


def test_installed_metadata_exposes_bundled_adapters():
    """The installed distribution declares the bundled adapters as entry points.

    Discovery must work from package metadata alone — no hardcoded
    name→module table anywhere in the CLI.
    """
    names = {ep.name for ep in registry._iter_adapter_entry_points()}
    assert {"mock", "robocasa", "libero"} <= names


# ---------------------------------------------------------------------------
# No hardcoded adapter-name table
# ---------------------------------------------------------------------------

def test_cli_common_has_no_hardcoded_adapter_table():
    import specola.cli_common as cli_common

    assert not hasattr(cli_common, "_BUILTIN_ADAPTER_MODULES")
    assert not hasattr(cli_common, "_import_builtin_adapter")
