"""Adapter auto-detection: registry.detect() and wrap_env()."""
import pytest

from specola import registry
from specola.adapters.fake import _MockAdapter


class _FakeLiberoEnv:
    pass


class _FakeMetaworldEnv:
    pass


# Spoof module paths — detection matches on type(env).__module__.
_FakeLiberoEnv.__module__ = "libero.libero.envs.env_wrapper"
_FakeMetaworldEnv.__module__ = "metaworld.envs.mujoco.sawyer_xyz"


@pytest.fixture
def scratch_registry(monkeypatch):
    """Isolated registry so tests never depend on installed entry points."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_iter_adapter_entry_points", lambda: [])
    return registry


def _register_fake(reg, name, prefix):
    reg.register(
        name=name,
        factory=lambda task, robot, target_only: _MockAdapter(),
        handles=lambda obj: (type(obj).__module__ or "").startswith(prefix),
        wrap=lambda env, **kw: _MockAdapter(),
    )


def test_detect_single_match(scratch_registry):
    _register_fake(scratch_registry, "fakelibero", "libero.")
    entry = scratch_registry.detect(_FakeLiberoEnv())
    assert entry.name == "fakelibero"


def test_detect_no_match_lists_candidates(scratch_registry):
    _register_fake(scratch_registry, "fakelibero", "libero.")
    with pytest.raises(registry.NoAdapterMatch) as excinfo:
        scratch_registry.detect(object())
    assert "fakelibero" in str(excinfo.value)
    assert "explicit" in str(excinfo.value).lower()  # escape-hatch hint


def test_detect_ambiguous_raises(scratch_registry):
    _register_fake(scratch_registry, "a1", "libero.")
    _register_fake(scratch_registry, "a2", "libero.")
    with pytest.raises(registry.AmbiguousAdapterMatch) as excinfo:
        scratch_registry.detect(_FakeLiberoEnv())
    assert "a1" in str(excinfo.value) and "a2" in str(excinfo.value)


def test_detect_walks_env_chain_outermost_first(scratch_registry):
    """A wrapper chain matches at the outermost link that has a handler."""
    _register_fake(scratch_registry, "fakelibero", "libero.")
    _register_fake(scratch_registry, "fakemw", "metaworld")
    outer = _FakeLiberoEnv()
    outer.env = _FakeMetaworldEnv()  # inner link would also match
    assert scratch_registry.detect(outer).name == "fakelibero"


def test_detect_skips_entries_without_handles(scratch_registry):
    scratch_registry.register(
        name="nohandles",
        factory=lambda task, robot, target_only: _MockAdapter(),
    )
    with pytest.raises(registry.NoAdapterMatch):
        scratch_registry.detect(_FakeLiberoEnv())


def test_wrap_env_constructs_adapter(scratch_registry):
    _register_fake(scratch_registry, "fakelibero", "libero.")
    adapter = scratch_registry.wrap_env(_FakeLiberoEnv())
    assert isinstance(adapter, _MockAdapter)


def test_wrap_env_without_wrap_callable_errors(scratch_registry):
    scratch_registry.register(
        name="nowrap",
        factory=lambda task, robot, target_only: _MockAdapter(),
        handles=lambda obj: True,
    )
    with pytest.raises(ValueError, match="wrap"):
        scratch_registry.wrap_env(_FakeLiberoEnv())


def test_bundled_adapters_register_detection():
    """Each live-sim adapter module registers handles matching its sim's
    env module prefix. Imports the adapter modules (light — sims are
    imported lazily inside factories), not the sims."""
    import specola.adapters.libero      # noqa: F401
    import specola.adapters.robocasa    # noqa: F401
    import specola.adapters.metaworld   # noqa: F401
    import specola.adapters.dmcontrol   # noqa: F401

    cases = {
        "libero":    "libero.libero.envs.env_wrapper",
        "robocasa":  "robocasa.environments.kitchen.kitchen",
        "metaworld": "metaworld.envs.mujoco.sawyer_xyz.v2",
        "dmcontrol": "dm_control.rl.control",
    }
    for name, module in cases.items():
        entry = registry.get(name)
        assert entry is not None, f"{name} not registered"
        assert entry.handles is not None, f"{name} has no handles"
        assert entry.wrap is not None, f"{name} has no wrap"
        probe = type("Probe", (), {})()
        type(probe).__module__ = module
        assert entry.handles(probe), f"{name} rejected {module}"
        assert not entry.handles(object()), f"{name} claimed plain object"
