"""Tests for the bounded bundled-adapter entry-point custody record."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from specola import registry
from specola.conformance import authority


ROOT = Path(__file__).parents[1].resolve()
_REAL_ITER_ENTRY_POINTS = authority._iter_entry_points


class _Distribution:
    name = "specola"
    version = "0.2.0"
    metadata = {"Name": "specola"}

    def __init__(self, direct_url=None, entry_points=()):
        self._direct_url = direct_url
        self.entry_points = tuple(entry_points)

    def read_text(self, name):
        if name == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None


class _EntryPoint:
    group = registry.ENTRY_POINT_GROUP

    def __init__(
        self,
        name="mock",
        value="specola.adapters.fake",
        loaded=None,
        factory=None,
        direct_url=None,
    ):
        self.name = name
        self.value = value
        self.dist = _Distribution(direct_url)
        self._loaded = loaded
        self._factory = factory

    def load(self):
        if self._loaded is None:
            return __import__(self.value.split(":", 1)[0], fromlist=["*"])
        sys.modules[self.value.split(":", 1)[0]] = self._loaded
        if self._factory is not None:
            registry.register(self.name, self._factory)
        return self._loaded


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    snapshot = dict(registry._REGISTRY)
    previous_module = sys.modules.get("specola.adapters.fake")
    sys.modules.pop("specola.adapters.fake", None)
    registry._REGISTRY.clear()
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint()])
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)
        if previous_module is None:
            sys.modules.pop("specola.adapters.fake", None)
        else:
            sys.modules["specola.adapters.fake"] = previous_module


def test_resolves_normal_entry_point_and_fresh_registry_factory():
    record = authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))

    assert record.entry_point_name == "mock"
    assert record.entry_point_value == "specola.adapters.fake"
    assert record.entry_point_module == "specola.adapters.fake"
    assert Path(record.entry_point_source_path).is_file()
    assert len(record.entry_point_source_sha256) == 64
    assert record.distribution_name == "specola"
    assert record.distribution_version == "0.2.0"
    assert len(record.project_source_sha256) == 64
    assert record.factory is registry.resolve("mock").factory

    validated = authority.validate_entry_point_authority(record)
    assert validated is not record
    assert validated.factory is registry.resolve("mock").factory


def test_canonical_json_contains_durable_facts_only():
    record = authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))

    assert record.to_json() == json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
    assert "factory" not in record.to_json()


def test_duplicate_installed_names_fail_closed(monkeypatch):
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint(), _EntryPoint()])

    with pytest.raises(authority.EntryPointAuthorityError, match="exactly one"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_missing_or_wrong_installed_name_fails_closed(monkeypatch):
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint(name="dataset")])

    with pytest.raises(authority.EntryPointAuthorityError, match="exactly one"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_selected_installed_name_must_be_declared_in_project(monkeypatch):
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint(name="unlisted")])

    with pytest.raises(authority.EntryPointAuthorityError, match="no adapter entry point"):
        authority.resolve_entry_point_authority("unlisted", repository_root=str(ROOT))


def test_missing_registry_registration_fails_closed(monkeypatch):
    loaded = types.ModuleType("specola.adapters.fake")
    loaded.__file__ = str(ROOT / "src/specola/adapters/fake.py")
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint(loaded=loaded)])

    with pytest.raises(authority.EntryPointAuthorityError, match="did not register"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_wrong_root_module_fails_closed(monkeypatch, tmp_path):
    loaded = types.ModuleType("specola.adapters.fake")
    loaded.__file__ = str(tmp_path / "ambient.py")
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [_EntryPoint(loaded=loaded)])

    with pytest.raises(authority.EntryPointAuthorityError, match="outside repository root"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_wrong_distribution_fails_closed(monkeypatch):
    entry_point = _EntryPoint()
    entry_point.dist.metadata = {"Name": "ambient-package"}
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [entry_point])

    with pytest.raises(authority.EntryPointAuthorityError, match="distribution mismatch"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_python38_entry_point_without_dist_uses_owned_distribution(monkeypatch):
    entry_point = _EntryPoint()
    delattr(entry_point, "dist")
    distribution = _Distribution(entry_points=(entry_point,))
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [entry_point])
    monkeypatch.setattr(
        authority.metadata,
        "distribution",
        lambda name: distribution,
    )

    record = authority.resolve_entry_point_authority(
        "mock", repository_root=str(ROOT)
    )

    assert record.distribution_name == "specola"
    assert record.distribution_version == "0.2.0"


@pytest.mark.parametrize("owned", [(), (_EntryPoint(value="specola.adapters.dataset"),)])
def test_python38_distribution_fallback_requires_exact_entry_point_ownership(
    monkeypatch, owned
):
    entry_point = _EntryPoint()
    delattr(entry_point, "dist")
    distribution = _Distribution(entry_points=owned)
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [entry_point])
    monkeypatch.setattr(
        authority.metadata,
        "distribution",
        lambda name: distribution,
    )

    with pytest.raises(authority.EntryPointAuthorityError, match="not uniquely owned"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_installed_entry_point_value_must_match_project_declaration(monkeypatch):
    monkeypatch.setattr(
        authority,
        "_iter_entry_points",
        lambda: [_EntryPoint(value="specola.adapters.fake:_make_mock")],
    )

    with pytest.raises(authority.EntryPointAuthorityError, match="differs from project declaration"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


def test_editable_distribution_root_mismatch_fails_closed(monkeypatch, tmp_path):
    entry_point = _EntryPoint(
        direct_url={"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    )
    monkeypatch.setattr(authority, "_iter_entry_points", lambda: [entry_point])

    with pytest.raises(authority.EntryPointAuthorityError, match="editable distribution root mismatch"):
        authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_point_value", "ambient.module"),
        ("entry_point_source_sha256", "0" * 64),
        ("distribution_name", "ambient-package"),
    ],
)
def test_durable_fact_drift_is_rejected(field, value):
    record = authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))
    forged = replace(record, **{field: value})

    with pytest.raises(authority.EntryPointAuthorityError, match="field drift"):
        authority.validate_entry_point_authority(forged)


def test_fresh_import_replaces_cached_wrong_registration():
    module = __import__("specola.adapters.fake", fromlist=["*"])
    real = registry.get("mock")
    assert real is not None
    poisoned = lambda *_args: "wrong"
    registry._REGISTRY["mock"] = replace(real, factory=poisoned)

    record = authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))

    assert record.factory is registry.resolve("mock").factory
    assert record.factory is not poisoned
    assert not isinstance(record.factory(None, None, False), str)
    assert sys.modules["specola.adapters.fake"] is not module


def test_real_installed_mock_entry_point(monkeypatch):
    """Exercise actual importlib.metadata when this checkout is installed."""

    real = [entry for entry in _REAL_ITER_ENTRY_POINTS() if entry.name == "mock"]
    if len(real) != 1:
        pytest.skip("installed specola mock entry point unavailable")
    module_name = real[0].value.split(":", 1)[0]
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        pytest.skip("installed entry-point module source unavailable")
    assert module_spec is not None
    origin = module_spec.origin
    if origin is None:
        pytest.skip("installed entry-point module source unavailable")
    assert origin is not None
    source = Path(origin).resolve()
    root = next((parent for parent in source.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        pytest.skip("installed entry-point project root unavailable")
    monkeypatch.setattr(authority, "_iter_entry_points", _REAL_ITER_ENTRY_POINTS)

    record = authority.resolve_entry_point_authority("mock", repository_root=str(root))

    assert record.distribution_name == "specola"
    assert record.entry_point_module == module_name


def _decorator(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


@_decorator
def _decorated_factory(*_args):
    return None


class _CallableFactory:
    def __call__(self, *_args):
        return None


@pytest.mark.parametrize(
    "factory",
    [
        _CallableFactory(),
        functools.partial(_decorated_factory),
        _decorated_factory,
        len,
    ],
)
def test_registered_callable_forms_are_not_inspected(monkeypatch, factory):
    loaded = types.ModuleType("specola.adapters.fake")
    loaded.__file__ = str(ROOT / "src/specola/adapters/fake.py")
    monkeypatch.setattr(
        authority,
        "_iter_entry_points",
        lambda: [_EntryPoint(loaded=loaded, factory=factory)],
    )

    record = authority.resolve_entry_point_authority("mock", repository_root=str(ROOT))

    assert record.factory is factory


def test_wrong_record_type_is_rejected():
    with pytest.raises(authority.EntryPointAuthorityError, match="wrong type"):
        authority.validate_entry_point_authority({})  # type: ignore
