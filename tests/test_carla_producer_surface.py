"""The CARLA client inventory is independent of adapter-used actor accessors."""

import ast
import importlib.metadata
from pathlib import Path

import pytest

from specola.conformance.surface import SurfaceValidationError, validate_known_surface


@pytest.fixture
def native():
    return pytest.importorskip("carla")


def _capture():
    from benchmarks.adapter_fidelity.generate_producer_surface import capture_carla
    return capture_carla()


def test_client_stubs_and_runtime_exports_are_independent_populations(native):
    snapshot = _capture()
    distribution = importlib.metadata.distribution("carla")
    installed_files = distribution.files
    assert installed_files is not None
    stub_files = [p for p in installed_files if str(p).endswith(".pyi")]
    assert stub_files
    members = {m.member_id for m in snapshot.members}
    for relative in stub_files:
        path = Path(str(distribution.locate_file(relative)))
        root = "carla.stub." + Path(str(relative)).with_suffix("").as_posix().replace("/", ".")
        declared_classes = {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
        assert {root + "." + name for name in declared_classes} <= members
    assert {m.name for m in snapshot.members if m.root == "carla.python"} == {
        name for name in vars(native) if not name.startswith("_")
    }
    assert "carla.python.Actor.get_acceleration" in members
    assert "carla.python.World.cast_ray" in members


def test_unused_client_query_fails_without_connecting(native, monkeypatch):
    before = _capture()

    def unused_query(world):
        raise AssertionError("inventory must not query a world")

    monkeypatch.setattr(native, "unused_read_surface_query", unused_query, raising=False)
    with pytest.raises(SurfaceValidationError, match="added=.*unused_read_surface_query"):
        validate_known_surface(before, _capture())


def test_client_capture_is_repeatable(native):
    validate_known_surface(_capture(), _capture())
