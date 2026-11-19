"""Reviewed profile policy diagnostics preserve native integrity failures."""

from importlib import metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from specola.adapters import mujoco_native
from specola.conformance.native_images import NativeImageError


@pytest.mark.parametrize("version, expected", [
    ("3.8.1", "3.12"), ("3.3.0", "3.11"),
    ("3.3.1", "3.11"), ("3.10.0", "3.11"), ("0.0.0", None),
])
def test_profile_python_tag_reads_the_reviewed_wheel_abi(version, expected):
    assert mujoco_native.profile_python_tag(version) == expected


def test_profile_python_tag_checks_the_complete_image_population(monkeypatch, tmp_path):
    profile = {"images": [
        {"relative_path": "mujoco/libmujoco.so"},
        {"relative_path": "mujoco/_first.cpython-312-x86_64-linux-gnu.so"},
        {"relative_path": "mujoco/_last.cpython-311-x86_64-linux-gnu.so"},
    ]}
    (tmp_path / "mujoco-native-test.json").write_text(json.dumps(profile))
    monkeypatch.setattr(mujoco_native, "_PROFILE_DIR", tmp_path)
    with pytest.raises(NativeImageError, match="inconsistent CPython ABI tags"):
        mujoco_native.profile_python_tag("test")


@pytest.mark.parametrize("running", [(3, 10), (3, 12)])
def test_population_message_distinguishes_python_and_image_mismatches(monkeypatch, running):
    monkeypatch.setattr(mujoco_native, "sys", SimpleNamespace(version_info=running))
    message = mujoco_native.population_mismatch_message("3.8.1")
    if running == (3, 10):
        assert "Specola" in message
        assert "CPython 3.12" in message
        assert "running 3.10" in message
        assert "conda create -n specola-libero python=3.12" in message
        assert "docs/install.md" in message
    else:
        assert message == "MuJoCo distribution native image population differs"


def test_preflight_leaves_missing_mujoco_to_existing_install_errors(monkeypatch):
    def absent(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", absent)
    mujoco_native.check_profile_python()


def test_same_python_population_mismatch_keeps_native_error(monkeypatch, tmp_path):
    monkeypatch.setattr(mujoco_native, "sys", SimpleNamespace(version_info=(3, 12)))
    dist = SimpleNamespace(version="3.8.1", files=[Path("unexpected.so")],
                           locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)
    guard = object.__new__(mujoco_native.MujocoNativeGuard)
    guard._version = "3.8.1"
    guard._images = {"expected.so": {}}
    with pytest.raises(NativeImageError) as caught:
        guard._check_distribution()
    assert str(caught.value) == "MuJoCo distribution native image population differs"
