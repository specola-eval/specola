"""Factory lifecycle isolation using value objects, without a CARLA server."""

from copy import deepcopy
from types import SimpleNamespace
import atexit
import sys

import pytest

import specola.adapters.carla as carla_adapter


class _World:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None,
            # A field outside the factory's two requested settings proves that
            # restoration uses the complete original value object.
            future_setting={"native": [17, 29]},
        )
        self.frame = 41
        self.applications = []
        self.fail_setup = False
        self.fail_restore = False
        self.ignore_restore_component = False

    def get_settings(self):
        return deepcopy(self.settings)

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame)

    def apply_settings(self, settings):
        self.applications.append(deepcopy(settings))
        self.settings = deepcopy(settings)
        if settings.synchronous_mode:
            self.frame += 2
            if self.fail_setup:
                raise RuntimeError("native setup failed after a partial update")
        else:
            if self.ignore_restore_component:
                self.settings.future_setting = {"native": [-1]}
            if self.fail_restore:
                raise RuntimeError("native restoration failed")
        return self.frame


@pytest.fixture
def factory(monkeypatch):
    world = _World()
    callbacks = []
    client = SimpleNamespace(set_timeout=lambda timeout: None, get_world=lambda: world)
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=lambda host, port: client))
    monkeypatch.setattr(atexit, "register", lambda callback: callbacks.append(callback))
    return world, callbacks, lambda: carla_adapter._make_carla(None, None, False)


def test_factory_preserves_and_restores_the_complete_original(factory):
    world, callbacks, make = factory
    original = deepcopy(world.settings)
    adapter = make()
    assert world.settings.synchronous_mode is True
    assert world.settings.fixed_delta_seconds == 0.05
    assert world.settings.future_setting == original.future_setting
    world.settings.future_setting = {"native": [99]}
    for callback in reversed(callbacks):
        callback()
    assert world.settings == original
    evidence = adapter._factory_lifecycle_evidence
    assert evidence["setup"] == {"frame_before": 41, "frame_after": 43}
    assert evidence["restorations"] == [
        {"frame_before": 43, "frame_after": 43, "settings_match": True}
    ]


@pytest.mark.parametrize("failure", ["fail_restore", "ignore_restore_component"])
def test_restoration_failure_is_recorded_and_reported(factory, failure, capsys):
    world, callbacks, make = factory
    adapter = make()
    setattr(world, failure, True)
    for callback in reversed(callbacks):
        callback()
    assert "CARLA world settings restoration failed" in capsys.readouterr().err
    assert adapter._factory_lifecycle_evidence["restorations"][0]["error"]


def test_failed_setup_restores_the_original_before_raising(factory):
    world, callbacks, make = factory
    original = deepcopy(world.settings)
    world.fail_setup = True
    with pytest.raises(RuntimeError, match="native setup failed after a partial update"):
        make()
    assert world.settings == original
    assert not callbacks


def test_failed_adapter_construction_restores_the_original(factory, monkeypatch):
    world, callbacks, make = factory
    original = deepcopy(world.settings)

    def failed_adapter(*args, **kwargs):
        raise RuntimeError("adapter construction failed")

    monkeypatch.setattr(carla_adapter, "CarlaAdapter", failed_adapter)
    with pytest.raises(RuntimeError, match="adapter construction failed"):
        make()
    assert world.settings == original
    assert not callbacks


def test_failed_camera_provider_installation_restores_the_original(factory, monkeypatch):
    world, callbacks, make = factory
    original = deepcopy(world.settings)

    def failed_provider(*args, **kwargs):
        raise RuntimeError("camera provider installation failed")

    monkeypatch.setattr(carla_adapter.CarlaAdapter, "set_camera_provider", failed_provider)
    with pytest.raises(RuntimeError, match="camera provider installation failed"):
        make()
    assert world.settings == original
    assert not callbacks


def test_changed_settings_between_reads_are_rejected_before_apply(factory, monkeypatch):
    world, callbacks, make = factory
    initial_get_settings = world.get_settings
    reads = 0

    def changed_get_settings():
        nonlocal reads
        reads += 1
        result = initial_get_settings()
        if reads == 2:
            result.future_setting = {"native": [301]}
        return result

    monkeypatch.setattr(world, "get_settings", changed_get_settings)
    with pytest.raises(RuntimeError, match="settings changed during factory setup"):
        make()
    assert not world.applications
    assert not callbacks
