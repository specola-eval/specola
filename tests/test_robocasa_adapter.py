"""Sim-free regressions for the RoboCasa registry factory."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import specola.adapters.robocasa as robocasa_adapter


@pytest.mark.parametrize("options,offscreen", (({}, True), ({"capture_frames": False}, False)))
def test_registry_factory_builds_the_renderer_only_when_frames_may_be_captured(monkeypatch, options, offscreen):
    created = []

    class FakeEnv:
        def __init__(self, task_name, **kwargs):
            self.kwargs = kwargs
            self.reset_called = False
            created.append(self)

        def reset(self):
            self.reset_called = True

    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(make=FakeEnv))
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.delenv("ROBOCASA_GENERATIVE_TEXTURES", raising=False)
    monkeypatch.setattr(robocasa_adapter, "RoboCasaAdapter", lambda env, **kwargs: env)

    env: Any = robocasa_adapter._make_robocasa("PickPlaceCounterToCabinet", None, True, **options)

    assert env.kwargs["has_offscreen_renderer"] is offscreen
    assert env.reset_called
    assert len(created) == 1
    assert env.kwargs["seed"] is None


def test_registry_factory_draws_the_kitchen_from_the_seed(monkeypatch):
    made = []

    def make(task_name, **kwargs):
        made.append(kwargs)
        return SimpleNamespace(reset=lambda: None)

    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(make=make))
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.delenv("ROBOCASA_GENERATIVE_TEXTURES", raising=False)
    monkeypatch.setattr(robocasa_adapter, "RoboCasaAdapter", lambda env, **kwargs: env)

    robocasa_adapter._make_robocasa("PickPlaceCounterToCabinet", None, True, capture_frames=False, seed=7)
    assert [kwargs["seed"] for kwargs in made] == [7]
