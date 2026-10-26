"""Check adapter bookkeeping when a native read or cleanup rejects.

These tests use Python stand-ins for the native resources; no simulator or
native scratch resource is constructed.
"""

from contextlib import ExitStack
from typing import Any

import pytest

from specola.adapters.mujoco import MujocoAdapter


def subject() -> Any:
    adapter: Any = object.__new__(MujocoAdapter)
    adapter._native_closed = False
    adapter._native_resources = ExitStack()
    adapter._native_projection = object()
    adapter._native_subject_model = object()
    adapter._episode_started = True
    adapter._last_observation = object()
    return adapter


def test_rejected_interval_invalidates_even_a_recorded_observation():
    adapter = subject()
    with pytest.raises(RuntimeError):
        with adapter._native_observation_scope():
            # Model the base class committing before the outer guard exits.
            adapter._episode_started = True
            adapter._last_observation = object()
            raise RuntimeError("reader rejected")
    assert adapter._episode_started is False
    assert adapter._last_observation is None


def test_rejection_before_a_record_preserves_previous_episode():
    adapter = subject()
    previous = adapter._last_observation
    with pytest.raises(RuntimeError, match="metadata refused"):
        with adapter._native_observation_scope():
            raise RuntimeError("metadata refused")
    assert adapter._episode_started is True
    assert adapter._last_observation is previous


@pytest.mark.parametrize("failure", [None, "callback"])
def test_close_retires_subject_and_can_be_repeated(failure):
    adapter = subject()
    calls = []
    adapter._native_resources.callback(calls.append, "first")

    def cleanup():
        calls.append("second")
        if failure == "callback":
            raise RuntimeError("cleanup rejected")

    adapter._native_resources.callback(cleanup)
    if failure:
        with pytest.raises(RuntimeError):
            adapter.close_native_extraction()
    else:
        adapter.close_native_extraction()
    assert calls == ["second", "first"]
    assert adapter._native_closed
    assert adapter._native_projection is None
    assert adapter._native_subject_model is None
    adapter.close_native_extraction()
    assert calls == ["second", "first"]
