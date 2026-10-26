"""Entity labels follow the adapter's population, not the reset snapshot.

Dynamic populations (CARLA actors, contact pairs) gain members after the
run starts; the run CLI and the session facade previously captured
``adapter.entity_labels`` once, so every later arrival displayed as
``e<id>`` in transitions and summaries.
"""

from __future__ import annotations

from specola.session import _LiveEntityLabels


class _GrowingAdapter:
    def __init__(self):
        self._labels = {0: "vehicle_audi_a2_25"}

    @property
    def entity_labels(self):
        return dict(self._labels)


def test_labels_refresh_on_miss():
    adapter = _GrowingAdapter()
    labels = _LiveEntityLabels(adapter)
    assert labels.get(0) == "vehicle_audi_a2_25"

    adapter._labels[7] = "vehicle_mini_cooper_31"
    assert labels.get(7) == "vehicle_mini_cooper_31"
    assert labels[7] == "vehicle_mini_cooper_31"


def test_unknown_id_still_falls_back():
    labels = _LiveEntityLabels(_GrowingAdapter())
    assert labels.get(99, "e99") == "e99"


def test_adapter_without_labels_is_empty():
    labels = _LiveEntityLabels(object())
    assert labels.get(0, "e0") == "e0"
    assert dict(labels) == {}


def test_tracker_keeps_live_view_by_reference():
    # The consumer must retain the mapping by reference: a defensive
    # dict() copy passes every _LiveEntityLabels unit test while
    # restoring e<N> fallbacks for entities born after reset.
    from specola.run_state import RunStateTracker

    labels = _LiveEntityLabels(_GrowingAdapter())
    tracker = RunStateTracker({}, labels=labels)
    assert tracker._labels is labels
