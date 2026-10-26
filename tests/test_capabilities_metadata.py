"""Reset/discovery metadata parity for the Monitor facade."""

from collections.abc import Mapping
from typing import List

import pytest

from specola.capabilities import reset_adapter
from specola.discovery_model import property_metadata_for
from specola.session import Monitor
from specola.adapters.fake import _MockAdapter


SPECS = """\
specs:
  - name: qpos_limit
    formula: "robot_*[qpos_joint1 @ joint angle] <= 1"
"""


class _FalseyMapping(Mapping):
    def __init__(self, values):
        self._data = dict(values)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return False


class _MetadataAdapter(_MockAdapter):
    def __init__(self, failure=False):
        super().__init__()
        self.failure = failure
        self.calls: List[int] = []
        self.falsey = _FalseyMapping(super().property_metadata(0))
        self.empty = {}
        setattr(self, "property_metadata", self._metadata_callback)

    def _metadata_callback(self, entity_id: int):
        self.calls.append(entity_id)
        if self.failure:
            raise ValueError("metadata source failed")
        if entity_id == 0:
            return self.falsey
        if entity_id == 1:
            return self.empty
        if entity_id == 2:
            return None
        return super().property_metadata(entity_id)


def test_reset_adapter_routes_metadata_through_discovery_helper():
    adapter = _MetadataAdapter()
    canonical = property_metadata_for(adapter, (0, 1, 2))

    assert canonical[0] is adapter.falsey
    assert canonical[1] is adapter.empty
    assert canonical[2] == {}

    result = reset_adapter(adapter)

    assert result.discovered_catalog.component_groups_for_entity(0)[
        "qpos_joint1"
    ] == "joint angle"
    assert result.discovered_catalog.component_groups_for_entity(1) == {}
    assert result.discovered_catalog.component_groups_for_entity(2) == {}
    assert set(adapter.calls) == set(result.initial_observation.perceived_entities)


def test_reset_adapter_metadata_failure_has_context_and_cause():
    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter _MetadataAdapter entity 0",
    ) as exc_info:
        reset_adapter(_MetadataAdapter(failure=True))

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "metadata source failed"


def test_monitor_watch_uses_same_contextual_metadata_failure(tmp_path):
    specs = tmp_path / "specs.yaml"
    specs.write_text(SPECS)

    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter _MetadataAdapter entity 0",
    ) as exc_info:
        Monitor(str(specs), results_dir=None).watch(
            _MetadataAdapter(failure=True)
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_monitor_watch_accepts_falsey_empty_and_none_metadata(tmp_path):
    specs = tmp_path / "specs.yaml"
    specs.write_text(SPECS)

    adapter = _MetadataAdapter()
    session = Monitor(str(specs), results_dir=None).watch(adapter)

    assert session.engine.managers
    assert set(session.engine.manager_specs().values()) == {"qpos_limit"}
    assert set(adapter.calls) == set(adapter.entity_labels)
    session.end()
