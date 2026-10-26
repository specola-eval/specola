from __future__ import annotations

import pytest

from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, _matches, filter_discovery
from specola.discovery_model import build_discovery_model


@pytest.fixture
def model():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    return build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )


def test_filter_unions_within_kind_and_intersects_across_kinds(model):
    view = filter_discovery(model, DiscoveryQuery(
        classes=("robot", "gripper"),
        signals=("*qpos*",),
    ))

    assert {entity.entity_class for entity in view.entities} == {
        "robot", "gripper",
    }
    assert all(
        prop.key.startswith(("qpos", "gripper_qpos"))
        for entity in view.entities for prop in entity.properties
    )


def test_zero_result_is_valid_and_keeps_denominators(model):
    view = filter_discovery(
        model, DiscoveryQuery(entities=("does-not-exist*",)),
    )
    assert view.entities == ()
    assert view.before.entity_count == 5
    assert view.after.entity_count == 0


def test_malformed_glob_is_rejected(model):
    with pytest.raises(ValueError, match="unclosed"):
        filter_discovery(model, DiscoveryQuery(signals=("[broken",)))


def test_entity_filter_accepts_monitor_reference_and_search_glob(model):
    by_reference = filter_discovery(
        model, DiscoveryQuery(entities=("OBJECT_1",)),
    )
    by_search = filter_discovery(
        model, DiscoveryQuery(search=("*TARGET*",)),
    )

    assert [entity.entity_id for entity in by_reference.entities] == [1]
    assert [entity.entity_id for entity in by_search.entities] == [1]


@pytest.mark.parametrize("pattern", ("[]", "[!]", "bad]"))
def test_other_malformed_bracket_forms_are_rejected(model, pattern):
    with pytest.raises(ValueError, match="bracket"):
        filter_discovery(model, DiscoveryQuery(signals=(pattern,)))


def test_fnmatch_bracket_members_and_caret_semantics_are_preserved(model):
    filter_discovery(model, DiscoveryQuery(signals=("[]a]",)))
    filter_discovery(model, DiscoveryQuery(signals=("[]]",)))
    filter_discovery(model, DiscoveryQuery(signals=("[^abc]",)))

    assert _matches("]", ("[]a]",))
    assert _matches("a", ("[]a]",))
    assert _matches("]", ("[]]",))
    assert _matches("^", ("[^abc]",))
    assert _matches("a", ("[^abc]",))
    assert not _matches("z", ("[^abc]",))
