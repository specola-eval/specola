"""Subsystem-tag metadata: catalog plumbing and @ selector matching.

Tags are adapter-supplied structural labels (arm/gripper/base/torso plus
left/right) carried per component key, parallel to display groups. The
selector filter ``@ label`` matches a key when the label equals its group
OR is one of its tags. Fixtures are sim-free.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from specola.adapter import PropertyMeta
from specola.components import ComponentCatalog
from specola.entities import EntityState, SpecConfigError, TrackState


def _entity(entity_id, entity_class, properties):
    return EntityState(
        entity_id=entity_id,
        entity_class=entity_class,
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def _entities():
    return {
        1: _entity(
            1,
            "robot",
            {
                "qvel_robot0_joint1": 0.1,
                "qvel_gripper0_finger_joint1": 0.0,
                "qvel_base_forward": 0.0,
            },
        ),
    }


def _metadata():
    return {
        1: {
            "qvel_robot0_joint1": PropertyMeta(
                group="joint velocities", tags=("arm", "right"),
            ),
            "qvel_gripper0_finger_joint1": PropertyMeta(
                group="joint velocities", tags=("gripper", "right"),
            ),
            "qvel_base_forward": PropertyMeta(
                group="joint velocities", tags=("base",),
            ),
        },
    }


@pytest.fixture
def catalog():
    return ComponentCatalog.from_entities(_entities(), metadata=_metadata())


# ---------------------------------------------------------------------------
# Task 1: PropertyMeta + catalog plumbing
# ---------------------------------------------------------------------------

def test_property_meta_tags_default_empty():
    assert PropertyMeta(group="g").tags == ()


def test_catalog_carries_component_tags(catalog):
    tags = catalog.component_tags_for_entity(1)
    assert tags["qvel_robot0_joint1"] == ("arm", "right")
    assert tags["qvel_gripper0_finger_joint1"] == ("gripper", "right")
    assert tags["qvel_base_forward"] == ("base",)


def test_unknown_entity_has_no_tags(catalog):
    assert catalog.component_tags_for_entity(999) == {}


def test_metadata_without_tags_attribute_is_tolerated():
    # from_entities accepts any object with .group; tags are best-effort.
    entities = {1: _entity(1, "robot", {"qvel_robot0_joint1": 0.1})}
    metadata = {1: {"qvel_robot0_joint1": SimpleNamespace(group="jv")}}
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    assert catalog.component_tags_for_entity(1) == {}
    assert catalog.component_groups_for_entity(1) == {
        "qvel_robot0_joint1": "jv",
    }


# ---------------------------------------------------------------------------
# Task 2: @ label matches group OR tag in resolve_selector
# ---------------------------------------------------------------------------

from specola.spec.selectors import parse_selector, resolve_selector


def _resolve(catalog, text):
    return resolve_selector(
        parse_selector(text),
        catalog.components_for_entity(1),
        catalog.component_groups_for_entity(1),
        catalog.component_tags_for_entity(1),
    )


def test_at_matches_subsystem_tag(catalog):
    assert _resolve(catalog, "* @ arm") == ("qvel_robot0_joint1",)


def test_at_matches_side_tag(catalog):
    assert _resolve(catalog, "* @ right") == (
        "qvel_gripper0_finger_joint1",
        "qvel_robot0_joint1",
    )


def test_at_still_matches_group(catalog):
    assert _resolve(catalog, "* @ joint velocities") == (
        "qvel_base_forward",
        "qvel_gripper0_finger_joint1",
        "qvel_robot0_joint1",
    )


def test_at_miss_lists_groups_and_tags(catalog):
    with pytest.raises(SpecConfigError) as excinfo:
        _resolve(catalog, "* @ torso")
    message = str(excinfo.value)
    assert "Available groups" in message
    assert "Available tags" in message
    assert "arm" in message


def test_three_argument_call_still_works(catalog):
    keys = resolve_selector(
        parse_selector("* @ joint velocities"),
        catalog.components_for_entity(1),
        catalog.component_groups_for_entity(1),
    )
    assert len(keys) == 3


# ---------------------------------------------------------------------------
# Task 3: tags flow through semantic compilation (compile_specs)
# ---------------------------------------------------------------------------

from specola.stl_parser import compile_specs


def test_tag_filter_compiles_end_to_end(catalog):
    compiled = compile_specs("robot_*[* @ arm] >= 0", catalog=catalog)
    assert compiled


def test_tag_filter_side_selects_arm_and_gripper(catalog):
    compiled = compile_specs(
        "robot_*[any: * @ right] >= 0", catalog=catalog
    )
    assert compiled


def test_unknown_tag_error_names_available_tags(catalog):
    with pytest.raises(SpecConfigError) as excinfo:
        compile_specs("robot_*[* @ wheels] >= 0", catalog=catalog)
    assert "Available tags" in str(excinfo.value)


def test_tag_filter_works_with_pinned_entity_binding(catalog):
    compiled = compile_specs("robot_1[* @ arm] >= 0", catalog=catalog)
    assert compiled


def test_classwide_error_lists_available_tags():
    entities = {
        1: _entity(1, "robot", {"qvel_robot0_joint1": 0.1}),
        2: _entity(2, "robot", {"qvel_robot1_joint1": 0.2}),
    }
    metadata = {
        1: {
            "qvel_robot0_joint1": PropertyMeta(
                group="joint velocities", tags=("arm", "right"),
            ),
        },
        2: {
            "qvel_robot1_joint1": PropertyMeta(
                group="joint velocities", tags=("arm", "left"),
            ),
        },
    }
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    with pytest.raises(SpecConfigError) as excinfo:
        compile_specs("robot_*[* @ wheels] >= 0", catalog=catalog)
    message = str(excinfo.value)
    assert "Available tags" in message
    assert "arm" in message


# ---------------------------------------------------------------------------
# Task 6: specola check key table renders tags next to the group
# ---------------------------------------------------------------------------

from specola.check_specs import _print_key_table
from specola.stl_parser import parse_stl


def test_key_table_shows_tags(catalog, capsys):
    formula = parse_stl(
        "robot_1[qvel_robot0_joint1] >= 0", catalog=catalog
    )
    _print_key_table(formula, catalog, _metadata(), _entities())
    out = capsys.readouterr().out
    assert "qvel_robot0_joint1" in out
    assert "[arm right]" in out


# ---------------------------------------------------------------------------
# `@ label` is one filter over two axes: ambiguity and key-table hoisting
# ---------------------------------------------------------------------------

def test_no_ambiguous_labels_when_groups_and_tags_are_disjoint(catalog):
    assert catalog.ambiguous_filter_labels() == ()


def test_label_that_is_both_a_group_and_a_tag_is_reported():
    # An adapter is free to emit either vocabulary, and nothing stops the
    # two from colliding. `@ base` would then silently union a group match
    # and a tag match, so the catalog reports the name for a warning.
    entities = {
        1: _entity(
            1,
            "robot",
            {"qvel_base_forward": 0.0, "qpos_base_yaw": 0.0},
        ),
    }
    metadata = {
        1: {
            # group named "base" ...
            "qvel_base_forward": PropertyMeta(group="base", tags=()),
            # ... and a tag named "base" on a different key
            "qpos_base_yaw": PropertyMeta(
                group="base angles", tags=("base",),
            ),
        },
    }
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    assert catalog.ambiguous_filter_labels() == ("base",)


def test_key_table_hoists_a_uniform_group(catalog, capsys):
    # Every matched key shares one group and tag set, so repeating it on
    # each row is noise -- it is stated once instead.
    formula = parse_stl(
        "robot_1[all: qvel_* @ joint velocities] >= 0", catalog=catalog
    )
    _print_key_table(formula, catalog, _metadata(), _entities())
    out = capsys.readouterr().out
    assert out.count("joint velocities") == 1
    assert "all in group" in out


def test_key_table_does_not_hoist_when_entities_disagree(capsys):
    # Rows show the representative entity's metadata, so hoisting must not
    # say "all" unless the claim holds for every entity the slot binds.
    # Two entities can spell one key with different units and groups.
    from specola.check_specs import _print_key_table
    from specola.stl_parser import parse_stl

    entities = {
        1: _entity(1, "robot", {"speed": 1.0}),
        2: _entity(2, "robot", {"speed": 2.0}),
    }
    metadata = {
        1: {"speed": PropertyMeta(
            group="body linear velocity", note="m/s", tags=("base",),
        )},
        2: {"speed": PropertyMeta(
            group="joint angular velocity", note="rad/s", tags=("arm",),
        )},
    }
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    formula = parse_stl("robot_*[speed] >= 0", catalog=catalog)

    _print_key_table(formula, catalog, metadata, entities)
    out = capsys.readouterr().out

    assert "all in group" not in out
    assert "body linear velocity" in out
