"""Discovery formatting: readable, nothing hidden."""

from __future__ import annotations

import subprocess
import sys


def _run_discover(*extra):
    return subprocess.run(
        [sys.executable, "-m", "specola.discover", "--sim", "mock", *extra],
        capture_output=True, text=True, timeout=60,
    )


import numpy as np

from specola.adapter import PropertyMeta
from specola.cli_common import (
    _fit_label,
    _group_properties,
    _print_class_summary,
    _print_entity_table,
    _print_example,
    _print_property_table,
)
from specola.entities import EntityState, TrackState

_REGION_PROPS = {
    "hw": 0.1, "hh": 0.1, "hd": 0.1,
    "x0": 0.9, "x1": 1.1, "y0": -0.1, "y1": 0.1, "z0": 1.1, "z1": 1.3,
}


def _entity(properties, position=(1.0, 2.0, 3.0), velocity=(0.1, 0.2, 0.3),
            entity_id=1, cls="object"):
    return EntityState(
        entity_id=entity_id,
        entity_class=cls,
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def test_group_properties_orders_groups_and_keeps_every_key():
    entity = _entity({
        "qpos_j_0": 1.0, "qpos_j_3": 0.99,
        "qfrc_applied_j_0": 0.0, "custom": 5.0,
    })
    metadata = {
        "qpos_j_0": PropertyMeta(group="joint position"),
        "qpos_j_3": PropertyMeta(group="joint orientation", note="quat w"),
        "qfrc_applied_j_0": PropertyMeta(group="applied forces"),
    }
    grouped = _group_properties(entity, metadata)
    labels = [group for group, _ in grouped]
    assert labels == ["joint position", "joint orientation", "applied forces", "other"]
    all_keys = [key for _, items in grouped for key, _ in items]
    assert sorted(all_keys) == sorted(entity.properties)


def _region(entity_id, extra=None):
    properties = dict(_REGION_PROPS)
    properties.update(extra or {})
    return _entity(properties, entity_id=entity_id, cls="region")


def test_regions_with_shared_keys_summarize_once(capsys):
    entities = {
        0: _entity({"qpos_j": 0.0}, entity_id=0),
        1: _region(1),
        2: _region(2),
        3: _region(3, extra={"door_angle": 0.4}),
    }
    _print_property_table(entities, labels=None, verbose=False)
    out = capsys.readouterr().out
    exact_set_lines = [
        line for line in out.splitlines()
        if line.strip() == "hd, hh, hw, x0, x1, y0, y1, z0, z1"
    ]
    assert len(exact_set_lines) == 1      # shared set printed once
    assert "2 region-like entities" in out
    assert "door_angle" in out            # deviating region still surfaced
    assert "[3]" in out                   # ... with its identity
    assert "[1]" not in out and "[2]" not in out


def test_regions_keep_full_listings_in_verbose(capsys):
    entities = {
        1: _region(1),
        2: _region(2),
    }
    _print_property_table(entities, labels=None, verbose=True)
    out = capsys.readouterr().out
    assert "[1]" in out and "[2]" in out
    assert out.count("= +0.9") >= 2       # x0 values per region


def test_property_lines_show_values_and_combined_notes(capsys):
    entity = _entity({"qpos_j_0": 1.0, "mass": 0.25})
    metadata = {
        "qpos_j_0": PropertyMeta(group="joint position", note="m"),
    }
    _print_property_table(
        {1: entity}, labels=None, verbose=False, metadata={1: metadata},
    )
    out = capsys.readouterr().out
    lines = out.splitlines()
    qpos_row = next(line for line in lines if "qpos_j_0" in line)
    assert "+1" in qpos_row and " m" in qpos_row and "joint position" in qpos_row
    mass_row = next(line for line in lines if "mass" in line)
    assert "+0.25" in mass_row        # keys without metadata get values too


def test_grouped_leftovers_render_as_aligned_table(capsys):
    # The grouped fallback is a KEY/VALUE/UNIT/GROUP table, matching the
    # matrix/family/sensor table style — not wrapped "key = value" prose.
    entity = _entity({"qpos_j_0": 1.0, "qpos_j_3": 0.99, "mass": 0.25})
    metadata = {
        "qpos_j_0": PropertyMeta(group="joint position", note="m"),
        "qpos_j_3": PropertyMeta(group="joint orientation"),
    }
    _print_property_table(
        {1: entity}, labels=None, verbose=False, metadata={1: metadata},
    )
    out = capsys.readouterr().out
    header = next(
        line for line in out.splitlines() if "KEY" in line
    )
    for column in ("KEY", "VALUE", "UNIT", "GROUP"):
        assert column in header
    assert "qpos_j_0 = +1" not in out
    # Rows align: every key starts at the same column as the KEY header.
    key_column = header.index("KEY")
    for key in ("qpos_j_0", "qpos_j_3", "mass"):
        row = next(line for line in out.splitlines() if key in line)
        assert row.index(key) == key_column
    # Group ordering preserved: position before orientation before other.
    body = out[out.index("KEY"):]
    assert body.index("qpos_j_0") < body.index("qpos_j_3") < body.index("mass")


def test_entity_without_metadata_renders_same_table(capsys):
    # Metadata-less entities get the same KEY table (all rows in "other"),
    # not a flat wrapped "key = value" preview.
    entity = _entity({"mass": 0.25, "temperature": 22.0})
    _print_property_table(
        {1: entity}, labels=None, verbose=False, metadata={},
    )
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "KEY" in line)
    for column in ("KEY", "VALUE", "UNIT", "GROUP"):
        assert column in header
    assert "mass = +0.25" not in out
    mass_row = next(line for line in out.splitlines() if "mass" in line)
    assert "+0.25" in mass_row and "other" in mass_row


def test_example_fallback_uses_an_extracted_key(capsys):
    entities = {
        0: _entity(
            {"qfrc_applied_bfoot": 0.0, "qvel_bfoot": 0.1},
            entity_id=0, cls="robot",
        ),
    }
    _print_example(entities, metadata={0: {
        "qvel_bfoot": PropertyMeta(group="joint angular velocity", note="rad/s"),
        "qfrc_applied_bfoot": PropertyMeta(group="applied torques", note="N·m"),
    }})
    out = capsys.readouterr().out
    assert "robot_*[qvel_bfoot]" in out


def test_example_region_containment_uses_metadata_position_keys(capsys):
    entities = {
        0: _region(0),
        1: _entity(
            {
                "xpos_obj_0": 1.0,
                "xpos_obj_1": 2.0,
                "xpos_obj_2": 3.0,
            },
            entity_id=1, cls="object",
        ),
    }
    metadata = {
        1: {
            "xpos_obj_0": PropertyMeta(group="body position", note="m"),
            "xpos_obj_1": PropertyMeta(group="body position", note="m"),
            "xpos_obj_2": PropertyMeta(group="body position", note="m"),
        },
    }
    _print_example(entities, metadata=metadata)
    out = capsys.readouterr().out
    assert "xpos_obj_0" in out
    assert "[px]" not in out
    assert "[py]" not in out
    assert "[pz]" not in out


def test_default_summary_counts_entities_without_listing_scene_rows():
    proc = _run_discover()
    out = proc.stdout
    assert proc.returncode == 0
    assert "Entities by semantic class" in out
    assert "region" in out
    assert "mock_cabinet_region" not in out
    assert "mock_robot_base" not in out


def test_fit_label_pads_and_truncates_with_marker():
    assert _fit_label("short", 10) == "short     "
    fitted = _fit_label("a" * 60, 10)
    assert len(fitted) == 10
    assert fitted.rstrip().endswith("…")
    assert fitted.endswith(" ")  # separator survives truncation


def test_entity_table_sizes_label_column_to_content(capsys):
    long_label = "cab_main_main_group_counter_left_group_fixture"  # 46 chars
    entities = {
        0: _entity({"qpos_j": 0.0}, entity_id=0),
        1: _entity({}),
    }
    labels = {0: long_label, 1: "tiny"}
    _print_entity_table(entities, labels=labels)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "cab_main" in line)
    assert long_label in row            # not truncated at 34 chars
    assert long_label + "(" not in row  # separator before position column


def test_filtered_mock_discovery_reaches_joint_detail():
    proc = _run_discover(
        "--class", "robot", "--signal", "qpos_*", "--verbose",
    )
    assert proc.returncode == 0
    assert "qpos_joint1 = +0.1" in proc.stdout
    assert "group=joint angle" in proc.stdout
    assert "Selector preview: robot_0[qpos_joint1]" in proc.stdout


def test_blocks_order_class_first_not_id_first(capsys):
    def entity(entity_id, cls, properties):
        return EntityState(
            entity_id=entity_id,
            entity_class=cls,
            position=np.asarray([0.0, 0.0, 0.0], dtype=float),
            velocity=np.asarray([0.0, 0.0, 0.0], dtype=float),
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties=properties,
        )

    # Ids deliberately reversed relative to the class display order.
    entities = {
        0: entity(0, "object", {"mass": 1.0}),
        1: entity(1, "gripper", {"grip": 0.5}),
        2: entity(2, "robot", {"qpos_j1": 0.1}),
    }

    _print_entity_table(entities, labels=None)
    table_out = capsys.readouterr().out
    _print_property_table(
        entities, labels=None, verbose=False, metadata={},
    )
    blocks_out = capsys.readouterr().out

    for out in (table_out, blocks_out):
        robot = out.index("robot")
        gripper = out.index("gripper")
        obj = out.index("object")
        assert robot < gripper < obj, out


# ---------------------------------------------------------------------------
# "0 currently common" is a statistic that predicts binding: a wildcard slot
# binds only entities exposing every key it names, so a class sharing no key
# cannot be covered by one class_* reference. The summary says so instead of
# leaving the reader to infer it (issue #72).
# ---------------------------------------------------------------------------

def test_class_sharing_no_key_states_the_binding_consequence(capsys):
    entities = {
        1: _entity({"xpos_obj_main_2": 0.9}, entity_id=1),
        2: _entity({"xpos_distr_main_2": 0.9}, entity_id=2),
    }
    _print_class_summary(entities)
    out = capsys.readouterr().out
    assert "0 currently common" in out
    assert "no exact key covers this class" in out
    assert "component glob" in out
    # The consequence is about *exact-key* references. A glob reference
    # binds the whole class, so the line must not claim that every
    # object_* reference binds one entity -- the remedy it recommends in
    # the same breath would falsify that.
    assert "one object_* reference binds a single entity" not in out


def test_class_sharing_a_key_states_no_consequence(capsys):
    entities = {
        1: _entity({"mass": 0.25, "temperature": 22.0}, entity_id=1),
        2: _entity({"mass": 0.40, "hinge_angle": 0.15}, entity_id=2),
    }
    _print_class_summary(entities)
    out = capsys.readouterr().out
    assert "1 currently common" in out
    assert "binds a single entity" not in out
