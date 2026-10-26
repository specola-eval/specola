"""Discovery formatting: joints x signals matrix, nothing hidden."""

from __future__ import annotations

import numpy as np
import pytest

from specola.adapter import PropertyMeta
from specola.cli_common import (
    _JOINT_SIGNALS,
    _print_matrix_table,
    _print_property_table,
    print_discovery,
)
from specola.entities import EntityState, TrackState
from specola.pager import pager_command


def _entity(properties, entity_id=0, cls="robot"):
    return EntityState(
        entity_id=entity_id,
        entity_class=cls,
        position=np.asarray([0.0, 0.0, 0.0], dtype=float),
        velocity=np.asarray([0.0, 0.0, 0.0], dtype=float),
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


# ---------------------------------------------------------------------------
# Synthetic multi-joint robot: 2 base-pose-family joints + 3 hinge
# articulation joints, 5 signals each, plus family-table keys (body pose,
# the three com-based body 6-vectors, a dim-1 and dim-3 sensor, and one
# actuator contributing all four of its keys) and one key with no
# metadata at all, to exercise the nothing-hidden fallback.
# ---------------------------------------------------------------------------

_BASE_TOKENS = [
    ("rootx", "base displacement (rel. spawn)", "m"),
    ("rooty", "base angle (rel. spawn)", "rad"),
]

_ARTICULATION_TOKENS = [
    ("shoulder", "joint angle", "rad"),
    ("elbow", "joint angle", "rad"),
    ("wrist", "joint angle", "rad"),
]

_SIGNALS = ["qpos", "qvel", "qacc", "qfrc_applied", "qfrc_passive"]


def _synthetic_matrix_fixture():
    properties = {}
    metadata = {}

    value = 0.101

    for token, group, note in _BASE_TOKENS + _ARTICULATION_TOKENS:
        for signal in _SIGNALS:
            key = f"{signal}_{token}"
            properties[key] = value
            value += 0.001

            if signal == "qpos":
                metadata[key] = PropertyMeta(group=group, note=note)
            else:
                metadata[key] = PropertyMeta(group="other-signal")

    # body pose family (2 tokens x 3 axes)
    for token in ("torso",):
        for axis in range(3):
            key = f"xpos_{token}_{axis}"
            properties[key] = value
            metadata[key] = PropertyMeta(group="body position", note="m")
            value += 0.001

    # xquat family with a mix of value magnitudes: a column-width bug
    # that sizes a column from its header label alone (not its formatted
    # values) undersizes the "x" column here, since "+0.0002462" (10
    # chars) is much wider than the 1-char header "x".
    for component, magnitude in (
        ("w", 0.9996), ("x", 0.0002462), ("y", 0.05), ("z", 0.01),
    ):
        key = f"xquat_torso_{component}"
        properties[key] = magnitude
        metadata[key] = PropertyMeta(
            group="body orientation", note=f"quat {component}",
        )

    # body force / acceleration families (1 token x 6 indices each), each
    # split across two unit groups the way cvel is
    for family, torque_group, force_group, torque_note, force_note in (
        ("cfrc_ext", "body external torque", "body external force",
         "N·m", "N"),
        ("cfrc_int", "body internal torque", "body internal force",
         "N·m", "N"),
        ("cacc", "body angular acceleration",
         "body com-frame linear acceleration", "rad/s²", "m/s²"),
    ):
        for index in range(6):
            key = f"{family}_torso_{index}"
            properties[key] = value
            metadata[key] = PropertyMeta(
                group=torque_group if index < 3 else force_group,
                note=torque_note if index < 3 else force_note,
            )
            value += 0.001

    # dim-1 sensor
    properties["sensordata_link1_touch"] = value
    metadata["sensordata_link1_touch"] = PropertyMeta(
        group="sensors: touch", note="N",
    )
    value += 0.001

    # dim-3 sensor
    for axis in range(3):
        key = f"sensordata_link1_accel_{axis}"
        properties[key] = value
        metadata[key] = PropertyMeta(group="sensors: accelerometer", note="m/s²")
        value += 0.001

    # one actuator: ctrl + force
    properties["ctrl_motor1"] = value
    metadata["ctrl_motor1"] = PropertyMeta(group="actuator command")
    value += 0.001

    properties["actuator_force_motor1"] = value
    metadata["actuator_force_motor1"] = PropertyMeta(
        group="actuator output torque", note="N·m",
    )
    value += 0.001

    properties["actuator_length_motor1"] = value
    metadata["actuator_length_motor1"] = PropertyMeta(
        group="actuator angle", note="rad",
    )
    value += 0.001

    properties["actuator_velocity_motor1"] = value
    metadata["actuator_velocity_motor1"] = PropertyMeta(
        group="actuator angular velocity", note="rad/s",
    )
    value += 0.001

    # gripper site position family
    for axis in range(3):
        key = f"site_xpos_gripper_{axis}"
        properties[key] = value
        metadata[key] = PropertyMeta(group="site position", note="m")
        value += 0.001

    # a key with no metadata at all -- the nothing-hidden fallback
    properties["custom_extra"] = value

    return properties, metadata


def test_matrix_path_renders_every_key_value_exactly_once(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # Nothing hidden, nothing duplicated: every key's value renders in
    # this entity's block -- matrix, family tables, nested
    # Sensors/Actuators tables, or the grouped leftover listing.
    #
    # Counted as whole whitespace-delimited tokens, not raw substrings:
    # the fixture's strictly increasing values still collide as string
    # prefixes (e.g. "+0.11" is a literal prefix of "+0.111".."+0.119"),
    # so a naive `out.count(formatted)` overcounts unrelated values that
    # merely start with the same digits.
    tokens = out.split()
    for key, value in properties.items():
        formatted = f"{value:+.4g}"
        assert tokens.count(formatted) == 1, (key, formatted)


def test_matrix_path_prints_joint_token_once_per_row_not_per_key(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # A flat rendering would repeat "shoulder" once per signal (five
    # times here: qpos_shoulder ... qfrc_passive_shoulder). The matrix
    # renders it once, as the row label.
    for token, _, _ in _ARTICULATION_TOKENS:
        assert out.count(token) == 1, (token, out.count(token))

    for token, _, _ in _BASE_TOKENS:
        assert out.count(token) == 1, (token, out.count(token))


def test_matrix_path_shows_key_rule_and_legend_lines(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "key = <signal>_<joint>" in out
    assert (
        "hinge rows: rad, rad/s, rad/s², N·m · "
        "slide rows: m, m/s, m/s², N"
    ) in out


def test_matrix_path_orders_base_pose_family_before_articulation(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    positions = {
        token: out.index(token)
        for token, _, _ in _BASE_TOKENS + _ARTICULATION_TOKENS
    }
    assert max(positions["rootx"], positions["rooty"]) < min(
        positions["shoulder"], positions["elbow"], positions["wrist"]
    )


def test_matrix_path_renders_sensor_actuator_tables_in_block(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "torso" in out
    assert "site_xpos" in out
    # Sensors/Actuators now render nested inside the entity block, with
    # no OWNER column (the enclosing block is the owner).
    assert "Sensors" in out
    assert "Actuators" in out
    assert out.count("link1_touch") == 1
    assert out.count("motor1") == 1
    assert "OWNER" not in out


def test_family_table_columns_stay_separated_for_wide_values(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    row = next(
        line for line in out.splitlines()
        if "+0.9996" in line and "+0.0002462" in line
    )
    # 1 row-label token + 4 xquat columns (w, x, y, z). A column sized
    # from its header label alone (not its widest formatted value) lets
    # "+0.9996" and "+0.0002462" run together with no separating space.
    assert len(row.split()) == 5, row


def test_matrix_path_leftover_key_without_metadata_still_shown(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "custom_extra" in out


def test_single_joint_entity_renders_matrix(capsys):
    # One scalar-keyed joint now meets the >=1 eligibility threshold:
    # single-joint articulated entities (cabinet doors, microwave doors)
    # get the same joints x signals matrix as multi-joint robots.
    entity = _entity({
        "qpos_joint1": 1.0,
        "qvel_joint1": 0.2,
    })
    metadata = {
        "qpos_joint1": PropertyMeta(group="joint angle", note="rad"),
        "qvel_joint1": PropertyMeta(group="joint angular velocity", note="rad/s"),
    }

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "key = <signal>_<joint>" in out
    assert out.count("joint1") == 1


def test_family_tables_render_without_matrix_eligibility(capsys):
    # A free-floating object has zero scalar hinge/slide joints, so no
    # matrix -- but its body-pose keys still belong in family tables,
    # not the wrapped grouped listing.
    entity = _entity(
        {
            "xpos_obj_main_0": 3.564,
            "xpos_obj_main_1": -0.379,
            "xpos_obj_main_2": 0.9357,
            "mass": 0.25,
        },
        cls="object",
    )
    metadata = {
        f"xpos_obj_main_{axis}": PropertyMeta(group="body position", note="m")
        for axis in range(3)
    }
    metadata["mass"] = PropertyMeta(group="other", note="kg")

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # Family table renders the token as a row label, not "key = value".
    assert "obj_main" in out
    assert "xpos_obj_main_0 =" not in out
    # Non-family scalar still falls through to the grouped listing.
    assert "mass" in out


def test_family_tables_render_for_entity_with_no_metadata(capsys):
    # An entity with no metadata at all still gets its family tables (the
    # split is by key prefix, not metadata); only the non-family leftover
    # falls to the flat preview, and every value renders exactly once.
    entity = _entity(
        {
            "xpos_obj_main_0": 3.564,
            "xpos_obj_main_1": -0.379,
            "xpos_obj_main_2": 0.9357,
            "sensordata_touch": 1.5,
            "mass": 0.25,
        },
        cls="object",
    )

    _print_property_table(
        {0: entity}, labels=None, verbose=False, metadata={},
    )
    out = capsys.readouterr().out

    # Family table for xpos; nested Sensors table for the sensor key;
    # both render the token, not a flat "key = value" line.
    assert "obj_main" in out
    assert "xpos_obj_main_0 =" not in out
    assert "Sensors" in out
    assert "sensordata_touch =" not in out
    # The non-family scalar is the only row in the leftover KEY table.
    mass_row = next(line for line in out.splitlines() if "mass" in line)
    assert "+0.25" in mass_row
    assert "mass = +0.25" not in out
    # Nothing rendered twice: each value appears once as a whole token.
    for value in ("+3.564", "-0.379", "+0.9357", "+1.5", "+0.25"):
        assert out.split().count(value) == 1, (value, out.split().count(value))


def test_verbose_mode_bypasses_matrix_even_when_eligible(capsys):
    properties, metadata = _synthetic_matrix_fixture()
    entity = _entity(properties)

    _print_property_table(
        {0: entity}, labels=None, verbose=True, metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # Verbose mode is byte-identical to the pre-matrix grouped rendering:
    # no matrix key-rule line, and every key appears in the old flat form.
    assert "key = <signal>_<joint>" not in out
    assert "qpos_shoulder" in out


def _matrix_values(qpos, qvel):
    values = {signal: None for signal in _JOINT_SIGNALS}
    values["qpos"] = qpos
    values["qvel"] = qvel
    return values


def test_matrix_table_columns_stay_separated_for_wide_values(capsys):
    # Mirrors test_family_table_columns_stay_separated_for_wide_values, but
    # against _print_matrix_table directly: a column sized from its header
    # label alone (not its widest formatted value) lets a wide qpos/qvel
    # pair run together with no separating space.
    rows = [
        (
            "joint1",
            "hinge (rad)",
            _matrix_values(1.234567e-10, 9.87654321e12),
        ),
        (
            "joint2",
            "hinge (rad)",
            _matrix_values(0.5, 0.25),
        ),
    ]

    _print_matrix_table(rows)
    out = capsys.readouterr().out

    row = next(
        line for line in out.splitlines()
        if "+1.235e-10" in line and "+9.877e+12" in line
    )
    # 1 joint token + 2 type-label words ("hinge", "(rad)") + 2 signal
    # columns (qpos, qvel).
    assert len(row.split()) == 5, row


# ---------------------------------------------------------------------------
# Nested per-entity Sensors / Actuators tables: each entity's own
# sensordata_*/ctrl_*/actuator_force_* keys render inside its block;
# orphan "sensors"/"actuators" entities render as ordinary blocks.
# ---------------------------------------------------------------------------

def _stub_adapter(labels, metadata):
    class _Adapter:
        entity_labels = labels

        @staticmethod
        def property_metadata(entity_id):
            return metadata.get(entity_id, {})

    return _Adapter()


def _sensor_actuator_fixture():
    # Entity 0: a matrix-eligible robot (2 joints) that also owns a sensor.
    # Entity 1: an orphan "sensors" entity (no label -> owner falls back to
    # the class name) with a second sensor.
    # Entity 2: an orphan "actuators" entity (no label) with one actuator.
    entities = {
        0: _entity(
            {
                "qpos_rootx": 0.11,
                "qpos_shoulder": 0.22,
                "sensordata_wrist_touch": 1.5,
            },
            entity_id=0,
            cls="robot",
        ),
        1: _entity(
            {"sensordata_ambient_light": 42.0},
            entity_id=1,
            cls="sensors",
        ),
        2: _entity(
            {
                "ctrl_motor1": 0.5,
                "actuator_force_motor1": 12.345,
            },
            entity_id=2,
            cls="actuators",
        ),
    }

    metadata = {
        0: {
            "qpos_rootx": PropertyMeta(
                group="base displacement (rel. spawn)", note="m",
            ),
            "qpos_shoulder": PropertyMeta(
                group="joint angle", note="rad",
            ),
            "sensordata_wrist_touch": PropertyMeta(
                group="sensors: touch", note="N",
            ),
        },
        1: {
            "sensordata_ambient_light": PropertyMeta(
                group="sensors: rangefinder", note="lux",
            ),
        },
        2: {
            "ctrl_motor1": PropertyMeta(group="actuator command"),
            "actuator_force_motor1": PropertyMeta(
                group="actuator output torque", note="N·m",
            ),
        },
    }

    labels = {0: "arm"}

    return entities, metadata, labels


def test_print_discovery_nests_sensor_tables_in_owner_blocks(capsys):
    entities, metadata, labels = _sensor_actuator_fixture()
    adapter = _stub_adapter(labels, metadata)

    print_discovery(adapter, entities, verbose=False)
    out = capsys.readouterr().out

    # Owned sensor renders inside the robot's block, before the orphan
    # "sensors" entity's block header.
    robot_block = out.index("[0] robot")
    orphan_block = out.index("[1] sensors")
    assert robot_block < out.index("wrist_touch") < orphan_block
    # Orphan sensor renders inside the orphan entity's own block.
    assert out.index("ambient_light") > orphan_block
    assert "lux" in out
    assert "OWNER" not in out


def test_print_discovery_nests_actuator_table_in_orphan_block(capsys):
    entities, metadata, labels = _sensor_actuator_fixture()
    adapter = _stub_adapter(labels, metadata)

    print_discovery(adapter, entities, verbose=False)
    out = capsys.readouterr().out

    actuator_block = out.index("[2] actuators")
    assert out.index("motor1") > actuator_block
    assert "N·m" in out
    assert "OWNER" not in out


def test_verbose_lists_sensor_keys_flat_with_no_summary_tables(capsys):
    entities, metadata, labels = _sensor_actuator_fixture()
    adapter = _stub_adapter(labels, metadata)

    print_discovery(adapter, entities, verbose=True)
    out = capsys.readouterr().out

    # Verbose keeps the flat grouped listing (which by design shows a
    # key in its group line and again in its exact-value line); the
    # scene-wide summary tables are gone, so no tabular NAME/OWNER
    # header appears anywhere.
    assert "sensordata_wrist_touch" in out
    assert "ctrl_motor1" in out
    assert "NAME" not in out
    assert "OWNER" not in out


def test_print_discovery_lists_sensor_actuator_keys_exactly_once(capsys):
    # Regression: sensor/actuator rows used to render twice -- once in the
    # owning entity's family tables, again in the per-sim summary tables.
    properties, metadata = _synthetic_matrix_fixture()
    entities = {0: _entity(properties)}
    adapter = _stub_adapter({0: "arm"}, {0: metadata})

    print_discovery(adapter, entities, verbose=False)
    out = capsys.readouterr().out

    # The example-formula footer may legitimately reuse one of these keys;
    # the once-only guarantee is about the tables above it.
    tables = out.split("Example formula")[0]

    for token in ("link1_touch", "link1_accel", "motor1"):
        assert tables.count(token) == 1, (token, tables.count(token))

    # Nothing hidden: every key's value still appears somewhere in the
    # full discovery output.
    for key, value in properties.items():
        formatted = f"{value:+.4g}"
        assert formatted in out, (key, formatted)


def test_print_discovery_sensor_actuator_tables_omitted_when_absent(capsys):
    entity = _entity({"mass": 1.0}, entity_id=0, cls="object")
    adapter = _stub_adapter({}, {})

    print_discovery(adapter, {0: entity}, verbose=False)
    out = capsys.readouterr().out

    assert "Sensors" not in out
    assert "Actuators" not in out


# ---------------------------------------------------------------------------
# Planar-XML end-to-end (mirrors test_mujoco_property_meta.py's fixture):
# 4 scalar hinge/slide joints on one subtree-scoped robot entity.
# ---------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

from specola.adapters.mujoco import BodyRule, MujocoAdapter  # noqa: E402
from test_mujoco_property_meta import _PLANAR_XML  # noqa: E402


def _planar_robot_entity_and_meta():
    class _PlanarEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(_PLANAR_XML)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

    adapter = MujocoAdapter(
        _PlanarEnv(),
        body_rules=[BodyRule(match="torso", cls="robot", joint_scope="subtree")],
    )
    observation = adapter.reset()
    entity_id = next(iter(observation.perceived_entities))
    entity = observation.perceived_entities[entity_id]
    meta = adapter.property_metadata(entity_id)
    return entity, meta


def test_planar_xml_matrix_has_one_row_per_joint(capsys):
    entity, meta = _planar_robot_entity_and_meta()

    _print_property_table(
        {entity.entity_id: entity}, labels=None, verbose=False,
        metadata={entity.entity_id: meta},
    )
    out = capsys.readouterr().out

    for token in ("rootx", "rootz", "rooty", "piston_slide"):
        matrix_rows = [
            line
            for line in out.splitlines()
            if line.split()[:1] == [token]
        ]
        assert len(matrix_rows) == 1, (token, len(matrix_rows))

    assert "key = <signal>_<joint>" in out
    assert (
        "hinge rows: rad, rad/s, rad/s², N·m · "
        "slide rows: m, m/s, m/s², N"
    ) in out


def test_pager_command_includes_dash_s(monkeypatch):
    monkeypatch.delenv("PAGER", raising=False)
    assert any("S" in flag for flag in pager_command() if flag.startswith("-"))


# ---------------------------------------------------------------------------
# Issue #68: the vocabulary a spec can filter on must be visible in discover
# ---------------------------------------------------------------------------

def _tagged_fixture():
    """The synthetic fixture with structural tags on the joint keys."""
    properties, metadata = _synthetic_matrix_fixture()

    tags = {
        "rootx": ("base",),
        "rooty": ("base",),
        "shoulder": ("arm", "right"),
        "elbow": ("arm", "right"),
        "wrist": ("arm", "right"),
    }

    for token, joint_tags in tags.items():
        key = f"qpos_{token}"
        existing = metadata[key]
        metadata[key] = PropertyMeta(
            group=existing.group,
            note=existing.note,
            tags=joint_tags,
        )

    return properties, metadata


def test_matrix_shows_structural_tags(capsys):
    properties, metadata = _tagged_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # `robot_*[qvel_* @ arm]` is only writable if `arm` is discoverable.
    assert "TAGS" in out
    assert "arm right" in out
    assert "base" in out


def test_matrix_omits_tags_column_when_no_joint_is_tagged(capsys):
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "TAGS" not in out


def test_family_tables_state_their_group_and_unit(capsys):
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    # xpos rendered as a family table; without this the 60 body-pose keys
    # on a real robot show neither unit nor group anywhere in discover.
    assert "body position" in out


def test_family_table_names_a_group_once_when_notes_are_component_labels(
    capsys,
):
    # xquat's notes are "quat w"/"quat x"/... -- component labels, not
    # units -- so pairing group with note would repeat one group four
    # times. The column headers already carry the components.
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert out.count("body orientation") == 1


def test_body_force_families_render_as_tables_with_both_groups(capsys):
    # Without a _FAMILY_PREFIXES entry each of these falls to the flat
    # listing: 6 rows per body instead of one 6-column row. The family
    # annotation must name both unit groups, as it does for cvel.
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    for family in ("cfrc_ext", "cfrc_int", "cacc"):
        assert family in out, family
        # every index of the family appears on one row, not six
        assert f"{family}_torso_0" not in out, family

    for group in (
        "body external torque (N·m)",
        "body external force (N)",
        "body internal torque (N·m)",
        "body internal force (N)",
        "body angular acceleration (rad/s²)",
        "body com-frame linear acceleration (m/s²)",
    ):
        assert group in out, group


def test_actuator_length_and_velocity_join_the_nested_actuator_table(capsys):
    # actuator_length/actuator_velocity are actuator families, so they are
    # consumed by the nested Actuators table rather than printed as their
    # own family tables -- the same treatment ctrl and actuator_force get.
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "Actuators" in out
    assert out.count("motor1") == 1
    assert "length" in out
    assert "velocity" in out
    # not rendered as standalone family tables
    assert "\n      actuator_length\n" not in out
    assert "\n      actuator_velocity\n" not in out


def test_passive_signal_is_a_matrix_column(capsys):
    from specola.cli_common import _JOINT_SIGNALS as signals

    assert "qfrc_passive" in signals

    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    assert "qfrc_passive" in out


def test_flat_listing_shows_tags_when_supplied(capsys):
    # The flat KEY table is the only listing a non-MuJoCo adapter reaches,
    # so an adapter that supplies tags must be able to show them there.
    # No bundled adapter hits this combination today: MuJoCo tags sit on
    # joints, which render in the matrix instead.
    from specola.cli_common import _print_grouped_block, _group_properties

    entity = _entity({"glucose_mgdl": 142.0, "insulin_U_per_hr": 0.85})
    metadata = {
        "glucose_mgdl": PropertyMeta(
            group="glucose", note="mg/dL", tags=("sensor",),
        ),
        "insulin_U_per_hr": PropertyMeta(
            group="insulin", note="U/hr", tags=("pump",),
        ),
    }

    _print_grouped_block(
        entity,
        _group_properties(entity, metadata),
        verbose=False,
        metadata=metadata,
    )
    out = capsys.readouterr().out

    assert "TAGS" in out
    assert "sensor" in out
    assert "pump" in out


def test_flat_listing_omits_tags_column_when_none_supplied(capsys):
    from specola.cli_common import _print_grouped_block, _group_properties

    entity = _entity({"glucose_mgdl": 142.0})
    metadata = {"glucose_mgdl": PropertyMeta(group="glucose", note="mg/dL")}

    _print_grouped_block(
        entity,
        _group_properties(entity, metadata),
        verbose=False,
        metadata=metadata,
    )
    out = capsys.readouterr().out

    assert "TAGS" not in out
    assert "glucose" in out


def test_nested_actuator_table_lists_every_group_on_the_row(capsys):
    """
    One actuator's four keys can sit in four different groups now that
    the actuator families split by unit. The GROUP column has to say so:
    showing only the first group labels a row `actuator command` when
    three of its four cells are in other groups entirely, which is what
    it did while every actuator key shared one `actuators` group.

    The UNIT column already accumulates all four notes, so this makes the
    two columns agree.
    """
    properties, metadata = _synthetic_matrix_fixture()

    _print_property_table(
        {0: _entity(properties)},
        labels=None,
        verbose=False,
        metadata={0: metadata},
    )
    out = capsys.readouterr().out

    row = next(
        line for line in out.splitlines()
        if "motor1" in line and "actuator" in line
    )

    for group in (
        "actuator command",
        "actuator output torque",
        "actuator angle",
        "actuator angular velocity",
    ):
        assert group in row, (group, row)
