"""
cli_common.py — Shared CLI Infrastructure
==========================================

Shared support for:

    write_specs.py
        Discovers entities and properties, validates authored formulas,
        and writes specification files.

    run_monitor.py
        Loads a simulator adapter and evaluates existing specifications.

This module is responsible for:

- terminal formatting;
- simulator adapter lookup and construction;
- resetting an adapter once at startup;
- presenting discovered entities and scalar properties;
- providing a dependency-free mock adapter.

It does not:

- define valid STL components;
- validate formulas;
- determine binding eligibility;
- assume a fixed entity or property schema;
- contain simulator-specific extraction logic.
"""

from __future__ import annotations

import os
import sys
import textwrap
from numbers import Real
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from .entities import EntityState, TrackState
from .query_config import QueryConfiguration


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

def _tty() -> bool:
    return (
        "NO_COLOR" not in os.environ
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )


class C:
    BOLD = "\033[1m" if _tty() else ""
    DIM = "\033[2m" if _tty() else ""
    CYAN = "\033[36m" if _tty() else ""
    GREEN = "\033[32m" if _tty() else ""
    AMBER = "\033[33m" if _tty() else ""
    RED = "\033[31m" if _tty() else ""
    MAGENTA = "\033[35m" if _tty() else ""
    RST = "\033[0m" if _tty() else ""


# Canonical state -> color-name mapping shared by every run renderer.
# Renderers translate the name to their own styling (ANSI helper here,
# Rich style in the Textual dashboard); color supplements the state
# words and is never the sole encoding.
STATE_COLOR_NAMES = {
    "satisfied": "green",
    "violated": "red",
    "pending": "amber",
    "undetermined": "amber",
    "rewarming": "cyan",
    "suspended": "dim",
    "missing_data": "magenta",
    "error": "red",
}


def bold(value: object) -> str:
    return f"{C.BOLD}{value}{C.RST}"


def dim(value: object) -> str:
    return f"{C.DIM}{value}{C.RST}"


def cyan(value: object) -> str:
    return f"{C.CYAN}{value}{C.RST}"


def green(value: object) -> str:
    return f"{C.GREEN}{value}{C.RST}"


def amber(value: object) -> str:
    return f"{C.AMBER}{value}{C.RST}"


def red(value: object) -> str:
    return f"{C.RED}{value}{C.RST}"


def magenta(value: object) -> str:
    return f"{C.MAGENTA}{value}{C.RST}"


# ---------------------------------------------------------------------------
# Display conventions
# ---------------------------------------------------------------------------

_CLASS_COLORS = {
    "robot": cyan,
    "pedestrian": cyan,
    "vehicle": cyan,
    "object": green,
    "obj": green,
    "gripper": amber,
    "region": amber,
    "goalzone": amber,
    "dangerzone": amber,
}

_REGION_HALF_EXTENTS = frozenset(
    {
        "hw",
        "hh",
        "hd",
    }
)

def _class_string(entity: EntityState) -> str:
    """Return the adapter-defined string class for display."""
    return entity.entity_class


def _is_region_like(
    entity: EntityState,
) -> bool:
    """
    Whether an entity should receive the specialized region presentation.

    A string class ending in "region" is treated as a region. An adapter may
    also use another class name while exposing all three conventional
    half-extent properties.
    """
    entity_class = entity.entity_class.lower()

    if entity_class.endswith("region"):
        return True

    return _REGION_HALF_EXTENTS.issubset(
        entity.properties.keys()
    )


def _format_vector(
    values: Sequence[float],
    precision: int = 3,
) -> str:
    return (
        "("
        + ", ".join(
            f"{float(value):.{precision}f}"
            for value in values
        )
        + ")"
    )


def _print_wrapped(
    prefix: str,
    text: str,
    *,
    width: int = 112,
    subsequent_indent: Optional[str] = None,
) -> None:
    if subsequent_indent is None:
        subsequent_indent = " " * len(prefix)

    lines = textwrap.wrap(
        text,
        width=max(20, width - len(prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )

    if not lines:
        print(prefix)
        return

    print(prefix + lines[0])

    for line in lines[1:]:
        print(subsequent_indent + line)


# ---------------------------------------------------------------------------
# Discovery presentation
# ---------------------------------------------------------------------------

def _fit_label(
    label: str,
    width: int,
) -> str:
    """
    Pad to ``width``, truncating with an explicit marker when needed.

    The final two columns are always separator space, so a truncated or
    maximal label can never collide with the next table column.
    """
    if len(label) <= width - 2:
        return f"{label:<{width}}"

    return label[: width - 3] + "…  "


def _label_width(
    values: Iterable[str],
    *,
    minimum: int,
    maximum: int = 48,
) -> int:
    longest = max(
        [len(value) for value in values]
        or [0]
    )

    return max(
        minimum,
        min(longest + 2, maximum),
    )


# Display order for entity blocks and the Entities overview table:
# class-first (robot, then gripper, then objects, then orphan
# sensor/actuator entities), id within a class. Region-like entities
# sort last; outside --verbose they only appear in their own tables.
_CLASS_DISPLAY_ORDER: Dict[str, int] = {
    "robot": 0,
    "gripper": 1,
    "object": 2,
    "sensors": 3,
    "actuators": 4,
}


def _display_sort_key(
    entity_id: int,
    entity: EntityState,
) -> Tuple[int, str, int]:
    if _is_region_like(entity):
        rank = 6
    else:
        rank = _CLASS_DISPLAY_ORDER.get(
            entity.entity_class,
            5,
        )

    return (rank, entity.entity_class, entity_id)


def _print_entity_table(
    entities: Mapping[int, EntityState],
    *,
    labels: Optional[Mapping[int, str]] = None,
) -> None:
    label_values = [
        (
            labels.get(entity_id, "")
            if labels is not None
            else ""
        )
        for entity_id in entities
    ]

    label_width = _label_width(
        label_values,
        minimum=len("BODY / LABEL") + 2,
    )

    rule_width = 14 + 18 + 8 + label_width + 25 + 6

    print()
    print(bold("  Entities"))

    if not entities:
        print(
            dim(
                "  No non-region entities in the "
                "initial observation."
            )
        )
        print()
        return

    print(dim("  " + "─" * rule_width))
    print(
        dim(
            f"  {'CLASS':<14}"
            f"{'PREFIX':<18}"
            f"{'ID':>5}   "
            f"{'BODY / LABEL':<{label_width}}"
            f"{'POSITION':<25}"
            f"{'PROPS':>6}"
        )
    )
    print(dim("  " + "─" * rule_width))

    for entity_id in sorted(
        entities,
        key=lambda eid: _display_sort_key(
            eid,
            entities[eid],
        ),
    ):
        entity = entities[entity_id]
        entity_class = _class_string(entity)

        color = _CLASS_COLORS.get(
            entity_class.lower(),
            dim,
        )

        label = (
            labels.get(entity_id, "")
            if labels is not None
            else ""
        )

        position = (
            "—"
            if entity.position is None
            else _format_vector(
                entity.position,
                precision=3,
            )
        )

        property_count = len(
            entity.properties
        )

        class_column = color(
            f"{entity_class:<14}"
        )

        prefix_column = bold(
            f"{entity_class + '_*':<18}"
        )

        print(
            f"  {class_column}"
            f"{prefix_column}"
            f"{entity_id:>5}   "
            f"{dim(_fit_label(label, label_width))}"
            f"{dim(f'{position:<25}')}"
            f"{property_count:>6}"
        )

    print(dim("  " + "─" * rule_width))
    print()


# Preferred display order: pose, then velocity, then acceleration, within
# each provenance. Groups an adapter emits that are absent here still
# render, sorted alphabetically after these -- see _group_properties.
_GROUP_ORDER = (
    "joint position",
    "joint orientation",
    "joint angle",
    "joint displacement",
    "joint linear velocity",
    "joint angular velocity",
    "joint linear acceleration",
    "joint angular acceleration",
    "base angle (rel. spawn)",
    "base displacement (rel. spawn)",
    "base linear velocity",
    "base angular velocity",
    "base linear acceleration",
    "base angular acceleration",
    "applied torques",
    "applied forces",
)

_OTHER_GROUP = "other"


def _group_properties(
    entity: EntityState,
    metadata: Optional[Mapping[str, Any]],
    keys: Optional[Iterable[str]] = None,
) -> List[Tuple[str, List[Tuple[str, Optional[str]]]]]:
    """
    Order an entity's property keys into display groups.

    Returns ``[(group_label, [(key, suffix_or_None), ...]), ...]`` covering
    every key the entity currently exposes; keys without metadata land in
    the trailing "other" group. ``keys`` restricts the keys considered
    (default: every key of ``entity.properties``) -- used to render the
    keys a matrix-eligible entity's matrix/family tables did not consume.
    """
    groups: Dict[str, List[Tuple[str, Optional[str]]]] = {}

    for key in sorted(
        entity.properties
        if keys is None
        else keys
    ):
        meta = (
            metadata.get(key)
            if metadata
            else None
        )

        if meta is None:
            group = _OTHER_GROUP
            suffix: Optional[str] = None
        else:
            group = meta.group or _OTHER_GROUP
            suffix = meta.note

        groups.setdefault(
            group,
            [],
        ).append((key, suffix))

    ordered: List[
        Tuple[str, List[Tuple[str, Optional[str]]]]
    ] = []

    for group in _GROUP_ORDER:
        if group in groups:
            ordered.append(
                (group, groups.pop(group))
            )

    for group in sorted(
        name
        for name in groups
        if name != _OTHER_GROUP
    ):
        ordered.append(
            (group, groups.pop(group))
        )

    if _OTHER_GROUP in groups:
        ordered.append(
            (_OTHER_GROUP, groups.pop(_OTHER_GROUP))
        )

    return ordered


def _print_grouped_block(
    entity: EntityState,
    grouped: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
    *,
    verbose: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Print one entity's already-grouped keys as an aligned KEY table.

    Shared by the pre-matrix grouped listing and by the matrix path's
    fallback for keys the matrix and family tables did not consume, so
    the table formatting stays identical between the two. Rows keep the
    ``_group_properties`` ordering: canonical groups first, then
    alphabetical, then "other" — the same order the wrapped listing used.
    """
    if not grouped:
        return

    rows = [
        (
            key,
            f"{float(entity.properties[key]):+.4g}",
            suffix or "—",
            group,
        )
        for group, items in grouped
        for key, suffix in items
    ]

    key_width = max(
        [len("KEY")]
        + [len(key) for key, _, _, _ in rows]
    ) + 2

    value_width = max(
        [len("VALUE")]
        + [len(value) for _, value, _, _ in rows]
    ) + 2

    unit_width = _label_width(
        [unit for _, _, unit, _ in rows],
        minimum=len("UNIT") + 2,
    )

    # Tags are filterable with `@` just as groups are, so an adapter that
    # supplies them needs them readable here -- this table is the only
    # listing a non-MuJoCo adapter reaches. Omitted when none are set.
    tag_labels = {
        key: " ".join(
            getattr((metadata or {}).get(key), "tags", ()) or ()
        )
        for key, _, _, _ in rows
    }

    show_tags = any(tag_labels.values())

    group_width = (
        _label_width(
            [group for _, _, _, group in rows],
            minimum=len("GROUP") + 2,
        )
        if show_tags
        else 0
    )

    print(dim(
        f"      {'KEY':<{key_width}}"
        f"{'VALUE':>{value_width}}"
        f"  {'UNIT':<{unit_width}}"
        + (f"{'GROUP':<{group_width}}TAGS" if show_tags else "GROUP")
    ))

    for key, value, unit, group in rows:
        print(
            f"      {key:<{key_width}}"
            f"{value:>{value_width}}"
            f"  {_fit_label(unit, unit_width)}"
            + (
                f"{_fit_label(group, group_width)}"
                f"{dim(tag_labels[key])}"
                if show_tags
                else f"{dim(group)}"
            )
        )

    if verbose:
        for group, items in grouped:
            for key, suffix in items:
                value = entity.properties[key]

                print(
                    dim(
                        f"        {key:<48} "
                        f"= {float(value):+.6g}"
                    )
                )


# ---------------------------------------------------------------------------
# Joints x signals matrix
# ---------------------------------------------------------------------------
#
# Per-DOF joint keys share the "<signal>_<joint-token>" structure (scalar
# hinge/slide joints -- exactly one DOF, so no trailing numeric index).
# The ordered tuple below mirrors adapters.mujoco._SIGNAL_ADDRESSES; it is
# duplicated rather than imported so this module stays usable without
# mujoco installed (adapter-optional).
_JOINT_SIGNALS: Tuple[str, ...] = (
    "qpos",
    "qvel",
    "qacc",
    "qfrc_actuator",
    "qfrc_constraint",
    "qfrc_bias",
    "qfrc_applied",
    "qfrc_smooth",
    "qfrc_passive",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_fluid",
    "qfrc_gravcomp",
)

# A scalar qpos key's PropertyMeta.group identifies its joint as matrix
# row-eligible: the "base ... (rel. spawn)" pair for a slide/hinge on the
# entity's own matched body (cheetah root joints, robosuite virtual mobile
# bases), "joint angle"/"joint displacement" for articulation further
# down the subtree. Each pair splits by unit, so no group mixes m with rad.
# Free/ball joints use other groups ("position", "orientation", ...) and
# are excluded -- their multi-DOF keys carry a numeric suffix and do not
# fit one matrix row per joint.
_BASE_POSE_GROUPS = frozenset(
    {
        "base angle (rel. spawn)",
        "base displacement (rel. spawn)",
    }
)

_ARTICULATION_GROUPS = frozenset(
    {
        "joint angle",
        "joint displacement",
    }
)

# Non-per-DOF families recognized by _family_key_split: body pose, body
# velocity, body force and acceleration, sensors, actuators, and the
# gripper's own site position. Tuple order is render order
# (_print_family_tables iterates it): pose, then the com-based 6-vectors,
# then sensors and actuators. Body-pose, body-6-vector and site families
# render as compact per-family tables alongside the matrix; sensor/actuator
# families are consumed there but render only in the nested per-entity
# Sensors/Actuators tables (_NESTED_TABLE_FAMILIES below).
_FAMILY_PREFIXES: Tuple[str, ...] = (
    "xpos",
    "xquat",
    "cvel",
    "cfrc_ext",
    "cfrc_int",
    "cacc",
    "sensordata",
    "ctrl",
    "actuator_force",
    "actuator_length",
    "actuator_velocity",
    "site_xpos",
)

# Families whose keys always carry the same fixed component suffixes.
_FAMILY_AXIS_LABELS: Dict[str, Tuple[str, ...]] = {
    "xpos": ("0", "1", "2"),
    "site_xpos": ("0", "1", "2"),
    "xquat": ("w", "x", "y", "z"),
    "cvel": tuple(str(index) for index in range(6)),
    "cfrc_ext": tuple(str(index) for index in range(6)),
    "cfrc_int": tuple(str(index) for index in range(6)),
    "cacc": tuple(str(index) for index in range(6)),
}

# Families with exactly one column per token (no trailing suffix to
# strip): the remainder after the prefix *is* the token.
_FAMILY_SINGLE_COLUMN: Dict[str, str] = {
    "ctrl": "ctrl",
    "actuator_force": "force",
    "actuator_length": "length",
    "actuator_velocity": "velocity",
}

# Families rendered in the nested per-entity Sensors and Actuators tables
# (see _sensor_actuator_rows / _print_sensor_actuator_table).
_SENSOR_FAMILIES: FrozenSet[str] = frozenset({"sensordata"})
_ACTUATOR_FAMILIES: FrozenSet[str] = frozenset(
    {
        "ctrl",
        "actuator_force",
        "actuator_length",
        "actuator_velocity",
    }
)

# Families deferred entirely to the nested Sensors/Actuators tables: the
# per-entity family path consumes their keys without printing, so each
# sensor/actuator key renders exactly once.
_NESTED_TABLE_FAMILIES: FrozenSet[str] = _SENSOR_FAMILIES | _ACTUATOR_FAMILIES


def _joint_type_label(meta: PropertyMeta) -> str:
    """
    "hinge (rad)" or "slide (m)" for one matrix row, from its qpos key's
    metadata. Group and note agree since the base-pose groups split by
    unit, but the note is read because it states the unit directly and
    the legend's hinge/slide split is a statement about units.
    """
    if meta.note == "rad":
        return "hinge (rad)"

    if meta.note == "m":
        return "slide (m)"

    # Defensive fallback: every matrix-eligible group is produced with a
    # "rad" or "m" note today, but do not silently mislabel a future one.
    return (
        "hinge (rad)"
        if "angle" in meta.group
        else "slide (m)"
    )


def _matrix_rows(
    entity: EntityState,
    metadata: Mapping[str, Any],
) -> List[Tuple[str, str, Dict[str, Optional[float]]]]:
    """
    Per-joint matrix rows for one entity, or ``[]`` below the eligibility
    threshold.

    Returns ``[(joint_token, type_label, {signal: value_or_None}), ...]``,
    base-pose-family rows first (each ordered alphabetically by token),
    then articulation rows, alphabetically by token. Empty when the
    entity has no scalar-keyed joints -- callers fall back to
    the flat/grouped listing in that case.
    """
    base_tokens: List[str] = []
    articulation_tokens: List[str] = []

    for key, meta in metadata.items():
        if (
            not key.startswith("qpos_")
            or key not in entity.properties
        ):
            continue

        token = key[len("qpos_"):]

        if meta.group in _BASE_POSE_GROUPS:
            base_tokens.append(token)
        elif meta.group in _ARTICULATION_GROUPS:
            articulation_tokens.append(token)

    tokens = sorted(base_tokens) + sorted(articulation_tokens)

    if not tokens:
        return []

    rows: List[Tuple[str, str, Dict[str, Optional[float]]]] = []

    for token in tokens:
        type_label = _joint_type_label(
            metadata[f"qpos_{token}"]
        )

        values = {
            signal: entity.properties.get(f"{signal}_{token}")
            for signal in _JOINT_SIGNALS
        }

        rows.append((token, type_label, values))

    return rows


def _matrix_consumed_keys(
    rows: List[Tuple[str, str, Dict[str, Optional[float]]]],
) -> FrozenSet[str]:
    """
    The per-DOF keys a joints x signals matrix renders, given its rows.

    Pure (no printing), so both ``_print_matrix_table`` (terminal) and the
    HTML report can agree on which keys the matrix already covers without
    reparsing. A signal only counts as consumed if it has at least one
    non-None value across ``rows`` -- the same column filter each renderer
    applies before showing that signal at all.
    """
    columns = [
        signal
        for signal in _JOINT_SIGNALS
        if any(
            values[signal] is not None
            for _, _, values in rows
        )
    ]

    return frozenset(
        f"{signal}_{token}"
        for token, _, values in rows
        for signal in columns
        if values[signal] is not None
    )


def _print_matrix_table(
    rows: List[Tuple[str, str, Dict[str, Optional[float]]]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> FrozenSet[str]:
    """
    Print the joints x signals matrix; return the keys it consumed.

    ``metadata`` supplies each joint's structural tags for the TAGS
    column. A spec can filter on a tag (``robot_*[qvel_* @ arm]``), so
    the tag vocabulary has to be readable here -- it appears nowhere else
    in discovery. The column is omitted when no joint carries a tag.
    """
    columns = [
        signal
        for signal in _JOINT_SIGNALS
        if any(
            values[signal] is not None
            for _, _, values in rows
        )
    ]

    token_width = max(
        [len(token) for token, _, _ in rows]
        + [len("JOINT")]
    ) + 2

    type_width = max(
        [len(label) for _, label, _ in rows]
        + [len("TYPE")]
    ) + 2

    tag_labels = {
        token: " ".join(
            getattr(
                (metadata or {}).get(f"qpos_{token}"),
                "tags",
                (),
            )
            or ()
        )
        for token, _, _ in rows
    }

    show_tags = any(tag_labels.values())

    tag_width = (
        max(
            [len(label) for label in tag_labels.values()]
            + [len("TAGS")]
        ) + 2
        if show_tags
        else 0
    )

    # Pre-format every cell so column widths account for the widest
    # actual value, not just the (usually much shorter) signal name --
    # a header-only width lets a wide value (e.g. "+1.234e-100") run
    # into its neighbor with no separating space.
    cells = {
        (token, signal): (
            "—"
            if values[signal] is None
            else f"{values[signal]:+.4g}"
        )
        for token, _, values in rows
        for signal in columns
    }

    column_widths = {
        signal: max(
            [len(signal)]
            + [
                len(cells[token, signal])
                for token, _, _ in rows
            ]
        ) + 2
        for signal in columns
    }

    print(dim("      key = <signal>_<joint>"))

    header = (
        f"      {'JOINT':<{token_width}}"
        f"{'TYPE':<{type_width}}"
        + (f"{'TAGS':<{tag_width}}" if show_tags else "")
        + "".join(
            f"{signal:>{column_widths[signal]}}"
            for signal in columns
        )
    )
    print(dim(header))

    for token, type_label, values in rows:
        row_text = (
            f"      {token:<{token_width}}"
            f"{type_label:<{type_width}}"
        )

        if show_tags:
            row_text += f"{tag_labels[token]:<{tag_width}}"

        for signal in columns:
            row_text += (
                f"{cells[token, signal]:>{column_widths[signal]}}"
            )

        print(row_text)

    print(
        dim(
            "      hinge rows: rad, rad/s, rad/s², N·m · "
            "slide rows: m, m/s, m/s², N"
        )
    )
    print()

    return _matrix_consumed_keys(rows)


def _family_key_split(
    key: str,
) -> Optional[Tuple[str, str]]:
    """
    ``(family, remainder)`` for one non-per-DOF key, or ``None``.

    Longest prefix first: none of ``_FAMILY_PREFIXES`` is currently a
    prefix of another, but matching longest-first keeps that guarantee
    from becoming a silent requirement on future additions.
    """
    for family in sorted(
        _FAMILY_PREFIXES,
        key=len,
        reverse=True,
    ):
        prefix = f"{family}_"

        if key.startswith(prefix):
            return family, key[len(prefix):]

    return None


def _family_token_component(
    family: str,
    remainder: str,
) -> Tuple[str, str]:
    """Split one family's key remainder into (row token, column label)."""
    if family in _FAMILY_AXIS_LABELS:
        token, _, component = remainder.rpartition("_")
        return token, component

    if family in _FAMILY_SINGLE_COLUMN:
        return remainder, _FAMILY_SINGLE_COLUMN[family]

    # sensordata: dim-1 sensors have no trailing index (one column, named
    # "value"); dim>1 sensors number every component 0..dim-1.
    token, separator, last = remainder.rpartition("_")

    if separator and last.isdigit():
        return token, last

    return remainder, "value"


def _family_columns(
    family: str,
    table: Mapping[str, Mapping[str, str]],
) -> List[str]:
    if family in _FAMILY_AXIS_LABELS:
        return [
            component
            for component in _FAMILY_AXIS_LABELS[family]
            if any(
                component in components
                for components in table.values()
            )
        ]

    present = {
        component
        for components in table.values()
        for component in components
    }

    if present and all(component.isdigit() for component in present):
        return [
            str(index)
            for index in sorted(int(component) for component in present)
        ]

    return sorted(present)


def _print_family_tables(
    entity: EntityState,
    exclude: FrozenSet[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> FrozenSet[str]:
    """
    Print one compact table per non-per-DOF family present on ``entity``.

    Rows are the family's body/site tokens (alphabetical); columns are the
    family's indices/components. Sensor/actuator families are consumed
    without printing -- they render in the entity's nested
    Sensors/Actuators tables instead (printed by the caller). Returns the
    keys these tables consumed, so the caller can fall back to the
    flat/grouped listing for anything still unaccounted for.
    """
    buckets: Dict[str, Dict[str, Dict[str, str]]] = {}
    consumed = set()

    for key in entity.properties:
        if key in exclude:
            continue

        split = _family_key_split(key)

        if split is None:
            continue

        family, remainder = split

        if family in _NESTED_TABLE_FAMILIES:
            consumed.add(key)
            continue

        token, component = _family_token_component(family, remainder)

        buckets.setdefault(family, {}).setdefault(token, {})[
            component
        ] = key

    for family in _FAMILY_PREFIXES:
        table = buckets.get(family)

        if not table:
            continue

        columns = _family_columns(family, table)
        token_width = max(
            [len(token) for token in table]
            + [len("KEY")]
        ) + 2

        # Pre-format every cell so column widths account for the widest
        # actual value, not just the (usually much shorter) component
        # label -- see the matching comment in _print_matrix_table.
        cells = {
            (token, component): (
                "—"
                if table[token].get(component) is None
                else f"{entity.properties[table[token][component]]:+.4g}"
            )
            for token in table
            for component in columns
        }

        column_widths = {
            component: max(
                [len(component)]
                + [
                    len(cells[token, component])
                    for token in table
                ]
            ) + 2
            for component in columns
        }

        print(bold(f"      {family}"))

        # A family table has no UNIT/GROUP column -- its cells are one
        # quantity across components -- so the vocabulary a spec filters
        # on would otherwise be invisible for every key it renders. One
        # family can still span several groups (cvel is angular on 0-2
        # and linear on 3-5), so every distinct pair is listed.
        notes_by_group: Dict[str, List[Optional[str]]] = {}
        for token in sorted(table):
            for component in columns:
                source_key = table[token].get(component)
                meta = (metadata or {}).get(source_key)

                if meta is None or not meta.group:
                    continue

                notes = notes_by_group.setdefault(meta.group, [])

                if meta.note not in notes:
                    notes.append(meta.note)

        described = []
        for group, notes in notes_by_group.items():
            # One note across the family is its unit and worth stating.
            # Several means `note` is carrying component labels instead
            # (xquat's "quat w"/"quat x"/...), which the column headers
            # already show, so the group alone is the useful part.
            if len(notes) == 1 and notes[0]:
                described.append(f"{group} ({notes[0]})")
            else:
                described.append(group)

        if described:
            print(dim(f"      {' · '.join(described)}"))

        header = f"      {'KEY':<{token_width}}" + "".join(
            f"{component:>{column_widths[component]}}"
            for component in columns
        )
        print(dim(header))

        for token in sorted(table):
            row_text = f"      {token:<{token_width}}"

            for component in columns:
                source_key = table[token].get(component)

                if source_key is not None:
                    consumed.add(source_key)

                row_text += (
                    f"{cells[token, component]:>{column_widths[component]}}"
                )

            print(row_text)

        print()

    return frozenset(consumed)


# ---------------------------------------------------------------------------
# Nested per-entity Sensors / Actuators tables
# ---------------------------------------------------------------------------
#
# These tables render inside each entity's own block, immediately after
# its family tables: _print_family_tables consumes the
# sensordata_*/ctrl_*/actuator_force_* keys without printing, and
# _print_property_table then renders them here from the entity's own
# properties. Sensors/actuators the adapter could not attribute to a
# matched body live on synthetic "sensors"/"actuators" entities, which
# render as ordinary blocks of their own.

def _sensor_actuator_rows(
    ents: Mapping[int, EntityState],
    metadata: Mapping[int, Mapping[str, Any]],
    families: FrozenSet[str],
) -> Dict[str, Dict[str, Any]]:
    """
    One row per name token across every entity in ``ents`` whose property
    keys split (via ``_family_key_split``) into one of ``families``.
    Callers pass a single-entity mapping to build one entity's nested
    table; token uniqueness is still scene-wide (the adapter's own
    collision guards enforce it), so a token never legitimately spans two
    owners.

    Returns ``{token: {"groups": [str, ...], "notes": [str, ...],
    "components": {component: (key, value)}}}``.

    Groups accumulate like notes rather than keeping only the first. One
    actuator's four keys span four groups once the actuator families
    split by unit (``actuator torque command``, ``actuator output torque``,
    ``actuator angle``, ``actuator angular velocity``), and reporting one
    of them would label the whole row wrongly.
    """
    rows: Dict[str, Dict[str, Any]] = {}

    for entity_id in sorted(ents):
        entity = ents[entity_id]
        entity_meta = metadata.get(entity_id) or {}

        for key, value in entity.properties.items():
            split = _family_key_split(key)

            if split is None:
                continue

            family, remainder = split

            if family not in families:
                continue

            token, component = _family_token_component(family, remainder)

            row = rows.setdefault(
                token,
                {
                    "groups": [],
                    "notes": [],
                    "components": {},
                },
            )
            row["components"][component] = (key, value)

            meta = entity_meta.get(key)

            if meta is None:
                continue

            if meta.group and meta.group not in row["groups"]:
                row["groups"].append(meta.group)

            if meta.note and meta.note not in row["notes"]:
                row["notes"].append(meta.note)

    return rows


def _print_sensor_actuator_table(
    title: str,
    rows: Mapping[str, Mapping[str, Any]],
    family_hint: str,
) -> None:
    """Print one nested "Sensors" or "Actuators" table; no-op if empty."""
    if not rows:
        return

    key_table = {
        token: {
            component: key
            for component, (key, _value) in row["components"].items()
        }
        for token, row in rows.items()
    }

    columns = _family_columns(family_hint, key_table)

    token_width = max(
        [len(token) for token in rows]
        + [len("NAME")]
    ) + 2

    unit_width = _label_width(
        [
            ", ".join(row["notes"]) or "—"
            for row in rows.values()
        ],
        minimum=len("UNIT") + 2,
    )

    cells = {
        (token, component): (
            "—"
            if component not in row["components"]
            else f"{row['components'][component][1]:+.4g}"
        )
        for token, row in rows.items()
        for component in columns
    }

    column_widths = {
        component: max(
            [len(component)]
            + [
                len(cells[token, component])
                for token in rows
            ]
        ) + 2
        for component in columns
    }

    print(bold(f"      {title}"))

    header = (
        f"      {'NAME':<{token_width}}"
        + "".join(
            f"{component:>{column_widths[component]}}"
            for component in columns
        )
        + f"  {'UNIT':<{unit_width}}GROUP"
    )
    print(dim(header))

    for token in sorted(rows):
        row = rows[token]

        row_text = f"      {token:<{token_width}}"

        for component in columns:
            row_text += (
                f"{cells[token, component]:>{column_widths[component]}}"
            )

        unit = ", ".join(row["notes"]) or "—"
        group = ", ".join(row["groups"]) or "—"

        row_text += (
            f"  {_fit_label(unit, unit_width)}"
            f"{dim(group)}"
        )

        print(row_text)

    print()


def _print_property_table(
    entities: Mapping[int, EntityState],
    *,
    labels: Optional[Mapping[int, str]],
    verbose: bool,
    metadata: Optional[
        Mapping[int, Mapping[str, Any]]
    ] = None,
) -> None:
    entities_with_properties = {
        entity_id: entity
        for entity_id, entity in entities.items()
        if entity.properties
    }

    if not entities_with_properties:
        print(
            dim(
                "  No dynamic scalar properties were "
                "discovered in the initial observation."
            )
        )
        print()
        return

    print(bold("  Dynamic scalar properties"))
    print(
        dim(
            "  Every property is a single scalar (vector quantities "
            "appear as indexed components); values are the current "
            "state at discovery time."
        )
    )
    print(
        dim(
            "  Properties are extracted by the adapter; availability "
            "may vary by entity and timestep."
        )
    )
    print(dim("  " + "─" * 106))

    # Region property keys are structural and usually identical across a
    # scene's regions; their values live in the region table. Outside
    # --verbose, summarize each shared key set once instead of repeating
    # it per region (196 identical blocks on a RoboCasa scene).
    region_key_groups: Dict[
        Tuple[str, ...], List[int]
    ] = {}

    if not verbose:
        for entity_id in sorted(
            entities_with_properties
        ):
            entity = entities_with_properties[
                entity_id
            ]

            if _is_region_like(entity):
                keys = tuple(
                    sorted(entity.properties)
                )

                region_key_groups.setdefault(
                    keys,
                    [],
                ).append(entity_id)

    # A lone or deviating region keeps its full block so it stays
    # identifiable; only actually-shared key sets are summarized.
    summarized_regions = {
        entity_id
        for ids in region_key_groups.values()
        if len(ids) > 1
        for entity_id in ids
    }

    for entity_id in sorted(
        entities_with_properties,
        key=lambda eid: _display_sort_key(
            eid,
            entities_with_properties[eid],
        ),
    ):
        if entity_id in summarized_regions:
            continue

        entity = entities_with_properties[
            entity_id
        ]

        label = (
            labels.get(entity_id, "")
            if labels is not None
            else ""
        )

        identity = (
            f"  [{entity_id}] "
            f"{entity.entity_class}"
        )

        if label:
            identity += f" — {label}"

        print(bold(identity))

        entity_meta = (
            metadata.get(entity_id)
            if metadata
            else None
        )

        # The matrix and family tables are a display-only reshaping of
        # the same grouped listing below. They never activate in
        # --verbose, which keeps the flat per-key rendering. The matrix
        # needs at least one scalar-keyed joint (_matrix_rows' own
        # eligibility gate); the family-table pass runs for every
        # entity. Anything the tables do not consume falls through to
        # the grouped listing (or the metadata-free preview) unchanged.
        if not verbose:
            matrix_rows = (
                _matrix_rows(entity, entity_meta)
                if entity_meta
                else []
            )

            consumed: FrozenSet[str] = frozenset()

            if matrix_rows:
                consumed = _print_matrix_table(
                    matrix_rows,
                    entity_meta,
                )

            consumed = consumed | _print_family_tables(
                entity,
                consumed,
                entity_meta,
            )

            for section_title, section_families, section_hint in (
                ("Sensors", _SENSOR_FAMILIES, "sensordata"),
                ("Actuators", _ACTUATOR_FAMILIES, "ctrl"),
            ):
                _print_sensor_actuator_table(
                    section_title,
                    _sensor_actuator_rows(
                        {entity_id: entity},
                        (
                            {entity_id: entity_meta}
                            if entity_meta
                            else {}
                        ),
                        section_families,
                    ),
                    section_hint,
                )

            leftover_keys = (
                set(entity.properties) - consumed
            )

            if leftover_keys:
                _print_grouped_block(
                    entity,
                    _group_properties(
                        entity,
                        entity_meta or None,
                        leftover_keys,
                    ),
                    verbose=False,
                    metadata=entity_meta or None,
                )
        elif entity_meta:
            _print_grouped_block(
                entity,
                _group_properties(
                    entity,
                    entity_meta,
                ),
                verbose=True,
                metadata=entity_meta,
            )
        else:
            _print_grouped_block(
                entity,
                _group_properties(
                    entity,
                    None,
                ),
                verbose=True,
            )

    for keys, ids in sorted(
        region_key_groups.items(),
        key=lambda item: -len(item[1]),
    ):
        if len(ids) < 2:
            continue

        print(
            bold(
                f"  {len(ids)} region-like entities "
                f"share one property set"
            )
        )

        _print_wrapped(
            "      ",
            ", ".join(keys),
            width=112,
        )

        print(
            dim(
                "      Centers and half extents are in the region "
                "table below; rerun with --verbose for per-region "
                "values."
            )
        )

    print(dim("  " + "─" * 106))
    print()


def _print_region_table(
    regions: Mapping[int, EntityState],
    *,
    labels: Optional[Mapping[int, str]],
) -> None:
    if not regions:
        return

    label_values = [
        (
            labels.get(
                entity_id,
                f"{regions[entity_id].entity_class}_{entity_id}",
            )
            if labels is not None
            else f"{regions[entity_id].entity_class}_{entity_id}"
        )
        for entity_id in regions
    ]

    label_width = _label_width(
        label_values,
        minimum=len("LABEL") + 2,
    )

    rule_width = 8 + 14 + label_width + 25 + 20

    print(bold("  Regions"))
    print(dim("  " + "─" * rule_width))
    print(
        dim(
            f"  {'ID':>5}   "
            f"{'CLASS':<14}"
            f"{'LABEL':<{label_width}}"
            f"{'CENTER':<25}"
            f"{'HALF EXTENTS'}"
        )
    )
    print(dim("  " + "─" * rule_width))

    for entity_id in sorted(regions):
        entity = regions[entity_id]

        label = (
            labels.get(
                entity_id,
                f"{entity.entity_class}_{entity_id}",
            )
            if labels is not None
            else f"{entity.entity_class}_{entity_id}"
        )

        center = (
            "—"
            if entity.position is None
            else _format_vector(
                entity.position,
                precision=3,
            )
        )

        extents = []

        for name in (
            "hw",
            "hh",
            "hd",
        ):
            value = entity.properties.get(
                name
            )

            if value is not None:
                extents.append(
                    f"{name}={float(value):.3f}"
                )

        extent_text = (
            ", ".join(extents)
            if extents
            else "not exposed"
        )

        is_target = label.startswith(
            "target_"
        )

        display_label = (
            green(_fit_label(label, label_width))
            if is_target
            else dim(_fit_label(label, label_width))
        )

        target_flag = (
            f"  {green('← TARGET')}"
            if is_target
            else ""
        )

        print(
            f"  {entity_id:>5}   "
            f"{entity.entity_class:<14}"
            f"{display_label}"
            f"{dim(f'{center:<25}')}"
            f"{dim(extent_text)}"
            f"{target_flag}"
        )

    print(dim("  " + "─" * rule_width))
    print()


def _print_class_summary(
    entities: Mapping[int, EntityState],
) -> None:
    by_class: Dict[str, List[EntityState]] = {}

    for entity in entities.values():
        by_class.setdefault(
            entity.entity_class,
            [],
        ).append(entity)

    print(bold("  Discovery summary"))

    if not by_class:
        print(
            amber(
                "  No monitorable entities were "
                "present in the initial observation."
            )
        )
        print()
        return

    for entity_class in sorted(by_class):
        class_entities = by_class[
            entity_class
        ]

        property_union = set()
        property_intersection: Optional[
            set[str]
        ] = None

        for entity in class_entities:
            keys = set(
                entity.properties.keys()
            )

            property_union.update(keys)

            if property_intersection is None:
                property_intersection = keys
            else:
                property_intersection.intersection_update(
                    keys
                )

        property_intersection = (
            property_intersection
            if property_intersection is not None
            else set()
        )

        print(
            f"  {bold(entity_class):<24} "
            f"{len(class_entities)} active "
            f"entity/entities"
        )

        if property_union:
            common_count = len(
                property_intersection
            )

            print(
                dim(
                    f"      {len(property_union)} distinct "
                    f"property key(s); "
                    f"{common_count} currently common to all "
                    f"active {entity_class} entities"
                )
            )

            # A count of 0 reads as a neutral statistic but predicts
            # binding: a wildcard slot binds only the entities exposing
            # every key it names, so with no shared key one class_*
            # reference cannot reach the whole class. Adapters that
            # qualify keys by body name land here by construction.
            if (
                common_count == 0
                and len(class_entities) > 1
            ):
                print(
                    amber(
                        f"      no shared key: no exact key covers "
                        f"this class, so {entity_class}_* naming one "
                        f"binds a single entity; a component glob "
                        f"covers all {len(class_entities)}"
                    )
                )

    print()


def _find_class(
    classes: Iterable[str],
    candidates: Sequence[str],
) -> Optional[str]:
    class_set = set(classes)

    for candidate in candidates:
        if candidate in class_set:
            return candidate

    return None


def _class_has_property(
    entities: Mapping[int, EntityState],
    entity_class: str,
    property_name: str,
) -> bool:
    return any(
        entity.entity_class == entity_class
        and property_name in entity.properties
        for entity in entities.values()
    )


def _class_has_z(
    entities: Mapping[int, EntityState],
    entity_class: Optional[str],
) -> bool:
    """Whether any entity of a class carries a 3-component position."""
    return any(
        entity.position is not None
        and len(entity.position) >= 3
        for entity in entities.values()
        if entity.entity_class == entity_class
    )


def _joint_velocity_key(
    keys: Sequence[str],
    metadata: Mapping[str, Any],
) -> Optional[str]:
    """First key whose metadata group is a joint-velocity group, if any."""
    return next(
        (
            key
            for key in keys
            if key in metadata
            and metadata[key].group.startswith("joint")
            and "velocit" in metadata[key].group
        ),
        None,
    )


_POSITION_GROUPS = frozenset(
    {
        "body position",
        "site position",
    }
)


def _position_keys(
    entities: Mapping[int, EntityState],
    entity_class: Optional[str],
    metadata: Optional[
        Mapping[int, Mapping[str, Any]]
    ],
) -> List[str]:
    """
    Position component keys for one entity class, in x/y/z order.

    Reads the first class member whose metadata tags a key with the
    "body position" or gripper "site position" group, then sorts that
    entity's tagged keys so the ``_0``/``_1``/``_2`` suffix order is
    stable (their shared token prefix makes plain alphabetical sort
    equivalent to axis order). Keys are never mixed across entities:
    different bodies/sites do not share axis-for-axis key names, so an
    example formula can only draw all of x/y/z from a single entity.

    Returns an empty list when no entity of the class carries position
    metadata, so callers can fall back to a generic example instead of
    printing a formula with keys that do not resolve.
    """
    if entity_class is None or not metadata:
        return []

    for entity_id, entity in entities.items():
        if entity.entity_class != entity_class:
            continue

        entity_meta = metadata.get(entity_id)

        if not entity_meta:
            continue

        keys = sorted(
            key
            for key, meta in entity_meta.items()
            if meta.group in _POSITION_GROUPS
            and key in entity.properties
        )

        if keys:
            return keys

    return []


def _print_example(
    entities: Mapping[int, EntityState],
    *,
    metadata: Optional[
        Mapping[int, Mapping[str, Any]]
    ] = None,
) -> None:
    if not entities:
        return

    classes = {
        entity.entity_class
        for entity in entities.values()
    }

    robot_class = _find_class(
        classes,
        (
            "robot",
            "agent",
        ),
    )

    object_class = _find_class(
        classes,
        (
            "object",
            "obj",
        ),
    )

    region_class = next(
        (
            entity.entity_class
            for entity in entities.values()
            if _is_region_like(entity)
        ),
        None,
    )

    gripper_class = _find_class(
        classes,
        (
            "gripper",
            "end_effector",
            "endeffector",
        ),
    )

    formula: Optional[str] = None
    note: Optional[str] = None

    region_containment_ok = (
        object_class is not None
        and region_class is not None
        and any(
            _REGION_HALF_EXTENTS.issubset(
                entity.properties.keys()
            )
            for entity in entities.values()
            if entity.entity_class
            == region_class
        )
    )

    region_object_keys = (
        _position_keys(
            entities,
            object_class,
            metadata,
        )
        if region_containment_ok
        else []
    )

    robot_distance_ok = (
        robot_class is not None
        and object_class is not None
    )

    robot_position_keys = (
        _position_keys(
            entities,
            robot_class,
            metadata,
        )
        if robot_distance_ok
        else []
    )

    distance_object_keys = (
        _position_keys(
            entities,
            object_class,
            metadata,
        )
        if robot_distance_ok
        else []
    )

    if (
        region_containment_ok
        and len(region_object_keys) >= 2
    ):
        obj_x, obj_y = (
            region_object_keys[0],
            region_object_keys[1],
        )

        obj_z = (
            region_object_keys[2]
            if len(region_object_keys) >= 3
            else None
        )

        z_terms = ""

        if (
            obj_z is not None
            and _class_has_z(entities, object_class)
            and _class_has_z(entities, region_class)
        ):
            z_terms = (
                ", "
                f"{object_class}_*[{obj_z}] - "
                f"{region_class}_*[z0], "
                f"{region_class}_*[z1] - "
                f"{object_class}_*[{obj_z}]"
            )

        formula = (
            f"once[0,30s]("
            f"Min("
            f"{object_class}_*[{obj_x}] - "
            f"{region_class}_*[x0], "
            f"{region_class}_*[x1] - "
            f"{object_class}_*[{obj_x}], "
            f"{object_class}_*[{obj_y}] - "
            f"{region_class}_*[y0], "
            f"{region_class}_*[y1] - "
            f"{object_class}_*[{obj_y}]"
            f"{z_terms}"
            f") >= 0"
            f")"
        )

        note = (
            f"every {object_class} binding is evaluated "
            f"against every {region_class} binding"
        )

    elif (
        robot_distance_ok
        and len(robot_position_keys) >= 2
        and len(distance_object_keys) >= 2
    ):
        robot_x, robot_y = (
            robot_position_keys[0],
            robot_position_keys[1],
        )

        robot_z = (
            robot_position_keys[2]
            if len(robot_position_keys) >= 3
            else None
        )

        object_x, object_y = (
            distance_object_keys[0],
            distance_object_keys[1],
        )

        object_z = (
            distance_object_keys[2]
            if len(distance_object_keys) >= 3
            else None
        )

        distance_terms = (
            f"({robot_class}_*[{robot_x}] - "
            f"{object_class}_*[{object_x}])^2 + "
            f"({robot_class}_*[{robot_y}] - "
            f"{object_class}_*[{object_y}])^2"
        )

        if (
            robot_z is not None
            and object_z is not None
            and _class_has_z(entities, robot_class)
            and _class_has_z(entities, object_class)
        ):
            distance_terms += (
                f" + "
                f"({robot_class}_*[{robot_z}] - "
                f"{object_class}_*[{object_z}])^2"
            )

        formula = (
            f"historically[0,5s]("
            f"sqrt({distance_terms}) - 0.30 >= 0"
            f")"
        )

        note = (
            f"{robot_class} remains at least 30 cm "
            f"from each {object_class}"
        )

    elif (
        gripper_class is not None
        and _class_has_property(
            entities,
            gripper_class,
            "recent_ee_force_norm",
        )
    ):
        formula = (
            f"historically[0,2s]("
            f"{gripper_class}_*"
            f"[recent_ee_force_norm] <= 25.0"
            f")"
        )

        note = (
            "end-effector force remains below "
            "the selected threshold"
        )

    else:
        # No built-in spatial alias exists in the core: fall back to an
        # extracted key from the scene itself. A blind alphabetical pick
        # would land on qfrc_* for MuJoCo robots — an applied-force input
        # that is trivially zero — so a key whose catalog metadata marks
        # it a joint velocity is preferred when the adapter provides one.
        fallback_class = (
            robot_class
            or sorted(classes)[0]
        )

        fallback_keys = sorted(
            {
                key
                for entity in entities.values()
                if entity.entity_class == fallback_class
                for key in entity.properties
            }
        )

        if not fallback_keys:
            return

        fallback_metadata: Dict[str, Any] = {}

        if metadata:
            for entity_id, entity in entities.items():
                if entity.entity_class != fallback_class:
                    continue

                entity_meta = metadata.get(entity_id)

                if entity_meta:
                    fallback_metadata.update(entity_meta)

        fallback_key = (
            _joint_velocity_key(
                fallback_keys,
                fallback_metadata,
            )
            or fallback_keys[0]
        )

        formula = (
            f"historically[0,5s]("
            f"{fallback_class}_*[{fallback_key}] <= 2.0"
            f")"
        )

        note = (
            f"cap on an extracted property ({fallback_key}); combine "
            "with the scene properties listed above"
        )

    print(
        bold("  Example formula")
        + dim(f"  ({note}):")
    )
    print(f"  {cyan(formula)}")
    print()


def print_discovery(
    adapter: SimulatorAdapter,
    ents: Dict[int, EntityState],
    verbose: bool = False,
    field_contract: Optional[bool] = None,
) -> None:
    """
    Present the initial entity and property discovery results.

    This is authoring-time introspection. It describes the current observation,
    not a promise that the entity population or property vocabulary will remain
    fixed during the episode.
    """
    print()
    print(bold("  Inspecting monitor-facing state …"))
    print()

    labels = getattr(
        adapter,
        "entity_labels",
        None,
    )

    _print_class_summary(
        ents
    )

    regions = {
        entity_id: entity
        for entity_id, entity in ents.items()
        if _is_region_like(entity)
    }

    non_regions = {
        entity_id: entity
        for entity_id, entity in ents.items()
        if entity_id not in regions
    }

    _print_entity_table(
        non_regions,
        labels=labels,
    )

    warning_fn = getattr(adapter, "discovery_warnings", None)
    if callable(warning_fn):
        warnings = tuple(warning_fn(ents))
        if warnings:
            print(bold("  Authoring warnings"))
            print()
            for warning in warnings:
                if not isinstance(warning, str) or not warning.strip():
                    raise TypeError(
                        "discovery_warnings() must return non-empty strings"
                    )
                _print_wrapped(
                    "  {}  ".format(amber("⚠")),
                    warning,
                )
            print()

    meta_fn = getattr(
        adapter,
        "property_metadata",
        None,
    )

    metadata: Dict[int, Mapping[str, Any]] = {}

    if callable(meta_fn):
        for entity_id in ents:
            try:
                entity_meta = meta_fn(entity_id) or {}
            except Exception:
                entity_meta = {}

            if entity_meta:
                metadata[entity_id] = entity_meta

    _print_property_table(
        ents,
        labels=labels,
        verbose=verbose,
        metadata=metadata,
    )

    _print_declared_dynamic_classes(
        adapter,
        ents,
    )

    _print_region_table(
        regions,
        labels=labels,
    )

    if field_contract is not None:
        _print_mujoco_field_contract(
            adapter,
            full=field_contract,
        )

    _print_example(
        ents,
        metadata=metadata,
    )


def _print_declared_dynamic_classes(
    adapter: SimulatorAdapter,
    ents: Dict[int, EntityState],
) -> None:
    """List capability-declared classes with no live instance at reset.

    Dynamic populations (contact pairs) may be empty in the reset
    snapshot while their class and component schema are authorable now;
    the declaration, not the snapshot, is the authority (section
    5.6.1). Without this section a reader of ``discover`` output would
    conclude the vocabulary does not exist.
    """
    capabilities = getattr(adapter, "capabilities", None)
    if capabilities is None:
        return
    declared = getattr(capabilities, "components_by_class", None)
    if not declared:
        return

    live_classes = {entity.entity_class for entity in ents.values()}
    absent = {
        entity_class: components
        for entity_class, components in declared.items()
        if entity_class not in live_classes
    }
    if not absent:
        return

    print(bold("  Declared dynamic classes (no live instance at reset)"))
    print()
    for entity_class in sorted(absent):
        components = ", ".join(sorted(absent[entity_class]))
        _print_wrapped(
            "    {}  ".format(cyan(entity_class)),
            "components: {}. Instances appear at runtime; a spec "
            "referencing this class compiles and waits.".format(
                components
            ),
        )
    print()


def _print_mujoco_field_contract(
    adapter: SimulatorAdapter,
    *,
    full: bool,
) -> None:
    """Print the live MuJoCo classification/materialisation boundary.

    This is deliberately duck-typed so the four-method adapter interface
    remains unchanged. Non-MuJoCo adapters simply have no field-contract
    section.
    """
    surface = getattr(adapter, "field_surface_report", None)
    statuses = getattr(adapter, "field_materialization", None)
    if surface is None or not isinstance(statuses, Mapping):
        return

    from .adapters.mujoco_contract import EXCLUDED_FIELDS, INCLUDED_FIELDS

    status_values = list(statuses.values())
    binding_present = sum(
        bool(getattr(status, "binding_present", False))
        for status in status_values
    )
    scene_applicable = sum(
        bool(getattr(status, "scene_applicable", False))
        for status in status_values
    )
    materialized = sum(
        bool(getattr(status, "emitted_keys", ()))
        for status in status_values
    )
    emitted_sinks = sum(
        len(tuple(getattr(status, "emitted_keys", ())))
        for status in status_values
    )
    applicable_without_output = sorted(
        str(getattr(status, "field_name", "unknown"))
        for status in status_values
        if bool(getattr(status, "scene_applicable", False))
        and not tuple(getattr(status, "emitted_keys", ()))
    )

    print()
    print(bold("  MuJoCo field contract"))
    print(
        "  {} global readers; {} present on this binding; "
        "{} scene-applicable; {} materialized into {} concrete sinks".format(
            len(INCLUDED_FIELDS),
            binding_present,
            scene_applicable,
            materialized,
            emitted_sinks,
        )
    )
    print(
        "  {} global wrapper-inclusive exclusions; {} exclusions present "
        "on this binding; {} unclassified".format(
            len(EXCLUDED_FIELDS),
            len(tuple(getattr(surface, "excluded", ()))),
            len(tuple(getattr(surface, "unclassified", ()))),
        )
    )
    if applicable_without_output:
        print(
            red(
                "  ERROR: applicable reader(s) without emitted sinks: "
                + ", ".join(applicable_without_output)
            )
        )
    else:
        print(green("  Every scene-applicable reader is eagerly materialized."))

    if not full:
        print(
            dim(
                "  Use --field-contract for reader provenance, capability "
                "reasons, exact emitted sinks, and exclusions."
            )
        )
        return

    print()
    print(bold("  Curated readers"))
    for field_name, record in sorted(INCLUDED_FIELDS.items()):
        status = statuses.get(field_name)
        present = bool(
            status is not None
            and getattr(status, "binding_present", False)
        )
        applicable = bool(
            status is not None
            and getattr(status, "scene_applicable", False)
        )
        emitted = tuple(
            getattr(status, "emitted_keys", ())
            if status is not None
            else ()
        )
        reason = (
            getattr(status, "reason", None)
            if status is not None
            else "adapter did not report this curated reader"
        )
        print()
        print(
            "  {}  binding={}  scene={}  emitted={}".format(
                bold(field_name),
                "yes" if present else "no",
                "yes" if applicable else "no",
                len(emitted),
            )
        )
        _print_wrapped("    owner:      ", record.owner_resolver_id)
        _print_wrapped("    address:    ", record.address_resolver_id)
        _print_wrapped("    unit/frame: ", "{}; {}".format(
            record.unit_resolver_id,
            record.frame,
        ))
        _print_wrapped(
            "    population: ",
            record.population_prerequisite,
        )
        if reason:
            _print_wrapped("    capability: ", str(reason))
        refusal = (
            getattr(status, "plan_refusal", None)
            if status is not None
            else None
        )
        if refusal:
            # Its own line, never folded into capability: this says the
            # field is on the scalar path, not that it extracts less.
            _print_wrapped("    plan:       ", str(refusal))
        if emitted:
            _print_wrapped("    sinks:      ", ", ".join(emitted))

    print()
    print(bold("  Declared exclusions"))
    for field_name, record in sorted(EXCLUDED_FIELDS.items()):
        _print_wrapped(
            "  {} [{}]: ".format(field_name, record.reason_code),
            record.concrete_detail,
            subsequent_indent="      ",
        )


# ---------------------------------------------------------------------------
# (moved to adapters/fake.py; re-exported for compatibility)
# ---------------------------------------------------------------------------

from .adapters.fake import _MockAdapter  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

def _load_sim(
    name: str,
    task: Optional[str] = None,
    robot: Optional[str] = None,
    target_only: bool = False,
    capture_frames: bool = True,
    seed: Optional[int] = None,
) -> SimulatorAdapter:
    """
    Construct one discovered simulator adapter.

    Discovery is entirely the registry's responsibility (direct
    registration, entry points, or an explicitly loaded adapter module);
    this module holds no adapter-name table.

    ``capture_frames=False`` reaches only a factory that declares the
    ``capture_frames`` parameter (see ``registry``), and a ``seed`` only a
    factory that declares ``seed``; others are called as before.

    Raises explicit exceptions rather than silently suppressing adapter
    import failures, which makes refactor and dependency errors visible.
    """
    import inspect

    from .registry import resolve

    entry = resolve(name)
    options = {}
    parameters = inspect.signature(entry.factory).parameters
    if not capture_frames and "capture_frames" in parameters:
        options["capture_frames"] = False
    if seed is not None and "seed" in parameters:
        options["seed"] = seed

    try:
        adapter = entry.factory(
            task,
            robot,
            target_only,
            **options,
        )
    except ImportError as exc:
        installation = (
            f" Install with: {entry.install_hint}"
            if entry.install_hint
            else ""
        )

        raise ImportError(
            f"Could not load simulator adapter "
            f"{name!r}: {exc}.{installation}"
        ) from exc

    if not isinstance(
        adapter,
        SimulatorAdapter,
    ):
        raise TypeError(
            f"Adapter factory {name!r} returned "
            f"{type(adapter).__name__}, expected "
            f"SimulatorAdapter"
        )

    return adapter


def _validate_dt_override(
    dt_override: Optional[float],
) -> Optional[float]:
    if dt_override is None:
        return None

    if (
        isinstance(dt_override, bool)
        or not isinstance(dt_override, Real)
    ):
        raise TypeError(
            "dt_override must be a real scalar or None"
        )

    value = float(
        dt_override
    )

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"dt_override must be finite and positive, "
            f"got {dt_override!r}"
        )

    return value


def _task_description(
    adapter: SimulatorAdapter,
) -> Optional[str]:
    """
    Read optional natural-language task metadata without requiring one backend.
    """
    sources = (
        adapter,
        getattr(
            adapter,
            "env",
            None,
        ),
    )

    for source in sources:
        if source is None:
            continue

        get_ep_meta = getattr(
            source,
            "get_ep_meta",
            None,
        )

        if callable(get_ep_meta):
            try:
                metadata = get_ep_meta()

                if isinstance(
                    metadata,
                    Mapping,
                ):
                    language = metadata.get(
                        "lang"
                    )

                    if language:
                        return str(language)

            except Exception:
                pass

        language_instruction = getattr(
            source,
            "language_instruction",
            None,
        )

        if language_instruction:
            return str(
                language_instruction
            )

    return None


def construct_sim(
    sim: str,
    task: Optional[str] = None,
    robot: Optional[str] = None,
    target_only: bool = False,
    header: str = "Specola",
    queries: Optional[QueryConfiguration] = None,
    capture_frames: bool = True,
    seed: Optional[int] = None,
) -> SimulatorAdapter:
    """Print the startup header, then construct one adapter without resetting it.

    ``queries`` is installed before any reset. ``specola run`` resets once,
    in ``Monitor.watch``, and reports readiness with ``print_ready``.
    ``capture_frames=False`` tells a factory that accepts it that no camera
    frame will be captured and no viewer opened. ``seed`` reaches a factory
    that draws its scene at construction (RoboCasa's kitchen, Meta-World's
    task variant).
    """
    print()
    from . import branding

    if branding.enabled():
        # The mark carries the name, so "Specola Evaluation" becomes
        # "─S─ Specola  Evaluation" rather than repeating it.
        if header == "Specola":
            label = ""
        elif header.startswith("Specola "):
            label = header[len("Specola "):]
        else:
            label = header
        print("  ", end="")
        branding.print_branded(branding.mark_text(label))
    else:
        print(bold(f"  {header}"))
    print(
        dim(
            "  ─────────────────────────────────────────────────"
        )
    )
    print()

    print(
        dim(
            f"  Loading simulator adapter: {sim} …"
        )
    )

    # Only a caller that captures no frame or gives a seed passes the
    # option at all, so every other caller keeps the exact three-argument
    # construction call.
    adapter = _load_sim(
        sim,
        task=task,
        robot=robot,
        target_only=target_only,
        **({} if capture_frames else {"capture_frames": False}),
        **({} if seed is None else {"seed": seed}),
    )

    if queries is not None:
        adapter.set_query_configuration(queries)
    return adapter


def print_ready(
    adapter: SimulatorAdapter,
    effective_dt: float,
    entity_count: int,
) -> None:
    """Print the readiness line for a completed reset, then the task line."""
    print(
        f"  {green('✓')} ready   "
        + dim(
            f"adapter_dt={adapter.dt}s   "
            f"spec_dt={effective_dt}s   "
            f"spatial_dims={int(adapter.spatial_dims)}   "
            f"entities={entity_count}"
        )
    )

    task_description = _task_description(
        adapter
    )

    if task_description:
        print()
        print(
            f"  {bold('Task:')} "
            f"{cyan(task_description)}"
        )


def load_sim(
    sim: str,
    task: Optional[str] = None,
    robot: Optional[str] = None,
    target_only: bool = False,
    dt_override: Optional[float] = None,
    header: str = "Specola",
    queries: Optional[QueryConfiguration] = None,
    allow_schema_only: bool = False,
    catalog_only: bool = False,
) -> Tuple[
    SimulatorAdapter,
    float,
    int,
    Dict[int, EntityState],
    List[str],
]:
    """
    Construct and reset one adapter, installing query arguments before reset.

    Discovery can opt into a guarded schema-only response before reset. When
    native query arguments are missing, SchemaOnlyDiscovery carries that
    response and no observation is read. Other callers retain the tuple result.

    ``catalog_only`` resets under an empty entity scope, so an adapter that
    reads a class entity by entity (MuJoCo geom pairs) reads none of them
    (issue #136); ``entities`` then omits them and
    ``capabilities.reset_catalog`` recovers them for a catalog.

    Returns
    -------
    adapter:
        Validated simulator adapter.

    dt:
        Effective specification timestep. This is ``dt_override`` when one was
        explicitly supplied; otherwise it is ``adapter.dt``.

    dims:
        Adapter spatial dimensionality.

    entities:
        Entities in the initial validated observation.

    entity_classes:
        String classes present in the initial observation.

    Notes
    -----
    ``entity_classes`` is discovery information, not a frozen episode
    vocabulary.
    """
    adapter = construct_sim(
        sim,
        task=task,
        robot=robot,
        target_only=target_only,
        header=header,
        queries=queries,
    )

    override = _validate_dt_override(
        dt_override
    )

    if allow_schema_only:
        from .conformance.authoring import SchemaOnlyDiscovery, prepare_schema_authoring
        document = prepare_schema_authoring(adapter)
        if document is not None:
            document["subject"].update(requested_adapter=sim, task=task,
                requested_factory_arguments={"robot": robot, "target_only": target_only,
                                             "dt_override": dt_override})
            raise SchemaOnlyDiscovery(document)
    if catalog_only:
        adapter.set_extraction_entity_scope({})
    observation = adapter.reset()

    effective_dt = (
        override
        if override is not None
        else float(adapter.dt)
    )

    dimensions = int(
        adapter.spatial_dims
    )

    entities = dict(
        observation.perceived_entities
    )

    entity_classes = sorted(
        {
            entity.entity_class
            for entity in entities.values()
        }
    )

    print_ready(adapter, effective_dt, len(entities))

    return (
        adapter,
        effective_dt,
        dimensions,
        entities,
        entity_classes,
    )
