"""
dataset.py — sim-free offline dataset adapter
=============================================
Monitor recorded trajectories (e.g. data released with a paper) with no
simulator at all: each row of a CSV file becomes part of one
MonitorObservation, and the engine neither knows nor cares that the
observations come from disk.

CSV format (long / tidy)
------------------------
Required columns:

    time          seconds (uniformly sampled; dt is inferred)
    entity_id     integer, stable per entity
    entity_class  string (``robot``, ``object``, ``vehicle``, ...)
    px, py        position; add ``pz`` for 3D data

Optional columns:

    vx, vy[, vz]  velocities; when absent they are finite-differenced
                  from positions between consecutive frames

Every numeric column except ``time``, ``entity_id``, and ``entity_class``
becomes an entity property (empty cells mean "this entity lacks that
property at that step", which per-slot binding eligibility handles
naturally). Position and velocity columns are no exception: ``px``/``py``
(and ``pz`` for 3D data) additionally populate ``EntityState.position``,
and ``vx``/``vy``/``vz`` additionally populate ``EntityState.velocity``.
When velocity columns are absent, the finite-differenced velocity is
still exposed as ``vx``/``vy``[/``vz``] properties, so the property
vocabulary stays stable regardless of whether the source data included
explicit velocities.

Frame times are parsed as decimal values. Spacing is uniform when every
interval agrees with the first within a relative tolerance of ``1e-12``;
there is no absolute tolerance that could hide nonuniform small intervals.

Usage
-----
    specola discover    --sim dataset --task path/to/trajectories.csv
    specola run --sim dataset --task path/to/trajectories.csv \\
                    --specs my_specs.yaml

The run ends when the recording does (the adapter owns its drive
behavior via ``make_advancer``). ROS 1 / ROS 2 bag readers are planned
variants of this adapter — see docs/ROADMAP.md.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..adapter import MonitorObservation, SimulatorAdapter
from ..entities import EntityState, TrackState
from ..registry import register as _register


_STRUCTURAL_COLUMNS = {
    "time", "entity_id", "entity_class",
    "px", "py", "pz", "vx", "vy", "vz",
}
_FRAME_SPACING_REL_TOLERANCE = Decimal("1e-12")


class DatasetAdapter(SimulatorAdapter):
    """Replay a recorded multi-entity trajectory file as observations."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._frames, self._dt, self._dims = _load_csv(path)
        self._cursor = 0
        self.path = path

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def spatial_dims(self) -> int:
        return self._dims

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def frames_remaining(self) -> int:
        return len(self._frames) - self._cursor

    # ------------------------------------------------------------------
    def _reset_impl(self) -> MonitorObservation:
        self._cursor = 0
        return self._emit()

    def _get_observation_impl(self) -> MonitorObservation:
        if self._cursor >= len(self._frames) - 1:
            raise RuntimeError(
                f"Dataset {self.path!r} is exhausted "
                f"({len(self._frames)} frames)."
            )
        self._cursor += 1
        return self._emit()

    def _emit(self) -> MonitorObservation:
        time, entities = self._frames[self._cursor]
        return MonitorObservation(
            timestep=self._cursor,
            time=time,
            perceived_entities=dict(entities),
        )

    # ------------------------------------------------------------------
    def make_advancer(self, mode, demo, seed):
        """
        Drive behavior: one recorded frame per step, stop at the end.

        The ``--task`` CSV is the recording, so there are no actions to
        drive: ``random`` (the CLI default) and ``hold`` both play it
        unchanged. ``replay`` and ``--demo`` are refused rather than
        accepted and ignored (#132).
        """
        if demo is not None:
            raise SystemExit(
                "The dataset adapter has no --demo replay: the --task CSV "
                "is the recording, so --demo does not apply."
            )
        if mode not in ("random", "hold"):
            raise SystemExit(
                f"--mode {mode} is not supported by the dataset adapter; "
                "the --task CSV is the recording and plays under the "
                "default --mode."
            )
        del seed

        def advance(step: int) -> bool:
            return self.frames_remaining > 1

        return advance


def _load_csv(path: str):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset {path!r} has no header row")
        columns = [c.strip() for c in reader.fieldnames]
        empty_columns = [
            index + 1 for index, column in enumerate(columns)
            if not column
        ]
        if empty_columns:
            raise ValueError(
                f"Dataset {path!r} has empty header column(s) at "
                f"position(s) {empty_columns}"
            )
        duplicate_columns = sorted(
            column for column, count in Counter(columns).items()
            if count > 1
        )
        if duplicate_columns:
            raise ValueError(
                f"Dataset {path!r} has duplicate normalized header(s): "
                f"{duplicate_columns}"
            )
        missing = {"time", "entity_id", "entity_class", "px", "py"} - set(
            columns
        )
        if missing:
            raise ValueError(
                f"Dataset {path!r} is missing required columns "
                f"{sorted(missing)}"
            )
        rows = []
        original_columns = tuple(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Dataset {path!r} row {row_number} has extra cells"
                )
            normalized_row = {}
            for original, normalized in zip(original_columns, columns):
                value = row.get(original)
                if value is None:
                    raise ValueError(
                        f"Dataset {path!r} row {row_number} is missing "
                        f"column {normalized!r}"
                    )
                normalized_row[normalized] = value
            rows.append(normalized_row)

    if not rows:
        raise ValueError(f"Dataset {path!r} contains no rows")

    dims = 3 if "pz" in columns else 2
    velocity_columns = {column for column in columns if column in {"vx", "vy", "vz"}}
    required_velocity_columns = {"vx", "vy"} | (
        {"vz"} if dims == 3 else set()
    )
    if velocity_columns and velocity_columns != required_velocity_columns:
        missing_velocity = sorted(required_velocity_columns - velocity_columns)
        extra_velocity = sorted(velocity_columns - required_velocity_columns)
        details = []
        if missing_velocity:
            details.append(f"missing {missing_velocity}")
        if extra_velocity:
            details.append(f"unexpected {extra_velocity}")
        raise ValueError(
            f"Dataset {path!r} has incomplete velocity column group for "
            f"{dims}D data ({'; '.join(details)}); provide "
            f"{sorted(required_velocity_columns)} together or omit all "
            "velocity columns"
        )
    has_velocity = bool(velocity_columns)
    property_columns = [
        c for c in columns if c not in _STRUCTURAL_COLUMNS
    ]

    # Group rows into frames by time (preserving file order of times).
    frame_times: List[Decimal] = []
    frame_rows: Dict[Decimal, List[dict]] = {}
    seen_rows: Dict[Tuple[Decimal, int], int] = {}
    previous_row_time: Optional[Decimal] = None
    for row_number, row in enumerate(rows, start=2):
        time_text = row["time"].strip()
        try:
            t = Decimal(time_text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"Dataset {path!r} row {row_number} column 'time' "
                f"must be a finite decimal, got {time_text!r}"
            ) from exc
        if not t.is_finite():
            raise ValueError(
                f"Dataset {path!r} row {row_number} column 'time' "
                f"must be finite, got {time_text!r}"
            )
        entity_id_text = row["entity_id"].strip()
        try:
            entity_id = int(entity_id_text)
        except ValueError as exc:
            raise ValueError(
                f"Dataset {path!r} row {row_number} column 'entity_id' "
                f"must be an integer, got {entity_id_text!r}"
            ) from exc
        if (t, entity_id) in seen_rows:
            first_row = seen_rows[(t, entity_id)]
            raise ValueError(
                f"Dataset {path!r} has duplicate (time, entity_id) row "
                f"at row {row_number}; first appeared at row {first_row} "
                f"for ({t!r}, {entity_id})"
            )
        seen_rows[(t, entity_id)] = row_number
        if previous_row_time is not None and t < previous_row_time:
            raise ValueError(
                f"Dataset {path!r} has out-of-order frame times: row "
                f"{row_number} has time {t!r} after {previous_row_time!r}; "
                "rows must be ordered by nondecreasing frame time"
            )
        previous_row_time = t
        if t not in frame_rows:
            frame_rows[t] = []
            frame_times.append(t)
        frame_rows[t].append(row)

    float_frame_times = [float(time) for time in frame_times]
    if any(
        not math.isfinite(time)
        for time in float_frame_times
    ) or any(
        later <= earlier
        for earlier, later in zip(
            float_frame_times, float_frame_times[1:]
        )
    ):
        raise ValueError(
            f"Dataset {path!r} frame times are not finite, strictly "
            "increasing binary floats"
        )

    deltas = [
        b - a for a, b in zip(frame_times, frame_times[1:])
    ]
    if not deltas:
        raise ValueError(
            f"Dataset {path!r} needs at least two timesteps"
        )
    decimal_dt = deltas[0]
    dt = float(decimal_dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Dataset {path!r} has a non-positive or non-representable "
            f"frame spacing {decimal_dt!r}"
        )
    if any(
        abs(delta - decimal_dt)
        > max(abs(delta), abs(decimal_dt))
        * _FRAME_SPACING_REL_TOLERANCE
        for delta in deltas
    ):
        raise ValueError(
            f"Dataset {path!r} has nonuniform frame times: "
            f"expected spacing {decimal_dt!r}, observed {deltas!r}"
        )
    frames = []
    previous_positions: Dict[int, np.ndarray] = {}

    for t, float_time in zip(frame_times, float_frame_times):
        entities: Dict[int, EntityState] = {}
        for row in frame_rows[t]:
            entity_id = int(row["entity_id"])
            position = np.array(
                [float(row["px"]), float(row["py"])]
                + ([float(row["pz"])] if dims == 3 else []),
                dtype=float,
            )

            if has_velocity:
                velocity = np.array(
                    [float(row["vx"]), float(row["vy"])]
                    + ([float(row["vz"])] if dims == 3 else []),
                    dtype=float,
                )
            else:
                prev = previous_positions.get(entity_id)
                velocity = (
                    (position - prev) / dt
                    if prev is not None
                    else np.zeros(dims)
                )
            previous_positions[entity_id] = position

            properties = {}
            for column in property_columns:
                value = (row.get(column) or "").strip()
                if value:
                    properties[column] = float(value)

            # Position/velocity columns are also exposed as properties,
            # reusing the arrays already parsed above (whether velocity
            # came from explicit columns or was finite-differenced).
            properties["px"] = float(position[0])
            properties["py"] = float(position[1])
            if dims == 3:
                properties["pz"] = float(position[2])

            properties["vx"] = float(velocity[0])
            properties["vy"] = float(velocity[1])
            if dims == 3:
                properties["vz"] = float(velocity[2])

            entities[entity_id] = EntityState(
                entity_id=entity_id,
                entity_class=str(row["entity_class"]).strip(),
                position=position,
                velocity=velocity,
                track_age=1,
                track_state=TrackState.CONFIRMED,
                properties=properties,
            )
        frames.append((float_time, entities))

    return frames, dt, dims


def _make_dataset(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> DatasetAdapter:
    """Registry factory. ``task`` is the path to the CSV file."""
    del robot, target_only
    if not task:
        raise ValueError(
            "The dataset adapter needs --task <path/to/trajectories.csv>"
        )
    return DatasetAdapter(task)


_register(
    name="dataset",
    factory=_make_dataset,
    description="Offline CSV trajectory dataset (no simulator)",
)
