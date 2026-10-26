"""Probe whether CARLA restores an observation payload without a tick.

Task 7 of ``docs/superpowers/plans/2026-10-13-state-replay.md``. The
criterion is the plan's decision 5: after ``set_transform``,
``set_target_velocity`` and ``set_target_angular_velocity`` on the
monitored actors, and without a tick, does the adapter's observation
equal the one captured when the world held that state?

Lifecycle, as in the plan's paired harness (Task 3, Step 4):

1. Build the adapter through the registry factory and populate the
   world with the adapter's own ``random`` advancer, recording the
   vehicle and walker entity sets after each tick (fleet churn).
2. ``adapter.reset()`` gives row 0; ten ``hold`` ticks give rows 1 to 10.
   Each row keeps the monitored actors' transform, velocity, angular
   velocity and acceleration, and the whole observation as a flat map.
3. Restore row 0 on every vehicle and walker, ``adapter.reset()``,
   ``get_observation()``; compare both with row 0. Then restore row k
   and ``get_observation()`` for k = 1 to 10. No tick in this pass.
4. A second client session lists the actors while the first is open,
   before and after the tick of step 5.
5. After the comparison: restore row 0, tick once, observe, to show
   whether the setters reached the server at all.

Needs a running CARLA server (``SPECOLA_CARLA_HOST``/``PORT``). Prints
a summary and writes the full report as JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import carla

from specola.adapters.carla import _VEHICLE_PREFIX, _WALKER_PREFIX, CarlaAdapter
from specola.registry import resolve

ROWS = 11  # row 0 at reset, rows 1-10 after each hold tick
# Steps 0-41: the advancer's first churn runs after the tick of step 40
# and becomes visible at step 41, so pass 1 starts on a settled set.
RANDOM_TICKS = 42
COUNTERS = ("timestep", "track_age")  # adapter counters, not simulator state


def flat(observation):
    """Every field of one MonitorObservation, keyed (entity id, name)."""
    out = {
        ("", "time"): observation.time,
        ("", "timestep"): observation.timestep,
        ("", "component_tags"): repr(dict(observation.component_tags)),
        ("", "extraction"): repr(observation.extraction),
    }
    for entity_id, entity in observation.perceived_entities.items():
        out[(entity_id, "class")] = entity.entity_class
        out[(entity_id, "track_age")] = entity.track_age
        out[(entity_id, "track_state")] = str(entity.track_state)
        for name, vector in (("position", entity.position), ("velocity", entity.velocity)):
            if vector is not None:
                for axis, value in enumerate(vector):
                    out[(entity_id, f"{name}[{axis}]")] = float(value)
        for key, value in entity.properties.items():
            out[(entity_id, key)] = float(value)
    return out


def moving_actors(world):
    return {
        actor.id: actor
        for actor in world.get_actors()
        if actor.type_id.startswith((_VEHICLE_PREFIX, _WALKER_PREFIX))
    }


def snapshot(world):
    """Kinematics of every vehicle and walker as the client last received them."""
    rows = {}
    for actor_id, actor in moving_actors(world).items():
        transform = actor.get_transform()
        location, rotation = transform.location, transform.rotation
        rows[actor_id] = {
            "type_id": actor.type_id,
            "transform": [location.x, location.y, location.z,
                          rotation.pitch, rotation.yaw, rotation.roll],
            **{
                name: [vector.x, vector.y, vector.z]
                for name, vector in (
                    ("velocity", actor.get_velocity()),
                    ("angular_velocity", actor.get_angular_velocity()),
                    ("acceleration", actor.get_acceleration()),
                )
            },
        }
    return rows


def restore(world, rows):
    """Apply one snapshot through the setters; return ids no longer alive."""
    actors = moving_actors(world)
    missing = []
    for actor_id, row in rows.items():
        actor = actors.get(actor_id)
        if actor is None:
            missing.append(actor_id)
            continue
        x, y, z, pitch, yaw, roll = row["transform"]
        actor.set_transform(carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
        ))
        actor.set_target_velocity(carla.Vector3D(*row["velocity"]))
        actor.set_target_angular_velocity(carla.Vector3D(*row["angular_velocity"]))
    return missing


def difference(a, b):
    """Absolute difference of two payload values; inf for a type or presence mismatch."""
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return 0.0 if a == b else math.inf
    return abs(float(a) - float(b))


def compare(expected, actual):
    """Largest absolute difference per (entity id, name) over the union of keys."""
    return {key: difference(expected.get(key), actual.get(key))
            for key in expected.keys() | actual.keys()}


def state_gap(expected, actual):
    """Largest difference over every key except the adapter counters."""
    return max(diff for key, diff in compare(expected, actual).items()
               if key[1] not in COUNTERS)


def group(entity_class, key):
    return f"{entity_class}.{key[1]}" if key[0] != "" else key[1]


def per_key(rows, restored, class_of):
    """Aggregate by (class, key) over entities and rows."""
    table: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {
        "range_in_pass1": 0.0, "max_abs_restore_vs_captured": 0.0,
        "worst_row": None, "cells": 0, "cells_differing": 0,
        "max_abs_restore_vs_row10": 0.0,
    })
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        name = group(class_of.get(key[0], ""), key)
        entry = table[name]
        numeric = [v for v in values if isinstance(v, (int, float))]
        if len(numeric) == len(values):
            entry["range_in_pass1"] = max(entry["range_in_pass1"],
                                          max(numeric) - min(numeric))
        elif len(set(map(repr, values))) > 1:
            entry["range_in_pass1"] = math.inf
    for k, (captured, observed) in enumerate(zip(rows, restored)):
        for key, diff in compare(captured, observed).items():
            entry = table[group(class_of.get(key[0], ""), key)]
            entry["cells"] += 1
            if diff > 0:
                entry["cells_differing"] += 1
            if diff > entry["max_abs_restore_vs_captured"]:
                entry["max_abs_restore_vs_captured"] = diff
                entry["worst_row"] = k
        for key, diff in compare(rows[-1], observed).items():
            entry = table[group(class_of.get(key[0], ""), key)]
            entry["max_abs_restore_vs_row10"] = max(
                entry["max_abs_restore_vs_row10"], diff)
    return dict(sorted(table.items()))


def transform_gap(a_rows, b_rows, prefix=("",), width=6):
    """Largest absolute difference of the first ``width`` transform entries over shared actors."""
    shared = [i for i in a_rows.keys() & b_rows.keys()
              if a_rows[i]["type_id"].startswith(prefix)]
    return max((abs(a - b)
                for i in shared
                for a, b in zip(a_rows[i]["transform"][:width], b_rows[i]["transform"][:width])),
               default=None)


def entity_sets(observation, adapter):
    sets = {adapter.vehicle_class: set(), adapter.walker_class: set()}
    for entity_id, entity in observation.perceived_entities.items():
        if entity.entity_class in sets:
            sets[entity.entity_class].add(entity_id)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="town to load; default: the loaded world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    adapter = resolve("carla").factory(args.task, None, False)
    assert isinstance(adapter, CarlaAdapter)
    world = adapter._world
    report = {
        "client": adapter._client.get_client_version(),
        "server": adapter._client.get_server_version(),
        "map": world.get_map().name,
        "dt": adapter.dt,
    }

    # 1. Populate through the adapter's own random advancer; record churn.
    advance_random = adapter.make_advancer("random", None, args.seed)
    report["random_actors_before_first_tick"] = len(moving_actors(world))
    adapter.reset()
    churn, previous = [], None
    for step in range(RANDOM_TICKS):
        advance_random(step)
        sets = entity_sets(adapter.get_observation(), adapter)
        record: dict[str, Any] = {"step": step, **{name: len(ids) for name, ids in sets.items()}}
        if previous is not None:
            record["added"] = sorted(
                i for name in sets for i in sets[name] - previous[name])
            record["removed"] = sorted(
                i for name in sets for i in previous[name] - sets[name])
        churn.append(record)
        previous = sets
    report["random_mode"] = churn

    # 2. Pass 1: reset, then ten hold ticks.
    advance_hold = adapter.make_advancer("hold", None, args.seed)
    rows, actor_rows, frames = [], [], []
    rows.append(flat(adapter.reset()))
    actor_rows.append(snapshot(world))
    frames.append(world.get_snapshot().frame)
    for step in range(1, ROWS):
        advance_hold(step)
        rows.append(flat(adapter.get_observation()))
        actor_rows.append(snapshot(world))
        frames.append(world.get_snapshot().frame)
    class_of = {key[0]: value for row in rows for key, value in row.items()
                if key[1] == "class"}
    report["population"] = {
        name: sum(1 for c in class_of.values() if c == name)
        for name in sorted(set(class_of.values()))
    }
    report["pass1_frames"] = frames
    report["restored_actors"] = len(actor_rows[0])

    # 3. Pass 2: restore each row without a tick.
    missing = restore(world, actor_rows[0])
    reset_observation = flat(adapter.reset())
    restored = [flat(adapter.get_observation())]
    report["row0_reset_observation"] = {
        "max_abs_vs_row0_without_counters": state_gap(rows[0], reset_observation),
        "max_abs_vs_row10_without_counters_and_time": state_gap(
            {k: v for k, v in rows[-1].items() if k != ("", "time")},
            {k: v for k, v in reset_observation.items() if k != ("", "time")}),
        "time": reset_observation[("", "time")],
    }

    # 4. A second client session, while the restored row 0 is pending.
    host = os.environ.get("SPECOLA_CARLA_HOST", "127.0.0.1")
    port = int(os.environ.get("SPECOLA_CARLA_PORT", "2000"))
    other = carla.Client(host, port)
    other.set_timeout(60.0)
    other_world = other.get_world()
    mine = {a.id: a.type_id for a in world.get_actors()}
    theirs = {a.id: a.type_id for a in other_world.get_actors()}
    second = {
        "same_actor_ids_and_types": mine == theirs,
        "actors_first_session": len(mine),
        "actors_second_session": len(theirs),
        "frame_first_session": world.get_snapshot().frame,
        "frame_second_session": other_world.get_snapshot().frame,
    }
    try:
        other_rows = snapshot(other_world)
        second["moving_actor_ids_equal"] = set(other_rows) == set(actor_rows[0])
        second["transform_max_abs_vs_row0"] = transform_gap(other_rows, actor_rows[0])
        second["transform_max_abs_vs_row10"] = transform_gap(other_rows, actor_rows[-1])
    except RuntimeError as error:
        second["transform_read_error"] = str(error)
    report["second_session"] = second

    for k in range(1, ROWS):
        missing += restore(world, actor_rows[k])
        restored.append(flat(adapter.get_observation()))
    report["missing_actor_ids_on_restore"] = sorted(set(missing))
    report["frame_after_pass2"] = world.get_snapshot().frame
    report["per_key"] = per_key(rows, restored, class_of)
    report["round_trips"] = all(
        entry["max_abs_restore_vs_captured"] == 0.0
        for name, entry in report["per_key"].items()
        if name.rsplit(".", 1)[-1] not in COUNTERS
    )

    # 5. One tick after restoring row 0: did the setters reach the server?
    restore(world, actor_rows[0])
    advance_hold(ROWS)
    ticked = snapshot(world)
    report["one_tick_after_restoring_row0"] = {
        name: {
            "location_max_abs_vs_row0": transform_gap(ticked, actor_rows[0], prefix, 3),
            "location_max_abs_vs_row10": transform_gap(ticked, actor_rows[-1], prefix, 3),
            "pass1_location_max_abs_row0_vs_row10": transform_gap(
                actor_rows[0], actor_rows[-1], prefix, 3),
        }
        for name, prefix in (("vehicle", _VEHICLE_PREFIX), ("walker", _WALKER_PREFIX))
    }

    # The second session again, once the tick has reached it.
    frame = world.get_snapshot().frame
    deadline = time.monotonic() + 5.0
    while other_world.get_snapshot().frame != frame and time.monotonic() < deadline:
        time.sleep(0.05)
    theirs = {a.id: a.type_id for a in other_world.get_actors()}
    mine = {a.id: a.type_id for a in world.get_actors()}
    second["after_one_tick"] = {
        "frame_first_session": frame,
        "frame_second_session": other_world.get_snapshot().frame,
        "same_actor_ids_and_types": mine == theirs,
        "actors_first_session": len(mine),
        "actors_second_session": len(theirs),
        "transform_max_abs_vs_first_session": transform_gap(snapshot(other_world), ticked),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    print(f"map {report['map']}  server {report['server']}  dt {report['dt']}")
    print(f"population at reset: {report['population']}")
    print("random mode (step: vehicles walkers +added -removed):")
    for record in churn:
        print(f"  {record['step']:>2}: {record['vehicle']:>3} {record['walker']:>3}"
              f" +{len(record.get('added', []))} -{len(record.get('removed', []))}")
    print(f"row 0 reset observation: {report['row0_reset_observation']}")
    print(f"second session: {second}")
    print(f"{'key':<34}{'range pass1':>14}{'restore diff':>14}{'row':>5}"
          f"{'differing':>12}{'vs row 10':>12}")
    for name, entry in report["per_key"].items():
        print(f"{name:<34}{entry['range_in_pass1']:>14.6g}"
              f"{entry['max_abs_restore_vs_captured']:>14.6g}"
              f"{str(entry['worst_row']):>5}"
              f"{entry['cells_differing']:>6}/{entry['cells']:<5}"
              f"{entry['max_abs_restore_vs_row10']:>12.6g}")
    print(f"one tick after restoring row 0: {report['one_tick_after_restoring_row0']}")
    print(f"second session after that tick: {second['after_one_tick']}")
    print(f"round_trips_without_tick: {report['round_trips']}")


if __name__ == "__main__":
    main()
