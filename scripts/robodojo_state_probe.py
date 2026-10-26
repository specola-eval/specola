"""Probe whether RoboDojo's Isaac Lab scene state restores an observation.

Task 5 of ``docs/superpowers/plans/2026-10-13-state-replay.md``; the
numbers are on ``docs/evidence/state-replay-robodojo-2026-10-14.md``.
RoboDojo builds its robots as Isaac Lab articulations in an
``InteractiveScene`` (``TaskEnv.sim.scene``) and its task objects as Isaac
Sim core prims that its own ``SceneManager`` holds. The question is
whether the scene's ``get_state``/``reset_to`` pair, restored without a
physics step (the plan's decision 5), reproduces the adapter's whole
observation.

Lifecycle, the paired harness's (plan Task 3, Step 4):

1. Build the adapter through the registry factory for ``--task``, with a
   query configuration that forms and reads every pair distance.
2. ``adapter.reset()``, then row 0: ``scene.get_state(is_relative=False)``
   flattened by ``isaac_state`` for the monitored environment. Ten times:
   the adapter's ``hold`` advancer, capture, observe.
3. Restore row 0 (``scene.reset_to`` for the monitored environment,
   ``SimulationContext.forward()`` when it exists, ``scene.update(0.0)``),
   ``adapter.reset()``; then restore each later row and observe.
4. Compare the two passes key by key with ``TOLERANCES`` below.

Then three checks, in this order: (a) one scene object lifted by 5 cm
and row 0 restored, to see whether the state carries scene objects; (b)
``reset_to`` with ``env_ids`` as a Python list; (c) the paired lifecycle
again on the robot keys, with every joint target offset so the arms
move (under ``hold`` they do not), which also shows whether a restore
brings the link poses along and whether live link poses lag the joint
positions.

    env -u PYTHONPATH -u ROBODOJO_PLANNER PYTHONNOUSERSITE=1 \\
        OMNI_KIT_ACCEPT_EULA=YES ROBODOJO_ROOT=<RoboDojo checkout> \\
        PYTHONPATH=$PWD/src ~/miniconda3/envs/RoboDojo/bin/python \\
        scripts/robodojo_state_probe.py --out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROWS = 11  # row 0 at reset, rows 1 to 10 after each hold step
LIFT = 0.05  # metres, check (a)
ARM_OFFSET = 0.1  # rad (m on prismatic joints) added to every joint target, check (c)

# Fixed before the first run (plan Task 5, Step 1): positions and joint
# positions within 1e-6, velocities within 1e-5, bit equality for every
# other key (orientations, wrenches, distances, counters, time).
TOLERANCES = (
    (r"\.(body_pos_w_\w+|root_pos_w_\w+|aabb_(min|max)_\d|particle_centroid_\d"
     r"|joint_pos_\w+|position)$", 1e-6),
    (r"\.(joint_vel_\w+|body_lin_vel_w_\w+|body_ang_vel_w_\w+|root_lin_vel_w_\w+"
     r"|root_ang_vel_w_\w+|velocity)$", 1e-5),
)

# Property stems the adapter emits, for the per-family summary.
STEMS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
    "body_ang_vel_w", "link_wrench", "root_pos_w", "root_quat_w",
    "root_lin_vel_w", "root_ang_vel_w", "aabb_min", "aabb_max",
    "particle_count", "particle_centroid", "mesh_distance_sampled",
    "mesh_distance_clipped",
)

QUERIES = """\
schema: specola.queries.v1
populations:
- binding: robodojo.mesh_pair
  arguments:
    distmax: 0.05
    robot_links: [link6, link7, link8]
    population: all
instances: []
"""

_MISSING = object()


def harness():
    """``benchmarks/replay_fidelity.py``, for its ``flatten`` and ``difference``."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "replay_fidelity.py"
    spec = importlib.util.spec_from_file_location("replay_fidelity", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tolerance(key):
    for pattern, bound in TOLERANCES:
        if re.search(pattern, key):
            return bound
    return None


def family(key):
    """``<entity class>.<property stem>`` for an entity key, the key itself otherwise."""
    match = re.match(r"^entity\[\d+:(\w+)\]\.(.+)$", key)
    if match is None:
        return key
    entity_class, prop = match.groups()
    for stem in STEMS:
        if prop.startswith(stem):
            return f"{entity_class}.{stem}"
    return f"{entity_class}.{prop}"


def numeric(value):
    return isinstance(value, (float, np.ndarray))


def span(values):
    """Largest change of one key over a pass: max minus min, per component."""
    stacked = np.asarray([np.ravel(value) for value in values], dtype=float)
    return float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))


def compare(live, replay, difference):
    """Every key whose replayed value differs from the live one on some row.

    ``max_abs_diff_from_live_last`` is the largest difference between a
    replayed row and the LAST live row: zero for a key the restore left
    where pass 1 ended.
    """
    differing = {}
    for step, (recorded, replayed) in enumerate(zip(live, replay)):
        for key in recorded.keys() | replayed.keys():
            a, b = recorded.get(key, _MISSING), replayed.get(key, _MISSING)
            absent = a is _MISSING or b is _MISSING
            equal, gap = (False, None) if absent else difference(a, b)
            if equal:
                continue
            record = differing.setdefault(key, {
                "family": family(key), "tolerance": tolerance(key),
                "steps_differing": 0, "max_abs_diff": None, "step": None,
            })
            record["steps_differing"] += 1
            if record["step"] is None or (gap is not None and gap > (record["max_abs_diff"] or -1.0)):
                record["max_abs_diff"], record["step"] = gap, step
            if absent:
                record["missing_in"] = "live" if a is _MISSING else "replay"
    for key, record in differing.items():
        bound, gap = record["tolerance"], record["max_abs_diff"]
        record["within_tolerance"] = bound is not None and gap is not None and gap <= bound
        last = live[-1].get(key, _MISSING)
        gaps = [difference(last, flat[key])[1] for flat in replay
                if last is not _MISSING and key in flat]
        record["max_abs_diff_from_live_last"] = (
            None if not gaps or None in gaps else max(gaps))
    return differing


def families(live, differing):
    """Per family: keys, keys differing beyond and within tolerance, largest difference and span."""
    table = {}
    keys = set().union(*live)
    for key in sorted(keys):
        row = table.setdefault(family(key), {
            "keys": 0, "beyond": 0, "within": 0, "max_abs_diff": 0.0, "max_key": None,
            "live_span": 0.0, "from_last": None, "non_numeric": 0,
        })
        row["keys"] += 1
        values = [flat.get(key) for flat in live]
        if all(numeric(value) for value in values):
            row["live_span"] = max(row["live_span"], span(values))
        record = differing.get(key)
        if record is None:
            continue
        row["within" if record["within_tolerance"] else "beyond"] += 1
        from_last = record["max_abs_diff_from_live_last"]
        if from_last is not None:
            row["from_last"] = max(row["from_last"] or 0.0, from_last)
        gap = record["max_abs_diff"]
        if gap is None:   # missing on one side, or not a number
            row["non_numeric"] += 1
        elif gap > row["max_abs_diff"]:
            row["max_abs_diff"], row["max_key"] = gap, key
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Kit appends sys.argv to its own command line (tests/test_robodojo_adapter.py).
    sys.argv = sys.argv[:1]

    from specola.adapters import isaac_state
    from specola.adapters.robodojo import RoboDojoAdapter, _to_numpy
    from specola.query_config import load_query_configuration
    from specola.registry import resolve

    tools = harness()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    queries = args.out.with_name("probe_queries.yaml")
    queries.write_text(QUERIES)

    adapter = resolve("robodojo").factory(args.task, None, False)
    assert isinstance(adapter, RoboDojoAdapter), type(adapter)
    adapter.set_query_configuration(load_query_configuration(queries))

    import torch

    rl_env = adapter._env.sim           # RoboDojo's IsaacRLEnv, a DirectRLEnv
    scene, context = rl_env.scene, rl_env.sim
    index = adapter.env_index
    device = scene.device
    ids = torch.tensor([index], dtype=torch.long, device=device)
    forward = getattr(context, "forward", None)
    layout = isaac_state.layout(scene.get_state(is_relative=False))

    def capture():
        return isaac_state.flatten(scene.get_state(is_relative=False), index)

    def restore(row, env_ids=ids):
        state = isaac_state.unflatten(layout, row)
        # unflatten builds host tensors; the 0.54.3 writers index-assign
        # them into buffers on the simulation device.
        state = {
            kind: {
                asset: ({key: value.to(device) for key, value in leaves.items()}
                        if isinstance(leaves, dict) else leaves.to(device))
                for asset, leaves in assets.items()
            }
            for kind, assets in state.items()
        }
        scene.reset_to(state, env_ids=env_ids, is_relative=False)
        if forward is not None:
            forward()
        scene.update(0.0)

    # Pass 1: live.
    live = [tools.flatten(adapter.reset())]
    rows = [capture()]
    advance = adapter.make_advancer("hold", None, 0)
    for step in range(ROWS - 1):
        advance(step)
        rows.append(capture())
        live.append(tools.flatten(adapter.get_observation()))

    # Pass 2: restored, no physics step.
    replay = []
    for k, row in enumerate(rows):
        restore(row)
        replay.append(tools.flatten(adapter.reset() if k == 0 else adapter.get_observation()))

    differing = compare(live, replay, tools.difference)
    table = families(live, differing)
    labels = dict(adapter.entity_labels)
    robot_assets = [asset for _, asset, _ in adapter._robot_groups()]
    scene_objects = adapter._scene_objects()

    # (a) Does the state carry scene objects? Lift one, restore row 0.
    def height(obj):
        return float(np.asarray(_to_numpy(obj.get_world_pose()[0]), dtype=float).reshape(-1)[2])

    check_a = None
    for _, entity_class, name, obj in scene_objects:
        if not (hasattr(obj, "set_world_pose") and hasattr(obj, "get_world_pose")):
            continue
        position, orientation = (np.asarray(_to_numpy(part), dtype=float).reshape(-1)
                                 for part in obj.get_world_pose())
        obj.set_world_pose(position=position + np.array([0.0, 0.0, LIFT]), orientation=orientation)
        lifted = height(obj)
        restore(rows[0])
        entity_id = adapter._entity_id_of[f"object:{name}"]
        check_a = {
            "object": name, "label": labels.get(entity_id), "class": entity_class,
            "type": type(obj).__name__, "z_before": float(position[2]), "z_lifted": lifted,
            "z_after_restoring_row_0": height(obj),
            "z_live_row_0": live[0].get(f"entity[{entity_id}:{entity_class}].root_pos_w_2"),
        }
        break

    # (b) env_ids as a list, the form the plan names.
    try:
        restore(rows[0], env_ids=[index])
        check_b = {"accepted": True,
                   "joint_state_equals_row_0": bool(np.array_equal(capture(), rows[0]))}
    except Exception as exc:  # report it, whatever it is
        check_b = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}

    # (c) A moving arm. Under hold the arms barely move, so the robot keys
    # above say little about restoring a moving arm. Offset every robot's
    # joint position targets, repeat the paired lifecycle, compare the
    # robot keys. Row 0 is at rest; replayed after row 10, it shows whether
    # a restore brings the link poses along. Rows 1 to 10 then show whether
    # the live link poses describe the joint positions of their own row
    # (no lag), of the row before (a lag of one physics step, decimation 1).
    restore(rows[0])
    for asset in {id(asset): asset for asset in robot_assets}.values():
        asset.set_joint_position_target(asset.data.joint_pos + ARM_OFFSET)
    moving_live, moving_rows = [tools.flatten(adapter.reset())], [capture()]
    for step in range(ROWS - 1):
        advance(step)
        moving_rows.append(capture())
        moving_live.append(tools.flatten(adapter.get_observation()))
    moving_replay = []
    for k, row in enumerate(moving_rows):
        restore(row)
        moving_replay.append(tools.flatten(adapter.reset() if k == 0 else adapter.get_observation()))

    def robot_keys(flat):
        return {key: value for key, value in flat.items() if re.match(r"^entity\[\d+:robot\]", key)}

    moving_live = [robot_keys(flat) for flat in moving_live]
    moving_replay = [robot_keys(flat) for flat in moving_replay]
    moving = compare(moving_live, moving_replay, tools.difference)
    body = [key for key in moving_live[0] if ".body_pos_w_" in key]

    def gap(a, b):
        return max(abs(a[key] - b[key]) for key in body)

    check_c = {
        "arm_offset": ARM_OFFSET,
        "families": families(moving_live, moving),
        "differing": dict(sorted(moving.items())),
        "body_pos_w_by_row": [
            {"row": k,
             "replay_vs_live_same_row": gap(moving_replay[k], moving_live[k]),
             "replay_vs_live_previous_row": gap(moving_replay[k], moving_live[k - 1]) if k else None,
             "replay_vs_live_next_row": gap(moving_replay[k], moving_live[k + 1]) if k + 1 < ROWS else None,
             "live_motion_since_previous_row": gap(moving_live[k], moving_live[k - 1]) if k else None}
            for k in range(ROWS)
        ],
        "replay_row_0_vs_live_row_10": gap(moving_replay[0], moving_live[-1]),
        "state_leaf_spans": _leaf_spans(layout, np.asarray(moving_rows)),
    }

    report = {
        "task": args.task,
        "rows": ROWS,
        "tolerances": [list(item) for item in TOLERANCES],
        "versions": _versions(),
        "sim_device": str(device),
        "fabric_enabled": bool(context.is_fabric_enabled()),
        "simulation_context_forward": forward is not None,
        "physics_dt": float(rl_env.physics_dt), "step_dt": float(rl_env.step_dt),
        "decimation": int(rl_env.cfg.decimation), "adapter_dt": float(adapter.dt),
        "scene": {
            "articulations": sorted(scene.articulations),
            "rigid_objects": sorted(scene.rigid_objects),
            "deformable_objects": sorted(scene.deformable_objects),
            "rigid_object_collections": sorted(scene.rigid_object_collections),
            "surface_grippers": sorted(scene.surface_grippers),
            "sensors": sorted(scene.sensors),
            "extras": sorted(scene.extras),
            "num_envs": int(scene.num_envs),
        },
        "state_layout": layout,
        "state_length": isaac_state.state_length(layout),
        "robot_assets_in_state": [
            next((name for name, art in scene.articulations.items() if art is asset), None)
            for asset in robot_assets
        ],
        "scene_objects": [[name, entity_class, type(obj).__name__] for _, entity_class, name, obj in scene_objects],
        "entities": {str(eid): label for eid, label in labels.items()},
        "families": table,
        "differing": dict(sorted(differing.items())),
        "check_a_scene_object": check_a,
        "check_b_list_env_ids": check_b,
        "check_c_moving_arm": check_c,
        "state_leaf_spans": _leaf_spans(layout, np.asarray(rows)),
    }
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(summary(report))
    return 0


def _leaf_spans(layout, rows):
    """Largest change of each state leaf over pass 1, ``asset/key``."""
    spans, offset = {}, 0
    for asset, _, key, shape, _ in layout:
        size = int(np.prod(shape))
        block = rows[:, offset:offset + size]
        spans[f"{asset}/{key}"] = float(np.max(np.ptp(block, axis=0))) if size else 0.0
        offset += size
    return spans


def _versions():
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ("isaaclab", "isaacsim", "torch", "warp-lang", "numpy"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    found["python"] = sys.version.split()[0]
    return found


def summary(report):
    lines = [f"robodojo state probe, {report['task']}: {report['rows'] - 1} hold steps, "
             f"state length {report['state_length']}, fabric {report['fabric_enabled']}, "
             f"forward {report['simulation_context_forward']}"]
    lines.append(f"  scene: {json.dumps(report['scene'])}")
    lines.append(f"  robots in state: {report['robot_assets_in_state']}")
    lines.append(f"  scene objects (not InteractiveScene assets): {report['scene_objects']}")
    lines.append(f"  pass 1 state leaf spans: {json.dumps(report['state_leaf_spans'])}")
    lines.append("  family: keys, beyond tolerance, within tolerance, max |diff| (key), "
                 "live span, max |replay - live row 10|, keys without a numeric difference")
    for name, row in report["families"].items():
        lines.append(f"    {name}: {row['keys']}, {row['beyond']}, {row['within']}, "
                     f"{row['max_abs_diff']:.3g} ({row['max_key']}), {row['live_span']:.3g}, "
                     f"{row['from_last']}, non-numeric {row['non_numeric']}")
    lines.append(f"  (a) scene object: {json.dumps(report['check_a_scene_object'], default=str)}")
    lines.append(f"  (b) list env_ids: {json.dumps(report['check_b_list_env_ids'])}")
    moving = report["check_c_moving_arm"]
    lines.append(f"  (c) moving arm, offset {moving['arm_offset']}: state leaf spans "
                 f"{json.dumps(moving['state_leaf_spans'])}")
    for name, row in moving["families"].items():
        lines.append(f"    {name}: {row['keys']}, {row['beyond']}, {row['within']}, "
                     f"{row['max_abs_diff']:.3g} ({row['max_key']}), {row['live_span']:.3g}, "
                     f"{row['from_last']}, non-numeric {row['non_numeric']}")
    lines.append(f"    replay row 0 vs live row 10, body_pos_w: {moving['replay_row_0_vs_live_row_10']:.3g}")
    for row in moving["body_pos_w_by_row"]:
        lines.append(f"    body_pos_w row {row['row']}: {json.dumps(row)}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
