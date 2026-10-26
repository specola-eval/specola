# RoboDojo state restore probe, 2026-10-14

Task 5 of the state-replay plan
(`docs/superpowers/plans/2026-10-13-state-replay.md`) asks whether the
`robodojo` adapter can have a Specola state tier built on Isaac Lab's
`InteractiveScene.get_state` and `reset_to`, through the helpers the
`isaaclab` adapter uses (`src/specola/adapters/isaac_state.py`). It cannot.
RoboDojo's Isaac Lab scene holds the two robot articulations and nothing
else; the bowls and the camera stand the adapter exports are Isaac Sim core
prims that RoboDojo's own `SceneManager` holds outside that scene. Restoring
the scene state left every bowl key at its value after the last live step,
beyond the tolerance fixed before the run, and a bowl lifted by 5 cm stayed
lifted through a restore. The adapter keeps refusing replay and
`state_signature()` stays None (decision 4).

The probe ran at commit `69b10fc` of branch `adapter-parity`, plus
`scripts/robodojo_state_probe.py` and the `make_advancer` docstring as
committed with this page. Its log, summary and JSON report are under
`results/state-replay-robodojo-2026-10-14/` in the worktree it ran in (not
committed); the numbers below are copied from them.

Maintainer decisions in force: option (a) of the plan's global
constraints (a state tier reproduces state-determined keys; step outputs
are listed with their measured differences), the `.npz` container, and
live observation semantics unchanged.

## What was run

| Environment | Packages | Physics | GPU |
|---|---|---|---|
| `RoboDojo` | RoboDojo `726e9aa`; `isaaclab` 0.54.3 (editable, RoboDojo's `third_party/IsaacLab`), `isaacsim` 5.1.0.0, `torch` 2.7.0+cu128, `warp-lang` 1.11.0, `numpy` 1.26.0, Python 3.11.15 | PhysX on the CPU (the environment's log reports its device as `cpu`), Fabric off | RTX 5090, driver 580.178.04; the Kit app runs on `cuda:0` |

The task is `stack_bowls`, layout 0 (`ROBODOJO_SEED` and `ROBODOJO_LAYOUT`
unset), built by the registry factory with one environment and
`ROBODOJO_PLANNER` unset, so no `cuRobo` planner is built. RoboDojo's
`env_cfg/sim/sim_config.yml` sets a physics step of 0.004 s and decimation
1, so one `hold` step is one physics step and the adapter's `dt` is
0.004 s. The probe installs a query configuration that forms and reads
every pair distance (`robodojo.mesh_pair`, `distmax: 0.05`,
`robot_links: [link6, link7, link8]`, `population: all`), so the compared
observation carries 21 entities: the two arms, three bowls, the camera
stand and 15 `mesh_pair` entities.

The probe ran from the worktree root, started only when
`pgrep -af "[k]it/python"` and
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` printed
nothing, in its own process group, which was killed after the probe
exited. `OUT` is `results/state-replay-robodojo-2026-10-14`:

```bash
setsid bash -c 'echo $$ > "$0"; exec timeout 1500 "$@"' "$OUT/probe.pgid" \
  env -u PYTHONPATH -u ROBODOJO_PLANNER PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  ROBODOJO_ROOT=/path/to/simulators/RoboDojo \
  PYTHONPATH=<worktree>/src ~/miniconda3/envs/RoboDojo/bin/python \
  scripts/robodojo_state_probe.py --out "$OUT/report.json" \
  > "$OUT/probe.log" 2>&1 < /dev/null
PGID=$(cat "$OUT/probe.pgid")
kill -TERM -- -"$PGID"
for i in $(seq 20); do kill -0 -- -"$PGID" 2>/dev/null || break; sleep 1; done
kill -0 -- -"$PGID" 2>/dev/null && kill -KILL -- -"$PGID"
```

The probe exited 0. After the kill the group was gone,
`pgrep -af "[k]it/python"` printed nothing and `nvidia-smi` listed no
compute process. Five runs gave the same numbers wherever they measured
the same quantity; the first two carried an earlier set of checks.

## The scene and its state

| Question | Answer |
|---|---|
| Does `self._env` reach an `InteractiveScene`? | Yes: `adapter._env.sim` is RoboDojo's `IsaacRLEnv` (a `DirectRLEnv`), `.sim.scene` its scene and `.sim.sim` its `SimulationContext` |
| What does the scene hold? | Articulations `robot0` and `robot1`; no rigid objects, deformable objects, rigid-object collections, surface grippers, sensors or extras; `num_envs` 1 |
| What does `get_state` give? | 8 float32 leaves, length 58: per robot `root_pose` [7], `root_velocity` [6], `joint_position` [8], `joint_velocity` [8] |
| Does it cover every asset the adapter exports? | No. The arms `left_arm` and `right_arm` are `robot0` and `robot1` (the same objects). `bowl0`, `bowl1` and `bowl2` (`RigidObject`, an Isaac Sim `SingleRigidPrim`, RoboDojo `env/scene_manager/objects/rigid.py:18`) and `camera_stand` (`GeometryObject`) are not in it, nor are the 15 `mesh_pair` entities, whose distances and centres follow those bodies |
| `SimulationContext.forward` in 0.54.3? | Present (`simulation_context.py:533`); it updates articulation kinematics only when Fabric is on (`:535`), and Fabric is off here |
| Does `reset_to` forward `env_ids` to surface grippers? | No, as in 6.1.17 (`interactive_scene.py:561`); RoboDojo's scene has none |
| Does `reset_to` take `env_ids` as a Python list? | No: `AttributeError: 'list' object has no attribute 'to'`. The 0.54.3 writers pass `env_ids` on to the PhysX views as their indices (`articulation.py:446`, `:616`); which call raised was not traced. The probe passes a `torch.long` tensor on the scene's device |

## What the probe does

The tolerance was fixed before the first run: positions (`body_pos_w_*`,
`root_pos_w_*`, `aabb_*`, the entity `position`) and joint positions
within 1e-6 absolute, velocities (`joint_vel_*`, `body_lin_vel_w_*`,
`body_ang_vel_w_*`, `root_lin_vel_w_*`, `root_ang_vel_w_*`, the entity
`velocity`) within 1e-5 absolute, bit equality for every other key
(quaternions, wrenches, distances, time, counters).

1. Builds the adapter through the registry factory,
   `resolve("robodojo").factory("stack_bowls", None, False)`, and
   installs the query configuration above.
2. Pass 1: `adapter.reset()` and row 0, `scene.get_state(is_relative=False)`
   flattened for environment 0 by `isaac_state.flatten`; then ten times
   the adapter's `make_advancer("hold", None, 0)` step, a capture and
   `get_observation()`. Each observation is flattened whole by the paired
   harness's `flatten` (`benchmarks/replay_fidelity.py`): time, timestep,
   and per entity its class, track age, track state, position, velocity
   and every property.
3. Pass 2, with no physics step: for each row, `isaac_state.unflatten`,
   `scene.reset_to(state, env_ids=<tensor [0]>, is_relative=False)`,
   `SimulationContext.forward()`, `scene.update(0.0)`; then
   `adapter.reset()` after row 0 and `get_observation()` after every
   later row. This is the harness's lifecycle: the adapter counts its own
   steps, so the reset restarts its clock.
4. Compares the passes key by key, and for every key that differs also
   the largest difference between a replayed row and live row 10.
5. Check (a): lifts `bowl0` by 5 cm with `set_world_pose`, restores row 0
   and reads the bowl's height.
6. Check (b): `reset_to` with `env_ids=[0]`.
7. Check (c): restores row 0, adds 0.1 to every joint position target of
   both arms (radians on revolute joints, metres on the prismatic finger
   joints, whose targets then lie beyond their limits), and repeats steps
   2 to 4 on the arms' keys, because under `hold` the arms do not move.

## Results under hold

Over pass 1 the arms' joint positions changed by at most 2.5e-16 rad and
their joint velocities by at most 9.1e-13 rad/s: under `hold` the arms
stood still. The bowls moved by micrometres; what moved them was not
traced.

| Keys | Count | Beyond tolerance | Within tolerance | Largest difference | Change over pass 1 |
|---|---|---|---|---|---|
| arm `joint_pos_*`, `joint_vel_*` | 32 | 0 | 0 | 0 | 2.5e-16 rad, 9.1e-13 rad/s |
| arm `body_pos_w_*` | 66 | 0 | 46 | 1.19e-7 m | 0 |
| arm `body_quat_w_*` | 88 | 76 | 0 | 1.19e-7 | 0 |
| arm `body_lin_vel_w_*`, `body_ang_vel_w_*` | 132 | 0 | 120 | 3.6e-12 m/s | 1.9e-14 m/s, 7.6e-14 rad/s |
| arm `link_wrench_*` | 132 | 120 | 0 | 3.0e-11 | 3.0e-11 |
| bowl `root_pos_w_*` and `position` | 12 | 3 | 7 | 8.9e-6 m (`bowl0`) | 8.9e-6 m |
| bowl `root_quat_w_*` | 12 | 12 | 0 | 1.79e-4 (`bowl0`) | 1.79e-4 |
| bowl `root_lin_vel_w_*` and `velocity` | 12 | 9 | 3 | 6.7e-5 m/s (`bowl0`) | 6.7e-5 m/s |
| bowl `root_ang_vel_w_*` | 9 | 9 | 0 | 3.29e-3 rad/s (`bowl0`) | 3.29e-3 rad/s |
| bowl `aabb_min_*`, `aabb_max_*` | 18 | 8 | 10 | 4.19e-5 m (`bowl0`) | 4.19e-5 m |
| `mesh_pair` `position` (centre of the pair) | 15 | 5 | 10 | 4.63e-6 m (`bowl0` pairs) | 4.63e-6 m |
| `mesh_pair` `mesh_distance_sampled`, `mesh_distance_clipped` | 30 | 0 | 0 | 0 | 0 |
| `camera_stand` pose, box and velocity keys | 15 | 0 | 0 | 0 | 0 |
| arm root `position` and `velocity`, `mesh_pair` `velocity` | 19 | 0 | 0 | 0 | 0 |
| time, timestep, class, track age and state of every entity | 65 | 0 | 0 | 0 | |

Every bowl key that differed had, on every row of pass 2, exactly the value
it had at live row 10 (largest difference from row 10: 0.0): the restore
did not move the bowls, so each replayed row reported where pass 1 ended.
The bowl numbers are therefore the bowls' own motion over the ten steps,
and grow with it. The `mesh_pair` centres differed where a bowl is one of
the pair. The pair distances did not change during pass 1, so their
equality says nothing about a restore.

The arm keys under `hold` came back within float32 round-off: link
positions within 1.19e-7 m (the float32 spacing at 1.0 is 1.19e-7) and the
link quaternions by the same amount, which the bit-equality rule counts as
differing. The `link_wrench_*` keys are incoming joint forces of the last
physics step, which `get_state` does not carry; in pass 2 they kept the
values of live row 10.

## The bowls are not in the state

Check (a) lifted `bowl0` (instance `bowl_10_1`) from z = 0.7898915 m to
0.8398915 m and restored row 0, in which the bowl stood at 0.7898915 m. After the
restore the bowl was at 0.8398915 m, where the lift had put it.

## The arms in motion

With the joint targets offset, the arms' joint positions changed by up to
0.106 rad and their joint velocities by up to 5.19 rad/s over pass 1.

| Arm keys | Count | Beyond tolerance | Within tolerance | Largest difference |
|---|---|---|---|---|
| `joint_pos_*`, `joint_vel_*` | 32 | 0 | 0 | 0, bit for bit |
| `body_pos_w_*` | 66 | 8 | 50 | 4.23e-5 m, finger link `link7` x, row 4 |
| `body_quat_w_*` | 88 | 80 | 0 | 2.98e-7 |
| `body_lin_vel_w_*` | 66 | 58 | 2 | 0.0231 m/s, `link2`, row 1 |
| `body_ang_vel_w_*` | 66 | 48 | 12 | 0.0934 rad/s, `link2`, row 1 |
| `link_wrench_*` | 132 | 120 | 0 | 579 N, `link1`; pass 2 kept the values of live row 10 |

The eight link-position keys beyond 1e-6 are the x and z components of the
finger links `link7` and `link8` of both arms (up to 4.23e-5 m and 1.9e-5 m
along x, 3.2e-6 m and 1.0e-6 m along z); the arm links `link1` to `link6`
came back within 1e-6. Link velocities differed on almost every link
while the joint velocities came back bit for bit. Neither cause was
traced.

Link positions per row, as the largest difference over the arms'
`body_pos_w_*` keys:

| Row | Replay against live, same row | Replay against live, previous row | Live motion since the previous row |
|---|---|---|---|
| 0 | 0.0 | | |
| 1 | 3.0e-7 m | 5.04e-3 m | 5.04e-3 m |
| 2 | 3.12e-5 m | 7.32e-3 m | 7.32e-3 m |
| 3 | 3.99e-5 m | 9.77e-3 m | 9.77e-3 m |
| 4 | 4.23e-5 m | 9.79e-3 m | 9.80e-3 m |
| 5 | 2.35e-5 m | 8.85e-3 m | 8.86e-3 m |
| 6 | 1.35e-5 m | 6.95e-3 m | 6.95e-3 m |
| 7 | 6.5e-6 m | 4.66e-3 m | 4.66e-3 m |
| 8 | 2.8e-6 m | 2.57e-3 m | 2.57e-3 m |
| 9 | 8.3e-7 m | 9.67e-4 m | 9.66e-4 m |
| 10 | 2.7e-7 m | 3.05e-4 m | 3.05e-4 m |

Replayed row 0 was restored straight after row 10, whose link positions
were up to 5.52 cm away, and matched live row 0 exactly: writing the joint
state moved the link poses, although `forward()` updates nothing with
Fabric off.

### The live kinematic lag

None was measured. Each replayed row matched the live row of the same
index within 4.23e-5 m and differed from the previous live row by that
step's whole motion (up to 9.8 mm), so a live observation's link poses
describe the joint positions of its own step, not those of the step
before. RoboDojo's `sim_step` calls `scene.update` after each physics step
(`env/environment/isaac/direct_rl_env.py:19`), which is where the adapter's
buffers are refreshed. Under `hold` alone the arms did not move, so the
`hold` pass could not show a lag; the measurement comes from check (c).

## Consequence

`state_signature()` stays at the base class's None, so a `specola run`
on `robodojo` writes no state file and `--mode replay` of a Specola state
recording is refused with "it has no state tier." The `make_advancer`
docstring states the finding, and the sim-free adapter test asserts that
`state_signature()` and `capture_state()` return None. With no tier the
paired harness cannot run on this adapter, so `benchmarks/replay_fidelity.py`
gains no `robodojo` entry in `LIVE_KINEMATIC_LAG` or `RESTORE_TOLERANCE`,
and the factory gains no `seed` (RoboDojo picks its layout from
`ROBODOJO_SEED` and `ROBODOJO_LAYOUT`).

A tier on this adapter would need at least a capture and restore of every
`SceneManager` object beside `get_state` (rigid and dynamic object poses
and velocities, articulated fixtures' joint states, garment and fluid
particles), and an account of the finger-link positions and link
velocities above. That is outside this task and was not built.

Sim-free tests after the change, in the `RoboDojo` environment:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
  ~/miniconda3/envs/RoboDojo/bin/python -m pytest tests/test_robodojo_adapter.py \
  tests/test_robodojo_scene.py tests/test_isaac_state.py tests/test_replay_fidelity.py \
  -m "not robodojo and not libero and not robocasa and not metaworld and not dmcontrol and not mujoco and not pybullet" -q
```

`50 passed, 2 skipped, 8 deselected, 1 warning`; the two skips need `pxr`.

## Limits

- One task and layout, one environment, PhysX on the CPU with Fabric off.
  A GPU simulation device, where RoboDojo turns Fabric on and `forward()`
  updates kinematics, was not measured.
- The bowls moved by micrometres under `hold`; their differences are their
  own motion and scale with it. Check (a) shows the mechanism, not a bound.
- Check (c) drives the arms by offsetting their joint targets, not with a
  policy, for ten physics steps.
- The causes of the finger-link position differences and of the link
  velocity differences were not traced.
