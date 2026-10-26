# CARLA state restore probe, 2026-10-14

Task 7 of the state-replay plan
(`docs/superpowers/plans/2026-10-13-state-replay.md`) asks whether the
`carla` adapter can have a Specola state tier. The criterion is the
plan's decision 5: after `set_transform`, `set_target_velocity` and
`set_target_angular_velocity` on the monitored actors, and without a
tick, the adapter's observation must equal the one captured when the
world held that state. It does not. Every value the observation reads
from CARLA came back as the last tick's value, whatever the setters
asked for. The adapter keeps refusing replay and `state_signature()`
stays None (decision 4).

## Setup

- Server: CARLA 0.9.16 (`simulators/CARLA_0.9.16`, whose `VERSION` file
  reads `0.9.16`) on the GPU workstation, started fresh for the recorded
  run.
- Client: `carla` 0.9.16 in the maintained `carla` conda environment,
  Python 3.12.13.
- Specola: branch `adapter-parity` at `5fd5bc9`, plus
  `scripts/carla_state_probe.py` as committed with this page, with
  `PYTHONPATH` set to the worktree's `src`.
- World: the server's default map, `Carla/Maps/Town10HD_Opt` (no
  `--task`). The registry factory put it in synchronous mode at
  `dt` = 0.05 s.
- Documentation: the 0.9.16 Python API reference that ships with the
  server, `PythonAPI/python_api.md`. For `carla.Actor.get_transform`,
  `get_velocity`, `get_angular_velocity` and `get_acceleration` it says
  that the value is the one the client received during the last tick
  and that the method does not call the simulator. The vehicle and walker
  `get_control`, `Vehicle.get_speed_limit`,
  `Vehicle.is_at_traffic_light` and every traffic-light getter the
  adapter reads carry the same statement. `set_target_velocity` and
  `set_target_angular_velocity` are "applied before the physics step."
  In the installed client, `carla.TrafficLight` has setters for the
  state and for the green, yellow and red times, and none for the
  elapsed time (`dir(carla.TrafficLight)`).

## Commands

`OUT` is the results directory,
`results/state-replay-carla-2026-10-14/` in the worktree the probe ran
in (not committed). The server was started as the surface-parity audit
plan (`docs/superpowers/plans/2026-10-02-surface-parity-audit.md`, Task
4 Step 2) describes, after `pgrep -af "[C]arlaUE4"` printed nothing:

```bash
cd /path/to/simulators/CARLA_0.9.16
setsid sh -c 'echo $$ > "$1"; exec ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound' \
  sh "$OUT/carla-server.pgid" < /dev/null > "$OUT/carla-server.txt" 2>&1 &
sleep 2
ps -o pid,pgid,sid,cmd -g "$(cat "$OUT/carla-server.pgid")"
```

`ps` showed 3379484 as the PGID and SID of both `CarlaUE4.sh` and
`CarlaUE4-Linux-Shipping`. A client poll
(`carla.Client("127.0.0.1", 2000).get_server_version()`, retried for up
to 120 s) printed `client 0.9.16 server 0.9.16` on its first attempt.
The probe, from the worktree root:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH=<worktree>/src \
  ~/miniconda3/envs/carla/bin/python scripts/carla_state_probe.py \
  --out results/state-replay-carla-2026-10-14/report.json
```

It exited 0. Its report is `state-replay-carla-2026-10-14.json` beside
this page; its printed summary is `$OUT/probe.txt`. The server was then
stopped:

```bash
PGID=$(cat "$OUT/carla-server.pgid")
kill -INT -- -"$PGID"
for i in $(seq 30); do kill -0 -- -"$PGID" 2>/dev/null || break; sleep 1; done
kill -0 -- -"$PGID" 2>/dev/null && kill -KILL -- -"$PGID"
sleep 2; kill -0 -- -"$PGID" 2>/dev/null && echo "CARLA group still alive" || echo "CARLA group gone"
```

It printed `CARLA group gone`, and `pgrep -af "[C]arlaUE4"` then printed
nothing. The server log ends with `Segmentation fault (core dumped)`
after the SIGINT, as on the fresh-install page
(`fresh-install-carla-2026-10-13.md`).

## What the probe does

1. Builds the adapter through the registry factory,
   `resolve("carla").factory(None, None, False)`.
2. Populates the world with the adapter's own `random` advancer (seed 0)
   and drives it for steps 0 to 41, observing after each tick. A fresh
   server has no vehicles or walkers and `hold` only ticks, so without
   this step the probe would have nothing to restore.
3. Pass 1: `adapter.reset()` gives row 0, and ten ticks of
   `make_advancer("hold", None, 0)` give rows 1 to 10. Each row keeps
   every vehicle's and walker's transform, velocity, angular velocity
   and acceleration as the `get_*` calls returned them, and the whole
   `MonitorObservation` as a flat map: time, timestep,
   `component_tags`, `extraction`, and per entity its class, track age,
   track state, position, velocity and every property.
4. Pass 2, with no tick: applies row 0 to every vehicle and walker with
   `set_transform`, `set_target_velocity` and
   `set_target_angular_velocity`, calls `adapter.reset()` and
   `get_observation()`, and compares the result with row 0; then, for k
   = 1 to 10, applies row k, calls `get_observation()` and compares with
   row k. The extra `get_observation()` after the reset shifts pass 2's
   timestep and track ages by one. Both are adapter counters, reported
   apart from the values read from CARLA.
5. Opens a second `carla.Client` to the same server while the first is
   open and lists the actors, before and after one tick.
6. Applies row 0 again, ticks once and reads the transforms, which
   shows whether the setters reached the server at all.

## Results

At pass 1's reset the observation held 25 regions, 15 traffic lights,
20 vehicles and 8 walkers (the advancer attempts 10 walkers; 2 spawns
did not succeed). Pass 1 covered frames 54 to 64.

### Pass 2 against pass 1

On every row of pass 2, every value the observation reads from CARLA
equalled row 10, the last tick: the largest absolute difference from row
10 over those keys was 0.0, and the world snapshot stayed at frame 64.
Against the row each step had restored, the observation differed
wherever pass 1 had moved. The reset observation of pass 2 differed from
row 0 by up to 26.70 (vehicle `acceleration_y`, m/s^2) and from row 10
by 0.0, time and counters aside.

Keys whose values changed during pass 1 are below. "Range" is the
largest change of one entity's value over rows 0 to 10. "Restore
difference" is the largest absolute difference between pass 2 and the
row it restored, with that row's index. "Cells" counts entity rows that
differ, out of all entity rows. The entity `position` and `velocity`
vectors carry the same numbers as `location_*` and `velocity_*` and are
left out.

| Key | Unit | Range in pass 1 | Restore difference (row) | Cells differing | Difference from row 10 |
|---|---|---|---|---|---|
| `time` | s | 0.5 | 0.5 (10) | 10/11 | 0.5 |
| `traffic_light.elapsed_time` | s | 0.5 | 0.5 (0) | 50/165 | 0 |
| `vehicle.location_x` | m | 4.045 | 4.045 (0) | 195/220 | 0 |
| `vehicle.location_y` | m | 4.101 | 4.101 (0) | 187/220 | 0 |
| `vehicle.location_z` | m | 0.736 | 0.728 (0) | 200/220 | 0 |
| `vehicle.rotation_pitch` | deg | 1.223 | 1.223 (3) | 200/220 | 0 |
| `vehicle.rotation_yaw` | deg | 4.679 | 4.679 (0) | 185/220 | 0 |
| `vehicle.rotation_roll` | deg | 0.213 | 0.213 (8) | 200/220 | 0 |
| `vehicle.velocity_x` | m/s | 1.781 | 1.781 (0) | 200/220 | 0 |
| `vehicle.velocity_y` | m/s | 2.506 | 2.386 (0) | 200/220 | 0 |
| `vehicle.velocity_z` | m/s | 3.220 | 3.220 (5) | 200/220 | 0 |
| `vehicle.angular_velocity_x` | deg/s | 11.206 | 8.391 (1) | 200/220 | 0 |
| `vehicle.angular_velocity_y` | deg/s | 3.653 | 2.769 (0) | 200/220 | 0 |
| `vehicle.angular_velocity_z` | deg/s | 11.784 | 11.712 (0) | 200/220 | 0 |
| `vehicle.acceleration_x` | m/s^2 | 10.591 | 6.455 (7) | 200/220 | 0 |
| `vehicle.acceleration_y` | m/s^2 | 26.700 | 26.700 (0) | 200/220 | 0 |
| `vehicle.acceleration_z` | m/s^2 | 26.659 | 16.030 (3) | 200/220 | 0 |
| `vehicle.control_throttle` | normalized | 0.85 | 0.85 (0) | 54/220 | 0 |
| `vehicle.control_steer` | normalized | 0.116 | 0.067 (0) | 145/220 | 0 |
| `vehicle.control_brake` | normalized | 0.7 | 0.7 (0) | 26/220 | 0 |
| `vehicle.control_gear` | index | 3 | 3 (7) | 24/220 | 0 |
| `walker.location_x` | m | 0.734 | 0.734 (0) | 80/88 | 0 |
| `walker.location_y` | m | 0.727 | 0.727 (0) | 80/88 | 0 |
| `walker.rotation_yaw` | deg | 4.579 | 4.579 (0) | 32/88 | 0 |
| `walker.control_speed` | m/s | 0.0088 | 0.0053 (0) | 35/88 | 0 |
| `walker.control_direction_x` | unit vector | 0.077 | 0.077 (0) | 44/88 | 0 |
| `walker.control_direction_y` | unit vector | 0.022 | 0.022 (0) | 49/88 | 0 |

`time` in pass 2 was 0.0 on every row: the adapter's reset took the
last tick's elapsed time as its origin, and without a tick the snapshot
time did not move.

Every other key kept one value per entity through pass 1 and compared
equal in pass 2: all region keys; every traffic-light key except
`elapsed_time` (including `state`, which did not change in these 0.5 s);
vehicle `class`, `track_state`, `extent_*`, `speed_limit`,
`is_at_traffic_light`, `control_hand_brake` and `control_reverse`;
walker `class`, `track_state`, `extent_*`, `location_z`,
`rotation_pitch`, `rotation_roll`, `control_direction_z`,
`control_jump`, and every walker velocity, angular velocity and
acceleration key (see "Walker velocities read zero" below); and the
observation's `component_tags` and `extraction`, both empty.

The adapter counters differed by construction: `timestep` by 1 on every
row of pass 2, and every entity's track age by 1.

### Second client session

Before a tick, the second session listed 0 actors and its snapshot
reported frame 0, while the first listed 59 actors at frame 64. After
one tick, both sessions were at frame 65 and listed the same 59 actor
ids with the same type ids, and every vehicle and walker transform was
identical between them (largest difference 0.0). A reconnect sees the
same actors from its first tick on, and nothing before it.

### Random mode

On steps 0 to 9, the ten ticks the task names, the observation held 20
vehicles and 8 walkers and no entity arrived or left. The set stayed
the same through step 40. At step 41 one vehicle entity left and one
arrived (adapter entity ids 64 and 68): the advancer's churn runs after
the tick of step 40 (`churn_period` = 40) and shows on the next tick.
Before the first tick, `world.get_actors()` listed no vehicle or walker,
although the advancer had already spawned them.

### One tick after restoring row 0

After the setters applied row 0 and one tick ran, vehicles were within
0.401 m of their row-0 locations, against 4.101 m between rows 0 and 10
and 3.700 m from row 10: the teleport reached the server and took effect
at that tick. Walkers were 0.807 m from row 0 and 0.0734 m from row 10,
about one tick of walking from where they had been, so the walkers'
teleport did not hold through the tick. The cause (the walker AI
controller placing its walker again, or something else) was not
investigated. A tier that restores and then ticks would break decision
5, and on this evidence would still not restore walkers.

### Walker velocities read zero

On every row, every walker's velocity, angular velocity and
acceleration were 0.0, while walker locations moved by up to 0.734 m
over the ten ticks (0.5 s). The adapter reads these keys from
`get_velocity()`, `get_angular_velocity()` and `get_acceleration()`, so
in live monitoring the walker `velocity_*` keys report zero for walkers
driven by `controller.ai.walker`. The cause was not investigated.

## Consequence

`state_signature()` stays at the base class's None, so `--mode replay`
of a Specola state recording on `carla` is refused with "it has no
state tier." The docstring of `CarlaAdapter.make_advancer` states the
probe's finding. CARLA's own recorder files (`Client.replay_file`)
remain the native follow-up the plan names; this probe does not bear
on them.

## Earlier runs

Two earlier runs used their own server instances, each started and
stopped as above.

- The first drove the `random` advancer to step 40 only. Its churn then
  fell between pass 1's rows 0 and 1, and one destroyed vehicle and one
  new vehicle showed as presence differences, so the probe now drives to
  step 41. Where it is comparable with the recorded run it agrees:
  walker ranges and one-tick distances equal to every printed digit,
  traffic-light elapsed time 0.5, and pass 2 equal to row 10 on every
  value read from CARLA.
- The second ran the probe before review added type annotations and an
  `isinstance` check on the factory's adapter. Its printed summary and
  its JSON report are byte-identical to the recorded run's (`diff` and
  `cmp`, both exit 0): with seed 0 on a freshly started server, this
  probe's numbers repeated exactly.
