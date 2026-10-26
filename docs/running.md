# Running the monitor

`specola run` drives one episode and streams robustness while it runs.
This page covers terminal controls, run states, and result files. For the
browser interface, see [Web dashboard](web-dashboard.md).

## Choose a terminal UI

The default Textual dashboard groups bindings under their specifications
and shows state, robustness history, and extrema.

| Key | Action |
|---|---|
| Space | expand/collapse a specification |
| Enter | inspect a binding |
| J | show the binding's schema-versioned JSON projection |
| T | show the literal transition history |
| L | show the simulator log |
| `/` | search |
| `f` | filter |
| `s` | sort |
| `q` / Ctrl-C | stop the run |

Collapsed specification rows show state counts. Expanded bindings show
signed, sign-colored `NOW`/`WORST` values; completed results use
`FINAL`/`WORST`. The full numerical trace is the saved Parquet/CSV
artifact, not the screen.

`--ui rich` is a lightweight Rich overview. In a non-interactive
terminal, or with `--ui plain` or `--no-live`, transitions print once as
they happen instead of periodic snapshot blocks. `--steps 0` runs until
Ctrl-C.

## Files written by a run

A completed run ends with a PASS/FAIL/UNDETERMINED summary and writes under
`results/<run_id>/`:

- one Parquet file per episode, or CSV with a metadata sidecar when
  `pyarrow` is unavailable; `specola run --help` shows the row schema
  and a pandas example
- `<episode>_states.npz`: on an adapter with a state tier, the
  simulator state at the reset and after every step (see
  [Replay a recorded run](#replay-a-recorded-run))
- `simulator.log`: simulator output
- `run-error.log`: on failure only, the phase, verbatim error, and
  any result that was safely written
- `recording_000/`: created with `--web` (see
  [web-dashboard.md](web-dashboard.md))

`--output-dir` chooses the result root and `--run-name` the run
directory. Ctrl-C finalizes the samples collected so far as partial
evidence. Results retain every per-step binding value regardless of the
selected UI.

`specola run` resets the adapter once: the monitoring session resets it after
replay setup and compiles your specifications against that reset, and the
`ready` line reports the entities it read.
Adapter guards, replay setup, initial specification compilation and drive-mode
setup finish before the CLI creates the run directory or starts its recorder
and dashboard, so a `--mode` or `--demo` the adapter refuses ends the run
there. A startup rejection reports `Results none` and retains its exception
trace and simulator log in the printed `specola-startup-` temporary directory.
After successful startup, the CLI appends the captured log to `simulator.log`
in the run directory. A failure while copying that log retains the original
startup diagnostics; the destination can contain a partial copy.

`--queries PATH` loads native query arguments from a versioned YAML or JSON
document before the first adapter reset. The same option is available to
`specola discover`, `specola check` and `specola write-specs`. Query arguments belong to the
extraction configuration.
The MuJoCo-family adapters require a population entry for
`mujoco.mj_geomDistance` with a `distmax` cutoff; `examples/geometry_queries.yaml`
is a starting point. Each collision-eligible geom pair then appears as a
`geom_pair` entity carrying `geom_distance_capped` (the native signed
distance, bounded above by the cutoff) and `geom_distance_clipped` (1 at or
beyond the cutoff), tagged with both body names and `robot`, `self_collision`
or `environment`. `specola discover` lists and reads every pair. `specola
check` lists every pair in its slot tables and reads only the pairs whose
values it prints. During `specola run` and a `Monitor` session, the reset
reads no pair: your specifications compile against a catalog that lists every
pair, and each observation after the reset reads only the pairs they can bind.
`geom_pair_*[geom_distance_capped @ robot]` keeps the robot pairs, and a run
whose specifications name no `geom_pair` reads none. The same holds with
`--web`: the dashboard's discovery snapshot describes that reset, so it
lists no pair entity (on a RoboCasa kitchen, reading every pair there cost
about two minutes of startup). Add `population: all` next to `distmax` to
read every pair at every reset and observation, for example to record them
all or to browse them in the dashboard. Keep the cutoff at or below 0.1 m on MuJoCo 3.8.1 and
3.10.0: those builds return exact zeros for some separated pairs at larger
cutoffs. The reader has custody rows for MuJoCo 3.3.0, 3.3.1, 3.8.1 and 3.10.0; another installed version is refused with a message naming the reviewed
ones.

The RoboDojo adapter takes a `robodojo.mesh_pair` population with a
`distmax` cutoff and an optional `robot_links` list of the arm links to
sample; `examples/robodojo_queries.yaml` is a starting point. Each pair of
robots and posed scene objects then appears as a `mesh_pair` entity carrying
`mesh_distance_sampled` and `mesh_distance_clipped`, tagged `a:<label>`,
`b:<label>` and `robot_pair`, `robot` or `environment`. Pairs follow the
same scope as `geom_pair`: `specola discover`, `specola check` and the first
observation of a run list every pair, later observations read only the pairs
your specifications can bind, and `population: all` reads every pair on every
observation. Instances that name pairs by entity label narrow the population
itself: only the named pairs exist, in discovery as well, and each one is read
whatever your specifications bind. Entity IDs follow the population, so take
the IDs for pinned `mesh_pair_<id>` selectors from a `specola discover` run
with the same `--queries` file.

Specola builds each RoboDojo robot with `need_planner: False`, so RoboDojo
skips building its motion planner. Only RoboDojo's evaluation client uses the
planner, to turn end-effector actions into joint targets, and a monitored run
never calls it. Set `ROBODOJO_PLANNER=1` to keep the planner when RoboDojo's
own evaluation loop drives an environment that Specola built.

With `--sim isaaclab`, Specola launches a headless Isaac Sim app before it
builds the environment and closes the app when the command exits, so
`specola discover`, `check` and `run` need no launch script. A Python program
that launched the app itself keeps it, and the adapter builds the environment
in that app. A task whose scene declares a camera needs `ENABLE_CAMERAS=1`,
which Isaac Lab reads at launch. In a non-interactive shell, set
`OMNI_KIT_ACCEPT_EULA=YES`, or the launch waits on the EULA prompt.

## Replay a recorded run

On an adapter with a state tier, every run writes `<episode>_states.npz`
beside the episode's result file, and the run summary lists it under the
`Results` line. `specola run --mode replay --demo <episode>_states.npz`
restores the recorded states on the adapter that recorded them, one per
step and with no physics step in between, and monitors each restored state
as a live run would. A live observation describes one instant: before each
one, the MuJoCo adapters recompute poses, velocities, the mass matrix and
actuator lengths at the current joint positions, and `pybullet` recomputes
its link frames, without collision detection, so contact forces and the
other outputs of a physics step keep the values the last step produced.
Replay refuses a recording made on another simulator,
task, model or engine build, or at another `--dt`; that message names the
`--dt` that matches. `--seed` (default 0) also picks the `robocasa` kitchen,
the `metaworld` task variant and the masses and friction that `isaaclab`
tasks randomise at startup and reset, so replay a recording from these
adapters with the `--seed` it was recorded with; another seed builds another
model, and replay refuses it. The run ends when the recording ends: with
`--steps 0` it completes there, and a positive `--steps` that the recording
cannot fill ends with `completion partial` and a `recording exhausted` line.
`--no-state-recording` stops a run from writing the file and leaves replay
from an existing recording unchanged. Robomimic HDF5 demonstrations remain
the native replay path for `libero` and `robocasa`. As of 2026-10-14 the
MuJoCo adapters (`libero`, `robocasa`, `metaworld`, `dmcontrol` and
`mamujoco`), `pybullet` and `isaaclab` have a state tier, except a
`pybullet` scene with a planar joint, and the [parity
audit](surface-parity.md) replays a recording on each. Every other bundled
adapter writes no state file and refuses a state recording with "it has no
state tier":

- `carla`: CARLA's `get_*` calls return the values of the last tick, so a
  restore without a tick changes nothing the observation reads
  ([probe](evidence/state-replay-carla-2026-10-14.md)).
- `robodojo`: Isaac Lab's scene state holds RoboDojo's two arms, not the
  bowls and fixtures RoboDojo keeps outside the scene
  ([probe](evidence/state-replay-robodojo-2026-10-14.md)).
- `mock` and `dataset` have no simulator state.

## Run and binding states

The run summary uses three specification-level states when no active binding
has a final Boolean verdict:

- `waiting`: the specification compiled, but no entity currently binds.
- `undetermined`: at least one binding exists, but its temporal window has not
  produced a final verdict.
- `not applicable`: the facade or run CLI explicitly skipped an absent entity
  class. This is recorded applicability, not a passing verdict.

Each binding in the live views has one of these states:

- `pending`: active, with no final verdict yet.
- `satisfied` / `violated`: active with a final verdict at this step.
- `suspended`: one or more bound entities are absent.
- `rewarming`: a suspended binding returned and its temporal history is being
  rebuilt from new samples.
- `missing_data`: the entities are present but a required value is missing.
- `error`: evaluation failed; the inspector and `run-error.log` retain the
  verbatim error.

`recovered` is an episode-summary transition: a binding was violated earlier
and later ended satisfied. It is not a separate live binding state.

## Open the simulator viewer

`--render` opens the simulator's own viewer beside the terminal UI
(MuJoCo-family adapters, needs a display). An adapter without a live viewer
refuses the flag at startup, before the CLI creates the run directory. On
the MuJoCo-family adapters the viewer opens and the run then fails
(`NativeImageError: Python namespace population changed`, #129): the
viewer adds something to the namespace the native guard pins, and which
import that is has not been established. Until #129 is fixed, use
`--web` for a live view. From library code, call
`session.adapter.render()` once per step and
`session.adapter.close_render()` when done.
