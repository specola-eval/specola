<h1 align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.svg"><img src="docs/brand/logo-light.svg" alt="Specola" height="132"></picture></h1>

<p align="center"><em>STL monitoring for simulators.</em></p>

<div align="center">

[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

[Quickstart](#quickstart) ·
[Evaluation loop](#use-specola-in-your-evaluation-loop) ·
[Workflow](#from-simulator-to-verdict) ·
[Dashboard](#inspect-live-and-recorded-runs) ·
[Documentation](docs/)

</div>

Specola connects Signal Temporal Logic specifications to a running robot
simulator. It discovers what the scene exposes, checks your specs against
that vocabulary before a rollout, then monitors one instance per matching
entity and streams a quantitative robustness value for each.

![Specola replaying a RoboCasa run with camera video, per-binding states,
robustness plots, transitions, and a scrubbable timeline](docs/images/dashboard-replay.jpg)

## Why Specola

A success label tells you whether a rollout reached its goal. It doesn't
tell you which object violated a clearance constraint, how close the
rollout came to failing, or whether a property held through a time window.
Specola evaluates those properties during the rollout, and every result
keeps the entity binding that produced it.

- Connect an existing simulator through one adapter.
- Discover the entities and signals the live simulator exposes.
- Check specifications against that vocabulary before running an episode.
- Create and maintain one online monitor per valid entity binding.
- Inspect robustness live, stop on a settled verdict, and replay the
  recorded run later.

One specification can cover every matching object in a scene:

```text
# Keep 30 cm of planar clearance between the robot and every object.
historically[0,5s]( sqrt((robot_*[xpos_mobilebase0_base_0] - object_*[xpos_obj_main_0])^2
            + (robot_*[xpos_mobilebase0_base_1] - object_*[xpos_obj_main_1])^2) - 0.30 >= 0 )
```

## Quickstart

Try Specola on its built-in mock scene, no simulator required:

```bash
git clone https://github.com/specola-eval/specola.git
cd specola
python3 -m venv .venv && source .venv/bin/activate
pip install ".[web]"
specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain
```

The `web` extra adds the [browser dashboard](#inspect-live-and-recorded-runs)
and needs Python 3.9 or later; on Python 3.8, run `pip install .` instead.

```text
Episode summary · Episode complete
150 steps · 7.500 s simulator time · completion complete
Specification results
  joint_settles  PASS
  object_mass_positive  PASS
  …
```

Without `--ui plain`, an interactive terminal shows the Textual dashboard;
`--ui rich` is lighter. Every run ends with a PASS/FAIL/UNDETERMINED
summary and writes one result file per episode under `results/<run_id>/`
(Parquet with the `report` extra, CSV otherwise). See
[Running Specola](docs/running.md) for keybindings, output modes, and the
result format, and [Install Specola](docs/install.md) for the extras and
each simulator's environment.

### Watch a robot demonstration in the dashboard

Replay a LIBERO demonstration that ships with Specola and watch its
monitors in your browser. LIBERO needs Python 3.12 and an environment of
its own, about 7.4 GB with PyTorch:

```bash
conda create -n specola-libero python=3.12 -y
conda activate specola-libero
pip install mujoco==3.8.1 robosuite==1.4.0 hf-libero==0.1.4
pip install ".[libero,web]"
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR="$HOME/.cache/specola/numba"
export MPLCONFIGDIR="$HOME/.cache/specola/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
specola run --sim libero --task libero_goal:0 \
  --queries examples/geometry_queries.yaml \
  --specs examples/libero_goal_00_demo_specs.yaml \
  --mode replay --demo examples/libero_goal_00_demo.hdf5 --web
```

On the first run, LIBERO asks whether to use a custom dataset path
(answer `n`) and downloads its scene assets, 409 MB. Open
<http://127.0.0.1:8090>: the robot opens the cabinet's middle drawer, and
the goal specification turns from violated to satisfied at step 128 of 138.
Press Ctrl-C to stop the dashboard.
[Install Specola](docs/install.md#libero) explains each setting.

## Use Specola in your evaluation loop

Keep your environment, policy, and rollout loop as they are. Your evaluation
code still owns the simulator reset. After that reset, `watch()` finds the
registered adapter by walking the environment's `.env` wrapper chain, resets
Specola's adapter state, and compiles the specifications against the scene the
simulator currently exposes:

```python
from specola import Monitor

monitor = Monitor("my_specs.yaml")                   # parse once
for episode in range(n_episodes):
    obs, info = env.reset()                          # your existing reset
    # watch(env) auto-detects the adapter from the env's wrapper chain;
    # pass an adapter yourself to override: watch(RoboCasaAdapter(env))
    with monitor.watch(env, meta={"task": task.name}) as session:
        for step in range(horizon):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            result = session.step()                  # once per env step
            print(result.rho("spec_name"))           # optional live margin
            # result.raw.evaluations has every per-binding margin
        session.end(success=env.check_success())     # one result file
```

If you interrupt an episode, it is recorded as truncated rather than
thrown away. An environment without a bundled adapter needs one written
first. Session methods, return types, and the lower-level engine API are
documented in [Python API](docs/python-api.md). To aggregate results by
any `meta` key, install the `report` extra and run:

```bash
specola report results/ --by task
```

## From simulator to verdict

The normal workflow is **discover → write → check → run**. This example
uses RoboCasa.

### 1. Install Specola in the simulator environment

Use one conda environment per simulator because their dependencies can
conflict. RoboCasa and robosuite install from their GitHub checkouts
before Specola; [Install Specola](docs/install.md#robocasa) gives the
commands.

### 2. Discover the live vocabulary

`discover` reports the entity classes, component keys, units, and tags that
can appear in a specification. The MuJoCo-family adapters, RoboCasa
among them, need a queries file that sets the cutoff for geometry
distances:

```bash
specola discover --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml
```

In a terminal it opens an explorer you can filter down; for scripting
there is JSON output:

```bash
specola discover --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml \
  --class robot --signal 'qvel_*' --verbose
specola discover --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml --format json
```

![Specola's discovery explorer showing the class tree, signal coverage,
entity-to-key mapping, and a ready selector](docs/images/discover-tui.svg)

### 3. Write and check specifications

Use the keys printed by `discover`:

```yaml
specs:
- name: no_drops
  formula: "historically[0,1s](object_*[xpos_obj_main_2] - 0.7 >= 0)"

- name: settles
  formula: "once[0,10s](robot_*[qvel_mobilebase0_joint_mobile_forward] <= limit)"
  constants: {limit: 0.5}
  backend: auto

# Bind by task identity instead of key name -- this is the drawer the
# episode is actually about, whatever the scene called it:
- name: drawer_stays_shut
  formula: "G[0,10s](fixture_*[qpos_*slidejoint @ fixture_ref:drawer] >= -0.01)"
```

Selectors support globs, explicit sets, set difference, group and tag
filters, `any:` and `all:` quantifiers, and scene-wide `**` expansion.
[Writing specifications](docs/writing-specs.md) documents each form with
worked examples.

You can also author interactively. `specola write-specs --sim robocasa
--queries examples/geometry_queries.yaml` adds draft, validation, and
save panels to the explorer:

![Specola's authoring explorer showing a validated draft with resolved
slots and keys](docs/images/write-specs-tui.svg)

Before running an episode, check every formula against the live simulator:

```bash
specola check --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml my_specs.yaml
```

The check explains selector expansion, resolved keys, and units. An
invalid specification fails here, before any rollout has run.

### 4. Run the monitor

```bash
specola run --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml \
  --specs my_specs.yaml --mode random --steps 300
```

## Inspect live and recorded runs

Install the `web` extra to serve the browser dashboard from the monitoring
process:

```bash
pip install ".[web]"
specola run --sim robocasa --queries examples/geometry_queries.yaml \
  --specs my_specs.yaml --mode random --web
# → Web dashboard: http://127.0.0.1:8090
```

Camera frames encode through `cv2` when the environment has it and fall
back to Pillow otherwise. The one-line encoder warning some
environments print is that fallback, not an error
([details](docs/web-dashboard.md#environment-requirements-and-common-warnings)).

The page shows camera frames, the specification and binding tree,
robustness plots, state transitions, and a scrubbable timeline. Each run
is also recorded under `results/<run_id>/recording_000/`, and the same
page replays a recording later with no simulator running:

```bash
specola view results/<run_id>/recording_000
```

Record every named camera with `--web-cameras all`. Cameras only feed the
video panes (monitoring is identical with one, many, or none), so the
single-camera default is also the fastest live view. The camera grid can
promote any view to the main panel:

![Specola's camera grid showing all six recorded RoboCasa
cameras](docs/images/dashboard-camera-grid.jpg)

See [Web dashboard](docs/web-dashboard.md) for the controls, recording
format, and measured performance.

## Bundled adapters

| `--sim` | Simulator and version | Python | `--task` format | Install extra |
|---|---|---|---|---|
| `libero` | LIBERO (`hf-libero` 0.1.4), robosuite 1.4.0, MuJoCo 3.8.1 | 3.12 | `suite:task_id` (`libero_object:0`) | `libero` |
| `robocasa` | RoboCasa 1.0.1 (`921c9a5`), robosuite `master` (`5ce6643`), MuJoCo 3.3.1 | 3.11 | Task name (`PickPlaceCounterToCabinet`) | `robocasa`, after the two checkouts |
| `metaworld` | Meta-World 3.1.1, Gymnasium 1.3.0, MuJoCo 3.3.0 | 3.11 | Environment name (`pick-place-v3`) | `metaworld` |
| `dmcontrol` | dm_control 1.0.43, MuJoCo 3.10.0 | 3.11 | `domain:task` (`walker:walk`) | `dmcontrol` |
| `mamujoco` | Gymnasium-Robotics 1.4.2 (MaMuJoCo), Gymnasium 1.3.0, MuJoCo 3.10.0 | 3.11 | `scenario:agent_conf` (`HalfCheetah:2x3`) | `mamujoco` |
| `isaaclab` | Isaac Lab 6.1.17 (`isaaclab_tasks` 1.10.9), Isaac Sim 6.0.1 | 3.12 | Gym id (`Isaac-Forge-PegInsert-Direct-v0`) | `isaaclab`, into the Isaac environment |
| `robodojo` | RoboDojo (`726e9aa`, 2026-09-19) with its bundled Isaac Lab 0.54.3, Isaac Sim 5.1.0 | 3.11 | Task name (`stack_bowls`) | `robodojo`, into RoboDojo's environment |
| `pybullet` | PyBullet 3.2.7 | 3.10 | URDF paths (`kuka_iiwa/model.urdf@fixed,cube_small.urdf`) | `pybullet` |
| `carla` | CARLA 0.9.16 client and server (start the server first) | 3.12 | Town name (`Town04`), optional | `carla` |
| `mock` | Internal: built-in fake scene | any | none | none |
| `dataset` | Internal: CSV trajectories, no simulator | any | Path to `.csv` | none |

The versions are the ones each adapter was installed and run with on
2026-10-07 to 2026-10-13 (evidence pages linked from
[Install Specola](docs/install.md)); other versions are untested, and the
MuJoCo-family adapters refuse a MuJoCo build other than 3.3.0, 3.3.1,
3.8.1 or 3.10.0. `mock` and `dataset` are internal adapters for
development and debugging. [Install Specola](docs/install.md) gives the
setup behind each extra. [Surface
parity](docs/surface-parity.md) records which commands, drive modes, and
flags each adapter accepts, as measured.

Drive modes are `random` (the default), `hold` (zero action), and
`replay`. An adapter refuses a mode it does not implement before the run
starts; `pybullet`, `robodojo` and `isaaclab` support `hold` only. Every
run on an adapter with a state tier writes a state recording, and `--mode
replay --demo <file>` replays it (seven adapters today; see [Running
Specola](docs/running.md#replay-a-recorded-run)). Replay also drives
`libero` or `robocasa` from a robomimic demonstration recorded on the same
model, and the other adapters refuse that format.

```bash
specola run --sim libero --task libero_goal:0 \
  --queries examples/geometry_queries.yaml \
  --specs my_specs.yaml --mode replay --demo demo.hdf5
```

`--render` opens the simulator's own viewer, and adapters without one
refuse the flag. On the MuJoCo-family adapters the viewer opens and the
run then fails (a known issue).

## Specification language

Specola evaluates past-time STL over entity references. A specification
template is compiled against the scene discovered at episode start and
expanded into one monitor per matching binding.

- `class_*[component]` binds every eligible entity; `class_3[component]`
  pins entity ID 3.
- `class{A}_*[x]` and `class{B}_*[y]` create distinct same-class slots.
- Predicates support `>= > <= < ==`, SymPy arithmetic such as `sqrt` and
  `Min`, and the Boolean operators `and or not -> <-> xor`.
- `historically[a,b]`, `once[a,b]`, and `since[a,b]` use trailing
  windows; unbounded `historically`, `once`, and `since` start at the
  current contiguous segment's origin. Nothing is predictive.
- `prev`, `rise`, and `fall` express one-step and edge behavior. Bounds
  can be steps (`once[0,100]`) or seconds (`once[0,5s]`).
- Native and optional RTAMT backends cover the same temporal operators.
  Explicit RTAMT rejects `<->`/`xor` over temporal subformulas; native
  supports those combinations. Legacy bounded `G`/`F` aliases remain
  accepted.

The full language is documented in
[Writing specifications](docs/writing-specs.md).

## Library API

For lower-level integration, drive the engine and adapter directly:

```python
from specola import MonitorEngine, reset_adapter

engine = MonitorEngine.load("my_specs.yaml")  # parse once
reset = reset_adapter(adapter)                # episode discovery
engine.start_episode(                         # compile per episode
    dt=adapter.dt,
    capabilities=reset.capabilities,
    catalog=reset.discovered_catalog,
)
result = engine.update(reset.initial_observation)
for evaluation in result.evaluations:
    print(evaluation.spec_name, evaluation.robustness,
          evaluation.status, evaluation.verdict_final)
```

Invalid references fail at episode-start compilation, not mid-run. An absent
entity class is different: the lower-level engine is strict by default, while
the facade and run CLI record it as `not_applicable` rather than silently
dropping the specification. Signatures and return types for both levels are in
[Python API](docs/python-api.md).

## Architecture

Adapters translate simulator state into observations; they do not step the
simulator. The engine parses specification templates once, compiles them
against each episode's discovered scene, and manages the resulting monitor
lifecycles as entities appear and disappear.

## Add a simulator

Implement `SimulatorAdapter` (`dt`, `spatial_dims`, `_reset_impl`, and
`_get_observation_impl`), register a name, and expose the module through
the adapter entry-point group:

```toml
[project.entry-points."specola.adapters"]
mysim = "my_package.my_adapter"
```

You can also load it directly with
`--adapter-module my_package.my_adapter`.

Start with the annotated `examples/custom_mujoco_adapter.py` template. For
MuJoCo-family simulators, subclass `MujocoAdapter` and override the body
rules; `src/specola/adapters/robocasa.py` is a complete example. The
adapter picks its own entity classes; Specola imposes no fixed ontology.
Bundling an adapter inside this repository adds two review gates that
your own module does not need; [Adding an adapter](docs/adding-an-adapter.md)
covers both paths.

## Installation and development

Specola requires Python 3.8 or later and runs on Linux x86_64: the
simulator adapters and the geometry reader's reviewed MuJoCo builds are
Linux only. The install builds one optional C extension when a C
compiler is present and installs without it otherwise. Core dependencies are NumPy, SymPy, PyYAML,
`jsonschema`, Rich, and Textual. The simulator adapters each need a
specific Python version; [Install Specola](docs/install.md) lists the
optional extras, the Python version per adapter, and each simulator's
setup.

To work on Specola itself, install it editable with the `dev` extra and
run the simulator-free tests:

```bash
pip install -e ".[dev]"
pytest
pytest tests/test_binding_parity.py
pytest tests/test_backend_selection.py
```

After changing adapter entry points in `pyproject.toml`, rerun
`pip install -e .` in each environment.

User guides live at the top of [docs/](docs/). Dated design records and
implementation plans live under `docs/superpowers/`; planned work is
tracked in GitHub issues.

## Project status

The API and result schema may still change between versions. Questions, bug reports, and
contributions are welcome through GitHub issues and pull requests.

## License

Specola is available under the [MIT License](LICENSE).
