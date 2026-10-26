# Install Specola

Specola installs from its repository; it is not on PyPI. You install the
core once to try the mock scene, then install it again into each
simulator's environment with that simulator's extra.

## Install the core and run the quickstart

Specola itself needs Python 3.8 or later. The simulator adapters need a
specific Python version, listed per adapter below, so create the
environment for the simulator you plan to use.

```bash
git clone https://github.com/specola-eval/specola.git
cd specola
conda create -n specola python=3.11 -y
conda activate specola
pip install ".[web,report]"
specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain
```

The run summary contains this line when the install works:

```text
150 steps · 7.500 s simulator time · completion complete
```

The extras add optional features to any environment:

| Extra | Adds |
|---|---|
| `web` | the live and replay browser dashboard (`--web`, `specola view`); needs Python 3.9 or later |
| `report` | Parquet result files and `specola report`; without it, results are CSV |
| `rtamt` | the RTAMT monitor backend |
| `dev` | pytest, Hypothesis, RTAMT, the dashboard and report dependencies, and h5py |

Use `pip install -e ".[...]"` instead if you edit Specola's source.

Specola runs on Linux x86_64. The install builds one optional C
extension (`specola._native_tls`, used by the guarded native
observation) when a C compiler is present; without one the install
still succeeds and that bridge is absent.

## Install a simulator adapter

Give each simulator its own conda environment, because simulators pin
conflicting dependencies. In that environment, install the simulator's
own prerequisites first, then Specola with the simulator's extra:

```bash
pip install ".[<extra>,web,report]"
```

Check which adapters find their simulator packages in the environment,
and which extras are installed:

```bash
specola doctor
```

Each adapter shows `packages found`, or names the simulator packages it
cannot find; in a terminal the install hint for each missing simulator
follows. The check locates packages without importing them, so
`packages found` does not prove the simulator starts; the requirements
below still apply.

Every MuJoCo-family adapter (`metaworld`, `dmcontrol`, `mamujoco`,
`libero`, `robocasa`) has three requirements the extra cannot meet for
you:

- **Linux on x86_64.** The geometry reader checks the installed MuJoCo
  build against the builds Specola has reviewed, and those are Linux
  x86_64 wheels.
- **The Python version in the table.** The reviewed wheels are specific
  to one Python version. On another version the install succeeds and
  the adapter stops at its first reset with
  `NativeImageError: MuJoCo distribution native image population differs`.
- **A queries file.** Pass `--queries examples/geometry_queries.yaml` to
  `discover`, `check` and `run`. The file sets the cutoff for geometry
  distances, which Specola does not choose for you.

| Adapter | Extra | Python | Simulator versions verified | MuJoCo | Fresh-install check |
|---|---|---|---|---|---|
| `metaworld` | `metaworld` | 3.11 | Meta-World 3.1.1, Gymnasium 1.3.0 | 3.3.0 | [2026-10-13](evidence/fresh-install-metaworld-2026-10-13.md) |
| `dmcontrol` | `dmcontrol` | 3.11 | dm_control 1.0.43 | 3.10.0 | [2026-10-13](evidence/fresh-install-dmcontrol-2026-10-13.md) |
| `mamujoco` | `mamujoco` | 3.11 | Gymnasium-Robotics 1.4.2, Gymnasium 1.3.0 | 3.10.0 | [2026-10-13](evidence/fresh-install-mamujoco-2026-10-13.md) |
| `libero` | `libero` | 3.12 | `hf-libero` 0.1.4, robosuite 1.4.0 | 3.8.1 | [2026-10-07](evidence/fresh-install-2026-10-07.md) |
| `robodojo` | `robodojo` | 3.11 (RoboDojo's environment) | RoboDojo `726e9aa` (2026-09-19), bundled Isaac Lab 0.54.3, Isaac Sim 5.1.0, torch 2.7.0+cu128 | none | [2026-10-07](evidence/fresh-install-robodojo-2026-10-07.md), into a copy of an existing RoboDojo environment |
| `robocasa` | `robocasa` | 3.11 | RoboCasa 1.0.1 (checkout `921c9a5`), robosuite `master` checkout `5ce6643` (reports 1.5.2) | 3.3.1 | [2026-10-13](evidence/fresh-install-robocasa-2026-10-13.md), with the robosuite and RoboCasa checkouts and assets already on the machine |
| `pybullet` | `pybullet` | 3.10 | PyBullet 3.2.7 | none | [2026-10-13](evidence/fresh-install-pybullet-2026-10-13.md) |
| `carla` | `carla` | 3.12 | CARLA 0.9.16 client and server | none | [2026-10-13](evidence/fresh-install-carla-2026-10-13.md), with the CARLA 0.9.16 server already on the machine |
| `isaaclab` | `isaaclab` | 3.12 (Isaac Sim's) | Isaac Lab 6.1.17, `isaaclab_tasks` 1.10.9, Isaac Sim 6.0.1, torch 2.10.0+cu128 | none | [2026-10-13](evidence/fresh-install-isaaclab-2026-10-13.md), into a copy of an existing Isaac Lab environment |

The `pybullet`, `carla` and `isaaclab` adapters read no MuJoCo geometry
and take no queries file.

### Meta-World

The `metaworld` extra installs Meta-World 3.1.1, MuJoCo 3.3.0 and
Gymnasium 1.3.0. Meta-World has no other prerequisite.

```bash
conda create -n metaworld python=3.11 -y
conda activate metaworld
pip install ".[metaworld,web,report]"
export MUJOCO_GL=egl
specola discover --sim metaworld --task pick-place-v3 \
  --queries examples/geometry_queries.yaml
```

Then run the example specifications:

```bash
specola run --sim metaworld --task pick-place-v3 \
  --queries examples/geometry_queries.yaml \
  --specs examples/metaworld_specs.yaml --steps 20
```

### DeepMind Control Suite

The `dmcontrol` extra installs dm_control 1.0.43 and MuJoCo 3.10.0. The
suite's models ship inside the package, so there is nothing else to
download.

```bash
conda create -n dmcontrol python=3.11 -y
conda activate dmcontrol
pip install ".[dmcontrol,web,report]"
export MUJOCO_GL=egl
specola discover --sim dmcontrol --task walker:walk \
  --queries examples/geometry_queries.yaml
```

The example specifications are in `examples/dmcontrol_specs.yaml`.

### Multi-Agent MuJoCo

The `mamujoco` extra installs gymnasium-robotics 1.4.2 (which brings
PettingZoo), Gymnasium 1.3.0 and MuJoCo 3.10.0. Tasks take the form
`scenario:agent_conf`.

```bash
conda create -n mamujoco python=3.11 -y
conda activate mamujoco
pip install ".[mamujoco,web,report]"
export MUJOCO_GL=egl
specola discover --sim mamujoco --task HalfCheetah:2x3 \
  --queries examples/geometry_queries.yaml
```

The example specifications are in `examples/mamujoco_specs.yaml`. At
import, gymnasium-robotics prints a notice about the Adroit hand
environments; it does not concern MaMuJoCo.

### LIBERO

Install LIBERO and its simulator versions before Specola. LIBERO runs
on robosuite 1.4.0 and MuJoCo 3.8.1, whose reviewed wheel is for
Python 3.12.

```bash
conda create -n libero python=3.12 -y
conda activate libero
pip install mujoco==3.8.1 robosuite==1.4.0 hf-libero==0.1.4
pip install ".[libero,web,report]"
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR="$HOME/.cache/specola/numba"
export MPLCONFIGDIR="$HOME/.cache/specola/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
specola discover --sim libero --task libero_goal:0 \
  --queries examples/geometry_queries.yaml
```

`NUMBA_CACHE_DIR` and `MPLCONFIGDIR` must be absolute paths to existing,
writable directories. Without `NUMBA_CACHE_DIR` the robosuite import
can fail with `no locator available`; without `MPLCONFIGDIR`,
`matplotlib` writes its cache to a temporary directory of its own.

On first use, hf-libero asks `Do you want to specify a custom path for
the dataset folder? (Y/N)`; answer `n` to keep its defaults. It then
downloads LIBERO's scene assets, 409 MB, from the Hugging Face Hub into
`~/.cache/libero/assets`. The environment with PyTorch, which LIBERO
depends on, takes 7.4 GB
([check](evidence/fresh-install-libero-quickstart-2026-10-16.md)).

### RoboDojo

RoboDojo brings its own environment. Run RoboDojo's installer
(`scripts/install.sh`), which builds a conda environment named
`RoboDojo` with Python 3.11, Isaac Sim 5.1, Isaac Lab and PyTorch, and
download its assets (`scripts/init_assets.sh`). Then install Specola
inside that environment. The `robodojo` extra adds only the dashboard
and report dependencies.

```bash
conda activate RoboDojo
pip install ".[robodojo]"
export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJO_ROOT=/path/to/RoboDojo
specola discover --sim robodojo --task stack_bowls \
  --queries examples/robodojo_queries.yaml
```

`OMNI_KIT_ACCEPT_EULA=YES` records that you have read and accepted
NVIDIA's Isaac Sim license agreement, so Kit starts without the prompt;
set it only once you have. `ROBODOJO_ROOT` points at your RoboDojo checkout. Run
the monitor with `--mode hold`; the adapter refuses `--mode random`.

The check behind this section installed Specola into a copy of a
working RoboDojo environment. RoboDojo's installer itself was not rerun
from zero.

### RoboCasa

RoboCasa is not on PyPI, and it runs on robosuite's `master` branch,
which differs from the robosuite 1.5.2 release on PyPI. Install both
from their GitHub checkouts as RoboCasa's README describes, then install
Specola. The `robocasa` extra installs neither; it adds `h5py`,
`pyelftools` and MuJoCo 3.3.1, the version RoboCasa 1.0.1 pins and
checks at import.

```bash
conda create -n robocasa python=3.11 -y
conda activate robocasa
git clone https://github.com/ARISE-Initiative/robosuite ../robosuite
pip install -e ../robosuite
git clone https://github.com/robocasa/robocasa ../robocasa
pip install -e ../robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
pip install ".[robocasa,web,report]"
export MUJOCO_GL=egl
specola discover --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml
```

The asset download is about 10 GB. The fresh-install check found the
checkouts and assets already on the machine, so the download has not
been checked.

Each run builds a different kitchen: the adapter passes RoboCasa no
seed, so the layout, style and objects change from run to run.
`discover` and `run --web` read every geometry pair in the kitchen,
from several hundred thousand to over a million, and took 1.5 to 4
minutes each in the check. A `specola run` without `--web` reads only
the pairs your specifications can bind:

```bash
specola run --sim robocasa --task PickPlaceCounterToCabinet \
  --queries examples/geometry_queries.yaml \
  --specs examples/robocasa_specs.yaml --steps 5
```

### PyBullet

The `pybullet` extra installs PyBullet 3.2.7, whose wheel includes the
engine and its example models (`pybullet_data`). PyPI has PyBullet 3.2.7
wheels for Python 3.11 and earlier only; on Python 3.12, pip builds it
from source, which the check did not try.

```bash
conda create -n pybullet python=3.10 -y
conda activate pybullet
pip install ".[pybullet,web,report]"
specola discover --sim pybullet \
  --task "plane.urdf@fixed,kuka_iiwa/model.urdf@fixed,cube_small.urdf"
```

`--task` is a comma-separated list of URDF files, found on disk or in
`pybullet_data`; `@fixed` gives a model a fixed base. Run with
`--mode hold`, which steps the physics with no new command. The adapter
refuses `run`'s default, `--mode random`, because a URDF scene declares
no actions to sample, and it refuses `--render`, because it has no live
viewer:

```bash
specola run --sim pybullet \
  --task "plane.urdf@fixed,kuka_iiwa/model.urdf@fixed,cube_small.urdf" \
  --specs examples/pybullet_specs.yaml --mode hold --steps 20
```

The dashboard's camera frames come from PyBullet's software renderer.
Set `PYBULLET_EGL=1` to render them with EGL on the GPU instead. Leave
it unset where EGL does not work: the adapter falls back to the software
renderer when it finds no `libEGL`, but a broken one can end the process
at the first frame.

### CARLA

CARLA is a server, which simulates the world, and a Python client
library, which Specola uses to connect to it. Install the server first,
as CARLA's
[quick-start guide](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
describes: download `CARLA_0.9.16.tar.gz` (about 8 GB) from the
[0.9.16 release](https://github.com/carla-simulator/carla/releases/tag/0.9.16)
and unpack it (about 19 GB). The guide recommends an NVIDIA GPU with at
least 8 GB of memory. Start the server without a window:

```bash
cd CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound
```

In another terminal, install Specola with the `carla` extra, which
installs the client, `carla` 0.9.16, to match the server. PyPI has that
client for Python 3.10 to 3.12.

```bash
conda create -n carla python=3.12 -y
conda activate carla
pip install ".[carla,web,report]"
specola discover --sim carla
```

Specola connects to `127.0.0.1:2000`, CARLA's default address; set
`SPECOLA_CARLA_HOST` and `SPECOLA_CARLA_PORT` to reach a server
elsewhere. It monitors the town the server has loaded, or loads the one
you name, such as `--task Town04`. While Specola runs, the world steps
in synchronous mode at 20 Hz, and Specola restores the server's previous
settings when it exits. Then run the example specifications:

```bash
specola run --sim carla --specs examples/carla_specs.yaml --steps 20
```

A freshly started server has no vehicles. In `run`'s default
`--mode random`, Specola spawns an autopilot fleet and pedestrians,
replaces fleet members during the run, and removes them when it exits.
`--mode hold` only advances the world, so specifications about vehicles
wait until another client adds traffic. The adapter refuses `--render`.

### Isaac Lab

Install Isaac Sim 6.0.1 and Isaac Lab before Specola, as Isaac Lab's
[pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
describes. Isaac Sim 6 requires Python 3.12 and installs from NVIDIA's
package index
(`pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com`);
Isaac Lab installs from its GitHub checkout with
`./isaaclab.sh --install`. Then install Specola in the same environment:

```bash
conda activate isaaclab
pip install ".[isaaclab,web,report]"
export OMNI_KIT_ACCEPT_EULA=YES
specola discover --sim isaaclab --task Isaac-Forge-PegInsert-Direct-v0
```

The `isaaclab` extra adds the dashboard and report dependencies; torch,
Warp and USD come from the Isaac stack. `OMNI_KIT_ACCEPT_EULA=YES`
accepts the Isaac Sim license agreement without a prompt. `--task` names
a registered Isaac Lab environment. Each command starts Isaac Sim
headless and builds one copy of the task, which took about 10 s per
command in the check. Then run the example specifications:

```bash
specola run --sim isaaclab --task Isaac-Forge-PegInsert-Direct-v0 \
  --specs examples/isaaclab_specs.yaml --mode hold --steps 20
```

`--mode hold` steps the task with zero actions; the default
`--mode random` samples its action space. The adapter refuses
`--render`.

The check behind this section installed Specola into a copy of a working
environment with Isaac Sim 6.0.1, Isaac Lab from its
`release/3.0.0-beta2` branch, and PyTorch 2.10.0 for CUDA 12.8. Isaac
Sim and Isaac Lab were not reinstalled from zero, so neither their
download nor Isaac Sim's first-run extension fetch, which Isaac Lab's
guide says can take upwards of 10 minutes, has been checked.
