# Fresh-environment install of specola[robocasa], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `2fa5ee7` of branch `adapter-parity`
plus the `robocasa` extra change recorded below. Logs:
`results/parity-audit-2026-10-02/logs/fresh-robocasa/` in the
`adapter-parity` worktree (not committed).

## Where RoboCasa comes from

RoboCasa is not on PyPI: `pip index versions robocasa` printed
`ERROR: No matching distribution found for robocasa`. Its README installs
it from a GitHub checkout with `pip install -e .`, after robosuite from
the `master` branch of its own checkout.

robosuite is on PyPI, and its newest release there is 1.5.2, the version
the `master` checkout also reports. They are not the same code. The
checkout is at `master` commit `5ce6643`, not at the `v1.5.2` tag
(`824ac14`). Against the PyPI 1.5.2 wheel, 15 files under its
`robosuite/` tree differ in content, among them
`environments/robot_env.py`,
`environments/manipulation/manipulation_env.py` and
`controllers/parts/mobile_base/joint_vel.py`, and 6 paths exist only in
the checkout. Their requirements differ too: the wheel requires
`mujoco>=3.3.0` and `mink==0.0.5`, the checkout `mujoco>=3.3.0,<3.10`
with `mink` as an optional extra. A pin of `robosuite==1.5.2` would
install the PyPI wheel, not what RoboCasa's README asks for, so the
extra does not name robosuite.

Both checkouts' `setup.py` files match upstream at their commits
(`robosuite` `5ce6643`, `robocasa` `921c9a5`; compared against
`raw.githubusercontent.com`).

## Extra

Before:

```toml
robocasa = ["robosuite", "robocasa", "h5py",
            "pyelftools==0.33; python_version >= '3.10' and sys_platform == 'linux'"]
```

After:

```toml
robocasa = ["h5py", "mujoco==3.3.1",
            "pyelftools==0.33; python_version >= '3.10' and sys_platform == 'linux'"]
```

The extra now carries only what Specola adds beside RoboCasa. RoboCasa
1.0.1 pins `mujoco==3.3.1` in its `setup.py` and asserts it at import;
3.3.1 is a build the geometry reader has a custody row for, and its
native profile records the cpython-311 Linux x86_64 wheel.

The maintained `robocasa` environment (Python 3.11.15, read with
`importlib.metadata` on 2026-10-13) has RoboCasa 1.0.1 and robosuite
1.5.2 as editable installs of `simulators/robocasa` and
`simulators/robosuite`, MuJoCo 3.3.1, `h5py` 3.16.0, `pyelftools` 0.33
and NumPy 2.2.5.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-robocasa python=3.11` | ok, Python 3.11.17 |
| 2 | `pip install --dry-run <repo>[robocasa]` with the old extra, before RoboCasa | **failed**, exit 1: `ERROR: No matching distribution found for robocasa; extra == "robocasa"`. pip had already chosen `robosuite` 1.5.2 from PyPI |
| 3 | `pip install -e simulators/robosuite` | ok; robosuite 1.5.2 (editable), `mujoco` 3.9.0 |
| 4 | `pip install -e simulators/robocasa` | ok; RoboCasa 1.0.1 (editable), `mujoco` downgraded to 3.3.1, `numpy` to 2.2.5, plus `lerobot` 0.3.3 and `torch` 2.7.1 (73 s) |
| 5 | `python -m robocasa.scripts.setup_macros` | the shared checkout already had `macros_private.py`; the script asked to overwrite it and was answered `n` |
| 6 | `pip install <repo>[robocasa,web,report]` (non-editable, new extra) | ok; `h5py` 3.16.0 and `mujoco` 3.3.1 already satisfied, `pyelftools` 0.33 installed |
| 7 | `pip check` | `No broken requirements found.` |
| 8 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 9 | `specola discover --sim robocasa --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml --no-pager` | `✓ ready   adapter_dt=0.05s   spec_dt=0.05s   spatial_dims=3   entities=738160`, in 98 s |
| 10 | `specola run --sim robocasa --task PickPlaceCounterToCabinet --queries ... --specs examples/robocasa_specs.yaml --steps 5 --ui plain --web 0` | `5 steps · 0.750 s simulator time · completion complete`; four specifications UNDETERMINED (windows longer than 5 steps); dashboard at `http://127.0.0.1:33173`, `GET /` returned 200; recording with 5 camera frames, geometry, meshes and poses written; process exited 0.41 s after SIGINT to its process group with status 0 |

Steps 8 to 10 ran from a scratch directory outside the repository with
`MUJOCO_GL=egl`, `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and
`PYTHONPATH` unset. Step 10 was launched with `setsid --wait` under
`set -m`, so SIGINT was at its default (`SigIgn: 0000000001001000`, no
`0x2` bit). The adapter needed no other environment variable.

## Findings

- With the new extra and RoboCasa's README order, the install works on
  the first attempt. The old extra could not work as an install path:
  before RoboCasa is installed it fails on `robocasa`, and after it, the
  bare names are satisfied by the editable installs and add nothing.
- The adapter's install hint said `pip install -e '.[robocasa]'`, which
  no longer installs RoboCasa. It now names the checkouts and points to
  the install guide.
- The scene changes on every construction. The registry factory passes
  no seed to `robosuite.make` and reads no seed variable (it reads
  `ROBOCASA_TASK`, `ROBOCASA_ROBOT` and `ROBOCASA_GENERATIVE_TEXTURES`),
  and RoboCasa draws the kitchen layout, style and objects from the
  unseeded environment random generator. Step 9 placed an orange and
  step 10 a baguette, and the counts below vary with the kitchen.
- `discover` lists every geometry pair as an entity: 738,160 entities,
  735,497 of them `geom_pair`, in 98 s. A `discover` in the maintained
  environment the same day, on another kitchen, listed 1,541,671
  entities in 218 s. The `run --web` in step 10 reported 513,392
  entities at readiness and took 203 s from launch to the end of the
  episode. Without `--web`, `run` reads only the pairs the
  specifications can bind: eleven runs of the step 10 command without
  `--web` (one here, ten in the maintained environment with this
  branch's source) reported 1,184 to 3,000 entities and took 8.8 to
  35.5 s. Whether the dashboard needs every pair was not examined.
- Both environments print robosuite's warning that it cannot import
  `robosuite_models`. The fresh one also prints that robosuite cannot
  load the `mink` whole-body IK (the maintained environment has `mink`
  0.0.5) and that `mimicgen` is not installed. None of the three
  concerns the `PandaOmron` robot this adapter uses.
- `Web: hardware h264 encoder unavailable; recording camera frames as
  JPEG images.` Frames were recorded.

## Not fresh

The robosuite and RoboCasa checkouts were the existing ones under
`simulators/`, installed editable as RoboCasa's README says, so the
kitchen assets (about 10 GB, `download_kitchen_assets`) and both
`macros_private.py` files were already in place and the download was not
exercised. The README creates its environment from `conda-forge`
(`conda create -c conda-forge`); this check used the default channel.
pip resolved against PyPI but took most wheels from the machine's pip
cache. The machine has a display (`DISPLAY=:1`) and NVIDIA EGL; the
runs used EGL, and a machine without EGL was not tried.
