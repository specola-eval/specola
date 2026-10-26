# Fresh-environment install of specola[isaaclab], 2026-10-13

Install of Specola into a copy of a working Isaac Lab environment, on the
GPU workstation, at commit `f4f8ac9` (branch `adapter-parity` at
`70d7993` plus a cherry-pick of `43ab145`, Drive Isaac Lab from
specola run) with the new `isaaclab` extra in `pyproject.toml` and the
new `examples/isaaclab_specs.yaml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-isaaclab/` in the
`adapter-parity` worktree (not committed).

## What "fresh" means here

Isaac Sim is a multi-gigabyte install from NVIDIA's package index, and
Isaac Lab installs from its own checkout. The user path is to install
both as Isaac Lab's pip installation guide describes, then Specola with
the `isaaclab` extra inside that environment. So the check cloned the
maintained `isaacsim` environment, removed the editable Specola from the
clone, and installed the repository non-editably. The Isaac Sim and
Isaac Lab stack was not reinstalled from zero.

The maintained environment, read with `importlib.metadata` and
`pip show` on 2026-10-13: Python 3.12.13, `isaacsim` 6.0.1.0 from
`pip install isaacsim[all,extscache]==6.0.1 --extra-index-url https://pypi.nvidia.com`,
`isaaclab` 6.1.17 and `isaaclab_tasks` 1.10.9 as editable installs of
`simulators/IsaacLab/source/` (checkout on branch `release/3.0.0-beta2`
at `6a7acb0`; its `VERSION` file reads 3.0.0), `torch` 2.10.0+cu128,
`warp-lang` 1.13.0, `gymnasium` 1.2.1, `numpy` 2.3.1 and `pyarrow`
23.0.1, with no pandas, so `specola report` is refused there. Isaac
Lab's guide (`docs/source/setup/installation/pip_installation.rst` in
the checkout) requires Python 3.12 for Isaac Sim 6 and installs
`torch==2.10.0` from the CUDA 12.8 index.

## Extra

```toml
isaaclab = ["flask>=3.1,<4", "flask-socketio>=5.5,<6", "python-socketio>=5.16,<6",
            "Pillow>=9", "pandas", "pyarrow"]
```

Like the `robodojo` extra, it carries the dashboard and report
dependencies only; torch, warp and `pxr` come from the Isaac stack, and
the extra names no Isaac package, so it cannot move the stack's pins. A
dry run (`pip install --dry-run <repo>[isaaclab,web,report]`) planned
`pandas-3.0.6 specola-0.2.0` and nothing else.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create --clone isaacsim -n fresh-isaaclab` | ok (31 packages, 250,619 files), with a clone artifact: conda laid its own `packaging` 26.1 and `setuptools` 83.0.0 files over the pip-installed `packaging` 26.0 and `setuptools` 78.1.0 the source runs, leaving `packaging-26.1.dist-info` beside `packaging-26.0.dist-info` and `setuptools-83.0.0-py3.12.egg-info` beside `setuptools-78.1.0.dist-info`. `pip` itself worked. Repaired by removing both packages and their metadata and running `pip install --no-deps packaging==26.0 setuptools==78.1.0`; afterwards `diff -rq` against the source environment found no difference in either package. Not a Specola matter |
| 2 | `pip uninstall specola` | ok (the editable install of the main checkout) |
| 3 | `pip install <repo>[isaaclab,web,report]` (non-editable) | ok; installed `pandas` 3.0.6 and `specola` 0.2.0, nothing else |
| 4 | `pip check` | the same nine conflicts as the source environment (`isaacsim-core` wants `torchaudio`, `torch==2.11.0` and `torchvision==0.26.0`; `isaacsim-kernel` pins `aiohttp`, `click`, `coverage`, `psutil` and `websockets`; `rl-games` wants `psutil<6`); Specola added none |
| 5 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 6 | `specola discover --sim isaaclab --task Isaac-Forge-PegInsert-Direct-v0 --no-pager` | `✓ ready   adapter_dt=0.06666666666666667s   spec_dt=0.06666666666666667s   spatial_dims=3   entities=3`: the arm, the peg (held asset) and the hole (fixed asset), 153 scalar properties; 9.5 s |
| 7 | `specola check --sim isaaclab --task ... examples/isaaclab_specs.yaml` | `✓ all specs compile and bind or wait` |
| 8 | `specola run --sim isaaclab --task ... --specs examples/isaaclab_specs.yaml --mode hold --steps 20 --ui plain --web 0` | `20 steps · 1.333 s simulator time · completion complete`; `peg_stays_in_hand` PASS (worst margin +0.017 m), `arm_joint_speed_within_limit` PASS (worst margin +0.994 rad/s, final +1.548); dashboard at `http://127.0.0.1:45339`, `GET /` returned 200; recording with snapshots, geometry, meshes and poses written; the episode ended 13.1 s after launch; process exited 0.41 s after SIGINT to its process group with status 0 |
| 9 | `specola report results/<step 8 run>` | exit 0; two rows, `sat_rate` 1.0 each |

Steps 5 to 9 ran from a scratch directory outside the repository with
`OMNI_KIT_ACCEPT_EULA=YES`, `PYTHONUNBUFFERED=1` and
`PYTHONNOUSERSITE=1` set and `PYTHONPATH` unset, each Isaac command in
its own session with no other Kit process running before it
(`pgrep -af "[k]it/python"` empty) and none left after it. Step 8 was
launched with `setsid --wait` under `set -m`, so SIGINT was at its
default (`SigIgn: 0000000001001000`, no `0x2` bit). The adapter needed
no environment variable beyond `OMNI_KIT_ACCEPT_EULA`. Each command
starts the Kit app itself, headless, and builds the task; `discover`
and `check` took 9.5 s each from launch to exit.

## Before the cherry-pick

A first pass ran steps 1 to 9 with the package built from `70d7993`.
Every step had the same outcome, but its run never stepped the scene:
the adapter had no drive hook, so `specola run` advanced nothing, and
both specifications passed on a frozen state. In that run's
`snapshots.jsonl.gz` no robustness value changed over the 20 steps; in
the step 8 run above, the peg margin moved from 0.01701 to 0.01699 m
and the joint-speed margin from 0.994 to 1.548 rad/s. Commit `43ab145`
gives the adapter its drive modes (`--mode hold` steps with zero
actions, `--mode random` samples the action space), and the package was
reinstalled from `f4f8ac9` (`pip uninstall`, then step 3 again) before
the run recorded above.

## Example specification

`examples/isaaclab_specs.yaml` is new, written from the step 6
vocabulary. `peg_stays_in_hand` keeps the peg's origin within 5 cm of
the fingertip center; FORGE resets with the peg grasped about 3.4 cm
above its origin. `arm_joint_speed_within_limit` bounds the seven arm
joint speeds by 2.175 rad/s, the lowest of the Panda's limits in the
joint table Isaac Lab prints at startup (124.6 deg/s for joints 1 to 4,
149.5 deg/s for 5 to 7). `check` warns that `object_*` binds one of the
two objects, because only the peg has the peg's key; that is the
intended binding.

## Findings

- The extra installs into the Isaac stack without moving any of its
  pins, and with it `report` works, which the maintained environment
  refuses for lack of pandas.
- The recording lists a `default` camera and holds no frame files, as
  the surface-parity audit's `web` cell for this adapter also records.
- `report` and the run summary use "worst" for different things.
  `report`'s `min_worst_rho` for `arm_joint_speed_within_limit` is
  1.548475, the robustness at the last step taken as the worst across
  bindings (`report.py`); the run summary's WORST, +0.994, is the worst
  over the run's steps.
- Kit prints warnings unrelated to Specola at every start, among them
  `Extensions config 'extension.toml' doesn't exist` for the checkout's
  `isaaclab_visualizers`.
- `conda create --clone` of an environment that pip has changed is not a
  faithful copy, as the RoboDojo check found.

## Not fresh

Isaac Sim, Isaac Lab and their extension cache came from the cloned
environment, so neither the multi-gigabyte download nor Isaac Sim's
first-run extension fetch was exercised. pip took pandas from the
machine's pip cache. The machine has an NVIDIA GPU with 32 GB of
memory.
