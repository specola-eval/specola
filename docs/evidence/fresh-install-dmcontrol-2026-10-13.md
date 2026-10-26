# Fresh-environment install of specola[dmcontrol], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `9bdba70` of branch `adapter-parity`
plus the new `dmcontrol` extra in `pyproject.toml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-dmcontrol/` and
`.../fresh-py312-dmcontrol/` in the `adapter-parity` worktree (not
committed).

## Extra

```toml
dmcontrol = ["dm_control==1.0.43", "mujoco==3.10.0",
             "pyelftools==0.33; python_version >= '3.10' and sys_platform == 'linux'"]
```

These are the versions of the maintained `dmcontrol` environment
(Python 3.11.15, read with `importlib.metadata` on 2026-10-13).
dm_control 1.0.43 requires `mujoco>=3.10.0`; 3.10.0 is the newest build
the geometry reader has a custody row for.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-dmcontrol python=3.11` | ok, Python 3.11.17 |
| 2 | `pip install <repo>[dmcontrol,web,report]` (non-editable) | ok, first attempt; resolved `dm_control` 1.0.43, `mujoco` 3.10.0, `pyelftools` 0.33 |
| 3 | `pip check` | `No broken requirements found.` |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 5 | `specola discover --sim dmcontrol --task walker:walk --queries examples/geometry_queries.yaml --no-pager` | `✓ ready   adapter_dt=0.025s   spec_dt=0.025s   spatial_dims=3   entities=22` |
| 6 | `specola check --sim dmcontrol --task walker:walk --queries ... examples/dmcontrol_specs.yaml` | `✓ all specs compile and bind or wait` |
| 7 | `specola run --sim dmcontrol --task walker:walk --queries ... --specs examples/dmcontrol_specs.yaml --steps 20 --ui plain --web 0` | `20 steps · 0.500 s simulator time · completion complete`; `torso_stays_standing` FAIL (`FINAL -0.553`: the torso fell below 1.2 m), `torso_stays_upright` UNDETERMINED (20 of 40 steps); dashboard at `http://127.0.0.1:37385`, `GET /` returned 200; recording with 20 camera frames written; process exited 0.10 s after SIGINT to its process group with status 0 |

`examples/dmcontrol_specs.yaml` is new. There was no example for this
adapter, and the specification the parity audit generated
(`body_*[cacc_world_0]`) binds a `body` class that today's `discover`
does not report, so the example was written from the step 5 vocabulary
instead: the torso height and uprightness terms of dm_control's own
walker reward.

The entity count changes between resets (22, 21 and 14 in three
`discover` and `check` runs): walker resets randomize the joint angles,
and the number of geometry pairs within the 0.05 m query horizon follows
the pose.

Steps 4 to 7 ran from a scratch directory outside the repository with
`MUJOCO_GL=egl`, `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and
`PYTHONPATH` unset. Step 7 was launched with `setsid --wait` under
`set -m`, so SIGINT was at its default. The adapter needed no other
environment variable.

## Python 3.12 check

The reader's native profile for MuJoCo 3.10.0
(`src/specola/data/mujoco-native-3.10.0.json`) lists the cpython-311 Linux
x86_64 extension modules by name. To see what that means for a user, the
same extra was installed into a throwaway `python=3.12` environment
(since removed):

| Step | Command | Result |
|---|---|---|
| 1 | `pip install <repo>[dmcontrol]` | ok; `dm_control` 1.0.43, `mujoco` 3.10.0 (cp312 wheel) |
| 2 | `specola discover --sim dmcontrol --task walker:walk --queries ...` | **failed**, exit 1: `specola.conformance.native_images.NativeImageError: MuJoCo distribution native image population differs` |

The extra cannot refuse a Python version, so the install guide states
it: the MuJoCo-family adapters need Python 3.11 (3.12 for LIBERO's
MuJoCo 3.8.1).

## Findings

- The extra installs a working adapter on the first attempt on
  Python 3.11. The adapter's install hint, `pip install dm_control`,
  would install `dm_control` 1.0.48 and `mujoco` 3.15.0 today
  (`pip install --dry-run --ignore-installed`), a build the reader
  refuses, and no `pyelftools`.
- On Python 3.12 the install succeeds and the adapter fails at its first
  reset. The message names neither the Python version nor the reviewed
  wheels, so a user cannot tell what to change. Not changed here
  (source edits were out of scope for this check).
- At exit, with output redirected to a file, the run printed an
  exception after `Dashboard closed`:
  `AttributeError: '_TeeStream' object has no attribute 'close'`, from
  `absl/logging/__init__.py`, line 1032, inside `logging.shutdown`.
  The `absl` logging handler captured `sys.stderr` while Specola's run log tee
  (`run_monitor.py`, `_TeeStream`) was installed, and at shutdown closes
  any stream that is no longer `sys.stderr` and is not a TTY. The exit
  status was still 0. It does not appear when stderr is a terminal.
  Not changed here.
- `Web warning: cv2 unavailable, using PIL JPEG encoder (slower).`, as
  on the other MuJoCo-family checks.

## Not fresh

pip resolved against PyPI but took most wheels from the machine's pip
cache. dm_control's suite assets ship inside its wheel. The machine has
a display (`DISPLAY=:1`) and NVIDIA EGL; the runs used EGL, and a
machine without EGL was not tried.
