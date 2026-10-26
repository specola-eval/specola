# Fresh-environment install of specola[metaworld], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `9bdba70` of branch `adapter-parity`
plus the new `metaworld` extra in `pyproject.toml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-metaworld/` in the
`adapter-parity` worktree (not committed).

## Extra

```toml
metaworld = ["metaworld==3.1.1", "mujoco==3.3.0", "gymnasium==1.3.0",
             "pyelftools==0.33; python_version >= '3.10' and sys_platform == 'linux'"]
```

These are the versions of the maintained `metaworld` environment
(Python 3.11.15, read with `importlib.metadata` on 2026-10-13).
metaworld 3.1.1 itself requires `mujoco==3.3.0`, a build the geometry
reader has a custody row for.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-metaworld python=3.11` | ok, Python 3.11.17 |
| 2 | `pip install <repo>[metaworld,web,report]` (non-editable) | ok, first attempt; resolved `metaworld` 3.1.1, `mujoco` 3.3.0, `gymnasium` 1.3.0, `pyelftools` 0.33, `pandas` 3.0.6, `pyarrow` 26.0.0 |
| 3 | `pip check` | `No broken requirements found.` |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 5 | `specola discover --sim metaworld --task pick-place-v3 --queries examples/geometry_queries.yaml --no-pager` | `✓ ready   adapter_dt=0.0125s   spec_dt=0.0125s   spatial_dims=3   entities=111` |
| 6 | `specola run --sim metaworld --task pick-place-v3 --queries ... --specs examples/metaworld_specs.yaml --steps 20 --ui plain --web 0` | `20 steps · 0.875 s simulator time · completion complete`, four specifications UNDETERMINED (windows longer than 20 steps); dashboard at `http://127.0.0.1:32817`, `GET /` returned 200; recording with 20 camera frames, geometry, meshes and poses written; process exited 0.10 s after SIGINT to its process group, printing `Dashboard closed` |

Steps 4 to 6 ran from a scratch directory outside the repository with
`MUJOCO_GL=egl`, `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and
`PYTHONPATH` unset. Step 6 was launched with `setsid --wait` under
`set -m`, so SIGINT was at its default (`SigIgn: 0000000001001000`, no
`0x2` bit). The adapter needed no other environment variable.

## Findings

- The extra installs a working adapter on the first attempt. Without it
  there was no supported install path: the adapter's install hint,
  `pip install metaworld`, would install `metaworld` 3.1.1 with
  `mujoco` 3.3.0 but `gymnasium` 1.4.0 and no `pyelftools`
  (`pip install --dry-run --ignore-installed`), and without
  `pyelftools` the adapter fails at reset as the LIBERO check found.
- The run prints `Web warning: cv2 unavailable, using PIL JPEG encoder
  (slower).` The `web` extra installs Pillow for this case; frames were
  recorded.
- The pin is to the Python version as well as to MuJoCo. The reader's
  native profile for MuJoCo 3.3.0 records the cpython-311 Linux x86_64
  wheel; see the [`dmcontrol` page](fresh-install-dmcontrol-2026-10-13.md)
  for what a different Python does.

## Not fresh

pip resolved against PyPI but took most wheels from the machine's pip
cache. Meta-World's assets ship inside its wheel, so nothing else was
downloaded. The machine has a display (`DISPLAY=:1`) and NVIDIA EGL; the
runs used EGL, and a machine without EGL was not tried.
