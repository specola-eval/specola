# Fresh-environment install of specola[pybullet], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `70d7993` of branch `adapter-parity`
plus the new `pybullet` extra in `pyproject.toml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-pybullet/` in the
`adapter-parity` worktree (not committed).

## Extra

```toml
pybullet = ["pybullet==3.2.7"]
```

This is the version of the maintained `pybullet` environment, read with
`importlib.metadata` on 2026-10-13: PyBullet 3.2.7 on Python 3.10.20.
That environment runs Python 3.10, not 3.11, so the check used 3.10.
PyBullet's wheel carries the engine and the example models
(`pybullet_data`) the task below loads; the adapter imports nothing else
outside Specola's core dependencies.

PyPI has no Python 3.12 wheel for PyBullet 3.2.7.
`pip download --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64 --no-deps pybullet==3.2.7`
failed with `ERROR: No matching distribution found for pybullet==3.2.7`,
and the same command for 3.11 downloaded
`pybullet-3.2.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`.
On 3.12, pip would build PyBullet from its source distribution; that
build was not tried.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-pybullet python=3.10` | ok, Python 3.10.22 |
| 2 | `pip install <repo>[pybullet,web,report]` (non-editable) | ok, first attempt; resolved `pybullet` 3.2.7 (cp310 wheel), `numpy` 2.2.6, `pandas` 2.3.3, `pyarrow` 25.0.1, `Pillow` 12.3.0 |
| 3 | `pip check` | `No broken requirements found.` |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 5 | `specola discover --sim pybullet --task "plane.urdf@fixed,kuka_iiwa/model.urdf@fixed,cube_small.urdf" --no-pager` | `✓ ready   adapter_dt=0.004166666666666667s   spec_dt=0.004166666666666667s   spatial_dims=3   entities=3`, 84 scalar properties |
| 6 | `specola check --sim pybullet --task ... examples/pybullet_specs.yaml` | `✓ all specs compile and bind or wait` |
| 7 | `specola run --sim pybullet --task ... --specs examples/pybullet_specs.yaml --mode hold --steps 20 --ui plain --web 0` | `20 steps · 0.083 s simulator time · completion complete`; `arm_eventually_settles` PASS, `joints_stay_slow` and `cube_stays_up` UNDETERMINED (20 of 480 steps); dashboard at `http://127.0.0.1:34613`, `GET /` returned 200; recording with 20 camera frames and snapshots written; process exited 0.10 s after SIGINT to its process group with status 0 |
| 8 | step 7 with `PYBULLET_EGL=1` | the same verdicts; PyBullet printed `Loaded EGL 1.5 after reload.`; 20 camera frames; the episode ended 1.6 s after launch against 3.8 s in step 7; exit 0.10 s after SIGINT, status 0 |
| 9 | `specola run --sim pybullet --task ... --specs ... --steps 20 --ui plain` (no `--mode`) | refused, exit 1: `--mode random is not supported by the pybullet adapter: a URDF scene declares no action space to sample and there is no replay path; use --mode hold to step physics under no new command.` |
| 10 | step 9 with `--mode hold --render` | refused, exit 1: `ValueError: --render is not supported by the pybullet adapter: it has no live viewer` |

Steps 4 to 10 ran from a scratch directory outside the repository with
`PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and `PYTHONPATH`
unset. Steps 7 and 8 were launched with `setsid --wait` under `set -m`,
so SIGINT was at its default (`SigIgn: 0000000001001000`, no `0x2` bit).
The adapter needed no environment variable; `PYBULLET_EGL=1` is an
optional switch to PyBullet's EGL renderer for camera frames.

## Findings

- The extra installs a working adapter on the first attempt. The
  adapter's install hint, `pip install pybullet`, resolves to the same
  3.2.7 today; the extra pins it.
- `run` defaults to `--mode random`, which this adapter refuses, so
  every documented `run` command for PyBullet carries `--mode hold`.
- The dashboard address is printed a moment before the server accepts
  connections. In a first run of step 7, a single fetch made as soon as
  the address appeared got no connection (`curl` code `000`); the run
  recorded in the table retried and got 200 on the second attempt,
  0.25 s later. A person opening the link sees no difference.
- `Web warning: cv2 unavailable, using PIL JPEG encoder (slower).` The
  `web` extra installs Pillow for this case; frames were recorded.

## Not fresh

pip resolved against PyPI but took most wheels from the machine's pip
cache. The models ship inside PyBullet's wheel, so nothing else was
downloaded. The machine has a display (`DISPLAY=:1`) and NVIDIA EGL;
step 8 used it, and a machine without EGL was not tried.
