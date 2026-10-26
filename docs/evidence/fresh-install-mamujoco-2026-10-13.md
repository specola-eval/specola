# Fresh-environment install of specola[mamujoco], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `9bdba70` of branch `adapter-parity`
plus the new `mamujoco` extra in `pyproject.toml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-mamujoco/` in the
`adapter-parity` worktree (not committed).

## Extra

```toml
mamujoco = ["gymnasium-robotics==1.4.2", "mujoco==3.10.0", "gymnasium==1.3.0",
            "pyelftools==0.33; python_version >= '3.10' and sys_platform == 'linux'"]
```

These are the versions of the maintained `mamujoco` environment
(Python 3.11.15, read with `importlib.metadata` on 2026-10-13).
gymnasium-robotics 1.4.2 accepts any `mujoco>=2.2.0`; the extra pins
3.10.0, the reviewed build that environment runs. gymnasium-robotics
1.4.2 declares no `mamujoco` extra (its `Provides-Extra` lines are
`testing`, `mujoco-py` and `mujoco-py-original`), and PettingZoo is one
of its base requirements, so the extra names the package without one.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-mamujoco python=3.11` | ok, Python 3.11.17 |
| 2 | `pip install <repo>[mamujoco,web,report]` (non-editable) | ok, first attempt; resolved `gymnasium-robotics` 1.4.2, `mujoco` 3.10.0, `gymnasium` 1.3.0, `PettingZoo` 1.27.0, `pyelftools` 0.33 |
| 3 | `pip check` | `No broken requirements found.` |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 5 | `specola discover --sim mamujoco --task HalfCheetah:2x3 --queries examples/geometry_queries.yaml --no-pager` | `✓ ready   adapter_dt=0.05s   spec_dt=0.05s   spatial_dims=3   entities=23` |
| 6 | `specola check --sim mamujoco --task HalfCheetah:2x3 --queries ... examples/mamujoco_specs.yaml` | `✓ all specs compile and bind or wait` |
| 7 | `specola run --sim mamujoco --task HalfCheetah:2x3 --queries ... --specs examples/mamujoco_specs.yaml --steps 20 --ui plain --web 0` | `20 steps · 1.000 s simulator time · completion complete`; `torso_stays_upright` PASS (`FINAL +0.981`); dashboard at `http://127.0.0.1:37495`, `GET /` returned 200; recording with 20 camera frames written; process exited 0.10 s after SIGINT to its process group with status 0 |

`examples/mamujoco_specs.yaml` is new. There was no example for this
adapter, and the specification the parity audit generated
(`body_*[cacc_world_0]`) binds a `body` class that today's `discover`
does not report, so the example was written from the step 5 vocabulary:
the torso's orientation component that turns negative when the cheetah
flips, which the environment does not terminate on.

Steps 4 to 7 ran from a scratch directory outside the repository with
`MUJOCO_GL=egl`, `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and
`PYTHONPATH` unset. Step 7 was launched with `setsid --wait` under
`set -m`, so SIGINT was at its default. The adapter needed no other
environment variable.

## Findings

- The extra installs a working adapter on the first attempt. The
  adapter's install hint, `pip install "gymnasium-robotics[mamujoco]"`,
  names an extra gymnasium-robotics 1.4.2 does not have. A
  `pip install --dry-run --ignore-installed` of that hint printed
  `WARNING: gymnasium-robotics 1.4.2 does not provide the extra
  'mamujoco'` and would install `mujoco` 3.15.0, a build the reader
  refuses, and `gymnasium` 1.4.0, without `pyelftools`. Not changed here
  (source edits were out of scope for this check).
- gymnasium-robotics prints a notice about the Adroit hand environments'
  reward functions at import (`AdroitHandRelocateDense-v1, ... use v2 in
  gymnasium-robotics>=1.4.3`). It concerns environments this adapter does
  not load.
- The fresh environment resolved PettingZoo 1.27.0; the maintained
  `mamujoco` environment has 1.26.1. The adapter ran on 1.27.0.
- The Python 3.11 requirement recorded on the
  [`dmcontrol` page](fresh-install-dmcontrol-2026-10-13.md) applies here
  too: the reader's MuJoCo 3.10.0 profile records the cpython-311 wheel.
- `Web warning: cv2 unavailable, using PIL JPEG encoder (slower).`, as
  on the other MuJoCo-family checks.

## Not fresh

pip resolved against PyPI but took most wheels from the machine's pip
cache (PettingZoo 1.27.0 was downloaded). The MaMuJoCo scenes ship
inside the gymnasium and gymnasium-robotics wheels. The machine has a
display (`DISPLAY=:1`) and NVIDIA EGL; the runs used EGL, and a machine
without EGL was not tried.
