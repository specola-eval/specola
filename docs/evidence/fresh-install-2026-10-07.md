# Fresh-environment install of specola[libero], 2026-10-07

First end-to-end install of Specola from the repository into an
environment that had never had it, on the GPU workstation, at commit
`d4ffd7d` of branch `adapter-parity` plus the `pyproject.toml` change
recorded below. Logs: `results/parity-audit-2026-10-02/logs/fresh-libero/`
(not committed).

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-libero python=3.12` | ok |
| 2 | `pip install mujoco==3.8.1 robosuite==1.4.0 hf-libero==0.1.4` | ok (the versions of the working `libero` env) |
| 3 | `pip install <repo>[libero,web,report]` (non-editable) | ok |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, three specifications PASS |
| 5 | `specola discover --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml` | **failed**: `NativeImageError: native ELF custody requires pyelftools 0.33` |
| 6 | `pyelftools` added to the `libero` and `robocasa` extras; reinstall | ok |
| 5 again | same `discover` | `✓ ready ... entities=10645` |
| 7 | `specola run --sim libero --task libero_goal:0 --queries ... --specs examples/libero_specs.yaml --steps 20 --ui plain --web 0` | `20 steps · 1.000 s simulator time · completion complete`; dashboard served, `GET /` returned 200; recording written |

Every LIBERO command ran with `MUJOCO_GL=egl`, `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR` set, as `CLAUDE.md` requires; the README does not yet say
so, and a user who omits them gets the failures those rules record.

## Findings

- **Missing dependency in the extras (fixed).** The MuJoCo-family
  adapters read the installed MuJoCo image through `pyelftools` at every
  reset. The `libero` and `robocasa` extras did not include it, so the
  adapter installed and then could not start. Both extras now carry the
  `native-extraction` requirement.
- **Buffered dashboard address.** With stdout redirected to a file and
  no `PYTHONUNBUFFERED=1`, the `Web dashboard: http://...` line did not
  appear for the whole run. A script that waits for that line hangs.
  `MISTAKES.md` records the same trap from 2026-09-23; the fix is a
  `flush=True` on that print (`run_monitor.py:1362`), not yet made.
- **No exit on SIGINT to the process alone.** After the run completed
  and the dashboard kept serving, `kill -INT <pid>` left the process
  alive for 30 s; it was killed. The #126 investigation found the
  likely cause in this harness, not Specola: a command started with
  `&` from a non-interactive shell begins with SIGINT ignored, so Python
  installs no handler and `kill -INT` does nothing (the run's log never
  printed "Dashboard closed"). The launch command here was of that
  form. Recorded in `MISTAKES.md`.

## Not fresh

The machine already had `~/.libero/config.yaml`, the LIBERO assets in
`~/.cache/libero/`, and the demonstration datasets. `hf-libero` prints
"Local assets not found. Downloading from HuggingFace Hub..." and then
finds them, so on a truly new machine step 5 would download assets
first; that path and the `config.yaml` creation were not exercised.
