# Fresh install of the README quickstart, 2026-10-16

Both parts of the README quickstart, run as written from a fresh clone of
commit `3598756` (branch `brand-ui`), on the GPU workstation. The shell
started from `env -i` with only the standard `PATH`, and `HOME` pointed at
an empty directory, so LIBERO's first-run prompt and its asset download ran
as they do for a new user. Two things were reused from the machine: pip's
download cache, so the install times below are not cold-network times, and
conda's settings. The LIBERO environment was created with
`conda create -p <dir>` instead of `-n specola-libero`, to keep it out of
the machine's environment list. Logs were kept in a temporary directory
and are not committed.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `python3 -m venv .venv`, `pip install ".[web]"` (Python 3.10.12) | ok |
| 2 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS; steps 1 and 2 took 10.2 s |
| 3 | `conda create python=3.12` | ok, Python 3.12.15, 5 s |
| 4 | `pip install mujoco==3.8.1 robosuite==1.4.0 hf-libero==0.1.4` | ok, 47 s; installs `torch` 2.14.1 |
| 5 | `pip install ".[libero,web]"` | ok, 3 s |
| 6 | `specola run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --specs examples/libero_goal_00_demo_specs.yaml --mode replay --demo examples/libero_goal_00_demo.hdf5 --web 8193`, with `MUJOCO_GL=egl`, `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` set and `n` on standard input | `138 steps · 7.100 s simulator time · completion complete`, five specifications PASS, the goal recovering at step 128; `GET /` returned 200; 31 s to the summary, the asset download included |

The environment from steps 3 to 5 takes 7.4 GB on disk.

## Findings

- **First-run prompt.** On its first import, with no `~/.libero/config.yaml`,
  hf-libero 0.1.4 asks `Do you want to specify a custom path for the
  dataset folder? (Y/N)` and reads the answer from standard input
  (`libero/libero/__init__.py:100-106`). Answering `n` writes the default
  config and the run continues. Not run here: with standard input closed,
  Python's `input()` raises `EOFError`, so a script needs to supply the
  answer.
- **Asset download.** After the prompt, hf-libero downloaded 586 files,
  409 MB, from the Hugging Face Hub into `~/.cache/libero/assets` in about
  10 s, unauthenticated. The 2026-10-07 install check found the assets
  already present and could not check this step.
- **Expected warnings.** robosuite warns that no private macro file exists,
  `huggingface_hub` warns that `local_dir_use_symlinks` is deprecated, and
  the dashboard reports `hardware h264 encoder unavailable; recording
  camera frames as JPEG images.` None stopped the run.
