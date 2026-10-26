# README screenshot capture

Regenerate the four README images from the checked-in capture scripts:
`capture_tui_svgs.py` exports the discover and write-specs Textual screens as
SVG, while `capture_dashboard.py` drives headless Chrome over `specola view`
for the two dashboard JPEGs. Run each from the RoboCasa environment:

```bash
PYTHONPATH=/absolute/path/to/reviewed/specola/src \
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
NUMBA_CACHE_DIR=/absolute/path/to/numba-cache \
MPLCONFIGDIR=/absolute/path/to/matplotlib-cache \
  ~/miniconda3/envs/robocasa/bin/python \
  docs/images/capture/capture_tui_svgs.py --out-dir docs/images

PYTHONPATH=/absolute/path/to/reviewed/specola/src \
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
NUMBA_CACHE_DIR=/absolute/path/to/numba-cache \
MPLCONFIGDIR=/absolute/path/to/matplotlib-cache \
  ~/miniconda3/envs/robocasa/bin/python \
  docs/images/capture/capture_dashboard.py \
  --recording results/20260731_234500_robocasa_PickPlaceCounterToCabinet/recording_000 \
  --out-dir docs/images
```

The browser captures are taken at step 900 of that completed 2,000-step
RoboCasa random-policy recording. Its manifest identifies six camera streams.
The inputs used for the 2026-08-10 refresh are pinned by SHA-256:

- `manifest.json`: `5d85a4b1d72ac95c21d58bbbe5a9962af1c207dde9d86a16711aa5d3e5463f18`
- `snapshots.jsonl.gz`: `64189871609a19f637dc2e6b3102708a9e71d274308ec7f68251fc0cbbac6008`

The TUI images were recaptured on 2026-08-11 after the examples moved from
legacy `G`/`F` aliases to the canonical causal `historically`/`once`
spellings:

- `discover-tui.svg`: `bff1a382d8a8ecc0170ce7db28f671b1d7729ebaf888c970d9fcfe9b8fc892fa`
- `write-specs-tui.svg`: `184f07ac4d8e631d639cda03dea2442b78dd7d4e4c673e32a8f6e1ca89d57610`

The scripts own viewport, seek position, and pictured UI state so the README
does not depend on manually staged screenshots.
