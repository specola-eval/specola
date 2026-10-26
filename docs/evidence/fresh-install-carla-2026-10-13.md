# Fresh-environment install of specola[carla], 2026-10-13

End-to-end install of Specola into an environment that had never had it,
on the GPU workstation, at commit `70d7993` of branch `adapter-parity`
plus the new `carla` extra in `pyproject.toml` and the new
`examples/carla_specs.yaml`. Logs:
`results/parity-audit-2026-10-02/logs/fresh-carla/` in the
`adapter-parity` worktree (not committed).

## Client and server

CARLA is a server, which simulates the world, and a Python client
library, which Specola imports to connect to it. The client is on PyPI;
the server is not. CARLA's 0.9.16 quick-start guide
(<https://carla.readthedocs.io/en/0.9.16/start_quickstart/>, read on
2026-10-13) has the user download the server package from the GitHub
release and install the client with pip. The release links
`CARLA_0.9.16.tar.gz` (<https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz>),
whose `content-length` was 8,346,095,504 bytes in a `curl -sI` the same
day; unpacked, the server on this machine takes 19 GB (`du -sh`). The
guide asks for about 20 GB of disk space and recommends a GPU
"equivalent to an NVIDIA 2070 or better with at least 8Gb of VRAM" for
the server.

The check used the existing unpacked server at
`simulators/CARLA_0.9.16`; its download was not exercised.

## Extra

```toml
carla = ["carla==0.9.16"]
```

This is the client of the maintained `carla` environment (Python
3.12.13, read with `importlib.metadata` on 2026-10-13), and the version
of the server. PyPI's file list for `carla` 0.9.16 has Linux x86_64
wheels for Python 3.10, 3.11 and 3.12 and no source distribution, and
the server package ships the same three wheels in
`PythonAPI/carla/dist/`. The maintained environment has no pandas or
pyarrow, so `specola report` is refused there; the `report` extra adds
both.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create -n fresh-carla python=3.12` | ok, Python 3.12.15 |
| 2 | `pip install <repo>[carla,web,report]` (non-editable) | ok, first attempt; resolved `carla` 0.9.16 (cp312 wheel), `numpy` 2.5.4, `pandas` 3.0.6, `pyarrow` 26.0.0, `Pillow` 12.3.0 |
| 3 | `pip check` | `No broken requirements found.` |
| 4 | `specola run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete`, four specifications PASS |
| 5 | start the server (below), poll with the fresh client | first poll: `client 0.9.16 server 0.9.16` |
| 6 | `specola discover --sim carla --no-pager` | `✓ ready   adapter_dt=0.05s   spec_dt=0.05s   spatial_dims=3   entities=40`: 25 regions and 15 traffic lights, 447 scalar properties; `vehicle` and `walker` listed as declared dynamic classes, not live at reset |
| 7 | `specola discover --sim carla --task Town04 --no-pager` | Town04 loaded; `entities=97` (54 regions, 43 traffic lights), 2.8 s |
| 8 | `specola check --sim carla examples/carla_specs.yaml`, first draft | **failed**, exit 1: `error: Selector 'vehicle_*[velocity_*]': class 'vehicle' has a declared component schema and a runtime population, so aggregation across entities ('**', or a min/max/sum call) is not known at compilation. Use an any:/all: keyword over one bound entity instead.` |
| 9 | the same, final example | `✓ all specs compile and bind or wait`, with `warning:  no reset-time binding; only declared runtime-populated class slot(s) are unsatisfied (vehicle), so the monitor compiles and waits` |
| 10 | `specola run --sim carla --specs examples/carla_specs.yaml --steps 20 --ui plain --web 0` | `mode=random`; `20 steps · 1.000 s simulator time · completion complete`; `vehicle_within_speed_limit` PASS on each of the 20 vehicles the fleet spawned, worst margin +14.5 km/h; dashboard at `http://127.0.0.1:44741`, `GET /` returned 200; recording with 18 camera frames from `junction_189` and snapshots written; process exited 0.10 s after SIGINT to its process group with status 0 |
| 11 | `specola report results/<step 10 run>` | exit 0; one row, `vehicle_within_speed_limit`, `sat_rate` 1.0, `min_worst_rho` 14.509235 |
| 12 | stop the server (below) | `CARLA group gone`; no `CarlaUE4` process left |

The server was started and stopped as the surface-parity audit plan
(`docs/superpowers/plans/2026-10-02-surface-parity-audit.md`, Task 4
Step 2) describes, from the unpacked package directory:

```bash
setsid sh -c 'echo $$ > "$1"; exec ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound' \
  sh <pgid file> < /dev/null > <server log> 2>&1 &
```

`ps` showed the recorded number as the PGID and SID of `CarlaUE4.sh`
and its `CarlaUE4-Linux-Shipping` child. The shutdown sent SIGINT to
that group, waited for it to empty, and checked it was gone. The server
log ends with `CommonUnixCrashHandler: Signal=11` and
`Segmentation fault (core dumped)` after the SIGINT, as the audit's own
server log does; the server's shutdown, not Specola, produced it.

Steps 4 and 6 to 11 ran from a scratch directory outside the repository
with `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1` set and `PYTHONPATH`
unset. Step 10 was launched with `setsid --wait` under `set -m`, so
SIGINT was at its default (`SigIgn: 0000000001001000`, no `0x2` bit).
The adapter needed no environment variable: it reads
`SPECOLA_CARLA_HOST` and `SPECOLA_CARLA_PORT` (default
`127.0.0.1:2000`, CARLA's default port) from `src/specola/adapters/carla.py`,
and the check used the defaults. A non-default port was not tried.

## Example specification

`examples/carla_specs.yaml` is new. A fresh server has no vehicles at
reset, so the example was written from the declared `vehicle` keys
step 6 lists: each vehicle's speed, from `velocity_x/y/z` in m/s, stays
under its `speed_limit`, which CARLA reports in km/h. The first draft
wrote the speed as `sqrt(sum(vehicle_*[velocity_*]^2))`, which step 8
refused because `sum()` over a dynamic class is not resolved before the
class is populated; the final example writes the three components out.

## Findings

- The extra installs a working client on the first attempt, and with it
  `report` works, which the maintained environment refuses for lack of
  pandas and pyarrow. The adapter's install hint, `pip install carla`,
  names no version; today it resolves to 0.9.16, the version of the
  server here.
- The default `--mode random` spawns the fleet the example binds. Under
  `--mode hold` the adapter only ticks the world, so on a fresh server
  the example would wait for vehicles another client adds; that path
  was not run here.
- Of 20 steps, 18 recorded a camera frame, as in the surface-parity
  audit's `web` cell; the bird's-eye cameras spawn on first use.
- `Web warning: cv2 unavailable, using PIL JPEG encoder (slower).` The
  `web` extra installs Pillow for this case.

## Not fresh

The server package was already unpacked on the machine, so its download
was not exercised. pip resolved against PyPI but took most wheels from
the machine's pip cache. The machine has an NVIDIA GPU with 32 GB of
memory; a GPU below CARLA's recommendation was not tried.
