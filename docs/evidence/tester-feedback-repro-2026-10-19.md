# Measure the LIBERO tester reports

On 2026-10-19, the reviewed harness completed eight configured LIBERO runs and one no-query control. The configured runs reached their requested endpoints and exited normally. The control refused startup and printed EGL cleanup errors. No run in this nine-process population exited from a native fatal signal.

## Measurement contract

You run `benchmarks/quickstart_timing.py` from this worktree with the LIBERO interpreter. The driver preserves the README task, replay file, specifications and configured query file. It uses an ephemeral dashboard port, supplies `n` followed by EOF to the first-run prompt, and enables unbuffered output and faulthandler. You retain each run's complete timestamped output in `run.log`, process-tree RSS samples in `memory.jsonl`, input hashes and cache observations in `provenance.json`, and final metrics in `summary.json`.

The initial source tree is `3153938041f6e333964524e5076af5367e150427`, measured with `git rev-parse HEAD:src` on 2026-10-19 at commit `c73dafaa563f91157c430bc7a7fee776954aabb1`. Commands `git rev-parse cf5c280:src` and `git rev-parse abd5bbe:src` returned that same tree identifier. Each native run also records its actual revision and source worktree status.

The source protocol differs from the task brief's sketch. `src/specola/dashboard/protocol.py:21` defines `Bundle`, whose wire representation carries both `step` and `frames`. The driver uses `step_bundle` instead of the sketch's separate `frame` and `line` events. It records `run_meta` and `episode_end`; the latter reports finalization even when the server does not send a second `run_meta` with `mode=ended`. A first frame means nonempty bytes received by the Python client. Browser decode and paint remain unmeasured.

The Python client attaches during the live run. If it first attaches after the run has ended, it records the ended metadata but does not request a historical frame. The browser requests that frame through a seek. Missing frame telemetry on a late attachment does not establish a browser first-frame failure; you inspect the retained mode before interpreting a missing value.

You measure the root process and descendants sampled through psutil 7.2.2. Summed RSS can count shared pages in more than one process, omit a descendant reparented before enumeration, and miss peaks between samples. It excludes GPU memory. Failed or partially unreadable samples have null RSS. You interpret `peak_sampled_rss_bytes` as a sampled maximum over valid samples, accompanied by valid and missing sample counts. The [version-matched psutil documentation](https://raw.githubusercontent.com/giampaolo/psutil/release-7.2.2/docs/index.rst) and installed source supply the process-enumeration and RSS contract. The Python wire client's event and acknowledgement behavior follows the [official Socket.IO client documentation](https://python-socketio.readthedocs.io/en/stable/client.html), cross-checked with installed python-socketio 5.16.3 and python-engineio 4.13.3.

The driver records whether each explicit cache directory is nonempty before launch. That observation does not establish the cache state of individual assets or compiled functions. You preserve run order and that uncertainty in comparisons.

## Run the fixed population

Use these commands with the reviewed harness. Keep the output directories separate when you measure the revised source. The population is three full configured runs, three Stop runs, one unset-GL run, one JPEG run, and one no-query control: eight configured runs plus one control.

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=/workspace/specola-worktrees/tester-feedback/src
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/specola-tester-feedback-numba
export MPLCONFIGDIR=/tmp/specola-tester-feedback-matplotlib
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
PYTHON=/opt/conda/envs/libero/bin/python
for i in 1 2 3; do
  "$PYTHON" benchmarks/quickstart_timing.py --out logs/tester-feedback/full-$i --ctrl-c-after 5
  "$PYTHON" benchmarks/quickstart_timing.py --out logs/tester-feedback/stop-$i --stop-after-step 40 --ctrl-c-after 5
done
env -u MUJOCO_GL "$PYTHON" benchmarks/quickstart_timing.py --out logs/tester-feedback/unset-gl --ctrl-c-after 5
"$PYTHON" benchmarks/quickstart_timing.py --out logs/tester-feedback/jpeg --web-video jpeg --ctrl-c-after 5
"$PYTHON" benchmarks/quickstart_timing.py --out logs/tester-feedback/no-query-control --no-query-control --ctrl-c-after 5
```

The no-query control omits only `--queries`, preserving the specifications and replay input. A startup refusal remains a measured outcome. The driver defaults to a 900 s limit with bounded termination and joined logging. It sends Ctrl-C five seconds after the episode-summary line, not five seconds after launch. An expired limit is recorded as a timeout.

## Full configured runs

Times are seconds from launch, rounded to three decimal places from each run's `summary.json`. The value `none` means the corresponding telemetry is absent, not zero elapsed time. Derive the rows and ranges with `/opt/conda/envs/libero/bin/python logs/tester-feedback/summarize.py`; retain its complete output in `logs/tester-feedback/summary-derived.log`.

| Run | Ready | Web dashboard | First frame | Summary | Ended on wire | Exit | Returncode | Crash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-1 | 7.283 | 7.521 | 8.599 | 20.453 | 20.452 | 25.767 | 0 | no |
| full-2 | 5.245 | 5.356 | 6.523 | 18.305 | 18.308 | 23.592 | 0 | no |
| full-3 | 5.396 | 5.505 | 6.686 | 18.380 | 18.379 | 23.702 | 0 | no |

## Stop after dashboard step 40

| Run | Ready | Web dashboard | First frame | Summary | Ended on wire | Exit | Returncode | Crash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop-1 | 5.178 | 5.303 | 6.419 | 9.385 | 9.385 | 14.640 | 0 | no |
| stop-2 | 5.235 | 5.348 | 6.491 | 9.434 | 9.435 | 14.687 | 0 | no |
| stop-3 | 5.180 | 5.290 | 6.391 | 9.352 | 9.353 | 14.672 | 0 | no |

## Crash-trigger variants and no-query control

| Run | Ready | Web dashboard | First frame | Summary | Ended on wire | Exit | Returncode | Crash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unset-gl | 5.229 | 5.340 | 6.481 | 18.183 | 18.182 | 23.513 | 0 | no |
| jpeg | 5.321 | 5.430 | 6.016 | 18.128 | 18.129 | 23.445 | 0 | no |
| no-query-control | none | none | none | none | none | 4.635 | 1 | no |

## Check scale and memory

The eight configured runs used task `libero_goal:0`, the shipped 138-state demonstration, six authored specifications, initial readiness entity count 370, and query cutoff `distmax: 0.05` from the hashed query file. Geometry telemetry reported 38 bodies, 239 geometries and 9,822,300 mesh bytes in each of these eight runs. The full runs, unset-GL variant and JPEG variant finalized at step 138. The three Stop runs acknowledged `stop_run` and finalized as `truncated` at step 41.

| Run | Sampled peak RSS, bytes | Valid RSS samples | Missing RSS samples | Finalized step | Wire status |
| --- | --- | --- | --- | --- | --- |
| full-1 | 2582872064 | 243 | 0 | 138 | completed |
| full-2 | 2377666560 | 221 | 0 | 138 | completed |
| full-3 | 2538024960 | 223 | 0 | 138 | completed |
| stop-1 | 2487951360 | 138 | 0 | 41 | truncated |
| stop-2 | 2391023616 | 138 | 0 | 41 | truncated |
| stop-3 | 2514849792 | 138 | 0 | 41 | truncated |
| unset-gl | 2372976640 | 221 | 0 | 138 | completed |
| jpeg | 2423595008 | 221 | 0 | 138 | completed |
| no-query-control | 1860456448 | 44 | 0 | none | none |

The first run began with empty explicit Numba and Matplotlib cache directories. Numba's directory was nonempty before the remaining eight invocations; Matplotlib's directory remained empty in the nine pre-launch checks. The first run's output reported `Assets already downloaded at /home/user/.cache/libero/assets`. No cold-assets measurement follows from these cache observations. A cache contribution to the slower first run remains a hypothesis because the campaign does not isolate it.

The eight configured manifests report `frame_store: jpeg`. The seven default-video runs printed `Web: hardware h264 encoder unavailable; recording camera frames as JPEG images.` The explicit JPEG variant also used JPEG. This population does not measure hardware video encoding. Installed versions recorded per run are hf-libero 0.1.4, MuJoCo 3.8.1, robosuite 1.4.0, Specola 0.2.0, psutil 7.2.2, python-socketio 5.16.3 and python-engineio 4.13.3.

## Findings

- Startup delay: first wire frames arrived in 6.016 to 8.599 s across the eight configured runs. The interval from launch to readiness, 5.178 to 7.283 s, was the largest of the three measured startup intervals in eight of eight configured runs. These measurements do not attribute its internal costs. The interval from dashboard announcement to first frame was 0.586 to 1.180 s. No extended startup stall was observed in this configured population.
- Stop and post-episode shutdown: the three Stop requests were acknowledged and finalized at step 41, 0.471 to 0.516 s after the requests. The eight configured processes remained serving after the episode as the current command specifies, then exited normally 0.243 to 0.328 s after the scheduled Ctrl-C. No Stop or interrupt hang reproduced in this population.
- Crash before the first step: no native fatal signal or faulthandler fatal stack appeared in the eight configured runs or the one control. The tester's crash trigger remains unknown. The sample does not estimate a rare-crash rate and does not cover hardware encoding.
- No-query control: the single control exited 1 before readiness, dashboard startup or any wire frame. Its startup diagnostic and two ignored destructor tracebacks are retained in `logs/tester-feedback/no-query-control/run.log`; the startup diagnostic file is also copied to that directory. Selected log lines are quoted below. These observations establish a required-query refusal and EGL cleanup errors on this control, without establishing the cause of the tester's reported crash.

```text
  4.029224 Error       QueryConfigurationRequired: query_configuration_required: mujoco.mj_geomDistance requires distmax
  4.135924 Exception ignored in: <function MjRenderContext.__del__ at 0x7d8596f33600>
  4.159219 OpenGL.raw.EGL._errors.EGLError: EGLError(
  4.159234 	err = EGL_NOT_INITIALIZED,
  4.159264 Exception ignored in: <function EGLGLContext.__del__ at 0x7d8596f334c0>
  4.159448 OpenGL.raw.EGL._errors.EGLError: EGLError(
  4.159459 	err = EGL_NOT_INITIALIZED,
```

The configured rows contain no harness errors or timeouts, and their logging and RSS observers joined before summarization. The nine provenance files contain the same source-tree identifier and empty source worktree status at launch. Subsequent implementation work does not change that recorded baseline.
