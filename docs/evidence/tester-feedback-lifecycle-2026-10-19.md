# Verify dashboard Stop and last-tab closure

The accepted Task 5 matrix contains nine baseline reruns, six Chrome cases on
the existing LIBERO environment, and one Chrome saved-view case on the fresh
Python 3.12 environment. The nine baseline reruns and seven browser cases pass.
The first baseline attempt also completes its native run, but its observer
cleanup fails; preserve it as a tenth baseline invocation, outside the accepted
nine. One fresh browser attempt fails before attaching, before its successful
retry. The original tester segmentation fault is not reproduced and its
cause remains unknown.

The [lifecycle design record](../superpowers/specs/2026-10-19-dashboard-lifecycle.md)
records ownership, timeouts and dependency semantics. The
[original reproduction record](tester-feedback-repro-2026-10-19.md) records the
Task 1 comparison population. Raw Task 5 output is rooted at
`logs/tester-feedback/task-5/`; the acceptance output below is under its
`native/` directory.

## Identify the tested source and environments

The base is `86b5c14`; HEAD during these runs is `c63a889`, whose later commit
adds evidence only. The accepted source includes the reviewed uncommitted Task
5 patch. `native/reviewed-source.patch` has SHA-256
`d9ddf1dcd79554316b2b6e250b6d9ff91b557fa97d012ef6d24ba4781c5fab8d`.
`native/source-provenance-reviewed.json` lists the SHA-256 of each of the 20
changed code, test and benchmark files. `native/final-source-check.json` verifies
20 of 20 hashes unchanged after acceptance. No source or harness edit overlaps
these workloads. The initial failed-observer attempt has its earlier hashes in
`native/source-provenance.json`.

Both interpreters report Python 3.12.13; both environments have hf-libero 0.1.4,
robosuite 1.4.0, MuJoCo 3.8.1, Flask 3.1.3 and simple-websocket 1.1.0. Exact
metadata is in `native/versions-libero.json` and `native/versions-fresh.json`.

| Package | Existing LIBERO environment | Fresh temporary environment |
| --- | --- | --- |
| Flask-SocketIO | 5.6.1 | 5.7.0 |
| python-socketio | 5.16.3 | 5.17.0 |
| python-engineio | 4.13.3 | 4.14.0 |
| `Werkzeug` | 3.1.8 | 3.1.9 |
| websocket-client test dependency | 1.9.0 | Added as 1.9.0 for the retry |

The fresh environment initially lacks `websocket-client`. Its first transport
invocation reports `1 failed, 1 passed in 5.23s`: polling passes; the websocket
client cannot connect and reports `websocket-client package not installed,
only polling transport is available`. The first fresh Chrome attempt fails
at the helper's `import websocket` with `ModuleNotFoundError: No module named
'websocket'`. Its retained result records empty owned process groups after
cleanup. Neither failure demonstrates a server transport defect.

The controller authorizes adding only `websocket-client==1.9.0` as a test
dependency with `--no-deps`. The before/after package inventories differ by
exactly that addition; no package is removed or replaced. `pip check` reports
`No broken requirements found.` Afterward, explicit websocket and polling
parameters both pass in 0.29 seconds, and the unchanged Chrome harness passes
saved view. The failed invocations remain in `native/fresh-transport.log` and
`native/browser-view-fresh/`; accepted retries are
`native/fresh-transport-with-client.log` and
`native/browser-view-fresh-with-client/`.

## Reproduce the sequential matrix

Run from this worktree. The commands below are the resumed inventory, with a
180-second bound for baseline children and a 180-second TERM watchdog plus a
20-second KILL backstop for each browser harness. The first baseline attempt
uses the first timing command with `baseline-full-1` instead of
`baseline-full-1-retry`; its evidence is preserved in the initial directory. Each timing
summary also records the complete child CLI, environment and working directory.
Each browser case retains `command.json`, `result.json`, `child.json`,
`run.log`, `chrome.log` and `before-close.png`.

```bash
set -euo pipefail
cd /workspace/specola-worktrees/tester-feedback
export PYTHONNOUSERSITE=1
export PYTHONPATH=/workspace/specola-worktrees/tester-feedback/src
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/specola-tester-feedback-numba
export MPLCONFIGDIR=/tmp/specola-tester-feedback-matplotlib
mkdir -p logs/tester-feedback/task-5/native

/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-full-1-retry --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-full-1-retry.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-stop-1 --stop-after-step 40 --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-stop-1.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-full-2 --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-full-2.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-stop-2 --stop-after-step 40 --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-stop-2.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-full-3 --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-full-3.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-stop-3 --stop-after-step 40 --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-stop-3.log 2>&1
env -u MUJOCO_GL /opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-unset-gl --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-unset-gl.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-jpeg --web-video jpeg --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-jpeg.log 2>&1
/opt/conda/envs/libero/bin/python benchmarks/quickstart_timing.py --out logs/tester-feedback/task-5/native/baseline-no-query-control --no-query-control --ctrl-c-after 5 --timeout 180 > logs/tester-feedback/task-5/native/baseline-no-query-control.log 2>&1

timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case stop --out logs/tester-feedback/task-5/native/browser-stop -- run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --specs examples/libero_goal_00_demo_specs.yaml --mode hold --steps 0 --web 0 --ui plain --run-name native-stop --output-dir logs/tester-feedback/task-5/native/browser-stop/results > logs/tester-feedback/task-5/native/browser-stop.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case active-close --out logs/tester-feedback/task-5/native/browser-active-close -- run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --specs examples/libero_goal_00_demo_specs.yaml --mode hold --steps 0 --web 0 --ui plain --run-name native-active-close --output-dir logs/tester-feedback/task-5/native/browser-active-close/results > logs/tester-feedback/task-5/native/browser-active-close.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case early-close --out logs/tester-feedback/task-5/native/browser-early-close -- run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --specs examples/libero_goal_00_demo_specs.yaml --mode hold --steps 0 --web 0 --ui plain --run-name native-early-close --output-dir logs/tester-feedback/task-5/native/browser-early-close/results > logs/tester-feedback/task-5/native/browser-early-close.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case completed --out logs/tester-feedback/task-5/native/browser-completed -- run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --specs examples/libero_goal_00_demo_specs.yaml --mode replay --demo examples/libero_goal_00_demo.hdf5 --steps 0 --web 0 --ui plain --run-name native-completed --output-dir logs/tester-feedback/task-5/native/browser-completed/results > logs/tester-feedback/task-5/native/browser-completed.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case discover --out logs/tester-feedback/task-5/native/browser-discover -- discover --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml --web 0 --ui plain --no-pager > logs/tester-feedback/task-5/native/browser-discover.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /opt/conda/envs/libero/bin/python benchmarks/dashboard_lifecycle_check.py --case view --out logs/tester-feedback/task-5/native/browser-view -- view logs/tester-feedback/task-5/native/browser-completed/results/native-completed/recording_000 --port 0 > logs/tester-feedback/task-5/native/browser-view.log 2>&1

timeout --signal=TERM --kill-after=20s 90s /tmp/specola-libero-check/bin/python -m pytest tests/test_dashboard_server.py::test_owned_transport_shutdown_joins_tasks_and_rebinds -q > logs/tester-feedback/task-5/native/fresh-transport.log 2>&1
```

The final command above is the initial fresh transport invocation. Its missing
test-client dependency is resolved with these exact additional commands, under
the same environment exports:

```bash
/tmp/specola-libero-check/bin/python -m pip freeze > logs/tester-feedback/task-5/native/fresh-before-test-client.txt
/tmp/specola-libero-check/bin/python -m pip install --no-deps websocket-client==1.9.0 > logs/tester-feedback/task-5/native/fresh-test-client-install.log 2>&1
/tmp/specola-libero-check/bin/python -m pip freeze > logs/tester-feedback/task-5/native/fresh-after-test-client.txt 2> logs/tester-feedback/task-5/native/fresh-after-test-client-stderr.log
/tmp/specola-libero-check/bin/python -m pip check > logs/tester-feedback/task-5/native/fresh-pip-check.log 2>&1
timeout --signal=TERM --kill-after=20s 90s /tmp/specola-libero-check/bin/python -m pytest tests/test_dashboard_server.py::test_owned_transport_shutdown_joins_tasks_and_rebinds -q > logs/tester-feedback/task-5/native/fresh-transport-with-client.log 2>&1
timeout --signal=TERM --kill-after=20s 180s /tmp/specola-libero-check/bin/python benchmarks/dashboard_lifecycle_check.py --case view --out logs/tester-feedback/task-5/native/browser-view-fresh-with-client -- view logs/tester-feedback/task-5/native/browser-completed/results/native-completed/recording_000 --port 0 > logs/tester-feedback/task-5/native/browser-view-fresh-with-client.log 2>&1
```

The failed fresh browser invocation uses the last command with
`browser-view-fresh` in place of `browser-view-fresh-with-client`, before the
installation. A separate read-only DevTools probe finds no connected page in
its 15-second observation window; it supplies no browser transport evidence.
`native/fresh-browser-transport-probe.txt` retains that limitation and error.
The passing parameterized test directly verifies websocket and polling on the
fresh server stack. The successful browser case verifies its Socket.IO session
and lifecycle without retaining its negotiated transport name.

## Compare the baseline phases

These nine accepted invocations retain eight configured variants from Task 1
(three full, three Stop, unset GL and explicit JPEG), plus one no-query control.
Every accepted child exits with code 0, each observer reports `errors: []`, and
both log-pump and memory-sampler threads join. Timings below are seconds from
the timing harness's launch origin, rounded to three decimal places. Ready,
first received camera frame, episode summary and CLI exit are distinct events;
exact values and sampled process-tree memory remain in each `summary.json` and
`memory.jsonl`.

| Case | Ready | First frame | Summary | CLI exit | Last step | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full-1-retry | 5.753 | 7.595 | 19.718 | 24.928 | 138 | completed |
| stop-1 | 5.684 | 7.586 | 10.952 | 16.193 | 41 | truncated |
| full-2 | 5.715 | 7.493 | 19.516 | 24.762 | 138 | completed |
| stop-2 | 5.695 | 7.449 | 10.766 | 16.030 | 41 | truncated |
| full-3 | 5.668 | 7.455 | 19.458 | 24.697 | 138 | completed |
| stop-3 | 5.718 | 7.477 | 10.740 | 15.955 | 41 | truncated |
| unset-gl | 5.727 | 7.483 | 19.228 | 24.477 | 138 | completed |
| `jpeg` | 5.750 | 7.033 | 19.454 | 24.669 | 138 | completed |
| no-query-control | 5.750 | 7.546 | 19.518 | 24.727 | 138 | completed |

Each Stop request is sent at received step 40 and ends at recorded step 41.
The no-query control now completes because Task 3 accepts the default geometry
query configuration; Task 1 rejected that input. Treat this as changed accepted
semantics, not identical startup behavior. The first full attempt completes
138 steps and exits with code 0, but reports
`Socket.IO close: RuntimeError('Socket.IO transport did not join; Socket.IO transport did not join')`.
A simulator-free reproduction and deterministic test identify an observer queue
sentinel lost during Engine.IO reset. The reviewed fix adds a fresh sentinel
before bounded joins without changing timing marks or CLI inputs.

## Check browser lifecycle and recording ownership

Native Stop and active-close cases use `--mode hold --steps 0`, bounded by the
outer watchdog, so they remain active until the browser action. Completed uses
the shipped replay and verifies visible `ended` plus a completed on-disk
manifest before closure. Early close gates evaluation until the listener is
closed and rebound. Saved-view cases reuse the completed recording and do not
construct a simulator.

| Chrome case | Recording steps / status | Last-tab request to CLI exit (seconds) | Error-trap observations |
| --- | --- | ---: | ---: |
| stop | 62 / truncated | 2.270 | 9 |
| active-close | 84 / truncated | 4.024 | 2 |
| `early-close` | 0 / truncated | 3.719 | 2 |
| completed | 138 / completed | 2.318 | 8 |
| discover | catalogue | 2.523 | 5 |
| view | 138 / saved completed | 2.117 | 8 |
| view-fresh-with-client | 138 / saved completed | 2.119 | 8 |

Seven of seven accepted browser cases rebind the actual released port, exit
with code 0, and record no live owned CLI or Chrome group member after cleanup.
Their final Python-thread census contains only `MainThread`; their owned server,
control and tracked background tasks are stopped. The 42 retained error-trap
observations contain empty error arrays and require the real `__specolaErrors`
array to exist. These observations cover the action boundaries in this harness;
they are not a claim about unobserved browser activity.

Each of seven cases survives the initial 2.5-second zero-viewer interval,
longer than the two-second reconnect grace. Stop, completed, discover and both
saved-view cases also survive refresh and closure of one of two simultaneously
connected viewers. Stop delivers and renders steps 1 and 62; completed and
both saved-view cases deliver and render steps 1 and 138. Their surviving
viewer performs another seek to step 1. Discovery renders its populated class
catalogue after the other viewer closes. Active and early close
exercise last-viewer closure without those refresh/multiple-viewer actions.

Five adapter-owning browser cases (Stop, active close, early close, completed
and discover) record one successful return for each render, native-extraction
and environment cleanup hook, on the measured execution owner. The four run
cases explicitly use `--ui plain`; discover also uses plain UI. This identity
instrumentation does not verify the Textual simulator-worker identity. Separate
focused driver tests cover the Textual bounded join and abandoned-worker path.
The listener closes independently of offline export work; Task 10 can use the
recorder's safe-finalization condition after listener closure.

Early close rebinds at 2.500815 seconds after its close request,
then releases the evaluation gate. Its 3.719000-second request-to-exit timing includes
the intentional 2.5-second gate wait. It is not a pure listener-shutdown timing.
The active-close timing includes simulation cleanup and recording finalization.
The 30-second driver and seven-second recorder budgets are failure bounds;
these successful native cases do not exercise a hung native call. Injected
focused tests verify the unfinalized/abandoned outcome without competing native
cleanup or mutable-buffer finalization.

## Inspect the retained images and limits

The controller inspects all six existing-environment images plus the fresh
saved view. The implementer also inspects Stop, completed, discover and the
existing saved view. Stop shows `stopped`, a camera image of the robot and table, the distinct
3D scene, bindings and replay controls. Completed and both saved-view images
show camera and 3D content, the 138-step trace and current replay step 1.
Completed shows the goal transition to satisfied at step 128. The active-close
image shows live state and step 58; the recording reaches step 84 between image
capture and final shutdown. Its visible `drops 0/15` counter is not loss-free
streaming evidence. Early close shows step 0 without camera frames or bindings,
consistent with the evaluation gate, and provides no native-rendering evidence.
Discovery shows the populated class catalogue, including 10,275 geometry-pair
entries, with no entity selected; this case does not verify signal-detail
rendering.

Several images show close or overlapping zero tick labels. These checks do not
establish that the entire UI is free of clipping, nor do they validate PDF
layout. Image-content checks are separate from the recorded canvas dimensions.
The six existing screenshots are `native/browser-<case>/before-close.png`;
the fresh screenshot is
`native/browser-view-fresh-with-client/before-close.png`.

A scan of 17 full CLI logs (ten baseline attempts and seven accepted browser
cases) finds none of `EGL_NOT_INITIALIZED`, `Segmentation fault`,
`Fatal Python error`, `Adapter cleanup warning:` or
`Adapter renderer cleanup warning:`. The exact file population and results are
in `native/native-log-scan.json`. A separate simulator-free standard
Flask-SocketIO/Werkzeug control reproduces
`AssertionError: write() before start_response` on websocket closure without
Specola's server. Its control script, full log and installed source excerpts
remain under `native/`; this stack trace is preserved and does not establish a
native crash cause.

## Retain the focused regression evidence

The reviewed focused implementation suite records 112 passes in
`fix-1-green.log`; the later ownership-focused run records 13 passes in
`fix-1-owner-green.log`. External SIGTERM and GNU timeout cleanup tests record
three passes in `fix-2-green-final-types.log`; their prior red run records two
expected failures and one pass. The timing observer suite records 15 passes in
`fix-3-green.log`; its focused red run records one expected failure and one real
transport pass, because the observed transport race is intermittent. These
suites overlap; do not sum them into a unique-test count. The final focused correction commands are retained in
`logs/tester-feedback/task-5/test-commands.txt`; the earlier logs preserve
their diagnostic failures and test results. Unrelated whole-branch
failures are outside this focused lifecycle change.
