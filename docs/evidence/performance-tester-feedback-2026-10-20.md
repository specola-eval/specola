# Measure the runtime and memory effect of the tester-feedback changes

Status: measured on 2026-10-20 against the `tester-feedback` branch. Every
number below comes from an accepted run whose command, source pins and raw
output are retained under `logs/tester-feedback/task-6/` (local, not
committed). The twelve offline report exports of the saved simulator outputs,
planned as the last matrix phase, did not run; the export cost is covered by
the fixture comparison and by the quickstart export phase instead. This record
reads accepted artifacts; it launches nothing.

The branch changed three things that can cost time: geometry policies moved
from the RoboDojo adapter into spec-owned functions (Task 9), every run now
exports CSV and PDF reports after cleanup (Task 10), and MuJoCo-family adapters
start with a 0.05 m geometry cutoff instead of refusing without `--queries`
(Task 3). Two optimizations followed from the measurements: the RoboDojo
geometry cache survives ordinary pose edits (commit `aa78b77`), and allocation
tracing during export is opt-in (commit `4da8b69`).

## Read the LIBERO quickstart as a tester runs it

Three baseline and three final runs of the README quickstart, each a fresh
process driven by `benchmarks/quickstart_timing.py`, which records log marks
and samples process-tree RSS every 0.1 s. The production command is

```bash
specola run --sim libero --task libero_goal:0 --queries examples/geometry_queries.yaml \
  --specs examples/libero_goal_00_demo_specs.yaml --mode replay \
  --demo examples/libero_goal_00_demo.hdf5 --web 0
```

in the `libero` conda env with `MUJOCO_GL=egl`, `PYTHONNOUSERSITE=1` and the
worktree on `PYTHONPATH`. The harness sends Ctrl-C five seconds after the
`Dashboard still serving` line. Baseline runs pin `src/specola/report_export.py`
at SHA-256 `9dabb72c…dd91cca` (allocation tracing on during export); final runs
pin `992cf5ab…03cfc291`, the content of commit `4da8b69`. Both sets pin
`src/specola/adapters/robodojo.py` at the cache-change content, which LIBERO
never imports.

| Mark (seconds from launch) | Baseline, three runs | Final, three runs |
| --- | --- | --- |
| Adapter ready | 8.16 / 6.18 / 6.15 | 6.15 / 6.12 / 6.23 |
| First dashboard frame at a client | 10.03 / 8.10 / 7.95 | 7.98 / 7.88 / 8.07 |
| Episode summary (138 steps) | 22.70 / 20.37 / 20.44 | 19.93 / 19.95 / 20.21 |
| Dashboard still serving (reports exported) | 50.28 / 47.97 / 48.04 | 26.01 / 26.03 / 26.29 |
| Summary to serving (export phase) | 27.58 / 27.60 / 27.60 | 6.08 / 6.07 / 6.08 |
| Peak sampled process-tree RSS (MiB) | 2524 / 2493 / 2532 | 2390 / 2521 / 2459 |

Median time from launch to a serving dashboard with reports on disk falls
from 48.04 s to 26.03 s; the export phase falls from 27.60 s to 6.08 s. The
replay itself (about 20 s for 138 steps, 145 ms per step including the
dashboard recorder) is unchanged, so the whole difference is the tracing
default. The first baseline run paid 2 s more to reach ready, which the two
later baseline runs and all final runs did not; that is cache warmth, not a
source difference. A no-query final run (`--queries` omitted, the new built-in
cutoff) reached the summary at 20.26 s and serving at 26.36 s, the same as the
explicit-query runs. Three Stop runs (Stop pressed after step 70) reached
serving at 16.44 to 16.63 s with reports present.

Raw records: `logs/tester-feedback/task-6/starting-quickstarts/full-{1,2,3}/summary.json`
and `logs/tester-feedback/task-6/final-matrix/quickstarts/{full-1,full-2,full-3,no-query,stop-1,stop-2,stop-3}/summary.json`;
launch and postflight custody under `starting-quickstarts-runner/` and
`final-matrix-continuation-runner/`.

## Read the export cost on its own

Six fresh offline processes exported the same retained 21-row, three-episode
fixture, alternating allocation tracing on and off. Median full-call duration
is 32.60 s with tracing and 7.05 s without; the rendered pages are
byte-identical. The exact arithmetic, worker arguments and the 17-page visual
review are recorded in
[automatic-reports-2026-10-19.md](automatic-reports-2026-10-19.md#read-the-measured-allocation-tracing-policy).
With tracing off the manifest's Python allocation peak is null, which the CLI
reports as unavailable, never as zero.

## Read the RoboDojo old-adapter versus spec-function comparison

The matched matrix runs the `stack_bowls` scene with the `arx_x5` robot, seed
and layout zero, in the `RoboDojo` conda env on the R580 driver. Each
arm/case group is three fresh processes; each process runs three ordered
episodes of five warmup plus twenty measured observations. `old` is the
historical adapter with its built-in containment and sampled-distance
computations; `new` is the branch with those computations declared as spec
functions over native inputs, including the geometry-cache change. The cases
select which functions the specification uses: `control` and `unused` declare
none, `containment` the polygon containment, `sampled-distance` the sampled
mesh distance, `both` both; `persist-*` add ordinary `Session` persistence.

Table medians are the median of three per-process medians, each pooling
sixty measured steps; p95 is the median of the three per-process p95s.
`Observation + monitor` is the `Session.step()` boundary.

| Arm | Case | Observation + monitor median / p95 (ms) | Total step median / p95 (ms) |
| --- | --- | --- | --- |
| old | control | 4.056 / 6.517 | 9.358 / 14.655 |
| new | control | 2.286 / 2.822 | 6.798 / 12.553 |
| new | unused | 2.248 / 2.716 | 6.864 / 10.504 |
| old | containment | 4.206 / 5.405 | 9.107 / 12.572 |
| new | containment | 4.554 / 6.721 | 9.490 / 13.980 |
| old | sampled-distance | 12.478 / 20.515 | 19.826 / 32.848 |
| new | sampled-distance | 10.499 / 18.929 | 17.400 / 28.692 |
| old | both | 11.699 / 18.673 | 18.190 / 32.620 |
| new | both | 12.829 / 23.919 | 19.829 / 32.246 |
| old | persist-control | 3.942 / 6.794 | 8.916 / 13.918 |
| new | persist-control | 2.255 / 2.666 | 6.702 / 10.281 |
| old | persist-both | 12.150 / 17.756 | 19.350 / 30.333 |
| new | persist-both | 12.320 / 19.557 | 19.206 / 29.759 |

Steady-state cost per observation is within about 1 ms of the old adapter in
every case that uses a function, and 1.7 ms lower when no function is used
(the old adapter computed its geometry whether or not a spec read it). The
direction varies by case; the three-process groups do not support a
confidence interval or a single speedup figure, and none is claimed. Before
the cache change the migrated functions cost about 183 ms per observation
(`both`) and 174 ms (`sampled-distance`), because every observation rebuilt
the scene; the final profiles record zero measured-step scene builds, the
same 600 sampler updates and 540 distance calls, and 53,760 bytes copied per
sixty observations where the starting migration copied 4.3 GB. Cached
operands were compared against fresh collections at 60 episode boundaries
before any timing was accepted.

Sampled process-tree RSS and driver allocation maxima are retained per
process in `logs/tester-feedback/task-6/final-analysis/process-tables.md`;
they are whole-worker lifetime maxima, including Isaac Sim, and were not
reduced to a single number here. The full analysis, with its nine-episode
breakdowns, attribution profiles and the bounded assessment of the remaining
optimization candidates (vectorization, batching, buffer reuse, transfers,
shared outputs, compiled kernels), is
`.superpowers/sdd/2026-10-19-tester-feedback-fixes/task-6-final-performance-analysis.md`
in the worktree, reproducible with the three `analyze*.py` and `render.py`
commands it names.

## Read the RoboCasa extraction cost with and without specification scope

One process per arm and scope on `PickPlaceCounterToCabinet` with
`PandaMobile`, seed 0, layout 1, style 1, three repetitions of ten warmup
plus twenty measured steps, in the `robocasa` conda env. The scoped runs use
the branch's default 0.05 m cutoff with a specification that binds no
geometry pair; the unscoped runs omit the specification filter entirely so
discovery reads the full eligible population.

| Arm | Scope | Observation median per repetition (ms) | Pair results / native queries per observation |
| --- | --- | --- | --- |
| old | scoped | 102.08 / 102.73 / 102.70 | 0 / 0 |
| new | scoped | 100.70 / 101.02 / 101.35 | 0 / 0 |
| old | unscoped | 7190.84 / 7598.19 / 7861.10 | 120,963 / 7,288 |
| new | unscoped | 7034.17 / 7455.41 / 7845.00 | 120,963 / 7,288 |

The branch does not change RoboCasa cost: the default cutoff registers the
same 120,963 eligible pairs the explicit file did, and a specification that
reads none of them executes no native query. Unscoped extraction, which only
standalone discovery performs, costs about 7.5 s per observation in this
950-geom scene on both arms; that is a pre-existing cost of reading every
pair, retained here as the reason the dashboard lists unread pairs as "not
measured" rather than reading them. Raw records:
`logs/tester-feedback/task-6/final-matrix/robocasa-{old,new}/{scoped,unscoped,unscoped-retry-1}/result.json`
(the first old unscoped attempt failed in the harness counter, not the
simulator, and was retried on corrected harness code with the same source
pins).

## Keep the limits in view

Fresh processes share the machine's filesystem and compiler caches, and the
old/new order is fixed, so these are matched descriptive comparisons, not
independent samples. The quickstart marks are log-receipt times at the
harness and do not decompose startup into import, native guard, environment
construction and encoder probing; the brief's startup remedies remain
hypotheses. GPU and OSMesa paths are different machines' problems; the
container matrix in
[docker-quickstart-2026-10-19.md](docker-quickstart-2026-10-19.md) has not
run. Memory figures are sampled maxima of whole process trees, never
adapter-exclusive allocation.
