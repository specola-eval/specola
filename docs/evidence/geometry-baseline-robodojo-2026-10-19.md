# Capture the old RoboDojo geometry path

Date: 2026-10-19. Status: reviewed harness, completed 36-worker baseline;
final result acceptance is pending controller review. Production remains
unchanged. The matrix establishes exercised sampled-kernel cost and retained
native operands, with a capped-only distance limitation.

**The 432 saved mesh calls return `0.05` with `clipped=True`.** This count
includes watch preparation and warmup; 270 calls belong to the 30 measured
mesh capture observations. These results do not establish uncapped native
numeric-distance parity. Before accepting the later migration, compare the
preserved old implementation and the new path on retained native meshes at
deliberate nearby, touching and overlapping poses, plus synthetic cutoff,
missing-input and degenerate fixtures. Keep those discriminating correctness
checks separate from this unchanged live performance workload.

## Scope and source custody

The production source remains at
`85890fa5fbe2630cab964bf0d0615baa1fcc0959`. No production module changes in
this phase. Run `benchmarks/robodojo_geometry_baseline.py` for one explicit
case, instrumentation mode and recording mode per fresh owned process.
The second helper, `benchmarks/robodojo_geometry_capture.py`, captures inputs
and instruments calls without retaining a second active implementation.

The existing `extraction_profile.py` has no RoboDojo builder and assumes a
MuJoCo model or PyBullet body counter. Its extraction-only timer does not
include `Session.step`, recording, native snapshots, cold phases or retained
device memory. The narrow driver reuses `adapter_overhead.py` hardware
provenance helpers and `dashboard_lifecycle_check.py` process-group enumeration
and cleanup. It uses the production registry factory, `Monitor.watch`,
`Session.step`, hold advancer and `Session.end`.

The original pre-fix sanity source archive is
`logs/tester-feedback/task-9a/old-specola-src-reviewed.tar.gz`, SHA256
`adc8c8ea62fa1fc65e40d66591bc508e29a9095c947a90aa1fc24cb2d75d09f5`.
It contains the old `src/`, extraction profiler and the two harness files
before the first review fix. It is preserved unchanged.
Retain the archive and its digest if later migration changes the checkout.
The archive supplies the RoboCasa extraction recipe below. For standalone
RoboDojo execution it also needs the unchanged `adapter_overhead.py`,
`dashboard_lifecycle_check.py` and `workbench_browser_check.py` helpers from
the pinned commit; those helpers are not present in this archive.
The production source and original extraction profiler are also reproducible
from the full commit above. Final capture provenance hashes the production Python
files, the two harness files, all three reused helpers, RoboDojo environment/configuration/utility files
and the selected task source/configuration before and after each workload.
It records the actual loaded Specola path and rejects another worktree.

## Workload and timing boundaries

Each case has the same `rigid_*[root_pos_w_2] >= 0.5` control. Containment adds
`rigid_*[footprint_in_bowl1] >= 0`. Sampled distance adds
`historically[0,1s](mesh_pair_*[mesh_distance_sampled] >= 0.005)`.
The both case adds both formulas. Mesh cases use cutoff `0.05` metres and
robot links `link6`, `link7`, `link8`; the other cases use an empty query
configuration and reject `ROBODOJO_DISTMAX` overrides. The formulas and query
configuration are written to each output directory and hashed in its result.
An unmatched specification slot fails the run.

The all-pair selector is an explicit benchmark choice. The initial retained
capture uses the example's `@ b:bowl0` selector: across its four measured
observations, neither selected arm-bowl pair survives the prefilter, so it
performs zero warmed kernel queries. Its two watch operations perform nine
queries each. The controller accepted the broader benchmark selector to
exercise warmed kernels, preserving cutoff, selected links, vertices and
estimator policy. It does not establish preservation of a historical
natural-language requirement or an absent task operand.

Construction, watch preparation, native reset, advance, observation plus
monitoring, total step and finalization have distinct timing fields.
Per-episode summaries report sample count, median, p95, total and throughput;
raw total-step latencies are retained. `--profile` records inclusive cold
preparation and warmed call attribution separately. Its mesh row census
records candidates, actual queries and prefilter skips for each instrumented
observation. Nested phase times must not be added together. Pose/transfer/refit
and kernel/result-transfer phases include the indicated computation and data
movement; they do not claim a pure CUDA transfer or kernel-only measurement.

`--capture` retains immutable inputs and actual old results. It does not
install timing attribution. Both capture and profile mark the run as
instrumented. The uninstrumented arm installs neither hook set and copies no
operand snapshots. Match it with a separate instrumented coverage run before
interpreting its latency as an exercised kernel workload. `--record` enables
the ordinary Session recording path; omission retains ordinary in-memory
session rows. The dashboard is disabled in each case. These runs measure no
browser, video, rendering or dashboard serialization cost.

## Native input retention

The capture retains USD producer points, face counts, face indices,
orientation, purpose, visibility, prim/root/frame paths and native transforms
before custom triangulation, merging and link selection. It also retains
collected rows before merging and link selection, row-to-object and
row-to-robot-link associations, merged geometry, float32 sample vertices,
body poses, actual per-row updates and queries, cutoff, output values and
clipped flags. Pair outputs and metadata are copied before projection can
remove an unrequested clipped flag.

For containment it captures the exact maker `calc_polygon` pose and bounding
box operands, polygon WKB including coordinate order and cardinality, height
extents, actual outer/inner WKB and old signed margins. Observation records
retain entity identities, labels, scalar outputs, metadata, bindings,
provisionality and robustness. NPZ arrays retain dtype and shape, are copied
before retention and keyed by a digest of dtype, shape and bytes. Result
validation checks input and capture file digests.

These snapshots establish computational operands for later parity. They are
not a simulator-state replay. RoboDojo's incomplete state restore is unused.
Native live end-to-end timings remain a separate workload.

## Sanity evidence

Read `logs/tester-feedback/task-9a/sanity-inspection.log` for the full numeric
inspection and `sanity-both-3/{command.json,provenance.json,worker.log,result.json}`
for execution provenance. The final bounded scene is `stack_bowls`, seed
family `0`, layout `0`, one environment, hold drive, two native episodes,
one warmup and two measured observations per episode.

The four measured observations each contain 21 entities, including 15 measured
mesh pairs. Each has 39 candidate row pairs, nine actual bidirectional sampled
queries and 30 prefilter skips. The same census holds for the two watch and
two warmup observations in this capture. Ten registered row meshes contain
653,301 sample vertices per episode. The plan contains 32 body rows. The
capture has 530 events and 495 retained arrays. It contains 270 raw producer
mesh events across the two episode watch operations.

| Artifact under `logs/tester-feedback/task-9a/sanity-both-3/` | SHA256 |
| --- | --- |
| `result.json` | `dce4785b8b0f1d4cc7a4ba06ca6e8f9f3ec74b99fe6af9b5e43e033520792a2a` |
| `capture.json` | `7460d9ff8e18667e95445fbd0495b705c476941ff3a7a66886240f105ff9e65a` |
| `operands.npz` | `2d60c151c97ee89c2b890b84ba9ce9196c70f935e854a65ff5a702c2d137e658` |
| `provenance.json` | `15f10a2b1516439046b15bcc4ff2792b0b5b0ef25478696423f6f6a7ca55e5aa` |

The earlier successful example-selector capture remains in `sanity-both-2/`:
`capture.json` SHA256
`168998ade73485d974143d67f9e6fc17623cb825a0059fc985ada8d32fcb519c`
and `operands.npz` SHA256
`9347b444b3fa2451399e260304d1726863130ed8964713567d7567a81bb3306b`.
The failed first sanity run remains in `sanity-both-1/`. Its real
`PropertyMeta` serializer error produced exit status zero through Kit, but the
supervisor rejected it using the failure marker and absent valid result.
An actual `PropertyMeta` regression test covers the corrected serializer.

The approved GPU context reports an NVIDIA GeForce RTX 5090, driver
`580.178.04`, UUID `GPU-45e46b7e-a53c-83dc-b7fa-5c7cf89e17fd`.
Native runtime imports report NumPy `1.26.0`, Shapely `2.1.2`, Torch
`2.7.0+cu128` and Warp `1.8.2`. The installed `warp-lang` metadata reports
`1.11.0`, which is not the runtime Warp. Kit loads
`isaacsim/extscache/omni.warp.core-1.8.2+lx64/warp/__init__.py`.
The runtime `context.py` SHA256 is
`ccbaa7d70f326b60ddb056e3e7c6cd1c0b34f9c10fc4966ce431c9bb77123b0f`;
native `warp.cu` SHA256 is
`48d6de26fba3ca802760b430020177ac9a868db050013eee83db00f12926e551`.

## Memory and native ownership

The supervisor samples RSS for its owned process group and records individual
PIDs plus aggregate sampled RSS. Worker checkpoints include descendant RSS
before watch, after warmup, after end, after releasing owned mesh/plan
references and after native environment close. The requested interval is
0.5 seconds; timestamps retain actual sampling times. This is sampled memory,
not a proof of an unsampled peak.

The PID-filtered `nvidia-smi` query retains its exact command, raw output and
MiB units. It reports driver process allocation, including pools. Torch
allocated/reserved statistics describe Torch only. The runtime Warp 1.8.2
public `get_mempool_used_mem_current(device)` and
`get_mempool_used_mem_high(device)` read the selected CUDA default pool's
`UsedMemCurrent` and `UsedMemHigh` attributes. These shared counters do not
measure adapter-exclusive live bytes. A public reserved-pool byte counter is
unavailable in the inspected exports; the release threshold is configuration.
See the pinned [Warp 1.8.2 source](https://github.com/NVIDIA/warp/blob/v1.8.2/warp/context.py)
and [allocator documentation](https://nvidia.github.io/warp/v1.8/modules/allocators.html#querying-memory-usage).

RoboDojo's current adapter base close calls render/native-extraction hooks;
it does not own Kit or clear the retained mesh estimator. In a fresh child the
harness verifies a `SimulationApp` owner, unregisters the factory's exact
environment/app exit callbacks, releases mesh/plan references while CUDA is
available, calls native environment close once, then calls app close once in
a finally block. It does not invoke destructors manually.

Both successful captures observe environment close return, app close entry,
`fast_shutdown=True`, then status zero with no forced cleanup and an empty
owned group. `native_cleanup_completed` remains false: app close return and
full extension cleanup are unverified. Isaac Sim 5.1 documents
`fast_shutdown` as "True to exit process immediately, false to shutdown each extension."
The harness preserves this production setting. A supervisor termination never
counts as native cleanup. See the maker's
[SimulationApp API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)
and [RoboDojo task close source](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/environment/task_env.py).

## Preserve the RoboCasa old-side recipe

The RoboCasa timing baseline is pending. The final archive described below
preserves the complete old source and recipe before migration. Extract it into a fresh directory and set its absolute
`src` path in `PYTHONPATH`; run the archived extraction profiler from that
directory. Use the unchanged `examples/robocasa_specs.yaml` from commit
`85890fa5fbe2630cab964bf0d0615baa1fcc0959`, with a retained digest, and the
existing RoboCasa environment. The prepared workload is
`PickPlaceCounterToCabinet`, seed `0`, layout `1`, style `1`, 20 steps,
three repetitions, first with `--specs examples/robocasa_specs.yaml`, then
without `--specs`, then separate `--profile` runs with distinct output names.
Keep `PYTHONNOUSERSITE=1`, `MUJOCO_GL=egl`, writable absolute Numba/Matplotlib
cache paths, full logs and bounded owned process groups. Execute it sequentially
with RoboDojo workloads. This profiler measures adapter extraction only; it
does not substitute for total future function, monitoring or dashboard cost.

The frozen specification is
`logs/tester-feedback/task-9a/robocasa-specs-old.yaml`, SHA256
`2d433a1698ac81e6f2c788e180cb3977c18541c30db71f3b9425e5ef06540677`.
The exact commands below are prepared, not executed. Run each child under the
reviewed owned-group supervisor (`start_new_session=True`, timeout, and
`stop_group` in a finally block), with full stdout/stderr logs. The archive
and source manifest identify the old implementation independently of later
worktree changes.

```bash
TASK9_WT=/workspace/specola-worktrees/tester-feedback
TASK9_OLD="$TASK9_WT/logs/tester-feedback/task-9a/old-source-extracted"
TASK9_RC="$TASK9_WT/logs/tester-feedback/task-9a/robocasa-old"
mkdir -p "$TASK9_OLD" "$TASK9_RC" "$TASK9_RC/numba" "$TASK9_RC/matplotlib"
tar -xzf "$TASK9_WT/logs/tester-feedback/task-9a/old-geometry-baseline-final.tar.gz" -C "$TASK9_OLD"

PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH="$TASK9_OLD/src:$TASK9_OLD" \
NUMBA_CACHE_DIR="$TASK9_RC/numba" MPLCONFIGDIR="$TASK9_RC/matplotlib" \
/usr/bin/time -v -o "$TASK9_RC/scoped.time" \
/opt/conda/envs/robocasa/bin/python "$TASK9_OLD/benchmarks/extraction_profile.py" \
  --env robocasa --task PickPlaceCounterToCabinet --seed 0 --layout 1 --style 1 \
  --steps 20 --reps 3 --specs "$TASK9_WT/logs/tester-feedback/task-9a/robocasa-specs-old.yaml" \
  --out "$TASK9_RC/scoped.json" > "$TASK9_RC/scoped.log" 2>&1

PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH="$TASK9_OLD/src:$TASK9_OLD" \
NUMBA_CACHE_DIR="$TASK9_RC/numba" MPLCONFIGDIR="$TASK9_RC/matplotlib" \
/usr/bin/time -v -o "$TASK9_RC/unscoped.time" \
/opt/conda/envs/robocasa/bin/python "$TASK9_OLD/benchmarks/extraction_profile.py" \
  --env robocasa --task PickPlaceCounterToCabinet --seed 0 --layout 1 --style 1 \
  --steps 20 --reps 3 --out "$TASK9_RC/unscoped.json" > "$TASK9_RC/unscoped.log" 2>&1
```

For separate instrumented replicates, add `--profile` and replace each scoped
or unscoped output stem, including the log and GNU time file, with
`scoped-profile` or `unscoped-profile`. Do not compare those observations to
uninstrumented ones as equal-cost measurements.

## Acceptance boundary

The controller approves the harness after scoped review and authorizes the
unchanged 36-worker inventory before execution. The matrix results below await
final result review. Task 9B migration remains outside this phase. Check every
argument against production, the actual native population and `MISTAKES.md`.
What am I assuming that I have not listed? Full simulator replay, uncapped
native distance parity, full-history temporal steady state, a larger RoboCasa
timing result and full extension cleanup remain unverified.

## Address the first harness review

The first independent review requests passive allocator checkpoints, stronger
artifact gates and actual native device provenance. The revised checkpoint
reads a loaded Torch module only when `torch.cuda.is_initialized()` is true.
Warp pool counters receive only a device already held by the actual distance
calculator; the checkpoint does not import Warp or resolve a new device.
Before the first mesh preparation, and throughout a control run, these
counters are unavailable. A retained device handle permits the counters after
owned mesh references are released. The public getter implementations can
call initialization helpers internally; the guards ensure those helpers do
not start a previously unused runtime. The pinned
[PyTorch 2.7 initialization query](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.is_initialized.html)
and installed 2.7.0 source support the Torch guard.

Validation now requires the exact input digest key set, matching
capture/profile/instrumentation/recording flags, an owned readable nonempty
recording file for each requested recording episode, and environment-close
return plus app-close entry or return. An unreturned app close also requires
the observed production `fast_shutdown=True` setting. Negative fixtures check
these gates independently. The focused suite reports 67 passed and one
deselected in `logs/tester-feedback/task-9a/fix-1-tests.log`.

Two bounded revised-source probes run sequentially, each with two native
episodes, one warmup and two measured steps per episode:

- `fix-1-control/`: no capture, profile or recording. Six entities, no mesh
  plan or registered samples, unavailable inactive allocator counters at each
  of its six episode checkpoints and both teardown checkpoints.
- `fix-1-both-profile-record/`: profile plus ordinary Session recording. Both
  episodes produce readable nonempty CSV result files. Each of the four
  measured observations executes nine queries and 30 prefilter skips out of
  39 candidate row pairs. Cold and warmed attribution are retained separately.

The exact worker argv is in each `command.json`. The executed coordinator
commands use the environment prefix in the task report's native command,
with `--case control` and output `fix-1-control`, or `--case both --profile
--record` and output `fix-1-both-profile-record`; the common episode/step/
warmup/timeout arguments remain `2`, `2`, `1`, `240`. Full logs, validation,
device readings, counters, file sizes and digests are retained in
`logs/tester-feedback/task-9a/fix-1-native-inspection.log`.

Each revised probe reports CPU for the RoboDojo environment, DirectRLEnv,
physics context, scene and both used `asset.data.body_pos_w` tensors. The
application requests GPU index zero and the sampled distance uses Warp
`cuda:0`. These are separate devices. Torch CUDA remains uninitialized and
its counters are unavailable, not measured zero. The control never requests
Warp pool counters; the mesh probe requests them only after real preparation.
Both probes observe environment close return, app-close entry and status zero
with no timeout, forced cleanup or owned survivors. Full app-close return
remains unverified.

| Revised probe artifact | SHA256 |
| --- | --- |
| `fix-1-control/result.json` | `59a060c1a4df946fa501abdf6453a0cef984d6379d292d542cf3b91284467dea` |
| `fix-1-both-profile-record/result.json` | `e83c37b7914ec5bc8a37181e647ba0428e8a39caae555b53138e2d716cc85d78` |
| Each probe's `provenance.json` | `24c795cf0ae926856739f8df9e2109cf374ad2b6989522b10ec7d070301a675a` |

New source-stability manifests also retain the three reused helpers:
`adapter_overhead.py` SHA256
`734045352b09130c7dd6ef53040a30a7a59fe939366dcf3196d833ee02358521`,
`dashboard_lifecycle_check.py` SHA256
`890a3baa5267c1678c2ecfbf1ceb20c7322bc94be8cf0ef49b29169f00423e38`,
and `workbench_browser_check.py` SHA256
`65e350582fec4b089ba0ab00d584215c6db11bce2ab9198236827b96092c0d5d`.
Their pinned-commit custody remains distinct from the older archive, which
omits them. The existing native captures also pass the revised artifact gates.

The retained all-pair capture and revised mesh probe contain 107 collected
meshes in episode zero and 157 in episode one, while their registered sample
counts remain unchanged. Native scene growth is a variable in retained-memory
interpretation; these runs do not attribute it to a function cache.
The matrix's five warmup steps at the captured 0.004-second timestep
do not fill a one-second temporal history. Its timing cells describe bounded
post-warmup observations, not established full-history monitor steady state.
Scoped review resolves the three findings and approves the unchanged
36-worker matrix before execution. The results below use the revised source.


## Completed native matrix

The exact executed argv, environment overrides/unsets, fixed source hashes,
output paths, recording flags and sampling settings are in
`.superpowers/sdd/2026-10-19-tester-feedback-fixes/task-9a-measurement-matrix.json`,
SHA256 `601d7c2da53bb7ea3e0b4444cb67d150ba19ac734a8c242261159aa8e7be6e10`.
Workers run sequentially with one native environment each. The first two run
directly; the remaining cells use the unchanged per-cell supervisor through
`logs/tester-feedback/task-9a/matrix-runner.py`. Each cell retains its outer
coordinator log, `worker.log`, exact worker `command.json`, provenance,
result, supervisor samples and any requested recording or capture.
The runner's full log and completion inventory remain beside those cells.

The population is 36 workers and 108 native episodes: 24 uninstrumented timing
workers (three fresh processes for each of four cases and two recording
flags), eight profile workers (four cases and two flags), and four capture
workers (four cases, recording off). Each worker has three episodes. Timing
and profile episodes have five warmup and 20 measured observations; capture
episodes have two warmup and five measured observations. This yields 1,980
measured observations: 1,440 timing, 480 profile and 60 capture.

Fresh-process construction can reuse shader, Warp and asset disk caches primed
by sanity. These are process-cold replicates, not disk-cold or compile-cache-cold
replicates. Five warmup steps at the observed `0.004`-second timestep do not
fill the one-second historical window. Read the timing as bounded post-warmup
sampling, not established full-history monitor steady state.

The 36 workers pass the source and artifact gates, exit zero, report no timeout
or forced termination, and leave no owned group members. Each records native
environment close return, maker app-close entry and `fast_shutdown=True`.
Full app-close return remains unverified and `native_cleanup_completed=false`
remains in each result. The sum of supervised cell elapsed durations is
693.787787 seconds; it includes per-cell instrumentation, sampling and teardown,
excludes inter-cell gaps, and is not simulation-only runtime.

The production/harness source fingerprint is
`006be97b6a38afc817f19d02b720f7673b8503ca9646f91af270ee3d4468882f`
in 36 of 36 workers. The native source fingerprint is
`3b9706b0e0330aa9a88aadc6c1eba112b5ac9b68029bcbaaa152a3bf175ab88e`
in the same population. Runtime Warp is Kit-bundled **1.8.2** throughout;
the separate installed `warp-lang` distribution metadata **1.11.0** does not
identify the loaded implementation. The environment, DirectRLEnv, physics
context, scene and two used robot `body_pos_w` tensors report CPU in each
worker. App GPU index zero and the mesh estimator's Warp `cuda:0` are separate
from those producer devices. Torch CUDA stays uninitialized.

### Actual geometry and query population

Each mesh case has 27 episodes across its timing, profile and capture workers.
Each episode reports 21 entities, 15 measured mesh pairs, 32 body rows, ten
registered row meshes and 653,301 registered vertices. Collected native mesh
rows grow from 107 to 157 to 207 over the three resets; registered samples
remain constant. Each nonmesh case has six entities and an unbuilt mesh plan;
its zero registered-row count says nothing about meshes in the native scene.

Across the 240 measured mesh profile observations, each census is 39 candidate
row pairs, nine actual bidirectional queries and 30 prefilter skips. The two
mesh captures retain the same census at 48 stages including watch and warmup,
of which 30 are measured observations. Their 432 saved query results are
capped at `0.05`, including 270 measured calls. Each measured mesh capture
observation submits 1,298,541 source-plus-destination vertex queries across
those nine pairs, counting repeated samples across pairs. This establishes
kernel execution under the saved cutoff, not an uncapped distance oracle.

| Capture case | Saved events | Retained arrays |
| --- | ---: | ---: |
| control | 147 | 97 |
| containment | 189 | 97 |
| both | 1,203 | 743 |
| sampled-distance | 1,161 | 743 |

### Uninstrumented observation latency

Each table row contains three fresh workers, nine episodes and 180 measured
observations. Each episode contributes a median and a p95 computed from its
20 observation-plus-monitor timings. The table reports the median of those
nine medians and the median and range of those nine p95s; it does not infer a
pooled observation p95. Throughput is 180 divided by the sum of measured
observation-plus-monitor durations. Recording arms are independent observations;
these results do not establish a causal speedup from enabling recording.
Raw total-step samples and their separate pooled percentiles remain in the
summary, along with every episode statistic and cold replicate identity.

| Case | Record | Median of episode medians (ms) | Median of episode p95s (ms) | Episode p95 range (ms) | Observation throughput (Hz) |
| --- | --- | ---: | ---: | ---: | ---: |
| control | off | 3.844 | 4.998 | 4.065–7.605 | 241.41 |
| control | on | 3.854 | 4.252 | 3.893–8.483 | 241.12 |
| containment | off | 4.192 | 5.225 | 4.215–8.137 | 227.41 |
| containment | on | 4.131 | 4.371 | 4.213–7.872 | 234.39 |
| sampled-distance | off | 11.531 | 13.715 | 11.372–59.035 | 66.49 |
| sampled-distance | on | 11.318 | 13.376 | 10.744–19.109 | 84.11 |
| both | off | 11.523 | 12.979 | 11.463–21.721 | 80.59 |
| both | on | 11.369 | 12.740 | 12.316–33.759 | 77.80 |

The p95 ranges retain slow observations rather than trimming them. For example,
the sampled-distance recording-off episode p95 reaches 59.035 ms. Read the
per-episode and per-cold-replicate values before generalizing these bounded
hold-driven results to other layouts, motions, tasks or pair populations.

### Construction and retained memory

Construction and first watch columns are medians of three process-cold values.
RSS and driver GPU columns are the maximum sampled peak among those workers,
not proof of an unsampled maximum. The final column retains each cold worker's
after-end episode-two minus episode-zero RSS difference in MiB.

| Case | Record | Construction (s) | First watch (s) | Sampled RSS peak (GiB) | Driver GPU peak (MiB) | After-end RSS changes (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| control | off | 11.473 | 0.060 | 6.061 | 5362 | 186.35, 188.46, 201.74 |
| control | on | 11.435 | 0.061 | 6.080 | 5330 | 199.14, 185.26, 188.02 |
| containment | off | 11.530 | 0.060 | 6.060 | 5362 | 195.42, 177.94, 190.79 |
| containment | on | 11.462 | 0.060 | 6.065 | 5362 | 181.30, 197.04, 186.59 |
| sampled-distance | off | 11.530 | 0.223 | 6.250 | 5586 | 224.82, 207.86, 210.61 |
| sampled-distance | on | 11.434 | 0.226 | 6.241 | 5586 | 210.11, 217.74, 213.34 |
| both | off | 11.433 | 0.235 | 6.236 | 5586 | 208.62, 213.17, 205.62 |
| both | on | 11.328 | 0.222 | 6.240 | 5618 | 207.39, 218.69, 219.20 |

Driver GPU samples are available at 1,276 of 1,276 sampled points over the
36 workers. Checkpoints also retain shared Warp default-pool current/high
counters after actual mesh preparation, reference release and native
environment close. Unused Warp and uninitialized Torch counters remain
unavailable. Adapter-exclusive live GPU bytes and reserved-pool bytes remain
unavailable. RSS growth coincides with the native collected-mesh growth above;
this experiment does not identify function-cache retention as its cause.

### Separate inclusive attribution

Each row contains one instrumented worker, three episodes and 60 measured
observations. Values are mean inclusive milliseconds per observation. `n/a`
means that phase is not called in the measured window. Nested times overlap
and must not be summed. Cold preparation attribution remains separately in
individual results. These values do not enter the uninstrumented table.

| Case | Record | Producer polygon | Margin | Pose/transfer/refit | Query/result transfer | Mesh pairs | Observation | Monitor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| both | off | 1.760 | 0.272 | 5.166 | 1.136 | 6.955 | 12.011 | 0.258 |
| both | on | 1.972 | 0.296 | 5.460 | 1.100 | 7.260 | 13.123 | 0.284 |
| containment | off | 1.673 | 0.241 | n/a | n/a | n/a | 4.323 | 0.086 |
| containment | on | 1.574 | 0.240 | n/a | n/a | n/a | 4.169 | 0.086 |
| control | off | 1.697 | n/a | n/a | n/a | n/a | 4.285 | 0.061 |
| control | on | 1.660 | n/a | n/a | n/a | n/a | 4.204 | 0.060 |
| sampled-distance | off | 1.900 | n/a | 6.049 | 1.075 | 7.851 | 13.217 | 0.291 |
| sampled-distance | on | 1.835 | n/a | 5.518 | 1.122 | 7.362 | 12.384 | 0.260 |

The control still computes producer polygons in the old path even when it
requests no derived containment or distance signal. The containment profile
population is 480 measured margin calls across containment and both cases.
The pose/transfer/refit boundary combines CPU posing, uploads and refits; the
query/result boundary combines GPU querying and result transfer. This harness
does not separate their pure transfer and computation costs.

### Recompute and preserve the result

From the worktree, the exact aggregation command is:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/workspace/specola-worktrees/tester-feedback/src \
/opt/conda/envs/RoboDojo/bin/python \
  logs/tester-feedback/task-9a/summarize-matrix.py \
  > logs/tester-feedback/task-9a/matrix-summary.log 2>&1
```

The local script is an analysis artifact, not an additional benchmark driver.
`matrix-summary.json` and the full raw `matrix-summary.log` retain the
estimator, eight timing groups, eight profile groups, 36 per-cell records and
artifact hash index. The script verifies 344 typed input/result/provenance/
recording artifacts, 1,680 retained NPZ array dtype/shape/content digests and
228 of 228 saved old containment margins recomputed from their exact WKB.
Outer coordinator logs are retained separately and named in the inventory;
they are not part of that 344-file typed-artifact denominator.
`final-evidence-digests.log` retains SHA256s for the aggregation script,
JSON, raw output, matrix inventory, archive audit and final documents.

The additional final baseline archive is
`logs/tester-feedback/task-9a/old-geometry-baseline-final.tar.gz`, SHA256
`3ef1d5baa194a5c438cf68b095a36a5d4fca54268ab773d50ed290331ad1066d`.
It contains old `src/`, both final reviewed harness files, all three reused
helpers, the unchanged extraction profiler, RoboCasa specification and
`pyproject.toml`. There are 201 files in the archive. The audit in
`final-archive-audit.log` verifies 137 required members against measured source
provenance plus profiler/specification/project hashes; 137 is an audited subset,
not the archive's file count. The original pre-fix sanity archive above remains
unchanged and is distinct from this complete reviewed baseline archive.

For the full RoboDojo driver, extract the final archive into an isolated Git
checkout of commit `85890fa5fbe2630cab964bf0d0615baa1fcc0959` and use that
checkout's absolute `src` path in `PYTHONPATH`. The provenance helper requires
Git context; a standalone extracted tar directory is insufficient for the full
driver. Use the saved matrix environment and commands with paths rebased to
that isolated checkout and separate writable logs/caches. This recipe preserves
the old computation before migration without claiming bit-identical live
simulator replay. No RoboCasa measurement or Task 9B migration runs in this phase.
