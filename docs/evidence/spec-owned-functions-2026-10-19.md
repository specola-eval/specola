# Check the spec-owned function migration

Status: shared-contract, authoring, native numerical, diagnostic resource and
LIBERO classification gates pass for the populations below. Final results review
is pending. Task 6 must profile and address or explicitly justify the measured
mesh cost, then repeat matched measurements before overall runtime acceptance.
Task 11 retains the native recording and rendered-browser integration gate.

Use the [architecture decision](../superpowers/specs/2026-10-19-simulator-provided-measurements.md)
for the native/derived boundary and the
[Task 9A baseline](geometry-baseline-robodojo-2026-10-19.md) for old behavior.
The comparison starts at `834587c`. The reviewed implementation changes sixteen
production paths, including deletion of the adapter sampler and its relocation
to `functions_mesh.py`. The old archive remains the comparison reference;
the active RoboDojo adapter has no legacy estimator backend.

## Reproduce the reviewed source

The prior production-review source manifest covers 146 paths:
`logs/tester-feedback/task-9b/production-fix-1/source-sha256.json`, SHA256
`fd5808843206a42f10afbff1b447bb621117d79d20bc20cac45322e0eb65b5e7`.
That manifest records the boundary before the test-fixture correction below.
The scoped production fix diff has SHA256
`71b8bd3d325475addf8ed557b805c1e3b63c71008a476e237134c60fd1612f2c`.
Earlier reviewed manifests and failed attempts remain under the task log root.

A subsequent fixture correction changes two test paths and leaves production
source unchanged. Its refreshed 146-path manifest is
`native-fixture-fix-1/source-sha256.json`, SHA256
`c516bbd5c7543e404bbeeb1d96fd3c28698d28c7f57a3955f1c1b418f1e161b2`.
The accepted workers enforce this refreshed manifest, also stored as
`native-review-source-sha256.json`, before and after execution. The reviewed
correction passes the fresh native retry recorded below.

The exact six-command native inventory, including environment and working
directory, is `production-fix-1/commands.json`, SHA256
`63c83153951816cd70a2e48a950f363aca07dde7447adced9a839a653f73501d`.
Log paths below are relative to `logs/tester-feedback/task-9b/` unless written in full.
Each worker uses the RoboDojo interpreter, absolute worktree `PYTHONPATH`,
`PYTHONNOUSERSITE=1`, layout/seed zero, `arx_x5`, headless mode and CUDA device
zero. The task factory is `registry.resolve('robodojo').factory('stack_bowls',
None, False)`. Launch settings retain `fast_shutdown=True`.

The accepted old source archive SHA256 is
`3ef1d5baa194a5c438cf68b095a36a5d4fca54268ab773d50ed290331ad1066d`.
Four exact retained capture directories supply JSON, NPZ and provenance data;
the accepted Task 9A summary pins their sizes and hashes. Every retained array
is checked against shape, type and content identity. External simulator source
stays read-only and must equal its accepted baseline pin.

## Check the shared contract and authoring paths

Functions load from explicit local YAML declarations, or `--functions PATH`
for standalone discovery and authoring. Local Python is trusted code with the
process's privileges. Loading hashes and compiles the same source bytes;
saving retains resolved absolute references, source digests, parameters and
output metadata. A browser request cannot load another local file.

The shared overlay feeds the existing scalar compiler and temporal evaluator.
Inputs retain observation-aligned values. Requested functions evaluate at most once
per observation and share outputs across specifications. Missing values retain
the monitor's suspension rules; exceptions and outputs that are not finite reject
evaluation. Reset, geometry revision changes and session end release function
caches. Native producer input capture follows requested dependencies.

Unused declarations expose static output schemas on existing native entities
and class capabilities. They do not prepare dynamic targets or capture native
inputs; unknown targets remain absent. Output collisions fail independently of
whether a function is requested. Result metadata separates loaded
`spec_functions` from requested `spec_functions_active`; discovery retains the
same distinction. Requested authoring preparation does not imply evaluation.

The final authoring suite passes 151 tests, skips one because pandas is absent
and excludes one baseline column-count assertion. Its full log is
`authoring-followup/authoring-tests-2.log`. The excluded assertion also fails on
`834587c`: the test expects three columns and the baseline source supplies two.
The baseline reproduction is `authoring-followup/textual-baseline.log`.

The eight new entrypoint cases cover CLI discovery, plain/Textual saving,
CLI check/run, exclusion-only formulas with no input capture, inactive output
collisions, unknown dynamic targets, run/discovery web wiring and restored
browser checking/saving with source digest rejection. Web tests execute actual
SocketIO handlers while replacing transport startup and waiting. They are not
rendered-browser validation; Task 11 owns that matrix.

Production review found a shared identity defect: after reset, a new native
entity could occupy an earlier authored ID. The retained probe reproduces that
failure and its correction in `production-fix-1/identity-before.log` and
`identity-after.log`. Fresh preparation now reserves the complete native catalog
and replaces conflicting authored reservations. The overlay rejects a new
native owner at an authored ID, even if the class matches. Regression native
populations of one, two, two and four receive authored IDs one, two, two and
four, with both specifications at robustness 3.0 throughout.

The fix suite passes 126 tests with one native test deselected; the final
fixture-focused run passes 69 with one native test deselected. Logs are
`production-fix-1/tests-final.log` and `refrozen-tests.log`. These populations
overlap and must not be summed. Compilation and manifest custody checks pass.
The final LSP output is `production-fix-1/diagnostics-final.json`; new test
structural diagnostics are resolved, while missing optional imports and
existing native-fixture diagnostics remain.

## Compare numerical values and attribution

The authored RoboDojo function owns pair selection, selected links,
collision-mesh preference, merge, prefilter, vertex sampling and cap flags.
The containment function retains the previous signed vertex-to-ring margin,
including zero on a covered boundary and the maker's polygon rounding.
The adapter supplies retained native rows, transforms, polygon bytes and
vertices. Rendering retains its own geometry/pose path.

| Gate | Required population | Current result |
|---|---|---|
| Real Warp kernel replay | 432 retained calls; every old call is capped | Passed; maximum absolute difference 0, cap flags equal |
| Full pair policy and attribution | 48 events across two mesh captures, including 30 measured observations | Passed |
| Scope and named-pair policy | One scoped and one named case per mesh capture | Passed |
| Containment output attribution | 30 observations across containment/both captures | Passed |
| Retained margin helper | 228 events | Passed |
| Deliberate retained-mesh poses | Near, touch, overlap and far through full policy and candidate YAML | Passed |
| Native in-memory USD cases | Eleven passes, zero skips/failures | Retry: eleven passes, zero skips/failures |
| English-intent computational fixtures | Seven tests covering three affected formulas and 30 migrated formula identities | Passed |

The old full-policy reference determines its own pairs, identities, mesh
population, attribution and query scope. Candidate targets cannot narrow its
expectation. Candidate policy runs through YAML loading, Monitor/Session,
prepared declarations and compiled scope. Helper equality alone does not
establish that full path. The seven intent fixtures distinguish inside,
boundary, outside, height failure, changing polygon, missing and degenerate
inputs; they do not establish native task trajectories.

The accepted correctness artifact is `native-correctness-gpu-2/result.json`,
SHA256 `497ef3fe82aaf2afff4b00a506e70d45d5090fe6d18d1000347505643209fd54`.
The controller's summary is `native-runner/controller-correctness-review.json`.
The four altered poses reuse native capture row 9, with 3,912 vertices and
1,304 triangles. Near returns `0.005000002682209015` m without clipping;
touch and overlap return zero without clipping; far returns `0.05` m with
clipping. These results compare the previous and moved approximations, rather
than certify exact geometric clearance or penetration depth.

Near/touch/overlap/far cases transform retained native meshes. They are
computational fixtures, not recorded trajectories or exact-clearance proofs.
The sampled distance can overestimate off-vertex clearance and does not measure
penetration depth. Containment margins retain their authored height and temporal
clauses; a nonnegative footprint margin alone does not establish bottom contact.

## Measure native work and ownership

The diagnostic workload covers five fresh processes: control, declared-but-unused,
containment, sampled-distance and both. Each has three episodes, two warmup
observations and twenty measured observations: fifteen episodes, thirty warmup
observations and 300 measured observations. Source/scene identities, requests,
function sets, geometry population and generated YAML digests accompany results.

The five cases pass their artifact gates. Full inspection is
`native-runner/workload-summary.json`, with per-case commands, logs, generated
YAML and results under `native-<case>/` and runner inspections under
`native-runner/workload-<case>/`. Control and declared-but-unused each have
sixty measured snapshots with no function inputs, scene plan, geometry snapshot,
function cache or sampler. Containment captures its input and cache in sixty
snapshots without a mesh plan or sampler. The two mesh cases each have sixty
snapshots with inputs, scene plan, geometry, cache and sampler.

Each table cell is one episode's median over twenty measured observations,
after two warmups. These are diagnostic results from one fresh process per
case, with shared warm filesystem caches. Observation plus monitor is nested
inside total step; do not add the timing levels.

| Case | Observation plus monitor medians, episodes 0/1/2 (ms) | Total-step medians, episodes 0/1/2 (ms) |
|---|---|---|
| Control | 2.174 / 2.157 / 2.357 | 5.804 / 5.942 / 6.375 |
| Declared but unused | 2.188 / 2.199 / 2.193 | 5.857 / 6.131 / 6.266 |
| Containment | 4.343 / 4.362 / 4.387 | 8.052 / 8.824 / 8.721 |
| Sampled distance | 159.353 / 179.157 / 173.188 | 165.020 / 184.828 / 179.767 |
| Both | 176.904 / 160.197 / 164.230 | 182.073 / 168.806 / 171.068 |

The mesh observation-plus-monitor medians are a material performance concern.
The retained old sampled-distance recording-off/on medians of
nine episode medians are 11.531/11.318 ms; the corresponding both values are
11.523/11.369 ms. The nine recording-off episode medians range from
10.573 to 31.228 ms for sampled distance and 10.948 to 17.396 ms for both;
`final-evidence/old-no-record-medians.json` retains those exact records and
hashes. Reproduce them with `python3
logs/tester-feedback/task-9b/final-evidence/old-medians.py`; its full output is
`final-evidence/old-medians.log`. Each old arm uses three cold workers, five warmups and
180 measured observations across nine episodes. The new diagnostic cases use
one cold worker and two warmups, with 60 measured observations across three
episodes and resource/checkpoint instrumentation. These differences prevent a
precise causal slowdown ratio. The new costs require profiling and an addressed
or explicitly justified outcome, followed by measurements with matched settings in Task 6.
They are not accepted final runtime.

The retained mesh revision arrays cover 120 measured observations in six
episodes. Within each episode, 19 of 19 adjacent transitions change revision:
114 of 114 transitions in total. Both cases record revisions 4–23, 27–46 and
50–69. `result-review/revisions.json` pins both source results. The frozen code
connects dirty geometry to plan/snapshot release and rebuilding; a changed
function-input revision clears and recreates the query cache. The triggering
USD notices and time spent in each stage remain unknown. Use
`result-review/task6-profile-candidates.md` for the bounded profiling handoff;
revision changes alone do not establish time attribution.

The mesh cases collect 107, 157 and 207 entries across resets, with vertex
counts 3,792,765, 3,806,725 and 3,820,685 and triangle counts 2,177,897,
2,192,527 and 2,207,157. Each of these six new case/episode populations matches
the corresponding nine old records: eighteen old mesh workers and fifty-four
episodes in total. `result-review/population-comparison.json` retains input
hashes and comparisons. Population growth therefore also exists in the old
baseline; these counts do not identify the cause of RSS growth.

Read each memory population in its measured scope. The RSS checkpoints below are bytes
at each episode end after function-cache release. Sampled RSS and driver GPU
peaks span each worker's complete supervisor sample stream. The driver values
sum only GPU PIDs present in that sample's owned process group; they include
retained pools and are not live allocation or absolute maxima.

| Case | After-end process-tree RSS, episodes 0/1/2 (bytes) | Sampled RSS peak (bytes) | Driver GPU sampled peak (MiB) | Available driver samples / samples |
|---|---|---:|---:|---:|
| control | 6,315,008,000 / 6,419,623,936 / 6,507,646,976 | 6,545,235,968 | 5,446 | 141 / 141 |
| unused | 6,311,096,320 / 6,413,590,528 / 6,502,789,120 | 6,542,626,816 | 5,446 | 140 / 140 |
| containment | 6,337,445,888 / 6,447,943,680 / 6,530,174,976 | 6,568,165,376 | 5,446 | 139 / 139 |
| sampled-distance | 6,775,836,672 / 6,937,677,824 / 7,032,139,776 | 7,056,060,416 | 5,574 | 195 / 195 |
| both | 6,794,620,928 / 6,937,296,896 / 7,030,112,256 | 7,054,979,072 | 5,574 | 197 / 197 |

The 812 of 812 available driver samples cover these five diagnostic workers.
`final-evidence/resource-summary.json` retains the input supervisor hashes,
counts and maxima. Reproduce this read-only aggregation from the worktree:

```bash
python3 logs/tester-feedback/task-9b/final-evidence/resources.py \
  > logs/tester-feedback/task-9b/final-evidence/resources.log 2>&1
```

Function caches are empty after each of the fifteen episode ends. Shared Warp
default-pool current usage returns to 3,724,799 bytes at each of the six mesh
episode ends. Process RSS still grows across resets in each case;
`result-review/diagnostic-summary.json` retains the byte counts. Native physics
and used robot pose tensors report CPU, while the mesh sampler uses Warp
`cuda:0`. Five supervisors report exit zero without forced cleanup and retain
environment-close return plus app-close entry, with app-close return unknown.
The 146-path source boundary remains stable. These observations establish the
bounded gates, not memory settling, exclusive ownership or a performance gain.

Each supervisor owns one process group with a 600 s timeout. Missing artifacts,
wrong mode/counts, results that are not finite, source drift, forced cleanup or nonzero
exit reject acceptance. Failed attempts remain evidence; a fresh retry uses a
fresh output directory. The first attempt could not access the GPU in the
default sandbox. Its device check is `native-runner/device-availability.json`;
the host-side probe is `native-runner/outside-sandbox-nvidia-smi.log`. The
interrupted attempt remains in `native-correctness/`; its supervisor records
forced cleanup, so it is not an accepted native result. The fresh retry uses
the exact recorded `native-runner/correctness-gpu-command.json` with output
`native-correctness-gpu-1/`. That run reaches eleven USD cases: ten pass, one
fails and none skip. The missing-stage fixture clears `env.stage`, but the
adapter then resolves the actual global Kit stage and returns geometry. This
is the existing acquisition order: environment stage, scene stage, then global
stage. The failure stops the worker before retained-kernel or full-policy parity.
The run is not an accepted correctness result despite process exit zero.

The fixture correction makes that adapter instance resolve no stage, preserving
the real global stage and production fallback. Four simulator-free cases check
the acquisition order and absence result. The covering suite passes seventy
tests with one native test deselected; the final four acquisition cases pass
again after a type-only fixture edit. Logs are `native-fixture-fix-1/tests.log`
and `acquisition-final.log`. The eleven native cases remain unchanged in count.
The accepted retry command is `native-fixture-fix-1/correctness-gpu-2-command.json`;
its sole command change is the fresh output directory `native-correctness-gpu-2/`.
That retry passes the numerical gates above. Its supervisor records exit zero,
no timeout, no interruption and no forced cleanup. The 146-path source manifest
remains stable at `c516bbd5c7543e404bbeeb1d96fd3c28698d28c7f57a3955f1c1b418f1e161b2`.

Report process-tree RSS, process GPU samples and public shared default-pool
counters. Reserved capacity and adapter-exclusive allocation remain
unavailable. Actual Kit runtime Warp 1.8.2 differs from installed distribution
metadata 1.11.0; new runs must retain both observations. Timing levels are nested
and must not be added.

The no-function path returns before authored reservation work. The unused
declaration path still captures no function inputs and prepares no dynamic
targets, but fresh preparation copies native catalog IDs into a `frozenset` and
filters the base reservation dictionary. Task 6 must measure this reset-only
traversal/copy. The old control still computed maker polygons; its measured
pose/upload/refit stage exceeded query/result cost. No speedup follows from
moving source code or from one cold process per case. Task 6 owns the matched
performance matrix.

Native acceptance requires environment-close return, released function caches,
a process-alive marker after adapter close, and actual entry into the existing
app-close callback at interpreter exit. `fast_shutdown=True` may prevent that
callback from returning. Current Stop/app ownership statements describe code,
not a measured live browser Stop. Forced termination is not graceful cleanup.
The failed GPU run records environment-close return, the process-alive marker
and app-close entry without forced cleanup. These markers do not establish a
live dashboard Stop or satisfy the failed numerical gate.
The accepted correctness retry also records environment-close return and
app-close entry; app-close return remains unknown. The repeated-episode
resource observations above cover three episodes per diagnostic case.

## Preserve the remaining limits

The accepted LIBERO inventory is `libero-review/inventory.json`, SHA256
`0bae4f8b89125f8a85769fed9e138242cff323dcabba5fc4e6a448c79375ac1f`.
From the worktree, the two sequential child commands are:

```bash
/opt/conda/envs/libero/bin/python -m pytest tests/test_spec_intent_viability.py tests/test_x2_spec_set_v2.py -q
/opt/conda/envs/libero/bin/python benchmarks/spec_intent_viability.py --require-complete
```

The exact environment, working directory, source boundary and six intent-specific input pins
are retained in that inventory and `native-runner/libero-intent/` command and
preflight records. Both 900-second owned runs exit zero without timeout,
interruption or forced cleanup. Source checks verify 146 manifest paths and
six additional intent inputs before and after each command; the final exact
command check finds zero matches.

Pytest reports `169 passed, 1 skipped, 3 warnings` across 170 nodes. Forty of
forty native reset/catalog/compile cases pass: indices zero through nine in each
of `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`. The skipped
node is `test_ordering_rows_compile_with_explicit_native_and_rtamt_backends`,
with reason `RTAMT functional backend unavailable in this environment`.
Optional RTAMT ordering compilation remains unverified. Full output is
`native-runner/libero-intent/tests/command.log`.

The ledger command reports 22 rows with status `complete`: twelve
`approximated`, three `expressible` and seven `not_expressible`. Its exact output
is `native-runner/libero-intent/ledger/command.log`; the combined inspection is
`native-runner/libero-intent/summary.json`. Native reset/catalog compilation
does not establish task trajectories or exercise spec-owned geometry-function
loading. Ledger completeness does not establish universal intent viability.

The architecture decision gives dispositions for eleven adapter registrations
from source, not eleven native validations. It retains region velocity as
named arithmetic over observed centers and fixed `_dt`, first observation
zero. Native instantaneous velocity is not an equivalent replacement.

Native root scale, time-sampled geometry and changes bypassing USD authoring
remain unverified. Revision invalidation alone does not establish numerical
support. Keep these limits visible when using this checkpoint. Actual native function
recording, dashboard Stop, save and rendered replay remain Task 11 obligations;
CLI and SocketIO handler coverage does not supply that acceptance.

## Verify artifact identities

All paths in this table are relative to `logs/tester-feedback/task-9b/`.
The exact source and command manifests above remain the launch authority.

| Artifact | SHA256 |
|---|---|
| `native-correctness-gpu-2/result.json` | `497ef3fe82aaf2afff4b00a506e70d45d5090fe6d18d1000347505643209fd54` |
| `native-runner/workload-summary.json` | `22b3a02bc8a47a75bfbbc4e8642ae478d9d686326984b4dc9457de06c9916617` |
| `result-review/diagnostic-summary.json` | `b331ea282078d1a58867bc4ed38130ce9f7d6e2be482ddd590672201155cf10b` |
| `result-review/revisions.json` | `5ccb26382b2f02e7e5a4df9051849b78cdc69cc6376281e28d8aaa1ca2d19ed1` |
| `result-review/population-comparison.json` | `d946def83ffce19a44a8315cbb0d301d418ff7b948a32d5e7f311c2003b02de1` |
| `final-evidence/resource-summary.json` | `daa3a0c02e2f871f208146d346ec85fc841cf1386f5021cb405ebe111339fdab` |
| `native-runner/libero-intent/summary.json` | `b828bd6ee7d1e5286325a2195f0d69eeb7b574873672842f6dbe5cdcfaac7cfc` |

The implementation report retains exact simulator-free test commands,
compilation and LSP outputs, baseline failure reproduction and the independent
source/harness/fix reviews. Overlapping test populations are not added.
The final results handoff is
`.superpowers/sdd/2026-10-19-tester-feedback-fixes/task-9b-results-handoff.md`.
