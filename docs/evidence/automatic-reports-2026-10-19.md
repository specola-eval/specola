# Verify automatic CSV and PDF reports

Status: the Task 10 implementation, scoped data fixes and 29-page visual
population pass independent review. This is bounded offline and CLI acceptance;
Task 6 performance and Task 11 native integration/installation gates remain open.
Use the [design record](../superpowers/specs/2026-10-19-automatic-run-reports.md)
for requirements and [reporting guide](../reporting.md) for commands and files.

## Recover the implementation boundary

Comparison base: `adc15ee`. Seven production paths implement the raw writer,
loader/CLI, exporter, completion integration, dependency declarations, branding
and packaged synthetic fixture. The initial review inventory has fourteen
paths: thirteen repository files plus its scratch implementation report. Fix 1
adds `tests/test_report_fixes.py`, giving fourteen repository implementation,
test and guide paths. `logs/tester-feedback/task-10/evidence/custody.json`
verifies those fourteen current files against the original and two scoped fix
manifests. It also rechecks four accepted PDF files, 29 PNGs and six selected raw
Parquet files. This documentation consolidation runs no tests or renders.

CSV with metadata is required, with additional Parquet when available. Empty
records retain headers. Directory regeneration validates raw columns, excludes
`report-*` outputs and selects each CSV/Parquet stem once, preferring CSV.
CSV parsing preserves saved finite float values, NaN/infinity, identity strings,
boolean missingness and the tested caller-integer range. Exact integer values
and missingness agree across measured CSV/Parquet routes; their data types need not
match. Saved formula text, constants and function provenance come from retained
metadata; missing historical text stays unavailable.

The exporter reuses the existing outcome reducer and emits a PDF, outcome and
task/spec tables, complete plot rows, trace availability, saved specifications,
image inventory and a last-published hash manifest. Satisfaction and native
success keep separate denominators. Explicit batches validate saved membership;
a directory scan does not establish benchmark completeness. Individual atomic
replacements do not provide one transaction across the bundle.

Preflight checks reporting dependencies, PDF/font support and the destination
before adapter construction. Export follows finalized rows and adapter cleanup,
before retained dashboard waiting. Completed, truncated and controlled failed
records keep their status; abandoned workers skip finalization/export. Report
errors preserve raw results and the original episode status; an otherwise
successful CLI run returns status 2. Source and lifecycle-spy checks do not
establish a native recording or camera integration result.

Images require an explicit raw/recording association and matching saved
identity/status. The existing JPEG reader supplies the latest indexed camera
frame with its actual sampled step and hash. Synthetic image evidence remains
labeled synthetic. Missing images, encoded video without a decoder and absent
component series/units retain availability explanations.

## Reproduce the measured checks

Paths below are relative to `logs/tester-feedback/task-10/`. Exact historical
argument vectors, environment and log paths are in `evidence/commands.json`.
Commands run from the tester-feedback worktree with this common prefix:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/workspace/specola-worktrees/tester-feedback/src \
MUJOCO_GL=egl \
MPLCONFIGDIR=/workspace/specola-worktrees/tester-feedback/logs/tester-feedback/task-10/mpl-cache \
NUMBA_CACHE_DIR=/workspace/specola-worktrees/tester-feedback/logs/tester-feedback/task-10/numba-cache \
/opt/conda/envs/libero/bin/python
```

The test populations overlap and are not added. Initial tests precede the
reviewed fixes; later scoped checks close their demonstrated defects.

| Population | Arguments after the prefix | Retained result and full log |
|---|---|---|
| Initial focused | `-m pytest tests/test_report_export.py tests/test_results_io.py tests/test_report.py tests/test_session_facade.py tests/test_verdict_consistency.py tests/test_run_startup.py tests/test_run_failures.py tests/test_branding.py -q -x` | 119 passed; `focused-tests-8.log` |
| Initial rendered regression | `-m pytest tests/test_report_rendered.py -q` | 1 passed; `rendered-regression-tests.log` |
| Fix 1 focused | `-m pytest tests/test_report_fixes.py tests/test_results_io.py tests/test_report.py tests/test_report_export.py tests/test_report_rendered.py -k 'not public_run_exports_after_close_and_skips_abandoned_workers' -q -x` | 53 passed, 1 deselected; `fix-1/focused-tests.log` |
| Fix 1 affected CLI | `-m pytest tests/test_verdict_consistency.py::test_never_bound_class_is_undetermined_everywhere tests/test_verdict_consistency.py::test_skipped_class_is_not_applicable_everywhere -q -x` | 2 passed; `fix-1/affected-integration-tests.log` |
| Fix 2 focused | `-m pytest tests/test_report_fixes.py -k 'integer or writer_loader_exact or csv_rejects_malformed or bounded_records or trace_endpoints or never_bound' -q -x` | 43 passed, 7 deselected; `fix-2/focused-tests.log` |

Fix 1 repairs finite float/NaN CSV conversion and per-source legacy task
identity. The acceptance harness now constructs never-bound/skipped markers
through the production row builder: six never-bound groups retain the actual
last-step semantics and explicit unavailable-trace records. Three finite groups
remain plotted. Fix 2 repairs the introduced signed-width narrowing of caller
integers with Python integers in object-backed CSV columns. Seven boundary
values over three routes give 21 cases, plus three nullable-cell cases; this
is not an exhaustive integer population. Eight introduced structural diagnostics
are closed across the four changed Python files; import-resolution diagnostics
remain in the separate analysis environment.

The original review, two fix reviews and implementation reports are under
`.superpowers/sdd/2026-10-19-tester-feedback-fixes/task-10*.md`. Exact numeric
and marker reproductions, failed attempts and recovery evidence remain in the
task logs. `fix-1/evidence-check-final.log` records equal present values and
missingness for the bounded CSV/Parquet comparison. The final scoped code review
is `task-10-fix-2-review.md`; it identifies no open finding in its four paths.

## Recover the accepted visual artifacts

The recorded acceptance command appends
`tests/report_acceptance.py --output logs/tester-feedback/task-10/acceptance-fix-1`
to the common prefix. Its complete log is `fix-1/acceptance.log`. Preserve that
frozen directory; select a fresh output path for reproduction. The generator
renders complete PDF files with `pdftoppm -scale-to 1700 -png` and retains
`pdfinfo`, font, text and render logs beside each PDF.

The independent review opens 29 of 29 enumerated PNGs individually at original
1203 × 1700 resolution, covering four PDF files. Endpoint clipping, legend/data
overlap, identity-losing page breaks and literal caption newline escapes are
closed in this population. The six unavailable groups retain readable identity,
reason and counts. Declared minimum font is 10 pt; the retained font logs show
15 embedded/subsetted/Unicode entries across the four PDF files. These checks do not
establish readability for untested labels or data.

Paths in the following table start at `acceptance-fix-1/`; PDF files are
`<fixture>/report/report-run.pdf`, with PNGs under `report/rendered/`.

| Fixture | PDF pages / inspected PNGs | PDF SHA256 |
|---|---:|---|
| dense | 19 / 19 | `ccc1660379280702e7d6d513d57691cb29cb17075a0b0b4297a1f9ddd493e1b4` |
| `empty` (saved formulas) | 2 / 2 | `99f4866e47efb66dc600208e565d14f16d630506dc3a104cfc089ad9bb670329` |
| `zero-no-specs` | 1 / 1 | `c71ad27c7698179e9d0433e883b621b2938d79db055daf644a79ed016aa01c8c` |
| `historical-failure` | 7 / 7 | `c988191ff283779335067447b15c3dab0d248f359a9c1bfbbcc2dd67eaae4f3d` |

`acceptance-fix-1/rendered-page-inventory.json` pins each PNG. Its generated
`visual_reviewed` flags precede inspection; the acceptance verdict is the
independent `task-10-visual-fix-1-review.md`, with inventory and trace checks
beside that review. The original four PDF files/27 PNGs in `acceptance-ready/`
remain superseded evidence and are not added to the accepted page count.

The dense synthetic fixture has 21 raw rows, twelve outcomes (one pass, one
fail, seven undetermined, three not applicable), satisfaction 1/2 and native
success 0/0 known episodes. Its scene image is a synthetic recording-writer
color grid. These fixtures are not native simulator or policy evaluations.

The accepted PDF files retain fix-1 source provenance. Fix 2 changes CSV integer
loading and annotations; the four manifests select six Parquet inputs, and the
review confirms unchanged plotting/layout. The generator input comparison is
bounded to its first dense export boundary with preflight/export stubbed;
`fix-2/generator-equivalence.log` states that scope. No PDF is regenerated at
fix 2, and these PDF files are not relabeled as products of the later source.

## Retain the measurement and integration limits

`report_build_seconds` and `report_build_python_tracemalloc_peak_bytes` cover
loading and CSV/PDF construction after initial exporter imports. They exclude
artifact hashing, manifest serialization and publication. The historical Task 10
exports below enable tracing. The later Task 6 comparison and opt-in policy are
recorded in the following section.
Whole-process lifetime RSS includes prior monitoring/native pools and is not
report-exclusive allocation. Python allocation peaks exclude native allocation.

| Fix-1 fixture | Build seconds | Python build allocation peak bytes |
|---|---:|---:|
| dense | 34.507951541338116 | 25,320,280 |
| empty with formulas | 2.9852014258503914 | 5,303,617 |
| zero rows/no specs | 2.501443930901587 | 5,414,070 |
| historical failure | 5.759824199136347 | 5,828,602 |

These bounded Task 10 build metrics do not measure the complete export call.
The following Task 6 comparison supplies a separate full-call measurement. Task 11
must verify native camera output, automatic finalized-recording association,
completion/Stop integration and the current wheel. The retained wheel build
predates both fixes and proves contents/declarations, not a fresh installation.
Missing-dependency readiness, Python 3.8/dependency-floor execution and native
resource release remain unverified. RoboDojo environment close does not imply
Kit application or retained native pools are released before process exit.

## Read the measured allocation-tracing policy

The Task 6 comparison uses six accepted fresh offline processes: three explicit
tracing-enabled exports alternating with three disabled exports. The retained
fixture has 21 raw rows, three episodes and three tasks. The full public call
includes lazy imports, CSV/PDF construction, hashing and publication, while
excluding the harness and surrounding CLI work. Median full-call durations are
32.601570778060704 seconds with tracing and 7.047831253614277 seconds without it.
Their ratio is 4.625759273300128 and their difference is 25.553739524446428 seconds.
These observations use a fixed alternating order and shared cache history;
they do not establish a universal speedup, worst-case cost or native-pipeline
memory saving. Whole-worker RSS high-water marks remain separate from Python
allocation peaks and cannot be added to native-process peaks.

Each of the six exports passes artifact and lifecycle acceptance, with no
forced cleanup. Six CSVs per export match the accepted current-schema reference.
The selected enabled and disabled PDFs each receive independent inspection of
17 of 17 pages at 300 DPI. Their corresponding rendered PNGs are byte-identical;
their PDF bytes differ, without an attributed cause. The other five timed PDFs
have artifact checks, not independent visual inspection.

You can reproduce the offline arithmetic without exporting again:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 logs/tester-feedback/task-6/export-timing-runner/summarize_exports.py > logs/tester-feedback/task-6/export-timing-runner/offline-summary.log 2>&1
```

Exact worker arguments and dependencies are pinned in
`logs/tester-feedback/task-6/export-fix-3/commands.json` and
`source-manifest.json`. Full per-worker evidence is under
`logs/tester-feedback/task-6/export-timing-fix-3/`; the summary command, its
27-input hash inventory, complete output and final custody are under
`export-timing-runner/`. The reference-preparation export is excluded from the
six timing observations. Three accepted current-default full CLI quickstarts
precede the policy change; final-policy CLI comparisons remain a separate gate.

The measured policy makes `export_report(..., trace_allocations=False)` the
default. You can request the previous allocation measurement with explicit
`trace_allocations=True`. The exporter never resets or stops tracing owned by
its caller. Disabled or caller-owned tracing leaves
`report_build_python_tracemalloc_peak_bytes` null and the CLI displays
`Python allocation peak unavailable`. Null does not mean zero allocation.
CSV/PDF content and whole-process lifetime RSS metrics retain their existing
contracts. The focused policy tests cover omitted, explicit-disabled and
explicit-enabled arguments across caller ownership and success/failure, plus
both available/unavailable formatting cases. Their exact command and complete
log are retained under `logs/tester-feedback/task-6/export-policy/`.

## Verify the review pins

The following paths are relative to `logs/tester-feedback/task-10/` unless they
start with `.superpowers/`. Artifact hashes permit recovery without restating
the complete review reports.

| Artifact | SHA256 |
|---|---|
| `review-manifest.json` | `ea13ba72cd15f82818edb4e88fa5c65c55dde13dedbf91bb48b9d488f42daf3e` |
| `review-diff.patch` | `cea105bde5f7924610cbb087fe5a207c80a3e9c87ccfb56655c78f7be6a7f013` |
| `fix-1/scoped-fix-manifest.json` | `e1fcdcce2a82c4949e56dc0af7bcf73d0446926f84e889fefdaaa85c874bede3` |
| `fix-1/scoped-fix.diff` | `d2806794a9ff5c0fe5738e971dc5386943235548f28f111a4090f5d2b13bb36b` |
| `fix-2/scoped-fix-manifest.json` | `de7b5495e8acdb0c7fd96b14e17fe759eae97ccd37f25f9e095b8cc90cb93fb6` |
| `fix-2/scoped-fix.diff` | `c8e48712483e23776ca78debfe5e29a5419fd942fd411c3420442761871ea5ac` |
| `acceptance-fix-1/rendered-page-inventory.json` | `e36454de512c315f5f647569a5db7cf04b76a9e56ea423a7dce83f9dd48c0406` |
| `fix-1/evidence-manifest.json` | `4270487d1abf632ffd5d067fde9011412baf7ce3cfda835a4577de59fb80261c` |
| `evidence/custody.json` | `c9fbc7635a2eac91e7d6a74269885a1a8372742fef383ef22f4010647587a1ea` |
| `evidence/commands.json` | `870ff542da55f4224166373cfcc02c33a3fc001d23b38e9f6db8eb5ebf1b5147` |

The required scoped root prose check exits zero with 244 advisory alerts in
its guide/shared-standards population. Explicit Vale checks of this evidence
and the design record produce zero alerts after one correction pass. Full
commands, logs, the two-file diff and final hashes are in
`.superpowers/sdd/2026-10-19-tester-feedback-fixes/task-10-evidence-handoff.md`
and `logs/tester-feedback/task-10/evidence/`. No global prose-clean claim is made.
