# Check the installed CLI and dashboard

Evidence checkpoint: 2026-10-20. The final seven-entry matrix has four accepted
entries, two unaccepted native entries and one unrun RoboDojo entry. Source and
installed-wheel mock verification passed. Whole Task 11 acceptance remains
incomplete; Task 8 owns the native browser scrub, native shutdown and pending
RoboDojo integration gates described below.

`specola doctor --verify` runs a deterministic mock fixture and selected
bounded simulator profiles in the invoking Python environment. Ordinary
`specola doctor` keeps its package inventory behavior. You can select a native
profile with repeated `--sim NAME` arguments; the mock checks still run.
Without a filter, verification selects adapters whose current-environment
inventory is ready. Inventory readiness does not establish native readiness.

```bash
specola doctor --verify --sim mock --output-dir /tmp/specola-check-new
specola doctor --verify --sim libero --demo --timeout 360 --output-dir /tmp/specola-libero-check-new
```

Install the `verify` extra and Chrome or Chromium for browser checks. The
verifier also checks Poppler's `pdfinfo`, `pdffonts`, `pdftotext` and
`pdftoppm` before PDF inspection. Missing selected prerequisites produce
`not-run`; missing profiles produce `unsupported`. A required result other
than `pass` makes verification exit with status 1. Unsupported camera and
scene hooks on the mock and other declared profiles are explicitly optional.
An existing evidence directory is rejected so earlier failures remain intact.

You receive `results.json`, full child command and process logs, screenshots,
rendered PDF pages and artifact hashes. Results distinguish automated PDF
parse, font and page-count checks from human page inspection. Each rendered
page starts with `visual_reviewed: false`. Screenshots retain the evidence
needed to inspect object identity, scene appearance and chart readability.
Automated chart checks measure drawn text boxes and reject overlapping labels.
They do not infer readability from canvas dimensions.

## Scope and native prerequisites

The enumerated CLI checklist is `discover`, `check`, `run`, `write-specs`,
`report`, `view` and `doctor`. The packaged mock fixture checks five native
mock entities, two bindings per authored specification, known pass/fail
outcomes and function robustness values 0.05 and 0.35. Its literal names
contain brackets and a backslash. Native profiles use declared tasks and drive
modes from `install_verification.PROFILES`; they do not establish coverage of
other tasks. RoboDojo uses the existing sampled mesh function with its 0.05 m
cutoff. Other ordinary native smoke profiles use an explicitly labelled finite
signal, which is separate from benchmark task success.

`--demo` selects the packaged LIBERO goal-0 demonstration and its current six
specifications. `--demo PATH` selects an existing local demonstration for the
same task. The packaged provenance and Apache 2.0 license accompany the data.
Verification does not download demonstration data, generate LIBERO config or
search sibling simulator environments.

The LIBERO preflight follows the inspected `hf_libero` 0.1.4 source. Its actual
asset resolver checks package-local assets before `~/.cache/libero/assets`;
the YAML `assets` field alone cannot establish resolution. The preflight
requires the selected task's immediate XML, BDDL and initial-state files and
sets `HF_HUB_OFFLINE=1` before child imports. The flag's documented semantics
come from [Hugging Face Hub 1.25.1](https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.25.1/docs/source/en/package_reference/environment_variables.md).
These file checks do not establish recursive mesh or texture completeness.
Configured relative asset paths are checked against the child working
directory. The config-directory override is resolved before isolation.

In that installed version, the fallback download helper returns before its Hub
call only when four directories exist under `~/.cache/libero/assets`:
`articulated_objects`, `stable_scanned_objects`, `turbosquid_objects` and
`stable_hope_objects`. Its fifth listed category, `scenes`, is outside that
early-return slice. The verifier also requires scenes and eight immediate XML
entry files for `open_the_middle_drawer_of_the_cabinet`: the tabletop scene,
wooden cabinet, flat stove, wine rack, bowl, cream cheese, wine bottle and plate.
These are presence checks, not recursive dependency validation. `LIBERO_CONFIG_PATH`
selects a directory containing `config.yaml`; a missing file can invoke an
interactive initializer. The verifier requires existing configuration instead.
The installed initializer and download-helper source, preserved with hashes
under `logs/tester-feedback/task-11/native-prerequisites/`, establish these
release-specific details. The [upstream initializer](https://raw.githubusercontent.com/huggingface/LIBERO/main/libero/libero/__init__.py)
corroborates the flow but does not identify the installed release. Hub 1.25.1
reads its offline flag at import, so it must be set before importing LIBERO.
That flag controls Hub requests, not arbitrary network activity in dependencies.

The RoboDojo profile requires the existing `arx_x5`, seed-family 0, layout
ordinal 0 `stack_bowls` configuration and immediate scene files. Its checks
follow checkout `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`; Kit still has to
resolve the actual asset graph. Other configurations remain outside this
bounded profile. Full source and package evidence for these preflights is
retained locally under `logs/tester-feedback/task-11/native-prerequisites/`.

The layout selector numerically sorts matching saved-layout filenames and
uses their ordinal position. An ordinal need not equal a filename suffix when
suffixes have gaps. The inspected default ordinal 0 selects
`Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json`, containing bowl0,
bowl1 and bowl2. Presence of its USD, MDL and HDR entries does not establish
that Kit can resolve the full asset graph. Staged reporting support for this
environment does not imply a conda dependency change or acceptance of the full
RoboDojo profile.

## Browser and process contract

Runtime checks recorded Chrome 153.0.8010.36. The version-matched
[Target protocol](https://raw.githubusercontent.com/chromium/chromium/153.0.8010.36/third_party/blink/public/devtools_protocol/domains/Target.pdl)
defines target creation and closure, while the
[Page protocol](https://raw.githubusercontent.com/chromium/chromium/153.0.8010.36/third_party/blink/public/devtools_protocol/domains/Page.pdl)
defines navigation, reload and screenshot commands. Closing a debugger socket
does not close its page. The verifier closes actual targets, observes their
disappearance, keeps another viewer connected, and independently checks final
server exit and port rebinding. Navigation errors and application readiness
are checked separately from command acknowledgement.

Chrome uses a fresh private `--user-data-dir` and `--remote-debugging-port=0`;
the verifier reads that profile's `DevToolsActivePort`. Chrome 136 and later
require a non-default profile for normal remote debugging, as documented in
the [Chrome announcement](https://developer.chrome.com/blog/remote-debugging-port).
The inspected [Chrome 153 server](https://raw.githubusercontent.com/chromium/chromium/153.0.8010.36/chrome/browser/devtools/remote_debugging_server.cc)
binds the normal port route to IPv4 or IPv6 loopback. websocket-client 1.9.0's
[handshake code](https://raw.githubusercontent.com/websocket-client/websocket-client/v1.9.0/websocket/_handshake.py)
supports `suppress_origin=True`; this client does not need wildcard origin
allowance. CDP receive loops enforce an overall monotonic deadline as well as
a socket timeout, so unrelated events cannot extend a command indefinitely.

The local Linux acceptance coordinator is separate from the packaged verifier.
It enables `PR_SET_CHILD_SUBREAPER` before launch, records owned descendant
identities across sessions, and applies finite TERM, KILL and reap deadlines.
The [Linux subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)
permits orphan adoption; it does not terminate processes. `Popen` owns the
command leader's wait status, and explicit waits reap adopted children.
An unrestricted second waiter can consume the leader's status; the inspected
[CPython 3.11.15 implementation](https://raw.githubusercontent.com/python/cpython/v3.11.15/Lib/subprocess.py)
can turn an unavailable status into zero. Cleanup must preserve one wait owner.

The [process-stat interface](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)
supplies parent, group, session and start time. The parser reads the whole
record and separates the final closing parenthesis before parsing fields;
command names can contain whitespace or parentheses. Ownership roots are the
dedicated coordinator and still-matching stored identities. A reused numeric
leader PID is not an unconditional root or wait exclusion. PID plus start time
is an observational check, not an atomic process handle. Fresh group membership
checks reduce stale-target mistakes but leave a check-to-signal interval.
Forced outer cleanup rejects acceptance even when the functional child exits
zero. The measured host kernel is 6.8.0-138-generic; research cross-checked
upstream Linux 6.8 source and man-pages 6.19. This is a bounded local harness,
not isolation of hostile processes.

## Final measured matrix

The source manifest covers 451 paths, with SHA256
`07b9b720769b123fd5735c20f42f6b23ea5c8ad542d097d0d399356e6676219f`.
Source review closed at fix round 5 with scoped specification and quality
passes. Native runtime failures below remain acceptance failures despite that
source-review result. Six of seven distinct matrix entries ran; the separate
interrupted LIBERO attempt does not increase that denominator.

| Entry | Final result | Measured limit |
| --- | --- | --- |
| Source mock/browser | Accepted, 79.422 s | Mock functions, literal authoring, complete/stopped reports, full histories, scrubbing, refresh, multiple viewers and final-tab close/rebind. Camera and scene are optional unsupported mock capabilities. |
| Source LIBERO replay/browser | Not accepted, 222.813 s | Replay/report and PDF checks pass; the first completed-run camera scrub times out. Later native browser stages are unrun. |
| Source RoboDojo functions/recording/browser | Not run | Existing native function benchmarks do not substitute for this integration profile. |
| Native USD first-recorder export | Functional test passes; acceptance rejected, 12.412 s | One pytest case passes with zero skips/failures and matching recording hashes; an adopted child requires outer TERM after driver exit. |
| Current wheel build | Accepted, 1.254 s | One CPython 3.12 Linux wheel built from the frozen source. |
| Isolated wheel install | Accepted, 13.621 s | The wheel and its `verify` dependencies installed into the temporary venv. |
| Installed-wheel mock/browser outside checkout | Accepted, 75.781 s | Mock workflows and five published reports pass with `PYTHONPATH` unset. Native wheel behavior is outside this result. |

These times are coordinator elapsed times, including orchestration. They are
not `report_build_seconds`, which excludes final hashing and manifest
publication. Source mock used Python 3.12.13, pandas 3.0.5 and websocket-client
1.9.0. Installed mock used Python 3.12.13, pandas 3.0.6, matplotlib 3.11.2,
websocket-client 1.9.2 and Pillow 13.0.0. Both recorded Chrome 153.0.8010.36.

The wheel is `specola-0.2.0-cp312-cp312-linux_x86_64.whl`, SHA256
`eb31cbe522dfc03f5aaf34496c0470dcc1beb1e5e5e7306550b293636d2bc51c`.
Verification ran from `/tmp/specola-verifier-wheel-r0ju6hqj`, with implementation
paths beneath `venv/lib/python3.12/site-packages/specola/`, including
`install_verification.py` and `verification_browser.py`. The original output
is `/tmp/specola-verifier-wheel-r0ju6hqj/acceptance/fix-5-retry-1/mock-1/`;
a retained copy is under
`logs/tester-feedback/task-11/fix-5-retry-1/runtime-runner/installed-wheel-evidence/`.
No simulator environment package changed during this installation.

The LIBERO run checked six current specifications over 138 observation times
and steps and confirmed changed non-time simulator state. Six report outcomes
are completed/pass; native task success remains unknown. Its browser recording
has 138 snapshots and 138 frontview JPEGs, and the failure screenshot shows
camera and 3D output at a displayed step 138/138. That image does not prove
receipt and painting of the requested scrub response. A source-reachable
view-acknowledgement/image-decode race can prevent the harness probe from
publishing an accepted observation, but the actual timeout cause is unknown:
the failed session did not retain probe, epoch, sequence and frame-identity
operands. The log's `last browser evaluation error: []` is not a dump of
application errors. It does not identify the failing predicate.

The independent USD case exercised an authored mesh change before first
recorder export. Its five recorded artifact hashes matched. Worker exit zero
did not override `forced_cleanup: true`: one adopted child remained alive and
received TERM. Its command/name were not captured, so its role is unknown.
Final checks found no survivors among that entry's five observed identities.
The separate continuation's four entries ended with no survivors among 60
observed identities and 451 matching source hashes. The earlier interrupted
LIBERO attempt retains an unknown incoming signal/sender and cause; outgoing
cleanup signals do not establish what initiated it. It is not classified as
a demonstrated production or simulator defect.

Task 8 carries the LIBERO scrub gate, native shutdown gate and pending full
RoboDojo profile. The controller parked Task 11 at its five-round cap instead
of authorizing a sixth source fix. This is a recorded continuation boundary,
not whole Task 11 acceptance or a waiver of native integration requirements.

## Visual review populations

Human inspection and generated automated flags remain separate. Generated
`visual_reviewed: false` fields were not rewritten to represent the review.

| Population | Individually inspected | Result and boundary |
| --- | --- | --- |
| Final source mock, fix5 | 30 PDF page images and five browser screenshots | Layout/readability pass. Six retained browser phases match full 12-step complete or 11-step stopped histories; the earlier partial-history finding is closed for these measured paths. |
| Current native LIBERO, fix5 retry1 | 25 PDF page images and one failure screenshot | Layout/readability pass for these images. Thirteen trace groups reconcile with 1,541 CSV rows. No successful native scrub or full browser workflow is implied. |
| Installed-wheel mock, fix5 retry1 | 30 PDF page images and five browser screenshots | Layout/readability pass for the installed wheel's own output. Twelve trace groups contain 132 CSV rows: 12 per complete trace and nine per stopped trace. Six browser phases match full histories and strokes. |

The original fix3 LIBERO diagnostic inspected a different 25-page report and
remains a distinct population. Earlier pages were not substituted for current
pages. Final source-mock browser screenshots select the positive chart; negative
values are visible in the tree and negative PDF panels were inspected, but a
negative browser-chart screenshot is not claimed for that batch. Source-mock
trace comparison accounts explicitly for tracker steps beginning at 1 and CSV
steps beginning at 0.

The installed-wheel comparisons apply the same explicit step conversion.
Its 35 inspected images and 70 selected retained artifacts are under
`logs/tester-feedback/task-11/final-visual-review/fix-5-retry-1/wheel-mock/`.
Installed implementation hashes were checked at their five recorded
site-packages paths. Source mock, current native and installed mock are three
distinct current populations; historical diagnostic images are not added to
their acceptance counts.

## Reproduction and custody

The following commands describe the executed interfaces; existing output
directories must not be reused. The reviewed coordinator inventories retain
the exact absolute argv, cwd, owned caches, environment, unsets and bounds.

```bash
PYTHONNOUSERSITE=1 python -m specola.cli doctor --verify --sim mock \
  --timeout 180 --output-dir /tmp/specola-mock-fresh --json
PYTHONNOUSERSITE=1 MUJOCO_GL=egl python -m specola.cli doctor --verify \
  --sim libero --demo --timeout 360 --output-dir /tmp/specola-libero-fresh --json
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /tmp/specola-verifier-wheel-r0ju6hqj/venv/bin/python -m specola.cli doctor \
  --verify --sim mock --timeout 180 --output-dir /tmp/specola-wheel-fresh --json
```

These are reproduction commands, not authorization to bypass native acceptance
supervision. Reporting needs writable absolute `MPLCONFIGDIR` and
`NUMBA_CACHE_DIR`; installation also uses an owned `PIP_CACHE_DIR`. Native
profiles run sequentially. The local coordinator validates fresh output,
selected profiles, explicit completion, required outcomes, report hashes and
source custody independently of worker exit status.

Paths below are relative to `logs/tester-feedback/task-11/`:

- `fix-5-source-manifest.json`: the frozen 451-path manifest pinned above.
- `acceptance-commands.json`: original final inventory, SHA256
  `709cee4312ec1f17863615655749bacb38e8139a0ff11d0d86afcd5caef18d81`.
- `fix-5-retry-1/acceptance-commands.json`: reviewed operational path-only retry,
  SHA256 `842d0d38a10650e758507160e3262289e83a12ce9ea49fc9ae7c94c7a058739e`.
- `acceptance-coordinator.py`: SHA256
  `448974d0212c8448f600d0f9c94be16f44b4dc0d79d0d8efefa5906492760e14`.
- `native-order-driver.py`: SHA256
  `b65c106712a2081169dac06e9c3034f943f57fb3d52b3a4b2dc1059038e73ebc`.
- `fix-5/runtime-runner/` and `fix-5-retry-1/runtime-runner/`: exact launches,
  summaries, before/after custody, complete logs and artifact inventories.
- `final-visual-review/`: separately pinned inspected images and CSV/history
  comparisons for each visual population.

The coordinator and native-order driver are versioned source harnesses at those
retained paths. Focused tests import them directly, so a fresh checkout requires
both files. Generated logs, recordings, wheel files and browser profiles are
outside that two-script source inventory.

The frozen manifest and wheel predate this final evidence-only document update.
Their recorded document digest is
`0628d54b4d1d00383ea81c3e67d3d9fd0f5bcae91786533116bebe526ebeb342`.
Do not regenerate old manifests or relabel their artifacts. This updated
document is hashed separately in the evidence-finalization record; production,
tests and harness source remain unchanged by that documentation step.

## Review boundary

Task 11 adds a narrow RoboDojo snapshot fix: mesh export freezes requested
function operands before releasing the mutable collected rows. Retained
observations remain immutable, and the adapter does not retain a second
collected-row copy after export. A simulator-free test covers unchanged and
invalidated call orders. A separate native USD test changes mesh points before
the first actual recorder export and checks the next function snapshot and
saved recording. That native test remains separate from the earlier eleven
Task 9B cases. Live replacement of geometry after the recorder's first export
is outside this fix; the recorder still publishes its initial scene payload.

Focused tests, native functional cases, browser acceptance, process cleanup,
installed-wheel acceptance and human visual review remain separate evidence
populations. The matrix above reports the measured boundaries rather than
promoting one population's pass to another.
