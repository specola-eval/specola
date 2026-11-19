# Check the LIBERO installation and unread geometry catalogue

Date: 2026-10-19. Scope: implementation checkpoints `dd3dfad` (default queries
and discovery) and `86b5c14` (installation and Python diagnostics). These checks
precede the dashboard lifecycle and spec-function migration. They do not
establish acceptance for those later changes.

Use this evidence alongside the [initial quickstart measurements](tester-feedback-repro-2026-10-19.md)
and the [accepted decisions](../superpowers/specs/2026-10-19-tester-feedback-decisions.md).
Full local logs remain under `logs/tester-feedback/`. The command inventories
`default-commands.txt` and `install-commands.txt` contain the executed commands,
working directories, environment variables and output paths. The digest manifest
`install-discovery-evidence-sha256.json` identifies the retained evidence files.

## Default queries and catalogue coverage

The measured native case is `libero_goal:0`, with `target_only=True` and the
built-in 0.05 m cutoff. The environment is Python 3.12.13, MuJoCo 3.8.1,
hf-libero 0.1.4, robosuite 1.4.0 and Flask-SocketIO 5.6.1. The browser is
Google Chrome 153.0.8010.36. The source measurement uses base `3ef8520` plus
the production patch later committed as `dd3dfad`; the patch SHA-256 is
`3e1775b441432197123f9ec159d10bcd102de01e7e7eccf07e8aad0de6a82e4a`.

| Measured population or operation | Result |
| --- | ---: |
| Observed reset entities | 370 |
| Catalogue entities | 10,645 |
| Catalogue properties | 31,798 |
| Eligible pairs | 10,275 |
| Unread pairs | 10,275 |
| Exact eligible pair IDs rendered by Chrome | 10,275 of 10,275 |
| Native distance calls during reset | 0 |
| Native distance calls during catalogue build and serialization | 0 |
| Serialized JSON | 26,210,613 bytes |
| Compressed payload (`gzip`) | 461,217 bytes |

The native-call counter uses `sys.setprofile` with the original
`mujoco.mj_geomDistance` callable. A focused fixture demonstrates a
positive count when a requested pair is measured. Zero-query listing therefore
does not rely on replacing the native query with a mock.

The following timings come from one corrected native/browser probe. They are
observations, not a speedup comparison or a general upper bound.

| Operation | Duration |
| --- | ---: |
| Adapter construction and reset, with profiling enabled | 6.73634 s |
| Catalogue and UI model construction | 0.27764 s |
| Filtering and JSON serialization | 0.48131 s |
| Compression (`gzip`) | 0.06194 s |
| Discover fetch, indexing, expansion and detail after dashboard page load | 1,148.2 ms |

The browser interval begins after the helper finishes opening the dashboard
page. It excludes initial page navigation and cannot be called time to first
paint. It measures the subsequent Discover operation. The initial probe's
incorrect error-array name and ambiguous timer scope are corrected before
these measurements are accepted.

The detail view displays `not measured`, cutoff `0.05 m`, and an unread clipped
value. A specification authored from an unread pair retains its entity ID and
passes validation. A separate browser check selects the entity and signal,
inserts its pinned selector, enters a threshold, and validates through the
visible controls. Its preview also displays `not measured`. Both checks assert
an empty `window.__specolaErrors` array. The inspected screenshots are
`default-browser.png` and `default-browser-ui.png`; machine results are
`default-browser-result.json` and `default-browser-ui-result.json`.

The final owning test selection records 277 passes and three robosuite
deprecation warnings in `default-final-tests.log`. It includes explicit cutoff
overrides, lazy discovery with `population: all`, later requested measurements,
retained reset metadata after later observations/resets, and observed NaN versus
unread values. A no-queries replay records
`20 steps · 1.200 s simulator time · completion complete` in
`default-libero-replay.log`. This is a bounded replay, not the full demonstration.

These measurements cover the named LIBERO scene and MuJoCo build. They do not
measure larger RoboCasa catalogues or other simulator environments. Catalogue
serialization and DOM size still grow with the eligible population.

## Install the pinned simulator packages

Two isolated environments under `/tmp` use Python 3.12.13 and 3.10.20. Each runs
its own interpreter with `pip install '.[libero,web]'`, using the command
inventory's cache, retry and timeout arguments. Both installations succeed and
resolve hf-libero 0.1.4, robosuite 1.4.0 and MuJoCo 3.8.1. Both `pip check`
commands report `No broken requirements found.` The logs are `pip-libero.log`,
`py310-libero.log`, `install-py312-check.log` and `install-py310-check.log`.

| Check | Python 3.12.13 | Python 3.10.20 |
| --- | --- | --- |
| Install the pinned extra | Pass | Pass |
| Dependency consistency | Pass | Pass |
| Bounded native replay | 20 steps, exit 0 | Refused before scene construction, exit 1 |
| Doctor Python status | Matching profile | Required 3.12, running 3.10 |
| Final focused tests | 112 passes, including native guard | 87 passes |

The fresh environments resolve Flask-SocketIO 5.7.0. Their complete selected
versions and imported source path are in `install-py312-versions.json` and
`install-py310-versions.json`. hf-libero declares Python `>=3`; MuJoCo 3.8.1
declares `>=3.10`. The reviewed `CPython` 3.12 requirement is Specola's native
profile policy, not hf-libero's declared Python requirement.

The Python 3.12 check uses the documented replay with `--steps 20 --ui plain`.
It exits zero and records
`20 steps · 1.200 s simulator time · completion complete` in
`install-py312-replay.log`. The drawer goal is still violated at that bound;
this check does not claim task success or completion of the 138-state demo.
Existing simulator and unbound-contact warnings remain in the raw log.

The Python 3.10 launch records this error in `install-py310-replay.log`:

```text
Error       NativeImageError: Specola's reviewed MuJoCo 3.8.1 native profile needs CPython 3.12 (running 3.10); create the supported environment with `conda create -n specola-libero python=3.12 -y` and install the simulator extra there; see docs/install.md
```

The matching doctor line in `install-py310-doctor.log` is:

```text
libero     packages found; needs Python 3.12 (running 3.10)
```

Focused regressions cover the four shipped native profiles, inconsistent ABI
tags, absent metadata, explicit interpreter matches/mismatches, constructor
non-execution on mismatch and unchanged same-Python image-integrity errors.
The fake-scene fixture explicitly isolates its metadata/interpreter defaults;
its mismatch tests still exercise the real preflight.

The runtime commands set the worktree's absolute `PYTHONPATH`. These checks
establish resolution and runtime behavior in isolated environments, not
installed-wheel import independence, an empty asset cache or browser rendering.
Those checks remain separate acceptance work.

## Compare the broader test selection

The recorded baseline selection has 180 failures, 3,808 passes, 48 skips,
507 deselected tests and two expected failures. After the installation checkpoint,
the same marker selection has 146 failures, 3,869 passes, 48 skips,
507 deselected tests and two expected failures. Exact commands and full output are
in `pytest-baseline.log`, `install-commands.txt` and `install-core.log`.

Comparing failure identifiers gives 144 retained failures, 36 resolved failures
and two additional failures. This spans the preceding integrated changes;
resolved failures are not attributed solely to the installation change. The
selection is not green. Marker exclusions also do not prove that no selected
test imports a native dependency.

The two additional tests pass together in isolation. Subsequent bounded
diagnosis reproduces a module-isolation issue after successful mock certification
and a Textual button-timing issue. Exact diagnostic commands are in
`diagnose-commands.log`; complete results remain in `diagnose-*` logs. The
relevant production behavior predates these changes. Test corrections and the
final broader acceptance run remain pending at this checkpoint; isolated
success does not erase the recorded suite failures.
