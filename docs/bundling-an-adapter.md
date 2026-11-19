# Bundling an adapter

A bundled adapter is a module under `src/specola/adapters/` registered in
this repository's `pyproject.toml` entry-point group. It ships to every
Specola user, so it meets more than the contract in
[Adding an adapter](adding-an-adapter.md): a capability matrix row, the
portability rules in [Adapter hooks](adapter-hooks.md), the review gates
below, and the tests and manual checks that decide when it is done.
Develop the adapter as your own module first; bundle at the end.

## Before any code

### Fill the capability matrix row

Add a row for your adapter to `docs/adapter-capabilities.md` before you
write code; the row is part of the change. The matrix lists every signal and
hook an adapter can carry: body position, orientation, linear and angular
velocity, joint state, wrenches, bounding boxes, pair distances and their
query binding, the drive, scene-geometry, frame-capture and label hooks.
For each cell write one of:

- `yes`, with the producer surface you read: package, version and the
  attribute or function (`<package> <version>` `<Class.attribute>`);
- `n/a`, with the reason the simulator has no such signal.

The matrix also defines `gap` (exposed by the simulator, not exported
yet); older rows carry it, a new row does not. Cite evidence by symbol
name, not line number. The columns are robotic; a simulator whose
signals fit none of them (an economic model, a queueing simulator) marks
the physical columns `n/a` with the reason and adds a paragraph under the
table listing every key it exports, with group and unit. Column J also
records the replay and state-tier status, as the existing rows do
(`replay refused, no state tier` with the reason when the simulator has
no state save and restore).

Silence is not allowed. A capability the simulator exposes and the adapter
omits is a defect, whatever the sibling adapters export. The shared key
vocabulary (`body_pos_w_<link>_<axis>`, `joint_pos_<joint>`,
`link_wrench_<link>_<i>` and the rest) is a floor for spec portability,
never a ceiling: an adapter exports what its simulator has (maintainer,
2026-09-24, after an adapter shipped without link velocities because a
sibling had none, and a progress specification had nothing to read).

Read the version-matched producer surface first: its documentation, then
the installed source. A `yes` without a test and a producer source is a
`no` in disguise. Where the simulator documents no unit for a quantity,
a unit derived from its equations is acceptable evidence if the module
docstring marks it derived and states every premise the derivation rests
on (a parameter's own unit included), and says so where a premise rests
on nothing the simulator states, recording it as pending; a unit taken from the model
publication the simulator implements counts as quoted, with the citation.
A quoted unit and a derived one are never presented alike: put the plain
unit in `PropertyMeta.unit` (unit checks and displays read it) and mark a
derived one in `note` (`"mg/kg (derived)"`).

### Settle shared semantics with the maintainer first

Before you touch anything other adapters and specifications share (the key
vocabulary, entity classes, group labels, binding keys, query bindings),
write every open semantic decision as a numbered list with a proposed
default and wait for the maintainer's answers. A group label or key shape
is published the moment a specification can select it.

An agent that cannot reach the maintainer records the list in the change
(a `QUESTIONS.md` at the worktree root, or the pull request description),
proceeds with each default, and leaves the change unmerged until the
maintainer has answered. The same holds for the gates below that name a
maintainer decision (enrollment exclusion, the verification profile):
implement the default, label it pending, and do not treat the label as
the authorization.

### Pick a reference adapter

[Adding an adapter](adding-an-adapter.md) says which base class to use.
The production adapters to copy from:

- `MujocoAdapter` subclasses: `adapters/robocasa.py` is the full
  production example, `adapters/dmcontrol.py` shows generic
  kinematic-tree classification, and `adapters/mamujoco.py` shows reusing
  another adapter's classification (`dmcontrol.kinematic_root_rules()`)
  when several simulators share one model. The base also gives a bundled
  subclass region materialization, the bounded geometry provider and the
  state tier. The subclass extension points are in
  [MuJoCo subclasses](adapter-hooks.md#mujoco-subclasses).
- Direct `SimulatorAdapter` implementations: `adapters/dataset.py` is an
  offline reader, `adapters/carla.py` the full production example with a
  dynamic actor population and a drive hook, `adapters/pybullet.py` a
  direct adapter with a state tier (its `close` override predates the
  rule in the hook table; follow the table), `adapters/isaaclab.py` and
  `adapters/robodojo.py` direct adapters over a vectorized simulator with
  the 3D-panel and function-input hooks.

### Follow the portability rules

Every rule in [Rules that keep specifications
portable](adapter-hooks.md#rules-that-keep-specifications-portable) binds
a bundled adapter: key naming, metadata
parity, groups and units, name collisions, instants and window means, and
the semantic boundary between state and verdicts. Implement
`property_metadata`, which `discover` reads to label every key, and a
closed `supported_classes`: with an open class vocabulary the
bindability guard (`tests/test_adapter_bindability.py`) only warns and
probes nothing.

## Pass the review gates

These gates are silent until they fail, so do them together, in one
commit, after the adapter works as your own module.

1. **Enrollment** (`src/specola/adapter_enrollment.py`). The path depends
   on the base class. A `MujocoAdapter` subclass is enrolled in
   `_REVIEWED_IMPLEMENTATIONS` (module path to adapter class and factory
   names) and installs the shared provider. A direct `SimulatorAdapter`
   has no provider, so the only path that runs today is the exclusion
   lists below, applied as the default and marked pending for the
   maintainer. In detail: Every adapter reset calls
   `guard_extraction_source` (`conformance/provider.py`), which reconciles
   the installed entry points against this table; a registered module
   missing from it raises `EnrollmentError: missing reviewed
   implementation enrollment: <name>` on every reset of every adapter in
   that install, and `tests/test_adapter_enrollment.py` fails. An enrolled
   row needs an extraction provider: MuJoCo
   subclasses install the shared one (`install_bounded_provider`, see [MuJoCo subclasses](adapter-hooks.md#mujoco-subclasses));
   a direct adapter has none today, so it is refused with
   `bundled_row_pending`. For a direct adapter the maintainer decides
   between writing a provider and listing the adapter in
   `EXTRACTION_SCOPE_EXCLUSIONS` and `_EXCLUDED_IMPLEMENTATIONS` instead,
   which runs it on the legacy capability wire and removes it from the
   extraction programme rather than certifying it (as `mock`, `dataset`,
   `robodojo`, `pybullet`, `carla` and `isaaclab` are; the last three
   moved there 2026-10-13 with certification open in #111, #112, #107).
2. **Capability declaration.** Add the entry-point name to `DECLARATIONS`
   in `tests/test_capability_declarations.py`. Declare each capability
   family the adapter supports (`rigid_body`, `articulation`,
   `deformable_state`, `contact`, `physical_sensor`) against at least one
   existing test that exercises extraction for that family. A family the
   adapter accepts scenes for but does not extract goes in
   `ACCEPTED_BUT_INCOMPLETE` with a test asserting the incompleteness
   marker. A simulated sensor counts as `physical_sensor`. An adapter that
   declares no physical family uses the table's `NO_PHYSICAL_CAPABILITIES`
   marker, as `dataset` does for signal passthrough. Declare the families
   that fit even when they cover only part of the simulator (a simulated
   sensor on a model whose state is none of the families), and record the
   unrepresented part as pending.
   A simulator that can be configured with sensors or observation types
   the adapter does not extract (a LiDAR, an occupancy grid) refuses those
   configurations in the factory with a message, or declares the family
   in `ACCEPTED_BUT_INCOMPLETE` with the marker test; accepting them with
   neither is the failure the table exists to prevent.
   The `test_<sim>_producer_surface.py`, `test_<sim>_source_inventory.py`
   and `test_<sim>_certification_row.py` modules some adapters have belong
   to the extraction certification programme, not to bundling; an adapter
   on the exclusion list does not need them.
3. **Bindability guard wiring.** `tests/test_adapter_bindability.py`
   covers a closed-vocabulary adapter automatically, probed with
   `scene=None`. Add a `_SCENES` entry when the factory needs a task, a
   `_PRODUCER_MODULES` entry when the simulator package is optional in the
   test environment (its absence is the only thing that skips), a
   `_PRECONDITIONS` entry for a launcher or checkout the import cannot
   express, and a `_MARKERS` entry if the adapter has a pytest marker.
4. **Marker and skips.** Add a pytest marker for the simulator in
   `pyproject.toml` next to the existing ones and gate simulator-backed
   tests with it. The marker does not skip by itself; each test module
   also `pytest.importorskip`s the simulator package, as
   `tests/test_mujoco_sensors.py` does.
5. **Installation verification.** `specola doctor --verify` reports an
   adapter without a bounded profile in `install_verification.PROFILES` as
   `unsupported`. The default is to document in `docs/install.md` that
   installation verification does not cover the adapter, marked pending;
   adding a profile is the maintainer's decision.
6. **Install extra and docs.** Add a `pyproject.toml` extra pinning the
   simulator package (every bundled simulator has one), a row and setup
   section in `docs/install.md`, and the simulator with the versions it
   was run with in the README's "Bundled adapters" table. The install
   table's fresh-install check is the maintainer's to run in a new env;
   mark it `not run`. `docs/surface-parity.md` is generated by
   `benchmarks/parity_audit/` and is not edited by hand; `docs/running.md`
   lists the adapters with a state tier and changes only if yours has one.
7. **Example specification.** Add `examples/<name>_specs.yaml`, whose
   specifications bind on the default task; `specola check` on it is part
   of the done checklist.
8. **Stale-source window.** Any `src/` change stales the
   diagnostic-catalogue pins: add a `date:` / `checkpoint:` / `reason:`
   entry to `benchmarks/diagnostic_catalogue/STALE_SOURCE.md`, as a bare
   block at the top of the file (newest first), rather than regenerating
   them.

## Environments

Run each simulator in its own conda env, named after it, with Specola
editable-installed into each; simulators pin conflicting dependencies, so
never co-install two. Entry points are read from the installed
distribution's metadata, not from `pyproject.toml` on disk, so a new
bundled entry point is visible only after `PYTHONNOUSERSITE=1 pip install
-e .` in each env, run from the main checkout once the change is merged
there. From any checkout other than the one the editable install points
at (a git worktree or a separate clone; `pip show specola` names it), put
that checkout's `src` first on `PYTHONPATH`
and load the adapter with `--adapter-module` until then; never
`pip install -e` from a worktree, which repoints the env at it.

The simulator's env also needs the test and dashboard dependencies: the
`dev` and `web` extras of `pyproject.toml` (pytest, hypothesis, pyarrow,
Flask and Socket.IO). Without Flask, `--web` prints `Web warning:
ModuleNotFoundError: No module named 'flask'; continuing without
dashboard.` and exits 0, which a scripted check can mistake for a pass.

To run the bundled gates from a worktree before the merge, give Python a
metadata-only copy of the installed distribution whose entry points come
from the worktree's `pyproject.toml`, and put it ahead of everything else
on `PYTHONPATH`; nothing in the env changes. Run it from the repository
root (it reads `pyproject.toml` by relative path):

```bash
SHIM=$(mktemp -d)
python - "$SHIM" <<'EOF'
import importlib.metadata as m, pathlib, shutil, sys, tomllib
shim = pathlib.Path(sys.argv[1])
installed = pathlib.Path(m.distribution("specola")._path)
target = shim / installed.name
shutil.copytree(installed, target)
groups = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["entry-points"]
(target / "entry_points.txt").write_text("".join(
    f"[{g}]\n" + "".join(f"{k} = {v}\n" for k, v in items.items()) + "\n"
    for g, items in groups.items()))
EOF
export PYTHONPATH="$SHIM:$PWD/src"
```

With it, `--sim <name>` resolves, `specola doctor` lists the adapter, and
the enrollment, capability-declaration and bindability tests see it.

## When the adapter is done

Mechanical checks, in the simulator's env, with `PYTHONPATH=<checkout>/src`
first when working from a worktree and the metadata copy above. First
the three commands in [Check that it
works](adding-an-adapter.md#check-that-it-works), with `--sim <name>` and `--web` on `run`; then:

```bash
specola doctor                             # lists the adapter, packages found
pytest tests/test_adapter_contract.py tests/test_supported_classes.py \
       tests/test_adapter_bindability.py tests/test_capability_declarations.py \
       tests/test_adapter_enrollment.py
pytest tests/ -m <your marker> --continue-on-collection-errors  # the adapter's own tests
pytest tests/ -q --continue-on-collection-errors   # compare with the base commit
```

Run the whole suite once on the base commit in the same env and compare
failure sets: a simulator env lacks other simulators' packages, and some
modules fail to collect there for reasons unrelated to your adapter, so
only a difference against the baseline is yours (hence
`--continue-on-collection-errors` on every suite command). Check out the
base commit in a separate worktree or clone for that run, rather than
extracting a tree, because some tests read git metadata. Add `-rfE` so
pytest lists failed test IDs, and run the comparison on a committed tree:
some tests refuse an uncommitted one and would show as new failures.

Your own tests cover, at minimum: observation-wide metadata parity across
every class the adapter can produce; unit homogeneity per group; a
sim-free vocabulary pin (`object.__new__(cls)` plus the attributes the
method reads, as `tests/test_supported_classes.py` does); name-collision
behaviour; a dynamic-key sequence and a mid-episode entity if the
population can change; a guard that no verdict or scoring quantity is
read, installed before `reset()` so the reset read is covered too, and
covering the attributes where the environment caches such values as well
as the functions that compute them; a test that scoped-out
keys are not read at all (make their native read raise), since the base
class's projection hides a missing prune from any test that only looks at
the scoped observation; for a drive hook, one test per wrapper kind over
the real simulator (an episode limit, an action rescaling, a pass-through
wrapper, and a subclass of the framework's automatic-reset wrapper), each
asserting that the wrapper still applies or is refused and that the
terminal observation survives; a pass-through wrapper's test counts the
calls that reach the wrapper, so a drive that bypasses it fails; for
`render`, tests that a pre-existing viewer survives `close`, that a viewer
the adapter created is released even after a native reset, and that
process-wide state is restored; for `close`, a spy that
asserts what was closed and what was left to its owner; and one
capability test per `yes` in the matrix row. Value tests assert every
exported key's value in the reset observation, one assertion per key, as
well as later steps, and compare against an oracle
the adapter does not read (the value the environment returns to its
caller, or an identity of the model), not against the attribute the
adapter reads, which would agree with a wrong label. Every value test must fail
on a wrong value: break the extraction on purpose (zero a key, swap two)
and confirm a test fails, including breaks that touch only the reset
observation, and exercise the conditions that make a value
non-trivial, choosing fixtures in which every asserted value is non-zero
and differs from its neighbours (an input event for a state it drives, contact for a force). Where a
listed test is vacuous for your simulator (no name can collide in a fixed
vocabulary), write the test that pins why it is vacuous instead. MuJoCo-family logic is
testable without the simulator package: build a tiny model inline with
`MjModel.from_xml_string(...)`, as `tests/test_mujoco_property_meta.py`
does.

Manual checks, none of which an assertion replaces. An agent may carry
them out with browser automation and image reading; record in the change
that an agent, not a person, made them. Without any browser, run the
headless dashboard auditor (`benchmarks/dashboard_live_check.py`, which
attaches to `specola run --web` as a socket.io client and checks every
bundle against the wire contract; it resolves `--sim` through entry
points, so it needs the installed adapter or the metadata copy above, and
it drives the default `random` mode, and on a simulator that steps
faster than a client attaches it counts dropped bundles against a budget
meant for slower scenes: raise `--max-delivery-drops` and record why) and replay a finished recording with
`specola view` (a client asks for playback with the `view` event's
`seek`, `play` and `rate` fields, `_apply_view` in
`src/specola/dashboard/server.py`; playback waits `min(dt / rate, 1 s)`
per step, so a simulator with a long step replays at one step per
second at most rates), and record the rendered-page pass as
not done.

- `specola discover` output is enough to write a specification against the
  adapter without reading its source: every key labelled, every class
  discoverable, units and groups guessable.
- A saved frame has been opened and looked at, beside the simulator's own
  render of the same state.
- `specola run --web` has been driven in a browser: every dropdown opened,
  the camera panel and the 3D panel checked (or their absence confirmed
  where the adapter declares none), and replay of a finished recording
  exercised as well as the live path. For a simulator whose episodes end
  in seconds, pick a configuration that runs long enough for the page to
  attach (a task or seed with a longer episode), so the live path
  is seen and not only the post-run view. This pass is unconditional, cameras
  or not: the dashboard consumes every adapter's bindings, labels and
  traces, and a pass first skipped "because the adapter has no visual
  surfaces" surfaced two real gaps.

Review the diff before the first commit of adapter code, not only at the
end (`/code-review` in Claude Code, or the equivalent). The review must
look at the repository you changed: Claude Code's `/code-review` with no
target reviews the session's working directory, which may be another
checkout, so when they differ give the reviewer the diff
(`git diff <base>..HEAD`) and the repository path explicitly. Then record what you learned that the next adapter author
needs in `MISTAKES.md` as a rule, not an incident.

## Further reading

- `adapter-capabilities.md`: the matrix every adapter has a row in.
- `src/specola/adapter.py`: the normative contract.
- `examples/custom_mujoco_adapter.py`: the annotated MuJoCo template.
- `superpowers/specs/2026-07-15-extract-everything-design.md`: why the full
  per-DOF, body, sensor and actuator signal set is derived this way.
- `superpowers/specs/2026-08-10-libero-spec-faithfulness-v2.md`: an
  evidence-driven example of the relation-input design, not a source of
  simulator-generic names.
- `ROADMAP.md`, `FUTURE-DIRECTIONS.md`: planned adapters and the deferred
  surfaces of landed ones.
