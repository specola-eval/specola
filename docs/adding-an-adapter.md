# Adding an adapter

An adapter projects one simulator's state into Specola observations. This
guide covers the two ways to ship one: as your own module, and bundled
inside this repository. Start with your own module; bundle at the end.

## The contract

Before the contract, the capability matrix: `adapter-capabilities.md`
lists every signal and hook an adapter can carry, one row per adapter.
Fill the row for yours from the simulator's own documentation and source
first, marking each capability either supported (with its source) or not
applicable (with the reason), then implement every supported cell. The
shared key names are a floor for spec portability, not a ceiling: an
adapter exports what its simulator has.

Read the `SimulatorAdapter` docstring in `src/specola/adapter.py` before
you write code. It is the normative statement of what an adapter may and
may not do: which fields you include, which you omit and why, and the rule
that success predicates, rewards, and task verdicts never feed extraction.

The minimum a subclass supplies:

- `super().__init__()` in the constructor;
- `dt` and `spatial_dims` properties;
- `_reset_impl()` returning the first observation of an episode;
- `_get_observation_impl()` returning the current observation;
- stable entity IDs within one episode;
- scalar simulator data exposed through `EntityState.properties`.

Both observation hooks return a `MonitorObservation(timestep, time,
perceived_entities)` where `perceived_entities` is a dict from integer
entity ID to `EntityState(entity_id, entity_class, position, velocity,
track_age, track_state, properties)` (`specola.entities`). `track_state`
is a `TrackState`; ground-truth simulators emit `CONFIRMED`. Each observation's timestep
is greater than the previous one within an episode. `reset()` validates all of this and
raises with a specific message, so run it early.

Optional hooks with defaults you can override: `supported_classes`,
`render`, `close_render`, `capture_frame`, `camera_names`,
`scene_geometry`, `scene_poses`.

Reference implementations:

- MuJoCo-family simulators: subclass `MujocoAdapter` and supply body
  rules. `examples/custom_mujoco_adapter.py` is the annotated template;
  `src/specola/adapters/robocasa.py` is a complete example.
- Everything else: `src/specola/adapters/isaaclab.py` (Isaac Sim),
  `src/specola/adapters/pybullet.py`, `src/specola/adapters/carla.py`.
  Each subclasses `SimulatorAdapter` directly.

## Your own module

Put the adapter in a module of your own, register a name with
`specola.registry.register(name, factory, description, install_hint)`,
and load it with `--adapter-module`:

```bash
specola write-specs --adapter-module my_package.my_adapter --sim mysim
specola run --adapter-module my_package.my_adapter --sim mysim \
    --specs my_specs.yaml --steps 200
```

Or declare an entry point in your own package's `pyproject.toml`, so the
name resolves without the flag:

```toml
[project.entry-points."specola.adapters"]
mysim = "my_package.my_adapter"
```

An adapter loaded this way runs with extraction status
`unaudited_third_party_provider`. Nothing else is required. Do all of
your development here.

## Bundling inside this repository

A module under `src/specola/adapters/` registered in this repository's
`pyproject.toml` entry-point group is a bundled adapter, and two review
gates apply. Both are silent until they fail, so do them together, in one
commit, after the adapter works as your own module.

1. **Enrollment.** Add the module to `_REVIEWED_IMPLEMENTATIONS` in
   `src/specola/adapter_enrollment.py`, mapping the module path to the
   adapter class name and factory name. Every adapter reset calls
   `guard_extraction_source`, which reconciles the installed entry points
   against this table. A registered module missing from the table raises
   on every reset of every adapter in that install, not only yours:

   ```
   EnrollmentError: missing reviewed implementation enrollment: <name>
   ```

   An enrolled row without an extraction provider is refused at reset
   (`bundled_row_pending`). Every enrolled row today is MuJoCo-family and
   carries the shared bounded provider, so no bundled adapter stands
   there. To run a bundled adapter on the legacy capability wire instead,
   list it in `EXTRACTION_SCOPE_EXCLUSIONS` and `_EXCLUDED_IMPLEMENTATIONS`
   in the same module, and leave it out of `_REVIEWED_IMPLEMENTATIONS`, as
   `mock`, `dataset`, `robodojo`, `pybullet`, `carla` and `isaaclab` are.
   That is a maintainer scope decision and it removes the adapter from the
   extraction programme rather than certifying it. The last three were
   moved there on 2026-10-13; their certification stays open in #111,
   #112 and #107.

2. **Capability declaration.** Add the entry-point name to `DECLARATIONS`
   in `tests/test_capability_declarations.py`. Declare each capability
   family the adapter supports (`rigid_body`, `articulation`,
   `deformable_state`, `contact`, `physical_sensor`) against at least one
   existing test that exercises extraction for that family. A family the
   adapter accepts scenes for but does not extract goes in
   `ACCEPTED_BUT_INCOMPLETE` with a test asserting the incompleteness
   marker. Silent acceptance is the failure this table exists to prevent.

Then reinstall the package in each conda env you test in (`pip install -e .`
inside the env, with `PYTHONNOUSERSITE=1`), because entry points are read
from the installed distribution, not from `pyproject.toml` on disk.

Add a pytest marker for the simulator in `pyproject.toml` next to the
existing ones, and gate simulator-backed tests with it so the sim-free
suite skips them cleanly.

## Checking your work

```bash
specola check --adapter-module my_package.my_adapter --sim mysim my_specs.yaml
pytest tests/test_capability_declarations.py      # bundled adapters only
```

`specola check` compiles specs against a discovered scene and reports
unbound selectors; it is the fastest way to see whether the entities and
properties you emit are the ones an author can name.
