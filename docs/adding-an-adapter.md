# Adding an adapter

An adapter projects one simulator's state into Specola observations. This
page is what an adapter needs to work: the contract, the factory,
registration, and the three commands that tell you it works. Every other
hook is optional; the list at the end says what each one adds and where
its rules live ([Adapter hooks](adapter-hooks.md)). An adapter bundled in
this repository also meets [Bundling an adapter](bundling-an-adapter.md).

Adding a simulator is additive. No registry, CLI, parser, monitor or runner
code changes; see the module docstring of `src/specola/registry.py`.
Adapters never control the simulator. Stepping, actions and rendering stay
in caller code, with one optional exception, the `make_advancer` drive hook.

Where this page names a symbol, resolve it by name (an editor, Serena,
`grep -n "def <name>"`); line numbers go stale, names do not.

## Read the normative contract

Read the `SimulatorAdapter` class docstring in `src/specola/adapter.py`
before you write code. It is the normative statement of what an adapter
may and may not do: which fields you include, which you omit and why, and
the rule that success predicates, rewards and task verdicts never feed
extraction. This page is a procedure; the docstring wins where they
differ.

## Choose the base class

- The simulator is MuJoCo-based (plain mujoco, robosuite, dm_control,
  gymnasium MuJoCo environments): subclass `MujocoAdapter`
  (`src/specola/adapters/mujoco.py`) and start from
  `examples/custom_mujoco_adapter.py`. You get observation extraction,
  entity-ID stability, joint, sensor and actuator discovery and contract
  validation; you supply body classification and simulator-specific
  naming.
- Anything else, including a simulator with no spatial notion at all:
  implement `SimulatorAdapter` directly. `adapters/fake.py` (the mock) is
  the dependency-free reference shape.

## Implement the contract

A subclass supplies:

- `super().__init__()` in the constructor;
- the `dt` property: finite, positive seconds between two observations
  the monitor receives (not the physics substep), since specification
  windows written in seconds convert to observation counts with it;
- `spatial_dims` (a
  positive integer, 2 or 3 in every bundled adapter; a simulator with no
  spatial notion returns 1 and leaves every `position` and `velocity`
  `None`, since the validator rejects 0);
- `_reset_impl()`, returning the first observation of an episode;
- `_get_observation_impl()`, returning the current observation;
- `register(...)` at module level (see Register the adapter).

Both observation hooks return a `MonitorObservation` (`adapter.py`) with
`timestep`, `time` and `perceived_entities`, a dict from integer entity ID
to `EntityState` (`src/specola/entities.py`). Build both with keyword
arguments; the positional order is not what a reader expects
(`EntityState(entity_id=..., entity_class=..., track_age=...,
track_state=..., position=..., velocity=..., properties=...)`).
Ground-truth simulators emit `TrackState.CONFIRMED`.

The base class validates every observation and raises with a specific
message, so run `reset()` and `get_observation()` early:

- within an episode each `timestep` is greater than the previous one and
  `time` does not decrease. `timestep` counts observation reads, not
  physics steps: two reads without a step in between still need
  increasing timesteps (the bindability probe reads twice without
  stepping), so keep a read counter and take `time` from the simulator's
  clock;
- an entity ID is stable within one episode and never changes class
  mid-episode. The base class cannot see the other half of the rule: an
  ID names one simulator object for the whole episode, and reusing it for
  a different object joins two histories into one without an error;
- `track_age` restarts with the episode; a first-seen step carried over
  from a previous episode makes it negative, which the base class
  rejects.

The entity and property-key population may change between observations
(contacts appear and vanish), and two entities of one class need not
expose the same keys.

Scalar simulator data goes in `EntityState.properties`. The core has no
built-in component vocabulary: every `[component]` in a specification
resolves against the keys the adapter extracted, and `specola discover`
output is the vocabulary an author works from. `position` and `velocity`
may be `None`; use `None` for a non-spatial entity rather than a
placeholder.

## Build the simulator in the factory

The factory you register (`factory(task, robot, target_only)`) constructs
the simulator and the adapter, and it is where simulator state is
prepared: import the simulator package inside the factory, not at module
top, build the environment, reset it so the scene exists, seed it, and
enable any renderer the adapter will capture from. `reset()` on the
adapter then reads that state; it does not create it.

The CLI inspects the factory's signature and forwards two optional
keyword arguments when present: `capture_frames=False` when the user
disabled frames, and `seed=<int>` when one was given. Only `specola run`
has `--seed` (default 0); `discover` and `check` pass none, so a factory
whose scene depends on a seed treats a missing seed as 0, or those
commands show a different scene from every run.

When the factory creates the environment it also owns closing it:
implement `close_native_extraction` (do not override `close`), or
register the close with `atexit`. An adapter constructed around an
environment the caller passed in leaves that environment open.

## Register the adapter

Call `register` at module level so importing the module registers the
adapter:

```python
from specola.registry import register

def _make(task, robot, target_only, *, seed=None, capture_frames=True):
    import mysimlib  # lazy: the module must import without the simulator
    env = mysimlib.make(task or "default", robot=robot or "default", seed=seed)
    env.reset()      # the adapter reads state; the factory creates it
    return MyAdapter(env)

register(
    name="mysim",
    factory=_make,
    description="My simulator (MyLib >= 1.0)",
    install_hint="pip install mysimlib",
    requires=("mysimlib",),
)
```

- `factory(task, robot, target_only)` returns an adapter; the meaning of
  `task` and `robot` is adapter-specific, so document it in the factory
  docstring.
- `requires` names the top-level packages the simulator needs; `specola
  doctor` checks they can be located without importing them.
- `handles(env) -> bool` and `wrap(env, ...)` let `Monitor.watch(env)`
  detect the adapter from an environment's wrapper chain.

Load the module with `--adapter-module`, or declare an entry point in
your own package so the name resolves without the flag:

```toml
[project.entry-points."specola.adapters"]
mysim = "my_package.my_adapter"
```

`specola doctor` lists only installed entry points, so an
`--adapter-module` adapter does not appear in it. A direct
`SimulatorAdapter` runs with extraction status
`unaudited_third_party_provider`, a `MujocoAdapter` subclass as
`unaudited_custom_adapter` (`adapter_enrollment.py`); nothing else is
required.

## Check that it works

Run these in the simulator's environment:

```bash
specola discover --adapter-module my_package.my_adapter --sim mysim --task <task>
specola check    --adapter-module my_package.my_adapter --sim mysim --task <task> my_specs.yaml
specola run      --adapter-module my_package.my_adapter --sim mysim --task <task> \
                 --specs my_specs.yaml --steps 200
```

`specola run` advances the simulator through the adapter's
`make_advancer` hook or, without one, by stepping the adapter's `env`
attribute (a gymnasium or robosuite environment). An adapter with
neither is never stepped, and `run` prints a drive warning: every step
reads the state the observation hook leaves. An adapter whose observation
hook advances the simulator itself provides a `make_advancer` that returns
`True` without stepping, as the mock does. Look at the values in the
results, not only the exit code.

`discover` lists every entity class and key a specification can select;
if a key you expect is missing, the adapter did not extract it. `check`
compiles your specifications against that vocabulary and shows, per
specification, the entities each slot can bind and the bindings at reset. `run` drives the simulator, monitors every binding, and writes the
results under `results/<run_id>/`.

## Optional hooks

Every hook below has a working default or is probed and skipped when
absent. Implement one when you want what it adds; its rules then become
part of your adapter's contract, and [Adapter hooks](adapter-hooks.md)
states them.

| Hook | What it adds |
|---|---|
| `property_metadata(entity_id)` | Groups, units and tags for each key, shown by `discover` and used by `@` filters |
| `supported_classes`, `capabilities` | A closed class or component vocabulary, so `check` rejects a misspelling before a run; schema facts for classes empty at reset |
| `entity_labels` | Human labels (`bowl0`) instead of `e<N>` in `check` and the dashboards |
| `make_advancer(mode, demo, seed)` | `specola run --mode hold`, `random` or `replay` on your simulator |
| `render`, `close_render` | `specola run --render`, the simulator's own live view |
| `capture_frame`, `camera_names` | Camera frames in the dashboards and recordings |
| `scene_geometry`, `scene_poses` | The web dashboard's 3D panel |
| `state_signature`, `capture_state`, `restore_state` | State recordings and `--mode replay` from them |
| `set_extraction_scope`, `set_extraction_entity_scope` | Skipping native reads of keys the loaded specifications do not need (the base already hides them) |
| `query_configuration` | `--queries` bindings that scope expensive pair work |
| `function_input_schemas` and related | Native operands for spec-owned functions |
| `discovery_warnings`, `entity_roles` and related | Scene facts and benchmark declarations in `discover` and the binding report |
| `step_on_main_thread` | Stepping on the main thread for simulators that require it |

[Adapter hooks](adapter-hooks.md) also has the rules that keep
specifications portable between adapters (key naming, metadata, the
semantic boundary between state and verdicts) and the `MujocoAdapter`
extension points.
