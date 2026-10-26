# Python API

Use this page when you run your own environment, policy, and rollout loop
and want Specola monitoring inside it. The CLI never sees your policy:
`specola run` can only drive a simulator with random actions, a held zero
action, or a replayed demonstration (`--mode random|hold|replay`). Your
inference script keeps stepping the environment; Specola watches.

Which entry point when:

- **Your own policy or inference script**: the `Monitor` facade on this
  page.
- **Scripted or demonstration-driven runs**: `specola run` ([Running
  Specola](running.md)).
- **Re-checking old runs against new specs**: `specola run --mode replay
  --demo demo.hdf5` re-drives the simulator from a recorded
  demonstration, and `specola run --sim dataset --task traj.csv` monitors
  a recorded trajectory with no simulator. Both accept a fresh `--specs`
  each invocation. `specola view` only serves a finished `--web`
  recording in the browser; it never re-evaluates specs.

Everything here needs an adapter for your simulator. The bundled ones are
listed in the [README](../README.md#bundled-adapters); writing one is
covered in [Add a simulator](../README.md#add-a-simulator).

## The evaluation loop

Every Specola call in the loop, with your code marked as comments:

```python
from specola import Monitor

monitor = Monitor("my_specs.yaml")        # parse specs once, fail fast on a bad file
for episode in range(n_episodes):
    obs, info = env.reset()               # the caller owns simulator reset
    with monitor.watch(env, meta={"task": task_name}) as session:
        for step in range(horizon):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            result = session.step()       # once per env.step()
            margin = result.rho("spec_name")
            # result.raw.evaluations carries every per-binding margin.
        session.end(success=your_success_flag)   # verdicts + one result file
```

### `Monitor`

`Monitor(specs, results_dir="results", run_id=None)`

| Argument | Meaning |
|---|---|
| `specs` | Path to a spec YAML (`specs:` list). Parsed immediately; a broken file raises here, before any rollout. |
| `results_dir` | Where episode result files land (`<results_dir>/<run_id>/`). `None` disables result files entirely. |
| `run_id` | Directory name for this run. Defaults to a timestamp salted with random hex, so parallel Monitors never collide. |

Construct it once and reuse it across episodes; parsing does not repeat.

### `Monitor.watch(env_or_adapter, meta=None, dt=None, **adapter_kwargs)`

Starts one monitored episode and returns a `Session`.

- Pass your raw environment and Specola finds the registered adapter by
  walking the `.env` wrapper chain; `adapter_kwargs` go to that adapter's
  constructor. Or pass an already-constructed `SimulatorAdapter`
  (then `adapter_kwargs` must be empty).
- The caller resets or otherwise prepares the simulator first. `watch()` does
  not call `env.reset()` or advance physics; it resets the adapter's episode
  state, discovers the simulator's current scene, and compiles every spec
  against it. Resetting the simulator after `watch()` would make that compiled
  scene stale.
- `meta` values must be `str`, `int`, `float`, or `bool`; they are stamped
  onto every row of the result file (keys that collide with reserved
  result columns raise).
- `dt` overrides the spec-evaluation timestep when it differs from the
  adapter's own, e.g. to match a recorded demo's control rate.
- Calling `watch()` while a previous session is still open ends that
  session as `truncated` instead of losing it.
- A spec whose wildcard slot names a class with a fixed population and no
  entity exposing the components it requires raises
  `UnboundStaticSlotError` (a `SpecConfigError`). Specs referencing a
  class outside the episode's vocabulary are still skipped and recorded
  as `not_applicable`; a slot that can never bind is an error, because
  the monitor would otherwise report `undetermined` for the whole
  episode.

### `Session`

| Member | Meaning |
|---|---|
| `step()` | Call once per environment step. Pulls the current observation from the adapter, updates every monitor, returns a `StepResult`. |
| `end(success=None, status="completed")` | Finalizes the episode, writes the result file, returns an `EpisodeRecord`. Idempotent. |
| `ended` | Whether `end()` has run. |
| `adapter` | The resolved `SimulatorAdapter`, e.g. for `session.adapter.render()`. |

As a context manager, `Session` ends itself: normal exit calls `end()`,
an exception ends the episode as `truncated`. An interrupted episode is
recorded, not thrown away.

### `StepResult`

| Member | Meaning |
|---|---|
| `violated(spec_name=None)` | `True` if any spec (or the named spec) has a finalized violation at this step. |
| `rho(spec_name)` | Minimum robustness across the spec's bindings at this step, `None` before the window has data. |
| `violations` | Tuple of `Violation(spec, monitor, binding, rho, step, time)`, one per violating binding. |
| `raw` | The underlying `EngineResult` (see below) with every per-binding evaluation. |

`Session.step()` is deliberately silent: synchronous console output in an
inference loop adds latency and can dominate the monitoring cost being
measured. Print or log `rho(name)` when one authored specification is of
interest, or iterate over `raw.evaluations` for per-binding values:

```python
result = session.step()
print(result.rho("gentle_contact"))
for evaluation in result.raw.evaluations:
    print(evaluation.spec_name, evaluation.binding, evaluation.robustness)
```

The CLI supplies its own terminal and browser presentation. The library API
keeps presentation under the caller's control; it has no `verbose=True`
switch.

### `EpisodeRecord`

Returned by `end()`: `episode_id`, `status` (`completed` or `truncated`),
`success`, `steps`, `meta`, `path` (the written result file), and
`specs`, a dict mapping each spec name to a `SpecOutcome` with `verdict`
(`pass` / `fail` / `undetermined` / `not_applicable`), `worst_rho`, and
`first_violation_step`. The result-file format itself is documented in
[Running Specola](running.md).

## The engine

Drop below the facade when you step the adapter yourself, want no result
files, or aggregate results your own way:

```python
from specola import MonitorEngine, reset_adapter

engine = MonitorEngine.load("my_specs.yaml")  # parse once
reset = reset_adapter(adapter)                # episode discovery
engine.start_episode(                         # compile per episode
    dt=adapter.dt,
    capabilities=reset.capabilities,
    catalog=reset.discovered_catalog,
)
result = engine.update(reset.initial_observation)
for evaluation in result.evaluations:
    print(evaluation.spec_name, evaluation.robustness,
          evaluation.status, evaluation.verdict_final)
```

`reset_adapter(adapter)` returns the episode's `initial_observation`,
`discovered_catalog`, and `capabilities`; `start_episode(...)` compiles
every spec against them and raises on a spec the scene cannot support
(pass `skip_unsatisfiable=True` to skip instead, as the facade does).
Then feed each step's `MonitorObservation`, from
`adapter.get_observation()`, to `engine.update()`.

`update()` returns an `EngineResult`: `evaluations` (one
`BindingEvaluation` per live binding, with `spec_name`, `binding`,
`robustness`, `status`, `verdict_final`, and per-component values),
plus lifecycle counts for the step (`created`, `resumed`,
`pruned_count`). The engine writes no files; persistence is yours.

## Ordered batches

Use `BatchedMonitorEngine` when caller code steps several environments and
already has one canonical `MonitorObservation` per environment:

```python
from specola import BatchEpisodeContext, BatchedMonitorEngine

contexts = tuple(
    BatchEpisodeContext(
        dt=adapter.dt,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )
    for adapter, reset in zip(adapters, reset_results)
)
batch = BatchedMonitorEngine.load("my_specs.yaml", contexts)

results = batch.step_batch(
    observations,
    reset_mask=(True,) * len(contexts),
)
```

The batch has a fixed number of ordered slots, including zero. Every call
requires exactly one observation and one built-in Boolean reset flag per
slot; results use the same order. A true flag resets that slot before its
observation is processed, making that observation the first sample of a new
episode. The first observation for every nonempty slot must therefore carry
a true reset flag.

When a reused slot's scalar compilation inputs change, pass an ordered
`reset_contexts` sequence containing a replacement `BatchEpisodeContext` at
that flagged index and `None` elsewhere. Omitting it reuses the contexts
currently stored by the batch, initially those given to `load`. A replacement
becomes the stored context only after compilation succeeds. Supplying a
context for an unflagged slot is rejected before any slot advances.

Each slot owns an independent scalar `MonitorEngine`, including its temporal
backend, bindings, entity identities, lifecycle counters, and reset state.
The current implementation deliberately loops over those scalar engines. It
is batched as an API, but it is not vectorized or claimed to be faster than a
user-side scalar loop. Adapters do not need a batched extraction method;
caller code may use one scalar adapter per environment or construct canonical
observations itself.

Input-shape and reset-flag validation occurs before any slot advances. A
reset or update error raises `BatchStepError`, whose `index`, `operation`,
and chained original exception identify the failing environment. Execution
otherwise matches an increasing-index scalar loop: earlier slots have
advanced and later slots have not. A reset failure invalidates the failing
slot while retaining its last successfully compiled context. An update
failure retains the same possibly partial per-manager progress as the scalar
engine; reset that slot before continuing unless the failure is known to have
occurred before monitor mutation. `batch.engine(index)` exposes a started slot's
scalar engine for detailed counts and manager inspection.

## Related pages

- [Writing specifications](writing-specs.md): the spec YAML and formula
  language.
- [Running Specola](running.md): the CLI, run states, and the result-file
  format.
- [README: Add a simulator](../README.md#add-a-simulator): the adapter
  contract for unsupported simulators.
