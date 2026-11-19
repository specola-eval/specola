# Adapter hooks

The optional hooks an adapter can implement, who calls each one and the
contract it carries, followed by the rules that keep specifications
portable between adapters. [Adding an adapter](adding-an-adapter.md) has
the required contract; each hook you implement adds its row here to it.
Adapters bundled in this repository also meet
[Bundling an adapter](bundling-an-adapter.md).

Where this page names a symbol, resolve it by name (an editor, Serena,
`grep -n "def <name>"`); line numbers go stale, names do not.

## Hooks the core calls

The base class defines the first group with defaults; the second group is
duck-typed (the core probes with `getattr`), so there is no base default
and an absent attribute means "not provided." Every hook is optional
unless the table says otherwise. Callers are modules in `src/specola/`.

### Defined on `SimulatorAdapter`

| Hook | Default | Called by | Contract |
|---|---|---|---|
| `dt`, `spatial_dims` | abstract | everything | Required; see [Adding an adapter](adding-an-adapter.md). |
| `supported_classes` | `None` (open vocabulary) | compilation, `discover`, the bindability guard | A closed `frozenset` of entity classes lets `specola check` reject a misspelt class before a run and enrolls the adapter in `tests/test_adapter_bindability.py`. A closed set must include every synthetic class the adapter can emit (MuJoCo's orphan `sensors` and `actuators`), or a reset that produces one is rejected by `_validate_entity_entry`. |
| `set_extraction_scope`, `set_extraction_entity_scope` | stores the scope; the base projects every observation to it | `Monitor.watch`, `run` | Spec-scoped extraction. Four layers: the base class applies an exact projection of `properties` to the scope before validation, so no adapter can expose an unscoped surface by mistake; the base projects `property_metadata` the same way (`property_metadata_for`); an adapter may additionally use the scope to skip native reads, which is where the optimization lives and is required of bundled live adapters (`MujocoAdapter` prunes early); and an adapter may prune whole entities under the entity scope (`RoboDojoAdapter._pair_in_scope`). Reset sees the scope of the previous session cleared. Design: `superpowers/specs/2026-08-02-spec-scoped-extraction.md`. |
| `set_query_configuration`, `query_configuration`, `query_configuration_epoch` | stores it; the epoch increments | `--queries`, `Monitor.watch(queries=...)` | Query bindings (`mujoco.mj_geomDistance`, `robodojo.mesh_pair`) scope expensive pair work. Validate the whole binding as soon as it is named, before deciding whether the work is on; a misspelt key must not pass as no query. Parse once per epoch. A configuration change takes effect at the next `reset()`. Reference: `RoboDojoAdapter._parse_population`. |
| `reset_catalog_only`, `last_observation`, `episode_started`, `episode_index`, `extraction_episode_epoch` | base bookkeeping | `session`, `discover`, the recorder | Read, not overridden. |
| `render`, `close_render` | no-op | `run --render` | On-screen rendering when the caller asks for it. A simulator with a native live view (a viewer window, a plotting window) has this capability, so the matrix row accounts for it like any other. The ownership rule of `close` applies: a viewer present before a draw is the caller's and stays open, unless an earlier draw of the adapter created it; a viewer the adapter created is the adapter's, tracked by the adapter's own reference, since a native reset can drop or replace the simulator's; and `close_render` restores any process-wide state rendering changed (an interactive plotting mode, a backend). `specola.viewer.ViewerOwnership` implements the viewer rules (wrap each draw in its `render`, call its `release` from `close_render`), and `SharedProcessState` the process-state rule, counted across every adapter in the process (`matplotlib_interactive()` is matplotlib's interactive mode; `pygame_session()` is pygame's initialization, quit on the last release only if pygame was not running before). A native viewer whose `close` calls `pygame.quit()` must not be closed while that state has other holders: release the adapter's surface and let the shared state decide. |
| `capture_frame(camera=None, *, width=320, height=240)`, `camera_names()` | `None`, `()` | the dashboards, the recorder | See Camera frames below. Keep the keyword signature: the recorder always passes `camera=`, `width=` and `height=`, so a narrower signature raises `TypeError` and disables recording. |
| `scene_geometry`, `scene_poses` | `None` | the web dashboard's 3D panel | See The 3D panel below. |
| `state_signature`, `capture_state`, `restore_state` | `None`, `None`, raises | state recording and `--mode replay` of a Specola state recording | The state tier. `state_signature` returns a JSON-serialisable dict carrying `length` (a positive integer, the length of the array `capture_state` returns) plus engine, version and model digest; `None` means no state tier and replay of a state recording is refused with that reason. `capture_state` returns a flat finite `float64` array of exactly that length, every time. `restore_state` puts the simulator into the state without a physics step, so the next observation reads it. Replay priming calls `state_signature` and `restore_state` after the factory and before the first `reset()`, so both work on a freshly constructed adapter. References: `MujocoAdapter`, `PyBulletAdapter`, `IsaacLabAdapter`; the writer's checks are in `state_recording.py`. |
| `function_input_schemas`, `set_function_inputs`, `function_entity_id`, `prepare_function_entity_ids` | `{}`, validates, allocates | the spec-function loader (`spec_functions.py`) | Native operands for spec-owned functions. Declare `function_input_schemas = {"<provider>": <schema int>}`. When `set_function_inputs` has requested a provider, every observation (reset included) carries `MonitorObservation.function_inputs = FunctionInputs(revision, values)`, with `values` an immutable mapping keyed by provider and `revision` a value that changes exactly when the operands' identity changes (a geometry rebuild), since the loader invalidates its caches on it. Authored entity IDs come from `function_entity_id(key, entity_class)`; the default allocator reserves the native IDs present at reset, not IDs a dynamic class allocates later, so an adapter whose population grows mid-episode allocates native IDs from a range the authored ones cannot reach. Keys share the adapter's identity namespace: document the key shape your native entities use. Reference: `RoboDojoAdapter._capture_function_inputs`. |
| `close` | calls `close_render` and `close_native_extraction` once each | every CLI command, after the run | Do not override `close`; implement `close_native_extraction` for owned resources, and only owned ones: an adapter constructed around an environment the caller passed in leaves that environment open, while one the factory built is closed (record which at construction). What the adapter owns includes resources the simulator creates on first use (a viewer, a renderer, a window); release them even when the native object has no `close` of its own. `Session.end()` does not close the adapter: a library caller closes it after the last episode. Whoever registers a release (an `atexit` entry, a handle) hands the exact token to whoever releases it; a release recomputed from the resource is a second guess (`MISTAKES.md`, "A resource built in one place is released in that place"). |

### Duck-typed

| Hook | Called by | Contract |
|---|---|---|
| `capabilities` (property) | compilation, `discover`, the spec-function loader, discovery persistence | An `AdapterCapabilities` (`src/specola/capabilities.py`). Absent, the core builds one from `supported_classes`. Declare it when the adapter has components that are schema facts rather than reset facts: a class whose population can be empty at reset (`runtime_populated_classes`, with `components_by_class`, a CLOSED `component_mode_by_class`, and `component_tags_by_class` and `component_groups_by_class` so a tag or group filter compiles without an instance), or outputs a spec function must not claim (the loader rejects a function whose outputs collide with declared components, with no instance present). References: `MujocoAdapter.capabilities`, `RoboDojoAdapter.capabilities`. |
| `property_metadata(entity_id)` | `capabilities.py`, `check_specs`, `run_monitor`, `discover` | `Dict[str, PropertyMeta]`. If implemented, `set(property_metadata(eid)) == set(entity.properties)` for every entity, including synthetic ones, on every observation. `PropertyMeta` fields, by keyword: `group`, `note`, `tags`, `unit`, `family`, `component`, `frame`; leave the last three unset unless the simulator declares them. See Rules below. |
| `MonitorObservation.component_tags` | binding eligibility (`bindings/eligibility.py`) | For every class with a CLOSED component schema whose tag vocabulary is declared in `component_tags_by_class` (a contact pair's two body labels, an `owner:<label>` association), each observation carries `component_tags = {entity_id: {property_key: (tag, ...)}}`; a slot with an `@` filter on such a class binds an entity only when the observation-time tags hold every filter label, so an entity that appears after reset is ineligible until its tags arrive. This holds for entities present at reset too: their `PropertyMeta.tags` are not consulted at runtime, so an adapter that declares the vocabulary and sends the tags only through metadata passes `specola check` and never binds (issue #150). Test an entity present at reset and one that appears mid-episode. |
| `entity_labels` (property) | `check_specs`, `write_specs`, the dashboards | `Dict[int, str]`, the human label behind each entity ID (`bowl0`, `left_arm`), a point-in-time snapshot. Without it the dashboards show `e<N>`. |
| `discovery_warnings(entities)` | `discover` | An iterable of non-empty strings shown with the discovery output, for scene facts an author must know (objects the task declares but the scene does not contain). |
| `unread_entity_metadata()` | `capabilities.py` | `{entity_id: (entity_class, metadata)}` for entities a scoped read skipped, so the catalog lists the whole scene. Needed only by adapters that prune entities under the scope. |
| `make_advancer(mode, demo, seed)` | `run` | Returns `advance(step) -> bool`, which drives one step and says whether the episode continues; when it returns `False` the loop breaks before that step's observation is read. For `--mode hold|random|replay` and `--demo`; refuse a mode you cannot honour rather than accepting and ignoring it. Set `advance.exhausted = True` when a replay source ran out, so the episode is recorded as `truncated` rather than `completed`. Step the environment the caller supplied, not an inner object behind its wrappers: wrappers carry action transformations and episode limits (a gymnasium `TimeLimit`), and both termination and truncation end the episode. The exception is a wrapper that resets the environment when an episode ends (an auto-reset wrapper): stepping through it replaces the terminal state before the adapter reads it, so detect that layer and step beneath it, or refuse the configuration, and keep the terminal observation: return `True` from the call whose step ends the episode, so its observation is read, and `False` from the next call. Detect a wrapper's role by type (`isinstance` against the framework's class), never by class name, so subclasses and re-exports are caught. If the native environment cannot accept what the caller's wrappers hand it (a rescaling wrapper passes an array where the native step needs a scalar), adapt at the native boundary, beneath the caller's chain, rather than bypassing the chain: put the conversion in a thin wrapper beneath every wrapper that transforms actions (`gymnasium.ActionWrapper` subclasses such as a rescaling wrapper); pass-through wrappers may sit beneath it, and `gymnasium.make` always puts its own (`PassiveEnvChecker`, `OrderEnforcing`) innermost, so check the order of action-transforming layers, never that your wrapper is the innermost layer. In an environment the factory builds, insert the conversion as the chain is built; in a chain the caller passed in, do not modify the caller's objects (no patched methods), but refuse with a message naming the wrapper the caller must add, and test a chain built by `gymnasium.make`; a caller's wrapper chain is reset by the caller before the adapter drives it. The rules in this row bind the adapter's drive whether it provides this hook or relies on the generic robosuite and gymnasium fallbacks in `run_monitor`. The fallbacks refuse an automatic-reset chain and drive a gymnasium action space through `gym_advancer` below, but they cannot supply a `hold_action` for a discrete space or adapt at the native boundary. So an adapter whose environment can arrive wrapped, which includes every gymnasium-style environment, provides the hook. For a gymnasium-style environment the hook returns `specola.drive.gym_advancer(env, mode, seed, demo, hold_action=...)`, which implements this row's rules (it steps the caller's outermost wrapper, refuses an automatic-reset chain by type, keeps the terminal observation, refuses `replay` and `--demo`). It drives `Box`, `Discrete` and `MultiDiscrete` spaces: random draws seeded values from the space; hold sends zeros on a `Box` and otherwise requires `hold_action`, the simulator's own no-op (an index named idle or no-op in its action table), because no index is neutral in general. Any other space is refused, and the adapter's own drive handles it; the run loop's own fallback calls the same function. Write a drive of your own only where the simulator's drive is not a gymnasium `step`, and then meet the rules yourself. Relying on the fallback is for an environment that cannot be wrapped, and even then decide it by running each drive mode once through the fallback, not by reading it (an environment can declare an action space the fallback reads correctly and still reject the fallback's array actions at runtime). Every bundled direct adapter provides the hook; the mock's steps nothing, since its scene advances on each observation read. Without the hook or an `env`, `run` prints a drive warning and never steps the simulator. |
| `step_on_main_thread` (attribute) | `run` | `True` when the simulator must step on the main thread (its callbacks or listeners deadlock when it is stepped from a worker thread). |
| `entity_roles`, `entity_role_basis`, `entity_role_provenance`, `entity_declared_names`, `entity_declared_categories`, `entity_declaration_sources`, `entity_fixture_names`, `entity_fixture_refs`, `entity_fixture_types`, `entity_fixture_provenance`, `entity_region_fixture_names`, `entity_region_fixture_refs`, `entity_region_keys` | the binding report in `run_monitor`, `discovery_model` | `Dict[int, ...]` each: what the benchmark declares about an entity (its role such as `target` or `goal`, what that role rests on, its declared name and category, the fixture it sits on). The `*_names`, `*_refs` and `entity_region_keys` mappings hold sequences; the others hold scalar labels. Only when the simulator declares these; never derived from a verdict. References: `adapters/libero.py`, `adapters/metaworld.py`. |
| `get_ep_meta()` on the adapter or its `env`, then `language_instruction` | `cli_common` task description | The natural-language task, read in that order: a mapping with a `"lang"` key first, then a string attribute. Shown in reports and the dashboards. |
| `field_surface_report`, `field_materialization` | `discover` | MuJoCo-family only; produced by the shared contract code, not by a subclass. |
| `env` (attribute) | `run`'s generic fallbacks, `Monitor.watch(env)` detection, the task description | The wrapped environment, when there is one. |

### MuJoCo subclasses

`MujocoAdapter` extension points, all overridable with `super()` merged in:
`_default_body_rules`, `_default_robot_bodies`, `_default_gripper_sites`,
`_discover_extra_regions`, `_semantic_policy`, `_declare_task_entities`,
`_declared_entity_states` with `_declared_entity_property_metadata`,
`_additional_joint_owners`, `_additional_body_owners`, `_entity_extra_tags`,
`_contact_pair_tags` with `_contact_tag_vocabulary`, and
`_gripper_properties` with `_gripper_property_meta`. The pairs are pairs:
whoever adds a key adds its metadata, and both call `super()` and merge,
the way `libero.py` and `robocasa.py` do, so metadata parity survives the
subclass's extra keys. A bundled MuJoCo subclass also installs the shared
bounded geometry provider at module level
(`mujoco_provider.install_bounded_provider(MyAdapter)`, as `robocasa.py`
does); without it enrollment refuses the row with `bundled_row_pending`.

## Rules that keep specifications portable

**Key naming: `<field>_<name>[_<index>]`.** Property keys derive from
simulator names by one rule, never hand-picked labels like `px` or
`speed`. The module-level `_property_token()` in `adapters/mujoco.py`
sanitizes a raw name into a token (punctuation to `_`, lowercase, repeats
collapsed, a leading digit prefixed); callers prefix the signal field:
`qpos_<joint>`, `xpos_<body>_<axis>`, `sensordata_<sensor>[_<axis>]`,
`ctrl_<actuator>`. Follow the same shape so specifications transfer between
simulators (two adapters for one robot emit the same gripper keys on purpose).
A simulator without named model objects names keys after the attribute
path it reads, tokenized the same way (a private attribute's leading
underscore dropped), and an unnamed state vector as `<field>_<index>`.
A flag follows the existing convention (`note="flag"`, values 0 and 1,
no unit).

**Metadata parity.** Every exposed key has a `PropertyMeta` and every
`PropertyMeta` names an exposed key, for every entity.
`MujocoAdapter.property_metadata()` enforces this by construction with
per-source helpers mirroring the extraction helpers key for key; copy the
shape of `tests/test_mujoco_property_meta.py::test_observation_wide_metadata_parity`
for a new adapter.

**`PropertyMeta` groups, units and tags.** `group` is
`<provenance> <quantity>`, singular: "joint angle," "base angular
velocity," "body position." The vocabulary is meant to be guessable. Keep
provenance in the group rather than a tag: a term takes one `@` filter, so
`@ position` plus `@ body` cannot be written. The force families stay
plural ("applied forces," "constraint torques"); they name distinct force
sources. A group is not presentation: a selector filters on it
(`robot_*[qacc_* @ joint angular acceleration]`), so keep every key in a
group in one unit (a `min` over a group mixing metres and radians compares
across units; split the family, as `base angles` and `base displacements`
do; `test_mujoco_property_meta.py::test_no_group_mixes_units` pins this
for MuJoCo adapters by comparing `note`; an adapter that sets `unit`
compares `unit`)
and treat a group label as published. `tags` are zero or more structural
labels per key (`("arm", "right")`), saying where on the robot the key is;
`@ label` matches a group or a tag, so never use one name for both
(`ComponentCatalog.ambiguous_filter_labels` warns). `note` and `unit`
derive from the physical quantity, never from the key string.

**Name collisions.** Two simulator objects whose names tokenize to one
key are a fact of real models (`a.b` and `a_b`). Resolve them
deterministically and visibly: `MujocoAdapter` gives colliding source
names indexed tokens (`a_b_model_index_0`, `a_b_model_index_1`;
`tests/test_mujoco_sensors.py::test_colliding_sensor_names_receive_indexed_tokens`).
A collision between emitted keys that cannot be told apart raises
`ValueError` naming both at `reset()` time, never a silent overwrite
(`MujocoAdapter._resolve_pose_body_ids`, `_add_values`,
`_claim_sensor_key`, `_claim_actuator_key`).

**`position` and `velocity` may be `None`.** The core imposes no spatial
requirement and resolves specifications from `properties`; use `None`
for non-spatial entities rather than a placeholder position.

**Instants, window means and events.** An observation describes one
instant (`MISTAKES.md`, "An observation must describe one instant"):
export the simulator's state at the end of the step. Some simulators
report a declared observation channel as a mean over the sample period
(an environment that integrates several internal steps per observation and returns their average).
Such a channel may be exported because it is what a policy receives, but
as its own key whose group and note both name the window (a note holding
only the unit does not), and never in place
of the instantaneous reading when the simulator holds one: export both.
At reset no window has elapsed; keep the key present (the docstring
requires every included field in every observation where its owner
exists) with the value the simulator itself returns to its caller for that
channel at reset (where it records a different value internally, the
returned one), and say in the docstring what that value is. A channel the
simulator reports nothing for at reset is left out, with the reason in the
docstring: a key absent from the reset observation cannot be bound by a
specification under a closed class vocabulary, because compilation checks
slots against the reset catalog.
A quantity sampled once per step can also miss an event inside the step
(a pulse input applied and consumed between two observations); export the
simulator's own accumulator or state for it where one exists, and do not
let an example specification claim an absence the observations cannot
establish.

**Semantic boundary.** Account for each included and omitted field with
provenance and a reason. Use simulator declarations for identity, role,
ordering and declared observation channels; never a success predicate,
reward or verdict API. That includes continuous scoring quantities: any
value the simulator computes to score, rank or judge the episode (a
reward, cost, penalty, risk or safety score, a success margin, a
termination flag), whether called as a function or read from a cache the
environment keeps, is not state. If a specification needs it, it is
restatable from exported state, and the specification computes it. Name
each quantity for what it is in the model's equations, not in a variable
name or comment: a deviation from a baseline is labelled as a deviation,
a filtered or delayed signal as filtered or delayed. Label approximations as approximations (a sampled
mesh distance is reported as sampled, never as exact clearance). Treat
engine introspection as an inventory only; ownership, units and physical
meaning come from explicit engine and simulator evidence. When supported
criteria depend on relations between entities, expose enough operand
identity and relation inputs to reconstruct them without emitting
criterion truth, and keep the candidate population independent of the
specifications. Two shapes, both already in use: a relation with a
measurable quantity (a distance, a gap, an adjacency flag) is a pair
entity class like `geom_pair` and `mesh_pair`, one entity per related
pair, whose keys carry the structural tags `a:<label>` and `b:<label>`;
an association (a goal that belongs to one agent, an object a container
holds) is a structural tag `owner:<label>` on each key of the owned
entity, so a selector filters on it (`goal_*[x @ owner:agent_0]`). Every
relation the simulator's API exposes (neighbours, ownership, containment,
lane membership) is accounted for in the matrix notes: included in one of
these shapes, or omitted with the reason. Tests cover inclusion and metadata parity, intentional
omissions, dynamic key sequences, and a guard that no criterion or verdict
API is called.

## Camera frames

`capture_frame(camera=None, *, width=320, height=240)` returns an HxWx3
uint8 RGB array or `None`; `camera` names a source from `camera_names()`,
`None` means the first name there (the adapter default); `width` and
`height` are a hint, and a simulator with fixed camera resolution returns
its native size. It is presentation only: render from simulator state and
never enable a benchmark observation channel or change the policy's
inputs. Two obligations sit behind the signature:

- **Render-context lifetime.** A simulator that changes textures or
  materials after loading its scene can push the change through the live
  render context only, so a renderer created later samples stale data.
  Construct the environment with its offscreen renderer alive before the
  first reset and capture through that context.
- **No process-wide side effects.** A renderer that is a plot rather than
  a 3D camera (a matplotlib history figure) renders on an
  offscreen canvas without switching the process's plotting backend or
  interactive mode, which would open windows or change other code's
  figures.
- **Content, not shape.** A rendering pipeline can return black or
  partial frames until it warms up; assert on `frame.any()` or a mean,
  never on shape and dtype. The first
  `capture_frame()` after `reset()` returns content.

`camera_names()` is empty whenever a frame cannot be attributed to the
monitored subject (one viewport over several parallel
environments) and whenever the caller has not enabled rendering. Promising
frames that never arrive is worse than declaring none. A swept vectorized
run records a trace per environment and reproduces the episode worth
studying at `num_envs=1`; say in the docstring whether that reproduction
is a replay (determinism established) or a close re-run.

Once per new adapter, render the same state through the simulator's native
pipeline and through `capture_frame`, open both, and compare orientation,
colours and textures. Stale-context corruption, RGB/BGR swaps and
upside-down cameras each shipped past automated checks before this step
existed.

## The 3D panel

`scene_geometry()` returns the scene's meshes once per episode after
reset; `scene_poses()` returns owning float32 `(xpos, xquat)` copies on
the simulator thread, one row per body of the geometry table, quaternions
`(w, x, y, z)`. Within an episode the geometry table is fixed and the
return must not alternate between `None` and arrays (pose records join
data lines by ordinal); decide viability once, at first use. The table may
be a stable subset of the scene while the monitored entity population
changes around it, but a scene whose renderable bodies themselves come and
go (actors spawned and destroyed during the episode) cannot implement the hooks. Verify quaternion
order against the installed simulator, not recall; frameworks differ
between `(w, x, y, z)` and `(x, y, z, w)`, versions of one framework have
switched, and both orders render plausibly at a glance. Formats
and the MuJoCo reference: `superpowers/specs/2026-08-03-streaming-pose-panel.md`;
`adapters/_usd_scene.py` is the shared USD mesh walk.

