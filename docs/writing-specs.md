# Writing specifications

Specifications are YAML files containing past-time STL formulas over the
entities and signals exposed by a simulator. The examples below use
RoboCasa, but the same workflow applies to every adapter.

## Workflow

```bash
# Find the names available in this simulator and task.
specola discover --sim robocasa --task PickPlaceCounterToCabinet

# Write the YAML yourself, or use the interactive authoring workspace.
specola write-specs --sim robocasa --task PickPlaceCounterToCabinet

# Resolve every selector and catch mistakes before running an episode.
specola check --sim robocasa --task PickPlaceCounterToCabinet my_specs.yaml

# Run the monitor.
specola run --sim robocasa --task PickPlaceCounterToCabinet \
  --specs my_specs.yaml --steps 300
```

Run `specola check` after every edit and once for each simulator and task
you plan to use. Selectors are compiled against the live scene, so the
same file can resolve differently elsewhere.

## Discover the available signals

`specola discover` shows the entity classes and component keys you can
reference. On an interactive terminal it opens an explorer:

![Specola discover with the gripper class selected, a signal highlighted,
and every way of selecting it listed below](images/discover-tui.svg)

The panel under the signal list, **Ways to select this**, is every
reference the language can express for what you have selected: the
wildcard and pinned forms, the glob, the explicit set, each `@ group` and
`@ tag` filter, the set difference, and the quantified and whole-class
forms. Each one is resolved against this scene before it is offered, and
a shape that cannot be expressed here is shown with the reason instead of
being hidden. Selecting a row shows what that shape means and a starter
specification you can paste.

Filter with `--class`, `--entity`, `--signal`, `--group`, `--role`,
`--tag`, or `--search`. Filters are case-insensitive shell globs.
Repeated filters of one kind form a union; different kinds intersect.
Add `--verbose` for current values, use `--format json` for the complete
machine-readable catalogue, or pass `--raw` to inspect raw scene entities.

Discovery gives you the simulator's names. You still choose the threshold
and temporal property.

## Write interactively

`specola write-specs` adds formula, constants, validation, and save panels
to the discovery explorer. Select a signal, highlight the form you want in
**Ways to select this**, insert it into the formula, add the STL operator and threshold, then validate the draft:

![Specola write-specs with a validated draft and its resolved slots and
keys](images/write-specs-tui.svg)

Validation reports the resolved slots, keys, units, current values,
warnings, and binding count. A statically empty selector is rejected;
a declared runtime class may validate with a wait-for-runtime warning.

Use `--ui plain` for prompts. Plain mode is selected automatically when
input or output is redirected. The Textual workspace needs a terminal at
least 80 columns by 30 rows. `--output PATH` chooses the initial path in
the Save prompt; it does not overwrite an existing file automatically.

## A spec file

```yaml
specs:
- name: no_drops
  formula: "historically[0,1s](object_*[xpos_obj_main_2] - 0.7 >= 0)"

- name: base_settles
  formula: "once[0,5s](robot_*[qvel_mobilebase0_joint_mobile_forward] <= limit)"
  constants: {limit: 0.5}   # named constants substituted into the formula
  backend: auto             # auto | native | rtamt (default auto)
  # symmetric: true         # for same-class pairs where order doesn't matter
```

Each spec needs a unique `name` and a `formula`. Duplicate names are
rejected. `constants`, `backend`, and `symmetric` are optional. Unknown
keys are ignored, so take care with their spelling.

A formula is STL over *entity references*. Each reference has three parts:

```
object_*[xpos_obj_main_2]
^^^^^^ ^  ^^^^^^^^^^^^^^^
class  |  component (sim-native key)
       binding token
```

`historically[0,5s](...)` means "held throughout the trailing 5 seconds" and
`once[0,5s](...)` means "true at least once in it". All windows look
backward, never ahead. Every monitor reports **robustness** (ρ): positive
means satisfied by that margin, negative means violated by that much, in
the units of the compared expression. ρ = 0 counts as satisfied, and
strict and non-strict comparisons produce the same ρ, so `>` and `>=`
behave identically.

The examples use robot `0`, gripper `4`, three objects, and four target
regions. RoboCasa randomizes the item assigned to `obj_main`, but its body
name remains stable, so the examples use body-derived keys rather than item
names. `examples/mock_specs.yaml` shows the same ideas with the bundled
mock simulator.

## From one monitor to many

The binding token controls how many monitors a formula creates.

### One entity, one signal

Pin the entity by ID:

```yaml
- name: base_creeps
  formula: "historically[0,5s](robot_0[qvel_mobilebase0_joint_mobile_forward] <= 0.1)"
```

This creates one monitor. IDs come from `specola discover` and are
per-scene, so pin only when the property is genuinely about one entity.

### Every entity of a class

Swap the ID for `*`:

```yaml
- name: regions_above_counter
  formula: "historically[0,5s](region_*[z0] - 0.7 >= 0)"
```

This creates four monitors, one per target region, each with its own ρ
stream. Violations are reported against the entity that caused them, and
bindings are created and pruned as entities enter and leave the scene. On
sims that distinguish task-target regions, discovery registers only the
task's goal regions by default; pass `--all-regions` to include every
fixture.

### Only entities that carry the signal

An entity binds only if it has *every* component the reference names.
Classes are not necessarily homogeneous. In this scene they are not: the
three objects have zero keys in common, because each object's keys are
named after its own body.

```yaml
- name: no_drops
  formula: "historically[0,1s](object_*[xpos_obj_main_2] - 0.7 >= 0)"
```

This creates one monitor, on object 1 only. The other objects expose
`xpos_distr_counter_main_2` and `xpos_distr_cab_main_2` instead, so they
are not bound. This is the most common surprise in practice: a spec that
looks scene-wide is often entity-specific because the key is.

To cover the whole class, glob the part that varies:

```yaml
- name: no_drops_any
  formula: "historically[0,1s](object_*[xpos_*_2] - 0.7 >= 0)"
```

This creates three monitors, one per object, each resolving to its own
body's key. The three forms differ in what you get back:

| Form | Binds | Verdicts |
|---|---|---|
| `object_*[xpos_obj_main_2]` | one object | one |
| `object_*[xpos_*_2]` | every object | one per object |
| `object_**[any: xpos_*_2]` | every object | one, for the class |

On an unfamiliar simulator, `specola discover` shows how many keys are
common to a class and which entities carry a selected key. A component
that matches no entity is a compile error that names the available keys.

### Two entities of the same class

One wildcard per class binds one entity at a time. To relate two
entities of the same class, label the slots. `{A}` and `{B}` never bind
the same entity, so nothing is compared against itself.

```yaml
# Distinct cabinet levels stay at least 10 cm apart vertically:
- name: levels_separated
  formula: "historically[0,5s](abs(region{A}_*[z0] - region{B}_*[z0]) - 0.1 >= 0)"
  symmetric: true
```

This creates six monitors: the unordered pairs of four regions. Without
`symmetric: true` you would get twelve, one per ordering. The flag is
trusted, not checked, so set it only when the expression really is
order-independent, as `abs` of a difference is. Labels are single
uppercase letters.

### Two classes, related by arithmetic

References from different classes combine in one expression, and the
cross-product of their bindings fans out. Arithmetic supports `sqrt`,
`Min`, `Max`, `abs`, powers, and common functions; anything unsupported
is a compile error, not a silent miscomputation. Subtracting one
reference from another is the workhorse, because the difference *is* the
robustness margin, in the units of the operands.

```yaml
# The robot keeps 30 cm from the task object:
- name: no_collision
  formula: "historically[0,5s](sqrt((robot_*[xpos_mobilebase0_base_0]-object_*[xpos_obj_main_0])^2
                       + (robot_*[xpos_mobilebase0_base_1]-object_*[xpos_obj_main_1])^2
                       + (robot_*[xpos_mobilebase0_base_2]-object_*[xpos_obj_main_2])^2) - 0.3 >= 0)"
```

This creates one monitor per eligible robot-object pair. Writing
`distance - 0.3 >= 0` rather than `distance >= 0.3` says the same thing
but makes ρ readable: it is metres of clearance, going negative by
exactly the incursion.

Chaining subtractions under `Min` gives box containment, each argument
one face margin:

```yaml
- name: reaches_goal
  formula: "once[0,30s](Min(object_*[xpos_obj_main_0]-region_*[x0], region_*[x1]-object_*[xpos_obj_main_0],
                         object_*[xpos_obj_main_1]-region_*[y0], region_*[y1]-object_*[xpos_obj_main_1],
                         object_*[xpos_obj_main_2]-region_*[z0], region_*[z1]-object_*[xpos_obj_main_2]) >= 0)"
```

This creates four monitors: one eligible object × four regions. ρ is the
distance to the nearest face: "how far inside" when positive, "how far
outside" when negative.

## Selecting many signals at once

Inside the brackets you can select a *set* of components instead of one:

```
robot_*[qacc_*]                                        # glob
robot_*[{qvel_robot0_joint1, qvel_robot0_joint2}]      # explicit set
robot_*[qfrc_actuator_* @ actuator torques]            # glob filtered by metadata group
object_*[qpos_obj_joint0_* \ {qpos_obj_joint0_3,       # set difference
                              qpos_obj_joint0_4,
                              qpos_obj_joint0_5,
                              qpos_obj_joint0_6}]
```

The grammar is `(any:|all:)? term (\ term)*`, all inside the brackets. A
term is an exact name, a glob, or a `{a, b, ...}` set, optionally
filtered by `@ <label>`. In a glob only `*` is special, and it crosses
underscores. A term that matches nothing is an error, including a `\`
clause that removes nothing.

When to use which:

- **Glob** for a family that shares a naming convention.
- **Explicit set** when you want exactly these keys and want the spec to
  break loudly if one is renamed away. A glob would quietly keep working
  on the survivors.
- **`@ label`** survives naming-convention changes that break globs.
- **`\` difference** to subtract from a match. Subtract the whole set
  you mean: an object's free joint packs translation into `qpos_*_0..2`
  and the quaternion into `_3..6`, and removing one quaternion component
  leaves six keys, not three.

```yaml
# Fail if either named joint disappears.
- name: shoulder_pair_bounded
  formula: "historically[0,5s](robot_*[all: {qvel_robot0_joint1, qvel_robot0_joint2}] <= 2.0)"

# Remove the quaternion components, leaving translation.
- name: object_translation_bounded
  formula: 'historically[0,5s](Max(abs(object_*[qpos_obj_joint0_* \ {qpos_obj_joint0_3,
                                                          qpos_obj_joint0_4,
                                                          qpos_obj_joint0_5,
                                                          qpos_obj_joint0_6}])) <= 3.0)'
```

### `@ label` filters on groups or tags

`@ label` matches a key when the label is its **group** or one of its
**tags**. They answer different questions:

| | group | tags |
|---|---|---|
| answers | what kind of quantity is this | where on the robot is it |
| per key | exactly one | zero or more |
| examples | `joint angle`, `actuator torques`, `body position` | `arm`, `gripper`, `base`, `left`, `right` |
| units | one unit per group | may span units |

A key has one group and any number of tags.
`qfrc_actuator_robot0_joint1` sits in the group
`actuator torques` and is tagged `arm right`, so `@ actuator torques` and
`@ arm` both select it. Groups follow `<provenance> <quantity>` and keep
units separate: force and torque, or displacement and angle, belong to
different groups.

Command keys are grouped by what the actuator command means. If the
actuator kind is unknown, it gets an actuator-specific group so controls
with unknown units are not combined. Its discovery note shows the control
range.

```yaml
# every actuator torque, whatever limb it is on
- name: torques_bounded
  formula: "historically[0,5s](robot_*[all: qfrc_actuator_* @ actuator torques] <= 100)"

# every arm signal of one kind, whatever the naming convention
- name: arm_speed
  formula: "historically[0,5s](robot_*[all: qvel_* @ arm] <= 2.0)"
```

#### Declaration and task-identity tags

Adapters that know which entity a task is *about* say so through tags, so
a spec can bind "the drawer this episode manipulates" without naming any
scene-specific key. The RoboCasa adapter emits:

| tag | source | example |
|---|---|---|
| `fixture_ref:<name>` | `env.fixture_refs` | the drawer tasks' target drawer |
| `task_ref:<attr>` | task env attributes (`env.stove`, `env.sink`, `env.microwave`, `env.coffee_machine`, `env.counter`) | the faucet task's sink |
| `task_ref:knob` | `env.knob` (per-episode burner draw) | exactly the sampled knob's joint keys |
| `source:fixture_flag` | fixture press-button state (`state_turned_on`) | microwave/coffee machine on/off |
| `declared_name:<name>` / `declared_category:<cat>` | `env.object_cfgs` | `drawer_obj`, `distr_counter_1` |

The same spec file then runs unchanged on any scene, and on any harness
that builds these environments, because the binding is by declared
identity, not by key name:

```yaml
# The drawer this task manipulates stays shut:
- name: drawer_stays_shut
  formula: "G[0,10s](fixture_*[qpos_*slidejoint @ fixture_ref:drawer] >= -0.01)"

# The burner knob this episode is about never overshoots its band:
- name: knob_in_band
  formula: "G[0,10s](abs(fixture_*[qpos_* @ task_ref:knob]) <= 5.93)"

# Microwave stays off until the task says otherwise:
- name: microwave_off
  formula: "G[0,10s](fixture_*[state_turned_on @ task_ref:microwave] <= 0.5)"

# This task's sink handle, whatever the scene named it:
- name: faucet_handle
  formula: "F[0,30s](fixture_*[qpos_*handle_joint @ task_ref:sink] > 0.40)"
```

`state_turned_on` and the `state_door_fraction_*` keys are
declaration-tier signals: the adapter transcribes fixture state the
native criteria consume (a press-button flag, a normalized door
fraction) that no raw MjData key carries. Their provenance is on the
tag (`source:fixture_flag`, `source:fixture_helper`), and robustness on
a 0/1 flag is sign-only: there is no margin gradient to read.

A group contains one unit, so it can usually be aggregated directly. Tags
describe location and can span units:
`robot_*[qpos_* @ base]` selects base x, y, and yaw, mixing metres and
radians. That is useful for selecting the whole pose but not for taking
its raw maximum. `specola check` warns when an aggregate mixes units.

### What a selected set means

A selected set behaves differently depending on where it appears.

#### Bare selector

A bare selector creates one monitor per matched key:

```yaml
- name: joint_accel_bounded          # 13 monitors: one per matched qacc key
  formula: "historically[0,5s](robot_*[qacc_*] <= 50)"
```

#### Aggregator

Inside `Min`, `Max`, or `sum`, the matched keys become arguments to one
monitor. Functions map elementwise, so
`sum(abs(robot_*[qfrc_actuator_*]))` becomes
`abs(k1) + abs(k2) + ...`.

```yaml
- name: peak_joint_accel             # 1 monitor, max over all 13 keys
  formula: "historically[0,5s](Max(robot_*[qacc_*]) <= 50)"
```

#### `all:` and `any:`

These quantifiers choose the correct aggregation for the comparison.
For example, `all: s <= c` becomes `max(s) <= c`.

```yaml
- name: actuator_torques_bounded
  formula: "historically[0,5s](robot_*[all: qfrc_actuator_* @ actuator torques] <= 100)"
```

Prefer the keywords when the selector side is one reference. If the
comparison changes from `<=` to `>=`, the keyword updates its aggregation;
a hand-written guard does not.

### One verdict for the whole scene

Sometimes the property is about the scene, not any one entity: *some*
object reaches the shelf, and you do not care which. `object_**` expands
at compile time to every discovered entity of the class, and the
quantifier collapses them into one verdict:

```yaml
- name: some_object_lifted
  formula: "once[0,30s](object_**[any: xpos_*_2] >= 1.1)"
```

This creates one monitor over all matched components and entities. The
glob is resolved separately for each entity, so it covers body-specific
keys that one `*` binding cannot.

`**` is valid only inside an aggregator or under `any:` or `all:`.
Expansion is fixed at episode start; disappearing entities follow the
normal suspension rules. The current release rejects two expanded classes
in one reduction and mixed nested reductions that would erase per-entity
grouping. Use explicit formulas for those cases. Equal nested reductions
such as `Max(Max(...))` remain legal.

## What to write specs about

The selectors above support several useful classes of property.

### Geometry from pose

The `no_collision` example compares the distance between body origins
with a clearance. Its ρ is metres. This measures proximity, not contact:
origin distance alone cannot tell whether two shapes touch.

### Measured contact

On MuJoCo-family adapters, `cfrc_ext_<body>_0..5` gives the external force
and torque on each exposed body:

```yaml
# Nothing hits the robot harder than 20 N, anywhere on it.
- name: no_hard_contact
  formula: "historically[0,5s](Max(abs(robot_*[cfrc_ext_* @ body external force])) <= 20)"
```

The group filter keeps force in newtons and leaves torque out. ρ is
newtons of headroom.

`cfrc_ext` is a per-body aggregate: it sums every influence on the body
and does not identify the counterpart. For pair-attributed contact, use
the dynamic `robot_contact` and `contact` classes. Each unordered body
pair is one entity with `contact_point_force_max`,
`contact_normal_force_max`, `contact_normal_force_sum`,
`contact_resultant_force` (all N), `contact_point_count`, and
`contact_min_dist` (m; negative means penetration):

```yaml
# No robot-involving contact pair carries more than 200 N at any point.
- name: contact_force_ceiling
  formula: "robot_contact_*[contact_point_force_max] < 200"
```

Pairs appear when bodies first touch and are labeled with both
participants. These classes may be empty at reset; discovery lists them
as declared dynamic classes, and their specifications wait for the first
pair. A bound on a `_max` key applies to every pair. A total summed across
pairs is not expressible from per-pair bounds.

Prefer the force keys for temporal formulas: `contact_min_dist` is
absent while a pair is separated, so a spec bound to it suspends and
resumes across contact gaps, where the force keys read a continuous
stream from first contact onward.

### Effort

Per-DOF force families describe the robot's own effort; `cfrc_ext`
describes external influence:

```yaml
# Total actuator effort stays inside a budget, whatever the limb.
- name: effort_budget
  formula: "historically[0,5s](sum(abs(robot_*[qfrc_actuator_* @ actuator torques])) <= 250)"

# No single joint is fighting its own passive dynamics hard.
- name: passive_load_bounded
  formula: "historically[0,5s](Max(abs(robot_*[qfrc_passive_* @ passive torques])) <= 8)"
```

`sum` gives a total budget; `Max` limits the worst component.
`qfrc_passive` components read zero when the model configures no matching
effect. Check their values before setting a bound, or use the total.

### Ordering and history

`left since[a,b] right` holds when the right side was true in the
trailing window and the left side has held from then until now. Omit the
bounds to search the complete contiguous segment:

```yaml
# The object has stayed above counter height ever since being lifted
# clear of it, at some point in the last 10 s.
- name: lifted_and_stayed_up
  formula: "(object_*[xpos_*_2] >= 0.75) since[0,10s] (object_*[xpos_*_2] >= 0.95)"

# Detect the moment the object first clears the counter height.
- name: lift_started
  formula: "rise(object_*[xpos_*_2] >= 0.95)"
```

### Declared sensors

Where a model declares them, `sensordata_<name>` keys carry the sensor's
own units.

## Temporal operators and robustness

All operators are causal and look backward. No verdict depends on future
input. Canonical spellings are:

- `historically[a,b](φ)`: `φ` held throughout the trailing window.
- `once[a,b](φ)`: `φ` held at least once in the trailing window.
- `historically(φ)` and `once(φ)`: running history from the current
  contiguous segment's origin.
- `left since[a,b] right` and `left since right`: `right` held once and
  `left` has held since then. Since is non-associative; parenthesize a
  nested since formula.
- `prev(φ)`: the preceding sample. It is false at segment origin.
- `rise(φ)` and `fall(φ)`: false-to-true and true-to-false edges. A
  predicate already true at segment origin counts as a rise.
- Boolean operators: `and or not -> <-> xor`.
- Comparisons: `>= > <= < ==`.

Bounds can be steps (`historically[0,100]`) or seconds
(`once[0,5s]`, rounded half-up to whole steps). An authored `[0,N]`
window covers exactly N samples. Legacy bounded `G`, `F`, `H`, and `P`
spellings remain accepted; `G` and `F` are Specola's historical aliases,
not future-time STL. `specola check` recommends canonical spellings for
new specifications.

ρ composes by the standard quantitative semantics:

| Form | ρ |
|---|---|
| `a >= b`, `a > b` | `a - b`, in the units of the operands |
| `a <= b`, `a < b` | `b - a` |
| `a == b` | `-abs(a - b)` |
| `not φ` | `-ρ(φ)` |
| `φ and ψ` | `min(ρ(φ), ρ(ψ))` |
| `φ or ψ` | `max(ρ(φ), ρ(ψ))` |
| `historically[a,b](φ)` | `min` of ρ(φ) over the window |
| `once[a,b](φ)` | `max` of ρ(φ) over the window |
| `prev(φ)` | preceding ρ(φ), or `-inf` at segment origin |
| `rise(φ)` | `min(ρ(φ,t), -ρ(φ,t-1))`; origin is ρ(φ,0) |
| `fall(φ)` | `min(-ρ(φ,t), ρ(φ,t-1))`; origin is -ρ(φ,0) |

`==` never reports a positive margin, and `!=` is rejected outright: it
has no meaningful continuous robustness. Write
`abs(lhs - rhs) >= epsilon` instead, which states the separation you
mean and gives ρ a scale.

### Combining different units

`and`, `or`, `Min`, and `Max` compare raw margins. Combining
`clearance_m` with `force_margin_N`, for example, ranks metres against
newtons. Normalize each margin by an appropriate scale first.
`specola check` prints the unit of every resolved key and warns about
mixed-unit aggregates. `G` and `F` reduce one predicate over time and do
not introduce this problem.

### Finality

Robustness streams from the first step. A verdict becomes
final when its window fills or when it settles, whichever comes first: a
violated `historically[0,b]` cannot recover before the window fills, and a
satisfied `once[0,b]` cannot be undone before it, so both are final the
moment they settle. The other direction of each waits for the window,
and so do nested windows, nonzero-lower `historically[a,b]` and
`once[a,b]` (legacy `G[a,b]` and `F[a,b]`), and bounded `since[a,b]`.
When their children are immediately defined, unbounded `historically`,
`once`, and `since`, plus `prev`, `rise`, and `fall`, have defined origin
values and add no history-window wait. Nested bounded children retain
their warm-up; `prev`, `rise`, and `fall` add one sample when they need a
preceding child value. While a
window fills, the monitor is *warming up*: the live display shows
`pend`, and the API reports the evaluation as warming up. An episode
that ends with a verdict still unsettled reports `undetermined` rather
than guessing. A spec with no temporal operator has no window and is
final from its first sample.

Final does not mean frozen. Once a full window begins sliding, old samples
age out and ρ can change sign again. The episode outcome uses the last
final value for each binding, so a binding that violated early and later
recovered can still pass.

### Episode outcome

A spec that binds several entities gets one verdict per binding. The
episode is FAIL if any final binding failed, PASS if none failed and at
least one passed, and UNDETERMINED if no binding reached a final verdict.

## Backends

`backend:` selects the temporal evaluation backend per spec:

| Formula shape | native (no extra deps) | rtamt (`pip install -e ".[rtamt]"`) |
|---|---|---|
| atomic predicates, Boolean combinations | yes | yes |
| bounded/unbounded `historically`, `once`, `since` | yes | yes |
| `prev`, `rise`, `fall` | yes | yes |
| `<->` / `xor` wrapping a temporal subformula | yes | rejected at compile |

`auto` uses native for the causal language. Explicit `rtamt` remains an
independent implementation for cross-checking. Shapes with no correct
encoding are rejected at compilation.

## Suspension

When a binding's entities disappear from the scene, its monitor
suspends: nothing is fed to the backend, and robustness is reported
absent. Absence is never interpreted as safety or violation. A suspended
binding is kept for a retention horizon that defaults to the spec's own
window length. If its entities return in time, the same monitor resumes
with its history reset, and
verdicts become final again once the window refills or the verdict
settles, whichever comes first (see [Finality](#finality)).

## Check before you run

`specola check` compiles every formula against the live simulator without
running an episode. It shows the selector expansion and each resolved key
with its unit, group, and tags.

A misspelled class, missing component, or empty selector is an error. A
mixed-unit aggregate is a warning: narrow the selector to one group or
normalize the margins before combining them.

Classes declared as runtime-populated may compile while empty and wait for
their first entity. In the `Monitor`/`Session` API, a class that is unknown
for an episode can instead be skipped and recorded as `not_applicable`.

A wildcard slot on a static class needs a candidate entity at episode
start. A class is static when the adapter declares a closed class
vocabulary and does not list the class as runtime-populated, so every
entity of that class exists at reset. If none of them exposes all the
components your reference names, compilation fails with an unbound static
slot error instead of waiting for an entity that cannot arrive. Both
`specola check` and `Monitor.watch()` report it, so a spec that names a
key belonging to another entity fails before the episode runs.

Run controls, terminal keybindings, and result files are documented in
[Running the monitor](running.md).

## Gotchas

- **Keys are body-specific.** `object_*[xpos_obj_main_2]` binds only the
  object whose body is `obj_main`. Glob the varying part to cover a
  class.
- **YAML quoting eats `\`.** In double quotes YAML treats `\` as an
  escape, so a set-difference selector silently loses its backslash. Use
  single quotes for any formula containing `\`.
- **Selectors compile per sim.** The same file can resolve differently,
  or fail, on a different sim or task. Run `specola check` per target.
- **Pinned IDs are per-scene.** `robot_0` is a fact about this task's
  scene, not a constant.
- **Windows count samples.** `[0,N]` in steps is exactly N samples; in
  seconds it converts with the effective dt (`--dt` or
  `Monitor.watch(dt=...)` changes it, and the compiled window with it).
- **`cvel_<body>_3..5` is not the body origin's velocity.** It is
  com-referenced; a body pinned to the world through its own origin
  still reads non-zero. The origin-exact velocity is not extracted.
- **`cacc_<body>_0..5` is proper acceleration.** At rest on the ground
  it reads +9.81 m/s² in z; in free fall it reads zero.
- **`cfrc_ext` covers the bodies an entity exposes**, not every body in
  the model. A body-scoped entity does not carry its children's contact
  keys.
- **Actuator force, length, and velocity carry no unit on free and ball
  joints**, where the gear vector makes the joint type an unreliable
  guide. On hinge and slide joints they carry the joint's units.

When a value does not read the way you expected, `specola discover`
prints the group and unit for every key.
