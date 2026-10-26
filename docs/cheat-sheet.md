# Spec-language cheat sheet

One page. The full guide is [Writing specifications](writing-specs.md).

## A minimal spec

```yaml
specs:
- name: no_drops                    # unique; duplicates rejected
  formula: "historically[0,1s](object_*[xpos_obj_main_2] - 0.7 >= 0)"
  constants: {limit: 0.5}           # optional; substituted by name
  backend: auto                     # optional; auto | native | rtamt
  # symmetric: true                 # optional; unordered same-class pairs
```

## Entity references

```
object_*[xpos_obj_main_2]
^^^^^^ ^  ^^^^^^^^^^^^^^^
class  |  component (sim-native key, from `specola discover`)
       binding token
```

| Token | Meaning | Monitors |
|---|---|---|
| `object_3[k]` | pin entity ID 3 | one |
| `object_*[k]` | every entity carrying `k` | one per entity |
| `object{A}_*[x]`, `object{B}_*[y]` | two distinct same-class slots | one per pair |
| `object_**[any: k]` | whole class collapsed | one, for the class |

## Component selectors (inside the brackets)

```
robot_*[qacc_*]                       glob (* crosses underscores)
robot_*[{qvel_a, qvel_b}]             explicit set
robot_*[qfrc_* @ actuator torques]    filter by group or tag
robot_*[qpos_j_* \ {qpos_j_3}]        set difference
robot_*[all: qvel_* @ arm]            quantifier picks the aggregation
```

Bare selector: one monitor per matched key. Inside `Min`/`Max`/`sum`:
arguments to one monitor. `all:`/`any:`: one verdict with the right
aggregation for the comparison's direction.

## Temporal operators

All windows trail; nothing looks ahead. Bounds in steps
(`historically[0,100]`) or seconds (`once[0,5s]`, rounded half-up to
whole steps). Both backends support the complete table. Explicit RTAMT
rejects `<->`/`xor` when either operand contains a temporal operator;
native supports those combinations.

| Operator | Reads as |
|---|---|
| `historically[a,b](φ)` | φ held throughout the trailing window |
| `once[a,b](φ)` | φ was true at least once in the trailing window |
| `historically(φ)` / `once(φ)` | running history since segment origin |
| `l since[a,b] r` / `l since r` | r held once and l has held since |
| `prev(φ)` | φ at the preceding sample; false at segment origin |
| `rise(φ)` / `fall(φ)` | false-to-true / true-to-false edge |

Legacy bounded `G`/`F` spellings remain accepted as past-time aliases;
`specola check` recommends the canonical names for new specifications.

Boolean: `and or not -> <-> xor`. Comparisons: `>= > <= < ==`.
Arithmetic: `sqrt`, `Min`, `Max`, `abs`, powers, and common SymPy
functions; unsupported functions are compile errors.

## Robustness in one line

ρ > 0 means satisfied by that margin, ρ < 0 violated by that much, in
the units of the compared expression; ρ = 0 counts as satisfied. Write
`distance - 0.3 >= 0` rather than `distance >= 0.3`: same verdict,
readable margin.

## Known traps

- **A spec that looks scene-wide is often entity-specific because the
  key is.** Keys are named after bodies, so `object_*[xpos_obj_main_2]`
  binds only the entity that carries that exact key. Glob the varying
  part (`xpos_*_2`) to cover the class, and read `specola check`'s
  binding count.
- **A term that matches nothing is a compile error**, including a `\`
  clause that removes nothing. This is deliberate: silence would mean
  monitoring less than you wrote.
- **`>` behaves like `>=`** (identical ρ), `==` never reports a positive
  margin, and `!=` is rejected. Write `abs(a - b) >= eps` for
  separation.
- **Mixed units aggregate silently in `Min`/`Max` across different
  signals** (metres vs newtons: the smaller raw number wins).
  `specola check` prints units and flags mixed aggregates; normalize
  per-predicate scales when crossing families. `historically`/`once` are safe, since
  they reduce one predicate over time.
- **`symmetric: true` is trusted, not checked.** Set it only when the
  expression really is order-independent (like `abs` of a difference),
  or half your pairs disappear wrongly.
- **Entity IDs are per-scene.** `object_3` in one task is not
  `object_3` in another. Pin only when the property is genuinely about
  one entity; re-run `specola check` per simulator and task.
- **Scene-wide existentials don't cross classes.** "Some object in some
  region" is not one verdict; write per-pair fan-out. `**` works only
  inside an aggregator or under `any:`/`all:`, and two expanded classes
  in one reduction are rejected.
- **Constants are scalars shared across the fan-out.** A parameter
  sweep is N suffixed spec entries, not a list value.
- **In replay, pass `--dt` matching the recording's control rate**
  (0.05 s for the released LIBERO sets). Seconds-valued windows
  otherwise scale by the simulator's model dt and a 1 s bound can
  become hundreds of replayed states.
- **Absence is neither satisfaction nor violation.** A binding whose
  entities leave the scene suspends; if they return in time it resumes
  with history reset. An episode ending with an unfilled window reports
  `undetermined`, not a guess.
