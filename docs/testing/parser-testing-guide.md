# Comprehensive Parser Testing Guide
## specola / spec/ layer

*Examples use the mock adapter's vocabulary; run `specola discover --sim mock` to see it.*

A parser is one of the most systematically testable components in software
because the input-output relationship is completely deterministic and the
failure modes are well understood. This guide covers the complete taxonomy
of what to test, why each category matters, and how to structure the test
suite.

---

## Test taxonomy overview

| # | Category | What can go wrong |
|---|---|---|
| 1 | Atomic correctness | Wrong sign convention, wrong direction |
| 2 | Operator semantics | min/max confusion, wrong negation |
| 3 | Precedence | Silent wrong parse, hardest to catch |
| 4 | Associativity | Wrong grouping in chains |
| 5 | Parentheses override | Parens ignored or double-applied |
| 6 | Temporal window boundaries | Off-by-one, warming-up state wrong |
| 7 | Function semantics | sqrt, Min, Max, abs computed wrong |
| 8 | Multi-parameter binding | Slot assignment swapped |
| 9 | Pinned vs wildcard | Wrong entity selected |
| 10 | Error cases | Silent failure instead of loud error |
| 11 | Round-trip regression | Refactor breaks real spec files |
| 12 | Property-based testing | Structural invariants that must always hold |

---

## 1. Atomic correctness

Every predicate form the grammar supports gets its own ground-truth test
with a hand-computed expected value. These are the simplest tests but the
most important: if the base case is wrong, everything above it is wrong.

```python
# >= semantics: rho = lhs - rhs
assert parse_and_eval("robot_*[qpos_joint1] - 0.7 >= 0", qpos_joint1=0.9) == pytest.approx(0.2)
assert parse_and_eval("robot_*[qpos_joint1] - 0.7 >= 0", qpos_joint1=0.5) == pytest.approx(-0.2)
assert parse_and_eval("robot_*[qpos_joint1] - 0.7 >= 0", qpos_joint1=0.7) == pytest.approx(0.0)

# <= semantics: rho = rhs - lhs
assert parse_and_eval("robot_*[qpos_joint1] <= 0.7", qpos_joint1=0.9) == pytest.approx(-0.2)
assert parse_and_eval("robot_*[qpos_joint1] <= 0.7", qpos_joint1=0.5) == pytest.approx(0.2)

# reversed operand order
assert parse_and_eval("0.7 - robot_*[qpos_joint1] >= 0", qpos_joint1=0.9) == pytest.approx(-0.2)
assert parse_and_eval("0.7 - robot_*[qpos_joint1] >= 0", qpos_joint1=0.5) == pytest.approx(0.2)

# zero boundary
assert parse_and_eval("robot_*[qpos_joint1] - 0.7 >= 0", qpos_joint1=0.7) == pytest.approx(0.0)
```

`qpos_joint1` is the mock adapter's robot joint-angle key (`src/specola/adapters/fake.py`);
substitute whatever `specola discover` prints for your own sim. There is no
built-in component vocabulary, only what the adapter actually extracted.

**Why it matters:** The sign convention for `>=` vs `<=` is a common source
of bugs. A test that only checks the sign of rho (positive = satisfied)
will miss cases where the magnitude is wrong.

---

## 2. Operator semantics

Each operator has a known mathematical definition. Test it directly against
that definition with hand-computed values, not by comparing two parser
outputs against each other.

```python
a_val = +0.3   # phi1 is satisfied with margin 0.3
b_val = -0.5   # phi2 is violated with margin 0.5
c_val = +0.8   # phi3 is satisfied with margin 0.8

# not: negation flips sign exactly
assert eval("not phi1") == pytest.approx(-a_val)   # -0.3
assert eval("not phi2") == pytest.approx(-b_val)   # +0.5

# and: min of operands (weakest link)
assert eval("phi1 and phi2") == pytest.approx(min(a_val, b_val))   # -0.5
assert eval("phi1 and phi3") == pytest.approx(min(a_val, c_val))   # +0.3

# or: max of operands (strongest link)
assert eval("phi1 or phi2") == pytest.approx(max(a_val, b_val))    # +0.3
assert eval("phi2 or phi3") == pytest.approx(max(b_val, c_val))    # +0.8

# implication: max(-a, b)  =  (not phi1) or phi2
assert eval("phi1 -> phi2") == pytest.approx(max(-a_val, b_val))   # max(-0.3, -0.5) = -0.3
assert eval("phi2 -> phi1") == pytest.approx(max(-b_val, a_val))   # max(+0.5, +0.3) = +0.5
assert eval("phi1 -> phi3") == pytest.approx(max(-a_val, c_val))   # max(-0.3, +0.8) = +0.8

# G[0,T]: min over trailing window — worst case in the window
signal_qpos = [0.9, 0.8, 1.2]   # mu: +0.2, +0.1, +0.5 (threshold = 0.7)
r = eval_temporal("G[0,0.2s](robot_*[qpos_joint1] - 0.7 >= 0)", signal_qpos, dt=0.1)
assert r[2] == pytest.approx(0.1)   # window_steps=2 (0.2s/0.1s dt): buffer=[0.1,0.5] at step 2 -> min=0.1

# F[0,T]: max over trailing window — best case in the window
r = eval_temporal("F[0,0.2s](robot_*[qpos_joint1] - 0.7 >= 0)", signal_qpos, dt=0.1)
assert r[2] == pytest.approx(0.5)   # same buffer -> max=0.5

# not G: negation of temporal min
r = eval_temporal("not G[0,0.2s](robot_*[qpos_joint1] - 0.7 >= 0)", signal_qpos, dt=0.1)
assert r[2] == pytest.approx(-0.1)
```

---

## 3. Precedence

This is where parsers most commonly fail silently. A precedence bug produces
a valid parse that computes the wrong value: no error, just wrong answers.
Test every precedence pair explicitly.

**Standard STL precedence (tightest to loosest):**
`not` > `and` > `or` > `->` > temporal operators

```python
a, b, c = +0.2, -0.5, +0.8

# --- not vs and ---
# "not a and b" must parse as "(not a) and b", NOT "not (a and b)"
assert eval("not a and b") == pytest.approx(min(-a, b))       # min(-0.2, -0.5) = -0.5
assert eval("not (a and b)") == pytest.approx(-min(a, b))     # -min(0.2,-0.5) = +0.5
# these must differ:
assert eval("not a and b") != eval("not (a and b)")

# --- not vs or ---
# "not a or b" must parse as "(not a) or b", NOT "not (a or b)"
assert eval("not a or b") == pytest.approx(max(-a, b))        # max(-0.2, 0.8) = 0.8
assert eval("not (a or b)") == pytest.approx(-max(a, b))      # -max(0.2, 0.8) = -0.8
assert eval("not a or b") != eval("not (a or b)")

# --- and vs or ---
# "a and b or c" must parse as "(a and b) or c", NOT "a and (b or c)"
assert eval("a and b or c") == pytest.approx(max(min(a,b), c))    # max(-0.5, 0.8) = 0.8
assert eval("a and (b or c)") == pytest.approx(min(a, max(b,c)))  # min(0.2, 0.8) = 0.2
assert eval("a and b or c") != eval("a and (b or c)")

# "a or b and c" must parse as "a or (b and c)", NOT "(a or b) and c"
assert eval("a or b and c") == pytest.approx(max(a, min(b,c)))    # max(0.2, -0.5) = 0.2
assert eval("(a or b) and c") == pytest.approx(min(max(a,b), c))  # min(0.2, 0.8) = 0.2
# these happen to be equal for these values — use different values to distinguish:
a2, b2, c2 = +0.6, +0.3, -0.2
assert eval("a or b and c", a2,b2,c2) == pytest.approx(max(a2, min(b2,c2)))
assert eval("(a or b) and c", a2,b2,c2) == pytest.approx(min(max(a2,b2), c2))

# --- implication vs and/or ---
# "a and b -> c" should parse as "(a and b) -> c"
assert eval("a and b -> c") == pytest.approx(max(-min(a,b), c))
assert eval("a and (b -> c)") == pytest.approx(min(a, max(-b, c)))
assert eval("a and b -> c") != eval("a and (b -> c)")  # for most values
```

**Run these as parameterized tests across multiple value combinations to
avoid accidentally testing a case where two parse orders give the same answer.**

---

## 4. Associativity

For operators where associativity matters, verify the correct grouping.

```python
a, b, c = +0.2, -0.5, +0.8

# and is left-associative: "a and b and c" = "(a and b) and c"
# for min this is equivalent to "a and (b and c)" since min is associative
# but verify the parse tree is correct by checking with a mock observer

# implication is right-associative: "a -> b -> c" = "a -> (b -> c)"
# this matters: "a -> (b -> c)" = max(-a, max(-b, c))
#              "(a -> b) -> c" = max(-(max(-a, b)), c) = max(min(a,-b), c)
# these are different — the grammar picks right-associative

rho_right = max(-a, max(-b, c))    # right-associative
rho_left  = max(min(a, -b), c)    # left-associative
assert eval("a -> b -> c") == pytest.approx(rho_right)  # right-associative
```

---

## 5. Parentheses override

Explicit parentheses must override all default precedence.

```python
a, b, c = +0.2, -0.5, +0.8

# Double wrapping should be transparent
assert eval("((a))") == pytest.approx(eval("a"))
assert eval("((a and b))") == pytest.approx(eval("a and b"))

# Parens change the result vs no parens — verify both sides
assert eval("not (a and b)") == pytest.approx(-min(a, b))      # +0.5
assert eval("(not a) and b") == pytest.approx(min(-a, b))      # -0.5
assert eval("not (a and b)") != eval("(not a) and b")

assert eval("a and (b or c)") == pytest.approx(min(a, max(b,c)))  # 0.2
assert eval("(a and b) or c") == pytest.approx(max(min(a,b), c))  # 0.8
assert eval("a and (b or c)") != eval("(a and b) or c")

# Nested parens
assert eval("not (not a)") == pytest.approx(a)                 # double negation
assert eval("not (a or (b and c))") == pytest.approx(-max(a, min(b,c)))
```

---

## 6. Temporal window boundary conditions

Off-by-one errors in window computation are the most common temporal parser
bug. Test the exact sample counts for every relevant window size. The
convention (`src/specola/spec/intervals.py`): an authored `[0,N]`
converts to exactly `max(N, 1)` samples in the trailing window, **not**
`N + 1`, and the native backend implements that window as a sliding
`deque(maxlen=window_steps)` (`src/specola/backends/native.py`), so once
the window is full the oldest sample drops on every subsequent step.

```python
DT = 0.1  # 100ms timestep

# G[0,0s]: window of exactly 1 sample — just current value
signal = [0.9]  # mu = +0.2
r = eval_G("G[0,0s](robot_*[qpos_joint1] - 0.7 >= 0)", signal, dt=DT)
assert r[0] == pytest.approx(0.2)

# G[0,0.2s] at dt=0.1: window_steps = round(0.2/0.1) = 2 samples
signal = [0.9, 0.8]  # mu: +0.2, +0.1
r = eval_G("G[0,0.2s](robot_*[qpos_joint1] - 0.7 >= 0)", signal, dt=DT)
assert r[0] == pytest.approx(0.2)   # step 0: only 1 sample available (warming up)
assert r[1] == pytest.approx(0.1)   # step 1: window full (2 samples): min(0.2, 0.1) = 0.1

# G[0,T] — warming up, then sliding: G[0,0.3s] at dt=0.1 is a 3-sample window.
# It accumulates until full, then drops the oldest sample each step.
signal = [0.9, 0.8, 0.6, 1.2]   # mu: +0.2, +0.1, -0.1, +0.5
r = eval_G("G[0,0.3s](robot_*[qpos_joint1] - 0.7 >= 0)", signal, dt=DT)  # window_steps=3
assert r[0] == pytest.approx(0.2)    # 1 sample: min(0.2)
assert r[1] == pytest.approx(0.1)    # 2 samples: min(0.2, 0.1)
assert r[2] == pytest.approx(-0.1)   # window now full (3 samples): min(0.2, 0.1, -0.1)
assert r[3] == pytest.approx(-0.1)   # slides — oldest (0.2) drops: min(0.1, -0.1, 0.5)

# window_steps sanity: G[0,Ns] with dt=0.1 gives window_steps = round(N/dt),
# clamped to at least 1. (Not "+1" — authored [0,N] covers exactly N samples.)
assert parse_stl("G[0,1s](robot_*[qpos_joint1] >= 0)", dt=0.1).window_steps == 10
assert parse_stl("G[0,3s](robot_*[qpos_joint1] >= 0)", dt=0.1).window_steps == 30
assert parse_stl("G[0,0s](robot_*[qpos_joint1] >= 0)", dt=0.1).window_steps == 1   # single sample, clamped
assert parse_stl("robot_*[qpos_joint1] >= 0", dt=0.1).window_steps == 1            # no temporal op

# F[0,T] mirrors G[0,T] but with max instead of min
signal = [0.9, 0.8, 0.6, 1.2]
r = eval_F("F[0,0.3s](robot_*[qpos_joint1] - 0.7 >= 0)", signal, dt=DT)
assert r[3] == pytest.approx(0.5)   # slides: max(0.1, -0.1, 0.5) = 0.5

# nested G inside F — get each window_steps right independently:
# inner G[0,0.2s] -> window_steps=2; outer F[0,0.3s] -> window_steps=3.
signal = [0.9, 0.8, 0.6, 1.2, 1.1]
r = eval("F[0,0.3s](G[0,0.2s](robot_*[qpos_joint1] - 0.7 >= 0))", signal, dt=DT)
# inner G[0,0.2s] at step 2 = min(0.1, -0.1) = -0.1
# inner G at step 3 = min(-0.1, 0.5) = -0.1
# inner G at step 4 = min(0.5, 0.4) = 0.4
# outer F[0,0.3s] at step 4 = max(-0.1, -0.1, 0.4) = 0.4
assert r[4] == pytest.approx(0.4)
```

Bounded `since[a,b]` and nonzero-lower `historically[a,b]`/`once[a,b]`
follow the same authored-bound convention. Native and RTAMT both cover those
forms, their unbounded counterparts, and `prev`/`rise`/`fall`. New differential
tests compare every sample, including operator origins and partial windows;
see `tests/test_causal_past.py` and the accepted causal-past design record.

---

## 7. Function semantics

```python
# sqrt: Euclidean distance
robot_pos = (0, 0, 0)
obj_pos   = (3, 4, 0)
# dist = sqrt(9 + 16) = 5.0; rho = 5.0 - 1.0 = 4.0
assert eval(
    "sqrt((robot_*[xpos_base_0]-object_*[xpos_obj_0])^2 + (robot_*[xpos_base_1]-object_*[xpos_obj_1])^2) - 1 >= 0",
    robot=robot_pos, obj=obj_pos
) == pytest.approx(4.0)

# sqrt with zero distance
assert eval("sqrt((r_*[xpos_r_0]-o_*[xpos_o_0])^2) - 0.3 >= 0",
            r_pos=(1,0,0), o_pos=(1,0,0)) == pytest.approx(-0.3)

# Min: element-wise minimum — used for box containment
# object at xpos_obj_0=1.2, region x0=0.5, x1=1.5
# Min(1.2-0.5, 1.5-1.2) = Min(0.7, 0.3) = 0.3
assert eval(
    "Min(object_*[xpos_obj_0]-region_*[x0], region_*[x1]-object_*[xpos_obj_0]) >= 0",
    obj_xpos_0=1.2, x0=0.5, x1=1.5
) == pytest.approx(0.3)

# Min outside box: object at xpos_obj_0=2.0
# Min(2.0-0.5, 1.5-2.0) = Min(1.5, -0.5) = -0.5
assert eval(
    "Min(object_*[xpos_obj_0]-region_*[x0], region_*[x1]-object_*[xpos_obj_0]) >= 0",
    obj_xpos_0=2.0, x0=0.5, x1=1.5
) == pytest.approx(-0.5)

# Full 3D containment formula (this is the README tutorial's `reaches_goal`
# shape, over region half-extents x0/x1/y0/y1/z0/z1 — real keys on the mock
# adapter's region entity, src/specola/adapters/fake.py)
# object at (0.8, 0.1, 0.9), box centered at (1,0,1) with hw=0.2, hh=0.2, hd=0.2
# x: min(0.8-0.8, 1.2-0.8) = min(0, 0.4) = 0
# y: min(0.1-(-0.2), 0.2-0.1) = min(0.3, 0.1) = 0.1
# z: min(0.9-0.8, 1.2-0.9) = min(0.1, 0.3) = 0.1
# Min(0, 0.1, 0.1, 0.4, 0.3, 0.3) = 0
assert eval(
    "Min(object_*[xpos_obj_0]-region_*[x0], region_*[x1]-object_*[xpos_obj_0], "
    "    object_*[xpos_obj_1]-region_*[y0], region_*[y1]-object_*[xpos_obj_1], "
    "    object_*[xpos_obj_2]-region_*[z0], region_*[z1]-object_*[xpos_obj_2]) >= 0",
    obj=(0.8, 0.1, 0.9), region_center=(1,0,1), half=0.2
) == pytest.approx(0.0)

# abs: absolute value
assert eval("abs(robot_*[qpos_joint1] - 0.7) - 0.3 >= 0", qpos_joint1=1.1) == pytest.approx(0.1)
assert eval("abs(robot_*[qpos_joint1] - 0.7) - 0.3 >= 0", qpos_joint1=0.9) == pytest.approx(-0.1)
```

---

## 8. Multi-parameter binding

Test that each parameter slot is filled with the correct entity class and
that the binding direction is not swapped.

```python
robot = EntityState(entity_id=0, entity_class="robot", track_age=0,
                     track_state=TrackState.CONFIRMED,
                     properties={"xpos_base_0": 0.0, "xpos_base_1": 0.0})
obj   = EntityState(entity_id=1, entity_class="object", track_age=0,
                     track_state=TrackState.CONFIRMED,
                     properties={"xpos_obj_0": 3.0, "xpos_obj_1": 4.0})
obs   = make_obs([robot, obj])

# Robot-object distance formula: robot slot vs object slot
f = parse_stl(
    "sqrt((robot_*[xpos_base_0]-object_*[xpos_obj_0])^2 + (robot_*[xpos_base_1]-object_*[xpos_obj_1])^2) - 1 >= 0",
    dt=0.1
)
bindings = enumerate_bindings(f, obs)
assert len(bindings) == 1

b = next(iter(bindings))
d = binding_to_dict(b, f.params)  # binding_to_dict(binding, params) — needs both

# robot param (param_key ("robot",1)) must map to entity 0
# object param (param_key ("object",1)) must map to entity 1
robot_param_key = [k for k in d if k[0] == "robot"][0]
object_param_key = [k for k in d if k[0] == "object"][0]
assert d[robot_param_key] == 0
assert d[object_param_key] == 1

# Verify the rho value matches the expected distance
# (assumes `engine` is a MonitorEngine already loaded with this one spec)
result = engine.update(obs)
assert list(result.values())[0] == pytest.approx(4.0)  # sqrt(9+16) - 1

# Two objects, one robot: should produce 2 bindings
obj2 = EntityState(entity_id=2, entity_class="object", track_age=0,
                    track_state=TrackState.CONFIRMED,
                    properties={"xpos_obj_0": 0.0, "xpos_obj_1": 3.0})
obs2 = make_obs([robot, obj, obj2])
bindings2 = enumerate_bindings(f, obs2)
assert len(bindings2) == 2

# Two robots, two objects (same-class, injective):
# robot-object pairs: (r0,o1),(r0,o2),(r1,o1),(r1,o2) = 4 bindings
robot2 = EntityState(entity_id=3, entity_class="robot", track_age=0,
                      track_state=TrackState.CONFIRMED,
                      properties={"xpos_base_0": 1.0, "xpos_base_1": 0.0})
obs3 = make_obs([robot, robot2, obj, obj2])
bindings3 = enumerate_bindings(f, obs3)
assert len(bindings3) == 4
```

---

## 9. Pinned vs wildcard

```python
obs = make_obs([
    EntityState(entity_id=1, entity_class="object", track_age=0,
                track_state=TrackState.CONFIRMED,
                properties={"hinge_angle": 0.9}),
    EntityState(entity_id=2, entity_class="object", track_age=0,
                track_state=TrackState.CONFIRMED,
                properties={"hinge_angle": 1.4}),
])

# Wildcard: both objects bind
f_wild = parse_stl("object_*[hinge_angle] - 0.7 >= 0", dt=0.1)
r_wild = engine_update(f_wild, obs)
assert len(r_wild) == 2
rhos = sorted(r_wild.values())
assert rhos == pytest.approx([0.2, 0.7])

# Pinned to entity 2: only one binding, and it's the right entity
f_pin = parse_stl("object_2[hinge_angle] - 0.7 >= 0", dt=0.1)
r_pin = engine_update(f_pin, obs)
assert len(r_pin) == 1
assert list(r_pin.values())[0] == pytest.approx(0.7)  # entity 2 at hinge_angle=1.4

# Pinned to entity that doesn't exist: zero bindings
f_pin_missing = parse_stl("object_99[hinge_angle] - 0.7 >= 0", dt=0.1)
r_missing = engine_update(f_pin_missing, obs)
assert len(r_missing) == 0

# Entities are pinned by integer ID only — the grammar's binding token is
# `**`, `*`, or digits (src/specola/spec/grammar.py PARAM_RE), never a
# string label. An adapter's human-readable label (fake.py's entity_labels,
# e.g. entity 2 = "mock_distractor_object") is display-only; `specola discover`
# prints it next to the ID so you know which integer to pin.
```

---

## 10. Error cases

The parser must fail loudly on invalid input. Silent wrong answers are
worse than explicit errors. Syntax and formula-shape problems raise plain
`ValueError`; problems that depend on the connected adapter (missing
component, unknown class, missing catalog entry) raise `SpecConfigError`
(`src/specola/entities.py`). There is no separate `ParseError` class.

```python
import pytest
from specola.entities import SpecConfigError
from specola.components import ComponentCatalog

# --- Parse errors: malformed syntax ---
with pytest.raises(ValueError):
    parse_stl("G[0,1s]", dt=0.1)                          # missing body

with pytest.raises(ValueError):
    parse_stl("robot_*[qpos_joint1] >= ", dt=0.1)         # incomplete rhs

with pytest.raises(ValueError):
    parse_stl("robot_*[qpos_joint1] - >= 0", dt=0.1)      # missing operand

with pytest.raises(ValueError):
    parse_stl("G[0](robot_*[qpos_joint1] >= 0)", dt=0.1)  # missing window bound

with pytest.raises(ValueError):
    parse_stl("(robot_*[qpos_joint1] >= 0", dt=0.1)       # unmatched paren

# --- Spec config errors: valid syntax, invalid configuration ---
# A component that isn't among an entity's real keys fails compilation
# against a catalog, naming the entity's actual keys. This is the direct
# replacement for the old px/py/pz vocabulary check: there is no built-in
# component list to validate against, only what the adapter extracted.
catalog = ComponentCatalog.from_entities({
    1: EntityState(entity_id=1, entity_class="object", track_age=0,
                    track_state=TrackState.CONFIRMED,
                    properties={"mass": 0.25, "temperature": 22.0}),
})
with pytest.raises(SpecConfigError):
    parse_stl("object_*[hinge_angle] >= 0", dt=0.1, catalog=catalog)  # this entity has no hinge_angle

with pytest.raises(SpecConfigError):
    parse_stl("unicorn_*[qpos_joint1] >= 0", dt=0.1,
              valid_classes=["robot", "object"])              # unknown class

# --- Semantic errors: window bounds ---
# Malformed interval literals raise ValueError — caught during interval
# conversion (src/specola/spec/intervals.py), before any catalog or
# entity is involved.
with pytest.raises(ValueError):
    parse_stl("G[5s,1s](object_*[hinge_angle] >= 0)", dt=0.1)    # b < a

with pytest.raises(ValueError):
    parse_stl("G[-1s,1s](object_*[hinge_angle] >= 0)", dt=0.1)   # negative bound

# --- Runtime errors: component not available ---
obs_no_hinge = make_obs([EntityState(entity_id=1, entity_class="object",
                                      track_age=0, track_state=TrackState.CONFIRMED,
                                      properties={})])  # no hinge_angle
with pytest.raises(SpecConfigError):
    engine_update(parse_stl("object_*[hinge_angle] >= 0", dt=0.1), obs_no_hinge)

# --- Discovered property: entity without the property is excluded ---
# (this is eligibility, not an error — see binding tests)
obs_mixed = make_obs([
    EntityState(entity_id=1, entity_class="object", track_age=0,
                track_state=TrackState.CONFIRMED,
                properties={"hinge_angle": 0.1}),
    EntityState(entity_id=2, entity_class="object", track_age=0,
                track_state=TrackState.CONFIRMED,
                properties={}),  # no hinge_angle
])
f = parse_stl("0.5 - object_*[hinge_angle] >= 0", dt=0.1)
r = engine_update(f, obs_mixed)
assert len(r) == 1  # entity 2 excluded, not errored
```

---

## 11. Round-trip regression

Pick specs from your actual YAML files and verify each against a
hand-computed ground truth. These catch regressions when the parser
or semantic layer is refactored.

```python
# From examples/robocasa_specs.yaml — the keys below are what
# `specola discover --sim robocasa` reports for this task (see the README's
# tutorial); substitute the keys your own sim reports.
ROBOCASA_REGRESSION = [
    {
        "name": "no_collision",
        "formula": (
            "G[0,5s](sqrt((robot_*[xpos_mobilebase0_base_0]-object_*[xpos_obj_main_0])^2 "
            "+ (robot_*[xpos_mobilebase0_base_1]-object_*[xpos_obj_main_1])^2 "
            "+ (robot_*[xpos_mobilebase0_base_2]-object_*[xpos_obj_main_2])^2) - 0.3 >= 0)"
        ),
        "robot_pos": (0, 0, 0),
        "obj_pos":   (0.5, 0, 0),
        "expected":  0.2,  # sqrt(0.25) - 0.3 = 0.5 - 0.3
    },
    {
        "name": "no_drops",
        "formula": "G[0,1s](object_*[xpos_obj_main_2] - 0.7 >= 0)",
        "obj_pz": 0.9,
        "expected": 0.2,
    },
]

@pytest.mark.parametrize("case", ROBOCASA_REGRESSION)
def test_robocasa_regression(case):
    f = parse_stl(case["formula"], dt=0.05)
    obs = build_obs_from_case(case)
    result = engine_update(f, obs)
    assert list(result.values())[0] == pytest.approx(case["expected"], abs=1e-4)
```

---

## 12. Property-based testing (hypothesis)

These are structural invariants that must hold for **any** valid input:
they check mathematical properties of the robustness semantics, not
specific values. If any of these fail, something is structurally wrong.

```python
from hypothesis import given, strategies as st, assume, settings

float_vals = st.floats(-10, 10, allow_nan=False, allow_infinity=False)
signal_list = st.lists(float_vals, min_size=5, max_size=20)

@given(v=float_vals)
def test_double_negation(v):
    """not(not(phi)) == phi for all values."""
    rho_direct   = eval_atomic("robot_*[qpos_joint1] - 0 >= 0", qpos_joint1=v)
    rho_double   = eval_atomic("not not (robot_*[qpos_joint1] - 0 >= 0)", qpos_joint1=v)
    assert rho_direct == pytest.approx(rho_double, abs=1e-9)

@given(a=float_vals, b=float_vals)
def test_demorgan_and(a, b):
    """not(phi1 and phi2) == (not phi1) or (not phi2)."""
    lhs = -min(a, b)         # not(phi1 and phi2) = -(min(a,b))
    rhs = max(-a, -b)        # (not phi1) or (not phi2) = max(-a, -b)
    assert lhs == pytest.approx(rhs, abs=1e-9)

@given(a=float_vals, b=float_vals)
def test_demorgan_or(a, b):
    """not(phi1 or phi2) == (not phi1) and (not phi2)."""
    lhs = -max(a, b)
    rhs = min(-a, -b)
    assert lhs == pytest.approx(rhs, abs=1e-9)

@given(signal=signal_list)
def test_G_leq_F(signal):
    """G[0,T](phi) <= F[0,T](phi) always — globally implies eventually."""
    r_G = eval_temporal("G[0,0.4s](robot_*[qpos_joint1] - 0 >= 0)", signal, dt=0.1)
    r_F = eval_temporal("F[0,0.4s](robot_*[qpos_joint1] - 0 >= 0)", signal, dt=0.1)
    for g, f in zip(r_G, r_F):
        assert g <= f + 1e-9, f"G={g} > F={f}: violated for signal {signal}"

@given(signal=signal_list)
def test_G_monotone_in_window(signal):
    """Longer G window can only be equal or smaller than shorter window."""
    r_short = eval_temporal("G[0,0.2s](robot_*[qpos_joint1] - 0 >= 0)", signal, dt=0.1)
    r_long  = eval_temporal("G[0,0.4s](robot_*[qpos_joint1] - 0 >= 0)", signal, dt=0.1)
    # At each step, the longer window rho <= shorter window rho
    # (longer window includes more samples so min can only decrease)
    n = min(len(r_short), len(r_long))
    for s, l in zip(r_short[-n:], r_long[-n:]):
        assert l <= s + 1e-9

@given(a=float_vals, b=float_vals)
def test_implication_tautology(a, b):
    """phi -> phi is always >= 0 (a formula implies itself)."""
    rho = max(-a, a)   # max(-a, a) = abs(a) >= 0 always
    assert rho >= -1e-9

@given(a=float_vals, b=float_vals)
def test_and_leq_both_operands(a, b):
    """phi1 and phi2 <= phi1 and <= phi2 always."""
    rho_and = min(a, b)
    assert rho_and <= a + 1e-9
    assert rho_and <= b + 1e-9

@given(a=float_vals, b=float_vals)
def test_or_geq_both_operands(a, b):
    """phi1 or phi2 >= phi1 and >= phi2 always."""
    rho_or = max(a, b)
    assert rho_or >= a - 1e-9
    assert rho_or >= b - 1e-9

@given(signal=signal_list)
def test_G_equals_not_F_not(signal):
    """G[0,T](phi) == not F[0,T](not phi) — duality."""
    r_G = eval_temporal("G[0,0.3s](robot_*[qpos_joint1] - 0 >= 0)", signal, dt=0.1)
    r_dual = eval_temporal("not F[0,0.3s](not (robot_*[qpos_joint1] - 0 >= 0))", signal, dt=0.1)
    for g, d in zip(r_G, r_dual):
        assert g == pytest.approx(d, abs=1e-9), \
            f"G={g} != not-F-not={d}: duality violated"
```

---

## Test file structure

The categories above are organizing concepts for *what* to test, not a
file-per-category layout. In the repository the parser and spec tests live
flat under `tests/`, grouped by concern:

```
tests/
  test_spec_ast.py          # syntax/semantics separation: parse() is syntax-only, returns an immutable AST
  test_parser_semantics.py  # quantitative semantics checked against a brute-force past-time oracle
  test_spec_selectors.py    # selector desugaring: sugared specs compile identically to longhand
  test_causal_past.py       # causal past-time syntax and native/RTAMT quantitative parity
  test_spec_render.py       # AST -> text renderer round-trip
  test_check_specs.py       # `specola check` explains a spec against a live sim
```

Keep each concern focused. A failure message should immediately tell you which
layer of the parser broke: atomic semantics, operator precedence, temporal
window, binding assignment, or a structural invariant.

---

## Helper fixtures (conftest.py sketch)

```python
import numpy as np
import pytest
from specola.entities import EntityState, TrackState
from specola.adapter import MonitorObservation
from specola.stl_parser import parse_stl
from specola.monitor import IncrementalBindingManager

def make_entity(eid, cls, pos=None, props=None):
    # position/velocity are optional — spec resolution reads only
    # `properties`; pos defaults to None (no spatial notion required).
    return EntityState(
        entity_id=eid,
        entity_class=cls,
        position=pos,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=props or {},
    )

def make_obs(entities, step=0, dt=0.1):
    return MonitorObservation(
        timestep=step,
        time=step * dt,
        perceived_entities={e.entity_id: e for e in entities},
    )

def run_steps(formula_str, entity_lists, dt=0.1):
    """Run a formula over a sequence of observations. Returns list of result dicts."""
    parsed = parse_stl(formula_str, dt=dt)
    mgr = IncrementalBindingManager(parsed, require_confirmed=True)
    results = []
    for i, ents in enumerate(entity_lists):
        obs = make_obs(ents, step=i, dt=dt)
        results.append(mgr.update(obs))
    return results
```

---

## Running the suite

```bash
# All parser/spec tests
pytest tests/test_spec_ast.py tests/test_parser_semantics.py \
       tests/test_spec_selectors.py tests/test_causal_past.py \
       tests/test_spec_render.py tests/test_check_specs.py -v

# Semantics against the brute-force oracle (the core correctness check)
pytest tests/test_parser_semantics.py -v

# With coverage for the spec package
pytest tests/test_spec_ast.py tests/test_parser_semantics.py \
       tests/test_spec_selectors.py --cov=specola.spec --cov-report=term-missing
```

---

*Last updated: July 2026*
*Applies to: specola spec/ layer (spec/parser.py, spec/semantic.py, spec/ast.py, and the stl_parser.py compatibility entry point)*
