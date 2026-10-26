# PyBullet native-accessor inventory

Date: 2026-08-15. Status: source-level inventory; this artifact is not a
backend-completeness certification.

## Population and provenance

The measured population is the PyBullet adapter source at
`src/specola/adapters/pybullet.py`, audited against PyBullet 3.2.7 source and
the locally installed extension. The inventory uses the finite set of native
accessor callsites actually present in the adapter, rather than the complete
PyBullet API. This scope is intentional: absence from this artifact does not
mean that a PyBullet API exists, is irrelevant, or is faithfully represented
by Specola.

The pinned evidence is
`pybullet_3.2.7_native_accessor.json`.
The instrument is
`pybullet_source_inventory.py`,
and the command/custody record is retained in
`benchmarks/results/adapter_fidelity/logs/pybullet-source-inventory-2026-08-16.log`.

| Evidence | Value |
|---|---|
| PyBullet package/source version | `3.2.7` |
| PyBullet API version reported by installed extension | `202010061` (direct measurement via `pybullet.getAPIVersion()` in the pinned environment) |
| source distribution SHA-256 | `042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f` |
| `examples/pybullet/pybullet.c` SHA-256 | `9c7be986f599fcb6e327542fa65786f5099f549a19bf7bae7ac8d544f52836d6` |
| installed extension SHA-256 | `54596239c69a8ed206759da5388dc69e3b9820815100fb20819d4fabdcda0bcf` |
| adapter source SHA-256 | `4a1d71e5a2ad2d7ad0bcd35f2fa98a3244bef477cbde36087a0505fc5b3a9eec` |

The `SpamMethods` table contains exactly 144 unique registered Python
accessors. The adapter uses 21 distinct accessors at 27 source callsites. All
27 used callsites resolve to a registered function in the pinned C table. The
visitor covers every `PyBulletAdapter` method and the reachable module-helper
closure (`_make_pybullet`, `_maybe_load_egl`, and `_resolve_urdf`); nested
functions receive their own qualified scope. This is a measured source
result, not a claim that the 144-accessor native surface is complete or
covered.
The audited qualified-symbol set is an acceptance invariant: it must equal
the enumerated expected scope set, including methods with no native callsites.
Missing scopes or collapsed call-count scopes therefore fail closed.

The C lexer removes comments while preserving line numbers before parsing the
table, so a commented-out registration cannot satisfy a live entry. Runtime
package/API/module-path/hash values are measured by importing PyBullet in the
selected environment; `--extension` is only an optional expected-path check.
The local module path is raw-custody-only: the durable pin records the
extension filename, ABI-bearing suffix, and SHA-256 instead. Tests inject a
verifier instead of replacing the production measurement path.
The upstream pinned 3.2.7 table contains its own conditional registration
directives; those are accepted only when the complete official C hash matches
the custody pin. Any synthetic or mutated C containing a directive is rejected
before registration parsing.

## Supported audit grammar (fail-closed)

The visitor intentionally recognizes a small source grammar; it is not a
general Python data-flow analysis. A bound native call must be a direct
attribute call on `self._p`, on the method's declared `pb` parameter, or on
the directly imported `pybullet` module in `_make_pybullet`. The only adapter
alias permitted is one top-level method-body statement `pb = self._p`, which
must precede every `pb` load. The `self._p = pybullet` initialization and an
unaliased `import pybullet` are the sole accepted producer bindings.

The pre-scan rejects rebinding or capture of these backend names through
tuple/list/star targets, annotated/chained/augmented assignments, named
expressions, `for`/`async for`, `with`, exception handlers, `del`, import
aliases, `global`/`nonlocal`, parameters, and pattern-match captures. It also
rejects storing a backend handle or bound backend attribute in another local.
Conditional, dead-code, nested-scope, lambda, comprehension, and unsupported
nested-function aliases/calls cannot satisfy a callsite. The existing
`PyBulletAdapter.make_advancer.advance` scope is the one explicit reachable
nested-function exception and is recorded under its qualified callsite key.
Module helper aliases are not resolved: reachable helpers must be called by
their function name directly. Any local binding or indirect call of a helper
symbol is rejected, including annotated, tuple/list/star, walrus, loop,
`with`/exception, deletion, import, `global`/`nonlocal`, and pattern-match
forms. Unsupported syntax fails closed instead of being treated as evidence
of absence.
Backend roots and aliases may only occur as the receiver of a direct
`receiver.accessor(...)` call, apart from the adapter's declared constants and
exception type uses. `getattr`-based receivers, stored handles, pass-through
arguments, and other indirect consumption are rejected.

## Classification

Classification is attached to a callsite, not only to an accessor name. The
same accessor can therefore be included in the monitor contract in one method
and excluded as camera support in another method. The instrument also records
the receiver binding (`self._p`, the adapter's `pb` alias, or the factory's
`pybullet` module) and tuple-index paths. A mutation that changes a tuple
index, removes a required call, or routes a call through an unrelated
receiver fails the inventory rather than being silently accepted.

| Boundary | Distinct accessors | Callsites | Classification |
|---|---:|---:|---|
| episode timing, identity and schema | 6 | 8 | included |
| observation and contact state | 5 | 5 | included |
| camera framing/capture | 5 | 6 | excluded from monitor state |
| renderer setup | 1 | 2 | excluded from monitor state |
| simulation control | 1 | 1 | excluded from monitor state |
| scene construction | 4 | 4 | excluded from monitor state |
| resource lifecycle | 1 | 1 | excluded from monitor state |

The complete callsite records, including source lines, receiver bindings,
native C registration lines, and tuple paths, live in the JSON pin. Important
paths include:

- `getPhysicsEngineParameters["fixedTimeStep"]` for the episode timestep;
- `getBodyInfo[1]` for body-name identity and liveness;
- `getJointInfo[1]`, `[2]`, and `[12]` for joint name, type, and link name;
- `getBasePositionAndOrientation[0]` and `[1]` for position and quaternion;
- `getBaseVelocity[0]` and `[1]` for linear and angular velocity;
- `getJointState[0]`, `[1]`, and `[3]` for position, velocity, and applied
  motor torque;
- `getLinkState[0]` for link world center-of-mass position;
- `getContactPoints[1]`, `[2]`, and `[9]` for body identities and normal
  force;
- `getCameraImage[2]` for the RGB/RGBA image component, which is excluded
  from physical monitor state.

## Boundary and gaps

The included rows establish only that the current adapter's finite accessor
projection is explicitly enumerated and source-linked. They do not establish
semantic correctness of every returned value, full producer coverage, or
support for native features not called by the adapter. In particular, this
inventory does not certify multi-DOF joint accessors, soft-body/deformable
mesh topology, dynamics/material fields, collision-shape geometry, visual
shape geometry, user data, or any of the other native APIs absent from the
adapter callsites. Those require separate capability denominators and real
scene evidence before a capability profile can pass.

Within the measured 27-callsite denominator, the unresolved count is zero.
That zero must not be read as zero unresolved fields in the PyBullet backend:
the instrument intentionally does not enumerate the unused native surface.

## Fixed-scene runtime row

`pybullet_3.2.7_certification_row_v1.json`
is the companion real-entry-point runtime row. It loads two checked fixtures
through the installed `pybullet` entry point and captures four fresh paired
reset/get-observation variants: unscoped, proper nonempty scope, empty scope,
and criterion poison at the proper scope. The exact fixed surface contains 43
class-qualified fields at 43 single-owner addresses and 86 call-major
witnesses. Variant output and native field-read dependencies are 86, 8, 0,
and 8; explicit host/device transfer denominators are empty. A separate
six-accessor invocation counter retains structural-getter multiplicity and
the exact native-pruning evidence that field dependencies cannot encode.

The fixed subject has a closed 43-field semantic manifest and one externally
supplied, content-hashed two-URDF configuration/native kind. Its exact 25-row
publication ledger contains 14 validated core `PASS` rows followed by 11
conservative applicable `UNEXECUTED`/`UNVERIFIED` conditional rows. The row is
still deliberately **uncertified** as a PyBullet backend: it does not classify
arbitrary URDF body, joint, link, or native-state kinds, infer conditional
capability presence or absence, or return an aggregate certification result.
