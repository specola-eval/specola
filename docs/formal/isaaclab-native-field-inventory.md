# Isaac Lab native-field inventory

Date: 2026-08-15. Status: source-level investigation plus the committed
2026-08-14 scalar-plan refactor (`5362fe1`). Certification status: unresolved.

## Producer boundary

This inventory covers the locally installed editable Isaac Lab source at:

- extension version `6.1.17`;
- Git commit `6a7acb0320a0bdc15b13e44e83b575e00797faf4`;
- description `v3.0.0-beta2.patch1-11-g6a7acb032`;
- branch `release/3.0.0-beta2`;
- externally supplied Isaac Sim package claim `6.0.1.0` (not a bundled runtime measurement);
- PhysX-backed data implementations.

The version, commit, backend and source population are all part of the
denominator. Results must not be generalized to another Isaac Lab checkout or
the Newton data implementations without rerunning the inventory. The
version-matched public reference is the
[Isaac Lab 3.0.0-beta2 asset API](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/api/lab/isaaclab.assets.html).

## Scene-kind denominator

`InteractiveScene.keys()` mechanically enumerates these populations, in this
order:

1. terrain;
2. articulations;
3. deformable objects;
4. rigid objects;
5. rigid-object collections;
6. sensors;
7. surface grippers;
8. extras.

The committed 2026-08-14 refactor compiles an Isaac-local plan from the
documented family mappings. It includes articulations, rigid objects, and an
explicitly incomplete deformable compatibility profile; rejects rigid-object
collections and surface grippers; records extras as excluded; and records
sensors as `unresolved_sensor_kind_not_materialized`. The latter is not a
claim that every sensor is a camera: RGB camera capture is independently
detected and remains separate from physical monitor extraction. Every mapped
key must appear in `scene.keys()`, and every nonstatic scene key is classified
once. A physical asset whose key happens to be `ground` or `light` remains a
physical asset rather than being discarded by name.

The refactor has not certified the semantic field denominator. In particular,
the `root_view.get_link_incoming_joint_force()` surface is not emitted at all:
it is retained as an unresolved semantic candidate rather than dynamically
changing discovery according to reset lifecycle. Isaac Lab's PhysX kernel
states that an incoming wrench can retain pre-reset data at timestamp zero
(source `isaaclab_physx/sensors/joint_wrench/kernels.py`, lines 24--38; see
also Specola issue #105). A future lifecycle-qualified manifest must use
per-environment episode progress, not a global counter. This intentional delta
from the historical adapter surface is not evidence of end-to-end fidelity.

## PhysX data API denominator

The initial field denominator is the union of public class-level annotated
attributes and public `@property` getters on each backend-neutral base data
class and its concrete PhysX implementation. This is a source-API name
denominator, not yet a semantic state denominator: it intentionally includes
aliases, defaults, commands, derived quantities, and metadata that later need
categorical classification.

| Family | Enumerated public names | Included | Excluded | Unresolved |
|---|---:|---:|---:|---:|
| Articulation data | 132 | 5 | 80 | 47 |
| Rigid-object data | 70 | 3 | 49 | 18 |
| Deformable-object data | 7 | 1 | 3 | 3 |
| Rigid-object-collection data | 86 | 0 | 86 | 0 |
| **Total** | **295** | **9** | **218** | **68** |

The partition is one explicit record per candidate, in exact pin order. The
checked policy records exactly 114
`shorthand_deprecated_compatibility`, 7
`non_state_operational_or_cache`, 7 `initial_or_default_template`, 4
`command_or_target`, and 86 `outside_supported_domain` exclusions. Unresolved records retain one of two reasons:
`configuration_dependent` (26 mutable configuration fields: 24 articulation
joint/tendon/body fields plus rigid-object `body_mass` and `body_inertia`) or
`dynamic_population` (42 fields across articulation, rigid-object, and
deformable families). The default/template spellings are excluded separately,
not counted as unresolved configuration fields. Alias
records carry explicit `canonical_of` targets; missing targets and cycles fail
validation. No prefix or name-pattern inference participates.

Current included spellings are exactly articulation
(`joint_pos`, `joint_vel`, `body_pos_w`, `root_pos_w`, `root_lin_vel_w`), rigid
object (`body_pos_w`, `root_pos_w`, `root_lin_vel_w`), and deformable object
(`root_pos_w`). Each binds to the current scalar reader and carries source
declaration provenance from base and concrete data classes. These nine are not
yet certifiable: 68 unresolved records remain. Collections are rejected as an
unsupported monitor domain rather than partially admitted. Topology, wrench
view methods, and sensors are separate denominators, outside these 295 names.

The pin also stores the canonical SHA-256 of the complete four-family
`source_declarations` mapping. Validation requires that digest to match the
trusted IsaacLab 6.1.17 schema even when no source root is supplied; when
source roots are supplied, the declarations and file hashes are checked
against the AST and files as well. A second trusted digest binds the ordered
IsaacLab producer-source records, so rootless validation also rejects a
changed or coordinately re-signed embedded source-file hash.

The pre-refactor articulation reader called
`root_view.get_link_incoming_joint_force()` when the asset exposed that
method. The current scalar plan records that accessor only as an unresolved
candidate; it does not call or emit it. It belongs to a separate
asset/view-method denominator and is not counted among the data-property names
above.

The tracked, rerunnable AST instrument is
`benchmarks/adapter_fidelity/isaac_source_inventory.py`.
Its pinned source evidence is
`benchmarks/adapter_fidelity/evidence_pins/isaaclab_6.1.17_physx_isaacsim_6.0.1.0.json`.
That pin retains the adapter SHA at generation time; later adapter refactors
must create a new clearly dated pin rather than rewriting this historical
evidence.
The historical overlap column was syntactic. The new records preserve that
source provenance while making the investigation partition explicit. The pin
is not an `exhaustively_classified` certification result because unresolved
members remain.

## Deformable result

The seven public names on the installed PhysX deformable-data surface are:

- `default_nodal_state_w`;
- `nodal_kinematic_target`;
- `nodal_pos_w`;
- `nodal_state_w`;
- `nodal_vel_w`;
- `root_pos_w`;
- `root_vel_w`.

The pre-refactor adapter accepted the deformable asset structurally because it
had `.data`, read only `root_pos_w`, and emitted that derived mean nodal
position. It did not read `root_vel_w` because the generic reader asked for
the rigid/articulation alias `root_lin_vel_w`. The current plan instead admits
the documented deformable family explicitly, preserves the derived
root-position compatibility behavior, and keeps velocity/nodal state/topology
outside its incomplete profile.

Therefore the current adapter does observe one derived deformable quantity,
but it does **not** extract garment nodal state or topology. The seven-name
data surface itself contains no connectivity field. Physics connectivity and
rest configuration require a separate asset/PhysX/USD denominator; render
mesh connectivity cannot substitute without verified correspondence.

## Remaining work after the scalar-plan classification

The tracked enumerator and source pin now carry the exact 295-record partition.
Remaining certification work:

- enumerate the unresolved sensor and arbitrary-extra semantic surface;
- enumerate asset/view methods used for link wrenches and deformable mesh
  structure;
- resolve the 26 configuration-dependent and 42 dynamic-population records;
- preserve the current emitted projection in the legacy-parity fixture.

## Live family execution

The live population is one installed PhysX scene per family, executed after
commit `5362fe1`:

- `Isaac-Forge-PegInsert-Direct-v0` contained three articulations and one
  terrain entry. The adapter materialized 63 scalar properties and 63
  matching metadata records, retained 90 raw-wrench fields as unresolved
  candidates, emitted zero wrench values, and left no scene key unclassified.
- A one-cube `RigidObjectCfg` scene materialized root position and velocity
  plus three body-position scalars and three matching metadata records.
- A one-cube `DeformableObjectCfg` scene materialized only the derived root
  position under `deformable_legacy_incomplete`; velocity, scalar fields,
  nodal state, and topology remained absent.
- A one-cube `RigidObjectCollectionCfg` scene failed reset with
  `Isaac Lab rigid_object_collections entry 'cubes' is unsupported for monitor
  extraction`, as required by the declared supported domain.

The full local logs are
`benchmarks/results/adapter_fidelity/logs/isaac-live-adapter-probe-2026-08-14.log`
and
`benchmarks/results/adapter_fidelity/logs/isaac-live-family-probes-2026-08-14.log`.
These four single-scene probes establish execution of the family-classification
paths; they do not close the 132/70/7/86 field classifications or certify the
backend.
