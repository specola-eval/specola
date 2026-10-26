# Adapter certification population

Date: 2026-08-13. Status: initial exact population inventory; certification
has not run.

## Denominator

The bundled entry-point denominator is the ten names declared under
`[project.entry-points."specola.adapters"]` in `pyproject.toml` at Specola
commit `32035e90547e3967ffaa38c033538fd8ca03dcf9`:

1. `mock`
2. `dataset`
3. `pybullet`
4. `dmcontrol`
5. `mamujoco`
6. `libero`
7. `robocasa`
8. `metaworld`
9. `isaaclab`
10. `carla`

This is an entry-point population, not ten independent physics engines.
`dmcontrol`, `mamujoco`, `libero`, `robocasa`, and `metaworld` share the
unregistered `MujocoAdapter` extraction mechanism and add distinct producer or
benchmark policy. `mock` and `dataset` are non-live producers.

Certification rows are finer than entry points. One row identifies an adapter,
producer and version or source digest, wrapper/configuration domain, and an
executable evidence population. One entry point may therefore require several
rows. A missing installation or live service is `unexecuted`, not
`not_applicable` and not a pass.

## Initial row inventory

| Entry point | Implementation family | Producer/configuration identity required | Existing strongest evidence boundary | Initial status |
|---|---|---|---|---|
| `mock` | Direct `SimulatorAdapter` | Specola commit and fixed synthetic scenario | Simulator-free unit and engine tests | Uncertified |
| `dataset` | Direct `SimulatorAdapter` | Specola commit and CSV schema/configuration | Simulator-free dataset tests | Uncertified |
| `pybullet` | Direct `SimulatorAdapter` | PyBullet package/API version and constructed scene population | Real PyBullet tests when installed | Uncertified; producer version not enforced |
| `dmcontrol` | Shared `MujocoAdapter` | `dm_control`, MuJoCo binding, task, and wrapper shape | Shared MuJoCo ledger plus adapter tests | Uncertified |
| `mamujoco` | Shared `MujocoAdapter` | MA-MuJoCo, MuJoCo binding, scenario and parallel-wrapper shape | Shared MuJoCo ledger plus adapter tests | Uncertified |
| `libero` | Shared `MujocoAdapter` plus declarations | `hf_libero`, robosuite, MuJoCo, task population, dataset/config source | Shared ledger, sim-free declaration tests, bounded real integration tests | Uncertified; producer/source pin incomplete |
| `robocasa` | Shared `MujocoAdapter` plus declarations | RoboCasa, robosuite, MuJoCo, generated task/scene population, role-table digest | Shared ledger, checked role artifact, sim-free and real tests | Uncertified |
| `metaworld` | Shared `MujocoAdapter` plus declarations | Exact Meta-World release/source map, MuJoCo binding, all supported tasks | Exact 50-task source map, shared ledger, sim-free and real tests | Uncertified |
| `isaaclab` | Direct `SimulatorAdapter` | Isaac Lab/Isaac Sim/PhysX versions, scene kinds, vectorized environment index and device | Simulator-free fake-asset tests and bounded live probes | Uncertified; runtime version not enforced |
| `carla` | Direct `SimulatorAdapter` | CARLA client/server versions, map, synchronous settings and sensor configuration | Simulator-free fake-world tests and separate stage probe | Uncertified; live server evidence required |

## Existing closed denominator that may be reused

The MuJoCo field contract covers the recorded union of public, non-callable
`MjData` binding fields for MuJoCo 3.2.3, 3.3.0, 3.3.1, 3.8.1, and 3.10.0,
plus three robosuite wrapper aliases. It does not cover all of `MjModel`, task
declarations, assets, accepted scene kinds, or a complete deformable-topology
contract. Those are separate certification denominators.

Shared evidence may be inherited by one of the five MuJoCo-family entry points
only after its actual binding version and applicability are recorded. The
adapter-specific declaration and reconciliation policy remains a separate row
obligation.

## Measured local baseline

On 2026-08-13, the `libero` conda environment reported:

```text
Python 3.12.13
numpy 2.4.6
pytest 9.1.1
mujoco 3.8.1
robosuite 1.4.0
hf_libero 0.1.4
```

This contradicts older workspace prose that described robosuite 1.4.1. The
runtime measurement is evidence for this run only; the durable producer pin
must be corrected from version-matched package/source inspection before the
LIBERO row can pass.

The base conda Python did not contain pytest. The first attempted baseline is
retained in
`benchmarks/results/adapter_fidelity/logs/baseline-sim-free.log`; it is an
environment failure, not a test result. The worktree-source core baseline used
the RTAMT-capable `robocasa` environment and excluded all simulator markers:

```text
8 failed, 1700 passed, 25 skipped, 157 deselected, 1 xfailed
```

All eight failures have the same pre-existing cause: the diagnostic catalogue
records historical-ledger SHA-256
`8efcc6e6c6182e22b2118ceedc3b678c65abb625492a6dc257586a94e2b7b903`, while
the checked file hashes to
`9dac9bd3ec66fbbaaff6d8c66a888e064945721334bec1626affe7f09c78e757`.
This is a measured baseline failure, not evidence that the certification work
passes. The complete output is retained in
`benchmarks/results/adapter_fidelity/logs/baseline-core.log`.

The local environment inventory measured on the same date is:

| Environment | Python | Relevant installed producers |
|---|---:|---|
| `robocasa` | 3.11.15 | RoboCasa 1.0.1; robosuite 1.5.2; MuJoCo 3.3.1 |
| `libero` | 3.12.13 | `hf_libero` 0.1.4; robosuite 1.4.0; MuJoCo 3.8.1 |
| `metaworld` | 3.11.15 | Meta-World 3.1.1; MuJoCo 3.3.0 |
| `dmcontrol` | 3.11.15 | `dm-control` 1.0.43; MuJoCo 3.10.0 |
| `mamujoco` | 3.11.15 | MuJoCo 3.10.0; MA-MuJoCo package/source identity unresolved |
| `isaacsim` | 3.12.13 | Isaac Lab 6.1.17; Isaac Sim 6.0.1.0 |
| `carla` | 3.12.13 | CARLA client 0.9.16; live server identity unresolved |

Installed-package versions establish the local runtime population only. They
do not establish API semantics, wrapper provenance, live server compatibility,
or certification coverage.

## Evidence needed per row

Every final row records:

- exact entry-point module, factory and adapter class;
- producer name, version/source digest, wrapper topology and supported
  configuration population;
- candidate-denominator enumeration method and digest;
- accepted native scene/item kinds and conditional obligations;
- inclusion, exclusion and unresolved counts;
- discovery, metadata and materialization result;
- structural trace-conformance result;
- criterion-noninterference and scoped-projection result where applicable;
- simulator-free and live-runtime commands, complete logs and dates;
- assurance label per obligation;
- `pass`, `fail`, `unexecuted`, or `not_applicable`, with the last permitted
  only when the obligation truly does not apply to the declared supported
  domain.

The executable obligation denominator is
`specola.conformance.ledger.SPECOLA_V1_CATALOGUE`: 14 core obligations and 11
conditional capability/facility obligations. A row cannot substitute a custom
catalogue for publication certification. Exact catalogue coverage,
obligation-specific evidence sufficiency, and the overall result are validated
by `validate_specola_certification` and `is_certified`.

## Status 2026-09-02: obligation `CORE.STATIC_SLOT_BINDABILITY` executed

Catalogue version 1.1 adds a fifteenth CORE obligation,
`CORE.STATIC_SLOT_BINDABILITY` (plan
`docs/superpowers/plans/2026-09-02-never-bound-static-slot.md`; probe
`src/specola/conformance/bindability.py`). The probe takes one
representative key per static class and component group, compiles a
one-slot specification against the reset catalog, then watches a fresh
adapter under extraction scope and requires an evaluation on the first
step. Every row driver ran in its own environment on 2026-09-02; logs
under `results/adapter-certification-2026-09/`.

| Entry point | Probe outcome | Cases | Row result for the obligation | Note |
|---|---|---|---|---|
| `mock` | pass | 6 | PASS | |
| `pybullet` | pass | 17 | PASS | two-URDF row scene |
| `libero` | pass | 200 | PASS | pin byte-identical across two runs |
| `dataset` | not applicable | 0 | UNEXECUTED | open class vocabulary; the static rule does not apply |
| `dmcontrol` | not applicable | 0 | UNEXECUTED | open class vocabulary |
| `metaworld` | not applicable | 0 | UNEXECUTED | open class vocabulary |
| `mamujoco` | not applicable | 0 | UNEXECUTED | open class vocabulary |
| `robocasa` | setup unavailable | 245 compiled, 0 decided | UNEXECUTED | the factory samples a new kitchen per reset, so the runtime leg watches a different scene than the compile leg; the probe reports this rather than a false pass or fail. Seeding the global generators before each factory call does not fix it: robosuite builds a per-environment `np.random.default_rng(seed)` and RoboCasa leaves that seed unset (`robosuite/environments/base.py:144-145`); measured again seeded on 2026-09-02 in `results/adapter-certification-2026-09/robocasa_probe_seeded.log`. The X15 pre-run check (`results/x15-freeze-prep-2026-09/check_entry/`) covers RoboCasa on fixed scenes; tracked as issue #122; the contract test marks RoboCasa as a strict expected failure |
| `carla` | not run | | UNEXECUTED | no live producer in this environment |
| `isaaclab` | not run | | UNEXECUTED | no live producer in this environment (Isaac Sim 5.1 driver block) |

The `carla` row now runs the probe whenever its live probe reports an
available server, so that row advances on its own once CARLA is
available. The `isaaclab` row is source-only: it has no app-backed
execution route, so no live probe path exists for it yet and its
obligation stays unexecuted by construction rather than by environment.

No adapter recorded FAIL. Overall certification status is unchanged:
`is_certified` needs every CORE obligation at PASS and every applicable
CONDITIONAL obligation at PASS, and the remaining obligations are still
`unexecuted` on every row except `mock` (see the 2026-08-20 amendment in
the certification design). The four open-vocabulary adapters cannot
pass this obligation until they declare `supported_classes`; that is an
adapter decision, not a probe defect. Full certification of all ten
entry points is the maintainer's standing target (2026-09-02); the work
packages are the ones in #107, #113 and #114.
