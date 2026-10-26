# Adapter capabilities

One row per adapter, one column per capability an adapter can carry. A
cell says `yes` (with where the adapter reads it), `gap` (the simulator
exposes the signal and the adapter does not export it yet), or `n/a`
(the simulator has no such signal, with the reason). Silence is not an
option: a new adapter fills its row before its first line of code, from
the version-matched documentation and source of its simulator, and every
`yes` ships with a parity test. The shared key names are a floor for
specification portability, never a ceiling.

Evidence for each cell is a symbol or line in `src/specola/adapters/`
as of 2026-09-24; `base` means inherited from `MujocoAdapter`. The gap
list at the end is the backlog this table produces.

## Capabilities

| id | capability | key shape or hook |
|---|---|---|
| A | body position | `body_pos_w_<link>_<axis>`, `xpos_<body>_<axis>`, `root_pos_w_<axis>` |
| B | body orientation | quaternion per body, `body_quat_w_<link>_<c>`, `xquat_<body>_<c>` |
| C | body linear velocity | `body_lin_vel_w_<link>_<axis>`, `cvel` linear half |
| D | body angular velocity | `body_ang_vel_w_<link>_<axis>`, `cvel` rotational half |
| E | joint position and velocity | `joint_pos_<joint>`, `joint_vel_<joint>`, `qpos_<joint>`, `qvel_<joint>` |
| F | joint or link force, wrench, contact force | `link_wrench_<link>_<i>`, `cfrc_*`, `qfrc_*`, `contact_*_force` |
| G | bounding box | `aabb_min_<axis>`, `aabb_max_<axis>` |
| H | pair distance or contact entities | `mesh_pair`, `geom_pair`, `contact` entity classes |
| I | query configuration binding | reads its population and instances by binding name |
| J | drive hook | `make_advancer(mode, demo, seed)` for `specola run --mode` |
| K | scene geometry and poses | `scene_geometry()`, `scene_poses()` for the 3D panel |
| L | frame capture | `capture_frame(camera)` for snapshots |
| M | units in property metadata | `PropertyMeta(unit=...)` on every key |
| N | entity labels | `entity_labels`, task-label aliases |

## Matrix

| adapter | A | B | C | D | E | F | G | H | I | J | K | L | M | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mujoco` (base class, `mujoco.py`) | yes `xpos` :6465 | yes `xquat` :6474 | yes `cvel` :285 | yes `cvel` :285 | yes `qpos`/`qvel` :251 | yes `cfrc_*`, `qfrc_*`, `contact_*_force` :290, :96 | gap: `mjModel.geom_aabb` exists; only region bounds :8254 | yes `contact`, `geom_pair` :65, :2762 | yes `mujoco.mj_geomDistance` :1310 (population only) | yes via runner fallback (`run_monitor.py:425` steps `mj_step`); replay of a Specola state recording through the state tier :9187 to :9266 (2026-10-14) | yes :9045, :9206 | yes :8924 | gap: `unit=` on `xquat` only :6477, others `note` | yes :1192, :8630 |
| `robocasa` | base | base | base | base | base | base | gap (base; `_parallelepiped_world_aabb` :1765 feeds regions only) | base | base | base runner; base state tier, replayed with the recording's `--seed` | base | base | gap (base) | yes :997, aliases :367 |
| `libero` | base | base | base | base | base | base | gap (base) | base + contact pairs :1053 | base | base runner; base state tier | base | base | gap (region half-extents :1247 only) | yes :1575 |
| `metaworld` | base | base | base | base | base + `gripper_qpos` :230 | base | gap (base) | base | base | base runner; base state tier, replayed with the recording's `--seed` | base | base | gap (base) | yes :480 |
| `dmcontrol` | base | base | base | base | base | base | gap (base) | base | base | yes random and hold :90; replay through the base state tier | base | base | gap (base) | base |
| `mamujoco` | base | base | base | base | base | base | gap (base) | base | base | yes random and hold :65; replay through the base state tier | base | base | gap (base) | base |
| `pybullet` | yes robot root :316, links :358 | gap: robot root only :328; `getLinkState` gives link quaternions | gap: robot root only :319; `getLinkState(computeLinkVelocity=1)` | gap: robot root only :323 | yes :382, :387 | yes `jointtorque` :396, per-body contact force :332 | gap: `getAABB` used only for camera framing :222 | gap: `getContactPoints` exposes pairs | gap (follows H) | yes hold :536; replay of a Specola state recording through the state tier :605 to :725 (2026-10-14), refused for a scene with a planar joint | gap: `getVisualShapeData` exposes meshes | yes :543 | gap (`note` only) | yes :168, :284 |
| `carla` | yes `location_*` :191 | n/a: CARLA rotations are Euler; exported as `rotation_*` :192 | yes `velocity_*` :195 | yes `angular_velocity_*` :197 | n/a: actors have no joints | n/a: no force API on actors | gap: `extent_*` half-sizes :216 without min/max | gap: collision sensor exposes contacts | gap (follows H) | yes :831; replay refused, no state tier (restore probe 2026-10-14, :842) | n/a: no mesh API on the client (:68) | yes :491 | gap (`note` only :184) | yes :353 |
| `isaaclab` | yes `body_pos_w_` (`isaaclab_extraction.py:238`) | gap: `body_quat_w` read for the scene plan only (:545) | gap: `body_link_vel_w` in `ArticulationData`; only root velocity as `EntityState.velocity` | gap (same tensor) | yes :231, :233 | yes `link_wrench_` :245 | n/a: Isaac Lab has no per-body box; RoboDojo's comes from its own metadata | gap: `ContactSensor` unresolved (:335) | gap (follows H) | yes hold :282 (`make_advancer`; random refused on an unbounded action box); replay of a Specola state recording through the state tier :377 to :410 (2026-10-14), any other `--demo` refused | yes :857, :915 | yes :405 | gap (`note` only :257) | yes :510 |
| `robodojo` | yes :682, :840 | yes :693, :859 | yes `body_link_vel_w` :875 (Isaac Lab 0.54.3, link frame) | yes :890 | yes :649, :659 | yes `body_incoming_joint_wrench_b` :908 | yes RoboDojo oriented-box metadata :1090 | yes `mesh_pair` :246 (sampled Warp distance) | yes `robodojo.mesh_pair` :1476 | yes hold only :2229 (policy drives); replay refused, no state tier (restore probe 2026-10-14, :2248) | yes :1760, :1773 | yes :1862 | yes :616 onward | yes :561, :1225 |
| `dataset` | yes `px`/`py`/`pz` :330 | n/a: columns are what the file has | yes `vx`/`vy`/`vz` :335 | n/a | n/a (arbitrary columns pass through :325) | n/a | n/a | n/a | n/a | yes plays the file :120 | n/a | n/a | gap: no `property_metadata` at all | n/a |
| `mock` (`fake.py`) | n/a (position on `EntityState` only :115) | n/a | n/a | n/a | yes :80, :84 | yes :88, :218 | n/a | n/a | n/a | n/a | n/a | n/a | gap (`note` only) | yes :39 |

The MuJoCo family binds a query population only; explicit instances are
refused (`mujoco_geom_pairs.py:76`). The MuJoCo `cvel` keys are
centre-of-mass frame 6-vectors, rotational components first
(`mujoco.py:276`).

## Benchmark-native reference quantities

A benchmark whose success predicates compare against values it captured
itself needs those captures exported, or a specification restating the
predicate restates a different one. The RoboDojo adapter exports, beside
the columns above, the quantities its reward helpers read through its own
accessors (2026-09-25): `native_pos_*`/`native_quat_*`
(`LayoutManager.get_instance_pose`), `reset_pos_*`/`reset_quat_*` and
`ee_home_*` (the poses `Func_Parser.init_state` captured at reset),
`ee_pos_*`/`ee_quat_*` (`RobotManager.get_real_endpose`),
`footprint_zmin`/`footprint_zmax` and `footprint_in_<label>`
(`calc_polygon` footprints and a signed containment margin) and
`model_id` (the asset index). See `_native_object_payload` and
`_native_robot_payload` in `robodojo.py`.

## Gap list

Each entry is a signal the simulator exposes and an adapter does not
export. Close them in the adapter, with the producer source and a parity
test, or move them to `n/a` with a verified reason.

- Bounding boxes (G): MuJoCo family from `geom_aabb`; PyBullet from
  `getAABB`; CARLA min/max from `location`, `rotation` and `extent`.
- PyBullet link state (B, C, D): quaternion and velocities per link from
  `getLinkState`, not only the robot root.
- Isaac Lab body state (B, C, D): `body_quat_w` and `body_link_vel_w`
  per link, the same tensors the RoboDojo adapter reads.
- Contacts and pairs (H, I): PyBullet `getContactPoints`, CARLA
  collision sensor, Isaac Lab `ContactSensor`; each then takes a query
  configuration population and, during a monitor session, reads only the
  pairs the compiled specifications can bind, the way the RoboDojo and
  MuJoCo adapters do (issue #136).
- PyBullet scene geometry (K): `getVisualShapeData` for the 3D panel.
- Units (M): every adapter except RoboDojo carries the unit in `note`
  rather than `unit=`; the dataset adapter has no metadata at all.

Recording raw observations for offline re-evaluation is a core
capability, not an adapter one; it is tracked as issue #127.
