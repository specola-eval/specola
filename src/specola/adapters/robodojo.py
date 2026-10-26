"""
robodojo.py — adapter for the RoboDojo manipulation benchmark.

RoboDojo (https://robodojo-benchmark.com/) is an eval-only benchmark of 42
Isaac Sim manipulation tasks. It is built on Isaac Lab but does not present
an Isaac Lab environment: a ``TaskEnv`` owns three managers -- a
``RobotManager`` holding one Isaac Lab ``Articulation`` per embodiment, a
``SceneManager`` holding per-environment dictionaries of Isaac Sim core
prims (rigid bodies, articulated fixtures, garments, geometry, fluids), and
a ``CameraManager``/``TiledCaptureManager`` pair holding per-environment
cameras. Nothing in that structure is reachable through the Isaac Lab scene
API the ``isaaclab`` adapter walks, which is why RoboDojo gets its own
adapter rather than a factory argument on that one.

What is shared with the ``isaaclab`` adapter is the property vocabulary:
``joint_pos_<joint>``, ``joint_vel_<joint>``, ``body_pos_w_<link>_<axis>``
and ``link_wrench_<link>_<i>`` mean the same thing here, deliberately, so a
specification written against one Isaac-family adapter transfers to the
other. The shared names are a floor, not a ceiling: this adapter also
exports what RoboDojo's Isaac Lab exposes and the other adapter does not,
such as link velocities (``body_lin_vel_w_<link>_<axis>``,
``body_ang_vel_w_<link>_<axis>``, link frame, world axes).

Two properties of RoboDojo shape this adapter.

**It is vectorized.** A RoboDojo run is ``num_envs`` independent task
instances behind one simulation app, each with its own layout seed, its own
scene objects and its own cameras. One adapter instance is bound to one
``env_index`` and reports that environment only; monitoring N of them means
N adapters and N engines. Unlike Isaac Lab's shared viewport, RoboDojo's
cameras are per environment (the tiled capture is indexed by environment),
so a frame *is* attributable to one monitored scene and ``camera_names``
reports cameras at any ``num_envs``.

**Its scene population is per episode, not per process.** Every
``TaskEnv.reset()`` reloads the layout: objects are cleared and respawned as
new Python objects, and which objects exist depends on the layout seed.
The scene is therefore re-scanned on every observation rather than indexed
once at construction. Entity ids are keyed by the layout's instance name,
which is what RoboDojo declares identity with, so an object that survives a
reset keeps its id and a new object gets a new one. The collection an
object lives in is its class rather than part of its identity: an instance
that changed collection mid-episode would be an entity changing class,
which the base adapter forbids and which is raised here instead of being
papered over with a second id.

Semantic boundary
-----------------
Only engine state is extracted. RoboDojo's ``RewardManager`` publishes
exactly the predicates a monitor would otherwise have to rediscover --
``is_stacked``, ``is_axis_up``, ``is_all_gripper_open`` -- and none of them
is read here, nor is ``get_score``/``success``. A monitor fed a benchmark's
own verdict agrees with it by construction and can only ever restate it.

Two tables of the reward manager are read, because they are state rather
than verdicts: the poses ``Func_Parser.init_state`` captures at reset
(every instance's pose, every arm's end-effector pose). RoboDojo's
``is_lift`` and ``all_robot_back_to_origin`` are relative to those
captures, and a specification that substitutes a constant or a later
re-read restates a different predicate (2026-09-24: a constant rest height
misread the conveyor lift in 34 of 40 layouts). They are exported as
``reset_*`` and ``ee_home_*`` beside the quantities the predicates compare
them with, which are read through RoboDojo's own accessors rather than
re-derived: ``native_*`` (``LayoutManager.get_instance_pose``, the
environment-frame pose every reward helper reads), ``ee_*``
(``RobotManager.get_real_endpose``), ``footprint_*`` (``calc_polygon``
over the asset's oriented-bbox vertices, the footprint of
``is_A_cover_B``) and ``model_id`` (the asset index
``get_label_cat_index`` orders by). What the predicates then compute from
them -- thresholds, strictness, conjunctions -- is left to the
specification.

``ObservationManager.get_obs`` is not read either. It is the policy-facing
observation: it is filtered by the run's observation config, it withholds
object state from the policy by design, and reading it would make the
monitor describe what the policy was told rather than what happened.
Identity is the one thing taken from RoboDojo's declarations: the task YAML
assigns labels (``bowl0``, ``bowl1``) through ``select_mode.label``, and
those labels become entity labels here -- names for things, not state.

Verification status (2026-09-21, live pass completed 2026-09-23)
----------------------------------------------------------------
Verified live on Isaac Sim 5.1 / Isaac Lab 0.54.3, ``stack_bowls``
layout 0. The factory builds the scene in ~45 s and the adapter reports
six entities -- two robots (159 keys each: 8 joints, 11 links with pose,
orientation and wrench) and four scene objects, three of them labelled
``bowl0``/``bowl1``/``bowl2`` from the task YAML -- with metadata parity
holding on every one. All three cameras return 480x640x3 frames with
content (mean 90-97, essentially every pixel non-zero), upright, in RGB
rather than BGR, each showing its own view; the head camera's three
bowls sit where the extracted positions say they do, which is the
cross-check between the two halves of this adapter.

Getting there required fixing the WORKSTATION, not this adapter: with
two NVIDIA Vulkan ICD manifests installed (a packaged
``/usr/share/vulkan/icd.d/nvidia_icd.json`` and an unowned
``/etc/vulkan/icd.d/nvidia_icd.json``), Isaac logs "Multiple Installable
Client Drivers (ICDs) are found for the same GPU" and rendering hangs
INTERMITTENTLY -- Kit's own log says "Hang detected" -- including inside
RoboDojo's own ``create_eval_env``/``reset()`` with no specola code in
the process. Scoping the loader to one manifest
(``VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json``) cleared
both the warning and the hang, and is what the runs above were made
with. This adapter deliberately does not set that variable for the
caller: a machine with duplicate ICDs has a problem worth fixing rather
than masking, and which manifest is the right one is the machine's
business.

A second live run on ``pick_from_conveyor_by_image`` covers the scene
types ``stack_bowls`` has none of: ten entities, including the conveyor
reported as a ``dynamic`` entity with a pose and no velocity, parity
holding on all of them.

Verified live on 2026-09-23 (Isaac Sim 5.1.0, driver 580.178.04, RTX
5090, ``stack_bowls`` layout 0): the ``specola run --web`` dashboard
pass, the 3D pose panel and the pair distances below. ``specola
discover`` with ``ROBODOJO_DISTMAX=0.5`` reports 20 entities, the six
above plus 14 ``mesh_pair`` rows (every robot-object and object-object
pair); the dashboard shows all 14 distances live, and the 3D panel
draws the room, table, both arms and the bowls from 32 bodies and 107
mesh geoms. The robot-pair distances read then (0.005 to 0.42 m, the
right arm 4.7 mm from the camera stand it is mounted beside) and the
arms drawn then had every arm link turned 120 degrees: ``_body_poses``
applied Isaac Lab 6.x's (x, y, z, w) reorder to Isaac Lab 0.54.3's
``body_link_pose_w``, whose quaternion is already (w, x, y, z). Fixed
2026-10-08, when link orientations matched the USD stage's to 0.03
degrees and the same configuration read 0.11 to 0.48 m, the right arm
12.0 cm from the camera stand. The first two camera frames of a run are grey
(RTX materials still streaming; colour from frame 5), which the
warm-up fence counts as content. The one-off setup a checkout needs is
``utils/update_embodiment_config_path.py``, which writes each robot's
``curobo.yml`` from its ``*_tmp.yml`` with absolute paths.

The same day found three lifecycle traps in the process that hosts
the app, none the driver's: Kit hangs at interpreter exit unless
``SimulationApp.close()`` runs first; Isaac Lab's kit file arms a 120 s
hang detector that fires while ``--web`` serves the dashboard after the
episode; and Ctrl-C on that serving dashboard was swallowed, because
the CLI waited on an untimed ``Event().wait()`` that only wakes when
the kernel hands the signal to the main thread itself, which in a
process with Kit's 160-odd threads it did not (specola's bug, fixed
in ``run_monitor``). The factory registers the close at launch, starts
Kit with the detector off, and restores Python's default SIGINT
handler so Ctrl-C follows the CLI's path into that close rather than
``SimulationApp``'s own handler, which shuts Kit down without
``close()`` (``_ensure_simulation_app``).

3D pose panel and pair distances
--------------------------------
``scene_geometry``/``scene_poses`` give the dashboard one row per robot
link and per posed scene object (rigid, dynamic, articulation, geometry),
with each body's USD meshes baked into its frame, plus static fixtures
(``/World/background`` and unowned children of the environment prim)
baked in world coordinates. Garments and fluids deform and get no row.
The mesh walk is shared with ``adapters/isaaclab.py``
(``adapters/_usd_scene.py``); the plan is rebuilt on every reset because
RoboDojo respawns scene objects per layout.

Isaac exposes no mesh-to-mesh distance to Python, so with ``distmax``
(or ``ROBODOJO_DISTMAX``) set the adapter samples one: every vertex of
each mesh against the other's surface via Warp's ``mesh_query_point``,
both directions, minimum (``adapters/_mesh_distance.py``). Every
unordered pair of robots and posed objects is then a ``mesh_pair``
entity carrying ``mesh_distance_sampled`` (metres, capped at
``distmax``, zero when a vertex lies inside the other mesh) and
``mesh_distance_clipped`` (1 at the cap), tagged with both labels and
``robot_pair`` (arm to arm, the bimanual clearance a safety spec selects
with ``@ robot_pair``), ``robot`` or ``environment``. Collision hulls are used when a body has any, else its
visual meshes. The value is exact when the closest approach touches a
vertex and an over-estimate otherwise; it is reported as sampled, never
as an exact clearance, and penetration depth is not measured. Without
``distmax`` no pair exists and no distance work is done.

The query configuration (``--queries``, ``Monitor.watch(queries=...)``)
scopes that work the way the MuJoCo family's does: a
``robodojo.mesh_pair`` population carries ``distmax`` (overriding the
constructor and the environment) and an optional ``robot_links`` list of
the arm links to sample, and its instances name the pairs to compute by
entity label (operands ``a`` and ``b``). Every pair is formed when no
instance is named. Under a monitor session's entity scope (issue #136)
an observation after reset then reads only the formed pairs a compiled
specification slot can bind, as the MuJoCo family does for
``geom_pair``; ``population: all`` in the population reads every formed
pair, and named instances are always all read, whatever the specs bind
(``_pair_in_scope``). A bounding-sphere test on each body's merged mesh
skips any pair that cannot lie within ``distmax``; it then reports the
cutoff, clipped, exactly as a queried far pair would. Measured on
``stack_bowls`` with ten parallel environments (2026-09-23): every pair
over every link cost seven times the unmonitored run; the sampled
distances during a clamped grasp read 1 to 24 cm, which was put down to
coarse finger meshes, so a grasp was read from the finger link's
joint-reaction force (0.1 N open, 23 N clamped) instead. Those grasp
distances were read with the arm links turned (the 2026-10-08 fix
above) and have not been re-measured, so neither the 1 to 24 cm nor
its explanation is established; the distances remain a clearance
measure.

Every posed scene object also carries ``aabb_min_<axis>`` and
``aabb_max_<axis>``: the world-axis bounding box of RoboDojo's own
oriented-bbox metadata for that instance, carried by the current pose.
RoboDojo's container goals (``is_A_in_B``, ``is_A_on_B_bottom``,
``is_A_cover_B``) are box tests against per-asset sizes that change with
the layout's asset pick, so a specification restating them needs the box,
not just the pose. The axis-aligned box over-admits a rotated footprint by
its corners; RoboDojo tests the rotated polygon.

``make_advancer`` supports ``--mode hold`` only. RoboDojo is driven by a
policy server over a WebSocket and its action interface is a per-robot
``MetaControl`` dictionary rather than a gym action vector, so a "random
action" drive mode would have to invent an action semantics the benchmark
does not define; ``--mode random`` and ``--mode replay`` say so instead of
guessing. What a monitored policy run looks like is wrapping the caller's
own env (``registry.wrap_env``) inside RoboDojo's eval loop.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..entities import EntityState, TrackState
from ..registry import register as _register
from ._usd_scene import (
    MeshCollector,
    link_prim_paths,
    merge_meshes,
    pack_scene,
    stage_is_metres_z_up,
)


# Scene-object collections queried per observation, in report order.
#
# SceneManager also answers to the type name "dynamic", which aliases the
# SAME dictionary as "rigid" (``_get_object_collections_by_type`` maps both
# to ``_rigid_and_dynamic_objects``). Querying both would report every rigid
# object twice, under two entity ids, so only "rigid" is listed here.
_OBJECT_COLLECTIONS: Tuple[str, ...] = (
    "rigid",
    "articulation",
    "garment",
    "geometry",
    "fluid",
)

# Entity classes for scene objects. "dynamic" is not a collection of its
# own -- RoboDojo stores DynamicObject alongside the rigid bodies -- but it
# IS a declared layout type with different physics (a USD xform the scene
# animates, not a PhysX body), so it is reported under its own class
# rather than filed as a rigid body that mysteriously has no velocity.
_OBJECT_ENTITY_CLASSES: FrozenSet[str] = frozenset(
    set(_OBJECT_COLLECTIONS) | {"dynamic"}
)

# Layout-record type names, which are capitalised where the collection names
# are not. "Dynamic" is included here because the label table is a lookup by
# instance name -- no object is reported twice by reading one more record
# list.
_LAYOUT_RECORD_TYPES: Tuple[str, ...] = (
    "Rigid",
    "Dynamic",
    "Geometry",
    "Articulation",
    "Garment",
    "Fluid",
)

# Scene-object classes whose prim is a rigid body with one root pose, the
# ones that get a 3D-panel row and take part in pair distances. Garments
# and fluids deform, so a single pose and a fixed mesh describe neither.
_POSED_OBJECT_CLASSES: FrozenSet[str] = frozenset(
    {"rigid", "dynamic", "articulation", "geometry"}
)

# Entity class of a pairwise distance between two posed bodies.
_PAIR_CLASS = "mesh_pair"

# Query-configuration binding for the sampled pair distances: the population
# takes ``distmax`` (metres), an optional ``robot_links`` list and an
# optional ``population: all``; instances name the pairs to query by entity
# label (operands ``a`` and ``b``).
_PAIR_BINDING = "robodojo.mesh_pair"

_AXES = ("0", "1", "2")
_QUAT_COMPONENTS = ("w", "x", "y", "z")
_WRENCH_AXES = ("0", "1", "2", "3", "4", "5")


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """
    Materialise one simulator array as numpy.

    RoboDojo mixes three array flavours in one scene: Isaac Lab's
    ``Articulation.data`` returns torch tensors on the simulation device,
    Isaac Sim core prims return numpy (or warp arrays when the prim's
    backend is warp), and the layout tables hold plain lists. ``torch`` is
    imported lazily because this module is imported by the registry at
    startup, where paying for a torch import on every ``specola``
    invocation would be wrong.
    """
    if value is None:
        return None

    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with Isaac Sim
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    # Warp arrays (``wp.array``) expose ``.numpy()``; so does anything that
    # already owns host memory. Anything else goes through np.asarray.
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return np.asarray(numpy_method())

    return np.asarray(value)


def _token(name: str) -> str:
    """
    Sanitise one simulator name into a property-key token.

    Mirrors ``MujocoAdapter._property_token`` and the ``isaaclab`` adapter:
    punctuation collapses to single underscores, case is dropped, and a
    leading digit is prefixed so the token is always a valid identifier
    fragment.
    """
    cleaned: List[str] = []
    for char in str(name):
        cleaned.append(char.lower() if char.isalnum() else "_")
    token = "".join(cleaned).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        return "unnamed"
    if token[0].isdigit():
        token = f"joint_{token}"
    return token


def _dof_kind(value: Any) -> Optional[str]:
    """Classify one PhysX DOF type as an angle or a displacement."""
    name = getattr(value, "name", None) or str(value)
    lowered = name.lower()
    if "rotation" in lowered or "revolute" in lowered:
        return "angle"
    if "translation" in lowered or "prismatic" in lowered:
        return "displacement"
    return None


def _joint_kinds(asset: Any) -> Optional[Tuple[Optional[str], ...]]:
    """
    Per-joint "angle"/"displacement" from the engine's own DOF types, or
    ``None`` when this asset does not publish them.

    The distinction decides the unit of ``joint_pos_*``/``joint_vel_*``, and
    a group must not mix units, so it is read from PhysX rather than guessed
    from a joint's name. Where it is unavailable the keys fall back to one
    joint-position group whose unit note says "rad or m" -- an honest
    "unknown", not a silent choice of one unit.
    """
    probes = (
        lambda: asset.root_physx_view.shared_metatype.dof_types,
        lambda: asset.dof_types,
        lambda: asset.get_dof_types(),
    )
    for probe in probes:
        try:
            raw = probe()
        except Exception:  # pragma: no cover - version/backend dependent
            continue
        if raw is None:
            continue
        kinds = tuple(_dof_kind(entry) for entry in raw)
        if kinds and any(kind is not None for kind in kinds):
            return kinds
    return None


def _side_tags(name: str) -> Tuple[str, ...]:
    """Structural side tag declared by a RobotManager arm name, if any."""
    lowered = str(name).lower()
    if lowered.startswith("left"):
        return ("left",)
    if lowered.startswith("right"):
        return ("right",)
    return ()


class RoboDojoAdapter(SimulatorAdapter):
    """
    Monitor one environment of a RoboDojo ``TaskEnv``.

    Args:
        env: A RoboDojo ``TaskEnv`` (or a task subclass of it, which is what
            ``task_registry.load_task_class`` returns) whose scene has been
            built -- that is, after the env's first ``reset()``. Passing one
            before the managers are initialised raises rather than reporting
            an empty scene.
        env_index: Which parallel environment to report. Must be in
            ``range(num_envs)``.
        dt: Seconds between successive ``get_observation()`` calls. Defaults
            to the simulator's own step interval; pass the real cadence when
            the caller samples less often than it steps (see ``dt``).
        robot_class: Entity class reported for each robot articulation.

    Scene objects are reported under RoboDojo's own type vocabulary --
    ``rigid``, ``articulation``, ``garment``, ``geometry``, ``fluid`` -- so
    an entity's class says what kind of thing the benchmark declared it to
    be rather than flattening every object into one class.
    """

    _OBJECT_CLASSES: FrozenSet[str] = _OBJECT_ENTITY_CLASSES

    # Render passes pumped before the first frame of an episode is
    # believed. Omniverse's pipeline warms up: the first frames after a
    # scene is built come back black or half-lit, and a dashboard fed one
    # of those shows what looks exactly like a dead camera. Later frames
    # need one fenced pass only.
    _WARMUP_RENDERS = 12

    # Step only on the main thread (#130). A physics step from a worker
    # thread writes USD; a notice listener (omni.anim.curve, via
    # Selection::getSelectedPrimPaths) then waits on an omni.usd rwlock
    # whose writer was measured to be the main thread, sitting in Textual's
    # event loop, and the step holds the GIL the main thread needs to go
    # on. ``specola run`` reads this to step on the main thread under the
    # Textual UI.
    step_on_main_thread = True

    def __init__(
        self,
        env: Any,
        env_index: int = 0,
        *,
        dt: Optional[float] = None,
        robot_class: str = "robot",
        distmax: Optional[float] = None,
        distance_device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._env = getattr(env, "unwrapped", env)
        self.robot_class = robot_class

        # Pair distances are opt-in: ``distmax`` (or ``ROBODOJO_DISTMAX``)
        # is the cutoff in metres above which a pair reports the cutoff
        # itself, clipped. Without it no pair entity exists and the mesh
        # walk is only made for the 3D panel.
        if distmax is None:
            import os

            raw = os.environ.get("ROBODOJO_DISTMAX")
            distmax = float(raw) if raw else None
        if distmax is not None and not (0.0 < float(distmax) < float("inf")):
            raise ValueError("distmax must be a finite positive number of metres")
        self.distmax = None if distmax is None else float(distmax)
        # (epoch, cutoff, wanted pairs, robot links, population: all) parsed
        # once per query configuration
        self._pair_config_cache: Optional[
            Tuple[Any, Optional[float], Optional[set], Optional[set], bool]
        ] = None
        self._distance_device = distance_device
        self._distance = None
        # Row plan for scene_geometry/scene_poses and the distance meshes,
        # built once per episode: RoboDojo respawns scene objects on reset.
        self._scene_plan: Optional[Dict[str, Any]] = None
        self._scene_plan_built = False

        for manager in ("robot_manager", "scene_manager"):
            if getattr(self._env, manager, None) is None:
                raise ValueError(
                    f"{type(env).__name__} exposes no {manager!r}; the "
                    "robodojo adapter expects a RoboDojo TaskEnv."
                )

        num_envs = int(getattr(self._env, "num_envs", 1))
        if not 0 <= env_index < num_envs:
            raise ValueError(
                f"env_index {env_index} is outside range(num_envs)="
                f"range({num_envs})"
            )
        self.env_index = int(env_index)

        if not getattr(self._env.robot_manager, "robot_key", None):
            raise ValueError(
                "RoboDojo RobotManager holds no articulation handles yet. "
                "Construct the adapter after the environment's first "
                "reset(), when the scene has been built."
            )

        self._dt_override = None if dt is None else float(dt)
        self._step = 0
        self._render_warmed = False

        # Identity key -> entity id, assigned on first sight and stable for
        # the adapter's lifetime, so an object that survives a reset keeps
        # its id. The class map is episode-scoped: reset() begins a new
        # identity namespace, and within one episode an id may not change
        # class.
        self._entity_id_of: Dict[str, int] = {}
        self._class_of: Dict[str, str] = {}
        self._next_id = 0

        # Step at which each entity id was first seen in this episode,
        # which is what track_age counts from. An object that appears
        # mid-episode -- RoboDojo respawns scene objects per layout -- is
        # one step old on its first observation, not as old as the episode.
        self._first_seen: Dict[int, int] = {}

        # Per-observation bookkeeping, refreshed by _observation().
        self._metadata: Dict[int, Dict[str, PropertyMeta]] = {}
        self._label_of: Dict[int, str] = {}
        # Robot key layouts by signature (_robot_payload); a layout is a
        # pure function of its signature, so entries never go stale.
        self._robot_layouts: Dict[Tuple[Any, ...], Any] = {}
        # Per-observation footprint cache and instance labels
        # (_native_object_payload).
        self._footprints: Dict[str, Any] = {}
        self._footprint_labels: Dict[str, str] = {}

    # -- adapter interface -------------------------------------------------

    @property
    def dt(self) -> float:
        """
        Seconds between observations.

        The default is the simulator's step interval: one
        ``TaskEnv.sim_step()`` advances physics by ``decimation`` substeps,
        which Isaac Lab reports as ``step_dt``. That is the *finest*
        cadence a caller can observe at, and it is right when the monitor
        is stepped from the env's own loop. A caller that samples once per
        policy action -- RoboDojo's eval loop steps the simulator several
        times per action -- observes more slowly than that and should pass
        ``dt`` explicitly, since every temporal bound in a specification is
        expressed in these seconds.
        """
        if self._dt_override is not None:
            return self._dt_override

        sim = getattr(self._env, "sim", None)
        step_dt = getattr(sim, "step_dt", None)
        if step_dt:
            return float(step_dt)

        physics_dt = getattr(sim, "physics_dt", None) or getattr(
            self._env, "dt", None
        )
        if not physics_dt:
            raise ValueError(
                "RoboDojo env publishes no step interval; construct the "
                "adapter with an explicit dt="
            )
        decimation = 1
        cfg = getattr(getattr(sim, "cfg", None), "decimation", None)
        if cfg:
            decimation = int(cfg)
        return float(physics_dt) * decimation

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def supported_classes(self) -> FrozenSet[str]:
        classes = frozenset({self.robot_class}) | self._OBJECT_CLASSES
        if self._active_distmax() is not None:
            classes = classes | {_PAIR_CLASS}
        return classes

    def _reset_impl(self) -> MonitorObservation:
        self._step = 0
        self._scene_plan = None
        self._scene_plan_built = False
        # A new episode is a new identity namespace: an instance name may
        # legitimately come back as a different kind of object under a
        # different layout seed. Ids are kept so a surviving object keeps
        # its identity in the traces, but every age restarts with the
        # episode -- a first-seen step from the previous one would make
        # track_age negative, which the base class rejects.
        self._class_of.clear()
        self._first_seen.clear()
        # RoboDojo rebuilds its cameras on reset, so the next frame is a
        # first frame again and has to be warmed like one.
        self._render_warmed = False
        return self._observation()

    def _get_observation_impl(self) -> MonitorObservation:
        self._step += 1
        return self._observation()

    def property_metadata(self, entity_id: int) -> Dict[str, PropertyMeta]:
        """
        Metadata for the keys of the most recent observation.

        Properties and metadata are built in one pass per entity (see
        ``_add``), so ``set(property_metadata(eid)) ==
        set(entity.properties)`` holds by construction rather than by two
        code paths agreeing.
        """
        return dict(self._metadata.get(entity_id, {}))

    @property
    def entity_labels(self) -> Dict[int, str]:
        """Task label (or instance name) behind each entity id."""
        return dict(self._label_of)

    # -- extraction helpers ------------------------------------------------

    def _add(
        self,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
        key: str,
        value: Any,
        meta: PropertyMeta,
        *,
        source: str,
    ) -> None:
        """
        Record one scalar and its metadata, rejecting key collisions.

        Two differently-named simulator objects can sanitise to one token
        (``finger.1`` and ``finger_1``). Dropping one silently would leave a
        specification binding to whichever happened to win, so the
        collision raises here, naming both sources.
        """
        previous = owners.get(key)
        if previous is not None and previous != source:
            raise ValueError(
                f"RoboDojo property key {key!r} is claimed by both "
                f"{previous!r} and {source!r}. Rename one of them in the "
                "simulator; specola will not silently drop either."
            )
        owners[key] = source
        properties[key] = float(value)
        metadata[key] = meta

    def _joint_payload(
        self,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
        names: Sequence[str],
        kinds: Optional[Tuple[Optional[str], ...]],
        positions: Optional[np.ndarray],
        velocities: Optional[np.ndarray],
        tags_of: Dict[str, Tuple[str, ...]],
    ) -> None:
        """Joint positions and velocities, grouped by engine DOF type."""
        for slot, name in enumerate(names):
            token = _token(name)
            tags = tags_of.get(name, ())
            kind = kinds[slot] if kinds is not None and slot < len(kinds) else None

            if kind == "angle":
                pos_meta = PropertyMeta(
                    group="joint angle", note="rad", unit="rad", tags=tags
                )
                vel_meta = PropertyMeta(
                    group="joint angular velocity",
                    note="rad/s",
                    unit="rad/s",
                    tags=tags,
                )
            elif kind == "displacement":
                pos_meta = PropertyMeta(
                    group="joint displacement", note="m", unit="m", tags=tags
                )
                vel_meta = PropertyMeta(
                    group="joint linear velocity",
                    note="m/s",
                    unit="m/s",
                    tags=tags,
                )
            else:
                # Unknown DOF type: one group, and a note that says the
                # unit is not established rather than asserting one.
                pos_meta = PropertyMeta(
                    group="joint position", note="rad or m", tags=tags
                )
                vel_meta = PropertyMeta(
                    group="joint velocity", note="rad/s or m/s", tags=tags
                )

            if positions is not None and slot < positions.shape[-1]:
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"joint_pos_{token}",
                    positions[slot],
                    pos_meta,
                    source=f"joint {name}",
                )
            if velocities is not None and slot < velocities.shape[-1]:
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"joint_vel_{token}",
                    velocities[slot],
                    vel_meta,
                    source=f"joint {name}",
                )

    def _root_pose_payload(
        self,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
        position: Optional[np.ndarray],
        orientation: Optional[np.ndarray],
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Root pose (and, where the prim has them, root velocities)."""
        if position is not None:
            for axis, label in enumerate(_AXES):
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"root_pos_w_{label}",
                    position[axis],
                    PropertyMeta(group="body position", note="m", unit="m"),
                    source="root pose",
                )
        if orientation is not None:
            for axis, label in enumerate(_QUAT_COMPONENTS):
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"root_quat_w_{label}",
                    orientation[axis],
                    PropertyMeta(
                        group="body orientation",
                        note=f"quat {label}",
                        component=label,
                    ),
                    source="root pose",
                )
        if linear_velocity is not None:
            for axis, label in enumerate(_AXES):
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"root_lin_vel_w_{label}",
                    linear_velocity[axis],
                    PropertyMeta(
                        group="body linear velocity", note="m/s", unit="m/s"
                    ),
                    source="root velocity",
                )
        if angular_velocity is not None:
            for axis, label in enumerate(_AXES):
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"root_ang_vel_w_{label}",
                    angular_velocity[axis],
                    PropertyMeta(
                        group="body angular velocity",
                        note="rad/s",
                        unit="rad/s",
                    ),
                    source="root velocity",
                )

    def _link_wrench(self, asset: Any) -> Optional[np.ndarray]:
        """
        Per-link 6-DOF joint reaction wrench as ``(bodies, 6)``, or None.

        Read from Isaac Lab's ``body_incoming_joint_wrench_b``, which is
        PhysX's ``get_link_incoming_joint_force`` -- the engine's own
        quantity, in the parent body frame. No task-side force copy is
        consulted: RoboDojo policies are evaluated on filtered observations,
        and a monitor reading those would be watching the policy's input
        rather than the simulation.
        """
        data = getattr(asset, "data", None)
        wrench = None
        if data is not None:
            try:
                wrench = _to_numpy(
                    getattr(data, "body_incoming_joint_wrench_b", None)
                )
            except Exception:  # pragma: no cover - backend dependent
                wrench = None

        if wrench is None:
            view = getattr(asset, "root_physx_view", None)
            getter = getattr(view, "get_link_incoming_joint_force", None)
            if getter is None:
                return None
            try:
                wrench = _to_numpy(getter())
            except Exception:  # pragma: no cover - backend dependent
                return None

        if wrench is None or wrench.ndim < 3:
            return None
        return np.asarray(wrench[self.env_index], dtype=float)

    def _robot_payload(
        self,
        asset: Any,
        robots: Sequence[Any],
    ) -> Tuple[
        Dict[str, float],
        Dict[str, PropertyMeta],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Every scalar reported for one robot articulation.

        ``robots`` are the RobotManager entries sharing this articulation --
        more than one when the robot config declares a coupled pair, which
        RoboDojo builds as a single Isaac Lab asset. Their declarations
        supply the structural tags (which joints are arm, which are gripper,
        which link is the end effector); the state itself comes from the
        articulation.
        """
        data = asset.data
        index = self.env_index

        joint_tags: Dict[str, Tuple[str, ...]] = {}
        body_tags: Dict[str, Tuple[str, ...]] = {}
        for robot in robots:
            side = _side_tags(getattr(robot, "arm_name", "") or "")
            for name in getattr(robot, "arm_joints_name", ()) or ():
                joint_tags[name] = ("arm",) + side
            for name in getattr(robot, "gripper_joints_name", ()) or ():
                joint_tags[name] = ("gripper",) + side
            ee_link = getattr(robot, "ee_link_name", None)
            if ee_link:
                body_tags[ee_link] = ("ee",) + side
            base_link = getattr(robot, "base_link", None)
            if base_link:
                body_tags[base_link] = ("base",) + side

        joint_pos = _to_numpy(getattr(data, "joint_pos", None))
        joint_vel = _to_numpy(getattr(data, "joint_vel", None))
        body_pos = _to_numpy(getattr(data, "body_pos_w", None))
        body_quat = _to_numpy(getattr(data, "body_quat_w", None))
        # Link-frame velocities, the frame body_pos_w is reported in (Isaac
        # Lab 0.54.3: body_pos_w is body_link_pos_w, while body_lin_vel_w
        # is the centre-of-mass velocity), shape (envs, bodies, 6).
        body_vel = _to_numpy(getattr(data, "body_link_vel_w", None))
        if body_vel is not None and (body_vel.ndim != 3 or body_vel.shape[2] != 6):
            body_vel = None
        # This environment's rows, in the order _robot_layout reads them.
        arrays = tuple(
            None if value is None else value[index]
            for value in (joint_pos, joint_vel, body_pos, body_quat, body_vel)
        ) + (self._link_wrench(asset),)

        # Which keys exist, their metadata and the key-collision check
        # depend only on names, DOF kinds, declared tags and array shapes,
        # so they are worked out once per distinct combination and an
        # observation only gathers values (rebuilding them was 0.4 ms of a
        # 1.8 ms stack_bowls observation, 2026-10-08).
        signature = (
            tuple(getattr(asset, "joint_names", ()) or ()),
            _joint_kinds(asset),
            tuple(getattr(asset, "body_names", ()) or ()),
            tuple(joint_tags.items()),
            tuple(body_tags.items()),
            tuple(None if value is None else value.shape for value in arrays),
        )
        layout = self._robot_layouts.get(signature)
        if layout is None:
            layout = self._robot_layouts[signature] = self._robot_layout(signature)
        slots, metadata = layout
        present = [value.reshape(-1) for value in arrays if value is not None]
        flat = np.concatenate(present).tolist() if present else []
        properties = {key: float(flat[offset]) for key, offset in slots}
        metadata = dict(metadata)
        # RoboDojo's end-effector poses are read per observation, outside the
        # cached layout. Their ee_* keys share no prefix with the layout's
        # joint_*/body_*/link_* keys, so a fresh owner table checks them.
        self._native_robot_payload(robots, properties, metadata, {})

        root_pos = _to_numpy(getattr(data, "root_pos_w", None))
        root_vel = _to_numpy(getattr(data, "root_lin_vel_w", None))
        position = (
            None
            if root_pos is None or root_pos.ndim < 2
            else np.asarray(root_pos[index][:3], dtype=float)
        )
        velocity = (
            None
            if root_vel is None or root_vel.ndim < 2
            else np.asarray(root_vel[index][:3], dtype=float)
        )
        return properties, metadata, position, velocity

    def _robot_layout(
        self, signature: Tuple[Any, ...]
    ) -> Tuple[Tuple[Tuple[str, int], ...], Dict[str, PropertyMeta]]:
        """
        The keys a robot payload with this signature emits, in order, each
        with the offset of its value in the payload's flattened arrays, and
        their metadata.

        It is the value pass run once on arrays of the payload's shapes
        whose elements are their own offsets, so key order, metadata, the
        shape guards and the collision check are those of the value pass by
        construction.
        """
        joint_names, kinds, body_names, joint_tag_items, body_tag_items, shapes = signature
        body_tags = dict(body_tag_items)
        offsets: List[Optional[np.ndarray]] = []
        base = 0
        for shape in shapes:
            if shape is None:
                offsets.append(None)
                continue
            size = int(np.prod(shape, dtype=np.int64))
            offsets.append(np.arange(base, base + size, dtype=np.float64).reshape(shape))
            base += size
        joint_pos, joint_vel, body_pos, body_quat, body_vel, wrench = offsets

        properties: Dict[str, float] = {}
        metadata: Dict[str, PropertyMeta] = {}
        owners: Dict[str, str] = {}
        self._joint_payload(
            properties,
            metadata,
            owners,
            joint_names,
            kinds,
            joint_pos,
            joint_vel,
            dict(joint_tag_items),
        )
        for slot, body in enumerate(body_names):
            token = _token(body)
            tags = body_tags.get(body, ())
            source = f"link {body}"
            if body_pos is not None and slot < body_pos.shape[0]:
                for axis, label in enumerate(_AXES):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"body_pos_w_{token}_{label}",
                        body_pos[slot][axis],
                        PropertyMeta(
                            group="body position",
                            note="m",
                            unit="m",
                            tags=tags,
                        ),
                        source=source,
                    )
            if body_quat is not None and slot < body_quat.shape[0]:
                # Isaac Lab reports body quaternions scalar-first
                # (w, x, y, z); so does Isaac Sim's core prim API, so both
                # halves of this adapter publish one convention.
                for axis, label in enumerate(_QUAT_COMPONENTS):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"body_quat_w_{token}_{label}",
                        body_quat[slot][axis],
                        PropertyMeta(
                            group="body orientation",
                            note=f"quat {label}",
                            component=label,
                            tags=tags,
                        ),
                        source=source,
                    )
            if body_vel is not None and slot < body_vel.shape[0]:
                for axis, label in enumerate(_AXES):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"body_lin_vel_w_{token}_{label}",
                        body_vel[slot][axis],
                        PropertyMeta(
                            group="body linear velocity",
                            note="m/s",
                            unit="m/s",
                            tags=tags,
                        ),
                        source=source,
                    )
                for axis, label in enumerate(_AXES, start=3):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"body_ang_vel_w_{token}_{label}",
                        body_vel[slot][axis],
                        PropertyMeta(
                            group="body angular velocity",
                            note="rad/s",
                            unit="rad/s",
                            tags=tags,
                        ),
                        source=source,
                    )
            if wrench is not None and slot < wrench.shape[0]:
                # Force and torque are split: a selector aggregates a whole
                # group, so N and N*m must not share one.
                for axis, label in enumerate(_WRENCH_AXES[:3]):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"link_wrench_{token}_{label}",
                        wrench[slot][axis],
                        PropertyMeta(
                            group="link forces", note="N", unit="N", tags=tags
                        ),
                        source=source,
                    )
                for axis, label in enumerate(_WRENCH_AXES[3:], start=3):
                    self._add(
                        properties,
                        metadata,
                        owners,
                        f"link_wrench_{token}_{label}",
                        wrench[slot][axis],
                        PropertyMeta(
                            group="link torques",
                            note="N·m",
                            unit="N·m",
                            tags=tags,
                        ),
                        source=source,
                    )
        return tuple((key, int(value)) for key, value in properties.items()), metadata

    def _object_payload(
        self,
        entity_class: str,
        obj: Any,
        name: Optional[str] = None,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, PropertyMeta],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Every scalar reported for one scene object.

        What is extracted follows what the object CAN answer, not what its
        class name suggests. RoboDojo's scene collections are not
        type-homogeneous -- ``_rigid_and_dynamic_objects`` holds PhysX
        rigid bodies next to ``DynamicObject``, a plain USD xform wrapper
        with a pose and no velocity at all -- so asking every member of
        that dictionary for ``get_linear_velocity()`` raises
        ``AttributeError`` on the conveyor tasks. An object that publishes
        no velocity is reported without velocity keys rather than with
        invented zeros, which would be indistinguishable from a genuinely
        stationary body.
        """
        properties: Dict[str, float] = {}
        metadata: Dict[str, PropertyMeta] = {}
        owners: Dict[str, str] = {}

        if entity_class == "fluid":
            return self._fluid_payload(obj, properties, metadata, owners)

        linear: Optional[np.ndarray] = None
        angular: Optional[np.ndarray] = None
        position, orientation = self._object_pose(obj)

        linear_getter = getattr(obj, "get_linear_velocity", None)
        if linear_getter is not None:
            linear = np.asarray(
                _to_numpy(linear_getter()), dtype=float
            ).reshape(-1)[:3]
        angular_getter = getattr(obj, "get_angular_velocity", None)
        if angular_getter is not None:
            angular = np.asarray(
                _to_numpy(angular_getter()), dtype=float
            ).reshape(-1)[:3]

        self._root_pose_payload(
            properties,
            metadata,
            owners,
            position,
            orientation,
            linear,
            angular,
        )
        if name is not None and entity_class in _POSED_OBJECT_CLASSES:
            self._bbox_payload(name, position, orientation, properties, metadata)
            self._native_object_payload(
                entity_class, name, properties, metadata, owners
            )

        joint_getter = getattr(obj, "get_joint_positions", None)
        joint_names = tuple(getattr(obj, "dof_names", ()) or ())
        if joint_getter is not None and joint_names:
            velocity_getter = getattr(obj, "get_joint_velocities", None)
            self._joint_payload(
                properties,
                metadata,
                owners,
                joint_names,
                _joint_kinds(obj),
                _to_numpy(joint_getter()),
                None if velocity_getter is None else _to_numpy(velocity_getter()),
                {},
            )

        return properties, metadata, position, linear

    @staticmethod
    def _object_pose(obj: Any) -> Tuple[np.ndarray, np.ndarray]:
        """World position and scalar-first quaternion of a scene object."""
        state_getter = getattr(obj, "get_state", None)
        if state_getter is not None:
            # Garment and geometry prims answer with the world-frame
            # root_pose [x, y, z, qw, qx, qy, qz] RoboDojo's own layout and
            # reward code reads for them.
            root_pose = np.asarray(
                _to_numpy(state_getter(is_relative=False)["root_pose"]),
                dtype=float,
            ).reshape(-1)
            return root_pose[:3], root_pose[3:7]
        # Isaac Sim core prims: get_world_pose() returns (position,
        # quaternion) with the quaternion scalar-first.
        pose = obj.get_world_pose()
        return (
            np.asarray(_to_numpy(pose[0]), dtype=float).reshape(-1)[:3],
            np.asarray(_to_numpy(pose[1]), dtype=float).reshape(-1)[:4],
        )

    def _bbox_payload(
        self,
        name: str,
        position: np.ndarray,
        orientation: np.ndarray,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
    ) -> None:
        """
        World-axis bounding box of one posed scene object, from RoboDojo's
        own oriented-bbox metadata carried by the live pose.

        RoboDojo's success predicates (``is_A_in_B``, ``is_A_on_B_bottom``,
        ``is_A_cover_B``) test an object's oriented bounding box against a
        container's, and the box is per asset: which bowl or vase a layout
        picked changes its size. A pose alone cannot restate those goals,
        so the eight object-frame vertices RoboDojo stores for each
        instance are rotated and translated by the current pose and
        reported as ``aabb_min_<axis>``/``aabb_max_<axis>``. An axis-aligned
        box over-admits a rotated footprint by its corners; that is the
        whole gap to RoboDojo's polygon test. Objects with no metadata
        (a test fixture, an asset without a box) report no box.
        """
        layout = getattr(getattr(self._env, "scene_manager", None), "layout_manager", None)
        getter = getattr(layout, "get_instance_metadata", None)
        if getter is None:
            return
        try:
            record = getter(self.env_index, inst_name=name)
            vertices = record["geometry"]["oriented_bbox"]["vertices"]
            local = np.asarray(vertices, dtype=float).reshape(-1, 3)
        except Exception:
            # No record, a record of another shape, or no box in it: the
            # object reports no box rather than failing the observation.
            return
        if len(local) == 0:
            return
        # Scalar-first, the convention _object_pose documents and the one
        # root_quat_w_<w,x,y,z> already publishes from the same source.
        w, x, y, z = (float(v) for v in np.asarray(orientation, dtype=float).reshape(-1)[:4])
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        world = local @ rotation.T + np.asarray(position, dtype=float).reshape(3)
        lo, hi = world.min(axis=0), world.max(axis=0)
        for axis in range(3):
            for key, value in ((f"aabb_min_{axis}", lo[axis]), (f"aabb_max_{axis}", hi[axis])):
                properties[key] = float(value)
                metadata[key] = PropertyMeta(
                    group="bounding box",
                    note="m; world-axis bounding box of RoboDojo's oriented bbox at the current pose",
                    unit="m",
                    tags=("bbox",),
                )

    # -- RoboDojo's own reward-state quantities -------------------------------

    def _reward_references(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        The reset references RoboDojo's reward code captured for this
        environment: ``(pre_state, robot_origin_endpose)``.

        ``Func_Parser.init_state`` records every rigid and articulated
        instance's pose (``pre_state[env][inst]["pose"]``, position then
        scalar-first quaternion) and every arm's end-effector pose
        (``robot_origin_endpose[env][arm]``) when an episode is reset.
        ``is_lift``, ``is_moved`` and ``all_robot_back_to_origin`` are
        defined against these, so a specification restating them needs the
        captured values themselves: a pose re-read later, or a constant,
        is a different quantity. These two tables are the only part of the
        reward manager this adapter reads -- they are poses, not verdicts;
        the check lists, rewards and success flags stay unread.
        """
        manager = getattr(self._env, "reward_manager", None)
        parser = getattr(manager, "func_parser", None) if manager is not None else None
        if parser is None:
            return {}, {}
        pre = getattr(parser, "pre_state", None)
        home = getattr(parser, "robot_origin_endpose", None)
        index = self.env_index
        return (
            dict(pre[index]) if pre is not None and index < len(pre) else {},
            dict(home[index]) if home is not None and index < len(home) else {},
        )

    def _pose_keys(
        self,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
        stem: str,
        pose: Any,
        group: str,
        note: str,
        source: str,
    ) -> None:
        """``<stem>_pos_<axis>`` and ``<stem>_quat_<w,x,y,z>`` from one
        pose ``[x, y, z, qw, qx, qy, qz]``; position only from a
        three-vector (``is_moved(update=True)`` stores one)."""
        values = _to_numpy(pose)
        if values is None:
            return
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 3:
            return
        for axis, label in enumerate(_AXES):
            self._add(
                properties, metadata, owners, f"{stem}_pos_{label}", values[axis],
                PropertyMeta(group=f"{group} position", note=f"m; {note}", unit="m"),
                source=source,
            )
        if values.size < 7:
            return
        for axis, label in enumerate(_QUAT_COMPONENTS, start=3):
            self._add(
                properties, metadata, owners, f"{stem}_quat_{label}", values[axis],
                PropertyMeta(
                    group=f"{group} orientation", note="quat", component=label,
                ),
                source=source,
            )

    def _footprint(self, name: str) -> Any:
        """
        ``(polygon, z_min, z_max)`` of one instance, exactly as RoboDojo's
        ``is_A_cover_B`` and ``is_A_on_B_bottom`` build it -- the instance's
        oriented-bbox vertices carried by ``LayoutManager.get_instance_pose``
        through ``utils.transformer.calc_polygon`` (XY convex hull, world
        z extent) -- or None when the instance has no pose or box, or the
        RoboDojo checkout is not importable. Cached per observation.
        """
        cache = self._footprints
        if name in cache:
            return cache[name]
        cache[name] = None
        try:
            from utils.transformer import calc_polygon  # RoboDojo checkout
        except ImportError:
            return None
        layout = getattr(self._env.scene_manager, "layout_manager", None)
        box_getter = getattr(layout, "get_instance_bbox_vertices", None)
        pose = self._native_pose(name)
        if box_getter is None or pose is None:
            return None
        vertices = box_getter(inst_name=name, env_idx=self.env_index)
        try:
            # Without oriented-bbox metadata RoboDojo falls back to the
            # object's get_bbox(), a (pos, ori, box) tuple its own
            # predicates cannot use either; no footprint then.
            points = np.asarray(vertices, dtype=float).reshape(-1, 3)
        except (TypeError, ValueError):
            return None
        cache[name] = calc_polygon(origin_pose=pose, origin_bbox_points=points)
        return cache[name]

    def _native_pose(self, name: str) -> Optional[np.ndarray]:
        """``[x, y, z, qw, qx, qy, qz]`` from ``LayoutManager.get_instance_pose``,
        or None. That accessor answers ``(None, None)`` for an unknown
        instance and a bare ``None`` for types it has no pose rule for
        (``dynamic``, the conveyor), so both are handled here."""
        layout = getattr(self._env.scene_manager, "layout_manager", None)
        getter = getattr(layout, "get_instance_pose", None)
        if getter is None:
            return None
        result = getter(self.env_index, inst_name=name)
        if not isinstance(result, tuple) or len(result) != 2:
            return None
        position, orientation = result
        if position is None or orientation is None:
            return None
        return np.concatenate([
            np.asarray(_to_numpy(position), dtype=float).reshape(-1)[:3],
            np.asarray(_to_numpy(orientation), dtype=float).reshape(-1)[:4],
        ])

    @staticmethod
    def _containment_margin(outer: Any, inner: Any) -> Optional[float]:
        """
        Signed margin, in metres, of polygon ``inner`` inside convex polygon
        ``outer``: the minimum over ``inner``'s vertices of the distance to
        ``outer``'s boundary, positive for a vertex ``outer`` covers (the
        boundary included), negative for one outside it.

        For convex ``outer`` and ``inner`` with an interior (RoboDojo's
        footprints are convex hulls of boxes) the margin is non-negative
        exactly when ``outer.contains(inner)``, the test ``is_A_cover_B``
        makes; its magnitude is the distance the footprint can move before
        the verdict flips, bounded by the nearest vertex.
        """
        from shapely.geometry import Point

        exterior = getattr(inner, "exterior", None)
        ring = getattr(outer, "exterior", None)
        if exterior is None or ring is None:
            return None
        margin = float("inf")
        for x, y in list(exterior.coords)[:-1]:
            point = Point(x, y)
            distance = float(ring.distance(point))
            margin = min(margin, distance if outer.covers(point) else -distance)
        return margin if np.isfinite(margin) else None

    def _native_object_payload(
        self,
        entity_class: str,
        name: str,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
    ) -> None:
        """
        The quantities RoboDojo's own success predicates read for one scene
        object, taken through RoboDojo's own accessors.

        ``native_pos_*``/``native_quat_*``: ``LayoutManager.get_instance_pose``,
        the pose every reward helper reads (the prim's pose relative to its
        environment, float32 in the engine), which differs from the world
        pose ``root_pos_w_*`` reports by the environment offset.
        ``reset_pos_*``/``reset_quat_*``: the reference pose the reward
        code holds for the instance (``_reward_references``), captured at
        reset; tasks that call ``update_object_state`` or
        ``is_moved(update=True)`` replace it mid-episode, and the key
        follows (the three tasks of the 2026-09-25 rerun do neither). ``model_id``: the
        asset's index in its category, the immutable identity
        ``get_label_cat_index`` orders by. ``footprint_zmin``/``_zmax`` and
        ``footprint_in_<label>``: the footprint RoboDojo's container
        predicates build (``_footprint``) and its signed containment margin
        inside the footprint of the object labelled ``<label>``
        (``_containment_margin``), for every other labelled object; that key
        is computed only where a specification asks for it.
        """
        layout = getattr(self._env.scene_manager, "layout_manager", None)
        pose = self._native_pose(name)
        if pose is None:
            return
        source = "RoboDojo reward state"
        self._pose_keys(
            properties, metadata, owners, "native", pose, "native pose",
            "LayoutManager.get_instance_pose, environment frame", source,
        )
        reset = self._reward_references()[0].get(name)
        if isinstance(reset, dict) and reset.get("pose") is not None:
            self._pose_keys(
                properties, metadata, owners, "reset", reset["pose"], "reset pose",
                "pose RoboDojo's reward code captured at reset", source,
            )
        record = None
        record_getter = getattr(layout, "get_instance_metadata", None)
        if record_getter is not None:
            record = record_getter(self.env_index, inst_name=name)
        if isinstance(record, dict) and record.get("model_id") is not None:
            self._add(
                properties, metadata, owners, "model_id", float(record["model_id"]),
                PropertyMeta(group="asset identity", note="index", unit="index"),
                source=source,
            )

        footprint = self._footprint(name)
        if footprint is None:
            return
        polygon, z_min, z_max = footprint
        for key, value in (("footprint_zmin", z_min), ("footprint_zmax", z_max)):
            self._add(
                properties, metadata, owners, key, float(value),
                PropertyMeta(
                    group="footprint height",
                    note="m; z extent of RoboDojo's calc_polygon footprint",
                    unit="m",
                ),
                source=source,
            )
        for other, label in sorted(self._footprint_labels.items()):
            if other == name:
                continue
            key = f"footprint_in_{_token(label)}"
            if not self._property_is_requested(entity_class, key):
                continue
            outer = self._footprint(other)
            if outer is None:
                continue
            margin = self._containment_margin(outer[0], polygon)
            if margin is None:
                continue
            self._add(
                properties, metadata, owners, key, margin,
                PropertyMeta(
                    group="footprint containment",
                    note="m; signed margin inside the named object's footprint",
                    unit="m",
                ),
                # Per container, so two instances sharing one label
                # collide loudly in _add instead of overwriting.
                source=f"footprint in {other}",
            )

    def _native_robot_payload(
        self,
        robots: Sequence[Any],
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
    ) -> None:
        """
        Each arm's end-effector pose as ``all_robot_back_to_origin`` reads
        it (``RobotManager.get_real_endpose``: the ``ee_link`` frame,
        environment-relative, scalar-first quaternion), and the pose the
        reward code captured at reset (``ee_home_*``). A coupled pair (two
        arms in one articulation) keys each arm by its name.
        """
        manager = self._env.robot_manager
        getter = getattr(manager, "get_real_endpose", None)
        if getter is None:
            return
        homes = self._reward_references()[1]
        for robot in robots:
            arm = str(getattr(robot, "arm_name", "") or "arm")
            stem = "ee" if len(robots) == 1 else f"ee_{_token(arm)}"
            source = f"end effector {arm}"
            pose = getter(robot, env_idx_list=[self.env_index]).get(self.env_index)
            self._pose_keys(
                properties, metadata, owners, stem, pose, "end-effector pose",
                "RobotManager.get_real_endpose, environment frame", source,
            )
            if homes.get(arm) is not None:
                self._pose_keys(
                    properties, metadata, owners, f"{stem}_home", homes[arm],
                    "end-effector home pose",
                    "end-effector pose RoboDojo's reward code captured at reset",
                    source,
                )

    def _fluid_payload(
        self,
        obj: Any,
        properties: Dict[str, float],
        metadata: Dict[str, PropertyMeta],
        owners: Dict[str, str],
    ) -> Tuple[
        Dict[str, float],
        Dict[str, PropertyMeta],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Particle count and particle centroid for one fluid.

        A fluid is a USD point instancer: its particles carry no identity
        the benchmark declares, and their number can change, so there is no
        honest per-particle property key. The centroid is reported as what
        it is -- an APPROXIMATION of where the fluid is, labelled as such in
        its unit note -- and it is deliberately not used as the entity's
        position, which stays ``None`` rather than implying the fluid has
        one.
        """
        positions, *_ = obj.get_particle_positions()
        particles = _to_numpy(positions)
        count = 0 if particles is None else int(np.size(particles) // 3)

        self._add(
            properties,
            metadata,
            owners,
            "particle_count",
            count,
            PropertyMeta(group="particle count", note="count", unit="count"),
            source="particle instancer",
        )

        if count:
            centroid = np.asarray(particles, dtype=float).reshape(-1, 3).mean(axis=0)
            for axis, label in enumerate(_AXES):
                self._add(
                    properties,
                    metadata,
                    owners,
                    f"particle_centroid_{label}",
                    centroid[axis],
                    PropertyMeta(
                        group="fluid particle centroid",
                        note="m (mean of particle positions; approximation)",
                        unit="m",
                    ),
                    source="particle instancer",
                )

        return properties, metadata, None, None

    # -- scene walk --------------------------------------------------------

    def _robot_groups(self) -> List[Tuple[str, Any, List[Any]]]:
        """
        One entry per robot articulation: ``(key, asset, robots)``.

        RobotManager keeps one ``robot_key`` entry per robot, but a coupled
        pair (one USD articulation carrying two arms) points both entries at
        the SAME Isaac Lab asset. Grouping by asset identity reports the
        engine's articulations rather than inventing a second entity that
        would republish every one of the first entity's joints.
        """
        manager = self._env.robot_manager
        groups: List[Tuple[str, Any, List[Any]]] = []
        index_of: Dict[int, int] = {}
        for robot, asset in zip(manager.robot_list, manager.robot_key):
            handle = id(asset)
            if handle in index_of:
                groups[index_of[handle]][2].append(robot)
                continue
            name = getattr(robot, "arm_name", None) or f"robot{len(groups)}"
            index_of[handle] = len(groups)
            groups.append((f"robot:{name}", asset, [robot]))
        return groups

    def _scene_objects(self) -> List[Tuple[str, str, str, Any]]:
        """
        This environment's scene objects as ``(key, class, name, obj)``.

        Re-read on every observation: RoboDojo respawns scene objects on
        every episode, and which objects exist depends on the layout seed.

        An object's class is the type its LAYOUT declares
        (``instance_type_by_env``), not the collection it happens to be
        stored in. The two differ: a ``Dynamic`` instance lives in the
        rigid collection, and calling it rigid would promise a velocity it
        does not have. A declared type outside this adapter's vocabulary
        falls back to the collection name, so an unfamiliar layout type
        cannot smuggle an undeclared class past ``supported_classes``.
        """
        manager = self._env.scene_manager
        declared = self._declared_types()
        found: List[Tuple[str, str, str, Any]] = []
        for collection in _OBJECT_COLLECTIONS:
            objects = manager.get_objects(
                env_ids=[self.env_index], object_type=collection
            )
            prefix = f"env{self.env_index}_{collection}_"
            for composite, obj in objects.items():
                if obj is None:
                    continue
                name = (
                    composite[len(prefix):]
                    if composite.startswith(prefix)
                    else composite
                )
                entity_class = declared.get(name, collection)
                if entity_class not in _OBJECT_ENTITY_CLASSES:
                    entity_class = collection
                found.append((f"object:{name}", entity_class, name, obj))
        return found

    def _declared_types(self) -> Dict[str, str]:
        """Layout-declared type per instance name, empty when unavailable."""
        manager = getattr(self._env.scene_manager, "layout_manager", None)
        types = getattr(manager, "instance_type_by_env", None)
        if not types or self.env_index >= len(types):
            return {}
        return dict(types[self.env_index] or {})

    def _labels(self) -> Dict[str, str]:
        """
        Task label per instance name, from the layout records.

        The task YAML's ``select_mode.label`` is what ``run_reward`` and
        ``get_score`` reference (``bowl0``, ``bowl1``), so it is the name a
        spec author has already seen. Unlabelled scene dressing -- clutter,
        the table, the room -- has no label and keeps its instance name.
        """
        manager = getattr(self._env.scene_manager, "layout_manager", None)
        if manager is None:
            return {}
        labels: Dict[str, str] = {}
        for record_type in _LAYOUT_RECORD_TYPES:
            try:
                records = manager.get_layout_records(self.env_index, record_type)
            except (KeyError, AttributeError, IndexError):
                continue
            for record in records or ():
                label = record.get("label")
                name = record.get("inst_name")
                if label and name:
                    labels[name] = label
        return labels

    def _entity_id(self, key: str, entity_class: str) -> int:
        """Stable id for one identity key, guarding mid-episode class change."""
        previous = self._class_of.get(key)
        if previous is not None and previous != entity_class:
            raise ValueError(
                f"RoboDojo entity {key!r} is reported as both {previous!r} "
                f"and {entity_class!r} in one episode. An instance name is "
                "an identity: either it moved between scene collections, or "
                "two collections hold one name and one of them has to be "
                "renamed."
            )
        self._class_of[key] = entity_class

        entity_id = self._entity_id_of.get(key)
        if entity_id is None:
            entity_id = self._next_id
            self._next_id += 1
            self._entity_id_of[key] = entity_id
        self._first_seen.setdefault(entity_id, self._step)
        return entity_id

    def _observation(self) -> MonitorObservation:
        entities: Dict[int, EntityState] = {}
        metadata: Dict[int, Dict[str, PropertyMeta]] = {}
        labels: Dict[int, str] = {}
        label_of_instance = self._labels()
        # Footprints are shared by every containment key of one observation
        # (see _native_object_payload), so they are computed once each.
        self._footprint_labels = label_of_instance
        self._footprints = {}

        for key, asset, robots in self._robot_groups():
            entity_id = self._entity_id(key, self.robot_class)
            props, meta, position, velocity = self._robot_payload(asset, robots)
            metadata[entity_id] = meta
            labels[entity_id] = "+".join(
                str(getattr(robot, "arm_name", "")) or "robot"
                for robot in robots
            )
            entities[entity_id] = EntityState(
                entity_id=entity_id,
                entity_class=self.robot_class,
                position=position,
                velocity=velocity,
                track_age=self._step - self._first_seen[entity_id],
                track_state=TrackState.CONFIRMED,
                properties=props,
            )

        for key, entity_class, name, obj in self._scene_objects():
            entity_id = self._entity_id(key, entity_class)
            props, meta, position, velocity = self._object_payload(
                entity_class, obj, name
            )
            metadata[entity_id] = meta
            labels[entity_id] = label_of_instance.get(name, name)
            entities[entity_id] = EntityState(
                entity_id=entity_id,
                entity_class=entity_class,
                position=position,
                velocity=velocity,
                track_age=self._step - self._first_seen[entity_id],
                track_state=TrackState.CONFIRMED,
                properties=props,
            )

        if self._active_distmax() is not None:
            self._pair_entities(entities, metadata, labels)

        self._metadata = metadata
        self._label_of = labels

        return MonitorObservation(
            timestep=self._step,
            time=self._step * self.dt,
            perceived_entities=entities,
        )

    # -- 3D pose panel and pair distances ----------------------------------

    def _stage(self) -> Any:
        stage = getattr(self._env, "stage", None)
        if stage is None:
            stage = getattr(getattr(self._env, "scene", None), "stage", None)
        if stage is None:
            try:
                import omni.usd  # type: ignore

                stage = omni.usd.get_context().get_stage()
            except Exception:  # noqa: BLE001 - no Kit in this process
                stage = None
        return stage

    def _ensure_scene_plan(self) -> Optional[Dict[str, Any]]:
        """
        Build, once per episode, the body-row plan shared by
        ``scene_geometry``, ``scene_poses`` and the pair distances.

        Row 0 is the world anchor; then every robot's links in
        ``_robot_groups`` order; then one row per posed scene object; then
        static fixtures (``/World/background`` and any child of this
        environment's prim that no robot or object owns). Deciding
        viability once per episode keeps the hooks from alternating
        between ``None`` and arrays inside one.
        """
        if self._scene_plan_built:
            return self._scene_plan
        self._scene_plan_built = True
        self._scene_plan = self._build_scene_plan()
        return self._scene_plan

    def _build_scene_plan(self) -> Optional[Dict[str, Any]]:
        stage = self._stage()
        if stage is None:
            return None
        try:
            if not stage_is_metres_z_up(stage):
                return None
        except ImportError:  # pragma: no cover - pxr ships with Isaac Sim
            return None

        bodies: List[Dict[str, Any]] = [{"name": "world", "parent": 0}]
        # (row, prim path, frame is the prim itself) for posed bodies
        posed: List[Tuple[int, Optional[str]]] = []
        robots: List[Tuple[str, Any, range]] = []
        objects: List[Tuple[str, str, int]] = []
        for key, asset, _ in self._robot_groups():
            names = tuple(getattr(asset, "body_names", ()) or ())
            positions = _to_numpy(getattr(asset.data, "body_pos_w", None))
            if not names or positions is None or positions.ndim != 3:
                continue
            paths = link_prim_paths(stage, asset, names, self.env_index)
            first = len(bodies)
            for name, path in zip(names, paths):
                bodies.append({"name": f"{key}/{name}", "parent": 0})
                posed.append((len(bodies) - 1, path))
            robots.append((key, asset, range(first, len(bodies))))
        for key, entity_class, name, obj in self._scene_objects():
            if entity_class not in _POSED_OBJECT_CLASSES:
                continue
            path = None
            for attr in ("_prim_path", "prim_path", "usd_prim_path"):
                value = getattr(obj, attr, None)
                if value:
                    path = str(value)
                    break
            bodies.append({"name": name, "parent": 0})
            posed.append((len(bodies) - 1, path))
            objects.append((key, name, len(bodies) - 1))
        if len(bodies) == 1:
            return None

        owned = {path for _, path in posed if path is not None}
        statics: List[Tuple[int, str]] = []
        env_path = f"/World/envs/env_{self.env_index}"
        env_prim = stage.GetPrimAtPath(env_path)
        candidates: List[Any] = []
        if env_prim and env_prim.IsValid():
            candidates.extend(env_prim.GetChildren())
        background = stage.GetPrimAtPath("/World/background")
        if background and background.IsValid():
            candidates.append(background)
        for child in candidates:
            prefix = str(child.GetPath())
            if any(path == prefix or path.startswith(prefix + "/") for path in owned):
                continue
            if any(prefix.startswith(path + "/") for path in owned):
                continue
            bodies.append({"name": child.GetName(), "parent": 0})
            statics.append((len(bodies) - 1, prefix))

        collector = MeshCollector(owned)
        for row, path in posed:
            prim = stage.GetPrimAtPath(path) if path else None
            if prim and prim.IsValid():
                collector.collect(prim, row, prim)
            else:
                collector.skipped["unresolved_body"] += 1
        for row, path in statics:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                collector.collect(prim, row, None)

        plan = {
            "bodies": bodies,
            "robots": robots,
            "objects": objects,
            "statics": statics,
            # Packed by the first scene_geometry() call: only the dashboard
            # reads it, and packing was 64 ms of a 214 ms build on
            # stack_bowls (2026-10-08).
            "geometry": None,
            "collected": collector.collected,
            "skipped": collector.skipped,
            "pairs": (),
        }
        if self._active_distmax() is not None:
            self._register_distance_meshes(plan, collector.collected)
        return plan

    def _pair_configuration(self) -> Tuple[Optional[float], Optional[set], Optional[set], bool]:
        """
        ``(cutoff, wanted pairs, robot links, population: all)`` for the
        current query configuration and fallback cutoff, parsed and
        validated once per (configuration epoch, ``self.distmax``): the
        lookups are asked several times per reset and per observation, the
        configuration is immutable between epochs, and the fallback is a
        public field.
        """
        key = (self.query_configuration_epoch, self.distmax)
        cached = self._pair_config_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3], cached[4]
        wanted = self._parse_configured_pairs()
        cutoff, robot_links, population_all = self._parse_population()
        self._pair_config_cache = (key, cutoff, wanted, robot_links, population_all)
        return cutoff, wanted, robot_links, population_all

    def _active_distmax(self) -> Optional[float]:
        """
        The pair-distance cutoff in force: the query configuration's
        ``robodojo.mesh_pair`` population wins over the constructor or
        ``ROBODOJO_DISTMAX``, so a run declares its clearance horizon next to
        its specifications, the way the MuJoCo family does with --queries.
        """
        return self._pair_configuration()[0]

    def _configured_pairs(self) -> Optional[set]:
        """
        Labels of the pairs the query configuration asks for, as unordered
        label pairs, or None when it names none (every pair is queried).
        """
        return self._pair_configuration()[1]

    def _parse_population(self) -> Tuple[Optional[float], Optional[set], bool]:
        """``(cutoff, robot links or None, population: all)`` from the
        population, validated."""
        arguments = self.query_configuration.population_arguments(_PAIR_BINDING)
        instances = self.query_configuration.instances(_PAIR_BINDING)
        if arguments is None and not instances:
            return self.distmax, None, False
        # The binding is named, so its whole shape is checked here, before
        # anything decides whether distances are on: a misspelt key must
        # not silently turn the request into no query.
        arguments = arguments or {}
        unknown_args = sorted(set(arguments) - {"distmax", "robot_links", "population"})
        if unknown_args:
            raise ValueError(
                f"{_PAIR_BINDING} population takes distmax, robot_links and population, "
                f"not {unknown_args}"
            )
        # The document schema admits only ``population: all``.
        population_all = arguments.get("population") == "all"
        robot_links = arguments.get("robot_links")
        if robot_links is not None and (
            not isinstance(robot_links, list) or not all(isinstance(n, str) for n in robot_links)
        ):
            raise ValueError(
                f"{_PAIR_BINDING} population argument robot_links must be a list of link names"
            )
        links = None if robot_links is None else set(robot_links)
        if arguments.get("distmax") is not None:
            # A JSON number only: a bool or a numeric string would coerce.
            value = arguments["distmax"]
            if type(value) not in (int, float) or not (0.0 < float(value) < float("inf")):
                raise ValueError(
                    f"{_PAIR_BINDING} distmax must be a finite positive number of metres, "
                    f"got {value!r}"
                )
            return float(value), links, population_all
        if self.distmax is None:
            raise ValueError(
                f"{_PAIR_BINDING} is configured but no cutoff is set: give the population a "
                "distmax (or pass distmax= / ROBODOJO_DISTMAX)"
            )
        return self.distmax, links, population_all

    def _parse_configured_pairs(self) -> Optional[set]:
        """
        Labels of the pairs the query configuration asks for, as unordered
        label pairs, or None when it names none (every pair is queried).

        An instance of binding ``robodojo.mesh_pair`` carries operands ``a``
        and ``b``, each an entity named by its label (``left_arm``,
        ``bowl0``). Named pairs are the only ones formed, and every one is
        read on every observation whatever the specifications bind; without
        names, a monitor session reads only the pairs its specifications
        can bind (``_pair_in_scope``).
        """
        instances = self.query_configuration.instances(_PAIR_BINDING)
        if not instances:
            return None
        wanted = set()
        for instance in instances:
            operands = instance.get("operands", {})
            if set(operands) != {"a", "b"}:
                raise ValueError(
                    f"{_PAIR_BINDING} instance {instance.get('id')!r} needs operands a and b, "
                    f"got {sorted(operands)}"
                )
            if instance.get("arguments"):
                raise ValueError(
                    f"{_PAIR_BINDING} instance {instance.get('id')!r} takes no arguments, "
                    f"got {sorted(instance['arguments'])}"
                )
            a, b = operands["a"], operands["b"]
            names = []
            for operand in (a, b):
                if operand.get("kind") != "entity":
                    raise ValueError(
                        f"{_PAIR_BINDING} instance {instance.get('id')!r}: operands must be "
                        f"of kind entity, got {operand.get('kind')!r}"
                    )
                if "name" not in operand:
                    raise ValueError(
                        f"robodojo.mesh_pair instance {instance.get('id')!r}: operands are "
                        "named by entity label (name), not by id"
                    )
                names.append(str(operand["name"]))
            wanted.add(frozenset(names))
        return wanted

    def _register_distance_meshes(self, plan: Dict[str, Any], collected: Any) -> None:
        from ._mesh_distance import SampledMeshDistance

        # Shape was checked when the configuration was parsed; here the
        # link names are resolved against the robots the plan actually holds.
        robot_links = self._pair_configuration()[2]
        if robot_links is not None:
            known = {
                name for _, asset, _ in plan["robots"]
                for name in (getattr(asset, "body_names", ()) or ())
            }
            unknown = sorted(robot_links - known)
            if unknown:
                raise ValueError(
                    f"{_PAIR_BINDING} population argument robot_links names links no robot "
                    f"has: {unknown}; known: {sorted(known)}"
                )

        query = SampledMeshDistance(device=self._distance_device)
        rows_of: Dict[str, List[int]] = {}
        # Bounding radius of each body's merged mesh about its own frame, so
        # a pair whose centres are further apart than the two radii plus the
        # cutoff is never queried: it reports the cutoff, clipped, exactly as
        # a queried far pair would.
        radius: Dict[int, float] = {}

        def register(key: str, row: int) -> None:
            merged = merge_meshes(collected, row)
            if merged is None:
                return
            query.add(str(row), merged[0], merged[1])
            rows_of.setdefault(key, []).append(row)
            vertices = np.asarray(merged[0], dtype=float).reshape(-1, 3)
            radius[row] = float(np.linalg.norm(vertices, axis=1).max()) if len(vertices) else 0.0

        for key, asset, rows in plan["robots"]:
            names = tuple(getattr(asset, "body_names", ()) or ())
            for name, row in zip(names, rows):
                if robot_links is not None and name not in robot_links:
                    continue
                register(key, row)
        for key, _, row in plan["objects"]:
            register(key, row)

        keys = [key for key, _, _ in plan["robots"]] + [key for key, _, _ in plan["objects"]]
        keys = [key for key in keys if key in rows_of]
        wanted = self._configured_pairs()
        name_of = {key: name for key, name, _ in plan["objects"]}
        for key, _, _ in plan["robots"]:
            name_of[key] = key[len("robot:"):]
        label_of_instance = self._labels()
        # Every unordered pair, robot-robot included: on a bimanual rig the
        # arm-arm clearance is the one distance a safety spec asks for
        # first. Its cost is links-squared per step, which is why the query
        # configuration can name the pairs and links a run needs.
        pairs = []
        matched = set()
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if wanted is not None:
                    labels = frozenset((
                        label_of_instance.get(name_of[a], name_of[a]),
                        label_of_instance.get(name_of[b], name_of[b]),
                    ))
                    if labels not in wanted:
                        continue
                    matched.add(labels)
                pairs.append((a, b))
        if wanted is not None and wanted - matched:
            # A named pair that produced nothing is a misspelt label or an
            # entity without a mesh; silently computing nothing would let a
            # spec bind to no pair and never fail.
            known = sorted(str(label_of_instance.get(name_of[k], name_of[k])) for k in keys)
            missing = sorted(sorted(pair) for pair in wanted - matched)
            raise ValueError(
                f"{_PAIR_BINDING} instances name pairs with no mesh on both sides: "
                f"{missing}; entities with a mesh: {known}"
            )
        plan["pairs"] = tuple(pairs)
        plan["rows_of"] = rows_of
        plan["radius"] = radius
        self._distance = query

    def _body_poses(self, plan: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Fresh ``(xpos, xquat)`` for every row; quaternions ``(w, x, y, z)``."""
        nbody = len(plan["bodies"])
        xpos = np.zeros((nbody, 3), dtype=np.float32)
        xquat = np.zeros((nbody, 4), dtype=np.float32)
        xquat[:, 0] = 1.0
        index = self.env_index
        for key, asset, rows in plan["robots"]:
            # The fields the robot payload publishes, so the panel and the
            # pair distances pose each link exactly as body_pos_w_* and
            # body_quat_w_* report it. (Isaac Lab 0.54.3's body_link_pose_w
            # is the same buffer, quaternion already (w, x, y, z); reading it
            # with Isaac Lab 6.x's (x, y, z, w) reorder turned every link.)
            body_pos = _to_numpy(getattr(asset.data, "body_pos_w", None))
            body_quat = _to_numpy(getattr(asset.data, "body_quat_w", None))
            if body_pos is None or body_quat is None:
                raise RuntimeError(f"RoboDojo robot {key!r} lost body poses mid-episode")
            xpos[rows] = body_pos[index]
            xquat[rows] = body_quat[index]
        objects = {key: obj for key, _, _, obj in self._scene_objects()}
        for key, _, row in plan["objects"]:
            obj = objects.get(key)
            if obj is None:
                raise RuntimeError(f"RoboDojo object {key!r} vanished mid-episode")
            position, orientation = self._object_pose(obj)
            xpos[row] = position
            xquat[row] = orientation
        return xpos, xquat

    def _pair_in_scope(self, key: str, tags: Tuple[str, ...]) -> bool:
        """
        Whether this observation reads the formed pair ``key`` (issue #136).

        Under a monitor session's entity scope, an observation after reset
        reads only the pairs a compiled specification slot can bind, with
        the selector semantics of ``MonitorEngine.extraction_entity_scope``
        that ``MujocoAdapter._geom_pairs_in_scope`` applies to
        ``geom_pair``: the slot's pin is absent or names the pair, and the
        pair's tags hold every label of its ``@`` filters. Both keys of a
        pair carry the pair's tags, so a filter on either is a test of
        them. A pin can only name a pair reset formed, since a session
        compiles against the reset catalog, where every formed pair has its
        id.

        Every formed pair is read during reset (discovery and compilation
        need them all), without a scope (discover, check, library use), with
        ``population: all``, and when the query configuration names pairs:
        named pairs are the population the user asked for, and every one is
        read whatever the specifications bind, so the union of the names and
        the scope is the names (the maintainer's default, 2026-10-07).
        """
        scope = self._extraction_entity_scope
        if (scope is None or not self._episode_started
                or self._configured_pairs() is not None or self._pair_configuration()[3]):
            return True
        entity_id = self._entity_id_of.get(key)
        return any(
            (pinned is None or pinned == entity_id)
            and all(label in tags for _component, labels in filters for label in labels)
            for pinned, filters in scope.get(_PAIR_CLASS, ())
        )

    def _pair_entities(
        self,
        entities: Dict[int, EntityState],
        metadata: Dict[int, Dict[str, PropertyMeta]],
        labels: Dict[int, str],
    ) -> None:
        plan = self._ensure_scene_plan()
        if plan is None or not plan["pairs"]:
            return
        query = self._distance
        active = self._active_distmax()
        if active is None or query is None:
            return
        name_of = {key: name for key, name, _ in plan["objects"]}
        for key, _, _ in plan["robots"]:
            name_of[key] = key[len("robot:"):]
        label_of_instance = self._labels()
        selected = []
        for a, b in plan["pairs"]:
            label_a = label_of_instance.get(name_of[a], name_of[a])
            label_b = label_of_instance.get(name_of[b], name_of[b])
            if a.startswith("robot:") and b.startswith("robot:"):
                kind = "robot_pair"
            elif a.startswith("robot:"):
                kind = "robot"
            else:
                kind = "environment"
            tags = (f"a:{label_a}", f"b:{label_b}", kind, "method:vertex_sample")
            if self._pair_in_scope(f"pair:{a}|{b}", tags):
                selected.append((a, b, label_a, label_b, tags))
        if not selected:
            return
        xpos, xquat = self._body_poses(plan)
        distmax = float(active)
        rows_of = plan["rows_of"]
        # Meshes are posed on demand: a row takes part in the prefilter
        # from its centre alone, and is transformed and refitted only when
        # some pair it belongs to survives.
        updated: set = set()
        eps = 8.0 * float(np.finfo(np.float32).eps)
        centres64 = xpos.astype(np.float64)
        radius = plan.get("radius", {})
        for a, b, label_a, label_b, tags in selected:
            best = distmax
            clipped = True
            for row_a in rows_of[a]:
                for row_b in rows_of[b]:
                    # Bounding-sphere prefilter: no query when the meshes
                    # cannot be within the cutoff; the pair then reports
                    # the cutoff, clipped, as a queried far pair would.
                    # The query holds world vertices in float32, so at
                    # RoboDojo's environment offsets (tens of metres) a
                    # vertex can round a few micrometres inward; the bound
                    # grows by a multiple of that rounding.
                    reach = radius.get(row_a, 0.0) + radius.get(row_b, 0.0)
                    pa, pb = centres64[row_a], centres64[row_b]
                    margin = eps * (float(np.abs(pa).max()) + float(np.abs(pb).max()) + reach)
                    if float(np.linalg.norm(pa - pb)) > reach + distmax + margin:
                        continue
                    for row in (row_a, row_b):
                        if row not in updated:
                            query.update(str(row), xpos[row], xquat[row])
                            updated.add(row)
                    value, hit_cap = query.distance(str(row_a), str(row_b), distmax)
                    if value < best:
                        best = value
                    clipped = clipped and hit_cap
            entity_id = self._entity_id(f"pair:{a}|{b}", _PAIR_CLASS)
            centre = 0.5 * (xpos[rows_of[a][0]] + xpos[rows_of[b][0]])
            props = {
                "mesh_distance_sampled": float(best),
                "mesh_distance_clipped": 1.0 if clipped else 0.0,
            }
            metadata[entity_id] = {
                "mesh_distance_sampled": PropertyMeta(
                    group="mesh pair distance",
                    note="m; min over sampled vertices of either mesh to the other's surface, capped at distmax; zero inside",
                    unit="m",
                    tags=tags,
                ),
                "mesh_distance_clipped": PropertyMeta(
                    group="mesh pair distance",
                    note="bool; 1 when no sampled vertex lies within distmax",
                    tags=tags,
                ),
            }
            labels[entity_id] = f"{label_a}|{label_b}"
            entities[entity_id] = EntityState(
                entity_id=entity_id,
                entity_class=_PAIR_CLASS,
                position=np.asarray(centre, dtype=float),
                velocity=None,
                track_age=self._step - self._first_seen[entity_id],
                track_state=TrackState.CONFIRMED,
                properties=props,
            )

    def scene_geometry(self) -> Optional[dict]:
        """
        Export this environment's USD meshes for the 3D panel.

        Robot links and posed scene objects get one row each with their
        meshes baked into the body frame; static fixtures are baked in
        world coordinates against identity rows. Collision hulls (purpose
        "guide") land in hidden group 0, visual meshes in visible group 1.
        Garments and fluids have no row: they deform.
        """
        plan = self._ensure_scene_plan()
        if plan is None:
            return None
        if plan["geometry"] is None:
            plan["geometry"] = pack_scene(plan["bodies"], plan["collected"], plan["skipped"])
            del plan["collected"]
        return plan["geometry"]

    def scene_poses(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Copy this step's body poses for the 3D panel, rows as in the plan."""
        plan = self._ensure_scene_plan()
        if plan is None:
            return None
        xpos, xquat = self._body_poses(plan)
        return xpos, xquat

    # -- drive hook --------------------------------------------------------

    def make_advancer(self, mode: str, demo: Optional[str], seed: int):
        """
        Drive one episode for ``specola run``.

        ``hold`` advances physics with the robots holding the position
        targets they already have -- RoboDojo's own reset settles the scene
        this way -- so the scene falls, slides and settles under no new
        command. It is the drive mode a scene-level specification (nothing
        leaves the table, the stack stays upright) can be exercised with
        without a policy.

        ``random`` and ``replay`` are refused rather than approximated.
        RoboDojo actions are per-robot ``MetaControl`` dictionaries in
        joint or end-effector space, so a uniform sample over an action box
        does not exist to draw; and its recordings are observation archives
        of an eval run, not simulator states a monitor could replay into
        this scene. Both are reported as unsupported so a run cannot
        silently monitor something other than what was asked for.

        Nor is there a Specola state tier: ``state_signature()`` stays None,
        so ``--mode replay`` of a state recording is refused before this
        method runs. The Isaac Lab scene's ``get_state`` holds the two robot
        articulations only; the task objects are Isaac Sim core prims that
        RoboDojo's ``SceneManager`` holds outside that scene. On
        ``stack_bowls`` (Isaac Lab 0.54.3, 2026-10-14), restoring ten
        ``hold`` steps through ``reset_to`` left every bowl key at its value
        after the last live step: bowl positions differed from the live ones
        by up to 8.9e-6 m, quaternion components by up to 1.8e-4 and angular
        velocities by up to 3.3e-3 rad/s, beyond the 1e-6 m and 1e-5
        tolerances fixed beforehand, and a bowl lifted by 5 cm stayed lifted
        through a restore. With the arms driven, joint positions and
        velocities came back bit for bit, while link positions differed by
        up to 4.2e-5 m and link angular velocities by up to 0.093 rad/s
        (``scripts/robodojo_state_probe.py``,
        ``docs/evidence/state-replay-robodojo-2026-10-14.md``).
        """
        del seed
        if demo is not None:
            raise SystemExit(
                "The robodojo adapter has no --demo replay: RoboDojo records "
                "observations, not simulator states."
            )
        if mode != "hold":
            raise SystemExit(
                f"--mode {mode} is not supported by the robodojo adapter; "
                "use --mode hold to advance physics under no new command, or "
                "monitor a policy run by wrapping the env from RoboDojo's "
                "own eval loop."
            )

        def advance(step: int) -> bool:
            # Physics only: no control is applied, and rendering is left to
            # the frame capture, which fences it for itself.
            self._env.sim_step(render=False)
            return True

        return advance

    # -- visualization -----------------------------------------------------

    def _cameras(self) -> List[Tuple[str, int]]:
        """
        This environment's RGB cameras as ``(name, camera index)``.

        RoboDojo renders every environment's cameras in one tiled pass and
        indexes the output by environment, so a frame here belongs to the
        monitored scene rather than to a grid of them -- the attribution
        problem that makes the ``isaaclab`` adapter report no camera under
        vectorization does not arise.

        A camera whose annotators do not include ``rgb`` is left out: it
        renders something (depth, a segmentation mask) that
        ``capture_frame`` cannot return, and advertising it would promise
        frames that never arrive.
        """
        cameras = getattr(self._env, "camera_manager", None)
        capture = getattr(self._env, "capture_manager", None)
        if cameras is None or capture is None:
            return []

        names_per_env = getattr(cameras, "camera_names", None) or []
        if self.env_index >= len(names_per_env):
            return []
        names = names_per_env[self.env_index]
        annotators = getattr(capture, "annotator_type", None) or []

        found: List[Tuple[str, int]] = []
        for index, name in enumerate(names):
            if index < len(annotators) and "rgb" not in (annotators[index] or ()):
                continue
            found.append((str(name), index))
        return found

    def camera_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self._cameras())

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        One HxWx3 uint8 RGB frame from this environment's camera.

        ``width``/``height`` are ignored: RoboDojo's cameras render at the
        resolution their config declares, and an adapter may not change what
        the simulator renders for the benchmark.

        The frame is fenced before it is read. Isaac Sim's annotators hand
        back whatever the render graph last wrote, which without a fence can
        be several frames stale -- a monitor video that lags the physics it
        is annotating. RoboDojo ships the fence
        (``render_sync.wait_for_latest_cameras``), which renders until the
        SDG dispatcher has delivered a new frame; ``env.render()`` pumps the
        renderer without advancing physics, so fencing observes the
        simulator rather than driving it.

        The first frame of an episode is retried until it has content.
        Omniverse renders a just-built scene black while it compiles the
        scene's materials, and a black frame is indistinguishable from a
        dead camera downstream -- but a fixed warm-up count is the wrong
        cure, because on a dressed RoboDojo scene each of those passes
        waits on the same compile. So the warm-up fences ONE pass at a
        time and stops at the first frame with any content, giving up
        after ``_WARMUP_RENDERS`` tries and returning what the renderer
        actually produced rather than pretending there is no camera.
        """
        del width, height

        cameras = self._cameras()
        if not cameras:
            return None

        index = cameras[0][1]
        if camera is not None:
            matches = [slot for name, slot in cameras if name == camera]
            if not matches:
                raise ValueError(
                    f"Unknown RoboDojo camera {camera!r}; this environment "
                    f"has {[name for name, _ in cameras]}"
                )
            index = matches[0]

        try:
            from env.camera_manager.capture.render_sync import (  # type: ignore
                wait_for_latest_cameras,
            )
        except ImportError:  # pragma: no cover - RoboDojo layout dependent
            wait_for_latest_cameras = None
        attempts = 1 if self._render_warmed else self._WARMUP_RENDERS
        frame: Optional[np.ndarray] = None
        for _ in range(attempts):
            if wait_for_latest_cameras is not None:
                wait_for_latest_cameras(
                    self._env, capture_manager=self._env.capture_manager
                )
            else:
                self._env.render()
            frame = self._read_frame(index)
            if frame is not None and frame.any():
                self._render_warmed = True
                break
        return frame

    def _read_frame(self, index: int) -> Optional[np.ndarray]:
        """Pull one camera's RGB buffer for this environment, as HxWx3."""
        data = self._env.capture_manager.step(
            env_ids=[self.env_index], cam_ids=[index]
        )
        if not data:
            return None
        frames = data[0].get("rgb")
        if not frames:
            return None

        frame = _to_numpy(frames[0]["data"])
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return None
        # The annotator delivers RGBA; the alpha channel is dropped rather
        # than reordered -- RoboDojo's RGB byte order is already RGB (fixed
        # alongside RoboTwin in the September 2026 release).
        return np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))


def _robodojo_root() -> str:
    """
    Locate the RoboDojo checkout, which is a repository rather than a
    package: ``env``, ``task`` and ``utils`` are top-level namespace
    packages that only import once its root is on ``sys.path``.
    """
    import os

    root = os.environ.get("ROBODOJO_ROOT")
    candidates = [root] if root else []
    candidates.append(os.getcwd())
    for candidate in candidates:
        if candidate and os.path.isfile(
            os.path.join(candidate, "env", "global_configs.py")
        ):
            return os.path.abspath(candidate)
    raise ValueError(
        "Could not locate a RoboDojo checkout. Set ROBODOJO_ROOT to it, or "
        "run specola from its root directory."
    )


def _ensure_simulation_app(headless: bool, device: str) -> Any:
    """
    Return the running Omniverse app, launching one if the caller has not.

    Isaac Sim allows exactly one app per process and several RoboDojo
    imports fail before it exists, so this runs before any RoboDojo module
    is imported. A caller that already launched one (RoboDojo's own eval
    client does, in ``main.py``, before importing anything else) gets it
    back untouched.
    """
    import argparse

    try:
        import omni.kit.app  # type: ignore

        existing = omni.kit.app.get_app()
        if existing is not None and existing.is_running():
            return existing
    except Exception:
        pass

    from isaaclab.app import AppLauncher  # type: ignore

    # Isaac Lab's kit file arms Kit's hang detector (120 s without an app
    # update ends in a "Kit appears to be hanging" dialog offering to
    # crash the process). This process idles Kit by design: --web keeps
    # serving the dashboard after the episode ends, and nothing pumps
    # the app while it waits. Off, since a stalled RoboDojo step is
    # reported by the monitor's own cadence, not by Kit.
    args = argparse.Namespace(
        headless=headless,
        device=device,
        enable_cameras=True,
        kit_args="--/app/hangDetector/enabled=false",
    )
    try:
        from env.camera_manager.capture.render_sync import (  # type: ignore
            add_zero_delay_kit_args,
        )

        # Must run before the app starts: the update-order setting that
        # makes a rendered frame current is read once, at startup.
        add_zero_delay_kit_args(args)
    except ImportError:  # pragma: no cover - RoboDojo layout dependent
        pass

    app = AppLauncher(args).app
    # An app this factory launched is this factory's to close. Left to
    # interpreter exit, Kit tears down with the camera/replicator graph
    # still live and wedges in plugin unload (omni.syntheticdata), which
    # on a desktop ends in Kit's "appears to be hanging" dialog two
    # minutes after the CLI has already written its output (Isaac Sim
    # 5.1.0, 2026-09-23). SimulationApp.close() stops the replicator
    # before unloading, which is the order every Isaac Lab script uses.
    import atexit
    import signal

    atexit.register(app.close)
    # SimulationApp also installs a SIGINT handler that calls
    # app.shutdown() and unload_all_plugins() directly and exits, skipping
    # close() and every atexit hook (RoboDojo's env.close among them).
    # Ctrl-C on a specola run has to reach the CLI as KeyboardInterrupt
    # instead, after which the hook above closes the app in the order
    # that is verified to shut down.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return app


def _make_robodojo(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> RoboDojoAdapter:
    """
    Build a RoboDojo environment for ``task`` and monitor environment 0.

    ``task`` is a RoboDojo task name -- the shared stem of
    ``task/RoboDojo/tasks/<task>.py`` and ``task/RoboDojo/config/<task>.yml``
    (``stack_bowls``, ``push_T``); ``bash scripts/robodojo.sh tasks`` lists
    them. ``robot`` names an embodiment config under ``env_cfg/robot``
    (``dual_x5``), overriding the one the eval config selects.
    ``target_only`` is unused: RoboDojo's support arms are part of the task,
    not an optional overlay, and dropping them would hide half the scene.

    The evaluation config is ``env_cfg/<ROBODOJO_ENV_CFG>.yml``, defaulting
    to ``arx_x5``; ``ROBODOJO_ROOT`` locates the checkout, ``ROBODOJO_SEED``
    names the layout family (the seed directory under
    ``Assets/Eval_Layout``, default 0) and ``ROBODOJO_LAYOUT`` picks the
    layout within it (default 0), which is also the value handed to
    ``reset()``, as the eval client does. The environment is created with
    ``num_envs=1``, given that layout, and reset once, so the adapter sees a
    populated scene rather than bare robots. Monitoring several parallel environments is done by constructing
    one adapter per index against an environment the caller already owns
    (``registry.wrap_env``), not through this factory, which serves the
    single-scene CLI paths.

    Every robot is built with ``need_planner: False``, so RoboDojo skips its
    cuRobo planner. Only ``RobotManager.solve_ik`` uses it, for the eval
    client's end-effector (``ee``) actions, and nothing a monitored run calls
    does. Set ``ROBODOJO_PLANNER=1`` to keep it when RoboDojo's own eval loop
    drives an environment this factory built.
    """
    del target_only

    if not task:
        raise ValueError(
            "The robodojo adapter requires --task naming a RoboDojo task, "
            "e.g. stack_bowls"
        )

    import importlib
    import os
    import sys

    root = _robodojo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    headless = os.environ.get("ROBODOJO_HEADLESS", "1") != "0"
    device = os.environ.get("ROBODOJO_DEVICE", "cuda:0")
    app = _ensure_simulation_app(headless=headless, device=device)

    from omegaconf import OmegaConf  # type: ignore

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR  # type: ignore
    from utils.load_file import load_yaml  # type: ignore
    from utils.pipeline_utils import (  # type: ignore
        process_config,
        process_randomization,
    )

    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_class = registry.load_task_class(task)

    eval_name = os.environ.get("ROBODOJO_ENV_CFG", "arx_x5")
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, eval_name + ".yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = 1

    selected = dict(eval_cfg["config"])
    if robot:
        selected["robot"] = robot

    def _section(kind: str) -> Any:
        return load_yaml(os.path.join(ENV_CONFIG_PATH, kind, selected[kind] + ".yml"))

    env_cfg = OmegaConf.create(
        {
            "sim": _section("sim"),
            "scene": _section("scene"),
            "camera": _section("camera"),
            "robot": _section("robot"),
            "task_env": load_yaml(
                registry.task_config_path(
                    os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name
                )
            ),
            "eval_cfg": eval_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task_name)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )
    # The eval client treats these two numbers as what they are: the seed
    # names the layout FAMILY (a directory of layouts under
    # Assets/Eval_Layout), and the value handed to reset() is the layout
    # id within it, which is also what seeds the episode's randomisation.
    eval_cfg["seed"] = int(os.environ.get("ROBODOJO_SEED", "0"))
    layout = int(os.environ.get("ROBODOJO_LAYOUT", "0"))
    env_cfg.sim.seed = [layout]

    # The planner's one consumer (RoboDojo 726e9aa): the flag is read at
    # env/robot_manager/robot_manager.py:51-64, the planner used only by
    # solve_ik at :325-338, whose one caller is the eval client's "ee"
    # action path, src/eval_client/eval_env.py:415. Set after
    # process_config, which may swap in another robot config.
    if os.environ.get("ROBODOJO_PLANNER") != "1":
        for robot_cfg in env_cfg.robot.robots:
            robot_cfg["need_planner"] = False

    # With the planner kept, the task constructor builds it. It captures
    # CUDA graphs, and torch.cuda.graph and cuRobo each run gc.collect()
    # before a capture: 38 full collections over the heap Kit and its
    # imports left, 9.7 s of a 14 s constructor (stack_bowls, Isaac Sim
    # 5.1, 2026-10-08). Freezing that heap first exempts it from those
    # collections; unfreezing afterwards hands it back to the collector,
    # so garbage among it is collected later rather than never, and the
    # constructed environment is the same. The one full collection after
    # it runs here: left to the collector, it fell inside a monitored
    # observation as a 0.28 s pause.
    import gc

    gc.freeze()
    try:
        env = task_class(env_cfg, app)
    finally:
        gc.unfreeze()
    gc.collect()

    # A RoboDojo task scene is a SAVED LAYOUT, not a procedural one: the
    # room, table, lights and every task object come from
    # Assets/Eval_Layout/<benchmark>/<config>/<seed>/<task>_<layout>.json,
    # and a TaskEnv reset without one builds the robots alone -- which
    # looks like a working scene until you notice the task has no objects
    # in it. The eval client selects the layout through SeedManager; this
    # does the same, for one layout, and then applies the saved poses the
    # way EvalEnv.setup_scene does so objects start where the layout put
    # them rather than wherever the spawn dropped them.
    from env.seed_manager.seed_manager import SeedManager  # type: ignore

    seed_manager = SeedManager(eval_cfg)
    seed_manager.init_eval()
    env.scene_manager.layout_manager.set_saved_layout(
        0, seed_manager.get_seed_scene_info(layout)
    )

    env.reset(seed=env_cfg.sim.seed)
    env.scene_manager.apply_saved_poses(env_idx_list=[0])
    # Robot-side episode initialisation, the part of EvalEnv.reset that is
    # about the robot rather than about a policy: the origin end pose and
    # the held control targets --mode hold advances under. The reward
    # manager's init_state() is deliberately NOT called; nothing in this
    # adapter reads a verdict.
    env.robot_manager.set_origin_endpose()
    env.robot_manager.set_robot_init_state()

    # Registered after the app's own close hook, so atexit (LIFO) tears
    # the cameras and scene down before the app that hosts them.
    import atexit

    atexit.register(env.close)

    return RoboDojoAdapter(env, env_index=0)


def _handles(obj: Any) -> bool:
    """
    Cheap sniff for a RoboDojo env — never imports RoboDojo itself.

    The module check has to cover ``src.eval_client.eval_env`` as well as
    the task modules: the env a caller actually holds inside an eval run
    is the ``EvalEnv`` subclass defined there, and that is the main reason
    to wrap an env rather than build one. The manager pair is the
    fallback for an env defined somewhere else again (a task subclass in a
    fork, a notebook); it is the same pair the constructor requires, so
    detection claims exactly the envs this adapter can monitor, and it is
    specific enough that no other simulator's env answers to it.
    """
    module = type(obj).__module__ or ""
    if module.startswith(
        ("task.RoboDojo.", "env.environment.", "src.eval_client.")
    ):
        return True
    return all(
        getattr(obj, manager, None) is not None
        for manager in ("robot_manager", "scene_manager")
    )


def _wrap(env: Any, **kwargs: Any) -> RoboDojoAdapter:
    """
    Wrap an environment the caller owns, e.g. from inside a RoboDojo eval
    loop. ``env_index=`` selects which parallel environment to monitor and
    ``dt=`` declares the caller's observation cadence.
    """
    return RoboDojoAdapter(env, **kwargs)


_register(
    name="robodojo",
    factory=_make_robodojo,
    description="RoboDojo manipulation benchmark (Isaac Sim / Isaac Lab)",
    install_hint=(
        "Install RoboDojo and its Isaac Sim stack into their own conda env "
        "(scripts/install.sh), download its assets (scripts/init_assets.sh), "
        "and set ROBODOJO_ROOT to the checkout."
    ),
    handles=_handles,
    wrap=_wrap,
    requires=("isaaclab", "shapely", "omegaconf"),
)
