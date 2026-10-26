"""
CARLA adapter: vehicles, walkers, traffic lights, and map regions.

The point of this adapter is binding lifecycle, not autonomous driving.
CARLA's actor population churns — vehicles and pedestrians spawn and are
destroyed continuously — which makes it the stress case for per-binding
monitor lifecycle (create/suspend/prune) that tabletop manipulation
scenes never exercise. Specs bind classes (``vehicle_*``, ``walker_*``)
and the engine follows individual actors through their lifetimes.

Entities and identity
---------------------
Four classes:

- ``vehicle`` / ``walker``: every live actor whose ``type_id`` starts
  with ``vehicle.`` / ``walker.pedestrian.``. Present while alive,
  absent from the observation after destruction; the engine's binding
  lifecycle handles the rest.
- ``traffic_light``: actors of ``traffic.traffic_light``. Static poles,
  dynamic state — the ``state`` key carries CARLA's own enum code
  (Red=0, Yellow=1, Green=2, Off=3, Unknown=4).
- ``region``: static map geometry discovered at reset — one entity per
  OpenDRIVE junction (bounding box) and one per crosswalk polygon
  (centroid plus axis-aligned half-extents derived from the polygon's
  own vertices; the derivation is noted in the metadata). The class's
  declared schema is the union of the two shapes, so a region spec
  naming a junction-only key (``rotation_*``) monitors junctions and
  leaves crosswalks ineligible — per-entity eligibility, visible in
  ``specola discover`` output, not an error.

CARLA's native id namespaces overlap (a junction id can equal an actor
id — observed live on Town10), so entity ids are adapter-assigned from a
monotonic counter keyed by ``(kind, native id)``. CARLA documents actor
ids as "unique during a given episode" (and fresh spawns after a destroy
got strictly larger ids in the live probe), so within the episode an
adapter monitors, a destroyed actor's entity id never resurfaces under
another class. Loading a different world starts a new episode and needs
a new adapter.

Other actor families (``traffic.speed_limit.*`` signs, sensors,
spectator, walker AI controllers) are not monitored in v1: signs are
static signage with no dynamic state the classes above do not carry,
and controllers/sensors are infrastructure, not scene entities.

Units and frame
---------------
All values are CARLA's own, unconverted, in Unreal's left-handed z-up
world frame: positions in metres, velocities m/s, accelerations m/s^2,
angular velocities deg/s, rotations in degrees (pitch, yaw, roll),
speed limits km/h. Converting handedness or units would falsify
sim-native values; the metadata notes carry the units.

Simulator control
-----------------
The adapter never ticks the world during observation. Stepping belongs
to the caller — or to :meth:`CarlaAdapter.make_advancer`, the shared
CLI's drive hook, which spawns an autopilot fleet plus wandering
pedestrians and periodically destroys and replaces actors so the churn
the adapter exists to showcase actually happens. Monitoring requires
synchronous mode with a fixed timestep (``dt`` refuses anything else):
in asynchronous mode the server free-runs and observation timing would
be whatever the network happened to deliver.

Cameras are caller-owned sensor actors delivering frames via callbacks
— the adapter never spawns one, but :meth:`CarlaAdapter.attach_camera`
adopts a sensor the caller spawned, and the bundled factory attaches a
bird's-eye view over the map's largest junction so ``specola run
--web`` has frames. No ``scene_geometry``/``scene_poses``: the 3D-panel
contract fixes the body table for a whole episode, which actor churn
violates by design; those stay at the base ``None`` defaults.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..entities import EntityState, TrackState
from ..registry import register as _register

_VEHICLE_PREFIX = "vehicle."
_WALKER_PREFIX = "walker.pedestrian."
_TRAFFIC_LIGHT_TYPE = "traffic.traffic_light"

# Dedicated Traffic Manager port for the bundled advancer's fleet; the
# default port (8000) is shared server state other clients may own.
_TM_PORT = 8055

# CARLA's own enum order, used verbatim as the ``state`` code.
_LIGHT_STATE_NOTE = "code: Red=0 Yellow=1 Green=2 Off=3 Unknown=4"

_AXES = ("x", "y", "z")
_ROTATION_AXES = ("pitch", "yaw", "roll")

_TOKEN_RE = re.compile(r"[^0-9a-z]+")


def _token(name: str) -> str:
    """Sanitize a CARLA name (``vehicle.audi.a2``) into a label token."""
    collapsed = _TOKEN_RE.sub("_", str(name).lower()).strip("_")
    if not collapsed:
        raise ValueError(f"CARLA name {name!r} sanitizes to an empty token")
    return collapsed


def _vector_items(prefix, vector, requested=None):
    values = {}
    if requested is None or f"{prefix}_x" in requested:
        values[f"{prefix}_x"] = float(vector.x)
    if requested is None or f"{prefix}_y" in requested:
        values[f"{prefix}_y"] = float(vector.y)
    if requested is None or f"{prefix}_z" in requested:
        values[f"{prefix}_z"] = float(vector.z)
    return values


def _rotation_items(rotation, requested=None):
    values = {}
    if requested is None or "rotation_pitch" in requested:
        values["rotation_pitch"] = float(rotation.pitch)
    if requested is None or "rotation_yaw" in requested:
        values["rotation_yaw"] = float(rotation.yaw)
    if requested is None or "rotation_roll" in requested:
        values["rotation_roll"] = float(rotation.roll)
    return values


def _wants(prefix, labels, requested):
    return any(f"{prefix}_{label}" in requested for label in labels)


def _crosswalk_polygons(
    points: Sequence[Any],
) -> List[List[Tuple[float, float, float]]]:
    """
    Split ``Map.get_crosswalks()``'s flat point list into polygons.

    CARLA delimits polygons by repeating each polygon's first point as
    its closer (verified live on Town10: point [4] equals point [0],
    [9] equals [5], ...). The repeated closer is dropped. A trailing
    unterminated run is kept as-is rather than silently discarded.
    """
    polygons: List[List[Tuple[float, float, float]]] = []
    current: List[Tuple[float, float, float]] = []
    for location in points:
        vertex = (float(location.x), float(location.y), float(location.z))
        if current and vertex == current[0]:
            polygons.append(current)
            current = []
            continue
        current.append(vertex)
    if current:
        polygons.append(current)
    return polygons


def _polygon_region(
    vertices: Sequence[Tuple[float, float, float]],
) -> Dict[str, float]:
    """Centroid and axis-aligned half-extents of one polygon."""
    array = np.asarray(vertices, dtype=float)
    center = array.mean(axis=0)
    half = (array.max(axis=0) - array.min(axis=0)) / 2.0
    values: Dict[str, float] = {}
    for axis, label in enumerate(_AXES):
        values[f"location_{label}"] = float(center[axis])
    for axis, label in enumerate(_AXES):
        values[f"extent_{label}"] = float(half[axis])
    return values


def _meta_triplet(
    prefix: str,
    group: str,
    note: str,
    labels: Tuple[str, ...] = _AXES,
    frame: Optional[str] = None,
) -> Dict[str, PropertyMeta]:
    return {
        f"{prefix}_{label}": PropertyMeta(group=group, note=note, frame=frame)
        for label in labels
    }


def _kinematics_meta() -> Dict[str, PropertyMeta]:
    meta: Dict[str, PropertyMeta] = {}
    meta.update(_meta_triplet("location", "actor position", "m"))
    meta.update(
        _meta_triplet("rotation", "actor rotation", "deg", _ROTATION_AXES)
    )
    meta.update(_meta_triplet("velocity", "actor velocity", "m/s"))
    meta.update(_meta_triplet("acceleration", "actor acceleration", "m/s^2"))
    meta.update(
        _meta_triplet("angular_velocity", "actor angular velocity", "deg/s")
    )
    return meta


_VEHICLE_META: Dict[str, PropertyMeta] = {
    **_kinematics_meta(),
    "control_throttle": PropertyMeta(group="vehicle control", note="normalized"),
    "control_steer": PropertyMeta(group="vehicle control", note="normalized"),
    "control_brake": PropertyMeta(group="vehicle control", note="normalized"),
    "control_gear": PropertyMeta(group="vehicle gear", note="index"),
    "control_hand_brake": PropertyMeta(
        group="vehicle control state", note="flag"
    ),
    "control_reverse": PropertyMeta(
        group="vehicle control state", note="flag"
    ),
    "speed_limit": PropertyMeta(group="speed limit", note="km/h"),
    "is_at_traffic_light": PropertyMeta(group="vehicle status", note="flag"),
    **_meta_triplet("extent", "actor extent", "m, half-size"),
}

_WALKER_META: Dict[str, PropertyMeta] = {
    **_kinematics_meta(),
    "control_speed": PropertyMeta(group="walker control", note="m/s"),
    **_meta_triplet(
        "control_direction", "walker control direction", "unit vector"
    ),
    "control_jump": PropertyMeta(group="walker control state", note="flag"),
    **_meta_triplet("extent", "actor extent", "m, half-size"),
}

_TRAFFIC_LIGHT_META: Dict[str, PropertyMeta] = {
    **_meta_triplet("location", "actor position", "m"),
    **_meta_triplet("rotation", "actor rotation", "deg", _ROTATION_AXES),
    "state": PropertyMeta(group="light state", note=_LIGHT_STATE_NOTE),
    "elapsed_time": PropertyMeta(group="light timing", note="s"),
    "green_time": PropertyMeta(group="light timing", note="s"),
    "yellow_time": PropertyMeta(group="light timing", note="s"),
    "red_time": PropertyMeta(group="light timing", note="s"),
    "pole_index": PropertyMeta(group="light pole", note="index"),
    **_meta_triplet(
        "trigger_offset", "trigger offset", "m, in actor frame"
    ),
    **_meta_triplet("trigger_extent", "trigger extent", "m, half-size"),
}

_JUNCTION_META: Dict[str, PropertyMeta] = {
    **_meta_triplet(
        "location", "region center", "m",
        frame="world (OpenDRIVE junction box center)",
    ),
    **_meta_triplet(
        "rotation", "region rotation", "deg", _ROTATION_AXES,
        frame="junction box axes",
    ),
    **_meta_triplet(
        "extent", "region extent", "m, half-size",
        frame="junction box axes",
    ),
}

_CROSSWALK_META: Dict[str, PropertyMeta] = {
    # The note column stays a pure unit so the shared region groups
    # keep one unit each; the DERIVED nature of these values is carried
    # per key in ``frame`` — a crosswalk's extents are a world-axis
    # AABB proxy over the polygon, not native box geometry, and a spec
    # author reading discovery output must be able to tell.
    **_meta_triplet(
        "location", "region center", "m",
        frame="world (polygon centroid, derived)",
    ),
    **_meta_triplet(
        "extent", "region extent", "m, half-size",
        frame="world axes (polygon AABB, derived)",
    ),
}


class CarlaAdapter(SimulatorAdapter):
    """Monitor one CARLA world. See the module docstring for semantics."""

    def __init__(
        self,
        world: Any,
        client: Any = None,
        vehicle_class: str = "vehicle",
        walker_class: str = "walker",
        traffic_light_class: str = "traffic_light",
        region_class: str = "region",
    ) -> None:
        super().__init__()
        self._world = world
        self._client = client
        self._factory_lifecycle_evidence: Optional[Dict[str, Any]] = None
        self.vehicle_class = vehicle_class
        self.walker_class = walker_class
        self.traffic_light_class = traffic_light_class
        self.region_class = region_class

        self._step = 0
        self._time_origin: Optional[float] = None
        # Attached cameras, insertion-ordered: name -> (sensor, frames).
        # The first attached is the adapter's default frame source.
        self._cameras: Dict[str, Tuple[Any, deque]] = {}
        # Lazy camera factory, invoked once on first camera use — each
        # sensor costs the server a render per tick, so nothing spawns
        # until something actually asks for frames.
        self._camera_provider: Any = None
        self._next_entity_id = 0
        # (kind, native id) -> entity id; assigned once, never reused, so
        # a CARLA actor id colliding with a junction id (observed live)
        # cannot alias two entities, and one entity id can never change
        # class mid-episode.
        self._entity_id_of: Dict[Tuple[str, Any], int] = {}
        self._class_of: Dict[int, str] = {}
        self._label_of: Dict[int, str] = {}
        self._meta_of: Dict[int, Dict[str, PropertyMeta]] = {}
        self._first_seen: Dict[int, int] = {}
        # entity id -> constant property dict for static regions.
        self._region_values: Dict[int, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # SimulatorAdapter surface
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        settings = self._world.get_settings()
        fixed = getattr(settings, "fixed_delta_seconds", None)
        if not getattr(settings, "synchronous_mode", False) or not fixed:
            raise ValueError(
                "CARLA monitoring requires synchronous mode with "
                "fixed_delta_seconds set; the world is asynchronous. "
                "Configure the world (the bundled factory does) before "
                "constructing the adapter."
            )
        return float(fixed)

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def supported_classes(self) -> FrozenSet[str]:
        return frozenset(
            {
                self.vehicle_class,
                self.walker_class,
                self.traffic_light_class,
                self.region_class,
            }
        )

    @property
    def entity_labels(self) -> Dict[int, str]:
        """Label behind each entity id, for discovery output."""
        return dict(self._label_of)

    @property
    def capabilities(self):
        """Declared authority for episode-start compilation.

        Every class's component schema is CLOSED and fixed by
        construction — the extraction methods emit exactly these keys —
        so specifications compile against the declaration rather than
        against whichever actors happened to exist at reset. That is
        what lets selector templates (``vehicle_*[all: velocity_*]``)
        expand class-level and bind actors born mid-episode, which is
        the point of this adapter. Vehicles and walkers are declared
        runtime-populated (a fresh world may hold none); entity ids stay
        open because actors keep arriving.
        """
        from ..capabilities import AdapterCapabilities, VocabularyMode

        meta_by_class = {
            self.vehicle_class: dict(_VEHICLE_META),
            self.walker_class: dict(_WALKER_META),
            self.traffic_light_class: dict(_TRAFFIC_LIGHT_META),
            self.region_class: {**_JUNCTION_META, **_CROSSWALK_META},
        }
        components_by_class = {
            entity_class: frozenset(meta)
            for entity_class, meta in meta_by_class.items()
        }
        return AdapterCapabilities(
            spatial_dims=3,
            class_mode=VocabularyMode.CLOSED,
            supported_classes=self.supported_classes,
            components_by_class=components_by_class,
            component_mode_by_class={
                entity_class: VocabularyMode.CLOSED
                for entity_class in components_by_class
            },
            component_tags_by_class={},
            # Groups are schema facts here: fixed key sets carry fixed
            # PropertyMeta groups, so published group filters resolve
            # statically even for runtime-populated classes.
            component_groups_by_class={
                entity_class: {
                    key: meta.group for key, meta in table.items()
                }
                for entity_class, table in meta_by_class.items()
            },
            runtime_populated_classes=frozenset(
                {self.vehicle_class, self.walker_class}
            ),
            entity_id_mode=VocabularyMode.OPEN,
        )

    def attach_camera(self, sensor: Any, name: str = "birdseye") -> None:
        """
        Adopt a caller-spawned RGB camera sensor as a named frame source.

        The adapter never spawns sensors itself — a sensor is a world
        actor and adding one changes what the simulator computes — so
        cameras are caller-owned; the bundled factory attaches one
        bird's-eye view per major junction, and the dashboard's camera
        picker lists every name. The sensor must not be listening yet.
        Attaching under an existing name replaces (and stops) that
        sensor; the first name ever attached is the adapter default.
        In synchronous mode CARLA delivers exactly one image per tick
        through the callback (verified live: ``image.frame`` matched
        the tick's frame id ten times out of ten), and only the newest
        frame per camera is kept.
        """
        listening = getattr(sensor, "is_listening", None)
        if callable(listening):
            listening = listening()
        if listening:
            raise ValueError(
                "attach_camera needs a sensor that is not listening yet; "
                "this one already streams to another callback"
            )
        key = str(name)
        frames: deque = deque(maxlen=1)

        def _on_image(image: Any) -> None:
            frames.append(
                (int(image.width), int(image.height), bytes(image.raw_data))
            )

        # Start the new sensor before stopping a replaced one: if
        # listen() raises (dead actor), the working camera must keep
        # working rather than freeze on its last frame. A one-tick
        # overlap of two listening sensors is harmless.
        sensor.listen(_on_image)
        previous = self._cameras.get(key)
        if previous is not None:
            # Same-name replacement: stop the old sensor, or it keeps
            # rendering and streaming server-side into an orphaned
            # queue forever.
            try:
                previous[0].stop()
            except RuntimeError:
                pass
        self._cameras[key] = (sensor, frames)

    def set_camera_provider(self, provider: Any) -> None:
        """
        Register a lazy camera factory: a callable returning
        ``[(name, sensor), ...]`` of freshly spawned, not-yet-listening
        sensors, invoked at most once, on the first ``camera_names`` or
        ``capture_frame`` call. Discovery and camera-less monitoring
        runs then create no sensors at all, so presentation cost is
        paid only when something consumes frames.
        """
        self._camera_provider = provider

    def _ensure_cameras(self) -> None:
        provider, self._camera_provider = self._camera_provider, None
        if provider is None:
            return
        try:
            spawned = provider() or []
        except RuntimeError:
            return
        for name, sensor in spawned:
            try:
                self.attach_camera(sensor, name=name)
            except (RuntimeError, ValueError):
                # A camera that cannot be adopted must not leak as an
                # orphaned server-side actor, and must not take the
                # already-adopted ones (or monitoring) down with it.
                try:
                    sensor.destroy()
                except RuntimeError:
                    pass

    def camera_names(self) -> Tuple[str, ...]:
        self._ensure_cameras()
        return tuple(self._cameras)

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        Latest frame of one attached camera as HxWx3 uint8 RGB.

        ``camera`` selects by name; ``None`` means the adapter default
        (the first camera attached). Resolution is fixed when the
        sensor is spawned, so the ``width``/``height`` hints are
        ignored (the base contract allows native-size frames). Returns
        None for unknown names and until the first tick delivers a
        frame. CARLA's raw buffer is BGRA; the slice reorders to RGB.
        """
        del width, height
        self._ensure_cameras()
        if not self._cameras:
            return None
        key = camera if camera is not None else next(iter(self._cameras))
        entry = self._cameras.get(key)
        if entry is None or not entry[1]:
            return None
        frame_width, frame_height, raw = entry[1][-1]
        array = np.frombuffer(raw, dtype=np.uint8).reshape(
            frame_height, frame_width, 4
        )
        return np.ascontiguousarray(array[:, :, 2::-1])

    def property_metadata(self, entity_id: int) -> Dict[str, PropertyMeta]:
        try:
            return dict(self._meta_of[entity_id])
        except KeyError:
            raise KeyError(f"unknown entity id {entity_id}") from None

    def _reset_impl(self) -> MonitorObservation:
        self._step = 0
        # Entities carried over from a previous episode keep their ids
        # (the actors are genuinely the same), but their age restarts
        # with the episode — a stale first-seen step from a previous
        # episode would make track_age negative, which the base class
        # rejects.
        self._first_seen = dict.fromkeys(self._first_seen, 0)
        snapshot = self._world.get_snapshot()
        self._time_origin = float(snapshot.timestamp.elapsed_seconds)
        self._discover_regions()
        return self._observation()

    def _get_observation_impl(self) -> MonitorObservation:
        self._step += 1
        return self._observation()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def _entity_id(
        self,
        kind: str,
        native_id: Any,
        entity_class: str,
        label: str,
        meta: Dict[str, PropertyMeta],
    ) -> int:
        key = (kind, native_id)
        entity_id = self._entity_id_of.get(key)
        if entity_id is None:
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            self._entity_id_of[key] = entity_id
            self._class_of[entity_id] = entity_class
            self._label_of[entity_id] = label
            self._meta_of[entity_id] = meta
            self._first_seen[entity_id] = self._step
        return entity_id

    # ------------------------------------------------------------------
    # Static regions
    # ------------------------------------------------------------------

    def _discover_regions(self) -> None:
        """Materialize junctions and crosswalks as region entities.

        Junctions are deduplicated by OpenDRIVE id from the topology's
        junction waypoints; crosswalks are indexed in the order the map
        reports their polygons, which is stable for a given map.
        """
        carla_map = self._world.get_map()

        junctions: Dict[int, Any] = {}
        for begin, _ in carla_map.get_topology():
            if getattr(begin, "is_junction", False):
                junction = begin.get_junction()
                if junction is not None:
                    junctions.setdefault(int(junction.id), junction)

        for native_id, junction in sorted(junctions.items()):
            box = junction.bounding_box
            values = {
                **_vector_items("location", box.location),
                **_rotation_items(box.rotation),
                **_vector_items("extent", box.extent),
            }
            entity_id = self._entity_id(
                "junction",
                native_id,
                self.region_class,
                f"junction_{native_id}",
                _JUNCTION_META,
            )
            self._region_values[entity_id] = values

        for index, polygon in enumerate(
            _crosswalk_polygons(carla_map.get_crosswalks())
        ):
            entity_id = self._entity_id(
                "crosswalk",
                index,
                self.region_class,
                f"crosswalk_{index}",
                _CROSSWALK_META,
            )
            self._region_values[entity_id] = _polygon_region(polygon)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> MonitorObservation:
        entities: Dict[int, EntityState] = {}

        for entity_id, values in self._region_values.items():
            requested = self._requested_property_keys(self.region_class)
            properties = values if requested is None else {
                key: value for key, value in values.items() if key in requested
            }
            entities[entity_id] = self._entity_state(
                entity_id,
                properties,
                velocity=None,
                position=np.array([values[f"location_{axis}"] for axis in _AXES]),
            )

        for actor in self._world.get_actors():
            type_id = str(actor.type_id)
            try:
                if type_id.startswith(_VEHICLE_PREFIX):
                    entity_class = self.vehicle_class
                    values, position, velocity = self._vehicle_properties(actor)
                    meta = _VEHICLE_META
                elif type_id.startswith(_WALKER_PREFIX):
                    entity_class = self.walker_class
                    values, position, velocity = self._walker_properties(actor)
                    meta = _WALKER_META
                elif type_id == _TRAFFIC_LIGHT_TYPE:
                    entity_class = self.traffic_light_class
                    values, position, velocity = self._traffic_light_properties(actor)
                    meta = _TRAFFIC_LIGHT_META
                else:
                    continue
            except RuntimeError:
                # CARLA raises RuntimeError both for a destroyed actor
                # and for transport or server failures. Only a genuinely
                # dead actor may be silently absent (matching the next
                # observation, where it is gone from get_actors);
                # masking a live-read failure would fabricate a
                # disappearance and suspend healthy bindings. is_alive
                # is client-cached and lags a same-tick external
                # destroy, so the server is asked before propagating.
                if (
                    not actor.is_alive
                    or self._world.get_actor(actor.id) is None
                ):
                    continue
                raise
            entity_id = self._entity_id(
                "actor",
                int(actor.id),
                entity_class,
                f"{_token(type_id)}_{int(actor.id)}",
                meta,
            )
            entities[entity_id] = self._entity_state(
                entity_id,
                values,
                velocity=velocity,
                position=position,
            )

        snapshot = self._world.get_snapshot()
        elapsed = float(snapshot.timestamp.elapsed_seconds)
        origin = self._time_origin if self._time_origin is not None else elapsed
        return MonitorObservation(
            timestep=self._step,
            time=elapsed - origin,
            perceived_entities=entities,
        )

    def _entity_state(
        self,
        entity_id: int,
        values: Dict[str, float],
        velocity: Optional[np.ndarray],
        position: Optional[np.ndarray],
    ) -> EntityState:
        return EntityState(
            entity_id=entity_id,
            entity_class=self._class_of[entity_id],
            position=position,
            velocity=velocity,
            track_age=self._step - self._first_seen[entity_id],
            track_state=TrackState.CONFIRMED,
            properties=values,
        )

    def _kinematics(self, actor: Any, entity_class: str):
        requested = self._requested_property_keys(entity_class)
        transform = actor.get_transform()
        location = transform.location
        native_velocity = actor.get_velocity()
        location_values = _vector_items("location", location)
        velocity_values = _vector_items("velocity", native_velocity)
        position = np.array(list(location_values.values()))
        velocity = np.array(list(velocity_values.values()))
        if requested is None:
            values = location_values
        else:
            values = {key: value for key, value in location_values.items()
                      if key in requested}
        if requested is None or _wants("rotation", _ROTATION_AXES, requested):
            values.update(_rotation_items(transform.rotation, requested))
        values.update(velocity_values if requested is None else {
            key: value for key, value in velocity_values.items()
            if key in requested
        })
        for prefix, getter in (
            ("acceleration", actor.get_acceleration),
            ("angular_velocity", actor.get_angular_velocity),
        ):
            if requested is None or _wants(prefix, _AXES, requested):
                values.update(_vector_items(prefix, getter(), requested))
        return values, position, velocity

    def _vehicle_properties(self, actor: Any):
        requested = self._requested_property_keys(self.vehicle_class)
        values, position, velocity = self._kinematics(actor, self.vehicle_class)
        control_keys = frozenset({
            "control_throttle", "control_steer", "control_brake",
            "control_gear", "control_hand_brake", "control_reverse",
        })
        if requested is None or requested & control_keys:
            control = actor.get_control()
            if requested is None or "control_throttle" in requested:
                values["control_throttle"] = float(control.throttle)
            if requested is None or "control_steer" in requested:
                values["control_steer"] = float(control.steer)
            if requested is None or "control_brake" in requested:
                values["control_brake"] = float(control.brake)
            if requested is None or "control_gear" in requested:
                values["control_gear"] = float(control.gear)
            if requested is None or "control_hand_brake" in requested:
                values["control_hand_brake"] = float(bool(control.hand_brake))
            if requested is None or "control_reverse" in requested:
                values["control_reverse"] = float(bool(control.reverse))
        if requested is None or "speed_limit" in requested:
            values["speed_limit"] = float(actor.get_speed_limit())
        if requested is None or "is_at_traffic_light" in requested:
            values["is_at_traffic_light"] = float(bool(actor.is_at_traffic_light()))
        if requested is None or _wants("extent", _AXES, requested):
            values.update(_vector_items(
                "extent", actor.bounding_box.extent, requested
            ))
        return values, position, velocity

    def _walker_properties(self, actor: Any):
        requested = self._requested_property_keys(self.walker_class)
        values, position, velocity = self._kinematics(actor, self.walker_class)
        if requested is None or requested & frozenset({
            "control_speed", "control_direction_x", "control_direction_y",
            "control_direction_z", "control_jump",
        }):
            control = actor.get_control()
            if requested is None or "control_speed" in requested:
                values["control_speed"] = float(control.speed)
            if requested is None or _wants("control_direction", _AXES, requested):
                direction = control.direction
                values.update(_vector_items(
                    "control_direction", direction, requested
                ))
            if requested is None or "control_jump" in requested:
                values["control_jump"] = float(bool(control.jump))
        if requested is None or _wants("extent", _AXES, requested):
            values.update(_vector_items(
                "extent", actor.bounding_box.extent, requested
            ))
        return values, position, velocity

    def _traffic_light_properties(self, actor: Any):
        requested = self._requested_property_keys(self.traffic_light_class)
        transform = actor.get_transform()
        location_values = _vector_items("location", transform.location)
        position = np.array(list(location_values.values()))
        if requested is None:
            values = location_values
        else:
            values = {key: value for key, value in location_values.items()
                      if key in requested}
        if requested is None or _wants("rotation", _ROTATION_AXES, requested):
            values.update(_rotation_items(transform.rotation, requested))
        for key, getter in (
            ("state", lambda: int(actor.state)),
            ("elapsed_time", actor.get_elapsed_time),
            ("green_time", actor.get_green_time),
            ("yellow_time", actor.get_yellow_time),
            ("red_time", actor.get_red_time),
            ("pole_index", actor.get_pole_index),
        ):
            if requested is None or key in requested:
                values[key] = float(getter())
        if requested is None or (
            _wants("trigger_offset", _AXES, requested)
            or _wants("trigger_extent", _AXES, requested)
        ):
            trigger = actor.trigger_volume
            if requested is None or _wants("trigger_offset", _AXES, requested):
                values.update(_vector_items(
                    "trigger_offset", trigger.location, requested
                ))
            if requested is None or _wants("trigger_extent", _AXES, requested):
                values.update(_vector_items(
                    "trigger_extent", trigger.extent, requested
                ))
        return values, position, None

    # ------------------------------------------------------------------
    # Drive hook for the shared CLI
    # ------------------------------------------------------------------

    def make_advancer(self, mode: str, demo: Optional[str], seed: int):
        """
        Drive the world for ``specola run``.

        ``hold`` ticks the world and nothing else — the scene runs on
        whatever traffic already exists. ``random`` maintains an
        autopilot fleet plus wandering pedestrians and periodically
        destroys a random fleet member and spawns a replacement, so the
        binding-lifecycle churn this adapter showcases actually occurs.
        ``replay`` (CARLA's own recorder files) is not supported yet, and
        ``--demo`` is refused under ``hold`` and ``random`` rather than
        ignored (#132). Nor is there a Specola state tier: on 0.9.16, after
        ``set_transform``, ``set_target_velocity`` and
        ``set_target_angular_velocity`` on every vehicle and walker and no
        tick, the observation equalled the last tick's on every key the
        adapter reads from CARLA, because those reads return the state of
        the last tick the client received
        (``docs/evidence/state-replay-carla-2026-10-14.md``).

        The fleet is owned by the advancer: everything it spawned is
        destroyed at interpreter exit (atexit), leaving the server
        reusable for the next client.
        """
        if mode == "replay":
            raise SystemExit(
                "--mode replay is not supported by the carla adapter yet"
            )
        if demo is not None:
            raise SystemExit(
                "The carla adapter has no --demo replay: it reads no "
                "demonstration format."
            )
        world = self._world
        if mode == "hold":
            def advance_hold(step: int) -> bool:
                world.tick()
                return True
            return advance_hold

        import atexit
        import random as _random

        import carla

        rng = _random.Random(seed)
        blueprints = list(
            world.get_blueprint_library().filter("vehicle.*")
        )
        walker_blueprints = list(
            world.get_blueprint_library().filter("walker.pedestrian.*")
        )
        controller_blueprint = world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        spawn_points = list(world.get_map().get_spawn_points())
        rng.shuffle(spawn_points)
        spawn_cursor = 0

        traffic_manager = None
        if self._client is not None:
            # A dedicated Traffic Manager port: the default TM (8000) is
            # shared server state another client may own, and TM exposes
            # no getters to record-and-restore its mode or seed. Owning
            # a separate instance keeps the sync flip and seed local.
            traffic_manager = self._client.get_trafficmanager(_TM_PORT)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(seed)

        fleet: List[Any] = []
        walkers: List[Tuple[Any, Any]] = []

        def spawn_vehicle() -> None:
            # Rotate through the shuffled spawn points rather than
            # consuming them: an unbounded run churns indefinitely, and
            # a finite pool would silently shrink the fleet once empty.
            # An occupied point makes try_spawn return None; the next
            # churn simply tries the next point.
            nonlocal spawn_cursor
            if not spawn_points:
                return
            blueprint = rng.choice(blueprints)
            point = spawn_points[spawn_cursor % len(spawn_points)]
            spawn_cursor += 1
            vehicle = world.try_spawn_actor(blueprint, point)
            if vehicle is None:
                return
            if traffic_manager is not None:
                vehicle.set_autopilot(True, traffic_manager.get_port())
            fleet.append(vehicle)

        def spawn_walker() -> None:
            blueprint = rng.choice(walker_blueprints)
            location = world.get_random_location_from_navigation()
            if location is None:
                return
            walker = world.try_spawn_actor(
                blueprint, carla.Transform(location)
            )
            if walker is None:
                return
            controller = world.try_spawn_actor(
                controller_blueprint, carla.Transform(), attach_to=walker
            )
            walkers.append((walker, controller))

        started_controllers: set = set()

        def start_walkers() -> None:
            # Controllers can only start after a tick has registered
            # them; started ids are tracked here because Boost.Python
            # actor objects do not accept ad-hoc attributes.
            for walker, controller in walkers:
                if controller is not None and controller.id not in (
                    started_controllers
                ):
                    controller.start()
                    destination = world.get_random_location_from_navigation()
                    if destination is not None:
                        controller.go_to_location(destination)
                    started_controllers.add(controller.id)

        def retire_vehicle(vehicle: Any) -> None:
            # A vehicle destroyed while still registered with the
            # Traffic Manager crashes the TM worker thread when it next
            # operates on the dead actor (std::terminate, observed
            # live), so deregister first. Every per-actor call is
            # fenced: the actor may have died server-side already.
            try:
                if traffic_manager is not None:
                    vehicle.set_autopilot(False, traffic_manager.get_port())
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError:
                pass

        def teardown() -> None:
            for walker, controller in walkers:
                try:
                    if controller is not None and controller.is_alive:
                        controller.stop()
                        controller.destroy()
                    if walker.is_alive:
                        walker.destroy()
                except RuntimeError:
                    pass
            for vehicle in fleet:
                retire_vehicle(vehicle)
            walkers.clear()
            fleet.clear()
            if traffic_manager is not None:
                # "Always disable sync mode before the script ends" —
                # a synchronous TM with nobody ticking blocks the server.
                traffic_manager.set_synchronous_mode(False)

        atexit.register(teardown)

        n_vehicles = 20
        n_walkers = 10
        churn_period = 40

        for _ in range(n_vehicles):
            spawn_vehicle()
        for _ in range(n_walkers):
            spawn_walker()

        def advance(step: int) -> bool:
            world.tick()
            if step == 1:
                start_walkers()
            if step and step % churn_period == 0:
                # The churn this adapter exists to demonstrate: retire a
                # random fleet vehicle and replace it next period.
                live = [v for v in fleet if v.is_alive]
                if live:
                    victim = rng.choice(live)
                    fleet.remove(victim)
                    retire_vehicle(victim)
                spawn_vehicle()
            return True

        return advance


def _make_carla(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> CarlaAdapter:
    """
    Connect to a running CARLA server and monitor its world.

    ``task`` optionally names a town to load (``Town04``); when omitted
    the currently loaded world is monitored as-is. ``robot`` is unused:
    CARLA has no embodiment to select. The server address comes from
    ``SPECOLA_CARLA_HOST``/``SPECOLA_CARLA_PORT`` (default
    127.0.0.1:2000) — the server is a separately launched local process
    (``CarlaUE4.sh -RenderOffScreen``).

    The factory switches the world to synchronous mode at 20 Hz for the
    monitoring session and restores the previous settings at interpreter
    exit, so a shared server is left the way it was found.
    """
    del robot, target_only

    import atexit
    import os

    import carla

    host = os.environ.get("SPECOLA_CARLA_HOST", "127.0.0.1")
    port = int(os.environ.get("SPECOLA_CARLA_PORT", "2000"))
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    if task:
        current = world.get_map().name
        if not current.endswith(task):
            world = client.load_world(task)

    # CARLA returns a value object. Keep the complete original untouched;
    # constructing defaults loses server-specific values and optional state.
    original = world.get_settings()
    settings = world.get_settings()
    if settings != original:
        raise RuntimeError("CARLA settings changed during factory setup")
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    lifecycle: Dict[str, Any] = {"setup": {}, "restorations": []}

    def restore() -> None:
        import sys

        record: Dict[str, Any] = {}
        lifecycle["restorations"].append(record)
        # A snapshot failure must not prevent the attempt to restore settings.
        try:
            record["frame_before"] = world.get_snapshot().frame
        except Exception as error:
            record["frame_before_error"] = str(error)
        try:
            world.apply_settings(original)
            # Bound native equality compares the complete EpisodeSettings
            # value. Do not substitute a hand-maintained field subset.
            record["settings_match"] = world.get_settings() == original
            if not record["settings_match"]:
                raise RuntimeError("CARLA restored settings differ from the complete original")
            record["frame_after"] = world.get_snapshot().frame
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            print("CARLA world settings restoration failed: " + record["error"], file=sys.stderr)

    try:
        lifecycle["setup"]["frame_before"] = world.get_snapshot().frame
        # ApplySettings may advance native frames while establishing a fixed
        # timestep. This is factory lifecycle work, outside observation.
        world.apply_settings(settings)
        lifecycle["setup"]["frame_after"] = world.get_snapshot().frame
        if world.get_settings() != settings:
            raise RuntimeError("CARLA applied settings differ from the requested configuration")
        adapter = CarlaAdapter(world, client=client)
        adapter._factory_lifecycle_evidence = lifecycle
        # Cameras remain lazy; provider installation is part of factory setup.
        adapter.set_camera_provider(lambda: _spawn_birdseye_cameras(world))
    except BaseException:
        restore()
        raise

    # Bird's-eye monitoring cameras over the map's major junctions.
    # Cameras are caller-owned sensor actors, and the factory is the
    # caller here — but they spawn lazily, on first camera use, so
    # discovery and camera-less runs pay no per-tick render cost.
    # Failed spawns degrade to fewer (or no) cameras rather than no
    # monitoring. The dashboard's camera picker lists one entry per
    # junction, largest first.

    def retire_cameras() -> None:
        for sensor, _frames in list(adapter._cameras.values()):
            try:
                sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass

    atexit.register(restore)
    atexit.register(retire_cameras)

    return adapter


_BIRDSEYE_CAMERAS = 3


def _spawn_birdseye_cameras(world: Any) -> List[Tuple[str, Any]]:
    """Top-down RGB cameras 50 m above the map's largest junctions.

    Junctions are the busiest fixed vantages a map offers without
    following any particular actor; the largest few, largest first,
    named ``junction_<opendrive id>`` so the picker entry says where
    the view is. 50 m with the default 90° field of view covers
    roughly a 100 m square — comfortably more than Town10's largest
    junction (about 40 m across, probed live). Each camera costs the
    server one extra render per tick, which is why the count is
    bounded rather than one per junction.
    """
    import carla

    junctions: Dict[int, Any] = {}
    for begin, _ in world.get_map().get_topology():
        if getattr(begin, "is_junction", False):
            junction = begin.get_junction()
            if junction is not None:
                junctions[int(junction.id)] = junction
    ranked = sorted(
        junctions.values(),
        key=lambda j: j.bounding_box.extent.x * j.bounding_box.extent.y,
        reverse=True,
    )[:_BIRDSEYE_CAMERAS]

    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", "960")
    blueprint.set_attribute("image_size_y", "540")

    cameras: List[Tuple[str, Any]] = []
    for junction in ranked:
        center = junction.bounding_box.location
        try:
            sensor = world.try_spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(
                        x=center.x, y=center.y, z=center.z + 50.0
                    ),
                    carla.Rotation(pitch=-90.0),
                ),
            )
        except RuntimeError:
            # One failed spawn must not discard the cameras already
            # spawned (they would leak server-side, unowned).
            continue
        if sensor is not None:
            cameras.append((f"junction_{int(junction.id)}", sensor))
    return cameras


_register(
    name="carla",
    factory=_make_carla,
    description=(
        "CARLA driving worlds: vehicles, walkers, traffic lights, and "
        "map regions, with actor churn driving binding lifecycle"
    ),
    install_hint=(
        "pip install carla, and launch a CARLA server locally "
        "(CarlaUE4.sh -RenderOffScreen)"
    ),
    requires=("carla",),
)
