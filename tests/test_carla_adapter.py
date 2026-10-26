"""Sim-free tests for the CARLA adapter.

CARLA's server is a 19 GB Unreal process, so these use stand-in objects
shaped like the 0.9.16 client API (shapes verified against the installed
wheel and a live Town10 probe; see scripts/carla_stage_probe.py at the
workspace root). What is pinned here is the adapter's own logic: identity
under churn, metadata parity, the region derivations, and the class/unit
conventions selectors depend on.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from specola.adapters.carla import (
    CarlaAdapter,
    _crosswalk_polygons,
    _polygon_region,
)
from specola.discovery_model import property_metadata_for


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Rot:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Vec()
        self.rotation = rotation or _Rot()


class _Box:
    def __init__(self, location=None, extent=None, rotation=None):
        self.location = location or _Vec()
        self.extent = extent or _Vec(1.0, 0.5, 0.4)
        self.rotation = rotation or _Rot()


class _VehicleControl:
    throttle = 0.3
    steer = -0.1
    brake = 0.0
    gear = 2
    hand_brake = False
    reverse = False


class _WalkerControl:
    speed = 1.4
    direction = _Vec(1.0, 0.0, 0.0)
    jump = False


class _Actor:
    def __init__(self, actor_id, type_id, location=None):
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = True
        self._transform = _Transform(location=location or _Vec(1.0, 2.0, 0.1))
        self.bounding_box = _Box()

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return _Vec(3.0, 0.0, 0.0)

    def get_acceleration(self):
        return _Vec(0.1, 0.0, 0.0)

    def get_angular_velocity(self):
        return _Vec(0.0, 0.0, 5.0)


class _Vehicle(_Actor):
    def __init__(self, actor_id, location=None):
        super().__init__(actor_id, "vehicle.audi.a2", location)

    def get_control(self):
        return _VehicleControl()

    def get_speed_limit(self):
        return 30.0

    def is_at_traffic_light(self):
        return False


class _Walker(_Actor):
    def __init__(self, actor_id):
        super().__init__(actor_id, "walker.pedestrian.0022")

    def get_control(self):
        return _WalkerControl()


class _TrafficLight(_Actor):
    def __init__(self, actor_id):
        super().__init__(actor_id, "traffic.traffic_light")
        self.state = 2  # int()-able, like carla.TrafficLightState.Green
        self.trigger_volume = _Box(location=_Vec(-7.0, 38.4, 1.0))

    def get_elapsed_time(self):
        return 1.68

    def get_green_time(self):
        return 10.0

    def get_yellow_time(self):
        return 3.0

    def get_red_time(self):
        return 2.0

    def get_pole_index(self):
        return 3


class _Waypoint:
    def __init__(self, junction=None):
        self.is_junction = junction is not None
        self._junction = junction

    def get_junction(self):
        return self._junction


class _Junction:
    def __init__(self, junction_id):
        self.id = junction_id
        self.bounding_box = _Box(
            location=_Vec(-47.4, -59.7, 0.0), extent=_Vec(19.8, 16.4, 0.0)
        )


class _Map:
    def __init__(self, junction_ids=(23,), crosswalks=()):
        self._junctions = [_Junction(jid) for jid in junction_ids]
        self._crosswalks = list(crosswalks)

    def get_topology(self):
        pairs = [(_Waypoint(j), _Waypoint()) for j in self._junctions]
        # The same junction typically appears behind several segments.
        return pairs + pairs

    def get_crosswalks(self):
        return self._crosswalks

    def get_spawn_points(self):
        return []


class _Timestamp:
    def __init__(self, elapsed):
        self.elapsed_seconds = elapsed
        self.delta_seconds = 0.05


class _Snapshot:
    def __init__(self, elapsed):
        self.timestamp = _Timestamp(elapsed)


class _Settings:
    def __init__(self, synchronous=True, dt=0.05):
        self.synchronous_mode = synchronous
        self.fixed_delta_seconds = dt


class _World:
    def __init__(self, actors=(), junction_ids=(23,), crosswalks=(),
                 synchronous=True, dt=0.05):
        self.actors = list(actors)
        self._map = _Map(junction_ids, crosswalks)
        self._settings = _Settings(synchronous, dt)
        self.elapsed = 100.0

    def get_actors(self):
        return list(self.actors)

    def get_actor(self, actor_id):
        return next(
            (actor for actor in self.actors if actor.id == actor_id), None
        )

    def get_map(self):
        return self._map

    def get_settings(self):
        return self._settings

    def get_snapshot(self):
        return _Snapshot(self.elapsed)


def _closed_square(x0, y0, side):
    return [
        _Vec(x0, y0, 0.0),
        _Vec(x0 + side, y0, 0.0),
        _Vec(x0 + side, y0 + side, 0.0),
        _Vec(x0, y0 + side, 0.0),
        _Vec(x0, y0, 0.0),  # CARLA closes each polygon with its first point
    ]


class _CountingProxy:
    def __init__(self, target, calls, prefix):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        value = getattr(self._target, name)
        path = f"{self._prefix}.{name}"
        if callable(value):
            def invoke(*args, **kwargs):
                self.calls[path] += 1
                return _counted(value(*args, **kwargs), self.calls, path)
            return invoke
        self.calls[path] += 1
        return _counted(value, self.calls, path)


def _counted(value, calls=None, prefix="actor"):
    calls = Counter() if calls is None else calls
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _CountingProxy(value, calls, prefix)


def _structural_reads(prefix, velocity):
    paths = [
        f"{prefix}.type_id", f"{prefix}.get_transform",
        f"{prefix}.get_transform.location",
        *(f"{prefix}.get_transform.location.{axis}" for axis in "xyz"),
        f"{prefix}.id", f"{prefix}.id",
    ] + velocity * [f"{prefix}.get_velocity", *(
        f"{prefix}.get_velocity.{axis}" for axis in "xyz"
    )]
    return Counter(paths)


def _scope_subject():
    actors = [_counted(actor, prefix=prefix) for actor, prefix in zip(
        (_Vehicle(100), _Walker(101), _TrafficLight(23)), ("vehicle", "walker", "light")
    )]
    world = _World(actors=actors, junction_ids=(23,), crosswalks=_closed_square(0, 0, 2))
    return world, CarlaAdapter(world), actors


def _assert_exact_projection(full, scoped, full_adapter, scoped_adapter, scope):
    assert (scoped.timestep, scoped.time, scoped.component_tags) == (
        full.timestep, full.time, full.component_tags
    )
    assert scoped_adapter.entity_labels == full_adapter.entity_labels
    assert set(scoped.perceived_entities) == set(full.perceived_entities)
    full_meta = property_metadata_for(full_adapter, full.perceived_entities)
    scoped_meta = property_metadata_for(scoped_adapter, scoped.perceived_entities)
    for entity_id, expected in full.perceived_entities.items():
        actual = scoped.perceived_entities[entity_id]
        allowed = None if scope is None else scope.get(expected.entity_class, ())
        assert (actual.entity_id, actual.entity_class, actual.track_age,
                actual.track_state) == (
            expected.entity_id, expected.entity_class, expected.track_age,
            expected.track_state,
        )
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.velocity, expected.velocity)
        expected_properties = expected.properties if allowed is None else {
            key: value for key, value in expected.properties.items()
            if key in allowed
        }
        assert actual.properties == expected_properties
        assert scoped_meta[entity_id] == {
            key: value for key, value in full_meta[entity_id].items()
            if allowed is None or key in allowed
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_crosswalk_polygons_split_on_repeated_first_point():
    points = _closed_square(0, 0, 2) + _closed_square(10, 10, 4)
    polygons = _crosswalk_polygons(points)
    assert [len(p) for p in polygons] == [4, 4]
    assert polygons[0][0] == (0.0, 0.0, 0.0)
    assert polygons[1][0] == (10.0, 10.0, 0.0)


def test_crosswalk_polygons_keep_unterminated_tail():
    points = _closed_square(0, 0, 2) + [_Vec(50, 50, 0), _Vec(51, 50, 0)]
    polygons = _crosswalk_polygons(points)
    assert [len(p) for p in polygons] == [4, 2]


def test_polygon_region_centroid_and_extent():
    values = _polygon_region(
        [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    )
    assert values["location_x"] == pytest.approx(2.0)
    assert values["location_y"] == pytest.approx(1.0)
    assert values["extent_x"] == pytest.approx(2.0)
    assert values["extent_y"] == pytest.approx(1.0)
    assert values["extent_z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Adapter behavior
# ---------------------------------------------------------------------------


def _adapter(world):
    return CarlaAdapter(world)


def test_dt_requires_synchronous_fixed_step():
    with pytest.raises(ValueError):
        _adapter(_World(synchronous=False, dt=None)).dt
    with pytest.raises(ValueError):
        _adapter(_World(synchronous=True, dt=None)).dt
    assert _adapter(_World()).dt == pytest.approx(0.05)


@pytest.mark.parametrize("mode", ["hold", "random"])
def test_make_advancer_refuses_demo(mode):
    # The stub world has no blueprint library: random's refusal has to come
    # before its fleet spawn touches the world.
    with pytest.raises(SystemExit, match="The carla adapter has no --demo replay"):
        _adapter(_World()).make_advancer(mode, "demo.hdf5", 0)


def test_metadata_parity_across_every_entity():
    world = _World(
        actors=[_Vehicle(100), _Walker(101), _TrafficLight(23)],
        junction_ids=(23, 134),
        crosswalks=_closed_square(0, 0, 2),
    )
    adapter = _adapter(world)
    observation = adapter.reset()
    assert len(observation.perceived_entities) == 6
    for entity_id, entity in observation.perceived_entities.items():
        assert set(adapter.property_metadata(entity_id)) == set(
            entity.properties
        ), adapter.entity_labels[entity_id]


def test_no_group_mixes_units():
    world = _World(
        actors=[_Vehicle(100), _Walker(101), _TrafficLight(23)],
        crosswalks=_closed_square(0, 0, 2),
    )
    adapter = _adapter(world)
    adapter.reset()
    units_by_group = {}
    for entity_id in adapter.entity_labels:
        for meta in adapter.property_metadata(entity_id).values():
            units_by_group.setdefault(meta.group, set()).add(meta.note)
    for group, units in units_by_group.items():
        assert len(units) == 1, f"group {group!r} mixes notes {units}"


def test_native_id_collision_across_kinds_does_not_alias():
    # Junction 23 and traffic-light actor 23 share a native id (observed
    # live on Town10); they must be distinct entities of distinct classes.
    world = _World(actors=[_TrafficLight(23)], junction_ids=(23,))
    adapter = _adapter(world)
    observation = adapter.reset()
    classes = sorted(
        entity.entity_class
        for entity in observation.perceived_entities.values()
    )
    assert classes == ["region", "traffic_light"]


def test_churn_retires_ids_and_never_reuses_them():
    vehicle = _Vehicle(100)
    world = _World(actors=[vehicle], junction_ids=())
    adapter = _adapter(world)
    first = adapter.reset()
    [vehicle_id] = list(first.perceived_entities)

    world.actors = []  # destroyed
    world.elapsed += 0.05
    second = adapter.get_observation()
    assert vehicle_id not in second.perceived_entities

    world.actors = [_Vehicle(102)]  # CARLA ids are monotonic
    world.elapsed += 0.05
    third = adapter.get_observation()
    [replacement_id] = list(third.perceived_entities)
    assert replacement_id != vehicle_id
    assert adapter.entity_labels[replacement_id] == "vehicle_audi_a2_102"

    # The original actor id maps back to its old entity id if ever seen
    # again, keeping identity stable rather than minting a duplicate.
    world.actors = [_Vehicle(100)]
    world.elapsed += 0.05
    fourth = adapter.get_observation()
    assert list(fourth.perceived_entities) == [vehicle_id]


def test_traffic_light_state_is_the_enum_code():
    world = _World(actors=[_TrafficLight(23)], junction_ids=())
    adapter = _adapter(world)
    observation = adapter.reset()
    [entity] = observation.perceived_entities.values()
    assert entity.properties["state"] == 2.0
    assert entity.properties["green_time"] == 10.0
    assert entity.properties["trigger_offset_x"] == pytest.approx(-7.0)


def test_time_is_relative_to_reset():
    world = _World(actors=[_Vehicle(100)], junction_ids=())
    adapter = _adapter(world)
    first = adapter.reset()
    assert first.time == pytest.approx(0.0)
    world.elapsed += 0.25
    second = adapter.get_observation()
    assert second.time == pytest.approx(0.25)
    assert second.timestep == 1


def test_supported_classes_are_closed():
    adapter = _adapter(_World())
    assert adapter.supported_classes == frozenset(
        {"vehicle", "walker", "traffic_light", "region"}
    )


def test_regions_carry_position_and_no_velocity():
    world = _World(
        actors=[],
        junction_ids=(23,),
        crosswalks=_closed_square(0, 0, 2),
    )
    adapter = _adapter(world)
    observation = adapter.reset()
    assert len(observation.perceived_entities) == 2
    for entity in observation.perceived_entities.values():
        assert entity.entity_class == "region"
        assert entity.position is not None
        assert entity.velocity is None


def test_second_reset_restarts_track_age():
    # An adapter may be reset again on the same persistent world; ages
    # restart with the episode even for entities first seen mid-episode
    # the time before (a stale first-seen step made track_age negative,
    # which base-class validation rejects).
    world = _World(actors=[_Vehicle(100)], junction_ids=())
    adapter = _adapter(world)
    adapter.reset()
    world.actors.append(_Vehicle(101))  # churned in mid-episode
    world.elapsed += 0.05
    adapter.get_observation()

    world.elapsed += 0.05
    observation = adapter.reset()
    assert all(
        entity.track_age == 0
        for entity in observation.perceived_entities.values()
    )


def test_dead_actor_is_absent_but_live_read_failure_propagates():
    # CARLA raises RuntimeError both for destroyed actors and for
    # transport/server failures. Only a genuinely dead actor may be
    # silently absent; masking a live failure would fabricate a
    # disappearance and suspend healthy bindings (external review).
    class _DeadVehicle(_Vehicle):
        def __init__(self, actor_id):
            super().__init__(actor_id)
            self.is_alive = False

        def get_transform(self):
            raise RuntimeError("trying to operate on a destroyed actor")

    world = _World(
        actors=[_Vehicle(100), _DeadVehicle(101)], junction_ids=()
    )
    adapter = _adapter(world)
    observation = adapter.reset()
    labels = [
        adapter.entity_labels[eid] for eid in observation.perceived_entities
    ]
    assert labels == ["vehicle_audi_a2_100"]

    class _FlakyVehicle(_Vehicle):
        def get_transform(self):
            raise RuntimeError("timeout waiting for simulator")

    flaky_world = _World(actors=[_FlakyVehicle(102)], junction_ids=())
    with pytest.raises(RuntimeError):
        _adapter(flaky_world).reset()


def test_region_metadata_distinguishes_derived_crosswalk_extents():
    # A crosswalk's extents are a derived world-axis AABB proxy; a
    # junction's are native box geometry. Discovery must let a spec
    # author tell them apart (approximation-labeling invariant).
    world = _World(
        actors=[],
        junction_ids=(23,),
        crosswalks=_closed_square(0, 0, 2),
    )
    adapter = _adapter(world)
    adapter.reset()
    frames = {}
    for eid, label in adapter.entity_labels.items():
        meta = adapter.property_metadata(eid)
        frames[label.split("_")[0]] = meta["extent_x"].frame
    assert "derived" in frames["crosswalk"]
    assert "AABB" in frames["crosswalk"]
    assert "derived" not in frames["junction"]


def test_camera_provider_is_lazy_and_invoked_once():
    calls = []

    def provider():
        calls.append(1)
        sensor = _FakeSensor()
        return [("junction_1", sensor)]

    adapter = _adapter(_World())
    adapter.set_camera_provider(provider)
    assert calls == []                      # nothing until first use
    assert adapter.camera_names() == ("junction_1",)
    adapter.camera_names()
    adapter.capture_frame()
    assert calls == [1]                     # invoked exactly once


def test_camera_provider_failure_degrades_to_no_cameras():
    def provider():
        raise RuntimeError("server went away")

    adapter = _adapter(_World())
    adapter.set_camera_provider(provider)
    assert adapter.camera_names() == ()
    assert adapter.capture_frame() is None


def test_unadoptable_provider_sensor_is_destroyed_others_kept():
    class _ListeningSensor(_FakeSensor):
        def __init__(self):
            super().__init__()
            self.destroyed = False

        def is_listening(self):
            return True

        def destroy(self):
            self.destroyed = True

    good = _FakeSensor()
    bad = _ListeningSensor()
    adapter = _adapter(_World())
    adapter.set_camera_provider(lambda: [("a", good), ("b", bad)])
    assert adapter.camera_names() == ("a",)
    assert bad.destroyed


def test_capabilities_declaration_matches_extraction():
    # The declared schemas are what specs compile against; drift between
    # declaration and extraction would validate specs the adapter can
    # never satisfy (or reject ones it serves).
    world = _World(
        actors=[_Vehicle(100), _Walker(101), _TrafficLight(23)],
        junction_ids=(23,),
        crosswalks=_closed_square(0, 0, 2),
    )
    adapter = _adapter(world)
    observation = adapter.reset()
    capabilities = adapter.capabilities
    for eid, entity in observation.perceived_entities.items():
        declared = capabilities.components_by_class[entity.entity_class]
        assert set(entity.properties) <= declared
        groups = capabilities.component_groups_by_class[entity.entity_class]
        meta = adapter.property_metadata(eid)
        for key in entity.properties:
            assert groups[key] == meta[key].group


def test_scope_none_preserves_full_output_and_native_reads():
    world_a, untouched, actors_a = _scope_subject()
    world_b, explicit, actors_b = _scope_subject()
    untouched.reset()
    explicit.reset()
    for actor in actors_a + actors_b:
        actor.calls.clear()
    explicit.set_extraction_scope(None)
    full = untouched.get_observation()
    same = explicit.get_observation()
    _assert_exact_projection(full, same, untouched, explicit, None)
    assert [actor.calls for actor in actors_a] == [actor.calls for actor in actors_b]
    vector_members = [count for actor in actors_a for path, count in actor.calls.items()
                      if ".location." in path or ".get_velocity." in path]
    assert set(vector_members) == {1}
    counts = sorted(len(entity.properties) for entity in full.perceived_entities.values())
    assert counts == [6, 9, 18, 23, 26]


def test_scope_is_exact_projection_and_prunes_native_reads():
    for scope, extra_reads in [
        (
            {
                "vehicle": frozenset({"speed_limit"}),
                "walker": frozenset({"control_speed"}),
                "traffic_light": frozenset({"state"}),
                "region": frozenset({"extent_x"}),
            },
            (
                Counter({"vehicle.get_speed_limit": 1}),
                Counter({
                    "walker.get_control": 1,
                    "walker.get_control.speed": 1,
                }),
                Counter({"light.state": 1}),
            ),
        ),
        ({}, (Counter(), Counter(), Counter())),
    ]:
        world, full_adapter, actors = _scope_subject()
        scoped_adapter = CarlaAdapter(world)
        full_adapter.reset()
        scoped_adapter.reset()
        full = full_adapter.get_observation()
        for actor in actors:
            actor.calls.clear()
        scoped_adapter.set_extraction_scope(scope)
        scoped = scoped_adapter.get_observation()
        _assert_exact_projection(full, scoped, full_adapter, scoped_adapter, scope)
        expected = (
            _structural_reads("vehicle", velocity=True) + extra_reads[0],
            _structural_reads("walker", velocity=True) + extra_reads[1],
            _structural_reads("light", velocity=False) + extra_reads[2],
        )
        assert tuple(actor.calls for actor in actors) == expected


def _check_scope_applies_to_dynamic_arrival():
    world = _World(actors=(), junction_ids=())
    adapter = CarlaAdapter(world)
    adapter.reset()
    adapter.set_extraction_scope({"vehicle": frozenset({"speed_limit"})})
    vehicle = _counted(_Vehicle(200), prefix="vehicle")
    world.actors.append(vehicle)
    [entity] = adapter.get_observation().perceived_entities.values()
    assert entity.properties == {"speed_limit": 30.0}
    assert entity.position.tolist() == [1.0, 2.0, 0.1]
    assert entity.velocity.tolist() == [3.0, 0.0, 0.0]
    assert vehicle.calls == _structural_reads(
        "vehicle", velocity=True
    ) + Counter({"vehicle.get_speed_limit": 1})


def test_scope_change_and_reset_rebuild_only_requested_properties():
    _check_scope_applies_to_dynamic_arrival()
    vehicle = _counted(_Vehicle(100), prefix="vehicle")
    world = _World(actors=[vehicle], junction_ids=())
    adapter = CarlaAdapter(world)
    adapter.reset()
    adapter.set_extraction_scope({"vehicle": frozenset({"control_throttle"})})
    vehicle.calls.clear()
    [entity] = adapter.get_observation().perceived_entities.values()
    assert entity.properties == {"control_throttle": 0.3}
    assert vehicle.calls == (
        _structural_reads("vehicle", velocity=True)
        + Counter({
            "vehicle.get_control": 1,
            "vehicle.get_control.throttle": 1,
        })
    )
    adapter.set_extraction_scope({"vehicle": frozenset({"speed_limit"})})
    vehicle.calls.clear()
    changed = adapter.reset()
    [entity] = changed.perceived_entities.values()
    assert entity.properties == {"speed_limit": 30.0}
    assert entity.track_age == 0
    assert vehicle.calls == _structural_reads(
        "vehicle", velocity=True
    ) + Counter({"vehicle.get_speed_limit": 1})


def test_scope_skips_unrequested_failure_but_propagates_requested_failure():
    class _OptionalFailure(_Vehicle):
        def get_acceleration(self):
            raise RuntimeError("acceleration transport failed")
    world = _World(actors=(), junction_ids=())
    adapter = CarlaAdapter(world)
    adapter.reset()
    world.actors.append(_OptionalFailure(100))
    adapter.set_extraction_scope({"vehicle": frozenset({"speed_limit"})})
    [entity] = adapter.get_observation().perceived_entities.values()
    assert entity.properties == {"speed_limit": 30.0}
    adapter.set_extraction_scope({"vehicle": frozenset({"acceleration_x"})})
    with pytest.raises(RuntimeError, match="acceleration transport failed"):
        adapter.get_observation()


@pytest.mark.parametrize("key,failing,value,path", [
    ("trigger_offset_x", "extent", -7.0, "location.x"),
    ("trigger_extent_y", "location", 0.5, "extent.y"),
])
def test_scope_reads_only_requested_trigger_sibling(key, failing, value, path):
    class _Trigger(_Box):
        def __getattribute__(self, name):
            if name == failing:
                raise RuntimeError(f"unrequested {name} read")
            return super().__getattribute__(name)
    light = _TrafficLight(23)
    light.trigger_volume = _Trigger(location=_Vec(-7.0, 38.4, 1.0))
    light = _counted(light, prefix="light")
    adapter = CarlaAdapter(_World(actors=[light], junction_ids=()))
    adapter.set_extraction_scope({"traffic_light": frozenset({key})})
    [entity] = adapter.reset().perceived_entities.values()
    assert entity.properties == {key: value}
    assert light.calls == _structural_reads("light", velocity=False) + Counter({
        "light.trigger_volume": 1,
        f"light.trigger_volume.{path.rsplit('.', 1)[0]}": 1,
        f"light.trigger_volume.{path}": 1,
    })


# ---------------------------------------------------------------------------
# Attached camera
# ---------------------------------------------------------------------------


class _FakeSensor:
    def __init__(self):
        self.callback = None
        self.is_alive = True

    def listen(self, callback):
        self.callback = callback


class _FakeImage:
    def __init__(self, width, height, raw):
        self.width = width
        self.height = height
        self.raw_data = raw


def test_no_camera_means_no_names_and_no_frames():
    adapter = _adapter(_World())
    assert adapter.camera_names() == ()
    assert adapter.capture_frame() is None


def test_attached_camera_converts_bgra_to_rgb():
    adapter = _adapter(_World())
    sensor = _FakeSensor()
    adapter.attach_camera(sensor)
    assert adapter.camera_names() == ("birdseye",)
    assert adapter.capture_frame() is None  # nothing delivered yet

    # One 1x2 image: pixels BGRA (1,2,3,4) and (5,6,7,8).
    sensor.callback(_FakeImage(2, 1, bytes([1, 2, 3, 4, 5, 6, 7, 8])))
    frame = adapter.capture_frame()
    assert frame.shape == (1, 2, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [3, 2, 1]
    assert frame[0, 1].tolist() == [7, 6, 5]


def test_unknown_camera_name_returns_none():
    adapter = _adapter(_World())
    sensor = _FakeSensor()
    adapter.attach_camera(sensor, name="birdseye")
    sensor.callback(_FakeImage(1, 1, bytes([9, 9, 9, 255])))
    assert adapter.capture_frame(camera="birdseye") is not None
    assert adapter.capture_frame(camera="chase") is None


def test_multiple_cameras_dispatch_by_name_with_first_as_default():
    adapter = _adapter(_World())
    first = _FakeSensor()
    second = _FakeSensor()
    adapter.attach_camera(first, name="junction_189")
    adapter.attach_camera(second, name="junction_23")
    assert adapter.camera_names() == ("junction_189", "junction_23")

    first.callback(_FakeImage(1, 1, bytes([1, 1, 1, 255])))
    second.callback(_FakeImage(1, 1, bytes([2, 2, 2, 255])))
    assert adapter.capture_frame(camera="junction_23")[0, 0].tolist() == [
        2, 2, 2,
    ]
    # None selects the adapter default: the first camera attached.
    assert adapter.capture_frame()[0, 0].tolist() == [1, 1, 1]


def test_only_newest_frame_is_kept():
    adapter = _adapter(_World())
    sensor = _FakeSensor()
    adapter.attach_camera(sensor)
    sensor.callback(_FakeImage(1, 1, bytes([1, 1, 1, 255])))
    sensor.callback(_FakeImage(1, 1, bytes([2, 2, 2, 255])))
    assert adapter.capture_frame()[0, 0].tolist() == [2, 2, 2]


def test_attach_refuses_listening_sensor_and_stops_replaced_one():
    class _ListeningSensor(_FakeSensor):
        def is_listening(self):
            return True

    adapter = _adapter(_World())
    with pytest.raises(ValueError):
        adapter.attach_camera(_ListeningSensor())

    class _StoppableSensor(_FakeSensor):
        def __init__(self):
            super().__init__()
            self.stopped = False

        def stop(self):
            self.stopped = True

    first = _StoppableSensor()
    second = _FakeSensor()
    adapter.attach_camera(first, name="birdseye")
    adapter.attach_camera(second, name="birdseye")  # same-name replacement
    assert first.stopped
    assert adapter.camera_names() == ("birdseye",)


def test_same_tick_external_destroy_with_stale_alive_flag_is_absent():
    # is_alive is client-cached: an actor destroyed by another client
    # the same tick can raise on reads while still reading alive. The
    # server-authoritative recheck treats it as the absence it is.
    class _StaleDeadVehicle(_Vehicle):
        def get_transform(self):
            raise RuntimeError("trying to operate on a destroyed actor")

    stale = _StaleDeadVehicle(103)
    world = _World(actors=[_Vehicle(100)], junction_ids=())

    class _RacyWorld(_World):
        def get_actors(self):
            return [stale] + list(self.actors)  # registry still lists it

        def get_actor(self, actor_id):
            if actor_id == stale.id:
                return None  # the server already knows it is gone
            return super().get_actor(actor_id)

    racy = _RacyWorld(actors=[_Vehicle(100)], junction_ids=())
    adapter = _adapter(racy)
    observation = adapter.reset()
    labels = [
        adapter.entity_labels[eid] for eid in observation.perceived_entities
    ]
    assert labels == ["vehicle_audi_a2_100"]
