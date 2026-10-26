"""
RoboDojo 3D-panel hooks and sampled pair distances, against a real
in-memory USD stage. Needs ``pxr`` and ``warp`` (both ship with any
Isaac install); skipped elsewhere.
"""

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
pytest.importorskip("warp")

from pxr import Usd, UsdGeom  # noqa: E402

from specola.adapters.robodojo import RoboDojoAdapter  # noqa: E402
from test_robodojo_adapter import (  # noqa: E402
    _Env,
    _PoseOnlyObject,
    _RigidObject,
    _Robot,
    _RobotAsset,
    _RobotManager,
    _SceneManager,
)

_CUBE_POINTS = [
    (x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
]
_CUBE_FACES = [
    [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
    [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
]


def _cube(stage, path, purpose=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(_CUBE_POINTS)
    mesh.GetFaceVertexCountsAttr().Set([4] * len(_CUBE_FACES))
    mesh.GetFaceVertexIndicesAttr().Set([i for face in _CUBE_FACES for i in face])
    if purpose:
        mesh.GetPurposeAttr().Set(purpose)
    return mesh


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for link in ("base_link", "link1", "gripper_link"):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_0/robot0/{link}")
        _cube(stage, f"/World/envs/env_0/robot0/{link}/collision", purpose="guide")
    for name in ("bowl_0", "cup_0"):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_0/objects/{name}")
        _cube(stage, f"/World/envs/env_0/objects/{name}/visual")
    _cube(stage, "/World/background/floor")
    return stage


class _PrimObject(_RigidObject):
    def __init__(self, prim_path, position):
        super().__init__(position, [1.0, 0.0, 0.0, 0.0], [0.0] * 3, [0.0] * 3)
        self._prim_path = prim_path

    def move(self, position):
        self._pose = (np.asarray(position, dtype=float), self._pose[1])


def _env(stage):
    asset = _RobotAsset(
        1, ["joint1", "joint2", "finger_joint1"], ["base_link", "link1", "gripper_link"],
        dof_types=["Rotation", "Rotation", "Translation"],
    )
    asset.root_physx_view.link_paths = [
        [f"/World/envs/env_0/robot0/{link}" for link in asset.body_names]
    ]
    asset.data.body_pos_w = np.array([[[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]]], dtype=float)
    asset.data.body_quat_w = np.tile([1.0, 0.0, 0.0, 0.0], (1, 3, 1))
    robot = _Robot("left_arm", ["joint1", "joint2"], ["finger_joint1"], "gripper_link", "base_link")
    bowl = _PrimObject("/World/envs/env_0/objects/bowl_0", [1.5, 0.0, 0.0])
    cup = _PrimObject("/World/envs/env_0/objects/cup_0", [1.5, 3.0, 0.0])
    scene = _SceneManager(
        {
            "rigid": {"bowl_0": bowl, "cup_0": cup},
            "geometry": {"marker_0": _PoseOnlyObject([0, 0, 5, 1, 0, 0, 0])},
        }
    )
    env = _Env(num_envs=1, robot_manager=_RobotManager([robot], [asset]), scene_manager=scene)
    env.stage = stage
    return env, bowl


def test_scene_geometry_rows_and_meshes():
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env)
    adapter.reset()
    geometry = adapter.scene_geometry()
    names = [body["name"] for body in geometry["json"]["bodies"]]
    assert names == [
        "world", "robot:left_arm/base_link", "robot:left_arm/link1",
        "robot:left_arm/gripper_link", "bowl_0", "cup_0", "marker_0", "background",
    ]
    assert geometry["json"]["ngeom"] == 6
    assert geometry["json"]["skipped"] == {"unresolved_body": 1}
    groups = {geom["body"]: geom["group"] for geom in geometry["json"]["geoms"]}
    assert groups[1] == 0 and groups[4] == 1 and groups[7] == 1
    xpos, xquat = adapter.scene_poses()
    assert xpos.shape == (8, 3) and xquat.shape == (8, 4)
    assert np.allclose(xpos[4], [1.5, 0.0, 0.0])
    assert np.allclose(xpos[3], [0.0, 0.0, 1.0])
    assert np.allclose(xquat[7], [1.0, 0.0, 0.0, 0.0])


def test_link_poses_are_the_published_body_poses():
    """RoboDojo's Isaac Lab (0.54.3) stores body_link_pose_w as position
    then a (w, x, y, z) quaternion. The panel and the pair distances pose a
    link with the quaternion the adapter publishes as body_quat_w_*, not a
    reordering of it (live stack_bowls read (z, w, x, y), 2026-10-08)."""
    env, _ = _env(_stage())
    data = env.robot_manager.robot_key[0].data
    yaw = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # 90 degrees about z
    data.body_quat_w = np.tile(yaw, (1, 3, 1))
    data.body_link_pose_w = np.concatenate((data.body_pos_w, data.body_quat_w), axis=-1)
    adapter = RoboDojoAdapter(env)
    observation = adapter.reset()
    _, xquat = adapter.scene_poses()
    assert np.allclose(xquat[1:4], yaw)
    robot = next(e for e in observation.perceived_entities.values() if e.entity_class == "robot")
    published = [robot.properties[f"body_quat_w_link1_{c}"] for c in "wxyz"]
    assert np.allclose(xquat[2], published)


def test_scene_hooks_none_without_stage():
    env, _ = _env(_stage())
    env.stage = None
    adapter = RoboDojoAdapter(env)
    adapter.reset()
    assert adapter.scene_geometry() is None
    assert adapter.scene_poses() is None


def _pairs(observation):
    return {
        observation.perceived_entities[eid].properties["mesh_distance_sampled"]: eid
        for eid, entity in observation.perceived_entities.items()
        if entity.entity_class == "mesh_pair"
    }


def test_pair_distances_follow_poses():
    env, bowl = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=1.0, distance_device="cpu")
    assert "mesh_pair" in adapter.supported_classes
    obs = adapter.reset()
    pairs = {
        adapter.entity_labels[eid]: entity
        for eid, entity in obs.perceived_entities.items()
        if entity.entity_class == "mesh_pair"
    }
    assert set(pairs) == {"left_arm|bowl_0", "left_arm|cup_0", "bowl_0|cup_0"}
    robot_bowl = pairs["left_arm|bowl_0"]
    assert robot_bowl.properties["mesh_distance_sampled"] == pytest.approx(0.5)
    assert robot_bowl.properties["mesh_distance_clipped"] == 0.0
    assert pairs["bowl_0|cup_0"].properties == {
        "mesh_distance_sampled": 1.0, "mesh_distance_clipped": 1.0,
    }
    meta = adapter.property_metadata(robot_bowl.entity_id)
    assert set(meta) == set(robot_bowl.properties)
    assert "robot" in meta["mesh_distance_sampled"].tags
    assert "environment" in adapter.property_metadata(pairs["bowl_0|cup_0"].entity_id)["mesh_distance_sampled"].tags

    bowl.move([1.2, 0.0, 0.0])
    obs = adapter.get_observation()
    moved = obs.perceived_entities[robot_bowl.entity_id]
    assert moved.properties["mesh_distance_sampled"] == pytest.approx(0.2)
    bowl.move([0.8, 0.0, 0.0])
    obs = adapter.get_observation()
    assert obs.perceived_entities[robot_bowl.entity_id].properties["mesh_distance_sampled"] == 0.0


def test_no_pairs_without_distmax(monkeypatch):
    monkeypatch.delenv("ROBODOJO_DISTMAX", raising=False)
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env)
    obs = adapter.reset()
    assert "mesh_pair" not in adapter.supported_classes
    assert not any(e.entity_class == "mesh_pair" for e in obs.perceived_entities.values())


def test_distmax_from_environment(monkeypatch):
    monkeypatch.setenv("ROBODOJO_DISTMAX", "0.25")
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    assert adapter.distmax == 0.25
    with pytest.raises(ValueError):
        RoboDojoAdapter(env, distmax=0.0)


def _queries(populations=(), instances=()):
    from specola.query_config import QueryConfiguration

    return QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": list(populations),
        "instances": list(instances),
    })


def test_far_pairs_are_not_queried():
    """A pair whose bounding spheres cannot meet within distmax reports the
    cap, clipped, without a mesh query; a near pair is still queried."""
    # Unit cubes (bounding radius 0.866): links at the origin column, bowl
    # moved 5 m out, cup at (1.5, 3). With a 2 m cutoff the sphere test
    # (0.866 + 0.866 + 2 = 3.73) admits only the links against the cup
    # (centres 3.35 to 3.5 apart) and rejects every pair with the bowl.
    env, bowl = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=2.0, distance_device="cpu")
    adapter.reset()
    bowl.move([5.0, 0.0, 0.0])
    queried = []
    real = adapter._distance.distance

    def counting(a, b, cutoff):
        queried.append((a, b))
        return real(a, b, cutoff)

    adapter._distance.distance = counting
    obs = adapter.get_observation()
    pairs = {adapter.entity_labels[eid]: e for eid, e in obs.perceived_entities.items()
             if e.entity_class == "mesh_pair"}
    assert pairs["left_arm|bowl_0"].properties == {"mesh_distance_sampled": 2.0, "mesh_distance_clipped": 1.0}
    assert pairs["bowl_0|cup_0"].properties == {"mesh_distance_sampled": 2.0, "mesh_distance_clipped": 1.0}
    # Exactly the three link rows against the cup's row were queried.
    assert len(queried) == 3
    assert len({a for a, _ in queried}) == 3 and len({b for _, b in queried}) == 1


def test_query_configuration_names_the_pairs_and_links():
    """The robodojo.mesh_pair population supplies the cutoff and the robot
    links to sample; its instances name the pairs to query by label."""
    env, bowl = _env(_stage())
    # Bowl cube at (1.5, 0, 2): its faces are 0.5 from gripper_link's
    # (z 1.0), 0.707 from link1's corner (z 0.5) and 1.118 from base_link's,
    # so a wrong link choice changes the reported distance.
    bowl.move([1.5, 0.0, 2.0])
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair",
                      "arguments": {"distmax": 1.5, "robot_links": ["gripper_link"]}}],
        instances=[{"binding": "robodojo.mesh_pair", "id": "grip_bowl", "arguments": {},
                    "operands": {"a": {"kind": "entity", "name": "left_arm"},
                                 "b": {"kind": "entity", "name": "bowl_0"}}}],
    ))
    assert "mesh_pair" in adapter.supported_classes
    obs = adapter.reset()
    pairs = {adapter.entity_labels[eid]: e for eid, e in obs.perceived_entities.items()
             if e.entity_class == "mesh_pair"}
    assert set(pairs) == {"left_arm|bowl_0"}
    plan = adapter._ensure_scene_plan()
    rows = plan["rows_of"]["robot:left_arm"]
    assert [plan["bodies"][row]["name"] for row in rows] == ["robot:left_arm/gripper_link"]
    assert pairs["left_arm|bowl_0"].properties["mesh_distance_sampled"] == pytest.approx(0.5)


def test_only_rows_of_surviving_pairs_are_posed():
    """Meshes no selected pair uses are never transformed or refitted, and
    a row shared by several surviving pairs is posed once per step."""
    env, bowl = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=1.0, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": 1.0}}],
        instances=[{"binding": "robodojo.mesh_pair", "id": "grip_bowl", "arguments": {},
                    "operands": {"a": {"kind": "entity", "name": "left_arm"},
                                 "b": {"kind": "entity", "name": "bowl_0"}}}],
    ))
    adapter.reset()
    plan = adapter._ensure_scene_plan()
    updated = []
    real = adapter._distance.update

    def counting(key, position, quat):
        updated.append(key)
        return real(key, position, quat)

    adapter._distance.update = counting
    adapter.get_observation()
    cup_row = plan["rows_of"]["object:cup_0"][0]
    assert str(cup_row) not in updated
    assert len(updated) == len(set(updated))
    # All three links are within reach of the bowl, so the bowl row and
    # each link row were posed exactly once.
    assert set(updated) == {str(r) for r in plan["rows_of"]["robot:left_arm"] + plan["rows_of"]["object:bowl_0"]}
    # Move the bowl out of reach: nothing is posed at all.
    updated.clear()
    bowl.move([5.0, 0.0, 0.0])
    adapter.get_observation()
    assert updated == []


def test_skipped_pair_matches_a_queried_far_pair_at_any_cutoff():
    """A cutoff float32 rounds down (0.03) must not turn a far pair into a
    hit; the prefiltered and the queried answer are byte-identical."""
    env, bowl = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=0.03, distance_device="cpu")
    adapter.reset()
    bowl.move([5.0, 0.0, 0.0])

    def far(observation):
        eid = next(eid for eid, e in observation.perceived_entities.items()
                   if adapter.entity_labels[eid] == "left_arm|bowl_0")
        return observation.perceived_entities[eid].properties

    filtered = far(adapter.get_observation())
    plan = adapter._ensure_scene_plan()
    plan["radius"] = {row: float("inf") for row in plan["radius"]}
    queried = far(adapter.get_observation())
    assert filtered == queried == {"mesh_distance_sampled": 0.03, "mesh_distance_clipped": 1.0}


def test_prefilter_is_conservative_under_float32_world_rounding():
    """Two slivers whose spheres just miss the cutoff in exact arithmetic,
    63 m from the origin: float32 world vertices round the sampled gap
    inside the cutoff, and the prefilter must not skip what the query
    would report (case found by external review, 2026-09-23)."""
    r1, r2 = 0.1647028923034668, 0.02746402844786644
    p1, p2 = 63.0, 63.24216842651367
    a = np.array([[r1, 0, 0], [r1 - 1e-4, 1e-4, 0], [r1 - 1e-4, 0, 1e-4]], np.float32)
    b = np.array([[-r2, 0, 0], [-r2 + 1e-4, 1e-4, 0], [-r2 + 1e-4, 0, 1e-4]], np.float32)
    stage = _stage()
    for path, vertices in (("/World/envs/env_0/robot0/gripper_link/collision", a),
                           ("/World/envs/env_0/objects/bowl_0/visual", b)):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        mesh.GetPointsAttr().Set([tuple(map(float, v)) for v in vertices])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    env, bowl = _env(stage)
    asset = env.robot_manager.robot_key[0]
    asset.data.body_pos_w[0, 2] = [p1, 0, 0]
    bowl.move([p2, 0, 0])
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair",
                      "arguments": {"distmax": 0.05, "robot_links": ["gripper_link"]}}],
        instances=[{"binding": "robodojo.mesh_pair", "id": "edge", "arguments": {},
                    "operands": {"a": {"kind": "entity", "name": "left_arm"},
                                 "b": {"kind": "entity", "name": "bowl_0"}}}],
    ))

    def only(observation):
        return next(e.properties for e in observation.perceived_entities.values()
                    if e.entity_class == "mesh_pair")

    filtered = only(adapter.reset())
    plan = adapter._ensure_scene_plan()
    plan["radius"] = {row: float("inf") for row in plan["radius"]}
    assert filtered == only(adapter.get_observation())
    assert filtered["mesh_distance_clipped"] == 0.0


@pytest.mark.parametrize("arguments, operands, message", [
    ({"distmax": 1.0, "robot_links": "gripper_link"}, None, "list of link names"),
    ({"distmax": 1.0, "robot_links": ["typo"]}, None, "no robot has"),
    ({"distmax": 1.0}, {"a": {"kind": "entity", "name": "left_arm"},
                        "b": {"kind": "entity", "name": "typo"}}, "no mesh on both sides"),
    ({"distmax": 1.0}, {"a": {"kind": "entity", "name": "left_arm"},
                        "b": {"kind": "entity", "name": "marker_0"}}, "no mesh on both sides"),
    ({"distmax": 1.0}, {"a": {"kind": "geom", "name": "left_arm"},
                        "b": {"kind": "entity", "name": "bowl_0"}}, "kind entity"),
    ({"distmax": 1.0, "robot_links": {"gripper_link": True}}, None, "list of link names"),
    ({"distmax": 1.0, "robot_links": 42}, None, "list of link names"),
    ({"distmax": 1.0, "robot_link": ["gripper_link"]}, None, "not \\['robot_link'\\]"),
    ({"distmax": 1.0}, {"a": {"kind": "entity", "name": "left_arm"},
                        "b": {"kind": "entity", "name": "bowl_0"},
                        "c": {"kind": "entity", "name": "cup_0"}}, "needs operands a and b"),
])
def test_malformed_pair_configuration_raises_at_reset(arguments, operands, message):
    """Every field the binding does not define, and every value of the
    wrong shape, is refused before a mesh is registered (a misspelt
    robot_links would otherwise silently restore the all-link cost)."""
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    instances = [] if operands is None else [
        {"binding": "robodojo.mesh_pair", "id": "bad", "arguments": {}, "operands": operands}]
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": arguments}],
        instances=instances))
    with pytest.raises(ValueError, match=message):
        adapter.reset()


@pytest.mark.parametrize("arguments, operands, message", [
    ({"distmx": 0.05}, None, "not \\['distmx'\\]"),
    ({"robot_links": 42}, None, "list of link names"),
    ({"robot_links": ["gripper_link"]}, None, "no cutoff is set"),
    ({}, {"a": {"kind": "entity", "name": "left_arm"}}, "needs operands a and b"),
    ({"distmax": True}, None, "finite positive number"),
    ({"distmax": "0.05"}, None, "finite positive number"),
    ({"distmax": [0.05]}, None, "finite positive number"),
    ({"distmax": {"m": 0.05}}, None, "finite positive number"),
    ({"distmax": 0}, None, "finite positive number"),
])
def test_pair_configuration_is_validated_before_activation(monkeypatch, arguments, operands, message):
    """With no constructor or environment cutoff, naming the binding at all
    validates it: a misspelt distmax must not silently mean "no pairs"
    (case found by external review, 2026-09-23)."""
    monkeypatch.delenv("ROBODOJO_DISTMAX", raising=False)
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    instances = [] if operands is None else [
        {"binding": "robodojo.mesh_pair", "id": "bad", "arguments": {}, "operands": operands}]
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": arguments}],
        instances=instances))
    with pytest.raises(ValueError, match=message):
        adapter.reset()


def test_pair_configuration_is_parsed_once_per_epoch():
    """The binding is parsed when the configuration or the fallback cutoff
    changes, not on every lookup or reset; an integer cutoff is accepted,
    a null one falls back."""
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=0.5, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": 1}}]))
    calls = []
    for name in ("_parse_population", "_parse_configured_pairs"):
        real = getattr(adapter, name)

        def counting(real=real, name=name):
            calls.append(name)
            return real()

        setattr(adapter, name, counting)
    adapter.reset()
    adapter.get_observation()
    adapter.get_observation()
    adapter.reset()
    assert adapter._active_distmax() == 1.0
    assert sorted(calls) == ["_parse_configured_pairs", "_parse_population"]
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": None}}]))
    assert adapter._active_distmax() == 0.5 and len(calls) == 4


def test_fallback_cutoff_change_is_seen_after_reset():
    """`distmax` is a public field; changing it between resets changes the
    cutoff in force, the pair class and the reported values (case found
    by external review, 2026-09-23)."""
    env, bowl = _env(_stage())
    adapter = RoboDojoAdapter(env, distmax=0.5, distance_device="cpu")
    adapter.reset()
    bowl.move([1.2, 0.0, 0.0])          # 0.2 from the links' cubes

    def robot_bowl(observation):
        eid = next(eid for eid, e in observation.perceived_entities.items()
                   if adapter.entity_labels[eid] == "left_arm|bowl_0")
        return observation.perceived_entities[eid].properties["mesh_distance_sampled"]

    adapter.distmax = 0.1
    assert adapter._active_distmax() == 0.1
    assert robot_bowl(adapter.reset()) == 0.1
    adapter.distmax = None
    assert "mesh_pair" not in adapter.supported_classes
    assert not any(e.entity_class == "mesh_pair" for e in adapter.reset().perceived_entities.values())
    adapter.distmax = 2.0
    assert robot_bowl(adapter.reset()) == pytest.approx(0.2)


def test_pair_instance_takes_no_arguments():
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": 1.0}}],
        instances=[{"binding": "robodojo.mesh_pair", "id": "bad", "arguments": {"ignored": True},
                    "operands": {"a": {"kind": "entity", "name": "left_arm"},
                                 "b": {"kind": "entity", "name": "bowl_0"}}}],
    ))
    with pytest.raises(ValueError, match="takes no arguments"):
        adapter.reset()


def test_distance_at_the_float32_cutoff_is_a_hit():
    """A sampled distance strictly below the Python cutoff is reported as
    such even when it equals the float32 rounding of that cutoff; only a
    miss reports the cap (case found by external review, 2026-09-23)."""
    from specola.adapters._mesh_distance import SampledMeshDistance

    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    cutoff = 0.03
    below = float(np.float32(cutoff))            # 0.029999999329..., below 0.03
    assert below < cutoff
    for gap, expect in ((below, (below, False)),
                        (float(np.nextafter(np.float32(cutoff), np.float32(0))), None),
                        (0.5, (cutoff, True))):
        query = SampledMeshDistance(device="cpu")
        query.add("a", tri, faces)
        query.add("b", tri, faces)
        query.update("a", np.zeros(3), np.array([1.0, 0, 0, 0]))
        query.update("b", np.array([0, 0, gap]), np.array([1.0, 0, 0, 0]))
        value, capped = query.distance("a", "b", cutoff)
        if expect is None:
            assert not capped and value == pytest.approx(gap)
        else:
            assert (value, capped) == expect
    # An exactly representable cutoff and one float32 rounds up behave alike.
    for cutoff in (0.5, 0.1):
        query = SampledMeshDistance(device="cpu")
        query.add("a", tri, faces)
        query.add("b", tri, faces)
        query.update("a", np.zeros(3), np.array([1.0, 0, 0, 0]))
        query.update("b", np.array([0, 0, 2.0]), np.array([1.0, 0, 0, 0]))
        assert query.distance("a", "b", cutoff) == (cutoff, True)


_ARM_BOWL = {"binding": "robodojo.mesh_pair", "id": "arm_bowl", "arguments": {},
             "operands": {"a": {"kind": "entity", "name": "left_arm"},
                          "b": {"kind": "entity", "name": "bowl_0"}}}
_ALL_PAIRS = {"left_arm|bowl_0", "left_arm|cup_0", "bowl_0|cup_0"}


@pytest.mark.parametrize("arguments, instances, formula, expected, queries", [
    # Issue #136: the specification decides which formed pairs are read.
    ({}, (), "mesh_pair_*[mesh_distance_sampled @ environment] >= 0.0", {"bowl_0|cup_0"}, 1),
    ({}, (), "robot_*[joint_pos_joint1] <= 10.0", set(), 0),
    ({}, (), "mesh_pair_*[mesh_distance_sampled] >= 0.0", _ALL_PAIRS, 7),
    # population: all reads every formed pair whatever the specification binds.
    ({"population": "all"}, (), "mesh_pair_*[mesh_distance_sampled @ environment] >= 0.0", _ALL_PAIRS, 7),
    # Named pairs are the population, read whatever the specification binds.
    ({}, (_ARM_BOWL,), "robot_*[joint_pos_joint1] <= 10.0", {"left_arm|bowl_0"}, 3),
])
def test_a_session_reads_only_the_pairs_its_specs_can_bind(tmp_path, arguments, instances, formula,
                                                           expected, queries):
    """Reset reads every formed pair; after it a monitor session reads the
    formed pairs a compiled slot can bind and queries only their rows. With
    a 5 m cutoff no pair is prefiltered: 3 link rows each against the bowl
    and the cup, plus bowl against cup, make 7 row queries for every pair."""
    from specola.session import Monitor

    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": 5.0, **arguments}}],
        instances=instances,
    ))
    specs = tmp_path / "specs.yaml"
    specs.write_text(f'specs:\n- name: s\n  formula: "{formula}"\n')
    session = Monitor(str(specs), results_dir=None).watch(adapter)

    def pairs():
        observation = adapter.last_observation
        assert observation is not None
        return {adapter.entity_labels[eid] for eid, entity in observation.perceived_entities.items()
                if entity.entity_class == "mesh_pair"}

    assert pairs() == ({"left_arm|bowl_0"} if instances else _ALL_PAIRS)
    queried = []
    query = adapter._distance
    assert query is not None
    real = query.distance

    def counting(a, b, max_distance):
        queried.append((a, b))
        return real(a, b, max_distance)

    query.distance = counting
    session.step()
    assert pairs() == expected
    assert len(queried) == queries
    session.end()


def test_pair_instance_must_name_its_operands():
    env, _ = _env(_stage())
    adapter = RoboDojoAdapter(env, distance_device="cpu")
    adapter.set_query_configuration(_queries(
        populations=[{"binding": "robodojo.mesh_pair", "arguments": {"distmax": 1.0}}],
        instances=[{"binding": "robodojo.mesh_pair", "id": "by_id", "arguments": {},
                    "operands": {"a": {"kind": "entity", "id": 0},
                                 "b": {"kind": "entity", "name": "bowl_0"}}}],
    ))
    with pytest.raises(ValueError):
        adapter.reset()
