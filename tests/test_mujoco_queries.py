"""Native distance-reader fixtures, separate from adapter completeness.

The population is the engine's default pair filter with a bounding-sphere
prune per read. The native effect witnesses cover plane, sphere and box
colliders. The plugin, thread-pool, stack and allocation-callback refusals
are source-traced guards, not executed native negative controls here.
"""

import ctypes
from importlib import metadata

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.mujoco

from specola.adapters.mujoco_queries import GeomDistanceResult, IsolatedGeomDistanceReader
from specola.conformance.query_bindings import QueryBindingError, QueryConfigurationRequired
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration


# Geom IDs in declaration order: floor 0, visual 1, fixture_box 2,
# a_sphere 3, a_child_sphere 4, a_weld_sphere 5, a_grand_sphere 6, b_sphere 7.
_SCENE = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <geom name="visual" type="sphere" size=".1" pos="0 0 .5" contype="0" conaffinity="0"/>
    <body name="fixture" pos="1.5 0 .2"><geom name="fixture_box" type="box" size=".1 .1 .1"/></body>
    <body name="a" pos="0 0 .3">
      <freejoint/>
      <geom name="a_sphere" type="sphere" size=".1"/>
      <body name="a_child" pos="0 0 .3"><joint type="hinge"/><geom name="a_child_sphere" type="sphere" size=".05"/></body>
      <body name="a_weld" pos=".3 0 0">
        <geom name="a_weld_sphere" type="sphere" size=".05"/>
        <body name="a_grand" pos=".3 0 0"><joint type="hinge"/><geom name="a_grand_sphere" type="sphere" size=".05"/></body>
      </body>
    </body>
    <body name="b" pos="0 1 .3"><freejoint/><geom name="b_sphere" type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""
FLOOR, VISUAL, FIXTURE, A, A_CHILD, A_WELD, A_GRAND, B = range(8)
_MOVING = (A, A_CHILD, A_WELD, A_GRAND, B)
_UNIVERSE = ({(FLOOR, geom) for geom in _MOVING} | {(FIXTURE, geom) for geom in _MOVING}
             | {(A, B), (A_CHILD, B), (A_WELD, B), (A_GRAND, B), (A_CHILD, A_GRAND)})


@pytest.fixture
def native_scene():
    if mujoco.__version__ != "3.8.1":
        pytest.skip("distance reader effect custody for this producer row is unexecuted")
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def reader_factory(request):
    def make(model, data, configuration):
        reader = IsolatedGeomDistanceReader(model, data, configuration)
        request.addfinalizer(reader.close)
        return reader
    return make


def _configuration(cutoff=2.0, instances=()):
    return QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": cutoff}}],
        "instances": list(instances),
    })


def _result(reader, data, pair):
    return next(item for item in reader.read(data) if item.geom_ids == pair and not item.instance_token)


def _move_body(model, data, body_name, position):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    address = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[address:address + 3] = position
    mujoco.mj_forward(model, data)


def test_population_is_the_engine_pair_filter(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration())
    assert set(reader.population_pairs()) == _UNIVERSE
    # Visual geom: bitmask incompatible. Floor and fixture: one weld group.
    # a and a_child: parent and child. a and a_weld: one weld group. The
    # filter compares weld groups, so a_weld is filtered against a_child (its
    # group is a_child's parent) and a against a_grand (a is the parent group
    # of a_grand) although neither is a direct parent. a_child and a_grand
    # are cousins and remain. Both orders never appear.
    pairs = set(reader.population_pairs())
    assert not {pair for pair in pairs if VISUAL in pair}
    assert (FLOOR, FIXTURE) not in pairs
    assert (A, A_CHILD) not in pairs and (A, A_WELD) not in pairs
    assert (A_CHILD, A_WELD) not in pairs
    assert (A, A_GRAND) not in pairs and (A_WELD, A_GRAND) not in pairs
    assert (A_CHILD, A_GRAND) in pairs
    assert all(first < second for first, second in reader.population_pairs())
    assert {result.geom_ids for result in reader.read(data)} == _UNIVERSE


def test_population_matches_independent_numpy_and_dispatch_oracle(native_scene, reader_factory):
    model, data = native_scene
    distribution = metadata.distribution("mujoco")
    files = [item for item in distribution.files if item.parent.as_posix() == "mujoco" and item.match("libmujoco.so.*")]
    assert len(files) == 1
    library = ctypes.CDLL(str(distribution.locate_file(files[0])))
    dimension = int(mujoco.mjtGeom.mjNGEOMTYPES)
    table = (ctypes.c_void_p * (dimension * dimension)).in_dll(library, "mjCOLLISIONFUNC")
    expected = set()
    for first in range(model.ngeom):
        for second in range(first + 1, model.ngeom):
            body_first, body_second = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
            compatible = ((int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                          | (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))) != 0
            weld_first, weld_second = int(model.body_weldid[body_first]), int(model.body_weldid[body_second])
            parent_weld_first = int(model.body_weldid[model.body_parentid[weld_first]])
            parent_weld_second = int(model.body_weldid[model.body_parentid[weld_second]])
            distinct = body_first != body_second and weld_first != weld_second
            # filterBodyPair, engine_collision_driver.c at the reviewed commit.
            related = (weld_first != 0 and weld_second != 0
                       and (weld_first == parent_weld_second or weld_second == parent_weld_first))
            low, high = sorted((int(model.geom_type[first]), int(model.geom_type[second])))
            if compatible and distinct and not related and table[low * dimension + high] is not None:
                expected.add((first, second))
    assert set(reader_factory(model, data, _configuration()).population_pairs()) == expected


def test_far_pairs_are_pruned_without_a_native_call(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(0.5))
    pruned = _result(reader, data, (A, B))
    assert not pruned.executed
    assert pruned.outcome == "inferred_above_cutoff"
    assert pruned.native_return is None and pruned.fromto is None
    assert pruned.capped == 0.5 and pruned.clipped == 1
    assert pruned.lower_bound == pytest.approx(1.0 - 0.2)
    assert pruned.body_ids == (int(model.geom_bodyid[A]), int(model.geom_bodyid[B]))
    executed = _result(reader, data, (FLOOR, A))
    assert executed.executed and executed.outcome == "native_below_cutoff"
    assert executed.native_return == pytest.approx(0.2)
    assert executed.lower_bound is not None


def test_pruned_pairs_agree_with_direct_native_clamp(native_scene, reader_factory):
    model, data = native_scene
    cutoff = 0.5
    reader = reader_factory(model, data, _configuration(cutoff))
    results = reader.read(data)
    pruned = [item for item in results if not item.executed]
    assert pruned
    buffer = np.zeros(6)
    for item in pruned:
        first, second = item.geom_ids
        # The reviewed 3.8.1 build clamps a beyond-cutoff pair to the cutoff.
        assert mujoco.mj_geomDistance(model, data, first, second, cutoff, buffer) == cutoff
        centers = np.linalg.norm(np.asarray(data.geom_xpos[first]) - np.asarray(data.geom_xpos[second]))
        assert centers - model.geom_rbound[first] - model.geom_rbound[second] >= cutoff


def test_plane_pairs_are_never_pruned(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(0.5))
    result = _result(reader, data, (FLOOR, A_CHILD))
    assert result.executed
    assert result.native_return == result.capped == 0.5
    assert result.clipped == 1 and result.outcome == "native_cutoff"
    # A plane has no bounding sphere, so its bound is not a prune criterion.
    assert result.lower_bound >= 0.5


def test_moving_pair_transitions_between_pruned_and_executed(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(1.0))
    executed = _result(reader, data, (A, B))
    assert executed.executed and executed.native_return == pytest.approx(0.8)
    assert executed.capped == pytest.approx(0.8) and executed.clipped == 0
    _move_body(model, data, "b", (0.0, 1.5, 0.3))
    pruned = _result(reader, data, (A, B))
    assert not pruned.executed and pruned.capped == 1.0 and pruned.clipped == 1
    _move_body(model, data, "b", (0.0, 0.3, 0.3))
    touching = _result(reader, data, (A, B))
    assert touching.executed and touching.native_return == pytest.approx(0.1)
    assert touching.outcome == "native_below_cutoff"


def test_witnesses_keep_requested_operand_order(native_scene, reader_factory):
    model, data = native_scene
    instance = {"binding": "mujoco.mj_geomDistance", "id": "reversed", "arguments": {"distmax": 2.0},
                "operands": {"geom1": {"kind": "geom", "id": B}, "geom2": {"kind": "geom", "id": A}}}
    reader = reader_factory(model, data, _configuration(instances=(instance,)))
    forward = _result(reader, data, (A, B))
    reverse = next(item for item in reader.read(data) if item.instance_token == "reversed")
    np.testing.assert_allclose(forward.fromto.restore(), [0, 0.1, 0.3, 0, 0.9, 0.3])
    np.testing.assert_allclose(reverse.fromto.restore(), [0, 0.9, 0.3, 0, 0.1, 0.3])


def test_explicit_instances_are_not_pruned(native_scene, reader_factory):
    model, data = native_scene
    instance = {"binding": "mujoco.mj_geomDistance", "id": "inspection", "arguments": {"distmax": 0.5},
                "operands": {"geom2": {"kind": "geom", "name": "b_sphere"}, "geom1": {"kind": "geom", "id": A}}}
    reader = reader_factory(model, data, _configuration(0.5, instances=(instance,)))
    result = next(item for item in reader.read(data) if item.instance_token == "inspection")
    assert result.geom_ids == (A, B) and result.distmax == 0.5
    assert result.executed and result.outcome == "native_cutoff" and result.lower_bound is None


def test_population_all_is_validated_and_never_reaches_the_native_call(native_scene, reader_factory):
    model, data = native_scene
    configuration = QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": 0.5, "population": "all"}}],
        "instances": [],
    })
    reader = reader_factory(model, data, configuration)
    assert reader.population_all and reader._cutoffs == (0.5,)
    assert {item.geom_ids for item in reader.read(data)} == _UNIVERSE
    assert not reader_factory(model, data, _configuration(0.5)).population_all


def test_population_argument_is_refused_on_an_explicit_instance(native_scene, reader_factory):
    model, data = native_scene
    instance = {"binding": "mujoco.mj_geomDistance", "id": "inspection",
                "arguments": {"distmax": 0.5, "population": "all"},
                "operands": {"geom1": {"kind": "geom", "id": A}, "geom2": {"kind": "geom", "id": B}}}
    with pytest.raises(QueryBindingError):
        reader_factory(model, data, _configuration(0.5, instances=(instance,)))


def test_a_pair_subset_reads_only_those_pairs_with_their_own_bounds(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(0.5))
    full = {item.geom_ids: item for item in reader.read(data)}
    with reader._session.observe(data, completion=("mj_kinematics",)) as snapshot:
        subset = reader._read_snapshot(snapshot, (np.array([FLOOR, A]), np.array([A, B])))
    assert [item.geom_ids for item in subset] == [(FLOOR, A), (A, B)]
    for item in subset:
        assert item.outcome == full[item.geom_ids].outcome
        assert item.lower_bound == pytest.approx(full[item.geom_ids].lower_bound)


def test_missing_required_configuration_fails_before_scratch_allocation(native_scene, reader_factory):
    model, data = native_scene
    with pytest.raises(QueryConfigurationRequired):
        reader_factory(model, data, QueryConfiguration.empty())


@pytest.mark.parametrize("operand", ({"kind": "geom", "id": -1}, {"kind": "geom", "id": 99},
                                   {"kind": "geom", "name": "absent"}))
def test_absent_native_operands_fail_during_binding(native_scene, reader_factory, operand):
    model, data = native_scene
    instance = {"binding": "mujoco.mj_geomDistance", "id": "bad", "arguments": {"distmax": 1.0},
                "operands": {"geom1": operand, "geom2": {"kind": "geom", "id": 1}}}
    with pytest.raises(QueryBindingError, match="absent or out of bounds"):
        reader_factory(model, data, _configuration(instances=(instance,)))


def test_native_float_conversion_is_validated_before_copy(native_scene, reader_factory):
    model, data = native_scene
    with pytest.raises(QueryBindingError, match="not representable"):
        reader_factory(model, data, _configuration(10 ** 400))


def test_query_copy_keeps_live_integration_state_and_high_water_counters(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(3.0))
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, state_spec))
    after = np.empty_like(before)
    mujoco.mj_getState(model, data, before, state_spec)
    # This denominator is the native integration state and each public
    # high-water field, not a claim to have checked arbitrary plugin state.
    counter_names = tuple(name for name in dir(data) if name.startswith("maxuse_"))
    counters = {name: np.array(getattr(data, name), copy=True) for name in counter_names}
    first = reader.read(data)
    assert reader.read(data) == first
    mujoco.mj_getState(model, data, after, state_spec)
    np.testing.assert_array_equal(before, after)
    for name in counter_names:
        np.testing.assert_array_equal(getattr(data, name), counters[name])


def test_query_read_does_not_change_subsequent_native_trajectory(native_scene, reader_factory):
    model, data = native_scene
    control = mujoco.MjData(model)
    mujoco.mj_copyData(control, model, data)
    reader = reader_factory(model, data, _configuration(3.0))
    reader.read(data)
    for _ in range(12):
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, control)
        np.testing.assert_array_equal(data.qpos, control.qpos)
        np.testing.assert_array_equal(data.qvel, control.qvel)


def test_native_mutation_does_not_rewrite_previous_results(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration())
    first = _result(reader, data, (FLOOR, A))
    before = first.fromto.document(), first.options.document()
    _move_body(model, data, "a", (0.0, 0.0, 0.4))
    model.opt.ccd_iterations += 1
    mujoco.mj_forward(model, data)
    later = _result(reader, data, (FLOOR, A))
    assert before == (first.fromto.document(), first.options.document())
    assert later.fromto != first.fromto and later.options != first.options


@pytest.mark.parametrize("mutate", (
    lambda model: model.geom_type.__setitem__(FLOOR, mujoco.mjtGeom.mjGEOM_BOX),
    lambda model: model.geom_contype.__setitem__(VISUAL, 1),
    lambda model: model.geom_rbound.__setitem__(A, 0.0),
))
def test_population_defining_model_change_requires_a_new_identity(native_scene, reader_factory, mutate):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration())
    mutate(model)
    with pytest.raises(QueryBindingError, match="reset query identity"):
        reader.read(data)


def test_replacing_the_native_query_reader_is_detected_before_execution(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration())
    reader._query = lambda *args: float("nan")
    with pytest.raises(QueryBindingError, match="reader identity changed"):
        reader.read(data)


@pytest.mark.parametrize("returned,witness", ((float("nan"), np.zeros(6)), (0.1, np.full(6, float("inf")))))
def test_nonfinite_native_results_remain_owned_raw_values(returned, witness):
    result = GeomDistanceResult((0, 1), (0, 0), 0.5, returned,
                                NativeValue.capture(witness), NativeValue.capture({}), True)
    assert result.outcome == "unresolved_nonfinite_native_result"
    assert result.capped is None and result.clipped is None
    assert result.fromto.restore().tobytes() == witness.tobytes()


def test_pruned_result_projections_are_the_cutoff_values():
    result = GeomDistanceResult((0, 1), (0, 0), 0.5, None, None, NativeValue.capture({}), True,
                                executed=False, lower_bound=0.7)
    assert result.outcome == "inferred_above_cutoff"
    assert not result.finite
    assert result.capped == 0.5 and result.clipped == 1.0


@pytest.mark.parametrize("cutoff", (0.0, -0.0, -0.025, -0.1))
def test_finite_nonpositive_cutoffs_preserve_native_results(native_scene, reader_factory, cutoff):
    model, data = native_scene
    configuration = _configuration(2.0, instances=[{
        "id": "raw_cutoff", "binding": "mujoco.mj_geomDistance",
        "arguments": {"distmax": cutoff},
        "operands": {"geom1": {"kind": "geom", "id": A},
                     "geom2": {"kind": "geom", "id": B}},
    }])
    reader = reader_factory(model, data, configuration)
    result, = [result for result in reader.read(data) if result.instance_token == "raw_cutoff"]
    assert np.float64(result.distmax).tobytes() == np.float64(cutoff).tobytes()
    assert result.native_return == cutoff
    assert result.capped == cutoff and result.clipped == 1
    reader.close()


def test_explicit_unsupported_pair_retains_native_cutoff(native_scene, reader_factory):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="one" type="plane" size="1 1 .1"/>
      <geom name="two" type="plane" size="1 1 .1" pos="0 0 1"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    instance = {"binding": "mujoco.mj_geomDistance", "id": "unsupported", "arguments": {"distmax": 0.5},
                "operands": {"geom1": {"kind": "geom", "id": 0}, "geom2": {"kind": "geom", "id": 1}}}
    reader = reader_factory(model, data, _configuration(instances=(instance,)))
    assert tuple(reader.population_pairs()) == ()
    result, = reader.read(data)
    assert result.native_return == result.capped == result.distmax
    assert result.clipped == 1
    assert not result.collider_available
    assert result.outcome == "unsupported_collider_native_cutoff"


def test_mixed_type_witness_ownership_on_separated_analytic_fixture(native_scene, reader_factory):
    model, data = native_scene
    reader = reader_factory(model, data, _configuration(3.0))
    result = _result(reader, data, (FIXTURE, A))
    fromto = result.fromto.restore().reshape(2, -1)
    box_center = np.asarray(data.geom_xpos[FIXTURE])
    sphere_center = np.asarray(data.geom_xpos[A])
    box_local = np.asarray(data.geom_xmat[FIXTURE]).reshape(3, 3).T @ (fromto[0] - box_center)
    assert np.max(np.abs(box_local) / model.geom_size[FIXTURE]) == pytest.approx(1.0)
    assert np.linalg.norm(fromto[1] - sphere_center) == pytest.approx(model.geom_size[A, 0])
    assert np.linalg.norm(fromto[1] - fromto[0]) == pytest.approx(result.native_return)
