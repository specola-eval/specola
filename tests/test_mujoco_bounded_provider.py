"""The bounded migration provider through the public adapter path.

These run the real reset and observation path of ``MujocoAdapter`` on the
native distance fixture: provider guards, prepared population, geometry
pair entities, structured samples and the post-read join. They are row
evidence for the bounded provider on the reviewed 3.8.1 build, not producer
surface completeness.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.mujoco

from specola.adapters import mujoco_geom_pairs
from specola.adapters.mujoco import MujocoAdapter
from specola.adapters.mujoco_geom_pairs import (
    CAPPED_KEY, CLIPPED_KEY, GEOM_PAIR_ENTITY_CLASS, GEOM_PAIR_ENTITY_BASE,
)
from specola.adapters.mujoco_provider import BOUNDED_PROVIDER
from specola.conformance.instances import InstanceValidationError
from specola.conformance.provider import ExtractionGuardError, registered_extraction_provider
from specola.conformance.query_bindings import QueryConfigurationRequired
from specola.query_config import QueryConfiguration
from test_mujoco_queries import _SCENE, _UNIVERSE, A, B, FLOOR, _move_body

CUTOFF = 0.5


class _Env:
    def __init__(self, xml=_SCENE):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


def _queries(cutoff=CUTOFF):
    return QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "mujoco.mj_geomDistance", "arguments": {"distmax": cutoff}}],
        "instances": [],
    })


@pytest.fixture
def adapter():
    if mujoco.__version__ != "3.8.1":
        pytest.skip("bounded provider row evidence is executed on the reviewed 3.8.1 build")
    env = _Env()
    adapter = MujocoAdapter(env)
    adapter.set_query_configuration(_queries())
    yield env, adapter
    adapter.close_native_extraction()


def _pairs(observation):
    return {entity.entity_id: entity for entity in observation.perceived_entities.values()
            if entity.entity_class == GEOM_PAIR_ENTITY_CLASS}


def _direct(env, first, second, cutoff=CUTOFF):
    buffer = np.zeros(6)
    return mujoco.mj_geomDistance(env.model, env.data, first, second, cutoff, buffer)


def test_family_classes_share_the_bounded_provider():
    from specola.adapters.libero import LIBEROAdapter
    from specola.adapters.robocasa import RoboCasaAdapter

    for cls in (MujocoAdapter, LIBEROAdapter, RoboCasaAdapter):
        assert registered_extraction_provider(cls) is BOUNDED_PROVIDER


def test_reset_emits_one_geom_pair_entity_per_universe_pair(adapter):
    env, adapter = adapter
    observation = adapter.reset()
    pairs = _pairs(observation)
    records = adapter._geom_pair_records
    assert set(records) == _UNIVERSE
    assert {record.entity_id for record in records.values()} == set(pairs)
    assert all(entity_id >= GEOM_PAIR_ENTITY_BASE for entity_id in pairs)
    for pair, record in records.items():
        entity = pairs[record.entity_id]
        assert set(entity.properties) == {CAPPED_KEY, CLIPPED_KEY}
        assert adapter.entity_labels[record.entity_id] == record.label
        assert record.label.startswith("geom_pair:")
        assert observation.component_tags[record.entity_id][CAPPED_KEY] == record.tags
        assert "environment" in record.tags
        # The scalar keys are the declared projections of the direct native call.
        direct = _direct(env, *pair)
        if entity.properties[CLIPPED_KEY] == 1.0:
            assert entity.properties[CAPPED_KEY] == CUTOFF and direct == CUTOFF
        else:
            assert entity.properties[CAPPED_KEY] == pytest.approx(direct)


def test_structured_samples_carry_native_and_pruned_outcomes(adapter):
    env, adapter = adapter
    observation = adapter.reset()
    records = adapter._geom_pair_records
    samples = {sample.address: sample for sample in observation.extraction.samples
               if sample.owner_kind == GEOM_PAIR_ENTITY_CLASS}
    assert len(samples) == 2 * len(records)
    binding = "mujoco.mj_geomDistance"
    pruned = samples[(binding + ".return", GEOM_PAIR_ENTITY_CLASS, (A, B), "", binding)]
    assert pruned.availability == "inferred_above_cutoff"
    assert pruned.scalar_keys == (CAPPED_KEY, CLIPPED_KEY)
    value = pruned.value.restore()
    assert value["native_return"] is None and value["lower_bound"] >= CUTOFF and value["distmax"] == CUTOFF
    witness = samples[(binding + ".fromto", GEOM_PAIR_ENTITY_CLASS, (A, B), "", binding)]
    assert witness.value.restore() is None and witness.scalar_keys == ()
    executed = samples[(binding + ".return", GEOM_PAIR_ENTITY_CLASS, (FLOOR, A), "", binding)]
    assert executed.availability == "native_below_cutoff"
    assert executed.value.restore()["native_return"] == pytest.approx(0.2)
    assert samples[(binding + ".fromto", GEOM_PAIR_ENTITY_CLASS, (FLOOR, A), "", binding)].value.restore().shape == (6,)


def test_keys_persist_while_a_pair_moves_through_the_cutoff(adapter):
    env, adapter = adapter
    adapter.reset()
    entity_id = adapter._geom_pair_records[(A, B)].entity_id
    _move_body(env.model, env.data, "b", (0.0, 0.3, 0.3))
    near = adapter.get_observation().perceived_entities[entity_id].properties
    assert near[CAPPED_KEY] == pytest.approx(0.1) and near[CLIPPED_KEY] == 0.0
    _move_body(env.model, env.data, "b", (0.0, 1.5, 0.3))
    far = adapter.get_observation().perceived_entities[entity_id].properties
    assert far[CAPPED_KEY] == CUTOFF and far[CLIPPED_KEY] == 1.0


def test_missing_query_configuration_fails_before_any_observation(adapter):
    env, adapter = adapter
    adapter.set_query_configuration(QueryConfiguration.empty())
    with pytest.raises(QueryConfigurationRequired):
        adapter.reset()
    assert adapter.last_observation is None


def test_capabilities_declare_the_pair_class_and_tags(adapter):
    env, adapter = adapter
    adapter.reset()
    capabilities = adapter.capabilities
    assert capabilities.components_by_class[GEOM_PAIR_ENTITY_CLASS] == frozenset({CAPPED_KEY, CLIPPED_KEY})
    tags = capabilities.component_tags_by_class[GEOM_PAIR_ENTITY_CLASS]
    assert {"environment", "body:a", "body:b"} <= tags


def test_tampered_scalar_projection_is_rejected_by_the_post_read_join(adapter, monkeypatch):
    env, adapter = adapter
    adapter.reset()
    honest = mujoco_geom_pairs.scalar_values

    def dishonest(result):
        values = honest(result)
        if values is not None and result.geom_ids == (FLOOR, A):
            values[CAPPED_KEY] = values[CAPPED_KEY] + 0.01
        return values

    monkeypatch.setattr(mujoco_geom_pairs, "scalar_values", dishonest)
    with pytest.raises(InstanceValidationError, match="differs from its declared native projection"):
        adapter.get_observation()


def test_dropped_pair_sample_is_rejected_by_the_post_read_join(adapter, monkeypatch):
    env, adapter = adapter
    adapter.reset()
    honest = mujoco_geom_pairs.geometry_samples
    monkeypatch.setattr(mujoco_geom_pairs, "geometry_samples",
                        lambda binding, result, record: honest(binding, result, record)[:1])
    with pytest.raises(InstanceValidationError, match="native instance population differs"):
        adapter.get_observation()


def test_changed_adapter_source_custody_stops_the_episode(adapter, monkeypatch):
    env, adapter = adapter
    adapter.reset()
    monkeypatch.setattr(BOUNDED_PROVIDER, "_adapter_source_custody", lambda instance: ())
    with pytest.raises(ExtractionGuardError, match="adapter_source_changed"):
        adapter.get_observation()


def test_failed_projection_stage_is_loud_and_preserves_the_last_observation(adapter, monkeypatch):
    """The write-free contract, ported from the retired live-forward injection."""
    from specola.adapters import mujoco_projection

    env, adapter = adapter
    accepted = adapter.reset()
    warmstart = np.array(env.data.qacc_warmstart, copy=True)
    original = mujoco_projection.MujocoProjection.observe

    def failing(self, data):
        raise RuntimeError("injected completion failure")

    monkeypatch.setattr(mujoco_projection.MujocoProjection, "observe", failing)
    _move_body(env.model, env.data, "b", (0.0, 0.3, 0.3))
    with pytest.raises(RuntimeError, match="injected completion failure"):
        adapter.get_observation()
    assert adapter.last_observation is accepted
    assert adapter.episode_started
    np.testing.assert_array_equal(env.data.qacc_warmstart, warmstart)
    monkeypatch.setattr(mujoco_projection.MujocoProjection, "observe", original)
    later = adapter.get_observation()
    assert later.timestep == accepted.timestep + 1


def test_adapter_legacy_scalars_remain_under_the_rule(adapter):
    env, adapter = adapter
    observation = adapter.reset()
    legacy = [entity for entity in observation.perceived_entities.values()
              if entity.entity_class != GEOM_PAIR_ENTITY_CLASS and entity.properties]
    assert legacy, "the base adapter still emits its legacy scalar properties"
    prepared_rule = observation.extraction.metadata
    assert prepared_rule


# -- specification scope over the pair universe (issue #136) -------------------
#
# Body "a" and its descendants are the robot here, so 13 of the 15 universe
# pairs are robot pairs and the two pairs of body "b" with the floor and the
# fixture are environment pairs.

ROBOT_CLEARANCE = 'geom_pair_*[geom_distance_capped @ robot] > -1'
BODY_B_CLEARANCE = 'geom_pair_*[geom_distance_capped @ body:b] > -1'


@pytest.fixture
def robot_adapter():
    if mujoco.__version__ != "3.8.1":
        pytest.skip("bounded provider row evidence is executed on the reviewed 3.8.1 build")
    made = []

    def make(queries=None):
        env = _Env()
        adapter = MujocoAdapter(env, robot_bodies=("a",))
        adapter.set_query_configuration(queries or _queries())
        made.append(adapter)
        return env, adapter

    yield make
    for adapter in made:
        adapter.close_native_extraction()


def _spec_file(tmp_path, formula=ROBOT_CLEARANCE, name="specs.yaml"):
    path = tmp_path / name
    path.write_text('specs:\n- name: clearance\n  formula: "{}"\n'.format(formula))
    return str(path)


def _tagged(adapter, tag):
    return {record.entity_id for record in adapter._geom_pair_records.values() if tag in record.tags}


def _pair_samples(observation):
    return [sample for sample in observation.extraction.samples if sample.owner_kind == GEOM_PAIR_ENTITY_CLASS]


def test_scope_emits_only_the_pairs_a_specification_can_bind(robot_adapter, tmp_path):
    from specola.session import Monitor

    env, adapter = robot_adapter()
    session = Monitor(_spec_file(tmp_path), results_dir=None).watch(adapter)
    every = {record.entity_id for record in adapter._geom_pair_records.values()}
    robot = _tagged(adapter, "robot")
    assert len(robot) == 13 and len(every) == 15
    assert adapter._extraction_entity_scope[GEOM_PAIR_ENTITY_CLASS] == frozenset(
        {(None, ((CAPPED_KEY, ("robot",)),))})
    result = session.step()
    observation = adapter.last_observation
    assert set(_pairs(observation)) == robot
    samples = _pair_samples(observation)
    assert len(samples) == 2 * len(robot)
    assert {sample.entity_id for sample in samples if sample.entity_id is not None} == robot
    assert len(result.raw.evaluations) == len(robot)


def test_no_scope_emits_every_pair(robot_adapter):
    env, adapter = robot_adapter()
    adapter.reset()
    every = {record.entity_id for record in adapter._geom_pair_records.values()}
    observation = adapter.get_observation()
    assert set(_pairs(observation)) == every
    assert len(_pair_samples(observation)) == 2 * len(every)
    # A key scope without an entity scope (library use) keeps the universe too.
    adapter.set_extraction_scope({GEOM_PAIR_ENTITY_CLASS: frozenset({CAPPED_KEY})})
    assert set(_pairs(adapter.get_observation())) == every


def test_population_all_keeps_every_pair_under_a_scope(robot_adapter, tmp_path):
    from specola.session import Monitor

    queries = QueryConfiguration.from_document({
        "schema": "specola.queries.v1",
        "populations": [{"binding": "mujoco.mj_geomDistance",
                         "arguments": {"distmax": CUTOFF, "population": "all"}}],
        "instances": [],
    })
    env, adapter = robot_adapter(queries)
    session = Monitor(_spec_file(tmp_path), results_dir=None).watch(adapter)
    every = {record.entity_id for record in adapter._geom_pair_records.values()}
    # The catalog-only reset reads the universe too.
    assert set(_pairs(adapter.last_observation)) == every
    result = session.step()
    observation = adapter.last_observation
    assert set(_pairs(observation)) == every
    assert len(_pair_samples(observation)) == 2 * len(every)
    # The extra pairs are emitted, not bound: binding still follows the tag filter.
    assert len(result.raw.evaluations) == len(_tagged(adapter, "robot"))


def test_check_lists_the_pairs_a_scoped_run_emits(robot_adapter, tmp_path, capsys):
    import re
    from specola.check_specs import _check_entry
    from specola.components import ComponentCatalog
    from specola.engine import MonitorEngine
    from specola.session import Monitor

    specs = _spec_file(tmp_path)
    env, adapter = robot_adapter()
    Monitor(specs, results_dir=None).watch(adapter).step()
    emitted = set(_pairs(adapter.last_observation))

    # specola check's path: a fresh adapter, unscoped, against its reset catalog.
    env, fresh = robot_adapter()
    entities = fresh.reset().perceived_entities
    metadata = {entity_id: fresh.property_metadata(entity_id) for entity_id in entities}
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    assert len([entity_id for entity_id in entities if entity_id >= GEOM_PAIR_ENTITY_BASE]) == 15
    ((_name, spec, _error),) = MonitorEngine.iter_spec_entries(specs)
    capsys.readouterr()
    assert _check_entry(spec, fresh.dt, catalog, metadata, entities, fresh.entity_labels,
                        capabilities=fresh.capabilities)
    output = capsys.readouterr().out
    slot_line = next((line for line in output.splitlines() if "geom_pair:" in line), output)
    listed = {int(entity_id) for entity_id in re.findall(r"geom_pair:[^\[,]*\[(\d+)\]", slot_line)}
    assert listed == emitted and len(listed) == 13


def test_scope_change_between_episodes_builds_a_new_population(robot_adapter, tmp_path, monkeypatch):
    from specola.session import Monitor

    prepared = []
    original = BOUNDED_PROVIDER.prepare

    def recording(adapter, *, reset):
        result = original(adapter, reset=reset)
        prepared.append((reset, result.population.instances))
        return result

    monkeypatch.setattr(BOUNDED_PROVIDER, "prepare", recording)
    env, adapter = robot_adapter()
    robot_monitor = Monitor(_spec_file(tmp_path), results_dir=None)
    body_b_monitor = Monitor(_spec_file(tmp_path, BODY_B_CLEARANCE, "b.yaml"), results_dir=None)

    robot_monitor.watch(adapter).step()
    robot = _tagged(adapter, "robot")
    assert set(_pairs(adapter.last_observation)) == robot
    first = prepared[-1][1]

    body_b_monitor.watch(adapter).step()
    body_b = _tagged(adapter, "body:b")
    assert body_b != robot and set(_pairs(adapter.last_observation)) == body_b
    second = prepared[-1][1]
    assert second is not first

    def pairs(instances):
        return {row.entity_id for row in instances if row.owner_kind == GEOM_PAIR_ENTITY_CLASS and row.entity_id}

    assert pairs(first) == robot and pairs(second) == body_b
    # Each catalog-only reset declared no pair; each read after it, its
    # scoped pairs.
    assert [len(pairs(instances)) for reset, instances in prepared if reset] == [0, 0]


def _catalog_rows(catalog):
    return {entity_id: (info.entity_class, info.components, dict(info.component_groups), dict(info.component_tags))
            for entity_id, info in catalog.entities.items()}


def test_watch_resets_catalog_only_and_compiles_against_every_pair(robot_adapter, tmp_path, monkeypatch):
    from specola import session as session_module
    from specola.capabilities import reset_adapter
    from specola.session import Monitor

    resets = []

    def recording(adapter):
        resets.append(reset_adapter(adapter))
        return resets[-1]

    monkeypatch.setattr(session_module, "reset_adapter", recording)
    env, adapter = robot_adapter()
    session = Monitor(_spec_file(tmp_path), results_dir=None).watch(adapter)
    (reset,) = resets
    assert not _pairs(reset.initial_observation) and not _pair_samples(reset.initial_observation)
    # The catalog equals the one an unscoped reset discovers, pairs included.
    env, unscoped = robot_adapter()
    expected = _catalog_rows(reset_adapter(unscoped).discovered_catalog)
    assert len([row for row in expected.values() if row[0] == GEOM_PAIR_ENTITY_CLASS]) == 15
    assert _catalog_rows(reset.discovered_catalog) == expected
    session.step()
    assert set(_pairs(adapter.last_observation)) == _tagged(adapter, "robot")


def test_a_reset_hook_sees_the_catalog_only_reset(robot_adapter, tmp_path):
    # Maintainer decision 2026-10-13: the dashboard's discovery document
    # no longer forces a full reset (365,423 entities and about 120 s on
    # RoboCasa); the hook sees the same reset as a run without one, with
    # every pair in the catalog and none read.
    from specola.session import Monitor

    env, adapter = robot_adapter()
    seen = []
    Monitor(_spec_file(tmp_path), results_dir=None).watch(adapter, _reset_hook=lambda _adapter, reset: seen.append(reset))
    assert not _pairs(seen[0].initial_observation)
    rows = _catalog_rows(seen[0].discovered_catalog)
    assert len([row for row in rows.values() if row[0] == GEOM_PAIR_ENTITY_CLASS]) == 15


def test_a_failed_watch_leaves_the_adapter_unscoped(robot_adapter, tmp_path):
    from specola.entities import SpecConfigError
    from specola.session import Monitor

    env, adapter = robot_adapter()
    monitor = Monitor(_spec_file(tmp_path, "geom_pair_*[no_such_key] > 0"), results_dir=None)
    with pytest.raises(SpecConfigError):
        monitor.watch(adapter)
    assert len(_pairs(adapter.reset())) == 15


def test_a_selector_scope_surviving_into_reset_reads_every_pair(robot_adapter, tmp_path):
    from specola.session import Monitor

    env, adapter = robot_adapter()
    Monitor(_spec_file(tmp_path), results_dir=None).watch(adapter).step()
    robot = _tagged(adapter, "robot")
    # A direct reset keeps the session's scope; its selectors filter on tags
    # the reset registers after declaring its population, so it reads all.
    assert len(_pairs(adapter.reset())) == 15
    assert set(_pairs(adapter.get_observation())) == robot


def test_check_reads_only_the_shown_pairs_and_reports_as_a_full_reset(robot_adapter, tmp_path, capsys, monkeypatch):
    import re
    import sys
    from specola import check_specs, cli_common

    made = []

    def load(sim, **kwargs):
        made.append(robot_adapter()[1])
        return made[-1]

    monkeypatch.setattr(cli_common, "_load_sim", load)
    monkeypatch.setattr(sys, "argv", ["specola check", _spec_file(tmp_path), "--sim", "fixture", "--no-pager"])

    def report():
        with pytest.raises(SystemExit) as exited:
            check_specs.main()
        assert exited.value.code == 0
        # The readiness line counts the entities the reset read.
        return [line for line in capsys.readouterr().out.splitlines() if "entities=" not in line]

    catalog_only = report()
    # One extra read, of the robot slot's representative pair only.
    assert len(_pairs(made[-1].last_observation)) == 1
    value_line = next(line for line in catalog_only if re.search(r"geom_pair_\d+\s+" + CAPPED_KEY, line))
    float(value_line.split()[-1])
    full_reset = check_specs.load_sim
    monkeypatch.setattr(check_specs, "load_sim", lambda *args, **kwargs: full_reset(*args, **{**kwargs, "catalog_only": False}))
    assert report() == catalog_only
    assert len(_pairs(made[-1].last_observation)) == 15


@pytest.mark.libero
def test_libero_reset_and_observation_with_pair_universe():
    pytest.importorskip("libero")
    from specola.adapters.libero import LIBEROAdapter, make_libero_env

    env, task = make_libero_env("libero_object", task_id=0)
    adapter = LIBEROAdapter(env, task=task)
    adapter.set_query_configuration(_queries(0.05))
    try:
        observation = adapter.reset()
        pairs = _pairs(observation)
        assert len(pairs) == len(adapter._geom_pair_records) > 2000
        executed = sum(1 for sample in observation.extraction.samples
                       if sample.owner_kind == GEOM_PAIR_ENTITY_CLASS and sample.availability != "inferred_above_cutoff")
        assert 0 < executed < len(pairs)
        env.step([0.0] * 7)
        later = adapter.get_observation()
        assert set(_pairs(later)) == set(pairs)
    finally:
        adapter.close_native_extraction()
