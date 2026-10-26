# tests/test_libero_integration.py
"""The README's five-line integration, run for real. Env-dependent:
skipped wherever LIBERO isn't installed (CI), runs in the lab env."""
import numpy as np
import pytest

libero = pytest.importorskip("libero")
from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

pytestmark = pytest.mark.libero

# The producer's source declarations supply tasks even when the adapter has
# never instantiated their suite. The separate registry check below uses the
# producer's generated Task declarations as an independent reconciliation.
_TASKS = tuple(
    (suite, task_id)
    for suite, tasks in libero_task_map.items()
    for task_id in range(len(tasks))
)


def test_task_matrix_matches_producer_declarations():
    expected = {
        (suite, task_id)
        for suite, tasks in benchmark.task_maps.items()
        for task_id, _task in enumerate(tasks.values())
    }
    assert len(_TASKS) == len(set(_TASKS))
    assert set(_TASKS) == expected
    assert set(libero_task_map) <= set(benchmark.get_benchmark_dict())
    for suite, declarations in benchmark.task_maps.items():
        assert list(declarations) == list(libero_task_map[suite])
        assert [task.name for task in declarations.values()] == list(declarations)


@pytest.mark.parametrize(
    ("suite", "task_id"),
    _TASKS,
    ids=[f"{suite}:{task_id}" for suite, task_id in _TASKS],
)
def test_all_tasks_reconcile_instantiated_declarations(suite, task_id):
    from specola.adapters.libero import LIBEROAdapter, make_libero_env

    env, task = make_libero_env(
        suite,
        task_id=task_id,
        camera_heights=64,
        camera_widths=64,
    )
    try:
        adapter = LIBEROAdapter(env, task=task)
        observation = adapter.reset()
        snapshot = adapter._declaration_snapshot()
        inner = adapter._libero_inner_env()

        parsed_object_names = {
            name
            for names in inner.parsed_problem["objects"].values()
            for name in names
        }
        parsed_fixture_names = {
            name
            for names in inner.parsed_problem["fixtures"].values()
            for name in names
        }
        runtime_object_names = {
            name for name, _root in snapshot.objects
        }
        runtime_fixture_names = {
            name for name, _root in snapshot.fixtures
        }
        arena_fixture_names = {
            record.name for record in snapshot.arena_fixtures
        }
        assert set(snapshot.parsed_object_names) == parsed_object_names
        assert set(snapshot.parsed_fixture_names) == parsed_fixture_names
        assert runtime_object_names == parsed_object_names
        assert parsed_fixture_names == (
            runtime_fixture_names | arena_fixture_names
        )
        assert arena_fixture_names == {inner.workspace_name}
        assert all(
            record.disposition == "arena_owned_compiled_body"
            for record in snapshot.arena_fixtures
        )

        declared_roots = {
            root
            for _name, root in snapshot.objects + snapshot.fixtures
        }
        registered_body_roots = {
            record.body_name
            for record in adapter._bodies
        }
        assert declared_roots <= registered_body_roots

        objects_of_interest = set(inner.obj_of_interest)
        expected_roles = {
            name: (
                "target"
                if name in objects_of_interest
                else "distractor"
            )
            for name in parsed_object_names
        }
        assert {
            name: role
            for name, (role, _basis, _provenance)
            in snapshot.role_by_object_name.items()
        } == expected_roles
        body_by_root = {
            record.body_name: record
            for record in adapter._bodies
        }
        for name, root in snapshot.objects:
            record = body_by_root[root]
            role = expected_roles[name]
            assert adapter.entity_roles[record.entity_id] == role
            assert (
                adapter.entity_role_basis[record.entity_id]
                == "structural_derivation"
            )
            metadata = adapter.property_metadata(record.entity_id)
            assert metadata
            assert all(
                "role:" + role in entry.tags
                for entry in metadata.values()
            )
        assert adapter._contact_tag_vocabulary() == frozenset(
            {"role:target", "role:distractor", "role:robot"}
        )

        instantiated_names = set(inner.object_sites_dict)
        assert {spec.name for spec in snapshot.regions} == instantiated_names
        assert {
            record.spec.name
            for record in adapter._regions
        } == instantiated_names
        assert set(snapshot.uninstantiated_region_names) == (
            set(inner.parsed_problem["regions"]) - instantiated_names
        )

        raw_site_by_name = {
            record.source_name: record
            for record in adapter._raw_sites
        }
        semantic_region_by_name = {
            record.spec.name: record
            for record in adapter._regions
        }
        assert instantiated_names <= set(raw_site_by_name)
        for name in instantiated_names:
            raw_site = raw_site_by_name[name]
            semantic_region = semantic_region_by_name[name]
            assert raw_site.entity_id != semantic_region.entity_id
            assert adapter.name_to_id[f"site:{name}"] == raw_site.entity_id
            assert adapter.name_to_id[name] == semantic_region.entity_id
            assert adapter.entity_labels[raw_site.entity_id] == (
                f"{name} (MuJoCo site)"
            )
            assert adapter.entity_labels[semantic_region.entity_id] == name
            assert raw_site.entity_id in observation.perceived_entities
            assert semantic_region.entity_id in observation.perceived_entities
            assert "declared_name:" + name in {
                tag
                for meta in adapter.property_metadata(
                    semantic_region.entity_id
                ).values()
                for tag in meta.tags
            }

        expected_object_relation_count = len(snapshot.objects) * (
            len(snapshot.objects) - 1
        )
        expected_object_region_relation_count = (
            len(snapshot.objects) * len(snapshot.regions)
        )
        assert len(snapshot.object_relations) == (
            expected_object_relation_count
        )
        assert len(snapshot.object_region_relations) == (
            expected_object_region_relation_count
        )
        source_by_name = {
            relation.source.name: relation.source
            for relation in (
                snapshot.object_relations
                + snapshot.object_region_relations
            )
        }
        assert set(source_by_name) <= parsed_object_names
        for name, source in source_by_name.items():
            assert source.body_id == inner.obj_body_id[name]
        expected_relation_keys = {
            relation.declaration_key
            for relation in (
                snapshot.object_relations
                + snapshot.object_region_relations
            )
        }
        relation_records = {
            record.spec.declaration_key: record
            for record in adapter._declared_entities
        }
        assert set(relation_records) == expected_relation_keys
        for declaration_key, record in relation_records.items():
            state = observation.perceived_entities[record.entity_id]
            metadata = adapter.property_metadata(record.entity_id)
            assert set(state.properties) == set(metadata)
            assert all(np.isfinite(value) for value in state.properties.values())
            assert declaration_key.startswith(
                state.entity_class + ":"
            )
    finally:
        env.close()


def test_five_line_integration(tmp_path):
    from specola import Monitor
    from specola.adapters.libero import make_libero_env

    specs = tmp_path / "specs.yaml"
    specs.write_text(
        'specs:\n  - name: gentle\n'
        '    formula: "historically[0,1s](gripper_*[recent_ee_force_norm] <= 200.0)"\n'
    )

    env, task = make_libero_env("libero_object", task_id=0)
    try:
        monitor = Monitor(str(specs), results_dir=str(tmp_path / "results"))

        session = monitor.watch(
            env,
            task=task,
            meta={"task": "libero_object:0"},
        )
        for _ in range(25):
            env.step(np.zeros(7))
            result = session.step()
        record = session.end(success=False)

        assert record.specs["gentle"].verdict in {
            "pass",
            "fail",
            "undetermined",
        }
        assert record.path is not None and record.path.exists()
    finally:
        env.close()


def test_perception_gate_can_filter_declared_fixtures():
    from specola.adapters.libero import LIBEROAdapter, make_libero_env

    env, task = make_libero_env(
        "libero_goal",
        task_id=0,
        camera_heights=64,
        camera_widths=64,
    )
    try:
        adapter = LIBEROAdapter(
            env,
            task=task,
            perception_gate=lambda entity, _env: (
                entity.entity_class not in {"object", "fixture"}
            ),
        )
        observation = adapter.reset()

        object_ids = {
            record.entity_id
            for record in adapter._bodies
            if record.entity_class in {"object", "fixture"}
        }
        assert object_ids
        assert object_ids.isdisjoint(observation.perceived_entities)
        qpos = adapter.field_materialization["qpos"]
        assert qpos.scene_applicable
        assert qpos.owner_omissions
    finally:
        env.close()


@pytest.mark.parametrize(
    (
        "suite",
        "task_id",
        "entity_label",
        "fixture_name",
        "joint_name",
        "metadata_group",
    ),
    [
        (
            "libero_goal",
            0,
            "wooden_cabinet_1_middle_region",
            "wooden_cabinet_1",
            "wooden_cabinet_1_middle_level",
            "joint displacement",
        ),
        (
            "libero_goal",
            7,
            "flat_stove_1",
            "flat_stove_1",
            "flat_stove_1_button",
            "joint angle",
        ),
        (
            "libero_10",
            3,
            "white_cabinet_1_bottom_region",
            "white_cabinet_1",
            "white_cabinet_1_bottom_level",
            "joint displacement",
        ),
        (
            "libero_10",
            9,
            "microwave_1",
            "microwave_1",
            "microwave_1_microjoint",
            "joint angle",
        ),
    ],
)
def test_declared_fixture_articulation_is_entity_scoped(
    suite,
    task_id,
    entity_label,
    fixture_name,
    joint_name,
    metadata_group,
):
    from specola.adapters.libero import LIBEROAdapter, make_libero_env

    env, task = make_libero_env(suite, task_id=task_id)
    try:
        adapter = LIBEROAdapter(env, task=task)
        observation = adapter.reset()
        entity_id = next(
            entity_id
            for entity_id, label in adapter.entity_labels.items()
            if label.split("  (")[0] == entity_label
        )
        entity = observation.perceived_entities[entity_id]
        key = f"qpos_{joint_name}"

        inner = env.env
        fixture = inner.fixtures_dict[fixture_name]
        raw_values = inner.object_states_dict[fixture_name].get_joint_state()
        raw_by_name = dict(zip(fixture.joints, raw_values))
        joint_id = adapter._name2id("joint", joint_name)
        start, stop = adapter._joint_slice(
            joint_id,
            "jnt_qposadr",
            "nq",
        )
        model_values = np.asarray(
            adapter._data.qpos[start:stop],
            dtype=float,
        ).reshape(-1)

        np.testing.assert_allclose(
            np.asarray(raw_by_name[joint_name]).reshape(-1),
            model_values,
        )
        assert entity.properties[key] == pytest.approx(float(model_values[0]))

        metadata = adapter.property_metadata(entity_id)
        assert metadata[key].group == metadata_group
        assert set(entity.properties) == set(metadata)
        assert (
            f"entity[{entity_id}].properties.{key}"
            in adapter.field_materialization["qpos"].emitted_keys
        )

        if entity_label.endswith("_region"):
            declared = inner.object_sites_dict[entity_label].joints
            expected_qpos = {
                f"qpos_{name}"
                for name in declared
            }
            actual_qpos = {
                name
                for name in entity.properties
                if name.startswith("qpos_")
            }
            assert actual_qpos == expected_qpos

        non_articulated_regions = [
            record
            for record in adapter._regions
            if not adapter._declared_joint_ids(record.spec.name)
        ]
        assert non_articulated_regions
        for record in non_articulated_regions:
            region = observation.perceived_entities[record.entity_id]
            assert set(region.properties) == set(
                adapter.property_metadata(record.entity_id)
            )
    finally:
        env.close()
