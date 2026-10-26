"""Exact Meta-World 3.1.1 integration over all 50 v3 task classes."""

from __future__ import annotations

import numpy as np
import pytest

metaworld = pytest.importorskip("metaworld")
pytestmark = pytest.mark.metaworld

from specola.adapters.metaworld import MetaworldAdapter
from specola.adapters.metaworld_declarations import (
    task_declaration,
    task_names,
)


_TASKS = task_names()
_TRACE_TASKS = (
    "assembly-v3",
    "button-press-v3",
    "door-lock-v3",
    "hammer-v3",
    "push-wall-v3",
    "sweep-v3",
)


def _environment(task_name):
    benchmark = metaworld.MT1(task_name, seed=0)
    environment = benchmark.train_classes[task_name]()
    environment.set_task(benchmark.train_tasks[0])
    environment.reset(seed=0)
    return environment


@pytest.mark.parametrize("task_name", _TASKS)
def test_all_50_tasks_expose_exact_declared_target_and_objects(task_name):
    environment = _environment(task_name)
    try:
        adapter = MetaworldAdapter(environment, task_name=task_name)
        observation = adapter.reset()
        declaration = task_declaration(task_name)
        object_states = [
            entity
            for entity in observation.perceived_entities.values()
            if entity.entity_class == "object"
        ]
        target_states = [
            entity
            for entity in observation.perceived_entities.values()
            if entity.entity_class == "target"
        ]

        assert len(object_states) == (
            np.asarray(environment._get_pos_objects()).size // 3
        )
        assert len(object_states) == len(declaration.objects)
        assert len(target_states) == 1
        np.testing.assert_array_equal(
            target_states[0].position,
            environment._get_pos_goal(),
        )
        assert not any(
            entity.entity_class == "region"
            for entity in observation.perceived_entities.values()
        )
        assert adapter.regions == []
        assert [
            item.source_kind
            for item in adapter.semantic_policy.declared_entities[1:]
        ] == [item.kind for item in declaration.objects]
        assert [
            item.source_name
            for item in adapter.semantic_policy.declared_entities[1:]
        ] == [item.declared_name for item in declaration.objects]
        for item in declaration.objects:
            entity_id = adapter.name_to_id[
                "declared:object:{}".format(item.index)
            ]
            role = "target" if item.index == 0 else "distractor"
            assert adapter.entity_roles[entity_id] == role
            assert adapter.entity_role_basis[entity_id] == (
                "structural_derivation"
            )
            provenance = adapter.entity_role_provenance[entity_id]
            if item.index == 0 and len(declaration.objects) > 1:
                assert "_get_id_main_object() source" in provenance
                assert "selects geom" in provenance
                assert "belongs to object 0 source-body subtree" in (
                    provenance
                )
            elif item.index == 0:
                assert "sole declared object" in provenance
            else:
                assert "non-primary ordinary-object complement" in (
                    provenance
                )
                assert "not task-irrelevant" in provenance
            metadata = adapter.property_metadata(entity_id)
            assert all(
                "role:" + role in entry.tags
                for entry in metadata.values()
            )
        body_rows = [
            body_id
            for _entity_id, body_id in adapter._body_owner_pairs()
        ]
        assert sorted(body_rows) == list(
            range(int(adapter._model.nbody))
        )
        assert len(body_rows) == len(set(body_rows))
        joint_rows = {
            joint_id
            for _body_id, _entity_id, joint_id
            in adapter._joint_owner_triples()
        }
        assert joint_rows == set(range(int(adapter._model.njnt)))

        for entity_id, entity in observation.perceived_entities.items():
            assert set(entity.properties) == set(
                adapter.property_metadata(entity_id)
            )
    finally:
        environment.close()


def test_hammer_roles_velocity_and_contact_ownership_are_native():
    import mujoco

    environment = _environment("hammer-v3")
    try:
        adapter = MetaworldAdapter(environment, task_name="hammer-v3")
        observation = adapter.reset()
        for index, name, role in (
            (0, "hammer", "target"),
            (1, "nail_link", "distractor"),
        ):
            entity_id = adapter.name_to_id[
                "declared:object:{}".format(index)
            ]
            state = observation.perceived_entities[entity_id]
            expected = np.zeros(6)
            body_id = adapter._name2id("body", name)
            mujoco.mj_objectVelocity(
                adapter._model,
                adapter._data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                expected,
                0,
            )
            np.testing.assert_allclose(
                state.velocity,
                expected[3:],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                [
                    state.properties["object_linvel_0"],
                    state.properties["object_linvel_1"],
                    state.properties["object_linvel_2"],
                ],
                expected[3:],
                atol=1e-12,
            )
            assert adapter._contact_role_by_body[body_id] == role
        assert adapter._contact_tag_vocabulary() == frozenset(
            {"role:target", "role:distractor", "role:robot"}
        )
    finally:
        environment.close()


@pytest.mark.parametrize("task_name", _TRACE_TASKS)
def test_declared_population_and_identity_stay_fixed_over_three_steps(task_name):
    environment = _environment(task_name)
    try:
        adapter = MetaworldAdapter(environment, task_name=task_name)
        initial = adapter.reset()
        declared_ids = {
            entity_id
            for entity_id, entity in initial.perceived_entities.items()
            if entity.entity_class in {"object", "target"}
        }
        declared_classes = {
            entity_id: initial.perceived_entities[entity_id].entity_class
            for entity_id in declared_ids
        }

        for _ in range(3):
            environment.step(np.zeros(4, dtype=np.float32))
            observation = adapter.get_observation()
            assert declared_ids <= set(observation.perceived_entities)
            assert {
                entity_id: observation.perceived_entities[
                    entity_id
                ].entity_class
                for entity_id in declared_ids
            } == declared_classes
    finally:
        environment.close()


def test_sweep_target_and_raw_goal_site_remain_distinct_channels():
    environment = _environment("sweep-v3")
    try:
        adapter = MetaworldAdapter(environment, task_name="sweep-v3")
        observation = adapter.reset()
        target = observation.perceived_entities[
            adapter.name_to_id["declared:target"]
        ]
        raw_goal = observation.perceived_entities[
            adapter.name_to_id["site:goal"]
        ]

        np.testing.assert_array_equal(target.position, environment._get_pos_goal())
        np.testing.assert_array_equal(
            raw_goal.position,
            environment.data.site("goal").xpos,
        )
        assert not np.array_equal(target.position, raw_goal.position)
        assert target.entity_class == "target"
        assert raw_goal.entity_class == "site"
    finally:
        environment.close()


def test_capture_flips_cameras_whose_up_points_world_down():
    # Meta-World's corner cameras are authored for mujoco-py's bottom-up
    # buffer; raw mujoco.Renderer output is upside down for them. The
    # adapter flips exactly the cameras whose runtime up vector points
    # at world-down, and leaves correctly authored ones alone.
    import mujoco
    import numpy as np
    from specola.adapters.metaworld import _make_metaworld
    from specola.adapters.mujoco import _raw_struct

    adapter = _make_metaworld(None, None, False)
    adapter.reset()
    model = _raw_struct(adapter._model, "_model")
    data = _raw_struct(adapter._data, "_data")
    renderer = mujoco.Renderer(model, height=48, width=64)
    try:
        for camera, expect_flip in (("corner", True),
                                    ("behindGripper", False)):
            got = adapter.capture_frame(camera=camera, width=64, height=48)
            renderer.update_scene(data, camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
            raw = renderer.render()
            reference = raw[::-1] if expect_flip else raw
            assert np.array_equal(got, reference), camera
    finally:
        renderer.close()
