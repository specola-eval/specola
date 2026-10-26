"""Ordered batched monitoring is exactly a loop over isolated scalar engines."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import List, Optional

import numpy as np  # type: ignore[import-not-found]
import pytest  # type: ignore[import-not-found]

from specola import (
    AdapterCapabilities,
    EntityState,
    MonitorEngine,
    SpecConfigError,
    TrackState,
    VocabularyMode,
)
from specola.backends.rtamt import rtamt_available
from specola.batch import (
    BatchEpisodeContext,
    BatchStepError,
    BatchedMonitorEngine,
)
from specola.components import ComponentCatalog


SPEC_TEMPLATE = """\
specs:
  - name: fan
    formula: "robot_*[qacc_*] <= 0"
    backend: {backend}
  - name: early_fail
    formula: "G[0,3s](robot_*[qacc_a] <= 0)"
    backend: {backend}
  - name: ordinary_fail
    formula: "F[0,3s](robot_*[qacc_b] >= 0)"
    backend: {backend}
"""


def _entity(
    entity_id,
    qacc_a=1.0,
    qacc_b=-1.0,
    *,
    qacc_c=None,
    omit=(),
    track_state=TrackState.CONFIRMED,
):
    properties = {"qacc_a": float(qacc_a), "qacc_b": float(qacc_b)}
    if qacc_c is not None:
        properties["qacc_c"] = float(qacc_c)
    for key in omit:
        properties.pop(key)
    return EntityState(
        entity_id=entity_id,
        entity_class="robot",
        track_age=1,
        track_state=track_state,
        properties=properties,
    )


def _observation(timestep, entities, *, dt=1.0):
    return SimpleNamespace(
        timestep=timestep,
        time=float(timestep) * dt,
        perceived_entities={entity.entity_id: entity for entity in entities},
    )


def _catalog(*, qacc_c=None):
    return ComponentCatalog.from_entities([_entity(1, qacc_c=qacc_c)])


def _spec_file(tmp_path, backend):
    if backend == "rtamt" and not rtamt_available():
        pytest.skip("RTAMT C++ backend is unavailable")
    path = tmp_path / f"specs-{backend}.yaml"
    path.write_text(SPEC_TEMPLATE.format(backend=backend))
    return str(path)


def _context(dt=1.0, *, entity_id=1, require_confirmed=True):
    return BatchEpisodeContext(
        dt=dt,
        catalog=ComponentCatalog.from_entities([_entity(entity_id)]),
        require_confirmed=require_confirmed,
    )


def _start_scalar(spec_file, context):
    engine = MonitorEngine.load(spec_file)
    engine.start_episode(
        dt=context.dt,
        capabilities=context.capabilities,
        catalog=context.catalog,
        require_confirmed=context.require_confirmed,
        skip_unsatisfiable=context.skip_unsatisfiable,
    )
    return engine


def _result_without_timing(result):
    return replace(result, update_seconds=0.0)


def _engine_contract(engine):
    return {
        "spec_names": tuple(engine.spec_names),
        "manager_names": tuple(engine.managers),
        "monitor_counts": engine.monitor_counts(),
        "manager_specs": engine.manager_specs(),
        "skipped_specs": engine.skipped_specs(),
        "spec_provenance": engine.spec_provenance,
        "specification_metadata": engine.specification_metadata(),
        "extraction_scope": engine.extraction_scope(),
        "unmatched_slots": engine.unmatched_slots(),
        "formulas": {
            name: (
                manager.formula.name,
                manager.formula.window_steps,
                tuple(parameter.param_key for parameter in manager.formula.params),
            )
            for name, manager in engine.managers.items()
        },
        "bindings": {
            name: manager.current_bindings
            for name, manager in engine.managers.items()
        },
    }


def _manager_samples(engine):
    return {
        name: tuple(
            instance.segment_samples
            for instance in manager.active_instances
        )
        for name, manager in engine.managers.items()
    }


def _assert_scalar_equivalent(
    batch, scalar_engines, batch_results, scalar_results
):
    assert len(batch_results) == len(scalar_engines) == len(scalar_results)
    for index, (batch_result, scalar_engine, scalar_result) in enumerate(
        zip(batch_results, scalar_engines, scalar_results)
    ):
        assert _result_without_timing(batch_result) == _result_without_timing(
            scalar_result
        ), index
        assert _engine_contract(batch.engine(index)) == _engine_contract(
            scalar_engine
        ), index


@pytest.mark.parametrize("backend", ["native", "rtamt"])
@pytest.mark.parametrize("batch_size", [0, 1, 4])
def test_batch_sizes_match_independent_scalar_engines_exactly(
    tmp_path, backend, batch_size
):
    spec_file = _spec_file(tmp_path, backend)
    contexts = tuple(
        _context(dt=1.0 / (index + 1), entity_id=index + 1)
        for index in range(batch_size)
    )
    batch = BatchedMonitorEngine.load(spec_file, contexts)
    scalars = [None] * batch_size
    results = ()

    for step in range(4):
        observations = tuple(
            _observation(
                step,
                [_entity(index + 1, qacc_a=1.0 + index, qacc_b=-1.0 - index)],
                dt=contexts[index].dt,
            )
            for index in range(batch_size)
        )
        reset_mask = tuple(step == 0 for _ in range(batch_size))

        results = batch.step_batch(observations, reset_mask)
        scalar_results = []
        for index in range(batch_size):
            if reset_mask[index]:
                scalars[index] = _start_scalar(spec_file, contexts[index])
            scalar = scalars[index]
            assert scalar is not None
            scalar_results.append(scalar.update(observations[index]))

        _assert_scalar_equivalent(batch, scalars, results, scalar_results)

    assert batch.batch_size == batch_size
    if batch_size:
        for index in range(batch_size):
            assert batch.engine(index).monitor_counts()["fan"] == 2
            assert len(
                [
                    name
                    for name in batch.engine(index).managers
                    if name.startswith("fan")
                ]
            ) == 2
            assert len(results[index].evaluations) == 4


def test_dynamic_entities_properties_and_staggered_resets_do_not_leak(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    contexts = (_context(), _context())
    batch = BatchedMonitorEngine.load(spec_file, contexts)
    scalars: List[Optional[MonitorEngine]] = [None, None]

    calls = [
        (
            (
                _observation(0, [_entity(1, qacc_a=1.0, qacc_b=-1.0)]),
                _observation(0, [_entity(1, qacc_a=-2.0, qacc_b=-3.0)]),
            ),
            (True, True),
        ),
        (
            (
                _observation(
                    1,
                    [
                        _entity(1, qacc_a=2.0, qacc_b=-2.0),
                        _entity(2, qacc_a=3.0, qacc_b=-3.0),
                    ],
                ),
                _observation(1, [_entity(1, qacc_a=-2.0, omit=("qacc_b",))]),
            ),
            (False, False),
        ),
        (
            (
                _observation(0, [_entity(1, qacc_a=-4.0, qacc_b=-4.0)]),
                _observation(2, [_entity(1, qacc_a=-2.0, qacc_b=-3.0)]),
            ),
            (True, False),
        ),
        (
            (
                _observation(1, [_entity(9, qacc_a=-5.0, qacc_b=-5.0)]),
                _observation(3, [_entity(1, qacc_a=-2.0, qacc_b=-3.0)]),
            ),
            (False, False),
        ),
    ]

    first_slot_zero_engine = None
    for call_index, (observations, reset_mask) in enumerate(calls):
        results = batch.step_batch(observations, reset_mask)
        scalar_results = []
        for index in range(2):
            if reset_mask[index]:
                scalars[index] = _start_scalar(spec_file, contexts[index])
            scalar = scalars[index]
            assert scalar is not None
            scalar_results.append(scalar.update(observations[index]))
        _assert_scalar_equivalent(batch, scalars, results, scalar_results)

        if call_index == 0:
            first_slot_zero_engine = batch.engine(0)
            assert batch.engine(0) is not batch.engine(1)
            for name in batch.engine(0).managers:
                assert (
                    batch.engine(0).managers[name]
                    is not batch.engine(1).managers[name]
                )
        elif call_index == 2:
            assert batch.engine(0) is not first_slot_zero_engine
            assert {binding for _, binding in results[0].created} == {(1,)}
            assert results[1].resumed

    slot_zero_bindings = set().union(
        *(manager.current_bindings for manager in batch.engine(0).managers.values())
    )
    slot_one_bindings = set().union(
        *(manager.current_bindings for manager in batch.engine(1).managers.values())
    )
    assert slot_zero_bindings == {(9,)}
    assert slot_one_bindings == {(1,)}


def test_slot_reuse_can_replace_the_scalar_episode_context(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    initial_context = _context(dt=1.0)
    batch = BatchedMonitorEngine.load(spec_file, (initial_context,))
    batch.step_batch((_observation(0, [_entity(1)]),), (True,))
    old_window = batch.engine(0).managers["early_fail"].formula.window_steps

    replacement = BatchEpisodeContext(
        dt=0.5,
        catalog=_catalog(qacc_c=0.25),
    )
    observation = _observation(
        0,
        [_entity(7, qacc_a=2.0, qacc_b=-2.0, qacc_c=0.25)],
        dt=0.5,
    )
    result = batch.step_batch(
        (observation,),
        (True,),
        reset_contexts=(replacement,),
    )[0]

    scalar = _start_scalar(spec_file, replacement)
    scalar_result = scalar.update(observation)
    assert _result_without_timing(result) == _result_without_timing(scalar_result)
    assert _engine_contract(batch.engine(0)) == _engine_contract(scalar)
    assert batch.engine(0).monitor_counts()["fan"] == 3
    assert (
        batch.engine(0).managers["early_fail"].formula.window_steps
        != old_window
    )


def test_slot_reset_matches_reused_scalar_engine_counters(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    context = _context()
    batch = BatchedMonitorEngine.load(spec_file, (context,))
    scalar = MonitorEngine.load(spec_file)

    for episode in range(2):
        scalar.start_episode(
            dt=context.dt,
            catalog=context.catalog,
        )
        observation = _observation(0, [_entity(episode + 1)])
        batch_result = batch.step_batch((observation,), (True,))[0]
        scalar_result = scalar.update(observation)

        assert _result_without_timing(batch_result) == _result_without_timing(
            scalar_result
        )
        assert batch_result.binding_recomputes == 4


def test_all_scalar_episode_context_arguments_match(tmp_path):
    path = tmp_path / "context-specs.yaml"
    path.write_text(
        "specs:\n"
        "  - name: visible\n"
        '    formula: "robot_*[qacc_a] <= 0"\n'
        "  - name: absent\n"
        '    formula: "ghost_*[value] <= 0"\n'
    )
    capabilities = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"robot"}),
        components_by_class={"robot": frozenset({"qacc_a", "qacc_b"})},
        component_mode_by_class={"robot": VocabularyMode.CLOSED},
    )
    ghost = EntityState(
        entity_id=2,
        entity_class="ghost",
        track_age=1,
        track_state=TrackState.CONFIRMED,
        properties={"value": 0.0},
    )
    catalog = ComponentCatalog.from_entities([_entity(1), ghost])
    context = BatchEpisodeContext(
        dt=0.5,
        capabilities=capabilities,
        catalog=catalog,
        require_confirmed=False,
        skip_unsatisfiable=True,
    )
    observation = _observation(
        0,
        [_entity(1, track_state=TrackState.TENTATIVE)],
        dt=0.5,
    )
    batch = BatchedMonitorEngine.load(str(path), (context,))
    batch_result = batch.step_batch((observation,), (True,))[0]
    scalar = _start_scalar(str(path), context)
    scalar_result = scalar.update(observation)

    assert _result_without_timing(batch_result) == _result_without_timing(
        scalar_result
    )
    assert _engine_contract(batch.engine(0)) == _engine_contract(scalar)
    assert batch.engine(0).skipped_specs().keys() == {"absent"}
    assert batch_result.evaluations


def test_shared_compile_context_does_not_share_runtime_state(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    shared = BatchEpisodeContext(
        dt=1.0,
        catalog=ComponentCatalog.from_entities([_entity(1), _entity(2)]),
    )
    batch = BatchedMonitorEngine.load(spec_file, (shared, shared))
    results = batch.step_batch(
        (
            _observation(0, [_entity(1, qacc_a=2.0)]),
            _observation(0, [_entity(2, qacc_a=-2.0)]),
        ),
        (True, True),
    )

    assert batch.engine(0) is not batch.engine(1)
    assert results[0] != results[1]
    assert {
        binding
        for manager in batch.engine(0).managers.values()
        for binding in manager.current_bindings
    } == {(1,)}
    assert {
        binding
        for manager in batch.engine(1).managers.values()
        for binding in manager.current_bindings
    } == {(2,)}


@pytest.mark.parametrize("backend", ["native", "rtamt"])
def test_early_and_ordinary_finality_match_scalar_sessions(tmp_path, backend):
    spec_file = _spec_file(tmp_path, backend)
    context = _context()
    batch = BatchedMonitorEngine.load(spec_file, (context,))
    scalar = _start_scalar(spec_file, context)

    first = None
    ordinary_evaluations = []
    for step in range(4):
        observation = _observation(step, [_entity(1)])
        batch_result = batch.step_batch((observation,), (step == 0,))[0]
        scalar_result = scalar.update(observation)
        assert _result_without_timing(batch_result) == _result_without_timing(
            scalar_result
        )
        first = batch_result if first is None else first
        ordinary_evaluations.append(
            next(
                evaluation
                for evaluation in batch_result.evaluations
                if evaluation.spec_name == "ordinary_fail"
            )
        )

    assert first is not None
    first_by_name = {
        evaluation.spec_name: evaluation
        for evaluation in first.evaluations
    }
    assert first_by_name["early_fail"].verdict_final
    assert first_by_name["early_fail"].segment_samples is not None
    assert first_by_name["early_fail"].window_steps is not None
    assert (
        first_by_name["early_fail"].segment_samples
        < first_by_name["early_fail"].window_steps
    )
    assert not first_by_name["ordinary_fail"].verdict_final
    first_ordinary_final = next(
        evaluation
        for evaluation in ordinary_evaluations
        if evaluation.verdict_final
    )
    assert (
        first_ordinary_final.segment_samples
        == first_ordinary_final.window_steps
    )


def test_per_environment_failure_preserves_index_and_scalar_loop_progress(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    contexts = (_context(), _context(), _context())
    batch = BatchedMonitorEngine.load(spec_file, contexts)
    initial = tuple(_observation(0, [_entity(index + 1)]) for index in range(3))
    batch.step_batch(initial, (True, True, True))

    with pytest.raises(BatchStepError) as caught:
        batch.step_batch(
            (
                _observation(1, [_entity(1)]),
                object(),
                _observation(1, [_entity(3)]),
            ),
            (False, False, False),
        )

    assert caught.value.index == 1
    assert caught.value.operation == "update"
    assert isinstance(caught.value.__cause__, AttributeError)
    assert "environment index 1" in str(caught.value)

    segment_samples = [
        min(
            instance.segment_samples
            for manager in batch.engine(index).managers.values()
            for instance in manager.active_instances
        )
        for index in range(3)
    ]
    assert segment_samples == [2, 1, 1]


def test_update_failure_retains_the_same_partial_progress_as_scalar(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    context = _context()
    batch = BatchedMonitorEngine.load(spec_file, (context,))
    scalar = _start_scalar(spec_file, context)
    initial = _observation(0, [_entity(1)])
    batch.step_batch((initial,), (True,))
    scalar.update(initial)
    slot_engine = batch.engine(0)

    invalid_entity = EntityState(
        entity_id=1,
        entity_class="robot",
        track_age=1,
        track_state=TrackState.CONFIRMED,
        properties={"qacc_a": 1.0, "qacc_b": None},  # type: ignore[dict-item]
    )
    invalid = _observation(1, [invalid_entity])
    with pytest.raises(BatchStepError) as batch_error:
        batch.step_batch((invalid,), (False,))
    with pytest.raises(SpecConfigError):
        scalar.update(invalid)

    assert batch_error.value.index == 0
    assert batch_error.value.operation == "update"
    assert isinstance(batch_error.value.__cause__, SpecConfigError)
    assert batch.engine(0) is slot_engine
    assert _manager_samples(batch.engine(0)) == _manager_samples(scalar)
    assert len(set(_manager_samples(batch.engine(0)).values())) > 1


def test_reset_failure_preserves_index_and_invalidates_only_that_slot(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    contexts = (_context(), _context(), _context())
    batch = BatchedMonitorEngine.load(spec_file, contexts)
    initial = tuple(_observation(0, [_entity(index + 1)]) for index in range(3))
    batch.step_batch(initial, (True, True, True))

    impossible = BatchEpisodeContext(
        dt=1.0,
        catalog=ComponentCatalog.from_entities([]),
    )
    with pytest.raises(BatchStepError) as caught:
        batch.step_batch(
            tuple(_observation(1, [_entity(index + 1)]) for index in range(3)),
            (False, True, False),
            reset_contexts=(None, impossible, None),
        )

    assert caught.value.index == 1
    assert caught.value.operation == "reset"
    assert batch.engine(0)
    with pytest.raises(RuntimeError, match="slot 1 has not started"):
        batch.engine(1)
    assert batch.engine(2)
    slot_zero_samples = min(
        instance.segment_samples
        for manager in batch.engine(0).managers.values()
        for instance in manager.active_instances
    )
    slot_two_samples = min(
        instance.segment_samples
        for manager in batch.engine(2).managers.values()
        for instance in manager.active_instances
    )
    assert slot_zero_samples == 2
    assert slot_two_samples == 1

    recovered = batch.step_batch(
        tuple(_observation(2, [_entity(index + 1)]) for index in range(3)),
        (False, True, False),
    )
    assert len(recovered) == 3
    assert batch.engine(1).monitor_counts()["fan"] == 2


def test_batch_input_validation_is_atomic(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    batch = BatchedMonitorEngine.load(spec_file, (_context(), _context()))
    initial = (_observation(0, [_entity(1)]), _observation(0, [_entity(2)]))
    batch.step_batch(initial, (True, True))

    before = tuple(_engine_contract(batch.engine(index)) for index in range(2))
    with pytest.raises(ValueError, match="2 observations.*1 reset flags"):
        batch.step_batch(initial, (False,))
    with pytest.raises(TypeError, match=r"reset_mask\[1\].*bool"):
        batch.step_batch(initial, (False, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"numpy\.bool"):
        batch.step_batch(initial, (False, np.bool_(False)))
    with pytest.raises(TypeError, match=r"reset_contexts\[0\].*builtins\.dict"):
        batch.step_batch(
            initial,
            (True, False),
            reset_contexts=({}, None),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"reset_contexts\[0\].*False"):
        batch.step_batch(
            initial,
            (False, False),
            reset_contexts=(_context(dt=0.5), None),
        )
    after = tuple(_engine_contract(batch.engine(index)) for index in range(2))
    assert after == before


def test_reset_fork_failure_is_indexed_and_invalidates_slot(
    tmp_path, monkeypatch
):
    spec_file = _spec_file(tmp_path, "native")
    batch = BatchedMonitorEngine.load(spec_file, (_context(),))
    batch.step_batch((_observation(0, [_entity(1)]),), (True,))

    def fail_fork():
        raise MemoryError("simulated fork failure")

    monkeypatch.setattr(batch._prototype, "_fork_unstarted", fail_fork)
    with pytest.raises(BatchStepError) as caught:
        batch.step_batch((_observation(1, [_entity(1)]),), (True,))

    assert caught.value.index == 0
    assert caught.value.operation == "reset"
    assert isinstance(caught.value.__cause__, MemoryError)
    with pytest.raises(RuntimeError, match="slot 0 has not started"):
        batch.engine(0)


def test_unstarted_slot_failure_is_indexed(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    batch = BatchedMonitorEngine.load(spec_file, (_context(),))

    with pytest.raises(BatchStepError) as caught:
        batch.step_batch((_observation(0, [_entity(1)]),), (False,))

    assert caught.value.index == 0
    assert caught.value.operation == "update"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_constructor_contexts_and_engine_indices_are_validated(tmp_path):
    spec_file = _spec_file(tmp_path, "native")
    with pytest.raises(TypeError, match=r"contexts\[0\].*builtins\.dict"):
        BatchedMonitorEngine.load(spec_file, ({},))  # type: ignore[arg-type]

    batch = BatchedMonitorEngine.load(spec_file, (_context(),))
    with pytest.raises(IndexError, match="index -1.*batch size 1"):
        batch.engine(-1)
    with pytest.raises(IndexError, match="index 1.*batch size 1"):
        batch.engine(1)
