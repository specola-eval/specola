"""
MonitorEngine: parse once, compile per episode, structured results.

Also the regression test for design-spec defect 10: the catalog and
class vocabulary are finally consulted in the real pipeline — an
impossible spec fails at episode start, not silently at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from specola.capabilities import reset_adapter
from specola.components import ComponentCatalog
from specola.engine import MonitorEngine
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.results import EvaluationStatus


def entity(eid, value):
    return EntityState(
        entity_id=eid, entity_class="sig",
        position=np.zeros(3), velocity=np.zeros(3),
        track_age=1, track_state=TrackState.CONFIRMED,
        properties={"a": float(value)},
    )


def obs(t, values):
    ents = {eid: entity(eid, v) for eid, v in values.items()}
    return SimpleNamespace(timestep=t, time=float(t), perceived_entities=ents)


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "specs.yaml"
    path.write_text(
        "specs:\n"
        "- name: hold\n"
        "  formula: \"G[0,3](sig_*[a] >= thresh)\"\n"
        "  constants: {thresh: 0.0}\n"
    )
    return str(path)


def test_engine_lifecycle_and_statuses(spec_file):
    engine = MonitorEngine.load(spec_file)
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    engine.start_episode(dt=1.0, catalog=catalog)

    r0 = engine.update(obs(0, {1: 1.0}))
    assert r0.created and r0.evaluations[0].status is EvaluationStatus.WARMING_UP
    assert not r0.evaluations[0].verdict_final

    engine.update(obs(1, {1: 1.0}))
    r2 = engine.update(obs(2, {1: 1.0}))
    assert r2.evaluations[0].status is EvaluationStatus.ACTIVE
    assert r2.evaluations[0].verdict_final
    assert r2.evaluations[0].robustness == pytest.approx(1.0)

    r3 = engine.update(obs(3, {}))         # entity gone → suspended
    assert r3.evaluations[0].status is EvaluationStatus.SUSPENDED
    assert r3.evaluations[0].robustness is None

    r4 = engine.update(obs(4, {1: 2.0}))   # returns → resumed, rewarming
    assert r4.resumed
    assert r4.evaluations[0].status is EvaluationStatus.WARMING_UP
    assert r4.evaluations[0].robustness == pytest.approx(2.0)


def test_impossible_spec_fails_at_episode_start(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "specs:\n- name: bad\n  formula: \"G[0,3](ghost_*[a] >= 0)\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    with pytest.raises(SpecConfigError, match="ghost"):
        engine.start_episode(dt=1.0, catalog=catalog)


def test_same_engine_recompiles_per_episode_dt(spec_file):
    engine = MonitorEngine.load(spec_file)
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )

    engine.start_episode(dt=1.0, catalog=catalog)
    w1 = engine.managers["hold"].formula.window_steps
    engine.start_episode(dt=0.5, catalog=catalog)
    w2 = engine.managers["hold"].formula.window_steps

    assert w1 == w2 == 3  # step bounds are dt-independent


def test_reset_adapter_packages_discovery():
    from specola.cli_common import _MockAdapter

    result = reset_adapter(_MockAdapter())
    assert result.capabilities.spatial_dims == 3
    assert "object" in result.discovered_catalog.classes
    assert result.initial_observation.perceived_entities


def test_open_capabilities_defer_unknown_references(spec_file, tmp_path):
    """§5.6.1: open vocabularies compile-and-wait instead of rejecting."""
    from specola.capabilities import AdapterCapabilities

    path = tmp_path / "open.yaml"
    path.write_text(
        "specs:\n"
        "- name: ghost_class\n"
        "  formula: \"G[0,3](ghost_*[a] >= 0)\"\n"
        "- name: future_pin\n"
        "  formula: \"sig_9[a] >= 0\"\n"
        "- name: future_component\n"
        "  formula: \"sig_*[not_yet] >= 0\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    open_caps = AdapterCapabilities(spatial_dims=3)  # all modes OPEN

    engine.start_episode(dt=1.0, catalog=catalog, capabilities=open_caps)
    result = engine.update(obs(0, {1: 1.0}))
    assert result.evaluations == ()  # deferred: no bindings yet, no error


def test_static_class_slot_without_candidate_is_rejected(tmp_path):
    """A wildcard slot on a static class must not compile-and-wait.

    The class vocabulary is CLOSED and ``sig`` is not runtime-populated,
    so no entity exposing ``not_yet`` can appear later in the episode.
    """
    from specola.capabilities import AdapterCapabilities, VocabularyMode
    from specola.entities import UnboundStaticSlotError

    path = tmp_path / "static.yaml"
    path.write_text(
        "specs:\n"
        "- name: dead_slot\n"
        "  formula: \"sig_*[not_yet] >= 0\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    caps = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"sig", "contact"}),
        runtime_populated_classes=frozenset({"contact"}),
    )

    with pytest.raises(UnboundStaticSlotError) as excinfo:
        engine.start_episode(
            dt=1.0,
            catalog=catalog,
            capabilities=caps,
            skip_unsatisfiable=True,
        )

    message = str(excinfo.value)
    assert "sig" in message and "not_yet" in message
    assert engine.skipped_specs() == {}  # never treated as inapplicable


@pytest.mark.parametrize("selector", ("missing", "missing_*", "{missing,absent}"))
@pytest.mark.parametrize("closed", (True, False))
def test_static_class_rejection_covers_every_selector_spelling(
    tmp_path, selector, closed
):
    """Spelling does not change the fact a static class has no candidate.

    The exact key, the glob and the list all name a slot no entity of a
    static class can ever satisfy, so all three earn the subtype
    ``x15_check`` records as unbindable. With an OPEN class vocabulary
    the same specs keep section 5.6.1 compile-and-wait.
    """
    from specola.capabilities import AdapterCapabilities, VocabularyMode
    from specola.entities import UnboundStaticSlotError

    path = tmp_path / "selector.yaml"
    path.write_text(
        "specs:\n"
        "- name: dead_slot\n"
        '  formula: "object_*[{}] >= 0"\n'.format(selector)
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities({
        1: EntityState(
            entity_id=1, entity_class="object",
            position=np.zeros(3), velocity=np.zeros(3),
            track_age=1, track_state=TrackState.CONFIRMED,
            properties={"mass": 1.0},
        ),
    })
    caps = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED if closed else VocabularyMode.OPEN,
        supported_classes=frozenset({"object"}) if closed else frozenset(),
    )

    if not closed:
        # Unchanged behaviour: the trivial key defers under section 5.6.1
        # and a selector matching nothing keeps the generic refusal, since
        # an OPEN class vocabulary can still gain an entity.
        try:
            engine.start_episode(dt=1.0, catalog=catalog, capabilities=caps)
        except UnboundStaticSlotError:
            pytest.fail("an open class vocabulary must not raise the subtype")
        except SpecConfigError:
            pass
        return

    with pytest.raises(UnboundStaticSlotError):
        engine.start_episode(
            dt=1.0,
            catalog=catalog,
            capabilities=caps,
            skip_unsatisfiable=True,
        )
    assert engine.skipped_specs() == {}


def test_runtime_populated_class_still_defers(tmp_path):
    """A class that legitimately starts empty keeps §5.6.1 deferral."""
    from specola.capabilities import AdapterCapabilities, VocabularyMode

    path = tmp_path / "dynamic.yaml"
    path.write_text(
        "specs:\n"
        "- name: waiting\n"
        "  formula: \"contact_*[x] >= 0\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    caps = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"sig", "contact"}),
        runtime_populated_classes=frozenset({"contact"}),
    )

    engine.start_episode(dt=1.0, catalog=catalog, capabilities=caps)
    assert engine.update(obs(0, {1: 1.0})).evaluations == ()


@pytest.mark.parametrize("runtime_populated", (frozenset(), frozenset({"region"})))
def test_declared_component_schema_still_rejects_a_static_class(
    tmp_path, runtime_populated
):
    """A declared CLOSED component schema does not license deferral.

    The schema stands in for a live instance only while the population
    can still grow. On a static class it cannot, so a slot the schema
    covers is rejected exactly as an open-vocabulary one is; the same
    class declared runtime-populated keeps §5.6.1 deferral.
    """
    from specola.capabilities import AdapterCapabilities, VocabularyMode
    from specola.entities import UnboundStaticSlotError

    path = tmp_path / "declared.yaml"
    path.write_text(
        "specs:\n"
        "- name: dead_region\n"
        "  formula: \"region_*[x] >= 0\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    caps = AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"sig", "region"}),
        components_by_class={"region": frozenset({"x"})},
        component_mode_by_class={"region": VocabularyMode.CLOSED},
        runtime_populated_classes=runtime_populated,
    )

    if runtime_populated:
        engine.start_episode(dt=1.0, catalog=catalog, capabilities=caps)
        assert engine.update(obs(0, {1: 1.0})).evaluations == ()
        return

    with pytest.raises(UnboundStaticSlotError) as excinfo:
        engine.start_episode(
            dt=1.0,
            catalog=catalog,
            capabilities=caps,
            skip_unsatisfiable=True,
        )
    assert "region" in str(excinfo.value) and "'x'" in str(excinfo.value)
    assert engine.skipped_specs() == {}


def test_failed_start_episode_invalidates_previous_episode(spec_file, tmp_path):
    engine = MonitorEngine.load(spec_file)
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    engine.start_episode(dt=1.0, catalog=catalog)
    engine.update(obs(0, {1: 1.0}))

    bad = MonitorEngine.load(spec_file)  # noqa: F841 (fresh engine ok)
    bad_path = tmp_path / "bad2.yaml"
    bad_path.write_text(
        "specs:\n- name: bad\n  formula: \"G[0,3](ghost_*[a] >= 0)\"\n"
    )
    engine._specs = MonitorEngine.load(str(bad_path))._specs
    with pytest.raises(SpecConfigError):
        engine.start_episode(dt=1.0, catalog=catalog)
    with pytest.raises(RuntimeError, match="episode"):
        engine.update(obs(1, {1: 1.0}))


def test_unmatched_slots_reports_possible_typos(spec_file, tmp_path):
    from specola.capabilities import AdapterCapabilities

    path = tmp_path / "open2.yaml"
    path.write_text(
        "specs:\n"
        "- name: ok\n"
        "  formula: \"G[0,3](sig_*[a] >= 0)\"\n"
        "- name: typo\n"
        "  formula: \"G[0,3](sgi_*[a] >= 0)\"\n"
    )
    engine = MonitorEngine.load(str(path))
    catalog = ComponentCatalog.from_entities(
        obs(0, {1: 1.0}).perceived_entities
    )
    engine.start_episode(
        dt=1.0, catalog=catalog,
        capabilities=AdapterCapabilities(spatial_dims=3),  # OPEN modes
    )
    for t in range(3):
        engine.update(obs(t, {1: 1.0}))

    unmatched = engine.unmatched_slots()
    assert "ok" not in unmatched
    assert [p.entity_class for p in unmatched["typo"]] == ["sgi"]


def test_iter_spec_entries_non_mapping_entry_reports_error(tmp_path):
    """A non-mapping spec entry (``specs: [hello]``) must not crash with a
    raw AttributeError from ``entry.get(...)`` — it gets a per-entry error
    tuple instead, same as any other malformed entry (final-review finding
    5), preserving ``iter_spec_entries``'s one-triple-per-entry contract.
    """
    path = tmp_path / "non_mapping.yaml"
    path.write_text("specs: [hello]\n")

    entries = MonitorEngine.iter_spec_entries(str(path))

    assert len(entries) == 1
    name, entry, error = entries[0]
    assert name == "<entry 1>"
    assert entry is None
    assert error == "Spec entry must be a mapping"


def test_load_aggregates_non_mapping_entry_error(tmp_path):
    """``load()``'s all-or-nothing contract still raises — with the new
    error folded into the aggregated message rather than a traceback.
    """
    path = tmp_path / "non_mapping.yaml"
    path.write_text("specs: [hello]\n")

    with pytest.raises(ValueError, match="Spec entry must be a mapping"):
        MonitorEngine.load(str(path))
