"""start_episode(skip_unsatisfiable=True): class-gap specs skip, others raise."""
import pytest

from specola.adapters.fake import _MockAdapter
from specola.capabilities import reset_adapter
from specola.engine import MonitorEngine
from specola.entities import SpecConfigError, UnknownClassError


GOOD = """\
specs:
  - name: joint_speed
    formula: "G[0,1s](robot_*[qvel_joint1] <= 2.0)"
  - name: microwave_check
    formula: "G[0,1s](microwave_*[door_angle] <= 0.1)"
"""


def _engine_and_reset(tmp_path, text):
    spec_file = tmp_path / "specs.yaml"
    spec_file.write_text(text)
    engine = MonitorEngine.load(str(spec_file))
    reset = reset_adapter(_MockAdapter())
    return engine, reset


def test_unknown_class_skipped_when_flagged(tmp_path):
    engine, reset = _engine_and_reset(tmp_path, GOOD)
    engine.start_episode(
        dt=0.05,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
        skip_unsatisfiable=True,
    )
    assert "microwave_check" in engine.skipped_specs()
    assert "joint_speed" not in engine.skipped_specs()
    # The surviving spec compiled and is serviceable.
    result = engine.update(reset.initial_observation)
    assert all(ev.spec_name != "microwave_check" for ev in result.evaluations)


def test_unknown_class_raises_by_default(tmp_path):
    engine, reset = _engine_and_reset(tmp_path, GOOD)
    with pytest.raises(UnknownClassError):
        engine.start_episode(
            dt=0.05,
            capabilities=reset.capabilities,
            catalog=reset.discovered_catalog,
        )


def test_unknown_class_error_is_specconfigerror():
    assert issubclass(UnknownClassError, SpecConfigError)


def test_manager_specs_maps_keys_to_authored_names(tmp_path):
    engine, reset = _engine_and_reset(
        tmp_path,
        'specs:\n  - name: joint_speed\n'
        '    formula: "G[0,1s](robot_*[qvel_joint1] <= 2.0)"\n',
    )
    engine.start_episode(
        dt=0.05,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )
    mapping = engine.manager_specs()
    assert set(mapping) == set(engine.managers)
    assert set(mapping.values()) == {"joint_speed"}


def test_manager_specs_maps_fanout_keys_to_one_authored_name(tmp_path):
    """A selector spec fans into several managers; all of them must map
    back to the one authored spec name (not just the non-fanning case
    covered above)."""
    from specola.adapter import PropertyMeta
    from specola.components import ComponentCatalog
    from specola.entities import EntityState, TrackState

    joint = PropertyMeta(group="joint angular acceleration")
    entities = {
        1: EntityState(
            entity_id=1,
            entity_class="robot",
            position=None,
            velocity=None,
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties={"qacc_a": 30.0, "qacc_b": 60.0},
        )
    }
    catalog = ComponentCatalog.from_entities(
        entities, metadata={1: {"qacc_a": joint, "qacc_b": joint}}
    )

    spec_file = tmp_path / "specs.yaml"
    spec_file.write_text(
        'specs:\n  - name: qacc\n    formula: "robot_*[qacc_*] <= 50"\n'
    )
    engine = MonitorEngine.load(str(spec_file))
    engine.start_episode(dt=1.0, catalog=catalog)

    assert len(engine.managers) == 2  # one per matched component
    mapping = engine.manager_specs()
    assert set(mapping) == set(engine.managers)
    assert set(mapping.values()) == {"qacc"}


def test_accessors_reset_after_failed_start_episode(tmp_path):
    """A start_episode call that raises must not leak the previous
    episode's skipped_specs()/manager_specs() — the same reset-then-raise
    contract engine.managers already has via self._managers = None."""
    engine, reset = _engine_and_reset(tmp_path, GOOD)

    # Episode 1: succeeds with one spec skipped.
    engine.start_episode(
        dt=0.05,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
        skip_unsatisfiable=True,
    )
    assert "microwave_check" in engine.skipped_specs()
    assert engine.manager_specs()  # non-empty: joint_speed's manager

    # Episode 2: same specs, default flag — raises on microwave_check.
    with pytest.raises(UnknownClassError):
        engine.start_episode(
            dt=0.05,
            capabilities=reset.capabilities,
            catalog=reset.discovered_catalog,
        )

    # The failed call must not leave episode 1's accessor data behind.
    assert engine.skipped_specs() == {}
    assert engine.manager_specs() == {}
    with pytest.raises(RuntimeError):
        engine.managers


def test_monitor_counts_reset_after_failed_start_episode(tmp_path):
    """monitor_counts() has the same reset-then-raise contract as
    skipped_specs()/manager_specs(): it is documented as "empty before
    the first start_episode() call", and a call that raises must leave it
    empty too rather than serving the previous episode's counts.

    Same latent bug 2bbf900 fixed for self._skipped and
    self._spec_of_manager, which that commit explicitly left open here.
    """
    engine, reset = _engine_and_reset(tmp_path, GOOD)

    # Episode 1: succeeds with one spec skipped, so counts are populated.
    engine.start_episode(
        dt=0.05,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
        skip_unsatisfiable=True,
    )
    assert engine.monitor_counts() == {"joint_speed": 1}

    # Episode 2: same specs, default flag — raises on microwave_check.
    with pytest.raises(UnknownClassError):
        engine.start_episode(
            dt=0.05,
            capabilities=reset.capabilities,
            catalog=reset.discovered_catalog,
        )

    assert engine.monitor_counts() == {}
