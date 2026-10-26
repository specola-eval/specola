"""Monitor/Session facade lifecycle on the mock adapter (no sim deps)."""
import logging
import math

import numpy as np
import pytest

from specola.adapter import MonitorObservation
from specola.adapters.fake import _MockAdapter
from specola.entities import EntityState, TrackState
from specola.session import Monitor

SPECS = """\
specs:
  - name: mass_cap
    formula: "G[0,1s](object_*[mass] <= 0.3)"
  - name: microwave_check
    formula: "G[0,1s](microwave_*[door_angle] <= 0.1)"
"""
# mock scene: object 1 mass=0.25 (pass), object 2 mass=0.40 (fail);
# no 'microwave' class -> not_applicable. dt=0.05 so G[0,1s] = 20 steps.


@pytest.fixture
def specs_file(tmp_path):
    f = tmp_path / "specs.yaml"
    f.write_text(SPECS)
    return str(f)


def _run_episode(monitor, steps=30, **watch_kwargs):
    session = monitor.watch(_MockAdapter(), **watch_kwargs)
    result = None
    for _ in range(steps):
        result = session.step()
    return session, result


def test_lifecycle_and_verdicts(specs_file, tmp_path):
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session, result = _run_episode(monitor, meta={"task": "mock", "seed": 3})
    assert result.violated()                       # mass 0.40 > 0.3, final
    assert result.violated("mass_cap")
    assert not result.violated("microwave_check")  # skipped, never violates
    assert result.rho("mass_cap") is not None and result.rho("mass_cap") < 0
    assert result.violations and result.violations[0].spec == "mass_cap"

    record = session.end(success=False)
    assert record.status == "completed"
    assert record.success is False
    assert record.specs["mass_cap"].verdict == "fail"
    assert record.specs["microwave_check"].verdict == "not_applicable"
    assert record.path is not None and record.path.exists()


def test_watch_does_not_reset_the_simulator(specs_file):
    """The facade starts an adapter episode around caller-prepared state."""

    class ResetSentinel:
        def reset(self):
            raise AssertionError("Monitor.watch() called env.reset()")

    adapter = _MockAdapter()
    adapter.env = ResetSentinel()
    monitor = Monitor(specs_file, results_dir=None)

    session = monitor.watch(adapter)
    session.step()
    session.end()


def test_parquet_schema(specs_file, tmp_path):
    pd = pytest.importorskip("pandas")
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "r"))
    session, _ = _run_episode(monitor, meta={"train_step": 30000})
    session.end(success=True)
    df = pd.read_parquet(session.end().path)  # idempotent end()
    for col in ["step", "time", "spec", "authored_spec", "binding", "rho",
                "suspended", "verdict_final", "window_steps", "episode_id",
                "status", "success", "not_applicable", "train_step"]:
        assert col in df.columns, col
    na = df[df["not_applicable"]]
    assert set(na["spec"]) == {"microwave_check"}
    assert (na["step"] == -1).all()
    assert set(df["train_step"]) == {30000}


def test_meta_key_collision_errors_at_watch(specs_file):
    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(ValueError, match="spec"):
        monitor.watch(_MockAdapter(), meta={"spec": "oops"})


def test_caller_cannot_replace_extraction_provenance(specs_file):
    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(ValueError, match="extraction_status"):
        monitor.watch(_MockAdapter(), meta={"extraction_status": "complete"})


def test_result_records_unaudited_status_from_episode_start(specs_file, tmp_path, monkeypatch):
    from specola import session as session_module

    class ThirdParty(_MockAdapter):
        pass

    written = []

    def capture(rows, path, metadata, **kwargs):
        written.append(dict(metadata))
        return None

    monkeypatch.setattr(session_module, "save_rows", capture)
    adapter = ThirdParty()
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session = monitor.watch(adapter)
    session.step()
    # A later adapter reset cannot rewrite the episode's captured provenance.
    adapter.reset()
    session.end()
    assert written[0]["extraction_status"] == "unaudited_third_party_provider"
    legacy = monitor.watch(_MockAdapter())
    legacy.step()
    legacy.end()
    assert "extraction_status" not in written[1]


def test_meta_key_collision_with_report_derived_column(specs_file):
    """Finding I3: a meta key named after a report-DERIVED column (not a
    row column) must also be rejected at watch(), naming the key. Pre-fix
    _validate_meta only checked CORE_COLUMNS, so meta={"verdict": ...}
    passed watch() and later corrupted the report.
    """
    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(ValueError, match="verdict"):
        monitor.watch(_MockAdapter(), meta={"verdict": "x"})


def test_reserved_meta_keys_superset_of_core_columns():
    from specola import results_io
    assert results_io.CORE_COLUMNS <= results_io.RESERVED_META_KEYS


class _CollidingLabelAdapter(_MockAdapter):
    """Two objects whose display labels both strip to the same category.

    mujoco-style labels "body  (category)" strip to the category in
    session._binding_label, so object 1 and object 2 share the display
    label "widget" while keeping distinct entity IDs (hence distinct
    binding_keys). Object 1 (mass 0.40) violates object_*[mass] <= 0.3 and
    leaves before the episode ends; object 2 (mass 0.25) passes and stays.
    A verdict keyed on the shared label keeps only object 2's later,
    positive last-final row and would wrongly report "pass".
    """

    entity_labels = {
        1: "obj_one  (widget)",
        2: "obj_two  (widget)",
    }

    def _observation(self) -> MonitorObservation:
        time = self._step * self.DT
        entities = {}
        if self._step < 25:   # failing object present early, then leaves
            entities[1] = EntityState(
                entity_id=1, entity_class="object",
                position=np.zeros(3, dtype=float),
                velocity=np.zeros(3, dtype=float),
                track_age=self._step, track_state=TrackState.CONFIRMED,
                properties={"mass": 0.40},
            )
        entities[2] = EntityState(   # passing object present throughout
            entity_id=2, entity_class="object",
            position=np.ones(3, dtype=float),
            velocity=np.zeros(3, dtype=float),
            track_age=self._step, track_state=TrackState.CONFIRMED,
            properties={"mass": 0.25},
        )
        return MonitorObservation(
            timestep=self._step, time=time, perceived_entities=entities,
        )


def test_colliding_binding_labels_keep_fail_verdict(specs_file):
    """Finding C2 (session side): a failing binding sharing a display label
    with a passing one must NOT be absorbed. mass_cap is G[0,1s] (20 steps
    at dt=0.05); object 1's last final rho (-0.1) settles by step 23 before
    it leaves, object 2's positive last final rho (+0.05) runs to step 39.
    Keyed on the shared label "widget" the later positive row wins and the
    verdict is a false "pass"; keyed on binding_key it is "fail".
    """
    monitor = Monitor(specs_file, results_dir=None)
    session = monitor.watch(_CollidingLabelAdapter())
    for _ in range(40):
        session.step()
    record = session.end()
    assert record.specs["mass_cap"].verdict == "fail"


def test_results_dir_none_writes_nothing(specs_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor = Monitor(specs_file, results_dir=None)
    session, _ = _run_episode(monitor, steps=5)
    record = session.end()
    assert record.path is None
    assert not (tmp_path / "results").exists()


def test_write_failure_leaves_the_record_with_no_path(specs_file, tmp_path):
    """
    The record survives a failed write by design (``end()`` stores it
    before writing), but it must not survive holding a path to a file
    that was never created: a caller that reads ``record.path`` would
    open nothing, or a stale file from an earlier run.
    """
    from specola import session as session_module

    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session, _ = _run_episode(monitor, steps=5)

    def _boom(*args, **kwargs):
        raise OSError("No space left on device")

    original = session_module.save_rows
    session_module.save_rows = _boom
    try:
        with pytest.raises(OSError):
            session.end(success=True)
    finally:
        session_module.save_rows = original

    # The record survived the failure...
    record = session.end()
    assert record.status == "completed"
    assert record.success is True
    assert record.steps == 5
    assert set(record.specs) == {"mass_cap", "microwave_check"}
    # ...but no longer claims a file that does not exist.
    assert record.path is None


def test_record_path_is_none_when_nothing_was_written(specs_file, tmp_path):
    """
    ``save_rows`` reports None when it wrote no file, and the record must
    carry that through rather than naming a file that does not exist.

    The record must not fall back to probing the filesystem for the path
    it *would* have written either: an earlier episode's file can occupy
    that name, and claiming it would attribute another episode's results
    to this one.
    """
    from specola import session as session_module

    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session, _ = _run_episode(monitor, steps=5)

    # A file already sitting at the path this episode was going to use.
    stale = session._path
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("someone else's results")

    original = session_module.save_rows
    session_module.save_rows = lambda *args, **kwargs: None
    try:
        record = session.end(success=True)
    finally:
        session_module.save_rows = original

    assert record.path is None
    assert stale.read_text() == "someone else's results"


def test_record_keeps_the_path_it_actually_wrote(specs_file, tmp_path):
    # The guard above must not throw away a real write: the normal path
    # still reports the file, including when save_rows redirects to a
    # different suffix than the one requested.
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session, _ = _run_episode(monitor, steps=5)

    record = session.end(success=True)

    assert record.path is not None
    assert record.path.exists()


def test_stale_session_auto_truncated(specs_file):
    monitor = Monitor(specs_file, results_dir=None)
    first = monitor.watch(_MockAdapter())
    first.step()
    second = monitor.watch(_MockAdapter())   # auto-ends first
    assert first.end().status == "truncated"
    second.end()


def test_context_manager_truncates_on_exception(specs_file):
    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(RuntimeError):
        with monitor.watch(_MockAdapter()) as session:
            session.step()
            raise RuntimeError("policy crashed")
    assert session.end().status == "truncated"


def test_adapter_kwargs_rejected_with_explicit_adapter(specs_file):
    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(TypeError):
        monitor.watch(_MockAdapter(), task="t")


def test_bad_specs_fail_at_construction(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text('specs:\n  - name: x\n    formula: "G[0,1s]("\n')
    with pytest.raises(ValueError):
        Monitor(str(f))


def test_golden_parquet_schema_superset(specs_file, tmp_path):
    """Refactor guard: every historical column survives with its name."""
    pd = pytest.importorskip("pandas")
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "r"))
    session, _ = _run_episode(monitor)
    df = pd.read_parquet(session.end().path)
    historical = {"step", "time", "spec", "binding", "rho",
                  "suspended", "verdict_final", "window_steps"}
    assert historical <= set(df.columns)


def test_watch_dt_override_reaches_engine(specs_file):
    """Regression: watch(dt=...) must actually change the compiled spec
    window, not silently fall back to the adapter's own dt.

    _MockAdapter.dt is 0.05s; mass_cap is G[0,1s], so the default window
    is 1s/0.05s = 20 steps. A dt=0.1 override must halve that to 10 steps
    (1s/0.1s) -- this is exactly the value that used to reach
    MonitorEngine.start_episode() via run_monitor's own dt variable
    before this facade existed.
    """
    monitor = Monitor(specs_file, results_dir=None)
    session = monitor.watch(_MockAdapter())
    assert session.engine.managers["mass_cap"].formula.window_steps == 20
    session.end()

    monitor2 = Monitor(specs_file, results_dir=None)
    session2 = monitor2.watch(_MockAdapter(), dt=0.1)
    assert session2.engine.managers["mass_cap"].formula.window_steps == 10
    session2.end()


def test_cli_phase_hook_tracks_reset_compile_finalize_and_write(
    specs_file, tmp_path
):
    phases = []
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results"))
    session = monitor.watch(_MockAdapter(), _phase_hook=phases.append)
    session.step()
    session.end()

    assert phases == [
        "adapter reset and scene discovery",
        "specification compilation",
        "result finalization",
        "result write",
    ]


class _DynamicClassAdapter(_MockAdapter):
    """Mock scene plus a declared class whose population stays empty."""

    @property
    def capabilities(self):
        from specola.capabilities import AdapterCapabilities, VocabularyMode

        return AdapterCapabilities(
            spatial_dims=3,
            class_mode=VocabularyMode.CLOSED,
            supported_classes=frozenset(self.supported_classes)
            | {"contact"},
            runtime_populated_classes=frozenset({"contact"}),
        )


def test_end_warns_about_slots_that_never_matched(tmp_path, caplog):
    """A compiled spec that never binds is reported, not silently
    recorded as undetermined.

    The report is a log record rather than a warning: the default
    warning filter shows one warning per call site, so a campaign loop
    would hear about its first episode only.
    """
    path = tmp_path / "dynamic.yaml"
    path.write_text(
        "specs:\n"
        "  - name: ghost_contact\n"
        '    formula: "contact_*[contact_point_force_max] > 0"\n'
    )
    monitor = Monitor(str(path), results_dir=None)
    session = monitor.watch(_DynamicClassAdapter())
    session.step()

    with caplog.at_level(logging.WARNING, logger="specola.session"):
        session.end()

    records = [item for item in caplog.records if "matched no entity" in item.getMessage()]
    assert len(records) == 1
    assert "ghost_contact" in records[0].getMessage()



def test_watch_accepts_an_adapter_with_the_one_argument_scope_override(specs_file):
    class LegacyAdapter(_MockAdapter):
        """A third-party adapter written against the one-argument hook."""

        def set_extraction_scope(self, scope):
            self.seen_scope = scope
            super().set_extraction_scope(scope)

    adapter = LegacyAdapter()
    monitor = Monitor(specs_file, results_dir=None)
    session = monitor.watch(adapter)
    assert adapter.seen_scope is not None
    session.step()
