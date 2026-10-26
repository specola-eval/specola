"""Candidate evidence cannot become ordinary observations or output artifacts."""

import asyncio
from contextvars import copy_context
from dataclasses import replace
from threading import Thread
from typing import Dict

import pytest

from specola.conformance.core_runner import EvidenceIdentity
from specola.conformance.extraction import SourceAddress, SurfaceClassification, SurfaceDisposition, manifest_digest
from specola.conformance.provider import ExtractionGuardError, ExtractionProvider, PreparedExtraction
from specola.conformance.surface import ApiMember, MemberKind, surface_digest
from specola.conformance.verification import VerificationSubject, _candidate_scope, current_verification, require_ordinary_execution
from specola.query_config import QueryConfiguration
from test_extraction_instances import _ProviderAdapter, _provider_fixture


CONFIGURATION = EvidenceIdentity("independent fixture scene declaration", "c" * 64)


class CandidateProvider(ExtractionProvider):
    def guard_source(self, adapter):
        assert isinstance(adapter, CandidateAdapter)
        adapter.events.append("source")

    def guard_runtime(self, adapter):
        assert isinstance(adapter, CandidateAdapter)
        adapter.events.append("runtime")

    def prepare(self, adapter, *, reset):
        assert isinstance(adapter, CandidateAdapter)
        surface, results, manifest, population, _ = _provider_fixture()
        evidence = surface.root_census
        surface = replace(surface, members=surface.members + (
            ApiMember("native", "unresolved_query", MemberKind.CALLABLE, "unresolved_query()"),
        ))
        results = replace(results, producer_surface_digest=surface_digest(surface))
        manifest = replace(manifest, producer_surface_digest=surface_digest(surface), classifications=manifest.classifications + (
            SurfaceClassification(SourceAddress("native.unresolved_query"), SurfaceDisposition.UNRESOLVED,
                                  "fixture query semantics remain unresolved", (evidence,)),
        ))
        population = replace(population, producer_surface_sha256=surface_digest(surface),
                             manifest_sha256=manifest_digest(manifest),
                             episode_epoch=adapter.extraction_episode_epoch,
                             configuration_epoch=adapter.query_configuration_epoch)
        prepared = PreparedExtraction(surface, results, manifest, population, lambda: None,
                                      configuration_identity=CONFIGURATION)
        adapter.prepared = prepared
        return prepared


class CandidateAdapter(_ProviderAdapter):
    _extraction_provider = CandidateProvider()
    prepared: PreparedExtraction

    def _get_observation_impl(self):
        observation = super()._get_observation_impl()
        population = self.prepared.population
        assert observation.extraction is not None
        extraction = replace(observation.extraction,
                             producer_surface_sha256=population.producer_surface_sha256,
                             manifest_sha256=population.manifest_sha256,
                             episode_epoch=population.episode_epoch,
                             configuration_epoch=population.configuration_epoch,
                             verification_only=True)
        if self.damage == "mode":
            extraction = replace(extraction, verification_only=False)
        if self.damage == "legacy":
            observation.perceived_entities[11].properties["undeclared"] = 7.0
        return replace(observation, extraction=extraction)


def _subject():
    surface, _, _, _, _ = _provider_fixture()
    return VerificationSubject(CandidateAdapter, surface.producer, CONFIGURATION, QueryConfiguration.empty())


def test_resolved_candidate_quantities_run_without_closing_other_members():
    adapter = CandidateAdapter()
    with pytest.raises(ValueError, match="unresolved"):
        adapter.reset()
    assert "read" not in adapter.events
    with _candidate_scope(_subject()):
        observation = adapter.reset()
        assert observation.extraction is not None
        assert observation.extraction.verification_only is True
        assert len(observation.extraction.samples) == 2
        assert adapter.prepared.manifest.classifications[-1].disposition is SurfaceDisposition.UNRESOLVED
    with pytest.raises(ExtractionGuardError, match="verification_scope_changed"):
        adapter.get_observation()


@pytest.mark.parametrize("damage", ("reader", "legacy", "mode"))
def test_candidate_scope_keeps_value_and_output_checks_active(damage):
    adapter = CandidateAdapter(damage)
    with _candidate_scope(_subject()):
        with pytest.raises((ValueError, ExtractionGuardError)):
            adapter.reset()
    assert adapter.last_observation is None


@pytest.mark.parametrize("field", ("adapter_class", "producer", "configuration", "queries"))
def test_candidate_subject_mismatch_rejects_before_read(field):
    from specola.adapters.fake import _MockAdapter

    subject = _subject()
    replacement = {
        "adapter_class": _MockAdapter,
        "producer": replace(subject.producer, version="different"),
        "configuration": EvidenceIdentity("other scene", "d" * 64),
        "queries": QueryConfiguration.from_document({
            "schema": "specola.queries.v1", "populations": [],
            "instances": [{"id": "other", "binding": "other", "arguments": {}, "operands": {}}],
        }),
    }[field]
    adapter = CandidateAdapter()
    with _candidate_scope(replace(subject, **{field: replacement})):
        with pytest.raises(ExtractionGuardError, match="verification_subject_mismatch"):
            adapter.reset()
    assert "read" not in adapter.events


def test_scope_does_not_cross_a_copied_thread_context():
    failures = []
    with _candidate_scope(_subject()):
        copied = copy_context()

        def probe():
            try:
                copied.run(current_verification)
            except ExtractionGuardError as error:
                failures.append(error.status)

        thread = Thread(target=probe)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert failures == ["verification_execution_owner_changed"]


def test_scope_does_not_cross_an_inherited_asyncio_task():
    async def parent():
        with _candidate_scope(_subject()):
            async def child():
                with pytest.raises(ExtractionGuardError, match="verification_execution_owner_changed"):
                    current_verification()
            await asyncio.create_task(child())
    asyncio.run(parent())


def test_scope_restores_on_failure_and_refuses_ordinary_execution():
    with pytest.raises(RuntimeError, match="fixture failure"):
        with _candidate_scope(_subject()):
            with pytest.raises(ExtractionGuardError, match="verification_only"):
                require_ordinary_execution()
            raise RuntimeError("fixture failure")
    assert current_verification() is None
    require_ordinary_execution()


@pytest.fixture
def specs_file(tmp_path):
    path = tmp_path / "candidate.yaml"
    path.write_text('specs:\n  - name: velocity\n    formula: "body_*[velocity_x] <= 2"\n')
    return str(path)


def test_candidate_monitoring_uses_public_paths_and_returns_flagged_memory_record(specs_file, tmp_path):
    from specola.session import Monitor

    adapter = CandidateAdapter()
    with _candidate_scope(_subject()):
        session = Monitor(specs_file, results_dir=None).watch(adapter)
        result = session.step()
        assert result.rho("velocity") == -1.0
        assert adapter.last_observation is not None
        assert adapter.last_observation.extraction is not None
        assert adapter.last_observation.extraction.verification_only is True
    with pytest.raises(ExtractionGuardError, match="verification_scope_changed"):
        session.step()
    record = session.end()
    assert record.meta["verification_only"] is True
    assert record.path is None
    assert set(path.name for path in tmp_path.iterdir()) == {"candidate.yaml"}


@pytest.mark.parametrize("output_request", ("results", "publication", "other_adapter"))
def test_candidate_monitor_refuses_ordinary_outputs_before_reset(specs_file, tmp_path, output_request):
    from specola.adapters.fake import _MockAdapter
    from specola.session import Monitor

    adapter = CandidateAdapter()
    events = []
    monitor = Monitor(specs_file, results_dir=str(tmp_path / "results") if output_request == "results" else None)
    hook = (lambda *args: events.append("published")) if output_request == "publication" else None
    with _candidate_scope(_subject()):
        with pytest.raises(ExtractionGuardError):
            monitor.watch(_MockAdapter() if output_request == "other_adapter" else adapter, _reset_hook=hook)
    assert adapter.events == []
    assert events == []
    assert not (tmp_path / "results").exists()


def test_ordinary_session_cannot_finalize_inside_candidate_scope(specs_file, tmp_path):
    from specola.session import Monitor

    session = Monitor(specs_file, results_dir=str(tmp_path / "results")).watch(_ProviderAdapter())
    with _candidate_scope(_subject()):
        with pytest.raises(ExtractionGuardError):
            session.end()
    assert not session.ended
    assert not (tmp_path / "results").exists()


def test_direct_session_construction_cannot_restore_candidate_metadata_as_ordinary(specs_file, tmp_path):
    from specola.session import Monitor, Session

    monitor = Monitor(specs_file, results_dir=None)
    with pytest.raises(ExtractionGuardError, match="verification_only"):
        Session(monitor._engine, _ProviderAdapter(), meta={"verification_only": True},
                episode_id="unscoped", path=tmp_path / "results.parquet")
    assert not (tmp_path / "results.parquet").exists()


@pytest.mark.parametrize("module", ("run_monitor", "check_specs", "discover"))
def test_ordinary_command_rejects_candidate_scope_before_argument_parsing(module, monkeypatch):
    import importlib
    import sys

    monkeypatch.setattr(sys, "argv", ["specola", "--invalid-argument"])
    command = importlib.import_module("specola." + module)
    with _candidate_scope(_subject()):
        with pytest.raises(ExtractionGuardError, match="verification_only"):
            command.main()


def test_candidate_scope_refuses_an_envelope_free_adapter_before_reset():
    from specola.adapters.fake import _MockAdapter

    adapter = _MockAdapter()
    with _candidate_scope(replace(_subject(), adapter_class=_MockAdapter)):
        with pytest.raises(ExtractionGuardError, match="verification_provider_required"):
            adapter.reset()
    assert adapter.last_observation is None


@pytest.mark.parametrize("kind", ("results", "recording"))
@pytest.mark.parametrize("scoped", (False, True))
def test_ordinary_writers_refuse_candidate_outputs_before_creating_files(tmp_path, kind, scoped):
    from contextlib import nullcontext
    from specola.dashboard.recording import RecordingWriter
    from specola.results_io import save_rows

    with _candidate_scope(_subject()) if scoped else nullcontext():
        with pytest.raises(ExtractionGuardError, match="verification_only"):
            metadata: Dict[str, object] = {} if scoped else {"verification_only": True}
            if kind == "results":
                save_rows([{"rho": -1.0}], tmp_path / "ordinary.parquet", metadata)
            else:
                RecordingWriter(tmp_path / "recording", metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("factory_failure", (False, True))
def test_existing_raw_runner_retains_candidate_mode_even_without_observations(factory_failure):
    import json
    from specola.conformance.verification import run_extraction_candidates
    from test_core_conformance_runner import _case, _subject as certification_subject

    def factory():
        if factory_failure:
            raise RuntimeError("candidate fixture factory failure")
        return CandidateAdapter()

    artifact = run_extraction_candidates("candidate fixture", certification_subject(CandidateAdapter),
                                         _subject(), (_case("candidate", factory),))
    document = json.loads(artifact.canonical_json)
    payload = document["payload"]
    assert payload["verification_only"] is True
    assert payload["verification_subject"]["configuration"]["sha256"] == CONFIGURATION.sha256
    assert payload["core"]["case_denominator"] == ["candidate"]
    serialized_observations = payload["core"]["executions"][0]["observations"]
    if factory_failure:
        assert serialized_observations == []
    else:
        assert len(serialized_observations) == 3
        assert all(json.loads(row["canonical_json"])["extraction"]["verification_only"] is True
                   for row in serialized_observations)
    assert current_verification() is None
