"""Audit reports never construct a simulator or become observations."""

from dataclasses import asdict, replace
import json
import sys

import pytest

from specola.adapter_enrollment import (
    AdapterDeclaration, bundled_adapter_declarations, required_extraction_declarations,
)
from specola.conformance.extraction_catalogue import SPECOLA_EXTRACTION_V1_CATALOGUE
from specola.conformance.provider import ExtractionGuardError


def test_audit_uses_the_distribution_denominator_without_importing_adapters(monkeypatch):
    from specola.conformance.audit import adapter_surface_audit
    from specola import registry

    declarations = bundled_adapter_declarations()
    required = required_extraction_declarations(declarations)

    def forbidden(*args, **kwargs):
        pytest.fail("audit must not resolve or import an adapter")

    monkeypatch.setattr(registry, "resolve", forbidden)
    monkeypatch.setattr(registry, "load_adapter_module", forbidden)
    document = adapter_surface_audit(required[0].name)
    assert document["audit_only"] is True
    assert document["schema"] == "specola.extraction-audit.v1"
    assert document["registry"]["bundled"] == [asdict(item) for item in declarations]
    assert document["registry"]["required"] == [asdict(item) for item in required]
    assert document["subject"]["actual_adapter_class"] is None
    assert document["native_identity_status"] == "not_measured"
    assert "entities" not in document and "schema_version" not in document
    assert {record["obligation_id"] for record in document["obligations"]} == set(
        SPECOLA_EXTRACTION_V1_CATALOGUE.definition_denominator)
    assert {record["result"] for record in document["obligations"]} == {"unexecuted"}


def test_unused_added_registration_remains_in_the_audit_denominator(monkeypatch):
    from specola.conformance import audit

    declarations = tuple(sorted(bundled_adapter_declarations() + (
        AdapterDeclaration("unused_new_native", "specola.adapters.unused_new_native"),
    )))
    monkeypatch.setattr(audit, "bundled_adapter_declarations", lambda: declarations)
    document = audit.adapter_surface_audit("unused_new_native")
    assert asdict(declarations[-1]) in document["registry"]["required"]
    assert document["status"] == "unresolved_implementation_enrollment"
    assert document["registry"]["source_reconciliation"]["status"] == "drift"


def test_unknown_registration_does_not_claim_an_unaudited_actual_class():
    from specola.conformance.audit import adapter_surface_audit

    document = adapter_surface_audit("not_an_installed_registration")
    assert document["status"] == "unresolved_adapter_subject"
    assert document["subject"]["actual_adapter_class"] is None


def test_contract_audit_preserves_each_quantity_schema_and_result_component():
    from specola.conformance.audit import contract_surface_audit
    from test_extraction_manifest import _fixture

    surface, results, manifest = _fixture()
    document = contract_surface_audit(surface, results, manifest)
    assert document["native_identity_status"] == "not_measured"
    assert document["candidate_addresses"] == [asdict(item.source) for item in manifest.classifications]
    assert {item["quantity_id"] for item in document["quantities"]} == {
        item.quantity_id for item in manifest.quantities}
    for quantity, encoded in zip(manifest.quantities, document["quantities"]):
        assert encoded["sources"] == [asdict(source) for source in quantity.sources]
        assert encoded["outcomes"][0]["native_schema"] == quantity.outcomes[0].native_schema.document()
    with pytest.raises(ValueError):
        contract_surface_audit(surface, replace(results, results=results.results[:-1]), manifest)


@pytest.mark.parametrize(("options", "expected"), [
    ((), {"robot": None, "target_only": True, "dt_override": None}),
    (("--robot", "requested_robot", "--all-regions", "--dt", "0.025"),
     {"robot": "requested_robot", "target_only": False, "dt_override": 0.025}),
])
def test_audit_cli_returns_before_explicit_module_import_factory_or_reset(monkeypatch, capsys, options, expected):
    from specola import cli_common, discover, registry
    from specola.adapter import SimulatorAdapter

    def forbidden(*args, **kwargs):
        pytest.fail("audit must not execute adapter or simulator lifecycle")

    monkeypatch.setattr(registry, "load_adapter_module", forbidden)
    monkeypatch.setattr(registry, "resolve", forbidden)
    monkeypatch.setattr(cli_common, "load_sim", forbidden)
    monkeypatch.setattr(SimulatorAdapter, "reset", forbidden)
    selected = required_extraction_declarations(bundled_adapter_declarations())[0].name
    monkeypatch.setattr(sys, "argv", ["specola discover", "--sim", selected, "--audit-surface",
                                      "--adapter-module", "must_not_import", "--format", "json", *options])
    discover.main()
    document = json.loads(capsys.readouterr().out)
    assert document["audit_only"] is True
    assert document["requested_adapter_modules"] == ["must_not_import"]
    assert document["subject"]["requested_factory_arguments"] == expected
    assert document["native_identity_status"] == "not_measured"


@pytest.mark.parametrize("option", ["--web", "--ui", "--class", "--entity", "--raw"])
def test_audit_refuses_ordinary_publication_or_value_filters_before_output(monkeypatch, tmp_path, option):
    from specola import discover

    arguments = ["specola discover", "--audit-surface", "--output", str(tmp_path / "report")]
    arguments += [option] if option in ("--web", "--raw") else [option, "plain" if option == "--ui" else "*"]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit, match="audit"):
        discover.main()
    assert list(tmp_path.iterdir()) == []


def _ordinary_document():
    from specola.adapters.fake import _MockAdapter
    from specola.discovery_model import build_discovery_model
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_serialize import discovery_document

    adapter = _MockAdapter()
    observation = adapter.reset()
    model = build_discovery_model(adapter, observation.perceived_entities, simulator="mock",
                                  task=None, effective_dt=adapter.dt)
    return discovery_document(filter_discovery(model, DiscoveryQuery()))


@pytest.mark.parametrize("flag", ["audit_only", "schema_only", "verification_only"])
def test_reports_cannot_be_restored_as_ordinary_discovery(flag):
    from specola.discovery_restore import DiscoveryRestoreError, discovery_model_from_document

    document = _ordinary_document()
    document[flag] = True
    with pytest.raises(DiscoveryRestoreError, match="ordinary"):
        discovery_model_from_document(document)


@pytest.mark.parametrize("flag", ["audit_only", "schema_only"])
@pytest.mark.parametrize("writer", ["results", "recording"])
def test_report_metadata_cannot_create_ordinary_artifacts(tmp_path, flag, writer):
    from specola.dashboard.recording import RecordingWriter
    from specola.results_io import save_rows

    with pytest.raises(ExtractionGuardError, match=flag):
        if writer == "results":
            save_rows([{"rho": 1.0}], tmp_path / "ordinary.parquet", {flag: True})
        else:
            RecordingWriter(tmp_path / "recording", {flag: True})
    assert list(tmp_path.iterdir()) == []
