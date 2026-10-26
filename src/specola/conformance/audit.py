"""Declaration-only extraction reports, separate from observations and results.

Metadata selection never imports an adapter entry point. A saved declaration
contract describes its pinned producer; it does not establish the identity,
availability or readiness of a currently running native producer.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..adapter_enrollment import (
    EnrollmentError, bundled_adapter_declarations, bundled_provider_enrollments,
    required_extraction_declarations,
)
from .extraction import (
    manifest_digest, validate_result_surface,
)
from .extraction_catalogue import SPECOLA_EXTRACTION_V1_CATALOGUE
from .ledger import ObligationResult
from .query_bindings import query_binding_document, validate_query_contract
from .surface import surface_document


def _source_reconciliation(declarations):
    # This is the checkout containing the imported Specola source. A wheel
    # without that checkout has no source file to cross-check here.
    path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if not path.is_file():
        return {"status": "source_unavailable", "path": None}
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        content = path.read_bytes()
        project = tomllib.loads(content.decode("utf-8"))["project"]
        if project.get("name") != "specola":
            return {"status": "source_unavailable", "path": None}
        expected = project["entry-points"]["specola.adapters"]
        actual = {item.name: item.entry_point_value for item in declarations}
        return {"status": "matched" if actual == expected else "drift", "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_declarations": expected, "installed_declarations": actual}
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        return {"status": "unresolved", "path": str(path),
                "error": type(error).__name__ + ": " + str(error)}


def contract_surface_audit(surface, results, manifest, *, bindings=(), declarations=()):
    """Project each validated draft quantity and result schema without values."""
    missing = validate_query_contract(surface, results, manifest, bindings=bindings,
                                      declarations=declarations, require_closed=False)
    candidates = validate_result_surface(surface, results)
    declared = {item.callable_id: item for item in declarations}
    quantities = json.loads(json.dumps([asdict(quantity) for quantity in manifest.quantities]))
    for quantity, encoded in zip(manifest.quantities, quantities):
        for outcome, encoded_outcome in zip(quantity.outcomes, encoded["outcomes"]):
            encoded_outcome["native_schema"] = outcome.native_schema.document()
    queries = []
    for binding in bindings:
        document = query_binding_document(binding, declared[binding.callable_id])
        document["binding"]["arguments"] = binding.arguments.document()
        document["binding"]["population_arguments"] = (
            binding.population_arguments.document() if binding.population_arguments is not None else None)
        queries.append(document)
    return {
        "native_identity_status": "not_measured", "contract_status": "declarations_only",
        "surface": surface_document(surface), "manifest_sha256": manifest_digest(manifest),
        "candidate_addresses": [asdict(address) for address in candidates],
        "classifications": [asdict(item) for item in manifest.classifications],
        "quantities": quantities, "query_bindings": queries,
        "unbound_callable_quantities": list(missing),
        "owner_population_status": "native_scene_not_bound",
    }


def adapter_surface_audit(name: str, *, task=None, robot=None, target_only=True,
                          dt_override=None, requested_adapter_modules=()):
    """Report metadata enrollment before resolving any adapter or producer."""
    declarations = bundled_adapter_declarations()
    required = required_extraction_declarations(declarations)
    selected = next((item for item in declarations if item.name == name), None)
    enrollment = None
    unresolved = []
    if selected is None:
        status = "unresolved_adapter_subject"
        unresolved.append("requested name has no bundled distribution declaration; the actual class was not constructed")
    elif selected not in required:
        status = "excluded_from_programme"
    else:
        try:
            enrollment = bundled_provider_enrollments((selected,))[0]
        except EnrollmentError as error:
            status = "unresolved_implementation_enrollment"
            unresolved.append(str(error))
        else:
            status = "bundled_adapter_pending"
            unresolved.append("packaged producer declarations and accepted extraction evidence have not been loaded")
    source = _source_reconciliation(declarations)
    if source["status"] in ("drift", "unresolved"):
        unresolved.append("installed adapter declarations have not reconciled with the available source project")
    obligations = []
    if selected in required:
        obligations = [{"obligation_id": identity, "result": ObligationResult.UNEXECUTED.value,
                        "reason": "no accepted extraction evidence loaded for this runtime subject"}
                       for identity in SPECOLA_EXTRACTION_V1_CATALOGUE.definition_denominator]
    return {
        "schema": "specola.extraction-audit.v1", "audit_only": True, "status": status,
        "subject": {"requested_adapter": name, "task": task,
                    "requested_factory_arguments": {"robot": robot, "target_only": target_only,
                                                    "dt_override": dt_override},
                    "registration": asdict(selected) if selected is not None else None,
                    "enrollment": asdict(enrollment) if enrollment is not None else None,
                    "actual_adapter_class": None},
        "registry": {"bundled": [asdict(item) for item in declarations],
                     "required": [asdict(item) for item in required],
                     "source_reconciliation": source},
        "requested_adapter_modules": list(requested_adapter_modules),
        "native_identity_status": "not_measured", "contracts": [],
        "obligations": obligations, "unresolved": unresolved,
    }


def render_surface_audit(document) -> str:
    """Render an explicit audit label and its complete diagnostic document."""
    return "Specola extraction audit\naudit_only: true\n" + json.dumps(
        document, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"
