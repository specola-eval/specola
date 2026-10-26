"""Guarded query authoring over declarations and native owner name tables.

The document is separate from discovery observations. Missing author arguments
cannot bypass producer custody, classification closure or query contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Callable, Optional, Tuple

from ..query_config import QueryConfiguration
from .audit import contract_surface_audit
from .core_runner import EvidenceIdentity
from .extraction import ExtractionManifest, ResultSurface, _evidence, _provenance, _text, _tuple
from .provider import ExtractionGuardError, guard_extraction_source
from .query_bindings import (
    CallableParameterSurface, ParameterRole, QueryBindingRecord,
    QueryConfigurationRequired, validate_query_arguments, validate_query_contract,
)
from .surface import ProducerSurface


@dataclass(frozen=True)
class AuthoringOwnerDomain:
    """Native indices and optional native names, without quantity values.

    The provider obtains this population from declarations and scene tables,
    independently of extraction output. An unnamed owner retains None; names
    need not be unique in producers that allow duplicate declarations.
    """

    resolver_id: str
    owner_kind: str
    population_rule: str
    owners: Tuple[Tuple[Tuple[int, ...], Optional[str]], ...]
    provenance: Tuple[EvidenceIdentity, ...]
    empty_reason: str = ""


@dataclass(frozen=True)
class PreparedAuthoring:
    """A closed contract and guarded native name populations for one subject."""

    surface: ProducerSurface
    results: ResultSurface
    manifest: ExtractionManifest
    bindings: Tuple[QueryBindingRecord, ...]
    declarations: Tuple[CallableParameterSurface, ...]
    queries: QueryConfiguration
    configuration_epoch: int
    configuration_identity: EvidenceIdentity
    owner_domains: Tuple[AuthoringOwnerDomain, ...]
    guard_after_authoring: Callable[[], None]

    def validate(self, adapter) -> None:
        self.validate_subject(adapter)
        _evidence(self.configuration_identity, "authoring configuration identity")
        missing = validate_query_contract(self.surface, self.results, self.manifest,
            bindings=self.bindings, declarations=self.declarations, require_closed=True)
        if missing:
            raise ExtractionGuardError("unbound_callable_quantities", missing)
        expected = {(quantity.owner_resolver_id, quantity.owner_kind) for quantity in self.manifest.quantities}
        for binding in self.bindings:
            for parameter in binding.parameters:
                if parameter.role is ParameterRole.OWNER:
                    assert parameter.owner_kind is not None
                    expected.add((binding.owner_resolver_id, parameter.owner_kind))
        _tuple(self.owner_domains, AuthoringOwnerDomain, "authoring owner domains")
        actual = set()
        for domain in self.owner_domains:
            for field in ("resolver_id", "owner_kind", "population_rule"):
                _text(getattr(domain, field), field)
            _provenance(domain.provenance, "authoring native owner population")
            address = (domain.resolver_id, domain.owner_kind)
            if address in actual:
                raise ValueError("duplicate authoring owner domain")
            actual.add(address)
            if type(domain.owners) is not tuple or type(domain.empty_reason) is not str:
                raise ValueError("invalid authoring owner population")
            if domain.owners and domain.empty_reason:
                raise ValueError("populated authoring domain declares an empty reason")
            if not domain.owners:
                _text(domain.empty_reason, "empty authoring owner population reason")
            indices = set()
            for owner in domain.owners:
                if (type(owner) is not tuple or len(owner) != 2 or type(owner[0]) is not tuple
                        or any(type(index) is not int for index in owner[0])
                        or (owner[1] is not None and type(owner[1]) is not str)):
                    raise ValueError("invalid authoring native owner address or name")
                if owner[0] in indices:
                    raise ValueError("duplicate authoring native owner address")
                indices.add(owner[0])
        if actual != expected:
            raise ValueError("authoring owner domains differ from quantity and query declarations")
        if not callable(self.guard_after_authoring):
            raise TypeError("authoring requires a post-capture native subject guard")

    def validate_subject(self, adapter) -> None:
        if (type(self.queries) is not QueryConfiguration or self.queries != adapter.query_configuration
                or type(self.configuration_epoch) is not int
                or self.configuration_epoch != adapter.query_configuration_epoch):
            raise ExtractionGuardError("authoring_subject_mismatch", ("query configuration changed during schema capture",))


class SchemaOnlyDiscovery(QueryConfigurationRequired):
    """A loader result that contains schemas instead of an observation."""

    def __init__(self, document):
        self.document = document
        super().__init__(tuple((item["binding_id"], tuple(item["arguments"]))
                               for item in document["missing_arguments"]))


def prepare_schema_authoring(adapter):
    """Return a schema document only for a guarded missing-argument subject.

    No observation preparation, reader or reset runs here. Fully configured
    subjects return None so the caller can start an ordinary guarded reset.
    Concrete providers retain responsibility for independent native population
    and configuration oracles; these structural checks cannot establish them.
    """
    from .verification import require_ordinary_execution

    require_ordinary_execution()
    provider = guard_extraction_source(adapter, reset=True)
    if provider is None:
        return None
    adapter._bind_extraction_subject(reset=True)
    prepared = provider.prepare_authoring(adapter)
    if prepared is None:
        return None
    if type(prepared) is not PreparedAuthoring:
        raise TypeError("extraction provider returned an invalid authoring subject")
    prepared.validate(adapter)
    required = None
    try:
        validate_query_arguments(prepared.queries, prepared.bindings, prepared.declarations,
                                 tuple(quantity.quantity_id for quantity in prepared.manifest.quantities))
    except QueryConfigurationRequired as error:
        required = error.missing
    prepared.guard_after_authoring()
    guard_extraction_source(adapter, reset=False)
    prepared.validate_subject(adapter)
    if required is None:
        return None
    contract = contract_surface_audit(prepared.surface, prepared.results, prepared.manifest,
                                     bindings=prepared.bindings, declarations=prepared.declarations)
    # The audit projection's metadata-only labels do not describe this guarded
    # name-table capture. It still carries no behavioral evidence or row pass.
    contract["native_identity_status"] = "provider_guarded"
    contract["contract_status"] = "closed_declarations"
    contract["owner_population_status"] = "native_names_bound"
    return {
        "schema": "specola.extraction-authoring.v1", "schema_only": True,
        "query_configuration_required": True, "status": "query_configuration_required",
        "subject": {"actual_adapter_class": type(adapter).__module__ + ":" + type(adapter).__qualname__,
                    "configuration": asdict(prepared.configuration_identity),
                    "query_configuration": prepared.queries.document(),
                    "configuration_epoch": prepared.configuration_epoch},
        "missing_arguments": [{"binding_id": binding, "arguments": list(arguments)}
                              for binding, arguments in required],
        "contract": contract,
        "owner_domains": json.loads(json.dumps([asdict(domain) for domain in prepared.owner_domains], allow_nan=False)),
    }


def render_schema_authoring(document) -> str:
    return "Specola query authoring\nschema_only: true\n" + json.dumps(
        document, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"
