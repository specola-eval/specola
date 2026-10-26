"""Quantity coverage declarations beside the historical field-based ledger.

Validation checks the declared joins, not the truth of native semantics. An
independently audited result census and executed native witnesses remain
separate obligations. A valid manifest is not an extraction-completeness pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

from .core_runner import EvidenceIdentity
from .ledger import ExactnessKind
from ..extraction_observation import StructuredMetadata
from .surface import ProducerSurface, SurfaceValidationError, surface_digest
from .surface import _evidence as _surface_evidence, _text as _surface_text
from .surface import _keys


if TYPE_CHECKING:
    from .query_bindings import ArgumentSchema


class ExtractionValidationError(SurfaceValidationError):
    """An extraction declaration has an incomplete or inconsistent join."""


def _text(value, label):
    try:
        _surface_text(value, label)
    except SurfaceValidationError as error:
        raise ExtractionValidationError(str(error)) from error


def _evidence(value, label):
    try:
        _surface_evidence(value, label)
    except SurfaceValidationError as error:
        raise ExtractionValidationError(str(error)) from error


class SurfaceDisposition(str, Enum):
    """Observation decisions, separate from syntactic producer membership."""

    EXPOSED = "exposed"
    REPRESENTED_ELSEWHERE = "represented_elsewhere"
    NOT_OBSERVATION = "not_observation"
    UNRESOLVED = "unresolved"


class NonObservationReason(str, Enum):
    """The accepted reasons for not treating an API member as an observation."""

    LIFECYCLE_OR_CONTROL_OPERATION = "lifecycle_or_control_operation"
    SOFTWARE_PLUMBING_OR_UTILITY = "software_plumbing_or_utility"
    BENCHMARK_CRITERION_EVALUATION = "benchmark_criterion_evaluation"


@dataclass(frozen=True, order=True)
class SourceAddress:
    """An API member, or one independently declared result beneath it."""

    member_id: str
    result_path: str = ""


@dataclass(frozen=True)
class ResultDeclaration:
    """Producer result schema, before any decision about exposure."""

    source: SourceAddress
    native_schema: str
    availability: str
    provenance: Tuple[EvidenceIdentity, ...]


@dataclass(frozen=True)
class ResultSurface:
    """An independently sourced result-path census tied to an API snapshot."""

    producer_surface_digest: str
    results: Tuple[ResultDeclaration, ...]
    census_provenance: Tuple[EvidenceIdentity, ...]


@dataclass(frozen=True)
class SurfaceClassification:
    """Exactly one disposition for each API member and declared result."""

    source: SourceAddress
    disposition: SurfaceDisposition
    rationale: str
    provenance: Tuple[EvidenceIdentity, ...]
    non_observation_reason: Optional[NonObservationReason] = None


@dataclass(frozen=True)
class QuantityOutcome:
    """One source-backed availability outcome and its typed result layout."""

    outcome_id: str
    availability: str
    native_schema: ArgumentSchema
    provenance: Tuple[EvidenceIdentity, ...]


@dataclass(frozen=True)
class QuantityRecord:
    """Native quantity semantics, output addressing and reader custody.

    Sources retain provenance, including transitive aliases. Direct coverage
    edges identify materialized native sources independently of those aliases.
    """

    quantity_id: str
    sources: Tuple[SourceAddress, ...]
    native_meaning: str
    owner_resolver_id: str
    index_domain: str
    unit: str
    frame: str
    exactness: ExactnessKind
    availability: str
    freshness: str
    transformation: str
    output_sink: str
    property_pattern: str
    metadata_resolver_id: str
    reader_id: str
    provenance: Tuple[EvidenceIdentity, ...]
    owner_kind: str
    outcomes: Tuple[QuantityOutcome, ...]
    approximation_relation: Optional[str] = None


@dataclass(frozen=True)
class CoverageLink:
    """A declared lossless mapping to another source or an exact quantity.

    A direct quantity target identifies materialization; a source target keeps
    the lossless alias relation. The mapping's truth requires native evidence.
    This record cannot close a
    source with an approximate quantity merely because that approximation is
    labeled. Additional approximate quantities retain their source links in
    ``QuantityRecord.sources`` without discharging coverage obligations.
    """

    source: SourceAddress
    mapping: str
    provenance: Tuple[EvidenceIdentity, ...]
    target_source: Optional[SourceAddress] = None
    target_quantity: Optional[str] = None


@dataclass(frozen=True)
class ExtractionManifest:
    """Versioned API/result-to-quantity join; no stored completeness verdict."""

    identity: str
    producer_surface_digest: str
    classifications: Tuple[SurfaceClassification, ...]
    quantities: Tuple[QuantityRecord, ...]
    coverage: Tuple[CoverageLink, ...]
    schema: str = "specola.extraction-manifest.v1"


def manifest_document(manifest: ExtractionManifest) -> dict:
    """Return the declared relation without introducing a stored verdict."""
    if type(manifest) is not ExtractionManifest:
        raise ExtractionValidationError("manifest must be an ExtractionManifest")
    return asdict(manifest)


def _record_values(record_type, document):
    return dict(_keys(document, (field.name for field in fields(record_type)), record_type.__name__))


def _restore_items(document, restore):
    if type(document) is not list:
        raise ExtractionValidationError("serialized record population must be a list")
    return tuple(restore(item) for item in document)


def _restore_evidence(document):
    return _restore_items(document, lambda item: EvidenceIdentity(**_record_values(EvidenceIdentity, item)))


def _restore_address(document):
    return SourceAddress(**_record_values(SourceAddress, document))


def result_surface_document(results: ResultSurface) -> dict:
    """Persist the independent result census without a completeness verdict."""
    if type(results) is not ResultSurface:
        raise ExtractionValidationError("results must be a ResultSurface")
    return {"schema": "specola.result-surface.v1", **asdict(results)}


def result_surface_from_document(surface: ProducerSurface, document: dict) -> ResultSurface:
    """Restore declared results and check their producer source addresses."""
    try:
        raw = dict(_keys(document, ("schema",) + tuple(field.name for field in fields(ResultSurface)),
                         "result surface"))
        if raw.pop("schema") != "specola.result-surface.v1":
            raise ExtractionValidationError("unknown result surface schema")

        def result(item):
            item = _record_values(ResultDeclaration, item)
            item["source"] = _restore_address(item["source"])
            item["provenance"] = _restore_evidence(item["provenance"])
            return ResultDeclaration(**item)

        raw["results"] = _restore_items(raw["results"], result)
        raw["census_provenance"] = _restore_evidence(raw["census_provenance"])
        restored = ResultSurface(**raw)
        validate_result_surface(surface, restored)
        return restored
    except (TypeError, ValueError) as error:
        raise ExtractionValidationError(str(error)) from error


def manifest_from_document(surface: ProducerSurface, results: ResultSurface, document: dict,
                           *, require_closed: bool = True) -> ExtractionManifest:
    """Restore exact semantic records and revalidate both coverage directions.

    Draft restoration is explicit and preserves unresolved dispositions. The
    result is a declaration, not permission to execute its readers.
    """
    from .query_bindings import ArgumentSchema

    try:
        raw = _record_values(ExtractionManifest, document)

        def classification(item):
            item = _record_values(SurfaceClassification, item)
            item["source"] = _restore_address(item["source"])
            item["disposition"] = SurfaceDisposition(item["disposition"])
            if item["non_observation_reason"] is not None:
                item["non_observation_reason"] = NonObservationReason(item["non_observation_reason"])
            item["provenance"] = _restore_evidence(item["provenance"])
            return SurfaceClassification(**item)

        def outcome(item):
            item = _record_values(QuantityOutcome, item)
            item["native_schema"] = ArgumentSchema(**_record_values(ArgumentSchema, item["native_schema"]))
            item["provenance"] = _restore_evidence(item["provenance"])
            return QuantityOutcome(**item)

        def quantity(item):
            item = _record_values(QuantityRecord, item)
            item["sources"] = _restore_items(item["sources"], _restore_address)
            item["exactness"] = ExactnessKind(item["exactness"])
            item["provenance"] = _restore_evidence(item["provenance"])
            item["outcomes"] = _restore_items(item["outcomes"], outcome)
            return QuantityRecord(**item)

        def link(item):
            item = _record_values(CoverageLink, item)
            item["source"] = _restore_address(item["source"])
            if item["target_source"] is not None:
                item["target_source"] = _restore_address(item["target_source"])
            item["provenance"] = _restore_evidence(item["provenance"])
            return CoverageLink(**item)

        raw["classifications"] = _restore_items(raw["classifications"], classification)
        raw["quantities"] = _restore_items(raw["quantities"], quantity)
        raw["coverage"] = _restore_items(raw["coverage"], link)
        restored = ExtractionManifest(**raw)
        validate_extraction_manifest(surface, results, restored, require_closed=require_closed)
        return restored
    except (TypeError, ValueError) as error:
        raise ExtractionValidationError(str(error)) from error


def _canonical_document(document) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def manifest_digest(manifest: ExtractionManifest) -> str:
    """Bind the complete semantic relation, including outcome declarations."""
    return hashlib.sha256(_canonical_document(manifest_document(manifest)).encode("utf-8")).hexdigest()


def validate_quantity_outcomes(quantity: QuantityRecord) -> None:
    """Reject absent, duplicate or untyped outcome declarations."""
    from .query_bindings import ArgumentSchema

    _text(quantity.owner_kind, "quantity owner kind")
    _tuple(quantity.outcomes, QuantityOutcome, "quantity outcomes")
    if not quantity.outcomes:
        raise ExtractionValidationError("quantity needs declared outcomes")
    seen = set()
    for outcome in quantity.outcomes:
        _text(outcome.outcome_id, "quantity outcome ID")
        _text(outcome.availability, "quantity outcome availability")
        _provenance(outcome.provenance, "quantity outcome")
        if type(outcome.native_schema) is not ArgumentSchema:
            raise ExtractionValidationError("quantity outcome needs a typed native schema")
        if outcome.outcome_id in seen:
            raise ExtractionValidationError("duplicate quantity outcome")
        seen.add(outcome.outcome_id)


def quantity_metadata(quantity: QuantityRecord) -> StructuredMetadata:
    """Join replay metadata to native result paths and declared outcomes.

    Encode each source address as a JSON pair so a delimiter in a native
    name cannot alias another member or result path. Legacy, non-enrolled
    metadata retains its existing free-text schema representation.
    """
    validate_quantity_outcomes(quantity)
    return StructuredMetadata(
        quantity.quantity_id, quantity.native_meaning,
        _canonical_document({"outcomes": {outcome.outcome_id: outcome.native_schema.document() for outcome in quantity.outcomes}}),
        quantity.unit, quantity.frame, quantity.transformation,
        tuple(_canonical_document([source.member_id, source.result_path]) for source in quantity.sources),
        hashlib.sha256(_canonical_document([asdict(item) for item in quantity.provenance]).encode("utf-8")).hexdigest(),
    )


def _provenance(value: Tuple[EvidenceIdentity, ...], label: str) -> None:
    if type(value) is not tuple or not value:
        raise ExtractionValidationError(label + " needs nonempty evidence")
    for item in value:
        _evidence(item, label)
    if len(set(value)) != len(value):
        raise ExtractionValidationError(label + " has duplicate evidence")


def _address(source: SourceAddress) -> None:
    if type(source) is not SourceAddress:
        raise ExtractionValidationError("source must be a SourceAddress")
    _text(source.member_id, "source member ID")
    if type(source.result_path) is not str or (source.result_path and not source.result_path.strip()):
        raise ExtractionValidationError("result path must be empty or nonempty text")


def _tuple(value, record_type, label):
    if type(value) is not tuple or any(type(item) is not record_type for item in value):
        raise ExtractionValidationError(label + " has the wrong record type")


def _partition(actual, expected, label):
    if len(set(actual)) != len(actual):
        raise ExtractionValidationError(label + " has duplicates")
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ExtractionValidationError(label + " differs: missing=" + str(sorted(missing)) + "; extra=" + str(sorted(extra)))


def validate_result_surface(surface: ProducerSurface, results: ResultSurface) -> Tuple[SourceAddress, ...]:
    """Reconcile a result census to its exact API surface, not to readers."""
    if type(results) is not ResultSurface:
        raise ExtractionValidationError("results must be a ResultSurface")
    if results.producer_surface_digest != surface_digest(surface):
        raise ExtractionValidationError("result census producer surface identity differs")
    _provenance(results.census_provenance, "result census")
    _tuple(results.results, ResultDeclaration, "result declarations")
    member_ids = {member.member_id for member in surface.members}
    addresses = []
    for declaration in results.results:
        _address(declaration.source)
        if declaration.source.member_id not in member_ids or not declaration.source.result_path:
            raise ExtractionValidationError("result declaration needs a known member and nonempty result path")
        _text(declaration.native_schema, "native result schema")
        _text(declaration.availability, "result availability")
        _provenance(declaration.provenance, "result declaration")
        addresses.append(declaration.source)
    if addresses != sorted(set(addresses)):
        raise ExtractionValidationError("result addresses must be sorted and unique")
    return tuple(sorted([SourceAddress(member) for member in member_ids] + addresses))


def validate_extraction_manifest(surface: ProducerSurface, results: ResultSurface, manifest: ExtractionManifest, *, require_closed: bool = True) -> None:
    """Check quantity coverage and source links without granting readiness.

    ``require_closed=False`` validates an audit draft while retaining unresolved
    declarations. It is not an ordinary-operation bypass or release result.
    Runtime readers and the truth of each semantic mapping require independent
    evidence under the extraction catalogue.
    """
    candidates = validate_result_surface(surface, results)
    if type(manifest) is not ExtractionManifest or manifest.schema != "specola.extraction-manifest.v1":
        raise ExtractionValidationError("unknown extraction manifest schema")
    _text(manifest.identity, "manifest identity")
    if manifest.producer_surface_digest != surface_digest(surface):
        raise ExtractionValidationError("manifest producer surface identity differs")
    if type(require_closed) is not bool:
        raise ExtractionValidationError("require_closed must be bool")
    if require_closed:
        for root in surface.roots:
            if root.unresolved_reason:
                raise ExtractionValidationError("unresolved producer root: " + root.name)
    _tuple(manifest.classifications, SurfaceClassification, "classifications")
    _tuple(manifest.quantities, QuantityRecord, "quantities")
    _tuple(manifest.coverage, CoverageLink, "coverage")
    for record in manifest.classifications:
        _address(record.source)
        if type(record.disposition) is not SurfaceDisposition:
            raise ExtractionValidationError("classification disposition has the wrong type")
        _text(record.rationale, "classification rationale")
        _provenance(record.provenance, "classification")
        if record.disposition is SurfaceDisposition.NOT_OBSERVATION:
            if type(record.non_observation_reason) is not NonObservationReason:
                raise ExtractionValidationError("not-observation needs an accepted reason")
        elif record.non_observation_reason is not None:
            raise ExtractionValidationError("observation carries a non-observation reason")
        if require_closed and record.disposition is SurfaceDisposition.UNRESOLVED:
            raise ExtractionValidationError("unresolved producer observation: " + str(record.source))
    _partition(tuple(record.source for record in manifest.classifications), candidates, "classification population")
    classifications = {record.source: record for record in manifest.classifications}
    quantities = {}
    for quantity in manifest.quantities:
        for name in ("quantity_id", "native_meaning", "owner_resolver_id", "index_domain", "unit", "frame",
                     "availability", "freshness", "transformation", "output_sink", "property_pattern", "metadata_resolver_id", "reader_id", "owner_kind"):
            _text(getattr(quantity, name), "quantity " + name)
        _provenance(quantity.provenance, "quantity")
        validate_quantity_outcomes(quantity)
        _tuple(quantity.sources, SourceAddress, "quantity sources")
        for address in quantity.sources:
            _address(address)
        if not quantity.sources or len(set(quantity.sources)) != len(quantity.sources):
            raise ExtractionValidationError("quantity needs unique nonempty source addresses")
        if quantity.quantity_id in quantities:
            raise ExtractionValidationError("duplicate quantity ID: " + quantity.quantity_id)
        if type(quantity.exactness) is not ExactnessKind:
            raise ExtractionValidationError("quantity exactness has the wrong type")
        if quantity.exactness is ExactnessKind.APPROXIMATE:
            _text(quantity.approximation_relation, "approximation relation")
        elif quantity.approximation_relation is not None:
            raise ExtractionValidationError("exact quantity carries an approximation relation")
        for address in quantity.sources:
            if address not in classifications or classifications[address].disposition in (SurfaceDisposition.NOT_OBSERVATION, SurfaceDisposition.UNRESOLVED):
                raise ExtractionValidationError("quantity source is not an exposed or represented observation")
        quantities[quantity.quantity_id] = quantity
    edges = {address: [] for address in candidates}
    seen = set()
    directly_covered = set()
    for link in manifest.coverage:
        _address(link.source)
        if link.source not in classifications:
            raise ExtractionValidationError("coverage source is not in the producer census")
        if classifications[link.source].disposition in (SurfaceDisposition.NOT_OBSERVATION, SurfaceDisposition.UNRESOLVED):
            raise ExtractionValidationError("coverage cannot discharge a non-observation or unresolved source")
        _text(link.mapping, "lossless coverage mapping")
        _provenance(link.provenance, "coverage")
        if (link.target_source is None) == (link.target_quantity is None):
            raise ExtractionValidationError("coverage needs exactly one target")
        if link.target_source is not None:
            _address(link.target_source)
            if link.target_source not in classifications or classifications[link.target_source].disposition in (SurfaceDisposition.NOT_OBSERVATION, SurfaceDisposition.UNRESOLVED):
                raise ExtractionValidationError("coverage has a dangling or non-observation target")
        else:
            _text(link.target_quantity, "coverage quantity")
            quantity = quantities.get(link.target_quantity)
            if quantity is None or link.source not in quantity.sources:
                raise ExtractionValidationError("coverage quantity/source join differs")
            if quantity.exactness is not ExactnessKind.EXACT:
                raise ExtractionValidationError("approximation cannot discharge native result coverage")
            directly_covered.add(quantity.quantity_id)
        key = (link.source, link.target_source, link.target_quantity)
        if key in seen:
            raise ExtractionValidationError("duplicate coverage edge")
        seen.add(key)
        edges[link.source].append(link)
    direct_sources = {link.source for link in manifest.coverage if link.target_quantity is not None}
    for quantity in quantities.values():
        if quantity.exactness is ExactnessKind.EXACT and quantity.quantity_id not in directly_covered:
            raise ExtractionValidationError("exact quantity has no coverage mapping")
        if quantity.exactness is ExactnessKind.APPROXIMATE and not direct_sources.intersection(quantity.sources):
            raise ExtractionValidationError("approximate quantity has no materialized source")

    active, reachable = set(), {}
    for record in manifest.classifications:
        if record.disposition in (SurfaceDisposition.EXPOSED, SurfaceDisposition.REPRESENTED_ELSEWHERE):
            if record.disposition is SurfaceDisposition.EXPOSED and not any(link.target_quantity is not None for link in edges[record.source]):
                raise ExtractionValidationError("exposed source needs a direct quantity mapping")
            pending = [(record.source, False)]
            while pending:
                address, finishing = pending.pop()
                if finishing:
                    targets = set()
                    for link in edges[address]:
                        if link.target_source is not None:
                            targets.update(reachable[link.target_source])
                        else:
                            targets.add(link.target_quantity)
                    reachable[address] = targets
                    active.remove(address)
                    continue
                if address in active:
                    raise ExtractionValidationError("cycle in quantity coverage")
                if address in reachable:
                    continue
                if not edges[address]:
                    raise ExtractionValidationError("observation has no path to an exact exposed quantity: " + str(address))
                active.add(address)
                pending.append((address, True))
                pending.extend((link.target_source, False) for link in edges[address] if link.target_source is not None)
    for quantity in quantities.values():
        if quantity.exactness is ExactnessKind.EXACT:
            for address in quantity.sources:
                if quantity.quantity_id not in reachable[address]:
                    raise ExtractionValidationError("quantity source has no coverage path to its declared output: " + quantity.quantity_id)
