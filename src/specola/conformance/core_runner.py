"""Raw, finite public-call evidence for adapter conformance assessment.

The runner records facts only. Structural predicates, outcomes, and custody
claims are pure derivations. ``artifact_tamper_resistant`` is deliberately
false: a caller can still replace valid raw facts. External signing and an
append-only evidence store are out of scope for this simulator-free module.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import marshal
import re
import traceback
import types

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, cast

import numpy as np

from ..adapter import MonitorObservation, SimulatorAdapter
from ..discovery_model import property_metadata_for
from ..entities import EntityState, TrackState
from ..extraction_observation import (
    ExtractionObservationError, extraction_document, extraction_from_document,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
STRUCTURAL_PREDICATE_SCHEMA = "specola-core-raw-facts-v2"
ARTIFACT_TAMPER_RESISTANT = False
STRUCTURAL_PREDICATE_CATALOGUE = (
    "LIFECYCLE.PRE_RESET.V1",
    "CALL.PUBLIC_CALLABLE.V1",
    "OBSERVATION.TYPE.V1",
    "OBSERVATION.TIME_INDEX.V1",
    "OBSERVATION.CROSS_CALL_TIME_INDEX.V1",
    "ENTITY.STRUCTURE.V1",
    "ENTITY.IDENTITY_CLASS_STABILITY.V1",
    "VECTOR.STRUCTURE.V1",
    "PROPERTY.SCALAR.V1",
    "TAG.RECONCILIATION.V1",
    "DISCOVERY.METADATA.V1",
    "SOURCE.MRO_STABLE.V1",
)


class ConformanceValidationError(ValueError):
    """Raised when supplied or captured evidence is internally inconsistent."""


class CallKind(str, Enum):
    RESET = "reset"
    GET_OBSERVATION = "get_observation"


class CallExpectation(str, Enum):
    RETURN_OBSERVATION = "return_observation"
    EXPECT_REJECT_BEFORE_RESET = "expect_reject_before_reset"


class PredicateStatus(str, Enum):
    SATISFIED_ON_TRACE = "satisfied_on_trace"
    VIOLATED = "violated"
    UNEXERCISED = "unexercised"
    UNEXECUTED = "unexecuted"


class ExecutionOutcome(str, Enum):
    COMPLETE = "complete"
    SETUP_UNAVAILABLE = "setup_unavailable"
    VIOLATION = "violation"


class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    VIOLATED = "violated"


@dataclass(frozen=True)
class EvidenceIdentity:
    """A declared identity and SHA-256, not a claimed resolved authority."""

    identity: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.identity, "evidence identity")
        _digest(self.sha256, "evidence SHA-256")


@dataclass(frozen=True)
class CertificationSubject:
    """Declarations supplied for a certification row; only direct source binds."""

    entry_point_declaration: str
    expected_adapter_module: str
    expected_adapter_qualname: str
    expected_adapter_source: EvidenceIdentity
    producer_declaration: str
    configuration_domain_declaration: str
    supported_domain_source_declaration: EvidenceIdentity

    def __post_init__(self) -> None:
        for value, label in (
            (self.entry_point_declaration, "entry point declaration"),
            (self.expected_adapter_module, "adapter module declaration"),
            (self.expected_adapter_qualname, "adapter qualname declaration"),
            (self.producer_declaration, "producer declaration"),
            (self.configuration_domain_declaration, "configuration declaration"),
        ):
            _text(value, label)
        _identity(self.expected_adapter_source, "expected adapter source")
        _identity(self.supported_domain_source_declaration, "supported-domain declaration")


@dataclass(frozen=True)
class TraceCase:
    """Finite direct-factory plan; trace identity remains a declaration."""

    case_id: str
    trace_source_declaration: EvidenceIdentity
    observations_per_episode: int
    driver_factory: Callable[[], SimulatorAdapter] = field(repr=False, compare=False)
    episode_count: int = 1
    probe_observe_before_reset: bool = False

    def __post_init__(self) -> None:
        _text(self.case_id, "case ID")
        _identity(self.trace_source_declaration, "trace source declaration")
        _positive(self.observations_per_episode, "observations per episode")
        _positive(self.episode_count, "episode count")
        if not isinstance(self.probe_observe_before_reset, bool):
            raise ConformanceValidationError("observe-before-reset probe must be bool")
        if not callable(self.driver_factory):
            raise ConformanceValidationError("driver factory must be callable")


@dataclass(frozen=True)
class PlannedCall:
    """One exact typed operation in a case denominator."""

    kind: CallKind
    episode_index: int
    observation_index: Optional[int]
    expectation: CallExpectation

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CallKind) or not isinstance(self.expectation, CallExpectation):
            raise ConformanceValidationError("planned call kind and expectation must be enums")
        _nonnegative(self.episode_index, "episode index")
        if self.expectation is CallExpectation.EXPECT_REJECT_BEFORE_RESET:
            if self.kind is not CallKind.GET_OBSERVATION or self.episode_index != 0:
                raise ConformanceValidationError("pre-reset rejection probe must be first-episode observation")
        elif self.kind is CallKind.RESET:
            if self.observation_index is not None:
                raise ConformanceValidationError("reset has no observation index")
        else:
            _nonnegative(self.observation_index, "observation index")


@dataclass(frozen=True)
class SourceMeasurement:
    """A source file measurement, or a recorded source-unavailability fact."""

    module: str
    qualname: str
    path: Optional[str]
    sha256: Optional[str]
    unavailable_reason: Optional[str]

    def __post_init__(self) -> None:
        _text(self.module, "source module")
        _text(self.qualname, "source qualname")
        if self.path is None:
            _text(self.unavailable_reason, "source unavailable reason")
            if self.sha256 is not None:
                raise ConformanceValidationError("unavailable source cannot carry digest")
        else:
            _text(self.path, "source path")
            _digest(self.sha256, "source SHA-256")
            if self.unavailable_reason is not None:
                raise ConformanceValidationError("available source cannot carry unavailable reason")


@dataclass(frozen=True)
class FrozenJsonArtifact:
    identity: str
    schema: str
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        for value, label in ((self.identity, "artifact identity"), (self.schema, "artifact schema")):
            _text(value, label)
        _text(self.canonical_json, "artifact canonical JSON")
        _digest(self.sha256, "artifact SHA-256")
        if _hash_text(self.canonical_json) != self.sha256:
            raise ConformanceValidationError("artifact digest mismatch")
        try:
            parsed = json.loads(self.canonical_json)
            canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ConformanceValidationError("artifact canonical JSON invalid: {}".format(error))
        if canonical != self.canonical_json or type(parsed) is not dict:
            raise ConformanceValidationError("artifact canonical JSON must round-trip exactly")
        if set(parsed) != {"identity", "schema", "payload"} or type(parsed["identity"]) is not str or type(parsed["schema"]) is not str or type(parsed["payload"]) is not dict:
            raise ConformanceValidationError("artifact envelope has exact identity/schema/payload keys")
        if parsed["identity"] != self.identity or parsed["schema"] != self.schema:
            raise ConformanceValidationError("artifact envelope disagrees with identity or schema")


def freeze_json_artifact(identity, schema, payload):
    _text(identity, "artifact identity")
    _text(schema, "artifact schema")
    if type(payload) is not dict:
        raise ConformanceValidationError("artifact payload must be exact dict")
    try:
        canonical = json.dumps({"identity": identity, "schema": schema, "payload": payload}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConformanceValidationError("artifact payload JSON invalid: {}".format(error))
    return FrozenJsonArtifact(identity, schema, canonical, _hash_text(canonical))

@dataclass(frozen=True)
class ProducerMeasurement:
    producer_name: str
    configuration_id: str
    type_source: SourceMeasurement
    producer: object = field(repr=False, compare=False)
    version: Optional[str] = None
    content_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for value, label in ((self.producer_name, "producer name"), (self.configuration_id, "producer configuration ID")):
            _text(value, label)
        if not isinstance(self.type_source, SourceMeasurement):
            raise ConformanceValidationError("producer type source must be SourceMeasurement")
        if self.version is None and self.content_sha256 is None:
            raise ConformanceValidationError("producer needs version or content digest")
        if self.version is not None:
            _text(self.version, "producer version")
        if self.content_sha256 is not None:
            _digest(self.content_sha256, "producer content digest")


def measure_producer(producer, producer_name, configuration_id, version=None, content_sha256=None):
    return ProducerMeasurement(producer_name, configuration_id, _source_for_class(type(producer)), producer, version, content_sha256)

def _callable_source(value, label):
    target = _callable_target(value, label)
    try:
        path = inspect.getsourcefile(target)
        if path is None:
            raise OSError("sourcefile is None")
        absolute = str(Path(path).resolve())
        return SourceMeasurement(target.__module__, target.__qualname__, absolute, hashlib.sha256(Path(absolute).read_bytes()).hexdigest(), None)
    except (OSError, TypeError) as error:
        raise ConformanceValidationError("{} source unavailable: {}: {}".format(label, type(error).__name__, error))


def _callable_target(value, label):
    """Return a source-measurable function/method's loaded implementation."""

    target = getattr(value, "__func__", value)
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        raise ConformanceValidationError("{} must be source-measurable function or method".format(label))
    return target


def _python_marshal_code_sha256(target):
    # Format 2 predates reference tracking, so custody ignores live refcounts.
    return hashlib.sha256(marshal.dumps(target.__code__, 2)).hexdigest()

@dataclass(frozen=True)
class DriverBundle:
    adapter: SimulatorAdapter
    producer: object = field(repr=False, compare=False)
    producer_measurement: ProducerMeasurement
    variant_id: str
    advance: Callable = field(repr=False, compare=False)
    native_snapshot: Callable = field(repr=False, compare=False)
    materialization_trace: Callable = field(repr=False, compare=False)
    interaction_counts: Callable = field(repr=False, compare=False)
    adapter_source: SourceMeasurement = field(init=False)
    hook_sources: Tuple[SourceMeasurement, ...] = field(init=False)
    hook_targets: Tuple[object, ...] = field(init=False, repr=False, compare=False)
    hook_codes: Tuple[object, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, SimulatorAdapter) or not isinstance(self.producer_measurement, ProducerMeasurement):
            raise ConformanceValidationError("bundle needs SimulatorAdapter and ProducerMeasurement")
        _text(self.variant_id, "bundle variant ID")
        if self.producer_measurement.producer is not self.producer:
            raise ConformanceValidationError("bundle producer disagrees with producer measurement")
        hooks = (self.advance, self.native_snapshot, self.materialization_trace, self.interaction_counts)
        if not all(callable(item) for item in hooks):
            raise ConformanceValidationError("bundle harness hooks must be callable")
        targets = tuple(_callable_target(item, "bundle hook") for item in hooks)
        object.__setattr__(self, "adapter_source", _source_for_class(type(self.adapter)))
        object.__setattr__(self, "hook_sources", tuple(_callable_source(item, "bundle hook") for item in hooks))
        object.__setattr__(self, "hook_targets", targets)
        object.__setattr__(self, "hook_codes", tuple(item.__code__ for item in targets))

@dataclass(frozen=True)
class FidelityCase:
    configuration_id: str
    case_id: str
    variant_denominator: Tuple[str, ...]
    profile_denominator: Tuple[str, ...]
    candidate_denominator: Tuple[str, ...]
    bundle_factory: Callable[[str], DriverBundle] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for value, label in ((self.configuration_id, "configuration ID"), (self.case_id, "fidelity case ID")):
            _text(value, label)
        for value, label in ((self.variant_denominator, "variant denominator"), (self.profile_denominator, "profile denominator"), (self.candidate_denominator, "candidate denominator")):
            _tuple(value, label, nonempty=True)
            if any(not isinstance(item, str) or not item.strip() for item in value) or len(set(value)) != len(value):
                raise ConformanceValidationError("{} must contain unique non-empty text".format(label))
        if not callable(self.bundle_factory):
            raise ConformanceValidationError("bundle factory must be callable")

def capture_fidelity_authority(cases):
    """Capture raw custody; marshal code digests are interpreter-specific facts."""

    _tuple(cases, "fidelity cases", nonempty=True)
    if any(not isinstance(case, FidelityCase) for case in cases):
        raise ConformanceValidationError("fidelity cases must be FidelityCase values")
    snapshots = tuple((case, case.configuration_id, case.case_id, case.variant_denominator, case.profile_denominator, case.candidate_denominator, case.bundle_factory, _callable_source(case.bundle_factory, "bundle factory"), _callable_target(case.bundle_factory, "bundle factory"), _callable_target(case.bundle_factory, "bundle factory").__code__) for case in cases)
    configurations = tuple(dict.fromkeys(item[1] for item in snapshots))
    case_ids = tuple(item[2] for item in snapshots)
    if len(set(case_ids)) != len(case_ids):
        raise ConformanceValidationError("case denominator must be unique")
    seen_adapters, seen_producers, bundles, records = [], [], [], []
    for case, configuration_id, case_id, variants, profiles, candidates, factory, factory_source, factory_target, factory_code in snapshots:
        for variant_id in variants:
            bundle = factory(variant_id)
            if not isinstance(bundle, DriverBundle):
                raise ConformanceValidationError("bundle factory must return DriverBundle")
            if any(bundle.adapter is item for item in seen_adapters) or any(bundle.producer is item for item in seen_producers):
                raise ConformanceValidationError("bundle factory reused adapter or producer identity")
            seen_adapters.append(bundle.adapter); seen_producers.append(bundle.producer)
            if bundle.variant_id != variant_id:
                raise ConformanceValidationError("bundle factory returned wrong variant")
            if bundle.producer_measurement.producer is not bundle.producer or bundle.producer_measurement.configuration_id != configuration_id or bundle.adapter_source != _source_for_class(type(bundle.adapter)) or bundle.producer_measurement.type_source != _source_for_class(type(bundle.producer)):
                raise ConformanceValidationError("bundle configuration or measured source disagrees with constructed objects")
            hooks = (bundle.advance, bundle.native_snapshot, bundle.materialization_trace, bundle.interaction_counts)
            if tuple(_callable_source(item, "bundle hook") for item in hooks) != bundle.hook_sources:
                raise ConformanceValidationError("bundle hook source drift")
            record = {"configuration_id": configuration_id, "case_id": case_id, "variant_id": variant_id, "variant_denominator": list(variants), "profile_denominator": list(profiles), "candidate_denominator": list(candidates), "bundle_factory_source": _source_data(factory_source), "python_marshal_code_sha256": _python_marshal_code_sha256(factory_target), "adapter_source": _source_data(bundle.adapter_source), "producer": {"name": bundle.producer_measurement.producer_name, "configuration_id": bundle.producer_measurement.configuration_id, "type_source": _source_data(bundle.producer_measurement.type_source), "version": bundle.producer_measurement.version, "content_sha256": bundle.producer_measurement.content_sha256}, "hooks": [{"name": name, "source": _source_data(source), "python_marshal_code_sha256": _python_marshal_code_sha256(target)} for name, source, target in zip(("advance", "native_snapshot", "materialization_trace", "interaction_counts"), bundle.hook_sources, bundle.hook_targets)]}
            records.append(record); bundles.append((bundle, bundle.adapter, bundle.producer, hooks, variant_id, record))
    for case, configuration_id, case_id, variants, profiles, candidates, factory, factory_source, factory_target, factory_code in snapshots:
        if (case.configuration_id, case.case_id, case.variant_denominator, case.profile_denominator, case.candidate_denominator, case.bundle_factory) != (configuration_id, case_id, variants, profiles, candidates, factory):
            raise ConformanceValidationError("fidelity case declaration drift")
        if _callable_source(factory, "bundle factory") != factory_source or _callable_target(factory, "bundle factory") is not factory_target or factory_target.__code__ is not factory_code:
            raise ConformanceValidationError("bundle factory source or code drift")
    for bundle, adapter, producer, initial_hooks, variant_id, record in bundles:
        hooks = (bundle.advance, bundle.native_snapshot, bundle.materialization_trace, bundle.interaction_counts)
        measurement = bundle.producer_measurement
        targets = tuple(_callable_target(item, "bundle hook") for item in hooks)
        if bundle.adapter is not adapter or bundle.producer is not producer or bundle.variant_id != variant_id or any(current is not initial for current, initial in zip(hooks, initial_hooks)) or any(target is not initial for target, initial in zip(targets, bundle.hook_targets)) or any(target.__code__ is not code for target, code in zip(targets, bundle.hook_codes)) or measurement.producer is not bundle.producer or measurement.producer_name != record["producer"]["name"] or measurement.configuration_id != record["producer"]["configuration_id"] or measurement.version != record["producer"]["version"] or measurement.content_sha256 != record["producer"]["content_sha256"] or bundle.adapter_source != _source_for_class(type(bundle.adapter)) or measurement.type_source != _source_for_class(type(bundle.producer)) or tuple(_callable_source(item, "bundle hook") for item in hooks) != bundle.hook_sources:
            raise ConformanceValidationError("retained bundle source drift")
    artifact = freeze_json_artifact("fidelity-authority-header", "specola-fidelity-authority-v1", {"configuration_denominator": list(configurations), "case_denominator": list(case_ids), "cases": records})
    validate_fidelity_authority(artifact)
    return artifact


def _authority_source(value, label):
    if type(value) is not dict or set(value) != {"module", "qualname", "path", "sha256", "unavailable_reason"}:
        raise ConformanceValidationError("{} must be exact source mapping".format(label))
    return SourceMeasurement(value["module"], value["qualname"], value["path"], value["sha256"], value["unavailable_reason"])


def _authority_texts(value, label):
    if type(value) is not list or not value or any(type(item) is not str or not item.strip() for item in value) or len(set(value)) != len(value):
        raise ConformanceValidationError("{} must be non-empty unique text list".format(label))
    return tuple(value)


def validate_fidelity_authority(artifact):
    """Validate the versioned authority payload beyond the generic JSON envelope."""

    if not isinstance(artifact, FrozenJsonArtifact) or artifact.identity != "fidelity-authority-header" or artifact.schema != "specola-fidelity-authority-v1":
        raise ConformanceValidationError("authority artifact identity or schema mismatch")
    payload = json.loads(artifact.canonical_json)["payload"]
    if type(payload) is not dict or set(payload) != {"configuration_denominator", "case_denominator", "cases"}:
        raise ConformanceValidationError("authority payload has exact denominator/cases keys")
    configurations = _authority_texts(payload["configuration_denominator"], "configuration denominator")
    case_ids = _authority_texts(payload["case_denominator"], "case denominator")
    if type(payload["cases"]) is not list or not payload["cases"]:
        raise ConformanceValidationError("authority cases must be non-empty list")
    definitions, observed_variants = {}, {}
    record_keys = {"configuration_id", "case_id", "variant_id", "variant_denominator", "profile_denominator", "candidate_denominator", "bundle_factory_source", "python_marshal_code_sha256", "adapter_source", "producer", "hooks"}
    for record in payload["cases"]:
        if type(record) is not dict or set(record) != record_keys:
            raise ConformanceValidationError("authority case record has exact keys")
        for key in ("configuration_id", "case_id", "variant_id"):
            if type(record[key]) is not str or not record[key].strip():
                raise ConformanceValidationError("authority {} must be text".format(key))
        variants = _authority_texts(record["variant_denominator"], "variant denominator")
        profiles = _authority_texts(record["profile_denominator"], "profile denominator")
        candidates = _authority_texts(record["candidate_denominator"], "candidate denominator")
        _authority_source(record["bundle_factory_source"], "bundle factory source")
        _digest(record["python_marshal_code_sha256"], "bundle factory marshal code SHA-256")
        _authority_source(record["adapter_source"], "adapter source")
        producer = record["producer"]
        if type(producer) is not dict or set(producer) != {"name", "configuration_id", "type_source", "version", "content_sha256"}:
            raise ConformanceValidationError("authority producer has exact keys")
        ProducerMeasurement(producer["name"], producer["configuration_id"], _authority_source(producer["type_source"], "producer type source"), object(), producer["version"], producer["content_sha256"])
        if producer["configuration_id"] != record["configuration_id"]:
            raise ConformanceValidationError("authority producer configuration mismatch")
        hooks = record["hooks"]
        if type(hooks) is not list or [item.get("name") if type(item) is dict else None for item in hooks] != ["advance", "native_snapshot", "materialization_trace", "interaction_counts"]:
            raise ConformanceValidationError("authority hooks must have exact ordered names")
        for hook in hooks:
            if set(hook) != {"name", "source", "python_marshal_code_sha256"}:
                raise ConformanceValidationError("authority hook has exact keys")
            _authority_source(hook["source"], "hook source")
            _digest(hook["python_marshal_code_sha256"], "hook marshal code SHA-256")
        definition = (record["configuration_id"], variants, profiles, candidates, record["bundle_factory_source"], record["python_marshal_code_sha256"])
        if record["case_id"] in definitions and definitions[record["case_id"]] != definition:
            raise ConformanceValidationError("authority case record denominator mismatch")
        definitions.setdefault(record["case_id"], definition)
        observed_variants.setdefault(record["case_id"], []).append(record["variant_id"])
    if tuple(definitions) != case_ids or tuple(dict.fromkeys(item[0] for item in definitions.values())) != configurations:
        raise ConformanceValidationError("authority denominator does not reconcile records")
    if any(tuple(observed_variants[case_id]) != definitions[case_id][1] for case_id in case_ids):
        raise ConformanceValidationError("authority variants do not reconcile records")


@dataclass(frozen=True)
class AdapterProvenance:
    """Ordered complete MRO measurements at one point in a run."""

    direct_module: str
    direct_qualname: str
    ordered_mro_sources: Tuple[SourceMeasurement, ...]

    def __post_init__(self) -> None:
        _text(self.direct_module, "direct module")
        _text(self.direct_qualname, "direct qualname")
        _tuple(self.ordered_mro_sources, "MRO sources", nonempty=True)
        for source in self.ordered_mro_sources:
            if not isinstance(source, SourceMeasurement):
                raise ConformanceValidationError("MRO sources must be SourceMeasurement values")


@dataclass(frozen=True)
class BoundCallableFact:
    """The callable obtained immediately before its corresponding invocation."""

    call: PlannedCall
    method_name: str
    owner_module: str
    owner_qualname: str
    callable_id: int
    callable: object = field(repr=False, compare=False)
    driver: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.call, PlannedCall):
            raise ConformanceValidationError("bound-call fact needs planned call")
        _text(self.method_name, "method name")
        _text(self.owner_module, "callable owner module")
        _text(self.owner_qualname, "callable owner qualname")
        _nonnegative(self.callable_id, "callable ID")
        if self.callable_id != id(self.callable):
            raise ConformanceValidationError("callable ID must match captured callable")


@dataclass(frozen=True)
class FrozenObservation:
    """Immediately canonicalized observation, with hash and round-trip check."""

    call: PlannedCall
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.call, PlannedCall):
            raise ConformanceValidationError("frozen observation needs planned call")
        _text(self.canonical_json, "canonical observation JSON")
        _digest(self.sha256, "canonical observation SHA-256")
        if _hash_text(self.canonical_json) != self.sha256:
            raise ConformanceValidationError("canonical observation digest mismatch")
        try:
            parsed = json.loads(self.canonical_json)
            roundtrip = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ConformanceValidationError("canonical observation JSON invalid: {}".format(error))
        if roundtrip != self.canonical_json or not isinstance(parsed, dict):
            raise ConformanceValidationError("canonical observation JSON must round-trip exactly")
        required = {"type", "timestep", "time", "perceived_entities", "component_tags"}
        if set(parsed) not in (required, required | {"extraction"}) or parsed.get("type") != "MonitorObservation":
            raise ConformanceValidationError("canonical observation payload is incomplete")
        if "extraction" in parsed:
            try:
                extraction_from_document(parsed["extraction"])
            except ExtractionObservationError as error:
                raise ConformanceValidationError("canonical structured observation is invalid: " + str(error)) from error


@dataclass(frozen=True)
class ExceptionFact:
    """A raw exception or an explicit unexpected-return fact for one call."""

    call: Optional[PlannedCall]
    phase: str
    exception_type: str
    message: str
    traceback_text: str
    expected: bool
    entity_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.call is not None and not isinstance(self.call, PlannedCall):
            raise ConformanceValidationError("exception call must be planned or None")
        for value, label in ((self.phase, "exception phase"), (self.exception_type, "exception type"), (self.message, "exception message"), (self.traceback_text, "exception traceback")):
            _text(value, label)
        if not isinstance(self.expected, bool):
            raise ConformanceValidationError("exception expected marker must be bool")
        if self.entity_id is not None:
            _nonnegative(self.entity_id, "metadata entity ID")


@dataclass(frozen=True)
class MetadataProbeFact:
    """A raw metadata callback attempt for one observed entity."""

    call: PlannedCall
    entity_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.call, PlannedCall):
            raise ConformanceValidationError("metadata probe needs planned call")
        if self.call.expectation is not CallExpectation.RETURN_OBSERVATION:
            raise ConformanceValidationError("metadata probe needs observation-producing call")
        _nonnegative(self.entity_id, "metadata entity ID")


@dataclass(frozen=True)
class CaseExecution:
    """Runner-owned raw facts only; no status, outcome, or predicate fields."""

    case_id: str
    trace_source_declaration: EvidenceIdentity
    plan: Tuple[PlannedCall, ...]
    attempted: Tuple[PlannedCall, ...]
    successful: Tuple[PlannedCall, ...]
    bound_callables: Tuple[BoundCallableFact, ...]
    observations: Tuple[FrozenObservation, ...]
    exceptions: Tuple[ExceptionFact, ...]
    provenance_before: Optional[AdapterProvenance]
    provenance_after: Optional[AdapterProvenance]
    metadata_probes: Tuple[MetadataProbeFact, ...] = ()

    def __post_init__(self) -> None:
        _text(self.case_id, "execution case ID")
        _identity(self.trace_source_declaration, "execution trace source")
        _tuple(self.plan, "call plan", nonempty=True)
        _tuple(self.attempted, "attempted calls")
        _tuple(self.successful, "successful calls")
        _tuple(self.bound_callables, "bound callable facts")
        _tuple(self.observations, "frozen observations")
        _tuple(self.exceptions, "exception facts")
        _tuple(self.metadata_probes, "metadata probe facts")
        if len(set(self.plan)) != len(self.plan):
            raise ConformanceValidationError("plan call IDs must be unique")
        if self.attempted != self.plan[: len(self.attempted)]:
            raise ConformanceValidationError("attempted calls must be plan prefix")
        if self.successful != self.attempted[: len(self.successful)]:
            raise ConformanceValidationError("successful calls must be attempted prefix")
        if tuple(item.call for item in self.bound_callables) != self.attempted:
            raise ConformanceValidationError("bound callables must exactly match attempted calls")
        producing = tuple(
            call for call in self.successful
            if call.expectation is CallExpectation.RETURN_OBSERVATION
        )
        if tuple(item.call for item in self.observations) != producing:
            raise ConformanceValidationError("observations must match successful observation-producing calls")
        for collection, type_ in (
            (self.plan, PlannedCall), (self.attempted, PlannedCall),
            (self.successful, PlannedCall), (self.bound_callables, BoundCallableFact),
            (self.observations, FrozenObservation), (self.exceptions, ExceptionFact),
            (self.metadata_probes, MetadataProbeFact),
        ):
            for item in collection:
                if not isinstance(item, type_):
                    raise ConformanceValidationError("raw fact has wrong type")
        exception_calls = [item.call for item in self.exceptions if item.call is not None]
        if any(call not in self.attempted for call in exception_calls):
            raise ConformanceValidationError("exception calls must be attempted")
        ordinary_exception_calls = [
            item.call for item in self.exceptions
            if item.call is not None and item.phase != "metadata"
        ]
        if len(set(ordinary_exception_calls)) != len(ordinary_exception_calls):
            raise ConformanceValidationError("at most one non-metadata exception fact per call")
        successful_observation_calls = {item.call for item in self.observations}
        if any(
            item.call in successful_observation_calls and not item.expected
            and item.entity_id is None
            for item in self.exceptions if item.call is not None
        ):
            raise ConformanceValidationError("call cannot have both observation and unexpected exception")
        exception_keys = [(item.call, item.phase, item.entity_id) for item in self.exceptions]
        if len(set(exception_keys)) != len(exception_keys):
            raise ConformanceValidationError("exception facts must have unique call/phase pairs")
        metadata_keys = [(item.call, item.entity_id) for item in self.metadata_probes]
        if len(set(metadata_keys)) != len(metadata_keys):
            raise ConformanceValidationError("metadata probes must be unique per call/entity")
        expected_metadata = _metadata_denominator(self.observations)
        if tuple(metadata_keys) != expected_metadata:
            raise ConformanceValidationError(
                "metadata probes must exactly match canonical observed-entity denominator"
            )
        metadata_key_set = set(metadata_keys)
        for item in self.exceptions:
            if item.phase == "metadata":
                if (
                    item.call is None or item.entity_id is None
                    or item.expected or (item.call, item.entity_id) not in metadata_key_set
                ):
                    raise ConformanceValidationError(
                        "metadata exception must identify an observed probe"
                    )
            elif item.entity_id is not None:
                raise ConformanceValidationError(
                    "entity ID is reserved for metadata exceptions"
                )
        for provenance in (self.provenance_before, self.provenance_after):
            if provenance is not None and not isinstance(provenance, AdapterProvenance):
                raise ConformanceValidationError("provenance has wrong type")


@dataclass(frozen=True)
class CoreConformanceArtifact:
    """Raw facts plus declarations. Use :func:`derive_assessment` to assess."""

    subject_declarations: CertificationSubject
    predicate_schema: str
    case_denominator: Tuple[str, ...]
    source_denominator: Tuple[EvidenceIdentity, ...]
    executions: Tuple[CaseExecution, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subject_declarations, CertificationSubject):
            raise ConformanceValidationError("artifact subject must be declarations")
        if self.predicate_schema != STRUCTURAL_PREDICATE_SCHEMA:
            raise ConformanceValidationError("unknown predicate schema")
        _tuple(self.case_denominator, "case denominator", nonempty=True)
        _tuple(self.source_denominator, "source denominator", nonempty=True)
        _tuple(self.executions, "executions", nonempty=True)
        for case_id in self.case_denominator:
            _text(case_id, "case denominator ID")
        for identity in self.source_denominator:
            _identity(identity, "source denominator identity")
        for execution in self.executions:
            if not isinstance(execution, CaseExecution):
                raise ConformanceValidationError("execution has wrong type")
        if tuple(item.case_id for item in self.executions) != self.case_denominator:
            raise ConformanceValidationError("executions must match case denominator")
        if tuple(item.trace_source_declaration for item in self.executions) != self.source_denominator:
            raise ConformanceValidationError("execution sources must match source denominator")
        if len(set(self.case_denominator)) != len(self.case_denominator):
            raise ConformanceValidationError("case denominator IDs must be unique")

    def to_dict(self) -> Dict[str, object]:
        """Serialize raw facts plus a freshly derived, non-stored assessment."""

        return {
            "subject_declarations": _subject_data(self.subject_declarations),
            "predicate_schema": self.predicate_schema,
            "case_denominator": list(self.case_denominator),
            "source_denominator": [_identity_data(item) for item in self.source_denominator],
            "executions": [_execution_data(item) for item in self.executions],
            "derived_assessment": _assessment_data(derive_assessment(self)),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PredicateResult:
    predicate_id: str
    call: Optional[PlannedCall]
    witness: str
    status: PredicateStatus
    detail: str


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    status: ClaimStatus
    detail: str


@dataclass(frozen=True)
class DerivedAssessment:
    """Purely derived assessment; never stored in :class:`CaseExecution`."""

    predicate_results: Tuple[PredicateResult, ...]
    claims: Tuple[ClaimResult, ...]
    outcome: ExecutionOutcome

    def __post_init__(self) -> None:
        _tuple(self.predicate_results, "predicate results")
        _tuple(self.claims, "derived claims")
        for result in self.predicate_results:
            if not isinstance(result, PredicateResult):
                raise ConformanceValidationError("predicate result has wrong type")
        keys = [(item.predicate_id, item.call, item.witness) for item in self.predicate_results]
        if len(set(keys)) != len(keys):
            raise ConformanceValidationError("predicate witnesses must be unique")


def run_core_conformance(subject: CertificationSubject, cases: Tuple[TraceCase, ...]) -> CoreConformanceArtifact:
    """Run exact plans and return only runner-captured raw facts."""

    if not isinstance(subject, CertificationSubject):
        raise ConformanceValidationError("subject must be CertificationSubject")
    _tuple(cases, "trace cases", nonempty=True)
    if len({case.case_id for case in cases}) != len(cases):
        raise ConformanceValidationError("trace case IDs must be unique")
    seen = set()
    retained = []
    executions = []
    for case in cases:
        execution, driver = _run_case(case, seen)
        executions.append(execution)
        if driver is not None:
            retained.append(driver)
    del retained
    return CoreConformanceArtifact(
        subject_declarations=subject,
        predicate_schema=STRUCTURAL_PREDICATE_SCHEMA,
        case_denominator=tuple(case.case_id for case in cases),
        source_denominator=tuple(case.trace_source_declaration for case in cases),
        executions=tuple(executions),
    )


def derive_assessment(artifact: CoreConformanceArtifact) -> DerivedAssessment:
    """Recompute finite predicates, claims, and outcome from raw facts."""

    if not isinstance(artifact, CoreConformanceArtifact):
        raise ConformanceValidationError("assessment needs CoreConformanceArtifact")
    results = []
    for execution in artifact.executions:
        results.extend(_derive_case(artifact.subject_declarations, execution))
    claims = _derive_claims(artifact)
    statuses = {item.status for item in results}
    setup_only = all(
        not item.attempted
        and any(exc.call is None and exc.phase == "factory" for exc in item.exceptions)
        for item in artifact.executions
    )
    if setup_only:
        outcome = ExecutionOutcome.SETUP_UNAVAILABLE
    elif PredicateStatus.VIOLATED in statuses or PredicateStatus.UNEXECUTED in statuses:
        outcome = ExecutionOutcome.VIOLATION
    else:
        outcome = ExecutionOutcome.COMPLETE
    return DerivedAssessment(tuple(results), tuple(claims), outcome)


def measure_adapter_provenance(adapter: SimulatorAdapter) -> AdapterProvenance:
    """Measure the direct class and complete ordered MRO source surface."""

    if not isinstance(adapter, SimulatorAdapter):
        raise ConformanceValidationError("object is not a SimulatorAdapter")
    cls = type(adapter)
    return AdapterProvenance(
        direct_module=cls.__module__,
        direct_qualname=cls.__qualname__,
        ordered_mro_sources=tuple(_source_for_class(item) for item in cls.__mro__),
    )


def _run_case(case: TraceCase, seen: set, before_observation=None, after_observation=None) -> Tuple[CaseExecution, Optional[SimulatorAdapter]]:
    plan = _plan(case)
    attempted, successful, bound, observations, exceptions = [], [], [], [], []
    metadata_probes = []
    driver = None
    before = after = None
    try:
        try:
            driver = case.driver_factory()
        except Exception as error:
            exceptions.append(_exception(None, "factory", error, False))
            return _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after), None
        if not isinstance(driver, SimulatorAdapter):
            exceptions.append(_manual(None, "factory_contract", "TypeError", "driver factory did not return a SimulatorAdapter", False))
            return _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after), None
        if id(driver) in seen:
            exceptions.append(_manual(None, "factory", "RuntimeError", "driver factory reused an adapter instance", False))
            return _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after), driver
        seen.add(id(driver))
        try:
            before = measure_adapter_provenance(driver)
        except Exception as error:
            exceptions.append(_exception(None, "pre_provenance", error, False))
            return _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after), driver
        for call in plan:
            method_name = "reset" if call.kind is CallKind.RESET else "get_observation"
            attempted.append(call)
            if call.kind is CallKind.GET_OBSERVATION and before_observation is not None:
                try:
                    before_observation(driver, call)
                except Exception as error:
                    exceptions.append(_exception(call, "before_observation", error, False))
                    break
            try:
                bound_callable = getattr(driver, method_name)
            except Exception as error:
                bound.append(_bound(call, method_name, None, driver))
                exceptions.append(_exception(call, "lookup", error, False))
                break
            bound.append(_bound(call, method_name, bound_callable, driver))
            if not callable(bound_callable):
                exceptions.append(_manual(call, "call", "TypeError", "public {} is not callable".format(method_name), False))
                break
            try:
                observation = bound_callable()
            except Exception as error:
                expected = call.expectation is CallExpectation.EXPECT_REJECT_BEFORE_RESET
                exceptions.append(_exception(call, "call", error, expected))
                if expected:
                    successful.append(call)
                    continue
                break
            if call.expectation is CallExpectation.EXPECT_REJECT_BEFORE_RESET:
                exceptions.append(_manual(call, "call", "UnexpectedReturn", "pre-reset observation returned instead of rejecting", False))
                break
            try:
                frozen = _freeze(call, observation)
            except Exception as error:
                exceptions.append(_exception(call, "capture", error, False))
                break
            successful.append(call)
            observations.append(frozen)
            if call.kind is CallKind.GET_OBSERVATION and after_observation is not None:
                try:
                    after_observation(driver, call)
                except Exception as error:
                    exceptions.append(_exception(call, "after_observation", error, False))
                    break
            _probe_metadata(driver, call, observation, metadata_probes, exceptions)
    finally:
        if driver is not None:
            try:
                after = measure_adapter_provenance(driver)
            except Exception as error:
                exceptions.append(_exception(None, "post_provenance", error, False))
        facts = _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after, metadata_probes)
    return facts, driver


def _facts(case, plan, attempted, successful, bound, observations, exceptions, before, after, metadata_probes=()):
    return CaseExecution(
        case_id=case.case_id,
        trace_source_declaration=case.trace_source_declaration,
        plan=tuple(plan), attempted=tuple(attempted), successful=tuple(successful),
        bound_callables=tuple(bound), observations=tuple(observations), exceptions=tuple(exceptions),
        provenance_before=before, provenance_after=after, metadata_probes=tuple(metadata_probes),
    )


def _metadata_denominator(observations):
    denominator = []
    for observation in observations:
        parsed = json.loads(observation.canonical_json)
        entities = _canonical_mapping(parsed.get("perceived_entities"))
        if entities is None:
            continue
        entity_ids = []
        for key, _ in entities:
            entity_id = _canonical_integer(key)
            if entity_id is None:
                raise ConformanceValidationError("observed entity denominator needs canonical nonnegative IDs")
            entity_ids.append(entity_id)
        denominator.extend((observation.call, entity_id) for entity_id in sorted(entity_ids))
    return tuple(denominator)


def _probe_metadata(driver, call, observation, probes, exceptions):
    """Probe the existing discovery metadata path for every observed entity."""

    for entity_id in sorted(observation.perceived_entities):
        entity_id = int(entity_id)
        probes.append(MetadataProbeFact(call, entity_id))
        try:
            property_metadata_for(driver, (entity_id,))
        except Exception as error:
            exceptions.append(_exception(call, "metadata", error, False, entity_id))


def _plan(case: TraceCase) -> Tuple[PlannedCall, ...]:
    calls = []
    if case.probe_observe_before_reset:
        calls.append(PlannedCall(CallKind.GET_OBSERVATION, 0, 0, CallExpectation.EXPECT_REJECT_BEFORE_RESET))
    for episode in range(case.episode_count):
        calls.append(PlannedCall(CallKind.RESET, episode, None, CallExpectation.RETURN_OBSERVATION))
        for index in range(case.observations_per_episode):
            calls.append(PlannedCall(CallKind.GET_OBSERVATION, episode, index, CallExpectation.RETURN_OBSERVATION))
    return tuple(calls)


def _derive_case(subject: CertificationSubject, facts: CaseExecution) -> Tuple[PredicateResult, ...]:
    results = []
    exceptions_by_call = {
        item.call: item for item in facts.exceptions
        if item.call is not None and item.entity_id is None
    }
    observations = {item.call: item for item in facts.observations}
    bindings = {item.call: item for item in facts.bound_callables}
    metadata_failures = {(item.call, item.entity_id) for item in facts.exceptions if item.phase == "metadata" and item.entity_id is not None}
    previous_by_episode = {}
    classes_by_episode = {}
    for call in facts.plan:
        exception = exceptions_by_call.get(call)
        binding = bindings.get(call)
        if binding is None:
            results.extend(_unexecuted(call))
            continue
        expected_function = (
            SimulatorAdapter.reset
            if call.kind is CallKind.RESET
            else SimulatorAdapter.get_observation
        )
        template_bound = _is_exact_template_binding(binding, expected_function)
        results.append(_result("CALL.PUBLIC_CALLABLE.V1", call, "call", template_bound, "bound callable identity and receiver"))
        if call.expectation is CallExpectation.EXPECT_REJECT_BEFORE_RESET:
            status = PredicateStatus.SATISFIED_ON_TRACE if exception and exception.expected else PredicateStatus.VIOLATED
            results.append(PredicateResult("LIFECYCLE.PRE_RESET.V1", call, "pre-reset", status, "expected rejection"))
            continue
        if exception is not None or call not in observations:
            results.extend(_unexecuted(call, skip_callable=True))
            continue
        parsed = json.loads(observations[call].canonical_json)
        results.extend(_observation_results(call, parsed))
        results.extend(_cross_call_results(call, parsed, previous_by_episode, classes_by_episode))
        results.extend(_metadata_results(call, facts.metadata_probes, metadata_failures))
    results.append(_source_result(subject, facts))
    present = {(item.predicate_id, item.call) for item in results}
    for call in facts.plan:
        for identifier in STRUCTURAL_PREDICATE_CATALOGUE:
            if identifier == "SOURCE.MRO_STABLE.V1" or (identifier, call) in present:
                continue
            results.append(
                PredicateResult(
                    identifier, call, "none", PredicateStatus.UNEXERCISED,
                    "no witness in this finite call surface",
                )
            )
    # ``PlannedCall`` is deliberately case-local.  Include the immutable case
    # denominator in every witness so a multi-case artifact has one unambiguous
    # (predicate, call, witness) row for every finite assessment fact.
    return tuple(
        PredicateResult(
            item.predicate_id, item.call,
            "case:{}|{}".format(facts.case_id, item.witness),
            item.status, item.detail,
        )
        for item in results
    )


def _unexecuted(call: PlannedCall, skip_callable: bool = False) -> Tuple[PredicateResult, ...]:
    ids = tuple(
        identifier for identifier in STRUCTURAL_PREDICATE_CATALOGUE
        if identifier != "SOURCE.MRO_STABLE.V1"
        and not (skip_callable and identifier == "CALL.PUBLIC_CALLABLE.V1")
    )
    return tuple(PredicateResult(identifier, call, "call", PredicateStatus.UNEXECUTED, "call did not yield assessable fact") for identifier in ids)


def _observation_results(call: PlannedCall, parsed: Dict[str, object]) -> Tuple[PredicateResult, ...]:
    results = [
        _result("OBSERVATION.TYPE.V1", call, "observation", parsed.get("type") == "MonitorObservation", "canonical observation type"),
        _result("OBSERVATION.TIME_INDEX.V1", call, "observation", _canonical_int(parsed.get("timestep")) and _canonical_finite(parsed.get("time")), "canonical time/index"),
    ]
    entities = _canonical_mapping(parsed.get("perceived_entities"))
    if entities is None:
        results.extend(_zero_witness(call))
        return tuple(results)
    if not entities:
        results.extend(_zero_witness(call))
    for key, entity in entities:
        witness = "entity:{}".format(key)
        valid_entity = _valid_entity(entity)
        results.append(_result("ENTITY.STRUCTURE.V1", call, witness, valid_entity, "entity shape"))
        if not valid_entity:
            continue
        results.append(_result("VECTOR.STRUCTURE.V1", call, witness + ":position", _canonical_vector(entity.get("position")), "position vector"))
        results.append(_result("VECTOR.STRUCTURE.V1", call, witness + ":velocity", _canonical_vector(entity.get("velocity")), "velocity vector"))
        properties = _canonical_mapping(entity.get("properties"))
        if not properties:
            results.append(PredicateResult("PROPERTY.SCALAR.V1", call, witness + ":none", PredicateStatus.UNEXERCISED, "no property witness"))
        else:
            for name, value in properties:
                results.append(_result("PROPERTY.SCALAR.V1", call, witness + ":property:" + str(name), _canonical_finite(value), "scalar property"))
    raw_tags = parsed.get("component_tags")
    tags = _canonical_mapping(raw_tags)
    if tags is None:
        results.append(_result("TAG.RECONCILIATION.V1", call, "surface", False, "component tags must be mapping"))
    elif not tags:
        results.append(PredicateResult("TAG.RECONCILIATION.V1", call, "none", PredicateStatus.UNEXERCISED, "no component-tag witness"))
    else:
        for entity_id, mapping in tags:
            nested = _canonical_mapping(mapping)
            if not nested:
                results.append(_result("TAG.RECONCILIATION.V1", call, "tag:{}".format(entity_id), False, "tag mapping"))
            else:
                for name, labels in nested:
                    results.append(_result(
                        "TAG.RECONCILIATION.V1", call, "tag:{}:{}".format(entity_id, name),
                        _valid_tag_pair(entities, entity_id, name, labels),
                        "entity/property-reconciled non-empty tuple labels",
                    ))
    return tuple(results)


def _cross_call_results(call, parsed, previous_by_episode, classes_by_episode):
    """Derive temporal and episode-local entity identity facts from one trace."""

    timestep = _canonical_integer(parsed.get("timestep"))
    time_value = _canonical_time(parsed.get("time"))
    previous = previous_by_episode.get(call.episode_index)
    temporal = timestep is not None and time_value is not None
    if temporal and previous is not None:
        temporal = timestep > previous[0] and time_value >= previous[1]
    if temporal:
        previous_by_episode[call.episode_index] = (timestep, time_value)
    results = [_result(
        "OBSERVATION.CROSS_CALL_TIME_INDEX.V1", call,
        "episode:{}".format(call.episode_index), temporal,
        "strict timestep and nondecreasing time within episode",
    )]
    entities = _canonical_mapping(parsed.get("perceived_entities"))
    if not entities:
        results.append(PredicateResult(
            "ENTITY.IDENTITY_CLASS_STABILITY.V1", call, "none",
            PredicateStatus.UNEXERCISED, "no entity identity witness",
        ))
        return tuple(results)
    prior_classes = classes_by_episode.setdefault(call.episode_index, {})
    for key, entity in entities:
        entity_id = _canonical_integer(entity.get("entity_id")) if isinstance(entity, dict) else None
        entity_class = _canonical_string(entity.get("entity_class")) if isinstance(entity, dict) else None
        valid = entity_id is not None and entity_class is not None
        typed_id = entity_id if valid else None
        prior = prior_classes.get(typed_id) if typed_id is not None else None
        stable = valid and (prior is None or prior == entity_class)
        results.append(_result(
            "ENTITY.IDENTITY_CLASS_STABILITY.V1", call, "entity:{}".format(key), stable,
            "episode-local entity ID retains class",
        ))
        if stable:
            prior_classes[typed_id] = entity_class
    return tuple(results)


def _zero_witness(call):
    return (
        PredicateResult("ENTITY.STRUCTURE.V1", call, "none", PredicateStatus.UNEXERCISED, "no entity witness"),
        PredicateResult("VECTOR.STRUCTURE.V1", call, "none", PredicateStatus.UNEXERCISED, "no vector witness"),
        PredicateResult("PROPERTY.SCALAR.V1", call, "none", PredicateStatus.UNEXERCISED, "no property witness"),
    )


def _metadata_results(call, probes, failures):
    results = []
    for probe in probes:
        if probe.call != call:
            continue
        witness = "entity:{}".format(probe.entity_id)
        failed = (probe.call, probe.entity_id) in failures
        results.append(PredicateResult(
            "DISCOVERY.METADATA.V1", call, witness,
            PredicateStatus.VIOLATED if failed else PredicateStatus.SATISFIED_ON_TRACE,
            "property metadata callback failed" if failed
            else "property metadata callback returned an accepted mapping",
        ))
    return tuple(results)


def _source_result(subject, facts):
    before, after = facts.provenance_before, facts.provenance_after
    valid = (
        before is not None and after is not None
        and before.direct_module == subject.expected_adapter_module
        and before.direct_qualname == subject.expected_adapter_qualname
        and before.ordered_mro_sources == after.ordered_mro_sources
        and before.ordered_mro_sources[0].path == subject.expected_adapter_source.identity
        and before.ordered_mro_sources[0].sha256 == subject.expected_adapter_source.sha256
    )
    return _result("SOURCE.MRO_STABLE.V1", None, "case:ordered-mro", valid, "pre/post ordered MRO source")


def _derive_claims(artifact):
    claims = [
        ClaimResult("ENTRY_POINT", ClaimStatus.UNVERIFIED, "direct factory not resolved from entry-point declaration"),
        ClaimResult("PRODUCER", ClaimStatus.UNVERIFIED, "producer declaration has no runtime authority"),
        ClaimResult("CONFIGURATION_DOMAIN", ClaimStatus.UNVERIFIED, "configuration declaration has no enumerated authority"),
        ClaimResult("SUPPORTED_DOMAIN", ClaimStatus.UNVERIFIED, "supported-domain declaration has no typed resolver"),
    ]
    for execution in artifact.executions:
        claims.append(ClaimResult("TRACE." + execution.case_id, ClaimStatus.UNVERIFIED, "trace declaration has no typed resolver"))
    return tuple(claims)


def _result(identifier, call, witness, passed, detail):
    return PredicateResult(identifier, call, witness, PredicateStatus.SATISFIED_ON_TRACE if passed else PredicateStatus.VIOLATED, detail)


def _bound(call, name, value, driver):
    function = _safe_attr(value, "__func__", value)
    module = _safe_attr(function, "__module__", type(value).__module__)
    qualname = _safe_attr(function, "__qualname__", type(value).__qualname__)
    return BoundCallableFact(
        call, name,
        module if isinstance(module, str) and module else type(value).__module__,
        qualname if isinstance(qualname, str) and qualname else type(value).__qualname__,
        id(value), value, driver,
    )


def _safe_attr(value, name, default):
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _is_exact_template_binding(binding, expected_function):
    """Assess captured identity, never display-only owner strings."""

    try:
        return (
            isinstance(binding.callable, types.MethodType)
            and inspect.ismethod(binding.callable)
            and getattr(binding.callable, "__func__", None) is expected_function
            and getattr(binding.callable, "__self__", None) is binding.driver
        )
    except Exception:
        return False


def _freeze(call, observation):
    text = json.dumps(_canonical(observation), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return FrozenObservation(call, text, _hash_text(text))


def _canonical(value):
    # NumPy scalar classes are transport details, not observation semantics.
    # Convert before recording or assessing so equivalent Python values have one
    # canonical representation.
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, MonitorObservation):
        result = {"type": "MonitorObservation", "timestep": _canonical(value.timestep), "time": _canonical(value.time), "perceived_entities": _canonical(value.perceived_entities), "component_tags": _canonical(value.component_tags)}
        # Keep the historical wire form when no typed sink exists. The nested
        # sink has its own versioned codec; repr would discard its raw types.
        if value.extraction is not None:
            result["extraction"] = extraction_document(value.extraction)
        return result
    if isinstance(value, EntityState):
        return {"type": "EntityState", "entity_id": _canonical(value.entity_id), "entity_class": _canonical(value.entity_class), "track_age": _canonical(value.track_age), "track_state": _canonical(value.track_state), "position": _canonical(value.position), "velocity": _canonical(value.velocity), "properties": _canonical(value.properties)}
    if isinstance(value, TrackState):
        return {"type": "TrackState", "value": value.value}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "values": _canonical(value.tolist())}
    if isinstance(value, Mapping):
        values = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return {"type": "mapping", "items": sorted(values, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [_canonical(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"type": type(value).__name__, "value": repr(value)}
    return {"type": type(value).__module__ + "." + type(value).__qualname__, "repr": repr(value)}


def _canonical_mapping(value):
    if not isinstance(value, dict) or value.get("type") != "mapping":
        return None
    items = value.get("items")
    if not isinstance(items, list):
        return None
    pairs = []
    for item in items:
        if not isinstance(item, list) or len(item) != 2:
            return None
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _canonical_int(value):
    return _canonical_integer(value) is not None


def _canonical_finite(value):
    return _canonical_time(value) is not None


def _canonical_vector(value):
    canonical_none = value == {"type": "NoneType", "value": "None"}
    return canonical_none or (isinstance(value, dict) and value.get("type") == "ndarray" and isinstance(value.get("values"), dict))


def _canonical_labels(value):
    return (
        isinstance(value, dict) and value.get("type") == "tuple"
        and isinstance(value.get("items"), list) and bool(value["items"])
        and all(_canonical_string(item) is not None for item in value["items"])
    )


def _canonical_scalar(value):
    """Recover a primitive from canonical scalar data without evaluating code."""

    if not isinstance(value, dict) or "value" not in value:
        return None
    if value.get("type") not in ("NoneType", "bool", "int", "float", "str"):
        return None
    try:
        return ast.literal_eval(value["value"])
    except (SyntaxError, ValueError, TypeError):
        return None


def _canonical_integer(value):
    scalar = _canonical_scalar(value)
    if _valid_timestep(scalar):
        return cast(int, scalar)
    return None


def _canonical_time(value):
    scalar = _canonical_scalar(value)
    if _valid_time(scalar):
        return float(cast(float, scalar))
    return None


def _canonical_string(value):
    scalar = _canonical_scalar(value)
    if _valid_entity_class(scalar):
        return scalar
    return None


def _valid_timestep(value):
    return type(value) is int and value >= 0


def _valid_time(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def _valid_entity_id(value):
    return _valid_timestep(value)


def _valid_entity_class(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_entity(entity):
    if not isinstance(entity, dict) or entity.get("type") != "EntityState":
        return False
    return (
        _canonical_integer(entity.get("entity_id")) is not None
        and _canonical_string(entity.get("entity_class")) is not None
        and _canonical_integer(entity.get("track_age")) is not None
        and _canonical_track_state(entity.get("track_state"))
    )


def _canonical_track_state(value):
    return (
        isinstance(value, dict)
        and value.get("type") == "TrackState"
        and value.get("value") in tuple(item.value for item in TrackState)
    )


def _valid_tag_pair(entities, entity_id, name, labels):
    """Check exact tag ownership, property reconciliation, and label shape."""

    normalized_entity_id = _canonical_integer(entity_id)
    property_name = _canonical_string(name)
    if (
        normalized_entity_id is None
        or property_name is None
        or not _canonical_labels(labels)
    ):
        return False
    for key, entity in entities:
        if _canonical_integer(key) != normalized_entity_id or not isinstance(entity, dict):
            continue
        properties = _canonical_mapping(entity.get("properties"))
        if properties is None:
            return False
        return any(_canonical_string(item_name) == property_name for item_name, _ in properties)
    return False


def _source_for_class(cls):
    try:
        path = inspect.getsourcefile(cls)
        if path is None:
            raise OSError("sourcefile is None")
        absolute = str(Path(path).resolve())
        return SourceMeasurement(cls.__module__, cls.__qualname__, absolute, hashlib.sha256(Path(absolute).read_bytes()).hexdigest(), None)
    except Exception as error:
        return SourceMeasurement(cls.__module__, cls.__qualname__, None, None, "{}: {}".format(type(error).__name__, error))


def _exception(call, phase, error, expected, entity_id=None):
    exception_type = type(error).__name__
    return ExceptionFact(
        call, phase, exception_type, str(error) or exception_type,
        traceback.format_exc(), expected, entity_id,
    )


def _manual(call, phase, type_, message, expected):
    return ExceptionFact(call, phase, type_, message, "runner fact: {}: {}".format(type_, message), expected)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ConformanceValidationError("{} must be non-empty text".format(label))


def _digest(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ConformanceValidationError("{} must be lowercase SHA-256".format(label))


def _identity(value, label):
    if not isinstance(value, EvidenceIdentity):
        raise ConformanceValidationError("{} must be EvidenceIdentity".format(label))


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConformanceValidationError("{} must be positive integer".format(label))


def _nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConformanceValidationError("{} must be nonnegative integer".format(label))


def _tuple(value, label, nonempty=False):
    if not isinstance(value, tuple) or (nonempty and not value):
        raise ConformanceValidationError("{} must be immutable tuple{}".format(label, " and non-empty" if nonempty else ""))


def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity_data(value):
    return {"identity": value.identity, "sha256": value.sha256}


def _source_data(value):
    return {"module": value.module, "qualname": value.qualname, "path": value.path, "sha256": value.sha256, "unavailable_reason": value.unavailable_reason}


def _subject_data(value):
    return {"entry_point_declaration": value.entry_point_declaration, "expected_adapter_module": value.expected_adapter_module, "expected_adapter_qualname": value.expected_adapter_qualname, "expected_adapter_source": _identity_data(value.expected_adapter_source), "producer_declaration": value.producer_declaration, "configuration_domain_declaration": value.configuration_domain_declaration, "supported_domain_source_declaration": _identity_data(value.supported_domain_source_declaration)}


def _call_data(value):
    return {"kind": value.kind.value, "episode_index": value.episode_index, "observation_index": value.observation_index, "expectation": value.expectation.value}


def _provenance_data(value):
    if value is None:
        return None
    return {"direct_module": value.direct_module, "direct_qualname": value.direct_qualname, "ordered_mro_sources": [_source_data(item) for item in value.ordered_mro_sources]}


def _execution_data(value):
    return {"case_id": value.case_id, "trace_source_declaration": _identity_data(value.trace_source_declaration), "plan": [_call_data(item) for item in value.plan], "attempted": [_call_data(item) for item in value.attempted], "successful": [_call_data(item) for item in value.successful], "bound_callables": [{"call": _call_data(item.call), "method_name": item.method_name, "owner_module": item.owner_module, "owner_qualname": item.owner_qualname, "callable_id": item.callable_id} for item in value.bound_callables], "observations": [{"call": _call_data(item.call), "canonical_json": item.canonical_json, "sha256": item.sha256} for item in value.observations], "metadata_probes": [{"call": _call_data(item.call), "entity_id": item.entity_id} for item in value.metadata_probes], "exceptions": [{"call": None if item.call is None else _call_data(item.call), "phase": item.phase, "entity_id": item.entity_id, "exception_type": item.exception_type, "message": item.message, "traceback_text": item.traceback_text, "expected": item.expected} for item in value.exceptions], "provenance_before": _provenance_data(value.provenance_before), "provenance_after": _provenance_data(value.provenance_after)}


def _assessment_data(value):
    return {"predicate_results": [{"predicate_id": item.predicate_id, "call": None if item.call is None else _call_data(item.call), "witness": item.witness, "status": item.status.value, "detail": item.detail} for item in value.predicate_results], "claims": [{"claim_id": item.claim_id, "status": item.status.value, "detail": item.detail} for item in value.claims], "outcome": value.outcome.value}
