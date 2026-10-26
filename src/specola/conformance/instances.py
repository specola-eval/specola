"""Join independently declared native instances to emitted observations.

This relation checks materialization and scalar projections. Producer closure,
native numerical fidelity and observation effects remain separate obligations
in the existing extraction catalogue; no result here grants readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from .core_runner import EvidenceIdentity
from .extraction import (
    ExtractionManifest, ExtractionValidationError, _provenance, _text,
    manifest_digest, quantity_metadata, validate_quantity_outcomes,
)
from .query_bindings import ArgumentSchema, QueryBindingError, QueryBindingRecord, materialized_query_outputs
from .surface import ProducerSurface
from ..extraction_observation import ExtractionObservation
from ..query_config import QueryConfiguration

if TYPE_CHECKING:
    from ..adapter import MonitorObservation


class InstanceValidationError(ExtractionValidationError):
    """The native declaration population differs from its materialization."""


@dataclass(frozen=True)
class NativePathStep:
    """Select a record field, sequence item or numeric-array element."""

    kind: str
    key: Union[str, int, Tuple[int, ...]]


@dataclass(frozen=True)
class ScalarProjection:
    """A declared scalar address and a closed native-value transformation."""

    key: str
    native_path: Tuple[NativePathStep, ...] = ()
    operation: str = "identity"
    argument: Optional[float] = None


@dataclass(frozen=True)
class ValueConstraint:
    """An outcome predicate on a decoded native component, not its bytes."""

    native_path: Tuple[NativePathStep, ...]
    relation: str
    argument: Optional[float] = None


@dataclass(frozen=True)
class ExpectedOutcome:
    """An admissible native outcome with independently resolved dimensions."""

    outcome_id: str
    layout: ArgumentSchema
    projections: Tuple[ScalarProjection, ...] = ()
    constraints: Tuple[ValueConstraint, ...] = ()


@dataclass(frozen=True)
class ExpectedInstance:
    """One native owner and configured binding, resolved before invoking readers."""

    quantity_id: str
    owner_kind: str
    owner_index: Tuple[int, ...]
    instance_token: str
    entity_id: Optional[int]
    outcomes: Tuple[ExpectedOutcome, ...]
    binding_id: str = ""

    @property
    def address(self):
        return self.quantity_id, self.owner_kind, self.owner_index, self.instance_token, self.binding_id


@dataclass(frozen=True)
class QuantityDomain:
    """Custody for a native instance population, including an empty one."""

    quantity_id: str
    population_rule: str
    provenance: Tuple[EvidenceIdentity, ...]
    empty_reason: str = ""


@dataclass(frozen=True)
class QueryPopulationDomain:
    """Custody for one binding's population of one result quantity."""

    binding_id: str
    quantity_id: str
    population_rule: str
    provenance: Tuple[EvidenceIdentity, ...]
    empty_reason: str = ""


@dataclass(frozen=True)
class RuntimePopulation:
    """The configured native subject, independent of emitted values.

    Providers resolve domains from producer declarations before reading any
    quantity. Legacy scalar slots have separate evidence while migrating
    existing scalar projections to the typed manifest. Neither population
    may be obtained by enumerating the observation being validated.

    ``legacy_scalar_rule`` is the bounded-migration alternative to an
    enumerated legacy population (maintainer decision 2026-09-07): a
    nonempty rule names the reviewed adapter source whose scalar emissions
    are accepted as the legacy population, with that source's identity in
    ``legacy_scalar_provenance``. The rule and enumerated slots are
    exclusive. A rule does not relax any typed projection check.
    """

    producer_surface_sha256: str
    manifest_sha256: str
    queries: QueryConfiguration
    episode_epoch: int
    configuration_epoch: int
    domains: Tuple[QuantityDomain, ...]
    instances: Tuple[ExpectedInstance, ...]
    legacy_scalar_slots: Tuple[Tuple[int, str], ...] = ()
    legacy_scalar_provenance: Tuple[EvidenceIdentity, ...] = ()
    query_domains: Tuple[QueryPopulationDomain, ...] = ()
    legacy_scalar_rule: str = ""

    @property
    def sha256(self) -> str:
        document = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _typed_tuple(value, member_type, label):
    if type(value) is not tuple or any(type(item) is not member_type for item in value):
        raise InstanceValidationError(label + " must be a tuple of " + member_type.__name__)


def _path(path) -> None:
    _typed_tuple(path, NativePathStep, "native path")
    for step in path:
        if step.kind == "field" and type(step.key) is str and step.key:
            continue
        if step.kind == "item" and type(step.key) is int and step.key >= 0:
            continue
        if (step.kind == "element" and type(step.key) is tuple
                and all(type(index) is int and index >= 0 for index in step.key)):
            continue
        raise InstanceValidationError("invalid typed native path step")


def _argument(operation: str, argument, allowed, unary) -> None:
    if operation not in allowed:
        raise InstanceValidationError("undeclared scalar operation or value relation: " + str(operation))
    if operation in unary:
        if argument is not None:
            raise InstanceValidationError("unary native operation carries an argument")
    elif type(argument) not in (int, float) or not math.isfinite(argument):
        raise InstanceValidationError("native operation needs a finite explicit argument")


def _scalar(value, path) -> Union[int, float, np.number]:
    try:
        for step in path:
            if step.kind == "field" and type(value) is dict:
                value = value[step.key]
            elif step.kind == "item" and type(value) in (list, tuple):
                value = value[step.key]
            elif step.kind == "element" and isinstance(value, np.ndarray):
                value = value[step.key]
            else:
                raise InstanceValidationError("native path kind does not match the result component")
    except (KeyError, IndexError, TypeError) as error:
        raise InstanceValidationError("native result component is absent: " + str(error)) from error
    if not (type(value) in (int, float) or isinstance(value, (np.integer, np.floating))):
        raise InstanceValidationError("native path does not select a real scalar")
    return value


# The declaration structure of the last accepted population (issue #136).
# Providers build a new RuntimePopulation per read around the same manifest,
# domain, instance and legacy-slot tuples, and the structural checks show
# those to be deeply immutable, so a verdict for the identical objects stands.
# The entry holds the objects themselves: an identity match cannot be a new
# object at a recycled address. Callers only read the returned mappings.
# ponytail: one entry; two subjects alternating in one process re-walk once per read.
_accepted_structure: Optional[tuple] = None

# Digest and replay metadata of the last manifest described (issue #136;
# maintainer decision 2026-10-12). Providers hand every read the same frozen
# manifest, so both are reused for the identical object and recomputed for a
# new one. The entries hold the manifest itself, as _accepted_structure does,
# and are kept only when its sequences are tuples, so a reused result cannot
# describe a sequence mutated in place.
# ponytail: one entry each; alternating manifests recompute once per read.
_digested: Optional[tuple] = None
_described: Optional[tuple] = None


def _frozen(manifest: ExtractionManifest) -> bool:
    return all(type(part) is tuple for part in (manifest.classifications, manifest.quantities, manifest.coverage))


def _manifest_digest(manifest: ExtractionManifest) -> str:
    global _digested
    held = _digested
    if held is not None and held[0] is manifest:
        return held[1]
    digest = manifest_digest(manifest)
    if _frozen(manifest):
        _digested = manifest, digest
    return digest


def _quantity_metadata(manifest: ExtractionManifest, quantities: dict) -> dict:
    """Replay metadata by quantity ID; ``quantities`` is the manifest's, by ID."""
    global _described
    held = _described
    if held is not None and held[0] is manifest:
        return held[1]
    metadata = {key: quantity_metadata(quantity) for key, quantity in quantities.items()}
    if _frozen(manifest):
        _described = manifest, metadata
    return metadata


# Layout verdicts of the current episode (issue #136; maintainer decision
# 2026-10-12): the set of accepted (quantity schema text, instance layout
# text, layout key) triples, held for the declaration objects that
# _accepted_structure is keyed on plus the population's episode and
# configuration epochs. A reset or reconfiguration changes an epoch, and a
# provider builds new declarations when it binds a new subject or scope, so
# either starts an empty set. RuntimePopulation records cannot carry it: the
# MuJoCo provider builds one per read.
# ponytail: one entry; two subjects alternating in one process revalidate once per read.
_accepted_layouts: Optional[tuple] = None


def _episode_layouts(manifest: ExtractionManifest, population: RuntimePopulation) -> set:
    global _accepted_layouts
    subject = manifest, population.domains, population.instances, population.legacy_scalar_slots
    epochs = population.episode_epoch, population.configuration_epoch
    held = _accepted_layouts
    if held is None or held[1] != epochs or any(old is not new for old, new in zip(held[0], subject)):
        held = _accepted_layouts = subject, epochs, set()
    return held[2]


def _validate_population(manifest: ExtractionManifest, population: RuntimePopulation):
    global _accepted_structure
    if type(manifest) is not ExtractionManifest or type(population) is not RuntimePopulation:
        raise InstanceValidationError("runtime join needs typed manifest and population records")
    if (population.manifest_sha256 != _manifest_digest(manifest)
            or population.producer_surface_sha256 != manifest.producer_surface_digest):
        raise InstanceValidationError("runtime population producer or manifest identity differs")
    if type(population.queries) is not QueryConfiguration:
        raise InstanceValidationError("runtime population needs a query configuration")
    for epoch in (population.episode_epoch, population.configuration_epoch):
        if type(epoch) is not int or epoch < 0:
            raise InstanceValidationError("runtime population epochs must be nonnegative integers")
    subject = manifest, population.domains, population.instances, population.legacy_scalar_slots
    accepted = _accepted_structure
    if accepted is not None and all(held is current for held, current in zip(accepted[0], subject)):
        quantities, instances, legacy = accepted[1]
    else:
        quantities, instances, legacy = _population_structure(manifest, population)
        # The manifest's quantity sequence is the one input not type-checked here.
        if type(manifest.quantities) is tuple:
            _accepted_structure = subject, (quantities, instances, legacy)
    if type(population.legacy_scalar_rule) is not str:
        raise InstanceValidationError("legacy scalar rule must be text")
    if population.legacy_scalar_rule and not population.legacy_scalar_rule.strip():
        raise InstanceValidationError("legacy scalar rule must be nonempty text")
    if population.legacy_scalar_rule and legacy:
        raise InstanceValidationError("legacy scalar rule and enumerated legacy slots are exclusive")
    if legacy or population.legacy_scalar_rule or population.legacy_scalar_provenance:
        _provenance(population.legacy_scalar_provenance, "legacy scalar population")
    return quantities, instances, legacy


def _population_structure(manifest: ExtractionManifest, population: RuntimePopulation):
    """Check the declarations a provider fixes when it binds its subject."""
    quantities = {quantity.quantity_id: quantity for quantity in manifest.quantities}
    if len(quantities) != len(manifest.quantities):
        raise InstanceValidationError("runtime manifest has duplicate quantity IDs")
    for quantity in quantities.values():
        validate_quantity_outcomes(quantity)
    _typed_tuple(population.domains, QuantityDomain, "quantity domains")
    domains = {}
    for domain in population.domains:
        _text(domain.quantity_id, "domain quantity ID")
        _text(domain.population_rule, "native population rule")
        _provenance(domain.provenance, "native population")
        if type(domain.empty_reason) is not str:
            raise InstanceValidationError("empty population reason must be text")
        if domain.quantity_id in domains:
            raise InstanceValidationError("duplicate native quantity domain")
        domains[domain.quantity_id] = domain
    if domains.keys() != quantities.keys():
        raise InstanceValidationError("native quantity domain population differs from the manifest")
    _typed_tuple(population.instances, ExpectedInstance, "native instances")
    instances, populated, possible_slots = {}, set(), set()
    for instance in population.instances:
        if instance.quantity_id not in quantities:
            raise InstanceValidationError("native instance has an undeclared quantity")
        quantity = quantities[instance.quantity_id]
        if instance.owner_kind != quantity.owner_kind:
            raise InstanceValidationError("native owner kind differs from quantity semantics")
        if (type(instance.owner_index) is not tuple or any(type(index) is not int for index in instance.owner_index)
                or type(instance.instance_token) is not str
                or type(instance.binding_id) is not str
                or (instance.entity_id is not None and type(instance.entity_id) is not int)):
            raise InstanceValidationError("invalid native instance address")
        if instance.address in instances:
            raise InstanceValidationError("duplicate expected native instance address")
        instances[instance.address] = instance
        populated.add(instance.quantity_id)
        _typed_tuple(instance.outcomes, ExpectedOutcome, "expected native outcomes")
        if not instance.outcomes:
            raise InstanceValidationError("native instance has no admissible outcomes")
        declared = {outcome.outcome_id for outcome in quantity.outcomes}
        seen, instance_slots = set(), set()
        for outcome in instance.outcomes:
            if outcome.outcome_id not in declared or outcome.outcome_id in seen:
                raise InstanceValidationError("undeclared or duplicate expected outcome")
            seen.add(outcome.outcome_id)
            if type(outcome.layout) is not ArgumentSchema:
                raise InstanceValidationError("expected native outcome needs a typed layout")
            _typed_tuple(outcome.projections, ScalarProjection, "scalar projections")
            keys = set()
            for projection in outcome.projections:
                _text(projection.key, "scalar projection key")
                _path(projection.native_path)
                _argument(projection.operation, projection.argument,
                          {"identity", "min_with_argument", "ge_indicator"}, {"identity"})
                if instance.entity_id is None or projection.key in keys:
                    raise InstanceValidationError("scalar projection needs a unique entity/key address")
                keys.add(projection.key)
                instance_slots.add((instance.entity_id, projection.key))
            _typed_tuple(outcome.constraints, ValueConstraint, "native value constraints")
            for constraint in outcome.constraints:
                _path(constraint.native_path)
                _argument(constraint.relation, constraint.argument,
                          {"finite", "lt", "le", "ge", "eq"}, {"finite"})
        if possible_slots.intersection(instance_slots):
            raise InstanceValidationError("scalar address is declared by multiple native instances")
        possible_slots.update(instance_slots)
    for quantity_id, domain in domains.items():
        if (quantity_id in populated and domain.empty_reason) or (quantity_id not in populated and not domain.empty_reason.strip()):
            raise InstanceValidationError("empty native population and its reason disagree")
    _typed_tuple(population.legacy_scalar_slots, tuple, "legacy scalar slots")
    legacy = set()
    for slot in population.legacy_scalar_slots:
        if len(slot) != 2 or type(slot[0]) is not int or type(slot[1]) is not str or not slot[1]:
            raise InstanceValidationError("invalid legacy scalar slot")
        if slot in legacy or slot in possible_slots:
            raise InstanceValidationError("duplicate or overlapping legacy scalar slot")
        legacy.add(slot)
    return quantities, instances, legacy


def _layout_key(document):
    """An encoded ``NativeValue`` document without its base64 ``data`` payloads.

    A ``data`` key occurs only in tagged binary payloads (floats, complex
    numbers, byte strings, array and NumPy-scalar contents); record field
    names are list items, not keys. The codec keeps each payload as
    canonical base64 text. Every other part of the document
    stays in the key: kind tags, record field names and order, item counts,
    dtype and shape, and integer, boolean and text values.
    """
    kind = type(document)
    if kind is dict:
        return kind, tuple((key, None if key == "data" else _layout_key(item)) for key, item in document.items())
    if kind is list:
        return kind, tuple(map(_layout_key, document))
    return kind, document


def validate_runtime_population(manifest: ExtractionManifest, population: RuntimePopulation) -> None:
    """Validate the declaration relation before executing a native reader."""
    try:
        _validate_population(manifest, population)
    except ExtractionValidationError as error:
        raise InstanceValidationError(str(error)) from error


def validate_query_population(
    surface: ProducerSurface, manifest: ExtractionManifest, population: RuntimePopulation,
    bindings: Tuple[QueryBindingRecord, ...],
) -> None:
    """Join validated callable outputs to their configured native instances.

    The caller first validates the complete manifest, binding declarations and
    arguments. This join checks attribution and emptiness; native owner oracles
    must separately establish each binding's declared population and layouts.
    """
    materialized = materialized_query_outputs(surface, manifest)
    query_quantities = set().union(*materialized.values())
    known = {binding.binding_id: binding for binding in bindings}
    by_binding = {key: materialized[value.callable_id] for key, value in known.items()}
    tokens = {item["id"]: item["binding"] for item in population.queries.instances()}
    resolvers = {binding.binding_id for binding in bindings if binding.population_resolver_id is not None}
    producing = resolvers | set(tokens.values())
    for quantity in query_quantities:
        participants = {key for key in known if quantity in by_binding[key]}
        if (len({known[key].callable_id for key in participants}) > 1
                and len(participants & producing) > 1):
            raise InstanceValidationError("combined quantity has more than one owner-producing binding")
    present = set()
    for row in population.instances:
        if row.quantity_id not in query_quantities:
            if row.binding_id or row.instance_token:
                raise InstanceValidationError("field instance carries a query binding or token")
            continue
        if row.binding_id not in known or row.quantity_id not in by_binding[row.binding_id]:
            raise InstanceValidationError("query instance has an undeclared binding or result quantity")
        if row.instance_token:
            if tokens.get(row.instance_token) != row.binding_id:
                raise InstanceValidationError("query instance token differs from its configured binding")
        elif row.binding_id not in resolvers:
            raise InstanceValidationError("query instance has no configured population binding")
        present.add((row.binding_id, row.quantity_id, row.instance_token))
    for token, binding in tokens.items():
        for quantity in by_binding[binding]:
            if (binding, quantity, token) not in present:
                raise InstanceValidationError("explicit query instance omitted a result quantity")
    _typed_tuple(population.query_domains, QueryPopulationDomain, "query population domains")
    domains = set()
    for domain in population.query_domains:
        _text(domain.binding_id, "query population binding")
        _text(domain.quantity_id, "query population quantity")
        _text(domain.population_rule, "query population rule")
        _provenance(domain.provenance, "query population")
        if type(domain.empty_reason) is not str:
            raise InstanceValidationError("empty query population reason must be text")
        key = domain.binding_id, domain.quantity_id
        if key in domains:
            raise InstanceValidationError("duplicate query population domain")
        domains.add(key)
        populated = (domain.binding_id, domain.quantity_id, "") in present
        if populated == bool(domain.empty_reason.strip()):
            raise InstanceValidationError("empty query population and its reason disagree")
    expected_domains = {(binding, quantity) for binding in resolvers for quantity in by_binding[binding]}
    if domains != expected_domains:
        raise InstanceValidationError("query population domain set differs from configured bindings")


def validate_materialization(manifest: ExtractionManifest, population: RuntimePopulation, observation: MonitorObservation) -> None:
    """Check both instance and scalar address relations before scope projection."""
    try:
        quantities, expected, legacy = _validate_population(manifest, population)
        extraction = observation.extraction
        if type(extraction) is not ExtractionObservation:
            raise InstanceValidationError("native reader omitted its structured observation")
        if (extraction.producer_surface_sha256 != population.producer_surface_sha256
                or extraction.manifest_sha256 != population.manifest_sha256
                or extraction.queries != population.queries
                or extraction.episode_epoch != population.episode_epoch
                or extraction.configuration_epoch != population.configuration_epoch):
            raise InstanceValidationError("native observation subject differs from its declared population")
        if extraction.timestep != observation.timestep or extraction.time != observation.time:
            raise InstanceValidationError("native observation instant differs from scalar observation")
        metadata = {item.quantity_id: item for item in extraction.metadata}
        if metadata != _quantity_metadata(manifest, quantities):
            raise InstanceValidationError("native semantic metadata differs from quantity declarations")
        samples = {sample.address: sample for sample in extraction.samples}
        if samples.keys() != expected.keys():
            missing, extra = expected.keys() - samples.keys(), samples.keys() - expected.keys()
            raise InstanceValidationError("native instance population differs; missing=" + repr(sorted(missing)) + "; undeclared=" + repr(sorted(extra)))
        slots = set(legacy)
        # Layout verdicts (issue #136; maintainer decisions 2026-10-07 and
        # 2026-10-12). The schemas run once per distinct pair of schema texts
        # and encoded document without its base64 payloads, per episode of
        # one declared population (_episode_layouts); a later sample with an
        # equal key, in this observation or a later one, reuses the verdict.
        # Field set and order, kind tags, dtype, shape and non-binary values
        # are still compared per sample, so a sample whose layout differs
        # from every accepted one is validated. Given up: schema constraints
        # on a later sample's base64 payload text. The current layouts declare
        # those only as strings, which the NativeValue codec already enforces.
        accepted_layouts = _episode_layouts(manifest, population)
        for address, sample in samples.items():
            instance = expected[address]
            if sample.entity_id != instance.entity_id:
                raise InstanceValidationError("native and scalar entity ownership differ")
            outcome = next((item for item in instance.outcomes if item.outcome_id == sample.availability), None)
            if outcome is None:
                raise InstanceValidationError("native reader returned an inadmissible outcome")
            declaration = next(item for item in quantities[sample.quantity_id].outcomes if item.outcome_id == outcome.outcome_id)
            document = sample.value.document()
            layout = declaration.native_schema.canonical_json, outcome.layout.canonical_json, _layout_key(document)
            if layout not in accepted_layouts:
                try:
                    declaration.native_schema.validate(document, sample.quantity_id)
                    # An instance layout identical to the quantity schema adds
                    # no constraint; validating it again only costs time.
                    if outcome.layout.canonical_json != declaration.native_schema.canonical_json:
                        outcome.layout.validate(document, sample.quantity_id)
                except QueryBindingError as error:
                    raise InstanceValidationError("native result layout differs: " + str(error)) from error
                accepted_layouts.add(layout)
            if sample.scalar_keys != tuple(item.key for item in outcome.projections):
                raise InstanceValidationError("native scalar links differ from declared projections")
            value = sample.value.restore() if outcome.projections or outcome.constraints else None
            for constraint in outcome.constraints:
                scalar = _scalar(value, constraint.native_path)
                argument = constraint.argument
                valid = math.isfinite(scalar)
                if valid and constraint.relation != "finite":
                    if constraint.relation == "lt":
                        valid = scalar < argument
                    elif constraint.relation == "le":
                        valid = scalar <= argument
                    elif constraint.relation == "ge":
                        valid = scalar >= argument
                    else:
                        valid = scalar == argument
                if not valid:
                    raise InstanceValidationError("native value violates its declared outcome policy")
            for projection in outcome.projections:
                scalar = _scalar(value, projection.native_path)
                if not math.isfinite(scalar):
                    raise InstanceValidationError("non-finite native value cannot enter a scalar projection")
                if projection.operation == "min_with_argument":
                    scalar = min(scalar, projection.argument)
                elif projection.operation == "ge_indicator":
                    scalar = float(scalar >= projection.argument)
                entity = observation.perceived_entities.get(instance.entity_id)
                if entity is None or projection.key not in entity.properties:
                    raise InstanceValidationError("declared scalar projection is missing")
                emitted = entity.properties[projection.key]
                if (type(emitted) not in (int, float) and not isinstance(emitted, (np.integer, np.floating))
                        or not math.isfinite(emitted) or emitted != scalar
                        or (scalar == 0 and math.copysign(1, emitted) != math.copysign(1, scalar))):
                    raise InstanceValidationError("scalar value differs from its declared native projection")
                slots.add((instance.entity_id, projection.key))
        actual_slots = {(entity_id, key) for entity_id, entity in observation.perceived_entities.items() for key in entity.properties}
        if population.legacy_scalar_rule:
            # Bounded migration: an entity that owns no typed projection is a
            # legacy owner whose scalars belong to the reviewed adapter source
            # named by the rule. An entity that owns a typed projection keeps
            # the exact census: it emits its declared projections and nothing
            # else, so a stray key on a typed owner is still rejected.
            typed_owners = {entity_id for entity_id, _ in slots}
            typed_actual = {slot for slot in actual_slots if slot[0] in typed_owners}
            if typed_actual != slots:
                raise InstanceValidationError("typed owner scalar population differs from its declared projections")
        elif actual_slots != slots:
            raise InstanceValidationError("scalar population differs from independently declared outputs")
    except ExtractionValidationError as error:
        raise InstanceValidationError(str(error)) from error
