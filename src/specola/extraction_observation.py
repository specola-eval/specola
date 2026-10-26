"""Versioned structured observations beside the legacy scalar interface.

The envelope records native quantity identity and detached values. Structural
validation does not establish producer completeness or native fidelity; those
come from the independent manifest, provider guards and executed evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .native_values import NativeValue
from .query_config import QueryConfiguration, _json_tree


class ExtractionObservationError(ValueError):
    """A structured observation cannot be reconciled with its declared shape."""


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ExtractionObservationError(label + " must be nonempty text")


def _sha(value, label):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ExtractionObservationError(label + " must be a SHA-256 digest")


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ExtractionObservationError(label + " must be a nonnegative integer")


@dataclass(frozen=True)
class StructuredMetadata:
    """Semantic identity retained with a structured quantity during replay."""

    quantity_id: str
    native_meaning: str
    native_schema: str
    unit: str
    frame: str
    transformation: str
    source_members: Tuple[str, ...]
    provenance_sha256: str

    def __post_init__(self):
        for field in ("quantity_id", "native_meaning", "native_schema", "unit", "frame", "transformation"):
            _text(getattr(self, field), field)
        _sha(self.provenance_sha256, "quantity provenance")
        if type(self.source_members) is not tuple or not self.source_members:
            raise ExtractionObservationError("quantity needs source members")
        for member in self.source_members:
            _text(member, "source member")
        if len(set(self.source_members)) != len(self.source_members):
            raise ExtractionObservationError("quantity has duplicate source members")


@dataclass(frozen=True)
class StructuredSample:
    """One quantity instance at a declared native owner/index address."""

    quantity_id: str
    owner_kind: str
    owner_index: Tuple[int, ...]
    value: NativeValue
    availability: str
    entity_id: Optional[int] = None
    scalar_keys: Tuple[str, ...] = ()
    instance_token: str = ""
    binding_id: str = ""

    def __post_init__(self):
        for field in ("quantity_id", "owner_kind", "availability"):
            _text(getattr(self, field), field)
        if type(self.owner_index) is not tuple or any(type(index) is not int for index in self.owner_index):
            raise ExtractionObservationError("native owner indices must be an integer tuple")
        if type(self.value) is not NativeValue:
            raise ExtractionObservationError("structured value must be a detached NativeValue")
        if self.entity_id is not None:
            _integer(self.entity_id, "entity ID")
        if type(self.scalar_keys) is not tuple or any(type(key) is not str or not key for key in self.scalar_keys):
            raise ExtractionObservationError("scalar projections must be named keys")
        if len(set(self.scalar_keys)) != len(self.scalar_keys):
            raise ExtractionObservationError("duplicate scalar projection key")
        if self.scalar_keys and self.entity_id is None:
            raise ExtractionObservationError("scalar projections need an entity owner")
        if type(self.instance_token) is not str:
            raise ExtractionObservationError("query instance token must be text")
        if type(self.binding_id) is not str:
            raise ExtractionObservationError("query binding identity must be text")

    @property
    def address(self):
        return self.quantity_id, self.owner_kind, self.owner_index, self.instance_token, self.binding_id


@dataclass(frozen=True)
class ExtractionObservation:
    """One typed extraction instant with immutable subject and query identity."""

    timestep: int
    time: float
    episode_epoch: int
    configuration_epoch: int
    producer_surface_sha256: str
    manifest_sha256: str
    queries: QueryConfiguration
    metadata: Tuple[StructuredMetadata, ...]
    samples: Tuple[StructuredSample, ...]
    verification_only: bool = False
    schema: str = "specola.extraction-observation.v1"

    def __post_init__(self):
        if self.schema != "specola.extraction-observation.v1":
            raise ExtractionObservationError("unknown extraction observation schema")
        for field in ("timestep", "episode_epoch", "configuration_epoch"):
            _integer(getattr(self, field), field)
        if type(self.time) not in (int, float) or not math.isfinite(self.time):
            raise ExtractionObservationError("extraction time must be finite")
        _sha(self.producer_surface_sha256, "producer surface")
        _sha(self.manifest_sha256, "extraction manifest")
        if type(self.queries) is not QueryConfiguration or type(self.verification_only) is not bool:
            raise ExtractionObservationError("extraction configuration or verification mode has the wrong type")
        if type(self.metadata) is not tuple or any(type(item) is not StructuredMetadata for item in self.metadata):
            raise ExtractionObservationError("structured metadata must be typed records")
        if type(self.samples) is not tuple or any(type(item) is not StructuredSample for item in self.samples):
            raise ExtractionObservationError("structured samples must be typed records")
        known = {item.quantity_id for item in self.metadata}
        if len(known) != len(self.metadata):
            raise ExtractionObservationError("duplicate structured quantity metadata")
        addresses, scalars = set(), set()
        for sample in self.samples:
            if sample.quantity_id not in known:
                raise ExtractionObservationError("structured sample has no quantity metadata")
            if sample.address in addresses:
                raise ExtractionObservationError("duplicate structured quantity-instance address")
            addresses.add(sample.address)
            for key in sample.scalar_keys:
                address = sample.entity_id, key
                if address in scalars:
                    raise ExtractionObservationError("scalar output is claimed by multiple structured quantities")
                scalars.add(address)

    def project_scalar_links(self, entities: Mapping) -> ExtractionObservation:
        """Retain native values while projecting links to retained scalar keys.

        The existing scalar scope does not select structured-only outputs.
        Keep their configured native subject intact instead of guessing a
        structured scope from property patterns or dropping replay evidence.
        """
        samples = tuple(replace(sample, scalar_keys=tuple(key for key in sample.scalar_keys
                        if sample.entity_id in entities and key in entities[sample.entity_id].properties))
                        for sample in self.samples)
        return replace(self, samples=samples)

    def validate_observation(self, timestep, time, entities: Mapping) -> None:
        if self.timestep != timestep or self.time != time:
            raise ExtractionObservationError("scalar and structured observation instants differ")
        for sample in self.samples:
            if sample.entity_id is not None and sample.entity_id not in entities:
                raise ExtractionObservationError("structured sample refers to an absent entity")
            properties = getattr(entities.get(sample.entity_id), "properties", None)
            if sample.scalar_keys and (not isinstance(properties, Mapping) or any(key not in properties for key in sample.scalar_keys)):
                raise ExtractionObservationError("structured scalar link refers to an absent property")


def extraction_document(observation: ExtractionObservation) -> dict:
    """Serialize the independently versioned envelope without raw NaN JSON."""
    if type(observation) is not ExtractionObservation:
        raise ExtractionObservationError("expected a typed extraction observation")
    return {
        "schema": observation.schema, "timestep": observation.timestep, "time": observation.time,
        "episode_epoch": observation.episode_epoch, "configuration_epoch": observation.configuration_epoch,
        "producer_surface_sha256": observation.producer_surface_sha256, "manifest_sha256": observation.manifest_sha256,
        "queries": observation.queries.document(), "verification_only": observation.verification_only,
        "metadata": [dict(asdict(item), source_members=list(item.source_members)) for item in observation.metadata],
        "samples": [dict(quantity_id=item.quantity_id, owner_kind=item.owner_kind, owner_index=list(item.owner_index),
                         value=item.value.document(), availability=item.availability, entity_id=item.entity_id,
                         scalar_keys=list(item.scalar_keys), instance_token=item.instance_token,
                         binding_id=item.binding_id) for item in observation.samples],
    }


def extraction_from_document(document: dict) -> ExtractionObservation:
    """Restore declared data types without importing producer classes."""
    if type(document) is not dict:
        raise ExtractionObservationError("extraction observation document must be an object")
    try:
        _json_tree(document)
        if type(document["samples"]) is not list or any(type(item) is not dict for item in document["samples"]):
            raise ExtractionObservationError("structured sample documents must be a list of objects")
        # The trailing binding identity preserves documents written before query
        # populations acquired independent addresses. Do not mutate caller data.
        document = dict(document, samples=[dict(item, binding_id=item.get("binding_id", ""))
                                          for item in document["samples"]])
        metadata = []
        for item in document["metadata"]:
            fields: Dict[str, Any] = dict(item)
            fields["source_members"] = tuple(item["source_members"])
            metadata.append(StructuredMetadata(**fields))
        samples = []
        for item in document["samples"]:
            fields = dict(item)
            fields.update(owner_index=tuple(item["owner_index"]), scalar_keys=tuple(item["scalar_keys"]),
                          value=NativeValue.from_document(item["value"]))
            samples.append(StructuredSample(**fields))
        fields = dict(document)
        fields.update(metadata=tuple(metadata), samples=tuple(samples),
                      queries=QueryConfiguration.from_document(document["queries"]))
        result = ExtractionObservation(**fields)
        if extraction_document(result) != document:
            raise ExtractionObservationError("extraction observation document has a noncanonical shape")
        return result
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        if isinstance(error, ExtractionObservationError):
            raise
        raise ExtractionObservationError("invalid extraction observation document: " + str(error)) from error
