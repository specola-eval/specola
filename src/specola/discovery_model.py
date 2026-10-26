"""Immutable reset-time discovery records shared by every UI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .adapter import SimulatorAdapter
from .entities import EntityState
from .extraction_observation import ExtractionObservation, ExtractionObservationError


RAW_ENTITY_CLASSES = frozenset({
    "body", "camera", "geom", "light", "simulation", "site",
})


@dataclass(frozen=True)
class DiscoveryProperty:
    key: str
    value: float
    unit: Optional[str]
    family: Optional[str]
    group: Optional[str]
    note: Optional[str]
    component: Optional[str]
    frame: Optional[str]
    tags: Tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryEntity:
    entity_id: int
    entity_class: str
    label: Optional[str]
    role: Optional[str]
    role_basis: Optional[str]
    role_provenance: Optional[str]
    declared_name: Optional[str]
    declared_category: Optional[str]
    declaration_source: Optional[str]
    fixture_names: Tuple[str, ...]
    fixture_refs: Tuple[str, ...]
    fixture_type: Optional[str]
    fixture_provenance: Optional[str]
    region_fixture_names: Tuple[str, ...]
    region_fixture_refs: Tuple[str, ...]
    region_key: Optional[str]
    position: Optional[Tuple[float, ...]]
    velocity: Optional[Tuple[float, ...]]
    track_age: int
    track_state: str
    raw: bool
    properties: Tuple[DiscoveryProperty, ...]
    # Precomputed casefolded haystack for free-text search; computed once
    # at model build so per-keystroke filtering never rebuilds it.
    search_blob: str = ""


def entity_search_blob(entity: "DiscoveryEntity") -> str:
    values = [
        entity.entity_class,
        str(entity.entity_id),
        entity.label or "",
        entity.role or "",
        entity.role_basis or "",
        entity.role_provenance or "",
        entity.declared_name or "",
        entity.declared_category or "",
        entity.declaration_source or "",
        " ".join(entity.fixture_names),
        " ".join(entity.fixture_refs),
        entity.fixture_type or "",
        entity.fixture_provenance or "",
        " ".join(entity.region_fixture_names),
        " ".join(entity.region_fixture_refs),
        entity.region_key or "",
    ]
    for prop in entity.properties:
        values.extend((
            prop.key,
            prop.family or "",
            prop.group or "",
            prop.unit or "",
            prop.component or "",
            prop.frame or "",
            prop.note or "",
            " ".join(prop.tags),
        ))
    return " ".join(values).casefold()


@dataclass(frozen=True)
class DiscoveryClass:
    entity_class: str
    live_count: int
    property_count: int
    common_property_count: int
    raw: bool


@dataclass(frozen=True)
class DeclaredDiscoveryClass:
    entity_class: str
    live_count: int
    components: Tuple[str, ...]
    tags: Tuple[str, ...]
    runtime_populated: bool


@dataclass(frozen=True)
class DiscoveryFieldMaterialization:
    field_name: str
    binding_present: bool
    scene_applicable: bool
    emitted_keys: Tuple[str, ...]
    reason: Optional[str]
    # Why this field could not be vectorized into a gather plan, if it
    # could not. A performance fact, distinct from reason, which says
    # what the field can extract at all.
    plan_refusal: Optional[str] = None


@dataclass(frozen=True)
class DiscoveryFieldContract:
    included: Tuple[str, ...]
    excluded: Tuple[str, ...]
    unclassified: Tuple[str, ...]
    overlap: Tuple[str, ...]
    materialization: Tuple[DiscoveryFieldMaterialization, ...]


@dataclass(frozen=True)
class DiscoveryModel:
    schema_version: int
    simulator: str
    adapter_type: str
    task: Optional[str]
    task_description: Optional[str]
    adapter_dt: float
    effective_dt: float
    spatial_dims: int
    entities: Tuple[DiscoveryEntity, ...]
    classes: Tuple[DiscoveryClass, ...]
    declared_classes: Tuple[DeclaredDiscoveryClass, ...]
    warnings: Tuple[str, ...]
    field_contract: Optional[DiscoveryFieldContract]
    # The adapter's exact capabilities record. declared_classes above
    # carries three of its nine fields, which is enough to describe the
    # scene and not enough to compile against it: without class_mode and
    # supported_classes a reconstructed check falls back to OPEN
    # vocabulary and defers references a CLOSED adapter would reject.
    capabilities: Optional[object] = None
    extraction: Optional[ExtractionObservation] = None


def _optional_mapping(adapter: SimulatorAdapter, name: str) -> Mapping:
    value = getattr(adapter, name, None)
    return value if isinstance(value, Mapping) else {}


def _vector(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    return tuple(float(component) for component in value)


def _property_record(key: str, value: Any, metadata: Any) -> DiscoveryProperty:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("discovery property {!r} is not a scalar number".format(key))
    scalar = float(value)
    # Non-finite values are valid observations and have an explicit JSON
    # representation; touching isfinite here documents that they are accepted.
    isfinite(scalar)
    group = getattr(metadata, "group", None)
    note = getattr(metadata, "note", None)
    # PropertyMeta.note is the authoritative unit/component annotation in
    # the current adapter contract. Richer third-party metadata may expose
    # dedicated fields; transport those without reconstructing semantics
    # from the property key.
    declared_unit = getattr(metadata, "unit", None)
    unit = declared_unit if declared_unit is not None else note
    family = getattr(metadata, "family", None)
    component = getattr(metadata, "component", None)
    frame = getattr(metadata, "frame", None)
    tags = tuple(getattr(metadata, "tags", ()) or ())
    if any(not isinstance(tag, str) or not tag for tag in tags):
        raise TypeError("property metadata tags must be non-empty strings")
    return DiscoveryProperty(
        key=key,
        value=scalar,
        unit=str(unit) if unit is not None else None,
        family=str(family) if family is not None else None,
        group=str(group) if group is not None else None,
        note=str(note) if note is not None else None,
        component=(
            str(component) if component is not None else None
        ),
        frame=str(frame) if frame is not None else None,
        tags=tags,
    )


def property_metadata_for(
    adapter: SimulatorAdapter,
    entities: Iterable[int],
) -> Dict[int, Mapping[str, Any]]:
    """Read adapter-declared property metadata once for shared consumers."""
    metadata: Dict[int, Mapping[str, Any]] = {}
    callback = getattr(adapter, "property_metadata", None)
    if not callable(callback):
        return metadata
    for entity_id in entities:
        try:
            value = callback(entity_id)
        except Exception as exc:
            raise RuntimeError(
                "property_metadata() failed for adapter {} entity {}".format(
                    type(adapter).__name__, entity_id
                )
            ) from exc
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise TypeError("property_metadata() must return a mapping")
        value = adapter._project_property_metadata(entity_id, value)
        metadata[entity_id] = value
    return metadata


def _warnings(
    adapter: SimulatorAdapter,
    entities: Mapping[int, EntityState],
) -> Tuple[str, ...]:
    callback = getattr(adapter, "discovery_warnings", None)
    if not callable(callback):
        return ()
    values = tuple(callback(dict(entities)))
    for warning in values:
        if not isinstance(warning, str) or not warning.strip():
            raise TypeError(
                "discovery_warnings() must return non-empty strings"
            )
    return values


def _classes(entities: Tuple[DiscoveryEntity, ...]) -> Tuple[DiscoveryClass, ...]:
    records = []
    for entity_class in sorted({entity.entity_class for entity in entities}):
        members = tuple(
            entity for entity in entities if entity.entity_class == entity_class
        )
        property_sets = [
            {prop.key for prop in entity.properties} for entity in members
        ]
        distinct = set().union(*property_sets) if property_sets else set()
        common = set.intersection(*property_sets) if property_sets else set()
        records.append(DiscoveryClass(
            entity_class=entity_class,
            live_count=len(members),
            property_count=len(distinct),
            common_property_count=len(common),
            raw=entity_class in RAW_ENTITY_CLASSES,
        ))
    return tuple(records)


def _declared_classes(
    adapter: SimulatorAdapter,
    classes: Tuple[DiscoveryClass, ...],
) -> Tuple[DeclaredDiscoveryClass, ...]:
    capabilities = getattr(adapter, "capabilities", None)
    if capabilities is None:
        return ()
    components = getattr(capabilities, "components_by_class", {}) or {}
    tags = getattr(capabilities, "component_tags_by_class", {}) or {}
    dynamic = frozenset(
        getattr(capabilities, "runtime_populated_classes", ()) or ()
    )
    live = {record.entity_class: record.live_count for record in classes}
    names = sorted(set(components) | set(dynamic))
    return tuple(
        DeclaredDiscoveryClass(
            entity_class=name,
            live_count=live.get(name, 0),
            components=tuple(sorted(components.get(name, ()))),
            tags=tuple(sorted(tags.get(name, ()))),
            runtime_populated=name in dynamic,
        )
        for name in names
    )


def _field_contract(
    adapter: SimulatorAdapter,
) -> Optional[DiscoveryFieldContract]:
    report = getattr(adapter, "field_surface_report", None)
    if report is None:
        return None
    materialization = getattr(adapter, "field_materialization", {}) or {}
    records = tuple(
        DiscoveryFieldMaterialization(
            field_name=str(name),
            binding_present=bool(getattr(record, "binding_present", False)),
            scene_applicable=bool(getattr(record, "scene_applicable", False)),
            emitted_keys=tuple(getattr(record, "emitted_keys", ()) or ()),
            reason=(
                str(getattr(record, "reason"))
                if getattr(record, "reason", None) is not None else None
            ),
            plan_refusal=(
                str(getattr(record, "plan_refusal"))
                if getattr(record, "plan_refusal", None) is not None else None
            ),
        )
        for name, record in sorted(materialization.items())
    )
    return DiscoveryFieldContract(
        included=tuple(getattr(report, "included", ()) or ()),
        excluded=tuple(getattr(report, "excluded", ()) or ()),
        unclassified=tuple(getattr(report, "unclassified", ()) or ()),
        overlap=tuple(getattr(report, "overlap", ()) or ()),
        materialization=records,
    )


def _derived_capabilities(adapter):
    """The adapter's capabilities, declared or derived.

    ``getattr(adapter, "capabilities", None)`` is not enough: an adapter
    that declares none still has a class vocabulary, derived from
    ``supported_classes``, and persisting None there would turn a CLOSED
    vocabulary into an OPEN one on the way back in.
    """
    from .capabilities import adapter_capabilities

    try:
        return adapter_capabilities(adapter)
    except Exception:
        return getattr(adapter, "capabilities", None)


def build_discovery_model(
    adapter: SimulatorAdapter,
    entities: Mapping[int, EntityState],
    *,
    simulator: str,
    task: Optional[str],
    effective_dt: float,
    task_description: Optional[str] = None,
    property_metadata: Optional[
        Mapping[int, Mapping[str, Any]]
    ] = None,
    capabilities: Optional[object] = None,
    extraction: Optional[ExtractionObservation] = None,
) -> DiscoveryModel:
    """Build the complete UI model without resetting the adapter."""
    if extraction is not None:
        if type(extraction) is not ExtractionObservation or extraction.verification_only:
            raise ExtractionObservationError("ordinary discovery requires a non-verification extraction observation")
        extraction.validate_observation(extraction.timestep, extraction.time, entities)
    labels = _optional_mapping(adapter, "entity_labels")
    roles = _optional_mapping(adapter, "entity_roles")
    role_basis = _optional_mapping(adapter, "entity_role_basis")
    role_provenance = _optional_mapping(adapter, "entity_role_provenance")
    declared_names = _optional_mapping(adapter, "entity_declared_names")
    declared_categories = _optional_mapping(
        adapter, "entity_declared_categories"
    )
    declaration_sources = _optional_mapping(
        adapter, "entity_declaration_sources"
    )
    fixture_names = _optional_mapping(adapter, "entity_fixture_names")
    fixture_refs = _optional_mapping(adapter, "entity_fixture_refs")
    fixture_types = _optional_mapping(adapter, "entity_fixture_types")
    fixture_provenance = _optional_mapping(
        adapter, "entity_fixture_provenance"
    )
    region_fixture_names = _optional_mapping(
        adapter, "entity_region_fixture_names"
    )
    region_fixture_refs = _optional_mapping(
        adapter, "entity_region_fixture_refs"
    )
    region_keys = _optional_mapping(adapter, "entity_region_keys")
    metadata = (
        property_metadata_for(adapter, entities)
        if property_metadata is None
        else property_metadata
    )

    records = []
    for entity_id in sorted(entities):
        entity = entities[entity_id]
        if entity.entity_id != entity_id:
            raise ValueError(
                "entity mapping key {} disagrees with entity_id {}".format(
                    entity_id, entity.entity_id
                )
            )
        entity_metadata = metadata.get(entity_id, {})
        properties = tuple(
            _property_record(key, entity.properties[key], entity_metadata.get(key))
            for key in sorted(entity.properties)
        )
        records.append(DiscoveryEntity(
            entity_id=entity_id,
            entity_class=entity.entity_class,
            label=str(labels[entity_id]) if entity_id in labels else None,
            role=str(roles[entity_id]) if entity_id in roles else None,
            role_basis=(
                str(role_basis[entity_id]) if entity_id in role_basis else None
            ),
            role_provenance=(
                str(role_provenance[entity_id])
                if entity_id in role_provenance else None
            ),
            declared_name=(
                str(declared_names[entity_id])
                if entity_id in declared_names else None
            ),
            declared_category=(
                str(declared_categories[entity_id])
                if entity_id in declared_categories else None
            ),
            declaration_source=(
                str(declaration_sources[entity_id])
                if entity_id in declaration_sources else None
            ),
            fixture_names=tuple(
                str(value)
                for value in fixture_names.get(entity_id, ())
            ),
            fixture_refs=tuple(
                str(value)
                for value in fixture_refs.get(entity_id, ())
            ),
            fixture_type=(
                str(fixture_types[entity_id])
                if entity_id in fixture_types else None
            ),
            fixture_provenance=(
                str(fixture_provenance[entity_id])
                if entity_id in fixture_provenance else None
            ),
            region_fixture_names=tuple(
                str(value)
                for value in region_fixture_names.get(entity_id, ())
            ),
            region_fixture_refs=tuple(
                str(value)
                for value in region_fixture_refs.get(entity_id, ())
            ),
            region_key=(
                str(region_keys[entity_id])
                if entity_id in region_keys else None
            ),
            position=_vector(entity.position),
            velocity=_vector(entity.velocity),
            track_age=int(entity.track_age),
            track_state=str(entity.track_state.value),
            raw=entity.entity_class in RAW_ENTITY_CLASSES,
            properties=properties,
        ))

    immutable_entities = tuple(
        replace(record, search_blob=entity_search_blob(record))
        for record in records
    )
    classes = _classes(immutable_entities)
    return DiscoveryModel(
        schema_version=1,
        simulator=simulator,
        adapter_type=type(adapter).__name__,
        task=task,
        task_description=task_description,
        adapter_dt=float(adapter.dt),
        effective_dt=float(effective_dt),
        spatial_dims=int(adapter.spatial_dims),
        entities=immutable_entities,
        classes=classes,
        declared_classes=_declared_classes(adapter, classes),
        warnings=_warnings(adapter, entities),
        field_contract=_field_contract(adapter),
        extraction=extraction,
        capabilities=(
            capabilities
            if capabilities is not None
            else _derived_capabilities(adapter)
        ),
    )
