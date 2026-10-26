"""Rebuild a discovery model from its serialized document.

The inverse of ``discovery_serialize`` for everything a check reads, and
the thing that lets a finished recording be browsed and checked with no
simulator installed. A document missing a field a check depends on is
refused rather than reconstructed approximately, because the consumer
compiles specifications against the result, and a silently degraded
scene produces a check whose verdict does not match what the run would
have said.

Two things are deliberately *not* round-trips, and saying so here is
cheaper than a reader discovering it:

``search_blob`` is recomputed from the restored fields rather than
carried, because it is derived and the derivation is the definition.

``field_contract`` is dropped entirely. It records which adapter fields
were included, excluded, unclassified or overlapping, and it is read by
exactly one thing -- ``discover --field-contract``'s MuJoCo diagnostic
display. Nothing in the check path consults it, so carrying it would add
bytes to every recording to serve a live-only diagnostic. A restored
model therefore has ``field_contract=None``, and that is not a
degradation of the check, only the absence of a display.

The capabilities record travels because it is the adapter's declared
authority over its own vocabulary, and a check is only equivalent to
the live one if it compiles against the same authority.

What that buys is worth stating precisely, because it is easy to
overstate. On the adapters available here no draft's verdict changes
when capabilities are dropped: the component catalogue and the
reference parser already reject an unknown class or key, and
``semantic.py`` treats absent capabilities as *closed* rather than open,
so losing them makes a check stricter rather than looser. The fields
that could diverge — ``component_mode_by_class`` and
``runtime_populated_classes``, which govern declared-but-empty classes
— were tried and did not flip a verdict either.

So this is a fidelity measure, not a fix for a demonstrated
over-acceptance: the two contexts are given identical inputs so they
cannot drift, rather than because a divergence has been seen. A
document without capabilities is refused rather than restored with
substitutes, which makes recordings written before the sidecar
browsable but not checkable — a stated limitation, chosen over a check
whose authority silently differs from the run's.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from .capabilities import AdapterCapabilities, VocabularyMode
from .extraction_observation import extraction_from_document
from .discovery_model import (
    DiscoveryClass,
    DiscoveryEntity,
    DiscoveryModel,
    DiscoveryProperty,
    DeclaredDiscoveryClass,
    entity_search_blob,
)
from .entities import EntityState, TrackState


class DiscoveryRestoreError(ValueError):
    """The document cannot be turned back into a usable model."""


def _number(value: Any) -> float:
    """Undo the serializer's non-finite spellings."""
    if isinstance(value, str):
        if value == "NaN":
            return math.nan
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        raise DiscoveryRestoreError(
            "not a number: {!r}".format(value)
        )
    return float(value)


def capabilities_from_document(document: Mapping) -> AdapterCapabilities:
    """Rebuild the exact capabilities the episode compiled against."""
    encoded = document.get("capabilities")
    if not encoded:
        raise DiscoveryRestoreError(
            "This recording carries no adapter capabilities, so a check "
            "against it cannot reproduce the live one. Recordings made "
            "before capabilities were persisted are browsable but not "
            "checkable."
        )
    try:
        return AdapterCapabilities(
            spatial_dims=int(encoded["spatial_dims"]),
            class_mode=VocabularyMode(encoded["class_mode"]),
            supported_classes=frozenset(encoded["supported_classes"]),
            components_by_class={
                name: frozenset(components)
                for name, components in encoded[
                    "components_by_class"
                ].items()
            },
            component_mode_by_class={
                name: VocabularyMode(value)
                for name, value in encoded[
                    "component_mode_by_class"
                ].items()
            },
            component_tags_by_class={
                name: frozenset(tags)
                for name, tags in encoded[
                    "component_tags_by_class"
                ].items()
            },
            component_groups_by_class={
                name: dict(groups)
                for name, groups in encoded[
                    "component_groups_by_class"
                ].items()
            },
            runtime_populated_classes=frozenset(
                encoded["runtime_populated_classes"]
            ),
            entity_id_mode=VocabularyMode(encoded["entity_id_mode"]),
            extraction_status=encoded.get("extraction_status"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DiscoveryRestoreError(
            "capabilities record is malformed: {}".format(exc)
        ) from exc


def _property(encoded: Mapping) -> DiscoveryProperty:
    return DiscoveryProperty(
        key=str(encoded["key"]),
        value=_number(encoded["value"]),
        unit=encoded.get("unit"),
        family=encoded.get("family"),
        group=encoded.get("group"),
        note=encoded.get("note"),
        component=encoded.get("component"),
        frame=encoded.get("frame"),
        tags=tuple(encoded.get("tags") or ()),
    )


def _entity(encoded: Mapping) -> DiscoveryEntity:
    record = DiscoveryEntity(
        entity_id=int(encoded["id"]),
        entity_class=str(encoded["class"]),
        label=encoded.get("label"),
        role=encoded.get("role"),
        role_basis=encoded.get("role_basis"),
        role_provenance=encoded.get("role_provenance"),
        declared_name=encoded.get("declared_name"),
        declared_category=encoded.get("declared_category"),
        declaration_source=encoded.get("declaration_source"),
        fixture_names=tuple(encoded.get("fixture_names") or ()),
        fixture_refs=tuple(encoded.get("fixture_refs") or ()),
        fixture_type=encoded.get("fixture_type"),
        fixture_provenance=encoded.get("fixture_provenance"),
        region_fixture_names=tuple(
            encoded.get("region_fixture_names") or ()
        ),
        region_fixture_refs=tuple(
            encoded.get("region_fixture_refs") or ()
        ),
        region_key=encoded.get("region_key"),
        position=(
            tuple(_number(value) for value in encoded["position"])
            if encoded.get("position") is not None else None
        ),
        velocity=(
            tuple(_number(value) for value in encoded["velocity"])
            if encoded.get("velocity") is not None else None
        ),
        track_age=int(encoded.get("track_age") or 0),
        track_state=str(encoded.get("track_state") or "confirmed"),
        raw=bool(encoded.get("raw")),
        properties=tuple(
            _property(prop) for prop in encoded.get("properties") or ()
        ),
    )
    # Recomputed rather than persisted: the blob is a search index, and
    # an index restored from disk can disagree with the record it
    # indexes if either side's construction changes.
    from dataclasses import replace

    return replace(record, search_blob=entity_search_blob(record))


def discovery_model_from_document(document: Mapping) -> DiscoveryModel:
    """Rebuild the model, or raise explaining why it cannot be."""
    if not isinstance(document, Mapping):
        raise DiscoveryRestoreError("document is not a mapping")
    for flag in ("verification_only", "audit_only", "schema_only"):
        if flag in document and document[flag] is not False:
            raise DiscoveryRestoreError("declaration or evidence-only document cannot become ordinary discovery: " + flag)
    if document.get("schema") in ("specola.extraction-audit.v1", "specola.extraction-authoring.v1"):
        raise DiscoveryRestoreError("declaration document cannot become ordinary discovery")
    if int(document.get("schema_version", 0)) != 1:
        raise DiscoveryRestoreError(
            "unsupported discovery schema_version {!r}".format(
                document.get("schema_version")
            )
        )
    try:
        extraction = extraction_from_document(document["extraction"]) if "extraction" in document else None
        if extraction is not None and extraction.verification_only:
            raise DiscoveryRestoreError("candidate extraction cannot become ordinary discovery")
        entities = tuple(
            _entity(encoded) for encoded in document["entities"]
        )
        classes = tuple(
            DiscoveryClass(
                entity_class=str(record["class"]),
                live_count=int(record["live_count"]),
                property_count=int(record["property_count"]),
                common_property_count=int(
                    record["common_property_count"]
                ),
                raw=bool(record["raw"]),
            )
            for record in document["classes"]
        )
        declared = tuple(
            DeclaredDiscoveryClass(
                entity_class=str(record["class"]),
                live_count=int(record["live_count"]),
                components=tuple(record["components"]),
                tags=tuple(record["tags"]),
                runtime_populated=bool(record["runtime_populated"]),
            )
            for record in document["declared_classes"]
        )
        timing = document["timing"]
        return DiscoveryModel(
            schema_version=1,
            simulator=str(document["simulator"]),
            adapter_type=str(document["adapter_type"]),
            task=document.get("task"),
            task_description=document.get("task_description"),
            adapter_dt=float(timing["adapter_dt"]),
            effective_dt=float(timing["effective_dt"]),
            spatial_dims=int(document["spatial_dims"]),
            entities=entities,
            classes=classes,
            declared_classes=declared,
            warnings=tuple(document.get("warnings") or ()),
            # Not carried by the sidecar: see the module docstring. Read
            # only by ``discover --field-contract``, never by a check.
            field_contract=None,
            capabilities=capabilities_from_document(document),
            extraction=extraction,
        )
    except DiscoveryRestoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise DiscoveryRestoreError(
            "discovery document is malformed: {}".format(exc)
        ) from exc


def entity_states_from_model(
    model: DiscoveryModel,
) -> Dict[int, EntityState]:
    """The ``EntityState`` mapping a checker needs, from the model.

    Values are the reset-time ones the document recorded. That is what a
    check wants: it compiles against the scene's shape, and the shape is
    fixed at reset even though the values move.
    """
    import numpy as np

    states: Dict[int, EntityState] = {}
    for entity in model.entities:
        states[entity.entity_id] = EntityState(
            entity_id=entity.entity_id,
            entity_class=entity.entity_class,
            track_age=entity.track_age,
            track_state=TrackState(entity.track_state),
            position=(
                np.asarray(entity.position, dtype=float)
                if entity.position is not None else None
            ),
            velocity=(
                np.asarray(entity.velocity, dtype=float)
                if entity.velocity is not None else None
            ),
            properties={
                prop.key: prop.value for prop in entity.properties
            },
        )
    return states


def property_metadata_from_model(
    model: DiscoveryModel,
) -> Dict[int, Dict[str, Any]]:
    """Rebuild the per-key metadata the catalogue and selectors read.

    Groups and tags are authoring vocabulary, not decoration: an
    ``@ label`` selector filters on them, so a catalogue rebuilt without
    them resolves differently from the live one.
    """
    from .adapter import PropertyMeta

    metadata: Dict[int, Dict[str, Any]] = {}
    for entity in model.entities:
        per_key: Dict[str, Any] = {}
        for prop in entity.properties:
            per_key[prop.key] = PropertyMeta(
                group=prop.group or "",
                note=prop.note,
                tags=tuple(prop.tags),
                unit=prop.unit,
                family=prop.family,
                component=prop.component,
                frame=prop.frame,
            )
        metadata[entity.entity_id] = per_key
    return metadata


def catalog_from_model(model: DiscoveryModel):
    """The ``ComponentCatalog`` a checker compiles against."""
    from .components import ComponentCatalog

    return ComponentCatalog.from_entities(
        entity_states_from_model(model),
        metadata=property_metadata_from_model(model),
    )


def labels_from_model(model: DiscoveryModel) -> Dict[int, str]:
    return {
        entity.entity_id: entity.label
        for entity in model.entities
        if entity.label
    }
