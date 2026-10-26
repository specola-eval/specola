"""Stable JSON representation of discovery views."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from .discovery_filter import DiscoveryView
from .extraction_observation import ExtractionObservationError, extraction_document


def _number(value: Optional[float]) -> Any:
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def capabilities_document(capabilities) -> Optional[dict]:
    """Declared capability fields in a stable order.

    ``declared_classes`` elsewhere in this document describes the scene
    and carries three of these. The rest travel so that a check rebuilt
    from this document compiles against the adapter's own declared
    authority rather than a reconstruction of it.

    No verdict divergence has been demonstrated from dropping them — the
    catalogue and the parser reject unknown classes and keys on their
    own, and absent capabilities are treated as closed rather than open.
    They are persisted so the live and offline contexts cannot drift,
    not because a drift has been observed.

    Sets become sorted lists and enums become their values, so the same
    capabilities always encode to the same bytes and the sidecar's
    digest means something.
    """
    if capabilities is None:
        return None

    def mode(value):
        return getattr(value, "value", value)

    document = {
        "spatial_dims": int(capabilities.spatial_dims),
        "class_mode": mode(capabilities.class_mode),
        "supported_classes": sorted(capabilities.supported_classes or ()),
        "components_by_class": {
            name: sorted(components)
            for name, components in sorted(
                (capabilities.components_by_class or {}).items()
            )
        },
        "component_mode_by_class": {
            name: mode(value)
            for name, value in sorted(
                (capabilities.component_mode_by_class or {}).items()
            )
        },
        "component_tags_by_class": {
            name: sorted(tags)
            for name, tags in sorted(
                (capabilities.component_tags_by_class or {}).items()
            )
        },
        "component_groups_by_class": {
            name: dict(sorted(groups.items()))
            for name, groups in sorted(
                (capabilities.component_groups_by_class or {}).items()
            )
        },
        "runtime_populated_classes": sorted(
            capabilities.runtime_populated_classes or ()
        ),
        "entity_id_mode": mode(capabilities.entity_id_mode),
    }
    status = getattr(capabilities, "extraction_status", None)
    if status is not None:
        document["extraction_status"] = status
    return document


def selector_palette_document(view: DiscoveryView) -> dict:
    """Every selector form this view's filter can express.

    The shape metadata travels — a consumer that wants to group by
    binding token or explain what a row means needs it — but the
    resolved ``bindings`` and ``monitors`` do not. The document already
    carries every entity and property, so the expansion is derivable
    from what is here, and duplicating it would put a count into the one
    place a browser reads its palette rows from.

    ``forms`` is empty exactly when ``reason`` is set: a filter spanning
    several classes has no single reference, because a reference names
    one class.
    """
    from .selector_palette import palette_for_view

    forms, reason = palette_for_view(view)
    return {
        "reason": reason,
        "forms": [
            {
                "reference": form.reference,
                "partner_reference": form.partner_reference,
                "binding_kind": form.binding_kind,
                "component_kind": form.component_kind,
                "entity_form": form.entity_form,
                "component_form": form.component_form,
                "quantifier": form.quantifier,
                "note": form.note,
                "available": form.available,
                "unavailable_reason": form.unavailable_reason,
                # Additive, nullable, and the version stays 1. Without
                # it a consumer cannot tell a row the compiler accepted
                # from one nothing checked -- and those disagree: ten of
                # RoboCasa's sixteen contact forms are bracket-available
                # and illegal.
                "verified": form.verified,
                "expansions": form.expansions,
            }
            for form in forms
        ],
    }


def discovery_document(
    view: DiscoveryView,
    *,
    include_field_contract: bool = False,
) -> dict:
    model = view.model
    if model.extraction is not None and model.extraction.verification_only:
        raise ExtractionObservationError("verification-only observations cannot become ordinary discovery")
    document = {
        # Precedent (issue #98): an added key that is nullable and that
        # no reader has to know about keeps the version -- a consumer
        # written against the old schema still parses the document. The
        # key-set pin in tests/test_discovery_serialize.py is what makes
        # each such extension a deliberate act rather than a drift.
        "schema_version": model.schema_version,
        "simulator": model.simulator,
        "adapter_type": model.adapter_type,
        "task": model.task,
        "task_description": model.task_description,
        "timing": {
            "adapter_dt": model.adapter_dt,
            "effective_dt": model.effective_dt,
        },
        "spatial_dims": model.spatial_dims,
        "filters": {
            "class": list(view.query.classes),
            "entity": list(view.query.entities),
            "signal": list(view.query.signals),
            "group": list(view.query.groups),
            "role": list(view.query.roles),
            "tag": list(view.query.tags),
            "search": list(view.query.search),
            "include_raw": view.query.include_raw,
        },
        "counts": {
            "before": {
                "entities": view.before.entity_count,
                "properties": view.before.property_count,
            },
            "after": {
                "entities": view.after.entity_count,
                "properties": view.after.property_count,
            },
        },
        "classes": [
            {
                "class": record.entity_class,
                "live_count": record.live_count,
                "property_count": record.property_count,
                "common_property_count": record.common_property_count,
                "raw": record.raw,
            }
            for record in model.classes
        ],
        "entities": [
            {
                "id": entity.entity_id,
                "class": entity.entity_class,
                "label": entity.label,
                "role": entity.role,
                "role_basis": entity.role_basis,
                "role_provenance": entity.role_provenance,
                "declared_name": entity.declared_name,
                "declared_category": entity.declared_category,
                "declaration_source": entity.declaration_source,
                "fixture_names": list(entity.fixture_names),
                "fixture_refs": list(entity.fixture_refs),
                "fixture_type": entity.fixture_type,
                "fixture_provenance": entity.fixture_provenance,
                "region_fixture_names": list(entity.region_fixture_names),
                "region_fixture_refs": list(entity.region_fixture_refs),
                "region_key": entity.region_key,
                "position": (
                    [_number(value) for value in entity.position]
                    if entity.position is not None else None
                ),
                "velocity": (
                    [_number(value) for value in entity.velocity]
                    if entity.velocity is not None else None
                ),
                "track_age": entity.track_age,
                "track_state": entity.track_state,
                "raw": entity.raw,
                "properties": [
                    {
                        "key": prop.key,
                        "value": _number(prop.value),
                        "unit": prop.unit,
                        "family": prop.family,
                        "group": prop.group,
                        "note": prop.note,
                        "component": prop.component,
                        "frame": prop.frame,
                        "tags": list(prop.tags),
                    }
                    for prop in entity.properties
                ],
            }
            for entity in view.entities
        ],
        "declared_classes": [
            {
                "class": record.entity_class,
                "live_count": record.live_count,
                "components": list(record.components),
                "tags": list(record.tags),
                "runtime_populated": record.runtime_populated,
            }
            for record in model.declared_classes
        ],
        "warnings": list(model.warnings),
        "selector_palette": selector_palette_document(view),
        "capabilities": capabilities_document(model.capabilities),
    }
    if model.extraction is not None:
        document["extraction"] = extraction_document(model.extraction)
    if include_field_contract:
        contract = model.field_contract
        document["field_contract"] = None if contract is None else {
            "included": list(contract.included),
            "excluded": list(contract.excluded),
            "unclassified": list(contract.unclassified),
            "overlap": list(contract.overlap),
            "materialization": [
                {
                    "field_name": record.field_name,
                    "binding_present": record.binding_present,
                    "scene_applicable": record.scene_applicable,
                    "emitted_keys": list(record.emitted_keys),
                    "reason": record.reason,
                    "plan_refusal": record.plan_refusal,
                }
                for record in contract.materialization
            ],
        }
    return document


def discovery_json(
    view: DiscoveryView,
    *,
    include_field_contract: bool = False,
) -> str:
    return json.dumps(
        discovery_document(
            view,
            include_field_contract=include_field_contract,
        ),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
