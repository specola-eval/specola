"""
capabilities.py — machine-readable adapter authority (design §5.6.1).

Separates what an adapter can theoretically emit (capabilities, with
open/closed vocabulary modes) from what was discovered at reset
(AdapterResetResult.discovered_catalog) and from what is visible in any
single observation. Closed vocabularies allow early rejection of
impossible specifications; open vocabularies defer unknown references.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, cast

from .adapter import MonitorObservation
from .components import ComponentCatalog, EntityComponentInfo
from .discovery_model import property_metadata_for


class VocabularyMode(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class AdapterCapabilities:
    spatial_dims: int
    class_mode: VocabularyMode = VocabularyMode.OPEN
    supported_classes: FrozenSet[str] = frozenset()
    components_by_class: Mapping[str, FrozenSet[str]] = field(
        default_factory=dict
    )
    component_mode_by_class: Mapping[str, VocabularyMode] = field(
        default_factory=dict
    )
    # Possible structural tags a declared-CLOSED-schema dynamic class may
    # carry at runtime (per-key ``PropertyMeta.tags`` content), declared
    # beside components_by_class. An ``@``-filtered exact-key selector on
    # such a class compiles against this vocabulary and keeps its filter
    # for per-step binding eligibility instead of desugaring at reset.
    # Empty for classes whose tags are resolvable from the reset catalog.
    component_tags_by_class: Mapping[str, FrozenSet[str]] = field(
        default_factory=dict
    )
    # Declared key -> PropertyMeta.group mapping for a CLOSED-schema
    # class, when the adapter's groups are schema facts (fixed key sets
    # imply fixed groups). Lets an ``@ group`` selector filter resolve
    # statically on a runtime-populated class, where the reset catalog
    # cannot be consulted. Empty when undeclared: group filters on such
    # classes then fail compilation naming the declared vocabularies.
    component_groups_by_class: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    # Entity classes whose instance population can legitimately be empty at
    # reset and gain members later in the same episode.
    runtime_populated_classes: FrozenSet[str] = frozenset()
    entity_id_mode: VocabularyMode = VocabularyMode.OPEN
    extraction_status: Optional[str] = None

    def __post_init__(self):
        if self.extraction_status is not None and self.extraction_status not in (
            "unaudited_custom_adapter", "unaudited_third_party_provider",
        ):
            raise ValueError("unsupported extraction capability status")


@dataclass(frozen=True)
class AdapterResetResult:
    initial_observation: MonitorObservation
    discovered_catalog: ComponentCatalog
    capabilities: AdapterCapabilities
    # The property metadata read during this reset. Carried out rather
    # than recomputed: ``property_metadata()`` reads live adapter state,
    # and a second call after the extraction scope is installed can see
    # a narrower surface than the one the episode compiled against.
    property_metadata: Mapping[int, Mapping[str, object]] = field(
        default_factory=dict
    )


def adapter_capabilities(adapter) -> AdapterCapabilities:
    """Combine declared vocabulary with the implementation's guarded status.

    An absent declaration retains the established supported-class default.
    The extraction status comes from the accepted reset subject; an adapter
    cannot promote its completeness by putting a label in its capabilities.
    """
    from .conformance.provider import adapter_extraction_status

    declared = getattr(adapter, "capabilities", None)
    if declared is None:
        supported = adapter.supported_classes
        declared = AdapterCapabilities(
            spatial_dims=int(adapter.spatial_dims),
            class_mode=VocabularyMode.CLOSED if supported is not None else VocabularyMode.OPEN,
            supported_classes=supported or frozenset(),
        )
    status = adapter_extraction_status(adapter)
    if isinstance(declared, AdapterCapabilities):
        return declared if declared.extraction_status == status else replace(declared, extraction_status=status)
    # Preserve structurally declared legacy capability objects while carrying
    # the status in the same standard representation used by persistence.
    return AdapterCapabilities(
        **{item.name: getattr(declared, item.name) for item in fields(AdapterCapabilities)
           if item.name != "extraction_status"},
        extraction_status=status,
    )


def reset_adapter(adapter) -> AdapterResetResult:
    """Reset an adapter and package the episode-start discovery result.

    Works with any SimulatorAdapter: the catalog is built from the
    initial observation's entities (model-level discovery) plus any the
    reset's entity scope left unread (``reset_catalog``), and
    capabilities come from the adapter's optional ``capabilities``
    attribute or are derived from ``supported_classes`` (closed class
    vocabulary when declared, open otherwise). ``property_metadata``
    covers the observed entities only.
    """
    observation = adapter.reset()

    metadata = property_metadata_for(
        adapter, observation.perceived_entities
    )

    catalog, _ = reset_catalog(
        adapter, observation.perceived_entities, metadata
    )

    return AdapterResetResult(
        initial_observation=observation,
        discovered_catalog=catalog,
        capabilities=adapter_capabilities(adapter),
        property_metadata=metadata,
    )


def reset_catalog(
    adapter,
    entities: Mapping[int, Any],
    metadata: Mapping[int, Mapping[str, Any]],
) -> Tuple[ComponentCatalog, Dict[int, Mapping[str, Any]]]:
    """Catalog a reset's entities and those its entity scope left unread.

    An adapter that reads a class entity by entity can reset under an
    entity scope that admits none of them (a monitor session's
    catalog-only reset, issue #136); its optional
    ``unread_entity_metadata()`` then maps each unread entity ID to its
    class and the property metadata an unscoped read would register. The
    catalog lists them exactly as it would list them read, so compiling
    and ``specola check`` see the whole scene. Returns the catalog and
    ``metadata`` extended by the unread entities' metadata.
    """
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    hook = getattr(adapter, "unread_entity_metadata", None)
    unread = cast(
        Mapping[int, Tuple[str, Mapping[str, Any]]],
        hook() if callable(hook) else {},
    )
    if not unread:
        return catalog, dict(metadata)
    infos = dict(catalog.entities)
    extended = dict(metadata)
    for entity_id, (entity_class, entity_metadata) in unread.items():
        if entity_id in infos:
            raise ValueError(
                "unread entity {} is also in the observation".format(entity_id)
            )
        extended[entity_id] = entity_metadata
        infos[entity_id] = EntityComponentInfo(
            entity_id,
            entity_class,
            entity_metadata,
            {key: meta.group for key, meta in entity_metadata.items()},
            {key: tuple(getattr(meta, "tags", ()) or ()) for key, meta in entity_metadata.items()},
        )
    return ComponentCatalog(infos), extended
