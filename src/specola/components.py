"""
components.py — Discovered Entity Components and Cached Resolution
==================================================================

Provides the shared component model used by the STL parser and binding layer.

The core package defines no built-in component vocabulary. Every
monitorable scalar signal is discovered from:

    EntityState.properties

Examples include:

    gx, gy, gz
    hw, hh, hd
    x0, x1, y0, y1, z0, z1
    grip
    hinge_angle
    joint_1_qpos
    joint_1_qvel
    recent_ee_force_norm

Adapters are responsible for assigning meaningful, stable property names.
Derived simulator-specific values, such as region bounds, should normally be
materialized by the adapter and exposed as direct properties.

ComponentCatalog
----------------
An episode-level description of the components exposed by discovered entities.
The parser uses it to validate class and component references.

ComponentResolver
-----------------
An observation-scoped resolver. It checks component availability and caches
values used repeatedly across predicates and bindings.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    cast,
)

from .entities import EntityState, SpecConfigError


_COMPONENT_NAME_RE = re.compile(
    r"^[a-z_][a-z0-9_]*$"
)

# Sentinel for cache lookups whose stored value may legitimately be falsy.
# ``supports`` caches booleans, so ``dict.get(key)`` alone cannot distinguish
# "cached False" from "absent" and would recompute every negative answer.
_CACHE_MISS = object()


def _require_component_str(
    component: Any,
) -> None:
    """
    Reject a non-string component name.

    Shared by ``validate_component_name`` and ``ComponentResolver.supports``
    so the two cannot drift. ``supports`` needs this check separately because
    it consults its cache before validating, and an unhashable argument must
    raise here rather than as a hashability error from building the cache key.
    """
    if not isinstance(component, str):
        raise TypeError(
            f"Component name must be str, got "
            f"{type(component).__name__}"
        )


# Bounded rather than unbounded. Adapter property keys are bounded by the
# scene, but this validator also takes spec- and caller-supplied names, and a
# rejected name caches its ``False`` just as a valid one caches ``True``; a
# long run over randomized object names would otherwise grow a process-lifetime
# cache. Hits concentrate on a handful of hot names, so eviction costs nothing
# in practice, and the bound holds the cache to roughly a megabyte.
@lru_cache(maxsize=16384)
def _component_name_matches(
    component: str,
) -> bool:
    """
    Whether a name matches the component syntax, memoized.

    Only the regex test is cached, not the validation as a whole: caching
    ``validate_component_name`` itself would route a non-hashable argument
    into a ``TypeError`` about hashability instead of the one this module
    raises.

    This is memoized because the resolver's per-observation caches sit
    *behind* validation: ``supports`` and ``value`` re-validate a name on
    every call, so a scene with P properties per entity re-runs the same
    regex O(P) times per binding per step, always with the same answer.
    """
    return _COMPONENT_NAME_RE.fullmatch(component) is not None


def validate_component_name(
    component: str,
) -> str:
    """
    Validate and return an authorable STL component name.

    Component names use lowercase snake-case syntax because the STL parser
    recognizes references such as:

        object_*[hinge_angle]
    """
    _require_component_str(component)

    if not _component_name_matches(component):
        raise ValueError(
            f"Invalid component name {component!r}. Component names must "
            f"match {_COMPONENT_NAME_RE.pattern!r}."
        )

    return component


def discovered_components_for(
    entity: EntityState,
) -> FrozenSet[str]:
    """
    Return every component directly available on an entity.

    The result is exactly the entity's adapter-provided property keys; the
    core defines no built-in component vocabulary of its own.
    """
    property_components = set()

    for property_name in entity.properties:
        validate_component_name(property_name)

        property_components.add(property_name)

    return frozenset(property_components)


class EntityComponentInfo:
    """Immutable component description for one discovered entity."""

    __slots__ = (
        "entity_id",
        "entity_class",
        "components",
        "component_groups",
        "component_tags",
    )

    def __init__(
        self,
        entity_id: int,
        entity_class: str,
        components: Iterable[str],
        component_groups: Optional[Mapping[str, str]] = None,
        component_tags: Optional[
            Mapping[str, Tuple[str, ...]]
        ] = None,
    ) -> None:
        if (
            not isinstance(entity_id, int)
            or isinstance(entity_id, bool)
        ):
            raise TypeError(
                "entity_id must be an integer"
            )

        if (
            not isinstance(entity_class, str)
            or not entity_class
        ):
            raise ValueError(
                "entity_class must be a non-empty string"
            )

        normalized_components = frozenset(
            validate_component_name(component)
            for component in components
        )

        normalized_groups = MappingProxyType(
            dict(component_groups or {})
        )

        normalized_tags = MappingProxyType(
            {
                key: tuple(value)
                for key, value in (component_tags or {}).items()
                if value
            }
        )

        self.entity_id = entity_id
        self.entity_class = entity_class
        self.components = normalized_components
        self.component_groups = normalized_groups
        self.component_tags = normalized_tags

    def supports(
        self,
        component: str,
    ) -> bool:
        """Whether this entity schema exposes a component."""
        return (
            validate_component_name(component)
            in self.components
        )

    def supports_all(
        self,
        components: Iterable[str],
    ) -> bool:
        """Whether this entity schema exposes every requested component."""
        return all(
            self.supports(component)
            for component in components
        )

    def __repr__(self) -> str:
        return (
            "EntityComponentInfo("
            f"entity_id={self.entity_id!r}, "
            f"entity_class={self.entity_class!r}, "
            f"components={sorted(self.components)!r}"
            ")"
        )


class ComponentCatalog:
    """
    Episode-level component catalog produced by adapter discovery.

    The catalog records components per concrete entity. Components for a class
    are computed as a union because heterogeneous entities of the same class
    may expose different properties.
    """

    def __init__(
        self,
        entities: Mapping[int, EntityComponentInfo],
    ) -> None:
        normalized_entities: Dict[
            int,
            EntityComponentInfo,
        ] = {}

        for dictionary_id, info in entities.items():
            if dictionary_id != info.entity_id:
                raise ValueError(
                    f"Catalog key {dictionary_id} does not match "
                    f"EntityComponentInfo.entity_id={info.entity_id}"
                )

            normalized_entities[
                dictionary_id
            ] = info

        class_entity_ids: Dict[
            str,
            set[int],
        ] = {}

        class_components: Dict[
            str,
            set[str],
        ] = {}

        for entity_id, info in normalized_entities.items():
            class_entity_ids.setdefault(
                info.entity_class,
                set(),
            ).add(entity_id)

            class_components.setdefault(
                info.entity_class,
                set(),
            ).update(info.components)

        self._entities = MappingProxyType(
            normalized_entities
        )

        self._class_entity_ids = MappingProxyType(
            {
                class_name: frozenset(entity_ids)
                for class_name, entity_ids
                in class_entity_ids.items()
            }
        )

        self._class_components = MappingProxyType(
            {
                class_name: frozenset(components)
                for class_name, components
                in class_components.items()
            }
        )

    @classmethod
    def from_entities(
        cls,
        entities: (
            Mapping[int, EntityState]
            | Iterable[EntityState]
        ),
        metadata: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
    ) -> "ComponentCatalog":
        """
        Build a catalog from entities discovered by an adapter.

        For simulator adapters, this should be called using the complete
        model-level entity discovery result after reset, not only entities
        currently visible to a perception subsystem.

        ``metadata`` is an optional, best-effort mapping from entity ID to
        that entity's ``adapter.property_metadata(entity_id)`` result (a
        mapping from component key to an object with a ``.group``
        attribute, e.g. ``specola.adapter.PropertyMeta``). A ``.tags``
        attribute (a tuple of structural labels, e.g. subsystem or side)
        is also read best-effort when present. Entities or keys absent
        from ``metadata`` simply carry neither a group label nor tags.
        """
        if isinstance(entities, Mapping):
            iterable = entities.values()
        else:
            iterable = entities

        metadata = metadata or {}

        discovered: Dict[
            int,
            EntityComponentInfo,
        ] = {}

        for entity in iterable:
            if not isinstance(entity, EntityState):
                raise TypeError(
                    f"Expected EntityState, got "
                    f"{type(entity).__name__}"
                )

            entity_metadata = metadata.get(
                entity.entity_id,
                {},
            )

            component_groups = {
                key: meta.group
                for key, meta in entity_metadata.items()
                if key in entity.properties
            }

            component_tags = {
                key: tuple(getattr(meta, "tags", ()) or ())
                for key, meta in entity_metadata.items()
                if key in entity.properties
                and getattr(meta, "tags", ())
            }

            info = EntityComponentInfo(
                entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                components=discovered_components_for(
                    entity
                ),
                component_groups=component_groups,
                component_tags=component_tags,
            )

            previous = discovered.get(
                entity.entity_id
            )

            if previous is not None:
                if (
                    previous.entity_class
                    != info.entity_class
                    or previous.components
                    != info.components
                ):
                    raise ValueError(
                        f"Conflicting discovery records for entity "
                        f"{entity.entity_id}"
                    )

                continue

            discovered[
                entity.entity_id
            ] = info

        return cls(discovered)

    @property
    def entities(
        self,
    ) -> Mapping[int, EntityComponentInfo]:
        """Read-only entity schema mapping."""
        return self._entities

    @property
    def classes(
        self,
    ) -> FrozenSet[str]:
        """All entity classes discovered in the catalog."""
        return frozenset(
            self._class_entity_ids
        )

    def has_entity(
        self,
        entity_id: int,
    ) -> bool:
        """Whether the catalog contains an entity ID."""
        return entity_id in self._entities

    def has_class(
        self,
        entity_class: str,
    ) -> bool:
        """Whether the catalog contains an entity class."""
        return entity_class in self._class_entity_ids

    def entity_info(
        self,
        entity_id: int,
    ) -> EntityComponentInfo:
        """Return the catalog entry for a concrete entity."""
        try:
            return self._entities[
                entity_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"Unknown catalog entity ID {entity_id}"
            ) from exc

    def entity_ids_for_class(
        self,
        entity_class: str,
    ) -> FrozenSet[int]:
        """All discovered entity IDs of one exact class."""
        return self._class_entity_ids.get(
            entity_class,
            frozenset(),
        )

    def components_for_entity(
        self,
        entity_id: int,
    ) -> FrozenSet[str]:
        """Every component exposed by one discovered entity."""
        return self.entity_info(
            entity_id
        ).components

    def component_groups_for_entity(
        self,
        entity_id: int,
    ) -> Mapping[str, str]:
        """
        Display-group label per component for one discovered entity.

        Returns an empty mapping for an unknown entity or for components
        with no adapter-supplied group. A group label is part of the
        authoring vocabulary — spec selectors can filter on it with
        ``@ <group>`` — so it is not presentation-only; absence never
        affects component availability.
        """
        info = self._entities.get(
            entity_id
        )

        if info is None:
            return {}

        return info.component_groups

    def component_tags_for_entity(
        self,
        entity_id: int,
    ) -> Mapping[str, Tuple[str, ...]]:
        """
        Structural tag tuple per component for one discovered entity.

        Returns an empty mapping for an unknown entity or for components
        with no adapter-supplied tags. Like groups, tags are best-effort
        metadata; absence never affects component availability.
        """
        info = self._entities.get(
            entity_id
        )

        if info is None:
            return {}

        return info.component_tags

    def ambiguous_filter_labels(self) -> Tuple[str, ...]:
        """
        Labels usable as ``@ label`` that name both a group and a tag.

        The selector filter matches a key when the label equals its group
        OR is one of its tags, so a name serving as both silently unions
        two different axes -- a group is one label per key and carries the
        key's unit, while tags are structural and may span units. Nothing
        prevents an adapter from colliding the two; this reports it so a
        caller can warn. Empty when the vocabularies are disjoint.
        """
        group_labels = set()
        tag_labels = set()

        for info in self._entities.values():
            group_labels.update(
                group
                for group in info.component_groups.values()
                if group
            )
            tag_labels.update(
                tag
                for key_tags in info.component_tags.values()
                for tag in key_tags
            )

        return tuple(sorted(group_labels & tag_labels))

    def components_for_class(
        self,
        entity_class: str,
    ) -> FrozenSet[str]:
        """
        Union of components exposed by at least one entity of a class.

        This is the appropriate parser-level validation rule for wildcard
        parameters. A wildcard component is valid when at least one entity of
        that class can satisfy it.
        """
        return self._class_components.get(
            entity_class,
            frozenset(),
        )

    def common_components_for_class(
        self,
        entity_class: str,
    ) -> FrozenSet[str]:
        """
        Components exposed by every discovered entity of a class.

        This may be useful for CLI presentation but must not be used as the
        wildcard-binding eligibility rule.
        """
        entity_ids = self.entity_ids_for_class(
            entity_class
        )

        if not entity_ids:
            return frozenset()

        iterator = iter(entity_ids)

        common = set(
            self.components_for_entity(
                next(iterator)
            )
        )

        for entity_id in iterator:
            common.intersection_update(
                self.components_for_entity(
                    entity_id
                )
            )

        return frozenset(common)

    def class_supports_component(
        self,
        entity_class: str,
        component: str,
    ) -> bool:
        """
        Whether at least one discovered entity of a class exposes a component.
        """
        return (
            validate_component_name(component)
            in self.components_for_class(
                entity_class
            )
        )

    def entity_supports_component(
        self,
        entity_id: int,
        component: str,
    ) -> bool:
        """Whether one catalog entity exposes a component."""
        info = self._entities.get(
            entity_id
        )

        if info is None:
            return False

        return info.supports(component)


class ComponentResolver:
    """
    Observation-scoped component resolver with lazy value caching.

    A resolver is created once per MonitorObservation and shared by binding
    eligibility checks and all predicate evaluations for that observation.
    """

    def __init__(
        self,
        entities: Mapping[int, EntityState],
    ) -> None:
        if not isinstance(entities, Mapping):
            raise TypeError(
                "entities must be a mapping from entity IDs to EntityState"
            )

        snapshot: Dict[
            int,
            EntityState,
        ] = {}

        for dictionary_id, entity in entities.items():
            if (
                not isinstance(dictionary_id, int)
                or isinstance(dictionary_id, bool)
            ):
                raise TypeError(
                    f"Entity mapping key must be int, got "
                    f"{type(dictionary_id).__name__}"
                )

            if not isinstance(entity, EntityState):
                raise TypeError(
                    f"Entity {dictionary_id} must be EntityState, got "
                    f"{type(entity).__name__}"
                )

            if dictionary_id != entity.entity_id:
                raise ValueError(
                    f"Entity mapping key {dictionary_id} does not match "
                    f"entity.entity_id={entity.entity_id}"
                )

            snapshot[
                dictionary_id
            ] = entity

        self._entities: Mapping[
            int,
            EntityState,
        ] = MappingProxyType(snapshot)

        self._support_cache: Dict[
            Tuple[int, str],
            bool,
        ] = {}

        self._value_cache: Dict[
            Tuple[int, str],
            float,
        ] = {}

        self._component_set_cache: Dict[
            int,
            FrozenSet[str],
        ] = {}

    @property
    def entities(
        self,
    ) -> Mapping[int, EntityState]:
        """Read-only observation entity mapping."""
        return self._entities

    @property
    def entity_ids(
        self,
    ) -> FrozenSet[int]:
        """Entity IDs present in this observation."""
        return frozenset(
            self._entities
        )

    def entity(
        self,
        entity_id: int,
    ) -> EntityState:
        """Return a present entity or raise a specification-facing error."""
        try:
            return self._entities[
                entity_id
            ]
        except KeyError as exc:
            raise SpecConfigError(
                f"Entity ID {entity_id} is not present in the current "
                f"observation"
            ) from exc

    def components_for_entity(
        self,
        entity_id: int,
    ) -> FrozenSet[str]:
        """All components available on one current entity."""
        cached = self._component_set_cache.get(
            entity_id
        )

        if cached is not None:
            return cached

        components = discovered_components_for(
            self.entity(entity_id)
        )

        self._component_set_cache[
            entity_id
        ] = components

        return components

    def supports(
        self,
        entity_id: int,
        component: str,
    ) -> bool:
        """
        Whether a current entity supports a component.

        An absent entity returns False so that binding enumeration can exclude
        it without raising an exception.

        The cache is consulted before the name is validated. That ordering is
        safe rather than merely faster: a key only enters the cache below,
        after validation has already passed for that name, so a hit implies a
        valid name. A miss still validates before doing anything else. The
        ``isinstance`` guard stays ahead of the lookup because an unhashable
        argument must raise this module's ``TypeError``, not one about
        hashability from the tuple hash.

        This path is hot for a reason worth recording: one observation-scoped
        resolver is shared between binding eligibility and predicate
        evaluation, so every (entity, component) pair is asked about at least
        twice per step -- once while ``slot_candidates`` decides eligibility
        and again while the bound predicate reads values.

        The guard is inlined rather than delegated to
        ``_require_component_str`` because the call overhead is measurable on
        this path (~5% of the sweep). Equality with the shared guard's message
        is pinned by
        ``test_components_vocabulary.py::test_supports_and_validate_reject_non_str_identically``
        instead of by structure.
        """
        if not isinstance(component, str):
            raise TypeError(
                f"Component name must be str, got "
                f"{type(component).__name__}"
            )

        cache_key = (
            entity_id,
            component,
        )

        cached = self._support_cache.get(
            cache_key,
            _CACHE_MISS,
        )

        if cached is not _CACHE_MISS:
            # Narrowing a sentinel of type ``object`` is beyond the checker;
            # every non-sentinel value stored below is a bool.
            return cast(bool, cached)

        component = validate_component_name(
            component
        )

        entity = self._entities.get(
            entity_id
        )

        if entity is None:
            supported = False

        else:
            supported = (
                component in entity.properties
            )

        self._support_cache[
            cache_key
        ] = supported

        return supported

    def supports_all(
        self,
        entity_id: int,
        components: Iterable[str],
    ) -> bool:
        """Whether an entity supports every requested component."""
        return all(
            self.supports(
                entity_id,
                component,
            )
            for component in components
        )

    def missing_components(
        self,
        entity_id: int,
        components: Iterable[str],
    ) -> Tuple[str, ...]:
        """Requested components not available on an entity."""
        return tuple(
            sorted(
                component
                for component in components
                if not self.supports(
                    entity_id,
                    component,
                )
            )
        )

    def value(
        self,
        entity_id: int,
        component: str,
    ) -> float:
        """
        Resolve one scalar component value.

        The value is computed once and then cached for the lifetime of this
        observation-scoped resolver.
        """
        component = validate_component_name(
            component
        )

        cache_key = (
            entity_id,
            component,
        )

        if cache_key in self._value_cache:
            return self._value_cache[
                cache_key
            ]

        entity = self.entity(
            entity_id
        )

        if not self.supports(
            entity_id,
            component,
        ):
            raise SpecConfigError(
                f"Entity {entity_id} ({entity.entity_class}) does not "
                f"support component {component!r}. Available components: "
                f"{sorted(entity.properties)}"
            )

        try:
            value = float(
                entity.properties[
                    component
                ]
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SpecConfigError(
                f"Entity {entity_id} property {component!r} is not "
                f"a real scalar: "
                f"{entity.properties[component]!r}"
            ) from exc

        if not math.isfinite(value):
            raise SpecConfigError(
                f"Component {component!r} for entity {entity_id} "
                f"({entity.entity_class}) is non-finite: {value!r}"
            )

        self._value_cache[
            cache_key
        ] = value

        return value

    def values(
        self,
        entity_id: int,
        components: Iterable[str],
    ) -> Dict[str, float]:
        """Resolve several components for one entity."""
        return {
            component: self.value(
                entity_id,
                component,
            )
            for component in components
        }

    def active_catalog(
        self,
    ) -> ComponentCatalog:
        """
        Build a catalog from the current observation.

        Simulator adapters should generally expose a fuller model-level
        catalog generated during reset. This method is useful for diagnostics
        and open-world perception adapters.
        """
        return ComponentCatalog.from_entities(
            self._entities
        )

    @property
    def cached_value_count(
        self,
    ) -> int:
        """Number of resolved entity-component values currently cached."""
        return len(
            self._value_cache
        )