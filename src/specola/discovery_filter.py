"""Typed discovery queries over immutable records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import Iterable, Tuple

from .discovery_model import DiscoveryEntity, DiscoveryModel


@dataclass(frozen=True)
class DiscoveryQuery:
    classes: Tuple[str, ...] = ()
    entities: Tuple[str, ...] = ()
    signals: Tuple[str, ...] = ()
    groups: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    search: Tuple[str, ...] = ()
    include_raw: bool = True

    @property
    def has_detail_scope(self) -> bool:
        return any((
            self.classes, self.entities, self.signals, self.groups,
            self.roles, self.tags, self.search,
        ))


@dataclass(frozen=True)
class DiscoveryCounts:
    entity_count: int
    property_count: int


@dataclass(frozen=True)
class DiscoveryView:
    model: DiscoveryModel
    query: DiscoveryQuery
    entities: Tuple[DiscoveryEntity, ...]
    before: DiscoveryCounts
    after: DiscoveryCounts


def _validate(patterns: Iterable[str]) -> None:
    for pattern in patterns:
        if not pattern:
            raise ValueError("filter patterns must not be empty")
        index = 0
        while index < len(pattern):
            if pattern[index] == "]":
                raise ValueError(
                    "unmatched bracket in filter pattern {!r}".format(pattern)
                )
            if pattern[index] != "[":
                index += 1
                continue
            content_start = index + 1
            if (
                content_start < len(pattern)
                and pattern[content_start] == "!"
            ):
                content_start += 1
            # fnmatch treats a closing bracket immediately after ``[`` or
            # ``[!`` as a literal class member. The following ``]`` closes
            # the class, so ``[]a]`` is valid and matches ``]`` or ``a``.
            if (
                content_start < len(pattern)
                and pattern[content_start] == "]"
            ):
                content_start += 1
            end = pattern.find("]", content_start)
            if end < 0:
                raise ValueError(
                    "unclosed bracket in filter pattern {!r}".format(pattern)
                )
            content = pattern[index + 1:end]
            if content.startswith("!"):
                content = content[1:]
            if not content or "[" in content:
                raise ValueError(
                    "malformed bracket in filter pattern {!r}".format(pattern)
                )
            index = end + 1


def _matches(value: str, patterns: Tuple[str, ...]) -> bool:
    if not patterns:
        return True
    folded = value.casefold()
    return any(fnmatchcase(folded, pattern.casefold()) for pattern in patterns)


def _entity_search_text(entity: DiscoveryEntity) -> str:
    """Precomputed at model build; fall back for hand-built entities."""
    if entity.search_blob:
        return entity.search_blob
    from .discovery_model import entity_search_blob

    return entity_search_blob(entity)


def _search_matches(text: str, patterns: Tuple[str, ...]) -> bool:
    if not patterns:
        return True
    for pattern in patterns:
        candidate = pattern.casefold()
        if not any(character in candidate for character in "*?["):
            candidate = "*" + candidate + "*"
        if fnmatchcase(text, candidate):
            return True
    return False


def filter_discovery(
    model: DiscoveryModel,
    query: DiscoveryQuery,
) -> DiscoveryView:
    """Apply unions within a filter kind and intersections across kinds."""
    all_patterns = (
        query.classes + query.entities + query.signals + query.groups
        + query.roles + query.tags + query.search
    )
    _validate(all_patterns)
    before = DiscoveryCounts(
        len(model.entities),
        sum(len(entity.properties) for entity in model.entities),
    )

    entities = []
    for entity in model.entities:
        if not query.include_raw and entity.raw:
            continue
        if not _matches(entity.entity_class, query.classes):
            continue
        if query.entities and not (
            any(
                _matches(candidate, query.entities)
                for candidate in (
                    str(entity.entity_id),
                    "{}_{}".format(entity.entity_class, entity.entity_id),
                    entity.label or "",
                    entity.declared_name or "",
                ) + entity.fixture_names + entity.fixture_refs
            )
        ):
            continue
        if not _matches(entity.role or "", query.roles):
            continue
        if not _search_matches(_entity_search_text(entity), query.search):
            continue

        properties = []
        for prop in entity.properties:
            if not _matches(prop.key, query.signals):
                continue
            if not _matches(prop.group or "", query.groups):
                continue
            if query.tags and not any(
                _matches(tag, query.tags) for tag in prop.tags
            ):
                continue
            properties.append(prop)

        property_filter = bool(query.signals or query.groups or query.tags)
        if property_filter and not properties:
            continue
        entities.append(replace(entity, properties=tuple(properties)))

    immutable = tuple(entities)
    after = DiscoveryCounts(
        len(immutable),
        sum(len(entity.properties) for entity in immutable),
    )
    return DiscoveryView(
        model=model,
        query=query,
        entities=immutable,
        before=before,
        after=after,
    )
