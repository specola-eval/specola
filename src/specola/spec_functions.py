"""Explicitly loaded local functions over retained native observation inputs.

Local Python is trusted code. Preparation declares scalar outputs without
running numerical queries; evaluation feeds the existing scalar monitor path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import math
import sys
from typing import Any

import numpy as np

from .adapter import MonitorObservation, PropertyMeta
from .capabilities import AdapterResetResult
from .components import ComponentCatalog, EntityComponentInfo
from .entities import EntityState, TrackState


def immutable(value):
    """Detach producer data; array storage cannot be made writable again."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(item) for item in value)
    return value


@dataclass(frozen=True)
class FunctionInputs:
    revision: Any
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionTarget:
    """One scalar-output owner, either native or created by the function."""
    key: str
    entity_id: int | None = None
    label: str = ""
    tags: tuple[str, ...] = ()
    position: Any = None


@dataclass(frozen=True)
class FunctionOutput:
    """Scalar results with an optional current position for a derived entity."""
    properties: Mapping[str, float]
    position: Any = None


@dataclass(frozen=True)
class FunctionDefinition:
    name: str
    declaration: dict
    implementation: Any
    digest: str

    @property
    def outputs(self):
        return self.declaration["outputs"]

    @property
    def entity_class(self):
        return self.declaration["entity_class"]


def load_functions(declarations, spec_path):
    """Load only an explicit local file reference, relative to its YAML file."""
    if declarations is None:
        return ()
    if not isinstance(declarations, dict):
        raise ValueError("functions must be a mapping")
    definitions = []
    modules = {}
    for name, raw in declarations.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("each function needs a name and declaration mapping")
        if set(raw) - {"python", "entity_class", "inputs", "parameters", "outputs", "sha256"}:
            raise ValueError(f"function {name}: unknown declaration keys")
        declaration = json.loads(json.dumps(raw))
        reference = declaration.get("python", "")
        if not isinstance(reference, str) or ":" not in reference or "://" in reference:
            raise ValueError(f"function {name}: python must be a local file.py:symbol")
        filename, symbol = reference.rsplit(":", 1)
        path = (Path(spec_path).resolve().parent / filename).resolve()
        if path.suffix != ".py" or not symbol.isidentifier():
            raise ValueError(f"function {name}: python must be a local file.py:symbol")
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"function {name}: cannot read {path}: {exc}") from exc
        digest = hashlib.sha256(source).hexdigest()
        if declaration.get("sha256", digest) != digest:
            raise ValueError(f"function {name}: source digest changed for {path}")
        declaration["python"] = f"{path}:{symbol}"
        declaration["sha256"] = digest
        if not isinstance(declaration.get("entity_class"), str) or not declaration["entity_class"]:
            raise ValueError(f"function {name}: entity_class is required")
        for field_name in ("inputs", "outputs"):
            if not isinstance(declaration.get(field_name), dict) or not declaration[field_name]:
                raise ValueError(f"function {name}: {field_name} must be a nonempty mapping")
        for key, requirement in declaration["inputs"].items():
            if not isinstance(requirement, dict) or set(requirement) != {"provider", "schema"}:
                raise ValueError(f"function {name}: input {key} needs provider and schema")
            if not isinstance(requirement["provider"], str) or type(requirement["schema"]) is not int:
                raise ValueError(f"function {name}: invalid input schema for {key}")
        for key, metadata in declaration["outputs"].items():
            if not isinstance(key, str) or not key.isidentifier() or not isinstance(metadata, dict):
                raise ValueError(f"function {name}: invalid output declaration")
            if set(metadata) - {"unit", "group", "note"}:
                raise ValueError(f"function {name}: unknown output metadata for {key}")
        parameters = declaration.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"function {name}: parameters must be a mapping")
        if path not in modules:
            module_name = f"_specola_authored_{digest}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError(f"function {name}: cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Compile the bytes just hashed, avoiding a stale bytecode cache.
            exec(compile(source, str(path), "exec"), module.__dict__)
            modules[path] = module
        implementation = getattr(modules[path], symbol, None)
        if not callable(implementation) or not callable(getattr(implementation, "prepare", None)):
            raise ValueError(f"function {name}: {symbol} needs callable prepare and evaluation")
        validate = getattr(implementation, "validate", None)
        if validate is not None:
            validate(immutable(parameters))
        definitions.append(FunctionDefinition(name, declaration, implementation, digest))
    return tuple(definitions)


def load_authoring_functions(path):
    """Load the function block explicitly requested by a local authoring CLI."""
    import yaml
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict) or "functions" not in document:
        raise ValueError(f"{path}: expected a functions mapping")
    return FunctionSet(load_functions(document["functions"], path))


def prepare_authoring_functions(adapter, functions):
    """Prepare unread output declarations using one catalog-only reset."""
    from .capabilities import reset_adapter
    adapter.set_function_inputs(functions.requirements)
    adapter.set_extraction_scope(None)
    adapter.set_extraction_entity_scope({})
    try:
        reset = reset_adapter(adapter)
    finally:
        adapter.set_extraction_entity_scope(None)
    prepared = functions.prepare(adapter, reset)
    adapter._specification_provenance = {
        "spec_functions": json.dumps(functions.declarations, sort_keys=True),
        "spec_functions_active": json.dumps([definition.name for definition in functions.definitions]),
    }
    return prepared


class FunctionSet:
    """Episode caches and output overlay for a fixed authored function set."""

    def __init__(self, definitions=(), specs=None):
        self._loaded_definitions = tuple(definitions)
        if specs is None:
            self.definitions = self._loaded_definitions
        else:
            from .spec.parser import parse
            from .spec.semantic import _collect_comparisons
            from .spec.selectors import _resolve_term
            references = [reference for spec in specs
                          for comparison in _collect_comparisons(getattr(spec, "node", None) or parse(spec.formula_text))
                          for reference in comparison.references]

            def requested(definition):
                keys = frozenset(definition.outputs)
                groups = {key: info.get("group", "authored function") for key, info in definition.outputs.items()}
                for reference in references:
                    if not (reference.head == definition.entity_class or reference.head.startswith(definition.entity_class + "_")):
                        continue
                    selector = reference.selector
                    # Target tags require native declarations. Retain possible
                    # tag matches until preparation and compiled scope resolve them.
                    included = _resolve_term(replace(selector.include, group=None), keys, groups, {})
                    for exclusion in selector.exclude:
                        included -= _resolve_term(exclusion, keys, groups, {})
                    if included:
                        return True
                return False

            self.definitions = tuple(definition for definition in self._loaded_definitions if requested(definition))
        self._plans = {}
        self._caches = {}
        self._revision = None
        self._scope = None
        self._entity_scope = None
        self.labels = {}

    @property
    def declarations(self):
        return {definition.name: definition.declaration for definition in self._loaded_definitions}

    @property
    def requirements(self):
        requirements = {}
        for definition in self.definitions:
            for requirement in definition.declaration["inputs"].values():
                provider, schema = requirement["provider"], requirement["schema"]
                if provider in requirements and requirements[provider] != schema:
                    raise ValueError(f"conflicting input schemas for {provider}")
                requirements[provider] = schema
        return requirements

    def close(self):
        """Release owned native references while the simulator is still alive."""
        self._caches.clear()
        self._revision = None

    def _inputs(self, definition, snapshot):
        return MappingProxyType({
            key: snapshot.values.get(requirement["provider"])
            for key, requirement in definition.declaration["inputs"].items()
        })

    def prepare(self, adapter, reset):
        if not self._loaded_definitions:
            adapter._authored_function_context = None
            return reset
        self.close()
        snapshot = reset.initial_observation.function_inputs
        if self.definitions and snapshot is None:
            raise ValueError("adapter did not capture declared function inputs at reset")
        catalog = dict(reset.discovered_catalog.entities)
        adapter.prepare_function_entity_ids(catalog)
        self._authored_owners = {}
        metadata = {key: dict(value) for key, value in reset.property_metadata.items()}
        declared_components = dict(reset.capabilities.components_by_class)
        groups = dict(reset.capabilities.component_groups_by_class)
        tags = dict(reset.capabilities.component_tags_by_class)
        self._plans = {}
        self.labels = {}
        active_names = {definition.name for definition in self.definitions}
        for definition in self._loaded_definitions:
            if definition.name in active_names:
                continue
            cls = definition.entity_class
            collisions = set(definition.outputs) & (
                set(reset.discovered_catalog.components_for_class(cls)) | set(declared_components.get(cls, ()))
            )
            if collisions:
                raise ValueError(f"function {definition.name}: output collision for class {cls}: {sorted(collisions)}")
            # Static schema publication needs no producer inputs or target
            # inference. Unknown dynamic entities remain absent from the catalog.
            for entity_id, info in tuple(catalog.items()):
                if info.entity_class != cls:
                    continue
                entity_meta = metadata.setdefault(entity_id, {})
                for key, output in definition.outputs.items():
                    entity_meta[key] = PropertyMeta(
                        unit=output.get("unit", ""), group=output.get("group", "authored function"),
                        note=f"declared function {definition.name}; value unread",
                    )
                catalog[entity_id] = EntityComponentInfo(
                    entity_id, cls, set(info.components) | set(definition.outputs),
                    component_groups={key: value.group for key, value in entity_meta.items()},
                    component_tags={key: value.tags for key, value in entity_meta.items()},
                )
            declared_components[cls] = frozenset(set(declared_components.get(cls, ())) | set(definition.outputs))
            groups[cls] = dict(groups.get(cls, {}), **{key: info.get("group", "authored function") for key, info in definition.outputs.items()})
        for definition in self.definitions:
            collisions = set(definition.outputs) & (set(reset.discovered_catalog.components_for_class(definition.entity_class)) | set(declared_components.get(definition.entity_class, ())))
            if collisions:
                raise ValueError(f"function {definition.name}: output collision for class {definition.entity_class}: {sorted(collisions)}")
            parameters = immutable(definition.declaration["parameters"])
            targets = definition.implementation.prepare(inputs=self._inputs(definition, snapshot), parameters=parameters)
            plan = {}
            for target in targets:
                if not isinstance(target, FunctionTarget) or target.key in plan:
                    raise ValueError(f"function {definition.name}: duplicate or invalid target")
                entity_id = target.entity_id
                if entity_id is None:
                    entity_id = adapter.function_entity_id(target.key, definition.entity_class)
                    if entity_id in reset.discovered_catalog.entities:
                        raise ValueError(f"function {definition.name}: authored entity ID {entity_id} collides with a native owner")
                    owner = (target.key, definition.entity_class)
                    if entity_id in self._authored_owners and self._authored_owners[entity_id] != owner:
                        raise ValueError(f"function {definition.name}: authored entity ID {entity_id} collides with another authored owner")
                    self._authored_owners[entity_id] = owner
                elif entity_id not in catalog or catalog[entity_id].entity_class != definition.entity_class:
                    raise ValueError(f"function {definition.name}: target {entity_id} is not a native {definition.entity_class}")
                target = replace(target, entity_id=entity_id)
                old = catalog.get(entity_id)
                old_components = set(old.components) if old else set()
                collision = old_components & set(definition.outputs)
                if collision:
                    raise ValueError(f"function {definition.name}: output collision on {entity_id}: {sorted(collision)}")
                entity_meta = metadata.setdefault(entity_id, {})
                for key, info in definition.outputs.items():
                    entity_meta[key] = PropertyMeta(
                        unit=info.get("unit", ""), group=info.get("group", "authored function"),
                        note=f"authored function {definition.name}; {info.get('note', '')}", tags=target.tags,
                    )
                catalog[entity_id] = EntityComponentInfo(
                    entity_id, definition.entity_class, old_components | set(definition.outputs),
                    component_groups={key: item.group for key, item in entity_meta.items()},
                    component_tags={key: item.tags for key, item in entity_meta.items()},
                )
                self.labels[entity_id] = target.label
                plan[target.key] = target
            self._plans[definition.name] = plan
            cls = definition.entity_class
            declared_components[cls] = frozenset(set(declared_components.get(cls, ())) | set(definition.outputs))
            groups[cls] = dict(groups.get(cls, {}), **{key: info.get("group", "authored function") for key, info in definition.outputs.items()})
            tags[cls] = frozenset(set(tags.get(cls, ())) | {tag for target in plan.values() for tag in target.tags})
        capabilities = replace(
            reset.capabilities,
            supported_classes=frozenset(set(reset.capabilities.supported_classes) | {d.entity_class for d in self._loaded_definitions}),
            components_by_class=declared_components, component_groups_by_class=groups, component_tags_by_class=tags,
        )
        self._revision = None if snapshot is None else snapshot.revision
        adapter._authored_function_context = {"declarations": self.declarations, "active_functions": [definition.name for definition in self.definitions], "input_schemas": dict(adapter.function_input_schemas), "labels": self.labels}
        self._metadata = metadata
        return replace(reset, discovered_catalog=ComponentCatalog(catalog), property_metadata=metadata, capabilities=capabilities)

    def set_scope(self, scope, entity_scope):
        self._scope = scope
        self._entity_scope = entity_scope

    def _requested(self, definition, target):
        if self._scope is None:
            return True
        keys = self._scope.get(definition.entity_class, ())
        if not set(keys) & set(definition.outputs):
            return False
        selectors = (self._entity_scope or {}).get(definition.entity_class, ())
        return any(
            (pin is None or pin == target.entity_id)
            and all(
                set(labels) <= (
                    set(self._metadata[target.entity_id].get(key, PropertyMeta(group="")).tags)
                    | {self._metadata[target.entity_id].get(key, PropertyMeta(group="")).group}
                )
                for key, labels in filters
            )
            for pin, filters in selectors
        )

    @property
    def prepared(self) -> bool:
        """True once ``prepare`` ran for the current episode's adapter."""
        return all(definition.name in self._plans for definition in self.definitions)

    def overlay(self, observation):
        if not self.definitions:
            return observation
        snapshot = observation.function_inputs
        if snapshot is None:
            raise ValueError("observation is missing declared function inputs")
        if snapshot.revision != self._revision:
            self.close()
            self._revision = snapshot.revision
        entities = dict(observation.perceived_entities)
        tags = {key: dict(value) for key, value in observation.component_tags.items()}
        for definition in self.definitions:
            targets = {key: target for key, target in self._plans[definition.name].items() if self._requested(definition, target)}
            if not targets:
                continue
            inputs = self._inputs(definition, snapshot)
            if any(value is None for value in inputs.values()):
                continue
            try:
                for target in targets.values():
                    observed_owner = observation.perceived_entities.get(target.entity_id)
                    if observed_owner is not None and (
                        target.entity_id in self._authored_owners
                        or observed_owner.entity_class != definition.entity_class
                    ):
                        raise ValueError(f"entity ID {target.entity_id} collides with a native owner")
                result = definition.implementation(
                    inputs=inputs, parameters=immutable(definition.declaration["parameters"]),
                    cache=self._caches.setdefault(definition.name, {}), targets=MappingProxyType(targets),
                )
                if not isinstance(result, Mapping) or set(result) - set(targets):
                    raise ValueError("function returned undeclared targets")
                for key, values in result.items():
                    if values is None:
                        continue
                    position = values.position if isinstance(values, FunctionOutput) else None
                    values = values.properties if isinstance(values, FunctionOutput) else values
                    if position is not None:
                        position = np.asarray(position, dtype=float)
                        if position.shape != (3,) or not np.isfinite(position).all():
                            raise ValueError("function entity position must be a finite 3-vector")
                    if not isinstance(values, Mapping) or set(values) != set(definition.outputs):
                        raise ValueError("function must return exactly its declared outputs, or None")
                    if any(not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) for value in values.values()):
                        raise ValueError("function output must be a finite scalar")
                    target = targets[key]
                    entity_id = target.entity_id
                    old = entities.get(entity_id)
                    props = dict(old.properties) if old is not None else {}
                    if set(props) & set(values):
                        raise ValueError("function output collides with observed native property")
                    props.update({key: float(value) for key, value in values.items()})
                    entities[entity_id] = replace(old, properties=props) if old is not None else EntityState(
                        entity_id=entity_id, entity_class=definition.entity_class,
                        position=position if position is not None else target.position, velocity=None, track_age=observation.timestep,
                        track_state=TrackState.CONFIRMED, properties=props,
                    )
                    tags.setdefault(entity_id, {}).update({key: target.tags for key in values})
            except Exception as exc:
                raise ValueError(f"function {definition.name} failed at observation {observation.timestep}: {exc}") from exc
        return replace(observation, perceived_entities=entities, component_tags=tags)
