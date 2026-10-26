"""Immutable, versioned query authoring without executable configuration.

The document defines the configured extraction subject, independently of
specifications and extraction scope. Native binding schemas and operand
resolvers validate its argument meanings before any query executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft202012Validator
from referencing import Registry
import yaml


class QueryConfigurationError(ValueError):
    """Query configuration is malformed, ambiguous or unsupported."""


def _json_tree(value: Any, active: Optional[set] = None) -> None:
    """Require an acyclic tree of exact JSON types before serialization."""
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise QueryConfigurationError("query arguments must be finite JSON values")
        return
    if type(value) not in (dict, list):
        raise QueryConfigurationError("query configuration contains a non-JSON value")
    active = set() if active is None else active
    if id(value) in active:
        raise QueryConfigurationError("query configuration contains a cycle")
    active.add(id(value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise QueryConfigurationError("query configuration keys must be strings")
        children = value.values()
    else:
        children = value
    for child in children:
        _json_tree(child, active)
    active.remove(id(value))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueryConfigurationError("duplicate query configuration key: " + key)
        result[key] = value
    return result


class _QueryLoader(yaml.SafeLoader):
    """Safe YAML with duplicate and non-string keys rejected."""


def _yaml_mapping(loader, node):
    pairs = []
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
            raise QueryConfigurationError("query configuration keys must be strings without YAML merge directives")
        pairs.append((loader.construct_object(key_node, deep=True), loader.construct_object(value_node, deep=True)))
    return _unique_pairs(pairs)


_QueryLoader.add_constructor("tag:yaml.org,2002:map", _yaml_mapping)


def _schema_document() -> dict:
    # read_text supports the project's Python 3.8 floor and wheel resources.
    with resources.open_text("specola.data", "query-configuration-v1.schema.json", encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = _schema_document()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=Registry())


def _validate_document(document: dict) -> None:
    _json_tree(document)
    errors = list(_validator().iter_errors(document))
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "<document>"
        raise QueryConfigurationError(path + ": " + error.message)
    populations, tokens = set(), set()
    for population in document["populations"]:
        binding = population["binding"]
        if not binding.strip() or binding != binding.strip() or binding in populations:
            raise QueryConfigurationError("empty or duplicate query population binding: " + binding)
        populations.add(binding)
    for instance in document["instances"]:
        if not instance["binding"].strip() or instance["binding"] != instance["binding"].strip():
            raise QueryConfigurationError("query instance binding must be nonempty")
        token_pattern = _validator().schema["properties"]["instances"]["items"]["properties"]["id"]["pattern"]
        if re.fullmatch(token_pattern, instance["id"]) is None:
            raise QueryConfigurationError("query instance token has invalid characters")
        if instance["id"] in tokens:
            raise QueryConfigurationError("duplicate query instance token: " + instance["id"])
        tokens.add(instance["id"])
        for operand in instance["operands"].values():
            if "id" in operand and type(operand["id"]) is not int:
                raise QueryConfigurationError("native operand ID must be an integer, not an integral float")
            if not operand["kind"].strip() or ("name" in operand and not operand["name"].strip()):
                raise QueryConfigurationError("native operand kind and name must be nonempty")


def _canonical(document: dict) -> str:
    _validate_document(document)
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True)
class QueryConfiguration:
    """A validated detached document with content-based configuration identity.

    Store canonical JSON rather than retaining mutable caller dictionaries.
    Accessors return fresh values, so a caller cannot change an installed
    configuration or the meaning of an existing monitored signal in place.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not str:
            raise QueryConfigurationError("canonical query configuration must be text")
        try:
            document = json.loads(self.canonical_json, object_pairs_hook=_unique_pairs)
            canonical = _canonical(document)
        except (json.JSONDecodeError, RecursionError) as error:
            raise QueryConfigurationError("invalid query configuration: " + str(error)) from error
        if canonical != self.canonical_json:
            raise QueryConfigurationError("query configuration is not canonical JSON")

    @classmethod
    def from_document(cls, document: dict) -> QueryConfiguration:
        try:
            return cls(_canonical(document))
        except RecursionError as error:
            raise QueryConfigurationError("query configuration nesting exceeds parser capacity") from error

    @classmethod
    def empty(cls) -> QueryConfiguration:
        return cls.from_document({"schema": "specola.queries.v1", "populations": [], "instances": []})

    def document(self) -> Dict[str, Any]:
        return json.loads(self.canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def population_arguments(self, binding: str) -> Optional[Dict[str, Any]]:
        for population in self.document()["populations"]:
            if population["binding"] == binding:
                return population["arguments"]
        return None

    def instances(self, binding: Optional[str] = None) -> Tuple[Dict[str, Any], ...]:
        return tuple(instance for instance in self.document()["instances"]
                     if binding is None or instance["binding"] == binding)


def load_query_configuration(path: Path) -> QueryConfiguration:
    """Read JSON or safe YAML, rejecting duplicate keys in either syntax."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        document = (json.loads(text, object_pairs_hook=_unique_pairs)
                    if Path(path).suffix.lower() == ".json" else yaml.load(text, Loader=_QueryLoader))
        return QueryConfiguration.from_document(document)
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError, RecursionError) as error:
        raise QueryConfigurationError("cannot load query configuration " + str(path) + ": " + str(error)) from error
