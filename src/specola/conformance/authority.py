"""Installed bundled-adapter entry-point authority.

An authority record is a custody record, not a tamper-resistant attestation.
It binds one installed ``specola.adapters`` entry point, its distribution, the
freshly imported module source, the project declaration, and the factory that
the registry would use for that adapter name.  It deliberately makes no claim
about adapter callable syntax or simulator behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
from urllib.parse import unquote, urlparse

try:
    from tomllib import load as _toml_load
except ImportError:  # Python 3.8--3.10
    from tomli import load as _toml_load  # type: ignore[reportMissingImports]

from .. import registry


ENTRY_POINT_GROUP = registry.ENTRY_POINT_GROUP
_DISTRIBUTION_NAME = "specola"
_MISSING = object()


class EntryPointAuthorityError(ValueError):
    """Raised when installed entry-point authority cannot be established."""


@dataclass(frozen=True)
class EntryPointAuthorityRecord:
    """Durable entry-point facts and the fresh registry factory.

    ``factory`` is intentionally omitted from serialization.  Validation
    fresh-resolves it instead of trusting a callable supplied by a caller.
    """

    entry_point_name: str
    entry_point_value: str
    entry_point_module: str
    entry_point_source_path: str
    entry_point_source_sha256: str
    distribution_name: str
    distribution_version: str
    project_source_sha256: str
    repo_root: str
    factory: Any = field(repr=False, compare=False)

    def to_dict(self) -> Dict[str, object]:
        """Return durable facts in a JSON-safe form."""

        return {
            "entry_point_name": self.entry_point_name,
            "entry_point_value": self.entry_point_value,
            "entry_point_module": self.entry_point_module,
            "entry_point_source_path": self.entry_point_source_path,
            "entry_point_source_sha256": self.entry_point_source_sha256,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "project_source_sha256": self.project_source_sha256,
            "repo_root": self.repo_root,
        }

    def to_json(self) -> str:
        """Return canonical JSON for durable facts."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EntryPointAuthorityError("{} must be a non-empty string".format(label))
    return value


def _root(value: Any) -> Path:
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    elif isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise EntryPointAuthorityError("repository root is not a filesystem path")
    try:
        root = Path(value).resolve()
    except (OSError, TypeError, ValueError) as error:
        raise EntryPointAuthorityError("invalid repository root: {}".format(error))
    if not root.is_dir():
        raise EntryPointAuthorityError("repository root is not a directory: {}".format(root))
    return root


def _source(path_value: Any, root: Path, label: str) -> Tuple[str, str]:
    if isinstance(path_value, bytes):
        path_value = os.fsdecode(path_value)
    elif isinstance(path_value, os.PathLike):
        path_value = os.fspath(path_value)
    if not isinstance(path_value, str):
        raise EntryPointAuthorityError("{} is not a filesystem path".format(label))
    try:
        path = Path(path_value).resolve()
        path.relative_to(root)
    except (OSError, TypeError, ValueError) as error:
        raise EntryPointAuthorityError(
            "{} is outside repository root: {} (root {})".format(label, path_value, root)
        ) from error
    if not path.is_file():
        raise EntryPointAuthorityError("{} is not a regular file: {}".format(label, path))
    try:
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EntryPointAuthorityError("cannot read {}: {}".format(label, error))


def _iter_entry_points() -> Tuple[Any, ...]:
    """Return the adapter group through Python 3.8 and modern APIs."""

    entries = metadata.entry_points()
    if hasattr(entries, "select"):
        return tuple(entries.select(group=ENTRY_POINT_GROUP))
    legacy = cast(Any, entries)
    return tuple(legacy.get(ENTRY_POINT_GROUP, ()))


def _entry_point(name: str) -> Any:
    matches = [entry for entry in _iter_entry_points() if getattr(entry, "name", None) == name]
    if len(matches) != 1:
        raise EntryPointAuthorityError(
            "expected exactly one installed entry point {!r}, found {}".format(name, len(matches))
        )
    entry = matches[0]
    if getattr(entry, "group", ENTRY_POINT_GROUP) != ENTRY_POINT_GROUP:
        raise EntryPointAuthorityError("installed entry point group mismatch")
    return entry


def _entry_point_value(value: Any, label: str) -> str:
    """Use importlib.metadata's entry-point value representation."""

    value = _text(value, label)
    try:
        return metadata.EntryPoint("authority", value, ENTRY_POINT_GROUP).value
    except ValueError as error:
        raise EntryPointAuthorityError("invalid {}: {}".format(label, error))


def _project_entry_point_value(root: Path, name: str) -> str:
    """Read one declared adapter entry point from the real project TOML."""

    project_path, _ = _source(root / "pyproject.toml", root, "project declaration")
    try:
        with Path(project_path).open("rb") as project_file:
            document = _toml_load(project_file)
        declared = document["project"]["entry-points"][ENTRY_POINT_GROUP][name]
    except (KeyError, TypeError) as error:
        raise EntryPointAuthorityError(
            "project declaration has no adapter entry point {!r}".format(name)
        ) from error
    except (OSError, ValueError) as error:
        raise EntryPointAuthorityError("cannot parse project declaration: {}".format(error))
    return _entry_point_value(declared, "project entry-point value")


def _distribution(entry: Any, root: Path) -> Tuple[str, str]:
    distribution = getattr(entry, "dist", None)
    if distribution is None:
        # Python 3.8's stdlib importlib.metadata EntryPoint has no
        # ``dist`` attribute. Resolve the known bundled distribution and
        # prove that its own entry-point population contains this exact
        # declaration before accepting its metadata.
        try:
            distribution = metadata.distribution(_DISTRIBUTION_NAME)
        except (metadata.PackageNotFoundError, OSError, ValueError) as error:
            raise EntryPointAuthorityError(
                "entry point distribution metadata is unavailable: {}".format(error)
            )
        owned_matches = [
            candidate
            for candidate in getattr(distribution, "entry_points", ())
            if getattr(candidate, "group", None) == ENTRY_POINT_GROUP
            and getattr(candidate, "name", None) == getattr(entry, "name", None)
            and getattr(candidate, "value", None) == getattr(entry, "value", None)
        ]
        if len(owned_matches) != 1:
            raise EntryPointAuthorityError(
                "entry point is not uniquely owned by the specola distribution"
            )
    metadata_name = getattr(getattr(distribution, "metadata", None), "get", lambda _key: None)("Name")
    name = _text(metadata_name or getattr(distribution, "name", None), "distribution name")
    if name.lower().replace("-", "_") != _DISTRIBUTION_NAME:
        raise EntryPointAuthorityError("entry point distribution mismatch: {}".format(name))
    read_text = getattr(distribution, "read_text", None)
    if callable(read_text):
        direct_url_text = read_text("direct_url.json")
        if direct_url_text is not None:
            if not isinstance(direct_url_text, (str, bytes, bytearray)):
                raise EntryPointAuthorityError("distribution direct_url.json is not text")
            try:
                direct_url = json.loads(direct_url_text)
                editable = direct_url.get("dir_info", {}).get("editable")
            except (AttributeError, TypeError, ValueError) as error:
                raise EntryPointAuthorityError("invalid distribution direct_url.json: {}".format(error))
            if editable:
                parsed = urlparse(_text(direct_url.get("url"), "editable distribution URL"))
                if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
                    raise EntryPointAuthorityError("editable distribution URL is not a local file URL")
                try:
                    editable_root = Path(unquote(parsed.path)).resolve()
                except (OSError, TypeError, ValueError) as error:
                    raise EntryPointAuthorityError(
                        "invalid editable distribution root: {}".format(error)
                    )
                if editable_root != root:
                    raise EntryPointAuthorityError("editable distribution root mismatch")
    return name, _text(getattr(distribution, "version", None), "distribution version")


def _module_name(value: str) -> str:
    return _text(value.split(":", 1)[0].strip(), "entry-point module")


def _fresh_registry_entry(entry_point: Any, name: str, module_name: str, root: Path) -> Tuple[Any, str, str]:
    """Import one target module afresh and retain its registration only.

    This narrowly scoped transaction prevents a cached registration from being
    used by the raw runner.  It does not inspect the callable it obtains.
    """

    registry_snapshot = dict(registry._REGISTRY)
    previous_module = sys.modules.pop(module_name, _MISSING)
    registry._REGISTRY.pop(name, None)
    try:
        entry_point.load()
        module = sys.modules.get(module_name)
        if module is None:
            raise EntryPointAuthorityError("entry point did not import its module")
        source_path, source_sha256 = _source(
            getattr(module, "__file__", None), root, "entry-point module"
        )
        registered = registry.get(name)
        if registered is None or registered.name != name:
            raise EntryPointAuthorityError("entry point module did not register its name")
    except Exception:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(registry_snapshot)
        sys.modules.pop(module_name, None)
        if previous_module is not _MISSING:
            sys.modules[module_name] = cast(Any, previous_module)
        raise

    registry._REGISTRY.clear()
    registry._REGISTRY.update(registry_snapshot)
    registry._REGISTRY[name] = registered
    return registered, source_path, source_sha256


def resolve_entry_point_authority(
    name: str,
    repository_root: Optional[str] = None,
) -> EntryPointAuthorityRecord:
    """Resolve one bundled adapter's installed entry point and fresh factory."""

    name = _text(name, "entry-point name")
    root = _root(repository_root or str(Path(__file__).parents[3]))
    _, project_sha256 = _source(root / "pyproject.toml", root, "project declaration")
    entry_point = _entry_point(name)
    value = _entry_point_value(getattr(entry_point, "value", None), "entry-point value")
    declared_value = _project_entry_point_value(root, name)
    if value != declared_value:
        raise EntryPointAuthorityError("installed entry-point value differs from project declaration")
    distribution_name, distribution_version = _distribution(entry_point, root)
    registered, source_path, source_sha256 = _fresh_registry_entry(
        entry_point, name, _module_name(value), root
    )
    return EntryPointAuthorityRecord(
        entry_point_name=name,
        entry_point_value=value,
        entry_point_module=_module_name(value),
        entry_point_source_path=source_path,
        entry_point_source_sha256=source_sha256,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        project_source_sha256=project_sha256,
        repo_root=str(root),
        factory=registered.factory,
    )


def validate_entry_point_authority(
    record: EntryPointAuthorityRecord,
    repository_root: Optional[str] = None,
) -> EntryPointAuthorityRecord:
    """Fresh-resolve durable facts and return the fresh registry factory."""

    if not isinstance(record, EntryPointAuthorityRecord):
        raise EntryPointAuthorityError("authority record has wrong type")
    root = _root(repository_root or record.repo_root)
    if record.repo_root != str(root):
        raise EntryPointAuthorityError("authority record repository root mismatch")
    current = resolve_entry_point_authority(record.entry_point_name, str(root))
    for key, value in record.to_dict().items():
        if value != current.to_dict()[key]:
            raise EntryPointAuthorityError("authority record field drift: {}".format(key))
    return current


__all__ = [
    "ENTRY_POINT_GROUP",
    "EntryPointAuthorityError",
    "EntryPointAuthorityRecord",
    "resolve_entry_point_authority",
    "validate_entry_point_authority",
]
