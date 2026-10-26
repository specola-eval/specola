"""One checker and one writer, shared by every authoring surface.

``write-specs`` has always validated a draft before letting it be kept,
and saved through a tmp+link/replace path that cannot half-publish a
file. The browser needs both behaviours. Two implementations of "is this
specification valid" would be two answers to the same question, and two
save paths would be two chances to lose someone's work, so this module
holds the single copy and the terminal writer defers to it.

A check runs against a ``CheckContext``: the catalogue, metadata,
entities, labels and capabilities of one scene. Where those come from is
the caller's business — a live adapter mid-run, or a recording's
persisted discovery document with no simulator present. That is what
makes an offline check meaningful rather than an approximation: it is
the same function over the same inputs.
"""

from __future__ import annotations

import contextlib
import io
import os
import secrets
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import yaml

# Serializes the report capture below. ``check_specs`` reports by
# printing, and the capture is process-global, so two concurrent checks
# would interleave their reports into each other's buffers -- and in a
# live run would swallow simulator output that belongs in the terminal.
# Checks are not on a hot path; serializing them costs nothing that
# matters and removes a class of bug that is very hard to see.
_CHECK_LOCK = threading.Lock()

AUTHORED_KEYS = ("name", "formula", "constants", "backend", "symmetric")


@dataclass(frozen=True)
class CheckContext:
    """Everything a draft is compiled against, and nothing live.

    Immutable by construction and by intent: a check must not be able to
    disturb a running episode, and a context built from a recording has
    no adapter to disturb in the first place.
    """

    dt: float
    catalog: Any
    metadata: Mapping[int, Mapping[str, Any]]
    entities: Mapping[int, Any]
    labels: Optional[Mapping[int, str]]
    capabilities: Any

    @classmethod
    def from_discovery_model(cls, model) -> "CheckContext":
        """Build a context from a persisted or restored scene."""
        from .discovery_restore import (
            catalog_from_model,
            entity_states_from_model,
            labels_from_model,
            property_metadata_from_model,
        )

        if model.capabilities is None:
            from .discovery_restore import DiscoveryRestoreError

            raise DiscoveryRestoreError(
                "This scene carries no adapter capabilities, so a check "
                "against it cannot reproduce the live one."
            )
        return cls(
            dt=model.effective_dt,
            catalog=catalog_from_model(model),
            metadata=property_metadata_from_model(model),
            entities=entity_states_from_model(model),
            labels=labels_from_model(model),
            capabilities=model.capabilities,
        )


@dataclass(frozen=True)
class DraftCheckResult:
    accepted: bool
    report: str
    error: Optional[str] = None


def normalise_name(name: str) -> str:
    """The writer's saved-name normalization, in one place."""
    return name.lower().replace(" ", "_")


def parse_constants(source: str) -> Dict[str, float]:
    """Parse the constants field and reuse the compiler's validation."""
    if not source.strip():
        return {}
    try:
        value = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid constants YAML: {}".format(exc)) from exc
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "Constants must be a YAML mapping such as {threshold: 0.25}"
        )

    from .spec.semantic import validate_constants

    return validate_constants(value)


def check_draft(
    context: CheckContext,
    *,
    name: str,
    formula: str,
    constants: Optional[Mapping[str, float]] = None,
    backend: str = "auto",
    symmetric: bool = False,
) -> DraftCheckResult:
    """Compile one draft against a scene and report what happened.

    The report is the checker's own text — resolved keys, units, groups,
    tags, binding counts, mixed-unit warnings — because that is what
    tells an author *why* a formula binds what it binds, and a shorter
    summary would drop the part people actually read.
    """
    from .check_specs import _check_entry

    entry = SimpleNamespace(
        name=name,
        formula_text=formula,
        constants=dict(constants or {}),
        symmetric=bool(symmetric),
        backend=backend,
    )
    buffer = io.StringIO()
    with _CHECK_LOCK:
        with contextlib.redirect_stdout(buffer):
            try:
                accepted = _check_entry(
                    entry,
                    context.dt,
                    context.catalog,
                    context.metadata,
                    context.entities,
                    context.labels,
                    capabilities=context.capabilities,
                )
                error = None
            except Exception as exc:            # noqa: BLE001
                # A checker crash is a rejected draft, not a dead
                # server. The message travels so the author sees
                # something better than silence.
                accepted = False
                error = "{}: {}".format(type(exc).__name__, exc)
    return DraftCheckResult(
        accepted=bool(accepted), report=buffer.getvalue(), error=error
    )


def spec_entry(
    *,
    name: str,
    formula: str,
    constants: Optional[Mapping[str, float]] = None,
    backend: Optional[str] = None,
    symmetric: bool = False,
) -> Dict[str, Any]:
    """One authored entry, carrying only keys the schema accepts.

    Unknown keys load without complaint, so an entry carrying a
    plausible-looking extra — ``window:`` is the one this UI's design
    invented — would read as configured and do nothing.
    """
    entry: Dict[str, Any] = {"name": name, "formula": formula}
    if constants:
        entry["constants"] = dict(constants)
    if backend and backend != "auto":
        entry["backend"] = backend
    if symmetric:
        entry["symmetric"] = True
    return entry


def write_specs_document(
    specs: List[Dict[str, object]],
    path: str,
    *,
    overwrite: bool = True,
) -> None:
    """Publish a specification document atomically.

    A reader never sees a partial file, and a file that appeared between
    the collision check and the write is refused rather than clobbered:
    ``os.link`` fails when the destination exists, which is what turns
    that race into a ``FileExistsError`` the caller can handle instead
    of silently destroying someone's specifications.
    """
    document = yaml.safe_dump(
        {"specs": specs},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    destination = Path(path)
    temporary_path: Optional[str] = None
    descriptor: Optional[int] = None
    try:
        while descriptor is None:
            temporary_path = str(destination.with_name(
                ".specola-{}.tmp".format(secrets.token_hex(8))
            ))
            try:
                descriptor = os.open(
                    temporary_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
            except FileExistsError:
                continue
        assert temporary_path is not None
        if overwrite and destination.exists():
            shutil.copystat(path, temporary_path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def available_renamed_path(path: str) -> str:
    """The first numbered sibling that does not exist."""
    source = Path(path)
    suffix = source.suffix
    stem = source.stem
    index = 2
    while True:
        candidate = source.with_name(
            "{}_{}{}".format(stem, index, suffix)
        )
        if not candidate.exists():
            return str(candidate)
        index += 1


def yaml_path(path: str) -> str:
    """Give a path a specification-file extension if it lacks one."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        return path
    return path + ".yaml"
