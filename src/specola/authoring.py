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
    functions: Optional[Mapping[str, Any]] = None
    input_schemas: Optional[Mapping[str, int]] = None

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
            functions=(model.authored_functions or {}).get("declarations", {}),
            input_schemas=(model.authored_functions or {}).get("input_schemas", {}),
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
        # Scanned first, without building anything: a mapping of names to
        # numbers needs no anchors, and aliases under nested merge keys
        # expand exponentially (700 bytes occupy the loader for minutes).
        for token in yaml.scan(source):
            if isinstance(token, (yaml.AnchorToken, yaml.AliasToken)):
                raise ValueError(
                    "Constants must be a plain YAML mapping such as "
                    "{threshold: 0.25}; anchors and aliases are not "
                    "supported")
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

    for function_name, declaration in (context.functions or {}).items():
        for requirement in declaration["inputs"].values():
            if (context.input_schemas or {}).get(requirement["provider"]) != requirement["schema"]:
                return DraftCheckResult(accepted=False, report="", error=f"function {function_name}: missing recorded input schema {requirement['provider']}")
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


# How long one browser-submitted draft may take to compile, and how much
# memory. Formula text arrives over the dashboard's socket from whoever
# can reach the port; sympy evaluates constant arithmetic exactly, and
# the compiler's own bounds on it (spec/semantic.py) are a list of known
# cases, not a proof. A draft that outruns them holds the interpreter
# lock in one arithmetic call, which would freeze the live run that
# shares this process. A child process can be killed.
DRAFT_CHECK_SECONDS = 30.0
DRAFT_CHECK_BYTES = 4 * 1024 ** 3


DRAFT_WORKER_START_SECONDS = 300.0
# A draft larger than this is refused. It also keeps the request within
# one pipe buffer, so writing it to an idle child cannot block.
DRAFT_MAX_BYTES = 32 * 1024


class DraftChecker:
    """``check_draft`` in a child process that can be killed.

    One child per scene. It rebuilds the scene from the discovery
    document the browser was sent, which is the slow part (17 s for a
    RoboCasa kitchen in a fresh interpreter, measured 2026-11-19), then
    answers drafts until it is closed. A draft that exceeds its time
    limit kills the child; the next draft starts another.
    """

    def __init__(self, discovery_gzip: bytes) -> None:
        self._discovery_gzip = discovery_gzip
        self._lock = threading.Lock()
        self._process = None
        self._replies = None
        self._closed = False

    def _start(self) -> None:
        import queue
        import subprocess
        import sys

        process = subprocess.Popen(
            [sys.executable, "-m", "specola.authoring"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        replies: "queue.Queue[Optional[bytes]]" = queue.Queue()

        def pump() -> None:
            for line in process.stdout:
                if line.startswith(b"{"):
                    replies.put(line)
            replies.put(None)                       # the child has gone

        threading.Thread(
            target=pump, name="specola-draft-worker", daemon=True).start()
        # Owned before the first write, so close() can kill a child that
        # never reads its scene and unblock the write below.
        self._process, self._replies = process, replies
        try:
            process.stdin.write(
                len(self._discovery_gzip).to_bytes(8, "big")
                + self._discovery_gzip)
            process.stdin.flush()
        except OSError:
            self._stop()
            raise
        ready = self._reply(DRAFT_WORKER_START_SECONDS)
        if ready is None or "ready" not in ready:
            self._stop()
            raise OSError("draft worker did not start: {}".format(
                (ready or {}).get("error", "no reply")))

    def _reply(self, timeout: float) -> Optional[dict]:
        import json
        import queue

        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            return None
        return json.loads(line) if line else {"error": "worker exited"}

    def _stop(self) -> None:
        process, self._process, self._replies = self._process, None, None
        if process is not None:
            process.kill()
            process.wait()

    def close(self) -> None:
        """Kill the child for good; later checks raise ``OSError``."""
        self._closed = True
        process = self._process
        if process is not None:
            # Before the lock: a check in flight holds it while it waits,
            # and the kill is what ends that wait.
            process.kill()
        with self._lock:
            self._stop()

    def check(
        self,
        *,
        name: str,
        formula: str,
        constants: Optional[Mapping[str, float]] = None,
        backend: str = "auto",
        symmetric: bool = False,
        timeout: float = DRAFT_CHECK_SECONDS,
    ) -> DraftCheckResult:
        """Check one draft, or reject it for running out of time or memory.

        Raises
        ------
        OSError
            If the child cannot be started. The caller decides whether
            to check in-process instead.
        """
        import json

        draft = json.dumps({
            "name": name, "formula": formula,
            "constants": dict(constants or {}), "backend": backend,
            "symmetric": bool(symmetric),
        }).encode("utf-8") + b"\n"
        if len(draft) > DRAFT_MAX_BYTES:
            return DraftCheckResult(
                accepted=False, report="",
                error="The draft is larger than {} bytes.".format(
                    DRAFT_MAX_BYTES))
        with self._lock:
            if self._closed:
                raise OSError("the draft checker is closed")
            if self._process is None or self._process.poll() is not None:
                self._start()
            try:
                self._process.stdin.write(draft)
                self._process.stdin.flush()
            except OSError:
                self._stop()
                raise
            verdict = self._reply(timeout)
            if verdict is None:
                self._stop()
                return DraftCheckResult(
                    accepted=False, report="",
                    error="Validation stopped after {:g} s. The formula "
                          "takes too long to compile; constant arithmetic "
                          "is exact, so look for a very large power or "
                          "number.".format(timeout),
                )
            if "accepted" not in verdict:
                self._stop()
                return DraftCheckResult(
                    accepted=False, report="",
                    error="Validation could not finish ({}). Constant "
                          "arithmetic is exact, so look for a very large "
                          "power or number.".format(verdict.get("error")),
                )
            return DraftCheckResult(
                accepted=bool(verdict["accepted"]),
                report=str(verdict["report"]), error=verdict["error"],
            )


def _draft_worker_main() -> None:
    """Child side of ``DraftChecker``: one JSON line per draft."""
    import json
    import sys

    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (DRAFT_CHECK_BYTES, DRAFT_CHECK_BYTES))
    except (ImportError, ValueError, OSError):
        pass                    # no such limit on this platform
    stream = sys.stdin.buffer
    out = sys.stdout.buffer
    # Anything the checker prints must not be read as a reply.
    sys.stdout = sys.stderr

    def send(message: dict) -> None:
        out.write(json.dumps(message).encode("utf-8") + b"\n")
        out.flush()

    try:
        size = int.from_bytes(stream.read(8), "big")
        from .dashboard.server import DiscoverySnapshot

        context = CheckContext.from_discovery_model(
            DiscoverySnapshot.from_gzip(stream.read(size)).model)
    except Exception as exc:                        # noqa: BLE001
        send({"error": "{}: {}".format(type(exc).__name__, exc)})
        return
    send({"ready": True})
    import signal

    for line in stream:
        if hasattr(signal, "alarm"):
            # The parent kills a slow check itself. This is for a parent
            # that died meanwhile: SIGALRM's default action ends the
            # process even inside one long arithmetic call.
            signal.alarm(int(DRAFT_CHECK_SECONDS) + 30)
        try:
            result = check_draft(context, **json.loads(line))
            send({"accepted": result.accepted, "report": result.report,
                  "error": result.error})
        except MemoryError:
            send({"error": "out of memory"})
            return
        except Exception as exc:                    # noqa: BLE001
            send({"accepted": False, "report": "",
                  "error": "{}: {}".format(type(exc).__name__, exc)})
        finally:
            if hasattr(signal, "alarm"):
                signal.alarm(0)


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
    functions: Optional[Mapping[str, Any]] = None,
) -> None:
    """Publish a specification document atomically.

    A reader never sees a partial file, and a file that appeared between
    the collision check and the write is refused rather than clobbered:
    ``os.link`` fails when the destination exists, which is what turns
    that race into a ``FileExistsError`` the caller can handle instead
    of silently destroying someone's specifications.
    """
    document = yaml.safe_dump(
        ({"functions": dict(functions), "specs": specs} if functions else {"specs": specs}),
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


if __name__ == "__main__":
    _draft_worker_main()
