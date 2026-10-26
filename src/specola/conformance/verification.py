"""Lexical execution context for collecting candidate extraction evidence.

The context permits unresolved classifications to remain in a draft while
resolved quantities undergo the ordinary producer, argument and value checks.
It grants no readiness and cannot cross threads, tasks or ordinary output
boundaries. The existing conformance runner owns candidate execution.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import current_thread
from typing import Optional

from .core_runner import EvidenceIdentity
from .provider import ExtractionGuardError
from .surface import ProducerIdentity, _evidence, _text
from ..query_config import QueryConfiguration


@dataclass(frozen=True)
class VerificationSubject:
    """Declared class, producer and configuration for one candidate row."""

    adapter_class: type
    producer: ProducerIdentity
    configuration: EvidenceIdentity
    queries: QueryConfiguration

    def __post_init__(self):
        from ..adapter import SimulatorAdapter

        if not isinstance(self.adapter_class, type) or not issubclass(self.adapter_class, SimulatorAdapter):
            raise TypeError("candidate subject needs an adapter class")
        if type(self.producer) is not ProducerIdentity:
            raise TypeError("candidate subject needs a producer identity")
        for name in ("name", "version", "backend"):
            _text(getattr(self.producer, name), "candidate producer " + name)
        _evidence(self.producer.content, "candidate producer contents")
        _evidence(self.configuration, "candidate configuration")
        if type(self.queries) is not QueryConfiguration:
            raise TypeError("candidate subject needs an immutable query configuration")


@dataclass(frozen=True, eq=False)
class _VerificationExecution:
    subject: VerificationSubject
    thread: object
    task: object


_EXECUTION: ContextVar[Optional[_VerificationExecution]] = ContextVar("specola_extraction_candidate", default=None)


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def current_verification(adapter=None) -> Optional[_VerificationExecution]:
    """Read the current scope, refusing inherited or mismatched execution."""
    execution = _EXECUTION.get()
    if execution is None:
        return None
    if execution.thread is not current_thread() or execution.task is not _task():
        raise ExtractionGuardError("verification_execution_owner_changed", ("candidate scope belongs to another thread or task",))
    if adapter is not None and type(adapter) is not execution.subject.adapter_class:
        raise ExtractionGuardError("verification_subject_mismatch", ("adapter class differs from the candidate subject",))
    return execution


def require_ordinary_execution(metadata=None) -> None:
    """Reject candidate or declaration-only data before ordinary publication."""
    if current_verification() is not None:
        raise ExtractionGuardError("verification_only", ("candidate evidence cannot enter an ordinary command",))
    if metadata is not None:
        for flag in ("verification_only", "audit_only", "schema_only"):
            if flag in metadata and metadata[flag] is not False:
                raise ExtractionGuardError(flag, ("evidence or declaration-only metadata cannot enter an ordinary output",))


@contextmanager
def _candidate_scope(subject: VerificationSubject):
    if type(subject) is not VerificationSubject:
        raise TypeError("candidate execution needs a VerificationSubject")
    if current_verification() is not None:
        raise ExtractionGuardError("verification_scope_nested", ("candidate execution cannot replace an active subject",))
    token = _EXECUTION.set(_VerificationExecution(subject, current_thread(), _task()))
    try:
        yield
    finally:
        _EXECUTION.reset(token)


def run_extraction_candidates(identity, certification_subject, verification_subject: VerificationSubject, cases):
    """Run the existing raw conformance plans as labeled candidate evidence.

    The outer artifact retains its mode even when a factory fails before the
    first observation. It stores raw facts and grants no completeness result.
    """
    from .core_runner import freeze_json_artifact, run_core_conformance

    with _candidate_scope(verification_subject):
        artifact = run_core_conformance(certification_subject, cases)
    adapter_class = verification_subject.adapter_class
    metadata = vars(type)
    subject_document = {
        "adapter_module": metadata["__module__"].__get__(adapter_class),
        "adapter_qualname": metadata["__qualname__"].__get__(adapter_class),
        "producer": asdict(verification_subject.producer),
        "configuration": asdict(verification_subject.configuration),
        "queries": verification_subject.queries.document(),
    }
    return freeze_json_artifact(identity, "specola.extraction-candidates.v1", {
        "verification_only": True, "verification_subject": subject_document,
        "core": artifact.to_dict(),
    })
