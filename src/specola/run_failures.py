"""Truthful failure records shared by terminal renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Optional


@dataclass(frozen=True)
class RunFailure:
    phase: str
    adapter: str
    task: Optional[str]
    error: str
    diagnostics_path: Optional[str]
    result_path: Optional[str]
    extraction_status: Optional[str] = None


def render_run_failure(failure: RunFailure) -> str:
    return (
        "Run failed\n"
        "Phase       {phase}\n"
        "Adapter     {adapter}\n"
        "Task        {task}\n"
        "Error       {error}\n"
        "{extraction}"
        "Results     {results}\n"
        "Diagnostics {diagnostics}\n"
    ).format(
        phase=failure.phase,
        adapter=failure.adapter,
        task=failure.task or "none",
        error=failure.error,
        extraction=("Observation none\nExtraction  " + failure.extraction_status + "\n"
                    if failure.extraction_status is not None else ""),
        results=failure.result_path or "none",
        diagnostics=failure.diagnostics_path or "none",
    )


def report_extraction_failure(exc, *, command: str, adapter: str, task: Optional[str]) -> None:
    """Retain a rejected check/discovery trace outside ordinary output paths."""
    directory = Path(tempfile.mkdtemp(prefix="specola-startup-"))
    diagnostics = directory / "run-error.log"
    diagnostics.write_text(traceback.format_exc(), encoding="utf-8")
    failure = RunFailure(
        phase=command + " extraction guard", adapter=adapter, task=task,
        error=type(exc).__name__ + ": " + str(exc),
        diagnostics_path=str(diagnostics), result_path=None,
        extraction_status=exc.status,
    )
    print(render_run_failure(failure), file=sys.stderr)
