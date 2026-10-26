"""Immutable completed/interrupted run summary projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .run_state import BindingState, RunSnapshot, recovery_transitions


@dataclass(frozen=True)
class BindingResult:
    specification: str
    monitor: str
    label: str
    state: BindingState
    final_robustness: Optional[float]
    worst_robustness: Optional[float]
    unit: Optional[str]
    recovery_count: int
    last_recovery_step: Optional[int]
    active: bool
    current_verdict_final: bool
    segment_samples: Optional[int]
    window_steps: Optional[int]


@dataclass(frozen=True)
class RunSummary:
    completion: str
    steps: int
    simulator_time: Optional[float]
    run_id: str
    spec_digest: Optional[str]
    result_path: Optional[str]
    bindings: Tuple[BindingResult, ...]


@dataclass(frozen=True)
class RenderedRunSummary:
    heading: str
    progress: str
    results_heading: str
    binding_lines: Tuple[str, ...]
    identity_lines: Tuple[str, ...]


def build_run_summary(record, snapshot, engine) -> RunSummary:
    completion = {
        "completed": "complete",
        "truncated": "partial",
        "failed": "failed",
    }.get(record.status, record.status)
    bindings = ()
    if snapshot is not None:
        bindings = tuple(
            BindingResult(
                specification=specification.name,
                monitor=binding.manager_name,
                label=binding.label,
                state=binding.state,
                final_robustness=binding.final_robustness,
                worst_robustness=binding.worst_robustness,
                unit=binding.unit,
                recovery_count=len(
                    recovery_transitions(binding.transitions)
                ),
                last_recovery_step=(
                    recovery_transitions(binding.transitions)[-1].step
                    if recovery_transitions(binding.transitions) else None
                ),
                active=binding.active,
                current_verdict_final=binding.final,
                segment_samples=binding.segment_samples,
                window_steps=binding.window_steps,
            )
            for specification in snapshot.specifications
            for binding in specification.bindings
        )
    path = (
        str(Path(record.path).resolve())
        if record.path is not None else None
    )
    return RunSummary(
        completion=completion,
        steps=record.steps,
        simulator_time=(
            snapshot.simulator_time if snapshot is not None else None
        ),
        run_id=record.episode_id,
        spec_digest=engine.spec_provenance.get("spec_digest"),
        result_path=path,
        bindings=bindings,
    )


def undetermined_reason(summary: RunSummary, specification: str) -> str:
    """Explain readiness from per-binding state, never global elapsed steps."""
    bindings = tuple(
        binding for binding in summary.bindings
        if binding.specification == specification and binding.active
    )
    if not bindings:
        return " (no live bindings)"
    unresolved = tuple(
        binding for binding in bindings
        if not binding.current_verdict_final
    )
    if any(binding.state is BindingState.SUSPENDED for binding in unresolved):
        return " (no final verdicts (windows unfilled or bindings suspended))"
    progress = {
        (binding.segment_samples, binding.window_steps)
        for binding in unresolved
        if binding.segment_samples is not None
        and binding.window_steps is not None
    }
    if len(progress) == 1 and all(
        binding.segment_samples is not None
        and binding.window_steps is not None
        for binding in unresolved
    ):
        samples, required = next(iter(progress))
        return " ({}/{} steps)".format(samples, required)
    return " (no final verdicts (windows unfilled or bindings suspended))"


def render_run_summary_sections(summary: RunSummary) -> RenderedRunSummary:
    heading = {
        "complete": "Episode complete",
        "partial": "Run interrupted",
        "failed": "Run failed",
    }.get(summary.completion, "Episode summary")
    progress = "{} steps{} · completion {}".format(
        summary.steps,
        (
            " · {:.3f} s simulator time".format(summary.simulator_time)
            if summary.simulator_time is not None else ""
        ),
        summary.completion,
    )
    binding_lines = []
    for binding in summary.bindings:
        unit = " " + binding.unit if binding.unit else ""
        final = (
            "—" if binding.final_robustness is None
            else "{:+.3f}{}".format(binding.final_robustness, unit)
        )
        worst = (
            "—" if binding.worst_robustness is None
            else "{:+.3f}{}".format(binding.worst_robustness, unit)
        )
        verdict = (
            "UNDETERMINED"
            if binding.final_robustness is None
            else "PASS" if binding.final_robustness >= 0 else "FAIL"
        )
        binding_lines.append(
            "  {} · {}  {} · {}  FINAL {}  WORST {}".format(
                binding.monitor,
                binding.label,
                binding.state.value,
                verdict,
                final,
                worst,
            )
        )
        if binding.recovery_count:
            binding_lines.append(
                "    {} recover{} · last at step {}".format(
                    binding.recovery_count,
                    "y" if binding.recovery_count == 1 else "ies",
                    binding.last_recovery_step,
                )
            )
    identity_lines = (
        "Run ID      " + summary.run_id,
        "Spec digest " + (summary.spec_digest or "unknown"),
        "Results     " + (summary.result_path or "none"),
    )
    return RenderedRunSummary(
        heading=heading,
        progress=progress,
        results_heading="Specification results",
        binding_lines=tuple(binding_lines),
        identity_lines=identity_lines,
    )


def render_run_summary(summary: RunSummary) -> str:
    sections = render_run_summary_sections(summary)
    lines = (
        (sections.heading, sections.progress, sections.results_heading)
        + sections.binding_lines
        + sections.identity_lines
    )
    return "\n".join(lines) + "\n"
