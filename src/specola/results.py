"""
results.py — structured evaluation results (design §5.2).

Robustness is reported with an explicit epistemic status; None is never
an implicit catch-all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .monitor_types import Binding


class EvaluationStatus(Enum):
    # The current window verdict is non-provisional; later sliding windows
    # may still change it.
    ACTIVE = "active"
    WARMING_UP = "warming_up"         # provisional robustness, window filling
    SUSPENDED = "suspended"           # binding invalid; no current robustness
    UNKNOWN_MISSING_DATA = "unknown_missing_data"
    ERROR = "error"


@dataclass(frozen=True)
class BindingEvaluation:
    spec_name: str
    binding: Binding
    robustness: Optional[float]       # None unless meaningfully current
    status: EvaluationStatus
    verdict_final: bool
    timestep: int
    time: float
    # Explicit temporal readiness. Renderers must not infer this from
    # robustness or global elapsed time because suspension resets the
    # contiguous segment.
    segment_samples: Optional[int] = None
    window_steps: Optional[int] = None
    error_message: Optional[str] = None
    component_values: Tuple[Tuple[int, str, float], ...] = ()


@dataclass(frozen=True)
class EngineResult:
    """One engine update: evaluations plus lifecycle events and metrics."""

    timestep: int
    time: float
    evaluations: Tuple[BindingEvaluation, ...]
    created: Tuple[Tuple[str, Binding], ...] = field(default=())
    resumed: Tuple[Tuple[str, Binding], ...] = field(default=())
    pruned_count: int = 0
    binding_recomputes: int = 0
    update_seconds: float = 0.0

    def for_spec(self, spec_name: str) -> Tuple[BindingEvaluation, ...]:
        return tuple(
            e for e in self.evaluations if e.spec_name == spec_name
        )

    @property
    def any_violation(self) -> bool:
        return any(
            e.robustness is not None
            and e.robustness < 0
            and e.verdict_final
            for e in self.evaluations
        )
