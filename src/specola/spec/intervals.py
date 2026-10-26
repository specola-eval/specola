"""
intervals.py — authored-bound → step conversion (dt applies here).

Shared by semantic compilation and the native backend so both layers
convert bounds identically.
"""

from __future__ import annotations

import math
from typing import Tuple

from .ast import Bound, Interval


def bound_to_steps(bound: Bound, dt: float) -> int:
    if bound.in_seconds:
        # Explicit round-half-up avoids Python's ties-to-even behavior.
        return int(math.floor(bound.value / dt + 0.5))
    return int(bound.value)


def convert_interval(bounds: Interval, dt: float) -> Tuple[int, int]:
    """Convert an authored interval to converted inclusive step bounds.

    Sample-count convention: authored [0,N] covers exactly max(N, 1)
    samples, so the backend window is [0, N-1].
    """
    lower = bound_to_steps(bounds.lower, dt)
    authored_upper = bound_to_steps(bounds.upper, dt)

    if authored_upper < lower:
        raise ValueError(
            f"Invalid interval [{bounds.lower.value},"
            f"{bounds.upper.value}]: upper bound < lower bound"
        )

    upper = authored_upper - 1 if authored_upper > 0 else 0
    if upper < lower:
        upper = lower

    return lower, upper
