"""
base.py — temporal backend protocol.

A backend consumes one sample of named atomic-predicate signals per
logical timestep and returns the formula's current robustness. There is
deliberately no advance-without-sample operation: suspended monitor
instances do not call their backend at all (RESET_AND_REWARM policy).
"""

from __future__ import annotations

from typing import Mapping

try:
    from typing import Protocol
except ImportError:  # Python 3.7
    Protocol = object  # type: ignore


class TemporalBackend(Protocol):
    def update(
        self,
        timestep: int,
        signals: Mapping[str, float],
    ) -> float:
        """Consume one sample per atomic predicate; return robustness."""

    def reset(self) -> None:
        """Discard all history (used when rewarming after suspension)."""
