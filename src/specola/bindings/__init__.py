"""
specola.bindings — parameter-binding eligibility and enumeration.

eligibility.slot_candidates   per-slot candidate sets (never class unions)
enumerate.enumerate_complete  complete injective enumeration — the
                              reference implementation for binding sets
"""

from __future__ import annotations

from .eligibility import slot_candidates  # noqa: F401
from .enumerate import enumerate_complete  # noqa: F401
from .incremental import IncrementalBindingSet  # noqa: F401
