"""Whole-file equality for evidence pins, without pytest's text diff.

A pin is hundreds of kilobytes of canonical JSON. When one drifts,
pytest's assertion rewriting hands both strings to ``difflib``, whose
quadratic diff ran for 20 minutes and past 30 GB of memory on the
mock row on 2026-10-13 (found by ``faulthandler_timeout``). Comparing
digests fails in a millisecond and names the first differing line.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def assert_pin_matches(pin: Path, current: str) -> None:
    pinned = pin.read_text(encoding="utf-8")
    if pinned == current:
        return
    for number, (a, b) in enumerate(zip(pinned.splitlines(), current.splitlines()), 1):
        if a != b:
            first = f"line {number}: pinned {a[:120]!r} vs current {b[:120]!r}"
            break
    else:
        first = f"lengths {len(pinned)} vs {len(current)} (one is a prefix of the other)"
    raise AssertionError(
        f"{pin.name} differs from the regenerated evidence "
        f"(sha256 {hashlib.sha256(pinned.encode()).hexdigest()[:12]} vs "
        f"{hashlib.sha256(current.encode()).hexdigest()[:12]}); {first}"
    )
