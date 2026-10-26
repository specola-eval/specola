"""
live_display.py — in-place live region for streaming CLI output.

Generic terminal mechanics only: a block of lines at the bottom of the
scrollback that is rewritten in place (VT100 cursor-up + erase-line),
while ordinary lines ("events") can still be inserted above it and
scroll away normally. No alternate screen is used, so the terminal's
scrollback stays intact and usable.

Contracts and limitations (deliberate):

- ANSI handling covers SGR sequences only (``\\x1b[...m``) — the only
  escapes this codebase emits (see ``cli_common.C``). Other escape
  classes in input lines would break the visible-width math.
- Foreign stdout written mid-run by third-party code (e.g. simulator
  warnings) shifts the region; the next repaint's cursor-up math will
  overwrite the wrong lines once, then self-heal on later repaints.
  The robust fix is redirecting foreign output, out of scope here.
- Repaints re-query the terminal size each time, so resizes are picked
  up without SIGWINCH handling.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Callable, List, Optional, Sequence

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # SGR only, by contract
_RESET = "\x1b[0m"
# Cursor to start of previous line, then erase that whole line.
_PREV_LINE_CLEAR = "\x1b[F\x1b[2K"


def supports_live(stream=None) -> bool:
    """Whether ``stream`` (default stdout) can host an in-place region."""
    if stream is None:
        stream = sys.stdout
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return os.environ.get("TERM") != "dumb"


def visible_truncate(line: str, width: int) -> str:
    """Cut ``line`` to ``width`` printable columns, keeping SGR escapes.

    Escapes are copied through for free; only printable characters count
    toward the width. A truncated line ends with an ellipsis and an
    unconditional SGR reset so a cut mid-color never bleeds into the
    next line.
    """
    if width <= 0:
        return _RESET
    visible = 0
    out: List[str] = []
    pos = 0
    truncated = False
    for match in _ANSI_RE.finditer(line):
        chunk = line[pos:match.start()]
        for ch in chunk:
            if visible == width - 1:
                truncated = True
                break
            out.append(ch)
            visible += 1
        if truncated:
            break
        out.append(match.group())
        pos = match.end()
    if not truncated:
        for ch in line[pos:]:
            if visible == width - 1:
                truncated = True
                break
            out.append(ch)
            visible += 1
    if not truncated:
        return line
    return "".join(out) + "…" + _RESET


class LiveDisplay:
    """One rewritable block of lines at the bottom of the scrollback.

    ``update()`` replaces the block (rate-capped); ``emit()`` inserts
    permanent lines above it; ``finalize()`` settles the block so
    subsequent ordinary prints scroll below it.
    """

    def __init__(
        self,
        stream=None,
        size_fn: Optional[Callable[[], os.terminal_size]] = None,
        time_fn: Optional[Callable[[], float]] = None,
        min_interval: float = 0.05,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._size_fn = size_fn or shutil.get_terminal_size
        self._time_fn = time_fn or time.monotonic
        self._min_interval = min_interval
        self._live_lines = 0
        self._pending: Optional[List[str]] = None
        self._last_paint: Optional[float] = None

    # ------------------------------------------------------------------
    def height_budget(self) -> int:
        """Max lines the live block should occupy on this terminal."""
        return max(4, self._size_fn().lines - 2)

    def update(self, lines: Sequence[str], force: bool = False) -> None:
        """Replace the live block with ``lines`` (repaint rate-capped)."""
        self._pending = list(lines)
        now = self._time_fn()
        if (
            not force
            and self._last_paint is not None
            and now - self._last_paint < self._min_interval
        ):
            return
        self._paint(now)

    def emit(self, *event_lines: str) -> None:
        """Insert permanent lines above the live block, then repaint it."""
        width = self._size_fn().columns
        parts = [_PREV_LINE_CLEAR * self._live_lines]
        for line in event_lines:
            parts.append(visible_truncate(line, width) + "\n")
        self._live_lines = 0
        self._stream.write("".join(parts))
        # Events always force the freshest block back underneath.
        if self._pending is not None:
            self._paint(self._time_fn())
        else:
            self._stream.flush()

    def finalize(self) -> None:
        """Paint any pending block, then release the region (idempotent).

        After this, ordinary prints scroll below the last painted block.
        """
        if self._pending is not None:
            self._paint(self._time_fn())
        self._pending = None
        self._live_lines = 0

    # ------------------------------------------------------------------
    def _paint(self, now: float) -> None:
        if self._pending is None:
            return
        width = self._size_fn().columns
        parts = [_PREV_LINE_CLEAR * self._live_lines]
        for line in self._pending:
            parts.append(visible_truncate(line, width) + "\n")
        self._stream.write("".join(parts))
        self._stream.flush()
        self._live_lines = len(self._pending)
        self._last_paint = now
