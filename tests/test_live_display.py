"""
Unit tests for live_display: region bookkeeping, truncation, event
interleaving, rate cap, finalize, and TTY gating. All against a fake
TTY stream with injected size/clock — no real terminal involved.
"""

from __future__ import annotations

import io
import os

import pytest

from specola.live_display import (
    LiveDisplay,
    supports_live,
    visible_truncate,
)

CLEAR = "\x1b[F\x1b[2K"
RESET = "\x1b[0m"


class FakeTty(io.StringIO):
    def isatty(self) -> bool:  # pragma: no cover - trivial
        return True


def make_display(cols=80, rows=24, min_interval=0.05):
    stream = FakeTty()
    clock = {"now": 0.0}

    def time_fn():
        return clock["now"]

    def size_fn():
        return os.terminal_size((cols, rows))

    display = LiveDisplay(
        stream=stream, size_fn=size_fn, time_fn=time_fn,
        min_interval=min_interval,
    )
    return display, stream, clock


# ---------------------------------------------------------------------------
# visible_truncate
# ---------------------------------------------------------------------------

def test_truncate_under_width_unchanged():
    assert visible_truncate("short", 80) == "short"


def test_truncate_plain_overwidth():
    out = visible_truncate("abcdefghij", 5)
    assert out == "abcd…" + RESET


def test_truncate_counts_visible_chars_only():
    colored = "\x1b[31m" + "abcdefghij" + RESET
    # 10 visible chars, width 20: escapes must not count.
    assert visible_truncate(colored, 20) == colored
    cut = visible_truncate(colored, 5)
    assert cut.startswith("\x1b[31mabcd")
    assert cut.endswith("…" + RESET)


# ---------------------------------------------------------------------------
# LiveDisplay
# ---------------------------------------------------------------------------

def test_update_paints_and_clears_previous_region():
    display, stream, _ = make_display()
    display.update(["a", "b", "c"])
    first = stream.getvalue()
    assert CLEAR not in first          # nothing to clear on first paint
    assert first == "a\nb\nc\n"

    display.update(["x", "y"], force=True)
    second = stream.getvalue()[len(first):]
    assert second.count(CLEAR) == 3    # exactly the previous region height
    assert second.endswith("x\ny\n")


def test_resize_applies_on_next_paint():
    stream = FakeTty()
    sizes = {"cols": 80}
    display = LiveDisplay(
        stream=stream,
        size_fn=lambda: os.terminal_size((sizes["cols"], 24)),
        time_fn=lambda: 0.0,
    )
    display.update(["m" * 40])
    sizes["cols"] = 10
    display.update(["m" * 40], force=True)
    tail = stream.getvalue().splitlines()[-1]
    # 9 visible + ellipsis + reset
    assert tail.replace(CLEAR.replace("\n", ""), "").endswith("…" + RESET)
    assert "m" * 10 not in tail


def test_emit_interleaves_events_above_block():
    display, stream, _ = make_display()
    display.update(["block1", "block2"])
    before = stream.getvalue()
    display.emit("EVENT")
    added = stream.getvalue()[len(before):]
    # Order: clear old region (2 lines), event line, repainted block.
    assert added.startswith(CLEAR * 2)
    event_pos = added.find("EVENT")
    block_pos = added.find("block1")
    assert -1 < event_pos < block_pos
    assert added.endswith("block1\nblock2\n")


def test_rate_cap_skips_then_force_paints():
    display, stream, clock = make_display(min_interval=1.0)
    display.update(["v1"])
    len1 = len(stream.getvalue())
    clock["now"] = 0.1
    display.update(["v2"])              # within interval: no paint
    assert len(stream.getvalue()) == len1
    display.update(["v3"], force=True)  # force: paints pending
    assert stream.getvalue().endswith("v3\n")


def test_finalize_paints_pending_and_releases_region():
    display, stream, clock = make_display(min_interval=1.0)
    display.update(["v1"])
    clock["now"] = 0.1
    display.update(["v2"])              # pending, unpainted
    display.finalize()
    assert stream.getvalue().endswith("v2\n")
    size = len(stream.getvalue())
    display.finalize()                  # idempotent
    assert len(stream.getvalue()) == size
    # Region released: a later update clears nothing.
    display.update(["after"], force=True)
    tail = stream.getvalue()[size:]
    assert CLEAR not in tail


def test_supports_live_gating():
    non_tty = io.StringIO()
    assert not supports_live(non_tty)
    tty = FakeTty()
    old = os.environ.get("TERM")
    try:
        os.environ["TERM"] = "xterm-256color"
        assert supports_live(tty)
        os.environ["TERM"] = "dumb"
        assert not supports_live(tty)
    finally:
        if old is None:
            os.environ.pop("TERM", None)
        else:
            os.environ["TERM"] = old
