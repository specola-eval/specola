"""The Specola mark in terminal output.

Three forms, one per surface:

- ``lockup``: the icon beside the name, tagline, and environment summary,
  for ``specola`` with no command;
- ``panel``: the icon above the name, then one row per adapter, for
  ``specola doctor``;
- ``version_line``: one line, for ``specola --version``.

Branding is printed only to an interactive UTF-8 terminal and never when
``NO_COLOR`` is set or ``TERM`` is ``dumb``; the callers fall back to plain text otherwise, so
piped output and logs stay unchanged. A terminal too narrow for the
lockup or panel gets the one-line mark and plain rows instead. The icon
art is generated from the Newsreader font by
``scripts/make_brand_assets.py``.
"""

from __future__ import annotations

import codecs
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import IO, List, Optional, Sequence, Tuple

from . import __version__
from ._brand_art import ICON_8, ICON_10

NAME = "Specola"
TAGLINE = "Specification monitoring"
# One-line mark: the icon's rule passing through the S. The circle on the
# spine needs more than one cell, so it appears only in the block icon.
MARK = "─S─"

# Brand navy for light backgrounds, light navy for dark ones. A mid navy
# that just clears 4.5:1 on pure black and pure white (#5277B1: 4.63 and
# 4.53) drops to 3.1-3.6:1 on common dark-grey terminal backgrounds
# (#282C34, #1E1F20), so an unknown background gets the terminal's own
# foreground colour.
COLORS = {"light": "#0F2747", "dark": "#A9C1E8", "unknown": "default"}

# Optional extras and the package whose presence shows each is installed.
EXTRAS: Tuple[Tuple[str, str], ...] = (
    ("rtamt", "rtamt"),
    ("report", "pyarrow"),
    ("web", "flask"),
)


def enabled(stream: Optional[IO[str]] = None) -> bool:
    """Whether branding may be printed to ``stream`` (default stdout).

    The art assumes a UTF-8 stream whose terminal renders East Asian
    ambiguous-width characters (``─``, ``○``, ``█``) one cell wide, the
    default outside CJK locales. ``NO_COLOR`` and ``TERM=dumb`` turn
    branding off.
    """
    stream = sys.stdout if stream is None else stream
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    # A text buffer without an encoding (io.StringIO) holds str, not bytes.
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        utf8 = codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False
    isatty = getattr(stream, "isatty", None)
    return utf8 and bool(isatty and isatty())


def theme() -> str:
    """``light``, ``dark``, or ``unknown`` terminal background.

    ``SPECOLA_THEME`` wins; otherwise ``COLORFGBG`` (``fg;bg``, set by
    some terminals) is read. Only the 16 standard colour codes are
    recognised: 0-6 and 8 are dark, 7 and 9-15 light. Any other value
    (an index into a 256-colour palette, say) is ``unknown``.
    """
    chosen = os.environ.get("SPECOLA_THEME", "").strip().lower()
    if chosen in ("light", "dark"):
        return chosen
    try:
        background = int(os.environ.get("COLORFGBG", "").split(";")[-1])
    except ValueError:
        return "unknown"
    if not 0 <= background <= 15:
        return "unknown"
    return "dark" if background in (0, 1, 2, 3, 4, 5, 6, 8) else "light"


def brand_color() -> str:
    return COLORS[theme()]


def mark_text(label: str = ""):
    """``─S─ Specola`` with an optional label, as a Rich ``Text``."""
    from rich.text import Text

    color = brand_color()
    text = Text.assemble((MARK + " ", color), (NAME, f"bold {color}"))
    if label:
        text.append(f"  {label}", style="dim")
    return text


def version_line(version: str = __version__):
    from rich.text import Text

    color = brand_color()
    return Text.assemble((MARK + " ", color), (NAME + " ", f"bold {color}"),
                         (version, "bold"), ("  " + TAGLINE, "dim"))


@dataclass(frozen=True)
class Environment:
    version: str
    python: str
    adapters: Sequence  # registry.AdapterStatus
    extras: Tuple[str, ...]

    @property
    def ready(self) -> List[str]:
        return [a.name for a in self.adapters if a.ready]


def environment(load_third_party: bool = True) -> Environment:
    from .registry import adapter_status

    extras = tuple(name for name, module in EXTRAS
                   if importlib.util.find_spec(module) is not None)
    return Environment(version=__version__, python=sys.version.split()[0],
                       adapters=adapter_status(load_third_party), extras=extras)


def _icon(rows: Tuple[str, ...], color: str):
    from rich.text import Text

    return [Text(line, style=color) for line in rows]


def status_label(status) -> str:
    """One adapter's doctor status in words.

    "packages found" means every simulator package the adapter declares
    can be located; nothing is imported, so it does not prove that the
    simulator starts.
    """
    if not status.checked:
        return "not checked"
    if status.load_error:
        return f"adapter failed to load: {status.load_error}"
    if status.missing:
        return "missing " + ", ".join(status.missing)
    return "packages found"


def lockup(env: Environment):
    """Icon left; name, tagline, and environment summary right."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    color = brand_color()
    right = [
        Text(""),
        Text(NAME, style=f"bold {color}"),
        Text(TAGLINE, style="dim"),
        Text(""),
        Text.assemble(("version  ", "dim"), (env.version, "bold"),
                      ("   python ", "dim"), env.python),
        Text.assemble(("adapters ", "dim"), (f"{len(env.ready)} of {len(env.adapters)}", "bold"),
                      " with packages found"),
        Text.assemble(("extras   ", "dim"), ", ".join(env.extras) or "none"),
        Text(""),
        Text.assemble(("details  ", "dim"), ("specola doctor", "bold")),
    ]
    grid = Table.grid(padding=(0, 3))
    grid.add_column(no_wrap=True)
    grid.add_column()
    grid.add_row(Group(*_icon(ICON_8, color)), Group(*right))
    return grid


def panel(env: Environment):
    """Icon above the name, then one row per adapter."""
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    color = brand_color()
    head = [Align.center(line) for line in _icon(ICON_10, color)]
    head += [Text(""),
             Align.center(Text(" ".join(NAME), style=f"bold {color}")),
             Align.center(Text(TAGLINE, style="dim"))]

    rows = Table(box=None, pad_edge=False, padding=(0, 2), header_style="dim")
    rows.add_column("adapter", no_wrap=True)
    rows.add_column("status")
    for status in env.adapters:
        style = "bold" if status.ready else ""
        rows.add_row(Text(status.name, style=style), Text(status_label(status), style=style))

    foot = Text.assemble(("specola ", "dim"), (env.version, "bold"),
                         ("  ·  python ", "dim"), env.python,
                         ("  ·  extras ", "dim"), ", ".join(env.extras) or "none")
    return Panel(Group(*head, Text(""), Align.center(rows), Text(""), Align.center(foot)),
                 border_style="dim", padding=(1, 4), expand=False)


def print_branded(renderable, stream: Optional[IO[str]] = None, narrow=None) -> None:
    """Print ``renderable``, or ``narrow`` when the terminal cannot fit it unwrapped."""
    from rich.console import Console

    console = Console(file=stream or sys.stdout, highlight=False)
    # Measured against unlimited width, since Rich clamps to the console's.
    natural = console.measure(renderable, options=console.options.update_width(10_000))
    if narrow is not None and natural.maximum > console.width:
        renderable = narrow
    console.print(renderable)
