"""
pager.py — optional auto-pager for long CLI output
==================================================
Pipes captured output through ``$PAGER`` (default ``less -FRXS``) when
stdout is a terminal, so long discovery output is read from the top.
``-F`` makes short output indistinguishable from a plain print; pipes
and redirects bypass paging entirely, keeping scripted output stable.
``-S`` chops long lines instead of wrapping them, so wide joints x
signals matrix rows stay one row per joint under the pager.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import signal
import subprocess
import sys
from typing import Iterator, List


def pager_command() -> List[str]:
    """Resolve the pager argv from ``$PAGER``, defaulting to less."""
    raw = os.environ.get("PAGER", "").strip()

    if not raw:
        return ["less", "-FRXS"]

    return shlex.split(raw)


def should_page(
    disable: bool = False,
) -> bool:
    return (
        not disable
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )


@contextlib.contextmanager
def maybe_page(
    disable: bool = False,
) -> Iterator[None]:
    """
    Capture prints inside the block and page them on a terminal.

    Color codes survive capture because ``cli_common.C`` resolves them
    at import time against the real stdout. On any pager failure the
    captured text prints plainly — output is never lost.
    """
    if not should_page(disable):
        yield
        return

    buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(buffer):
            yield
    except BaseException:
        sys.stdout.write(buffer.getvalue())
        raise

    text = buffer.getvalue()

    # Ctrl-C belongs to the pager while it owns the terminal (less uses
    # it to interrupt searches; quit is 'q'). Ignore SIGINT here so the
    # parent does not die with a KeyboardInterrupt traceback.
    previous_handler = signal.signal(
        signal.SIGINT,
        signal.SIG_IGN,
    )

    try:
        subprocess.run(
            pager_command(),
            input=text.encode(
                "utf-8",
                errors="replace",
            ),
            check=False,
        )
    except (FileNotFoundError, OSError):
        sys.stdout.write(text)
    finally:
        signal.signal(
            signal.SIGINT,
            previous_handler,
        )
