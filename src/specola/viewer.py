"""
Ownership of on-screen viewers and of the process-wide state they change.

An adapter's ``render`` drives the simulator's own live view. Two rules
decide what ``close_render`` may release (``docs/adapter-hooks.md``,
the ``render`` row): a viewer present before a draw belongs to the
caller and stays open, unless an earlier draw of the adapter created it;
a viewer the adapter's rendering created belongs to the adapter, which keeps its own reference
because a native reset can drop or replace the simulator's. A setting
rendering changes for the whole process (matplotlib's interactive mode)
is shared by every adapter in it, so it is saved by the first holder and
restored by the last, whatever order they close in.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

class ViewerOwnership:
    """Track which viewers an adapter's rendering created.

    ``current`` returns the simulator's viewer reference (or ``None``);
    ``close`` releases one viewer. Wrap every draw in ``render`` and call
    ``release`` from ``close_render``.
    """

    def __init__(self, current: Callable[[], Any], close: Callable[[Any], None]) -> None:
        self._current = current
        self._close = close
        self._owned: List[Any] = []

    def render(self, draw: Callable[[], None]) -> None:
        """Run ``draw`` and claim the viewer if this draw created it.

        A viewer present before the draw is the caller's unless an earlier
        draw of this adapter created it, so a viewer the caller opens
        between renders is never claimed.
        """
        before = self._current()
        try:
            draw()
        finally:
            # A draw that opens a window and then raises still owns it.
            viewer = self._current()
            if (viewer is not None and viewer is not before
                    and not any(viewer is owned for owned in self._owned)):
                self._owned.append(viewer)

    def release(self) -> None:
        """Close every viewer the adapter created, once; leave the caller's.

        Every viewer is attempted even when one close raises; the first
        error is raised after the rest are closed.
        """
        owned, self._owned = self._owned, []
        error = None
        for viewer in owned:
            try:
                self._close(viewer)
            except BaseException as exc:  # an interrupt still closes the rest
                error = error or exc
        if error is not None:
            raise error


class SharedProcessState:
    """A process-wide setting several adapters' rendering may change.

    The first ``acquire`` saves the current value; the last ``release``
    restores it. Holders are any objects (usually the adapter); a repeated
    ``release`` is a no-op.
    """

    def __init__(self, read: Callable[[], Any], write: Callable[[Any], None]) -> None:
        self._read = read
        self._write = write
        self._holders: set = set()
        self._saved: Any = None
        self._lock = threading.Lock()

    def acquire(self, holder: Any) -> None:
        with self._lock:
            if not self._holders:
                self._saved = self._read()
            self._holders.add(id(holder))

    def release(self, holder: Any) -> None:
        with self._lock:
            if id(holder) not in self._holders:
                return
            self._holders.discard(id(holder))
            if not self._holders:
                self._write(self._saved)


_matplotlib_interactive: Optional[SharedProcessState] = None


def matplotlib_interactive() -> SharedProcessState:
    """The process's matplotlib interactive mode, as one shared state."""
    global _matplotlib_interactive
    if _matplotlib_interactive is None:
        import matplotlib

        _matplotlib_interactive = SharedProcessState(matplotlib.is_interactive, matplotlib.interactive)
    return _matplotlib_interactive


_pygame_session: Optional[SharedProcessState] = None


def pygame_session() -> SharedProcessState:
    """pygame's process-wide initialization, as one shared state.

    The first holder records whether pygame was already running; the last
    release calls ``pygame.quit()`` only if it was not, so a caller's own
    pygame display survives. A native viewer whose ``close`` quits pygame
    must not be closed while this state has other holders.
    """
    global _pygame_session
    if _pygame_session is None:
        import pygame

        def restore(was_running: Any) -> None:
            if not was_running:
                pygame.quit()

        def running() -> bool:
            # get_init is True only after pygame.init(); a caller may have
            # started the display alone.
            return bool(pygame.get_init() or pygame.display.get_init())

        _pygame_session = SharedProcessState(running, restore)
    return _pygame_session
