"""Non-interactive Rich Live renderer for typed run snapshots."""

from __future__ import annotations

from time import monotonic

from .run_present import render_run_snapshot


class RichRunLive:
    def __init__(self, refresh_rate: float = 10.0) -> None:
        try:
            from rich.live import Live
            from rich.panel import Panel
        except ImportError as exc:
            raise RuntimeError(
                "Rich run UI requires the bounded UI dependencies"
            ) from exc
        from . import branding

        self._panel = Panel
        self._title = branding.mark_text("Run") if branding.enabled() else "Specola Run"
        self._refresh_interval = 1.0 / refresh_rate
        self._next_refresh = 0.0
        self._live = Live(
            self._panel("Starting monitor…", title=self._title),
            auto_refresh=False,
            transient=False,
        )

    def __enter__(self):
        self._live.__enter__()
        return self

    def update(self, snapshot) -> None:
        now = monotonic()
        refresh = now >= self._next_refresh
        self._live.update(
            self._panel(
                "\n".join(render_run_snapshot(snapshot)),
                title=self._title,
            ),
            refresh=refresh,
        )
        if refresh:
            self._next_refresh = now + self._refresh_interval

    def emit(self, line: str) -> None:
        self._live.console.print(line)

    def __exit__(self, exc_type, exc, traceback):
        return self._live.__exit__(exc_type, exc, traceback)
