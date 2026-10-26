"""Renderer-neutral presentation helpers for run snapshots."""

from __future__ import annotations

from .run_state import BindingSnapshot, RunSnapshot


def format_binding(binding: BindingSnapshot) -> str:
    unit = " " + binding.unit if binding.unit else ""
    rho = (
        "—" if binding.current_robustness is None
        else "{:+.3f}".format(binding.current_robustness + 0.0)
    )
    worst = (
        "—" if binding.worst_robustness is None
        else "{:+.3f}".format(binding.worst_robustness + 0.0)
    )
    return "    {:<30} {:<13} NOW={}{} WORST={}{}".format(
        binding.label,
        binding.state.value,
        rho,
        unit,
        worst,
        unit,
    )


def render_run_snapshot(snapshot: RunSnapshot) -> list:
    lines = [
        "  step={}  t={:.3f}s".format(
            snapshot.step, snapshot.simulator_time
        ),
        "  Specifications",
    ]
    for specification in snapshot.specifications:
        if specification.state_counts:
            counts = " · ".join(
                "{} {}".format(count, state.value)
                for state, count in specification.state_counts
            )
        else:
            counts = specification.state.value.replace("_", " ")
        lines.append("  ▼ {:<28} {}".format(specification.name, counts))
        for binding in specification.bindings:
            lines.append(format_binding(binding))
    return lines


class OverlayPanel:
    """Base text plus a dismissible overlay for one text widget.

    Both terminal apps show durable panel content (inspector, authoring
    preview) that a transient overlay (help) may cover; Escape restores
    the covered text. The widget only needs an ``update(str)`` method.
    """

    def __init__(self, widget, text: str) -> None:
        self._widget = widget
        self.text = text
        self._overlay = None

    def set_text(self, text: str) -> None:
        self.text = text
        if self._overlay is None:
            self._widget.update(text)

    def show_overlay(self, text: str) -> None:
        self._overlay = text
        self._widget.update(text)

    def dismiss_overlay(self) -> bool:
        """Restore the covered text; True if an overlay was showing."""
        if self._overlay is None:
            return False
        self._overlay = None
        self._widget.update(self.text)
        return True
