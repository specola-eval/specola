"""Scrolling plain renderer for typed run transitions."""

from __future__ import annotations

from typing import List

from .run_state import RunSnapshot, RunTransition


def _colored_state(value: str) -> str:
    from . import cli_common

    helper = cli_common.STATE_COLOR_NAMES.get(value)
    if helper is None:
        return value
    return getattr(cli_common, helper)(value)


def transition_line(
    specification: str,
    binding_label: str,
    transition: RunTransition,
) -> str:
    previous = (
        transition.from_state.value
        if transition.from_state is not None else "—"
    )
    return (
        "step {step:<5d} {specification} · {binding} · "
        "{previous} → {current}"
    ).format(
        step=transition.step,
        specification=specification,
        binding=binding_label or "global",
        previous=previous,
        current=_colored_state(transition.to_state.value),
    )


def render_plain_transitions(
    snapshot: RunSnapshot,
    *,
    summary_only: bool = False,
) -> List[str]:
    """Render only literal transitions; never periodic full snapshots."""
    if summary_only:
        return []
    labels = {
        (specification.name, binding.binding_key): binding.label
        for specification in snapshot.specifications
        for binding in specification.bindings
    }
    return [
        transition_line(
            specification,
            labels.get((specification, binding_key), binding_key),
            transition,
        )
        for specification, binding_key, transition in snapshot.transitions
    ]
