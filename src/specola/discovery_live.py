"""Explicit, non-interactive Rich discovery overview."""

from __future__ import annotations

from .discovery_filter import DiscoveryView
from .discovery_render import palette_lines
from .selector_palette import palette_for_view


def render_rich_discovery(view: DiscoveryView, *, console=None) -> None:
    try:
        from rich.console import Console
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
    except ImportError as exc:
        raise RuntimeError(
            "Rich discovery requires the bounded UI dependencies"
        ) from exc

    class_table = Table(title="Monitor-facing classes")
    class_table.add_column("Class")
    class_table.add_column("Entities", justify="right")
    class_table.add_column("Signals", justify="right")
    live_classes = {
        record.entity_class: record for record in view.model.classes
    }
    declared_classes = {
        record.entity_class: record
        for record in view.model.declared_classes
    }
    for entity_class in sorted(set(live_classes) | set(declared_classes)):
        live = live_classes.get(entity_class)
        declared = declared_classes.get(entity_class)
        if live is not None and live.raw and not view.query.include_raw:
            continue
        class_table.add_row(
            entity_class,
            str(
                live.live_count
                if live is not None else declared.live_count
            ),
            str(
                live.property_count
                if live is not None else len(declared.components)
            ),
        )
    role_table = Table(title="Declared roles")
    role_table.add_column("Role")
    role_table.add_column("Entities", justify="right")
    roles = {}
    for entity in view.entities:
        if entity.role:
            roles[entity.role] = roles.get(entity.role, 0) + 1
    for role, count in sorted(roles.items()):
        role_table.add_row(role, str(count))

    subtitle = "{} entities · {} selected scalar properties".format(
        view.after.entity_count, view.after.property_count
    )
    identity = "{} · {} · reset snapshot".format(
        view.model.simulator.upper(),
        view.model.task or view.model.adapter_type,
    )
    body = [identity, class_table]
    if roles:
        body.append(role_table)
    if view.model.warnings:
        body.append("Warnings\n" + "\n".join(
            "• " + warning for warning in view.model.warnings
        ))
    # Text, not a plain string: Rich parses square brackets as style
    # markup, so a raw "object_*[mass]" renders as "object_*" with the
    # component silently eaten — losing exactly the part of the
    # reference that decides what gets monitored.
    from rich.text import Text

    forms, refusal = palette_for_view(view)
    if refusal is not None:
        body.append(Text("Ways to select this\n  " + refusal))
    elif forms:
        body.append(Text(
            "Ways to select this\n" + "\n".join(palette_lines(forms))
        ))
    body.append(
        "Filter with --class/--role/--signal to narrow the selection."
    )
    output = console or Console()
    output.print(Panel(
        Group(*body),
        title="Specola Discover",
        subtitle=subtitle,
    ))
