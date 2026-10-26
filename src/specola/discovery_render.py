"""Concise and detailed plain discovery presentation."""

from __future__ import annotations

from collections import Counter
from typing import List

import yaml

from .discovery_filter import DiscoveryView
from .discovery_selector import first_selector_preview, selector_preview
from .selector_palette import (
    default_form,
    palette_for_view,
    starter_yaml_for_form,
)


def palette_lines(forms) -> List[str]:
    """Render selector forms the way every non-interactive surface does.

    Forms only: the reference, and for a two-slot form its partner.
    A form the grammar cannot express here keeps its place with the
    reason attached, because an author needs to know a shape is
    unavailable as much as they need the ones that are.
    """
    out: List[str] = []
    for form in forms:
        text = "  " + form.reference
        if form.partner_reference:
            text += "  ↔  " + form.partner_reference
        if not form.available:
            text += "    — {}".format(form.unavailable_reason)
        out.append(text)
    return out


_palette_lines = palette_lines


def _position(values):
    if values is None:
        return "—"
    return "(" + ", ".join("{:.3f}".format(value) for value in values) + ")"


def render_discovery(
    view: DiscoveryView,
    *,
    detailed: bool = False,
    verbose: bool = False,
    include_cli_guidance: bool = True,
) -> str:
    """Render a bounded semantic summary or explicitly selected detail."""
    model = view.model
    lines: List[str] = []
    heading = "Specola Discovery — {}".format(model.simulator)
    lines.extend((heading, "=" * len(heading)))
    if model.task:
        lines.append("Task: {}".format(model.task))
    if model.task_description and model.task_description != model.task:
        lines.append("Instruction: {}".format(model.task_description))
    lines.append(
        "Timing: adapter_dt={}s  spec_dt={}s  spatial_dims={}".format(
            model.adapter_dt, model.effective_dt, model.spatial_dims
        )
    )
    lines.append(
        "Catalogue: {} entities, {} scalar properties".format(
            view.before.entity_count, view.before.property_count
        )
    )
    lines.append("")

    if not detailed:
        lines.append("Entities by semantic class")
        semantic = [record for record in model.classes if not record.raw]
        for record in semantic:
            lines.append(
                "  {:<18} {:>4} entities  {:>5} signals".format(
                    record.entity_class,
                    record.live_count,
                    record.property_count,
                )
            )

        roles = Counter(
            entity.role for entity in model.entities
            if not entity.raw and entity.role
        )
        if roles:
            lines.append("")
            lines.append("Declared task roles")
            for role, count in sorted(roles.items()):
                lines.append("  {:<24} {:>4} entities".format(role, count))

        groups = Counter()
        for entity in model.entities:
            if entity.raw:
                continue
            groups.update(
                prop.group or "ungrouped" for prop in entity.properties
            )
        if groups:
            lines.append("")
            lines.append("Quantity families")
            ordered_groups = sorted(groups.items())
            for group, count in ordered_groups[:24]:
                lines.append("  {:<36} {:>5} signals".format(group, count))
            if len(ordered_groups) > 24:
                guidance = (
                    "use --group GLOB"
                    if include_cli_guidance
                    else "refine with the discovery command below"
                )
                lines.append(
                    "  … {} more families; {}".format(
                        len(ordered_groups) - 24,
                        guidance,
                    )
                )

        absent = [
            record for record in model.declared_classes
            if record.live_count == 0
        ]
        if absent:
            lines.append("")
            lines.append("Declared dynamic classes (not live at reset)")
            for record in absent:
                lines.append(
                    "  {}: {}".format(
                        record.entity_class,
                        ", ".join(record.components) or "open schema",
                    )
                )

        preview = first_selector_preview(view)
        if preview.example is not None:
            lines.append("")
            lines.append(
                "Example specification "
                "(production-verified; {} reset-time binding{})".format(
                    preview.binding_count,
                    "" if preview.binding_count == 1 else "s",
                )
            )
            if include_cli_guidance:
                lines.extend(
                    "  " + line for line in preview.example.splitlines()
                )
            else:
                example = yaml.safe_load(preview.example)
                constants = yaml.safe_dump(
                    example["constants"],
                    default_flow_style=True,
                    sort_keys=True,
                ).strip()
                lines.extend((
                    "  Spec name prompt:",
                    "    {}".format(example["name"]),
                    "  Formula prompt:",
                    "    {}".format(example["formula"]),
                    "  Constants prompt:",
                    "    {}".format(constants),
                ))
    else:
        lines.append(
            "Selected: {} of {} entities; {} of {} properties".format(
                view.after.entity_count,
                view.before.entity_count,
                view.after.property_count,
                view.before.property_count,
            )
        )
        if not view.entities:
            lines.append("No matches. Broaden a filter or use --raw.")
        for entity in view.entities:
            identity = entity.label or str(entity.entity_id)
            suffix = "  role={}".format(entity.role) if entity.role else ""
            lines.append("")
            lines.append(
                "{}_{}  {}{}  position={}".format(
                    entity.entity_class,
                    entity.entity_id,
                    identity,
                    suffix,
                    _position(entity.position),
                )
            )
            if not entity.properties:
                lines.append("  no selected properties")
            for prop in entity.properties:
                metadata = "  group={}".format(prop.group or "other")
                if prop.unit:
                    metadata += "  unit={}".format(prop.unit)
                if prop.tags:
                    metadata += "  tags={}".format(",".join(prop.tags))
                value = " = {:+.6g}".format(prop.value) if verbose else ""
                lines.append("  {}{}{}".format(prop.key, value, metadata))

        preview = selector_preview(view)
        lines.append("")
        if preview.selector:
            lines.append(
                "Selector preview: {}  ({} reset-time binding{})".format(
                    preview.selector,
                    preview.binding_count,
                    "" if preview.binding_count == 1 else "s",
                )
            )
            if preview.example:
                lines.append("")
                lines.extend(
                    "  " + line for line in preview.example.splitlines()
                )
        else:
            lines.append("Selector preview: {}".format(preview.explanation))

        forms, refusal = palette_for_view(view)
        lines.append("")
        if refusal is not None:
            lines.append("Ways to select this: {}".format(refusal))
        else:
            lines.append("Ways to select this")
            lines.extend(_palette_lines(forms))
            # `default_form`, not "the first available one": a role
            # scope must land on the row that reproduces the role, and
            # reimplementing that choice here is how the same filter came
            # to produce a different starter than the explorer's. The
            # scope is derived exactly as `palette_for_view` derives it.
            from .discovery_selector import scope_for_view

            scope, _reason = scope_for_view(view)
            starter = next(
                (form for form in [
                    default_form(forms, scope) if scope else None
                ] if form),
                None
            )
            if starter is not None:
                lines.append("")
                lines.append("Starter specification")
                lines.extend(
                    "  " + line
                    for line in starter_yaml_for_form(
                        starter
                    ).rstrip().splitlines()
                )

    if model.warnings:
        lines.append("")
        lines.append("Authoring warnings")
        lines.extend("  ⚠ {}".format(warning) for warning in model.warnings)

    if include_cli_guidance:
        lines.append("")
        lines.extend((
            "Details: --class CLASS --entity GLOB --signal GLOB "
            "--group GLOB --role ROLE --tag TAG",
            "Raw scene: --raw    Values: --verbose    "
            "Machine data: --format json",
            "Author specs with `specola write-specs`; "
            "evaluate with `specola run`.",
        ))
    return "\n".join(lines) + "\n"
