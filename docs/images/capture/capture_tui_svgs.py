"""Re-export the two README TUI screenshots as Textual SVGs.

Mirrors specola.discover.main / specola.write_specs.main construction, then
drives the app headlessly with Textual's pilot and calls export_screenshot().

Terminal geometry recovered from the committed SVGs' viewBox (rich SVG export:
width = 12.2*cols + 18, height = 24.4*rows + 50):
  discover-tui.svg    1482 x  928.4  ->  120 x 36
  write-specs-tui.svg 1482 x 1074.8  ->  120 x 42

Usage:
  PYTHONNOUSERSITE=1 python capture_tui_svgs.py --sim robocasa \
      --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml \
      --out-dir docs/images
  ... --sim mock            # sim-free mechanics check
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

DISCOVER_SIZE = (120, 36)
WRITE_SPECS_SIZE = (120, 42)

SIGNAL_PATTERN = "gripper_qpos_1"
ENTITY_CLASS = "gripper"
SPEC_NAME = "gripper_opens"
FORMULA = "once[0,10s]({selector} >= 0.03)"

# Mock fallback so the script's mechanics can be checked without a simulator.
MOCK_SIGNAL_PATTERN = "mass"
MOCK_ENTITY_CLASS = "object"
MOCK_FORMULA = "once[0,10s]({selector} >= 0.03)"


def build_context(sim: str, task: str | None, queries_path: Path | None = None):
    """Return (view, dt, catalog, metadata, entities, labels, capabilities)."""
    from specola.components import ComponentCatalog
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_model import (
        build_discovery_model,
        property_metadata_for,
    )

    if sim == "mock":
        from specola.cli_common import _MockAdapter

        adapter = _MockAdapter()
        entities = adapter.reset().perceived_entities
        dt = adapter.dt
        task_description = None
    else:
        from specola.cli_common import _task_description, load_sim
        from specola.query_config import load_query_configuration

        # MuJoCo-family adapters require a query configuration (the
        # mj_geomDistance cutoff); discover and write-specs take it with
        # --queries, so the capture does too.
        queries = (load_query_configuration(queries_path)
                   if queries_path is not None else None)
        adapter, dt, _dims, entities, _classes = load_sim(
            sim, task=task, robot=None, target_only=True,
            dt_override=None, header="Specola Discovery", queries=queries,
        )
        task_description = _task_description(adapter)

    metadata = property_metadata_for(adapter, entities)
    model = build_discovery_model(
        adapter, entities, simulator=sim, task=task, effective_dt=dt,
        task_description=task_description, property_metadata=metadata,
    )
    view = filter_discovery(model, DiscoveryQuery(include_raw=False))
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    return (
        view, dt, catalog, metadata, entities,
        getattr(adapter, "entity_labels", None),
        getattr(adapter, "capabilities", None),
    )


def _find_signal(app, pattern):
    for signal in app.signal_by_key.values():
        if signal.pattern == pattern:
            return signal
    raise SystemExit(
        "signal {!r} not in scope; have: {}".format(
            pattern,
            sorted({s.pattern for s in app.signal_by_key.values()})[:20],
        )
    )


async def _select_signal(app, pilot, entity_class, pattern):
    """Select the class scope and one of its signals, as a user would."""
    from specola.discovery_selector import DiscoveryScope

    scope = DiscoveryScope(entity_class)
    app._select_scope(scope)
    await pilot.pause()
    node = app._nodes_by_scope.get(scope)
    if node is not None:
        node.expand()
        await pilot.pause()
    app._show_signal(_find_signal(app, pattern))
    await pilot.pause(0.2)
    # The single selector preview was replaced by the palette: the panel
    # now lists every form the selection can express, and the row it
    # lands on is the one the preview used to produce.
    if app.selected_form is None:
        raise SystemExit("selector palette did not resolve")
    return app.selected_form.reference


async def capture_discover(context, out_path, entity_class, pattern):
    from specola.explore import build_explorer

    app = build_explorer(context[0])
    async with app.run_test(size=DISCOVER_SIZE) as pilot:
        await pilot.pause()
        selector = await _select_signal(app, pilot, entity_class, pattern)
        await pilot.pause(0.2)
        out_path.write_text(app.export_screenshot(), encoding="utf-8")
    return selector


async def capture_write_specs(context, out_path, entity_class, pattern,
                              formula_template):
    from textual.widgets import Button
    from specola.write_specs import build_write_specs_app

    view, dt, catalog, metadata, entities, labels, capabilities = context
    app = build_write_specs_app(
        view, dt=dt, catalog=catalog, metadata=metadata, entities=entities,
        labels=labels, capabilities=capabilities,
        suggested_output="robocasa_specs.yaml",
    )
    async with app.run_test(size=WRITE_SPECS_SIZE) as pilot:
        await pilot.pause()
        selector = await _select_signal(app, pilot, entity_class, pattern)

        app.query_one("#spec_name").value = SPEC_NAME
        app.query_one("#formula").text = formula_template.format(
            selector=selector
        )
        app.query_one("#constants").text = "{}"
        await pilot.pause()

        app.query_one("#validate", Button).press()
        await pilot.pause(0.5)

        validation = str(app.query_one("#validation").render())
        if "accepted" not in validation:
            raise SystemExit("draft not accepted:\n" + validation)

        # Scroll the workspace to the draft + validation panels, which is what
        # the README screenshot documents.
        workspace = app.query_one("#writer_workspace")
        workspace.scroll_end(animate=False)
        await pilot.pause(0.4)

        out_path.write_text(app.export_screenshot(), encoding="utf-8")
    return validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="robocasa")
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", choices=("discover", "write-specs"), default=None)
    ap.add_argument("--queries", type=Path, default=None,
                    help="query configuration, required for MuJoCo-family "
                         "simulators (e.g. examples/geometry_queries.yaml)")
    args = ap.parse_args()

    if args.sim == "mock":
        entity_class, pattern = MOCK_ENTITY_CLASS, MOCK_SIGNAL_PATTERN
        formula = MOCK_FORMULA
        args.task = None
    else:
        entity_class, pattern, formula = ENTITY_CLASS, SIGNAL_PATTERN, FORMULA

    context = build_context(args.sim, args.task, args.queries)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "discover"):
        path = args.out_dir / "discover-tui.svg"
        selector = asyncio.run(
            capture_discover(context, path, entity_class, pattern)
        )
        print("wrote {} ({} bytes) selector={}".format(
            path, path.stat().st_size, selector
        ))

    if args.only in (None, "write-specs"):
        path = args.out_dir / "write-specs-tui.svg"
        asyncio.run(
            capture_write_specs(
                context, path, entity_class, pattern, formula
            )
        )
        print("wrote {} ({} bytes)".format(path, path.stat().st_size))


if __name__ == "__main__":
    sys.exit(main())
