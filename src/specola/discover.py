"""Read-only discovery CLI over one immutable reset snapshot."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import textwrap
from pathlib import Path

from .pager import maybe_page


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specola discover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Explore the entities, task roles, and scalar signals exposed "
            "to the STL monitor."
        ),
        epilog=textwrap.dedent(
            """
            specola discover --sim mock
            specola discover --sim libero --task libero_object:0 --queries examples/geometry_queries.yaml --class object
            specola discover --sim libero --task libero_object:0 \
              --queries examples/geometry_queries.yaml --class robot --signal 'qfrc_passive_*'
            specola discover --sim mock --format json
            """
        ),
    )
    parser.add_argument("--sim", default="mock")
    parser.add_argument("--audit-surface", action="store_true",
                        help="Report extraction declarations and missing evidence without constructing a simulator")
    parser.add_argument("--queries", type=Path, default=None, metavar="PATH",
                        help="Native query configuration YAML or JSON, separate from scope")
    parser.add_argument(
        "--adapter-module", action="append", default=[], metavar="MODULE",
        help="Import MODULE so its adapter registration runs (repeatable)",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument(
        "--all-regions", action="store_true",
        help="Include regions from every fixture, not only task targets",
    )
    parser.add_argument(
        "--class", dest="classes", action="append", default=[],
        metavar="GLOB", help="Select entity classes (repeatable)",
    )
    parser.add_argument(
        "--entity", dest="entities", action="append", default=[],
        metavar="GLOB", help="Select entity IDs or labels (repeatable)",
    )
    parser.add_argument(
        "--signal", dest="signals", action="append", default=[],
        metavar="GLOB", help="Select scalar property keys (repeatable)",
    )
    parser.add_argument(
        "--group", dest="groups", action="append", default=[],
        metavar="GLOB", help="Select declared property groups (repeatable)",
    )
    parser.add_argument(
        "--role", dest="roles", action="append", default=[],
        metavar="GLOB", help="Select declared task roles (repeatable)",
    )
    parser.add_argument(
        "--tag", dest="tags", action="append", default=[],
        metavar="GLOB", help="Select structural property tags (repeatable)",
    )
    parser.add_argument(
        "--search", dest="search", action="append", default=[],
        metavar="TEXT", help="Search labels and structured metadata",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Include raw scene bodies, geoms, sites, cameras, and lights",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show current values in an explicitly selected detail scope",
    )
    parser.add_argument(
        "--field-contract", action="store_true",
        help="Show the complete MuJoCo reader/exclusion audit",
    )
    parser.add_argument(
        "--format", choices=("table", "json"), default="table",
        help="Output presentation (default: table)",
    )
    parser.add_argument(
        "--ui", choices=("textual", "rich", "plain"), default=None,
        help="Human UI; defaults to Textual on a TTY and plain otherwise",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write the selected output to PATH instead of stdout",
    )
    parser.add_argument(
        "--web", nargs="?", const=8090, type=int, default=None,
        metavar="PORT",
        help="Browse this scene in a browser on 127.0.0.1 (default port "
             "8090, 0 picks a free port; needs the 'web' extra). The "
             "URL is announced on stderr so --format json's single "
             "document keeps stdout to itself. Serves until Ctrl-C.",
    )
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--no-pager", action="store_true",
        help="Print directly instead of paging long table output",
    )
    return parser


def _is_tty() -> bool:
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def _serve_discovery(args, model) -> None:
    """Serve one scene in the browser, with no run behind it.

    A scene with no episode still has everything Discover needs, so this
    reuses the dashboard rather than growing a second server. What it
    lacks is a recording, so the snapshot is handed over in memory and
    the directory it names holds only what authoring writes.

    The URL goes to stderr. ``--format json`` writes one document to
    stdout by contract, and announcing a URL there would corrupt it for
    every consumer that pipes it.
    """
    import sys
    import threading

    try:
        from .dashboard.server import DashboardServer
    except ImportError as exc:
        raise SystemExit(
            "specola discover --web needs the 'web' extra "
            "(pip install 'specola[web]'): {}".format(exc)
        )

    from .capabilities import adapter_capabilities
    from .dashboard.recording import RecordingWriter
    from .discovery_serialize import discovery_json
    from .discovery_filter import DiscoveryQuery, filter_discovery
    import gzip

    # A synthetic identity, because there is no run to borrow one from.
    # It names the scene rather than an episode, and gives authoring a
    # directory of its own instead of writing beside somebody's results.
    from datetime import datetime

    session = "discover_{}_{}".format(
        args.sim,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    directory = args.output_dir / session if getattr(
        args, "output_dir", None
    ) else Path("results") / session
    recording_dir = directory / "recording_000"
    recording_dir.mkdir(parents=True, exist_ok=True)
    RecordingWriter(recording_dir, {
        "run_id": session,
        "simulator": args.sim,
        "task": args.task,
        "mode": "discover",
        "cameras": [],
        "status": "discover-only",
    })

    payload = gzip.compress(discovery_json(
        filter_discovery(model, DiscoveryQuery(include_raw=True))
    ).encode("utf-8"), 6)

    server = DashboardServer(recording_dir, mode="view")
    server.set_discovery_snapshot(payload)

    # Everything from here goes to stderr. Flask and werkzeug print
    # their startup banners on stdout from the serving thread, which
    # would append to a --format json document that is already complete
    # and turn one valid document into two things. Redirecting for the
    # whole serving lifetime catches the thread's later prints too,
    # which redirecting only across start() would not.
    with contextlib.redirect_stdout(sys.stderr):
        url = server.start(args.web)
        print("Discover: {} (Ctrl-C to exit)".format(url), file=sys.stderr)
        print("Scene: {}".format(recording_dir), file=sys.stderr)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


def main() -> None:
    from .conformance.verification import require_ordinary_execution
    require_ordinary_execution()
    args = _parser().parse_args()
    if args.audit_surface:
        if (args.web is not None or args.ui is not None or args.raw or args.verbose or args.field_contract
                or any((args.classes, args.entities, args.signals, args.groups, args.roles, args.tags, args.search))):
            raise SystemExit("--audit-surface cannot publish ordinary UI/web discovery or apply observation-value filters")
        from .conformance.audit import adapter_surface_audit, render_surface_audit
        document = adapter_surface_audit(args.sim, task=args.task, robot=args.robot,
                                         target_only=not args.all_regions, dt_override=args.dt,
                                         requested_adapter_modules=args.adapter_module)
        if args.queries is not None:
            from .query_config import load_query_configuration
            document["requested_query_configuration"] = load_query_configuration(args.queries).document()
            document["query_contract_status"] = "native_bindings_not_loaded"
        text = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
                if args.format == "json" else render_surface_audit(document))
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        return
    if args.format == "json" and args.ui is not None:
        raise SystemExit("--format json cannot be combined with --ui")
    if args.web is not None:
        if args.ui == "textual":
            raise SystemExit(
                "--web cannot be combined with --ui textual; the browser "
                "and a full-screen terminal app cannot both own the "
                "terminal. Use --web alone, or --ui plain beside it."
            )
        if not (0 <= args.web <= 65535):
            raise SystemExit("--web PORT must be 0-65535")
    if args.output is not None and args.ui in ("textual", "rich"):
        raise SystemExit("--output is supported by plain and JSON output")
    if args.ui == "textual" and not _is_tty():
        raise SystemExit(
            "--ui textual requires interactive stdin and stdout; "
            "use --ui plain or --ui rich for redirected output"
        )

    filters = any((
        args.classes, args.entities, args.signals, args.groups,
        args.roles, args.tags, args.search,
    ))
    if args.verbose and not (filters or args.raw):
        raise SystemExit(
            "--verbose requires a detail filter or --raw; "
            "this prevents an accidental full-scene dump"
        )

    # Structured stdout must remain a single JSON document. Loading progress
    # and simulator diagnostics are therefore redirected as one operation.
    output_stream = sys.stderr if args.format == "json" else sys.stdout
    authoring_document = None
    loaded = None
    task_description = ""
    with contextlib.redirect_stdout(output_stream):
        from .registry import load_adapter_module
        for module_name in args.adapter_module:
            load_adapter_module(module_name)

        from .cli_common import _task_description, load_sim
        from .query_config import load_query_configuration
        queries = load_query_configuration(args.queries) if args.queries is not None else None
        from .conformance.provider import ExtractionGuardError
        from .conformance.authoring import SchemaOnlyDiscovery, render_schema_authoring
        from .run_failures import report_extraction_failure
        try:
            loaded = load_sim(
                args.sim,
                task=args.task,
                robot=args.robot,
                target_only=not args.all_regions,
                dt_override=args.dt,
                header="Specola Discovery", queries=queries, allow_schema_only=True,
            )
        except SchemaOnlyDiscovery as exc:
            authoring_document = exc.document
        except ExtractionGuardError as exc:
            report_extraction_failure(exc, command="discover", adapter=args.sim, task=args.task)
            raise SystemExit(1) from None
        else:
            task_description = _task_description(loaded[0])

    if authoring_document is not None:
        if (args.web is not None or args.ui is not None or args.raw or args.verbose
                or args.field_contract or filters):
            raise SystemExit("schema-only discovery cannot publish ordinary UI/web results or apply observation-value filters")
        text = (json.dumps(authoring_document, sort_keys=True, indent=2, allow_nan=False) + "\n"
                if args.format == "json" else render_schema_authoring(authoring_document))
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        return

    assert loaded is not None
    adapter, effective_dt, _dims, entities, _classes = loaded
    from .discovery_filter import DiscoveryQuery, filter_discovery
    from .discovery_model import build_discovery_model

    model = build_discovery_model(
        adapter,
        entities,
        simulator=args.sim,
        task=args.task,
        effective_dt=effective_dt,
        task_description=task_description,
        extraction=adapter.last_observation.extraction if adapter.last_observation is not None else None,
    )
    include_raw = args.raw or filters or args.format == "json"
    query = DiscoveryQuery(
        classes=tuple(args.classes),
        entities=tuple(args.entities),
        signals=tuple(args.signals),
        groups=tuple(args.groups),
        roles=tuple(args.roles),
        tags=tuple(args.tags),
        search=tuple(args.search),
        include_raw=include_raw,
    )
    try:
        view = filter_discovery(model, query)
    except ValueError as exc:
        raise SystemExit("invalid discovery filter: {}".format(exc))

    if args.format == "json":
        from .discovery_serialize import discovery_json
        text = discovery_json(
            view,
            include_field_contract=args.field_contract,
        )
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        # The document is complete on stdout before anything is served,
        # so `--format json --web` honours both contracts: a consumer
        # piping stdout gets exactly one document, and the browser still
        # opens. The URL was never going to stdout anyway.
        if args.web is not None:
            _serve_discovery(args, model)
        return

    if args.web is not None:
        _serve_discovery(args, model)
        return

    ui = args.ui or ("textual" if _is_tty() else "plain")
    if ui == "textual":
        from .explore import run_explorer
        run_explorer(view)
        return
    if ui == "rich":
        from .discovery_live import render_rich_discovery
        render_rich_discovery(view)
        return

    from .discovery_render import render_discovery
    detailed = filters or args.raw
    text = render_discovery(view, detailed=detailed, verbose=args.verbose)
    if args.field_contract:
        from .cli_common import _print_mujoco_field_contract

        contract_stream = io.StringIO()
        with contextlib.redirect_stdout(contract_stream):
            _print_mujoco_field_contract(adapter, full=True)
        text += contract_stream.getvalue()
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        return

    with maybe_page(disable=args.no_pager):
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
