"""Serve a finished run recording in the browser (no simulator).

"view", not "replay": --mode replay already means re-driving
demonstration states through the simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="specola view",
        description="Serve a finished --web run recording in the browser.",
    )
    ap.add_argument(
        "recording", type=Path,
        help="Recording directory (results/<run>/recording_000)",
    )
    ap.add_argument(
        "--port", type=int, default=8090, metavar="PORT",
        help="Port on 127.0.0.1 (default: 8090; 0 picks a free port)",
    )
    args = ap.parse_args()
    if not (0 <= args.port <= 65535):
        ap.error("--port must be 0-65535")
    if not (args.recording / "manifest.json").is_file():
        ap.error("not a recording directory: {}".format(args.recording))

    try:
        from .dashboard.server import DashboardServer, wait_for_interrupt
    except ImportError as exc:
        raise SystemExit(
            "specola view needs the 'web' extra "
            "(pip install 'specola[web]'): {}".format(exc)
        )

    server = DashboardServer(args.recording, mode="view")
    url = server.start(args.port)
    print("Serving {} at {} (Ctrl-C to exit)".format(args.recording, url),
          flush=True)
    wait_for_interrupt()
