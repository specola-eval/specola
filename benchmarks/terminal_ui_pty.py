"""Measure one terminal command through a real pseudo-terminal.

This harness records transport-level evidence. Newline bytes are not terminal
rows after cursor movement, and operating-system read chunks are not paints.
The output therefore does not claim screen-aware input-to-visible latency or
rows-per-paint.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pty
import selectors
import struct
import subprocess
import termios
import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _process_sample(pid: int) -> Optional[Tuple[float, int]]:
    """Return process CPU seconds and resident bytes from Linux procfs."""
    try:
        stat = Path("/proc/{}/stat".format(pid)).read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat.rsplit(") ", 1)[1].split()
    ticks = os.sysconf("SC_CLK_TCK")
    pages = os.sysconf("SC_PAGE_SIZE")
    cpu_seconds = (int(fields[11]) + int(fields[12])) / ticks
    resident_bytes = int(fields[21]) * pages
    return cpu_seconds, resident_bytes


def measure(
    command: Sequence[str],
    *,
    cwd: Path,
    rows: int,
    columns: int,
    timeout: float,
    environment: Mapping[str, str],
) -> Tuple[dict, bytes]:
    """Run ``command`` under a sized PTY and return metrics plus transcript."""
    controller, follower = pty.openpty()
    fcntl.ioctl(
        follower,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", rows, columns, 0, 0),
    )
    child_environment = dict(environment)
    child_environment.update({
        "TERM": "xterm-256color",
        "LINES": str(rows),
        "COLUMNS": str(columns),
    })
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=child_environment,
        stdin=follower,
        stdout=follower,
        stderr=follower,
        close_fds=True,
    )
    os.close(follower)
    selector = selectors.DefaultSelector()
    selector.register(controller, selectors.EVENT_READ)
    chunks: List[Tuple[float, bytes]] = []
    peak_rss_bytes = 0
    final_cpu_seconds = 0.0
    timed_out = False
    terminate_started: Optional[float] = None
    exit_observed_at: Optional[float] = None
    stream_open = True

    try:
        while True:
            now = time.monotonic()
            sample = _process_sample(process.pid)
            if sample is not None:
                final_cpu_seconds = max(final_cpu_seconds, sample[0])
                peak_rss_bytes = max(peak_rss_bytes, sample[1])
            if (
                now - started > timeout
                and process.poll() is None
                and terminate_started is None
            ):
                timed_out = True
                process.terminate()
                terminate_started = now
            elif (
                terminate_started is not None
                and now - terminate_started > 2.0
                and process.poll() is None
            ):
                process.kill()
            events = selector.select(timeout=0.02)
            for _key, _mask in events:
                try:
                    chunk = os.read(controller, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    chunks.append((time.monotonic() - started, chunk))
                elif stream_open:
                    selector.unregister(controller)
                    stream_open = False
            if process.poll() is not None:
                if exit_observed_at is None:
                    exit_observed_at = now
                if (
                    not stream_open
                    or now - exit_observed_at >= 0.2
                ):
                    if stream_open:
                        selector.unregister(controller)
                        stream_open = False
                    break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    finally:
        selector.close()
        os.close(controller)

    elapsed = time.monotonic() - started
    transcript = b"".join(chunk for _timestamp, chunk in chunks)
    chunk_sizes = [float(len(chunk)) for _timestamp, chunk in chunks]
    metrics = {
        "command": list(command),
        "cwd": str(cwd),
        "terminal": {"rows": rows, "columns": columns},
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "first_output_ms": chunks[0][0] * 1000 if chunks else None,
        "last_output_ms": chunks[-1][0] * 1000 if chunks else None,
        "terminal_bytes": len(transcript),
        "newline_bytes": transcript.count(b"\n"),
        "pty_read_chunks": len(chunks),
        "pty_read_chunk_bytes": {
            "p50": _percentile(chunk_sizes, 0.50),
            "p95": _percentile(chunk_sizes, 0.95),
            "p99": _percentile(chunk_sizes, 0.99),
            "maximum": max(chunk_sizes) if chunk_sizes else None,
        },
        "child_cpu_seconds": final_cpu_seconds,
        "child_cpu_percent_of_one_core": (
            100.0 * final_cpu_seconds / elapsed if elapsed else None
        ),
        "peak_rss_bytes": peak_rss_bytes,
        "transcript_sha256": hashlib.sha256(transcript).hexdigest(),
        "measurement_scope": {
            "terminal_bytes": "PTY transport bytes",
            "newline_bytes": (
                "literal newline bytes; not displayed-row or paint count"
            ),
            "pty_read_chunks": (
                "operating-system reads; not presentation frames"
            ),
            "not_measured": [
                "screen-aware terminal rows per paint",
                "input-to-visible latency",
                "source-to-visible latency",
                "descendant-process CPU and RSS",
            ],
        },
    }
    return metrics, transcript


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure terminal output and process cost through a PTY."
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--columns", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label")
    parser.add_argument("--source-events", type=int)
    parser.add_argument("--requested-refresh-hz", type=float)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("command required after --")
    if args.rows <= 0 or args.columns <= 0:
        raise SystemExit("terminal rows and columns must be positive")
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")

    metrics, transcript = measure(
        command,
        cwd=args.cwd.resolve(),
        rows=args.rows,
        columns=args.columns,
        timeout=args.timeout,
        environment=os.environ,
    )
    metrics["label"] = args.label
    metrics["declared_denominator"] = {
        "source_events": args.source_events,
        "requested_refresh_hz": args.requested_refresh_hz,
        "notes": args.note,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    args.transcript.write_bytes(transcript)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
