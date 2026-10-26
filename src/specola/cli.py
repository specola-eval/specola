"""
cli.py — the ``specola`` command, dispatching to one subcommand module each.

Every subcommand module (``run_monitor``, ``discover``, ``check_specs``,
``write_specs``, ``report``) owns a zero-argument ``main()`` that builds its
own ``argparse`` parser and reads ``sys.argv``. This dispatcher does not
duplicate or wrap those parsers: it strips the subcommand token, rewrites
``sys.argv[0]`` so each parser's usage line reads ``specola <subcommand>``,
and calls that module's ``main()`` unchanged. Adding a flag to a subcommand
therefore needs no edit here.

Subcommand modules are imported lazily, one per invocation. ``run_monitor``
pulls in the live display and results-writing stack, which ``specola check``
has no use for; importing all five eagerly would make every subcommand pay
for the heaviest one.

When a subcommand leaves an Isaac Sim (Kit) app running in the process,
``main()`` also owns the exit status, because Kit's shutdown would end the
process with status 0 whatever the subcommand raised
(``_end_kit_process``).
"""

from __future__ import annotations

import os
import sys

from . import __version__

# Subcommand name -> (module attribute path relative to this package, one-line
# help). Order is the order shown in usage text: the two commands that answer
# "what can I even monitor here?" come before the ones that monitor it.
_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "discover": ("discover", "report the entities and signals a simulator exposes"),
    "check": ("check_specs", "statically check spec templates against a simulator"),
    "run": ("run_monitor", "monitor a simulator episode and stream robustness"),
    "write-specs": ("write_specs", "generate a starter spec file for a simulator"),
    "report": ("report", "summarize saved results from a previous run"),
    "view": ("run_view", "serve a finished --web run recording in the browser"),
    "doctor": ("doctor", "report which adapters find their simulator packages"),
}


def _usage() -> str:
    width = max(len(name) for name in _SUBCOMMANDS)
    lines = [
        "usage: specola <command> [options]",
        "",
        "STL monitoring for simulators.",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in _SUBCOMMANDS.items()]
    lines += [
        "",
        "Run 'specola <command> --help' for the options of one command.",
    ]
    return "\n".join(lines)


def _kit_app_running() -> bool:
    """Whether this process hosts a running Omniverse Kit app (Isaac Sim)."""
    # A Kit app launched in this process imported omni.kit.app first, so a
    # module that was never imported means no app. Importing it here only to
    # ask would load Kit's bindings into every exiting command.
    kit_app = sys.modules.get("omni.kit.app")
    if kit_app is None:
        return False
    try:
        app = kit_app.get_app()
        return app is not None and app.is_running()
    except Exception:  # noqa: BLE001 - a broken binding means no usable app
        return False


def _end_kit_process(code: int) -> None:
    """
    Flush the standard streams and end a failing process before Kit can.

    The Isaac Lab and RoboDojo factories register ``SimulationApp.close``
    at exit. Under Isaac Lab's ``fastShutdown = true`` that close ends the
    process through Kit's quick-terminate path with status 0, so a
    ``SystemExit(1)`` already in flight ends as a success; on Isaac Sim 5.1
    it also drops whatever Python still buffers for stdout. A zero status
    is left to that close, which tears Kit down cleanly. A nonzero one ends
    here with ``os._exit``, skipping the remaining exit hooks: Isaac Sim
    6's own ``SimulationApp.close(exit_code=...)`` does the same for a
    nonzero code, and Isaac Sim 5.1's ``close()`` takes no exit code.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (ValueError, OSError, AttributeError):
            pass
    if code:
        os._exit(code)


def main() -> None:
    try:
        _dispatch()
    except SystemExit as exc:
        if _kit_app_running():
            code = exc.code
            if code is not None and not isinstance(code, int):
                # The interpreter prints a non-integer code and exits 1;
                # os._exit skips that, so do it here.
                print(code, file=sys.stderr)
                code = 1
            _end_kit_process(code or 0)
        raise
    except BaseException as exc:
        if _kit_app_running():
            sys.excepthook(type(exc), exc, exc.__traceback__)
            _end_kit_process(130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise
    if _kit_app_running():
        _end_kit_process(0)


def _dispatch() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        # The branded summary appears only for bare `specola` in a
        # terminal; --help and piped output print the usage alone.
        if not argv:
            from . import branding

            if branding.enabled():
                # Bundled adapters only: the help screen must not run
                # third-party import code. `specola doctor` checks them all.
                branding.print_branded(
                    branding.lockup(branding.environment(load_third_party=False)),
                    narrow=branding.version_line())
                print()
        print(_usage())
        return

    if argv[0] in ("-V", "--version"):
        from . import branding

        if branding.enabled():
            branding.print_branded(branding.version_line())
        else:
            print(f"specola {__version__}")
        return

    name = argv[0]
    entry = _SUBCOMMANDS.get(name)
    if entry is None:
        print(f"specola: unknown command {name!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    module_name, _ = entry

    # Each subcommand parser derives its own prog from sys.argv[0] unless it
    # sets prog= explicitly (they all do, to "specola <name>"). Rewriting
    # argv[0] keeps the two consistent and makes error text from argparse
    # point at the real invocation rather than at the dispatcher.
    sys.argv = [f"specola {name}", *argv[1:]]

    from importlib import import_module

    module = import_module(f".{module_name}", package=__package__)
    module.main()


if __name__ == "__main__":
    main()
