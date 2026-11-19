"""``specola doctor``: which adapters find their simulator packages.

Lists every discoverable adapter with whether the simulator packages it
declares can be located (checked without importing them, so a found
package does not prove the simulator starts), plus the installed extras.
In a terminal it prints the branded panel; otherwise plain text, one
adapter per line, so the output can be read by scripts.
"""

from __future__ import annotations

import argparse
import sys

from . import branding


def plain(env: branding.Environment) -> str:
    lines = [f"specola {env.version}  python {env.python}  "
             f"extras {', '.join(env.extras) or 'none'}", ""]
    width = max([len(a.name) for a in env.adapters] + [7])
    for status in env.adapters:
        lines.append(f"{status.name:<{width}}  {branding.status_label(status)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="specola doctor",
        description="Report which simulator adapters find their simulator "
                    "packages in this environment and which optional extras "
                    "are installed.",
    )
    parser.add_argument("--verify", action="store_true", help="Run bounded functional and rendered-browser installation checks")
    parser.add_argument("--sim", action="append", default=None, metavar="NAME", help="Verify a current-environment simulator (repeatable; default: mock plus available adapters)")
    parser.add_argument("--output-dir", type=str, help="New directory for retained verification logs, screenshots and results.json")
    parser.add_argument("--timeout", type=float, default=180, help="Per-child timeout in seconds (default: 180)")
    parser.add_argument("--browser", help="Chrome or Chromium executable for rendered verification")
    parser.add_argument("--json", action="store_true", help="Print the verification result as JSON")
    parser.add_argument("--demo", nargs="?", const="packaged", help="Verify LIBERO replay from PATH, or the packaged goal-0 demonstration when omitted")
    args = parser.parse_args()
    if not args.verify and (args.sim or args.output_dir or args.browser or args.json or args.demo or args.timeout != 180):
        parser.error("verification options require --verify")
    if args.verify:
        import json
        from .install_verification import verify_installation
        try:
            result = verify_installation(args.output_dir, simulators=args.sim, timeout=args.timeout, browser=args.browser, demo=args.demo)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"Installation verification could not start: {exc}\n")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            for outcome in result["outcomes"]:
                print(f"{outcome['simulator']}/{outcome['capability']}: {outcome['status']} ({outcome['reason']})")
            print("Evidence: " + result["evidence_directory"])
        raise SystemExit(0 if result["ready"] else 1)

    env = branding.environment()
    if branding.enabled():
        from rich.console import Group
        from rich.text import Text

        branding.print_branded(branding.panel(env),
                               narrow=Group(branding.mark_text(), Text(""), Text(plain(env))))
        hints = [s for s in env.adapters if s.missing and s.install_hint]
        if hints:
            print()
            for status in hints:
                print(f"  {status.name}: {status.install_hint}")
    else:
        print(plain(env))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
