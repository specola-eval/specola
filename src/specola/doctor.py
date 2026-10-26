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
    parser.parse_args()

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
