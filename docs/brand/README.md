# Brand assets

The Specola icon (an S crossed by a horizontal rule, with an open circle
where the rule meets the spine) appears in the CLI, the terminal UIs, and
the web dashboard.

## Source

Every asset comes from the Newsreader font (SIL Open Font License 1.1) at
`opsz 72`, `wght 600`, the font the project page uses for its name. The
icon is the font's S, 80 units tall in a 100-unit box, crossed by a rule
from x 2 to 98, 3.5 units thick, with an open circle of radius 6.5 at the
centre. The name has its tracking tightened by 0.02 em, as on the page.

## Regenerate

```bash
pip install pillow numpy uharfbuzz fonttools
python scripts/make_brand_assets.py
```

The first run downloads Newsreader from a pinned google/fonts commit into
`~/.cache/specola/`. The script writes:

- `docs/brand/logo-{light,dark}.svg`: the icon stacked over the name,
  outlined, in brand navy and in light navy, for the README heading.
- `src/specola/_brand_art.py`: the S as terminal block characters at two
  sizes, with the rule and circle drawn as a line of text (block
  characters cannot hold the rule at terminal resolution).
- `src/specola/dashboard/static/brand/icon-{16,32,44,64}.png`: the icon in
  the dashboard's light navy on a transparent background. The dashboard
  tab bar shows `icon-44.png` at 22 CSS pixels. The rule and circle are
  drawn as shapes at each size: a 2 px stroke at 32 and 44 px, and at
  16 px a 1 px rule with the circle left out, as the design brief allows.
- `src/specola/dashboard/static/brand/favicon-{16,32}.png`: the same icon
  on a brand navy tile, so the browser tab icon reads on light and dark
  tab bars alike.

Specola itself imports none of these packages at run time for branding.

The S in these assets is about 83% of the icon's height, because the rule
overhangs it on both sides. An S cropped from the brand-direction image
(`specola-paper/planning/project-page/reference/brand-direction.png`) filled
the full height until 2026-10-16; it was replaced so the tool and the README
use the same S.

## Colours

| Use | Colour |
|---|---|
| Light backgrounds | `#0F2747` (brand navy) |
| Dark backgrounds | `#A9C1E8` (light navy) |
| Unknown background | the terminal's own foreground colour |

A mid navy can just clear a 4.5:1 contrast ratio on pure black and pure
white (`#5277B1` measures 4.63 and 4.53), but it drops to 3.1 to 3.6 on
common dark-grey terminal backgrounds such as `#282C34` and `#1E1F20`, so
the terminal mark is coloured only when the background is known. Set
`SPECOLA_THEME=light` or `SPECOLA_THEME=dark` to choose; otherwise the
terminal's `COLORFGBG` hint is used when its background is one of the 16
standard colours. Status text in `specola doctor` uses the terminal's
foreground colour, with bold for adapters whose packages were found.

## Where branding appears

- The README heading shows the logo centred, in brand navy or light navy
  to match the reader's GitHub theme.
- `specola`, `specola --version`, `specola doctor`, and the startup line
  of the commands that build a simulator print the mark only to an
  interactive UTF-8 terminal, and never when `NO_COLOR` is set or `TERM`
  is `dumb`. Piped output stays plain. A terminal too narrow for the icon gets the
  one-line mark and plain rows.
- The live run panel (`specola run --ui rich`) puts the mark in its title
  under the same rule.
- The Textual apps (`discover`, `write-specs`, the run dashboard) show the
  one-line mark `─S─` in their header whenever they run. Textual removes
  colour under `NO_COLOR` itself; the characters stay.

The block art assumes the terminal renders East Asian ambiguous-width
characters (`─`, `○`, `█`) one cell wide, the default outside CJK
locales. If your terminal renders them two cells wide, or its font lacks
the quadrant block characters, set `NO_COLOR`. Below about 43 columns
the one-line fallback wraps onto a second line.

## Deliberate departure from the design brief

The design brief
(`specola-paper/planning/project-page/2026-09-25-design-brief.md`) sets
the name, icon, colours and tagline. The dashboard tab bar sets the name
in the dashboard's sans-serif at 12.5 px rather than the brief's display
serif, which does not hold up at that size.
