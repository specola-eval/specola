#!/usr/bin/env python3
"""Generate every brand asset from the Newsreader font.

The icon is Newsreader's S (SIL Open Font License 1.1, opsz 72, wght 600)
crossed by a horizontal rule, with an open circle where the rule meets the
spine, as on the project page. In a 100-unit box the S is 80 units tall and
centred, the rule runs from x 2 to 98 at y 50, 3.5 units thick, and the
circle has radius 6.5. The script writes:

- ``docs/brand/logo-{light,dark}.svg``: the icon stacked over the
  "Specola" wordmark (-0.02 em tracking), outlined, in brand navy and in
  light navy, for the README heading.
- ``src/specola/_brand_art.py``: the icon as quadrant block characters
  at 8 and 10 rows, split at its middle by a text rule ``───○───``. Block
  characters cannot hold the rule and circle at terminal resolution, so
  only the S is rasterized and the rule is a line of text.
- ``src/specola/dashboard/static/brand/icon-{16,32,44,64}.png``: the icon
  in the dashboard's light navy on a transparent background. The rule and
  circle are drawn as crisp shapes at each size (the brand brief asks for
  dedicated small cuts, because downscaling erases them).
- ``src/specola/dashboard/static/brand/favicon-{16,32}.png``: the same
  light navy icon on a brand navy tile, for browser tabs.

The font is downloaded once, from a pinned google/fonts commit, into
``~/.cache/specola/``. Requires Pillow, NumPy, uharfbuzz (for kerning) and
fontTools. Run from the repository root:

    pip install pillow numpy uharfbuzz fonttools
    python scripts/make_brand_assets.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "docs" / "brand"
ART = ROOT / "src" / "specola" / "_brand_art.py"
STATIC = ROOT / "src" / "specola" / "dashboard" / "static" / "brand"
FONT_URL = ("https://raw.githubusercontent.com/google/fonts/"
            "991ce1de6075188e6b8977a5aa9fcd3610a4e946/ofl/newsreader/Newsreader%5Bopsz,wght%5D.ttf")
FONT = Path.home() / ".cache" / "specola" / "Newsreader[opsz,wght].ttf"
AXES = {"wght": 600, "opsz": 72}
TRACKING = -0.02          # em, as the project page's wordmark

# Quadrant characters keyed by (top-left, top-right, bottom-left, bottom-right).
QUAD = {
    (0, 0, 0, 0): " ", (1, 0, 0, 0): "▘", (0, 1, 0, 0): "▝", (0, 0, 1, 0): "▖",
    (0, 0, 0, 1): "▗", (1, 1, 0, 0): "▀", (0, 0, 1, 1): "▄", (1, 0, 1, 0): "▌",
    (0, 1, 0, 1): "▐", (1, 0, 0, 1): "▚", (0, 1, 1, 0): "▞", (1, 1, 1, 0): "▛",
    (1, 1, 0, 1): "▜", (1, 0, 1, 1): "▙", (0, 1, 1, 1): "▟", (1, 1, 1, 1): "█",
}
THRESHOLD = 0.35          # ink coverage above which a quadrant is filled
CELL_ASPECT = 2.0         # terminal cell height / width
DASHBOARD_NAVY = (0xA9, 0xC1, 0xE8)
BRAND_NAVY = (0x0F, 0x27, 0x47)
COLOURS = {"light": "#0F2747", "dark": "#A9C1E8"}
SMALL = 44                # sizes at or below this get the small-cut strokes
FRAME = (96, 80)          # the icon's ink, x 2-98 by y 10-90 of the 100-unit box
PX = 6                    # raster pixels per icon unit


def load_font() -> hb.Font:
    if not FONT.exists():
        FONT.parent.mkdir(parents=True, exist_ok=True)
        part = FONT.with_suffix(".part")
        urllib.request.urlretrieve(FONT_URL, part)
        part.replace(FONT)  # an interrupted download never poses as the font
    font = hb.Font(hb.Face(hb.Blob.from_file_path(str(FONT))))
    font.set_variations(AXES)
    return font


def outline(font: hb.Font, text: str) -> tuple[str, tuple[float, float, float, float]]:
    """Shaped ``text`` as one SVG path in font units with y down, and its ink bounds."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    pen, bounds = SVGPathPen(None), BoundsPen(None)
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        t = (1, 0, 0, -1, x + pos.x_offset, -pos.y_offset)
        font.draw_glyph_with_pen(info.codepoint, TransformPen(pen, t))
        font.draw_glyph_with_pen(info.codepoint, TransformPen(bounds, t))
        x += pos.x_advance + TRACKING * font.face.upem
    return pen.getCommands(), bounds.bounds


def icon_svg(font: hb.Font, colour: str, theme: str) -> str:
    """The icon in its 100-unit box."""
    d, (x0, y0, x1, y1) = outline(font, "S")
    s = 80 / (y1 - y0)
    tx, ty = 50 - s * (x0 + x1) / 2, 10 - s * y0
    # A mask keeps the circle's centre transparent on any background.
    return (f'<mask id="hole-{theme}"><rect x="-10" y="-10" width="120" height="120" fill="#fff"/>'
            f'<circle cx="50" cy="50" r="6.5"/></mask>'
            f'<g mask="url(#hole-{theme})" fill="{colour}">'
            f'<path transform="matrix({s:.6f} 0 0 {s:.6f} {tx:.3f} {ty:.3f})" d="{d}"/>'
            f'<line x1="2" x2="98" y1="50" y2="50" stroke="{colour}" stroke-width="3.5"/></g>'
            f'<circle cx="50" cy="50" r="6.5" fill="none" stroke="{colour}" stroke-width="3.5"/>')


def write_logo(font: hb.Font) -> None:
    """The icon, 1.6 em, centred 0.12 em above the wordmark's ink."""
    em = font.face.upem
    d, (x0, y0, x1, y1) = outline(font, "Specola")
    box, pad = 1.6 * em, 10
    ix, iy = (x0 + x1) / 2 - box / 2, y0 - 0.12 * em - box
    left, top = x0 - pad, iy - pad
    width, height = x1 - x0 + 2 * pad, y1 - iy + 2 * pad
    for theme, colour in COLOURS.items():
        (BRAND / f"logo-{theme}.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.1f} {top:.1f} {width:.1f} {height:.1f}" '
            f'role="img" aria-label="Specola">\n'
            f"<!-- Generated by scripts/make_brand_assets.py. Outlines from Newsreader "
            f"(SIL Open Font License 1.1), opsz 72, wght 600. -->\n"
            f'<g transform="translate({ix:.1f} {iy:.1f}) scale({box / 100:.3f})">{icon_svg(font, colour, theme)}</g>\n'
            f'<path fill="{colour}" d="{d}"/>\n</svg>\n')


def s_image() -> Image.Image:
    """The icon's S alone, black on white, framed by the icon's ink (FRAME)."""
    upem = TTFont(FONT)["head"].unitsPerEm
    _, (_, y0, _, y1) = outline(load_font(), "S")
    pil = ImageFont.truetype(str(FONT), size=round(80 * PX * upem / (y1 - y0)))
    pil.set_variation_by_axes([AXES[a.axisTag] for a in TTFont(FONT)["fvar"].axes])
    width, height = FRAME[0] * PX, FRAME[1] * PX
    left, top, right, _ = pil.getbbox("S")
    image = Image.new("L", (width, height), 255)
    ImageDraw.Draw(image).text(((width - (right - left)) / 2 - left, -top), "S", font=pil, fill=0)
    return image


def block_rows(image: Image.Image, rows: int) -> list[str]:
    width, height = image.size
    cols = max(1, round(width / height * rows * CELL_ASPECT))
    grey = image.resize((cols * 2, rows * 2), Image.BOX)
    filled = (1.0 - np.asarray(grey, dtype=float) / 255.0) > THRESHOLD
    lines = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            key = (int(filled[2 * r, 2 * c]), int(filled[2 * r, 2 * c + 1]),
                   int(filled[2 * r + 1, 2 * c]), int(filled[2 * r + 1, 2 * c + 1]))
            line += QUAD[key]
        lines.append(line)  # full width: the rule overhangs the S on both sides
    return lines


def icon_art(image: Image.Image, rows: int) -> list[str]:
    """Block art with the rule row inserted at the middle."""
    cells = block_rows(image, rows)
    width = len(cells[0])
    centre = width // 2
    rule = "─" * centre + "○" + "─" * (width - centre - 1)
    middle = rows // 2
    return cells[:middle] + [rule] + cells[middle:]


def write_art(image: Image.Image) -> None:
    body = [
        '"""Terminal block art for the Specola icon.',
        "",
        "Generated by scripts/make_brand_assets.py from the Newsreader font;",
        "do not edit by hand.",
        '"""',
        "",
    ]
    for rows in (8, 10):
        body.append(f"ICON_{rows} = (")
        body += [f"    {line!r}," for line in icon_art(image, rows)]
        body.append(")")
        body.append("")
    ART.write_text("\n".join(body))


def render(image: Image.Image, size: int, colour, tile=None) -> Image.Image:
    """The icon at ``size`` px in ``colour``, on transparency or a rounded ``tile``."""
    alpha = Image.eval(image, lambda v: 255 - v)
    inner = size if tile is None else round(size * 0.75)
    unit = inner / max(FRAME)                     # canvas pixels per icon unit
    resized = alpha.resize((round(FRAME[0] * unit), round(FRAME[1] * unit)), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if tile is not None:
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=max(2, size // 5),
                               fill=tile + (255,))
    glyph = Image.new("RGBA", resized.size, colour + (255,))
    glyph.putalpha(resized)
    left, top = (size - resized.width) // 2, (size - resized.height) // 2
    canvas.alpha_composite(glyph, (left, top))
    # 44 px is shown at 22 CSS px and 32 px is the 2x favicon, so both need a
    # 2 px stroke to show a 1 CSS px line; larger sizes follow the vector.
    stroke = 1 if size <= 16 else 2 if size <= SMALL else round(3.5 * unit)
    y = top + round(FRAME[1] / 2 * unit)
    draw.line((left, y, left + resized.width - 1, y), fill=colour + (255,), width=stroke)
    if size > 16:  # at 16 px the circle is omitted, as the brief permits
        cx = left + round(FRAME[0] / 2 * unit)
        r = max(round(8.25 * unit), stroke + 2)   # outer radius, stroke drawn inward
        hole = tile + (255,) if tile is not None else (0, 0, 0, 0)
        draw.ellipse((cx - r, y - r, cx + r, y + r), fill=hole,
                     outline=colour + (255,), width=stroke)
    return canvas


def write_pngs(image: Image.Image) -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 44, 64):
        render(image, size, DASHBOARD_NAVY).save(STATIC / f"icon-{size}.png")
    # Browser tabs can be light or dark, so the favicon carries its own
    # navy tile rather than relying on the tab colour behind it.
    for size in (16, 32):
        render(image, size, DASHBOARD_NAVY, tile=BRAND_NAVY).save(STATIC / f"favicon-{size}.png")


def main() -> int:
    write_logo(load_font())
    image = s_image()
    write_art(image)
    write_pngs(image)
    print(f"wrote {BRAND.relative_to(ROOT)}/logo-{{light,dark}}.svg, {ART.relative_to(ROOT)}"
          f" and {STATIC.relative_to(ROOT)}/{{icon,favicon}}-*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
