"""Generate the DefenseMode application icon.

Design: a dark rounded-square tile (matches the app's dark theme) with a
soft glow, a modern microphone silhouette and five voice-waveform bars in
the app's blue->green accent gradient.  Drawn at 1024px and downscaled so
every size down to 16px stays crisp.

Usage: python tools/make_defense_icon.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
TILE_RADIUS = 232

# App accent colors (defense/ui.py): blue -> green gradient, dark navy tile.
TILE_TOP = (32, 41, 63)
TILE_BOTTOM = (14, 18, 28)
BLUE = (79, 140, 255)
GREEN = (62, 207, 142)
MIC_TOP = (255, 255, 255)
MIC_BOTTOM = (196, 209, 236)


def lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vertical_gradient(width: int, height: int, top: tuple, bottom: tuple) -> Image.Image:
    column = Image.new("RGB", (1, height))
    for y in range(height):
        column.putpixel((0, y), lerp(top, bottom, y / max(height - 1, 1)))
    return column.resize((width, height))


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def build_icon() -> Image.Image:
    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # Tile: diagonal-feel dark gradient clipped to a rounded square.
    gradient = vertical_gradient(SIZE, SIZE, TILE_TOP, TILE_BOTTOM).convert("RGBA")
    icon.paste(gradient, (0, 0), rounded_mask(SIZE, TILE_RADIUS))

    # Soft accent glow behind the artwork so the tile is not flat black.
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((212, 132, 812, 732), fill=BLUE + (52,))
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    clipped = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    clipped.paste(glow, (0, 0), rounded_mask(SIZE, TILE_RADIUS))
    icon = Image.alpha_composite(icon, clipped)

    draw = ImageDraw.Draw(icon)

    # Microphone capsule with its own vertical gradient.
    capsule_left, capsule_right = 392, 632
    capsule_top, capsule_bottom = 176, 566
    capsule_gradient = vertical_gradient(capsule_right - capsule_left, capsule_bottom - capsule_top, MIC_TOP, MIC_BOTTOM)
    capsule_mask = Image.new("L", capsule_gradient.size, 0)
    ImageDraw.Draw(capsule_mask).rounded_rectangle(
        (0, 0, capsule_gradient.size[0] - 1, capsule_gradient.size[1] - 1), radius=120, fill=255
    )
    icon.paste(capsule_gradient, (capsule_left, capsule_top), capsule_mask)

    # Holder arc wrapping the lower half of the capsule.
    arc_box = (292, 126, 732, 830)
    draw.arc(arc_box, start=12, end=168, fill=MIC_BOTTOM, width=44)
    # Stem connecting arc bottom to the waveform.
    stem_top = arc_box[3] - 6
    draw.rounded_rectangle((489, stem_top, 535, stem_top + 96), radius=23, fill=MIC_BOTTOM)

    # Voice waveform: five rounded bars, blue -> green, centered under the mic.
    bar_width = 44
    gap = 46
    heights = [88, 168, 252, 168, 88]
    total = 5 * bar_width + 4 * gap
    start_x = (SIZE - total) // 2
    center_y = 872
    for index, height in enumerate(heights):
        t = index / (len(heights) - 1)
        color = lerp(BLUE, GREEN, t)
        x0 = start_x + index * (bar_width + gap)
        draw.rounded_rectangle(
            (x0, center_y - height // 2, x0 + bar_width, center_y + height // 2),
            radius=bar_width // 2,
            fill=color,
        )
    return icon


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    icon = build_icon()
    icon.save(out_dir / "defense_mode_preview.png")
    icon.save(
        out_dir / "defense_mode.ico",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)],
    )
    for size in (256, 32, 16):
        icon.resize((size, size), Image.LANCZOS).save(out_dir / f"preview_{size}.png")
    print(f"icon written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
