"""Side-panel backdrop library for the meeting translator.

Backdrops are an axis of their own, deliberately independent from the layout
and palette selectors: the app already ships ``4 layouts x 4 palettes = 16``
combinations, and the artwork choice must stay freely combinable with both.

Image files live in ``local_assets/backgrounds/``, a local-only, git-ignored
directory. Only file names are referenced here -- never commit the artwork,
and never publish it to the public repository or a public release. A missing
file degrades to placeholder text rather than an error, so the app still runs
on machines without the private assets.
"""

from __future__ import annotations

from pathlib import Path

BACKDROP_DIR = Path(__file__).parent / "local_assets" / "backgrounds"

BACKDROP_AUTO = "auto"
BACKDROP_NONE = "none"

# Palette key -> backdrop used when the user leaves the choice on "auto".
THEME_DEFAULT_BACKDROP: dict[str, str] = {
    "shizuku": "shizuku",
    "light": "shizuku",
    "dark": "stellar",
    "warm": "amber",
}

DEFAULT_BACKDROP = BACKDROP_AUTO
FALLBACK_BACKDROP = "shizuku"

# key -> display label, file name and the credit shown under the artwork.
BACKDROP_LIBRARY: dict[str, dict[str, str]] = {
    "shizuku": {
        "label": "水晶雫 · 蓝白",
        "file": "shizuku.jpg",
        "credit": "水晶雫 · 蓝白水晶",
    },
    "sakura": {
        "label": "樱花 · 粉白",
        "file": "sakura.jpg",
        "credit": "樱花 · 春日粉白",
    },
    "stellar": {
        "label": "星空 · 深蓝紫",
        "file": "stellar.jpg",
        "credit": "星空 · 银河深蓝",
    },
    "amber": {
        "label": "暖阳 · 琥珀",
        "file": "amber.jpg",
        "credit": "暖阳 · 金色黄昏",
    },
    "mint": {
        "label": "薄荷 · 青绿",
        "file": "mint.jpg",
        "credit": "薄荷 · 竹林清绿",
    },
    "violet": {
        "label": "紫夜 · 霓虹",
        "file": "violet.jpg",
        "credit": "紫夜 · 霓虹都市",
    },
}


def backdrop_options() -> list[tuple[str, str]]:
    """``(label, key)`` pairs for the backdrop combo boxes."""
    options: list[tuple[str, str]] = [("自动跟随主题", BACKDROP_AUTO)]
    options.extend((entry["label"], key) for key, entry in BACKDROP_LIBRARY.items())
    options.append(("无背景（纯色）", BACKDROP_NONE))
    return options


def resolve_backdrop(theme: str, choice: str | None) -> str:
    """Map a stored choice to a concrete backdrop key.

    ``auto`` follows the palette, ``none`` hides the artwork, and an unknown or
    missing key falls back so a deleted local file never blanks the panel.
    """
    if choice == BACKDROP_NONE:
        return BACKDROP_NONE
    if choice and choice != BACKDROP_AUTO and choice in BACKDROP_LIBRARY:
        return choice
    return THEME_DEFAULT_BACKDROP.get(theme, FALLBACK_BACKDROP)


def backdrop_path(key: str) -> Path | None:
    """Absolute path of a backdrop image, or None when it is unavailable."""
    entry = BACKDROP_LIBRARY.get(key)
    if not entry:
        return None
    path = BACKDROP_DIR / entry["file"]
    return path if path.is_file() else None


def backdrop_credit(key: str) -> str:
    """Short human-readable name shown under the artwork."""
    entry = BACKDROP_LIBRARY.get(key)
    return entry.get("credit", "") if entry else ""


def backdrop_label(key: str) -> str:
    entry = BACKDROP_LIBRARY.get(key)
    return entry.get("label", "") if entry else ""


__all__ = [
    "BACKDROP_AUTO",
    "BACKDROP_DIR",
    "BACKDROP_LIBRARY",
    "BACKDROP_NONE",
    "DEFAULT_BACKDROP",
    "FALLBACK_BACKDROP",
    "THEME_DEFAULT_BACKDROP",
    "backdrop_credit",
    "backdrop_label",
    "backdrop_options",
    "backdrop_path",
    "resolve_backdrop",
]
