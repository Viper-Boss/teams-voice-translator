"""Exercise every layout x backdrop combination and report what rendered.

Run from the project root:

    ..\\teams-voice-translator\\.venv\\Scripts\\python.exe tools\\verify_backdrops.py

Confirms that the backdrop selector is a genuinely independent third axis:
all four layouts must be able to show all six artworks, "none" must hide the
panel everywhere, and "auto" must keep following the palette.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from teams_voice_translator.backdrops import (  # noqa: E402
    BACKDROP_NONE,
    backdrop_credit,
    backdrop_options,
    resolve_backdrop,
)
from teams_voice_translator.ui import MainWindow  # noqa: E402

LAYOUTS = ("crystal", "signal", "studio", "fluent")
PALETTES = ("shizuku", "light", "dark", "warm")
ARTWORK_KEYS = [key for _label, key in backdrop_options()]


def main() -> int:
    os.environ["APPDATA"] = str(PROJECT_ROOT / "test-artifacts" / "backdrop-verify-appdata")
    app = QApplication.instance() or QApplication([])
    with patch.object(MainWindow, "start_hotkeys"):
        window = MainWindow()
    window.resize(1460, 900)
    window.show()
    for _ in range(10):
        app.processEvents()

    failures: list[str] = []

    print("=== layout x backdrop matrix (palette = shizuku) ===")
    window._select_data(window.theme_quick, "shizuku")
    header = f"{'':8s}" + "".join(f"{k[:7]:>9s}" for k in ARTWORK_KEYS)
    print(header)
    for layout_name in LAYOUTS:
        window._select_data(window.layout_quick, layout_name)
        window.apply_layout()
        row = f"{layout_name:8s}"
        for key in ARTWORK_KEYS:
            window._select_data(window.backdrop_quick, key)
            app.processEvents()
            pixmap = window.artwork.pixmap()
            loaded = bool(pixmap and not pixmap.isNull())
            visible = window.artwork_panel.isVisible()
            if key == BACKDROP_NONE:
                ok = not visible
                mark = "hidden" if ok else "SHOWN!"
            else:
                ok = loaded and visible
                mark = f"{pixmap.width()}x{pixmap.height()}" if ok else "FAIL"
            if not ok:
                failures.append(f"{layout_name}/{key}: visible={visible} loaded={loaded}")
            row += f"{mark:>9s}"
        print(row)

    print()
    print("=== artwork panel width per layout ===")
    widths = {}
    window._select_data(window.backdrop_quick, "shizuku")
    for layout_name in LAYOUTS:
        window._select_data(window.layout_quick, layout_name)
        window.apply_layout()
        app.processEvents()
        widths[layout_name] = window.artwork_panel.width()
        print(f"  {layout_name:8s} {widths[layout_name]:>4d} px")

    print()
    print("=== auto follows palette ===")
    window._select_data(window.backdrop_quick, "auto")
    for palette_name in PALETTES:
        window._select_data(window.theme_quick, palette_name)
        window.apply_theme()
        app.processEvents()
        expected = backdrop_credit(resolve_backdrop(palette_name, "auto"))
        actual = window.art_credit.text().splitlines()[0]
        ok = expected == actual
        if not ok:
            failures.append(f"auto/{palette_name}: expected {expected!r}, got {actual!r}")
        print(f"  [{'OK ' if ok else 'BAD'}] {palette_name:8s} -> {actual}")

    print()
    print("=== settings thumbnail ===")
    for key in ("sakura", "auto", BACKDROP_NONE):
        window._select_data(window.backdrop_quick, key)
        app.processEvents()
        preview = window.backdrop_preview.pixmap()
        has = bool(preview and not preview.isNull())
        expect_none = key == BACKDROP_NONE
        ok = (not has) if expect_none else has
        if not ok:
            failures.append(f"preview/{key}: loaded={has}")
        print(f"  [{'OK ' if ok else 'BAD'}] {key:8s} thumbnail={'shown' if has else 'empty'}")

    window.close()
    window.deleteLater()
    app.processEvents()

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for item in failures:
            print("  -", item)
        return 1
    print("All layout x backdrop combinations behaved as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
