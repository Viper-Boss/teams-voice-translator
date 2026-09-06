"""Render the four reference-based VoiceStudio layouts for visual QA."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from teams_voice_translator.ui import MainWindow  # noqa: E402


PAIRS = (
    ("crystal", "shizuku"),
    ("fluent", "light"),
    ("signal", "dark"),
    ("studio", "warm"),
)


def render(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(output_dir / "appdata")
    app = QApplication.instance() or QApplication([])
    with patch.object(MainWindow, "start_hotkeys"):
        window = MainWindow()
    window.show()
    app.processEvents()
    for layout_name, palette_name in PAIRS:
        window._select_data(window.layout_quick, layout_name)
        window._select_data(window.theme_quick, palette_name)
        window.apply_layout()
        window.apply_theme()
        for page_name, page in (
            ("meeting", window.meeting_tab),
            ("voice", window.voicestudio_tab),
            ("settings", window.settings_tab),
        ):
            window.tabs.setCurrentWidget(page)
            app.processEvents()
            target = output_dir / f"{layout_name}-{palette_name}-{page_name}.png"
            if not window.grab().save(str(target)):
                raise RuntimeError(f"Unable to save {target}")
        # Preserve the historical preview filenames for documentation links.
        window.tabs.setCurrentWidget(window.voicestudio_tab)
        app.processEvents()
        legacy_target = output_dir / f"{layout_name}-{palette_name}.png"
        if not window.grab().save(str(legacy_target)):
            raise RuntimeError(f"Unable to save {legacy_target}")
    window.close()
    window.deleteLater()
    app.processEvents()


if __name__ == "__main__":
    render(Path("test-artifacts") / "reference-ui")
