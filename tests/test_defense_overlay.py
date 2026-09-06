from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from teams_voice_translator.defense.overlay import (  # noqa: E402
    SubtitleOverlay,
    current_token_index,
    highlight_html,
    token_chunks,
)
from teams_voice_translator.defense.settings import DefenseSettings  # noqa: E402


class KaraokeTimingTest(unittest.TestCase):
    def test_token_chunks_keep_words(self) -> None:
        chunks = token_chunks("The main contribution is novel.")
        self.assertEqual(chunks, ["The ", "main ", "contribution ", "is ", "novel."])

    def test_index_progression_monotonic(self) -> None:
        chunks = token_chunks("One two three four five.")
        indexes = [current_token_index(chunks, p / 10) for p in range(11)]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], len(chunks) - 1)

    def test_highlight_marks_current_word_only(self) -> None:
        chunks = token_chunks("Hello world.")
        html = highlight_html(chunks, 1)
        self.assertIn("FFD400", html)
        self.assertIn("world.", html)

    def test_empty_chunks_safe(self) -> None:
        self.assertEqual(current_token_index([], 0.5), -1)


class SubtitleOverlayTest(unittest.TestCase):
    app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.settings = DefenseSettings(base_dir=Path(cls._tmp.name))
        cls.overlay = SubtitleOverlay(cls.settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.overlay.stop()
        cls._tmp.cleanup()

    def test_show_sentence_displays_both_lines(self) -> None:
        self.overlay.show_sentence("主要贡献是新算法。", "The main contribution is a novel algorithm.", 6.0)
        self.assertIn("主要贡献", self.overlay.zh_label.text())
        self.assertIn("The main", self.overlay.en_label.text())
        self.assertEqual(self.overlay._duration, 6.0)

    def test_tick_moves_highlight(self) -> None:
        import time as _time

        self.overlay.show_sentence("你好。", "Hello beautiful world.", 4.0)
        self.overlay._t0 = _time.monotonic() - 3.0  # 75% 进度
        self.overlay._tick()
        html = self.overlay.en_label.text()
        self.assertIn("FFD400", html)
        self.assertIn("world.", html)

    def test_set_duration_matches_sentence(self) -> None:
        self.overlay.show_sentence("你好。", "Hello beautiful world.", 0.0)
        self.overlay.set_duration("Hello beautiful world.", 5.5)
        self.assertEqual(self.overlay._duration, 5.5)
        self.overlay.set_duration("another sentence", 9.0)
        self.assertEqual(self.overlay._duration, 5.5, "不同句子的时长不应误套用")

    def test_stop_clears_and_hides(self) -> None:
        self.overlay.show_sentence("你好。", "Hello.", 2.0)
        self.overlay.stop()
        self.assertFalse(self.overlay.isVisible())
        self.assertEqual(self.overlay._chunks, [])


if __name__ == "__main__":
    unittest.main()
