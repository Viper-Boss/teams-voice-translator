from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.defense.records import new_meeting_folder, write_transcripts


class FakeSegment:
    def __init__(self, role: str, source: str, translation: str, latency_ms: int = 0) -> None:
        import time

        self.role = role
        self.source = source
        self.translation = translation
        self.latency_ms = latency_ms
        self.at = time.time()


class RecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_meeting_folder_layout(self) -> None:
        folder = new_meeting_folder(self.base)
        self.assertTrue(folder.name.startswith("答辩_"))
        self.assertTrue((folder / "录音").is_dir())
        self.assertTrue((folder / "字幕").is_dir())

    def test_write_transcripts_files(self) -> None:
        folder = new_meeting_folder(self.base)
        segments = [
            FakeSegment("me", "我的方法是两步反演。", "My method is a two-step inversion.", 1800),
            FakeSegment("committee", "Why is regularization needed?", "为什么需要正则化？"),
        ]
        paths = write_transcripts(folder, segments)
        for key in ("中文字幕", "英文字幕", "双语字幕", "会议全记录"):
            self.assertIn(key, paths)
            self.assertTrue(paths[key].exists())
        zh = paths["中文字幕"].read_text(encoding="utf-8-sig")
        self.assertIn("我的方法是两步反演。", zh)
        self.assertIn("为什么需要正则化？", zh)
        self.assertIn("评委", zh)
        en = paths["英文字幕"].read_text(encoding="utf-8-sig")
        self.assertIn("Why is regularization needed?", en)
        self.assertIn("two-step inversion", en)
        md = paths["会议全记录"].read_text(encoding="utf-8")
        self.assertIn("# 答辩会议全记录", md)
        self.assertIn("1.8s", md)
        self.assertIn("| 时间 | 谁 | 中文 | 英文 | 延迟 |", md)


if __name__ == "__main__":
    unittest.main()
