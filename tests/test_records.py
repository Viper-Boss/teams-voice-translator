import json
import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.records import SubtitleSession, srt_timestamp


class RecordTests(unittest.TestCase):
    def test_srt_timestamp(self):
        self.assertEqual(srt_timestamp(0), "00:00:00,000")
        self.assertEqual(srt_timestamp(3661.234), "01:01:01,234")

    def test_cache_always_keeps_both_languages_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            session = SubtitleSession(Path(temp) / "cache")
            session.add("老师您好", "Hello, Professor.", "F8 原声")
            cached = json.loads(session.cache_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(cached["chinese"], "老师您好")
            self.assertEqual(cached["english"], "Hello, Professor.")
            self.assertEqual(cached["source"], "F8 原声")

    def test_export_can_select_language_and_format(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = SubtitleSession(root / "cache")
            session.add("老师您好", "Hello, Professor.", "F8 原声")
            session.add("我们开始吧", "Let's begin.", "F9 翻译")
            txt_path, srt_path = session.export(root / "out", language="both", file_format="both")
            txt = txt_path.read_text(encoding="utf-8")
            srt = srt_path.read_text(encoding="utf-8")
            self.assertIn("中文：老师您好", txt)
            self.assertIn("English: Hello, Professor.", txt)
            self.assertIn("1\n", srt)
            self.assertIn("2\n", srt)
            self.assertIn("我们开始吧\nLet's begin.", srt)
            zh_path = session.export(root / "out", language="zh", file_format="srt")[0]
            en_path = session.export(root / "out", language="en", file_format="txt")[0]
            self.assertNotIn("Hello, Professor.", zh_path.read_text(encoding="utf-8"))
            self.assertNotIn("老师您好", en_path.read_text(encoding="utf-8"))

    def test_cleanup_removes_temporary_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            session = SubtitleSession(Path(temp))
            session.add("中文", "English", "test")
            self.assertTrue(session.cache_path.exists())
            session.cleanup()
            self.assertFalse(session.cache_path.exists())

    def test_all_export_formats_keep_speaker_and_bilingual_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = SubtitleSession(root / "cache")
            session.add("请解释一下", "Could you explain that?", "我 / F9")
            paths = session.export(
                root / "out",
                language="both",
                file_format="all",
                include_source=True,
                include_time=True,
            )
            self.assertEqual(
                {path.suffix for path in paths},
                {".srt", ".txt", ".vtt", ".md", ".jsonl"},
            )
            markdown = next(path for path in paths if path.suffix == ".md").read_text(
                encoding="utf-8"
            )
            self.assertIn("我 / F9", markdown)
            self.assertIn("Could you explain that?", markdown)

    def test_crash_cache_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = SubtitleSession(root)
            old.add("老师好", "Hello, Professor.", "我")
            recovered = SubtitleSession(root)
            candidates = SubtitleSession.discover_recoverable(
                root, exclude=recovered.cache_path
            )
            self.assertIn(old.cache_path, candidates)
            self.assertEqual(recovered.import_cache(old.cache_path), 1)
            self.assertEqual(recovered.records[0].english, "Hello, Professor.")

    def test_cache_paths_are_unique_for_back_to_back_sessions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = SubtitleSession(root)
            second = SubtitleSession(root)
            self.assertNotEqual(first.cache_path, second.cache_path)


if __name__ == "__main__":
    unittest.main()
