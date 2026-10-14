"""讲稿朗读台的数据模型 + 合成音频磁盘缓存。

这两块是「断网也能讲完 20 分钟 PPT」的地基，所以它们的失败模式要比功能本身
更值得钉住：缓存坏了必须降级成未命中，不能抛；句子切错了宁可切粗，不能把
"Fig. 3" 拆成两句。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.defense.script_desk import (
    ScriptDocument,
    ScriptLine,
    cumulative_bounds,
    detect_language,
    line_at_elapsed,
    split_paragraphs,
    split_sentences,
)
from teams_voice_translator.defense.tts_cache import TtsDiskCache, cache_key


class LanguageDetectionTests(unittest.TestCase):
    def test_chinese_and_english_are_told_apart(self):
        self.assertEqual(detect_language("各位老师好，我来汇报我的研究。"), "zh")
        self.assertEqual(detect_language("Good morning, let me present my work."), "en")

    def test_english_with_a_few_chinese_terms_stays_english(self):
        text = "The baseline is a CNN. 对照组 uses the same data."
        self.assertEqual(detect_language(text), "en")

    def test_empty_text_defaults_to_english(self):
        self.assertEqual(detect_language(""), "en")
        self.assertEqual(detect_language("   "), "en")


class SentenceSplitTests(unittest.TestCase):
    def test_chinese_punctuation_splits(self):
        lines = split_sentences("各位老师好。今天汇报我的工作！有三个部分？")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].endswith("。"))

    def test_english_sentences_split(self):
        lines = split_sentences("First point. Second point. Third one.")
        self.assertEqual(len(lines), 3)

    def test_abbreviations_are_not_split(self):
        """把 Fig. 3 切成两句，现场点读就会断在半句上。"""
        lines = split_sentences("As shown in Fig. 3 the error drops. That is the key result.")
        self.assertEqual(len(lines), 2)
        self.assertIn("Fig. 3", lines[0])

    def test_decimal_numbers_survive(self):
        lines = split_sentences("Accuracy reached 92.7 percent overall.")
        self.assertEqual(len(lines), 1)

    def test_blank_lines_separate_paragraphs(self):
        lines = split_sentences("第一段没有句号\n\n第二段也没有")
        self.assertEqual(len(lines), 2)

    def test_runaway_sentence_is_hard_wrapped(self):
        """有人整段不打标点；一句合成半分钟，点一下要等很久。"""
        text = "word " * 300
        lines = split_sentences(text, max_chars=200)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= 200 for line in lines))

    def test_whitespace_is_normalised(self):
        lines = split_sentences("  多余   空格   全都   去掉。  ")
        self.assertEqual(lines, ["多余 空格 全都 去掉。"])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("\n\n  \n"), [])

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            split_sentences("abc", max_chars=0)


class ScriptDocumentTests(unittest.TestCase):
    def test_document_indexes_every_line(self):
        doc = ScriptDocument.from_text("One. Two. Three.")
        self.assertEqual([line.index for line in doc.lines], [0, 1, 2])
        self.assertEqual(len(doc), 3)

    def test_language_is_detected_but_can_be_forced(self):
        self.assertEqual(ScriptDocument.from_text("Hello there.").language, "en")
        self.assertEqual(ScriptDocument.from_text("Hello there.", language="zh").language, "zh")

    def test_every_line_inherits_the_document_language(self):
        doc = ScriptDocument.from_text("各位老师好。今天汇报。")
        self.assertTrue(all(line.language == "zh" for line in doc.lines))

    def test_progress_reflects_render_state(self):
        doc = ScriptDocument.from_text("One. Two. Three.")
        self.assertIn("已渲染 0", doc.progress_text())
        for line in doc.lines:
            line.rendered = True
            line.seconds = 2.0
        self.assertIn("已渲染 3", doc.progress_text())
        self.assertIn("现场可离线朗读", doc.progress_text())

    def test_failures_are_counted_and_shown(self):
        doc = ScriptDocument.from_text("One. Two.")
        doc.lines[1].error = "配额不足"
        self.assertEqual(doc.failed_count, 1)
        self.assertIn("失败 1", doc.progress_text())

    def test_pending_puts_failed_lines_last(self):
        """一次「全部渲染」不该被同一个坏句子反复拖住。"""
        doc = ScriptDocument.from_text("One. Two. Three.")
        doc.lines[0].error = "超时"
        pending = doc.pending()
        self.assertEqual([line.index for line in pending], [1, 2, 0])

    def test_rendered_lines_are_not_pending(self):
        doc = ScriptDocument.from_text("One. Two.")
        doc.lines[0].rendered = True
        self.assertEqual([line.index for line in doc.pending()], [1])

    def test_status_and_preview_are_display_safe(self):
        line = ScriptLine(index=0, paragraph=0, order=0, text="x" * 200)
        self.assertEqual(len(line.preview), 80)
        self.assertIn("未渲染", line.status_text)
        line.rendered, line.seconds = True, 3.25
        self.assertIn("3.2", line.status_text)
        line.error = "失败了"
        self.assertIn("✗", line.status_text)

    def test_empty_document_says_so(self):
        self.assertIn("还没有导入", ScriptDocument.from_text("").progress_text())


class _FakeSettings(dict):
    pass


class CacheKeyTests(unittest.TestCase):
    def base(self):
        return _FakeSettings(
            tts_model="qwen-audio-3.0-tts-plus", voice="v1", tts_rate=1.0,
            tts_pitch=1.0, tts_volume=55, tts_instruction="", tts_language_hint="en",
            tts_sample_rate=24000, tts_enable_ssml=False,
        )

    def test_same_text_and_params_give_the_same_key(self):
        self.assertEqual(cache_key("Hello.", self.base()), cache_key("Hello.", self.base()))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(cache_key("  Hello.  ", self.base()), cache_key("Hello.", self.base()))

    def test_different_text_gives_a_different_key(self):
        self.assertNotEqual(cache_key("Hello.", self.base()), cache_key("Goodbye.", self.base()))

    def test_every_voice_affecting_param_changes_the_key(self):
        """否则改完语速重播，听到的还是旧速度 —— 比不缓存更糟。"""
        original = cache_key("Hello.", self.base())
        for field, value in (
            ("tts_model", "cosyvoice-v3.5-plus"), ("voice", "v2"), ("tts_rate", 0.9),
            ("tts_pitch", 1.1), ("tts_volume", 70), ("tts_instruction", "Speak calmly."),
            ("tts_language_hint", "zh"), ("tts_sample_rate", 16000), ("tts_enable_ssml", True),
        ):
            changed = self.base()
            changed[field] = value
            self.assertNotEqual(original, cache_key("Hello.", changed), field)


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = TtsDiskCache(Path(self.tmp.name))

    def test_round_trip(self):
        pcm = b"\x01\x02" * 100
        self.assertTrue(self.cache.put("a" * 64, pcm))
        self.assertTrue(self.cache.has("a" * 64))
        self.assertEqual(self.cache.get("a" * 64), pcm)

    def test_missing_key_is_a_miss_not_an_error(self):
        self.assertIsNone(self.cache.get("b" * 64))
        self.assertFalse(self.cache.has("b" * 64))

    def test_odd_length_payload_is_refused(self):
        """半个采样点的 PCM 播出来是噪声，不如当没有。"""
        self.assertFalse(self.cache.put("c" * 64, b"\x01\x02\x03"))
        self.assertFalse(self.cache.has("c" * 64))

    def test_empty_payload_is_refused(self):
        self.assertFalse(self.cache.put("d" * 64, b""))

    def test_truncated_file_is_treated_as_a_miss_and_removed(self):
        """上次写到一半断电，留下的半截文件不能当成有效音频。"""
        key = "e" * 64
        self.cache.put(key, b"\x01\x02" * 10)
        path = self.cache._path_for(key)
        path.write_bytes(b"\x01\x02\x03")
        self.assertIsNone(self.cache.get(key))
        self.assertFalse(path.exists())

    def test_discard_removes_one_entry(self):
        key = "f" * 64
        self.cache.put(key, b"\x00\x00" * 8)
        self.cache.discard(key)
        self.assertFalse(self.cache.has(key))
        self.cache.discard(key)  # 再删一次不该抛

    def test_clear_empties_everything(self):
        for index in range(5):
            self.cache.put(f"{index:064d}", b"\x00\x00" * 8)
        self.assertEqual(self.cache.clear(), 5)
        self.assertEqual(self.cache.stats()["count"], 0)

    def test_eviction_keeps_total_under_the_limit(self):
        import os, time

        small = TtsDiskCache(Path(self.tmp.name) / "small")
        payload = b"\x00\x00" * 512  # 1 KiB
        now = time.time()
        for index in range(12):
            key = f"{index:064d}"
            self.assertTrue(small.put(key, payload))
            # mtime 只有秒级精度，手动拉开才能让 LRU 有确定顺序
            os.utime(small._path_for(key), (now + index, now + index))
        # 先写满再压低上限：eviction 在 put 内部跑，一开始就压低会边写边删
        small.limit_bytes = 4096
        small._evict_if_needed()
        self.assertLessEqual(small.total_bytes(), small.limit_bytes)
        self.assertGreater(small.stats()["count"], 0)
        # 最旧的先走，最新的留下
        self.assertFalse(small.has(f"{0:064d}"))
        self.assertTrue(small.has(f"{11:064d}"))

    def test_stats_report_usage(self):
        self.cache.put("g" * 64, b"\x00\x00" * 50)
        stats = self.cache.stats()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["bytes"], 100)

    def test_unwritable_root_disables_the_cache_instead_of_raising(self):
        """缓存永远是锦上添花：建不出目录就整体停用，不能让答辩中断。"""
        blocker = Path(self.tmp.name) / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        cache = TtsDiskCache(blocker)
        self.assertFalse(cache.enabled)
        self.assertFalse(cache.put("h" * 64, b"\x00\x00"))
        self.assertIsNone(cache.get("h" * 64))
        self.assertFalse(cache.has("h" * 64))


class ParagraphModelTests(unittest.TestCase):
    """段是一级单位：一段 = 一页 PPT，用户靠敲回车来划分。"""

    def test_每个换行都开始新的一段(self):
        self.assertEqual(split_paragraphs("第一页\n第二页\n第三页"), ["第一页", "第二页", "第三页"])

    def test_连续空行不产生空段(self):
        self.assertEqual(split_paragraphs("A\n\n\nB"), ["A", "B"])

    def test_段内空白被归一(self):
        self.assertEqual(split_paragraphs("  多余   空格  "), ["多余 空格"])

    def test_空文本没有段(self):
        self.assertEqual(split_paragraphs(""), [])
        self.assertEqual(split_paragraphs("  \n \n "), [])

    def test_一段可以含多句(self):
        doc = ScriptDocument.from_text("First. Second.\nThird only.")
        self.assertEqual(len(doc.paragraphs), 2)
        self.assertEqual([len(p.lines) for p in doc.paragraphs], [2, 1])

    def test_句子同时带全篇序号和段内序号(self):
        doc = ScriptDocument.from_text("A one. A two.\nB one.")
        self.assertEqual([l.index for l in doc.lines], [0, 1, 2])
        self.assertEqual([l.paragraph for l in doc.lines], [0, 0, 1])
        self.assertEqual([l.order for l in doc.lines], [0, 1, 0])

    def test_段号与下标一致(self):
        doc = ScriptDocument.from_text("A.\nB.\nC.")
        self.assertEqual([p.index for p in doc.paragraphs], [0, 1, 2])

    def test_整段渲染完才算就绪(self):
        """缺一句中间就会断，不能算能上台。"""
        doc = ScriptDocument.from_text("One. Two.")
        para = doc.paragraphs[0]
        self.assertFalse(para.ready)
        para.lines[0].rendered = True
        self.assertFalse(para.ready)
        para.lines[1].rendered = True
        self.assertTrue(para.ready)

    def test_段状态显示进度和失败(self):
        doc = ScriptDocument.from_text("One. Two.")
        para = doc.paragraphs[0]
        self.assertIn("0/2", para.status_text)
        para.lines[0].rendered, para.lines[0].seconds = True, 2.0
        self.assertIn("1/2", para.status_text)
        para.lines[1].error = "超时"
        self.assertIn("失败", para.status_text)

    def test_段标题取开头且截断(self):
        doc = ScriptDocument.from_text("x" * 200 + ".")
        self.assertLessEqual(len(doc.paragraphs[0].title), 42)

    def test_按下标取段和取句(self):
        doc = ScriptDocument.from_text("A.\nB.")
        self.assertIsNotNone(doc.paragraph_at(1))
        self.assertIsNone(doc.paragraph_at(9))
        self.assertIsNone(doc.paragraph_at(-1))
        self.assertIsNotNone(doc.line_at(1))
        self.assertIsNone(doc.line_at(9))

    def test_进度同时报段数和句数(self):
        doc = ScriptDocument.from_text("A one. A two.\nB one.")
        self.assertIn("2 段 / 3 句", doc.progress_text())


class CursorTrackingTests(unittest.TestCase):
    """光标跟随：整段一次播出去，界面靠各句时长定位高亮。"""

    def lines(self, *durations):
        out = []
        for order, seconds in enumerate(durations):
            line = ScriptLine(index=order, paragraph=0, order=order, text=f"s{order}")
            line.seconds = seconds
            out.append(line)
        return out

    def test_累计边界是首尾相接的(self):
        bounds = cumulative_bounds(self.lines(1.0, 2.0, 0.5))
        self.assertEqual(bounds, [(0.0, 1.0), (1.0, 3.0), (3.0, 3.5)])

    def test_空列表没有边界(self):
        self.assertEqual(cumulative_bounds([]), [])

    def test_按播放秒数定位句子(self):
        bounds = cumulative_bounds(self.lines(1.0, 2.0, 0.5))
        self.assertEqual(line_at_elapsed(bounds, 0.0), 0)
        self.assertEqual(line_at_elapsed(bounds, 0.9), 0)
        self.assertEqual(line_at_elapsed(bounds, 1.0), 1)
        self.assertEqual(line_at_elapsed(bounds, 2.9), 1)
        self.assertEqual(line_at_elapsed(bounds, 3.2), 2)

    def test_越界钳到两端而不是抛(self):
        bounds = cumulative_bounds(self.lines(1.0, 1.0))
        self.assertEqual(line_at_elapsed(bounds, -5), 0)
        self.assertEqual(line_at_elapsed(bounds, 999), 1)
        self.assertEqual(line_at_elapsed([], 1.0), -1)

    def test_零时长句子不会卡住光标(self):
        """渲染失败的句子 seconds 是 0，不能让光标永远停在它上面。"""
        bounds = cumulative_bounds(self.lines(1.0, 0.0, 1.0))
        self.assertEqual(line_at_elapsed(bounds, 1.5), 2)


if __name__ == "__main__":
    unittest.main()
