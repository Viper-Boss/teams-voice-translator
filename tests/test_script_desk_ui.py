"""讲稿朗读台界面、字幕分页、音色自动跟随、重播缓存三级查找。"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from teams_voice_translator.defense.overlay import (
    SubtitleOverlay,
    paginate,
    page_for_index,
    token_chunks,
)
from teams_voice_translator.defense.pipeline import CachedUtterance, DefenseEngine
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.tts_cache import cache_key
from teams_voice_translator.defense.ui import SettingsDialog


class PaginationTests(unittest.TestCase):
    def test_short_text_is_one_page(self):
        chunks = token_chunks("Hello there world.")
        self.assertEqual(paginate(chunks, 200), [(0, len(chunks))])

    def test_long_text_splits_into_pages(self):
        chunks = token_chunks("word " * 200)
        pages = paginate(chunks, 50)
        self.assertGreater(len(pages), 1)
        # 分页必须覆盖全部词块，不能漏也不能重
        self.assertEqual(pages[0][0], 0)
        self.assertEqual(pages[-1][1], len(chunks))
        for (_s1, e1), (s2, _e2) in zip(pages, pages[1:]):
            self.assertEqual(e1, s2)

    def test_single_oversized_chunk_gets_its_own_page(self):
        """否则打包循环会原地死循环。"""
        pages = paginate(["x" * 500 + " ", "ok "], 50)
        self.assertEqual(len(pages), 2)

    def test_empty_chunks_paginate_to_nothing(self):
        self.assertEqual(paginate([], 100), [])

    def test_invalid_page_size_is_rejected(self):
        with self.assertRaises(ValueError):
            paginate(["a "], 0)

    def test_page_lookup_finds_the_containing_page(self):
        pages = [(0, 5), (5, 10), (10, 12)]
        self.assertEqual(page_for_index(pages, 0), 0)
        self.assertEqual(page_for_index(pages, 7), 1)
        self.assertEqual(page_for_index(pages, 11), 2)

    def test_out_of_range_index_is_clamped_not_raised(self):
        pages = [(0, 5), (5, 10)]
        self.assertEqual(page_for_index(pages, 99), 1)
        self.assertEqual(page_for_index(pages, -3), 0)
        self.assertEqual(page_for_index([], 3), 0)


class OverlayGrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.overlay = SubtitleOverlay(self.settings)
        self.addCleanup(self.overlay.stop)

    def test_long_sentence_makes_the_card_taller(self):
        """回归：show_sentence 原来完全不改尺寸，长句直接被 128px 裁掉。"""
        self.overlay.show_sentence("短句。", "Short.", 1.0)
        short_height = self.overlay.height()
        self.overlay.show_sentence("很长的中文句子。" * 6, "This is a much longer sentence. " * 8, 20.0)
        self.assertGreater(self.overlay.height(), short_height)

    def test_height_is_capped_so_it_cannot_cover_the_slides(self):
        self.overlay.show_sentence("中文。" * 200, "word " * 2000, 60.0)
        screen = self.app.primaryScreen().availableGeometry().height()
        self.assertLessEqual(self.overlay.height(), int(screen * 0.5))

    def test_bottom_edge_stays_put_while_growing(self):
        self.overlay.show_sentence("短。", "Short.", 1.0)
        bottom = self.overlay.y() + self.overlay.height()
        self.overlay.show_sentence("长。" * 30, "This is much longer. " * 10, 20.0)
        self.assertAlmostEqual(self.overlay.y() + self.overlay.height(), bottom, delta=2)

    def test_highlight_follows_onto_later_pages(self):
        """长文本时正在读的那个词必须还在可视区里。"""
        english = "word " * 400
        self.overlay.show_sentence("中文。", english, 100.0)
        first_page = self.overlay.en_label.text()
        self.overlay._render_page(len(token_chunks(english)) - 1)
        last_page = self.overlay.en_label.text()
        self.assertNotEqual(first_page, last_page)
        self.assertIn("FFD400", last_page)


class VoiceAutoSwitchTests(unittest.TestCase):
    """换模型要自动跟上音色；但用户手动选的音色不能被改掉。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.settings.set("voice_library", [
            {"voice_id": "voice-qa", "target_model": "qwen-audio-3.0-tts-plus",
             "name": "qa", "channel": "legacy", "created_at": "2026-09-22 10:00"},
            {"voice_id": "voice-qa-new", "target_model": "qwen-audio-3.0-tts-plus",
             "name": "qa2", "channel": "legacy", "created_at": "2026-09-22 20:00"},
            {"voice_id": "voice-cosy", "target_model": "cosyvoice-v3.5-plus",
             "name": "cosy", "channel": "legacy", "created_at": "2026-09-22 11:00"},
        ])

    def dialog(self):
        dlg = SettingsDialog(None, self.settings)
        self.addCleanup(dlg.close)
        return dlg

    def select_model(self, dlg, model):
        dlg.tts_model_combo.setCurrentIndex(dlg.tts_model_combo.findData(model))

    def test_switching_model_picks_a_bound_voice(self):
        dlg = self.dialog()
        # 默认模型已经是 qwen-audio-3.0-tts-plus，所以要先离开它，
        # 否则 setCurrentIndex 不换下标、currentIndexChanged 根本不发。
        self.select_model(dlg, "cosyvoice-v3.5-plus")
        dlg._refresh_voice_combo(keep="voice-cosy")
        self.select_model(dlg, "qwen-audio-3.0-tts-plus")
        self.assertEqual(dlg._current_voice_id(), "voice-qa-new")
        self.assertIn("已自动切换", dlg.voice_hint.text())

    def test_newest_bound_voice_wins(self):
        """批量复刻后同一模型常有好几个音色，最新的通常是刚建来试听的那个。"""
        dlg = self.dialog()
        self.assertEqual(
            [e["voice_id"] for e in dlg._voices_bound_to("qwen-audio-3.0-tts-plus")],
            ["voice-qa-new", "voice-qa"],
        )

    def test_manual_voice_choice_is_never_overridden(self):
        """回归：自动切换曾经挂在 _update_voice_hint 上，手选音色的瞬间就被改回去。"""
        dlg = self.dialog()
        self.select_model(dlg, "qwen-audio-3.0-tts-plus")
        dlg._refresh_voice_combo(keep="voice-cosy")
        dlg._update_voice_hint()
        self.assertEqual(dlg._current_voice_id(), "voice-cosy")
        self.assertIn("不是同一绑定模型", dlg.voice_hint.text())

    def test_no_bound_voice_keeps_the_warning(self):
        dlg = self.dialog()
        dlg._refresh_voice_combo(keep="voice-cosy")
        self.select_model(dlg, "cosyvoice-v2")
        self.assertEqual(dlg._current_voice_id(), "voice-cosy")
        self.assertIn("重新复刻", dlg.voice_hint.text())


class ScriptDeskPanelTests(unittest.TestCase):
    """三栏面板：左原文 / 中段落 / 右句子，段是一级单位。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.engine = MagicMock()
        self.engine.script_line_ready.return_value = False
        self.engine.script_pcm.return_value = None

    def panel(self, text="", *, ready=False):
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        if ready:
            self.engine.script_line_ready.return_value = True
            self.engine.script_pcm.return_value = b"\x00\x00" * 12000  # 1 秒
        widget = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(widget.shutdown)
        if text:
            widget.source_edit.setPlainText(text)
            widget._split_clicked()
        return widget

    def test_newlines_become_paragraphs(self):
        """一段 = 一页 PPT，用户靠敲回车划分。"""
        panel = self.panel("第一页内容。\n第二页内容。\n第三页内容。")
        self.assertEqual(panel.paragraph_table.rowCount(), 3)

    def test_selecting_a_paragraph_fills_the_sentence_column(self):
        panel = self.panel("A one. A two.\nB only.")
        panel.paragraph_table.selectRow(0)
        self.assertEqual(panel.line_table.rowCount(), 2)
        panel.paragraph_table.selectRow(1)
        self.assertEqual(panel.line_table.rowCount(), 1)

    def test_chinese_script_is_accepted_directly(self):
        """旧的「讲稿预习」只吃中文且必须先整段翻译；这里中英文都直接切。"""
        panel = self.panel("各位老师好。今天汇报我的工作。")
        self.assertEqual(panel.document.language, "zh")

    def test_language_can_be_forced(self):
        panel = self.panel()
        panel.language_combo.setCurrentIndex(panel.language_combo.findData("en"))
        panel.source_edit.setPlainText("各位老师好。")
        panel._split_clicked()
        self.assertEqual(panel.document.language, "en")

    def test_empty_input_is_refused(self):
        panel = self.panel()
        panel._split_clicked()
        self.assertIn("请先粘贴", panel.status_label.text())
        self.assertEqual(panel.paragraph_table.rowCount(), 0)

    def test_render_all_on_empty_document_says_split_first(self):
        """回归：空文档点「渲染全部」曾经显示「所有句子都已渲染」。"""
        panel = self.panel()
        panel._render_all()
        self.assertIn("请先切分", panel.status_label.text())

    def test_already_cached_lines_show_as_rendered(self):
        panel = self.panel("One. Two.", ready=True)
        self.assertTrue(all(line.rendered for line in panel.document.lines))
        self.assertIn("已渲染 2", panel.progress_label.text())

    def test_unrendered_paragraph_is_not_played(self):
        """缺一句还硬播，现场就是讲到一半突然跳过一句。"""
        panel = self.panel("One. Two.")
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        self.engine.play_script_segment.assert_not_called()
        self.assertIn("还没渲染", panel.status_label.text())

    def test_rendered_paragraph_plays_as_one_buffer(self):
        """整段拼接播出去；句与句之间插入换气停顿（默认 350 ms），不再一口气念到底。"""
        panel = self.panel("One. Two.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        self.engine.play_script_segment.assert_called_once()
        sent = self.engine.play_script_segment.call_args[0][0]
        gap = int(24000 * 350 / 1000) * 2
        self.assertEqual(len(sent), 2 * len(self.engine.script_pcm.return_value) + gap)
        self.settings.set("script_gap_ms", 0)
        panel._play_paragraph()
        sent = self.engine.play_script_segment.call_args[0][0]
        self.assertEqual(len(sent), 2 * len(self.engine.script_pcm.return_value))

    def test_play_single_line_only_sends_that_line(self):
        panel = self.panel("One. Two.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel.line_table.selectRow(1)
        panel._play_line()
        sent = self.engine.play_script_segment.call_args[0][0]
        self.assertEqual(len(sent), len(self.engine.script_pcm.return_value))

    def _engine_at(self, panel, played, active=True):
        """模拟引擎的真实播放进度：光标和换段都以它为准。"""
        self.engine.cached_playback_state.return_value = {
            "tag": panel._segment_tag, "played": played, "total": panel._play_total, "active": active}

    def test_cursor_follows_real_playback_position(self):
        panel = self.panel("One. Two. Three.", ready=True)   # 每句 0.5 秒，句间 0.35 秒
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        self.assertEqual(panel.line_table.currentRow(), 0)
        self._engine_at(panel, 1.0)   # 第二句中间（0.85~1.35）
        panel._advance_cursor()
        self.assertEqual(panel.line_table.currentRow(), 1)

    def test_cached_lines_get_real_duration_not_zero(self):
        """回归：从缓存认出的句子时长是 0，连播一开始就判定播完、直接跳下一段。"""
        panel = self.panel("A one.\nB one.", ready=True)
        self.assertTrue(all(abs(line.seconds - 0.5) < 1e-6 for line in panel.document.lines))
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, 0.0)
        panel._advance_cursor()
        self.assertEqual(panel.paragraph_table.currentRow(), 0, "刚开始播，不能跳到下一段")
        self.assertEqual(self.engine.play_script_segment.call_count, 1)

    def test_cursor_stops_at_the_end_when_not_chaining(self):
        panel = self.panel("One. Two.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        self._engine_at(panel, panel._play_total, active=False)
        panel._advance_cursor()
        self.assertFalse(panel._cursor_timer.isActive())
        self.assertEqual(panel._playing, [])

    def test_chained_playback_moves_to_the_next_paragraph(self):
        panel = self.panel("A one.\nB one.\nC one.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, panel._play_total, active=False)
        panel._advance_cursor()
        self.assertEqual(panel.paragraph_table.currentRow(), 1)
        self.assertEqual(self.engine.play_script_segment.call_count, 2)
        sent = self.engine.play_script_segment.call_args[0][0]
        self.assertEqual(len(sent), int(24000 * 0.8) * 2 + 24000, "下一段前面留段间停顿")

    def test_chained_playback_stops_after_the_last_paragraph(self):
        panel = self.panel("Only one.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, panel._play_total, active=False)
        panel._advance_cursor()
        self.assertIn("最后一段", panel.status_label.text())

    def test_interrupted_by_other_speech_does_not_chain(self):
        panel = self.panel("A one.\nB one.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, 0.1)
        panel._advance_cursor()   # 自己这段开始播了
        self.engine.cached_playback_state.return_value = {"tag": "other", "played": 0, "total": 1, "active": True}
        panel._advance_cursor()
        self.assertEqual(panel._playing, [])
        self.assertEqual(self.engine.play_script_segment.call_count, 1)

    def test_next_paragraph_moves_selection_without_playing(self):
        panel = self.panel("A.\nB.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._next_paragraph()
        self.assertEqual(panel.paragraph_table.currentRow(), 1)
        self.engine.play_script_segment.assert_not_called()

    def test_stop_clears_the_cursor_state(self):
        panel = self.panel("One. Two.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        panel._stop_playback()
        self.assertFalse(panel._cursor_timer.isActive())
        self.assertEqual(panel._playing, [])
        self.engine.interrupt_speech.assert_called_once()

    def test_render_worker_marks_success_and_failure_per_line(self):
        panel = self.panel("Good. Bad.")

        def render(text, language, cancel=None):
            if text.startswith("Bad"):
                raise RuntimeError("配额不足")
            return b"\x00\x00" * 100, 1.5

        self.engine.render_script_line.side_effect = render
        results = []
        panel.render_progress.disconnect()
        panel.render_progress.connect(results.append)
        panel._render_worker(tuple(panel.document.lines), threading.Event())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["error"], "")
        self.assertIn("配额不足", results[1]["error"])

    def test_one_failure_does_not_abort_the_rest(self):
        panel = self.panel("Bad. Good one. Good two.")
        calls = []

        def render(text, language, cancel=None):
            calls.append(text)
            if text.startswith("Bad"):
                raise RuntimeError("超时")
            return b"\x00\x00" * 10, 0.5

        self.engine.render_script_line.side_effect = render
        panel.render_progress.disconnect()
        panel.render_done.disconnect()
        panel._render_worker(tuple(panel.document.lines), threading.Event())
        self.assertEqual(len(calls), 3)

    def test_cancelling_stops_rendering(self):
        panel = self.panel("One. Two. Three.")
        cancel = threading.Event()
        cancel.set()
        panel.render_progress.disconnect()
        panel.render_done.disconnect()
        panel._render_worker(tuple(panel.document.lines), cancel)
        self.engine.render_script_line.assert_not_called()

    def test_progress_updates_both_columns(self):
        panel = self.panel("One. Two.")
        panel._on_render_progress({"index": 0, "seconds": 2.5, "error": ""})
        self.assertTrue(panel.document.lines[0].rendered)
        self.assertIn("1/2", panel.paragraph_table.item(0, 2).text())
        panel._on_render_progress({"index": 1, "error": "没配额"})
        self.assertIn("失败", panel.paragraph_table.item(0, 2).text())

    def test_progress_ignores_out_of_range_index(self):
        panel = self.panel("One.")
        panel._on_render_progress({"index": 99, "error": ""})  # 不该抛

    def test_shutdown_cancels_rendering(self):
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        panel = ScriptDeskPanel(self.settings, self.engine)
        self.assertFalse(panel._cancel.is_set())
        panel.shutdown()
        self.assertTrue(panel._cancel.is_set())


class ReplayLookupTests(unittest.TestCase):
    """重播的三级查找：内存 → 磁盘 → 分块拼接。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def engine(self):
        from teams_voice_translator.defense.tts_cache import TtsDiskCache

        fake = MagicMock()
        fake._pcm_cache = {}
        fake.disk_cache = TtsDiskCache(Path(self.tmp.name))
        fake._tts_settings.return_value = {
            "tts_model": "qwen-audio-3.0-tts-plus", "voice": "v1", "tts_rate": 1.0,
            "tts_pitch": 1.0, "tts_volume": 55, "tts_instruction": "",
            "tts_language_hint": "en", "tts_sample_rate": 24000, "tts_enable_ssml": False,
        }
        fake.lookup_pcm = DefenseEngine.lookup_pcm.__get__(fake)
        return fake

    def test_memory_hit_wins(self):
        eng = self.engine()
        eng._pcm_cache["Hello."] = CachedUtterance(pcm=b"\x01\x02", zh="你好")
        self.assertEqual(eng.lookup_pcm("Hello."), b"\x01\x02")

    def test_disk_hit_when_memory_is_empty(self):
        """内存缓存改一次设置就被清空；磁盘那份要能顶上。"""
        eng = self.engine()
        eng.disk_cache.put(cache_key("Hello.", eng._tts_settings()), b"\x03\x04")
        self.assertEqual(eng.lookup_pcm("Hello."), b"\x03\x04")

    def test_long_answer_is_reassembled_from_its_chunks(self):
        """回归：超过 450 字符的回答按段缓存，但重播按钮传的是整段译文，
        于是长回答永远命不中 —— 这就是「重播还要等好久」的真正原因。"""
        from teams_voice_translator.defense.speech_policy import speech_chunks

        eng = self.engine()
        long_text = ". ".join(f"This is sentence number {i} of the answer" for i in range(40)) + "."
        chunks = speech_chunks(long_text)
        self.assertGreater(len(chunks), 1, "测试文本必须超过分块上限")
        for index, chunk in enumerate(chunks):
            eng._pcm_cache[chunk] = CachedUtterance(pcm=bytes([index, index]), zh="")
        merged = eng.lookup_pcm(long_text)
        self.assertEqual(merged, b"".join(bytes([i, i]) for i in range(len(chunks))))

    def test_partial_chunk_coverage_is_a_miss(self):
        """缺一段就整体判未命中，不能播一半就停。"""
        from teams_voice_translator.defense.speech_policy import speech_chunks

        eng = self.engine()
        long_text = ". ".join(f"This is sentence number {i} of the answer" for i in range(40)) + "."
        chunks = speech_chunks(long_text)
        eng._pcm_cache[chunks[0]] = CachedUtterance(pcm=b"\x01\x01", zh="")
        self.assertIsNone(eng.lookup_pcm(long_text))

    def test_blank_text_is_a_miss(self):
        self.assertIsNone(self.engine().lookup_pcm("   "))


if __name__ == "__main__":
    unittest.main()
