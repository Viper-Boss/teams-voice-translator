"""第 16 批：像真人的小失误开关 + 讲稿/预案音频「渲染一次一直能用」。"""
from __future__ import annotations

import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.defense.humanize import (  # noqa: E402
    LEVEL_CHOICES, humanize_sentence, humanize_text, rate_for,
)
from teams_voice_translator.defense.script_desk import ScriptDocument  # noqa: E402
from teams_voice_translator.defense.tts_cache import RenderManifest, TtsDiskCache  # noqa: E402

WORDS = ("the model we proposed achieves better accuracy on the test set compared with "
         "several methods in our experiments and the results show clear improvement").split()


def _sentences(n, seed=7):
    rng = random.Random(seed)
    return [" ".join(rng.sample(WORDS, 9)).capitalize() + "." for _ in range(n)]


class HumanizeTests(unittest.TestCase):
    def test_关闭时原样不动(self):
        for sentence in _sentences(200):
            self.assertEqual(humanize_sentence(sentence, "off"), sentence)
        self.assertEqual(rate_for("off"), 0.0)
        self.assertEqual([value for value, _label in LEVEL_CHOICES], ["off", "rare", "natural", "more"])

    def test_比例接近档位_大约百分之二到六(self):
        sentences = _sentences(8000)
        for level, low, high in (("rare", 0.01, 0.03), ("natural", 0.02, 0.05), ("more", 0.04, 0.08)):
            changed = sum(humanize_sentence(s, level) != s for s in sentences) / len(sentences)
            self.assertTrue(low <= changed <= high, (level, changed))

    def test_同一句结果固定_渲染和播放命中同一缓存(self):
        for sentence in _sentences(300):
            self.assertEqual(humanize_sentence(sentence, "more"), humanize_sentence(sentence, "more"))

    def test_只加不删_原句的词都还在(self):
        for sentence in _sentences(3000):
            changed = humanize_sentence(sentence, "more")
            if changed == sentence:
                continue
            original = [w.strip(".,").lower() for w in sentence.split()]
            now = [w.strip(".,").lower() for w in changed.split()]
            it = iter(now)
            self.assertTrue(all(any(word == n for n in it) for word in original), changed)

    def test_数字缩写术语不重复不打断(self):
        protected = ["Transformer"]
        rng = random.Random(3)
        risky = ["CNN", "3.5", "BERT-base", "Transformer", "2026"]
        for _ in range(4000):
            words = rng.sample(WORDS, 7)
            words.insert(rng.randrange(len(words)), rng.choice(risky))
            sentence = " ".join(words).capitalize() + "."
            changed = humanize_sentence(sentence, "more", protected=protected)
            if changed == sentence:
                continue
            for token in risky:
                self.assertLessEqual(changed.count(token), sentence.count(token), changed)
                self.assertNotIn(token + "...", changed)

    def test_中文和短句不动(self):
        self.assertEqual(humanize_sentence("我们提出的模型效果更好，准确率提升了很多。", "more"),
                         "我们提出的模型效果更好，准确率提升了很多。")
        self.assertEqual(humanize_sentence("Thank you very much.", "more"), "Thank you very much.")

    def test_一段最多一处(self):
        text = " ".join(_sentences(400))
        changed = humanize_text(text, "more")
        self.assertLessEqual(len(re.findall(r"\.\.\.|\bum,|\buh,|\bUm,|\bUh,", changed)), 2)


class ManifestTests(unittest.TestCase):
    def test_登记簿跨实例保存(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = TtsDiskCache(tmp)
            manifest = RenderManifest(cache)
            entry = RenderManifest.entry_id("script", "Hello.", "en", {"tts_model": "m", "voice": "v"})
            manifest.set(entry, "abc")
            self.assertEqual(RenderManifest(TtsDiskCache(tmp)).get(entry), "abc")
            other = RenderManifest.entry_id("script", "Hello.", "en", {"tts_model": "m", "voice": "v2"})
            self.assertNotEqual(entry, other)


class EngineRenderOnceTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.settings import DefenseSettings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.settings = DefenseSettings(base_dir=self.base)
        self.settings.set("tts_voice_id", "voice-test")
        self.engine = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(self.engine.shutdown)
        self.engine.client = object()

    def _render(self, text):
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x01\x00" * 100) as synth:
            self.engine.render_script_line(text, "en")
        return synth

    def test_改了音量语气_旧渲染照样能播_重开软件也在(self):
        text = "Our method improves the accuracy on every benchmark we tested."
        self._render(text)
        self.assertEqual(self.engine.script_render_state(text, "en"), "exact")
        self.settings.update({"tts_volume": 80, "tts_instruction": "Speak slowly."})
        self.assertEqual(self.engine.script_render_state(text, "en"), "saved")
        self.assertTrue(self.engine.script_pcm(text, "en"))
        # 模拟重开软件：新引擎、同一个目录
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.settings import DefenseSettings
        settings2 = DefenseSettings(base_dir=self.base)
        engine2 = DefenseEngine(settings2, EngineCallbacks())
        self.addCleanup(engine2.shutdown)
        self.assertEqual(engine2.script_render_state(text, "en"), "saved")
        self.assertTrue(engine2.script_pcm(text, "en"))

    def test_换了音色不混用旧声音(self):
        text = "The second chapter describes the dataset in detail."
        self._render(text)
        self.settings.update({"prerender_tts_model": "cosyvoice-v3.5-plus", "prerender_voice_id": "other-voice"})
        self.assertEqual(self.engine.script_render_state(text, "en"), "")
        self.assertIsNone(self.engine.script_pcm(text, "en"))

    def test_重渲染用新参数_不再标旧设置(self):
        text = "We collected more than two hundred interviews for this study."
        self._render(text)
        self.settings.set("tts_volume", 70)
        self.assertEqual(self.engine.script_render_state(text, "en"), "saved")
        synth = self._render(text)
        self.assertEqual(synth.call_count, 1)
        self.assertEqual(self.engine.script_render_state(text, "en"), "exact")

    def test_小失误开关_讲稿发给TTS的是加工后的文本(self):
        self.settings.update({"humanize_level": "more", "prerender_instruction": "follow"})
        texts = [s.replace(".", "") + " in our final experiments." for s in _sentences(400, seed=11)]
        spoken = []
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   side_effect=lambda _c, _s, t, **_k: spoken.append(t) or b"\x01\x00" * 10):
            for text in texts:
                self.engine.render_script_line(text, "en")
        self.assertTrue(any(s != t for s, t in zip(spoken, texts)), "400 句里至少应有一处小失误")
        self.assertTrue(all(self.engine.script_render_state(t, "en") == "exact" for t in texts))

    def test_现场发声_字幕不变_只改送出的文本_有冷却(self):
        self.settings.set("humanize_level", "more")
        with patch("teams_voice_translator.defense.pipeline.humanize_text",
                   side_effect=lambda text, level, protected=(): "The... " + text):
            first = self.engine._prepare_utterances("the model works well on our data.", with_filler=False)
            second = self.engine._prepare_utterances("the model works well on our data.", with_filler=False)
        self.assertEqual(first[0][1], "the model works well on our data.")
        self.assertTrue(first[0][0].startswith("The... "))
        self.assertEqual(second[0][0], second[0][1], "12 秒冷却内不再出第二处")

    def test_预案回答也能用旧设置的渲染(self):
        text = "It is the main limitation of the current work."
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x01\x00" * 50):
            self.engine.render_answer(text)
        self.settings.set("tts_rate", 1.1)
        self.assertTrue(self.engine.answer_ready(text))
        self.assertTrue(self.engine.answer_pcm(text))


class ScriptDeskStaleUiTests(EngineRenderOnceTests.__mro__[0]):
    """讲稿台：重开后旧设置的句子显示就绪，按钮可按新设置重渲染。"""

    def test_旧设置显示就绪_可选重渲染(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        QApplication.instance() or QApplication([])
        text = "Our method improves the accuracy on every benchmark we tested."
        self._render(text)
        self.settings.set("tts_volume", 90)
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.source_edit.setPlainText(text)
        panel.language_combo.setCurrentIndex(panel.language_combo.findData("en"))
        panel._split_clicked()
        line = panel.document.lines[0]
        self.assertTrue(line.rendered)
        self.assertTrue(line.stale)
        self.assertFalse(panel.rerender_stale_button.isHidden())
        with patch.object(panel, "_start_render") as start:
            panel._rerender_stale()
        start.assert_called_once()
        self.assertEqual(list(start.call_args[0][0]), [line])

    # 基类的其他用例不在这里重复跑
    test_改了音量语气_旧渲染照样能播_重开软件也在 = None
    test_换了音色不混用旧声音 = None
    test_重渲染用新参数_不再标旧设置 = None
    test_小失误开关_讲稿发给TTS的是加工后的文本 = None
    test_现场发声_字幕不变_只改送出的文本_有冷却 = None
    test_预案回答也能用旧设置的渲染 = None


class PrerenderInstructionTests(EngineRenderOnceTests.__mro__[0]):
    """讲稿台默认用和声音对比 B「自然答辩语气」完全相同的参数，并整段一次合成。"""

    def test_默认和B一样_可切A或跟随设置(self):
        self.settings.update({"tts_model": "qwen-audio-3.0-tts-plus", "tts_rate": 0.9,
                              "tts_instruction": "Custom style."})
        values = self.engine.script_settings("en")
        self.assertEqual(values["tts_instruction"], "")
        self.assertEqual((values["tts_rate"], values["tts_pitch"]), (1.0, 1.0))
        self.assertEqual(self.engine.answer_settings()["tts_instruction"], "")
        self.settings.set("prerender_instruction", "none")
        self.assertEqual(self.engine.script_settings("en")["tts_instruction"], "")
        self.settings.set("prerender_instruction", "follow")
        values = self.engine.script_settings("en")
        self.assertEqual(values["tts_instruction"], "Custom style.")
        self.assertEqual(values["tts_rate"], 0.9)

    def test_讲稿台显示实际参数并可切换(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        QApplication.instance() or QApplication([])
        self.settings.update({"tts_model": "qwen-audio-3.0-tts-plus"})
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.refresh_render_state()
        self.assertIn("语气指令 无", panel.voice_params_label.text())
        combo = panel.instruction_mode_combo
        combo.setCurrentIndex(combo.findData("none"))
        panel._on_instruction_mode_chosen(combo.currentIndex())
        self.assertEqual(self.settings.get("prerender_instruction"), "none")
        self.assertIn("语气指令 无", panel.voice_params_label.text())

    def test_声音对比B试听的音色_讲稿台标出来(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        QApplication.instance() or QApplication([])
        self.settings.update({"voice_library": [
            {"voice_id": "v-b", "target_model": "qwen-audio-3.0-tts-plus", "name": "我的B"}]})
        self.settings.set("voice_ab_last_b", {"voice_id": "v-b", "model": "qwen-audio-3.0-tts-plus"})
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        labels = [panel.voice_combo.itemText(i) for i in range(panel.voice_combo.count())]
        self.assertTrue(any("声音对比 B 刚用过" in label and "我的B" in label for label in labels), labels)

    test_改了音量语气_旧渲染照样能播_重开软件也在 = None
    test_换了音色不混用旧声音 = None
    test_重渲染用新参数_不再标旧设置 = None
    test_小失误开关_讲稿发给TTS的是加工后的文本 = None
    test_现场发声_字幕不变_只改送出的文本_有冷却 = None
    test_预案回答也能用旧设置的渲染 = None


def _tone_with_pauses(durations, pause=0.3, rate=24000):
    """每句一段 440Hz 的「说话」，句间一段静音。返回 (pcm, 每个静音中心的采样位置)。"""
    import math
    from array import array

    samples = array("h")
    centers = []
    for index, seconds in enumerate(durations):
        if index:
            start = len(samples)
            samples.extend([0] * int(pause * rate))
            centers.append(start + int(pause * rate) // 2)
        for n in range(int(seconds * rate)):
            samples.append(int(8000 * math.sin(2 * math.pi * 440 * n / rate)))
    return samples.tobytes(), centers


class GroupRenderTests(EngineRenderOnceTests.__mro__[0]):
    def test_按句切在停顿里_拼回去等于原音频(self):
        from teams_voice_translator.defense.script_desk import split_pcm_by_text

        texts = ["My name is Zheng Sui.", "Today I will present my thesis on graph learning.",
                 "It has three parts."]
        pcm, centers = _tone_with_pauses([1.2, 2.6, 1.1])
        slices = split_pcm_by_text(pcm, texts)
        self.assertEqual(b"".join(slices), pcm)
        cut = 0
        for piece, center in zip(slices, centers):
            cut += len(piece) // 2
            self.assertLess(abs(cut - center), int(0.3 * 24000 / 2), "切点应落在句间静音里")

    def test_一段只请求一次_逐句都能播(self):
        texts = ["My name is Zheng Sui.", "Today I will present my thesis.", "It has three parts."]
        pcm, _ = _tone_with_pauses([1.0, 1.5, 0.9])
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=pcm) as synth:
            seconds = self.engine.render_script_group(texts, "en")
            self.engine.render_script_group(texts, "en")
        self.assertEqual(synth.call_count, 1, "已渲染的不再请求")
        self.assertEqual(synth.call_args[0][2], " ".join(texts), "整段一次送给 TTS，和声音对比 B 一样")
        self.assertEqual(len(seconds), 3)
        self.assertEqual(self.engine.script_group_state(texts, "en"), "exact")
        self.assertEqual(b"".join(self.engine.script_group_slices(texts, "en")), pcm)

    def test_讲稿台整段渲染_光标逐句_以前逐句渲染的照样能播(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        QApplication.instance() or QApplication([])
        texts = ["My name is Zheng Sui.", "Today I will present my thesis.", "It has three parts."]
        # 先按旧方式逐句渲染
        self.settings.set("script_render_unit", "sentence")
        for text in texts:
            self._render(text)
        self.settings.set("script_render_unit", "group")
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.source_edit.setPlainText(" ".join(texts))
        panel.language_combo.setCurrentIndex(panel.language_combo.findData("en"))
        panel._split_clicked()
        lines = panel.document.lines
        self.assertTrue(all(line.rendered and line.stale for line in lines), "旧的逐句渲染照样能播")
        pcm, _ = _tone_with_pauses([1.0, 1.5, 0.9])
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=pcm) as synth:
            panel._render_worker(tuple(tuple(u) for u in panel._render_units(lines)),
                                 __import__("threading").Event(), True)
        self.assertEqual(synth.call_count, 1)
        panel.refresh_render_state()
        self.assertTrue(all(line.rendered and not line.stale for line in lines))
        joined, bounds = panel._collect_pcm(lines)
        self.assertEqual(joined, pcm, "同一块切出来的句子首尾相接，不再插停顿")
        self.assertEqual(len(bounds), 3)

    test_改了音量语气_旧渲染照样能播_重开软件也在 = None
    test_换了音色不混用旧声音 = None
    test_重渲染用新参数_不再标旧设置 = None
    test_小失误开关_讲稿发给TTS的是加工后的文本 = None
    test_现场发声_字幕不变_只改送出的文本_有冷却 = None
    test_预案回答也能用旧设置的渲染 = None


class AudioManageTests(EngineRenderOnceTests.__mro__[0]):
    """右键管理：单句单独重做 / 换语气 / 删除，本段清空，一键清空。"""

    TEXTS = ["My name is Zheng Sui.", "Today I will present my thesis.", "It has three parts."]

    def _panel_rendered(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        QApplication.instance() or QApplication([])
        self.settings.update({"tts_model": "qwen-audio-3.0-tts-plus"})
        pcm, _ = _tone_with_pauses([1.0, 1.5, 0.9])
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=pcm):
            self.engine.render_script_group(self.TEXTS, "en")
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.source_edit.setPlainText(" ".join(self.TEXTS))
        panel.language_combo.setCurrentIndex(panel.language_combo.findData("en"))
        panel._split_clicked()
        self.assertTrue(all(line.rendered for line in panel.document.lines))
        return panel

    def _wait(self, panel):
        import time as _t
        from PySide6.QtWidgets import QApplication

        deadline = _t.monotonic() + 5
        while panel._rendering and _t.monotonic() < deadline:
            QApplication.processEvents()
            _t.sleep(0.01)
        QApplication.processEvents()
        self.assertFalse(panel._rendering)

    def test_单句换语气单独生成_优先使用_可恢复(self):
        panel = self._panel_rendered()
        line = panel.document.lines[0]
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x05\x00" * 2400) as synth:
            panel._regenerate_line(line, "Speak warmly.", "亲切")
            self._wait(panel)
            panel._regenerate_line(line)          # 再点一次：又是新的一版
            self._wait(panel)
        self.assertEqual(synth.call_count, 2)
        self.assertEqual(synth.call_args_list[0][0][1]["tts_instruction"], "Speak warmly.")
        self.assertEqual(line.custom, "单独重做")
        self.assertIn("单独", line.status_text)
        pcm, bounds = panel._collect_pcm(panel.document.lines)
        self.assertIn(b"\x05\x00" * 2400, pcm)
        panel._drop_override(line)
        self.assertEqual(line.custom, "")
        self.assertTrue(line.rendered, "恢复整段里的这句")

    def test_删除这句_整段那块还在时提示_本段清空和一键清空(self):
        from teams_voice_translator.defense import ui

        panel = self._panel_rendered()
        line = panel.document.lines[1]
        panel._delete_lines([line], whole_blocks=False)
        self.assertTrue(line.rendered, "整段合成的那块不因删一句而断开")
        self.assertIn("整段合成的音频里还在", panel.status_label.text())
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.No):
            panel._clear_paragraph(panel.document.paragraphs[0])
        self.assertTrue(all(l.rendered for l in panel.document.lines), "点了取消不删")
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes):
            panel._clear_paragraph(panel.document.paragraphs[0])
        self.assertFalse(any(l.rendered for l in panel.document.lines))
        self.assertEqual(self.engine.script_group_state(self.TEXTS, "en"), "")

    def test_右键菜单能弹出(self):
        from PySide6.QtCore import QPoint
        from teams_voice_translator.defense import ui

        panel = self._panel_rendered()
        panel.paragraph_table.selectRow(0)
        panel._refresh_lines()
        titles = []
        def grab(menu, _point):
            for action in menu.actions():
                titles.append(action.text())
                if action.menu() is not None:
                    titles.extend(sub.text() for sub in action.menu().actions())

        with patch.object(panel, "_exec_menu", grab):
            rect = panel.line_table.visualItemRect(panel.line_table.item(0, 1))
            panel._line_menu(rect.center())
            rect = panel.paragraph_table.visualItemRect(panel.paragraph_table.item(0, 1))
            panel._paragraph_menu(rect.center())
        joined = " | ".join(titles)
        for word in ("单独重新生成", "换个语气", "删除这句", "清空第 1 段", "一键清空", "播放本段"):
            self.assertIn(word, joined)

    def test_一键清空_逐句模式也能删(self):
        from teams_voice_translator.defense import ui

        self.settings.set("script_render_unit", "sentence")
        for text in self.TEXTS:
            self._render(text)
        panel = self._panel_rendered()
        self.settings.set("script_render_unit", "sentence")
        panel.refresh_render_state()
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes):
            panel._clear_all_audio()
        self.assertFalse(any(l.rendered for l in panel.document.lines))
        self.assertIsNone(self.engine.script_pcm(self.TEXTS[0], "en"))

    test_改了音量语气_旧渲染照样能播_重开软件也在 = None
    test_换了音色不混用旧声音 = None
    test_重渲染用新参数_不再标旧设置 = None
    test_小失误开关_讲稿发给TTS的是加工后的文本 = None
    test_现场发声_字幕不变_只改送出的文本_有冷却 = None
    test_预案回答也能用旧设置的渲染 = None


class ScriptDocStaleTests(unittest.TestCase):
    def test_旧设置显示_且进度提示已存本机(self):
        doc = ScriptDocument.from_text("First sentence here. Second one.")
        for line in doc.lines:
            line.rendered = True
            line.seconds = 1.0
        doc.lines[0].stale = True
        self.assertIn("旧设置", doc.lines[0].status_text)
        text = doc.progress_text()
        self.assertIn("旧设置渲染", text)
        self.assertIn("重开软件直接用", text)


class SettingsUiTests(unittest.TestCase):
    def test_设置页有单独开关并保存(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense import ui

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(base_dir=Path(tmp))
            self.assertEqual(settings.get("humanize_level"), "off")
            dialog = ui.SettingsDialog(None, settings)
            combo = dialog.humanize_combo
            combo.setCurrentIndex(combo.findData("natural"))
            with patch("teams_voice_translator.defense.ui.QMessageBox"):
                dialog._save()
            self.assertEqual(settings.get("humanize_level"), "natural")
            dialog.deleteLater()


class QaBankManageTests(unittest.TestCase):
    """问答库：试听（只走监听）、渲染这条 / 重做一版 / 换语气 / 恢复默认 / 删音频、改了自动删旧声音、布局可调。"""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.qa_bank import QaBankStore, QaEntry
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import QaPanel

        QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.settings = DefenseSettings(base_dir=base)
        self.settings.update({"tts_voice_id": "voice-test", "tts_model": "qwen-audio-3.0-tts-plus"})
        QaBankStore(base).save([
            QaEntry("S1", "Why simulate?", "Real data are rare.", question_zh="为什么仿真？", answer_zh="真实数据少。"),
            QaEntry("S2", "How fast?", "About six milliseconds."),
        ])
        self.engine = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(self.engine.shutdown)
        self.engine.client = object()
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.open_bank_dialog()
        self.dialog = self.panel._bank_dialog
        self.addCleanup(self.dialog.hide)
        self.pcm = b"\x03\x00" * 24000   # 1 秒

    def _wait(self):
        import time as _t
        from PySide6.QtWidgets import QApplication

        deadline = _t.monotonic() + 5
        while self.dialog._rendering and _t.monotonic() < deadline:
            QApplication.processEvents()
            _t.sleep(0.01)
        QApplication.processEvents()
        self.assertFalse(self.dialog._rendering)

    def _synth(self):
        return patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=self.pcm)

    def test_渲染这条_重做一版_换语气_恢复默认_删音频(self):
        entry = self.panel.bank[0]
        self.dialog._load_detail(0)
        with self._synth() as synth:
            self.dialog.rerender_entry(entry)
            self._wait()
            self.assertIn("已渲染", self.dialog.table.item(0, 4).text())
            self.assertIn("1s", self.dialog.table.item(0, 4).text())
            self.dialog.rerender_entry(self.panel.bank[0])          # 再点：重做一版
            self._wait()
            self.assertIn("重做一版", self.dialog.table.item(0, 4).text())
            self.dialog.render_with_tone(self.panel.bank[0], "Speak warmly.", "亲切交流")
            self._wait()
        self.assertEqual(synth.call_count, 3)
        self.assertEqual(synth.call_args_list[2][0][1]["tts_instruction"], "Speak warmly.")
        self.assertIn("亲切交流", self.dialog.table.item(0, 4).text())
        self.assertEqual(self.engine.answer_state(entry.answer), "custom")
        self.dialog.restore_default(self.panel.bank[0])
        self.assertIn("已渲染", self.dialog.table.item(0, 4).text())
        self.dialog.delete_audio([self.panel.bank[0]])
        self.assertEqual(self.dialog.table.item(0, 4).text(), "—")
        self.assertFalse(self.engine.answer_ready(entry.answer))

    def test_改了英文回答_旧声音自动删_保存并渲染(self):
        with self._synth():
            self.engine.render_answer("Real data are rare.")
        self.dialog.refresh()
        self.dialog._load_detail(0)
        self.dialog.a_en_edit.setText("Real MFL data are rare.")
        with self._synth() as synth:
            self.dialog.save_and_render()
            self._wait()
        self.assertEqual(synth.call_count, 1)
        self.assertFalse(self.engine.answer_ready("Real data are rare."), "旧回答的声音删掉了")
        self.assertTrue(self.engine.answer_ready("Real MFL data are rare."))

    def test_试听只走监听设备_没渲染先问再渲染再播(self):
        from teams_voice_translator.defense import ui

        written = []

        class FakePlayer:
            def write(self, data):
                written.append(len(data))

            def close(self):
                pass

        self.dialog.preview_player_factory = FakePlayer
        entry = self.panel.bank[1]
        self.dialog._load_detail(1)
        with self._synth(), patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes):
            self.dialog.preview_entry(entry)
            self._wait()
        import time as _t
        deadline = _t.monotonic() + 3
        while sum(written) < len(self.pcm) and _t.monotonic() < deadline:
            _t.sleep(0.01)
        self.assertEqual(sum(written), len(self.pcm))

    def test_列宽可拖_右键菜单齐全(self):
        from PySide6.QtWidgets import QHeaderView

        header = self.dialog.table.horizontalHeader()
        self.assertEqual(header.sectionResizeMode(2), QHeaderView.Interactive)
        self.dialog.table.setColumnWidth(2, 333)
        self.dialog._save_layout()
        self.assertEqual(self.settings.get("qa_bank_column_widths")[2], 333)
        titles = []

        def grab(menu, _point):
            for action in menu.actions():
                titles.append(action.text())
                if action.menu() is not None:
                    titles.extend(sub.text() for sub in action.menu().actions())

        self.dialog.table.selectRow(0)
        rect = self.dialog.table.visualItemRect(self.dialog.table.item(0, 2))
        with patch.object(self.dialog, "_exec_menu", grab):
            self.dialog._table_menu(rect.center())
        joined = " | ".join(titles)
        for word in ("试听", "渲染这条", "换语气", "放到助答面板", "删除这条预案"):
            self.assertIn(word, joined)


if __name__ == "__main__":
    unittest.main()
