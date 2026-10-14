"""快捷用语、情景模式、后台预渲染、助答弹窗。全部离线，引擎用 MagicMock。

测试里的问答都是编的，不含真实答辩内容。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from teams_voice_translator.defense import scene_modes
from teams_voice_translator.defense.qa_bank import QaBankStore, QaEntry
from teams_voice_translator.defense.quick_phrases import (
    CATEGORY_LABELS, DEFAULT_PHRASES, QuickPhrase, QuickPhraseStore, next_phrase_id, parse_phrases,
)
from teams_voice_translator.defense.settings import DEFENSE_DEFAULTS, DefenseSettings


def _wait(app, predicate, rounds=300):
    for _ in range(rounds):
        app.processEvents()
        if predicate():
            return True
        threading.Event().wait(0.01)
    app.processEvents()
    return predicate()


class QuickPhraseTests(unittest.TestCase):
    def test_默认用语够多且完整(self):
        self.assertGreaterEqual(len(DEFAULT_PHRASES), 35)
        ids = [p.id for p in DEFAULT_PHRASES]
        self.assertEqual(len(ids), len(set(ids)))
        for phrase in DEFAULT_PHRASES:
            self.assertIn(phrase.category, CATEGORY_LABELS)
            self.assertTrue(phrase.zh and phrase.en, phrase)
            self.assertLess(len(phrase.en), 450, "单句不能超过一个合成块")
        self.assertTrue(all(any(p.category == c for p in DEFAULT_PHRASES) for c in CATEGORY_LABELS))

    def test_用户点名的四句都在(self):
        zh = " ".join(p.zh for p in DEFAULT_PHRASES)
        for words in ("让我想想", "谢谢教授", "源码", "换个问题"):
            self.assertIn(words, zh)
        self.assertGreaterEqual(sum(p.star for p in DEFAULT_PHRASES), 4)

    def test_存取与容错(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuickPhraseStore(Path(tmp))
            self.assertEqual(len(store.load()), len(DEFAULT_PHRASES), "没存过就用默认")
            store.save([QuickPhrase("x", "hard", "不会", "I don't know.", True)])
            loaded = store.load()
            self.assertEqual([(p.id, p.en, p.star) for p in loaded], [("x", "I don't know.", True)])
            (Path(tmp) / "quick_phrases.json").write_text("{坏", encoding="utf-8")
            self.assertEqual(len(store.load()), len(DEFAULT_PHRASES))
            self.assertEqual(len(store.reset()), len(DEFAULT_PHRASES))

    def test_解析跳过坏条目_未知分类归位_重复id改名(self):
        phrases = parse_phrases({"phrases": [
            {"id": "a", "category": "nope", "zh": "甲", "en": "A."},
            {"id": "a", "category": "hard", "zh": "乙", "en": "B."},
            {"id": "c", "zh": "没英文"},
            "garbage",
        ]})
        self.assertEqual([p.id for p in phrases], ["a", "a-2"])
        self.assertEqual(phrases[0].category, "think")
        self.assertEqual(next_phrase_id(phrases), "u1")


class SceneModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def test_默认设置就是推荐模式(self):
        mode = scene_modes.current_mode(self.settings)
        self.assertIsNotNone(mode, scene_modes.current_values(self.settings))
        self.assertEqual(mode.id, "recommended")

    def test_内置模式用到的模型都在下拉清单里(self):
        from teams_voice_translator.defense.chat_models import USE_QA, USE_TRANSLATE, models_for
        from teams_voice_translator.defense.livetranslate import LIVETRANSLATE_MODELS
        from teams_voice_translator.defense.ui import ASR_MODELS

        asr = {m for m, _ in ASR_MODELS}
        translate = {m for m, _ in models_for(USE_TRANSLATE)}
        qa = {m for m, _ in models_for(USE_QA)}
        live = {m for m, _ in LIVETRANSLATE_MODELS}
        names = set()
        for mode in scene_modes.BUILTIN_MODES:
            self.assertEqual(set(mode.values), set(scene_modes.SCENE_KEYS), mode.id)
            self.assertIn(mode.values["asr_model"], asr)
            self.assertIn(mode.values["committee_asr_model"], asr)
            self.assertIn(mode.values["llm_model"], translate, "翻译模型必须能关掉思考")
            self.assertIn(mode.values["qa_model"], qa)
            self.assertIn(mode.values["livetranslate_model"], live)
            self.assertIn(mode.values["my_translate_engine"], ("llm", "livetranslate"))
            names.add(mode.name)
        self.assertEqual(len(names), len(scene_modes.BUILTIN_MODES))

    def test_备用模式真的换了东西(self):
        recommended = scene_modes.BUILTIN_MODES[0].values
        for mode in scene_modes.BUILTIN_MODES[1:]:
            self.assertNotEqual(mode.values, recommended, mode.id)

    def test_切换_显示改动_识别当前模式(self):
        mode = next(m for m in scene_modes.BUILTIN_MODES if m.id == "legacy_all")
        changes = scene_modes.apply_mode(self.settings, mode)
        keys = {k for k, _o, _n in changes}
        self.assertEqual(keys, {"asr_model", "committee_asr_model", "llm_model", "qa_model"})
        self.assertIn("qwen-plus", scene_modes.describe_changes(changes))
        self.assertEqual(scene_modes.current_mode(self.settings).id, "legacy_all")
        reloaded = DefenseSettings(base_dir=Path(self.tmp.name))
        self.assertEqual(reloaded.get("llm_model"), "qwen-plus", "切换要落盘")
        self.assertEqual(scene_modes.apply_mode(self.settings, mode), [])

    def test_手动改过就是自定义_可以存成我的模式(self):
        self.settings.set("llm_model", "qwen3.7-plus")
        self.assertIsNone(scene_modes.current_mode(self.settings))
        saved = scene_modes.save_custom_mode(self.settings, "我的备用")
        self.assertEqual(scene_modes.current_mode(self.settings).id, saved.id)
        again = scene_modes.save_custom_mode(self.settings, "我的备用")
        self.assertEqual(again.id, saved.id, "同名覆盖")
        self.assertEqual(len(scene_modes.custom_modes(self.settings)), 1)
        self.assertTrue(scene_modes.delete_custom_mode(self.settings, saved.id))
        self.assertEqual(scene_modes.custom_modes(self.settings), [])

    def test_音色下拉与切换(self):
        self.settings.update({
            "tts_voice_id": "v-main", "tts_model": "qwen-audio-3.0-tts-plus",
            "voice_library": [
                {"voice_id": "v-main", "target_model": "qwen-audio-3.0-tts-plus", "name": "主音色"},
                {"voice_id": "v-backup", "target_model": "cosyvoice-v3.5-plus", "name": "备用"},
                {"name": "坏条目"},
            ],
        })
        options = scene_modes.voice_options(self.settings)
        self.assertEqual([o["voice_id"] for o in options], ["v-main", "v-backup"])
        self.assertEqual(scene_modes.apply_voice(self.settings, "v-backup"), ("v-backup", "cosyvoice-v3.5-plus"))
        self.assertEqual(self.settings.get("tts_model"), "cosyvoice-v3.5-plus", "合成模型跟着音色换")
        self.assertIsNone(scene_modes.apply_voice(self.settings, "nope"))

    def test_默认锁定(self):
        self.assertTrue(self.settings.get("scene_locked"))


class PrerendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _make(self, engine):
        from teams_voice_translator.defense.ui import Prerenderer

        pre = Prerenderer(engine)
        finished = []
        pre.finished.connect(finished.append)
        return pre, finished

    def test_跳过已渲染_逐条渲染(self):
        engine = MagicMock()
        engine.answer_ready.side_effect = lambda text: text == "ready"
        pre, finished = self._make(engine)
        self.assertEqual(pre.enqueue(["ready", "a", "b", "a", "", "  "]), 2)
        self.assertTrue(_wait(self.app, lambda: finished))
        self.assertEqual([c.args[0] for c in engine.render_answer.call_args_list], ["a", "b"])
        self.assertIn("2 条", finished[0])
        self.assertFalse(pre.running)
        # 跑完后还能再加
        self.assertEqual(pre.enqueue(["c"]), 1)
        self.assertTrue(_wait(self.app, lambda: len(finished) == 2))

    def test_连续失败三次就停_不反复花钱(self):
        engine = MagicMock()
        engine.answer_ready.return_value = False
        engine.render_answer.side_effect = RuntimeError("no network")
        pre, finished = self._make(engine)
        pre.enqueue([f"t{i}" for i in range(10)])
        self.assertTrue(_wait(self.app, lambda: finished))
        self.assertEqual(engine.render_answer.call_count, 3)
        self.assertIn("暂停", finished[0])
        self.assertEqual(pre.pending(), 0)


class QuickPhrasePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from teams_voice_translator.defense.ui import QuickPhrasePanel

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = False
        self.engine.speak_prepared.return_value = True
        self.panel = QuickPhrasePanel(self.settings, self.engine)

    def test_按分类建标签页_点按钮读英文(self):
        self.assertEqual(self.panel.tabs.count(), len(CATEGORY_LABELS))
        phrase = self.panel.phrases[0]
        self.panel._buttons[phrase.id].click()
        args, kwargs = self.engine.speak_prepared.call_args
        self.assertEqual(args[0], phrase.en)
        self.assertEqual(kwargs["zh"], phrase.zh)
        self.assertIn("等待发声", self.panel.status_label.text())
        self.assertIn(f"0/{len(self.panel.phrases)}", self.panel.ready_label.text())

    def test_编辑后保存并触发预渲染(self):
        changed = []
        self.panel.on_changed = lambda: changed.append(1)
        self.panel.set_phrases([QuickPhrase("u1", "hard", "我不会", "I'm not sure.", True)])
        self.assertEqual(self.panel.tabs.count(), 1)
        self.assertEqual(changed, [1])
        saved = json.loads((Path(self.tmp.name) / "quick_phrases.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["phrases"][0]["en"], "I'm not sure.")
        self.assertEqual([p.id for p in self.panel.starred()], ["u1"])

    def test_编辑窗口_中文生成英文_没英文不让存(self):
        from teams_voice_translator.defense.ui import QuickPhraseDialog

        dialog = QuickPhraseDialog(self.panel, [QuickPhrase("u1", "hard", "我得查一下", "")])
        dialog.calibrator = lambda direction, zh, en, context: "Let me check that."
        dialog.table.selectRow(0)
        from unittest.mock import patch
        with patch("teams_voice_translator.defense.ui.QMessageBox.warning") as warning:
            dialog._save()
            warning.assert_called_once()
        dialog.translate_selected()
        self.assertTrue(_wait(self.app, lambda: dialog.translate_button.isEnabled()))
        result = dialog.result_phrases()
        self.assertEqual(result[0].en, "Let me check that.")
        self.assertEqual(result[0].category, "hard")


class AnswerPopupTests(unittest.TestCase):
    """评委提问 → 弹窗显示问题和预案 → 没改就播预渲染 / 改了中文先校准英文再现场合成。"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from teams_voice_translator.defense.ui import QaPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        QaBankStore(Path(self.tmp.name)).save([
            QaEntry("A1", "How did you choose the loss weights?", "They were fixed for all groups.",
                    answer_zh="所有组都用同一组权重。"),
        ])
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = True
        self.engine.speak_prepared.return_value = True
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.star_phrases_provider = lambda: [QuickPhrase("t1", "think", "让我想想", "Let me think.", True)]
        self.spoken_phrases = []
        self.panel.phrase_player = self.spoken_phrases.append

    def ask(self):
        self.panel.match_chat_factory = lambda: (lambda messages: '{"id": "A1", "confidence": 0.9}')
        self.panel.on_committee_done("How did you pick the loss weights?", "你的损失权重是怎么选的？")
        return _wait(self.app, lambda: self.panel._current_entry is not None)

    def test_命中后弹窗显示问题中英和预案(self):
        self.assertTrue(self.ask())
        popup = self.panel.popup
        self.assertIsNotNone(popup)
        self.assertTrue(popup.isVisible())
        self.assertIn("loss weights", popup.question_en.text())
        self.assertIn("损失权重", popup.question_zh.text())
        self.assertIn("预案问题", popup.source_label.text())
        self.assertIs(self.panel.answer_split.parent(), popup, "回答区整块搬进弹窗")
        self.assertTrue(self.panel.answer_placeholder.isVisibleTo(self.panel))
        # 应急快捷语一键可点
        buttons = [w for w in popup.star_box.findChildren(type(popup.play_button)) if w.text() == "让我想想"]
        self.assertEqual(len(buttons), 1)
        buttons[0].click()
        self.assertEqual(self.spoken_phrases[0].en, "Let me think.")

    def test_没改直接回答_播放预案原文(self):
        self.assertTrue(self.ask())
        self.panel.popup.play_button.click()
        self.assertEqual(self.engine.speak_prepared.call_args.args[0], "They were fixed for all groups.")
        self.assertEqual(self.engine.speak_prepared.call_args.kwargs["zh"], "所有组都用同一组权重。")

    def test_改了中文_先校准英文再朗读(self):
        self.assertTrue(self.ask())
        calls = []

        def calibrator():
            def run(direction, edited, other, context):
                calls.append((direction, edited))
                return "They were fixed, and I did not tune them per group."
            return run

        self.panel.calibrator_factory = calibrator
        self.panel.answer_zh.setText("权重是固定的，没有按组单独调。")
        self.panel._on_zh_edited(user=True)
        self.panel.popup.play_button.click()
        self.assertTrue(_wait(self.app, lambda: self.engine.speak_prepared.called))
        self.assertEqual(calls, [("zh2en", "权重是固定的，没有按组单独调。")])
        self.assertEqual(self.engine.speak_prepared.call_args.args[0],
                         "They were fixed, and I did not tune them per group.")
        self.assertEqual(self.engine.speak_prepared.call_args.kwargs["zh"], "权重是固定的，没有按组单独调。")
        self.assertFalse(self.panel._zh_dirty)

    def test_校准失败不朗读_再点一次读现在的英文(self):
        self.assertTrue(self.ask())

        def calibrator():
            def run(*_args):
                raise RuntimeError("timeout")
            return run

        self.panel.calibrator_factory = calibrator
        self.panel.answer_zh.setText("改过的中文")
        self.panel._on_zh_edited(user=True)
        self.panel._play_answer()
        self.assertTrue(_wait(self.app, lambda: "失败" in self.panel.status_label.text()))
        self.engine.speak_prepared.assert_not_called()
        self.panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_args.args[0], "They were fixed for all groups.")

    def test_删除中文仍能朗读英文(self):
        self.assertTrue(self.ask())
        self.panel.answer_zh.clear()
        self.panel._on_zh_edited(user=True)
        self.panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_args.args[0], "They were fixed for all groups.")
        self.assertFalse(self.panel._zh_dirty)

    def test_重新生成失败保留上一条可朗读回答(self):
        self.assertTrue(self.ask())

        class FailingAdvisor:
            model = "offline-test"

            def generate(self, *_args, **_kwargs):
                raise RuntimeError("模拟超时")

        self.panel.advisor_factory = lambda **_kwargs: FailingAdvisor()
        self.panel.generate("A new question?")
        self.assertTrue(_wait(self.app, lambda: "生成失败" in self.panel.status_label.text()))
        self.assertEqual(self.panel.answer_en.text(), "They were fixed for all groups.")
        self.assertIn("上一条", self.panel.play_button.text())
        self.panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_args.args[0], "They were fixed for all groups.")

    def test_英文空白时禁用朗读(self):
        self.assertTrue(self.ask())
        self.panel.answer_en.clear()
        self.assertFalse(self.panel.play_button.isEnabled())
        self.panel._play_answer()
        self.engine.speak_prepared.assert_not_called()

    def test_不回答_收起后回答区回到面板(self):
        self.assertTrue(self.ask())
        self.panel.popup.skip_button.click()
        self.assertFalse(self.panel.popup.isVisible())
        self.assertIs(self.panel.answer_split.parent(), self.panel.answer_host)
        self.assertFalse(self.panel.answer_placeholder.isVisibleTo(self.panel))
        self.engine.speak_prepared.assert_not_called()
        self.assertTrue(self.settings.get("qa_popup_geometry"), "记住弹窗位置")

    def test_关掉弹窗开关就不弹(self):
        self.panel.popup_check.setChecked(False)
        self.assertTrue(self.ask())
        self.assertTrue(self.panel.popup is None or not self.panel.popup.isVisible())

    def test_手动搜预案不弹窗(self):
        self.panel.search_edit.setText("loss weights")
        self.panel._search_bank()
        self.assertTrue(self.panel.popup is None or not self.panel.popup.isVisible())


class SceneBarWindowTests(unittest.TestCase):
    """首页情景模式：默认锁住；解锁后切换立即生效并自动重新锁上。"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import DefenseWindow

        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.settings = DefenseSettings(base_dir=Path(cls._tmp.name))
        cls.window = DefenseWindow(cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls._tmp.cleanup()

    def setUp(self):
        scene_modes.apply_mode(self.settings, scene_modes.BUILTIN_MODES[0])
        self.settings.set("scene_locked", True)
        self.window._refresh_scene_bar()
        self.window.engine.apply_settings = MagicMock()

    def _index_of(self, mode_id):
        return self.window.scene_combo.findData(mode_id)

    def test_默认锁定_下拉不可用(self):
        self.assertFalse(self.window.scene_combo.isEnabled())
        self.assertFalse(self.window.voice_scene_combo.isEnabled())
        self.assertIn("已锁定", self.window.scene_lock_button.text())
        self.assertEqual(self.window.scene_combo.currentData(), "recommended")

    def test_锁着时即使收到选择也不切换(self):
        self.window._on_scene_chosen(self._index_of("legacy_all"))
        self.assertEqual(self.settings.get("llm_model"), "qwen3.8-flash")
        self.window.engine.apply_settings.assert_not_called()

    def test_解锁_切换生效_自动重新锁上(self):
        self.window.scene_lock_button.click()
        self.assertTrue(self.window.scene_combo.isEnabled())
        self.window._on_scene_chosen(self._index_of("deep_qa"))
        self.assertEqual(self.settings.get("qa_model"), "qwen3.8-max")
        self.assertTrue(self.settings.get("qa_thinking"))
        self.window.engine.apply_settings.assert_called_once()
        self.assertEqual(self.window.qa_panel.model_combo.currentData(), "qwen3.8-max", "助答下拉跟着变")
        self.assertTrue(self.window.qa_panel.thinking_check.isChecked())
        self.assertTrue(self.settings.get("scene_locked"))
        self.assertFalse(self.window.scene_combo.isEnabled())
        self.assertEqual(self.window.scene_combo.currentData(), "deep_qa")

    def test_面板上手动改模型显示自定义(self):
        combo = self.window.qa_panel.model_combo
        combo.setCurrentIndex(combo.findData("qwen3.7-plus"))
        self.assertEqual(self.window.scene_combo.currentData(), scene_modes.CUSTOM_ID)

    def test_快捷用语面板和预渲染已接好(self):
        window = self.window
        self.assertIsNotNone(window.phrase_panel)
        self.assertIs(window.qa_panel.prerenderer, window.prerenderer)
        self.assertEqual(window._auto_prerender(), 0, "没配音色时不自动渲染（不花钱）")

    def test_批量预渲染须主动开启且关闭会取消队列(self):
        panel = self.window.phrase_panel
        panel.auto_render_check.setChecked(False)
        self.assertFalse(DEFENSE_DEFAULTS["prerender_auto"])
        self.assertFalse(self.window._prerender_allowed())
        self.assertIs(panel.on_auto_render_disabled.__self__, self.window.prerenderer)
        with patch.object(self.window.prerenderer, "stop") as stop:
            with patch.object(panel, "on_auto_render_disabled", stop):
                panel.auto_render_check.setChecked(True)
                panel.auto_render_check.setChecked(False)
            stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
