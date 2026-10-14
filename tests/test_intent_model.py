import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication
from teams_voice_translator.defense.channels import BackupChannel, primary_endpoint
from teams_voice_translator.defense.question_intent import INTENT_MODEL, INTENT_MODELS, parse_intent
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import QaPanel, DefenseWindow


REPLY = json.dumps(dict(complete=True, is_question=True,
                       intent="Explain the assumptions.", subquestions=["Why these assumptions?"], reason="完整"))


class IntentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = DefenseSettings(Path(self.temp.name))
        self.settings.shared.update({"workspace_id": "offline-test"})
        self.settings.values.update(qa_popup=False, qa_bank_enabled=False)
        self.panel = QaPanel(self.settings, MagicMock())
        self.addCleanup(self.panel.shutdown)
        self.panel.generate = MagicMock()

    def select(self, model):
        index = self.panel.intent_model_combo.findData(model)
        self.assertGreaterEqual(index, 0)
        self.panel.intent_model_combo.setCurrentIndex(index)

    def test_default_strong_model_and_clear_order(self):
        self.assertEqual(INTENT_MODEL, "qwen3.8-max")
        self.assertEqual([self.panel.intent_model_combo.itemData(i) for i in range(3)],
                         ["qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash"])
        self.assertEqual(self.panel.intent_model_combo.currentData(), INTENT_MODEL)
        self.assertFalse(self.panel.intent_model_combo.isEditable())
        self.assertEqual(self.settings.get("llm_model"), "qwen3.8-flash")
        self.assertEqual(self.settings.get("qa_model"), "qwen3.8-flash")

    def test_selection_and_thinking_persist_without_changing_translation_or_answer(self):
        self.select("qwen3.7-plus")
        self.panel.intent_thinking_check.setChecked(True)
        saved = DefenseSettings(Path(self.temp.name))
        self.assertEqual(saved.get("qa_intent_model"), "qwen3.7-plus")
        self.assertTrue(saved.get("qa_intent_thinking"))
        self.assertEqual(saved.get("llm_model"), "qwen3.8-flash")
        self.assertEqual(saved.get("qa_model"), "qwen3.8-flash")
        self.assertFalse(saved.get("qa_thinking"))

    def test_actual_request_payload_uses_each_selected_model_and_separate_thinking(self):
        from teams_voice_translator.defense.translator import ContextTranslator
        for model, _ in INTENT_MODELS:
            self.select(model)
            for thinking in (False, True):
                self.panel.intent_thinking_check.setChecked(thinking)
                cancel = threading.Event()
                with patch.object(self.settings, "get_api_key", return_value="offline-key"), \
                     patch("teams_voice_translator.defense.ui.backup_channel", return_value=None), \
                     patch.object(ContextTranslator, "_post_stream", return_value=REPLY) as post:
                    result = self.panel._default_intent_chat(cancel)([{"role": "user", "content": "Why?"}])
                self.assertEqual(result, REPLY)
                payload = post.call_args.args[0]
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["enable_thinking"], thinking)
                self.assertIs(post.call_args.args[2], cancel)

    def test_model_change_discards_old_understanding_but_retains_all_segments(self):
        p = self.panel
        p.collect_committee_segment(1, "The first part of my question.")
        p.collect_committee_segment(2, "Why did you choose that assumption?")
        old_revision = p._question_revision
        self.select("qwen3.8-flash")
        self.assertGreater(p._question_revision, old_revision)
        p._on_question_intent_ready((old_revision, False, parse_intent(REPLY), ""))
        p.generate.assert_not_called()
        self.assertEqual(len(p._question_parts), 2)
        self.assertTrue(p._question_collect_timer.isActive())

    def test_disabled_understanding_disables_only_its_controls(self):
        p = self.panel
        p.intent_check.setChecked(False)
        self.assertFalse(p.intent_model_combo.isEnabled())
        self.assertFalse(p.intent_thinking_check.isEnabled())
        self.assertTrue(p.model_combo.isEnabled())
        self.assertTrue(p.thinking_check.isEnabled())
        p.collect_committee_segment(1, "Why these assumptions?")
        p.submit_collected_question()
        p.generate.assert_called_once_with("Why these assumptions?")

    def test_scene_and_answer_settings_do_not_override_understanding_choice(self):
        self.select("qwen3.7-plus")
        self.panel.intent_thinking_check.setChecked(True)
        self.settings.update({"qa_model": "qwen-plus", "qa_thinking": False, "llm_model": "qwen-plus"})
        self.panel.sync_from_settings()
        self.assertEqual(self.panel.model_combo.currentData(), "qwen-plus")
        self.assertEqual(self.panel.intent_model_combo.currentData(), "qwen3.7-plus")
        self.assertTrue(self.panel.intent_thinking_check.isChecked())

    def test_fast_backup_cannot_replace_independently_selected_strong_model(self):
        backup = BackupChannel("https://offline.example/chat", "offline", "offline", model="qwen3.8-flash")
        with patch.object(self.settings, "get_api_key", return_value="offline-key"), \
             patch("teams_voice_translator.defense.ui.backup_channel", return_value=backup), \
             patch("teams_voice_translator.defense.ui.ContextTranslator") as factory:
            self.panel._default_intent_chat(threading.Event())([])
        self.assertEqual(factory.call_args.kwargs["model"], "qwen3.8-max")
        self.assertEqual(factory.call_args.kwargs["backup"].model, "")
        self.assertEqual(backup.model, "qwen3.8-flash", "Do not mutate the global backup config")
        identical = BackupChannel(primary_endpoint(self.settings.workspace_id), "offline", "same", model="qwen3.8-flash")
        with patch.object(self.settings, "get_api_key", return_value="offline-key"), \
             patch("teams_voice_translator.defense.ui.backup_channel", return_value=identical), \
             patch("teams_voice_translator.defense.ui.ContextTranslator") as factory:
            self.panel._default_intent_chat(threading.Event())([])
        self.assertIsNone(factory.call_args.kwargs["backup"])

    def test_slow_understanding_does_not_block_committee_subtitle_translation(self):
        window = DefenseWindow(self.settings)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(window.close)
        self.addCleanup(release.set)
        def chat(messages):
            entered.set()
            release.wait(2)
            return REPLY
        p = window.qa_panel
        p.generate = MagicMock()
        p.intent_chat_factory = lambda cancel: chat
        window._on_committee_recognized(7, "Why these assumptions?")
        p.submit_collected_question()
        self.assertTrue(entered.wait(1))
        window._on_committee_done(7, "Why these assumptions?", "为什么采用这些假设？")
        self.assertEqual(window.live_committee_zh.text(), "为什么采用这些假设？")
        self.assertEqual(p._question_parts[7][1], "为什么采用这些假设？")
        p.generate.assert_not_called()
        release.set()
        deadline = time.monotonic() + 2
        while not p.generate.called and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(p.generate.called)


if __name__ == "__main__":
    unittest.main()
