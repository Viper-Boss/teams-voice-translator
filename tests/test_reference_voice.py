import base64
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from test_batch17 import _Base, _wait
import test_qa_bank as fixtures
from teams_voice_translator.defense.speech_policy import speech_settings, NATURAL_INSTRUCTION
from teams_voice_translator.defense.ui import VoiceCompareDialog, ScriptDeskPanel
from test_tts_result_audio import ResultResponse


class ReferenceVoiceUiTests(_Base):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.settings.shared.update({"workspace_id": "fake-workspace"})
        self.plus = {"name": "A original", "target_model": "qwen-audio-3.0-tts-plus", "voice_id": "qwen-audio-3.0-tts-plus-original"}
        self.flash = {"name": "B different", "target_model": "qwen-audio-3.1-tts-flash", "voice_id": "qwen-audio-3.1-tts-flash-different"}
        self.settings.update({"tts_model": self.flash["target_model"], "tts_voice_id": self.flash["voice_id"],
            "prerender_tts_model": self.plus["target_model"], "prerender_voice_id": self.plus["voice_id"],
            "voice_library": [self.flash, self.plus], "backup_tts_enabled": True})

    def test_a_defaults_to_render_voice_and_apply_copies_original_parameters_without_synthesis(self):
        dialog = VoiceCompareDialog(None, self.settings)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.voices[0].currentData()["voice_id"], self.plus["voice_id"])
        self.assertEqual(dialog.voices[1].currentData()["voice_id"], self.flash["voice_id"])
        applied = []
        dialog.voice_applied.connect(lambda: applied.append(True))
        with patch("teams_voice_translator.defense.ui.voice_probe") as synth:
            dialog.apply_a_button.click()
        synth.assert_not_called()
        self.assertTrue(applied)
        self.assertEqual(self.settings.get("tts_model"), self.plus["target_model"])
        self.assertEqual(self.settings.get("tts_voice_id"), self.plus["voice_id"])
        self.assertEqual(self.settings.get("prerender_voice_id"), self.plus["voice_id"])
        self.assertEqual(self.settings.get("prerender_instruction"), "none")
        self.assertFalse(self.settings.get("backup_tts_enabled"))
        self.assertEqual(speech_settings(self.settings)["tts_instruction"], "")

    def test_qwen_clone_default_preserves_reference_while_explicit_instruction_is_respected(self):
        self.assertEqual(speech_settings(self.settings)["tts_instruction"], "")
        self.settings.set("tts_instruction", "Please sound thoughtful.")
        self.assertEqual(speech_settings(self.settings)["tts_instruction"], "Please sound thoughtful.")
        self.assertEqual(speech_settings(self.settings, model="qwen-audio-3.0-tts-plus", voice=self.plus["voice_id"])["voice"], self.plus["voice_id"])

    def test_busy_apply_cannot_change_voice_during_comparison(self):
        dialog = VoiceCompareDialog(None, self.settings)
        self.addCleanup(dialog.close)
        dialog._busy(True)
        dialog._apply_a_voice()
        self.assertEqual(self.settings.get("tts_voice_id"), self.flash["voice_id"])
        self.assertFalse(dialog.apply_a_button.isEnabled())

    def test_default_b_flash_reaches_http_without_a_delivery_instruction(self):
        dialog = VoiceCompareDialog(None, self.settings)
        self.addCleanup(dialog.close)
        pcm = b"\x01\x00" * 240
        response = ResultResponse([{"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}])
        self.settings.set("tts_instruction", NATURAL_INSTRUCTION)
        with patch.object(self.settings, "get_api_key", return_value="fake-key"), \
                patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post, \
                patch("teams_voice_translator.aliyun.requests.get") as download, \
                patch("teams_voice_translator.audio.MultiOutputPlayer", return_value=MagicMock()):
            dialog.b_button.click()
            self.assertTrue(_wait(lambda: not dialog._is_busy))
        post.assert_called_once()
        download.assert_not_called()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], self.flash["target_model"])
        self.assertEqual(payload["input"]["voice"], self.flash["voice_id"])
        self.assertEqual(payload["input"]["text"], dialog.text_edit.toPlainText())
        self.assertNotIn("instruction", payload["input"])
        self.assertNotIn("instructions", payload["input"])
        self.assertNotIn("enable_ssml", payload["input"])
        self.assertEqual(dialog._cached_audio[1], pcm)
        self.assertIn("无（复刻原声）", dialog.parameters_label.text())

    def test_plus_a_b_and_manuscript_use_identical_http_parameters(self):
        from teams_voice_translator.aliyun import BailianClient
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        self.settings.update({"tts_model": self.plus["target_model"], "tts_voice_id": self.plus["voice_id"],
            "prerender_tts_model": "", "prerender_voice_id": "", "prerender_instruction": "b",
            "humanize_level": "natural"})
        dialog = VoiceCompareDialog(None, self.settings)
        self.addCleanup(dialog.close)
        engine = DefenseEngine(self.settings, EngineCallbacks())
        engine.client = BailianClient("fake-key", "fake-workspace")
        text = "This concludes my report."
        dialog.text_edit.setPlainText(text)
        pcm = b"\x01\x00" * 240
        event = {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}
        with patch.object(self.settings, "get_api_key", return_value="fake-key"), \
                patch("teams_voice_translator.aliyun.requests.post", side_effect=lambda *a, **k: ResultResponse([event])) as post, \
                patch("teams_voice_translator.audio.MultiOutputPlayer", return_value=MagicMock()), \
                patch("teams_voice_translator.defense.pipeline.humanize_text", return_value="Um, altered text.") as humanize:
            dialog.a_button.click()
            self.assertTrue(_wait(lambda: not dialog._is_busy))
            dialog.b_button.click()
            self.assertTrue(_wait(lambda: not dialog._is_busy))
            engine.render_script_group([text], "en")
        humanize.assert_not_called()
        self.assertEqual(post.call_count, 3)
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0], payloads[2])
        self.assertEqual(engine.answer_settings()["tts_instruction"], "")
        self.assertEqual(engine._tts_settings(), engine.script_settings("en"))

    def test_other_model_delivery_and_explicit_qwen_instruction_still_work(self):
        from teams_voice_translator.defense.speech_policy import natural_delivery_instruction
        self.assertEqual(natural_delivery_instruction("cosyvoice-v3.5-plus"), NATURAL_INSTRUCTION)
        self.settings.set("tts_instruction", "Keep my original accent and speak more quietly.")
        self.assertEqual(speech_settings(self.settings)["tts_instruction"], self.settings.get("tts_instruction"))


class ReferenceVoiceCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
    setUp = fixtures.EnginePreparedAnswerTests.setUp
    def test_original_mode_sends_unmodified_text_and_blank_instruction(self):
        self.settings.update({"prerender_instruction": "none", "humanize_level": "natural", "tts_rate": 1.2})
        values = self.engine.script_settings("en")
        self.assertEqual(values["tts_instruction"], "")
        self.assertEqual((values["tts_rate"], values["tts_pitch"]), (1.0, 1.0))
        with patch("teams_voice_translator.defense.pipeline.humanize_text") as humanize:
            self.assertEqual(self.engine._humanize_prerender("My original text.", "en"), "My original text.")
        humanize.assert_not_called()

    def test_follow_mode_can_still_use_explicit_text_humanization(self):
        self.settings.update({"prerender_instruction": "follow", "humanize_level": "natural",
            "tts_model": "qwen-audio-3.0-tts-plus"})
        with patch("teams_voice_translator.defense.pipeline.humanize_text", return_value="Um, my text.") as humanize:
            self.assertEqual(self.engine._humanize_prerender("My text.", "en"), "Um, my text.")
        humanize.assert_called_once()

    def test_rerender_original_keeps_old_b_cache_and_retires_custom_version_only_after_success(self):
        from teams_voice_translator.defense.ui import ScriptDeskPanel
        from teams_voice_translator.defense.tts_cache import cache_key
        self.settings.update({"prerender_instruction": "b", "humanize_level": "off"})
        text = "This concludes my report. Thank you for listening."
        bad, good = b"\x01\x00" * 2400, b"\x02\x00" * 2400
        # Simulate the old B policy so the saved paid audio differs from the
        # new original-voice parameters, even though model/voice are unchanged.
        with patch("teams_voice_translator.defense.pipeline.natural_delivery_instruction", return_value=NATURAL_INSTRUCTION), \
                patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=bad):
            self.engine.render_script_group(["This concludes my report.", "Thank you for listening."], "en")
            self.engine.render_script_override("This concludes my report.", "en", instruction=NATURAL_INSTRUCTION)
            old_group = cache_key(text, self.engine.script_settings("en"))
        old_override = self.engine.script_overrides.get(self.engine._override_entry("This concludes my report.", "en")).partition("\t")[0]
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.source_edit.setPlainText(text)
        panel._split_clicked()
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=good) as synth:
            panel.reference_render_button.click()
            self.assertTrue(_wait(lambda: not panel._rendering))
        self.assertEqual(synth.call_count, 1)
        self.assertEqual(synth.call_args.args[1]["tts_instruction"], "")
        self.assertEqual(synth.call_args.args[2], text)
        self.assertTrue(self.engine.disk_cache.has(old_group))
        self.assertTrue(self.engine.disk_cache.has(old_override))
        self.assertIsNone(self.engine.script_override("This concludes my report.", "en"))
        self.assertEqual(self.engine.script_group_pcm(["This concludes my report.", "Thank you for listening."], "en"), good)

    def test_failure_preserves_old_override_and_all_paid_audio(self):
        self.settings.set("prerender_instruction", "b")
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", return_value=b"\x01\x00" * 2400):
            self.engine.render_script_override("Thank you for listening.", "en")
        old = self.engine.script_overrides.get(self.engine._override_entry("Thank you for listening.", "en"))
        panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(panel.shutdown)
        panel.source_edit.setPlainText("Thank you for listening.")
        panel._split_clicked()
        with patch("teams_voice_translator.defense.tts_session.synthesize_once", side_effect=RuntimeError("offline")):
            panel.reference_render_button.click()
            self.assertTrue(_wait(lambda: not panel._rendering))
        self.assertEqual(self.engine.script_overrides.get(self.engine._override_entry("Thank you for listening.", "en")), old)
        self.assertTrue(self.engine.disk_cache.has(old.partition("\t")[0]))
