import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from teams_voice_translator.ui import MainWindow


class VoiceModelUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["APPDATA"] = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_tts_model_switch_saves_and_restores_voice(self):
        w = self.window
        models = [w.tts_model.itemText(i) for i in range(w.tts_model.count())]
        if len(models) < 2:
            self.skipTest("TTS 模型下拉框不足 2 项")
        first, second = models[0], models[1]

        w.tts_model.setCurrentText(first)
        w.voice.setText("voice-for-" + first)
        w.on_tts_model_changed(first)

        w.tts_model.setCurrentText(second)
        w.on_tts_model_changed(second)
        self.assertEqual(w.settings.values["tts_voices"][first], "voice-for-" + first)
        self.assertEqual(w._active_tts_model, second)

        w.voice.setText("voice-for-" + second)
        w.tts_model.setCurrentText(first)
        w.on_tts_model_changed(first)
        self.assertEqual(w.voice.text(), "voice-for-" + first)
        self.assertEqual(w.settings.values["tts_voices"][second], "voice-for-" + second)

    @patch("teams_voice_translator.ui.QMessageBox.information")
    def test_clone_finished_switches_model_and_saves_voice(self, _mock_box):
        w = self.window
        target = "qwen3-tts-vc-realtime-2026-01-15"
        if w.tts_model.findText(target) < 0:
            self.skipTest(f"当前 TTS 模型列表不包含 {target}")
        w.tts_model.setCurrentText("qwen-audio-3.0-tts-plus")
        w.on_tts_model_changed("qwen-audio-3.0-tts-plus")
        w.clone_target_model = target

        w.on_clone_finished(True, "cloned-voice-123", "")

        self.assertEqual(w.tts_model.currentText(), target)
        self.assertEqual(w.voice.text(), "cloned-voice-123")
        self.assertEqual(w.settings.values["tts_voices"][target], "cloned-voice-123")

    def test_profile_saves_and_loads_tts_model_and_voices(self):
        w = self.window
        model = w.tts_model.itemText(0)
        w.tts_model.setCurrentText(model)
        w.voice.setText("profile-voice")
        w.on_tts_model_changed(model)
        w.translation_domain.setText("MBA 课堂")
        w.profile_name.setCurrentText("MBA 课堂")

        w.save_selected_profile()
        profile = w.profile_store.get("MBA 课堂")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["tts_model"], model)
        self.assertEqual(profile["tts_voices"][model], "profile-voice")

        w.tts_model.setCurrentText(w.tts_model.itemText(1))
        w.on_tts_model_changed(w.tts_model.currentText())
        w.voice.setText("other-voice")
        w.translation_domain.setText("其他领域")

        w._load_profile("MBA 课堂")
        self.assertEqual(w.tts_model.currentText(), model)
        self.assertEqual(w.voice.text(), "profile-voice")
        self.assertEqual(w.translation_domain.text(), "MBA 课堂")


if __name__ == "__main__":
    unittest.main()
