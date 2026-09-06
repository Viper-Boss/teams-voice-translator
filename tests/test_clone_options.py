import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from teams_voice_translator.api_payloads import build_voice_clone_payload
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import CloneVoiceDialog
from teams_voice_translator.voice_sample import voice_sample_duration, normalized_voice_sample


class CloneOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        self.no_oss = patch.object(CloneVoiceDialog, "_maybe_auto_probe_oss")
        self.no_oss.start(); self.addCleanup(self.no_oss.stop)

    def dialog(self, model="cosyvoice-v3.5-plus"):
        dialog = CloneVoiceDialog(None, self.settings, model)
        self.addCleanup(dialog.close)
        return dialog

    def test_duration_defaults_to_actual_length_and_caps_at_thirty(self):
        dialog = self.dialog()
        for duration, expected in ((3.5, 3.5), (13.125, 13.125), (30, 30), (65.7, 30)):
            dialog._set_sample(Path("test.wav"), duration)
            self.assertEqual(dialog.sample_length_spin.value(), expected)
        self.assertFalse(dialog.preprocess_check.isChecked())
        self.assertFalse(dialog.volume_normalization_check.isChecked())

    def test_models_only_enable_supported_options(self):
        qwen = self.dialog("qwen3-tts-vc-2026-01-22")
        self.assertFalse(qwen.preprocess_check.isEnabled())
        self.assertFalse(qwen.volume_normalization_check.isEnabled())
        older = self.dialog("cosyvoice-v3-plus")
        self.assertFalse(older.preprocess_check.isEnabled())
        self.assertTrue(older.volume_normalization_check.isEnabled())

    def test_sample_duration_ignores_wheel_even_when_focused(self):
        from PySide6.QtCore import QPointF, QPoint, Qt
        from PySide6.QtGui import QWheelEvent
        dialog = self.dialog()
        dialog._set_sample(Path('test.wav'), 30)
        dialog.show()
        spin = dialog.sample_length_spin
        spin.setFocus()
        self.app.processEvents()
        for delta in (-120, 120):
            event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, delta),
                                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
            self.app.sendEvent(spin, event)
            self.assertEqual(spin.value(), 30)

    def test_selected_options_are_snapshotted_before_worker_starts(self):
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 18.125)
        dialog.preprocess_check.setChecked(True)
        dialog.volume_normalization_check.setChecked(True)
        with patch.object(dialog, "_collect_oss_values", return_value={}), \
             patch("teams_voice_translator.defense.ui.threading.Thread") as thread:
            dialog._start_clone()
        options = thread.call_args.kwargs["args"][-1]
        self.assertEqual(options, {"max_seconds": 18.125, "min_seconds": 3.0,
                                  "preprocess": True, "volume_normalization": True})
        dialog.sample_length_spin.setValue(10)
        self.assertEqual(options["max_seconds"], 18.125)

    def test_http_payload_types_and_unsupported_fields(self):
        args = dict(prefix="sample", audio_url="https://example.com/a.wav", max_seconds=18.125,
                    preprocess=True, volume_normalization=True)
        payload = build_voice_clone_payload(target_model="cosyvoice-v3.5-plus", **args)["input"]
        self.assertEqual(payload["max_prompt_audio_length"], 18.125)
        self.assertIs(payload["enable_preprocess"], True)
        self.assertEqual(payload["enable_volume_normalization"], "true")
        older = build_voice_clone_payload(target_model="cosyvoice-v3-plus", **args)["input"]
        self.assertNotIn("max_prompt_audio_length", older)
        self.assertNotIn("enable_preprocess", older)
        self.assertEqual(older["enable_volume_normalization"], "true")
        with self.assertRaises(ValueError):
            build_voice_clone_payload(target_model="cosyvoice-v3.5-plus", **dict(args, max_seconds=31))

    def test_original_duration_is_read_before_thirty_second_conversion(self):
        source = Path(self.tmp.name) / "source.wav"
        for duration, expected in ((3.5, 3.5), (18.125, 18.125), (35, 30)):
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
                handle.writeframes(bytes(int(duration * 16000) * 2))
            self.assertEqual(voice_sample_duration(source), duration)
            with normalized_voice_sample(source, max_seconds=min(duration, 30), min_seconds=3) as result:
                with wave.open(str(result)) as handle:
                    self.assertAlmostEqual(handle.getnframes() / handle.getframerate(), expected, places=3)

    def test_stale_file_inspection_cannot_replace_new_selection(self):
        dialog = self.dialog()
        dialog._clear_sample("old")
        old = dialog._sample_generation
        dialog._clear_sample("new")
        dialog._on_sample_checked(old, "old.wav", 12, "")
        self.assertIsNone(dialog._sample_path)


if __name__ == "__main__": unittest.main()
