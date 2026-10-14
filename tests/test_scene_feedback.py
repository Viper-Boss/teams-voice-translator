"""Scene background stays independent; Q&A actions visibly acknowledge completion."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_qa_review as qa_fixtures
from teams_voice_translator.defense.pipeline import CachedUtterance, DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings


class SceneBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = DefenseSettings(Path(self.temp.name))
        self.settings.set("ambience_mode", "rain")
        self.engine = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(self.engine.shutdown)

    def test_f8_and_live_monitor_mute_do_not_silence_scene_background(self):
        self.engine._session_active = True
        self.engine._direct_active = True
        self.settings.set("live_monitor_enabled", False)
        written = threading.Event()
        stop = threading.Event()
        def write(pcm):
            self.assertTrue(any(pcm))
            written.set()
            stop.set()
        self.engine.player = Mock(write=write)
        worker = threading.Thread(target=self.engine._ambience_bed_loop, args=(stop,))
        worker.start()
        try:
            self.assertTrue(written.wait(3), "F8 must not gate the background layer")
        finally:
            stop.set()
            worker.join(2)

    def test_pause_and_esc_do_not_cancel_background_layer(self):
        self.engine._session_active = True
        self.engine.player = Mock()
        stop = self.engine._ambience_stop
        self.engine.pause_speech()
        self.assertFalse(stop.is_set())
        self.engine.interrupt_speech()
        self.assertFalse(stop.is_set())
        self.assertEqual(self.settings.get("ambience_mode"), "rain")

    def test_cached_answer_honors_scene_switch_and_volume_during_playback(self):
        raw = b"\x10\x00" * 960
        outputs = []
        def tone():
            mode, volume = self.settings.get("ambience_mode"), self.settings.get("ambience_volume")
            return SimpleNamespace(mix=lambda pcm: pcm if mode == "off" else bytes([volume, 0]) * (len(pcm)//2))
        def write(pcm):
            outputs.append(pcm)
            if len(outputs) == 1:
                self.settings.set("ambience_mode", "off")
            elif len(outputs) == 2:
                self.settings.update({"ambience_mode": "rain", "ambience_volume": 60})
        self.settings.set("ambience_volume", 20)
        self.engine.player = Mock(write=write)
        with patch.object(self.engine, "_ensure_player"), \
                patch.object(self.engine, "_current_room_tone", side_effect=tone), \
                patch.object(self.engine, "_ambience_bed_on", return_value=True):
            self.engine._play_cached(CachedUtterance(raw * 3), "Prepared answer.")
            self.engine._cached_thread.join(3)
        self.assertFalse(self.engine._cached_thread.is_alive())
        self.assertEqual(outputs, [b"\x14\x00" * 960, raw, b"\x3c\x00" * 960])

    def test_streaming_answer_uses_global_background_without_dirtying_clean_audio(self):
        raw = b"\x10\x00" * 960
        self.engine.player = Mock()
        self.engine._on_first_audio("My answer.")
        self.engine._on_tts_audio(raw)
        self.assertNotEqual(self.engine.player.write.call_args.args[0], raw)
        self.assertEqual(bytes(self.engine._live_utterance["buf"]), raw)
        self.settings.set("ambience_mode", "off")
        self.engine._on_tts_audio(raw)
        self.assertEqual(self.engine.player.write.call_args.args[0], raw)


class BankActionFeedbackTests(unittest.TestCase):
    setUp = qa_fixtures.UiTests.setUp

    def test_transfer_has_button_and_status_feedback_without_playing_audio(self):
        self.dialog.table.selectRow(0)
        self.dialog.to_panel_button.click()
        self.assertEqual(self.dialog.to_panel_button.text(), "✓ 已放入助答面板")
        self.assertIn("已放入助答面板", self.dialog.detail_status.text())
        self.assertEqual(self.panel.answer_en.text(), self.panel.bank[0].answer)
        self.engine.speak_prepared.assert_not_called()

    def test_transfer_failure_is_visible_and_does_not_claim_success(self):
        self.dialog.table.selectRow(0)
        with patch.object(self.panel, "show_bank_entry", side_effect=RuntimeError("test failure")):
            self.dialog.to_panel_button.click()
        self.assertIn("失败", self.dialog.detail_status.text())
        self.assertNotIn("已放入", self.dialog.to_panel_button.text())

    def test_render_progress_is_visible_next_to_current_entry_buttons(self):
        self.dialog.table.selectRow(0)
        entry = self.panel.bank[0]
        self.dialog._on_render_progress(entry.id, "正在合成…")
        self.assertIn("正在合成", self.dialog.detail_status.text())

    def test_editing_resets_prior_success_button(self):
        self.dialog.table.selectRow(0)
        self.dialog.to_panel_button.click()
        self.dialog.a_en_edit.setText("I changed this answer.")
        self.assertEqual(self.dialog.to_panel_button.text(), "放到助答面板")
        self.dialog.to_panel_button.click()
        self.assertIn("尚未保存", self.dialog.detail_status.text())


if __name__ == "__main__":
    unittest.main()
