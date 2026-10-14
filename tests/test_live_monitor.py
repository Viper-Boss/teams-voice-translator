import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from teams_voice_translator.audio import MultiOutputPlayer, DirectAudioBridge
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings


class LiveMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = DefenseSettings(Path(self.tmp.name))
        self.engine = DefenseEngine(self.settings, EngineCallbacks())

    def tearDown(self):
        self.engine.shutdown()
        self.tmp.cleanup()

    def test_live_translation_muted_but_typed_and_other_tts_stay_audible(self):
        self.settings.set('live_monitor_enabled', False)
        self.engine._on_my_segment('现场中文')
        live_job = self.engine._my_queue.get_nowait()
        self.engine._on_my_live_translation('同传中文', 'Live translation.')
        simultaneous_job = self.engine._my_queue.get_nowait()
        with patch.object(self.engine, '_submit', side_effect=lambda fn, label: fn()), \
                patch.object(self.engine, '_ensure_voice'):
            self.engine.send_text('键盘中文', translate=True)
        typed_job = self.engine._my_queue.get_nowait()
        self.assertTrue(live_job.live_input)
        self.assertTrue(simultaneous_job.live_input)
        self.assertFalse(typed_job.live_input)
        player = MagicMock()
        self.engine.player = player
        pcm = b'\x01\x00' * 24
        for job, muted in ((live_job, True), (simultaneous_job, True), (typed_job, False), (None, False)):
            self.engine._speech_jobs.clear()
            if job is not None:
                self.engine._speech_jobs['Answer.'] = job
            self.engine._on_first_audio('Answer.')
            player.reset_mock()
            self.engine._on_tts_audio(pcm)
            if muted:
                player.write.assert_called_once_with(pcm, include_monitor=False)
            else:
                player.write.assert_called_once_with(pcm)
        self.engine._speech_jobs['Answer.'] = live_job
        self.engine._on_first_audio('Answer.')
        self.settings.set('live_monitor_enabled', True)
        player.reset_mock()
        self.engine._on_tts_audio(pcm)
        player.write.assert_called_once_with(pcm)

    def test_output_mute_keeps_meeting_audio_and_recovers_monitor_without_reopening(self):
        meeting, monitor = MagicMock(), MagicMock()
        pcm = b'\x01\x00' * 24
        with patch('teams_voice_translator.audio.sd.RawOutputStream', side_effect=[meeting, monitor]) as create:
            with MultiOutputPlayer([7,44], 24000) as player:
                player.write(pcm, include_monitor=False)
                meeting.write.assert_called_with(pcm)
                monitor.write.assert_called_with(bytes(len(pcm)))
                player.write(pcm)
                monitor.write.assert_called_with(pcm)
                self.assertEqual(create.call_count, 2)

    def test_f8_switch_applies_to_running_bridge_without_restarting_microphone(self):
        player = MagicMock()
        ready = threading.Event()
        player.write.side_effect = lambda *args, **kwargs: ready.set()
        with patch('teams_voice_translator.audio.MultiOutputPlayer', return_value=player), \
                patch('teams_voice_translator.audio.MicrophoneCapture') as capture:
            bridge = DirectAudioBridge()
            bridge.start(1,7,48000,monitor_devices=[44],
                         monitor_enabled=lambda: self.settings.get('live_monitor_enabled'))
            try:
                enqueue = capture.call_args.args[1]
                for enabled in (False, True):
                    self.settings.set('live_monitor_enabled', enabled)
                    ready.clear()
                    enqueue(b'\x01\x00' * 20)
                    self.assertTrue(ready.wait(2))
                    player.write.assert_called_with(b'\x01\x00' * 20, include_monitor=enabled)
                capture.return_value.start.assert_called_once()
            finally:
                bridge.stop()

    def test_recovery_preserves_live_mute_and_other_playback_monitoring(self):
        player = MagicMock()
        def ensure():
            self.engine.player = player
        with patch.object(self.engine, '_ensure_player', side_effect=ensure):
            self.engine._recover_player(b'\x01\x00', include_monitor=False)
            player.write.assert_called_with(b'\x01\x00', include_monitor=False)
            self.engine._recover_player(b'\x02\x00')
            player.write.assert_called_with(b'\x02\x00')

    def test_main_switch_persists_independently_without_changing_other_monitoring(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense import ui
        app = QApplication.instance() or QApplication([])
        with patch.object(ui, 'apply_default_loopback', return_value=False), \
                patch.object(ui.DefenseEngine, 'start'), \
                patch.object(ui.DefenseWindow, '_auto_prerender'), \
                patch.object(ui.DefenseEngine, 'apply_settings') as restart:
            window = ui.DefenseWindow(self.settings)
            try:
                self.assertTrue(window.live_monitor_button.isChecked())
                window.live_monitor_button.click()
                self.assertFalse(self.settings.get('live_monitor_enabled'))
                self.assertTrue(self.settings.get('monitor_enabled'))
                self.assertFalse(DefenseSettings(Path(self.tmp.name)).get('live_monitor_enabled'))
                window.live_monitor_button.click()
                self.assertTrue(self.settings.get('live_monitor_enabled'))
                restart.assert_not_called()
            finally:
                window.close()
                app.processEvents()


if __name__ == '__main__':
    unittest.main()
