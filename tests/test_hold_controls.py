import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from teams_voice_translator.defense.ui import DefenseWindow
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.audio import MicrophoneCapture


class HoldControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        with patch('teams_voice_translator.defense.ui.DefenseEngine') as factory:
            self.window = DefenseWindow(self.settings)
            self.engine = factory.return_value
        self.window.show(); self.app.processEvents()
        self.addCleanup(self.window.close)

    def test_mouse_is_hold_to_talk_when_continuous_is_off(self):
        for mode in ('direct', 'translate'):
            button = getattr(self.window, mode + '_button')
            QTest.mousePress(button, Qt.LeftButton)
            self.assertEqual(self.window._input_mode, mode)
            QTest.mouseRelease(button, Qt.LeftButton)
            self.assertIsNone(self.window._input_mode)
            self.engine.set_listening.assert_called_with(False)
            self.engine.stop_direct.assert_called()

    def test_release_outside_button_stops_capture(self):
        button = self.window.translate_button
        QTest.mousePress(button, Qt.LeftButton)
        QTest.mouseRelease(button, Qt.LeftButton, pos=QPoint(-20, -20))
        self.assertIsNone(self.window._input_mode)

    def test_continuous_click_switch_click_off_never_returns_to_previous_mode(self):
        self.window.continuous_check.setChecked(True)
        self.assertIsNone(self.window._input_mode)
        for first, second in (('direct', 'translate'), ('translate', 'direct')):
            self.window._set_input_mode(first)
            QTest.mouseClick(getattr(self.window, second + '_button'), Qt.LeftButton)
            self.assertEqual(self.window._input_mode, second)
            QTest.mouseClick(getattr(self.window, second + '_button'), Qt.LeftButton)
            self.assertIsNone(self.window._input_mode)
            self.assertTrue(self.window.continuous_check.isChecked())

    def test_keyboard_and_mouse_share_mode_rules(self):
        self.window.signals.input_pressed.emit('translate', 'keyboard'); self.app.processEvents()
        QTest.mousePress(self.window.translate_button, Qt.LeftButton)
        QTest.mouseRelease(self.window.translate_button, Qt.LeftButton)
        self.assertEqual(self.window._input_mode, 'translate')
        self.window.signals.input_released.emit('translate', 'keyboard'); self.app.processEvents()
        self.assertIsNone(self.window._input_mode)

    def test_idle_keeps_continuous_preference_visible_and_start_waits_for_selection(self):
        self.settings.set('continuous_enabled', True)
        self.window._on_state_changed('idle')
        self.assertTrue(self.window.continuous_check.isChecked())
        self.engine.state = 'idle'
        self.window._on_toggle_defense()
        self.assertIsNone(self.window._input_mode)
        self.engine.set_listening.assert_called_once_with(False)
        self.engine.start_direct.assert_not_called()
        QTest.mouseClick(self.window.translate_button, Qt.LeftButton)
        self.assertEqual(self.window._input_mode, 'translate')
        QTest.mouseClick(self.window.translate_button, Qt.LeftButton)
        self.assertIsNone(self.window._input_mode)
        self.engine.set_listening.assert_called_with(False)

    def test_checking_continuous_does_not_start_either_mode(self):
        self.engine.reset_mock()
        self.window.continuous_check.setChecked(True)
        self.assertIsNone(self.window._input_mode)
        self.assertFalse(self.window.translate_button.isChecked())
        self.assertFalse(self.window.direct_button.isChecked())
        self.engine.set_listening.assert_not_called()
        self.engine.start_direct.assert_not_called()
        self.engine.start_defense.assert_not_called()
        QTest.mouseClick(self.window.direct_button, Qt.LeftButton)
        self.assertEqual(self.window._input_mode, 'direct')
        self.engine.start_direct.assert_called_once()

    def test_start_with_continuous_off_and_click_stays_off_after_state_update(self):
        self.engine.state = 'idle'
        self.window._on_toggle_defense()
        self.engine.set_listening.assert_called_with(False)
        QTest.mousePress(self.window.translate_button, Qt.LeftButton)
        self.window._on_state_changed('listening')
        QTest.mouseRelease(self.window.translate_button, Qt.LeftButton)
        self.assertFalse(self.window.continuous_check.isChecked())
        self.assertIsNone(self.window._input_mode)
        self.engine.set_listening.assert_called_with(False)


class CaptureOwnershipTests(unittest.TestCase):
    def test_engine_start_does_not_auto_listen_with_saved_continuous_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp))
            settings.set('continuous_enabled', True)
            settings.set('auto_record', False)
            engine = DefenseEngine(settings, EngineCallbacks())
            self.addCleanup(engine.shutdown)
            with patch.object(engine, '_build_clients'), patch.object(engine, 'reload_context'), \
                 patch.object(engine, '_ensure_voice'), patch.object(engine, '_start_committee_listening'), \
                 patch.object(engine, '_do_toggle_listening') as listen:
                engine._do_start_defense()
            listen.assert_not_called()
            self.assertEqual(engine.state, 'standby')
            self.assertFalse(engine._listening)

    def test_open_read_stop_close_stay_on_one_owner_thread(self):
        operations = []
        class Stream:
            def __init__(self, **kwargs):
                self.check('open'); assert 'callback' not in kwargs
            def check(self, name): operations.append((name, threading.get_ident()))
            def start(self): self.check('start')
            def read(self, count): self.check('read'); time.sleep(.002); return bytes(count * 2), False
            def stop(self): self.check('stop')
            def close(self): self.check('close')
        pcm = []
        with patch('teams_voice_translator.audio.sd.RawInputStream', Stream):
            capture = MicrophoneCapture(None, pcm.append)
            capture.start(); time.sleep(.015); capture.stop()
        self.assertTrue(pcm)
        self.assertEqual(len(set(t for _, t in operations)), 1)
        self.assertNotEqual(operations[0][1], threading.get_ident())
        self.assertEqual([name for name, _ in operations][-2:], ['stop', 'close'])
        self.assertFalse(capture._thread.is_alive())

    def test_release_discards_pending_start_and_gates_pcm_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DefenseEngine(DefenseSettings(Path(tmp)), EngineCallbacks())
            pending = []; engine._submit = lambda action, label: pending.append(action)
            engine._do_set_listening = Mock()
            engine.set_listening(True); engine.set_listening(False)
            engine.asr_mic = Mock()
            engine._on_mic_audio(b'1234')
            engine.asr_mic.send_audio.assert_not_called()
            for action in pending: action()
            engine._do_set_listening.assert_called_once_with(False)
            engine.shutdown()

    def test_release_while_waiting_for_asr_never_opens_microphone(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp)); settings.get_api_key = lambda: 'test'
            engine = DefenseEngine(settings, EngineCallbacks()); engine._session_active = True
            engine._listen_requested = True
            engine._submit = lambda action, label: None
            asr = Mock(); asr.wait_ready.side_effect = lambda: engine.set_listening(False)
            with patch('teams_voice_translator.defense.pipeline.create_realtime_asr', return_value=asr), \
                 patch('teams_voice_translator.defense.pipeline.MicrophoneCapture') as mic:
                engine._do_set_listening(True)
            mic.assert_not_called(); asr.close.assert_called()
            self.assertFalse(engine._listening); engine.shutdown()

    def test_direct_stop_does_not_resume_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DefenseEngine(DefenseSettings(Path(tmp)), EngineCallbacks())
            engine._direct_active = True; engine._resume_listening_after_direct = True
            engine._do_set_listening = Mock(); engine._do_toggle_listening = Mock()
            engine._do_stop_direct()
            engine._do_set_listening.assert_not_called(); engine._do_toggle_listening.assert_not_called()
            engine.shutdown()

    def test_own_speech_is_not_sent_back_to_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DefenseEngine(DefenseSettings(Path(tmp)), EngineCallbacks())
            engine.asr_mic = Mock(); engine._suppress_until = time.monotonic() + 1
            engine._on_mic_audio(b'1234')
            engine.asr_mic.send_audio.assert_called_once_with(bytes(4))
            engine.shutdown()


if __name__ == '__main__': unittest.main()
