import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import RehearsalDialog, VoiceCompareDialog


class Player:
    def __init__(self):
        self.parts = []
        self.owners = []
        self.closed = False
    def __enter__(self):
        self.owners.append(threading.get_ident())
        return self
    def write(self, data):
        self.owners.append(threading.get_ident())
        time.sleep(.01)
        self.parts.append(bytes(data))
    def close(self):
        self.owners.append(threading.get_ident())
        self.closed = True
    def __exit__(self, *_args): self.close()


class RehearsalPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        self.settings.update({'tts_model': 'cosyvoice-v3.5-plus', 'tts_voice_id': 'test'})
        self.settings.get_api_key = lambda: 'offline'
        self.pcm = bytes(range(256)) * 375
        self.player = Player()
        self.client = Mock()
        self.client._stream_legacy_tts.side_effect = lambda t, s, cb, c: cb(self.pcm)
        for p in (patch('teams_voice_translator.defense.ui.BailianClient', return_value=self.client),
                  patch('teams_voice_translator.audio.MultiOutputPlayer', return_value=self.player)):
            p.start(); self.addCleanup(p.stop)

    def spin(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.005)
        self.app.processEvents()
        self.assertTrue(predicate(), 'Timed out waiting for playback state')

    def pump(self, seconds=.08):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.005)

    def rehearsal(self):
        dialog = RehearsalDialog(None, self.settings)
        dialog._set_rows(['我的研究将物理约束与测量数据结合，以提高有噪声条件下的缺陷反演稳定性。'],
                         ['My research combines physical constraints with the information in the measured signals to improve the stability of defect inversion under noisy conditions.'])
        dialog.show(); self.app.processEvents()
        self.addCleanup(dialog.close)
        return dialog

    def test_pause_stops_current_sentence_and_progress_then_resumes_without_loss(self):
        dialog = self.rehearsal()
        dialog._play_all()
        self.spin(lambda: len(self.player.parts) >= 5)
        QTest.mouseClick(dialog.pause_button, Qt.LeftButton)
        self.pump()
        count = len(self.player.parts)
        cursor = dialog.subtitle_en.text()
        self.pump()
        self.assertEqual(len(self.player.parts), count)
        self.assertEqual(dialog.subtitle_en.text(), cursor)
        self.assertTrue(dialog.subtitle_overlay.isVisible())
        self.assertIn('FFD400', cursor)
        self.assertIn('FFD400', dialog.result_list.cellWidget(0, 1).text())
        self.assertEqual(dialog.pause_button.text(), '继续')
        QTest.mouseClick(dialog.pause_button, Qt.LeftButton)
        self.spin(lambda: not dialog._play_thread.is_alive())
        self.assertEqual(b''.join(self.player.parts), self.pcm)
        self.assertFalse(dialog.pause_button.isEnabled())
        self.assertTrue(self.player.closed)
        self.assertEqual(len(set(self.player.owners)), 1)

    def test_stop_while_paused_exits_and_late_updates_do_not_show_subtitles(self):
        dialog = self.rehearsal()
        dialog._play_all()
        self.spin(lambda: len(self.player.parts) >= 2)
        QTest.mouseClick(dialog.pause_button, Qt.LeftButton)
        QTest.mouseClick(dialog.stop_button, Qt.LeftButton)
        self.spin(lambda: not dialog._play_thread.is_alive())
        self.assertFalse(dialog.subtitle_overlay.isVisible())
        self.assertLess(len(b''.join(self.player.parts)), len(self.pcm))

    def test_long_rows_reflow_when_columns_narrow(self):
        dialog = self.rehearsal()
        height = dialog.result_list.rowHeight(0)
        dialog.resize(680, 640)
        self.pump()
        self.assertGreater(dialog.result_list.rowHeight(0), height)
        self.assertEqual(dialog.result_list.textElideMode(), Qt.ElideNone)

    def test_comparison_stop_targets_new_token_for_first_play_and_replay(self):
        dialog = VoiceCompareDialog(None, self.settings)
        self.addCleanup(dialog.close)
        dialog.show()
        original_token = dialog._cancel
        dialog._probe(False)
        self.spin(lambda: len(self.player.parts) >= 3)
        self.assertIsNot(dialog._cancel, original_token)
        self.assertTrue(dialog.subtitle_overlay.isVisible())
        QTest.mouseClick(dialog.stop_button, Qt.LeftButton)
        self.spin(lambda: not dialog._is_busy)
        self.assertTrue(dialog._cancel.is_set())
        self.assertFalse(dialog.subtitle_overlay.isVisible())
        self.assertLess(len(b''.join(self.player.parts)), len(self.pcm))
        requests = self.client._stream_legacy_tts.call_count
        self.player.parts.clear()
        dialog._cached_audio[0] = self.pcm
        dialog._replay(0)
        self.spin(lambda: len(self.player.parts) >= 3)
        QTest.mouseClick(dialog.stop_button, Qt.LeftButton)
        self.spin(lambda: not dialog._is_busy)
        self.assertFalse(dialog.subtitle_overlay.isVisible())
        self.assertLess(len(b''.join(self.player.parts)), len(self.pcm))
        self.assertEqual(self.client._stream_legacy_tts.call_count, requests)


if __name__ == '__main__': unittest.main()
