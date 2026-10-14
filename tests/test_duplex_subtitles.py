import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication
from teams_voice_translator.defense.committee_process import _recognize
from teams_voice_translator.defense.overlay import SubtitleOverlay
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks


class DuplexSubtitlesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))

    def test_real_font_pixel_pages_fit_and_cover_the_full_text(self):
        overlay = SubtitleOverlay(self.settings)
        self.addCleanup(overlay.stop)
        text = ('Magnetic flux leakage, or MFL, inspection can continuously measure magnetic-field '
                'responses in ferromagnetic pipelines. However, in practical integrity assessment, '
                'simply detecting whether a defect exists is not sufficient. ') * 4
        overlay.show_sentence('', text, 30)
        chunks = []
        pages = overlay._pixel_pages(overlay._chunks, overlay.en_label)
        self.assertGreater(len(pages), 1)
        for first, last in pages:
            overlay._render_page(first)
            self.app.processEvents()
            self.assertGreaterEqual(overlay.en_label.height(),
                                    overlay.en_label.heightForWidth(overlay.en_label.width()))
            self.assertLessEqual(overlay.en_label.geometry().bottom(), overlay.height())
            chunks.extend(overlay._chunks[first:last])
        self.assertEqual(''.join(chunks).strip(), text.strip())

    def test_actual_playback_progress_advances_highlight_and_pages_and_pause_preserves_it(self):
        overlay = SubtitleOverlay(self.settings)
        self.addCleanup(overlay.stop)
        text = ' '.join(f'word{i}' for i in range(200))
        overlay.show_sentence('', text, 100)
        first = overlay.en_label.text()
        overlay.update_playback_progress(text, .8)
        self.assertNotEqual(first, overlay.en_label.text())
        current = overlay.en_label.text()
        overlay.pause()
        overlay._last_activity = 0
        overlay._tick()
        self.assertTrue(overlay.isVisible())
        self.assertEqual(current, overlay.en_label.text())
        overlay.resume()
        overlay.update_playback_progress(text, .95)
        self.assertNotEqual(current, overlay.en_label.text())
        overlay.stop()
        overlay.update_playback_progress(text, .97)
        self.assertFalse(overlay.isVisible())

    def test_chinese_without_spaces_has_moving_cursor_and_saved_position_is_on_screen(self):
        self.settings.shared.update({'overlay_x': -9999, 'overlay_y': 9999})
        overlay = SubtitleOverlay(self.settings)
        self.addCleanup(overlay.stop)
        overlay.show_sentence('', '大家好，今天介绍我的研究工作。', 8)
        first = overlay.en_label.text()
        overlay.update_playback_progress('大家好，今天介绍我的研究工作。', .7)
        self.assertNotEqual(first, overlay.en_label.text())
        area = overlay.screen().availableGeometry()
        self.assertGreaterEqual(overlay.y(), area.top())
        self.assertLessEqual(overlay.y() + overlay.height(), area.bottom() + 1)

    def test_committee_recognized_event_does_not_wait_for_translation(self):
        found = []
        engine = DefenseEngine(self.settings, EngineCallbacks(on_committee_recognized=lambda *v: found.append(v)))
        self.addCleanup(engine.shutdown)
        engine._direct_active = True
        engine._suppress_until = time.monotonic() + 60
        engine._on_committee_segment('Can you explain the limitation?')
        self.assertEqual(found[0][1], 'Can you explain the limitation?')
        self.assertEqual(engine._committee_queue.qsize(), 1)

    def test_asr_timeout_reconnects_and_late_failed_session_does_not_emit(self):
        audio, events, stop = queue.Queue(), queue.Queue(), threading.Event()
        sessions = []
        def factory(**kwargs):
            session = MagicMock()
            session.kwargs = kwargs
            if not sessions:
                session.send_audio.side_effect = lambda pcm: kwargs['on_error']('request timeout after 23 seconds')
            else:
                session.send_audio.side_effect = lambda pcm: kwargs['on_segment']('Recovered question.', 'neutral')
            sessions.append(session)
            return session
        config = {'kwargs': {'values': {'asr_model': 'offline'}}, 'backup': {}}
        worker = threading.Thread(target=_recognize, args=(config,audio,events,stop,factory), daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 3
            while not sessions and time.monotonic() < deadline:
                time.sleep(.01)
            audio.put(b'\x01\x00'*100)
            while len(sessions) < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(len(sessions), 2)
            sessions[0].kwargs['on_segment']('Late old question.', 'neutral')
            audio.put(b'\x02\x00'*100)
            segments = []
            while time.monotonic() < deadline:
                kind,args = events.get(timeout=1)
                if kind == 'segment':
                    segments.append(args[0])
                    break
            self.assertEqual(segments, ['Recovered question.'])
        finally:
            stop.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
