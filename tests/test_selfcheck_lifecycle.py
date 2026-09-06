"""Exercise self-check completion under a running Qt event loop without cloud calls."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QThread, Slot, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import SelfCheckDialog


class ObservedDialog(SelfCheckDialog):
    def __init__(self, settings):
        self.deliveries = []
        self.finished_threads = []
        super().__init__(None, settings)

    @Slot(int, bool, str)
    def _finish_row(self, row, ok, message):
        self.deliveries.append((row, QThread.currentThread() == self.thread()))
        super()._finish_row(row, ok, message)

    @Slot()
    def _check_finished(self):
        self.finished_threads.append(QThread.currentThread() == self.thread())
        super()._check_finished()


class SelfCheckLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = DefenseSettings(Path(self.tmp.name))
        settings.shared.update({"workspace_id": "offline-test"})
        settings.set("tts_voice_id", "offline-test-voice")
        settings.get_api_key = lambda: "offline-test-key"
        self.dialog = ObservedDialog(settings)
        self.dialog.show()
        self.app.processEvents()
        self.addCleanup(self.cleanup_dialog)
        self.patches = [patch("teams_voice_translator.defense.ui._safe_list_audio_devices", return_value=([], [])),
                        patch("teams_voice_translator.defense.ui._safe_list_loopback_devices", return_value=[]),
                        patch("teams_voice_translator.defense.ui.ContextTranslator")]
        for p in self.patches:
            obj = p.start()
            self.addCleanup(p.stop)
            if p is self.patches[-1]: obj.return_value.translate.return_value = "Hello."

    def cleanup_dialog(self):
        self.dialog.close()
        if self.dialog._thread:
            self.dialog._thread.join(2)
        self.dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def wait_finished(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.app.processEvents()
            if self.dialog.run_button.isEnabled() and not self.dialog._thread.is_alive():
                return
            time.sleep(.002)
        self.fail("Self-check did not finish")

    def test_playback_completion_and_repeated_checks_stay_on_gui_thread(self):
        calls = []
        def audio(*args, **kwargs):
            calls.append(threading.get_ident())
            time.sleep(.002)  # return after the simulated audio worker finishes
        with patch("teams_voice_translator.defense.ui.voice_probe", side_effect=audio):
            for _ in range(30):
                self.dialog._run()
                self.wait_finished()
                self.assertEqual(self.dialog.table.rowCount(), 8)
                self.assertIn("已在本机", self.dialog.table.item(7, 1).text())
        self.assertEqual(len(calls), 30)
        self.assertTrue(all(t != threading.get_ident() for t in calls))
        self.assertEqual(len(self.dialog.deliveries), 240)
        self.assertTrue(all(on_gui for _, on_gui in self.dialog.deliveries))
        self.assertEqual(self.dialog.finished_threads, [True] * 30)

    def test_close_during_playback_cancels_and_ignores_late_completion(self):
        entered = threading.Event()
        cancelled = threading.Event()
        def audio(*args, **kwargs):
            entered.set()
            if kwargs["cancel_event"].wait(2): cancelled.set()
        with patch("teams_voice_translator.defense.ui.voice_probe", side_effect=audio):
            self.dialog._run()
            self.assertTrue(entered.wait(2))
            self.dialog.close()
            self.assertTrue(cancelled.wait(1))
            self.dialog._thread.join(2)
            self.app.processEvents()
        self.assertTrue(self.dialog._closing)
        self.assertFalse(self.dialog.run_button.isEnabled())
        self.assertNotIn(7, [row for row, _ in self.dialog.deliveries])
        self.assertFalse(self.dialog.finished_threads)

    def test_repeated_click_cannot_start_parallel_self_checks(self):
        entered, release = threading.Event(), threading.Event()
        def audio(*args, **kwargs):
            entered.set(); release.wait(2)
        with patch("teams_voice_translator.defense.ui.voice_probe", side_effect=audio) as probe:
            self.dialog._run()
            self.assertTrue(entered.wait(2))
            thread = self.dialog._thread
            try:
                self.dialog._run()
                self.assertIs(thread, self.dialog._thread)
                probe.assert_called_once()
            finally:
                release.set()
            self.wait_finished()

    def test_unexpected_worker_failure_does_not_leave_button_disabled(self):
        with patch.object(self.dialog.settings, "get_api_key", side_effect=RuntimeError("credential store unavailable")):
            self.dialog._run()
            self.wait_finished()
        self.assertIn("自检中断", self.dialog.table.item(7, 1).text())


if __name__ == "__main__":
    unittest.main()
