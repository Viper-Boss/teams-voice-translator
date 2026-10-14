import threading
from unittest.mock import Mock, MagicMock, patch

from test_batch17 import _Base, _wait
from PySide6.QtCore import Qt, QThread
from teams_voice_translator.defense.ui import SettingsDialog, ScriptDeskPanel, SelfCheckDialog


class ConnectionTestLifecycleTests(_Base):
    def setUp(self):
        super().setUp()
        self.dialog = SettingsDialog(None, self.settings)
        self.addCleanup(self.dialog.close)
        self.dialog.workspace_edit.setText("fake-workspace")
        self.dialog.api_key_edit.setText("fake-key")
        self.dialog._current_voice_id = lambda: "fake-voice"

    def test_worker_completes_on_gui_and_reenables_button_with_stage_results(self):
        threads = []
        self.dialog.connection_event.connect(lambda *_: threads.append(QThread.currentThread() == self.dialog.thread()))
        translator = Mock()
        translator.translate.return_value = "Hello."
        asr = Mock()
        with patch("teams_voice_translator.defense.ui.ContextTranslator", return_value=translator), \
                patch("teams_voice_translator.defense.ui.create_realtime_asr", return_value=asr), \
                patch("teams_voice_translator.defense.ui.voice_probe", return_value=b"\x01\x00") as voice:
            self.dialog._run_connection_test()
            self.assertTrue(_wait(lambda: not self.dialog._connection_running))
        self.assertTrue(self.dialog.test_button.isEnabled())
        self.assertNotIn("测试中…", self.dialog.test_result_label.text())
        self.assertIn("连接测试已结束", self.dialog.test_result_label.text())
        self.assertTrue(threads and all(threads))
        asr.close.assert_called_once()
        self.assertEqual(voice.call_count, 1)
        self.assertIsNotNone(translator.translate.call_args.kwargs["cancel_event"])

    def test_timeout_cancels_subsequent_asr_and_paid_voice_then_ignores_old_results(self):
        gate, entered = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        translator = Mock()
        def blocked(*_args, **_kwargs):
            entered.set()
            gate.wait(5)
            return "Hello."
        translator.translate.side_effect = blocked
        self.dialog.CONNECTION_TIMEOUT_MS = 30
        with patch("teams_voice_translator.defense.ui.ContextTranslator", return_value=translator), \
                patch("teams_voice_translator.defense.ui.create_realtime_asr") as asr, \
                patch("teams_voice_translator.defense.ui.voice_probe") as voice:
            self.dialog._run_connection_test()
            self.assertTrue(_wait(entered.is_set))
            serial = self.dialog._connection_serial
            self.assertTrue(_wait(lambda: not self.dialog._connection_running))
            self.assertIn("超时", self.dialog.test_result_label.text())
            self.assertTrue(self.dialog.test_button.isEnabled())
            gate.set()
            self.assertTrue(_wait(lambda: self.dialog._connection_cancel.is_set()))
            self.dialog._on_connection_event(serial, "LATE RESULT", True)
            self.assertNotIn("LATE RESULT", self.dialog.test_result_label.text())
            asr.assert_not_called()
            voice.assert_not_called()

    def test_unexpected_worker_exception_finishes_ui(self):
        with patch.object(self.dialog, "_test_worker", side_effect=RuntimeError("unexpected failure")):
            self.dialog._run_connection_test()
            self.assertTrue(_wait(lambda: not self.dialog._connection_running))
        self.assertIn("unexpected failure", self.dialog.test_result_label.text())
        self.assertTrue(self.dialog.test_button.isEnabled())

    def test_asr_failure_still_closes_client_and_reports_tts_error_in_full(self):
        translator = Mock()
        translator.translate.return_value = "Hello."
        asr = Mock()
        asr.wait_ready.side_effect = RuntimeError("ASR not ready")
        message = "Invalid voice " + "x" * 180 + " request-id-tail"
        with patch("teams_voice_translator.defense.ui.ContextTranslator", return_value=translator), \
                patch("teams_voice_translator.defense.ui.create_realtime_asr", return_value=asr), \
                patch("teams_voice_translator.defense.ui.voice_probe", side_effect=RuntimeError(message)):
            self.dialog._run_connection_test()
            self.assertTrue(_wait(lambda: not self.dialog._connection_running))
        asr.close.assert_called_once()
        self.assertIn("ASR not ready", self.dialog.test_result_label.text())
        self.assertIn("request-id-tail", self.dialog.test_result_label.text())

    def test_closing_dialog_cancels_current_test_and_restores_button(self):
        with patch.object(self.dialog, "_connection_test_entry"):
            self.dialog._run_connection_test()
        self.dialog.reject()
        self.assertTrue(self.dialog._connection_cancel.is_set())
        self.assertFalse(self.dialog._connection_running)
        self.assertTrue(self.dialog.test_button.isEnabled())


class ScriptDetailsTests(_Base):
    def setUp(self):
        super().setUp()
        engine = MagicMock()
        engine.script_render_unit.return_value = "sentence"
        engine.script_pcm.return_value = None
        self.panel = ScriptDeskPanel(self.settings, engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.source_edit.setPlainText("I welcome your valuable comments and suggestions, and I am ready to address any questions about the method, experiments, and limitations.")
        self.panel._split_clicked()

    def test_full_sentence_and_full_error_are_displayed_and_wrapped(self):
        line = self.panel.document.paragraphs[0].lines[0]
        line.error = "语音服务未返回有效音频或可下载地址（请求编号 " + "test-request-" * 8 + "tail）"
        self.panel._refresh_lines()
        self.assertEqual(self.panel.line_table.item(0, 1).text(), line.text)
        self.assertGreater(len(line.text), 80)
        self.assertEqual(self.panel.line_table.item(0, 2).toolTip(), line.error)
        self.assertIn("tail", self.panel.line_table.item(0, 2).text())
        self.assertEqual(self.panel.line_table.textElideMode(), Qt.ElideNone)
        self.assertGreaterEqual(self.panel.line_table.columnWidth(2), 300)
        self.assertGreater(self.panel.line_table.rowHeight(0), 30)

    def test_self_check_error_has_full_tooltip_and_wrap(self):
        dialog = SelfCheckDialog(None, self.settings)
        self.addCleanup(dialog.close)
        dialog._add_row("音色试听", None, "")
        message = "failed: " + "request details " * 18 + "TAIL"
        dialog._finish_row(0, False, message)
        self.assertEqual(dialog.table.item(0, 1).toolTip(), message)
        self.assertIn("TAIL", dialog.table.item(0, 1).text())
        self.assertEqual(dialog.table.textElideMode(), Qt.ElideNone)
