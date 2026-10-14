import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import ScriptDeskPanel


class ScriptClearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        self.engine = MagicMock()
        self.engine.script_pcm.return_value = None
        self.engine.script_group_slices.return_value = None
        self.panel = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.source_edit.setPlainText('Wrong first page.\nWrong second page.')
        self.panel._split_clicked()

    def flush(self):
        for _ in range(10):
            self.app.processEvents()
            time.sleep(.005)

    def test_button_clears_all_columns_stops_playback_and_persists_with_importable_backup(self):
        panel = self.panel
        panel._playing = list(panel.document.lines)
        panel._continue_all = True
        panel._cursor_timer.start()
        panel._chinese_backup = '旧稿'
        panel.undo_translate_button.setEnabled(True)
        panel.clear_script_button.click()
        self.assertEqual(panel.source_edit.toPlainText(), '')
        self.assertEqual(panel.paragraph_table.rowCount(), 0)
        self.assertEqual(panel.line_table.rowCount(), 0)
        self.assertEqual(panel._playing, [])
        self.assertFalse(panel._continue_all)
        self.assertFalse(panel._cursor_timer.isActive())
        self.assertFalse(panel.undo_translate_button.isEnabled())
        self.assertEqual(panel._chinese_backup, '')
        self.engine.interrupt_speech.assert_called_once()
        self.engine.delete_script_audio.assert_not_called()
        self.assertEqual(json.loads(panel._store_path.read_text(encoding='utf-8'))['text'], '')
        backups = list((Path(self.tmp.name)/'script-backups').glob('*.txt'))
        self.assertEqual(len(backups), 1)
        self.assertIn('Wrong second page.', backups[0].read_text(encoding='utf-8'))
        reopened = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(reopened.shutdown)
        self.assertEqual(reopened.paragraph_table.rowCount(), 0)

    def test_empty_source_split_clears_old_rows_instead_of_refusing(self):
        self.panel.source_edit.clear()
        self.panel._split_clicked()
        self.assertEqual(self.panel.paragraph_table.rowCount(), 0)
        backups = list((Path(self.tmp.name)/'script-backups').glob('*.txt'))
        self.assertIn('Wrong first page.', backups[0].read_text(encoding='utf-8'))

    def test_translation_finishing_after_clear_cannot_restore_old_script(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def translate(text, previous):
            entered.set()
            release.wait(3)
            returned.set()
            return 'Old translated script.'
        self.panel.translate_runner = translate
        self.panel.source_edit.setPlainText('旧中文讲稿。')
        self.panel._translate_source()
        try:
            self.assertTrue(entered.wait(2))
            self.panel.clear_script_button.click()
            self.panel.source_edit.setPlainText('New correct script.')
            self.panel._split_clicked()
        finally:
            release.set()
        self.assertTrue(returned.wait(2))
        self.flush()
        self.assertEqual(self.panel.source_edit.toPlainText(), 'New correct script.')
        self.assertEqual(self.panel.document.lines[0].text, 'New correct script.')
        self.assertFalse(self.panel.undo_translate_button.isEnabled())

    def test_render_finishing_after_clear_cannot_modify_new_script(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def render(*args, **kwargs):
            entered.set()
            release.wait(3)
            returned.set()
            return [2.5]
        self.engine.render_script_group.side_effect = render
        self.engine.script_render_unit.return_value = 'group'
        self.engine.script_groups.side_effect = lambda texts: [texts]
        self.panel._start_render(self.panel.document.lines)
        try:
            self.assertTrue(entered.wait(2))
            self.panel.clear_script_button.click()
            self.panel.source_edit.setPlainText('New correct script.')
            self.panel._split_clicked()
        finally:
            release.set()
        self.assertTrue(returned.wait(2))
        self.flush()
        self.assertFalse(self.panel.document.lines[0].rendered)
        self.assertEqual(self.panel.document.lines[0].seconds, 0)
        self.assertTrue(self.panel.render_all_button.isEnabled())
        self.assertIn('已切分', self.panel.status_label.text())

    def test_disk_failure_does_not_pretend_script_was_cleared(self):
        original = self.panel.source_edit.toPlainText()
        with patch.object(self.panel, '_write_script', side_effect=PermissionError('locked')):
            self.panel.clear_script_button.click()
        self.assertEqual(self.panel.source_edit.toPlainText(), original)
        self.assertEqual(self.panel.paragraph_table.rowCount(), 2)
        self.assertIn('清空讲稿失败', self.panel.status_label.text())
        self.engine.interrupt_speech.assert_not_called()


if __name__ == '__main__':
    unittest.main()
