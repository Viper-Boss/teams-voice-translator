"""Offline tests of atomic spoken-term edits and GUI lifecycle."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from teams_voice_translator.defense.glossary_audit import AUDIT_MODEL, audit_terms


def answer(changes):
    return json.dumps({'changes':changes})


class AuditTests(unittest.TestCase):
    def run_audit(self, original, target):
        return audit_terms([{'source':'中文','target':original}],
            post=lambda _:answer([{'id':0,'target':target,'reason':'口语写法'}]), cancel_event=threading.Event())

    def test_redundant_acronym_and_parallel_slash_can_be_edited(self):
        self.assertEqual(self.run_audit('Knowledge distillation (KD)', 'Knowledge distillation')[0]['target'], 'Knowledge distillation')
        self.assertEqual(self.run_audit('Query/Key/Value matrices (Q/K/V)', 'query, key, and value matrices')[0]['target'], 'query, key, and value matrices')

    def test_variables_units_and_formulas_cannot_be_removed(self):
        for value in ('Defect depth (D)', 'Defect length (L)', 'Defect width (W)',
                      'R-squared (R²)', 'Temperature (°C)', 'Speed (mm/s)', 'Ratio L/D'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_audit(value, 'Bad rewrite')

    def test_variable_parentheses_retained_when_other_text_corrected(self):
        self.assertEqual(len(self.run_audit('defect depth (D)', 'Defect depth (D)')), 1)

    def test_fixed_flash_model_and_no_changes_valid(self):
        payloads = []
        def post(payload): payloads.append(payload); return answer([])
        result = audit_terms([{'source':'深度','target':'Defect depth (D)'}],post=post,cancel_event=threading.Event())
        self.assertEqual(result, [])
        self.assertEqual(payloads[0]['model'], AUDIT_MODEL)
        self.assertIn('不得修改', payloads[0]['messages'][0]['content'])

    def test_malformed_ids_duplicates_and_empty_rejected(self):
        for changes in ([{'id':True,'target':'X'}],[{'id':4,'target':'X'}],
                        [{'id':0,'target':''}], [{'id':0,'target':'X'},{'id':0,'target':'Y'}]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit_terms([{'source':'S','target':'T'}],post=lambda _:answer(changes),cancel_event=threading.Event())

    def test_later_batch_failure_does_not_return_partial_changes(self):
        terms = [{'source':f'S{i}','target':'Full term (FT)'} for i in range(36)]
        replies = iter([answer([{'id':0,'target':'Full term'}]),'malformed'])
        with self.assertRaises(ValueError):
            audit_terms(terms,post=lambda _:next(replies),cancel_event=threading.Event())
        self.assertEqual(terms[0]['target'],'Full term (FT)')

    def test_cancel_after_response_does_not_apply(self):
        cancel = threading.Event()
        def post(_): cancel.set(); return answer([{'id':0,'target':'Short term'}])
        with self.assertRaisesRegex(ValueError,'已停止'):
            audit_terms([{'source':'S','target':'Long term'}],post=post,cancel_event=cancel)


class UiTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import GlossaryDialog
        self.app = QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.settings.set_glossary([{'source':'查询/键/值矩阵','target':'Query/Key/Value matrices (Q/K/V)'},
                                   {'source':'缺陷深度','target':'Defect depth (D)'}])
        self.dialog = GlossaryDialog(None,self.settings)
        self.addCleanup(self.dialog.deleteLater)

    def wait(self):
        end = time.monotonic()+3
        while self.dialog._audit_running and time.monotonic()<end:
            self.app.processEvents(); time.sleep(.005)
        self.assertFalse(self.dialog._audit_running)

    def test_auto_edit_save_and_undo_preserves_source_order_and_symbols(self):
        d = self.dialog
        d.post_factory = lambda:lambda _:answer([{'id':0,'target':'query, key, and value matrices','reason':'并列口语'}])
        d.audit(); self.wait()
        self.assertEqual(d.terms()[0]['source'],'查询/键/值矩阵')
        self.assertEqual(d.terms()[0]['target'],'query, key, and value matrices')
        self.assertEqual(d.terms()[1]['target'],'Defect depth (D)')
        self.assertTrue(d._dirty)
        self.assertNotEqual(self.settings.glossary_terms(),d.terms())
        d.save(); d.undo_audit()
        self.assertEqual(d.terms()[0]['target'],'Query/Key/Value matrices (Q/K/V)')
        self.assertTrue(d._dirty, 'undo after saving must require another save')

    def test_edits_while_running_are_never_overwritten(self):
        gate = threading.Event()
        def post(_): gate.wait(1); return answer([{'id':0,'target':'Edited'}])
        d = self.dialog; d.post_factory = lambda:post
        d.audit(); d.table.item(0,1).setText('User edit'); gate.set(); self.wait()
        self.assertEqual(d.table.item(0,1).text(),'User edit')
        self.assertIn('未应用',d.gen_status.text())

    def test_failure_keeps_table_and_reenables_controls(self):
        d = self.dialog; old=d.terms()
        d.post_factory = lambda:lambda _:'invalid'
        d.audit(); self.wait()
        self.assertEqual(old,d.terms())
        self.assertTrue(d.audit_button.isEnabled())
        self.assertTrue(d.generate_button.isEnabled())

    def test_stop_does_not_apply_late_response(self):
        d = self.dialog; gate=threading.Event(); old=d.terms()
        def post(_): gate.wait(1);return answer([{'id':0,'target':'Edited'}])
        d.post_factory=lambda:post
        d.audit(); d._cancel.set(); gate.set(); self.wait()
        self.assertEqual(old,d.terms())

    def test_blank_rows_are_preserved(self):
        d=self.dialog; d._add_row()
        d.post_factory=lambda:lambda _:answer([{'id':0,'target':'query, key, and value matrices'}])
        d.audit(); self.wait()
        self.assertEqual(d.table.rowCount(),3)
        d.undo_audit(); self.assertEqual(d.table.rowCount(),3)
