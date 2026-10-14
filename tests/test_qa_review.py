"""Offline answer verification and lossless library exchange."""
import json,os,tempfile,threading,time,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock,patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from teams_voice_translator.defense.qa_bank import QaEntry,parse_bank_file
from teams_voice_translator.defense.qa_review import review_answers,export_bank,template_bank,REVIEW_MODEL


def entry(key='Q1'):
    return QaEntry(key,'What is the task?','You can say: My study estimates dimensions.',
        question_zh='任务是什么？',answer_zh='建议用户说：我的研究估计尺寸。',
        note='private reminder',aliases=['Task?'],points=['估计尺寸'],priority=2,category='task',basis='check',source='第 1 页')


def result(key='Q1',status='fix',**extra):
    return {'id':key,'status':status,'issues':['编辑建议不能朗读'],'evidence':[],
            'answer':'My study estimates dimensions.','answer_zh':'我的研究估计尺寸。',**extra}


class ReviewTests(unittest.TestCase):
    def check(self,results,entries=None,**kwargs):
        return review_answers(entries or [entry()],thinking=False,
            post=lambda _:json.dumps({'results':results}),context_for=lambda _:[],
            cancel_event=threading.Event(),**kwargs)

    def test_thinking_switch_and_evidence_attached_per_entry(self):
        self.assertEqual(REVIEW_MODEL, 'qwen3.8-max')
        for thinking in (True,False):
            seen=[]
            def post(payload):seen.append(payload);return json.dumps({'results':[result(evidence=['第 1 页'])]})
            review_answers([entry()],thinking=thinking,post=post,context_for=lambda _:[('第 1 页','source text')],cancel_event=threading.Event())
            self.assertEqual(seen[0]['model'],REVIEW_MODEL)
            self.assertEqual(seen[0]['enable_thinking'],thinking)
            self.assertIn('source text',seen[0]['messages'][1]['content'])

    def test_meta_instruction_rejected_in_rewritten_answers(self):
        for text in ('建议用户说：xxx','The user should say that...', 'Suggested answer: Example'):
            with self.subTest(text=text),self.assertRaises(ValueError):self.check([result(answer=text)])

    def test_meta_instruction_in_original_cannot_be_pass(self):
        self.assertEqual(self.check([result(status='pass')])[0]['status'],'check')

    def test_unverified_answer_not_provided_for_application(self):
        checked=self.check([result(status='check')])[0]
        self.assertNotIn('answer',checked)

    def test_invented_evidence_and_omissions_rejected(self):
        for rows in ([result(evidence=['第 99 页'])],[],[result(),result()],[result(key='unknown')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):self.check(rows)

    def test_later_batch_failure_returns_no_partial_results(self):
        entries=[entry(f'Q{i}') for i in range(6)]
        replies=iter([json.dumps({'results':[result(f'Q{i}') for i in range(5)]}),'bad'])
        with self.assertRaises(ValueError):
            review_answers(entries,thinking=False,post=lambda _:next(replies),context_for=lambda _:[],cancel_event=threading.Event())
        self.assertIn('You can say',entries[0].answer)

    def test_cancel_after_response_does_not_return_results(self):
        cancel=threading.Event()
        def post(_):cancel.set();return json.dumps({'results':[result()]})
        with self.assertRaisesRegex(ValueError,'已停止'):
            review_answers([entry()],thinking=False,post=post,context_for=lambda _:[],cancel_event=cancel)

    def test_export_round_trip_all_fields_no_accounts_or_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'qa.json';original=entry()
            export_bank([original],path)
            loaded,skipped=parse_bank_file(path)
            self.assertEqual(skipped,0)
            self.assertEqual(asdict(loaded[0]),asdict(original))
            self.assertEqual(set(json.loads(path.read_text(encoding='utf-8')) ),{'version','items'})
            self.assertFalse(path.with_name('qa.json.writing').exists())

    def test_template_is_importable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'template.json';export_bank(template_bank(),path)
            entries,skipped=parse_bank_file(path)
            self.assertEqual(len(entries),1);self.assertEqual(skipped,0)
            self.assertIn('格式示例',entries[0].note)


class UiTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import QaPanel,QaBankDialog
        self.app=QApplication.instance() or QApplication([])
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.settings=DefenseSettings(base_dir=Path(self.tmp.name))
        self.engine=MagicMock();self.engine.answer_ready.return_value=False
        self.panel=QaPanel(self.settings,self.engine);self.addCleanup(self.panel.shutdown)
        self.panel.bank_store.save([entry()]);self.panel.reload_bank()
        self.dialog=QaBankDialog(self.panel);self.dialog.refresh()
        self.addCleanup(self.dialog.shutdown);self.addCleanup(self.dialog.deleteLater)
        self.seen=[]
        def post(payload):
            self.seen.append(payload)
            return json.dumps({'results':[result(e['id']) for e in json.loads(payload['messages'][1]['content'])]})
        self.dialog.review_post_factory=lambda cancel:post
        class Preview:
            def __init__(self,entries,reviews):self.reviews=reviews
            def exec(self):return 1
            def chosen(self):return self.reviews
        self.dialog._answer_review_dialog=lambda entries,reviews:Preview(entries,reviews)

    def wait(self):
        end=time.monotonic()+3
        while self.dialog._review_running and time.monotonic()<end:
            self.app.processEvents();time.sleep(.005)
        self.assertFalse(self.dialog._review_running)

    def test_apply_makes_backup_preserves_question_metadata_and_invalidates_audio(self):
        original=asdict(self.panel.bank[0])
        self.dialog.start_answer_review();self.wait()
        updated=asdict(self.panel.bank[0]);self.assertEqual(updated['answer'],'My study estimates dimensions.')
        for key in original:
            if key not in ('answer','answer_zh'):self.assertEqual(updated[key],original[key])
        files=list((self.settings.base_dir/'qa-bank-backups').glob('*.json'))
        self.assertEqual(len(files),1)
        self.assertEqual(asdict(parse_bank_file(files[0])[0][0]),original)
        self.engine.delete_answer_audio.assert_called_once_with(original['answer'])

    def test_changed_bank_during_check_is_not_overwritten(self):
        gate=threading.Event()
        def post(_):gate.wait(1);return json.dumps({'results':[result()]})
        self.dialog.review_post_factory=lambda cancel:post
        self.dialog.start_answer_review()
        self.panel.bank[0].answer='manual edit';gate.set();self.wait()
        self.assertEqual(self.panel.bank[0].answer,'manual edit')

    def test_stop_discards_late_results(self):
        gate=threading.Event();original=self.panel.bank[0].answer
        def post(_):gate.wait(1);return json.dumps({'results':[result()]})
        self.dialog.review_post_factory=lambda cancel:post
        self.dialog.start_answer_review();self.dialog._review_cancel.set();gate.set();self.wait()
        self.assertEqual(self.panel.bank[0].answer,original)

    def test_export_button_and_template_roundtrip(self):
        destination=Path(self.tmp.name)/'export.json'
        with patch('teams_voice_translator.defense.ui.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
            self.dialog.export_file()
        loaded,skipped=parse_bank_file(destination)
        self.assertEqual(asdict(loaded[0]),asdict(self.panel.bank[0]))
        self.dialog.import_entries(loaded,'replace')
        self.assertEqual(len(self.panel.bank),1)

    def test_thinking_setting_independent_and_selected_scope(self):
        self.dialog.review_thinking_check.setChecked(False)
        self.assertFalse(self.settings.get('qa_review_thinking'))
        self.dialog.table.selectRow(0);self.dialog.start_answer_review(True);self.wait()
        self.assertFalse(self.seen[0]['enable_thinking'])

    def test_check_status_cannot_be_selected_in_preview(self):
        from teams_voice_translator.defense.qa_review_ui import AnswerReviewDialog
        from PySide6.QtCore import Qt
        d=AnswerReviewDialog(self.dialog,[entry()],[result(status='check')])
        self.assertEqual(d.chosen(),[])
        self.assertFalse(d.table.item(0,0).flags() & Qt.ItemIsUserCheckable)
        d.deleteLater()

    def test_close_cancels_review_without_applying_late_answer(self):
        gate=threading.Event();original=self.panel.bank[0].answer
        def post(_):gate.wait(1);return json.dumps({'results':[result()]})
        self.dialog.review_post_factory=lambda cancel:post
        self.dialog.start_answer_review();self.dialog.close();gate.set();self.wait()
        self.assertEqual(self.panel.bank[0].answer,original)
