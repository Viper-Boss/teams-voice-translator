"""第十四批：问答库中英完整 + 导入其他 AI 问答 + 批量生成容错 + 多文件导入 + 推荐高亮 +
设置与情景模式联动 + 讲稿台合并 + 暂停/重播 + 早开口 + 持续背景音 + 异常兜底。

全部离线：大模型、合成、声卡都用假的。测试里的问答都是编的，不含真实答辩内容。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from teams_voice_translator.defense import scene_modes
from teams_voice_translator.defense.qa_bank import QaBankStore, QaEntry
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.thesis_study import (
    MAX_QA_COUNT, StudyNotes, StudySection, StudyStore, apply_chinese, needs_chinese,
    parse_fill_zh_response, parse_import_response, parse_qa_response, salvage_items, split_import_text,
)


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(predicate, rounds=400):
    app = _app()
    for _ in range(rounds):
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _qa_json(n: int, start: int = 1) -> str:
    return json.dumps({"items": [
        {"q_en": f"Question {i}?", "q_zh": f"问题{i}", "a_en": f"Answer {i}.", "a_zh": f"回答{i}", "pri": 2}
        for i in range(start, start + n)
    ]})


# ---------------------------------------------------------------- JSON 容错
class SalvageTests(unittest.TestCase):
    def test_未转义引号的条目也能救回来(self):
        raw = ('{"items": [{"q_en": "Why?", "a_en": "Because "windows" help.", "pri": 3},'
               ' {"q_en": "Data?", "a_en": "Simulated.", "pri": 2}]}')
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        entries = parse_qa_response(raw)
        self.assertEqual([e.question for e in entries], ["Why?", "Data?"])
        self.assertIn('"windows"', entries[0].answer)

    def test_被截断的最后一条丢掉_不要半截回答(self):
        raw = '{"items": [{"q_en": "A?", "a_en": "One."}, {"q_en": "B?", "a_en": "Two and th'
        self.assertEqual([i["q_en"] for i in salvage_items(raw)], ["A?"])

    def test_完全不是JSON就报错(self):
        with self.assertRaises(Exception):
            parse_qa_response("对不起，我无法完成")

    def test_单次请求上限与总上限分开(self):
        self.assertGreaterEqual(MAX_QA_COUNT, 200)


class ImportParseTests(unittest.TestCase):
    def test_长材料按空行切块_不切开一组问答(self):
        text = "\n\n".join(f"Q{i}: 问题{i}\nA{i}: " + "回答" * 50 for i in range(60))
        chunks = split_import_text(text, limit=1500)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 1500 for c in chunks))
        self.assertEqual("".join(chunks).count("Q"), 60)

    def test_识别结果编号I开头_标记外部导入(self):
        entries = parse_import_response(_qa_json(2), start=4)
        self.assertEqual([e.id for e in entries], ["I4", "I5"])
        self.assertEqual(entries[0].category, "外部导入")
        self.assertIn("导入", entries[0].note)


class FillChineseTests(unittest.TestCase):
    def test_只补空着的中文_你写的不覆盖(self):
        entries = [QaEntry("S1", "Q?", "A."), QaEntry("S2", "Q2?", "A2.", question_zh="我的问", answer_zh="")]
        found = parse_fill_zh_response(json.dumps({"items": [
            {"id": "S1", "q_zh": "问", "a_zh": "答"}, {"id": "S2", "q_zh": "别覆盖", "a_zh": "答2"}]}))
        updated, changed = apply_chinese(entries, found)
        self.assertEqual(changed, 2)
        self.assertEqual((updated[0].question_zh, updated[0].answer_zh), ("问", "答"))
        self.assertEqual((updated[1].question_zh, updated[1].answer_zh), ("我的问", "答2"))
        self.assertFalse(any(needs_chinese(e) for e in updated))


# ---------------------------------------------------------------- 问答库窗口
class BankDialogTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.ui import QaPanel

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        QaBankStore(Path(self.tmp.name)).save([
            QaEntry("S1", "Why simulate?", "Real data are rare.", question_zh="为什么仿真？", answer_zh="真实数据少。"),
            QaEntry("S2", "How fast?", "About six milliseconds."),
        ])
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = False
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.open_bank_dialog()
        self.dialog = self.panel._bank_dialog
        self.addCleanup(self.dialog.hide)

    def test_列表有中文问题_缺中文的标出来_窗口能最小化(self):
        from PySide6.QtCore import Qt

        self.assertEqual(self.dialog.table.item(0, 2).text(), "为什么仿真？")
        self.assertEqual(self.dialog.table.item(1, 2).text(), "（缺中文）")
        self.assertIn("缺中文", self.dialog.info_label.text())
        self.assertTrue(self.dialog.windowFlags() & Qt.WindowMinimizeButtonHint)

    def test_选中一条_下面显示完整中英问答_改了能保存(self):
        self.dialog.table.setCurrentCell(0, 2)
        self.dialog._load_detail(0)
        self.assertEqual(self.dialog.a_en_edit.text(), "Real data are rare.")
        self.assertEqual(self.dialog.a_zh_edit.text(), "真实数据少。")
        self.dialog.a_en_edit.setText("Real MFL data are rare and expensive.")
        self.assertTrue(self.dialog.save_detail_button.text().endswith("*"))
        self.dialog.save_detail()
        self.assertEqual(QaBankStore(Path(self.tmp.name)).load()[0].answer, "Real MFL data are rare and expensive.")
        self.assertIn("重新生成", self.dialog.detail_status.text())

    def test_英文不能存空(self):
        self.dialog._load_detail(1)
        self.dialog.a_en_edit.setText("")
        with mock.patch("teams_voice_translator.defense.ui.QMessageBox.warning") as warning:
            self.assertIsNone(self.dialog.save_detail())
        warning.assert_called_once()

    def test_补全缺的中文(self):
        translator = MagicMock()
        translator.complete.return_value = json.dumps({"items": [{"id": "S2", "q_zh": "多快？", "a_zh": "大约六毫秒。"}]})
        self.dialog.fill_translator_factory = lambda: translator
        self.dialog.fill_missing_chinese()
        self.assertTrue(_wait(lambda: self.dialog.fill_button.isEnabled()))
        saved = QaBankStore(Path(self.tmp.name)).load()
        self.assertEqual((saved[1].question_zh, saved[1].answer_zh), ("多快？", "大约六毫秒。"))
        self.assertEqual(saved[0].question_zh, "为什么仿真？", "已有中文不动")
        self.assertEqual(self.dialog.table.item(1, 2).text(), "多快？")

    def test_删除所选(self):
        self.dialog.table.selectRow(1)
        with mock.patch("teams_voice_translator.defense.ui.QMessageBox.question",
                        return_value=__import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.Yes):
            self.dialog.delete_selected()
        self.assertEqual([e.id for e in self.panel.bank], ["S1"])

    def test_搜索过滤(self):
        self.dialog.search_edit.setText("六毫秒")
        self.assertTrue(self.dialog.table.isRowHidden(0))
        self.dialog.search_edit.setText("six")
        self.assertFalse(self.dialog.table.isRowHidden(1))


class ImportAiDialogTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.ui import ImportQaDialog, QaPanel

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        QaBankStore(Path(self.tmp.name)).save([QaEntry("S1", "Question 1?", "Old answer.")])
        self.panel = QaPanel(self.settings, MagicMock())
        self.addCleanup(self.panel.shutdown)
        self.dialog = ImportQaDialog(None, self.panel)

    def test_本程序格式直接读_不调用大模型(self):
        self.dialog.translator_factory = MagicMock(side_effect=AssertionError("不该调用大模型"))
        self.dialog.source_edit.setPlainText("Q: What is new?\nA: A hybrid method.\nZH: 混合方法。")
        self.dialog.start_parse()
        self.assertEqual(self.dialog.table.rowCount(), 1)
        self.assertIn("没有调用大模型", self.dialog.status_label.text())

    def test_乱格式交给大模型_确认后导入_重复的默认不勾(self):
        translator = MagicMock()
        translator.complete.return_value = _qa_json(3)
        self.dialog.translator_factory = lambda: translator
        self.dialog.source_edit.setPlainText("1. 老师可能问：为什么……\n回答：因为……\n\n2. ……")
        self.dialog.start_parse()
        self.assertTrue(_wait(lambda: self.dialog.import_button.isEnabled()))
        self.assertEqual(self.dialog.table.rowCount(), 3)
        from PySide6.QtCore import Qt
        self.assertEqual(self.dialog.table.item(0, 0).checkState(), Qt.Unchecked, "Question 1 库里已有")
        self.dialog.table.item(1, 3).setText("Answer two, edited.")
        added = self.dialog.import_selected()
        self.assertEqual(added, 2)
        bank = QaBankStore(Path(self.tmp.name)).load()
        self.assertEqual([e.id for e in bank], ["S1", "I1", "I2"])
        self.assertEqual(bank[1].answer, "Answer two, edited.")
        self.assertEqual(bank[1].question_zh, "问题2")

    def test_某段识别失败自动重试一次(self):
        translator = MagicMock()
        translator.complete.side_effect = ["不是 JSON", _qa_json(1, start=7)]
        self.dialog.translator_factory = lambda: translator
        self.dialog.source_edit.setPlainText("随便一段材料")
        self.dialog.start_parse()
        self.assertTrue(_wait(lambda: self.dialog.parse_button.isEnabled()))
        self.assertEqual(translator.complete.call_count, 2)
        self.assertEqual(self.dialog.table.rowCount(), 1)


# ---------------------------------------------------------------- 论文预习窗口
class StudyDialogTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.thesis_context import ThesisStore
        from teams_voice_translator.defense.ui import QaPanel, StudyDialog

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.settings = DefenseSettings(base_dir=base)
        ThesisStore(base).save(source_path="a.pdf; b.docx", chunks=[], page_count=3, full_text="thesis text", brief="")
        StudyStore(base).save(StudyNotes([StudySection("节", "Sec", "中文", "English")], "m", "t"))
        self.panel = QaPanel(self.settings, MagicMock())
        self.addCleanup(self.panel.shutdown)
        self.dialog = StudyDialog(None, self.settings, self.panel)
        self.addCleanup(self.dialog.shutdown)

    def test_能最小化_题数上限_多文件名_字号按钮(self):
        from PySide6.QtCore import Qt

        self.assertTrue(self.dialog.windowFlags() & Qt.WindowMinimizeButtonHint)
        self.assertEqual(self.dialog.qa_count_spin.maximum(), MAX_QA_COUNT)
        self.assertIn("a.pdf、b.docx", self.dialog.info_label.text())
        before = self.settings.get("study_font_px")
        self.dialog.font_up_button.click()
        self.assertEqual(self.settings.get("study_font_px"), before + 1)
        self.assertIn(f"{before + 1}px", self.dialog.styleSheet())
        self.assertIn("问答库", self.dialog.bank_button.text())

    def test_某批JSON坏了_重试后跳过_继续生成(self):
        translator = MagicMock()
        replies = iter(["坏的", "还是坏的", _qa_json(10), _qa_json(5, start=20)])
        translator.complete.side_effect = lambda *a, **k: next(replies)
        self.dialog.translator_factory = lambda model: translator
        self.dialog.qa_count_spin.setValue(15)
        results = []
        self.dialog.qa_done.connect(lambda added, error: results.append((added, error)))
        self.dialog.start_qa()
        self.assertTrue(_wait(lambda: results, rounds=800))
        added, error = results[0]
        self.assertEqual(added, 15)
        self.assertIn("1 批出错已跳过", error)
        self.assertEqual(len(QaBankStore(Path(self.tmp.name)).load()), 15)

    def test_连续三批失败就停下_不再花钱(self):
        translator = MagicMock()
        translator.complete.return_value = "完全不是 JSON"
        self.dialog.translator_factory = lambda model: translator
        self.dialog.qa_count_spin.setValue(50)
        results = []
        self.dialog.qa_done.connect(lambda added, error: results.append((added, error)))
        self.dialog.start_qa()
        self.assertTrue(_wait(lambda: results, rounds=800))
        self.assertEqual(results[0][0], 0)
        self.assertIn("连续 3 批失败", results[0][1])
        self.assertEqual(translator.complete.call_count, 6, "每批最多试两次，三批就停")


# ---------------------------------------------------------------- 推荐模型高亮
class HighlightTests(unittest.TestCase):
    def test_推荐模型加星加粗(self):
        from PySide6.QtCore import Qt
        from teams_voice_translator.defense.ui import (ASR_MODELS, RECOMMENDED_ASR, NoWheelComboBox,
                                                       fill_model_combo)

        _app()
        combo = NoWheelComboBox()
        fill_model_combo(combo, ASR_MODELS, "fun-asr-realtime", RECOMMENDED_ASR)
        top = combo.findData(RECOMMENDED_ASR[0])
        self.assertTrue(combo.itemText(top).startswith("★ "))
        self.assertNotIn("★ 推荐：★", combo.itemText(top))
        self.assertTrue(combo.itemData(top, Qt.FontRole).bold())
        other = combo.findData("fun-asr-realtime")
        self.assertFalse(combo.itemText(other).startswith("★"))
        self.assertEqual(combo.currentData(), "fun-asr-realtime")


# ---------------------------------------------------------------- 主窗口：多文件、联动、暂停、异常兜底
class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teams_voice_translator.defense.ui import DefenseWindow

        _app()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.settings = DefenseSettings(base_dir=Path(cls._tmp.name))
        cls.window = DefenseWindow(cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls._tmp.cleanup()

    def test_顶栏没有讲稿预习_讲稿台有翻译和暂停(self):
        self.assertFalse(hasattr(self.window, "rehearsal_button"))
        desk = self.window.script_panel
        self.assertTrue(hasattr(desk, "translate_button"))
        self.assertTrue(hasattr(desk, "pause_button"))

    def test_讲稿台中文稿逐段翻成英文_可撤销(self):
        desk = self.window.script_panel
        calls = []

        def runner(chinese, previous):
            calls.append((chinese, previous))
            return f"EN({chinese})"

        desk.translate_runner = runner
        desk.source_edit.setPlainText("第一段。\n\n第二段。")
        desk._translate_source()
        self.assertTrue(_wait(lambda: desk.undo_translate_button.isEnabled()))
        self.assertEqual(desk.source_edit.toPlainText(), "EN(第一段。)\n\nEN(第二段。)")
        self.assertEqual(calls[1], ("第二段。", "EN(第一段。)"), "带上一段英文保持衔接")
        desk._undo_translate()
        self.assertEqual(desk.source_edit.toPlainText(), "第一段。\n\n第二段。")

    def test_多个文件一次导入(self):
        from teams_voice_translator.defense import ui
        ppt = MagicMock(brief="ppt brief", terms=[{"source": "漏磁", "target": "MFL"}], slide_count=12,
                        source_path="deck.pptx")
        thesis = MagicMock(brief="thesis brief", terms=[{"source": "蒸馏", "target": "distillation"}],
                           chunks=[], page_count=3, char_count=3000, source_path="a.pdf; b.docx",
                           full_text="text")
        finished = []
        self.window.signals.import_finished.connect(lambda message, thesis_flag: finished.append(message))
        with mock.patch.object(ui, "BailianClient"), mock.patch.object(ui, "ContextTranslator"), \
                mock.patch.object(ui, "import_presentation", return_value=ppt) as import_ppt, \
                mock.patch.object(ui, "import_thesis", return_value=thesis) as import_paper, \
                mock.patch.object(self.window, "_ensure_pdf_support", return_value=True), \
                mock.patch.object(ui.QMessageBox, "information"):
            self.window.import_files(["deck.pptx", "a.pdf", "b.docx"])
            self.assertTrue(_wait(lambda: finished))
        import_ppt.assert_called_once()
        self.assertEqual(import_paper.call_args.args[1], ["a.pdf", "b.docx"], "论文类文件合成一份")
        self.assertEqual(self.settings.get("defense_brief"), "thesis brief", "有论文时以论文摘要为准")
        sources = {t["source"] for t in self.settings.glossary_terms()}
        self.assertTrue({"漏磁", "蒸馏"} <= sources)
        self.assertIn("✓ PPT deck.pptx", finished[0])
        self.assertTrue(self.window.ppt_button.isEnabled())

    def test_暂停按钮_面板和记录区和输入栏同步(self):
        engine = self.window.engine
        panel = self.window.qa_panel
        with mock.patch.object(type(engine), "speech_paused", new_callable=mock.PropertyMock, return_value=False), \
                mock.patch.object(engine, "pause_speech") as pause:
            self.window.timeline_pause_button.click()
        pause.assert_called_once()
        self.assertEqual(panel.pause_button.text(), "▶ 继续")
        self.assertIn("继续", self.window.input_pause_button.text())
        self.window._on_cancel()
        self.assertEqual(panel.pause_button.text(), "⏸ 暂停", "Esc 也解除暂停")

    def test_重播上一次读的(self):
        panel = self.window.qa_panel
        with mock.patch.object(self.window.engine, "speak_prepared", return_value=True) as speak:
            panel.answer_en.setText("This is the answer.")
            panel._play_answer()
            panel.replay_last()
        self.assertEqual(speak.call_count, 2)
        self.assertEqual(speak.call_args.args[0], "This is the answer.")
        self.assertIn("重播", speak.call_args.kwargs["label"])

    def test_界面里漏掉的异常只提示_不退出(self):
        messages = []
        self.window.signals.error_message.connect(messages.append)
        sys.excepthook(ValueError, ValueError("boom"), None)
        _wait(lambda: messages)
        self.assertTrue(any("内部错误已拦截" in m for m in messages))


class SettingsSceneLinkTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.dialog = SettingsDialog(None, self.settings)

    def test_设置页显示当前情景模式_选模式填好各项_手改变自定义(self):
        dialog = self.dialog
        self.assertEqual(dialog.scene_combo.currentData(), "recommended")
        dialog._on_settings_scene_chosen(dialog.scene_combo.findData("legacy_all"))
        self.assertEqual(dialog.llm_combo.currentData(), "qwen-plus")
        self.assertEqual(dialog.committee_asr_combo.currentData(), "qwen3-asr-flash-realtime")
        self.assertEqual(dialog.scene_combo.currentData(), "legacy_all")
        dialog.llm_combo.setCurrentIndex(dialog.llm_combo.findData("qwen3.7-plus"))
        self.assertEqual(dialog.scene_combo.currentData(), scene_modes.CUSTOM_ID)

    def test_同传模式来回切(self):
        dialog = self.dialog
        dialog._on_settings_scene_chosen(dialog.scene_combo.findData("livetranslate"))
        self.assertEqual(dialog._engine_values()["my_translate_engine"], "livetranslate")
        self.assertEqual(dialog.scene_combo.currentData(), "livetranslate")
        dialog._on_settings_scene_chosen(dialog.scene_combo.findData("recommended"))
        self.assertEqual(dialog._engine_values()["my_translate_engine"], "llm")


# ---------------------------------------------------------------- 引擎：早开口、暂停、持续背景音
class EngineSpeechTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.engine = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(self.engine.shutdown)

    def _run_job(self, deltas: list[str]):
        from teams_voice_translator.defense.pipeline import _Job

        batches = []
        tts = MagicMock()
        tts.speak_batch.side_effect = lambda texts: batches.append(list(texts)) or True
        self.engine.tts = tts

        class FakeTranslator:
            def translate(self, memory, direction, source, on_delta=None, cancel_event=None):
                so_far = ""
                for piece in deltas:
                    so_far += piece
                    on_delta(piece, so_far)
                return so_far

        self.engine.translator = FakeTranslator()
        self.engine._translate_my_job(_Job(seg_id=1, source="中文", t0=time.monotonic()))
        return batches

    def test_早开口_第一句先送_后面补上_不重复(self):
        batches = self._run_job(["The method has three parts. ", "First we simulate data. ",
                                 "Then we train the model."])
        self.assertEqual(len(batches), 2)
        self.assertEqual(" ".join(batches[0]), "The method has three parts.")
        self.assertEqual(" ".join(batches[1]), "First we simulate data. Then we train the model.")

    def test_第一句太短就和下一句拼一起再开口(self):
        batches = self._run_job(["Yes. ", "It is a hybrid method that works. ", "Done."])
        self.assertEqual(" ".join(batches[0]), "Yes. It is a hybrid method that works.")
        self.assertEqual(" ".join(batches[1]), "Done.")

    def test_关掉早开口就等整段(self):
        self.settings.set("speech_early_start", False)
        batches = self._run_job(["The method has three parts. ", "First we simulate data."])
        self.assertEqual(len(batches), 1)

    def test_余下部分计算(self):
        from teams_voice_translator.defense.pipeline import remainder_after

        self.assertEqual(remainder_after("A b c d e. F g.", "A b c d e."), "F g.")
        self.assertEqual(remainder_after("A  b c d e. F g.", "A b c d e."), "F g.")
        self.assertEqual(remainder_after("Only this.", "Only this."), "")

    def test_断句静音不再强制700ms(self):
        self.settings.set("vad_silence_ms", 500)
        self.assertEqual(self.engine._vad_silence_ms(), 500)
        self.settings.set("vad_silence_ms", 100)
        self.assertEqual(self.engine._vad_silence_ms(), 400)

    def test_暂停时实时语音停住_继续后接着写(self):
        player = MagicMock()
        self.engine.player = player
        self.engine.pause_speech()
        worker = threading.Thread(target=self.engine._on_tts_audio, args=(b"\x01\x00" * 100,))
        worker.start()
        time.sleep(0.15)
        player.write.assert_not_called()
        self.engine.resume_speech()
        worker.join(2)
        player.write.assert_called_once()

    def test_暂停时按Esc_这块不再播(self):
        player = MagicMock()
        self.engine.player = player
        self.engine.pause_speech()
        worker = threading.Thread(target=self.engine._on_tts_audio, args=(b"\x01\x00" * 100,))
        worker.start()
        time.sleep(0.1)
        self.engine.interrupt_speech()
        worker.join(2)
        player.write.assert_not_called()
        self.assertFalse(self.engine.speech_paused)

    def test_持续背景音_没人说话也在写_说话时让开(self):
        self.settings.set("ambience_mode", "subtle")
        player = MagicMock()
        player.write.side_effect = lambda data: time.sleep(0.01)
        self.engine.player = player
        self.engine._session_active = True
        self.engine._start_ambience_bed()
        self.assertTrue(self.engine._ambience_bed_on())
        time.sleep(0.2)
        self.assertGreater(player.write.call_count, 3)
        first = player.write.call_args.args[0]
        self.assertEqual(len(first), int(24000 * 0.04) * 2)
        self.assertTrue(any(first), "底噪不是全零")
        # 刚写过语音：底噪让开
        self.engine._last_voice_write = time.monotonic() + 5
        time.sleep(0.05)
        count = player.write.call_count
        time.sleep(0.15)
        self.assertEqual(player.write.call_count, count)
        self.engine._stop_ambience_bed()
        self.assertFalse(self.engine._ambience_bed_on())

    def test_关掉背景音就不起线程(self):
        self.settings.set("ambience_mode", "off")
        self.engine._session_active = True
        self.engine._start_ambience_bed()
        self.assertIsNone(self.engine._ambience_thread)


if __name__ == "__main__":
    unittest.main()
