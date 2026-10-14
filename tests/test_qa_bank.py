"""问答库：导入解析、本地检索、大模型匹配（全部 mock，不发真实请求）、预案播放。

测试里的问答都是编的，不含任何真实答辩内容。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.defense.memory import MeetingMemory
from teams_voice_translator.defense.qa import build_qa_messages
from teams_voice_translator.defense.qa_bank import (
    LEXICAL_STRONG,
    QaBankError,
    QaBankStore,
    QaEntry,
    QaIndex,
    build_match_messages,
    match_question,
    merge_entries,
    parse_bank_file,
    parse_json_bank,
    parse_match,
    parse_text_bank,
    reference_pairs,
    tokenize,
)


def sample_entries() -> list[QaEntry]:
    return [
        QaEntry("A1", "Why did you choose a transformer backbone instead of a CNN?",
                "Because windows capture local boundaries.", question_zh="为什么选 Transformer？"),
        QaEntry("A2", "All your data are simulated. How do you know it works on real pipelines?",
                "It is the main limitation; next I will test measured data.",
                aliases=["Did you validate with field data?"]),
        QaEntry("A3", "Why did pruning not reduce inference time?",
                "Runtime depends on memory access, not only parameters.", note="记得说 GPU"),
        QaEntry("A4", "How did you choose the loss weights?",
                "They were fixed for all groups so the comparison is fair."),
    ]


class ParseJsonTests(unittest.TestCase):
    def test_预案库格式_字段全部映射(self):
        text = json.dumps({
            "categories": [{"key": "B", "name": "数据与仿真"}],
            "items": [{
                "id": "B1", "cat": "B", "pri": 3, "basis": "limit",
                "q_en": "Is the data simulated?", "q_zh": "数据是仿真的吗？",
                "points": ["承认", "给下一步"], "a_en": "Yes, it is.", "a_zh": "是的。",
                "src": "5.2", "flag": "别辩解",
            }],
        }, ensure_ascii=False)
        entries, skipped = parse_json_bank(text)
        self.assertEqual(skipped, 0)
        entry = entries[0]
        self.assertEqual((entry.id, entry.question, entry.answer), ("B1", "Is the data simulated?", "Yes, it is."))
        self.assertEqual(entry.category, "数据与仿真")
        self.assertEqual(entry.priority, 3)
        self.assertEqual(entry.note, "别辩解")
        self.assertEqual(entry.answer_zh, "是的。")
        self.assertEqual(entry.points, ["承认", "给下一步"])

    def test_纯数组和通用字段名(self):
        entries, _ = parse_json_bank('[{"question": "Q?", "answer": "A."}]')
        self.assertEqual(entries[0].id, "Q1")

    def test_缺问题或缺回答的条目被跳过而不是报错(self):
        entries, skipped = parse_json_bank('[{"question": "Q?"}, {"answer": "A."}, {"q": "x", "a": "y"}, 5]')
        self.assertEqual(len(entries), 1)
        self.assertEqual(skipped, 3)

    def test_坏JSON给出人能看懂的错误(self):
        with self.assertRaises(QaBankError):
            parse_json_bank("{broken")
        with self.assertRaises(QaBankError):
            parse_json_bank('{"foo": 1}')


class ParseTextTests(unittest.TestCase):
    def test_英文前缀_续行_可选字段(self):
        text = (
            "1. Q: Why simulation?\n"
            "A: Because labels\n"
            "are exact.\n"
            "ZH: 标签精确\n"
            "Alt: Why not real data?\n"
            "\n"
            "- **Q**: Second question?\n"
            "A: Second answer.\n"
        )
        entries, skipped = parse_text_bank(text)
        self.assertEqual(skipped, 0)
        self.assertEqual(entries[0].answer, "Because labels are exact.")
        self.assertEqual(entries[0].answer_zh, "标签精确")
        self.assertEqual(entries[0].aliases, ["Why not real data?"])
        self.assertEqual(entries[1].question, "Second question?")

    def test_中文前缀(self):
        entries, _ = parse_text_bank("问：为什么用小波？\n答：因为能保留边界。")
        self.assertEqual(entries[0].question, "为什么用小波？")
        self.assertEqual(entries[0].answer, "因为能保留边界。")

    def test_回答里带冒号的普通句子不会被当成新字段(self):
        entries, _ = parse_text_bank("Q: What is the result?\nA: Two things.\nFirst: accuracy improves.")
        self.assertEqual(entries[0].answer, "Two things. First: accuracy improves.")

    def test_只有问题没有回答计入跳过(self):
        entries, skipped = parse_text_bank("Q: lonely\nQ: ok?\nA: yes")
        self.assertEqual(len(entries), 1)
        self.assertEqual(skipped, 1)

    def test_文件入口_空文件报错_gbk可读(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.txt"
            empty.write_text("nothing here", encoding="utf-8")
            with self.assertRaises(QaBankError):
                parse_bank_file(empty)
            gbk = Path(tmp) / "gbk.md"
            gbk.write_bytes("问：中文问题？\n答：中文回答。".encode("gbk"))
            entries, _ = parse_bank_file(gbk)
            self.assertEqual(entries[0].answer, "中文回答。")


class StoreAndMergeTests(unittest.TestCase):
    def test_保存再读回完全一致(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QaBankStore(Path(tmp))
            self.assertEqual(store.load(), [])
            store.save(sample_entries())
            self.assertEqual(store.load(), sample_entries())

    def test_文件损坏当空库而不是崩溃(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QaBankStore(Path(tmp))
            store.path.write_text("{not json", encoding="utf-8")
            self.assertEqual(store.load(), [])

    def test_追加导入编号冲突加后缀_不覆盖已有(self):
        existing = sample_entries()[:2]
        incoming = [QaEntry("A1", "new?", "new."), QaEntry("A1", "newer?", "newer.")]
        merged = merge_entries(existing, incoming)
        self.assertEqual([e.id for e in merged], ["A1", "A2", "A1-2", "A1-3"])
        self.assertEqual(merged[0].answer, existing[0].answer)


class LexicalTests(unittest.TestCase):
    def test_停用词和客套话不产生匹配(self):
        self.assertEqual(QaIndex(sample_entries()).search("Thank you, okay, well, can you tell us?"), [])

    def test_同义问法排在第一(self):
        hits = QaIndex(sample_entries()).search("why was the pruned model not faster at inference")
        self.assertEqual(hits[0][0].id, "A3")

    def test_别问法参与匹配(self):
        hits = QaIndex(sample_entries()).search("did you validate it with field data")
        self.assertEqual(hits[0][0].id, "A2")

    def test_中文二字组(self):
        self.assertIn("为什", tokenize("为什么"))
        hits = QaIndex(sample_entries()).search("为什么选Transformer")
        self.assertEqual(hits[0][0].id, "A1")

    def test_分数在0到1之间_原问题几乎满分(self):
        entries = sample_entries()
        score = QaIndex(entries).search(entries[3].question)[0][1]
        self.assertGreater(score, LEXICAL_STRONG)
        self.assertLessEqual(score, 1.0)


class ParseMatchTests(unittest.TestCase):
    ids = ["A1", "A2"]

    def test_标准JSON(self):
        self.assertEqual(parse_match('{"id": "A2", "confidence": 0.9}', self.ids), ("A2", 0.9))

    def test_包在多余文字里的JSON(self):
        self.assertEqual(parse_match('Sure! {"id":"A1","confidence":1.4} done', self.ids), ("A1", 1.0))

    def test_不存在的编号当作没命中(self):
        self.assertEqual(parse_match('{"id": "Z9", "confidence": 0.9}', self.ids)[0], None)

    def test_NONE(self):
        self.assertEqual(parse_match('{"id": "NONE", "confidence": 0.1}', self.ids)[0], None)
        self.assertEqual(parse_match("NONE, it is A1-ish", self.ids)[0], None)

    def test_没给JSON但只提到一个编号(self):
        entry_id, confidence = parse_match("The answer is A1.", self.ids)
        self.assertEqual(entry_id, "A1")
        self.assertGreater(confidence, 0)

    def test_提到多个编号不敢选(self):
        self.assertEqual(parse_match("A1 or A2", self.ids)[0], None)

    def test_A1不会误认A10(self):
        self.assertEqual(parse_match("A10", ["A1", "A10"])[0], "A10")


class MatchQuestionTests(unittest.TestCase):
    def test_大模型命中_候选第一个就是它(self):
        chat = MagicMock(return_value='{"id": "A3", "confidence": 0.93}')
        result = match_question(sample_entries(), "Pruning did not speed things up, why?", chat)
        self.assertTrue(result.matched)
        self.assertEqual(result.entry.id, "A3")
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.candidates[0][0].id, "A3")

    def test_置信度不够当没命中(self):
        chat = MagicMock(return_value='{"id": "A3", "confidence": 0.4}')
        result = match_question(sample_entries(), "Something about speed", chat)
        self.assertFalse(result.matched)

    def test_小库整库送给大模型_词面零分的条目也在里面(self):
        chat = MagicMock(return_value='{"id": "NONE", "confidence": 0}')
        match_question(sample_entries(), "pruning speed", chat)
        user = chat.call_args[0][0][1]["content"]
        for entry in sample_entries():
            self.assertIn(f"[{entry.id}]", user)

    def test_大库只送最像的前N条(self):
        entries = [QaEntry(f"X{i}", f"topic{i} question", "a") for i in range(30)]
        entries.append(QaEntry("HIT", "pruning inference time", "b"))
        chat = MagicMock(return_value='{"id": "HIT", "confidence": 0.9}')
        result = match_question(entries, "pruning inference time", chat, max_candidates=5)
        user = chat.call_args[0][0][1]["content"]
        self.assertIn("[HIT]", user)
        self.assertLessEqual(user.count("["), 5)
        self.assertEqual(result.entry.id, "HIT")

    def test_断网时退回本地_高分才接受(self):
        entries = sample_entries()
        boom = MagicMock(side_effect=RuntimeError("network down"))
        strong = match_question(entries, entries[3].question, boom)
        self.assertEqual((strong.method, strong.entry.id), ("lexical", "A4"))
        self.assertIn("network", strong.error)
        weak = match_question(entries, "weights?", boom)
        self.assertFalse(weak.matched)
        self.assertTrue(weak.error)

    def test_没有大模型只用本地(self):
        entries = sample_entries()
        self.assertEqual(match_question(entries, entries[0].question, None).method, "lexical")

    def test_空库或空问题直接返回(self):
        chat = MagicMock()
        self.assertFalse(match_question([], "x", chat).matched)
        self.assertFalse(match_question(sample_entries(), "  ", chat).matched)
        chat.assert_not_called()

    def test_追问带上前文(self):
        messages = build_match_messages("Can you explain that?", sample_entries(), ["About your pruning results."])
        user = messages[1]["content"]
        self.assertIn("About your pruning results.", user)
        self.assertIn("Latest remark: Can you explain that?", user)

    def test_提醒字段永远不进提示词(self):
        user = build_match_messages("q", sample_entries())[1]["content"]
        self.assertNotIn("记得说 GPU", user)


class ReferenceTests(unittest.TestCase):
    def test_没命中时相关预案作为参考放进用户消息(self):
        refs = reference_pairs([(sample_entries()[0], 0.3), (sample_entries()[1], 0.0)])
        self.assertEqual(len(refs), 1)
        messages = build_qa_messages(MeetingMemory(), "Why transformers?", refs)
        self.assertNotIn("windows capture", messages[0]["content"])
        self.assertIn("windows capture", messages[1]["content"])
        self.assertTrue(messages[1]["content"].rstrip().endswith("Why transformers?"))

    def test_不给参考时提示词与原来一致(self):
        memory = MeetingMemory()
        self.assertEqual(build_qa_messages(memory, "Q?"), build_qa_messages(memory, "Q?", None))
        self.assertNotIn("事先准备", build_qa_messages(memory, "Q?")[1]["content"])


class EnginePreparedAnswerTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.settings import DefenseSettings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.settings.set("tts_voice_id", "voice-test")
        self.done = []
        self.engine = DefenseEngine(self.settings, EngineCallbacks(
            on_my_done=lambda seg, text, latency: self.done.append(text)))
        self.addCleanup(self.engine.shutdown)
        # 命令线程改成同步执行，播放和合成全部 mock
        self.engine._submit = lambda action, _desc: action()
        self.engine._play_cached = MagicMock()
        self.engine._queue_speech = MagicMock()
        self.engine._ensure_voice = MagicMock()
        self.engine.client = object()

    def test_预渲染后秒播_不走现场合成(self):
        text = "It is the main limitation."
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x01\x00" * 100) as synth:
            self.assertFalse(self.engine.answer_ready(text))
            self.assertEqual(self.engine.render_answer(text), 1)
            self.assertEqual(self.engine.render_answer(text), 0, "已渲染的不应再请求")
            self.assertEqual(synth.call_count, 1)
        self.assertTrue(self.engine.answer_ready(text))
        self.assertTrue(self.engine.speak_prepared(text, label="预案 A2", zh="局限"))
        self.engine._play_cached.assert_called_once()
        self.engine._queue_speech.assert_not_called()
        self.assertEqual(self.done, [text])
        self.assertIn(text, self.engine.memory.render_history())

    def test_改过的回答现场合成(self):
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x01\x00" * 100):
            self.engine.render_answer("Original answer.")
        self.assertFalse(self.engine.speak_prepared("Original answer, edited."))
        self.engine._queue_speech.assert_called_once_with("Original answer, edited.")
        self.engine._ensure_voice.assert_called_once()
        self.engine._play_cached.assert_not_called()

    def test_长回答按块渲染_整段才算就绪(self):
        long_text = " ".join(f"Sentence number {i} explains one more detail of the method." for i in range(20))
        with patch("teams_voice_translator.defense.tts_session.synthesize_once",
                   return_value=b"\x01\x00" * 10) as synth:
            rendered = self.engine.render_answer(long_text)
        self.assertGreater(rendered, 1)
        self.assertEqual(synth.call_count, rendered)
        self.assertTrue(self.engine.answer_ready(long_text))
        self.assertEqual(len(self.engine.answer_pcm(long_text)), 20 * rendered)

    def test_渲染可取消_没填音色报错(self):
        cancel = threading.Event()
        cancel.set()
        from teams_voice_translator.aliyun import ApiError
        with self.assertRaises(ApiError):
            self.engine.render_answer("Hello there.", cancel=cancel)
        self.settings.set("tts_voice_id", "")
        with self.assertRaises(ApiError):
            self.engine.render_answer("Hello there.")


class QaPanelFlowTests(unittest.TestCase):
    """界面流程：先查库 → 命中弹预案 / 没命中交给大模型；朗读读的是框里的字。"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import QaPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        QaBankStore(Path(self.tmp.name)).save(sample_entries())
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = True
        self.engine.speak_prepared.return_value = True
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.generated = []
        self.panel.generate = lambda question, references=None: self.generated.append((question, references))

    def run_match(self, reply, question):
        self.panel.match_chat_factory = lambda: (lambda messages: reply)
        received = []
        self.panel.qa_match.connect(received.append)
        self.panel.on_committee_done(question, "")
        for _ in range(200):
            self.app.processEvents()
            if received:
                break
            threading.Event().wait(0.01)
        self.app.processEvents()
        return received

    def test_启动时读入本机问答库(self):
        self.assertEqual(len(self.panel.bank), 4)
        self.assertIn("4", self.panel.bank_button.text())

    def test_命中后弹出预案_提醒单独显示不进回答框(self):
        self.run_match('{"id": "A3", "confidence": 0.9}', "Why was pruning not faster?")
        self.assertEqual(self.panel.answer_en.text(), sample_entries()[2].answer)
        self.assertIn("A3", self.panel.source_label.text())
        self.assertIn("记得说 GPU", self.panel.note_label.text())
        self.assertNotIn("记得说 GPU", self.panel.answer_en.text())
        self.assertEqual(self.generated, [], "命中时不应再调用大模型生成")

    def test_没命中交给大模型_并附上相关预案(self):
        self.run_match('{"id": "NONE", "confidence": 0}', "Why was pruning not faster in your tests?")
        self.assertEqual(len(self.generated), 1)
        question, references = self.generated[0]
        self.assertIn("pruning", question)
        self.assertTrue(references)

    def test_关掉先查问答库就直接生成(self):
        self.panel.bank_check.setChecked(False)
        self.panel.on_committee_done("Why was pruning not faster?", "")
        self.assertEqual(len(self.generated), 1)
        self.assertFalse(DefenseSettingsReload(self.tmp.name).get("qa_bank_enabled"))

    def test_朗读的是框里改过的文字(self):
        self.run_match('{"id": "A4", "confidence": 0.9}', "How did you pick the loss weights?")
        self.panel.answer_en.setText("They were fixed, and I did not tune them.")
        self.assertTrue(self.panel.restore_button.isEnabled())
        self.panel._play_answer()
        text = self.engine.speak_prepared.call_args[0][0]
        self.assertEqual(text, "They were fixed, and I did not tune them.")
        self.assertEqual(self.engine.speak_prepared.call_args[1]["zh"], "", "改过的回答不配原来的中文")
        self.panel._restore_entry_text()
        self.assertEqual(self.panel.answer_en.text(), sample_entries()[3].answer)

    def test_过时的匹配结果被丢弃(self):
        from teams_voice_translator.defense.qa_bank import MatchResult
        self.panel._match_serial = 5
        self.panel._on_qa_match((4, "old", MatchResult(sample_entries()[0], 0.9, "llm")))
        self.assertEqual(self.panel.answer_en.text(), "")

    def test_手动搜预案(self):
        self.panel.search_edit.setText("loss weights")
        self.panel._search_bank()
        self.assertEqual(self.panel._current_entry.id, "A4")
        self.assertTrue(self.panel.candidate_combo.isVisible() or self.panel.candidate_combo.count() > 0)

    def test_问答库窗口导入追加(self):
        self.panel.open_bank_dialog()
        dialog = self.panel._bank_dialog
        count = dialog.import_entries([QaEntry("A1", "Extra?", "Extra.")], "append")
        self.assertEqual(count, 5)
        self.assertEqual(self.panel.bank[-1].id, "A1-2")
        self.assertEqual(dialog.table.rowCount(), 5)
        dialog.hide()

    def test_预渲染在后台逐条执行并更新表格(self):
        self.engine.answer_ready.return_value = False
        self.panel.open_bank_dialog()
        dialog = self.panel._bank_dialog
        finished = []
        dialog.render_finished.connect(finished.append)
        dialog.start_render(self.panel.bank)
        for _ in range(300):
            self.app.processEvents()
            if finished:
                break
            threading.Event().wait(0.01)
        self.assertEqual(self.engine.render_answer.call_count, 4)
        self.assertIn("成功 4 条", finished[0])
        dialog.hide()


class QaPanelChineseTests(unittest.TestCase):
    """中英对照：预案自带中文直接显示；没带就机器翻译；改了英文自动重译；字号可调。"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import QaPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = False
        self.engine.speak_prepared.return_value = False
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.calls = []

        def fake_translate(text):
            self.calls.append(text)
            return f"译：{text}"

        self.panel.translate_factory = lambda: fake_translate
        self.panel._translate_timer.setInterval(10)

    def wait(self, condition, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            threading.Event().wait(0.01)
        self.app.processEvents()
        return condition()

    def test_预案自带中文_直接并排显示_不调用翻译(self):
        entry = QaEntry("Z1", "Q?", "It is fine.", answer_zh="没问题。", points=["要点一", "要点二"])
        self.panel.show_bank_entry(entry)
        self.assertEqual(self.panel.answer_zh.text(), "没问题。")
        self.assertFalse(self.panel.answer_zh.isReadOnly(), "中文框可以改，改完能让英文跟着改")
        self.assertIn("要点一", self.panel.points_label.text())
        self.assertEqual(self.calls, [])

    def test_预案没带中文_自动机器翻译(self):
        self.panel.show_bank_entry(QaEntry("Z2", "Q?", "Plain answer."))
        self.assertTrue(self.wait(lambda: self.panel.answer_zh.text() == "译：Plain answer."))
        self.assertIn("机器翻译", self.panel.answer_zh_caption.text())

    def test_改英文后停手自动重译_朗读时配新中文(self):
        self.panel.show_bank_entry(QaEntry("Z3", "Q?", "Original.", answer_zh="原文。"))
        self.panel.answer_en.setText("Edited answer.")
        self.assertTrue(self.wait(lambda: self.panel.answer_zh.text() == "译：Edited answer."))
        self.panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_args[1]["zh"], "译：Edited answer.")
        # 改回原文：直接用预案自带的中文，不再请求
        count = len(self.calls)
        self.panel.answer_en.setText("Original.")
        self.app.processEvents()
        self.assertEqual(self.panel.answer_zh.text(), "原文。")
        self.assertEqual(len(self.calls), count)

    def test_过时的翻译不覆盖(self):
        self.panel.answer_en.blockSignals(True)
        self.panel.answer_en.setText("Newest.")
        self.panel.answer_en.blockSignals(False)
        self.panel._translate_serial = 7
        self.panel._on_translate_ready((6, "Older.", "旧的", ""))
        self.assertNotEqual(self.panel.answer_zh.text(), "旧的")
        self.assertEqual(self.panel._zh_cache.get("Older."), "旧的", "过时的结果仍然缓存起来备用")

    def test_没配API时说明原因而不是卡在翻译中(self):
        self.panel.translate_factory = lambda: None
        self.panel.show_bank_entry(QaEntry("Z4", "Q?", "No api."))
        self.assertIn("未配置", self.panel.answer_zh_caption.text())

    def test_大模型生成的回答自带中文(self):
        self.panel._on_qa_answer("Generated.", "生成的。")
        self.assertEqual(self.panel.answer_zh.text(), "生成的。")
        self.panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_args[1]["zh"], "生成的。")

    def test_字号可调_有上下限_会保存(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        start = self.panel._font_px
        self.panel.change_font_size(+2)
        self.assertEqual(self.panel._font_px, start + 2)
        self.assertIn(f"font-size: {start + 2}px", self.panel.answer_en.styleSheet())
        self.assertEqual(DefenseSettings(base_dir=Path(self.tmp.name)).get("qa_font_px"), start + 2)
        for _ in range(30):
            self.panel.change_font_size(+2)
        self.assertEqual(self.panel._font_px, self.panel.FONT_MAX)
        for _ in range(30):
            self.panel.change_font_size(-2)
        self.assertEqual(self.panel._font_px, self.panel.FONT_MIN)


def DefenseSettingsReload(base):
    from teams_voice_translator.defense.settings import DefenseSettings
    return DefenseSettings(base_dir=Path(base))


if __name__ == "__main__":
    unittest.main()
