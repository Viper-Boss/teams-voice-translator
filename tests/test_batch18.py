"""第 18 批：术语表 / 问答库查重——只提示、逐组确认、不起冲突。"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.defense.dedupe import (  # noqa: E402
    QaDedupeSession, find_qa_pairs, find_term_groups, same_source_conflicts,
)
from teams_voice_translator.defense.qa_bank import QaBankStore, QaEntry  # noqa: E402

TERMS = [
    {"source": "漏磁检测", "target": "Magnetic Flux Leakage (MFL) inspection"},
    {"source": "漏磁 检测", "target": "MFL testing"},
    {"source": "漏磁探测", "target": "Magnetic flux leakage (MFL) inspection"},
    {"source": "MFL", "target": "MFL"},
    {"source": "漏磁检测技术", "target": "Magnetic Flux Leakage (MFL) inspection technology"},
    {"source": "漏场", "target": "Magnetic Field Leakage (MFL)"},
    {"source": "小波去噪", "target": "Wavelet denoising"},
]


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(predicate, seconds=5.0):
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TermGroupTests(unittest.TestCase):
    def test_中文相同必须合并_其余按对给出原因(self):
        groups = find_term_groups(TERMS)
        first = groups[0]
        self.assertTrue(first.must_merge)
        self.assertEqual(first.indices, [0, 1])
        reasons = {tuple(g.indices): g.reason for g in groups[1:]}
        self.assertIn("英文写法相同", reasons[(0, 2)])
        self.assertIn("包含", reasons[(0, 4)])
        self.assertIn("同一个缩写 MFL", reasons[(0, 5)])
        self.assertNotIn((0, 3), reasons, "只是共用缩写（MFL 和 漏磁检测(MFL)）是正常的两个术语")
        self.assertFalse(any(6 in g.indices for g in groups))
        self.assertEqual(same_source_conflicts(TERMS), [[0, 1]])

    def test_都保留过的不再提示_中文相同的永远提示(self):
        groups = find_term_groups(TERMS)
        ignored = {g.key(TERMS) for g in groups}
        again = find_term_groups(TERMS, ignored)
        self.assertEqual([g.indices for g in again], [[0, 1]])

    def test_导入合并时忽略空格大小写(self):
        from teams_voice_translator.defense.thesis_context import merge_terms

        merged, added = merge_terms([{"source": "漏磁检测", "target": "A"}],
                                    [{"source": "漏磁 检测", "target": "B"}, {"source": "mfl", "target": "MFL"},
                                     {"source": "MFL", "target": "MFL"}])
        self.assertEqual(added, 1)
        self.assertEqual([t["source"] for t in merged], ["漏磁检测", "mfl"])


class TermDedupeDialogTests(unittest.TestCase):
    def setUp(self):
        _app()

    def test_中文相同只能留一条_可以改写法_都保留记住(self):
        from PySide6.QtCore import Qt
        from teams_voice_translator.defense import ui

        dialog = ui.TermDedupeDialog(None, TERMS[:3] + TERMS[6:], set())
        self.assertEqual(dialog.group_count, 2)
        # 第一组：中文相同，两条都勾 → 不让过
        dialog.table.item(1, 0).setCheckState(Qt.Checked)
        with patch.object(ui.QMessageBox, "information") as info:
            dialog.apply_group()
        info.assert_called_once()
        dialog.table.item(1, 0).setCheckState(Qt.Unchecked)
        dialog.table.item(0, 2).setText("MFL inspection")
        self.assertFalse(dialog.keep_all_button.isEnabled())
        dialog.apply_group()
        # 第二组：英文相同（漏磁检测 / 漏磁探测）→ 都保留
        self.assertTrue(dialog.keep_all_button.isEnabled())
        dialog.keep_all()
        dialog.finish()
        result = dialog.result_terms()
        self.assertEqual([t["source"] for t in result], ["漏磁检测", "漏磁探测", "小波去噪"])
        self.assertEqual(result[0]["target"], "MFL inspection")
        self.assertEqual(len(dialog.new_ignored), 1)


class GlossarySaveTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def test_中文重复时保存被拦下_合并后才写入(self):
        from teams_voice_translator.defense import ui

        dialog = ui.GlossaryDialog(None, self.settings)
        self.addCleanup(dialog.deleteLater)
        dialog._fill(TERMS[:2])
        self.assertEqual(len(dialog.terms()), 2, "不再悄悄丢掉重复的那条")

        class Cancelled:
            group_count = 1
            new_ignored = set()

            def exec(self):
                from PySide6.QtWidgets import QDialog
                return QDialog.Rejected

        dialog._dedupe_dialog = lambda terms: Cancelled()
        with patch.object(ui.QMessageBox, "information"):
            dialog.save()
        self.assertEqual(self.settings.glossary_terms(), [], "取消查重就不保存")

        class Merged(Cancelled):
            def exec(self):
                from PySide6.QtWidgets import QDialog
                return QDialog.Accepted

            def result_terms(self):
                return [TERMS[0]]

        dialog._dedupe_dialog = lambda terms: Merged()
        with patch.object(ui.QMessageBox, "information"):
            dialog.save()
        self.assertEqual(self.settings.glossary_terms(), [TERMS[0]])


def _bank():
    return [
        QaEntry("A", "Why did you choose magnetic flux leakage inspection?", "Because MFL is cheap."),
        QaEntry("B", "Why choose magnetic flux leakage inspection?", "Because MFL is cheap."),
        QaEntry("C", "Why did you choose MFL inspection?", "MFL suits pipelines."),
        QaEntry("D", "What is the accuracy of your model on real data?", "About 90 percent."),
        QaEntry("E", "How accurate is the model on real measured data?", "It reaches 90 percent."),
        QaEntry("F", "What are the limitations of this study?", "Simulated data only."),
    ]


class QaPairTests(unittest.TestCase):
    def test_只比较问题_相近的配对_只查新导入的(self):
        pairs = find_qa_pairs(_bank())
        keys = {(p.left, p.right) for p in pairs}
        self.assertIn(("A", "B"), keys)
        self.assertIn(("D", "E"), keys)
        self.assertFalse(any("F" in k for k in keys))
        only = find_qa_pairs(_bank(), only_ids={"E"})
        self.assertEqual({(p.left, p.right) for p in only}, {("D", "E")})

    def test_合并保留回答_问法变成其他问法_连锁自动换人(self):
        entries = _bank()
        pairs = find_qa_pairs(entries, threshold=0.2)
        session = QaDedupeSession(entries, pairs)
        first = session.current()
        self.assertEqual((first.left, first.right), ("A", "B"))
        session.decide("merge_left")
        merged = session.by_id()["A"]
        self.assertIn("Why choose magnetic flux leakage inspection?", merged.aliases)
        self.assertEqual(merged.answer, "Because MFL is cheap.")
        for pair in session.pending():
            self.assertNotIn("B", (pair.left, pair.right), "B 已经并进 A，后面的对自动换成 A")
        self.assertEqual(session.audio_to_delete(), [], "B 的回答和 A 一样，声音还在用，不能删")

    def test_删掉的条目回答没人用了才删声音(self):
        entries = _bank()
        session = QaDedupeSession(entries, find_qa_pairs(entries))
        while session.current() is not None and {session.current().left, session.current().right} != {"D", "E"}:
            session.decide("skip")
        session.decide("keep_left")
        self.assertEqual(session.audio_to_delete(), ["It reaches 90 percent."])


class QaDedupeDialogTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import QaPanel

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        QaBankStore(Path(self.tmp.name)).save(_bank())
        self.engine = MagicMock()
        self.engine.answer_ready.return_value = True
        self.engine._session_active = False
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)

    def test_逐对确认_完成才写盘_删掉没人用的声音_记住都保留(self):
        from teams_voice_translator.defense.ui import QaDedupeDialog

        dialog = QaDedupeDialog(None, self.panel)
        self.assertGreaterEqual(dialog.pair_count, 2)
        while dialog.session.current() is not None:
            pair = dialog.session.current()
            if {pair.left, pair.right} == {"D", "E"}:
                dialog.decide("keep_left")
            elif {pair.left, pair.right} == {"A", "B"}:
                dialog.decide("merge_left")
            else:
                dialog.decide("both")
        self.assertEqual(len(QaBankStore(Path(self.tmp.name)).load()), 6, "还没点完成，什么都没写")
        dialog.finish()
        ids = [e.id for e in QaBankStore(Path(self.tmp.name)).load()]
        self.assertNotIn("B", ids)
        self.assertNotIn("E", ids)
        self.engine.delete_answer_audio.assert_called_once_with("It reaches 90 percent.")
        self.assertEqual(len(self.panel.bank), len(ids))

    def test_都保留的记住_下次不再提示(self):
        from teams_voice_translator.defense.ui import QaDedupeDialog

        dialog = QaDedupeDialog(None, self.panel)
        count = dialog.pair_count
        dialog.decide("both")
        dialog.finish()
        self.assertEqual(len(self.settings.get("qa_dedupe_ignored")), 1)
        self.assertEqual(QaDedupeDialog(None, self.panel).pair_count, count - 1)
        self.engine.delete_answer_audio.assert_not_called()

    def test_取消什么都不变_后台渲染时不让完成(self):
        from teams_voice_translator.defense import ui

        dialog = ui.QaDedupeDialog(None, self.panel)
        dialog.decide("keep_left")
        dialog.reject()
        self.assertEqual(len(QaBankStore(Path(self.tmp.name)).load()), 6)
        dialog = ui.QaDedupeDialog(None, self.panel)
        dialog.decide("keep_left")
        self.panel._bank_dialog = MagicMock(_rendering=True)
        with patch.object(ui.QMessageBox, "information") as info:
            dialog.finish()
        info.assert_called_once()
        self.assertEqual(len(QaBankStore(Path(self.tmp.name)).load()), 6)

    def test_大模型判断拿不准的(self):
        from teams_voice_translator.defense.ui import QaDedupeDialog

        dialog = QaDedupeDialog(None, self.panel)
        uncertain = [p for p in dialog.pairs if not p.exact and p.score < 0.62]
        if not uncertain:
            self.skipTest("这组样例里没有拿不准的对")
        dialog.verify_chat_factory = lambda: (lambda m: '[{"n": 0, "same": true}]')
        dialog.verify_uncertain()
        self.assertTrue(_wait(lambda: uncertain[0].verdict == "same"))

    def test_导入后提示_答辩进行中不弹(self):
        asked = []
        self.panel._ask_dedupe = lambda parent, count, strong, on_yes: asked.append((count, strong))
        self.assertGreater(self.panel.offer_dedupe({"E"}), 0)
        self.assertEqual(len(asked), 1)
        self.engine._session_active = True
        self.assertEqual(self.panel.offer_dedupe({"E"}), 0)
        self.assertEqual(len(asked), 1)
        self.assertEqual(self.panel.offer_dedupe(set()), 0)


if __name__ == "__main__":
    unittest.main()
