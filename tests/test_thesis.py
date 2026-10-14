"""导入论文全文、论文预习、中英校准、助答引用原文。全部离线：大模型调用都是假的。

测试里的「论文」都是现编的几句话，不含任何真实论文内容。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.memory import MeetingMemory
from teams_voice_translator.defense.qa import build_qa_messages
from teams_voice_translator.defense.qa_bank import QaBankStore, QaEntry
from teams_voice_translator.defense.thesis_context import (
    ThesisChunk,
    ThesisIndex,
    ThesisPage,
    ThesisStore,
    build_thesis_payload,
    chunk_pages,
    excerpt_pairs,
    extract_docx_text,
    extract_thesis_pages,
    import_thesis,
    looks_glued,
    merge_terms,
    parse_thesis_response,
    strip_references,
)
from teams_voice_translator.defense.thesis_study import (
    StudyNotes,
    StudySection,
    StudyStore,
    build_calibrate_messages,
    build_qa_payload,
    build_study_payload,
    calibrate,
    clean_calibrated,
    next_id_start,
    parse_qa_response,
    parse_study_response,
)

FAKE_BODY = (
    "Chapter 1 Introduction. Pipelines need inspection. "
    "Chapter 2 Data. We simulate magnetic flux leakage signals with a dipole model and add ten percent noise. "
    "Chapter 3 Method. A Swin Transformer regresses depth, length and width with a range constraint. "
    "Chapter 4 Lightweight. Knowledge distillation transfers the teacher to a MobileNetV3 student. "
    "Pruning reduced storage but not the measured inference time on the GPU. "
)


def make_docx(path: Path, paragraphs: list[str]) -> None:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def study_json(n: int = 3) -> str:
    return json.dumps({"sections": [
        {"title_zh": f"第{i}节", "title_en": f"Section {i}", "zh": f"中文内容{i}", "en": f"English body {i}"}
        for i in range(1, n + 1)
    ]}, ensure_ascii=False)


def qa_json(n: int, tag: str = "") -> str:
    return json.dumps({"items": [
        {"q_en": f"Question {tag}{i}?", "q_zh": f"问题{i}？", "a_en": f"Answer {i}.", "a_zh": f"回答{i}。",
         "points": ["要点"], "pri": 3, "basis": "thesis", "src": "3.2"}
        for i in range(1, n + 1)
    ]}, ensure_ascii=False)


# ================================================================ 读文件
class ExtractTests(unittest.TestCase):
    def test_docx_without_extra_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thesis.docx"
            make_docx(path, ["Title", "First paragraph.", "第二段。"])
            self.assertEqual(extract_docx_text(path), "Title\n\nFirst paragraph.\n\n第二段。")
            pages = extract_thesis_pages(path)
            self.assertEqual(pages[0].number, 0)

    def test_broken_docx_and_unknown_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.docx"
            bad.write_bytes(b"not a zip")
            with self.assertRaises(ApiError):
                extract_docx_text(bad)
            with self.assertRaises(ApiError):
                extract_thesis_pages(Path(tmp) / "x.pptx")

    def test_pdf_pages_keep_page_numbers(self):
        try:
            import pypdf  # noqa: F401
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("需要 pypdf 和 reportlab")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pdf"
            pdf = canvas.Canvas(str(path))
            pdf.drawString(72, 720, "Page one talks about wavelet denoising.")
            pdf.showPage()
            pdf.drawString(72, 720, "Page two talks about knowledge distillation.")
            pdf.save()
            pages = extract_thesis_pages(path)
            self.assertEqual([p.number for p in pages], [1, 2])
            self.assertIn("distillation", pages[1].text)

    def test_references_cut_only_near_the_end(self):
        pages = [ThesisPage(1, "Contents References ...... 77 " + "intro " * 50),
                 ThesisPage(2, "body " * 300),
                 ThesisPage(3, "Conclusion text. References [1] Someone. A paper."),
                 ThesisPage(4, "[2] Another reference.")]
        kept = strip_references(pages)
        self.assertEqual([p.number for p in kept], [1, 2, 3])
        self.assertEqual(kept[-1].text, "Conclusion text.")
        self.assertIn("Contents References", kept[0].text, "目录里的那一行不能当成参考文献标题")

    def test_glued_text_detection(self):
        self.assertTrue(looks_glued("Thisisalongwordwithoutanyspacesatall" * 100))
        self.assertFalse(looks_glued("This is normal text with spaces. " * 100))
        self.assertFalse(looks_glued("短"))


# ================================================================ 切块与检索
class ChunkIndexTests(unittest.TestCase):
    def test_chunks_respect_size_and_page(self):
        pages = [ThesisPage(5, "Sentence number one is here. " * 80)]
        chunks = chunk_pages(pages, size=300, overlap=50)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.page == 5 for c in chunks))
        self.assertTrue(all(len(c.text) <= 450 for c in chunks))

    def test_search_finds_relevant_chunk(self):
        chunks = chunk_pages([ThesisPage(i + 1, text) for i, text in enumerate(FAKE_BODY.split("Chapter")) if text])
        index = ThesisIndex(chunks)
        hits = index.search("Why did pruning not reduce the inference time?", limit=2)
        self.assertIn("Pruning", hits[0][0].text)
        label, _text = excerpt_pairs(hits)[0]
        self.assertTrue(label.startswith("第"))
        self.assertEqual(index.search("thank you"), [])


# ================================================================ 导入 + 提炼
class ImportThesisTests(unittest.TestCase):
    def test_payload_disables_thinking_and_trims(self):
        payload = build_thesis_payload("x" * 10, "qwen3.8-flash")
        self.assertIs(payload.get("enable_thinking"), False)
        self.assertEqual(payload["messages"][1]["content"], "x" * 10)

    def test_parse_dedupes_and_requires_brief(self):
        content = "```json\n" + json.dumps({"brief": "摘要", "terms": [
            {"source": "漏磁", "target": "MFL"}, {"source": "漏磁", "target": "dup"}, {"source": "", "target": "x"}]},
            ensure_ascii=False) + "\n```"
        brief, terms = parse_thesis_response(content)
        self.assertEqual((brief, terms), ("摘要", [{"source": "漏磁", "target": "MFL"}]))
        with self.assertRaises(ApiError):
            parse_thesis_response('{"terms": []}')

    def test_import_with_fake_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.docx"
            make_docx(path, FAKE_BODY.split(". "))
            seen = {}

            def post(payload):
                seen.update(payload)
                return json.dumps({"brief": "这是一篇假论文。", "terms": [{"source": "知识蒸馏", "target": "knowledge distillation"}]},
                                  ensure_ascii=False)

            context = import_thesis(None, path, model="qwen3.8-flash", post=post)
            self.assertEqual(context.brief, "这是一篇假论文。")
            self.assertEqual(context.terms[0]["target"], "knowledge distillation")
            self.assertTrue(context.chunks)
            self.assertIn("Swin Transformer", context.full_text)
            self.assertIn("Swin Transformer", seen["messages"][1]["content"])

    def test_multi_file_import_skips_bad_file_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.txt"
            bad = Path(tmp) / "missing.txt"
            good.write_text("A fake thesis about a sensor and its calibration.", encoding="utf-8")
            failures = []
            response = json.dumps({"brief": "假论文摘要", "terms": []}, ensure_ascii=False)
            context = import_thesis(
                None, [bad, good], model="qwen3.8-flash",
                post=lambda _payload: response,
                on_file_error=lambda path, exc: failures.append((path, exc)),
            )
            self.assertEqual(context.source_path, str(good))
            self.assertEqual(context.full_text.strip(), good.read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][0], bad)
            self.assertIsInstance(failures[0][1], Exception)

            with self.assertRaises(ApiError):
                import_thesis(None, [bad, Path(tmp) / "also-missing.txt"],
                              model="qwen3.8-flash", post=lambda _payload: response)

    def test_merge_terms_keeps_user_versions_first(self):
        merged, added = merge_terms([{"source": "漏磁", "target": "MFL (mine)"}],
                                    [{"source": "漏磁", "target": "other"}, {"source": "剪枝", "target": "pruning"}])
        self.assertEqual(merged, [{"source": "漏磁", "target": "MFL (mine)"}, {"source": "剪枝", "target": "pruning"}])
        self.assertEqual(added, 1)

    def test_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ThesisStore(Path(tmp))
            self.assertIsNone(store.load_index())
            self.assertEqual(store.full_text(), "")
            store.save(source_path="t.pdf", chunks=[ThesisChunk(0, 3, "wavelet denoising text")],
                       page_count=9, full_text="full", brief="b")
            self.assertEqual(store.full_text(), "full")
            self.assertEqual(store.load_index().search("wavelet")[0][0].page, 3)
            store.path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(store.load_index())


# ================================================================ 预习与校准（纯逻辑）
class StudyLogicTests(unittest.TestCase):
    def test_study_payload_thinking_switch(self):
        on = build_study_payload("text", "qwen3.8-max", thinking=True, glossary=[{"source": "漏磁", "target": "MFL"}])
        off = build_study_payload("text", "qwen3.8-max", thinking=False)
        self.assertIs(on.get("enable_thinking"), True)
        self.assertIn("漏磁 → MFL", on["messages"][1]["content"])
        self.assertIs(off.get("enable_thinking"), False)

    def test_parse_study_and_render_english(self):
        notes = parse_study_response(study_json(2), model="m")
        self.assertEqual(len(notes.sections), 2)
        self.assertIn("## Section 1\nEnglish body 1", notes.render("en"))
        self.assertIn("中文内容2", notes.render("zh"))
        with self.assertRaises(ApiError):
            parse_study_response('{"sections": []}')

    def test_qa_payload_lists_existing_questions(self):
        notes = StudyNotes([StudySection("t", "T", "中", "Eng")])
        payload = build_qa_payload("text", notes, ["Old question?"], "qwen3.8-max", count=99, thinking=False)
        user = payload["messages"][1]["content"]
        self.assertIn("- Old question?", user)
        self.assertIn("Eng", user)
        self.assertIn("60 个问题", payload["messages"][0]["content"], "数量有上限")

    def test_parse_qa_ids_and_basis(self):
        raw = json.loads(qa_json(2))
        raw["items"][1]["basis"] = "whatever"
        raw["items"].append({"q_en": "", "a_en": "x"})
        entries = parse_qa_response(json.dumps(raw), id_prefix="S", start=5)
        self.assertEqual([e.id for e in entries], ["S5", "S6"])
        self.assertEqual(entries[1].basis, "reason")
        self.assertTrue(entries[0].note, "自动生成的要带核对提醒")
        self.assertEqual(next_id_start(["S1", "S9", "L2", "A1"], "S"), 10)
        self.assertEqual(next_id_start([], "L"), 1)

    def test_calibrate_messages_and_cleanup(self):
        messages = build_calibrate_messages("zh2en", "新的中文", "Old English", context="Why?",
                                            glossary=[{"source": "剪枝", "target": "pruning"}])
        self.assertIn("改写英文", messages[0]["content"])
        self.assertIn("新的中文", messages[1]["content"])
        self.assertIn("剪枝 → pruning", messages[1]["content"])
        self.assertIn("改写中文", build_calibrate_messages("en2zh", "E", "中")[0]["content"])
        with self.assertRaises(ValueError):
            build_calibrate_messages("xx", "a", "b")
        self.assertEqual(clean_calibrated('英文：“It works.”'), "It works.")
        self.assertEqual(clean_calibrated("```\nHello\n```"), "Hello")

    def test_calibrate_uses_fast_chat_without_thinking(self):
        translator = MagicMock()
        translator.chat.return_value = "English: Fixed answer."
        self.assertEqual(calibrate(translator, "zh2en", "中文", "old"), "Fixed answer.")
        self.assertFalse(translator.chat.call_args.kwargs["thinking"])

    def test_study_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyStore(Path(tmp))
            self.assertIsNone(store.load())
            notes = StudyNotes([StudySection("中", "En", "z", "e")], "m", "t0", "t1")
            store.save(notes)
            self.assertEqual(store.load(), notes)


# ================================================================ 助答提示词
class QaPromptTests(unittest.TestCase):
    def test_notes_in_system_excerpts_in_user(self):
        messages = build_qa_messages(MeetingMemory(), "Why pruning?", notes="## Results\nPruning was slow.",
                                     excerpts=[("第 88 页", "Inference time did not follow parameter count.")])
        self.assertIn("Pruning was slow.", messages[0]["content"])
        self.assertIn("（第 88 页）Inference time", messages[1]["content"])
        self.assertTrue(messages[1]["content"].rstrip().endswith("Why pruning?"))
        plain = build_qa_messages(MeetingMemory(), "Q?")
        self.assertNotIn("论文理解（事实", plain[0]["content"].replace("论文理解、", ""))


# ================================================================ 界面
class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.settings = DefenseSettings(base_dir=self.base)

    def wait(self, condition, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            threading.Event().wait(0.01)
        self.app.processEvents()
        return condition()

    def make_panel(self):
        from teams_voice_translator.defense.ui import QaPanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QaPanel(self.settings, engine)
        self.addCleanup(panel.shutdown)
        return panel

    def seed_thesis(self):
        ThesisStore(self.base).save(
            source_path="t.pdf", page_count=4, full_text=FAKE_BODY, brief="b",
            chunks=chunk_pages([ThesisPage(i + 1, t) for i, t in enumerate(FAKE_BODY.split("Chapter")) if t]),
        )
        StudyStore(self.base).save(StudyNotes([StudySection("结果", "Results", "剪枝没变快", "Pruning was not faster.")], "m", "t"))

    def test_generate_passes_notes_and_excerpts_and_shows_pages(self):
        self.seed_thesis()
        panel = self.make_panel()
        captured = {}

        class FakeAdvisor:
            model = "fake"

            def __init__(self, **_kw):
                pass

            def generate(self, memory, question, **kwargs):
                captured.update(kwargs)
                return "Because of memory access.", "因为访存。"

        panel.advisor_factory = FakeAdvisor
        panel.generate("Why did pruning not reduce inference time?")
        self.assertTrue(self.wait(lambda: panel.answer_en.text() == "Because of memory access."))
        self.assertIn("Pruning was not faster.", captured["notes"])
        self.assertTrue(captured["excerpts"])
        self.assertIn("原文 第", panel.source_label.text())
        self.assertIn("论文理解", panel.source_label.text())

    def test_save_live_answer_then_update_it(self):
        panel = self.make_panel()
        panel._question = "How long did training take?"
        panel._on_qa_answer("About two hours.", "大约两小时。")
        saved = panel.save_to_bank()
        self.assertEqual(saved.id, "L1")
        self.assertEqual(QaBankStore(self.base).load()[0].answer_zh, "大约两小时。")
        panel.answer_en.setText("About three hours.")
        again = panel.save_to_bank()
        self.assertEqual(again.id, "L1", "来自问答库的条目再保存是更新，不是新增")
        bank = QaBankStore(self.base).load()
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0].answer, "About three hours.")

    def test_save_without_question_refuses(self):
        panel = self.make_panel()
        panel.answer_en.setText("Orphan answer.")
        self.assertIsNone(panel.save_to_bank())

    def test_chinese_edit_recalibrates_english(self):
        panel = self.make_panel()
        panel.show_bank_entry(QaEntry("Z1", "Q?", "Old English.", answer_zh="旧中文。"))
        panel.calibrator_factory = lambda: (lambda direction, edited, other, context: f"New English for {edited}")
        panel.answer_zh.setText("新中文。")
        panel._on_zh_edited(user=True)
        self.assertTrue(panel.zh_to_en_button.isEnabled())
        panel._calibrate_english()
        self.assertTrue(self.wait(lambda: panel.answer_en.text() == "New English for 新中文。"))
        self.assertEqual(panel.answer_zh.text(), "新中文。", "你改的中文不能被自动重译覆盖")
        self.assertTrue(panel.restore_button.isEnabled())

    def test_programmatic_chinese_update_is_not_a_user_edit(self):
        panel = self.make_panel()
        panel._on_qa_answer("Hi.", "你好。")
        self.assertFalse(panel.zh_to_en_button.isEnabled())

    def make_study(self, translator):
        from teams_voice_translator.defense.ui import StudyDialog

        panel = self.make_panel()
        dialog = StudyDialog(None, self.settings, panel)
        dialog.translator_factory = lambda model: translator
        self.addCleanup(dialog.shutdown)
        return dialog, panel

    def test_study_flow_edit_calibrate_save(self):
        self.seed_thesis()
        StudyStore(self.base).path.unlink()
        translator = MagicMock()
        translator.complete.return_value = study_json(3)
        dialog, panel = self.make_study(translator)
        self.assertEqual(dialog.editors, [])
        dialog.start_study()
        self.assertTrue(self.wait(lambda: len(dialog.editors) == 3))
        self.assertIsNotNone(panel.study_notes, "预习完助答立即能用")
        payload = translator.complete.call_args[0][0]
        self.assertEqual(payload["model"], "qwen3.8-max")
        # 改中文 → 校准英文 → 保存
        dialog.calibrator = lambda direction, edited, other, context: f"EN({edited})"
        dialog.editors[1].zh.setText("改过的中文")
        dialog.calibrate_section(1, "zh2en")
        self.assertTrue(self.wait(lambda: dialog.editors[1].en.text() == "EN(改过的中文)"))
        dialog.save_notes()
        stored = StudyStore(self.base).load()
        self.assertEqual(stored.sections[1].en, "EN(改过的中文)")
        self.assertTrue(stored.edited_at)
        self.assertIn("EN(改过的中文)", panel.study_notes.render("en"))

    def test_study_without_thesis_explains(self):
        translator = MagicMock()
        dialog, _panel = self.make_study(translator)
        dialog.start_study()
        self.assertIn("还没有导入论文", dialog.status_label.text())
        translator.complete.assert_not_called()

    def test_generate_qa_in_batches_into_bank(self):
        self.seed_thesis()
        QaBankStore(self.base).save([QaEntry("A1", "Existing?", "Yes.")])
        translator = MagicMock()
        translator.complete.side_effect = [qa_json(10, "a"), qa_json(10, "b")]
        dialog, panel = self.make_study(translator)
        dialog.qa_count_spin.setValue(12)
        dialog.start_qa()
        self.assertTrue(self.wait(lambda: "已向问答库加入" in dialog.status_label.text() or "出错" in dialog.status_label.text()))
        bank = QaBankStore(self.base).load()
        self.assertEqual([e.id for e in bank][:3], ["A1", "S1", "S2"])
        self.assertEqual(len(bank), 1 + 12, "第二批只要 2 条")
        self.assertEqual(translator.complete.call_count, 2)
        second_user = translator.complete.call_args_list[1][0][0]["messages"][1]["content"]
        self.assertIn("Question a1?", second_user, "第二批要知道第一批已经生成过哪些问题")
        self.assertEqual(len(panel.bank), 13)

    def test_window_thesis_import_updates_everything(self):
        from teams_voice_translator.defense.ui import DefenseWindow

        window = DefenseWindow(self.settings)
        self.addCleanup(window.close)
        path = self.base / "t.docx"
        make_docx(path, FAKE_BODY.split(". "))
        translator = MagicMock()
        translator.complete.return_value = json.dumps(
            {"brief": "假论文摘要。", "terms": [{"source": "知识蒸馏", "target": "knowledge distillation"}]},
            ensure_ascii=False)
        done = []
        window.signals.import_finished.disconnect()
        window.signals.import_finished.connect(lambda message, thesis: done.append((message, thesis)))
        window._thesis_import_worker(str(path), "qwen3.8-flash", translator)
        self.assertTrue(self.wait(lambda: bool(done)))
        self.assertTrue(done[0][1])
        self.assertIn("新增 1 条", done[0][0])
        self.assertEqual(self.settings.get("defense_brief"), "假论文摘要。")
        self.assertIn({"source": "知识蒸馏", "target": "knowledge distillation"}, self.settings.glossary_terms())
        self.assertIsNotNone(ThesisStore(self.base).load_index())


if __name__ == "__main__":
    unittest.main()
