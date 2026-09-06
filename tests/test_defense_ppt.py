from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.ppt_context import (
    MAX_CONTEXT_CHARS,
    build_context_payload,
    extract_document_text,
    parse_context_response,
)


class ParseContextResponseTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        content = json.dumps(
            {
                "brief": "论文研究缺陷反演。",
                "terms": [{"source": "有限元", "target": "finite element method"}],
            },
            ensure_ascii=False,
        )
        brief, terms = parse_context_response(content)
        self.assertEqual(brief, "论文研究缺陷反演。")
        self.assertEqual(terms[0]["target"], "finite element method")

    def test_fenced_json_with_prose(self) -> None:
        content = "好的，以下是结果：\n```json\n" + json.dumps(
            {"brief": "主题", "terms": []}, ensure_ascii=False
        ) + "\n```\n请查收。"
        brief, terms = parse_context_response(content)
        self.assertEqual(brief, "主题")
        self.assertEqual(terms, [])

    def test_filters_incomplete_terms(self) -> None:
        content = json.dumps(
            {"brief": "b", "terms": [{"source": "a", "target": ""}, {"source": "x", "target": "y"}]},
            ensure_ascii=False,
        )
        _brief, terms = parse_context_response(content)
        self.assertEqual(len(terms), 1)

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ApiError):
            parse_context_response("这不是 JSON")

    def test_missing_brief_raises(self) -> None:
        with self.assertRaises(ApiError):
            parse_context_response(json.dumps({"terms": []}))


class BuildContextPayloadTest(unittest.TestCase):
    def test_truncates_long_document(self) -> None:
        payload = build_context_payload("字" * (MAX_CONTEXT_CHARS + 500))
        user = payload["messages"][1]["content"]
        self.assertEqual(len(user), MAX_CONTEXT_CHARS)

    def test_instruction_mentions_term_cap(self) -> None:
        payload = build_context_payload("内容")
        self.assertIn("40", payload["messages"][0]["content"])


class ExtractDocumentTextTest(unittest.TestCase):
    def test_plain_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.md"
            path.write_text("# 论文\n\n有限元方法", encoding="utf-8")
            text, count = extract_document_text(path)
            self.assertIn("有限元方法", text)
            self.assertEqual(count, 0)

    def test_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deck.ppt"
            path.write_text("old", encoding="utf-8")
            with self.assertRaises(ApiError):
                extract_document_text(path)

    def test_pptx_extraction(self) -> None:
        pptx = __import__("pptx")
        presentation = pptx.Presentation()
        layout = presentation.slide_layouts[1]
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = "缺陷反演研究"
        body = slide.placeholders[1]
        body.text = "有限元\n正向模型"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "defense.pptx"
            presentation.save(str(path))
            text, count = extract_document_text(path)
        self.assertEqual(count, 1)
        self.assertIn("缺陷反演研究", text)
        self.assertIn("有限元", text)


if __name__ == "__main__":
    unittest.main()
