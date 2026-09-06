from __future__ import annotations

import unittest
from unittest import mock

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.memory import MeetingMemory, Turn
from teams_voice_translator.defense.qa import NOT_A_QUESTION, QaAdvisor, build_qa_messages, parse_answer


class ParseAnswerTest(unittest.TestCase):
    def test_standard_en_zh(self) -> None:
        english, chinese = parse_answer("EN: We model it as an inverse problem.\nZH: 我们把它建模为反演问题。")
        self.assertEqual(english, "We model it as an inverse problem.")
        self.assertEqual(chinese, "我们把它建模为反演问题。")

    def test_same_line_en_zh(self) -> None:
        english, chinese = parse_answer("EN: Yes, exactly. ZH: 是的。")
        self.assertEqual(english, "Yes, exactly.")
        self.assertEqual(chinese, "是的。")

    def test_multiline_sections(self) -> None:
        content = "EN: First sentence.\nSecond sentence.\nZH: 第一句。\n第二句。"
        english, chinese = parse_answer(content)
        self.assertIn("First sentence.", english)
        self.assertIn("Second sentence.", english)
        self.assertIn("第二句。", chinese)

    def test_not_a_question(self) -> None:
        english, chinese = parse_answer(f"ZH: {NOT_A_QUESTION}")
        self.assertEqual(english, "")
        self.assertEqual(chinese, "")

    def test_free_form_fallback_becomes_english(self) -> None:
        english, chinese = parse_answer("Thank you for the question. Our method handles noise via regularization.")
        self.assertIn("regularization", english)
        self.assertEqual(chinese, "")

    def test_empty_content(self) -> None:
        self.assertEqual(parse_answer(""), ("", ""))


class BuildQaMessagesTest(unittest.TestCase):
    def test_contains_brief_glossary_and_question(self) -> None:
        memory = MeetingMemory(brief="本文研究缺陷反演。", glossary=[{"source": "反演", "target": "inversion"}])
        memory.add_turn(Turn(role="me", source="我的方法是两步反演。", translation="My method is two-step inversion."))
        messages = build_qa_messages(memory, "Why is your regularization necessary?")
        system = messages[0]["content"]
        self.assertIn("缺陷反演", system)
        self.assertIn("inversion", system)
        self.assertIn(NOT_A_QUESTION, system)
        user = messages[1]["content"]
        self.assertIn("两步反演", user)
        self.assertIn("Why is your regularization necessary?", user)


class QaAdvisorTest(unittest.TestCase):
    def test_generate_parses_reply(self) -> None:
        advisor = QaAdvisor(api_key="sk-test", workspace_id="ws-demo", model="qwen-plus")
        response = mock.Mock()
        response.ok = True
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.iter_lines.return_value = [
            "data: " + mock_json({"choices": [{"delta": {"content": "EN: It regularizes the inversion."}}]}),
            "data: " + mock_json({"choices": [{"delta": {"content": "\nZH: 它对反演做了正则化。"}}]}),
            "data: [DONE]",
        ]
        with mock.patch("teams_voice_translator.defense.translator.requests.post", return_value=response):
            english, chinese = advisor.generate(MeetingMemory(), "How do you stabilize it?")
        self.assertEqual(english, "It regularizes the inversion.")
        self.assertIn("正则化", chinese)

    def test_generate_requires_credentials(self) -> None:
        with self.assertRaises(ApiError):
            QaAdvisor(api_key="", workspace_id="ws")


def mock_json(payload: dict) -> str:
    import json

    return json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
