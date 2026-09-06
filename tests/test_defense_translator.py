from __future__ import annotations

import json
import unittest
from unittest import mock

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.memory import MeetingMemory
from teams_voice_translator.defense.translator import ContextTranslator, SentenceSplitter


def sse_lines(chunks: list[str]) -> list[str]:
    lines = []
    for piece in chunks:
        event = {"choices": [{"delta": {"content": piece}}]}
        lines.append("data: " + json.dumps(event))
    lines.append("data: [DONE]")
    return lines


class FakeStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200, payload: str = "") -> None:
        self._lines = lines
        self.ok = status_code < 400
        self.status_code = status_code
        self.text = payload

    def iter_lines(self, decode_unicode: bool = True) -> list[str]:
        return list(self._lines)

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None


class SentenceSplitterTest(unittest.TestCase):
    def test_hard_endings_split(self) -> None:
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("这是第一句。这是第二句！"), ["这是第一句。", "这是第二句！"])
        self.assertEqual(splitter.flush(), [])

    def test_english_abbreviation_not_split(self) -> None:
        splitter = SentenceSplitter()
        pieces = splitter.feed("This uses e.g. three methods. Next one starts here.")
        self.assertEqual(pieces, ["This uses e.g. three methods."])
        self.assertEqual(splitter.flush(), ["Next one starts here."])

    def test_english_period_needs_uppercase_or_digit(self) -> None:
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("The value is 3."), [])
        self.assertEqual(splitter.feed("4 meters. The result holds."), ["The value is 3.4 meters."])
        self.assertEqual(splitter.flush(), ["The result holds."])

    def test_semicolon_and_question(self) -> None:
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("First part; second part? "), ["First part;", "second part?"])

    def test_flush_returns_rest(self) -> None:
        splitter = SentenceSplitter()
        splitter.feed("unfinished sentence")
        self.assertEqual(splitter.flush(), ["unfinished sentence"])

    def test_streaming_deltas(self) -> None:
        splitter = SentenceSplitter()
        collected: list[str] = []
        collected.extend(splitter.feed("Hello "))
        collected.extend(splitter.feed("world. "))
        collected.extend(splitter.feed("Again."))
        self.assertEqual(collected, ["Hello world."])
        self.assertEqual(splitter.flush(), ["Again."])


class ContextTranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.translator = ContextTranslator(api_key="sk-test", workspace_id="ws-demo", model="qwen-plus")

    def test_requires_credentials(self) -> None:
        with self.assertRaises(ApiError):
            ContextTranslator(api_key="", workspace_id="ws")
        with self.assertRaises(ApiError):
            ContextTranslator(api_key="sk", workspace_id="")

    def test_payload_shape(self) -> None:
        memory = MeetingMemory(brief="答辩主题")
        memory.set_glossary([{"source": "有限元", "target": "FEM"}])
        payload = self.translator.build_payload(memory, "zh2en", "测试句子")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "qwen-plus")
        system = payload["messages"][0]["content"]
        self.assertIn("答辩主题", system)
        self.assertIn("FEM", system)

    def test_unknown_direction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.translator.build_payload(MeetingMemory(), "zh2fr", "text")

    def test_translate_streams_and_returns_text(self) -> None:
        response = FakeStreamResponse(sse_lines(["Good ", "morning."]))
        with mock.patch("teams_voice_translator.defense.translator.requests.post", return_value=response) as poster:
            deltas: list[str] = []
            result = self.translator.translate(
                MeetingMemory(), "zh2en", "早上好。", on_delta=lambda piece, so_far: deltas.append(so_far)
            )
        self.assertEqual(result, "Good morning.")
        self.assertEqual(deltas, ["Good ", "Good morning."])
        request = poster.call_args
        self.assertIn("compatible-mode/v1/chat/completions", request.args[0])
        self.assertTrue(request.kwargs["json"]["stream"])

    def test_translate_http_error(self) -> None:
        response = FakeStreamResponse([], status_code=403, payload="denied")
        with mock.patch("teams_voice_translator.defense.translator.requests.post", return_value=response):
            with self.assertRaises(ApiError):
                self.translator.translate(MeetingMemory(), "zh2en", "测试")

    def test_translate_empty_result(self) -> None:
        response = FakeStreamResponse(["data: [DONE]"])
        with mock.patch("teams_voice_translator.defense.translator.requests.post", return_value=response):
            with self.assertRaises(ApiError):
                self.translator.translate(MeetingMemory(), "zh2en", "测试")


if __name__ == "__main__":
    unittest.main()
