import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teams_voice_translator.aliyun import BailianClient  # noqa: E402
from teams_voice_translator.longform import (  # noqa: E402
    BilingualSegment,
    LongFormReaderDialog,
    build_long_form_translation_payload,
    apply_pronunciation_dictionary,
    estimate_speech_seconds,
    highlighted_sentence_html,
    parse_long_form_translation,
    parse_pronunciation_dictionary,
    split_source_sentences,
)


class TestLongFormHelpers(unittest.TestCase):
    def test_sentence_split_retains_chinese_punctuation(self) -> None:
        chunks = split_source_sentences("第一句。第二句！\n第三句没有句号")
        self.assertEqual(chunks, ["第一句。", "第二句！", "第三句没有句号"])

    def test_context_payload_contains_full_document_terms_and_alignment(self) -> None:
        payload = build_long_form_translation_payload(
            "有限元方法很重要。它可以求解此问题。",
            ["有限元方法很重要。", "它可以求解此问题。"],
            {
                "summary_model": "qwen-flash",
                "long_form_model": "qwen-plus",
                "translation_domain": "engineering",
                "translation_style": "academic",
            },
            terms=[{"source": "有限元", "target": "finite element method"}],
            memories=[],
        )
        self.assertEqual(payload["model"], "qwen-plus")
        request = json.loads(payload["messages"][1]["content"])
        self.assertEqual(request["document_context"], "有限元方法很重要。它可以求解此问题。")
        self.assertEqual(request["required_terms"][0]["target"], "finite element method")
        self.assertEqual([item["index"] for item in request["sentences"]], [1, 2])

    def test_aligned_response_parser_reorders_by_index_and_rejects_missing(self) -> None:
        content = '```json\n{"segments":[{"index":2,"english":"B"},{"index":1,"english":"A"}]}\n```'
        self.assertEqual(parse_long_form_translation(content, 2), ["A", "B"])
        with self.assertRaisesRegex(ValueError, "缺少序号"):
            parse_long_form_translation('{"segments":[{"index":1,"english":"A"}]}', 2)

    def test_estimate_and_highlight(self) -> None:
        self.assertGreater(estimate_speech_seconds("This is a short sentence."), 1.0)
        rendered = highlighted_sentence_html("Hello world.", 0.6)
        self.assertIn("#fff1a8", rendered)
        self.assertIn("#8b5cf6", rendered)

    def test_pronunciation_dictionary_is_tts_only_and_longest_first(self) -> None:
        raw = '{"Open":"open","OpenAI":"Open A I"}'
        self.assertEqual(parse_pronunciation_dictionary(raw)["OpenAI"], "Open A I")
        self.assertEqual(apply_pronunciation_dictionary("Use OpenAI.", raw), "Use Open A I.")

    @patch("teams_voice_translator.aliyun.requests.post")
    def test_client_long_form_uses_single_context_request(self, post: Mock) -> None:
        response = Mock(ok=True)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"segments":[{"index":1,"english":"Hello."}]}'}}]
        }
        post.return_value = response
        client = BailianClient("key", "workspace")
        result = client.translate_long_form(
            "你好。",
            ["你好。"],
            {
                "summary_model": "qwen-plus",
                "translation_terms": "[]",
                "translation_memories": "[]",
                "translation_domain": "classroom",
                "translation_style": "polite",
            },
        )
        self.assertEqual(result, ["Hello."])
        self.assertEqual(post.call_count, 1)

    @patch("teams_voice_translator.aliyun.requests.post")
    def test_client_long_form_batches_large_scripts_after_context_brief(self, post: Mock) -> None:
        context = Mock(ok=True)
        context.json.return_value = {
            "choices": [{"message": {"content": '{"summary":"lesson"}'}}]
        }
        first = Mock(ok=True)
        first.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {"index": index, "english": f"E{index}"}
                                    for index in range(1, 19)
                                ]
                            }
                        )
                    }
                }
            ]
        }
        second = Mock(ok=True)
        second.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {"index": index, "english": f"L{index}"}
                                    for index in range(1, 8)
                                ]
                            }
                        )
                    }
                }
            ]
        }
        post.side_effect = [context, first, second]
        client = BailianClient("key", "workspace")
        sentences = [f"第{index}句。" for index in range(25)]
        result = client.translate_long_form(
            "".join(sentences),
            sentences,
            {
                "long_form_model": "qwen-plus",
                "translation_terms": "[]",
                "translation_memories": "[]",
            },
        )
        self.assertEqual(len(result), 25)
        self.assertEqual(post.call_count, 3)


class TestLongFormReaderDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.client = Mock()
        self.dialog = LongFormReaderDialog(
            [
                BilingualSegment("第一句话。", "The first sentence.", 3.0),
                BilingualSegment("第二句话。", "The second sentence.", 4.0),
            ],
            client=self.client,
            settings={"tts_sample_rate": 24000, "teams_output_device": None},
        )

    def tearDown(self) -> None:
        self.dialog.stop()
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_dialog_has_bilingual_cards_and_timeline_controls(self) -> None:
        self.assertEqual(len(self.dialog.cards), 2)
        self.assertEqual(self.dialog.cards[0].chinese.text(), "第一句话。")
        self.assertEqual(self.dialog.cards[0].english.text(), "The first sentence.")
        self.assertEqual(self.dialog.pause_button.text(), "⏸ 暂停")
        self.assertEqual(self.dialog.slider.maximum(), 7000)

    def test_same_fraction_drives_both_language_highlights(self) -> None:
        self.dialog.apply_state_update("progress:0:0.5")
        self.assertIn("#8b5cf6", self.dialog.cards[0].chinese.text())
        self.assertIn("#8b5cf6", self.dialog.cards[0].english.text())
        self.assertTrue(self.dialog.cards[0].property("active"))

    def test_click_target_interrupts_current_sentence(self) -> None:
        current_cancel = threading.Event()
        self.dialog.active_cancel_event = current_cancel
        self.dialog.worker_thread = Mock()
        self.dialog.worker_thread.is_alive.return_value = True
        self.dialog.play_from(1)
        self.assertTrue(current_cancel.is_set())
        self.assertEqual(self.dialog.requested_index, 1)

    def test_sentence_pcm_is_cached_for_replay(self) -> None:
        def stream(_text, _settings, on_audio, _cancel):
            on_audio(b"1234")
            return 4

        self.client.stream_tts.side_effect = stream
        cancel = threading.Event()
        first = self.dialog._get_or_synthesize_audio(0, cancel)
        second = self.dialog._get_or_synthesize_audio(0, cancel)
        self.assertEqual(first, b"1234")
        self.assertEqual(second, first)
        self.client.stream_tts.assert_called_once()

    def test_finished_clock_uses_actual_durations(self) -> None:
        self.dialog.segments[0].actual_seconds = 2.0
        self.dialog.segments[1].actual_seconds = 3.0
        self.dialog.apply_state_update("finished")
        self.assertEqual(self.dialog.slider.maximum(), 5000)
        self.assertEqual(self.dialog.elapsed.text(), "00:05")

    @patch("teams_voice_translator.longform.QFileDialog.getSaveFileName")
    def test_reader_exports_bilingual_srt(self, choose: Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lesson.srt"
            choose.return_value = (str(output), "双语字幕 (*.srt)")
            self.dialog.export_script()
            content = output.read_text(encoding="utf-8-sig")
            self.assertIn("第一句话。", content)
            self.assertIn("The first sentence.", content)
            self.assertIn("00:00:00,000 --> 00:00:03,000", content)


if __name__ == "__main__":
    unittest.main()
