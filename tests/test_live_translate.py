import base64
import json
import time
import unittest

from teams_voice_translator.live_translate import QwenLiveTranslate


class LiveTranslateTests(unittest.TestCase):
    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(json.loads(message))

    def test_stream_events_produce_bilingual_result_and_audio(self):
        sources = []
        translations = []
        audio = []
        client = QwenLiveTranslate(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3.5-livetranslate-flash-realtime",
            voice_mode="default",
            on_source_preview=sources.append,
            on_translation_preview=translations.append,
            on_audio=audio.append,
        )
        client._committed_at = time.perf_counter()
        client._on_message(
            None,
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "你好老师",
                }
            ),
        )
        client._on_message(
            None,
            json.dumps(
                {
                    "type": "response.audio_transcript.done",
                    "transcript": "Hello, Professor.",
                }
            ),
        )
        client._on_message(
            None,
            json.dumps(
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(b"pcm").decode("ascii"),
                }
            ),
        )
        client._on_message(
            None,
            json.dumps({"type": "response.done", "response": {"usage": {"x": 1}}}),
        )

        self.assertEqual(sources[-1], "你好老师")
        self.assertEqual(translations[-1], "Hello, Professor.")
        self.assertEqual(audio, [b"pcm"])
        self.assertTrue(client.response_done.is_set())
        self.assertIsNotNone(client.first_audio_seconds)

    def test_unexpected_close_unblocks_waiters(self):
        errors = []
        client = QwenLiveTranslate(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3.5-livetranslate-flash-realtime",
            on_error=errors.append,
        )

        client._on_close(None, 1006, "network lost")

        self.assertTrue(client.ready.is_set())
        self.assertTrue(client.response_done.is_set())
        self.assertTrue(client.session_finished.is_set())
        self.assertIn("意外关闭", errors[-1])

    def test_manual_session_resets_state_for_second_turn(self):
        client = QwenLiveTranslate(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3.5-livetranslate-flash-realtime",
        )
        client.ws = self.FakeWebSocket()
        client.opened.set()
        client.ready.set()
        client.source_text = "上一句"
        client.translated_text = "Previous sentence."
        client.usage = {"x": 1}
        client.response_done.set()

        client.begin_turn()

        self.assertFalse(client.response_done.is_set())
        self.assertEqual(client.source_text, "")
        self.assertEqual(client.translated_text, "")
        self.assertEqual(client.usage, {})

    def test_continuous_session_emits_one_result_per_response(self):
        results = []
        client = QwenLiveTranslate(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3.5-livetranslate-flash-realtime",
            continuous=True,
            on_result=results.append,
        )
        client._on_message(None, json.dumps({"type": "input_audio_buffer.speech_started"}))
        client._on_message(
            None,
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "你好",
                }
            ),
        )
        client._on_message(
            None,
            json.dumps(
                {"type": "response.audio_transcript.done", "transcript": "Hello."}
            ),
        )
        client._on_message(None, json.dumps({"type": "response.done", "response": {}}))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_text, "你好")
        self.assertEqual(results[0].translated_text, "Hello.")
        self.assertEqual(client.completed_turns, 1)


if __name__ == "__main__":
    unittest.main()
