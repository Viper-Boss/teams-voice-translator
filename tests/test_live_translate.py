import base64
import json
import time
import unittest

from teams_voice_translator.live_translate import QwenLiveTranslate


class LiveTranslateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
