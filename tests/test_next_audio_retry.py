"""Offline coverage for AudioGen reference calls and empty-audio recovery."""
import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from teams_voice_translator.aliyun import ApiError, BailianClient
from teams_voice_translator.defense.tts_session import TtsLegacySession, TtsNextSession, create_tts_session
from teams_voice_translator.defense.speech_policy import speech_settings
from teams_voice_translator.defense.tts_cache import cache_key


def wav_bytes(seconds=0.1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x01\x00" * int(24000 * seconds))
    return stream.getvalue()


class NextAudioTests(unittest.TestCase):
    def test_next_is_not_sent_to_legacy_tts(self):
        session = create_tts_session(object(), {"tts_model": "qwen-audio-3.1-tts-next",
                                                 "voice": "", "tts_reference_audio": "sample.wav"},
                                     on_audio=lambda _pcm: None)
        self.assertIsInstance(session, TtsNextSession)

    def test_reference_audio_payload_and_complete_file_download(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.wav"
            sample.write_bytes(wav_bytes())
            client = BailianClient("test-key", "test-workspace")
            post = Mock(ok=True)
            post.json.return_value = {"output": {"audio": {"url": "https://example.com/result.wav"}}}
            download = Mock(ok=True)
            download.iter_content.return_value = [wav_bytes()]
            received = []
            with patch("teams_voice_translator.aliyun.requests.post", return_value=post) as call, \
                 patch("teams_voice_translator.aliyun.requests.get", return_value=download):
                count = client._stream_next_audio("Hello there.", {
                    "tts_reference_audio": str(sample), "tts_volume": 50, "tts_rate": 1.0,
                    "tts_language_hint": "en",
                }, received.append, threading.Event())
            payload = call.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "qwen-audio-3.1-tts-next")
            self.assertIn("@voice1", payload["input"]["text_prompt"])
            self.assertNotIn("voice", payload["input"])
            self.assertTrue(payload["input"]["references"][0]["audio_data"].startswith("data:audio/wav;base64,"))
            self.assertEqual(count, len(received[0]))
            self.assertEqual(received[0], b"\x01\x00" * 2400)

    def test_missing_reference_does_not_call_cloud(self):
        client = BailianClient("test-key", "test-workspace")
        with patch("teams_voice_translator.aliyun.requests.post") as post:
            with self.assertRaisesRegex(ApiError, "参考录音"):
                client._stream_next_audio("Hello", {"tts_reference_audio": ""},
                                          lambda _pcm: None, threading.Event())
        post.assert_not_called()

    def test_changing_reference_invalidates_prepared_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.wav"
            sample.write_bytes(wav_bytes(0.1))
            config = {"tts_model": "qwen-audio-3.1-tts-next", "tts_reference_audio": str(sample)}
            before = cache_key("Hello", speech_settings(config))
            sample.write_bytes(wav_bytes(0.2))
            after = cache_key("Hello", speech_settings(config))
            self.assertNotEqual(before, after)

    def test_empty_audio_is_retried_before_playback(self):
        attempts = []
        delivered = []
        statuses = []
        done = threading.Event()
        client = Mock()

        def stream(_text, _settings, on_audio, _token):
            attempts.append(1)
            if len(attempts) < 3:
                return 0
            on_audio(b"\x01\x00" * 100)
            return 200

        client._stream_legacy_tts.side_effect = stream
        session = TtsLegacySession(client, {"tts_model": "qwen-audio-3.0-tts-plus",
                                            "voice": "v", "speech_mode": "natural",
                                            "tts_sample_rate": 24000},
                                   on_audio=delivered.append, on_status=statuses.append,
                                   on_utterance_done=lambda *_: done.set())
        session.start()
        try:
            self.assertTrue(session.speak("Hello."))
            self.assertTrue(done.wait(8), "空音频后应自动恢复发声")
        finally:
            session.close()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(delivered)
        self.assertEqual(len(statuses), 2)


if __name__ == "__main__":
    unittest.main()
