import base64
import io
import json
import threading
import unittest
import wave
from unittest.mock import Mock, patch

from teams_voice_translator.aliyun import BailianClient


class FakeWebSocket:
    def __init__(self, events):
        self.events = [json.dumps(event) for event in events]
        self.sent = []
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def recv(self):
        return self.events.pop(0)

    def send(self, message):
        self.sent.append(json.loads(message))

    def close(self):
        self.closed = True


class Qwen3TtsTests(unittest.TestCase):
    def setUp(self):
        self.client = BailianClient("sk-test", "llm-test")
        self.settings = {
            "tts_model": "qwen3-tts-vc-realtime-2026-01-15",
            "voice": "voice-test",
            "tts_sample_rate": 24000,
            "tts_volume": 55,
            "tts_rate": 1.0,
            "tts_pitch": 1.0,
            "tts_language_hint": "en",
        }

    def test_qwen3_realtime_streams_pcm_delta(self):
        pcm = b"\x01\x02\x03\x04"
        socket = FakeWebSocket(
            [
                {"type": "session.created"},
                {"type": "session.updated"},
                {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()},
                {"type": "response.done", "response": {"status": "completed"}},
                {"type": "session.finished"},
            ]
        )
        chunks = []
        with patch("teams_voice_translator.aliyun.websocket.create_connection", return_value=socket):
            count = self.client.stream_tts(
                "Hello",
                self.settings,
                chunks.append,
                threading.Event(),
            )
        self.assertEqual(count, len(pcm))
        self.assertEqual(chunks, [pcm])
        self.assertTrue(socket.closed)
        self.assertEqual(socket.sent[0]["type"], "session.update")
        self.assertEqual(socket.sent[1]["type"], "input_text_buffer.append")
        self.assertEqual(socket.sent[2]["type"], "input_text_buffer.commit")
        self.assertEqual(socket.sent[-1]["type"], "session.finish")

    def test_qwen3_http_streams_pcm_sse(self):
        pcm = b"\x10\x20"
        response = Mock()
        response.ok = True
        response.iter_lines.return_value = [
            "data: " + json.dumps(
                {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}
            ),
            "data: [DONE]",
        ]
        settings = dict(self.settings)
        settings["tts_model"] = "qwen3-tts-vc-2026-01-22"
        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post:
            count = self.client.stream_tts(
                "Hello",
                settings,
                chunks.append,
                threading.Event(),
            )
        self.assertEqual(count, len(pcm))
        self.assertEqual(chunks, [pcm])
        self.assertEqual(
            post.call_args.kwargs["headers"]["X-DashScope-WorkSpace"],
            "llm-test",
        )

    def test_qwen3_http_streams_wav_sse(self):
        pcm = b"\x10\x20\x30\x40"
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        wav_bytes = wav_buffer.getvalue()
        response = Mock()
        response.ok = True
        response.iter_lines.return_value = [
            "data: " + json.dumps(
                {"output": {"audio": {"data": base64.b64encode(wav_bytes).decode()}}}
            ),
            "data: [DONE]",
        ]
        settings = dict(self.settings)
        settings["tts_model"] = "qwen3-tts-vc-2026-01-22"
        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response):
            count = self.client.stream_tts(
                "Hello",
                settings,
                chunks.append,
                threading.Event(),
            )
        self.assertEqual(count, len(pcm))
        self.assertEqual(chunks, [pcm])

    def test_qwen3_http_falls_back_to_url(self):
        pcm = b"\x01\x02\x03\x04\x05\x06"
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        wav_bytes = wav_buffer.getvalue()
        url_response = Mock()
        url_response.ok = True
        url_response.content = wav_bytes
        sse_response = Mock()
        sse_response.ok = True
        sse_response.iter_lines.return_value = [
            "data: " + json.dumps(
                {"output": {"audio": {"url": "https://dashscope.example.com/audio.wav"}}}
            ),
            "data: [DONE]",
        ]
        settings = dict(self.settings)
        settings["tts_model"] = "qwen3-tts-vc-2026-01-22"
        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=sse_response):
            with patch("teams_voice_translator.aliyun.requests.get", return_value=url_response) as get:
                count = self.client.stream_tts(
                    "Hello",
                    settings,
                    chunks.append,
                    threading.Event(),
                )
        self.assertEqual(count, len(pcm))
        self.assertEqual(chunks, [pcm])
        get.assert_called_once()
        self.assertIn("dashscope.example.com", get.call_args.args[0])
        url_response.close.assert_called_once()

    def test_qwen3_http_does_not_play_data_and_url_twice(self):
        pcm = b"\x10\x20\x30\x40"
        response = Mock()
        response.ok = True
        response.iter_lines.return_value = [
            "data: " + json.dumps(
                {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}
            ),
            "data: " + json.dumps(
                {"output": {"audio": {"url": "https://dashscope.example.com/same.wav"}}}
            ),
            "data: [DONE]",
        ]
        settings = dict(self.settings)
        settings["tts_model"] = "qwen3-tts-vc-2026-01-22"
        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response):
            with patch("teams_voice_translator.aliyun.requests.get") as get:
                count = self.client.stream_tts(
                    "Hello", settings, chunks.append, threading.Event()
                )
        self.assertEqual(count, len(pcm))
        self.assertEqual(chunks, [pcm])
        get.assert_not_called()

    def test_qwen3_clone_returns_voice_and_fallback_warning(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "output": {
                "voice": "voice-test",
                "target_model": "qwen3-tts-vc-realtime-2026-01-15",
                "fallback_mode": True,
                "fallback_reason": "no_valid_asr_segments",
            }
        }
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post:
            voice, warning = self.client.clone_qwen_voice(
                target_model="qwen3-tts-vc-realtime-2026-01-15",
                preferred_name="my_voice",
                audio_data="data:audio/wav;base64,UklGRg==",
                language="zh",
            )
        self.assertEqual(voice, "voice-test")
        self.assertIn("fallback", warning)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen-voice-enrollment")


if __name__ == "__main__":
    unittest.main()
