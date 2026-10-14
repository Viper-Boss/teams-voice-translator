"""Actual client response parsing, using service-shaped HTTP fixtures only."""
import base64
import io
import json
import threading
import unittest
import wave
from unittest.mock import Mock, patch
from requests.structures import CaseInsensitiveDict

from teams_voice_translator.aliyun import BailianClient, ApiError
import test_funasr_cosyvoice as fixtures
from teams_voice_translator.defense.tts_session import synthesize_once


class ResultResponse:
    ok = True
    status_code = 200
    def __init__(self, events, *, json_mode=False, as_bytes=False):
        self.events = events
        self.json_mode = json_mode
        self.as_bytes = as_bytes
        self.headers = CaseInsensitiveDict({"content-type": "application/json" if json_mode else "text/event-stream"})
        self.closed = False
    def json(self):
        return self.events[0]
    def iter_lines(self, decode_unicode=True):
        for item in self.events:
            line = "data: " + json.dumps(item)
            yield line.encode() if self.as_bytes else line
    def close(self):
        self.closed = True


class AudioResponse:
    ok = True
    status_code = 200
    def __init__(self, blob, *, content_type="application/octet-stream", cancel=None):
        self.blob = blob
        self.headers = {"Content-Type": content_type}
        self.closed = False
        self.cancel = cancel
    def iter_content(self, chunk_size):
        yield self.blob[:len(self.blob)//2]
        if self.cancel:
            self.cancel.set()
        yield self.blob[len(self.blob)//2:]
    def close(self):
        self.closed = True


class TtsResultTests(unittest.TestCase):
    def setUp(self):
        self.client = BailianClient("fake-key", "fake-workspace")
        self.settings = fixtures.CosyVoiceTtsTests().settings(tts_model="qwen-audio-3.1-tts-flash", tts_instruction="Speak calmly.")
        self.cancel = threading.Event()
        self.pcm = b"\x01\x00\x02\x00" * 64
    def event(self, **audio):
        return {"request_id": "test-request", "output": {"finish_reason": "stop", "audio": audio}}
    def stream(self, response, download=None):
        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post, \
                patch("teams_voice_translator.aliyun.requests.get", return_value=download) as get:
            total = self.client._stream_legacy_tts("Hello.", self.settings, chunks.append, self.cancel)
        return total, chunks, post, get

    def test_url_only_sse_retrieves_same_request_pcm_once(self):
        response = ResultResponse([self.event(data="", url="https://oss.example/result.pcm")])
        download = AudioResponse(self.pcm)
        total, chunks, post, get = self.stream(response, download)
        self.assertEqual(total, len(self.pcm))
        self.assertEqual(chunks, [self.pcm])
        post.assert_called_once()
        get.assert_called_once()
        self.assertNotIn("headers", get.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["json"]["input"]["instruction"], "Speak calmly.")
        self.assertTrue(response.closed and download.closed)

    def test_non_streaming_json_url_is_supported(self):
        response = ResultResponse([self.event(data="", url="https://oss.example/result.pcm")], json_mode=True)
        self.assertEqual(self.stream(response, AudioResponse(self.pcm))[1], [self.pcm])

    def test_sse_bytes_are_decoded_even_without_response_encoding(self):
        response = ResultResponse([self.event(data=base64.b64encode(self.pcm).decode())], as_bytes=True)
        self.assertEqual(self.stream(response)[1], [self.pcm])

    def test_data_then_final_url_is_not_played_twice(self):
        response = ResultResponse([
            {"output": {"audio": {"data": base64.b64encode(self.pcm).decode()}}},
            self.event(data="", url="https://oss.example/result.pcm")])
        _, chunks, _, get = self.stream(response)
        self.assertEqual(chunks, [self.pcm])
        get.assert_not_called()

    def test_downloaded_wav_is_converted_to_pcm(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(self.pcm)
        response = ResultResponse([self.event(url="https://oss.example/result.wav")])
        self.assertEqual(self.stream(response, AudioResponse(buf.getvalue(), content_type="audio/wav"))[1], [self.pcm])

    def test_empty_or_odd_download_is_error(self):
        for blob in (b"", b"x"):
            with self.subTest(blob=blob), self.assertRaises(ApiError):
                self.stream(ResultResponse([self.event(url="https://oss.example/result.pcm")]), AudioResponse(blob))

    def test_html_response_is_not_played_as_pcm_noise(self):
        download = AudioResponse(b"<html>error</html>", content_type="text/html")
        with self.assertRaisesRegex(ApiError, "错误页面"):
            self.stream(ResultResponse([self.event(url="https://oss.example/result.pcm")]), download)
        self.assertTrue(download.closed)

    def test_cancel_during_download_emits_no_partial_audio(self):
        response = ResultResponse([self.event(url="https://oss.example/result.pcm")])
        total, chunks, _, _ = self.stream(response, AudioResponse(self.pcm, cancel=self.cancel))
        self.assertEqual((total, chunks), (0, []))

    def test_provider_error_is_not_masked_as_empty_audio(self):
        with self.assertRaisesRegex(ApiError, "InvalidParameter.*voice"):
            self.stream(ResultResponse([{"code": "InvalidParameter", "message": "voice not found"}]))

    def test_empty_response_exposes_request_id(self):
        with self.assertRaisesRegex(ApiError, "test-request"):
            self.stream(ResultResponse([self.event(data="")]))

    def test_actual_synthesize_once_accepts_url_only_result_without_retry_post(self):
        response = ResultResponse([self.event(url="https://oss.example/result.pcm")])
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post, \
                patch("teams_voice_translator.aliyun.requests.get", return_value=AudioResponse(self.pcm)):
            pcm = synthesize_once(self.client, self.settings, "Hello.", timeout=3)
        self.assertEqual(pcm, self.pcm)
        post.assert_called_once()
