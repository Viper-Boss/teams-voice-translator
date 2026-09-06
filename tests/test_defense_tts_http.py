from __future__ import annotations

import threading
import time
import unittest

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.tts_session import (
    TtsHttpSession,
    TtsLegacySession,
    TtsSession,
    create_tts_session,
)


def http_settings() -> dict:
    return {
        "tts_model": "qwen3-tts-vc-2026-01-22",
        "voice": "qwen-clone-voice-id",
        "tts_language_hint": "en",
        "tts_sample_rate": 24000,
        "tts_volume": 55,
        "tts_rate": 1.0,
        "tts_pitch": 1.0,
    }


class FakeHttpClient:
    """Scripted stand-in for BailianClient._stream_qwen3_tts_http."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls: list[str] = []

    def _stream_qwen3_tts_http(self, text, settings, on_audio, cancel_event) -> int:
        self.calls.append(text)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ApiError("模拟合成失败")
        for chunk in (b"\x01\x02", b"\x03\x04", b"\x05\x06"):
            if cancel_event.is_set():
                break
            on_audio(chunk)
        return 6


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class TtsHttpSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audio: list[bytes] = []
        self.first: list[str] = []
        self.done: list[tuple[str, int]] = []
        self.errors: list[str] = []

    def make_session(self, client) -> TtsHttpSession:
        return TtsHttpSession(
            client,
            http_settings(),
            on_audio=self.audio.append,
            on_error=self.errors.append,
            on_first_audio=self.first.append,
            on_utterance_done=lambda text, total: self.done.append((text, total)),
        )

    def test_streams_audio_and_reports_duration_bytes(self) -> None:
        client = FakeHttpClient()
        session = self.make_session(client)
        session.start()
        session.speak("Hello there.")
        self.assertTrue(wait_for(lambda: len(self.done) == 1))
        session.close()
        self.assertEqual(client.calls, ["Hello there."])
        self.assertEqual(self.first, ["Hello there."])
        self.assertEqual(len(self.audio), 3)
        self.assertEqual(self.done, [("Hello there.", 6)])
        self.assertEqual(self.errors, [])

    def test_retries_once_then_reports_error(self) -> None:
        client = FakeHttpClient(fail_times=2)
        session = self.make_session(client)
        session.start()
        session.speak("Will fail.")
        self.assertTrue(wait_for(lambda: bool(self.errors), timeout=8))
        session.close()
        self.assertEqual(client.calls, ["Will fail.", "Will fail."], "失败后应重试一次")
        self.assertEqual(self.audio, [])
        self.assertIn("跳过", self.errors[0])

    def test_interrupt_drops_audio_and_recovers(self) -> None:
        client = FakeHttpClient()
        session = self.make_session(client)
        session.start()
        session.speak("Interrupt me.")
        session.interrupt()
        self.assertTrue(session.speak("Speak after interrupt."))
        self.assertTrue(wait_for(lambda: any(text == "Speak after interrupt." for text, _ in self.done)))
        session.close()
        self.assertEqual(client.calls[-1], "Speak after interrupt.")
        self.assertEqual(self.errors, [])

    def test_batch_synthesizes_concurrently_but_plays_in_order(self) -> None:
        class ParallelClient(FakeHttpClient):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def _stream_qwen3_tts_http(self, text, settings, on_audio, cancel_event):
                with self.lock:
                    self.calls.append(text)
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(.08 if text == 'First.' else .04)
                on_audio({'First.': b'AA', 'Second.': b'BB', 'Third.': b'CC'}[text])
                with self.lock:
                    self.active -= 1
                return 2

        client = ParallelClient()
        session = self.make_session(client)
        session.start()
        self.assertTrue(session.speak_batch(['First.', 'Second.', 'Third.']))
        self.assertTrue(wait_for(lambda: len(self.done) == 3))
        session.close()
        self.assertGreaterEqual(client.max_active, 2)
        self.assertEqual(self.first, ['First.', 'Second.', 'Third.'])
        self.assertEqual(self.audio, [b'AA', b'BB', b'CC'])
        self.assertEqual([text for text, _size in self.done], ['First.', 'Second.', 'Third.'])


class TtsFactoryTest(unittest.TestCase):
    def test_routes_by_model_protocol(self) -> None:
        client = FakeHttpClient()
        realtime = create_tts_session(
            client,
            {"tts_model": "qwen3-tts-vc-realtime-2026-01-15", "voice": "v1"},
            on_audio=lambda chunk: None,
        )
        self.assertIsInstance(realtime, TtsSession)
        realtime.close()

        hd = create_tts_session(
            client,
            {"tts_model": "qwen3-tts-vc-2026-01-22", "voice": "v1"},
            on_audio=lambda chunk: None,
        )
        self.assertIsInstance(hd, TtsHttpSession)
        hd.close()

    def test_requires_voice(self) -> None:
        with self.assertRaises(Exception):
            create_tts_session(
                FakeHttpClient(),
                {"tts_model": "qwen3-tts-vc-2026-01-22", "voice": ""},
                on_audio=lambda chunk: None,
            )

    def test_cosyvoice_routes_to_legacy_session(self) -> None:
        from teams_voice_translator.aliyun import ApiError as _ApiError

        class LegacyClient(FakeHttpClient):
            def _stream_legacy_tts(self, text, settings, on_audio, cancel_event) -> int:
                on_audio(b"ab")
                return 2

        received: list[bytes] = []
        session = create_tts_session(
            LegacyClient(),
            {"tts_model": "cosyvoice-v3.5-plus", "voice": "cosy-voice-id", "tts_seed": 0},
            on_audio=received.append,
        )
        self.assertIsInstance(session, TtsLegacySession)
        errors: list[str] = []
        session = create_tts_session(
            LegacyClient(),
            {"tts_model": "cosyvoice-v3.5-plus", "voice": "cosy-voice-id", "tts_seed": 0},
            on_audio=received.append,
            on_error=errors.append,
        )
        session.start()
        session.speak("CosyVoice test.")
        self.assertTrue(wait_for(lambda: len(received) == 1))
        session.close()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
