from __future__ import annotations

import base64
import json
import threading
import time
import unittest
from unittest import mock

import websocket

from teams_voice_translator.aliyun import BailianClient
from teams_voice_translator.defense.tts_session import TtsSession


def fake_audio(seed: int) -> str:
    return base64.b64encode(bytes([seed, 1, 2, 3])).decode("ascii")


class FakeWs:
    """Scripted realtime WebSocket: session lifecycle + per-commit events."""

    instances: list["FakeWs"] = []

    def __init__(self, plan) -> None:
        self.plan = plan
        self.sent: list[dict] = []
        self.inbox: list[dict] = [{"type": "session.created"}]
        self.lock = threading.Lock()
        self.commits = 0
        self.closed = False
        FakeWs.instances.append(self)

    def settimeout(self, _value) -> None:
        return None

    def send(self, data: str) -> None:
        event = json.loads(data)
        with self.lock:
            self.sent.append(event)
            kind = event.get("type")
            if kind == "session.update":
                self.inbox.append({"type": "session.updated"})
            elif kind == "input_text_buffer.commit":
                self.commits += 1
                self.inbox.extend(self.plan(self.commits))

    def recv(self):
        for _ in range(400):
            with self.lock:
                if self.inbox:
                    return json.dumps(self.inbox.pop(0))
            time.sleep(0.005)
        raise websocket.WebSocketTimeoutException()

    def close(self) -> None:
        self.closed = True


def good_plan(commit_index: int) -> list[dict]:
    return [
        {"type": "response.audio.delta", "delta": fake_audio(commit_index * 2)},
        {"type": "response.audio.delta", "delta": fake_audio(commit_index * 2 + 1)},
        {"type": "response.done", "response": {"status": "completed"}},
    ]


def error_plan(_commit_index: int) -> list[dict]:
    return [{"type": "error", "error": {"code": "bad_request", "message": "boom"}}]


def tts_settings() -> dict:
    return {
        "tts_model": "qwen3-tts-vc-realtime-2026-01-15",
        "voice": "qwen-clone-voice-id",
        "tts_language_hint": "en",
        "tts_sample_rate": 24000,
        "tts_volume": 55,
        "tts_rate": 1.0,
        "tts_pitch": 1.0,
    }


def wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class TtsSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeWs.instances = []
        self.client = BailianClient("sk-test", "ws-demo")
        self.audio: list[bytes] = []
        self.first_audio_texts: list[str] = []
        self.statuses: list[str] = []
        self.errors: list[str] = []

    def _make_session(self) -> TtsSession:
        return TtsSession(
            self.client,
            tts_settings(),
            on_audio=self.audio.append,
            on_status=self.statuses.append,
            on_error=self.errors.append,
            on_first_audio=self.first_audio_texts.append,
            connect_timeout=10,
            utterance_timeout=20,
        )

    def test_rejects_missing_voice(self) -> None:
        settings = tts_settings()
        settings["voice"] = ""
        with self.assertRaises(Exception):
            TtsSession(self.client, settings, on_audio=self.audio.append)

    def test_two_sentences_share_one_connection(self) -> None:
        with mock.patch(
            "teams_voice_translator.defense.tts_session.websocket.create_connection",
            side_effect=lambda *_args, **_kwargs: FakeWs(good_plan),
        ):
            session = self._make_session()
            session.start()
            self.assertTrue(session.speak("Hello there."))
            self.assertTrue(session.speak("Second sentence."))
            self.assertTrue(wait_for(lambda: len(self.first_audio_texts) >= 2))
            self.assertTrue(session.wait_until_idle(timeout=10))
            session.close()
        self.assertEqual(len(FakeWs.instances), 1, "同一会话的两句话不应重建连接")
        ws = FakeWs.instances[0]
        updates = [event for event in ws.sent if event.get("type") == "session.update"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["session"]["voice"], "qwen-clone-voice-id")
        appends = [event for event in ws.sent if event.get("type") == "input_text_buffer.append"]
        self.assertEqual(len(appends), 2)
        self.assertEqual(len(self.audio), 4)
        self.assertEqual(self.errors, [])
        self.assertNotIn("session.finish", {event.get("type") for event in ws.sent if event})

    def test_reconnect_and_retry_on_error(self) -> None:
        plans = [error_plan, good_plan, good_plan]

        def factory(*_args, **_kwargs) -> FakeWs:
            plan = plans.pop(0) if len(plans) > 1 else plans[0]
            return FakeWs(plan)

        with mock.patch(
            "teams_voice_translator.defense.tts_session.websocket.create_connection",
            side_effect=factory,
        ):
            session = self._make_session()
            session.start()
            session.speak("Will fail once.")
            self.assertTrue(wait_for(lambda: len(self.audio) >= 2))
            self.assertTrue(session.wait_until_idle(timeout=10))
            session.close()
        self.assertEqual(len(FakeWs.instances), 2, "失败后应在新连接上重试")
        self.assertEqual(self.errors, [], "重试成功后不应报错")
        self.assertTrue(any("重试" in message or "重连" in message for message in self.statuses))

    def test_interrupt_clears_queue_and_recovers(self) -> None:
        with mock.patch(
            "teams_voice_translator.defense.tts_session.websocket.create_connection",
            side_effect=lambda *_args, **_kwargs: FakeWs(good_plan),
        ):
            session = self._make_session()
            session.start()
            session.speak("Interrupt me.")
            session.interrupt()
            self.assertTrue(wait_for(lambda: not session._drop_audio.is_set(), timeout=5))
            self.assertTrue(session.speak("Speak after interrupt."))
            self.assertTrue(wait_for(lambda: len(self.audio) >= 2))
            self.assertTrue(session.wait_until_idle(timeout=10))
            session.close()
        self.assertEqual(self.errors, [])

    def test_failed_after_retry_reports_error(self) -> None:
        with mock.patch(
            "teams_voice_translator.defense.tts_session.websocket.create_connection",
            side_effect=lambda *_args, **_kwargs: FakeWs(error_plan),
        ):
            session = self._make_session()
            session.start()
            session.speak("Always failing.")
            self.assertTrue(wait_for(lambda: bool(self.errors), timeout=15))
            session.close()
        self.assertEqual(len(FakeWs.instances), 2, "重试一次后放弃")
        self.assertTrue(any("跳过" in message for message in self.errors))


if __name__ == "__main__":
    unittest.main()
