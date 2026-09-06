import base64
import json
import threading
import unittest
from unittest.mock import patch

import websocket as websocket_module

from teams_voice_translator.aliyun import (
    BailianClient,
    FunASRRealtime,
    QwenRealtimeASR,
    create_realtime_asr,
)
from teams_voice_translator.api_payloads import (
    build_fun_asr_finish_task,
    build_fun_asr_run_task,
)


class FakeWebSocketApp:
    instances = []

    def __init__(self, url, header=None, on_open=None, on_message=None, on_error=None, on_close=None):
        self.url = url
        self.header = header or []
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent = []
        self.closed = False
        FakeWebSocketApp.instances.append(self)

    def send(self, data, opcode=None):
        self.sent.append((data, opcode))

    def run_forever(self, **_kwargs):
        pass

    def close(self):
        self.closed = True


class AutoReadyWebSocketApp(FakeWebSocketApp):
    def run_forever(self, **_kwargs):
        self.on_open(self)
        self.on_message(self, json.dumps({"header": {"event": "task-started"}, "payload": {}}))


def make_fun_asr(**overrides):
    previews = []
    segments = []
    statuses = []
    errors = []
    kwargs = {
        "api_key": "sk-test",
        "workspace_id": "llm-test",
        "model": "fun-asr-realtime",
        "language": "zh",
        "vad_threshold": 0.0,
        "vad_silence_ms": 500,
        "on_preview": lambda text, emotion: previews.append((text, emotion)),
        "on_status": statuses.append,
        "on_error": errors.append,
        "on_segment": lambda text, emotion: segments.append(text),
    }
    kwargs.update(overrides)
    asr = FunASRRealtime(**kwargs)
    return asr, previews, segments, statuses, errors


class FunAsrPayloadTests(unittest.TestCase):
    def test_run_task_structure(self):
        event = build_fun_asr_run_task()
        self.assertEqual(event["header"]["action"], "run-task")
        self.assertEqual(event["header"]["streaming"], "duplex")
        self.assertTrue(event["header"]["task_id"])
        payload = event["payload"]
        self.assertEqual(payload["task_group"], "audio")
        self.assertEqual(payload["task"], "asr")
        self.assertEqual(payload["function"], "recognition")
        self.assertEqual(payload["model"], "fun-asr-realtime")
        self.assertEqual(
            payload["parameters"],
            {"format": "pcm", "sample_rate": 16000, "heartbeat": True},
        )
        self.assertEqual(payload["input"], {})

    def test_run_task_with_hotwords(self):
        event = build_fun_asr_run_task(
            context_terms=["有限元", "模态分析"],
            vocabulary_id="vocab-test",
            language="zh",
        )
        payload = event["payload"]
        self.assertEqual(payload["parameters"]["vocabulary_id"], "vocab-test")
        self.assertEqual(payload["parameters"]["language_hints"], ["zh"])
        self.assertIn(
            "有限元",
            payload["input"]["context"][0]["content"][0]["text"],
        )
        self.assertNotIn("vocabulary", payload["parameters"])

    def test_finish_task_structure(self):
        event = build_fun_asr_finish_task("task-1")
        self.assertEqual(event["header"]["action"], "finish-task")
        self.assertEqual(event["header"]["task_id"], "task-1")


class FunAsrRealtimeTests(unittest.TestCase):
    def setUp(self):
        FakeWebSocketApp.instances = []

    def test_full_recognition_flow(self):
        asr, previews, segments, _statuses, errors = make_fun_asr(
            context_terms=["有限元"]
        )
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp", FakeWebSocketApp):
            asr.start()
        app = FakeWebSocketApp.instances[0]
        self.assertIn("wss://llm-test.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference", app.url)
        self.assertIn("Authorization: Bearer sk-test", app.header)

        app.on_open(app)
        run_task = json.loads(app.sent[0][0])
        self.assertEqual(run_task["header"]["action"], "run-task")
        self.assertIn(
            "有限元",
            run_task["payload"]["input"]["context"][0]["content"][0]["text"],
        )
        self.assertTrue(run_task["payload"]["parameters"]["heartbeat"])
        self.assertEqual(asr.task_id, run_task["header"]["task_id"])

        app.on_message(app, json.dumps({"header": {"event": "task-started"}, "payload": {}}))
        asr.wait_ready(timeout=1.0)

        asr.send_audio(b"\x00\x01")
        self.assertEqual(app.sent[-1], (b"\x00\x01", websocket_module.ABNF.OPCODE_BINARY))

        app.on_message(
            app,
            json.dumps(
                {
                    "header": {"event": "result-generated"},
                    "payload": {"output": {"sentence": {"text": "今天我们讲", "sentence_end": False}}},
                }
            ),
        )
        self.assertEqual(previews[-1][0], "今天我们讲")

        app.on_message(
            app,
            json.dumps(
                {
                    "header": {"event": "result-generated"},
                    "payload": {
                        "output": {
                            "sentence": {"text": "今天我们讲有限元", "sentence_end": True, "words": []}
                        },
                        "usage": {"duration": 2},
                    },
                }
            ),
        )
        self.assertEqual(segments, ["今天我们讲有限元"])

        app.on_message(
            app,
            json.dumps(
                {
                    "header": {"event": "result-generated"},
                    "payload": {"output": {"sentence": {"text": "", "heartbeat": True, "sentence_id": 0}}},
                }
            ),
        )
        self.assertEqual(len(segments), 1)

        app.on_message(app, json.dumps({"header": {"event": "task-finished"}, "payload": {}}))
        result = asr.finish(timeout=1.0)
        self.assertEqual(result, "今天我们讲有限元")
        finish = json.loads(app.sent[-1][0])
        self.assertEqual(finish["header"]["action"], "finish-task")
        self.assertEqual(finish["header"]["task_id"], asr.task_id)
        self.assertFalse(errors)

    def test_task_failed_sets_error(self):
        asr, _previews, _segments, _statuses, errors = make_fun_asr()
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp", FakeWebSocketApp):
            asr.start()
        app = FakeWebSocketApp.instances[0]
        app.on_open(app)
        app.on_message(
            app,
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "error_code": "CLIENT_ERROR",
                        "error_message": "request timeout",
                    },
                    "payload": {},
                }
            ),
        )
        self.assertTrue(errors)
        with self.assertRaises(Exception):
            asr.wait_ready(timeout=1.0)

    def test_combine_previews_disabled(self):
        asr, previews, segments, _statuses, _errors = make_fun_asr(combine_previews=False)
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp", FakeWebSocketApp):
            asr.start()
        app = FakeWebSocketApp.instances[0]
        app.on_open(app)
        app.on_message(app, json.dumps({"header": {"event": "task-started"}, "payload": {}}))
        for text in ("第一句。", "第二句。"):
            app.on_message(
                app,
                json.dumps(
                    {
                        "header": {"event": "result-generated"},
                        "payload": {"output": {"sentence": {"text": text, "sentence_end": True}}},
                    }
                ),
            )
        self.assertEqual(previews[-1][0], "第二句。")
        self.assertEqual(segments, ["第一句。", "第二句。"])

    def test_unexpected_disconnect_reconnects_on_next_audio(self):
        asr, _previews, _segments, statuses, errors = make_fun_asr()
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp", AutoReadyWebSocketApp):
            asr.start()
            asr.wait_ready(timeout=1.0)
            first = AutoReadyWebSocketApp.instances[0]
            first.on_close(first, 1006, "network lost")
            asr.send_audio(b"\x00\x01")
        self.assertEqual(len(AutoReadyWebSocketApp.instances), 2)
        second = AutoReadyWebSocketApp.instances[1]
        self.assertEqual(second.sent[-1], (b"\x00\x01", websocket_module.ABNF.OPCODE_BINARY))
        self.assertTrue(any("自动重连" in status for status in statuses))
        self.assertFalse(errors)


class CreateRealtimeAsrTests(unittest.TestCase):
    def base_values(self, model):
        return {
            "asr_model": model,
            "vad_threshold": 0.0,
            "vad_silence_ms": 500,
            "translation_terms": '[{"source":"有限元","target":"finite element method"}]',
        }

    def test_factory_returns_fun_asr_with_hotwords(self):
        asr = create_realtime_asr(
            api_key="sk-test",
            workspace_id="llm-test",
            values=self.base_values("fun-asr-realtime"),
            language="zh",
            on_preview=lambda *_: None,
            on_status=lambda *_: None,
            on_error=lambda *_: None,
        )
        self.assertIsInstance(asr, FunASRRealtime)
        self.assertEqual(asr.context_terms, ["有限元"])

    def test_factory_fun_asr_english_uses_targets(self):
        asr = create_realtime_asr(
            api_key="sk-test",
            workspace_id="llm-test",
            values=self.base_values("fun-asr-realtime"),
            language="en",
            on_preview=lambda *_: None,
            on_status=lambda *_: None,
            on_error=lambda *_: None,
        )
        self.assertIsInstance(asr, FunASRRealtime)
        self.assertEqual(asr.context_terms, ["finite element method"])

    def test_factory_returns_qwen_for_qwen_models(self):
        asr = create_realtime_asr(
            api_key="sk-test",
            workspace_id="llm-test",
            values=self.base_values("qwen3-asr-flash-realtime"),
            language="zh",
            on_preview=lambda *_: None,
            on_status=lambda *_: None,
            on_error=lambda *_: None,
        )
        self.assertIsInstance(asr, QwenRealtimeASR)

    def test_factory_routes_duplex_models_to_funasr_class(self):
        from teams_voice_translator.api_payloads import is_duplex_asr_model

        for model in (
            "fun-asr-realtime-2026-02-28",
            "paraformer-realtime-v2",
            "qwen-audio-3.0-asr-flash-streaming",
        ):
            self.assertTrue(is_duplex_asr_model(model), model)
            asr = create_realtime_asr(
                api_key="sk-test",
                workspace_id="llm-test",
                values=self.base_values(model),
                language="zh",
                on_preview=lambda *_: None,
                on_status=lambda *_: None,
                on_error=lambda *_: None,
            )
            self.assertIsInstance(asr, FunASRRealtime, model)
        self.assertFalse(is_duplex_asr_model("qwen3-asr-flash-realtime-2025-10-27"))

    def test_run_task_heartbeat_only_for_fun_asr(self):
        from teams_voice_translator.api_payloads import build_fun_asr_run_task

        fun_payload = build_fun_asr_run_task(model="fun-asr-realtime")
        self.assertTrue(fun_payload["payload"]["parameters"]["heartbeat"])
        streaming_payload = build_fun_asr_run_task(model="qwen-audio-3.0-asr-flash-streaming")
        self.assertNotIn("heartbeat", streaming_payload["payload"]["parameters"])


class FakeSseResponse:
    def __init__(self, audio: bytes):
        self.ok = True
        encoded = base64.b64encode(audio).decode()
        self._lines = [
            f'data: {{"output": {{"audio": {{"data": "{encoded}"}}}}}}',
            "data: [DONE]",
        ]
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        self.closed = True


class CosyVoiceTtsTests(unittest.TestCase):
    def settings(self, **overrides):
        values = {
            "tts_model": "cosyvoice-v3-flash",
            "voice": "longanyang",
            "tts_sample_rate": 24000,
            "tts_volume": 55,
            "tts_rate": 1.0,
            "tts_pitch": 1.0,
            "tts_seed": 0,
            "tts_language_hint": "en",
            "tts_instruction": "",
            "tts_emotion_tag": "[sad]",
        }
        values.update(overrides)
        return values

    def test_cosyvoice_streams_via_legacy_http_without_emotion_tag(self):
        client = BailianClient("sk-test", "llm-test")
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            return FakeSseResponse(b"\x01\x02")

        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", fake_post):
            total = client.stream_tts("Hello", self.settings(), chunks.append, threading.Event())
        self.assertEqual(total, 2)
        self.assertEqual(chunks, [b"\x01\x02"])
        self.assertEqual(captured["json"]["model"], "cosyvoice-v3-flash")
        self.assertEqual(captured["json"]["input"]["text"], "Hello")
        self.assertIn("SpeechSynthesizer", captured["url"])

    def test_qwen_audio_keeps_emotion_tag(self):
        client = BailianClient("sk-test", "llm-test")
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs["json"]
            return FakeSseResponse(b"\x01")

        with patch("teams_voice_translator.aliyun.requests.post", fake_post):
            client.stream_tts(
                "Hello",
                self.settings(tts_model="qwen-audio-3.0-tts-flash"),
                lambda _chunk: None,
                threading.Event(),
            )
        self.assertEqual(captured["json"]["input"]["text"], "[sad] Hello")

    def test_cosyvoice_v35_strips_aigc_fields(self):
        client = BailianClient("sk-test", "llm-test")
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs["json"]
            return FakeSseResponse(b"\x01\x02")

        chunks = []
        with patch("teams_voice_translator.aliyun.requests.post", fake_post):
            client.stream_tts(
                "Hello",
                self.settings(
                    tts_model="cosyvoice-v3.5-plus",
                    voice="cosyvoice-v3.5-plus-myvoice-xxx",
                ),
                chunks.append,
                threading.Event(),
            )
        self.assertEqual(chunks, [b"\x01\x02"])
        payload_input = captured["json"]["input"]
        self.assertEqual(payload_input["text"], "Hello")
        for key in ("enable_aigc_tag", "aigc_propagator", "aigc_propagate_id"):
            self.assertNotIn(key, payload_input)


if __name__ == "__main__":
    unittest.main()
