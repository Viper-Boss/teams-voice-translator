"""2026-09 模型更新：MiniMax 语音接入 + 大模型思考模式控制。

最要紧的一条不变量：**实时翻译请求绝不能让模型先「思考」。**
新一代模型（qwen3.8 / 3.7、deepseek-v4、glm-5.2）默认开思考，正文要等
思考完才吐。翻译请求不关掉，界面就会沉默几秒到几十秒。
"""
from __future__ import annotations

import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.api_payloads import (
    TTS_CAPABILITIES,
    build_minimax_clone_payload,
    build_minimax_tts_payload,
    is_minimax_tts_model,
    minimax_voice_setting,
    parse_minimax_tts_response,
    tts_supports,
    valid_minimax_voice_id,
)
from teams_voice_translator.defense.chat_models import (
    CHAT_MODEL_TABLE,
    USE_QA,
    USE_TRANSLATE,
    chat_model,
    models_for,
    thinking_fields,
    translation_problem,
)
from teams_voice_translator.defense.clone_batch import (
    MINIMAX_PREVIEW_TEXT,
    build_tasks,
    minimax_voice_name,
    minimum_sample_seconds,
)

MINIMAX = "MiniMax/speech-2.8-hd"


# ============================================================== MiniMax 语音
class MiniMaxPayloadTests(unittest.TestCase):
    def test_all_four_models_are_registered_for_cloning(self):
        for model in ("MiniMax/speech-2.8-hd", "MiniMax/speech-2.8-turbo",
                      "MiniMax/speech-02-hd", "MiniMax/speech-02-turbo"):
            self.assertIn(model, TTS_CAPABILITIES)
            self.assertTrue(tts_supports(model, "clone"), model)
            self.assertFalse(tts_supports(model, "instruction"), model)
            self.assertFalse(tts_supports(model, "ssml"), model)
            self.assertTrue(is_minimax_tts_model(model))

    def test_family_check_does_not_match_other_vendors(self):
        self.assertFalse(is_minimax_tts_model("MiniMax-M3"))  # 那是对话模型
        self.assertFalse(is_minimax_tts_model("qwen-audio-3.0-tts-plus"))

    def test_settings_live_under_input_and_ask_for_pcm(self):
        """官方示例里 voice_setting / audio_setting 都在 input 下，不在顶层。"""
        payload = build_minimax_tts_payload("Hello.", model=MINIMAX, voice="defenseEN1")
        self.assertEqual(payload["model"], MINIMAX)
        body = payload["input"]
        self.assertEqual(body["text"], "Hello.")
        self.assertEqual(body["voice_setting"]["voice_id"], "defenseEN1")
        self.assertEqual(body["audio_setting"]["format"], "pcm")
        self.assertEqual(body["audio_setting"]["channel"], 1)
        self.assertEqual(body["audio_setting"]["sample_rate"], 24000)
        self.assertNotIn("voice_setting", payload)

    def test_empty_voice_is_rejected_before_any_request(self):
        with self.assertRaises(ValueError):
            build_minimax_tts_payload("Hi.", model=MINIMAX, voice="  ")

    def test_non_minimax_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_minimax_tts_payload("Hi.", model="cosyvoice-v3.5-plus", voice="v")

    def test_parameters_are_mapped_into_minimax_ranges(self):
        neutral = minimax_voice_setting(voice="v", rate=1.0, volume=50, pitch=1.0)
        self.assertEqual((neutral["speed"], neutral["vol"], neutral["pitch"]), (1.0, 1.0, 0))
        # 本工具音调是倍率，MiniMax 是半音：2 倍 = 升 12 个半音
        self.assertEqual(minimax_voice_setting(voice="v", pitch=2.0)["pitch"], 12)
        self.assertEqual(minimax_voice_setting(voice="v", pitch=0.5)["pitch"], -12)
        # 越界钳住，不把非法值发出去
        extreme = minimax_voice_setting(voice="v", rate=9, volume=9999, pitch=99)
        self.assertEqual(extreme["speed"], 2.0)
        self.assertEqual(extreme["vol"], 10.0)
        self.assertEqual(extreme["pitch"], 12)
        self.assertGreater(minimax_voice_setting(voice="v", volume=0)["vol"], 0)

    def test_hex_audio_is_decoded(self):
        pcm = b"\x01\x02\x03\x04"
        body = {"output": {"base_resp": {"status_code": 0}, "data": {"audio": pcm.hex(), "status": 2}}}
        self.assertEqual(parse_minimax_tts_response(body), pcm)

    def test_business_error_code_is_raised(self):
        body = {"output": {"base_resp": {"status_code": 1004, "status_msg": "auth failed"}}}
        with self.assertRaises(ValueError) as ctx:
            parse_minimax_tts_response(body)
        self.assertIn("1004", str(ctx.exception))

    def test_missing_or_broken_audio_is_raised(self):
        for body in (
            {"output": {"data": {}}},
            {"output": {"data": {"audio": "zz-not-hex"}}},
            {"output": {"data": {"audio": "010203"}}},  # 3 字节 = 半个采样点
            {},
        ):
            with self.assertRaises(ValueError):
                parse_minimax_tts_response(body)


class MiniMaxCloneTests(unittest.TestCase):
    def test_payload_matches_the_documented_shape(self):
        payload = build_minimax_clone_payload(
            target_model=MINIMAX, voice_id="defenseEN123n0",
            audio_url="https://oss/sample.wav", preview_text="Good morning.",
        )
        body = payload["input"]
        self.assertEqual(body["action"], "voice_clone")
        self.assertEqual(body["voice_id"], "defenseEN123n0")
        self.assertEqual(body["audio_url"], "https://oss/sample.wav")
        self.assertEqual(body["text"], "Good morning.")
        self.assertFalse(body["need_noise_reduction"])

    def test_voice_id_rules(self):
        self.assertTrue(valid_minimax_voice_id("defenseEN1790000000000n0"))
        for bad in ("short", "1startsWithDigit", "ends_with_", "has space here", "中文名字的音色"):
            self.assertFalse(valid_minimax_voice_id(bad), bad)

    def test_generated_names_are_valid_and_unique(self):
        names = {minimax_voice_name("en", now_ms=1790000000000, salt=i) for i in range(5)}
        self.assertEqual(len(names), 5, "同一毫秒批量提交也不能重名，重名会覆盖前一个音色")
        self.assertTrue(all(valid_minimax_voice_id(n) for n in names))
        self.assertIn("ZH", minimax_voice_name("zh", now_ms=1))

    def test_preview_text_is_required_and_trimmed(self):
        with self.assertRaises(ValueError):
            build_minimax_clone_payload(target_model=MINIMAX, voice_id="defenseEN1n0",
                                        audio_url="u", preview_text="  ")
        payload = build_minimax_clone_payload(target_model=MINIMAX, voice_id="defenseEN1n0",
                                              audio_url="u", preview_text="x" * 5000)
        self.assertEqual(len(payload["input"]["text"]), 1000)

    def test_preview_texts_are_short(self):
        """试听文本按字计费。"""
        self.assertTrue(all(len(text) < 80 for text in MINIMAX_PREVIEW_TEXT.values()))

    def test_minimax_needs_ten_second_samples(self):
        self.assertEqual(minimum_sample_seconds(MINIMAX), 10.0)
        tasks, skipped = build_tasks([MINIMAX, "qwen-audio-3.0-tts-plus"], [8], sample_seconds=26.0)
        self.assertEqual([t.model for t in tasks], ["qwen-audio-3.0-tts-plus"])
        self.assertIn("10", skipped[0].reason)

    def test_minimax_shares_the_oss_upload_with_qwen_audio(self):
        """同为 OSS 通道：同一时长的 MiniMax 与 qwen-audio 共用一次上传。"""
        from teams_voice_translator.defense.clone_batch import group_tasks

        tasks, _ = build_tasks([MINIMAX, "qwen-audio-3.0-tts-plus"], [18], sample_seconds=26.0)
        self.assertEqual(len(group_tasks(tasks)), 1)


class MiniMaxClientTests(unittest.TestCase):
    def client(self):
        from teams_voice_translator.aliyun import BailianClient

        return BailianClient("sk-test", "llm-test", proxy_mode="direct")

    def settings(self):
        return {"tts_model": MINIMAX, "voice": "defenseEN1n0", "tts_sample_rate": 24000,
                "tts_volume": 50, "tts_rate": 1.0, "tts_pitch": 1.0}

    def test_synthesis_posts_to_the_minimax_endpoint_and_delivers_pcm(self):
        client = self.client()
        pcm = b"\x10\x00" * 50
        response = MagicMock(ok=True)
        response.json.return_value = {"output": {"base_resp": {"status_code": 0},
                                                 "data": {"audio": pcm.hex()}}}
        received = []
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post:
            total = client.stream_tts("Hello.", self.settings(), received.append, threading.Event())
        self.assertEqual(total, len(pcm))
        self.assertEqual(b"".join(received), pcm)
        url = post.call_args.args[0]
        self.assertTrue(url.endswith("/api/v1/services/aigc/multimodal-generation/generation"))
        # 不开 SSE 流式，避免最后一帧附带整段拼接音频导致整句重播
        self.assertNotIn("X-DashScope-SSE", post.call_args.kwargs["headers"])

    def test_synthesis_error_becomes_api_error(self):
        from teams_voice_translator.aliyun import ApiError

        client = self.client()
        response = MagicMock(ok=True)
        response.json.return_value = {"output": {"base_resp": {"status_code": 2013, "status_msg": "bad voice"}}}
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response):
            with self.assertRaises(ApiError):
                client.stream_tts("Hi.", self.settings(), lambda b: None, threading.Event())

    def test_clone_returns_the_requested_voice_id(self):
        client = self.client()
        response = MagicMock(ok=True)
        response.json.return_value = {"output": {"base_resp": {"status_code": 0}, "demo_audio": "https://demo"}}
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response):
            voice, demo = client.clone_minimax_voice(
                target_model=MINIMAX, voice_id="defenseEN1n0",
                audio_url="https://oss/a.wav", preview_text="Hi there.",
            )
        self.assertEqual(voice, "defenseEN1n0")
        self.assertEqual(demo, "https://demo")

    def test_clone_business_failure_is_raised(self):
        from teams_voice_translator.aliyun import ApiError

        client = self.client()
        response = MagicMock(ok=True)
        response.json.return_value = {"output": {"base_resp": {"status_code": 1, "status_msg": "sample too short"}}}
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response):
            with self.assertRaises(ApiError) as ctx:
                client.clone_minimax_voice(target_model=MINIMAX, voice_id="defenseEN1n0",
                                           audio_url="u", preview_text="Hi.")
        self.assertIn("sample too short", str(ctx.exception))

    def test_session_dispatch_picks_the_minimax_session(self):
        from teams_voice_translator.defense.tts_session import TtsMiniMaxSession, create_tts_session

        session = create_tts_session(object(), {**self.settings(), "speech_mode": "natural"},
                                     on_audio=lambda b: None)
        self.assertIsInstance(session, TtsMiniMaxSession)


# ============================================================== 大模型
class ChatModelTableTests(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [item.model for item in CHAT_MODEL_TABLE]
        self.assertEqual(len(ids), len(set(ids)))

    def test_current_generation_is_listed_first(self):
        first = [item.model for item in CHAT_MODEL_TABLE[:4]]
        self.assertEqual(first, ["qwen3.8-max", "qwen3.7-plus",
                                 "qwen3.8-flash", "qwen3.7-flash"])

    def test_chat_model_families_stay_together(self):
        ids = [item.model for item in CHAT_MODEL_TABLE]
        self.assertEqual(ids[:9], [
            "qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash", "qwen3.7-flash",
            "qwen3-max", "qwen-max", "qwen-plus", "qwen-flash", "qwen-turbo",
        ])
        self.assertEqual(ids[9:11], ["deepseek-v4-pro", "deepseek-v4-flash"])
        self.assertEqual(ids[12:14], ["kimi-k3", "kimi-k2.7-code"])

    def test_legacy_models_are_kept_so_old_settings_still_load(self):
        for model in ("qwen-plus", "qwen-max", "qwen3-max"):
            self.assertTrue(chat_model(model).legacy, model)

    def test_translation_list_excludes_models_that_cannot_stop_thinking(self):
        translate = {m for m, _ in models_for(USE_TRANSLATE)}
        qa = {m for m, _ in models_for(USE_QA)}
        self.assertNotIn("kimi-k3", translate)       # 仅思考
        self.assertNotIn("MiniMax-M3", translate)    # 开关格式未实测，不敢发
        self.assertIn("kimi-k3", qa)
        self.assertIn("MiniMax-M3", qa)
        self.assertTrue(translate < qa)

    def test_translation_problem_explains_itself(self):
        self.assertIn("仅思考", translation_problem("kimi-k3"))
        self.assertEqual(translation_problem("qwen3.8-flash"), "")
        self.assertEqual(translation_problem("some-unknown-model"), "")


class ThinkingPolicyTests(unittest.TestCase):
    def test_new_models_get_thinking_explicitly_disabled(self):
        """核心回归：这些模型默认开思考，翻译请求必须显式关掉。"""
        for model in ("qwen3.8-flash", "qwen3.8-max", "qwen3.7-plus", "qwen3.7-flash",
                      "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.2"):
            self.assertEqual(thinking_fields(model, want_thinking=False),
                             {"enable_thinking": False}, model)

    def test_legacy_models_get_nothing_extra(self):
        """旧模型本来就不思考；多发字段在旧模型上有报错风险。"""
        for model in ("qwen-plus", "qwen-max", "qwen-turbo"):
            self.assertEqual(thinking_fields(model, want_thinking=False), {}, model)

    def test_unknown_models_are_left_untouched(self):
        self.assertEqual(thinking_fields("brand-new-model", want_thinking=False), {})
        self.assertEqual(thinking_fields("brand-new-model", want_thinking=True), {})

    def test_models_without_a_known_switch_are_left_untouched(self):
        self.assertEqual(thinking_fields("MiniMax-M3", want_thinking=False), {})
        self.assertEqual(thinking_fields("kimi-k3", want_thinking=True), {})

    def test_opt_in_thinking_carries_a_capped_budget(self):
        fields = thinking_fields("qwen3.8-max", want_thinking=True, budget=99999)
        self.assertTrue(fields["enable_thinking"])
        self.assertEqual(fields["thinking_budget"], 32768)


class TranslatorPayloadTests(unittest.TestCase):
    def translator(self, model):
        from teams_voice_translator.defense.translator import ContextTranslator

        return ContextTranslator(api_key="sk", workspace_id="ws", model=model)

    def test_translation_payload_turns_thinking_off(self):
        from teams_voice_translator.defense.memory import MeetingMemory

        payload = self.translator("qwen3.8-flash").build_payload(MeetingMemory(), "zh2en", "你好")
        self.assertIs(payload["enable_thinking"], False)

    def test_legacy_translation_payload_is_unchanged(self):
        from teams_voice_translator.defense.memory import MeetingMemory

        payload = self.translator("qwen-plus").build_payload(MeetingMemory(), "zh2en", "你好")
        self.assertNotIn("enable_thinking", payload)

    def stream(self, lines):
        response = MagicMock(ok=True)
        response.iter_lines.return_value = iter(lines)
        response.__enter__ = lambda self_: self_
        response.__exit__ = lambda *a: False
        return response

    def test_reasoning_is_reported_but_never_spoken(self):
        """思考内容不能混进译文——否则会被当成英文念出来。"""
        events = [
            {"choices": [{"delta": {"reasoning_content": "用户想让我翻译……"}}]},
            {"choices": [{"delta": {"reasoning_content": "应该用第一人称。"}}]},
            {"choices": [{"delta": {"content": "Hello."}, "finish_reason": "stop"}]},
        ]
        lines = [f"data: {json.dumps(e, ensure_ascii=False)}" for e in events] + ["data: [DONE]"]
        translator = self.translator("qwen3.8-max")
        seen = []
        with patch("teams_voice_translator.defense.translator.requests.post", return_value=self.stream(lines)):
            text = translator._post_stream({}, None, None, on_reasoning=seen.append)
        self.assertEqual(text, "Hello.")
        self.assertTrue(seen)
        self.assertGreater(seen[-1], 0)

    def test_qa_chat_can_opt_into_thinking(self):
        translator = self.translator("qwen3.8-max")
        captured = {}

        def fake_post(payload, on_piece, cancel, **kwargs):
            captured.update(payload)
            captured["deadline"] = kwargs.get("deadline_seconds")
            return "EN: ok\nZH: 好"

        translator._post_stream = fake_post
        translator.chat([{"role": "user", "content": "q"}], thinking=True)
        self.assertTrue(captured["enable_thinking"])
        self.assertIn("thinking_budget", captured)
        self.assertEqual(captured["deadline"], 180, "思考模式要放宽超时，否则 60 秒会误判超时")

    def test_qa_chat_defaults_to_no_thinking(self):
        translator = self.translator("qwen3.8-max")
        captured = {}
        translator._post_stream = lambda payload, *a, **k: captured.update(payload) or "EN: a"
        translator.chat([{"role": "user", "content": "q"}])
        self.assertIs(captured["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
