"""模型能力表是唯一事实来源，UI / speech_policy / payload 必须都跟着它走。

以前三个模块各自手写一份集合，结果：
  - qwen-audio-3.1-tts-flash 三处都漏了，下拉框里根本选不到；
  - 默认模型 qwen3-tts-vc-realtime-* 既无指令也无 SSML，语气预设永远是灰的。
这些测试把「表」和「用表的地方」钉在一起，避免再次漂移。
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.api_payloads import (
    INSTRUCTION_MODELS,
    SSML_MODELS,
    TTS_CAPABILITIES,
    VOICE_PREPROCESS_MODELS,
    build_tts_payload,
    build_voice_clone_payload,
    is_qwen_audio_tts_model,
    tts_supports,
)
from teams_voice_translator.defense import speech_policy


class CapabilityTableTests(unittest.TestCase):
    def test_derived_sets_match_the_table(self):
        for model, caps in TTS_CAPABILITIES.items():
            self.assertEqual(model in INSTRUCTION_MODELS, "instruction" in caps, model)
            self.assertEqual(model in SSML_MODELS, "ssml" in caps, model)
            self.assertEqual(model in VOICE_PREPROCESS_MODELS, "preprocess" in caps, model)

    def test_unknown_model_supports_nothing(self):
        self.assertFalse(tts_supports("totally-made-up-model", "instruction"))
        self.assertFalse(tts_supports("", "ssml"))

    def test_speech_policy_reexports_the_same_sets(self):
        self.assertIs(speech_policy.INSTRUCTION_MODELS, INSTRUCTION_MODELS)
        self.assertIs(speech_policy.SSML_MODELS, SSML_MODELS)

    def test_ui_dropdown_only_offers_known_models(self):
        from teams_voice_translator.defense.ui import TTS_MODELS

        listed = [model for model, _blurb in TTS_MODELS]
        self.assertEqual(len(listed), len(set(listed)), "下拉框有重复模型")
        for model in listed:
            self.assertIn(model, TTS_CAPABILITIES, f"{model} 不在能力表里")

    def test_ui_dropdown_offers_every_instruction_capable_model(self):
        """支持语气指令的模型必须都能选到，否则等于没做。"""
        from teams_voice_translator.defense.ui import TTS_MODELS

        listed = {model for model, _blurb in TTS_MODELS}
        self.assertTrue(INSTRUCTION_MODELS <= listed, INSTRUCTION_MODELS - listed)

    def test_qwen_audio_three_one_is_fully_wired(self):
        """3.1-flash 是之前漏得最彻底的一个，单独钉住。"""
        model = "qwen-audio-3.1-tts-flash"
        self.assertTrue(tts_supports(model, "instruction"))
        self.assertTrue(tts_supports(model, "ssml"))
        self.assertTrue(tts_supports(model, "preprocess"))
        self.assertTrue(is_qwen_audio_tts_model(model))
        payload = build_tts_payload(
            "Hello", model=model, voice="v", instruction="Speak calmly."
        )["input"]
        self.assertEqual(payload["instruction"], "Speak calmly.")
        self.assertNotIn("instructions", payload)
        clone = build_voice_clone_payload(
            target_model=model, prefix="p", audio_url="https://e/a.wav",
            max_seconds=18.0, preprocess=True, volume_normalization=True,
        )["input"]
        self.assertEqual(clone["max_prompt_audio_length"], 18.0)
        self.assertEqual(clone["enable_volume_normalization"], "true")


class SessionDispatchTests(unittest.TestCase):
    """能力表里的每个模型都必须能被 create_tts_session 分发出去。

    回归：qwen-audio-3.1-tts-flash 曾经在能力表和下拉框里都有，但
    create_tts_session 里写死了 `qwen-audio-3.0-tts` 前缀，选中它一开口
    就报「不支持的语音合成模型」—— 界面全绿，实际用不了。
    """

    def test_every_listed_model_can_build_a_session(self):
        from teams_voice_translator.aliyun import ApiError
        from teams_voice_translator.defense.tts_session import create_tts_session

        for model in TTS_CAPABILITIES:
            settings = {"tts_model": model, "voice": "voice-test", "speech_mode": "natural"}
            try:
                session = create_tts_session(object(), settings, on_audio=lambda chunk: None)
            except ApiError as exc:
                self.fail(f"{model} 无法分发到任何会话实现：{exc}")
            self.assertIsNotNone(session, model)

    def test_unknown_model_is_still_rejected_loudly(self):
        from teams_voice_translator.aliyun import ApiError
        from teams_voice_translator.defense.tts_session import create_tts_session

        with self.assertRaises(ApiError):
            create_tts_session(
                object(), {"tts_model": "not-a-real-model", "voice": "v"},
                on_audio=lambda chunk: None,
            )

    def test_qwen_audio_family_covers_every_minor_version(self):
        for model in ("qwen-audio-3.0-tts-plus", "qwen-audio-3.1-tts-flash", "qwen-audio-3.9-tts-x"):
            self.assertTrue(is_qwen_audio_tts_model(model), model)
        self.assertFalse(is_qwen_audio_tts_model("qwen3-tts-vc-2026-01-22"))
        self.assertFalse(is_qwen_audio_tts_model("cosyvoice-v3.5-plus"))


class SentenceGroupingTests(unittest.TestCase):
    def test_ssml_join_keeps_one_prosody_curve(self):
        text = speech_policy.join_sentences_ssml(
            ["First point.", "Second point."], "qwen-audio-3.0-tts-plus"
        )
        self.assertTrue(text.startswith("<speak>"))
        self.assertIn(f'<break time="{speech_policy.SENTENCE_BREAK_MS}ms"/>', text)

    def test_single_sentence_is_not_wrapped(self):
        self.assertEqual(
            speech_policy.join_sentences_ssml(["Only one."], "qwen-audio-3.0-tts-plus"),
            "Only one.",
        )

    def test_model_without_ssml_degrades_to_plain_text(self):
        text = speech_policy.join_sentences_ssml(
            ["A.", "B."], "qwen3-tts-vc-realtime-2026-01-15"
        )
        self.assertEqual(text, "A. B.")
        self.assertNotIn("<speak>", text)

    def test_ssml_escapes_user_text(self):
        text = speech_policy.join_sentences_ssml(
            ["R&D on <T>.", "Done."], "qwen-audio-3.0-tts-plus"
        )
        self.assertIn("R&amp;D", text)
        self.assertNotIn("<T>", text)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            speech_policy.join_sentences_ssml(["", "   "], "qwen-audio-3.0-tts-plus")

    def test_grouping_respects_both_bounds(self):
        groups = speech_policy.group_sentences(["a", "b", "c", "d", "e"], max_per_group=2)
        self.assertEqual(groups, [["a", "b"], ["c", "d"], ["e"]])

    def test_grouping_splits_on_character_limit(self):
        long = "x" * 300
        groups = speech_policy.group_sentences([long, long], max_per_group=3, limit=450)
        self.assertEqual(len(groups), 2)

    def test_oversized_single_sentence_stands_alone(self):
        long = "y" * 600
        groups = speech_policy.group_sentences(["short.", long], max_per_group=3, limit=450)
        self.assertEqual(groups, [["short."], [long]])

    def test_blank_sentences_are_dropped(self):
        self.assertEqual(
            speech_policy.group_sentences(["a", "", "  ", "b"], max_per_group=5),
            [["a", "b"]],
        )

    def test_opening_fillers_are_short_enough_to_feel_spontaneous(self):
        self.assertTrue(speech_policy.OPENING_FILLERS_EN)
        for filler in speech_policy.OPENING_FILLERS_EN:
            self.assertLessEqual(len(filler.split()), 5, filler)


if __name__ == "__main__":
    unittest.main()
