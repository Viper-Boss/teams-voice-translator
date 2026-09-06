import unittest

from teams_voice_translator.api_payloads import (
    build_asr_session_update,
    build_live_translate_session_update,
    build_qwen3_tts_http_payload,
    build_qwen3_tts_realtime_session_update,
    build_qwen_voice_clone_payload,
    build_translation_payload,
    build_tts_payload,
    build_voice_clone_payload,
    cosyvoice_instruction_units,
    parse_json_list,
)


class PayloadTests(unittest.TestCase):
    def test_asr_low_latency_session(self):
        payload = build_asr_session_update(language="zh", threshold=0.0, silence_ms=400)
        self.assertEqual(payload["session"]["sample_rate"], 16000)
        self.assertEqual(payload["session"]["turn_detection"]["silence_duration_ms"], 400)

    def test_translation_controls(self):
        terms = parse_json_list('[{"source":"有限元","target":"finite element method"}]', "术语表")
        payload = build_translation_payload(
            "有限元",
            terms=terms,
            domain="academic discussion",
        )
        self.assertEqual(payload["translation_options"]["terms"], terms)
        self.assertEqual(payload["translation_options"]["domains"], "academic discussion")

    def test_tts_uses_pcm_for_live_output(self):
        payload = build_tts_payload(
            "Hello",
            model="qwen-audio-3.0-tts-flash",
            voice="loongjohn",
            emotion_tag="[amazed]",
        )
        self.assertEqual(payload["input"]["format"], "pcm")
        self.assertTrue(payload["input"]["text"].startswith("[amazed]"))

    def test_cosyvoice_instruction_uses_official_weighted_limit(self):
        self.assertEqual(cosyvoice_instruction_units("请用 English"), 4 + 8)
        with self.assertRaisesRegex(ValueError, "100"):
            build_tts_payload(
                "Hello",
                model="cosyvoice-v3.5-plus",
                voice="voice-test",
                instruction="请" * 51,
            )

    def test_instruction_parameter_name_matches_model_family(self):
        cosy = build_tts_payload(
            "Hello", model="cosyvoice-v3.5-plus", voice="voice", instruction="Speak warmly."
        )
        qwen = build_tts_payload(
            "Hello", model="qwen-audio-3.0-tts-plus", voice="voice", instruction="Speak warmly."
        )
        self.assertEqual(cosy["input"]["instruction"], "Speak warmly.")
        self.assertNotIn("instructions", cosy["input"])
        self.assertEqual(qwen["input"]["instructions"], "Speak warmly.")
        self.assertNotIn("instruction", qwen["input"])

    def test_live_translate_push_to_talk_with_voice_clone(self):
        payload = build_live_translate_session_update(
            phrases={"有限元": "finite element method"},
            voice_mode="once",
        )
        session = payload["session"]
        self.assertIsNone(session["turn_detection"])
        self.assertEqual(session["translation"]["language"], "en")
        self.assertEqual(
            session["translation"]["corpus"]["phrases"]["有限元"],
            "finite element method",
        )
        self.assertEqual(session["voice_clone_options"]["frequency"], "once")
        self.assertEqual(session["input_audio_transcription"]["language"], "zh")

    def test_live_translate_fixed_voice_requires_id(self):
        with self.assertRaises(ValueError):
            build_live_translate_session_update(voice_mode="fixed", voice="")

    def test_live_translate_continuous_mode_uses_server_vad(self):
        payload = build_live_translate_session_update(
            continuous=True,
            vad_threshold=0.15,
            vad_silence_ms=700,
        )
        turn_detection = payload["session"]["turn_detection"]
        self.assertEqual(turn_detection["type"], "server_vad")
        self.assertEqual(turn_detection["threshold"], 0.15)
        self.assertEqual(turn_detection["silence_duration_ms"], 700)

    def test_voice_clone_payload(self):
        payload = build_voice_clone_payload(
            target_model="qwen-audio-3.0-tts-flash",
            prefix="myvoice",
            audio_url="https://example.com/a.wav",
            preprocess=True,
        )
        self.assertEqual(payload["model"], "voice-enrollment")
        self.assertEqual(payload["input"]["action"], "create_voice")
        self.assertTrue(payload["input"]["enable_preprocess"])

    def test_live_translate_voice_clone_omits_unsupported_preprocess_fields(self):
        payload = build_voice_clone_payload(
            target_model="qwen3.5-livetranslate-flash-realtime",
            prefix="myvoice",
            audio_url="https://example.com/a.wav",
            language="zh",
            max_seconds=30,
            preprocess=True,
        )
        self.assertEqual(
            payload["input"],
            {
                "action": "create_voice",
                "target_model": "qwen3.5-livetranslate-flash-realtime",
                "prefix": "myvoice",
                "url": "https://example.com/a.wav",
            },
        )

    def test_qwen3_voice_clone_uses_official_qwen_enrollment_schema(self):
        payload = build_qwen_voice_clone_payload(
            target_model="qwen3-tts-vc-realtime-2026-01-15",
            preferred_name="my_voice",
            audio_data="data:audio/wav;base64,UklGRg==",
            language="zh",
            transcript="这是我的样音。",
        )
        self.assertEqual(payload["model"], "qwen-voice-enrollment")
        self.assertEqual(payload["input"]["action"], "create")
        self.assertEqual(payload["input"]["preferred_name"], "my_voice")
        self.assertEqual(payload["input"]["audio"]["data"], "data:audio/wav;base64,UklGRg==")
        self.assertEqual(payload["input"]["text"], "这是我的样音。")

    def test_qwen3_realtime_tts_uses_commit_mode_and_english(self):
        payload = build_qwen3_tts_realtime_session_update(
            voice="voice-123",
            language="en",
            rate=1.1,
            pitch=0.95,
        )
        self.assertEqual(payload["type"], "session.update")
        self.assertEqual(payload["session"]["mode"], "commit")
        self.assertEqual(payload["session"]["language_type"], "English")
        self.assertEqual(payload["session"]["speech_rate"], 1.1)

    def test_qwen3_http_tts_binds_voice_and_model(self):
        payload = build_qwen3_tts_http_payload(
            "Hello",
            model="qwen3-tts-vc-2026-01-22",
            voice="voice-123",
            language="en",
        )
        self.assertEqual(payload["model"], "qwen3-tts-vc-2026-01-22")
        self.assertEqual(payload["input"]["voice"], "voice-123")
        self.assertEqual(payload["input"]["language_type"], "English")


if __name__ == "__main__":
    unittest.main()
