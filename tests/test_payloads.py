import unittest

from teams_voice_translator.api_payloads import (
    build_asr_session_update,
    build_translation_payload,
    build_tts_payload,
    build_voice_clone_payload,
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

    def test_voice_clone_payload(self):
        payload = build_voice_clone_payload(
            target_model="qwen-audio-3.0-tts-flash",
            prefix="myvoice",
            audio_url="https://example.com/a.wav",
        )
        self.assertEqual(payload["model"], "voice-enrollment")
        self.assertEqual(payload["input"]["action"], "create_voice")


if __name__ == "__main__":
    unittest.main()
