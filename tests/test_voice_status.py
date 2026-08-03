import unittest
from unittest.mock import Mock, patch

from teams_voice_translator.aliyun import ApiError, BailianClient


class VoiceStatusTests(unittest.TestCase):
    def setUp(self):
        self.client = BailianClient("sk-test", "ws-test")

    def test_waits_until_cloned_voice_is_ready(self):
        self.client.query_voice = Mock(
            side_effect=[
                {"status": "DEPLOYING"},
                {"status": "OK", "target_model": "qwen-audio-3.0-tts-flash"},
            ]
        )
        statuses = []
        with patch("teams_voice_translator.aliyun.time.sleep"):
            detail = self.client.wait_for_voice_ready(
                "qwen-audio-3.0-tts-flash-myvoice-123",
                on_status=statuses.append,
            )
        self.assertEqual(detail["status"], "OK")
        self.assertTrue(statuses)

    def test_rejected_cloned_voice_has_actionable_error(self):
        self.client.query_voice = Mock(return_value={"status": "UNDEPLOYED"})
        with self.assertRaisesRegex(ApiError, "UNDEPLOYED"):
            self.client.wait_for_voice_ready("qwen-audio-3.0-tts-flash-myvoice-123")

    def test_model_mismatch_explains_engine_error(self):
        self.client.query_voice = Mock(
            return_value={
                "status": "OK",
                "target_model": "qwen-audio-3.0-tts-plus",
            }
        )
        detail = self.client._explain_cloned_voice_error(
            "Engine return error code: 431",
            voice_id="qwen-audio-3.0-tts-plus-myvoice-123",
            model="qwen-audio-3.0-tts-flash",
        )
        self.assertIn("必须完全一致", detail)


if __name__ == "__main__":
    unittest.main()
