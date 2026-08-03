import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teams_voice_translator.config import DEFAULTS, SettingsStore


class SettingsTests(unittest.TestCase):
    def test_live_translate_is_the_default_f9_engine(self):
        self.assertEqual(DEFAULTS["translation_engine"], "live")
        self.assertFalse(DEFAULTS["continuous_f9_enabled"])
        self.assertEqual(
            DEFAULTS["live_translate_model"],
            "qwen3.5-livetranslate-flash-realtime",
        )

    def test_json_never_contains_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SettingsStore(Path(temp))
            store.values["workspace_id"] = "ws-test"
            store.values["api_key"] = "must-not-be-written"
            store.save()
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(raw["workspace_id"], "ws-test")
            self.assertNotIn("api_key", raw)

    def test_oss_secret_is_saved_only_to_keyring(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SettingsStore(Path(temp))
            store.update(
                {
                    "oss_region": "cn-beijing",
                    "oss_bucket": "voice-private",
                }
            )
            with patch("teams_voice_translator.config.keyring.set_password") as set_password:
                store.set_oss_access_key_id("LTAI-test")
                store.set_oss_access_key_secret("never-write-this-secret")
            self.assertEqual(set_password.call_count, 2)
            raw = store.path.read_text(encoding="utf-8")
            self.assertNotIn("LTAI-test", raw)
            self.assertNotIn("never-write-this-secret", raw)


if __name__ == "__main__":
    unittest.main()
