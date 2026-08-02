import json
import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.config import SettingsStore


class SettingsTests(unittest.TestCase):
    def test_json_never_contains_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SettingsStore(Path(temp))
            store.values["workspace_id"] = "ws-test"
            store.values["api_key"] = "must-not-be-written"
            store.save()
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(raw["workspace_id"], "ws-test")
            self.assertNotIn("api_key", raw)


if __name__ == "__main__":
    unittest.main()
