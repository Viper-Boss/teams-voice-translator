from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.defense.settings import DEFENSE_DEFAULTS, DefenseSettings


class DefenseSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmp.name)
        self.settings = DefenseSettings(base_dir=self.base_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_defaults(self) -> None:
        self.assertEqual(self.settings.get("tts_model"), "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(self.settings.get("llm_model"), "qwen-plus")
        self.assertEqual(self.settings.get("tts_voice_id"), "")

    def test_round_trip(self) -> None:
        self.settings.set("tts_voice_id", "voice-123")
        self.settings.set("vad_silence_ms", 350)
        reloaded = DefenseSettings(base_dir=self.base_dir)
        self.assertEqual(reloaded.get("tts_voice_id"), "voice-123")
        self.assertEqual(reloaded.get("vad_silence_ms"), 350)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.settings.set("not_a_defense_key", 1)

    def test_get_falls_through_to_shared(self) -> None:
        self.settings.shared.update({"workspace_id": "llm-demo"})
        reloaded = DefenseSettings(base_dir=self.base_dir)
        self.assertEqual(reloaded.workspace_id, "llm-demo")
        self.assertEqual(reloaded.get("workspace_id"), "llm-demo")

    def test_glossary_round_trip(self) -> None:
        terms = [{"source": "有限元", "target": "FEM"}]
        self.settings.set_glossary(terms)
        self.assertEqual(self.settings.glossary_terms(), terms)

    def test_glossary_invalid_json_returns_empty(self) -> None:
        self.settings.values["glossary"] = "{broken"
        self.assertEqual(self.settings.glossary_terms(), [])

    def test_glossary_filters_bad_entries(self) -> None:
        self.settings.set_glossary([{"source": "", "target": "x"}, {"source": "a", "target": "b"}])
        self.assertEqual(self.settings.glossary_terms(), [{"source": "a", "target": "b"}])

    def test_file_has_no_secrets(self) -> None:
        self.settings.set("tts_voice_id", "voice-abc")
        raw = (self.base_dir / "defense_settings.json").read_text(encoding="utf-8")
        self.assertNotIn("api_key", raw)
        self.assertNotIn("sk-", raw)
        for key in DEFENSE_DEFAULTS:
            self.assertIsInstance(key, str)


if __name__ == "__main__":
    unittest.main()
