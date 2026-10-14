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
        self.assertEqual(self.settings.get("tts_model"), "qwen-audio-3.0-tts-plus")
        # 2026-09：qwen-plus / qwen3-max 已列为旧版，默认换到当前代。
        self.assertEqual(self.settings.get("llm_model"), "qwen3.8-flash")
        self.assertEqual(self.settings.get("qa_model"), "qwen3.8-flash")
        self.assertEqual(self.settings.get("asr_model"), "qwen-audio-3.1-asr-flash-streaming")
        self.assertEqual(self.settings.get("committee_asr_model"), "qwen-audio-3.1-asr-flash-streaming")
        self.assertFalse(self.settings.get("qa_thinking"))
        self.assertEqual(self.settings.get("tts_voice_id"), "")
        self.assertEqual(self.settings.get("translation_style"), "accurate")

    def test_default_tts_model_supports_instruction_and_ssml(self) -> None:
        """回归：旧默认 qwen3-tts-vc-realtime-* 两样都不支持，语气预设全是灰的。"""
        from teams_voice_translator.api_payloads import tts_supports

        model = self.settings.get("tts_model")
        self.assertTrue(tts_supports(model, "instruction"))
        self.assertTrue(tts_supports(model, "ssml"))

    def test_clone_language_defaults_to_english(self) -> None:
        """跨语种复刻质量下降；答辩说英文，默认就该用英文样音。"""
        self.assertEqual(self.settings.get("clone_language"), "en")

    def test_old_default_asr_migrates_once_but_user_choice_is_kept(self) -> None:
        """还停在旧默认的换成推荐模型；用户主动选的别的模型不碰；迁移只做一次。"""
        path = self.base_dir / "defense_settings.json"
        path.write_text(json.dumps({
            "asr_model": "qwen3-asr-flash-realtime",
            "committee_asr_model": "fun-asr-realtime",
        }), encoding="utf-8")
        migrated = DefenseSettings(base_dir=self.base_dir)
        self.assertEqual(migrated.get("asr_model"), "qwen-audio-3.1-asr-flash-streaming")
        self.assertEqual(migrated.get("committee_asr_model"), "fun-asr-realtime")
        # 迁移后用户又主动选回旧模型：不能再被改掉
        migrated.set("asr_model", "qwen3-asr-flash-realtime")
        again = DefenseSettings(base_dir=self.base_dir)
        self.assertEqual(again.get("asr_model"), "qwen3-asr-flash-realtime")

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
