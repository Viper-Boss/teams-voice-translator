from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import voice_ab


class VoiceAbTests(unittest.TestCase):
    def test_candidate_parser_preserves_colons_inside_voice_id(self) -> None:
        self.assertEqual(
            voice_ab.parse_candidate("model-name:voice:variant"),
            ("model-name", "voice:variant"),
        )

    def test_default_mode_never_synthesizes(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.object(
            voice_ab, "synthesize", side_effect=AssertionError("cloud call attempted")
        ):
            result = voice_ab.main(
                [
                    "--candidate", "model-a:voice-a",
                    "--candidate", "model-b:voice-b",
                    "--out", folder,
                ]
            )
            self.assertEqual(result, 0)
            self.assertFalse((Path(folder) / "mapping.json").exists())

    def test_execute_requires_two_successful_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.object(
            voice_ab, "synthesize", side_effect=[b"\x00\x00" * 16, RuntimeError("failed")]
        ), patch(
            "teams_voice_translator.defense.settings.DefenseSettings", return_value=object()
        ):
            result = voice_ab.main(
                [
                    "--execute",
                    "--candidate", "model-a:voice-a",
                    "--candidate", "model-b:voice-b",
                    "--out", folder,
                ]
            )
            self.assertEqual(result, 1)
            payload = json.loads((Path(folder) / "mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["candidates"]), 1)
            self.assertEqual(len(payload["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
