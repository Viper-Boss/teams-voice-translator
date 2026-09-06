from __future__ import annotations

import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import Mock

from teams_voice_translator.defense.ambience import AMBIENCE_PRESETS, RoomTone
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings


class AmbienceTests(unittest.TestCase):
    def test_off_is_bit_exact_and_has_no_tail(self) -> None:
        tone = RoomTone("off")
        pcm = b"\x01\x00\xfe\xff"
        self.assertEqual(tone.mix(pcm), pcm)
        self.assertEqual(tone.tail(), b"")

    def test_subtle_room_tone_is_low_level_and_fades_after_speech(self) -> None:
        tone = RoomTone("subtle", seed=1)
        mixed = tone.mix(bytes(2400 * 2))
        samples = array("h", mixed)
        self.assertTrue(any(samples))
        self.assertLess(max(abs(value) for value in samples), 500)

        tail = tone.tail()
        expected = int(24000 * AMBIENCE_PRESETS["subtle"].tail_ms / 1000) * 2
        self.assertEqual(len(tail), expected)
        tail_samples = array("h", tail)
        window = max(1, len(tail_samples) // 8)
        start_energy = sum(abs(value) for value in tail_samples[:window])
        end_energy = sum(abs(value) for value in tail_samples[-window:])
        self.assertLess(end_energy, start_energy)

    def test_pipeline_keeps_clean_cache_but_outputs_tone_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp))
            settings.set("ambience_mode", "subtle")
            engine = DefenseEngine(settings, EngineCallbacks())
            engine.player = Mock()
            raw = bytes(1920)
            engine._on_first_audio("Hello.")
            engine._on_tts_audio(raw)
            self.assertNotEqual(engine.player.write.call_args_list[0].args[0], raw)
            self.assertEqual(bytes(engine._live_utterance["buf"]), raw)
            engine._on_utterance_done("Hello.", len(raw))
            self.assertGreater(engine.player.write.call_count, 1)
            self.assertEqual(engine._pcm_cache["Hello."].pcm, raw)
            engine.player = None
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
