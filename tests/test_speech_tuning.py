"""Offline regressions for audio framing and controlled, monitor-only comparisons."""
import base64
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from xml.etree import ElementTree

from teams_voice_translator.aliyun import ApiError, BailianClient
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.speech_policy import pause_comparison_text, speech_settings, speech_chunks
from teams_voice_translator.defense.tts_session import TtsLegacySession


class SpeechTuningTests(unittest.TestCase):
    def test_fragmented_pcm_preserves_every_sample(self):
        for mode in ("natural", "streaming"):
            with self.subTest(mode=mode):
                client = Mock()
                def stream(text, settings, cb, cancel):
                    for part in (b"1", b"23", b"4", b"567", b"8"):
                        cb(part)
                client._stream_legacy_tts.side_effect = stream
                audio, errors = [], []
                session = TtsLegacySession(client, {"voice": "test", "speech_mode": mode},
                                           on_audio=audio.append, on_error=errors.append)
                try:
                    session.start(); session.speak("test")
                    self.assertTrue(session.wait_until_idle(2))
                    self.assertEqual(b"".join(audio), b"12345678")
                    self.assertTrue(all(len(part) % 2 == 0 for part in audio))
                    self.assertFalse(errors)
                finally:
                    session.close()

    def test_incomplete_sample_cannot_be_reported_as_completed(self):
        client = Mock()
        client._stream_legacy_tts.side_effect = lambda t, s, cb, c: cb(b"123")
        for mode in ("natural", "streaming"):
            errors, audio = [], []
            done = Mock()
            session = TtsLegacySession(client, {"voice": "test", "speech_mode": mode},
                on_audio=audio.append, on_error=errors.append, on_utterance_done=done, max_attempts=1)
            try:
                session.start(); session.speak("test")
                self.assertTrue(session.wait_until_idle(2))
                self.assertIn("不完整", errors[0]); done.assert_not_called()
                if mode == "natural": self.assertFalse(audio)
            finally:
                session.close()

    def test_old_sample_rate_cannot_change_export_pitch_or_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp))
            settings.set("tts_sample_rate", 48000)
            engine = DefenseEngine(settings, EngineCallbacks())
            try:
                engine.archive_folder = Path(tmp)
                (Path(tmp) / "录音").mkdir()
                engine._meeting_pcm = bytearray(48000)
                self.assertEqual(engine.meeting_audio_seconds(), 1.0)
                engine._submit = lambda action, label: action()
                engine.export_meeting_audio()
                with wave.open(str(Path(tmp) / "录音" / "英文语音_全场.wav")) as handle:
                    self.assertEqual(handle.getframerate(), 24000)
                    self.assertEqual(handle.getnframes(), 24000)
            finally:
                engine.shutdown()

    def test_pause_markup_escapes_text_without_changing_scientific_content(self):
        text = 'Fig. 2 shows x < 3.14 & y > 2.\nThe result is stable.'
        xml = pause_comparison_text(text, "cosyvoice-v3.5-plus")
        root = ElementTree.fromstring(xml)
        self.assertEqual(len(root.findall("break")), 1)
        self.assertEqual(root[0].attrib, {"time": "250ms"})
        self.assertEqual("".join(root.itertext()), text.replace("\n", ""))
        with self.assertRaises(ValueError):
            pause_comparison_text(text, "qwen3-tts-vc-2026-01-22")
        with self.assertRaises(ValueError):
            pause_comparison_text("one line", "cosyvoice-v3.5-plus")

    def test_ssml_flag_reaches_real_http_payload_only_when_requested(self):
        client = BailianClient("test", "test")
        response = Mock(ok=True)
        response.iter_lines.return_value = ['data: ' + json.dumps({"output": {"audio": {
            "data": base64.b64encode(b"1234").decode()}}}), 'data: [DONE]']
        settings = speech_settings({"tts_model": "cosyvoice-v3.5-plus", "tts_voice_id": "test"})
        settings["tts_enable_ssml"] = True
        text = pause_comparison_text("First.\nSecond.", settings["tts_model"])
        with patch("teams_voice_translator.aliyun.requests.post", return_value=response) as post:
            client.stream_tts(text, settings, lambda pcm: None, threading.Event())
            payload = post.call_args.kwargs["json"]["input"]
            self.assertTrue(payload["enable_ssml"])
            self.assertEqual(payload["text"], text)
            self.assertEqual(payload["sample_rate"], 24000)
            self.assertEqual(payload["format"], "pcm")
            settings.pop("tts_enable_ssml")
            client.stream_tts("plain", settings, lambda pcm: None, threading.Event())
            self.assertNotIn("enable_ssml", post.call_args.kwargs["json"]["input"])

    def test_probe_failure_makes_exactly_one_request(self):
        from teams_voice_translator.defense.ui import voice_probe
        client = Mock()
        client._stream_legacy_tts.side_effect = ApiError("offline")
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp))
            with patch("teams_voice_translator.defense.ui.BailianClient", return_value=client), \
                 patch("teams_voice_translator.audio.MultiOutputPlayer", return_value=MagicMock()):
                with self.assertRaisesRegex(ApiError, "offline"):
                    voice_probe(settings, "ws", "key", "voice", "cosyvoice-v3.5-plus")
            client._stream_legacy_tts.assert_called_once()

    def test_cached_comparison_replay_never_calls_cloud_and_invalidates_on_edit(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import VoiceCompareDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            dialog = VoiceCompareDialog(None, DefenseSettings(Path(tmp)))
            dialog._cached_audio[0] = b"1234"
            player = MagicMock()
            with patch("teams_voice_translator.defense.ui.voice_probe") as synth, \
                 patch("teams_voice_translator.audio.MultiOutputPlayer", return_value=player) as factory:
                dialog._replay(0)
                deadline = time.monotonic() + 2
                while not dialog.a_button.isEnabled() and time.monotonic() < deadline:
                    app.processEvents(); time.sleep(.01)
                self.assertTrue(dialog.a_button.isEnabled())
                synth.assert_not_called()
                factory.assert_called_once_with([None], 24000)
                player.__enter__.return_value.write.assert_called_once_with(b"1234")
            dialog.text_edit.setPlainText("Changed.")
            self.assertFalse(dialog._cached_audio)
            self.assertFalse(dialog.replay_buttons[0].isEnabled())
            dialog.close()

    def test_long_answer_keeps_figure_reference_with_following_words(self):
        text = "We tested this method repeatedly and present the result in Fig. 2 for comparison. The next experiment uses more data."
        chunks = speech_chunks(text, limit=88)
        self.assertIn("Fig. 2", chunks[0])
        self.assertTrue(all(len(chunk) <= 88 for chunk in chunks))
        with self.assertRaises(ValueError): speech_chunks(text, 0)


if __name__ == "__main__":
    unittest.main()
