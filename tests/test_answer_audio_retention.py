"""Paid audio survives batch stop, model switches and restart; synthesis is mocked."""
import threading
import unittest
from unittest.mock import patch

import test_qa_bank as fixtures
import test_batch16 as ui_fixtures
from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.tts_cache import RenderManifest

SYNTH = "teams_voice_translator.defense.tts_session.synthesize_once"


class AnswerAudioRetentionTests(unittest.TestCase):
    setUp = fixtures.EnginePreparedAnswerTests.setUp

    def change_voice(self):
        self.settings.update({"prerender_tts_model": "cosyvoice-v3.5-plus", "prerender_voice_id": "different-voice"})

    def test_completed_audio_survives_model_switch_and_restart_without_any_synthesis(self):
        text, pcm = "I used a controlled evaluation.", b"\x01\x00" * 48000
        with patch(SYNTH, return_value=pcm) as synth:
            self.engine.render_answer(text)
            original = self.engine.answer_audio_source(text)
            self.change_voice()
            self.assertTrue(self.engine.answer_ready(text))
            self.assertEqual(self.engine.answer_state(text), "other")
            self.assertEqual(self.engine.answer_pcm(text), pcm)
            self.assertEqual(self.engine.answer_audio_seconds(text), 2)
            self.assertEqual(self.engine.answer_audio_source(text), original)
            self.assertEqual(self.engine.render_answer(text), 0)
            self.assertTrue(self.engine.speak_prepared(text))
            self.assertEqual(synth.call_count, 1)
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        restarted = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(restarted.shutdown)
        self.assertTrue(restarted.answer_ready(text))
        self.assertEqual(restarted.answer_pcm(text), pcm)

    def test_cancellation_keeps_completed_chunk_and_resume_only_synthesizes_missing_chunk(self):
        text = "First part. Second part."
        chunks = ["First part.", "Second part."]
        cancel = threading.Event()
        first, second = b"\x01\x00" * 24, b"\x02\x00" * 24
        def complete_inflight(*args, **kwargs):
            self.assertIsNone(kwargs.get("cancel_event"), "stop must not throw away the already submitted chunk")
            cancel.set()
            return first
        with patch("teams_voice_translator.defense.pipeline.speech_chunks", return_value=chunks), \
                patch(SYNTH, side_effect=complete_inflight) as synth:
            with self.assertRaises(ApiError):
                self.engine.render_answer(text, cancel=cancel)
            self.assertEqual(synth.call_count, 1)
            self.assertEqual(self.engine.answer_state(text), "partial")
            self.change_voice()
            self.assertEqual(self.engine.answer_state(text), "partial")
            self.assertGreater(self.engine.answer_audio_seconds(text), 0)
            self.settings.update({"prerender_tts_model": "", "prerender_voice_id": ""})
            cancel.clear()
            with patch(SYNTH, return_value=second) as remaining:
                self.assertEqual(self.engine.render_answer(text, cancel=cancel), 1)
                remaining.assert_called_once()
                self.assertEqual(remaining.call_args.args[2], chunks[1])
            self.assertEqual(self.engine.answer_pcm(text), first + second)

    def test_legacy_audio_is_found_from_voice_library_without_rebilling(self):
        text, pcm = "My legacy answer.", b"\x01\x00" * 48000
        with patch(SYNTH, return_value=pcm):
            self.engine.render_answer(text)
        source = self.engine.answer_audio_source(text)
        self.engine.answer_catalog.remove(self.engine._answer_catalog_id(text))
        self.engine.answer_catalog = RenderManifest(self.engine.disk_cache, "answer_audio_catalog.json")
        self.settings.set("voice_library", [{"target_model": source["model"], "voice_id": source["voice"]}])
        self.change_voice()
        with patch(SYNTH, side_effect=AssertionError("must not contact cloud")):
            self.assertTrue(self.engine.answer_ready(text))
            self.assertEqual(self.engine.answer_pcm(text), pcm)
            self.assertEqual(self.engine.render_answer(text), 0)

    def test_different_models_are_never_spliced_into_one_answer(self):
        text, chunks = "First. Second.", ["First.", "Second."]
        pcm = b"\x01\x00" * 48
        with patch("teams_voice_translator.defense.pipeline.speech_chunks", return_value=chunks), \
                patch(SYNTH, return_value=pcm):
            self.engine.render_answer(text)
            self.change_voice()
            # Explicitly request a new version and fail on its second chunk.
            with patch(SYNTH, side_effect=[b"\x02\x00" * 48, ApiError("cancelled")]):
                with self.assertRaises(ApiError):
                    self.engine.render_answer_override(text)
            self.assertEqual(self.engine.answer_pcm(text), pcm * 2)

    def test_cache_write_failure_is_not_reported_as_success(self):
        with patch(SYNTH, return_value=b"\x01\x00" * 10), \
                patch.object(self.engine.disk_cache, "put", return_value=False):
            with self.assertRaisesRegex(ApiError, "未能保存"):
                self.engine.render_answer("I must save this audio.")

    def test_explicit_delete_removes_versions_across_models(self):
        text = "I can delete this answer."
        with patch(SYNTH, return_value=b"\x01\x00" * 20):
            self.engine.render_answer(text)
            self.change_voice()
            self.engine.render_answer_override(text)
        self.engine.delete_answer_audio(text)
        self.assertFalse(self.engine.answer_ready(text))


class BankDurationRetentionTests(unittest.TestCase):
    def setUp(self):
        # Real cache and engine, with the existing bank-dialog fixture.
        ui_fixtures.QaBankManageTests.setUp(self)

    def test_complete_callback_keeps_seconds_before_batch_finishes(self):
        entry = self.panel.bank[0]
        with patch(SYNTH, return_value=b"\x01\x00" * 24000):
            self.engine.render_answer(entry.answer)
        self.dialog._on_render_progress(entry.id, "✓ 已渲染")
        self.assertIn("1s", self.dialog.table.item(0, 4).text())
        self.settings.update({"prerender_tts_model": "cosyvoice-v3.5-plus", "prerender_voice_id": "other"})
        self.dialog.refresh()
        shown = self.dialog.table.item(0, 4).text()
        self.assertIn("1s", shown)
        self.assertIn("已渲染", shown)
        self.assertIn(self.engine.answer_audio_source(entry.answer)["model"], shown)
        self.dialog._on_render_progress(entry.id, "渲染中…")
        self.assertIn("1s", self.dialog.table.item(0, 4).text())


if __name__ == "__main__":
    unittest.main()
