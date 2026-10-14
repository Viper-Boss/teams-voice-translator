"""Single foreground voice, persistent questions, and popup generation. All offline."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
import unittest

import test_batch15 as playback_fixtures
import test_quick_features as popup_fixtures
from test_batch15 import _wait
from teams_voice_translator.defense.pipeline import CachedUtterance, DefenseEngine
from teams_voice_translator.defense.ui import DefenseWindow


class VoiceOwnershipTests(unittest.TestCase):
    _play = playback_fixtures.EnginePlaybackTests._play

    def setUp(self):
        # Borrow setup without inheriting the fixture's tests.
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        from pathlib import Path
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.pipeline import EngineCallbacks
        self.settings = DefenseSettings(Path(self.tmp.name))
        self.controls = []
        self.input_stops = []
        self.engine = DefenseEngine(self.settings, EngineCallbacks(
            on_speech_control=self.controls.append, on_input_stopped=lambda: self.input_stops.append(True)))
        self.addCleanup(self.engine.shutdown)
        self.engine._submit = Mock()
        self.player = Mock()
        self.player.write.side_effect = lambda data, **kwargs: time.sleep(.015)
        self.engine.player = self.player

    def test_answer_immediately_gates_direct_and_stops_input_buttons_callback(self):
        bridge = Mock()
        bridge.running = threading.Event()
        bridge.running.set()
        self.engine.bridge = bridge
        self.engine._direct_requested = True
        self.engine._direct_active = True
        self.engine.answer_ready = lambda text: True
        self.engine.speak_prepared("I used the same protocol.")
        self.assertFalse(bridge.running.is_set())
        self.assertFalse(self.engine._direct_requested)
        self.assertTrue(self.input_stops)
        self.assertFalse(self.engine._speech_paused.is_set())

    def test_cached_script_holds_for_f8_and_resumes_same_position(self):
        self._play(1.2, "script")
        self.assertTrue(_wait(lambda: self.engine.cached_playback_state()["played"] > .08))
        self.engine.start_direct()
        time.sleep(.08)
        position = self.engine.cached_playback_state()["played"]
        time.sleep(.12)
        self.assertEqual(self.engine.cached_playback_state()["played"], position)
        self.assertTrue(self.engine.cached_playback_state()["active"])
        self.engine.resume_speech()
        self.assertFalse(self.engine._direct_requested)
        self.assertTrue(_wait(lambda: self.engine.cached_playback_state()["played"] > position))
        self.assertIn("resume", self.controls)

    def test_f9_can_speak_while_old_script_stays_paused_until_continue(self):
        self._play(1.2, "script")
        self.assertTrue(_wait(lambda: self.engine.cached_playback_state()["played"] > .08))
        self.engine.set_listening(True)
        self.assertTrue(self.engine.speech_paused, "旧讲稿按钮必须保持暂停，F9 新译文仍能播放")
        time.sleep(.08)
        position = self.engine.cached_playback_state()["played"]
        self.player.reset_mock()
        self.engine._on_tts_audio(b"\x02\x00" * 100)
        self.player.write.assert_called_once()
        self.engine.set_listening(False)
        time.sleep(.1)
        self.assertEqual(self.engine.cached_playback_state()["played"], position)
        self.engine.resume_speech()
        self.assertTrue(_wait(lambda: self.engine.cached_playback_state()["played"] > position))

    def test_switch_to_explicit_playback_discards_mic_tail_but_not_committee(self):
        recognized = []
        self.engine.callbacks.on_committee_recognized = lambda sid, text: recognized.append(text)
        self.engine._listen_requested = True
        self.engine._claim_playback()
        self.engine._on_my_segment("late microphone sentence")
        self.engine._on_my_live_translation("late", "Late translation.")
        self.assertTrue(self.engine._my_queue.empty())
        self.engine._on_committee_segment("Why did you use this method?")
        self.assertEqual(recognized, ["Why did you use this method?"])
        self.assertFalse(self.engine._committee_queue.empty())

    def test_pending_f8_start_is_invalidated_by_later_playback(self):
        commands = []
        self.engine._submit = lambda action, desc: commands.append((action, desc))
        self.engine._do_start_direct = Mock()
        self.engine.start_direct()
        self.engine._claim_playback()
        commands[0][0]()
        self.engine._do_start_direct.assert_not_called()

    def test_new_uncached_answer_cancels_old_cached_voice(self):
        self._play(1.2, "old")
        self.assertTrue(_wait(lambda: self.engine.cached_playback_state()["played"] > .04))
        self.engine.answer_ready = lambda text: False
        self.engine.speak_prepared("A different answer without cached audio.")
        self.assertTrue(_wait(lambda: not self.engine.cached_playback_state()["active"]))

    def test_pending_script_request_is_cancelled_by_later_f8(self):
        commands = []
        self.engine._submit = lambda action, desc: commands.append(action)
        self.engine._play_cached = Mock()
        self.engine.play_script_segment(b"\x01\x00" * 100, text="script")
        script = commands[-1]
        self.engine.start_direct()
        script()
        self.engine._play_cached.assert_not_called()


class PopupGenerationTests(unittest.TestCase):
    setUpClass = popup_fixtures.AnswerPopupTests.__dict__["setUpClass"]
    setUp = popup_fixtures.AnswerPopupTests.setUp

    def test_manual_open_works_with_auto_popup_disabled_and_shared_editor(self):
        self.panel.popup_check.setChecked(False)
        self.panel.open_popup_button.click()
        self.assertTrue(self.panel.popup.isVisible())
        self.assertIs(self.panel.answer_split.parent(), self.panel.popup)
        self.panel.popup.hide()
        self.panel.open_popup_button.click()
        self.assertTrue(self.panel.popup.isVisible())

    def test_popup_generation_reuses_paper_references_and_first_person_advisor(self):
        from teams_voice_translator.defense.memory import MeetingMemory
        from teams_voice_translator.defense.qa import build_qa_messages

        self.engine.memory = MeetingMemory()
        self.panel.auto_check.setChecked(False)
        self.panel.on_committee_done("How did you choose the loss weights?", "如何选择权重？")
        advisor = Mock(model="qwen-test")
        advisor.generate.return_value = ("I used the same weights for every group.", "我对所有组使用相同的权重。")
        self.panel.advisor_factory = Mock(return_value=advisor)
        self.panel.study_notes = Mock()
        self.panel.study_notes.render.return_value = "Paper notes: fixed weights."
        self.panel.thesis_index = Mock()
        self.panel.thesis_index.search.return_value = [object()]
        self.panel.open_popup()
        popup = self.panel.popup
        popup.thinking_check.setChecked(True)
        with patch("teams_voice_translator.defense.ui.excerpt_pairs", return_value=[("page 4", "The weights were fixed.")]):
            popup.generate_button.click()
            self.assertTrue(_wait(lambda: self.panel.answer_en.text().startswith("I used")))
        kwargs = advisor.generate.call_args.kwargs
        self.assertTrue(kwargs["thinking"])
        self.assertEqual(kwargs["notes"], "Paper notes: fixed weights.")
        self.assertEqual(kwargs["excerpts"], [("page 4", "The weights were fixed.")])
        self.assertTrue(kwargs["references"])
        messages = build_qa_messages(MeetingMemory(), "Question?", **{k: kwargs[k] for k in ("notes", "excerpts", "references")})
        self.assertIn("第一人称", messages[0]["content"])
        self.engine.speak_prepared.assert_not_called()
        popup.play_button.click()
        self.assertEqual(self.engine.speak_prepared.call_args.args[0], "I used the same weights for every group.")

    def test_popup_generate_submits_all_pending_segments_bypassing_bank_match(self):
        self.panel.auto_check.setChecked(False)
        self.panel.intent_check.setChecked(False)
        self.panel.collect_committee_segment(1, "How were samples drawn?")
        self.panel.collect_committee_segment(2, "Could the validation set contain the same conditions?")
        self.panel.update_committee_translation(1, "样本如何抽取？")
        self.panel.open_popup_button.click()
        self.assertIn("validation", self.panel.popup.question_en.text())
        with patch.object(self.panel, "generate") as generate, patch.object(self.panel, "match_then_answer") as match:
            self.panel.popup.generate_button.click()
        self.assertEqual(generate.call_args.args[0], "How were samples drawn? Could the validation set contain the same conditions?")
        match.assert_not_called()


class WindowFlowTests(unittest.TestCase):
    def setUp(self):
        from test_defense_ui import DefenseUiTest
        DefenseUiTest.setUpClass()
        self.window = DefenseUiTest.window
        self.addCleanup(DefenseUiTest.tearDownClass)
        self.window.qa_panel.auto_check.setChecked(False)
        self.window.engine._submit = Mock()

    def test_f8_continuous_survives_script_pause_and_release_then_toggles_off(self):
        w = self.window
        w.continuous_check.setChecked(True)
        w.script_panel._playing = [object()]
        w.script_panel._paused_at = 0
        with patch.object(w.script_panel, "pause") as pause:
            w._input_press("direct", "mouse")
        pause.assert_called_once()
        w._input_release("direct", "mouse")
        self.assertEqual(w._input_mode, "direct")
        w._input_press("direct", "keyboard")
        self.assertIsNone(w._input_mode)
        w.script_panel._playing = []

    def test_f8_hold_still_stops_on_release_when_continuous_unchecked(self):
        w = self.window
        w.continuous_check.setChecked(False)
        w._input_press("direct", "mouse")
        self.assertEqual(w._input_mode, "direct")
        w._input_release("direct", "mouse")
        self.assertIsNone(w._input_mode)

    def test_playback_clears_buttons_held_inputs_and_late_state_cannot_relatch(self):
        w = self.window
        w.continuous_check.setChecked(True)
        w._input_press("direct", "keyboard")
        w.engine.resume_speech()
        self.assertIsNone(w._input_mode)
        self.assertFalse(w.direct_button.isChecked())
        self.assertFalse(w.translate_button.isChecked())
        self.assertEqual(w._held_inputs, {})
        w._on_state_changed("direct")
        self.assertFalse(w.direct_button.isChecked())
        self.assertTrue(w.continuous_check.isChecked())

    def test_question_original_visible_before_translation_then_updated_without_duplicates(self):
        w = self.window
        w._on_committee_recognized(700, "How did you verify independent sampling?")
        self.assertIn("independent sampling", w.committee_history.toPlainText())
        self.assertIn("正在翻译", w.committee_history.toPlainText())
        w._on_committee_recognized(701, "How did you avoid leakage?")
        w._on_committee_done(700, "How did you verify independent sampling?", "如何验证独立采样？")
        w._on_committee_done(701, "How did you avoid leakage?", "如何避免泄漏？")
        text = w.committee_history.toPlainText()
        self.assertEqual(text.count("independent sampling"), 1)
        self.assertIn("独立采样", text)
        self.assertIn("avoid leakage", text)
        self.assertNotIn("正在翻译", text)


class DirectOutputLockTests(unittest.TestCase):
    def test_direct_writer_rechecks_gate_after_acquiring_shared_output_lock(self):
        from teams_voice_translator.audio import DirectAudioBridge
        player = MagicMock()
        lock = threading.RLock()
        lock.acquire()
        bridge = DirectAudioBridge()
        with patch("teams_voice_translator.audio.MultiOutputPlayer", return_value=player), \
                patch("teams_voice_translator.audio.MicrophoneCapture") as capture:
            try:
                bridge.start(1, 7, 48000, output_lock=lock)
                capture.call_args.args[1](b"\x01\x00" * 50)
                time.sleep(.06)
                player.write.assert_not_called()
                bridge.running.clear()
                lock.release()
                bridge.thread.join(timeout=1)
                self.assertFalse(bridge.thread.is_alive())
                player.write.assert_not_called()
            finally:
                bridge.stop()
