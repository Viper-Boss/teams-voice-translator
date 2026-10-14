import unittest
from unittest.mock import Mock, patch

import test_qa_bank as fixtures
import test_quick_features as ui_fixtures

SYNTH = "teams_voice_translator.defense.tts_session.synthesize_once"


class QuickPlaybackRecoveryTests(unittest.TestCase):
    setUp = fixtures.EnginePreparedAnswerTests.setUp

    def test_history_replay_uses_prerender_audio_across_model_switch(self):
        text, pcm = "Let me organize my thoughts.", b"\x01\x00" * 100
        with patch(SYNTH, return_value=pcm) as synth:
            self.engine.render_answer(text)
            self.settings.update({"prerender_tts_model": "cosyvoice-v3.5-plus", "prerender_voice_id": "other"})
            self.engine.replay_english(text)
        self.assertEqual(synth.call_count, 1)
        self.engine._play_cached.assert_called_once()
        self.assertEqual(self.engine._play_cached.call_args.args[0].pcm, pcm)
        self.engine._queue_speech.assert_not_called()

    def test_uncached_phrase_recovers_paused_speech_instead_of_silently_queuing(self):
        self.engine.pause_speech()
        self.engine._queue_speech.side_effect = lambda text: self.assertFalse(self.engine.speech_paused)
        self.engine.speak_prepared("I need to check that result.", label="快捷用语")
        self.engine._queue_speech.assert_called_once()
        self.assertFalse(self.engine.speech_paused)

    def test_same_phrase_paused_is_resumed_even_when_duplicate_request_is_ignored(self):
        self.engine.speak_prepared("Let me think.")
        self.engine.pause_speech()
        self.engine.speak_prepared("Let me think.")
        self.assertFalse(self.engine.speech_paused)
        self.assertEqual(self.engine._queue_speech.call_count, 1)

    def test_cached_phrase_really_writes_nonzero_audio_with_live_monitor_off(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine
        text, pcm = "I will answer in two parts.", b"\x10\x00" * 1920
        with patch(SYNTH, return_value=pcm):
            self.engine.render_answer(text)
        self.settings.set("live_monitor_enabled", False)
        self.engine._play_cached = DefenseEngine._play_cached.__get__(self.engine)
        self.engine.player = Mock()
        self.engine._ensure_player = Mock()
        self.engine.pause_speech()
        self.engine.speak_prepared(text, label="快捷用语")
        self.engine._cached_thread.join(3)
        self.assertFalse(self.engine._cached_thread.is_alive())
        self.assertEqual(b"".join(c.args[0] for c in self.engine.player.write.call_args_list), pcm)
        self.assertTrue(all(not c.kwargs for c in self.engine.player.write.call_args_list),
                        "the F8/F9 mute switch must not mute prepared phrase monitoring")


class QuickPlaybackStatusTests(unittest.TestCase):
    setUpClass = classmethod(ui_fixtures.QuickPhrasePanelTests.setUpClass.__func__)
    setUp = ui_fixtures.QuickPhrasePanelTests.setUp

    def test_only_real_playback_callback_marks_phrase_as_started(self):
        phrase = self.panel.phrases[0]
        self.panel.speak(phrase)
        self.assertIn("等待发声", self.panel.status_label.text())
        self.panel.on_voice_started("Other audio.")
        self.assertIn("等待发声", self.panel.status_label.text())
        self.panel.on_voice_started(phrase.en)
        self.assertIn("已开始播放", self.panel.status_label.text())


if __name__ == "__main__":
    unittest.main()
