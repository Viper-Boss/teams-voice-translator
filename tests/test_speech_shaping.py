"""句间换气 + 开场衔接词：送给 TTS 的文本和给人看的文本必须能分开。

最容易出的事故是二者混用：悬浮字幕上冒出 <speak>、黄色进度对不上、
重播缓存按带标记的串做键。这里把这条边界钉死。
"""
from __future__ import annotations

import random
import unittest
from unittest.mock import MagicMock

from teams_voice_translator.defense.pipeline import DefenseEngine
from teams_voice_translator.defense.speech_policy import (
    FillerPicker,
    OPENING_FILLERS_EN,
    apply_opening_filler,
    join_sentences_ssml,
)

SUPPORTS_SSML = "qwen-audio-3.0-tts-plus"
NO_SSML = "qwen3-tts-vc-realtime-2026-01-15"


class _FakeEngine:
    """只装配 _prepare_utterances 真正用到的东西，不启动任何线程或声卡。"""

    def __init__(self, values):
        self.settings = MagicMock()
        self.settings.get.side_effect = lambda key, default=None: values.get(key, default)
        self._filler_picker = FillerPicker(rng=random.Random(0))
        self._spoken_display = {}
        self._speech_jobs = {}
        self._last_humanized_at = 0.0

    # 按真实名字借用，因为这些方法内部会互相调用 self._xxx。
    _prepare_utterances = DefenseEngine._prepare_utterances
    _sentence_break_ms = DefenseEngine._sentence_break_ms
    _split_sentences = staticmethod(DefenseEngine._split_sentences)
    _register_utterances = DefenseEngine._register_utterances
    _unregister_utterances = DefenseEngine._unregister_utterances
    _display_for = DefenseEngine._display_for
    _humanize_live = DefenseEngine._humanize_live
    humanize_level = DefenseEngine.humanize_level
    HUMANIZE_COOLDOWN = DefenseEngine.HUMANIZE_COOLDOWN
    speech_features = DefenseEngine.speech_features

    # 测试里用的短别名
    prepare = _prepare_utterances
    break_ms = _sentence_break_ms
    register = _register_utterances
    unregister = _unregister_utterances
    display_for = _display_for
    features = speech_features


def engine(**values):
    values.setdefault("tts_model", SUPPORTS_SSML)
    values.setdefault("sentence_break_ms", 0)
    values.setdefault("opening_filler", "off")
    return _FakeEngine(values)


class FillerTests(unittest.TestCase):
    def test_filler_is_prepended_to_the_first_chunk_only(self):
        self.assertEqual(apply_opening_filler("Body text.", "Sure."), "Sure. Body text.")

    def test_empty_filler_leaves_text_untouched(self):
        self.assertEqual(apply_opening_filler("Body.", ""), "Body.")
        self.assertEqual(apply_opening_filler("  Body.  ", "   "), "Body.")

    def test_empty_body_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_opening_filler("   ", "Sure.")

    def test_picker_never_repeats_immediately(self):
        picker = FillerPicker(rng=random.Random(7))
        previous = ""
        for _ in range(60):
            current = picker.pick()
            self.assertIn(current, OPENING_FILLERS_EN)
            self.assertNotEqual(current, previous)
            previous = current

    def test_picker_with_single_choice_degrades_instead_of_hanging(self):
        picker = FillerPicker(choices=("Sure.",))
        self.assertEqual(picker.pick(), "Sure.")
        self.assertEqual(picker.pick(), "Sure.")

    def test_picker_rejects_empty_pool(self):
        with self.assertRaises(ValueError):
            FillerPicker(choices=())

    def test_reset_clears_the_repeat_guard(self):
        picker = FillerPicker(rng=random.Random(1))
        first = picker.pick()
        picker.reset()
        # reset 后第一个可以和 reset 前相同；只要不抛异常、仍在池子里即可。
        self.assertIn(picker.pick(), OPENING_FILLERS_EN)
        self.assertIn(first, OPENING_FILLERS_EN)


class BreakGateTests(unittest.TestCase):
    def test_unsupported_model_forces_zero(self):
        self.assertEqual(engine(tts_model=NO_SSML, sentence_break_ms=160).break_ms(), 0)

    def test_supported_model_passes_value_through(self):
        self.assertEqual(engine(sentence_break_ms=160).break_ms(), 160)

    def test_value_is_clamped_and_garbage_falls_back_to_zero(self):
        self.assertEqual(engine(sentence_break_ms=99999).break_ms(), 1000)
        self.assertEqual(engine(sentence_break_ms=-5).break_ms(), 0)
        self.assertEqual(engine(sentence_break_ms="abc").break_ms(), 0)
        self.assertEqual(engine(sentence_break_ms=None).break_ms(), 0)


class PrepareUtteranceTests(unittest.TestCase):
    def test_everything_off_is_a_passthrough(self):
        prepared = engine().prepare("First point. Second point.")
        self.assertEqual(len(prepared), 1)
        spoken, display = prepared[0]
        self.assertEqual(spoken, display)
        self.assertNotIn("<speak>", spoken)

    def test_break_wraps_spoken_text_but_leaves_display_plain(self):
        prepared = engine(sentence_break_ms=160).prepare("First point. Second point.")
        spoken, display = prepared[0]
        self.assertIn('<break time="160ms"/>', spoken)
        self.assertTrue(spoken.startswith("<speak>"))
        self.assertNotIn("<speak>", display)
        self.assertEqual(display, "First point. Second point.")

    def test_single_sentence_is_still_wrapped_when_break_is_on(self):
        """SSML 模式一旦打开，每一次请求都必须是合法 SSML，不能混着发纯文本。"""
        spoken, _display = engine(sentence_break_ms=160).prepare("Only one.")[0]
        self.assertTrue(spoken.startswith("<speak>"))

    def test_unsupported_model_never_emits_markup(self):
        spoken, display = engine(
            tts_model=NO_SSML, sentence_break_ms=160
        ).prepare("First. Second.")[0]
        self.assertEqual(spoken, display)
        self.assertNotIn("<speak>", spoken)

    def test_filler_appears_in_both_spoken_and_display(self):
        """衔接词是真的会被说出来的，所以字幕必须照实显示，不能偷偷藏起来。"""
        spoken, display = engine(opening_filler="auto").prepare("Body text.")[0]
        self.assertEqual(spoken, display)
        self.assertTrue(any(display.startswith(f) for f in OPENING_FILLERS_EN), display)
        self.assertTrue(display.endswith("Body text."))

    def test_filler_only_leads_the_first_chunk(self):
        long_text = ". ".join(f"Sentence number {index} here" for index in range(60)) + "."
        prepared = engine(opening_filler="auto").prepare(long_text)
        self.assertGreater(len(prepared), 1, "测试文本应当超过分块上限")
        leads = [any(d.startswith(f) for f in OPENING_FILLERS_EN) for _s, d in prepared]
        self.assertTrue(leads[0])
        self.assertFalse(any(leads[1:]))

    def test_filler_and_break_compose(self):
        spoken, display = engine(
            opening_filler="auto", sentence_break_ms=200
        ).prepare("First. Second.")[0]
        self.assertTrue(spoken.startswith("<speak>"))
        self.assertIn('<break time="200ms"/>', spoken)
        self.assertTrue(any(display.startswith(f) for f in OPENING_FILLERS_EN))
        self.assertNotIn("<speak>", display)

    def test_empty_text_is_rejected(self):
        from teams_voice_translator.aliyun import ApiError

        with self.assertRaises((ApiError, ValueError)):
            engine().prepare("   ")


class RegistrationTests(unittest.TestCase):
    def test_display_lookup_round_trips_and_cleans_up(self):
        eng = engine(sentence_break_ms=160)
        prepared = eng.prepare("First. Second.")
        spoken_texts = eng.register(prepared, None)
        spoken = spoken_texts[0]
        self.assertEqual(eng.display_for(spoken), prepared[0][1])
        eng.unregister(spoken_texts)
        # 清理后查不到映射，退化成返回自身，不会抛。
        self.assertEqual(eng.display_for(spoken), spoken)

    def test_passthrough_text_is_not_stored(self):
        """送出文本和显示文本相同时不占映射表，避免无谓增长。"""
        eng = engine()
        eng.register(eng.prepare("Plain text."), None)
        self.assertEqual(eng._spoken_display, {})

    def test_jobs_are_keyed_by_spoken_text(self):
        eng = engine(sentence_break_ms=160)
        job = object()
        spoken_texts = eng.register(eng.prepare("First. Second."), job)
        self.assertEqual(eng._speech_jobs[spoken_texts[0]], job)
        eng.unregister(spoken_texts)
        self.assertEqual(eng._speech_jobs, {})


class CapabilityReportTests(unittest.TestCase):
    def test_recommended_model_reports_everything(self):
        features = engine().features()
        self.assertTrue(features["instruction"])
        self.assertTrue(features["ssml"])
        self.assertTrue(features["prefetch"])
        self.assertTrue(features["clone"])

    def test_realtime_vc_model_reports_what_it_lacks(self):
        features = engine(tts_model=NO_SSML).features()
        self.assertFalse(features["instruction"])
        self.assertFalse(features["ssml"])
        self.assertFalse(features["prefetch"], "实时 WebSocket 会话是单连接串行的")
        self.assertTrue(features["clone"])

    def test_unknown_model_reports_nothing_supported(self):
        features = engine(tts_model="made-up").features()
        self.assertFalse(features["instruction"])
        self.assertFalse(features["ssml"])
        self.assertFalse(features["clone"])


class SsmlEscapingTests(unittest.TestCase):
    def test_wrap_single_is_off_by_default(self):
        self.assertEqual(join_sentences_ssml(["One."], SUPPORTS_SSML), "One.")

    def test_terminology_with_angle_brackets_is_escaped(self):
        spoken, _display = engine(sentence_break_ms=160).prepare(
            "We use <EOS> tokens. It works."
        )[0]
        self.assertNotIn("<EOS>", spoken)
        self.assertIn("&lt;EOS&gt;", spoken)


if __name__ == "__main__":
    unittest.main()
