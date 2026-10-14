import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication
from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.memory import MeetingMemory
from teams_voice_translator.defense.qa import QaAdvisor, build_qa_messages
from teams_voice_translator.defense.qa_bank import QaEntry, MatchResult, match_question
from teams_voice_translator.defense.question_intent import (
    QuestionIntent, analyze_question, build_intent_messages, parse_intent,
)
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import QaPanel


def reply(*, complete=True, is_question=True):
    return json.dumps(dict(complete=complete, is_question=is_question,
                          intent="Explain the noise assumption and field validation.",
                          subquestions=["Why Gaussian noise?", "How was field validation performed?"],
                          reason="还有后半段" if not complete else "完整问题"))


class WholeQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.settings = DefenseSettings(Path(tmp.name))
        self.settings.values.update(qa_popup=False, qa_bank_enabled=False)
        self.engine = MagicMock()
        self.engine.memory = MeetingMemory()
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.generate = MagicMock()
        self.requests = []
        def chat(messages):
            self.requests.append(messages)
            return reply()
        self.panel.intent_chat_factory = lambda cancel: chat

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertTrue(predicate())

    def test_every_segment_goes_directly_to_intent_without_waiting_for_chinese(self):
        p = self.panel
        p.collect_committee_segment(1, "You assumed Gaussian noise. Why?")
        p.update_committee_translation(1, "故意错误的中文译文")
        p.collect_committee_segment(2, "And how did you validate it in the field?")
        p.generate.assert_not_called()
        p._auto_submit_collected_question()
        self.wait_for(lambda: p.generate.called)
        expected = "You assumed Gaussian noise. Why? And how did you validate it in the field?"
        self.assertEqual(p.generate.call_args.args[0], expected)
        data = json.loads(self.requests[0][1]["content"])
        self.assertEqual(data["complete_original_transcript"], expected)
        self.assertNotIn("故意错误", str(self.requests))
        self.assertEqual(len(p._question_intent.subquestions), 2)

    def test_incomplete_question_stays_collected_until_more_speech_arrives(self):
        p = self.panel
        received = []
        p.question_intent_ready.connect(received.append)
        p.intent_chat_factory = lambda cancel: lambda messages: reply(complete=False)
        p.collect_committee_segment(1, "I have two questions. First, why Gaussian noise?")
        p._auto_submit_collected_question()
        self.wait_for(lambda: bool(received))
        p.generate.assert_not_called()
        self.assertEqual(len(p._question_parts), 1)
        self.assertFalse(p._question_collect_timer.isActive(), "Do not repeatedly bill without new speech")
        p.intent_chat_factory = lambda cancel: lambda messages: reply()
        p.collect_committee_segment(2, "Second, how did you validate it?")
        p._auto_submit_collected_question()
        self.wait_for(lambda: p.generate.called)
        self.assertIn("First,", p.generate.call_args.args[0])
        self.assertIn("Second,", p.generate.call_args.args[0])

    def test_new_preview_invalidates_old_intent_result_and_waits_for_final(self):
        p = self.panel
        p.collect_committee_segment(1, "Why Gaussian noise?")
        old_revision = p._question_revision
        p.note_committee_activity("And how did you")
        p._on_question_intent_ready((old_revision, False, parse_intent(reply()), ""))
        p._auto_submit_collected_question()
        p.generate.assert_not_called()
        p.submit_collected_question()
        self.assertTrue(p._question_manual_pending)
        p.collect_committee_segment(2, "And how did you validate it?")
        self.wait_for(lambda: p.generate.called)
        self.assertIn("validate it", p.generate.call_args.args[0])

    def test_auto_off_manual_submit_still_generates_after_bank_miss(self):
        p = self.panel
        p.auto_check.setChecked(False)
        p.bank_check.setChecked(True)
        p.bank = [QaEntry(id="a", question="A different question?", answer="I have a different answer.")]
        p.match_chat_factory = lambda: lambda messages: '{"id":"NONE","confidence":0}'
        p.collect_committee_segment(1, "Why Gaussian noise?")
        p._auto_submit_collected_question()
        p.generate.assert_not_called()
        p.submit_collected_question()
        self.wait_for(lambda: p.generate.called)

    def test_network_failure_keeps_whole_original_and_shutdown_rejects_late_results(self):
        p = self.panel
        p.collect_committee_segment(1, "First question?")
        p.collect_committee_segment(2, "Second question?")
        p._on_question_intent_ready((p._question_revision, False, None, "offline"))
        p.generate.assert_called_once_with("First question? Second question?")
        p.generate.reset_mock()
        p.collect_committee_segment(3, "Next question?")
        revision = p._question_revision
        p.clear_collected_question()
        p._on_question_intent_ready((revision, False, parse_intent(reply()), ""))
        p.generate.assert_not_called()
        self.assertFalse(p._question_collect_timer.isActive())

    def test_nonquestion_does_not_generate_and_queued_old_answer_is_rejected(self):
        p = self.panel
        p.collect_committee_segment(1, "Thank you for presenting.")
        p._on_question_intent_ready((p._question_revision, False, parse_intent(reply(is_question=False)), ""))
        p.generate.assert_not_called()
        self.assertFalse(p._question_parts)
        old = threading.Event()
        p._on_qa_generation_ready((old, "Old answer", "旧回答"))
        self.assertEqual(p.answer_en.text(), "")

    def test_late_chinese_translation_updates_submitted_full_question(self):
        p = self.panel
        p.intent_check.setChecked(False)
        p.collect_committee_segment(1, "First question?")
        p.collect_committee_segment(2, "Second question?")
        p.submit_collected_question()
        p.update_committee_translation(2, "第二个问题？")
        p.update_committee_translation(1, "第一个问题？")
        self.assertEqual(p._question_zh, "第一个问题？\n第二个问题？")

    def test_bad_prepared_answer_is_not_presented_as_ready_for_reading(self):
        p = self.panel
        entry = QaEntry("bad", "Why?", "The candidate should explain the limitation.")
        p._on_qa_match((p._match_serial, "Why?", MatchResult(entry, .95, "llm")))
        p.generate.assert_called_once_with("Why?", references=[(entry.question, entry.answer)])
        self.assertEqual(p.answer_en.text(), "")


class IntentAndFirstPersonTests(unittest.TestCase):
    def test_invalid_or_missing_completion_cannot_silently_end_the_turn(self):
        for content in ("bad", "{}", '{"complete":"false","is_question":true,"intent":"why","subquestions":[]}'):
            with self.subTest(content=content), self.assertRaises(ApiError):
                parse_intent(content)

    def test_context_is_advisory_and_original_is_preserved_in_qa_and_match(self):
        original = "Why not Gaussian noise? How would non-Gaussian noise change the result?"
        messages = build_intent_messages(original, ["Earlier context"])
        self.assertEqual(json.loads(messages[1]["content"])["complete_original_transcript"], original)
        analyzed = analyze_question(original, lambda msgs: reply())
        messages = build_qa_messages(MeetingMemory(), original, intent=analyzed.context())
        self.assertIn(original, messages[1]["content"])
        self.assertIn("逐项回答", messages[1]["content"])
        self.assertIn("第一人称", messages[0]["content"])
        bank = [QaEntry(id="a", question="Why not Gaussian noise?", answer="I used a different noise model.")]
        seen = []
        match_question(bank, original, lambda messages: seen.extend(messages) or '{"id":"NONE","confidence":0}',
                       intent=analyzed.context())
        self.assertIn(original, seen[1]["content"])
        self.assertIn(bank[0].answer, seen[1]["content"])
        self.assertIn("ALL", seen[0]["content"])

    def test_multicomponent_offline_match_does_not_return_a_partial_answer(self):
        bank = [QaEntry(id="a", question="Why did you use Gaussian noise in the training data?", answer="I assumed Gaussian noise.")]
        result = match_question(bank, bank[0].question, None, allow_lexical_match=False)
        self.assertFalse(result.matched)
        self.assertTrue(result.candidates)

    def test_third_person_is_repaired_once_and_still_bad_output_is_rejected(self):
        advisor = QaAdvisor(api_key="test", workspace_id="test")
        advisor.translator.chat = MagicMock(side_effect=[
            "EN: The candidate should explain the limitation.\nZH: 建议用户说当前局限。",
            "EN: I have not yet validated this in the field.\nZH: 我还没有在现场验证。",
        ])
        result = advisor.generate(MeetingMemory(), "Have you validated it in the field?")
        self.assertTrue(result[0].startswith("I have"))
        self.assertEqual(advisor.translator.chat.call_count, 2)
        advisor.translator.chat = MagicMock(return_value="EN: The author believes it is useful.\nZH: 作者认为它有用。")
        with self.assertRaises(ApiError):
            advisor.generate(MeetingMemory(), "What did you find?")
        self.assertEqual(advisor.translator.chat.call_count, 2)

    def test_normal_first_person_answer_uses_only_one_request(self):
        advisor = QaAdvisor(api_key="test", workspace_id="test")
        advisor.translator.chat = MagicMock(return_value="EN: I used physical constraints.\nZH: 我采用物理约束。")
        self.assertTrue(advisor.generate(MeetingMemory(), "How?")[0].startswith("I used"))
        advisor.translator.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
