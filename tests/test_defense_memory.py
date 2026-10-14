from __future__ import annotations

import unittest

from teams_voice_translator.defense.memory import MeetingMemory, Turn


class MeetingMemoryTest(unittest.TestCase):
    def test_add_turns_and_history_window(self) -> None:
        memory = MeetingMemory(max_turns=3)
        for index in range(5):
            memory.add_turn(Turn(role="me", source=f"中文{index}", translation=f"English {index}"))
        history = memory.render_history()
        self.assertIn("中文4", history)
        self.assertNotIn("中文0", history)
        self.assertEqual(len(memory.turns), 5)

    def test_system_prompt_contains_brief_and_glossary(self) -> None:
        memory = MeetingMemory(
            brief="本文研究有限元缺陷反演。",
            glossary=[{"source": "有限元", "target": "finite element method"}, {"source": "", "target": "drop"}],
        )
        prompt = memory.system_prompt("zh2en")
        self.assertIn("有限元缺陷反演", prompt)
        self.assertIn("finite element method", prompt)
        self.assertNotIn("drop", prompt)
        zh_prompt = memory.system_prompt("en2zh")
        self.assertIn("中文", zh_prompt)

    def test_build_messages_marks_roles(self) -> None:
        memory = MeetingMemory()
        memory.add_turn(Turn(role="committee", source="What is your contribution?", translation="你的贡献是什么？"))
        memory.add_turn(Turn(role="me", source="主要贡献是新算法。", translation="The main contribution is a new algorithm."))
        messages = memory.build_messages("zh2en", "它比基线方法更快。")
        self.assertEqual(messages[0]["role"], "system")
        user = messages[1]["content"]
        self.assertIn("[评委]", user)
        self.assertIn("[我]", user)
        self.assertIn("它比基线方法更快。", user)

    def test_empty_turn_not_added(self) -> None:
        memory = MeetingMemory()
        memory.add_turn(Turn(role="me", source="", translation=""))
        self.assertEqual(len(memory.turns), 0)

    def test_clear_turns(self) -> None:
        memory = MeetingMemory()
        memory.add_turn(Turn(role="me", source="你好", translation="Hello"))
        memory.clear_turns()
        self.assertEqual(memory.turns, [])

    def test_optional_spoken_style_keeps_factual_constraints(self) -> None:
        memory = MeetingMemory()
        original = memory.system_prompt("zh2en")
        self.assertIn("去掉口头语和重复", original)
        memory.set_translation_style("spoken")
        spoken = memory.system_prompt("zh2en")
        self.assertIn("较短的分句", spoken)
        self.assertIn("数字、单位、专有名词必须准确", spoken)
        memory.set_translation_style("thoughtful")
        thoughtful = memory.system_prompt("zh2en")
        self.assertIn("偶尔", thoughtful)
        self.assertIn("不要故意说错数字、术语或结论", thoughtful)
        self.assertNotIn("Well 或 So", memory.system_prompt("en2zh"))
        memory.set_translation_style("invalid")
        self.assertIn("去掉口头语和重复", memory.system_prompt("zh2en"))


if __name__ == "__main__":
    unittest.main()
