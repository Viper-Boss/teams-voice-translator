from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    """One bilingual utterance from either side of the defense."""

    role: str  # "me" or "committee"
    source: str  # 我=中文 / 评委=英文
    translation: str  # 我=英文 / 评委=中文
    latency_ms: int = 0
    at: float = field(default_factory=time.time)

    @property
    def speaker_label(self) -> str:
        return "我" if self.role == "me" else "评委"


class MeetingMemory:
    """Rolling record of the whole defense, injected into every translation.

    Translation is never sentence-by-sentence in isolation: every request
    carries the thesis brief, the mandatory glossary and the most recent
    bilingual turns so pronouns, topics and wording stay coherent.
    """

    def __init__(self, *, brief: str = "", glossary: list[dict[str, str]] | None = None, max_turns: int = 14) -> None:
        self._lock = threading.Lock()
        self.brief = brief.strip()
        self.glossary: list[dict[str, str]] = []
        self.max_turns = max(2, int(max_turns))
        self.turns: list[Turn] = []
        self.set_glossary(glossary)

    # ---------------------------------------------------------------- state
    def set_brief(self, brief: str) -> None:
        with self._lock:
            self.brief = brief.strip()

    def set_glossary(self, glossary: list[dict[str, str]]) -> None:
        cleaned: list[dict[str, str]] = []
        for item in glossary or []:
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if source and target:
                cleaned.append({"source": source, "target": target})
        with self._lock:
            self.glossary = cleaned

    def add_turn(self, turn: Turn) -> None:
        if not turn.source.strip() and not turn.translation.strip():
            return
        with self._lock:
            self.turns.append(turn)

    def clear_turns(self) -> None:
        with self._lock:
            self.turns.clear()

    def snapshot_turns(self) -> list[Turn]:
        with self._lock:
            return list(self.turns)

    def snapshot_glossary(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self.glossary)

    # ---------------------------------------------------------------- prompt
    def render_history(self, max_turns: int | None = None) -> str:
        limit = self.max_turns if max_turns is None else max_turns
        with self._lock:
            recent = self.turns[-limit:]
        if not recent:
            return "（对话刚开始，暂无历史）"
        lines: list[str] = []
        for turn in recent:
            if turn.role == "me":
                lines.append(f"[我] 中文：{turn.source}")
                lines.append(f"[我] 英文（你此前的译文）：{turn.translation}")
            else:
                lines.append(f"[评委] 英文：{turn.source}")
                if turn.translation:
                    lines.append(f"[评委] 中文（参考）：{turn.translation}")
        return "\n".join(lines)

    def glossary_block(self) -> str:
        terms = self.snapshot_glossary()
        if not terms:
            return "（暂无强制术语表）"
        return "\n".join(f"  {item['source']} → {item['target']}" for item in terms)

    def system_prompt(self, direction: str) -> str:
        if direction == "en2zh":
            return "\n".join(
                [
                    "你是论文答辩现场的实时译员。评委专家用英文提问和点评，"
                    "你负责把评委的最新英文发言翻译成中文，显示给答辩人看。",
                    "",
                    "论文背景摘要：",
                    self.brief or "（尚未导入答辩 PPT）",
                    "",
                    "术语表（中文译法必须保持一致）：",
                    self.glossary_block(),
                    "",
                    "翻译要求：",
                    "1. 中文自然流畅、简洁易懂，适合快速阅读。",
                    "2. 专业术语使用中文学术界的标准说法，并与术语表保持一致。",
                    "3. 忠实原意：不添加、不解释、不遗漏；数字与专有名词必须准确。",
                    "4. 只输出译文本身，不要引号、标注或任何解释。",
                ]
            )
        return "\n".join(
            [
                "你是硕士学位论文答辩现场的实时口译员。答辩人（学生）用中文发言，"
                "你负责把答辩人的最新一句中文精准翻译成英文，由语音系统当场说出。",
                "",
                "论文背景摘要：",
                self.brief or "（尚未导入答辩 PPT）",
                "",
                "术语表（这些术语的英文译法是强制的，必须逐字一致）：",
                self.glossary_block(),
                "",
                "翻译要求：",
                "1. 用答辩人本人的第一人称口吻。",
                "2. 自然的学术口语：像答辩人当面解释研究，清晰、直接、语法准确；避免书面腔和播报腔。用自然标点保留停顿、强调、疑问和转折，不添加情绪标签或拟声词。",
                "3. 忠实原文：不添加、不解释、不遗漏；数字、单位、专有名词必须准确；"
                "原文含糊或缺失的部分不要编造。",
                "4. 结合对话历史保持指代、话题和逻辑连贯；评委的提问也在历史中。",
                "5. 只输出译文本身，不要引号、标注或任何解释。",
            ]
        )

    def build_messages(self, direction: str, text: str, *, max_turns: int | None = None) -> list[dict[str, str]]:
        user = "\n".join(
            [
                "===== 对话历史（最新在最后）=====",
                self.render_history(max_turns),
                "===== 结束 =====",
                "",
                f"请翻译下面这一句：\n{text.strip()}",
            ]
        )
        return [
            {"role": "system", "content": self.system_prompt(direction)},
            {"role": "user", "content": user},
        ]
