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


# 「我的英文口吻」：F9 现场翻译的系统提示第 5 条 + 讲稿台「中文稿翻成英文」的风格提示。
# (id, 界面名, 现场翻译规则, 讲稿翻译提示)。所有风格都必须：不改数字、术语、结论，不编造内容。
TRANSLATION_STYLES: tuple[tuple[str, str, str, str], ...] = (
    ("accurate", "准确自然（原有译法）",
     "5. 原文来自实时语音识别，可能有同音错字、术语识别错误、断句不当或口头语（嗯、那个、就是说）：先结合论文背景、术语表和上下文还原答辩人的本意，去掉口头语和重复，再翻译；只纠正识别错误，不改变意思、不补充内容。",
     ""),
    ("spoken", "更像口语（短句、自然转折）",
     "5. 译成现场说得出口的英文：适度用常见缩写和较短的分句，保留原文有意义的停顿、自我修正与犹豫。识别造成的同音错字、术语错误仍要纠正；删掉无意义的重复，不要把句子润色成书面论文。",
     "用现场说得出口的英文：较短的分句、常见缩写、自然转折，不要写成书面论文。"),
    ("thoughtful", "轻微思考（偶尔犹豫，不改术语和数字）",
     "5. 像学生现场边想边回答：用较短的分句和自然转折，偶尔可有一个简短的 Well 或 So 作为思考衔接；不要每句都加，不要机械重复。偶尔可在非关键措辞上短暂重启并自我重述，例如用 'what I mean is' 换个说法；原文已有自我修正时可保留。不要故意说错数字、术语或结论，也不要凭空编造内容或观点。识别错误仍须纠正。",
     "像学生现场边想边讲：较短分句、自然转折，偶尔（不是每句）用一个 Well 或 So 衔接；不要改动数字、术语和结论。"),
    ("confident", "自信简洁（结论先行、短句）",
     "5. 语气自信、干脆：先说结论再补理由，用短句和主动语态，少用 maybe / kind of 这类含糊词和多余修饰；但原文本身不确定的地方要照实保留，不要把推测说成定论。识别错误仍须纠正。",
     "语气自信干脆：结论先行，短句、主动语态，少用含糊词；原文不确定的地方照实保留。"),
    ("humble", "谦逊礼貌（面对评委客气）",
     "5. 语气谦逊礼貌、尊重评委：适当用 I think / I believe / in my understanding 表达观点，回应建议时自然地表示感谢或认同；不要每句都加客套话，不要过度道歉，不改变原意。识别错误仍须纠正。",
     "语气谦逊礼貌：适当用 I think / I believe 表达观点，不过度客套，不改变原意。"),
    ("simple", "简单好读（常用词、短句，英语不熟也好念）",
     "5. 用最常见的英文词和简单句式：一句只讲一件事，避免长从句、生僻词和复杂时态，让答辩人自己也容易读懂、念顺；专业术语仍按术语表，不用简单词替换术语。识别错误仍须纠正。",
     "用最常见的词和简单句式，一句只讲一件事，避免长从句和生僻词；专业术语仍按术语表。"),
    ("academic", "学术严谨（用词精确、稍正式）",
     "5. 用严谨的学术英语：用词精确、逻辑连接清楚（therefore / however / in contrast），可以稍正式，但仍是能说出口的口头表达，不要写成论文长句。识别错误仍须纠正。",
     "学术严谨：用词精确、逻辑连接清楚，稍正式但仍适合朗读，不要写成论文长句。"),
    ("explain", "讲解式（像给评委讲清楚，带层次）",
     "5. 像在给评委讲解：原文有多个要点时用 first / then / finally、the key point is 这类衔接把层次讲清楚；不增加原文没有的内容，不为了层次硬拆句子。识别错误仍须纠正。",
     "像给评委讲解：多个要点时用 first / then / finally 等衔接把层次讲清楚，不增加原文没有的内容。"),
)
STYLE_IDS = frozenset(item[0] for item in TRANSLATION_STYLES)
STYLE_RULES = {item[0]: item[2] for item in TRANSLATION_STYLES}
STYLE_DESK_HINTS = {item[0]: item[3] for item in TRANSLATION_STYLES}


class MeetingMemory:
    """Rolling record of the whole defense, injected into every translation.

    Translation is never sentence-by-sentence in isolation: every request
    carries the thesis brief, the mandatory glossary and the most recent
    bilingual turns so pronouns, topics and wording stay coherent.
    """

    def __init__(self, *, brief: str = "", glossary: list[dict[str, str]] | None = None,
                 max_turns: int = 14, translation_style: str = "accurate") -> None:
        self._lock = threading.Lock()
        self.brief = brief.strip()
        self.glossary: list[dict[str, str]] = []
        self.max_turns = max(2, int(max_turns))
        self.translation_style = translation_style if translation_style in STYLE_IDS else "accurate"
        self.turns: list[Turn] = []
        self.set_glossary(glossary)

    # ---------------------------------------------------------------- state
    def set_brief(self, brief: str) -> None:
        with self._lock:
            self.brief = brief.strip()

    def set_translation_style(self, style: str) -> None:
        with self._lock:
            self.translation_style = style if style in STYLE_IDS else "accurate"

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

    GLOSSARY_ALWAYS = 60      # 术语很多时：前 60 条（最重要的）每次都带
    GLOSSARY_MAX = 160        # 每次最多带这么多条，再多只会拖慢首字延迟

    def glossary_block(self, text: str = "") -> str:
        """术语表放进提示词。条数不多时全带；自动生成出几百条时，只带前 60 条 +
        这句话和最近几轮里真的出现了的术语（总数封顶），避免每句翻译都背一大段术语。"""
        terms = self.snapshot_glossary()
        if not terms:
            return "（暂无强制术语表）"
        if len(terms) > self.GLOSSARY_MAX:
            recent = " ".join(f"{t.source} {t.translation}" for t in self.snapshot_turns()[-4:])
            hay = f"{text} {recent}".casefold()
            picked = terms[:self.GLOSSARY_ALWAYS]
            for item in terms[self.GLOSSARY_ALWAYS:]:
                if len(picked) >= self.GLOSSARY_MAX:
                    break
                if item["source"].casefold() in hay or item["target"].casefold() in hay:
                    picked.append(item)
            terms = picked
        return "\n".join(f"  {item['source']} → {item['target']}" for item in terms)

    def system_prompt(self, direction: str, text: str = "") -> str:
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
                    self.glossary_block(text),
                    "",
                    "翻译要求：",
                    "1. 中文自然流畅、简洁易懂，适合快速阅读。",
                    "2. 专业术语使用中文学术界的标准说法，并与术语表保持一致。",
                    "3. 忠实原意：不添加、不解释、不遗漏；数字与专有名词必须准确。",
                    "4. 原文来自实时语音识别，可能有近音词或术语识别错误：结合论文背景、术语表和上下文按本意理解后再翻译。",
                    "5. 只输出译文本身，不要引号、标注或任何解释。",
                ]
            )
        with self._lock:
            style = self.translation_style
        style_rule = STYLE_RULES.get(style, STYLE_RULES["accurate"])
        return "\n".join(
            [
                "你是硕士学位论文答辩现场的实时口译员。答辩人（学生）用中文发言，"
                "你负责把答辩人的最新一句中文精准翻译成英文，由语音系统当场说出。",
                "",
                "论文背景摘要：",
                self.brief or "（尚未导入答辩 PPT）",
                "",
                "术语表（这些术语的英文译法是强制的，必须逐字一致）：",
                self.glossary_block(text),
                "",
                "翻译要求：",
                "1. 用答辩人本人的第一人称口吻。",
                "2. 自然的学术口语：像答辩人当面解释研究，清晰、直接、语法准确；避免书面腔和播报腔。用自然标点保留停顿、强调、疑问和转折，不添加情绪标签或拟声词。",
                "3. 忠实原文：不添加、不解释、不遗漏；数字、单位、专有名词必须准确；"
                "原文含糊或缺失的部分不要编造。",
                "4. 结合对话历史保持指代、话题和逻辑连贯；评委的提问也在历史中。",
                style_rule,
                "6. 只输出译文本身，不要引号、标注或任何解释。",
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
            {"role": "system", "content": self.system_prompt(direction, text)},
            {"role": "user", "content": user},
        ]
