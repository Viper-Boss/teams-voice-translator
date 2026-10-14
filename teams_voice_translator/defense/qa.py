from __future__ import annotations

import re
from typing import Any

from .memory import MeetingMemory
from .translator import ContextTranslator
from ..aliyun import ApiError

NOT_A_QUESTION = "[非提问]"

_SYSTEM_PROMPT = """你就是正在参加论文答辩的学生本人，正在直接面对评委回答。
根据论文背景、术语表和完整提问作答；输出的是你亲口说的话，不是给学生的指导建议。

论文背景摘要：
{brief}
{notes}
术语表（英文表达必须与之一致）：
{glossary}

要求：
1. EN 和 ZH 都以我/I/my 的第一人称直接回答；只有确指团队工作才用 we/our。
   禁止用“用户/学生/作者/答辩者应该……”或“The candidate should / You can say / The author believes”等第三人称旁白。
   不写“建议回答”、角色标签、指导性括号或舞台提示。简洁的正式学术口语，一般 2~4 句；多部分问题可适当延长以完整回答。
2. 严格基于论文背景、论文理解、论文原文摘录与对话历史，数字以原文摘录和论文理解为准；没有依据的细节不要编造；确无依据时，给一个诚实的应对思路\
（例如承认当前局限并说明后续改进方向）。
3. 如果评委的最新发言不是提问（只是点评、过渡语或与候选人无关），只输出一行：ZH: {not_a_question}
4. 必须严格按以下两行格式输出，不要有任何其他内容：
EN: <可直接朗读的英文回答>
ZH: <对应的中文意思>
5. 最新发言可能是连续多句的完整问题。先理解前文背景、限定条件、转折和所有子问题，再逐项回答；
   不要只回答最后一句。保留评委的否定、比较对象和要求解释的原因；不能确认意图时由我第一人称请求澄清。"""


NON_FIRST_PERSON = re.compile(
    r'\b(?:the (?:candidate|student|user) (?:should|could|can|needs|believes|thinks|uses|proposes)|'
    r'the author (?:should|believes|thinks|uses|proposes)|'
    r'you (?:can|should|could) (?:say|answer|respond|explain)|suggested (?:answer|response))\b|'
    r'(?:建议|推荐)(?:用户|学生|答辩者|答辩人)(?:说|回答)|'
    r'(?:作者|学生|答辩者|答辩人)(?:认为|应该|应当|需要|可以回答|采用)|你可以这样回答', re.I)


_NOTES_BLOCK = """
答辩人事先预习、亲自校对过的论文理解（事实和数字以此为准）：
{notes}
"""

_EXCERPT_HEADER = "===== 论文原文摘录（按与问题的相关度排列，括号内是出处）====="

_REFERENCE_HEADER = (
    "===== 候选人事先准备的相关问答（与当前问题未必相同，只作事实和措辞参考；"
    "其中的数字和结论优先于你自己的推断，与当前问题无关的部分不要照搬）====="
)


def build_qa_messages(
    memory: MeetingMemory,
    question: str,
    references: list[tuple[str, str]] | None = None,
    notes: str = "",
    excerpts: list[tuple[str, str]] | None = None,
    intent: str = "",
) -> list[dict[str, str]]:
    system = _SYSTEM_PROMPT.format(
        brief=memory.brief or "（尚未导入答辩 PPT，回答时不要编造论文细节）",
        notes=_NOTES_BLOCK.format(notes=notes.strip()) if notes.strip() else "",
        glossary=memory.glossary_block(question),
        not_a_question=NOT_A_QUESTION,
    )
    lines = [
        "===== 对话历史（最新在最后）=====",
        memory.render_history(),
        "===== 结束 =====",
        "",
    ]
    if excerpts:
        lines.append(_EXCERPT_HEADER)
        for label, text in excerpts:
            lines.extend([f"（{label}）{text}", ""])
        lines.append("===== 结束 =====")
        lines.append("")
    if references:
        # 问答库没命中时，最相关的几条预案仍然是最可靠的事实来源。放在用户
        # 消息里而不是系统提示末尾，免得冲淡系统提示最后那条输出格式要求。
        lines.append(_REFERENCE_HEADER)
        for q, a in references:
            lines.extend([f"Q: {q}", f"A: {a}", ""])
        lines.append("===== 结束 =====")
        lines.append("")
    lines.append(f"评委的最新发言：\n{question.strip()}")
    if intent:
        lines.append("整题意图与子问题（仅帮助理解，以评委原话为准，逐项回答，不朗读这些标签）：\n" + intent)
    user = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_answer(content: str) -> tuple[str, str]:
    """Split a model reply into (english, chinese).

    Returns ``("", "")`` when the model judged the utterance not to be a
    question.  Free-form replies without markers are treated as the English
    answer so the feature degrades gracefully instead of failing.
    """
    text = (content or "").strip()
    if not text:
        return "", ""
    en_match = re.search(r"EN\s*[:：]\s*(.+?)(?=\n\s*ZH\s*[:：]|\s*ZH\s*[:：]|$)", text, re.DOTALL | re.IGNORECASE)
    zh_match = re.search(r"ZH\s*[:：]\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    english = en_match.group(1).strip() if en_match else ""
    chinese = zh_match.group(1).strip() if zh_match else ""
    if NOT_A_QUESTION in text and not english:
        return "", ""
    if NOT_A_QUESTION in chinese and len(chinese) <= len(NOT_A_QUESTION) + 2:
        return "", ""
    if not english and not chinese:
        return text, ""
    if not english:
        english = chinese
    return english, chinese


class QaAdvisor:
    """Generates speakable first-person answers to committee questions."""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str = "qwen-plus",
        proxy: str = "",
        proxy_mode: str = "",
        timeout: int = 60,
        backup=None,
    ) -> None:
        self.translator = ContextTranslator(
            api_key=api_key,
            workspace_id=workspace_id,
            model=model or "qwen-plus",
            proxy=proxy,
            proxy_mode=proxy_mode,
            timeout=timeout,
            backup=backup,
        )

    @property
    def model(self) -> str:
        return self.translator.model

    def build_messages(
        self, memory: MeetingMemory, question: str, references: list[tuple[str, str]] | None = None,
        notes: str = "", excerpts: list[tuple[str, str]] | None = None,
        intent: str = "",
    ) -> list[dict[str, str]]:
        return build_qa_messages(memory, question, references, notes=notes, excerpts=excerpts, intent=intent)

    def generate(
        self,
        memory: MeetingMemory,
        question: str,
        *,
        on_delta: Any = None,
        cancel_event=None,
        thinking: bool = False,
        on_reasoning: Any = None,
        references: list[tuple[str, str]] | None = None,
        notes: str = "",
        excerpts: list[tuple[str, str]] | None = None,
        intent: str = "",
    ) -> tuple[str, str]:
        """thinking=True 时答得更准但更慢：评委还在说话时就开始生成，
        多等几秒通常能接受；赶时间就关掉。"""
        messages = self.build_messages(memory, question, references, notes=notes, excerpts=excerpts, intent=intent)
        content = self.translator.chat(
            messages,
            on_delta=on_delta,
            cancel_event=cancel_event,
            thinking=thinking,
            on_reasoning=on_reasoning,
        )
        answer = parse_answer(content)
        if NON_FIRST_PERSON.search("\n".join(answer)):
            corrected = self.translator.chat(
                messages + [{"role": "assistant", "content": content}, {"role": "user", "content":
                    "刚才含有第三人称旁白或给学生的建议。请改成我本人面对评委直接说出的第一人称回答。"
                    "保留事实、数字、局限和所有子问题，不能增加研究结果。只输出 EN: 和 ZH: 两行。"}],
                cancel_event=cancel_event, thinking=False)
            answer = parse_answer(corrected)
            if NON_FIRST_PERSON.search("\n".join(answer)):
                raise ApiError("回答仍含第三人称旁白，请重新生成或修改后朗读")
        return answer
