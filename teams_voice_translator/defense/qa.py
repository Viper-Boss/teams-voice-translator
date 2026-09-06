from __future__ import annotations

import re
from typing import Any

from .memory import MeetingMemory
from .translator import ContextTranslator

NOT_A_QUESTION = "[非提问]"

_SYSTEM_PROMPT = """你是论文答辩现场候选人的智能助答助手。评委刚刚向候选人提出了问题，\
请根据论文背景、术语表和对话历史，给候选人一个可以当场口头回答的答案。

论文背景摘要：
{brief}

术语表（英文表达必须与之一致）：
{glossary}

要求：
1. 以答辩人第一人称口吻回答；正式学术口语；简洁自信，一般 2~4 句、不超过 80 个英文单词。
2. 严格基于论文背景与对话历史；没有依据的细节不要编造；确无依据时，给一个诚实的应对思路\
（例如承认当前局限并说明后续改进方向）。
3. 如果评委的最新发言不是提问（只是点评、过渡语或与候选人无关），只输出一行：ZH: {not_a_question}
4. 必须严格按以下两行格式输出，不要有任何其他内容：
EN: <可直接朗读的英文回答>
ZH: <对应的中文意思>"""


def build_qa_messages(memory: MeetingMemory, question: str) -> list[dict[str, str]]:
    system = _SYSTEM_PROMPT.format(
        brief=memory.brief or "（尚未导入答辩 PPT，回答时不要编造论文细节）",
        glossary=memory.glossary_block(),
        not_a_question=NOT_A_QUESTION,
    )
    user = "\n".join(
        [
            "===== 对话历史（最新在最后）=====",
            memory.render_history(),
            "===== 结束 =====",
            "",
            f"评委的最新发言：\n{question.strip()}",
        ]
    )
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
    ) -> None:
        self.translator = ContextTranslator(
            api_key=api_key,
            workspace_id=workspace_id,
            model=model or "qwen-plus",
            proxy=proxy,
            proxy_mode=proxy_mode,
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        return self.translator.model

    def build_messages(self, memory: MeetingMemory, question: str) -> list[dict[str, str]]:
        return build_qa_messages(memory, question)

    def generate(
        self,
        memory: MeetingMemory,
        question: str,
        *,
        on_delta: Any = None,
        cancel_event=None,
    ) -> tuple[str, str]:
        content = self.translator.chat(
            self.build_messages(memory, question),
            on_delta=on_delta,
            cancel_event=cancel_event,
        )
        return parse_answer(content)
