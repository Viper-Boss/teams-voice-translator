"""本地问答库：导入、检索、匹配评委提问。

设计要点：

* **先匹配、后生成。** 评委提问时先在用户提前准备的问答里找「问的是同一件事」
  的条目；找到就直接给出准备好的答案（可预渲染成本地音频，点一下秒播），
  找不到才交给大模型现场生成。
* **两级匹配。** 本地词面检索（离线、瞬时）给候选排序（库很大时先预筛），
  再让快速大模型判断「是不是同一个问题」。词面检索抗不住换说法和语音识别错字，
  大模型抗得住；大模型断网时退回词面检索，只接受高分命中。
* **宁可不命中，也不要错命中。** 错命中会让你当场念一段答非所问的话，
  比现场生成慢几秒糟糕得多。所以「相关话题」不算命中，阈值偏保守。
* 问答库是真实答辩内容，只存在本机配置目录，不进仓库、不写日志。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

BANK_FILE = "qa_bank.json"
BANK_VERSION = 1

# 匹配严格度：(大模型置信度门槛, 断网时词面分门槛)。适中 = 原来的默认值。
STRICTNESS: dict[str, tuple[float, float]] = {
    "strict": (0.75, 0.65),
    "medium": (0.6, 0.55),
    "loose": (0.45, 0.45),
}
STRICTNESS_CHOICES: tuple[tuple[str, str], ...] = (
    ("strict", "严格"),
    ("medium", "★ 适中（默认）"),
    ("loose", "宽松"),
)


def strictness_thresholds(level: str) -> tuple[float, float]:
    return STRICTNESS.get(str(level or "medium"), STRICTNESS["medium"])


# 大模型给出的置信度低于这个值就当没命中。0.6 偏保守：宁可多走一次现场生成。
MIN_LLM_CONFIDENCE = 0.6
# 大模型不可用（断网、没配 Key）时，词面分数高于这个值才敢直接用。
# 这个分数是加权 Dice（0~1），0.55 大致对应「问题里的关键词大半都对上了」。
LEXICAL_STRONG = 0.55
# 同时要求问题本身至少有这么多个有效词：「weights?」这种一个词的碎片
# 和短条目的 Dice 分可能很高，但凭它离线直接弹答案太冒险。
MIN_LEXICAL_TOKENS = 3
# 送给大模型挑选的候选数。库不超过这个数就整库送过去——换了说法的问题
# 词面分可能很低，预筛会把正确答案筛掉。60 条问题约 3~4k token，快速模型
# 一次判断约一秒，一场答辩几十次调用费用可以忽略。库更大时才按词面取前 60。
MAX_LLM_CANDIDATES = 60


@dataclass
class QaEntry:
    id: str
    question: str                 # 英文问法（评委多半这么问）
    answer: str                   # 要朗读的英文回答
    question_zh: str = ""
    answer_zh: str = ""           # 中文意思，只显示不朗读
    points: list[str] = field(default_factory=list)   # 中文要点，只显示
    aliases: list[str] = field(default_factory=list)  # 其他问法，帮助匹配
    category: str = ""
    priority: int = 0             # 0~3，被问到的可能性
    basis: str = ""               # thesis / reason / check / limit
    source: str = ""              # 论文出处
    note: str = ""                # 给自己看的提醒，**绝不朗读**

    def display_zh(self) -> str:
        if self.answer_zh.strip():
            return self.answer_zh.strip()
        if self.points:
            return "；".join(p.strip() for p in self.points if p.strip())
        return self.question_zh.strip()

    def match_text(self) -> str:
        return " ".join([self.question, *self.aliases, self.question_zh])


BASIS_LABELS = {
    "thesis": ("论文有依据", "#3ecf8e"),
    "reason": ("推理回答·论文未写", "#f5a623"),
    "check": ("需核实", "#ff6b6b"),
    "limit": ("承认局限", "#b18cff"),
}


class QaBankError(ValueError):
    """导入文件格式不对。消息直接给用户看。"""


# ------------------------------------------------------------------ 导入解析
def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _first(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entry_from_dict(item: dict, index: int, categories: dict[str, str]) -> QaEntry | None:
    question = _first(item, "question", "q_en", "q", "question_en")
    answer = _first(item, "answer", "a_en", "a", "answer_en")
    if not question or not answer:
        return None
    cat = _first(item, "category", "cat")
    try:
        priority = int(item.get("priority", item.get("pri", 0)) or 0)
    except (TypeError, ValueError):
        priority = 0
    return QaEntry(
        id=_first(item, "id") or f"Q{index}",
        question=question,
        answer=answer,
        question_zh=_first(item, "question_zh", "q_zh"),
        answer_zh=_first(item, "answer_zh", "a_zh"),
        points=_as_list(item.get("points")),
        aliases=_as_list(item.get("aliases") or item.get("alt")),
        category=categories.get(cat, cat),
        priority=max(0, min(3, priority)),
        basis=_first(item, "basis"),
        source=_first(item, "source", "src"),
        note=_first(item, "note", "flag"),
    )


def parse_json_bank(text: str) -> tuple[list[QaEntry], int]:
    """返回 (条目, 跳过的条数)。兼容本工具导出的格式和「预案库 50 题」的 JSON。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QaBankError(f"JSON 格式错误（第 {exc.lineno} 行）：{exc.msg}") from exc
    categories: dict[str, str] = {}
    if isinstance(data, dict):
        for cat in data.get("categories", []) or []:
            if isinstance(cat, dict) and cat.get("key"):
                categories[str(cat["key"])] = str(cat.get("name", cat["key"]))
        items = data.get("items", data.get("entries"))
    else:
        items = data
    if not isinstance(items, list):
        raise QaBankError("JSON 里没有找到问答列表（需要 items 数组，或整个文件就是数组）")
    entries: list[QaEntry] = []
    skipped = 0
    for index, item in enumerate(items, 1):
        entry = _entry_from_dict(item, index, categories) if isinstance(item, dict) else None
        if entry is None:
            skipped += 1
        else:
            entries.append(entry)
    return entries, skipped


_FIELD_PREFIXES = [
    ("question", ("q", "question", "问", "问题")),
    ("answer", ("a", "answer", "答", "回答")),
    ("answer_zh", ("zh", "中文", "中文意思")),
    ("aliases", ("alt", "alias", "别问法", "其他问法")),
    ("note", ("note", "备注", "提醒")),
]
_PREFIX_RE = re.compile(
    r"^\s*(?:[-*#>]+\s*)?(?:\d+[.)、]\s*)?(?:\*\*)?(?P<key>[A-Za-z]+|[\u4e00-\u9fff]{1,4})(?:\*\*)?\s*[:：]\s*(?P<rest>.*)$"
)


def _field_for(key: str) -> str | None:
    lowered = key.lower()
    for name, prefixes in _FIELD_PREFIXES:
        if lowered in prefixes:
            return name
    return None


def parse_text_bank(text: str) -> tuple[list[QaEntry], int]:
    """纯文本 / Markdown：

        Q: Why did you choose Swin Transformer?
        A: The defect information has two scales...
        ZH: （可选）中文意思
        Alt: （可选）另一种问法

    「问：/答：」同样可以。一个 Q 开始一条新问答；没有前缀的行接在上一个字段后面。
    """
    entries: list[QaEntry] = []
    skipped = 0
    current: dict[str, list[str]] | None = None
    last_field: str | None = None

    def flush() -> None:
        nonlocal current, skipped
        if current is None:
            return
        question = " ".join(current.get("question", [])).strip()
        answer = " ".join(current.get("answer", [])).strip()
        if question and answer:
            entries.append(QaEntry(
                id=f"Q{len(entries) + 1}",
                question=question,
                answer=answer,
                answer_zh=" ".join(current.get("answer_zh", [])).strip(),
                aliases=[a for a in current.get("aliases", []) if a],
                note=" ".join(current.get("note", [])).strip(),
            ))
        else:
            skipped += 1
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _PREFIX_RE.match(line)
        name = _field_for(match.group("key")) if match else None
        if name is not None:
            rest = match.group("rest").strip()
            if name == "question":
                flush()
                current = {}
            if current is None:
                current = {}
            current.setdefault(name, []).append(rest)
            last_field = name
        elif current is not None and last_field is not None:
            if last_field == "aliases":
                current.setdefault("aliases", []).append(line)
            else:
                current.setdefault(last_field, []).append(line)
    flush()
    return entries, skipped


def parse_bank_file(path: str | Path) -> tuple[list[QaEntry], int]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gbk", errors="replace")
    except OSError as exc:
        raise QaBankError(f"读不了这个文件：{exc}") from exc
    if path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        entries, skipped = parse_json_bank(text)
    else:
        entries, skipped = parse_text_bank(text)
    if not entries:
        raise QaBankError("文件里没有识别到任何「问题 + 回答」。JSON 需要 question/answer 字段；文本需要 Q:/A: 开头的行。")
    return entries, skipped


def merge_entries(existing: list[QaEntry], incoming: list[QaEntry]) -> list[QaEntry]:
    """追加导入：编号撞了就加后缀，绝不覆盖已有条目。"""
    taken = {e.id for e in existing}
    merged = list(existing)
    for entry in incoming:
        new_id = entry.id
        suffix = 2
        while new_id in taken:
            new_id = f"{entry.id}-{suffix}"
            suffix += 1
        taken.add(new_id)
        merged.append(QaEntry(**{**asdict(entry), "id": new_id}))
    return merged


# ------------------------------------------------------------------ 本机存储
class QaBankStore:
    """问答库保存在本机配置目录。原子写入，读失败当空库。"""

    def __init__(self, base_dir: Path) -> None:
        self.path = Path(base_dir) / BANK_FILE

    def load(self) -> list[QaEntry]:
        if not self.path.exists():
            return []
        try:
            entries, _ = parse_json_bank(self.path.read_text(encoding="utf-8"))
        except (OSError, QaBankError):
            return []
        return entries

    def save(self, entries: Iterable[QaEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": BANK_VERSION, "items": [asdict(e) for e in entries]}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)


# ------------------------------------------------------------------ 词面检索
_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being do does did
    you your yours we our i me my it its this that these those what which who whom how why when where there
    can could would should will shall may might must have has had not no so than then also just very about
    into over under more most such some any each other please explain tell us describe question think
    let say said thank thanks okay ok yes well really""".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _stem(word: str) -> str:
    for suffix in ("ation", "ing", "ies", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    """英文词（去停用词、粗略词干）+ 中文二字组。只用于排序，不求精确。"""
    lowered = (text or "").lower()
    tokens = {_stem(w) for w in _WORD_RE.findall(lowered) if len(w) > 1 and w not in _STOPWORDS}
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class QaIndex:
    def __init__(self, entries: list[QaEntry]) -> None:
        self.entries = list(entries)
        self._tokens = [tokenize(e.match_text()) for e in self.entries]
        count = max(len(self.entries), 1)
        df: dict[str, int] = {}
        for tokens in self._tokens:
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        # 平滑 IDF：所有条目都有的词（thesis、model…）权重接近 0
        self._idf = {t: math.log((count + 1) / (n + 0.5)) for t, n in df.items()}
        self._default_idf = math.log((count + 1) / 0.5)

    def _weight(self, tokens: set[str]) -> float:
        return sum(self._idf.get(t, self._default_idf) for t in tokens)

    def search(self, query: str, limit: int = 5) -> list[tuple[QaEntry, float]]:
        """加权 Dice 相似度，0~1。只返回分数 > 0 的条目。"""
        q = tokenize(query)
        if not q:
            return []
        q_weight = self._weight(q)
        scored: list[tuple[QaEntry, float]] = []
        for entry, tokens in zip(self.entries, self._tokens):
            common = q & tokens
            if not common:
                continue
            overlap = self._weight(common)
            denom = q_weight + self._weight(tokens)
            # 浮点求和顺序随集合迭代顺序变化，可能算出 1.0000000000000002：钳到 1
            score = min(1.0, 2 * overlap / denom) if denom > 0 else 0.0
            scored.append((entry, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


# ------------------------------------------------------------------ 大模型匹配
MATCH_SYSTEM_PROMPT = """You match what a thesis-defense committee member just said to the candidate's prepared Q&A bank.

Rules:
1. Pick an entry only if it asks for the SAME information as the committee member's latest question, even if the wording differs or the speech recognition has errors.
2. A merely related topic is NOT a match. If the question asks something the entry does not answer, output NONE.
3. If the latest remark is not a question (a comment, a greeting, a transition), output NONE.
4. Use the earlier remarks only as context for a short follow-up such as "Can you explain that?".
5. The latest turn may contain multiple sentences and subquestions. Match only if
   the prepared answer covers ALL requested information and constraints, not just
   the final sentence or one subquestion. Use the original turn as the authority.

Output exactly one line of JSON and nothing else:
{"id": "<entry id or NONE>", "confidence": <number from 0 to 1>}"""


def build_match_messages(
    question: str, candidates: list[QaEntry], context: list[str] | None = None, *, intent: str = ""
) -> list[dict[str, str]]:
    lines = []
    for entry in candidates:
        parts = [f"[{entry.id}] {entry.question}"]
        if entry.aliases:
            parts.append("also asked as: " + " / ".join(entry.aliases))
        # 中文问法只在条目本身没有英文问法时附上，免得整库翻倍拖慢判断
        if entry.question_zh and not re.search(r"[A-Za-z]", entry.question):
            parts.append("中文: " + entry.question_zh)
        parts.append("Prepared answer: " + entry.answer)
        lines.append(" | ".join(parts))
    earlier = [c.strip() for c in (context or []) if c and c.strip()]
    user = "\n".join([
        "Prepared questions:",
        *lines,
        "",
        "Earlier remarks by the committee (oldest first):" if earlier else "Earlier remarks: (none)",
        *earlier,
        "",
        f"Latest remark: {question.strip()}",
        "Interpretation of the whole turn (advisory; retain every original constraint): " + intent if intent else "",
    ])
    return [
        {"role": "system", "content": MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_match(content: str, valid_ids: Iterable[str]) -> tuple[str | None, float]:
    """容错解析。任何看不懂的输出都当作「没命中」。"""
    valid = {str(v) for v in valid_ids}
    text = (content or "").strip()
    data = None
    found = re.search(r"\{.*?\}", text, re.DOTALL)
    if found:
        try:
            data = json.loads(found.group(0))
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        raw_id = str(data.get("id", "")).strip().strip("[]")
        try:
            confidence = float(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if raw_id in valid:
            return raw_id, confidence
        return None, confidence
    # 没给 JSON：只在恰好提到一个合法编号、且没说 NONE 时接受，置信度按保守值
    if "NONE" in text.upper():
        return None, 0.0
    mentioned = [v for v in valid if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text)]
    if len(mentioned) == 1:
        return mentioned[0], MIN_LLM_CONFIDENCE
    return None, 0.0


@dataclass
class MatchResult:
    entry: QaEntry | None
    confidence: float = 0.0
    method: str = "none"          # llm / lexical / none
    candidates: list[tuple[QaEntry, float]] = field(default_factory=list)
    error: str = ""

    @property
    def matched(self) -> bool:
        return self.entry is not None


def match_question(
    entries: list[QaEntry],
    question: str,
    chat: Callable[[list[dict[str, str]]], str] | None,
    *,
    context: list[str] | None = None,
    min_confidence: float = MIN_LLM_CONFIDENCE,
    max_candidates: int = MAX_LLM_CANDIDATES,
    lexical_strong: float = LEXICAL_STRONG,
    intent: str = "",
    allow_lexical_match: bool = True,
) -> MatchResult:
    """先词面检索缩小范围，再让大模型判断。chat=None 或调用失败时只用词面结果。"""
    question = (question or "").strip()
    if not entries or not question:
        return MatchResult(entry=None)
    index = QaIndex(entries)
    query = " ".join([*(context or [])[-1:], question])
    ranked = index.search(query, limit=len(entries))
    display = ranked[:5]
    if len(entries) <= max_candidates:
        # 库小就全部给大模型看：换了说法的问题词面分可能是 0，不能被预筛掉
        seen = {e.id for e, _ in ranked}
        pool = [e for e, _ in ranked] + [e for e in entries if e.id not in seen]
    else:
        pool = [e for e, _ in ranked[:max_candidates]]

    enough_words = len(tokenize(question)) >= MIN_LEXICAL_TOKENS

    def lexical_fallback(error: str) -> MatchResult:
        if allow_lexical_match and ranked and ranked[0][1] >= lexical_strong and enough_words:
            return MatchResult(ranked[0][0], ranked[0][1], "lexical", display, error)
        return MatchResult(None, ranked[0][1] if ranked else 0.0, "none", display, error)

    if chat is None or not pool:
        return lexical_fallback("")
    try:
        reply = chat(build_match_messages(question, pool, context, intent=intent))
    except Exception as exc:  # noqa: BLE001 —— 网络/鉴权/超时都退回本地匹配
        return lexical_fallback(" ".join(str(exc).split())[:120])
    chosen_id, confidence = parse_match(reply, (e.id for e in pool))
    if chosen_id is None or confidence < min_confidence:
        return MatchResult(None, confidence, "none", display)
    chosen = next(e for e in pool if e.id == chosen_id)
    # 下拉候选里把大模型选中的排第一
    others = [(e, s) for e, s in display if e.id != chosen.id]
    return MatchResult(chosen, confidence, "llm", [(chosen, confidence), *others][:5])


def reference_pairs(candidates: list[tuple[QaEntry, float]], limit: int = 3) -> list[tuple[str, str]]:
    """没命中时，把最相关的几条预案当参考交给现场生成，让措辞和事实跟你准备的一致。"""
    return [(e.question, e.answer) for e, score in candidates[:limit] if score > 0]
