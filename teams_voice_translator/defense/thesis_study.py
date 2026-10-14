"""论文预习：答辩前用强模型（默认 qwen3.8-max）通读论文，产出可人工校对的中英
「论文理解」，并批量生成预案问答；答辩现场用快模型（qwen3.8-flash）作答时，
把这份理解当作事实依据。

分工的道理：
* 预习在答辩前做，不赶时间，可以用最强的模型、打开深度思考，把全文吃透；
* 现场评委在等，只能用快模型；快模型不必再读 20 万字原文，只读这份几千字的、
  你亲自校对过的理解（加上按问题检索出的几段原文），又快又不容易编造。

「校准」：你改了中文或英文的任意一边，点按钮让快模型把另一边改成一致的说法。
英文侧会尽量保留原来的措辞和简单口语风格，只改意思不一致的地方。

所有结果都只存在本机配置目录（``thesis_study.json``）；生成的问答直接进问答库。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..aliyun import ApiError
from .chat_models import thinking_fields
from .qa_bank import QaEntry
from .thesis_context import input_char_limit

STUDY_FILE = "thesis_study.json"
DEFAULT_STUDY_MODEL = "qwen3.8-max"
DEFAULT_QA_COUNT = 20
MAX_QA_COUNT = 300          # 界面上一次最多要多少题（分批生成）
PER_REQUEST_MAX = 60        # 单次请求最多要多少题：再多输出就会被截断
QA_BATCH = 10   # 一次请求生成的题数：太多会被输出长度截断，JSON 也更容易出错

_STUDY_INSTRUCTION = """你是这位硕士研究生的答辩教练，也是这个领域的审稿专家。下面是他的学位论文全文。
请彻底读懂这篇论文，写一份给答辩人本人和现场助答系统使用的「论文理解」，中英双语。

要求：
1. 分 6~10 个小节，必须包含：研究问题与贡献、数据、方法、关键实验结果与数字、局限与未来工作、
   评委最可能追问的薄弱点（每个薄弱点给出诚实的应对思路）。其余小节按论文实际内容安排。
2. 每节中文 150~400 字；英文与中文意思完全一致，用简单清楚的学术口语英文（答辩人英语一般）。
3. 所有数字、模型名、指标名必须与论文原文一致，不确定的宁可不写，**绝不编造**。
4. 论文里前后不一致的数字、没有解释的反常结果，要在「薄弱点」里明确指出。

只输出一个 JSON 对象，不要任何解释：
{"sections": [{"title_zh": "...", "title_en": "...", "zh": "...", "en": "..."}]}"""

_QA_INSTRUCTION = """你是这位硕士研究生的答辩教练。根据论文全文和已校对的「论文理解」，
预测评委最可能问的 {count} 个问题，并写出答辩人可以直接朗读的回答。

要求：
1. 问题用评委真实会用的英文问法，覆盖动机、数据、方法、实验、结果解读、局限、与已有工作的区别、工程应用；
   尖锐问题（论文的薄弱点、反常数据）要优先。
2. 下面列出的已有问题**不要重复**，意思相同的也不要。
3. 英文回答：第一人称，简单清楚的口语英文，60~110 个词，数字与论文一致；论文没写的不要编造，
   a_en 和 a_zh 都是我本人直接回答；禁止“建议用户说”“作者认为”“The candidate should”等第三人称旁白。
   需要推理时用 "A possible reason is..." 这类谨慎说法；确实是局限就大方承认并给出下一步。
4. "a_zh" 是英文回答的完整中文翻译；"points" 是 2~4 条中文要点。
5. "pri"：被问到的可能性 1~3；"basis"：thesis（论文有依据）/ reason（推理，论文没写）/ limit（承认局限）；
   "src"：论文里的章节或表号。

只输出一个 JSON 对象，不要任何解释：
{{"items": [{{"q_en": "...", "q_zh": "...", "a_en": "...", "a_zh": "...", "points": ["..."],
"pri": 3, "basis": "thesis", "src": "..."}}]}}"""

_CALIBRATE_TO_EN = """你是论文答辩的口译。答辩人修改了中文，请据此改写英文，使英文与中文意思完全一致。
- 原英文里与新中文一致的部分，尽量保留原来的措辞；
- 用简单清楚的口语英文，第一人称；术语与术语表一致；数字照抄中文；
- 只输出改写后的英文，不要任何解释。"""

_CALIBRATE_TO_ZH = """你是论文答辩的口译。答辩人修改了英文，请据此改写中文，使中文与英文意思完全一致。
- 原中文里与新英文一致的部分，尽量保留原来的措辞；
- 中文自然流畅；术语与术语表一致；数字照抄英文；
- 只输出改写后的中文，不要任何解释。"""


# ------------------------------------------------------------------ 数据
@dataclass
class StudySection:
    title_zh: str
    title_en: str
    zh: str
    en: str


@dataclass
class StudyNotes:
    sections: list[StudySection] = field(default_factory=list)
    model: str = ""
    created_at: str = ""
    edited_at: str = ""

    def render(self, language: str = "en") -> str:
        """给助答提示词用：英文侧（回答要说英文）。"""
        blocks = []
        for section in self.sections:
            title = section.title_en if language == "en" else section.title_zh
            body = section.en if language == "en" else section.zh
            if body.strip():
                blocks.append(f"## {title.strip()}\n{body.strip()}")
        return "\n\n".join(blocks)


class StudyStore:
    def __init__(self, base_dir: Path) -> None:
        self.path = Path(base_dir) / STUDY_FILE

    def load(self) -> StudyNotes | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        sections = [
            StudySection(str(s.get("title_zh", "")), str(s.get("title_en", "")),
                         str(s.get("zh", "")), str(s.get("en", "")))
            for s in data.get("sections", []) or [] if isinstance(s, dict)
        ]
        if not sections:
            return None
        return StudyNotes(sections, str(data.get("model", "")), str(data.get("created_at", "")),
                          str(data.get("edited_at", "")))

    def save(self, notes: StudyNotes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(notes), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)


# ------------------------------------------------------------------ 解析工具
def _json_object(content: str, what: str) -> dict:
    raw = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ApiError(f"{what}返回的不是 JSON，请重试")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ApiError(f"{what}返回的 JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise ApiError(f"{what}返回格式不对")
    return data


_KEY_RE = re.compile(r'"([A-Za-z_]+)"\s*:\s*')


def _loose_fields(text: str) -> dict[str, Any]:
    """逐字段宽松提取一个条目：回答里有没转义的英文引号（模型常犯）也能取出来。

    以「逗号或左花括号后面的 "键":」为字段边界，值取到下一个字段边界为止。
    """
    keys = [m for m in _KEY_RE.finditer(text)
            if m.start() == 0 or re.search(r"[{,]\s*$", text[:m.start()])]
    result: dict[str, Any] = {}
    for index, match in enumerate(keys):
        end = keys[index + 1].start() if index + 1 < len(keys) else len(text)
        value = text[match.end():end].strip().rstrip(",").strip()
        if index + 1 == len(keys):
            value = value.rstrip("}]").strip().rstrip(",").strip()
        key = match.group(1)
        if value.startswith("["):
            try:
                result[key] = json.loads(value[:value.rfind("]") + 1])
            except (json.JSONDecodeError, ValueError):
                result[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', value)
        elif value.startswith('"'):
            inner = value[1:]
            if inner.endswith('"'):
                inner = inner[:-1]
            try:
                result[key] = json.loads(f'"{inner}"')
            except (json.JSONDecodeError, ValueError):
                result[key] = inner.replace('\\n', "\n").replace('\\"', '"')
        else:
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                result[key] = value
    return result


def salvage_items(content: str) -> list[dict[str, Any]]:
    """整体 JSON 解析失败时，一条一条地救：坏一条丢一条，不连累整批。"""
    raw = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    match = re.search(r'"items"\s*:\s*\[', raw)
    body = raw[match.end():] if match else raw[raw.find("[") + 1:] if "[" in raw else raw
    items: list[dict[str, Any]] = []
    pieces = re.split(r"\}\s*,\s*(?=\{\s*\")", body)
    for position, piece in enumerate(pieces):
        piece = piece.strip()
        start = piece.find("{")
        if start == -1:
            continue
        if position == len(pieces) - 1 and not re.search(r"\}\s*(\]\s*\}?)?\s*$", piece):
            break   # 最后一条被输出长度截断了：半截回答不能要
        inner = piece[start + 1:]
        inner = re.sub(r"\}\s*\]\s*\}?\s*$", "", inner).rstrip().rstrip("}").rstrip()
        candidate = "{" + re.sub(r",\s*$", "", inner) + "}"
        try:
            item = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
        except (json.JSONDecodeError, ValueError):
            item = _loose_fields(inner)
        if isinstance(item, dict) and item:
            items.append(item)
    return items


def _items_from(content: str, what: str) -> list[dict[str, Any]]:
    try:
        data = _json_object(content, what)
        items = data.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    except ApiError:
        pass
    items = salvage_items(content)
    if not items:
        raise ApiError(f"{what}返回的内容无法解析，请重试")
    return items


def _glossary_block(terms: list[dict[str, str]] | None) -> str:
    rows = [f"{t['source']} → {t['target']}" for t in (terms or []) if t.get("source") and t.get("target")]
    return "\n".join(rows[:120]) or "（无）"


# ------------------------------------------------------------------ 预习
def build_study_payload(full_text: str, model: str, *, thinking: bool, glossary=None) -> dict:
    limit = input_char_limit(model)
    user = f"术语表（英文写法以此为准）：\n{_glossary_block(glossary)}\n\n===== 论文全文 =====\n{full_text[:limit]}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _STUDY_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }
    payload.update(thinking_fields(model, want_thinking=thinking, budget=16384 if thinking else 0))
    return payload


def parse_study_response(content: str, *, model: str = "") -> StudyNotes:
    data = _json_object(content, "论文预习")
    sections = []
    for item in data.get("sections") or []:
        if not isinstance(item, dict):
            continue
        section = StudySection(
            title_zh=str(item.get("title_zh", "")).strip(),
            title_en=str(item.get("title_en", "")).strip(),
            zh=str(item.get("zh", "")).strip(),
            en=str(item.get("en", "")).strip(),
        )
        if section.zh or section.en:
            sections.append(section)
    if not sections:
        raise ApiError("论文预习没有返回任何小节，请重试")
    return StudyNotes(sections, model, time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------ 批量生成问答
def build_qa_payload(full_text: str, notes: StudyNotes | None, existing: list[str], model: str,
                     *, count: int, thinking: bool, glossary=None) -> dict:
    count = max(1, min(PER_REQUEST_MAX, int(count)))
    limit = input_char_limit(model)
    existing_block = "\n".join(f"- {q}" for q in existing[:200]) or "（暂无）"
    notes_block = notes.render("en") if notes else "（尚未预习）"
    user = "\n\n".join([
        f"术语表（英文写法以此为准）：\n{_glossary_block(glossary)}",
        f"===== 已校对的论文理解 =====\n{notes_block}",
        f"===== 已有问题（不要重复）=====\n{existing_block}",
        f"===== 论文全文 =====\n{full_text[:max(20_000, limit - 20_000)]}",
    ])
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _QA_INSTRUCTION.format(count=count)},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
    }
    payload.update(thinking_fields(model, want_thinking=thinking, budget=16384 if thinking else 0))
    return payload


def parse_qa_response(content: str, *, id_prefix: str = "S", start: int = 1,
                      category: str = "预习生成", note: str = "由论文预习自动生成，读之前请核对",
                      what: str = "预案生成") -> list[QaEntry]:
    entries: list[QaEntry] = []
    for item in _items_from(content, what):
        q_en = str(item.get("q_en", "")).strip()
        a_en = str(item.get("a_en", "")).strip()
        if not q_en or not a_en:
            continue
        try:
            pri = max(0, min(3, int(item.get("pri", 2) or 2)))
        except (TypeError, ValueError):
            pri = 2
        basis = str(item.get("basis", "") or "").strip()
        entries.append(QaEntry(
            id=f"{id_prefix}{start + len(entries)}",
            question=q_en,
            answer=a_en,
            question_zh=str(item.get("q_zh", "") or "").strip(),
            answer_zh=str(item.get("a_zh", "") or "").strip(),
            points=[str(p).strip() for p in (item.get("points") or []) if str(p).strip()],
            category=category,
            priority=pri,
            basis=basis if basis in ("thesis", "reason", "check", "limit") else "reason",
            source=str(item.get("src", "") or "").strip(),
            note=note,
        ))
    if not entries:
        raise ApiError(f"{what}没有返回任何问答，请重试")
    return entries


# ------------------------------------------------------------------ 导入其他 AI 写的问答
_IMPORT_INSTRUCTION = """你是论文答辩助手。下面是用户从其他 AI 或自己整理的答辩问答材料，格式可能很乱
（编号列表、Markdown、表格、中英混杂、只有中文等）。请从中识别出每一组「问题 + 回答」，整理成统一格式：

1. 只整理材料里已有的问答：不要新编问题，不要遗漏；没有回答的问题跳过；一个问题有多段回答就合并成一个回答。
2. q_en：问题的英文。原文是英文就保留；只有中文就译成评委会用的自然英文问法。q_zh：问题的中文。
3. a_en：可以直接朗读的英文回答，第一人称。原文有英文就保留原意，只修正明显的语法错误；只有中文就译成
   简单清楚的口语英文。a_zh：a_en 的完整中文意思。
4. points：2~4 条中文要点（可以为空）；pri：被问到的可能性 1~3，看不出来就写 2。
5. 专有名词和数字与材料保持一致（参考术语表），不要编造材料里没有的内容。
6. 字符串里的英文双引号必须写成 \\"。

只输出一个 JSON 对象，不要任何解释：
{"items": [{"q_en": "...", "q_zh": "...", "a_en": "...", "a_zh": "...", "points": ["..."], "pri": 2}]}"""

IMPORT_CHUNK_CHARS = 6000


def split_import_text(text: str, limit: int = IMPORT_CHUNK_CHARS) -> list[str]:
    """按空行把长材料切成几块，每块不超过 limit 字，尽量不把一组问答切开。"""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        if current and len(current) + len(paragraph) + 2 > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def build_import_payload(text: str, model: str, *, glossary=None) -> dict:
    user = f"术语表（英文写法以此为准）：\n{_glossary_block(glossary)}\n\n===== 问答材料 =====\n{text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _IMPORT_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    payload.update(thinking_fields(model, want_thinking=False))
    return payload


def parse_import_response(content: str, *, start: int = 1) -> list[QaEntry]:
    return parse_qa_response(content, id_prefix="I", start=start, category="外部导入",
                             note="从其他 AI 的问答导入，读之前请核对", what="问答识别")


def next_id_start(existing_ids: list[str], prefix: str) -> int:
    numbers = [int(m.group(1)) for i in existing_ids if (m := re.fullmatch(rf"{re.escape(prefix)}(\d+)", i))]
    return (max(numbers) + 1) if numbers else 1


# ------------------------------------------------------------------ 中英校准
def build_calibrate_messages(direction: str, edited: str, other: str, *, context: str = "",
                             glossary=None) -> list[dict[str, str]]:
    """direction='zh2en'：中文改了，改写英文；'en2zh'：英文改了，改写中文。"""
    if direction not in ("zh2en", "en2zh"):
        raise ValueError(direction)
    system = _CALIBRATE_TO_EN if direction == "zh2en" else _CALIBRATE_TO_ZH
    edited_label, other_label = ("修改后的中文", "原来的英文") if direction == "zh2en" else ("修改后的英文", "原来的中文")
    user = "\n\n".join(part for part in [
        f"术语表：\n{_glossary_block(glossary)}",
        f"背景（这段话在回答的问题 / 所属小节）：{context}" if context.strip() else "",
        f"{edited_label}：\n{edited.strip()}",
        f"{other_label}（供参考，可能已过时）：\n{other.strip() or '（空）'}",
    ] if part)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_calibrated(text: str) -> str:
    """去掉模型偶尔加的引号、「英文：」前缀、代码块。"""
    text = (text or "").strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(英文|中文|English|Chinese)\s*[:：]\s*", "", text).strip()
    if len(text) >= 2 and text[0] in "\"“「'" and text[-1] in "\"”」'":
        text = text[1:-1].strip()
    return text


def calibrate(translator, direction: str, edited: str, other: str, *, context: str = "",
              glossary=None, cancel_event=None) -> str:
    """用快模型（不思考）校准另一种语言。translator 是 ContextTranslator。"""
    messages = build_calibrate_messages(direction, edited, other, context=context, glossary=glossary)
    result = clean_calibrated(translator.chat(messages, cancel_event=cancel_event, thinking=False))
    if not result:
        raise ApiError("校准没有返回内容")
    return result


def run_payload(translator, payload: dict[str, Any], *, thinking: bool, cancel_event=None,
                on_reasoning=None) -> str:
    """长任务：思考可能要好几分钟，给足时间。"""
    return translator.complete(payload, cancel_event=cancel_event, on_reasoning=on_reasoning,
                               deadline_seconds=900 if thinking else 420)


# ------------------------------------------------------------------ 给问答库补中文
_FILL_ZH_INSTRUCTION = """你是论文答辩的口译。下面每一条是答辩预案的英文问题和英文回答，请给出准确、通顺的中文翻译。
专有名词参考术语表；数字、模型名、指标名保持原样。只翻译，不增删内容。
字符串里的英文双引号必须写成 \\"。
只输出一个 JSON 对象，不要任何解释：
{"items": [{"id": "原样照抄", "q_zh": "问题的中文", "a_zh": "回答的完整中文"}]}"""

FILL_ZH_BATCH = 8


def needs_chinese(entry: QaEntry) -> bool:
    return not entry.question_zh.strip() or not entry.answer_zh.strip()


def build_fill_zh_payload(entries: list[QaEntry], model: str, *, glossary=None) -> dict:
    rows = [{"id": e.id, "q_en": e.question, "a_en": e.answer} for e in entries]
    user = (f"术语表：\n{_glossary_block(glossary)}\n\n===== 待翻译 =====\n"
            + json.dumps(rows, ensure_ascii=False, indent=1))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _FILL_ZH_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    payload.update(thinking_fields(model, want_thinking=False))
    return payload


def parse_fill_zh_response(content: str) -> dict[str, tuple[str, str]]:
    """返回 {id: (问题中文, 回答中文)}。某条坏了就跳过那一条。"""
    result: dict[str, tuple[str, str]] = {}
    for item in _items_from(content, "补中文"):
        entry_id = str(item.get("id", "") or "").strip()
        if not entry_id:
            continue
        result[entry_id] = (str(item.get("q_zh", "") or "").strip(), str(item.get("a_zh", "") or "").strip())
    return result


def apply_chinese(entries: list[QaEntry], translations: dict[str, tuple[str, str]]) -> tuple[list[QaEntry], int]:
    """只补空着的中文，你已经写好的中文不覆盖。返回 (新列表, 补了几条)。"""
    updated: list[QaEntry] = []
    changed = 0
    for entry in entries:
        q_zh, a_zh = translations.get(entry.id, ("", ""))
        new_q = entry.question_zh or q_zh
        new_a = entry.answer_zh or a_zh
        if new_q != entry.question_zh or new_a != entry.answer_zh:
            changed += 1
            entry = QaEntry(**{**asdict(entry), "question_zh": new_q, "answer_zh": new_a})
        updated.append(entry)
    return updated, changed
