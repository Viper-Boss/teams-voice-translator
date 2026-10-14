"""像真人的小失误：偶尔卡一下壳、重读一个词、说半句重来。

真人说话不会每句都语法完美、一口气说完。GPT 语音模式听着像真人，很大一部分
就是因为它偶尔会「The... the model」「we used a, um, we used a」。这里只改**送给
TTS 的文本**，字幕、记录和缓存显示的仍是原句。

规则（稳定优先）：
* 按句独立掷骰子，概率由档位决定（很少 2% / 自然 3.5% / 较多 6%）。
* 同一块文本（一次合成请求）最多出一次失误，避免连着出错一听就假。
* **数字、缩写（全大写）、术语表里的词绝不重复或打断**——这些读错比不读还糟。
* 少于 5 个词的短句不动；中文暂不处理（答辩是英文发声）。
* 结果由句子内容决定（同一句同一档位永远同样的失误），所以提前渲染和现场发声
  命中同一份缓存，重播也是同一个「版本」。
"""
from __future__ import annotations

import hashlib
import random
import re

LEVELS: dict[str, float] = {
    "off": 0.0,
    "rare": 0.02,
    "natural": 0.035,
    "more": 0.06,
}

LEVEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("off", "关闭（每句都说得很完整）"),
    ("rare", "很少 · 约 2% 的句子卡一下"),
    ("natural", "自然 · 约 3.5% 的句子（推荐）"),
    ("more", "较多 · 约 6% 的句子"),
)

# 可以放心在句首重复的虚词：重复它们不改变意思，也最像真人卡壳。
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "we", "i", "this", "that", "these", "those", "so", "and", "but",
    "in", "on", "our", "it", "my", "for", "to", "with", "as", "there", "if", "when",
    "because", "then", "also", "what", "which", "one", "first", "here",
})
PREPOSITIONS = frozenset({"in", "on", "for", "with", "of", "to", "from", "by", "at", "about"})
FILLERS = ("um,", "um,", "uh,")

_WORD_CORE = re.compile(r"^[A-Za-z][A-Za-z'’]*$")
_PUNCT_END = re.compile(r"[,.;:!?)\]\"”’]$")


def rate_for(level: str) -> float:
    return LEVELS.get(str(level or "off"), 0.0)


def _core(word: str) -> str:
    return word.strip("\"'“”‘’()[]")


def _protected_words(protected) -> set[str]:
    words: set[str] = set()
    for term in protected or ():
        for piece in re.split(r"[\s\-/]+", str(term or "")):
            piece = _core(piece).strip(".,;:!?").lower()
            if piece:
                words.add(piece)
    return words


def _safe(word: str, protected: set[str]) -> bool:
    """这个词能不能被重复或在它前后打断。"""
    core = _core(word)
    if not core or _PUNCT_END.search(word):
        return False
    if not _WORD_CORE.match(core):          # 数字、符号、混写（BERT-base、3D）
        return False
    if len(core) > 1 and core.isupper():    # 缩写（CNN、SVM）
        return False
    if len(core) > 10:
        return False
    return core.lower() not in protected


def _lower_first(word: str) -> str:
    if word == "I" or word.startswith("I'") or word.startswith("I’"):
        return word
    return word[:1].lower() + word[1:]


def _rng(sentence: str, level: str) -> random.Random:
    digest = hashlib.sha1(f"{level}\x1f{sentence}".encode("utf-8", errors="replace")).hexdigest()
    return random.Random(int(digest[:12], 16))


def _repeat_start(words, protected, rng):
    first = words[0]
    if _core(first).lower() not in FUNCTION_WORDS or not _safe(first, protected):
        return None
    return [f"{first}...", _lower_first(first)] + words[1:]


def _restart_phrase(words, protected, rng):
    size = rng.choice((2, 3))
    if len(words) < size + 3:
        size = 2
    head = words[:size]
    if len(words) < size + 3 or not all(_safe(w, protected) for w in head):
        return None
    again = [_lower_first(head[0])] + head[1:]
    return head[:-1] + [f"{head[-1]}..."] + again + words[size:]


def _filler(words, protected, rng):
    filler = rng.choice(FILLERS)
    # 优先放在句中第一个逗号后面（真人停顿最常见的位置），但别太靠后。
    limit = max(2, int(len(words) * 0.7))
    for index in range(1, limit):
        if words[index - 1].endswith(",") and index < len(words) - 2 and _safe(words[index], protected):
            return words[:index] + [filler] + words[index:]
    if not _safe(words[0], protected):
        return None
    lead = filler[:1].upper() + filler[1:]
    return [lead, _lower_first(words[0])] + words[1:]


def _self_correct(words, protected, rng):
    """「in the... sorry, in the second chapter」：介词短语说到一半重来。"""
    for index in range(1, len(words) - 3):
        prep, nxt = words[index], words[index + 1]
        if (_core(prep).lower() in PREPOSITIONS and _safe(prep, protected)
                and _safe(nxt, protected) and not words[index - 1].endswith(",")):
            return (words[:index] + [prep, f"{nxt}...", "sorry,", prep, nxt] + words[index + 2:])
    return None


_KINDS = (
    (_repeat_start, 35),
    (_restart_phrase, 30),
    (_filler, 25),
    (_self_correct, 10),
)


def humanize_sentence(sentence: str, level: str, *, protected=()) -> str:
    """按档位给一句话（可能）加一个小失误。不加时原样返回。"""
    rate = rate_for(level)
    text = str(sentence or "")
    if rate <= 0 or not text.strip():
        return text
    words = text.split()
    if len(words) < 5 or re.search(r"[一-鿿]", text):
        return text
    rng = _rng(text.strip(), level)
    if rng.random() >= rate:
        return text
    guard = _protected_words(protected)
    kinds = list(_KINDS)
    # 先按权重抽一种；这种放不进去就依次试别的，全都不行就不改。
    first = rng.choices(kinds, weights=[w for _k, w in kinds], k=1)[0]
    ordered = [first] + [item for item in kinds if item is not first]
    for kind, _weight in ordered:
        result = kind(words, guard, rng)
        if result:
            return " ".join(result)
    return text


def split_for_humanize(text: str) -> list[str]:
    """按句末标点切句，保留标点；不切小数和常见缩写。"""
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", str(text or "").strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and re.search(r"\b(?:e\.g|i\.e|Fig|Figs|Eq|Eqs|Dr|Mr|Mrs|Prof|vs|etc|al)\.$", merged[-1], re.I):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [piece for piece in merged if piece]


def humanize_text(text: str, level: str, *, protected=(), max_events: int = 1) -> str:
    """一块文本里逐句掷骰子，最多 max_events 处失误。"""
    if rate_for(level) <= 0 or not str(text or "").strip():
        return text
    sentences = split_for_humanize(text)
    out: list[str] = []
    events = 0
    for sentence in sentences:
        if events < max_events:
            changed = humanize_sentence(sentence, level, protected=protected)
            if changed != sentence:
                events += 1
            out.append(changed)
        else:
            out.append(sentence)
    return " ".join(out) if events else text


def protected_terms(settings) -> list[str]:
    """术语表里的英文（目标）和原文都保护起来。"""
    terms: list[str] = []
    try:
        for item in settings.glossary_terms():
            terms.append(str(item.get("target", "")))
            terms.append(str(item.get("source", "")))
    except Exception:  # noqa: BLE001 —— 术语表坏了也不能影响发声
        pass
    return [term for term in terms if term]
