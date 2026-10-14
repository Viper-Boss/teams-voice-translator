"""查重：术语表和问答库里「一样的」「差不多的」找出来，交给你逐组确认。

原则（避免起冲突）：
* **只提示、不自动删。** 每一组都要你点确认；整个查重过程在内存里进行，点「完成」才写盘，
  中途「取消」什么都不会变。
* **术语表硬规则：同一个中文只能有一种英文。** 中文相同的两条必须合成一条才能保存，
  否则翻译时模型不知道该用哪个英文——这是唯一不能「都保留」的情况。
* **问答库删条目时，声音跟着处理但不误删：** 只有被删那条的英文回答没有别的条目在用时，
  才删它渲染好的声音（相同的回答文本共用同一份音频）。
* 「都保留」会被记住（按内容记，不按编号），以后不再提示同一对。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field

from .qa_bank import QaEntry, tokenize

# ------------------------------------------------------------------ 通用
_PUNCT = re.compile(r"[\s\-_·•.,，。;；:：'\"“”‘’()（）\[\]【】/\\]+")


def norm(text: str) -> str:
    """比较用：去空白和标点、统一大小写。"""
    return _PUNCT.sub("", str(text or "")).casefold()


def pair_key(a: str, b: str) -> str:
    left, right = sorted((norm(a), norm(b)))
    return hashlib.sha1(f"{left}\x1f{right}".encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ 术语表
_ABBR = re.compile(r"\(([A-Za-z][A-Za-z0-9\-]{1,11})\)|（([A-Za-z][A-Za-z0-9\-]{1,11})）")


def _target_core(target: str) -> str:
    return norm(_ABBR.sub("", target))


def _abbrevs(term: dict[str, str]) -> set[str]:
    found = {m.group(1) or m.group(2) for m in _ABBR.finditer(term["target"])}
    for text in (term["source"], term["target"]):
        stripped = text.strip()
        if re.fullmatch(r"[A-Z][A-Z0-9\-]{1,11}", stripped):
            found.add(stripped)
    return {a.upper() for a in found if a}


@dataclass
class TermGroup:
    indices: list[int]
    reason: str
    must_merge: bool = False      # 中文相同：必须合成一条

    def key(self, terms: list[dict[str, str]]) -> str:
        texts = sorted(norm(terms[i]["source"]) + "=" + norm(terms[i]["target"]) for i in self.indices)
        return hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]


def _expansion(target: str, abbr: str) -> str:
    """「Magnetic Flux Leakage (MFL) inspection」里缩写前面的全称部分。"""
    match = re.search(r"(.*?)[(（]\s*" + re.escape(abbr) + r"\s*[)）]", target, re.IGNORECASE)
    return norm(match.group(1)) if match else ""


def find_term_groups(terms: list[dict[str, str]], ignored: set[str] | None = None) -> list[TermGroup]:
    """需要你看一眼的组，中文相同的排最前（必须处理）。

    * 中文相同（去空格、大小写后）：一组，必须合成一条；
    * 其余按「对」给出，每对说明原因：英文相同 / 同一个缩写却是不同全称 / 中文和英文都是一条包含另一条。
      只是共用一个缩写（MFL 和 漏磁检测(MFL)）不算，那是正常的两个术语。
    """
    ignored = ignored or set()
    src = [norm(t["source"]) for t in terms]
    core = [_target_core(t["target"]) for t in terms]
    groups: list[TermGroup] = []
    by_src: dict[str, list[int]] = {}
    for i, key in enumerate(src):
        by_src.setdefault(key, []).append(i)
    same_src = {i for members in by_src.values() if len(members) > 1 for i in members}
    for members in by_src.values():
        if len(members) > 1:
            groups.append(TermGroup(members, "中文相同：翻译时不知道用哪个英文，必须合成一条", must_merge=True))
    pairs: list[TermGroup] = []
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            if src[i] == src[j]:
                continue
            reason = ""
            if core[i] and core[i] == core[j]:
                reason = "英文写法相同，中文说法不同"
            else:
                shared = {m.group(1) or m.group(2) for m in _ABBR.finditer(terms[i]["target"])} & \
                         {m.group(1) or m.group(2) for m in _ABBR.finditer(terms[j]["target"])}
                for abbr in shared:
                    a, b = _expansion(terms[i]["target"], abbr), _expansion(terms[j]["target"], abbr)
                    if a and b and a != b:
                        reason = f"同一个缩写 {abbr} 写成了不同的全称"
                        break
                if not reason:
                    a, b, ca, cb = src[i], src[j], core[i], core[j]
                    if (len(a) >= 2 and len(b) >= 2 and (a in b or b in a)
                            and ca and cb and (ca in cb or cb in ca)):
                        reason = "中文和英文都是一条包含另一条"
            if not reason:
                continue
            if i in same_src and j in same_src:
                continue
            group = TermGroup([i, j], reason)
            if group.key(terms) not in ignored:
                pairs.append(group)
    return groups + pairs


def same_source_conflicts(terms: list[dict[str, str]]) -> list[list[int]]:
    """中文相同（去空格、大小写后）的条目编号，每组两条以上。保存前必须为空。"""
    seen: dict[str, list[int]] = {}
    for i, term in enumerate(terms):
        seen.setdefault(norm(term["source"]), []).append(i)
    return [ids for ids in seen.values() if len(ids) > 1]


# ------------------------------------------------------------------ 问答库
STRONG = 0.62       # 这么像基本就是同一个问题
MAYBE = 0.38        # 这之间的「拿不准」，可以让大模型再判断


@dataclass
class QaPair:
    left: str              # 条目编号
    right: str
    score: float
    exact: bool = False
    verdict: str = ""      # "" 未判断 / same / different（大模型给的）

    @property
    def level(self) -> str:
        if self.exact or self.score >= STRONG or self.verdict == "same":
            return "很可能重复"
        return "可能相近"


def _question_text(entry: QaEntry) -> str:
    return " ".join([entry.question, entry.question_zh, *entry.aliases])


def qa_pair_key(a: QaEntry, b: QaEntry) -> str:
    return pair_key(a.question, b.question)


def find_qa_pairs(entries: list[QaEntry], *, only_ids: set[str] | None = None,
                  ignored: set[str] | None = None, threshold: float = MAYBE) -> list[QaPair]:
    """问题意思相近的两两配对（只看问题，不看回答）。only_ids：只找和这些条目有关的（导入后用）。"""
    ignored = ignored or set()
    tokens = [tokenize(_question_text(e)) for e in entries]
    count = max(len(entries), 1)
    df: dict[str, int] = {}
    for toks in tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((count + 1) / (c + 0.5)) for t, c in df.items()}

    def weight(toks: set[str]) -> float:
        return sum(idf.get(t, 1.0) for t in toks)

    weights = [weight(t) for t in tokens]
    normalized = [norm(e.question) for e in entries]
    pairs: list[QaPair] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if only_ids is not None and a.id not in only_ids and b.id not in only_ids:
                continue
            if qa_pair_key(a, b) in ignored:
                continue
            exact = bool(normalized[i]) and normalized[i] == normalized[j]
            common = tokens[i] & tokens[j]
            if not exact and not common:
                continue
            denom = weights[i] + weights[j]
            score = 1.0 if exact else (2 * weight(common) / denom if denom > 0 else 0.0)
            if exact or score >= threshold:
                pairs.append(QaPair(a.id, b.id, round(score, 3), exact=exact))
    pairs.sort(key=lambda p: (-p.exact, -p.score))
    return pairs


def build_verify_messages(pairs: list[QaPair], by_id: dict[str, QaEntry]) -> list[dict[str, str]]:
    lines = []
    for number, pair in enumerate(pairs):
        a, b = by_id[pair.left], by_id[pair.right]
        lines.append(f"{number}. A: {a.question} ｜ B: {b.question}")
    system = ("下面每一行是论文答辩问答库里的两个问题。判断每一对是不是在问**同一件事**"
              "（换了说法、中英文不同也算同一件事；只是话题相关、问的点不同不算）。"
              "只输出 JSON 数组，例如 [{\"n\": 0, \"same\": true}]。")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def parse_verify(content: str, pairs: list[QaPair]) -> None:
    raw = content or ""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        if 0 <= number < len(pairs):
            pairs[number].verdict = "same" if item.get("same") else "different"


def merge_qa(keep: QaEntry, other: QaEntry) -> QaEntry:
    """合并成一条：回答、中文、提醒以 keep 为准；other 的问法记进「其他问法」，匹配时也能命中。"""
    aliases: list[str] = []
    seen = {norm(keep.question)}
    for text in [*keep.aliases, other.question, *other.aliases]:
        key = norm(text)
        if text.strip() and key not in seen:
            seen.add(key)
            aliases.append(text.strip())
    points = list(keep.points) or list(other.points)
    note = keep.note
    if other.note.strip() and other.note.strip() not in note:
        note = f"{note}；{other.note.strip()}" if note.strip() else other.note.strip()
    return QaEntry(**{**asdict(keep), "aliases": aliases, "points": points, "note": note,
                      "priority": max(keep.priority, other.priority),
                      "question_zh": keep.question_zh or other.question_zh,
                      "answer_zh": keep.answer_zh or (other.answer_zh if other.answer.strip() == keep.answer.strip()
                                                      else keep.answer_zh)})


@dataclass
class QaDedupeSession:
    """在内存里逐对处理，完成时才给出最终结果：处理到一半不写盘，取消就什么都不变。"""

    entries: list[QaEntry]
    pairs: list[QaPair]
    removed: dict[str, str] = field(default_factory=dict)   # 被删的编号 -> 保留下来的编号
    ignored: set[str] = field(default_factory=set)
    done: set[frozenset] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.entries = list(self.entries)
        self._original_answers = {e.id: e.answer.strip() for e in self.entries}

    def _resolve(self, entry_id: str) -> str:
        while entry_id in self.removed:
            entry_id = self.removed[entry_id]
        return entry_id

    def by_id(self) -> dict[str, QaEntry]:
        return {e.id: e for e in self.entries}

    def pending(self) -> list[QaPair]:
        """还要处理的对：被删掉的一方换成保留下来的那条；变成同一条的、处理过的自动跳过。"""
        result, seen = [], set()
        for pair in self.pairs:
            left, right = self._resolve(pair.left), self._resolve(pair.right)
            key = frozenset((left, right))
            if left == right or key in self.done or key in seen:
                continue
            seen.add(key)
            result.append(QaPair(left, right, pair.score, pair.exact, pair.verdict))
        return result

    def current(self) -> QaPair | None:
        pending = self.pending()
        return pending[0] if pending else None

    def decide(self, action: str) -> None:
        """action：keep_left / keep_right / merge_left / merge_right / both（都保留，以后不提示）/ skip。"""
        pair = self.current()
        if pair is None:
            return
        entries = self.by_id()
        left, right = entries[pair.left], entries[pair.right]
        if action == "both":
            self.ignored.add(qa_pair_key(left, right))
        elif action == "skip":
            pass          # 这次先不管，下次查重还会提示
        elif action in ("keep_left", "merge_left"):
            self._replace(merge_qa(left, right) if action == "merge_left" else left, drop=right.id)
        elif action in ("keep_right", "merge_right"):
            self._replace(merge_qa(right, left) if action == "merge_right" else right, drop=left.id)
        else:
            raise ValueError(action)
        self.done.add(frozenset((pair.left, pair.right)))

    def _replace(self, keep: QaEntry, *, drop: str) -> None:
        self.entries = [keep if e.id == keep.id else e for e in self.entries if e.id != drop]
        self.removed[drop] = keep.id

    def audio_to_delete(self) -> list[str]:
        """被删条目的英文回答里，已经没有任何保留条目在用的那些（这些声音可以删）。
        合并时保留条目的回答没变，所以它的声音不受影响。"""
        still_used = {e.answer.strip() for e in self.entries}
        answers = {self._original_answers[i] for i in self.removed if i in self._original_answers}
        return sorted(a for a in answers if a and a not in still_used)
