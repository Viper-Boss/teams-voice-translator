"""从论文和答辩 PPT 里把**全部**术语找出来（不只是导入时那 40~80 条最重要的）。

做法：把全文按段切成约 12000 字一块，逐块让大模型列出这块里出现的所有领域术语；
每块都带上已有术语表，保证同一个概念前后用同一种英文写法。最后合并去重，
按「在几块里出现过」排序——出现越多越重要，排在前面（前 50 条同时作为语音识别热词）。

结果不直接写进术语表：先给你一张对照表，新增的默认勾上、和已有写法冲突的默认不勾，
你确认后再加入。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .chat_models import thinking_fields

CHUNK_CHARS = 12000
PER_CHUNK_MAX = 80
ASR_HOTWORD_COUNT = 50

_INSTRUCTION = """你是一名学术口译顾问，正在为一位硕士研究生的论文答辩准备「强制术语表」。
下面是论文或答辩 PPT 的一部分。请把这部分里出现的**所有**领域术语都列出来，最多 {max_terms} 条：
方法名、模型名、网络结构、算法、数据集、评价指标、物理量、设备 / 传感器、专有缩写都要收；
「深度」「长度」「方法」「结果」这类普通词不要收。

每条要求：
- "source"：中文说法——答辩人**开口会怎么说**就怎么写（例如「漏磁检测」「知识蒸馏」）；
  本来就用英文或缩写说的（例如 "MobileNetV3"、"RMSE"），source 和 target 写成一样；
- "target"：适合答辩直接朗读的英文写法，专有名称、大小写和论文一致；全称和常用缩写选一种，不要写成「全称 (缩写)」重复朗读；
  变量、单位或公式等有实际含义的括号要保留，例如 Defect depth (D)、Defect length (L)、Defect width (W)；
  并列词不用斜杠，例如 query, key, and value matrices；数学除法、单位和正式名称中的斜杠不要误删；
- 已有术语表里有的概念，英文必须沿用已有写法，不要另起一个；
- 按在这部分里的重要性从高到低排序。

只输出 JSON 数组，不要任何解释：
[{{"source": "中文说法", "target": "English term"}}]"""


def split_for_terms(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """按段落切块，尽量不把一段拦腰切开。"""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = ""
        while len(paragraph) > size:
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def build_terms_payload(chunk: str, model: str, existing: list[dict[str, str]], label: str = "") -> dict:
    known = "\n".join(f"  {t['source']} → {t['target']}" for t in existing[:300]) or "（暂无）"
    user = (f"已有术语表（英文写法以此为准）：\n{known}\n\n"
            f"===== {label or '论文片段'} =====\n{chunk}")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _INSTRUCTION.format(max_terms=PER_CHUNK_MAX)},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
    }
    payload.update(thinking_fields(model, want_thinking=False))
    return payload


def parse_terms(content: str) -> list[dict[str, str]]:
    """容错解析：代码块、前后废话、单引号都尽量救回来；坏条目跳过。"""
    raw = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    body = raw[start:end + 1]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # 截断或多了尾逗号：逐个对象抢救
        data = []
        for match in re.finditer(r"\{[^{}]*\}", body):
            try:
                data.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    terms: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        source = " ".join(str(item.get("source", "")).split())
        target = " ".join(str(item.get("target", "")).split())
        if source and target and len(source) <= 60 and len(target) <= 120:
            terms.append({"source": source, "target": target})
    return terms


@dataclass
class TermCandidate:
    source: str
    target: str
    hits: int = 1                     # 在几块里出现过
    existing_target: str = ""         # 已有术语表里同一中文的写法（冲突时非空）
    status: str = "new"               # new / conflict / same

    def as_term(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass
class GenerationResult:
    candidates: list[TermCandidate] = field(default_factory=list)
    chunks: int = 0
    failed_chunks: int = 0

    @property
    def new(self) -> list[TermCandidate]:
        return [c for c in self.candidates if c.status == "new"]

    @property
    def conflicts(self) -> list[TermCandidate]:
        return [c for c in self.candidates if c.status == "conflict"]

    @property
    def same(self) -> list[TermCandidate]:
        return [c for c in self.candidates if c.status == "same"]


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def aggregate(batches: list[list[dict[str, str]]], existing: list[dict[str, str]]) -> list[TermCandidate]:
    """合并各块结果：同一中文取出现最多的英文写法；和已有术语比对出新增 / 冲突 / 相同。"""
    order: list[str] = []
    variants: dict[str, dict[str, list]] = {}
    for batch in batches:
        seen_in_batch: set[str] = set()
        for term in batch:
            key = _key(term["source"])
            if not key:
                continue
            if key not in variants:
                variants[key] = {}
                order.append(key)
            bucket = variants[key].setdefault(term["target"], [term["source"], 0])
            if key not in seen_in_batch:
                bucket[1] += 1
            seen_in_batch.add(key)
    known = {_key(t["source"]): t["target"] for t in existing}
    known_targets = {_key(t["target"]) for t in existing}
    result: list[TermCandidate] = []
    for position, key in enumerate(order):
        target, (source, hits) = max(variants[key].items(), key=lambda kv: (kv[1][1], -len(kv[0])))
        hits = sum(h for _s, h in variants[key].values())
        candidate = TermCandidate(source=source, target=target, hits=hits)
        if key in known:
            candidate.existing_target = known[key]
            candidate.status = "same" if _key(known[key]) == _key(target) else "conflict"
        elif _key(target) in known_targets and _key(source) == _key(target):
            candidate.status = "same"    # 纯英文术语，已有表里有同样的英文
        result.append((position, candidate))  # type: ignore[arg-type]
    result.sort(key=lambda item: (-item[1].hits, item[0]))  # type: ignore[index]
    return [candidate for _pos, candidate in result]  # type: ignore[misc]


def generate_terms(sources: list[tuple[str, str]], existing: list[dict[str, str]], *, model: str,
                   post: Callable[[dict], str], on_progress: Callable[[str], None] | None = None,
                   cancel_event=None) -> GenerationResult:
    """sources = [(名称, 全文), ...]；post(payload) -> 模型返回的正文（调用方负责网络和超时）。"""
    notify = on_progress or (lambda _m: None)
    jobs: list[tuple[str, str]] = []
    for label, text in sources:
        pieces = split_for_terms(text)
        for number, piece in enumerate(pieces, start=1):
            jobs.append((f"{label} 第 {number}/{len(pieces)} 块", piece))
    result = GenerationResult(chunks=len(jobs))
    batches: list[list[dict[str, str]]] = []
    running = [dict(t) for t in existing]
    for index, (label, piece) in enumerate(jobs, start=1):
        if cancel_event is not None and cancel_event.is_set():
            break
        notify(f"正在找术语：{index}/{len(jobs)}（{label}）…")
        try:
            terms = parse_terms(post(build_terms_payload(piece, model, running, label)))
        except Exception:  # noqa: BLE001 —— 一块失败不影响其他块
            result.failed_chunks += 1
            continue
        batches.append(terms)
        known = {_key(t["source"]) for t in running}
        for term in terms:   # 后面的块沿用前面找到的写法
            if _key(term["source"]) not in known:
                running.append(term)
                known.add(_key(term["source"]))
    result.candidates = aggregate(batches, existing)
    return result
