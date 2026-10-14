"""导入论文全文：提炼摘要 + 术语，并为助答建立原文索引。

做三件事：

1. **读全文。** 支持 PDF（需要 pypdf）、Word .docx（不需要额外依赖，直接读
   docx 里的 XML）、.txt / .md。末尾的参考文献会被切掉——它对术语和答题都是
   噪声，还白占上下文。
2. **一次调用提炼摘要和术语。** 用翻译那个快速模型（默认 qwen3.8-flash，
   上下文足够放下整篇硕士论文），关掉思考。术语按「中文说法 → 英文写法」给出，
   中文要写成答辩人**开口会说的样子**，因为它还要当语音识别的热词用。
3. **切块建索引。** 评委提问、问答库又没命中时，按问题检索最相关的几段原文，
   连同页码一起交给助答大模型：答案里的数字和结论以原文为准，界面上也能看到
   引用了哪几页。

全部结果只存在本机配置目录（``thesis_index.json``），不进仓库、不写日志。
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

from ..aliyun import ApiError, _extract_api_error
from .chat_models import chat_model, thinking_fields
from .qa_bank import tokenize

try:  # 顶层导入，PyInstaller 才能静态发现并打进 EXE
    import pypdf
except ImportError:  # pragma: no cover - 依赖缺失时给出安装提示
    pypdf = None

INDEX_FILE = "thesis_index.json"
MAX_THESIS_TERMS = 80
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
MAX_INPUT_CHARS = 300_000
SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")

_THESIS_INSTRUCTION = """你是一名学术口译顾问。下面是一位硕士研究生的学位论文全文（参考文献已删去）。
答辩现场会有语音识别和实时口译，请提取两样东西：

1. "brief"：用 6~10 句中文概括论文的研究问题、数据、方法、主要实验结论和局限，
   供现场口译员和助答系统理解背景。关键数字（误差、倍数等）要保留。
2. "terms"：论文里真正使用的领域术语中英对照表，最多 {max_terms} 条，按重要性从高到低排序
   （答辩中最可能说到的放前面）。要求：
   - "source" 写中文说法：答辩人**开口会怎么说**就怎么写（例如「漏磁」「磁偶极子模型」），
     英文缩写或模型名本身就是说法的，source 可以和 target 相同（例如 "MobileNetV3"）；
   - "target" 写论文里的英文写法，大小写和缩写与论文一致；
   - 只收领域术语、方法名、模型名、指标名，不收「深度」「长度」这类普通词。

只输出一个 JSON 对象，不要任何解释：
{{"brief": "...", "terms": [{{"source": "中文说法", "target": "English term"}}]}}"""


# ------------------------------------------------------------------ 读文件
@dataclass
class ThesisPage:
    number: int          # 从 1 开始；非 PDF 文件为 0
    text: str


def _normalize(text: str) -> str:
    # PDF 抽出来的文字每行都断开；合并成段，保留空行作为段落边界
    # PDF 里的数学符号常被抽成落单的代理字符，写文件和发请求都会出错
    text = re.sub("[\ud800-\udfff]", "", text)
    text = text.replace("\r", "\n").replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # 行尾连字符断词
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # 单个换行 = 同一段
    return text.strip()


def _pypdf():
    """程序启动时没有 pypdf、之后在界面里在线装上的，这里能重新拿到。"""
    global pypdf
    if pypdf is None:
        from .deps import load_pypdf
        pypdf = load_pypdf()
    return pypdf


def extract_pdf_pages(path: Path) -> list[ThesisPage]:
    module = _pypdf()
    if module is None:
        raise ApiError(
            "读取 PDF 需要 pypdf 组件（5.0 以上）：重新导入时按提示在线安装，"
            "或者改为导入论文的 Word 版（.docx，不需要额外安装）"
        )
    try:
        reader = module.PdfReader(str(path))
    except Exception as exc:
        raise ApiError(f"无法打开 PDF：{exc}") from exc
    pages: list[ThesisPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _normalize(text)
        if text:
            pages.append(ThesisPage(index, text))
    if not pages:
        raise ApiError("PDF 里没有可提取的文字（可能是扫描件）。请改用 Word 版或文字版 PDF")
    return pages


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx_text(path: Path) -> str:
    """不依赖 python-docx：直接读 word/document.xml 的段落文字。"""
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ApiError(f"无法读取 Word 文件：{exc}") from exc
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ApiError(f"Word 文件内容损坏：{exc}") from exc
    paragraphs: list[str] = []
    for para in root.iter(f"{_W}p"):
        pieces = []
        for node in para.iter():
            if node.tag == f"{_W}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_W}tab":
                pieces.append(" ")
        line = "".join(pieces).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def extract_thesis_pages(path: Path) -> list[ThesisPage]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(path)
    if suffix == ".docx":
        text = extract_docx_text(path)
    elif suffix in (".txt", ".md"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="replace")
        except OSError as exc:
            raise ApiError(f"无法读取文件：{exc}") from exc
    else:
        raise ApiError("论文支持 .pdf、.docx、.txt、.md；PPT 请用 .pptx")
    text = text.strip()
    if not text:
        raise ApiError("文件里没有可提取的文字")
    return [ThesisPage(0, text)]


_REFERENCE_HEADING = re.compile(r"(?:^|\n\n|\s)(References|REFERENCES|Bibliography|参考文献)\s*(?:\n|\[1\]|$)")


def strip_references(pages: list[ThesisPage]) -> list[ThesisPage]:
    """从后往前找「参考文献」标题，只在全文后 40% 里找，避免误删目录里的那一行。"""
    total = sum(len(p.text) for p in pages)
    if total == 0:
        return pages
    seen = 0
    cut_page = cut_offset = None
    for index, page in enumerate(pages):
        for match in _REFERENCE_HEADING.finditer(page.text):
            position = seen + match.start(1)
            if position >= total * 0.6:
                cut_page, cut_offset = index, match.start(1)
        seen += len(page.text)
    if cut_page is None:
        return pages
    kept = pages[:cut_page]
    head = pages[cut_page].text[:cut_offset].strip()
    if head:
        kept.append(ThesisPage(pages[cut_page].number, head))
    return kept


# ------------------------------------------------------------------ 切块与检索
@dataclass
class ThesisChunk:
    id: int
    page: int
    text: str


def chunk_pages(pages: list[ThesisPage], *, size: int = CHUNK_CHARS,
                overlap: int = CHUNK_OVERLAP) -> list[ThesisChunk]:
    """按页切，页内按句子凑到约 size 字，相邻块重叠一点，防止答案正好被切断。"""
    chunks: list[ThesisChunk] = []
    for page in pages:
        text = re.sub(r"\s+", " ", page.text).strip()
        sentences = re.split(r"(?<=[.!?。！？；;])\s+", text)
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > size:
                chunks.append(ThesisChunk(len(chunks), page.number, current.strip()))
                current = current[-overlap:] if overlap else ""
            current = f"{current} {sentence}".strip()
            while len(current) > size * 1.5:  # 超长无标点段落硬切
                chunks.append(ThesisChunk(len(chunks), page.number, current[:size]))
                current = current[size - overlap:]
        if current.strip():
            chunks.append(ThesisChunk(len(chunks), page.number, current.strip()))
    return chunks


class ThesisIndex:
    """BM25。分词沿用问答库的（英文词干 + 中文二字组），离线、毫秒级。"""

    def __init__(self, chunks: list[ThesisChunk]) -> None:
        self.chunks = list(chunks)
        self._docs = [self._counts(c.text) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: dict[str, int] = {}
        for doc in self._docs:
            for token in doc:
                df[token] = df.get(token, 0) + 1
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    @staticmethod
    def _counts(text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        lowered = text.lower()
        for token in tokenize(text):
            counts[token] = max(1, lowered.count(token)) if len(token) > 2 else 1
        return counts

    def search(self, query: str, limit: int = 3) -> list[tuple[ThesisChunk, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        k1, b = 1.5, 0.75
        scored = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if not tf:
                    continue
                norm = tf + k1 * (1 - b + b * length / (self._avg or 1))
                score += self._idf.get(term, 0.0) * tf * (k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


def excerpt_pairs(hits: list[tuple[ThesisChunk, float]]) -> list[tuple[str, str]]:
    """(出处标签, 原文) —— 交给助答提示词。"""
    return [(f"第 {c.page} 页" if c.page else f"第 {c.id + 1} 段", c.text) for c, _ in hits]


# ------------------------------------------------------------------ 本机存储
class ThesisStore:
    def __init__(self, base_dir: Path) -> None:
        self.path = Path(base_dir) / INDEX_FILE

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def load_index(self) -> ThesisIndex | None:
        data = self.load()
        chunks = []
        for item in data.get("chunks", []) or []:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                chunks.append(ThesisChunk(int(item.get("id", len(chunks))), int(item.get("page", 0) or 0),
                                          str(item["text"])))
        return ThesisIndex(chunks) if chunks else None

    def full_text(self) -> str:
        return str(self.load().get("full_text", "") or "")

    def save(self, *, source_path: str, chunks: list[ThesisChunk], page_count: int,
             full_text: str = "", brief: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": source_path,
            "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "page_count": page_count,
            "brief": brief,
            # 预习要把全文交给大模型，所以原文也存一份（只在本机）
            "full_text": full_text,
            "chunks": [asdict(c) for c in chunks],
        }
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------ 提炼
def input_char_limit(model: str) -> int:
    """按模型上下文保守估算能送多少字（中文约 1.5 字/token，英文更多）。"""
    info = chat_model(model)
    context_k = info.context_k if info is not None else 128
    return max(20_000, min(MAX_INPUT_CHARS, int(context_k * 1000 * 1.5)))


def build_thesis_payload(text: str, model: str) -> dict:
    limit = input_char_limit(model)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _THESIS_INSTRUCTION.format(max_terms=MAX_THESIS_TERMS)},
            {"role": "user", "content": text[:limit]},
        ],
        "temperature": 0.2,
    }
    # 提炼术语不需要「思考」，关掉能省几十秒
    payload.update(thinking_fields(model, want_thinking=False))
    return payload


def parse_thesis_response(content: str) -> tuple[str, list[dict[str, str]]]:
    raw = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ApiError("论文分析返回的不是 JSON，请重试")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ApiError(f"论文分析返回的 JSON 解析失败：{exc}") from exc
    brief = str(data.get("brief", "")).strip()
    if not brief:
        raise ApiError("论文分析没有返回摘要")
    terms: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data.get("terms") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and target and source not in seen:
            seen.add(source)
            terms.append({"source": source, "target": target})
    return brief, terms[:MAX_THESIS_TERMS]


def merge_terms(existing: list[dict[str, str]], new: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """已有的在前、不覆盖（那是你自己改过的），新的追加在后。返回 (合并结果, 新增条数)。"""
    def key(text: str) -> str:   # 忽略空格、标点、大小写：「漏磁 检测」和「漏磁检测」算同一个
        return re.sub(r"[\s\-_·.,，。()（）]+", "", str(text or "")).casefold()

    merged = [dict(t) for t in existing]
    sources = {key(t["source"]) for t in existing}
    added = 0
    for term in new:
        if key(term["source"]) not in sources:
            merged.append(dict(term))
            sources.add(key(term["source"]))
            added += 1
    return merged, added


def looks_glued(text: str) -> bool:
    """英文字母很多但空格极少 = 单词粘连（PDF 用定位代替空格时，旧版 pypdf 会这样）。"""
    sample = text[:20000]
    letters = sum(ch.isascii() and ch.isalpha() for ch in sample)
    if letters < 2000:
        return False
    return sample.count(" ") / letters < 0.08


@dataclass
class ThesisContext:
    brief: str = ""
    terms: list[dict[str, str]] = field(default_factory=list)
    chunks: list[ThesisChunk] = field(default_factory=list)
    page_count: int = 0
    char_count: int = 0
    source_path: str = ""
    full_text: str = ""


def import_thesis(
    client,
    path,
    *,
    model: str,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    post=None,
    on_file_error: Callable[[Path, Exception], None] | None = None,
) -> ThesisContext:
    """读论文 → 切块 → 一次大模型调用提炼摘要和术语。

    post(payload) -> 正文：界面里传 ContextTranslator.complete（流式，不怕长时间等待）；
    测试里传假的。不传时退回普通 POST。
    """
    notify = on_status or (lambda message: None)
    # 可以一次传多个文件（论文正文 + 附录 / 补充材料）：按顺序拼成一份全文
    paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
    pages: list[ThesisPage] = []
    imported_paths: list[Path] = []
    for index, one in enumerate(paths, start=1):
        notify(f"正在读取论文（{index}/{len(paths)}）：{one.name}…" if len(paths) > 1 else "正在读取论文…")
        try:
            file_pages = strip_references(extract_thesis_pages(one))
            if not any(page.text.strip() for page in file_pages):
                raise ApiError("文件里没有可提取的文字")
        except Exception as exc:
            if len(paths) == 1:
                raise
            if on_file_error is not None:
                on_file_error(one, exc)
            continue
        pages.extend(file_pages)
        imported_paths.append(one)
    full_text = "\n\n".join(p.text for p in pages)
    if not full_text.strip():
        raise ApiError("论文里没有可提取的文字")
    chunks = chunk_pages(pages)
    if looks_glued(full_text):
        notify("提示：这份 PDF 抽出来的英文单词粘在一起（旧版 pypdf 常见），术语和检索效果会变差；"
               "建议升级 pypdf 或改导入 Word 版")
    if cancel_event is not None and cancel_event.is_set():
        raise ApiError("已取消")
    notify(f"正在让 {model} 通读全文（约 {len(full_text) // 1000} 千字）并提炼术语，通常需要 20~60 秒…")
    payload = build_thesis_payload(full_text, model)
    if post is None:
        import requests

        endpoint = getattr(client, "endpoint", None) or f"{client.root}/compatible-mode/v1/chat/completions"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
            proxies=getattr(client, "proxies", None),
        )
        if not response.ok:
            raise ApiError(_extract_api_error(response))
        try:
            content = str(response.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            raise ApiError(f"论文分析返回格式异常：{response.text[:300]}") from exc
    else:
        content = post(payload)
    if cancel_event is not None and cancel_event.is_set():
        raise ApiError("已取消")
    brief, terms = parse_thesis_response(content)
    page_count = max((p.number for p in pages), default=0)
    return ThesisContext(brief=brief, terms=terms, chunks=chunks, page_count=page_count,
                         char_count=len(full_text), source_path="; ".join(str(p) for p in imported_paths),
                         full_text=full_text)
