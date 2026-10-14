"""讲稿朗读台的数据模型（纯逻辑，不依赖 Qt）。

用途：答辩开场要讲 20 分钟 PPT。这段内容是提前写好的，不该依赖现场实时翻译——
网络抖一下、接口抽一下，人就卡在台上了。所以这里的做法是：

    整篇讲稿 → 按换行切成「段」→ 段内切成「句」→ 提前逐句渲染到本地 → 现场点播

**段是一级单位，一段对应一页 PPT。** 用户在原文里用换行划分段落，
讲到哪一页就播哪一段；句是段内的朗读与光标粒度。

和「讲稿预习」的区别：那个只能输入中文、必须先整段翻译、按播放键从头念到尾。
这个中英文都能直接导入、按段/按句点播、可以跳着讲，而且渲染过的段断网也能出声。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 常见缩写，后面的句点不算句末。
_ABBREVIATIONS = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|cf|Fig|Figs|Eq|Eqs|No|Dr|Mr|Mrs|Ms|Prof|St|al)\.$",
    re.IGNORECASE,
)

LANGUAGE_LABELS = {"zh": "中文", "en": "英文"}

# 中文句末标点自带边界，不需要后面跟空格；英文句点必须谨慎。
_CJK_END = "。！？；…"
_CLOSERS = "”’\"')）】》"


def detect_language(text: str) -> str:
    """按汉字比例粗判语种。只用于选 TTS 的 language_hint，判错也不致命。"""
    stripped = "".join(text.split())
    if not stripped:
        return "en"
    cjk = sum(1 for char in stripped if "一" <= char <= "鿿")
    return "zh" if cjk * 4 >= len(stripped) else "en"


def split_paragraphs(text: str) -> list[str]:
    """按换行切段。**一个换行就是一段**，连续空行忽略。

    用户是靠敲回车来划分 PPT 页的，所以不能要求必须空一行——
    那样每分一页都得按两下回车，和他们的实际写法不符。
    """
    blocks = []
    for raw in re.split(r"[\r\n]+", text or ""):
        block = " ".join(raw.split())
        if block:
            blocks.append(block)
    return blocks


def split_sentences(text: str, *, max_chars: int = 400) -> list[str]:
    """把一段切成可以逐句点读的句子。

    超长的句子（有人整段不打标点）按 max_chars 硬切，否则一句要合成半分钟。
    """
    if max_chars < 1:
        raise ValueError("单句字符上限必须为正整数")
    sentences: list[str] = []
    for block in split_paragraphs(text):
        for piece in _split_block(block):
            sentences.extend(_hard_wrap(piece, max_chars))
    return sentences


def _split_block(block: str) -> list[str]:
    """逐字符扫描找句末。

    不能按空白切词再判断——中文整段没有空格，那样整块会变成一个 token。
    """
    pieces: list[str] = []
    start = 0
    position = 0
    length = len(block)
    while position < length:
        char = block[position]
        if char in _CJK_END or char in "!?;":
            end = position + 1
            while end < length and block[end] in _CLOSERS:
                end += 1
            # 英文 ! ? ; 后面要跟空白或结尾才算句末，"a;b" 不切
            if char in "!?;" and end < length and not block[end].isspace():
                position += 1
                continue
            pieces.append(block[start:end].strip())
            start = position = end
            continue
        if char == ".":
            end = position + 1
            while end < length and block[end] in _CLOSERS:
                end += 1
            tail = block[start:end].strip()
            following = block[end:end + 2]
            is_decimal = position + 1 < length and block[position + 1].isdigit()
            ends_sentence = end >= length or (following[:1].isspace() and not is_decimal)
            if ends_sentence and not _ABBREVIATIONS.search(tail) and not is_decimal:
                pieces.append(tail)
                start = position = end
                continue
        position += 1
    if block[start:].strip():
        pieces.append(block[start:].strip())
    return [piece for piece in pieces if piece]


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    out: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


@dataclass
class ScriptLine:
    """段里的一句，以及它的渲染状态。"""

    index: int            # 全篇顺序号，方便「全部朗读」直接走一条线
    paragraph: int        # 属于第几段
    order: int            # 段内第几句
    text: str
    language: str = "en"
    rendered: bool = False
    seconds: float = 0.0
    error: str = ""
    stale: bool = False   # 同一音色、旧参数（语速/音量/语气/小失误）渲染的：能播，可选重渲染
    custom: str = ""      # 右键单独重做 / 换语气后的版本名，优先于整段里的这句

    @property
    def status_text(self) -> str:
        if self.rendered and self.custom:
            return f"✓ {self.seconds:.1f}s（单独·{self.custom}）"
        if self.rendered and self.stale:
            return f"✓ {self.seconds:.1f}s（旧设置）"
        if self.error:
            return f"✗ {self.error}"
        if self.rendered:
            return f"✓ {self.seconds:.1f}s"
        return "· 未渲染"

    @property
    def preview(self) -> str:
        return self.text if len(self.text) <= 80 else self.text[:79] + "…"


@dataclass
class ScriptParagraph:
    """一段 = 一页 PPT。"""

    index: int
    lines: list[ScriptLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    @property
    def title(self) -> str:
        head = self.text
        return head if len(head) <= 42 else head[:41] + "…"

    @property
    def rendered_count(self) -> int:
        return sum(1 for line in self.lines if line.rendered)

    @property
    def seconds(self) -> float:
        return sum(line.seconds for line in self.lines)

    @property
    def ready(self) -> bool:
        """整段都渲染好了才算能上台播——缺一句中间就会断。"""
        return bool(self.lines) and all(line.rendered for line in self.lines)

    @property
    def status_text(self) -> str:
        if not self.lines:
            return "空段"
        failed = sum(1 for line in self.lines if line.error)
        if failed:
            return f"✗ {failed}/{len(self.lines)} 句失败"
        if self.ready:
            return f"✓ 就绪 {self.seconds:.0f}s"
        return f"{self.rendered_count}/{len(self.lines)} 句"


@dataclass
class ScriptDocument:
    """一篇讲稿。语种整篇统一，不做逐句混排——混排会让音色在中英间跳。"""

    language: str = "en"
    paragraphs: list[ScriptParagraph] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, *, language: str = "", max_chars: int = 400) -> "ScriptDocument":
        language = language or detect_language(text)
        paragraphs: list[ScriptParagraph] = []
        running = 0
        for para_index, block in enumerate(split_paragraphs(text)):
            lines: list[ScriptLine] = []
            for order, piece in enumerate(split_sentences(block, max_chars=max_chars)):
                lines.append(ScriptLine(
                    index=running, paragraph=para_index, order=order,
                    text=piece, language=language,
                ))
                running += 1
            if lines:
                paragraphs.append(ScriptParagraph(index=len(paragraphs), lines=lines))
        # 段号可能因为空段被跳过，重排一遍保证 index == 下标
        for position, paragraph in enumerate(paragraphs):
            paragraph.index = position
            for line in paragraph.lines:
                line.paragraph = position
        return cls(language=language, paragraphs=paragraphs)

    @property
    def lines(self) -> list[ScriptLine]:
        """全篇句子，按讲述顺序。"""
        return [line for paragraph in self.paragraphs for line in paragraph.lines]

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def rendered_count(self) -> int:
        return sum(1 for line in self.lines if line.rendered)

    @property
    def failed_count(self) -> int:
        return sum(1 for line in self.lines if line.error)

    @property
    def total_seconds(self) -> float:
        return sum(line.seconds for line in self.lines)

    def paragraph_at(self, index: int) -> ScriptParagraph | None:
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None

    def line_at(self, index: int) -> ScriptLine | None:
        lines = self.lines
        if 0 <= index < len(lines):
            return lines[index]
        return None

    def pending(self) -> list[ScriptLine]:
        """还需要渲染的句子：没渲染过、且上次没失败过的先来。

        失败过的排在后面，这样一次「全部渲染」不会被同一个坏句子反复拖住。
        """
        lines = self.lines
        fresh = [line for line in lines if not line.rendered and not line.error]
        retry = [line for line in lines if not line.rendered and line.error]
        return fresh + retry

    def progress_text(self) -> str:
        if not self.paragraphs:
            return "还没有导入讲稿"
        parts = [f"{len(self.paragraphs)} 段 / {len(self)} 句", f"已渲染 {self.rendered_count}"]
        if self.failed_count:
            parts.append(f"失败 {self.failed_count}")
        if self.rendered_count:
            minutes, seconds = divmod(int(self.total_seconds), 60)
            parts.append(f"已渲染时长 {minutes}分{seconds:02d}秒")
        stale = sum(1 for line in self.lines if line.rendered and line.stale)
        if stale:
            parts.append(f"{stale} 句是旧设置渲染的（照样能播）")
        if self.rendered_count == len(self) and len(self):
            parts.append("已存本机，重开软件直接用，现场可离线朗读")
        return " · ".join(parts)


def cumulative_bounds(lines) -> list[tuple[float, float]]:
    """每句在拼接音频里的 [起, 止] 秒。光标跟随就是靠它定位的。"""
    bounds: list[tuple[float, float]] = []
    clock = 0.0
    for line in lines:
        bounds.append((clock, clock + line.seconds))
        clock += line.seconds
    return bounds


def line_at_elapsed(bounds, elapsed: float) -> int:
    """播到第 elapsed 秒时光标应该停在第几句。越界钳到两端。"""
    if not bounds:
        return -1
    if elapsed <= bounds[0][0]:
        return 0
    for index, (start, end) in enumerate(bounds):
        if start <= elapsed < end:
            return index
    return len(bounds) - 1


def split_pcm_by_text(pcm: bytes, texts, *, sample_rate: int = 24000,
                      search_seconds: float = 0.45) -> list[bytes]:
    """把整段合成的音频按句切开，用于逐句光标、单句点播和进度显示。

    整段一起合成（和「声音对比」B 一样）韵律最连贯，但只有一整块音频。先按字数
    比例估出每句的边界，再在估计点前后 search_seconds 内找最安静的 20 ms 当切点——
    真人说话句与句之间有停顿，切在停顿里就不会切断一个词。切出来的各块首尾相接
    正好等于原音频，所以连着播和整段播放听起来完全一样。
    """
    from array import array

    texts = [str(text or "") for text in texts]
    if not texts:
        return []
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if len(texts) == 1:
        return [pcm]
    total = len(pcm) // 2
    samples = array("h")
    samples.frombytes(pcm)
    weights = [len(text.strip()) + 8 for text in texts]   # +8：每句句尾的停顿也占时间
    whole = float(sum(weights)) or 1.0
    frame = max(1, int(sample_rate * 0.02))
    window = int(sample_rate * search_seconds)
    minimum = int(sample_rate * 0.2)
    cuts: list[int] = []
    running = 0
    previous = 0
    for weight in weights[:-1]:
        running += weight
        estimate = int(total * running / whole)
        low = max(previous + minimum, estimate - window)
        high = min(total - minimum, estimate + window)
        if high <= low:
            cut = min(max(estimate, previous), total)
        else:
            best, best_score = estimate, None
            position = low
            while position + frame <= high:
                chunk = samples[position:position + frame]
                energy = sum(abs(value) for value in chunk) / frame
                # 能量最低优先；差不多安静时选离估计点近的
                score = energy + abs(position + frame // 2 - estimate) * 0.002
                if best_score is None or score < best_score:
                    best_score, best = score, position + frame // 2
                position += frame // 2
            cut = best
        cut = max(previous, min(cut, total))
        cuts.append(cut)
        previous = cut
    bounds = [0] + cuts + [total]
    return [pcm[bounds[i] * 2:bounds[i + 1] * 2] for i in range(len(texts))]
