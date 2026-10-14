from __future__ import annotations

import html
import re
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QTextDocument
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFrame, QWidget

HIGHLIGHT_COLOR = "#FFD400"
HIGHLIGHT_BG = "rgba(255,212,0,0.22)"
ZH_COLOR = "#c9d2e3"
EN_COLOR = "#f4f6fb"
CARD_BG = "rgba(10,13,20,205)"
IDLE_HIDE_SECONDS = 3.0
TICK_MS = 60
# 字幕最高占屏幕多少：再高就挡住 PPT 了。超过这个高度改为分页跟随。
MAX_HEIGHT_RATIO = 0.42
MIN_HEIGHT = 128
# 英文一屏最多排几行；超出就翻页，保证正在读的那个词永远看得见。
MAX_EN_LINES = 3


def paginate(chunks: list[str], max_chars: int) -> list[tuple[int, int]]:
    """把词块按「一屏能放下多少字符」切成若干页，返回每页的 [start, end)。

    分页而不是像素滚动，是因为 QLabel 拿不到富文本里某个词的坐标。
    按页走还有个好处：视线落点稳定，不会随着每个词往前挪一点点而抖。
    """
    if not chunks:
        return []
    if max_chars < 1:
        raise ValueError("每页字符数必须为正整数")
    pages: list[tuple[int, int]] = []
    start = 0
    length = 0
    for index, chunk in enumerate(chunks):
        size = len(chunk)
        # 单个词就超长时让它自己占一页，否则会死循环。
        if length and length + size > max_chars:
            pages.append((start, index))
            start, length = index, 0
        length += size
    pages.append((start, len(chunks)))
    return pages


def page_for_index(pages: list[tuple[int, int]], index: int) -> int:
    """当前高亮词落在第几页。越界时钳到首/末页，不抛。"""
    if not pages:
        return 0
    for position, (start, end) in enumerate(pages):
        if start <= index < end:
            return position
    return len(pages) - 1 if index >= pages[-1][1] else 0


def token_chunks(text: str) -> list[str]:
    """Split into word chunks while keeping the separating whitespace."""
    return re.findall(r"[\u3400-\u9fff]\s*|[^\s\u3400-\u9fff]+\s*", text)


def chunk_weights(chunks: list[str]) -> list[int]:
    # Spoken length ≈ visible characters; +1 models the natural micro-pause
    # between words so timing stays proportional.
    return [len(chunk.strip()) + 1 for chunk in chunks]


def current_token_index(chunks: list[str], progress: float) -> int:
    if not chunks:
        return -1
    weights = chunk_weights(chunks)
    total = sum(weights)
    progress = min(max(progress, 0.0), 1.0)
    elapsed_target = progress * total
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if elapsed_target < cumulative:
            return index
    return len(chunks) - 1


def highlight_html(chunks: list[str], index: int) -> str:
    parts: list[str] = []
    for position, chunk in enumerate(chunks):
        escaped = html.escape(chunk.rstrip()) + (" " if chunk.endswith(" ") else "")
        if position == index:
            parts.append(
                f"<span style='color:{HIGHLIGHT_COLOR}; background-color:{HIGHLIGHT_BG};'>{escaped}</span>"
            )
        else:
            parts.append(escaped)
    return "".join(parts).replace("  ", " ")


class SubtitleOverlay(QWidget):
    """歌词式双语字幕悬浮窗：中文原句 + 英文合成朗读的逐词黄色高亮。"""

    def __init__(self, settings) -> None:
        super().__init__(None)
        self.settings = settings
        self.setWindowTitle("答辩字幕")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._drag_offset = None
        self._chunks: list[str] = []
        self._t0: float | None = None
        self._duration = 0.0
        self._last_activity = 0.0
        self._position_callback = None
        self._paused_at: float | None = None
        self._estimated = False
        self._page_cache = {}
        self._chinese = ""

        card = QFrame(self)
        card.setObjectName("subtitleCard")
        card.setStyleSheet(
            f"#subtitleCard {{ background: {CARD_BG}; border-radius: 14px; }}"
            "QLabel { background: transparent; border: none; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 12, 22, 14)
        layout.setSpacing(4)

        self.zh_label = QLabel("")
        self.zh_label.setAlignment(Qt.AlignCenter)
        self.zh_label.setWordWrap(True)
        self.zh_label.setTextFormat(Qt.PlainText)
        self.zh_label.setStyleSheet(f"color: {ZH_COLOR}; font-size: 17px;")
        layout.addWidget(self.zh_label)

        self.en_label = QLabel("")
        self.en_label.setAlignment(Qt.AlignCenter)
        self.en_label.setWordWrap(True)
        self.en_label.setTextFormat(Qt.RichText)
        self.en_label.setStyleSheet(f"color: {EN_COLOR}; font-size: 25px; font-weight: 600;")
        layout.addWidget(self.en_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        # 宽度固定，高度随内容变（_fit_height 会改），所以不能用 setFixedSize。
        self.setFixedWidth(940)
        self.setFixedHeight(MIN_HEIGHT)

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._restore_position()

    # -------------------------------------------------------------- content
    # ------------------------------------------------------------- 排版
    def _chars_per_page(self) -> int:
        """按当前字体实测一屏能放多少字符，而不是写死一个数。

        字号、DPI 缩放、窗口宽度都会变，写死的话在 150% 缩放下就不准了。
        """
        metrics = self.en_label.fontMetrics()
        char_width = max(1, metrics.averageCharWidth())
        per_line = max(8, (self.width() - 44) // char_width)
        return int(per_line * MAX_EN_LINES)

    def _render_page(self, index: int) -> None:
        """只渲染 index 所在的那一页，并把高亮位置换算成页内下标。"""
        if not self._chunks:
            self.en_label.setText("")
            return
        self.ensurePolished()
        pages = self._pixel_pages(self._chunks, self.en_label)
        start, end = pages[page_for_index(pages, index)]
        self.en_label.setText(highlight_html(self._chunks[start:end], index - start))
        chinese = token_chunks(self._chinese)
        if chinese:
            zh_pages = self._pixel_pages(chinese, self.zh_label)
            page = min(len(zh_pages) - 1, int(max(index, 0) / max(1, len(self._chunks)) * len(zh_pages)))
            first, last = zh_pages[page]
            self.zh_label.setText("".join(chinese[first:last]))
        else:
            self.zh_label.clear()
        self._fit_height()

    def _measure(self, label, text, *, rich=False):
        label.ensurePolished()
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(label.font())
        doc.setTextWidth(max(1, self.width() - 48))
        if rich:
            doc.setHtml(text)
        else:
            doc.setPlainText(text)
        return doc.size().height()

    def _pixel_pages(self, chunks, label):
        key = (tuple(chunks), self.width(), label.font().toString())
        if key in self._page_cache:
            return self._page_cache[key]
        limit = label.fontMetrics().lineSpacing() * MAX_EN_LINES + 2
        pages, start = [], 0
        while start < len(chunks):
            low, high = start + 1, len(chunks)
            end = low
            while low <= high:
                middle = (low + high) // 2
                if self._measure(label, "".join(chunks[start:middle])) <= limit:
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            pages.append((start, end))
            start = end
        if len(self._page_cache) > 8:
            self._page_cache.clear()
        self._page_cache[key] = pages
        return pages

    def _fit_height(self) -> None:
        """按内容长高，底边不动（从下往上长），并且封顶。

        原来 show_sentence 完全不改尺寸，构造时 setFixedSize(940,128) 之后
        长句直接被裁掉 —— 这就是「长文本字幕不滚动」的真正原因。
        """
        bottom = self.y() + self.height()
        self.zh_label.setVisible(bool(self.zh_label.text()))
        zh_height = int(self._measure(self.zh_label, self.zh_label.text()) + 6) if self.zh_label.text() else 0
        en_height = int(self._measure(self.en_label, self.en_label.text(), rich=True) + 8)
        self.zh_label.setFixedHeight(zh_height)
        self.en_label.setFixedHeight(en_height)
        height = max(MIN_HEIGHT, zh_height + en_height + 34)
        if height != self.height():
            self.setFixedHeight(height)
            self.move(self.x(), bottom - height)
        self._clamp_position()

    def _clamp_position(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(max(area.left(), min(self.x(), area.right() - self.width() + 1)),
                  max(area.top(), min(self.y(), area.bottom() - self.height() + 1)))

    # -------------------------------------------------------------- content
    def show_sentence(self, chinese: str, english: str, duration: float = 0.0) -> None:
        if not english.strip():
            return
        self._chinese = chinese.strip()
        self._chunks = token_chunks(english.strip())
        self._duration = max(0.0, float(duration))
        # 时长还没到（现场合成的要等音频齐了才知道）：先按语速估一个，高亮照样走，
        # 也不会因为「长时间没进度」被当成空闲自动隐藏。真实时长一到就替换。
        self._estimated = self._duration <= 0
        if self._estimated:
            self._duration = max(1.5, len(english.split()) / 2.6)
        self._paused_at = None
        self._t0 = time.monotonic()
        self._last_activity = time.monotonic()
        self._render_page(0)
        if not self.isVisible():
            self.show()

    def set_duration(self, english: str, seconds: float) -> None:
        # 比较时忽略空白差异：以前差一个空格就对不上，时长一直是 0，高亮不动、字幕过会儿自己消失
        if " ".join(english.split()) == " ".join("".join(self._chunks).split()):
            self._duration = max(0.0, float(seconds))
            self._estimated = False
            self._last_activity = time.monotonic()

    def pause(self) -> None:
        """暂停朗读：高亮停在当前词。"""
        if self._chunks and self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self._paused_at is not None and self._t0 is not None:
            self._t0 += time.monotonic() - self._paused_at
        self._paused_at = None
        self._last_activity = time.monotonic()

    def show_progress(self, chinese: str, english: str, progress: float) -> None:
        """Externally paced rehearsal subtitles; pause leaves the cursor still."""
        self._t0 = None
        self._chinese = chinese
        self._chunks = token_chunks(english)
        self._last_activity = time.monotonic()
        self._render_page(current_token_index(self._chunks, progress))
        self.show()

    def update_playback_progress(self, english: str, progress: float) -> None:
        if (self.isVisible() and self._chunks
                and " ".join(english.split()) == " ".join("".join(self._chunks).split())):
            self._t0 = None
            self._last_activity = time.monotonic()
            self._render_page(current_token_index(self._chunks, progress))

    def stop(self) -> None:
        """Esc/打断：立刻熄灭高亮并隐藏。"""
        self._t0 = None
        self._paused_at = None
        self._duration = 0.0
        self._chunks = []
        self._chinese = ""
        self.en_label.setText("")
        self.zh_label.setText("")
        self.hide()
        self._save_position()

    def _tick(self) -> None:
        if not self.isVisible():
            return
        now = time.monotonic()
        if self._paused_at is not None:
            self._last_activity = now   # 暂停中不算空闲，不自动隐藏
            return
        if self._t0 is None:
            if now - self._last_activity > IDLE_HIDE_SECONDS:
                self.hide()
            return
        if self._duration > 0:
            progress = (now - self._t0) / self._duration
            index = current_token_index(self._chunks, progress)
            # 跟随高亮翻页：正在读的那个词永远在可视区内。
            self._render_page(index)
            if progress >= 1.0 and self._estimated and progress < 3.0:
                return   # 估的时长到了但真实时长还没来：停在最后一个词等一等
            if progress >= 1.0:
                self._t0 = None
                self._last_activity = now
                return
        self._last_activity = max(self._last_activity, now)
        if now - self._last_activity > IDLE_HIDE_SECONDS:
            self.hide()
            self._save_position()

    # ------------------------------------------------------------ dragging
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            self._save_position()

    def mouseDoubleClickEvent(self, _event) -> None:
        self.stop()

    # ------------------------------------------------------------ position
    def set_position_callback(self, callback) -> None:
        self._position_callback = callback

    def place_at_bottom(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - self.height() - 90
        saved = self._saved_position()
        if saved is not None:
            x, y = saved
        self.move(x, y)
        self._clamp_position()

    def _saved_position(self) -> tuple[int, int] | None:
        x = self.settings.shared.get("overlay_x")
        y = self.settings.shared.get("overlay_y")
        if isinstance(x, int) and isinstance(y, int):
            return x, y
        return None

    def _restore_position(self) -> None:
        self.place_at_bottom()

    def _save_position(self) -> None:
        if not self.isVisible() and self._saved_position() is not None:
            return
        try:
            self.settings.shared.update({"overlay_x": self.x(), "overlay_y": self.y()})
        except Exception:
            pass
        if self._position_callback is not None:
            try:
                self._position_callback(self.x(), self.y())
            except Exception:
                pass
