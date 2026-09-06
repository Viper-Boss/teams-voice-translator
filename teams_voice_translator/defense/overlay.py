from __future__ import annotations

import html
import re
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFrame, QWidget

HIGHLIGHT_COLOR = "#FFD400"
HIGHLIGHT_BG = "rgba(255,212,0,0.22)"
ZH_COLOR = "#c9d2e3"
EN_COLOR = "#f4f6fb"
CARD_BG = "rgba(10,13,20,205)"
IDLE_HIDE_SECONDS = 3.0
TICK_MS = 60


def token_chunks(text: str) -> list[str]:
    """Split into word chunks while keeping the separating whitespace."""
    return re.findall(r"\S+\s*", text)


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
        self.zh_label.setStyleSheet(f"color: {ZH_COLOR}; font-size: 17px;")
        layout.addWidget(self.zh_label)

        self.en_label = QLabel("")
        self.en_label.setAlignment(Qt.AlignCenter)
        self.en_label.setWordWrap(True)
        self.en_label.setStyleSheet(f"color: {EN_COLOR}; font-size: 25px; font-weight: 600;")
        layout.addWidget(self.en_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        self.setFixedSize(940, 128)

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._restore_position()

    # -------------------------------------------------------------- content
    def show_sentence(self, chinese: str, english: str, duration: float = 0.0) -> None:
        if not english.strip():
            return
        self.zh_label.setText(chinese.strip())
        self._chunks = token_chunks(english.strip())
        self._duration = max(0.0, float(duration))
        self._t0 = time.monotonic()
        self._last_activity = time.monotonic()
        self.en_label.setText(html.escape(english.strip()))
        if not self.isVisible():
            self.show()

    def set_duration(self, english: str, seconds: float) -> None:
        if english.strip() == "".join(self._chunks).strip():
            self._duration = max(0.0, float(seconds))
            self._last_activity = time.monotonic()

    def show_progress(self, chinese: str, english: str, progress: float) -> None:
        """Externally paced rehearsal subtitles; pause leaves the cursor still."""
        self._t0 = None
        self.zh_label.setText(chinese)
        self._chunks = token_chunks(english)
        self.en_label.setText(highlight_html(self._chunks, current_token_index(self._chunks, progress)))
        # Long sentences wrap and grow upwards from the existing bottom edge.
        bottom = self.y() + self.height()
        width = self.width() - 44
        height = max(128, self.zh_label.heightForWidth(width) + self.en_label.heightForWidth(width) + 42)
        self.setFixedHeight(height)
        self.move(self.x(), bottom - height)
        self.show()

    def stop(self) -> None:
        """Esc/打断：立刻熄灭高亮并隐藏。"""
        self._t0 = None
        self._duration = 0.0
        self._chunks = []
        self.en_label.setText("")
        self.zh_label.setText("")
        self.hide()
        self._save_position()

    def _tick(self) -> None:
        if not self.isVisible() or self._t0 is None:
            return
        now = time.monotonic()
        if self._duration > 0:
            progress = (now - self._t0) / self._duration
            index = current_token_index(self._chunks, progress)
            self.en_label.setText(highlight_html(self._chunks, index))
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
