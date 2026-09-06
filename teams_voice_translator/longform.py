from __future__ import annotations

import html
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .audio import MultiOutputPlayer


_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_DISPLAY_TOKEN_RE = re.compile(
    r"\s+|[\u3400-\u9fff]|[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|.",
    re.DOTALL,
)


@dataclass
class BilingualSegment:
    chinese: str
    english: str
    estimated_seconds: float
    actual_seconds: float | None = None


def parse_pronunciation_dictionary(raw: str) -> dict[str, str]:
    """Parse display-text -> spoken-text replacements without changing captions."""
    if not str(raw or "").strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"发音词典不是有效 JSON：{exc}") from exc
    if isinstance(data, dict):
        pairs = data.items()
    elif isinstance(data, list):
        pairs = (
            (item.get("text", item.get("source", "")), item.get("spoken", item.get("target", "")))
            for item in data
            if isinstance(item, dict)
        )
    else:
        raise ValueError("发音词典必须是 JSON 对象或对象数组")
    result: dict[str, str] = {}
    for source, spoken in pairs:
        source, spoken = str(source).strip(), str(spoken).strip()
        if source and spoken:
            result[source] = spoken
    return result


def apply_pronunciation_dictionary(text: str, raw: str) -> str:
    """Return TTS-only text with longest dictionary entries replaced first."""
    replacements = parse_pronunciation_dictionary(raw)
    if not replacements:
        return text
    pattern = re.compile(
        "|".join(re.escape(source) for source in sorted(replacements, key=len, reverse=True))
    )
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def split_source_sentences(text: str, max_len: int = 260) -> list[str]:
    """Split Chinese presentation text while retaining sentence punctuation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    for paragraph in normalized.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidates = [match.group(0).strip() for match in _SENTENCE_RE.finditer(paragraph)]
        if not candidates:
            candidates = [paragraph]
        for candidate in candidates:
            while len(candidate) > max_len:
                cut = max(
                    candidate.rfind(mark, 0, max_len)
                    for mark in ("，", "、", ",", " ")
                )
                if cut < max_len // 3:
                    cut = max_len
                else:
                    cut += 1
                head, candidate = candidate[:cut].strip(), candidate[cut:].strip()
                if head:
                    result.append(head)
            if candidate:
                result.append(candidate)
    return result


def estimate_speech_seconds(text: str, rate: float = 1.0) -> float:
    """Estimate English speech duration for the pre-playback timeline."""
    words = len(_ENGLISH_WORD_RE.findall(text))
    punctuation_pauses = len(re.findall(r"[,;:，；：.!?。！？]", text)) * 0.14
    effective_rate = max(0.5, min(2.0, float(rate or 1.0)))
    return max(1.2, (words / (2.55 * effective_rate)) + punctuation_pauses + 0.35)


def format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def build_long_form_translation_payload(
    full_text: str,
    sentences: list[str],
    settings: dict[str, Any],
    *,
    terms: list[dict[str, str]],
    memories: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a context-first, strictly aligned long-form translation request."""
    style_names = {
        "polite": "natural and polite classroom speech",
        "concise": "concise spoken English",
        "academic": "clear formal academic English",
        "literal": "faithful translation with minimal paraphrasing",
    }
    domain = str(settings.get("translation_domain", "")).strip()
    style = style_names.get(str(settings.get("translation_style", "")), "natural spoken English")
    request = {
        "document_context": full_text,
        "domain": domain,
        "style": style,
        "required_terms": terms,
        "translation_memory": memories,
        "sentences": [
            {"index": index, "chinese": sentence}
            for index, sentence in enumerate(sentences, start=1)
        ],
    }
    system = (
        "You are a senior Chinese-English simultaneous-interpreting editor. "
        "Read the entire document before translating. Keep terminology, pronouns, tense, "
        "names and argument flow consistent across all sentences. Prefer natural spoken "
        "English suitable for an online lesson, never translate each sentence in isolation. "
        "Obey required_terms exactly when applicable and use translation_memory as examples. "
        "Return JSON only in this exact shape: "
        '{"segments":[{"index":1,"english":"..."}]}. '
        "Return exactly one non-empty English segment for every input index, in order. "
        "Do not merge, split, omit, explain, or include Markdown."
    )
    return {
        "model": str(
            settings.get("long_form_model") or settings.get("summary_model") or "qwen-plus"
        ).strip(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        "temperature": 0.15,
    }


def build_long_form_context_payload(
    full_text: str,
    settings: dict[str, Any],
    *,
    terms: list[dict[str, str]],
    memories: list[dict[str, str]],
) -> dict[str, Any]:
    """Create a compact translation brief before batching a very long script."""
    request = {
        "document": full_text,
        "domain": str(settings.get("translation_domain", "")).strip(),
        "required_terms": terms,
        "translation_memory": memories,
    }
    return {
        "model": str(
            settings.get("long_form_model") or settings.get("summary_model") or "qwen-plus"
        ).strip(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Read the complete Chinese lesson script and produce a compact English "
                    "translation brief. Preserve names, terminology, referents, tense and the "
                    "argument sequence. Return JSON only with keys summary, terminology, "
                    "entities, style_notes and continuity_notes. Do not translate every sentence."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }


def parse_long_form_translation(content: str, sentence_count: int) -> list[str]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("整段翻译没有返回可解析的 JSON")
    data = json.loads(cleaned[start : end + 1])
    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw_segments, list):
        raise ValueError("整段翻译缺少 segments 数组")
    by_index: dict[int, str] = {}
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        english = str(item.get("english", "")).strip()
        if english:
            by_index[index] = english
    missing = [index for index in range(1, sentence_count + 1) if index not in by_index]
    if missing:
        raise ValueError(f"整段翻译句子没有一一对齐，缺少序号：{missing}")
    return [by_index[index] for index in range(1, sentence_count + 1)]


def _spoken_token_positions(tokens: list[str]) -> list[int]:
    result: list[int] = []
    for index, token in enumerate(tokens):
        if token.isspace():
            continue
        if re.fullmatch(r"[^\w\u3400-\u9fff]+", token):
            continue
        result.append(index)
    return result


def highlighted_sentence_html(text: str, progress: float) -> str:
    """Yellow sentence highlight with a purple current spoken token."""
    tokens = _DISPLAY_TOKEN_RE.findall(text)
    spoken = _spoken_token_positions(tokens)
    current = -1
    if spoken:
        current = spoken[min(len(spoken) - 1, max(0, int(progress * len(spoken))))]
    parts: list[str] = []
    for index, token in enumerate(tokens):
        escaped = html.escape(token).replace("\n", "<br>")
        if index == current:
            parts.append(
                '<span style="background-color:#8b5cf6;color:#ffffff;'
                'font-weight:700;border-radius:3px;">' + escaped + "</span>"
            )
        else:
            parts.append(escaped)
    return (
        '<div style="background-color:#fff1a8;color:#202431;line-height:1.55;'
        'padding:4px 6px;border-radius:5px;">' + "".join(parts) + "</div>"
    )


class ClickableFrame(QFrame):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class SentenceCard(ClickableFrame):
    edited = Signal(int, str)
    retranslate_requested = Signal(int)

    def __init__(self, index: int, segment: BilingualSegment, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.segment = segment
        self._active = False
        self.setObjectName("longFormSentence")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        root = QHBoxLayout(self)
        root.setContentsMargins(13, 10, 13, 10)
        root.setSpacing(12)

        self.number = QLabel(f"{index + 1:02d}")
        self.number.setObjectName("sentenceNumber")
        self.number.setFixedWidth(28)
        self.number.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        root.addWidget(self.number)

        self.chinese = QLabel(segment.chinese)
        self.chinese.setObjectName("longFormChinese")
        self.chinese.setWordWrap(True)
        self.chinese.setTextFormat(Qt.RichText)
        self.chinese.setAttribute(Qt.WA_TransparentForMouseEvents)
        root.addWidget(self.chinese, 5)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setObjectName("sentenceDivider")
        root.addWidget(divider)

        self.english = QLabel(segment.english)
        self.english.setObjectName("longFormEnglish")
        self.english.setWordWrap(True)
        self.english.setTextFormat(Qt.RichText)
        self.english.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.english_editor = QPlainTextEdit(segment.english)
        self.english_editor.setObjectName("longFormEnglishEditor")
        self.english_editor.setMinimumHeight(72)
        editor_page = QWidget()
        editor_layout = QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(5)
        editor_layout.addWidget(self.english_editor)
        editor_actions = QHBoxLayout()
        editor_actions.addStretch()
        self.retranslate_button = QPushButton("↻ 结合全文重译本句")
        self.retranslate_button.clicked.connect(
            lambda checked=False: self.retranslate_requested.emit(self.index)
        )
        editor_actions.addWidget(self.retranslate_button)
        editor_layout.addLayout(editor_actions)
        self.english_stack = QStackedWidget()
        self.english_stack.addWidget(self.english)
        self.english_stack.addWidget(editor_page)
        root.addWidget(self.english_stack, 5)

        self.duration = QLabel(f"预计\n{segment.estimated_seconds:.0f}s")
        self.duration.setObjectName("sentenceDuration")
        self.duration.setFixedWidth(48)
        self.duration.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        root.addWidget(self.duration)

    def set_active(self, active: bool, progress: float = 0.0) -> None:
        property_changed = self._active != active
        self._active = active
        self.setProperty("active", active)
        if active:
            self.chinese.setText(highlighted_sentence_html(self.segment.chinese, progress))
            self.english.setText(highlighted_sentence_html(self.segment.english, progress))
        else:
            self.chinese.setText(html.escape(self.segment.chinese))
            self.english.setText(html.escape(self.segment.english))
        if property_changed:
            self.style().unpolish(self)
            self.style().polish(self)

    def set_actual_duration(self, seconds: float) -> None:
        self.segment.actual_seconds = seconds
        self.duration.setText(f"实际\n{seconds:.1f}s")

    def set_edit_mode(self, enabled: bool) -> None:
        if enabled:
            self.english_editor.setPlainText(self.segment.english)
        else:
            self.commit_edit()
        self.english_stack.setCurrentIndex(1 if enabled else 0)

    def commit_edit(self) -> bool:
        updated = self.english_editor.toPlainText().strip()
        if not updated or updated == self.segment.english:
            return False
        self.segment.english = updated
        self.segment.actual_seconds = None
        self.english.setText(html.escape(updated))
        self.duration.setText(f"预计\n{self.segment.estimated_seconds:.0f}s")
        self.edited.emit(self.index, updated)
        return True

    def set_translation(self, english: str, estimated_seconds: float) -> None:
        self.segment.english = english.strip()
        self.segment.estimated_seconds = estimated_seconds
        self.segment.actual_seconds = None
        self.english.setText(html.escape(self.segment.english))
        self.english_editor.setPlainText(self.segment.english)
        self.duration.setText(f"预计\n{estimated_seconds:.0f}s")


class LongFormReaderDialog(QDialog):
    sentence_changed = Signal(int, int)
    state_changed = Signal(str)
    translation_updated = Signal(int, str)
    pronunciations_changed = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        segments: list[BilingualSegment],
        *,
        client: Any,
        tts_client: Any | None = None,
        settings: dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.segments = segments
        self.client = client
        self.tts_client = tts_client or client
        self.settings = dict(settings)
        self.cards: list[SentenceCard] = []
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.control_lock = threading.Lock()
        self.requested_index: int | None = None
        self.active_cancel_event: threading.Event | None = None
        self.worker_thread: threading.Thread | None = None
        self.worker_stop_event = threading.Event()
        self.worker_generation = 0
        self.current_index = 0
        self.active_index = -1
        self.playing = False
        self.paused = False
        self.edit_mode = False
        self.rehearsal_only = False
        self.audio_cache: dict[str, bytes] = {}
        self.audio_errors: dict[str, Exception] = {}
        self.audio_inflight: dict[str, threading.Event] = {}
        self.cache_lock = threading.Lock()
        self.prefetch_cancel_event = threading.Event()
        self.estimated_total = sum(segment.estimated_seconds for segment in segments)
        self._build_ui()
        self.state_changed.connect(self.apply_state_update)
        self.translation_updated.connect(self._apply_retranslation)
        self.setWindowTitle("双语整段朗读台 · 全文语境模式")
        self.setMinimumSize(900, 620)
        self.resize(1120, 760)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 15, 18, 16)
        root.setSpacing(10)
        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("双语整段朗读台")
        title.setObjectName("readerTitle")
        raw_terms = str(self.settings.get("translation_terms", "")).strip()
        try:
            term_count = len(json.loads(raw_terms)) if raw_terms else 0
        except (TypeError, ValueError, json.JSONDecodeError):
            term_count = 0
        domain = str(self.settings.get("translation_domain", "")).strip()
        context_badges = ["全文语境翻译", f"已锁定 {term_count} 条专有术语"]
        if domain:
            context_badges.append(f"领域：{domain}")
        context_badges.append("点击任意句即可从该句继续")
        subtitle = QLabel(" · ".join(context_badges))
        subtitle.setObjectName("readerSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch()
        self.pronunciation_button = QPushButton("发音词典")
        self.pronunciation_button.setToolTip("只改变送给语音模型的读法，不改变屏幕译文")
        self.pronunciation_button.clicked.connect(self.edit_pronunciations)
        top.addWidget(self.pronunciation_button)
        self.rehearsal_checkbox = QCheckBox("仅本机试听")
        self.rehearsal_checkbox.setToolTip("校对时不发送到 Teams/VB-CABLE，只从本机监听设备播放")
        self.rehearsal_checkbox.toggled.connect(self.set_rehearsal_only)
        top.addWidget(self.rehearsal_checkbox)
        self.edit_button = QPushButton("✎ 校对译文")
        self.edit_button.setCheckable(True)
        self.edit_button.clicked.connect(self.toggle_edit_mode)
        top.addWidget(self.edit_button)
        self.state_label = QLabel("准备播放")
        self.state_label.setObjectName("readerState")
        top.addWidget(self.state_label)
        root.addLayout(top)

        headings = QHBoxLayout()
        headings.setContentsMargins(53, 0, 64, 0)
        zh = QLabel("中文原文")
        en = QLabel("English · 全文语境译文")
        for label in (zh, en):
            label.setObjectName("readerColumnTitle")
        headings.addWidget(zh, 5)
        headings.addSpacing(24)
        headings.addWidget(en, 5)
        root.addLayout(headings)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("readerScroll")
        content = QWidget()
        self.card_layout = QVBoxLayout(content)
        self.card_layout.setContentsMargins(5, 5, 5, 5)
        self.card_layout.setSpacing(7)
        for index, segment in enumerate(self.segments):
            card = SentenceCard(index, segment)
            card.clicked.connect(lambda checked=False, i=index: self._card_clicked(i))
            card.edited.connect(self._translation_edited)
            card.retranslate_requested.connect(self.retranslate_sentence)
            self.cards.append(card)
            self.card_layout.addWidget(card)
        self.card_layout.addStretch()
        self.scroll.setWidget(content)
        root.addWidget(self.scroll, 1)

        timeline = QFrame()
        timeline.setObjectName("readerTimeline")
        controls = QVBoxLayout(timeline)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(7)
        slider_row = QHBoxLayout()
        self.elapsed = QLabel("00:00")
        self.elapsed.setObjectName("timelineClock")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(1, int(self.estimated_total * 1000)))
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self._seek_from_slider)
        self.total = QLabel(format_clock(self.estimated_total))
        self.total.setObjectName("timelineClock")
        slider_row.addWidget(self.elapsed)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.total)
        controls.addLayout(slider_row)

        buttons = QHBoxLayout()
        self.position_label = QLabel(f"共 {len(self.segments)} 句 · 预计 {format_clock(self.estimated_total)}")
        self.position_label.setObjectName("readerPosition")
        buttons.addWidget(self.position_label)
        buttons.addStretch()
        self.restart_button = QPushButton("↺ 从头开始")
        self.restart_button.clicked.connect(lambda: self.play_from(0))
        self.export_button = QPushButton("⇩ 导出讲稿")
        self.export_button.clicked.connect(self.export_script)
        self.pause_button = QPushButton("⏸ 暂停")
        self.pause_button.setObjectName("readerPrimaryButton")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button = QPushButton("⏹ 停止")
        self.stop_button.clicked.connect(self.stop)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.restart_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        controls.addLayout(buttons)
        root.addWidget(timeline)

        self.setStyleSheet(READER_STYLE_SHEET)

    def start(self) -> None:
        self.play_from(0)

    def _card_clicked(self, index: int) -> None:
        if not self.edit_mode:
            self.play_from(index)

    def play_from(self, index: int) -> None:
        if not self.segments:
            return
        if self.edit_mode:
            return
        index = max(0, min(len(self.segments) - 1, int(index)))
        with self.control_lock:
            self.requested_index = index
            if self.active_cancel_event is not None:
                self.active_cancel_event.set()
        self.pause_event.set()
        self.paused = False
        self.pause_button.setText("⏸ 暂停")
        if (
            self.worker_thread is None
            or not self.worker_thread.is_alive()
            or self.worker_stop_event.is_set()
        ):
            old_stop = self.worker_stop_event
            old_stop.set()
            self.worker_stop_event = threading.Event()
            self.worker_generation += 1
            generation = self.worker_generation
            self.worker_thread = threading.Thread(
                target=self._playback_worker,
                args=(self.worker_stop_event, generation),
                name="long-form-reader",
                daemon=True,
            )
            self.worker_thread.start()

    def toggle_pause(self) -> None:
        if not self.playing:
            self.play_from(self.current_index)
            return
        self.paused = not self.paused
        if self.paused:
            self.pause_event.clear()
            self.pause_button.setText("▶ 继续")
            self.state_label.setText("已暂停")
            self.state_changed.emit("paused")
        else:
            self.pause_event.set()
            self.pause_button.setText("⏸ 暂停")
            self.state_label.setText("继续朗读…")
            self.state_changed.emit("playing")

    def stop(self) -> None:
        self.worker_stop_event.set()
        self.pause_event.set()
        with self.control_lock:
            self.requested_index = None
            if self.active_cancel_event is not None:
                self.active_cancel_event.set()
        self.playing = False
        self.paused = False
        self.pause_button.setText("▶ 继续")
        self.state_label.setText("已停止 · 点击任意句继续")
        self.state_changed.emit("stopped")

    def toggle_edit_mode(self, checked: bool) -> None:
        self.edit_mode = bool(checked)
        if self.edit_mode:
            self.stop()
            self.edit_button.setText("✓ 完成校对")
            self.state_label.setText("校对模式 · 可编辑英文或单句重译")
        else:
            for card in self.cards:
                card.commit_edit()
            self.edit_button.setText("✎ 校对译文")
            self.state_label.setText("校对已保存 · 点击任意句播放")
        for card in self.cards:
            card.set_edit_mode(self.edit_mode)

    def edit_pronunciations(self) -> None:
        current = str(self.settings.get("tts_pronunciations", ""))
        value, accepted = QInputDialog.getMultiLineText(
            self,
            "TTS 发音词典",
            'JSON 对象：屏幕文字 → 实际读法。例如 {"OpenAI":"Open A I"}',
            current,
        )
        if not accepted:
            return
        try:
            parse_pronunciation_dictionary(value)
        except ValueError as exc:
            QMessageBox.warning(self, "发音词典格式错误", str(exc))
            return
        self.settings["tts_pronunciations"] = value.strip()
        self.pronunciations_changed.emit(value.strip())
        self._invalidate_audio_cache()
        self.state_label.setText("发音词典已应用 · 本次朗读缓存已刷新")

    def set_rehearsal_only(self, checked: bool) -> None:
        if self.playing:
            self.stop()
        self.rehearsal_only = bool(checked)
        self.state_label.setText(
            "仅本机试听 · 点击任意句继续" if checked else "Teams 输出已启用 · 点击任意句继续"
        )

    @staticmethod
    def _srt_clock(seconds: float) -> str:
        millis = max(0, int(round(seconds * 1000)))
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def export_script(self) -> None:
        path, selected = QFileDialog.getSaveFileName(
            self,
            "导出双语讲稿",
            "bilingual-script.srt",
            "双语字幕 (*.srt);;双语文本 (*.txt);;讲稿工程数据 (*.json)",
        )
        if not path:
            return
        output = Path(path)
        if not output.suffix:
            output = output.with_suffix(".srt" if "字幕" in selected else ".txt")
        durations = [item.actual_seconds or item.estimated_seconds for item in self.segments]
        suffix = output.suffix.lower()
        if suffix == ".srt":
            lines: list[str] = []
            cursor = 0.0
            for index, (segment, duration) in enumerate(zip(self.segments, durations), start=1):
                lines.extend(
                    [
                        str(index),
                        f"{self._srt_clock(cursor)} --> {self._srt_clock(cursor + duration)}",
                        segment.chinese,
                        segment.english,
                        "",
                    ]
                )
                cursor += duration
            content = "\n".join(lines)
        elif suffix == ".json":
            project_settings = {
                key: self.settings.get(key)
                for key in (
                    "long_form_model",
                    "translation_domain",
                    "translation_style",
                    "translation_terms",
                    "translation_memories",
                    "tts_model",
                    "tts_provider",
                    "voice",
                    "voicestudio_model",
                    "voicestudio_voice",
                    "tts_rate",
                    "tts_pitch",
                    "tts_pronunciations",
                )
            }
            content = json.dumps(
                {
                    "format": "teams-voice-translator-long-form-v1",
                    "settings": project_settings,
                    "segments": [
                        {
                            "chinese": segment.chinese,
                            "english": segment.english,
                            "estimated_seconds": segment.estimated_seconds,
                            "actual_seconds": segment.actual_seconds,
                        }
                        for segment in self.segments
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        else:
            content = "\n\n".join(
                f"{index}. {segment.chinese}\n{segment.english}"
                for index, segment in enumerate(self.segments, start=1)
            )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.state_label.setText(f"已导出：{output.name}")

    def _translation_edited(self, index: int, english: str) -> None:
        segment = self.segments[index]
        segment.english = english
        segment.estimated_seconds = estimate_speech_seconds(
            english, float(self.settings.get("tts_rate", 1.0))
        )
        segment.actual_seconds = None
        self.cards[index].set_translation(english, segment.estimated_seconds)
        self._invalidate_audio_cache(index)
        self._refresh_total_duration()

    def retranslate_sentence(self, index: int) -> None:
        if not 0 <= index < len(self.segments):
            return
        self.cards[index].retranslate_button.setEnabled(False)
        self.state_label.setText(f"正在结合全文重译第 {index + 1} 句…")

        def worker() -> None:
            try:
                source = self.segments[index].chinese
                full_text = "\n".join(segment.chinese for segment in self.segments)
                english = self.client.translate_long_form(full_text, [source], self.settings)[0]
                self.translation_updated.emit(index, english)
            except Exception as exc:
                self.error.emit(str(exc))
                self.state_changed.emit(f"retranslate_failed:{index}")

        threading.Thread(target=worker, name=f"long-form-retranslate-{index}", daemon=True).start()

    def _apply_retranslation(self, index: int, english: str) -> None:
        if not 0 <= index < len(self.segments):
            return
        estimate = estimate_speech_seconds(
            english, float(self.settings.get("tts_rate", 1.0))
        )
        self.cards[index].set_translation(english, estimate)
        self.segments[index].estimated_seconds = estimate
        self._invalidate_audio_cache(index)
        self._refresh_total_duration()
        self.cards[index].retranslate_button.setEnabled(True)
        self.state_label.setText(f"第 {index + 1} 句已结合全文重译")

    def _take_requested_index(self) -> int | None:
        with self.control_lock:
            requested = self.requested_index
            self.requested_index = None
            return requested

    def _has_requested_index(self) -> bool:
        with self.control_lock:
            return self.requested_index is not None

    def _cache_key(self, index: int) -> str:
        spoken = apply_pronunciation_dictionary(
            self.segments[index].english,
            str(self.settings.get("tts_pronunciations", "")),
        )
        options = {
            key: self.settings.get(key)
            for key in (
                "tts_model",
                "tts_provider",
                "voice",
                "voicestudio_model",
                "voicestudio_voice",
                "tts_sample_rate",
                "tts_volume",
                "tts_rate",
                "tts_pitch",
                "tts_seed",
                "tts_instruction",
                "tts_emotion_tag",
            )
        }
        raw = json.dumps([spoken, options], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _invalidate_audio_cache(self, index: int | None = None) -> None:
        self.prefetch_cancel_event.set()
        self.prefetch_cancel_event = threading.Event()
        with self.cache_lock:
            if index is None:
                self.audio_cache.clear()
                self.audio_errors.clear()
            else:
                # Keys contain content/settings, so old entries are harmless but
                # removing all keeps memory bounded after repeated proofreading.
                self.audio_cache.clear()
                self.audio_errors.clear()

    def _get_or_synthesize_audio(
        self,
        index: int,
        cancel_event: threading.Event,
    ) -> bytes:
        key = self._cache_key(index)
        with self.cache_lock:
            cached = self.audio_cache.get(key)
            if cached is not None:
                return cached
            existing = self.audio_inflight.get(key)
            if existing is None:
                existing = threading.Event()
                self.audio_inflight[key] = existing
                producer = True
            else:
                producer = False
        if not producer:
            while not existing.wait(0.08):
                if cancel_event.is_set():
                    return b""
            with self.cache_lock:
                cached = self.audio_cache.get(key)
                error = self.audio_errors.get(key)
            if cached is not None:
                return cached
            if error is not None:
                # A speculative prefetch can fail because of a brief network
                # interruption. Foreground playback gets one clean retry.
                with self.cache_lock:
                    self.audio_errors.pop(key, None)
                return self._get_or_synthesize_audio(index, cancel_event)
            return b""

        try:
            parts: list[bytes] = []
            spoken = apply_pronunciation_dictionary(
                self.segments[index].english,
                str(self.settings.get("tts_pronunciations", "")),
            )
            count = self.tts_client.stream_tts(spoken, self.settings, parts.append, cancel_event)
            pcm = b"".join(parts)
            if cancel_event.is_set():
                return b""
            if count == 0 or not pcm:
                raise RuntimeError(f"第 {index + 1} 句没有返回音频数据")
            with self.cache_lock:
                self.audio_cache[key] = pcm
                self.audio_errors.pop(key, None)
            return pcm
        except Exception as exc:
            if cancel_event.is_set():
                return b""
            with self.cache_lock:
                self.audio_errors[key] = exc
            raise
        finally:
            with self.cache_lock:
                event = self.audio_inflight.pop(key, None)
            if event is not None:
                event.set()

    def _schedule_prefetch(self, index: int) -> None:
        if index >= len(self.segments) or self.prefetch_cancel_event.is_set():
            return
        key = self._cache_key(index)
        with self.cache_lock:
            if key in self.audio_cache or key in self.audio_inflight:
                return
        cancel = self.prefetch_cancel_event

        def worker() -> None:
            try:
                self._get_or_synthesize_audio(index, cancel)
                if not cancel.is_set():
                    self.state_changed.emit(f"prefetched:{index}")
            except Exception:
                # Foreground playback will surface a real error if this sentence
                # is reached; background prefetch must not interrupt the lesson.
                pass

        threading.Thread(target=worker, name=f"long-form-prefetch-{index}", daemon=True).start()

    def _playback_worker(self, worker_stop: threading.Event, generation: int) -> None:
        index = self._take_requested_index()
        if index is None:
            index = self.current_index
        if self.rehearsal_only:
            devices = [self.settings.get("monitor_output_device")]
        else:
            devices = [self.settings.get("teams_output_device")]
        if not self.rehearsal_only and self.settings.get("monitor_enabled"):
            devices.append(self.settings.get("monitor_output_device"))
        sample_rate = int(self.settings.get("tts_sample_rate", 24000))
        self.playing = True
        self.state_changed.emit("playing")
        try:
            with MultiOutputPlayer(devices, sample_rate) as player:
                while index < len(self.segments) and not worker_stop.is_set():
                    requested = self._take_requested_index()
                    if requested is not None:
                        index = requested
                    self.current_index = index
                    self.sentence_changed.emit(index + 1, len(self.segments))
                    self.state_changed.emit(f"synthesizing:{index}")
                    cancel = threading.Event()
                    with self.control_lock:
                        self.active_cancel_event = cancel
                    pcm = self._get_or_synthesize_audio(index, cancel)
                    with self.control_lock:
                        if self.active_cancel_event is cancel:
                            self.active_cancel_event = None
                    if worker_stop.is_set():
                        break
                    if self._has_requested_index():
                        continue
                    if not pcm:
                        continue
                    duration = len(pcm) / float(sample_rate * 2)
                    self.state_changed.emit(f"duration:{index}:{duration}")
                    self._schedule_prefetch(index + 1)
                    frame_bytes = max(2, int(sample_rate * 2 * 0.04))
                    offset = 0
                    while offset < len(pcm) and not worker_stop.is_set():
                        if self._has_requested_index():
                            break
                        while not self.pause_event.wait(0.08):
                            if worker_stop.is_set() or self._has_requested_index():
                                break
                        if worker_stop.is_set() or self._has_requested_index():
                            break
                        chunk = pcm[offset : offset + frame_bytes]
                        player.write(chunk)
                        offset += len(chunk)
                        fraction = min(1.0, offset / len(pcm))
                        self.state_changed.emit(f"progress:{index}:{fraction}")
                    if self._has_requested_index():
                        continue
                    index += 1
                if index >= len(self.segments) and not worker_stop.is_set():
                    self.state_changed.emit("finished")
        except Exception as exc:
            if not worker_stop.is_set():
                self.error.emit(str(exc))
                self.state_changed.emit("error")
        finally:
            with self.control_lock:
                if generation == self.worker_generation:
                    self.active_cancel_event = None
                    self.playing = False

    def _seek_from_slider(self) -> None:
        target = self.slider.value() / 1000.0
        cumulative = 0.0
        for index, segment in enumerate(self.segments):
            duration = segment.actual_seconds or segment.estimated_seconds
            if target < cumulative + duration:
                self.play_from(index)
                return
            cumulative += duration
        self.play_from(len(self.segments) - 1)

    def apply_state_update(self, state: str) -> None:
        if state.startswith("synthesizing:"):
            index = int(state.split(":", 1)[1])
            self._activate_card(index, 0.0)
            self.state_label.setText(f"正在合成第 {index + 1}/{len(self.segments)} 句…")
        elif state.startswith("duration:"):
            _, raw_index, raw_duration = state.split(":", 2)
            self.cards[int(raw_index)].set_actual_duration(float(raw_duration))
        elif state.startswith("progress:"):
            _, raw_index, raw_fraction = state.split(":", 2)
            index, fraction = int(raw_index), float(raw_fraction)
            self._activate_card(index, fraction)
            self.state_label.setText(f"正在朗读第 {index + 1}/{len(self.segments)} 句")
            self._update_timeline(index, fraction)
        elif state == "finished":
            self.state_label.setText("朗读完成")
            self.pause_button.setText("▶ 再次播放")
            self.current_index = 0
            durations = [item.actual_seconds or item.estimated_seconds for item in self.segments]
            actual_total = sum(durations)
            self.slider.setMaximum(max(1, int(actual_total * 1000)))
            self.slider.setValue(self.slider.maximum())
            self.elapsed.setText(format_clock(actual_total))
            self.total.setText(format_clock(actual_total))
            self.position_label.setText(
                f"共 {len(self.segments)} 句 · 实际/估算 {format_clock(actual_total)}"
            )
        elif state == "error":
            self.state_label.setText("朗读发生错误")
        elif state.startswith("retranslate_failed:"):
            index = int(state.split(":", 1)[1])
            if 0 <= index < len(self.cards):
                self.cards[index].retranslate_button.setEnabled(True)
            self.state_label.setText(f"第 {index + 1} 句重译失败")

    def _activate_card(self, index: int, fraction: float) -> None:
        if self.active_index != index:
            if 0 <= self.active_index < len(self.cards):
                self.cards[self.active_index].set_active(False)
            self.active_index = index
        self.cards[index].set_active(True, fraction)
        self.scroll.ensureWidgetVisible(self.cards[index], 20, 80)

    def _update_timeline(self, index: int, fraction: float) -> None:
        durations = [item.actual_seconds or item.estimated_seconds for item in self.segments]
        total = sum(durations)
        elapsed = sum(durations[:index]) + durations[index] * fraction
        self.slider.setMaximum(max(1, int(total * 1000)))
        self.slider.blockSignals(True)
        self.slider.setValue(min(self.slider.maximum(), int(elapsed * 1000)))
        self.slider.blockSignals(False)
        self.elapsed.setText(format_clock(elapsed))
        self.total.setText(format_clock(total))
        self.position_label.setText(
            f"第 {index + 1}/{len(self.segments)} 句 · "
            f"本句 {durations[index]:.1f}s"
        )

    def _refresh_total_duration(self) -> None:
        durations = [item.actual_seconds or item.estimated_seconds for item in self.segments]
        self.estimated_total = sum(durations)
        self.slider.setMaximum(max(1, int(self.estimated_total * 1000)))
        self.total.setText(format_clock(self.estimated_total))
        self.position_label.setText(
            f"共 {len(self.segments)} 句 · 预计 {format_clock(self.estimated_total)}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.prefetch_cancel_event.set()
        self.stop()
        super().closeEvent(event)


READER_STYLE_SHEET = """
QDialog { background: #f5f7fb; color: #202431; font-family: 'Microsoft YaHei UI'; }
QLabel#readerTitle { color: #12284a; font-size: 25px; font-weight: 800; }
QLabel#readerSubtitle { color: #68758c; font-size: 13px; }
QLabel#readerState { background: #e8f2ff; color: #1766bd; border: 1px solid #b8d8ff;
    border-radius: 15px; padding: 8px 14px; font-weight: 700; }
QLabel#readerColumnTitle { color: #34445f; font-size: 14px; font-weight: 800; padding: 3px; }
QScrollArea#readerScroll { border: 1px solid #d8dfeb; border-radius: 12px; background: #edf1f7; }
QScrollArea#readerScroll > QWidget > QWidget { background: #edf1f7; }
QFrame#longFormSentence { background: white; border: 1px solid #d8dfeb; border-radius: 10px; }
QFrame#longFormSentence:hover { border: 1px solid #7eaff0; background: #fbfdff; }
QFrame#longFormSentence[active="true"] { border: 2px solid #f1b928; background: #fffdf4; }
QLabel#sentenceNumber { color: #1766bd; font-size: 13px; font-weight: 800; }
QLabel#longFormChinese { color: #202431; font-size: 15px; }
QLabel#longFormEnglish { color: #233553; font-family: 'Segoe UI'; font-size: 15px; }
QPlainTextEdit#longFormEnglishEditor { background: #fbfdff; color: #233553;
    border: 1px solid #a9c6eb; border-radius: 7px; padding: 7px;
    font-family: 'Segoe UI'; font-size: 14px; selection-background-color: #8b5cf6; }
QLabel#sentenceDuration { color: #748096; font-size: 11px; }
QFrame#sentenceDivider { color: #e1e6ef; }
QFrame#readerTimeline { background: white; border: 1px solid #d8dfeb; border-radius: 12px; }
QLabel#timelineClock { color: #34445f; font-family: 'Consolas'; font-weight: 700; }
QLabel#readerPosition { color: #56647a; font-weight: 600; }
QPushButton { min-height: 32px; padding: 4px 14px; border: 1px solid #c8d3e3;
    border-radius: 7px; background: #f4f7fb; color: #263a58; font-weight: 600; }
QPushButton:hover { background: #e8f2ff; border-color: #7eaff0; }
QPushButton#readerPrimaryButton { background: #176dcc; color: white; border-color: #176dcc; }
QSlider::groove:horizontal { height: 7px; background: #dce4f0; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #6f55d9; border-radius: 3px; }
QSlider::handle:horizontal { width: 17px; margin: -5px 0; background: #ffffff;
    border: 2px solid #6f55d9; border-radius: 8px; }
"""
