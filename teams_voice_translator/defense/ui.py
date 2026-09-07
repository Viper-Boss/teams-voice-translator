from __future__ import annotations

import base64
import html
import json
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot, QEvent
from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QScrollArea,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..aliyun import ApiError, BailianClient, create_realtime_asr
from ..audio import (list_audio_devices, list_loopback_devices,
                     choose_committee_loopback, is_program_output_loopback)
from ..hotkeys import KEY_MAP
from .hotkeys_poll import PollingHotkeys
from ..longform import split_source_sentences
from ..oss_upload import OssTemporaryUploader, OssUploadError
from ..voice_sample import SUPPORTED_AUDIO_SUFFIXES, VoiceSampleError, normalized_voice_sample, inspect_voice_sample, voice_sample_duration
from ..api_payloads import VOICE_PREPROCESS_MODELS
from .overlay import SubtitleOverlay, token_chunks, current_token_index, highlight_html
from .pipeline import DefenseEngine, EngineCallbacks
from .ppt_context import import_presentation
from .qa import QaAdvisor
from .settings import DefenseSettings
from .translator import ContextTranslator
from .tts_session import create_tts_session
from .speech_policy import (speech_settings, NATURAL_INSTRUCTION, INSTRUCTION_MODELS,
                            INSTRUCTION_PRESETS, SPEECH_SAMPLE_RATE, pause_comparison_text)
from .ambience import AMBIENCE_PRESETS


CUSTOM_INSTRUCTION = "__custom__"


def _safe_list_audio_devices() -> tuple[list, list]:
    try:
        return list_audio_devices()
    except Exception as exc:
        print(f"音频设备枚举失败：{exc}")
        return [], []



def _safe_list_loopback_devices() -> list:
    try:
        return list_loopback_devices()
    except Exception as exc:
        print(f"回环设备枚举失败：{exc}")
        return []


def app_icon() -> QIcon:
    """Window/exe icon; works both from source tree and PyInstaller bundle."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    path = root / "assets" / "defense_mode.ico"
    if path.exists():
        return QIcon(str(path))
    fallback = Path(__file__).resolve().parents[2] / "assets" / "defense_mode.ico"
    return QIcon(str(fallback)) if fallback.exists() else QIcon()


DEFAULT_READING_PASSAGE = (
    "各位老师好，我是本次答辩的学生，今天向各位老师汇报我的研究工作。"
    "我将从研究背景、研究方法和实验结果三个方面进行说明，恳请各位老师批评指正。"
)
MAX_RECORD_SECONDS = 30

def clone_channel_for_model(model: str) -> str:
    """复刻通道由模型家族决定：qwen3-tts-vc 走 Base64 直传，其余走 OSS 中转。"""
    return "qwen3" if str(model).startswith("qwen3-tts-vc") else "legacy"


CHANNEL_LABELS = {
    "qwen3": "Qwen3 直传（qwen3-tts-vc-* · Base64 上传，无需 OSS，秒级完成）",
    "legacy": "旧版 OSS 中转（cosyvoice-* / qwen-audio-tts-* · 上传 OSS + 等待部署 1~3 分钟）",
}


def voice_probe(
    settings: DefenseSettings,
    workspace: str,
    api_key: str,
    voice: str,
    tts_model: str,
    *,
    text: str = "Hello, this is a quick voice check.",
    timeout: float = 45.0,
    overrides: dict | None = None,
    cancel_event: threading.Event | None = None,
    on_progress=None,
) -> bytes:
    """Synthesize one short sentence on the monitor device only.

    Raises on any failure so callers can report it; success means the voice,
    the TTS model and the credentials all work.  Never plays to the cable, so
    it is safe to run outside a meeting.
    """
    from ..audio import MultiOutputPlayer
    from .rehearsal import play_pcm

    client = BailianClient(api_key, workspace, proxy=settings.ws_proxy, proxy_mode=settings.proxy_mode)
    monitor = settings.get("monitor_output_device")
    payload = speech_settings(settings, model=tts_model, voice=voice)
    if overrides:
        payload.update(overrides)
    errors = []
    audio_seen = threading.Event()
    captured = bytearray()
    player = None
    session = None
    def play(chunk):
        if cancel_event is not None and cancel_event.is_set():
            raise ApiError("试听已停止")
        captured.extend(chunk)
        audio_seen.set()
    try:
        session = create_tts_session(
            client, payload, on_audio=play, on_error=errors.append,
            connect_timeout=15, utterance_timeout=timeout, max_attempts=1,
        )
        session.start()
        if not session.speak(text):
            raise ApiError("试听任务未能提交")
        deadline = time.monotonic() + timeout
        while not session.wait_until_idle(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("试听已停止")
            if time.monotonic() >= deadline:
                raise ApiError("试听等待超时")
        if errors:
            raise ApiError(errors[-1])
        if not audio_seen.is_set():
            raise ApiError("试听未收到有效音频")
        if cancel_event is not None and cancel_event.is_set():
            raise ApiError("试听已停止")
        stop = cancel_event if cancel_event is not None else threading.Event()
        player = MultiOutputPlayer([monitor], SPEECH_SAMPLE_RATE)
        player.__enter__()
        if on_progress:
            on_progress(0.0)
        if not play_pcm(captured, player, stop, threading.Event(), on_progress or (lambda _p: None)):
            raise ApiError("试听已停止")
        return bytes(captured)
    finally:
        if session is not None:
            session.close()
        if player is not None:
            player.close()

ACCENT = "#60a5fa"
OK_GREEN = "#3ecf8e"
WARN_ORANGE = "#f5a623"
ERR_RED = "#ff5d5d"
PANEL_BG = "#0b1220"
PANEL_BG_2 = "#131f32"
TEXT_MAIN = "#e8eaf0"
TEXT_DIM = "#9aaec9"

STATE_TEXT = {
    "idle": ("未开始", "#5a6272"),
    "starting": ("正在打开通道…", WARN_ORANGE),
    "standby": ("待命", ACCENT),
    "listening": ("F9 翻译中", ERR_RED),
    "direct": ("F5 原声直通", OK_GREEN),
}


class NoWheelComboBox(QComboBox):
    """悬停时滚轮不改变选中值，防止滚动设置页时误改关键配置；点开列表后仍可正常滚动。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumContentsLength(12)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class TalkButton(QPushButton):
    """Release even if the mouse leaves the button before being let go."""
    hold_released = Signal()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.hold_released.emit()

    def hideEvent(self, event):
        self.hold_released.emit()
        super().hideEvent(event)


class NoWheelSpinBox(QSpinBox):
    """同上：数字框忽略滚轮，避免误改数值。"""

    def wheelEvent(self, event) -> None:
        event.ignore()



class TranslatorSignals(QObject):
    """Marshals engine callbacks from worker threads into the Qt loop."""

    state_changed = Signal(str)
    status_message = Signal(str)
    error_message = Signal(str)
    mic_preview = Signal(str)
    committee_preview = Signal(str)
    my_segment = Signal(int, str)
    my_delta = Signal(int, str)
    my_done = Signal(int, str, int)
    committee_done = Signal(int, str, str)
    tts_sentence_started = Signal(str, str)
    tts_sentence_duration = Signal(str, float)
    input_pressed = Signal(str, str)
    input_released = Signal(str, str)
    cancel_requested = Signal()
    committee_device = Signal(str)
    latency_ready = Signal(int, int)


def _format_clock(value: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(value))


class DefenseWindow(QMainWindow):
    def __init__(self, settings: DefenseSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or DefenseSettings()
        self.signals = TranslatorSignals()
        self.engine = DefenseEngine(
            self.settings,
            EngineCallbacks(
                on_state=self.signals.state_changed.emit,
                on_status=self.signals.status_message.emit,
                on_error=self.signals.error_message.emit,
                on_mic_preview=lambda text: self.signals.mic_preview.emit(text),
                on_committee_preview=lambda text: self.signals.committee_preview.emit(text),
                on_my_segment=lambda seg_id, text: self.signals.my_segment.emit(seg_id, text),
                on_my_delta=lambda seg_id, text: self.signals.my_delta.emit(seg_id, text),
                on_my_done=lambda seg_id, text, latency: self.signals.my_done.emit(seg_id, text, latency),
                on_committee_done=lambda seg_id, en, zh: self.signals.committee_done.emit(seg_id, en, zh),
                on_tts_sentence_started=lambda text, zh: self.signals.tts_sentence_started.emit(text, zh),
                on_tts_sentence_duration=lambda text, seconds: self.signals.tts_sentence_duration.emit(text, seconds),
                on_committee_device=lambda name: self.signals.committee_device.emit(name),
                on_latency=self.signals.latency_ready.emit,
            ),
        )
        self.hotkeys: PollingHotkeys | None = None
        self._timeline_rows: dict[int, int] = {}
        self._recent_spoken: list[str] = []
        self.recent_buttons: list[QPushButton] = []
        self._pending_audio_prompt = False
        self._input_mode = None
        self._held_inputs = {}
        self.overlay: SubtitleOverlay | None = None
        self._build_ui()
        self._connect_signals()
        self.engine.start()

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        from .. import __version__

        self.setWindowTitle(f"答辩模式 · 实时中英互译 v{__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1380, 860)
        self.setMinimumSize(1060, 680)
        self._apply_dark_theme()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_top_bar())

        captions = QSplitter(Qt.Horizontal)
        captions.addWidget(self._build_my_panel())
        captions.addWidget(self._build_committee_panel())
        captions.addWidget(self._build_qa_panel())
        captions.setSizes([520, 470, 400])

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(captions)
        self.timeline_panel = self._build_timeline()
        main_split.addWidget(self.timeline_panel)
        main_split.setSizes([380, 240])
        layout.addWidget(main_split, 1)

        layout.addWidget(self._build_recent_bar())
        layout.addWidget(self._build_input_bar())

        self.overlay = SubtitleOverlay(self.settings)  # 第一句语音开始时自动出现

    ACTIVE_BUTTON_STYLE = (
        "QPushButton:checked { background: #2fae74; border-color: #2fae74; color: white; }"
    )

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        stack = QVBoxLayout(bar)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(10)
        stack.addLayout(row)

        title = QLabel("答辩模式  /  LIVE")
        title.setStyleSheet(f"font-size: 23px; font-weight: 700; color: {TEXT_MAIN};")
        row.addWidget(title)

        self.state_pill = QLabel("未开始")
        self.state_pill.setAlignment(Qt.AlignCenter)
        self.state_pill.setFixedWidth(130)
        self._set_pill("idle")
        row.addWidget(self.state_pill)

        self.latency_label = QLabel("")
        self.latency_label.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(self.latency_label)

        row.addStretch(1)

        self.focus_check = QCheckBox("专注模式")
        self.focus_check.setToolTip("隐藏助答和时间轴，放大双方字幕")
        self.focus_check.toggled.connect(lambda enabled: (
            self.qa_panel.setVisible(not enabled), self.timeline_panel.setVisible(not enabled)))
        row.addWidget(self.focus_check)
        row = QHBoxLayout()
        row.setSpacing(8)
        stack.addLayout(row)
        self.continuous_check = QCheckBox("持续监听")
        self.continuous_check.setChecked(bool(self.settings.get("continuous_enabled", False)))
        self.continuous_check.setToolTip("仅改变按钮操作方式，不自动开启原声或翻译。关闭：按住生效，松开停止。开启：单击开、再单击关；切换模式不会自动跳回。")
        self.continuous_check.toggled.connect(self._on_continuous_toggled)
        row.addWidget(self.continuous_check)

        self.overlay_check = QCheckBox("字幕悬浮窗")
        self.overlay_check.setChecked(bool(self.settings.get("overlay_enabled", True)))
        self.overlay_check.toggled.connect(self._on_overlay_toggled)
        row.addWidget(self.overlay_check)

        self.direct_button = TalkButton(f"原声 ({self._hotkey_label('direct_hotkey', 'F5')})")
        self.direct_button.setCheckable(True)
        self.direct_button.setToolTip("按住将原声送入会议，松开停止；开启持续监听后改为单击开关。")
        self.direct_button.setStyleSheet(self.ACTIVE_BUTTON_STYLE)
        self.direct_button.pressed.connect(lambda: self._input_press("direct", "mouse"))
        self.direct_button.released.connect(lambda: self._input_release("direct", "mouse"))
        self.direct_button.hold_released.connect(lambda: self._input_release("direct", "mouse"))
        self.direct_button.clicked.connect(self._refresh_input_buttons)
        row.addWidget(self.direct_button)

        self.translate_button = TalkButton(f"翻译 ({self._hotkey_label('translate_hotkey', 'F9')})")
        self.translate_button.setCheckable(True)
        self.translate_button.setToolTip("按住识别中文，松开停止收音并处理已说内容；开启持续监听后改为单击开关。")
        self.translate_button.setStyleSheet(self.ACTIVE_BUTTON_STYLE)
        self.translate_button.pressed.connect(lambda: self._input_press("translate", "mouse"))
        self.translate_button.released.connect(lambda: self._input_release("translate", "mouse"))
        self.translate_button.hold_released.connect(lambda: self._input_release("translate", "mouse"))
        self.translate_button.clicked.connect(self._refresh_input_buttons)
        row.addWidget(self.translate_button)

        self.start_button = QPushButton("开始答辩")
        self.start_button.setCheckable(True)
        self.start_button.setMinimumHeight(36)
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._on_toggle_defense)
        row.addWidget(self.start_button)

        row.addStretch(1)
        self.check_button = QPushButton("自检")
        self.check_button.clicked.connect(self._open_self_check)
        row.addWidget(self.check_button)

        self.ppt_button = QPushButton("导入 PPT")
        self.ppt_button.clicked.connect(self._on_import_ppt)
        row.addWidget(self.ppt_button)

        self.rehearsal_button = QPushButton("讲稿预习")
        self.rehearsal_button.clicked.connect(self._open_rehearsal)
        row.addWidget(self.rehearsal_button)

        self.voice_compare_button = QPushButton("声音对比")
        self.voice_compare_button.clicked.connect(lambda: VoiceCompareDialog(self, self.settings).exec())
        row.addWidget(self.voice_compare_button)
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self._open_settings)
        row.addWidget(self.settings_button)
        return bar

    def _hotkey_label(self, key: str, fallback: str) -> str:
        value = str(self.settings.get(key, fallback) or fallback)
        return value.upper()

    def _build_my_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(self._panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QLabel("01  我的发言 · 中文 → 英文")
        header.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        layout.addWidget(header)

        self.mic_preview = QLabel("准备好后开始答辩，按 F9 或开启持续监听。")
        self.mic_preview.setWordWrap(True)
        self.mic_preview.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")
        layout.addWidget(self.mic_preview)

        live_caption = QLabel("英文译文")
        live_caption.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(live_caption)

        self.live_english = QLabel("Your words, in your voice.")
        self.live_english.setWordWrap(True)
        self.live_english.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 20px; font-weight: 500;")
        layout.addWidget(self.live_english)

        self.my_history = QTextBrowser()
        self.my_history.setOpenExternalLinks(False)
        self.my_history.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        layout.addWidget(self.my_history, 1)
        return panel

    def _build_committee_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(self._panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header = QLabel("02  评委提问 · 英文 → 中文")
        header.setStyleSheet(f"color: {OK_GREEN}; font-weight: 600;")
        header_row.addWidget(header)
        header_row.addStretch(1)
        self.committee_device_label = QLabel("")
        self.committee_device_label.setStyleSheet("color: #ffd400;")
        header_row.addWidget(self.committee_device_label)
        layout.addLayout(header_row)

        self.committee_preview = QLabel("（开始答辩后自动监听：微信/Teams 里对方说话会转成中文字幕）")
        self.committee_preview.setWordWrap(True)
        self.committee_preview.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")
        layout.addWidget(self.committee_preview)

        self.live_committee_zh = QLabel("")
        self.live_committee_zh.setWordWrap(True)
        self.live_committee_zh.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 20px; font-weight: 500;")
        layout.addWidget(self.live_committee_zh)

        self.committee_history = QTextBrowser()
        self.committee_history.setOpenExternalLinks(False)
        self.committee_history.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        layout.addWidget(self.committee_history, 1)
        return panel

    def _build_qa_panel(self) -> QWidget:
        self.qa_panel = QaPanel(self.settings, self.engine)
        return self.qa_panel

    def _build_timeline(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header_row = QHBoxLayout()
        caption = QLabel("对话记录  /  上下文连续保留")
        caption.setStyleSheet(f"color: {TEXT_DIM};")
        header_row.addWidget(caption)
        header_row.addStretch(1)
        self.memory_label = QLabel("")
        self.memory_label.setStyleSheet(f"color: {TEXT_DIM};")
        header_row.addWidget(self.memory_label)
        self.auto_record_check = QCheckBox("自动录音")
        self.auto_record_check.setChecked(bool(self.settings.get("auto_record", True)))
        self.auto_record_check.setToolTip("勾选后点「开始答辩」自动录音，字幕与音频按次存入会议文件夹")
        self.auto_record_check.toggled.connect(
            lambda checked: self.settings.set("auto_record", bool(checked))
        )
        header_row.addWidget(self.auto_record_check)
        save_archive_button = QPushButton("保存本场记录")
        save_archive_button.clicked.connect(self._on_save_archive)
        header_row.addWidget(save_archive_button)
        save_audio_button = QPushButton("保存英文语音")
        save_audio_button.setToolTip("把本场所有合成的英文语音存为一个 WAV（内存临时缓存）")
        save_audio_button.clicked.connect(self.engine.export_meeting_audio)
        header_row.addWidget(save_audio_button)
        open_folder_button = QPushButton("打开记录文件夹")
        open_folder_button.clicked.connect(self._on_open_archive_folder)
        header_row.addWidget(open_folder_button)
        clear_button = QPushButton("清空记忆")
        clear_button.clicked.connect(self._on_clear_memory)
        header_row.addWidget(clear_button)
        layout.addLayout(header_row)

        self.timeline = QTableWidget(0, 6)
        self.timeline.setHorizontalHeaderLabels(["时间", "谁", "原文", "译文", "延迟", "重播"])
        self.timeline.setShowGrid(False)
        self.timeline.setAlternatingRowColors(True)
        self.timeline.verticalHeader().setDefaultSectionSize(48)
        self.timeline.setColumnWidth(5, 58)
        self.timeline.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.timeline.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.timeline.verticalHeader().setVisible(False)
        self.timeline.setEditTriggers(QTableWidget.NoEditTriggers)
        self.timeline.setSelectionBehavior(QTableWidget.SelectRows)
        self.timeline.setStyleSheet(
            "QTableWidget { background: " + PANEL_BG_2 + "; color: " + TEXT_MAIN + "; gridline-color: #2a3040; }"
            "QHeaderView::section { background: #1d2230; color: " + TEXT_DIM + "; border: none; padding: 4px; }"
        )
        layout.addWidget(self.timeline)
        return container

    def _build_recent_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        caption = QLabel("最近 5 句：")
        caption.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(caption)
        for index in range(5):
            button = QPushButton("")
            button.setCheckable(False)
            button.setStyleSheet(
                "QPushButton { background: #2b2410; border: 1px solid #f5a623;"
                " color: #ffd97a; text-align: left; padding: 4px 10px; }"
                "QPushButton:hover { background: #3d3317; }"
            )
            button.setVisible(False)
            button.clicked.connect(
                lambda _checked=False, pos=index: self._on_replay_recent(pos)
            )
            self.recent_buttons.append(button)
            row.addWidget(button, 1)
        row.addStretch(0)
        return bar

    def _on_tts_sentence_started(self, english: str, chinese: str) -> None:
        # 只驱动悬浮字幕；最近5句由“我”的每句完成（_on_my_done）驱动，
        # 重复播放/缓存重播不会改变顺序。
        if self.overlay is not None and self.overlay_check.isChecked():
            self.overlay.show_sentence(chinese, english)

    def _push_recent(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        if sentence in self._recent_spoken:
            self._recent_spoken.remove(sentence)
        self._recent_spoken.insert(0, sentence)
        del self._recent_spoken[5:]
        self._rebuild_recent_buttons()

    def _rebuild_recent_buttons(self) -> None:
        for index, button in enumerate(self.recent_buttons):
            if index < len(self._recent_spoken):
                sentence = self._recent_spoken[index]
                shown = sentence if len(sentence) <= 30 else sentence[:29] + "…"
                button.setText(f"▶ {shown}")
                button.setToolTip(sentence)
                button.setVisible(True)
            else:
                button.setVisible(False)

    def _on_replay_recent(self, position: int) -> None:
        if position >= len(self._recent_spoken):
            return
        sentence = self._recent_spoken[position]
        self.engine.replay_english(sentence)
        self.statusBar().showMessage(f"重播：{sentence[:40]}")

    def _build_input_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("应急输入：手动输入中文（回车翻译并用你的音色说出），勾选后直接朗读英文原文")
        self.text_input.returnPressed.connect(self._on_send_text)
        row.addWidget(self.text_input, 1)

        self.direct_speak_checkbox = QCheckBox("直接朗读英文")
        self.direct_speak_checkbox.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(self.direct_speak_checkbox)

        send_button = QPushButton("发送")
        send_button.clicked.connect(self._on_send_text)
        row.addWidget(send_button)

        stop_button = QPushButton("停止语音 (Esc)")
        stop_button.clicked.connect(self.signals.cancel_requested.emit)
        row.addWidget(stop_button)
        return bar

    # -------------------------------------------------------------- theme
    def _apply_dark_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setFont(QFont("Microsoft YaHei UI", 10))
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(PANEL_BG))
        palette.setColor(QPalette.WindowText, QColor(TEXT_MAIN))
        palette.setColor(QPalette.Base, QColor(PANEL_BG_2))
        palette.setColor(QPalette.Text, QColor(TEXT_MAIN))
        palette.setColor(QPalette.Button, QColor("#232936"))
        palette.setColor(QPalette.ButtonText, QColor(TEXT_MAIN))
        palette.setColor(QPalette.Highlight, QColor(ACCENT))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
        app.setPalette(palette)
        app.setStyleSheet(
            f"""
            QMainWindow, QDialog {{ background: {PANEL_BG}; }}
            QPushButton {{
                background: #232936; color: {TEXT_MAIN};
                border: 1px solid #2e3648; border-radius: 6px; padding: 6px 14px;
            }}
            QPushButton:hover {{ background: #2b3345; }}
            QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {PANEL_BG_2}; color: {TEXT_MAIN};
                border: 1px solid #2e3648; border-radius: 6px; padding: 5px 8px;
            }}
            QTableWidget {{ alternate-background-color: #171b25; }}
            QLabel {{ color: {TEXT_MAIN}; }}
            QPushButton#primaryButton {{ background: #2563eb; border-color: #3b82f6; color: white; font-weight: 600; }}
            QPushButton#primaryButton:hover {{ background: #3b82f6; }}
            QPushButton:disabled {{ color: #677a94; background: #172236; border-color: #24324a; }}
            QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: #60a5fa; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ width: 9px; background: transparent; margin: 0; }}
            QScrollBar::handle:vertical {{ background: #36465f; border-radius: 4px; min-height: 28px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QSplitter::handle {{ background: #24324a; }}
            QStatusBar {{ color: #9aaec9; border-top: 1px solid #24324a; padding: 4px; }}
            QToolTip {{ background: #25344e; color: #eef4ff; border: 1px solid #496184; padding: 6px; }}
            """
        )

    def _panel_style(self) -> str:
        return (
            f"QFrame {{ background: {PANEL_BG_2}; border: 1px solid #232a3a; border-radius: 10px; }}"
            "QLabel { border: none; }"
        )

    def _set_pill(self, state: str) -> None:
        text, color = STATE_TEXT.get(state, ("未知", TEXT_DIM))
        self.state_pill.setText(text)
        self.state_pill.setStyleSheet(
            f"background: {color}; color: white; border-radius: 12px; font-weight: 600; padding: 3px 0;"
        )

    # ----------------------------------------------------------- signals io
    def _connect_signals(self) -> None:
        self.signals.input_pressed.connect(self._input_press, Qt.QueuedConnection)
        self.signals.input_released.connect(self._input_release, Qt.QueuedConnection)
        self.signals.state_changed.connect(self._on_state_changed)
        self.signals.status_message.connect(self.statusBar().showMessage)
        self.signals.error_message.connect(self._show_error)
        self.signals.mic_preview.connect(self.mic_preview.setText)
        self.signals.committee_preview.connect(self.committee_preview.setText)
        self.signals.my_segment.connect(self._on_my_segment)
        self.signals.my_delta.connect(self._on_my_delta)
        self.signals.my_done.connect(self._on_my_done)
        self.signals.committee_done.connect(self._on_committee_done)
        self.signals.tts_sentence_started.connect(self._on_tts_sentence_started)
        self.signals.tts_sentence_duration.connect(self._on_tts_sentence_duration)
        self.signals.cancel_requested.connect(self._on_cancel)
        self.signals.committee_device.connect(self._on_committee_device)
        self.signals.latency_ready.connect(self._on_latency)

    def _on_latency(self, seg_id, milliseconds):
        self.latency_label.setText(f"识别结束 → 发声  {milliseconds / 1000:.1f} s")
        self.latency_label.setStyleSheet(f"color: {TEXT_DIM};")
        row = self._timeline_rows.get(seg_id)
        if row is not None:
            self.timeline.setItem(row, 4, QTableWidgetItem(f"{milliseconds / 1000:.1f}s"))

    def _on_tts_sentence_duration(self, english: str, seconds: float) -> None:
        if self.overlay is not None:
            self.overlay.set_duration(english, seconds)

    def _on_committee_device(self, name: str) -> None:
        self._active_loopback_name = name
        self.committee_device_label.setText(f"监听中：{name}")

    def _show_selected_committee_device(self) -> None:
        name = str(self.settings.shared.get("loopback_device_name", "") or "").strip()
        self._active_loopback_name = ""
        self.committee_device_label.setText(
            f"已选择：{name}（开始答辩后监听）" if name else ""
        )

    # -------------------------------------------------------------- hotkeys
    def start_hotkeys(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.hotkeys = PollingHotkeys(
            direct_key=str(self.settings.get("direct_hotkey", "f5")),
            translate_key=str(self.settings.get("translate_hotkey", "f9")),
            cancel_key=str(self.settings.get("cancel_hotkey", "esc")),
            direct_press=lambda: self.signals.input_pressed.emit("direct", "keyboard"),
            direct_release=lambda: self.signals.input_released.emit("direct", "keyboard"),
            translate_press=lambda: self.signals.input_pressed.emit("translate", "keyboard"),
            translate_release=lambda: self.signals.input_released.emit("translate", "keyboard"),
            cancel=self.signals.cancel_requested.emit,
        )
        self.hotkeys.start()
        self._refresh_hotkey_labels()

    def _refresh_hotkey_labels(self) -> None:
        self.direct_button.setText(f"原声 ({self._hotkey_label('direct_hotkey', 'F5')})")
        self.translate_button.setText(f"翻译 ({self._hotkey_label('translate_hotkey', 'F9')})")

    def _on_cancel(self) -> None:
        if self.overlay is not None:
            self.overlay.stop()
        self.engine.interrupt_speech()

    def _on_continuous_toggled(self, checked: bool) -> None:
        self.settings.set("continuous_enabled", bool(checked))
        self._held_inputs.clear()
        if not checked:
            self._set_input_mode(None)

    @Slot(str, str)
    def _input_press(self, mode, source):
        if self.continuous_check.isChecked():
            self._set_input_mode(None if self._input_mode == mode else mode)
            return
        self._held_inputs[(source, mode)] = True
        self._set_input_mode(mode)

    @Slot(str, str)
    def _input_release(self, mode, source):
        self._held_inputs.pop((source, mode), None)
        if not self.continuous_check.isChecked() and self._input_mode == mode:
            if not any(key[1] == mode for key in self._held_inputs):
                self._set_input_mode(None)

    def _set_input_mode(self, mode):
        if mode == self._input_mode:
            return
        self._input_mode = mode
        if mode == "direct":
            self.engine.set_listening(False)
            self.engine.start_direct()
        elif mode == "translate":
            self.engine.stop_direct()
            self.engine.set_listening(True)
        else:
            self.engine.set_listening(False)
            self.engine.stop_direct()
        self._refresh_input_buttons()

    def _refresh_input_buttons(self, *_args):
        self.direct_button.setChecked(self._input_mode == "direct")
        self.translate_button.setChecked(self._input_mode == "translate")

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            if hasattr(self, "_held_inputs") and hasattr(self, "continuous_check"):
                if not self.continuous_check.isChecked():
                    for source, mode in list(self._held_inputs):
                        if source == "mouse":
                            self._input_release(mode, source)
        super().changeEvent(event)

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.settings.set("overlay_enabled", bool(checked))
        if self.overlay is not None:
            if checked:
                self.overlay.place_at_bottom()
                self.overlay.show()
            else:
                self.overlay.stop()

    # ------------------------------------------------------------ callbacks
    def _on_state_changed(self, state: str) -> None:
        if state == "idle":
            self._input_mode = None
            self._held_inputs.clear()
            self._show_selected_committee_device()
        self._set_pill(state)
        self.start_button.setChecked(state != "idle")
        self.start_button.setText("结束答辩" if state != "idle" else "开始答辩")
        direct = state == "direct"
        for widget, checked in (
            (self.translate_button, state == "listening"),
            (self.direct_button, direct),
        ):
            widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(False)
        # 持续监听开关只反映用户偏好（settings），不再被播放状态反向改写。
        desired = bool(self.settings.get("continuous_enabled", False))
        self.continuous_check.blockSignals(True)
        self.continuous_check.setChecked(desired)
        self.continuous_check.blockSignals(False)
        if state == "standby":
            self.mic_preview.setText("（按住「翻译」或 F9 说话，松开停止；勾选「持续监听」后改为单击开关）")
        if state == "idle" and self._pending_audio_prompt:
            self._pending_audio_prompt = False
            self._prompt_meeting_audio()

    def _prompt_meeting_audio(self) -> None:
        seconds = self.engine.meeting_audio_seconds()
        if seconds <= 0:
            return
        minutes = seconds / 60
        box = QMessageBox(self)
        box.setWindowTitle("本场缓存语音")
        box.setText(
            f"本场共合成 {minutes:.1f} 分钟英文语音（临时缓存在内存）。" + "\n" + "要怎么处理？"
        )
        save_button = box.addButton("保存到会议文件夹", QMessageBox.AcceptRole)
        keep_button = box.addButton("保留在内存（下一场开始前有效）", QMessageBox.ActionRole)
        discard_button = box.addButton("丢弃", QMessageBox.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_button:
            self.engine.export_meeting_audio()
            folder = self.engine.archive_folder_path()
            if folder:
                self.memory_label.setText(f"记录目录：{folder}")
        elif clicked is keep_button:
            self.statusBar().showMessage("缓存语音保留在内存，可用「保存英文语音」按钮随时导出")
        elif clicked is discard_button:
            self.engine.discard_meeting_audio()

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        self.latency_label.setText(message[:60])
        self.latency_label.setStyleSheet(f"color: {ERR_RED};")

    def _on_my_segment(self, seg_id: int, text: str) -> None:
        self.mic_preview.setText(text)
        self.live_english.setText("…")

    def _on_my_delta(self, seg_id: int, so_far: str) -> None:
        self.live_english.setText(so_far)

    def _on_my_done(self, seg_id: int, english: str, latency_ms: int) -> None:
        self.live_english.setText(english)
        self.mic_preview.setText("（已翻译；等英文播完后，按住「翻译」继续说话）")
        seconds = latency_ms / 1000 if latency_ms else 0
        color = OK_GREEN if seconds and seconds <= 2.5 else (WARN_ORANGE if seconds <= 4 else ERR_RED)
        if seconds:
            self.latency_label.setText(f"上一句端到端延迟：{seconds:.1f} 秒")
            self.latency_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.my_history.append(
            f"<p style='margin:2px 0'><span style='color:{TEXT_DIM}'>#{seg_id} · {seconds:.1f}s</span><br>"
            f"<span style='font-size:14px'>{html.escape(english)}</span></p>"
        )
        self._ensure_timeline_row(seg_id, "我", english, latency_ms, translation=english)
        self._push_recent(english)

    def _on_committee_done(self, seg_id: int, english: str, chinese: str) -> None:
        self.live_committee_zh.setText(chinese)
        self.committee_history.append(
            f"<p style='margin:2px 0'><span style='color:{TEXT_DIM};font-size:12px'>{html.escape(english)}</span><br>"
            f"<span style='color:{OK_GREEN};font-size:14px'>{html.escape(chinese)}</span></p>"
        )
        self._ensure_timeline_row(seg_id, "评委", english, 0, translation=chinese)
        if hasattr(self, "qa_panel"):
            self.qa_panel.on_committee_done(english, chinese)

    def _ensure_timeline_row(
        self,
        seg_id: int,
        who: str,
        text: str,
        latency_ms: int,
        *,
        translation: str | None = None,
    ) -> None:
        record = next((item for item in self.engine.segments() if item.seg_id == seg_id), None)
        zh_source = record.source if record is not None else ""
        shown_source = zh_source or ("（手动输入）" if who == "我" else text)
        row = self._timeline_rows.get(seg_id)
        if row is None:
            # 最新在最上面：已有行整体下移
            for key in self._timeline_rows:
                self._timeline_rows[key] += 1
            row = 0
            self._timeline_rows[seg_id] = row
            self.timeline.insertRow(row)
            items = [
                QTableWidgetItem(_format_clock(time.time())),
                QTableWidgetItem(who),
                QTableWidgetItem(shown_source),
                QTableWidgetItem(translation or ""),
                QTableWidgetItem(f"{latency_ms / 1000:.1f}s" if latency_ms else ""),
            ]
            for column, item in enumerate(items):
                item.setToolTip(item.text())
                self.timeline.setItem(row, column, item)
            if who == "我" and (translation or "").strip():
                english_text = translation
                play_button = QPushButton("▶")
                play_button.setFixedWidth(44)
                play_button.setToolTip("从本地缓存重播这句英文（不请求服务器）")
                play_button.clicked.connect(lambda _checked=False, t=english_text: self.engine.replay_english(t))
                self.timeline.setCellWidget(row, 5, play_button)
            self.timeline.scrollToTop()
        else:
            if translation is not None:
                self.timeline.item(row, 3).setText(translation)
            if latency_ms:
                self.timeline.item(row, 4).setText(f"{latency_ms / 1000:.1f}s")
        turns = len(self.engine.memory.snapshot_turns())
        self.memory_label.setText(f"上下文记忆：{turns} 轮对话")

    # ------------------------------------------------------------ handlers
    def _on_toggle_defense(self) -> None:
        if self.engine.state == "idle":
            self.engine.set_listening(False)
            self.engine.start_defense()
        else:
            self._held_inputs.clear()
            self._set_input_mode(None)
            self._pending_audio_prompt = True
            self.engine.stop_defense()

    def _on_save_archive(self) -> None:
        self.engine.save_archive_now()
        folder = self.engine.archive_folder_path()
        if folder:
            self.memory_label.setText(f"记录目录：{folder}")

    def _on_open_archive_folder(self) -> None:
        folder = self.engine.archive_folder_path()
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(self, "记录文件夹", "还没有会议记录文件夹（点「开始答辩」后自动创建）")
            return
        try:
            os.startfile(folder)  # type: ignore[attr-defined]  # Windows-only
        except Exception as exc:
            QMessageBox.information(self, "记录文件夹", f"打开失败：{exc}" + "\n" + folder)

    def _on_clear_memory(self) -> None:
        if QMessageBox.question(self, "清空记忆", "确定清空整场对话记忆和时间轴？翻译上下文将从零开始。") == QMessageBox.Yes:
            self.engine.clear_memory()
            self.timeline.setRowCount(0)
            self._timeline_rows.clear()
            self.my_history.clear()
            self.committee_history.clear()
            self.memory_label.setText("")

    def _on_send_text(self) -> None:
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self.engine.send_text(text, translate=not self.direct_speak_checkbox.isChecked())

    # ------------------------------------------------------------- dialogs
    def _open_settings(self) -> None:
        try:
            dialog = SettingsDialog(
                self, self.settings, active_loopback_name=self.engine.loopback_device_name()
            )
        except Exception as exc:
            QMessageBox.critical(self, "设置", f"设置页打开失败：{exc}")
            return
        if dialog.exec() == QDialog.Accepted:
            selected = str(self.settings.shared.get("loopback_device_name", "") or "").strip()
            self.committee_device_label.setText(
                f"正在切换：{selected}" if selected and self.engine.state != "idle"
                else (f"已选择：{selected}（开始答辩后监听）" if selected else "")
            )
            try:
                self.engine.apply_settings()
            except Exception as exc:
                self.signals.error_message.emit(f"重新加载术语/摘要失败：{exc}")
            self.memory_label.setText("")
            try:
                self.start_hotkeys()  # 快捷键可能改了：重新挂载并刷新按钮文字
            except Exception as exc:
                self.signals.error_message.emit(f"快捷键重挂失败：{exc}")

    def _open_self_check(self) -> None:
        SelfCheckDialog(self, self.settings).exec()

    def _open_rehearsal(self) -> None:
        RehearsalDialog(self, self.settings).exec()

    def _on_import_ppt(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "选择答辩 PPT", "", "演示文稿 (*.pptx);;文本 (*.txt *.md)"
        )
        if not path:
            return
        try:
            client = BailianClient(
                self.settings.get_api_key(),
                self.settings.workspace_id,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=int(self.settings.shared.get("request_timeout", 45)),
            )
        except ApiError as exc:
            QMessageBox.warning(self, "导入 PPT", str(exc))
            return
        self.ppt_button.setEnabled(False)
        self.statusBar().showMessage("正在读取 PPT 并提炼术语，通常需要 10~30 秒…")

        def worker() -> None:
            try:
                context = import_presentation(client, Path(path), model=str(self.settings.get("llm_model")))
            except Exception as exc:
                self.signals.error_message.emit(f"PPT 导入失败：{exc}")
                return
            self.settings.set("defense_brief", context.brief)
            terms = self.settings.glossary_terms()
            merged = {item["source"]: item["target"] for item in terms}
            for item in context.terms:
                merged.setdefault(item["source"], item["target"])
            self.settings.set_glossary([{"source": k, "target": v} for k, v in merged.items()])
            self.settings.set("context_source_path", context.source_path)
            self.engine.reload_context()
            self.signals.status_message.emit(
                f"PPT 导入完成：{context.slide_count} 页，术语共 {len(self.settings.glossary_terms())} 条，"
                "可在设置中查看论文摘要与术语表"
            )

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------------- close
    def closeEvent(self, event) -> None:
        if hasattr(self, "qa_panel"):
            self.qa_panel.shutdown()
        if self.overlay is not None:
            self.overlay.stop()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.engine.shutdown()
        super().closeEvent(event)


QA_ACCENT = "#f5a623"

# 阿里云百炼常用模型清单：(模型 ID, 一句话简介)。下拉显示 "ID · 简介"，存储/传参用 ID。
CHAT_MODELS: list[tuple[str, str]] = [
    ("qwen-plus", "均衡稳妥，翻译/问答首选"),
    ("qwen3.7-plus", "新一代均衡（3.7 代）"),
    ("qwen3.8-flash", "新一代高速（3.8 代），延迟优先"),
    ("qwen3.8-max", "最新最强，推理最深，稍慢"),
    ("qwen-flash", "快且便宜"),
    ("qwen-turbo", "最快最省"),
    ("qwen3-plus", "上一代均衡"),
    ("qwen3-max", "上一代最强"),
    ("qwen-max", "旧旗舰"),
]
ASR_MODELS: list[tuple[str, str]] = [
    ("qwen3-asr-flash-realtime", "综合最准，推荐"),
    ("qwen3-asr-flash-realtime-2026-02-10", "同上，锁定新快照"),
    ("qwen3-asr-flash-realtime-2025-10-27", "同上，锁定旧快照"),
    ("qwen-audio-3.0-asr-flash-streaming", "支持术语热词加权，专有名词多选它"),
    ("fun-asr-realtime", "轻量快，方言口音友好"),
    ("fun-asr-realtime-2025-11-07", "FunASR 锁定快照"),
    ("fun-asr-realtime-2026-02-28", "FunASR 新快照（中英日）"),
]
TTS_MODELS: list[tuple[str, str]] = [
    ("qwen3-tts-vc-realtime-2026-01-15", "低延迟；不支持语气指令"),
    ("qwen3-tts-vc-realtime-2025-11-27", "实时旧快照，备用"),
    ("qwen3-tts-vc-2026-01-22", "完整合成；可对照跨语种自然度"),
    ("cosyvoice-v3.5-plus", "支持克隆与语气指令；需对应音色"),
    ("cosyvoice-v3.5-flash", "v3.5 提速便宜版"),
    ("cosyvoice-v3-plus", "上一代克隆"),
    ("cosyvoice-v3-flash", "上一代快版"),
    ("cosyvoice-v2", "旧版"),
    ("cosyvoice-v1", "最旧版，仅存量"),
]


def fill_model_combo(combo: QComboBox, options: list[tuple[str, str]], current: str) -> None:
    """Fill a non-editable model dropdown so a click opens the list directly."""
    combo.setEditable(False)
    combo.clear()
    for model, blurb in options:
        combo.addItem(f"{model} · {blurb}", model)
    if current and combo.findData(current) < 0:
        combo.addItem(f"{current} · 自定义", current)
    combo.setCurrentIndex(max(combo.findData(current), 0))


class QaPanel(QFrame):
    """AI 助答：监听评委提问 → 大模型生成第一人称口头回答 → 中英字幕 → 一键播放。"""

    qa_status = Signal(str)
    qa_answer = Signal(str, str)

    def __init__(self, settings: DefenseSettings, engine: DefenseEngine) -> None:
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.advisor_factory = QaAdvisor  # 测试可替换
        self._job_cancel = threading.Event()
        self._answer_en = ""
        self._answer_zh = ""
        self._question = ""
        self._build_ui()
        self.qa_status.connect(self._on_qa_status)
        self.qa_answer.connect(self._on_qa_answer)

    def _build_ui(self) -> None:
        self.setStyleSheet(self._panel_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("AI 助答")
        title.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        header.addWidget(title)
        self.auto_check = QCheckBox("自动生成")
        self.auto_check.setChecked(bool(self.settings.get("qa_auto", True)))
        self.auto_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.auto_check.toggled.connect(
            lambda checked: self.settings.set("qa_auto", bool(checked))
        )
        header.addWidget(self.auto_check)
        header.addStretch(1)
        self.model_combo = NoWheelComboBox()
        fill_model_combo(self.model_combo, CHAT_MODELS, str(self.settings.get("qa_model", "qwen-plus")))
        self.model_combo.setMinimumWidth(120)
        self.model_combo.currentIndexChanged.connect(
            lambda _index: self.settings.set("qa_model", self.model_combo.currentData())
        )
        header.addWidget(self.model_combo)
        layout.addLayout(header)

        layout.addWidget(QLabel("评委提问"))
        self.question_label = QLabel("（监听评委发言中，检测到提问会自动生成回答）")
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.question_label)

        layout.addWidget(QLabel("中文回答"))
        self.answer_zh = QLabel("")
        self.answer_zh.setWordWrap(True)
        self.answer_zh.setStyleSheet(f"color: {OK_GREEN}; font-size: 14px; font-weight: 600;")
        layout.addWidget(self.answer_zh)

        layout.addWidget(QLabel("English（播放的就是这段）"))
        self.answer_en = QLabel("")
        self.answer_en.setWordWrap(True)
        self.answer_en.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 14px;")
        layout.addWidget(self.answer_en)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("▶ 一键回答")
        self.play_button.setMinimumHeight(40)
        self.play_button.setStyleSheet(
            f"QPushButton {{ background: {OK_GREEN}; border: none; color: #08120c;"
            f" font-weight: 700; font-size: 15px; }}"
            f"QPushButton:hover {{ background: #4fe0a4; }}"
        )
        self.play_button.clicked.connect(self._play_answer)
        buttons.addWidget(self.play_button, 1)
        self.regen_button = QPushButton("↻ 重新生成")
        self.regen_button.clicked.connect(self._regenerate)
        buttons.addWidget(self.regen_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- triggers
    def on_committee_done(self, english: str, chinese: str) -> None:
        self._question = english or chinese
        self.question_label.setText(self._question)
        if self.auto_check.isChecked() and self._question.strip():
            self.generate(self._question)

    def _regenerate(self) -> None:
        if not self._question.strip():
            self.qa_status.emit("还没有评委提问，无法生成")
            return
        self.generate(self._question)

    def generate(self, question: str) -> None:
        try:
            advisor = self.advisor_factory(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.model_combo.currentData()),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
            )
        except ApiError as exc:
            self.qa_status.emit(str(exc))
            return
        self._job_cancel.set()
        cancel = threading.Event()
        self._job_cancel = cancel
        self._answer_en = ""
        self._answer_zh = ""
        self.answer_en.setText("")
        self.answer_zh.setText("")
        self.qa_status.emit(f"正在用 {advisor.model} 生成回答…")
        memory = self.engine.memory

        def worker() -> None:
            try:
                english, chinese = advisor.generate(memory, question, cancel_event=cancel)
            except Exception as exc:
                if "已取消" not in str(exc):
                    self.qa_status.emit(f"生成失败：{exc}")
                return
            if not english and not chinese:
                self.qa_status.emit("评委不是在提问，未生成回答")
                return
            self.qa_answer.emit(english, chinese)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- display
    def _on_qa_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")

    def _on_qa_answer(self, english: str, chinese: str) -> None:
        self._answer_en = english
        self._answer_zh = chinese
        self.answer_en.setText(english)
        self.answer_zh.setText(chinese or "（模型未提供中文对照）")
        self.qa_status.emit("回答已生成，点击下方按钮播放（Esc 可随时打断）")

    def _play_answer(self) -> None:
        if not self._answer_en.strip():
            self.qa_status.emit("还没有可播放的回答")
            return
        self.engine.speak_as_me(self._answer_en, label="AI 助答")
        self.qa_status.emit("正在用你的音色播放回答…（Esc 停止）")

    def shutdown(self) -> None:
        self._job_cancel.set()

    def _panel_style(self) -> str:
        return (
            f"QFrame {{ background: {PANEL_BG_2}; border: 1px solid #232a3a; border-radius: 10px; }}"
            "QLabel { border: none; }"
        )


class VoiceCompareDialog(QDialog):
    """Same English text, selectable bound voices, monitor-only listening."""
    finished_probe = Signal(str)
    playback_progress = Signal(object, str, float)

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self._cancel = threading.Event()
        self._closed = False
        self._is_busy = False
        self.subtitle_overlay = SubtitleOverlay(settings)
        self.subtitle_overlay.setParent(self, self.subtitle_overlay.windowFlags())
        self.playback_progress.connect(self._show_progress, Qt.QueuedConnection)
        self.setWindowTitle("声音对比 · 同一句英文，听音色与语气")
        self._cached_audio = {}
        self.resize(820, 560)
        layout = QVBoxLayout(self)
        note = QLabel("A 使用原始语气，B 可选择语气或停顿。每次合成最多请求一次；合成成功后可免费重听，仅在本机试听设备播放。更改文本或选项会清除试听缓存。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.text_edit = QPlainTextEdit("Thank you for the question. The key idea is to combine physical constraints with the information in the measured signals. Let me explain why this matters.")
        self.text_edit.setMaximumHeight(140)
        layout.addWidget(self.text_edit)
        self.voices = []
        entries = settings.get("voice_library", [])
        entries = [e for e in entries if isinstance(e, dict) and e.get("voice_id") and e.get("target_model")]
        current = str(settings.get("tts_voice_id", ""))
        if current and not any(e["voice_id"] == current for e in entries):
            entries.insert(0, {"voice_id": current, "target_model": settings.get("tts_model"), "name": "当前音色"})
        for caption in ("A · 原始语气", "B · 对比音色"):
            row = QHBoxLayout()
            row.addWidget(QLabel(caption))
            combo = NoWheelComboBox()
            combo.setMinimumContentsLength(24)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            for entry in entries:
                combo.addItem(f"{entry.get('name', '音色')} · {entry['target_model']}", entry)
            selected = next((i for i, e in enumerate(entries) if e["voice_id"] == current), 0)
            combo.setCurrentIndex(selected)
            self.voices.append(combo)
            row.addWidget(combo, 1)
            layout.addLayout(row)
        self.result = QLabel("建议先选同一音色，保持语速、音调 1.0；再比较不同模型。Qwen VC 的 B 不添加语气指令。")
        self.result.setWordWrap(True)
        layout.addWidget(self.result)
        self.b_style = NoWheelComboBox()
        for label, value in (("B：自然答辩语气", "instruction"),
                             ("B：仅在换行处停顿 250 ms", "pause"),
                             ("B：原始语气（比较不同模型）", "original")):
            self.b_style.addItem(label, value)
        layout.addWidget(self.b_style)
        buttons = QHBoxLayout()
        self.a_button = QPushButton("试听 A")
        self.b_button = QPushButton("试听 B")
        self.b_button.setObjectName("primaryButton")
        self.a_button.clicked.connect(lambda: self._probe(False))
        self.b_button.clicked.connect(lambda: self._probe(True))
        stop = self.stop_button = QPushButton("停止试听")
        stop.clicked.connect(self._stop_probe)
        for button in (self.a_button, self.b_button, stop):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.finished_probe.connect(self._finished, Qt.QueuedConnection)
        replay_row = QHBoxLayout()
        self.replay_buttons = []
        for index, name in enumerate(("重听 A · 免费", "重听 B · 免费")):
            button = QPushButton(name)
            button.setEnabled(False)
            button.clicked.connect(lambda checked=False, i=index: self._replay(i))
            self.replay_buttons.append(button)
            replay_row.addWidget(button)
        layout.addLayout(replay_row)
        self.text_edit.textChanged.connect(self._invalidate_cache)
        self.b_style.currentIndexChanged.connect(self._invalidate_cache)
        for combo in self.voices:
            combo.currentIndexChanged.connect(self._invalidate_cache)

    def _invalidate_cache(self, *_args):
        self._cached_audio.clear()
        for button in self.replay_buttons:
            button.setEnabled(False)

    def _busy(self, busy):
        self._is_busy = busy
        for control in (self.a_button, self.b_button, self.text_edit, self.b_style, *self.voices):
            control.setEnabled(not busy)
        for index, button in enumerate(self.replay_buttons):
            button.setEnabled(not busy and index in self._cached_audio)

    def _replay(self, index):
        if self._is_busy:
            return
        from .rehearsal import play_pcm
        pcm = self._cached_audio.get(index)
        if not pcm:
            return
        self._cancel = threading.Event()
        token = self._cancel
        text = self.text_edit.toPlainText().strip()
        monitor = self.settings.get("monitor_output_device")
        self._busy(True)
        self.result.setText("正在重听缓存音频，不调用云端…")
        def worker():
            from ..audio import MultiOutputPlayer
            try:
                with MultiOutputPlayer([monitor], SPEECH_SAMPLE_RATE) as player:
                    self.playback_progress.emit(token, text, 0.0)
                    if not play_pcm(pcm, player, token, threading.Event(),
                                    lambda progress: self.playback_progress.emit(token, text, progress)):
                        raise ApiError("试听已停止")
                message = "重听完成，未产生合成请求。"
            except Exception as exc:
                message = f"重听停止或失败：{exc}"
            try:
                self.finished_probe.emit(message)
            except RuntimeError:
                pass
        self._worker = threading.Thread(target=worker, name="defense-voice-replay", daemon=True)
        self._worker.start()

    def _probe(self, natural):
        if self._is_busy:
            return
        entry = self.voices[int(natural)].currentData()
        text = self.text_edit.toPlainText().strip()
        if not entry or not text:
            self.result.setText("请先在设置里创建或添加音色，并输入英文试听文本。")
            return
        if len(text) > 450:
            self.result.setText("对比试听请控制在 450 字符以内，保证两次内容一致。")
            return
        model = entry["target_model"]
        subtitle_text = text
        style = self.b_style.currentData() if natural else "original"
        if style == "pause":
            try:
                text = pause_comparison_text(text, model)
            except ValueError as exc:
                self.result.setText(str(exc))
                return
        self._cancel = threading.Event()
        token = self._cancel
        self._cached_audio.pop(int(natural), None)
        self._busy(True)
        self.result.setText("正在合成，请稍候…")
        model = entry["target_model"]
        voice = entry["voice_id"]
        values = {
            "speech_mode": "natural", "tts_rate": 1.0, "tts_pitch": 1.0,
            "tts_instruction": NATURAL_INSTRUCTION if style == "instruction" and model in INSTRUCTION_MODELS else "",
            "tts_enable_ssml": style == "pause",
        }
        workspace = self.settings.workspace_id
        api_key = self.settings.get_api_key()
        def worker():
            try:
                pcm = voice_probe(self.settings, workspace, api_key, voice, model,
                            text=text, timeout=90, overrides=values, cancel_event=token,
                            on_progress=lambda progress: self.playback_progress.emit(token, subtitle_text, progress))
                if token.is_set():
                    raise ApiError("试听已停止")
                self._cached_audio[int(natural)] = pcm
                message = "试听已完成。比较音色相似度、停顿、句尾变化和术语发音；此处不会自动切换正式答辩音色。"
            except Exception as exc:
                message = "试听已停止" if token.is_set() else f"试听失败：{exc}"
            try:
                self.finished_probe.emit(message)
            except RuntimeError:
                pass
        self._worker = threading.Thread(target=worker, name="defense-voice-comparison", daemon=True)
        self._worker.start()

    @Slot(object, str, float)
    def _show_progress(self, token, text, progress):
        if not self._closed and token is self._cancel and not token.is_set():
            self.subtitle_overlay.show_progress("声音对比 · 英文试听", text, progress)

    def _stop_probe(self):
        # Resolve the current token here; it is replaced for every probe/replay.
        self._cancel.set()
        self.subtitle_overlay.stop()
        self.result.setText("已停止试听" if not self._is_busy else "正在停止试听…")

    @Slot(str)
    def _finished(self, message):
        if self._closed:
            return
        self._busy(False)
        self.result.setText(message)

    def done(self, result):
        self._stop_probe()
        self._closed = True
        self.subtitle_overlay.close()
        super().done(result)


class _SampleRecorder:
    """Collects microphone audio as 48 kHz mono PCM16 for voice cloning."""

    def __init__(self, device: int | None) -> None:
        self.device = device
        self.frames = bytearray()
        self._stream = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        import sounddevice as sd

        self.frames = bytearray()
        self._stream = sd.RawInputStream(
            device=self.device,
            samplerate=48000,
            channels=1,
            dtype="int16",
            blocksize=4800,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, _frames, _time, _status) -> None:
        self.frames.extend(bytes(indata))

    def stop(self) -> float:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        return len(self.frames) / 2 / 48000


class CloneVoiceDialog(QDialog):
    """录音或选择样音 → 一键创建克隆音色 → 自动填入 voice_id。

    复刻通道由 TTS 模型家族决定：qwen3-tts-vc-* 走 Qwen3 Base64 直传；
    cosyvoice-* / qwen-audio-tts-* 走旧版 voice-enrollment（OSS 中转 + 部署轮询）。
    """

    clone_finished = Signal(str, str, bool)
    clone_progress = Signal(str)
    oss_probe_done = Signal(bool, str)
    sample_checked = Signal(int, str, float, str)

    def __init__(self, parent: QWidget, settings: DefenseSettings, tts_model: str) -> None:
        super().__init__(parent)
        self.settings = settings
        self.tts_model = (tts_model or str(settings.get("tts_model"))).strip()
        self.channel = clone_channel_for_model(self.tts_model)
        self.created_voice_id = ""
        self._recorder: _SampleRecorder | None = None
        self._sample_path: Path | None = None
        self._sample_seconds = 0.0
        self._sample_generation = 0
        self._record_started_at = 0.0
        self._cloning = False
        self._build_ui()
        self.clone_finished.connect(self._on_clone_finished)
        self.clone_progress.connect(self._on_clone_progress)
        self.oss_probe_done.connect(self._on_oss_probe_done)
        self.sample_checked.connect(self._on_sample_checked, Qt.QueuedConnection)
        self._update_channel_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    def _build_ui(self) -> None:
        from .. import __version__

        self.setWindowTitle(f"一键克隆音色 v{__version__}")
        self.resize(720, 820)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)

        hint = QLabel(
            "在安静房间用平时说话的音量自然朗读下面的文字（约 15 秒）。\n"
            "要求：单人、无背景音乐，保留自然停顿和语气；按平常答辩方式说，不要刻意播音。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        library_hint = QLabel("已克隆音色库（双击或选中后点按钮即可切换；从列表删除不会注销阿里云上的音色）")
        library_hint.setWordWrap(True)
        layout.addWidget(library_hint)
        self.library_table = QTableWidget(0, 4)
        self.library_table.setHorizontalHeaderLabels(["名称", "voice_id", "绑定模型", "创建时间"])
        self.library_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.library_table.verticalHeader().setVisible(False)
        self.library_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.library_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.library_table.setMaximumHeight(150)
        self.library_table.doubleClicked.connect(self._use_selected_voice)
        layout.addWidget(self.library_table)
        library_buttons = QHBoxLayout()
        use_button = QPushButton("使用选中音色")
        use_button.clicked.connect(self._use_selected_voice)
        library_buttons.addWidget(use_button)
        save_current_button = QPushButton("把当前 voice_id 存入列表")
        save_current_button.clicked.connect(self._save_current_voice)
        library_buttons.addWidget(save_current_button)
        delete_button = QPushButton("删除选中记录")
        delete_button.clicked.connect(self._delete_selected_voice)
        library_buttons.addWidget(delete_button)
        library_buttons.addStretch(1)
        layout.addLayout(library_buttons)
        self._ensure_current_voice_in_library()
        self._refresh_library()

        layout.addWidget(QLabel("朗读文字（可替换成你的答辩开场白，音色更贴合现场状态）"))
        self.passage_edit = QPlainTextEdit()
        self.passage_edit.setPlainText(DEFAULT_READING_PASSAGE)
        layout.addWidget(self.passage_edit, 1)

        self.use_transcript = QCheckBox("我按上面的文字朗读（提供原文可提高相似度）")
        self.use_transcript.setChecked(False)
        self.use_transcript.setToolTip("仅当文字与样音逐字一致时勾选。上传其他录音时不要使用默认朗读稿，否则可能降级复刻。")
        layout.addWidget(self.use_transcript)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("复刻通道（由模型自动决定，无需手选）"))
        self.channel_combo = NoWheelComboBox()
        channel_row.addWidget(self.channel_combo, 1)
        layout.addLayout(channel_row)
        self.channel_note = QLabel("")
        self.channel_note.setWordWrap(True)
        self.channel_note.setStyleSheet(f"color: {WARN_ORANGE};")
        layout.addWidget(self.channel_note)

        self.oss_group = QFrame(self)
        oss_form = QFormLayout(self.oss_group)
        oss_form.setContentsMargins(0, 0, 0, 0)
        self.oss_region_edit = QLineEdit(str(self.settings.shared.get("oss_region", "cn-beijing")))
        oss_form.addRow("OSS 地域", self.oss_region_edit)
        self.oss_bucket_edit = QLineEdit(str(self.settings.shared.get("oss_bucket", "")))
        self.oss_bucket_edit.setPlaceholderText("私有 Bucket 名称（样音用完即删）")
        oss_form.addRow("OSS Bucket", self.oss_bucket_edit)
        has_ak = bool(self.settings.shared.get_oss_access_key_id())
        self.oss_ak_edit = QLineEdit()
        self.oss_ak_edit.setEchoMode(QLineEdit.Password)
        self.oss_ak_edit.setPlaceholderText("已保存在凭据管理器，留空表示不修改" if has_ak else "OSS AccessKey ID（RAM 最小权限即可）")
        oss_form.addRow("AccessKey ID", self.oss_ak_edit)
        self.oss_sk_edit = QLineEdit()
        self.oss_sk_edit.setEchoMode(QLineEdit.Password)
        self.oss_sk_edit.setPlaceholderText("AccessKey Secret（同上，留空不修改）")
        oss_form.addRow("AccessKey Secret", self.oss_sk_edit)
        probe_row = QHBoxLayout()
        self.oss_probe_button = QPushButton("检测 OSS 并记住")
        self.oss_probe_button.setToolTip("真实上传/删除一个临时小文件验证连通性；通过后配置自动保存，以后不用再填")
        self.oss_probe_button.clicked.connect(self._probe_oss_clicked)
        probe_row.addWidget(self.oss_probe_button)
        self.oss_status_label = QLabel("")
        self.oss_status_label.setWordWrap(True)
        probe_row.addWidget(self.oss_status_label, 1)
        oss_form.addRow(probe_row)
        layout.addWidget(self.oss_group)

        row = QHBoxLayout()
        self.record_button = QPushButton("● 开始录音")
        self.record_button.clicked.connect(self._toggle_record)
        row.addWidget(self.record_button)
        self.file_button = QPushButton("从文件选择…")
        self.file_button.clicked.connect(self._pick_file)
        row.addWidget(self.file_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.sample_label = QLabel("尚未准备样音")
        self.sample_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.sample_label)

        self.sample_options = QFrame()
        options = QFormLayout(self.sample_options)
        options.setContentsMargins(0, 0, 0, 0)
        self.sample_length_spin = NoWheelDoubleSpinBox()
        self.sample_length_spin.setDecimals(3)
        self.sample_length_spin.setRange(self._minimum_sample_seconds(), 30.0)
        self.sample_length_spin.setValue(30.0)
        self.sample_length_spin.setSuffix(" 秒")
        self.sample_length_spin.setToolTip("选择文件后自动取音频长度，最长 30 秒；可手动缩短，使用音频开头这一段。")
        options.addRow("参考音频时长", self.sample_length_spin)
        self.preprocess_check = QCheckBox("启用预处理（降噪、音频增强、音量规整）")
        self.preprocess_check.setChecked(False)
        self.preprocess_check.setEnabled(self.tts_model in VOICE_PREPROCESS_MODELS)
        options.addRow(self.preprocess_check)
        self.volume_normalization_check = QCheckBox("启用音量归一化")
        self.volume_normalization_check.setChecked(False)
        self.volume_normalization_check.setEnabled(self.channel == "legacy")
        options.addRow(self.volume_normalization_check)
        option_note = QLabel("录音棚或安静环境默认关闭两项处理，保留原始音色。灰色选项表示当前模型不支持；预处理本身也可能调整音量。")
        option_note.setWordWrap(True)
        options.addRow(option_note)
        layout.addWidget(self.sample_options)

        self.clone_button = QPushButton("创建克隆音色")
        self.clone_button.setEnabled(False)
        self.clone_button.setMinimumHeight(38)
        self.clone_button.clicked.connect(self._start_clone)
        outer.addWidget(self.clone_button)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        outer.addWidget(self.result_label)

        close = QPushButton("关闭")
        close.clicked.connect(self.reject)
        outer.addWidget(close)

    # --------------------------------------------------------- OSS 检测
    def _set_oss_status(self, message: str, color: str) -> None:
        self.oss_status_label.setText(message)
        self.oss_status_label.setStyleSheet(f"color: {color};")

    def _stored_oss_values(self) -> dict[str, str] | None:
        shared = self.settings.shared
        values = {
            "region": str(shared.get("oss_region", "cn-beijing") or "cn-beijing").strip(),
            "bucket": str(shared.get("oss_bucket", "") or "").strip(),
            "access_key_id": shared.get_oss_access_key_id(),
            "access_key_secret": shared.get_oss_access_key_secret(),
        }
        if not all(values.values()):
            return None
        return values

    def _persist_oss_values(self, values: dict[str, str]) -> None:
        # 凭据只进 Windows 凭据管理器；地域/Bucket 进配置文件（非敏感）。
        self.settings.shared.set_oss_access_key_id(values["access_key_id"])
        self.settings.shared.set_oss_access_key_secret(values["access_key_secret"])
        self.settings.shared.update(
            {"oss_region": values["region"], "oss_bucket": values["bucket"]}
        )

    def _probe_oss_clicked(self) -> None:
        values = self._current_oss_values()
        if values is None:
            self._set_oss_status("❌ 请先完整填写 地域 / Bucket / AccessKey ID / Secret", ERR_RED)
            return
        self._run_oss_probe(values)

    def _current_oss_values(self) -> dict[str, str] | None:
        """输入框优先，空缺处回退到已保存配置；四项齐全才返回。"""
        shared = self.settings.shared
        values = {
            "region": self.oss_region_edit.text().strip() or str(shared.get("oss_region", "cn-beijing")),
            "bucket": self.oss_bucket_edit.text().strip() or str(shared.get("oss_bucket", "")),
            "access_key_id": self.oss_ak_edit.text().strip() or shared.get_oss_access_key_id(),
            "access_key_secret": self.oss_sk_edit.text().strip() or shared.get_oss_access_key_secret(),
        }
        if not all(values.values()):
            return None
        return values

    def _maybe_auto_probe_oss(self) -> None:
        """进入旧版通道且配置齐全时自动检测一次，绿了就代表可以直接克隆。"""
        if self.channel != "legacy":
            return
        values = self._current_oss_values()
        if values is None:
            stored = self._stored_oss_values()
            if stored is None:
                self._set_oss_status("⚠️ 填写下方 OSS 信息后点「检测 OSS 并记住」，绿了以后就不用再填", WARN_ORANGE)
                return
            values = stored
        self._run_oss_probe(values)

    def _run_oss_probe(self, values: dict[str, str]) -> None:
        self.oss_probe_button.setEnabled(False)
        self._set_oss_status("正在检测 OSS 连通性（上传/删除临时小文件）…", TEXT_DIM)
        threading.Thread(target=self._oss_probe_worker, args=(values,), daemon=True).start()

    def _oss_probe_worker(self, values: dict[str, str]) -> None:
        import tempfile as _tempfile

        try:
            probe_path = Path(_tempfile.gettempdir()) / f"oss_probe_{int(time.time())}.wav"
            with wave.open(str(probe_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes(b"\x00\x00" * 800)
            uploader = OssTemporaryUploader(**values)
            uploaded = uploader.upload(probe_path)
            uploader.delete(uploaded.key)
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
        except Exception as exc:
            message = " ".join(str(exc).split())[:200]
            self.oss_probe_done.emit(False, message)
            return
        self._persist_oss_values(values)
        self.oss_probe_done.emit(True, "连通正常，配置已记住——以后直接录音克隆即可，无需再填")

    def _on_oss_probe_done(self, ok: bool, message: str) -> None:
        self.oss_probe_button.setEnabled(True)
        if ok:
            self._set_oss_status(f"✅ {message}", OK_GREEN)
        else:
            self._set_oss_status(f"❌ OSS 不可用：{message}", ERR_RED)

    def done(self, result: int) -> None:  # 关闭时把地域/Bucket 存下来（凭据仅在检测通过或克隆时保存）
        self._sample_generation += 1
        if self._recorder is not None and self._recorder.recording:
            self._recorder.stop()
        self._timer.stop()
        region = self.oss_region_edit.text().strip()
        bucket = self.oss_bucket_edit.text().strip()
        if region or bucket:
            try:
                self.settings.shared.update({"oss_region": region, "oss_bucket": bucket})
            except Exception:
                pass
        super().done(result)

    # ------------------------------------------------------------ recording
    def _toggle_record(self) -> None:
        if self._recorder is not None and self._recorder.recording:
            self._finish_recording()
            return
        self._clear_sample("正在启动录音…")
        try:
            self._recorder = _SampleRecorder(self.settings.shared.get("input_device"))
            self._recorder.start()
        except Exception as exc:
            self._recorder = None
            self.sample_label.setText(f"录音启动失败：{exc}")
            self.sample_label.setStyleSheet(f"color: {ERR_RED};")
            return
        self._record_started_at = time.monotonic()
        self.record_button.setText("■ 停止录音")
        self._timer.start()

    def _tick(self) -> None:
        if self._recorder is None or not self._recorder.recording:
            return
        elapsed = time.monotonic() - self._record_started_at
        self.sample_label.setText(f"录音中… {elapsed:.0f} 秒（{MAX_RECORD_SECONDS} 秒自动停止）")
        if elapsed >= MAX_RECORD_SECONDS:
            self._finish_recording()

    def _finish_recording(self) -> None:
        self._timer.stop()
        recorder = self._recorder
        if recorder is None:
            return
        duration = recorder.stop()
        self._recorder = None
        self.record_button.setText("● 重新录音")
        if duration < self._minimum_sample_seconds():
            self._clear_sample(f"录音太短（{duration:.1f} 秒），至少需要 {self._minimum_sample_seconds():g} 秒，请重录")
            self.sample_label.setStyleSheet(f"color: {ERR_RED};")
            return
        path = Path(tempfile.gettempdir()) / f"defense_voice_sample_{int(time.time())}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframesraw(bytes(recorder.frames))
        self._set_sample(path, duration)

    # ----------------------------------------------------------- file input
    def _pick_file(self) -> None:
        suffixes = " ".join("*" + suffix for suffix in sorted(SUPPORTED_AUDIO_SUFFIXES))
        path, _selected = QFileDialog.getOpenFileName(self, "选择样音录音", "", f"音频 ({suffixes})")
        if not path:
            return
        self.use_transcript.setChecked(False)
        self._clear_sample("正在检查样音…")
        generation = self._sample_generation
        def worker():
            try:
                duration = voice_sample_duration(path)
                message = ""
            except Exception as exc:
                duration, message = 0.0, str(exc)
            try:
                self.sample_checked.emit(generation, path, duration, message)
            except RuntimeError:
                pass
        threading.Thread(target=worker, name="defense-sample-info", daemon=True).start()

    @Slot(int, str, float, str)
    def _on_sample_checked(self, generation, path, duration, message):
        if generation != self._sample_generation:
            return
        if message or duration < self._minimum_sample_seconds():
            self._clear_sample(message or f"样音至少需要 {self._minimum_sample_seconds():g} 秒，请重录")
            self.sample_label.setStyleSheet(f"color: {ERR_RED};")
            return
        self._set_sample(Path(path), duration)

    def _minimum_sample_seconds(self):
        return 3.0 if self.tts_model in VOICE_PREPROCESS_MODELS else 5.0

    # ---------------------------------------------------------------- state
    def _clear_sample(self, message: str) -> None:
        self._sample_generation += 1
        self._sample_path = None
        self._sample_seconds = 0.0
        self.clone_button.setEnabled(False)
        self.sample_label.setText(message)
        self.sample_label.setStyleSheet(f"color: {TEXT_DIM};")

    def _set_sample(self, path: Path, seconds: float) -> None:
        self._sample_path = path
        self._sample_seconds = seconds
        self.sample_length_spin.setMaximum(min(30.0, seconds))
        self.sample_length_spin.setValue(min(30.0, seconds))
        self.sample_label.setText(f"样音已就绪：{seconds:.3f} 秒 · 默认使用前 {min(30.0, seconds):.3f} 秒")
        self.sample_label.setStyleSheet(f"color: {OK_GREEN};")
        self.clone_button.setEnabled(True)

    # ---------------------------------------------------------------- clone
    # ------------------------------------------------------- 复刻通道逻辑
    def _update_channel_ui(self) -> None:
        # 防呆：通道由 TTS 模型家族唯一决定，界面上只出现唯一正确项，选无可选。
        expected = clone_channel_for_model(self.tts_model)
        self.channel = expected
        self.channel_combo.clear()
        self.channel_combo.addItem(CHANNEL_LABELS[expected], expected)
        self.oss_group.setVisible(expected == "legacy")
        self.use_transcript.setEnabled(expected == "qwen3")
        if expected == "legacy":
            self.channel_note.setText(
                "旧版通道流程：样音 → 私有 OSS 中转 → 提交复刻 → 等待部署（1~3 分钟）→ 自动删除样音。"
            )
            self._maybe_auto_probe_oss()
        else:
            self.channel_note.setText("Qwen3 直传流程：样音 Base64 上传 → 秒级创建，无需 OSS。")
            self._set_oss_status("✅ 直传通道：录完音点「创建克隆音色」即可，无需 OSS。", OK_GREEN)

    def _collect_oss_values(self) -> dict[str, str]:
        region = self.oss_region_edit.text().strip() or str(self.settings.shared.get("oss_region", "cn-beijing"))
        bucket = self.oss_bucket_edit.text().strip()
        access_key_id = self.oss_ak_edit.text().strip() or self.settings.shared.get_oss_access_key_id()
        access_key_secret = self.oss_sk_edit.text().strip() or self.settings.shared.get_oss_access_key_secret()
        if not bucket:
            raise OssUploadError("请填写 OSS Bucket 名称")
        if not (region and access_key_id and access_key_secret):
            raise OssUploadError("OSS 配置不完整：需要地域、Bucket、AccessKey ID 和 Secret")
        # 凭据只进 Windows 凭据管理器，不落盘。
        if self.oss_ak_edit.text().strip():
            self.settings.shared.set_oss_access_key_id(self.oss_ak_edit.text().strip())
        if self.oss_sk_edit.text().strip():
            self.settings.shared.set_oss_access_key_secret(self.oss_sk_edit.text().strip())
        self.settings.shared.update({"oss_region": region, "oss_bucket": bucket})
        return {
            "region": region,
            "bucket": bucket,
            "access_key_id": access_key_id,
            "access_key_secret": access_key_secret,
        }

    def _start_clone(self) -> None:
        if self._cloning or self._sample_path is None:
            return
        expected = clone_channel_for_model(self.tts_model)
        if self.channel != expected:
            self._set_result(
                f"复刻通道与模型不匹配：{self.tts_model} 属于「{CHANNEL_LABELS[expected].split('（')[0]}」通道。"
                f"请把上方「复刻通道」切回后重试。（刚才的报错 preprocess service not found 就是这个原因）",
                ERR_RED,
            )
            return
        source = self._sample_path
        clone_options = {
            "max_seconds": self.sample_length_spin.value(),
            "min_seconds": self._minimum_sample_seconds(),
            "preprocess": self.preprocess_check.isEnabled() and self.preprocess_check.isChecked(),
            "volume_normalization": self.volume_normalization_check.isEnabled() and self.volume_normalization_check.isChecked(),
        }
        transcript = self.passage_edit.toPlainText().strip() if self.use_transcript.isChecked() else ""
        self._cloning = True
        self.clone_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.file_button.setEnabled(False)
        if self.channel == "legacy":
            try:
                oss_values = self._collect_oss_values()
            except OssUploadError as exc:
                self._cloning = False
                self.clone_button.setEnabled(True)
                self.record_button.setEnabled(True)
                self.file_button.setEnabled(True)
                self._set_result(str(exc), ERR_RED)
                return
            self._set_result("正在上传样音到 OSS 并提交复刻，之后还需等待部署（共 1~3 分钟）…", TEXT_DIM)
            preferred = f"def{int(time.time()) % 10_000_000}"
            threading.Thread(
                target=self._legacy_clone_worker, args=(source, preferred, oss_values, clone_options), daemon=True
            ).start()
            return
        self._set_result("正在上传样音并创建音色（约 10~30 秒）…", TEXT_DIM)
        threading.Thread(target=self._clone_worker, args=(source, transcript, clone_options), daemon=True).start()

    def _legacy_clone_worker(self, source: Path, preferred: str, oss_values: dict[str, str], options: dict) -> None:
        try:
            client = BailianClient(
                self.settings.get_api_key(),
                self.settings.workspace_id,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=120,
            )
            with normalized_voice_sample(source, max_seconds=options["max_seconds"], min_seconds=options["min_seconds"]) as wav:
                self.clone_progress.emit(inspect_voice_sample(wav))
                uploader = OssTemporaryUploader(**oss_values)
                uploaded = uploader.upload(wav)
            self.clone_progress.emit("样音已上传，正在提交复刻并等待百炼部署（请勿关闭窗口）…")
            try:
                voice_id = client.clone_voice(
                    target_model=self.tts_model,
                    prefix=preferred,
                    audio_url=uploaded.signed_url,
                    language="zh",
                    max_seconds=options["max_seconds"],
                    preprocess=options["preprocess"],
                    volume_normalization=options["volume_normalization"],
                )
                client.wait_for_voice_ready(
                    voice_id,
                    timeout=240.0,
                    on_status=lambda message: self.clone_progress.emit(message),
                )
            finally:
                try:
                    uploader.delete(uploaded.key)
                except Exception:
                    self.clone_progress.emit("OSS 临时样音删除失败，可稍后在 OSS 控制台手动删除 temporary/ 下的对象。")
        except (ApiError, VoiceSampleError, OssUploadError) as exc:
            self.clone_finished.emit("", str(exc), True)
        except Exception as exc:  # pragma: no cover - defensive
            self.clone_finished.emit("", f"克隆失败：{exc}", True)
        else:
            self.clone_finished.emit(voice_id, "", False)

    def _on_clone_progress(self, message: str) -> None:
        self._set_result(message, TEXT_DIM)

    def _clone_worker(self, source: Path, transcript: str, options: dict) -> None:
        try:
            client = BailianClient(
                self.settings.get_api_key(),
                self.settings.workspace_id,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=120,
            )
            with normalized_voice_sample(source, max_seconds=options["max_seconds"], min_seconds=options["min_seconds"]) as wav:
                self.clone_progress.emit(inspect_voice_sample(wav))
                encoded = base64.b64encode(wav.read_bytes()).decode("ascii")
            preferred = f"defense_{int(time.time()) % 10_000_000}"
            voice_id, warning = client.clone_qwen_voice(
                target_model=self.tts_model,
                preferred_name=preferred,
                audio_data="data:audio/wav;base64," + encoded,
                language="zh",
                transcript=transcript,
            )
        except (ApiError, VoiceSampleError) as exc:
            self.clone_finished.emit("", str(exc), True)
        except Exception as exc:  # pragma: no cover - defensive
            self.clone_finished.emit("", f"克隆失败：{exc}", True)
        else:
            self.clone_finished.emit(voice_id, warning, False)

    # ------------------------------------------------------------- 音色库
    def _ensure_current_voice_in_library(self) -> None:
        """默认就把当前使用的音色展示在列表里，无需手动存入。"""
        voice = str(self.settings.get("tts_voice_id", "") or "").strip()
        if not voice:
            return
        entries = self._library_entries()
        if any(entry.get("voice_id") == voice for entry in entries):
            return
        target_model = str(self.settings.get("tts_model", ""))
        entries.append(
            {
                "voice_id": voice,
                "target_model": target_model,
                "name": "当前音色（自动收录）",
                "channel": clone_channel_for_model(target_model),
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
        self._save_library(entries)

    def _library_entries(self) -> list[dict]:
        value = self.settings.get("voice_library", [])
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict) and entry.get("voice_id")]

    def _save_library(self, entries: list[dict]) -> None:
        self.settings.set("voice_library", entries)

    def _refresh_library(self) -> None:
        current = str(self.settings.get("tts_voice_id", "") or "")
        entries = self._library_entries()
        self.library_table.setRowCount(0)
        for entry in entries:
            row = self.library_table.rowCount()
            self.library_table.insertRow(row)
            name = str(entry.get("name") or "未命名")
            if entry.get("voice_id") == current:
                name += "（当前）"
            values = [
                name,
                str(entry.get("voice_id", "")),
                str(entry.get("target_model", "")),
                str(entry.get("created_at", "")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.library_table.setItem(row, column, item)

    @staticmethod
    def _voice_family(model: str) -> str:
        model = model.lower()
        if model.startswith("qwen3-tts-vc"):
            return "qwen3"
        if model.startswith("cosyvoice"):
            return "cosyvoice"
        if model.startswith("qwen-audio"):
            return "qwen-audio"
        return model

    def _selected_voice_entry(self) -> dict | None:
        row = self.library_table.currentRow()
        entries = self._library_entries()
        if row < 0 or row >= len(entries):
            return None
        return entries[row]

    def _apply_voice(self, voice_id: str, target_model: str) -> None:
        self.created_voice_id = voice_id
        self._set_result(f"已选择音色：{voice_id}（绑定模型 {target_model}），关闭后自动填入 voice_id 输入框", OK_GREEN)
        self.accept()

    def _use_selected_voice(self) -> None:
        entry = self._selected_voice_entry()
        if entry is None:
            QMessageBox.information(self, "音色库", "请先在列表中选中一个音色")
            return
        voice_id = str(entry.get("voice_id", ""))
        if str(entry.get("target_model", "")) != self.tts_model:
            answer = QMessageBox.question(
                self,
                "音色与模型不匹配",
                f"该音色绑定的是 {entry.get('target_model', '未知模型')}，与当前 TTS 模型 "
                f"{self.tts_model} 绑定模型不同。请返回设置页选择该音色对应的模型。仍要选用吗？",
            )
            if answer != QMessageBox.Yes:
                return
        self._apply_voice(voice_id, str(entry.get("target_model", "")))

    def _save_current_voice(self) -> None:
        voice_id = str(self.settings.get("tts_voice_id", "") or "").strip()
        if not voice_id:
            QMessageBox.information(self, "音色库", "当前设置里还没有 voice_id 可存")
            return
        entries = self._library_entries()
        if any(entry.get("voice_id") == voice_id for entry in entries):
            QMessageBox.information(self, "音色库", "该 voice_id 已在列表中")
            return
        target_model = str(self.settings.get("tts_model", ""))
        entries.append(
            {
                "voice_id": voice_id,
                "target_model": target_model,
                "name": f"手动存入 {time.strftime('%m-%d %H:%M')}",
                "channel": clone_channel_for_model(target_model),
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
        self._save_library(entries)
        self._refresh_library()

    def _delete_selected_voice(self) -> None:
        row = self.library_table.currentRow()
        entries = self._library_entries()
        if row < 0 or row >= len(entries):
            return
        removed = entries.pop(row)
        self._save_library(entries)
        self._refresh_library()
        self._set_result(f"已从列表移除 {removed.get('voice_id')}（不影响阿里云上的音色本体）", TEXT_DIM)

    def _remember_voice(self, voice_id: str) -> None:
        entries = [entry for entry in self._library_entries() if entry.get("voice_id") != voice_id]
        entries.append(
            {
                "voice_id": voice_id,
                "target_model": self.tts_model,
                "name": f"克隆 {time.strftime('%m-%d %H:%M')}",
                "channel": self.channel,
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
        self._save_library(entries)
        self._refresh_library()

    def _on_clone_finished(self, voice_id: str, message: str, is_error: bool) -> None:
        self._cloning = False
        self.clone_button.setEnabled(self._sample_path is not None)
        self.record_button.setEnabled(True)
        self.file_button.setEnabled(True)
        if is_error:
            self._set_result(message, ERR_RED)
            return
        self._remember_voice(voice_id)
        self.created_voice_id = voice_id
        text = f"✅ 音色已创建：{voice_id}（已自动填入设置页的 voice_id 输入框）"
        self._set_result(text, OK_GREEN)
        if message:
            self.result_label.setText(text + f"\n⚠️ {message}")
            self.result_label.setStyleSheet(f"color: {WARN_ORANGE};")

    def _set_result(self, message: str, color: str) -> None:
        self.result_label.setText(message)
        self.result_label.setStyleSheet(f"color: {color};")

    def closeEvent(self, event) -> None:  # also covers reject via dialog machinery
        if self._recorder is not None and self._recorder.recording:
            self._recorder.stop()
        self._timer.stop()
        super().closeEvent(event)


def _annotate_device(name: str, direction: str) -> str:
    """给设备名加方向与用途括号，避免 CABLE Input/Output 分不清。"""
    low = name.lower()
    if "cable output" in low:
        return f"{name}（虚拟声卡·输入端，勿在“程序输出”里选它）"
    if "cable input" in low or low.startswith("cable in"):
        return f"{name}（虚拟声卡·输出端，选它送进会议）"
    if direction == "输入":
        note = "输入·麦克风" if any(word in low for word in ("mic", "麦克风", "麦克风")) else "输入设备"
    elif direction == "回环":
        note = "回环·录制此设备的播放声"
    else:
        note = "输出·扬声器/耳机" if any(word in low for word in ("speaker", "扬声器", "headset", "耳机", "digital")) else "输出设备"
    return f"{name}（{note}）"


def _device_combo(
    combo: QComboBox,
    items: list,
    current,
    allow_default: str | None = None,
    *,
    direction: str = "输出",
    highlight_name: str = "",
) -> None:
    combo.clear()
    if allow_default is not None:
        combo.addItem(allow_default, None)
    for item in items:
        host_api = getattr(item, "host_api", "WASAPI 回环")
        combo.addItem(f"[{item.index}] {_annotate_device(item.name, direction)} · {host_api}", item.index)
    if current is not None:
        position = combo.findData(current)
        if position >= 0:
            combo.setCurrentIndex(position)
    if highlight_name:
        low = highlight_name.lower()
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data is not None and low in str(combo.itemText(index)).lower():
                combo.setItemData(index, QColor("#ffd400"), Qt.ForegroundRole)
                combo.setItemData(index, QColor("#3a3210"), Qt.BackgroundRole)
                combo.setCurrentIndex(index)
                break


class SettingsDialog(QDialog):
    test_line = Signal(str)
    test_done = Signal()

    def __init__(
        self, parent: QWidget, settings: DefenseSettings, *, active_loopback_name: str = ""
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.active_loopback_name = active_loopback_name
        from .. import __version__

        self.setWindowTitle(f"设置 · 答辩模式 v{__version__}")
        self.resize(1120, 780)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)

        left_form = QFormLayout()
        right_form = QFormLayout()
        left_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        right_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        left_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        right_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        left_wrap = QWidget()
        left_wrap.setLayout(left_form)
        right_wrap = QWidget()
        right_wrap.setLayout(right_form)
        columns = QHBoxLayout()
        columns.addWidget(left_wrap, 1)
        columns.addWidget(right_wrap, 1)
        layout.addLayout(columns)
        form = left_form  # 占位：下方逐行改挂到 left/right
        self.workspace_edit = QLineEdit(str(settings.shared.get("workspace_id", "")))
        left_form.addRow("百炼 Workspace ID", self.workspace_edit)
        self.workspace_hint = QLabel("")
        left_form.addRow("", self.workspace_hint)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("已保存在 Windows 凭据管理器，留空表示不修改")
        left_form.addRow("百炼 API Key", self.api_key_edit)

        self.api_key_hint = QLabel("")
        left_form.addRow("", self.api_key_hint)

        test_row = QHBoxLayout()
        self.test_button = QPushButton("测试连接")
        self.test_button.setToolTip("依次验证：翻译大模型 → 实时识别 → 克隆音色合成（用当前输入框里的值）")
        self.test_button.clicked.connect(self._run_connection_test)
        test_row.addWidget(self.test_button)
        test_row.addStretch(1)
        left_form.addRow("", test_row)
        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setTextFormat(Qt.RichText)
        left_form.addRow("", self.test_result_label)
        self.test_line.connect(self._on_test_line)
        self.test_done.connect(lambda: self.test_button.setEnabled(True))
        self._test_lines: list[str] = []
        self.workspace_edit.textChanged.connect(lambda _text: self._update_credential_hints())
        self.api_key_edit.textChanged.connect(lambda _text: self._update_credential_hints())
        self._update_credential_hints()

        self.voice_combo = NoWheelComboBox()
        self.voice_combo.setMaxVisibleItems(12)
        self._refresh_voice_combo()
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        clone_button = QPushButton("一键克隆音色…")
        clone_button.clicked.connect(self._open_clone_voice)
        voice_row.addWidget(clone_button)
        add_voice_button = QPushButton("＋")
        add_voice_button.setFixedWidth(34)
        add_voice_button.setToolTip("手动添加控制台复刻的 voice_id 到音色库")
        add_voice_button.clicked.connect(self._add_manual_voice)
        voice_row.addWidget(add_voice_button)
        left_form.addRow("克隆音色 voice_id", voice_row)
        self.voice_hint = QLabel("")
        self.voice_hint.setWordWrap(True)
        left_form.addRow("", self.voice_hint)

        self.tts_model_combo = NoWheelComboBox()
        fill_model_combo(self.tts_model_combo, TTS_MODELS, str(settings.get("tts_model")))
        self.tts_model_combo.setToolTip(
            "支持截图中的 CosyVoice 与 Qwen HTTP 路线。切换模型必须同时选择绑定到该模型的音色。"
        )
        left_form.addRow("TTS 模型（与 voice_id 绑定一致）", self.tts_model_combo)
        self.tts_model_combo.currentIndexChanged.connect(lambda _index: self._update_voice_hint())
        self.voice_combo.currentIndexChanged.connect(lambda _index: self._update_voice_hint())
        self._update_voice_hint()

        self.rate_spin = NoWheelDoubleSpinBox()
        self.rate_spin.setRange(0.5, 2.0)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setValue(float(settings.get("tts_rate", 1.0)))
        self.rate_spin.setSuffix(" x")
        self.rate_spin.setToolTip("答辩建议 0.90~0.95，听着更从容；对高清模型不生效")
        left_form.addRow("语速", self.rate_spin)

        self.pitch_spin = NoWheelDoubleSpinBox()
        self.pitch_spin.setRange(0.5, 2.0)
        self.pitch_spin.setSingleStep(0.05)
        self.pitch_spin.setDecimals(2)
        self.pitch_spin.setValue(float(settings.get("tts_pitch", 1.0)))
        self.pitch_spin.setSuffix(" x")
        self.pitch_spin.setToolTip("低于 1.0 更低沉稳重；仅实时版和 CosyVoice 生效，高清模型忽略")
        left_form.addRow("音调", self.pitch_spin)

        self.volume_spin = NoWheelSpinBox()
        self.volume_spin.setRange(0, 100)
        self.volume_spin.setValue(int(settings.get("tts_volume", 55)))
        self.volume_spin.setToolTip("会议里偏小就调高；对高清模型不生效")
        left_form.addRow("音量", self.volume_spin)

        current_instruction = str(settings.get("tts_instruction", "") or "").strip()
        self.instruction_preset_combo = NoWheelComboBox()
        for label, instruction in INSTRUCTION_PRESETS:
            self.instruction_preset_combo.addItem(label, instruction)
        self.instruction_preset_combo.addItem("自定义指令…", CUSTOM_INSTRUCTION)
        preset_index = self.instruction_preset_combo.findData(current_instruction)
        if preset_index < 0:
            preset_index = self.instruction_preset_combo.findData(CUSTOM_INSTRUCTION)
        self.instruction_preset_combo.setCurrentIndex(preset_index)
        left_form.addRow("语气风格", self.instruction_preset_combo)

        self.instruction_edit = QLineEdit(current_instruction)
        self.instruction_edit.setPlaceholderText(
            "自然模式留空会使用推荐语气；也可输入英文自定义指令（CosyVoice 限 100 字符单位）"
        )
        left_form.addRow("语气指令", self.instruction_edit)
        self.instruction_note = QLabel("")
        self.instruction_note.setWordWrap(True)
        left_form.addRow("", self.instruction_note)
        self.instruction_preset_combo.currentIndexChanged.connect(self._apply_instruction_preset)
        self.instruction_edit.textEdited.connect(self._mark_instruction_custom)
        self.tts_model_combo.currentIndexChanged.connect(self._update_instruction_support)
        self._update_instruction_support()

        self.ambience_combo = NoWheelComboBox()
        for key, preset in AMBIENCE_PRESETS.items():
            self.ambience_combo.addItem(preset.label, key)
        ambience_mode = str(settings.get("ambience_mode", "off") or "off")
        self.ambience_combo.setCurrentIndex(max(0, self.ambience_combo.findData(ambience_mode)))
        left_form.addRow("背景环境", self.ambience_combo)
        ambience_note = QLabel(
            "在英文语音下混入极轻的合成房间底噪，并在句尾自然淡出，避免像录音一样突然切成绝对静音。"
            "它不是音乐；正式答辩建议先选“极轻”，会议软件强降噪时可能听不见。"
        )
        ambience_note.setWordWrap(True)
        ambience_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", ambience_note)
        self.speech_mode_combo = NoWheelComboBox()
        self.speech_mode_combo.addItem("自然度优先 · 完整翻译 + 音频缓冲", "natural")
        self.speech_mode_combo.addItem("低延迟 · 分句流式发声", "streaming")
        self.speech_mode_combo.setCurrentIndex(max(0, self.speech_mode_combo.findData(settings.get("speech_mode", "natural"))))
        left_form.addRow("发声方式", self.speech_mode_combo)
        mode_note = QLabel("自然模式等待完整译文和音频后播放；断句停顿至少 700 ms。Qwen VC 不支持语气指令，语速/音调先保持 1.0。")
        mode_note.setWordWrap(True)
        mode_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", mode_note)

        self.output_dir_edit = QLineEdit(str(settings.get("output_directory", "")))
        self.output_dir_edit.setPlaceholderText("留空 = 我的文档/DefenseMode；每次答辩自动建 答辩_日期_时间 子文件夹")
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        browse_button = QPushButton("浏览…")
        browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_button)
        right_form.addRow("记录保存位置", output_row)

        self.llm_combo = NoWheelComboBox()
        fill_model_combo(self.llm_combo, CHAT_MODELS, str(settings.get("llm_model")))
        form.addRow("翻译大模型", self.llm_combo)

        self.asr_combo = NoWheelComboBox()
        fill_model_combo(self.asr_combo, ASR_MODELS, str(settings.get("asr_model")))
        right_form.addRow("我方识别模型", self.asr_combo)

        self.direct_hotkey_combo = self._hotkey_combo("direct_hotkey", "f5")
        right_form.addRow("原声快捷键（按住说话）", self.direct_hotkey_combo)
        self.translate_hotkey_combo = self._hotkey_combo("translate_hotkey", "f9")
        right_form.addRow("翻译快捷键（按一下开/关）", self.translate_hotkey_combo)
        self.cancel_hotkey_combo = self._hotkey_combo("cancel_hotkey", "esc")
        right_form.addRow("停止快捷键", self.cancel_hotkey_combo)

        self.proxy_combo = NoWheelComboBox()
        self.proxy_combo.addItem("直连国内（推荐，开会开 VPN 时选这个）", "direct")
        self.proxy_combo.addItem("跟随系统代理", "system")
        self.proxy_combo.addItem("手动设置代理", "manual")
        current_mode = self.settings.proxy_mode
        self.proxy_combo.setCurrentIndex(max(self.proxy_combo.findData(current_mode), 0))
        self.proxy_edit = QLineEdit(str(settings.shared.get("http_proxy", "")))
        self.proxy_edit.setPlaceholderText("手动模式生效，例如 127.0.0.1:7890")
        self.proxy_edit.setEnabled(current_mode == "manual")
        self.proxy_combo.currentIndexChanged.connect(
            lambda _index: self.proxy_edit.setEnabled(self.proxy_combo.currentData() == "manual")
        )
        proxy_row = QHBoxLayout()
        proxy_row.addWidget(self.proxy_combo, 1)
        proxy_row.addWidget(self.proxy_edit, 1)
        right_form.addRow("网络代理", proxy_row)

        self.silence_spin = NoWheelSpinBox()
        self.silence_spin.setRange(200, 1500)
        self.silence_spin.setSingleStep(50)
        self.silence_spin.setValue(int(settings.get("vad_silence_ms", 500)))
        self.silence_spin.setSuffix(" ms")
        right_form.addRow("断句静音（越小结句越快）", self.silence_spin)

        self.turns_spin = NoWheelSpinBox()
        self.turns_spin.setRange(4, 40)
        self.turns_spin.setValue(int(settings.get("max_context_turns", 14)))
        right_form.addRow("上下文携带句数", self.turns_spin)

        inputs, outputs = _safe_list_audio_devices()
        loopbacks = [device for device in _safe_list_loopback_devices()
                     if not is_program_output_loopback(device.name)]
        self._loopbacks_cache = loopbacks
        self.mic_combo = NoWheelComboBox()
        _device_combo(
            self.mic_combo, inputs, settings.shared.get("input_device"),
            allow_default="系统默认输入", direction="输入",
        )
        self.mic_locked = True
        self.mic_lock_button = QPushButton("🔒 已锁定")
        self.mic_lock_button.setFixedWidth(92)
        self.mic_lock_button.clicked.connect(self._toggle_mic_lock)
        mic_lock_row = QHBoxLayout()
        mic_lock_row.addWidget(self.mic_lock_button)
        mic_lock_row.addWidget(self.mic_combo, 1)
        right_form.addRow("麦克风", mic_lock_row)
        self.cable_combo = NoWheelComboBox()
        self.cable_note = QLabel("")
        self.cable_lock_button = QPushButton("🔒 已锁定")
        self.cable_lock_button.setFixedWidth(92)
        self.cable_lock_button.clicked.connect(self._toggle_cable_lock)
        self.cable_locked = True
        self._outputs_cache = outputs
        self.cable_locked_index = self._apply_cable_lock(self.cable_combo, outputs)
        cable_lock_row = QHBoxLayout()
        cable_lock_row.addWidget(self.cable_lock_button)
        cable_lock_row.addWidget(self.cable_combo, 1)
        right_form.addRow("程序输出（已锁定虚拟声卡）", cable_lock_row)
        right_form.addRow("", self.cable_note)
        self.loopback_combo = NoWheelComboBox()
        self.loopback_note = QLabel("")
        self.loopback_locked = True
        chosen_loopback = choose_committee_loopback(
            loopbacks, settings.shared.get("loopback_device"),
            str(settings.shared.get("loopback_device_name", "") or ""),
        )
        _device_combo(
            self.loopback_combo, loopbacks, chosen_loopback.index if chosen_loopback else None,
            allow_default="系统默认输出（推荐：微信/Teams 默认从这里出声）", direction="回环",
            highlight_name=self.active_loopback_name,
        )
        if self.active_loopback_name:
            low = self.active_loopback_name.lower()
            for index in range(self.loopback_combo.count()):
                if low in self.loopback_combo.itemText(index).lower():
                    self.loopback_combo.setCurrentIndex(index)
                    break
        self.loopback_locked_index = self.loopback_combo.currentData()
        loopback_lock_row = QHBoxLayout()
        self.loopback_lock_button = QPushButton("🔒 已锁定")
        self.loopback_lock_button.setFixedWidth(92)
        self.loopback_lock_button.clicked.connect(self._toggle_loopback_lock)
        loopback_lock_row.addWidget(self.loopback_lock_button)
        loopback_lock_row.addWidget(self.loopback_combo, 1)
        right_form.addRow("对方声音回环（对方声音从哪个设备播就选哪个）", loopback_lock_row)
        right_form.addRow("", self.loopback_note)
        self._apply_loopback_lock_state()
        self.monitor_check = QCheckBox("本机试听（戴耳机防止回声）")
        self.monitor_check.setChecked(bool(settings.get("monitor_enabled", True)))
        right_form.addRow("", self.monitor_check)
        self.monitor_combo = NoWheelComboBox()
        _device_combo(
            self.monitor_combo, outputs, settings.get("monitor_output_device"),
            allow_default="系统默认输出", direction="输出",
        )
        self.monitor_locked = True
        self.monitor_lock_button = QPushButton("🔒 已锁定")
        self.monitor_lock_button.setFixedWidth(92)
        self.monitor_lock_button.clicked.connect(self._toggle_monitor_lock)
        monitor_lock_row = QHBoxLayout()
        monitor_lock_row.addWidget(self.monitor_lock_button)
        monitor_lock_row.addWidget(self.monitor_combo, 1)
        right_form.addRow("试听设备", monitor_lock_row)
        self._apply_mic_lock_state()
        self._apply_monitor_lock_state()

        glossary_caption = QLabel("强制术语表（中 → 英，翻译时逐字一致）")
        layout.addWidget(glossary_caption)
        self.terms_table = QTableWidget(0, 2)
        self.terms_table.setHorizontalHeaderLabels(["中文", "英文"])
        self.terms_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.terms_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.terms_table.setMinimumHeight(180)
        for item in settings.glossary_terms():
            row = self.terms_table.rowCount()
            self.terms_table.insertRow(row)
            self.terms_table.setItem(row, 0, QTableWidgetItem(item["source"]))
            self.terms_table.setItem(row, 1, QTableWidgetItem(item["target"]))
        layout.addWidget(self.terms_table, 1)

        term_buttons = QHBoxLayout()
        add_button = QPushButton("添加术语")
        add_button.clicked.connect(self._add_term)
        remove_button = QPushButton("删除选中")
        remove_button.clicked.connect(self._remove_terms)
        import_button = QPushButton("导入 JSON")
        import_button.clicked.connect(self._import_terms)
        term_buttons.addWidget(add_button)
        term_buttons.addWidget(remove_button)
        term_buttons.addWidget(import_button)
        term_buttons.addStretch(1)
        layout.addLayout(term_buttons)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存")
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        outer.addLayout(buttons)

    # ------------------------------------------------------ connection test
    def _on_test_line(self, html: str) -> None:
        self._test_lines.append(html)
        self.test_result_label.setText("<br>".join(self._test_lines))

    def _emit_test_line(self, html: str) -> None:
        try:
            self.test_line.emit(html)
        except RuntimeError:
            pass  # 对话框已关闭，丢弃结果即可

    def _run_connection_test(self) -> None:
        workspace = self.workspace_edit.text().strip()
        api_key = self.api_key_edit.text().strip() or self.settings.get_api_key()
        if not workspace or not api_key:
            self._test_lines = [f"<span style='color:{ERR_RED}'>❌ 请先填写 Workspace ID 和 API Key</span>"]
            self.test_result_label.setText(self._test_lines[0])
            return
        self.test_button.setEnabled(False)
        self._test_lines = [f"<span style='color:{TEXT_DIM}'>测试中…</span>"]
        self.test_result_label.setText(self._test_lines[0])
        args = (
            workspace,
            api_key,
            self._current_voice_id(),
            str(self.llm_combo.currentData()),
            str(self.asr_combo.currentData()),
            str(self.tts_model_combo.currentData()),
        )
        threading.Thread(target=self._test_worker, args=args, daemon=True).start()

    def _test_worker(self, workspace: str, api_key: str, voice: str, llm_model: str, asr_model: str, tts_model: str) -> None:
        proxy = self.settings.ws_proxy

        def brief(message: str) -> str:
            message = " ".join(str(message).split())
            return message[:140] + ("…" if len(message) > 140 else "")

        try:
            from .memory import MeetingMemory

            translator = ContextTranslator(
                api_key=api_key,
                workspace_id=workspace,
                model=llm_model,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
            )
            result = translator.translate(MeetingMemory(), "zh2en", "你好。")
            self._emit_test_line(
                f"<span style='color:{OK_GREEN}'>✅ 翻译大模型 {llm_model} 可用（返回：{result[:30]}）</span>"
            )
        except Exception as exc:
            self._emit_test_line(f"<span style='color:{ERR_RED}'>❌ 翻译大模型 {llm_model} 失败：{brief(exc)}</span>")
        try:
            asr = create_realtime_asr(
                api_key=api_key,
                workspace_id=workspace,
                values={"asr_model": asr_model, "vad_threshold": 0.0, "vad_silence_ms": 500, "translation_terms": ""},
                language="zh",
                on_preview=lambda *_args: None,
                on_status=lambda *_args: None,
                on_error=lambda *_args: None,
                proxy=proxy,
            )
            asr.start()
            asr.wait_ready(timeout=12)
            asr.close()
            self._emit_test_line(f"<span style='color:{OK_GREEN}'>✅ 实时识别 {asr_model} 连接正常</span>")
        except Exception as exc:
            self._emit_test_line(f"<span style='color:{ERR_RED}'>❌ 实时识别 {asr_model} 失败：{brief(exc)}</span>")
        if voice:
            try:
                voice_probe(self.settings, workspace, api_key, voice, tts_model)
                self._emit_test_line(
                    f"<span style='color:{OK_GREEN}'>✅ 克隆音色有效（{tts_model}），已在本机播放试听，请确认是你的音色</span>"
                )
            except Exception as exc:
                self._emit_test_line(f"<span style='color:{ERR_RED}'>❌ 音色合成失败：{brief(exc)}</span>")
        else:
            self._emit_test_line(
                f"<span style='color:{WARN_ORANGE}'>⚠️ 未填写 voice_id，已跳过音色测试</span>"
            )
        try:
            self.test_done.emit()
        except RuntimeError:
            pass

    def _refresh_voice_combo(self, keep: str | None = None) -> None:
        current = keep if keep is not None else str(self.settings.get("tts_voice_id", "") or "")
        combo = self.voice_combo
        combo.clear()
        entries = self.settings.get("voice_library", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("voice_id"):
                    continue
                label = f"{entry.get('name', '未命名')} · {entry.get('target_model', '')}"
                combo.addItem(label, entry.get("voice_id"))
        if current and combo.findData(current) < 0:
            combo.addItem(f"{current} · 未入库", current)
        combo.setCurrentIndex(max(combo.findData(current), 0))

    def _add_manual_voice(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        voice_id, ok = QInputDialog.getText(self, "添加控制台音色", "粘贴控制台复刻的 voice_id：")
        voice_id = (voice_id or "").strip()
        if not ok or not voice_id:
            return
        target_model = str(self.tts_model_combo.currentData())
        entries = self.settings.get("voice_library", [])
        if not isinstance(entries, list):
            entries = []
        if any(isinstance(e, dict) and e.get("voice_id") == voice_id for e in entries):
            QMessageBox.information(self, "音色库", "该 voice_id 已在音色库中")
            self._refresh_voice_combo(keep=voice_id)
            return
        entries.append(
            {
                "voice_id": voice_id,
                "target_model": target_model,
                "name": f"手动添加 {time.strftime('%m-%d %H:%M')}",
                "channel": clone_channel_for_model(target_model),
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
        self.settings.set("voice_library", entries)
        self._refresh_voice_combo(keep=voice_id)

    def _current_voice_id(self) -> str:
        return str(self.voice_combo.currentData() or "").strip()

    def _apply_cable_lock(self, combo: QComboBox, outputs: list) -> int | None:
        """程序输出防呆：默认锁定 CABLE Input（虚拟声卡，黄色高亮）；解锁仅供高级用途。"""
        self._outputs_cache = outputs
        cable = next((d for d in outputs if "cable input" in d.name.lower()), None)
        self.cable_locked_index = cable.index if cable is not None else None
        _device_combo(combo, outputs, None, direction="输出")
        if cable is None:
            combo.setEnabled(True)
            self.cable_locked = False
            self.cable_lock_button.setText("🔒 无虚拟声卡")
            self.cable_lock_button.setEnabled(False)
            self.cable_note.setText("❌ 未检测到 CABLE Input：请安装 VB-CABLE 驱动并重启 Windows")
            self.cable_note.setStyleSheet(f"color: {ERR_RED};")
            return None
        position = combo.findData(cable.index)
        combo.setCurrentIndex(max(position, 0))
        combo.setItemData(position, QColor("#ffd400"), Qt.ForegroundRole)
        combo.setItemData(position, QColor("#3a3210"), Qt.BackgroundRole)
        self._apply_cable_lock_state(combo)
        return cable.index

    def _apply_cable_lock_state(self, combo: QComboBox) -> None:
        if self.cable_locked:
            combo.setEnabled(False)
            combo.setStyleSheet("QComboBox { color: #ffd400; font-weight: 600; }")
            self.cable_lock_button.setText("🔒 已锁定")
            self.cable_lock_button.setToolTip("当前已锁定，点击解锁手动选择")
            self.cable_note.setText("✅ 已锁定虚拟声卡 CABLE Input：所有合成的英文都从这里送进会议")
            self.cable_note.setStyleSheet(f"color: {OK_GREEN};")
        else:
            combo.setEnabled(True)
            combo.setStyleSheet("")
            self.cable_lock_button.setText("🔓 已解锁")
            self.cable_lock_button.setToolTip("当前已解锁，点击重新锁定虚拟声卡")
            self.cable_note.setText("⚠️ 已解锁手动选择（高级用途；正常情况请重新锁定）")
            self.cable_note.setStyleSheet(f"color: {WARN_ORANGE};")

    def _toggle_cable_lock(self) -> None:
        self.cable_locked = not self.cable_locked
        self._apply_cable_lock(self.cable_combo, self._outputs_cache)

    def _update_voice_hint(self) -> None:
        voice = self._current_voice_id()
        model = str(self.tts_model_combo.currentData())
        if not voice:
            self.voice_hint.setText("尚未填写音色：点右侧「一键克隆音色」创建，或从音色库选择后保存。")
            self.voice_hint.setStyleSheet(f"color: {WARN_ORANGE};")
            return
        entries = self.settings.get("voice_library", [])
        bound = None
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("voice_id") == voice:
                    bound = entry
                    break
        if bound is None:
            self.voice_hint.setText(
                f"音色 {voice} 的绑定模型未知（控制台复刻的？）。确保它与 {model} 完全匹配，否则合成可能报错。"
            )
            self.voice_hint.setStyleSheet(f"color: {TEXT_DIM};")
            return
        bound_model = str(bound.get("target_model", ""))
        if bound_model != model:
            self.voice_hint.setText(
                f"⚠️ 音色绑定的是 {bound_model}，与当前 TTS 模型 {model} 不是同一绑定模型，请选对应音色或重新复刻。"
            )
            self.voice_hint.setStyleSheet(f"color: {ERR_RED};")
        else:
            self.voice_hint.setText(f"✅ 音色与模型匹配（{bound_model}）。")
            self.voice_hint.setStyleSheet(f"color: {OK_GREEN};")

    def _browse_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择记录保存位置", self.output_dir_edit.text() or "")
        if chosen:
            self.output_dir_edit.setText(chosen)

    def _update_credential_hints(self) -> None:
        workspace = self.workspace_edit.text().strip()
        if workspace:
            self.workspace_hint.setText(f"✅ 已填写：{workspace}")
            self.workspace_hint.setStyleSheet(f"color: {OK_GREEN};")
        else:
            self.workspace_hint.setText("❌ 未填写：形如 llm-xxxxxxxx 的业务空间 ID")
            self.workspace_hint.setStyleSheet(f"color: {ERR_RED};")
        typed = self.api_key_edit.text().strip()
        saved = bool(self.settings.get_api_key())
        if typed:
            self.api_key_hint.setText("✅ 已输入新 Key，点下方「保存」后写入 Windows 凭据管理器")
            self.api_key_hint.setStyleSheet(f"color: {WARN_ORANGE};")
        elif saved:
            self.api_key_hint.setText("✅ API Key 已保存在 Windows 凭据管理器，无需重复输入")
            self.api_key_hint.setStyleSheet(f"color: {OK_GREEN};")
        else:
            self.api_key_hint.setText("❌ 尚未设置 API Key：粘贴百炼 API Key 后点「保存」")
            self.api_key_hint.setStyleSheet(f"color: {ERR_RED};")

    def _toggle_loopback_lock(self) -> None:
        if not self.loopback_locked:
            self.loopback_locked_index = self.loopback_combo.currentData()
        self.loopback_locked = not self.loopback_locked
        self._apply_loopback_lock_state()

    def _apply_instruction_preset(self, _index: int = -1) -> None:
        instruction = self.instruction_preset_combo.currentData()
        if instruction != CUSTOM_INSTRUCTION:
            self.instruction_edit.setText(str(instruction or ""))

    def _mark_instruction_custom(self, _text: str) -> None:
        index = self.instruction_preset_combo.findData(CUSTOM_INSTRUCTION)
        if index >= 0 and self.instruction_preset_combo.currentIndex() != index:
            self.instruction_preset_combo.blockSignals(True)
            self.instruction_preset_combo.setCurrentIndex(index)
            self.instruction_preset_combo.blockSignals(False)

    def _update_instruction_support(self, _index: int = -1) -> None:
        model = str(self.tts_model_combo.currentData() or "")
        supported = model in INSTRUCTION_MODELS
        self.instruction_preset_combo.setEnabled(supported)
        self.instruction_edit.setEnabled(supported)
        if supported:
            self.instruction_note.setText(
                "该模型支持自然语言语气控制。预设只改变发声风格，不修改译文；自定义内容会原样发送。"
            )
            self.instruction_note.setStyleSheet(f"color: {OK_GREEN};")
        else:
            self.instruction_note.setText(
                "当前模型不支持本工具的语气指令，已保存的内容会保留但不会发送。"
            )
            self.instruction_note.setStyleSheet(f"color: {TEXT_DIM};")

    @staticmethod
    def _apply_device_lock_state(combo: QComboBox, button: QPushButton, locked: bool, label: str) -> None:
        combo.setEnabled(not locked)
        if locked:
            combo.setStyleSheet("QComboBox { color: #ffd400; font-weight: 600; }")
            button.setText("🔒 已锁定")
            button.setToolTip(f"当前已锁定，点击解锁更换{label}")
        else:
            combo.setStyleSheet("")
            button.setText("🔓 已解锁")
            button.setToolTip(f"当前已解锁，选择{label}后点击重新锁定")

    def _toggle_mic_lock(self) -> None:
        self.mic_locked = not self.mic_locked
        self._apply_mic_lock_state()

    def _apply_mic_lock_state(self) -> None:
        self._apply_device_lock_state(
            self.mic_combo, self.mic_lock_button, self.mic_locked, "麦克风"
        )

    def _toggle_monitor_lock(self) -> None:
        self.monitor_locked = not self.monitor_locked
        self._apply_monitor_lock_state()

    def _apply_monitor_lock_state(self) -> None:
        self._apply_device_lock_state(
            self.monitor_combo, self.monitor_lock_button, self.monitor_locked, "试听设备"
        )

    def _apply_loopback_lock_state(self) -> None:
        if self.loopback_locked:
            position = self.loopback_combo.findData(self.loopback_locked_index)
            if position >= 0:
                self.loopback_combo.setCurrentIndex(position)
            elif self.active_loopback_name:
                low = self.active_loopback_name.lower()
                for index in range(self.loopback_combo.count()):
                    if low in self.loopback_combo.itemText(index).lower():
                        self.loopback_combo.setCurrentIndex(index)
                        self.loopback_locked_index = self.loopback_combo.currentData()
                        break
            self.loopback_combo.setEnabled(False)
            self.loopback_combo.setStyleSheet("QComboBox { color: #ffd400; font-weight: 600; }")
            self.loopback_lock_button.setText("🔒 已锁定")
            self.loopback_lock_button.setToolTip("当前已锁定，点击解锁手动选择")
            self.loopback_note.setText(
                f"✅ 已锁定实际监听设备：{self.loopback_combo.currentText() or '（开始答辩后自动标出）'}"
            )
            self.loopback_note.setStyleSheet(f"color: {OK_GREEN};")
        else:
            self.loopback_combo.setEnabled(True)
            self.loopback_combo.setStyleSheet("")
            self.loopback_lock_button.setText("🔓 已解锁")
            self.loopback_lock_button.setToolTip("当前已解锁，点击重新锁定")
            self.loopback_note.setText("⚠️ 已解锁手动选择；微信/Teams 的声音从哪个设备播出就选哪个")
            self.loopback_note.setStyleSheet(f"color: {WARN_ORANGE};")

    def _hotkey_combo(self, key: str, fallback: str) -> QComboBox:
        combo = NoWheelComboBox()
        for name in sorted(KEY_MAP):
            combo.addItem(name.upper(), name)
        current = str(self.settings.get(key, fallback) or fallback).lower()
        combo.setCurrentIndex(max(combo.findData(current), 0))
        return combo

    def _open_clone_voice(self) -> None:
        dialog = CloneVoiceDialog(self, self.settings, str(self.tts_model_combo.currentData()))
        dialog.exec()
        if dialog.created_voice_id:
            self._refresh_voice_combo(keep=dialog.created_voice_id)

    def _add_term(self) -> None:
        row = self.terms_table.rowCount()
        self.terms_table.insertRow(row)
        self.terms_table.setFocus()

    def _remove_terms(self) -> None:
        rows = sorted({index.row() for index in self.terms_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.terms_table.removeRow(row)

    def _import_terms(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(self, "导入术语 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "导入术语", f"读取失败：{exc}")
            return
        if not isinstance(data, list):
            QMessageBox.warning(self, "导入术语", "JSON 必须是数组")
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            row = self.terms_table.rowCount()
            self.terms_table.insertRow(row)
            self.terms_table.setItem(row, 0, QTableWidgetItem(str(item.get("source", ""))))
            self.terms_table.setItem(row, 1, QTableWidgetItem(str(item.get("target", ""))))

    def _collect_terms(self) -> list[dict[str, str]]:
        terms: list[dict[str, str]] = []
        for row in range(self.terms_table.rowCount()):
            source = self.terms_table.item(row, 0)
            target = self.terms_table.item(row, 1)
            if source is None or target is None:
                continue
            source_text = source.text().strip()
            target_text = target.text().strip()
            if source_text and target_text:
                terms.append({"source": source_text, "target": target_text})
        return terms

    def _save(self) -> None:
        from ..api_payloads import cosyvoice_instruction_units
        model = str(self.tts_model_combo.currentData())
        if model.startswith("cosyvoice") and cosyvoice_instruction_units(self.instruction_edit.text().strip()) > 100:
            QMessageBox.warning(self, "语气指令过长", "CosyVoice 限 100 字符单位，汉字计 2。请缩短后保存。")
            return
        keys = [self.direct_hotkey_combo.currentData(), self.translate_hotkey_combo.currentData(), self.cancel_hotkey_combo.currentData()]
        if len(set(keys)) != len(keys):
            QMessageBox.warning(self, "快捷键冲突", "原声、翻译、停止需要三个不同的快捷键。")
            return
        for entry in self.settings.get("voice_library", []):
            if isinstance(entry, dict) and entry.get("voice_id") == self._current_voice_id() and entry.get("target_model") and entry["target_model"] != model:
                QMessageBox.warning(self, "音色与模型不匹配", f"该音色绑定 {entry['target_model']}，请选择对应模型或为当前模型重新复刻。")
                return
        if self.api_key_edit.text().strip():
            self.settings.set_api_key(self.api_key_edit.text())
            self.api_key_edit.clear()  # 已写入凭据管理器，避免明文留在输入框
        self._update_credential_hints()
        self.settings.shared.update(
            {
                "workspace_id": self.workspace_edit.text().strip(),
                "input_device": self.mic_combo.currentData(),
                "teams_output_device": (
                    self.cable_combo.currentData()
                    if not self.cable_locked and self.cable_locked_index is None
                    else (
                        self.cable_locked_index
                        if self.cable_locked
                        else self.cable_combo.currentData()
                    )
                ),
                "loopback_device": self.loopback_combo.currentData(),
                "loopback_device_name": next(
                    (device.name for device in self._loopbacks_cache
                     if device.index == self.loopback_combo.currentData()), ""
                ),
                "proxy_mode": str(self.proxy_combo.currentData()),
                "http_proxy": self.proxy_edit.text().strip(),
            }
        )
        self.settings.update(
            {
                "tts_voice_id": self._current_voice_id(),
                "tts_model": str(self.tts_model_combo.currentData()),
                "llm_model": str(self.llm_combo.currentData()),
                "asr_model": str(self.asr_combo.currentData()),
                "vad_silence_ms": self.silence_spin.value(),
                "max_context_turns": self.turns_spin.value(),
                "monitor_enabled": self.monitor_check.isChecked(),
                "monitor_output_device": self.monitor_combo.currentData(),
                "tts_rate": self.rate_spin.value(),
                "tts_pitch": self.pitch_spin.value(),
                "tts_volume": self.volume_spin.value(),
                "tts_instruction": self.instruction_edit.text().strip(),
                "ambience_mode": str(self.ambience_combo.currentData() or "off"),
                "speech_mode": self.speech_mode_combo.currentData(),
                "output_directory": self.output_dir_edit.text().strip(),
                "direct_hotkey": str(self.direct_hotkey_combo.currentData()),
                "translate_hotkey": str(self.translate_hotkey_combo.currentData()),
                "cancel_hotkey": str(self.cancel_hotkey_combo.currentData()),
            }
        )
        self.settings.set_glossary(self._collect_terms())
        self.accept()


class SelfCheckDialog(QDialog):
    """Pre-defense checklist: config, connectivity, voice, devices."""

    row_finished = Signal(int, bool, str)
    check_finished = Signal()
    CHECKS = ("Workspace ID 已填写", "API Key 已保存", "克隆音色 voice_id",
              "VB-CABLE 虚拟声卡", "麦克风设备", "评委声音回环",
              "百炼连接（翻译模型）", "克隆音色试听（仅本机）")

    def __init__(self, parent: QWidget, settings: DefenseSettings) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("答辩前自检")
        self._cancel = threading.Event()
        self._thread = None
        self._closing = False
        self.row_finished.connect(self._finish_row, Qt.QueuedConnection)
        self.check_finished.connect(self._check_finished, Qt.QueuedConnection)
        self.resize(680, 460)
        layout = QVBoxLayout(self)
        caption = QLabel("逐项检查；答辩开始前建议全部为绿色。")
        layout.addWidget(caption)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["检查项", "结果"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.run_button = QPushButton("开始自检")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)

    def _add_row(self, name: str, ok: bool | None, message: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        if ok is None:
            status = "检查中…"
        else:
            status = ("✅ " if ok else "❌ ") + message
        item = QTableWidgetItem(status)
        if ok is True:
            item.setForeground(QColor(OK_GREEN))
        elif ok is False:
            item.setForeground(QColor(ERR_RED))
        self.table.setItem(row, 1, item)
        self.table.scrollToBottom()

    @Slot(int, bool, str)
    def _finish_row(self, row: int, ok: bool, message: str) -> None:
        if self._closing:
            return
        item = QTableWidgetItem(("✅ " if ok else "❌ ") + message)
        item.setForeground(QColor(OK_GREEN if ok else ERR_RED))
        self.table.setItem(row, 1, item)

    def _run(self) -> None:
        if self._closing or (self._thread is not None and self._thread.is_alive()):
            return
        logging.getLogger("defense.lifecycle").info("self_check_started")
        self.run_button.setEnabled(False)
        self.table.setRowCount(0)
        for name in self.CHECKS:
            self._add_row(name, None, "")
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._worker, args=(self._cancel,),
                                        name="defense-self-check", daemon=True)
        self._thread.start()

    @Slot()
    def _check_finished(self) -> None:
        if not self._closing:
            logging.getLogger("defense.lifecycle").info("self_check_finished_on_gui_thread")
            self.run_button.setEnabled(True)

    def done(self, result) -> None:
        self._closing = True
        self._cancel.set()
        logging.getLogger("defense.lifecycle").info("self_check_closed")
        super().done(result)

    def _worker(self, token: threading.Event) -> None:
        """Only plain data and queued signals here: never touch a QWidget."""
        try:
            self._perform_checks(token)
        except Exception as exc:
            if not token.is_set():
                try:
                    self.row_finished.emit(7, False, f"自检中断：{exc}"[:160])
                except RuntimeError:
                    pass  # parent window already destroyed
        finally:
            if not token.is_set():
                try:
                    self.check_finished.emit()
                except RuntimeError:
                    pass

    def _perform_checks(self, token: threading.Event) -> None:
        settings = self.settings
        def finish(row, ok, message):
            if not token.is_set():
                logging.getLogger("defense.lifecycle").info("self_check_result row=%s ok=%s", row, ok)
                try:
                    self.row_finished.emit(row, ok, message)
                except RuntimeError:
                    token.set()

        if token.is_set():
            return
        workspace = settings.workspace_id
        api_key = settings.get_api_key()
        voice = str(settings.get("tts_voice_id", "") or "").strip()
        finish(0, bool(workspace), workspace or "未填写，请在设置中填写")

        finish(1, bool(api_key), "已从 Windows 凭据管理器读取" if api_key else "未设置")

        finish(2, bool(voice), voice or "未填写（与 TTS 模型绑定）")

        if token.is_set():
            return
        inputs, outputs = _safe_list_audio_devices()
        cable_out = next((d for d in outputs if "cable input" in d.name.lower()), None)
        cable_in = next((d for d in inputs if "cable output" in d.name.lower()), None)
        finish(
            3,
            cable_out is not None and cable_in is not None,
            "已找到 CABLE Input / CABLE Output"
            if cable_out is not None and cable_in is not None
            else "未检测到 VB-CABLE，请安装驱动后重启 Windows",
        )

        selected_mic = settings.shared.get("input_device")
        mic_ok = any(d.index == selected_mic for d in inputs) if selected_mic is not None else bool(inputs)
        finish(4, mic_ok, "可选并已选择" if mic_ok else "未找到可用麦克风")

        if token.is_set():
            return
        loopbacks = [device for device in _safe_list_loopback_devices()
                     if not is_program_output_loopback(device.name)]
        selected_loop = settings.shared.get("loopback_device")
        resolved_loop = choose_committee_loopback(
            loopbacks, selected_loop,
            str(settings.shared.get("loopback_device_name", "") or ""),
        )
        loop_ok = resolved_loop is not None
        finish(
            5,
            loop_ok,
            f"已就绪：{resolved_loop.name}" if loop_ok
            else "无真实扬声器/耳机回环可选；CABLE 不能用于评委监听",
        )

        if token.is_set():
            return
        if workspace and api_key:
            try:
                from .memory import MeetingMemory

                translator = ContextTranslator(
                    api_key=api_key,
                    workspace_id=workspace,
                    model=str(settings.get("llm_model")),
                    proxy=settings.ws_proxy,
                    proxy_mode=settings.proxy_mode,
                )
                result = translator.translate(MeetingMemory(), "zh2en", "你好。", cancel_event=token)
                finish(6, True, f"模型返回：{result[:40]}")
            except Exception as exc:
                finish(6, False, str(exc)[:120])
        else:
            finish(6, False, "缺少 Workspace 或 API Key")

        if token.is_set():
            return
        if workspace and api_key and voice:
            try:
                voice_probe(settings, workspace, api_key, voice, str(settings.get("tts_model")), cancel_event=token)
                finish(7, True, "已在本机试听设备播放，请确认是你的音色")
            except Exception as exc:
                finish(7, False, str(exc)[:160])
        else:
            finish(7, False, "缺少配置，跳过")


class RehearsalDialog(QDialog):
    """Paste the Chinese opening script, translate it coherently, rehearse aloud."""

    worker_invoke = Signal(object)

    def __init__(self, parent: QWidget, settings: DefenseSettings) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("讲稿预习 · 整段连贯翻译 + 你的音色朗读")
        self.worker_invoke.connect(self._invoke, Qt.QueuedConnection)
        self._closed = False
        self._generation = 0
        self._active_row = -1
        self._last_word = -1
        self.subtitle_overlay = SubtitleOverlay(settings)
        self.subtitle_overlay.setParent(self, self.subtitle_overlay.windowFlags())
        self.resize(960, 640)
        self._cancel = threading.Event()
        self._stop_play = threading.Event()
        self._pause_play = threading.Event()
        self._sentences: list[str] = []
        self._translations: list[str] = []
        layout = QVBoxLayout(self)

        caption = QLabel("粘贴中文开场/讲稿 → 可结合 PPT 背景整段翻译 → 选择朗读中文原稿或英文译文，用克隆音色逐句练习。")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("中文讲稿"))
        self.source_edit = QPlainTextEdit()
        self.source_edit.setPlaceholderText("把准备好的中文讲稿粘贴到这里…")
        left_layout.addWidget(self.source_edit)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("连贯英文译文（逐句）"))
        self.result_list = QTableWidget(0, 2)
        self.result_list.setHorizontalHeaderLabels(["中文", "英文"])
        self.result_list.setWordWrap(True)
        self.result_list.setTextElideMode(Qt.ElideNone)
        self.result_list.setStyleSheet("QTableWidget::item:selected { background: #243552; color: #e8eaf0; }")
        self.result_list.setEditTriggers(QTableWidget.NoEditTriggers)
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.timeout.connect(self._resize_rows)
        self.result_list.horizontalHeader().sectionResized.connect(lambda *_: self._row_timer.start(0))
        self.result_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.result_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        right_layout.addWidget(self.result_list)
        split.addWidget(right)
        split.setSizes([460, 460])
        layout.addWidget(split, 1)

        self.subtitle_zh = QLabel("播放时在这里显示中文原句")
        self.subtitle_en = QLabel("英文字幕随朗读逐词高亮；黄色位置按音频进度估算")
        for label in (self.subtitle_zh, self.subtitle_en):
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)
            layout.addWidget(label)
        self.subtitle_en.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 18px;")

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        play_options = QHBoxLayout()
        play_options.addWidget(QLabel("朗读语言"))
        self.play_language_combo = NoWheelComboBox()
        self.play_language_combo.addItem("英文译文（答辩练习）", "en")
        self.play_language_combo.addItem("中文原稿（语气检查）", "zh")
        self.play_language_combo.currentIndexChanged.connect(self._on_play_language_changed)
        play_options.addWidget(self.play_language_combo)
        self.play_language_note = QLabel("英文需要先完成整段翻译")
        self.play_language_note.setStyleSheet(f"color: {TEXT_DIM};")
        play_options.addWidget(self.play_language_note)
        play_options.addStretch(1)
        layout.addLayout(play_options)

        buttons = QHBoxLayout()
        self.translate_button = QPushButton("① 整段连贯翻译")
        self.translate_button.clicked.connect(self._translate)
        buttons.addWidget(self.translate_button)
        self.play_button = QPushButton("② 从头朗读英文")
        self.play_button.clicked.connect(self._play_all)
        buttons.addWidget(self.play_button)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        buttons.addWidget(self.pause_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self._stop)
        buttons.addWidget(self.stop_button)
        self.to_meeting_check = QCheckBox("同时输出到会议（默认仅本机练习）")
        buttons.addWidget(self.to_meeting_check)
        buttons.addStretch(1)
        export_button = QPushButton("导出双语 TXT")
        export_button.clicked.connect(self._export)
        buttons.addWidget(export_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _on_play_language_changed(self, *_args) -> None:
        chinese = self.play_language_combo.currentData() == "zh"
        self.play_button.setText("② 从头朗读中文" if chinese else "② 从头朗读英文")
        self.play_language_note.setText("可直接朗读上方中文讲稿，无需先翻译" if chinese else "英文需要先完成整段翻译")

    @Slot(object)
    def _invoke(self, fn):
        if not self._closed:
            fn()

    def _post(self, generation, fn):
        try:
            self.worker_invoke.emit(lambda: fn() if generation == self._generation else None)
        except RuntimeError:
            pass

    def _resize_rows(self):
        self.result_list.resizeRowsToContents()
        for row in range(self.result_list.rowCount()):
            label = self.result_list.cellWidget(row, 1)
            if label is not None:
                doc = QTextDocument()
                doc.setDefaultFont(label.font())
                doc.setHtml(label.text())
                doc.setTextWidth(max(40, self.result_list.columnWidth(1) - 20))
                self.result_list.setRowHeight(row, max(self.result_list.rowHeight(row), int(doc.size().height()) + 18))

    def _set_rows(self, sentences, translations):
        self._sentences = sentences
        self._translations = translations
        self.result_list.setRowCount(0)
        for row, (zh, en) in enumerate(zip(sentences, translations)):
            self.result_list.insertRow(row)
            self.result_list.setItem(row, 0, QTableWidgetItem(zh))
            item = QTableWidgetItem("")
            item.setToolTip(en)
            self.result_list.setItem(row, 1, item)
            label = QLabel(html.escape(en))
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)
            label.setMargin(8)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setStyleSheet(f"color: {TEXT_MAIN}; background: transparent; font-size: 14px;")
            self.result_list.setCellWidget(row, 1, label)
        self._resize_rows()

    def _show_play_progress(self, row, progress):
        spoken_lines = getattr(self, "_play_lines", [])
        if self._stop_play.is_set() or row >= len(spoken_lines):
            return
        spoken = spoken_lines[row]
        chunks = token_chunks(spoken)
        word = current_token_index(chunks, progress)
        if row == self._active_row and word == self._last_word:
            return
        if row != self._active_row:
            old_label = self.result_list.cellWidget(self._active_row, 1) if self._active_row >= 0 else None
            if old_label is not None and getattr(self, "_play_language", "en") == "en":
                old_label.setText(html.escape(self._translations[self._active_row]))
            if row < self.result_list.rowCount():
                self.result_list.selectRow(row)
                self.result_list.scrollToItem(self.result_list.item(row, 1))
        self._active_row, self._last_word = row, word
        rendered = highlight_html(chunks, word)
        language = getattr(self, "_play_language", "en")
        chinese_lines = getattr(self, "_play_chinese_lines", [])
        chinese = chinese_lines[row] if row < len(chinese_lines) else ""
        if language == "en" and row < self.result_list.rowCount():
            label = self.result_list.cellWidget(row, 1)
            if label is not None:
                label.setText(rendered)
        self.subtitle_zh.setText(html.escape(chinese if language == "en" else "中文原稿"))
        self.subtitle_en.setText(rendered)
        self.subtitle_overlay.show_progress(chinese if language == "en" else "", spoken, progress)

    def _finish_play(self, message):
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.play_button.setEnabled(True)
        self.translate_button.setEnabled(True)
        self.play_language_combo.setEnabled(True)
        self.progress_label.setText(message)

    # ------------------------------------------------------------- translate
    def _translate(self) -> None:
        text = self.source_edit.toPlainText().strip()
        if len(text) < 10:
            QMessageBox.information(self, "讲稿预习", "请先粘贴讲稿内容（至少 10 个字）")
            return
        try:
            client = BailianClient(
                self.settings.get_api_key(),
                self.settings.workspace_id,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=int(self.settings.shared.get("request_timeout", 45)),
            )
        except ApiError as exc:
            QMessageBox.warning(self, "讲稿预习", str(exc))
            return
        self._stop()
        self._cancel = threading.Event()
        cancel = self._cancel
        generation = self._generation
        self.translate_button.setEnabled(False)
        self.progress_label.setText("正在通读全文并翻译…")
        sentences = split_source_sentences(text)
        translate_settings = {
            "translation_terms": self.settings.get("glossary", ""),
            "translation_memories": "",
            "translation_domain": "Academic thesis defense",
        }

        def worker() -> None:
            try:
                translations = client.translate_long_form(
                    text,
                    sentences,
                    translate_settings,
                    cancel_event=cancel,
                    on_progress=lambda message: self._post(generation,
                        lambda m=message: self.progress_label.setText(m)
                    ),
                )
            except Exception as exc:
                if not cancel.is_set():
                    self._post(generation, lambda m=str(exc): self.progress_label.setText(f"翻译失败：{m}"))
                self._post(generation, lambda: self.translate_button.setEnabled(True))
                return
            if cancel.is_set():
                return

            def apply_rows() -> None:
                self._set_rows(sentences, translations)
                self.progress_label.setText(f"翻译完成，共 {len(sentences)} 句，可以开始朗读练习")
                self.translate_button.setEnabled(True)

            self._post(generation, apply_rows)

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------- play
    def _play_all(self) -> None:
        if getattr(self, "_play_thread", None) is not None and self._play_thread.is_alive():
            return
        language = str(self.play_language_combo.currentData() or "en")
        if language == "zh":
            source = self.source_edit.toPlainText().strip()
            spoken_lines = split_source_sentences(source)
            if not spoken_lines:
                QMessageBox.information(self, "讲稿预习", "请先在左侧粘贴要朗读的中文讲稿")
                return
            chinese_lines = list(spoken_lines)
        elif not self._translations:
            QMessageBox.information(self, "讲稿预习", "请先完成整段翻译")
            return
        else:
            spoken_lines = list(self._translations)
            chinese_lines = list(self._sentences)
        if not str(self.settings.get("tts_voice_id", "") or "").strip():
            QMessageBox.warning(self, "讲稿预习", "请先在设置中填写克隆音色 voice_id")
            return
        self._stop_play = threading.Event()
        self._pause_play.clear()
        self._generation += 1
        self._active_row = self._last_word = -1
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.play_button.setEnabled(False)
        self.translate_button.setEnabled(False)
        self.play_language_combo.setEnabled(False)
        self._play_language = language
        self._play_lines = spoken_lines
        self._play_chinese_lines = chinese_lines
        devices = [self.settings.get("monitor_output_device")]
        if self.to_meeting_check.isChecked():
            devices.insert(0, self.settings.shared.get("teams_output_device"))
        args = (spoken_lines, devices, speech_settings(self.settings, language=language), self._stop_play, self._generation)
        self._play_thread = threading.Thread(target=self._play_worker, args=args, daemon=True)
        self._play_thread.start()

    def _play_worker(self, spoken_lines, devices, payload, stop, generation) -> None:
        from ..audio import MultiOutputPlayer
        from .speech_policy import speech_chunks
        from .rehearsal import play_pcm
        player = session = None
        message = "朗读结束"
        try:
            client = BailianClient(self.settings.get_api_key(), self.settings.workspace_id,
                                   proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode)
            player = MultiOutputPlayer(devices, 24000)
            player.__enter__()
            errors = []
            captured = bytearray()
            session = create_tts_session(client, payload, on_audio=captured.extend, on_error=errors.append)
            self._rehearsal_session = session
            session.start()
            for index, spoken_text in enumerate(spoken_lines):
                while self._pause_play.is_set() and not stop.is_set():
                    stop.wait(.02)
                if stop.is_set():
                    break
                captured.clear()
                self._post(generation, lambda p=index, total=len(spoken_lines):
                    self.progress_label.setText(f"正在准备第 {p + 1}/{total} 句音频…"))
                for chunk in speech_chunks(spoken_text):
                    if not session.speak(chunk):
                        raise ApiError("讲稿发声队列未就绪")
                deadline = time.monotonic() + 120
                while not session.wait_until_idle(0.1):
                    if stop.is_set():
                        session.interrupt()
                        break
                    if time.monotonic() > deadline:
                        raise ApiError("讲稿合成超时")
                if errors:
                    raise ApiError(errors[-1])
                if stop.is_set():
                    break
                if not captured:
                    raise ApiError("讲稿合成未返回有效音频")
                self._post(generation, lambda p=index, total=len(spoken_lines):
                    self.progress_label.setText("已暂停" if self._pause_play.is_set() else f"正在朗读第 {p + 1}/{total} 句"))
                self._post(generation, lambda p=index: self._show_play_progress(p, 0.0))
                if not play_pcm(bytes(captured), player, stop, self._pause_play,
                                lambda progress, p=index: self._post(generation,
                                    lambda: self._show_play_progress(p, progress))):
                    break
            message = "朗读已停止" if stop.is_set() else "朗读结束"
        except Exception as exc:
            message = "朗读已停止" if stop.is_set() else f"朗读失败：{exc}"
        finally:
            if session is not None:
                session.close()
            if player is not None:
                player.close()
            self._rehearsal_session = None
            self._post(generation, lambda: self._finish_play(message))

    def done(self, result):
        self._stop()
        self._closed = True
        self.subtitle_overlay.close()
        super().done(result)

    def _toggle_pause(self) -> None:
        if not self.pause_button.isEnabled():
            return
        if self._pause_play.is_set():
            self._pause_play.clear()
            self.pause_button.setText("暂停")
            self.progress_label.setText("继续朗读")
        else:
            self._pause_play.set()
            self.pause_button.setText("继续")
            self.progress_label.setText("已暂停")

    def _stop(self) -> None:
        self._stop_play.set()
        self._generation += 1
        session = getattr(self, "_rehearsal_session", None)
        if session is not None:
            session.interrupt()
        self._pause_play.clear()
        self._cancel.set()
        self.subtitle_overlay.stop()
        self._finish_play("朗读已停止")

    def _export(self) -> None:
        if not self._translations:
            QMessageBox.information(self, "讲稿预习", "还没有可导出的翻译")
            return
        path, _selected = QFileDialog.getSaveFileName(self, "导出双语讲稿", "defense_script.txt", "TXT (*.txt)")
        if not path:
            return
        lines: list[str] = []
        for zh, en in zip(self._sentences, self._translations):
            lines.append(zh)
            lines.append(en)
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.progress_label.setText(f"已导出：{path}")


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    if "--smoke-test" in sys.argv:
        from PySide6.QtGui import QFontDatabase
        for font in ("msyh.ttc", "segoeui.ttf"):
            QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / font))
        with tempfile.TemporaryDirectory(prefix="defense-smoke-") as folder:
            window = DefenseWindow(DefenseSettings(Path(folder)))
            window.show()
            app.processEvents()
            if "--smoke-output" in sys.argv:
                output = Path(sys.argv[sys.argv.index("--smoke-output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(output / "packaged-main.png"))
                (output / "smoke-result.json").write_text(json.dumps({"ui_created": True, "network_calls": False, "hotkeys_started": False}), encoding="utf-8")
            window.close()
            return 0
    window = DefenseWindow()
    window.show()
    window.start_hotkeys()
    return app.exec()
