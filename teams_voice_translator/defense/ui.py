from __future__ import annotations

import base64
from dataclasses import asdict, replace
import datetime
import html
import json
import re
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot, QEvent
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPalette, QColor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QScrollArea,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..aliyun import ApiError, BailianClient, create_realtime_asr
from ..audio import (list_audio_devices, list_loopback_devices,
                     choose_committee_loopback, is_program_output_loopback, is_virtual_endpoint,
                     useful_devices, resolve_monitor_device, resolve_program_output)
from ..hotkeys import KEY_MAP
from .hotkeys_poll import PollingHotkeys
from ..longform import split_source_sentences
from ..oss_upload import OssTemporaryUploader, OssUploadError
from ..voice_sample import SUPPORTED_AUDIO_SUFFIXES, VoiceSampleError, normalized_voice_sample, inspect_voice_sample, voice_sample_duration
from ..api_payloads import TTS_CAPABILITIES, VOICE_PREPROCESS_MODELS, is_minimax_tts_model, tts_supports
from .chat_models import USE_QA, USE_TRANSLATE, models_for, thinking_fields, translation_problem
from .clone_batch import (
    MAX_CLONE_SECONDS,
    MINIMAX_PREVIEW_TEXT,
    minimax_voice_name,
    build_tasks,
    clone_channel_for_model,
    describe_plan,
    group_tasks,
    minimum_sample_seconds,
    needs_oss,
)
from .overlay import SubtitleOverlay, token_chunks, current_token_index, highlight_html
from .pipeline import DefenseEngine, EngineCallbacks
from .ppt_context import extract_document_text, import_presentation
from .glossary_gen import ASR_HOTWORD_COUNT, generate_terms
from .dedupe import (STRONG, QaDedupeSession, build_verify_messages, find_qa_pairs, find_term_groups,
                     parse_verify, same_source_conflicts)
from .channels import (BACKUP_ROUTES, MODES as CHANNEL_MODES, PRIMARY_REGION, REGION_LABELS,
                       TTS_MODES as CHANNEL_TTS_MODES, backup_channel, backup_endpoint, channel_summary,
                       get_backup_api_key, primary_endpoint, route_label, set_backup_api_key, uses_main_key)
from .qa_bank import STRICTNESS_CHOICES, strictness_thresholds
from .usage import USAGE, balance_summary, format_tokens, load_history, query_balance, query_month_bill, save_session
from .qa import QaAdvisor, NON_FIRST_PERSON
from .question_intent import INTENT_MODEL, INTENT_MODELS, analyze_question
from .livetranslate import LIVETRANSLATE_MODELS
from .quick_phrases import (CATEGORIES as PHRASE_CATEGORIES, CATEGORY_LABELS as PHRASE_CATEGORY_LABELS,
                            QuickPhrase, QuickPhraseStore, default_phrases, next_phrase_id)
from .quick_phrases import (PhraseRecordings, RECORD_RATE, build_suggest_messages, clean_recording,
                            parse_suggestions)
from . import scene_modes
from . import deps
from .thesis_context import ThesisStore, excerpt_pairs, extract_thesis_pages, import_thesis, merge_terms
from .glossary_audit import AUDIT_MODEL, audit_terms
from .qa_review import REVIEW_MODEL, review_answers, export_bank, template_bank
from .qa_review_ui import AnswerReviewDialog
from .thesis_study import (DEFAULT_QA_COUNT, DEFAULT_STUDY_MODEL, FILL_ZH_BATCH, MAX_QA_COUNT, QA_BATCH,
                           StudyNotes, StudySection, StudyStore, apply_chinese, build_fill_zh_payload,
                           build_import_payload, build_qa_payload, build_study_payload,
                           calibrate, needs_chinese, next_id_start, parse_fill_zh_response,
                           parse_import_response, parse_qa_response, parse_study_response,
                           run_payload, split_import_text)
from .qa_bank import (BASIS_LABELS, QaBankError, QaBankStore, QaEntry, QaIndex,
                      match_question, merge_entries, parse_bank_file, parse_json_bank,
                      parse_text_bank, reference_pairs)
from .script_desk import (LANGUAGE_LABELS, ScriptDocument, cumulative_bounds,
                          line_at_elapsed)
from .settings import DefenseSettings
from .translator import ContextTranslator
from .tts_session import create_tts_session
from .speech_policy import (speech_settings, natural_delivery_instruction, INSTRUCTION_MODELS,
                            INSTRUCTION_PRESETS, SPEECH_SAMPLE_RATE, pause_comparison_text)
from .humanize import LEVEL_CHOICES as HUMANIZE_CHOICES
from .memory import STYLE_DESK_HINTS, TRANSLATION_STYLES
from .ambience import (AMBIENCE_PRESETS, GROUP_ORDER as AMBIENCE_GROUPS, MAX_LEVEL_DB as AMBIENCE_MAX_DB,
                       RoomTone, VOLUME_DEFAULT as AMBIENCE_VOLUME_DEFAULT, at_cap as ambience_at_cap,
                       effective_level_db as ambience_level_db)


CUSTOM_INSTRUCTION = "__custom__"
from .ambience import BACKGROUND_CHOICES, background_mode, make_room_tone


def _safe_list_audio_devices() -> tuple[list, list]:
    try:
        return list_audio_devices()
    except Exception as exc:
        print(f"音频设备枚举失败：{exc}")
        return [], []



def apply_default_loopback(settings) -> bool:
    """一次性修正评委监听设备：没选过、或选的是远程桌面/虚拟声卡（如 ToDesk）时，
    默认改成 8 号设备（用户指定：这台电脑上真正出声的那个）。之后仍可在设置里手动改。"""
    if bool(settings.get("loopback_default_v2", False)):
        return False
    devices = [d for d in _safe_list_loopback_devices() if not is_program_output_loopback(d.name)]
    if not devices:
        return False
    saved_index = settings.shared.get("loopback_device")
    saved_name = str(settings.shared.get("loopback_device_name", "") or "")
    saved = None
    if saved_index is not None or saved_name:
        saved = choose_committee_loopback(devices, saved_index, saved_name)
    target = next((d for d in devices if d.index == 8), None)
    changed = False
    if target is not None and (saved is None or is_virtual_endpoint(saved.name)):
        settings.shared.update({"loopback_device": 8, "loopback_device_name": target.name})
        changed = True
    settings.set("loopback_default_v2", True)
    return changed


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

# clone_channel_for_model 现在住在 clone_batch 里（批量编排也要用它，放这里会循环导入）。
# 这里保留同名引用，既有调用点和测试不用改。


CHANNEL_LABELS = {
    "qwen3": "Qwen3 直传（qwen3-tts-vc-* · Base64 上传，无需 OSS，秒级完成）",
    "legacy": "OSS 中转（cosyvoice-* / qwen-audio-tts-* 需等待部署 1~3 分钟；MiniMax 同步完成）",
}


def _has_tts_source(settings: DefenseSettings) -> bool:
    if str(settings.get("tts_model", "")) == "qwen-audio-3.1-tts-next":
        return Path(str(settings.get("tts_reference_audio", "") or "")).is_file()
    return bool(str(settings.get("tts_voice_id", "") or "").strip())


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
    if tts_model == "qwen-audio-3.1-tts-next":
        timeout = max(timeout, 180.0)
    monitor = resolve_monitor_device(settings)
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
    "direct": ("原声直通", OK_GREEN),
}


class NoWheelComboBox(QComboBox):
    """悬停时滚轮不改变选中值，防止滚动设置页时误改关键配置；点开列表后仍可正常滚动。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumContentsLength(12)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

    def wheelEvent(self, event) -> None:
        event.ignore()


class ModelPickCombo(NoWheelComboBox):
    """可以下拉选、也可以直接输入的模型框。text() / setText() 和 QLineEdit 一样，拿到的是模型 ID。"""

    def __init__(self, options, current: str = "", *, same_label: str = "和主通道相同", parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.addItem(same_label, "")
        for model, blurb in options:
            self.addItem(f"{model} · {blurb}", model)
        self.setText(current)

    def text(self) -> str:
        shown = self.currentText().strip()
        index = self.findText(shown)
        if index >= 0:
            return str(self.itemData(index) or "")
        return shown.split(" · ", 1)[0].strip()

    def setText(self, value: str) -> None:  # noqa: N802
        value = str(value or "").strip()
        index = self.findData(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setEditText(value)


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


class NoWheelSlider(QSlider):
    """滑块同样忽略滚轮：滚动设置页时不会误改背景音量。"""

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
    committee_recognized = Signal(int, str)
    tts_sentence_started = Signal(str, str)
    tts_sentence_duration = Signal(str, float)
    tts_sentence_progress = Signal(str, float)
    input_pressed = Signal(str, str)
    input_released = Signal(str, str)
    cancel_requested = Signal()
    committee_device = Signal(str)
    latency_ready = Signal(int, int)
    import_finished = Signal(str, bool)   # 提示文字, 是否导入了论文全文
    pdf_support_ready = Signal(str, bool, str)   # 待导入的论文路径, 是否装好, 安装输出
    speech_control = Signal(str)                 # pause / resume / stop
    input_stopped = Signal()


def _format_clock(value: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(value))


class DefenseWindow(QMainWindow):
    def __init__(self, settings: DefenseSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or DefenseSettings()
        try:
            apply_default_loopback(self.settings)
        except Exception:
            pass
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
                on_committee_recognized=self.signals.committee_recognized.emit,
                on_tts_sentence_started=lambda text, zh: self.signals.tts_sentence_started.emit(text, zh),
                on_tts_sentence_duration=lambda text, seconds: self.signals.tts_sentence_duration.emit(text, seconds),
                on_tts_sentence_progress=self.signals.tts_sentence_progress.emit,
                on_committee_device=lambda name: self.signals.committee_device.emit(name),
                on_latency=self.signals.latency_ready.emit,
                on_speech_control=lambda action: self.signals.speech_control.emit(action),
                on_input_stopped=self.signals.input_stopped.emit,
            ),
        )
        self.hotkeys: PollingHotkeys | None = None
        self._timeline_rows: dict[int, int] = {}
        self._committee_transcripts: dict[int, list[str]] = {}
        self._recent_spoken: list[str] = []
        self.recent_buttons: list[QPushButton] = []
        self._pending_audio_prompt = False
        self._input_mode = None
        self._held_inputs = {}
        self.overlay: SubtitleOverlay | None = None
        self._build_ui()
        self._connect_signals()
        self.engine.start()
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(1000)
        self.recording_timer.timeout.connect(self._refresh_recording_status)
        self.recording_timer.start()
        self._wire_quick_features()
        self._install_error_hooks()
        # 启动几秒后再开始后台预渲染：先让界面和引擎起来
        QTimer.singleShot(4000, self._auto_prerender)

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        from .. import __version__

        self.setWindowTitle(f"留子答辩器 · 实时中英互译 v{__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1380, 860)
        self.setMinimumSize(1060, 680)
        # 大屏幕：默认铺到可用区域的九成，助答栏跟着变宽，中英对照看得清
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            width = max(1380, int(area.width() * 0.9))
            height = max(860, int(area.height() * 0.9))
            if width <= area.width() and height <= area.height():
                self.resize(width, height)
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
        # 助答栏占近一半宽：左右两栏中英对照需要空间。splitter 可拖动调整。
        captions.setSizes([300, 270, 520])

        # 讲稿朗读台放在主界面标签页里，不单开窗口：开场讲 PPT 时要和实时字幕
        # 并排随时切换，弹窗会挡住后面的内容、也容易被误关。
        self.stage_tabs = QTabWidget()
        self.stage_tabs.addTab(captions, "实时答辩")
        self.script_panel = ScriptDeskPanel(self.settings, self.engine)
        self.script_panel.on_playback_changed = self._on_script_playback_changed
        self.stage_tabs.addTab(self.script_panel, "讲稿朗读台")

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(self.stage_tabs)
        self.timeline_panel = self._build_timeline()
        # 底部左边对话记录、右边快捷用语：答不上来时一眼能点到
        self.phrase_panel = QuickPhrasePanel(self.settings, self.engine)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.timeline_panel)
        bottom.addWidget(self.phrase_panel)
        bottom.setSizes([520, 700])
        main_split.addWidget(bottom)
        # 答辩现场最常看的是上面三栏（尤其助答的英文回答框），底部可以拖大
        main_split.setSizes([540, 210])
        main_split.setStretchFactor(0, 4)
        main_split.setStretchFactor(1, 1)
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

        title = QLabel("留子答辩器  /  LIVE")
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
        row.addWidget(self._build_scene_bar())
        row.addSpacing(12)

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

        self.direct_button = TalkButton(f"原声 ({self._hotkey_label('direct_hotkey', 'F8')})")
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

        self.live_monitor_button = QPushButton()
        self.live_monitor_button.setCheckable(True)
        self.live_monitor_button.setStyleSheet(self.ACTIVE_BUTTON_STYLE)
        self.live_monitor_button.clicked.connect(self._toggle_live_monitor)
        row.addWidget(self.live_monitor_button)
        self._sync_live_monitor()

        row.addStretch(1)
        # 阿里云余额 / 本次用量 / 消费记录（顶栏中间那块空位）
        self.usage_bar = UsageBar(self.settings)
        row.addWidget(self.usage_bar, 2)
        self.check_button = QPushButton("自检")
        self.check_button.clicked.connect(self._open_self_check)
        row.addWidget(self.check_button)

        self.ppt_button = QPushButton("导入论文 / PPT")
        self.ppt_button.setToolTip(
            "论文（PDF / Word / txt）：提炼摘要和术语（自动加入识别和翻译术语库），并建立原文索引供助答引用；\n"
            "PPT：提炼摘要和术语。"
        )
        self.ppt_button.clicked.connect(self._on_import_ppt)
        row.addWidget(self.ppt_button)

        self.glossary_button = QPushButton("术语表")
        self.glossary_button.setToolTip("查看 / 修改术语表，或从论文和 PPT 自动生成全部术语（翻译和识别都按这里的写法）")
        self.glossary_button.clicked.connect(self._open_glossary)
        row.addWidget(self.glossary_button)

        self.study_button = QPushButton("论文预习")
        self.study_button.setToolTip("答辩前用强模型通读论文，生成可人工校对的中英「论文理解」和预案问答")
        self.study_button.clicked.connect(self._open_study)
        row.addWidget(self.study_button)

        # 「讲稿预习」已并入「讲稿朗读台」页签（中文稿翻英文、朗读、暂停都在那里），顶栏不再单列

        self.voice_compare_button = QPushButton("声音对比")
        self.voice_compare_button.clicked.connect(self._open_voice_compare)
        row.addWidget(self.voice_compare_button)
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self._open_settings)
        row.addWidget(self.settings_button)
        background_row = QHBoxLayout()
        self.background_toggle = QPushButton("背景声")
        self.background_toggle.setCheckable(True)
        self.background_toggle.clicked.connect(self._toggle_background)
        background_row.addWidget(self.background_toggle)
        self.background_combo = NoWheelComboBox()
        for key, label in BACKGROUND_CHOICES:
            self.background_combo.addItem(label, key)
        self.background_combo.setMinimumWidth(250)
        self.background_combo.currentIndexChanged.connect(self._on_background_changed)
        background_row.addWidget(self.background_combo)
        background_row.addWidget(QLabel("背景音量"))
        self.background_volume = NoWheelSlider(Qt.Horizontal)
        self.background_volume.setRange(0,100)
        self.background_volume.setFixedWidth(140)
        self._background_volume_timer = QTimer(self)
        self._background_volume_timer.setSingleShot(True)
        self._background_volume_timer.timeout.connect(self._on_background_changed)
        self.background_volume.valueChanged.connect(lambda _: self._background_volume_timer.start(200))
        background_row.addWidget(self.background_volume)
        self.background_level_label = QLabel()
        background_row.addWidget(self.background_level_label)
        background_row.addWidget(QLabel("实录背景 · 开始答辩后输出"))
        background_row.addStretch(1)
        stack.addLayout(background_row)
        self._last_background_mode = "room"
        self._sync_background_controls()
        return bar

    def _sync_live_monitor(self):
        available = bool(self.settings.get("monitor_enabled", True))
        enabled = available and bool(self.settings.get("live_monitor_enabled", True))
        self.live_monitor_button.setChecked(enabled)
        self.live_monitor_button.setEnabled(available)
        self.live_monitor_button.setText("F8/F9 监听：开" if enabled else "F8/F9 监听：关")
        self.live_monitor_button.setToolTip(
            "仅控制原声和麦克风翻译的本机监听，会议输出照常。朗读台、打字说话、问答朗读和试听不受影响。"
            if available else "请先在设置 → 声卡中启用「本机试听」并选择试听耳机。"
        )

    def _toggle_live_monitor(self, checked):
        try:
            self.settings.set("live_monitor_enabled", bool(checked))
        except Exception as exc:
            self.signals.error_message.emit(f"F8/F9 监听切换失败：{exc}")
        self._sync_live_monitor()

    def _sync_background_controls(self):
        mode = background_mode(self.settings.get("ambience_mode", "off"))
        for widget in (self.background_toggle,self.background_combo,self.background_volume):
            widget.blockSignals(True)
        self.background_toggle.setChecked(mode != "off")
        self.background_toggle.setText("背景声：开" if mode != "off" else "背景声：关")
        self.background_combo.setCurrentIndex(max(0,self.background_combo.findData(mode)))
        self.background_volume.setValue(int(self.settings.get("ambience_volume",50)))
        self.background_level_label.setText(f"{self.background_volume.value()}%")
        if mode != "off":
            self._last_background_mode = mode
        for widget in (self.background_toggle,self.background_combo,self.background_volume):
            widget.blockSignals(False)

    def _toggle_background(self, checked):
        self._set_background(self._last_background_mode if checked else "off")

    def _on_background_changed(self, *_args):
        self._set_background(str(self.background_combo.currentData() or "off"))

    def _set_background(self, mode):
        self._background_volume_timer.stop()
        try:
            self.settings.update({"ambience_mode":mode,"ambience_volume":self.background_volume.value()})
            self.engine.refresh_ambience()
        except Exception as exc:
            self.signals.error_message.emit(f"背景声切换失败：{exc}")
        self._sync_background_controls()

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
        self.timeline_pause_button = QPushButton("⏸ 暂停朗读")
        self.timeline_pause_button.setToolTip("暂停正在读的语音（重播、助答、快捷用语都算）；再点一次接着读")
        self.timeline_pause_button.clicked.connect(self._toggle_pause)
        header_row.addWidget(self.timeline_pause_button)
        header_row.addStretch(1)
        self.memory_label = QLabel("")
        self.memory_label.setStyleSheet(f"color: {TEXT_DIM};")
        header_row.addWidget(self.memory_label)
        self.auto_record_check = QCheckBox("自动录音")
        self.auto_record_check.setChecked(bool(self.settings.get("auto_record", True)))
        self.auto_record_check.setToolTip("开始答辩时自动录音；答辩中勾选或取消立即生效。老师声音与麦克风分别保存")
        self.auto_record_check.toggled.connect(self.engine.set_auto_record)
        header_row.addWidget(self.auto_record_check)
        save_archive_button = QPushButton("保存本场记录")
        save_archive_button.clicked.connect(self._on_save_archive)
        header_row.addWidget(save_archive_button)
        save_audio_button = QPushButton("保存英文语音")
        save_audio_button.setToolTip("保存软件合成的英文语音；老师原声请在记录文件夹的录音目录查找")
        save_audio_button.clicked.connect(self.engine.export_meeting_audio)
        header_row.addWidget(save_audio_button)
        open_folder_button = QPushButton("打开记录文件夹")
        open_folder_button.clicked.connect(self._on_open_archive_folder)
        header_row.addWidget(open_folder_button)
        clear_button = QPushButton("清空记忆")
        clear_button.clicked.connect(self._on_clear_memory)
        header_row.addWidget(clear_button)
        layout.addLayout(header_row)

        self.recording_status_label = QLabel("录音未开启")
        self.recording_status_label.setWordWrap(True)
        self.recording_status_label.setToolTip("显示实际写入录音文件的时长；等待声音时请确认会议声音从所选评委设备播放")
        layout.addWidget(self.recording_status_label)

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

    def _refresh_recording_status(self) -> None:
        state, message = self.engine.recording_status()
        color = {"recording": "#2fd19b", "error": "#ff6262",
                 "waiting": "#f5a623"}.get(state, TEXT_DIM)
        self.recording_status_label.setText(message)
        self.recording_status_label.setStyleSheet(f"color: {color};")

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
        if hasattr(self, "phrase_panel"):
            self.phrase_panel.on_voice_started(english)

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

        self.input_pause_button = QPushButton("⏸ 暂停")
        self.input_pause_button.setToolTip("暂停正在读的语音；再点一次从停下的地方接着读")
        self.input_pause_button.clicked.connect(self._toggle_pause)
        row.addWidget(self.input_pause_button)

        stop_button = QPushButton("停止语音 (Esc)")
        stop_button.clicked.connect(self.signals.cancel_requested.emit)
        row.addWidget(stop_button)
        return bar

    def _toggle_pause(self) -> None:
        self.qa_panel.toggle_pause()

    def _on_pause_changed(self, paused: bool) -> None:
        if hasattr(self, "timeline_pause_button"):
            self.timeline_pause_button.setText("▶ 继续朗读" if paused else "⏸ 暂停朗读")
        if hasattr(self, "input_pause_button"):
            self.input_pause_button.setText("▶ 继续" if paused else "⏸ 暂停")

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
        self.signals.input_stopped.connect(self._on_input_stopped)
        self.signals.state_changed.connect(self._on_state_changed)
        self.signals.status_message.connect(self.statusBar().showMessage)
        self.signals.error_message.connect(self._show_error)
        self.signals.mic_preview.connect(self.mic_preview.setText)
        self.signals.committee_preview.connect(self._on_committee_preview)
        self.signals.my_segment.connect(self._on_my_segment)
        self.signals.my_delta.connect(self._on_my_delta)
        self.signals.my_done.connect(self._on_my_done)
        self.signals.committee_done.connect(self._on_committee_done)
        self.signals.committee_recognized.connect(self._on_committee_recognized)
        self.signals.tts_sentence_started.connect(self._on_tts_sentence_started)
        self.signals.tts_sentence_duration.connect(self._on_tts_sentence_duration)
        self.signals.tts_sentence_progress.connect(self._on_tts_sentence_progress)
        self.signals.cancel_requested.connect(self._on_cancel)
        self.signals.committee_device.connect(self._on_committee_device)
        self.signals.latency_ready.connect(self._on_latency)
        self.signals.import_finished.connect(self._on_import_finished)
        self.signals.pdf_support_ready.connect(self._on_pdf_support_ready)
        self.signals.speech_control.connect(self._on_speech_control)

    def _on_latency(self, seg_id, milliseconds):
        self.latency_label.setText(f"识别结束 → 发声  {milliseconds / 1000:.1f} s")
        self.latency_label.setStyleSheet(f"color: {TEXT_DIM};")
        row = self._timeline_rows.get(seg_id)
        if row is not None:
            self.timeline.setItem(row, 4, QTableWidgetItem(f"{milliseconds / 1000:.1f}s"))

    def _on_tts_sentence_duration(self, english: str, seconds: float) -> None:
        if self.overlay is not None:
            self.overlay.set_duration(english, seconds)

    def _on_tts_sentence_progress(self, english: str, progress: float) -> None:
        if self.overlay is not None and self.overlay_check.isChecked():
            self.overlay.update_playback_progress(english, progress)

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
            direct_key=str(self.settings.get("direct_hotkey", "f8")),
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
        self.direct_button.setText(f"原声 ({self._hotkey_label('direct_hotkey', 'F8')})")
        self.translate_button.setText(f"翻译 ({self._hotkey_label('translate_hotkey', 'F9')})")

    def _on_cancel(self) -> None:
        if self.overlay is not None:
            self.overlay.stop()
        self.engine.interrupt_speech()
        if hasattr(self, "qa_panel"):
            self.qa_panel._sync_pause_button(False)   # Esc 也会解除暂停

    def _on_continuous_toggled(self, checked: bool) -> None:
        self.settings.set("continuous_enabled", bool(checked))
        self._held_inputs.clear()
        if not checked:
            self._set_input_mode(None)

    @Slot(str, str)
    def _script_active(self) -> bool:
        """讲稿台正在播（含暂停中）。"""
        panel = getattr(self, "script_panel", None)
        return bool(panel is not None and panel._playing)

    def _hold_mode(self) -> bool:
        return not self.continuous_check.isChecked()

    def _input_press(self, mode, source):
        if self._script_active() and self.script_panel.is_playing():
            # 讲稿正在读，老师打断要你说话：先把讲稿停在原处，免得两个声音叠在一起
            self.script_panel.pause()
            self.statusBar().showMessage("讲稿已暂停，F8/F9 按当前持续监听设置操作；点讲稿台「继续」接着讲", 8000)
        if not self._hold_mode():
            self._set_input_mode(None if self._input_mode == mode else mode)
            return
        self._held_inputs[(source, mode)] = True
        self._set_input_mode(mode)

    @Slot(str, str)
    def _input_release(self, mode, source):
        self._held_inputs.pop((source, mode), None)
        if self._hold_mode() and self._input_mode == mode:
            if not any(key[1] == mode for key in self._held_inputs):
                self._set_input_mode(None)

    def _on_script_playback_changed(self, playing: bool) -> None:
        """播放优先，但不改变用户选定的按住/持续操作方式。"""
        if playing:
            self._on_input_stopped()
            self.statusBar().showMessage("讲稿播放中，F8/F9 已关闭；需要说话可重新开启，讲稿会暂停", 8000)

    def _on_input_stopped(self) -> None:
        self._input_mode = None
        self._held_inputs.clear()
        self._refresh_input_buttons()

    def _on_speech_control(self, action: str) -> None:
        """引擎的暂停/继续/停止：字幕悬浮窗、讲稿台、助答面板一起跟上。"""
        if self.overlay is not None:
            if action == "pause":
                self.overlay.pause()
            elif action == "resume":
                self.overlay.resume()
            elif action == "stop":
                self.overlay.stop()
        if hasattr(self, "script_panel"):
            self.script_panel.on_external_speech_control(action)
        if hasattr(self, "qa_panel"):
            self.qa_panel.on_external_speech_control(action)

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
            self.qa_panel.clear_collected_question()
        self._set_pill(state)
        self.start_button.setChecked(state != "idle")
        self.start_button.setText("结束答辩" if state != "idle" else "开始答辩")
        direct = state == "direct" and self.engine._direct_requested is not False
        for widget, checked in (
            (self.translate_button, state == "listening" and self.engine._listen_requested is not False),
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

    def _on_committee_recognized(self, seg_id: int, english: str) -> None:
        self.committee_preview.setText(english)
        self.live_committee_zh.setText("正在翻译评委问题…")
        self._update_committee_history(seg_id, english, "正在翻译…")
        self._ensure_timeline_row(seg_id, "评委", english, 0)
        self.qa_panel.collect_committee_segment(seg_id, english)

    def _on_committee_preview(self, text: str) -> None:
        self.committee_preview.setText(text)
        self.qa_panel.note_committee_activity(text)

    def _on_committee_done(self, seg_id: int, english: str, chinese: str) -> None:
        self.live_committee_zh.setText(chinese)
        self._update_committee_history(seg_id, english, chinese)
        self._ensure_timeline_row(seg_id, "评委", english, 0, translation=chinese)
        if hasattr(self, "qa_panel"):
            self.qa_panel.update_committee_translation(seg_id, chinese)
        if hasattr(self, "phrase_panel"):
            try:
                self.phrase_panel.suggest_for(english or chinese)
            except Exception:  # noqa: BLE001 —— 推荐是锦上添花，出错不能影响字幕和助答
                pass

    def _update_committee_history(self, seg_id: int, english: str, chinese: str) -> None:
        self._committee_transcripts[seg_id] = [english, chinese]
        while len(self._committee_transcripts) > 200:
            del self._committee_transcripts[next(iter(self._committee_transcripts))]
        bar = self.committee_history.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        previous = bar.value()
        self.committee_history.setHtml("".join(
            f"<p style='margin:2px 0'><span style='color:{TEXT_DIM};font-size:12px'>{html.escape(en)}</span><br>"
            f"<span style='color:{OK_GREEN};font-size:14px'>{html.escape(zh)}</span></p>"
            for en, zh in self._committee_transcripts.values()))
        bar.setValue(bar.maximum() if follow else previous)

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
            self._committee_transcripts.clear()
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
            self._sync_live_monitor()
            self._sync_background_controls()
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
            # 设置页里可能改了模型或音色：情景模式显示、助答下拉、预渲染状态都跟着更新
            self._refresh_scene_bar()
            self.qa_panel.sync_from_settings()
            self.phrase_panel.refresh_ready()
            self.script_panel.reload_voice_options()
            self.script_panel.refresh_render_state()
            self._auto_prerender()

    def _open_self_check(self) -> None:
        SelfCheckDialog(self, self.settings).exec()

    def _open_voice_compare(self) -> None:
        dialog = VoiceCompareDialog(self, self.settings)
        dialog.voice_applied.connect(self._on_comparison_voice_applied)
        dialog.exec()

    def _on_comparison_voice_applied(self) -> None:
        self.engine._claim_playback()
        self.engine.interrupt_speech()
        self._refresh_scene_bar()
        self.qa_panel.sync_from_settings()
        self.phrase_panel.refresh_ready()
        self.script_panel.reload_voice_options()
        combo = self.script_panel.instruction_mode_combo
        combo.setCurrentIndex(max(0, combo.findData("none")))
        self.script_panel.refresh_render_state()
        self.statusBar().showMessage("已统一使用 A 的原始音色；已有音频保留，变音段落可点「原声重做本段」", 12000)

    def _open_rehearsal(self) -> None:
        RehearsalDialog(self, self.settings).exec()

    def _on_import_ppt(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "选择论文和 / 或答辩 PPT（可以多选，按住 Ctrl 点选）", "",
            "论文或 PPT (*.pdf *.docx *.pptx *.txt *.md);;论文 (*.pdf *.docx *.txt *.md);;演示文稿 (*.pptx)",
        )
        self.import_files(paths)

    def import_files(self, paths: list[str]) -> None:
        """一次导入多个文件：PPT 各自提炼术语；论文类文件合成一份全文建索引。"""
        paths = [p for p in (paths or []) if p]
        if not paths:
            return
        decks = [p for p in paths if Path(p).suffix.lower() == ".pptx"]
        papers = [p for p in paths if Path(p).suffix.lower() != ".pptx"]
        if len(paths) == 1:
            if decks:
                self._import_presentation(decks[0])
            else:
                self._import_thesis(papers[0])
            return
        pdf = next((p for p in papers if Path(p).suffix.lower() == ".pdf"), None)
        if pdf is not None and not self._ensure_pdf_support(pdf):
            return
        model = str(self.settings.get("llm_model", "") or "qwen3.8-flash")
        try:
            client = BailianClient(
                self.settings.get_api_key(), self.settings.workspace_id,
                proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode,
                timeout=int(self.settings.shared.get("request_timeout", 45)),
            )
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(), workspace_id=self.settings.workspace_id, model=model,
                proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode, timeout=90,
            )
        except ApiError as exc:
            QMessageBox.warning(self, "导入", str(exc))
            return
        self.ppt_button.setEnabled(False)
        self.statusBar().showMessage(f"正在导入 {len(paths)} 个文件…")

        def worker() -> None:
            lines: list[str] = []
            thesis_done = False
            for index, deck in enumerate(decks, start=1):
                self.signals.status_message.emit(f"正在读取 PPT（{index}/{len(decks)}）：{Path(deck).name}…")
                try:
                    context = import_presentation(client, Path(deck), model=model)
                except Exception as exc:  # noqa: BLE001 —— 一个文件坏了不影响其他文件
                    lines.append(f"✗ {Path(deck).name}：{exc}")
                    continue
                if not papers:
                    self.settings.set("defense_brief", context.brief)   # 有论文时以论文摘要为准
                merged, added = merge_terms(self.settings.glossary_terms(), context.terms)
                self.settings.set_glossary(merged)
                lines.append(f"✓ PPT {Path(deck).name}：{context.slide_count} 页，新增术语 {added} 条")
            if papers:
                try:
                    context = import_thesis(
                        None, list(papers), model=model,
                        on_status=self.signals.status_message.emit,
                        post=lambda payload: translator.complete(payload, deadline_seconds=420),
                        on_file_error=lambda failed, exc: lines.append(f"✗ 论文 {failed.name}：{exc}"),
                    )
                    ThesisStore(self.settings.base_dir).save(
                        source_path=context.source_path, chunks=context.chunks, page_count=context.page_count,
                        full_text=context.full_text, brief=context.brief,
                    )
                    self.settings.set("defense_brief", context.brief)
                    merged, added = merge_terms(self.settings.glossary_terms(), context.terms)
                    self.settings.set_glossary(merged)
                    self.settings.set("context_source_path", context.source_path)
                    names = "、".join(Path(p).name for p in context.source_path.split("; "))
                    lines.append(f"✓ 论文 {names}：约 {context.char_count // 1000} 千字，"
                                 f"{len(context.chunks)} 段原文索引，新增术语 {added} 条")
                    thesis_done = True
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"✗ 论文：{exc}")
            try:
                self.engine.reload_context()
            except Exception as exc:  # noqa: BLE001
                lines.append(f"重新加载术语失败：{exc}")
            total = len(self.settings.glossary_terms())
            self.signals.import_finished.emit(
                "导入完成：\n" + "\n".join(lines) + f"\n\n术语库现有 {total} 条，已自动用于语音识别和翻译。",
                thesis_done or bool(lines),
            )

        self.import_worker = threading.Thread(target=worker, name="multi-import", daemon=True)
        self.import_worker.start()

    def _on_import_finished(self, message: str, thesis: bool) -> None:
        dialog = getattr(self, "_glossary_dialog", None)
        if dialog is not None:
            dialog.reload()   # 导入刚往术语表里加了词；术语表窗口有没保存的修改时不会被覆盖
        self.ppt_button.setEnabled(True)
        if not message:
            return
        self.statusBar().showMessage(message, 15000)
        if thesis:
            self.qa_panel.reload_thesis()
            QMessageBox.information(self, "论文导入完成", message)

    def _import_presentation(self, path: str) -> None:
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
                self.signals.import_finished.emit("", False)
                return
            self.settings.set("defense_brief", context.brief)
            merged, _added = merge_terms(self.settings.glossary_terms(), context.terms)
            self.settings.set_glossary(merged)
            self.settings.set("context_source_path", context.source_path)
            self.engine.reload_context()
            self.signals.import_finished.emit(
                f"PPT 导入完成：{context.slide_count} 页，术语共 {len(self.settings.glossary_terms())} 条，"
                "可在设置中查看论文摘要与术语表", False,
            )

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------ PDF 组件自动安装
    def _ensure_pdf_support(self, path: str) -> bool:
        """读 PDF 前确认有可用的 pypdf。可以直接导入返回 True；否则询问/安装，
        装好后自动接着导入，这里返回 False。"""
        status = deps.pypdf_status()
        if status.ok:
            return True
        old = status.state == "old"
        what = (f"当前的 pypdf {status.version} 版本太旧，读这份论文会把英文单词粘在一起"
                if old else "读取 PDF 需要一个叫 pypdf 的小组件（约 0.3 MB），当前还没有安装")
        if not deps.can_auto_install():
            # 打包好的 EXE 不能在线装：打包时本应已包含，缺了说明是旧版 EXE
            if old:
                return True   # 旧版也能读，只是效果差；导入后会有提示
            QMessageBox.warning(
                self, "导入论文",
                f"{what}。\n\n这个程序版本没有带上它，请改为导入论文的 Word 版（.docx，不需要任何组件），"
                "或者换用最新打包的程序。",
            )
            return False
        buttons = QMessageBox.Yes | QMessageBox.No
        if old:
            buttons |= QMessageBox.Ignore
        tip = "\n\n选「忽略」则用旧版继续导入。" if old else "\n\n也可以选「否」，改为导入论文的 Word 版（.docx）。"
        answer = QMessageBox.question(
            self, "安装 PDF 组件",
            f"{what}。\n\n是否现在自动在线安装？只装到本程序自己的运行环境里，"
            f"通常十几秒，需要联网（先用清华镜像，失败再用官方源）。{tip}",
            buttons, QMessageBox.Yes,
        )
        if answer == QMessageBox.Ignore:
            return True
        if answer != QMessageBox.Yes:
            return False
        self._install_pdf_support(path)
        return False

    def _install_pdf_support(self, path: str) -> None:
        self.ppt_button.setEnabled(False)
        self.statusBar().showMessage("正在在线安装 PDF 组件 pypdf…")
        proxy = self.settings.proxy if self.settings.proxy_mode == "manual" else ""

        def worker() -> None:
            try:
                ok, output = deps.install_package(
                    deps.PYPDF_SPEC, proxy=proxy, on_status=self.signals.status_message.emit,
                )
            except Exception as exc:   # 兜底：安装线程绝不能把异常丢给 Qt
                ok, output = False, str(exc)
            self.signals.pdf_support_ready.emit(path, ok, output)

        self.pdf_install_worker = threading.Thread(target=worker, name="pdf-support-install", daemon=True)
        self.pdf_install_worker.start()

    def _on_pdf_support_ready(self, path: str, ok: bool, output: str) -> None:
        self.ppt_button.setEnabled(True)
        if not ok:
            detail = output.strip().splitlines()[-1] if output.strip() else "没有返回信息"
            self.statusBar().showMessage("PDF 组件安装失败", 15000)
            QMessageBox.warning(
                self, "安装 PDF 组件",
                "在线安装没有成功，请检查网络（设置里的代理）后再导入一次，"
                "或者改为导入论文的 Word 版（.docx）。\n\n"
                f"最后的提示：{detail[:300]}",
            )
            return
        if deps.load_pypdf() is None:
            self.statusBar().showMessage("PDF 组件已更新，重启程序后生效", 15000)
            QMessageBox.information(
                self, "安装 PDF 组件",
                "新版 pypdf 已经装好，但程序里还在用旧版。请关闭程序重新打开，再导入论文。",
            )
            return
        self.statusBar().showMessage("PDF 组件安装完成，继续导入论文…", 8000)
        self._import_thesis(path)

    def _import_thesis(self, path: str) -> None:
        if Path(path).suffix.lower() == ".pdf" and not self._ensure_pdf_support(path):
            return
        model = str(self.settings.get("llm_model", "") or "qwen3.8-flash")
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=model,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=90,
            )
        except ApiError as exc:
            QMessageBox.warning(self, "导入论文", str(exc))
            return
        self.ppt_button.setEnabled(False)
        self.statusBar().showMessage("正在读取论文…")
        self.thesis_worker = threading.Thread(
            target=self._thesis_import_worker, args=(path, model, translator), daemon=True,
        )
        self.thesis_worker.start()

    def _thesis_import_worker(self, path: str, model: str, translator) -> None:
        try:
            self._thesis_import_body(path, model, translator)
        except Exception as exc:  # noqa: BLE001 —— 保存/合并术语出错也要把按钮还回来
            self.signals.error_message.emit(f"论文导入失败：{exc}")
            self.signals.import_finished.emit("", False)

    def _thesis_import_body(self, path: str, model: str, translator) -> None:
        try:
            context = import_thesis(
                None, Path(path), model=model,
                on_status=self.signals.status_message.emit,
                post=lambda payload: translator.complete(payload, deadline_seconds=420),
            )
        except Exception as exc:
            self.signals.error_message.emit(f"论文导入失败：{exc}")
            self.signals.import_finished.emit("", False)
            return
        ThesisStore(self.settings.base_dir).save(
            source_path=context.source_path, chunks=context.chunks, page_count=context.page_count,
            full_text=context.full_text, brief=context.brief,
        )
        self.settings.set("defense_brief", context.brief)
        merged, added = merge_terms(self.settings.glossary_terms(), context.terms)
        self.settings.set_glossary(merged)
        self.settings.set("context_source_path", context.source_path)
        self.engine.reload_context()
        pages = f"{context.page_count} 页、" if context.page_count else ""
        preview = "、".join(t["source"] for t in context.terms[:12])
        self.signals.import_finished.emit(
            f"论文导入完成：{pages}约 {context.char_count // 1000} 千字，已建立 {len(context.chunks)} 段原文索引。\n"
            f"提炼出术语 {len(context.terms)} 条，新增 {added} 条（术语库现有 {len(merged)} 条），"
            f"已自动用于语音识别和翻译。\n例如：{preview}\n\n"
            "论文摘要已更新。可以在设置里核对术语；下一步建议点「论文预习」，用强模型通读一遍。",
            True,
        )

    def _open_glossary(self) -> None:
        dialog = getattr(self, "_glossary_dialog", None)
        if dialog is None:
            dialog = GlossaryDialog(self, self.settings, on_saved=self._glossary_saved)
            self._glossary_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _glossary_saved(self) -> None:
        try:
            self.engine.reload_context()
        except Exception as exc:  # noqa: BLE001
            self.signals.error_message.emit(f"重新加载术语失败：{exc}")

    def _open_study(self) -> None:
        if getattr(self, "study_dialog", None) is None:
            self.study_dialog = StudyDialog(self, self.settings, self.qa_panel)
        self.study_dialog.refresh()
        self.study_dialog.show()
        self.study_dialog.raise_()
        self.study_dialog.activateWindow()

    # ---------------------------------------------------------------- close
    # ------------------------------------------------------ 兜底：任何没接住的异常都不许让程序退出
    def _install_error_hooks(self) -> None:
        """界面槽函数或后台线程里漏掉的异常：照常写崩溃日志（入口里装的钩子），
        再在状态栏提示一句，程序继续跑。答辩中一个按钮出错，不能把整个程序带走。"""
        previous_hook = sys.excepthook
        previous_thread_hook = threading.excepthook
        signals = self.signals

        def notify(exc_type, exc_value) -> None:
            if exc_type in (KeyboardInterrupt, SystemExit):
                return
            try:
                signals.error_message.emit(f"内部错误已拦截，程序继续运行：{exc_type.__name__}: {str(exc_value)[:120]}")
            except Exception:
                pass   # 窗口已经在关闭：只写日志

        def hook(exc_type, exc_value, exc_tb) -> None:
            try:
                previous_hook(exc_type, exc_value, exc_tb)
            except Exception:
                pass
            notify(exc_type, exc_value)

        def thread_hook(args) -> None:
            try:
                previous_thread_hook(args)
            except Exception:
                pass
            notify(args.exc_type, args.exc_value)

        sys.excepthook = hook
        threading.excepthook = thread_hook
        self._previous_hooks = (previous_hook, previous_thread_hook)

    def _restore_error_hooks(self) -> None:
        hooks = getattr(self, "_previous_hooks", None)
        if hooks:
            sys.excepthook, threading.excepthook = hooks
            self._previous_hooks = None

    # ------------------------------------------------------ 快捷用语 / 预渲染
    def _wire_quick_features(self) -> None:
        self.prerenderer = Prerenderer(self.engine)
        self.prerenderer.progress.connect(lambda text: self.phrase_panel.status_label.setText(text))
        self.prerenderer.item_done.connect(self._on_prerender_item)
        self.prerenderer.finished.connect(self._on_prerender_finished)
        self.qa_panel.prerenderer = self.prerenderer
        self.qa_panel.star_phrases_provider = self.phrase_panel.starred
        self.qa_panel.phrase_player = self.phrase_panel.speak
        self.qa_panel.on_bank_changed = self._auto_prerender
        self.phrase_panel.on_changed = self._auto_prerender
        self.phrase_panel.on_auto_render_disabled = self.prerenderer.stop
        self.qa_panel.on_pause_changed = self._on_pause_changed
        # 助答面板上直接改模型 / 深度思考：情景模式显示跟着变成「自定义」或对应模式
        self.qa_panel.model_combo.currentIndexChanged.connect(lambda _i: self._refresh_scene_bar())
        self.qa_panel.thinking_check.toggled.connect(lambda _c: self._refresh_scene_bar())

    def _prerender_allowed(self) -> bool:
        if not bool(self.settings.get("prerender_auto", False)):
            return False
        if not _has_tts_source(self.settings):
            return False
        if not self.settings.workspace_id:
            return False
        try:
            return bool(self.settings.get_api_key())
        except Exception:
            return False

    def _auto_prerender(self) -> int:
        """把还没渲染的快捷用语（先）和问答库答案（后）交给后台队列。"""
        if getattr(self, "prerenderer", None) is None or not self._prerender_allowed():
            return 0
        texts = self.phrase_panel.texts() + [e.answer for e in self.qa_panel.bank]
        added = self.prerenderer.enqueue(texts)
        if added:
            self.statusBar().showMessage(
                f"后台开始预渲染 {added} 条（快捷用语 + 问答库答案），用当前音色，完成后现场秒播", 8000)
        return added

    def _on_prerender_item(self, text: str) -> None:
        self.phrase_panel.refresh_ready(only_text=text)
        self.qa_panel.entry_ready_text(text)

    def _on_prerender_finished(self, summary: str) -> None:
        self.phrase_panel.status_label.setText(summary)
        self.phrase_panel.refresh_ready()
        if self.qa_panel._bank_dialog is not None and self.qa_panel._bank_dialog.isVisible():
            self.qa_panel._bank_dialog.refresh()

    # ------------------------------------------------------ 情景模式
    def _build_scene_bar(self) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        caption = QLabel("情景模式")
        caption.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(caption)
        self.scene_combo = NoWheelComboBox()
        self.scene_combo.setMinimumWidth(200)
        self.scene_combo.activated.connect(self._on_scene_chosen)
        row.addWidget(self.scene_combo)
        voice_caption = QLabel("音色")
        voice_caption.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(voice_caption)
        self.voice_scene_combo = NoWheelComboBox()
        self.voice_scene_combo.setMinimumWidth(180)
        self.voice_scene_combo.setToolTip("音色库里的全部音色；合成模型会自动换成该音色绑定的那个")
        self.voice_scene_combo.activated.connect(self._on_voice_chosen)
        row.addWidget(self.voice_scene_combo)
        self.scene_lock_button = QPushButton("")
        self.scene_lock_button.setCheckable(True)
        self.scene_lock_button.setMinimumWidth(96)
        self.scene_lock_button.toggled.connect(self._on_scene_lock_toggled)
        row.addWidget(self.scene_lock_button)
        self.scene_more_button = QToolButton()
        self.scene_more_button.setText("⋯")
        self.scene_more_button.setToolTip("保存 / 删除我的模式")
        self.scene_more_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.scene_more_button)
        menu.addAction("把当前设置存为我的模式…", self._save_scene_mode)
        menu.addAction("删除我的模式…", self._delete_scene_mode)
        self.scene_more_button.setMenu(menu)
        row.addWidget(self.scene_more_button)
        self._refresh_scene_bar()
        return box

    def _scene_locked(self) -> bool:
        return bool(self.settings.get("scene_locked", True))

    def _refresh_scene_bar(self) -> None:
        if not hasattr(self, "scene_combo"):
            return
        modes = scene_modes.all_modes(self.settings)
        current = scene_modes.current_mode(self.settings, modes)
        combo = self.scene_combo
        combo.blockSignals(True)
        combo.clear()
        if current is None:
            combo.addItem("自定义（设置里手动改过）", scene_modes.CUSTOM_ID)
            combo.setItemData(0, "当前模型组合和任何模式都不一致", Qt.ToolTipRole)
        for mode in modes:
            combo.addItem(mode.name if mode.builtin else f"我的：{mode.name}", mode.id)
            combo.setItemData(combo.count() - 1, mode.blurb, Qt.ToolTipRole)
        combo.setCurrentIndex(max(combo.findData(current.id if current else scene_modes.CUSTOM_ID), 0))
        combo.blockSignals(False)
        locked = self._scene_locked()
        blurb = current.blurb if current else "当前模型组合和任何模式都不一致"
        combo.setToolTip(blurb + ("\n\n已锁定：点右边的锁解锁后才能切换" if locked else
                                  "\n\n选一个就立即生效（正在答辩时会断开重连 1~2 秒），生效后自动重新锁上"))

        voice = self.voice_scene_combo
        voice.blockSignals(True)
        voice.clear()
        options = scene_modes.voice_options(self.settings)
        current_voice = str(self.settings.get("tts_voice_id", "") or "").strip()
        for option in options:
            voice.addItem(option["label"], option["voice_id"])
        if not options:
            voice.addItem("未设置音色（去设置里复刻）", "")
        voice.setCurrentIndex(max(voice.findData(current_voice), 0))
        voice.blockSignals(False)

        self.scene_lock_button.blockSignals(True)
        self.scene_lock_button.setChecked(locked)
        self.scene_lock_button.blockSignals(False)
        self.scene_lock_button.setText("🔒 已锁定" if locked else "🔓 可切换")
        self.scene_lock_button.setToolTip(
            "锁定时不能切换情景模式和音色，防止答辩中误触。点一下解锁；切换生效后会自动重新锁上。"
            if locked else "点一下重新锁上"
        )
        self.scene_lock_button.setStyleSheet(
            "" if locked else "QPushButton { background: #3d3317; border: 1px solid #f5a623; color: #ffd97a; }"
        )
        for widget in (combo, voice, self.scene_more_button):
            widget.setEnabled(not locked)

    def _on_scene_lock_toggled(self, locked: bool) -> None:
        self.settings.set("scene_locked", bool(locked))
        self._refresh_scene_bar()
        if not locked:
            self.statusBar().showMessage("情景模式已解锁：选好模式或音色后会自动重新锁上", 8000)

    def _relock_scene(self) -> None:
        self.settings.set("scene_locked", True)
        self._refresh_scene_bar()

    def _on_scene_chosen(self, index: int) -> None:
        if self._scene_locked():
            self._refresh_scene_bar()
            return
        mode_id = self.scene_combo.itemData(index)
        mode = next((m for m in scene_modes.all_modes(self.settings) if m.id == mode_id), None)
        if mode is None:
            return
        changes = scene_modes.apply_mode(self.settings, mode)
        if changes:
            try:
                self.engine.apply_settings()
            except Exception as exc:
                self.signals.error_message.emit(f"切换模式后重新加载失败：{exc}")
            self.qa_panel.sync_from_settings()
            message = f"已切换到「{mode.name}」：{scene_modes.describe_changes(changes)}"
        else:
            message = f"当前已经是「{mode.name}」"
        self._relock_scene()
        self.statusBar().showMessage(message, 15000)

    def _on_voice_chosen(self, index: int) -> None:
        if self._scene_locked():
            self._refresh_scene_bar()
            return
        voice_id = str(self.voice_scene_combo.itemData(index) or "")
        if not voice_id or voice_id == str(self.settings.get("tts_voice_id", "") or ""):
            self._relock_scene()
            return
        applied = scene_modes.apply_voice(self.settings, voice_id)
        if applied is None:
            self._relock_scene()
            return
        try:
            self.engine.apply_settings()
        except Exception as exc:
            self.signals.error_message.emit(f"换音色后重新加载失败：{exc}")
        self._relock_scene()
        self.phrase_panel.refresh_ready()
        self.script_panel.reload_voice_options()
        self.script_panel.refresh_render_state()
        queued = self._auto_prerender()
        suffix = "，新音色的预渲染已加入后台队列" if queued else ""
        self.statusBar().showMessage(f"已换音色（合成模型 {applied[1]}）{suffix}", 15000)

    def _save_scene_mode(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "存为我的模式", "给当前的模型组合起个名字：")
        if not ok or not name.strip():
            return
        mode = scene_modes.save_custom_mode(self.settings, name)
        self._relock_scene()
        self.statusBar().showMessage(f"已保存模式「{mode.name}」", 8000)

    def _delete_scene_mode(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        mine = scene_modes.custom_modes(self.settings)
        if not mine:
            QMessageBox.information(self, "删除我的模式", "还没有保存过自己的模式")
            return
        names = [m.name for m in mine]
        name, ok = QInputDialog.getItem(self, "删除我的模式", "删除哪一个？", names, 0, False)
        if not ok:
            return
        mode = mine[names.index(name)]
        scene_modes.delete_custom_mode(self.settings, mode.id)
        self._relock_scene()
        self.statusBar().showMessage(f"已删除模式「{mode.name}」", 8000)

    def closeEvent(self, event) -> None:
        self.recording_timer.stop()
        self._restore_error_hooks()
        try:
            save_session(self.settings.base_dir)   # 本场用量写进「消费记录」
        except Exception:  # noqa: BLE001
            pass
        if getattr(self, "study_dialog", None) is not None:
            self.study_dialog.shutdown()
        if hasattr(self, "qa_panel"):
            self.qa_panel.shutdown()
        if getattr(self, "prerenderer", None) is not None:
            self.prerenderer.stop()
        if hasattr(self, "script_panel"):
            # 渲染线程还会 emit 信号，而信号连到的是即将销毁的表格。
            self.script_panel.shutdown()
        if self.overlay is not None:
            self.overlay.stop()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.engine.shutdown()
        super().closeEvent(event)


QA_ACCENT = "#f5a623"

# 阿里云百炼常用模型清单：(模型 ID, 一句话简介)。下拉显示 "ID · 简介"，存储/传参用 ID。
# 大模型清单来自 chat_models 能力表（2026-09 按百炼文档核对）。
# 翻译清单剔除了关不掉思考的模型 —— 实时翻译先「思考」几秒是致命的。
TRANSLATE_MODELS: list[tuple[str, str]] = models_for(USE_TRANSLATE)
QA_MODELS: list[tuple[str, str]] = models_for(USE_QA)
CHAT_MODELS = TRANSLATE_MODELS  # 兼容旧引用
ASR_MODELS: list[tuple[str, str]] = [
    ("qwen-audio-3.1-asr-flash-streaming", "★ 推荐：官方最新，热词+上下文，术语表自动带上"),
    ("qwen-audio-3.0-asr-flash-streaming", "上一版，同样带术语表"),
    ("qwen3-asr-flash-realtime", "旧默认；带不上术语表，专有名词易错"),
    ("qwen3-asr-flash-realtime-2026-02-10", "同上，锁定新快照"),
    ("qwen3-asr-flash-realtime-2025-10-27", "同上，锁定旧快照"),
    ("fun-asr-realtime", "轻量快，方言口音友好"),
    ("fun-asr-realtime-2026-02-28", "FunASR 新快照（中英日）"),
    ("fun-asr-realtime-2025-11-07", "FunASR 锁定旧快照"),
]
TTS_MODELS: list[tuple[str, str]] = [
    ("qwen-audio-3.0-tts-plus", "★ 答辩首选：语气指令 + SSML + 并行预生成，复刻免费"),
    ("qwen-audio-3.1-tts-next", "参考录音直接生成·完整音频返回；适合讲稿预渲染，不用 voice_id"),
    ("qwen-audio-3.1-tts-flash", "同上，更新更快（首包约 300ms）"),
    ("qwen-audio-3.0-tts-flash", "3.0 提速版"),
    ("qwen3-tts-vc-2026-01-22", "⚠ 完整合成；可对照跨语种自然度，无语气指令"),
    ("qwen3-tts-vc-realtime-2026-01-15", "⚠ 低延迟，但无语气指令/无 SSML/无并行预生成"),
    ("qwen3-tts-vc-realtime-2025-11-27", "⚠ 实时旧快照，同样无语气指令"),
    ("cosyvoice-v3.5-plus", "支持克隆与语气指令；需对应音色"),
    ("cosyvoice-v3.5-flash", "v3.5 提速便宜版"),
    ("cosyvoice-v3-plus", "上一代克隆"),
    ("cosyvoice-v3-flash", "上一代快版"),
    ("cosyvoice-v2", "旧版"),
    ("cosyvoice-v1", "最旧版，仅存量"),
    ("MiniMax/speech-2.8-hd", "MiniMax 高清：另一家引擎，适合做对照；首次合成解锁 ¥9.9，样音≥10 秒"),
    ("MiniMax/speech-2.8-turbo", "MiniMax 提速版：更便宜更快"),
    ("MiniMax/speech-02-hd", "MiniMax 上一代高清"),
    ("MiniMax/speech-02-turbo", "MiniMax 上一代提速"),
]

# 复刻样音语种。跨语种复刻（中文样音 → 合成英文）官方明确说明质量下降：
# 模型要同时还原音色并迁移口腔习惯，第二件事做不好就是「说不上哪里怪」。
CLONE_LANGUAGES: list[tuple[str, str]] = [
    ("en", "英文样音 → 英文音色（答辩现场用，强烈推荐）"),
    ("zh", "中文样音 → 中文音色（讲稿预习朗读中文原稿用）"),
]

# 官方最佳区间 10~20 秒。更长不提升音色保真度，反而增加不稳定性
# （超长参考音频可能导致模型不吐 EOS 而卡住）。
RECOMMENDED_SAMPLE_SECONDS = 18.0
MAX_SAMPLE_SECONDS = 30.0


# 各个下拉里效果最好的那几个：加 ★、金色粗体，一眼就能挑到（2026-09 按百炼文档和实测选定）
RECOMMENDED_ASR = ("qwen-audio-3.1-asr-flash-streaming",)
RECOMMENDED_TRANSLATE = ("qwen3.8-flash",)
RECOMMENDED_QA = ("qwen3.8-flash", "qwen3.8-max")
RECOMMENDED_STUDY = ("qwen3.8-max",)
RECOMMENDED_TTS = ("qwen-audio-3.0-tts-plus",)


def highlight_item(combo: QComboBox, index: int) -> None:
    font = QFont(combo.font())
    font.setBold(True)
    combo.setItemData(index, font, Qt.FontRole)
    combo.setItemData(index, QColor(QA_ACCENT), Qt.ForegroundRole)


def fill_model_combo(combo: QComboBox, options: list[tuple[str, str]], current: str,
                     recommended: tuple[str, ...] = ()) -> None:
    """Fill a non-editable model dropdown so a click opens the list directly."""
    combo.setEditable(False)
    combo.clear()
    for model, blurb in options:
        if model in recommended:
            clean = blurb[2:] if blurb.startswith("★ ") else blurb
            combo.addItem(f"★ {model} · {clean}", model)
            highlight_item(combo, combo.count() - 1)
        else:
            combo.addItem(f"{model} · {blurb}", model)
    if current and combo.findData(current) < 0:
        combo.addItem(f"{current} · 自定义", current)
    combo.setCurrentIndex(max(combo.findData(current), 0))


def QFontMetricsForPx(px: int):
    """指定像素字号的字体度量，用来给固定高度的标签算行高。"""
    from PySide6.QtGui import QFontMetrics

    font = QFont()
    font.setPixelSize(int(px))
    return QFontMetrics(font)


class AnswerEditor(QPlainTextEdit):
    """可以直接改的回答框。保留 text()/setText()，和原来的 QLabel 用法兼容。"""

    def text(self) -> str:  # noqa: D401 —— 与 QLabel 同名，方便调用方不分类型
        return self.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802
        self.setPlainText(text or "")


class QaPanel(QFrame):
    """AI 助答：评委提问 → 先匹配本地问答库 → 命中就弹出预案，没命中再让大模型生成。

    朗读永远要你点按钮，程序不会自己开口。预案回答可以先改再读；读的就是
    回答框里此刻的文字。
    """

    qa_status = Signal(str)
    qa_answer = Signal(str, str)
    qa_generation_ready = Signal(object)
    qa_match = Signal(object)
    question_intent_ready = Signal(object)
    translate_ready = Signal(object)
    calibrate_ready = Signal(object)
    calibrate_zh_ready = Signal(object)

    FONT_MIN, FONT_MAX = 13, 32

    def __init__(self, settings: DefenseSettings, engine: DefenseEngine) -> None:
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.advisor_factory = QaAdvisor  # 测试可替换
        # 测试可替换：返回 chat(messages)->str，或 None 表示只用本地词面匹配
        self.match_chat_factory = self._default_match_chat
        self.intent_chat_factory = self._default_intent_chat
        # 测试可替换：返回 translate(english)->chinese，或 None 表示没法翻译
        self.translate_factory = self._default_translate
        self._translate_serial = 0
        self._zh_cache: dict[str, str] = {}
        try:
            self._font_px = int(self.settings.get("qa_font_px", 18) or 18)
        except (TypeError, ValueError):
            self._font_px = 18
        self._font_px = max(self.FONT_MIN, min(self.FONT_MAX, self._font_px))
        self.bank_store = QaBankStore(settings.base_dir)
        self.bank: list[QaEntry] = self.bank_store.load()
        # 论文原文索引（导入论文后才有）和预习笔记（论文预习后才有）
        self.thesis_store = ThesisStore(settings.base_dir)
        self.study_store = StudyStore(settings.base_dir)
        self.thesis_index = None
        self.study_notes: StudyNotes | None = None
        self._last_pages: list[str] = []
        self._calibrate_serial = 0
        self.reload_thesis()
        self._job_cancel = threading.Event()
        self._match_serial = 0
        self._answer_en = ""
        self._answer_zh = ""
        self._question = ""
        self._question_zh = ""
        # 中文框被你改过、英文还没跟着改：朗读前先按中文校准英文，再读
        self._zh_dirty = False
        self._play_after_calibrate = False
        # 由主窗口接上：后台预渲染队列、常用快捷语、播放快捷语、问答库变化回调
        self.prerenderer = None
        self.star_phrases_provider = None
        self.phrase_player = None
        self.on_bank_changed = None
        self.on_pause_changed = None
        self.popup: AnswerPopup | None = None
        self._popup_wanted = False
        self._recent: list[tuple[float, str]] = []
        self._question_parts = {}
        self._submitted_question_parts = {}
        self._question_partial = ""
        self._question_manual_pending = False
        self._question_revision = 0
        self._question_intent_cancel = threading.Event()
        self._question_intent = None
        self._question_force_answer = False
        self._question_collect_timer = QTimer(self)
        self._question_collect_timer.setSingleShot(True)
        try:
            collect_wait = int(self.settings.get("qa_collect_wait_ms", 3000))
        except (TypeError, ValueError):
            collect_wait = 3000
        self._question_collect_timer.setInterval(max(1000, min(15000, collect_wait)))
        self._question_collect_timer.timeout.connect(self._auto_submit_collected_question)
        self._current_entry: QaEntry | None = None
        self._candidates: list[QaEntry] = []
        self._bank_dialog: QaBankDialog | None = None
        self._build_ui()
        self.qa_status.connect(self._on_qa_status)
        self.qa_answer.connect(self._on_qa_answer)
        self.qa_generation_ready.connect(self._on_qa_generation_ready)
        self.qa_match.connect(self._on_qa_match)
        self.question_intent_ready.connect(self._on_question_intent_ready)
        self.translate_ready.connect(self._on_translate_ready)
        self.calibrate_ready.connect(self._on_calibrate_ready)
        self.calibrate_zh_ready.connect(self._on_calibrate_zh_ready)
        # 朗读按钮的状态：idle 绿 / pending 黄（在准备语音）/ playing 红（在读）/ paused
        self._play_state = "idle"
        self._play_clicked_at = 0.0
        self._played_text = ""
        self._answer_locked = False
        self._calibrate_zh_serial = 0
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(150)
        self._state_timer.timeout.connect(self._poll_play_state)
        # 改中文停手 1.5 秒后自动让英文跟上（手动按钮随时可点）
        self._zh_sync_timer = QTimer(self)
        self._zh_sync_timer.setSingleShot(True)
        self._zh_sync_timer.setInterval(1500)
        self._zh_sync_timer.timeout.connect(self._auto_zh_to_en)
        # 改英文时停手 1.2 秒再翻译，边打字边请求会白白浪费调用
        self._translate_timer = QTimer(self)
        self._translate_timer.setSingleShot(True)
        self._translate_timer.setInterval(1200)
        self._translate_timer.timeout.connect(self._translate_current)
        self._refresh_bank_button()
        self._apply_font_size()

    def _build_ui(self) -> None:
        self.setStyleSheet(self._panel_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("AI 助答")
        title.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        header.addWidget(title)
        self.auto_check = QCheckBox("自动")
        self.auto_check.setToolTip("收集评委连续多句，停止说话约 3 秒后才提交整题。长停顿时可关闭自动，等评委说完点「提交整题」。朗读仍需你点。")
        self.auto_check.setChecked(bool(self.settings.get("qa_auto", True)))
        self.auto_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.auto_check.toggled.connect(self._on_question_auto_toggled)
        header.addWidget(self.auto_check)
        self.submit_question_button = QPushButton("提交整题")
        self.submit_question_button.setToolTip("确认评委已经说完，把已收集的全部问题一起提交；关闭自动时也可使用。")
        self.submit_question_button.clicked.connect(self.submit_collected_question)
        self.bank_check = QCheckBox("先查问答库")
        self.bank_check.setToolTip(
            "评委提问时先在你导入的问答库里找「问的是同一件事」的条目；\n"
            "找到就直接弹出准备好的答案，找不到才让大模型现场生成。"
        )
        self.bank_check.setChecked(bool(self.settings.get("qa_bank_enabled", True)))
        self.bank_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.bank_check.toggled.connect(
            lambda checked: self.settings.set("qa_bank_enabled", bool(checked))
        )
        header.addWidget(self.bank_check)
        self.strictness_combo = NoWheelComboBox()
        for value, label in STRICTNESS_CHOICES:
            self.strictness_combo.addItem(label, value)
        self.strictness_combo.setCurrentIndex(max(0, self.strictness_combo.findData(
            str(self.settings.get("qa_match_strictness", "medium") or "medium"))))
        self.strictness_combo.setToolTip(
            "问答库匹配有多严：\n"
            "严格：大模型把握 ≥ 0.75 才用预案（断网时关键词相似度 ≥ 0.65）——宁可现场生成也不错用；\n"
            "★ 适中（默认）：≥ 0.6（断网 ≥ 0.55）；\n"
            "宽松：≥ 0.45（断网 ≥ 0.45）——更容易直接用预案，但偶尔会用到问的不是同一件事的那条。\n"
            "只算「问的是同一件事」，话题相关不算。")
        self.strictness_combo.currentIndexChanged.connect(
            lambda _i: self.settings.set("qa_match_strictness", str(self.strictness_combo.currentData())))
        header.addWidget(self.strictness_combo)
        self.thinking_check = QCheckBox("深度思考")
        self.thinking_check.setChecked(bool(self.settings.get("qa_thinking", False)))
        self.thinking_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.thinking_check.setToolTip(
            "只影响「问答库没命中、现场生成」的情况。\n"
            "开启后模型先推理再作答，答得更准但要多等几秒到几十秒。"
        )
        self.thinking_check.toggled.connect(
            lambda checked: self.settings.set("qa_thinking", bool(checked))
        )
        header.addWidget(self.thinking_check)
        self.popup_check = QCheckBox("弹窗")
        self.popup_check.setChecked(bool(self.settings.get("qa_popup", True)))
        self.popup_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.popup_check.setToolTip(
            "评委提问后弹出置顶小窗：评委的问题（中英）+ 回答 +「回答 / 不回答」。\n"
            "Windows 10 以上这个小窗不会出现在屏幕共享和录屏里。"
        )
        self.popup_check.toggled.connect(lambda checked: self.settings.set("qa_popup", bool(checked)))
        header.addWidget(self.popup_check)
        self.open_popup_button = QPushButton("打开弹窗")
        self.open_popup_button.clicked.connect(self.open_popup)
        header.addWidget(self.open_popup_button)
        header.addStretch(1)
        self.font_down_button = QPushButton("A-")
        self.font_down_button.setToolTip("字小一点")
        self.font_down_button.setMinimumWidth(52)
        self.font_down_button.clicked.connect(lambda: self.change_font_size(-2))
        header.addWidget(self.font_down_button)
        self.font_up_button = QPushButton("A+")
        self.font_up_button.setToolTip("字大一点（大屏幕可以调到 24 以上）")
        self.font_up_button.setMinimumWidth(52)
        self.font_up_button.clicked.connect(lambda: self.change_font_size(2))
        header.addWidget(self.font_up_button)
        self.bank_button = QPushButton("问答库")
        self.bank_button.setToolTip("导入问答文件、预渲染答案音频、浏览全部预案")
        self.bank_button.clicked.connect(self.open_bank_dialog)
        layout.addLayout(header)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("现场生成模型"))
        self.model_combo = NoWheelComboBox()
        fill_model_combo(self.model_combo, QA_MODELS, str(self.settings.get("qa_model", "qwen3.8-max")), RECOMMENDED_QA)
        self.model_combo.setMinimumWidth(120)
        self.model_combo.currentIndexChanged.connect(
            lambda _index: self.settings.set("qa_model", self.model_combo.currentData())
        )
        model_row.addWidget(self.model_combo, 1)
        # 问答库按钮放到这一行：标题行勾选框多，挤在一起时按钮上的字会被截掉
        model_row.addWidget(self.bank_button)
        layout.addLayout(model_row)

        question_row = QHBoxLayout()
        self.intent_check = QCheckBox("整题意图理解")
        self.intent_check.setChecked(bool(self.settings.get("qa_intent_enabled", True)))
        self.intent_check.setToolTip(
            "用下面单独选择的模型理解评委全部英文原话，不依赖中文翻译。字幕仍照常翻译。\n"
            "自动判断是否问完、核心意图和所有子问题，再匹配问答库或生成答案。\n"
            "每次整题理解会调用模型。长停顿可能无法准确判断，可关闭「自动」后手动提交。")
        self.intent_check.toggled.connect(self._on_intent_toggled)
        question_row.addWidget(self.intent_check)
        self.intent_model_combo = NoWheelComboBox()
        self.intent_model_combo.setAccessibleName("问题理解模型")
        self.intent_model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.intent_model_combo.setMinimumContentsLength(16)
        fill_model_combo(self.intent_model_combo, list(INTENT_MODELS),
                         str(self.settings.get("qa_intent_model", INTENT_MODEL) or INTENT_MODEL), (INTENT_MODEL,))
        self.intent_model_combo.setToolTip(
            "独立选择理解评委整题意图的模型。\n"
            "Max：复杂问题优先；Plus：均衡；Flash：优先速度。\n"
            "不改变字幕翻译模型，也不改变上面的回答生成模型。")
        self.intent_model_combo.currentIndexChanged.connect(self._on_intent_config_changed)
        question_row.addWidget(self.intent_model_combo, 1)
        self.intent_thinking_check = QCheckBox("理解时深度思考")
        self.intent_thinking_check.setChecked(bool(self.settings.get("qa_intent_thinking", False)))
        self.intent_thinking_check.setToolTip("只影响问题理解，适合复杂或有隐含条件的提问；可能增加等待时间。字幕翻译继续运行。")
        self.intent_thinking_check.toggled.connect(self._on_intent_config_changed)
        question_row.addWidget(self.intent_thinking_check)
        question_row.addWidget(self.submit_question_button)
        layout.addLayout(question_row)
        self._update_intent_controls()

        # 中间内容放进滚动区：面板高度不够时滚动，而不是让控件互相压叠
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 4, 0)
        body_layout.setSpacing(6)

        # 顺序按现场要看的优先级排：答案出处与可信度 → 红色提醒 → 中文要点 →
        # 换一条/手动搜 → 评委原话（原话在中间那栏字幕里也有，放最后）
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.source_label.setTextFormat(Qt.RichText)
        body_layout.addWidget(self.source_label)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(f"color: {ERR_RED};")
        self.note_label.setVisible(False)
        body_layout.addWidget(self.note_label)

        # 预案里的中文要点（先理解再读）。完整中文翻译在下面英文框的右边。
        self.points_label = QLabel("")
        self.points_label.setWordWrap(True)
        self.points_label.setToolTip("中文要点：只给你看，不会朗读")
        self.points_label.setVisible(False)
        body_layout.addWidget(self.points_label)

        pick_row = QHBoxLayout()
        self.candidate_combo = NoWheelComboBox()
        self.candidate_combo.setToolTip("匹配到的前几条预案；选错了在这里换一条")
        self.candidate_combo.activated.connect(self._on_candidate_chosen)
        pick_row.addWidget(self.candidate_combo, 1)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("手动搜预案：关键词 回车")
        self.search_edit.returnPressed.connect(self._search_bank)
        pick_row.addWidget(self.search_edit, 1)
        body_layout.addLayout(pick_row)
        self.candidate_combo.setVisible(False)

        self.question_label = QLabel("（监听评委发言中，检测到提问会先查问答库）")
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet(f"color: {TEXT_DIM};")
        body_layout.addWidget(self.question_label)
        self.intent_label = QLabel("")
        self.intent_label.setWordWrap(True)
        self.intent_label.setTextFormat(Qt.PlainText)
        self.intent_label.setStyleSheet(f"color: {QA_ACCENT};")
        self.intent_label.hide()
        body_layout.addWidget(self.intent_label)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setWidget(body)
        scroll.setMinimumHeight(64)
        scroll.setMaximumHeight(300)
        layout.addWidget(scroll, 2)

        # 回答区不放进滚动区：现场最要紧的就是这两栏，任何时候都要一眼看到。
        # 左英文（读的就是它，可改），右中文（逐句对照，只看不读）。中间可拖。
        self.answer_split = QSplitter(Qt.Horizontal)
        self.answer_split.setChildrenCollapsible(False)

        en_box = QWidget()
        en_layout = QVBoxLayout(en_box)
        en_layout.setContentsMargins(0, 0, 0, 0)
        en_layout.setSpacing(4)
        self.answer_en_caption = QLabel("English · 朗读的就是这段，可以直接改")
        en_layout.addWidget(self.answer_en_caption)
        self.answer_en = AnswerEditor()
        self.answer_en.setPlaceholderText("English 回答会出现在这里。")
        self.answer_en.setToolTip("点「朗读」读的就是这里此刻的文字，可以直接改")
        self.answer_en.setMinimumHeight(72)
        self.answer_en.textChanged.connect(self._on_answer_edited)
        en_layout.addWidget(self.answer_en, 1)
        self.answer_split.addWidget(en_box)

        zh_box = QWidget()
        zh_layout = QVBoxLayout(zh_box)
        zh_layout.setContentsMargins(0, 0, 0, 0)
        zh_layout.setSpacing(4)
        zh_caption_row = QHBoxLayout()
        self.answer_zh_caption = QLabel("中文 · 可以改，改了朗读时英文自动跟上")
        zh_caption_row.addWidget(self.answer_zh_caption, 1)
        self.en_to_zh_button = QPushButton("按英文改中文")
        self.en_to_zh_button.setToolTip("你改了左边英文：让大模型只改中文里意思对不上的地方")
        self.en_to_zh_button.clicked.connect(self._calibrate_chinese)
        zh_caption_row.addWidget(self.en_to_zh_button)
        self.zh_to_en_button = QPushButton("按中文改英文")
        self.zh_to_en_button.setToolTip("你在中文框里改了意思：点这里让大模型把左边英文改成一致的说法")
        self.zh_to_en_button.setEnabled(False)
        self.zh_to_en_button.clicked.connect(self._calibrate_english)
        zh_caption_row.addWidget(self.zh_to_en_button)
        zh_layout.addLayout(zh_caption_row)
        self.answer_zh = AnswerEditor()
        self.answer_zh.setPlaceholderText("左边英文的中文意思会出现在这里。可以直接改中文，再点「按中文改英文」。")
        self.answer_zh.setToolTip("中文不会被读出来。改了英文这里会自动重译；改了中文，点「朗读」时会先让英文跟上再读")
        self.answer_zh.setMinimumHeight(72)
        self.answer_zh.textChanged.connect(lambda: self._on_zh_edited())
        zh_layout.addWidget(self.answer_zh, 1)
        self.answer_split.addWidget(zh_box)
        self.answer_split.setSizes([500, 500])
        # 回答区放在一个宿主里：弹窗打开时整块搬进弹窗（同一对编辑框，不用来回同步），
        # 弹窗收起时再搬回来。
        self.answer_host = QWidget()
        self.answer_host_layout = QVBoxLayout(self.answer_host)
        self.answer_host_layout.setContentsMargins(0, 0, 0, 0)
        self.answer_placeholder = QPushButton("回答正显示在弹窗里 · 点这里收回到面板")
        self.answer_placeholder.setVisible(False)
        self.answer_placeholder.clicked.connect(self.close_popup)
        self.answer_host_layout.addWidget(self.answer_placeholder)
        self.answer_host_layout.addWidget(self.answer_split, 1)
        layout.addWidget(self.answer_host, 4)

        # 状态行固定两行高：自动换行的标签在窗口很矮时会和上下控件叠在一起
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.status_label.setFixedHeight(self.status_label.fontMetrics().lineSpacing() * 2 + 4)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("▶ 朗读")
        self.play_button.setMinimumHeight(46)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._play_answer)
        buttons.addWidget(self.play_button, 1)
        self.pause_button = QPushButton("⏸ 暂停")
        self.pause_button.setToolTip("暂停正在读的语音；再点一次从停下的地方接着读（Esc 是彻底停止）")
        self.pause_button.setMinimumHeight(46)
        self.pause_button.clicked.connect(self.toggle_pause)
        buttons.addWidget(self.pause_button)
        self.replay_button = QPushButton("↻ 重播")
        self.replay_button.setToolTip("把刚才读过的那段再读一遍（读过一次后已在本机，秒播）")
        self.replay_button.setMinimumHeight(46)
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.replay_last)
        buttons.addWidget(self.replay_button)
        self.stop_button = QPushButton("■ 停止")
        self.stop_button.setToolTip("立刻停止朗读（同 Esc）")
        self.stop_button.setMinimumHeight(46)
        self.stop_button.clicked.connect(self.stop_speech)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        # 第二排：编辑类按钮。一排放 7 个时面板一窄字就显示不全
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.restore_button = QPushButton("↺ 还原预案")
        self.restore_button.setToolTip("撤销你对回答的修改，恢复问答库里的原文（原文有预渲染音频时可秒播）")
        self.restore_button.clicked.connect(self._restore_entry_text)
        self.restore_button.setEnabled(False)
        buttons.addWidget(self.restore_button)
        self.regen_button = QPushButton("↻ 大模型生成")
        self.regen_button.setToolTip("不用预案，让大模型针对这个问题现场生成（会参考最相关的几条预案）")
        self.regen_button.clicked.connect(self._regenerate)
        buttons.addWidget(self.regen_button)
        self.save_bank_button = QPushButton("存入问答库")
        self.save_bank_button.setToolTip(
            "把这道题和现在框里的中英文回答存进本地问答库：\n"
            "来自问答库的就更新那一条，现场生成的就新增一条。以后同样的问题直接命中。"
        )
        self.save_bank_button.clicked.connect(self.save_to_bank)
        buttons.addWidget(self.save_bank_button)
        layout.addLayout(buttons)
        self._fit_buttons()

    # ------------------------------------------------------------ 问答库
    def reload_bank(self) -> None:
        self.bank = self.bank_store.load()
        self._refresh_bank_button()
        if self.on_bank_changed is not None:
            self.on_bank_changed()

    def sync_from_settings(self) -> None:
        """情景模式 / 设置页改了助答模型后，面板上的下拉和勾选跟着变。"""
        self.model_combo.blockSignals(True)
        fill_model_combo(self.model_combo, QA_MODELS, str(self.settings.get("qa_model", "qwen3.8-flash")), RECOMMENDED_QA)
        self.model_combo.blockSignals(False)
        self.thinking_check.blockSignals(True)
        self.thinking_check.setChecked(bool(self.settings.get("qa_thinking", False)))
        self.thinking_check.blockSignals(False)
        old_intent = (self.intent_model_combo.currentData(), self.intent_thinking_check.isChecked(), self.intent_check.isChecked())
        self.intent_model_combo.blockSignals(True)
        fill_model_combo(self.intent_model_combo, list(INTENT_MODELS),
                         str(self.settings.get("qa_intent_model", INTENT_MODEL) or INTENT_MODEL), (INTENT_MODEL,))
        self.intent_model_combo.blockSignals(False)
        self.intent_thinking_check.blockSignals(True)
        self.intent_thinking_check.setChecked(bool(self.settings.get("qa_intent_thinking", False)))
        self.intent_thinking_check.blockSignals(False)
        self.intent_check.blockSignals(True)
        self.intent_check.setChecked(bool(self.settings.get("qa_intent_enabled", True)))
        self.intent_check.blockSignals(False)
        self._update_intent_controls()
        if old_intent != (self.intent_model_combo.currentData(), self.intent_thinking_check.isChecked(), self.intent_check.isChecked()):
            self._invalidate_question_intent()
            if self._question_parts and self.auto_check.isChecked():
                self._question_collect_timer.start()

    # ------------------------------------------------------------ 查重
    def open_dedupe(self, parent=None, only_ids: set[str] | None = None) -> None:
        dialog = QaDedupeDialog(parent or self.window(), self, only_ids=only_ids)
        if dialog.pair_count == 0:
            QMessageBox.information(parent or self.window(), "问答库查重",
                                    "没有发现意思相近的问题" + ("（只查了刚导入的）" if only_ids else ""))
            return
        dialog.exec()

    def offer_dedupe(self, new_ids: set[str] | None, parent=None) -> int:
        """导入 / 生成后：新进来的条目和已有的有相近的，问一声要不要现在核对。返回相近的对数。
        答辩进行中（开始答辩后）不弹，免得打断；之后随时可以点「🔍 查重」。"""
        if new_ids is not None and not new_ids:
            return 0
        if getattr(self.engine, "_session_active", False) is True:
            return 0
        try:
            ignored = set(self.settings.get("qa_dedupe_ignored", []) or [])
            pairs = find_qa_pairs(list(self.bank), only_ids=new_ids, ignored=ignored)
        except Exception:  # noqa: BLE001
            return 0
        if not pairs:
            return 0
        strong = sum(1 for p in pairs if p.level == "很可能重复")
        self._ask_dedupe(parent, len(pairs), strong, lambda: self.open_dedupe(parent, only_ids=new_ids))
        return len(pairs)

    def _ask_dedupe(self, parent, count: int, strong: int, on_yes) -> None:
        """非模态提示：不挡住导入窗口，也不会卡住后台流程；点「现在核对」才打开查重。"""
        box = QMessageBox(parent or self.window())
        box.setWindowTitle("发现相近的问题")
        box.setText(f"刚加入的条目里有 {count} 对和已有问题意思相近（其中 {strong} 对很可能重复）。现在核对吗？\n"
                    "（只提示，不会自动删；也可以之后在问答库里点「🔍 查重」。）")
        yes = box.addButton("现在核对", QMessageBox.AcceptRole)
        box.addButton("以后再说", QMessageBox.RejectRole)
        box.setModal(False)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box.buttonClicked.connect(lambda button: on_yes() if button is yes else None)
        self._dedupe_prompt = box
        box.show()

    def _fit_buttons(self) -> None:
        """按按钮上文字的实际宽度设最小宽度：窗口再窄也不会把字截掉（会先挤别的）。"""
        for name in ("bank_button", "font_down_button", "font_up_button", "pause_button", "replay_button",
                     "stop_button", "restore_button", "regen_button", "save_bank_button"):
            button = getattr(self, name, None)
            if button is None:
                continue
            metrics = QFontMetrics(button.font())
            button.setMinimumWidth(metrics.horizontalAdvance(button.text()) + 34)

    def _refresh_bank_button(self) -> None:
        self.bank_button.setText(f"问答库（{len(self.bank)}）")
        self._fit_buttons()

    def open_bank_dialog(self) -> None:
        if self._bank_dialog is None:
            self._bank_dialog = QaBankDialog(self)
        self._bank_dialog.refresh()
        self._bank_dialog.show()
        self._bank_dialog.raise_()
        self._bank_dialog.activateWindow()

    def _default_match_chat(self):
        """用翻译那个快速模型做匹配：只需判断「是不是同一个问题」，不需要大模型。"""
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=20,
                backup=backup_channel(self.settings),
            )
        except ApiError:
            return None
        cancel = self._job_cancel

        def chat(messages):
            return translator.chat(messages, cancel_event=cancel, thinking=False)

        return chat

    # ------------------------------------------------------------- triggers
    def _default_intent_chat(self, cancel):
        model = str(self.settings.get("qa_intent_model", INTENT_MODEL) or INTENT_MODEL)
        thinking = bool(self.settings.get("qa_intent_thinking", False))
        backup = backup_channel(self.settings)
        if backup is not None:
            # A faster global backup model must not silently replace the strong
            # model selected specifically for interpreting the whole question.
            if backup.endpoint == primary_endpoint(self.settings.workspace_id):
                backup = None  # Same endpoint + same model would duplicate billing.
            else:
                backup = replace(backup, model="")
        translator = ContextTranslator(
            api_key=self.settings.get_api_key(), workspace_id=self.settings.workspace_id,
            model=model, proxy=self.settings.ws_proxy,
            proxy_mode=self.settings.proxy_mode, timeout=60,
            backup=backup,
        )
        return lambda messages: translator.chat(messages, cancel_event=cancel, thinking=thinking)

    def _update_intent_controls(self):
        enabled = self.intent_check.isChecked()
        self.intent_model_combo.setEnabled(enabled)
        self.intent_thinking_check.setEnabled(enabled)

    def _on_intent_config_changed(self, *_args):
        self.settings.update({"qa_intent_model": str(self.intent_model_combo.currentData() or INTENT_MODEL),
                              "qa_intent_thinking": self.intent_thinking_check.isChecked()})
        self._invalidate_question_intent()
        if self._question_parts and self.auto_check.isChecked():
            self._question_collect_timer.start()

    def _invalidate_question_intent(self):
        self._question_revision += 1
        self._question_intent_cancel.set()

    def _on_intent_toggled(self, checked):
        self.settings.set("qa_intent_enabled", bool(checked))
        self._update_intent_controls()
        self._invalidate_question_intent()
        if self._question_parts and self.auto_check.isChecked():
            self._question_collect_timer.start()

    def _on_question_auto_toggled(self, checked):
        self.settings.set("qa_auto", bool(checked))
        self._invalidate_question_intent()
        self._question_collect_timer.stop()
        if checked and self._question_parts:
            self._question_collect_timer.start()

    def note_committee_activity(self, text):
        text = text.strip()
        if not text:
            return
        if text != self._question_partial:
            self._question_partial = text
            self._invalidate_question_intent()
        if self._question_parts and self.auto_check.isChecked():
            self._question_collect_timer.start()

    def collect_committee_segment(self, seg_id, english):
        if not english.strip():
            return
        if not self._question_parts:
            self._job_cancel.set()
            self._match_serial += 1
            self._question_force_answer = False
            self._question_intent = None
            self.intent_label.hide()
        self._invalidate_question_intent()
        self._question_partial = ""
        self._question_parts[seg_id] = [english.strip(), ""]
        text = " ".join(part[0] for part in self._question_parts.values())
        self.question_label.setText(text)
        self._refresh_popup()
        self.qa_status.emit(f"正在收集整题（{len(self._question_parts)} 段），评委说完可点「提交整题」")
        if self._question_manual_pending:
            self._question_manual_pending = False
            self.submit_collected_question()
        elif self.auto_check.isChecked():
            self._question_collect_timer.start()

    def update_committee_translation(self, seg_id, chinese):
        if seg_id in self._question_parts:
            self._question_parts[seg_id][1] = chinese.strip()
            self._refresh_popup()
        elif seg_id in self._submitted_question_parts:
            self._submitted_question_parts[seg_id][1] = chinese.strip()
            self._question_zh = "\n".join(part[1] for part in self._submitted_question_parts.values() if part[1])
            self._refresh_popup()

    def submit_collected_question(self):
        self._analyze_collected_question(manual=True)

    def _auto_submit_collected_question(self):
        if self.auto_check.isChecked():
            self._analyze_collected_question(manual=False)

    def _analyze_collected_question(self, *, manual):
        self._question_collect_timer.stop()
        if self._question_partial:
            if manual:
                self._question_manual_pending = True
            self.qa_status.emit("最后一段仍在识别，等原话识别完整后再提交整题")
            return
        if not self._question_parts:
            return
        english = " ".join(part[0] for part in self._question_parts.values())
        if not self.intent_check.isChecked():
            self._finish_collected_question()
            return
        self._invalidate_question_intent()
        revision = self._question_revision
        cancel = threading.Event()
        self._question_intent_cancel = cancel
        recent = [s for t, s in self._recent if time.monotonic() - t <= 45][-2:]
        model = str(self.settings.get("qa_intent_model", INTENT_MODEL) or INTENT_MODEL)
        thinking_label = "深度理解" if self.settings.get("qa_intent_thinking", False) else "理解"
        self.qa_status.emit(f"正在用 {model} {thinking_label}评委整题意图与子问题（直接使用英文原话）…")

        def worker():
            intent, error = None, ""
            try:
                intent = analyze_question(english, self.intent_chat_factory(cancel), recent)
            except Exception as exc:
                error = str(exc)
            if not cancel.is_set():
                self.question_intent_ready.emit((revision, manual, intent, error))

        threading.Thread(target=worker, name="committee-question-intent", daemon=True).start()

    def _on_question_intent_ready(self, payload):
        revision, manual, intent, error = payload
        if revision != self._question_revision or not self._question_parts:
            return
        if intent is None:
            # Preserve every segment if interpretation is unavailable; never
            # fall back to only the latest sentence or repeatedly bill a retry.
            self._finish_collected_question()
            self.qa_status.emit(f"整题理解暂不可用，已按全部原话处理：{error}")
            return
        if not manual and not intent.complete:
            self.qa_status.emit("评委的问题尚未完整，继续收集… " + intent.reason)
            return
        if not manual and not intent.is_question:
            self.clear_collected_question()
            self.qa_status.emit("评委这段话没有提问，已保留原话，未生成回答")
            return
        self._finish_collected_question(intent)

    def _finish_collected_question(self, intent=None):
        english = " ".join(part[0] for part in self._question_parts.values())
        chinese = "\n".join(part[1] for part in self._question_parts.values() if part[1])
        self._submitted_question_parts = {key: list(part) for key, part in self._question_parts.items()}
        self._question_parts.clear()
        self._question_partial = ""
        self._question_manual_pending = False
        self._question_collect_timer.stop()
        self._question_intent = intent
        self.intent_label.setText("整题理解（请对照原话）：\n" + intent.context() if intent else "")
        self.intent_label.setVisible(bool(intent))
        self._invalidate_question_intent()
        direct_generate = getattr(self, "_generate_collected_directly", False)
        self._generate_collected_directly = False
        self.on_committee_done(english, chinese, force=True, direct_generate=direct_generate)

    def clear_collected_question(self):
        self._generate_collected_directly = False
        self._question_collect_timer.stop()
        self._invalidate_question_intent()
        self._question_parts.clear()
        self._submitted_question_parts.clear()
        self._question_partial = ""
        self._question_manual_pending = False
        self._question_intent = None
        self._question_force_answer = False
        self._job_cancel.set()
        self._match_serial += 1

    def on_committee_done(self, english: str, chinese: str, *, force: bool = False, direct_generate: bool = False) -> None:
        text = (english or chinese or "").strip()
        # 新问题来了：解除上一题的「已点播锁定」，允许读新答案
        self._answer_locked = False
        self._played_text = ""
        self._question = text
        self._question_force_answer = force
        self._question_zh = (chinese or "").strip() if english else ""
        self.question_label.setText(text)
        if not text:
            return
        now = time.monotonic()
        # 追问往往只有半句（"Can you explain that?"），把前面 45 秒内的几句带上
        self._recent = [(t, s) for t, s in self._recent if now - t <= 45][-2:]
        context = [s for _, s in self._recent]
        self._recent.append((now, text))
        if not force and not self.auto_check.isChecked():
            return
        self._popup_wanted = True   # 评委提问触发的：有结果就弹窗（手动搜/手动生成不弹）
        if direct_generate:
            self._regenerate()
            self._maybe_popup()
        elif self.bank_check.isChecked() and self.bank:
            self.match_then_answer(text, context)
        else:
            self.generate(text)
            self._maybe_popup()

    def match_then_answer(self, question: str, context: list[str] | None = None) -> None:
        self._job_cancel.set()
        self._job_cancel = threading.Event()
        self._match_serial += 1
        serial = self._match_serial
        bank = list(self.bank)
        chat = self.match_chat_factory()
        intent = self._question_intent
        intent_text = intent.context() if intent else ""
        # Offline lexical similarity cannot establish that a multi-part question
        # is fully answered by one prepared entry. Use entries as references only.
        allow_lexical = not intent or len(intent.subquestions) <= 1
        self.qa_status.emit(f"正在问答库（{len(bank)} 条）里找这个问题…")

        def worker() -> None:
            min_confidence, lexical = strictness_thresholds(self.settings.get("qa_match_strictness", "medium"))
            result = match_question(bank, question, chat, context=context,
                                    min_confidence=min_confidence, lexical_strong=lexical,
                                    intent=intent_text, allow_lexical_match=allow_lexical)
            self.qa_match.emit((serial, question, result))

        threading.Thread(target=worker, name="qa-bank-match", daemon=True).start()

    def _on_qa_match(self, payload) -> None:
        serial, question, result = payload
        if serial != self._match_serial or self._answer_locked:
            return  # 评委又说了新的一句 / 你已经在读当前答案：这个结果已经过时
        if result.matched:
            if NON_FIRST_PERSON.search(result.entry.answer + "\n" + result.entry.answer_zh):
                self.qa_status.emit("命中的预案含第三人称旁白，正在按评委整题改成我本人可直接说的回答…")
                self.generate(question, references=[(result.entry.question, result.entry.answer)])
                self._maybe_popup()
                return
            self._show_entry(result.entry, confidence=result.confidence,
                             method=result.method, candidates=[e for e, _ in result.candidates])
            self._maybe_popup()
            return
        self._fill_candidates([e for e, _ in result.candidates])
        reason = "（大模型不可用，只做了本地匹配）" if result.error else ""
        references = reference_pairs(result.candidates)
        self.source_label.setText("")
        if self.auto_check.isChecked() or self._question_force_answer:
            self.qa_status.emit(f"问答库没有这道题{reason}，交给大模型现场生成…")
            self.generate(question, references=references)
            self._maybe_popup()
        else:
            self.qa_status.emit(f"问答库没有这道题{reason}")

    def _regenerate(self) -> None:
        if self._question_parts:
            self._generate_collected_directly = True
            self.submit_collected_question()
            return
        self._answer_locked = False
        if not self._question.strip():
            self.qa_status.emit("还没有评委提问，无法生成")
            return
        references = []
        if self.bank:
            references = reference_pairs(QaIndex(self.bank).search(self._question, limit=3))
        self.generate(self._question, references=references)

    def generate(self, question: str, references: list[tuple[str, str]] | None = None) -> None:
        try:
            advisor = self.advisor_factory(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.model_combo.currentData()),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                backup=backup_channel(self.settings),
            )
        except ApiError as exc:
            self.qa_status.emit(str(exc))
            return
        self._job_cancel.set()
        cancel = threading.Event()
        self._job_cancel = cancel
        self._match_serial += 1  # 作废还在路上的匹配结果
        self._play_after_calibrate = False
        old_answer = bool(self.answer_en.text().strip())
        self.play_button.setText("▶ 朗读上一条" if old_answer else "▶ 朗读")
        if old_answer:
            self.source_label.setText("⚠ 新回答生成中；下方仍是上一条回答，请核对问题后再朗读")
        thinking = self.thinking_check.isChecked()
        self.qa_status.emit(
            f"正在用 {advisor.model} {'深度思考' if thinking else '生成回答'}…"
            + (" 旧回答仍可朗读。" if old_answer else "")
        )
        self._refresh_popup()
        memory = self.engine.memory
        last_report = [0]

        def report_reasoning(chars: int) -> None:
            # 每多想 200 字报一次，别把状态栏刷爆
            if chars - last_report[0] >= 200:
                last_report[0] = chars
                self.qa_status.emit(f"{advisor.model} 思考中…（已推理约 {chars} 字）")

        kwargs = {"cancel_event": cancel, "thinking": thinking, "on_reasoning": report_reasoning}
        if self._question_intent and question == self._question:
            kwargs["intent"] = self._question_intent.context()
        if references:
            kwargs["references"] = references
        # 论文预习笔记 + 按问题检索的论文原文：快模型也有事实依据
        if self.study_notes is not None:
            kwargs["notes"] = self.study_notes.render("en")
        self._last_pages = []
        if self.thesis_index is not None:
            hits = self.thesis_index.search(question, limit=3)
            if hits:
                excerpts = excerpt_pairs(hits)
                kwargs["excerpts"] = excerpts
                self._last_pages = [label for label, _ in excerpts]

        def worker() -> None:
            try:
                english, chinese = advisor.generate(memory, question, **kwargs)
            except Exception as exc:
                if "已取消" not in str(exc):
                    self.qa_status.emit(f"生成失败：{exc}")
                return
            if cancel.is_set():
                return
            if not english and not chinese:
                self.qa_status.emit("评委不是在提问，未生成回答")
                return
            self.qa_generation_ready.emit((cancel, english, chinese))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- display
    def _fill_candidates(self, entries: list[QaEntry]) -> None:
        self._candidates = list(entries)
        self.candidate_combo.clear()
        for entry in self._candidates:
            self.candidate_combo.addItem(f"{entry.id} · {entry.question}", entry.id)
        self.candidate_combo.setVisible(bool(self._candidates))

    def show_bank_entry(self, entry: QaEntry) -> None:
        """从问答库窗口双击、或手动搜到时调用。"""
        self._match_serial += 1
        self._show_entry(entry, method="manual", candidates=self._candidates or [entry])

    def _show_entry(self, entry: QaEntry, *, confidence: float | None = None,
                    method: str = "manual", candidates: list[QaEntry] | None = None) -> None:
        self._current_entry = entry
        self._answer_en = entry.answer
        self._answer_zh = entry.answer_zh.strip()
        if self._answer_zh:
            self._zh_cache[entry.answer.strip()] = self._answer_zh
        if candidates is not None:
            self._fill_candidates(candidates)
            index = self.candidate_combo.findData(entry.id)
            if index >= 0:
                self.candidate_combo.setCurrentIndex(index)
        self.answer_en.blockSignals(True)
        self.answer_en.setText(entry.answer)
        self.answer_en.blockSignals(False)
        self.play_button.setEnabled(bool(entry.answer.strip()))
        self.play_button.setText("▶ 朗读")
        points = [p.strip() for p in entry.points if p.strip()]
        if points:
            self.points_label.setText("要点：" + "；".join(points))
        elif entry.question_zh:
            self.points_label.setText("问题：" + entry.question_zh)
        else:
            self.points_label.setText("")
        self.points_label.setVisible(bool(self.points_label.text()))
        if self._answer_zh:
            self._translate_serial += 1  # 作废还在路上的旧翻译
            self.answer_zh.setText(self._answer_zh)
            self.answer_zh_caption.setText("中文 · 可以改，改了朗读时英文自动跟上")
        else:
            # 预案没带中文（比如只写了 Q/A 的文本文件）：机器翻译一份
            self._request_translation(entry.answer)
        self.note_label.setText(f"⚠ 提醒（不会朗读）：{entry.note}" if entry.note else "")
        self.note_label.setVisible(bool(entry.note))
        self.restore_button.setEnabled(False)
        self._zh_dirty = False
        self._play_after_calibrate = False

        how = {"llm": "大模型判定", "lexical": "本地关键词匹配（离线）", "manual": "手动选择"}.get(method, method)
        parts = [f"<b style='color:{OK_GREEN}'>📌 预案 {html.escape(entry.id)}</b>", html.escape(how)]
        if confidence is not None and method != "manual":
            parts.append(f"置信度 {confidence:.2f}")
        basis = BASIS_LABELS.get(entry.basis)
        if basis:
            parts.append(f"<span style='color:{basis[1]}'>{basis[0]}</span>")
        ready = self._entry_ready(entry)
        parts.append("已预渲染·可秒播" if ready else "未预渲染·朗读时现场合成")
        self.source_label.setText(" · ".join(parts)
                                  + f"<br><span style='color:{TEXT_DIM}'>预案问题：{html.escape(entry.question)}</span>")
        self.qa_status.emit("命中问答库：看一眼没问题就点「朗读」，也可以先改（Esc 打断）")
        self._refresh_popup()

    def _entry_ready(self, entry: QaEntry) -> bool:
        try:
            return self.engine.answer_ready(entry.answer)
        except Exception:
            return False

    def _on_candidate_chosen(self, index: int) -> None:
        if 0 <= index < len(self._candidates):
            self._match_serial += 1
            self._show_entry(self._candidates[index], method="manual")

    def _search_bank(self) -> None:
        query = self.search_edit.text().strip()
        if not query:
            return
        if not self.bank:
            self.qa_status.emit("问答库是空的，先点右上角「问答库」导入")
            return
        hits = QaIndex(self.bank).search(query, limit=5)
        if not hits:
            self.qa_status.emit(f"没有找到和「{query}」相关的预案")
            return
        self._fill_candidates([e for e, _ in hits])
        self.show_bank_entry(hits[0][0])

    def _on_answer_edited(self) -> None:
        text = self.answer_en.text().strip()
        self.play_button.setEnabled(bool(text))
        entry = self._current_entry
        edited = entry is not None and text != entry.answer.strip()
        self.restore_button.setEnabled(edited)
        if edited:
            self.qa_status.emit("回答已修改：朗读时会现场合成改过的版本（约需 1~3 秒）")
        cached = self._zh_cache.get(text)
        if cached is not None:
            self._translate_timer.stop()
            self._translate_serial += 1
            self.answer_zh.setText(cached)
            self.answer_zh_caption.setText("中文 · 可以改，改了朗读时英文自动跟上")
        elif text:
            self.answer_zh_caption.setText("中文翻译 · 英文改过了，停手后自动重新翻译…")
            self._translate_timer.start()

    # ------------------------------------------------------------ 中文对照
    def _default_translate(self):
        """用翻译模型把英文回答译成中文，带术语表和对话上下文，和字幕翻译一致。"""
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=30,
            )
        except ApiError:
            return None
        memory = self.engine.memory

        def translate(text: str) -> str:
            return translator.translate(memory, "en2zh", text)

        return translate

    def _translate_current(self) -> None:
        text = self.answer_en.text().strip()
        if text and text not in self._zh_cache:
            self._request_translation(text)

    def _request_translation(self, text: str) -> None:
        text = (text or "").strip()
        self._translate_serial += 1
        serial = self._translate_serial
        if not text:
            return
        translate = self.translate_factory()
        if translate is None:
            self.answer_zh.setText("")
            self.answer_zh_caption.setText("中文翻译 · 未配置百炼 API，无法自动翻译")
            return
        self.answer_zh_caption.setText("中文翻译 · 翻译中…")

        def worker() -> None:
            try:
                chinese = translate(text)
            except Exception as exc:  # noqa: BLE001 —— 翻译失败只影响对照，不影响朗读
                self.translate_ready.emit((serial, text, "", " ".join(str(exc).split())[:60]))
                return
            self.translate_ready.emit((serial, text, (chinese or "").strip(), ""))

        threading.Thread(target=worker, name="qa-answer-zh", daemon=True).start()

    def _on_translate_ready(self, payload) -> None:
        serial, text, chinese, error = payload
        if chinese:
            self._zh_cache[text] = chinese
        if serial != self._translate_serial or text != self.answer_en.text().strip():
            return  # 英文又改了，这份翻译已经过时
        if chinese:
            self.answer_zh.setText(chinese)
            self.answer_zh_caption.setText("中文 · 可以改（机器翻译）")
        else:
            self.answer_zh_caption.setText(f"中文翻译 · 翻译失败：{error}")

    # ------------------------------------------------------------ 字号
    def change_font_size(self, delta: int) -> None:
        self._font_px = max(self.FONT_MIN, min(self.FONT_MAX, self._font_px + delta))
        self.settings.set("qa_font_px", self._font_px)
        self._apply_font_size()

    def _apply_font_size(self) -> None:
        px = self._font_px
        small = max(12, px - 4)
        box = "border: 1px solid #2a3348; border-radius: 6px; padding: 6px;"
        self.answer_en.setStyleSheet(
            f"QPlainTextEdit {{ background: #0f1626; color: {TEXT_MAIN}; font-size: {px}px; {box} }}"
        )
        self.answer_zh.setStyleSheet(
            f"QPlainTextEdit {{ background: #0d1a17; color: #c9f2dc; font-size: {px}px; {box} }}"
        )
        caption = f"color: {TEXT_DIM}; font-size: {small}px;"
        self.answer_en_caption.setStyleSheet(caption)
        self.answer_zh_caption.setStyleSheet(caption)
        self.points_label.setStyleSheet(f"color: {OK_GREEN}; font-size: {max(13, px - 2)}px; font-weight: 600;")
        self.note_label.setStyleSheet(f"color: {ERR_RED}; font-size: {max(13, px - 3)}px;")
        self.source_label.setStyleSheet(f"font-size: {max(12, px - 4)}px;")
        self.question_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: {max(12, px - 4)}px;")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: {small}px;")
        self.status_label.setFixedHeight(QFontMetricsForPx(small).lineSpacing() * 2 + 6)
        self.play_button.setStyleSheet(
            f"QPushButton {{ background: {OK_GREEN}; border: none; color: #08120c;"
            f" font-weight: 700; font-size: {max(15, px - 1)}px; }}"
            f"QPushButton:hover {{ background: #4fe0a4; }}"
        )
        self.play_button.setMinimumHeight(max(46, int(px * 2.4)))
        self._fit_buttons()
        if getattr(self, "_play_state", "idle") != "idle":
            self._set_play_state(self._play_state)

    def _restore_entry_text(self) -> None:
        if self._current_entry is not None:
            self._show_entry(self._current_entry, method="manual")

    def _on_qa_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        if self.popup is not None and self.popup.isVisible():
            self.popup.status_label.setText(message)

    def _on_qa_generation_ready(self, payload):
        cancel, english, chinese = payload
        if cancel is self._job_cancel and not cancel.is_set():
            self._on_qa_answer(english, chinese)

    def _on_qa_answer(self, english: str, chinese: str) -> None:
        if self._answer_locked:
            # 你已经点了朗读：屏幕上的必须一直是正在读的那一条，新生成的丢掉
            self.qa_status.emit("新回答已丢弃：你已经在读当前这一条（要新的请点「大模型生成」）")
            return
        self._answer_en = english
        self._answer_zh = chinese
        self._current_entry = None
        self.restore_button.setEnabled(False)
        self.note_label.setVisible(False)
        self.points_label.setVisible(False)
        if chinese:
            self._zh_cache[english.strip()] = chinese
        self.answer_en.blockSignals(True)
        self.answer_en.setText(english)
        self.answer_en.blockSignals(False)
        self.play_button.setEnabled(bool(english.strip()))
        self.play_button.setText("▶ 朗读")
        self._translate_serial += 1
        self.answer_zh.setText(chinese or "（模型未提供中文对照）")
        self.answer_zh_caption.setText("中文 · 可以改，改了朗读时英文自动跟上")
        grounds = []
        if self.study_notes is not None:
            grounds.append("论文理解")
        if self._last_pages:
            grounds.append("原文 " + "、".join(dict.fromkeys(self._last_pages)))
        basis = f" · 依据：{'；'.join(grounds)}" if grounds else " · 未导入论文，仅凭 PPT 摘要"
        self.source_label.setText(f"<b style='color:{QA_ACCENT}'>🤖 大模型现场生成</b>{html.escape(basis)} · 请核对后再读")
        self._zh_dirty = False
        self._play_after_calibrate = False
        self.qa_status.emit("回答已生成，核对后点「朗读」（可先修改；Esc 可随时打断）")
        self._refresh_popup()

    def _play_answer(self) -> None:
        text = self.answer_en.text().strip()
        if self._play_state in ("pending", "playing", "paused"):
            self.qa_status.emit("正在读（或正在准备），不会再读第二遍；要停请点「停止」，要重读请点「重播」")
            return
        if text and text == self._played_text:
            self.qa_status.emit("这段已经读过了，不会重复读；确实要再读一遍请点「↻ 重播」")
            return
        if not text:
            self.qa_status.emit("英文回答框为空：请等待生成完成，或先输入英文；中文对照不会被朗读")
            return
        if self._zh_dirty and self.answer_zh.text().strip():
            # 你改的是中文：读的是英文，所以先让英文跟上你的中文，再用你的声音现场合成
            self._play_after_calibrate = True
            self._lock_answer()
            self._set_play_state("pending", "⏳ 正在按中文改英文…")
            self.qa_status.emit("你改过中文：先按中文调整英文（约 1~3 秒），调整完自动朗读…")
            self._calibrate_english()
            return
        entry = self._current_entry
        label = f"预案 {entry.id}" if entry is not None else "AI 助答"
        # 字幕浮窗里配的中文：必须对应此刻这段英文，对不上宁可不配
        zh = self._zh_cache.get(text, "")
        self._last_played = (text, label, zh)
        self._played_text = text
        self._lock_answer()
        self.replay_button.setEnabled(True)
        self._sync_pause_button(False)
        self._play_clicked_at = time.monotonic()
        self._play_initial_write = float(self.engine.last_voice_write)
        self._set_play_state("pending")
        local = self.engine.speak_prepared(text, label=label, zh=zh)
        if local:
            self.qa_status.emit("已提交本地朗读，等待播放设备开始发声…（Esc 停止）")
        else:
            self.qa_status.emit("已提交现场合成，收到音频后开始播放…（Esc 停止）")

    # ------------------------------------------------------------ 暂停 / 继续 / 重播
    def toggle_pause(self) -> None:
        try:
            paused = bool(self.engine.speech_paused)
            if paused:
                self.engine.resume_speech()
            else:
                self.engine.pause_speech()
        except Exception as exc:  # noqa: BLE001
            self.qa_status.emit(f"暂停/继续失败：{exc}")
            return
        self._sync_pause_button(not paused)
        self.qa_status.emit("已暂停：再点「继续」接着读（Esc 彻底停止）" if not paused else "继续朗读…")

    def _sync_pause_button(self, paused: bool) -> None:
        text = "▶ 继续" if paused else "⏸ 暂停"
        self.pause_button.setText(text)
        if self.popup is not None:
            self.popup.pause_button.setText(text)
        if self.on_pause_changed is not None:
            self.on_pause_changed(paused)

    def replay_last(self) -> None:
        last = getattr(self, "_last_played", None)
        if not last:
            self.qa_status.emit("还没有读过任何回答")
            return
        text, label, zh = last
        try:
            if self.engine.speech_paused:
                self.engine.resume_speech()
        except Exception:
            pass
        self._sync_pause_button(False)
        self._play_clicked_at = time.monotonic()
        self._play_initial_write = float(self.engine.last_voice_write)
        self._set_play_state("pending")
        local = self.engine.speak_prepared(text, label=f"{label}·重播", zh=zh, force=True)
        self.qa_status.emit("重播：从本机直接播放（Esc 停止）" if local else "重播：本机还没有这段声音，正在现场合成…")

    # ------------------------------------------------------------ 论文与预习
    def reload_thesis(self) -> None:
        self.thesis_index = self.thesis_store.load_index()
        self.study_notes = self.study_store.load()

    # ------------------------------------------------------------ 中文改 → 英文跟着改
    def _on_zh_edited(self, user: bool | None = None) -> None:
        if user is None:
            user = self.answer_zh.hasFocus()
        if not user:
            return
        self._zh_dirty = bool(self.answer_zh.text().strip())
        if self._zh_dirty:
            self._zh_sync_timer.start()   # 停手 1.5 秒后英文自动跟上
        has_text = bool(self.answer_zh.text().strip() and self.answer_en.text().strip())
        self.zh_to_en_button.setEnabled(has_text)
        if has_text:
            self.answer_zh_caption.setText("中文 · 你改了中文：点「朗读」会先让英文跟上再读，也可以先点右边按钮看看")
        else:
            self.answer_zh_caption.setText("中文已清空 · 仍可直接朗读左侧英文")

    def _default_calibrator(self):
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=30,
            )
        except ApiError:
            return None
        glossary = self.settings.glossary_terms()

        def run(direction: str, edited: str, other: str, context: str) -> str:
            return calibrate(translator, direction, edited, other, context=context, glossary=glossary)

        return run

    def _calibrate_english(self) -> None:
        chinese = self.answer_zh.text().strip()
        english = self.answer_en.text().strip()
        if not chinese:
            return
        run = getattr(self, "calibrator_factory", self._default_calibrator)()
        if run is None:
            self.qa_status.emit("未配置百炼 API，无法按中文改英文")
            return
        self._calibrate_serial += 1
        serial = self._calibrate_serial
        self.zh_to_en_button.setEnabled(False)
        self.qa_status.emit("正在按你改的中文调整英文回答…")
        context = self._current_entry.question if self._current_entry is not None else self._question

        def worker() -> None:
            try:
                result = run("zh2en", chinese, english, context)
            except Exception as exc:  # noqa: BLE001
                self.calibrate_ready.emit((serial, chinese, "", " ".join(str(exc).split())[:80]))
                return
            self.calibrate_ready.emit((serial, chinese, result, ""))

        threading.Thread(target=worker, name="qa-calibrate", daemon=True).start()

    def _on_calibrate_ready(self, payload) -> None:
        serial, chinese, english, error = payload
        if serial != self._calibrate_serial:
            return
        if english and chinese != self.answer_zh.text().strip() and not self._play_after_calibrate:
            return   # 你还在改中文：这份结果已经过时，等停手后的下一次
        if not english:
            self.zh_to_en_button.setEnabled(True)
            if self._play_state == "pending" and self._play_after_calibrate:
                self._set_play_state("idle")
            if self._play_after_calibrate:
                # 别卡在这里：放弃这次校准，再点一次「朗读」就直接读现在的英文
                self._play_after_calibrate = False
                self._zh_dirty = False
                self.qa_status.emit(f"按中文改英文失败（{error}），没有朗读。再点一次「朗读」会直接读左边现在的英文")
            else:
                self.qa_status.emit(f"按中文改英文失败：{error}")
            return
        # 先记下这对中英，英文框变化时就不会再把你改好的中文「重译」掉
        self._zh_cache[english.strip()] = chinese
        self.answer_en.setText(english)
        self.answer_zh_caption.setText("中文 · 可以改（英文已按你的中文调整）")
        self._zh_dirty = False
        self.zh_to_en_button.setEnabled(True)
        if self._play_after_calibrate:
            self._play_after_calibrate = False
            self._set_play_state("idle")
            self._play_answer()
            return
        self.qa_status.emit("英文已按你的中文调整。核对后点「朗读」；想长期保留就点「存入问答库」")

    def _auto_zh_to_en(self) -> None:
        if self._zh_dirty and self.answer_zh.text().strip() and self._play_state == "idle":
            self._calibrate_english()

    def _calibrate_chinese(self) -> None:
        """按英文改中文：只改中文里和英文意思对不上的地方（手动按钮）。"""
        english = self.answer_en.text().strip()
        chinese = self.answer_zh.text().strip()
        if not english:
            return
        run = getattr(self, "calibrator_factory", self._default_calibrator)()
        if run is None:
            self.qa_status.emit("未配置百炼 API，无法按英文改中文")
            return
        self._calibrate_zh_serial += 1
        serial = self._calibrate_zh_serial
        self.en_to_zh_button.setEnabled(False)
        self.qa_status.emit("正在按英文调整中文…")
        context = self._current_entry.question if self._current_entry is not None else self._question

        def worker() -> None:
            try:
                result = run("en2zh", english, chinese, context)
            except Exception as exc:  # noqa: BLE001
                self.calibrate_zh_ready.emit((serial, english, "", " ".join(str(exc).split())[:80]))
                return
            self.calibrate_zh_ready.emit((serial, english, result, ""))

        threading.Thread(target=worker, name="qa-calibrate-zh", daemon=True).start()

    def _on_calibrate_zh_ready(self, payload) -> None:
        serial, english, chinese, error = payload
        self.en_to_zh_button.setEnabled(True)
        if serial != self._calibrate_zh_serial or english != self.answer_en.text().strip():
            return
        if not chinese:
            self.qa_status.emit(f"按英文改中文失败：{error}")
            return
        self._zh_cache[english] = chinese
        self._translate_serial += 1
        self.answer_zh.setText(chinese)
        self._zh_dirty = False
        self.answer_zh_caption.setText("中文 · 可以改（已按英文调整）")
        self.qa_status.emit("中文已按英文调整")

    # ------------------------------------------------------------ 朗读按钮状态
    PLAY_STYLES = {
        "idle": ("#3ecf8e", "#08120c", "#4fe0a4"),
        "pending": ("#f5c542", "#1a1400", "#f7d56a"),
        "playing": ("#e5484d", "#ffffff", "#ef6a6e"),
        "paused": ("#b8474b", "#ffffff", "#c95a5e"),
    }

    def _set_play_state(self, state: str, text: str | None = None) -> None:
        self._play_state = state
        labels = {"idle": "▶ 朗读", "pending": "⏳ 正在准备语音…", "playing": "🔊 正在朗读…", "paused": "⏸ 已暂停"}
        label = text or labels.get(state, "▶ 朗读")
        if state == "idle" and self._played_text and self._played_text == self.answer_en.text().strip():
            label = "✓ 已读过（再读点重播）"
        bg, fg, hover = self.PLAY_STYLES.get(state, self.PLAY_STYLES["idle"])
        px = max(15, self._font_px - 1)
        style = (f"QPushButton {{ background: {bg}; border: none; color: {fg}; font-weight: 700; font-size: {px}px; }}"
                 f"QPushButton:hover {{ background: {hover}; }}")
        for button in self._play_buttons():
            button.setText(label)
            button.setStyleSheet(style)
        if state in ("pending", "playing", "paused"):
            if not self._state_timer.isActive():
                self._state_timer.start()
        else:
            self._state_timer.stop()

    def _play_buttons(self) -> list:
        buttons = [self.play_button]
        if self.popup is not None:
            buttons.append(self.popup.play_button)
        return buttons

    def _poll_play_state(self) -> None:
        try:
            last = float(self.engine.last_voice_write)
            speaking = bool(self.engine.is_speaking())
            paused = bool(self.engine.speech_paused)
        except Exception:
            self._set_play_state("idle")
            return
        now = time.monotonic()
        if self._play_state == "pending":
            if last >= self._play_clicked_at and last > getattr(self, "_play_initial_write", 0.0):
                self._set_play_state("playing")
            elif now - self._play_clicked_at > 90:
                self._set_play_state("idle")
                self.qa_status.emit("等了很久没有声音：请检查网络或音色设置")
        elif self._play_state in ("playing", "paused"):
            if paused:
                if self._play_state != "paused":
                    self._set_play_state("paused")
            elif not speaking:
                self._set_play_state("idle")
                self.qa_status.emit("读完了。要再读一遍请点「↻ 重播」")
            elif self._play_state == "paused":
                self._set_play_state("playing")

    def stop_speech(self) -> None:
        self.engine.interrupt_speech()
        self._set_play_state("idle")

    def on_external_speech_control(self, action: str) -> None:
        if action == "stop":
            self._sync_pause_button(False)
            if self._play_state != "idle":
                # 中途停掉的不算读过：允许再点一次朗读
                self._played_text = ""
                self._set_play_state("idle")
        elif action == "pause" and self._play_state == "playing":
            self._set_play_state("paused")
        elif action == "resume" and self._play_state == "paused":
            self._set_play_state("playing")

    def _lock_answer(self) -> None:
        """点了朗读：取消还在生成的新回答和还没回来的匹配结果，屏幕上的就是要读的。"""
        self._answer_locked = True
        self._job_cancel.set()
        self._match_serial += 1

    # ------------------------------------------------------------ 存入问答库
    def save_to_bank(self) -> QaEntry | None:
        english = self.answer_en.text().strip()
        if not english:
            self.qa_status.emit("回答框是空的，没有可保存的内容")
            return None
        chinese = self.answer_zh.text().strip()
        entry = self._current_entry
        entries = list(self.bank)
        if entry is not None and any(e.id == entry.id for e in entries):
            updated = QaEntry(**{**asdict(entry), "answer": english, "answer_zh": chinese})
            entries = [updated if e.id == entry.id else e for e in entries]
            message = f"已更新问答库 {entry.id}（改过的回答需要重新预渲染才能秒播）"
            saved = updated
        else:
            question = self._question.strip()
            if not question:
                self.qa_status.emit("不知道这是回答哪个问题的：请等评委提问后再保存")
                return None
            start = next_id_start([e.id for e in entries], "L")
            saved = QaEntry(
                id=f"L{start}", question=question, answer=english, answer_zh=chinese,
                category="现场保存", priority=2, basis="reason",
                note="现场生成后保存，建议赛前再核对一遍",
            )
            entries.append(saved)
            message = f"已存入问答库 {saved.id}，以后同样的问题会直接命中"
        self.bank_store.save(entries)
        self.reload_bank()
        self._current_entry = saved
        self.restore_button.setEnabled(False)
        self.qa_status.emit(message)
        if self._bank_dialog is not None:
            self._bank_dialog.refresh()
        return saved

    def shutdown(self) -> None:
        self.clear_collected_question()
        self._match_serial += 1
        self._calibrate_serial += 1
        self._translate_serial += 1
        self._translate_timer.stop()
        if self._bank_dialog is not None:
            self._bank_dialog.shutdown()
        if self.popup is not None:
            self.popup.close()

    # ------------------------------------------------------------ 弹窗
    def _maybe_popup(self) -> None:
        if not self._popup_wanted:
            return
        self._popup_wanted = False
        if not self.popup_check.isChecked():
            return
        self.open_popup()

    def open_popup(self) -> None:
        if self.popup is None:
            self.popup = AnswerPopup(self)
        self.popup.present()

    def close_popup(self) -> None:
        if self.popup is not None:
            self.popup.hide()

    def _refresh_popup(self) -> None:
        if self.popup is not None and self.popup.isVisible():
            self.popup.refresh()

    def entry_ready_text(self, text: str) -> None:
        """后台预渲染完成一条：正在显示的就是这条时，把「未预渲染」改成「可秒播」。"""
        entry = self._current_entry
        if entry is not None and entry.answer.strip() == text.strip():
            label = self.source_label
            label.setText(label.text().replace("未预渲染·朗读时现场合成", "已预渲染·可秒播"))
            self._refresh_popup()

    def _panel_style(self) -> str:
        return (
            f"QFrame {{ background: {PANEL_BG_2}; border: 1px solid #232a3a; border-radius: 10px; }}"
            "QLabel { border: none; }"
        )


def exclude_from_capture(widget: QWidget) -> bool:
    """让窗口在屏幕共享 / 录屏里不可见，本机照常显示（Windows 10 2004 及以上）。

    助答弹窗里是你的准备答案，万一共享的是整个屏幕，评委也看不到它。
    其他系统或老版本 Windows 调用失败就算了，窗口照常工作。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        wda_exclude_from_capture = 0x11
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), wda_exclude_from_capture))
    except Exception:
        return False


class AnswerPopup(QWidget):
    """评委提问后的置顶小窗：问题（中英）+ 回答（可改）+「回答 / 不回答」。

    回答的两个编辑框不是复制品：弹窗打开时把助答面板里那一整块搬进来，收起时
    搬回去。所以在哪边改都是同一份文字，朗读、还原、按中文改英文逻辑全部复用面板。
    """

    def __init__(self, panel: "QaPanel") -> None:
        super().__init__(panel.window(), Qt.Tool | Qt.WindowStaysOnTopHint)
        self.panel = panel
        # 弹出来不抢焦点：你可能正在 Teams / PPT 里操作
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("评委提问 · 要回答吗？")
        self.setStyleSheet(f"AnswerPopup {{ background: {PANEL_BG_2}; }} QLabel {{ color: {TEXT_MAIN}; }}")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.source_label = QLabel("")
        self.source_label.setTextFormat(Qt.RichText)
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        question_scroll = QScrollArea()
        question_scroll.setWidgetResizable(True)
        question_scroll.setMaximumHeight(240)
        question_scroll.setMinimumHeight(90)
        question_body = QWidget()
        question_layout = QVBoxLayout(question_body)
        question_layout.setContentsMargins(0, 0, 0, 0)
        question_scroll.setWidget(question_body)
        layout.addWidget(question_scroll)
        ask = QLabel("评委问：")
        ask.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        question_layout.addWidget(ask)
        self.question_en = QLabel("")
        self.question_en.setWordWrap(True)
        self.question_en.setTextInteractionFlags(Qt.TextSelectableByMouse)
        question_layout.addWidget(self.question_en)
        self.question_zh = QLabel("")
        self.question_zh.setWordWrap(True)
        question_layout.addWidget(self.question_zh)
        self.bank_question = QLabel("")
        self.bank_question.setWordWrap(True)
        question_layout.addWidget(self.bank_question)
        generate_row = QHBoxLayout()
        self.generate_button = QPushButton("现场大模型生成答案")
        self.generate_button.setToolTip("根据当前评委整题，结合已导入论文、预习笔记和预案，用第一人称生成；核对后点回答播放")
        self.generate_button.clicked.connect(self.panel._regenerate)
        generate_row.addWidget(self.generate_button)
        self.thinking_check = QCheckBox("深度思考")
        self.thinking_check.setChecked(panel.thinking_check.isChecked())
        self.thinking_check.toggled.connect(panel.thinking_check.setChecked)
        panel.thinking_check.toggled.connect(self.thinking_check.setChecked)
        generate_row.addWidget(self.thinking_check)
        self.model_label = QLabel("")
        generate_row.addWidget(self.model_label, 1)
        layout.addLayout(generate_row)

        self.answer_slot = QVBoxLayout()
        self.answer_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.answer_slot, 1)

        self.star_box = QWidget()
        self.star_row = QHBoxLayout(self.star_box)
        self.star_row.setContentsMargins(0, 0, 0, 0)
        self.star_row.setSpacing(6)
        layout.addWidget(self.star_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("▶ 回答（用我的声音读）")
        self.play_button.setToolTip("没改过：直接播放提前渲染好的声音；改过英文或中文：现场重新合成后播放")
        self.play_button.clicked.connect(self.panel._play_answer)
        buttons.addWidget(self.play_button, 2)
        self.pause_button = QPushButton("⏸ 暂停")
        self.pause_button.clicked.connect(self.panel.toggle_pause)
        buttons.addWidget(self.pause_button)
        self.replay_button = QPushButton("↻ 重播")
        self.replay_button.clicked.connect(self.panel.replay_last)
        buttons.addWidget(self.replay_button)
        self.stop_button = QPushButton("■ 停止")
        self.stop_button.clicked.connect(self.panel.stop_speech)
        buttons.addWidget(self.stop_button)
        self.calibrate_button = QPushButton("按中文改英文")
        self.calibrate_button.clicked.connect(self.panel._calibrate_english)
        buttons.addWidget(self.calibrate_button)
        self.restore_button = QPushButton("↺ 还原预案")
        self.restore_button.clicked.connect(self.panel._restore_entry_text)
        buttons.addWidget(self.restore_button)
        self.skip_button = QPushButton("✕ 不回答 / 收起")
        self.skip_button.setToolTip("收起这个小窗（回答回到右侧助答面板，不会朗读）")
        self.skip_button.clicked.connect(self.hide)
        buttons.addWidget(self.skip_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------ 内容
    def refresh(self) -> None:
        panel = self.panel
        px = panel._font_px
        pending = " ".join(part[0] for part in panel._question_parts.values())
        self.question_en.setText(pending or panel._question or "（评委原话见中间字幕栏）")
        self.question_en.setStyleSheet(f"color: {TEXT_MAIN}; font-size: {px + 2}px; font-weight: 600;")
        chinese = "\n".join(part[1] for part in panel._question_parts.values() if part[1]) if pending else panel._question_zh
        self.question_zh.setText(chinese)
        self.question_zh.setVisible(bool(chinese))
        self.model_label.setText(f"{panel.model_combo.currentData()} · 第一人称回答")
        self.model_label.setToolTip("与主助答面板共用已导入论文、预习笔记和问答库；未导入论文时会明确提示")
        self.generate_button.setEnabled(bool(pending or panel._question.strip()))
        self.question_zh.setStyleSheet(f"color: #c9f2dc; font-size: {px}px;")
        entry = panel._current_entry
        # 预案问题已经在顶部来源行里（「预案问题：…」），这里不重复
        self.bank_question.setVisible(False)
        source = panel.source_label.text()
        self.source_label.setText(source or f"<span style='color:{TEXT_DIM}'>正在准备回答…</span>")
        self.source_label.setStyleSheet(f"font-size: {max(12, px - 4)}px;")
        self.restore_button.setEnabled(entry is not None)
        self.status_label.setText(panel.status_label.text())
        self.status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: {max(12, px - 4)}px;")
        self.play_button.setStyleSheet(panel.play_button.styleSheet())
        if getattr(panel, "_play_state", "idle") != "idle" or panel._played_text:
            self.play_button.setText(panel.play_button.text())
        else:
            self.play_button.setText("▶ 回答（用我的声音读）")
        self.play_button.setMinimumHeight(panel.play_button.minimumHeight())
        self._rebuild_stars()

    def _rebuild_stars(self) -> None:
        while self.star_row.count():
            item = self.star_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        provider = self.panel.star_phrases_provider
        player = self.panel.phrase_player
        phrases = provider() if provider is not None else []
        self.star_box.setVisible(bool(phrases) and player is not None)
        if not phrases or player is None:
            return
        caption = QLabel("应急：")
        caption.setStyleSheet(f"color: {TEXT_DIM};")
        self.star_row.addWidget(caption)
        for phrase in phrases[:6]:
            button = QPushButton(phrase.label())
            button.setToolTip(phrase.en)
            button.setStyleSheet(
                "QPushButton { background: #2b2410; border: 1px solid #f5a623; color: #ffd97a; padding: 4px 8px; }"
            )
            button.clicked.connect(lambda _checked=False, p=phrase: player(p))
            self.star_row.addWidget(button)
        self.star_row.addStretch(1)

    # ------------------------------------------------------------ 显示 / 收起
    def attach(self) -> None:
        split = self.panel.answer_split
        if split.parent() is not self:
            self.answer_slot.addWidget(split)
            self.panel.answer_placeholder.setVisible(True)

    def detach(self) -> None:
        split = self.panel.answer_split
        if split.parent() is self:
            self.panel.answer_host_layout.addWidget(split, 1)
            self.panel.answer_placeholder.setVisible(False)

    def present(self) -> None:
        first = not self.isVisible()
        self.attach()
        self.refresh()
        if first:
            self._restore_geometry()
        self.show()
        self.raise_()
        exclude_from_capture(self)

    def _restore_geometry(self) -> None:
        raw = str(self.panel.settings.get("qa_popup_geometry", "") or "")
        try:
            x, y, w, h = (int(v) for v in raw.split(","))
        except ValueError:
            x = y = w = h = 0
        screen = QApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        if w > 200 and h > 200:
            visible = any(s.availableGeometry().contains(x + 40, y + 20) for s in QApplication.screens())
            if visible:
                self.setGeometry(x, y, w, h)
                return
        if area is not None:
            width = min(960, int(area.width() * 0.5))
            height = min(680, int(area.height() * 0.6))
            self.setGeometry(area.right() - width - 40, area.top() + 60, width, height)

    def _save_geometry(self) -> None:
        g = self.geometry()
        try:
            self.panel.settings.set("qa_popup_geometry", f"{g.x()},{g.y()},{g.width()},{g.height()}")
        except Exception:
            pass

    def hideEvent(self, event) -> None:  # noqa: N802
        self._save_geometry()
        self.detach()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.hide()
        event.accept()


class Prerenderer(QObject):
    """后台预渲染队列：问答库答案和快捷用语用当前音色提前合成、钉在本机。

    只有一个工作线程、按顺序合成，不和现场发声抢并发。连续失败 3 次就停
    （多半是没网、没配音色或欠费），不在后台反复重试花钱。
    """

    progress = Signal(str)
    item_done = Signal(str)
    finished = Signal(str)

    MAX_STREAK = 3

    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._cancel = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def enqueue(self, texts) -> int:
        """把还没渲染的加入队列，返回新加入的条数。已在本机的直接跳过。"""
        added = 0
        start = False
        with self._lock:
            for raw in texts:
                text = (raw or "").strip()
                if not text or text in self._queue:
                    continue
                try:
                    if self.engine.answer_ready(text):
                        continue
                except Exception:
                    pass
                self._queue.append(text)
                added += 1
            if self._queue and not self._running:
                self._running = True
                self._cancel = threading.Event()
                start = True
        if start:
            self._thread = threading.Thread(target=self._run, name="prerender", daemon=True)
            self._thread.start()
        return added

    def stop(self) -> None:
        with self._lock:
            self._queue.clear()
        self._cancel.set()

    def _run(self) -> None:
        cancel = self._cancel
        done = failed = streak = 0
        last_error = ""
        while True:
            with self._lock:
                if not self._queue or cancel.is_set():
                    self._queue.clear() if cancel.is_set() else None
                    self._running = False
                    break
                text = self._queue.pop(0)
                left = len(self._queue)
            self.progress.emit(f"后台预渲染中：还剩 {left + 1} 条（用当前音色，完成后现场秒播）")
            try:
                self.engine.render_answer(text, cancel=cancel)
            except Exception as exc:  # noqa: BLE001
                if cancel.is_set():
                    continue
                failed += 1
                streak += 1
                last_error = " ".join(str(exc).split())[:80]
                if streak >= self.MAX_STREAK:
                    with self._lock:
                        self._queue.clear()
                        self._running = False
                    self.finished.emit(f"后台预渲染已暂停：连续失败 {streak} 次（{last_error}）")
                    return
                continue
            streak = 0
            done += 1
            self.item_done.emit(text)
        if cancel.is_set():
            self.finished.emit(f"后台预渲染已停止：完成 {done} 条")
        elif done or failed:
            tail = f"，失败 {failed} 条（{last_error}）" if failed else ""
            self.finished.emit(f"后台预渲染完成：{done} 条{tail}")


class QuickPhrasePanel(QFrame):
    """快捷用语：答不上来、没听清、要争取时间时点一下，用你的声音说出来。"""

    FONT_PX = 15

    def __init__(self, settings: DefenseSettings, engine) -> None:
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.store = QuickPhraseStore(settings.base_dir)
        self.recordings = PhraseRecordings(settings.base_dir)
        self.phrases: list[QuickPhrase] = self.store.load()
        self.suggest_chat_factory = self._default_suggest_chat   # 测试可替换
        self._suggest_serial = 0
        self.on_changed = None       # 主窗口接上：改了用语后触发后台预渲染
        self.on_auto_render_disabled = None
        self._buttons: dict[str, QPushButton] = {}
        self.setStyleSheet(
            f"QuickPhrasePanel {{ background: {PANEL_BG_2}; border: 1px solid #232a3a; border-radius: 10px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        header = QHBoxLayout()
        title = QLabel("快捷用语")
        title.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        header.addWidget(title)
        hint = QLabel("点一下就用你的声音说（Esc 停）")
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        header.addWidget(hint)
        header.addStretch(1)
        self.ready_label = QLabel("")
        self.ready_label.setStyleSheet(f"color: {TEXT_DIM};")
        header.addWidget(self.ready_label)
        self.auto_render_check = QCheckBox("自动预渲染")
        self.auto_render_check.setChecked(bool(settings.get("prerender_auto", False)))
        self.auto_render_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.auto_render_check.setToolTip(
            "开着：启动后、换音色后、问答库或快捷用语有变化时，在后台用当前音色把\n"
            "快捷用语和问答库答案提前合成好（会产生少量语音合成费用），现场点了秒播、断网也能播。"
        )
        self.auto_render_check.toggled.connect(self._on_auto_render_toggled)
        header.addWidget(self.auto_render_check)
        self.suggest_check = QCheckBox("按问题推荐")
        self.suggest_check.setChecked(bool(settings.get("phrase_suggest", True)))
        self.suggest_check.setStyleSheet(f"color: {TEXT_DIM};")
        self.suggest_check.setToolTip("评委每问一个问题，用快模型（flash）从快捷用语里挑出最适合这道题的几条，"
                                      "显示在最上面一排（答不上来 / 争取时间时直接点）")
        self.suggest_check.toggled.connect(lambda c: self.settings.set("phrase_suggest", bool(c)))
        header.addWidget(self.suggest_check)
        self.record_button = QPushButton("🎙 录自己的声音…")
        self.record_button.setToolTip("给快捷用语录你自己的真声（不用克隆音色），录过的点了就播录音；也可以在按钮上右键录")
        self.record_button.clicked.connect(lambda: self.open_recorder(None))
        header.addWidget(self.record_button)
        self.edit_button = QPushButton("编辑…")
        self.edit_button.clicked.connect(self.open_editor)
        header.addWidget(self.edit_button)
        layout.addLayout(header)
        # 按评委问题推荐的几条（flash 挑的）
        self.suggest_box = QWidget()
        self.suggest_layout = QHBoxLayout(self.suggest_box)
        self.suggest_layout.setContentsMargins(0, 0, 0, 0)
        self.suggest_label = QLabel("💡 这题可能用得上：")
        self.suggest_label.setStyleSheet(f"color: {WARN_ORANGE}; font-weight: 600;")
        self.suggest_layout.addWidget(self.suggest_label)
        self._suggest_buttons: list[QPushButton] = []
        self.suggest_box.setVisible(False)
        layout.addWidget(self.suggest_box)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.status_label)
        self.rebuild()

    def _on_auto_render_toggled(self, checked: bool) -> None:
        self.settings.set("prerender_auto", bool(checked))
        if checked and self.on_changed is not None:
            self.on_changed()
        elif not checked and self.on_auto_render_disabled is not None:
            self.on_auto_render_disabled()

    # ------------------------------------------------------------ 按钮
    def rebuild(self) -> None:
        current = self.tabs.currentIndex()
        self.tabs.clear()
        self._buttons = {}
        for category, label in PHRASE_CATEGORIES:
            items = [p for p in self.phrases if p.category == category]
            if not items:
                continue
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(4, 4, 4, 4)
            grid.setSpacing(6)
            for index, phrase in enumerate(items):
                button = QPushButton(self._button_text(phrase))
                button.setMinimumHeight(34)
                button.clicked.connect(lambda _checked=False, p=phrase: self.speak(p))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                button.customContextMenuRequested.connect(
                    lambda pos, p=phrase, b=button: self._phrase_menu(p, b, pos))
                grid.addWidget(button, index // 3, index % 3)
                self._buttons[phrase.id] = button
            grid.setRowStretch(grid.rowCount(), 1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, f"{label}（{len(items)}）")
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
        self.refresh_ready()

    def _button_text(self, phrase: QuickPhrase) -> str:
        mark = "🎙 " if self.recordings.has(phrase.id) else ""
        return mark + ("★ " if phrase.star else "") + phrase.label()

    def _phrase_ready(self, phrase: QuickPhrase) -> bool:
        if self.recordings.has(phrase.id):
            return True
        try:
            return bool(self.engine.answer_ready(phrase.en))
        except Exception:
            return False

    def refresh_ready(self, only_text: str | None = None) -> None:
        ready_count = 0
        for phrase in self.phrases:
            button = self._buttons.get(phrase.id)
            ready = self._phrase_ready(phrase)
            ready_count += ready
            if button is None or (only_text is not None and phrase.en.strip() != only_text.strip()):
                continue
            border = "1px solid #2fae74" if ready else "1px dashed #56607a"
            button.setStyleSheet(
                f"QPushButton {{ background: #16233a; border: {border}; color: {TEXT_MAIN};"
                f" font-size: {self.FONT_PX}px; text-align: left; padding: 4px 10px; }}"
                "QPushButton:hover { background: #1f3150; }"
            )
            state = ("🎙 你自己录的声音 · 点了直接播（右键可重录 / 删除）" if self.recordings.has(phrase.id)
                     else "已预渲染 · 点了秒播" if ready else "未预渲染 · 点了现场合成（约 1~3 秒）")
            button.setToolTip(f"{phrase.en}\n\n{phrase.zh}\n{state}")
        self.ready_label.setText(f"已预渲染 {ready_count}/{len(self.phrases)}")

    def speak(self, phrase: QuickPhrase) -> None:
        text = phrase.en.strip()
        if not text:
            return
        self._pending_phrase_text = text
        recorded = self.recordings.load(phrase.id)
        if recorded:
            try:
                self.engine.play_script_segment(recorded, text=text, zh=phrase.zh, tag=f"phrase-{phrase.id}")
            except Exception as exc:  # noqa: BLE001
                self.status_label.setText(f"播放录音失败：{exc}")
                return
            self.status_label.setText(f"🎙 已提交录音，等待发声：{phrase.label()}")
            return
        try:
            local = self.engine.speak_prepared(text, label="快捷用语", zh=phrase.zh)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"播放失败：{exc}")
            return
        how = "本地音频已提交，等待发声" if local else "现场合成中，收到音频后播放"
        self.status_label.setText(f"{how}：{text}")

    def on_voice_started(self, text: str) -> None:
        pending = getattr(self, "_pending_phrase_text", "")
        if pending and (text.strip() == pending or pending.startswith(text.strip())):
            self.status_label.setText(f"▶ 已开始播放：{pending}")
            self._pending_phrase_text = ""

    def starred(self) -> list[QuickPhrase]:
        return [p for p in self.phrases if p.star]

    # ------------------------------------------------------------ 自己录音
    def _phrase_menu(self, phrase: QuickPhrase, button: QPushButton, pos) -> None:
        menu = QMenu(button)
        menu.addAction("▶ 播放（给评委）", lambda: self.speak(phrase))
        menu.addAction("🎙 录我自己的声音…", lambda: self.open_recorder(phrase))
        if self.recordings.has(phrase.id):
            menu.addAction("🗑 删除录音（改回克隆音色）", lambda: self.delete_recording(phrase))
        self._exec_menu(menu, button.mapToGlobal(pos))

    def _exec_menu(self, menu, point) -> None:
        menu.exec(point)   # 单独成方法：测试里替换掉

    def open_recorder(self, phrase: QuickPhrase | None) -> None:
        dialog = PhraseRecordDialog(self, phrase)
        dialog.exec()
        self.rebuild()

    def delete_recording(self, phrase: QuickPhrase) -> None:
        self.recordings.delete(phrase.id)
        self.rebuild()
        self.status_label.setText(f"已删除「{phrase.label()}」的录音，改回用克隆音色读")

    # ------------------------------------------------------------ 按问题推荐
    def _default_suggest_chat(self):
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(), workspace_id=self.settings.workspace_id,
                model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode, timeout=15,
            )
        except ApiError:
            return None
        return lambda messages: translator.chat(messages, thinking=False)

    def suggest_for(self, question: str) -> None:
        """评委问完一题：后台用 flash 挑几条最合适的快捷用语，放在最上面一排。"""
        question = (question or "").strip()
        if not question or not self.suggest_check.isChecked() or not self.phrases:
            return
        chat = self.suggest_chat_factory()
        if chat is None:
            return
        self._suggest_serial += 1
        serial = self._suggest_serial
        phrases = list(self.phrases)
        relay = self._suggest_relay()

        def worker() -> None:
            try:
                ids = parse_suggestions(chat(build_suggest_messages(question, phrases)), phrases)
            except Exception:  # noqa: BLE001 —— 推荐失败就不显示，不影响别的
                ids = []
            relay.done.emit(serial, ids)

        threading.Thread(target=worker, name="phrase-suggest", daemon=True).start()

    def _suggest_relay(self):
        relay = getattr(self, "_relay", None)
        if relay is None:
            relay = _SuggestRelay()
            relay.done.connect(self._show_suggestions)
            self._relay = relay
        return relay

    def _show_suggestions(self, serial: int, ids) -> None:
        if serial != self._suggest_serial:
            return   # 评委又问了新的问题
        for button in self._suggest_buttons:
            button.deleteLater()
        self._suggest_buttons = []
        by_id = {p.id: p for p in self.phrases}
        chosen = [by_id[i] for i in ids if i in by_id]
        for phrase in chosen:
            label = self._button_text(phrase)
            button = QPushButton(label if len(label) <= 16 else label[:15] + "…")   # 太长就省略，完整内容看悬停
            button.setToolTip(f"{phrase.label()}\n\n{phrase.en}")
            button.setStyleSheet(
                f"QPushButton {{ background: #2a2410; border: 1px solid {WARN_ORANGE}; color: {TEXT_MAIN};"
                f" font-size: {self.FONT_PX}px; padding: 4px 10px; }}"
                "QPushButton:hover { background: #3a3216; }")
            button.clicked.connect(lambda _c=False, p=phrase: self.speak(p))
            self.suggest_layout.addWidget(button)
            self._suggest_buttons.append(button)
        self.suggest_box.setVisible(bool(chosen))

    def texts(self) -> list[str]:
        """需要预渲染的英文：自己录过音的不用再合成。"""
        return [p.en for p in self.phrases if not self.recordings.has(p.id)]

    # ------------------------------------------------------------ 编辑
    def open_editor(self) -> None:
        dialog = QuickPhraseDialog(self, self.phrases)
        if dialog.exec() == QDialog.Accepted:
            self.set_phrases(dialog.result_phrases())

    def set_phrases(self, phrases: list[QuickPhrase]) -> None:
        self.phrases = list(phrases)
        self.store.save(self.phrases)
        self.rebuild()
        if self.on_changed is not None:
            self.on_changed()


class _SuggestRelay(QObject):
    done = Signal(int, object)


class PhraseRecordDialog(QDialog):
    """给一条快捷用语录你自己的真声：录 → 试听（只在耳机里）→ 保存。"""

    level = Signal(float)

    def __init__(self, panel: QuickPhrasePanel, phrase: QuickPhrase | None) -> None:
        super().__init__(panel.window())
        self.panel = panel
        self.settings = panel.settings
        self.setWindowTitle("🎙 快捷用语 · 录自己的声音")
        self.resize(720, 360)
        self._frames = bytearray()
        self._capture = None
        self._started = 0.0
        self._pcm = b""
        self._preview_stop = threading.Event()
        self.capture_factory = None    # 测试可替换：(callback) -> 有 start/stop 的对象
        self.player_factory = None     # 测试可替换：() -> 有 write/close 的播放器
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("录哪一条"))
        self.phrase_combo = NoWheelComboBox()
        for item in panel.phrases:
            mark = "🎙 " if panel.recordings.has(item.id) else ""
            self.phrase_combo.addItem(f"{mark}{PHRASE_CATEGORY_LABELS.get(item.category, '')} · {item.label()}", item.id)
        if phrase is not None:
            self.phrase_combo.setCurrentIndex(max(0, self.phrase_combo.findData(phrase.id)))
        self.phrase_combo.currentIndexChanged.connect(lambda _i: self._show_script())
        row.addWidget(self.phrase_combo, 1)
        layout.addLayout(row)
        self.script_label = QLabel("")
        self.script_label.setWordWrap(True)
        self.script_label.setStyleSheet(f"font-size: 18px; color: {TEXT_MAIN}; padding: 8px;")
        layout.addWidget(self.script_label, 1)
        self.state_label = QLabel("点「开始录音」，照着上面的英文读一遍，读完点「停止」。首尾的空白会自动剪掉。")
        self.state_label.setWordWrap(True)
        self.state_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.state_label)
        buttons = QHBoxLayout()
        self.record_button = QPushButton("● 开始录音")
        self.record_button.setMinimumHeight(40)
        self.record_button.clicked.connect(self.toggle_record)
        buttons.addWidget(self.record_button)
        self.preview_button = QPushButton("▶ 试听（只在耳机里）")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview)
        buttons.addWidget(self.preview_button)
        self.save_button = QPushButton("保存")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.save_button)
        self.delete_button = QPushButton("删除这条的录音")
        self.delete_button.clicked.connect(self.delete)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._show_script()

    def _phrase(self) -> QuickPhrase | None:
        pid = self.phrase_combo.currentData()
        return next((p for p in self.panel.phrases if p.id == pid), None)

    def _show_script(self) -> None:
        phrase = self._phrase()
        if phrase is None:
            self.script_label.setText("还没有快捷用语")
            return
        self.script_label.setText(f"{phrase.en}\n\n（{phrase.zh}）")
        has = self.panel.recordings.has(phrase.id)
        self.delete_button.setEnabled(has)
        existing = self.panel.recordings.load(phrase.id) if has else None
        if existing and not self._frames:
            self._pcm = existing
            self.preview_button.setEnabled(True)
            self.state_label.setText(f"这条已经录过（{len(existing) / (RECORD_RATE * 2):.1f} 秒），可以试听或重录")

    def toggle_record(self) -> None:
        if self._capture is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        self._frames = bytearray()
        self._pcm = b""
        try:
            if self.capture_factory is not None:
                capture = self.capture_factory(self._frames.extend)
            else:
                from ..audio import MicrophoneCapture
                capture = MicrophoneCapture(self.settings.shared.get("input_device"), self._frames.extend,
                                            sample_rate=RECORD_RATE)
            capture.start()
        except Exception as exc:  # noqa: BLE001
            self.state_label.setText(f"麦克风打不开：{exc}")
            return
        self._capture = capture
        self._started = time.monotonic()
        self.record_button.setText("■ 停止")
        self.preview_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self._timer.start()

    def _stop(self) -> None:
        capture, self._capture = self._capture, None
        self._timer.stop()
        try:
            capture.stop()
        except Exception:  # noqa: BLE001
            pass
        self.record_button.setText("● 重新录")
        self._pcm = clean_recording(bytes(self._frames), sample_rate=RECORD_RATE)
        seconds = len(self._pcm) / (RECORD_RATE * 2)
        if seconds < 0.3:
            self.state_label.setText("没录到声音（太短或太小），检查一下麦克风再录一次")
            return
        self.state_label.setText(f"录好了：{seconds:.1f} 秒。先试听，满意就点「保存」")
        self.preview_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def _tick(self) -> None:
        seconds = time.monotonic() - self._started
        self.state_label.setText(f"录音中… {seconds:.0f} 秒（读完点「停止」）")
        if seconds > 60:
            self._stop()

    def preview(self) -> None:
        from ..audio import MultiOutputPlayer
        from .rehearsal import play_pcm

        pcm = self._pcm
        if not pcm:
            return
        self._preview_stop.set()
        stop = threading.Event()
        self._preview_stop = stop
        factory = self.player_factory or (
            lambda: MultiOutputPlayer([resolve_monitor_device(self.settings)], RECORD_RATE))

        def worker() -> None:
            player = None
            try:
                player = factory()
                if hasattr(player, "__enter__"):
                    player.__enter__()
                play_pcm(pcm, player, stop, threading.Event(), lambda _p: None)
            except Exception:  # noqa: BLE001
                pass
            finally:
                if player is not None:
                    try:
                        player.close()
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=worker, name="phrase-record-preview", daemon=True).start()

    def save(self) -> None:
        phrase = self._phrase()
        if phrase is None or not self._pcm:
            return
        try:
            self.panel.recordings.save(phrase.id, self._pcm)
        except OSError as exc:
            QMessageBox.warning(self, "保存录音", str(exc))
            return
        self.state_label.setText(f"已保存。以后点「{phrase.label()}」就播你自己的声音")
        self.save_button.setEnabled(False)
        self.delete_button.setEnabled(True)

    def delete(self) -> None:
        phrase = self._phrase()
        if phrase is None:
            return
        self.panel.recordings.delete(phrase.id)
        self._pcm = b""
        self.preview_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.state_label.setText("已删除录音，这条改回用克隆音色读")

    def closeEvent(self, event) -> None:
        if self._capture is not None:
            self._stop()
        self._preview_stop.set()
        super().closeEvent(event)


class QuickPhraseDialog(QDialog):
    """编辑快捷用语：中文（按钮上显示）/ 英文（读出来的）/ 分类 / 常用。"""

    translated = Signal(int, str, str)   # 行号, 英文, 错误

    COLUMNS = ["分类", "中文（按钮上显示）", "English（读出来的）", "常用"]

    def __init__(self, panel: QuickPhrasePanel, phrases: list[QuickPhrase]) -> None:
        super().__init__(panel.window())
        self.panel = panel
        self.setWindowTitle("编辑快捷用语")
        self.resize(1100, 640)
        self._ids: list[str] = []
        self.calibrator = None   # 测试可替换：run(direction, edited, other, context) -> str
        layout = QVBoxLayout(self)
        tip = QLabel(
            "中文只显示在按钮上给你看，读出来的是英文。只写了中文的行，选中后点「按中文生成英文」。\n"
            "勾选「常用」的会出现在助答弹窗底部，一键可点。保存后会在后台用你的音色预渲染。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(tip)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)
        for phrase in phrases:
            self._append_row(phrase)
        row = QHBoxLayout()
        add_button = QPushButton("新增一条")
        add_button.clicked.connect(lambda: self._append_row(
            QuickPhrase(next_phrase_id(self.result_phrases()), self._current_category(), "", ""), edit=True))
        row.addWidget(add_button)
        delete_button = QPushButton("删除所选")
        delete_button.clicked.connect(self.delete_selected)
        row.addWidget(delete_button)
        self.translate_button = QPushButton("按中文生成英文（所选行）")
        self.translate_button.clicked.connect(self.translate_selected)
        row.addWidget(self.translate_button)
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self.reset_defaults)
        row.addWidget(reset_button)
        row.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(self.status_label)
        save_button = QPushButton("保存")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self._save)
        row.addWidget(save_button)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        row.addWidget(cancel_button)
        layout.addLayout(row)
        self.translated.connect(self._on_translated)

    def _current_category(self) -> str:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            combo = self.table.cellWidget(rows[0], 0)
            if combo is not None:
                return str(combo.currentData())
        return PHRASE_CATEGORIES[0][0]

    def _append_row(self, phrase: QuickPhrase, edit: bool = False) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        combo = NoWheelComboBox()
        for key, label in PHRASE_CATEGORIES:
            combo.addItem(label, key)
        combo.setCurrentIndex(max(combo.findData(phrase.category), 0))
        self.table.setCellWidget(row, 0, combo)
        self.table.setItem(row, 1, QTableWidgetItem(phrase.zh))
        self.table.setItem(row, 2, QTableWidgetItem(phrase.en))
        star = QTableWidgetItem("")
        star.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        star.setCheckState(Qt.Checked if phrase.star else Qt.Unchecked)
        self.table.setItem(row, 3, star)
        self._ids.append(phrase.id)
        if edit:
            self.table.setCurrentCell(row, 1)
            self.table.editItem(self.table.item(row, 1))
        return row

    def result_phrases(self) -> list[QuickPhrase]:
        phrases: list[QuickPhrase] = []
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            zh_item, en_item, star_item = (self.table.item(row, c) for c in (1, 2, 3))
            phrases.append(QuickPhrase(
                self._ids[row],
                str(combo.currentData()) if combo is not None else PHRASE_CATEGORIES[0][0],
                (zh_item.text() if zh_item else "").strip(),
                (en_item.text() if en_item else "").strip(),
                bool(star_item and star_item.checkState() == Qt.Checked),
            ))
        return phrases

    def delete_selected(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
            del self._ids[row]

    def reset_defaults(self) -> None:
        answer = QMessageBox.question(self, "恢复默认", "用内置的默认用语替换当前列表吗？（点保存后才生效）")
        if answer != QMessageBox.Yes:
            return
        self.table.setRowCount(0)
        self._ids = []
        for phrase in default_phrases():
            self._append_row(phrase)

    def _make_calibrator(self):
        if self.calibrator is not None:
            return self.calibrator
        settings = self.panel.settings
        try:
            translator = ContextTranslator(
                api_key=settings.get_api_key(),
                workspace_id=settings.workspace_id,
                model=str(settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=settings.ws_proxy,
                proxy_mode=settings.proxy_mode,
                timeout=30,
            )
        except ApiError:
            return None

        def run(direction: str, edited: str, other: str, context: str) -> str:
            return calibrate(translator, direction, edited, other, context=context)

        return run

    def translate_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        jobs = []
        for row in rows:
            zh_item = self.table.item(row, 1)
            zh = (zh_item.text() if zh_item else "").strip()
            if zh:
                en_item = self.table.item(row, 2)
                jobs.append((row, zh, (en_item.text() if en_item else "").strip()))
        if not jobs:
            self.status_label.setText("先选中写了中文的行")
            return
        run = self._make_calibrator()
        if run is None:
            self.status_label.setText("未配置百炼 API，没法自动生成英文")
            return
        self.translate_button.setEnabled(False)
        self.status_label.setText(f"正在生成 {len(jobs)} 条英文…")
        context = "论文答辩现场的一句过渡语或客套话：口语化、礼貌、简短，适合直接说出口"

        def worker() -> None:
            for row, zh, en in jobs:
                try:
                    result = run("zh2en", zh, en, context)
                except Exception as exc:  # noqa: BLE001
                    self.translated.emit(row, "", " ".join(str(exc).split())[:60])
                    continue
                self.translated.emit(row, result, "")
            self.translated.emit(-1, "", "")

        threading.Thread(target=worker, name="phrase-translate", daemon=True).start()

    def _on_translated(self, row: int, english: str, error: str) -> None:
        if row < 0:
            self.translate_button.setEnabled(True)
            if not self.status_label.text().startswith("失败"):
                self.status_label.setText("英文已生成，核对后点保存")
            return
        if error:
            self.status_label.setText(f"失败：{error}")
            return
        if row < self.table.rowCount():
            self.table.setItem(row, 2, QTableWidgetItem(english))

    def _save(self) -> None:
        phrases = self.result_phrases()
        missing = [p.zh or "（空行）" for p in phrases if not p.en]
        if missing:
            QMessageBox.warning(self, "还有没写英文的", "这些行还没有英文，不能朗读：\n" + "\n".join(missing[:8])
                                + "\n\n选中它们点「按中文生成英文」，或者删掉。")
            return
        self.accept()


MANAGE_FONT_MIN, MANAGE_FONT_MAX = 11, 26


def enable_window_buttons(dialog: QWidget) -> None:
    """对话框默认只有关闭按钮：加上最小化、最大化，长时间开着的管理窗口要能收起来。"""
    dialog.setWindowFlags(
        Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint
        | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
    )


def manage_font_px(settings) -> int:
    try:
        px = int(settings.get("study_font_px", 14) or 14)
    except (TypeError, ValueError):
        px = 14
    return max(MANAGE_FONT_MIN, min(MANAGE_FONT_MAX, px))


def font_buttons(owner, on_change) -> tuple[QPushButton, QPushButton]:
    """A- / A+ 两个按钮。管理窗口共用一个字号设置（study_font_px）。"""
    down = QPushButton("A-")
    down.setToolTip("字小一点")
    down.setMinimumWidth(44)
    up = QPushButton("A+")
    up.setToolTip("字大一点")
    up.setMinimumWidth(44)

    def change(delta: int) -> None:
        px = max(MANAGE_FONT_MIN, min(MANAGE_FONT_MAX, manage_font_px(owner.settings) + delta))
        owner.settings.set("study_font_px", px)
        on_change(px)

    down.clicked.connect(lambda: change(-1))
    up.clicked.connect(lambda: change(1))
    return down, up


class QaBankDialog(QDialog):
    """问答库管理：上面是列表（中英文问题都看得到），下面是选中那条的完整中英问答，
    可以直接改、保存、删；缺中文的一键让大模型补上；也能导入其他 AI 写的问答。

    不是模态窗口：预渲染几十条要几分钟，期间主界面照常可用。关闭窗口只是
    隐藏，渲染会在后台继续；退出程序时才真正停下。
    """

    render_progress = Signal(str, str)   # entry_id, 状态文字
    render_finished = Signal(str)
    fill_progress = Signal(str)
    fill_done = Signal(object, str)      # {id: (q_zh, a_zh)}, 错误
    review_progress = Signal(str)
    review_done = Signal(object, str)

    COLUMNS = ["编号", "可能性", "问题（中文）", "问题（英文）", "本地音频"]
    AUDIO_COL = 4

    def __init__(self, panel: QaPanel) -> None:
        super().__init__(panel.window())
        enable_window_buttons(self)
        self.panel = panel
        self.settings = panel.settings
        self._render_cancel = threading.Event()
        self._fill_cancel = threading.Event()
        self._rendering = False
        self._current_row = -1
        self._loading_detail = False
        self.fill_translator_factory = self._make_translator   # 测试可替换
        self.review_post_factory = None
        self._review_cancel = threading.Event()
        self._review_running = False
        self.setWindowTitle("问答库 · 预案管理")
        self.resize(1320, 900)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        top.addWidget(self.info_label, 1)
        self.font_down_button, self.font_up_button = font_buttons(self, self._apply_font)
        top.addWidget(self.font_down_button)
        top.addWidget(self.font_up_button)
        layout.addLayout(top)

        row = QHBoxLayout()
        self.import_button = QPushButton("导入问答文件…")
        self.import_button.setToolTip("本程序格式的 JSON，或者每条写成 Q: / A: / ZH: 的文本")
        self.import_button.clicked.connect(self.import_file)
        row.addWidget(self.import_button)
        self.import_ai_button = QPushButton("导入其他 AI 的问答…")
        self.import_ai_button.setToolTip("把 ChatGPT / Kimi / 豆包等写的问答粘贴进来或选文件，大模型识别成一问一答，你确认后再入库")
        self.import_ai_button.clicked.connect(self.open_import_ai)
        row.addWidget(self.import_ai_button)
        self.fill_button = QPushButton("补全缺的中文")
        self.fill_button.setToolTip("没有中文的问题和回答，用快模型翻译补上（已有的中文不动）")
        self.fill_button.clicked.connect(self.fill_missing_chinese)
        row.addWidget(self.fill_button)
        self.dedupe_button = QPushButton("🔍 查重")
        self.dedupe_button.setToolTip("找出意思相近的问题，两两并排，你选保留、合并还是都保留")
        self.dedupe_button.clicked.connect(lambda: self.panel.open_dedupe(self))
        row.addWidget(self.dedupe_button)
        self.render_button = QPushButton("预渲染全部答案")
        self.render_button.setToolTip("用当前音色把每条答案提前合成好存在本机，现场点一下秒播，断网也能读")
        self.render_button.clicked.connect(self.render_all)
        row.addWidget(self.render_button)
        self.render_selected_button = QPushButton("渲染所选")
        self.render_selected_button.setToolTip("只渲染表格里选中的几条（可以 Ctrl / Shift 多选）；已渲染的跳过")
        self.render_selected_button.clicked.connect(self.render_selected)
        row.addWidget(self.render_selected_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_render)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        self.delete_audio_button = QPushButton("删除所选的音频")
        self.delete_audio_button.setToolTip("只删本机渲染好的声音，问答本身保留")
        self.delete_audio_button.clicked.connect(self.delete_selected_audio)
        row.addWidget(self.delete_audio_button)
        self.delete_button = QPushButton("删除所选")
        self.delete_button.clicked.connect(self.delete_selected)
        row.addWidget(self.delete_button)
        self.clear_button = QPushButton("清空问答库")
        self.clear_button.clicked.connect(self.clear_bank)
        row.addWidget(self.clear_button)
        layout.addLayout(row)

        review_row = QHBoxLayout()
        self.export_bank_button = QPushButton("导出完整问答库…")
        self.export_bank_button.clicked.connect(self.export_file)
        review_row.addWidget(self.export_bank_button)
        self.export_template_button = QPushButton("导出格式模板…")
        self.export_template_button.clicked.connect(self.export_template)
        review_row.addWidget(self.export_template_button)
        self.review_all_button = QPushButton("Max 检查全部回答")
        self.review_all_button.clicked.connect(lambda: self.start_answer_review(False))
        review_row.addWidget(self.review_all_button)
        self.review_selected_button = QPushButton("Max 检查所选")
        self.review_selected_button.clicked.connect(lambda: self.start_answer_review(True))
        review_row.addWidget(self.review_selected_button)
        self.review_thinking_check = QCheckBox("思考模式")
        self.review_thinking_check.setChecked(bool(self.settings.get("qa_review_thinking", True)))
        self.review_thinking_check.toggled.connect(lambda checked:self.settings.set("qa_review_thinking",checked))
        review_row.addWidget(self.review_thinking_check)
        self.review_stop_button = QPushButton("停止检查")
        self.review_stop_button.setEnabled(False)
        self.review_stop_button.clicked.connect(lambda:self._review_cancel.set())
        review_row.addWidget(self.review_stop_button)
        review_row.addWidget(QLabel(f"{REVIEW_MODEL} · 结合论文摘录 · 会产生费用"),1)
        layout.addLayout(review_row)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索：问题或回答里的中文、英文关键词")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        # 每一列都能拖动调宽（拖表头之间的竖线），宽度记在本机，下次打开还是这样
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)   # 最后一列（本地音频）补满右边空白
        header.setMinimumSectionSize(40)
        widths = self.settings.get("qa_bank_column_widths", []) or []
        if not (isinstance(widths, list) and len(widths) == len(self.COLUMNS)):
            widths = [60, 70, 480, 480, 150]
        for col, width in enumerate(widths):
            self.table.setColumnWidth(col, max(40, int(width)))
        self._save_layout_timer = QTimer(self)
        self._save_layout_timer.setSingleShot(True)
        self._save_layout_timer.setInterval(600)
        self._save_layout_timer.timeout.connect(self._save_layout)
        header.sectionResized.connect(lambda *_a: self._save_layout_timer.start())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.currentCellChanged.connect(lambda row, _c, _pr, _pc: self._load_detail(row))
        split.addWidget(self.table)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        self.detail_title = QLabel("选中上面一条，在这里看完整的中英问答，可以直接改")
        self.detail_title.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        detail_layout.addWidget(self.detail_title)
        self.q_en_edit = AnswerEditor()
        self.q_zh_edit = AnswerEditor()
        self.a_en_edit = AnswerEditor()
        self.a_zh_edit = AnswerEditor()
        # 左英文、右中文；中间的竖线可以左右拖，问题和回答之间的横线可以上下拖
        self.detail_split = QSplitter(Qt.Horizontal)
        self.detail_split.setChildrenCollapsible(False)
        for q_editor, q_caption, a_editor, a_caption in (
            (self.q_en_edit, "问题 · English", self.a_en_edit, "回答 · English（朗读的就是这段）"),
            (self.q_zh_edit, "问题 · 中文", self.a_zh_edit, "回答 · 中文（只看不读）"),
        ):
            column = QSplitter(Qt.Vertical)
            column.setChildrenCollapsible(False)
            for editor, caption in ((q_editor, q_caption), (a_editor, a_caption)):
                box = QWidget()
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(0, 0, 0, 0)
                box_layout.setSpacing(2)
                label = QLabel(caption)
                label.setStyleSheet(f"color: {TEXT_DIM};")
                box_layout.addWidget(label)
                box_layout.addWidget(editor, 1)
                column.addWidget(box)
                editor.textChanged.connect(self._mark_detail_dirty)
            column.setSizes([90, 220])
            self.detail_split.addWidget(column)
        sizes = self.settings.get("qa_bank_detail_split", []) or []
        self.detail_split.setSizes([int(v) for v in sizes] if isinstance(sizes, list) and len(sizes) == 2 else [640, 640])
        self.detail_split.splitterMoved.connect(lambda *_a: self._save_layout_timer.start())
        detail_layout.addWidget(self.detail_split, 1)
        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(f"color: {ERR_RED};")
        detail_layout.addWidget(self.note_label)
        actions = QHBoxLayout()
        self.en_to_zh_button = QPushButton("按英文回答改中文 →")
        self.en_to_zh_button.clicked.connect(lambda: self._calibrate("en2zh"))
        actions.addWidget(self.en_to_zh_button)
        self.zh_to_en_button = QPushButton("← 按中文回答改英文")
        self.zh_to_en_button.clicked.connect(lambda: self._calibrate("zh2en"))
        actions.addWidget(self.zh_to_en_button)
        actions.addStretch(1)
        self.detail_status = QLabel("")
        self.detail_status.setStyleSheet(f"color: {TEXT_DIM};")
        self.detail_status.setWordWrap(True)
        actions.addWidget(self.detail_status, 2)
        self.to_panel_button = QPushButton("放到助答面板")
        self.to_panel_button.clicked.connect(lambda: self._on_double_click(self._current_row, 0))
        actions.addWidget(self.to_panel_button)
        self.save_detail_button = QPushButton("保存这条")
        self.save_detail_button.setObjectName("primaryButton")
        self.save_detail_button.clicked.connect(self.save_detail)
        actions.addWidget(self.save_detail_button)
        self.save_render_button = QPushButton("保存并渲染")
        self.save_render_button.setToolTip("保存这条，英文回答改过的话马上用当前音色重新渲染")
        self.save_render_button.clicked.connect(self.save_and_render)
        actions.addWidget(self.save_render_button)
        detail_layout.addLayout(actions)

        # 本条的声音：试听 / 渲染 / 换语气 / 删除
        audio_row = QHBoxLayout()
        self.audio_label = QLabel("本地音频：—")
        self.audio_label.setStyleSheet(f"color: {QA_ACCENT};")
        audio_row.addWidget(self.audio_label, 1)
        self.preview_button = QPushButton("▶ 试听")
        self.preview_button.setToolTip("只在你自己的耳机/音箱里放（监听设备），不会传给 Teams / 评委")
        self.preview_button.clicked.connect(lambda: self.preview_entry(self._current_entry()))
        audio_row.addWidget(self.preview_button)
        self.preview_stop_button = QPushButton("■ 停止试听")
        self.preview_stop_button.clicked.connect(self.stop_preview)
        audio_row.addWidget(self.preview_stop_button)
        self.render_one_button = QPushButton("↻ 渲染这条")
        self.render_one_button.setToolTip("没渲染过就渲染；渲染过的再点会重新合成一版（读法会有细微不同）")
        self.render_one_button.clicked.connect(lambda: self.rerender_entry(self._current_entry()))
        audio_row.addWidget(self.render_one_button)
        self.tone_button = QPushButton("🎭 换语气渲染 ▾")
        self.tone_button.setToolTip("只给这一条换个语气重新合成，其他条不受影响；不满意可以「恢复默认版本」")
        self.tone_menu = QMenu(self.tone_button)
        self.tone_menu.aboutToShow.connect(lambda: self._fill_tone_menu(self.tone_menu, self._current_entry()))
        self.tone_button.setMenu(self.tone_menu)
        audio_row.addWidget(self.tone_button)
        self.delete_one_audio_button = QPushButton("🗑 删这条的音频")
        self.delete_one_audio_button.clicked.connect(lambda: self.delete_audio([self._current_entry()]))
        audio_row.addWidget(self.delete_one_audio_button)
        detail_layout.addLayout(audio_row)
        split.addWidget(detail)
        sizes = self.settings.get("qa_bank_split", []) or []
        split.setSizes([int(v) for v in sizes] if isinstance(sizes, list) and len(sizes) == 2 else [440, 420])
        split.splitterMoved.connect(lambda *_a: self._save_layout_timer.start())
        self.main_split = split
        layout.addWidget(split, 1)
        self._preview_stop = threading.Event()
        self._preview_after_render = ""
        self.preview_player_factory = None   # 测试可替换：() -> 有 write/close 的播放器

        hint = QLabel(
            "单击一行在下面看完整中英问答；双击放到助答面板；右键：试听 / 渲染 / 换语气 / 删音频。"
            "表头之间、上下两块之间、中英文之间的分隔线都可以拖动。改了英文回答后旧声音自动删掉，点「保存并渲染」重新生成。\n"
            "「导入问答文件」认本程序的 JSON 或 Q:/A:/ZH: 文本；其他格式用「导入其他 AI 的问答」。问答库只存在本机，不会上传。"
        )
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.render_progress.connect(self._on_render_progress)
        self.render_finished.connect(self._on_render_finished)
        self.fill_progress.connect(lambda text: self.info_label.setText(text))
        self.fill_done.connect(self._on_fill_done)
        self.review_progress.connect(self.info_label.setText)
        self.review_done.connect(self._on_answer_reviewed)
        self.calibrated = _CalibrateRelay()
        self.calibrated.done.connect(self._on_calibrated)
        self._apply_font(manage_font_px(self.settings))
        self._set_detail_enabled(False)

    # ------------------------------------------------------------ 外观
    def _apply_font(self, px: int) -> None:
        self.setStyleSheet(f"QWidget {{ font-size: {px}px; }}")
        for editor in (self.q_en_edit, self.q_zh_edit, self.a_en_edit, self.a_zh_edit):
            editor.setStyleSheet(f"QPlainTextEdit {{ font-size: {px + 1}px; padding: 4px; }}")
        self.table.resizeRowsToContents()

    # ------------------------------------------------------------ 表格
    def refresh(self) -> None:
        entries = self.panel.bank
        keep = self._current_row
        self.table.blockSignals(True)
        self.table.setRowCount(len(entries))
        ready_count = 0
        missing_zh = 0
        for row, entry in enumerate(entries):
            ready = self.panel._entry_ready(entry)
            ready_count += int(ready)
            stars = "★" * entry.priority if entry.priority else ""
            zh = entry.question_zh.strip()
            if not zh or not entry.answer_zh.strip():
                missing_zh += 1
            values = [entry.id, stars, zh or "（缺中文）", entry.question, self._audio_text(entry)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (2, 3):
                    item.setToolTip(f"{entry.question}\n{entry.question_zh}\n\n{entry.answer}\n\n{entry.answer_zh}".strip())
                if col == 2 and not zh:
                    item.setForeground(QColor(TEXT_DIM))
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
        chars = sum(len(e.answer) for e in entries)
        tail = f"，{missing_zh} 条缺中文（点「补全缺的中文」）" if missing_zh else ""
        self.info_label.setText(
            f"共 {len(entries)} 条预案，已预渲染 {ready_count} 条{tail}。全部答案约 {chars} 个字符。"
        )
        self._apply_filter()
        if 0 <= keep < len(entries):
            self.table.setCurrentCell(keep, 2)
            self._load_detail(keep)
        else:
            self._current_row = -1
            self._set_detail_enabled(False)

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        for row, entry in enumerate(self.panel.bank):
            if not query:
                self.table.setRowHidden(row, False)
                continue
            hay = " ".join([entry.id, entry.question, entry.question_zh, entry.answer, entry.answer_zh]).lower()
            self.table.setRowHidden(row, query not in hay)

    def _row_of(self, entry_id: str) -> int:
        for row, entry in enumerate(self.panel.bank):
            if entry.id == entry_id:
                return row
        return -1

    def _on_double_click(self, row: int, _col: int) -> None:
        self.to_panel_button.setText("放到助答面板")
        self.to_panel_button.setStyleSheet("")
        if not 0 <= row < len(self.panel.bank):
            self.detail_status.setText("请先选中一条问答")
            return
        entry = self.panel.bank[row]
        try:
            self.panel.show_bank_entry(entry)
        except Exception as exc:  # noqa: BLE001
            self.detail_status.setText(f"放入助答面板失败：{exc}")
            self.detail_status.setStyleSheet(f"color: {ERR_RED};")
            return
        message = f"✓ {entry.id} 已放入助答面板，回到主窗口即可看到；点「朗读」才会播放。"
        if row == self._current_row and self.save_detail_button.text().endswith("*"):
            message += " 当前编辑尚未保存，本次放入的是已保存版本。"
        self.detail_status.setText(message)
        self.detail_status.setStyleSheet(f"color: {OK_GREEN}; font-weight: bold;")
        self.to_panel_button.setText("✓ 已放入助答面板")
        self.to_panel_button.setStyleSheet(f"QPushButton {{ color: {OK_GREEN}; border: 1px solid {OK_GREEN}; }}")
        self.panel.qa_status.emit(message)

    # ------------------------------------------------------------ 单条详情
    def _set_detail_enabled(self, enabled: bool) -> None:
        for widget in (self.q_en_edit, self.q_zh_edit, self.a_en_edit, self.a_zh_edit,
                       self.save_detail_button, self.en_to_zh_button, self.zh_to_en_button, self.to_panel_button,
                       self.save_render_button, self.preview_button, self.render_one_button, self.tone_button,
                       self.delete_one_audio_button):
            widget.setEnabled(enabled)
        if not enabled:
            self.audio_label.setText("本地音频：—")
        if not enabled:
            self._loading_detail = True
            for editor in (self.q_en_edit, self.q_zh_edit, self.a_en_edit, self.a_zh_edit):
                editor.setText("")
            self._loading_detail = False
            self.note_label.setText("")

    def _load_detail(self, row: int) -> None:
        if row == self._current_row and self.save_detail_button.text().endswith("*"):
            return
        if not (0 <= row < len(self.panel.bank)):
            return
        if self.save_detail_button.text().endswith("*") and 0 <= self._current_row < len(self.panel.bank):
            answer = QMessageBox.question(self, "还没保存", "上一条改过还没保存，要保存吗？",
                                          QMessageBox.Save | QMessageBox.Discard, QMessageBox.Save)
            if answer == QMessageBox.Save:
                self.save_detail(reselect=False)
        self._current_row = row
        entry = self.panel.bank[row]
        self._loading_detail = True
        self.q_en_edit.setText(entry.question)
        self.q_zh_edit.setText(entry.question_zh)
        self.a_en_edit.setText(entry.answer)
        self.a_zh_edit.setText(entry.answer_zh)
        self._loading_detail = False
        self.detail_title.setText(f"{entry.id} · {entry.category or '预案'} · {'★' * entry.priority}")
        self.note_label.setText(f"提醒（不会朗读）：{entry.note}" if entry.note else "")
        self.save_detail_button.setText("保存这条")
        self.detail_status.setText("")
        self.detail_status.setStyleSheet(f"color: {TEXT_DIM};")
        self.to_panel_button.setText("放到助答面板")
        self.to_panel_button.setStyleSheet("")
        self._set_detail_enabled(True)
        self._update_audio_label()

    def _mark_detail_dirty(self) -> None:
        if not self._loading_detail:
            self.save_detail_button.setText("保存这条 *")
            self.to_panel_button.setText("放到助答面板")
            self.to_panel_button.setStyleSheet("")

    def save_detail(self, reselect: bool = True) -> QaEntry | None:
        row = self._current_row
        if not (0 <= row < len(self.panel.bank)):
            return None
        question = self.q_en_edit.text().strip()
        answer = self.a_en_edit.text().strip()
        if not question or not answer:
            QMessageBox.warning(self, "不能保存", "英文问题和英文回答都不能为空（读出来的是英文回答）")
            return None
        entries = list(self.panel.bank)
        old = entries[row]
        entries[row] = QaEntry(**{**asdict(old), "question": question, "answer": answer,
                                  "question_zh": self.q_zh_edit.text().strip(),
                                  "answer_zh": self.a_zh_edit.text().strip()})
        self.save_detail_button.setText("保存这条")
        self.panel.bank_store.save(entries)
        self.panel.reload_bank()
        if reselect:
            self.refresh()
        changed_voice = old.answer.strip() != answer
        if changed_voice:
            # 旧回答的声音不会再用到：删掉，免得占空间，也免得以为「已渲染」
            try:
                self.panel.engine.delete_answer_audio(old.answer)
            except Exception:  # noqa: BLE001
                pass
            if reselect:
                self.refresh()
        self.detail_status.setText("已保存" + ("；英文回答改了，旧声音已删，点「渲染这条」重新生成" if changed_voice else ""))
        self._update_audio_label()
        return entries[row]

    def save_and_render(self) -> None:
        entry = self.save_detail()
        if entry is None:
            return
        if self.panel._entry_ready(entry):
            self.detail_status.setText("已保存，这条的声音已经是最新的")
            return
        self.start_render([entry])

    def _calibrate(self, direction: str) -> None:
        edited = (self.a_zh_edit if direction == "zh2en" else self.a_en_edit).text().strip()
        other = (self.a_en_edit if direction == "zh2en" else self.a_zh_edit).text().strip()
        if not edited:
            return
        run = self.panel._default_calibrator() if not hasattr(self, "calibrator") else self.calibrator
        if run is None:
            self.detail_status.setText("未配置百炼 API，无法校准")
            return
        row = self._current_row
        context = self.q_en_edit.text().strip()
        self.detail_status.setText("正在校准…")
        relay = self.calibrated

        def worker() -> None:
            try:
                result = run(direction, edited, other, context)
            except Exception as exc:  # noqa: BLE001
                relay.done.emit(row, direction, "", " ".join(str(exc).split())[:80])
                return
            relay.done.emit(row, direction, result, "")

        threading.Thread(target=worker, name="bank-calibrate", daemon=True).start()

    def _on_calibrated(self, row: int, direction: str, result: str, error: str) -> None:
        if row != self._current_row:
            return
        if not result:
            self.detail_status.setText(f"校准失败：{error}")
            return
        (self.a_en_edit if direction == "zh2en" else self.a_zh_edit).setText(result)
        self.detail_status.setText("已校准，核对后点「保存这条」")

    def delete_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            return
        answer = QMessageBox.question(self, "删除", f"删除选中的 {len(rows)} 条预案？")
        if answer != QMessageBox.Yes:
            return
        keep = [e for i, e in enumerate(self.panel.bank) if i not in set(rows)]
        self.save_detail_button.setText("保存这条")
        self._current_row = -1
        self.import_entries(keep, "replace")

    # ------------------------------------------------------------ 导入
    def _flush_export_detail(self):
        return not self.save_detail_button.text().endswith("*") or self.save_detail() is not None

    def export_file(self):
        if not self._flush_export_detail():return
        path,_=QFileDialog.getSaveFileName(self,"导出完整问答库","问答库.json","JSON (*.json)")
        if not path:return
        try:
            export_bank(self.panel.bank,path)
            self.info_label.setText(f"已导出 {len(self.panel.bank)} 条完整问答；可再次导入，不含账号或音频：{path}")
        except OSError as exc:QMessageBox.warning(self,"导出失败",str(exc))

    def export_template(self):
        path,_=QFileDialog.getSaveFileName(self,"导出问答格式模板","问答库格式模板.json","JSON (*.json)")
        if not path:return
        try:
            export_bank(template_bank(),path)
            self.info_label.setText("已导出可导入的格式示例。question/answer 是英文，*_zh 是中文；提醒只写 note，不写进 answer。")
        except OSError as exc:QMessageBox.warning(self,"导出失败",str(exc))

    def _backup_bank(self):
        path=self.settings.base_dir / "qa-bank-backups" / f"qa-bank-{time.time_ns()}.json"
        export_bank(self.panel.bank,path)
        return path

    def start_answer_review(self,selected=False):
        if self._review_running:return
        if self._rendering or not self.fill_button.isEnabled():
            self.info_label.setText("请先停止渲染或等补全中文完成，再检查回答")
            return
        if not self._flush_export_detail():return
        entries=list(self._selected_entries() if selected else self.panel.bank)
        if not entries:
            self.info_label.setText("请先选择要检查的问答" if selected else "问答库为空")
            return
        if len({e.id for e in self.panel.bank})!=len(self.panel.bank):
            self.info_label.setText("编号重复，请先处理重复编号")
            return
        self._review_cancel=threading.Event();cancel=self._review_cancel
        try:
            index=ThesisStore(self.settings.base_dir).load_index()
            if self.review_post_factory:
                post=self.review_post_factory(cancel)
            else:
                translator=ContextTranslator(api_key=self.settings.get_api_key(),workspace_id=self.settings.workspace_id,
                    model=REVIEW_MODEL,proxy=self.settings.ws_proxy,proxy_mode=self.settings.proxy_mode,timeout=90)
                post=lambda payload:translator.complete(payload,cancel_event=cancel,deadline_seconds=600)
        except Exception as exc:
            self.info_label.setText(f"无法开始检查：{exc}")
            return
        self._review_snapshot=[asdict(e) for e in self.panel.bank]
        self._review_entries=entries
        thinking=self.review_thinking_check.isChecked()
        self._review_running=True
        for button in (self.review_all_button,self.review_selected_button):button.setEnabled(False)
        self.review_stop_button.setEnabled(True)
        self.info_label.setText(f"准备检查 {len(entries)} 条；"+("结合论文摘录" if index else "未导入论文，事实可能只能标为需核实"))
        done,progress=self.review_done,self.review_progress
        def worker():
            try:
                def context_for(entry):
                    return excerpt_pairs(index.search(entry.question+' '+entry.question_zh,limit=3)) if index else []
                def notify(message):
                    try:progress.emit(message)
                    except RuntimeError:cancel.set()
                reviews=review_answers(entries,thinking=thinking,post=post,context_for=context_for,
                    cancel_event=cancel,on_progress=notify)
                error=""
            except Exception as exc:reviews,error=None," ".join(str(exc).split())[:240]
            try:done.emit(reviews,error)
            except RuntimeError:pass
        threading.Thread(target=worker,name="qa-answer-review",daemon=True).start()

    def _answer_review_dialog(self,entries,reviews):
        return AnswerReviewDialog(self,entries,reviews)

    def _on_answer_reviewed(self,reviews,error):
        self._review_running=False
        for button in (self.review_all_button,self.review_selected_button):button.setEnabled(True)
        self.review_stop_button.setEnabled(False)
        if reviews is None or self._review_cancel.is_set():
            self.info_label.setText(f"检查未应用，问答库未修改：{error or '已停止'}");return
        if [asdict(e) for e in self.panel.bank]!=self._review_snapshot or self.save_detail_button.text().endswith("*"):
            self.info_label.setText("检查期间问答已修改，本次结果未应用，避免覆盖；请重新检查");return
        dialog=self._answer_review_dialog(self._review_entries,reviews)
        if dialog.exec()!=QDialog.Accepted:
            self.info_label.setText("检查完成，未应用修改");return
        chosen=dialog.chosen()
        if not chosen:
            self.info_label.setText("检查完成，没有勾选修正");return
        if [asdict(e) for e in self.panel.bank]!=self._review_snapshot or self.save_detail_button.text().endswith("*"):
            self.info_label.setText("问答已修改，未应用旧检查结果");return
        changes={r['id']:r for r in chosen}
        updated=[QaEntry(**{**asdict(e),'answer':changes[e.id]['answer'],'answer_zh':changes[e.id]['answer_zh']})
                 if e.id in changes else e for e in self.panel.bank]
        try:
            backup=self._backup_bank()
            self.panel.bank_store.save(updated)
        except OSError as exc:
            self.info_label.setText(f"保存失败，原问答未替换：{exc}");return
        old=list(self.panel.bank)
        self.panel.reload_bank()
        remaining={e.answer for e in updated}
        for e in old:
            if e.id in changes and e.answer not in remaining:
                try:self.panel.engine.delete_answer_audio(e.answer)
                except Exception:pass
        self.refresh()
        self.info_label.setText(f"已应用 {len(chosen)} 条修正，修改的英文需重新渲染；原库备份：{backup}")

    def import_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择问答文件", "", "问答库 (*.json *.txt *.md);;所有文件 (*)"
        )
        if not path:
            return
        try:
            entries, skipped = parse_bank_file(path)
        except QaBankError as exc:
            QMessageBox.warning(self, "导入失败", f"{exc}\n\n不是本程序格式的话，请用「导入其他 AI 的问答」让大模型识别。")
            return
        mode = "replace"
        if self.panel.bank:
            box = QMessageBox(self)
            box.setWindowTitle("导入方式")
            box.setText(f"读到 {len(entries)} 条问答。现有问答库有 {len(self.panel.bank)} 条，要怎么处理？")
            replace = box.addButton("替换现有", QMessageBox.AcceptRole)
            append = box.addButton("追加到后面", QMessageBox.ActionRole)
            box.addButton("取消", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is replace:
                mode = "replace"
            elif box.clickedButton() is append:
                mode = "append"
            else:
                return
        try:
            count = self.import_entries(entries, mode, check=True)
        except OSError as exc:
            QMessageBox.warning(self,"导入失败",f"备份或保存失败，未完成导入：{exc}")
            return
        extra = f"，跳过 {skipped} 条缺问题或缺回答的" if skipped else ""
        self.panel.qa_status.emit(f"问答库已导入，现有 {count} 条{extra}")

    def open_import_ai(self) -> None:
        dialog = ImportQaDialog(self, self.panel)
        dialog.exec()
        self.refresh()

    def import_entries(self, entries: list[QaEntry], mode: str = "replace", *, check: bool = False) -> int:
        """写入问答库。check=True（从文件导入时）：导入后查一下有没有和已有问题相近的。"""
        before = {e.id for e in self.panel.bank}
        merged = merge_entries(self.panel.bank, entries) if mode == "append" else list(entries)
        if self.panel.bank and mode == "replace":
            self._backup_bank()
        self.panel.bank_store.save(merged)
        self.panel.reload_bank()
        self.refresh()
        if check and entries:
            self.panel.offer_dedupe(({e.id for e in merged} - before) if mode == "append" else None, self)
        return len(merged)

    def clear_bank(self) -> None:
        if not self.panel.bank:
            return
        answer = QMessageBox.question(self, "清空问答库", f"确定清空全部 {len(self.panel.bank)} 条预案？（已渲染的音频保留在缓存里）")
        if answer != QMessageBox.Yes:
            return
        self._current_row = -1
        self.import_entries([], "replace")

    # ------------------------------------------------------------ 补中文
    def _make_translator(self):
        return ContextTranslator(
            api_key=self.settings.get_api_key(),
            workspace_id=self.settings.workspace_id,
            model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
            proxy=self.settings.ws_proxy,
            proxy_mode=self.settings.proxy_mode,
            timeout=90,
        )

    def fill_missing_chinese(self) -> None:
        todo = [e for e in self.panel.bank if needs_chinese(e)]
        if not todo:
            self.info_label.setText("所有预案都有中文了")
            return
        try:
            translator = self.fill_translator_factory()
        except ApiError as exc:
            QMessageBox.warning(self, "补全中文", str(exc))
            return
        model = str(self.settings.get("llm_model", "") or "qwen3.8-flash")
        glossary = self.settings.glossary_terms()
        self._fill_cancel = threading.Event()
        cancel = self._fill_cancel
        self.fill_button.setEnabled(False)

        def worker() -> None:
            found: dict[str, tuple[str, str]] = {}
            errors = 0
            for start in range(0, len(todo), FILL_ZH_BATCH):
                if cancel.is_set():
                    break
                batch = todo[start:start + FILL_ZH_BATCH]
                self.fill_progress.emit(f"正在补中文：{start}/{len(todo)} 条…")
                try:
                    content = translator.complete(build_fill_zh_payload(batch, model, glossary=glossary),
                                                  cancel_event=cancel, deadline_seconds=180)
                    found.update(parse_fill_zh_response(content))
                except Exception:  # noqa: BLE001 —— 一批失败不连累其他批
                    errors += 1
            self.fill_done.emit(found, f"{errors} 批失败" if errors else "")

        threading.Thread(target=worker, name="bank-fill-zh", daemon=True).start()

    def _on_fill_done(self, found, error: str) -> None:
        self.fill_button.setEnabled(True)
        entries, changed = apply_chinese(self.panel.bank, found or {})
        if changed:
            self.panel.bank_store.save(entries)
            self.panel.reload_bank()
        self.refresh()
        tail = f"（{error}，可以再点一次补剩下的）" if error else ""
        self.info_label.setText(self.info_label.text() + f"  已补中文 {changed} 条{tail}")

    # ------------------------------------------------------------ 预渲染
    def render_all(self) -> None:
        if self._rendering:
            return
        todo = [e for e in self.panel.bank if not self.panel._entry_ready(e)]
        if not todo:
            self.panel.qa_status.emit("全部预案都已预渲染")
            return
        chars = sum(len(e.answer) for e in todo)
        answer = QMessageBox.question(
            self, "预渲染答案",
            f"将用当前音色合成 {len(todo)} 条答案，约 {chars} 个字符，会产生少量语音合成费用。\n"
            "合成期间可以关掉这个窗口，渲染会在后台继续。开始吗？",
        )
        if answer != QMessageBox.Yes:
            return
        self.start_render(todo)

    def start_render(self, todo: list[QaEntry], action=None) -> None:
        """后台逐条渲染。action(entry, cancel) 默认是普通渲染；重做 / 换语气时传别的。"""
        if self._rendering:
            return
        self._render_cancel = threading.Event()
        cancel = self._render_cancel
        self._rendering = True
        self.render_button.setEnabled(False)
        self.render_selected_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        engine = self.panel.engine
        run = action or (lambda entry, cancel_event: engine.render_answer(entry.answer, cancel=cancel_event))

        def worker() -> None:
            done = failed = 0
            for entry in todo:
                if cancel.is_set():
                    break
                self.render_progress.emit(entry.id, "渲染中…")
                try:
                    run(entry, cancel)
                except Exception as exc:
                    if cancel.is_set():
                        break
                    failed += 1
                    self.render_progress.emit(entry.id, f"✗ {' '.join(str(exc).split())[:40]}")
                    continue
                done += 1
                self.render_progress.emit(entry.id, "✓ 已渲染")
            if cancel.is_set():
                summary = f"预渲染已停止：完成 {done} 条，已保存的音频全部保留；再次渲染只补缺失部分"
            else:
                summary = f"预渲染完成：成功 {done} 条" + (f"，失败 {failed} 条（可再点一次重试）" if failed else "")
            self.render_finished.emit(summary)

        threading.Thread(target=worker, name="qa-bank-render", daemon=True).start()

    def stop_render(self) -> None:
        self._render_cancel.set()
        self.stop_button.setText("正在收尾…")
        self.stop_button.setEnabled(False)
        message = "正在停止：保存当前已提交的音频后停下，已完成的音频全部保留。"
        self.detail_status.setText(message)
        self.info_label.setText(message)

    # ------------------------------------------------------------ 单条 / 所选的声音管理
    def _current_entry(self):
        row = self._current_row
        return self.panel.bank[row] if 0 <= row < len(self.panel.bank) else None

    def _selected_entries(self) -> list:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.panel.bank[row] for row in rows if 0 <= row < len(self.panel.bank)]

    def _audio_text(self, entry) -> str:
        engine = self.panel.engine
        try:
            state = engine.answer_state(entry.answer)
        except Exception:  # noqa: BLE001
            state = None
        if not isinstance(state, str):
            state = "exact" if self.panel._entry_ready(entry) else ""
        if not state:
            return "—"
        if state == "partial":
            label = "◐ 已保存部分音频"
            source_of = getattr(engine, "answer_audio_source", None)
            source = source_of(entry.answer) if callable(source_of) else None
            if isinstance(source, dict) and source.get("model"):
                label += f" · {source['model']}"
            return label
        try:
            seconds = engine.answer_audio_seconds(entry.answer)
            seconds = float(seconds) if isinstance(seconds, (int, float)) else 0.0
        except Exception:  # noqa: BLE001
            seconds = 0.0
        length = f" {seconds:.0f}s" if seconds >= 1 else ""
        if state == "other":
            source = engine.answer_audio_source(entry.answer)
            model = source.get("model", "其他模型") if isinstance(source, dict) else "其他模型"
            voice = source.get("voice", "") if isinstance(source, dict) else ""
            return f"✓ 已渲染{length} · {model}" + (f" · 音色…{voice[-8:]}" if voice else "")
        if state == "custom":
            style = ""
            try:
                found = engine.answer_override(entry.answer)
                style = found[1] if found else ""
            except Exception:  # noqa: BLE001
                pass
            return f"✓{length} · {style or '单独版本'}"
        if state == "saved":
            return f"✓{length} · 旧设置"
        return f"✓ 已渲染{length}"

    def _update_audio_label(self) -> None:
        entry = self._current_entry()
        if entry is None:
            self.audio_label.setText("本地音频：—")
            return
        text = self._audio_text(entry)
        if text == "—":
            self.audio_label.setText("本地音频：还没渲染（点「渲染这条」，或「试听」时会先渲染）")
            self.render_one_button.setText("↻ 渲染这条")
        else:
            self.audio_label.setText(f"本地音频：{text}")
            self.render_one_button.setText("↻ 重新渲染一版")

    def render_selected(self) -> None:
        if self._rendering:
            return
        entries = [e for e in self._selected_entries() if not self.panel._entry_ready(e)]
        if not entries:
            self.panel.qa_status.emit("选中的都已渲染（要重做请右键「重新渲染一版」）")
            self.info_label.setText("选中的都已渲染；要重做某条请右键「重新渲染一版」")
            return
        self.start_render(entries)

    def rerender_entry(self, entry) -> None:
        """没渲染 → 渲染；渲染过 → 再合成一版（删掉旧的，只留新的）。"""
        if entry is None or self._rendering:
            return
        engine = self.panel.engine
        if not self.panel._entry_ready(entry):
            self.start_render([entry])
            return
        self.start_render([entry], action=lambda e, cancel: engine.render_answer_override(
            e.answer, instruction=None, style="重做一版", cancel=cancel))

    def render_with_tone(self, entry, instruction: str, style: str) -> None:
        if entry is None or self._rendering:
            return
        engine = self.panel.engine
        self.start_render([entry], action=lambda e, cancel: engine.render_answer_override(
            e.answer, instruction=instruction, style=style, cancel=cancel))

    def restore_default(self, entry) -> None:
        if entry is None:
            return
        self.panel.engine.clear_answer_override(entry.answer)
        self.refresh()
        self.detail_status.setText(f"{entry.id} 已恢复默认版本")

    def delete_audio(self, entries) -> None:
        entries = [e for e in entries if e is not None]
        if not entries or self._rendering:
            return
        if len(entries) > 1:
            answer = QMessageBox.question(self, "删除音频", f"删除选中 {len(entries)} 条已渲染的声音？问答本身保留。")
            if answer != QMessageBox.Yes:
                return
        for entry in entries:
            try:
                self.panel.engine.delete_answer_audio(entry.answer)
            except Exception:  # noqa: BLE001
                pass
        self.refresh()
        self.detail_status.setText(f"已删除 {len(entries)} 条的音频（问答保留，需要时再渲染）")

    def delete_selected_audio(self) -> None:
        self.delete_audio(self._selected_entries())

    def _fill_tone_menu(self, menu, entry) -> None:
        menu.clear()
        if entry is None:
            return
        for label, instruction in INSTRUCTION_PRESETS:
            if not instruction:
                values = self.panel.engine.answer_settings()
                label = "自然答辩（和声音对比 B 一样）"
                instruction = natural_delivery_instruction(str(values.get("tts_model", "")), "en")
            menu.addAction(label, lambda i=instruction, l=label: self.render_with_tone(entry, i, l.split(" · ")[0]))
        try:
            custom = self.panel.engine.answer_override(entry.answer)
        except Exception:  # noqa: BLE001
            custom = None
        if custom:
            menu.addSeparator()
            menu.addAction("↩ 恢复默认版本", lambda: self.restore_default(entry))

    def _table_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if not 0 <= row < len(self.panel.bank):
            return
        if row not in {index.row() for index in self.table.selectedIndexes()}:
            self.table.selectRow(row)
        selected = self._selected_entries()
        entry = self.panel.bank[row]
        menu = QMenu(self.table)
        if len(selected) > 1:
            menu.addAction(f"渲染选中的 {len(selected)} 条", self.render_selected)
            menu.addAction(f"🗑 删除选中 {len(selected)} 条的音频", lambda: self.delete_audio(selected))
            menu.addSeparator()
            menu.addAction(f"删除选中的 {len(selected)} 条预案…", self.delete_selected)
        else:
            menu.addAction("▶ 试听（只在自己耳机里放）", lambda: self.preview_entry(entry))
            menu.addAction("■ 停止试听", self.stop_preview)
            menu.addSeparator()
            ready = self.panel._entry_ready(entry)
            menu.addAction("↻ 重新渲染一版" if ready else "↻ 渲染这条", lambda: self.rerender_entry(entry))
            self._fill_tone_menu(menu.addMenu("🎭 换语气渲染"), entry)
            if ready:
                menu.addAction("🗑 删除这条的音频", lambda: self.delete_audio([entry]))
            menu.addSeparator()
            menu.addAction("放到助答面板（答辩时读给评委）", lambda: self._on_double_click(row, 0))
            menu.addAction("删除这条预案…", self.delete_selected)
        self._exec_menu(menu, self.table.viewport().mapToGlobal(pos))

    def _exec_menu(self, menu, point) -> None:
        menu.exec(point)   # 单独成方法：测试里替换掉

    # ------------------------------------------------------------ 试听（只走监听设备）
    def preview_entry(self, entry) -> None:
        if entry is None:
            return
        pcm = None
        try:
            pcm = self.panel.engine.answer_pcm(entry.answer)
        except Exception:  # noqa: BLE001
            pcm = None
        if not pcm:
            if self._rendering:
                self.detail_status.setText("正在渲染别的条目，等它结束再试听")
                return
            answer = QMessageBox.question(self, "还没渲染",
                                          f"{entry.id} 还没渲染，先用当前音色渲染再试听？（会产生少量合成费用）")
            if answer != QMessageBox.Yes:
                return
            self._preview_after_render = entry.id
            self.start_render([entry])
            return
        self._play_preview(entry, pcm)

    def _play_preview(self, entry, pcm: bytes) -> None:
        from ..audio import MultiOutputPlayer
        from .rehearsal import play_pcm

        self.stop_preview()
        stop = threading.Event()
        self._preview_stop = stop
        factory = self.preview_player_factory or (
            lambda: MultiOutputPlayer([resolve_monitor_device(self.settings)], SPEECH_SAMPLE_RATE))
        self.detail_status.setText(f"正在试听 {entry.id}（{len(pcm) / (SPEECH_SAMPLE_RATE * 2):.0f} 秒，只在你自己耳机里放）")

        def worker() -> None:
            player = None
            try:
                player = factory()
                if hasattr(player, "__enter__"):
                    player.__enter__()
                play_pcm(pcm, player, stop, threading.Event(), lambda _p: None)
            except Exception as exc:  # noqa: BLE001
                self.render_progress.emit("", f"试听失败：{' '.join(str(exc).split())[:60]}")
            finally:
                if player is not None:
                    try:
                        player.close()
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=worker, name="qa-bank-preview", daemon=True).start()

    def stop_preview(self) -> None:
        self._preview_stop.set()

    # ------------------------------------------------------------ 布局记忆
    def _save_layout(self) -> None:
        try:
            self.settings.update({
                "qa_bank_column_widths": [self.table.columnWidth(c) for c in range(len(self.COLUMNS))],
                "qa_bank_split": list(self.main_split.sizes()),
                "qa_bank_detail_split": list(self.detail_split.sizes()),
            })
        except Exception:  # noqa: BLE001
            pass

    def _on_render_progress(self, entry_id: str, state: str) -> None:
        if not entry_id:
            self.detail_status.setText(state)
            return
        row = self._row_of(entry_id)
        if row >= 0:
            cached = self._audio_text(self.panel.bank[row])
            shown = cached if state.startswith("✓") and cached != "—" else state
            if not state.startswith("✓") and cached != "—":
                shown = f"{cached}（{state}）"
            self.table.setItem(row, self.AUDIO_COL, QTableWidgetItem(shown))
        entry = self._current_entry()
        if entry is not None and entry.id == entry_id:
            self.detail_status.setText(f"{entry_id}：{state}")
        self.panel.qa_status.emit(f"预渲染 {entry_id}：{state}")

    def _on_render_finished(self, summary: str) -> None:
        self._rendering = False
        self.render_button.setEnabled(True)
        self.render_selected_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("停止")
        self.refresh()
        self.info_label.setText(self.info_label.text() + "  " + summary)
        self._update_audio_label()
        self.detail_status.setText(summary)
        waiting, self._preview_after_render = self._preview_after_render, ""
        if waiting:
            row = self._row_of(waiting)
            entry = self.panel.bank[row] if row >= 0 else None
            pcm = self.panel.engine.answer_pcm(entry.answer) if entry is not None else None
            if pcm:
                self._play_preview(entry, pcm)
        self.panel.qa_status.emit(summary)
        entry = self.panel._current_entry
        if entry is not None and self.panel._entry_ready(entry):
            label = self.panel.source_label
            label.setText(label.text().replace("未预渲染·朗读时现场合成", "已预渲染·可秒播"))

    def closeEvent(self, event) -> None:
        self._review_cancel.set()
        super().closeEvent(event)

    def reject(self) -> None:
        self._review_cancel.set()
        super().reject()

    def shutdown(self) -> None:
        self._render_cancel.set()
        self._fill_cancel.set()
        self._review_cancel.set()
        self._preview_stop.set()


class _CalibrateRelay(QObject):
    done = Signal(int, str, str, str)


class ImportQaDialog(QDialog):
    """导入其他 AI（ChatGPT、Kimi、豆包……）写的答辩问答：粘贴或选文件 → 大模型识别成
    一问一答（中英都补齐）→ 表格里逐条确认、可改 → 勾选的导入问答库。"""

    parsed = Signal(object, str, bool)   # 条目列表, 错误, 是否全部结束
    progress = Signal(str)

    COLUMNS = ["导入", "问题（英文）", "问题（中文）", "回答（英文）", "回答（中文）"]

    def __init__(self, parent: QWidget, panel: QaPanel) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.panel = panel
        self.settings = panel.settings
        self._cancel = threading.Event()
        self._entries: list[QaEntry] = []
        self.translator_factory = self._make_translator   # 测试可替换
        self.setWindowTitle("导入其他 AI 的问答 · 识别后确认再入库")
        self.resize(1300, 880)
        layout = QVBoxLayout(self)

        tip = QLabel("① 把其他 AI 写的问答粘贴到下面（或点「从文件读取」，支持 txt / md / docx / pdf）；"
                     "② 点「识别」，大模型整理成一问一答，缺的中文或英文会补上；"
                     "③ 在表格里核对、可以直接改，勾选要的，点「导入所选」。")
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(tip)

        split = QSplitter(Qt.Vertical)
        self.source_edit = QPlainTextEdit()
        self.source_edit.setPlaceholderText("在这里粘贴问答材料……格式不限：编号列表、Markdown、中英混杂、只有中文都可以")
        split.addWidget(self.source_edit)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        split.addWidget(self.table)
        split.setSizes([260, 520])
        layout.addWidget(split, 1)

        row = QHBoxLayout()
        self.file_button = QPushButton("从文件读取…")
        self.file_button.clicked.connect(self.load_file)
        row.addWidget(self.file_button)
        self.parse_button = QPushButton("识别")
        self.parse_button.setObjectName("primaryButton")
        self.parse_button.clicked.connect(self.start_parse)
        row.addWidget(self.parse_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self._cancel.set())
        row.addWidget(self.stop_button)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(self.status_label, 1)
        self.import_button = QPushButton("导入所选到问答库")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_selected)
        row.addWidget(self.import_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        row.addWidget(close_button)
        layout.addLayout(row)

        self.parsed.connect(self._on_parsed)
        self.progress.connect(self.status_label.setText)
        px = manage_font_px(self.settings)
        self.setStyleSheet(f"QWidget {{ font-size: {px}px; }}")

    def _make_translator(self):
        return ContextTranslator(
            api_key=self.settings.get_api_key(),
            workspace_id=self.settings.workspace_id,
            model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
            proxy=self.settings.ws_proxy,
            proxy_mode=self.settings.proxy_mode,
            timeout=90,
        )

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择问答材料", "", "问答材料 (*.txt *.md *.docx *.pdf *.json);;所有文件 (*)"
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".json":
                text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
            else:
                text = "\n\n".join(page.text for page in extract_thesis_pages(Path(path)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        self.source_edit.setPlainText(text)
        self.status_label.setText(f"已读入 {Path(path).name}，约 {len(text)} 字。点「识别」")

    def start_parse(self) -> None:
        text = self.source_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("先粘贴问答材料，或者从文件读取")
            return
        # 已经是本程序格式（Q:/A: 或导出的 JSON）的，不用花钱调大模型
        local = self._local_parse(text)
        if local:
            self._entries = []
            self._on_parsed(local, "", True)
            self.status_label.setText(f"按本程序格式直接读出 {len(local)} 条（没有调用大模型），核对后导入")
            return
        try:
            translator = self.translator_factory()
        except ApiError as exc:
            self.status_label.setText(str(exc))
            return
        chunks = split_import_text(text)
        model = str(self.settings.get("llm_model", "") or "qwen3.8-flash")
        glossary = self.settings.glossary_terms()
        self._cancel = threading.Event()
        cancel = self._cancel
        self._entries = []
        self.table.setRowCount(0)
        self.parse_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.import_button.setEnabled(False)

        def worker() -> None:
            start = 1
            failures = 0
            for index, chunk in enumerate(chunks, start=1):
                if cancel.is_set():
                    break
                self.progress.emit(f"{model} 正在识别第 {index}/{len(chunks)} 段…")
                entries = None
                for _attempt in range(2):   # 一段失败自动重试一次
                    try:
                        content = translator.complete(build_import_payload(chunk, model, glossary=glossary),
                                                      cancel_event=cancel, deadline_seconds=240)
                        entries = parse_import_response(content, start=start)
                        break
                    except Exception:  # noqa: BLE001
                        if cancel.is_set():
                            break
                if entries is None:
                    failures += 1
                    continue
                start += len(entries)
                self.parsed.emit(entries, "", False)
            self.parsed.emit([], f"{failures} 段识别失败" if failures else "", True)

        threading.Thread(target=worker, name="import-qa", daemon=True).start()

    @staticmethod
    def _local_parse(text: str) -> list[QaEntry]:
        stripped = text.lstrip()
        try:
            if stripped.startswith(("{", "[")):
                return parse_json_bank(stripped)[0]
            if re.search(r"^\s*(Q|问)\s*[:：]", text, re.MULTILINE) and re.search(r"^\s*(A|答)\s*[:：]", text, re.MULTILINE):
                return parse_text_bank(text)[0]
        except Exception:  # noqa: BLE001
            return []
        return []

    def _on_parsed(self, entries, error: str, finished: bool) -> None:
        existing = {e.question.strip().lower() for e in self.panel.bank}
        for entry in entries or []:
            self._entries.append(entry)
            row = self.table.rowCount()
            self.table.insertRow(row)
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            duplicate = entry.question.strip().lower() in existing
            check.setCheckState(Qt.Unchecked if duplicate else Qt.Checked)
            if duplicate:
                check.setToolTip("问答库里已经有这道题")
                check.setText("已有")
            self.table.setItem(row, 0, check)
            for col, value in enumerate((entry.question, entry.question_zh, entry.answer, entry.answer_zh), start=1):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.resizeRowsToContents()
        if finished:
            self.parse_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.import_button.setEnabled(bool(self._entries))
            tail = f"（{error}）" if error else ""
            self.status_label.setText(f"识别出 {len(self._entries)} 条{tail}。核对、修改后勾选要的，点「导入所选」")

    def selected_entries(self) -> list[QaEntry]:
        chosen: list[QaEntry] = []
        for row, entry in enumerate(self._entries):
            check = self.table.item(row, 0)
            if check is None or check.checkState() != Qt.Checked:
                continue
            cells = [(self.table.item(row, col).text() if self.table.item(row, col) else "").strip()
                     for col in range(1, 5)]
            if not cells[0] or not cells[2]:
                continue
            chosen.append(QaEntry(**{**asdict(entry), "question": cells[0], "question_zh": cells[1],
                                     "answer": cells[2], "answer_zh": cells[3]}))
        return chosen

    def import_selected(self) -> int:
        chosen = self.selected_entries()
        if not chosen:
            self.status_label.setText("没有勾选任何一条（英文问题和英文回答都要有）")
            return 0
        start = next_id_start([e.id for e in self.panel.bank], "I")
        renumbered = [QaEntry(**{**asdict(e), "id": f"I{start + i}"}) for i, e in enumerate(chosen)]
        merged = merge_entries(self.panel.bank, renumbered)
        self.panel.bank_store.save(merged)
        self.panel.reload_bank()
        if self.panel._bank_dialog is not None:
            self.panel._bank_dialog.refresh()
        self.panel.offer_dedupe({e.id for e in renumbered}, self)
        self.status_label.setText(f"已导入 {len(renumbered)} 条（编号 I 开头），问答库现有 {len(merged)} 条")
        self.import_button.setEnabled(False)
        return len(renumbered)

    def reject(self) -> None:
        self._cancel.set()
        super().reject()


class StudySectionEditor(QFrame):
    """「论文理解」里的一节：左英文、右中文，都能改；改了哪边就能让另一边跟着改。"""

    def __init__(self, index: int, section: StudySection, on_calibrate, on_changed,
                 font_px: int = 16) -> None:
        super().__init__()
        self.index = index
        self.setStyleSheet("QFrame { border: 1px solid #232a3a; border-radius: 8px; }"
                           "QLabel { border: none; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        head = QHBoxLayout()
        self.title_zh = QLineEdit(section.title_zh)
        self.title_zh.setPlaceholderText("中文标题")
        self.title_en = QLineEdit(section.title_en)
        self.title_en.setPlaceholderText("English title")
        head.addWidget(self.title_en, 1)
        head.addWidget(self.title_zh, 1)
        layout.addLayout(head)
        boxes = QHBoxLayout()
        self.en = AnswerEditor()
        self.en.setText(section.en)
        self.en.setMinimumHeight(150)
        self.zh = AnswerEditor()
        self.zh.setText(section.zh)
        self.zh.setMinimumHeight(150)
        for box in (self.en, self.zh):
            # 字号跟助答面板的 A-/A+ 走，大屏上一样看得清
            box.setStyleSheet(f"QPlainTextEdit {{ font-size: {font_px}px; padding: 6px; }}")
        boxes.addWidget(self.en, 1)
        boxes.addWidget(self.zh, 1)
        layout.addLayout(boxes)
        actions = QHBoxLayout()
        self.en_to_zh = QPushButton("按英文改中文 →")
        self.en_to_zh.setToolTip("你改了左边英文：让大模型把右边中文改成一致的意思")
        self.en_to_zh.clicked.connect(lambda: on_calibrate(self.index, "en2zh"))
        self.zh_to_en = QPushButton("← 按中文改英文")
        self.zh_to_en.setToolTip("你改了右边中文：让大模型把左边英文改成一致的说法")
        self.zh_to_en.clicked.connect(lambda: on_calibrate(self.index, "zh2en"))
        actions.addWidget(self.en_to_zh)
        actions.addStretch(1)
        actions.addWidget(self.zh_to_en)
        layout.addLayout(actions)
        for widget in (self.en, self.zh):
            widget.textChanged.connect(on_changed)
        for widget in (self.title_en, self.title_zh):
            widget.textEdited.connect(lambda _text: on_changed())

    def section(self) -> StudySection:
        return StudySection(self.title_zh.text().strip(), self.title_en.text().strip(),
                            self.zh.text().strip(), self.en.text().strip())

    def set_busy(self, busy: bool) -> None:
        self.en_to_zh.setEnabled(not busy)
        self.zh_to_en.setEnabled(not busy)


class StudyDialog(QDialog):
    """论文预习：用强模型通读论文 → 中英「论文理解」（可改、可互相校准）→ 批量生成预案问答。

    非模态：通读一篇硕士论文、打开深度思考可能要好几分钟，期间主界面照常能用。
    关窗口只是隐藏，后台任务继续；退出程序时才停。
    """

    study_done = Signal(object, str)      # StudyNotes 或 None, 错误
    qa_progress = Signal(str)
    qa_done = Signal(int, str)            # 新增条数, 错误
    calibrated = Signal(int, str, str, str)  # 小节序号, 方向, 结果, 错误

    def __init__(self, parent, settings: DefenseSettings, qa_panel: QaPanel) -> None:
        super().__init__(parent)
        self.settings = settings
        self.qa_panel = qa_panel
        self.thesis_store = ThesisStore(settings.base_dir)
        self.study_store = StudyStore(settings.base_dir)
        self.notes: StudyNotes | None = self.study_store.load()
        self.editors: list[StudySectionEditor] = []
        self._dirty = False
        self._cancel = threading.Event()
        # 测试可替换：返回一个有 complete()/chat() 的对象（ContextTranslator）
        self.translator_factory = self._make_translator
        enable_window_buttons(self)
        self.setWindowTitle("论文预习 · 答辩前用强模型吃透论文")
        self.resize(1280, 860)
        self._build_ui()
        self._apply_font(manage_font_px(settings), rebuild=False)
        self.study_done.connect(self._on_study_done)
        self.qa_progress.connect(self._set_status)
        self.qa_done.connect(self._on_qa_done)
        self.calibrated.connect(self._on_calibrated)
        self._rebuild_editors()

    # ------------------------------------------------------------ 界面
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("预习模型"))
        self.model_combo = NoWheelComboBox()
        fill_model_combo(self.model_combo, QA_MODELS, str(self.settings.get("study_model", DEFAULT_STUDY_MODEL)), RECOMMENDED_STUDY)
        self.model_combo.setMinimumWidth(260)
        row.addWidget(self.model_combo, 1)
        self.thinking_check = QCheckBox("深度思考（更准，更慢）")
        self.thinking_check.setChecked(bool(self.settings.get("study_thinking", True)))
        row.addWidget(self.thinking_check)
        self.study_button = QPushButton("开始预习（通读全文）")
        self.study_button.clicked.connect(self.start_study)
        row.addWidget(self.study_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.font_down_button, self.font_up_button = font_buttons(self, self._apply_font)
        row.addWidget(self.font_down_button)
        row.addWidget(self.font_up_button)
        layout.addLayout(row)

        qa_row = QHBoxLayout()
        qa_row.addWidget(QLabel("生成预案问答"))
        self.qa_count_spin = NoWheelSpinBox()
        self.qa_count_spin.setRange(1, MAX_QA_COUNT)
        self.qa_count_spin.setValue(int(self.settings.get("study_qa_count", DEFAULT_QA_COUNT) or DEFAULT_QA_COUNT))
        qa_row.addWidget(self.qa_count_spin)
        qa_row.addWidget(QLabel("题"))
        self.qa_button = QPushButton("生成并加入问答库")
        self.qa_button.setToolTip("用上面的模型，根据论文全文和下面这份（你校对过的）理解，生成问库里还没有的问题和中英回答")
        self.qa_button.clicked.connect(self.start_qa)
        qa_row.addWidget(self.qa_button)
        self.import_ai_button = QPushButton("导入其他 AI 的问答…")
        self.import_ai_button.setToolTip("把 ChatGPT / Kimi / 豆包等写的问答粘贴进来，大模型识别成一问一答，你确认后入库")
        self.import_ai_button.clicked.connect(self.open_import_ai)
        qa_row.addWidget(self.import_ai_button)
        self.bank_button = QPushButton("查看 / 编辑问答库")
        self.bank_button.setToolTip("逐条看中英文问题和回答，可以改、删、补中文、预渲染")
        self.bank_button.clicked.connect(self.qa_panel.open_bank_dialog)
        qa_row.addWidget(self.bank_button)
        qa_row.addStretch(1)
        self.save_button = QPushButton("保存修改")
        self.save_button.clicked.connect(self.save_notes)
        qa_row.addWidget(self.save_button)
        layout.addLayout(qa_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.status_label)

        hint = QLabel("左边英文、右边中文，都可以直接改。改完一边点下面的按钮，让另一边跟着改成一致的意思；"
                      "最后点「保存修改」。现场助答会把这份理解当作事实依据。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(hint)

        self.sections_host = QWidget()
        self.sections_layout = QVBoxLayout(self.sections_host)
        self.sections_layout.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sections_host)
        layout.addWidget(scroll, 1)

    def refresh(self) -> None:
        data = self.thesis_store.load()
        if data.get("full_text"):
            pages = f"{data.get('page_count')} 页，" if data.get("page_count") else ""
            names = "、".join(Path(p).name for p in str(data.get("source_path", "")).split("; ") if p)
            thesis = f"论文：{names}（{pages}{len(data.get('chunks', []))} 段，导入于 {data.get('imported_at', '')}）"
        else:
            thesis = "还没有导入论文全文：先在主界面点「导入论文 / PPT」选论文 PDF 或 Word。"
        if self.notes is not None:
            study = f"论文理解：{len(self.notes.sections)} 节，{self.notes.model} 生成于 {self.notes.created_at}" + (
                f"，最后修改 {self.notes.edited_at}" if self.notes.edited_at else "")
        else:
            study = "论文理解：还没有预习。"
        self.info_label.setText(f"{thesis}\n{study}\n问答库现有 {len(self.qa_panel.bank)} 条。")
        self.bank_button.setText(f"查看 / 编辑问答库（{len(self.qa_panel.bank)}）")

    def _rebuild_editors(self) -> None:
        while self.sections_layout.count():
            item = self.sections_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.editors = []
        for index, section in enumerate(self.notes.sections if self.notes else []):
            editor = StudySectionEditor(index, section, self.calibrate_section, self._mark_dirty,
                                        font_px=manage_font_px(self.settings) + 1)
            self.editors.append(editor)
            self.sections_layout.addWidget(editor)
        self.sections_layout.addStretch(1)
        self._dirty = False
        self.refresh()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.save_button.setText("保存修改 *")

    def _apply_font(self, px: int, rebuild: bool = True) -> None:
        self.setStyleSheet(f"QWidget {{ font-size: {px}px; }}")
        for editor in self.editors:
            for box in (editor.en, editor.zh):
                box.setStyleSheet(f"QPlainTextEdit {{ font-size: {px + 1}px; padding: 6px; }}")

    def open_import_ai(self) -> None:
        ImportQaDialog(self, self.qa_panel).exec()
        self.refresh()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        for button in (self.study_button, self.qa_button, self.save_button):
            button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if busy:
            self._bank_ids_before = {e.id for e in self.qa_panel.bank}

    # ------------------------------------------------------------ 模型
    def _make_translator(self, model: str):
        return ContextTranslator(
            api_key=self.settings.get_api_key(),
            workspace_id=self.settings.workspace_id,
            model=model,
            proxy=self.settings.ws_proxy,
            proxy_mode=self.settings.proxy_mode,
            timeout=120,
        )

    def _selected_model(self) -> str:
        model = str(self.model_combo.currentData() or DEFAULT_STUDY_MODEL)
        self.settings.update({"study_model": model, "study_thinking": self.thinking_check.isChecked(),
                              "study_qa_count": self.qa_count_spin.value()})
        return model

    # ------------------------------------------------------------ 预习
    def start_study(self) -> None:
        full_text = self.thesis_store.full_text()
        if not full_text.strip():
            self._set_status("还没有导入论文全文：先在主界面点「导入论文 / PPT」")
            return
        if self._dirty or (self.notes and self.notes.edited_at):
            answer = QMessageBox.question(self, "重新预习", "重新预习会覆盖现在这份（含你改过的）论文理解，继续吗？")
            if answer != QMessageBox.Yes:
                return
        model = self._selected_model()
        thinking = self.thinking_check.isChecked()
        try:
            translator = self.translator_factory(model)
        except ApiError as exc:
            self._set_status(str(exc))
            return
        self._cancel = threading.Event()
        cancel = self._cancel
        self._set_busy(True)
        self._set_status(f"{model} 正在通读论文{'（深度思考）' if thinking else ''}，通常需要 1~5 分钟，可以先关掉这个窗口…")
        glossary = self.settings.glossary_terms()
        last = [0]

        def on_reasoning(chars: int) -> None:
            if chars - last[0] >= 2000:
                last[0] = chars
                self.qa_progress.emit(f"{model} 思考中…（已推理约 {chars} 字）")

        def worker() -> None:
            try:
                payload = build_study_payload(full_text, model, thinking=thinking, glossary=glossary)
                content = run_payload(translator, payload, thinking=thinking, cancel_event=cancel,
                                      on_reasoning=on_reasoning)
                notes = parse_study_response(content, model=model)
            except Exception as exc:  # noqa: BLE001
                self.study_done.emit(None, " ".join(str(exc).split())[:200])
                return
            self.study_done.emit(notes, "")

        threading.Thread(target=worker, name="thesis-study", daemon=True).start()

    def _on_study_done(self, notes, error: str) -> None:
        self._set_busy(False)
        if notes is None:
            self._set_status(f"预习失败：{error}" if "已取消" not in error else "预习已停止")
            return
        self.notes = notes
        self.study_store.save(notes)
        self.qa_panel.reload_thesis()
        self._rebuild_editors()
        self._set_status(f"预习完成：{len(notes.sections)} 节。请逐节核对，改完点「保存修改」；"
                         "然后可以「生成并加入问答库」。")

    def save_notes(self) -> None:
        if not self.editors:
            self._set_status("还没有论文理解可保存")
            return
        sections = [editor.section() for editor in self.editors]
        base = self.notes or StudyNotes()
        self.notes = StudyNotes(sections, base.model, base.created_at, time.strftime("%Y-%m-%d %H:%M:%S"))
        self.study_store.save(self.notes)
        self.qa_panel.reload_thesis()
        self._dirty = False
        self.save_button.setText("保存修改")
        self.refresh()
        self._set_status("已保存。现场助答从现在起按这份理解作答。")

    # ------------------------------------------------------------ 中英校准
    def calibrate_section(self, index: int, direction: str) -> None:
        if not (0 <= index < len(self.editors)):
            return
        editor = self.editors[index]
        section = editor.section()
        edited, other = (section.zh, section.en) if direction == "zh2en" else (section.en, section.zh)
        if not edited:
            return
        run = self.qa_panel._default_calibrator() if not hasattr(self, "calibrator") else self.calibrator
        if run is None:
            self._set_status("未配置百炼 API，无法校准")
            return
        editor.set_busy(True)
        self._set_status(f"正在校准「{section.title_zh or section.title_en}」…")
        context = section.title_en or section.title_zh

        def worker() -> None:
            try:
                result = run(direction, edited, other, context)
            except Exception as exc:  # noqa: BLE001
                self.calibrated.emit(index, direction, "", " ".join(str(exc).split())[:120])
                return
            self.calibrated.emit(index, direction, result, "")

        threading.Thread(target=worker, name="study-calibrate", daemon=True).start()

    def _on_calibrated(self, index: int, direction: str, result: str, error: str) -> None:
        if not (0 <= index < len(self.editors)):
            return
        editor = self.editors[index]
        editor.set_busy(False)
        if not result:
            self._set_status(f"校准失败：{error}")
            return
        (editor.en if direction == "zh2en" else editor.zh).setText(result)
        self._set_status("已校准。核对无误后记得点「保存修改」。")

    # ------------------------------------------------------------ 批量生成问答
    def start_qa(self) -> None:
        full_text = self.thesis_store.full_text()
        if not full_text.strip():
            self._set_status("还没有导入论文全文：先在主界面点「导入论文 / PPT」")
            return
        if self._dirty:
            self.save_notes()
        model = self._selected_model()
        thinking = self.thinking_check.isChecked()
        try:
            translator = self.translator_factory(model)
        except ApiError as exc:
            self._set_status(str(exc))
            return
        total = self.qa_count_spin.value()
        notes = self.notes
        glossary = self.settings.glossary_terms()
        panel = self.qa_panel
        self._cancel = threading.Event()
        cancel = self._cancel
        self._set_busy(True)

        def worker() -> None:
            added = 0
            failures = 0          # 累计失败的批
            streak = 0            # 连续失败的批：连着坏 3 批多半是网络 / 账号问题，别再花钱了
            last_error = ""
            while added < total and not cancel.is_set():
                batch = min(QA_BATCH, total - added)  # 分批：一次太多会被输出长度截断
                self.qa_progress.emit(
                    f"{model} 正在生成预案问答（{added}/{total}）…" + (f" 已跳过 {failures} 批出错的" if failures else ""))
                entries = None
                for attempt in range(2):   # 一批出错（比如 JSON 坏了）自动重试一次
                    try:
                        existing = panel.bank_store.load()
                        payload = build_qa_payload(full_text, notes, [e.question for e in existing], model,
                                                   count=batch, thinking=thinking, glossary=glossary)
                        content = run_payload(translator, payload, thinking=thinking, cancel_event=cancel)
                        start = next_id_start([e.id for e in existing], "S")
                        entries = parse_qa_response(content, id_prefix="S", start=start)[:batch]
                        panel.bank_store.save(existing + entries)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = " ".join(str(exc).split())[:160]
                        if cancel.is_set():
                            break
                        if attempt == 0:
                            self.qa_progress.emit(f"这一批出错了（{last_error[:60]}），自动重试一次…")
                if cancel.is_set():
                    break
                if entries is None:
                    failures += 1
                    streak += 1
                    if streak >= 3:
                        self.qa_done.emit(added, f"连续 3 批失败，已停下：{last_error}")
                        return
                    continue
                streak = 0
                added += len(entries)
            error = f"其中 {failures} 批出错已跳过（最后一次：{last_error}）" if failures else ""
            self.qa_done.emit(added, "已取消" if cancel.is_set() and not error else error)

        threading.Thread(target=worker, name="study-qa", daemon=True).start()

    def _on_qa_done(self, added: int, error: str) -> None:
        self._set_busy(False)
        self.qa_panel.reload_bank()
        if self.qa_panel._bank_dialog is not None:
            self.qa_panel._bank_dialog.refresh()
        self.refresh()
        before = getattr(self, "_bank_ids_before", None)
        if added and before is not None:
            self.qa_panel.offer_dedupe({e.id for e in self.qa_panel.bank} - before, self)
        if error and "已取消" not in error:
            self._set_status(f"已加入 {added} 条。{error}")
        else:
            self._set_status(f"已向问答库加入 {added} 条预案（编号 S 开头）。点主界面「问答库」可逐条查看、预渲染。")

    def stop(self) -> None:
        self._cancel.set()
        self._set_status("正在停止…")

    def shutdown(self) -> None:
        self._cancel.set()


class VoiceCompareDialog(QDialog):
    """Same English text, selectable bound voices, monitor-only listening."""
    finished_probe = Signal(str)
    playback_progress = Signal(object, str, float)
    voice_applied = Signal()

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
            preferred = str(settings.get("prerender_voice_id", "") or current) if not self.voices else current
            selected = next((i for i, e in enumerate(entries) if e["voice_id"] == preferred), 0)
            combo.setCurrentIndex(selected)
            self.voices.append(combo)
            row.addWidget(combo, 1)
            layout.addLayout(row)
        self.result = QLabel("Qwen 自然模式使用复刻原声，不额外加语气指令。对比不同模型时，请保持同一句文本、语速和音调 1.0。")
        self.result.setWordWrap(True)
        layout.addWidget(self.result)
        self.b_style = NoWheelComboBox()
        for label, value in (("B：自然答辩（Qwen 使用复刻原声）", "instruction"),
                             ("B：仅在换行处停顿 250 ms", "pause"),
                             ("B：原始语气（比较不同模型）", "original")):
            self.b_style.addItem(label, value)
        layout.addWidget(self.b_style)
        self.parameters_label = QLabel("")
        self.parameters_label.setWordWrap(True)
        layout.addWidget(self.parameters_label)
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
        self.apply_a_button = QPushButton("使用 A 原声：应用到现场和讲稿")
        self.apply_a_button.setToolTip("复制 A 的模型和复刻音色，语速音调 1.0、不加语气指令；不会删除或自动重渲染已有音频")
        self.apply_a_button.clicked.connect(self._apply_a_voice)
        layout.addWidget(self.apply_a_button)
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
        for control in (self.a_button, self.b_button, self.apply_a_button, self.text_edit, self.b_style, *self.voices):
            control.setEnabled(not busy)
        for index, button in enumerate(self.replay_buttons):
            button.setEnabled(not busy and index in self._cached_audio)

    def _apply_a_voice(self) -> None:
        if self._is_busy:
            return
        entry = self.voices[0].currentData()
        if not isinstance(entry, dict) or not entry.get("voice_id") or not entry.get("target_model"):
            self.result.setText("请先选择 A 的音色")
            return
        self.settings.update({"tts_model": entry["target_model"], "tts_voice_id": entry["voice_id"],
            "prerender_tts_model": entry["target_model"], "prerender_voice_id": entry["voice_id"],
            "tts_instruction": "", "tts_rate": 1.0, "tts_pitch": 1.0,
            "prerender_instruction": "none", "backup_tts_enabled": False})
        self.voice_applied.emit()
        self.result.setText("已使用 A 的原始音色：现场与讲稿一致、不加语气指令。已有音频仍保留；变音段落请在讲稿台点「原声重做本段」。")

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
        self._busy(True)
        self.result.setText("正在重听缓存音频，不调用云端…")
        def worker():
            from ..audio import MultiOutputPlayer
            try:
                monitor = resolve_monitor_device(self.settings)
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
        if natural and style == "instruction":
            # 记下 B 用的音色：讲稿台的音色下拉会标出来，方便选成同一个
            try:
                self.settings.set("voice_ab_last_b", {"voice_id": voice, "model": model})
            except Exception:  # noqa: BLE001
                pass
        values = {
            "speech_mode": "natural", "tts_rate": 1.0, "tts_pitch": 1.0,
            "tts_instruction": natural_delivery_instruction(model) if style == "instruction" else "",
            "tts_enable_ssml": style == "pause",
        }
        self.parameters_label.setText(
            f"本次 {'B' if natural else 'A'}：{model} · 音色 …{voice[-10:]} · 语速 1.0 · 音调 1.0 · "
            f"语气指令：{values['tts_instruction'] or '无（复刻原声）'}"
        )
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
    # 批量结果用 object 承载一个只读 dict：worker 线程构造后不再改动，
    # 排队信号送回 GUI 线程读取，不违反「worker 不碰 Qt 控件」的规矩。
    batch_result = Signal(object)
    batch_done = Signal(int, int)

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
        self._batch_cancel = threading.Event()
        self._oss_probed = False
        self._build_ui()
        self.clone_finished.connect(self._on_clone_finished)
        self.clone_progress.connect(self._on_clone_progress)
        self.oss_probe_done.connect(self._on_oss_probe_done)
        self.sample_checked.connect(self._on_sample_checked, Qt.QueuedConnection)
        self.batch_result.connect(self._on_batch_result, Qt.QueuedConnection)
        self.batch_done.connect(self._on_batch_done, Qt.QueuedConnection)
        self._update_channel_ui()
        self._update_batch_plan()
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
        self.clone_language_combo = NoWheelComboBox()
        for code, blurb in CLONE_LANGUAGES:
            self.clone_language_combo.addItem(blurb, code)
        saved_language = str(self.settings.get("clone_language", "en") or "en")
        self.clone_language_combo.setCurrentIndex(max(self.clone_language_combo.findData(saved_language), 0))
        self.clone_language_combo.setToolTip(
            "样音语种要和你之后要合成的语种一致。答辩说英文就录一段你自己念的英文（带口音没关系），"
            "不要用中文样音去合成英文。"
        )
        options.addRow("样音语种", self.clone_language_combo)

        self.sample_length_spin = NoWheelDoubleSpinBox()
        self.sample_length_spin.setDecimals(3)
        self.sample_length_spin.setRange(self._minimum_sample_seconds(), MAX_SAMPLE_SECONDS)
        self.sample_length_spin.setValue(RECOMMENDED_SAMPLE_SECONDS)
        self.sample_length_spin.setSuffix(" 秒")
        self.sample_length_spin.setToolTip(
            f"官方最佳区间 10~20 秒，默认取前 {RECOMMENDED_SAMPLE_SECONDS:g} 秒。"
            "更长不会更像，反而可能降低稳定性；可手动调整，使用音频开头这一段。"
        )
        options.addRow("参考音频时长", self.sample_length_spin)
        self.preprocess_check = QCheckBox("启用预处理（降噪、音频增强、音量规整）")
        self.preprocess_check.setChecked(False)
        self.preprocess_check.setEnabled(self.tts_model in VOICE_PREPROCESS_MODELS)
        options.addRow(self.preprocess_check)
        self.volume_normalization_check = QCheckBox("启用音量归一化")
        self.volume_normalization_check.setChecked(False)
        self.volume_normalization_check.setEnabled(self.channel == "legacy")
        options.addRow(self.volume_normalization_check)
        option_note = QLabel(
            "录音棚或安静环境默认关闭两项处理，保留原始音色。灰色选项表示当前模型不支持；预处理本身也可能调整音量。\n"
            "建议录 3~5 条候选各复刻一个音色，再用同一句英文盲听对比挑最像的那个。"
        )
        option_note.setWordWrap(True)
        options.addRow(option_note)
        layout.addWidget(self.sample_options)

        # ---------------- 批量复刻：一条录音 → 多个音色 ----------------
        self.batch_group = QGroupBox("批量复刻（同一条录音，一次建多个音色）")
        self.batch_group.setCheckable(True)
        self.batch_group.setChecked(False)
        batch_layout = QVBoxLayout(self.batch_group)
        batch_hint = QLabel(
            "做盲听对比时，要比的是「同一条录音在不同模型/时长下的表现」。"
            "一个一个手点很容易中途改错一个选项，对比里就混进了第二个变量。\n"
            "勾好下面的组合，软件会用完全相同的条件逐个提交；"
            "同一时长的多个模型共用一次转码和一次 OSS 上传。"
        )
        batch_hint.setWordWrap(True)
        batch_hint.setStyleSheet(f"color: {TEXT_DIM};")
        batch_layout.addWidget(batch_hint)

        batch_layout.addWidget(QLabel("要复刻到哪些模型（音色与模型绑定，不能跨模型复用）"))
        self.batch_model_list = QListWidget()
        self.batch_model_list.setMaximumHeight(150)
        for model, blurb in TTS_MODELS:
            if not tts_supports(model, "clone"):
                continue
            item = QListWidgetItem(f"{model} · {blurb}")
            item.setData(Qt.UserRole, model)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if model == self.tts_model else Qt.Unchecked)
            self.batch_model_list.addItem(item)
        self.batch_model_list.itemChanged.connect(lambda _item: self._update_batch_plan())
        batch_layout.addWidget(self.batch_model_list)

        preset_row = QHBoxLayout()
        recommend_button = QPushButton("勾选对比推荐组合")
        recommend_button.setToolTip(
            "勾上 qwen-audio-3.0-tts-plus 与 cosyvoice-v3.5-plus，"
            "配合下面两个时长，正好是一组能分离「模型」和「时长」两个变量的对比。"
        )
        recommend_button.clicked.connect(self._check_recommended_models)
        preset_row.addWidget(recommend_button)
        clear_button = QPushButton("全部取消")
        clear_button.clicked.connect(lambda: self._set_all_batch_models(False))
        preset_row.addWidget(clear_button)
        preset_row.addStretch(1)
        batch_layout.addLayout(preset_row)

        batch_layout.addWidget(QLabel("每个模型各用哪些参考时长复刻一份"))
        self.batch_duration_recommended = QCheckBox("18 秒（官方推荐区间 10~20 秒）")
        self.batch_duration_recommended.setChecked(True)
        self.batch_duration_recommended.stateChanged.connect(lambda _s: self._update_batch_plan())
        batch_layout.addWidget(self.batch_duration_recommended)
        self.batch_duration_full = QCheckBox("全长（最多 30 秒）")
        self.batch_duration_full.setChecked(True)
        self.batch_duration_full.setToolTip(
            "更长不一定更像 —— 官方最佳区间是 10~20 秒。勾上它就是为了用耳朵验证这件事。"
        )
        self.batch_duration_full.stateChanged.connect(lambda _s: self._update_batch_plan())
        batch_layout.addWidget(self.batch_duration_full)
        self.batch_duration_custom = QCheckBox("使用上面「参考音频时长」里的值")
        self.batch_duration_custom.stateChanged.connect(lambda _s: self._update_batch_plan())
        batch_layout.addWidget(self.batch_duration_custom)
        self.sample_length_spin.valueChanged.connect(lambda _v: self._update_batch_plan())

        self.batch_plan_label = QLabel("")
        self.batch_plan_label.setWordWrap(True)
        batch_layout.addWidget(self.batch_plan_label)

        self.batch_result_table = QTableWidget(0, 4)
        self.batch_result_table.setHorizontalHeaderLabels(["模型", "时长", "结果", "voice_id"])
        self.batch_result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.batch_result_table.verticalHeader().setVisible(False)
        self.batch_result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_result_table.setMaximumHeight(160)
        self.batch_result_table.setVisible(False)
        batch_layout.addWidget(self.batch_result_table)

        self.batch_group.toggled.connect(lambda _on: self._update_batch_plan())
        layout.addWidget(self.batch_group)

        self.clone_button = QPushButton("创建克隆音色")
        self.clone_button.setEnabled(False)
        self.clone_button.setMinimumHeight(38)
        self.clone_button.clicked.connect(self._start_clone)
        outer.addWidget(self.clone_button)

        self.batch_button = QPushButton("批量创建")
        self.batch_button.setEnabled(False)
        self.batch_button.setMinimumHeight(38)
        self.batch_button.setVisible(False)
        self.batch_button.clicked.connect(self._start_batch_clone)
        outer.addWidget(self.batch_button)

        self.batch_cancel_button = QPushButton("停止批量（当前这个建完后停）")
        self.batch_cancel_button.setVisible(False)
        self.batch_cancel_button.clicked.connect(self._cancel_batch)
        outer.addWidget(self.batch_cancel_button)

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
        return minimum_sample_seconds(self.tts_model)

    # ---------------------------------------------------------------- state
    def _clear_sample(self, message: str) -> None:
        self._sample_generation += 1
        self._sample_path = None
        self._sample_seconds = 0.0
        self.clone_button.setEnabled(False)
        self.sample_label.setText(message)
        self.sample_label.setStyleSheet(f"color: {TEXT_DIM};")
        self._update_batch_plan()

    def _set_sample(self, path: Path, seconds: float) -> None:
        self._sample_path = path
        self._sample_seconds = seconds
        # 上限仍是音频实际长度（不超过服务端的 30 秒），但默认值取官方最佳区间的
        # 18 秒，而不是「能用多长用多长」—— 更长不会更像。
        usable = min(MAX_SAMPLE_SECONDS, seconds)
        self.sample_length_spin.setMaximum(usable)
        self.sample_length_spin.setValue(min(RECOMMENDED_SAMPLE_SECONDS, usable))
        self.sample_label.setText(
            f"样音已就绪：{seconds:.3f} 秒 · 默认使用前 {self.sample_length_spin.value():.3f} 秒"
        )
        self.sample_label.setStyleSheet(f"color: {OK_GREEN};")
        self.clone_button.setEnabled(True)
        self._update_batch_plan()

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
        language = str(self.clone_language_combo.currentData() or "en")
        clone_options = {
            "max_seconds": self.sample_length_spin.value(),
            "min_seconds": self._minimum_sample_seconds(),
            "preprocess": self.preprocess_check.isEnabled() and self.preprocess_check.isChecked(),
            "volume_normalization": self.volume_normalization_check.isEnabled() and self.volume_normalization_check.isChecked(),
            "language": language,
        }
        # 记住这次的选择，下次打开对话框直接沿用。
        self.settings.set("clone_language", language)
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

    # ------------------------------------------------------------ 批量复刻
    def _selected_batch_models(self) -> list[str]:
        models = []
        for row in range(self.batch_model_list.count()):
            item = self.batch_model_list.item(row)
            if item.checkState() == Qt.Checked:
                models.append(str(item.data(Qt.UserRole)))
        return models

    def _set_all_batch_models(self, checked: bool) -> None:
        self.batch_model_list.blockSignals(True)
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.batch_model_list.count()):
            self.batch_model_list.item(row).setCheckState(state)
        self.batch_model_list.blockSignals(False)
        self._update_batch_plan()

    def _check_recommended_models(self) -> None:
        """两个都支持语气指令和 SSML，一个 Qwen 家族一个 CosyVoice 家族，对比最有信息量。"""
        wanted = {"qwen-audio-3.0-tts-plus", "cosyvoice-v3.5-plus"}
        self.batch_model_list.blockSignals(True)
        for row in range(self.batch_model_list.count()):
            item = self.batch_model_list.item(row)
            item.setCheckState(Qt.Checked if str(item.data(Qt.UserRole)) in wanted else Qt.Unchecked)
        self.batch_model_list.blockSignals(False)
        self._update_batch_plan()

    def _selected_batch_durations(self) -> list[float]:
        durations: list[float] = []
        if self.batch_duration_recommended.isChecked():
            durations.append(RECOMMENDED_SAMPLE_SECONDS)
        if self.batch_duration_full.isChecked():
            durations.append(MAX_CLONE_SECONDS)
        if self.batch_duration_custom.isChecked():
            durations.append(float(self.sample_length_spin.value()))
        return durations

    def _current_batch_tasks(self):
        """没有样音时返回空列表，不抛 —— 这个函数每次勾选都会被调用。"""
        if self._sample_seconds <= 0:
            return [], []
        return build_tasks(
            self._selected_batch_models(),
            self._selected_batch_durations(),
            sample_seconds=self._sample_seconds,
            language=str(self.clone_language_combo.currentData() or "en"),
            preprocess=self.preprocess_check.isChecked(),
            volume_normalization=self.volume_normalization_check.isChecked(),
        )

    def _update_batch_plan(self) -> None:
        batch_on = self.batch_group.isChecked()
        self.batch_button.setVisible(batch_on)
        self.clone_button.setVisible(not batch_on)
        if not batch_on:
            self._update_channel_ui()
            return
        tasks, skipped = self._current_batch_tasks()
        if self._sample_seconds <= 0:
            self.batch_plan_label.setText("先录音或选择样音文件，再勾选要复刻的组合。")
            self.batch_plan_label.setStyleSheet(f"color: {TEXT_DIM};")
            self.batch_button.setEnabled(False)
            return
        lines = [describe_plan(tasks, skipped)]
        lines.extend(f"  • {task.label}" for task in tasks)
        # 跳过的组合必须说清楚，否则用户会以为软件默默吞了任务。
        lines.extend(f"  ✗ {item.model} · {item.seconds:g}s —— {item.reason}" for item in skipped)
        self.batch_plan_label.setText("\n".join(lines))
        self.batch_plan_label.setStyleSheet(f"color: {OK_GREEN if tasks else WARN_ORANGE};")
        self.batch_button.setEnabled(bool(tasks) and not self._cloning)
        # OSS 只在确实有任务走旧版通道时才要求，全 qwen3 直传就不该拦人。
        self._apply_batch_channel_ui(needs_oss(tasks))

    def _apply_batch_channel_ui(self, requires_oss: bool) -> None:
        self.oss_group.setVisible(requires_oss)
        if requires_oss:
            self.channel_note.setText(
                "本批含旧版通道模型：样音 → 私有 OSS 中转 → 提交复刻 → 等待部署 → 自动删除样音。"
                "同一时长的多个模型只上传一次。"
            )
            self._maybe_auto_probe_oss_for_batch()
        else:
            self.channel_note.setText("本批全部走 Qwen3 直传，无需 OSS。")

    def _maybe_auto_probe_oss_for_batch(self) -> None:
        if self._oss_probed:
            return
        self._oss_probed = True
        values = self._current_oss_values() or self._stored_oss_values()
        if values is None:
            self._set_oss_status("⚠️ 本批需要 OSS：填好下方信息后点「检测 OSS 并记住」", WARN_ORANGE)
            return
        self._run_oss_probe(values)

    def _cancel_batch(self) -> None:
        self._batch_cancel.set()
        self.batch_cancel_button.setEnabled(False)
        self._set_result("已请求停止：当前这个音色建完后就不再继续。", WARN_ORANGE)

    def _start_batch_clone(self) -> None:
        if self._cloning or self._sample_path is None:
            return
        tasks, _skipped = self._current_batch_tasks()
        if not tasks:
            self._set_result("当前勾选不会创建任何音色，请至少选一个模型和一个时长。", ERR_RED)
            return
        oss_values = None
        if needs_oss(tasks):
            try:
                oss_values = self._collect_oss_values()
            except OssUploadError as exc:
                self._set_result(str(exc), ERR_RED)
                return

        language = str(self.clone_language_combo.currentData() or "en")
        self.settings.set("clone_language", language)
        transcript = self.passage_edit.toPlainText().strip() if self.use_transcript.isChecked() else ""

        self._cloning = True
        self._batch_cancel = threading.Event()
        self.batch_button.setEnabled(False)
        self.batch_cancel_button.setEnabled(True)
        self.batch_cancel_button.setVisible(True)
        self.clone_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.batch_result_table.setRowCount(0)
        self.batch_result_table.setVisible(True)
        self._set_result(f"批量复刻已开始，共 {len(tasks)} 个音色…", TEXT_DIM)
        threading.Thread(
            target=self._batch_clone_worker,
            args=(self._sample_path, transcript, tuple(tasks), oss_values, self._batch_cancel),
            name="defense-clone-batch",
            daemon=True,
        ).start()

    def _batch_clone_worker(self, source: Path, transcript: str, tasks, oss_values, cancel) -> None:
        """按 (时长, 通道) 分组执行：每组只转码一次、只上传一次 OSS。

        单个任务失败不能中断整批 —— 用户等了几分钟，不该因为第三个模型
        报错就丢掉前两个已经成功的音色。每个任务的成败单独回报。
        """
        succeeded = 0
        attempted = 0
        try:
            client = BailianClient(
                self.settings.get_api_key(),
                self.settings.workspace_id,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=120,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.batch_result.emit({"model": "", "seconds": 0.0, "error": f"无法连接百炼：{exc}"})
            self.batch_done.emit(0, len(tasks))
            return

        for seconds, channel, group in group_tasks(tasks):
            if cancel.is_set():
                break
            floor = min(task.min_seconds for task in group)
            uploader = None
            uploaded = None
            encoded = ""
            try:
                self.clone_progress.emit(f"正在准备 {seconds:g} 秒样音（{len(group)} 个模型共用）…")
                with normalized_voice_sample(source, max_seconds=seconds, min_seconds=floor) as wav:
                    self.clone_progress.emit(inspect_voice_sample(wav))
                    if channel == "qwen3":
                        encoded = "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode("ascii")
                    else:
                        uploader = OssTemporaryUploader(**(oss_values or {}))
                        uploaded = uploader.upload(wav)
            except Exception as exc:
                # 整组的样音都没准备好，组内每个任务都要单独回报失败。
                for task in group:
                    attempted += 1
                    self.batch_result.emit(
                        {"model": task.model, "seconds": task.seconds, "error": f"样音准备失败：{exc}"}
                    )
                continue

            try:
                for task in group:
                    if cancel.is_set():
                        break
                    attempted += 1
                    self.clone_progress.emit(f"正在创建 {task.label} …")
                    try:
                        if channel == "qwen3":
                            voice_id, warning = client.clone_qwen_voice(
                                target_model=task.model,
                                preferred_name=f"defense_{int(time.time() * 1000) % 100_000_000}",
                                audio_data=encoded,
                                language=task.language,
                                transcript=transcript,
                            )
                        elif is_minimax_tts_model(task.model):
                            # 同步复刻；voice_id 自己起，带序号防同一毫秒内重名覆盖。
                            voice_id, _demo = client.clone_minimax_voice(
                                target_model=task.model,
                                voice_id=minimax_voice_name(task.language, salt=attempted),
                                audio_url=uploaded.signed_url,
                                preview_text=MINIMAX_PREVIEW_TEXT.get(
                                    task.language, MINIMAX_PREVIEW_TEXT["en"]
                                ),
                                noise_reduction=task.preprocess,
                                volume_normalization=task.volume_normalization,
                            )
                            warning = "MiniMax 首次用该音色合成时百炼会收一次性 ¥9.9 解锁费"
                        else:
                            voice_id = client.clone_voice(
                                target_model=task.model,
                                prefix=f"def{int(time.time() * 1000) % 10_000_000}",
                                audio_url=uploaded.signed_url,
                                language=task.language,
                                max_seconds=task.seconds,
                                preprocess=task.preprocess,
                                volume_normalization=task.volume_normalization,
                            )
                            client.wait_for_voice_ready(
                                voice_id,
                                timeout=240.0,
                                on_status=lambda message: self.clone_progress.emit(message),
                            )
                            warning = ""
                    except Exception as exc:
                        self.batch_result.emit(
                            {"model": task.model, "seconds": task.seconds, "error": " ".join(str(exc).split())[:200]}
                        )
                        continue
                    succeeded += 1
                    self.batch_result.emit(
                        {
                            "model": task.model,
                            "seconds": task.seconds,
                            "voice_id": voice_id,
                            "language": task.language,
                            "warning": warning,
                            "error": "",
                        }
                    )
            finally:
                if uploader is not None and uploaded is not None:
                    try:
                        uploader.delete(uploaded.key)
                    except Exception:
                        self.clone_progress.emit(
                            "OSS 临时样音删除失败，可稍后在控制台手动删除 temporary/ 下的对象。"
                        )
        self.batch_done.emit(succeeded, attempted)

    @Slot(object)
    def _on_batch_result(self, payload: dict) -> None:
        model = str(payload.get("model", ""))
        seconds = float(payload.get("seconds", 0.0) or 0.0)
        error = str(payload.get("error", ""))
        voice_id = str(payload.get("voice_id", ""))
        row = self.batch_result_table.rowCount()
        self.batch_result_table.insertRow(row)
        self.batch_result_table.setItem(row, 0, QTableWidgetItem(model))
        self.batch_result_table.setItem(row, 1, QTableWidgetItem(f"{seconds:g}s" if seconds else "-"))
        status = QTableWidgetItem("❌ " + error if error else ("⚠️ " + payload["warning"] if payload.get("warning") else "✅ 成功"))
        status.setForeground(QColor(ERR_RED if error else OK_GREEN))
        self.batch_result_table.setItem(row, 2, status)
        self.batch_result_table.setItem(row, 3, QTableWidgetItem(voice_id or "-"))
        self.batch_result_table.scrollToBottom()
        if voice_id:
            self._remember_voice(
                voice_id,
                model=model,
                language=str(payload.get("language", "en")),
                seconds=seconds,
            )
            # 和设置页当前模型一致的那个，顺手填进 voice_id 输入框。
            if model == self.tts_model and not self.created_voice_id:
                self.created_voice_id = voice_id

    @Slot(int, int)
    def _on_batch_done(self, succeeded: int, attempted: int) -> None:
        self._cloning = False
        self.batch_cancel_button.setVisible(False)
        self.batch_cancel_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.clone_button.setEnabled(self._sample_path is not None)
        self._update_batch_plan()
        failed = attempted - succeeded
        if succeeded and not failed:
            self._set_result(
                f"✅ 批量完成：{succeeded} 个音色已创建，全部记入音色库。"
                "接下来用 tools/voice_ab.py 做盲听对比。",
                OK_GREEN,
            )
        elif succeeded:
            self._set_result(
                f"⚠️ 批量结束：成功 {succeeded} 个，失败 {failed} 个。成功的已记入音色库，"
                "失败原因见上表，可单独重试。",
                WARN_ORANGE,
            )
        else:
            self._set_result("❌ 批量失败：没有创建任何音色，请看上表的失败原因。", ERR_RED)

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
                if is_minimax_tts_model(self.tts_model):
                    # MiniMax 复刻是同步的，成功即可用；不能走 wait_for_voice_ready，
                    # 那个查询接口不认识 MiniMax 的 voice_id，会一直等到超时。
                    language = options.get("language", "en")
                    voice_id, _demo = client.clone_minimax_voice(
                        target_model=self.tts_model,
                        voice_id=minimax_voice_name(language),
                        audio_url=uploaded.signed_url,
                        preview_text=MINIMAX_PREVIEW_TEXT.get(language, MINIMAX_PREVIEW_TEXT["en"]),
                        noise_reduction=options["preprocess"],
                        volume_normalization=options["volume_normalization"],
                    )
                else:
                    voice_id = client.clone_voice(
                        target_model=self.tts_model,
                        prefix=preferred,
                        audio_url=uploaded.signed_url,
                        language=options.get("language", "en"),
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
                language=options.get("language", "en"),
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
        if str(self.settings.get("tts_model", "")) == "qwen-audio-3.1-tts-next":
            return
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

    def _remember_voice(self, voice_id: str, *, model: str = "", language: str = "",
                        seconds: float = 0.0) -> None:
        entries = [entry for entry in self._library_entries() if entry.get("voice_id") != voice_id]
        # 记下样音语种和参考时长：做 A/B 时一眼能分清哪个是英文音色、哪个是
        # 18 秒版，避免拿中文音色去合成英文又得出「差别不大」的错误结论。
        model = model or self.tts_model
        language = language or str(self.clone_language_combo.currentData() or "en")
        seconds = seconds or float(self.sample_length_spin.value())
        entries.append(
            {
                "voice_id": voice_id,
                "target_model": model,
                "language": language,
                "sample_seconds": round(float(seconds), 3),
                "name": f"克隆 {language.upper()} {seconds:g}s {time.strftime('%m-%d %H:%M')}",
                "channel": clone_channel_for_model(model),
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
        # 关窗要让批量线程停下来：它还会 emit 信号，而信号连到的是即将销毁的控件。
        # 取消是协作式的 —— 当前这一个建完后退出，不会把阿里云上已创建的音色丢掉。
        self._batch_cancel.set()
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


def _http_proxies(settings):
    """和翻译请求同一套代理规则：直连模式显式不走系统代理，填了代理就用它。"""
    try:
        if settings.proxy_mode == "direct":
            return {"http": None, "https": None}
        if settings.ws_proxy:
            return {"http": settings.ws_proxy, "https": settings.ws_proxy}
    except Exception:  # noqa: BLE001
        pass
    return None


class _BalanceRelay(QObject):
    done = Signal(int, object, str)  # 请求编号、(amount, currency) 或 None、错误
    bill = Signal(object, str)      # 账单行 或 None, 错误


def _usage_summary(snap: dict) -> str:
    parts = [f"{format_tokens(snap['tokens'])} tokens"]
    if snap["tts_chars"]:
        parts.append(f"合成 {format_tokens(snap['tts_chars'])} 字")
    if snap["asr_seconds"]:
        parts.append(f"识别 {snap['asr_seconds'] / 60:.0f} 分钟")
    return " · ".join(parts)


class UsageBar(QFrame):
    """顶栏上的用量条：阿里云余额、本次花了多少、本次 token / 合成字数 / 识别时长，外加「消费记录」按钮。"""

    BALANCE_EVERY_MS = 60 * 1000
    QUERY_TIMEOUT_MS = 25 * 1000

    def __init__(self, settings: DefenseSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setStyleSheet("UsageBar { border: 1px solid #2a3348; border-radius: 8px; }"
                           "QLabel { border: none; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 4, 2)
        layout.setSpacing(8)
        column = QVBoxLayout()
        column.setSpacing(0)
        self.balance_label = QLabel("余额：—")
        self.balance_label.setStyleSheet(f"color: {OK_GREEN}; font-weight: 600;")
        self.session_label = QLabel("本次：0 tokens")
        self.session_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        column.addWidget(self.balance_label)
        column.addWidget(self.session_label)
        layout.addLayout(column, 1)
        self.refresh_button = QPushButton("刷新余额")
        self.refresh_button.clicked.connect(self.query_balance)
        layout.addWidget(self.refresh_button)
        self.history_button = QPushButton("消费记录")
        self.history_button.setToolTip("本机每一场用了多少，阿里云余额和每月账单")
        self.history_button.clicked.connect(self.open_history)
        layout.addWidget(self.history_button)
        self.relay = _BalanceRelay()
        self.relay.done.connect(self._on_balance)
        self._querying = False
        self._query_id = 0
        self.last_error = ""
        self._query_timeout = QTimer(self)
        self._query_timeout.setSingleShot(True)
        self._query_timeout.timeout.connect(self._balance_timeout)
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._balance_timer = QTimer(self)
        self._balance_timer.setInterval(self.BALANCE_EVERY_MS)
        self._balance_timer.timeout.connect(self.query_balance)
        self._balance_timer.start()
        self.balance_query = None    # 测试可替换：() -> (amount, currency)
        self.refresh()
        QTimer.singleShot(1500, self.query_balance)

    def _credentials(self) -> tuple[str, str]:
        shared = self.settings.shared
        try:
            return shared.get_oss_access_key_id(), shared.get_oss_access_key_secret()
        except Exception:  # noqa: BLE001
            return "", ""

    def refresh(self) -> None:
        snap = USAGE.snapshot()
        text = f"本次：{_usage_summary(snap)}"
        self.session_label.setText(text)
        self.session_label.setToolTip(
            f"大模型 {snap['llm_calls']} 次：输入 {snap['input_tokens']} / 输出 {snap['output_tokens']} tokens\n"
            f"语音合成 {snap['tts_chars']} 字；语音识别 {snap['asr_seconds'] / 60:.1f} 分钟\n"
            "（本机按接口返回的用量计数；钱以阿里云账单为准）")
        status = (" · 查询中…" if self._querying else
                  " · 刷新失败，显示上次结果" if self.last_error and snap["balance_now"] is not None else
                  " · 查询失败（点消费记录）" if self.last_error else "")
        self.balance_label.setText(balance_summary(snap) + status)
        warning = self.last_error or (snap["balance_now"] is not None and snap["balance_now"] <= 5)
        self.balance_label.setStyleSheet(f"color: {WARN_ORANGE if warning else OK_GREEN}; font-weight: 600;")
        self.balance_label.setToolTip(self.balance_status(snap))
        self.refresh_button.setEnabled(not self._querying)
        self.refresh_button.setText("查询中…" if self._querying else "刷新余额")
        dialog = getattr(self, "_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.refresh()

    def balance_status(self, snap: dict | None = None) -> str:
        snap = snap if snap is not None else USAGE.snapshot()
        updated = snap.get("balance_updated_at")
        stamp = time.strftime("%H:%M:%S", time.localtime(updated)) if updated else "尚无成功查询"
        state = "正在查询余额…" if self._querying else f"查询失败：{self.last_error}" if self.last_error else "余额查询已完成" if updated else "填好 AccessKey 后点击查询"
        return (f"{state}；最后成功更新：{stamp}。每分钟自动刷新，可手动刷新。\n"
                "用量会实时累计，账户出账和可用余额有延迟；余额不变不代表没有费用。"
                "余额变化从本次首次成功查询起计算，包含充值和账户内其他产品，不能作为本软件的准确费用。")

    def _balance_timeout(self) -> None:
        if self._querying:
            self._on_balance(self._query_id, None, "查询超时，请点击刷新余额重试")

    def query_balance(self) -> None:
        if self._querying:
            return
        key_id, secret = self._credentials()
        if self.balance_query is None and not (key_id and secret):
            self.last_error = "还没填完整的 AccessKey，请在消费记录中保存后重试"
            self.refresh()
            return
        self._query_id += 1
        request_id = self._query_id
        self._querying = True
        self._query_timeout.start(self.QUERY_TIMEOUT_MS)
        self.refresh()
        query = self.balance_query or (lambda: query_balance(
            key_id, secret, proxies=_http_proxies(self.settings)))
        relay = self.relay

        def worker() -> None:
            try:
                result, error = query(), ""
            except Exception as exc:  # noqa: BLE001
                result, error = None, " ".join(str(exc).split())[:160]
            try:
                relay.done.emit(request_id, result, error)
            except RuntimeError:  # 窗口已关闭/销毁
                pass

        threading.Thread(target=worker, name="balance-query", daemon=True).start()

    def _on_balance(self, request_id: int, result, error: str) -> None:
        if not self._querying or request_id != self._query_id:
            return  # 超时请求迟到的结果不能覆盖新查询
        self._query_timeout.stop()
        self._querying = False
        if result is not None:
            try:
                amount, currency = result
                USAGE.set_balance(float(amount), str(currency))
            except (TypeError, ValueError, OverflowError):
                error = "费用中心返回的余额格式无效"
        self.last_error = error or ("费用中心没有返回余额" if result is None else "")
        self.refresh()

    def open_history(self) -> None:
        dialog = getattr(self, "_dialog", None)
        if dialog is None:
            dialog = UsageDialog(self.window(), self.settings, self)
            self._dialog = dialog
        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


class UsageDialog(QDialog):
    """消费记录：阿里云 AccessKey、本次明细、本机历史每一场、阿里云每月账单。"""

    def __init__(self, parent: QWidget, settings: DefenseSettings, bar: UsageBar) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.settings = settings
        self.bar = bar
        self.setWindowTitle("消费记录")
        self.resize(1100, 820)
        self.bill_query = None     # 测试可替换：(cycle) -> rows
        layout = QVBoxLayout(self)

        key_box = QFrame()
        key_box.setStyleSheet("QFrame { border: 1px solid #2a3348; border-radius: 6px; } QLabel { border: none; }")
        key_layout = QGridLayout(key_box)
        key_layout.addWidget(QLabel("阿里云 AccessKey（查余额和账单用；和 OSS 上传共用同一对）"), 0, 0, 1, 4)
        self.ak_edit = QLineEdit()
        self.ak_edit.setPlaceholderText("AccessKey ID（已保存时留空即可）")
        self.sk_edit = QLineEdit()
        self.sk_edit.setEchoMode(QLineEdit.Password)
        self.sk_edit.setPlaceholderText("AccessKey Secret（存进 Windows 凭据管理器，不写进任何文件）")
        key_layout.addWidget(self.ak_edit, 1, 0, 1, 2)
        key_layout.addWidget(self.sk_edit, 1, 2, 1, 2)
        self.save_key_button = QPushButton("保存并查余额")
        self.save_key_button.clicked.connect(self._save_keys)
        key_layout.addWidget(self.save_key_button, 2, 3)
        self.refresh_button = QPushButton("刷新余额")
        self.refresh_button.clicked.connect(self.bar.query_balance)
        key_layout.addWidget(self.refresh_button, 3, 3)
        self.key_hint = QLabel(
            "建议在阿里云 RAM 控制台新建一个子账号，只给「AliyunBSSReadOnlyAccess」（只读费用）权限，"
            "用它的 AccessKey，别用主账号的。")
        self.key_hint.setWordWrap(True)
        self.key_hint.setStyleSheet(f"color: {TEXT_DIM};")
        key_layout.addWidget(self.key_hint, 2, 0, 1, 3)
        layout.addWidget(key_box)

        self.session_label = QLabel("")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)
        self.session_table = QTableWidget(0, 4)
        self.session_table.setHorizontalHeaderLabels(["类型", "模型", "次数", "用量"])
        self.session_table.horizontalHeader().setStretchLastSection(True)
        self.session_table.setMaximumHeight(200)
        layout.addWidget(self.session_table)

        tabs = QTabWidget()
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["开始", "时长", "tokens（入/出）", "合成字数", "识别分钟", "余额减少（非准确费用）"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.history_table, "本机每一场")
        bill_page = QWidget()
        bill_layout = QVBoxLayout(bill_page)
        bill_row = QHBoxLayout()
        bill_row.addWidget(QLabel("月份"))
        self.month_combo = NoWheelComboBox()
        today = datetime.date.today()
        year, month = today.year, today.month
        for _ in range(12):
            self.month_combo.addItem(f"{year}-{month:02d}", f"{year}-{month:02d}")
            month -= 1
            if month == 0:
                year, month = year - 1, 12
        bill_row.addWidget(self.month_combo)
        self.bill_button = QPushButton("查询阿里云账单")
        self.bill_button.clicked.connect(self.query_bill)
        bill_row.addWidget(self.bill_button)
        bill_row.addStretch(1)
        bill_layout.addLayout(bill_row)
        self.bill_table = QTableWidget(0, 2)
        self.bill_table.setHorizontalHeaderLabels(["产品", "金额"])
        self.bill_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        bill_layout.addWidget(self.bill_table, 1)
        self.bill_status = QLabel("按产品汇总的应付金额（税前），以阿里云费用中心为准。")
        self.bill_status.setStyleSheet(f"color: {TEXT_DIM};")
        bill_layout.addWidget(self.bill_status)
        tabs.addTab(bill_page, "阿里云每月账单")
        layout.addWidget(tabs, 1)
        self.relay = _BalanceRelay()
        self.relay.bill.connect(self._on_bill)

    def refresh(self) -> None:
        snap = USAGE.snapshot()
        self.key_hint.setText(self.bar.balance_status(snap))
        self.refresh_button.setEnabled(not self.bar._querying)
        self.refresh_button.setText("查询中…" if self.bar._querying else "刷新余额")
        self.save_key_button.setEnabled(not self.bar._querying)
        balance = balance_summary(snap)
        started = time.strftime("%H:%M", time.localtime(snap["started_at"]))
        self.session_label.setText(f"本次（{started} 打开软件至今）：{_usage_summary(snap)}。{balance}")
        # 余额提示持续更新；用量没有变化时不重建表格，保留用户正在看的行。
        if getattr(self, "_last_snapshot", None) == snap:
            return
        self._last_snapshot = snap
        rows = []
        for model, (calls, inp, out) in snap["llm"].items():
            rows.append(("大模型", model, calls, f"输入 {inp} / 输出 {out} tokens"))
        for model, (calls, chars) in snap["tts"].items():
            rows.append(("语音合成", model, calls, f"{chars} 字"))
        for model, seconds in snap["asr"].items():
            rows.append(("语音识别", model, "—", f"{seconds / 60:.1f} 分钟"))
        self.session_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                self.session_table.setItem(row, col, QTableWidgetItem(str(value)))
        history = load_history(self.settings.base_dir)
        self.history_table.setRowCount(len(history))
        for row, item in enumerate(history):
            start = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(item.get("started_at", 0) or 0)))
            minutes = (float(item.get("ended_at", 0) or 0) - float(item.get("started_at", 0) or 0)) / 60
            spent = item.get("spent")
            values = [start, f"{minutes:.0f} 分钟", f"{item.get('input_tokens', 0)} / {item.get('output_tokens', 0)}",
                      str(item.get("tts_chars", 0)), f"{float(item.get('asr_seconds', 0) or 0) / 60:.1f}",
                      f"¥{spent:.2f}" if isinstance(spent, (int, float)) else "—（没查余额）"]
            for col, value in enumerate(values):
                self.history_table.setItem(row, col, QTableWidgetItem(value))

    def _save_keys(self) -> None:
        shared = self.settings.shared
        key_id, secret = self.ak_edit.text().strip(), self.sk_edit.text().strip()
        try:
            if key_id:
                shared.set_oss_access_key_id(key_id)
            if secret:
                shared.set_oss_access_key_secret(secret)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存 AccessKey", f"写入 Windows 凭据管理器失败：{exc}")
            return
        self.ak_edit.clear()
        self.sk_edit.clear()
        self.key_hint.setText("已保存，正在查余额…")
        self.bar.query_balance()

    def query_bill(self) -> None:
        cycle = str(self.month_combo.currentData())
        key_id, secret = self.bar._credentials()
        query = self.bill_query or (lambda c: query_month_bill(
            key_id, secret, c, proxies=_http_proxies(self.settings)))
        self.bill_button.setEnabled(False)
        self.bill_status.setText(f"正在查询 {cycle} 的账单…")
        relay = self.relay

        def worker() -> None:
            try:
                relay.bill.emit(query(cycle), "")
            except Exception as exc:  # noqa: BLE001
                relay.bill.emit(None, " ".join(str(exc).split())[:200])

        threading.Thread(target=worker, name="bill-query", daemon=True).start()

    def _on_bill(self, rows, error: str) -> None:
        self.bill_button.setEnabled(True)
        if rows is None:
            self.bill_status.setText(f"查询失败：{error}")
            return
        self.bill_table.setRowCount(len(rows))
        total = 0.0
        for row, item in enumerate(rows):
            self.bill_table.setItem(row, 0, QTableWidgetItem(item["product"]))
            self.bill_table.setItem(row, 1, QTableWidgetItem(f"¥{item['amount']:.2f}"))
            total += item["amount"]
        self.bill_status.setText(f"合计 ¥{total:.2f}（税前，以阿里云费用中心为准）" if rows else "这个月没有消费记录")


class TermDedupeDialog(QDialog):
    """术语表查重：一组一组看。中文相同的必须合成一条；其他的可以删、改，或者都保留（以后不再提示）。

    全部在内存里做，点「完成」才把结果交回术语表窗口（那边还要点保存才写盘）。
    """

    def __init__(self, parent: QWidget, terms: list[dict[str, str]], ignored: set[str]) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.setWindowTitle("术语表查重")
        self.resize(900, 520)
        self._items = [{"uid": i, "source": t["source"], "target": t["target"], "alive": True}
                       for i, t in enumerate(terms)]
        self.ignored = set(ignored)
        self.new_ignored: set[str] = set()
        self._groups = [[g.indices, g.reason, g.must_merge, g.key(terms)]
                        for g in find_term_groups(terms, self.ignored)]
        self._position = 0
        layout = QVBoxLayout(self)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.progress_label)
        self.reason_label = QLabel("")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet(f"color: {WARN_ORANGE}; font-weight: 600;")
        layout.addWidget(self.reason_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["保留", "中文说法（可改）", "英文写法（可改）"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 260)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.hint_label)
        row = QHBoxLayout()
        self.apply_button = QPushButton("按勾选处理这组")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.clicked.connect(self.apply_group)
        row.addWidget(self.apply_button)
        self.keep_all_button = QPushButton("都保留（以后不再提示）")
        self.keep_all_button.clicked.connect(self.keep_all)
        row.addWidget(self.keep_all_button)
        self.skip_button = QPushButton("先跳过")
        self.skip_button.clicked.connect(self.skip)
        row.addWidget(self.skip_button)
        row.addStretch(1)
        cancel = QPushButton("取消（什么都不改）")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self.finish_button = QPushButton("完成")
        self.finish_button.clicked.connect(self.finish)
        row.addWidget(self.finish_button)
        layout.addLayout(row)
        self._show()

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def _alive(self, indices) -> list[dict]:
        return [self._items[i] for i in indices if self._items[i]["alive"]]

    def _current(self):
        while self._position < len(self._groups):
            indices, reason, must, key = self._groups[self._position]
            members = self._alive(indices)
            if len(members) >= 2:
                return indices, reason, must, key, members
            self._position += 1
        return None

    def _show(self) -> None:
        current = self._current()
        if current is None:
            self.table.setRowCount(0)
            self.reason_label.setText("✅ 都处理完了" if self._groups else "✅ 没有发现重复或相近的术语")
            self.progress_label.setText("")
            self.hint_label.setText("点「完成」把结果放回术语表（那边再点保存才生效）。")
            for button in (self.apply_button, self.keep_all_button, self.skip_button):
                button.setEnabled(False)
            return
        _indices, reason, must, _key, members = current
        self.progress_label.setText(f"第 {self._position + 1} / {len(self._groups)} 组")
        self.reason_label.setText(reason)
        self.table.setRowCount(len(members))
        for row, item in enumerate(members):
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if (row == 0 or not must) else Qt.Unchecked)
            check.setData(Qt.UserRole, item["uid"])
            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(item["source"]))
            self.table.setItem(row, 2, QTableWidgetItem(item["target"]))
        self.keep_all_button.setEnabled(not must)
        self.apply_button.setEnabled(True)
        self.skip_button.setEnabled(not must)
        self.hint_label.setText(
            "中文一样只能留一条：勾一条保留（可以先把英文改成你要的写法），其余会删掉。" if must else
            "取消勾选的会删掉；两条都有用就点「都保留」。改了文字也会一起生效。")

    def apply_group(self) -> None:
        current = self._current()
        if current is None:
            return
        _indices, _reason, must, _key, _members = current
        kept = [row for row in range(self.table.rowCount()) if self.table.item(row, 0).checkState() == Qt.Checked]
        if must and len(kept) != 1:
            QMessageBox.information(self, "术语表查重", "中文相同的只能保留一条，请只勾一条。")
            return
        if not kept:
            QMessageBox.information(self, "术语表查重", "至少保留一条；都不要的话请回术语表里直接删除。")
            return
        for row in range(self.table.rowCount()):
            item = self._items[int(self.table.item(row, 0).data(Qt.UserRole))]
            if row in kept:
                item["source"] = self.table.item(row, 1).text().strip() or item["source"]
                item["target"] = self.table.item(row, 2).text().strip() or item["target"]
            else:
                item["alive"] = False
        self._position += 1
        self._show()

    def keep_all(self) -> None:
        current = self._current()
        if current is None or current[2]:
            return
        self.new_ignored.add(current[3])
        self._position += 1
        self._show()

    def skip(self) -> None:
        current = self._current()
        if current is None or current[2]:
            return
        self._position += 1
        self._show()

    def result_terms(self) -> list[dict[str, str]]:
        return [{"source": i["source"], "target": i["target"]} for i in self._items if i["alive"]]

    def finish(self) -> None:
        if same_source_conflicts(self.result_terms()):
            QMessageBox.warning(self, "术语表查重", "还有中文相同的术语没合并（可能是刚才改文字造成的），请先处理完。")
            self._groups = [[g.indices, g.reason, g.must_merge, g.key(self.result_terms())]
                            for g in find_term_groups(self.result_terms(), self.ignored | self.new_ignored)]
            self._items = [{"uid": i, **t, "alive": True} for i, t in enumerate(self.result_terms())]
            self._position = 0
            self._show()
            return
        self.accept()


class QaDedupeDialog(QDialog):
    """问答库查重：意思相近的问题两两并排，你选保留哪条、合并还是都保留。

    合并 = 保留一条的回答，另一条的问法记成「其他问法」（以后这样问也能命中）。
    处理过程只在内存里，点「完成」才写盘；删掉的条目的声音只有在没人再用时才清理。
    """

    verified = Signal(str)

    def __init__(self, parent: QWidget, panel, *, only_ids: set[str] | None = None) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.panel = panel
        self.settings = panel.settings
        self.setWindowTitle("问答库查重")
        self.resize(1200, 720)
        self.verify_chat_factory = panel._default_match_chat     # 测试可替换
        ignored = set(self.settings.get("qa_dedupe_ignored", []) or [])
        entries = list(panel.bank)
        self.pairs = find_qa_pairs(entries, only_ids=only_ids, ignored=ignored)
        self.session = QaDedupeSession(entries, self.pairs)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        top.addWidget(self.summary_label, 1)
        self.verify_button = QPushButton("让大模型判断拿不准的（flash）")
        self.verify_button.setToolTip("把「可能相近」的那些对交给快模型判断是不是同一个问题，结果标在每一对上，最后仍由你决定")
        self.verify_button.clicked.connect(self.verify_uncertain)
        top.addWidget(self.verify_button)
        layout.addLayout(top)
        self.pair_label = QLabel("")
        self.pair_label.setStyleSheet(f"color: {WARN_ORANGE}; font-weight: 600;")
        layout.addWidget(self.pair_label)
        columns = QSplitter(Qt.Horizontal)
        self.left_view = QTextBrowser()
        self.right_view = QTextBrowser()
        columns.addWidget(self.left_view)
        columns.addWidget(self.right_view)
        layout.addWidget(columns, 1)
        row = QHBoxLayout()
        self.action_buttons = {}
        for action, text, tip in (
            ("merge_left", "合并 · 用左边的回答", "留左边这条，右边的问法记成其他问法"),
            ("merge_right", "合并 · 用右边的回答", "留右边这条，左边的问法记成其他问法"),
            ("keep_left", "只留左边", "删掉右边这条"),
            ("keep_right", "只留右边", "删掉左边这条"),
            ("both", "都保留（以后不提示）", "问的不是同一件事"),
            ("skip", "先跳过", "这次不处理，下次查重还会出现"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, a=action: self.decide(a))
            row.addWidget(button)
            self.action_buttons[action] = button
        layout.addLayout(row)
        bottom = QHBoxLayout()
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self.progress_label, 1)
        cancel = QPushButton("取消（什么都不改）")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        self.finish_button = QPushButton("完成（写入问答库）")
        self.finish_button.setObjectName("primaryButton")
        self.finish_button.clicked.connect(self.finish)
        bottom.addWidget(self.finish_button)
        layout.addLayout(bottom)
        self.verified.connect(self._on_verified)
        self._render()

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def _entry_html(self, entry: QaEntry, side: str) -> str:
        ready = False
        try:
            ready = bool(self.panel.engine.answer_ready(entry.answer))
        except Exception:  # noqa: BLE001
            pass
        esc = html.escape
        aliases = "<br>".join(esc(a) for a in entry.aliases)
        return (f"<h3>{side} · {esc(entry.id)} {'★' * entry.priority}</h3>"
                f"<p><b>{esc(entry.question)}</b><br><span style='color:{TEXT_DIM}'>{esc(entry.question_zh)}</span></p>"
                + (f"<p style='color:{TEXT_DIM}'>其他问法：<br>{aliases}</p>" if aliases else "")
                + f"<p>{esc(entry.answer)}</p><p style='color:{TEXT_DIM}'>{esc(entry.answer_zh)}</p>"
                + f"<p style='color:{OK_GREEN if ready else TEXT_DIM}'>{'✓ 已渲染声音' if ready else '— 未渲染'}</p>")

    def _render(self) -> None:
        pending = self.session.pending()
        strong = sum(1 for p in pending if p.level == "很可能重复")
        self.summary_label.setText(
            f"还剩 {len(pending)} 对（很可能重复 {strong}，可能相近 {len(pending) - strong}）。只比较问题，不比较回答。"
            if self.pairs else "✅ 没有发现意思相近的问题")
        current = pending[0] if pending else None
        for button in self.action_buttons.values():
            button.setEnabled(current is not None)
        self.verify_button.setEnabled(any(not p.exact and p.score < STRONG and not p.verdict for p in pending))
        removed = len(self.session.removed)
        self.progress_label.setText(f"已处理：删除 / 合并 {removed} 条，都保留 {len(self.session.ignored)} 对")
        if current is None:
            self.pair_label.setText("✅ 都处理完了，点「完成」写入问答库" if self.pairs else "")
            self.left_view.setHtml("")
            self.right_view.setHtml("")
            return
        by_id = self.session.by_id()
        verdict = {"same": " · 大模型：是同一个问题", "different": " · 大模型：不是同一个问题"}.get(current.verdict, "")
        self.pair_label.setText(f"{current.level}（相似度 {current.score:.2f}）{verdict}")
        self.left_view.setHtml(self._entry_html(by_id[current.left], "左"))
        self.right_view.setHtml(self._entry_html(by_id[current.right], "右"))

    def decide(self, action: str) -> None:
        self.session.decide(action)
        self._render()

    def verify_uncertain(self) -> None:
        chat = self.verify_chat_factory()
        if chat is None:
            self.pair_label.setText("没法调用大模型（没配 API Key？），请自己判断")
            return
        uncertain = [p for p in self.pairs if not p.exact and p.score < STRONG and not p.verdict]
        if not uncertain:
            return
        self.verify_button.setEnabled(False)
        by_id = {e.id: e for e in self.panel.bank}
        messages = build_verify_messages(uncertain[:60], by_id)

        def worker() -> None:
            try:
                parse_verify(chat(messages), uncertain[:60])
                self.verified.emit("")
            except Exception as exc:  # noqa: BLE001
                self.verified.emit(" ".join(str(exc).split())[:120])

        threading.Thread(target=worker, name="qa-dedupe-verify", daemon=True).start()

    def _on_verified(self, error: str) -> None:
        if error:
            self.summary_label.setText(f"大模型判断失败：{error}（不影响，自己判断就行）")
        # 大模型说「不是同一个」的往后放，说「是」的往前放
        self.pairs.sort(key=lambda p: (p.verdict == "different", not (p.exact or p.verdict == "same"), -p.score))
        self._render()

    def finish(self) -> None:
        dialog = self.panel._bank_dialog
        if dialog is not None and getattr(dialog, "_rendering", False):
            QMessageBox.information(self, "问答库查重", "问答库正在后台渲染声音，等它结束（或点停止）再完成查重。")
            return
        session = self.session
        if session.removed or session.entries != list(self.panel.bank):
            self.panel.bank_store.save(session.entries)
            for text in session.audio_to_delete():
                try:
                    self.panel.engine.delete_answer_audio(text)
                except Exception:  # noqa: BLE001
                    pass
        if session.ignored:
            known = list(self.settings.get("qa_dedupe_ignored", []) or [])
            self.settings.set("qa_dedupe_ignored", known + [k for k in session.ignored if k not in known])
        self.panel.reload_bank()
        if dialog is not None:
            dialog.refresh()
        self.panel.qa_status.emit(f"问答库查重完成：删除 / 合并 {len(session.removed)} 条，现有 {len(session.entries)} 条")
        self.accept()


class GlossaryReviewDialog(QDialog):
    """自动生成的术语先在这里过一遍：新增的默认勾上，和已有写法冲突的默认不勾。"""

    def __init__(self, parent: QWidget, result) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.setWindowTitle("核对自动生成的术语")
        self.resize(1000, 720)
        self.result = result
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"从 {result.chunks} 块内容里找到 {len(result.candidates)} 条术语："
            f"新增 {len(result.new)} 条（默认勾上）、和已有写法不同 {len(result.conflicts)} 条（默认不勾，勾上会改成新写法）、"
            f"已有 {len(result.same)} 条（不显示）。"
            + (f"\n⚠️ 有 {result.failed_chunks} 块没处理成功，可以再生成一次补上。" if result.failed_chunks else "")
            + "\n英文可以直接在表里改；排序按在论文里出现的次数，越常出现越靠前。"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["加入", "中文说法", "英文写法（可改）", "说明"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for col, width in enumerate((56, 260, 380, 240)):
            self.table.setColumnWidth(col, width)
        self.table.verticalHeader().setVisible(False)
        self._rows = [c for c in result.candidates if c.status in ("new", "conflict")]
        for candidate in self._rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if candidate.status == "new" else Qt.Unchecked)
            self.table.setItem(row, 0, check)
            source = QTableWidgetItem(candidate.source)
            source.setFlags(source.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, source)
            self.table.setItem(row, 2, QTableWidgetItem(candidate.target))
            note = (f"已有写法：{candidate.existing_target}" if candidate.status == "conflict"
                    else f"新增 · 出现 {candidate.hits} 次")
            note_item = QTableWidgetItem(note)
            note_item.setFlags(note_item.flags() & ~Qt.ItemIsEditable)
            if candidate.status == "conflict":
                note_item.setForeground(QColor(WARN_ORANGE))
            self.table.setItem(row, 3, note_item)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, state in (("全选", Qt.Checked), ("全不选", Qt.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, s=state: self._check_all(s))
            row.addWidget(button)
        row.addStretch(1)
        cancel = QPushButton("都不要")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        ok = QPushButton("加入勾选的")
        ok.setObjectName("primaryButton")
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        layout.addLayout(row)

    def _check_all(self, state) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)

    def chosen(self) -> list[dict[str, str]]:
        terms = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.Checked:
                continue
            source = self.table.item(row, 1).text().strip()
            target = self.table.item(row, 2).text().strip()
            if source and target:
                terms.append({"source": source, "target": target})
        return terms


class GlossaryDialog(QDialog):
    """术语表（从设置里搬出来的独立窗口）：编辑、排序、导入导出，以及从论文和 PPT 自动生成全部术语。

    术语表同时用在三处：翻译时强制译法、语音识别热词（前 50 条）、助答 / 预习 / 校准的提示词。
    不是模态窗口：生成要几分钟，期间主界面照常可用。
    """

    gen_progress = Signal(str)
    gen_done = Signal(object, str)
    audit_done = Signal(object, str)

    def __init__(self, parent: QWidget, settings: DefenseSettings, on_saved=None) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.settings = settings
        self.on_saved = on_saved
        self._dirty = False
        self._loading = False
        self._cancel = threading.Event()
        self._extra_files: list[str] = []
        self.post_factory = None      # 测试可替换：() -> post(payload) -> 正文
        self._audit_running = False
        self._audit_undo = None
        self.setWindowTitle("术语表 · 翻译和识别都按这里的写法")
        self.resize(1180, 860)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "中文说法 → 英文写法。翻译时逐字按这里的英文说，评委那边也按这里译回中文；"
            f"<b>前 {ASR_HOTWORD_COUNT} 条</b>同时作为语音识别热词（专有名词更不容易听错），重要的请往上排。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ---- 自动生成 ----
        gen_box = QFrame()
        gen_box.setStyleSheet("QFrame { border: 1px solid #2a3348; border-radius: 6px; }"
                              "QLabel, QCheckBox { border: none; }")
        gen_layout = QVBoxLayout(gen_box)
        gen_title = QLabel("🤖 从论文和答辩 PPT 自动找出全部术语")
        gen_title.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        gen_layout.addWidget(gen_title)
        source_row = QHBoxLayout()
        self.use_thesis_check = QCheckBox("")
        self.use_thesis_check.setChecked(True)
        source_row.addWidget(self.use_thesis_check)
        self.pick_files_button = QPushButton("再加论文 / PPT 文件…")
        self.pick_files_button.setToolTip("PPT（.pptx）、论文（PDF / Word / txt），可以多选；只读取文字，不上传文件本身")
        self.pick_files_button.clicked.connect(self._pick_files)
        source_row.addWidget(self.pick_files_button)
        self.files_label = QLabel("")
        self.files_label.setStyleSheet(f"color: {TEXT_DIM};")
        source_row.addWidget(self.files_label, 1)
        gen_layout.addLayout(source_row)
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("用哪个模型"))
        self.model_combo = NoWheelComboBox()
        fill_model_combo(self.model_combo, QA_MODELS, str(settings.get("llm_model", "") or "qwen3.8-flash"),
                         RECOMMENDED_QA)
        self.model_combo.setToolTip("找术语是体力活，flash 就够；想更全可以选 plus / max（更慢、稍贵）")
        run_row.addWidget(self.model_combo, 1)
        self.generate_button = QPushButton("开始生成")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self.generate)
        run_row.addWidget(self.generate_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self._cancel.set())
        run_row.addWidget(self.stop_button)
        gen_layout.addLayout(run_row)
        self.gen_status = QLabel("生成完先给你核对，确认后才加入；已有的术语不会被改，除非你勾选了「改成新写法」。")
        self.gen_status.setWordWrap(True)
        self.gen_status.setStyleSheet(f"color: {TEXT_DIM};")
        gen_layout.addWidget(self.gen_status)
        layout.addWidget(gen_box)

        # ---- 表格 ----
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索中文或英文…")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)
        audit_row = QHBoxLayout()
        self.audit_button = QPushButton("Flash 检查并修正")
        self.audit_button.setToolTip("调用 Flash 检查整张表，自动修正英文朗读写法；保留变量、单位和公式。会产生接口费用。")
        self.audit_button.clicked.connect(self.audit)
        audit_row.addWidget(self.audit_button)
        self.audit_undo_button = QPushButton("撤销本次修正")
        self.audit_undo_button.setEnabled(False)
        self.audit_undo_button.clicked.connect(self.undo_audit)
        audit_row.addWidget(self.audit_undo_button)
        audit_note = QLabel("修正重复释义与书面斜杠；保留 (D)、(L)、(W) 等科学符号。修改后点保存生效。")
        audit_note.setWordWrap(True)
        audit_row.addWidget(audit_note, 1)
        layout.addLayout(audit_row)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["中文说法", "英文写法"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(0, 420)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        for text, slot, tip in (
            ("添加", self._add_row, "在末尾加一行，直接在表里填"),
            ("删除所选", self._remove_rows, "可以 Ctrl / Shift 多选"),
            ("↑ 上移", lambda: self._move(-1), ""),
            ("↓ 下移", lambda: self._move(1), ""),
            ("⤒ 置顶", self._to_top, f"排进前 {ASR_HOTWORD_COUNT} 条就会作为识别热词"),
            ("导入 JSON…", self._import_json, "[{\"source\": \"中文\", \"target\": \"English\"}]"),
            ("导出 JSON…", self._export_json, "备份，或发给同学"),
            ("清空", self._clear_all, ""),
            ("🔍 查重", self.open_dedupe, "找出中文相同、英文相同、缩写不一致、互相包含的术语，逐组确认"),
        ):
            button = QPushButton(text)
            if tip:
                button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {TEXT_DIM};")
        buttons.addWidget(self.count_label)
        self.save_button = QPushButton("保存")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        self.gen_progress.connect(self.gen_status.setText)
        self.gen_done.connect(self._on_generated)
        self.audit_done.connect(self._on_audited)
        self.reload()

    def _raw_rows(self):
        return [(self.table.item(r, c).text() if self.table.item(r, c) else "")
                for r in range(self.table.rowCount()) for c in (0, 1)]

    def audit(self) -> None:
        if self._audit_running or not self.generate_button.isEnabled():
            return
        terms = self.terms()
        if not terms:
            self.gen_status.setText("术语表为空，没有可检查的内容")
            return
        self._cancel = threading.Event()
        cancel = self._cancel
        try:
            post = self.post_factory() if self.post_factory else self._default_post(model=AUDIT_MODEL)
        except Exception as exc:
            self.gen_status.setText(f"无法开始检查：{exc}")
            return
        self._audit_snapshot = self._raw_rows()
        self._audit_was_dirty = self._dirty
        self._audit_rows = [r for r in range(self.table.rowCount())
                            if all(self._audit_snapshot[r*2+c].strip() for c in (0, 1))]
        self._audit_running = True
        self.audit_button.setEnabled(False)
        self.audit_undo_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.gen_status.setText("Flash 正在检查，成功后自动修改英文；中文匹配键保持不变…")
        done, progress = self.audit_done, self.gen_progress

        def worker():
            try:
                def notify(message):
                    try:
                        progress.emit(message)
                    except RuntimeError:
                        cancel.set()
                patches = audit_terms(terms, post=post, cancel_event=cancel, on_progress=notify)
                error = ""
            except Exception as exc:
                patches, error = None, " ".join(str(exc).split())[:240]
            try:
                done.emit(patches, error)
            except RuntimeError:
                pass
        threading.Thread(target=worker, name="glossary-audit", daemon=True).start()

    def _on_audited(self, patches, error):
        self._audit_running = False
        self.audit_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.audit_undo_button.setEnabled(self._audit_undo is not None)
        if self._cancel.is_set() or patches is None:
            self.gen_status.setText(f"检查未应用，术语没有修改：{error or '已停止'}")
            return
        if self._raw_rows() != self._audit_snapshot:
            self.gen_status.setText("检查期间你修改了表格，本次结果未应用，避免覆盖；请重新检查")
            return
        if not patches:
            self.gen_status.setText("Flash 检查完成，没有发现需要修正的英文写法")
            return
        self._loading = True
        try:
            for patch in patches:
                self.table.item(self._audit_rows[patch['id']], 1).setText(patch['target'])
        finally:
            self._loading = False
        self._audit_undo = (list(self._audit_snapshot), self._audit_was_dirty, self._raw_rows())
        self.audit_undo_button.setEnabled(True)
        self._set_dirty(True)
        self._restyle()
        self._apply_filter()
        details = "\n".join(f"{self._audit_snapshot[self._audit_rows[p['id']]*2+1]} → {p['target']}：{p['reason']}" for p in patches)
        self.gen_status.setToolTip(details)
        self.gen_status.setText(f"已自动修正 {len(patches)} 条英文（悬停查看改动），变量/单位/公式受保护；点保存生效，也可撤销")

    def undo_audit(self) -> None:
        if self._audit_running or self._audit_undo is None:
            return
        old, dirty, applied = self._audit_undo
        if self._raw_rows() != applied:
            self.gen_status.setText("修正后你又编辑了表格，未撤销，避免覆盖后续修改")
            return
        self._fill([{'source':old[i], 'target':old[i+1]} for i in range(0, len(old), 2)])
        self._set_dirty(dirty or self.settings.glossary_terms() != self.terms())
        self._apply_filter()
        self._audit_undo = None
        self.audit_undo_button.setEnabled(False)
        self.gen_status.setText("已撤销本次修正；保存后生效")

    def reject(self) -> None:
        self._cancel.set()
        super().reject()

    # ------------------------------------------------------------ 数据
    def reload(self) -> None:
        """从设置重新读（导入论文 / PPT 可能刚往里加了术语）。有没保存的修改时不覆盖。"""
        if self._dirty:
            return
        self._fill(self.settings.glossary_terms())
        store = ThesisStore(self.settings.base_dir)
        text = store.full_text()
        names = str(self.settings.get("context_source_path", "") or "")
        if text.strip():
            shown = "、".join(Path(p).name for p in names.split("; ") if p) or "已导入的论文"
            self.use_thesis_check.setText(f"已导入的论文：{shown}（约 {len(text) // 1000} 千字）")
            self.use_thesis_check.setEnabled(True)
        else:
            self.use_thesis_check.setText("还没导入论文（主界面「导入论文 / PPT」，或在右边直接选文件）")
            self.use_thesis_check.setChecked(False)
            self.use_thesis_check.setEnabled(False)
        self._set_dirty(False)

    def _fill(self, terms: list[dict[str, str]]) -> None:
        self._loading = True
        self.table.setRowCount(0)
        for term in terms:
            self._append(term.get("source", ""), term.get("target", ""))
        self._loading = False
        self._restyle()

    def _append(self, source: str, target: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(source))
        self.table.setItem(row, 1, QTableWidgetItem(target))

    def terms(self) -> list[dict[str, str]]:
        """表里的全部术语（空行跳过）。中文重复的不在这里悄悄丢掉——保存时会拦下来让你合并。"""
        result = []
        for row in range(self.table.rowCount()):
            source = (self.table.item(row, 0).text() if self.table.item(row, 0) else "").strip()
            target = (self.table.item(row, 1).text() if self.table.item(row, 1) else "").strip()
            if source and target:
                result.append({"source": source, "target": target})
        return result

    def _restyle(self) -> None:
        """前 50 条（识别热词）行号标出来。"""
        self.table.setVerticalHeaderLabels([
            f"{row + 1}{' 🔥' if row < ASR_HOTWORD_COUNT else ''}" for row in range(self.table.rowCount())])
        count = len(self.terms())
        self.count_label.setText(f"共 {count} 条 · 前 {min(count, ASR_HOTWORD_COUNT)} 条是识别热词 🔥")

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.save_button.setText("保存 *" if dirty else "保存")

    def _on_item_changed(self, _item) -> None:
        if not self._loading:
            self._set_dirty(True)
            self._restyle()

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        for row in range(self.table.rowCount()):
            hay = " ".join((self.table.item(row, c).text() if self.table.item(row, c) else "") for c in (0, 1)).lower()
            self.table.setRowHidden(row, bool(query) and query not in hay)

    # ------------------------------------------------------------ 编辑
    def _add_row(self) -> None:
        self._append("", "")
        self.table.scrollToBottom()
        self.table.setCurrentCell(self.table.rowCount() - 1, 0)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))
        self._set_dirty(True)
        self._restyle()

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _remove_rows(self) -> None:
        rows = self._selected_rows()
        for row in reversed(rows):
            self.table.removeRow(row)
        if rows:
            self._set_dirty(True)
            self._restyle()

    def _swap_rows(self, a: int, b: int) -> None:
        self._loading = True
        for col in (0, 1):
            first, second = self.table.takeItem(a, col), self.table.takeItem(b, col)
            self.table.setItem(a, col, second)
            self.table.setItem(b, col, first)
        self._loading = False

    def _move(self, step: int) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        order = rows if step < 0 else list(reversed(rows))
        for row in order:
            target = row + step
            if 0 <= target < self.table.rowCount():
                self._swap_rows(row, target)
        self.table.clearSelection()
        for row in rows:
            target = min(max(row + step, 0), self.table.rowCount() - 1)
            self.table.selectRow(target)
        self._set_dirty(True)
        self._restyle()

    def _to_top(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        terms = [(self.table.item(r, 0).text(), self.table.item(r, 1).text()) for r in range(self.table.rowCount())]
        picked = [terms[r] for r in rows]
        rest = [t for i, t in enumerate(terms) if i not in set(rows)]
        self._fill([{"source": s, "target": t} for s, t in picked + rest])
        self._set_dirty(True)

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入术语 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "导入术语", f"读取失败：{exc}")
            return
        if not isinstance(data, list):
            QMessageBox.warning(self, "导入术语", "JSON 必须是数组：[{\"source\": \"中文\", \"target\": \"English\"}]")
            return
        new = [{"source": str(i.get("source", "")).strip(), "target": str(i.get("target", "")).strip()}
               for i in data if isinstance(i, dict)]
        merged, added = merge_terms(self.terms(), [t for t in new if t["source"] and t["target"]])
        self._fill(merged)
        self._set_dirty(True)
        self.gen_status.setText(f"已导入，新增 {added} 条（同名的保留原写法），记得点保存")
        self._hint_duplicates()

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出术语表", "术语表.json", "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.terms(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "导出术语表", str(exc))
            return
        self.gen_status.setText(f"已导出 {len(self.terms())} 条到 {path}")

    def _clear_all(self) -> None:
        if not self.table.rowCount():
            return
        if QMessageBox.question(self, "清空术语表", "清空全部术语？点保存后才真正生效。") != QMessageBox.Yes:
            return
        self._fill([])
        self._set_dirty(True)

    def _ignored(self) -> set[str]:
        return set(self.settings.get("glossary_dedupe_ignored", []) or [])

    def open_dedupe(self, *, required: bool = False) -> bool:
        """打开查重。返回 True 表示处理完并已放回表格。"""
        dialog = self._dedupe_dialog(self.terms())
        if dialog.group_count == 0 and not required:
            self.gen_status.setText("✅ 没有发现重复或相近的术语")
            return True
        if dialog.exec() != QDialog.Accepted:
            return False
        self._fill(dialog.result_terms())
        self._set_dirty(True)
        if dialog.new_ignored:
            known = list(self.settings.get("glossary_dedupe_ignored", []) or [])
            self.settings.set("glossary_dedupe_ignored", known + [k for k in dialog.new_ignored if k not in known])
        self.gen_status.setText("查重已处理，核对后点「保存」生效")
        return True

    def _dedupe_dialog(self, terms):   # 单独成方法：测试里替换掉
        return TermDedupeDialog(self, terms, self._ignored())

    def _hint_duplicates(self) -> None:
        try:
            groups = find_term_groups(self.terms(), self._ignored())
        except Exception:  # noqa: BLE001
            return
        if groups:
            must = sum(1 for g in groups if g.must_merge)
            extra = f"（其中 {must} 组中文相同，保存前必须合并）" if must else ""
            self.gen_status.setText(self.gen_status.text() + f"  ·  发现 {len(groups)} 组可能重复{extra}，点「🔍 查重」核对")

    def save(self) -> None:
        terms = self.terms()
        if same_source_conflicts(terms):
            QMessageBox.information(
                self, "中文重复", "有中文说法相同但英文不同的术语，翻译时会不知道用哪个。先合并成一条再保存。")
            if not self.open_dedupe(required=True):
                return
            terms = self.terms()
            if same_source_conflicts(terms):
                return
        self.settings.set_glossary(terms)
        self._set_dirty(False)
        self._restyle()
        self.gen_status.setText(f"已保存 {len(terms)} 条，识别和翻译马上按新术语表来")
        if self.on_saved is not None:
            try:
                self.on_saved()
            except Exception:  # noqa: BLE001
                pass

    def closeEvent(self, event) -> None:
        if self._dirty:
            answer = QMessageBox.question(self, "还没保存", "术语表改过还没保存，要保存吗？",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                          QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.save()
            else:
                self._set_dirty(False)
        self._cancel.set()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        self.reload()
        super().showEvent(event)

    # ------------------------------------------------------------ 自动生成
    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择论文 / PPT", "", "论文和 PPT (*.pptx *.pdf *.docx *.txt *.md);;所有文件 (*)")
        if paths:
            self._extra_files = list(paths)
            self.files_label.setText("已选：" + "、".join(Path(p).name for p in paths))

    def _collect_sources(self) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []
        if self.use_thesis_check.isChecked():
            text = ThesisStore(self.settings.base_dir).full_text()
            if text.strip():
                sources.append(("论文", text))
        for path in self._extra_files:
            name = Path(path).name
            if Path(path).suffix.lower() == ".pptx":
                text, _count = extract_document_text(Path(path))
            else:
                text = "\n".join(page.text for page in extract_thesis_pages(Path(path)))
            if text.strip():
                sources.append((name, text))
        return sources

    def _default_post(self, *, model=None):
        translator = ContextTranslator(
            api_key=self.settings.get_api_key(), workspace_id=self.settings.workspace_id,
            model=model or str(self.model_combo.currentData() or "qwen3.8-flash"),
            proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode, timeout=90,
        )
        cancel = self._cancel
        return lambda payload: translator.complete(payload, cancel_event=cancel, deadline_seconds=240)

    def generate(self) -> None:
        if self._audit_running:
            return
        try:
            sources = self._collect_sources()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取文件失败", str(exc))
            return
        if not sources:
            QMessageBox.information(self, "自动生成术语", "先勾选已导入的论文，或者点「再加论文 / PPT 文件…」选文件。")
            return
        self._cancel = threading.Event()
        cancel = self._cancel
        try:
            post = self.post_factory() if self.post_factory else self._default_post()
        except ApiError as exc:
            QMessageBox.warning(self, "自动生成术语", str(exc))
            return
        chars = sum(len(text) for _l, text in sources)
        model = str(self.model_combo.currentData() or "qwen3.8-flash")
        existing = self.terms()
        self.generate_button.setEnabled(False)
        self.audit_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.gen_status.setText(f"准备生成：共约 {chars // 1000} 千字…")

        def worker() -> None:
            try:
                result = generate_terms(sources, existing, model=model, post=post,
                                        on_progress=self.gen_progress.emit, cancel_event=cancel)
            except Exception as exc:  # noqa: BLE001
                self.gen_done.emit(None, " ".join(str(exc).split())[:160])
                return
            self.gen_done.emit(result, "")

        threading.Thread(target=worker, name="glossary-generate", daemon=True).start()

    def _on_generated(self, result, error: str) -> None:
        self.generate_button.setEnabled(True)
        self.audit_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if result is None:
            self.gen_status.setText(f"生成失败：{error}")
            return
        if not (result.new or result.conflicts):
            self.gen_status.setText(f"没有找到新术语（已有的 {len(result.same)} 条都在表里了）"
                                    + ("；已停止" if self._cancel.is_set() else ""))
            return
        dialog = self._review_dialog(result)
        if dialog.exec() != QDialog.Accepted:
            self.gen_status.setText("这次生成的术语都没加入")
            return
        chosen = dialog.chosen()
        current = self.terms()
        index = {t["source"]: i for i, t in enumerate(current)}
        added = changed = 0
        for term in chosen:
            if term["source"] in index:
                current[index[term["source"]]]["target"] = term["target"]
                changed += 1
            else:
                current.append(term)
                added += 1
        self._fill(current)
        self._set_dirty(True)
        self.gen_status.setText(f"新增 {added} 条、改写 {changed} 条，核对后点「保存」生效"
                                + (f"（有 {result.failed_chunks} 块失败，可再生成一次）" if result.failed_chunks else ""))
        self._hint_duplicates()

    def _review_dialog(self, result):   # 单独成方法：测试里替换掉
        return GlossaryReviewDialog(self, result)


class _VoiceDeleteRelay(QObject):
    done = Signal(object, str)     # voice_id, 错误（空 = 成功）


class VoiceLibraryDialog(QDialog):
    """音色库管理：改名、删除（只从本机音色库删 / 连阿里云上的一起删）、设为现场或提前渲染用。

    音色库只存在本机配置里（发给同学的包里没有），同学复刻了自己的音色后也能在这里清理。
    删掉的音色如果正被「现场发声」或「提前渲染」使用，会自动换成同模型的另一个音色（没有就清空），
    不会留下一个指向已删除音色的设置。
    """

    def __init__(self, parent: QWidget, settings: DefenseSettings) -> None:
        super().__init__(parent)
        enable_window_buttons(self)
        self.settings = settings
        self.changed = False
        self.client_factory = self._default_client     # 测试可替换
        self.setWindowTitle("音色库管理")
        self.resize(1000, 560)
        layout = QVBoxLayout(self)
        intro = QLabel("音色库只保存在这台电脑上。删除分两种：<b>只从本机删</b>（云上还在，以后可以用「＋」加回来）；"
                       "<b>连阿里云上一起删</b>（彻底删掉、不可恢复，释放复刻配额）。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["名称（双击可改）", "绑定模型", "voice_id", "创建时间", "正在使用"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for col, width in enumerate((240, 230, 260, 130, 120)):
            self.table.setColumnWidth(col, width)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_renamed)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, slot, tip in (
            ("设为现场发声", lambda: self._assign("live"), "现场发声的模型也会跟着换成这个音色绑定的模型"),
            ("设为提前渲染", lambda: self._assign("prerender"), "讲稿台 / 问答库 / 快捷用语用它"),
            ("🗑 只从本机删", lambda: self.delete_selected(remote=False), "云上的音色还在"),
            ("🗑 连阿里云上一起删…", lambda: self.delete_selected(remote=True), "彻底删除，不可恢复"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.status_label)
        self.relay = _VoiceDeleteRelay()
        self.relay.done.connect(self._on_remote_deleted)
        self._loading = False
        self.refresh()

    def _entries(self) -> list[dict]:
        entries = self.settings.get("voice_library", []) or []
        return [e for e in entries if isinstance(e, dict) and str(e.get("voice_id", "")).strip()]

    def refresh(self) -> None:
        self._loading = True
        entries = self._entries()
        live = str(self.settings.get("tts_voice_id", "") or "")
        pre = str(self.settings.get("prerender_voice_id", "") or "")
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            voice_id = str(entry.get("voice_id"))
            used = "、".join(label for label, value in (("现场发声", live), ("提前渲染", pre)) if value == voice_id)
            values = [str(entry.get("name", "") or "未命名"), str(entry.get("target_model", "") or ""),
                      voice_id, str(entry.get("created_at", "") or ""), used or "—"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col != 0:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setData(Qt.UserRole, voice_id)
                if col == 4 and used:
                    item.setForeground(QColor(OK_GREEN))
                self.table.setItem(row, col, item)
        self._loading = False
        self.status_label.setText(f"共 {len(entries)} 个音色")

    def _selected_ids(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [str(self.table.item(r, 0).data(Qt.UserRole)) for r in rows if self.table.item(r, 0)]

    def _on_renamed(self, item) -> None:
        if self._loading or item.column() != 0:
            return
        voice_id = str(item.data(Qt.UserRole))
        name = item.text().strip()
        entries = self.settings.get("voice_library", []) or []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("voice_id") == voice_id and name:
                entry["name"] = name
        self.settings.set("voice_library", entries)
        self.changed = True

    def _assign(self, which: str) -> None:
        ids = self._selected_ids()
        if len(ids) != 1:
            self.status_label.setText("请选中一个音色")
            return
        entry = next(e for e in self._entries() if e.get("voice_id") == ids[0])
        model = str(entry.get("target_model", "") or "")
        if which == "live":
            self.settings.update({"tts_voice_id": ids[0], **({"tts_model": model} if model else {})})
        else:
            self.settings.update({"prerender_voice_id": ids[0], "prerender_tts_model": model})
        self.changed = True
        self.refresh()
        self.status_label.setText(f"已设为{'现场发声' if which == 'live' else '提前渲染'}：{entry.get('name', ids[0])}")

    def _replacement(self, removed: set[str], model: str) -> str:
        """被删的音色正在用时，换成同一模型里最新的另一个音色。"""
        candidates = [e for e in self._entries()
                      if e.get("voice_id") not in removed and str(e.get("target_model", "")) == model]
        candidates.sort(key=lambda e: str(e.get("created_at", "")), reverse=True)
        return str(candidates[0]["voice_id"]) if candidates else ""

    def _remove_local(self, ids: set[str]) -> list[str]:
        """从本机音色库删，并修好指向它们的设置。返回提示文字。"""
        notes: list[str] = []
        entries = self._entries()
        models = {e["voice_id"]: str(e.get("target_model", "")) for e in entries}
        live = str(self.settings.get("tts_voice_id", "") or "")
        pre = str(self.settings.get("prerender_voice_id", "") or "")
        updates: dict = {}
        if live in ids:
            new = self._replacement(ids, models.get(live, str(self.settings.get("tts_model", ""))))
            updates["tts_voice_id"] = new
            notes.append(f"现场发声已换成 {new}" if new else "现场发声的音色已清空，请重新选一个")
        if pre in ids:
            new = self._replacement(ids, models.get(pre, ""))
            updates.update({"prerender_voice_id": new, "prerender_tts_model": models.get(pre, "") if new else ""})
            notes.append("提前渲染已换成同模型的另一个音色" if new else "提前渲染改回「和现场发声相同」")
        last_b = self.settings.get("voice_ab_last_b", {}) or {}
        if isinstance(last_b, dict) and last_b.get("voice_id") in ids:
            updates["voice_ab_last_b"] = {}
        updates["voice_library"] = [e for e in (self.settings.get("voice_library", []) or [])
                                    if not (isinstance(e, dict) and e.get("voice_id") in ids)]
        self.settings.update(updates)
        self.changed = True
        return notes

    def delete_selected(self, *, remote: bool) -> None:
        ids = self._selected_ids()
        if not ids:
            self.status_label.setText("先在表里选中要删的音色（可以多选）")
            return
        entries = {e["voice_id"]: e for e in self._entries()}
        names = "、".join(str(entries[i].get("name", i)) for i in ids if i in entries)
        if remote:
            text = (f"彻底删除 {len(ids)} 个音色：{names}？\n\n会同时在阿里云上删除，删了不能恢复，"
                    "所有用这个音色的地方（包括别的电脑）都不能再用。")
        else:
            text = f"从本机音色库删除 {len(ids)} 个音色：{names}？\n（阿里云上的音色还在，以后可以用「＋」按 voice_id 加回来。）"
        if self._confirm(text) is not True:
            return
        if not remote:
            notes = self._remove_local(set(ids))
            self.refresh()
            self.status_label.setText(f"已从本机删除 {len(ids)} 个音色。" + " ".join(notes))
            return
        try:
            client = self.client_factory()
        except ApiError as exc:
            self.status_label.setText(f"没法连接阿里云：{exc}")
            return
        self.status_label.setText(f"正在阿里云上删除 {len(ids)} 个音色…")
        relay = self.relay
        jobs = [(i, str(entries.get(i, {}).get("target_model", ""))) for i in ids]

        def worker() -> None:
            for voice_id, model in jobs:
                try:
                    client.delete_voice(voice_id, model)
                    relay.done.emit(voice_id, "")
                except Exception as exc:  # noqa: BLE001
                    relay.done.emit(voice_id, " ".join(str(exc).split())[:160])

        threading.Thread(target=worker, name="voice-delete", daemon=True).start()

    def _on_remote_deleted(self, voice_id, error: str) -> None:
        if error:
            self.status_label.setText(f"{voice_id} 在阿里云上没删成：{error}（本机音色库里保留，免得找不回来）")
            self.status_label.setStyleSheet(f"color: {ERR_RED};")
            return
        notes = self._remove_local({voice_id})
        self.refresh()
        self.status_label.setText(f"已彻底删除 {voice_id}。" + " ".join(notes))
        self.status_label.setStyleSheet(f"color: {OK_GREEN};")

    def _confirm(self, text: str) -> bool:   # 单独成方法：测试里替换掉
        return QMessageBox.question(self, "删除音色", text) == QMessageBox.Yes

    def _default_client(self):
        return BailianClient(self.settings.get_api_key(), self.settings.workspace_id,
                             proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode)


class SettingsDialog(QDialog):
    device_test_done = Signal(str, bool)
    test_line = Signal(str)
    test_done = Signal()
    connection_event = Signal(int, str, bool)
    CONNECTION_TIMEOUT_MS = 60_000
    ambience_preview_done = Signal(str)   # 试听背景放完 / 出错（错误文字）
    channel_test_done = Signal(str)       # 双通道测试结果

    def __init__(
        self, parent: QWidget, settings: DefenseSettings, *, active_loopback_name: str = ""
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.active_loopback_name = active_loopback_name
        self._show_all_devices = False  # 每次打开先精简；完整接口只用于本次排查。
        from .. import __version__

        self.setWindowTitle(f"设置 · 留子答辩器 v{__version__}")
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
        self.test_done.connect(self._finish_connection_test)
        self.connection_event.connect(self._on_connection_event, Qt.QueuedConnection)
        self._connection_serial = 0
        self._connection_running = False
        self._connection_cancel = threading.Event()
        self._connection_watchdog = QTimer(self)
        self._connection_watchdog.setSingleShot(True)
        self._connection_watchdog.timeout.connect(self._connection_timeout)
        self.finished.connect(self._cancel_connection_test)
        self._test_lines: list[str] = []
        self.workspace_edit.textChanged.connect(lambda _text: self._update_credential_hints())
        self.api_key_edit.textChanged.connect(lambda _text: self._update_credential_hints())
        self._update_credential_hints()
        left_form.addRow("双通道（第二候选，防止某个接口卡住）", self._build_channel_box())

        # ---- 两套「模型 + 音色」分开装框、分颜色：蓝 = 现场发声（要快），琥珀 = 提前渲染（要好听）----
        def voice_frame(title: str, subtitle: str, color: str, background: str):
            frame = QFrame()
            frame.setObjectName("voiceFrame")
            frame.setStyleSheet(
                f"QFrame#voiceFrame {{ background: {background}; border: 1px solid {color}; border-radius: 8px; }}"
                "QFrame#voiceFrame QLabel { background: transparent; border: none; }")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 8, 12, 10)
            head = QLabel(f"<b style='color:{color}'>{title}</b>　<span style='color:{TEXT_DIM}'>{subtitle}</span>")
            box.addWidget(head)
            form = QFormLayout()
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            box.addLayout(form)
            return frame, form

        live_frame, live_form = voice_frame(
            "🎤 现场发声", "答辩实时翻译、助答现场生成用；要快，建议 flash 模型", "#4f9cf0", "#0f1d33")
        self.voice_combo = NoWheelComboBox()
        self.voice_combo.setMaxVisibleItems(12)
        self._refresh_voice_combo()
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        self.clone_voice_button = QPushButton("一键克隆音色…")
        self.clone_voice_button.clicked.connect(self._open_clone_voice)
        voice_row.addWidget(self.clone_voice_button)
        add_voice_button = QPushButton("＋")
        add_voice_button.setFixedWidth(34)
        add_voice_button.setToolTip("手动添加控制台复刻的 voice_id 到音色库")
        add_voice_button.clicked.connect(self._add_manual_voice)
        voice_row.addWidget(add_voice_button)
        library_button = QPushButton("🗂 音色库 / 删除…")
        library_button.setToolTip("改名、删除音色（只从本机删，或连阿里云上一起删）、设为现场或提前渲染用")
        library_button.clicked.connect(self._open_voice_library)
        voice_row.addWidget(library_button)

        self.tts_model_combo = NoWheelComboBox()
        fill_model_combo(self.tts_model_combo, TTS_MODELS, str(settings.get("tts_model")), RECOMMENDED_TTS)
        self.tts_model_combo.setToolTip(
            "普通 TTS 要选绑定到模型的音色；Next 另选本机参考录音，不使用 voice_id。"
        )
        live_form.addRow("TTS 模型", self.tts_model_combo)
        live_form.addRow("克隆音色（只列和上面模型匹配的，其余灰色）", voice_row)
        self.voice_hint = QLabel("")
        self.voice_hint.setWordWrap(True)
        live_form.addRow("", self.voice_hint)
        self.tts_model_combo.currentIndexChanged.connect(lambda _index: self._on_tts_model_changed())
        self.reference_audio_edit = QLineEdit(str(settings.get("tts_reference_audio", "") or ""))
        self.reference_audio_edit.setPlaceholderText("选择你自己的 30 秒以内 WAV/MP3/OGG 录音")
        self.reference_audio_button = QPushButton("选择…")
        self.reference_audio_button.clicked.connect(self._browse_reference_audio)
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_audio_edit, 1)
        reference_row.addWidget(self.reference_audio_button)
        self.reference_audio_box = QWidget()
        self.reference_audio_box.setLayout(reference_row)
        live_form.addRow("Next 参考录音", self.reference_audio_box)
        self.reference_audio_label = live_form.labelForField(self.reference_audio_box)
        self._update_reference_audio_ui()
        self.voice_combo.currentIndexChanged.connect(lambda _index: self._update_voice_hint())
        self._update_voice_hint()

        # 模型能力一览：换模型时哪些自然度手段可用一眼能看到，
        # 不用等到发现语气指令是灰的才回头猜原因。
        self.capability_label = QLabel("")
        self.capability_label.setWordWrap(True)
        self.capability_label.setToolTip(
            "这一行只说明当前模型「支不支持」，不代表你已经开启。开关在下面各项里。"
        )
        live_form.addRow("模型能力", self.capability_label)
        left_form.addRow(live_frame)

        pre_frame, pre_form = voice_frame(
            "📼 提前渲染", "讲稿台、问答库、快捷用语提前合成用；不赶时间，建议 plus 模型", "#e0a43a", "#2a2010")
        self.prerender_model_combo = NoWheelComboBox()
        current_pre_model = str(settings.get("prerender_tts_model", "") or "")
        fill_model_combo(self.prerender_model_combo, TTS_MODELS, current_pre_model, RECOMMENDED_TTS)
        self.prerender_model_combo.insertItem(0, "和现场发声相同（不单独设置）", "")
        self.prerender_model_combo.setCurrentIndex(max(0, self.prerender_model_combo.findData(current_pre_model)))
        pre_form.addRow("TTS 模型", self.prerender_model_combo)
        self.prerender_voice_combo = NoWheelComboBox()
        self.prerender_voice_combo.setToolTip(
            "讲稿台、问答库答案、快捷用语提前渲染时用的音色。建议：上面「现场发声」选 flash 模型的音色（快），"
            "这里选 plus 模型的音色（更好听，反正提前渲染不赶时间）。音色库里要有对应模型复刻的音色。"
        )
        self._refresh_prerender_voice_combo(str(settings.get("prerender_voice_id", "") or ""))
        pre_form.addRow("克隆音色（只列和上面模型匹配的，其余灰色）", self.prerender_voice_combo)
        self.prerender_hint = QLabel("")
        self.prerender_hint.setWordWrap(True)
        pre_form.addRow("", self.prerender_hint)
        self.prerender_model_combo.currentIndexChanged.connect(lambda _i: self._on_prerender_model_changed())
        self.prerender_voice_combo.currentIndexChanged.connect(lambda _i: self._on_prerender_voice_changed())
        self._update_prerender_hint()
        left_form.addRow(pre_frame)
        self._voice_frames = (live_frame, pre_frame)
        self.tts_model_combo.currentIndexChanged.connect(self._update_capability_label)

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
            "Qwen 复刻音色留空会保留原声口音；也可输入自定义语气（CosyVoice 限 100 字符单位）"
        )
        left_form.addRow("语气指令", self.instruction_edit)
        self.instruction_note = QLabel("")
        self.instruction_note.setWordWrap(True)
        left_form.addRow("", self.instruction_note)
        self.instruction_preset_combo.currentIndexChanged.connect(self._apply_instruction_preset)
        self.instruction_edit.textEdited.connect(self._mark_instruction_custom)
        self.tts_model_combo.currentIndexChanged.connect(self._update_instruction_support)
        self._update_instruction_support()

        # ---- 句间换气：用 SSML <break> 显式控制，而不是让模型看标点猜 ----
        self.sentence_break_spin = NoWheelSpinBox()
        self.sentence_break_spin.setRange(0, 1000)
        self.sentence_break_spin.setSingleStep(20)
        self.sentence_break_spin.setSuffix(" ms")
        self.sentence_break_spin.setSpecialValueText("0 ms（交给模型决定）")
        self.sentence_break_spin.setValue(int(settings.get("sentence_break_ms", 0) or 0))
        self.sentence_break_spin.setToolTip(
            "一段回答里句与句之间的换气长度。0 表示保持原有行为，由模型按标点自己决定。"
            "想要更连贯可以试 140~180 ms；超过 300 ms 就不像换气而像冷场了。"
        )
        left_form.addRow("句间换气", self.sentence_break_spin)
        self.sentence_break_note = QLabel("")
        self.sentence_break_note.setWordWrap(True)
        left_form.addRow("", self.sentence_break_note)

        # ---- 开场衔接词：默认关闭，因为它会替你多说一个词 ----
        self.filler_combo = NoWheelComboBox()
        self.filler_combo.addItem("关闭（只朗读译文本身）", "off")
        self.filler_combo.addItem("自动加开场衔接词（Right, so — / Sure. …）", "auto")
        filler_mode = str(settings.get("opening_filler", "off") or "off")
        self.filler_combo.setCurrentIndex(max(0, self.filler_combo.findData(filler_mode)))
        self.filler_combo.setToolTip(
            "真人被问到问题不会在 0.2 秒后吐出语法完美的长句。开启后会在每段回答前面"
            "随机加一个短衔接词，和正文一起合成。注意：这是你没说过的词。"
        )
        left_form.addRow("开场衔接词", self.filler_combo)
        filler_note = QLabel(
            "默认关闭。开启等于让软件替你多说一个词；字幕会照实显示它，朗读前你能看到。"
        )
        filler_note.setWordWrap(True)
        filler_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", filler_note)

        # ---- 像真人的小失误：偶尔卡壳 / 重读 / 说半句重来（单独开关，默认关）----
        self.humanize_combo = NoWheelComboBox()
        for value, label in HUMANIZE_CHOICES:
            self.humanize_combo.addItem(label, value)
        self.humanize_combo.setCurrentIndex(max(0, self.humanize_combo.findData(
            str(settings.get("humanize_level", "off") or "off"))))
        self.humanize_combo.setToolTip(
            "像 GPT 语音那样偶尔出一点真人才有的小失误，例如：\n"
            "  The... the model we proposed\n  We used a... we used a transformer\n"
            "  The results, um, show that\n  in the... sorry, in the second chapter\n"
            "只改读出来的声音，字幕和记录还是原句；数字、缩写、术语表里的词绝不重复或打断；"
            "一段话最多一处，现场两处之间至少隔 12 秒。"
        )
        left_form.addRow("像真人的小失误", self.humanize_combo)
        humanize_note = QLabel(
            "影响 F9 现场翻译的声音。提前渲染选「跟随设置」时，也用于讲稿、问答库和快捷用语；"
            "A 原声模式及 Qwen 的 B 自然模式保留原文，自己录的快捷用语不受影响。"
            "已有音频不会自动变；改变提前渲染选项后，需重做相应条目才会生效。"
        )
        humanize_note.setWordWrap(True)
        humanize_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", humanize_note)

        self.translation_style_combo = NoWheelComboBox()
        for style_id, style_label, _rule, _hint in TRANSLATION_STYLES:
            self.translation_style_combo.addItem(style_label, style_id)
        style = str(settings.get("translation_style", "accurate") or "accurate")
        self.translation_style_combo.setCurrentIndex(max(0, self.translation_style_combo.findData(style)))
        self.translation_style_combo.setToolTip(
            "影响两处：① F9 现场翻译（你说中文 → 读出来的英文）；② 讲稿台「中文稿翻成英文」。\n"
            "不影响：评委字幕、问答库 / 助答生成的回答、已经渲染好的声音（讲稿要重新翻译、重新渲染才会变）。\n"
            "所有口吻都不会改数字、术语和结论。若使用千问同传，译文由同传模型直接给出，此选项暂不生效。"
        )
        left_form.addRow("我的英文口吻", self.translation_style_combo)
        style_note = QLabel(
            "口吻管的是「翻成什么样的英文」：影响 F9 现场翻译，和讲稿台「中文稿翻成英文」（翻过的讲稿要重新翻、重新渲染才变）。"
            "讲稿台读出来的声音语气在讲稿台「语气」里单独选。"
            "想听到停顿和轻微呼吸感，可同时选上面的“轻微犹豫”语气风格，"
            "并在支持 SSML 的模型上把句间换气设为约 160–220 ms。"
            "真实吸气声由模型决定，无法保证每次都有。"
        )
        style_note.setWordWrap(True)
        style_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", style_note)

        self.tts_model_combo.currentIndexChanged.connect(self._update_ssml_support)
        self._update_ssml_support()
        self._update_capability_label()

        self.ambience_combo = NoWheelComboBox()
        self.ambience_combo.setMaxVisibleItems(20)
        for key, label in BACKGROUND_CHOICES:
            self.ambience_combo.addItem(label,key)
        ambience_mode = background_mode(settings.get("ambience_mode", "off"))
        self.ambience_combo.setCurrentIndex(max(0, self.ambience_combo.findData(ambience_mode)))
        left_form.addRow("背景环境", self.ambience_combo)

        self.ambience_volume_slider = NoWheelSlider(Qt.Horizontal)
        self.ambience_volume_slider.setRange(0, 100)
        self.ambience_volume_slider.setValue(int(settings.get("ambience_volume", AMBIENCE_VOLUME_DEFAULT) or 0))
        self.ambience_volume_label = QLabel("")
        self.ambience_preview_button = QPushButton("▶ 试听 6 秒")
        self.ambience_preview_button.setToolTip("只在你的试听设备里放，不会传给会议")
        self.ambience_preview_button.clicked.connect(self._preview_ambience)
        volume_row = QHBoxLayout()
        volume_row.addWidget(self.ambience_volume_slider, 1)
        volume_row.addWidget(self.ambience_volume_label)
        volume_row.addWidget(self.ambience_preview_button)
        left_form.addRow("背景音量", volume_row)
        self.ambience_preview_label = QLabel("")
        self.ambience_preview_label.setWordWrap(True)
        self.ambience_preview_label.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", self.ambience_preview_label)
        self._ambience_preview_stop = threading.Event()
        self._ambience_previewing = False
        self.ambience_preview_done.connect(self._on_ambience_preview_done)
        self.ambience_volume_slider.valueChanged.connect(lambda _v: self._update_ambience_label())
        self.ambience_combo.currentIndexChanged.connect(lambda _i: self._update_ambience_label())
        self._update_ambience_label()
        ambience_note = QLabel(
            "背景使用内置录音循环播放，不再生成白噪声。咖啡馆录音包含背景谈话，办公室包含电话铃；正式答辩可关闭背景声。"
            f"音量有硬上限：无论哪个场景、滑块拉多大，背景都不超过约 {AMBIENCE_MAX_DB:.0f} dB，"
            "会议软件强降噪时可能听不见背景声。"
        )
        ambience_note.setWordWrap(True)
        ambience_note.setStyleSheet(f"color: {TEXT_DIM};")
        left_form.addRow("", ambience_note)
        self.speech_mode_combo = NoWheelComboBox()
        self.speech_mode_combo.addItem("自然度优先 · 完整翻译 + 音频缓冲", "natural")
        self.speech_mode_combo.addItem("低延迟 · 分句流式发声", "streaming")
        self.speech_mode_combo.setCurrentIndex(max(0, self.speech_mode_combo.findData(settings.get("speech_mode", "natural"))))
        left_form.addRow("发声方式", self.speech_mode_combo)

        # ---- 语音缓存用量：一小时的会到底占多少，要能一眼看见 ----
        self.cache_usage_label = QLabel("")
        self.cache_usage_label.setWordWrap(True)
        left_form.addRow("语音缓存", self.cache_usage_label)
        cache_row = QHBoxLayout()
        refresh_cache_button = QPushButton("刷新用量")
        refresh_cache_button.clicked.connect(self._refresh_cache_usage)
        cache_row.addWidget(refresh_cache_button)
        clear_cache_button = QPushButton("清理缓存")
        clear_cache_button.setToolTip(
            "删除实时翻译产生的语音缓存。**讲稿朗读台预渲染的音频默认保留**，"
            "否则现场断网就讲不了了。"
        )
        clear_cache_button.clicked.connect(self._clear_cache_clicked)
        cache_row.addWidget(clear_cache_button)
        cache_row.addStretch(1)
        left_form.addRow("", cache_row)
        self.disk_cache_spin = NoWheelSpinBox()
        self.disk_cache_spin.setRange(64, 20480)
        self.disk_cache_spin.setSingleStep(256)
        self.disk_cache_spin.setSuffix(" MB")
        self.disk_cache_spin.setValue(int(settings.get("disk_cache_mb", 1024) or 1024))
        self.disk_cache_spin.setToolTip(
            "磁盘缓存上限。一小时会议实际发声约 25 分钟 ≈ 72MB，讲稿 20 分钟 ≈ 58MB，"
            "1GB 绰绰有余。超限时只淘汰实时翻译的缓存，不动讲稿。"
        )
        left_form.addRow("磁盘缓存上限", self.disk_cache_spin)
        self._refresh_cache_usage()
        self.early_start_check = QCheckBox("说完第一句就开口（早开口，推荐）")
        self.early_start_check.setChecked(bool(settings.get("speech_early_start", True)))
        self.early_start_check.setToolTip(
            "自然模式原来要等整段译文出完、整段合成完才出声。\n"
            "打开后译文第一句（至少 5 个词）一完整就先合成播放，后面的边播边合成，开口快 1~3 秒。"
        )
        left_form.addRow("", self.early_start_check)
        mode_note = QLabel("自然模式：译文第一句好了就开口（可关）；断句停顿按下面「静音判定」的设置（最低 400 ms）。"
                           "Qwen VC 不支持语气指令，语速/音调先保持 1.0。")
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
        fill_model_combo(self.llm_combo, TRANSLATE_MODELS, str(settings.get("llm_model")), RECOMMENDED_TRANSLATE)
        form.addRow("翻译大模型", self.llm_combo)

        # 情景模式：和首页右上角那个是同一套。这里选一个就把下面几项填好；
        # 手动改了其中任何一项，这里自动变成「自定义」。保存后首页跟着变。
        self.scene_combo = NoWheelComboBox()
        self.scene_combo.setToolTip("和首页右上角的情景模式是同一个设置：选一个就把识别 / 翻译 / 助答模型一起填好")
        right_form.addRow("情景模式（一键填好下面的模型）", self.scene_combo)

        self.asr_combo = NoWheelComboBox()
        fill_model_combo(self.asr_combo, ASR_MODELS, str(settings.get("asr_model")), RECOMMENDED_ASR)
        right_form.addRow("我方识别模型", self.asr_combo)

        self.committee_asr_combo = NoWheelComboBox()
        fill_model_combo(self.committee_asr_combo, ASR_MODELS, str(settings.get("committee_asr_model")), RECOMMENDED_ASR)
        right_form.addRow("评委识别模型", self.committee_asr_combo)

        self.qa_model_combo = NoWheelComboBox()
        fill_model_combo(self.qa_model_combo, QA_MODELS, str(settings.get("qa_model", "qwen3.8-flash")), RECOMMENDED_QA)
        right_form.addRow("助答现场生成模型", self.qa_model_combo)
        self.qa_thinking_check = QCheckBox("助答深度思考（更准但要多等十几秒）")
        self.qa_thinking_check.setChecked(bool(settings.get("qa_thinking", False)))
        right_form.addRow("", self.qa_thinking_check)

        # 我的发言翻译方式：大模型（默认）或千问同传（测试中）
        self.engine_combo = NoWheelComboBox()
        self.engine_combo.addItem(
            "★ 分开翻译：识别 + 上面的「翻译大模型」（默认 qwen3.8-flash，带术语表和整场上下文）", "llm"
        )
        highlight_item(self.engine_combo, 0)
        for model, blurb in LIVETRANSLATE_MODELS:
            self.engine_combo.addItem(f"千问同传 {model} · {blurb}（测试中）", f"livetranslate:{model}")
        current_engine = str(settings.get("my_translate_engine", "llm") or "llm")
        current_key = (
            f"livetranslate:{settings.get('livetranslate_model', LIVETRANSLATE_MODELS[0][0])}"
            if current_engine == "livetranslate" else "llm"
        )
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(current_key)))
        self.engine_combo.setToolTip(
            "同传：边说边译，不用等一句说完；术语表会作为同传热词。\n"
            "此时上面的「我方识别模型」和「翻译大模型」对我的发言不起作用。\n"
            "译文照样用你的克隆音色播放。"
        )
        right_form.addRow("我的发言翻译方式", self.engine_combo)
        self._fill_scene_combo()
        for combo in (self.llm_combo, self.asr_combo, self.committee_asr_combo, self.qa_model_combo, self.engine_combo):
            combo.currentIndexChanged.connect(lambda _i: self._sync_scene_combo())
        self.qa_thinking_check.toggled.connect(lambda _c: self._sync_scene_combo())
        self.scene_combo.activated.connect(self._on_settings_scene_chosen)

        self.direct_hotkey_combo = self._hotkey_combo("direct_hotkey", "f8")
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
        self._all_inputs, self._all_outputs = inputs, outputs
        loopbacks = [device for device in _safe_list_loopback_devices()
                     if not is_program_output_loopback(device.name)]
        self._loopbacks_cache = loopbacks
        self.mic_combo = NoWheelComboBox()
        _device_combo(
            self.mic_combo, self._device_list("mic", settings.shared.get("input_device")),
            settings.shared.get("input_device"), allow_default="系统默认输入", direction="输入",
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
        self.cable_locked_index = self._apply_cable_lock(
            self.cable_combo, self._device_list("cable", self._saved_cable_index(outputs)))
        cable_lock_row = QHBoxLayout()
        cable_lock_row.addWidget(self.cable_lock_button)
        cable_lock_row.addWidget(self.cable_combo, 1)
        right_form.addRow("程序输出（已锁定虚拟声卡）", cable_lock_row)
        right_form.addRow("", self.cable_note)
        self.loopback_combo = NoWheelComboBox()
        self.loopback_note = QLabel("")
        self.loopback_locked = True
        apply_default_loopback(settings)
        has_saved = settings.shared.get("loopback_device") is not None or bool(
            str(settings.shared.get("loopback_device_name", "") or ""))
        chosen_loopback = choose_committee_loopback(
            loopbacks, settings.shared.get("loopback_device"),
            str(settings.shared.get("loopback_device_name", "") or ""),
        )
        # 选哪个锁哪个：有保存的就显示保存的；只有从没选过时才跟随「正在监听的设备」
        _device_combo(
            self.loopback_combo, loopbacks, chosen_loopback.index if chosen_loopback else None,
            allow_default="系统默认输出（推荐：微信/Teams 默认从这里出声）", direction="回环",
            highlight_name="" if has_saved else self.active_loopback_name,
        )
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
        monitor_index = settings.get("monitor_output_device")
        saved_monitor_name = str(settings.get("monitor_output_device_name", "") or "")
        saved_monitor_host = str(settings.get("monitor_output_host_api", "") or "")
        monitor_missing = False
        if monitor_index is not None and saved_monitor_name:
            matches = [d for d in self._all_outputs if d.name == saved_monitor_name
                       and (not saved_monitor_host or d.host_api == saved_monitor_host)]
            found = next((d for d in matches if d.index == monitor_index), None)
            found = found or (matches[0] if len(matches) == 1 else None)
            monitor_index = found.index if found else None
            monitor_missing = found is None
        _device_combo(
            self.monitor_combo, self._device_list("monitor", monitor_index),
            monitor_index, allow_default="系统默认输出", direction="输出",
        )
        self.monitor_locked = True
        self.monitor_lock_button = QPushButton("🔒 已锁定")
        self.monitor_lock_button.setFixedWidth(92)
        self.monitor_lock_button.clicked.connect(self._toggle_monitor_lock)
        monitor_lock_row = QHBoxLayout()
        monitor_lock_row.addWidget(self.monitor_lock_button)
        monitor_lock_row.addWidget(self.monitor_combo, 1)
        self.device_test_button = QPushButton("▶ 播放测试")
        self.device_test_button.setToolTip("在当前选择的试听设备播放两声提示音；无需保存，不调用云端")
        self.device_test_button.clicked.connect(self._test_monitor_device)
        monitor_lock_row.addWidget(self.device_test_button)
        right_form.addRow("试听设备", monitor_lock_row)
        self.device_test_label = QLabel("先选耳机，再点播放测试确认出声位置。测试无需保存；正式播放需点击右下角「保存」。")
        self.device_test_label.setWordWrap(True)
        if monitor_missing:
            self.device_test_label.setText(f"原试听设备已断开或无法确定：{saved_monitor_name}。请重新选择并保存；正式播放不会自动切换设备。")
        right_form.addRow("", self.device_test_label)
        self._device_testing = False
        self._device_test_stop = threading.Event()
        self.device_test_done.connect(self._on_device_test_done)
        self._apply_mic_lock_state()
        self._apply_monitor_lock_state()

        glossary_caption = QLabel("术语表已经搬到主界面顶部的「术语表」按钮（独立窗口，可以从论文和 PPT 自动生成全部术语）。")
        glossary_caption.setWordWrap(True)
        layout.addWidget(glossary_caption)
        term_buttons = QHBoxLayout()
        open_glossary = QPushButton("打开术语表…")
        open_glossary.clicked.connect(self._open_glossary_from_settings)
        term_buttons.addWidget(open_glossary)
        term_buttons.addStretch(1)
        layout.addLayout(term_buttons)
        try:
            self._reorganize_into_tabs(layout, columns, left_form, right_form, left_wrap, right_wrap,
                                       glossary_caption, term_buttons)
        except Exception as exc:  # noqa: BLE001 —— 排版失败就保持原来的两栏，不影响设置功能
            logging.getLogger(__name__).warning("设置页分页失败：%s", exc)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存")
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        outer.addLayout(buttons)

    # ------------------------------------------------------ 分页整理
    SETTINGS_TABS = (
        ("声音", ("克隆音色", "TTS 模型", "Next 参考录音", "模型能力", "提前渲染音色", "语速", "音调", "音量",
                 "语气风格", "语气指令", "句间换气", "开场衔接词", "背景环境", "发声方式")),
        ("识别与翻译", ("情景模式", "我方识别模型", "评委识别模型", "翻译大模型", "我的英文口吻",
                       "我的发言翻译方式", "助答现场生成模型", "断句静音", "上下文携带句数")),
        ("快捷键", ("原声快捷键", "翻译快捷键", "停止快捷键")),
        ("声卡", ("麦克风", "程序输出", "对方声音回环", "试听设备")),
        ("账号与网络", ("百炼 Workspace ID", "百炼 API Key", "双通道", "网络代理", "记录保存位置")),
        ("缓存", ("语音缓存", "磁盘缓存上限")),
    )
    HIDDEN_DEVICE_LABELS = ("麦克风", "程序输出", "对方声音回环", "试听设备")

    def _reorganize_into_tabs(self, layout, columns, left_form, right_form, left_wrap, right_wrap,
                              glossary_caption, term_buttons) -> None:
        """原来两大栏几十行挤在一起：按用途分成几个标签页。控件本身不变，只是换了位置。"""
        def group_of(label_text: str) -> str | None:
            for name, prefixes in self.SETTINGS_TABS:
                if any(label_text.startswith(prefix) for prefix in prefixes):
                    return name
            return None

        overrides = {id(self.early_start_check): "声音"}
        overrides.update({id(frame): "声音" for frame in getattr(self, "_voice_frames", ())})
        rows: list[tuple[str, QWidget | None, object]] = []
        for form in (left_form, right_form):
            previous = None
            while form.rowCount():
                taken = form.takeRow(0)
                label = taken.labelItem.widget() if taken.labelItem is not None else None
                item = taken.fieldItem
                field = None if item is None else (item.widget() or item.layout())
                text = label.text().strip() if isinstance(label, QLabel) else ""
                group = overrides.get(id(field)) or (group_of(text) if text else None) or previous or "声音"
                previous = group
                rows.append((group, label, field))

        tabs = QTabWidget()
        pages: dict[str, QFormLayout] = {}
        device_box = QWidget()
        device_form = QFormLayout(device_box)
        device_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        device_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for name, _prefixes in self.SETTINGS_TABS:
            page = QWidget()
            form = QFormLayout(page)
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            pages[name] = form
            if name == "声卡":
                self.show_devices_check = QCheckBox(
                    "显示全部接口（排查用：同一设备会重复出现；取消勾选恢复精简列表）")
                self.show_devices_check.setChecked(False)
                form.addRow(self.show_devices_check)
                form.addRow(device_box)
                self.device_box = device_box
            tabs.addTab(page, name)
        in_device_box = False
        for group, label, field in rows:
            text = label.text().strip() if isinstance(label, QLabel) else ""
            target = pages[group]
            if group == "声卡":
                if text:
                    in_device_box = any(text.startswith(prefix) for prefix in self.HIDDEN_DEVICE_LABELS)
                if field is self.monitor_check:
                    in_device_box = False
                if in_device_box:
                    target = device_form
            if field is None:
                continue
            if label is not None:
                target.addRow(label, field)
            else:
                target.addRow(field)
        self.show_devices_check.toggled.connect(self._on_show_devices_toggled)
        device_box.setVisible(True)   # 四个声卡选择一直显示；筛掉的是下拉里没用的条目

        glossary_page = QWidget()
        glossary_layout = QVBoxLayout(glossary_page)
        layout.removeWidget(glossary_caption)
        glossary_layout.addWidget(glossary_caption)
        layout.removeItem(term_buttons)
        glossary_layout.addLayout(term_buttons)
        glossary_layout.addStretch(1)
        tabs.addTab(glossary_page, "术语表")

        layout.removeItem(columns)
        for wrap in (left_wrap, right_wrap):
            wrap.hide()
            wrap.setParent(None)
            wrap.deleteLater()
        layout.insertWidget(0, tabs, 1)
        self.settings_tabs = tabs

    def _update_ambience_label(self) -> None:
        mode = str(self.ambience_combo.currentData() or "off")
        volume = int(self.ambience_volume_slider.value())
        self.ambience_volume_slider.setEnabled(mode != "off")
        self.ambience_preview_button.setEnabled(mode != "off")
        if mode == "off":
            self.ambience_volume_label.setText("（已关闭）")
            self.ambience_volume_label.setStyleSheet(f"color: {TEXT_DIM};")
            return
        level = ambience_level_db(mode, volume)
        default = "（场景默认）" if volume == AMBIENCE_VOLUME_DEFAULT else ""
        if ambience_at_cap(mode, volume):
            self.ambience_volume_label.setText(f"{volume}%  约 {level:.0f} dB · 已到上限，再大会影响评委听清")
            self.ambience_volume_label.setStyleSheet(f"color: {WARN_ORANGE};")
        else:
            self.ambience_volume_label.setText(f"{volume}%{default}  约 {level:.0f} dB（上限 {AMBIENCE_MAX_DB:.0f} dB）")
            self.ambience_volume_label.setStyleSheet(f"color: {TEXT_DIM};")

    def _reference_voice(self, seconds: float = 5.0) -> bytes:
        """找一句你渲染过的声音（讲稿 / 预案，最近的那句）叠在背景上，才听得出背景和人声的真实比例。"""
        root = Path(self.settings.base_dir) / "tts_cache" / "pinned"
        try:
            files = sorted(root.rglob("*.pcm"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return b""
        for path in files[:20]:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) >= SPEECH_SAMPLE_RATE * 2 and len(data) % 2 == 0:
                return data[:int(SPEECH_SAMPLE_RATE * seconds) * 2]
        return b""

    def _build_ambience_preview(self, mode: str, volume: int) -> tuple[bytes, bool]:
        """(试听音频, 是否叠了人声)。6 秒背景（场景事件前几秒各来一次），第 1.5 秒起叠一句你的声音。"""
        tone = make_room_tone(mode, sample_rate=SPEECH_SAMPLE_RATE, volume=volume)
        bed = bytearray(tone.demo(SPEECH_SAMPLE_RATE * 6))
        voice = self._reference_voice()
        if voice:
            from array import array
            start = int(SPEECH_SAMPLE_RATE * 1.5) * 2
            mixed = array("h", bytes(bed[start:start + len(voice)]).ljust(len(voice), b"\0"))
            speech = array("h", voice)
            for i in range(len(speech)):
                mixed[i] = max(-32768, min(32767, mixed[i] + speech[i]))
            bed[start:start + len(voice)] = mixed.tobytes()
        return bytes(bed) + tone.tail(800), bool(voice)

    def _preview_ambience(self) -> None:
        """在试听设备上放一段当前场景 + 音量（不进会议）。再点一次停止。"""
        from ..audio import MultiOutputPlayer
        from .rehearsal import play_pcm

        if self._ambience_previewing:
            self._ambience_preview_stop.set()
            return
        mode = str(self.ambience_combo.currentData() or "off")
        if mode == "off":
            return
        try:
            pcm, with_voice = self._build_ambience_preview(mode, int(self.ambience_volume_slider.value()))
        except Exception as exc:
            self.ambience_preview_label.setText(f"背景录音无法读取：{exc}")
            return
        device = self.monitor_combo.currentData() if hasattr(self, "monitor_combo") else None
        stop = threading.Event()
        self._ambience_preview_stop = stop
        self._ambience_previewing = True
        self.ambience_preview_button.setText("■ 停止试听")
        self.ambience_preview_label.setText(
            f"正在试听（约 {len(pcm) / (SPEECH_SAMPLE_RATE * 2):.0f} 秒，只在试听设备里放）："
            + ("背景 + 你渲染过的一句话，听的是两者的真实比例。" if with_voice else
               "只有背景——本来就很轻（比说话低 20 dB 以上），戴耳机、把音量调大一点听；"
               "讲稿台或问答库渲染过一句之后，这里会叠上你的声音一起放。"))
        self.ambience_preview_label.setStyleSheet(f"color: {OK_GREEN};")
        relay = self.ambience_preview_done
        factory = getattr(self, "ambience_player_factory", None) or (
            lambda: MultiOutputPlayer([device], SPEECH_SAMPLE_RATE))

        def worker() -> None:
            player = None
            error = ""
            try:
                player = factory()
                if hasattr(player, "__enter__"):
                    player.__enter__()
                play_pcm(pcm, player, stop, threading.Event(), lambda _p: None)
            except Exception as exc:  # noqa: BLE001
                error = " ".join(str(exc).split())[:160] or type(exc).__name__
            finally:
                if player is not None:
                    try:
                        player.close()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                relay.emit(error)
            except RuntimeError:
                pass   # 设置窗口已经关了

        threading.Thread(target=worker, name="ambience-preview", daemon=True).start()

    def _on_ambience_preview_done(self, error: str) -> None:
        self._ambience_previewing = False
        self.ambience_preview_button.setText("▶ 试听 6 秒")
        if error:
            self.ambience_preview_label.setText(f"试听没放出来：{error}。检查「声卡」页的试听设备（选你的耳机或音箱）。")
            self.ambience_preview_label.setStyleSheet(f"color: {ERR_RED};")
        else:
            self.ambience_preview_label.setText("试听结束。正式答辩时这层背景会一直在，别人说话时也不会停。")
            self.ambience_preview_label.setStyleSheet(f"color: {TEXT_DIM};")

    def _test_monitor_device(self) -> None:
        if self._device_testing:
            self._device_test_stop.set()
            return
        from .device_test import play_device_test
        device = self.monitor_combo.currentData()
        name = self.monitor_combo.currentText()
        stop = threading.Event()
        self._device_test_stop = stop
        self._device_testing = True
        self.device_test_button.setText("■ 停止测试")
        self.device_test_label.setText(f"正在测试：{name}")
        relay = self.device_test_done
        play = getattr(self, "device_test_player", None) or play_device_test
        def worker():
            error, complete = "", False
            try:
                complete = play(device, stop)
            except Exception as exc:
                error = " ".join(str(exc).split())[:200] or type(exc).__name__
            try:
                relay.emit(error, complete)
            except RuntimeError:
                pass
        threading.Thread(target=worker, name="output-device-test", daemon=True).start()

    def _on_device_test_done(self, error: str, complete: bool) -> None:
        self._device_testing = False
        self.device_test_button.setText("▶ 播放测试")
        self.device_test_label.setText(
            f"测试失败：{error}。可在「显示全部接口」中尝试同一设备的其他接口。" if error else
            "测试音已发送，请确认耳机是否听到两声提示音；没有声音时请检查接线、音量或换设备。" if complete else
            "测试已停止。")

    def done(self, result) -> None:   # 保存或取消关窗口时，停掉还在放的背景试听
        device_stop = getattr(self, "_device_test_stop", None)
        if device_stop is not None:
            device_stop.set()
        stop = getattr(self, "_ambience_preview_stop", None)
        if stop is not None:
            stop.set()
        super().done(result)

    def _on_show_devices_toggled(self, checked: bool) -> None:
        self._show_all_devices = bool(checked)
        self.settings.set("show_all_audio_devices", bool(checked))
        self._refill_device_combos()

    def _device_list(self, kind: str, keep=None) -> list:
        """下拉里列哪些设备：默认筛掉重复和没用的；勾了「显示全部设备」就全列。"""
        items = self._all_inputs if kind == "mic" else self._all_outputs
        if self._show_all_devices:
            return list(items)
        try:
            return useful_devices(items, kind=kind, keep_index=keep)
        except Exception:  # noqa: BLE001 —— 筛选出错就全列，绝不让设置打不开
            return list(items)

    def _saved_cable_index(self, outputs):
        saved_name = str(self.settings.get("teams_output_device_name", "") or "").strip()
        if saved_name:
            found = next((d for d in outputs if d.name.strip() == saved_name), None)
            if found is not None:
                return found.index
        return self.settings.shared.get("teams_output_device")

    def _refill_device_combos(self) -> None:
        """切换「显示全部设备」后重填三个下拉，已选的保持不变。"""
        mic = self.mic_combo.currentData()
        monitor = self.monitor_combo.currentData()
        cable = self.cable_combo.currentData()
        _device_combo(self.mic_combo, self._device_list("mic", mic), mic,
                      allow_default="系统默认输入", direction="输入")
        _device_combo(self.monitor_combo, self._device_list("monitor", monitor), monitor,
                      allow_default="系统默认输出", direction="输出")
        self.cable_locked_index = self._apply_cable_lock(
            self.cable_combo, self._device_list("cable", cable), keep=cable)
        self._apply_mic_lock_state()
        self._apply_monitor_lock_state()

    # ------------------------------------------------------ connection test
    def _on_test_line(self, html: str) -> None:
        if self._test_lines and ("测试中…" in self._test_lines[-1] or self._test_lines[-1].startswith("正在检查")):
            self._test_lines.pop()
        self._test_lines.append(html)
        self.test_result_label.setText("<br>".join(self._test_lines))

    @Slot(int, str, bool)
    def _on_connection_event(self, serial: int, message: str, done: bool) -> None:
        if serial != self._connection_serial or not self._connection_running:
            return
        if message:
            self._on_test_line(message)
        if done:
            self._finish_connection_test()

    @Slot()
    def _finish_connection_test(self) -> None:
        self._connection_running = False
        self._connection_watchdog.stop()
        self.test_button.setEnabled(True)
        self._on_test_line("连接测试已结束，请查看上面的各项结果。")

    def _cancel_connection_test(self, *_args) -> None:
        self._connection_cancel.set()
        self._connection_running = False
        self._connection_watchdog.stop()
        self.test_button.setEnabled(True)

    def _connection_timeout(self) -> None:
        if self._connection_running:
            self._connection_cancel.set()
            self._on_test_line(f"<span style='color:{ERR_RED}'>检查超时，已停止后续检查；可点击测试连接重试。</span>")
            self._finish_connection_test()

    def _emit_test_line(self, html: str) -> None:
        try:
            self.test_line.emit(html)
        except RuntimeError:
            pass  # 对话框已关闭，丢弃结果即可

    def _run_connection_test(self) -> None:
        if self._connection_running:
            return
        workspace = self.workspace_edit.text().strip()
        api_key = self.api_key_edit.text().strip() or self.settings.get_api_key()
        if not workspace or not api_key:
            self._test_lines = [f"<span style='color:{ERR_RED}'>❌ 请先填写 Workspace ID 和 API Key</span>"]
            self.test_result_label.setText(self._test_lines[0])
            return
        self.test_button.setEnabled(False)
        self._connection_serial += 1
        self._connection_running = True
        self._connection_cancel = threading.Event()
        self._connection_watchdog.start(self.CONNECTION_TIMEOUT_MS)
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
        threading.Thread(target=self._connection_test_entry,
                         args=(args, self._connection_serial, self._connection_cancel), daemon=True).start()

    def _connection_test_entry(self, args, serial, cancel) -> None:
        try:
            self._test_worker(*args, request_id=serial, cancel_event=cancel)
        except Exception as exc:
            try:
                self.connection_event.emit(serial, f"检查异常：{html.escape(str(exc))}", False)
            except RuntimeError:
                pass
        finally:
            try:
                self.connection_event.emit(serial, "", True)
            except RuntimeError:
                pass

    def _test_worker(self, workspace: str, api_key: str, voice: str, llm_model: str, asr_model: str, tts_model: str,
                     *, request_id=None, cancel_event=None) -> None:
        proxy = self.settings.ws_proxy
        def emit(message):
            if cancel_event is not None and cancel_event.is_set():
                return
            try:
                if request_id is None:
                    self._emit_test_line(message)
                else:
                    self.connection_event.emit(request_id, message, False)
            except RuntimeError:
                pass

        def brief(message: str) -> str:
            message = " ".join(str(message).split())
            return html.escape(message[:2048])

        if cancel_event is not None and cancel_event.is_set():
            return
        emit("正在检查翻译模型…")
        try:
            from .memory import MeetingMemory

            translator = ContextTranslator(
                api_key=api_key,
                workspace_id=workspace,
                model=llm_model,
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=15,
            )
            # 测首字延迟：实时翻译能不能用，看的就是这个数，而不是「能不能返回」。
            started = time.monotonic()
            first_piece = [0.0]

            def on_delta(_piece: str, _so_far: str) -> None:
                if not first_piece[0]:
                    first_piece[0] = time.monotonic() - started

            result = translator.translate(
                MeetingMemory(), "zh2en", "你好，我来介绍一下我的研究。", on_delta=on_delta, cancel_event=cancel_event
            )
            total = time.monotonic() - started
            latency = first_piece[0] or total
            problem = translation_problem(llm_model)
            if problem:
                color, verdict = WARN_ORANGE, f"⚠️ {problem}"
            elif latency > 3.0:
                color, verdict = WARN_ORANGE, "⚠️ 首字偏慢，实时翻译会有明显停顿，建议换 flash 系列"
            else:
                color, verdict = OK_GREEN, "✅ 适合实时翻译"
            emit(
                f"<span style='color:{color}'>{verdict} · 翻译大模型 {llm_model} "
                f"首字 {latency:.1f}s / 全句 {total:.1f}s（返回：{result[:30]}）</span>"
            )
        except Exception as exc:
            emit(f"<span style='color:{ERR_RED}'>❌ 翻译大模型 {llm_model} 失败：{brief(exc)}</span>")
        if cancel_event is not None and cancel_event.is_set():
            return
        emit("正在检查识别连接…")
        asr = None
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
            emit(f"<span style='color:{OK_GREEN}'>✅ 实时识别 {asr_model} 连接正常</span>")
        except Exception as exc:
            emit(f"<span style='color:{ERR_RED}'>❌ 实时识别 {asr_model} 失败：{brief(exc)}</span>")
        finally:
            if asr is not None:
                try:
                    asr.close()
                except Exception:
                    pass
        if cancel_event is not None and cancel_event.is_set():
            return
        if voice or (tts_model == "qwen-audio-3.1-tts-next" and _has_tts_source(self.settings)):
            emit("正在检查音色合成与本机试听…")
            try:
                voice_probe(self.settings, workspace, api_key, voice, tts_model, cancel_event=cancel_event)
                emit(
                    f"<span style='color:{OK_GREEN}'>✅ 语音模型可用（{tts_model}），已在本机播放试听，请确认声音效果</span>"
                )
            except Exception as exc:
                emit(f"<span style='color:{ERR_RED}'>❌ 音色合成失败：{brief(exc)}</span>")
        else:
            emit(
                f"<span style='color:{WARN_ORANGE}'>⚠️ 未配置克隆音色或 Next 参考录音，已跳过试听</span>"
            )
        try:
            if request_id is None:
                self.test_done.emit()
        except RuntimeError:
            pass

    def _refresh_voice_combo(self, keep: str | None = None) -> None:
        """音色下拉：和当前 TTS 模型匹配的排上面可选；绑定别的模型的放在分隔线下面、灰色点不动。

        音色和合成模型是绑定的，选了不匹配的一合成就报错。当前已选的那个即使不匹配也保持可选，
        下面的红字会提示换哪个。
        """
        current = keep if keep is not None else str(self.settings.get("tts_voice_id", "") or "")
        model_combo = getattr(self, "tts_model_combo", None)
        model = str(model_combo.currentData() if model_combo is not None else self.settings.get("tts_model", "")) or ""
        combo = self.voice_combo
        combo.blockSignals(True)
        combo.clear()
        matched, others = [], []
        entries = self.settings.get("voice_library", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("voice_id"):
                    continue
                bound = str(entry.get("target_model", "") or "")
                (matched if not bound or bound == model else others).append(entry)
        for entry in matched:
            combo.addItem(f"{entry.get('name', '未命名')} · {entry.get('target_model', '')}", entry.get("voice_id"))
        if current and all(entry.get("voice_id") != current for entry in matched + others):
            combo.addItem(f"{current} · 未入库", current)
        if not matched:
            combo.addItem(f"（音色库里没有绑定 {model or '这个模型'} 的音色：点右边「一键克隆音色」）", "")
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        if others:
            combo.insertSeparator(combo.count())
            combo.addItem("—— 下面这些绑定的是别的模型，当前模型用不了 ——", "")
            header = combo.model().item(combo.count() - 1)
            if header is not None:
                header.setEnabled(False)
            for entry in others:
                voice_id = entry.get("voice_id")
                combo.addItem(f"{entry.get('name', '未命名')} · {entry.get('target_model', '')}（换到这个模型才能用）",
                              voice_id)
                item = combo.model().item(combo.count() - 1)
                if item is not None and voice_id != current:
                    item.setEnabled(False)
                    item.setForeground(QColor("#6b7280"))
        index = combo.findData(current) if current else -1
        if index < 0:
            index = next((i for i in range(combo.count()) if combo.itemData(i)
                          and combo.model().item(i) is not None and combo.model().item(i).isEnabled()), 0)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

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

    def _apply_cable_lock(self, combo: QComboBox, outputs: list, keep=None) -> int | None:
        """程序输出防呆：锁定你选的那一个（黄色高亮）。

        第一次用、或保存的设备不在了：自动选 CABLE Input。以前每次都重新去找第一个
        CABLE Input，结果你选好的设备一打开设置又「自动跑到 11」。
        """
        self._outputs_cache = outputs
        cables = [d for d in outputs if "cable input" in d.name.lower()]
        saved_name = str(self.settings.get("teams_output_device_name", "") or "").strip()
        saved_index = self.settings.shared.get("teams_output_device")
        chosen = None
        if keep is not None:
            chosen = next((d for d in outputs if d.index == keep), None)
        if chosen is None and saved_name:
            chosen = next((d for d in outputs if d.name.strip() == saved_name), None)
        if chosen is None and saved_index is not None and not saved_name:
            chosen = next((d for d in outputs if d.index == saved_index and "cable" in d.name.lower()), None)
        if chosen is None and cables:
            chosen = cables[0]
        cable = chosen
        self.cable_locked_index = cable.index if cable is not None else None
        _device_combo(combo, outputs, None, direction="输出")
        for device in cables:
            position = combo.findData(device.index)
            if position >= 0:
                combo.setItemData(position, QColor("#ffd400"), Qt.ForegroundRole)
        if cable is None:
            combo.setEnabled(True)
            self.cable_locked = False
            self.cable_lock_button.setText("🔒 无虚拟声卡")
            self.cable_lock_button.setEnabled(False)
            self.cable_note.setText("❌ 未检测到 CABLE Input：请安装 VB-CABLE 驱动并重启 Windows")
            self.cable_note.setStyleSheet(f"color: {ERR_RED};")
            return None
        if not self.cable_lock_button.isEnabled():
            # 之前没检测到虚拟声卡、现在有了：恢复锁定
            self.cable_lock_button.setEnabled(True)
            self.cable_locked = True
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
            name = self.cable_combo.currentText()
            self.cable_note.setText(f"✅ 已锁定：{name}（所有合成的英文都从这里送进会议）"
                                    if "cable" in name.lower() else
                                    f"✅ 已锁定：{name}（注意：不是 CABLE 虚拟声卡，会议里可能听不到）")
            self.cable_note.setStyleSheet(f"color: {OK_GREEN};")
        else:
            combo.setEnabled(True)
            combo.setStyleSheet("")
            self.cable_lock_button.setText("🔓 已解锁")
            self.cable_lock_button.setToolTip("当前已解锁，点击重新锁定虚拟声卡")
            self.cable_note.setText("⚠️ 已解锁手动选择（高级用途；正常情况请重新锁定）")
            self.cable_note.setStyleSheet(f"color: {WARN_ORANGE};")

    def _toggle_cable_lock(self) -> None:
        """解锁 → 随便选；再点锁定 → 锁住你刚选的那个（不再跳回第一个 CABLE Input）。"""
        if self.cable_locked:
            self.cable_locked = False
            self._apply_cable_lock_state(self.cable_combo)
            return
        self.cable_locked = True
        keep = self.cable_combo.currentData()
        self._apply_cable_lock(self.cable_combo, self._outputs_cache, keep=keep)

    # --------------------------------------------------------- 语音缓存用量
    @staticmethod
    def _format_mb(value: int) -> str:
        megabytes = value / (1024 * 1024)
        return f"{megabytes:.0f} MB" if megabytes >= 10 else f"{megabytes:.1f} MB"

    def _cache_view(self):
        """按设置目录现开一个只读视图，避免设置页依赖 engine 实例。"""
        from .tts_cache import TtsDiskCache

        return TtsDiskCache(self.settings.base_dir, limit_mb=self.disk_cache_spin.value())

    def _refresh_cache_usage(self) -> None:
        try:
            stats = self._cache_view().stats()
            spool = Path(self.settings.base_dir) / "meeting_audio.pcm"
            meeting = spool.stat().st_size if spool.is_file() else 0
        except OSError as exc:
            self.cache_usage_label.setText(f"读取缓存用量失败：{exc}")
            self.cache_usage_label.setStyleSheet(f"color: {WARN_ORANGE};")
            return
        if not stats["enabled"]:
            self.cache_usage_label.setText("缓存目录不可写，本机已停用语音缓存（不影响答辩，只是重播会重新合成）")
            self.cache_usage_label.setStyleSheet(f"color: {WARN_ORANGE};")
            return
        parts = [
            f"实时翻译 {self._format_mb(stats['auto_bytes'])}（{stats['auto_count']} 条）",
            f"讲稿预渲染 {self._format_mb(stats['pinned_bytes'])}（{stats['pinned_count']} 条，受保护不淘汰）",
            f"本场录音 {self._format_mb(meeting)}",
            f"磁盘剩余 {stats['free_bytes'] / (1024 ** 3):.1f} GB",
        ]
        text = " · ".join(parts)
        color = OK_GREEN
        # 快满了要提前说，别等到现场写不进去
        if stats["bytes"] > stats["limit_bytes"] * 0.9:
            text += "\n⚠️ 接近上限，超出后会淘汰最久未用的实时翻译缓存（讲稿不受影响）"
            color = WARN_ORANGE
        if stats["last_skip"]:
            text += f"\n⚠️ {stats['last_skip']}"
            color = ERR_RED
        self.cache_usage_label.setText(text)
        self.cache_usage_label.setStyleSheet(f"color: {color};")

    def _clear_cache_clicked(self) -> None:
        view = self._cache_view()
        stats = view.stats()
        if not stats["count"]:
            self._refresh_cache_usage()
            QMessageBox.information(self, "语音缓存", "缓存已经是空的")
            return
        box = QMessageBox(self)
        box.setWindowTitle("清理语音缓存")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"实时翻译缓存 {self._format_mb(stats['auto_bytes'])}（{stats['auto_count']} 条）\n"
            f"讲稿预渲染 {self._format_mb(stats['pinned_bytes'])}（{stats['pinned_count']} 条）"
        )
        box.setInformativeText(
            "讲稿预渲染是「断网也能讲」的依据，删了就要重新渲染。默认只清实时翻译那部分。"
        )
        keep = box.addButton("只清实时翻译", QMessageBox.AcceptRole)
        everything = box.addButton("连讲稿一起清", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            removed = view.clear()
        elif clicked is everything:
            removed = view.clear(include_pinned=True)
        else:
            return
        self._refresh_cache_usage()
        QMessageBox.information(self, "语音缓存", f"已删除 {removed} 个缓存文件")

    # ------------------------------------------------------ 提前渲染的模型 + 音色
    # ------------------------------------------------------ 双通道
    def _build_channel_box(self) -> QWidget:
        box = QFrame()
        box.setObjectName("channelBox")
        box.setStyleSheet("QFrame#channelBox { border: 1px solid #3d4a63; border-radius: 8px; }"
                          "QFrame#channelBox QLabel, QFrame#channelBox QCheckBox { border: none; }")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 8, 12, 10)
        self.channel_summary_label = QLabel("")
        self.channel_summary_label.setWordWrap(True)
        self.channel_summary_label.setStyleSheet(f"color: {QA_ACCENT}; font-weight: 600;")
        layout.addWidget(self.channel_summary_label)
        self.backup_check = QCheckBox("开启第二通道：实时翻译、问答库匹配、生成回答这三类请求同时（或稍后）发给第二个入口，谁先出字用谁")
        self.backup_check.setChecked(bool(self.settings.get("backup_channel_enabled", False)))
        layout.addWidget(self.backup_check)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.backup_mode_combo = NoWheelComboBox()
        for value, label in CHANNEL_MODES:
            self.backup_mode_combo.addItem(label, value)
        self.backup_mode_combo.setCurrentIndex(max(0, self.backup_mode_combo.findData(
            str(self.settings.get("backup_channel_mode", "race") or "race"))))
        form.addRow("怎么用", self.backup_mode_combo)
        self.backup_route_combo = NoWheelComboBox()
        for value, label in BACKUP_ROUTES:
            self.backup_route_combo.addItem(label, value)
        self.backup_route_combo.setCurrentIndex(max(0, self.backup_route_combo.findData(
            str(self.settings.get("backup_route", "beijing-public") or "beijing-public"))))
        form.addRow("第二通道走哪里", self.backup_route_combo)
        self.backup_key_edit = QLineEdit()
        self.backup_key_edit.setEchoMode(QLineEdit.Password)
        self.backup_key_edit.setPlaceholderText(
            "该地域的 API Key（已保存时留空即可；存进 Windows 凭据管理器）" if get_backup_api_key()
            else "该地域的 API Key（各地域的 Key 互相独立，北京的 Key 在别的地域用不了）")
        form.addRow("第二通道 API Key", self.backup_key_edit)
        self.backup_workspace_edit = QLineEdit(str(self.settings.get("backup_workspace_id", "") or ""))
        self.backup_workspace_edit.setPlaceholderText("该地域的业务空间 ID（可空；填了走专属入口）")
        self.backup_workspace_edit.setToolTip("日本东京、德国法兰克福必须填写该地域的业务空间 ID")
        form.addRow("第二通道业务空间 ID", self.backup_workspace_edit)
        self.backup_model_edit = ModelPickCombo(TRANSLATE_MODELS, str(self.settings.get("backup_model", "") or ""),
                                                same_label="和主通道相同（同一个模型）")
        self.backup_model_edit.setToolTip("下拉选一个（比如主通道 plus、这里选 flash），也可以直接输入模型名；"
                                          "别的地域没有这个模型时填一个那边有的")
        form.addRow("第二通道模型（可以只换模型：plus / flash…）", self.backup_model_edit)
        self.backup_asr_check = QCheckBox("识别备用通道：主模型故障时切换（独立开关，仅北京；未完成的句子需重说）")
        self.backup_asr_check.setChecked(bool(self.settings.get("backup_asr_enabled", False)))
        form.addRow(self.backup_asr_check)
        self.backup_asr_combo = NoWheelComboBox()
        fill_model_combo(self.backup_asr_combo, ASR_MODELS,
                         str(self.settings.get("backup_asr_model", "qwen3-asr-flash-realtime")), "qwen3-asr-flash-realtime")
        form.addRow("备用识别模型", self.backup_asr_combo)
        self.backup_tts_check = QCheckBox("合成第二通道：先返回完整有效音频的一路胜出（独立开关，仅北京；可能增加费用）")
        self.backup_tts_check.setChecked(bool(self.settings.get("backup_tts_enabled", False)))
        form.addRow(self.backup_tts_check)
        self.backup_tts_voice_combo = NoWheelComboBox()
        self.backup_tts_voice_combo.addItem("沿用当前模型和音色（同模型重试）", ("", ""))
        selected_pair = (str(self.settings.get("backup_tts_model", "") or ""),
                         str(self.settings.get("backup_tts_voice_id", "") or ""))
        for entry in self.settings.get("voice_library", []) or []:
            if isinstance(entry, dict) and entry.get("voice_id") and entry.get("target_model"):
                pair = (str(entry["target_model"]), str(entry["voice_id"]))
                self.backup_tts_voice_combo.addItem(
                    f"{entry.get('name') or entry['voice_id']} · {entry['target_model']}", pair)
        idx = self.backup_tts_voice_combo.findData(selected_pair)
        if idx < 0 and all(selected_pair):
            self.backup_tts_voice_combo.addItem(f"已保存音色 · {selected_pair[0]}", selected_pair)
            idx = self.backup_tts_voice_combo.count() - 1
        self.backup_tts_voice_combo.setCurrentIndex(max(0, idx))
        form.addRow("备用合成模型 / 绑定音色", self.backup_tts_voice_combo)
        self.backup_tts_mode_combo = NoWheelComboBox()
        for value, label in CHANNEL_TTS_MODES:
            self.backup_tts_mode_combo.addItem(label, value)
        self.backup_tts_mode_combo.setCurrentIndex(max(0, self.backup_tts_mode_combo.findData(
            str(self.settings.get("backup_tts_mode", "delayed") or "delayed"))))
        form.addRow("合成第二通道怎么用", self.backup_tts_mode_combo)
        layout.addLayout(form)
        test_row = QHBoxLayout()
        self.channel_test_button = QPushButton("测试两个通道（各发一句，看谁快）")
        self.channel_test_button.clicked.connect(self._test_channels)
        test_row.addWidget(self.channel_test_button)
        test_row.addStretch(1)
        layout.addLayout(test_row)
        self.channel_test_label = QLabel("")
        self.channel_test_label.setWordWrap(True)
        layout.addWidget(self.channel_test_label)
        note = QLabel(
            "<b>只说一次</b>：以先吐出正文的通道为准，另一个立刻取消、内容全部丢弃；定下来以后就算中途断了也不会换通道接着说。"
            "识别和合成由各自的开关控制，使用北京账号；无需更换地区，也可换模型。合成的不同模型必须有对应的克隆音色。<br>"
            "合成第二通道<b>只管现场发声</b>：讲稿台 / 问答库 / 快捷用语的提前渲染不走它，已经渲染好的音频不受这个开关影响；"
            "默认主通道 2 秒没出声或失败才启用，主通道正常时不多花钱、声音也不会换。"
            "识别是故障切换：主模型报错、断线，或者一直有人声却 6 秒没有任何识别结果时切到备用模型，"
            "切换那几秒说的话需要重说，不会把同一句识别两次。测试按钮仅测试大模型通道，不会测试音频。<br>"
            "阿里云百炼的地域只有：华北2（北京）、中国香港、新加坡、日本（东京）、美国（弗吉尼亚）、德国（法兰克福）——"
            "没有杭州 / 上海。第一通道固定北京（识别、合成、复刻都在北京）。第二通道推荐「北京 · 公共入口」：同一个 Key，换一条线路；"
            "选别的地域要在那边单独开通并建 Key，模型也可能不一样。")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(note)
        for widget in (self.backup_check, self.backup_asr_check, self.backup_tts_check):
            widget.toggled.connect(lambda _c: self._update_channel_ui())
        self.backup_route_combo.currentIndexChanged.connect(lambda _i: self._update_channel_ui())
        self.backup_model_edit.editTextChanged.connect(lambda _t: self._update_channel_ui())
        self.backup_mode_combo.currentIndexChanged.connect(lambda _i: self._update_channel_ui())
        self.channel_test_done.connect(self._on_channel_test_done)
        self._update_channel_ui()
        return box

    def _update_channel_ui(self) -> None:
        enabled = self.backup_check.isChecked()
        route = str(self.backup_route_combo.currentData() or "beijing-public")
        other_region = not uses_main_key(route)
        self.backup_mode_combo.setEnabled(enabled)
        self.backup_asr_combo.setEnabled(self.backup_asr_check.isChecked())
        self.backup_tts_voice_combo.setEnabled(self.backup_tts_check.isChecked())
        self.backup_tts_mode_combo.setEnabled(self.backup_tts_check.isChecked())
        for widget in (self.backup_route_combo, self.channel_test_button):
            widget.setEnabled(enabled)
        for widget in (self.backup_key_edit, self.backup_workspace_edit):
            widget.setEnabled(enabled and other_region)
        self.backup_model_edit.setEnabled(enabled)
        primary = f"第一通道：{REGION_LABELS[PRIMARY_REGION]} · 业务空间专属入口"
        if not enabled:
            second = "第二通道：未开启"
        else:
            label = route_label(route)
            missing = other_region and not (self.backup_key_edit.text().strip() or get_backup_api_key())
            second_model = self.backup_model_edit.text()
            second = f"第二通道：{label}" + (f" · {second_model}" if second_model else "") + (
                "（还没填这个地域的 API Key，不会生效）" if missing else "")
            if route in ("ap-northeast-1", "eu-central-1") and not self.backup_workspace_edit.text().strip():
                second += "（还需填写该地域业务空间 ID）"
        self.channel_summary_label.setText(f"{primary}　｜　{second}")

    def _test_channels(self) -> None:
        """两个通道各发一句最短的请求，报告各自多久出第一个字（费用可以忽略）。"""
        route = str(self.backup_route_combo.currentData() or "beijing-public")
        main_key = self.api_key_edit.text().strip() or self.settings.get_api_key()
        workspace = self.workspace_edit.text().strip() or self.settings.workspace_id
        second_key = main_key if uses_main_key(route) else (self.backup_key_edit.text().strip() or get_backup_api_key())
        model = str(self.llm_combo.currentData() or "qwen3.8-flash")
        second_model = self.backup_model_edit.text().strip() or model
        if not main_key or not workspace:
            self.channel_test_label.setText("先填好百炼 Workspace ID 和 API Key")
            return
        if not second_key:
            self.channel_test_label.setText("第二通道还没有 API Key")
            return
        try:
            second_endpoint = backup_endpoint(route, self.backup_workspace_edit.text().strip(), workspace)
        except ValueError as exc:
            self.channel_test_label.setText(str(exc))
            return
        jobs = [
            ("第一通道", primary_endpoint(workspace), main_key, model),
            ("第二通道", second_endpoint, second_key, second_model),
        ]
        self.channel_test_button.setEnabled(False)
        self.channel_test_label.setText("正在测试…")
        self.channel_test_label.setStyleSheet(f"color: {TEXT_DIM};")
        relay = self.channel_test_done
        proxy, proxy_mode = self.settings.ws_proxy, self.settings.proxy_mode
        runner = getattr(self, "channel_test_runner", None)

        def one(name, endpoint, key, model_id) -> str:
            began = time.monotonic()
            first = []
            try:
                if runner is not None:
                    runner(endpoint, key, model_id)
                else:
                    translator = ContextTranslator(api_key=key, workspace_id=workspace, model=model_id,
                                                   proxy=proxy, proxy_mode=proxy_mode, timeout=20)
                    payload = {"model": model_id, "messages": [{"role": "user", "content": "Reply with the word OK."}],
                               "temperature": 0, "stream": True}
                    payload.update(thinking_fields(model_id, want_thinking=False))
                    translator._stream_once(endpoint, key, payload,
                                            lambda _p, _s: first.append(time.monotonic()) if not first else None,
                                            None, deadline_seconds=20)
            except Exception as exc:  # noqa: BLE001
                return f"{name} ❌ {' '.join(str(exc).split())[:120]}"
            seconds = (first[0] if first else time.monotonic()) - began
            return f"{name} ✅ {seconds:.2f} 秒出字（{model_id}）"

        def worker() -> None:
            results: list[str] = [""] * len(jobs)
            threads = []
            for index, job in enumerate(jobs):
                def task(i=index, j=job):
                    results[i] = one(*j)
                thread = threading.Thread(target=task, daemon=True)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join(40)
            try:
                relay.emit("\n".join(r or "超时" for r in results))
            except RuntimeError:
                pass

        threading.Thread(target=worker, name="channel-test", daemon=True).start()

    def _on_channel_test_done(self, text: str) -> None:
        self.channel_test_button.setEnabled(self.backup_check.isChecked())
        self.channel_test_label.setText(text)
        self.channel_test_label.setStyleSheet(f"color: {ERR_RED if '❌' in text else OK_GREEN};")

    def _open_voice_library(self) -> None:
        dialog = VoiceLibraryDialog(self, self.settings)
        dialog.exec()
        if dialog.changed:
            # 音色库变了：两套下拉都按最新的音色库和设置重建
            model_index = self.tts_model_combo.findData(str(self.settings.get("tts_model", "") or ""))
            if model_index >= 0:
                self.tts_model_combo.blockSignals(True)
                self.tts_model_combo.setCurrentIndex(model_index)
                self.tts_model_combo.blockSignals(False)
            self._refresh_voice_combo(keep=str(self.settings.get("tts_voice_id", "") or ""))
            self._update_voice_hint()
            pre_model = str(self.settings.get("prerender_tts_model", "") or "")
            self.prerender_model_combo.blockSignals(True)
            self.prerender_model_combo.setCurrentIndex(max(0, self.prerender_model_combo.findData(pre_model)))
            self.prerender_model_combo.blockSignals(False)
            self._refresh_prerender_voice_combo(str(self.settings.get("prerender_voice_id", "") or ""))
            self._update_prerender_hint()

    def _prerender_pair(self) -> tuple[str, str]:
        """要保存的 (提前渲染模型, 音色)。模型选了但没有能用的音色时当作「和现场相同」，免得渲染失败。"""
        model = str(self.prerender_model_combo.currentData() or "")
        data = self.prerender_voice_combo.currentData() or ("", "")
        if not model or not data or not data[1] or data[0] != model:
            return "", ""
        return model, data[1]

    def _refresh_prerender_voice_combo(self, keep_voice: str = "") -> None:
        """提前渲染音色：和上面选的模型匹配的排上面可选，其余放分隔线下面、灰色点不动。
        模型选「和现场发声相同」时音色也跟现场一致，这里不用选。"""
        combo = self.prerender_voice_combo
        model = str(self.prerender_model_combo.currentData() or "")
        combo.blockSignals(True)
        combo.clear()
        if not model:
            combo.addItem("和现场发声相同", ("", ""))
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        combo.setEnabled(True)
        options = scene_modes.voice_options(self.settings)
        matched = [o for o in options if o["model"] == model]
        others = [o for o in options if o["model"] != model]
        for option in matched:
            combo.addItem(option["label"], (option["model"], option["voice_id"]))
        if not matched:
            combo.addItem(f"（音色库里没有绑定 {model} 的音色：先用这个模型复刻一个）", ("", ""))
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        if others:
            combo.insertSeparator(combo.count())
            combo.addItem("—— 下面这些绑定的是别的模型，这个模型用不了 ——", ("", ""))
            header = combo.model().item(combo.count() - 1)
            if header is not None:
                header.setEnabled(False)
            for option in others:
                combo.addItem(option["label"] + "（换到这个模型才能用）", (option["model"], option["voice_id"]))
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
                    item.setForeground(QColor("#6b7280"))
        index = next((i for i in range(combo.count())
                      if combo.itemData(i) and tuple(combo.itemData(i)) == (model, keep_voice)), -1)
        if index < 0:
            index = next((i for i in range(combo.count()) if combo.itemData(i) and combo.itemData(i)[1]
                          and combo.model().item(i) is not None and combo.model().item(i).isEnabled()), 0)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _on_prerender_model_changed(self) -> None:
        current = self.prerender_voice_combo.currentData() or ("", "")
        self._refresh_prerender_voice_combo(current[1] if current else "")
        self._update_prerender_hint()

    def _on_prerender_voice_changed(self) -> None:
        self._update_prerender_hint()

    def _update_prerender_hint(self) -> None:
        model = str(self.prerender_model_combo.currentData() or "")
        data = self.prerender_voice_combo.currentData() or ("", "")
        if not model:
            self.prerender_hint.setText("讲稿台 / 问答库 / 快捷用语会用上面「现场发声」的模型和音色。")
            self.prerender_hint.setStyleSheet(f"color: {TEXT_DIM};")
        elif not data[1]:
            self.prerender_hint.setText(f"❌ 没有绑定 {model} 的音色：提前渲染会失败。换一个模型，或用这个模型复刻一个音色。")
            self.prerender_hint.setStyleSheet(f"color: {ERR_RED};")
        else:
            self.prerender_hint.setText(f"✅ 提前渲染用 {model} + 这个音色（和现场发声分开，讲稿台里也能快速换）。")
            self.prerender_hint.setStyleSheet(f"color: {OK_GREEN};")

    def _update_voice_hint(self) -> None:
        voice = self._current_voice_id()
        model = str(self.tts_model_combo.currentData())
        if model == "qwen-audio-3.1-tts-next":
            self.voice_hint.setText("Next 每次读取下方参考录音；现有 voice_id 不能用于这个模型。音频生成是非流式，建议先预渲染讲稿。")
            self.voice_hint.setStyleSheet(f"color: {WARN_ORANGE};")
            return
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
            suffix = (
                "请用该模型重新复刻一个。"
                if not self._voices_bound_to(model)
                else "音色库里有绑定该模型的音色，可在上方下拉框选择。"
            )
            self.voice_hint.setText(
                f"⚠️ 音色绑定的是 {bound_model}，与当前 TTS 模型 {model} 不是同一绑定模型。{suffix}"
            )
            self.voice_hint.setStyleSheet(f"color: {ERR_RED};")
        else:
            self.voice_hint.setText(f"✅ 音色与模型匹配（{bound_model}）。")
            self.voice_hint.setStyleSheet(f"color: {OK_GREEN};")

    def _on_tts_model_changed(self) -> None:
        """换模型时自动跟上对应音色。

        只挂在模型下拉框上，不能放进 _update_voice_hint —— 那个函数在**音色**
        下拉框变化时也会跑，放进去就会在用户手动选了某个音色的瞬间又把它改掉。
        用户的显式选择永远优先于自动行为。
        """
        model = str(self.tts_model_combo.currentData() or "")
        self._update_reference_audio_ui()
        switched = self._auto_pick_voice_for_model(model)
        if not switched:
            # 按新模型重新分组：匹配的放上面，不匹配的灰掉
            self._refresh_voice_combo(keep=self._current_voice_id())
        self._update_voice_hint()
        if switched:
            self.voice_hint.setText(f"已自动切换到绑定 {model} 的音色：{switched}")
            self.voice_hint.setStyleSheet(f"color: {OK_GREEN};")

    def _update_reference_audio_ui(self) -> None:
        visible = str(self.tts_model_combo.currentData() or "") == "qwen-audio-3.1-tts-next"
        self.reference_audio_box.setVisible(visible)
        self.reference_audio_label.setVisible(visible)
        self.voice_combo.setEnabled(not visible)
        self.clone_voice_button.setEnabled(not visible)

    def _browse_reference_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Next 参考录音", "", "音频 (*.wav *.mp3 *.ogg *.opus)")
        if path:
            self.reference_audio_edit.setText(path)

    def _voices_bound_to(self, model: str) -> list[dict]:
        """音色库里绑定到该模型的条目，最近创建的排前面。"""
        entries = self.settings.get("voice_library", [])
        if not isinstance(entries, list):
            return []
        matched = [
            entry for entry in entries
            if isinstance(entry, dict)
            and str(entry.get("target_model", "")) == model
            and str(entry.get("voice_id", "")).strip()
        ]
        matched.sort(key=lambda entry: str(entry.get("created_at", "")), reverse=True)
        return matched

    def _auto_pick_voice_for_model(self, model: str) -> str:
        """换模型时自动选上绑定该模型的音色，省得每次手动换。

        选最近创建的那个：批量复刻之后音色库里同一模型往往有好几个，
        最新的通常就是用户刚建、正准备试听的那个。
        """
        candidates = self._voices_bound_to(model)
        if not candidates:
            return ""
        voice_id = str(candidates[0]["voice_id"])
        if voice_id == self._current_voice_id():
            return ""
        # 下拉框不可编辑，必须走 _refresh_voice_combo 才能正确带上 userData。
        # 屏蔽信号：这个方法是从 _update_voice_hint 里调的，不挡会递归。
        self.voice_combo.blockSignals(True)
        try:
            self._refresh_voice_combo(keep=voice_id)
        finally:
            self.voice_combo.blockSignals(False)
        return voice_id if self._current_voice_id() == voice_id else ""

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

    def _update_ssml_support(self, _index: int = -1) -> None:
        model = str(self.tts_model_combo.currentData() or "")
        supported = tts_supports(model, "ssml")
        self.sentence_break_spin.setEnabled(supported)
        if supported:
            self.sentence_break_note.setText(
                "该模型支持 SSML，可以显式控制句间换气。0 ms 保持原有行为。"
            )
            self.sentence_break_note.setStyleSheet(f"color: {OK_GREEN};")
        else:
            # 不清零已保存的值：换回支持的模型时设置还在。
            self.sentence_break_note.setText(
                "当前模型不支持 SSML，句间换气不会生效（设置值会保留）。句子韵律仍由模型按标点决定。"
            )
            self.sentence_break_note.setStyleSheet(f"color: {TEXT_DIM};")

    def _update_capability_label(self, _index: int = -1) -> None:
        """一行显示当前模型支持什么。只讲能力，不讲开关状态。"""
        model = str(self.tts_model_combo.currentData() or "")
        if model == "qwen-audio-3.1-tts-next":
            self.capability_label.setText(
                f"<span style='color:{OK_GREEN}'>✓ 本机参考录音 · ✓ 并行预生成</span>"
                f"<br><span style='color:{WARN_ORANGE}'>完整音频返回，不能用已有 voice_id；现场实时答辩可能比常规 TTS 慢，建议先渲染讲稿。</span>"
            )
            return
        # 并行预生成看会话类型：实时 WebSocket 是单连接串行，不能并发写同一会话。
        prefetch = not model.startswith("qwen3-tts-vc-realtime")
        items = (
            ("语气指令", tts_supports(model, "instruction")),
            ("SSML 停顿", tts_supports(model, "ssml")),
            ("并行预生成", prefetch),
            ("声音复刻", tts_supports(model, "clone")),
        )
        parts = []
        for name, ok in items:
            color = OK_GREEN if ok else ERR_RED
            parts.append(f"<span style='color:{color}'>{'✓' if ok else '✗'} {name}</span>")
        missing = [name for name, ok in items if not ok]
        text = " · ".join(parts)
        if missing:
            text += (
                f"<br><span style='color:{WARN_ORANGE}'>缺少：{'、'.join(missing)}。"
                f"换成 qwen-audio-3.0-tts-plus 四项齐全（需用该模型重新复刻音色）。</span>"
            )
        if not model:
            text = f"<span style='color:{TEXT_DIM}'>未选择模型</span>"
        elif model not in TTS_CAPABILITIES:
            text = (
                f"<span style='color:{WARN_ORANGE}'>未知模型 {model}：按「全部不支持」处理，"
                "只会发送最基本的合成参数。</span>"
            )
        self.capability_label.setText(text)

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

    def _open_glossary_from_settings(self) -> None:
        parent = self.parent()
        opener = getattr(parent, "_open_glossary", None)
        if callable(opener):
            opener()
        else:
            GlossaryDialog(self, self.settings).exec()

    def _save(self) -> None:
        if self.backup_check.isChecked():
            try:
                backup_endpoint(str(self.backup_route_combo.currentData()), self.backup_workspace_edit.text().strip(),
                                self.workspace_edit.text().strip() or self.settings.workspace_id)
            except ValueError as exc:
                QMessageBox.warning(self, "第二通道配置不完整", str(exc))
                return
        # 备用合成模型不支持 SSML 换气也不用拦：第二通道会自动去掉换气标记只读文字。
        from ..api_payloads import cosyvoice_instruction_units
        model = str(self.tts_model_combo.currentData())
        if model == "qwen-audio-3.1-tts-next":
            source = Path(self.reference_audio_edit.text().strip()).expanduser()
            if source.suffix.lower() not in {".wav", ".mp3", ".ogg", ".opus"} or not source.is_file():
                QMessageBox.warning(self, "缺少参考录音", "Next 模型需要选择本机 WAV、MP3 或 OGG/Opus 录音。")
                return
            try:
                duration = voice_sample_duration(source)
                if not 0 < duration <= 30 or source.stat().st_size > 10 * 1024 * 1024:
                    raise VoiceSampleError("参考录音必须在 30 秒以内、10 MB 以内")
            except (VoiceSampleError, OSError) as exc:
                QMessageBox.warning(self, "参考录音无效", str(exc))
                return
        if model.startswith("cosyvoice") and cosyvoice_instruction_units(self.instruction_edit.text().strip()) > 100:
            QMessageBox.warning(self, "语气指令过长", "CosyVoice 限 100 字符单位，汉字计 2。请缩短后保存。")
            return
        keys = [self.direct_hotkey_combo.currentData(), self.translate_hotkey_combo.currentData(), self.cancel_hotkey_combo.currentData()]
        if len(set(keys)) != len(keys):
            QMessageBox.warning(self, "快捷键冲突", "原声、翻译、停止需要三个不同的快捷键。")
            return
        for entry in self.settings.get("voice_library", []):
            if model != "qwen-audio-3.1-tts-next" and isinstance(entry, dict) and entry.get("voice_id") == self._current_voice_id() and entry.get("target_model") and entry["target_model"] != model:
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
                "teams_output_device": self.cable_combo.currentData(),
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
                "teams_output_device_name": next(
                    (device.name.strip() for device in self._outputs_cache
                     if device.index == self.cable_combo.currentData()), ""),
                "tts_model": str(self.tts_model_combo.currentData()),
                "tts_reference_audio": self.reference_audio_edit.text().strip(),
                "backup_channel_enabled": self.backup_check.isChecked(),
                "backup_channel_mode": str(self.backup_mode_combo.currentData() or "race"),
                "backup_route": str(self.backup_route_combo.currentData() or "beijing-public"),
                "backup_workspace_id": self.backup_workspace_edit.text().strip(),
                "backup_model": self.backup_model_edit.text().strip(),
                "backup_tts_mode": str(self.backup_tts_mode_combo.currentData() or "delayed"),
                "backup_asr_enabled": self.backup_asr_check.isChecked(),
                "backup_asr_model": str(self.backup_asr_combo.currentData()),
                "backup_tts_enabled": self.backup_tts_check.isChecked(),
                "backup_tts_model": self.backup_tts_voice_combo.currentData()[0],
                "backup_tts_voice_id": self.backup_tts_voice_combo.currentData()[1],
                "prerender_tts_model": self._prerender_pair()[0],
                "prerender_voice_id": self._prerender_pair()[1],
                "llm_model": str(self.llm_combo.currentData()),
                "asr_model": str(self.asr_combo.currentData()),
                "committee_asr_model": str(self.committee_asr_combo.currentData()),
                "qa_model": str(self.qa_model_combo.currentData()),
                "qa_thinking": self.qa_thinking_check.isChecked(),
                "speech_early_start": self.early_start_check.isChecked(),
                **self._engine_values(),
                "vad_silence_ms": self.silence_spin.value(),
                "max_context_turns": self.turns_spin.value(),
                "teams_output_host_api": next((d.host_api for d in self._all_outputs if d.index == self.cable_combo.currentData()), ""),
                "monitor_enabled": self.monitor_check.isChecked(),
                "monitor_output_device": self.monitor_combo.currentData(),
                "monitor_output_device_name": next((d.name for d in self._all_outputs if d.index == self.monitor_combo.currentData()), ""),
                "monitor_output_host_api": next((d.host_api for d in self._all_outputs if d.index == self.monitor_combo.currentData()), ""),
                "tts_rate": self.rate_spin.value(),
                "tts_pitch": self.pitch_spin.value(),
                "tts_volume": self.volume_spin.value(),
                "tts_instruction": self.instruction_edit.text().strip(),
                "sentence_break_ms": int(self.sentence_break_spin.value()),
                "disk_cache_mb": int(self.disk_cache_spin.value()),
                "opening_filler": str(self.filler_combo.currentData() or "off"),
                "humanize_level": str(self.humanize_combo.currentData() or "off"),
                "translation_style": str(self.translation_style_combo.currentData() or "accurate"),
                "ambience_mode": str(self.ambience_combo.currentData() or "off"),
                "ambience_volume": int(self.ambience_volume_slider.value()),
                "speech_mode": self.speech_mode_combo.currentData(),
                "output_directory": self.output_dir_edit.text().strip(),
                "direct_hotkey": str(self.direct_hotkey_combo.currentData()),
                "translate_hotkey": str(self.translate_hotkey_combo.currentData()),
                "cancel_hotkey": str(self.cancel_hotkey_combo.currentData()),
            }
        )
        backup_key = self.backup_key_edit.text().strip()
        if backup_key:
            try:
                set_backup_api_key(backup_key)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "第二通道", f"第二通道 API Key 没能存进 Windows 凭据管理器：{exc}")
        self.accept()

    # ------------------------------------------------------------ 情景模式联动
    def _dialog_scene_values(self) -> dict:
        values = {
            "asr_model": str(self.asr_combo.currentData() or ""),
            "committee_asr_model": str(self.committee_asr_combo.currentData() or ""),
            "llm_model": str(self.llm_combo.currentData() or ""),
            "qa_model": str(self.qa_model_combo.currentData() or ""),
            "qa_thinking": self.qa_thinking_check.isChecked(),
            "my_translate_engine": "llm",
            "livetranslate_model": str(self.settings.get("livetranslate_model", LIVETRANSLATE_MODELS[0][0])),
        }
        values.update(self._engine_values())
        return values

    def _matching_scene(self):
        values = self._dialog_scene_values()
        for mode in scene_modes.all_modes(self.settings):
            same = True
            for key in scene_modes.SCENE_KEYS:
                if key == "livetranslate_model" and values.get("my_translate_engine") != "livetranslate":
                    continue   # 不用同传时，同传模型是哪个无所谓
                if scene_modes._normalize(key, mode.values.get(key)) != scene_modes._normalize(key, values.get(key)):
                    same = False
                    break
            if same:
                return mode
        return None

    def _fill_scene_combo(self) -> None:
        combo = self.scene_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("自定义（和任何模式都不一样）", scene_modes.CUSTOM_ID)
        for mode in scene_modes.all_modes(self.settings):
            combo.addItem(mode.name if mode.builtin else f"我的：{mode.name}", mode.id)
            combo.setItemData(combo.count() - 1, mode.blurb, Qt.ToolTipRole)
            if mode.id == "recommended":
                highlight_item(combo, combo.count() - 1)
        combo.blockSignals(False)
        self._sync_scene_combo()

    def _sync_scene_combo(self) -> None:
        mode = self._matching_scene()
        self.scene_combo.blockSignals(True)
        self.scene_combo.setCurrentIndex(max(0, self.scene_combo.findData(mode.id if mode else scene_modes.CUSTOM_ID)))
        self.scene_combo.blockSignals(False)
        self.scene_combo.setToolTip(mode.blurb if mode else "当前这几项和任何情景模式都对不上：首页会显示「自定义」")

    def _on_settings_scene_chosen(self, index: int) -> None:
        mode_id = self.scene_combo.itemData(index)
        mode = next((m for m in scene_modes.all_modes(self.settings) if m.id == mode_id), None)
        if mode is None:
            self._sync_scene_combo()
            return
        values = mode.values

        def pick(combo: QComboBox, value: str) -> None:
            index_of = combo.findData(value)
            if index_of < 0:
                combo.addItem(f"{value} · 自定义", value)
                index_of = combo.count() - 1
            combo.setCurrentIndex(index_of)

        for combo in (self.llm_combo, self.asr_combo, self.committee_asr_combo, self.qa_model_combo, self.engine_combo):
            combo.blockSignals(True)
        self.qa_thinking_check.blockSignals(True)
        try:
            pick(self.asr_combo, str(values["asr_model"]))
            pick(self.committee_asr_combo, str(values["committee_asr_model"]))
            pick(self.llm_combo, str(values["llm_model"]))
            pick(self.qa_model_combo, str(values["qa_model"]))
            self.qa_thinking_check.setChecked(bool(values["qa_thinking"]))
            key = (f"livetranslate:{values['livetranslate_model']}"
                   if values["my_translate_engine"] == "livetranslate" else "llm")
            pick(self.engine_combo, key)
        finally:
            for combo in (self.llm_combo, self.asr_combo, self.committee_asr_combo, self.qa_model_combo, self.engine_combo):
                combo.blockSignals(False)
            self.qa_thinking_check.blockSignals(False)
        self._sync_scene_combo()

    def _engine_values(self) -> dict:
        key = str(self.engine_combo.currentData() or "llm")
        if key.startswith("livetranslate:"):
            return {"my_translate_engine": "livetranslate", "livetranslate_model": key.split(":", 1)[1]}
        return {"my_translate_engine": "llm"}


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
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.ElideNone)
        self.table.horizontalHeader().sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
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
        item.setToolTip(status)
        self.table.resizeRowsToContents()
        self.table.scrollToBottom()

    @Slot(int, bool, str)
    def _finish_row(self, row: int, ok: bool, message: str) -> None:
        if self._closing:
            return
        item = QTableWidgetItem(("✅ " if ok else "❌ ") + message)
        item.setForeground(QColor(OK_GREEN if ok else ERR_RED))
        self.table.setItem(row, 1, item)
        item.setToolTip(message)
        self.table.resizeRowsToContents()

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
        has_source = _has_tts_source(settings)
        finish(0, bool(workspace), workspace or "未填写，请在设置中填写")

        finish(1, bool(api_key), "已从 Windows 凭据管理器读取" if api_key else "未设置")

        finish(2, has_source, "Next 本机参考录音已就绪" if has_source and str(settings.get("tts_model")) == "qwen-audio-3.1-tts-next" else (voice or "未填写（与 TTS 模型绑定）"))

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
        if workspace and api_key and has_source:
            try:
                voice_probe(settings, workspace, api_key, voice, str(settings.get("tts_model")), cancel_event=token)
                finish(7, True, "已在本机试听设备播放，请确认是你的音色")
            except Exception as exc:
                finish(7, False, str(exc)[:160])
        else:
            finish(7, False, "缺少配置，跳过")


class ScriptDeskPanel(QWidget):
    """讲稿朗读台：三栏（原文 / 段 / 句），嵌在主窗口标签页里。

    段是一级单位，**一段对应一页 PPT**。讲到哪一页点哪一段，光标在右栏
    按各句时长自动往前走。

    一条硬规则：点播只从本地缓存读。读不到就明说「还没渲染」，绝不临时
    去请求服务器 —— 台上点一句等五秒，比一句提示难受得多。
    """

    render_progress = Signal(object)
    render_done = Signal(object, int, int)
    translate_done = Signal(object, object, str)  # 取消令牌，英文段落列表 或 None，错误

    def __init__(self, settings: DefenseSettings, engine) -> None:
        super().__init__(None)
        self.settings = settings
        self.engine = engine
        self.document = ScriptDocument()
        self._paused_at = 0.0
        self._chinese_backup = ""
        self.translate_runner = None   # 测试可替换：run(chinese, previous_english) -> english
        self.on_playback_changed = None   # 主窗口接上：讲稿开始/停止播放时通知（切换按住说话）
        self._segment_serial = 0
        self._segment_tag = ""
        self._awaiting_start = 0.0
        self._store_path = Path(settings.base_dir) / "script_desk.json"
        self._rendering = False
        self._cancel = threading.Event()
        self._translate_cancel = threading.Event()
        # 光标跟随：一次播一段，定时器按各句时长把高亮往前推。
        self._playing: list = []
        self._bounds: list[tuple[float, float]] = []
        self._play_started = 0.0
        self._play_total = 0.0
        self._continue_all = False
        self._duration_units = {}
        self._timeline_offsets = {}
        self._timeline_total = 0.0
        self._timeline_complete = False
        self._timeline_position = 0.0
        self._timeline_segment_offset = 0.0
        self._cursor_line_index = None
        self._cursor_paragraph_index = None
        self._build_ui()
        self.render_progress.connect(self._on_render_progress, Qt.QueuedConnection)
        self.render_done.connect(self._on_render_done, Qt.QueuedConnection)
        self.translate_done.connect(self._on_translate_done, Qt.QueuedConnection)
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(80)
        self._cursor_timer.timeout.connect(self._advance_cursor)
        self._restore_saved_script()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        hint = QLabel(
            "讲稿按换行分段，**一段 = 一页 PPT**。切分后先点「渲染全部」，"
            "现场就只读本地音频，不依赖网络。讲稿和渲染好的声音都存在本机，下次打开还在。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        outer.addWidget(hint)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("渲染音色"))
        self.voice_combo = NoWheelComboBox()
        self.voice_combo.setMinimumWidth(320)
        self.voice_combo.setToolTip(
            "讲稿台、问答库、快捷用语提前渲染用的音色和合成模型（和设置里的「提前渲染音色」是同一个）。\n"
            "换一个音色后，之前渲染的声音不会删，换回来立刻能用；新音色需要再点一次渲染。"
        )
        self.voice_combo.activated.connect(self._on_voice_chosen)
        voice_row.addWidget(self.voice_combo, 2)
        self.voice_lock_button = QPushButton("")
        self.voice_lock_button.setCheckable(True)
        self.voice_lock_button.setMinimumWidth(96)
        self.voice_lock_button.toggled.connect(self._on_voice_lock_toggled)
        voice_row.addWidget(self.voice_lock_button)
        voice_row.addSpacing(16)
        voice_row.addWidget(QLabel("句间停顿"))
        self.gap_combo = NoWheelComboBox()
        for ms in (0, 200, 350, 500, 700):
            self.gap_combo.addItem(f"{ms} ms" + ("（推荐）" if ms == 350 else ""), ms)
        self.gap_combo.setToolTip("逐句合成后拼成一段时，句与句之间留的换气停顿（期间是背景底噪，不是死寂）")
        self.gap_combo.setCurrentIndex(max(0, self.gap_combo.findData(int(self.settings.get("script_gap_ms", 350) or 0))))
        self.gap_combo.currentIndexChanged.connect(
            lambda _i: self.settings.set("script_gap_ms", int(self.gap_combo.currentData())))
        self.gap_combo.currentIndexChanged.connect(lambda _i: self._rebuild_timeline())
        voice_row.addWidget(self.gap_combo)
        voice_row.addWidget(QLabel("段间停顿"))
        self.paragraph_gap_combo = NoWheelComboBox()
        for ms in (300, 500, 800, 1200, 1800):
            self.paragraph_gap_combo.addItem(f"{ms} ms" + ("（推荐）" if ms == 800 else ""), ms)
        self.paragraph_gap_combo.setToolTip("连播时换一页 PPT 之间留的停顿")
        self.paragraph_gap_combo.setCurrentIndex(max(0, self.paragraph_gap_combo.findData(
            int(self.settings.get("script_paragraph_gap_ms", 800) or 800))))
        self.paragraph_gap_combo.currentIndexChanged.connect(
            lambda _i: self.settings.set("script_paragraph_gap_ms", int(self.paragraph_gap_combo.currentData())))
        self.paragraph_gap_combo.currentIndexChanged.connect(lambda _i: self._rebuild_timeline())
        voice_row.addWidget(self.paragraph_gap_combo)
        voice_row.addStretch(1)
        outer.addLayout(voice_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("语气"))
        self.instruction_mode_combo = NoWheelComboBox()
        self.instruction_mode_combo.addItem("和声音对比 B 一样（Qwen 使用复刻原声，语速 1.0）", "b")
        self.instruction_mode_combo.addItem("和声音对比 A / 原始一样（不加语气，语速 1.0）", "none")
        self.instruction_mode_combo.addItem("跟随设置里的语气指令和语速", "follow")
        self.instruction_mode_combo.setToolTip(
            "声音对比 B 自然模式：Qwen 使用复刻原声，不加语气指令、不加工文字；语速音调 1.0。\n"
            "选第一项后讲稿台用的就是同一套参数（音色请在上面选成和 B 相同的那个）。\n"
            "换了之后已渲染的句子照样能播（显示「旧设置」），想用新设置点「按当前设置重渲染」。"
        )
        self.instruction_mode_combo.setCurrentIndex(max(0, self.instruction_mode_combo.findData(
            str(self.settings.get("prerender_instruction", "b") or "b"))))
        self.instruction_mode_combo.activated.connect(self._on_instruction_mode_chosen)
        style_row.addWidget(self.instruction_mode_combo, 2)
        style_row.addSpacing(16)
        style_row.addWidget(QLabel("合成方式"))
        self.render_unit_combo = NoWheelComboBox()
        self.render_unit_combo.addItem("整段一起合成（和 B 一样连贯，推荐）", "group")
        self.render_unit_combo.addItem("逐句合成再拼接（旧方式）", "sentence")
        self.render_unit_combo.setToolTip(
            "整段合成：一段（一页 PPT，超过 450 字符才拆开）一次请求读完，句与句之间的换气和语调由模型自己接，"
            "和声音对比 B 一模一样；逐句光标、单句点播照常可用（自动切在句间停顿里）。\n"
            "逐句合成：每句单独合成再拼起来，每句都是「冷启动」，名字等短句最容易读怪。"
        )
        self.render_unit_combo.setCurrentIndex(max(0, self.render_unit_combo.findData(
            str(self.settings.get("script_render_unit", "group") or "group"))))
        self.render_unit_combo.activated.connect(self._on_render_unit_chosen)
        style_row.addWidget(self.render_unit_combo, 1)
        style_row.addStretch(1)
        outer.addLayout(style_row)
        self.voice_params_label = QLabel("")
        self.voice_params_label.setWordWrap(True)
        self.voice_params_label.setStyleSheet(f"color: {TEXT_DIM};")
        outer.addWidget(self.voice_params_label)
        self.reload_voice_options()

        columns = QSplitter(Qt.Horizontal)
        outer.addWidget(columns, 1)

        # ---------------- 左栏：原文 ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("语种"))
        self.language_combo = NoWheelComboBox()
        self.language_combo.addItem("自动判断", "")
        self.language_combo.addItem("英文", "en")
        self.language_combo.addItem("中文", "zh")
        lang_row.addWidget(self.language_combo, 1)
        left_layout.addLayout(lang_row)
        left_layout.addWidget(QLabel("讲稿原文（换行分段，中英文均可）"))
        self.source_edit = QPlainTextEdit()
        self.source_edit.setPlaceholderText(
            "每按一次回车就是新的一段，对应下一页 PPT。\n"
            "中英文都可以直接粘贴，不需要先翻译。"
        )
        left_layout.addWidget(self.source_edit, 1)
        source_row = QHBoxLayout()
        load_button = QPushButton("导入文件…")
        load_button.clicked.connect(self._load_file)
        source_row.addWidget(load_button)
        split_button = QPushButton("切分")
        split_button.clicked.connect(self._split_clicked)
        source_row.addWidget(split_button)
        self.translate_button = QPushButton("中文稿翻成英文")
        self.translate_button.setToolTip("把左边的中文讲稿逐段翻成英文（带术语表和论文背景，前后连贯），段落不变；原来的中文可以撤销找回")
        self.translate_button.clicked.connect(self._translate_source)
        source_row.addWidget(self.translate_button)
        self.undo_translate_button = QPushButton("撤销翻译")
        self.undo_translate_button.setEnabled(False)
        self.undo_translate_button.clicked.connect(self._undo_translate)
        source_row.addWidget(self.undo_translate_button)
        left_layout.addLayout(source_row)
        self.clear_script_button = QPushButton("清空讲稿")
        self.clear_script_button.setToolTip("清空原文、段落和句子，停止讲稿播放与后台处理；原稿自动备份，已渲染音频保留。")
        self.clear_script_button.clicked.connect(self._clear_script)
        left_layout.addWidget(self.clear_script_button)
        columns.addWidget(left)

        # ---------------- 中栏：段 ----------------
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(6, 0, 6, 0)
        paragraph_header = QHBoxLayout()
        paragraph_title = QLabel("段落 / PPT 页（单击查看，双击朗读整段；右键：重做 / 删音频）")
        paragraph_title.setWordWrap(True)
        paragraph_header.addWidget(paragraph_title, 1)
        self.timing_label = QLabel("总时长 00:00\n当前位置 00:00 · 剩余 00:00")
        self.timing_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timing_label.setStyleSheet(f"color: {OK_GREEN}; font-weight: bold;")
        self.timing_label.setToolTip(
            "按本地真实音频计算，包含实际插入的句间、段间停顿。\n"
            "当前位置是全稿中的播放位置；从中段开始时，前面的内容视为跳过。\n"
            "剩余时间是从当前位置到全稿末尾的音频时长，不包含暂停和手动等待。"
        )
        paragraph_header.addWidget(self.timing_label)
        middle_layout.addLayout(paragraph_header)
        self.paragraph_table = QTableWidget(0, 3)
        self.paragraph_table.setHorizontalHeaderLabels(["页", "段落开头", "状态"])
        self.paragraph_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.paragraph_table.verticalHeader().setVisible(False)
        self.paragraph_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.paragraph_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.paragraph_table.itemSelectionChanged.connect(self._on_paragraph_selected)
        self.paragraph_table.doubleClicked.connect(self._play_clicked_paragraph)
        self.paragraph_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.paragraph_table.customContextMenuRequested.connect(self._paragraph_menu)
        middle_layout.addWidget(self.paragraph_table, 1)
        columns.addWidget(middle)

        # ---------------- 右栏：句 ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.addWidget(QLabel("本段句子（双击朗读；右键：单独重做 / 换语气 / 删除）"))
        self.line_table = QTableWidget(0, 3)
        self.line_table.setHorizontalHeaderLabels(["#", "句子", "状态"])
        self.line_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.line_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.line_table.setColumnWidth(2, 330)
        self.line_table.setTextElideMode(Qt.ElideNone)
        self._line_resize_timer = QTimer(self)
        self._line_resize_timer.setSingleShot(True)
        self._line_resize_timer.setInterval(30)
        self._line_resize_timer.timeout.connect(self.line_table.resizeRowsToContents)
        self.line_table.horizontalHeader().sectionResized.connect(lambda *_: self._line_resize_timer.start())
        self.line_table.verticalHeader().setVisible(False)
        self.line_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.line_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.line_table.setWordWrap(True)
        self.line_table.doubleClicked.connect(self._play_clicked_line)
        self.line_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.line_table.customContextMenuRequested.connect(self._line_menu)
        right_layout.addWidget(self.line_table, 1)
        columns.addWidget(right)
        columns.setSizes([380, 300, 420])

        self.progress_label = QLabel("还没有导入讲稿")
        self.progress_label.setWordWrap(True)
        outer.addWidget(self.progress_label)

        play_row = QHBoxLayout()
        for label, slot, tip in (
            ("▶ 本句", self._play_line, "只朗读右栏选中的那一句"),
            ("▶ 本段", self._play_paragraph, "朗读整段（一页 PPT），光标逐句跟随"),
            ("▶▶ 从本段连播", self._play_from_here, "从当前段一直播到结尾，段与段自动衔接"),
            ("⏸ 暂停", self._toggle_pause, "暂停朗读，再点一次从停下的地方接着读"),
            ("⏭ 下一段", self._next_paragraph, "选中下一段但不朗读"),
            ("■ 停止", self._stop_playback, "立刻停止朗读"),
        ):
            button = QPushButton(label)
            button.setMinimumHeight(36)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            play_row.addWidget(button)
            if slot == self._toggle_pause:
                self.pause_button = button
        outer.addLayout(play_row)

        render_row = QHBoxLayout()
        self.render_all_button = QPushButton("渲染全部（答辩前做一次）")
        self.render_all_button.setMinimumHeight(34)
        self.render_all_button.clicked.connect(self._render_all)
        render_row.addWidget(self.render_all_button)
        self.render_one_button = QPushButton("只渲染本段")
        self.render_one_button.clicked.connect(self._render_paragraph)
        render_row.addWidget(self.render_one_button)
        self.reference_render_button = QPushButton("原声重做本段")
        self.reference_render_button.setToolTip("按当前渲染音色的 A 原声参数生成本段；缺少对应音频时产生合成费用。不加语气或文字加工，旧音频保留")
        self.reference_render_button.clicked.connect(self._render_reference_paragraph)
        render_row.addWidget(self.reference_render_button)
        self.rerender_stale_button = QPushButton("按当前设置重渲染")
        self.rerender_stale_button.setToolTip(
            "这些句子是用同一个音色、但以前的语速/音量/语气/小失误设置渲染的，现在照样能播。\n"
            "想让它们也用上新设置就点这里（会重新请求服务器，按字数计费）；不点也完全不影响答辩。"
        )
        self.rerender_stale_button.setVisible(False)
        self.rerender_stale_button.clicked.connect(self._rerender_stale)
        render_row.addWidget(self.rerender_stale_button)
        self.cancel_render_button = QPushButton("停止渲染")
        self.cancel_render_button.setVisible(False)
        self.cancel_render_button.clicked.connect(lambda: self._cancel.set())
        render_row.addWidget(self.cancel_render_button)
        outer.addLayout(render_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------- 导入切分
    def _load_file(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "导入讲稿", "", "文本 (*.txt *.md);;所有文件 (*.*)"
        )
        if not path:
            return
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                self.source_edit.setPlainText(Path(path).read_text(encoding=encoding))
                self._set_status(f"已导入 {Path(path).name}", TEXT_DIM)
                return
            except (UnicodeDecodeError, LookupError):
                continue
            except OSError as exc:
                self._set_status(f"读取失败：{exc}", ERR_RED)
                return
        self._set_status("无法识别文件编码，请另存为 UTF-8 后重试", ERR_RED)

    def _split_clicked(self) -> None:
        text = self.source_edit.toPlainText().strip()
        if not text:
            if self.document.paragraphs:
                self._clear_script()
                return
            self._set_status("请先粘贴或导入讲稿", ERR_RED)
            return
        self._invalidate_script_tasks()
        if self._playing:
            self._stop_playback()
        self._timeline_position = 0.0
        self.document = ScriptDocument.from_text(
            text, language=str(self.language_combo.currentData() or "")
        )
        if not self.document.paragraphs:
            self._set_status("没有切出任何段落，请检查讲稿内容", ERR_RED)
            return
        # 切分后查一遍本地缓存：之前渲染过的句子不必重来（时长也按本地音频算准）。
        self.refresh_render_state()
        self._save_script()
        if self.paragraph_table.rowCount():
            self.paragraph_table.selectRow(0)
        label = LANGUAGE_LABELS.get(self.document.language, self.document.language)
        self._set_status(
            f"已切分为 {len(self.document.paragraphs)} 段 / {len(self.document)} 句（按{label}合成）",
            OK_GREEN,
        )

    # --------------------------------------------------------------- 刷新
    def _refresh_paragraphs(self) -> None:
        keep = self.paragraph_table.currentRow()
        self.paragraph_table.setRowCount(len(self.document.paragraphs))
        for row, paragraph in enumerate(self.document.paragraphs):
            self.paragraph_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            title = QTableWidgetItem(paragraph.title)
            self.paragraph_table.setItem(row, 1, title)
            status = QTableWidgetItem(paragraph.status_text)
            status.setForeground(QColor(OK_GREEN if paragraph.ready else TEXT_DIM))
            self.paragraph_table.setItem(row, 2, status)
        self.paragraph_table.resizeColumnToContents(0)
        if 0 <= keep < self.paragraph_table.rowCount():
            self.paragraph_table.selectRow(keep)
        self.progress_label.setText(self.document.progress_text())
        self._refresh_lines()
        self._rebuild_timeline()

    @staticmethod
    def _format_script_time(seconds: float) -> str:
        value = max(0, int(seconds + 0.5))
        hours, rest = divmod(value, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def _rebuild_timeline(self) -> None:
        # Only metadata here: never read/cache whole audio buffers on the UI timer.
        clock = 0.0
        offsets = {}
        complete = True
        gap = max(0, int(self.settings.get("script_gap_ms", 350) or 0)) / 1000
        paragraph_gap = max(0, int(self.settings.get("script_paragraph_gap_ms", 800) or 0)) / 1000
        for position, paragraph in enumerate(self.document.paragraphs):
            if position:
                clock += paragraph_gap
            previous_unit = None
            for number, line in enumerate(paragraph.lines):
                unit = self._duration_units.get(line.index)
                ready = line.rendered and line.seconds > 0 and unit is not None
                complete = complete and ready
                if number and unit is not None and previous_unit is not None and unit != previous_unit:
                    clock += gap
                offsets[line.index] = clock
                if ready:
                    clock += line.seconds
                previous_unit = unit
        self._timeline_offsets = offsets
        self._timeline_total = clock
        self._timeline_complete = complete
        self._timeline_position = min(self._timeline_position, clock)
        self._update_timing_label()

    def _update_timing_label(self) -> None:
        fmt = self._format_script_time
        if not self._timeline_complete:
            self.timing_label.setText(
                f"总时长待定（音频未齐）\n已缓存 {fmt(sum(l.seconds for l in self.document.lines if l.rendered))} · 剩余 --:--"
            )
            return
        position = min(self._timeline_total, max(0.0, self._timeline_position))
        self.timing_label.setText(
            f"总时长 {fmt(self._timeline_total)}\n当前位置 {fmt(position)} · 剩余 {fmt(self._timeline_total - position)}"
        )

    def _sync_timing_position(self, state=None) -> None:
        if state is None:
            state = self.engine.cached_playback_state()
        if state.get("tag") == self._segment_tag and self._segment_tag:
            played = min(self._play_total, max(0.0, float(state.get("played", 0.0))))
            self._timeline_position = self._timeline_segment_offset + played
            self._update_timing_label()

    def _refresh_lines(self) -> None:
        paragraph = self._current_paragraph()
        lines = paragraph.lines if paragraph else []
        keep = self.line_table.currentRow()
        self.line_table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            self.line_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            preview = QTableWidgetItem(line.text)
            self.line_table.setItem(row, 1, preview)
            status = QTableWidgetItem(line.status_text)
            status.setToolTip(line.error or line.status_text)
            status.setForeground(
                QColor(ERR_RED if line.error else (OK_GREEN if line.rendered else TEXT_DIM))
            )
            self.line_table.setItem(row, 2, status)
        self.line_table.resizeColumnToContents(0)
        self.line_table.resizeRowsToContents()
        if 0 <= keep < self.line_table.rowCount():
            self.line_table.selectRow(keep)

    def _on_paragraph_selected(self) -> None:
        self._refresh_lines()
        if self.line_table.rowCount():
            self.line_table.selectRow(0)

    def _current_paragraph(self):
        return self.document.paragraph_at(self.paragraph_table.currentRow())

    def _current_line(self):
        paragraph = self._current_paragraph()
        row = self.line_table.currentRow()
        if paragraph is None or not 0 <= row < len(paragraph.lines):
            return None
        return paragraph.lines[row]

    # --------------------------------------------------------------- 朗读
    def _collect_pcm(self, lines, *, lead_ms: int = 0):
        """把若干句的本地音频拼起来，句与句之间插入换气停顿。返回 (pcm, 每句起止秒)。

        缺任何一句就整体判失败：缺一句还硬播，现场就是讲到一半突然跳过一句。
        起止秒按真实音频长度算——以前用记录的时长，缓存里读出来的句子时长是 0，
        结果光标不动、连播时一开始就判定「播完了」直接跳到下一段。
        """
        rate = SPEECH_SAMPLE_RATE * 2
        gap = bytes(int(SPEECH_SAMPLE_RATE * max(0, int(self.settings.get("script_gap_ms", 350) or 0)) / 1000) * 2)
        parts: list[bytes] = []
        bounds: list[tuple[float, float]] = []
        clock = 0.0
        if lead_ms > 0:
            lead = bytes(int(SPEECH_SAMPLE_RATE * lead_ms / 1000) * 2)
            parts.append(lead)
            clock += len(lead) / rate
        audio: dict = {}
        for paragraph_index in dict.fromkeys(line.paragraph for line in lines):
            paragraph = self.document.paragraph_at(paragraph_index)
            if paragraph is not None:
                audio.update(self._paragraph_audio(paragraph))
        self._duration_units.update({index: entry[0] for index, entry in audio.items()})
        previous_unit = None
        for position, line in enumerate(lines):
            entry = audio.get(line.index)
            pcm = entry[1] if entry else b""
            unit = entry[0] if entry else None
            if not pcm:
                line.rendered = False
                self._refresh_paragraphs()
                self._set_status(
                    f"第 {line.paragraph + 1} 段第 {line.order + 1} 句用当前音色还没渲染，请先渲染本段", WARN_ORANGE
                )
                return None, []
            # 同一块整段合成切出来的句子首尾相接，不插停顿（停顿本来就在音频里）
            if position and gap and unit != previous_unit:
                parts.append(gap)
                clock += len(gap) / rate
            previous_unit = unit
            seconds = len(pcm) / rate
            line.seconds = seconds
            bounds.append((clock, clock + seconds))
            parts.append(pcm)
            clock += seconds
        self._rebuild_timeline()
        return (b"".join(parts) if parts else None), bounds

    def _start_sequence(self, lines, *, continue_all: bool, lead_ms: int = 0) -> None:
        pcm, bounds = self._collect_pcm(lines, lead_ms=lead_ms)
        if pcm is None:
            if not self._playing:
                self._notify_playback(False)
            return
        self._playing = list(lines)
        self._bounds = bounds
        self._cursor_line_index = lines[0].index
        self._cursor_paragraph_index = lines[0].paragraph
        self.paragraph_table.blockSignals(True)
        self.paragraph_table.selectRow(lines[0].paragraph)
        self.paragraph_table.blockSignals(False)
        self._refresh_lines()
        self._play_total = len(pcm) / (SPEECH_SAMPLE_RATE * 2)
        self._timeline_segment_offset = self._timeline_offsets[lines[0].index] - bounds[0][0]
        self._timeline_position = self._timeline_segment_offset
        self._update_timing_label()
        self._continue_all = continue_all
        self._paused_at = 0.0
        self.pause_button.setText("⏸ 暂停")
        self._segment_serial += 1
        self._segment_tag = f"script-{self._segment_serial}"
        self._awaiting_start = time.monotonic()
        self.engine.play_script_segment(pcm, text=" ".join(l.text for l in lines), tag=self._segment_tag)
        self._cursor_timer.start()
        self._highlight(lines[0].order if lines else 0)
        self._notify_playback(True)

    def is_playing(self) -> bool:
        return bool(self._playing) and not self._paused_at

    def _notify_playback(self, playing: bool) -> None:
        # These gaps have already been baked into the current PCM buffer.
        self.gap_combo.setEnabled(not self._playing)
        self.paragraph_gap_combo.setEnabled(not self._playing)
        if self.on_playback_changed is not None:
            try:
                self.on_playback_changed(playing)
            except Exception:
                pass

    def _highlight(self, position: int) -> None:
        """把右栏光标移到段内第 position 句。"""
        if not 0 <= position < self.line_table.rowCount():
            return
        self.line_table.selectRow(position)
        item = self.line_table.item(position, 1)
        if item is not None:
            self.line_table.scrollToItem(item)

    def _advance_cursor(self) -> None:
        """光标跟着引擎的真实播放进度走：暂停时不动，播完才换下一段。"""
        if not self._playing:
            self._cursor_timer.stop()
            return
        state = self.engine.cached_playback_state()
        mine = state.get("tag") == self._segment_tag
        if self._awaiting_start:
            if mine:
                self._awaiting_start = 0.0
            elif time.monotonic() - self._awaiting_start > 8.0:
                self._awaiting_start = 0.0
                self._cursor_timer.stop()
                self._playing = []
                self._notify_playback(False)
                self._set_status("播放没有开始（可能被其他语音打断或设备异常），请再点一次", WARN_ORANGE)
                return
            else:
                return
        if not mine:
            # 被别的语音（助答、快捷用语、Esc）打断了：光标停下，不自动续播
            self._cursor_timer.stop()
            self._playing = []
            self._continue_all = False
            self._notify_playback(False)
            return
        elapsed = float(state.get("played", 0.0))
        self._sync_timing_position(state)
        if not state.get("active"):
            self._cursor_timer.stop()
            finished = self._playing
            self._playing = []
            if self._continue_all and elapsed >= self._play_total - 0.2:
                self._advance_to_next_paragraph(finished)
            else:
                self._notify_playback(False)
            return
        index = line_at_elapsed(self._bounds, elapsed)
        line = self._playing[index] if 0 <= index < len(self._playing) else None
        if line is None:
            return
        # 连播跨段时，中栏也要跟着翻页
        if line.paragraph != self._cursor_paragraph_index:
            self._cursor_paragraph_index = line.paragraph
            self.paragraph_table.blockSignals(True)
            self.paragraph_table.selectRow(line.paragraph)
            self.paragraph_table.blockSignals(False)
            self._refresh_lines()
        if line.index != self._cursor_line_index:
            self._cursor_line_index = line.index
            if line.paragraph == self.paragraph_table.currentRow():
                self._highlight(line.order)

    def _advance_to_next_paragraph(self, finished) -> None:
        last = finished[-1] if finished else None
        nxt = (last.paragraph + 1) if last else 0
        paragraph = self.document.paragraph_at(nxt)
        if paragraph is None:
            self._set_status("已讲到最后一段", OK_GREEN)
            self._notify_playback(False)
            return
        self.paragraph_table.blockSignals(True)
        self.paragraph_table.selectRow(nxt)
        self.paragraph_table.blockSignals(False)
        self._refresh_lines()
        gap = int(self.settings.get("script_paragraph_gap_ms", 800) or 0)
        self._start_sequence(paragraph.lines, continue_all=True, lead_ms=gap)

    def _play_line(self) -> None:
        line = self._current_line()
        if line is None:
            self._set_status("请先在右栏选中一句", WARN_ORANGE)
            return
        self._start_sequence([line], continue_all=False)

    def _play_clicked_line(self, index) -> None:
        paragraph = self._current_paragraph()
        if paragraph is not None and 0 <= index.row() < len(paragraph.lines):
            self._start_sequence([paragraph.lines[index.row()]], continue_all=False)

    def _play_clicked_paragraph(self, index) -> None:
        # Use the double-click event's row, never the timer-controlled selection.
        paragraph = self.document.paragraph_at(index.row())
        if paragraph is not None:
            self._start_sequence(paragraph.lines, continue_all=False)

    def _play_paragraph(self) -> None:
        paragraph = self._current_paragraph()
        if paragraph is None:
            self._set_status("请先在中栏选中一段", WARN_ORANGE)
            return
        self._start_sequence(paragraph.lines, continue_all=False)

    def _play_from_here(self) -> None:
        paragraph = self._current_paragraph()
        if paragraph is None:
            self._set_status("请先在中栏选中一段", WARN_ORANGE)
            return
        self._start_sequence(paragraph.lines, continue_all=True)

    def _next_paragraph(self) -> None:
        row = self.paragraph_table.currentRow()
        nxt = row + 1 if row >= 0 else 0
        if nxt >= len(self.document.paragraphs):
            self._set_status("已经是最后一段了", WARN_ORANGE)
            return
        self.paragraph_table.selectRow(nxt)

    def _stop_playback(self) -> None:
        if self._playing:
            self._sync_timing_position()
        self._cursor_timer.stop()
        was_playing = bool(self._playing)
        self._playing = []
        self._continue_all = False
        self._paused_at = 0.0
        self._segment_tag = ""
        self.pause_button.setText("⏸ 暂停")
        self.engine.interrupt_speech()   # 同时熄掉字幕悬浮窗
        if was_playing:
            self._notify_playback(False)
        self._set_status("已停止朗读", TEXT_DIM)

    def _toggle_pause(self) -> None:
        """暂停：声音、右栏光标、字幕悬浮窗一起停；继续从停下的地方接着走。"""
        if self._paused_at:
            self.resume()
            return
        if not self._playing:
            self._set_status("现在没有在朗读", WARN_ORANGE)
            return
        self.pause()

    def pause(self) -> None:
        if not self._playing or self._paused_at:
            return
        self.engine.pause_speech()
        self._sync_timing_position()
        self._paused_at = time.monotonic()
        self.pause_button.setText("▶ 继续")
        self._set_status("已暂停：点「继续」接着读", WARN_ORANGE)
        self._notify_playback(False)

    def resume(self) -> None:
        if not self._paused_at:
            return
        self.engine.resume_speech()
        self._paused_at = 0.0
        self.pause_button.setText("⏸ 暂停")
        if self._playing:
            self._cursor_timer.start()
            self._notify_playback(True)
        self._set_status("继续朗读", TEXT_DIM)

    def on_external_speech_control(self, action: str) -> None:
        """别处按了暂停/继续/Esc：按钮文字跟着变。"""
        if action == "stop" and self._playing and not self._awaiting_start:
            self._sync_timing_position()
            self._cursor_timer.stop()
            self._playing = []
            self._continue_all = False
            self._paused_at = 0.0
            self.pause_button.setText("⏸ 暂停")
            self._notify_playback(False)
        elif action == "pause" and self._playing and not self._paused_at:
            self._paused_at = time.monotonic()
            self.pause_button.setText("▶ 继续")
        elif action == "resume" and self._paused_at:
            self._paused_at = 0.0
            self.pause_button.setText("⏸ 暂停")

    # --------------------------------------------------------------- 渲染音色
    def reload_voice_options(self) -> None:
        """音色下拉：音色库里的全部音色 + 「和现场发声相同」。"""
        combo = self.voice_combo
        combo.blockSignals(True)
        combo.clear()
        live_model = str(self.settings.get("tts_model", "") or "")
        combo.addItem(f"和现场发声相同（{live_model or '未设置'}）", ("", ""))
        last_b = self.settings.get("voice_ab_last_b", {}) or {}
        for option in scene_modes.voice_options(self.settings):
            label = option["label"]
            if option["model"] in RECOMMENDED_TTS:
                label = "★ " + label
            if isinstance(last_b, dict) and last_b.get("voice_id") == option["voice_id"] \
                    and last_b.get("model") == option["model"]:
                label += "  ← 声音对比 B 刚用过"
            combo.addItem(label, (option["model"], option["voice_id"]))
            if option["model"] in RECOMMENDED_TTS:
                highlight_item(combo, combo.count() - 1)
        current = (str(self.settings.get("prerender_tts_model", "") or ""),
                   str(self.settings.get("prerender_voice_id", "") or ""))
        index = next((i for i in range(combo.count()) if tuple(combo.itemData(i)) == current), 0)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        locked = bool(self.settings.get("script_voice_locked", True))
        self.voice_lock_button.blockSignals(True)
        self.voice_lock_button.setChecked(locked)
        self.voice_lock_button.blockSignals(False)
        self._apply_voice_lock(locked)
        self._update_voice_params()

    def _on_instruction_mode_chosen(self, _index: int) -> None:
        self.settings.set("prerender_instruction", str(self.instruction_mode_combo.currentData() or "b"))
        self.refresh_render_state()
        self._set_status("已切换语气；已渲染的句子照样能播，想听新效果点「按当前设置重渲染」", OK_GREEN)

    def _on_render_unit_chosen(self, _index: int) -> None:
        self.settings.set("script_render_unit", str(self.render_unit_combo.currentData() or "group"))
        self.refresh_render_state()
        self._set_status("已切换合成方式；以前逐句渲染的句子照样能播（显示「旧设置」），"
                         "点「按当前设置重渲染」改成整段合成", OK_GREEN)

    # ---------------- 整段 / 逐句 两种合成方式共用的取音频逻辑 ----------------
    def _group_mode(self) -> bool:
        unit = getattr(self.engine, "script_render_unit", None)
        try:
            return callable(unit) and hasattr(self.engine, "script_group_slices") and unit() == "group"
        except Exception:  # noqa: BLE001
            return False

    def _units_of(self, paragraph) -> list:
        """一段拆成几块合成：整段模式按引擎的分块（通常整段一块），逐句模式一句一块。"""
        lines = list(paragraph.lines)
        if not self._group_mode():
            return [[line] for line in lines]
        try:
            groups = self.engine.script_groups([line.text for line in lines])
        except Exception:  # noqa: BLE001
            return [[line] for line in lines]
        units, position = [], 0
        for group in groups:
            units.append(lines[position:position + len(group)])
            position += len(group)
        if position < len(lines):
            units.append(lines[position:])
        return [unit for unit in units if unit]

    def _paragraph_audio(self, paragraph) -> dict:
        """{line.index: (块标识, 这句的音频, 是否旧设置)}。只读本地，不请求服务器。"""
        result: dict = {}
        group_mode = self._group_mode()
        state_of = getattr(self.engine, "script_render_state", None)
        for number, unit in enumerate(self._units_of(paragraph)):
            if group_mode:
                texts = [line.text for line in unit]
                language = unit[0].language
                try:
                    slices = self.engine.script_group_slices(texts, language)
                except Exception:  # noqa: BLE001
                    slices = None
                if isinstance(slices, list) and len(slices) == len(unit) and all(slices):
                    stale = self.engine.script_group_state(texts, language) == "saved"
                    for line, pcm in zip(unit, slices):
                        result[line.index] = ((paragraph.index, number), pcm, stale)
                    continue
            # 逐句模式；或整段还没渲染时，退回以前逐句渲染好的音频（照样能播）
            for line in unit:
                pcm = self.engine.script_pcm(line.text, line.language)
                if not pcm:
                    continue
                if group_mode:
                    stale = True
                else:
                    try:
                        stale = callable(state_of) and state_of(line.text, line.language) == "saved"
                    except Exception:  # noqa: BLE001
                        stale = False
                result[line.index] = (("line", line.index), pcm, stale)
        # 单独重做 / 换过语气的句子优先；前后留一点换气（块标识不同会插句间停顿）
        override_of = getattr(self.engine, "script_override", None)
        if callable(override_of):
            for line in paragraph.lines:
                try:
                    found = override_of(line.text, line.language)
                except Exception:  # noqa: BLE001
                    found = None
                if (isinstance(found, tuple) and len(found) == 2
                        and isinstance(found[0], (bytes, bytearray)) and found[0]):
                    result[line.index] = (("override", line.index), bytes(found[0]), False, str(found[1]))
        return result

    def _update_voice_params(self) -> None:
        """把讲稿台实际用的合成参数写出来，方便和「声音对比」逐项对照。"""
        label = getattr(self, "voice_params_label", None)
        if label is None:
            return
        try:
            values = self.engine.script_settings("en")
        except Exception:  # noqa: BLE001
            label.setText("")
            return
        voice = str(values.get("voice", "") or "")
        instruction = str(values.get("tts_instruction", "") or "")
        label.setText(
            f"讲稿台实际参数：模型 {values.get('tts_model') or '未设置'} · 音色 …{voice[-10:] or '未设置'} · "
            f"语速 {values.get('tts_rate')} · 音调 {values.get('tts_pitch')} · "
            f"语气指令 {('「' + instruction[:40] + ('…' if len(instruction) > 40 else '') + '」') if instruction else '无'}"
            + ("　（与 A 原声参数一致：不加语气指令、不加工文本；已有缓存需重做才会变化）"
               if self.settings.get("prerender_instruction", "b") == "none" else
               "　（已有缓存保留原来的声音；新参数需重做相应段落才生效）")
        )

    def _apply_voice_lock(self, locked: bool) -> None:
        self.voice_combo.setEnabled(not locked)
        self.voice_lock_button.setText("🔒 已锁定" if locked else "🔓 可切换")
        self.voice_lock_button.setToolTip("锁定时不能换音色，防止误触；点一下解锁" if locked else "点一下重新锁上")

    def _on_voice_lock_toggled(self, locked: bool) -> None:
        self.settings.set("script_voice_locked", bool(locked))
        self._apply_voice_lock(locked)

    def _on_voice_chosen(self, index: int) -> None:
        if bool(self.settings.get("script_voice_locked", True)):
            self.reload_voice_options()
            return
        model, voice = self.voice_combo.itemData(index) or ("", "")
        self.settings.update({"prerender_tts_model": model, "prerender_voice_id": voice})
        self.voice_lock_button.setChecked(True)   # 换完自动锁回
        self.refresh_render_state()
        ready = self.document.rendered_count
        self._set_status(
            f"已换成「{self.voice_combo.currentText()}」：本机已有这个音色的 {ready}/{len(self.document)} 句，"
            "其余点「渲染」生成（原来音色的声音还在，换回去立刻能用）", OK_GREEN)

    def refresh_render_state(self) -> None:
        """按当前渲染音色重新查一遍本地音频：状态和时长都以本机实际文件为准。

        以前只在渲染时记一个「已渲染」标记：换了音色后标记还在，于是「渲染」说已渲染、
        「播放」又说没渲染。
        """
        rate = SPEECH_SAMPLE_RATE * 2
        self._duration_units = {}
        for paragraph in self.document.paragraphs:
            audio = self._paragraph_audio(paragraph)
            self._duration_units.update({index: entry[0] for index, entry in audio.items()})
            for line in paragraph.lines:
                entry = audio.get(line.index)
                pcm = entry[1] if entry else b""
                line.rendered = bool(pcm)
                line.seconds = len(pcm) / rate if pcm else 0.0
                line.stale = bool(entry and entry[2])
                line.custom = entry[3] if entry and len(entry) > 3 else ""
                if pcm:
                    line.error = ""
        self._refresh_paragraphs()
        self._update_stale_button()
        self._update_voice_params()

    def _stale_lines(self) -> list:
        return [line for line in self.document.lines if line.rendered and getattr(line, "stale", False)]

    def _update_stale_button(self) -> None:
        button = getattr(self, "rerender_stale_button", None)
        if button is None:
            return
        stale = len(self._stale_lines())
        button.setVisible(bool(stale))
        button.setText(f"按当前设置重渲染 {stale} 句")

    def _rerender_stale(self) -> None:
        self.refresh_render_state()
        stale = self._stale_lines()
        if not stale:
            self._set_status("没有需要按新设置重渲染的句子", OK_GREEN)
            return
        self._start_render(stale)

    # --------------------------------------------------------------- 本地保存讲稿
    def _write_script(self, data: dict) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._store_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self._store_path)

    def _save_script(self) -> None:
        try:
            self._write_script({
                "text": self.source_edit.toPlainText(),
                "language": str(self.language_combo.currentData() or ""),
            })
        except OSError as exc:
            self._set_status(f"讲稿保存失败：{exc}", ERR_RED)

    def _invalidate_script_tasks(self) -> None:
        self._cancel.set()
        self._cancel = threading.Event()
        self._translate_cancel.set()
        self._translate_cancel = threading.Event()
        self._rendering = False
        self.render_all_button.setEnabled(True)
        self.render_one_button.setEnabled(True)
        self.cancel_render_button.setVisible(False)
        self.translate_button.setEnabled(True)

    def _clear_script(self) -> None:
        text = self.source_edit.toPlainText()
        if not text.strip():
            text = "\n".join(paragraph.text for paragraph in self.document.paragraphs)
        backup = None
        try:
            if text.strip():
                directory = self._store_path.parent / "script-backups"
                directory.mkdir(parents=True, exist_ok=True)
                backup = directory / f"before-clear-{time.time_ns()}.txt"
                backup.write_text(text, encoding="utf-8")
            # Persist first: do not claim success when the old script cannot be replaced.
            self._write_script({"text": "", "language": ""})
        except OSError as exc:
            self._set_status(f"清空讲稿失败，原稿仍保留：{exc}", ERR_RED)
            return
        self._invalidate_script_tasks()
        if self._playing:
            self._stop_playback()
        self._cursor_timer.stop()
        self._playing = []
        self._bounds = []
        self._continue_all = False
        self._paused_at = self._awaiting_start = self._play_started = self._play_total = 0.0
        self._segment_tag = ""
        self.pause_button.setText("⏸ 暂停")
        self._timeline_position = self._timeline_segment_offset = 0.0
        self.document = ScriptDocument()
        self.source_edit.clear()
        self.language_combo.setCurrentIndex(0)
        self._chinese_backup = ""
        self.undo_translate_button.setEnabled(False)
        self.refresh_render_state()
        message = "已清空讲稿，可以重新粘贴或导入。"
        if backup is not None:
            message += f" 原稿备份：{backup}"
        self._set_status(message, OK_GREEN)

    def _restore_saved_script(self) -> None:
        """上次的讲稿自动恢复并切分；渲染过的句子直接显示「就绪」。"""
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        text = str(data.get("text", "") or "")
        if not text.strip():
            return
        self.source_edit.setPlainText(text)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(str(data.get("language", "") or ""))))
        try:
            self._split_clicked()
        except Exception:
            pass

    # ----------------------------------------------------------- 中文稿翻英文
    def _default_translate_runner(self):
        try:
            translator = ContextTranslator(
                api_key=self.settings.get_api_key(),
                workspace_id=self.settings.workspace_id,
                model=str(self.settings.get("llm_model", "") or "qwen3.8-flash"),
                proxy=self.settings.ws_proxy,
                proxy_mode=self.settings.proxy_mode,
                timeout=60,
            )
        except ApiError:
            return None
        glossary = self.settings.glossary_terms()
        brief = str(self.settings.get("defense_brief", "") or "")[:600]

        style = str(self.settings.get("translation_style", "accurate") or "accurate")
        style_hint = STYLE_DESK_HINTS.get(style, "")

        def run(chinese: str, previous: str) -> str:
            context = ("硕士论文答辩开场讲稿，要口语化、适合朗读。" + style_hint
                       + (f"论文背景：{brief}。" if brief else "")
                       + (f"上一段的英文（保持衔接）：{previous}" if previous else ""))
            return calibrate(translator, "zh2en", chinese, "", context=context, glossary=glossary)

        return run

    def _translate_source(self) -> None:
        text = self.source_edit.toPlainText()
        if not text.strip():
            self._set_status("先在左边粘贴中文讲稿", WARN_ORANGE)
            return
        if not re.search(r"[\u4e00-\u9fff]", text):
            self._set_status("左边看起来已经是英文了，不需要翻译", WARN_ORANGE)
            return
        run = self.translate_runner or self._default_translate_runner()
        if run is None:
            self._set_status("未配置百炼 API，无法翻译", ERR_RED)
            return
        paragraphs = text.split("\n")
        self._translate_cancel.set()
        cancel = self._translate_cancel = threading.Event()
        self.translate_button.setEnabled(False)
        self._set_status(f"正在逐段翻译（共 {sum(1 for p in paragraphs if p.strip())} 段）…", TEXT_DIM)

        def worker() -> None:
            english: list[str] = []
            previous = ""
            try:
                for paragraph in paragraphs:
                    if cancel.is_set():
                        return
                    if not paragraph.strip():
                        english.append("")
                        continue
                    previous = run(paragraph.strip(), previous)
                    english.append(previous)
            except Exception as exc:  # noqa: BLE001
                self.translate_done.emit(cancel, None, " ".join(str(exc).split())[:120])
                return
            self.translate_done.emit(cancel, english, "")

        threading.Thread(target=worker, name="script-translate", daemon=True).start()

    def _on_translate_done(self, cancel, english, error: str) -> None:
        if cancel is not self._translate_cancel or cancel.is_set():
            return
        self.translate_button.setEnabled(True)
        if english is None:
            self._set_status(f"翻译失败：{error}", ERR_RED)
            return
        self._chinese_backup = self.source_edit.toPlainText()
        self.source_edit.setPlainText("\n".join(english))
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData("en")))
        self.undo_translate_button.setEnabled(True)
        self._save_script()
        self._set_status("已翻成英文，核对后点「切分」再「渲染全部」。不满意可以「撤销翻译」找回中文", OK_GREEN)

    def _undo_translate(self) -> None:
        if not self._chinese_backup:
            return
        self.source_edit.setPlainText(self._chinese_backup)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData("zh")))
        self._chinese_backup = ""
        self.undo_translate_button.setEnabled(False)
        self._set_status("已恢复中文讲稿", TEXT_DIM)

    # --------------------------------------------------------------- 渲染
    def _render_paragraph(self) -> None:
        paragraph = self._current_paragraph()
        if paragraph is None:
            self._set_status("请先在中栏选中一段", WARN_ORANGE)
            return
        self.refresh_render_state()
        todo = [line for line in paragraph.lines if not line.rendered]
        if not todo:
            self._set_status(f"第 {paragraph.index + 1} 段用「{self.voice_combo.currentText()}」已经渲染好了", OK_GREEN)
            return
        self._start_render(todo)

    def _render_all(self) -> None:
        if not self.document.paragraphs:
            self._set_status("请先切分讲稿", ERR_RED)
            return
        self.refresh_render_state()
        pending = self.document.pending()
        if not pending:
            self._set_status("所有句子都已用当前音色渲染好，现场可以离线朗读", OK_GREEN)
            return
        self._start_render(pending)

    def _render_reference_paragraph(self) -> None:
        if self._rendering:
            return
        paragraph = self._current_paragraph()
        if paragraph is None or not paragraph.lines:
            self._set_status("请先选中需要重做的段落", WARN_ORANGE)
            return
        self._stop_playback()
        self.settings.set("prerender_instruction", "none")
        self.instruction_mode_combo.setCurrentIndex(max(0, self.instruction_mode_combo.findData("none")))
        units = self._units_of(paragraph)
        jobs = []
        for unit in units:
            lines = list(unit)
            def render(cancel, chosen=lines):
                if self._group_mode():
                    seconds = self.engine.render_script_group([line.text for line in chosen], chosen[0].language, cancel=cancel)
                else:
                    seconds = [self.engine.render_script_line(line.text, line.language, cancel=cancel)[1] for line in chosen]
                # Only retire a custom B version once the A version is safely cached.
                for line in chosen:
                    self.engine.clear_script_override(line.text, line.language, keep_audio=True)
                return seconds
            jobs.append((lines, render))
        self._start_jobs(jobs, "按 A 原声参数重做本段，已有音频保留…")

    # --------------------------------------------------------------- 右键管理
    def _line_menu(self, pos) -> None:
        row = self.line_table.rowAt(pos.y())
        paragraph = self._current_paragraph()
        if paragraph is None or not 0 <= row < len(paragraph.lines):
            return
        self.line_table.selectRow(row)
        line = paragraph.lines[row]
        menu = QMenu(self.line_table)
        menu.addAction("▶ 播放这句", lambda: self._start_sequence([line], continue_all=False))
        menu.addSeparator()
        menu.addAction("↻ 这句单独重新生成（换一版）", lambda: self._regenerate_line(line))
        tone = menu.addMenu("🎭 这句换个语气单独生成")
        for label, instruction in INSTRUCTION_PRESETS:
            if not instruction:
                values = self.engine.script_settings(line.language)
                label = "自然答辩（和声音对比 B 一样）"
                instruction = natural_delivery_instruction(str(values.get("tts_model", "")), line.language)
            tone.addAction(label, lambda l=label, i=instruction: self._regenerate_line(line, i, l))
        if getattr(line, "custom", ""):
            menu.addAction("↩ 不要单独版本，恢复整段里的这句", lambda: self._drop_override(line))
        menu.addAction("🗑 删除这句的音频", lambda: self._delete_lines([line], whole_blocks=False))
        menu.addSeparator()
        self._add_paragraph_actions(menu, paragraph)
        self._exec_menu(menu, self.line_table.viewport().mapToGlobal(pos))

    def _paragraph_menu(self, pos) -> None:
        row = self.paragraph_table.rowAt(pos.y())
        paragraph = self.document.paragraph_at(row)
        if paragraph is None:
            return
        self.paragraph_table.selectRow(row)
        menu = QMenu(self.paragraph_table)
        menu.addAction("▶ 播放本段", lambda: self._start_sequence(paragraph.lines, continue_all=False))
        menu.addSeparator()
        self._add_paragraph_actions(menu, paragraph)
        self._exec_menu(menu, self.paragraph_table.viewport().mapToGlobal(pos))

    def _exec_menu(self, menu, point) -> None:
        menu.exec(point)   # 单独成方法：测试里替换掉，离屏环境不弹真菜单

    def _add_paragraph_actions(self, menu, paragraph) -> None:
        menu.addAction(f"↻ 第 {paragraph.index + 1} 段整段重新生成", lambda: self._regenerate_paragraph(paragraph))
        menu.addAction(f"🗑 清空第 {paragraph.index + 1} 段的音频", lambda: self._clear_paragraph(paragraph))
        menu.addAction("🗑 一键清空全部讲稿音频…", self._clear_all_audio)

    def _regenerate_line(self, line, instruction: str | None = None, style: str = "") -> None:
        if self._rendering:
            self._set_status("正在渲染，等这一轮结束再操作", WARN_ORANGE)
            return
        label = style or "单独重做"

        def job(cancel):
            return [self.engine.render_script_override(
                line.text, line.language, instruction=instruction, style=label, cancel=cancel)]

        self._start_jobs([((line,), job)], f"正在单独生成第 {line.paragraph + 1} 段第 {line.order + 1} 句（{label}）…")

    def _drop_override(self, line) -> None:
        self.engine.clear_script_override(line.text, line.language)
        self.refresh_render_state()
        self._set_status("已恢复成整段里的这句", OK_GREEN)

    def _regenerate_paragraph(self, paragraph) -> None:
        if self._rendering:
            self._set_status("正在渲染，等这一轮结束再操作", WARN_ORANGE)
            return
        self._delete_lines(paragraph.lines, whole_blocks=True, quiet=True)
        self._start_render(list(paragraph.lines))

    def _delete_lines(self, lines, *, whole_blocks: bool, quiet: bool = False) -> None:
        """删音频：单句 → 它的单独版本和逐句音频；整段合成的那块只有删整段 / 清空时才删。"""
        if self._rendering:
            self._set_status("正在渲染，等这一轮结束再操作", WARN_ORANGE)
            return
        lines = list(lines)
        blocks_left = False
        for line in lines:
            try:
                self.engine.delete_script_line_audio(line.text, line.language)
            except Exception:  # noqa: BLE001
                pass
        if self._group_mode():
            wanted = {line.index for line in lines}
            for paragraph in self.document.paragraphs:
                for unit in self._units_of(paragraph):
                    if not any(line.index in wanted for line in unit):
                        continue
                    if whole_blocks or len(unit) == 1:
                        self.engine.delete_script_group_audio([l.text for l in unit], unit[0].language)
                    else:
                        blocks_left = True
        self.refresh_render_state()
        if quiet:
            return
        if blocks_left:
            self._set_status("已删除这句的单独版本 / 逐句音频。这句在整段合成的音频里还在（删一句会让整段断开）；"
                             "不想要这句的读法就右键「这句单独重新生成」或「换个语气」替换它", WARN_ORANGE)
        else:
            self._set_status(f"已删除 {len(lines)} 句的音频", OK_GREEN)

    def _clear_paragraph(self, paragraph) -> None:
        answer = QMessageBox.question(self, "清空本段",
                                      f"删除第 {paragraph.index + 1} 段（{len(paragraph.lines)} 句）已渲染的音频？删了要重新渲染（会计费）。")
        if answer != QMessageBox.Yes:
            return
        self._delete_lines(paragraph.lines, whole_blocks=True, quiet=True)
        self._set_status(f"已清空第 {paragraph.index + 1} 段的音频", OK_GREEN)

    def _clear_all_audio(self) -> None:
        if not self.document.paragraphs:
            return
        answer = QMessageBox.question(self, "一键清空",
                                      f"删除整篇讲稿（{len(self.document)} 句）用当前音色渲染的全部音频？\n"
                                      "别的音色渲染的不受影响。删了要重新渲染（会计费）。")
        if answer != QMessageBox.Yes:
            return
        self._delete_lines(self.document.lines, whole_blocks=True, quiet=True)
        self._set_status("已清空整篇讲稿的音频", OK_GREEN)

    def _start_jobs(self, jobs, status: str) -> None:
        """后台跑若干个合成任务：[(句子们, fn(cancel) -> 每句秒数)]，进度照常回到界面。"""
        if self._rendering:
            return
        self._rendering = True
        self._cancel = threading.Event()
        self.render_all_button.setEnabled(False)
        self.render_one_button.setEnabled(False)
        self.cancel_render_button.setVisible(True)
        self._set_status(status, TEXT_DIM)

        def worker(cancel=self._cancel) -> None:
            succeeded = attempted = 0
            for lines, fn in jobs:
                if cancel.is_set():
                    break
                attempted += len(lines)
                try:
                    seconds = list(fn(cancel))
                except Exception as exc:  # noqa: BLE001
                    message = " ".join(str(exc).split())[:2048]
                    for line in lines:
                        self.render_progress.emit({"index": line.index, "error": message, "keep": True, "cancel": cancel})
                    continue
                succeeded += len(lines)
                for line, value in zip(lines, seconds + [0.0] * len(lines)):
                    self.render_progress.emit({"index": line.index, "seconds": value, "error": "", "refresh": True, "cancel": cancel})
            self.render_done.emit(cancel, succeeded, attempted)

        threading.Thread(target=worker, name="defense-script-custom", daemon=True).start()

    def _render_units(self, lines) -> list:
        """要渲染的句子 → 要合成的块（整段模式下一句没渲染就整块重做）。"""
        wanted = {line.index for line in lines}
        units, seen = [], set()
        for paragraph in self.document.paragraphs:
            if not any(line.index in wanted for line in paragraph.lines):
                continue
            for unit in self._units_of(paragraph):
                ids = tuple(line.index for line in unit)
                if ids not in seen and any(i in wanted for i in ids):
                    seen.add(ids)
                    units.append(unit)
        return units

    def _start_render(self, lines) -> None:
        if self._rendering:
            return
        self._rendering = True
        self._cancel = threading.Event()
        self.render_all_button.setEnabled(False)
        self.render_one_button.setEnabled(False)
        self.cancel_render_button.setVisible(True)
        group_mode = self._group_mode()
        units = self._render_units(lines) if group_mode else [[line] for line in lines]
        count = sum(len(unit) for unit in units)
        how = f"（整段合成，{len(units)} 次请求）" if group_mode else ""
        self._set_status(f"正在渲染 {count} 句{how}…", TEXT_DIM)
        threading.Thread(
            target=self._render_worker,
            args=(tuple(tuple(unit) for unit in units), self._cancel, group_mode),
            name="defense-script-render",
            daemon=True,
        ).start()

    def _render_worker(self, units, cancel, group_mode: bool = False) -> None:
        """逐块串行渲染。一块失败继续下一块，不要让整篇卡在某一处。"""
        succeeded = 0
        attempted = 0
        if units and not isinstance(units[0], (tuple, list)):
            units = tuple((line,) for line in units)   # 兼容旧调用：直接传句子
        for unit in units:
            if cancel.is_set():
                break
            if group_mode:
                attempted += len(unit)
                try:
                    seconds_list = self.engine.render_script_group(
                        [line.text for line in unit], unit[0].language, cancel=cancel)
                except Exception as exc:
                    message = " ".join(str(exc).split())[:160]
                    for line in unit:
                        self.render_progress.emit({"index": line.index, "error": message, "cancel": cancel})
                    continue
                succeeded += len(unit)
                for line, seconds in zip(unit, list(seconds_list) + [0.0] * len(unit)):
                    self.render_progress.emit({"index": line.index, "seconds": seconds, "error": "", "cancel": cancel})
                continue
            line = unit[0]
            attempted += 1
            try:
                _pcm, seconds = self.engine.render_script_line(
                    line.text, line.language, cancel=cancel
                )
            except Exception as exc:
                self.render_progress.emit(
                    {"index": line.index, "error": " ".join(str(exc).split())[:2048], "cancel": cancel}
                )
                continue
            succeeded += 1
            self.render_progress.emit({"index": line.index, "seconds": seconds, "error": "", "cancel": cancel})
        self.render_done.emit(cancel, succeeded, attempted)

    @Slot(object)
    def _on_render_progress(self, payload: dict) -> None:
        cancel = payload.get("cancel")
        if cancel is not None and cancel.is_set():
            return
        line = self.document.line_at(int(payload.get("index", -1)))
        if line is None:
            return
        line.error = str(payload.get("error", ""))
        if line.error and line.rendered and (payload.get("keep") or getattr(line, "stale", False)):
            # 重渲染失败：旧参数的音频还在，照样能播，别把它标成未渲染。
            self._refresh_paragraphs()
            return
        line.rendered = not line.error
        line.stale = False
        line.seconds = float(payload.get("seconds", 0.0) or 0.0)
        self._refresh_paragraphs()

    @Slot(object, int, int)
    def _on_render_done(self, cancel, succeeded: int, attempted: int) -> None:
        if cancel is not self._cancel:
            return
        self._rendering = False
        self.render_all_button.setEnabled(True)
        self.render_one_button.setEnabled(True)
        self.cancel_render_button.setVisible(False)
        self.refresh_render_state()   # 以本机实际文件为准（单独版本、整段切分的时长都在这里重算）
        failed = attempted - succeeded
        if succeeded and not failed:
            self._set_status(f"✅ {succeeded} 句已渲染到本地，现场断网也能朗读", OK_GREEN)
        elif succeeded:
            self._set_status(
                f"⚠️ 成功 {succeeded} 句，失败 {failed} 句。失败原因在右栏状态列，可再点一次渲染重试。",
                WARN_ORANGE,
            )
        elif attempted:
            self._set_status("❌ 没有渲染成功任何句子，请看右栏的失败原因", ERR_RED)

    def _set_status(self, message: str, color: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

    def shutdown(self) -> None:
        """主窗口关闭时调用：停掉渲染线程和光标定时器。"""
        self._cancel.set()
        self._translate_cancel.set()
        self._cursor_timer.stop()


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
        if not _has_tts_source(self.settings):
            QMessageBox.warning(self, "讲稿预习", "请先在设置中选择克隆音色或 Next 参考录音")
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
        try:
            devices = [resolve_monitor_device(self.settings)]
            if self.to_meeting_check.isChecked():
                devices.insert(0, resolve_program_output(self.settings))
        except Exception as exc:
            self._finish_play(f"朗读失败：{exc}")
            return
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
    app.setApplicationName("留子答辩器")
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
