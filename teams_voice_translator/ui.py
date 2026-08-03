from __future__ import annotations

import logging
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .aliyun import ApiError, BailianClient, QwenRealtimeASR
from .audio import (
    DirectAudioBridge,
    DualTrackRecorder,
    MicrophoneCapture,
    MultiOutputPlayer,
    Pcm16Resampler,
    SystemAudioCapture,
    list_audio_devices,
    list_loopback_devices,
)
from .config import SettingsStore
from .hotkeys import HoldHotkeys
from .live_translate import QwenLiveTranslate
from .oss_upload import OssTemporaryUploader, UploadedVoiceSample
from .records import SubtitleSession, default_output_directory
from .profiles import CourseProfileStore
from .api_payloads import parse_json_list
from .voice_sample import SUPPORTED_AUDIO_SUFFIXES, normalized_voice_sample
from . import __version__


log = logging.getLogger(__name__)


class UiSignals(QObject):
    direct_down = Signal()
    direct_up = Signal()
    translate_down = Signal()
    translate_up = Signal()
    cancel = Signal()
    asr_preview = Signal(str, str)
    live_translation_preview = Signal(str)
    status = Signal(str)
    error = Signal(str)
    translation_ready = Signal(str, str, float, str)
    direct_caption_finished = Signal(str)
    teacher_preview = Signal(str)
    teacher_finished = Signal(str)
    tts_finished = Signal()
    test_finished = Signal(bool, str)
    clone_finished = Signal(bool, str, str)
    summary_ready = Signal(bool, str)


class HoldButton(QPushButton):
    hold_pressed = Signal()
    hold_released = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.hold_pressed.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.hold_released.emit()
        super().mouseReleaseEvent(event)


class SendTextEdit(QPlainTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            modifiers = event.modifiers()
            if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
                self.insertPlainText("\n")
                event.accept()
                return
            if modifiers in (Qt.NoModifier, Qt.KeypadModifier):
                self.send_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class SubtitleOverlay(QWidget):
    geometry_saved = Signal(int, int, int, int)

    def __init__(self) -> None:
        super().__init__(None)
        self._drag_offset = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("双语悬浮字幕")
        self._always_on_top = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.panel = QFrame()
        self.panel.setObjectName("subtitlePanel")
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(22, 13, 22, 13)
        panel_layout.setSpacing(5)
        self.chinese_label = QLabel("等待语音或文字输入…")
        self.chinese_label.setObjectName("overlayChinese")
        self.chinese_label.setAlignment(Qt.AlignCenter)
        self.chinese_label.setWordWrap(True)
        self.english_label = QLabel("")
        self.english_label.setObjectName("overlayEnglish")
        self.english_label.setAlignment(Qt.AlignCenter)
        self.english_label.setWordWrap(True)
        panel_layout.addWidget(self.chinese_label)
        panel_layout.addWidget(self.english_label)
        root.addWidget(self.panel)
        self.configure(82, True, 28, 22)

    def configure(
        self,
        opacity: int,
        always_on_top: bool,
        chinese_font_size: int,
        english_font_size: int,
    ) -> None:
        was_visible = self.isVisible()
        self._always_on_top = always_on_top
        flags = Qt.Tool | Qt.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setWindowOpacity(max(20, min(100, opacity)) / 100.0)
        self.panel.setStyleSheet(
            "QFrame#subtitlePanel { background: rgba(13, 18, 28, 225); "
            "border: 1px solid rgba(255,255,255,80); border-radius: 14px; }"
            f"QLabel#overlayChinese {{ color: white; font-family: 'Microsoft YaHei UI'; "
            f"font-size: {chinese_font_size}px; font-weight: 700; background: transparent; }}"
            f"QLabel#overlayEnglish {{ color: #dce9ff; font-family: 'Segoe UI'; "
            f"font-size: {english_font_size}px; font-weight: 600; background: transparent; }}"
        )
        if was_visible:
            self.show()

    def set_texts(self, chinese: str, english: str) -> None:
        self.chinese_label.setText(chinese.strip() or "等待语音或文字输入…")
        self.english_label.setText(english.strip())
        self.english_label.setVisible(bool(english.strip()))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            geometry = self.geometry()
            self.geometry_saved.emit(geometry.x(), geometry.y(), geometry.width(), geometry.height())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SubtitleExportDialog(QDialog):
    def __init__(
        self,
        output_directory: Path,
        language: str,
        file_format: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("保存本次字幕")
        self.resize(590, 250)
        layout = QVBoxLayout(self)
        note = QLabel(
            "运行期间的临时缓存始终保留完整中文、英文和时间轴。这里的选择只影响导出的文件。"
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        form = QFormLayout()
        self.language = QComboBox()
        self.language.addItem("中英双语", "both")
        self.language.addItem("仅中文", "zh")
        self.language.addItem("仅英文", "en")
        self.language.setCurrentIndex(max(0, self.language.findData(language)))
        self.file_format = QComboBox()
        self.file_format.addItem("SRT + TXT（推荐）", "both")
        self.file_format.addItem("仅 SRT（电影字幕）", "srt")
        self.file_format.addItem("仅 TXT（阅读记录）", "txt")
        self.file_format.addItem("仅 WebVTT（网页字幕）", "vtt")
        self.file_format.addItem("仅 Markdown（课堂笔记）", "md")
        self.file_format.addItem("仅 JSONL（结构化数据）", "jsonl")
        self.file_format.addItem("全部格式", "all")
        self.file_format.setCurrentIndex(max(0, self.file_format.findData(file_format)))
        self.directory = QLineEdit(str(output_directory))
        browse = QPushButton("选择…")
        browse.clicked.connect(self.choose_directory)
        directory_row = QHBoxLayout()
        directory_row.addWidget(self.directory)
        directory_row.addWidget(browse)
        form.addRow("导出语言", self.language)
        form.addRow("文件格式", self.file_format)
        form.addRow("保存目录", directory_row)
        self.include_source = QCheckBox("在字幕中标注“我 / 老师”等说话来源")
        self.include_source.setChecked(True)
        self.include_time = QCheckBox("在笔记和结构化文件中保留绝对时间")
        self.include_time.setChecked(True)
        form.addRow("说话人", self.include_source)
        form.addRow("绝对时间", self.include_time)
        layout.addLayout(form)
        format_hint = QLabel("SRT 可导入播放器、剪映和 Premiere；TXT 适合阅读、检索和复制。")
        format_hint.setObjectName("hint")
        layout.addWidget(format_hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.Save).setText("保存字幕")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.validate_and_accept)
        layout.addWidget(buttons)

    def choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择字幕保存目录", self.directory.text())
        if selected:
            self.directory.setText(selected)

    def validate_and_accept(self) -> None:
        if not self.directory.text().strip():
            QMessageBox.warning(self, "保存目录", "请选择字幕保存目录。")
            return
        self.accept()


class GlossaryDialog(QDialog):
    def __init__(self, raw_json: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("课程术语表")
        self.resize(680, 430)
        root = QVBoxLayout(self)
        note = QLabel("中文术语和标准英文译法会同时用于你说中文、老师说英文两个方向。")
        note.setObjectName("hint")
        root.addWidget(note)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["中文术语", "英文标准译法"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.table)
        try:
            values = json.loads(raw_json) if raw_json.strip() else []
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    self.add_row(str(item.get("source", "")), str(item.get("target", "")))
        actions = QHBoxLayout()
        add = QPushButton("新增术语")
        remove = QPushButton("删除选中")
        add.clicked.connect(lambda: self.add_row("", ""))
        remove.clicked.connect(self.remove_selected)
        actions.addWidget(add)
        actions.addWidget(remove)
        actions.addStretch()
        root.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.Save).setText("保存术语表")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def add_row(self, source: str, target: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(source))
        self.table.setItem(row, 1, QTableWidgetItem(target))

    def remove_selected(self) -> None:
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def serialized(self) -> str:
        result = []
        for row in range(self.table.rowCount()):
            source_item = self.table.item(row, 0)
            target_item = self.table.item(row, 1)
            source = source_item.text().strip() if source_item else ""
            target = target_item.text().strip() if target_item else ""
            if source and target:
                result.append({"source": source, "target": target})
        return json.dumps(result, ensure_ascii=False, indent=2) if result else ""


class MeetingSummaryDialog(QDialog):
    def __init__(self, summary: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI 课堂总结 · 可编辑")
        self.resize(780, 620)
        root = QVBoxLayout(self)
        note = QLabel("总结由完整的双向字幕生成。请检查专有名词和作业要求后再保存。")
        note.setObjectName("hint")
        root.addWidget(note)
        self.editor = QPlainTextEdit(summary)
        root.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.Save).setText("保存 Markdown 总结")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


class OssSettingsDialog(QDialog):
    def __init__(self, settings: SettingsStore, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("本地样音临时上传 · 阿里云 OSS")
        self.resize(650, 360)
        root = QVBoxLayout(self)
        note = QLabel(
            "仅首次配置。请选择一个私有 OSS Bucket；程序把样音转换为 WAV 后临时上传，"
            "生成 15 分钟签名地址，并在创建音色后立即删除。AccessKey Secret 只保存在 Windows 凭据管理器。"
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        form = QFormLayout()
        self.region = QLineEdit(str(settings.get("oss_region", "cn-beijing")))
        self.region.setPlaceholderText("例如 cn-beijing、cn-hangzhou")
        self.bucket = QLineEdit(str(settings.get("oss_bucket", "")))
        self.bucket.setPlaceholderText("已创建的私有 Bucket 名称")
        self.access_key_id = QLineEdit(settings.get_oss_access_key_id())
        self.access_key_id.setPlaceholderText("建议使用仅限临时目录的 RAM AccessKey")
        self.access_key_secret = QLineEdit()
        self.access_key_secret.setEchoMode(QLineEdit.Password)
        if settings.get_oss_access_key_secret():
            self.access_key_secret.setPlaceholderText("已保存；留空表示继续使用")
        form.addRow("OSS 地域 Region", self.region)
        form.addRow("Bucket", self.bucket)
        form.addRow("AccessKey ID", self.access_key_id)
        form.addRow("AccessKey Secret", self.access_key_secret)
        root.addLayout(form)

        permission = QLabel(
            "RAM 最小权限：仅允许该 Bucket 下 teams-voice-translator/temporary/* 的 "
            "oss:PutObject、oss:GetObject、oss:DeleteObject。请勿填写阿里云主账号 AccessKey。"
        )
        permission.setWordWrap(True)
        permission.setObjectName("hint")
        root.addWidget(permission)

        links = QHBoxLayout()
        open_oss = QPushButton("打开阿里云 OSS 控制台")
        open_oss.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://oss.console.aliyun.com/bucket"))
        )
        open_ram = QPushButton("打开 RAM 用户管理")
        open_ram.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://ram.console.aliyun.com/users"))
        )
        links.addWidget(open_oss)
        links.addWidget(open_ram)
        links.addStretch()
        root.addLayout(links)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.Save).setText("保存 OSS 设置")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.submit)
        root.addWidget(buttons)

    def submit(self) -> None:
        region = self.region.text().strip().lower()
        bucket = self.bucket.text().strip()
        access_key_id = self.access_key_id.text().strip()
        secret = self.access_key_secret.text().strip()
        if not region or not all(char.isalnum() or char == "-" for char in region):
            QMessageBox.warning(self, "OSS 地域无效", "请填写类似 cn-beijing 的 Region。")
            return
        if not bucket or not all(char.isalnum() or char in "-." for char in bucket):
            QMessageBox.warning(self, "Bucket 无效", "请填写已创建的 OSS Bucket 名称。")
            return
        if not access_key_id:
            QMessageBox.warning(self, "AccessKey 缺失", "请填写 RAM AccessKey ID。")
            return
        if not secret and not self.settings.get_oss_access_key_secret():
            QMessageBox.warning(self, "AccessKey 缺失", "请填写 RAM AccessKey Secret。")
            return
        self.settings.update(
            {
                "oss_region": region,
                "oss_bucket": bucket,
            }
        )
        self.settings.set_oss_access_key_id(access_key_id)
        if secret:
            self.settings.set_oss_access_key_secret(secret)
        self.accept()


class VoiceCloneDialog(QDialog):
    create_requested = Signal(dict)

    def __init__(
        self,
        model: str,
        settings: SettingsStore,
        configure_oss: Callable[[], bool],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.configure_oss = configure_oss
        self.setWindowTitle("创建克隆音色 · 百炼官方接口")
        self.resize(720, 430)
        layout = QVBoxLayout(self)
        note = QLabel(
            "直接选择 iPhone M4A、MP3、WAV 等本地录音即可。程序会自动裁剪并转换成 "
            "24 kHz 单声道 16-bit PCM WAV，临时上传到你自己的 OSS，创建音色后立即删除。"
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        form = QFormLayout()
        self.model = QComboBox()
        self.model.addItems([
            "qwen-audio-3.0-tts-flash",
            "qwen-audio-3.0-tts-plus",
            "qwen3.5-livetranslate-flash-realtime",
        ])
        self.model.setCurrentText(model)
        self.prefix = QLineEdit("myvoice")
        self.url = QLineEdit()
        self.url.setPlaceholderText("选择本地录音；也兼容原有 HTTPS 样音地址")
        source_row = QHBoxLayout()
        source_row.addWidget(self.url)
        choose_file = QPushButton("选择本地录音")
        choose_file.clicked.connect(self.choose_local_file)
        source_row.addWidget(choose_file)
        oss_settings = QPushButton("OSS 设置")
        oss_settings.clicked.connect(self.open_oss_settings)
        source_row.addWidget(oss_settings)
        self.language = QComboBox()
        self.language.addItem("中文 zh", "zh")
        self.language.addItem("英语 en", "en")
        self.max_seconds = QDoubleSpinBox()
        self.max_seconds.setRange(5.0, 30.0)
        self.max_seconds.setValue(20.0)
        self.max_seconds.setSuffix(" 秒")
        self.preprocess = QCheckBox("开启降噪、增强和音量归一化")
        self.model.currentTextChanged.connect(self.refresh_model_options)
        form.addRow("目标模型", self.model)
        form.addRow("音色前缀", self.prefix)
        form.addRow("样音文件", source_row)
        form.addRow("样音语言", self.language)
        form.addRow("最大取样长度", self.max_seconds)
        form.addRow("样音预处理", self.preprocess)
        layout.addLayout(form)
        self.refresh_model_options()
        self.oss_status = QLabel()
        self.oss_status.setObjectName("hint")
        self.refresh_oss_status()
        layout.addWidget(self.oss_status)
        links = QHBoxLayout()
        docs = QPushButton("打开声音复刻官方文档")
        docs.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide")
            )
        )
        links.addWidget(docs)
        links.addStretch()
        layout.addLayout(links)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("创建音色")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.submit)
        layout.addWidget(buttons)

    def choose_local_file(self) -> None:
        formats = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_AUDIO_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择用于复刻声音的本地录音",
            "",
            f"音频文件 ({formats});;所有文件 (*.*)",
        )
        if path:
            self.url.setText(path)

    def refresh_model_options(self) -> None:
        supported = self.model.currentText() in {
            "qwen-audio-3.0-tts-flash",
            "qwen-audio-3.0-tts-plus",
        }
        self.preprocess.setEnabled(supported)
        if not supported:
            self.preprocess.setChecked(False)
            self.preprocess.setToolTip("LiveTranslate 不支持服务端样音预处理；本地格式标准化仍会自动执行。")
        else:
            self.preprocess.setToolTip("")

    def open_oss_settings(self) -> None:
        self.configure_oss()
        self.refresh_oss_status()

    def refresh_oss_status(self) -> None:
        config = self.settings.get_oss_config()
        if self.settings.has_oss_config():
            self.oss_status.setText(
                f"本地样音自动上传已就绪：{config['bucket']} · {config['region']}；上传后自动删除。"
            )
        else:
            self.oss_status.setText("首次选择本地样音前，请点击“OSS 设置”完成一次性配置。")

    def submit(self) -> None:
        source = self.url.text().strip()
        local_path = Path(source).expanduser()
        if local_path.is_file():
            if not self.settings.has_oss_config() and not self.configure_oss():
                return
            self.refresh_oss_status()
            audio_path = str(local_path.resolve())
            audio_url = ""
        elif source.startswith("https://"):
            audio_path = ""
            audio_url = source
        else:
            QMessageBox.warning(
                self,
                "样音文件无效",
                "请选择本地录音文件，或填写可公开访问的 HTTPS 音频地址。",
            )
            return
        self.create_requested.emit(
            {
                "target_model": self.model.currentText(),
                "prefix": self.prefix.text().strip(),
                "audio_path": audio_path,
                "audio_url": audio_url,
                "language": self.language.currentData(),
                "max_seconds": self.max_seconds.value(),
                "preprocess": self.preprocess.isChecked(),
            }
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = SettingsStore()
        self.signals = UiSignals()
        self.direct_bridge = DirectAudioBridge()
        self.direct_caption_queue: queue.Queue[bytes | None] | None = None
        self.direct_caption_active = False
        self.teacher_capture = SystemAudioCapture()
        self.teacher_audio_queue: queue.Queue[bytes | None] | None = None
        self.teacher_segment_queue: queue.Queue[str | None] | None = None
        self.teacher_thread: threading.Thread | None = None
        self.teacher_translation_thread: threading.Thread | None = None
        self.teacher_active = False
        self.teacher_cancel_event = threading.Event()
        self.tts_active = threading.Event()
        self.translation_capture: MicrophoneCapture | None = None
        self.translation_queue: queue.Queue[bytes | None] | None = None
        self.pipeline_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state = "idle"
        self.last_translation = ""
        self.awaiting_confirmation = False
        self.clone_dialog: VoiceCloneDialog | None = None
        self.clone_target_model = ""
        self.hotkeys: HoldHotkeys | None = None
        self.recorder = DualTrackRecorder()
        self.recording_started_at: float | None = None
        self.subtitle_session = SubtitleSession()
        self.subtitle_exported = True
        self.profile_store = CourseProfileStore(self.settings.base_dir)
        self._build_ui()
        self.overlay = SubtitleOverlay()
        self._connect_signals()
        self.overlay.geometry_saved.connect(self._save_overlay_geometry)
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(500)
        self.record_timer.timeout.connect(self._update_recording_status)
        self.refresh_audio_devices()
        self.load_settings_into_ui()
        self.start_hotkeys()
        self.setWindowTitle("Teams 双向课堂翻译")
        self.resize(1280, 790)
        self._set_status("就绪 · F8 原声 / F9 翻译")
        QTimer.singleShot(300, self.offer_cache_recovery)
        if self.settings.get("auto_start_teacher_caption"):
            QTimer.singleShot(700, self.start_teacher_caption)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Teams 双向课堂翻译")
        title.setObjectName("title")
        subtitle = QLabel("你说中文 → 英文克隆音色 · 老师说英文 → 中文字幕 · 双向录音与课堂总结")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.status_badge = QLabel("初始化…")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.meeting_tab = self._build_meeting_tab()
        self.settings_tab = self._build_settings_tab()
        self.tabs.addTab(self.meeting_tab, "会议控制台")
        self.tabs.addTab(self.settings_tab, "设置")
        root.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.setStyleSheet(STYLE_SHEET)

    def _build_meeting_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        transcript_grid = QGridLayout()
        self.zh_group = QGroupBox("中文 · 你的原文 / 老师译文")
        zh_layout = QVBoxLayout(self.zh_group)
        self.chinese_text = QPlainTextEdit()
        self.chinese_text.setPlaceholderText("按住 F9 说中文，识别结果会实时显示在这里…")
        self.chinese_text.setReadOnly(True)
        zh_layout.addWidget(self.chinese_text)
        self.emotion_label = QLabel("识别情绪：—")
        self.emotion_label.setObjectName("hint")
        zh_layout.addWidget(self.emotion_label)
        self.en_group = QGroupBox("English · 你的译文 / 老师原文")
        en_layout = QVBoxLayout(self.en_group)
        self.english_text = QPlainTextEdit()
        self.english_text.setPlaceholderText("松开 F9 后，英文译文会显示在这里…")
        en_layout.addWidget(self.english_text)
        edit_hint = QLabel("可在“先确认后播放”模式下修改英文，再点击播放。")
        edit_hint.setObjectName("hint")
        en_layout.addWidget(edit_hint)
        transcript_grid.addWidget(self.zh_group, 0, 0)
        transcript_grid.addWidget(self.en_group, 0, 1)
        layout.addLayout(transcript_grid)

        typed_group = QGroupBox("键盘输入 · 不方便开口时使用")
        typed_layout = QHBoxLayout(typed_group)
        self.typed_input = SendTextEdit()
        self.typed_input.setMaximumHeight(76)
        self.typed_input.setPlaceholderText("输入中文后按 Enter 发送；Ctrl+Enter 换行…")
        typed_layout.addWidget(self.typed_input, 4)
        typed_actions = QVBoxLayout()
        self.typed_mode = QComboBox()
        self.typed_mode.addItem("中文翻译成英文后发送", "translate")
        self.typed_mode.addItem("按输入原文直接朗读", "direct")
        self.typed_send_button = QPushButton("发送文字语音  Enter")
        self.typed_send_button.setObjectName("primaryButton")
        typed_actions.addWidget(self.typed_mode)
        typed_actions.addWidget(self.typed_send_button)
        typed_layout.addLayout(typed_actions, 2)
        layout.addWidget(typed_group)

        controls = QHBoxLayout()
        self.direct_button = HoldButton("按住直接说话\nF8")
        self.direct_button.setObjectName("directButton")
        self.translate_button = HoldButton("按住翻译说话\nF9")
        self.translate_button.setObjectName("translateButton")
        controls.addWidget(self.direct_button, 2)
        controls.addWidget(self.translate_button, 2)
        side = QVBoxLayout()
        self.play_button = QPushButton("播放/发送当前英文")
        self.teacher_button = QPushButton("▶ 开始听老师 / Teams")
        self.teacher_button.setObjectName("teacherButton")
        self.stop_button = QPushButton("停止 / 取消  Esc")
        side.addWidget(self.play_button)
        side.addWidget(self.teacher_button)
        side.addWidget(self.stop_button)
        controls.addLayout(side, 1)
        layout.addLayout(controls)

        utility = QHBoxLayout()
        self.record_button = QPushButton("● 开始录音")
        self.record_button.setObjectName("recordButton")
        self.subtitle_display_mode_quick = QComboBox()
        self.subtitle_display_mode_quick.addItem("字幕：中英双语", "both")
        self.subtitle_display_mode_quick.addItem("字幕：仅中文", "zh")
        self.subtitle_display_mode_quick.addItem("字幕：仅英文", "en")
        self.profile_quick = QComboBox()
        self.profile_quick.addItem("课程：默认", "")
        for profile_name in self.profile_store.names():
            self.profile_quick.addItem(f"课程：{profile_name}", profile_name)
        self.overlay_toggle_button = QPushButton("显示悬浮字幕")
        self.open_output_button = QPushButton("打开保存目录")
        self.export_subtitles_button = QPushButton("保存本次字幕…")
        self.summary_button = QPushButton("生成课堂总结")
        self.clear_button = QPushButton("清空文本")
        utility.addWidget(self.record_button)
        utility.addWidget(self.subtitle_display_mode_quick)
        utility.addWidget(self.profile_quick)
        utility.addWidget(self.overlay_toggle_button)
        utility.addWidget(self.open_output_button)
        utility.addWidget(self.export_subtitles_button)
        utility.addWidget(self.summary_button)
        utility.addStretch()
        utility.addWidget(self.clear_button)
        layout.addLayout(utility)

        history_group = QGroupBox("本次会议双向时间轴（双击一行可重新载入）")
        history_layout = QVBoxLayout(history_group)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索本次字幕中的中文、英文或说话人…")
        history_layout.addWidget(self.history_search)
        self.history = QTableWidget(0, 5)
        self.history.setHorizontalHeaderLabels(["时间", "来源", "中文", "英文", "耗时"])
        self.history.horizontalHeader().setStretchLastSection(False)
        self.history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.history.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        history_layout.addWidget(self.history)
        layout.addWidget(history_group)
        return tab

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        grid = QGridLayout()

        api_group = QGroupBox("1. 百炼 API（华北 2 · 北京）")
        api_form = QFormLayout(api_group)
        self.workspace_id = QLineEdit()
        self.workspace_id.setPlaceholderText("百炼业务空间 Workspace ID")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("留空表示继续使用 Windows 凭据管理器中已保存的 Key")
        self.show_key = QCheckBox("显示")
        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key)
        key_row.addWidget(self.show_key)
        self.test_api_button = QPushButton("测试 API 连通性")
        api_form.addRow("Workspace ID", self.workspace_id)
        api_form.addRow("API Key", key_row)
        api_form.addRow("", self.test_api_button)
        api_note = QLabel("API Key 不会写入 settings.json，而是保存到 Windows 凭据管理器。")
        api_note.setObjectName("hint")
        api_note.setWordWrap(True)
        api_form.addRow("", api_note)

        audio_group = QGroupBox("2. 音频设备")
        audio_form = QFormLayout(audio_group)
        self.input_device = QComboBox()
        self.teams_output_device = QComboBox()
        self.loopback_device = QComboBox()
        self.monitor_enabled = QCheckBox("同时在本机扬声器试听翻译后的英文")
        self.monitor_output_device = QComboBox()
        self.direct_sample_rate = QComboBox()
        for rate in (44100, 48000):
            self.direct_sample_rate.addItem(f"{rate} Hz", rate)
        self.refresh_devices_button = QPushButton("刷新设备列表")
        self.audio_diagnostics_button = QPushButton("运行音频设备诊断")
        audio_form.addRow("物理麦克风", self.input_device)
        audio_form.addRow("Teams 虚拟输出", self.teams_output_device)
        audio_form.addRow("老师/Teams 系统声", self.loopback_device)
        audio_form.addRow("监听", self.monitor_enabled)
        audio_form.addRow("本机试听设备", self.monitor_output_device)
        audio_form.addRow("原声采样率", self.direct_sample_rate)
        audio_form.addRow("", self.refresh_devices_button)
        audio_form.addRow("", self.audio_diagnostics_button)
        cable_hint = QLabel("推荐安装 VB-CABLE，并在这里选 “CABLE Input”；Teams 麦克风选 “CABLE Output”。")
        cable_hint.setObjectName("hint")
        cable_hint.setWordWrap(True)
        audio_form.addRow("", cable_hint)

        grid.addWidget(api_group, 0, 0)
        grid.addWidget(audio_group, 0, 1)

        live_group = QGroupBox("3. F9 极速语音直译 · 中文语音直接生成英文字幕和语音")
        live_form = QFormLayout(live_group)
        self.translation_engine = QComboBox()
        self.translation_engine.addItem("极速直译（推荐，单模型低延迟）", "live")
        self.translation_engine.addItem("传统流水线（ASR → Qwen-MT → TTS）", "classic")
        self.live_translate_model = QComboBox()
        self.live_translate_model.setEditable(True)
        self.live_translate_model.addItems([
            "qwen3.5-livetranslate-flash-realtime",
            "qwen3.5-livetranslate-flash-realtime-2026-05-19",
        ])
        self.live_voice_clone_mode = QComboBox()
        self.live_voice_clone_mode.addItem("服务端复刻一次（推荐）", "once")
        self.live_voice_clone_mode.addItem("不复刻，使用默认音色（最快）", "default")
        self.live_voice_clone_mode.addItem("每轮动态复刻（多人场景）", "always")
        self.live_voice_clone_mode.addItem("使用预先复刻的固定音色", "fixed")
        self.live_voice = QLineEdit()
        self.live_voice.setPlaceholderText("固定 voice_id，例如 qwen-translate-vc-…；其他模式可留空")
        live_voice_row = QHBoxLayout()
        live_voice_row.addWidget(self.live_voice)
        self.clone_live_voice_button = QPushButton("创建直译专属音色")
        live_voice_row.addWidget(self.clone_live_voice_button)
        live_form.addRow("F9 翻译引擎", self.translation_engine)
        live_form.addRow("直译模型", self.live_translate_model)
        live_form.addRow("声音复刻", self.live_voice_clone_mode)
        live_form.addRow("直译 voice", live_voice_row)
        live_hint = QLabel(
            "极速模式通过一个 WebSocket 直接完成中文识别、英文翻译和英文语音流式输出。"
            "请在 API Key 权限中授权 qwen3.5-livetranslate-flash-realtime。"
            "固定音色需针对该模型单独创建，不能复用普通 TTS voice_id。"
        )
        live_hint.setObjectName("hint")
        live_hint.setWordWrap(True)
        live_form.addRow("", live_hint)
        grid.addWidget(live_group, 1, 0, 1, 2)

        asr_group = QGroupBox("4. 字幕、F8 与传统模式实时语音识别")
        asr_form = QFormLayout(asr_group)
        self.asr_model = QComboBox()
        self.asr_model.addItems([
            "qwen3-asr-flash-realtime",
            "qwen3-asr-flash-realtime-2026-02-10",
        ])
        self.asr_language = QComboBox()
        self.asr_language.addItem("中文（普通话/四川话/闽南语/吴语）", "zh")
        self.asr_language.addItem("粤语", "yue")
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setDecimals(2)
        self.vad_silence_ms = QSpinBox()
        self.vad_silence_ms.setRange(200, 2000)
        self.vad_silence_ms.setSingleStep(100)
        self.vad_silence_ms.setSuffix(" ms")
        self.direct_caption_enabled = QCheckBox("F8 原声直通时也识别中文并在松开后生成英文翻译")
        self.teacher_caption_enabled = QCheckBox("监听 Teams 系统声：老师英文原文 + 中文翻译")
        self.auto_start_teacher_caption = QCheckBox("软件启动后自动开始监听老师")
        asr_form.addRow("模型", self.asr_model)
        asr_form.addRow("说话语言", self.asr_language)
        asr_form.addRow("VAD 灵敏度阈值", self.vad_threshold)
        asr_form.addRow("静音断句时间", self.vad_silence_ms)
        asr_form.addRow("原声字幕", self.direct_caption_enabled)
        asr_form.addRow("老师字幕", self.teacher_caption_enabled)
        asr_form.addRow("自动监听", self.auto_start_teacher_caption)
        vad_hint = QLabel("官方低延迟建议：threshold=0.0、silence=400ms；默认给你 500ms，降低误断句。")
        vad_hint.setObjectName("hint")
        vad_hint.setWordWrap(True)
        asr_form.addRow("", vad_hint)

        mt_group = QGroupBox("5. 字幕、键盘输入与传统模式机器翻译")
        mt_form = QFormLayout(mt_group)
        self.translation_model = QComboBox()
        self.translation_model.addItems(["qwen-mt-flash", "qwen-mt-plus", "qwen-mt-turbo", "qwen-mt-lite"])
        self.summary_model = QComboBox()
        self.summary_model.setEditable(True)
        self.summary_model.addItems(["qwen-plus", "qwen-max", "qwen-flash"])
        self.translation_domain = QLineEdit()
        self.translation_terms = QPlainTextEdit()
        self.translation_terms.setMaximumHeight(65)
        self.translation_terms.setPlaceholderText('[{"source":"有限元","target":"finite element method"}]')
        self.translation_memories = QPlainTextEdit()
        self.translation_memories.setMaximumHeight(65)
        self.translation_memories.setPlaceholderText('[{"source":"老师您好","target":"Hello, Professor."}]')
        self.speak_mode = QComboBox()
        self.speak_mode.addItem("立即翻译并发送（最快）", "auto")
        self.speak_mode.addItem("先确认/编辑，再手动发送（稳妥）", "confirm")
        self.translation_style = QComboBox()
        self.translation_style.addItem("自然礼貌课堂英语", "polite")
        self.translation_style.addItem("简洁日常口语", "concise")
        self.translation_style.addItem("正式学术表达", "academic")
        self.translation_style.addItem("尽量逐字忠实", "literal")
        self.glossary_button = QPushButton("表格方式编辑课程术语")
        mt_form.addRow("模型", self.translation_model)
        mt_form.addRow("课堂总结模型", self.summary_model)
        mt_form.addRow("领域提示（英文）", self.translation_domain)
        mt_form.addRow("表达风格", self.translation_style)
        mt_form.addRow("术语表 JSON", self.translation_terms)
        mt_form.addRow("", self.glossary_button)
        mt_form.addRow("翻译记忆 JSON", self.translation_memories)
        mt_form.addRow("发送方式", self.speak_mode)

        grid.addWidget(asr_group, 2, 0)
        grid.addWidget(mt_group, 2, 1)

        tts_group = QGroupBox("6. 键盘输入与传统模式语音合成")
        tts_form = QFormLayout(tts_group)
        self.tts_model = QComboBox()
        self.tts_model.addItems(["qwen-audio-3.0-tts-flash", "qwen-audio-3.0-tts-plus"])
        self.voice = QLineEdit()
        self.voice.setPlaceholderText("克隆 voice_id，或系统音色如 loongjohn")
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice)
        self.clone_voice_button = QPushButton("创建克隆音色")
        voice_row.addWidget(self.clone_voice_button)
        self.tts_volume = QSpinBox()
        self.tts_volume.setRange(0, 100)
        self.tts_rate = QDoubleSpinBox()
        self.tts_rate.setRange(0.5, 2.0)
        self.tts_rate.setSingleStep(0.05)
        self.tts_pitch = QDoubleSpinBox()
        self.tts_pitch.setRange(0.5, 2.0)
        self.tts_pitch.setSingleStep(0.05)
        self.tts_seed = QSpinBox()
        self.tts_seed.setRange(0, 65535)
        self.tts_emotion = QComboBox()
        for label, value in [
            ("自然（无标签）", ""),
            ("悲伤 [sad]", "[sad]"),
            ("惊讶 [amazed]", "[amazed]"),
            ("低沉大声 [deep and loud shouting]", "[deep and loud shouting]"),
        ]:
            self.tts_emotion.addItem(label, value)
        self.tts_instruction = QLineEdit()
        self.enable_aigc_tag = QCheckBox("在生成音频中嵌入官方 AIGC 隐性标识")
        self.aigc_propagator = QLineEdit()
        self.aigc_propagate_id = QLineEdit()
        tts_form.addRow("模型", self.tts_model)
        tts_form.addRow("音色 voice", voice_row)
        tts_form.addRow("音量 0–100", self.tts_volume)
        tts_form.addRow("语速 0.5–2.0", self.tts_rate)
        tts_form.addRow("音调 0.5–2.0", self.tts_pitch)
        tts_form.addRow("随机种子", self.tts_seed)
        tts_form.addRow("情绪标签", self.tts_emotion)
        tts_form.addRow("Free-style 指令", self.tts_instruction)
        tts_form.addRow("AIGC 标识", self.enable_aigc_tag)
        tts_form.addRow("ContentPropagator", self.aigc_propagator)
        tts_form.addRow("PropagateID", self.aigc_propagate_id)

        hotkey_group = QGroupBox("7. 快捷键与网络")
        hotkey_form = QFormLayout(hotkey_group)
        self.direct_hotkey = QComboBox()
        self.translate_hotkey = QComboBox()
        self.cancel_hotkey = QComboBox()
        for box in (self.direct_hotkey, self.translate_hotkey):
            box.addItems([f"f{i}" for i in range(1, 13)])
        self.cancel_hotkey.addItems(["esc"] + [f"f{i}" for i in range(1, 13)])
        self.request_timeout = QSpinBox()
        self.request_timeout.setRange(10, 180)
        self.request_timeout.setSuffix(" 秒")
        self.http_proxy = QLineEdit()
        self.http_proxy.setPlaceholderText("通常留空；例如 http://127.0.0.1:7890")
        hotkey_form.addRow("按住原声", self.direct_hotkey)
        hotkey_form.addRow("按住翻译", self.translate_hotkey)
        hotkey_form.addRow("停止/取消", self.cancel_hotkey)
        hotkey_form.addRow("接口超时", self.request_timeout)
        hotkey_form.addRow("HTTP 代理", self.http_proxy)

        grid.addWidget(tts_group, 3, 0)
        grid.addWidget(hotkey_group, 3, 1)

        overlay_group = QGroupBox("7. 歌词式双语悬浮字幕")
        overlay_form = QFormLayout(overlay_group)
        self.overlay_enabled = QCheckBox("启用悬浮字幕（仅在本机显示）")
        self.overlay_always_on_top = QCheckBox("始终置顶")
        self.overlay_opacity = QSpinBox()
        self.overlay_opacity.setRange(20, 100)
        self.overlay_opacity.setSuffix(" %")
        self.overlay_chinese_font_size = QSpinBox()
        self.overlay_chinese_font_size.setRange(14, 64)
        self.overlay_chinese_font_size.setSuffix(" px")
        self.overlay_english_font_size = QSpinBox()
        self.overlay_english_font_size.setRange(12, 56)
        self.overlay_english_font_size.setSuffix(" px")
        self.overlay_width = QSpinBox()
        self.overlay_width.setRange(420, 1800)
        self.overlay_width.setSuffix(" px")
        self.overlay_height = QSpinBox()
        self.overlay_height.setRange(90, 500)
        self.overlay_height.setSuffix(" px")
        overlay_form.addRow("显示", self.overlay_enabled)
        overlay_form.addRow("窗口层级", self.overlay_always_on_top)
        overlay_form.addRow("透明度", self.overlay_opacity)
        overlay_form.addRow("中文字幕字号", self.overlay_chinese_font_size)
        overlay_form.addRow("英文字幕字号", self.overlay_english_font_size)
        overlay_form.addRow("悬浮窗宽度", self.overlay_width)
        overlay_form.addRow("悬浮窗高度", self.overlay_height)
        overlay_hint = QLabel("可直接拖动悬浮窗改变位置；修改透明度和字号时会立即预览。")
        overlay_hint.setObjectName("hint")
        overlay_hint.setWordWrap(True)
        overlay_form.addRow("", overlay_hint)

        records_group = QGroupBox("8. 录音与字幕保存")
        records_form = QFormLayout(records_group)
        self.output_directory = QLineEdit()
        self.output_directory.setPlaceholderText(str(default_output_directory()))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_directory)
        self.choose_output_button = QPushButton("选择…")
        output_row.addWidget(self.choose_output_button)
        records_form.addRow("保存目录", output_row)
        record_hint = QLabel(
            "录音保存为单声道 WAV。字幕运行时始终完整缓存中文、英文和时间轴；点击“保存本次字幕”后再选择语言和格式。"
        )
        record_hint.setObjectName("hint")
        record_hint.setWordWrap(True)
        records_form.addRow("", record_hint)

        grid.addWidget(overlay_group, 4, 0)
        grid.addWidget(records_group, 4, 1)

        appearance_group = QGroupBox("9. 外观与实时字幕显示")
        appearance_form = QFormLayout(appearance_group)
        self.theme = QComboBox()
        self.theme.addItem("浅色界面", "light")
        self.theme.addItem("深色黑色界面", "dark")
        self.subtitle_display_mode = QComboBox()
        self.subtitle_display_mode.addItem("中英双语", "both")
        self.subtitle_display_mode.addItem("仅中文", "zh")
        self.subtitle_display_mode.addItem("仅英文", "en")
        appearance_form.addRow("界面主题", self.theme)
        appearance_form.addRow("实时字幕", self.subtitle_display_mode)
        appearance_hint = QLabel("实时显示方式不会删减缓存；保存文件时仍可重新选择纯中文、纯英文或双语。")
        appearance_hint.setObjectName("hint")
        appearance_hint.setWordWrap(True)
        appearance_form.addRow("", appearance_hint)
        grid.addWidget(appearance_group, 5, 0, 1, 2)

        profile_group = QGroupBox("10. 课程配置")
        profile_form = QFormLayout(profile_group)
        self.profile_name = QComboBox()
        self.profile_name.setEditable(True)
        self.profile_name.addItem("")
        self.profile_name.addItems(self.profile_store.names())
        profile_actions = QHBoxLayout()
        self.profile_load_button = QPushButton("载入")
        self.profile_save_button = QPushButton("保存/覆盖")
        self.profile_delete_button = QPushButton("删除")
        profile_actions.addWidget(self.profile_load_button)
        profile_actions.addWidget(self.profile_save_button)
        profile_actions.addWidget(self.profile_delete_button)
        profile_form.addRow("课程名称", self.profile_name)
        profile_form.addRow("操作", profile_actions)
        profile_hint = QLabel("课程配置会保存领域、术语、翻译记忆、表达风格、音色与说话指令；不同老师可使用不同配置。")
        profile_hint.setObjectName("hint")
        profile_hint.setWordWrap(True)
        profile_form.addRow("", profile_hint)
        grid.addWidget(profile_group, 6, 0, 1, 2)
        layout.addLayout(grid)
        actions = QHBoxLayout()
        self.save_button = QPushButton("保存全部设置")
        self.save_button.setObjectName("primaryButton")
        official = QPushButton("打开极速直译官方文档")
        official.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://help.aliyun.com/zh/model-studio/qwen3-5-livetranslate-flash-realtime")
            )
        )
        vb_cable = QPushButton("打开 VB-CABLE 官网")
        vb_cable.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/")))
        actions.addWidget(self.save_button)
        actions.addWidget(official)
        actions.addWidget(vb_cable)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return tab

    def _connect_signals(self) -> None:
        self.signals.direct_down.connect(self.start_direct)
        self.signals.direct_up.connect(self.stop_direct)
        self.signals.translate_down.connect(self.start_translation)
        self.signals.translate_up.connect(self.stop_translation_capture)
        self.signals.cancel.connect(self.cancel_all)
        self.signals.asr_preview.connect(self.update_asr_preview)
        self.signals.live_translation_preview.connect(self.update_live_translation_preview)
        self.signals.status.connect(self._set_status)
        self.signals.error.connect(self.show_error)
        self.signals.translation_ready.connect(self.on_translation_ready)
        self.signals.direct_caption_finished.connect(self.on_direct_caption_finished)
        self.signals.teacher_preview.connect(self.update_teacher_preview)
        self.signals.teacher_finished.connect(self.on_teacher_finished)
        self.signals.tts_finished.connect(self.on_tts_finished)
        self.signals.test_finished.connect(self.on_test_finished)
        self.signals.clone_finished.connect(self.on_clone_finished)
        self.signals.summary_ready.connect(self.on_summary_ready)
        self.direct_button.hold_pressed.connect(self.start_direct)
        self.direct_button.hold_released.connect(self.stop_direct)
        self.translate_button.hold_pressed.connect(self.start_translation)
        self.translate_button.hold_released.connect(self.stop_translation_capture)
        self.stop_button.clicked.connect(self.cancel_all)
        self.play_button.clicked.connect(self.play_current_english)
        self.teacher_button.clicked.connect(self.toggle_teacher_caption)
        self.clear_button.clicked.connect(self.clear_text)
        self.typed_send_button.clicked.connect(self.send_typed_text)
        self.typed_input.send_requested.connect(self.send_typed_text)
        self.english_text.textChanged.connect(self.sync_overlay_from_editors)
        self.record_button.clicked.connect(self.toggle_recording)
        self.overlay_toggle_button.clicked.connect(self.toggle_overlay)
        self.open_output_button.clicked.connect(self.open_output_directory)
        self.export_subtitles_button.clicked.connect(self.export_current_subtitles)
        self.summary_button.clicked.connect(self.generate_meeting_summary)
        self.choose_output_button.clicked.connect(self.choose_output_directory)
        self.save_button.clicked.connect(self.save_settings_from_ui)
        self.refresh_devices_button.clicked.connect(self.refresh_audio_devices)
        self.audio_diagnostics_button.clicked.connect(self.run_audio_diagnostics)
        self.test_api_button.clicked.connect(self.test_api)
        self.clone_voice_button.clicked.connect(
            lambda: self.open_clone_dialog(self.tts_model.currentText())
        )
        self.clone_live_voice_button.clicked.connect(
            lambda: self.open_clone_dialog(self.live_translate_model.currentText())
        )
        self.translation_engine.currentIndexChanged.connect(self.update_translation_engine_ui)
        self.live_voice_clone_mode.currentIndexChanged.connect(self.update_translation_engine_ui)
        self.glossary_button.clicked.connect(self.open_glossary_dialog)
        self.profile_load_button.clicked.connect(self.load_selected_profile)
        self.profile_save_button.clicked.connect(self.save_selected_profile)
        self.profile_delete_button.clicked.connect(self.delete_selected_profile)
        self.profile_quick.currentIndexChanged.connect(self.load_quick_profile)
        self.history.cellDoubleClicked.connect(self.load_history_row)
        self.history_search.textChanged.connect(self.filter_history)
        self.show_key.toggled.connect(
            lambda checked: self.api_key.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        )
        self.subtitle_display_mode_quick.currentIndexChanged.connect(self._display_mode_from_quick)
        self.subtitle_display_mode.currentIndexChanged.connect(self._display_mode_from_settings)
        self.theme.currentIndexChanged.connect(self.apply_theme)
        for control in (
            self.overlay_enabled,
            self.overlay_always_on_top,
            self.overlay_opacity,
            self.overlay_chinese_font_size,
            self.overlay_english_font_size,
            self.overlay_width,
            self.overlay_height,
        ):
            if isinstance(control, QCheckBox):
                control.toggled.connect(self.apply_overlay_settings)
            else:
                control.valueChanged.connect(self.apply_overlay_settings)

    def current_settings(self) -> dict[str, Any]:
        values = dict(self.settings.values)
        values.update(
            {
                "workspace_id": self.workspace_id.text().strip(),
                "input_device": self.input_device.currentData(),
                "teams_output_device": self.teams_output_device.currentData(),
                "loopback_device": self.loopback_device.currentData(),
                "monitor_enabled": self.monitor_enabled.isChecked(),
                "monitor_output_device": self.monitor_output_device.currentData(),
                "direct_sample_rate": self.direct_sample_rate.currentData(),
                "asr_model": self.asr_model.currentText(),
                "asr_language": self.asr_language.currentData(),
                "vad_threshold": self.vad_threshold.value(),
                "vad_silence_ms": self.vad_silence_ms.value(),
                "direct_caption_enabled": self.direct_caption_enabled.isChecked(),
                "teacher_caption_enabled": self.teacher_caption_enabled.isChecked(),
                "teacher_asr_language": "en",
                "auto_start_teacher_caption": self.auto_start_teacher_caption.isChecked(),
                "translation_engine": self.translation_engine.currentData(),
                "live_translate_model": self.live_translate_model.currentText().strip(),
                "live_voice_clone_mode": self.live_voice_clone_mode.currentData(),
                "live_voice": self.live_voice.text().strip(),
                "translation_model": self.translation_model.currentText(),
                "summary_model": self.summary_model.currentText().strip(),
                "source_language": "Chinese",
                "target_language": "English",
                "translation_terms": self.translation_terms.toPlainText().strip(),
                "translation_memories": self.translation_memories.toPlainText().strip(),
                "translation_domain": self.translation_domain.text().strip(),
                "translation_style": self.translation_style.currentData(),
                "speak_mode": self.speak_mode.currentData(),
                "confirm_before_speak": self.speak_mode.currentData() == "confirm",
                "tts_model": self.tts_model.currentText(),
                "voice": self.voice.text().strip(),
                "tts_sample_rate": 24000,
                "tts_volume": self.tts_volume.value(),
                "tts_rate": self.tts_rate.value(),
                "tts_pitch": self.tts_pitch.value(),
                "tts_seed": self.tts_seed.value(),
                "tts_language_hint": "en",
                "tts_instruction": self.tts_instruction.text().strip(),
                "tts_emotion_tag": self.tts_emotion.currentData(),
                "enable_aigc_tag": self.enable_aigc_tag.isChecked(),
                "aigc_propagator": self.aigc_propagator.text().strip(),
                "aigc_propagate_id": self.aigc_propagate_id.text().strip(),
                "direct_hotkey": self.direct_hotkey.currentText(),
                "translate_hotkey": self.translate_hotkey.currentText(),
                "cancel_hotkey": self.cancel_hotkey.currentText(),
                "request_timeout": self.request_timeout.value(),
                "http_proxy": self.http_proxy.text().strip(),
                "overlay_enabled": self.overlay_enabled.isChecked(),
                "overlay_opacity": self.overlay_opacity.value(),
                "overlay_always_on_top": self.overlay_always_on_top.isChecked(),
                "overlay_chinese_font_size": self.overlay_chinese_font_size.value(),
                "overlay_english_font_size": self.overlay_english_font_size.value(),
                "overlay_width": self.overlay_width.value(),
                "overlay_height": self.overlay_height.value(),
                "subtitle_display_mode": self.subtitle_display_mode.currentData(),
                "theme": self.theme.currentData(),
                "active_profile": self.profile_quick.currentData() or "",
                "output_directory": self.output_directory.text().strip(),
            }
        )
        return values

    def load_settings_into_ui(self) -> None:
        v = self.settings.values
        self.workspace_id.setText(v["workspace_id"])
        self._select_data(self.input_device, v["input_device"])
        self._select_data(self.teams_output_device, v["teams_output_device"])
        self._select_data(self.loopback_device, v["loopback_device"])
        self.monitor_enabled.setChecked(v["monitor_enabled"])
        self._select_data(self.monitor_output_device, v["monitor_output_device"])
        self._select_data(self.direct_sample_rate, v["direct_sample_rate"])
        self.asr_model.setCurrentText(v["asr_model"])
        self._select_data(self.asr_language, v["asr_language"])
        self.vad_threshold.setValue(float(v["vad_threshold"]))
        self.vad_silence_ms.setValue(int(v["vad_silence_ms"]))
        self.direct_caption_enabled.setChecked(bool(v["direct_caption_enabled"]))
        self.teacher_caption_enabled.setChecked(bool(v["teacher_caption_enabled"]))
        self.auto_start_teacher_caption.setChecked(bool(v["auto_start_teacher_caption"]))
        self._select_data(self.translation_engine, v.get("translation_engine", "live"))
        self.live_translate_model.setCurrentText(
            v.get("live_translate_model", "qwen3.5-livetranslate-flash-realtime")
        )
        self._select_data(self.live_voice_clone_mode, v.get("live_voice_clone_mode", "once"))
        self.live_voice.setText(v.get("live_voice", ""))
        self.translation_model.setCurrentText(v["translation_model"])
        self.summary_model.setCurrentText(v["summary_model"])
        self.translation_domain.setText(v["translation_domain"])
        self.translation_terms.setPlainText(v["translation_terms"])
        self.translation_memories.setPlainText(v["translation_memories"])
        self._select_data(self.translation_style, v["translation_style"])
        speak_mode = v.get("speak_mode") or ("confirm" if v.get("confirm_before_speak") else "auto")
        self._select_data(self.speak_mode, speak_mode)
        self.tts_model.setCurrentText(v["tts_model"])
        self.voice.setText(v["voice"])
        self.tts_volume.setValue(int(v["tts_volume"]))
        self.tts_rate.setValue(float(v["tts_rate"]))
        self.tts_pitch.setValue(float(v["tts_pitch"]))
        self.tts_seed.setValue(int(v["tts_seed"]))
        self._select_data(self.tts_emotion, v["tts_emotion_tag"])
        self.tts_instruction.setText(v["tts_instruction"])
        self.enable_aigc_tag.setChecked(v["enable_aigc_tag"])
        self.aigc_propagator.setText(v["aigc_propagator"])
        self.aigc_propagate_id.setText(v["aigc_propagate_id"])
        self.direct_hotkey.setCurrentText(v["direct_hotkey"])
        self.translate_hotkey.setCurrentText(v["translate_hotkey"])
        self.cancel_hotkey.setCurrentText(v["cancel_hotkey"])
        self.request_timeout.setValue(int(v["request_timeout"]))
        self.http_proxy.setText(v["http_proxy"])
        self.overlay_enabled.setChecked(bool(v["overlay_enabled"]))
        self.overlay_opacity.setValue(int(v["overlay_opacity"]))
        self.overlay_always_on_top.setChecked(bool(v["overlay_always_on_top"]))
        self.overlay_chinese_font_size.setValue(int(v["overlay_chinese_font_size"]))
        self.overlay_english_font_size.setValue(int(v["overlay_english_font_size"]))
        self.overlay_width.setValue(int(v["overlay_width"]))
        self.overlay_height.setValue(int(v["overlay_height"]))
        self._select_data(self.subtitle_display_mode, v["subtitle_display_mode"])
        self._select_data(self.subtitle_display_mode_quick, v["subtitle_display_mode"])
        self._select_data(self.theme, v["theme"])
        self.output_directory.setText(v["output_directory"])
        self._select_data(self.profile_quick, v.get("active_profile", ""))
        self.profile_name.setCurrentText(v.get("active_profile", ""))
        self.apply_overlay_settings()
        self.apply_subtitle_display_mode()
        self.apply_theme()
        self.update_translation_engine_ui()

    @staticmethod
    def _select_data(box: QComboBox, value: Any) -> None:
        index = box.findData(value)
        if index >= 0:
            box.setCurrentIndex(index)

    def update_translation_engine_ui(self, *_args) -> None:
        live = self.translation_engine.currentData() == "live"
        if live:
            self._select_data(self.speak_mode, "auto")
        self.speak_mode.setEnabled(not live)
        self.speak_mode.setToolTip(
            "极速模式会边生成边播放，因此固定为自动发送。" if live else ""
        )
        self.translate_button.setText(
            "按住极速翻译说话\nF9" if live else "按住传统翻译说话\nF9"
        )
        for control in (
            self.live_translate_model,
            self.live_voice_clone_mode,
            self.clone_live_voice_button,
        ):
            control.setEnabled(live)
        self.live_voice.setEnabled(
            live and self.live_voice_clone_mode.currentData() == "fixed"
        )

    def apply_overlay_settings(self, *_args) -> None:
        if not hasattr(self, "overlay"):
            return
        values = self.current_settings()
        self.overlay.configure(
            values["overlay_opacity"],
            values["overlay_always_on_top"],
            values["overlay_chinese_font_size"],
            values["overlay_english_font_size"],
        )
        width = values["overlay_width"]
        height = values["overlay_height"]
        if self.overlay.isVisible():
            x, y = self.overlay.x(), self.overlay.y()
        else:
            x, y = values.get("overlay_x"), values.get("overlay_y")
            if x is None or y is None:
                screen = QApplication.primaryScreen()
                available = screen.availableGeometry() if screen is not None else None
                if available is not None:
                    x = available.x() + max(0, (available.width() - width) // 2)
                    y = available.y() + max(0, available.height() - height - 90)
                else:
                    x, y = 100, 100
        self.overlay.setGeometry(int(x), int(y), width, height)
        if values["overlay_enabled"]:
            self.overlay.show()
            self.overlay_toggle_button.setText("隐藏悬浮字幕")
        else:
            self.overlay.hide()
            self.overlay_toggle_button.setText("显示悬浮字幕")

    def _display_mode_from_quick(self, *_args) -> None:
        mode = self.subtitle_display_mode_quick.currentData()
        self.subtitle_display_mode.blockSignals(True)
        self._select_data(self.subtitle_display_mode, mode)
        self.subtitle_display_mode.blockSignals(False)
        self.apply_subtitle_display_mode()

    def _display_mode_from_settings(self, *_args) -> None:
        mode = self.subtitle_display_mode.currentData()
        self.subtitle_display_mode_quick.blockSignals(True)
        self._select_data(self.subtitle_display_mode_quick, mode)
        self.subtitle_display_mode_quick.blockSignals(False)
        self.apply_subtitle_display_mode()

    def apply_subtitle_display_mode(self) -> None:
        mode = self.subtitle_display_mode.currentData() or "both"
        self.zh_group.setVisible(mode in {"zh", "both"})
        self.en_group.setVisible(mode in {"en", "both"})
        self.sync_overlay_from_editors()

    def _visible_subtitle_texts(self, chinese: str, english: str) -> tuple[str, str]:
        mode = self.subtitle_display_mode.currentData() or "both"
        if mode == "zh":
            return chinese, ""
        if mode == "en":
            return "", english
        return chinese, english

    def apply_theme(self, *_args) -> None:
        theme = self.theme.currentData() if hasattr(self, "theme") else "light"
        self.setStyleSheet(DARK_STYLE_SHEET if theme == "dark" else STYLE_SHEET)

    def toggle_overlay(self) -> None:
        self.overlay_enabled.setChecked(not self.overlay.isVisible())
        self.apply_overlay_settings()
        self.settings.update({"overlay_enabled": self.overlay_enabled.isChecked()})

    def _save_overlay_geometry(self, x: int, y: int, width: int, height: int) -> None:
        self.overlay_width.blockSignals(True)
        self.overlay_height.blockSignals(True)
        self.overlay_width.setValue(width)
        self.overlay_height.setValue(height)
        self.overlay_width.blockSignals(False)
        self.overlay_height.blockSignals(False)
        self.settings.update(
            {
                "overlay_x": x,
                "overlay_y": y,
                "overlay_width": width,
                "overlay_height": height,
            }
        )

    def output_path(self) -> Path:
        configured = self.output_directory.text().strip()
        return Path(configured).expanduser() if configured else default_output_directory()

    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择录音和字幕保存目录",
            str(self.output_path()),
        )
        if selected:
            self.output_directory.setText(selected)

    def open_output_directory(self) -> None:
        path = self.output_path()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def cache_subtitle_record(self, chinese: str, english: str, source: str) -> None:
        try:
            self.subtitle_session.add(chinese, english, source)
            self.subtitle_exported = False
        except OSError as exc:
            log.error("Subtitle cache failed: %s", exc)
            self._set_status(f"字幕临时缓存失败：{exc}")

    def export_current_subtitles(self) -> bool:
        if not self.subtitle_session.records:
            QMessageBox.information(self, "保存字幕", "当前还没有可保存的字幕。")
            return False
        values = self.current_settings()
        dialog = SubtitleExportDialog(
            self.output_path() / "subtitles",
            values["subtitle_export_language"],
            values["subtitle_export_format"],
            self,
        )
        dialog.include_source.setChecked(bool(values["subtitle_export_include_source"]))
        dialog.include_time.setChecked(bool(values["subtitle_export_include_time"]))
        if dialog.exec() != QDialog.Accepted:
            return False
        language = dialog.language.currentData()
        file_format = dialog.file_format.currentData()
        include_source = dialog.include_source.isChecked()
        include_time = dialog.include_time.isChecked()
        directory = Path(dialog.directory.text().strip())
        try:
            paths = self.subtitle_session.export(
                directory,
                language=language,
                file_format=file_format,
                include_source=include_source,
                include_time=include_time,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "字幕保存失败", str(exc))
            return False
        self.settings.update(
            {
                "subtitle_export_language": language,
                "subtitle_export_format": file_format,
                "subtitle_export_include_source": include_source,
                "subtitle_export_include_time": include_time,
            }
        )
        self.subtitle_exported = True
        QMessageBox.information(
            self,
            "字幕保存完成",
            "已保存：\n" + "\n".join(str(path) for path in paths),
        )
        return True

    def toggle_recording(self) -> None:
        if self.recorder.running.is_set():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        try:
            self.recorder.start(
                self.input_device.currentData(),
                self.loopback_device.currentData(),
                self.output_path() / "recordings",
                sample_rate=16000,
            )
        except Exception as exc:
            self.show_error(
                f"双向录音启动失败：{exc}\n请检查麦克风权限、老师系统声设备，或关闭声卡独占模式。"
            )
            return
        self.recording_started_at = time.monotonic()
        self.record_timer.start()
        self.record_button.setProperty("recording", True)
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self._update_recording_status()

    def _update_recording_status(self) -> None:
        if self.recording_started_at is None:
            return
        elapsed = int(time.monotonic() - self.recording_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self.record_button.setText(f"■ 停止录音  {minutes:02d}:{seconds:02d}")

    def stop_recording(self) -> None:
        if not self.recorder.running.is_set() and self.recording_started_at is None:
            return
        paths = self.recorder.stop()
        self.record_timer.stop()
        self.recording_started_at = None
        self.record_button.setText("● 开始录音")
        self.record_button.setProperty("recording", False)
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        if paths:
            self._set_status("三轨录音已保存：麦克风 / 老师系统声 / 混合会议")

    def refresh_audio_devices(self) -> None:
        saved_input = self.input_device.currentData() if hasattr(self, "input_device") else None
        saved_output = self.teams_output_device.currentData() if hasattr(self, "teams_output_device") else None
        saved_monitor = self.monitor_output_device.currentData() if hasattr(self, "monitor_output_device") else None
        saved_loopback = self.loopback_device.currentData() if hasattr(self, "loopback_device") else None
        try:
            inputs, outputs = list_audio_devices()
            loopbacks = list_loopback_devices()
        except Exception as exc:
            self.show_error(f"读取音频设备失败：{exc}")
            return
        self.input_device.clear()
        self.teams_output_device.clear()
        self.monitor_output_device.clear()
        self.loopback_device.clear()
        self.input_device.addItem("系统默认输入", None)
        self.teams_output_device.addItem("系统默认输出", None)
        self.monitor_output_device.addItem("系统默认输出", None)
        self.loopback_device.addItem("Windows 默认扬声器回环", None)
        for item in inputs:
            self.input_device.addItem(item.label, item.index)
        for item in outputs:
            self.teams_output_device.addItem(item.label, item.index)
            self.monitor_output_device.addItem(item.label, item.index)
        for item in loopbacks:
            self.loopback_device.addItem(item.label, item.index)
        self._select_data(self.input_device, saved_input if saved_input is not None else self.settings.get("input_device"))
        self._select_data(self.teams_output_device, saved_output if saved_output is not None else self.settings.get("teams_output_device"))
        self._select_data(self.monitor_output_device, saved_monitor if saved_monitor is not None else self.settings.get("monitor_output_device"))
        self._select_data(self.loopback_device, saved_loopback if saved_loopback is not None else self.settings.get("loopback_device"))

    def run_audio_diagnostics(self) -> None:
        try:
            inputs, outputs = list_audio_devices()
            loopbacks = list_loopback_devices()
            selected_input = self.input_device.currentText()
            selected_virtual = self.teams_output_device.currentText()
            selected_loopback = self.loopback_device.currentText()
            report = [
                "音频设备诊断通过。",
                "",
                f"普通输入设备：{len(inputs)} 个",
                f"普通输出设备：{len(outputs)} 个",
                f"WASAPI 回环设备：{len(loopbacks)} 个",
                "",
                f"物理麦克风：{selected_input}",
                f"Teams 虚拟输出：{selected_virtual}",
                f"老师系统声：{selected_loopback}",
            ]
            if not loopbacks:
                report.extend(["", "警告：没有找到 WASAPI 回环设备，老师字幕和系统声录音不可用。"])
            if "CABLE" not in selected_virtual.upper():
                report.extend(["", "提示：Teams 虚拟输出似乎不是 VB-CABLE，请确认 Teams 麦克风路由。"])
            QMessageBox.information(self, "音频设备诊断", "\n".join(report))
        except Exception as exc:
            QMessageBox.critical(self, "音频设备诊断失败", str(exc))

    def open_glossary_dialog(self) -> None:
        dialog = GlossaryDialog(self.translation_terms.toPlainText(), self)
        if dialog.exec() == QDialog.Accepted:
            self.translation_terms.setPlainText(dialog.serialized())

    def _refresh_profile_boxes(self, selected: str = "") -> None:
        self.profile_name.blockSignals(True)
        self.profile_quick.blockSignals(True)
        self.profile_name.clear()
        self.profile_name.addItem("")
        self.profile_quick.clear()
        self.profile_quick.addItem("课程：默认", "")
        for name in self.profile_store.names():
            self.profile_name.addItem(name)
            self.profile_quick.addItem(f"课程：{name}", name)
        self.profile_name.setCurrentText(selected)
        self._select_data(self.profile_quick, selected)
        self.profile_name.blockSignals(False)
        self.profile_quick.blockSignals(False)

    def save_selected_profile(self) -> None:
        name = self.profile_name.currentText().strip()
        if not name:
            name, ok = QInputDialog.getText(self, "保存课程配置", "课程名称：")
            if not ok:
                return
            name = name.strip()
        try:
            self.profile_store.save(name, self.current_settings())
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "课程配置保存失败", str(exc))
            return
        self._refresh_profile_boxes(name)
        self.settings.update({"active_profile": name})
        self._set_status(f"课程配置已保存：{name}")

    def _load_profile(self, name: str) -> bool:
        profile = self.profile_store.get(name)
        if profile is None:
            return False
        self._select_data(
            self.translation_engine,
            profile.get("translation_engine", "live") or "live",
        )
        self.translation_domain.setText(str(profile.get("translation_domain", "")))
        self.translation_terms.setPlainText(str(profile.get("translation_terms", "")))
        self.translation_memories.setPlainText(str(profile.get("translation_memories", "")))
        self._select_data(self.translation_style, profile.get("translation_style", "polite"))
        self._select_data(
            self.live_voice_clone_mode,
            profile.get("live_voice_clone_mode", "once") or "once",
        )
        self.live_voice.setText(str(profile.get("live_voice", "")))
        self.tts_instruction.setText(str(profile.get("tts_instruction", "")))
        self.voice.setText(str(profile.get("voice", "")))
        self.profile_name.setCurrentText(name)
        self.profile_quick.blockSignals(True)
        self._select_data(self.profile_quick, name)
        self.profile_quick.blockSignals(False)
        self.settings.update({"active_profile": name})
        self.update_translation_engine_ui()
        self._set_status(f"已载入课程配置：{name}")
        return True

    def load_selected_profile(self) -> None:
        name = self.profile_name.currentText().strip()
        if not name or not self._load_profile(name):
            QMessageBox.information(self, "课程配置", "请选择一个已经保存的课程配置。")

    def load_quick_profile(self, *_args) -> None:
        name = self.profile_quick.currentData() or ""
        if name:
            self._load_profile(str(name))
        else:
            self.settings.update({"active_profile": ""})

    def delete_selected_profile(self) -> None:
        name = self.profile_name.currentText().strip()
        if not name:
            return
        choice = QMessageBox.question(
            self,
            "删除课程配置",
            f"确认删除课程配置“{name}”？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            self.profile_store.delete(name)
        except OSError as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self._refresh_profile_boxes("")
        self.settings.update({"active_profile": ""})

    def load_history_row(self, row: int, _column: int) -> None:
        chinese = self.history.item(row, 2)
        english = self.history.item(row, 3)
        self.chinese_text.setPlainText(chinese.text() if chinese else "")
        self.english_text.setPlainText(english.text() if english else "")
        self.sync_overlay_from_editors()
        self._set_status("已载入历史句子，可修改英文后重新播放")

    def filter_history(self, query: str) -> None:
        needle = query.strip().casefold()
        for row in range(self.history.rowCount()):
            haystack = " ".join(
                self.history.item(row, column).text()
                for column in range(self.history.columnCount())
                if self.history.item(row, column) is not None
            ).casefold()
            self.history.setRowHidden(row, bool(needle) and needle not in haystack)

    def generate_meeting_summary(self) -> None:
        if not self.subtitle_session.records:
            QMessageBox.information(self, "课堂总结", "当前还没有双向字幕记录。")
            return
        values = self.current_settings()
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先设置 Workspace ID 和 API Key，再生成课堂总结。")
            return
        transcript = "\n".join(
            f"[{record.at:%H:%M:%S}] {record.source}\n"
            f"中文：{record.chinese}\nEnglish: {record.english}"
            for record in self.subtitle_session.records
        )
        if len(transcript) > 80000:
            transcript = transcript[-80000:]
        prompt = (
            "请根据下面的双向课堂记录，用中文输出 Markdown 总结。必须包括：\n"
            "1. 课程主题与简要结论；2. 按主题整理的知识点；3. 老师的解释和回答；"
            "4. 作业、截止时间和待办事项；5. 值得复习的英文表达与专业术语；"
            "6. 没有听清或需要再次确认的问题。\n"
            "只使用记录中明确出现的信息，不要补造内容。\n\n"
            f"{transcript}"
        )
        self.summary_button.setEnabled(False)
        self._set_status("正在生成课堂总结…")

        def worker() -> None:
            try:
                result = self._make_client(values).complete(
                    prompt, model=values["summary_model"] or "qwen-plus"
                )
                self.signals.summary_ready.emit(True, result)
            except Exception as exc:
                self.signals.summary_ready.emit(False, str(exc))

        threading.Thread(target=worker, name="meeting-summary", daemon=True).start()

    def on_summary_ready(self, ok: bool, result: str) -> None:
        self.summary_button.setEnabled(True)
        if not ok:
            self._set_status("课堂总结生成失败")
            QMessageBox.critical(self, "课堂总结生成失败", result)
            return
        dialog = MeetingSummaryDialog(result, self)
        if dialog.exec() != QDialog.Accepted:
            self._set_status("课堂总结已生成但未保存")
            return
        directory = self.output_path() / "summaries"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"meeting_summary_{datetime.now():%Y%m%d_%H%M%S}.md"
        try:
            path.write_text(dialog.editor.toPlainText().strip() + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "课堂总结保存失败", str(exc))
            return
        self._set_status(f"课堂总结已保存：{path.name}")
        QMessageBox.information(self, "课堂总结保存完成", str(path))

    def offer_cache_recovery(self) -> None:
        paths = SubtitleSession.discover_recoverable(exclude=self.subtitle_session.cache_path)
        if not paths:
            return
        choice = QMessageBox.question(
            self,
            "发现未保存的会议字幕",
            f"发现 {len(paths)} 个上次异常退出遗留的字幕缓存。是否恢复到本次时间轴？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice != QMessageBox.Yes:
            return
        imported = 0
        for path in paths:
            imported += self.subtitle_session.import_cache(path)
        if imported:
            self.subtitle_session.normalize_recovered_timeline()
            self.history.setRowCount(0)
            for record in self.subtitle_session.records:
                self._append_history_row(
                    record.at.strftime("%H:%M:%S"),
                    record.source,
                    record.chinese,
                    record.english,
                    "恢复",
                )
            self.subtitle_exported = False
            self._set_status(f"已恢复 {imported} 条未保存字幕")

    def save_settings_from_ui(self) -> None:
        if self.direct_hotkey.currentText() == self.translate_hotkey.currentText():
            QMessageBox.warning(self, "快捷键冲突", "原声和翻译快捷键不能相同。")
            return
        try:
            if self.api_key.text().strip():
                self.settings.set_api_key(self.api_key.text().strip())
                self.api_key.clear()
            self.settings.update(self.current_settings())
            self.apply_overlay_settings()
            self.apply_subtitle_display_mode()
            self.apply_theme()
            self.start_hotkeys()
            self._set_status("设置已保存")
        except Exception as exc:
            self.show_error(f"保存失败：{exc}")

    def start_hotkeys(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        values = self.current_settings()
        self.hotkeys = HoldHotkeys(
            direct_key=values["direct_hotkey"],
            translate_key=values["translate_hotkey"],
            cancel_key=values["cancel_hotkey"],
            direct_press=self.signals.direct_down.emit,
            direct_release=self.signals.direct_up.emit,
            translate_press=self.signals.translate_down.emit,
            translate_release=self.signals.translate_up.emit,
            cancel=self.signals.cancel.emit,
        )
        self.hotkeys.start()

    def _can_begin(self, new_state: str) -> bool:
        with self.state_lock:
            if self.state != "idle":
                return False
            self.state = new_state
            return True

    def _set_idle(self) -> None:
        with self.state_lock:
            self.state = "idle"
        self.direct_button.setProperty("active", False)
        self.translate_button.setProperty("active", False)
        self.direct_button.style().unpolish(self.direct_button)
        self.direct_button.style().polish(self.direct_button)
        self.translate_button.style().unpolish(self.translate_button)
        self.translate_button.style().polish(self.translate_button)

    def _make_client(self, values: dict[str, Any] | None = None) -> BailianClient:
        values = values or self.current_settings()
        return BailianClient(
            self.settings.get_api_key(),
            values["workspace_id"],
            timeout=values["request_timeout"],
            proxy=values["http_proxy"],
        )

    def toggle_teacher_caption(self) -> None:
        if self.teacher_active:
            self.stop_teacher_caption()
        else:
            self.start_teacher_caption()

    def start_teacher_caption(self) -> None:
        if self.teacher_active:
            return
        values = self.current_settings()
        if not values["teacher_caption_enabled"]:
            self._set_status("请先在设置中启用老师/Teams 字幕")
            return
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先设置 Workspace ID 和 API Key，再开启老师字幕。")
            return
        self.teacher_cancel_event.clear()
        self.teacher_audio_queue = queue.Queue(maxsize=500)
        self.teacher_segment_queue = queue.Queue()

        def capture_audio(chunk: bytes) -> None:
            if self.tts_active.is_set() or self.teacher_audio_queue is None:
                return
            try:
                self.teacher_audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self.teacher_audio_queue.get_nowait()
                    self.teacher_audio_queue.put_nowait(chunk)
                except queue.Empty:
                    pass

        try:
            device = self.teacher_capture.start(
                values["loopback_device"], capture_audio, target_rate=16000, block_ms=100
            )
        except Exception as exc:
            self.show_error(f"老师系统声监听启动失败：{exc}")
            return
        self.teacher_active = True
        self.teacher_button.setText("■ 停止听老师")
        self.teacher_button.setProperty("active", True)
        self.teacher_button.style().unpolish(self.teacher_button)
        self.teacher_button.style().polish(self.teacher_button)
        self._set_status(f"正在听老师 · {device.name}")
        self.teacher_translation_thread = threading.Thread(
            target=self._teacher_translation_worker,
            args=(values,),
            name="teacher-translation",
            daemon=True,
        )
        self.teacher_translation_thread.start()
        self.teacher_thread = threading.Thread(
            target=self._teacher_asr_worker,
            args=(values,),
            name="teacher-asr",
            daemon=True,
        )
        self.teacher_thread.start()

    def stop_teacher_caption(self) -> None:
        if not self.teacher_active:
            return
        self.teacher_capture.stop()
        if self.teacher_audio_queue is not None:
            try:
                self.teacher_audio_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.teacher_audio_queue.get_nowait()
                    self.teacher_audio_queue.put_nowait(None)
                except queue.Empty:
                    pass
        self._set_status("正在结束老师字幕并整理最后一句…")

    def _teacher_asr_worker(self, values: dict[str, Any]) -> None:
        asr: QwenRealtimeASR | None = None
        message = "老师字幕已停止"
        try:
            client = self._make_client(values)

            def completed_segment(text: str, _emotion: str) -> None:
                if text.strip() and self.teacher_segment_queue is not None:
                    self.teacher_segment_queue.put(text.strip())

            asr = QwenRealtimeASR(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                model=values["asr_model"],
                language="en",
                vad_threshold=values["vad_threshold"],
                vad_silence_ms=values["vad_silence_ms"],
                on_preview=lambda text, _emotion: self.signals.teacher_preview.emit(text),
                on_status=lambda _status: None,
                on_error=lambda error: log.warning("Teacher ASR: %s", error),
                on_segment=completed_segment,
                combine_previews=False,
            )
            asr.start()
            asr.wait_ready()
            assert self.teacher_audio_queue is not None
            while not self.teacher_cancel_event.is_set():
                chunk = self.teacher_audio_queue.get()
                if chunk is None:
                    break
                asr.send_audio(chunk)
            if not self.teacher_cancel_event.is_set():
                asr.finish()
        except Exception as exc:
            log.exception("Teacher caption pipeline failed")
            message = f"老师字幕失败：{exc}"
        finally:
            if asr is not None:
                asr.close()
            self.teacher_capture.stop()
            if self.teacher_segment_queue is not None:
                self.teacher_segment_queue.put(None)
            if self.teacher_translation_thread is not None:
                self.teacher_translation_thread.join(timeout=20.0)
            self.signals.teacher_finished.emit(message)

    def _teacher_translation_worker(self, values: dict[str, Any]) -> None:
        try:
            client = self._make_client(values)
            segment_queue = self.teacher_segment_queue
            assert segment_queue is not None
            while not self.teacher_cancel_event.is_set():
                english = segment_queue.get()
                if english is None:
                    break
                started_at = time.perf_counter()
                chinese = client.translate_between(
                    english,
                    values,
                    source_language="English",
                    target_language="Chinese",
                )
                self.signals.translation_ready.emit(
                    chinese,
                    english,
                    time.perf_counter() - started_at,
                    "老师 / Teams",
                )
        except Exception as exc:
            log.exception("Teacher translation failed")
            self.signals.status.emit(f"老师翻译失败：{exc}")

    def update_teacher_preview(self, english: str) -> None:
        if self.awaiting_confirmation:
            return
        self.english_text.setPlainText(english)
        self.emotion_label.setText("来源：老师 / Teams 系统声")
        self.sync_overlay_from_editors()

    def on_teacher_finished(self, message: str) -> None:
        self.teacher_active = False
        self.teacher_audio_queue = None
        self.teacher_segment_queue = None
        self.teacher_button.setText("▶ 开始听老师 / Teams")
        self.teacher_button.setProperty("active", False)
        self.teacher_button.style().unpolish(self.teacher_button)
        self.teacher_button.style().polish(self.teacher_button)
        if message and not self.teacher_cancel_event.is_set():
            self._set_status(message)

    def start_direct(self) -> None:
        if not self._can_begin("direct"):
            return
        values = self.current_settings()
        self.cancel_event.clear()
        self.awaiting_confirmation = False
        captions_enabled = bool(
            values["direct_caption_enabled"]
            and self.settings.get_api_key()
            and values["workspace_id"]
        )
        self.direct_caption_active = captions_enabled
        tap = None
        if captions_enabled:
            self.chinese_text.clear()
            self.english_text.clear()
            self.sync_overlay_from_editors()
            self.direct_caption_queue = queue.Queue(maxsize=400)

            def caption_tap(chunk: bytes) -> None:
                if self.direct_caption_queue is None:
                    return
                try:
                    self.direct_caption_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        self.direct_caption_queue.get_nowait()
                        self.direct_caption_queue.put_nowait(chunk)
                    except queue.Empty:
                        pass

            tap = caption_tap
        try:
            self.direct_bridge.start(
                values["input_device"],
                values["teams_output_device"],
                values["direct_sample_rate"],
                tap=tap,
            )
            self.direct_button.setProperty("active", True)
            self.direct_button.style().polish(self.direct_button)
            if captions_enabled:
                self.pipeline_thread = threading.Thread(
                    target=self._direct_caption_worker,
                    args=(values, time.perf_counter()),
                    name="direct-caption-pipeline",
                    daemon=True,
                )
                self.pipeline_thread.start()
                self._set_status("原声直通中 · 同时识别中文字幕 · 松开后翻译")
            elif values["direct_caption_enabled"]:
                self._set_status("原声直通中 · 未配置 API，字幕暂不可用")
            else:
                self._set_status("原声通道开启 · 松开结束")
        except Exception as exc:
            self.direct_caption_active = False
            if self.direct_caption_queue is not None:
                try:
                    self.direct_caption_queue.put_nowait(None)
                except queue.Full:
                    pass
            self._set_idle()
            self.show_error(f"原声通道启动失败：{exc}")

    def stop_direct(self) -> None:
        with self.state_lock:
            if self.state != "direct":
                return
            if self.direct_caption_active:
                self.state = "processing"
        self.direct_bridge.stop()
        if self.direct_caption_active and self.direct_caption_queue is not None:
            try:
                self.direct_caption_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.direct_caption_queue.get_nowait()
                    self.direct_caption_queue.put_nowait(None)
                except queue.Empty:
                    pass
            self._set_status("原声已发送 · 正在整理字幕并翻译…")
        else:
            self._set_idle()
            self._set_status("就绪 · F8 原声 / F9 翻译")

    def _direct_caption_worker(self, values: dict[str, Any], started_at: float) -> None:
        asr: QwenRealtimeASR | None = None
        message = ""
        try:
            client = self._make_client(values)
            asr = QwenRealtimeASR(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                model=values["asr_model"],
                language=values["asr_language"],
                vad_threshold=values["vad_threshold"],
                vad_silence_ms=values["vad_silence_ms"],
                on_preview=lambda text, emotion: self.signals.asr_preview.emit(text, emotion),
                on_status=lambda _status: None,
                on_error=lambda error: log.warning("Direct ASR: %s", error),
            )
            asr.start()
            asr.wait_ready()
            resampler = Pcm16Resampler(int(values["direct_sample_rate"]), 16000)
            assert self.direct_caption_queue is not None
            while not self.cancel_event.is_set():
                chunk = self.direct_caption_queue.get()
                if chunk is None:
                    break
                converted = resampler.process(chunk)
                if converted:
                    asr.send_audio(converted)
            if self.cancel_event.is_set():
                return
            chinese = asr.finish()
            if not chinese:
                message = "原声已发送 · 本次没有识别到有效字幕"
                return
            self.signals.asr_preview.emit(chinese, "final")
            english = client.translate(chinese, values)
            self.signals.translation_ready.emit(
                chinese,
                english,
                time.perf_counter() - started_at,
                "我 / F8 原声",
            )
            message = "原声已发送 · 中文和英文字幕已生成"
        except Exception as exc:
            log.exception("Direct caption pipeline failed")
            message = f"原声已发送 · 字幕生成失败：{exc}"
        finally:
            if asr is not None:
                asr.close()
            self.signals.direct_caption_finished.emit(message)

    def on_direct_caption_finished(self, message: str) -> None:
        self.direct_caption_active = False
        self.direct_caption_queue = None
        self._set_idle()
        if message and not self.cancel_event.is_set():
            self._set_status(message)

    def start_translation(self) -> None:
        if not self._can_begin("capturing"):
            return
        values = self.current_settings()
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self._set_idle()
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先在设置页填写 Workspace ID 和 API Key。")
            return
        self.cancel_event.clear()
        self.awaiting_confirmation = False
        self.chinese_text.clear()
        self.english_text.clear()
        self.overlay.set_texts("", "")
        self.translation_queue = queue.Queue()
        self.translate_button.setProperty("active", True)
        self.translate_button.style().polish(self.translate_button)
        started_at = time.perf_counter()

        def enqueue(chunk: bytes) -> None:
            if self.translation_queue is not None:
                self.translation_queue.put(chunk)

        try:
            self.translation_capture = MicrophoneCapture(
                values["input_device"], enqueue, sample_rate=16000, block_ms=100
            )
            self.translation_capture.start()
        except Exception as exc:
            self._set_idle()
            self.show_error(f"麦克风启动失败：{exc}\n可尝试在 Windows 声音设置中把麦克风格式改为 16k/48kHz。")
            return
        self._set_status("正在听你说中文… · 松开后翻译")
        self.pipeline_thread = threading.Thread(
            target=self._translation_worker,
            args=(values, started_at),
            name="translation-pipeline",
            daemon=True,
        )
        self.pipeline_thread.start()

    def stop_translation_capture(self) -> None:
        with self.state_lock:
            if self.state != "capturing":
                return
            self.state = "processing"
        if self.translation_capture is not None:
            self.translation_capture.stop()
            self.translation_capture = None
        if self.translation_queue is not None:
            self.translation_queue.put(None)
        self._set_status("正在整理中文并翻译…")

    def _translation_worker(self, values: dict[str, Any], started_at: float) -> None:
        asr: QwenRealtimeASR | None = None
        try:
            if values.get("translation_engine") == "live":
                self._live_translation_worker(values, started_at)
                return
            client = self._make_client(values)
            asr = QwenRealtimeASR(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                model=values["asr_model"],
                language=values["asr_language"],
                vad_threshold=values["vad_threshold"],
                vad_silence_ms=values["vad_silence_ms"],
                on_preview=lambda text, emotion: self.signals.asr_preview.emit(text, emotion),
                on_status=self.signals.status.emit,
                on_error=lambda error: log.warning("ASR: %s", error),
            )
            asr.start()
            asr.wait_ready()
            assert self.translation_queue is not None
            while not self.cancel_event.is_set():
                chunk = self.translation_queue.get()
                if chunk is None:
                    break
                asr.send_audio(chunk)
            if self.cancel_event.is_set():
                asr.close()
                return
            chinese = asr.finish()
            if not chinese:
                raise ApiError("没有识别到有效语音，请靠近麦克风后重试")
            self.signals.asr_preview.emit(chinese, "final")
            self.signals.status.emit("中文识别完成，正在翻译英文…")
            english = client.translate(chinese, values)
            elapsed = time.perf_counter() - started_at
            self.last_translation = english
            self.awaiting_confirmation = bool(values["confirm_before_speak"])
            self.signals.translation_ready.emit(chinese, english, elapsed, "我 / F9 翻译")
            if not values["confirm_before_speak"] and not self.cancel_event.is_set():
                self._stream_speech(client, english, values)
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.error.emit(str(exc))
        finally:
            if asr is not None:
                asr.close()
            self.signals.tts_finished.emit()

    def _live_translation_worker(self, values: dict[str, Any], started_at: float) -> None:
        client = self._make_client(values)
        terms = parse_json_list(values.get("translation_terms", ""), "术语表")
        phrases = {item["source"]: item["target"] for item in terms}
        devices = [values["teams_output_device"]]
        if values["monitor_enabled"]:
            devices.append(values["monitor_output_device"])

        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        playback_errors: list[Exception] = []
        playback_thread: threading.Thread | None = None
        session: QwenLiveTranslate | None = None
        self.tts_active.set()
        try:
            with MultiOutputPlayer(devices, 24000) as player:
                def play_audio() -> None:
                    failed = False
                    while True:
                        item = audio_queue.get()
                        try:
                            if item is None:
                                return
                            if not failed:
                                player.write(item)
                        except Exception as exc:
                            failed = True
                            playback_errors.append(exc)
                        finally:
                            audio_queue.task_done()

                playback_thread = threading.Thread(
                    target=play_audio,
                    name="live-translate-audio-output",
                    daemon=True,
                )
                playback_thread.start()
                session = QwenLiveTranslate(
                    api_key=client.api_key,
                    workspace_id=client.workspace_id,
                    model=values["live_translate_model"],
                    source_language="zh",
                    target_language="en",
                    phrases=phrases,
                    voice_mode=values["live_voice_clone_mode"],
                    voice=values.get("live_voice", ""),
                    audio_enabled=True,
                    on_source_preview=lambda text: self.signals.asr_preview.emit(text, "live"),
                    on_translation_preview=self.signals.live_translation_preview.emit,
                    on_audio=audio_queue.put,
                    on_status=self.signals.status.emit,
                    on_error=lambda error: log.warning("LiveTranslate: %s", error),
                )
                session.start()
                session.wait_ready()
                assert self.translation_queue is not None
                while not self.cancel_event.is_set():
                    chunk = self.translation_queue.get()
                    if chunk is None:
                        break
                    session.send_audio(chunk)
                if self.cancel_event.is_set():
                    return

                self.signals.status.emit("中文已提交 · 正在直接生成英文字幕和语音…")
                result = session.commit_and_wait(
                    timeout=max(30.0, float(values["request_timeout"])),
                    cancel_event=self.cancel_event,
                )
                session.finish()
                session = None

                audio_queue.put(None)
                audio_queue.join()
                playback_thread.join(timeout=1.0)
                if playback_errors:
                    raise ApiError(f"极速直译音频播放失败：{playback_errors[0]}")

                chinese = result.source_text
                english = result.translated_text
                elapsed = time.perf_counter() - started_at
                self.last_translation = english
                self.awaiting_confirmation = False
                self.signals.translation_ready.emit(
                    chinese,
                    english,
                    elapsed,
                    "我 / F9 极速直译",
                )
                if result.first_audio_seconds is not None:
                    self.signals.status.emit(
                        f"极速直译完成 · 松开 F9 后 {result.first_audio_seconds:.2f}s 开始出声"
                    )
        finally:
            if session is not None:
                session.close()
            if self.cancel_event.is_set():
                while True:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except queue.Empty:
                        break
            if playback_thread is not None and playback_thread.is_alive():
                audio_queue.put(None)
                audio_queue.join()
                playback_thread.join(timeout=1.0)
            self.tts_active.clear()

    def _stream_speech(self, client: BailianClient, english: str, values: dict[str, Any]) -> None:
        self.signals.status.emit("正在合成并发送英文到 Teams…")
        devices = [values["teams_output_device"]]
        if values["monitor_enabled"]:
            devices.append(values["monitor_output_device"])
        self.tts_active.set()
        try:
            with MultiOutputPlayer(devices, values["tts_sample_rate"]) as player:
                count = client.stream_tts(english, values, player.write, self.cancel_event)
        finally:
            self.tts_active.clear()
        if count == 0 and not self.cancel_event.is_set():
            raise ApiError("语音合成没有返回音频数据")

    def play_current_english(self) -> None:
        text = self.english_text.toPlainText().strip()
        if not text:
            self.show_error("当前没有可播放的英文。")
            return
        if not self._can_begin("speaking"):
            return
        values = self.current_settings()
        self.cancel_event.clear()
        self.awaiting_confirmation = False

        def worker() -> None:
            try:
                self._stream_speech(self._make_client(values), text, values)
            except Exception as exc:
                self.signals.error.emit(str(exc))
            finally:
                self.signals.tts_finished.emit()

        threading.Thread(target=worker, name="manual-tts", daemon=True).start()

    def send_typed_text(self) -> None:
        text = self.typed_input.toPlainText().strip()
        if not text:
            self.show_error("请先在键盘输入框中填写要说的内容。")
            return
        if not self._can_begin("typed"):
            self._set_status("当前任务尚未结束，请稍候或按 Esc 取消")
            return
        values = self.current_settings()
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self._set_idle()
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先在设置页填写 Workspace ID 和 API Key。")
            return
        self.cancel_event.clear()
        self.awaiting_confirmation = False
        mode = self.typed_mode.currentData()
        self.chinese_text.setPlainText(text)
        self.english_text.clear()
        self.sync_overlay_from_editors()
        self._set_status("正在翻译键盘文字…" if mode == "translate" else "正在合成输入文字…")
        threading.Thread(
            target=self._typed_text_worker,
            args=(text, mode, values),
            name="typed-text-tts",
            daemon=True,
        ).start()

    def _typed_text_worker(self, text: str, mode: str, values: dict[str, Any]) -> None:
        started_at = time.perf_counter()
        try:
            client = self._make_client(values)
            if mode == "translate":
                spoken = client.translate(text, values)
                self.awaiting_confirmation = bool(values["confirm_before_speak"])
                self.signals.translation_ready.emit(
                    text,
                    spoken,
                    time.perf_counter() - started_at,
                    "我 / 键盘翻译",
                )
                if not self.awaiting_confirmation and not self.cancel_event.is_set():
                    self._stream_speech(client, spoken, values)
            else:
                direct_values = dict(values)
                direct_values["tts_language_hint"] = "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"
                if direct_values["tts_language_hint"] == "zh":
                    direct_values["tts_instruction"] = ""
                self.signals.translation_ready.emit(
                    text,
                    "",
                    time.perf_counter() - started_at,
                    "我 / 键盘原文",
                )
                self._stream_speech(client, text, direct_values)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.tts_finished.emit()

    def cancel_all(self) -> None:
        self.cancel_event.set()
        self.teacher_cancel_event.set()
        if self.translation_capture is not None:
            self.translation_capture.stop()
            self.translation_capture = None
        if self.translation_queue is not None:
            try:
                self.translation_queue.put_nowait(None)
            except Exception:
                pass
        if self.direct_caption_queue is not None:
            try:
                self.direct_caption_queue.put_nowait(None)
            except Exception:
                pass
        self.direct_bridge.stop()
        if self.teacher_active:
            self.stop_teacher_caption()
        self._set_idle()
        self._set_status("已停止 · F8 原声 / F9 翻译")

    def update_asr_preview(self, text: str, emotion: str) -> None:
        self.chinese_text.setPlainText(text)
        self.emotion_label.setText(f"识别情绪：{emotion or '—'}")
        self.sync_overlay_from_editors()

    def update_live_translation_preview(self, text: str) -> None:
        self.english_text.setPlainText(text)
        self.emotion_label.setText("来源：我 / F9 极速直译")
        self.sync_overlay_from_editors()

    def on_translation_ready(self, chinese: str, english: str, elapsed: float, source: str) -> None:
        is_teacher = source.startswith("老师")
        if not (is_teacher and self.awaiting_confirmation):
            self.chinese_text.setPlainText(chinese)
            self.english_text.setPlainText(english)
            self.emotion_label.setText(f"来源：{source}")
            self.sync_overlay_from_editors()
        self._append_history_row(
            datetime.now().strftime("%H:%M:%S"),
            source,
            chinese,
            english,
            f"{elapsed:.2f}s",
        )
        self.cache_subtitle_record(chinese, english, source)
        if self.awaiting_confirmation and not is_teacher:
            self._set_status("译文已生成 · 修改后点击“播放/发送当前英文”")

    def _append_history_row(
        self, at: str, source: str, chinese: str, english: str, elapsed: str
    ) -> None:
        row = self.history.rowCount()
        self.history.insertRow(row)
        for column, value in enumerate(
            [at, source, chinese, english, elapsed]
        ):
            self.history.setItem(row, column, QTableWidgetItem(value))
        self.history.scrollToBottom()

    def on_tts_finished(self) -> None:
        self._set_idle()
        if not self.cancel_event.is_set():
            if self.awaiting_confirmation and self.english_text.toPlainText().strip():
                self._set_status("译文已生成 · 修改后点击“播放/发送当前英文”")
            elif self.teacher_active:
                self._set_status("正在听老师 · F8/F9 仍可随时使用")
            else:
                self._set_status("就绪 · F8 原声 / F9 翻译")

    def clear_text(self) -> None:
        self.awaiting_confirmation = False
        self.chinese_text.clear()
        self.english_text.clear()
        self.typed_input.clear()
        self.emotion_label.setText("识别情绪：—")
        self.overlay.set_texts("", "")

    def sync_overlay_from_editors(self) -> None:
        chinese, english = self._visible_subtitle_texts(
            self.chinese_text.toPlainText(),
            self.english_text.toPlainText(),
        )
        self.overlay.set_texts(chinese, english)

    def test_api(self) -> None:
        self.save_settings_from_ui()
        self.test_api_button.setEnabled(False)
        self._set_status("正在测试百炼接口…")
        values = self.current_settings()

        def worker() -> None:
            try:
                client = self._make_client(values)
                if values.get("translation_engine") == "live":
                    session = QwenLiveTranslate(
                        api_key=client.api_key,
                        workspace_id=client.workspace_id,
                        model=values["live_translate_model"],
                        voice_mode="default",
                        audio_enabled=False,
                    )
                    try:
                        session.start()
                        session.wait_ready()
                        self.signals.test_finished.emit(
                            True,
                            f"连接成功：{values['live_translate_model']} 极速直译 WebSocket 已就绪。",
                        )
                    finally:
                        session.finish()
                else:
                    result = client.translate("你好，这是一次连接测试。", values)
                    self.signals.test_finished.emit(True, f"连接成功，翻译返回：{result}")
            except Exception as exc:
                self.signals.test_finished.emit(False, str(exc))

        threading.Thread(target=worker, name="api-test", daemon=True).start()

    def on_test_finished(self, ok: bool, message: str) -> None:
        self.test_api_button.setEnabled(True)
        self._set_status("API 测试成功" if ok else "API 测试失败")
        (QMessageBox.information if ok else QMessageBox.critical)(self, "API 测试", message)

    def open_clone_dialog(self, model: str | None = None) -> None:
        self.save_settings_from_ui()
        self.clone_target_model = model or self.tts_model.currentText()
        self.clone_dialog = VoiceCloneDialog(
            self.clone_target_model,
            self.settings,
            self.open_oss_settings,
            self,
        )
        self.clone_dialog.create_requested.connect(self.create_voice)
        self.clone_dialog.exec()

    def open_oss_settings(self) -> bool:
        dialog = OssSettingsDialog(self.settings, self)
        return dialog.exec() == QDialog.Accepted

    def create_voice(self, options: dict[str, Any]) -> None:
        assert self.clone_dialog is not None
        for button in self.clone_dialog.findChildren(QPushButton):
            button.setEnabled(False)
        self._set_status("正在准备本地样音…" if options.get("audio_path") else "正在通过百炼创建克隆音色…")
        values = self.current_settings()
        oss_config = self.settings.get_oss_config() if options.get("audio_path") else None

        def worker() -> None:
            uploader: OssTemporaryUploader | None = None
            uploaded: UploadedVoiceSample | None = None
            cleanup_warning = ""
            voice_id = ""
            error = ""
            try:
                request_options = dict(options)
                local_path = str(request_options.pop("audio_path", "") or "").strip()
                if local_path:
                    self.signals.status.emit("正在把本地样音转换为标准 WAV…")
                    with normalized_voice_sample(
                        local_path,
                        max_seconds=float(request_options["max_seconds"]),
                    ) as normalized_path:
                        self.signals.status.emit("正在将标准 WAV 临时上传到 OSS…")
                        assert oss_config is not None
                        uploader = OssTemporaryUploader(**oss_config)
                        uploaded = uploader.upload(normalized_path)
                        request_options["audio_url"] = uploaded.signed_url
                        self.signals.status.emit("临时样音已上传 · 正在通过百炼创建固定音色…")
                        voice_id = self._make_client(values).clone_voice(**request_options)
                else:
                    voice_id = self._make_client(values).clone_voice(**request_options)
            except Exception as exc:
                error = str(exc)
            finally:
                if uploader is not None and uploaded is not None:
                    try:
                        uploader.delete(uploaded.key)
                    except Exception as exc:
                        log.warning("Unable to delete temporary OSS voice sample %s: %s", uploaded.key, exc)
                        cleanup_warning = (
                            "OSS 临时样音删除失败，请手动删除对象："
                            f"{uploaded.key}\n错误：{exc}"
                        )
            self.signals.clone_finished.emit(not bool(error), voice_id if not error else error, cleanup_warning)

        threading.Thread(target=worker, name="voice-clone", daemon=True).start()

    def on_clone_finished(self, ok: bool, message: str, cleanup_warning: str) -> None:
        if ok:
            if self.clone_target_model.startswith("qwen3.5-livetranslate"):
                self.live_voice.setText(message)
                self._select_data(self.live_voice_clone_mode, "fixed")
            else:
                self.voice.setText(message)
                if message.startswith("qwen-audio-3.0-tts-plus-"):
                    self.tts_model.setCurrentText("qwen-audio-3.0-tts-plus")
                elif message.startswith("qwen-audio-3.0-tts-flash-"):
                    self.tts_model.setCurrentText("qwen-audio-3.0-tts-flash")
            self.save_settings_from_ui()
            if self.clone_dialog is not None:
                self.clone_dialog.accept()
            detail = f"voice_id：\n{message}\n\n已自动填入并保存。"
            if cleanup_warning:
                detail += f"\n\n注意：{cleanup_warning}"
            (QMessageBox.warning if cleanup_warning else QMessageBox.information)(
                self,
                "克隆音色创建成功",
                detail,
            )
            self._set_status("克隆音色已创建")
        else:
            if self.clone_dialog is not None:
                for button in self.clone_dialog.findChildren(QPushButton):
                    button.setEnabled(True)
            if cleanup_warning:
                message += f"\n\n{cleanup_warning}"
            QMessageBox.critical(self, "创建音色失败", message)
            self._set_status("创建音色失败")

    def show_error(self, message: str) -> None:
        log.error(message)
        self.cancel_event.set()
        if self.translation_capture is not None:
            self.translation_capture.stop()
            self.translation_capture = None
        if self.direct_caption_queue is not None:
            try:
                self.direct_caption_queue.put_nowait(None)
            except Exception:
                pass
        self.direct_bridge.stop()
        self._set_idle()
        self._set_status("发生错误")
        QMessageBox.critical(self, "错误", message)

    def _set_status(self, text: str) -> None:
        self.status_badge.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.subtitle_session.records and not self.subtitle_exported:
            choice = QMessageBox.question(
                self,
                "字幕尚未保存",
                f"本次有 {len(self.subtitle_session.records)} 条完整中英文字幕仍在临时缓存中。\n"
                "是否现在选择语言和格式并保存？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Yes and not self.export_current_subtitles():
                event.ignore()
                return
        self.cancel_all()
        self.stop_recording()
        self.overlay.close()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.subtitle_session.cleanup()
        event.accept()


STYLE_SHEET = """
QMainWindow, QWidget { background: #f4f7fb; color: #172033; font-family: "Microsoft YaHei UI"; font-size: 13px; }
QLabel#title { font-size: 28px; font-weight: 800; color: #102a56; }
QLabel#subtitle { color: #60708f; font-size: 13px; }
QLabel#statusBadge { background: #e8f2ff; color: #1769c2; border: 1px solid #b9d7ff; border-radius: 14px; padding: 7px 13px; font-weight: 700; }
QLabel#hint { color: #6e7b94; font-size: 12px; }
QTabWidget::pane { border: 1px solid #dbe3ef; background: white; border-radius: 10px; top: -1px; }
QTabBar::tab { padding: 10px 22px; margin-right: 4px; background: #e8edf5; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: white; color: #1564c0; font-weight: 700; }
QGroupBox { background: white; border: 1px solid #dbe3ef; border-radius: 10px; margin-top: 12px; padding: 14px 10px 10px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #25395f; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: #fbfcfe; border: 1px solid #cfd9e8; border-radius: 6px; padding: 6px; selection-background-color: #2d7bd5; }
QPushButton { background: #eef3fa; border: 1px solid #c9d5e6; border-radius: 7px; padding: 8px 13px; font-weight: 600; }
QPushButton:hover { background: #e1ebf8; }
QPushButton:pressed { background: #d1e1f5; }
QPushButton#primaryButton { background: #1769c2; color: white; border: none; }
QPushButton#directButton, QPushButton#translateButton { min-height: 92px; color: white; font-size: 19px; font-weight: 800; border: none; border-radius: 12px; }
QPushButton#directButton { background: #264d73; }
QPushButton#translateButton { background: #1769c2; }
QPushButton#directButton[active="true"], QPushButton#translateButton[active="true"] { background: #e05a47; }
QPushButton#teacherButton[active="true"] { background: #13835f; color: white; border-color: #0e6d4e; }
QPushButton#recordButton[recording="true"] { background: #d84343; color: white; border-color: #bd3131; }
QHeaderView::section { background: #eef3fa; border: none; border-bottom: 1px solid #d8e0eb; padding: 7px; font-weight: 700; }
"""

DARK_STYLE_SHEET = """
QMainWindow, QWidget { background: #11151c; color: #e8edf6; font-family: "Microsoft YaHei UI"; font-size: 13px; }
QLabel#title { font-size: 28px; font-weight: 800; color: #f4f7ff; }
QLabel#subtitle { color: #9da9bc; font-size: 13px; }
QLabel#statusBadge { background: #17365c; color: #8dc4ff; border: 1px solid #285b91; border-radius: 14px; padding: 7px 13px; font-weight: 700; }
QLabel#hint { color: #9aa7bb; font-size: 12px; }
QTabWidget::pane { border: 1px solid #303948; background: #171c24; border-radius: 10px; top: -1px; }
QTabBar::tab { padding: 10px 22px; margin-right: 4px; background: #242b36; color: #aeb9ca; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #171c24; color: #74b7ff; font-weight: 700; }
QGroupBox { background: #191f28; border: 1px solid #333d4d; border-radius: 10px; margin-top: 12px; padding: 14px 10px 10px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #dce7f8; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: #10151c; color: #edf2fa; border: 1px solid #3c485a; border-radius: 6px; padding: 6px; selection-background-color: #286fb8; }
QComboBox QAbstractItemView { background: #171d26; color: #edf2fa; selection-background-color: #286fb8; }
QPushButton { background: #283140; color: #edf2fa; border: 1px solid #46546a; border-radius: 7px; padding: 8px 13px; font-weight: 600; }
QPushButton:hover { background: #344156; }
QPushButton:pressed { background: #202937; }
QPushButton#primaryButton { background: #1971ca; color: white; border: none; }
QPushButton#directButton, QPushButton#translateButton { min-height: 92px; color: white; font-size: 19px; font-weight: 800; border: none; border-radius: 12px; }
QPushButton#directButton { background: #315b82; }
QPushButton#translateButton { background: #1971ca; }
QPushButton#directButton[active="true"], QPushButton#translateButton[active="true"] { background: #d95649; }
QPushButton#teacherButton[active="true"] { background: #167c5d; color: white; border-color: #27a67e; }
QPushButton#recordButton[recording="true"] { background: #c43f3f; color: white; border-color: #e05a5a; }
QHeaderView::section { background: #252e3b; color: #e5ecf7; border: none; border-bottom: 1px solid #3b4657; padding: 7px; font-weight: 700; }
QScrollBar:vertical { background: #161c24; width: 12px; }
QScrollBar::handle:vertical { background: #485568; min-height: 28px; border-radius: 5px; }
"""


def configure_logging(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(base_dir / "teams-voice-translator.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run() -> int:
    store = SettingsStore()
    configure_logging(store.base_dir)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Teams Voice Translator")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    return app.exec()
