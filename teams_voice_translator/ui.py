from __future__ import annotations

import base64
import logging
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, QSignalBlocker, Signal
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import qtawesome as qta

from .aliyun import ApiError, BailianClient, FunASRRealtime, QwenRealtimeASR, create_realtime_asr
from .backdrops import (
    BACKDROP_NONE,
    DEFAULT_BACKDROP,
    backdrop_credit,
    backdrop_options,
    backdrop_path,
    resolve_backdrop,
)
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
from .longform import (
    BilingualSegment,
    LongFormReaderDialog,
    estimate_speech_seconds,
    parse_pronunciation_dictionary,
    split_source_sentences,
)
from .oss_upload import OssTemporaryUploader, UploadedVoiceSample
from .records import SubtitleSession, default_output_directory
from .profiles import CourseProfileStore
from .api_payloads import (
    COSYVOICE_V3_5_PLUS_MODEL,
    COSYVOICE_V3_FLASH_MODEL,
    FUN_ASR_REALTIME_MODEL,
    QWEN3_TTS_VC_HTTP_MODEL,
    QWEN3_TTS_VC_REALTIME_MODEL,
    is_cosyvoice_model,
    is_qwen3_tts_vc_model,
    is_qwen3_tts_vc_realtime_model,
    parse_json_list,
)
from .voice_sample import SUPPORTED_AUDIO_SUFFIXES, normalized_voice_sample
from .voicestudio import VoiceStudioClient, VoiceStudioEngine, VoiceStudioVoice
from .voicestudio_manager import (
    GITHUB_RELEASES_PAGE,
    VoiceStudioManager,
    VoiceStudioManagerError,
    VoiceStudioRelease,
    VoiceStudioRuntime,
    VoiceStudioTaskCancelled,
)
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
    continuous_finished = Signal(str)
    long_text_progress = Signal(int, int)
    long_text_state = Signal(str)
    long_form_progress = Signal(int, str)
    long_form_ready = Signal(object, object)
    voicestudio_catalog = Signal(bool, object, object, str)
    voicestudio_runtime = Signal(object)
    voicestudio_task_progress = Signal(str, int)
    voicestudio_task_finished = Signal(bool, str, object)
    voicestudio_shutdown_progress = Signal(str)
    voicestudio_shutdown_finished = Signal(str)


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


class ScrollSafeComboBox(QComboBox):
    """Let the settings page scroll without changing a hovered selection."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class ScrollSafeSpinBox(QSpinBox):
    """Ignore wheel edits; values remain editable with click/keyboard."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class ScrollSafeDoubleSpinBox(QDoubleSpinBox):
    """Ignore wheel edits; values remain editable with click/keyboard."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class ElidedLabel(QLabel):
    """Single-line label that keeps the full value in its tooltip."""

    def __init__(self, text: str = "") -> None:
        self._full_text = text
        super().__init__(text)
        self.setToolTip(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._refresh_elision()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_elision()

    def _refresh_elision(self) -> None:
        available = max(20, self.width() - 10)
        QLabel.setText(
            self,
            self.fontMetrics().elidedText(self._full_text, Qt.ElideRight, available),
        )


class ArtworkLabel(QLabel):
    """Crop a local-only backdrop image to a tall panel without distortion."""

    def __init__(self, image_path: Path | None = None) -> None:
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        # Narrow layouts host the same panel, so keep the floor small enough
        # for the compact workspaces.
        self.setMinimumWidth(140)
        self.setObjectName("artwork")
        if image_path is not None:
            self.set_backdrop(image_path)

    def set_backdrop(self, image_path: Path | None) -> bool:
        """Swap the displayed image.

        Returns False and clears the label when the file is missing or
        unreadable, so callers can fall back to placeholder text instead of
        showing an empty box on machines without the private artwork.
        """
        if image_path is None or not Path(image_path).is_file():
            self._source = QPixmap()
            self.clear()
            return False
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._source = QPixmap()
            self.clear()
            return False
        self._source = pixmap
        self._render()
        return True

    def _render(self) -> None:
        if self._source.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        source = self._source
        wanted_ratio = self.width() / self.height()
        crop_width = max(1, min(source.width(), int(source.height() * wanted_ratio)))
        cropped = source.copy(0, 0, crop_width, source.height())
        self.setPixmap(
            cropped.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


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
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("hint")
        layout.addWidget(self.note)
        form = QFormLayout()
        self.model = QComboBox()
        self.model.addItems([
            QWEN3_TTS_VC_REALTIME_MODEL,
            QWEN3_TTS_VC_HTTP_MODEL,
            COSYVOICE_V3_5_PLUS_MODEL,
            COSYVOICE_V3_FLASH_MODEL,
            "qwen-audio-3.0-tts-plus",
            "qwen-audio-3.0-tts-flash",
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
        self.oss_settings_button = QPushButton("OSS 设置")
        self.oss_settings_button.clicked.connect(self.open_oss_settings)
        source_row.addWidget(self.oss_settings_button)
        self.language = QComboBox()
        self.language.addItem("中文 zh", "zh")
        self.language.addItem("英语 en", "en")
        self.max_seconds = QDoubleSpinBox()
        self.max_seconds.setRange(5.0, 30.0)
        self.max_seconds.setValue(20.0)
        self.max_seconds.setSuffix(" 秒")
        self.preprocess = QCheckBox("开启降噪、增强和音量归一化")
        self.transcript = QLineEdit()
        self.transcript.setPlaceholderText("可留空；准确填写录音原文可进一步提高 Qwen3 克隆质量")
        self.model.currentTextChanged.connect(self.refresh_model_options)
        form.addRow("目标模型", self.model)
        form.addRow("音色名称/前缀", self.prefix)
        form.addRow("样音文件", source_row)
        form.addRow("样音语言", self.language)
        form.addRow("样音原文（可空）", self.transcript)
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
        model = self.model.currentText()
        qwen3 = is_qwen3_tts_vc_model(model)
        supported = model in {
            "qwen-audio-3.0-tts-flash",
            "qwen-audio-3.0-tts-plus",
        }
        self.preprocess.setEnabled(supported)
        self.transcript.setEnabled(qwen3)
        self.oss_settings_button.setEnabled(not qwen3)
        if not supported:
            self.preprocess.setChecked(False)
            self.preprocess.setToolTip("该模型不支持旧版服务端预处理；本地格式标准化仍会自动执行。")
        else:
            self.preprocess.setToolTip("")
        if qwen3:
            mode = "低延迟会议" if is_qwen3_tts_vc_realtime_model(model) else "高清自然打字发声"
            self.note.setText(
                f"当前创建 Qwen3-TTS-VC {mode}专属音色。直接选择 iPhone M4A、MP3 或 WAV；"
                "程序会转成标准 WAV 并以 Base64 直传百炼，不需要 OSS，也不会留下临时云端样音。"
            )
        else:
            self.note.setText(
                "直接选择 iPhone M4A、MP3、WAV 等本地录音即可。程序会自动裁剪并转换成 "
                "24 kHz 单声道 16-bit PCM WAV；旧版模型会临时上传到你的 OSS，完成后删除。"
            )
        if hasattr(self, "oss_status"):
            self.refresh_oss_status()

    def open_oss_settings(self) -> None:
        self.configure_oss()
        self.refresh_oss_status()

    def refresh_oss_status(self) -> None:
        if is_qwen3_tts_vc_model(self.model.currentText()):
            self.oss_status.setText("Qwen3 声音复刻使用官方 Base64 直传，本次不读取 OSS 配置。")
            return
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
            if (
                not is_qwen3_tts_vc_model(self.model.currentText())
                and not self.settings.has_oss_config()
                and not self.configure_oss()
            ):
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
                "transcript": self.transcript.text().strip(),
            }
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = SettingsStore()
        self.signals = UiSignals()
        self.voicestudio_manager = VoiceStudioManager()
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
        self.live_translate_session: QwenLiveTranslate | None = None
        self.live_translate_session_key: tuple[Any, ...] | None = None
        self.live_translate_session_lock = threading.Lock()
        self.continuous_stop_event = threading.Event()
        self.direct_key_held = False
        self.switch_to_direct_after_continuous = False
        self.cancel_event = threading.Event()
        self.long_text_thread: threading.Thread | None = None
        self.long_form_dialog: LongFormReaderDialog | None = None
        self.long_form_prepare_dialog: QProgressDialog | None = None
        self.long_form_prepare_cancel_event = threading.Event()
        self.long_form_prepare_generation = 0
        self.state_lock = threading.Lock()
        self.state = "idle"
        self.last_translation = ""
        self.awaiting_confirmation = False
        self.clone_dialog: VoiceCloneDialog | None = None
        self.clone_target_model = ""
        self.voicestudio_voices: list[VoiceStudioVoice] = []
        self.voicestudio_engines: list[VoiceStudioEngine] = []
        self._voicestudio_runtime_refreshing = False
        self._voicestudio_task_running = False
        self._voicestudio_task_cancellable = False
        self._voicestudio_task_cancel_event = threading.Event()
        self._voicestudio_task_success: Callable[[object], None] | None = None
        self._voicestudio_task_success_message = ""
        self._shutdown_in_progress = False
        self._shutdown_authorized = False
        self._shutdown_dialog: QProgressDialog | None = None
        self._subtitle_close_checked = False
        self.hotkeys: HoldHotkeys | None = None
        self.recorder = DualTrackRecorder()
        self.recording_started_at: float | None = None
        self.subtitle_session = SubtitleSession()
        self.subtitle_exported = True
        self.profile_store = CourseProfileStore(self.settings.base_dir)
        self._build_ui()
        self._active_tts_model = self.tts_model.currentText()
        self.overlay = SubtitleOverlay()
        self._connect_signals()
        self.overlay.geometry_saved.connect(self._save_overlay_geometry)
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(500)
        self.record_timer.timeout.connect(self._update_recording_status)
        self.voicestudio_monitor_timer = QTimer(self)
        self.voicestudio_monitor_timer.setInterval(2500)
        self.voicestudio_monitor_timer.timeout.connect(self.refresh_voicestudio_runtime)
        self.refresh_audio_devices()
        self.load_settings_into_ui()
        self._sync_voicestudio_backend_switch()
        self.start_hotkeys()
        self.voicestudio_monitor_timer.start()
        self.setWindowTitle(f"Teams 双向课堂翻译 v{__version__}")
        self.setMinimumSize(1020, 720)
        self.apply_layout()
        self._set_status(self._ready_status())
        QTimer.singleShot(300, self.offer_cache_recovery)
        self._voicestudio_refresh_timer = QTimer(self)
        self._voicestudio_refresh_timer.setSingleShot(True)
        self._voicestudio_refresh_timer.timeout.connect(lambda: self.refresh_voicestudio_runtime(include_latest=True))
        self._voicestudio_refresh_timer.start(650)
        if self.settings.get("tts_provider") == "voicestudio":
            QTimer.singleShot(900, self.refresh_voicestudio_catalog)
        if self.settings.get("auto_start_teacher_caption"):
            QTimer.singleShot(700, self.start_teacher_caption)

    @staticmethod
    def _add_theme_options(combo: QComboBox) -> None:
        combo.addItem("冰川水晶蓝", "shizuku")
        combo.addItem("午夜蓝黑", "dark")
        combo.addItem("琥珀暖白", "warm")
        combo.addItem("Fluent 云白", "light")

    @staticmethod
    def _add_layout_options(combo: QComboBox) -> None:
        combo.addItem("水晶极光", "crystal")
        combo.addItem("午夜工作台", "signal")
        combo.addItem("暖色工作室", "studio")
        combo.addItem("Fluent 控制台", "fluent")

    @staticmethod
    def _add_backdrop_options(combo: QComboBox) -> None:
        """Fill a backdrop selector. Labels and order come from backdrops.py."""
        for label, key in backdrop_options():
            combo.addItem(label, key)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        self.app_root = central
        root = QVBoxLayout(central)
        self.root_layout = root
        root.setContentsMargins(14, 10, 14, 12)
        root.setSpacing(7)
        self.header_frame = QFrame()
        self.header_frame.setObjectName("appHeader")
        header = QHBoxLayout(self.header_frame)
        self.header_layout = header
        header.setContentsMargins(8, 4, 8, 4)
        header.setSpacing(10)
        logo_path = Path(__file__).parent / "local_assets" / "shizuku_logo.png"
        self.brand_logo = QLabel()
        self.brand_logo.setObjectName("brandLogo")
        self.brand_logo.setFixedSize(62, 62)
        logo = QPixmap(str(logo_path))
        if not logo.isNull():
            self.brand_logo.setPixmap(
                logo.scaled(self.brand_logo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        header.addWidget(self.brand_logo)
        self.header_title_container = QWidget()
        self.header_title_container.setObjectName("headerTitleContainer")
        self.header_title_container.setMinimumWidth(470)
        self.header_title_container.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        title_box = QVBoxLayout(self.header_title_container)
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        self.title_label = QLabel(f"Teams 双向课堂翻译 v{__version__}")
        self.title_label.setObjectName("title")
        self.subtitle_label = QLabel("VOICE WORKSTATION  ·  中英双向翻译、克隆音色与课堂记录")
        self.subtitle_label.setObjectName("subtitle")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        header.addWidget(self.header_title_container, 1)
        header.addStretch()
        self.layout_quick = ScrollSafeComboBox()
        self.layout_quick.setObjectName("layoutQuick")
        self.layout_quick.setFixedWidth(124)
        self.layout_quick.setToolTip("界面布局：可与任意配色自由组合")
        self._add_layout_options(self.layout_quick)
        header.addWidget(self.layout_quick)
        self.theme_quick = ScrollSafeComboBox()
        self.theme_quick.setObjectName("themeQuick")
        self.theme_quick.setFixedWidth(124)
        self.theme_quick.setToolTip("界面配色：不会改变当前布局")
        self._add_theme_options(self.theme_quick)
        header.addWidget(self.theme_quick)
        self.backdrop_quick = ScrollSafeComboBox()
        self.backdrop_quick.setObjectName("backdropQuick")
        self.backdrop_quick.setFixedWidth(132)
        self.backdrop_quick.setToolTip("侧边栏背景立绘：与布局和配色完全独立")
        self._add_backdrop_options(self.backdrop_quick)
        header.addWidget(self.backdrop_quick)
        self.engine_scope_button = QPushButton("☁ 云端引擎")
        self.engine_scope_button.setObjectName("engineScopeButton")
        self.engine_scope_button.setFixedWidth(112)
        self.engine_scope_button.setToolTip("切换语音输出后端：阿里云百炼 / VoiceStudio 本地服务")
        header.addWidget(self.engine_scope_button)
        self.status_badge = ElidedLabel("初始化…")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setMinimumWidth(170)
        self.status_badge.setMaximumWidth(230)
        header.addWidget(self.status_badge)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainPages")
        self.meeting_tab = self._build_meeting_tab()
        self.settings_tab = self._build_settings_tab()
        self.voicestudio_tab = self._build_voicestudio_tab()
        self.tabs.addTab(
            self.meeting_tab,
            qta.icon("fa5s.headset", color="#1688d4"),
            "会议控制台",
        )
        self.tabs.addTab(
            self.voicestudio_tab,
            qta.icon("fa5s.wave-square", color="#1688d4"),
            "本地声音工作台",
        )
        self.tabs.addTab(
            self.settings_tab,
            qta.icon("fa5s.cog", color="#52718f"),
            "设置",
        )

        self.main_navigation = QFrame()
        self.main_navigation.setObjectName("mainNavigation")
        self.main_navigation.setMinimumWidth(188)
        self.main_navigation.setMaximumWidth(230)
        main_nav_layout = QVBoxLayout(self.main_navigation)
        main_nav_layout.setContentsMargins(12, 18, 12, 14)
        main_nav_layout.setSpacing(10)
        self.main_nav_brand = QLabel("MIDNIGHT\nSIGNAL")
        self.main_nav_brand.setObjectName("mainNavBrand")
        main_nav_layout.addWidget(self.main_nav_brand)
        main_nav_layout.addSpacing(18)
        self.main_nav_buttons: list[QPushButton] = []
        for index, text in enumerate(
            ("🎧  会议控制台", "🎙  本地声音工作台", "⚙  设置")
        ):
            button = QPushButton(text)
            button.setObjectName("mainNavButton")
            button.setCheckable(True)
            button.setMinimumHeight(54)
            button.clicked.connect(lambda checked=False, page=index: self._set_main_page(page))
            main_nav_layout.addWidget(button)
            self.main_nav_buttons.append(button)
        main_nav_layout.addStretch()
        self.main_nav_health = QLabel("●  本地服务\n    正在检测")
        self.main_nav_health.setObjectName("mainNavHealth")
        main_nav_layout.addWidget(self.main_nav_health)

        self.app_footer = QFrame()
        self.app_footer.setObjectName("appFooter")
        footer_layout = QHBoxLayout(self.app_footer)
        footer_layout.setContentsMargins(10, 4, 10, 4)
        self.footer_status = QLabel("●  系统状态：就绪")
        self.footer_status.setObjectName("footerStatus")
        footer_layout.addWidget(self.footer_status)
        footer_layout.addStretch()
        self.footer_version = QLabel(f"v{__version__}  ·  LOCAL VOICE WORKSTATION")
        self.footer_version.setObjectName("hint")
        footer_layout.addWidget(self.footer_version)

        self.main_shell = QWidget()
        self.main_shell.setObjectName("mainShell")
        self.main_shell_grid = QGridLayout(self.main_shell)
        self.main_shell_grid.setContentsMargins(0, 0, 0, 0)
        self.main_shell_grid.setSpacing(7)
        root.addWidget(self.main_shell, 1)
        root.addWidget(self.app_footer)
        self.tabs.currentChanged.connect(self._sync_main_navigation)
        self._arrange_app_shell("crystal")
        self.setCentralWidget(central)
        self.setStyleSheet(SHIZUKU_STYLE_SHEET)

    def _set_main_page(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)
        self._sync_main_navigation(index)

    def _sync_main_navigation(self, index: int) -> None:
        for button_index, button in enumerate(self.main_nav_buttons):
            button.setChecked(button_index == index)

    def _arrange_app_header(self, layout_name: str) -> None:
        header = self.header_layout
        self._take_all_layout_items(header)
        # Layout/palette/backdrop are edited on the Settings page. Once a
        # combo is removed from this layout it must also be hidden, otherwise
        # Qt leaves it at its stale geometry and it floats over the title.
        self.layout_quick.hide()
        self.theme_quick.hide()
        self.backdrop_quick.hide()
        crystal = layout_name == "crystal"
        self.brand_logo.setVisible(crystal)
        self.header_title_container.setVisible(crystal)
        if crystal:
            header.addWidget(self.brand_logo)
            header.addWidget(self.header_title_container, 1)
            header.addStretch()
        elif layout_name in {"fluent", "signal"}:
            header.addWidget(self.status_badge)
            header.addStretch()
        else:
            header.addStretch()
        header.addWidget(self.engine_scope_button)
        if layout_name not in {"fluent", "signal"}:
            header.addWidget(self.status_badge)

    def _arrange_app_shell(self, layout_name: str) -> None:
        """Rebuild the whole-window navigation shell for each reference UI."""
        if not hasattr(self, "main_shell_grid"):
            return
        grid = self.main_shell_grid
        self._take_all_layout_items(grid)
        for index in range(4):
            grid.setColumnStretch(index, 0)
            grid.setRowStretch(index, 0)
            grid.setColumnMinimumWidth(index, 0)

        sidebar_mode = layout_name in {"signal", "fluent"}
        self.main_navigation.setVisible(sidebar_mode)
        self.tabs.tabBar().setVisible(not sidebar_mode)
        self.header_frame.setVisible(True)
        self.header_frame.setMinimumHeight(94 if layout_name == "crystal" else 44)
        self.header_frame.setMaximumHeight(104 if layout_name == "crystal" else 48)
        self.app_footer.setVisible(layout_name == "studio")
        self.main_nav_brand.setVisible(layout_name == "signal")
        self.main_nav_brand.setText("MIDNIGHT\nSIGNAL" if layout_name == "signal" else "VOICE\nWORKSTATION")
        self.main_navigation.setMinimumWidth(250 if layout_name == "signal" else 210)
        self.main_navigation.setMaximumWidth(265 if layout_name == "signal" else 230)
        self._sync_main_navigation(self.tabs.currentIndex())
        self._arrange_app_header(layout_name)

        if layout_name == "crystal":
            self.brand_logo.setFixedSize(62, 62)
            self.header_layout.setContentsMargins(18, 8, 18, 6)
            self.engine_scope_button.setFixedWidth(118)
            self.status_badge.setMinimumWidth(188)
        else:
            self.brand_logo.setFixedSize(38, 38)
            self.header_layout.setContentsMargins(8, 4, 8, 4)
            self.engine_scope_button.setFixedWidth(112)
            self.status_badge.setMinimumWidth(170)

        grid.addWidget(self.header_frame, 0, 0, 1, 2)
        if sidebar_mode:
            grid.addWidget(self.main_navigation, 1, 0)
            grid.addWidget(self.tabs, 1, 1)
            grid.setColumnStretch(1, 1)
        else:
            grid.addWidget(self.tabs, 1, 0, 1, 2)
            grid.setColumnStretch(0, 1)
        grid.setRowStretch(1, 1)

    def _build_voicestudio_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(9)

        hero = QGroupBox("VoiceStudio · 本地化 ElevenLabs 替代后端")
        hero.setObjectName("voiceStudioHero")
        hero_layout = QVBoxLayout(hero)
        hero_top = QHBoxLayout()
        intro = QLabel(
            "声音克隆与模型运行留在本机；本软件负责课堂翻译、Teams 路由、字幕、重播和整段朗读。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("voiceStudioIntro")
        hero_top.addWidget(intro, 1)
        self.voicestudio_status = QLabel("尚未检测")
        self.voicestudio_status.setObjectName("voiceStudioStatus")
        hero_top.addWidget(self.voicestudio_status)
        hero_layout.addLayout(hero_top)

        self.voicestudio_hero = hero
        hero.hide()
        root.addWidget(hero)

        self.voicestudio_inner_tabs = QTabWidget()
        self.voicestudio_inner_tabs.setObjectName("voiceStudioInnerTabs")
        manager_page = QWidget()
        manager_page_layout = QVBoxLayout(manager_page)
        manager_page_layout.setContentsMargins(0, 0, 0, 0)
        self.voicestudio_manager_scroll = QScrollArea()
        self.voicestudio_manager_scroll.setObjectName("voiceStudioManagerScroll")
        self.voicestudio_manager_scroll.setWidgetResizable(True)
        self.voicestudio_manager_scroll.setFrameShape(QFrame.NoFrame)
        self.voicestudio_manager_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        manager_content = QWidget()
        manager_content.setObjectName("voiceStudioManagerContent")
        manager_layout = QVBoxLayout(manager_content)
        manager_layout.setSizeConstraint(QLayout.SetMinimumSize)
        manager_layout.setContentsMargins(4, 6, 4, 4)
        manager_layout.setSpacing(8)

        connection = QGroupBox("安装位置与服务连接")
        connection.setObjectName("voiceStudioConnectionCard")
        self.voicestudio_connection_card = connection
        connection_layout = QFormLayout(connection)
        self.voicestudio_install_dir = QLineEdit()
        self.voicestudio_install_dir.setPlaceholderText(r"例如 D:\Apps\VoiceStudio；留空使用 MSI 默认位置")
        install_path_row = QHBoxLayout()
        install_path_row.addWidget(self.voicestudio_install_dir, 1)
        self.voicestudio_choose_install_dir_button = QPushButton("选择安装目录…")
        self.voicestudio_choose_install_dir_button.setMinimumWidth(190)
        self.voicestudio_choose_install_dir_button.setToolTip("选择 VoiceStudio 的程序安装目录")
        install_path_row.addWidget(self.voicestudio_choose_install_dir_button)
        connection_layout.addRow("安装路径", install_path_row)

        self.voicestudio_executable = QLineEdit()
        self.voicestudio_executable.setPlaceholderText("自动检测 VoiceStudio.exe，也可手动指定")
        exe_row = QHBoxLayout()
        exe_row.addWidget(self.voicestudio_executable, 1)
        self.voicestudio_choose_exe_button = QPushButton("选择程序…")
        self.voicestudio_choose_exe_button.setMinimumWidth(190)
        self.voicestudio_choose_exe_button.setToolTip("手动选择 VoiceStudio.exe")
        exe_row.addWidget(self.voicestudio_choose_exe_button)
        connection_layout.addRow("桌面程序", exe_row)

        self.voicestudio_url = QLineEdit()
        self.voicestudio_url.setPlaceholderText("http://127.0.0.1:3900")
        url_row = QHBoxLayout()
        url_row.addWidget(self.voicestudio_url, 1)
        self.voicestudio_refresh_button = QPushButton("刷新状态与声音库")
        self.voicestudio_refresh_button.setMinimumWidth(190)
        self.voicestudio_refresh_button.setToolTip("刷新本地 API、版本、进程与声音库")
        url_row.addWidget(self.voicestudio_refresh_button)
        connection_layout.addRow("本地服务", url_row)
        for control in (
            self.voicestudio_install_dir,
            self.voicestudio_executable,
            self.voicestudio_url,
            self.voicestudio_choose_install_dir_button,
            self.voicestudio_choose_exe_button,
            self.voicestudio_refresh_button,
        ):
            control.setMinimumHeight(38)
        manager_layout.addWidget(connection)

        versions = QGroupBox("版本与运行状态")
        versions.setObjectName("voiceStudioVersionsCard")
        self.voicestudio_versions_card = versions
        versions_layout = QGridLayout(versions)
        self.voicestudio_versions_layout = versions_layout
        self.voicestudio_installed_title = QLabel("已安装版本")
        self.voicestudio_installed_version = QLabel("未检测")
        self.voicestudio_installed_version.setObjectName("versionValue")
        self.voicestudio_api_title = QLabel("本地 API")
        self.voicestudio_api_version = QLabel("未连接")
        self.voicestudio_api_version.setObjectName("versionValue")
        self.voicestudio_latest_title = QLabel("GitHub 最新版")
        self.voicestudio_latest_version = QLabel("正在查询…")
        self.voicestudio_latest_version.setObjectName("versionValue")
        self.voicestudio_installed_meta = QLabel("等待检测本地安装")
        self.voicestudio_api_meta = QLabel("服务端点 http://127.0.0.1:3900")
        self.voicestudio_latest_meta = QLabel("联网后自动检查更新")
        self.voicestudio_latest_meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.voicestudio_latest_meta.setToolTip("联网后显示 GitHub 官方发布文件的完整 SHA-256，可用鼠标选中复制。")
        for meta in (
            self.voicestudio_installed_meta,
            self.voicestudio_api_meta,
            self.voicestudio_latest_meta,
        ):
            meta.setObjectName("versionMeta")
        self.voicestudio_version_cards: list[QFrame] = []
        for title, value, meta in (
            (self.voicestudio_installed_title, self.voicestudio_installed_version, self.voicestudio_installed_meta),
            (self.voicestudio_api_title, self.voicestudio_api_version, self.voicestudio_api_meta),
            (self.voicestudio_latest_title, self.voicestudio_latest_version, self.voicestudio_latest_meta),
        ):
            card = QFrame()
            card.setObjectName("versionFactCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(5)
            title.setObjectName("versionFactTitle")
            # QLabel inherits QWidget's page background in Qt stylesheets.
            # Keeping these children explicitly transparent prevents the
            # unwanted horizontal bands previously visible in each card.
            for label in (title, value, meta):
                label.setAutoFillBackground(False)
            card_layout.addWidget(title)
            card_layout.addWidget(value)
            card_layout.addWidget(meta)
            card_layout.addStretch()
            self.voicestudio_version_cards.append(card)
        self.voicestudio_voice_count_card = QFrame()
        self.voicestudio_voice_count_card.setObjectName("versionFactCard")
        voice_count_layout = QVBoxLayout(self.voicestudio_voice_count_card)
        voice_count_layout.setContentsMargins(12, 10, 12, 10)
        voice_count_title = QLabel("本地声音数量")
        voice_count_title.setObjectName("versionFactTitle")
        self.voicestudio_voice_count = QLabel("0")
        self.voicestudio_voice_count.setObjectName("versionValue")
        voice_count_layout.addWidget(voice_count_title)
        voice_count_layout.addWidget(self.voicestudio_voice_count)
        voice_count_layout.addStretch()
        self._arrange_voicestudio_versions("crystal")
        manager_layout.addWidget(versions)

        actions = QGroupBox("VoiceStudio 生命周期管理")
        actions.setObjectName("voiceStudioActionsCard")
        self.voicestudio_actions_card = actions
        actions_layout = QVBoxLayout(actions)
        self.voicestudio_actions_layout = actions_layout
        action_grid = QGridLayout()
        self.voicestudio_action_grid = action_grid
        action_grid.setHorizontalSpacing(9)
        action_grid.setVerticalSpacing(9)
        self.voicestudio_install_button = QToolButton()
        self.voicestudio_install_button.setObjectName("voiceStudioPrimary")
        self.voicestudio_import_button = QToolButton()
        self.voicestudio_download_button = QToolButton()
        self.voicestudio_upgrade_button = QToolButton()
        self.voicestudio_start_button = QToolButton()
        self.voicestudio_stop_button = QToolButton()
        self.voicestudio_restart_button = QToolButton()
        self.voicestudio_uninstall_button = QToolButton()
        self.voicestudio_uninstall_button.setObjectName("dangerButton")
        # Segmented backend switch: pick the active TTS backend right here
        # instead of a one-way "set VoiceStudio as backend" button.
        self.voicestudio_backend_switch = QFrame()
        self.voicestudio_backend_switch.setObjectName("voiceStudioBackendSwitch")
        backend_switch_layout = QHBoxLayout(self.voicestudio_backend_switch)
        backend_switch_layout.setContentsMargins(3, 3, 3, 3)
        backend_switch_layout.setSpacing(2)
        self.voicestudio_backend_group = QButtonGroup(self)
        self.voicestudio_backend_group.setExclusive(True)
        self.voicestudio_use_aliyun_button = QPushButton("百炼云语音")
        self.voicestudio_use_aliyun_button.setCheckable(True)
        self.voicestudio_use_aliyun_button.setProperty("backendSegment", True)
        self.voicestudio_use_aliyun_button.setIcon(qta.icon("ph.cloud", color="#4a6b8a"))
        self.voicestudio_use_button = QPushButton("VoiceStudio 本地")
        self.voicestudio_use_button.setCheckable(True)
        self.voicestudio_use_button.setProperty("backendSegment", True)
        self.voicestudio_use_button.setIcon(qta.icon("ph.hard-drives", color="#4a6b8a"))
        self.voicestudio_backend_group.addButton(self.voicestudio_use_aliyun_button)
        self.voicestudio_backend_group.addButton(self.voicestudio_use_button)
        backend_switch_layout.addWidget(self.voicestudio_use_aliyun_button)
        backend_switch_layout.addWidget(self.voicestudio_use_button)
        self.voicestudio_docs_button = QPushButton("本地 API 文档")
        lifecycle_buttons = (
            self.voicestudio_install_button,
            self.voicestudio_import_button,
            self.voicestudio_download_button,
            self.voicestudio_start_button,
            self.voicestudio_stop_button,
            self.voicestudio_restart_button,
            self.voicestudio_upgrade_button,
            self.voicestudio_uninstall_button,
        )
        self.voicestudio_lifecycle_buttons = lifecycle_buttons
        # One coherent Phosphor outline family replaces the former mixture of
        # heavy Font Awesome glyphs. The restrained per-action accents match
        # the Crystal dashboard while remaining crisp at any DPI.
        self.voicestudio_tile_icons = (
            ("ph.package", "#2878e8"),
            ("ph.file-arrow-up", "#2c9b72"),
            ("ph.github-logo", "#243b63"),
            ("ph.play-circle", "#35a36f"),
            ("ph.stop-circle", "#e05b62"),
            ("ph.arrow-clockwise", "#e6942d"),
            ("ph.arrow-fat-line-up", "#8a62d4"),
            ("ph.trash", "#df514b"),
        )
        for index, button in enumerate(lifecycle_buttons):
            button.setMinimumHeight(42)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setArrowType(Qt.NoArrow)
            button.setAutoRaise(False)
            action_grid.addWidget(button, index // 4, index % 4)
        for column in range(4):
            action_grid.setColumnStretch(column, 1)
        actions_layout.addLayout(action_grid)

        utility_actions = QHBoxLayout()
        self.voicestudio_utility_actions = utility_actions
        self.voicestudio_use_aliyun_button.setMinimumHeight(32)
        self.voicestudio_use_button.setMinimumHeight(32)
        self.voicestudio_backend_switch.setMinimumHeight(38)
        utility_actions.addWidget(self.voicestudio_backend_switch)
        self.voicestudio_docs_button.setMinimumHeight(38)
        utility_actions.addWidget(self.voicestudio_docs_button)
        utility_actions.addStretch()
        actions_layout.addLayout(utility_actions)
        self.voicestudio_launch_button = self.voicestudio_start_button
        self.voicestudio_start_button.setEnabled(False)
        self.voicestudio_stop_button.setEnabled(False)
        self.voicestudio_restart_button.setEnabled(False)
        self.voicestudio_upgrade_button.setEnabled(False)
        self.voicestudio_uninstall_button.setEnabled(False)

        self.voicestudio_task_bar = QFrame()
        self.voicestudio_task_bar.setObjectName("voiceStudioTaskBar")
        task_bar_layout = QVBoxLayout(self.voicestudio_task_bar)
        task_bar_layout.setContentsMargins(11, 9, 11, 9)
        task_bar_layout.setSpacing(7)
        task_header = QHBoxLayout()
        task_header.setSpacing(8)
        self.voicestudio_progress_label = QLabel("就绪")
        self.voicestudio_progress_label.setObjectName("voiceStudioTaskLabel")
        self.voicestudio_progress_label.setWordWrap(True)
        self.voicestudio_progress_label.hide()
        task_header.addWidget(self.voicestudio_progress_label, 1)
        self.voicestudio_cancel_task_button = QPushButton("取消任务")
        self.voicestudio_cancel_task_button.setObjectName("voiceStudioCancelTask")
        self.voicestudio_cancel_task_button.setMinimumWidth(92)
        self.voicestudio_cancel_task_button.setMinimumHeight(32)
        task_header.addWidget(self.voicestudio_cancel_task_button)
        task_bar_layout.addLayout(task_header)
        self.voicestudio_progress = QProgressBar()
        self.voicestudio_progress.setRange(0, 100)
        self.voicestudio_progress.setValue(0)
        self.voicestudio_progress.setFormat("%p%")
        self.voicestudio_progress.setMinimumHeight(20)
        self.voicestudio_progress.hide()
        task_bar_layout.addWidget(self.voicestudio_progress)
        self.voicestudio_task_bar.hide()
        actions_layout.addWidget(self.voicestudio_task_bar)
        manager_layout.addWidget(actions)

        process_group = QGroupBox("后台进程信息 · 正常为绿色，异常为红色")
        process_group.setObjectName("voiceStudioProcessCard")
        self.voicestudio_process_card = process_group
        process_layout = QVBoxLayout(process_group)
        self.voicestudio_process_table = QTableWidget(0, 7)
        self.voicestudio_process_table.setHorizontalHeaderLabels(
            ["状态", "角色", "进程", "PID", "内存", "运行时间", "命令行"]
        )
        self.voicestudio_process_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.voicestudio_process_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.voicestudio_process_table.verticalHeader().setVisible(False)
        process_header = self.voicestudio_process_table.horizontalHeader()
        for column in (0, 1, 2, 3, 4, 5):
            process_header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        process_header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.voicestudio_process_table.setMinimumHeight(205)
        process_layout.addWidget(self.voicestudio_process_table)
        manager_layout.addWidget(process_group, 1)
        self.voicestudio_manager_scroll.setWidget(manager_content)
        manager_page_layout.addWidget(self.voicestudio_manager_scroll)

        voice_page = QWidget()
        workspace = QHBoxLayout(voice_page)
        workspace.setContentsMargins(4, 6, 4, 4)
        workspace.setSpacing(9)
        library = QGroupBox("本地声音库")
        library.setObjectName("voiceStudioLibraryCard")
        self.voicestudio_library_card = library
        library_layout = QGridLayout(library)
        self.voicestudio_library_layout = library_layout
        self.voicestudio_library_controls = QFrame()
        self.voicestudio_library_controls.setObjectName("voiceLibraryControls")
        selectors = QFormLayout(self.voicestudio_library_controls)
        self.voicestudio_library_selectors = selectors
        self.voicestudio_model = ScrollSafeComboBox()
        self.voicestudio_model.setEditable(True)
        self.voicestudio_model.addItem("tts-1")
        self.voicestudio_voice = ScrollSafeComboBox()
        self.voicestudio_voice.setEditable(True)
        self.voicestudio_voice.addItem("默认音色", "default")
        selectors.addRow("本地模型 / 引擎", self.voicestudio_model)
        selectors.addRow("声音档案", self.voicestudio_voice)
        self.voicestudio_library_quick_actions = QHBoxLayout()
        self.voicestudio_library_quick_actions.setSpacing(7)
        selectors.addRow("", self.voicestudio_library_quick_actions)
        self.voicestudio_voice_table = QTableWidget(0, 5)
        self.voicestudio_voice_table.setHorizontalHeaderLabels(
            ["名称", "声音 ID", "类型", "语言", "引擎"]
        )
        self.voicestudio_voice_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.voicestudio_voice_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.voicestudio_voice_table.verticalHeader().setVisible(False)
        header = self.voicestudio_voice_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        library_layout.addWidget(self.voicestudio_library_controls, 0, 0)
        library_layout.addWidget(self.voicestudio_voice_table, 1, 0)
        library_layout.setRowStretch(1, 1)
        workspace.addWidget(library, 3)

        preview = QGroupBox("本机试听与使用说明")
        preview.setObjectName("voiceStudioPreviewCard")
        self.voicestudio_preview_card = preview
        preview_layout = QVBoxLayout(preview)
        self.voicestudio_preview_text = QPlainTextEdit()
        self.voicestudio_preview_text.setPlaceholderText("输入一小段中文或英文，试听当前本地音色…")
        self.voicestudio_preview_text.setPlainText(
            "Hello, this is my private local voice for our online lesson."
        )
        self.voicestudio_preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.voicestudio_preview_text)
        preview_controls = QGridLayout()
        preview_controls.addWidget(QLabel("语速"), 0, 0)
        self.voicestudio_preview_rate = QSlider(Qt.Horizontal)
        self.voicestudio_preview_rate.setRange(50, 200)
        self.voicestudio_preview_rate.setValue(100)
        self.voicestudio_preview_rate_value = QLabel("1.00×")
        self.voicestudio_preview_rate_value.setMinimumWidth(46)
        self.voicestudio_preview_rate.valueChanged.connect(
            lambda value: self.voicestudio_preview_rate_value.setText(f"{value / 100:.2f}×")
        )
        preview_controls.addWidget(self.voicestudio_preview_rate, 0, 1)
        preview_controls.addWidget(self.voicestudio_preview_rate_value, 0, 2)
        preview_controls.addWidget(QLabel("音量"), 1, 0)
        self.voicestudio_preview_volume = QSlider(Qt.Horizontal)
        self.voicestudio_preview_volume.setRange(0, 100)
        self.voicestudio_preview_volume.setValue(80)
        self.voicestudio_preview_volume_value = QLabel("80%")
        self.voicestudio_preview_volume_value.setMinimumWidth(46)
        self.voicestudio_preview_volume.valueChanged.connect(
            lambda value: self.voicestudio_preview_volume_value.setText(f"{value}%")
        )
        preview_controls.addWidget(self.voicestudio_preview_volume, 1, 1)
        preview_controls.addWidget(self.voicestudio_preview_volume_value, 1, 2)
        preview_layout.addLayout(preview_controls)
        self.voicestudio_preview_button = QPushButton("▶ 仅在本机试听当前音色")
        self.voicestudio_preview_button.setObjectName("voiceStudioPrimary")
        self.voicestudio_preview_actions = QHBoxLayout()
        self.voicestudio_preview_actions.setSpacing(7)
        self.voicestudio_preview_actions.addWidget(self.voicestudio_preview_button)
        preview_layout.addLayout(self.voicestudio_preview_actions)
        notes = QLabel(
            "使用流程：\n"
            "1. 安装并启动 VoiceStudio，首次按它的向导下载本地模型。\n"
            "2. 在 VoiceStudio 中克隆/设计声音。\n"
            "3. 回到这里点“检测并同步”，选择音色，再设为当前语音后端。\n\n"
            "生效范围：传统 F9 翻译、键盘发声、历史重播和双语整段朗读。"
            "极速直译仍使用百炼的一体化实时音频通道。"
        )
        notes.setWordWrap(True)
        notes.setObjectName("hint")
        preview_layout.addWidget(notes)
        preview_layout.addStretch()
        workspace.addWidget(preview, 2)
        # Keep the former inner-tab object as a hidden compatibility surface
        # for saved state and older UI automation.  The visible workspace is a
        # real dashboard whose cards are rearranged into four distinct layouts.
        self.voicestudio_inner_tabs.addTab(manager_page, "运行管理")
        self.voicestudio_inner_tabs.addTab(voice_page, "声音库与试听")
        self.voicestudio_inner_tabs.setParent(tab)
        self.voicestudio_inner_tabs.hide()
        self._take_all_layout_items(manager_layout)
        self._take_all_layout_items(workspace)

        self.voicestudio_dashboard_scroll = QScrollArea()
        self.voicestudio_dashboard_scroll.setObjectName("voiceStudioDashboardScroll")
        self.voicestudio_dashboard_scroll.setWidgetResizable(True)
        self.voicestudio_dashboard_scroll.setFrameShape(QFrame.NoFrame)
        self.voicestudio_dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.voicestudio_dashboard_content = QWidget()
        self.voicestudio_dashboard_content.setObjectName("voiceStudioDashboardContent")
        self.voicestudio_dashboard_grid = QGridLayout(self.voicestudio_dashboard_content)
        self.voicestudio_dashboard_grid.setSizeConstraint(QLayout.SetMinimumSize)
        self.voicestudio_dashboard_grid.setContentsMargins(4, 4, 4, 4)
        self.voicestudio_dashboard_grid.setHorizontalSpacing(10)
        self.voicestudio_dashboard_grid.setVerticalSpacing(10)

        self.voicestudio_nav_card = QFrame()
        self.voicestudio_nav_card.setObjectName("voiceStudioNavCard")
        self.voicestudio_nav_card.setMinimumWidth(168)
        nav_layout = QVBoxLayout(self.voicestudio_nav_card)
        nav_layout.setContentsMargins(14, 18, 14, 14)
        nav_layout.setSpacing(10)
        self.voicestudio_nav_title = QLabel("LOCAL VOICE")
        self.voicestudio_nav_title.setObjectName("voiceStudioNavTitle")
        nav_layout.addWidget(self.voicestudio_nav_title)
        self.voicestudio_nav_runtime_button = QPushButton("  运行管理")
        self.voicestudio_nav_runtime_button.setObjectName("voiceStudioNavButton")
        self.voicestudio_nav_runtime_button.setIcon(qta.icon("fa5s.toolbox", color="#1688d4"))
        self.voicestudio_nav_runtime_button.setIconSize(QSize(20, 20))
        self.voicestudio_nav_runtime_button.setCheckable(True)
        self.voicestudio_nav_runtime_button.setChecked(True)
        self.voicestudio_nav_voices_button = QPushButton("  声音库与试听")
        self.voicestudio_nav_voices_button.setObjectName("voiceStudioNavButton")
        self.voicestudio_nav_voices_button.setIcon(qta.icon("fa5s.music", color="#1688d4"))
        self.voicestudio_nav_voices_button.setIconSize(QSize(20, 20))
        self.voicestudio_nav_voices_button.setCheckable(True)
        nav_layout.addWidget(self.voicestudio_nav_runtime_button)
        nav_layout.addWidget(self.voicestudio_nav_voices_button)
        nav_layout.addStretch()
        self.voicestudio_nav_health = QLabel("●  本地引擎\n    隐私优先 · 等待检测")
        self.voicestudio_nav_health.setObjectName("navHealthCard")
        nav_layout.addWidget(self.voicestudio_nav_health)
        self.voicestudio_nav_runtime_button.clicked.connect(
            lambda: self.voicestudio_dashboard_scroll.ensureWidgetVisible(
                self.voicestudio_connection_card, 16, 16
            )
        )
        self.voicestudio_nav_voices_button.clicked.connect(
            lambda: self.voicestudio_dashboard_scroll.ensureWidgetVisible(
                self.voicestudio_library_card, 16, 16
            )
        )
        self.voicestudio_nav_runtime_button.clicked.connect(
            lambda checked=False: self.voicestudio_nav_voices_button.setChecked(False)
        )
        self.voicestudio_nav_voices_button.clicked.connect(
            lambda checked=False: self.voicestudio_nav_runtime_button.setChecked(False)
        )

        self.voicestudio_page_heading = QFrame()
        self.voicestudio_page_heading.setObjectName("voiceStudioPageHeading")
        page_heading_layout = QVBoxLayout(self.voicestudio_page_heading)
        page_heading_layout.setContentsMargins(8, 4, 8, 8)
        page_heading_layout.setSpacing(2)
        self.voicestudio_page_title = QLabel("本地声音工作台  ·······")
        self.voicestudio_page_title.setObjectName("voiceStudioPageTitle")
        page_heading_layout.addWidget(self.voicestudio_page_title)
        self.voicestudio_page_subtitle = QLabel(
            "管理本地语音服务，提供稳定的声音合成、克隆与播放能力"
        )
        self.voicestudio_page_subtitle.setObjectName("hint")
        page_heading_layout.addWidget(self.voicestudio_page_subtitle)

        self.voicestudio_dashboard_scroll.setWidget(self.voicestudio_dashboard_content)
        root.addWidget(self.voicestudio_dashboard_scroll, 1)
        self._arrange_voicestudio_workspace("crystal")
        return tab

    def _build_meeting_tab(self) -> QWidget:
        tab = QWidget()
        shell = QHBoxLayout(tab)
        self.meeting_shell = shell
        shell.setContentsMargins(7, 7, 7, 7)
        shell.setSpacing(9)
        workspace = QWidget()
        self.meeting_workspace = workspace
        layout = QGridLayout(workspace)
        self.meeting_grid = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        # 当前句子：始终置顶，便于开会时快速扫一眼。
        self.transcript_panel = QWidget()
        self.transcript_panel.setObjectName("transcriptPanel")
        transcript_grid = QGridLayout(self.transcript_panel)
        transcript_grid.setContentsMargins(0, 0, 0, 0)
        transcript_grid.setSpacing(7)
        self.zh_group = QGroupBox("中文 · 你的原文 / 老师译文")
        self.zh_group.setObjectName("transcriptCard")
        zh_layout = QVBoxLayout(self.zh_group)
        self.chinese_text = QPlainTextEdit()
        self.chinese_text.setPlaceholderText("按住 F9，或开启持续翻译后直接说中文…")
        self.chinese_text.setReadOnly(True)
        self.chinese_text.setObjectName("currentText")
        self.chinese_text.setMinimumHeight(50)
        self.chinese_text.setMaximumHeight(64)
        zh_layout.addWidget(self.chinese_text)
        self.emotion_label = QLabel("识别情绪：—")
        self.emotion_label.setObjectName("hint")
        zh_layout.addWidget(self.emotion_label)
        self.en_group = QGroupBox("English · 你的译文 / 老师原文")
        self.en_group.setObjectName("transcriptCard")
        en_layout = QVBoxLayout(self.en_group)
        self.english_text = QPlainTextEdit()
        self.english_text.setPlaceholderText("松开 F9 后，英文译文会显示在这里…")
        self.english_text.setObjectName("currentText")
        self.english_text.setMinimumHeight(50)
        self.english_text.setMaximumHeight(64)
        en_layout.addWidget(self.english_text)
        edit_hint = QLabel("可在“先确认后播放”模式下修改英文，再点击播放。")
        edit_hint.setObjectName("hint")
        en_layout.addWidget(edit_hint)
        transcript_grid.addWidget(self.zh_group, 0, 0)
        transcript_grid.addWidget(self.en_group, 0, 1)
        transcript_grid.setColumnStretch(0, 1)
        transcript_grid.setColumnStretch(1, 1)

        # 核心控制区：主操作和辅助操作分成两个独立横排，避免窗口压缩时相互覆盖。
        self.voice_group = QGroupBox("实时语音控制")
        self.voice_group.setObjectName("controlDeck")
        self.voice_group.setMinimumHeight(182)
        controls = QVBoxLayout(self.voice_group)
        controls.setContentsMargins(10, 14, 10, 10)
        controls.setSpacing(8)
        primary_controls = QHBoxLayout()
        primary_controls.setSpacing(8)
        self.direct_button = HoldButton("F8  按住说原声")
        self.direct_button.setObjectName("directButton")
        self.direct_button.setFixedHeight(68)
        self.translate_button = HoldButton("F9  按住翻译说话")
        self.translate_button.setObjectName("translateButton")
        self.translate_button.setFixedHeight(68)
        self.continuous_f9_toggle = QCheckBox("F9 持续翻译")
        self.continuous_f9_toggle.setToolTip(
            "开启后，按一次 F9 持续监听；服务端检测停顿并自动逐句翻译。"
            "再按 F9、按 Esc 或按 F8 即停止。"
        )
        primary_controls.addWidget(self.direct_button, 1)
        primary_controls.addWidget(self.translate_button, 1)
        controls.addLayout(primary_controls)

        secondary_controls = QHBoxLayout()
        secondary_controls.setSpacing(8)
        self.play_button = QPushButton("▶ 播放当前英文")
        self.teacher_button = QPushButton("🎧 听老师 / Teams")
        self.teacher_button.setObjectName("teacherButton")
        self.stop_button = QPushButton("■ 停止  Esc")
        self.stop_button.setObjectName("dangerButton")
        for control in (
            self.continuous_f9_toggle,
            self.play_button,
            self.teacher_button,
            self.stop_button,
        ):
            control.setMinimumHeight(32)
            secondary_controls.addWidget(control, 1)
        controls.addLayout(secondary_controls)

        # 键盘与长文放到切换页，减少首页纵向占用。
        self.input_tabs = QTabWidget()
        self.input_tabs.setObjectName("inputTabs")
        typed_page = QWidget()
        typed_layout = QHBoxLayout(typed_page)
        typed_layout.setSpacing(10)
        typed_layout.setContentsMargins(8, 7, 8, 7)
        self.typed_input = SendTextEdit()
        self.typed_input.setFixedHeight(66)
        self.typed_input.setPlaceholderText("输入中文按 Enter；Ctrl+Enter 换行…")
        typed_layout.addWidget(self.typed_input, 4)

        typed_actions = QVBoxLayout()
        typed_actions.setSpacing(6)
        typed_actions.setContentsMargins(0, 0, 0, 0)
        self.typed_mode = ScrollSafeComboBox()
        self.typed_mode.addItem("中文翻译成英文后发送", "translate")
        self.typed_mode.addItem("按输入原文直接朗读", "direct")
        self.typed_send_button = QPushButton("发送文字语音  Enter ↵")
        self.typed_send_button.setObjectName("primaryButton")
        self.typed_send_button.setMinimumHeight(38)
        typed_actions.addWidget(self.typed_mode)
        typed_actions.addWidget(self.typed_send_button)
        typed_layout.addLayout(typed_actions, 2)
        typed_layout.setAlignment(typed_actions, Qt.AlignVCenter)

        long_text_page = QWidget()
        long_text_layout = QVBoxLayout(long_text_page)
        long_text_layout.setContentsMargins(8, 7, 8, 7)
        long_text_layout.setSpacing(6)
        self.long_text_input = QPlainTextEdit()
        self.long_text_input.setPlaceholderText(
            "把演讲稿、课堂稿粘贴到这里，点击开始朗读；系统会按句子切分后逐句播放。"
        )
        self.long_text_input.setMinimumHeight(48)
        self.long_text_input.setMaximumHeight(62)
        long_text_layout.addWidget(self.long_text_input)

        long_text_controls = QHBoxLayout()
        long_text_controls.setSpacing(8)
        long_text_controls.setAlignment(Qt.AlignTop)
        self.long_text_play = QPushButton("▶ 双语整段朗读")
        self.long_text_play.setObjectName("primaryButton")
        self.long_text_pause = QPushButton("⏸ 暂停")
        self.long_text_pause.setEnabled(False)
        self.long_text_pause.setVisible(False)
        self.long_text_stop = QPushButton("⏹ 停止")
        self.long_text_stop.setEnabled(False)
        self.long_text_stop.setVisible(False)
        self.long_text_progress = QProgressBar()
        self.long_text_progress.setRange(0, 0)
        self.long_text_progress.setTextVisible(True)
        self.long_text_progress.setFormat("%v / %m 句")
        self.long_text_progress.setFixedHeight(32)
        self.long_text_status = QLabel("就绪")
        self.long_text_status.setObjectName("hint")
        long_text_controls.addWidget(self.long_text_play)
        long_text_controls.addWidget(self.long_text_pause)
        long_text_controls.addWidget(self.long_text_stop)
        long_text_controls.addWidget(self.long_text_progress, 3)
        long_text_controls.addWidget(self.long_text_status)
        long_text_layout.addLayout(long_text_controls)
        self.input_tabs.addTab(typed_page, "⌨ 快捷输入")
        self.input_tabs.addTab(long_text_page, "▤ 长文本朗读")
        self.input_tabs.setMaximumHeight(132)

        # 常用工具带。
        self.utility_frame = QFrame()
        self.utility_frame.setObjectName("utilityBar")
        utility = QHBoxLayout(self.utility_frame)
        utility.setContentsMargins(7, 5, 7, 5)
        utility.setSpacing(5)
        self.record_button = QPushButton("● 录音")
        self.record_button.setObjectName("recordButton")
        self.subtitle_display_mode_quick = ScrollSafeComboBox()
        self.subtitle_display_mode_quick.addItem("字幕：中英双语", "both")
        self.subtitle_display_mode_quick.addItem("字幕：仅中文", "zh")
        self.subtitle_display_mode_quick.addItem("字幕：仅英文", "en")
        self.profile_quick = ScrollSafeComboBox()
        self.profile_quick.addItem("课程：默认", "")
        for profile_name in self.profile_store.names():
            self.profile_quick.addItem(f"课程：{profile_name}", profile_name)
        self.overlay_toggle_button = QPushButton("悬浮字幕")
        self.open_output_button = QPushButton("保存目录")
        self.export_subtitles_button = QPushButton("保存字幕")
        self.summary_button = QPushButton("课堂总结")
        self.clear_button = QPushButton("清空")
        utility.addWidget(self.record_button)
        utility.addWidget(self.subtitle_display_mode_quick)
        utility.addWidget(self.profile_quick)
        utility.addWidget(self.overlay_toggle_button)
        utility.addWidget(self.open_output_button)
        utility.addWidget(self.export_subtitles_button)
        utility.addWidget(self.summary_button)
        utility.addWidget(self.clear_button)

        self.history_group = QGroupBox("会议时间轴  ·  双击载入，点击 ▶ 重播")
        self.history_group.setObjectName("historyCard")
        history_layout = QVBoxLayout(self.history_group)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索本次字幕中的中文、英文或说话人…")
        history_layout.addWidget(self.history_search)
        self.history = QTableWidget(0, 6)
        self.history.setHorizontalHeaderLabels(["时间", "来源", "中文", "英文", "耗时", "重播"])
        self.history.horizontalHeader().setStretchLastSection(False)
        self.history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.history.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        # Column 5 holds replay *widgets*; ResizeToContents only measures item
        # text (none here) and would clip the buttons, so it stays Fixed and
        # _append_history_row widens it to the real button sizeHint.
        self.history.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.history.setColumnWidth(5, 96)
        self.history.verticalHeader().setDefaultSectionSize(28)
        self.history.verticalHeader().setMinimumSectionSize(28)
        # 默认展示最近 5 句话；更早的记录仍可向上滚动查看。
        self.history_visible_rows = 5
        self.history.setMinimumHeight(32 + self.history_visible_rows * 28 + 6)
        self.history_group.setMinimumHeight(self.history.minimumHeight() + 62)
        history_layout.addWidget(self.history, 1)

        self._arrange_meeting_workspace("crystal")

        self.artwork_panel = QFrame()
        self.artwork_panel.setObjectName("artPanel")
        self.artwork_panel.setFixedWidth(238)
        art_layout = QVBoxLayout(self.artwork_panel)
        art_layout.setContentsMargins(12, 14, 12, 12)
        art_layout.setSpacing(8)
        self.meeting_nav_title = QLabel("MEETING DESK")
        self.meeting_nav_title.setObjectName("meetingNavTitle")
        art_layout.addWidget(self.meeting_nav_title)
        self.meeting_nav_live_button = QPushButton("  实时翻译")
        self.meeting_nav_live_button.setObjectName("meetingNavButton")
        self.meeting_nav_live_button.setIcon(qta.icon("fa5s.microphone-alt", color="#1688d4"))
        self.meeting_nav_input_button = QPushButton("  快捷输入与朗读")
        self.meeting_nav_input_button.setObjectName("meetingNavButton")
        self.meeting_nav_input_button.setIcon(qta.icon("fa5s.keyboard", color="#1688d4"))
        self.meeting_nav_history_button = QPushButton("  会议记录")
        self.meeting_nav_history_button.setObjectName("meetingNavButton")
        self.meeting_nav_history_button.setIcon(qta.icon("fa5s.history", color="#1688d4"))
        self.meeting_nav_buttons = (
            self.meeting_nav_live_button,
            self.meeting_nav_input_button,
            self.meeting_nav_history_button,
        )
        for index, button in enumerate(self.meeting_nav_buttons):
            button.setIconSize(QSize(19, 19))
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.setChecked(index == 0)
            art_layout.addWidget(button)
        self.meeting_nav_live_button.clicked.connect(
            lambda: self._activate_meeting_section(0)
        )
        self.meeting_nav_input_button.clicked.connect(
            lambda: self._activate_meeting_section(1)
        )
        self.meeting_nav_history_button.clicked.connect(
            lambda: self._activate_meeting_section(2)
        )
        # The image itself is chosen by apply_backdrop() once settings load, so
        # the panel is built empty and filled on demand.
        self.artwork = ArtworkLabel()
        art_layout.addWidget(self.artwork, 1)
        self.artwork_shortcut = QLabel("F8  原声直通\nF9  中文翻译\nEsc  随时停止")
        self.artwork_shortcut.setObjectName("shortcutCard")
        self.artwork_shortcut.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        art_layout.addWidget(self.artwork_shortcut)
        self.art_credit = QLabel("")
        self.art_credit.setObjectName("artCredit")
        self.art_credit.setAlignment(Qt.AlignCenter)
        self.art_credit.setWordWrap(True)
        art_layout.addWidget(self.art_credit)
        self._arrange_meeting_shell("crystal")
        return tab

    def _activate_meeting_section(self, section: int) -> None:
        for index, button in enumerate(self.meeting_nav_buttons):
            button.setChecked(index == section)
        if section == 0:
            self.direct_button.setFocus()
        elif section == 1:
            self.input_tabs.setCurrentIndex(0)
            self.typed_input.setFocus()
        else:
            self.history_search.setFocus()

    @staticmethod
    def _take_all_layout_items(layout) -> None:
        """Detach layout items without deleting the reusable widgets."""
        while layout.count():
            layout.takeAt(0)

    def _arrange_meeting_shell(self, layout_name: str) -> None:
        """Use the artwork as a real navigation rail in Crystal Aurora."""
        if not hasattr(self, "artwork_panel"):
            return
        self._take_all_layout_items(self.meeting_shell)
        crystal = layout_name == "crystal"
        self.meeting_nav_title.setVisible(crystal)
        for button in self.meeting_nav_buttons:
            button.setVisible(crystal)
        if crystal:
            self.meeting_shell.addWidget(self.artwork_panel, 0)
            self.meeting_shell.addWidget(self.meeting_workspace, 1)
        else:
            self.meeting_shell.addWidget(self.meeting_workspace, 1)
            self.meeting_shell.addWidget(self.artwork_panel, 0)

    def _arrange_meeting_workspace(self, layout_name: str) -> None:
        """Place the same meeting controls into one of four real workspaces."""
        grid = self.meeting_grid
        self._take_all_layout_items(grid)
        for index in range(6):
            grid.setRowStretch(index, 0)
            grid.setColumnStretch(index, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        if layout_name == "signal":
            # Console-first: primary controls dominate the upper-left while the
            # current sentence and quick utilities remain visible beside them.
            grid.addWidget(self.voice_group, 0, 0, 2, 1)
            grid.addWidget(self.transcript_panel, 0, 1)
            grid.addWidget(self.utility_frame, 1, 1)
            grid.addWidget(self.input_tabs, 2, 0, 1, 2)
            grid.addWidget(self.history_group, 3, 0, 1, 2)
            grid.setRowStretch(3, 1)
        elif layout_name == "studio":
            # Writer-friendly: the bilingual sentence stays on top and typing /
            # long-form work sits beside the speech controls.
            grid.addWidget(self.transcript_panel, 0, 0, 1, 2)
            grid.addWidget(self.input_tabs, 1, 0)
            grid.addWidget(self.voice_group, 1, 1)
            grid.addWidget(self.utility_frame, 2, 0, 1, 2)
            grid.addWidget(self.history_group, 3, 0, 1, 2)
            grid.setRowStretch(3, 1)
        elif layout_name == "fluent":
            # Command-center: global tools come first, followed by the active
            # bilingual context and a balanced two-column action row.
            grid.addWidget(self.utility_frame, 0, 0, 1, 2)
            grid.addWidget(self.transcript_panel, 1, 0, 1, 2)
            grid.addWidget(self.voice_group, 2, 0)
            grid.addWidget(self.input_tabs, 2, 1)
            grid.addWidget(self.history_group, 3, 0, 1, 2)
            grid.setRowStretch(3, 1)
        else:
            # Crystal Aurora: the original spacious, presentation-oriented
            # composition with its local artwork panel.
            grid.addWidget(self.transcript_panel, 0, 0, 1, 2)
            grid.addWidget(self.voice_group, 1, 0, 1, 2)
            grid.addWidget(self.input_tabs, 2, 0, 1, 2)
            grid.addWidget(self.utility_frame, 3, 0, 1, 2)
            grid.addWidget(self.history_group, 4, 0, 1, 2)
            grid.setRowStretch(4, 1)

    def _arrange_voicestudio_versions(self, layout_name: str) -> None:
        """Reflow version facts without forcing the dashboard wider."""
        layout = self.voicestudio_versions_layout
        self._take_all_layout_items(layout)
        for index in range(6):
            layout.setColumnStretch(index, 0)
            layout.setRowStretch(index, 0)
        horizontal = layout_name != "signal"
        cards = list(self.voicestudio_version_cards)
        if layout_name == "fluent":
            cards.append(self.voicestudio_voice_count_card)
        self.voicestudio_voice_count_card.setVisible(layout_name == "fluent")
        self.voicestudio_status.setVisible(layout_name != "crystal")
        if horizontal:
            for column, card in enumerate(cards):
                layout.addWidget(card, 0, column)
                layout.setColumnStretch(column, 1)
            if layout_name != "crystal":
                layout.addWidget(self.voicestudio_status, 1, 0, 1, len(cards))
        else:
            for row, card in enumerate(self.voicestudio_version_cards):
                layout.addWidget(card, row, 0)
            layout.addWidget(self.voicestudio_status, 3, 0)
            layout.setColumnStretch(0, 1)

    def _arrange_voicestudio_actions(self, layout_name: str) -> None:
        """Match each reference's lifecycle button geometry with vector icons."""
        grid = self.voicestudio_action_grid
        utility = self.voicestudio_utility_actions
        self._take_all_layout_items(grid)
        self._take_all_layout_items(utility)
        self._take_all_layout_items(self.voicestudio_preview_actions)
        self._take_all_layout_items(self.voicestudio_library_quick_actions)
        for index in range(8):
            grid.setColumnStretch(index, 0)
            grid.setColumnMinimumWidth(index, 0)

        if layout_name == "crystal":
            labels = (
                "安装服务", "导入 MSI", "GitHub 下载", "启动服务",
                "停止服务", "重启服务", "升级服务", "卸载服务",
            )
            columns, button_height, tile, icon_size = 4, 84, True, 30
        elif layout_name == "fluent":
            labels = (
                "安装", "导入安装包", "GitHub 下载", "启动",
                "停止", "重启", "升级", "卸载",
            )
            columns, button_height, tile, icon_size = 8, 40, False, 16
        elif layout_name == "signal":
            labels = (
                "安装", "导入 MSI", "GitHub 下载", "启动",
                "停止", "重启", "升级", "卸载",
            )
            columns, button_height, tile, icon_size = 3, 46, False, 17
        else:
            labels = (
                "安装", "导入安装包", "GitHub 下载", "启动",
                "停止", "重启", "升级", "卸载",
            )
            columns, button_height, tile, icon_size = 3, 48, False, 17

        for index, (button, label) in enumerate(zip(self.voicestudio_lifecycle_buttons, labels)):
            icon_name, color = self.voicestudio_tile_icons[index]
            if button.objectName() == "voiceStudioPrimary":
                color = "#ffffff"
            button.setIcon(
                qta.icon(
                    icon_name,
                    color=color,
                    color_disabled="#aeb8c6",
                    color_active=color,
                )
            )
            button.setIconSize(QSize(icon_size, icon_size))
            button.setText(label)
            button.setToolButtonStyle(
                Qt.ToolButtonTextUnderIcon if tile else Qt.ToolButtonTextBesideIcon
            )
            button.setMinimumHeight(button_height)
            button.setProperty("actionTile", tile)
            button.style().unpolish(button)
            button.style().polish(button)
            grid.addWidget(button, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)

        if layout_name == "crystal":
            self.voicestudio_library_quick_actions.addWidget(
                self.voicestudio_preview_button
            )
            self.voicestudio_library_quick_actions.addWidget(
                self.voicestudio_backend_switch
            )
            self.voicestudio_library_quick_actions.addWidget(
                self.voicestudio_docs_button
            )
        else:
            utility.addWidget(self.voicestudio_backend_switch)
            utility.addWidget(self.voicestudio_docs_button)
            utility.addStretch()
            self.voicestudio_preview_actions.addWidget(
                self.voicestudio_preview_button
            )

    def _arrange_voicestudio_connection(self, layout_name: str) -> None:
        """Use the compact path actions shown by each reference shell."""
        if layout_name == "studio":
            labels = ("▣  打开", "▣  打开", "↗  打开")
            width = 96
        elif layout_name == "signal":
            labels = ("↗", "↗", "⧉")
            width = 42
        elif layout_name == "fluent":
            labels = ("▣", "▣", "↻")
            width = 44
        else:
            labels = ("", "", "")
            width = 42
        icons = ("fa5s.folder-open", "fa5s.file", "fa5s.sync-alt")
        for button, label, icon_name in zip(
            (
                self.voicestudio_choose_install_dir_button,
                self.voicestudio_choose_exe_button,
                self.voicestudio_refresh_button,
            ),
            labels,
            icons,
        ):
            button.setText(label)
            button.setIcon(qta.icon(icon_name, color="#3d6286"))
            button.setIconSize(QSize(18, 18))
            button.setMinimumWidth(width)
            button.setMaximumWidth(width)

    def _arrange_voicestudio_library(self, layout_name: str) -> None:
        """Switch between table and right-side voice-inspector presentations."""
        inspector = layout_name in {"signal", "fluent"}
        library_layout = self.voicestudio_library_layout
        self._take_all_layout_items(library_layout)
        for index in range(3):
            library_layout.setColumnStretch(index, 0)
            library_layout.setRowStretch(index, 0)
        if layout_name == "crystal":
            self.voicestudio_library_controls.setMinimumWidth(330)
            self.voicestudio_library_controls.setMaximumWidth(390)
            library_layout.addWidget(self.voicestudio_library_controls, 0, 0)
            library_layout.addWidget(self.voicestudio_voice_table, 0, 1)
            library_layout.setColumnStretch(1, 1)
        else:
            self.voicestudio_library_controls.setMinimumWidth(0)
            self.voicestudio_library_controls.setMaximumWidth(16777215)
            library_layout.addWidget(self.voicestudio_library_controls, 0, 0)
            library_layout.addWidget(self.voicestudio_voice_table, 1, 0)
            library_layout.setRowStretch(1, 1)
        self.voicestudio_voice_table.horizontalHeader().setVisible(not inspector)
        for column in range(5):
            self.voicestudio_voice_table.setColumnHidden(
                column, inspector and column not in {0, 3}
            )
        self.voicestudio_voice_table.verticalHeader().setDefaultSectionSize(
            48 if inspector else 34
        )
        if inspector:
            header = self.voicestudio_voice_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        else:
            header = self.voicestudio_voice_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            for column in (2, 3, 4):
                header.setSectionResizeMode(column, QHeaderView.ResizeToContents)

    def _arrange_voicestudio_workspace(self, layout_name: str) -> None:
        """Build four genuinely different VoiceStudio workspaces.

        Layout and palette intentionally remain independent: this method only
        controls hierarchy, card placement and density, so every one of the
        four layouts can be combined with every one of the four palettes.
        """
        if not hasattr(self, "voicestudio_dashboard_grid"):
            return
        grid = self.voicestudio_dashboard_grid
        self._take_all_layout_items(grid)
        for index in range(8):
            grid.setRowStretch(index, 0)
            grid.setColumnStretch(index, 0)
            grid.setColumnMinimumWidth(index, 0)

        connection = self.voicestudio_connection_card
        versions = self.voicestudio_versions_card
        actions = self.voicestudio_actions_card
        processes = self.voicestudio_process_card
        library = self.voicestudio_library_card
        preview = self.voicestudio_preview_card
        nav = self.voicestudio_nav_card
        page_heading = self.voicestudio_page_heading

        # Only the Crystal reference has a second-level navigation rail inside
        # the VoiceStudio page. Fluent and Signal already use the application
        # sidebar, so showing another rail would be visually and logically
        # incorrect.
        nav.setVisible(layout_name == "crystal")
        preview.setVisible(layout_name != "crystal")
        page_heading.setVisible(layout_name == "signal")
        self._arrange_voicestudio_versions(layout_name)
        self._arrange_voicestudio_actions(layout_name)
        self._arrange_voicestudio_connection(layout_name)
        self._arrange_voicestudio_library(layout_name)
        crystal = layout_name == "crystal"
        self.voicestudio_connection_card.setMaximumHeight(205 if crystal else 16777215)
        self.voicestudio_actions_card.setMaximumHeight(205 if crystal else 16777215)
        self.voicestudio_process_table.setMinimumHeight(125 if crystal else 205)
        self.voicestudio_process_card.setMaximumHeight(185 if crystal else 16777215)
        self.voicestudio_voice_table.setMinimumHeight(155 if crystal else 190)

        if layout_name == "fluent":
            # Microsoft Fluent command centre: navigation on the left,
            # operational dashboard in the middle and a voice inspector on
            # the right, matching the second reference design.
            connection.setTitle("安装路径与服务端点")
            versions.setTitle("服务概览")
            actions.setTitle("快速操作")
            processes.setTitle("进程监控")
            library.setTitle("声音检视器")
            preview.setTitle("试听设置")
            self.voicestudio_dashboard_content.setMinimumWidth(980)
            grid.addWidget(versions, 0, 0, 1, 2)
            grid.addWidget(connection, 1, 0, 1, 2)
            grid.addWidget(actions, 2, 0, 1, 2)
            grid.addWidget(processes, 3, 0, 2, 2)
            grid.addWidget(library, 0, 2, 4, 1)
            grid.addWidget(preview, 4, 2)
            grid.setColumnMinimumWidth(2, 330)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setRowStretch(3, 1)
        elif layout_name == "signal":
            # Midnight Signal workstation: a dense three-column operating
            # surface after the vertical navigation rail.
            connection.setTitle("服务配置")
            versions.setTitle("版本信息")
            actions.setTitle("生命周期管理")
            processes.setTitle("服务进程状态")
            library.setTitle("语音库")
            preview.setTitle("语音试听")
            self.voicestudio_dashboard_content.setMinimumWidth(1000)
            grid.addWidget(page_heading, 0, 0, 1, 3)
            grid.addWidget(connection, 1, 0)
            grid.addWidget(versions, 1, 1)
            grid.addWidget(library, 1, 2, 2, 1)
            grid.addWidget(actions, 2, 0, 1, 2)
            grid.addWidget(processes, 3, 0, 2, 2)
            grid.addWidget(preview, 3, 2, 2, 1)
            grid.setColumnMinimumWidth(2, 330)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setRowStretch(3, 1)
        elif layout_name == "studio":
            # Warm Studio: a calm balanced two-column layout with environment,
            # lifecycle and voices on the left, status and preview on the right.
            connection.setTitle("环境与路径")
            versions.setTitle("版本信息")
            actions.setTitle("生命周期控制")
            processes.setTitle("后台进程状态")
            library.setTitle("声音库")
            preview.setTitle("文本试听")
            self.voicestudio_dashboard_content.setMinimumWidth(1040)
            grid.addWidget(connection, 0, 0)
            grid.addWidget(versions, 0, 1)
            grid.addWidget(actions, 1, 0)
            grid.addWidget(processes, 1, 1)
            grid.addWidget(library, 2, 0)
            grid.addWidget(preview, 2, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setRowStretch(2, 1)
        else:
            # Crystal Aurora: airy navigation-led dashboard with strong
            # version cards and wide monitoring/library regions.
            connection.setTitle("本地运行管理")
            versions.setTitle("版本与运行状态")
            actions.setTitle("安装与服务操作")
            processes.setTitle("后台进程")
            library.setTitle("本地声音库预览")
            preview.setTitle("试听与使用")
            self.voicestudio_dashboard_content.setMinimumWidth(1180)
            grid.addWidget(nav, 0, 0, 5, 1)
            grid.addWidget(connection, 0, 1)
            grid.addWidget(actions, 0, 2)
            grid.addWidget(versions, 1, 1, 1, 2)
            grid.addWidget(processes, 2, 1, 1, 2)
            grid.addWidget(library, 3, 1, 1, 2)
            grid.setColumnMinimumWidth(0, 185)
            grid.setColumnStretch(1, 5)
            grid.setColumnStretch(2, 4)
            grid.setRowStretch(3, 1)

    def _arrange_settings_shell(self, layout_name: str) -> None:
        if not hasattr(self, "settings_nav_card"):
            return
        crystal = layout_name == "crystal"
        self.settings_nav_card.setVisible(crystal)
        self.settings_shell.setContentsMargins(8 if crystal else 0, 8 if crystal else 0, 8 if crystal else 0, 8 if crystal else 0)
        self.settings_shell.setSpacing(10 if crystal else 0)
        self.settings_content_layout.setContentsMargins(
            12 if crystal else 0,
            8 if crystal else 0,
            12 if crystal else 0,
            10 if crystal else 0,
        )
        self.settings_page_title.setVisible(crystal)

    def _arrange_settings_workspace(self, layout_name: str) -> None:
        """Keep settings responsive inside both top-nav and sidebar shells."""
        if not hasattr(self, "settings_grid"):
            return
        grid = self.settings_grid
        adv_grid = self.settings_advanced_grid
        self._take_all_layout_items(grid)
        self._take_all_layout_items(adv_grid)
        for index in range(12):
            grid.setColumnStretch(index, 0)
            grid.setRowStretch(index, 0)
        groups = self.settings_primary_groups
        sidebar_mode = layout_name in {"signal", "fluent"}
        if sidebar_mode:
            for row, group in enumerate(groups):
                grid.addWidget(group, row, 0)
            grid.addWidget(self.advanced_toggle, len(groups), 0)
            grid.addWidget(self.advanced_panel, len(groups) + 1, 0)
            grid.setColumnStretch(0, 1)
            for row, group in enumerate(self.settings_advanced_groups):
                adv_grid.addWidget(group, row, 0)
            adv_grid.setColumnStretch(0, 1)
        else:
            (
                api_group,
                audio_group,
                live_group,
                asr_group,
                mt_group,
                tts_group,
                hotkey_group,
                overlay_group,
                records_group,
                appearance_group,
                profile_group,
            ) = groups
            grid.addWidget(api_group, 0, 0)
            grid.addWidget(audio_group, 0, 1)
            grid.addWidget(live_group, 1, 0, 1, 2)
            grid.addWidget(asr_group, 2, 0)
            grid.addWidget(mt_group, 2, 1)
            grid.addWidget(tts_group, 3, 0)
            grid.addWidget(hotkey_group, 3, 1)
            grid.addWidget(overlay_group, 4, 0)
            grid.addWidget(records_group, 4, 1)
            grid.addWidget(appearance_group, 5, 0, 1, 2)
            grid.addWidget(profile_group, 6, 0, 1, 2)
            grid.addWidget(self.advanced_toggle, 7, 0, 1, 2)
            grid.addWidget(self.advanced_panel, 8, 0, 1, 2)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            advanced = self.settings_advanced_groups
            adv_grid.addWidget(advanced[0], 0, 0)
            adv_grid.addWidget(advanced[1], 0, 1)
            adv_grid.addWidget(advanced[2], 1, 0)
            adv_grid.addWidget(advanced[3], 1, 1)
            adv_grid.addWidget(advanced[4], 2, 0)
            adv_grid.addWidget(advanced[5], 2, 1)
            adv_grid.setColumnStretch(0, 1)
            adv_grid.setColumnStretch(1, 1)

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        self.settings_shell = outer
        self.settings_nav_card = QFrame()
        self.settings_nav_card.setObjectName("settingsNavCard")
        self.settings_nav_card.setMinimumWidth(182)
        self.settings_nav_card.setMaximumWidth(210)
        self.settings_nav_layout = QVBoxLayout(self.settings_nav_card)
        self.settings_nav_layout.setContentsMargins(14, 18, 14, 14)
        self.settings_nav_layout.setSpacing(9)
        self.settings_nav_title = QLabel("SETTINGS")
        self.settings_nav_title.setObjectName("settingsNavTitle")
        self.settings_nav_layout.addWidget(self.settings_nav_title)

        self.settings_content_panel = QFrame()
        self.settings_content_panel.setObjectName("settingsContentPanel")
        content_outer = QVBoxLayout(self.settings_content_panel)
        self.settings_content_layout = content_outer
        content_outer.setContentsMargins(10, 8, 10, 10)
        content_outer.setSpacing(8)
        top_actions = QHBoxLayout()
        self.settings_page_title = QLabel("应用设置")
        self.settings_page_title.setObjectName("settingsPageTitle")
        top_actions.addWidget(self.settings_page_title)
        top_actions.addStretch()
        self.top_glossary_button = QPushButton("编辑术语与提示词")
        self.save_button = QPushButton("保存全部设置")
        self.save_button.setObjectName("primaryButton")
        top_actions.addWidget(self.top_glossary_button)
        top_actions.addWidget(self.save_button)
        content_outer.addLayout(top_actions)
        scroll = QScrollArea()
        self.settings_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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
        self.input_device = ScrollSafeComboBox()
        self.teams_output_device = ScrollSafeComboBox()
        self.loopback_device = ScrollSafeComboBox()
        self.monitor_enabled = QCheckBox("同时在本机扬声器试听翻译后的英文")
        self.monitor_output_device = ScrollSafeComboBox()
        self.direct_sample_rate = ScrollSafeComboBox()
        for rate in (44100, 48000):
            self.direct_sample_rate.addItem(f"{rate} Hz", rate)
        self.refresh_devices_button = QPushButton("刷新设备列表")
        self.audio_diagnostics_button = QPushButton("运行音频设备诊断")
        audio_form.addRow("物理麦克风", self.input_device)
        audio_form.addRow("Teams 虚拟输出", self.teams_output_device)
        audio_form.addRow("老师/Teams 系统声", self.loopback_device)
        audio_form.addRow("监听", self.monitor_enabled)
        audio_form.addRow("本机试听设备", self.monitor_output_device)
        audio_form.addRow("", self.refresh_devices_button)
        cable_hint = QLabel("推荐安装 VB-CABLE，并在这里选 “CABLE Input”；Teams 麦克风选 “CABLE Output”。")
        cable_hint.setObjectName("hint")
        cable_hint.setWordWrap(True)
        audio_form.addRow("", cable_hint)

        grid.addWidget(api_group, 0, 0)
        grid.addWidget(audio_group, 0, 1)

        live_group = QGroupBox("3. F9 翻译引擎（按住 F9 说中文 → 英文语音）")
        live_form = QFormLayout(live_group)
        self.translation_engine = ScrollSafeComboBox()
        self.translation_engine.addItem("极速直译（推荐，单模型低延迟）", "live")
        self.translation_engine.addItem("传统流水线（ASR → Qwen-MT → TTS）", "classic")
        self.translation_engine.setVisible(False)
        self.engine_live_radio = QRadioButton(
            "极速直译：单模型直接出英文语音，延迟最低（推荐日常开会）"
        )
        self.engine_classic_radio = QRadioButton(
            "传统流水线：识别 → 翻译 → 合成三段独立，可配克隆音色，最像本人"
        )
        engine_options = QVBoxLayout()
        engine_options.addWidget(self.engine_live_radio)
        engine_options.addWidget(self.engine_classic_radio)
        live_form.addRow("引擎", engine_options)
        self.live_translate_model = ScrollSafeComboBox()
        self.live_translate_model.setEditable(True)
        self.live_translate_model.addItems([
            "qwen3.5-livetranslate-flash-realtime",
            "qwen3.5-livetranslate-flash-realtime-2026-05-19",
        ])
        self.live_voice_clone_mode = ScrollSafeComboBox()
        self.live_voice_clone_mode.addItem("服务端复刻一次（推荐，首句校准）", "once")
        self.live_voice_clone_mode.addItem("不复刻，使用默认音色（最快）", "default")
        self.live_voice_clone_mode.addItem("每轮动态复刻（多人场景）", "always")
        self.live_voice_clone_mode.addItem("使用预先复刻的固定音色（高级）", "fixed")
        self.live_voice = QLineEdit()
        self.live_voice.setPlaceholderText("固定 voice_id，例如 qwen-translate-vc-…；其他模式可留空")
        live_voice_row = QHBoxLayout()
        live_voice_row.addWidget(self.live_voice)
        self.clone_live_voice_button = QPushButton("实验性创建固定音色")
        self.clone_live_voice_button.setToolTip(
            "百炼当前可能拒绝为 Qwen3.5 LiveTranslate 预创建固定音色；"
            "推荐使用“服务端复刻一次”，无需上传样音。"
        )
        live_voice_row.addWidget(self.clone_live_voice_button)
        self.live_options = QWidget()
        live_options_form = QFormLayout(self.live_options)
        live_options_form.setContentsMargins(0, 0, 0, 0)
        live_options_form.addRow("直译模型", self.live_translate_model)
        live_options_form.addRow("声音复刻", self.live_voice_clone_mode)
        live_options_form.addRow("直译 voice", live_voice_row)
        live_form.addRow(self.live_options)
        live_hint = QLabel(
            "以上“直译模型 / 声音复刻 / 直译 voice”只在选中极速直译时生效；"
            "选传统流水线后它们会自动变灰，识别、翻译、合成模型请用下方第 4 / 5 / 6 组配置。\n"
            "极速模式通过一个 WebSocket 直接完成中文识别、英文翻译和英文语音流式输出。"
            "请在 API Key 权限中授权 qwen3.5-livetranslate-flash-realtime。"
            "推荐选择“服务端复刻一次”：第一句用于建立音色校准，首句可能使用默认过渡音色，"
            "第二句及后续会在同一连接中复用。会议页还可开启 F9 持续翻译，由服务端自动检测停顿。"
            "固定 voice_id 仅供已经拥有兼容音色的高级用户使用。"
        )
        live_hint.setObjectName("hint")
        live_hint.setWordWrap(True)
        live_form.addRow("", live_hint)
        grid.addWidget(live_group, 1, 0, 1, 2)

        asr_group = QGroupBox("4. 语音识别 ASR（字幕 / F8 / 传统流水线共用）")
        asr_form = QFormLayout(asr_group)
        self.asr_model = ScrollSafeComboBox()
        self.asr_model.addItems([
            "qwen3-asr-flash-realtime",
            "qwen3-asr-flash-realtime-2026-02-10",
            FUN_ASR_REALTIME_MODEL,
        ])
        self.asr_language = ScrollSafeComboBox()
        self.asr_language.addItem("中文（普通话/四川话/闽南语/吴语）", "zh")
        self.asr_language.addItem("粤语", "yue")
        self.vad_threshold = ScrollSafeDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setDecimals(2)
        self.vad_silence_ms = ScrollSafeSpinBox()
        self.vad_silence_ms.setRange(200, 2000)
        self.vad_silence_ms.setSingleStep(100)
        self.vad_silence_ms.setSuffix(" ms")
        self.direct_caption_enabled = QCheckBox("F8 原声直通时也识别中文并在松开后生成英文翻译")
        self.teacher_caption_enabled = QCheckBox("监听 Teams 系统声：老师英文原文 + 中文翻译")
        self.auto_start_teacher_caption = QCheckBox("软件启动后自动开始监听老师")
        asr_form.addRow("模型", self.asr_model)
        asr_form.addRow("说话语言", self.asr_language)
        asr_form.addRow("原声字幕", self.direct_caption_enabled)
        asr_form.addRow("老师字幕", self.teacher_caption_enabled)
        asr_form.addRow("自动监听", self.auto_start_teacher_caption)
        asr_hint = QLabel(
            "选择 fun-asr-realtime 时，课程术语表（右上角“编辑术语与提示词”维护）"
            "会自动作为识别热词，按识别语言取中文或英文，课堂专业词识别更准。"
        )
        asr_hint.setObjectName("hint")
        asr_hint.setWordWrap(True)
        asr_form.addRow("", asr_hint)

        mt_group = QGroupBox("5. 机器翻译 Qwen-MT（字幕 / 键盘输入 / 传统流水线共用）")
        mt_form = QFormLayout(mt_group)
        self.translation_model = ScrollSafeComboBox()
        self.translation_model.addItems(["qwen-mt-flash", "qwen-mt-plus", "qwen-mt-turbo", "qwen-mt-lite"])
        self.summary_model = ScrollSafeComboBox()
        self.summary_model.setEditable(True)
        self.summary_model.addItems(["qwen-plus", "qwen-max", "qwen-flash"])
        self.long_form_model = ScrollSafeComboBox()
        self.long_form_model.setEditable(True)
        self.long_form_model.addItems(["qwen-plus", "qwen-max", "qwen-flash"])
        self.translation_domain = QLineEdit()
        self.translation_terms = QPlainTextEdit()
        self.translation_terms.setMaximumHeight(65)
        self.translation_terms.setPlaceholderText('[{"source":"有限元","target":"finite element method"}]')
        self.translation_memories = QPlainTextEdit()
        self.translation_memories.setMaximumHeight(65)
        self.translation_memories.setPlaceholderText('[{"source":"老师您好","target":"Hello, Professor."}]')
        self.tts_pronunciations = QPlainTextEdit()
        self.tts_pronunciations.setMaximumHeight(65)
        self.tts_pronunciations.setPlaceholderText('{"OpenAI":"Open A I","Qwen":"Q wen"}')
        self.speak_mode = ScrollSafeComboBox()
        self.speak_mode.addItem("立即翻译并发送（最快）", "auto")
        self.speak_mode.addItem("先确认/编辑，再手动发送（稳妥）", "confirm")
        self.translation_style = ScrollSafeComboBox()
        self.translation_style.addItem("自然礼貌课堂英语", "polite")
        self.translation_style.addItem("简洁日常口语", "concise")
        self.translation_style.addItem("正式学术表达", "academic")
        self.translation_style.addItem("尽量逐字忠实", "literal")
        self.glossary_button = QPushButton("表格方式编辑课程术语")
        mt_form.addRow("模型", self.translation_model)
        mt_form.addRow("领域提示（英文）", self.translation_domain)
        mt_form.addRow("发送方式", self.speak_mode)
        mt_hint = QLabel(
            "课程术语、翻译记忆、表达风格等较少改动的选项已收进底部“高级设置”；"
            "也可以直接用右上角“编辑术语与提示词”。"
        )
        mt_hint.setObjectName("hint")
        mt_hint.setWordWrap(True)
        mt_form.addRow("", mt_hint)

        grid.addWidget(asr_group, 2, 0)
        grid.addWidget(mt_group, 2, 1)

        tts_group = QGroupBox("6. 语音合成 TTS · 克隆音色（键盘发声 / 传统流水线共用）")
        tts_form = QFormLayout(tts_group)
        self.tts_provider = ScrollSafeComboBox()
        self.tts_provider.addItem("百炼云语音", "aliyun")
        self.tts_provider.addItem("VoiceStudio 本地语音", "voicestudio")
        self.tts_model = ScrollSafeComboBox()
        self.tts_model.addItems([
            QWEN3_TTS_VC_REALTIME_MODEL,
            QWEN3_TTS_VC_HTTP_MODEL,
            COSYVOICE_V3_5_PLUS_MODEL,
            COSYVOICE_V3_FLASH_MODEL,
            "qwen-audio-3.0-tts-plus",
            "qwen-audio-3.0-tts-flash",
        ])
        self.voice = QLineEdit()
        self.voice.setPlaceholderText("克隆 voice_id，或系统音色如 loongjohn")
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice)
        self.clone_voice_button = QPushButton("创建克隆音色")
        voice_row.addWidget(self.clone_voice_button)
        self.tts_volume = ScrollSafeSpinBox()
        self.tts_volume.setRange(0, 100)
        self.tts_rate = ScrollSafeDoubleSpinBox()
        self.tts_rate.setRange(0.5, 2.0)
        self.tts_rate.setSingleStep(0.05)
        self.tts_pitch = ScrollSafeDoubleSpinBox()
        self.tts_pitch.setRange(0.5, 2.0)
        self.tts_pitch.setSingleStep(0.05)
        self.tts_seed = ScrollSafeSpinBox()
        self.tts_seed.setRange(0, 65535)
        self.tts_emotion = ScrollSafeComboBox()
        for label, value in [
            ("自然（无标签）", ""),
            ("悲伤 [sad]", "[sad]"),
            ("惊讶 [amazed]", "[amazed]"),
            ("低沉大声 [deep and loud shouting]", "[deep and loud shouting]"),
        ]:
            self.tts_emotion.addItem(label, value)
        self.tts_instruction = QLineEdit()
        self.tts_instruction.setMaxLength(100)
        self.tts_instruction.setToolTip("CosyVoice 3.5 Plus 指令最多 100 个字符。")
        self.enable_aigc_tag = QCheckBox("在生成音频中嵌入官方 AIGC 隐性标识")
        self.aigc_propagator = QLineEdit()
        self.aigc_propagate_id = QLineEdit()
        self.tts_hint = QLabel()
        self.tts_hint.setObjectName("hint")
        self.tts_hint.setWordWrap(True)
        tts_form.addRow("语音后端", self.tts_provider)
        tts_form.addRow("百炼模型", self.tts_model)
        tts_form.addRow("百炼音色 voice", voice_row)
        tts_form.addRow("音量 0–100", self.tts_volume)
        tts_form.addRow("语速 0.5–2.0", self.tts_rate)
        tts_form.addRow("音调 0.5–2.0", self.tts_pitch)
        tts_form.addRow("", self.tts_hint)

        hotkey_group = QGroupBox("7. 快捷键")
        hotkey_form = QFormLayout(hotkey_group)
        self.direct_hotkey = ScrollSafeComboBox()
        self.translate_hotkey = ScrollSafeComboBox()
        self.cancel_hotkey = ScrollSafeComboBox()
        for box in (self.direct_hotkey, self.translate_hotkey):
            box.addItems([f"f{i}" for i in range(1, 13)])
        self.cancel_hotkey.addItems(["esc"] + [f"f{i}" for i in range(1, 13)])
        self.request_timeout = ScrollSafeSpinBox()
        self.request_timeout.setRange(10, 180)
        self.request_timeout.setSuffix(" 秒")
        self.http_proxy = QLineEdit()
        self.http_proxy.setPlaceholderText("通常留空；例如 http://127.0.0.1:7890")
        hotkey_form.addRow("按住原声", self.direct_hotkey)
        hotkey_form.addRow("按住翻译", self.translate_hotkey)
        hotkey_form.addRow("停止/取消", self.cancel_hotkey)

        grid.addWidget(tts_group, 3, 0)
        grid.addWidget(hotkey_group, 3, 1)

        overlay_group = QGroupBox("7. 歌词式双语悬浮字幕")
        overlay_form = QFormLayout(overlay_group)
        self.overlay_enabled = QCheckBox("启用悬浮字幕（仅在本机显示）")
        self.overlay_always_on_top = QCheckBox("始终置顶")
        self.overlay_opacity = ScrollSafeSpinBox()
        self.overlay_opacity.setRange(20, 100)
        self.overlay_opacity.setSuffix(" %")
        self.overlay_chinese_font_size = ScrollSafeSpinBox()
        self.overlay_chinese_font_size.setRange(14, 64)
        self.overlay_chinese_font_size.setSuffix(" px")
        self.overlay_english_font_size = ScrollSafeSpinBox()
        self.overlay_english_font_size.setRange(12, 56)
        self.overlay_english_font_size.setSuffix(" px")
        self.overlay_width = ScrollSafeSpinBox()
        self.overlay_width.setRange(420, 1800)
        self.overlay_width.setSuffix(" px")
        self.overlay_height = ScrollSafeSpinBox()
        self.overlay_height.setRange(90, 500)
        self.overlay_height.setSuffix(" px")
        overlay_form.addRow("显示", self.overlay_enabled)
        overlay_form.addRow("窗口层级", self.overlay_always_on_top)
        overlay_form.addRow("透明度", self.overlay_opacity)
        overlay_hint = QLabel(
            "可直接拖动悬浮窗改变位置，修改透明度立即预览；字号和尺寸在底部“高级设置”里调整。"
        )
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
        self.ui_layout = ScrollSafeComboBox()
        self._add_layout_options(self.ui_layout)
        self.theme = ScrollSafeComboBox()
        self._add_theme_options(self.theme)
        self.backdrop = ScrollSafeComboBox()
        self._add_backdrop_options(self.backdrop)
        # A live thumbnail keeps the backdrop choice observable from the
        # settings page, where the meeting artwork panel is off-screen.
        self.backdrop_preview = QLabel()
        self.backdrop_preview.setObjectName("backdropPreview")
        self.backdrop_preview.setFixedSize(74, 111)
        self.backdrop_preview.setAlignment(Qt.AlignCenter)
        backdrop_row = QHBoxLayout()
        backdrop_row.setSpacing(8)
        backdrop_row.addWidget(self.backdrop, 1)
        backdrop_row.addWidget(self.backdrop_preview)
        self.subtitle_display_mode = ScrollSafeComboBox()
        self.subtitle_display_mode.addItem("中英双语", "both")
        self.subtitle_display_mode.addItem("仅中文", "zh")
        self.subtitle_display_mode.addItem("仅英文", "en")
        appearance_form.addRow("界面布局", self.ui_layout)
        appearance_form.addRow("界面配色", self.theme)
        appearance_form.addRow("侧边栏背景", backdrop_row)
        appearance_form.addRow("实时字幕", self.subtitle_display_mode)
        appearance_hint = QLabel(
            "4 套布局与 4 套配色可以自由组合，共 16 种外观；背景立绘是第三个独立选项，"
            "同样可与它们任意搭配。选“自动跟随主题”时背景会随配色切换，"
            "选“无背景”则隐藏侧边栏立绘。实时显示方式不会删减缓存。"
        )
        appearance_hint.setObjectName("hint")
        appearance_hint.setWordWrap(True)
        appearance_form.addRow("", appearance_hint)
        grid.addWidget(appearance_group, 5, 0, 1, 2)

        profile_group = QGroupBox("10. 课程配置")
        profile_form = QFormLayout(profile_group)
        self.profile_name = ScrollSafeComboBox()
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

        self.advanced_toggle = QPushButton("高级设置（不常用，点击展开）▾")
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setCheckable(True)
        self.advanced_panel = QWidget()
        self.advanced_panel.setVisible(False)
        adv_grid = QGridLayout(self.advanced_panel)
        self.settings_advanced_grid = adv_grid
        adv_grid.setContentsMargins(0, 0, 0, 0)

        adv_audio = QGroupBox("音频 · 高级")
        adv_audio_form = QFormLayout(adv_audio)
        adv_audio_form.addRow("原声采样率", self.direct_sample_rate)
        adv_audio_form.addRow("", self.audio_diagnostics_button)

        adv_asr = QGroupBox("识别 · 高级")
        adv_asr_form = QFormLayout(adv_asr)
        adv_asr_form.addRow("VAD 灵敏度阈值", self.vad_threshold)
        adv_asr_form.addRow("静音断句时间", self.vad_silence_ms)
        vad_hint = QLabel(
            "官方低延迟建议：threshold=0.0、silence=400ms；默认 500ms，降低误断句。"
            "fun-asr-realtime 由服务端自动断句，这两个参数不生效。"
        )
        vad_hint.setObjectName("hint")
        vad_hint.setWordWrap(True)
        adv_asr_form.addRow("", vad_hint)

        adv_mt = QGroupBox("翻译 · 高级")
        adv_mt_form = QFormLayout(adv_mt)
        adv_mt_form.addRow("课堂总结模型", self.summary_model)
        adv_mt_form.addRow("长文语境翻译模型", self.long_form_model)
        adv_mt_form.addRow("表达风格", self.translation_style)
        adv_mt_form.addRow("术语表 JSON", self.translation_terms)
        adv_mt_form.addRow("", self.glossary_button)
        adv_mt_form.addRow("翻译记忆 JSON", self.translation_memories)

        adv_tts = QGroupBox("合成 · 高级")
        adv_tts_form = QFormLayout(adv_tts)
        adv_tts_form.addRow("随机种子", self.tts_seed)
        adv_tts_form.addRow("情绪标签", self.tts_emotion)
        adv_tts_form.addRow("Free-style 指令", self.tts_instruction)
        adv_tts_form.addRow("TTS 发音词典 JSON", self.tts_pronunciations)
        adv_tts_form.addRow("AIGC 标识", self.enable_aigc_tag)
        adv_tts_form.addRow("ContentPropagator", self.aigc_propagator)
        adv_tts_form.addRow("PropagateID", self.aigc_propagate_id)

        adv_net = QGroupBox("网络 · 高级")
        adv_net_form = QFormLayout(adv_net)
        adv_net_form.addRow("接口超时", self.request_timeout)
        adv_net_form.addRow("HTTP 代理", self.http_proxy)

        adv_overlay = QGroupBox("悬浮字幕 · 高级")
        adv_overlay_form = QFormLayout(adv_overlay)
        adv_overlay_form.addRow("中文字幕字号", self.overlay_chinese_font_size)
        adv_overlay_form.addRow("英文字幕字号", self.overlay_english_font_size)
        adv_overlay_form.addRow("悬浮窗宽度", self.overlay_width)
        adv_overlay_form.addRow("悬浮窗高度", self.overlay_height)

        adv_grid.addWidget(adv_audio, 0, 0)
        adv_grid.addWidget(adv_asr, 0, 1)
        adv_grid.addWidget(adv_mt, 1, 0)
        adv_grid.addWidget(adv_tts, 1, 1)
        adv_grid.addWidget(adv_net, 2, 0)
        adv_grid.addWidget(adv_overlay, 2, 1)

        grid.addWidget(self.advanced_toggle, 7, 0, 1, 2)
        grid.addWidget(self.advanced_panel, 8, 0, 1, 2)
        self.advanced_toggle.toggled.connect(self.toggle_advanced_panel)

        self.settings_grid = grid
        self.settings_primary_groups = (
            api_group,
            audio_group,
            live_group,
            asr_group,
            mt_group,
            tts_group,
            hotkey_group,
            overlay_group,
            records_group,
            appearance_group,
            profile_group,
        )
        self.settings_advanced_groups = (
            adv_audio,
            adv_asr,
            adv_mt,
            adv_tts,
            adv_net,
            adv_overlay,
        )

        layout.addLayout(grid)
        actions = QHBoxLayout()
        official = QPushButton("打开极速直译官方文档")
        official.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://help.aliyun.com/zh/model-studio/qwen3-5-livetranslate-flash-realtime")
            )
        )
        vb_cable = QPushButton("打开 VB-CABLE 官网")
        vb_cable.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/")))
        actions.addWidget(official)
        actions.addWidget(vb_cable)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        scroll.setWidget(content)
        content_outer.addWidget(scroll, 1)

        nav_specs = (
            ("常用与 API", "fa5s.sliders-h", api_group),
            ("翻译与语音", "fa5s.language", live_group),
            ("字幕与记录", "fa5s.closed-captioning", overlay_group),
            ("外观与课程", "fa5s.palette", appearance_group),
        )
        self.settings_nav_buttons: list[QPushButton] = []
        for index, (label, icon_name, target) in enumerate(nav_specs):
            button = QPushButton(f"  {label}")
            button.setObjectName("settingsNavButton")
            button.setIcon(qta.icon(icon_name, color="#1688d4"))
            button.setIconSize(QSize(19, 19))
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setMinimumHeight(43)
            button.clicked.connect(
                lambda checked=False, page=index, widget=target: self._activate_settings_section(
                    page, widget
                )
            )
            self.settings_nav_layout.addWidget(button)
            self.settings_nav_buttons.append(button)
        self.settings_nav_layout.addStretch()
        settings_hint = QLabel("所有配置自动保存在本机\n切换主题不会改变功能")
        settings_hint.setObjectName("navHealthCard")
        settings_hint.setWordWrap(True)
        self.settings_nav_layout.addWidget(settings_hint)
        outer.addWidget(self.settings_nav_card)
        outer.addWidget(self.settings_content_panel, 1)
        return tab

    def _activate_settings_section(self, section: int, target: QWidget) -> None:
        for index, button in enumerate(self.settings_nav_buttons):
            button.setChecked(index == section)
        self.settings_scroll.ensureWidgetVisible(target, 18, 18)

    def _connect_signals(self) -> None:
        self.signals.direct_down.connect(self.handle_direct_down)
        self.signals.direct_up.connect(self.handle_direct_up)
        self.signals.translate_down.connect(self.handle_translate_down)
        self.signals.translate_up.connect(self.handle_translate_up)
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
        self.signals.continuous_finished.connect(self.on_continuous_translation_finished)
        self.signals.long_text_progress.connect(self._on_long_text_progress)
        self.signals.long_text_state.connect(self._on_long_text_state)
        self.signals.long_form_progress.connect(self._on_long_form_progress)
        self.signals.long_form_ready.connect(self._open_long_form_reader)
        self.signals.voicestudio_catalog.connect(self._apply_voicestudio_catalog)
        self.signals.voicestudio_runtime.connect(self._apply_voicestudio_runtime)
        self.signals.voicestudio_task_progress.connect(self._on_voicestudio_task_progress)
        self.signals.voicestudio_task_finished.connect(self._on_voicestudio_task_finished)
        self.signals.voicestudio_shutdown_progress.connect(self._on_voicestudio_shutdown_progress)
        self.signals.voicestudio_shutdown_finished.connect(self._on_voicestudio_shutdown_finished)
        self.direct_button.hold_pressed.connect(self.handle_direct_down)
        self.direct_button.hold_released.connect(self.handle_direct_up)
        self.translate_button.hold_pressed.connect(self.handle_translate_down)
        self.translate_button.hold_released.connect(self.handle_translate_up)
        self.stop_button.clicked.connect(self.cancel_all)
        self.play_button.clicked.connect(self.play_current_english)
        self.teacher_button.clicked.connect(self.toggle_teacher_caption)
        self.clear_button.clicked.connect(self.clear_text)
        self.typed_send_button.clicked.connect(self.send_typed_text)
        self.typed_input.send_requested.connect(self.send_typed_text)
        self.long_text_play.clicked.connect(self.start_long_text_speech)
        self.long_text_pause.clicked.connect(self.pause_or_resume_long_text)
        self.long_text_stop.clicked.connect(self.stop_long_text_speech)
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
        self.clone_voice_button.clicked.connect(self.open_current_voice_manager)
        self.tts_provider.currentIndexChanged.connect(self.update_tts_model_ui)
        self.tts_provider.currentIndexChanged.connect(
            self._sync_voicestudio_backend_switch
        )
        self.engine_scope_button.clicked.connect(self.toggle_tts_provider)
        self.tts_model.currentTextChanged.connect(self.on_tts_model_changed)
        self.clone_live_voice_button.clicked.connect(
            lambda: self.open_clone_dialog(self.live_translate_model.currentText())
        )
        self.translation_engine.currentIndexChanged.connect(self.update_translation_engine_ui)
        self.engine_live_radio.toggled.connect(self.on_engine_radio_toggled)
        self.engine_classic_radio.toggled.connect(self.on_engine_radio_toggled)
        self.live_voice_clone_mode.currentIndexChanged.connect(self.update_translation_engine_ui)
        self.direct_hotkey.currentTextChanged.connect(self.update_translation_engine_ui)
        self.translate_hotkey.currentTextChanged.connect(self.update_translation_engine_ui)
        self.cancel_hotkey.currentTextChanged.connect(self.update_translation_engine_ui)
        self.continuous_f9_toggle.toggled.connect(self.on_continuous_f9_setting_changed)
        self.glossary_button.clicked.connect(self.open_glossary_dialog)
        self.top_glossary_button.clicked.connect(self.open_glossary_dialog)
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
        self.ui_layout.currentIndexChanged.connect(self._layout_from_settings)
        self.layout_quick.currentIndexChanged.connect(self._layout_from_quick)
        self.theme.currentIndexChanged.connect(self._theme_from_settings)
        self.theme_quick.currentIndexChanged.connect(self._theme_from_quick)
        self.backdrop.currentIndexChanged.connect(self._backdrop_from_settings)
        self.backdrop_quick.currentIndexChanged.connect(self._backdrop_from_quick)
        self.voicestudio_refresh_button.clicked.connect(self.refresh_voicestudio_catalog)
        self.voicestudio_start_button.clicked.connect(self.launch_voicestudio)
        self.voicestudio_stop_button.clicked.connect(self.stop_voicestudio)
        self.voicestudio_restart_button.clicked.connect(self.restart_voicestudio)
        self.voicestudio_install_button.clicked.connect(self.install_voicestudio)
        self.voicestudio_import_button.clicked.connect(self.import_voicestudio_installer)
        self.voicestudio_upgrade_button.clicked.connect(self.upgrade_voicestudio)
        self.voicestudio_uninstall_button.clicked.connect(self.uninstall_voicestudio)
        self.voicestudio_cancel_task_button.clicked.connect(self.cancel_voicestudio_task)
        self.voicestudio_choose_install_dir_button.clicked.connect(self.choose_voicestudio_install_dir)
        self.voicestudio_choose_exe_button.clicked.connect(self.choose_voicestudio_executable)
        self.voicestudio_download_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_RELEASES_PAGE))
        )
        self.voicestudio_docs_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://voicestudio.sh/docs/quickstart"))
        )
        self.voicestudio_use_aliyun_button.clicked.connect(
            lambda: self.use_tts_backend("aliyun")
        )
        self.voicestudio_use_button.clicked.connect(
            lambda: self.use_tts_backend("voicestudio")
        )
        self.voicestudio_preview_button.clicked.connect(self.preview_voicestudio_voice)
        self.voicestudio_voice_table.cellClicked.connect(self.select_voicestudio_voice_row)
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
        tts_voices = dict(values.get("tts_voices") or {})
        if self.tts_model.currentText() and self.voice.text().strip():
            tts_voices[self.tts_model.currentText()] = self.voice.text().strip()
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
                "continuous_f9_enabled": self.continuous_f9_toggle.isChecked(),
                "translation_model": self.translation_model.currentText(),
                "summary_model": self.summary_model.currentText().strip(),
                "long_form_model": self.long_form_model.currentText().strip(),
                "source_language": "Chinese",
                "target_language": "English",
                "translation_terms": self.translation_terms.toPlainText().strip(),
                "translation_memories": self.translation_memories.toPlainText().strip(),
                "translation_domain": self.translation_domain.text().strip(),
                "translation_style": self.translation_style.currentData(),
                "speak_mode": self.speak_mode.currentData(),
                "confirm_before_speak": self.speak_mode.currentData() == "confirm",
                "tts_provider": self.tts_provider.currentData(),
                "tts_model": self.tts_model.currentText(),
                "voice": self.voice.text().strip(),
                "tts_voices": tts_voices,
                "tts_sample_rate": 24000,
                "tts_volume": self.tts_volume.value(),
                "tts_rate": self.tts_rate.value(),
                "tts_pitch": self.tts_pitch.value(),
                "tts_seed": self.tts_seed.value(),
                "tts_language_hint": "en",
                "tts_instruction": self.tts_instruction.text().strip(),
                "tts_pronunciations": self.tts_pronunciations.toPlainText().strip(),
                "tts_emotion_tag": self.tts_emotion.currentData(),
                "voicestudio_url": self.voicestudio_url.text().strip(),
                "voicestudio_model": self.voicestudio_model.currentText().strip() or "tts-1",
                "voicestudio_voice": self._current_voicestudio_voice_id(),
                "voicestudio_executable": self.voicestudio_executable.text().strip(),
                "voicestudio_install_dir": self.voicestudio_install_dir.text().strip(),
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
                "ui_layout": self.layout_quick.currentData(),
                "theme": self.theme_quick.currentData(),
                "backdrop": self.backdrop_quick.currentData(),
                "active_profile": self.profile_quick.currentData() or "",
                "output_directory": self.output_directory.text().strip(),
            }
        )
        return values

    def _current_voicestudio_voice_id(self) -> str:
        text = self.voicestudio_voice.currentText().strip()
        index = self.voicestudio_voice.findText(text, Qt.MatchExactly)
        if index >= 0:
            return str(self.voicestudio_voice.itemData(index) or text or "default")
        return text or "default"

    def load_settings_into_ui(self) -> None:
        v = self.settings.values
        voices = dict(v.get("tts_voices") or {})
        if v.get("tts_model") and v.get("voice") and v["tts_model"] not in voices:
            voices[v["tts_model"]] = v["voice"]
            self.settings.values["tts_voices"] = voices
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
        self.continuous_f9_toggle.setChecked(bool(v.get("continuous_f9_enabled", False)))
        self.translation_model.setCurrentText(v["translation_model"])
        self.summary_model.setCurrentText(v["summary_model"])
        self.long_form_model.setCurrentText(v.get("long_form_model", v["summary_model"]))
        self.translation_domain.setText(v["translation_domain"])
        self.translation_terms.setPlainText(v["translation_terms"])
        self.translation_memories.setPlainText(v["translation_memories"])
        self._select_data(self.translation_style, v["translation_style"])
        speak_mode = v.get("speak_mode") or ("confirm" if v.get("confirm_before_speak") else "auto")
        self._select_data(self.speak_mode, speak_mode)
        self._select_data(self.tts_provider, v.get("tts_provider", "aliyun"))
        self.tts_model.setCurrentText(v["tts_model"])
        self.voice.setText(voices.get(v["tts_model"], v["voice"]))
        self.tts_volume.setValue(int(v["tts_volume"]))
        self.tts_rate.setValue(float(v["tts_rate"]))
        self.voicestudio_preview_volume.setValue(int(v["tts_volume"]))
        self.voicestudio_preview_rate.setValue(int(round(float(v["tts_rate"]) * 100)))
        self.tts_pitch.setValue(float(v["tts_pitch"]))
        self.tts_seed.setValue(int(v["tts_seed"]))
        self._select_data(self.tts_emotion, v["tts_emotion_tag"])
        self.tts_instruction.setText(v["tts_instruction"])
        self.tts_pronunciations.setPlainText(v.get("tts_pronunciations", ""))
        self.voicestudio_url.setText(v.get("voicestudio_url", "http://127.0.0.1:3900"))
        self.voicestudio_model.setCurrentText(v.get("voicestudio_model", "tts-1"))
        local_voice = str(v.get("voicestudio_voice", "default") or "default")
        if self.voicestudio_voice.findData(local_voice) < 0:
            self.voicestudio_voice.addItem(local_voice, local_voice)
        self._select_data(self.voicestudio_voice, local_voice)
        self.voicestudio_executable.setText(v.get("voicestudio_executable", ""))
        self.voicestudio_install_dir.setText(v.get("voicestudio_install_dir", ""))
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
        self._select_data(self.ui_layout, v.get("ui_layout", "crystal"))
        self._select_data(self.layout_quick, v.get("ui_layout", "crystal"))
        self._select_data(self.theme, v["theme"])
        self._select_data(self.theme_quick, v["theme"])
        self._select_data(self.backdrop, v.get("backdrop", DEFAULT_BACKDROP))
        self._select_data(self.backdrop_quick, v.get("backdrop", DEFAULT_BACKDROP))
        self.output_directory.setText(v["output_directory"])
        self._select_data(self.profile_quick, v.get("active_profile", ""))
        self.profile_name.setCurrentText(v.get("active_profile", ""))
        self.apply_overlay_settings()
        self.apply_subtitle_display_mode()
        self.apply_layout()
        self.apply_theme()
        self.apply_backdrop()
        self.update_translation_engine_ui()
        self.update_tts_model_ui()

    @staticmethod
    def _select_data(box: QComboBox, value: Any) -> None:
        index = box.findData(value)
        if index >= 0:
            box.setCurrentIndex(index)

    def on_engine_radio_toggled(self, *_args) -> None:
        self._select_data(
            self.translation_engine,
            "live" if self.engine_live_radio.isChecked() else "classic",
        )

    def toggle_advanced_panel(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setText(
            "高级设置（点击收起）▴" if expanded else "高级设置（不常用，点击展开）▾"
        )

    def update_translation_engine_ui(self, *_args) -> None:
        live = self.translation_engine.currentData() == "live"
        fixed_voice = self.live_voice_clone_mode.currentData() == "fixed"
        continuous = live and self.continuous_f9_toggle.isChecked()
        if live:
            self._select_data(self.speak_mode, "auto")
        self.speak_mode.setEnabled(not live)
        self.speak_mode.setToolTip(
            "极速模式会边生成边播放，因此固定为自动发送。" if live else ""
        )
        direct_key = self.direct_hotkey.currentText().upper()
        translate_key = self.translate_hotkey.currentText().upper()
        cancel_key = self.cancel_hotkey.currentText().upper()
        self.direct_button.setText(f"按住直接说话\n{direct_key}")
        self.translate_button.setText(
            f"开启/关闭持续翻译\n{translate_key}"
            if continuous
            else (
                f"按住极速翻译说话\n{translate_key}"
                if live
                else f"按住传统翻译说话\n{translate_key}"
            )
        )
        self.continuous_f9_toggle.setText(
            f"{translate_key} 持续翻译\n按一次开启/关闭"
        )
        self.stop_button.setText(f"停止 / 取消  {cancel_key}")
        self.chinese_text.setPlaceholderText(
            f"按住 {translate_key}，或开启持续翻译后直接说中文…"
        )
        self.english_text.setPlaceholderText(
            f"松开 {translate_key} 后，英文译文会显示在这里…"
        )
        self.continuous_f9_toggle.setEnabled(live)
        self.engine_live_radio.blockSignals(True)
        self.engine_classic_radio.blockSignals(True)
        self.engine_live_radio.setChecked(live)
        self.engine_classic_radio.setChecked(not live)
        self.engine_live_radio.blockSignals(False)
        self.engine_classic_radio.blockSignals(False)
        self.live_options.setEnabled(live)
        self.live_voice.setEnabled(live and fixed_voice)
        self.clone_live_voice_button.setVisible(fixed_voice)
        self.clone_live_voice_button.setEnabled(live and fixed_voice)

    def update_tts_model_ui(self, *_args) -> None:
        local = self.tts_provider.currentData() == "voicestudio"
        model = self.tts_model.currentText()
        qwen3 = is_qwen3_tts_vc_model(model)
        realtime = is_qwen3_tts_vc_realtime_model(model)
        cosyvoice = is_cosyvoice_model(model)
        self.tts_model.setEnabled(not local)
        self.voice.setEnabled(not local)
        for control in (self.tts_volume, self.tts_rate, self.tts_pitch):
            control.setEnabled(not qwen3 or realtime)
        for control in (
            self.tts_seed,
            self.tts_instruction,
        ):
            control.setEnabled(not qwen3)
        for control in (
            self.enable_aigc_tag,
            self.aigc_propagator,
            self.aigc_propagate_id,
        ):
            control.setEnabled(not qwen3 and not model.startswith("cosyvoice-v3.5"))
        self.tts_emotion.setEnabled(not qwen3 and not cosyvoice)
        if local:
            self.tts_volume.setEnabled(False)
            self.tts_rate.setEnabled(True)
            self.tts_pitch.setEnabled(False)
            self.tts_seed.setEnabled(False)
            self.tts_instruction.setEnabled(False)
            self.tts_emotion.setEnabled(False)
            self.enable_aigc_tag.setEnabled(False)
            self.aigc_propagator.setEnabled(False)
            self.aigc_propagate_id.setEnabled(False)
            self.tts_hint.setText(
                "VoiceStudio 本地后端已选中：模型和声音档案在“本地声音工作台”管理。"
                "传统 F9、键盘发声、重播和整段朗读会走本机 3900 端口；极速直译不变。"
            )
        elif realtime:
            self.tts_hint.setText(
                "Qwen3-TTS-VC Realtime：推荐会议/F9 传统流水线，边生成边播放；支持音量、语速和音调。"
                "音色必须专门绑定此实时模型。"
            )
        elif qwen3:
            self.tts_hint.setText(
                "Qwen3-TTS-VC 高清版：更接近原声，适合键盘输入和重播，但首段等待更长；"
                "音色必须专门绑定此高清模型。"
            )
        elif cosyvoice:
            if model == COSYVOICE_V3_5_PLUS_MODEL:
                self.tts_hint.setText(
                    "CosyVoice V3.5 Plus：克隆相似度最高的版本，语气停顿都能复刻。"
                    "它没有系统音色，必须先点“创建克隆音色”传 10–20 秒清晰录音建专属音色。"
                )
            else:
                self.tts_hint.setText(
                    "CosyVoice V3 Flash：轻快便宜，但克隆偏机械；追求像本人请改用 v3.5-plus。"
                    "可用系统音色（如 longanyang）或克隆音色；不支持情绪标签。"
                )
        else:
            self.tts_hint.setText(
                "Qwen-Audio 兼容模式：保留随机种子、情绪标签和 Free-style 指令。"
            )
        self.clone_voice_button.setText(
            "打开本地声音工作台"
            if local
            else ("创建 Qwen3 专属音色" if qwen3 else "创建克隆音色")
        )
        self._sync_engine_scope_button()

    def _sync_engine_scope_button(self) -> None:
        if not hasattr(self, "engine_scope_button"):
            return
        local = self.tts_provider.currentData() == "voicestudio"
        self.engine_scope_button.setText("◉ 本地引擎" if local else "☁ 云端引擎")
        self.engine_scope_button.setProperty("localEngine", local)
        style = self.engine_scope_button.style()
        style.unpolish(self.engine_scope_button)
        style.polish(self.engine_scope_button)

    def toggle_tts_provider(self) -> None:
        local = self.tts_provider.currentData() == "voicestudio"
        target = "aliyun" if local else "voicestudio"
        self._select_data(self.tts_provider, target)
        self.update_tts_model_ui()
        self.settings.update({"tts_provider": target})
        if target == "voicestudio":
            self._set_status("已切换到本地引擎 · VoiceStudio")
            self.refresh_voicestudio_runtime()
        else:
            self._set_status("已切换到云端引擎 · 阿里云百炼")

    def open_current_voice_manager(self) -> None:
        if self.tts_provider.currentData() == "voicestudio":
            self.tabs.setCurrentWidget(self.voicestudio_tab)
            return
        self.open_clone_dialog(self.tts_model.currentText())

    def on_tts_model_changed(self, model: str) -> None:
        voices = dict(self.settings.get("tts_voices", {}) or {})
        if self._active_tts_model and self.voice.text().strip():
            voices[self._active_tts_model] = self.voice.text().strip()
        self.settings.values["tts_voices"] = voices
        self._active_tts_model = model
        if model in voices:
            self.voice.setText(str(voices[model]))
        elif model == self.settings.get("tts_model"):
            self.voice.setText(str(self.settings.get("voice", "")))
        else:
            self.voice.clear()
        self.update_tts_model_ui()

    def _hotkey_name(self, setting: str) -> str:
        control = {
            "direct_hotkey": self.direct_hotkey,
            "translate_hotkey": self.translate_hotkey,
            "cancel_hotkey": self.cancel_hotkey,
        }[setting]
        return control.currentText().upper()

    def _ready_status(self) -> str:
        backend = "本地 VoiceStudio" if self.tts_provider.currentData() == "voicestudio" else "百炼语音"
        return (
            f"就绪 · {self._hotkey_name('direct_hotkey')} 原声 / "
            f"{self._hotkey_name('translate_hotkey')} 翻译 · {backend}"
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

    def _layout_from_quick(self, *_args) -> None:
        if hasattr(self, "ui_layout"):
            blocker = QSignalBlocker(self.ui_layout)
            self._select_data(self.ui_layout, self.layout_quick.currentData())
            del blocker
        self.apply_layout()
        self.apply_theme()
        self.settings.update({"ui_layout": self.layout_quick.currentData()})

    def _layout_from_settings(self, *_args) -> None:
        blocker = QSignalBlocker(self.layout_quick)
        self._select_data(self.layout_quick, self.ui_layout.currentData())
        del blocker
        self.apply_layout()
        self.apply_theme()
        self.settings.update({"ui_layout": self.ui_layout.currentData()})

    def apply_layout(self, *_args) -> None:
        layout_name = (
            self.layout_quick.currentData() if hasattr(self, "layout_quick") else "crystal"
        ) or "crystal"
        self.app_root.setProperty("uiLayout", layout_name)
        self._arrange_app_shell(str(layout_name))
        self._arrange_meeting_shell(str(layout_name))
        self._arrange_meeting_workspace(str(layout_name))
        self._arrange_voicestudio_workspace(str(layout_name))
        self._arrange_settings_shell(str(layout_name))
        self._arrange_settings_workspace(str(layout_name))

        metrics = {
            "crystal": (12, 8, 12, 10, 7, 64, 166, 122, (1460, 960)),
            "signal": (6, 5, 6, 6, 5, 60, 166, 124, (1460, 900)),
            "studio": (12, 8, 12, 9, 7, 64, 174, 174, (1420, 900)),
            "fluent": (5, 4, 5, 5, 4, 56, 156, 156, (1460, 900)),
        }[str(layout_name)]
        left, top, right, bottom, spacing, primary_height, deck_height, input_height, size = metrics
        self.root_layout.setContentsMargins(left, top, right, bottom)
        self.root_layout.setSpacing(spacing)
        self.header_layout.setSpacing(spacing + 2)
        self.meeting_shell.setContentsMargins(spacing, spacing, spacing, spacing)
        self.meeting_shell.setSpacing(spacing + 2)
        self.meeting_grid.setSpacing(spacing)
        self.direct_button.setFixedHeight(primary_height)
        self.translate_button.setFixedHeight(primary_height)
        self.voice_group.setMinimumHeight(deck_height)
        self.input_tabs.setMaximumHeight(input_height)

        # Every layout hosts the artwork panel so the backdrop selector stays
        # meaningful outside Crystal; only the width and the shortcut card
        # adapt to the space each layout can spare. Visibility is owned by
        # apply_backdrop(), which hides the panel in all layouts at once when
        # the user picks "no backdrop".
        art_widths = {"crystal": 208, "signal": 168, "studio": 190, "fluent": 168}
        self.artwork_panel.setFixedWidth(art_widths.get(str(layout_name), 190))
        self.artwork_shortcut.setVisible(layout_name == "crystal")
        self.brand_logo.setVisible(layout_name == "crystal")
        self.title_label.setVisible(layout_name == "crystal")
        self.subtitle_label.setVisible(layout_name == "crystal")
        self.tabs.setDocumentMode(layout_name in {"signal", "fluent"})
        if not self.isMaximized():
            self.resize(*size)

    def _theme_from_quick(self, *_args) -> None:
        if hasattr(self, "theme"):
            blocker = QSignalBlocker(self.theme)
            self._select_data(self.theme, self.theme_quick.currentData())
            del blocker
        self.apply_theme()
        self.settings.update({"theme": self.theme_quick.currentData()})

    def _theme_from_settings(self, *_args) -> None:
        blocker = QSignalBlocker(self.theme_quick)
        self._select_data(self.theme_quick, self.theme.currentData())
        del blocker
        self.apply_theme()
        self.settings.update({"theme": self.theme.currentData()})

    def apply_theme(self, *_args) -> None:
        theme = self.theme_quick.currentData() if hasattr(self, "theme_quick") else "shizuku"
        layout_name = (
            self.layout_quick.currentData() if hasattr(self, "layout_quick") else "crystal"
        ) or "crystal"
        palette = {
            "dark": DARK_STYLE_SHEET,
            "warm": WARM_STYLE_SHEET,
            "light": STYLE_SHEET,
        }.get(theme, SHIZUKU_STYLE_SHEET)
        self.setStyleSheet(palette + LAYOUT_STYLE_SHEETS.get(str(layout_name), ""))
        # An "auto" backdrop follows the palette, so repaint after a switch.
        if self.current_backdrop_choice() == "auto":
            self.apply_backdrop()

    def current_backdrop_choice(self) -> str:
        """Stored backdrop choice; may be "auto" or "none"."""
        combo = getattr(self, "backdrop_quick", None)
        return combo.currentData() if combo is not None else DEFAULT_BACKDROP

    def _backdrop_from_quick(self, *_args) -> None:
        if hasattr(self, "backdrop"):
            blocker = QSignalBlocker(self.backdrop)
            self._select_data(self.backdrop, self.backdrop_quick.currentData())
            del blocker
        self.apply_backdrop()
        self.settings.update({"backdrop": self.backdrop_quick.currentData()})

    def _backdrop_from_settings(self, *_args) -> None:
        blocker = QSignalBlocker(self.backdrop_quick)
        self._select_data(self.backdrop_quick, self.backdrop.currentData())
        del blocker
        self.apply_backdrop()
        self.settings.update({"backdrop": self.backdrop.currentData()})

    def apply_backdrop(self, *_args) -> None:
        """Paint the side artwork and the settings thumbnail for the choice."""
        if not hasattr(self, "artwork"):
            return
        theme = self.theme_quick.currentData() if hasattr(self, "theme_quick") else "shizuku"
        key = resolve_backdrop(theme, self.current_backdrop_choice())
        if key == BACKDROP_NONE:
            self.artwork.set_backdrop(None)
            self.artwork.setText("VOICE\nWORKSTATION")
            self.art_credit.setText("已关闭背景立绘")
            self.backdrop_preview.setPixmap(QPixmap())
            self.backdrop_preview.setText("无")
            self.artwork_panel.setVisible(False)
            return
        self.artwork_panel.setVisible(True)
        path = backdrop_path(key)
        if self.artwork.set_backdrop(path):
            self.artwork.setText("")
            self.art_credit.setText(f"{backdrop_credit(key)}\nAI 生成 · 仅本机使用")
            self._paint_backdrop_preview(path)
        else:
            self.artwork.setText("VOICE\nWORKSTATION")
            self.art_credit.setText("背景图缺失（local_assets/backgrounds）")
            self.backdrop_preview.setPixmap(QPixmap())
            self.backdrop_preview.setText("缺失")

    def _paint_backdrop_preview(self, path: Path) -> None:
        """Show a small crop of the chosen artwork next to the combo box."""
        source = QPixmap(str(path))
        if source.isNull():
            self.backdrop_preview.setPixmap(QPixmap())
            self.backdrop_preview.setText("缺失")
            return
        scaled = source.scaled(
            self.backdrop_preview.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        self.backdrop_preview.setPixmap(scaled)
        self.backdrop_preview.setText("")

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
        self.long_form_model.setCurrentText(
            str(profile.get("long_form_model", "qwen-plus") or "qwen-plus")
        )
        self._select_data(
            self.live_voice_clone_mode,
            profile.get("live_voice_clone_mode", "once") or "once",
        )
        self.live_voice.setText(str(profile.get("live_voice", "")))
        self._select_data(self.tts_provider, profile.get("tts_provider", "aliyun") or "aliyun")
        tts_voices = profile.get("tts_voices") or {}
        if isinstance(tts_voices, dict) and tts_voices:
            merged = dict(self.settings.get("tts_voices", {}) or {})
            merged.update(tts_voices)
            self.settings.values["tts_voices"] = merged
        profile_model = str(profile.get("tts_model", "") or "")
        if profile_model and self.tts_model.findText(profile_model) >= 0:
            self.tts_model.setCurrentText(profile_model)
        self.tts_instruction.setText(str(profile.get("tts_instruction", "")))
        self.tts_pronunciations.setPlainText(str(profile.get("tts_pronunciations", "")))
        voice = str(profile.get("voice", "") or "")
        if voice or not profile_model:
            # Legacy profiles only store a plain voice; keep applying it as-is.
            self.voice.setText(voice)
        if voice and profile_model and profile_model == self.tts_model.currentText():
            merged = dict(self.settings.get("tts_voices", {}) or {})
            merged[profile_model] = voice
            self.settings.values["tts_voices"] = merged
        local_model = str(profile.get("voicestudio_model", "") or "")
        if local_model:
            self.voicestudio_model.setCurrentText(local_model)
        local_voice = str(profile.get("voicestudio_voice", "") or "")
        if local_voice:
            if self.voicestudio_voice.findData(local_voice) < 0:
                self.voicestudio_voice.addItem(local_voice, local_voice)
            self._select_data(self.voicestudio_voice, local_voice)
        self.profile_name.setCurrentText(name)
        self.profile_quick.blockSignals(True)
        self._select_data(self.profile_quick, name)
        self.profile_quick.blockSignals(False)
        self.settings.update({"active_profile": name})
        self.update_translation_engine_ui()
        self.update_tts_model_ui()
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
            parse_pronunciation_dictionary(self.tts_pronunciations.toPlainText())
            if self.api_key.text().strip():
                self.settings.set_api_key(self.api_key.text().strip())
                self.api_key.clear()
            self.settings.update(self.current_settings())
            self._discard_manual_live_session(graceful=True)
            self.apply_overlay_settings()
            self.apply_subtitle_display_mode()
            self.apply_layout()
            self.apply_theme()
            self.start_hotkeys()
            self.update_translation_engine_ui()
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

    def handle_direct_down(self) -> None:
        self.direct_key_held = True
        with self.state_lock:
            state = self.state
        if state in {"continuous", "continuous_stopping"}:
            self.switch_to_direct_after_continuous = True
            self.cancel_event.set()
            self.stop_continuous_translation()
            self._set_status("正在停止 F9 持续翻译并切换到 F8 原声…")
            return
        self.start_direct()

    def handle_direct_up(self) -> None:
        self.direct_key_held = False
        self.switch_to_direct_after_continuous = False
        self.stop_direct()

    def handle_translate_down(self) -> None:
        if self.continuous_f9_toggle.isChecked():
            with self.state_lock:
                state = self.state
            if state in {"continuous", "continuous_stopping"}:
                self.stop_continuous_translation()
            else:
                self.start_continuous_translation()
            return
        self.start_translation()

    def handle_translate_up(self) -> None:
        if not self.continuous_f9_toggle.isChecked():
            self.stop_translation_capture()

    def on_continuous_f9_setting_changed(self, checked: bool) -> None:
        with self.state_lock:
            active = self.state in {"continuous", "continuous_stopping"}
        if active and not checked:
            self.stop_continuous_translation()
        self.settings.update({"continuous_f9_enabled": bool(checked)})
        self.update_translation_engine_ui()

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

    def _live_session_signature(
        self,
        values: dict[str, Any],
        client: BailianClient,
        phrases: dict[str, str],
    ) -> tuple[Any, ...]:
        return (
            client.workspace_id,
            client.api_key,
            values["live_translate_model"],
            values["live_voice_clone_mode"],
            values.get("live_voice", ""),
            tuple(sorted(phrases.items())),
        )

    def _get_manual_live_session(
        self,
        values: dict[str, Any],
        client: BailianClient,
        phrases: dict[str, str],
        on_audio: Callable[[bytes], None],
    ) -> QwenLiveTranslate:
        signature = self._live_session_signature(values, client, phrases)
        stale: QwenLiveTranslate | None = None
        with self.live_translate_session_lock:
            session = self.live_translate_session
            if (
                session is not None
                and (
                    self.live_translate_session_key != signature
                    or not session.opened.is_set()
                    or bool(session.error_text)
                )
            ):
                stale = session
                session = None
                self.live_translate_session = None
                self.live_translate_session_key = None
        if stale is not None:
            stale.close()
        if session is None:
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
                on_audio=on_audio,
                on_status=self.signals.status.emit,
                on_error=lambda error: log.warning("LiveTranslate: %s", error),
                proxy=values["http_proxy"],
            )
            session.start()
            session.wait_ready()
            with self.live_translate_session_lock:
                self.live_translate_session = session
                self.live_translate_session_key = signature
        session.on_audio = on_audio
        session.on_source_preview = lambda text: self.signals.asr_preview.emit(text, "live")
        session.on_translation_preview = self.signals.live_translation_preview.emit
        session.on_status = self.signals.status.emit
        session.begin_turn()
        return session

    def _discard_manual_live_session(
        self, expected: QwenLiveTranslate | None = None, *, graceful: bool = False
    ) -> None:
        with self.live_translate_session_lock:
            session = self.live_translate_session
            if expected is not None and session is not expected:
                return
            self.live_translate_session = None
            self.live_translate_session_key = None
        if session is not None:
            if graceful:
                session.finish(timeout=3.0)
            else:
                session.close()

    def _make_client(self, values: dict[str, Any] | None = None) -> BailianClient:
        values = values or self.current_settings()
        return BailianClient(
            self.settings.get_api_key(),
            values["workspace_id"],
            timeout=values["request_timeout"],
            proxy=values["http_proxy"],
        )

    def _make_tts_client(self, values: dict[str, Any] | None = None) -> Any:
        values = values or self.current_settings()
        if values.get("tts_provider") == "voicestudio":
            return VoiceStudioClient(
                values.get("voicestudio_url", "http://127.0.0.1:3900"),
                timeout=values.get("request_timeout", 45),
            )
        return self._make_client(values)

    def refresh_voicestudio_catalog(self) -> None:
        self.voicestudio_refresh_button.setEnabled(False)
        self.voicestudio_status.setText("正在连接本地服务…")
        self.refresh_voicestudio_runtime(include_latest=True)
        values = self.current_settings()

        def worker() -> None:
            try:
                client = VoiceStudioClient(
                    values.get("voicestudio_url", "http://127.0.0.1:3900"),
                    timeout=values.get("request_timeout", 45),
                )
                health = client.health()
                voices, engines = client.catalog()
                version = str(health.get("version") or "未知版本")
                device = str(health.get("device") or health.get("compute") or "本机")
                message = f"已连接 · v{version} · {device} · {len(voices)} 个声音"
                self.signals.voicestudio_catalog.emit(True, voices, engines, message)
            except Exception as exc:
                self.signals.voicestudio_catalog.emit(False, [], [], str(exc))

        threading.Thread(target=worker, name="voicestudio-catalog", daemon=True).start()

    def _apply_voicestudio_catalog(
        self,
        ok: bool,
        raw_voices: object,
        raw_engines: object,
        message: str,
    ) -> None:
        self.voicestudio_refresh_button.setEnabled(True)
        self.voicestudio_status.setText(message)
        self.voicestudio_status.setProperty("connected", ok)
        self.voicestudio_status.style().unpolish(self.voicestudio_status)
        self.voicestudio_status.style().polish(self.voicestudio_status)
        if not ok:
            self._set_status("VoiceStudio 未连接")
            return

        self.voicestudio_voices = [
            item for item in (raw_voices if isinstance(raw_voices, list) else [])
            if isinstance(item, VoiceStudioVoice)
        ]
        self.voicestudio_engines = [
            item for item in (raw_engines if isinstance(raw_engines, list) else [])
            if isinstance(item, VoiceStudioEngine)
        ]
        self.voicestudio_voice_count.setText(str(len(self.voicestudio_voices)))
        self.voicestudio_api_meta.setText(
            f"已同步 {len(self.voicestudio_voices)} 个声音 · {len(self.voicestudio_engines)} 个引擎"
        )
        selected_model = self.voicestudio_model.currentText().strip() or "tts-1"
        selected_voice = self._current_voicestudio_voice_id()
        model_ids = ["tts-1"]
        for engine in self.voicestudio_engines:
            if engine.engine_id not in model_ids:
                model_ids.append(engine.engine_id)
        self.voicestudio_model.clear()
        self.voicestudio_model.addItems(model_ids)
        self.voicestudio_model.setCurrentText(
            selected_model if selected_model in model_ids else "tts-1"
        )

        self.voicestudio_voice.clear()
        self.voicestudio_voice.addItem("默认音色", "default")
        for voice in self.voicestudio_voices:
            label = voice.name
            if voice.kind:
                label += f" · {voice.kind}"
            self.voicestudio_voice.addItem(label, voice.voice_id)
        if self.voicestudio_voice.findData(selected_voice) < 0 and selected_voice != "default":
            self.voicestudio_voice.addItem(str(selected_voice), str(selected_voice))
        self._select_data(self.voicestudio_voice, selected_voice)

        self.voicestudio_voice_table.setRowCount(0)
        for voice in self.voicestudio_voices:
            row = self.voicestudio_voice_table.rowCount()
            self.voicestudio_voice_table.insertRow(row)
            for column, value in enumerate(
                [voice.name, voice.voice_id, voice.kind, voice.language, voice.engine]
            ):
                self.voicestudio_voice_table.setItem(row, column, QTableWidgetItem(value))
        self.voicestudio_manager.adopt_running_processes()
        self.refresh_voicestudio_runtime()
        self._set_status(message)

    def select_voicestudio_voice_row(self, row: int, _column: int) -> None:
        if not 0 <= row < self.voicestudio_voice_table.rowCount():
            return
        item = self.voicestudio_voice_table.item(row, 1)
        if item is None:
            return
        voice_id = item.text().strip()
        if self.voicestudio_voice.findData(voice_id) >= 0:
            self._select_data(self.voicestudio_voice, voice_id)

    def use_tts_backend(self, provider: str) -> None:
        """Switch the active TTS backend from the VoiceStudio segmented control."""
        if self.tts_provider.currentData() == provider:
            self._sync_voicestudio_backend_switch()
            return
        self._select_data(self.tts_provider, provider)
        self.update_tts_model_ui()
        self.save_settings_from_ui()
        self._sync_voicestudio_backend_switch()
        if provider == "voicestudio":
            self._set_status(
                f"已启用 VoiceStudio · {self.voicestudio_model.currentText()} · "
                f"{self.voicestudio_voice.currentText()}"
            )
        else:
            self._set_status("已启用百炼云语音")

    # Backwards-compatible alias kept for scripts/tests using the old name.
    def use_voicestudio_backend(self) -> None:
        self.use_tts_backend("voicestudio")

    def _sync_voicestudio_backend_switch(self, *_args) -> None:
        """Reflect the current tts_provider selection on the segmented control."""
        if not hasattr(self, "voicestudio_backend_group"):
            return
        local = self.tts_provider.currentData() == "voicestudio"
        blocker = QSignalBlocker(self.voicestudio_backend_group)
        self.voicestudio_use_button.setChecked(local)
        self.voicestudio_use_aliyun_button.setChecked(not local)
        del blocker

    def choose_voicestudio_executable(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 VoiceStudio.exe",
            self.voicestudio_executable.text().strip() or str(Path.home()),
            "VoiceStudio (VoiceStudio.exe);;Windows 程序 (*.exe)",
        )
        if path:
            self.voicestudio_executable.setText(path)

    def choose_voicestudio_install_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 VoiceStudio 安装目录",
            self.voicestudio_install_dir.text().strip() or str(Path.home()),
        )
        if path:
            self.voicestudio_install_dir.setText(path)
            self.settings.update(self.current_settings())

    @staticmethod
    def _format_process_uptime(seconds: int) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def refresh_voicestudio_runtime(self, include_latest: bool = False) -> None:
        if self._voicestudio_runtime_refreshing or self._shutdown_in_progress:
            return
        self._voicestudio_runtime_refreshing = True
        values = self.current_settings()
        signals = self.signals

        def worker() -> None:
            try:
                result: object = self.voicestudio_manager.runtime_snapshot(
                    values.get("voicestudio_url", "http://127.0.0.1:3900"),
                    values.get("voicestudio_executable", ""),
                    values.get("voicestudio_install_dir", ""),
                    include_latest=include_latest,
                )
            except Exception as exc:
                result = exc
            try:
                signals.voicestudio_runtime.emit(result)
            except RuntimeError:
                # The window can be destroyed while a network/process probe is
                # finishing.  Its result is no longer useful and must not make
                # the daemon worker print a noisy "C++ object deleted" trace.
                return

        threading.Thread(target=worker, name="voicestudio-runtime", daemon=True).start()

    def _apply_voicestudio_runtime(self, raw_snapshot: object) -> None:
        self._voicestudio_runtime_refreshing = False
        if not isinstance(raw_snapshot, VoiceStudioRuntime):
            log.warning("VoiceStudio runtime refresh failed: %s", raw_snapshot)
            return
        self._last_voicestudio_runtime = raw_snapshot
        installation = raw_snapshot.installation
        installed_version = installation.version or ("已安装" if installation.installed else "未安装")
        self.voicestudio_installed_version.setText(installed_version)
        self.voicestudio_installed_meta.setText(
            "本地程序已就绪" if installation.installed else "尚未安装 VoiceStudio"
        )
        self.voicestudio_api_version.setText(
            (f"{raw_snapshot.api_version} · {raw_snapshot.api_device}".strip(" ·"))
            if raw_snapshot.api_ok
            else "未连接"
        )
        if raw_snapshot.latest_release is not None:
            release = raw_snapshot.latest_release
            checksum = release.expected_sha256.strip().lower()
            self.voicestudio_latest_version.setText(release.tag)
            if checksum:
                self.voicestudio_latest_meta.setText(f"SHA-256: {checksum}")
                self.voicestudio_latest_meta.setToolTip(
                    f"GitHub 官方发布文件 SHA-256：\n{checksum}\n\n可用鼠标选中复制。"
                )
            else:
                self.voicestudio_latest_meta.setText("SHA-256: 官方未提供")
                self.voicestudio_latest_meta.setToolTip(
                    "当前 GitHub Release 没有提供可核验的 SHA-256。"
                )
        if installation.executable and not self.voicestudio_executable.hasFocus():
            self.voicestudio_executable.setText(installation.executable)
        if installation.install_dir and not self.voicestudio_install_dir.hasFocus():
            self.voicestudio_install_dir.setText(installation.install_dir)

        running = bool(raw_snapshot.processes)
        if raw_snapshot.api_ok:
            state = "ok"
            if running:
                text = f"运行正常 · {len(raw_snapshot.processes)} 个进程"
            else:
                text = "服务在线 · 远程或外部托管"
        elif running:
            state = "error"
            text = f"运行异常 · {len(raw_snapshot.processes)} 个进程，但 API 未就绪"
        else:
            state = "stopped"
            text = "已停止"
        self.voicestudio_status.setText(text)
        self.voicestudio_status.setProperty("connected", raw_snapshot.api_ok)
        self.voicestudio_status.setProperty("runtimeState", state)
        self.voicestudio_status.setToolTip(raw_snapshot.api_message)
        self.voicestudio_status.style().unpolish(self.voicestudio_status)
        self.voicestudio_status.style().polish(self.voicestudio_status)
        self.main_nav_health.setText(f"●  本地服务\n    {text}")
        self.main_nav_health.setProperty("connected", raw_snapshot.api_ok)
        self.main_nav_health.setProperty("runtimeState", state)
        self.main_nav_health.style().unpolish(self.main_nav_health)
        self.main_nav_health.style().polish(self.main_nav_health)
        self.voicestudio_nav_health.setText(f"●  本地引擎\n    {text}")
        self.voicestudio_nav_health.setProperty("runtimeState", state)
        self.voicestudio_nav_health.style().unpolish(self.voicestudio_nav_health)
        self.voicestudio_nav_health.style().polish(self.voicestudio_nav_health)
        self.footer_status.setText(f"●  系统状态：{text}")

        self.voicestudio_process_table.setRowCount(0)
        normal_color = QColor("#158553")
        abnormal_color = QColor("#cf3535")
        for process in raw_snapshot.processes:
            row = self.voicestudio_process_table.rowCount()
            self.voicestudio_process_table.insertRow(row)
            normal = raw_snapshot.api_ok
            values = [
                "● 正常" if normal else "● 异常",
                process.role,
                process.name,
                str(process.pid),
                f"{process.memory_mb:.1f} MB",
                self._format_process_uptime(process.uptime_seconds),
                process.command_line,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(normal_color if normal else abnormal_color)
                if column == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.voicestudio_process_table.setItem(row, column, item)
        self._update_voicestudio_action_states()

    def _update_voicestudio_action_states(self) -> None:
        snapshot = getattr(self, "_last_voicestudio_runtime", None)
        installed = bool(snapshot and snapshot.installation.installed)
        running = bool(snapshot and snapshot.processes)
        busy = self._voicestudio_task_running or self._shutdown_in_progress
        self.voicestudio_install_button.setEnabled(not busy)
        self.voicestudio_import_button.setEnabled(not busy)
        self.voicestudio_download_button.setEnabled(not busy)
        self.voicestudio_upgrade_button.setEnabled(not busy and installed)
        self.voicestudio_uninstall_button.setEnabled(not busy and installed)
        self.voicestudio_start_button.setEnabled(not busy and installed and not running)
        self.voicestudio_stop_button.setEnabled(not busy and running)
        self.voicestudio_restart_button.setEnabled(not busy and running)
        self.voicestudio_refresh_button.setEnabled(not busy)

    def _on_voicestudio_task_progress(self, message: str, percent: int) -> None:
        self.voicestudio_progress_label.setText(message)
        self.voicestudio_progress_label.show()
        self.voicestudio_progress.show()
        self.voicestudio_task_bar.show()
        for button in self.voicestudio_lifecycle_buttons:
            button.hide()
        self.voicestudio_cancel_task_button.setVisible(self._voicestudio_task_cancellable)
        self.voicestudio_cancel_task_button.setEnabled(
            self._voicestudio_task_cancellable
            and not self._voicestudio_task_cancel_event.is_set()
        )
        if percent < 0:
            self.voicestudio_progress.setRange(0, 0)
            self.voicestudio_progress.setFormat("处理中…")
        else:
            self.voicestudio_progress.setRange(0, 100)
            self.voicestudio_progress.setValue(max(0, min(100, percent)))
            self.voicestudio_progress.setFormat("%p%")

    def _run_voicestudio_task(
        self,
        title: str,
        worker: Callable[[], object],
        *,
        on_success: Callable[[object], None] | None = None,
        success_message: str = "",
        cancellable: bool = False,
    ) -> None:
        if self._voicestudio_task_running:
            QMessageBox.information(self, "VoiceStudio 正忙", "请等待当前安装或管理任务完成。")
            return
        self._voicestudio_task_running = True
        self._voicestudio_task_cancellable = cancellable
        self._voicestudio_task_cancel_event = threading.Event()
        self._voicestudio_task_success = on_success
        self._voicestudio_task_success_message = success_message
        self._on_voicestudio_task_progress(title, -1)
        self._update_voicestudio_action_states()

        def runner() -> None:
            try:
                result = worker()
                if self._voicestudio_task_cancel_event.is_set():
                    raise VoiceStudioTaskCancelled("VoiceStudio 操作已取消。")
            except VoiceStudioTaskCancelled as exc:
                self.signals.voicestudio_task_finished.emit(
                    False,
                    str(exc),
                    {"cancelled": True},
                )
            except Exception as exc:
                self.signals.voicestudio_task_finished.emit(False, str(exc), None)
            else:
                self.signals.voicestudio_task_finished.emit(True, title, result)

        threading.Thread(target=runner, name="voicestudio-management", daemon=True).start()

    def _on_voicestudio_task_finished(self, ok: bool, message: str, result: object) -> None:
        callback = self._voicestudio_task_success
        success_message = self._voicestudio_task_success_message
        cancelled = isinstance(result, dict) and bool(result.get("cancelled"))
        self._voicestudio_task_running = False
        self._voicestudio_task_cancellable = False
        self._voicestudio_task_success = None
        self._voicestudio_task_success_message = ""
        if ok:
            # Queue status probes before an install-complete dialog opens;
            # QMessageBox runs a nested Qt event loop, so these timers still
            # update the cards while the user is reading the confirmation.
            self._schedule_voicestudio_refreshes(include_catalog=True)
            self.voicestudio_progress.setRange(0, 100)
            self.voicestudio_progress.setValue(100)
            self.voicestudio_progress_label.setText(success_message or "操作完成")
            if callback is not None:
                callback(result)
        elif cancelled:
            self.voicestudio_progress.setRange(0, 100)
            self.voicestudio_progress.setValue(0)
            self.voicestudio_progress_label.setText("操作已取消")
        else:
            self.voicestudio_progress.setRange(0, 100)
            self.voicestudio_progress.setValue(0)
            self.voicestudio_progress_label.setText("操作失败")
            QMessageBox.critical(self, "VoiceStudio 管理失败", message)
        self._update_voicestudio_action_states()
        if not ok:
            self._schedule_voicestudio_refreshes(include_catalog=False)
        QTimer.singleShot(1800 if cancelled else 3500, self._hide_voicestudio_task_progress)

    def cancel_voicestudio_task(self) -> None:
        if not self._voicestudio_task_running or not self._voicestudio_task_cancellable:
            return
        self._voicestudio_task_cancel_event.set()
        self.voicestudio_cancel_task_button.setEnabled(False)
        self.voicestudio_progress_label.setText("正在取消，请等待当前步骤安全结束…")

    def _schedule_voicestudio_refreshes(self, *, include_catalog: bool = False) -> None:
        """Refresh again after MSI/child-process state has settled on Windows."""
        for delay in (0, 1500, 4000, 8000):
            QTimer.singleShot(
                delay,
                lambda latest=delay == 0: self.refresh_voicestudio_runtime(
                    include_latest=latest
                ),
            )
        if include_catalog:
            for delay in (2200, 5500, 9000):
                QTimer.singleShot(delay, self.refresh_voicestudio_catalog)

    def _hide_voicestudio_task_progress(self) -> None:
        if self._voicestudio_task_running:
            return
        self.voicestudio_task_bar.hide()
        self.voicestudio_progress.hide()
        self.voicestudio_progress_label.hide()
        self.voicestudio_cancel_task_button.hide()
        for button in self.voicestudio_lifecycle_buttons:
            button.show()
        self._update_voicestudio_action_states()

    def install_voicestudio(self) -> None:
        snapshot = getattr(self, "_last_voicestudio_runtime", None)
        if snapshot and snapshot.installation.installed:
            choice = QMessageBox.question(
                self,
                "VoiceStudio 已安装",
                "已经检测到 VoiceStudio。继续会用 GitHub 最新 MSI 执行覆盖安装，是否继续？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if choice != QMessageBox.Yes:
                return
        install_dir = self.voicestudio_install_dir.text().strip()
        self._run_voicestudio_task(
            "正在准备自动下载安装…",
            lambda: self.voicestudio_manager.install(
                install_dir,
                self.signals.voicestudio_task_progress.emit,
                self._voicestudio_task_cancel_event,
            ),
            on_success=self._after_voicestudio_install,
            success_message="VoiceStudio 安装完成",
            cancellable=True,
        )

    def _after_voicestudio_install(self, result: object) -> None:
        executable = str(getattr(result, "executable", "") or "")
        install_dir = str(getattr(result, "install_dir", "") or "")
        if executable:
            self.voicestudio_executable.setText(executable)
        if install_dir:
            self.voicestudio_install_dir.setText(install_dir)
        self.settings.update(self.current_settings())
        QMessageBox.information(
            self,
            "安装完成",
            "VoiceStudio 已安装。首次启动会继续准备本地 Python 环境和模型，请按它的向导完成。",
        )

    def import_voicestudio_installer(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 VoiceStudio MSI 安装包",
            str(Path.home() / "Downloads"),
            "Windows Installer (*.msi)",
        )
        if not path:
            return
        installer = Path(path)

        def verify() -> object:
            actual = self.voicestudio_manager.file_sha256(
                installer,
                self.signals.voicestudio_task_progress.emit,
                self._voicestudio_task_cancel_event,
            )
            try:
                release = self.voicestudio_manager.latest_release(force=True)
                matched = (
                    actual == release.expected_sha256.lower()
                    if release.expected_sha256
                    else None
                )
                error = ""
            except Exception as exc:
                release = None
                matched = None
                error = str(exc)
            return {
                "installer": installer,
                "actual": actual,
                "release": release,
                "matched": matched,
                "error": error,
            }

        self._run_voicestudio_task(
            "正在校验导入的安装包…",
            verify,
            on_success=self._confirm_manual_voicestudio_installer,
            success_message="安装包指纹计算完成",
            cancellable=True,
        )

    def _confirm_manual_voicestudio_installer(self, raw_result: object) -> None:
        result = raw_result if isinstance(raw_result, dict) else {}
        installer = result.get("installer")
        actual = str(result.get("actual") or "")
        release = result.get("release")
        matched = result.get("matched")
        error = str(result.get("error") or "")
        if not isinstance(installer, Path):
            return
        box = QMessageBox(self)
        box.setWindowTitle("VoiceStudio 安装包校验")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.button(QMessageBox.Yes).setText("仍然安装" if matched is not True else "安装")
        box.button(QMessageBox.Cancel).setText("取消")
        if matched is True and isinstance(release, VoiceStudioRelease):
            box.setIcon(QMessageBox.Information)
            box.setText(
                "<b><font color='#158553'>SHA-256 与 GitHub 官方最新版本一致</font></b><br>"
                f"官方版本：{release.tag}<br>现在安装吗？"
            )
        elif matched is False and isinstance(release, VoiceStudioRelease):
            box.setIcon(QMessageBox.Critical)
            box.setText(
                "<b><font color='#cf3535'>红色警告：SHA-256 与 GitHub 官方最新版本不一致</font></b><br>"
                "该文件可能是旧版本、被重新打包或已经损坏。只有确认来源可信时才继续。"
            )
        else:
            box.setIcon(QMessageBox.Warning)
            box.setText(
                "<b>暂时无法取得 GitHub 官方 SHA-256，无法确认文件一致性。</b><br>"
                "只有确认安装包来自官方发布页时才继续。"
            )
        official_hash = (
            release.expected_sha256
            if isinstance(release, VoiceStudioRelease) and release.expected_sha256
            else "GitHub 未返回或暂时无法读取"
        )
        latest = release.tag if isinstance(release, VoiceStudioRelease) else "未知"
        box.setDetailedText(
            f"文件：{installer}\nGitHub 最新版本：{latest}\n"
            f"本地 SHA-256：{actual}\n官方 SHA-256：{official_hash}"
            + (f"\n读取错误：{error}" if error else "")
        )
        if box.exec() != QMessageBox.Yes:
            return
        version_label = release.tag if isinstance(release, VoiceStudioRelease) and matched is True else "手动导入包"
        self._run_voicestudio_task(
            "正在启动 Windows Installer…",
            lambda: self.voicestudio_manager.install_package(
                installer,
                self.voicestudio_install_dir.text().strip(),
                self.signals.voicestudio_task_progress.emit,
                version_label,
                self._voicestudio_task_cancel_event,
            ),
            on_success=self._after_voicestudio_install,
            success_message="手动导入安装完成",
            cancellable=True,
        )

    def upgrade_voicestudio(self) -> None:
        if QMessageBox.question(
            self,
            "升级 VoiceStudio",
            "升级前会先关闭 VoiceStudio 的桌面程序和全部后台子进程，然后下载 GitHub 最新 MSI。继续吗？",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        self._run_voicestudio_task(
            "正在准备升级…",
            lambda: self.voicestudio_manager.upgrade(
                self.voicestudio_install_dir.text().strip(),
                self.signals.voicestudio_task_progress.emit,
                lambda text: self.signals.voicestudio_task_progress.emit(text, -1),
                self._voicestudio_task_cancel_event,
            ),
            on_success=self._after_voicestudio_install,
            success_message="VoiceStudio 已升级到最新版本",
            cancellable=True,
        )

    def uninstall_voicestudio(self) -> None:
        if QMessageBox.warning(
            self,
            "卸载 VoiceStudio",
            "将关闭所有 VoiceStudio 进程并卸载程序。声音、模型、项目和生成记录默认保留，"
            "如需清除数据请在 VoiceStudio 的“设置 → 存储”中单独执行。\n\n确定卸载吗？",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self._run_voicestudio_task(
            "正在准备卸载…",
            lambda: self.voicestudio_manager.uninstall(
                self.signals.voicestudio_task_progress.emit,
                lambda text: self.signals.voicestudio_task_progress.emit(text, -1),
                self._voicestudio_task_cancel_event,
            ),
            success_message="VoiceStudio 程序已卸载，用户数据已保留",
            cancellable=True,
        )

    def launch_voicestudio(self) -> None:
        def started(result: object) -> None:
            self.settings.update(self.current_settings())
            self.voicestudio_status.setText(f"VoiceStudio 正在启动 · PID {result}")
            QTimer.singleShot(2500, self.refresh_voicestudio_runtime)
            QTimer.singleShot(5000, self.refresh_voicestudio_catalog)

        self._run_voicestudio_task(
            "正在启动 VoiceStudio…",
            lambda: self.voicestudio_manager.start(
                self.voicestudio_executable.text().strip(),
                self.voicestudio_install_dir.text().strip(),
            ),
            on_success=started,
            success_message="VoiceStudio 已启动，正在等待本地 API 就绪",
        )

    def stop_voicestudio(self) -> None:
        self._run_voicestudio_task(
            "正在停止 VoiceStudio…",
            lambda: self.voicestudio_manager.stop_all(
                lambda text: self.signals.voicestudio_task_progress.emit(text, -1),
                managed_only=False,
            ),
            success_message="VoiceStudio 桌面程序和后台进程已停止",
        )

    def restart_voicestudio(self) -> None:
        self._run_voicestudio_task(
            "正在重启 VoiceStudio…",
            lambda: self.voicestudio_manager.restart(
                self.voicestudio_executable.text().strip(),
                self.voicestudio_install_dir.text().strip(),
                lambda text: self.signals.voicestudio_task_progress.emit(text, -1),
            ),
            on_success=lambda _result: QTimer.singleShot(5000, self.refresh_voicestudio_catalog),
            success_message="VoiceStudio 已重新启动",
        )

    def preview_voicestudio_voice(self) -> None:
        text = self.voicestudio_preview_text.toPlainText().strip()
        if not text:
            self.show_error("请先输入一小段试听文字。")
            return
        if not self._can_begin("speaking"):
            return
        values = self.current_settings()
        values["tts_provider"] = "voicestudio"
        values["tts_rate"] = self.voicestudio_preview_rate.value() / 100.0
        values["tts_volume"] = self.voicestudio_preview_volume.value()
        values["tts_language_hint"] = (
            "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"
        )
        self.cancel_event.clear()
        self._set_status("VoiceStudio 正在本机合成试听…")

        def worker() -> None:
            self.tts_active.set()
            try:
                device = values.get("monitor_output_device")
                if device is None:
                    device = values.get("teams_output_device")
                with MultiOutputPlayer([device], 24_000) as player:
                    count = self._make_tts_client(values).stream_tts(
                        text, values, player.write, self.cancel_event
                    )
                if count == 0 and not self.cancel_event.is_set():
                    raise RuntimeError("VoiceStudio 没有返回试听音频")
            except Exception as exc:
                self.signals.error.emit(str(exc))
            finally:
                self.tts_active.clear()
                self.signals.tts_finished.emit()

        threading.Thread(target=worker, name="voicestudio-preview", daemon=True).start()

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
        asr: QwenRealtimeASR | FunASRRealtime | None = None
        message = "老师字幕已停止"
        try:
            client = self._make_client(values)

            def completed_segment(text: str, _emotion: str) -> None:
                if text.strip() and self.teacher_segment_queue is not None:
                    self.teacher_segment_queue.put(text.strip())

            asr = create_realtime_asr(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                values=values,
                language="en",
                on_preview=lambda text, _emotion: self.signals.teacher_preview.emit(text),
                on_status=lambda _status: None,
                on_error=lambda error: log.warning("Teacher ASR: %s", error),
                on_segment=completed_segment,
                combine_previews=False,
                proxy=values["http_proxy"],
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
            self._set_status(self._ready_status())

    def _direct_caption_worker(self, values: dict[str, Any], started_at: float) -> None:
        asr: QwenRealtimeASR | FunASRRealtime | None = None
        message = ""
        try:
            client = self._make_client(values)
            asr = create_realtime_asr(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                values=values,
                language=values["asr_language"],
                on_preview=lambda text, emotion: self.signals.asr_preview.emit(text, emotion),
                on_status=lambda _status: None,
                on_error=lambda error: log.warning("Direct ASR: %s", error),
                proxy=values["http_proxy"],
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
                f"我 / {str(values['direct_hotkey']).upper()} 原声",
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

    def start_continuous_translation(self) -> None:
        values = self.current_settings()
        if values.get("translation_engine") != "live":
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error(
                f"{self._hotkey_name('translate_hotkey')} 持续翻译仅支持“极速直译”引擎。"
            )
            return
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先在设置页填写 Workspace ID 和 API Key。")
            return
        if not self._can_begin("continuous"):
            return

        self._discard_manual_live_session(graceful=True)
        self.cancel_event.clear()
        self.continuous_stop_event.clear()
        self.switch_to_direct_after_continuous = False
        self.awaiting_confirmation = False
        self.translation_queue = queue.Queue(maxsize=400)
        self.chinese_text.clear()
        self.english_text.clear()
        self.overlay.set_texts("", "")

        def enqueue(chunk: bytes) -> None:
            if self.translation_queue is None:
                return
            try:
                self.translation_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self.translation_queue.get_nowait()
                    self.translation_queue.put_nowait(chunk)
                except (queue.Empty, queue.Full):
                    pass

        try:
            self.translation_capture = MicrophoneCapture(
                values["input_device"], enqueue, sample_rate=16000, block_ms=100
            )
            self.translation_capture.start()
        except Exception as exc:
            self.translation_capture = None
            self.translation_queue = None
            self._set_idle()
            self.show_error(f"持续翻译麦克风启动失败：{exc}")
            return

        self.translate_button.setProperty("active", True)
        self.translate_button.style().unpolish(self.translate_button)
        self.translate_button.style().polish(self.translate_button)
        self._set_status("F9 持续翻译已开启 · 直接说中文，停顿后自动翻译")
        self.pipeline_thread = threading.Thread(
            target=self._continuous_translation_worker,
            args=(values,),
            name="continuous-live-translation",
            daemon=True,
        )
        self.pipeline_thread.start()

    def stop_continuous_translation(self) -> None:
        with self.state_lock:
            if self.state == "continuous":
                self.state = "continuous_stopping"
            elif self.state != "continuous_stopping":
                return
        self.continuous_stop_event.set()
        if self.translation_capture is not None:
            self.translation_capture.stop()
            self.translation_capture = None
        if self.translation_queue is not None:
            try:
                self.translation_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.translation_queue.get_nowait()
                    self.translation_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
        self._set_status("正在结束 F9 持续翻译并处理最后一句…")

    def _continuous_translation_worker(self, values: dict[str, Any]) -> None:
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
        result_count = 0
        message = "F9 持续翻译已关闭"
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
                    name="continuous-live-audio-output",
                    daemon=True,
                )
                playback_thread.start()

                def handle_result(result) -> None:
                    nonlocal result_count
                    result_count += 1
                    self.last_translation = result.translated_text
                    self.signals.translation_ready.emit(
                        result.source_text,
                        result.translated_text,
                        result.first_audio_seconds or 0.0,
                        "我 / F9 持续直译",
                    )
                    if values["live_voice_clone_mode"] == "once" and result_count == 1:
                        self.signals.status.emit(
                            "第 1 段翻译完成 · 音色校准已建立，后续句子会复用你的音色"
                        )
                    else:
                        self.signals.status.emit(
                            f"F9 持续翻译中 · 已完成 {result_count} 句，继续说即可"
                        )

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
                    continuous=True,
                    vad_threshold=float(values["vad_threshold"]),
                    vad_silence_ms=int(values["vad_silence_ms"]),
                    on_source_preview=lambda text: self.signals.asr_preview.emit(text, "live"),
                    on_translation_preview=self.signals.live_translation_preview.emit,
                    on_audio=audio_queue.put,
                    on_result=handle_result,
                    on_status=self.signals.status.emit,
                    on_error=lambda error: log.warning("Continuous LiveTranslate: %s", error),
                    proxy=values["http_proxy"],
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
                    session.close()
                else:
                    session.finish(timeout=max(15.0, float(values["request_timeout"])))
                session = None
                audio_queue.put(None)
                audio_queue.join()
                playback_thread.join(timeout=1.0)
                if playback_errors:
                    raise ApiError(f"持续翻译音频播放失败：{playback_errors[0]}")
        except Exception as exc:
            log.exception("Continuous LiveTranslate failed")
            message = f"F9 持续翻译已停止：{exc}"
            if not self.cancel_event.is_set():
                self.signals.error.emit(str(exc))
        finally:
            if session is not None:
                session.close()
            if playback_thread is not None and playback_thread.is_alive():
                audio_queue.put(None)
                audio_queue.join()
                playback_thread.join(timeout=1.0)
            self.tts_active.clear()
            self.signals.continuous_finished.emit(message)

    def on_continuous_translation_finished(self, message: str) -> None:
        self.translation_capture = None
        self.translation_queue = None
        self.continuous_stop_event.clear()
        self.translate_button.setProperty("active", False)
        self.translate_button.style().unpolish(self.translate_button)
        self.translate_button.style().polish(self.translate_button)
        self._set_idle()
        should_start_direct = self.switch_to_direct_after_continuous and self.direct_key_held
        self.switch_to_direct_after_continuous = False
        if should_start_direct:
            self.start_direct()
        elif not self.cancel_event.is_set():
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
        asr: QwenRealtimeASR | FunASRRealtime | None = None
        try:
            if values.get("translation_engine") == "live":
                self._live_translation_worker(values, started_at)
                return
            client = self._make_client(values)
            asr = create_realtime_asr(
                api_key=client.api_key,
                workspace_id=client.workspace_id,
                values=values,
                language=values["asr_language"],
                on_preview=lambda text, emotion: self.signals.asr_preview.emit(text, emotion),
                on_status=self.signals.status.emit,
                on_error=lambda error: log.warning("ASR: %s", error),
                proxy=values["http_proxy"],
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
                self._stream_speech(self._make_tts_client(values), english, values)
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
        keep_session = False
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
                session = self._get_manual_live_session(values, client, phrases, audio_queue.put)
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
                keep_session = True

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
                    if values["live_voice_clone_mode"] == "once" and session.completed_turns == 1:
                        self.signals.status.emit(
                            "第 1 段翻译完成 · 音色校准已建立；下一次 F9 会复用你的音色"
                        )
                    else:
                        self.signals.status.emit(
                            f"极速直译完成 · 松开 F9 后 {result.first_audio_seconds:.2f}s 开始出声"
                        )
        finally:
            if session is not None and not keep_session:
                self._discard_manual_live_session(session)
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

    def _stream_speech(self, client: Any, english: str, values: dict[str, Any]) -> None:
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
                self._stream_speech(self._make_tts_client(values), text, values)
            except Exception as exc:
                self.signals.error.emit(str(exc))
            finally:
                self.signals.tts_finished.emit()

        threading.Thread(target=worker, name="manual-tts", daemon=True).start()

    @staticmethod
    def split_long_text(text: str, max_len: int = 280) -> list[str]:
        """Compatibility wrapper for the context-aware sentence splitter."""
        return split_source_sentences(text, max_len=max_len)

    def start_long_text_speech(self) -> None:
        text = self.long_text_input.toPlainText().strip()
        if not text:
            self.show_error("请先在长文本框里粘贴要朗读的内容。")
            return
        values = self.current_settings()
        if not self.settings.get_api_key() or not values["workspace_id"]:
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先在设置页填写 Workspace ID 和 API Key。")
            return
        sentences = split_source_sentences(text)
        if not sentences:
            self.show_error("没有可分句朗读的有效内容。")
            return
        self.long_text_progress.setMaximum(len(sentences))
        self.long_text_progress.setValue(0)
        self.long_text_play.setEnabled(False)
        self.long_text_pause.setEnabled(False)
        self.long_text_stop.setEnabled(False)
        self.long_text_status.setText("正在分析全文语境与术语…")
        if self.long_form_prepare_dialog is not None:
            self.long_form_prepare_dialog.blockSignals(True)
            self.long_form_prepare_dialog.close()
            self.long_form_prepare_dialog = None
        self.long_form_prepare_generation += 1
        generation = self.long_form_prepare_generation
        self.long_form_prepare_cancel_event.set()
        self.long_form_prepare_cancel_event = threading.Event()
        values["_long_form_prepare_generation"] = generation
        self.long_form_prepare_dialog = QProgressDialog(
            "正在通读全文、统一专有术语并生成逐句英文…",
            "取消",
            0,
            0,
            self,
        )
        self.long_form_prepare_dialog.setWindowTitle("准备双语整段朗读")
        self.long_form_prepare_dialog.setAutoClose(False)
        self.long_form_prepare_dialog.setMinimumDuration(0)
        self.long_form_prepare_dialog.resize(520, 120)
        self.long_form_prepare_dialog.canceled.connect(self._cancel_long_form_prepare)
        self.long_form_prepare_dialog.show()
        self.long_text_thread = threading.Thread(
            target=self._prepare_long_form_reader,
            args=(text, sentences, values, generation, self.long_form_prepare_cancel_event),
            name="long-form-prepare",
            daemon=True,
        )
        self.long_text_thread.start()

    def _prepare_long_form_reader(
        self,
        full_text: str,
        sentences: list[str],
        values: dict[str, Any],
        generation: int,
        cancel_event: threading.Event,
    ) -> None:
        try:
            client = self._make_client(values)
            english_sentences = client.translate_long_form(
                full_text,
                sentences,
                values,
                cancel_event=cancel_event,
                on_progress=lambda message: self.signals.long_form_progress.emit(
                    generation, message
                ),
            )
            if cancel_event.is_set() or generation != self.long_form_prepare_generation:
                return
            segments = [
                BilingualSegment(
                    chinese=chinese,
                    english=english,
                    estimated_seconds=estimate_speech_seconds(
                        english,
                        float(values.get("tts_rate", 1.0)),
                    ),
                )
                for chinese, english in zip(sentences, english_sentences, strict=True)
            ]
            self.signals.long_form_ready.emit(segments, values)
        except Exception as exc:
            if cancel_event.is_set() or generation != self.long_form_prepare_generation:
                return
            self.signals.error.emit(str(exc))
            self.signals.long_text_state.emit("prepare_failed")

    def _cancel_long_form_prepare(self) -> None:
        self.long_form_prepare_cancel_event.set()
        self.long_form_prepare_generation += 1
        if self.long_form_prepare_dialog is not None:
            self.long_form_prepare_dialog.blockSignals(True)
            self.long_form_prepare_dialog.close()
            self.long_form_prepare_dialog = None
        self.long_text_play.setEnabled(True)
        self.long_text_status.setText("已取消整段翻译准备")

    def _on_long_form_progress(self, generation: int, message: str) -> None:
        if generation != self.long_form_prepare_generation:
            return
        self.long_text_status.setText(message)
        if self.long_form_prepare_dialog is not None:
            self.long_form_prepare_dialog.setLabelText(message)

    def _open_long_form_reader(
        self,
        raw_segments: object,
        raw_values: object,
    ) -> None:
        segments = list(raw_segments) if isinstance(raw_segments, list) else []
        values = dict(raw_values) if isinstance(raw_values, dict) else self.current_settings()
        if int(values.get("_long_form_prepare_generation", -1)) != self.long_form_prepare_generation:
            return
        if not segments:
            self._on_long_text_state("prepare_failed")
            return
        if self.long_form_prepare_dialog is not None:
            self.long_form_prepare_dialog.blockSignals(True)
            self.long_form_prepare_dialog.close()
            self.long_form_prepare_dialog = None
        if self.long_form_dialog is not None:
            self.long_form_dialog.close()
        dialog = LongFormReaderDialog(
            segments,
            client=self._make_client(values),
            tts_client=self._make_tts_client(values),
            settings=values,
            parent=self,
        )
        self.long_form_dialog = dialog
        dialog.error.connect(self.show_error)
        dialog.pronunciations_changed.connect(self.tts_pronunciations.setPlainText)
        dialog.sentence_changed.connect(self._on_long_text_progress)
        dialog.finished.connect(self._long_form_dialog_closed)
        self.long_text_progress.setMaximum(len(segments))
        self.long_text_progress.setValue(0)
        self.long_text_play.setEnabled(True)
        self.long_text_status.setText("双语朗读窗口已打开")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.start()

    def _long_form_dialog_closed(self, _result: int) -> None:
        self.long_form_dialog = None
        self.long_text_play.setEnabled(True)
        self.long_text_status.setText("朗读窗口已关闭")

    def pause_or_resume_long_text(self) -> None:
        if self.long_form_dialog is not None:
            self.long_form_dialog.toggle_pause()

    def stop_long_text_speech(self) -> None:
        if self.long_form_dialog is not None:
            self.long_form_dialog.stop()
        self.long_text_play.setEnabled(True)
        self.long_text_pause.setEnabled(False)
        self.long_text_pause.setText("⏸ 暂停")
        self.long_text_stop.setEnabled(False)
        self.long_text_status.setText("已停止")

    def _on_long_text_progress(self, done: int, total: int) -> None:
        self.long_text_progress.setMaximum(total)
        self.long_text_progress.setValue(done)
        self.long_text_status.setText(f"{done}/{total} 句")

    def _on_long_text_state(self, state: str) -> None:
        if state == "finished":
            self.long_text_play.setEnabled(True)
            self.long_text_pause.setEnabled(False)
            self.long_text_pause.setText("⏸ 暂停")
            self.long_text_stop.setEnabled(False)
            self.long_text_status.setText("完成")
        elif state == "playing":
            self.long_text_status.setText("朗读中…")
        elif state == "prepare_failed":
            if self.long_form_prepare_dialog is not None:
                self.long_form_prepare_dialog.blockSignals(True)
                self.long_form_prepare_dialog.close()
                self.long_form_prepare_dialog = None
            self.long_text_play.setEnabled(True)
            self.long_text_pause.setEnabled(False)
            self.long_text_stop.setEnabled(False)
            self.long_text_status.setText("整段翻译准备失败")
        else:
            self.long_text_status.setText(state)

    def _replay_history_row(self, row: int) -> None:
        english_item = self.history.item(row, 3)
        text = english_item.text() if english_item else ""
        if not text:
            chinese_item = self.history.item(row, 2)
            text = chinese_item.text() if chinese_item else ""
        if not text:
            self.show_error("该句没有可重播的内容。")
            return
        self.english_text.setPlainText(text)
        self.play_current_english()

    def send_typed_text(self) -> None:
        text = self.typed_input.toPlainText().strip()
        if not text:
            self.show_error("请先在键盘输入框中填写要说的内容。")
            return
        if not self._can_begin("typed"):
            self._set_status("当前任务尚未结束，请稍候或按 Esc 取消")
            return
        values = self.current_settings()
        mode = self.typed_mode.currentData()
        needs_bailian = mode == "translate" or values.get("tts_provider") != "voicestudio"
        if needs_bailian and (not self.settings.get_api_key() or not values["workspace_id"]):
            self._set_idle()
            self.tabs.setCurrentWidget(self.settings_tab)
            self.show_error("请先在设置页填写 Workspace ID 和 API Key。")
            return
        self.cancel_event.clear()
        self.awaiting_confirmation = False
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
            if mode == "translate":
                client = self._make_client(values)
                spoken = client.translate(text, values)
                self.awaiting_confirmation = bool(values["confirm_before_speak"])
                self.signals.translation_ready.emit(
                    text,
                    spoken,
                    time.perf_counter() - started_at,
                    "我 / 键盘翻译",
                )
                if not self.awaiting_confirmation and not self.cancel_event.is_set():
                    self._stream_speech(self._make_tts_client(values), spoken, values)
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
                self._stream_speech(self._make_tts_client(direct_values), text, direct_values)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.tts_finished.emit()

    def cancel_all(self) -> None:
        self.cancel_event.set()
        self.teacher_cancel_event.set()
        self.switch_to_direct_after_continuous = False
        self.direct_key_held = False
        self.continuous_stop_event.set()
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
        self._discard_manual_live_session()
        if self.teacher_active:
            self.stop_teacher_caption()
        self._set_idle()
        self._set_status(
            f"已停止 · {self._hotkey_name('direct_hotkey')} 原声 / "
            f"{self._hotkey_name('translate_hotkey')} 翻译"
        )

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
        replay_button = QPushButton("▶ 重播")
        replay_button.setObjectName("historyReplayButton")
        replay_button.setToolTip("重新合成并播放这一句话")
        replay_button.setMinimumWidth(68)
        replay_button.setMinimumHeight(24)
        replay_button.clicked.connect(lambda _checked, r=row: self._replay_history_row(r))
        self.history.setCellWidget(row, 5, replay_button)
        needed = replay_button.sizeHint().width() + 18
        if self.history.columnWidth(5) < needed:
            self.history.setColumnWidth(5, needed)
        self.history.scrollToBottom()

    def on_tts_finished(self) -> None:
        self._set_idle()
        if not self.cancel_event.is_set():
            if self.awaiting_confirmation and self.english_text.toPlainText().strip():
                self._set_status("译文已生成 · 修改后点击“播放/发送当前英文”")
            elif self.teacher_active:
                self._set_status("正在听老师 · F8/F9 仍可随时使用")
            else:
                self._set_status(self._ready_status())

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
                        proxy=values["http_proxy"],
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
        if self.clone_target_model.startswith("qwen3.5-livetranslate"):
            choice = QMessageBox.question(
                self,
                "实验性固定音色",
                "百炼当前可能拒绝为 Qwen3.5 LiveTranslate 预创建固定音色，并返回 "
                "“preprocess service not found”。\n\n"
                "推荐取消，然后选择“服务端复刻一次”：无需样音，直接按 F9 说话即可自动复刻。\n\n"
                "仍要尝试固定音色创建吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                self._select_data(self.live_voice_clone_mode, "once")
                self.save_settings_from_ui()
                return
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
        qwen3 = is_qwen3_tts_vc_model(str(options.get("target_model", "")))
        oss_config = (
            self.settings.get_oss_config()
            if options.get("audio_path") and not qwen3
            else None
        )

        def worker() -> None:
            uploader: OssTemporaryUploader | None = None
            uploaded: UploadedVoiceSample | None = None
            cleanup_warning = ""
            voice_id = ""
            preserve_upload = False
            error = ""
            try:
                request_options = dict(options)
                local_path = str(request_options.pop("audio_path", "") or "").strip()
                transcript = str(request_options.pop("transcript", "") or "").strip()
                client = self._make_client(values)
                if local_path:
                    self.signals.status.emit("正在把本地样音转换为标准 WAV…")
                    with normalized_voice_sample(
                        local_path,
                        max_seconds=float(request_options["max_seconds"]),
                    ) as normalized_path:
                        if qwen3:
                            self.signals.status.emit("正在把标准 WAV 直传百炼并创建 Qwen3 专属音色…")
                            data_url = (
                                "data:audio/wav;base64,"
                                + base64.b64encode(normalized_path.read_bytes()).decode("ascii")
                            )
                            voice_id, qwen_warning = client.clone_qwen_voice(
                                target_model=request_options["target_model"],
                                preferred_name=request_options["prefix"],
                                audio_data=data_url,
                                language=request_options["language"],
                                transcript=transcript,
                            )
                            cleanup_warning = qwen_warning
                        else:
                            self.signals.status.emit("正在将标准 WAV 临时上传到 OSS…")
                            assert oss_config is not None
                            uploader = OssTemporaryUploader(**oss_config)
                            uploaded = uploader.upload(normalized_path)
                            request_options["audio_url"] = uploaded.signed_url
                            self.signals.status.emit("临时样音已上传 · 正在通过百炼创建固定音色…")
                            voice_id = client.clone_voice(**request_options)
                else:
                    if qwen3:
                        voice_id, qwen_warning = client.clone_qwen_voice(
                            target_model=request_options["target_model"],
                            preferred_name=request_options["prefix"],
                            audio_data=request_options["audio_url"],
                            language=request_options["language"],
                            transcript=transcript,
                        )
                        cleanup_warning = qwen_warning
                    else:
                        voice_id = client.clone_voice(**request_options)
                if voice_id.startswith(("qwen-audio-", "cosyvoice-")):
                    client.wait_for_voice_ready(
                        voice_id,
                        timeout=120.0,
                        on_status=self.signals.status.emit,
                    )
            except Exception as exc:
                error = str(exc)
                preserve_upload = bool(voice_id and "DEPLOYING" in error)
            finally:
                if uploader is not None and uploaded is not None and not preserve_upload:
                    try:
                        uploader.delete(uploaded.key)
                    except Exception as exc:
                        log.warning("Unable to delete temporary OSS voice sample %s: %s", uploaded.key, exc)
                        cleanup_warning = (
                            "OSS 临时样音删除失败，请手动删除对象："
                            f"{uploaded.key}\n错误：{exc}"
                        )
                elif uploader is not None and uploaded is not None:
                    cleanup_warning = (
                        "音色尚未确认可用，程序为避免中断百炼处理，已保留 OSS 临时样音："
                        f"{uploaded.key}。确认音色状态后可手动删除。"
                    )
            self.signals.clone_finished.emit(not bool(error), voice_id if not error else error, cleanup_warning)

        threading.Thread(target=worker, name="voice-clone", daemon=True).start()

    def on_clone_finished(self, ok: bool, message: str, cleanup_warning: str) -> None:
        if ok:
            if self.clone_target_model.startswith("qwen3.5-livetranslate"):
                self.live_voice.setText(message)
                self._select_data(self.live_voice_clone_mode, "fixed")
            else:
                target = self.clone_target_model
                voices = dict(self.settings.get("tts_voices", {}) or {})
                voices[target] = message
                self.settings.values["tts_voices"] = voices
                if is_qwen3_tts_vc_model(target):
                    self.tts_model.setCurrentText(target)
                elif message.startswith("qwen-audio-3.0-tts-plus-"):
                    self.tts_model.setCurrentText("qwen-audio-3.0-tts-plus")
                elif message.startswith("qwen-audio-3.0-tts-flash-"):
                    self.tts_model.setCurrentText("qwen-audio-3.0-tts-flash")
                elif is_cosyvoice_model(target):
                    self.tts_model.setCurrentText(target)
                self.voice.setText(message)
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
            if (
                self.clone_target_model.startswith("qwen3.5-livetranslate")
                and "preprocess service not found" in message.lower()
            ):
                self._select_data(self.live_voice_clone_mode, "once")
                self.live_voice.clear()
                self.save_settings_from_ui()
                if self.clone_dialog is not None:
                    self.clone_dialog.reject()
                detail = (
                    "百炼当前没有为 Qwen3.5 LiveTranslate 提供可用的预创建固定音色处理服务。\n\n"
                    "程序已自动切换为“服务端复刻一次”。现在无需上传录音，回到会议控制台后"
                    "直接按住 F9 说话；第一段建立音色校准，第二段及后续会在同一连接中复用。"
                )
                if cleanup_warning:
                    detail += f"\n\n注意：{cleanup_warning}"
                QMessageBox.information(self, "已切换到可用的声音复刻方式", detail)
                self._set_status("已切换为服务端复刻一次 · 直接按 F9 说话")
                return
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
        if self._shutdown_in_progress:
            event.ignore()
            return
        if self._voicestudio_task_running:
            event.ignore()
            QMessageBox.information(
                self,
                "VoiceStudio 操作尚未完成",
                "安装、升级或进程管理仍在进行。请等待进度条完成后再关闭软件。",
            )
            return
        if not self._subtitle_close_checked and self.subtitle_session.records and not self.subtitle_exported:
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
        self._subtitle_close_checked = True
        if not self._shutdown_authorized and self.voicestudio_manager.has_managed_processes():
            event.ignore()
            self._begin_voicestudio_shutdown()
            return
        self._finish_close_cleanup()
        event.accept()

    def _begin_voicestudio_shutdown(self) -> None:
        self._shutdown_in_progress = True
        self.voicestudio_monitor_timer.stop()
        self._update_voicestudio_action_states()
        dialog = QProgressDialog("正在检查 VoiceStudio 后台进程…", "", 0, 0, self)
        dialog.setWindowTitle("正在安全关闭")
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(520)
        dialog.show()
        self._shutdown_dialog = dialog

        def worker() -> None:
            error = ""
            try:
                self.voicestudio_manager.stop_all(
                    self.signals.voicestudio_shutdown_progress.emit,
                    managed_only=True,
                )
            except Exception as exc:
                error = str(exc)
            self.signals.voicestudio_shutdown_finished.emit(error)

        threading.Thread(target=worker, name="voicestudio-app-shutdown", daemon=True).start()

    def _on_voicestudio_shutdown_progress(self, message: str) -> None:
        if self._shutdown_dialog is not None:
            self._shutdown_dialog.setLabelText(message)

    def _on_voicestudio_shutdown_finished(self, error: str) -> None:
        if self._shutdown_dialog is not None:
            if error:
                self._shutdown_dialog.setLabelText(
                    f"后台进程清理出现问题：{error}\n正在继续关闭当前软件…"
                )
            else:
                self._shutdown_dialog.setLabelText("VoiceStudio 后台进程已全部关闭，正在退出当前软件…")
        self._shutdown_authorized = True
        self._shutdown_in_progress = False
        QTimer.singleShot(350, self.close)

    def _finish_close_cleanup(self) -> None:
        self.voicestudio_monitor_timer.stop()
        self.cancel_all()
        self.long_form_prepare_cancel_event.set()
        if self.long_form_dialog is not None:
            self.long_form_dialog.close()
        self.stop_recording()
        self.overlay.close()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.subtitle_session.cleanup()
        if self._shutdown_dialog is not None:
            self._shutdown_dialog.close()
            self._shutdown_dialog = None


STYLE_SHEET = """
QMainWindow, QWidget { background: #f1f4f9; color: #1f2a44; font-family: "Microsoft YaHei UI"; font-size: 12px; }
QLabel#title { font-size: 21px; font-weight: 800; color: #0f2742; }
QLabel#subtitle { color: #5d6b83; font-size: 11px; letter-spacing: 1px; }
QLabel#brandLogo { background: transparent; }
QFrame#appHeader { background: rgba(255,255,255,215); border: none; border-bottom: 1px solid #e3e9f2; }
QFrame#mainNavigation { background: rgba(255,255,255,235); border: 1px solid #e3e9f2; border-radius: 12px; }
QLabel#mainNavBrand { color: #0f2742; font-size: 18px; font-weight: 800; letter-spacing: 2px; }
QLabel#mainNavHealth { background: #ecf8f1; color: #188a52; border: 1px solid #bde5cd; border-radius: 8px; padding: 10px; font-weight: 700; }
QLabel#mainNavHealth[runtimeState="error"] { background: #fdecec; color: #c0392b; border-color: #f2b8b8; }
QLabel#mainNavHealth[runtimeState="stopped"] { background: #f2f4f7; color: #6b7688; border-color: #dde3ec; }
QPushButton#mainNavButton { background: transparent; border: 1px solid transparent; text-align: left; font-size: 14px; padding: 10px 13px; border-radius: 8px; }
QPushButton#mainNavButton:hover { background: #eef4fd; border-color: #d5e4f8; }
QPushButton#mainNavButton:checked { background: #e3edfe; color: #1d4ed8; border-color: #b3c9f5; font-weight: 800; }
QFrame#appFooter { background: rgba(255,255,255,185); border-top: 1px solid #e3e9f2; }
QLabel#footerStatus { color: #188a52; font-weight: 700; }
QLabel#statusBadge { background: #e9f1fe; color: #1d4ed8; border: 1px solid #bcd3fb; border-radius: 14px; padding: 7px 13px; font-weight: 700; }
QLabel#voiceStudioStatus { background: #fdf3d8; color: #96650a; border: 1px solid #f2d894; border-radius: 12px; padding: 6px 11px; font-weight: 700; }
QLabel#voiceStudioStatus[connected="true"] { background: #e2f7ec; color: #13744c; border-color: #9ed9bc; }
QLabel#voiceStudioStatus[runtimeState="error"] { background: #fdecec; color: #c0392b; border-color: #f2b8b8; }
QLabel#voiceStudioIntro { color: #3d5a7f; font-size: 13px; }
QFrame#voiceStudioPageHeading { background: transparent; border: none; }
QLabel#voiceStudioPageTitle { color: #0f2742; font-size: 22px; font-weight: 800; letter-spacing: 1px; }
QLabel#hint { color: #7c879c; font-size: 12px; }
QTabWidget::pane { border: 1px solid #e3e9f2; background: #ffffff; border-radius: 12px; top: -1px; }
QTabBar::tab { padding: 10px 22px; margin-right: 4px; background: transparent; color: #5d6b83; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #ffffff; color: #1d4ed8; font-weight: 700; border-bottom: 2px solid #2563eb; }
QTabBar::tab:hover:!selected { background: #eef4fd; }
QGroupBox { background: #ffffff; border: 1px solid #e8edf3; border-radius: 14px; margin-top: 14px; padding: 16px 12px 12px; font-weight: 700; }
QGroupBox#voiceStudioHero { background: #f6faff; border: 1px solid #bcd7f5; }
QFrame#voiceStudioNavCard { background: #f7fbff; border: 1px solid #d4e3f3; border-radius: 12px; }
QFrame#settingsNavCard { background: #f7fbff; border: 1px solid #d4e3f3; border-radius: 14px; }
QFrame#settingsContentPanel { background: rgba(255,255,255,175); border: 1px solid #d8e6f3; border-radius: 15px; }
QFrame#versionFactCard { background: #fbfdff; border: 1px solid #dde7f4; border-radius: 10px; }
QFrame#versionFactCard QLabel { background: transparent; border: none; padding: 0px; }
QFrame#voiceStudioTaskBar { background: #f7fbff; border: 1px solid #c9deef; border-radius: 11px; }
QLabel#voiceStudioTaskLabel { background: transparent; color: #315b7d; font-weight: 700; }
QPushButton#voiceStudioCancelTask { background: #fff6f5; color: #c33d3d; border: 1px solid #efb9b6; }
QPushButton#voiceStudioCancelTask:hover { background: #ffe9e7; border-color: #df8f89; }
QPushButton#voiceStudioCancelTask:disabled { background: #f2f4f7; color: #9aa5b8; border-color: #e0e5ec; }
QLabel#versionFactTitle { color: #4a6480; font-weight: 700; }
QLabel#versionValue { color: #1d4ed8; font-size: 18px; font-weight: 800; }
QLabel#versionMeta { color: #8493a8; font-size: 10px; }
QLabel#voiceStudioNavTitle { color: #1d4ed8; font-size: 15px; font-weight: 800; letter-spacing: 1px; }
QLabel#meetingNavTitle, QLabel#settingsNavTitle { color: #1d4ed8; font-size: 14px; font-weight: 800; letter-spacing: 1px; }
QLabel#settingsPageTitle { color: #15395d; font-size: 20px; font-weight: 800; }
QLabel#navHealthCard { background: #edf8f3; color: #287653; border: 1px solid #c6e9d6; border-radius: 9px; padding: 10px; font-size: 11px; }
QLabel#navHealthCard[runtimeState="error"] { background: #fff0f0; color: #bb3c3c; border-color: #f5c3c3; }
QLabel#navHealthCard[runtimeState="stopped"] { background: #f2f5f8; color: #718096; border-color: #dce4ec; }
QPushButton#voiceStudioNavButton, QPushButton#meetingNavButton, QPushButton#settingsNavButton { text-align: left; background: transparent; border: 1px solid transparent; padding: 10px 12px; border-radius: 9px; }
QPushButton#voiceStudioNavButton:hover, QPushButton#meetingNavButton:hover, QPushButton#settingsNavButton:hover { background: #e8f2ff; border-color: #b9d7ff; }
QPushButton#voiceStudioNavButton:checked, QPushButton#meetingNavButton:checked, QPushButton#settingsNavButton:checked { background: #e5f1ff; color: #126fc2; border-color: #b9d7ff; font-weight: 800; }
QScrollArea#voiceStudioDashboardScroll, QWidget#voiceStudioDashboardContent { border: none; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #21395c; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: #ffffff; border: 1px solid #d9e1ec; border-radius: 7px; padding: 6px; selection-background-color: #2563eb; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #2563eb; }
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #f1f4f8; color: #9aa5b8; border-color: #e5eaf2; }
QComboBox QAbstractItemView { background: #ffffff; color: #1f2a44; selection-background-color: #e3edfe; selection-color: #1d4ed8; border: 1px solid #d9e1ec; border-radius: 7px; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #7c879c; margin-right: 7px; }
QCheckBox, QRadioButton { spacing: 6px; padding: 2px; }
QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled { color: #9aa5b8; }
QPushButton { background: #f4f6fa; border: 1px solid #dde4ee; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QToolButton { background: #f4f6fa; color: #24415f; border: 1px solid #dde4ee; border-radius: 8px; padding: 7px 11px; font-weight: 600; }
QPushButton:hover { background: #e9eef7; border-color: #c9d6ea; }
QToolButton:hover { background: #e9eef7; border-color: #c9d6ea; }
QPushButton:pressed { background: #dde6f3; }
QToolButton:pressed { background: #dde6f3; }
QPushButton:disabled { background: #f1f4f8; color: #9aa5b8; border-color: #e5eaf2; }
QToolButton:disabled { background: #f1f4f8; color: #9aa5b8; border-color: #e5eaf2; }
QPushButton#primaryButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3b5bdb, stop:1 #4c6ef5); color: white; border: none; }
QPushButton#primaryButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3451d1, stop:1 #4263eb); }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary { background: #1d4ed8; color: white; border: none; }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary:hover { background: #1b45bd; }
QFrame#voiceStudioBackendSwitch { background: #e6ecf4; border: 1px solid #d3dde9; border-radius: 10px; }
QPushButton[backendSegment="true"] { background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 14px; font-weight: 600; color: #4a6b8a; }
QPushButton[backendSegment="true"]:hover { color: #1d4ed8; }
QPushButton[backendSegment="true"]:checked { background: #ffffff; color: #1d4ed8; border: 1px solid #c9d6ea; }
QPushButton[actionTile="true"], QToolButton[actionTile="true"] { font-size: 12px; font-weight: 700; padding: 10px 6px 8px; border-radius: 13px; background: #ffffff; border: 1px solid #d7e5f2; color: #193a5a; }
QPushButton[actionTile="true"]:hover, QToolButton[actionTile="true"]:hover { background: #f2f8ff; border-color: #8ec4f0; }
QPushButton[actionTile="true"]:pressed, QToolButton[actionTile="true"]:pressed { background: #e5f2ff; border-color: #69ace4; }
QToolButton[actionTile="true"]:disabled { background: #f5f7fa; color: #a5b0bf; border-color: #e7ecf2; }
QPushButton#advancedToggle { background: transparent; border: 1px dashed #bfcadd; color: #4c5b76; font-weight: 600; }
QPushButton#advancedToggle:hover { background: #eef3fb; }
QPushButton#advancedToggle:checked { background: #e7eefb; border-style: solid; color: #1d4ed8; }
QPushButton#directButton, QPushButton#translateButton { min-height: 58px; color: white; font-size: 17px; font-weight: 800; border: none; border-radius: 12px; }
QPushButton#directButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2f4f8f, stop:1 #3b5fc9); }
QPushButton#directButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2a4780, stop:1 #3451b8); }
QPushButton#translateButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e86b8, stop:1 #18a4d4); }
QPushButton#translateButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0c74a0, stop:1 #1490bc); }
QPushButton#directButton[active="true"], QPushButton#translateButton[active="true"] { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #d93b2e, stop:1 #f0654a); }
QPushButton#teacherButton[active="true"] { background: #188a52; color: white; border-color: #137a48; }
QPushButton#recordButton[recording="true"] { background: #d84343; color: white; border-color: #bd3131; }
QPushButton#dangerButton, QToolButton#dangerButton { color: #c0392b; }
QPushButton#dangerButton, QToolButton#dangerButton:hover { background: #fdecec; border-color: #f2b8b8; }
QPushButton#engineScopeButton { background: #e9f1fe; color: #1d4ed8; border: 1px solid #bcd3fb; font-weight: 800; }
QPushButton#engineScopeButton[localEngine="true"] { background: #e2f7ec; color: #13744c; border-color: #9ed9bc; }
QFrame#utilityBar { background: rgba(255,255,255,200); border: 1px solid #e3e9f2; border-radius: 10px; }
QFrame#artPanel { background: rgba(255,255,255,230); border: 1px solid #dce7f0; border-radius: 14px; }
QLabel#artwork { background: #eaf6fc; border: 1px solid #c8e5f2; border-radius: 9px; }
QLabel#backdropPreview { background: #eaf6fc; border: 1px solid #c8e5f2; border-radius: 6px; color: #76839c; font-size: 11px; }
QLabel#shortcutCard { background: #e8f3fb; color: #24435f; border-left: 3px solid #2a9bd4; border-radius: 6px; padding: 8px; font-weight: 700; line-height: 1.45; }
QLabel#artCredit { color: #7c8c9c; background: transparent; font-size: 9px; }
QGroupBox#controlDeck { border-color: #c9dcea; }
QGroupBox#transcriptCard { padding-top: 14px; }
QTabWidget#inputTabs::pane { background: rgba(255,255,255,220); border: 1px solid #e3e9f2; border-radius: 9px; }
QTabWidget#inputTabs QTabBar::tab { padding: 6px 16px; }
QHeaderView::section { background: #f1f5fa; color: #3d5a7f; border: none; border-bottom: 1px solid #e0e7f0; padding: 7px; font-weight: 700; }
QTableWidget { alternate-background-color: #f7fafd; }
QTableWidget::item { padding: 2px; }
QScrollBar:vertical { background: transparent; width: 12px; }
QScrollBar::handle:vertical { background: #c9d4e4; min-height: 28px; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #b0c0d6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSlider::groove:horizontal { height: 6px; background: #dfe6f0; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: white; border: 2px solid #2563eb; border-radius: 7px; }
"""

DARK_STYLE_SHEET = """
QMainWindow, QWidget { background: #11151c; color: #e8edf6; font-family: "Microsoft YaHei UI"; font-size: 12px; }
QLabel#title { font-size: 22px; font-weight: 800; color: #f4f7ff; }
QLabel#subtitle { color: #9da9bc; font-size: 11px; letter-spacing: 1px; }
QLabel#brandLogo { background: transparent; }
QFrame#appHeader { background: #0f1620; border: none; border-bottom: 1px solid #2b394a; }
QFrame#mainNavigation { background: #0c1520; border: 1px solid #29394c; border-radius: 4px; }
QLabel#mainNavBrand { color: #edf5ff; font-size: 18px; font-weight: 800; letter-spacing: 3px; }
QLabel#mainNavHealth { background: #15261f; color: #6bd18d; border: 1px solid #294d3b; border-radius: 5px; padding: 10px; font-weight: 700; }
QLabel#mainNavHealth[runtimeState="error"] { background: #321a20; color: #ff9292; border-color: #6c303b; }
QLabel#mainNavHealth[runtimeState="stopped"] { background: #171e27; color: #8190a4; border-color: #303b4a; }
QPushButton#mainNavButton { background: transparent; color: #bdc8d8; border: 1px solid transparent; text-align: left; font-size: 14px; padding: 10px 13px; }
QPushButton#mainNavButton:hover { background: #182536; border-color: #2f4761; }
QPushButton#mainNavButton:checked { background: #172840; color: #79baff; border-color: #5589c7; font-weight: 800; }
QFrame#appFooter { background: #0f1620; border-top: 1px solid #2b394a; }
QLabel#footerStatus { color: #6bd18d; font-weight: 700; }
QLabel#statusBadge { background: #17365c; color: #8dc4ff; border: 1px solid #285b91; border-radius: 14px; padding: 7px 13px; font-weight: 700; }
QLabel#voiceStudioStatus { background: #3c3018; color: #f1c86d; border: 1px solid #6b5527; border-radius: 12px; padding: 6px 11px; font-weight: 700; }
QLabel#voiceStudioStatus[connected="true"] { background: #183d30; color: #83ddb8; border-color: #2d7258; }
QLabel#voiceStudioStatus[runtimeState="error"] { background: #4a2024; color: #ff9d9d; border-color: #8b3b43; }
QLabel#voiceStudioIntro { color: #bac9dc; font-size: 13px; }
QFrame#voiceStudioPageHeading { background: transparent; border: none; }
QLabel#voiceStudioPageTitle { color: #f4f7ff; font-size: 22px; font-weight: 800; letter-spacing: 1px; }
QLabel#hint { color: #9aa7bb; font-size: 12px; }
QTabWidget::pane { border: 1px solid #303948; background: #171c24; border-radius: 10px; top: -1px; }
QTabBar::tab { padding: 10px 22px; margin-right: 4px; background: #242b36; color: #aeb9ca; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #171c24; color: #74b7ff; font-weight: 700; }
QGroupBox { background: #191f28; border: 1px solid #333d4d; border-radius: 12px; margin-top: 14px; padding: 16px 12px 12px; font-weight: 700; }
QGroupBox#voiceStudioHero { background: #17222f; border: 1px solid #315d87; }
QFrame#voiceStudioNavCard { background: #121b26; border: 1px solid #33465c; border-radius: 12px; }
QFrame#settingsNavCard { background: #121b26; border: 1px solid #33465c; border-radius: 8px; }
QFrame#settingsContentPanel { background: #151c25; border: 1px solid #2e3c4d; border-radius: 8px; }
QFrame#versionFactCard { background: #141e2a; border: 1px solid #33465c; border-radius: 7px; }
QFrame#voiceStudioTaskBar { background: #151f2b; border-color: #35495f; }
QLabel#voiceStudioTaskLabel { color: #b9d7f2; }
QPushButton#voiceStudioCancelTask { background: #321d23; color: #ff9a9a; border-color: #6b3540; }
QPushButton#voiceStudioCancelTask:hover { background: #44232b; border-color: #95505d; }
QLabel#versionFactTitle { color: #9eb0c5; font-weight: 700; }
QLabel#versionValue { color: #82bdff; font-size: 18px; font-weight: 800; }
QLabel#versionMeta { color: #7d8da3; font-size: 10px; }
QLabel#voiceStudioNavTitle { color: #74b7ff; font-size: 15px; font-weight: 800; letter-spacing: 1px; }
QLabel#meetingNavTitle, QLabel#settingsNavTitle { color: #74b7ff; font-size: 14px; font-weight: 800; letter-spacing: 1px; }
QLabel#settingsPageTitle { color: #f1f6ff; font-size: 20px; font-weight: 800; }
QLabel#navHealthCard { background: #17291f; color: #70cf96; border: 1px solid #2d553f; border-radius: 6px; padding: 10px; font-size: 11px; }
QLabel#navHealthCard[runtimeState="error"] { background: #321a20; color: #ff9292; border-color: #6c303b; }
QLabel#navHealthCard[runtimeState="stopped"] { background: #171e27; color: #8190a4; border-color: #303b4a; }
QPushButton#voiceStudioNavButton, QPushButton#meetingNavButton, QPushButton#settingsNavButton { text-align: left; background: transparent; border: 1px solid transparent; padding: 10px 12px; }
QPushButton#voiceStudioNavButton:hover, QPushButton#meetingNavButton:hover, QPushButton#settingsNavButton:hover { background: #213149; border-color: #3b6d9e; }
QPushButton#voiceStudioNavButton:checked, QPushButton#meetingNavButton:checked, QPushButton#settingsNavButton:checked { background: #182e48; color: #8bc5ff; border-color: #3c6b9e; font-weight: 800; }
QScrollArea#voiceStudioDashboardScroll, QWidget#voiceStudioDashboardContent { border: none; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #dce7f8; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: #10151c; color: #edf2fa; border: 1px solid #3c485a; border-radius: 6px; padding: 6px; selection-background-color: #286fb8; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #4d9bff; }
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #171d26; color: #5d6b80; border-color: #2a3341; }
QComboBox QAbstractItemView { background: #171d26; color: #edf2fa; selection-background-color: #286fb8; }
QCheckBox, QRadioButton { spacing: 6px; padding: 2px; }
QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled { color: #5d6b80; }
QPushButton { background: #283140; color: #edf2fa; border: 1px solid #46546a; border-radius: 7px; padding: 8px 13px; font-weight: 600; }
QToolButton { background: #283140; color: #edf2fa; border: 1px solid #46546a; border-radius: 7px; padding: 7px 11px; font-weight: 600; }
QPushButton:hover { background: #344156; }
QToolButton:hover { background: #344156; }
QPushButton:pressed { background: #202937; }
QToolButton:pressed { background: #202937; }
QPushButton:disabled { background: #1c232e; color: #5d6b80; border-color: #2a3341; }
QToolButton:disabled { background: #1c232e; color: #5d6b80; border-color: #2a3341; }
QPushButton#primaryButton { background: #1971ca; color: white; border: none; }
QPushButton#primaryButton:hover { background: #2b82db; }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary { background: #1971ca; color: white; border: none; }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary:hover { background: #2b82db; }
QFrame#voiceStudioBackendSwitch { background: #161d27; border: 1px solid #2a3341; border-radius: 10px; }
QPushButton[backendSegment="true"] { background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 14px; font-weight: 600; color: #8fa2b8; }
QPushButton[backendSegment="true"]:hover { color: #d5e4f5; }
QPushButton[backendSegment="true"]:checked { background: #22303f; color: #6db3f2; border: 1px solid #3c6b9e; }
QPushButton[actionTile="true"], QToolButton[actionTile="true"] { font-size: 12px; font-weight: 700; padding: 10px 6px 8px; border-radius: 13px; background: #1a212c; border: 1px solid #344457; color: #dde6f2; }
QPushButton[actionTile="true"]:hover, QToolButton[actionTile="true"]:hover { background: #22303f; border-color: #3c6b9e; }
QPushButton[actionTile="true"]:pressed, QToolButton[actionTile="true"]:pressed { background: #1c2836; }
QToolButton[actionTile="true"]:disabled { background: #171d26; color: #5d6b80; border-color: #2a3341; }
QPushButton#advancedToggle { background: transparent; border: 1px dashed #3d4a5e; color: #9fb0c6; font-weight: 600; }
QPushButton#advancedToggle:hover { background: #1f2733; }
QPushButton#advancedToggle:checked { background: #223247; border-style: solid; color: #8dc4ff; }
QPushButton#directButton, QPushButton#translateButton { min-height: 58px; color: white; font-size: 17px; font-weight: 800; border: none; border-radius: 10px; }
QPushButton#directButton { background: #315b82; }
QPushButton#translateButton { background: #1971ca; }
QPushButton#directButton[active="true"], QPushButton#translateButton[active="true"] { background: #d95649; }
QPushButton#teacherButton[active="true"] { background: #167c5d; color: white; border-color: #27a67e; }
QPushButton#recordButton[recording="true"] { background: #c43f3f; color: white; border-color: #e05a5a; }
QPushButton#dangerButton, QToolButton#dangerButton { color: #ff9d93; }
QPushButton#engineScopeButton { background: #17365c; color: #8dc4ff; border: 1px solid #285b91; font-weight: 800; }
QPushButton#engineScopeButton[localEngine="true"] { background: #183d30; color: #83ddb8; border-color: #2d7258; }
QFrame#utilityBar { background: #171d26; border: 1px solid #303b4b; border-radius: 9px; }
QFrame#artPanel { background: #171d26; border: 1px solid #364255; border-radius: 12px; }
QLabel#artwork { background: #0f1821; border: 1px solid #35475a; border-radius: 8px; }
QLabel#backdropPreview { background: #0f1821; border: 1px solid #35475a; border-radius: 6px; color: #9aa7bb; font-size: 11px; }
QLabel#shortcutCard { background: #172b3d; color: #b9ddfa; border-left: 3px solid #3ba9dd; border-radius: 5px; padding: 8px; font-weight: 700; }
QLabel#artCredit { color: #7e8da1; background: transparent; font-size: 9px; }
QTabWidget#inputTabs::pane { background: #151b23; border: 1px solid #303b4b; border-radius: 8px; }
QTabWidget#inputTabs QTabBar::tab { padding: 6px 16px; }
QHeaderView::section { background: #252e3b; color: #e5ecf7; border: none; border-bottom: 1px solid #3b4657; padding: 7px; font-weight: 700; }
QScrollBar:vertical { background: #161c24; width: 12px; }
QScrollBar::handle:vertical { background: #485568; min-height: 28px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSlider::groove:horizontal { height: 5px; background: #2a3442; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #6e48d7; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #edf2fa; border: 2px solid #8c66ea; border-radius: 7px; }
"""

# These palette variants deliberately share widget geometry, so switching a
# theme never shifts the meeting controls. The Shizuku artwork is optional and
# loaded from an ignored local-only directory.
SHIZUKU_STYLE_SHEET = (
    STYLE_SHEET
    + """
QWidget#appRoot { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8fcff, stop:0.30 #eaf7ff, stop:0.68 #dff4fb, stop:1 #e8ecff); }
QFrame#appHeader { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgba(255,255,255,242), stop:0.55 rgba(229,247,255,225), stop:1 rgba(219,233,255,218)); border-bottom: 1px solid #bddfed; }
QFrame#mainNavigation { background: rgba(251,254,255,230); border-color: #c9e0ef; }
QGroupBox { background: rgba(255,255,255,226); border-color: #cfe4f1; }
QFrame#voiceStudioNavCard, QFrame#settingsNavCard { background: rgba(248,253,255,232); border-color: #c9e2f1; }
QFrame#settingsContentPanel { background: rgba(255,255,255,160); border-color: #c9e2f1; }
QFrame#versionFactCard { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(255,255,255,245), stop:1 rgba(235,247,255,228)); border-color: #cce2f3; }
QTabWidget#mainPages::pane { background: rgba(255,255,255,118); border-color: #c6e0ef; }
QTabWidget#mainPages > QTabBar::tab { background: rgba(255,255,255,135); border: 1px solid rgba(191,219,238,160); }
QTabWidget#mainPages > QTabBar::tab:selected { background: rgba(255,255,255,242); color: #087cb8; border-color: #a9d4e9; border-bottom: 3px solid #1ca6d5; }
QFrame#artPanel { background: rgba(249,253,255,232); border-color: #c7e0ef; }
QWidget#voiceStudioDashboardContent { background: transparent; }
QMainWindow, QWidget { background: #eaf6fc; color: #0f3b57; }
QLabel#title { color: #0f3b57; }
QPushButton#primaryButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e86b8, stop:1 #22b8cf); }
QPushButton#primaryButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0c74a0, stop:1 #18a4d4); }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary { background: #0b7ba3; }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary:hover { background: #096a8c; }
QFrame#voiceStudioBackendSwitch { background: #d7ebf5; border: 1px solid #bcdcee; border-radius: 10px; }
QPushButton[backendSegment="true"] { background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 14px; font-weight: 600; color: #33617a; }
QPushButton[backendSegment="true"]:hover { color: #0b7ba3; }
QPushButton[backendSegment="true"]:checked { background: #ffffff; color: #0b7ba3; border: 1px solid #a8d4e8; }
QPushButton#directButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #155e8c, stop:1 #1d86b8); }
QPushButton#directButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #125478, stop:1 #1976a3); }
QPushButton#translateButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e86b8, stop:1 #22b8cf); }
QPushButton#translateButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0c74a0, stop:1 #18a4d4); }
QTabBar::tab:selected { color: #0b7ba3; border-bottom-color: #0e86b8; }
QLabel#statusBadge { background: #dff2fb; color: #0b7ba3; border-color: #a3d5ea; }
QPushButton#engineScopeButton { background: #dff2fb; color: #0b7ba3; border-color: #a3d5ea; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #0e86b8; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { selection-background-color: #0e86b8; }
QPushButton#mainNavButton:checked { background: #dff2fb; color: #0b7ba3; border-color: #a3d5ea; }
QPushButton#advancedToggle:checked { background: #dff2fb; color: #0b7ba3; }
QLabel#shortcutCard { background: #e8f6fb; color: #125478; border-left-color: #0e86b8; }
QSlider::sub-page:horizontal { background: #0e86b8; }
QSlider::handle:horizontal { border-color: #0e86b8; }
"""
)

WARM_STYLE_SHEET = (
    STYLE_SHEET
    + """
QWidget#appRoot { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fffdf8, stop:0.55 #f8f1e7, stop:1 #fff9ef); }
QFrame#appHeader, QFrame#appFooter { background: rgba(255,253,248,225); border-color: #eadbc5; }
QFrame#mainNavigation { background: rgba(255,251,244,235); border-color: #ead8bd; }
QGroupBox { background: rgba(255,253,249,240); border-color: #eadbc5; }
QWidget#voiceStudioDashboardContent { background: transparent; }
QMainWindow, QWidget { background: #faf5ee; color: #3f2a15; }
QLabel#title { color: #3f2a15; }
QPushButton#primaryButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #c2410c, stop:1 #ea580c); }
QPushButton#primaryButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #a93708, stop:1 #d64d0a); }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary { background: #b45309; }
QPushButton#voiceStudioPrimary, QToolButton#voiceStudioPrimary:hover { background: #9a4807; }
QFrame#voiceStudioBackendSwitch { background: #f0e6d6; border: 1px solid #e3d2b8; border-radius: 10px; }
QPushButton[backendSegment="true"] { background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 14px; font-weight: 600; color: #7a5c38; }
QPushButton[backendSegment="true"]:hover { color: #b45309; }
QPushButton[backendSegment="true"]:checked { background: #fffdf8; color: #b45309; border: 1px solid #dfc9a6; }
QPushButton#directButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #7c4a12, stop:1 #a3612b); }
QPushButton#directButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6b400f, stop:1 #8f5323); }
QPushButton#translateButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #c2410c, stop:1 #ea580c); }
QPushButton#translateButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #a93708, stop:1 #d64d0a); }
QTabBar::tab:selected { color: #92400e; border-bottom-color: #ea580c; }
QLabel#statusBadge { background: #fbeedd; color: #92400e; border-color: #e5c9a8; }
QPushButton#engineScopeButton { background: #fbeedd; color: #92400e; border-color: #e5c9a8; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #ea580c; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { selection-background-color: #ea580c; }
QPushButton#mainNavButton:checked { background: #fbeedd; color: #92400e; border-color: #e5c9a8; }
QPushButton#advancedToggle:checked { background: #fbeedd; color: #92400e; }
QLabel#shortcutCard { background: #fdf1e3; color: #7c4a12; border-left-color: #ea580c; }
QSlider::sub-page:horizontal { background: #ea580c; }
QSlider::handle:horizontal { border-color: #ea580c; }
"""
)


# Layout and palette are intentionally independent. These sheets only change
# geometry, density and hierarchy; the four palette sheets above own colors.
LAYOUT_STYLE_SHEETS = {
    "crystal": """
        QLabel#title { font-size: 31px; letter-spacing: 0px; }
        QLabel#subtitle { font-size: 12px; letter-spacing: 1px; }
        QTabWidget#mainPages > QTabBar::tab { min-width: 180px; padding: 14px 32px; font-size: 15px; }
        QTabWidget#mainPages::pane { border-radius: 18px; padding: 6px; }
        QFrame#appHeader { border-radius: 0px; }
        QFrame#voiceStudioNavCard, QFrame#settingsNavCard { border-radius: 16px; }
        QFrame#settingsContentPanel { border-radius: 17px; }
        QGroupBox#voiceStudioConnectionCard, QGroupBox#voiceStudioActionsCard { min-height: 180px; }
        QGroupBox#voiceStudioVersionsCard { min-height: 108px; }
        QFrame#versionFactCard { border-radius: 11px; }
        QGroupBox { border-radius: 15px; margin-top: 15px; padding: 17px 13px 13px; }
        QPushButton { border-radius: 9px; padding: 8px 14px; }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { border-radius: 8px; padding: 7px; }
        QFrame#utilityBar, QFrame#artPanel { border-radius: 12px; }
        QPushButton#directButton, QPushButton#translateButton { border-radius: 14px; font-size: 18px; }
        QPushButton#voiceStudioNavButton, QPushButton#meetingNavButton, QPushButton#settingsNavButton { min-height: 45px; font-size: 13px; }
        QLabel#meetingNavTitle, QLabel#settingsNavTitle, QLabel#voiceStudioNavTitle { margin: 2px 4px 8px 4px; }
        QHeaderView::section { min-height: 28px; }
    """,
    "signal": """
        QLabel#title { font-size: 20px; letter-spacing: 1px; }
        QLabel#subtitle { font-size: 10px; letter-spacing: 2px; }
        QTabWidget::pane { border-radius: 4px; }
        QTabWidget#mainPages::pane { border: none; padding: 0px; }
        QFrame#appHeader { border-radius: 0px; }
        QFrame#mainNavigation { border-radius: 0px; }
        QFrame#versionFactCard { border-radius: 5px; }
        QGroupBox#voiceStudioConnectionCard, QGroupBox#voiceStudioVersionsCard { min-height: 245px; }
        QTabBar::tab { padding: 7px 16px; margin-right: 2px; border-radius: 3px; }
        QGroupBox { border-radius: 5px; margin-top: 11px; padding: 12px 9px 9px; }
        QGroupBox::title { left: 9px; padding: 0 4px; }
        QPushButton { border-radius: 4px; padding: 7px 11px; }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { border-radius: 3px; padding: 5px; }
        QFrame#utilityBar { border-radius: 4px; }
        QPushButton#directButton, QPushButton#translateButton { border-radius: 5px; font-size: 16px; }
        QHeaderView::section { padding: 6px; }
    """,
    "studio": """
        QLabel#title { font-size: 21px; }
        QLabel#subtitle { font-size: 11px; letter-spacing: 1px; }
        QTabWidget::pane { border-radius: 12px; }
        QTabWidget#mainPages::pane { border-left: none; border-right: none; border-radius: 0px; }
        QFrame#appHeader { border-radius: 0px; }
        QFrame#versionFactCard { border-radius: 9px; }
        QGroupBox#voiceStudioConnectionCard, QGroupBox#voiceStudioVersionsCard { min-height: 150px; }
        QTabBar::tab { padding: 9px 20px; margin-right: 5px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
        QGroupBox { border-radius: 10px; margin-top: 14px; padding: 16px 12px 12px; }
        QPushButton { border-radius: 8px; padding: 8px 13px; }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { border-radius: 7px; padding: 7px; }
        QFrame#utilityBar { border-radius: 10px; }
        QPushButton#directButton, QPushButton#translateButton { border-radius: 10px; font-size: 16px; }
    """,
    "fluent": """
        QLabel#title { font-size: 19px; }
        QLabel#statusBadge { border-radius: 7px; padding: 6px 10px; }
        QTabWidget::pane { border-radius: 7px; }
        QTabWidget#mainPages::pane { border: none; padding: 0px; }
        QFrame#appHeader { border-radius: 0px; }
        QFrame#mainNavigation { border-radius: 0px; }
        QFrame#versionFactCard { border-radius: 5px; }
        QGroupBox#voiceStudioVersionsCard { min-height: 120px; }
        QTabBar::tab { padding: 7px 14px; margin-right: 2px; border-radius: 5px; }
        QGroupBox { border-radius: 7px; margin-top: 11px; padding: 12px 9px 9px; }
        QGroupBox::title { left: 9px; padding: 0 4px; }
        QPushButton { border-radius: 6px; padding: 6px 10px; }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { border-radius: 5px; padding: 5px; }
        QFrame#utilityBar { border-radius: 6px; }
        QPushButton#directButton, QPushButton#translateButton { border-radius: 7px; font-size: 15px; }
        QHeaderView::section { padding: 6px; }
    """,
}


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
