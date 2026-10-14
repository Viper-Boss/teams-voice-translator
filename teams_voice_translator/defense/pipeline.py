from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from ..aliyun import ApiError, BailianClient, create_realtime_asr
from ..audio import (
        DirectAudioBridge,
        DualTrackRecorder,
        MicrophoneCapture,
        MultiOutputPlayer,
        SystemAudioCapture,
        WavRecorder,
        choose_committee_loopback,
        list_loopback_devices,
        resolve_monitor_device,
        resolve_program_output,
    )
from .records import default_base_dir, new_meeting_folder, write_transcripts
from .memory import MeetingMemory, Turn
from .settings import DefenseSettings
from .translator import ContextTranslator, SentenceSplitter
from .tts_session import TtsHttpSession, TtsSession, create_tts_session
from .asr_standby import create_defense_asr
from .committee_process import CommitteeASRProcess
from .speech_policy import (
    FillerPicker,
    natural_delivery_instruction,
    SPEECH_SAMPLE_RATE,
    apply_opening_filler,
    group_sentences,
    join_sentences_ssml,
    speech_chunks,
    speech_settings,
)
from ..api_payloads import is_qwen3_tts_vc_realtime_model, is_qwen_audio_tts_model, tts_supports
from .pcm_spool import PcmSpool
from .tts_cache import RenderManifest, TtsDiskCache, cache_key
from .humanize import humanize_text, protected_terms, rate_for
from .ambience import RoomTone, RecordedRoomTone, make_room_tone, background_mode
from .usage import USAGE
from .channels import backup_channel

STATE_IDLE = "idle"  # 未开始
STATE_STARTING = "starting"  # 正在打开答辩通道
STATE_STANDBY = "standby"  # 答辩通道已就绪
STATE_LISTENING = "listening"  # F9 翻译中
STATE_DIRECT = "direct"  # 原声直通（默认 F8）


@dataclass
class EngineCallbacks:
    on_state: Callable[[str], None] = lambda state: None
    on_status: Callable[[str], None] = lambda message: None
    on_error: Callable[[str], None] = lambda message: None
    on_mic_preview: Callable[[str], None] = lambda text: None
    on_committee_preview: Callable[[str], None] = lambda text: None
    on_my_segment: Callable[[int, str], None] = lambda seg_id, text: None
    on_my_delta: Callable[[int, str], None] = lambda seg_id, text: None
    on_my_done: Callable[[int, str, int], None] = lambda seg_id, text, latency: None
    on_committee_done: Callable[[int, str, str], None] = lambda seg_id, text, translation: None
    on_committee_recognized: Callable[[int, str], None] = lambda seg_id, text: None
    on_tts_sentence_started: Callable[[str, str], None] = lambda text, zh: None
    on_tts_sentence_duration: Callable[[str, float], None] = lambda text, seconds: None
    on_tts_sentence_progress: Callable[[str, float], None] = lambda text, progress: None
    on_committee_device: Callable[[str], None] = lambda name: None
    on_latency: Callable[[int, int], None] = lambda seg_id, latency: None
    # 朗读控制：pause / resume / stop。字幕悬浮窗跟着停、跟着走、跟着熄灭。
    on_speech_control: Callable[[str], None] = lambda action: None
    on_input_stopped: Callable[[], None] = lambda: None


@dataclass
class CachedUtterance:
    """一句已合成英文的本地 PCM 缓存：重播不再请求服务器。"""

    pcm: bytes
    zh: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class SegmentRecord:
    seg_id: int
    role: str  # "me" | "committee"
    source: str
    translation: str = ""
    latency_ms: int = 0
    at: float = field(default_factory=time.time)


@dataclass
class _Job:
    seg_id: int
    source: str
    t0: float
    cancel: threading.Event = field(default_factory=threading.Event)
    first_audio_at: float | None = None
    # 同传模式下译文已经由同传模型给出，不再调用翻译大模型
    preset_english: str | None = None
    live_input: bool = False


class DefenseEngine:
    """Defense-mode orchestrator.

    Thread layout: one serialized command worker owns every start/stop
    transition (hotkeys and buttons only enqueue commands), two translation
    workers keep my speech and the committee's speech independent, and one
    persistent TTS session plays sentences in order.  All external contact
    happens through the callbacks in :class:`EngineCallbacks`.
    """

    def __init__(self, settings: DefenseSettings, callbacks: EngineCallbacks) -> None:
        self.settings = settings
        self.callbacks = callbacks
        self.memory = MeetingMemory(
            brief=str(settings.get("defense_brief", "") or ""),
            glossary=settings.glossary_terms(),
            max_turns=int(settings.get("max_context_turns", 14)),
            translation_style=str(settings.get("translation_style", "accurate") or "accurate"),
        )

        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._cmd_queue: queue.Queue[tuple[Callable[[], None], str]] = queue.Queue()
        self._cmd_thread: threading.Thread | None = None

        self.client: BailianClient | None = None
        self.translator: ContextTranslator | None = None
        self.tts: TtsSession | None = None
        # 设置切换后旧合成线程可能仍在网络请求中。回调只允许写入创建它的那一代。
        self._tts_generation = 0
        self.player: MultiOutputPlayer | None = None

        self._session_active = False
        self._listening = False
        self._listen_requested = None
        self._listen_request_serial = 0
        self._direct_requested = None
        self._cached_input_pause = False
        self._accept_mic_tail = True
        self._direct_request_serial = 0
        self._direct_active = False
        self._resume_listening_after_direct = False

        self.archive_folder = None
        self.recorder: DualTrackRecorder | None = None
        self.mic_recorder: WavRecorder | None = None
        self._player_lock = threading.RLock()
        self._pcm_cache: dict[str, CachedUtterance] = {}
        # 按**字节**限，不按条数。原来限 300 条毫无意义：一条可能是 0.5 秒，
        # 也可能是 20 秒的长回答 —— 300 条长回答接近 300MB 常驻内存。
        # 现在所有音频同时也在磁盘上，内存这份纯粹是加速，淘汰掉没有损失。
        self._cache_limit_bytes = max(
            8, int(settings.get("memory_cache_mb", 96) or 96)
        ) * 1024 * 1024
        self._cache_bytes = 0
        # 内存缓存关掉程序就没了，改一次设置也会被清空（apply_settings 会 clear）。
        # 磁盘缓存跨重启存活，是「重播不再请求服务器」和「讲稿离线朗读」的地基。
        self.disk_cache = TtsDiskCache(
            settings.base_dir, limit_mb=int(settings.get("disk_cache_mb", 1024) or 1024)
        )
        # 整场合成语音改为边产生边写盘，内存里只留一个小缓冲。
        # 一小时的会按实际发声 25 分钟算就是 ~72MB，不能一直挂在内存里。
        self.render_manifest = RenderManifest(self.disk_cache)
        self._group_slice_memo: dict = {}
        # 讲稿里单独重做 / 换语气的句子：{登记号: "缓存键\t语气名"}，优先于整段切出来的那句
        self.script_overrides = RenderManifest(self.disk_cache, "script_overrides.json")
        # 问答库里重做 / 换语气的回答：{登记号: "键1,键2\t语气名"}，优先于默认渲染
        self.answer_overrides = RenderManifest(self.disk_cache, "answer_overrides.json")
        # Full-answer versions are indexed independently of the selected voice.
        self.answer_catalog = RenderManifest(self.disk_cache, "answer_audio_catalog.json")
        self._answer_catalog_lock = threading.RLock()
        self._answer_legacy_checked = set()
        self._meeting_spool = PcmSpool(Path(settings.base_dir) / "meeting_audio.pcm")
        self._live_utterance: dict[str, Any] | None = None
        self._utterance_offsets: list[dict[str, Any]] = []
        self.asr_mic = None
        self.asr_committee = None
        self.mic_capture: MicrophoneCapture | None = None
        self.loopback_capture: SystemAudioCapture | None = None
        self.bridge: DirectAudioBridge | None = None

        self._my_queue: queue.Queue[_Job | None] = queue.Queue(maxsize=32)
        self._committee_queue: queue.Queue[tuple[int, str] | None] = queue.Queue(maxsize=32)
        self._workers: list[threading.Thread] = []
        self._segments: dict[int, SegmentRecord] = {}
        self._segment_lock = threading.Lock()
        self._next_segment_id = 1
        self._active_job: _Job | None = None
        self._job_lock = threading.RLock()
        self._playback_cancel = threading.Event()
        self._committee_cancel = threading.Event()
        self._speech_jobs: dict[str, _Job] = {}
        # 送给 TTS 的文本可能带 SSML 标记或开场衔接词，和字幕要显示、要缓存的
        # 文本不再是同一个串。回调里拿到的永远是「送出去的那一串」，所以这里
        # 存一张 送出文本 → 显示文本 的映射，在回调边界上换回来。
        # 不这样做的后果很直接：悬浮字幕上会出现 <speak>、黄色进度对不上、
        # 重播缓存按标记串做键，同一句话换个开场词就命不中。
        self._spoken_display: dict[str, str] = {}
        self._filler_picker = FillerPicker()
        self._last_humanized_at = 0.0
        self._speech_generation = 0
        self._suppress_until = 0.0
        self._room_tone = RoomTone(str(settings.get("ambience_mode", "off") or "off"))
        self._ambience_cancel = threading.Event()
        # 持续背景环境音：答辩通道开着且选了背景环境时，没人说话也一直输出极轻的
        # 房间底噪，语音只是叠在上面。不然评委会听到「说话时有底噪、一停就绝对静音」。
        self._ambience_stop = threading.Event()
        self._ambience_thread: threading.Thread | None = None
        self._last_voice_write = 0.0
        self._last_prepared: tuple[str, float] = ("", 0.0)
        self._cached_thread: threading.Thread | None = None
        # 本地整段播放的真实进度：讲稿台的光标和「下一段」都以它为准，不再自己掐表
        self._cached_state = {"tag": "", "played": 0.0, "total": 0.0, "active": False}
        # 朗读暂停：置位时播放停在当前位置，清除后从原处接着播（Esc 仍然是彻底停止）
        self._speech_paused = threading.Event()
        self._closed = False
        self._stopping = False

    # ---------------------------------------------------------------- public
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def segments(self) -> list[SegmentRecord]:
        with self._segment_lock:
            return sorted(self._segments.values(), key=lambda item: item.seg_id)

    def clear_memory(self) -> None:
        self.memory.clear_turns()
        self._pcm_cache.clear()
        self._cache_bytes = 0
        with self._segment_lock:
            self._segments.clear()

    def reload_context(self) -> None:
        """Re-read brief/glossary/window from settings after the user edits them."""
        self.memory.set_brief(str(self.settings.get("defense_brief", "") or ""))
        self.memory.set_glossary(self.settings.glossary_terms())
        self.memory.max_turns = max(2, int(self.settings.get("max_context_turns", 14)))
        self.memory.set_translation_style(str(self.settings.get("translation_style", "accurate") or "accurate"))

    def start(self) -> None:
        if self._closed or self._workers:
            return
        if self._cmd_thread is None:
            self._cmd_thread = threading.Thread(target=self._command_loop, name="defense-commands", daemon=True)
            self._cmd_thread.start()
        for target, name in (
            (self._my_worker, "defense-translate-me"),
            (self._committee_worker, "defense-translate-committee"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._workers.append(thread)

    def shutdown(self) -> None:
        if self._closed:
            return
        self.interrupt_speech()
        self._committee_cancel.set()
        try:
            self._cmd_queue.put_nowait((self._do_stop_defense, "shutdown"))
        except queue.Full:
            pass
        # Command is queued before the flag flips so the loop is guaranteed to
        # drain the stop command before it sees the closed flag and exits.
        self._closed = True
        if self._cmd_thread is not None:
            self._cmd_thread.join(timeout=10)
        else:
            try:
                self._do_stop_defense()
            finally:
                self._meeting_spool.close()
        # 命令线程可能还在等待音频驱动退出。它会在真正退出时关闭 spool，
        # 不能在此处提前关闭，否则迟到的录音回调可能重新打开文件。

    # --------------------------------------------------------------- commands
    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state
        self.callbacks.on_state(state)

    def _submit(self, action: Callable[[], None], description: str) -> None:
        if self._closed:
            return
        if self._cmd_thread is None:
            self.start()
        self._cmd_queue.put_nowait((action, description))

    def _command_loop(self) -> None:
        try:
            while True:
                try:
                    action, description = self._cmd_queue.get(timeout=0.2)
                except queue.Empty:
                    if self._closed:
                        break
                    continue
                if self._closed and description != "shutdown":
                    continue
                try:
                    action()
                except ApiError as exc:
                    if self.state == STATE_STARTING:
                        self._set_state(STATE_IDLE)
                    self.callbacks.on_error(str(exc))
                except Exception as exc:  # pragma: no cover - defensive
                    if self.state == STATE_STARTING:
                        self._set_state(STATE_IDLE)
                    self.callbacks.on_error(f"{description}失败：{exc}")
        finally:
            # shutdown 的 join 超时后，仍须等设备线程真正退出才能关录音文件。
            self._meeting_spool.close()

    def start_defense(self) -> None:
        self._submit(self._do_start_defense, "打开答辩通道")

    def stop_defense(self) -> None:
        self._submit(self._do_stop_defense, "关闭答辩通道")

    def toggle_listening(self) -> None:
        self.set_listening(not bool(self._listen_requested or self._listening))

    def set_listening(self, enabled: bool) -> None:
        if enabled and not self._listen_requested:
            self._pause_for_input("translate")
            self._accept_mic_tail = True
        self._listen_requested = bool(enabled)
        self._listen_request_serial += 1
        serial = self._listen_request_serial
        self._submit(lambda: self._do_set_listening(enabled) if serial == self._listen_request_serial else None, "切换监听")

    def _do_set_listening(self, enabled: bool) -> None:
        if enabled and not self._listening:
            self._do_toggle_listening()
        elif not enabled and self._listening:
            self._do_toggle_listening()

    def start_direct(self) -> None:
        self._pause_for_input("direct")
        self._direct_requested = True
        self._direct_request_serial += 1
        serial = self._direct_request_serial
        self._submit(lambda: self._do_start_direct() if serial == self._direct_request_serial else None, "开启原声直通")

    def stop_direct(self) -> None:
        self._direct_requested = False
        if self.bridge is not None:
            self.bridge.running.clear()  # gate PCM immediately, not after queued shutdown
        self._direct_request_serial += 1
        serial = self._direct_request_serial
        self._submit(lambda: self._do_stop_direct() if serial == self._direct_request_serial else None, "关闭原声直通")

    def _pause_for_input(self, mode: str) -> None:
        # Preserve cached PCM/cursor; F9's new TTS may play while that PCM is held.
        self.interrupt_speech(user=False, keep_cached=True)
        self._cached_input_pause = bool(self._cached_state.get("active"))
        if mode == "direct":
            self._speech_paused.set()
        self.callbacks.on_speech_control("pause")

    def _claim_playback(self) -> None:
        """Explicit playback owns the voice channel; committee ASR is untouched."""
        live = self._direct_requested or self._listen_requested or self._direct_active or self._listening
        self._accept_mic_tail = False
        if self._listen_requested or self._listening:
            self.set_listening(False)
        if self._direct_requested or self._direct_active or self.bridge is not None:
            self.stop_direct()
        if live or getattr(self, "_cached_input_pause", False):
            self.interrupt_speech(user=False, keep_cached=True)
        self.callbacks.on_input_stopped()

    # ------------------------------------------------ 暂停 / 继续
    @property
    def speech_paused(self) -> bool:
        return self._speech_paused.is_set() or self._cached_input_pause

    @property
    def last_voice_write(self) -> float:
        return self._last_voice_write

    def is_speaking(self) -> bool:
        """此刻是否在出声（或马上要出声）：最近写过语音、本地播放线程还在、或还有排队的句子。"""
        if time.monotonic() - self._last_voice_write < 0.6:
            return True
        thread = self._cached_thread
        if thread is not None and thread.is_alive():
            return True
        return bool(self._speech_jobs)

    def pause_speech(self) -> None:
        """暂停正在播的语音，停在原处。排队的句子也等着，不会丢。"""
        self._speech_paused.set()
        self.callbacks.on_speech_control("pause")
        self.callbacks.on_status("朗读已暂停（点「继续」从停下的地方接着读，Esc 彻底停止）")

    def resume_speech(self) -> None:
        was_paused = self._speech_paused.is_set() or getattr(self, "_cached_input_pause", False)
        self._claim_playback()
        self._cached_input_pause = False
        if was_paused:
            self._speech_paused.clear()
            self.callbacks.on_speech_control("resume")
            self.callbacks.on_status("继续朗读")

    def _wait_if_paused(self, cancel: threading.Event | None = None) -> bool:
        """暂停期间在这里等。返回 False 表示等的过程中被 Esc / 关闭取消了。"""
        while self._speech_paused.is_set():
            if self._closed or (cancel is not None and cancel.is_set()):
                return False
            time.sleep(0.03)
        return True

    def interrupt_speech(self, *, user: bool = True, keep_cached: bool = False) -> None:
        """Esc: silence the current and queued speech immediately.

        user=False：内部换播下一段时的打断（不熄字幕、不清除防重复记录）。
        """
        self._speech_paused.clear()
        self._ambience_cancel.set()
        if user:
            self._last_prepared = ("", 0.0)
            try:
                self.callbacks.on_speech_control("stop")
            except Exception:
                pass
        with self._job_lock:
            self._speech_generation += 1
            if not keep_cached:
                self._playback_cancel.set()
                self._cached_input_pause = False
            job = self._active_job
            if job is not None:
                job.cancel.set()
            while True:
                try:
                    queued = self._my_queue.get_nowait()
                    if queued is not None:
                        queued.cancel.set()
                except queue.Empty:
                    break
            self._speech_jobs.clear()
            # 映射表和任务表同生共死；留着会让下一次合成命中已取消那一句的显示文本。
            self._spoken_display.clear()
            self._filler_picker.reset()
            if self.tts is not None:
                self.tts.interrupt()
        self.callbacks.on_status("已停止当前语音")

    # ------------------------------------------------------ 发声文本的构造
    def speech_features(self) -> dict[str, Any]:
        """当前模型实际能用哪些自然度手段。UI 直接拿这个显示 ✓/✗。

        判断只看模型能力表，不看用户开没开 —— 「支不支持」和「开没开」是
        两件事，界面上要分开显示，否则用户会以为自己开了却没效果。
        """
        model = str(self.settings.get("tts_model", "") or "")
        return {
            "model": model,
            "instruction": tts_supports(model, "instruction"),
            "ssml": tts_supports(model, "ssml"),
            # 并行预生成由会话类型决定：实时 WebSocket 会话是单连接串行的，
            # 不能并发写同一个会话，所以只有走 HTTP 的模型才有预生成。
            "prefetch": not is_qwen3_tts_vc_realtime_model(model),
            "clone": tts_supports(model, "clone"),
        }

    def _sentence_break_ms(self) -> int:
        """未启用、模型不支持、或值非法时一律返回 0（= 维持原有行为）。"""
        if not tts_supports(str(self.settings.get("tts_model", "") or ""), "ssml"):
            return 0
        try:
            value = int(self.settings.get("sentence_break_ms", 0) or 0)
        except (TypeError, ValueError):
            return 0
        # 上限 1000ms：再长就不是换气而是冷场了。
        return max(0, min(1000, value))

    def _prepare_utterances(self, text: str, *, with_filler: bool = True) -> list[tuple[str, str]]:
        """把一段译文变成 [(送给TTS的文本, 用于字幕和缓存的文本), ...]。

        两项加工都是可选的，任何一项关闭或模型不支持时，送出文本 == 显示文本，
        行为和改动前完全一致。
        """
        chunks = speech_chunks(text)
        if not chunks:
            raise ApiError("没有可朗读的内容")
        filler = ""
        if with_filler and str(self.settings.get("opening_filler", "off") or "off") == "auto":
            filler = self._filler_picker.pick()
        break_ms = self._sentence_break_ms()
        model = str(self.settings.get("tts_model", "") or "")

        prepared: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            # 衔接词只加在整段的第一块，不是每一块都加。
            display = apply_opening_filler(chunk, filler) if index == 0 and filler else chunk
            spoken = self._humanize_live(display)
            if break_ms:
                sentences = self._split_sentences(spoken)
                spoken = join_sentences_ssml(
                    sentences, model, break_ms=break_ms, wrap_single=True
                )
            prepared.append((spoken, display))
        return prepared

    HUMANIZE_COOLDOWN = 12.0   # 秒：现场发声两次小失误之间至少隔这么久，连着出错一听就假

    def humanize_level(self) -> str:
        level = str(self.settings.get("humanize_level", "off") or "off")
        return level if rate_for(level) > 0 else "off"

    def _humanize_live(self, display: str) -> str:
        """现场发声：按档位偶尔加一个小失误（只改送给 TTS 的文本，字幕不变）。"""
        level = self.humanize_level()
        if level == "off":
            return display
        if time.monotonic() - self._last_humanized_at < self.HUMANIZE_COOLDOWN:
            return display
        try:
            spoken = humanize_text(display, level, protected=protected_terms(self.settings))
        except Exception:  # noqa: BLE001 —— 加失误失败就照原文读，绝不影响发声
            return display
        if spoken != display:
            self._last_humanized_at = time.monotonic()
        return spoken

    def _humanize_prerender(self, text: str, language: str) -> str:
        """提前渲染（讲稿 / 预案）：同一句同一档位结果固定，渲染和播放算出同一个缓存键。"""
        level = self.humanize_level()
        mode = self.settings.get("prerender_instruction", "b")
        # A and Qwen's default B compare the literal text. Keep their manuscript
        # path identical too; "follow" still honors the user's text humanization.
        render_model = self.prerender_voice()[0] or str(self.settings.get("tts_model", ""))
        reference_delivery = mode == "none" or (mode == "b" and is_qwen_audio_tts_model(render_model))
        if level == "off" or language != "en" or reference_delivery:
            return text
        try:
            return humanize_text(text, level, protected=protected_terms(self.settings))
        except Exception:  # noqa: BLE001
            return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        splitter = SentenceSplitter()
        sentences = splitter.feed(text) + splitter.flush()
        return sentences or [text]

    def _register_utterances(self, prepared, job) -> list[str]:
        """登记 送出文本 → 显示文本 的映射，返回要送给 TTS 的文本列表。"""
        spoken_texts: list[str] = []
        for spoken, display in prepared:
            if spoken != display:
                self._spoken_display[spoken] = display
            if job is not None:
                self._speech_jobs[spoken] = job
            spoken_texts.append(spoken)
        return spoken_texts

    def _unregister_utterances(self, spoken_texts) -> None:
        for spoken in spoken_texts:
            self._speech_jobs.pop(spoken, None)
            self._spoken_display.pop(spoken, None)

    def _display_for(self, text: str) -> str:
        """回调边界：把送出去的文本换回给人看的文本。没登记过就是它自己。"""
        return self._spoken_display.get(text, text)

    def _queue_speech(self, text):
        prepared = self._prepare_utterances(text)
        spoken_texts = self._register_utterances(prepared, None)
        if self.tts is None or not self.tts.speak_batch(spoken_texts):
            self._unregister_utterances(spoken_texts)
            raise ApiError("发声队列未就绪，请稍后重播")

    def send_text(self, text: str, *, translate: bool = True) -> None:
        text = text.strip()
        if not text:
            return
        self._claim_playback()
        self.interrupt_speech()
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            self._ensure_voice()
            if translate:
                seg_id = self._register_segment("me", text)
                self.callbacks.on_my_segment(seg_id, text)
                self._my_queue.put_nowait(_Job(seg_id=seg_id, source=text, t0=time.monotonic()))
            else:
                seg_id = self._register_segment("me", "（手动输入）")
                self._queue_speech(text)
                self.memory.add_turn(Turn(role="me", source="", translation=text))
                self._store_segment(seg_id, text, latency_ms=0)
                self.callbacks.on_my_done(seg_id, text, 0)

        self._submit(action, "发送文本")

    def retranslate(self, seg_id: int) -> None:
        record = self._segments.get(seg_id)
        if record is None or record.role != "me" or not record.source.strip():
            return
        source = record.source
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            self._ensure_voice()
            self._my_queue.put_nowait(_Job(seg_id=seg_id, source=source, t0=time.monotonic()))

        self._submit(action, "重新翻译")

    def lookup_pcm(self, text: str) -> bytes | None:
        """内存 → 磁盘 → 分块拼接，三级查找。找不到返回 None。

        为什么需要「分块拼接」这一级：`speech_chunks()` 会把超过 450 字符的
        回答切成几段分别合成，缓存也是按段存的。而重播按钮传进来的是**整段
        译文**，所以长回答永远命不中整句键 —— 这就是「重播还要等好久」的
        真正原因。把各段拼起来即可，音频本来就是顺序播放的。
        """
        text = (text or "").strip()
        if not text:
            return None
        cached = self._pcm_cache.get(text)
        if cached is not None:
            return cached.pcm
        settings = self._tts_settings()
        disk = self.disk_cache.get(cache_key(text, settings))
        if disk:
            return disk

        pieces: list[bytes] = []
        for chunk in speech_chunks(text):
            part = self._pcm_cache.get(chunk)
            data = part.pcm if part is not None else self.disk_cache.get(cache_key(chunk, settings))
            if not data:
                return None
            pieces.append(data)
        return b"".join(pieces) if pieces else None

    # ------------------------------------------------------------ 讲稿朗读台
    def script_settings(self, language: str) -> dict[str, Any]:
        """讲稿用的合成参数：除语种提示外与现场发声完全一致。

        一致很重要——预渲染时用一套参数、现场用另一套，缓存就全部命不中，
        提前渲染也就白做了。
        """
        model, voice = self.prerender_voice()
        values = speech_settings(self.settings, language=language, model=model, voice=voice)
        values["tts_enable_ssml"] = False  # 讲稿逐句点读，不需要跨句停顿
        mode = str(self.settings.get("prerender_instruction", "b") or "b")
        if mode in ("b", "none"):
            # 和「声音对比」完全同一套参数：语速、音调固定 1.0；
            # b follows B's automatic delivery policy; Qwen uses the voice's
            # original delivery. none = A / original (no instruction).
            values["tts_rate"] = 1.0
            values["tts_pitch"] = 1.0
            instruction = ""
            if mode == "b":
                instruction = natural_delivery_instruction(str(values.get("tts_model", "")), language)
            values["tts_instruction"] = instruction
        return values

    def prerender_voice(self) -> tuple[str | None, str | None]:
        """提前渲染（讲稿台 / 问答库 / 快捷用语）用的合成模型和音色。

        答辩现场要快，建议用 flash；提前渲染不赶时间，用 plus 更好听。两者分开设置；
        提前渲染没单独设时（留空）就跟现场一致。
        """
        model = str(self.settings.get("prerender_tts_model", "") or "").strip()
        voice = str(self.settings.get("prerender_voice_id", "") or "").strip()
        if not model or not voice:
            return None, None
        return model, voice

    def render_script_line(self, text: str, language: str, *, cancel=None) -> tuple[bytes, float]:
        """合成讲稿里的一句并落盘。命中缓存就直接返回，不再请求服务器。"""
        from .tts_session import synthesize_once

        text = (text or "").strip()
        if not text:
            raise ApiError("空句子无需渲染")
        settings = self.script_settings(language)
        spoken = self._humanize_prerender(text, language)
        key = cache_key(spoken, settings)
        pcm = self.disk_cache.get(key)
        if not pcm:
            if self.client is None:
                self._build_clients()
            if not str(settings.get("voice", "")).strip():
                raise ApiError("请先在设置里填写克隆音色 voice_id")
            pcm = synthesize_once(self.client, settings, spoken, cancel_event=cancel)
            # pinned：讲稿音频永不参与 LRU 淘汰，否则现场讲到一半可能已经
            # 被实时翻译产生的新缓存挤掉，「断网也能讲」就成了空话。
            if not self.disk_cache.put(key, pcm, pinned=True):
                raise ApiError("音频未能保存到本地，原有版本仍保留；请检查磁盘空间")
        self.render_manifest.set(RenderManifest.entry_id("script", text, language, settings), key)
        return pcm, len(pcm) / max(SPEECH_SAMPLE_RATE * 2, 1)

    def _script_keys(self, text: str, language: str) -> tuple[str, str]:
        """(按当前全部参数的键, 同音色最后一次渲染登记的键)。"""
        text = (text or "").strip()
        settings = self.script_settings(language)
        exact = cache_key(self._humanize_prerender(text, language), settings)
        saved = self.render_manifest.get(RenderManifest.entry_id("script", text, language, settings))
        return exact, saved

    def script_render_state(self, text: str, language: str) -> str:
        """'exact' = 按当前参数渲染过；'saved' = 同一音色以前渲染过（语速/音量/语气/小失误
        等参数后来改过），照样能播；'' = 这个音色还没渲染。"""
        exact, saved = self._script_keys(text, language)
        if self.disk_cache.has(exact):
            if saved != exact:   # 老版本渲染的没登记过：顺手补登记，以后改参数也不丢
                self.render_manifest.set(RenderManifest.entry_id(
                    "script", (text or "").strip(), language, self.script_settings(language)), exact)
            return "exact"
        if saved and self.disk_cache.has(saved):
            return "saved"
        return ""

    # ---------------- 整段合成（和声音对比 B 一样，一次请求读完一段）----------------
    SCRIPT_GROUP_LIMIT = 450   # 字符；和现场发声同一上限，低于接口 600 字符的限制

    def script_render_unit(self) -> str:
        return "sentence" if str(self.settings.get("script_render_unit", "group") or "group") == "sentence" else "group"

    def script_groups(self, texts) -> list[list[str]]:
        """一段里的句子怎么分块合成：尽量整段一次，超过上限才在句边界拆开。"""
        return group_sentences([str(t).strip() for t in texts], max_per_group=40, limit=self.SCRIPT_GROUP_LIMIT)

    @staticmethod
    def _group_display(texts, language: str) -> str:
        joiner = "" if language == "zh" else " "
        return joiner.join(str(t).strip() for t in texts if str(t).strip())

    def _script_group_keys(self, texts, language: str) -> tuple[str, str, str]:
        """(整段显示文本, 当前参数的键, 同音色最后一次渲染的键)。"""
        display = self._group_display(texts, language)
        settings = self.script_settings(language)
        exact = cache_key(self._humanize_prerender(display, language), settings)
        saved = self.render_manifest.get(RenderManifest.entry_id("script-group", display, language, settings))
        return display, exact, saved

    def script_group_state(self, texts, language: str) -> str:
        display, exact, saved = self._script_group_keys(texts, language)
        if not display:
            return ""
        if self.disk_cache.has(exact):
            return "exact"
        if saved and self.disk_cache.has(saved):
            return "saved"
        return ""

    def script_group_pcm(self, texts, language: str) -> bytes | None:
        display, exact, saved = self._script_group_keys(texts, language)
        if not display:
            return None
        pcm = self.disk_cache.get(exact)
        if not pcm and saved and saved != exact:
            pcm = self.disk_cache.get(saved)
        return pcm

    def script_group_slices(self, texts, language: str) -> list[bytes] | None:
        """整段音频按句切开（切在句间停顿里）。没渲染返回 None，不请求服务器。"""
        from .script_desk import split_pcm_by_text

        display, exact, saved = self._script_group_keys(texts, language)
        if not display:
            return None
        # 按缓存键记住切好的结果：重开讲稿台、改设置刷新状态时不必把整篇音频再读一遍
        key = exact if self.disk_cache.has(exact) else (saved if saved and self.disk_cache.has(saved) else "")
        if not key:
            return None
        memo_key = (key, tuple(texts))
        slices = self._group_slice_memo.get(memo_key)
        if slices is None:
            pcm = self.disk_cache.get(key)
            if not pcm:
                return None
            slices = split_pcm_by_text(pcm, list(texts), sample_rate=SPEECH_SAMPLE_RATE)
            if len(self._group_slice_memo) > 256:
                self._group_slice_memo.clear()
            self._group_slice_memo[memo_key] = slices
        return slices

    def render_script_group(self, texts, language: str, *, cancel=None) -> list[float]:
        """整段一次合成并落盘（pinned）。返回每句切出来的秒数。"""
        from .tts_session import synthesize_once

        texts = [str(t).strip() for t in texts if str(t).strip()]
        display = self._group_display(texts, language)
        if not display:
            raise ApiError("空段落无需渲染")
        settings = self.script_settings(language)
        spoken = self._humanize_prerender(display, language)
        key = cache_key(spoken, settings)
        if not self.disk_cache.has(key):
            if self.client is None:
                self._build_clients()
            if not str(settings.get("voice", "")).strip():
                raise ApiError("请先在设置里填写克隆音色 voice_id")
            pcm = synthesize_once(self.client, settings, spoken, cancel_event=cancel)
            if not self.disk_cache.put(key, pcm, pinned=True):
                raise ApiError("音频没能写入本地缓存（磁盘空间不足？）")
        self.render_manifest.set(RenderManifest.entry_id("script-group", display, language, settings), key)
        slices = self.script_group_slices(texts, language) or []
        rate = SPEECH_SAMPLE_RATE * 2
        return [len(piece) / rate for piece in slices]

    # ---------------- 讲稿音频管理：单句单独重做 / 换语气 / 删除 ----------------
    def _override_entry(self, text: str, language: str) -> str:
        return RenderManifest.entry_id("override", (text or "").strip(), language, self.script_settings(language))

    def script_override(self, text: str, language: str) -> tuple[bytes, str] | None:
        """这句单独做过的版本 (音频, 语气名)；没有返回 None。只认当前音色。"""
        raw = self.script_overrides.get(self._override_entry(text, language))
        if not raw:
            return None
        key, _sep, style = raw.partition("\t")
        pcm = self.disk_cache.get(key)
        return (pcm, style or "单独版本") if pcm else None

    def render_script_override(self, text: str, language: str, *, instruction: str | None = None,
                               style: str = "", cancel=None) -> float:
        """单独重新合成这一句（可换语气），**总是重新请求**，得到新的一版。返回秒数。"""
        from .tts_session import synthesize_once

        text = (text or "").strip()
        if not text:
            raise ApiError("空句子无需渲染")
        settings = dict(self.script_settings(language))
        if instruction is not None:
            if instruction and not tts_supports(str(settings.get("tts_model", "")), "instruction"):
                raise ApiError("当前渲染音色的合成模型不支持语气指令，请换 qwen-audio / CosyVoice 音色")
            settings["tts_instruction"] = instruction
        if self.client is None:
            self._build_clients()
        if not str(settings.get("voice", "")).strip():
            raise ApiError("请先在设置里填写克隆音色 voice_id")
        # 单独版本的键和整段/逐句的分开，并带上时间戳：每点一次都是新的一版，不会覆盖别的
        key = cache_key(f"{text}\x1foverride\x1f{time.time_ns()}", settings)
        pcm = synthesize_once(self.client, settings, text, cancel_event=cancel)
        if not self.disk_cache.put(key, pcm, pinned=True):
            raise ApiError("音频没能写入本地缓存（磁盘空间不足？）")
        entry = self._override_entry(text, language)
        old = self.script_overrides.get(entry)
        self.script_overrides.set(entry, f"{key}\t{style or '单独重做'}")
        if old:
            self.disk_cache.discard(old.partition("\t")[0])
        return len(pcm) / (SPEECH_SAMPLE_RATE * 2)

    def clear_script_override(self, text: str, language: str, *, keep_audio: bool = False) -> bool:
        entry = self._override_entry(text, language)
        raw = self.script_overrides.get(entry)
        if not raw:
            return False
        if keep_audio:
            self.script_overrides.set(f"{entry}:previous:{time.time_ns()}", raw)
        else:
            self.disk_cache.discard(raw.partition("\t")[0])
        self.script_overrides.remove(entry)
        return True

    def delete_script_line_audio(self, text: str, language: str) -> None:
        """删掉这句逐句渲染的音频和单独版本（整段合成的那块另由 delete_script_group_audio 删）。"""
        self.clear_script_override(text, language)
        text = (text or "").strip()
        exact, saved = self._script_keys(text, language)
        for key in {exact, saved} - {""}:
            self.disk_cache.discard(key)
        self.render_manifest.remove(RenderManifest.entry_id("script", text, language, self.script_settings(language)))

    def delete_script_group_audio(self, texts, language: str) -> None:
        display, exact, saved = self._script_group_keys(texts, language)
        if not display:
            return
        for key in {exact, saved} - {""}:
            self.disk_cache.discard(key)
        self.render_manifest.remove(RenderManifest.entry_id(
            "script-group", display, language, self.script_settings(language)))
        self._group_slice_memo.clear()

    def script_line_ready(self, text: str, language: str) -> bool:
        """这一句是否已经在本地，可以断网朗读。"""
        return bool(self.script_render_state(text, language))

    def script_pcm(self, text: str, language: str) -> bytes | None:
        """取这一句已渲染好的音频（当前参数优先，退回同音色的旧渲染）。
        没有就返回 None，**不去请求服务器**。"""
        exact, saved = self._script_keys(text, language)
        pcm = self.disk_cache.get(exact)
        if not pcm and saved and saved != exact:
            pcm = self.disk_cache.get(saved)
        return pcm

    def play_script_line(self, text: str, language: str, *, zh: str = "") -> None:
        """点播讲稿里的一句：只从本地读，读不到就明说，不偷偷去请求服务器。

        现场点一句要等五秒，比直接告诉你「这句没渲染」糟糕得多。
        """
        self._claim_playback()
        self.interrupt_speech(user=False)
        generation = self._speech_generation
        def action() -> None:
            if generation != self._speech_generation:
                return
            pcm = self.script_pcm(text, language)
            if not pcm:
                self.callbacks.on_error("这一句还没有渲染，请先在讲稿台点「渲染」")
                return
            self._play_cached(CachedUtterance(pcm=pcm, zh=zh), text)

        self._submit(action, "讲稿朗读")

    def cached_playback_state(self) -> dict:
        """{tag, played(秒), total(秒), active}：本地整段播放播到哪了（暂停时 played 不涨）。"""
        return dict(self._cached_state)

    def play_script_segment(self, pcm: bytes, *, text: str = "", zh: str = "", tag: str = "") -> None:
        """把整段拼好的音频一次播出去。

        整段拼接而不是一句一句排队，是因为句与句之间不会多出调度间隙——
        听起来就是连着讲下来的。界面上的光标改由定时器按各句时长推进。
        """
        if not pcm:
            self.callbacks.on_error("这一段还没有渲染完整，无法朗读")
            return
        self._claim_playback()
        self.interrupt_speech(user=False)
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            self._play_cached(CachedUtterance(pcm=pcm, zh=zh), text, tag=tag)

        self._submit(action, "讲稿整段朗读")

    # ------------------------------------------------------------ 问答库预案
    # 预案回答复用讲稿台的英文合成参数：与现场发声只差「不发 SSML」，默认
    # 句间停顿为 0 时两套缓存键完全相同，现场说过的句子也能直接命中。
    def answer_settings(self) -> dict[str, Any]:
        return self.script_settings("en")

    def _answer_override_entry(self, text: str) -> str:
        return RenderManifest.entry_id("answer-override", (text or "").strip(), "en", self.answer_settings())

    def answer_override(self, text: str) -> tuple[list[str], str] | None:
        """这条回答单独重做 / 换语气的版本 (各块的键, 语气名)；文件不全返回 None。"""
        raw = self.answer_overrides.get(self._answer_override_entry(text))
        if not raw:
            return None
        joined, _sep, style = raw.partition("\t")
        keys = [key for key in joined.split(",") if key]
        if keys and all(self.disk_cache.has(key) for key in keys):
            return keys, style or "单独版本"
        return None

    def _answer_catalog_id(self, text: str) -> str:
        return RenderManifest.entry_id("answer-audio", (text or "").strip(), "en", {})

    def _answer_versions(self, text: str) -> list[dict]:
        try:
            values = json.loads(self.answer_catalog.get(self._answer_catalog_id(text)) or "[]")
        except (ValueError, TypeError):
            return []
        return [v for v in values if isinstance(v, dict) and isinstance(v.get("keys"), list)
                and isinstance(v.get("model"), str) and isinstance(v.get("voice"), str)
                and v["keys"] and all(isinstance(k, str) and len(k) == 64 and all(c in "0123456789abcdef" for c in k)
                                      for k in v["keys"])] if isinstance(values, list) else []

    def _remember_answer_audio(self, text: str, settings: dict, keys: list[str], *, style: str = "") -> None:
        if not keys or not any(self.disk_cache.has(k) for k in keys):
            return
        # Preserve older complete versions even when a later attempt is cancelled.
        with self._answer_catalog_lock:
            values = [v for v in self._answer_versions(text) if v["keys"] != keys]
            values.append({"model": str(settings.get("tts_model", "")), "voice": str(settings.get("voice", "")),
                           "keys": list(keys), "style": style})
            self.answer_catalog.set(self._answer_catalog_id(text), json.dumps(values, ensure_ascii=False))

    def _discover_legacy_answer_audio(self, text: str, settings: dict) -> None:
        pairs = {(str(settings.get("tts_model", "")), str(settings.get("voice", "")))}
        pairs.add((str(self.settings.get("tts_model", "")), str(self.settings.get("tts_voice_id", ""))))
        for voice in self.settings.get("voice_library", []) or []:
            if isinstance(voice, dict):
                pairs.add((str(voice.get("target_model", "")), str(voice.get("voice_id", ""))))
        marker = (text, tuple(sorted(pairs)))
        if marker in self._answer_legacy_checked:
            return
        self._answer_legacy_checked.add(marker)
        for model, voice in sorted(pairs):
            if not model or not voice:
                continue
            values = {**settings, "tts_model": model, "voice": voice}
            keys = []
            for chunk in speech_chunks(text):
                entry = RenderManifest.entry_id("answer", chunk, "en", values)
                saved = self.render_manifest.get(entry)
                exact = cache_key(self._humanize_prerender(chunk, "en"), values)
                keys.append(saved if saved and self.disk_cache.has(saved) else exact)
            self._remember_answer_audio(text, values, keys)
            override = self.answer_overrides.get(RenderManifest.entry_id("answer-override", text, "en", values))
            if override:
                raw, _, style = override.partition("\t")
                self._remember_answer_audio(text, values, [k for k in raw.split(",") if k], style=style)

    def _current_answer_keys(self, text: str) -> list[str]:
        """每块回答用哪份音频：单独版本优先；当前参数有就用当前的，否则用同音色最后一次渲染的。"""
        override = self.answer_override(text)
        if override:
            self._remember_answer_audio(text, self.answer_settings(), override[0], style=override[1])
            return override[0]
        settings = self.answer_settings()
        keys: list[str] = []
        for chunk in speech_chunks((text or "").strip()):
            exact = cache_key(self._humanize_prerender(chunk, "en"), settings)
            entry = RenderManifest.entry_id("answer", chunk, "en", settings)
            saved = self.render_manifest.get(entry)
            if self.disk_cache.has(exact):
                if saved != exact:
                    self.render_manifest.set(entry, exact)
            elif saved and self.disk_cache.has(saved):
                exact = saved
            keys.append(exact)
        self._remember_answer_audio(text, settings, keys)
        return keys

    def _answer_keys(self, text: str) -> list[str]:
        text = (text or "").strip()
        current = self._current_answer_keys(text)
        if current and all(self.disk_cache.has(k) for k in current):
            return current
        settings = self.answer_settings()
        versions = self._answer_versions(text)
        if not any(all(self.disk_cache.has(k) for k in v["keys"]) for v in versions):
            self._discover_legacy_answer_audio(text, settings)
            versions = self._answer_versions(text)
        # Never splice chunks from different models/voices into one answer.
        for version in reversed(versions):
            if all(self.disk_cache.has(k) for k in version["keys"]):
                return version["keys"]
        if any(self.disk_cache.has(k) for k in current):
            return current
        for version in reversed(versions):
            if any(self.disk_cache.has(k) for k in version["keys"]):
                return version["keys"]
        return current

    def answer_audio_source(self, text: str) -> dict:
        keys = self._answer_keys(text)
        for version in reversed(self._answer_versions(text)):
            if version["keys"] == keys:
                return {"model": version["model"], "voice": version["voice"], "style": version.get("style", "")}
        return {}

    def answer_ready(self, text: str) -> bool:
        """这条回答是否已经整段在本地，可以不请求服务器直接播。"""
        keys = self._answer_keys(text)
        return bool(keys) and all(self.disk_cache.has(key) for key in keys)

    def answer_pcm(self, text: str) -> bytes | None:
        """先找预渲染的整段，再退回现场发声留下的缓存。都没有返回 None。"""
        keys = self._answer_keys(text)
        if keys:
            pieces = [self.disk_cache.get(key) for key in keys]
            if all(pieces):
                return b"".join(pieces)
        return self.lookup_pcm(text)

    def render_answer(self, text: str, *, cancel=None) -> int:
        """把一条预案回答合成并钉在本地（pinned，不参与淘汰）。返回新合成的块数。

        在调用方的后台线程里同步执行；不碰直播会话和声卡。
        """
        from .tts_session import synthesize_once

        text = (text or "").strip()
        if not text:
            raise ApiError("空回答无需渲染")
        if cancel is not None and cancel.is_set():
            raise ApiError("已取消")
        if self.answer_ready(text):
            return 0
        settings = self.answer_settings()
        if str(settings.get("tts_model", "")) == "qwen-audio-3.1-tts-next":
            if not Path(str(settings.get("tts_reference_audio", "") or "")).is_file():
                raise ApiError("请先在设置中选择 Next 本机参考录音")
        elif not str(settings.get("voice", "")).strip():
            raise ApiError("请先在设置里填写克隆音色 voice_id")
        rendered = 0
        chunks = list(speech_chunks(text))
        keys = self._current_answer_keys(text)
        for chunk, key in zip(chunks, keys):
            if cancel is not None and cancel.is_set():
                raise ApiError("已取消")
            spoken = self._humanize_prerender(chunk, "en")
            entry = RenderManifest.entry_id("answer", chunk, "en", settings)
            if self.disk_cache.has(key):
                self.render_manifest.set(entry, key)
                continue
            if self.client is None:
                self._build_clients()
            # Batch stop gates the next chunk; finish and save the request already
            # submitted, rather than throwing away paid audio as it arrives.
            pcm = synthesize_once(self.client, settings, spoken)
            if not self.disk_cache.put(key, pcm, pinned=True):
                raise ApiError("音频未能保存到本地，请检查磁盘空间；已保存的部分仍保留")
            self.render_manifest.set(entry, key)
            self._remember_answer_audio(text, settings, keys)
            rendered += 1
        return rendered

    def answer_state(self, text: str) -> str:
        """custom = 单独重做/换过语气；exact = 按当前参数渲染；saved = 同音色旧设置；
        partial = 只渲染了一部分；'' = 没渲染。"""
        text = (text or "").strip()
        if not text:
            return ""
        selected = self._answer_keys(text)
        current = self._current_answer_keys(text)
        if selected != current and selected and all(self.disk_cache.has(k) for k in selected):
            return "other"
        if selected != current and any(self.disk_cache.has(k) for k in selected):
            return "partial"
        if self.answer_override(text):
            return "custom"
        settings = self.answer_settings()
        states = []
        for chunk in speech_chunks(text):
            exact = cache_key(self._humanize_prerender(chunk, "en"), settings)
            if self.disk_cache.has(exact):
                states.append("exact")
                continue
            saved = self.render_manifest.get(RenderManifest.entry_id("answer", chunk, "en", settings))
            states.append("saved" if saved and self.disk_cache.has(saved) else "")
        if all(states):
            return "exact" if all(state == "exact" for state in states) else "saved"
        return "partial" if any(states) else ""

    def answer_audio_seconds(self, text: str) -> float:
        """本地音频有多长（只看文件大小，不读内容，几百条也很快）。"""
        keys = self._answer_keys(text)
        return sum(self.disk_cache.size_of(key) for key in keys) / (SPEECH_SAMPLE_RATE * 2)

    def render_answer_override(self, text: str, *, instruction: str | None = None,
                               style: str = "", cancel=None) -> float:
        """这条回答单独重新合成（可换语气），**总是重新请求**，得到新的一版。返回秒数。"""
        from .tts_session import synthesize_once

        text = (text or "").strip()
        if not text:
            raise ApiError("空回答无需渲染")
        settings = dict(self.answer_settings())
        if instruction is not None:
            if instruction and not tts_supports(str(settings.get("tts_model", "")), "instruction"):
                raise ApiError("当前渲染音色的合成模型不支持语气指令，请换 qwen-audio / CosyVoice 音色")
            settings["tts_instruction"] = instruction
        if not str(settings.get("voice", "")).strip():
            raise ApiError("请先在设置里填写克隆音色 voice_id")
        keys, total = [], 0
        stamp = time.time_ns()
        for number, chunk in enumerate(speech_chunks(text)):
            if cancel is not None and cancel.is_set():
                raise ApiError("已取消")
            if self.client is None:
                self._build_clients()
            pcm = synthesize_once(self.client, settings, chunk, cancel_event=cancel)
            key = cache_key(f"{chunk}\x1fanswer-override\x1f{stamp}\x1f{number}", settings)
            if not self.disk_cache.put(key, pcm, pinned=True):
                raise ApiError("音频没能写入本地缓存（磁盘空间不足？）")
            keys.append(key)
            total += len(pcm)
        entry = self._answer_override_entry(text)
        old = self.answer_overrides.get(entry)
        self.answer_overrides.set(entry, ",".join(keys) + f"\t{style or '重做一版'}")
        self._remember_answer_audio(text, settings, keys, style=style or "重做一版")
        if old:
            for key in old.partition("\t")[0].split(","):
                if key and key not in keys:
                    self.disk_cache.discard(key)
        return total / (SPEECH_SAMPLE_RATE * 2)

    def clear_answer_override(self, text: str) -> bool:
        entry = self._answer_override_entry(text)
        raw = self.answer_overrides.get(entry)
        if not raw:
            return False
        for key in raw.partition("\t")[0].split(","):
            if key:
                self.disk_cache.discard(key)
        self.answer_overrides.remove(entry)
        return True

    def delete_answer_audio(self, text: str) -> None:
        """删掉这条回答在本机的全部音频（默认版本、旧设置版本、单独版本）。"""
        text = (text or "").strip()
        if not text:
            return
        self.clear_answer_override(text)
        for version in self._answer_versions(text):
            for key in version["keys"]:
                self.disk_cache.discard(key)
        self.answer_catalog.remove(self._answer_catalog_id(text))
        settings = self.answer_settings()
        for chunk in speech_chunks(text):
            exact = cache_key(self._humanize_prerender(chunk, "en"), settings)
            entry = RenderManifest.entry_id("answer", chunk, "en", settings)
            saved = self.render_manifest.get(entry)
            for key in {exact, saved} - {""}:
                self.disk_cache.discard(key)
            self.render_manifest.remove(entry)

    DUPLICATE_WINDOW = 2.5   # 秒：这么短时间内同一段再点一次，一律当成误触

    def speak_prepared(self, text: str, *, label: str = "预案回答", zh: str = "",
                       force: bool = False) -> bool:
        """播放一条（可能改过的）回答：本地有就秒播，没有才现场合成。

        返回值只表示「提交时本地是否已有整段音频」，给界面提示用；真正播放
        在命令线程里进行，和其他发声共用同一条顺序队列。

        **同一段话绝不会叠着读两遍**：它还在准备或正在播放时再提交一次（连点、
        弹窗和面板各点一下），直接忽略。force=True（「重播」按钮）先停掉当前的再读。
        """
        text = (text or "").strip()
        if not text:
            return False
        # Clicking an explicit answer/phrase playback is a request to speak now,
        # including after F8/F9 paused a script or the user paused another answer.
        self.resume_speech()
        now = time.monotonic()
        last_text, last_at = self._last_prepared
        if not force and text == last_text and (now - last_at < self.DUPLICATE_WINDOW or self.is_speaking()):
            self.callbacks.on_status("这段正在读（或刚提交过），不会再读第二遍；要重读请点「重播」")
            return self.answer_ready(text)
        if force:
            self.interrupt_speech()
        else:
            self.interrupt_speech(user=False)
        self._last_prepared = (text, now)
        ready = self.answer_ready(text)
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                self.callbacks.on_status("朗读请求在开始前被其他操作取消；如需播放，请再点朗读")
                return
            pcm = self.answer_pcm(text)
            if pcm is None:
                # 改过的回答或没预渲染：走和 AI 助答完全相同的现场合成路径
                self._ensure_voice()
            seg_id = self._register_segment("me", f"（{label}）")
            if pcm is not None:
                self._play_cached(CachedUtterance(pcm=pcm, zh=zh), text)
            else:
                self._queue_speech(text)
            self.memory.add_turn(Turn(role="me", source=f"（{label}）", translation=text))
            self._store_segment(seg_id, text, latency_ms=0)
            self.callbacks.on_my_done(seg_id, text, 0)

        self._submit(action, "播放预案回答")
        return ready

    def replay_english(self, text: str) -> None:
        text = text.strip()
        self._claim_playback()
        self.interrupt_speech()
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            pcm = self.answer_pcm(text)
            if pcm:
                zh = self._pcm_cache[text].zh if text in self._pcm_cache else ""
                self._play_cached(CachedUtterance(pcm=pcm, zh=zh), text)
                return
            self._ensure_voice()
            self._queue_speech(text)
            self.callbacks.on_status("该句不在本地缓存，已请求服务器重新合成（完成后自动入缓存）")

        self._submit(action, "重播")

    def _play_cached(self, cached: CachedUtterance, text: str, *, tag: str = "") -> None:
        if not cached.pcm:
            self.callbacks.on_error("本地语音缓存为空，请重新合成后再朗读")
            return
        self._claim_playback()
        self.interrupt_speech(user=False)
        with self._player_lock:
            self._ensure_player()
        token = threading.Event()
        self._playback_cancel = token
        sample_rate = SPEECH_SAMPLE_RATE
        duration = len(cached.pcm) / max(sample_rate * 2, 1)
        self._suppress_until = time.monotonic() + duration + 0.6
        def write_block(data: bytes) -> bool:
            """每块单独拿锁：整段播放期间绝不长时间占着播放锁。

            以前是整段播完才放锁，结果讲稿台一播，「开始答辩」「原声直通」都要等它
            播完才能拿到播放器——按钮看起来就是卡死了。
            """
            while not token.is_set() and not self._closed:
                with self._player_lock:
                    if not (self._speech_paused.is_set() or getattr(self, "_cached_input_pause", False)):
                        if self.player is None:
                            self._ensure_player()
                        try:
                            self.player.write(data)
                        except Exception:
                            self._recover_player(data)
                        return True
                time.sleep(0.03)
            return False

        state = {"tag": tag, "played": 0.0, "total": duration, "active": True}
        self._cached_state = state

        def write_loop() -> None:
            try:
                block = sample_rate * 2 // 25
                started = False
                for offset in range(0, len(cached.pcm), block):
                    while self._speech_paused.is_set() or getattr(self, "_cached_input_pause", False):
                        # 暂停：不占锁、不出声；持续底噪线程会接手背景音
                        if token.is_set() or self._closed:
                            return
                        time.sleep(0.03)
                    if token.is_set() or self._closed:
                        return
                    piece = cached.pcm[offset:offset + block]
                    # Long prepared answers must honor the global switch/volume
                    # immediately too, rather than retaining a start-time snapshot.
                    mixed = self._current_room_tone().mix(piece)
                    self._last_voice_write = time.monotonic()
                    if not write_block(mixed):
                        return
                    state["played"] += len(piece) / (sample_rate * 2)
                    # 放锁后让一下：Python 的锁不讲先来后到，不歇这 1 毫秒，别的线程可能一直抢不到
                    time.sleep(0.001)
                    if not started:
                        started = True
                        import logging
                        logging.getLogger("defense.lifecycle").info(
                            "cached_playback_started tag=%s seconds=%.2f devices=%s monitor=%s",
                            tag, duration, getattr(self.player, "devices", "configured"), self.settings.get("monitor_enabled", True))
                        self.callbacks.on_tts_sentence_started(text, cached.zh)
                        self.callbacks.on_tts_sentence_duration(text, duration)
                        self.callbacks.on_status("本地语音已开始播放（Esc 停止）")
                    self.callbacks.on_tts_sentence_progress(text, state["played"] / max(duration, .001))
                # 持续底噪开着时不做淡出：淡到 0 再被底噪接上，反而听得出一个「坑」
                tail = b"" if self._ambience_bed_on() else self._current_room_tone().tail()
                for offset in range(0, len(tail), block):
                    if token.is_set() or self._closed or not write_block(tail[offset:offset + block]):
                        return
            except Exception as exc:
                import logging
                logging.getLogger("defense.lifecycle").exception("cached_playback_failed tag=%s", tag)
                self.callbacks.on_error(f"本地重播播放失败：{exc}")
            else:
                self.callbacks.on_status("已从本地缓存重播（未请求服务器）")
            finally:
                state["active"] = False

        thread = threading.Thread(target=write_loop, name="defense-cache-playback", daemon=True)
        self._cached_thread = thread
        thread.start()

    def speak_as_me(self, text: str, *, label: str = "AI 助答") -> None:
        """Speak prepared English as the candidate and record it into context."""
        text = text.strip()
        if not text:
            return
        self._claim_playback()
        self.interrupt_speech()
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            self._ensure_voice()
            seg_id = self._register_segment("me", f"（{label}）")
            self._queue_speech(text)
            self.memory.add_turn(Turn(role="me", source=f"（{label}）", translation=text))
            self._store_segment(seg_id, text, latency_ms=0)
            self.callbacks.on_my_done(seg_id, text, 0)

        self._submit(action, "播放回答")

    # ---------------------------------------------------------- command impls
    def _build_clients(self) -> None:
        api_key = self.settings.get_api_key()
        workspace = self.settings.workspace_id
        if not workspace:
            raise ApiError("尚未设置 Workspace ID，请先在设置中填写")
        proxy_mode = self.settings.proxy_mode
        self.client = BailianClient(
            api_key,
            workspace,
            timeout=int(self.settings.shared.get("request_timeout", 45)),
            proxy=self.settings.ws_proxy,
            proxy_mode=proxy_mode,
        )
        self.translator = ContextTranslator(
            api_key=api_key,
            workspace_id=workspace,
            model=str(self.settings.get("llm_model", "qwen-plus")),
            proxy=self.settings.ws_proxy,
            proxy_mode=proxy_mode,
            timeout=int(self.settings.shared.get("request_timeout", 45)),
            backup=backup_channel(self.settings),   # 开了第二通道：谁先出字用谁，只说一次
        )

    def _tts_settings(self) -> dict[str, Any]:
        values = speech_settings(self.settings)
        # 只有确实要发 SSML 时才打开这个开关：开着却送纯文本，某些模型会把
        # 整段当标记解析。_sentence_break_ms() 已经把「模型不支持」折算成 0。
        values["tts_enable_ssml"] = bool(self._sentence_break_ms())
        return values

    def _live_tts_settings(self) -> dict[str, Any]:
        """现场发声会话用的参数 = 合成参数 + 合成第二通道。

        第二通道只加在这里：讲稿台 / 问答库 / 快捷用语的提前渲染、声音对比都用 _tts_settings() /
        script_settings()，不带这几项，所以不会被换成备用音色，缓存键也不受这个开关影响。
        """
        values = self._tts_settings()
        if bool(self.settings.get("backup_tts_enabled", False)):
            values.update({
                "backup_tts_enabled": True,
                "backup_tts_model": str(self.settings.get("backup_tts_model", "") or ""),
                "backup_tts_voice_id": str(self.settings.get("backup_tts_voice_id", "") or ""),
                "backup_tts_mode": str(self.settings.get("backup_tts_mode", "delayed") or "delayed"),
            })
        return values

    def apply_settings(self) -> None:
        def action():
            was_active = self._session_active
            was_listening = self._listening
            self._do_stop_defense()
            self.client = None
            self.translator = None
            self._pcm_cache.clear()
            self._cache_bytes = 0
            self.reload_context()
            if was_active:
                self._do_start_defense(reset_audio=False)
                if was_listening and not self._listening:
                    self._do_toggle_listening()
            self.callbacks.on_status("设置已应用，发声参数与设备已更新")
        self._submit(action, "应用设置")

    def _ensure_player(self) -> None:
        with self._player_lock:
            if self.player is None:
                candidate = MultiOutputPlayer(self._output_devices(), SPEECH_SAMPLE_RATE)
                candidate.__enter__()
                self.player = candidate

    def _ensure_voice(self) -> None:
        next_model = str(self.settings.get("tts_model", "")) == "qwen-audio-3.1-tts-next"
        if next_model and not Path(str(self.settings.get("tts_reference_audio", "") or "")).is_file():
            raise ApiError("Next 模型需要先在设置中选择本机参考录音")
        if not next_model and not str(self.settings.get("tts_voice_id", "") or "").strip():
            raise ApiError("尚未填写克隆音色 voice_id，请在设置中填写控制台复刻的音色")
        for entry in ([] if next_model else self.settings.get("voice_library", [])):
            if isinstance(entry, dict) and entry.get("voice_id") == self.settings.get("tts_voice_id"):
                bound = entry.get("target_model")
                if bound and bound != self.settings.get("tts_model"):
                    raise ApiError(f"音色绑定模型为 {bound}，请在设置中选择对应模型")
        if self.tts is None:
            self._build_clients()
        self._ensure_player()
        if self.tts is None:
            generation = self._tts_generation

            def current() -> bool:
                return (generation == self._tts_generation
                        and not self._stopping and not self._closed)

            self.tts = create_tts_session(
                self.client,
                self._live_tts_settings(),
                on_audio=lambda pcm: self._on_tts_audio(pcm) if current() else None,
                on_status=lambda message: self.callbacks.on_status(message) if current() else None,
                on_error=lambda message: self.callbacks.on_error(message) if current() else None,
                on_first_audio=lambda text: self._on_first_audio(text) if current() else None,
                on_utterance_done=lambda text, size: self._on_utterance_done(text, size) if current() else None,
                on_utterance_failed=lambda text: self._on_utterance_failed(text) if current() else None,
                on_audio_ready=lambda text, size: self.callbacks.on_tts_sentence_duration(
                    self._display_for(text), size / (SPEECH_SAMPLE_RATE * 2)
                ) if current() else None,
            )
            self.tts.start()
            if isinstance(self.tts, TtsHttpSession):
                self.callbacks.on_status(
                    "自然发声：后台最多并发预生成 3 句，按原顺序连续播放"
                )

    def _output_devices(self) -> list[int | None]:
        cable = resolve_program_output(self.settings)
        devices: list[int | None] = [cable]
        if bool(self.settings.get("monitor_enabled", True)):
            devices.append(resolve_monitor_device(self.settings))
        return devices

    def _do_start_defense(self, *, reset_audio: bool = True) -> None:
        if self._session_active:
            return
        self._stopping = False
        if reset_audio:
            self._meeting_spool.reset()
            self._utterance_offsets = []
        self._committee_cancel = threading.Event()
        while True:
            try:
                self._committee_queue.get_nowait()
            except queue.Empty:
                break
        self._set_state(STATE_STARTING)
        try:
            self._build_clients()
            self.reload_context()
            self._ensure_voice()
            if self._closed:
                raise ApiError("程序正在关闭")
        except Exception:
            # 设备或音色初始化到一半失败时，不能把播放器/TTS 留给下一次启动。
            self._do_stop_defense()
            self.client = None
            self.translator = None
            raise
        # Committee listening: explicit loopback device wins; otherwise fall back
        # to the system default output so 微信/Teams 对方声音默认就能被听见。
        loopback_device = self.settings.shared.get("loopback_device")
        try:
            self._start_committee_listening(
                int(loopback_device) if loopback_device is not None else None
            )
        except Exception as exc:
            self.callbacks.on_error(f"评委声音监听启动失败（不影响翻译发声）：{exc}")
        self._session_active = True
        self._start_ambience_bed()
        self._set_state(STATE_STANDBY)
        if bool(self.settings.get("auto_record", True)):
            try:
                self._start_recording()
            except Exception as exc:
                self.callbacks.on_error(f"录音启动失败（不影响翻译）：{exc}")
        self.callbacks.on_status("答辩通道已就绪：请选择「原声」或「翻译」；未勾选持续监听时需按住说话")

    def _do_stop_defense(self) -> None:
        self._stopping = True
        self._stop_ambience_bed()
        self._tts_generation += 1
        self._listen_requested = False
        self._direct_requested = False
        self._resume_listening_after_direct = False
        self.interrupt_speech()
        self._committee_cancel.set()
        errors = []
        # Detach each resource even if another device raises during shutdown.
        for name, method in (("mic_capture", "stop"), ("asr_mic", "close"),
                             ("bridge", "stop"), ("loopback_capture", "stop"),
                             ("asr_committee", "close"), ("tts", "close")):
            resource = getattr(self, name)
            setattr(self, name, None)
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        with self._player_lock:
            player, self.player = self.player, None
            if player is not None:
                try:
                    player.close()
                except Exception as exc:
                    errors.append(f"player: {exc}")
        for name in ("recorder", "mic_recorder"):
            recorder = getattr(self, name)
            setattr(self, name, None)
            if recorder is not None:
                try:
                    recorder.stop()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        self._listening = False
        self._direct_active = False
        self._session_active = False
        self._set_state(STATE_IDLE)
        try:
            self._save_transcripts()
        except Exception as exc:
            errors.append(f"记录保存: {exc}")
        if errors:
            self.callbacks.on_error("关闭通道时出现问题，其他通道已释放：" + "; ".join(errors))

    def _do_toggle_listening(self) -> None:
        if self._listening:
            self._stop_listening()
            self._set_state(STATE_STANDBY)
            self.callbacks.on_status("翻译已暂停")
            return
        if not self._session_active:
            self._do_start_defense()
            if self._listening:
                return
        if self._direct_active:
            self._resume_listening_after_direct = False
            self._do_stop_direct()
        api_key = self.settings.get_api_key()
        workspace = self.settings.workspace_id
        if self.settings.get("my_translate_engine", "llm") == "livetranslate":
            self._start_live_translate(api_key, workspace)
            return
        values = {
            "asr_model": self.settings.get("asr_model"),
            "vad_threshold": float(self.settings.shared.get("vad_threshold", 0.0)),
            "vad_silence_ms": self._vad_silence_ms(),
            "translation_terms": self.settings.get("glossary", ""),
            "asr_context_prefix": "本次硕士论文答辩中可能出现的专业术语：",
        }
        self.asr_mic = self._create_asr(
            api_key=api_key,
            workspace_id=workspace,
            values=values,
            language="zh",
            on_preview=lambda text, _emotion: self.callbacks.on_mic_preview(text),
            on_status=self.callbacks.on_status,
            on_error=self._on_asr_error,
            on_segment=self._on_my_segment,
            combine_previews=False,
            proxy=self.settings.ws_proxy,
        )
        try:
            self.asr_mic.start()
            self.asr_mic.wait_ready()
            if self._listen_requested is False or self._closed:
                self.asr_mic.close()
                self.asr_mic = None
                self._set_state(STATE_STANDBY)
                return
            self.mic_capture = MicrophoneCapture(
                self.settings.shared.get("input_device"), self._on_mic_audio,
                sample_rate=16000, block_ms=100,
                on_error=lambda message: self.callbacks.on_error(f"麦克风采集已停止：{message}；请松开后重试"),
            )
            self.mic_capture.start()
        except Exception:
            if self.mic_capture is not None:
                self.mic_capture.stop()
                self.mic_capture = None
            self.asr_mic.close()
            self.asr_mic = None
            self._set_state(STATE_STANDBY)
            raise
        self._listening = True
        self._set_state(STATE_LISTENING)
        self.callbacks.on_status("正在聆听：说完一句停顿即翻译")

    def _start_live_translate(self, api_key: str, workspace: str) -> None:
        """同传模式：麦克风直接进同传模型，出来的英文走原有发声链路。"""
        from .livetranslate import LiveTranslateRealtime, glossary_phrases

        model = str(self.settings.get("livetranslate_model", "") or "qwen3.8-livetranslate-flash-realtime")
        self.asr_mic = LiveTranslateRealtime(
            api_key=api_key,
            workspace_id=workspace,
            model=model,
            source_language="zh",
            target_language="en",
            phrases=glossary_phrases(self.settings.glossary_terms()),
            on_preview=self.callbacks.on_mic_preview,
            on_translation=self._on_my_live_translation,
            on_status=self.callbacks.on_status,
            on_error=lambda message: self.callbacks.on_error(f"同传：{message}"),
            proxy=self.settings.ws_proxy,
        )
        self._open_mic_for(self.asr_mic)
        self.callbacks.on_status(f"同传模式（{model}）：边说边译，译文用你的音色播放")

    def _open_mic_for(self, client) -> None:
        """连上识别/同传客户端后再开麦克风；中途失败全部回滚。"""
        try:
            client.start()
            client.wait_ready()
            if self._listen_requested is False or self._closed:
                client.close()
                self.asr_mic = None
                self._set_state(STATE_STANDBY)
                return
            self.mic_capture = MicrophoneCapture(
                self.settings.shared.get("input_device"), self._on_mic_audio,
                sample_rate=16000, block_ms=100,
                on_error=lambda message: self.callbacks.on_error(f"麦克风采集已停止：{message}；请松开后重试"),
            )
            self.mic_capture.start()
        except Exception:
            if self.mic_capture is not None:
                self.mic_capture.stop()
                self.mic_capture = None
            client.close()
            self.asr_mic = None
            self._set_state(STATE_STANDBY)
            raise
        self._listening = True
        self._set_state(STATE_LISTENING)

    def _on_my_live_translation(self, source: str, english: str) -> None:
        english = (english or "").strip()
        if not english or self._stopping or self._closed or not getattr(self, "_accept_mic_tail", True):
            return
        source = (source or "").strip() or "（同传）"
        seg_id = self._register_segment("me", source)
        self.callbacks.on_my_segment(seg_id, source)
        try:
            self._my_queue.put_nowait(_Job(
                seg_id=seg_id, source=source, t0=time.monotonic(), preset_english=english, live_input=True,
            ))
        except queue.Full:
            self.callbacks.on_error("翻译队列已满，请暂停说话并检查网络")

    def _stop_listening(self) -> None:
        self._listening = False
        if self.mic_capture is not None:
            self.mic_capture.stop()
            self.mic_capture = None
        if self.asr_mic is not None:
            asr = self.asr_mic
            self.asr_mic = None
            try:
                asr.finish(timeout=8.0)
            finally:
                asr.close()
        self._listening = False

    def _do_start_direct(self) -> None:
        if self._direct_active:
            return
        if not self._session_active:
            self._do_start_defense()
        if self._direct_requested is False or self._closed:
            return
        if self._listening:
            self._stop_listening()
        self._ambience_cancel.set()
        self._resume_listening_after_direct = False
        self.bridge = DirectAudioBridge()
        self.bridge.start(
            self.settings.shared.get("input_device"),
            resolve_program_output(self.settings),
            int(self.settings.shared.get("direct_sample_rate", 48000)),
            monitor_devices=[resolve_monitor_device(self.settings)] if self.settings.get("monitor_enabled", True) else [],
            monitor_enabled=lambda: bool(self.settings.get("live_monitor_enabled", True)),
            output_lock=self._player_lock,
        )
        self._suppress_until = time.monotonic() + 1.0
        self._direct_active = True
        self._set_state(STATE_DIRECT)
        key = str(self.settings.get("direct_hotkey", "f8") or "f8").upper()
        self.callbacks.on_status(f"原声直通中（松开 {key} 结束）")

    def _do_stop_direct(self) -> None:
        if self.bridge is None and not self._direct_active:
            self._resume_listening_after_direct = False
            return
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None
        self._direct_active = False
        self._suppress_until = max(self._suppress_until, time.monotonic() + 0.8)
        self._resume_listening_after_direct = False
        self._set_state(STATE_STANDBY if self._session_active else STATE_IDLE)
        self.callbacks.on_status("原声直通结束")

    # ------------------------------------------------------- 录音与归档
    def _base_directory(self) -> Path:
        configured = str(self.settings.get("output_directory", "") or "").strip()
        base = Path(configured) if configured else default_base_dir()
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _start_recording(self) -> None:
        if self.archive_folder is not None:
            return
        self.archive_folder = new_meeting_folder(self._base_directory())
        audio_dir = self.archive_folder / "录音"
        loopback_device = self.settings.shared.get("loopback_device")
        if loopback_device is not None:
            self.recorder = DualTrackRecorder()
            self.recorder.start(
                self.settings.shared.get("input_device"),
                int(loopback_device),
                audio_dir,
                sample_rate=16000,
            )
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.mic_recorder = WavRecorder()
            self.mic_recorder.start(
                self.settings.shared.get("input_device"),
                audio_dir / f"我的麦克风_{stamp}.wav",
                16000,
            )
        self.callbacks.on_status(f"本场记录目录：{self.archive_folder}")

    def _stop_recording(self) -> None:
        if self.recorder is not None:
            recorder, self.recorder = self.recorder, None
            recorder.stop()
        if self.mic_recorder is not None:
            mic, self.mic_recorder = self.mic_recorder, None
            mic.stop()

    def _save_transcripts(self) -> None:
        if self.archive_folder is None:
            return
        paths = write_transcripts(self.archive_folder, self.segments())
        self.callbacks.on_status(f"本场字幕已保存：{paths['会议全记录']}")

    def save_archive_now(self) -> None:
        def action() -> None:
            if self.archive_folder is None:
                self.callbacks.on_status("尚未开始答辩，没有可保存的记录")
                return
            self._save_transcripts()

        self._submit(action, "保存本场记录")

    def loopback_device_name(self) -> str:
        capture = self.loopback_capture
        if capture is not None and capture.device is not None:
            return capture.device.name
        return ""

    def archive_folder_path(self) -> str:
        return str(self.archive_folder) if self.archive_folder is not None else ""

    # ------------------------------------------------------------ committee
    def _start_committee_listening(self, loopback_device: int | None) -> None:
        configured_name = str(self.settings.shared.get("loopback_device_name", "") or "")
        device_choice = choose_committee_loopback(
            list_loopback_devices(), loopback_device, configured_name,
        )
        if device_choice is None:
            raise RuntimeError("没有可用的真实扬声器/耳机回环；CABLE 是程序输出，不能用于评委监听")
        if loopback_device != device_choice.index:
            self.callbacks.on_status(f"评委监听设备编号已变化，已改用：{device_choice.name}")
        values = {
            "asr_model": self.settings.get("committee_asr_model"),
            "vad_threshold": float(self.settings.shared.get("vad_threshold", 0.0)),
            "vad_silence_ms": self._vad_silence_ms(),
            "translation_terms": self.settings.get("glossary", ""),
            # 评委说英文：提示也用英文，术语取术语表的英文一侧
            "asr_context_prefix": "Technical terms that may be spoken in this master's thesis defense: ",
        }
        self.asr_committee = CommitteeASRProcess(
            backup={"backup_asr_enabled": self.settings.get("backup_asr_enabled", False),
                    "backup_asr_model": self.settings.get("backup_asr_model", "")},
            api_key=self.settings.get_api_key(),
            workspace_id=self.settings.workspace_id,
            values=values,
            language="en",
            on_preview=lambda text, _emotion: self.callbacks.on_committee_preview(text),
            on_status=lambda message: self.callbacks.on_status(f"评委识别：{message}"),
            on_error=lambda message: self.callbacks.on_error(f"评委识别：{message}"),
            on_usage=USAGE.add_asr,
            on_segment=self._on_committee_segment,
            combine_previews=False,
            proxy=self.settings.ws_proxy,
        )
        try:
            self.asr_committee.start()
            self.asr_committee.wait_ready()
            self.loopback_capture = SystemAudioCapture()
            device = self.loopback_capture.start(device_choice.index, self._on_loopback_audio, target_rate=16000)
        except Exception:
            if self.loopback_capture is not None:
                self.loopback_capture.stop()
                self.loopback_capture = None
            self.asr_committee.close()
            self.asr_committee = None
            raise
        self.settings.shared.update({
            "loopback_device": device.index,
            "loopback_device_name": device.name,
        })
        self.callbacks.on_committee_device(device.name)
        self.callbacks.on_status(f"对方声音监听中：{device.name}（微信/Teams 的声音需从此设备播出）")

    def _on_loopback_audio(self, pcm: bytes) -> None:
        if self._closed or self._stopping:
            return
        asr = self.asr_committee
        if asr is None:
            return
        try:
            asr.send_audio(pcm)
        except Exception:
            pass

    def _on_committee_segment(self, text: str, _emotion: str = "") -> None:
        text = text.strip()
        if not text or self._stopping or self._closed:
            return
        seg_id = self._register_segment("committee", text)
        self.callbacks.on_committee_recognized(seg_id, text)
        try:
            self._committee_queue.put_nowait((seg_id, text))
        except queue.Full:
            self.callbacks.on_error("评委字幕队列已满，请检查翻译连接")

    # ------------------------------------------------------------- mic / tts
    def _on_mic_audio(self, pcm: bytes) -> None:
        if self._listen_requested is False or self._closed or self._stopping:
            return
        asr = self.asr_mic
        if asr is None:
            return
        try:
            if time.monotonic() < self._suppress_until:
                pcm = bytes(len(pcm))  # keep VAD timing without re-translating our own speaker output
            asr.send_audio(pcm)
            USAGE.add_asr(self._billed_asr_model(asr, "asr_model"), len(pcm) / 32000)
        except Exception:
            pass

    def _on_asr_error(self, message: str) -> None:
        self.callbacks.on_error(f"实时识别：{message}")

    def _billed_asr_model(self, asr, key: str) -> str:
        """识别用量记在真正在用的模型上（切到备用模型后记备用的）。"""
        current = getattr(asr, "current_model", "")
        return current if isinstance(current, str) and current else str(self.settings.get(key, "") or "")

    def _create_asr(self, **kwargs):
        if self.settings.get("backup_asr_enabled", False):
            return create_defense_asr(self.settings, **kwargs)
        return create_realtime_asr(**kwargs)

    def _on_my_segment(self, text: str, _emotion: str = "") -> None:
        text = text.strip()
        if not text or self._stopping or self._closed or not getattr(self, "_accept_mic_tail", True):
            return
        seg_id = self._register_segment("me", text)
        self.callbacks.on_my_segment(seg_id, text)
        try:
            self._my_queue.put_nowait(_Job(seg_id=seg_id, source=text, t0=time.monotonic(), live_input=True))
        except queue.Full:
            self.callbacks.on_error("翻译队列已满，请暂停说话并检查网络")

    def _on_tts_audio(self, pcm: bytes) -> None:
        if self._stopping or self._closed:
            return
        if self._speech_paused.is_set():
            generation = self._speech_generation
            if not self._wait_if_paused() or generation != self._speech_generation:
                return  # 暂停期间按了 Esc：这块不再播
        self._suppress_until = time.monotonic() + 0.6
        if self._live_utterance is not None and pcm:
            self._live_utterance["buf"].extend(pcm)
        if pcm:
            room_tone = self._current_room_tone()
            output_pcm = room_tone.mix(pcm)
            self._last_voice_write = time.monotonic()
            with self._player_lock:
                if self._direct_requested or self._speech_paused.is_set():
                    return
                player = self.player
                if player is None:
                    return
                try:
                    self._write_live_audio(player, output_pcm)
                except Exception:
                    # 死流绝不再写第二次（PortAudio 会在原生层崩溃）：关旧流、重建、重放本块。
                    self._recover_player(output_pcm, include_monitor=self._live_audio_monitor())

    def _recover_player(self, pcm: bytes, *, include_monitor: bool = True) -> None:
        try:
            if self.player is not None:
                self.player.close()
        except Exception:
            pass
        self.player = None
        try:
            self._ensure_player()
            if self.player is not None and pcm:
                if include_monitor:
                    self.player.write(pcm)
                else:
                    self.player.write(pcm, include_monitor=False)
            self.callbacks.on_status("播放设备异常，已自动重建输出流并恢复播放")
        except Exception as exc:
            if self.player is not None:
                self.player.close()
                self.player = None
            self.callbacks.on_error(f"播放设备异常且重建失败，本段音频已丢弃：{exc}")
            raise ApiError("播放设备不可用，请检查输出设备后重播") from exc

    def _write_live_audio(self, player, pcm: bytes) -> None:
        if not self._live_audio_monitor():
            player.write(pcm, include_monitor=False)
        else:
            player.write(pcm)

    def _live_audio_monitor(self) -> bool:
        holder = self._live_utterance
        return not (holder is not None and holder.get("live_input", False)
                    and not self.settings.get("live_monitor_enabled", True))

    def _on_first_audio(self, text: str) -> None:
        if self._stopping or self._closed:
            return
        self._ambience_cancel.clear()
        job = self._speech_jobs.get(text)
        text = self._display_for(text)
        if job is not None and job.first_audio_at is None:
            job.first_audio_at = time.monotonic()
            latency = int((job.first_audio_at - job.t0) * 1000)
            with self._segment_lock:
                if job.seg_id in self._segments:
                    self._segments[job.seg_id].latency_ms = latency
            self.callbacks.on_latency(job.seg_id, latency)
        self._live_utterance = {
            "text": text,
            "zh": job.source if job is not None else "",
            "buf": bytearray(),
            "live_input": bool(job is not None and job.live_input),
        }
        self.callbacks.on_tts_sentence_started(text, job.source if job is not None else "")

    def _on_utterance_done(self, text: str, total_bytes: int) -> None:
        if self._stopping or self._closed:
            return
        job = self._speech_jobs.pop(text, None)
        more_speech_pending = bool(self._speech_jobs)
        took_backup = getattr(self.tts, "took_backup", None)
        # 第二通道换了模型 / 音色合成的这句：本场重播照用，但不写进主模型的磁盘缓存
        other_voice = took_backup(text) is True if callable(took_backup) else False
        text = self._spoken_display.pop(text, text)
        holder = self._live_utterance
        if holder is not None and holder.get("text") == text and holder["buf"]:
            self._store_pcm(text, bytes(holder["buf"]), str(holder.get("zh", "")), disk=not other_voice)
        self._live_utterance = None
        if total_bytes > 0:
            # 后一句无论来自同一长回答还是下一次连续发言，都只保留轻微换气，
            # 避免每个 ASR 句段末尾再额外等待完整房间底噪淡出。
            self._write_ambience_tail(140 if more_speech_pending else None)
        sample_rate = SPEECH_SAMPLE_RATE
        seconds = total_bytes / max(sample_rate * 2, 1)
        self.callbacks.on_tts_sentence_duration(text, seconds)

    def _on_utterance_failed(self, text: str) -> None:
        self._speech_jobs.pop(text, None)
        text = self._spoken_display.pop(text, text)
        holder = self._live_utterance
        if holder is not None and holder.get("text") == text:
            self._live_utterance = None

    def _current_room_tone(self) -> RoomTone:
        mode = background_mode(self.settings.get("ambience_mode", "off"))
        try:
            volume = int(self.settings.get("ambience_volume", 50))
        except (TypeError, ValueError):
            volume = 50
        if (self._room_tone.mode != mode or getattr(self._room_tone, "volume", 50) != volume
                or (mode != "off" and not isinstance(self._room_tone, RecordedRoomTone))):
            self._room_tone = make_room_tone(mode, sample_rate=SPEECH_SAMPLE_RATE, volume=volume)
        return self._room_tone

    def _write_ambience_tail(self, duration_ms: int | None = None) -> None:
        if self._ambience_bed_on():
            return  # 持续底噪会自然接上，不需要淡出尾巴
        room_tone = self._current_room_tone()
        tail = room_tone.tail(duration_ms)
        if not tail:
            return
        block = SPEECH_SAMPLE_RATE * 2 // 25
        for offset in range(0, len(tail), block):
            with self._player_lock:   # 每块单独拿锁，别让尾音挡住别的操作
                if self._ambience_cancel.is_set() or self._closed or self.player is None:
                    return
                try:
                    self.player.write(tail[offset:offset + block])
                except Exception:
                    # 环境音收尾与正文使用相同的死流恢复路径，避免设备刚释放时
                    # 继续写入已失效的 PortAudio 流而触发原生层闪退。
                    self._recover_player(b"")
                    return
                self._suppress_until = time.monotonic() + 0.6

    # ------------------------------------------------ 持续背景环境音
    AMBIENCE_BLOCK_SECONDS = 0.04
    AMBIENCE_VOICE_GAP = 0.25   # 语音停下多久后底噪接手

    def _ambience_mode(self) -> str:
        return background_mode(self.settings.get("ambience_mode", "off"))

    def refresh_ambience(self) -> None:
        def action():
            if self._session_active and self._ambience_mode() != "off":
                self._start_ambience_bed()
        self._submit(action, "切换背景声")

    def _ambience_bed_on(self) -> bool:
        thread = self._ambience_thread
        return bool(thread is not None and thread.is_alive() and self._ambience_mode() != "off")

    def _start_ambience_bed(self) -> None:
        if self._ambience_mode() == "off":
            return
        thread = self._ambience_thread
        if thread is not None and thread.is_alive():
            return
        stop = threading.Event()
        self._ambience_stop = stop
        thread = threading.Thread(target=self._ambience_bed_loop, args=(stop,),
                                  name="defense-ambience-bed", daemon=True)
        self._ambience_thread = thread
        thread.start()

    def _stop_ambience_bed(self) -> None:
        self._ambience_stop.set()
        thread, self._ambience_thread = self._ambience_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)

    def _ambience_bed_loop(self, stop: threading.Event) -> None:
        """没人说话时往会议输出写极轻的房间底噪。

        * 只在拿得到播放锁时写一小块（40 ms），写完马上放锁并让一下，语音随时能插进来；
        * 合成语音刚写过（250 ms 内）时，由语音混音承接背景；
        * F8 原声使用独立的输出流，背景层继续播放，不因真麦克风自带环境声而停掉；
        * 设备出错按和语音相同的死流恢复路径处理，恢复不了就歇 2 秒再试，绝不在原地狂刷。
        """
        samples = int(SPEECH_SAMPLE_RATE * self.AMBIENCE_BLOCK_SECONDS)
        while not stop.is_set() and not self._closed:
            if self._ambience_mode() == "off" or not self._session_active or self._stopping:
                stop.wait(0.3)
                continue
            if time.monotonic() - self._last_voice_write < self.AMBIENCE_VOICE_GAP:
                stop.wait(0.03)
                continue
            if not self._player_lock.acquire(blocking=False):
                stop.wait(0.01)
                continue
            failed = False
            try:
                if stop.is_set() or self._closed:
                    return
                if self.player is None:
                    self._ensure_player()
                bed = self._current_room_tone().bed(samples)
                try:
                    self.player.write(bed)
                except Exception:
                    try:
                        self._recover_player(b"")
                    except Exception:
                        failed = True
            except Exception:
                failed = True
            finally:
                self._player_lock.release()
            # 让出锁：Python 的锁不保证先来先得，不歇一下语音线程可能一直抢不到
            stop.wait(2.0 if failed else 0.004)

    def _vad_silence_ms(self) -> int:
        """说完多久算一句结束。自然模式以前强制至少 700 ms，现在尊重设置（最低 400 ms）。"""
        try:
            value = int(self.settings.get("vad_silence_ms", 500) or 500)
        except (TypeError, ValueError):
            value = 500
        return max(400, value)

    def _trim_memory_cache(self) -> None:
        """按字节把内存缓存压回上限内，最旧的先走（dict 保序即插入序）。

        丢掉的只是加速副本，磁盘上那份还在，`lookup_pcm` 会接住。
        """
        while self._cache_bytes > self._cache_limit_bytes and self._pcm_cache:
            # dict 保持插入顺序，第一个就是最旧的
            oldest = next(iter(self._pcm_cache))
            self._cache_bytes -= len(self._pcm_cache.pop(oldest).pcm)

    def _store_pcm(self, text: str, pcm: bytes, zh: str = "", *, pinned: bool = False, disk: bool = True) -> None:
        # 同文本覆盖旧缓存（模型/参数可能已变）；超上限时按最旧淘汰。
        previous = self._pcm_cache.pop(text, None)
        if previous is not None:
            self._cache_bytes -= len(previous.pcm)
        self._pcm_cache[text] = CachedUtterance(pcm=pcm, zh=zh)
        self._cache_bytes += len(pcm)
        # 同步落盘。写失败不影响本场使用，内存里那份仍然有效。
        if disk:
            self.disk_cache.put(cache_key(text, self._tts_settings()), pcm, pinned=pinned)
        self._trim_memory_cache()
        # 整场会议合成语音：直接写进落盘暂存区，不再在内存里累积。
        start, end = self._meeting_spool.append(pcm)
        if end > start:
            self._utterance_offsets.append({"text": text, "start": start, "end": end, "zh": zh})

    def meeting_audio_seconds(self) -> float:
        return self._meeting_spool.seconds(SPEECH_SAMPLE_RATE)

    def cache_report(self) -> dict:
        """给界面看的用量快照：内存、磁盘、整场录音各占多少。"""
        stats = dict(self.disk_cache.stats())
        stats.update({
            "memory_bytes": self._cache_bytes,
            "memory_limit_bytes": self._cache_limit_bytes,
            "meeting_bytes": self._meeting_spool.total_bytes(),
            "meeting_enabled": self._meeting_spool.enabled,
            "meeting_error": self._meeting_spool.error,
        })
        return stats

    def discard_meeting_audio(self) -> None:
        def action() -> None:
            self._meeting_spool.reset()
            self._utterance_offsets = []
            self.callbacks.on_status("已丢弃本场缓存语音")

        self._submit(action, "丢弃缓存语音")

    def export_meeting_audio(self) -> None:
        def action() -> None:
            if not self._meeting_spool.total_bytes():
                if not self._meeting_spool.enabled and self._meeting_spool.error:
                    self.callbacks.on_error(
                        f"整场录音未能写入磁盘，已跳过：{self._meeting_spool.error}"
                    )
                else:
                    self.callbacks.on_status("本场没有可导出的合成语音")
                return
            folder = self.archive_folder or new_meeting_folder(self._base_directory())
            path = folder / "录音" / "英文语音_全场.wav"
            # 分块流式写，不把整场音频读进内存再复制一份。
            self._meeting_spool.write_wav(path, SPEECH_SAMPLE_RATE)
            seconds = self.meeting_audio_seconds()
            self.callbacks.on_status(f"整场英文语音已保存：{path}（{seconds / 60:.1f} 分钟）")

        self._submit(action, "导出整场英文语音")

    # --------------------------------------------------------------- workers
    def _register_segment(self, role: str, source: str) -> int:
        with self._segment_lock:
            seg_id = self._next_segment_id
            self._next_segment_id += 1
            self._segments[seg_id] = SegmentRecord(seg_id=seg_id, role=role, source=source)
        return seg_id

    def _store_segment(self, seg_id: int, translation: str, *, latency_ms: int) -> None:
        with self._segment_lock:
            record = self._segments.get(seg_id)
            if record is not None:
                record.translation = translation
                record.latency_ms = latency_ms

    def _my_worker(self) -> None:
        while not self._closed:
            with self._job_lock:
                try:
                    job = self._my_queue.get_nowait()
                except queue.Empty:
                    job = None
                if job is not None:
                    self._active_job = job
            if job is None:
                time.sleep(0.02)
                continue
            if self.translator is None:
                self.callbacks.on_error("翻译引擎尚未就绪，请先开始答辩")
                self._active_job = None
                continue
            try:
                self._translate_my_job(job)
            except Exception as exc:
                self.callbacks.on_error(f"第 {job.seg_id} 句处理失败：{exc}")
            finally:
                self._active_job = None

    EARLY_START_MIN_WORDS = 5

    def _translate_my_job(self, job: _Job) -> None:
        assert self.translator is not None or job.preset_english is not None
        splitter = SentenceSplitter()
        natural = self.settings.get("speech_mode", "natural") == "natural"
        # 早开口：自然模式本来要等整段译文出完、整段合成完才出声。打开后，译文流里
        # 第一句（至少 5 个词）一完整就先送去合成播放，后面的在它播放期间合成。
        early = (natural and job.preset_english is None
                 and bool(self.settings.get("speech_early_start", True)))
        early_buffer: list[str] = []
        early_spoken: list[str] = []

        def speak_prepared_batch(text: str, *, with_filler: bool) -> None:
            prepared = self._prepare_utterances(text, with_filler=with_filler)
            with self._job_lock:
                if job.cancel.is_set() or self._closed:
                    raise ApiError("已跳过")
                if self.tts is None:
                    raise ApiError("发声队列未就绪，请稍后重播")
                spoken_texts = self._register_utterances(prepared, job)
                if not self.tts.speak_batch(spoken_texts):
                    self._unregister_utterances(spoken_texts)
                    raise ApiError("发声队列已满或已关闭，请稍后重播")

        def speak(sentence):
            with self._job_lock:
                if job.cancel.is_set() or self._closed:
                    raise ApiError("已跳过")
                if self.tts is not None:
                    self._speech_jobs[sentence] = job
                    if not self.tts.speak(sentence):
                        self._speech_jobs.pop(sentence, None)
                        raise ApiError("发声队列已满或已关闭，请稍后重播")

        def dispatch(piece: str, so_far: str) -> None:
            if job.cancel.is_set():
                raise ApiError("已跳过")
            self.callbacks.on_my_delta(job.seg_id, so_far)
            if not natural:
                for sentence in splitter.feed(piece):
                    speak(sentence)
            elif early and not early_spoken:
                for sentence in splitter.feed(piece):
                    early_buffer.append(sentence)
                    head = " ".join(early_buffer).strip()
                    if len(head.split()) >= self.EARLY_START_MIN_WORDS:
                        speak_prepared_batch(head, with_filler=True)
                        early_spoken.append(head)
                        break

        try:
            if job.preset_english is not None:
                english = job.preset_english
                dispatch(english, english)
            else:
                english = self.translator.translate(
                    self.memory,
                    "zh2en",
                    job.source,
                    on_delta=dispatch,
                    cancel_event=job.cancel,
                )
            if job.cancel.is_set():
                raise ApiError("已跳过")
            if natural and early_spoken:
                rest = remainder_after(english, early_spoken[0])
                if rest:
                    speak_prepared_batch(rest, with_filler=False)
            elif natural:
                speak_prepared_batch(english, with_filler=True)
            else:
                # 逐句流式模式不加工：它的卖点就是尽快开口，任何拼接都会推迟首字。
                for sentence in splitter.flush():
                    speak(sentence)
        except ApiError as exc:
            if "已跳过" in str(exc) or "已取消" in str(exc):
                self.callbacks.on_status(f"第 {job.seg_id} 句已停止")
                return
            self.callbacks.on_error(f"第 {job.seg_id} 句翻译失败：{exc}")
            return
        latency_ms = 0
        if job.first_audio_at is not None:
            latency_ms = int((job.first_audio_at - job.t0) * 1000)
        self._store_segment(job.seg_id, english, latency_ms=latency_ms)
        self.memory.add_turn(Turn(role="me", source=job.source, translation=english, latency_ms=latency_ms))
        self.callbacks.on_my_done(job.seg_id, english, latency_ms)

    def _committee_worker(self) -> None:
        while not self._closed:
            try:
                item = self._committee_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            seg_id, source = item
            if self.translator is None:
                self.callbacks.on_committee_done(seg_id, source, "（中文翻译未就绪，已保留评委原话）")
                continue
            try:
                token = self._committee_cancel
                if token.is_set():
                    continue
                chinese = self.translator.translate(self.memory, "en2zh", source, cancel_event=token)
                if token.is_set():
                    continue
            except Exception as exc:
                if token.is_set() or self._closed:
                    continue
                self.callbacks.on_error(f"评委字幕翻译失败：{exc}")
                self.callbacks.on_committee_done(seg_id, source, "（中文翻译暂时失败，已保留英文问题）")
                continue
            self._store_segment(seg_id, chinese, latency_ms=0)
            self.memory.add_turn(Turn(role="committee", source=source, translation=chinese))
            self.callbacks.on_committee_done(seg_id, source, chinese)


def remainder_after(full: str, spoken: str) -> str:
    """早开口已经读了开头 spoken，返回整段译文 full 里还没读的部分。

    先按原文前缀比；对不上（空白、标点被分句器整理过）就按词数对齐。
    宁可少读半句也不要把已经读过的再读一遍——重复一句评委立刻就听得出来。
    """
    full = (full or "").strip()
    spoken = (spoken or "").strip()
    if not spoken:
        return full
    if full.startswith(spoken):
        return full[len(spoken):].strip()
    words = full.split()
    count = len(spoken.split())
    if count >= len(words):
        return ""
    if [w.strip(".,;:!?\"'") for w in words[:count]] == [w.strip(".,;:!?\"'") for w in spoken.split()]:
        return " ".join(words[count:]).strip()
    # 最后一招：在全文里找已读部分的最后几个词
    tail = " ".join(spoken.split()[-4:])
    index = full.find(tail)
    if index >= 0:
        return full[index + len(tail):].strip()
    return ""
