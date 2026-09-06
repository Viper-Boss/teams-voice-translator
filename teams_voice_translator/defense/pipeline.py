from __future__ import annotations

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
    )
from .records import default_base_dir, new_meeting_folder, write_transcripts
from .memory import MeetingMemory, Turn
from .settings import DefenseSettings
from .translator import ContextTranslator, SentenceSplitter
from .tts_session import TtsHttpSession, TtsSession, create_tts_session
from .speech_policy import speech_settings, speech_chunks, SPEECH_SAMPLE_RATE
from .ambience import RoomTone

STATE_IDLE = "idle"  # 未开始
STATE_STARTING = "starting"  # 正在打开答辩通道
STATE_STANDBY = "standby"  # 答辩通道已就绪
STATE_LISTENING = "listening"  # F9 翻译中
STATE_DIRECT = "direct"  # F5 原声直通


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
    on_tts_sentence_started: Callable[[str, str], None] = lambda text, zh: None
    on_tts_sentence_duration: Callable[[str, float], None] = lambda text, seconds: None
    on_committee_device: Callable[[str], None] = lambda name: None
    on_latency: Callable[[int, int], None] = lambda seg_id, latency: None


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
        )

        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._cmd_queue: queue.Queue[tuple[Callable[[], None], str]] = queue.Queue()
        self._cmd_thread: threading.Thread | None = None

        self.client: BailianClient | None = None
        self.translator: ContextTranslator | None = None
        self.tts: TtsSession | None = None
        self.player: MultiOutputPlayer | None = None

        self._session_active = False
        self._listening = False
        self._listen_requested = None
        self._listen_request_serial = 0
        self._direct_requested = None
        self._direct_request_serial = 0
        self._direct_active = False
        self._resume_listening_after_direct = False

        self.archive_folder = None
        self.recorder: DualTrackRecorder | None = None
        self.mic_recorder: WavRecorder | None = None
        self._player_lock = threading.RLock()
        self._pcm_cache: dict[str, CachedUtterance] = {}
        self._cache_limit = 300
        self._live_utterance: dict[str, Any] | None = None
        self._meeting_pcm = bytearray()
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
        self._speech_generation = 0
        self._suppress_until = 0.0
        self._room_tone = RoomTone(str(settings.get("ambience_mode", "off") or "off"))
        self._ambience_cancel = threading.Event()
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
        with self._segment_lock:
            self._segments.clear()

    def reload_context(self) -> None:
        """Re-read brief/glossary/window from settings after the user edits them."""
        self.memory.set_brief(str(self.settings.get("defense_brief", "") or ""))
        self.memory.set_glossary(self.settings.glossary_terms())
        self.memory.max_turns = max(2, int(self.settings.get("max_context_turns", 14)))

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
            self._do_stop_defense()

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

    def start_defense(self) -> None:
        self._submit(self._do_start_defense, "打开答辩通道")

    def stop_defense(self) -> None:
        self._submit(self._do_stop_defense, "关闭答辩通道")

    def toggle_listening(self) -> None:
        self._submit(self._do_toggle_listening, "切换翻译")

    def set_listening(self, enabled: bool) -> None:
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

    def interrupt_speech(self) -> None:
        """Esc: silence the current and queued speech immediately."""
        self._ambience_cancel.set()
        with self._job_lock:
            self._speech_generation += 1
            self._playback_cancel.set()
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
            if self.tts is not None:
                self.tts.interrupt()
        self.callbacks.on_status("已停止当前语音")

    def _queue_speech(self, text):
        chunks = speech_chunks(text)
        if self.tts is None or not self.tts.speak_batch(chunks):
            raise ApiError("发声队列未就绪，请稍后重播")

    def send_text(self, text: str, *, translate: bool = True) -> None:
        text = text.strip()
        if not text:
            return
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

    def replay_english(self, text: str) -> None:
        text = text.strip()
        generation = self._speech_generation

        def action() -> None:
            if generation != self._speech_generation:
                return
            cached = self._pcm_cache.get(text)
            if cached is not None:
                self._play_cached(cached, text)
                return
            self._ensure_voice()
            self._queue_speech(text)
            self.callbacks.on_status("该句不在本地缓存，已请求服务器重新合成（完成后自动入缓存）")

        self._submit(action, "重播")

    def _play_cached(self, cached: CachedUtterance, text: str) -> None:
        self.interrupt_speech()
        self._ensure_player()
        token = threading.Event()
        self._playback_cancel = token
        sample_rate = SPEECH_SAMPLE_RATE
        duration = len(cached.pcm) / max(sample_rate * 2, 1)
        self._suppress_until = time.monotonic() + duration + 0.6
        self.callbacks.on_tts_sentence_started(text, cached.zh)
        self.callbacks.on_tts_sentence_duration(text, duration)
        room_tone = RoomTone(str(self.settings.get("ambience_mode", "off") or "off"))
        def write_loop() -> None:
            try:
                with self._player_lock:
                    block = sample_rate * 2 // 25
                    for offset in range(0, len(cached.pcm), block):
                        if token.is_set() or self._closed:
                            return
                        if self.player is None:
                            return
                        try:
                            self.player.write(room_tone.mix(cached.pcm[offset:offset + block]))
                        except Exception:
                            self._recover_player(b"")
                    tail = room_tone.tail()
                    for offset in range(0, len(tail), block):
                        if token.is_set() or self._closed or self.player is None:
                            return
                        self.player.write(tail[offset:offset + block])
            except Exception as exc:
                self.callbacks.on_error(f"本地重播播放失败：{exc}")
            else:
                self.callbacks.on_status("已从本地缓存重播（未请求服务器）")

        threading.Thread(target=write_loop, name="defense-cache-playback", daemon=True).start()

    def speak_as_me(self, text: str, *, label: str = "AI 助答") -> None:
        """Speak prepared English as the candidate and record it into context."""
        text = text.strip()
        if not text:
            return
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
        )

    def _tts_settings(self) -> dict[str, Any]:
        return speech_settings(self.settings)

    def apply_settings(self) -> None:
        def action():
            was_active = self._session_active
            was_listening = self._listening
            self._do_stop_defense()
            self.client = None
            self.translator = None
            self._pcm_cache.clear()
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
        if not str(self.settings.get("tts_voice_id", "") or "").strip():
            raise ApiError("尚未填写克隆音色 voice_id，请在设置中填写控制台复刻的音色")
        for entry in self.settings.get("voice_library", []):
            if isinstance(entry, dict) and entry.get("voice_id") == self.settings.get("tts_voice_id"):
                bound = entry.get("target_model")
                if bound and bound != self.settings.get("tts_model"):
                    raise ApiError(f"音色绑定模型为 {bound}，请在设置中选择对应模型")
        if self.tts is None:
            self._build_clients()
        self._ensure_player()
        if self.tts is None:
            self.tts = create_tts_session(
                self.client,
                self._tts_settings(),
                on_audio=self._on_tts_audio,
                on_status=self.callbacks.on_status,
                on_error=self.callbacks.on_error,
                on_first_audio=self._on_first_audio,
                on_utterance_done=self._on_utterance_done,
                on_audio_ready=lambda text, size: self.callbacks.on_tts_sentence_duration(text, size / (SPEECH_SAMPLE_RATE * 2)),
            )
            self.tts.start()
            if isinstance(self.tts, TtsHttpSession):
                self.callbacks.on_status(
                    "自然发声：完整合成后播放，首句等待时间较长"
                )

    def _output_devices(self) -> list[int | None]:
        cable = self.settings.shared.get("teams_output_device")
        devices: list[int | None] = [cable]
        if bool(self.settings.get("monitor_enabled", True)):
            devices.append(self.settings.get("monitor_output_device"))
        return devices

    def _do_start_defense(self, *, reset_audio: bool = True) -> None:
        if self._session_active:
            return
        self._stopping = False
        if reset_audio:
            self._meeting_pcm = bytearray()
            self._utterance_offsets = []
        self._committee_cancel = threading.Event()
        while True:
            try:
                self._committee_queue.get_nowait()
            except queue.Empty:
                break
        self._set_state(STATE_STARTING)
        self._build_clients()
        self.reload_context()
        self._ensure_voice()
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
        self._set_state(STATE_STANDBY)
        if bool(self.settings.get("auto_record", True)):
            try:
                self._start_recording()
            except Exception as exc:
                self.callbacks.on_error(f"录音启动失败（不影响翻译）：{exc}")
        self.callbacks.on_status("答辩通道已就绪：请选择「原声」或「翻译」；未勾选持续监听时需按住说话")

    def _do_stop_defense(self) -> None:
        self._stopping = True
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
        values = {
            "asr_model": self.settings.get("asr_model"),
            "vad_threshold": float(self.settings.shared.get("vad_threshold", 0.0)),
            "vad_silence_ms": max(700, int(self.settings.get("vad_silence_ms", 500))) if self.settings.get("speech_mode", "natural") == "natural" else int(self.settings.get("vad_silence_ms", 500)),
            "translation_terms": self.settings.get("glossary", ""),
        }
        self.asr_mic = create_realtime_asr(
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
            self.settings.shared.get("teams_output_device"),
            int(self.settings.shared.get("direct_sample_rate", 48000)),
        )
        self._suppress_until = time.monotonic() + 1.0
        self._direct_active = True
        self._set_state(STATE_DIRECT)
        self.callbacks.on_status("原声直通中（松开 F5 结束）")

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
            "vad_silence_ms": max(700, int(self.settings.get("vad_silence_ms", 500))) if self.settings.get("speech_mode", "natural") == "natural" else int(self.settings.get("vad_silence_ms", 500)),
            "translation_terms": self.settings.get("glossary", ""),
        }
        self.asr_committee = create_realtime_asr(
            api_key=self.settings.get_api_key(),
            workspace_id=self.settings.workspace_id,
            values=values,
            language="en",
            on_preview=lambda text, _emotion: self.callbacks.on_committee_preview(text),
            on_status=self.callbacks.on_status,
            on_error=self._on_asr_error,
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
        if self._direct_requested is True or self._direct_active or time.monotonic() < self._suppress_until:
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
        except Exception:
            pass

    def _on_asr_error(self, message: str) -> None:
        self.callbacks.on_error(f"实时识别：{message}")

    def _on_my_segment(self, text: str, _emotion: str = "") -> None:
        text = text.strip()
        if not text or self._stopping or self._closed:
            return
        seg_id = self._register_segment("me", text)
        self.callbacks.on_my_segment(seg_id, text)
        try:
            self._my_queue.put_nowait(_Job(seg_id=seg_id, source=text, t0=time.monotonic()))
        except queue.Full:
            self.callbacks.on_error("翻译队列已满，请暂停说话并检查网络")

    def _on_tts_audio(self, pcm: bytes) -> None:
        self._suppress_until = time.monotonic() + 0.6
        if self._live_utterance is not None and pcm:
            self._live_utterance["buf"].extend(pcm)
        if pcm:
            room_tone = self._current_room_tone()
            output_pcm = room_tone.mix(pcm)
            with self._player_lock:
                player = self.player
                if player is None:
                    return
                try:
                    player.write(output_pcm)
                except Exception:
                    # 死流绝不再写第二次（PortAudio 会在原生层崩溃）：关旧流、重建、重放本块。
                    self._recover_player(b"")

    def _recover_player(self, pcm: bytes) -> None:
        try:
            if self.player is not None:
                self.player.close()
        except Exception:
            pass
        self.player = None
        try:
            self._ensure_player()
            if self.player is not None and pcm:
                self.player.write(pcm)
            self.callbacks.on_status("播放设备异常，已自动重建输出流并恢复播放")
        except Exception as exc:
            if self.player is not None:
                self.player.close()
                self.player = None
            self.callbacks.on_error(f"播放设备异常且重建失败，本段音频已丢弃：{exc}")
            raise ApiError("播放设备不可用，请检查输出设备后重播") from exc

    def _on_first_audio(self, text: str) -> None:
        self._ambience_cancel.clear()
        job = self._speech_jobs.get(text)
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
        }
        self.callbacks.on_tts_sentence_started(text, job.source if job is not None else "")

    def _on_utterance_done(self, text: str, total_bytes: int) -> None:
        job = self._speech_jobs.pop(text, None)
        more_in_same_answer = job is not None and any(item is job for item in self._speech_jobs.values())
        holder = self._live_utterance
        if holder is not None and holder.get("text") == text and holder["buf"]:
            self._store_pcm(text, bytes(holder["buf"]), str(holder.get("zh", "")))
        self._live_utterance = None
        if total_bytes > 0:
            self._write_ambience_tail(140 if more_in_same_answer else None)
        sample_rate = SPEECH_SAMPLE_RATE
        seconds = total_bytes / max(sample_rate * 2, 1)
        self.callbacks.on_tts_sentence_duration(text, seconds)

    def _current_room_tone(self) -> RoomTone:
        mode = str(self.settings.get("ambience_mode", "off") or "off")
        if self._room_tone.mode != mode:
            self._room_tone = RoomTone(mode, sample_rate=SPEECH_SAMPLE_RATE)
        return self._room_tone

    def _write_ambience_tail(self, duration_ms: int | None = None) -> None:
        room_tone = self._current_room_tone()
        tail = room_tone.tail(duration_ms)
        if not tail:
            return
        block = SPEECH_SAMPLE_RATE * 2 // 25
        with self._player_lock:
            for offset in range(0, len(tail), block):
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

    def _store_pcm(self, text: str, pcm: bytes, zh: str = "") -> None:
        # 同文本覆盖旧缓存（模型/参数可能已变）；超上限时按最旧淘汰。
        if text in self._pcm_cache:
            self._pcm_cache.pop(text)
        self._pcm_cache[text] = CachedUtterance(pcm=pcm, zh=zh)
        while len(self._pcm_cache) > self._cache_limit:
            self._pcm_cache.pop(next(iter(self._pcm_cache)))
        # 整场会议合成语音：按顺序累积，供会后整体导出。
        start = len(self._meeting_pcm)
        self._meeting_pcm.extend(pcm)
        self._utterance_offsets.append({"text": text, "start": start, "end": len(self._meeting_pcm), "zh": zh})

    def meeting_audio_seconds(self) -> float:
        sample_rate = SPEECH_SAMPLE_RATE
        return len(self._meeting_pcm) / max(sample_rate * 2, 1)

    def discard_meeting_audio(self) -> None:
        def action() -> None:
            self._meeting_pcm = bytearray()
            self._utterance_offsets = []
            self.callbacks.on_status("已丢弃本场缓存语音")

        self._submit(action, "丢弃缓存语音")

    def export_meeting_audio(self) -> None:
        def action() -> None:
            import wave

            if not self._meeting_pcm:
                self.callbacks.on_status("本场没有可导出的合成语音")
                return
            folder = self.archive_folder or new_meeting_folder(self._base_directory())
            path = folder / "录音" / "英文语音_全场.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SPEECH_SAMPLE_RATE)
                handle.writeframes(bytes(self._meeting_pcm))
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

    def _translate_my_job(self, job: _Job) -> None:
        assert self.translator is not None
        splitter = SentenceSplitter()
        natural = self.settings.get("speech_mode", "natural") == "natural"

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

        try:
            english = self.translator.translate(
                self.memory,
                "zh2en",
                job.source,
                on_delta=dispatch,
                cancel_event=job.cancel,
            )
            if job.cancel.is_set():
                raise ApiError("已跳过")
            sentences = speech_chunks(english) if natural else splitter.flush()
            if natural:
                with self._job_lock:
                    if job.cancel.is_set() or self._closed:
                        raise ApiError("已跳过")
                    if self.tts is None:
                        raise ApiError("发声队列未就绪，请稍后重播")
                    for sentence in sentences:
                        self._speech_jobs[sentence] = job
                    if not self.tts.speak_batch(sentences):
                        for sentence in sentences:
                            self._speech_jobs.pop(sentence, None)
                        raise ApiError("发声队列已满或已关闭，请稍后重播")
            else:
                for sentence in sentences:
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
                continue
            try:
                token = self._committee_cancel
                if token.is_set():
                    continue
                chinese = self.translator.translate(self.memory, "en2zh", source, cancel_event=token)
                if token.is_set():
                    continue
            except Exception as exc:
                self.callbacks.on_error(f"评委字幕翻译失败：{exc}")
                continue
            self._store_segment(seg_id, chinese, latency_ms=0)
            self.memory.add_turn(Turn(role="committee", source=source, translation=chinese))
            self.callbacks.on_committee_done(seg_id, source, chinese)
