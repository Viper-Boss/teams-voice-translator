from __future__ import annotations

import base64
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any

import websocket

from ..aliyun import ApiError, _websocket_proxy_options
from ..api_payloads import build_qwen3_tts_realtime_session_update


class _Cancelled(Exception):
    pass


class _QueuedSession:
    """One serial worker, per-utterance cancellation and accurate idle tracking."""

    def __init__(self, client, settings: dict[str, Any], *, on_audio: Callable,
                 on_status=None, on_error=None, on_first_audio=None,
                 on_utterance_done=None, on_audio_ready=None, connect_timeout=15.0,
                 utterance_timeout=120.0, max_attempts=2):
        self.client = client
        self.settings = dict(settings)
        self.model = str(settings.get("tts_model", "")).strip()
        self.voice = str(settings.get("voice", "")).strip()
        if not self.voice:
            raise ApiError("需要先填写与当前模型绑定的克隆音色 voice_id")
        self.on_audio = on_audio
        self.on_status = on_status or (lambda message: None)
        self.on_error = on_error or (lambda message: None)
        self.on_first_audio = on_first_audio
        self.on_utterance_done = on_utterance_done
        self.on_audio_ready = on_audio_ready
        self.connect_timeout = float(connect_timeout)
        self.utterance_timeout = float(utterance_timeout)
        self.max_attempts = max(1, min(2, int(max_attempts)))
        self._queue = queue.Queue(maxsize=64)
        self._state_lock = threading.RLock()
        self._cancel = threading.Event()
        self._closed = threading.Event()
        self._drop_audio = threading.Event()
        self._active_cancel = None
        self._thread = None
        self._speaking = False
        self._last_error = ""
        self._emitted_bytes = 0

    @property
    def last_error(self):
        return self._last_error

    def start(self):
        with self._state_lock:
            if self._cancel.is_set() or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="defense-tts", daemon=True)
            self._thread.start()

    def speak(self, text):
        text = text.strip()
        with self._state_lock:
            if not text or self._cancel.is_set():
                return False
            try:
                self._queue.put_nowait((text, threading.Event()))
                return True
            except queue.Full:
                self.on_error("语音合成队列已满，请暂停说话，等待当前回答播放完毕")
                return False

    def speak_batch(self, texts):
        """Queue one ordered batch; HTTP subclasses prefetch it concurrently."""
        cleaned = tuple(str(text).strip() for text in texts if str(text).strip())
        if not cleaned:
            return False
        with self._state_lock:
            if self._cancel.is_set():
                return False
            try:
                self._queue.put_nowait((cleaned, threading.Event()))
                return True
            except queue.Full:
                self.on_error("语音合成队列已满，请暂停说话，等待当前回答播放完毕")
                return False
    def interrupt(self):
        with self._state_lock:
            self._drop_audio.set()
            if self._active_cancel is not None:
                self._active_cancel.set()
            while True:
                try:
                    _, event = self._queue.get_nowait()
                    event.set()
                    self._queue.task_done()
                except queue.Empty:
                    break
            if self._active_cancel is None:
                self._drop_audio.clear()

    def close(self, timeout=6.0):
        self._cancel.set()
        self.interrupt()
        # The worker owns the connection. It observes cancellation on each recv
        # (one-second timeout), avoiding concurrent websocket close/write calls.
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is None or not thread.is_alive():
            self._closed.set()
        # Never discard a still-live thread or clear its cancellation token.

    def wait_until_idle(self, timeout=60.0):
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(min(remaining, 0.1))
        return True

    def _disconnect(self):
        pass

    def _run(self):
        try:
            while not self._cancel.is_set():
                # Claim the item and publish its token atomically with interrupt.
                with self._state_lock:
                    try:
                        text, token = self._queue.get_nowait()
                    except queue.Empty:
                        text = None
                    if text is not None:
                        self._active_cancel = token
                        self._speaking = True
                        self._drop_audio.clear()
                if text is None:
                    self._cancel.wait(0.02)
                    continue
                try:
                    self._process_item(text, token)
                finally:
                    with self._state_lock:
                        self._active_cancel = None
                        self._speaking = False
                        self._drop_audio.clear()
                        self._queue.task_done()
        finally:
            self._disconnect()
            self._closed.set()

    def _check_cancel(self, token):
        if self._cancel.is_set() or token.is_set():
            raise _Cancelled()

    def _process_item(self, item, token):
        if isinstance(item, tuple):
            for text in item:
                self._check_cancel(token)
                self._speak_with_retry(text, token)
            return
        self._speak_with_retry(item, token)

    def _speak_with_retry(self, text, token):
        self._last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            self._emitted_bytes = 0
            try:
                self._check_cancel(token)
                self._synthesize(text, token)
                return
            except _Cancelled:
                self._disconnect()
                return
            except Exception as exc:
                self._disconnect()
                if self._cancel.is_set() or token.is_set():
                    return
                # Once listeners heard any audio, replaying from the beginning
                # would duplicate speech. Only retry before audible delivery.
                if attempt == self.max_attempts or self._emitted_bytes:
                    self._last_error = str(exc)
                    self.on_error(f"这一句合成或播放失败，已跳过：{exc}")
                    return
                self.on_status(f"语音合成异常（{exc}），正在自动重试…")
                token.wait(0.2)

    def _synthesize(self, text, token):
        natural = self.settings.get("speech_mode") == "natural"
        buffer = bytearray()
        pending = bytearray()
        received = 0
        deadline = time.monotonic() + self.utterance_timeout
        first = True

        def deliver(chunk):
            nonlocal first
            # Bound a blocking PortAudio write to 40 ms of mono PCM16.
            block = max(2, int(self.settings.get("tts_sample_rate", 24000)) * 2 // 25)
            for offset in range(0, len(chunk), block):
                self._check_cancel(token)
                if first:
                    first = False
                    if self.on_first_audio:
                        self.on_first_audio(text)
                    if natural and self.on_audio_ready:
                        self.on_audio_ready(text, received)
                part = chunk[offset:offset + block]
                self._emitted_bytes += len(part)
                self.on_audio(part)

        def receive(chunk):
            nonlocal received
            self._check_cancel(token)
            if time.monotonic() > deadline:
                raise ApiError("等待合成音频超时")
            if not chunk:
                return
            received += len(chunk)
            if received > 16 * 1024 * 1024:
                raise ApiError("单段音频过长，请把回答分成较短段落")
            if natural:
                buffer.extend(chunk)
            else:
                # Transport packets may split a 16-bit sample between bytes.
                pending.extend(chunk)
                size = len(pending) // 2 * 2
                if size:
                    deliver(bytes(pending[:size]))
                    del pending[:size]

        self._stream_call(text, receive, token)
        self._check_cancel(token)
        if not received:
            raise ApiError("语音服务未返回有效音频")
        if received % 2:
            raise ApiError("语音服务返回了不完整的 PCM 音频")
        if natural:
            if len(buffer) % 2:
                raise ApiError("语音服务返回了不完整的 PCM 音频")
            deliver(buffer)
        self._check_cancel(token)
        if self.on_utterance_done:
            self.on_utterance_done(text, received)


class TtsSession(_QueuedSession):
    """Qwen VC realtime connection, recovered lazily for each queued sentence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws = None

    @property
    def endpoint(self):
        return f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={self.model}"

    def _send(self, event):
        if self._ws is None:
            raise ApiError("语音合成连接尚未建立")
        self._ws.send(json.dumps(event, ensure_ascii=False))

    def _receive(self):
        try:
            message = self._ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        if not message:
            raise ApiError("语音合成连接已断开")
        try:
            event = json.loads(message)
        except (ValueError, TypeError):
            return None
        if not isinstance(event, dict):
            return None
        if event.get("type") == "error":
            detail = event.get("error") or {}
            raise ApiError(f"{detail.get('code', 'TTS error')}: {detail.get('message', '')}")
        return event

    def _wait_for(self, kind, deadline, token):
        while time.monotonic() < deadline:
            self._check_cancel(token)
            event = self._receive()
            if event and event.get("type") == kind:
                return
        raise ApiError(f"等待 {kind} 超时，请检查凭据和网络")

    def _connect(self, token):
        options = {
            "header": [f"Authorization: Bearer {self.client.api_key}",
                       f"X-DashScope-WorkSpace: {self.client.workspace_id}"],
            "timeout": min(15, max(self.connect_timeout, 1)),
        }
        options.update(_websocket_proxy_options(getattr(self.client, "proxy", "")))
        self._ws = websocket.create_connection(self.endpoint, **options)
        self._ws.settimeout(1.0)
        self._wait_for("session.created", time.monotonic() + self.connect_timeout, token)
        self._send(build_qwen3_tts_realtime_session_update(
            voice=self.voice, language=self.settings.get("tts_language_hint", "en"),
            sample_rate=self.settings.get("tts_sample_rate", 24000),
            volume=self.settings.get("tts_volume", 55),
            rate=self.settings.get("tts_rate", 1.0), pitch=self.settings.get("tts_pitch", 1.0)))
        self._wait_for("session.updated", time.monotonic() + self.connect_timeout, token)
        self.on_status("语音合成通道已就绪")

    def _disconnect(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _stream_call(self, text, on_audio, token):
        if self._ws is None:
            self._connect(token)
        self._check_cancel(token)
        self._send({"event_id": uuid.uuid4().hex, "type": "input_text_buffer.append", "text": text})
        self._send({"event_id": uuid.uuid4().hex, "type": "input_text_buffer.commit"})
        deadline = time.monotonic() + self.utterance_timeout
        while time.monotonic() < deadline:
            self._check_cancel(token)
            event = self._receive()
            if event is None:
                continue
            if event.get("type") == "response.audio.delta" and event.get("delta"):
                on_audio(base64.b64decode(event["delta"], validate=True))
            elif event.get("type") == "response.done":
                response = event.get("response") or {}
                if response.get("status") in ("failed", "cancelled", "incomplete"):
                    raise ApiError(f"合成未完成：{response.get('status_details', response.get('status'))}")
                return
        raise ApiError("等待合成音频超时")


class TtsHttpSession(_QueuedSession):
    """Qwen HTTP session with the same cancellation and buffering discipline."""

    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_qwen3_tts_http(text, self.settings, on_audio, cancel_event)

    def _buffer_one(self, text, token):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            captured = bytearray()
            try:
                self._check_cancel(token)
                self._stream_call(text, captured.extend, token)
                self._check_cancel(token)
                if not captured:
                    raise ApiError("语音服务未返回有效音频")
                if len(captured) % 2:
                    raise ApiError("语音服务返回了不完整的 PCM 音频")
                if len(captured) > 16 * 1024 * 1024:
                    raise ApiError("单段音频过长，请把回答分成较短段落")
                return bytes(captured)
            except _Cancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    self.on_status(f"语音合成异常（{exc}），正在后台重试…")
                    token.wait(.2)
        raise ApiError(str(last_error or "语音合成失败"))

    def _deliver_buffered(self, text, pcm, token):
        self._check_cancel(token)
        if self.on_first_audio:
            self.on_first_audio(text)
        if self.on_audio_ready:
            self.on_audio_ready(text, len(pcm))
        block = max(2, int(self.settings.get("tts_sample_rate", 24000)) * 2 // 25)
        for offset in range(0, len(pcm), block):
            self._check_cancel(token)
            part = pcm[offset:offset + block]
            self._emitted_bytes += len(part)
            self.on_audio(part)
        if self.on_utterance_done:
            self.on_utterance_done(text, len(pcm))

    def _process_item(self, item, token):
        if not isinstance(item, tuple):
            return super()._process_item(item, token)
        # Synthesis runs ahead on at most three workers; the single session
        # worker remains the only audio producer and preserves sentence order.
        workers = min(3, len(item))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="defense-tts-prefetch") as pool:
            futures = [pool.submit(self._buffer_one, text, token) for text in item]
            for text, future in zip(item, futures):
                try:
                    pcm = future.result()
                    self._deliver_buffered(text, pcm, token)
                except _Cancelled:
                    return
                except Exception as exc:
                    self._last_error = str(exc)
                    self.on_error(f"这一句后台合成失败，已跳过：{exc}")
                    if self.on_utterance_done:
                        self.on_utterance_done(text, 0)


class TtsLegacySession(TtsHttpSession):
    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_legacy_tts(text, self.settings, on_audio, cancel_event)


def create_tts_session(client, settings, **kwargs):
    from ..api_payloads import is_cosyvoice_model, is_qwen3_tts_vc_model, is_qwen3_tts_vc_realtime_model
    model = str(settings.get("tts_model", ""))
    if is_qwen3_tts_vc_realtime_model(model):
        return TtsSession(client, settings, **kwargs)
    if is_qwen3_tts_vc_model(model):
        return TtsHttpSession(client, settings, **kwargs)
    if is_cosyvoice_model(model) or model.startswith("qwen-audio-3.0-tts"):
        return TtsLegacySession(client, settings, **kwargs)
    raise ApiError(f"不支持的语音合成模型：{model}")
