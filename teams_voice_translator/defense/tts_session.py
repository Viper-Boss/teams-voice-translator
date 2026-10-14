from __future__ import annotations

import base64
import json
import queue
import re
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


class _BatchCompletion:
    """Keep Queue.unfinished_tasks accurate while one batch is prefetched per sentence."""

    def __init__(self, count, on_done):
        self.remaining = count
        self.on_done = on_done
        self.lock = threading.Lock()

    def complete_one(self):
        done = False
        with self.lock:
            if self.remaining <= 0:
                return
            self.remaining -= 1
            done = self.remaining == 0
        if done:
            self.on_done()


class _QueuedSession:
    """One serial worker, per-utterance cancellation and accurate idle tracking."""

    def __init__(self, client, settings: dict[str, Any], *, on_audio: Callable,
                 on_status=None, on_error=None, on_first_audio=None,
                 on_utterance_done=None, on_utterance_failed=None, on_audio_ready=None, connect_timeout=15.0,
                 utterance_timeout=120.0, max_attempts=2):
        self.client = client
        self.settings = dict(settings)
        self.model = str(settings.get("tts_model", "")).strip()
        self.voice = str(settings.get("voice", "")).strip()
        if not self.voice and self.model != "qwen-audio-3.1-tts-next":
            raise ApiError("需要先填写与当前模型绑定的克隆音色 voice_id")
        self.on_audio = on_audio
        self.on_status = on_status or (lambda message: None)
        self.on_error = on_error or (lambda message: None)
        self.on_first_audio = on_first_audio
        self.on_utterance_done = on_utterance_done
        self.on_utterance_failed = on_utterance_failed
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

    def _count(self, texts) -> None:
        if self.settings.get("backup_tts_enabled", False):
            return  # Each underlying session accounts for its actual request.
        try:
            from .usage import USAGE
            chars = sum(len(re.sub(r"<[^>]+>", "", str(t))) for t in texts)
            USAGE.add_tts(self.model, chars)
        except Exception:  # noqa: BLE001 —— 记账失败不影响发声
            pass

    def speak(self, text):
        text = text.strip()
        with self._state_lock:
            if not text or self._cancel.is_set():
                return False
            try:
                self._queue.put_nowait((text, threading.Event()))
                self._count([text])
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
                self._count(cleaned)
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
        for attempt in range(1, max(self.max_attempts, 4) + 1):
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
                retry_limit = 4 if "语音服务未返回有效音频" in str(exc) else self.max_attempts
                if attempt >= retry_limit or self._emitted_bytes:
                    self._last_error = str(exc)
                    self.on_error(f"这一句合成或播放失败，已跳过：{exc}")
                    if self.on_utterance_failed:
                        self.on_utterance_failed(text)
                    return
                self.on_status(f"语音服务暂未返回音频，正在自动重试（{attempt}/{retry_limit - 1}）…" if retry_limit == 4 else f"语音合成异常（{exc}），正在自动重试…")
                token.wait(min(0.2 * 2 ** (attempt - 1), 1.0))

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A feeder keeps submitting later utterances while the owner thread is
        # playing the current one. Six prepared slots bound PCM memory; only
        # three cloud syntheses run at once.
        self._prepared = queue.Queue()
        self._prefetch_slots = threading.BoundedSemaphore(6)
        self._prefetch_tokens = set()

    def interrupt(self):
        with self._state_lock:
            for token in tuple(self._prefetch_tokens):
                token.set()
        super().interrupt()

    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_qwen3_tts_http(text, self.settings, on_audio, cancel_event)

    def _buffer_one(self, text, token):
        last_error = None
        for attempt in range(1, max(self.max_attempts, 4) + 1):
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
                retry_limit = 4 if "语音服务未返回有效音频" in str(exc) else self.max_attempts
                if attempt < retry_limit:
                    self.on_status(f"语音服务暂未返回音频，正在后台重试（{attempt}/{retry_limit - 1}）…" if retry_limit == 4 else f"语音合成异常（{exc}），正在后台重试…")
                    token.wait(min(0.2 * 2 ** (attempt - 1), 1.0))
                else:
                    break
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
                    if self.on_utterance_failed:
                        self.on_utterance_failed(text)

    def _finish_prefetch_batch(self, token):
        with self._state_lock:
            self._prefetch_tokens.discard(token)
        self._queue.task_done()

    def _prefetch_loop(self, pool):
        while not self._cancel.is_set():
            with self._state_lock:
                try:
                    item, token = self._queue.get_nowait()
                except queue.Empty:
                    item = None
                if item is not None:
                    self._prefetch_tokens.add(token)
            if item is None:
                self._cancel.wait(.02)
                continue

            texts = item if isinstance(item, tuple) else (item,)
            tracker = _BatchCompletion(
                len(texts), lambda t=token: self._finish_prefetch_batch(t)
            )
            for text in texts:
                acquired = False
                while not self._cancel.is_set() and not token.is_set():
                    acquired = self._prefetch_slots.acquire(timeout=.05)
                    if acquired:
                        break
                if not acquired or self._cancel.is_set() or token.is_set():
                    if acquired:
                        self._prefetch_slots.release()
                    tracker.complete_one()
                    continue
                try:
                    future = pool.submit(self._buffer_one, text, token)
                    self._prepared.put((text, future, token, tracker))
                except Exception:
                    self._prefetch_slots.release()
                    tracker.complete_one()

    def _run(self):
        """Prefetch across consecutive answers, then play through one ordered owner."""
        if self.settings.get("speech_mode") != "natural":
            return super()._run()
        pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="defense-tts-prefetch")
        feeder = threading.Thread(
            target=self._prefetch_loop,
            args=(pool,),
            name="defense-tts-prefetch-feed",
            daemon=True,
        )
        feeder.start()
        try:
            while True:
                if self._cancel.is_set() and not feeder.is_alive() and self._prepared.empty():
                    break
                try:
                    text, future, token, tracker = self._prepared.get(timeout=.05)
                except queue.Empty:
                    continue
                with self._state_lock:
                    self._active_cancel = token
                    self._speaking = True
                self._emitted_bytes = 0
                try:
                    pcm = future.result()
                    self._deliver_buffered(text, pcm, token)
                except _Cancelled:
                    pass
                except Exception as exc:
                    self._last_error = str(exc)
                    self.on_error(f"这一句后台合成失败，已跳过：{exc}")
                    if self.on_utterance_failed:
                        self.on_utterance_failed(text)
                finally:
                    with self._state_lock:
                        self._active_cancel = None
                        self._speaking = False
                    self._prefetch_slots.release()
                    tracker.complete_one()
                    self._prepared.task_done()
        finally:
            self._cancel.set()
            with self._state_lock:
                for token in tuple(self._prefetch_tokens):
                    token.set()
            feeder.join(timeout=2)
            pool.shutdown(wait=True, cancel_futures=True)
            self._disconnect()
            self._closed.set()


class TtsLegacySession(TtsHttpSession):
    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_legacy_tts(text, self.settings, on_audio, cancel_event)


class TtsMiniMaxSession(TtsHttpSession):
    """MiniMax 走 HTTP 同步合成，复用 HTTP 会话的并行预生成与有序播放。"""

    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_minimax_tts(text, self.settings, on_audio, cancel_event)


class TtsNextSession(TtsHttpSession):
    """Non-streaming AudioGen reference synthesis; reuse ordered HTTP prefetch playback."""

    def __init__(self, *args, **kwargs):
        kwargs["utterance_timeout"] = max(180.0, float(kwargs.get("utterance_timeout", 180.0)))
        super().__init__(*args, **kwargs)

    def _stream_call(self, text, on_audio, cancel_event):
        self.client._stream_next_audio(text, self.settings, on_audio, cancel_event)


TTS_BACKUP_DELAY = 2.0          # 「稍后」模式：主通道这么久没有任何音频才启动第二通道
TTS_PREFER_PRIMARY = 0.35       # 第二通道换了模型 / 音色时，它先好了也再等主通道这一小会儿，声音尽量统一


def plain_text_for_tts(text: str) -> str:
    """去掉 SSML 标记只留文字：第二通道的模型不认 SSML 时用，免得把标记念出来。"""
    if "<" not in text:
        return text
    import html

    plain = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", plain).strip() or text


class TtsBackupSession(TtsHttpSession):
    """现场发声的第二通道：主 / 第二通道各合成一份，**只交出一份**，另一份取消、音频丢掉。

    * 自然发声（整句合成完才播）：谁先合成出**完整有效**的整句用谁；
    * 快速发声（边合成边播）：谁先吐出第一包音频用谁，定了就不再换——前半句已经播出去了；
    * 「稍后」模式（默认）按**有没有音频出来**判断，不是按整句有没有合成完：主通道
      TTS_BACKUP_DELAY 秒内没有任何新音频、或者直接失败，才启动第二通道。主通道正常时
      第二通道根本不会发请求，不多花钱，声音也不会换。
    * 「同时」模式：两个一起发；第二通道换了模型 / 音色时，它先好了也再礼让主通道一小会儿。
    只用于现场发声；讲稿台 / 问答库 / 快捷用语的提前渲染不走这里（见 pipeline._live_tts_settings）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner: dict[int, Any] = {}
        self._inner_lock = threading.Lock()
        self._backup_texts: set[str] = set()
        self._announced: set[str] = set()
        self.channel_wins = [0, 0]

    # ---- 配置
    def _channel_settings(self):
        """(主通道参数, 第二通道参数, 第二通道不可用的原因)。"""
        from ..api_payloads import INSTRUCTION_MODELS, SSML_MODELS

        primary = {**self.settings, "backup_tts_enabled": False}
        secondary = dict(primary)
        model = str(self.settings.get("backup_tts_model", "") or self.model)
        voice = str(self.settings.get("backup_tts_voice_id", "") or "").strip()
        issue = ""
        if model != self.model and not voice:
            issue = "备用合成模型不同，必须选择与其绑定的备用音色"
        secondary.update(tts_model=model, voice=voice or self.voice)
        if model not in INSTRUCTION_MODELS:
            secondary["tts_instruction"] = ""
        if model not in SSML_MODELS:
            secondary["tts_enable_ssml"] = False      # 文字也会去掉标记，见 _stream_call
        return primary, secondary, issue

    def took_backup(self, text) -> bool:
        """这句是不是第二通道用**另一个模型 / 音色**合成的（是的话别写进主模型的磁盘缓存）。取一次就清。"""
        with self._inner_lock:
            if text in self._backup_texts:
                self._backup_texts.discard(text)
                return True
            return False

    def _say_once(self, key: str, message: str) -> None:
        with self._inner_lock:
            if key in self._announced:
                return
            self._announced.add(key)
        self.on_status(message)

    # ---- 单个通道的一次合成（测试替换这里）
    def _channel_call(self, index, settings, text, emit, cancel_event):
        with self._inner_lock:
            inner = self._inner.get(index)
            if inner is None:
                inner = create_tts_session(self.client, settings, on_audio=lambda _pcm: None)
                self._inner[index] = inner
        if isinstance(inner, TtsHttpSession):
            # HTTP 家族：每次调用互不相干，可以并发，音频一包一包出来
            inner._stream_call(text, emit, cancel_event)
        else:
            # 实时 WebSocket 版是单连接串行的：单开一个短命会话，整句一次交出
            emit(synthesize_once(self.client, settings, text, timeout=self.utterance_timeout,
                                 cancel_event=cancel_event))

    def _stream_call(self, text, on_audio, cancel_event):
        self._check_cancel(cancel_event)
        primary, secondary, issue = self._channel_settings()
        if issue:
            self._say_once("issue", f"备用合成配置不适用，仅使用主通道：{issue}")
            self._account(primary, text)
            self._channel_call(0, primary, text, on_audio, cancel_event)
            return
        streaming = self.settings.get("speech_mode") != "natural"
        mode = str(self.settings.get("backup_tts_mode", "delayed") or "delayed")
        same_voice = (secondary["tts_model"], secondary["voice"]) == (primary["tts_model"], primary.get("voice"))
        grace = 0.0 if same_voice else TTS_PREFER_PRIMARY
        configs = (primary, secondary)
        strip_ssml = bool(primary.get("tts_enable_ssml")) and not secondary.get("tts_enable_ssml")
        texts = (text, plain_text_for_tts(text) if strip_ssml else text)
        events: queue.Queue = queue.Queue()
        flags = (threading.Event(), threading.Event())
        started, alive, done = [False, False], [False, False], [False, False]
        bufs = (bytearray(), bytearray())
        errors: dict[int, BaseException] = {}

        def run(index):
            try:
                self._channel_call(index, configs[index], texts[index],
                                   lambda chunk, i=index: events.put(("audio", i, bytes(chunk))) if chunk else None,
                                   flags[index])
                events.put(("done", index, None))
            except BaseException as exc:  # noqa: BLE001
                events.put(("error", index, exc))

        def launch(index):
            started[index] = alive[index] = True
            self._account(configs[index], texts[index])
            threading.Thread(target=run, args=(index,), name=f"tts-channel-{index}", daemon=True).start()

        winner = None
        candidate = None            # 第二通道已经够格，正在礼让主通道
        grace_until = 0.0

        def decide(index):
            nonlocal winner
            winner = index
            flags[1 - index].set()
            with self._inner_lock:
                self.channel_wins[index] += 1
                if index == 1 and not same_voice:
                    self._backup_texts.add(text)
                else:
                    self._backup_texts.discard(text)
            if index == 1:
                why = "主通道失败" if 0 in errors else "主通道慢"
                self._say_once("failed" if 0 in errors else "slow",
                               f"语音合成：这句用了第二通道（{secondary['tts_model']}，{why}）")
            if streaming and bufs[index]:
                chunk = bytes(bufs[index])
                bufs[index].clear()
                on_audio(chunk)

        def fail_all():
            if isinstance(errors.get(0), _Cancelled):
                raise errors[0]
            raise ApiError(f"两路合成都失败：主通道 {errors.get(0)}；备用 {errors.get(1)}")

        began = time.monotonic()
        last_primary = began
        launch(0)
        if mode != "delayed":
            launch(1)
        try:
            while True:
                self._check_cancel(cancel_event)
                now = time.monotonic()
                if now - began >= self.utterance_timeout:
                    raise ApiError("两路合成等待超时")
                if winner is None and candidate is not None and now >= grace_until:
                    decide(candidate)
                    candidate = None
                    if done[winner]:
                        break
                if winner is None and not started[1] and now - last_primary >= TTS_BACKUP_DELAY:
                    launch(1)
                try:
                    kind, index, payload = events.get(timeout=.02)
                except queue.Empty:
                    continue
                if winner is not None and index != winner:
                    continue                                  # 输家的尾巴：一个字节都不要
                if kind == "audio":
                    if index == 0:
                        last_primary = time.monotonic()
                    if winner == index and streaming:
                        on_audio(payload)
                        continue
                    bufs[index].extend(payload)
                    if streaming and winner is None:
                        if index == 0:
                            candidate = None
                            decide(0)
                        elif candidate is None:
                            if grace and alive[0]:
                                candidate, grace_until = 1, time.monotonic() + grace
                            else:
                                decide(1)
                    continue
                alive[index] = False
                if kind == "done" and not streaming and (not bufs[index] or len(bufs[index]) % 2):
                    kind, payload = "error", ApiError("通道未返回完整有效的 PCM 音频")
                if kind == "done" and len(bufs[index]) > 16 * 1024 * 1024:
                    kind, payload = "error", ApiError("通道音频过长")
                if kind == "done":
                    done[index] = True
                    if winner == index:
                        break
                    if winner is None and not bufs[index]:
                        kind, payload = "error", ApiError("通道未返回完整有效的 PCM 音频")
                    elif winner is None:
                        if index == 0 or not (grace and alive[0]):
                            candidate = None
                            decide(index)
                            break
                        if candidate is None:
                            candidate, grace_until = 1, time.monotonic() + grace
                        continue
                if kind == "error":
                    errors[index] = payload
                    if winner == index:
                        raise payload
                    if index == 0 and candidate == 1:
                        candidate = None
                        decide(1)
                        if done[1]:
                            break
                        continue
                    if index == 0 and not started[1]:
                        launch(1)                             # 主通道没出声就失败：马上用第二通道
                        continue
                    if not any(alive):
                        fail_all()
            if not streaming:
                self._check_cancel(cancel_event)
                on_audio(bytes(bufs[winner]))
        finally:
            for flag in flags:
                flag.set()

    def _account(self, settings, text) -> None:
        try:
            from .usage import USAGE
            USAGE.add_tts(str(settings.get("tts_model", "")), len(re.sub(r"<[^>]+>", "", str(text))))
        except Exception:  # noqa: BLE001 —— 记账失败不影响发声
            pass


def synthesize_once(client, settings, text, *, timeout=120.0, cancel_event=None,
                    on_status=None):
    """同步合成一句，返回裸 PCM16。不碰正在直播的那个会话。

    讲稿预渲染要在后台一句一句地合成，绝不能借用 live 会话——那个会话的
    声卡流只能由它自己的拥有者线程写，插队进去就是 PortAudio 崩溃。
    所以这里每次新建一个短命会话，只取音频、不接播放器。
    """
    text = str(text or "").strip()
    if cancel_event is not None and cancel_event.is_set():
        raise ApiError("已取消")
    if str(settings.get("tts_model", "")) == "qwen-audio-3.1-tts-next":
        timeout = max(timeout, 180.0)
    if not text:
        raise ApiError("没有可合成的内容")
    captured = bytearray()
    errors: list[str] = []
    finished = threading.Event()

    session = create_tts_session(
        client,
        settings,
        on_audio=captured.extend,
        on_status=on_status or (lambda message: None),
        on_error=errors.append,
        on_utterance_done=lambda *_args: finished.set(),
        on_utterance_failed=lambda *_args: finished.set(),
        connect_timeout=15,
        utterance_timeout=timeout,
        max_attempts=2,
    )
    session.start()
    try:
        if not session.speak(text):
            raise ApiError("合成任务未能提交")
        deadline = time.monotonic() + timeout
        while not finished.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("已取消")
            if time.monotonic() > deadline:
                raise ApiError("合成超时")
    finally:
        session.close()

    if errors:
        raise ApiError(errors[0])
    if not captured:
        raise ApiError("语音服务未返回有效音频")
    if len(captured) % 2:
        raise ApiError("语音服务返回了不完整的 PCM 音频")
    return bytes(captured)


def create_tts_session(client, settings, **kwargs):
    """按模型家族挑选会话实现。

    这里曾经硬编码 `qwen-audio-3.0-tts` 前缀，于是 qwen-audio-3.1-tts-flash
    虽然在能力表和下拉框里都有，一开口就报「不支持的语音合成模型」。
    判断一律走 api_payloads 里的家族函数，不要在这里再写前缀。
    """
    from ..api_payloads import (
        is_cosyvoice_model,
        is_minimax_tts_model,
        is_qwen3_tts_vc_model,
        is_qwen3_tts_vc_realtime_model,
        is_qwen_audio_tts_model,
    )
    if settings.get("backup_tts_enabled", False):
        return TtsBackupSession(client, settings, **kwargs)
    model = str(settings.get("tts_model", ""))
    if model == "qwen-audio-3.1-tts-next":
        return TtsNextSession(client, settings, **kwargs)
    if is_qwen3_tts_vc_realtime_model(model):
        return TtsSession(client, settings, **kwargs)
    if is_qwen3_tts_vc_model(model):
        return TtsHttpSession(client, settings, **kwargs)
    if is_cosyvoice_model(model) or is_qwen_audio_tts_model(model):
        return TtsLegacySession(client, settings, **kwargs)
    if is_minimax_tts_model(model):
        return TtsMiniMaxSession(client, settings, **kwargs)
    raise ApiError(f"不支持的语音合成模型：{model}")
