from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import websocket

from .aliyun import ApiError
from .api_payloads import build_live_translate_session_update


@dataclass(frozen=True)
class LiveTranslateResult:
    source_text: str
    translated_text: str
    first_audio_seconds: float | None
    usage: dict[str, Any]


class QwenLiveTranslate:
    """One push-to-talk Qwen3.5 LiveTranslate WebSocket session."""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str,
        source_language: str = "zh",
        target_language: str = "en",
        phrases: dict[str, str] | None = None,
        voice_mode: str = "once",
        voice: str = "",
        audio_enabled: bool = True,
        on_source_preview: Callable[[str], None] | None = None,
        on_translation_preview: Callable[[str], None] | None = None,
        on_audio: Callable[[bytes], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.model = model
        self.on_source_preview = on_source_preview or (lambda _text: None)
        self.on_translation_preview = on_translation_preview or (lambda _text: None)
        self.on_audio = on_audio or (lambda _pcm: None)
        self.on_status = on_status or (lambda _text: None)
        self.on_error = on_error or (lambda _text: None)
        self.session_payload = build_live_translate_session_update(
            source_language=source_language,
            target_language=target_language,
            phrases=phrases,
            voice_mode=voice_mode,
            voice=voice,
            audio_enabled=audio_enabled,
        )

        self.ready = threading.Event()
        self.response_done = threading.Event()
        self.session_finished = threading.Event()
        self.opened = threading.Event()
        self.error_text = ""
        self.source_text = ""
        self.translated_text = ""
        self.usage: dict[str, Any] = {}
        self.first_audio_seconds: float | None = None
        self._committed_at: float | None = None
        self.ws: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            f"/api-ws/v1/realtime?model={self.model}"
        )

    def start(self) -> None:
        self.ws = websocket.WebSocketApp(
            self.url,
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(
            target=lambda: self.ws.run_forever(ping_interval=20, ping_timeout=10),
            name="qwen-live-translate-websocket",
            daemon=True,
        )
        self.thread.start()

    def wait_ready(self, timeout: float = 12.0) -> None:
        if not self.ready.wait(timeout):
            raise ApiError(self.error_text or "极速直译连接超时，请检查模型权限和网络")
        self._raise_if_error()

    def send_audio(self, pcm: bytes) -> None:
        if not self.ws or not self.opened.is_set():
            raise ApiError("极速直译连接尚未就绪")
        self._send("input_audio_buffer.append", audio=base64.b64encode(pcm).decode("ascii"))

    def commit_and_wait(
        self,
        *,
        timeout: float = 45.0,
        cancel_event: threading.Event | None = None,
    ) -> LiveTranslateResult:
        self._raise_if_error()
        if not self.ws or not self.opened.is_set():
            raise ApiError("极速直译连接已断开")
        self._committed_at = time.perf_counter()
        self._send("input_audio_buffer.commit")
        deadline = time.monotonic() + timeout
        while not self.response_done.wait(0.1):
            self._raise_if_error()
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("已取消极速直译")
            if time.monotonic() >= deadline:
                raise ApiError("极速直译等待响应超时")
        self._raise_if_error()
        if not self.translated_text.strip():
            raise ApiError("极速直译没有返回有效译文")
        return LiveTranslateResult(
            source_text=self.source_text.strip(),
            translated_text=self.translated_text.strip(),
            first_audio_seconds=self.first_audio_seconds,
            usage=dict(self.usage),
        )

    def finish(self, timeout: float = 15.0) -> None:
        if self.ws and self.opened.is_set():
            try:
                self._send("session.finish")
                self.session_finished.wait(timeout)
            except Exception:
                pass
        self.close()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)

    def _send(self, kind: str, **payload: Any) -> None:
        assert self.ws is not None
        event = {"event_id": f"event_{uuid.uuid4().hex}", "type": kind, **payload}
        self.ws.send(json.dumps(event, ensure_ascii=False))

    def _raise_if_error(self) -> None:
        if self.error_text:
            raise ApiError(self.error_text)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self.opened.set()
        payload = dict(self.session_payload)
        payload["event_id"] = f"event_{uuid.uuid4().hex}"
        ws.send(json.dumps(payload, ensure_ascii=False))

    def _on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        kind = event.get("type", "")
        if kind == "session.updated":
            self.ready.set()
            self.on_status("极速直译已连接 · 正在接收中文语音")
        elif kind == "conversation.item.input_audio_transcription.text":
            text = f"{event.get('text', '')}{event.get('stash', '')}".strip()
            if text:
                self.source_text = text
                self.on_source_preview(text)
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = (event.get("transcript") or "").strip()
            if text:
                self.source_text = text
                self.on_source_preview(text)
        elif kind in {"response.audio_transcript.text", "response.text.text"}:
            text = f"{event.get('text', '')}{event.get('stash', '')}".strip()
            if text:
                self.translated_text = text
                self.on_translation_preview(text)
        elif kind == "response.audio_transcript.done":
            text = (event.get("transcript") or "").strip()
            if text:
                self.translated_text = text
                self.on_translation_preview(text)
        elif kind == "response.text.done":
            text = (event.get("text") or "").strip()
            if text:
                self.translated_text = text
                self.on_translation_preview(text)
        elif kind == "response.audio.delta":
            encoded = event.get("delta") or ""
            if encoded:
                if self.first_audio_seconds is None and self._committed_at is not None:
                    self.first_audio_seconds = time.perf_counter() - self._committed_at
                self.on_audio(base64.b64decode(encoded))
        elif kind == "response.done":
            self.usage = (event.get("response") or {}).get("usage") or {}
            self.response_done.set()
        elif kind == "session.finished":
            self.session_finished.set()
        elif kind == "error":
            error = event.get("error") or {}
            self.error_text = error.get("message") or str(error)
            self.on_error(self.error_text)
            self.ready.set()
            self.response_done.set()
            self.session_finished.set()

    def _on_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        self.error_text = str(error)
        self.on_error(self.error_text)
        self.ready.set()
        self.response_done.set()
        self.session_finished.set()

    def _on_close(self, _ws: websocket.WebSocketApp, _code: int, _message: str) -> None:
        self.opened.clear()
        if not self.response_done.is_set() and not self.session_finished.is_set() and not self.error_text:
            detail = f"（{_code}: {_message}）" if _code or _message else ""
            self.error_text = f"极速直译连接意外关闭{detail}"
            self.on_error(self.error_text)
            self.ready.set()
            self.response_done.set()
        self.session_finished.set()
