"""千问同传（qwen3.8-livetranslate-flash-realtime）：我说中文 → 直接出英文。

和原来「识别 → 大模型翻译」两步走的区别：同传模型边听边译，平均时延约 2.3 秒
（官方数据），不用等你一句话说完再开始翻译。

**只用它的文字输出。** 发声仍然走你的克隆音色（原有语音合成链路），这样音色、
缓存、重播、字幕全部不变；它自带的语音输出和「自动复刻说话人」这次不用。

对外接口与 ``QwenRealtimeASR`` 相同（start / wait_ready / send_audio / finish /
close），所以答辩引擎可以直接替换，麦克风、回声抑制、停止逻辑都不用改。

协议依据（百炼文档，2026-09）：
* 连接 ``wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...``
* 音频：``input_audio_buffer.append``，16 kHz 单声道 PCM16，base64
* 3.8 的配置字段是 ``output_modalities``，3.5 是 ``modalities``
* 3.8 译文事件 ``response.text.delta`` / ``response.text.done``；
  3.5 是 ``response.text.text``；两者原文转写都有
  ``conversation.item.input_audio_transcription.*``
* 术语 ``translation.corpus.phrases``（原文→译文，最多 1000 条）——文档只在 3.5
  的字段表里写了，3.8 未写明，所以服务端若拒绝配置就自动去掉可选字段重试一次。
"""
from __future__ import annotations

import base64
import json
import threading
import uuid
from typing import Any, Callable

import websocket

from ..aliyun import ApiError, _websocket_proxy_options

LIVETRANSLATE_MODELS: list[tuple[str, str]] = [
    ("qwen3.8-livetranslate-flash-realtime", "最新同传，官方平均时延约 2.3 秒"),
    ("qwen3.5-livetranslate-flash-realtime", "上一代同传"),
]
DEFAULT_LIVETRANSLATE_MODEL = LIVETRANSLATE_MODELS[0][0]
MAX_CORPUS_PHRASES = 1000


def is_v38(model: str) -> bool:
    return "3.8" in (model or "")


def build_session_update(
    model: str,
    *,
    source_language: str = "zh",
    target_language: str = "en",
    phrases: dict[str, str] | None = None,
    minimal: bool = False,
) -> dict[str, Any]:
    """只要文字输出。minimal=True 时只留两个必填项，用于服务端拒绝可选字段时重试。"""
    modalities_key = "output_modalities" if is_v38(model) else "modalities"
    session: dict[str, Any] = {
        modalities_key: ["text"],
        "translation": {"language": target_language},
    }
    if not minimal:
        session["input_audio_transcription"] = {"language": source_language}
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in (phrases or {}).items()
            if str(k).strip() and str(v).strip()
        }
        if cleaned:
            limited = dict(list(cleaned.items())[:MAX_CORPUS_PHRASES])
            session["translation"]["corpus"] = {"phrases": limited}
    return {
        "event_id": f"event_{uuid.uuid4().hex}",
        "type": "session.update",
        "session": session,
    }


class LiveTranslateState:
    """把服务端事件拼成「一段原文 + 一段译文」。纯逻辑，方便离线测试。

    一次 response 对应一段译文。原文转写和译文是两条独立的事件流，时间上不
    严格对齐：译文完成时，取上一段译文之后累计到的原文（已完成的 + 正在识别
    的）作为这段的原文，只用于界面显示和对话记录。
    """

    def __init__(self) -> None:
        self.completed_source: list[str] = []
        self.partial_source = ""
        self.translation = ""

    def source_preview(self) -> str:
        return ("".join(self.completed_source) + self.partial_source).strip()

    def on_event(self, event: dict[str, Any]) -> tuple[str, Any]:
        """返回 (动作, 数据)。动作：preview / translation_delta / segment / ready /
        error / finished / ""（忽略）。"""
        kind = str(event.get("type", ""))
        if kind == "session.updated":
            return "ready", None
        if kind == "conversation.item.input_audio_transcription.delta":
            self.partial_source += str(event.get("delta", "") or "")
            return "preview", self.source_preview()
        if kind == "conversation.item.input_audio_transcription.text":
            # 3.5：text 是已确定部分，stash 是暂定部分
            self.partial_source = f"{event.get('text', '')}{event.get('stash', '')}"
            return "preview", self.source_preview()
        if kind == "conversation.item.input_audio_transcription.completed":
            transcript = str(event.get("transcript", "") or "").strip()
            if transcript:
                self.completed_source.append(transcript)
            self.partial_source = ""
            return "preview", self.source_preview()
        if kind == "response.text.delta":
            self.translation += str(event.get("delta", "") or "")
            return "translation_delta", self.translation
        if kind == "response.text.text":
            self.translation = f"{event.get('text', '')}{event.get('stash', '')}"
            return "translation_delta", self.translation
        if kind in ("response.text.done", "response.audio_transcript.done"):
            final = str(event.get("text") or event.get("transcript") or self.translation or "")
            return self._close_segment(final)
        if kind == "response.done":
            # 有的实现不单独发 text.done：收尾时手里还有译文就当一段
            if self.translation.strip():
                return self._close_segment(self.translation)
            return "", None
        if kind == "session.finished":
            return "finished", None
        if kind == "error":
            error = event.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else None
            return "error", message or event.get("message") or str(error or event)
        return "", None

    def _close_segment(self, final: str) -> tuple[str, Any]:
        english = final.strip()
        source = self.source_preview()
        self.translation = ""
        self.completed_source = []
        self.partial_source = ""
        if not english:
            return "", None
        return "segment", (source, english)


class LiveTranslateRealtime:
    """与 QwenRealtimeASR 同接口，但 on_segment 给出的是 (中文原文, 英文译文)。"""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str = DEFAULT_LIVETRANSLATE_MODEL,
        source_language: str = "zh",
        target_language: str = "en",
        phrases: dict[str, str] | None = None,
        on_preview: Callable[[str], None],
        on_translation: Callable[[str, str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_translation_delta: Callable[[str], None] | None = None,
        proxy: str = "",
    ) -> None:
        if not api_key.strip():
            raise ApiError("尚未设置百炼 API Key")
        if not workspace_id.strip():
            raise ApiError("尚未设置百炼 Workspace ID")
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id.strip()
        self.model = model or DEFAULT_LIVETRANSLATE_MODEL
        self.source_language = source_language
        self.target_language = target_language
        self.phrases = dict(phrases or {})
        self.on_preview = on_preview
        self.on_translation = on_translation
        self.on_translation_delta = on_translation_delta
        self.on_status = on_status
        self.on_error = on_error
        self.proxy = proxy.strip()
        self.state = LiveTranslateState()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.opened = threading.Event()
        self.error_text = ""
        self._retried_minimal = False
        self.ws: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None

    def url(self) -> str:
        return (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            f"/api-ws/v1/realtime?model={self.model}"
        )

    def start(self) -> None:
        ws = websocket.WebSocketApp(
            self.url(),
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_ws_error,
            on_close=self._on_close,
        )
        self.ws = ws
        run_options = {"ping_interval": 20, "ping_timeout": 10}
        run_options.update(_websocket_proxy_options(self.proxy))
        self.thread = threading.Thread(
            target=lambda: ws.run_forever(**run_options),
            name="livetranslate-websocket",
            daemon=True,
        )
        self.thread.start()

    def wait_ready(self, timeout: float = 12.0) -> None:
        if not self.ready.wait(timeout):
            if self.error_text:
                raise ApiError(self.error_text)
            raise ApiError("同传连接超时，请检查 Workspace ID、API Key、网络，以及账号是否开通了同传模型")
        if self.error_text:
            raise ApiError(self.error_text)

    def send_audio(self, pcm: bytes) -> None:
        if not self.ws or not self.opened.is_set():
            raise ApiError("同传尚未连接")
        self.ws.send(json.dumps({
            "event_id": f"event_{uuid.uuid4().hex}",
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }))

    def finish(self, timeout: float = 8.0) -> str:
        if self.ws and self.opened.is_set():
            try:
                self.ws.send(json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"}))
                self.finished.wait(timeout)
            except Exception:
                pass
        self.close()
        return self.state.source_preview()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)

    # ------------------------------------------------------------ websocket
    def _send_session(self, ws, *, minimal: bool) -> None:
        event = build_session_update(
            self.model,
            source_language=self.source_language,
            target_language=self.target_language,
            phrases=self.phrases,
            minimal=minimal,
        )
        ws.send(json.dumps(event, ensure_ascii=False))

    def _on_open(self, ws) -> None:
        self.opened.set()
        self.on_status(f"同传已连接（{self.model}）")
        self._send_session(ws, minimal=False)

    def _on_message(self, ws, message: str | bytes) -> None:
        if isinstance(message, bytes):
            return
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        action, data = self.state.on_event(event)
        if action == "ready":
            self.ready.set()
        elif action == "preview":
            if data:
                self.on_preview(data)
        elif action == "translation_delta":
            if self.on_translation_delta is not None and data:
                self.on_translation_delta(data)
        elif action == "segment":
            source, english = data
            self.on_translation(source, english)
        elif action == "finished":
            self.finished.set()
        elif action == "error":
            text = str(data)
            if not self.ready.is_set() and not self._retried_minimal:
                # 可选字段（术语、原文语种）可能不被这个版本接受：去掉后再试一次
                self._retried_minimal = True
                self.on_status(f"同传配置被拒绝（{text[:60]}），去掉术语等可选项重试…")
                try:
                    self._send_session(ws, minimal=True)
                    return
                except Exception:
                    pass
            self.error_text = text
            self.on_error(text)
            self.ready.set()
            self.finished.set()

    def _on_ws_error(self, _ws, error: Any) -> None:
        self.error_text = str(error)
        self.on_error(self.error_text)
        self.ready.set()
        self.finished.set()

    def _on_close(self, _ws, _status_code, _message) -> None:
        self.opened.clear()
        self.finished.set()


def glossary_phrases(terms: list[dict[str, str]]) -> dict[str, str]:
    """术语表（中→英）直接当同传热词：原文→指定译法。"""
    return {t["source"]: t["target"] for t in terms if t.get("source") and t.get("target")}
