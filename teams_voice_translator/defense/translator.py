from __future__ import annotations

import json
import threading
import re
import time
from collections.abc import Callable
from typing import Any

import requests

from ..aliyun import ApiError, _extract_api_error
from .memory import MeetingMemory

DIRECTIONS = ("zh2en", "en2zh")

DEFAULT_LLM_MODEL = "qwen-plus"


class SentenceSplitter:
    """Cut streaming translation text into speakable sentences.

    Feeding LLM deltas as they arrive and dispatching each finished sentence
    to TTS immediately is what keeps long answers flowing without waiting for
    the whole utterance.  A period only splits when followed by whitespace and
    a capital/quote/digit (or enough buffer), so "e.g." and "Fig." survive.
    """

    HARD_ENDINGS = "。！？；!?;…"
    SOFT_ENDING = "."
    _MAX_SOFT_BUFFER = 80

    def __init__(self, *, min_length: int = 3) -> None:
        self.min_length = max(1, int(min_length))
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        self._buffer += piece
        sentences: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> list[str]:
        rest = self._buffer.strip()
        self._buffer = ""
        return [rest] if rest else []

    def _find_cut(self) -> int | None:
        for index, char in enumerate(self._buffer):
            if char in self.HARD_ENDINGS:
                return index + 1
            if char == self.SOFT_ENDING:
                prefix = self._buffer[:index + 1]
                if re.search(r"\b(?:e\.g|i\.e|Fig|Figs|Eq|Eqs|Dr|Mr|Mrs|Prof|vs|etc)\.$", prefix, re.I):
                    continue
                following = self._buffer[index + 1 :]
                stripped = following.lstrip()
                if not stripped:
                    # Ends with ". " and nothing after yet: wait for more text
                    # unless the buffer already grew past the soft cap.
                    if len(self._buffer) > self._MAX_SOFT_BUFFER:
                        return index + 1
                    continue
                if following[0] in " \t\n" and (stripped[0].isupper() or stripped[0].isdigit() or stripped[0] in "\"'“"):
                    if index + 1 >= self.min_length:
                        return index + 1
        return None


class ContextTranslator:
    """Streaming LLM translation with the whole-meeting context attached."""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.2,
        timeout: int = 45,
        proxy: str = "",
        proxy_mode: str = "",
    ) -> None:
        if not api_key.strip():
            raise ApiError("尚未设置百炼 API Key")
        if not workspace_id.strip():
            raise ApiError("尚未设置百炼 Workspace ID")
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id.strip()
        self.model = model or DEFAULT_LLM_MODEL
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        proxy = proxy.strip()
        if proxy_mode == "direct":
            # 与 BailianClient 一致：直连模式显式禁用系统/VPN 代理。
            self.proxies = {"http": None, "https": None}
        elif proxy:
            self.proxies = {"http": proxy, "https": proxy}
        else:
            self.proxies = None

    @property
    def endpoint(self) -> str:
        return f"https://{self.workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

    def build_payload(
        self,
        memory: MeetingMemory,
        direction: str,
        text: str,
        *,
        stream: bool = True,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        if direction not in DIRECTIONS:
            raise ValueError(f"未知翻译方向：{direction}")
        return {
            "model": self.model,
            "messages": memory.build_messages(direction, text, max_turns=max_turns),
            "temperature": self.temperature,
            "stream": stream,
        }

    def _post_stream(
        self,
        payload: dict[str, Any],
        on_piece: Callable[[str, str], None],
        cancel_event: threading.Event | None,
    ) -> str:
        """POST a streaming chat request and return the concatenated content."""
        if cancel_event is not None and cancel_event.is_set():
            raise ApiError("已取消")
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
                stream=True,
                timeout=(10, max(self.timeout, 60)),
                proxies=self.proxies,
            )
        except requests.RequestException as exc:
            raise ApiError(f"翻译服务连接失败：{exc}") from exc
        with response:
            if not response.ok:
                raise ApiError(_extract_api_error(response))
            parts: list[str] = []
            completed = False
            deadline = time.monotonic() + max(self.timeout, 60)
            for raw_line in response.iter_lines(decode_unicode=True):
                if cancel_event is not None and cancel_event.is_set():
                    raise ApiError("已取消")
                if not raw_line:
                    continue
                if time.monotonic() > deadline:
                    raise ApiError("翻译响应超时，请稍后重试")
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    completed = True
                    break
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    raise ApiError(f"翻译服务错误：{event['error']}")
                if event.get("code"):
                    raise ApiError(f"{event.get('code')}: {event.get('message', '')}")
                choices = event.get("choices") or []
                if not choices:
                    continue
                reason = choices[0].get("finish_reason")
                if reason in ("length", "content_filter"):
                    raise ApiError("译文被截断或过滤，未将其作为完整回答播放，请缩短后重试")
                if reason == "stop":
                    completed = True
                piece = ((choices[0].get("delta") or {}).get("content")) or ""
                if not piece:
                    continue
                parts.append(piece)
                if on_piece is not None:
                    on_piece(piece, "".join(parts))
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("已取消")
            if parts and not completed:
                raise ApiError("翻译连接提前结束，译文不完整，请重试")
        return "".join(parts)

    def translate(
        self,
        memory: MeetingMemory,
        direction: str,
        text: str,
        *,
        on_delta: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Translate one utterance; stream deltas to ``on_delta(piece, so_far)``."""
        payload = self.build_payload(memory, direction, text, stream=True, max_turns=max_turns)
        result = self._post_stream(payload, on_delta, cancel_event).strip()
        if not result:
            raise ApiError("翻译模型返回了空文本")
        return result

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Stream a free-form chat completion (used by the Q&A advisor)."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        result = self._post_stream(payload, on_delta, cancel_event).strip()
        if not result:
            raise ApiError("模型返回了空内容")
        return result
