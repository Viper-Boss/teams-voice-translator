from __future__ import annotations

import json
import threading
import re
import time
from collections.abc import Callable
from typing import Any

import requests

from ..aliyun import ApiError, _extract_api_error
from .chat_models import thinking_fields
from .usage import record_llm_usage

# 流式请求带上 stream_options.include_usage，最后一块会返回 token 用量（用于「本次用了多少 token」）。
# 万一哪个模型不认这个参数（返回 400 且提到 stream_options），自动去掉重试一次，并从此不再带。
_USAGE_OPTION = {"enabled": True}
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
        backup=None,
    ) -> None:
        # backup：channels.BackupChannel 或 None。有它时答辩现场的请求走「两个通道谁先出字用谁」。
        self.backup = backup
        self.last_channel = ""       # 最近一次用上的是哪个通道（界面 / 日志用）
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
        payload = {
            "model": self.model,
            "messages": memory.build_messages(direction, text, max_turns=max_turns),
            "temperature": self.temperature,
            "stream": stream,
        }
        # 实时翻译绝不能先「思考」：新一代模型默认开思考，正文要等思考完才吐，
        # 界面会沉默几秒到几十秒。显式关掉；未知/旧模型则什么都不发。
        payload.update(thinking_fields(self.model, want_thinking=False))
        return payload

    def _post_stream(
        self,
        payload: dict[str, Any],
        on_piece: Callable[[str, str], None],
        cancel_event: threading.Event | None,
        *,
        on_reasoning: Callable[[int], None] | None = None,
        deadline_seconds: float | None = None,
    ) -> str:
        """发一个流式请求。配了第二通道就两个通道赛跑，否则和原来一样只走主通道。"""
        if self.backup is None:
            self.last_channel = "主通道"
            return self._stream_once(self.endpoint, self.api_key, payload, on_piece, cancel_event,
                                     on_reasoning=on_reasoning, deadline_seconds=deadline_seconds)
        return self._race(payload, on_piece, cancel_event, on_reasoning=on_reasoning,
                          deadline_seconds=deadline_seconds)

    def _race(self, payload, on_piece, cancel_event, *, on_reasoning=None, deadline_seconds=None) -> str:
        cancels = [threading.Event(), threading.Event()]
        try:
            return self._race_impl(payload, on_piece, cancel_event, on_reasoning=on_reasoning,
                                   deadline_seconds=deadline_seconds, cancels=cancels)
        finally:
            for flag in cancels:
                flag.set()

    def _race_impl(self, payload, on_piece, cancel_event, *, on_reasoning=None,
                   deadline_seconds=None, cancels) -> str:
        """两个通道赛跑，**只用一个**。

        * 以第一段正文定胜负：先吐出正文的通道赢，另一个马上取消，它收到的东西一概不用；
        * 所有回调都在调用方自己的线程里执行（和单通道时一样），不会有两个线程同时往下游送字；
        * 胜负定了之后赢家中途出错就照常报错，**不换通道续说**（不然前半句会被说两遍）；
        * 还没出正文就出错的那个直接出局，等另一个；两个都失败才报错（报主通道的错）。
        """
        import queue as _queue

        backup = self.backup
        if cancel_event is not None and cancel_event.is_set():
            raise ApiError("已取消")
        events: _queue.Queue = _queue.Queue(maxsize=128)
        labels = ["主通道", f"第二通道（{backup.label}）"]
        started = [False, False]
        finished: dict[int, tuple[str | None, Exception | None]] = {}
        variants = [payload, {**payload, "model": backup.model} if backup.model else payload]
        if backup.model:
            for key in ("enable_thinking", "thinking_budget"):
                variants[1].pop(key, None)
            variants[1].update(thinking_fields(backup.model,
                want_thinking=bool(payload.get("enable_thinking", False)),
                budget=int(payload.get("thinking_budget", 0) or 0)))
        targets = [(self.endpoint, self.api_key), (backup.endpoint, backup.api_key)]

        def run(index: int) -> None:
            def publish(kind, value):
                while not cancels[index].is_set():
                    try:
                        events.put((kind, index, value), timeout=.05)
                        return
                    except _queue.Full:
                        continue

            def piece(text: str, _so_far: str) -> None:
                publish("piece", text)

            def reasoning(chars: int) -> None:
                publish("reason", chars)

            try:
                result = self._stream_once(targets[index][0], targets[index][1], variants[index], piece,
                                           cancels[index], on_reasoning=reasoning,
                                           deadline_seconds=deadline_seconds)
                publish("done", (result, None))
            except Exception as exc:  # noqa: BLE001
                publish("done", (None, exc))

        def start(index: int) -> None:
            started[index] = True
            threading.Thread(target=run, args=(index,), name=f"llm-channel-{index}", daemon=True).start()

        start(0)
        if backup.mode != "delayed":
            start(1)
        began = time.monotonic()
        winner: int | None = None
        parts: list[str] = []
        while True:
            if time.monotonic() - began >= (deadline_seconds or max(self.timeout, 60)):
                raise ApiError("两路大模型响应超时，请稍后重试")
            if cancel_event is not None and cancel_event.is_set():
                for flag in cancels:
                    flag.set()
                raise ApiError("已取消")
            if not started[1] and winner is None and (
                    time.monotonic() - began >= backup.delay or (0 in finished and finished[0][1] is not None)):
                start(1)
            try:
                kind, index, value = events.get(timeout=0.05)
            except _queue.Empty:
                continue
            if kind == "piece":
                if winner is None:
                    if not value.strip():
                        continue
                    winner = index
                    cancels[1 - index].set()          # 输家立刻停，它的内容一律丢弃
                    self.last_channel = labels[index]
                if index != winner:
                    continue
                parts.append(value)
                if on_piece is not None:
                    on_piece(value, "".join(parts))
            elif kind == "reason":
                if on_reasoning is not None and winner in (None, index):
                    on_reasoning(value)
            else:
                finished[index] = value
                result, error = value
                if winner == index:
                    if error is not None:
                        raise error          # 已经出过字了：不换通道续说
                    return result or ""
                if winner is not None:
                    continue                 # 输家收尾，不管它
                all_done = all(i in finished for i in (0, 1) if started[i])
                if error is None:
                    if result and result.strip():
                        self.last_channel = labels[index]   # 没有流式片段却有结果（极少见）
                        return result
                    error = ApiError("模型返回了空内容")
                    finished[index] = (result, error)
                if index == 0 and not started[1]:
                    start(1)                 # 主通道还没出字就失败：马上发第二通道
                    continue
                if all_done:
                    raise finished[0][1] if finished.get(0, (None, None))[1] is not None else error

    def _stream_once(
        self,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
        on_piece: Callable[[str, str], None],
        cancel_event: threading.Event | None,
        *,
        on_reasoning: Callable[[int], None] | None = None,
        deadline_seconds: float | None = None,
    ) -> str:
        """POST a streaming chat request and return the concatenated content.

        思考模型会先在 `reasoning_content` 里吐推理过程，`content` 为空。
        这部分不进译文，但通过 on_reasoning 报告已思考的字数，让界面能显示
        「思考中」而不是看起来卡死。
        """
        if cancel_event is not None and cancel_event.is_set():
            raise ApiError("已取消")
        if _USAGE_OPTION["enabled"] and "stream_options" not in payload:
            payload = {**payload, "stream_options": {"include_usage": True}}
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
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
                message = _extract_api_error(response)
                if "stream_options" in payload and response.status_code == 400 and "stream_options" in message:
                    _USAGE_OPTION["enabled"] = False
                    retry = {k: v for k, v in payload.items() if k != "stream_options"}
                    return self._stream_once(endpoint, api_key, retry, on_piece, cancel_event,
                                             on_reasoning=on_reasoning, deadline_seconds=deadline_seconds)
                raise ApiError(message)
            parts: list[str] = []
            usage = None
            completed = False
            reasoning_chars = 0
            budget = deadline_seconds if deadline_seconds else max(self.timeout, 60)
            deadline = time.monotonic() + budget
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
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                reason = choices[0].get("finish_reason")
                if reason in ("length", "content_filter"):
                    raise ApiError("译文被截断或过滤，未将其作为完整回答播放，请缩短后重试")
                if reason == "stop":
                    completed = True
                delta = choices[0].get("delta") or {}
                thought = delta.get("reasoning_content") or ""
                if thought:
                    reasoning_chars += len(thought)
                    if on_reasoning is not None:
                        on_reasoning(reasoning_chars)
                piece = delta.get("content") or ""
                if not piece:
                    continue
                parts.append(piece)
                if on_piece is not None:
                    on_piece(piece, "".join(parts))
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("已取消")
            if parts and not completed:
                raise ApiError("翻译连接提前结束，译文不完整，请重试")
        prompt_chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []) if isinstance(m, dict))
        record_llm_usage(str(payload.get("model", self.model)), usage,
                         prompt_chars=prompt_chars, output_chars=sum(len(p) for p in parts) + reasoning_chars)
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

    def complete(
        self,
        payload: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
        on_reasoning: Callable[[int], None] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        deadline_seconds: float = 600,
    ) -> str:
        """发一个现成的请求体（流式），返回完整正文。给论文导入 / 预习这类长任务用：
        流式读取不会因为模型长时间思考而读超时，总时限由 deadline_seconds 控制。"""
        body = dict(payload)
        body["stream"] = True
        body.setdefault("model", self.model)
        result = self._post_stream(
            body,
            on_delta or (lambda _piece, _so_far: None),
            cancel_event,
            on_reasoning=on_reasoning,
            deadline_seconds=deadline_seconds,
        ).strip()
        if not result:
            raise ApiError("模型返回了空内容")
        return result

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        thinking: bool = False,
        thinking_budget: int = 4096,
        on_reasoning: Callable[[int], None] | None = None,
    ) -> str:
        """Stream a free-form chat completion (used by the Q&A advisor).

        thinking=False（默认）：尽量不思考，答案来得快。
        thinking=True：允许思考，答得更准但更慢；预算封顶，超时放宽到 3 分钟。
        """
        chosen = model or self.model
        payload = {
            "model": chosen,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        payload.update(thinking_fields(chosen, want_thinking=thinking, budget=thinking_budget))
        result = self._post_stream(
            payload,
            on_delta,
            cancel_event,
            on_reasoning=on_reasoning,
            deadline_seconds=180 if thinking else None,
        ).strip()
        if not result:
            raise ApiError("模型返回了空内容")
        return result
