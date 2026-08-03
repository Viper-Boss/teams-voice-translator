from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import requests
import websocket

from .api_payloads import (
    build_asr_session_update,
    build_translation_payload,
    build_tts_payload,
    build_voice_clone_payload,
    parse_json_list,
)


class ApiError(RuntimeError):
    pass


def _extract_api_error(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("message") or body.get("error", {}).get("message") or str(body)
    except Exception:
        detail = response.text[:500]
    return f"HTTP {response.status_code}: {detail}"


class BailianClient:
    def __init__(self, api_key: str, workspace_id: str, *, timeout: int = 45, proxy: str = "") -> None:
        if not api_key.strip():
            raise ApiError("尚未设置百炼 API Key")
        if not workspace_id.strip():
            raise ApiError("尚未设置百炼 Workspace ID")
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id.strip()
        self.timeout = timeout
        self.proxies = {"http": proxy, "https": proxy} if proxy.strip() else None

    @property
    def root(self) -> str:
        return f"https://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"

    def translate(self, text: str, settings: dict[str, Any]) -> str:
        return self.translate_between(
            text,
            settings,
            source_language=settings["source_language"],
            target_language=settings["target_language"],
        )

    def translate_between(
        self,
        text: str,
        settings: dict[str, Any],
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        try:
            terms = parse_json_list(settings.get("translation_terms", ""), "术语表")
            memories = parse_json_list(settings.get("translation_memories", ""), "翻译记忆")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ApiError(str(exc)) from exc
        if source_language.lower().startswith("english") and target_language.lower().startswith("chinese"):
            terms = [{"source": item["target"], "target": item["source"]} for item in terms]
            memories = [{"source": item["target"], "target": item["source"]} for item in memories]
        style_prompts = {
            "polite": "Natural, polite classroom conversation",
            "concise": "Concise everyday spoken language",
            "academic": "Formal academic discussion",
            "literal": "Faithful translation with minimal paraphrasing",
        }
        domain = settings.get("translation_domain", "").strip()
        style = style_prompts.get(settings.get("translation_style", ""), "")
        if style:
            domain = f"{domain}; style: {style}" if domain else f"Style: {style}"
        payload = build_translation_payload(
            text,
            model=settings["translation_model"],
            source_lang=source_language,
            target_lang=target_language,
            terms=terms,
            memories=memories,
            domain=domain,
        )
        response = requests.post(
            f"{self.root}/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
            proxies=self.proxies,
        )
        if not response.ok:
            raise ApiError(_extract_api_error(response))
        try:
            result = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ApiError(f"翻译接口返回格式异常：{response.text[:500]}") from exc
        if not result:
            raise ApiError("翻译接口返回了空文本")
        return result

    def stream_tts(
        self,
        text: str,
        settings: dict[str, Any],
        on_audio: Callable[[bytes], None],
        cancel_event: threading.Event,
    ) -> int:
        payload = build_tts_payload(
            text,
            model=settings["tts_model"],
            voice=settings["voice"],
            sample_rate=settings["tts_sample_rate"],
            volume=settings["tts_volume"],
            rate=settings["tts_rate"],
            pitch=settings["tts_pitch"],
            seed=settings["tts_seed"],
            language_hint=settings["tts_language_hint"],
            instruction=settings.get("tts_instruction", ""),
            emotion_tag=settings.get("tts_emotion_tag", ""),
            enable_aigc_tag=settings.get("enable_aigc_tag", False),
            aigc_propagator=settings.get("aigc_propagator", ""),
            aigc_propagate_id=settings.get("aigc_propagate_id", ""),
        )
        response = requests.post(
            f"{self.root}/api/v1/services/audio/tts/SpeechSynthesizer",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "enable",
            },
            json=payload,
            stream=True,
            timeout=(10, self.timeout),
            proxies=self.proxies,
        )
        if not response.ok:
            detail = _extract_api_error(response)
            if "Engine return error code: 431" in detail and settings["voice"].strip():
                detail = self._explain_cloned_voice_error(
                    detail,
                    voice_id=settings["voice"].strip(),
                    model=settings["tts_model"],
                )
            raise ApiError(detail)
        total = 0
        for raw_line in response.iter_lines(decode_unicode=True):
            if cancel_event.is_set():
                response.close()
                break
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("code"):
                raise ApiError(f"{event.get('code')}: {event.get('message', '')}")
            audio = (event.get("output") or {}).get("audio") or {}
            encoded = audio.get("data") or ""
            if encoded:
                chunk = base64.b64decode(encoded)
                total += len(chunk)
                on_audio(chunk)
        return total

    def complete(self, prompt: str, *, model: str = "qwen-plus") -> str:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You turn bilingual online-class transcripts into accurate Chinese meeting notes. Do not invent facts.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post(
            f"{self.root}/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=max(self.timeout, 90),
            proxies=self.proxies,
        )
        if not response.ok:
            raise ApiError(_extract_api_error(response))
        try:
            result = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ApiError(f"会议总结接口返回格式异常：{response.text[:500]}") from exc
        if not result:
            raise ApiError("会议总结接口返回了空文本")
        return result

    def clone_voice(
        self,
        *,
        target_model: str,
        prefix: str,
        audio_url: str,
        language: str,
        max_seconds: float,
        preprocess: bool,
    ) -> str:
        if not prefix.isalnum() or len(prefix) > 10:
            raise ApiError("音色前缀只能含字母和数字，且最长 10 个字符")
        payload = build_voice_clone_payload(
            target_model=target_model,
            prefix=prefix,
            audio_url=audio_url,
            language=language,
            max_seconds=max_seconds,
            preprocess=preprocess,
        )
        response = requests.post(
            f"{self.root}/api/v1/services/audio/tts/customization",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
            proxies=self.proxies,
        )
        if not response.ok:
            raise ApiError(_extract_api_error(response))
        result = response.json()
        voice_id = (result.get("output") or {}).get("voice_id")
        if not voice_id:
            raise ApiError(f"创建音色失败：{result}")
        return voice_id

    def query_voice(self, voice_id: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/api/v1/services/audio/tts/customization",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": "voice-enrollment",
                "input": {"action": "query_voice", "voice_id": voice_id.strip()},
            },
            timeout=self.timeout,
            proxies=self.proxies,
        )
        if not response.ok:
            raise ApiError(_extract_api_error(response))
        output = response.json().get("output") or {}
        if not isinstance(output, dict):
            raise ApiError("查询音色状态时，百炼返回了异常数据")
        return output

    def wait_for_voice_ready(
        self,
        voice_id: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            detail = self.query_voice(voice_id)
            status = str(detail.get("status", "")).upper()
            if status == "OK":
                return detail
            if status == "UNDEPLOYED":
                raise ApiError(
                    "百炼未通过该复刻音色的处理/审核（UNDEPLOYED）。"
                    "请换一段 10–20 秒、无背景声且只有你本人说话的清晰录音后重新创建。"
                )
            if on_status is not None:
                on_status("音色已提交 · 正在等待百炼处理完成（DEPLOYING）…")
            if time.monotonic() >= deadline:
                raise ApiError(
                    "音色仍在百炼处理中（DEPLOYING），暂时不能合成。"
                    "为避免处理失败，OSS 临时样音会先保留；稍后可重新查询或创建。"
                )
            time.sleep(poll_interval)

    def _explain_cloned_voice_error(self, original: str, *, voice_id: str, model: str) -> str:
        if not voice_id.startswith(("qwen-audio-", "cosyvoice-")):
            return original
        try:
            detail = self.query_voice(voice_id)
        except Exception:
            return original
        status = str(detail.get("status", "")).upper()
        target_model = str(detail.get("target_model", "")).strip()
        if status == "DEPLOYING":
            return (
                "复刻音色仍在百炼处理中（DEPLOYING），目前不能发声。"
                "请稍等一会再试；新版本创建音色时会自动等到可用。"
            )
        if status == "UNDEPLOYED":
            return (
                "复刻音色处理失败或未通过审核（UNDEPLOYED），所以语音合成返回 431。"
                "请使用 10–20 秒、安静环境、只有你本人连续说话的样音重新创建。"
            )
        if target_model and target_model != model:
            return (
                f"复刻音色绑定的是 {target_model}，当前语音合成模型却是 {model}。"
                "两者必须完全一致，请切换模型或重新创建对应音色。"
            )
        if status == "OK":
            return (
                "该复刻音色状态为 OK，但百炼合成引擎仍返回 431。"
                "请先把音色切换为系统音色 loongjohn 验证；若系统音色可用，"
                "请用新版重新创建一次克隆音色。"
            )
        return original


class QwenRealtimeASR:
    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str,
        language: str,
        vad_threshold: float,
        vad_silence_ms: int,
        on_preview: Callable[[str, str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_segment: Callable[[str, str], None] | None = None,
        combine_previews: bool = True,
    ) -> None:
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.model = model
        self.language = language
        self.vad_threshold = vad_threshold
        self.vad_silence_ms = vad_silence_ms
        self.on_preview = on_preview
        self.on_status = on_status
        self.on_error = on_error
        self.on_segment = on_segment
        self.combine_previews = combine_previews
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.opened = threading.Event()
        self.error_text = ""
        self.final_segments: list[str] = []
        self._completed_items: set[str] = set()
        self._latest_preview = ""
        self.ws: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        url = (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            f"/api-ws/v1/realtime?model={self.model}"
        )
        self.ws = websocket.WebSocketApp(
            url,
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Beta: realtime=v1",
            ],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(
            target=lambda: self.ws.run_forever(ping_interval=20, ping_timeout=10),
            name="qwen-asr-websocket",
            daemon=True,
        )
        self.thread.start()

    def wait_ready(self, timeout: float = 12.0) -> None:
        if not self.ready.wait(timeout):
            if self.error_text:
                raise ApiError(self.error_text)
            raise ApiError("实时识别连接超时，请检查 Workspace ID、API Key 和网络")
        if self.error_text:
            raise ApiError(self.error_text)

    def send_audio(self, pcm: bytes) -> None:
        if not self.ws or not self.opened.is_set():
            raise ApiError("实时识别尚未连接")
        event = {
            "event_id": f"event_{uuid.uuid4().hex}",
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
        self.ws.send(json.dumps(event))

    def finish(self, timeout: float = 15.0) -> str:
        if self.ws and self.opened.is_set():
            self.ws.send(
                json.dumps(
                    {"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"}
                )
            )
            self.finished.wait(timeout)
        self.close()
        combined = "".join(segment.strip() for segment in self.final_segments if segment.strip()).strip()
        return combined or self._latest_preview.strip()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)

    def _on_open(self, ws) -> None:
        self.opened.set()
        self.on_status("实时识别已连接")
        event = build_asr_session_update(
            language=self.language,
            threshold=self.vad_threshold,
            silence_ms=self.vad_silence_ms,
        )
        event["event_id"] = f"event_{uuid.uuid4().hex}"
        ws.send(json.dumps(event))

    def _on_message(self, _ws, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        kind = event.get("type", "")
        if kind == "session.updated":
            self.ready.set()
        elif kind == "conversation.item.input_audio_transcription.text":
            preview = f"{event.get('text', '')}{event.get('stash', '')}".strip()
            if preview:
                self._latest_preview = preview
                self.on_preview(preview, event.get("emotion", "neutral"))
        elif kind == "conversation.item.input_audio_transcription.completed":
            item_id = event.get("item_id", str(len(self.final_segments)))
            transcript = (event.get("transcript") or "").strip()
            if transcript and item_id not in self._completed_items:
                self._completed_items.add(item_id)
                self.final_segments.append(transcript)
                self._latest_preview = transcript
                preview = "".join(self.final_segments) if self.combine_previews else transcript
                self.on_preview(preview, event.get("emotion", "neutral"))
                if self.on_segment is not None:
                    self.on_segment(transcript, event.get("emotion", "neutral"))
        elif kind == "session.finished":
            self.finished.set()
        elif kind == "error":
            error = event.get("error") or {}
            self.error_text = error.get("message") or str(error)
            self.on_error(self.error_text)
            self.ready.set()
            self.finished.set()

    def _on_error(self, _ws, error: Any) -> None:
        self.error_text = str(error)
        self.on_error(self.error_text)
        self.ready.set()
        self.finished.set()

    def _on_close(self, _ws, _status_code, _message) -> None:
        self.opened.clear()
        self.finished.set()
