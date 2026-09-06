from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import requests
import websocket

from .audio import decode_audio_to_pcm16_mono
from .longform import (
    build_long_form_context_payload,
    build_long_form_translation_payload,
    parse_long_form_translation,
)
from .api_payloads import (
    build_asr_session_update,
    build_fun_asr_finish_task,
    build_fun_asr_run_task,
    build_qwen3_tts_http_payload,
    build_qwen3_tts_realtime_session_update,
    build_qwen_voice_clone_payload,
    build_translation_payload,
    build_tts_payload,
    build_voice_clone_payload,
    is_cosyvoice_model,
    is_duplex_asr_model,
    is_qwen3_tts_vc_model,
    is_qwen3_tts_vc_realtime_model,
    parse_json_list,
)


class ApiError(RuntimeError):
    pass


def _websocket_proxy_options(proxy: str) -> dict[str, Any]:
    """Parse a ``host:port`` or ``http://host:port`` proxy string into
    websocket-client run_forever keyword arguments.
    """
    options: dict[str, Any] = {}
    if not proxy.strip():
        return options
    parsed = urlparse(proxy if "://" in proxy else f"http://{proxy}")
    if parsed.hostname:
        options["http_proxy_host"] = parsed.hostname
        options["http_proxy_port"] = parsed.port or 80
        options["proxy_type"] = "http"
        if parsed.username:
            options["http_proxy_auth"] = (parsed.username, parsed.password or "")
    return options


def _extract_api_error(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("message") or body.get("error", {}).get("message") or str(body)
    except Exception:
        detail = response.text[:500]
    return f"HTTP {response.status_code}: {detail}"


class BailianClient:
    def __init__(
        self,
        api_key: str,
        workspace_id: str,
        *,
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
        self.timeout = timeout
        proxy = proxy.strip()
        if proxy_mode == "direct":
            # 国内直连：显式置空，禁止 requests 使用系统/VPN 代理，避免流量绕到海外。
            self.proxies = {"http": None, "https": None}
            self.proxy = ""
        elif proxy:
            self.proxies = {"http": proxy, "https": proxy}
            self.proxy = proxy
        else:
            self.proxies = None
            self.proxy = ""

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

    def translate_long_form(
        self,
        full_text: str,
        sentences: list[str],
        settings: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Translate an aligned script in one context-aware model request.

        Unlike ``translate_between``, the model sees the complete document and
        every numbered sentence at once.  Course terminology, translation
        memory, domain and style are part of the same request so references and
        specialist wording remain consistent from beginning to end.
        """
        try:
            terms = parse_json_list(settings.get("translation_terms", ""), "术语表")
            memories = parse_json_list(settings.get("translation_memories", ""), "翻译记忆")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ApiError(str(exc)) from exc
        cancel_event = cancel_event or threading.Event()

        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise ApiError("已取消整段翻译准备")

        def post_content(payload: dict[str, Any]) -> str:
            check_cancelled()
            response = requests.post(
                f"{self.root}/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=max(self.timeout, 90),
                proxies=self.proxies,
            )
            check_cancelled()
            if not response.ok:
                raise ApiError(_extract_api_error(response))
            try:
                return str(response.json()["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ApiError(f"整段语境翻译返回格式异常：{exc}") from exc

        # Short and medium scripts keep the single-request path. Very long
        # scripts first become one compact context brief, then translate in
        # aligned batches so the same source text is not duplicated until the
        # model context overflows.
        if len(full_text) <= 6000 and len(sentences) <= 24:
            if on_progress is not None:
                on_progress("正在通读全文并统一术语…")
            payload = build_long_form_translation_payload(
                full_text,
                sentences,
                settings,
                terms=terms,
                memories=memories,
            )
            try:
                return parse_long_form_translation(post_content(payload), len(sentences))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ApiError(f"整段语境翻译返回格式异常：{exc}") from exc

        if on_progress is not None:
            on_progress("长稿模式：正在提取全文语境、人物与术语…")
        context_payload = build_long_form_context_payload(
            full_text,
            settings,
            terms=terms,
            memories=memories,
        )
        context_brief = post_content(context_payload).strip()
        if not context_brief:
            raise ApiError("长稿语境分析返回了空内容")

        batch_size = 18
        translated: list[str] = []
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        for batch_number, start in enumerate(range(0, len(sentences), batch_size), start=1):
            check_cancelled()
            batch = sentences[start : start + batch_size]
            if on_progress is not None:
                on_progress(f"长稿模式：正在翻译第 {batch_number}/{total_batches} 批…")
            payload = build_long_form_translation_payload(
                context_brief,
                batch,
                settings,
                terms=terms,
                memories=memories,
            )
            try:
                translated.extend(parse_long_form_translation(post_content(payload), len(batch)))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ApiError(
                    f"长稿第 {batch_number}/{total_batches} 批返回格式异常：{exc}"
                ) from exc
        return translated

    def stream_tts(
        self,
        text: str,
        settings: dict[str, Any],
        on_audio: Callable[[bytes], None],
        cancel_event: threading.Event,
    ) -> int:
        model = settings["tts_model"]
        if is_qwen3_tts_vc_realtime_model(model):
            return self._stream_qwen3_tts_realtime(text, settings, on_audio, cancel_event)
        if is_qwen3_tts_vc_model(model):
            return self._stream_qwen3_tts_http(text, settings, on_audio, cancel_event)
        return self._stream_legacy_tts(text, settings, on_audio, cancel_event)

    def _stream_legacy_tts(
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
            emotion_tag=(
                ""
                if is_cosyvoice_model(settings["tts_model"])
                else settings.get("tts_emotion_tag", "")
            ),
            enable_aigc_tag=settings.get("enable_aigc_tag", False),
            aigc_propagator=settings.get("aigc_propagator", ""),
            aigc_propagate_id=settings.get("aigc_propagate_id", ""),
            enable_ssml=settings.get("tts_enable_ssml", False),
        )
        if settings["tts_model"].startswith("cosyvoice-v3.5"):
            for key in ("enable_aigc_tag", "aigc_propagator", "aigc_propagate_id"):
                payload["input"].pop(key, None)
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
        try:
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
                    chunk = base64.b64decode(encoded, validate=True)
                    total += len(chunk)
                    on_audio(chunk)
            return total
        finally:
            response.close()

    def _stream_qwen3_tts_http(
        self,
        text: str,
        settings: dict[str, Any],
        on_audio: Callable[[bytes], None],
        cancel_event: threading.Event,
    ) -> int:
        if not settings["voice"].strip():
            raise ApiError("Qwen3-TTS-VC 需要先创建并填写与当前模型绑定的克隆音色 voice")
        payload = build_qwen3_tts_http_payload(
            text,
            model=settings["tts_model"],
            voice=settings["voice"],
            language=settings.get("tts_language_hint", "en"),
        )
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "enable",
                "X-DashScope-WorkSpace": self.workspace_id,
            },
            json=payload,
            stream=True,
            timeout=(10, max(self.timeout, 90)),
            proxies=self.proxies,
        )
        try:
            if not response.ok:
                raise ApiError(_extract_api_error(response))
            total = 0
            saw_stream_audio = False
            deferred_audio_url = ""
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
                audio_meta = (event.get("output") or {}).get("audio") or {}
                encoded = audio_meta.get("data") or ""
                if encoded:
                    saw_stream_audio = True
                    chunk = base64.b64decode(encoded, validate=True)
                    pcm = decode_audio_to_pcm16_mono(chunk, sample_rate=24000)
                    if pcm:
                        total += len(pcm)
                        on_audio(pcm)
                audio_url = audio_meta.get("url")
                if audio_url and not saw_stream_audio:
                    deferred_audio_url = str(audio_url)
            # Some HTTP responses provide a complete-file URL instead of SSE audio
            # chunks. Use it only as a fallback; downloading it after streamed data
            # would replay the whole utterance a second time.
            if not saw_stream_audio and deferred_audio_url and not cancel_event.is_set():
                url_response = None
                try:
                    url_response = requests.get(
                        deferred_audio_url,
                        timeout=(10, self.timeout),
                        proxies=self.proxies,
                    )
                    url_response.raise_for_status()
                    if cancel_event.is_set():
                        return total
                    pcm = decode_audio_to_pcm16_mono(url_response.content, sample_rate=24000)
                    if pcm:
                        total += len(pcm)
                        on_audio(pcm)
                except requests.RequestException as exc:
                    raise ApiError(f"下载合成音频失败：{exc}") from exc
                finally:
                    if url_response is not None:
                        url_response.close()
            return total
        finally:
            response.close()

    def _stream_qwen3_tts_realtime(
        self,
        text: str,
        settings: dict[str, Any],
        on_audio: Callable[[bytes], None],
        cancel_event: threading.Event,
    ) -> int:
        voice = settings["voice"].strip()
        if not voice:
            raise ApiError("Qwen3-TTS-VC Realtime 需要先创建并填写与当前模型绑定的克隆音色 voice")
        model = settings["tts_model"]
        url = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}"
        connection_options: dict[str, Any] = {
            "header": [
                f"Authorization: Bearer {self.api_key}",
                f"X-DashScope-WorkSpace: {self.workspace_id}",
                "User-Agent: teams-voice-translator/1.3",
            ],
            "timeout": min(15, self.timeout),
        }
        connection_options.update(_websocket_proxy_options(self.proxy))
        ws = None
        total = 0
        finish_sent = False
        deadline = time.monotonic() + max(self.timeout, 90)
        try:
            ws = websocket.create_connection(url, **connection_options)
            ws.settimeout(1.0)

            def send(event: dict[str, Any]) -> None:
                assert ws is not None
                ws.send(json.dumps(event, ensure_ascii=False))

            def receive() -> dict[str, Any] | None:
                assert ws is not None
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    return None
                if not message:
                    return None
                if isinstance(message, bytes):
                    message = message.decode("utf-8", errors="replace")
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    return None
                if event.get("type") == "error":
                    detail = event.get("error") or {}
                    raise ApiError(
                        f"{detail.get('code', 'Qwen3-TTS WebSocket error')}: "
                        f"{detail.get('message', event)}"
                    )
                return event

            while time.monotonic() < deadline:
                event = receive()
                if event and event.get("type") == "session.created":
                    break
            else:
                raise ApiError("Qwen3-TTS WebSocket 建立会话超时")

            send(
                build_qwen3_tts_realtime_session_update(
                    voice=voice,
                    language=settings.get("tts_language_hint", "en"),
                    sample_rate=settings.get("tts_sample_rate", 24000),
                    volume=settings.get("tts_volume", 50),
                    rate=settings.get("tts_rate", 1.0),
                    pitch=settings.get("tts_pitch", 1.0),
                )
            )
            while time.monotonic() < deadline:
                event = receive()
                if event and event.get("type") == "session.updated":
                    break
            else:
                raise ApiError("Qwen3-TTS WebSocket 更新会话超时")

            send(
                {
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "type": "input_text_buffer.append",
                    "text": text,
                }
            )
            send(
                {
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "type": "input_text_buffer.commit",
                }
            )
            while time.monotonic() < deadline:
                if cancel_event.is_set() and not finish_sent:
                    send({"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"})
                    finish_sent = True
                event = receive()
                if event is None:
                    continue
                event_type = event.get("type")
                if event_type == "response.audio.delta" and not cancel_event.is_set():
                    encoded = event.get("delta") or ""
                    if encoded:
                        chunk = base64.b64decode(encoded)
                        total += len(chunk)
                        on_audio(chunk)
                elif event_type == "response.done":
                    response = event.get("response") or {}
                    if response.get("status") == "failed":
                        detail = response.get("status_details") or {}
                        raise ApiError(f"Qwen3-TTS 合成失败：{detail}")
                    if not finish_sent:
                        send({"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"})
                        finish_sent = True
                elif event_type == "session.finished":
                    break
            else:
                raise ApiError("Qwen3-TTS WebSocket 等待音频超时")
        except websocket.WebSocketBadStatusException as exc:
            raise ApiError(f"Qwen3-TTS WebSocket 鉴权失败：{exc}") from exc
        except websocket.WebSocketException as exc:
            raise ApiError(f"Qwen3-TTS WebSocket 连接失败：{exc}") from exc
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
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
        volume_normalization: bool = False,
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
            volume_normalization=volume_normalization,
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

    def clone_qwen_voice(
        self,
        *,
        target_model: str,
        preferred_name: str,
        audio_data: str,
        language: str,
        transcript: str = "",
    ) -> tuple[str, str]:
        if not is_qwen3_tts_vc_model(target_model):
            raise ApiError("Qwen 音色复刻目标模型无效")
        if not preferred_name or len(preferred_name) > 16 or not all(
            char.isalnum() or char == "_" for char in preferred_name
        ):
            raise ApiError("Qwen3 音色名称只能含字母、数字和下划线，且最长 16 个字符")
        payload = build_qwen_voice_clone_payload(
            target_model=target_model,
            preferred_name=preferred_name,
            audio_data=audio_data,
            language=language,
            transcript=transcript,
        )
        response = requests.post(
            f"{self.root}/api/v1/services/audio/tts/customization",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=max(self.timeout, 90),
            proxies=self.proxies,
        )
        if not response.ok:
            detail = _extract_api_error(response)
            if response.status_code == 403:
                detail += (
                    "。请在百炼 API Key 的模型权限中授权 qwen-voice-enrollment，"
                    f"并授权目标模型 {target_model}。"
                )
            raise ApiError(detail)
        result = response.json()
        output = result.get("output") or {}
        voice = str(output.get("voice") or "").strip()
        if not voice:
            raise ApiError(f"创建 Qwen3 音色失败：{result}")
        returned_model = str(output.get("target_model") or "").strip()
        if returned_model and returned_model != target_model:
            raise ApiError(
                f"百炼返回的音色绑定模型为 {returned_model}，与请求的 {target_model} 不一致"
            )
        warning = ""
        if output.get("fallback_mode"):
            reason = output.get("fallback_reason") or "样音质量不足"
            warning = (
                f"百炼以 fallback 模式创建了音色（{reason}），相似度可能降低。"
                "建议换用 10–20 秒安静、连续、无混响的单人录音重新创建。"
            )
        return voice, warning

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
        proxy: str = "",
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
        self.proxy = proxy.strip()
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
        ws = websocket.WebSocketApp(
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
        self.ws = ws
        run_options = {"ping_interval": 20, "ping_timeout": 10}
        run_options.update(_websocket_proxy_options(self.proxy))
        self.thread = threading.Thread(
            target=lambda: ws.run_forever(**run_options),
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


class FunASRRealtime:
    """Fun-ASR realtime recognition over the DashScope duplex WebSocket protocol.

    Same public interface as QwenRealtimeASR so the UI workers can swap
    engines transparently. Audio is sent as binary frames (mono PCM16);
    results arrive as result-generated events with sentence_end flags.
    """

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
        context_terms: list[str] | None = None,
        vocabulary_id: str = "",
        proxy: str = "",
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
        self.context_terms = list(context_terms or [])
        self.vocabulary_id = vocabulary_id.strip()
        self.proxy = proxy.strip()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.opened = threading.Event()
        self.error_text = ""
        self.final_segments: list[str] = []
        self._latest_preview = ""
        self.task_id = ""
        self.ws: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None
        self._connection_lock = threading.Lock()
        self._closing = False

    def start(self) -> None:
        self._closing = False
        self._connect()

    def _connect(self) -> None:
        self.ready.clear()
        self.finished.clear()
        self.opened.clear()
        self.error_text = ""
        self.task_id = ""
        url = (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/inference"
        )
        self.ws = websocket.WebSocketApp(
            url,
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        run_options = {"ping_interval": 20, "ping_timeout": 10}
        run_options.update(_websocket_proxy_options(self.proxy))
        self.thread = threading.Thread(
            target=lambda: self.ws.run_forever(**run_options),
            name="funasr-websocket",
            daemon=True,
        )
        self.thread.start()

    def wait_ready(self, timeout: float = 12.0) -> None:
        if not self.ready.wait(timeout):
            if self.error_text:
                raise ApiError(self.error_text)
            raise ApiError("Fun-ASR 实时识别连接超时，请检查 Workspace ID、API Key 和网络")
        if self.error_text:
            raise ApiError(self.error_text)

    def send_audio(self, pcm: bytes) -> None:
        if not self.opened.is_set():
            self._reconnect()
        if not self.ws or not self.opened.is_set():
            raise ApiError("实时识别尚未连接")
        try:
            self.ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            self.opened.clear()
            self._reconnect()
            if not self.ws or not self.opened.is_set():
                raise ApiError("实时识别重连失败")
            self.ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)

    def _reconnect(self) -> None:
        if self._closing:
            raise ApiError("实时识别已经关闭")
        with self._connection_lock:
            if self.opened.is_set():
                return
            old_ws = self.ws
            if old_ws is not None:
                try:
                    old_ws.close()
                except Exception:
                    pass
            self.on_status("实时识别连接已断开，正在自动重连…")
            self._connect()
            self.wait_ready(timeout=12.0)
            self.on_status("实时识别已重新连接")

    def finish(self, timeout: float = 15.0) -> str:
        if self.ws and self.opened.is_set() and self.task_id:
            self.ws.send(json.dumps(build_fun_asr_finish_task(self.task_id), ensure_ascii=False))
            self.finished.wait(timeout)
        self.close()
        combined = "".join(segment.strip() for segment in self.final_segments if segment.strip()).strip()
        return combined or self._latest_preview.strip()

    def close(self) -> None:
        self._closing = True
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)

    def _on_open(self, ws) -> None:
        if ws is not self.ws:
            return
        self.opened.set()
        self.on_status("实时识别已连接")
        event = build_fun_asr_run_task(
            model=self.model,
            context_terms=self.context_terms,
            vocabulary_id=self.vocabulary_id,
            language=self.language,
            heartbeat=True,
        )
        self.task_id = event["header"]["task_id"]
        ws.send(json.dumps(event, ensure_ascii=False))

    def _on_message(self, _ws, message: str | bytes) -> None:
        if _ws is not self.ws:
            return
        if isinstance(message, bytes):
            return
        try:
            event = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        header = event.get("header") or {}
        kind = header.get("event", "")
        if kind == "task-started":
            self.ready.set()
        elif kind == "result-generated":
            sentence = ((event.get("payload") or {}).get("output") or {}).get("sentence") or {}
            if sentence.get("heartbeat"):
                return
            text = str(sentence.get("text") or "").strip()
            if not text:
                return
            self._latest_preview = text
            if sentence.get("sentence_end"):
                self.final_segments.append(text)
                preview = "".join(self.final_segments) if self.combine_previews else text
                self.on_preview(preview, "neutral")
                if self.on_segment is not None:
                    self.on_segment(text, "neutral")
            else:
                preview = (
                    "".join(self.final_segments) + text if self.combine_previews else text
                )
                self.on_preview(preview, "neutral")
        elif kind == "task-finished":
            self.finished.set()
        elif kind == "task-failed":
            self.error_text = header.get("error_message") or header.get("error_code") or str(event)
            self.on_error(self.error_text)
            self.ready.set()
            self.finished.set()

    def _on_error(self, _ws, error: Any) -> None:
        if _ws is not self.ws or self._closing:
            return
        self.error_text = str(error)
        # Keep the session recoverable: wait_ready() still raises the original
        # error for a failed initial connection, while an established session
        # will reconnect when the next audio chunk arrives.
        if self.ready.is_set():
            self.on_status(f"实时识别连接异常：{self.error_text}；下一段音频将自动重连")
        else:
            self.on_error(self.error_text)
        self.ready.set()
        self.finished.set()

    def _on_close(self, _ws, _status_code, _message) -> None:
        if _ws is not self.ws:
            return
        self.opened.clear()
        self.finished.set()
        if not self._closing:
            self.on_status("实时识别连接已断开；收到下一段音频时会自动重连")


def _fun_asr_context_terms(values: dict[str, Any], language: str) -> list[str]:
    """Reuse the course glossary as Fun-ASR dynamic recognition context."""
    try:
        terms = parse_json_list(str(values.get("translation_terms", "")), "术语表")
    except (ValueError, json.JSONDecodeError):
        return []
    words: list[str] = []
    for item in terms:
        word = item["target"] if language.strip().lower().startswith("en") else item["source"]
        word = word.strip()
        if word:
            words.append(word)
    return words[:50]


def create_realtime_asr(
    *,
    api_key: str,
    workspace_id: str,
    values: dict[str, Any],
    language: str,
    on_preview: Callable[[str, str], None],
    on_status: Callable[[str], None],
    on_error: Callable[[str], None],
    on_segment: Callable[[str, str], None] | None = None,
    combine_previews: bool = True,
    proxy: str = "",
) -> QwenRealtimeASR | FunASRRealtime:
    """Pick the realtime ASR engine from settings and build the client."""
    model = str(values["asr_model"])
    if is_duplex_asr_model(model):
        return FunASRRealtime(
            api_key=api_key,
            workspace_id=workspace_id,
            model=model,
            language=language,
            vad_threshold=float(values["vad_threshold"]),
            vad_silence_ms=int(values["vad_silence_ms"]),
            on_preview=on_preview,
            on_status=on_status,
            on_error=on_error,
            on_segment=on_segment,
            combine_previews=combine_previews,
            context_terms=_fun_asr_context_terms(values, language),
            proxy=proxy,
        )
    return QwenRealtimeASR(
        api_key=api_key,
        workspace_id=workspace_id,
        model=model,
        language=language,
        vad_threshold=float(values["vad_threshold"]),
        vad_silence_ms=int(values["vad_silence_ms"]),
        on_preview=on_preview,
        on_status=on_status,
        on_error=on_error,
        on_segment=on_segment,
        combine_previews=combine_previews,
        proxy=proxy,
    )
