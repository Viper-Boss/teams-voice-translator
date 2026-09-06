from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import requests

from .audio import decode_audio_to_pcm16_mono


class VoiceStudioError(RuntimeError):
    """A human-readable failure from the local VoiceStudio service."""


@dataclass(frozen=True)
class VoiceStudioVoice:
    voice_id: str
    name: str
    kind: str = ""
    language: str = ""
    engine: str = ""


@dataclass(frozen=True)
class VoiceStudioEngine:
    engine_id: str
    name: str
    installed: bool | None = None
    available: bool | None = None


def normalize_voicestudio_url(value: str) -> str:
    """Return the server root without a trailing /v1 path."""
    raw = str(value or "").strip() or "http://127.0.0.1:3900"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VoiceStudioError("VoiceStudio 地址无效，应类似 http://127.0.0.1:3900")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


def _items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        result: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, dict):
                copy = dict(item)
                copy.setdefault("id", key)
                result.append(copy)
        return result
    return []


def parse_voice_catalog(payload: Any) -> tuple[list[VoiceStudioVoice], list[VoiceStudioEngine]]:
    """Normalize the intentionally extensible VoiceStudio discovery response."""
    data = payload if isinstance(payload, dict) else {}
    raw_voices = data.get("voices", data.get("data", []))
    voices: list[VoiceStudioVoice] = []
    for item in _items(raw_voices):
        voice_id = str(
            item.get("voice_id")
            or item.get("profile_id")
            or item.get("id")
            or item.get("value")
            or ""
        ).strip()
        if not voice_id:
            continue
        voices.append(
            VoiceStudioVoice(
                voice_id=voice_id,
                name=str(item.get("name") or item.get("display_name") or item.get("label") or voice_id),
                kind=str(item.get("type") or item.get("kind") or ""),
                language=str(item.get("language") or item.get("lang") or ""),
                engine=str(item.get("engine") or item.get("engine_id") or item.get("model") or ""),
            )
        )

    engines: list[VoiceStudioEngine] = []
    for item in _items(data.get("engines", [])):
        engine_id = str(item.get("id") or item.get("engine_id") or item.get("model") or "").strip()
        if not engine_id:
            continue
        engines.append(
            VoiceStudioEngine(
                engine_id=engine_id,
                name=str(item.get("name") or item.get("display_name") or item.get("label") or engine_id),
                installed=item.get("installed") if isinstance(item.get("installed"), bool) else None,
                available=item.get("available") if isinstance(item.get("available"), bool) else None,
            )
        )
    return voices, engines


class VoiceStudioClient:
    """Small client for VoiceStudio's local OpenAI-compatible speech API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:3900",
        api_key: str = "local",
        *,
        timeout: float = 60.0,
    ) -> None:
        self.root = normalize_voicestudio_url(base_url)
        self.api_root = f"{self.root}/v1"
        self.api_key = str(api_key or "local").strip() or "local"
        self.timeout = max(5.0, float(timeout))
        self.session = requests.Session()
        # A loopback/Tailscale speech server should not be sent through a browser proxy.
        self.session.trust_env = False

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _error(response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("message") or payload.get("error")
                if isinstance(detail, dict):
                    detail = detail.get("message") or detail.get("detail") or str(detail)
                if detail:
                    return str(detail)
        except (ValueError, requests.RequestException):
            pass
        text = response.text.strip()
        return text[:600] if text else response.reason

    def health(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.root}/health",
                headers=self.headers,
                timeout=min(self.timeout, 12.0),
            )
        except requests.RequestException as exc:
            raise VoiceStudioError(
                "没有连接到 VoiceStudio。请先启动 VoiceStudio，并确认本地 API 端口为 3900。"
            ) from exc
        if not response.ok:
            raise VoiceStudioError(f"VoiceStudio 健康检查失败（HTTP {response.status_code}）：{self._error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise VoiceStudioError("VoiceStudio 健康检查没有返回 JSON") from exc
        return data if isinstance(data, dict) else {"status": str(data)}

    def catalog(self) -> tuple[list[VoiceStudioVoice], list[VoiceStudioEngine]]:
        try:
            response = self.session.get(
                f"{self.api_root}/audio/voices",
                headers=self.headers,
                timeout=min(self.timeout, 20.0),
            )
        except requests.RequestException as exc:
            raise VoiceStudioError("读取 VoiceStudio 本地声音库失败，请确认程序正在运行。") from exc
        if not response.ok:
            raise VoiceStudioError(f"读取 VoiceStudio 声音库失败（HTTP {response.status_code}）：{self._error(response)}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise VoiceStudioError("VoiceStudio 声音库返回了无法解析的数据") from exc
        return parse_voice_catalog(payload)

    def stream_tts(
        self,
        text: str,
        settings: dict[str, Any],
        on_audio: Callable[[bytes], None],
        cancel_event: threading.Event,
    ) -> int:
        text = str(text or "").strip()
        if not text:
            raise VoiceStudioError("没有可合成的文字")
        payload = {
            "model": str(settings.get("voicestudio_model") or "tts-1"),
            "voice": str(settings.get("voicestudio_voice") or "default"),
            "input": text,
            "response_format": "pcm",
            "speed": max(0.25, min(4.0, float(settings.get("tts_rate", 1.0)))),
        }
        try:
            response = self.session.post(
                f"{self.api_root}/audio/speech",
                headers={**self.headers, "Accept": "application/octet-stream"},
                json=payload,
                stream=True,
                timeout=(8.0, self.timeout),
            )
        except requests.RequestException as exc:
            raise VoiceStudioError("VoiceStudio 本地语音合成连接失败") from exc
        if not response.ok:
            raise VoiceStudioError(f"VoiceStudio 合成失败（HTTP {response.status_code}）：{self._error(response)}")

        content_type = response.headers.get("Content-Type", "").lower()
        encoded = any(token in content_type for token in ("wav", "mpeg", "mp3", "flac", "ogg", "opus", "aac"))
        buffered = bytearray()
        emitted = 0
        try:
            for chunk in response.iter_content(chunk_size=16_384):
                if cancel_event.is_set():
                    response.close()
                    break
                if not chunk:
                    continue
                if encoded:
                    buffered.extend(chunk)
                    continue
                if not buffered:
                    buffered.extend(chunk)
                    if len(buffered) < 12:
                        continue
                    header = bytes(buffered[:12])
                    if header.startswith((b"RIFF", b"ID3", b"fLaC", b"OggS")) or header[:2] == b"\xff\xfb":
                        encoded = True
                        continue
                    on_audio(bytes(buffered))
                    emitted += 1
                    buffered.clear()
                else:
                    on_audio(chunk)
                    emitted += 1

            if cancel_event.is_set():
                return emitted
            if encoded:
                pcm = decode_audio_to_pcm16_mono(bytes(buffered), sample_rate=24_000)
                if pcm:
                    on_audio(pcm)
                    emitted += 1
            elif buffered:
                on_audio(bytes(buffered))
                emitted += 1
        except requests.RequestException as exc:
            raise VoiceStudioError("VoiceStudio 返回音频时连接中断") from exc
        finally:
            response.close()
        return emitted
