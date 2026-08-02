from __future__ import annotations

import json
from typing import Any


def build_asr_session_update(
    *, language: str = "zh", threshold: float = 0.0, silence_ms: int = 500
) -> dict[str, Any]:
    return {
        "event_id": "event_session_update",
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "sample_rate": 16000,
            "input_audio_transcription": {"language": language},
            "turn_detection": {
                "type": "server_vad",
                "threshold": float(threshold),
                "silence_duration_ms": int(silence_ms),
            },
        },
    }

def parse_json_list(raw: str, field_name: str) -> list[dict[str, str]]:
    raw = raw.strip()
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是 JSON 数组")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("source"), str) or not isinstance(item.get("target"), str):
            raise ValueError(f"{field_name} 第 {index} 项必须包含字符串 source 和 target")
        result.append({"source": item["source"], "target": item["target"]})
    return result


def build_translation_payload(
    text: str,
    *,
    model: str = "qwen-mt-flash",
    source_lang: str = "Chinese",
    target_lang: str = "English",
    terms: list[dict[str, str]] | None = None,
    memories: list[dict[str, str]] | None = None,
    domain: str = "",
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    if terms:
        options["terms"] = terms
    if memories:
        options["tm_list"] = memories
    if domain.strip():
        options["domains"] = domain.strip()
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "translation_options": options,
    }


def build_tts_payload(
    text: str,
    *,
    model: str,
    voice: str,
    sample_rate: int = 24000,
    volume: int = 50,
    rate: float = 1.0,
    pitch: float = 1.0,
    seed: int = 0,
    language_hint: str = "en",
    instruction: str = "",
    emotion_tag: str = "",
    enable_aigc_tag: bool = False,
    aigc_propagator: str = "",
    aigc_propagate_id: str = "",
) -> dict[str, Any]:
    spoken_text = f"{emotion_tag.strip()} {text}".strip() if emotion_tag.strip() else text
    payload_input: dict[str, Any] = {
        "text": spoken_text,
        "voice": voice.strip(),
        "format": "pcm",
        "sample_rate": int(sample_rate),
        "volume": int(volume),
        "rate": float(rate),
        "pitch": float(pitch),
        "seed": int(seed),
        "language_hints": [language_hint],
        "enable_aigc_tag": bool(enable_aigc_tag),
    }
    if instruction.strip():
        payload_input["instruction"] = instruction.strip()
    if enable_aigc_tag and aigc_propagator.strip():
        payload_input["aigc_propagator"] = aigc_propagator.strip()
    if enable_aigc_tag and aigc_propagate_id.strip():
        payload_input["aigc_propagate_id"] = aigc_propagate_id.strip()
    return {"model": model, "input": payload_input}


def build_voice_clone_payload(
    *,
    target_model: str,
    prefix: str,
    audio_url: str,
    language: str = "zh",
    max_seconds: float = 20.0,
    preprocess: bool = False,
) -> dict[str, Any]:
    return {
        "model": "voice-enrollment",
        "input": {
            "action": "create_voice",
            "target_model": target_model,
            "prefix": prefix.strip(),
            "url": audio_url.strip(),
            "language_hints": [language],
            "max_prompt_audio_length": float(max_seconds),
            "enable_preprocess": bool(preprocess),
        },
    }
