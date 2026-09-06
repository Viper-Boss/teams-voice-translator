from __future__ import annotations

import json
import uuid
from typing import Any


QWEN3_TTS_VC_REALTIME_MODEL = "qwen3-tts-vc-realtime-2026-01-15"
QWEN3_TTS_VC_HTTP_MODEL = "qwen3-tts-vc-2026-01-22"
FUN_ASR_REALTIME_MODEL = "fun-asr-realtime"
COSYVOICE_V3_FLASH_MODEL = "cosyvoice-v3-flash"
COSYVOICE_V3_5_PLUS_MODEL = "cosyvoice-v3.5-plus"
VOICE_PREPROCESS_MODELS = frozenset({
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
    "cosyvoice-v3.5-plus", "cosyvoice-v3.5-flash", "cosyvoice-v3-flash",
})


def is_qwen3_tts_vc_model(model: str) -> bool:
    return model.startswith("qwen3-tts-vc-")


def is_qwen3_tts_vc_realtime_model(model: str) -> bool:
    return model.startswith("qwen3-tts-vc-realtime-")


def is_fun_asr_model(model: str) -> bool:
    return model.startswith("fun-asr")


def is_duplex_asr_model(model: str) -> bool:
    """实时识别里走 DashScope duplex task 协议（run-task/finish-task）的模型。"""
    normalized = model.strip().lower()
    return normalized.startswith(("fun-asr", "paraformer", "gummy")) or (
        normalized == "qwen-audio-3.0-asr-flash-streaming"
    )


def is_cosyvoice_model(model: str) -> bool:
    return model.startswith("cosyvoice-")


def cosyvoice_instruction_units(text: str) -> int:
    """Count CosyVoice instruction units (CJK ideographs count as two)."""
    return sum(
        2
        if ("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff")
        else 1
        for char in text
    )


def build_fun_asr_run_task(
    *,
    model: str = FUN_ASR_REALTIME_MODEL,
    sample_rate: int = 16000,
    context_terms: list[str] | None = None,
    vocabulary_id: str = "",
    language: str = "",
    heartbeat: bool = True,
    task_id: str = "",
) -> dict[str, Any]:
    """Fun-ASR realtime run-task over the DashScope duplex WebSocket protocol."""
    normalized_model = model.strip().lower()
    parameters: dict[str, Any] = {
        "format": "pcm",
        "sample_rate": int(sample_rate),
    }
    if normalized_model.startswith("fun-asr"):
        # heartbeat 是 Fun-ASR 专属参数，其他 duplex 模型发送未知参数可能直接报错。
        parameters["heartbeat"] = bool(heartbeat)
    if vocabulary_id.strip():
        parameters["vocabulary_id"] = vocabulary_id.strip()
    normalized_language = language.strip().lower()
    if normalized_language:
        # Fun-ASR treats Cantonese and other Chinese dialects as Chinese here.
        parameters["language_hints"] = ["zh" if normalized_language == "yue" else normalized_language]

    terms = [str(term).strip() for term in (context_terms or []) if str(term).strip()]
    input_payload: dict[str, Any] = {}
    if terms:
        # Fun-ASR does not support the inline `vocabulary` field. Its realtime
        # API does support dynamic context, which is a better fit for a course
        # glossary that can change from meeting to meeting.
        context_text = "本次课堂专业术语：" + "、".join(terms)
        input_payload["context"] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": context_text[:400]}],
            }
        ]
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id or str(uuid.uuid4()),
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "parameters": parameters,
            "input": input_payload,
        },
    }


def build_fun_asr_finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def qwen_tts_language_type(language: str) -> str:
    mapping = {
        "zh": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "es": "Spanish",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "ru": "Russian",
    }
    return mapping.get(language.strip().lower(), "Auto")


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


def build_live_translate_session_update(
    *,
    source_language: str = "zh",
    target_language: str = "en",
    phrases: dict[str, str] | None = None,
    voice_mode: str = "once",
    voice: str = "",
    audio_enabled: bool = True,
    continuous: bool = False,
    vad_threshold: float = 0.0,
    vad_silence_ms: int = 500,
) -> dict[str, Any]:
    """Build the official Qwen LiveTranslate realtime session config.

    Push-to-talk uses manual commits. Continuous F9 mode uses the official
    server-side VAD so one WebSocket can detect and translate many utterances.
    """

    translation: dict[str, Any] = {"language": target_language}
    if phrases:
        translation["corpus"] = {"phrases": phrases}

    session: dict[str, Any] = {
        "modalities": ["text", "audio"] if audio_enabled else ["text"],
        "input_audio_format": "pcm",
        "output_audio_format": "pcm",
        "input_audio_transcription": {
            "model": "qwen3-asr-flash-realtime",
            "language": source_language,
        },
        "translation": translation,
        "turn_detection": (
            {
                "type": "server_vad",
                "threshold": float(vad_threshold),
                "silence_duration_ms": int(vad_silence_ms),
            }
            if continuous
            else None
        ),
    }

    if voice_mode in {"once", "always"}:
        session.update(
            {
                "voice": "default",
                "enable_voice_clone": True,
                "voice_clone_options": {"frequency": voice_mode},
            }
        )
    elif voice_mode == "fixed":
        if not voice.strip():
            raise ValueError("固定复刻音色模式需要填写 LiveTranslate voice_id")
        session.update(
            {
                "voice": voice.strip(),
                "enable_voice_clone": True,
                "voice_clone_options": {"frequency": "never"},
            }
        )
    elif voice.strip():
        session["voice"] = voice.strip()

    return {
        "event_id": "event_session_update",
        "type": "session.update",
        "session": session,
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
    enable_ssml: bool = False,
) -> dict[str, Any]:
    if is_cosyvoice_model(model) and cosyvoice_instruction_units(instruction.strip()) > 100:
        raise ValueError("CosyVoice instruction 不可超过 100 字符单位（汉字按 2 个计算）")
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
    if enable_ssml:
        payload_input["enable_ssml"] = True
    if enable_aigc_tag and aigc_propagator.strip():
        payload_input["aigc_propagator"] = aigc_propagator.strip()
    if enable_aigc_tag and aigc_propagate_id.strip():
        payload_input["aigc_propagate_id"] = aigc_propagate_id.strip()
    return {"model": model, "input": payload_input}


def build_qwen3_tts_http_payload(
    text: str,
    *,
    model: str,
    voice: str,
    language: str = "en",
) -> dict[str, Any]:
    return {
        "model": model,
        "input": {
            "text": text,
            "voice": voice.strip(),
            "language_type": qwen_tts_language_type(language),
        },
    }


def build_qwen3_tts_realtime_session_update(
    *,
    voice: str,
    language: str = "en",
    sample_rate: int = 24000,
    volume: int = 50,
    rate: float = 1.0,
    pitch: float = 1.0,
) -> dict[str, Any]:
    return {
        "event_id": f"event_{uuid.uuid4().hex}",
        "type": "session.update",
        "session": {
            "voice": voice.strip(),
            "mode": "commit",
            "language_type": qwen_tts_language_type(language),
            "response_format": "pcm",
            "sample_rate": int(sample_rate),
            "speech_rate": float(rate),
            "volume": int(volume),
            "pitch_rate": float(pitch),
        },
    }


def build_qwen_voice_clone_payload(
    *,
    target_model: str,
    preferred_name: str,
    audio_data: str,
    language: str = "zh",
    transcript: str = "",
) -> dict[str, Any]:
    payload_input: dict[str, Any] = {
        "action": "create",
        "target_model": target_model,
        "preferred_name": preferred_name.strip(),
        "audio": {"data": audio_data.strip()},
        "language": language,
    }
    if transcript.strip():
        payload_input["text"] = transcript.strip()
    return {"model": "qwen-voice-enrollment", "input": payload_input}


def build_voice_clone_payload(
    *,
    target_model: str,
    prefix: str,
    audio_url: str,
    language: str = "zh",
    max_seconds: float = 20.0,
    preprocess: bool = False,
    volume_normalization: bool = False,
) -> dict[str, Any]:
    payload_input: dict[str, Any] = {
        "action": "create_voice",
        "target_model": target_model,
        "prefix": prefix.strip(),
        "url": audio_url.strip(),
    }
    # The HTTP endpoint requires strings for this field, unlike Python SDK's bool.
    if target_model.startswith(("cosyvoice-", "qwen-audio-3.0-tts-")):
        payload_input["enable_volume_normalization"] = "true" if volume_normalization else "false"
    if target_model in VOICE_PREPROCESS_MODELS:
        if not 3.0 <= float(max_seconds) <= 30.0:
            raise ValueError("复刻参考时长需在 3–30 秒之间")
        payload_input.update(
            {
                "language_hints": [language],
                "max_prompt_audio_length": float(max_seconds),
                "enable_preprocess": bool(preprocess),
            }
        )
    return {"model": "voice-enrollment", "input": payload_input}
