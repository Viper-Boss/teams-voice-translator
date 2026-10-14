from __future__ import annotations

import json
import re
import uuid
from typing import Any


QWEN3_TTS_VC_REALTIME_MODEL = "qwen3-tts-vc-realtime-2026-01-15"
QWEN3_TTS_VC_HTTP_MODEL = "qwen3-tts-vc-2026-01-22"
FUN_ASR_REALTIME_MODEL = "fun-asr-realtime"
COSYVOICE_V3_FLASH_MODEL = "cosyvoice-v3-flash"
COSYVOICE_V3_5_PLUS_MODEL = "cosyvoice-v3.5-plus"
QWEN_AUDIO_TTS_PLUS_MODEL = "qwen-audio-3.0-tts-plus"
QWEN_AUDIO_TTS_NEXT_MODEL = "qwen-audio-3.1-tts-next"

# ---------------------------------------------------------------- 模型能力表
#
# 这张表是唯一事实来源。以前 api_payloads / speech_policy / ui 各自手写一份集合，
# 结果 qwen-audio-3.1-tts-flash 三处都漏掉了，而默认模型 qwen3-tts-vc-realtime-*
# 既不支持语气指令也不支持 SSML —— 用户选了语气预设却发现控件是灰的。
# 新增模型时只改这里。
TTS_CAPABILITIES: dict[str, frozenset[str]] = {
    # AudioGen：每次请求携带本机参考音频，不能使用 voice_id；完整文件返回。
    QWEN_AUDIO_TTS_NEXT_MODEL: frozenset({"reference"}),
    # qwen-audio 3.x：支持指令控制、SSML、情感标签，复刻免费，自然度最好
    "qwen-audio-3.0-tts-plus": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    "qwen-audio-3.1-tts-flash": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    "qwen-audio-3.0-tts-flash": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    # CosyVoice
    "cosyvoice-v3.5-plus": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    "cosyvoice-v3.5-flash": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    "cosyvoice-v3-plus": frozenset({"ssml", "clone"}),
    "cosyvoice-v3-flash": frozenset({"instruction", "ssml", "preprocess", "clone"}),
    "cosyvoice-v2": frozenset({"ssml", "clone"}),
    "cosyvoice-v1": frozenset({"clone"}),
    # Qwen3-TTS 声音复刻专用模型：低延迟，但没有指令/SSML 通道
    "qwen3-tts-vc-realtime-2026-01-15": frozenset({"clone"}),
    "qwen3-tts-vc-realtime-2025-11-27": frozenset({"clone"}),
    "qwen3-tts-vc-2026-01-22": frozenset({"clone"}),
    # MiniMax（百炼代理）：复刻与合成都是独立接口，见 build_minimax_* 。
    # 无语气指令、无 SSML；只有 HTTP，没有 WebSocket 流式。
    # 首次用复刻音色合成时百炼收一次性 ¥9.9 解锁费。
    "MiniMax/speech-2.8-hd": frozenset({"clone"}),
    "MiniMax/speech-2.8-turbo": frozenset({"clone"}),
    "MiniMax/speech-02-hd": frozenset({"clone"}),
    "MiniMax/speech-02-turbo": frozenset({"clone"}),
}

MINIMAX_MIN_SAMPLE_SECONDS = 10.0  # 官方：样音 10 秒 ~ 5 分钟


def is_minimax_tts_model(model: str) -> bool:
    return str(model).startswith("MiniMax/speech-")


def minimax_voice_setting(*, voice: str, rate: float = 1.0, volume: int = 50,
                          pitch: float = 1.0) -> dict[str, Any]:
    """把本工具的参数换算成 MiniMax 的取值体系。

    本工具：语速倍率 0.5~2.0、音量 0~100（50 为常态）、音调倍率 0.5~2.0。
    MiniMax：speed [0.5,2]、vol (0,10] 默认 1、pitch 为半音 [-12,12] 整数。
    音调按 12·log2(倍率) 换成半音 —— 直接把 1.1 塞进 pitch 会被当成 1 个半音。
    """
    import math

    speed = min(2.0, max(0.5, float(rate)))
    vol = min(10.0, max(0.1, float(volume) / 50.0))
    ratio = max(0.5, min(2.0, float(pitch)))
    semitones = int(round(12 * math.log2(ratio)))
    return {
        "voice_id": voice.strip(),
        "speed": round(speed, 3),
        "vol": round(vol, 3),
        "pitch": max(-12, min(12, semitones)),
    }


def build_minimax_tts_payload(
    text: str,
    *,
    model: str,
    voice: str,
    sample_rate: int = 24000,
    volume: int = 50,
    rate: float = 1.0,
    pitch: float = 1.0,
) -> dict[str, Any]:
    """百炼 MiniMax 同步语音合成。voice_setting / audio_setting 都在 input 下。

    固定要 pcm + 单声道 + 与本工具一致的采样率，播放链路才能直接吃；
    不开流式：流式的最后一帧可能带整段拼接音频，处理不好会整句重播一遍。
    """
    if not is_minimax_tts_model(model):
        raise ValueError(f"不是 MiniMax 语音模型：{model}")
    if not voice.strip():
        raise ValueError("MiniMax 合成需要复刻音色 voice_id")
    return {
        "model": model,
        "input": {
            "text": text,
            "voice_setting": minimax_voice_setting(voice=voice, rate=rate, volume=volume, pitch=pitch),
            "audio_setting": {
                "sample_rate": int(sample_rate),
                "format": "pcm",
                "channel": 1,
            },
        },
    }


def parse_minimax_tts_response(body: dict[str, Any]) -> bytes:
    """从 MiniMax 同步合成响应里取出 PCM。音频是 hex 编码，不是 base64。

    出问题一律抛 ValueError，由 aliyun 层包成 ApiError（这里不反向依赖 aliyun）：
    业务码非 0、没有音频、hex 解不开、字节数为奇数（半个采样点）。
    """
    output = (body or {}).get("output") or {}
    base = output.get("base_resp") or {}
    code = base.get("status_code", 0)
    if code not in (0, "0", None):
        raise ValueError(f"MiniMax 合成失败（{code}）：{base.get('status_msg', '')}")
    encoded = ((output.get("data") or {}).get("audio")) or ""
    if not encoded:
        raise ValueError("MiniMax 合成未返回音频")
    try:
        pcm = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValueError("MiniMax 返回的音频不是合法的 hex 编码") from exc
    if not pcm or len(pcm) % 2:
        raise ValueError("MiniMax 返回了不完整的 PCM 音频")
    return pcm


_MINIMAX_VOICE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{6,254}[A-Za-z0-9]$")


def valid_minimax_voice_id(voice_id: str) -> bool:
    """官方规则：8~256 字符、字母开头、仅字母数字-_、不能以 - 或 _ 结尾。"""
    return bool(_MINIMAX_VOICE_ID.match(str(voice_id)))


def build_minimax_clone_payload(
    *,
    target_model: str,
    voice_id: str,
    audio_url: str,
    preview_text: str,
    noise_reduction: bool = False,
    volume_normalization: bool = False,
) -> dict[str, Any]:
    """百炼 MiniMax 声音复刻。voice_id 由调用方自己起名，成功后直接可用。

    注意 text 是「复刻完成后的试听文本」，不是样音的转写。
    """
    if not is_minimax_tts_model(target_model):
        raise ValueError(f"不是 MiniMax 语音模型：{target_model}")
    if not valid_minimax_voice_id(voice_id):
        raise ValueError("MiniMax 音色名须 8~256 字符、字母开头、仅含字母数字-_，且不以 - 或 _ 结尾")
    preview = (preview_text or "").strip()
    if not preview:
        raise ValueError("MiniMax 复刻需要一段试听文本")
    return {
        "model": target_model,
        "input": {
            "action": "voice_clone",
            "voice_id": voice_id,
            "audio_url": audio_url.strip(),
            "text": preview[:1000],
            "need_noise_reduction": bool(noise_reduction),
            "need_volume_normalization": bool(volume_normalization),
        },
    }

INSTRUCTION_MODELS = frozenset(m for m, caps in TTS_CAPABILITIES.items() if "instruction" in caps)
SSML_MODELS = frozenset(m for m, caps in TTS_CAPABILITIES.items() if "ssml" in caps)
VOICE_PREPROCESS_MODELS = frozenset(m for m, caps in TTS_CAPABILITIES.items() if "preprocess" in caps)


def tts_supports(model: str, capability: str) -> bool:
    """未知模型一律按「不支持」处理，宁可少发字段也不要让请求整体报错。"""
    return capability in TTS_CAPABILITIES.get(str(model).strip(), frozenset())


def is_qwen_audio_tts_model(model: str) -> bool:
    """Qwen-Audio 3.x legacy TTS models (HTTP uses singular instruction)."""
    return str(model).startswith("qwen-audio-3") and str(model) != QWEN_AUDIO_TTS_NEXT_MODEL


def is_qwen3_tts_vc_model(model: str) -> bool:
    return model.startswith("qwen3-tts-vc-")


def is_qwen3_tts_vc_realtime_model(model: str) -> bool:
    return model.startswith("qwen3-tts-vc-realtime-")


def is_fun_asr_model(model: str) -> bool:
    return model.startswith("fun-asr")


def is_qwen_audio_streaming_asr(model: str) -> bool:
    """qwen-audio-3.x-asr-flash-streaming：与 Fun-ASR 同协议，另支持即时热词。"""
    normalized = model.strip().lower()
    return normalized.startswith("qwen-audio-3") and normalized.endswith("-asr-flash-streaming")


def is_duplex_asr_model(model: str) -> bool:
    """实时识别里走 DashScope duplex task 协议（run-task/finish-task）的模型。"""
    normalized = model.strip().lower()
    return normalized.startswith(("fun-asr", "paraformer", "gummy")) or is_qwen_audio_streaming_asr(normalized)


# 即时热词权重。官方示例用 5；取 4 留一点余地，避免把近音的普通词也硬拉成术语。
ASR_HOTWORD_WEIGHT = 4
DEFAULT_ASR_CONTEXT_PREFIX = "本次课堂专业术语："


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
    context_prefix: str = "",
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
    if terms and is_qwen_audio_streaming_asr(normalized_model):
        # qwen-audio-3.x 流式识别支持即时热词（官方格式 {"词": 权重}）。
        # Fun-ASR 不支持这个字段，发了可能直接报错，所以只给 qwen-audio。
        parameters["vocabulary"] = {term: ASR_HOTWORD_WEIGHT for term in dict.fromkeys(terms)}
    if terms:
        # Fun-ASR does not support the inline `vocabulary` field. Its realtime
        # API does support dynamic context, which is a better fit for a course
        # glossary that can change from meeting to meeting.
        prefix = context_prefix or DEFAULT_ASR_CONTEXT_PREFIX
        separator = ", " if prefix.isascii() else "、"
        context_text = prefix + separator.join(terms)
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
    if target_model.startswith("cosyvoice-") or is_qwen_audio_tts_model(target_model):
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
