"""Shared, explicit speech policy for live speech and voice comparisons."""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

SPEECH_SAMPLE_RATE = 24000
SSML_MODELS = {
    "cosyvoice-v3.5-plus", "cosyvoice-v3.5-flash", "cosyvoice-v3-plus",
    "cosyvoice-v3-flash", "cosyvoice-v2",
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
}

NATURAL_INSTRUCTION = "Speak naturally in English, with calm confidence, varied intonation and conversational pauses."
INSTRUCTION_MODELS = {
    "cosyvoice-v3.5-plus", "cosyvoice-v3.5-flash", "cosyvoice-v3-flash",
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
}


def speech_settings(settings, *, model=None, voice=None) -> dict:
    model = str(model or settings.get("tts_model", ""))
    mode = settings.get("speech_mode", "natural")
    instruction = str(settings.get("tts_instruction", "") or "").strip()
    if not instruction and mode == "natural" and model in INSTRUCTION_MODELS:
        instruction = NATURAL_INSTRUCTION
    return {
        "tts_model": model,
        "voice": str(voice if voice is not None else settings.get("tts_voice_id", "")),
        "speech_mode": mode,
        "tts_language_hint": "en",
        "tts_sample_rate": SPEECH_SAMPLE_RATE,
        "tts_volume": int(settings.get("tts_volume", 55)),
        "tts_rate": float(settings.get("tts_rate", 1.0)),
        "tts_pitch": float(settings.get("tts_pitch", 1.0)),
        "tts_seed": 0, "tts_instruction": instruction, "tts_emotion_tag": "",
        "enable_aigc_tag": False, "aigc_propagator": "", "aigc_propagate_id": "",
    }


def pause_comparison_text(text: str, model: str) -> str:
    """Explicit line breaks mark user-chosen pauses; ordinary punctuation stays intact."""
    if model not in SSML_MODELS:
        raise ValueError("此模型不支持本工具的 SSML 停顿对比，请选择 CosyVoice 音色")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("请在希望停顿的位置按回车，至少输入两行英文")
    return '<speak>' + '<break time="250ms"/>'.join(escape(line) for line in lines) + '</speak>'


def speech_chunks(text: str, limit: int = 450) -> list[str]:
    """Keep whole answers together; bound long requests at punctuation or words.

    The bound stays below Qwen VC's 600-character request limit and does not
    split decimals or common academic abbreviations just to reduce latency.
    """
    if limit < 1:
        raise ValueError("语音分段长度必须为正整数")
    result = []
    rest = text.strip()
    while len(rest) > limit:
        candidates = [m for m in re.finditer(r"[.!?;。！？；][\"'”’]?\s+", rest[:limit + 1])
                      if not re.search(r"\b(?:e\.g|i\.e|Fig|Figs|Eq|Eqs|Dr|Mr|Mrs|Prof|vs|etc)\.$",
                                       rest[:m.start() + 1], re.I)]
        cut = candidates[-1].end() if candidates else 0
        if cut < limit // 3:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        result.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        result.append(rest)
    return result
