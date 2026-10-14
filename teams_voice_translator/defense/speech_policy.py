"""Shared, explicit speech policy for live speech and voice comparisons."""
from __future__ import annotations

import random
import re
from pathlib import Path
from xml.sax.saxutils import escape

from ..api_payloads import INSTRUCTION_MODELS, SSML_MODELS, tts_supports, is_qwen_audio_tts_model  # noqa: F401

SPEECH_SAMPLE_RATE = 24000

# 句间换气：真人连着说两句话不会有整整一拍的静音，也不会完全没有间隔。
# 160ms 是排练中听感最接近连续说话的值，调整前请先做 A/B。
SENTENCE_BREAK_MS = 160

NATURAL_INSTRUCTION = "Speak naturally in English, with calm confidence, varied intonation and conversational pauses."
CHINESE_NATURAL_INSTRUCTION = "请用自然、平静、自信的中文口语朗读，保留真实停顿和语调变化。"
INSTRUCTION_PRESETS = (
    ("自动自然答辩（推荐）", ""),
    ("平静自信 · 学术答辩", NATURAL_INSTRUCTION),
    ("亲切交流 · 像面对面回答", "Speak like a real person in a face-to-face conversation: warm, relaxed, and spontaneous."),
    ("思考后回答 · 自然收尾", "Sound thoughtful and responsive, with a brief reflective pause and natural sentence endings."),
    ("轻微犹豫 · 像现场思考", "Thoughtful student, slight hesitations and quiet breaths; clear academic English."),
    ("像真人聊天 · 不是在念稿", "Talk live like a real person, not reading: relaxed, natural breaths, small pauses."),
    ("强调重点 · 不要播音腔", "Use clear academic speech, naturally emphasizing key results without sounding like a narrator."),
    ("简洁坚定 · 结论与贡献", "Speak concisely and firmly, with restrained emotion and a natural conversational rhythm."),
    ("友好礼貌 · 有轻微情绪", "Speak warmly and politely, with subtle emotion, varied rhythm, and natural pauses."),
)

# 开场衔接词：真人被问到问题时不会在 0.2 秒后开始吐出语法完美的长句。
# 这些提前用克隆音色合成好缓存在本地，零延迟播出，既掩盖后续合成延迟，
# 也让整段听感从「机器应答」变成「人在思考后开口」。
OPENING_FILLERS_EN = (
    "Right, so —",
    "Yes, so —",
    "Okay, so —",
    "Sure.",
    "That's a good question.",
    "Right.",
)


def natural_delivery_instruction(model: str, language: str = "en") -> str:
    """Automatic delivery must not override a Qwen voice's reference accent.

    Use this in comparisons and pre-rendering too, which otherwise bypass the
    live defaults. Explicit user instructions remain a separate choice.
    """
    if model not in INSTRUCTION_MODELS or is_qwen_audio_tts_model(model):
        return ""
    return CHINESE_NATURAL_INSTRUCTION if language == "zh" else NATURAL_INSTRUCTION


def speech_settings(settings, *, model=None, voice=None, language="en") -> dict:
    model = str(model or settings.get("tts_model", ""))
    mode = settings.get("speech_mode", "natural")
    instruction = str(settings.get("tts_instruction", "") or "").strip()
    if model not in INSTRUCTION_MODELS:
        instruction = ""
    elif language == "zh":
        # 英文答辩语气预设可能明确要求 English；朗读中文原稿时换成中文指令，
        # 避免模型被相互冲突的语言要求影响发音和韵律。
        instruction = natural_delivery_instruction(model, language) if mode == "natural" else ""
    elif not instruction and mode == "natural":
        # A cloned voice already supplies pronunciation/accent. Do not silently
        # impose a new English delivery instruction when the user left it blank.
        instruction = natural_delivery_instruction(model, language)
    reference = str(settings.get("tts_reference_audio", "") or "").strip()
    reference_signature = ""
    if model == "qwen-audio-3.1-tts-next" and reference:
        try:
            stat = Path(reference).stat()
            reference_signature = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            pass
    return {
        "tts_model": model,
        "voice": str(voice if voice is not None else settings.get("tts_voice_id", "")),
        "tts_reference_audio": reference,
        "tts_reference_audio_signature": reference_signature,
        "speech_mode": mode,
        "tts_language_hint": language,
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


class FillerPicker:
    """轮换开场衔接词，且不连着重复同一个。

    每次都随机会出现「Right, so — Right, so —」这种一听就假的连击；
    固定顺序又会被评委听出规律。记住上一个，从剩下的里面挑。
    """

    def __init__(self, choices=OPENING_FILLERS_EN, rng=None) -> None:
        self._choices = tuple(str(item).strip() for item in choices if str(item).strip())
        if not self._choices:
            raise ValueError("衔接词池不能为空")
        self._rng = rng or random.Random()
        self._last = ""

    def pick(self) -> str:
        pool = [item for item in self._choices if item != self._last] or list(self._choices)
        self._last = self._rng.choice(pool)
        return self._last

    def reset(self) -> None:
        self._last = ""


def apply_opening_filler(text: str, filler: str) -> str:
    """把衔接词接在整段回答的最前面，和正文一起送进同一次合成请求。

    不单独发一次请求：那样会多一次网络往返，而且衔接词和正文之间会出现
    一个明显的接缝 —— 恰好是我们想消除的东西。合在一起由模型统一走韵律。
    """
    body = (text or "").strip()
    filler = (filler or "").strip()
    if not body:
        raise ValueError("没有可朗读的正文")
    if not filler:
        return body
    return f"{filler} {body}"


def join_sentences_ssml(sentences, model: str, *, break_ms: int = SENTENCE_BREAK_MS,
                        wrap_single: bool = False) -> str:
    """把连续几句合成一次 SSML 请求，让句与句共享一条语调曲线。

    逐句独立合成时每一句对 TTS 都是冷启动：它不知道上一句说到哪、语调收在哪里。
    真人说三句话是一条连续曲线，逐句拼接是三条曲线首尾相接 —— 这正是「拼接感」
    的来源。合成一次请求后由模型自己决定句间韵律，只在句边界插一个换气。

    模型不支持 SSML 时返回纯文本拼接，调用方无需分支。
    """
    lines = [str(item).strip() for item in sentences if str(item).strip()]
    if not lines:
        raise ValueError("没有可合成的句子")
    if not tts_supports(model, "ssml"):
        return " ".join(lines)
    if len(lines) == 1 and not wrap_single:
        return lines[0]
    if break_ms < 0:
        raise ValueError("句间换气时长不能为负")
    joiner = f'<break time="{int(break_ms)}ms"/>'
    return "<speak>" + joiner.join(escape(line) for line in lines) + "</speak>"


def group_sentences(sentences, *, max_per_group: int = 3, limit: int = 450) -> list[list[str]]:
    """按「最多几句 + 不超过字符上限」分组，供 join_sentences_ssml 逐组合成。

    分组是自然度和首句延迟之间的权衡：组越大韵律越连贯、开口越晚。
    默认 3 句对应约 0.5~1 秒的额外首句延迟，符合「自然度优先」的既定取舍。
    """
    if max_per_group < 1:
        raise ValueError("每组句数必须为正整数")
    if limit < 1:
        raise ValueError("分组字符上限必须为正整数")
    groups: list[list[str]] = []
    current: list[str] = []
    length = 0
    for raw in sentences:
        line = str(raw).strip()
        if not line:
            continue
        # 单句就超限时自己成组，交给 speech_chunks 再切，不要和别的句子挤在一起。
        if current and (len(current) >= max_per_group or length + len(line) + 1 > limit):
            groups.append(current)
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        groups.append(current)
    return groups


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
