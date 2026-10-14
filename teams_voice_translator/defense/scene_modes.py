"""情景模式：像手机的情景模式一样，一键切换整套「识别 / 翻译 / 助答」模型组合。

用途是**应急顶替**：答辩中某个模型临时连不上（限流、服务波动、账号没开通），
不用进设置页一个个改，在首页下拉里换一个备用组合就行。

* 每个模式把下面 ``SCENE_KEYS`` 全部写死，所以「当前是哪个模式」可以精确判断；
  设置页里手动改过、和任何模式都对不上时显示「自定义」。
* 音色（语音合成模型 + voice_id）不放进模式里：音色和合成模型绑定，是否有备用
  音色取决于你复刻过哪些。首页另有一个音色下拉，列出音色库里的全部音色。
* 切换有锁（``scene_locked``，默认锁上）：解锁 → 选择 → 生效后自动重新锁上，
  防止答辩中误触。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCENE_KEYS: tuple[str, ...] = (
    "asr_model",
    "committee_asr_model",
    "llm_model",
    "qa_model",
    "qa_thinking",
    "my_translate_engine",
    "livetranslate_model",
)

KEY_LABELS = {
    "asr_model": "我的识别",
    "committee_asr_model": "评委识别",
    "llm_model": "翻译",
    "qa_model": "助答",
    "qa_thinking": "助答深度思考",
    "my_translate_engine": "我的发言翻译方式",
    "livetranslate_model": "同传模型",
}

ASR_31 = "qwen-audio-3.1-asr-flash-streaming"
ASR_30 = "qwen-audio-3.0-asr-flash-streaming"
ASR_QWEN3 = "qwen3-asr-flash-realtime"
LIVE_38 = "qwen3.8-livetranslate-flash-realtime"


@dataclass(frozen=True)
class SceneMode:
    id: str
    name: str
    blurb: str
    values: dict[str, Any] = field(default_factory=dict)
    builtin: bool = True


def _values(asr: str = ASR_31, committee_asr: str | None = None, llm: str = "qwen3.8-flash",
            qa: str = "qwen3.8-flash", qa_thinking: bool = False, engine: str = "llm",
            live: str = LIVE_38) -> dict[str, Any]:
    return {
        "asr_model": asr,
        "committee_asr_model": committee_asr or asr,
        "llm_model": llm,
        "qa_model": qa,
        "qa_thinking": qa_thinking,
        "my_translate_engine": engine,
        "livetranslate_model": live,
    }


BUILTIN_MODES: tuple[SceneMode, ...] = (
    SceneMode("recommended", "★ 推荐·稳定",
              "识别 3.1（带术语表）+ 翻译/助答 qwen3.8-flash + 分开翻译。平时就用这个。",
              _values()),
    SceneMode("livetranslate", "同传·低延迟",
              "我的发言改用千问同传 3.8 边听边译（更快，但没有整场上下文）；评委那边不变。",
              _values(engine="livetranslate")),
    SceneMode("backup_asr", "备用识别 A",
              "3.1 识别连不上时用：换上一版 qwen-audio-3.0（同一套接口，术语表照样带上）。",
              _values(asr=ASR_30)),
    SceneMode("backup_asr_b", "备用识别 B",
              "3.x 识别都不行时用：换 qwen3-asr-flash-realtime（另一套接口；专有名词容易错）。",
              _values(asr=ASR_QWEN3)),
    SceneMode("backup_llm", "备用大模型",
              "翻译或助答报错/很慢时用：换上一代 qwen3.7-flash，识别不变。",
              _values(llm="qwen3.7-flash", qa="qwen3.7-flash")),
    SceneMode("legacy_all", "全备用（老模型）",
              "新一代模型整体出问题时用：识别 qwen3-asr-flash-realtime + 翻译/助答旧版 qwen-plus（不思考，稳定）。",
              _values(asr=ASR_QWEN3, llm="qwen-plus", qa="qwen-plus")),
    SceneMode("deep_qa", "深度助答（排练用）",
              "助答换 qwen3.8-max 并开启深度思考，答得更深但要多等十几秒；其余同推荐。",
              _values(qa="qwen3.8-max", qa_thinking=True)),
)

CUSTOM_ID = "custom"


def _normalize(key: str, value: Any) -> Any:
    if key == "qa_thinking":
        return bool(value)
    return str(value or "").strip()


def custom_modes(settings) -> list[SceneMode]:
    raw = settings.get("scene_custom_modes", []) or []
    modes: list[SceneMode] = []
    if not isinstance(raw, list):
        return modes
    for item in raw:
        if not isinstance(item, dict):
            continue
        values = item.get("values")
        mode_id = str(item.get("id", "") or "").strip()
        name = str(item.get("name", "") or "").strip()
        if not mode_id or not name or not isinstance(values, dict):
            continue
        if not all(key in values for key in SCENE_KEYS):
            continue
        modes.append(SceneMode(mode_id, name, str(item.get("blurb", "") or "我保存的模式"),
                               {k: values[k] for k in SCENE_KEYS}, builtin=False))
    return modes


def all_modes(settings) -> list[SceneMode]:
    return list(BUILTIN_MODES) + custom_modes(settings)


def current_values(settings) -> dict[str, Any]:
    return {key: _normalize(key, settings.get(key)) for key in SCENE_KEYS}


def current_mode(settings, modes: list[SceneMode] | None = None) -> SceneMode | None:
    """和当前设置完全一致的模式；对不上（设置页里手动改过）返回 None。"""
    now = current_values(settings)
    for mode in modes if modes is not None else all_modes(settings):
        if all(_normalize(k, mode.values.get(k)) == now[k] for k in SCENE_KEYS if not _ignorable(k, now)):
            return mode
    return None


def _ignorable(key: str, values: dict[str, Any]) -> bool:
    """不用同传时，同传模型选的是哪个不影响任何行为，比较时忽略。"""
    return key == "livetranslate_model" and values.get("my_translate_engine") != "livetranslate"


def diff_mode(settings, mode: SceneMode) -> list[tuple[str, Any, Any]]:
    """切过去会改哪些：[(键, 旧值, 新值)]。"""
    now = current_values(settings)
    changes = []
    for key in SCENE_KEYS:
        new = _normalize(key, mode.values.get(key))
        if now[key] != new:
            changes.append((key, now[key], new))
    return changes


def describe_changes(changes: list[tuple[str, Any, Any]]) -> str:
    def show(value: Any) -> str:
        if isinstance(value, bool):
            return "开" if value else "关"
        return {"llm": "分开翻译", "livetranslate": "千问同传"}.get(value, str(value))

    return "；".join(f"{KEY_LABELS.get(k, k)} {show(old)} → {show(new)}" for k, old, new in changes)


def apply_mode(settings, mode: SceneMode) -> list[tuple[str, Any, Any]]:
    changes = diff_mode(settings, mode)
    if changes:
        settings.update({key: new for key, _old, new in changes})
    return changes


def save_custom_mode(settings, name: str) -> SceneMode:
    """把当前设置存成一个自己的模式；同名覆盖。"""
    name = name.strip()
    if not name:
        raise ValueError("模式名称不能为空")
    raw = [item for item in (settings.get("scene_custom_modes", []) or []) if isinstance(item, dict)]
    existing = next((item for item in raw if item.get("name") == name), None)
    if existing is not None:
        mode_id = str(existing.get("id"))
        raw = [item for item in raw if item is not existing]
    else:
        used = {str(item.get("id")) for item in raw}
        n = 1
        while f"my{n}" in used:
            n += 1
        mode_id = f"my{n}"
    values = current_values(settings)
    raw.append({"id": mode_id, "name": name, "blurb": "我保存的模式", "values": values})
    settings.set("scene_custom_modes", raw)
    return SceneMode(mode_id, name, "我保存的模式", values, builtin=False)


def delete_custom_mode(settings, mode_id: str) -> bool:
    raw = [item for item in (settings.get("scene_custom_modes", []) or []) if isinstance(item, dict)]
    kept = [item for item in raw if item.get("id") != mode_id]
    if len(kept) == len(raw):
        return False
    settings.set("scene_custom_modes", kept)
    return True


# ------------------------------------------------------------------ 音色
def voice_options(settings) -> list[dict[str, str]]:
    """音色库里能用的音色：[{voice_id, model, label}]。当前音色不在库里也列出来。"""
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    current_voice = str(settings.get("tts_voice_id", "") or "").strip()
    current_model = str(settings.get("tts_model", "") or "").strip()
    for entry in settings.get("voice_library", []) or []:
        if not isinstance(entry, dict):
            continue
        voice_id = str(entry.get("voice_id", "") or "").strip()
        if not voice_id or voice_id in seen:
            continue
        seen.add(voice_id)
        model = str(entry.get("target_model", "") or "").strip() or current_model
        name = str(entry.get("name", "") or "").strip() or voice_id[-8:]
        options.append({"voice_id": voice_id, "model": model, "label": f"{name} · {model}"})
    if current_voice and current_voice not in seen:
        options.insert(0, {"voice_id": current_voice, "model": current_model,
                           "label": f"当前音色 · {current_model}"})
    return options


def apply_voice(settings, voice_id: str) -> tuple[str, str] | None:
    """换成音色库里的另一个音色，合成模型跟着换成它绑定的那个。返回 (voice_id, model)。"""
    for option in voice_options(settings):
        if option["voice_id"] == voice_id:
            settings.update({"tts_voice_id": option["voice_id"], "tts_model": option["model"]})
            return option["voice_id"], option["model"]
    return None
