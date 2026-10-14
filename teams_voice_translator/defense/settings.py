from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import SettingsStore
from .question_intent import INTENT_MODEL

APP_DIR_NAME = "TeamsVoiceTranslator"
DEFENSE_FILE = "defense_settings.json"

# 旧默认 qwen3-tts-vc-realtime-2026-01-15 低延迟但不支持语气指令、不支持 SSML，
# 而且在 create_tts_session 里走单连接串行的 TtsSession，拿不到并行预生成。
# qwen-audio-3.0-tts-plus 四样全有，复刻免费。切换后必须用新模型重新复刻音色
# （音色与模型绑定，旧 voice_id 不能跨模型使用）。
LEGACY_TTS_MODEL = "qwen3-tts-vc-realtime-2026-01-15"
LEGACY_ASR_MODEL = "qwen3-asr-flash-realtime"
RECOMMENDED_ASR_MODEL = "qwen-audio-3.1-asr-flash-streaming"
ASR_DEFAULTS_VERSION = 2
RECOMMENDED_TTS_MODEL = "qwen-audio-3.0-tts-plus"

DEFENSE_DEFAULTS: dict[str, Any] = {
    "speech_mode": "natural",
    "tts_model": RECOMMENDED_TTS_MODEL,
    "tts_voice_id": "",
    "tts_reference_audio": "",
    "tts_volume": 55,
    "tts_rate": 1.0,
    "tts_pitch": 1.0,
    "tts_instruction": "",
    # 句间换气：0 = 交给模型按标点自己决定（原有行为）。设为正数时用 SSML
    # <break> 显式控制，仅对支持 SSML 的模型生效。
    "sentence_break_ms": 0,
    # 开场衔接词：默认关闭。它会在译文前面加一个你没说过的词（"Right, so —"），
    # 属于替你开口，必须由你自己决定要不要开。
    "opening_filler": "off",
    # 像真人的小失误（卡壳 / 重读 / 说半句重来）：off / rare≈2% / natural≈3.5% / more≈6% 的句子。
    # 只改送给 TTS 的文本，字幕照常；数字、缩写、术语表的词永远不动。默认关，由你决定开不开。
    "humanize_level": "off",
    # 讲稿台 / 问答库 / 快捷用语提前渲染用哪套语气参数：
    # b = B 自然模式（Qwen 使用复刻原声，无语气指令/文字加工，语速音调 1.0）；
    # none = 和 A / 原始一样（不发指令，语速音调 1.0）；follow = 跟随设置（和现场发声一致）。
    "prerender_instruction": "b",
    # 讲稿怎么合成：group = 一段（≤450 字符）一次合成，和声音对比 B 一样连贯；
    # sentence = 逐句合成再拼接（旧方式，句与句之间是硬接缝）。
    "script_render_unit": "group",
    # 声音对比里最后一次用「B 自然答辩语气」试听的音色 {voice_id, model}：讲稿台音色下拉会标出来
    "voice_ab_last_b": {},
    # 设置「声卡」页下拉是否列出全部设备（默认只列有用的，见 audio.useful_devices）
    "show_all_audio_devices": False,
    # 评委每问一题，用 flash 从快捷用语里挑最适合的几条放在最上面
    "phrase_suggest": True,
    # 查重时点了「都保留」的组 / 对（按内容记），以后不再提示
    "glossary_dedupe_ignored": [],
    "qa_dedupe_ignored": [],
    # 先查问答库的匹配严格度：strict / medium（默认，= 原来的 0.6 / 0.55）/ loose
    "qa_match_strictness": "medium",
    # 第二通道（实时翻译、问答库匹配、生成回答；不含合成和识别）。默认关。
    "backup_channel_enabled": False,
    "backup_channel_mode": "race",          # race 同时发谁快用谁 / delayed 主通道 1.5 秒没出字再发
    "backup_route": "beijing-public",       # beijing-public 或其他地域 ID（channels.BACKUP_ROUTES）
    "backup_workspace_id": "",              # 其他地域的业务空间 ID（可空）
    "backup_model": "",                     # 第二通道用的模型（空 = 和主通道相同）
    "backup_asr_enabled": False,
    "backup_asr_model": "qwen-audio-3.0-asr-flash-streaming",   # 和主模型一样带得上术语表
    "backup_tts_enabled": False,
    "backup_tts_mode": "delayed",           # delayed 主通道 2 秒没出声才用第二通道 / race 同时合成
    "backup_tts_model": "",
    "backup_tts_voice_id": "",
    # 问答库窗口的列宽、上下 / 左右分隔位置（拖过之后记住）
    "qa_bank_column_widths": [],
    "qa_bank_split": [],
    "qa_bank_detail_split": [],
    # 译文风格：准确自然 / 更口语 / 轻微思考。只影响我的中文转英文，评委翻译不变。
    "translation_style": "accurate",
    # 语音缓存容量。一小时的会实际发声约 25 分钟 ≈ 72MB，讲稿 20 分钟 ≈ 58MB，
    # 1GB 上限绰绰有余；内存那份只是加速副本，淘汰掉磁盘上还在。
    "disk_cache_mb": 1024,
    "memory_cache_mb": 96,
    "ambience_mode": "off",
    # 背景环境音量 0~100，50 = 场景默认；最终电平被 ambience.MAX_LEVEL_DB 截住
    "ambience_volume": 50,
    "tts_language_hint": "en",
    "tts_sample_rate": 24000,
    # 2026-09：百炼官方推荐的实时识别换成 qwen-audio-3.1-asr-flash-streaming。
    # 它支持即时热词和上下文，术语表（PPT 导入时自动填）会一起带上；
    # 旧默认 qwen3-asr-flash-realtime 走另一套协议，术语表用不上。
    "asr_model": RECOMMENDED_ASR_MODEL,
    "committee_asr_model": RECOMMENDED_ASR_MODEL,
    "asr_defaults_version": ASR_DEFAULTS_VERSION,
    # 翻译在关键路径上，要的是快和稳；助答不在关键路径（评委还在说话时就能开始生成），
    # 多花两秒无所谓，但答案质量差一档就是现场翻车，所以两者分开配置。
    # 2026-09 核对百炼：qwen-plus / qwen3-max 已列为旧版。新一代默认会「思考」，
    # 翻译请求里由 chat_models.thinking_fields 显式关掉，保证首字延迟。
    "llm_model": "qwen3.8-flash",
    # 助答现场生成也默认用 flash：评委在等，快比深更要紧；深的内容靠问答库。
    # 面板上的下拉框随时可以换成 qwen3.8-max，「深度思考」另外勾选。
    "qa_model": "qwen3.8-flash",
    "qa_thinking": False,
    "qa_review_thinking": True,
    # 问答库：评委提问时先在本地预案里找同一个问题，找不到才现场生成。
    # 库本身存在配置目录的 qa_bank.json，不放在这个设置文件里。
    "qa_bank_enabled": True,
    # 我的发言怎么翻：llm = 识别后交给翻译大模型（带整场上下文，默认）；
    # livetranslate = 千问同传模型边听边译（测试中），译文仍用克隆音色播放。
    "my_translate_engine": "llm",
    # 论文预习：答辩前用强模型通读论文（慢但准），现场助答仍用快模型
    "study_model": "qwen3.8-max",
    "study_thinking": True,
    "study_qa_count": 20,
    "livetranslate_model": "qwen3.8-livetranslate-flash-realtime",
    # 助答回答区字号（像素）。大屏幕可以调大，面板上 A−/A+ 调整。
    "qa_font_px": 18,
    "qa_auto": True,
    "qa_collect_wait_ms": 3000,
    "qa_intent_enabled": True,
    # 整题理解、字幕翻译、回答生成三者独立；不随情景模式更改。
    "qa_intent_model": INTENT_MODEL,
    "qa_intent_thinking": False,
    # 命中问答库 / 开始现场生成时，弹出置顶小窗：评委问题 + 回答 + 「回答 / 不回答」
    "qa_popup": True,
    "qa_popup_geometry": "",
    # 批量预渲染会调用付费 TTS；由用户在界面主动开启后才运行。
    "prerender_auto": False,
    # 首页情景模式：锁上时不能切换，防止答辩中误触（切换生效后自动重新锁上）
    "scene_locked": True,
    # 自然模式「早开口」：译文第一句一完整就先合成播放，后面的边播边合成
    "speech_early_start": True,
    # 提前渲染（讲稿台 / 问答库 / 快捷用语）单独用的合成模型和音色。留空 = 和现场一样。
    # 建议：现场用 flash（快），提前渲染用 plus（好听，不赶时间）。
    "prerender_tts_model": "",
    "prerender_voice_id": "",
    # 讲稿台：句与句之间、段与段之间的停顿（毫秒）。逐句合成再拼接，不加停顿就是一口气念到底。
    "script_gap_ms": 350,
    "script_paragraph_gap_ms": 800,
    "script_voice_locked": True,
    # 设置页里的声卡选择默认收起（本软件固定搭配 VB-CABLE 使用，一般不用动）
    "show_audio_devices": False,
    "loopback_default_v2": False,
    # 程序输出设备的名字：Windows 的设备编号会变，按名字找回用户锁定的那一个
    "teams_output_device_name": "",
    "teams_output_host_api": "",
    # 论文预习窗口字号（像素），窗口里 A-/A+ 调
    "study_font_px": 14,
    "scene_custom_modes": [],
    # 样音语种。跨语种复刻（中文样音合成英文）质量明显下降，答辩说英文就用英文样音。
    "clone_language": "en",
    "vad_silence_ms": 500,
    "max_context_turns": 14,
    "monitor_enabled": True,
    "live_monitor_enabled": True,  # Only F8/F9; other playback keeps its monitor routing.
    "monitor_output_device": None,
    "monitor_output_device_name": "",
    "monitor_output_host_api": "",
    "defense_brief": "",
    "glossary": "",
    "context_source_path": "",
    "overlay_enabled": True,
    "auto_record": True,
    "continuous_enabled": False,
    "output_directory": "",
    # 原声键默认 F8：F5 是 PowerPoint「从头放映」，答辩时一按就把 PPT 打乱了。
    # F8 在 PowerPoint / Teams / 微信里都不做事。
    "direct_hotkey": "f8",
    "hotkey_defaults_version": 2,
    "translate_hotkey": "f9",
    "cancel_hotkey": "esc",
    "voice_library": [],
}


class DefenseSettings:
    """Defense-mode settings.

    Devices and the API key/workspace are shared with the main app through the
    regular ``settings.json`` and the Windows credential store, so nothing has
    to be configured twice.  Defense-specific values live in a separate
    ``defense_settings.json`` next to it and never contain secrets.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._save_lock = threading.RLock()
        self.shared = SettingsStore(base_dir)
        if base_dir is None:
            base_dir = self.shared.base_dir
        self.base_dir = base_dir
        self.path = base_dir / DEFENSE_FILE
        self.values = deepcopy(DEFENSE_DEFAULTS)
        self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key in DEFENSE_DEFAULTS:
                    self.values[key] = value
            self._migrate_asr_defaults(loaded)
            self._migrate_hotkey_defaults(loaded)

    def _migrate_hotkey_defaults(self, loaded: dict) -> None:
        """一次性迁移：原声键还是旧默认 F5 的，换成不和 PPT 冲突的键（优先 F8）。

        只动「值恰好是 F5」的；换成的键不能和翻译键、停止键重复。
        """
        try:
            version = int(loaded.get("hotkey_defaults_version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        if version >= 2:
            return
        if str(self.values.get("direct_hotkey", "") or "").lower() == "f5":
            taken = {str(self.values.get(k, "") or "").lower() for k in ("translate_hotkey", "cancel_hotkey")}
            for candidate in ("f8", "f7", "f6", "f4"):
                if candidate not in taken:
                    self.values["direct_hotkey"] = candidate
                    break
        self.values["hotkey_defaults_version"] = 2

    def _migrate_asr_defaults(self, loaded: dict) -> None:
        """一次性迁移：还停在旧默认识别模型的，换成推荐模型。

        只动「值恰好等于旧默认」的：用户主动选过别的模型（比如 fun-asr）就不碰。
        迁移只改内存里的值，下一次任何保存都会连同版本号一起写回。
        """
        try:
            version = int(loaded.get("asr_defaults_version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        if version >= ASR_DEFAULTS_VERSION:
            return
        for key in ("asr_model", "committee_asr_model"):
            if self.values.get(key) == LEGACY_ASR_MODEL:
                self.values[key] = RECOMMENDED_ASR_MODEL
        self.values["asr_defaults_version"] = ASR_DEFAULTS_VERSION

    def save(self) -> None:
        with self._save_lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            contents = json.dumps(self.values, ensure_ascii=False, indent=2)
            for attempt in range(5):
                try:
                    temp.write_text(contents, encoding="utf-8")
                    temp.replace(self.path)
                    break
                except PermissionError:
                    # Windows scanners may hold a just-written file briefly.
                    # Keep the lock and old JSON intact; genuine denial still raises.
                    if attempt == 4:
                        raise
                    time.sleep(.025 * (2 ** attempt))

    # --------------------------------------------------------------- access
    def get(self, key: str, default: Any = None) -> Any:
        if key in DEFENSE_DEFAULTS:
            return self.values.get(key, default)
        return self.shared.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._save_lock:
            if key not in DEFENSE_DEFAULTS:
                raise KeyError(f"非答辩模式配置项：{key}（请在共享设置中修改）")
            self.values[key] = value
            self.save()

    def update(self, values: dict[str, Any]) -> None:
        with self._save_lock:
            for key, value in values.items():
                if key in DEFENSE_DEFAULTS:
                    self.values[key] = value
            self.save()

    # -------------------------------------------------------------- shared
    def get_api_key(self) -> str:
        return self.shared.get_api_key()

    def set_api_key(self, api_key: str) -> None:
        self.shared.set_api_key(api_key)

    @property
    def workspace_id(self) -> str:
        return str(self.shared.get("workspace_id", "") or "").strip()

    @property
    def proxy_mode(self) -> str:
        return str(self.shared.get("proxy_mode", "direct") or "direct")

    @property
    def proxy(self) -> str:
        return str(self.shared.get("http_proxy", "") or "").strip()

    @property
    def ws_proxy(self) -> str:
        """Proxy for WebSocket clients; they never follow the system proxy."""
        return self.proxy if self.proxy_mode == "manual" else ""

    def glossary_terms(self) -> list[dict[str, str]]:
        raw = str(self.get("glossary", "") or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        result: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict):
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                if source and target:
                    result.append({"source": source, "target": target})
        return result

    def set_glossary(self, terms: list[dict[str, str]]) -> None:
        self.set("glossary", json.dumps(terms, ensure_ascii=False))
