from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import keyring


APP_NAME = "TeamsVoiceTranslator"
KEYRING_SERVICE = "TeamsVoiceTranslator.AlibabaCloud"

DEFAULTS: dict[str, Any] = {
    "workspace_id": "",
    "input_device": None,
    "teams_output_device": None,
    "loopback_device": None,
    "monitor_enabled": False,
    "monitor_output_device": None,
    "direct_sample_rate": 48000,
    "asr_model": "qwen3-asr-flash-realtime",
    "asr_language": "zh",
    "vad_threshold": 0.0,
    "vad_silence_ms": 500,
    "translation_model": "qwen-mt-flash",
    "translation_engine": "live",
    "live_translate_model": "qwen3.5-livetranslate-flash-realtime",
    "live_voice_clone_mode": "once",
    "live_voice": "",
    "summary_model": "qwen-plus",
    "source_language": "Chinese",
    "target_language": "English",
    "translation_terms": "",
    "translation_memories": "",
    "translation_domain": "Education, academic discussion and online meetings",
    "confirm_before_speak": False,
    "tts_model": "qwen-audio-3.0-tts-flash",
    "voice": "loongjohn",
    "tts_sample_rate": 24000,
    "tts_volume": 55,
    "tts_rate": 1.0,
    "tts_pitch": 1.0,
    "tts_seed": 0,
    "tts_language_hint": "en",
    "tts_instruction": "Speak natural, clear conversational American English for an online lesson.",
    "tts_emotion_tag": "",
    "enable_aigc_tag": False,
    "aigc_propagator": "",
    "aigc_propagate_id": "",
    "direct_hotkey": "f8",
    "translate_hotkey": "f9",
    "cancel_hotkey": "esc",
    "request_timeout": 45,
    "http_proxy": "",
    "overlay_enabled": False,
    "overlay_opacity": 82,
    "overlay_always_on_top": True,
    "overlay_chinese_font_size": 28,
    "overlay_english_font_size": 22,
    "overlay_x": None,
    "overlay_y": None,
    "overlay_width": 760,
    "overlay_height": 150,
    "direct_caption_enabled": True,
    "teacher_caption_enabled": True,
    "teacher_asr_language": "en",
    "auto_start_teacher_caption": False,
    "subtitle_display_mode": "both",
    "subtitle_export_language": "both",
    "subtitle_export_format": "both",
    "subtitle_export_include_source": True,
    "subtitle_export_include_time": True,
    "active_profile": "",
    "translation_style": "polite",
    "speak_mode": "auto",
    "theme": "light",
    "output_directory": "",
}


class SettingsStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            root = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
            base_dir = root / APP_NAME
        self.base_dir = base_dir
        self.path = self.base_dir / "settings.json"
        self.values = deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if key in DEFAULTS and key != "api_key":
                        self.values[key] = value
                if "speak_mode" not in loaded and loaded.get("confirm_before_speak"):
                    self.values["speak_mode"] = "confirm"
        except (OSError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe = {key: value for key, value in self.values.items() if key != "api_key"}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key in DEFAULTS:
                self.values[key] = value
        self.save()

    def get_api_key(self) -> str:
        env_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if env_key:
            return env_key
        try:
            return (keyring.get_password(KEYRING_SERVICE, "api_key") or "").strip()
        except Exception:
            return ""

    def set_api_key(self, api_key: str) -> None:
        api_key = api_key.strip()
        if not api_key:
            return
        keyring.set_password(KEYRING_SERVICE, "api_key", api_key)
