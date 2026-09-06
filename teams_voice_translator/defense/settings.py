from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import SettingsStore

APP_DIR_NAME = "TeamsVoiceTranslator"
DEFENSE_FILE = "defense_settings.json"

DEFENSE_DEFAULTS: dict[str, Any] = {
    "speech_mode": "natural",
    "tts_model": "qwen3-tts-vc-realtime-2026-01-15",
    "tts_voice_id": "",
    "tts_volume": 55,
    "tts_rate": 1.0,
    "tts_pitch": 1.0,
    "tts_instruction": "",
    "ambience_mode": "off",
    "tts_language_hint": "en",
    "tts_sample_rate": 24000,
    "asr_model": "qwen3-asr-flash-realtime",
    "committee_asr_model": "qwen3-asr-flash-realtime",
    "llm_model": "qwen-plus",
    "qa_model": "qwen-plus",
    "qa_auto": True,
    "vad_silence_ms": 500,
    "max_context_turns": 14,
    "monitor_enabled": True,
    "monitor_output_device": None,
    "defense_brief": "",
    "glossary": "",
    "context_source_path": "",
    "overlay_enabled": True,
    "auto_record": True,
    "continuous_enabled": False,
    "output_directory": "",
    "direct_hotkey": "f5",
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

    def save(self) -> None:
        with self._save_lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.path)

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
