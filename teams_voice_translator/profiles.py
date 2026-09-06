from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROFILE_FIELDS = (
    "translation_engine",
    "translation_domain",
    "translation_terms",
    "translation_memories",
    "translation_style",
    "long_form_model",
    "live_voice_clone_mode",
    "live_voice",
    "tts_provider",
    "tts_model",
    "tts_voices",
    "voicestudio_model",
    "voicestudio_voice",
    "tts_instruction",
    "tts_pronunciations",
    "voice",
)

PROFILE_FIELD_DEFAULTS: dict[str, Any] = {"tts_voices": {}}


def _profile_field_value(values: dict[str, Any], field: str) -> Any:
    """Read one profile field, normalising type and default per field."""
    value = values.get(field, PROFILE_FIELD_DEFAULTS.get(field, ""))
    if field == "tts_voices":
        if not isinstance(value, dict):
            return {}
        return {
            str(model): str(voice)
            for model, voice in value.items()
            if str(voice).strip()
        }
    return value


class CourseProfileStore:
    """Small, human-readable collection of per-course language settings."""

    def __init__(self, base_directory: Path) -> None:
        self.path = base_directory / "course_profiles.json"
        self.profiles: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        for name, values in raw.items():
            if isinstance(name, str) and name.strip() and isinstance(values, dict):
                self.profiles[name.strip()] = {
                    field: _profile_field_value(values, field)
                    for field in PROFILE_FIELDS
                }

    def save_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.profiles, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    def names(self) -> list[str]:
        return sorted(self.profiles, key=str.casefold)

    def get(self, name: str) -> dict[str, Any] | None:
        value = self.profiles.get(name.strip())
        return dict(value) if value is not None else None

    def save(self, name: str, values: dict[str, Any]) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("课程配置名称不能为空")
        self.profiles[clean_name] = {
            field: _profile_field_value(values, field) for field in PROFILE_FIELDS
        }
        self.save_file()

    def delete(self, name: str) -> bool:
        if self.profiles.pop(name.strip(), None) is None:
            return False
        self.save_file()
        return True
