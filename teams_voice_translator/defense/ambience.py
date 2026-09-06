"""Very low level synthetic room tone for less abrupt virtual-mic audio."""
from __future__ import annotations

import math
import random
import sys
from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class AmbiencePreset:
    label: str
    level_db: float
    tail_ms: int
    smooth: float
    hum: float = 0.0


AMBIENCE_PRESETS: dict[str, AmbiencePreset] = {
    "off": AmbiencePreset("关闭 · 保持纯净语音", -120.0, 0, 0.0),
    "subtle": AmbiencePreset("极轻房间底噪（推荐）", -54.0, 650, 0.90, 0.01),
    "room": AmbiencePreset("安静房间 · 稍明显", -49.0, 800, 0.86, 0.025),
    "meeting": AmbiencePreset("会议室空调 · 明显", -45.0, 950, 0.94, 0.08),
}


class RoomTone:
    """Generate continuous, deterministic low-level PCM16 room tone.

    It is intentionally synthetic rather than a looped recording: there is no
    repeating sample boundary, no asset to publish, and the state continues
    across speech blocks so the noise floor does not jump between packets.
    """

    def __init__(self, mode: str = "off", *, sample_rate: int = 24000, seed: int = 9137) -> None:
        self.mode = mode if mode in AMBIENCE_PRESETS else "off"
        self.sample_rate = int(sample_rate)
        self.spec = AMBIENCE_PRESETS[self.mode]
        self._random = random.Random(seed)
        self._smooth = 0.0
        self._phase = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _noise(self, count: int, *, fade: bool = False) -> array:
        result = array("h")
        if not self.enabled or count <= 0:
            return result
        peak = 32767.0 * (10.0 ** (self.spec.level_db / 20.0))
        fade_start = int(count * 0.55) if fade else count
        phase_step = 2.0 * math.pi * 50.0 / max(self.sample_rate, 1)
        for index in range(count):
            white = self._random.uniform(-1.0, 1.0)
            self._smooth = self.spec.smooth * self._smooth + (1.0 - self.spec.smooth) * white
            shaped = self._smooth * 1.8 + white * 0.12
            if self.spec.hum:
                shaped += math.sin(self._phase) * self.spec.hum
                self._phase = (self._phase + phase_step) % (2.0 * math.pi)
            gain = 1.0
            if fade and index >= fade_start:
                gain = max(0.0, (count - index - 1) / max(count - fade_start, 1))
            value = int(max(-32768, min(32767, shaped * peak * gain)))
            result.append(value)
        return result

    @staticmethod
    def _from_bytes(pcm: bytes) -> array:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples

    @staticmethod
    def _to_bytes(samples: array) -> bytes:
        if sys.byteorder != "little":
            samples.byteswap()
        return samples.tobytes()

    def mix(self, pcm: bytes) -> bytes:
        if not self.enabled or not pcm:
            return pcm
        usable = len(pcm) - (len(pcm) % 2)
        samples = self._from_bytes(pcm[:usable])
        noise = self._noise(len(samples))
        for index, addition in enumerate(noise):
            samples[index] = max(-32768, min(32767, samples[index] + addition))
        return self._to_bytes(samples) + pcm[usable:]

    def tail(self, duration_ms: int | None = None) -> bytes:
        if not self.enabled:
            return b""
        milliseconds = self.spec.tail_ms if duration_ms is None else max(0, int(duration_ms))
        count = int(self.sample_rate * milliseconds / 1000)
        return self._to_bytes(self._noise(count, fade=True))


__all__ = ["AMBIENCE_PRESETS", "AmbiencePreset", "RoomTone"]
