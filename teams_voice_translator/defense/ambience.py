"""合成的背景环境声：让虚拟麦克风里不是「绝对静音 ↔ 突然有人说话」的录音感。

全部实时合成、不带任何录音素材：没有循环接缝，也没有版权问题。每个场景由几种成分叠加：

* 底噪：平滑过的白噪声（房间空气声），可叠一点 50 Hz 电源嗡声；
* 人声嗡嗡（babble）：几路带通噪声各自慢慢起伏，远处听起来就像一群人在说话，但听不出任何字；
* 低频轰鸣（rumble）：车流、车厢、空调机组；
* 偶发事件：提示音「叮咚」、拉杆箱滚轮、键盘声、翻书、杯碟、雨滴……都很稀疏、很轻。

**音量有硬上限**：不管选哪个场景、音量调多大，底噪整体不超过 MAX_LEVEL_DB，单个样本的峰值
不超过 PEAK_CAP_DB。典型 TTS 语音约 −20 dBFS，上限保证背景比语音低 15 dB 以上，评委始终听得清。
"""
from __future__ import annotations

import math
import random
import sys
import audioop
import wave
from functools import lru_cache
from pathlib import Path
from array import array
from dataclasses import dataclass, field

# 整体电平上限（与各场景 level_db 同一口径）和逐样本峰值上限
MAX_LEVEL_DB = -36.0
PEAK_CAP_DB = -26.0
# 音量滑块：0~100，50 = 场景默认；每格对应的分贝变化
VOLUME_DEFAULT = 50
VOLUME_MIN_DB = -18.0   # 滑块 0
VOLUME_MAX_DB = 12.0    # 滑块 100（再被 MAX_LEVEL_DB 截住）


@dataclass(frozen=True)
class AmbiencePreset:
    label: str
    level_db: float
    tail_ms: int
    smooth: float
    hum: float = 0.0
    babble: float = 0.0        # 人声嗡嗡强度（相对底噪）
    rumble: float = 0.0        # 低频轰鸣强度
    events: tuple[str, ...] = field(default_factory=tuple)
    group: str = "房间"


AMBIENCE_PRESETS: dict[str, AmbiencePreset] = {
    "off": AmbiencePreset("关闭 · 保持纯净语音", -120.0, 0, 0.0),
    # ---- 房间类：正式答辩推荐 ----
    "subtle": AmbiencePreset("极轻房间底噪（推荐）", -54.0, 650, 0.90, 0.01),
    "room": AmbiencePreset("安静房间 · 稍明显", -49.0, 800, 0.86, 0.025),
    "meeting": AmbiencePreset("会议室空调 · 明显", -45.0, 950, 0.94, 0.08),
    "library": AmbiencePreset("图书馆 · 很静，偶尔翻书和脚步", -52.0, 800, 0.90, 0.0,
                              events=("page", "steps")),
    "dorm": AmbiencePreset("宿舍 / 家里 · 电脑风扇、窗外远处车流", -49.0, 850, 0.93, 0.03, rumble=0.35),
    "office": AmbiencePreset("开放办公室 · 远处人声和键盘", -47.0, 900, 0.90, 0.02, babble=0.35,
                             events=("keys",)),
    # ---- 户外 / 窗边 ----
    "rain": AmbiencePreset("窗外下雨", -46.0, 900, 0.80, 0.0, rumble=0.2, events=("rain",), group="户外"),
    "street": AmbiencePreset("临街窗边 · 远处车流", -45.0, 900, 0.95, 0.0, rumble=0.7, group="户外"),
    "cafe": AmbiencePreset("咖啡馆 · 人声嗡嗡、偶尔杯碟", -44.0, 950, 0.88, 0.0, babble=0.6,
                           events=("cup",), group="户外"),
    # ---- 交通场景（热闹，排练或特殊情况用）----
    "rail": AmbiencePreset("高铁站候车厅 · 大厅人声、偶尔提示音", -42.0, 1000, 0.90, 0.0, babble=0.8,
                           events=("chime", "steps"), group="交通"),
    "airport": AmbiencePreset("机场候机厅 · 广播提示音、拉杆箱", -42.0, 1000, 0.90, 0.0, babble=0.75,
                              events=("chime", "roll"), group="交通"),
    "train": AmbiencePreset("高铁车厢内 · 持续低频轰鸣", -43.0, 1000, 0.97, 0.0, rumble=1.0, babble=0.2,
                            group="交通"),
}

GROUP_ORDER = ("房间", "户外", "交通")

# 人声嗡嗡三路带通的（快, 慢）一阶低通系数：相减得到大致 300~1500 Hz 的「说话频段」
_BABBLE_BANDS = ((0.35, 0.08), (0.45, 0.12), (0.30, 0.07))

# 各类偶发事件的平均间隔（秒）
_EVENT_EVERY = {
    "chime": 70.0, "roll": 25.0, "keys": 5.0, "page": 25.0, "steps": 18.0, "cup": 9.0,
}


def volume_gain_db(volume: int) -> float:
    volume = max(0, min(100, int(volume)))
    if volume >= VOLUME_DEFAULT:
        return (volume - VOLUME_DEFAULT) / (100 - VOLUME_DEFAULT) * VOLUME_MAX_DB
    return (VOLUME_DEFAULT - volume) / VOLUME_DEFAULT * VOLUME_MIN_DB


def effective_level_db(mode: str, volume: int = VOLUME_DEFAULT) -> float:
    """场景 + 音量滑块之后的实际电平，永远不超过 MAX_LEVEL_DB。"""
    spec = AMBIENCE_PRESETS.get(mode, AMBIENCE_PRESETS["off"])
    if mode == "off" or mode not in AMBIENCE_PRESETS:
        return -120.0
    return min(MAX_LEVEL_DB, spec.level_db + volume_gain_db(volume))


def at_cap(mode: str, volume: int) -> bool:
    spec = AMBIENCE_PRESETS.get(mode)
    return bool(spec) and mode != "off" and spec.level_db + volume_gain_db(volume) >= MAX_LEVEL_DB


class RoomTone:
    """连续、可复现的低电平 PCM16 环境声。状态跨块延续，底噪不会在块与块之间跳变。"""

    def __init__(self, mode: str = "off", *, sample_rate: int = 24000, seed: int = 9137,
                 volume: int = VOLUME_DEFAULT) -> None:
        self.mode = mode if mode in AMBIENCE_PRESETS else "off"
        self.volume = max(0, min(100, int(volume)))
        self.sample_rate = int(sample_rate)
        self.spec = AMBIENCE_PRESETS[self.mode]
        self._random = random.Random(seed)
        self._smooth = 0.0
        self._phase = 0.0
        self._rumble = 0.0
        # 人声嗡嗡：3 路带通（两级一阶低通相减），各自有慢变的包络
        self._babble_lp = [[0.0, 0.0] for _ in range(3)]
        self._babble_env = [0.3, 0.5, 0.4]
        self._babble_target = [0.5, 0.5, 0.5]
        self._babble_count = 0
        self._overlay: list[float] = []      # 正在播放的偶发事件（尚未输出的部分）
        self._level_db = effective_level_db(self.mode, self.volume)
        self._peak_cap = 32767.0 * (10.0 ** (PEAK_CAP_DB / 20.0))

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    # ------------------------------------------------------------ 偶发事件
    def _schedule_events(self, count: int) -> None:
        seconds = count / max(self.sample_rate, 1)
        for kind in self.spec.events:
            if kind == "rain":
                continue   # 雨滴逐块生成，见 _rain
            every = _EVENT_EVERY.get(kind, 30.0)
            if self._random.random() < seconds / every:
                self._add_overlay(self._render_event(kind), self._random.randint(0, max(0, count - 1)))

    def _add_overlay(self, samples: list[float], offset: int = 0) -> None:
        need = offset + len(samples)
        if len(self._overlay) < need:
            self._overlay.extend([0.0] * (need - len(self._overlay)))
        for index, value in enumerate(samples):
            self._overlay[offset + index] += value

    def _tone(self, freq: float, seconds: float, amp: float, decay: float) -> list[float]:
        count = int(self.sample_rate * seconds)
        step = 2.0 * math.pi * freq / self.sample_rate
        attack = max(1, int(self.sample_rate * 0.01))
        out = []
        for n in range(count):
            env = min(1.0, n / attack) * math.exp(-decay * n / self.sample_rate)
            out.append(amp * env * (math.sin(step * n) + 0.25 * math.sin(2 * step * n)))
        return out

    def _clicks(self, count: int, spacing: tuple[float, float], amp: float, width_ms: float = 3.0) -> list[float]:
        rng = self._random
        out: list[float] = []
        width = max(1, int(self.sample_rate * width_ms / 1000))
        for _ in range(count):
            gap = int(self.sample_rate * rng.uniform(*spacing))
            burst = [amp * rng.uniform(-1, 1) * (1 - i / width) for i in range(width)]
            out.extend(burst)
            out.extend([0.0] * gap)
        return out

    def _render_event(self, kind: str) -> list[float]:
        rng = self._random
        if kind == "chime":   # 车站 / 机场广播前的「叮——咚」
            first = self._tone(659.0, 0.9, 0.9, 3.2)
            second = self._tone(523.0, 1.1, 0.9, 2.6)
            gap = int(self.sample_rate * 0.42)
            result = [0.0] * max(gap + len(second), len(first))
            for i, v in enumerate(first):
                result[i] += v
            for i, v in enumerate(second):
                result[gap + i] += v
            return result
        if kind == "roll":    # 拉杆箱：一段低沉滚动 + 规律的地砖缝咔嗒
            seconds = rng.uniform(1.5, 3.0)
            count = int(self.sample_rate * seconds)
            out, smooth = [], 0.0
            period = int(self.sample_rate * rng.uniform(0.10, 0.16))
            for n in range(count):
                smooth = 0.985 * smooth + 0.015 * rng.uniform(-1, 1)
                fade = min(1.0, n / (0.3 * self.sample_rate), (count - n) / (0.4 * self.sample_rate))
                click = 0.6 * rng.uniform(-1, 1) if n % period < 40 else 0.0
                out.append((smooth * 6.0 + click) * 0.5 * fade)
            return out
        if kind == "keys":
            return self._clicks(rng.randint(3, 12), (0.08, 0.22), 0.55)
        if kind == "steps":
            return self._clicks(rng.randint(4, 8), (0.45, 0.6), 0.45, width_ms=18.0)
        if kind == "page":    # 翻书：一小段柔和的「沙」声
            count = int(self.sample_rate * rng.uniform(0.25, 0.4))
            return [0.5 * rng.uniform(-1, 1) * math.sin(math.pi * n / count) for n in range(count)]
        if kind == "cup":     # 杯碟轻碰：短促的高音
            return self._tone(rng.uniform(2200.0, 3400.0), 0.12, 0.5, 30.0)
        return []

    def _rain(self, count: int) -> list[float]:
        drops = [0.0] * count
        rng = self._random
        expected = count / self.sample_rate * 90.0     # 每秒约 90 滴
        n = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
        for _ in range(n):
            pos = rng.randint(0, max(0, count - 1))
            amp = rng.uniform(0.2, 0.9)
            for k in range(min(24, count - pos)):
                drops[pos + k] += amp * rng.uniform(-1, 1) * (1 - k / 24)
        return drops

    # ------------------------------------------------------------ 主合成
    def _noise(self, count: int, *, fade: bool = False) -> array:
        result = array("h")
        if not self.enabled or count <= 0:
            return result
        spec = self.spec
        rng = self._random
        level = 32767.0 * (10.0 ** (self._level_db / 20.0))
        cap = self._peak_cap
        fade_start = int(count * 0.55) if fade else count
        phase_step = 2.0 * math.pi * 50.0 / max(self.sample_rate, 1)
        if spec.events and not fade:
            self._schedule_events(count)
        rain = self._rain(count) if "rain" in spec.events and not fade else None
        overlay = self._overlay
        babble = spec.babble
        rumble = spec.rumble
        env_step = max(1, int(self.sample_rate * 0.12))
        for index in range(count):
            white = rng.uniform(-1.0, 1.0)
            self._smooth = spec.smooth * self._smooth + (1.0 - spec.smooth) * white
            shaped = self._smooth * 1.8 + white * 0.12
            if spec.hum:
                shaped += math.sin(self._phase) * spec.hum
                self._phase = (self._phase + phase_step) % (2.0 * math.pi)
            if rumble:
                self._rumble = 0.996 * self._rumble + 0.004 * white
                shaped += self._rumble * 18.0 * rumble
            if babble:
                self._babble_count += 1
                if self._babble_count >= env_step:
                    self._babble_count = 0
                    for voice in range(3):
                        if rng.random() < 0.35:
                            self._babble_target[voice] = rng.uniform(0.05, 1.0)
                mix = 0.0
                for voice in range(3):
                    lp = self._babble_lp[voice]
                    fast, slow = _BABBLE_BANDS[voice]
                    noise = rng.uniform(-1.0, 1.0)
                    lp[0] += fast * (noise - lp[0])
                    lp[1] += slow * (noise - lp[1])
                    self._babble_env[voice] += 0.0006 * (self._babble_target[voice] - self._babble_env[voice])
                    mix += (lp[0] - lp[1]) * self._babble_env[voice]
                shaped += mix * 2.2 * babble
            if overlay:
                shaped += overlay[index] if index < len(overlay) else 0.0
            if rain is not None:
                shaped += rain[index]
            gain = 1.0
            if fade and index >= fade_start:
                gain = max(0.0, (count - index - 1) / max(count - fade_start, 1))
            value = shaped * level * gain
            if value > cap:
                value = cap
            elif value < -cap:
                value = -cap
            result.append(int(value))
        if overlay:
            del overlay[:count]
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

    def demo(self, samples: int) -> bytes:
        """试听用：和 bed 一样，但把这个场景的偶发事件（提示音、拉杆箱……）各安排一次在前几秒，
        不然试听 6 秒多半一个都碰不到，听不出场景的区别。"""
        if not self.enabled:
            return b""
        offset = int(self.sample_rate * 0.8)
        for kind in self.spec.events:
            if kind == "rain":
                continue
            self._add_overlay(self._render_event(kind), offset)
            offset += int(self.sample_rate * 1.6)
        return self.bed(samples)

    def bed(self, samples: int) -> bytes:
        """纯底噪（不叠在语音上）：没人说话时持续输出，让会议里的「房间」一直在。"""
        return self._to_bytes(self._noise(int(samples)))

    def tail(self, duration_ms: int | None = None) -> bytes:
        if not self.enabled:
            return b""
        milliseconds = self.spec.tail_ms if duration_ms is None else max(0, int(duration_ms))
        count = int(self.sample_rate * milliseconds / 1000)
        return self._to_bytes(self._noise(count, fade=True))


BACKGROUND_CHOICES = (
    ("off", "关闭背景声"),
    ("room", "室内设备 · 风机实录"),
    ("office", "办公室 · 键盘、打印机与电话"),
    ("rain", "雨打窗户 · 实录"),
    ("cafe", "咖啡馆 · 人声与杯碟实录"),
)


def background_mode(mode):
    mode = str(mode or 'off')
    if mode in ('subtle', 'meeting', 'dorm'):
        return 'room'
    return mode if mode in dict(BACKGROUND_CHOICES) else 'off'


@lru_cache(maxsize=4)
def _recording(mode, sample_rate):
    path = Path(__file__).resolve().parent.parent / 'ambience_assets' / f'{mode}.wav'
    with wave.open(str(path), 'rb') as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError('内置背景录音格式无效')
        data = source.readframes(source.getnframes())
        rate = source.getframerate()
    if rate != sample_rate:
        data, _ = audioop.ratecv(data, 2, 1, rate, sample_rate, None)
    samples = RoomTone._from_bytes(data)
    if len(samples) < sample_rate:
        raise ValueError('内置背景录音过短')
    # Remove DC and crossfade the end into the beginning for a quiet loop seam.
    mean = sum(samples) / len(samples)
    values = [x - mean for x in samples]
    seam = min(int(sample_rate * .1), len(values) // 8)
    crossfade = [values[-seam+i] * (1-i/seam) + values[i] * (i/seam) for i in range(seam)]
    return tuple(values[seam:-seam] + crossfade)


class RecordedRoomTone(RoomTone):
    """Loop real bundled recordings; no procedural noise is added."""
    def __init__(self, mode, *, sample_rate=24000, volume=VOLUME_DEFAULT):
        super().__init__(mode, sample_rate=sample_rate, volume=volume)
        source = _recording(mode, sample_rate)
        rms = math.sqrt(sum(x*x for x in source) / len(source))
        if rms <= 0:
            raise ValueError('内置背景录音为空音频')
        gain = min(32767 * 10 ** (self._level_db / 20) / rms,
                   self._peak_cap / max(abs(x) for x in source))
        self._recorded_samples = array('h', (int(x * gain) for x in source))
        self._offset = 0

    def _noise(self, count, *, fade=False):
        values = self._recorded_samples
        result = array('h')
        for i in range(count):
            value = values[(self._offset + i) % len(values)]
            if fade:
                value = int(value * max(0, min(1, (count-i-1) / max(1,count*.45))))
            result.append(value)
        self._offset = (self._offset + count) % len(values)
        return result


def make_room_tone(mode, *, sample_rate=24000, volume=VOLUME_DEFAULT):
    mode = background_mode(mode)
    if mode == 'off':
        return RoomTone('off', sample_rate=sample_rate, volume=volume)
    return RecordedRoomTone(mode, sample_rate=sample_rate, volume=volume)


__all__ = ["AMBIENCE_PRESETS", "AmbiencePreset", "GROUP_ORDER", "MAX_LEVEL_DB", "PEAK_CAP_DB",
           "RoomTone", "VOLUME_DEFAULT", "at_cap", "effective_level_db", "volume_gain_db"]
