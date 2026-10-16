from __future__ import annotations

from contextlib import nullcontext

import audioop
import io
import queue
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sounddevice as sd

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover - exercised by packaged dependency checks
    pyaudio = None


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    inputs: int
    outputs: int

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} · {self.host_api}"


@dataclass(frozen=True)
class LoopbackDevice:
    index: int
    name: str
    channels: int
    sample_rate: int

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} · WASAPI 回环"


def list_audio_devices() -> tuple[list[AudioDevice], list[AudioDevice]]:
    host_apis = sd.query_hostapis()
    inputs: list[AudioDevice] = []
    outputs: list[AudioDevice] = []
    for index, raw in enumerate(sd.query_devices()):
        host_name = host_apis[raw["hostapi"]]["name"]
        item = AudioDevice(
            index=index,
            name=raw["name"],
            host_api=host_name,
            inputs=int(raw["max_input_channels"]),
            outputs=int(raw["max_output_channels"]),
        )
        if item.inputs > 0:
            inputs.append(item)
        if item.outputs > 0:
            outputs.append(item)
    return inputs, outputs


def list_loopback_devices() -> list[LoopbackDevice]:
    """Return Windows WASAPI loopback endpoints exposed by PyAudioWPatch."""
    if pyaudio is None:
        return []
    backend = pyaudio.PyAudio()
    try:
        result: list[LoopbackDevice] = []
        for raw in backend.get_loopback_device_info_generator():
            result.append(
                LoopbackDevice(
                    index=int(raw["index"]),
                    name=str(raw["name"]),
                    channels=max(1, int(raw.get("maxInputChannels") or 2)),
                    sample_rate=max(8000, int(raw.get("defaultSampleRate") or 48000)),
                )
            )
        return result
    finally:
        backend.terminate()


def is_program_output_loopback(name: str) -> bool:
    """Identify VB-CABLE render endpoints that carry this app's own voice."""
    lowered = str(name or "").casefold()
    return any(marker in lowered for marker in (
        "vb-audio virtual cable", "cable input", "cable in 16ch", "virtual cable",
    ))


def choose_committee_loopback(
    devices: list[LoopbackDevice], configured_index=None, configured_name: str = "",
) -> LoopbackDevice | None:
    """Resolve the saved physical endpoint despite unstable Windows indices."""
    allowed = [device for device in devices if not is_program_output_loopback(device.name)]
    wanted = str(configured_name or "").casefold().replace("[loopback]", "").strip()
    if wanted:
        for device in allowed:
            current = device.name.casefold().replace("[loopback]", "").strip()
            if current == wanted:
                return device
    if configured_index is not None:
        for device in allowed:
            if device.index == int(configured_index):
                return device
    # 没保存过（或保存的已不存在）：优先真实扬声器/耳机，别落到远程桌面、虚拟声卡上——
    # 例如 ToDesk 装的虚拟声卡常被 Windows 设成默认输出，监听它就听不到评委。
    real = [device for device in allowed if not is_virtual_endpoint(device.name)]
    if real:
        return real[0]
    return allowed[0] if allowed else None


VIRTUAL_ENDPOINT_MARKERS = (
    "todesk", "virtual", "voicemeeter", "vb-audio", "oray", "sunlogin", "向日葵",
    "remote audio", "steam streaming", "nvidia broadcast", "obs",
)


def is_virtual_endpoint(name: str) -> bool:
    """远程桌面 / 虚拟声卡一类的播放端点：自动挑选评委监听设备时尽量避开。"""
    lowered = str(name or "").casefold()
    return any(marker in lowered for marker in VIRTUAL_ENDPOINT_MARKERS)


_JUNK_DEVICE_MARKERS = (
    "声音映射器", "sound mapper", "主声音", "primary sound", "()",
)
_MIX_MARKERS = ("立体声混音", "stereo mix", "what u hear", "波形混音")


def _is_cable_input(name: str) -> bool:
    lowered = str(name or "").casefold()
    return "cable input" in lowered or "cable in 16ch" in lowered


def useful_devices(devices, *, kind: str, keep_index=None) -> list:
    """设置里下拉框默认只列有用的设备，别把几十条重复驱动全摊出来。

    * 同一个设备 Windows 会按 MME / DirectSound / WASAPI / WDM-KS 各列一遍。本程序按
      24 kHz 直接打开设备，优先 MME；仅其他接口提供的独立端点仍保留。
    * 去掉「声音映射器」「主声音驱动」这类转发项和名字为空的项。
    * kind="mic"：去掉虚拟声卡、远程桌面、立体声混音（选了它们要么没声、要么把自己的声音录进去）。
    * kind="monitor"：去掉虚拟声卡、远程桌面（试听要用真的耳机/音箱）。
    * kind="cable"：只列 VB-CABLE 的 CABLE Input；一个都没有时全列，方便看清楚缺什么。
    * 当前已选的设备（keep_index）即使被筛掉也会保留在最前面，保证「选哪个锁哪个」不跳。
    """
    items = list(devices or [])
    def rank(d):
        host = str(getattr(d, "host_api", "")).casefold()
        return 0 if "mme" in host else 1 if "wasapi" in host else 2 if "directsound" in host else 3
    base = sorted(items, key=rank)
    base = [d for d in base if not any(m in str(d.name).casefold() for m in _JUNK_DEVICE_MARKERS)]
    if kind == "mic":
        base = [d for d in base if not is_virtual_endpoint(d.name) and not _is_cable_input(d.name)
                and "cable output" not in str(d.name).casefold()
                and not any(m in str(d.name).casefold() for m in _MIX_MARKERS)]
    elif kind == "monitor":
        base = [d for d in base if not is_virtual_endpoint(d.name) and not _is_cable_input(d.name)]
    elif kind == "cable":
        cables = [d for d in base if _is_cable_input(d.name)] or [d for d in items if _is_cable_input(d.name)]
        base = cables or base
    # 同名去重（MME 会把长名字截成 31 个字符，截断后相同的算同一个）
    seen: set[str] = set()
    result = []
    for device in base:
        key = str(device.name).strip().casefold()[:31]
        if key in seen:
            continue
        seen.add(key)
        result.append(device)
    if keep_index is not None and all(d.index != keep_index for d in result):
        kept = next((d for d in items if d.index == keep_index), None)
        if kept is not None:
            key = str(kept.name).strip().casefold()[:31]
            result = [d for d in result if str(d.name).strip().casefold()[:31] != key]
            result.insert(0, kept)
    return result


def resolve_program_output(settings):
    """Resolve the virtual cable by identity instead of trusting an old index."""
    index = settings.shared.get("teams_output_device")
    name = str(settings.get("teams_output_device_name", "") or "")
    host = str(settings.get("teams_output_host_api", "") or "")
    if not name:
        return index
    _, outputs = list_audio_devices()
    matches = [d for d in outputs if d.name == name and (not host or d.host_api == host)]
    exact = next((d for d in matches if d.index == index), None)
    if exact:
        return exact.index
    if matches:
        # Legacy configurations did not record the driver interface.
        return next((d.index for d in matches if "mme" in d.host_api.casefold()), matches[0].index)
    raise RuntimeError(f"会议输出设备已断开：{name}。请重新选择虚拟声卡，防止声音发往其他设备。")


def resolve_monitor_device(settings):
    """Follow the saved endpoint identity when Windows renumbers devices; never reroute silently."""
    index = settings.get("monitor_output_device")
    name = str(settings.get("monitor_output_device_name", "") or "")
    host = str(settings.get("monitor_output_host_api", "") or "")
    if index is None or not name:
        return index  # System default or legacy configuration, until explicitly saved.
    _, outputs = list_audio_devices()
    matches = [d for d in outputs if d.name == name and (not host or d.host_api == host)]
    exact = next((d for d in matches if d.index == index), None)
    if exact:
        return exact.index
    if len(matches) == 1:
        return matches[0].index
    raise RuntimeError(f"试听设备已断开或无法确定：{name}。请在设置中重新选择试听设备。")


class MicrophoneCapture:
    def __init__(
        self,
        device: int | None,
        callback: Callable[[bytes], None],
        *,
        sample_rate: int = 16000,
        block_ms: int = 100,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.on_error = on_error
        self.stream: sd.RawInputStream | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._error = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("麦克风仍在运行或关闭中")
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._read_loop, name="microphone-reader", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("麦克风启动超时")
        if self._error:
            self.stop()
            raise RuntimeError(f"麦克风启动失败：{self._error}")

    def _read_loop(self) -> None:
        # The same Python thread owns open/read/close. No Python callback pointer
        # is handed to PortAudio, and no network I/O runs in its native callback.
        blocksize = int(self.sample_rate * self.block_ms / 1000)
        stream = None
        try:
            stream = sd.RawInputStream(device=self.device, samplerate=self.sample_rate,
                                       channels=1, dtype="int16", blocksize=blocksize)
            self.stream = stream
            stream.start()
            self._ready.set()
            while not self._stop.is_set():
                pcm, _overflowed = stream.read(blocksize)
                if not self._stop.is_set():
                    self.callback(bytes(pcm))
        except Exception as exc:
            self._error = exc
            if self._ready.is_set() and not self._stop.is_set() and self.on_error:
                self.on_error(str(exc))
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self.stream = None
            self._ready.set()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        # Keep ownership if a driver/read is still blocked. Never free a stream
        # or reset the stop token under a live reader.


class MultiOutputPlayer:
    def __init__(self, devices: list[int | None], sample_rate: int) -> None:
        unique: list[int | None] = []
        for device in devices:
            if device not in unique:
                unique.append(device)
        self.devices = unique
        self.sample_rate = sample_rate
        self.streams: list[sd.RawOutputStream] = []
        self._retired_streams = []
        self._stream_formats = {}
        self._monitor_streams = set()
        self._io_lock = threading.RLock()

    def __enter__(self) -> "MultiOutputPlayer":
        try:
            for index, device in enumerate(self.devices):
                rate, channels = self.sample_rate, 1
                try:
                    stream = sd.RawOutputStream(device=device, samplerate=rate, channels=channels, dtype="int16")
                except sd.PortAudioError as exc:
                    if len(exc.args) < 2 or exc.args[1] not in (-9997, -9998):
                        raise
                    # Some WDM-KS headphones require native rate/stereo. Retry the
                    # SAME endpoint with conversion, never the default/another device.
                    info = sd.query_devices(device, 'output')
                    rate = int(info['default_samplerate'])
                    channels = min(2, int(info['max_output_channels']))
                    stream = sd.RawOutputStream(device=device, samplerate=rate, channels=channels, dtype="int16")
                self.streams.append(stream)
                self._stream_formats[id(stream)] = [rate, channels, None]
                if index > 0:
                    self._monitor_streams.add(id(stream))
                stream.start()
            return self
        except Exception:
            self.close()
            raise

    def write(self, pcm: bytes, *, include_monitor: bool = True) -> None:
        """写入所有健康流；某条流报错时立即剔除（防止继续写入死流导致原生崩溃）并抛出一次。"""
        with self._io_lock:
            self._write_locked(pcm, include_monitor=include_monitor)

    def _write_locked(self, pcm, *, include_monitor=True):
        if not self.streams:
            raise RuntimeError("输出流已关闭或没有可用设备")
        failures = []
        healthy = []
        for stream in self.streams:
            try:
                fmt = self._stream_formats.get(id(stream), [self.sample_rate,1,None])
                converted = pcm
                if fmt[0] != self.sample_rate:
                    converted, fmt[2] = audioop.ratecv(converted,2,1,self.sample_rate,fmt[0],fmt[2])
                if fmt[1] == 2:
                    converted = audioop.tostereo(converted,2,1,1)
                if not include_monitor and id(stream) in self._monitor_streams:
                    # Keep the endpoint clock running without leaking this utterance.
                    converted = bytes(len(converted))
                stream.write(converted)
                healthy.append(stream)
            except Exception as exc:
                self._retired_streams.append(stream)
                failures.append(f"设备 {getattr(stream, 'device', '?')}: {exc}")
        if failures:
            self.streams = healthy
            raise RuntimeError("; ".join(failures))

    def close(self) -> None:
        with self._io_lock:
            healthy, retired = self.streams, self._retired_streams
            self.streams = []
            self._retired_streams = []
            self._stream_formats = {}
            self._monitor_streams.clear()
            for stream in healthy:
                try:
                    stream.stop()
                except Exception:
                    pass
            for stream in healthy + retired:
                try:
                    stream.close()
                except Exception:
                    pass

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


class DirectAudioBridge:
    """Low-latency physical microphone -> virtual output bridge."""

    def __init__(self) -> None:
        self.capture: MicrophoneCapture | None = None
        self.player: MultiOutputPlayer | None = None
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=24)
        self.thread: threading.Thread | None = None
        self.running = threading.Event()

    def start(
        self,
        input_device: int | None,
        output_device: int | None,
        sample_rate: int,
        tap: Callable[[bytes], None] | None = None,
        *,
        monitor_devices: list[int | None] | None = None,
        monitor_enabled: Callable[[], bool] | None = None,
        output_lock=None,
    ) -> None:
        if self.running.is_set():
            return
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("原声输出线程仍在关闭中")
        self.audio_queue = queue.Queue(maxsize=24)
        self.player = MultiOutputPlayer([output_device, *(monitor_devices or [])], sample_rate)
        self.player.__enter__()
        self.running.set()

        def write_loop() -> None:
            assert self.player is not None
            player = self.player
            try:
                while self.running.is_set():
                    try:
                        item = self.audio_queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if item is None or not self.running.is_set():
                        break
                    with output_lock if output_lock is not None else nullcontext():
                        if not self.running.is_set():
                            break
                        if monitor_enabled is None:
                            player.write(item)
                        else:
                            player.write(item, include_monitor=monitor_enabled())
            finally:
                player.close()

        self.thread = threading.Thread(target=write_loop, name="direct-audio-output", daemon=True)
        self.thread.start()

        def enqueue(chunk: bytes) -> None:
            if not self.running.is_set():
                return
            if tap is not None:
                try:
                    tap(chunk)
                except Exception:
                    # Never let a caption/recording side channel interrupt the
                    # low-latency microphone path.
                    pass
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass

        try:
            self.capture = MicrophoneCapture(
                input_device,
                enqueue,
                sample_rate=sample_rate,
                block_ms=20,
            )
            self.capture.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.running.clear()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.player is not None and (self.thread is None or not self.thread.is_alive()):
            self.player.close()
            self.player = None


class Pcm16Resampler:
    """Stateful mono PCM16 resampler suitable for consecutive ASR chunks."""

    def __init__(self, source_rate: int, target_rate: int = 16000) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.state = None

    def process(self, pcm: bytes) -> bytes:
        if self.source_rate == self.target_rate:
            return pcm
        converted, self.state = audioop.ratecv(
            pcm,
            2,
            1,
            self.source_rate,
            self.target_rate,
            self.state,
        )
        return converted


def pcm16_to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    if channels == 2:
        return audioop.tomono(pcm, 2, 0.5, 0.5)
    frame_width = channels * 2
    usable = len(pcm) - (len(pcm) % frame_width)
    source = memoryview(pcm)[:usable]
    result = bytearray((usable // frame_width) * 2)
    output_index = 0
    for frame_index in range(0, usable, frame_width):
        total = 0
        for channel in range(channels):
            start = frame_index + channel * 2
            total += int.from_bytes(source[start : start + 2], "little", signed=True)
        sample = max(-32768, min(32767, round(total / channels)))
        result[output_index : output_index + 2] = sample.to_bytes(2, "little", signed=True)
        output_index += 2
    return bytes(result)


def wav_payload_offset(data: bytes) -> int | None:
    """Return the byte offset where audio payload starts inside a WAV blob.

    ``None`` means the data does not start with a RIFF/WAVE header, so the
    caller can treat it as raw PCM instead.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        body = offset + 8
        if chunk_id == b"data":
            return body
        offset = body + size + (size & 1)
    return None


def is_compressed_audio(data: bytes) -> bool:
    """Best-effort detection of MP3/Ogg/FLAC payloads that need FFmpeg."""
    if data.startswith((b"ID3", b"OggS", b"fLaC")):
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def decode_audio_to_pcm16_mono(data: bytes, *, sample_rate: int = 24000) -> bytes:
    """Decode WAV/MP3/other audio bytes into PCM16 mono at ``sample_rate``.

    Bytes that look like raw PCM are returned unchanged.  WAV payloads are
    parsed with the standard library; compressed formats fall back to the
    bundled FFmpeg binary.
    """
    if not data:
        return b""
    if wav_payload_offset(data) is not None:
        try:
            return _wav_bytes_to_pcm(data, sample_rate=sample_rate)
        except (EOFError, OSError, wave.Error):
            return data
    if not is_compressed_audio(data):
        return data
    return _ffmpeg_to_pcm(data, sample_rate=sample_rate)


def _wav_bytes_to_pcm(data: bytes, *, sample_rate: int) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
    if channels > 1:
        frames = pcm16_to_mono(frames, channels)
    if rate and rate != sample_rate:
        frames = Pcm16Resampler(rate, sample_rate).process(frames)
    return frames


def _ffmpeg_to_pcm(data: bytes, *, sample_rate: int) -> bytes:
    import imageio_ffmpeg  # imported lazily: only needed for this fallback

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=data,
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"无法解码 TTS 音频：{exc}") from exc
    return _wav_bytes_to_pcm(result.stdout, sample_rate=sample_rate)


class SystemAudioCapture:
    """Capture the selected Windows playback endpoint through WASAPI loopback."""

    def __init__(self) -> None:
        self.backend = None
        self.stream = None
        self.running = threading.Event()
        self.device: LoopbackDevice | None = None

    def start(
        self,
        device_index: int | None,
        callback: Callable[[bytes], None],
        *,
        target_rate: int = 16000,
        block_ms: int = 100,
    ) -> LoopbackDevice:
        if self.running.is_set():
            assert self.device is not None
            return self.device
        if pyaudio is None:
            raise RuntimeError("缺少 PyAudioWPatch，无法使用 Windows 系统声回环")
        self.backend = pyaudio.PyAudio()
        try:
            if device_index is None:
                raw = self.backend.get_default_wasapi_loopback()
            else:
                raw = self.backend.get_device_info_by_index(int(device_index))
            self.device = LoopbackDevice(
                index=int(raw["index"]),
                name=str(raw["name"]),
                channels=max(1, int(raw.get("maxInputChannels") or 2)),
                sample_rate=max(8000, int(raw.get("defaultSampleRate") or 48000)),
            )
            resampler = Pcm16Resampler(self.device.sample_rate, target_rate)

            def on_audio(in_data, _frame_count, _time_info, status_flags):
                if self.running.is_set() and in_data:
                    try:
                        mono = pcm16_to_mono(in_data, self.device.channels)
                        converted = resampler.process(mono)
                        if converted:
                            callback(converted)
                    except Exception:
                        # Audio callbacks must never propagate into PortAudio.
                        pass
                return (None, pyaudio.paContinue)

            self.stream = self.backend.open(
                format=pyaudio.paInt16,
                channels=self.device.channels,
                rate=self.device.sample_rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=max(128, int(self.device.sample_rate * block_ms / 1000)),
                stream_callback=on_audio,
            )
            self.running.set()
            self.stream.start_stream()
            return self.device
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.running.clear()
        if self.stream is not None:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.backend is not None:
            try:
                self.backend.terminate()
            except Exception:
                pass
            self.backend = None
        self.device = None


class PcmWaveSink:
    """Non-blocking mono PCM16 WAV writer for real-time audio taps."""

    def __init__(self) -> None:
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
        self.thread: threading.Thread | None = None
        self.path: Path | None = None
        self.running = threading.Event()
        self._stop_requested = threading.Event()
        self.error = ""
        self.frames_written = 0
        self.sample_rate = 16000
        self.last_audio_at = 0.0

    def start(self, path: Path, sample_rate: int) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("录音写入线程尚未停止")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.audio_queue = queue.Queue(maxsize=256)
        # Never truncate an earlier paid-for/irreplaceable recording.
        raw_file = path.open("xb")
        wav_file = wave.open(raw_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        self.sample_rate = sample_rate
        self.error = ""
        self.frames_written = 0
        self.last_audio_at = 0.0
        self._stop_requested.clear()
        self.running.set()

        def writer() -> None:
            try:
                last_flush = 0.0
                # On stop, drain the entire queue instead of discarding a block
                # to make space for a sentinel.
                while not self._stop_requested.is_set() or not self.audio_queue.empty():
                    try:
                        chunk = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if not chunk:
                        continue
                    wav_file.writeframesraw(chunk)
                    self.frames_written += len(chunk) // 2
                    if time.monotonic() - last_flush >= 1.0:
                        # writeframes patches the WAV header using the public API.
                        # An open recording remains playable up to the last flush.
                        wav_file.writeframes(b"")
                        raw_file.flush()
                        last_flush = time.monotonic()
            except Exception as exc:
                self.error = f"录音写入失败：{exc}"
            finally:
                try:
                    wav_file.close()
                except Exception as exc:
                    self.error = self.error or f"录音保存失败：{exc}"
                finally:
                    try:
                        raw_file.close()
                    except Exception as exc:
                        self.error = self.error or f"录音文件关闭失败：{exc}"
                    finally:
                        self.running.clear()

        self.thread = threading.Thread(target=writer, name="pcm-wave-sink", daemon=True)
        self.thread.start()

    def write(self, pcm: bytes) -> None:
        if not self.running.is_set() or self._stop_requested.is_set():
            return
        self.last_audio_at = time.monotonic()
        try:
            self.audio_queue.put_nowait(pcm)
        except queue.Full:
            self.error = "录音写入过慢，部分音频未保存，请检查磁盘"

    def stop(self) -> Path | None:
        self._stop_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                self.error = "录音文件仍在写入，请勿关闭程序"
                raise RuntimeError(self.error)
            self.thread = None
        if self.error:
            raise RuntimeError(self.error)
        return self.path


def mix_mono_wav(first: Path, second: Path, destination: Path) -> Path:
    """Mix two mono PCM16 WAV files without loading a full meeting into RAM."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(first), "rb") as left, wave.open(str(second), "rb") as right:
        if left.getframerate() != right.getframerate():
            raise ValueError("分轨录音采样率不一致，无法生成混合录音")
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(left.getframerate())
            while True:
                a = left.readframes(8192)
                b = right.readframes(8192)
                if not a and not b:
                    break
                size = max(len(a), len(b))
                if len(a) < size:
                    a += b"\0" * (size - len(a))
                if len(b) < size:
                    b += b"\0" * (size - len(b))
                output.writeframesraw(audioop.add(a, b, 2))
    return destination


class DualTrackRecorder:
    """Record microphone and system audio to separate tracks plus a mixed WAV."""

    def __init__(self) -> None:
        self.microphone = WavRecorder()
        self.system_capture = SystemAudioCapture()
        self.system_sink = PcmWaveSink()
        self.running = threading.Event()
        self.paths: dict[str, Path] = {}

    def start(
        self,
        microphone_device: int | None,
        loopback_device: int | None,
        directory: Path,
        *,
        sample_rate: int = 16000,
    ) -> dict[str, Path]:
        if self.running.is_set():
            return dict(self.paths)
        stamp = recording_stamp()
        self.paths = {
            "microphone": directory / f"microphone_{stamp}.wav",
            "system": directory / f"teacher_system_{stamp}.wav",
            "mixed": directory / f"meeting_mixed_{stamp}.wav",
        }
        try:
            self.microphone.start(microphone_device, self.paths["microphone"], sample_rate)
            self.system_sink.start(self.paths["system"], sample_rate)
            self.system_capture.start(
                loopback_device,
                self.system_sink.write,
                target_rate=sample_rate,
                block_ms=100,
            )
            self.running.set()
            return dict(self.paths)
        except Exception:
            try:
                self.stop(create_mix=False)
            except Exception:
                pass
            raise

    def stop(self, *, create_mix: bool = True) -> dict[str, Path]:
        errors = []
        # One failed device must not prevent the other track from being saved.
        for resource in (self.system_capture, self.microphone, self.system_sink):
            try:
                resource.stop()
            except Exception as exc:
                errors.append(str(exc))
        microphone_path = self.microphone.path
        system_path = self.system_sink.path
        self.running.clear()
        if not errors and create_mix and microphone_path and system_path and microphone_path.exists() and system_path.exists():
            mix_mono_wav(microphone_path, system_path, self.paths["mixed"])
        if errors:
            raise RuntimeError("；".join(errors))
        return dict(self.paths)


def recording_stamp() -> str:
    """Unique even when settings cause several restarts within one second."""
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:12]


class WavRecorder(PcmWaveSink):
    """Continuously record one microphone to a mono 16-bit PCM WAV file."""

    def __init__(self) -> None:
        super().__init__()
        self.stream: sd.RawInputStream | None = None

    def start(self, device: int | None, path: Path, sample_rate: int = 48000) -> None:
        if self.running.is_set():
            return
        super().start(path, sample_rate)

        def on_audio(indata, _frames, _time, _status) -> None:
            self.write(bytes(indata))

        try:
            self.stream = sd.RawInputStream(
                device=device,
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=int(sample_rate * 0.1),
                callback=on_audio,
            )
            self.stream.start()
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise

    def stop(self) -> Path | None:
        errors = []
        if self.stream is not None:
            stream, self.stream = self.stream, None
            try:
                stream.stop()
            except Exception as exc:
                errors.append(str(exc))
            try:
                stream.close()
            except Exception as exc:
                errors.append(str(exc))
        try:
            super().stop()
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            self.error = "；".join(errors)
            raise RuntimeError(self.error)
        return self.path
