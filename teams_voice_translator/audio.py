from __future__ import annotations

import audioop
import io
import queue
import subprocess
import threading
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
    return allowed[0] if allowed else None


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
        self._io_lock = threading.RLock()

    def __enter__(self) -> "MultiOutputPlayer":
        try:
            for device in self.devices:
                stream = sd.RawOutputStream(
                    device=device,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                )
                self.streams.append(stream)
                stream.start()
            return self
        except Exception:
            self.close()
            raise

    def write(self, pcm: bytes) -> None:
        """写入所有健康流；某条流报错时立即剔除（防止继续写入死流导致原生崩溃）并抛出一次。"""
        with self._io_lock:
            self._write_locked(pcm)

    def _write_locked(self, pcm):
        if not self.streams:
            raise RuntimeError("输出流已关闭或没有可用设备")
        failures = []
        healthy = []
        for stream in self.streams:
            try:
                stream.write(pcm)
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
    ) -> None:
        if self.running.is_set():
            return
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("原声输出线程仍在关闭中")
        self.audio_queue = queue.Queue(maxsize=24)
        self.player = MultiOutputPlayer([output_device], sample_rate)
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
                    player.write(item)
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

    def start(self, path: Path, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.audio_queue = queue.Queue(maxsize=256)
        wav_file = wave.open(str(path), "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        def writer() -> None:
            try:
                while True:
                    chunk = self.audio_queue.get()
                    if chunk is None:
                        break
                    wav_file.writeframesraw(chunk)
            finally:
                wav_file.close()

        self.thread = threading.Thread(target=writer, name="pcm-wave-sink", daemon=True)
        self.thread.start()

    def write(self, pcm: bytes) -> None:
        try:
            self.audio_queue.put_nowait(pcm)
        except queue.Full:
            pass

    def stop(self) -> Path | None:
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(None)
            except queue.Empty:
                pass
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
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
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
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
            self.stop(create_mix=False)
            raise

    def stop(self, *, create_mix: bool = True) -> dict[str, Path]:
        self.system_capture.stop()
        microphone_path = self.microphone.stop()
        system_path = self.system_sink.stop()
        self.running.clear()
        if create_mix and microphone_path and system_path and microphone_path.exists() and system_path.exists():
            mix_mono_wav(microphone_path, system_path, self.paths["mixed"])
        return dict(self.paths)


class WavRecorder:
    """Continuously record one microphone to a mono 16-bit PCM WAV file."""

    def __init__(self) -> None:
        self.stream: sd.RawInputStream | None = None
        self.thread: threading.Thread | None = None
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
        self.running = threading.Event()
        self.path: Path | None = None
        self.sample_rate = 48000

    def start(self, device: int | None, path: Path, sample_rate: int = 48000) -> None:
        if self.running.is_set():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=128)
        wav_file = wave.open(str(path), "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        self.running.set()

        def writer() -> None:
            try:
                while True:
                    chunk = self.audio_queue.get()
                    if chunk is None:
                        break
                    wav_file.writeframesraw(chunk)
            finally:
                wav_file.close()

        self.thread = threading.Thread(target=writer, name="wav-recorder-writer", daemon=True)
        self.thread.start()

        def on_audio(indata, _frames, _time, _status) -> None:
            try:
                self.audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                # Prefer a short gap over blocking PortAudio's callback thread.
                pass

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
            self.running.clear()
            self.audio_queue.put_nowait(None)
            if self.thread is not None:
                self.thread.join(timeout=2.0)
                self.thread = None
            raise

    def stop(self) -> Path | None:
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
        if self.running.is_set():
            self.running.clear()
            try:
                self.audio_queue.put_nowait(None)
            except queue.Full:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(None)
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None
        return self.path
