from __future__ import annotations

import audioop
import queue
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


class MicrophoneCapture:
    def __init__(
        self,
        device: int | None,
        callback: Callable[[bytes], None],
        *,
        sample_rate: int = 16000,
        block_ms: int = 100,
    ) -> None:
        self.device = device
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.stream: sd.RawInputStream | None = None

    def start(self) -> None:
        blocksize = int(self.sample_rate * self.block_ms / 1000)

        def on_audio(indata, _frames, _time, status) -> None:
            if status:
                # PortAudio status is diagnostic; dropping one block is safer than
                # doing logging or UI work inside the real-time callback.
                pass
            self.callback(bytes(indata))

        self.stream = sd.RawInputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=on_audio,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


class MultiOutputPlayer:
    def __init__(self, devices: list[int | None], sample_rate: int) -> None:
        unique: list[int | None] = []
        for device in devices:
            if device not in unique:
                unique.append(device)
        self.devices = unique
        self.sample_rate = sample_rate
        self.streams: list[sd.RawOutputStream] = []

    def __enter__(self) -> "MultiOutputPlayer":
        try:
            for device in self.devices:
                stream = sd.RawOutputStream(
                    device=device,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                )
                stream.start()
                self.streams.append(stream)
            return self
        except Exception:
            self.close()
            raise

    def write(self, pcm: bytes) -> None:
        for stream in self.streams:
            stream.write(pcm)

    def close(self) -> None:
        for stream in self.streams:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self.streams.clear()

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
        self.audio_queue = queue.Queue(maxsize=24)
        self.player = MultiOutputPlayer([output_device], sample_rate)
        self.player.__enter__()
        self.running.set()

        def write_loop() -> None:
            assert self.player is not None
            while self.running.is_set():
                item = self.audio_queue.get()
                if item is None:
                    break
                self.player.write(item)

        self.thread = threading.Thread(target=write_loop, name="direct-audio-output", daemon=True)
        self.thread.start()

        def enqueue(chunk: bytes) -> None:
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
            self.thread = None
        if self.player is not None:
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
