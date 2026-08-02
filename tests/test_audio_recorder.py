import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from teams_voice_translator.audio import (
    DirectAudioBridge,
    Pcm16Resampler,
    WavRecorder,
    mix_mono_wav,
    pcm16_to_mono,
)


class FakeInputStream:
    def __init__(self, *, callback, **_kwargs):
        self.callback = callback

    def start(self):
        self.callback(b"\x01\x00" * 160, 160, None, None)

    def stop(self):
        pass

    def close(self):
        pass


class FakeCapture:
    def __init__(self, _device, callback, **_kwargs):
        self.callback = callback

    def start(self):
        self.callback(b"\x02\x00" * 32)

    def stop(self):
        pass


class FakePlayer:
    def __init__(self, _devices, _sample_rate):
        self.written = []

    def __enter__(self):
        return self

    def write(self, chunk):
        self.written.append(chunk)

    def close(self):
        pass


class RecorderTests(unittest.TestCase):
    def test_direct_bridge_taps_audio_without_blocking_output(self):
        tapped = []
        bridge = DirectAudioBridge()
        with (
            patch("teams_voice_translator.audio.MicrophoneCapture", FakeCapture),
            patch("teams_voice_translator.audio.MultiOutputPlayer", FakePlayer),
        ):
            bridge.start(None, None, 48000, tap=tapped.append)
            time.sleep(0.02)
            bridge.stop()
        self.assertEqual(tapped, [b"\x02\x00" * 32])

    def test_direct_audio_can_be_resampled_for_asr(self):
        source = b"\x01\x00" * 480
        converted = Pcm16Resampler(48000, 16000).process(source)
        self.assertEqual(len(converted), 160 * 2)

    def test_stereo_loopback_is_downmixed_to_mono(self):
        stereo = (
            int(1000).to_bytes(2, "little", signed=True)
            + int(3000).to_bytes(2, "little", signed=True)
        ) * 4
        mono = pcm16_to_mono(stereo, 2)
        self.assertEqual(len(mono), 8)
        self.assertEqual(int.from_bytes(mono[:2], "little", signed=True), 2000)

    def test_two_tracks_can_be_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "mic.wav"
            second = root / "system.wav"
            mixed = root / "mixed.wav"
            for path, sample in ((first, 1000), (second, 2000)):
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(sample.to_bytes(2, "little", signed=True) * 20)
            mix_mono_wav(first, second, mixed)
            with wave.open(str(mixed), "rb") as handle:
                self.assertEqual(handle.getframerate(), 16000)
                sample = int.from_bytes(handle.readframes(1), "little", signed=True)
            self.assertEqual(sample, 3000)

    def test_recorder_writes_valid_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recording.wav"
            recorder = WavRecorder()
            with patch("teams_voice_translator.audio.sd.RawInputStream", FakeInputStream):
                recorder.start(None, path, 16000)
                time.sleep(0.02)
                saved = recorder.stop()
            self.assertEqual(saved, path)
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertEqual(wav_file.getnframes(), 160)


if __name__ == "__main__":
    unittest.main()
