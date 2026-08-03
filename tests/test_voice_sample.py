import math
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import imageio_ffmpeg

from teams_voice_translator.voice_sample import VoiceSampleError, normalized_voice_sample


def write_tone(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frames = bytearray()
        for index in range(int(seconds * sample_rate)):
            value = int(9000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        audio.writeframes(bytes(frames))


class VoiceSampleTests(unittest.TestCase):
    def test_iphone_m4a_is_converted_to_standard_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_wav = root / "iphone-source.wav"
            source_m4a = root / "iphone-recording.m4a"
            write_tone(source_wav, 6.0)
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_wav),
                    "-c:a",
                    "aac",
                    str(source_m4a),
                ],
                check=True,
                capture_output=True,
            )

            with normalized_voice_sample(source_m4a, max_seconds=20) as normalized:
                self.assertTrue(normalized.exists())
                with wave.open(str(normalized), "rb") as audio:
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getsampwidth(), 2)
                    self.assertEqual(audio.getframerate(), 24000)
                    self.assertGreaterEqual(audio.getnframes() / audio.getframerate(), 5.9)
            self.assertFalse(normalized.exists())

    def test_sample_shorter_than_five_seconds_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "short.wav"
            write_tone(source, 1.0)
            with self.assertRaises(VoiceSampleError):
                with normalized_voice_sample(source):
                    pass


if __name__ == "__main__":
    unittest.main()
