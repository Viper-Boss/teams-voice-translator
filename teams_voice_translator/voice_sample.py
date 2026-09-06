from __future__ import annotations

import subprocess
import tempfile
import wave
import math
import audioop
import re
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import imageio_ffmpeg


SUPPORTED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".mp4",
    ".caf",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}


class VoiceSampleError(RuntimeError):
    pass


def inspect_voice_sample(path: str | Path) -> str:
    """Local signal checks, without uploading audio or claiming speaker quality."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise VoiceSampleError("样音检查需要单声道 PCM16 WAV")
        pcm = handle.readframes(handle.getnframes())
    if not pcm or audioop.max(pcm, 2) < 16:
        raise VoiceSampleError("样音几乎无声，请检查麦克风后重新录制")
    level = 20 * math.log10(max(1, audioop.rms(pcm, 2)) / 32768)
    samples = array("h", pcm)
    clipped = sum(abs(value) >= 32760 for value in samples) / len(samples)
    notes = []
    if level < -35:
        notes.append("样音音量偏低，建议靠近麦克风后重录")
    if clipped > 0.005:
        notes.append("样音存在削波，建议降低麦克风增益后重录")
    return "；".join(notes) or "样音音量与削波检查通过（不代表相似度已验证）"


def _validate_source(path: Path) -> None:
    if not path.is_file():
        raise VoiceSampleError(f"找不到本地样音文件：{path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = "、".join(sorted(suffix.lstrip(".").upper() for suffix in SUPPORTED_AUDIO_SUFFIXES))
        raise VoiceSampleError(f"不支持 {path.suffix or '无扩展名'} 文件，请选择：{supported}")
    if path.stat().st_size <= 0:
        raise VoiceSampleError("样音文件为空")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise VoiceSampleError("样音文件超过 100 MB，请先裁剪后再试")


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
    except (OSError, wave.Error) as exc:
        raise VoiceSampleError(f"无法读取转换后的 WAV：{exc}") from exc
    return frames / rate if rate else 0.0


def voice_sample_duration(source: str | Path) -> float:
    """Read original duration without truncating, denoising or changing gain."""
    path = Path(source).expanduser().resolve()
    _validate_source(path)
    if path.suffix.lower() == ".wav":
        try:
            return _wav_duration_seconds(path)
        except VoiceSampleError:
            pass  # e.g. float WAV supported by FFmpeg but not wave
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path),
             "-map", "0:a:0", "-t", "0", "-f", "null", "-"],
            capture_output=True, timeout=15, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceSampleError("无法读取音频时长，请检查文件或转换为 WAV") from exc
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr.decode(errors="replace"))
    if not match:
        raise VoiceSampleError("音频没有有效时长信息，请转换为 WAV 后再试")
    hours, minutes, seconds = map(float, match.groups())
    return hours * 3600 + minutes * 60 + seconds


@contextmanager
def normalized_voice_sample(
    source: str | Path,
    *,
    max_seconds: float = 20.0,
    min_seconds: float = 5.0,
) -> Iterator[Path]:
    """Convert a local sample to a short, predictable PCM WAV for enrollment."""

    source_path = Path(source).expanduser().resolve()
    _validate_source(source_path)
    max_seconds = min(30.0, max(float(min_seconds), float(max_seconds)))

    with tempfile.TemporaryDirectory(prefix="teams-voice-sample-") as temp_dir:
        output = Path(temp_dir) / "voice-sample.wav"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            f"{max_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.decode("utf-8", errors="replace").strip()
            raise VoiceSampleError(f"样音转换失败：{details or exc}") from exc
        except OSError as exc:
            raise VoiceSampleError(f"无法启动内置音频转换器：{exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise VoiceSampleError("样音转换超时，请改用较短的本地音频") from exc

        if not output.is_file() or output.stat().st_size == 0:
            raise VoiceSampleError("音频转换没有生成有效的 WAV 文件")
        duration = _wav_duration_seconds(output)
        if duration + 1 / 24000 < min_seconds:
            raise VoiceSampleError(
                f"有效样音只有 {duration:.1f} 秒；请提供至少 {min_seconds:g} 秒的连续清晰人声"
            )
        yield output
