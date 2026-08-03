from __future__ import annotations

import subprocess
import tempfile
import wave
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


@contextmanager
def normalized_voice_sample(
    source: str | Path,
    *,
    max_seconds: float = 20.0,
) -> Iterator[Path]:
    """Convert a local sample to a short, predictable PCM WAV for enrollment."""

    source_path = Path(source).expanduser().resolve()
    _validate_source(source_path)
    max_seconds = min(30.0, max(5.0, float(max_seconds)))

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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.decode("utf-8", errors="replace").strip()
            raise VoiceSampleError(f"样音转换失败：{details or exc}") from exc
        except OSError as exc:
            raise VoiceSampleError(f"无法启动内置音频转换器：{exc}") from exc

        if not output.is_file() or output.stat().st_size == 0:
            raise VoiceSampleError("音频转换没有生成有效的 WAV 文件")
        duration = _wav_duration_seconds(output)
        if duration < 5.0:
            raise VoiceSampleError(
                f"有效样音只有 {duration:.1f} 秒；请提供至少 5 秒、建议 10–20 秒的连续清晰人声"
            )
        yield output
