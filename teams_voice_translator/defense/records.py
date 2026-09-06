from __future__ import annotations

import time
from pathlib import Path

SUBTITLE_DIR = "字幕"
AUDIO_DIR = "录音"


def default_base_dir() -> Path:
    return Path.home() / "Documents" / "DefenseMode"


def new_meeting_folder(base_dir: Path, started_at: float | None = None) -> Path:
    """每次会议一个文件夹：答辩_YYYYMMDD_HHMMSS，内含 录音/ 与 字幕/ 两个子目录。"""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at or time.time()))
    folder = Path(base_dir) / f"答辩_{stamp}"
    (folder / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    (folder / SUBTITLE_DIR).mkdir(parents=True, exist_ok=True)
    return folder


def _fmt_clock(value: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(value))


def write_transcripts(folder: Path, segments) -> dict[str, Path]:
    """把整场记录写成四个文件：会议全记录.md + 中/英/双语字幕 TXT。

    ``segments`` 的元素需要具备 role/source/translation/at/latency_ms 属性
    （即 pipeline.SegmentRecord）。"我"：source=中文、translation=英文；
    "评委"：source=英文、translation=中文。
    """
    items = list(segments)
    subtitle_dir = folder / SUBTITLE_DIR
    zh_lines: list[str] = []
    en_lines: list[str] = []
    both_lines: list[str] = []
    md_lines: list[str] = [
        "# 答辩会议全记录",
        "",
        f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 记录条数：{len(items)} 条（我 + 评委）",
        "",
        "| 时间 | 谁 | 中文 | 英文 | 延迟 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for seg in items:
        who = "我" if seg.role == "me" else "评委"
        clock = _fmt_clock(seg.at)
        if seg.role == "me":
            chinese = seg.source
            english = seg.translation
        else:
            chinese = seg.translation
            english = seg.source
        chinese = (chinese or "").strip()
        english = (english or "").strip()
        latency = f"{seg.latency_ms / 1000:.1f}s" if seg.latency_ms else ""
        zh_lines.append(f"[{clock}] {who}：{chinese}")
        en_lines.append(f"[{clock}] {who}: {english}")
        both_lines.append(f"[{clock}] {who}")
        both_lines.append(f"  中：{chinese}")
        both_lines.append(f"  英：{english}")
        md_lines.append(
            f"| {clock} | {who} | {chinese.replace('|', '/')} | {english.replace('|', '/')} | {latency} |"
        )

    paths = {
        "中文字幕": subtitle_dir / "中文字幕.txt",
        "英文字幕": subtitle_dir / "英文字幕.txt",
        "双语字幕": subtitle_dir / "双语字幕.txt",
        "会议全记录": folder / "会议全记录.md",
    }
    paths["中文字幕"].write_text("\n".join(zh_lines) + "\n", encoding="utf-8-sig")
    paths["英文字幕"].write_text("\n".join(en_lines) + "\n", encoding="utf-8-sig")
    paths["双语字幕"].write_text("\n".join(both_lines) + "\n", encoding="utf-8-sig")
    paths["会议全记录"].write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return paths
