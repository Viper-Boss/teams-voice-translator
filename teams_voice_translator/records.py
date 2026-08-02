from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time


def default_output_directory() -> Path:
    return Path.home() / "Documents" / "Teams Voice Translator"


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


@dataclass(frozen=True)
class SubtitleRecord:
    index: int
    at: datetime
    offset_seconds: float
    chinese: str
    english: str
    source: str


class SubtitleSession:
    """Lossless bilingual session cache with selectable final exports."""

    def __init__(self, cache_directory: Path | None = None) -> None:
        self.started_wall = datetime.now()
        self.started_clock = time.monotonic()
        self.records: list[SubtitleRecord] = []
        stamp = self.started_wall.strftime("%Y%m%d_%H%M%S_%f")
        cache_directory = cache_directory or Path(tempfile.gettempdir()) / "TeamsVoiceTranslator" / "captions"
        cache_directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_directory / f"caption_cache_{stamp}.jsonl"

    def add(self, chinese: str, english: str, source: str = "") -> SubtitleRecord:
        record = SubtitleRecord(
            index=len(self.records) + 1,
            at=datetime.now(),
            offset_seconds=max(0.0, time.monotonic() - self.started_clock),
            chinese=chinese.strip(),
            english=english.strip(),
            source=source.strip(),
        )
        self.records.append(record)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "index": record.index,
                        "time": record.at.isoformat(timespec="milliseconds"),
                        "offset_seconds": round(record.offset_seconds, 3),
                        "source": record.source,
                        "chinese": record.chinese,
                        "english": record.english,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return record

    def export(
        self,
        output_directory: Path,
        *,
        language: str = "both",
        file_format: str = "both",
        include_source: bool = True,
        include_time: bool = True,
    ) -> list[Path]:
        if language not in {"zh", "en", "both"}:
            raise ValueError(f"Unsupported subtitle language: {language}")
        if file_format not in {"srt", "txt", "vtt", "md", "jsonl", "both", "all"}:
            raise ValueError(f"Unsupported subtitle format: {file_format}")
        output_directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_wall.strftime("%Y%m%d_%H%M%S")
        suffix = {"zh": "中文", "en": "English", "both": "中英双语"}[language]
        base = output_directory / f"meeting_subtitles_{stamp}_{suffix}"
        paths: list[Path] = []
        if file_format in {"txt", "both", "all"}:
            txt_path = base.with_suffix(".txt")
            txt_path.write_text(
                self._render_txt(language, include_source=include_source, include_time=include_time),
                encoding="utf-8",
            )
            paths.append(txt_path)
        if file_format in {"srt", "both", "all"}:
            srt_path = base.with_suffix(".srt")
            srt_path.write_text(
                self._render_srt(language, include_source=include_source), encoding="utf-8"
            )
            paths.append(srt_path)
        if file_format in {"vtt", "all"}:
            vtt_path = base.with_suffix(".vtt")
            vtt_path.write_text(
                self._render_vtt(language, include_source=include_source), encoding="utf-8"
            )
            paths.append(vtt_path)
        if file_format in {"md", "all"}:
            md_path = base.with_suffix(".md")
            md_path.write_text(
                self._render_markdown(
                    language, include_source=include_source, include_time=include_time
                ),
                encoding="utf-8",
            )
            paths.append(md_path)
        if file_format in {"jsonl", "all"}:
            jsonl_path = base.with_suffix(".jsonl")
            jsonl_path.write_text(
                self._render_jsonl(
                    language, include_source=include_source, include_time=include_time
                ),
                encoding="utf-8",
            )
            paths.append(jsonl_path)
        return paths

    def _selected_lines(self, record: SubtitleRecord, language: str) -> list[str]:
        if language == "zh":
            return [record.chinese] if record.chinese else []
        if language == "en":
            return [record.english] if record.english else []
        return [line for line in (record.chinese, record.english) if line]

    def _render_txt(self, language: str, *, include_source: bool, include_time: bool) -> str:
        chunks: list[str] = []
        for record in self.records:
            lines = self._selected_lines(record, language)
            if not lines:
                continue
            labels: list[str] = []
            if include_time:
                labels.append(f"{record.at:%Y-%m-%d %H:%M:%S}")
            if include_source and record.source:
                labels.append(record.source)
            if labels:
                chunks.append(f"[{' · '.join(labels)}]")
            if language in {"zh", "both"} and record.chinese:
                chunks.append(f"中文：{record.chinese}")
            if language in {"en", "both"} and record.english:
                chunks.append(f"English: {record.english}")
            chunks.append("")
        return "\n".join(chunks)

    def _render_srt(self, language: str, *, include_source: bool = False) -> str:
        chunks: list[str] = []
        export_index = 0
        for index, record in enumerate(self.records):
            lines = self._selected_lines(record, language)
            if not lines:
                continue
            export_index += 1
            estimated = min(8.0, max(2.5, sum(len(line) for line in lines) / 12.0))
            end = record.offset_seconds + estimated
            if index + 1 < len(self.records):
                next_start = self.records[index + 1].offset_seconds
                if next_start > record.offset_seconds + 0.5:
                    end = min(end, next_start - 0.05)
            chunks.extend(
                [
                    str(export_index),
                    f"{srt_timestamp(record.offset_seconds)} --> {srt_timestamp(end)}",
                    *([f"[{record.source}]"] if include_source and record.source else []),
                    *lines,
                    "",
                ]
            )
        return "\n".join(chunks)

    def _render_vtt(self, language: str, *, include_source: bool) -> str:
        chunks: list[str] = ["WEBVTT", ""]
        for index, record in enumerate(self.records):
            lines = self._selected_lines(record, language)
            if not lines:
                continue
            estimated = min(8.0, max(2.5, sum(len(line) for line in lines) / 12.0))
            end = record.offset_seconds + estimated
            if index + 1 < len(self.records):
                next_start = self.records[index + 1].offset_seconds
                if next_start > record.offset_seconds + 0.5:
                    end = min(end, next_start - 0.05)
            chunks.extend(
                [
                    f"{srt_timestamp(record.offset_seconds).replace(',', '.')} --> "
                    f"{srt_timestamp(end).replace(',', '.')}",
                    *([f"[{record.source}]"] if include_source and record.source else []),
                    *lines,
                    "",
                ]
            )
        return "\n".join(chunks)

    def _render_markdown(
        self, language: str, *, include_source: bool, include_time: bool
    ) -> str:
        title = f"# 会议记录 {self.started_wall:%Y-%m-%d %H:%M}\n"
        blocks: list[str] = [title]
        for record in self.records:
            lines = self._selected_lines(record, language)
            if not lines:
                continue
            heading: list[str] = []
            if include_time:
                heading.append(f"{record.at:%H:%M:%S}")
            if include_source and record.source:
                heading.append(record.source)
            if heading:
                blocks.append(f"## {' · '.join(heading)}")
            if language in {"zh", "both"} and record.chinese:
                blocks.append(f"- 中文：{record.chinese}")
            if language in {"en", "both"} and record.english:
                blocks.append(f"- English: {record.english}")
            blocks.append("")
        return "\n".join(blocks)

    def _render_jsonl(
        self, language: str, *, include_source: bool, include_time: bool
    ) -> str:
        rows: list[str] = []
        for record in self.records:
            row: dict[str, object] = {
                "index": record.index,
                "offset_seconds": round(record.offset_seconds, 3),
            }
            if include_time:
                row["time"] = record.at.isoformat(timespec="milliseconds")
            if include_source:
                row["source"] = record.source
            if language in {"zh", "both"}:
                row["chinese"] = record.chinese
            if language in {"en", "both"}:
                row["english"] = record.english
            rows.append(json.dumps(row, ensure_ascii=False))
        return "\n".join(rows) + ("\n" if rows else "")

    @staticmethod
    def discover_recoverable(
        cache_directory: Path | None = None, *, exclude: Path | None = None
    ) -> list[Path]:
        root = cache_directory or Path(tempfile.gettempdir()) / "TeamsVoiceTranslator" / "captions"
        if not root.exists():
            return []
        candidates = []
        for path in root.glob("caption_cache_*.jsonl"):
            if exclude is not None and path == exclude:
                continue
            try:
                if path.stat().st_size > 0:
                    candidates.append(path)
            except OSError:
                pass
        return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)

    def import_cache(self, path: Path) -> int:
        imported = 0
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0
        for line in lines:
            try:
                raw = json.loads(line)
                record = SubtitleRecord(
                    index=len(self.records) + 1,
                    at=datetime.fromisoformat(str(raw["time"])),
                    offset_seconds=float(raw.get("offset_seconds", 0.0)),
                    chinese=str(raw.get("chinese", "")).strip(),
                    english=str(raw.get("english", "")).strip(),
                    source=str(raw.get("source", "")).strip(),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self.records.append(record)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "index": record.index,
                            "time": record.at.isoformat(timespec="milliseconds"),
                            "offset_seconds": round(record.offset_seconds, 3),
                            "source": record.source,
                            "chinese": record.chinese,
                            "english": record.english,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            imported += 1
        if imported:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return imported

    def normalize_recovered_timeline(self) -> None:
        """Order recovered sessions and make future records continue their wall-clock timeline."""
        if not self.records:
            return
        ordered = sorted(self.records, key=lambda record: record.at)
        first = ordered[0].at
        self.records = [
            SubtitleRecord(
                index=index,
                at=record.at,
                offset_seconds=max(0.0, (record.at - first).total_seconds()),
                chinese=record.chinese,
                english=record.english,
                source=record.source,
            )
            for index, record in enumerate(ordered, start=1)
        ]
        self.started_wall = first
        elapsed_wall = max(0.0, (datetime.now() - first).total_seconds())
        self.started_clock = time.monotonic() - elapsed_wall
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(
                    json.dumps(
                        {
                            "index": record.index,
                            "time": record.at.isoformat(timespec="milliseconds"),
                            "offset_seconds": round(record.offset_seconds, 3),
                            "source": record.source,
                            "chinese": record.chinese,
                            "english": record.english,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def cleanup(self) -> None:
        try:
            self.cache_path.unlink(missing_ok=True)
        except OSError:
            pass
