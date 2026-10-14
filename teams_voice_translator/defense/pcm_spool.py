"""整场合成语音的落盘暂存区（append-only）。

为什么需要它：原来整场录音是一个 `bytearray`，`_store_pcm` 每次 `.extend()`，
整场会议都不释放。一小时的会按实际发声 25 分钟算就是 ~72MB 常驻；
导出时还要 `bytes(bytearray)` 完整复制一份，峰值直接翻倍。
bytearray 扩容时还会临时占用两倍空间 —— 答辩进行到一半内存抖一下，
是最不该发生的事。

改法：音频**边产生边写文件**，内存里只留一个小缓冲（默认 4MB）攒够了再刷盘，
导出时从文件分块流式写 WAV，全程不做整份复制。

失败策略：录音是「会后回听」的附加功能，**不是答辩跑起来的必要条件**。
所以这里任何磁盘错误都只是让录音功能降级或关闭，绝不往上抛。
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path

DEFAULT_FLUSH_BYTES = 4 * 1024 * 1024
# 分块读写，别为了导出一次性把整场音频读进内存。
COPY_CHUNK = 1 * 1024 * 1024


class PcmSpool:
    """顺序追加 PCM16，记录每段的字节区间；落盘失败则自动停用。"""

    def __init__(self, path, *, flush_bytes: int = DEFAULT_FLUSH_BYTES) -> None:
        self.path = Path(path)
        # 下限 4KB：避免每来一小段就刷一次盘，但也别把调用方给的值整个吃掉
        # （原来写的是 max(64KB, ...)，小于 64KB 的设置会被静默忽略）。
        self.flush_bytes = max(4 * 1024, int(flush_bytes))
        self._lock = threading.RLock()
        self._buffer = bytearray()
        self._written = 0          # 已落盘字节数
        self._handle = None
        self._enabled = True
        self._error = ""
        # 这里只准备路径，不提前长期占用文件句柄。引擎可能在设置页或自检时
        # 被创建，却一直没有真正开始合成；Windows 上若此时临时目录先清理，
        # 一个空的 meeting_audio.pcm 句柄也足以导致 WinError 32。
        # 第一次需要落盘时再打开，既不改变录音内容，也让空闲引擎可安全销毁。
        self._prepare_empty_file()

    # ------------------------------------------------------------------ 生命周期
    def _prepare_empty_file(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            self._enabled = False
            self._error = str(exc)
            self._handle = None

    def _ensure_open_locked(self) -> bool:
        if self._handle is not None:
            return True
        if not self._enabled:
            return False
        try:
            self._handle = open(self.path, "wb")
            return True
        except OSError as exc:
            self._disable(exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def error(self) -> str:
        return self._error

    def _disable(self, exc: Exception) -> None:
        """写盘出问题就整体停用，并丢掉缓冲——绝不让录音把答辩拖下水。"""
        self._enabled = False
        self._error = " ".join(str(exc).split())[:200]
        self._buffer = bytearray()
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ 写入
    def append(self, pcm: bytes) -> tuple[int, int]:
        """追加一段，返回它在整场音频里的 [起, 止] 字节偏移。

        停用状态下返回 (total, total) —— 调用方拿到一个空区间，
        不需要为「录音坏了」写分支。
        """
        with self._lock:
            if not self._enabled or not pcm:
                total = self.total_bytes()
                return total, total
            start = self._written + len(self._buffer)
            self._buffer.extend(pcm)
            if len(self._buffer) >= self.flush_bytes:
                self._flush_locked()
            return start, start + len(pcm)

    def _flush_locked(self) -> None:
        if not self._enabled or not self._buffer:
            return
        if not self._ensure_open_locked():
            return
        try:
            self._handle.write(self._buffer)
            self._handle.flush()
        except OSError as exc:
            self._disable(exc)
            return
        self._written += len(self._buffer)
        self._buffer = bytearray()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    # ------------------------------------------------------------------ 读取
    def total_bytes(self) -> int:
        with self._lock:
            return self._written + len(self._buffer)

    def seconds(self, sample_rate: int) -> float:
        return self.total_bytes() / max(int(sample_rate) * 2, 1)

    def write_wav(self, target, sample_rate: int) -> int:
        """把整场音频流式写成 WAV，返回写入的字节数。

        分块搬运，不把整场读进内存 —— 一小时的会可能上百 MB。
        """
        with self._lock:
            self._flush_locked()
            total = self.total_bytes()
            if not total:
                return 0
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with wave.open(str(target), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(int(sample_rate))
                if self._written and self.path.is_file():
                    with open(self.path, "rb") as source:
                        while True:
                            chunk = source.read(COPY_CHUNK)
                            if not chunk:
                                break
                            out.writeframes(chunk)
                            copied += len(chunk)
                if self._buffer:
                    out.writeframes(bytes(self._buffer))
                    copied += len(self._buffer)
            return copied

    def read_range(self, start: int, end: int) -> bytes:
        """取某一段音频。越界自动裁剪，读不到返回空。"""
        with self._lock:
            self._flush_locked()
            total = self.total_bytes()
            start = max(0, min(int(start), total))
            end = max(start, min(int(end), total))
            if start == end or not self._enabled:
                return b""
            try:
                with open(self.path, "rb") as source:
                    source.seek(start)
                    return source.read(end - start)
            except OSError:
                return b""

    # ------------------------------------------------------------------ 清理
    def reset(self) -> None:
        """丢弃本场录音，重新开一个空文件。"""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            self._buffer = bytearray()
            self._written = 0
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self._enabled = True
            self._error = ""
            self._prepare_empty_file()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            handle, self._handle = self._handle, None
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "bytes": self.total_bytes(),
                "buffered": len(self._buffer),
                "error": self._error,
            }
