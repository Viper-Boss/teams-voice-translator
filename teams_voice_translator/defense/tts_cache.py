"""合成音频的本地磁盘缓存。

为什么需要它：
  1. 原来的 `_pcm_cache` 只在内存里，关掉程序就没了，改一次设置也会被清空。
     用户问「重播为什么还要等」——因为没命中就真的又去请求服务器了。
  2. 讲稿朗读台要在答辩前把每一句都渲染好。现场万一网络抖动或者接口抽风，
     已经渲染过的句子必须还能出声。这是这个模块存在的主要理由。

键由「文本 + 所有会影响声音的参数」共同决定。换音色、换模型、改语速都会
得到不同的键 —— 否则你改完语速重播，听到的还是旧速度的音频，比不缓存更糟。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path

# 影响最终声音的参数。新增会影响音色/韵律的设置项时必须加进来，
# 否则缓存会把不同参数的结果混为一谈。
VOICE_AFFECTING_KEYS = (
    "tts_model",
    "voice",
    "tts_reference_audio",
    "tts_reference_audio_signature",
    "tts_rate",
    "tts_pitch",
    "tts_volume",
    "tts_instruction",
    "tts_language_hint",
    "tts_sample_rate",
    "tts_enable_ssml",
)

DEFAULT_LIMIT_MB = 1024
# 磁盘剩余低于这个数就停止写缓存。答辩当天把系统盘写满，
# 比丢掉缓存严重得多 —— Windows 在剩余空间见底时什么都可能出问题。
MIN_FREE_BYTES = 1024 * 1024 * 1024

# 两个子目录：auto 会被 LRU 淘汰，pinned 永远不淘汰。
# 讲稿朗读台预渲染的音频进 pinned —— 否则现场讲到一半，
# 提前渲染好的段可能已经被实时翻译产生的新缓存挤掉了，
# 「断网也能讲」就成了空话。
AUTO_DIR = "auto"
PINNED_DIR = "pinned"


def cache_key(text: str, settings) -> str:
    """文本 + 声音参数 → 稳定的 64 位十六进制键。

    参数按固定顺序拼接，不用 dict 的迭代顺序（那个不保证跨版本稳定）。
    """
    parts = [str(text or "").strip()]
    for key in VOICE_AFFECTING_KEYS:
        value = settings.get(key, "") if hasattr(settings, "get") else ""
        parts.append(f"{key}={value}")
    # 合成第二通道的开关 / 模型 / 音色**不进缓存键**：它只管现场发声，开了也不能让已经渲染好的
    # 讲稿、预案、快捷用语全部失效。第二通道用另一个音色合成的句子由引擎负责不写进磁盘缓存。
    blob = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


class TtsDiskCache:
    """按键存取裸 PCM16。写入是原子的，读失败一律当未命中。

    缓存永远是「有更好，没有也能跑」的东西：任何一步出错都不该让答辩中断，
    所以这里所有的异常都被吞掉并降级成未命中。
    """

    def __init__(self, base_dir, *, limit_mb: int = DEFAULT_LIMIT_MB) -> None:
        self.root = Path(base_dir) / "tts_cache"
        self.limit_bytes = max(16, int(limit_mb)) * 1024 * 1024
        self._lock = threading.RLock()
        self._enabled = True
        self._last_skip = ""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 目录建不出来（权限、磁盘满）就整体停用，不要每次读写都抛。
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _path_for(self, key: str, *, pinned: bool = False) -> Path:
        # 两级目录：单目录放几万个文件在 Windows 上会明显变慢。
        bucket = PINNED_DIR if pinned else AUTO_DIR
        return self.root / bucket / key[:2] / f"{key}.pcm"

    def _existing_path(self, key: str) -> Path | None:
        """pinned 优先：同一段音频两边都有时，用不会被淘汰的那份。"""
        for pinned in (True, False):
            path = self._path_for(key, pinned=pinned)
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
        return None

    def free_bytes(self) -> int:
        """缓存所在盘的剩余空间。取不到就返回一个大数，不因此停掉缓存。"""
        try:
            return shutil.disk_usage(self.root).free
        except OSError:
            return 1 << 62

    def size_of(self, key: str) -> int:
        """这份音频的字节数（只看文件大小，不读内容）；没有返回 0。"""
        if not self._enabled or not key:
            return 0
        path = self._existing_path(key)
        if path is None:
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def has(self, key: str) -> bool:
        if not self._enabled:
            return False
        return self._existing_path(key) is not None

    def get(self, key: str) -> bytes | None:
        if not self._enabled:
            return None
        path = self._existing_path(key)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if not data or len(data) % 2:
            # 半截文件（上次写到一半断电）当未命中，并顺手删掉。
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        try:
            os.utime(path, None)  # 记一次访问时间，LRU 用
        except OSError:
            pass
        return data

    def put(self, key: str, pcm: bytes, *, pinned: bool = False) -> bool:
        """写入。pinned=True 的条目永远不会被 LRU 淘汰。"""
        if not self._enabled or not pcm or len(pcm) % 2:
            return False
        # 盘快满了就不再写。宁可这一句没缓存，也不能把用户的系统盘塞爆。
        if self.free_bytes() - len(pcm) < MIN_FREE_BYTES:
            self._last_skip = "磁盘剩余空间不足，已暂停写入语音缓存"
            return False
        path = self._path_for(key, pinned=pinned)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再 replace：中途崩溃不会留下半截的 .pcm
            temp = path.with_suffix(".tmp")
            temp.write_bytes(pcm)
            temp.replace(path)
        except OSError:
            return False
        # 升级成 pinned 时，把 auto 里的旧副本删掉，别占两份空间。
        if pinned:
            try:
                self._path_for(key, pinned=False).unlink(missing_ok=True)
            except OSError:
                pass
        self._evict_if_needed()
        return True

    def discard(self, key: str) -> None:
        for pinned in (True, False):
            try:
                self._path_for(key, pinned=pinned).unlink(missing_ok=True)
            except OSError:
                pass

    def _entries(self, *, only: str = "") -> list[tuple[float, int, Path]]:
        """only 取 AUTO_DIR / PINNED_DIR 可限定子树，留空表示全部。"""
        root = self.root / only if only else self.root
        entries: list[tuple[float, int, Path]] = []
        try:
            for path in root.rglob("*.pcm"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                entries.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            return []
        return entries

    def total_bytes(self) -> int:
        return sum(size for _mtime, size, _path in self._entries())

    def _evict_if_needed(self) -> None:
        """超限时按最久未访问淘汰，**只淘汰 auto 子树**。

        pinned 里的是讲稿预渲染，淘汰掉就等于现场断网讲不下去，
        所以它只计入用量、不参与淘汰。
        """
        with self._lock:
            pinned = sum(size for _m, size, _p in self._entries(only=PINNED_DIR))
            entries = self._entries(only=AUTO_DIR)
            total = pinned + sum(size for _m, size, _p in entries)
            if total <= self.limit_bytes:
                return
            entries.sort(key=lambda item: item[0])  # 旧的在前
            for _mtime, size, path in entries:
                if total <= self.limit_bytes:
                    break
                try:
                    path.unlink()
                    total -= size
                except OSError:
                    continue

    def clear(self, *, include_pinned: bool = False) -> int:
        """清空缓存，返回删掉的文件数。

        默认保留 pinned —— 用户点「清理缓存」通常是想回收空间，
        不是想把提前渲染好的讲稿一起删掉。
        """
        removed = 0
        with self._lock:
            targets = self._entries() if include_pinned else self._entries(only=AUTO_DIR)
            for _mtime, _size, path in targets:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def stats(self) -> dict:
        auto = self._entries(only=AUTO_DIR)
        pinned = self._entries(only=PINNED_DIR)
        auto_bytes = sum(size for _m, size, _p in auto)
        pinned_bytes = sum(size for _m, size, _p in pinned)
        return {
            "enabled": self._enabled,
            "count": len(auto) + len(pinned),
            "bytes": auto_bytes + pinned_bytes,
            "auto_count": len(auto),
            "auto_bytes": auto_bytes,
            "pinned_count": len(pinned),
            "pinned_bytes": pinned_bytes,
            "limit_bytes": self.limit_bytes,
            "free_bytes": self.free_bytes(),
            "last_skip": self._last_skip,
        }


MANIFEST_FILE = "render_manifest.json"


class RenderManifest:
    """「这句话用这个音色最后一次渲染成了哪份音频」的登记簿。

    为什么需要它：缓存键包含语速、音量、语气指令等全部参数。答辩前渲染好讲稿，
    之后只要在设置里动一下音量或换个语气预设，键就全变了——重开软件一看全是
    「未渲染」，得再花钱渲染一遍。登记簿按「文本 + 语种 + 合成模型 + 音色」记下
    最后一次渲染的键：参数变了但音色没变时，旧音频照样能播（界面标成「旧参数」），
    要不要按新参数重渲染由你决定。换了音色则不算，免得一段话里混着两个声音。
    """

    def __init__(self, cache: TtsDiskCache, filename: str = MANIFEST_FILE) -> None:
        self.cache = cache
        self.path = cache.root / filename
        self._lock = threading.RLock()
        self._data: dict[str, str] | None = None

    @staticmethod
    def entry_id(scope: str, text: str, language: str, settings) -> str:
        getter = settings.get if hasattr(settings, "get") else (lambda _k, _d="": "")
        blob = "\x1f".join((
            str(scope), str(language or ""), str(getter("tts_model", "") or ""),
            str(getter("voice", "") or ""), str(text or "").strip(),
        )).encode("utf-8", errors="replace")
        return hashlib.sha256(blob).hexdigest()

    def _load(self) -> dict[str, str]:
        if self._data is None:
            import json

            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def get(self, entry: str) -> str:
        with self._lock:
            return self._load().get(entry, "")

    def remove(self, entry: str) -> None:
        with self._lock:
            data = self._load()
            if entry in data:
                del data[entry]
                self._save(data)

    def set(self, entry: str, key: str) -> None:
        with self._lock:
            data = self._load()
            if data.get(entry) == key:
                return
            data[entry] = key
            self._save(data)

    def _save(self, data: dict) -> None:
        import json

        with self._lock:
            if not self.cache.enabled:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.path.with_suffix(".tmp")
                temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                temp.replace(self.path)
            except OSError:
                pass
