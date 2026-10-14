"""容量与稳定性：一小时会议不能把内存或磁盘撑爆。

三条底线：
  1. 整场录音不在内存里累积，边产生边落盘；
  2. 内存缓存按**字节**限，不按条数；
  3. 讲稿预渲染的音频**永不被 LRU 淘汰**，否则「断网也能讲」是空话。

所有失败路径都必须降级而不是抛 —— 缓存和录音都是附加功能，
不能因为它们出问题把答辩带崩。
"""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock

from teams_voice_translator.defense.pcm_spool import PcmSpool
from teams_voice_translator.defense.pipeline import CachedUtterance, DefenseEngine
from teams_voice_translator.defense.tts_cache import (
    AUTO_DIR,
    MIN_FREE_BYTES,
    PINNED_DIR,
    TtsDiskCache,
)

SAMPLE_RATE = 24000
ONE_SECOND = b"\x00\x00" * SAMPLE_RATE  # 48000 字节


class SpoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "meeting.pcm"

    def spool(self, **kwargs):
        spool = PcmSpool(self.path, **kwargs)
        self.addCleanup(spool.close)
        return spool

    def test_offsets_are_contiguous(self):
        spool = self.spool()
        first = spool.append(b"\x01\x02" * 10)
        second = spool.append(b"\x03\x04" * 5)
        self.assertEqual(first, (0, 20))
        self.assertEqual(second, (20, 30))
        self.assertEqual(spool.total_bytes(), 30)

    def test_audio_lands_on_disk_not_in_memory(self):
        """这是整件事的重点：一小时的会不能全挂在内存里。"""
        spool = self.spool(flush_bytes=64 * 1024)
        for _ in range(10):
            spool.append(ONE_SECOND)  # 共 480KB，远超 64KB 阈值
        self.assertGreater(self.path.stat().st_size, 0)
        # 内存里只剩不足一个阈值的尾巴
        self.assertLess(spool.stats()["buffered"], 64 * 1024)
        self.assertEqual(spool.total_bytes(), 10 * len(ONE_SECOND))

    def test_small_writes_stay_buffered_until_the_threshold(self):
        spool = self.spool(flush_bytes=1024 * 1024)
        spool.append(b"\x00\x00" * 100)
        self.assertEqual(spool.stats()["buffered"], 200)

    def test_seconds_matches_pcm_length(self):
        spool = self.spool()
        spool.append(ONE_SECOND * 3)
        self.assertAlmostEqual(spool.seconds(SAMPLE_RATE), 3.0, places=3)

    def test_export_writes_a_playable_wav(self):
        spool = self.spool(flush_bytes=64 * 1024)
        for _ in range(5):
            spool.append(ONE_SECOND)
        target = Path(self.tmp.name) / "out.wav"
        written = spool.write_wav(target, SAMPLE_RATE)
        self.assertEqual(written, 5 * len(ONE_SECOND))
        with wave.open(str(target)) as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), SAMPLE_RATE)
            self.assertEqual(handle.getnframes(), 5 * SAMPLE_RATE)

    def test_export_includes_the_unflushed_tail(self):
        """最后几秒还在缓冲里，导出时不能漏掉。"""
        spool = self.spool(flush_bytes=1024 * 1024)
        spool.append(ONE_SECOND)
        target = Path(self.tmp.name) / "tail.wav"
        self.assertEqual(spool.write_wav(target, SAMPLE_RATE), len(ONE_SECOND))

    def test_export_of_empty_spool_writes_nothing(self):
        spool = self.spool()
        self.assertEqual(spool.write_wav(Path(self.tmp.name) / "empty.wav", SAMPLE_RATE), 0)

    def test_read_range_clamps_instead_of_raising(self):
        spool = self.spool(flush_bytes=8 * 1024)
        spool.append(b"\x01\x02" * 100)
        self.assertEqual(len(spool.read_range(0, 50)), 50)
        self.assertEqual(len(spool.read_range(150, 9999)), 50)
        self.assertEqual(spool.read_range(999, 1000), b"")
        self.assertEqual(spool.read_range(30, 10), b"")

    def test_reset_starts_a_fresh_recording(self):
        spool = self.spool(flush_bytes=8 * 1024)
        spool.append(ONE_SECOND)
        spool.reset()
        self.assertEqual(spool.total_bytes(), 0)
        self.assertTrue(spool.enabled)
        spool.append(b"\x00\x00" * 4)
        self.assertEqual(spool.total_bytes(), 8)

    def test_unwritable_path_degrades_instead_of_raising(self):
        """录音是会后回听的附加功能，不能因为它把答辩带崩。"""
        blocker = Path(self.tmp.name) / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        spool = PcmSpool(blocker / "nested" / "meeting.pcm")
        self.addCleanup(spool.close)
        self.assertFalse(spool.enabled)
        self.assertEqual(spool.append(ONE_SECOND), (0, 0))
        self.assertEqual(spool.total_bytes(), 0)
        self.assertTrue(spool.error)

    def test_write_failure_mid_session_disables_and_drops_buffer(self):
        spool = self.spool(flush_bytes=8 * 1024)
        spool.append(b"\x00\x00" * 8)

        class _Broken:
            def write(self, _data):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

            def close(self):
                pass

        spool._handle = _Broken()
        spool.append(ONE_SECOND)  # 触发刷盘失败
        self.assertFalse(spool.enabled)
        self.assertIn("disk full", spool.error)
        # 停用后继续 append 不抛
        self.assertEqual(spool.append(ONE_SECOND), (spool.total_bytes(), spool.total_bytes()))


class PinnedCacheTests(unittest.TestCase):
    """讲稿预渲染必须扛得住 LRU。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = TtsDiskCache(Path(self.tmp.name))

    def test_pinned_and_auto_live_in_separate_subtrees(self):
        self.cache.put("a" * 64, b"\x00\x00" * 4, pinned=True)
        self.cache.put("b" * 64, b"\x00\x00" * 4)
        self.assertTrue((self.cache.root / PINNED_DIR).exists())
        self.assertTrue((self.cache.root / AUTO_DIR).exists())

    def test_both_kinds_are_readable(self):
        self.cache.put("a" * 64, b"\x01\x01" * 4, pinned=True)
        self.cache.put("b" * 64, b"\x02\x02" * 4)
        self.assertEqual(self.cache.get("a" * 64), b"\x01\x01" * 4)
        self.assertEqual(self.cache.get("b" * 64), b"\x02\x02" * 4)

    def test_eviction_never_touches_pinned(self):
        """核心保证：现场讲到一半，提前渲染好的段不能已经被挤掉。"""
        import os, time

        payload = b"\x00\x00" * 512  # 1 KiB
        now = time.time()
        for index in range(8):
            key = f"p{index:063d}"
            self.cache.put(key, payload, pinned=True)
            os.utime(self.cache._path_for(key, pinned=True), (now - 9999, now - 9999))
        for index in range(20):
            key = f"a{index:063d}"
            self.cache.put(key, payload)
            os.utime(self.cache._path_for(key), (now + index, now + index))
        self.cache.limit_bytes = 6 * 1024
        self.cache._evict_if_needed()
        stats = self.cache.stats()
        # pinned 一个都没少，哪怕它们的 mtime 最老
        self.assertEqual(stats["pinned_count"], 8)
        self.assertLess(stats["auto_count"], 20)

    def test_pinned_counts_toward_usage(self):
        self.cache.put("a" * 64, b"\x00\x00" * 512, pinned=True)
        stats = self.cache.stats()
        self.assertEqual(stats["pinned_bytes"], 1024)
        self.assertEqual(stats["bytes"], 1024)

    def test_upgrading_to_pinned_drops_the_auto_copy(self):
        key = "c" * 64
        self.cache.put(key, b"\x00\x00" * 8)
        self.cache.put(key, b"\x00\x00" * 8, pinned=True)
        self.assertFalse(self.cache._path_for(key, pinned=False).exists())
        self.assertTrue(self.cache._path_for(key, pinned=True).exists())

    def test_clear_keeps_pinned_by_default(self):
        """点「清理缓存」是想回收空间，不是想删掉提前渲染的讲稿。"""
        self.cache.put("a" * 64, b"\x00\x00" * 4, pinned=True)
        self.cache.put("b" * 64, b"\x00\x00" * 4)
        self.assertEqual(self.cache.clear(), 1)
        self.assertTrue(self.cache.has("a" * 64))
        self.assertFalse(self.cache.has("b" * 64))
        self.assertEqual(self.cache.clear(include_pinned=True), 1)
        self.assertFalse(self.cache.has("a" * 64))

    def test_discard_removes_both_copies(self):
        key = "d" * 64
        self.cache.put(key, b"\x00\x00" * 4, pinned=True)
        self.cache.discard(key)
        self.assertFalse(self.cache.has(key))

    def test_low_disk_space_stops_writing_instead_of_filling_the_disk(self):
        """答辩当天把系统盘写满，比丢掉缓存严重得多。"""
        self.cache.free_bytes = lambda: MIN_FREE_BYTES // 2
        self.assertFalse(self.cache.put("e" * 64, b"\x00\x00" * 4))
        self.assertIn("磁盘剩余空间不足", self.cache.stats()["last_skip"])

    def test_stats_split_auto_and_pinned(self):
        self.cache.put("a" * 64, b"\x00\x00" * 4, pinned=True)
        self.cache.put("b" * 64, b"\x00\x00" * 8)
        stats = self.cache.stats()
        self.assertEqual(stats["pinned_count"], 1)
        self.assertEqual(stats["auto_count"], 1)
        self.assertGreater(stats["free_bytes"], 0)


class MemoryCacheBudgetTests(unittest.TestCase):
    """内存缓存按字节限——按条数限毫无意义。"""

    def engine(self, limit_mb=1):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = MagicMock()
        fake._pcm_cache = {}
        fake._cache_bytes = 0
        fake._cache_limit_bytes = limit_mb * 1024 * 1024
        fake._utterance_offsets = []
        # 录音用真的 spool：假的返回 MagicMock，解包会炸
        fake._meeting_spool = PcmSpool(Path(tmp.name) / "m.pcm")
        self.addCleanup(fake._meeting_spool.close)
        fake._trim_memory_cache = DefenseEngine._trim_memory_cache.__get__(fake)
        fake._store_pcm = DefenseEngine._store_pcm.__get__(fake)
        fake._tts_settings.return_value = {}
        return fake

    def test_byte_budget_is_enforced(self):
        eng = self.engine(limit_mb=1)
        for index in range(40):
            eng._store_pcm(f"line {index}", ONE_SECOND)  # 每条 48KB，共 1.9MB
        self.assertLessEqual(eng._cache_bytes, eng._cache_limit_bytes)
        self.assertLess(len(eng._pcm_cache), 40)

    def test_oldest_entries_are_dropped_first(self):
        eng = self.engine(limit_mb=1)
        for index in range(40):
            eng._store_pcm(f"line {index}", ONE_SECOND)
        self.assertNotIn("line 0", eng._pcm_cache)
        self.assertIn("line 39", eng._pcm_cache)

    def test_byte_counter_tracks_overwrites(self):
        """同一句重复缓存时计数不能虚高，否则最后会把新条目全误淘汰。"""
        eng = self.engine(limit_mb=1)
        for _ in range(5):
            eng._store_pcm("same line", ONE_SECOND)
        self.assertEqual(len(eng._pcm_cache), 1)
        self.assertEqual(eng._cache_bytes, len(ONE_SECOND))

    def test_a_single_oversized_entry_does_not_loop_forever(self):
        eng = self.engine(limit_mb=1)
        eng._store_pcm("huge", ONE_SECOND * 40)  # 单条就超限
        self.assertLessEqual(len(eng._pcm_cache), 1)

    def test_script_renders_are_pinned_on_disk(self):
        eng = self.engine()
        eng._store_pcm("script line", ONE_SECOND, pinned=True)
        self.assertTrue(eng.disk_cache.put.call_args.kwargs["pinned"])

    def test_live_speech_is_not_pinned(self):
        eng = self.engine()
        eng._store_pcm("live line", ONE_SECOND)
        self.assertFalse(eng.disk_cache.put.call_args.kwargs["pinned"])


class HourLongMeetingTests(unittest.TestCase):
    """一小时会议的量级不会让内存失控。"""

    def test_meeting_audio_never_accumulates_in_memory(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        spool = PcmSpool(Path(tmp.name) / "meeting.pcm", flush_bytes=256 * 1024)
        self.addCleanup(spool.close)
        # 25 分钟实际发声 ≈ 1500 秒 ≈ 72MB，用 1 秒一段模拟 300 段
        for _ in range(300):
            spool.append(ONE_SECOND)
        self.assertEqual(spool.total_bytes(), 300 * len(ONE_SECOND))
        # 不管累积多少，内存里始终不超过一个刷盘阈值
        self.assertLess(spool.stats()["buffered"], 256 * 1024)


class CacheUsagePanelTests(unittest.TestCase):
    """设置页的用量显示与清理按钮。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        from teams_voice_translator.defense.settings import DefenseSettings

        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def dialog(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        dlg = SettingsDialog(None, self.settings)
        self.addCleanup(dlg.close)
        return dlg

    def seed(self, dlg, *, auto=0, pinned=0):
        view = dlg._cache_view()
        for index in range(auto):
            view.put(f"a{index:063d}", b"\x00\x00" * 512)
        for index in range(pinned):
            view.put(f"p{index:063d}", b"\x00\x00" * 512, pinned=True)
        return view

    def test_usage_splits_realtime_and_script(self):
        dlg = self.dialog()
        self.seed(dlg, auto=2, pinned=3)
        dlg._refresh_cache_usage()
        text = dlg.cache_usage_label.text()
        self.assertIn("实时翻译", text)
        self.assertIn("3 条，受保护不淘汰", text)
        self.assertIn("磁盘剩余", text)

    def test_meeting_recording_size_is_shown(self):
        dlg = self.dialog()
        (Path(self.tmp.name) / "meeting_audio.pcm").write_bytes(b"\x00\x00" * 1024)
        dlg._refresh_cache_usage()
        self.assertIn("本场录音", dlg.cache_usage_label.text())

    def test_warns_before_the_limit_is_reached(self):
        """快满了要提前说，别等到现场写不进去。"""
        dlg = self.dialog()
        dlg.disk_cache_spin.setValue(64)
        view = dlg._cache_view()
        payload = b"\x00\x00" * (512 * 1024)  # 1 MiB
        for index in range(60):
            view.put(f"b{index:063d}", payload)
        dlg._refresh_cache_usage()
        self.assertIn("接近上限", dlg.cache_usage_label.text())

    def test_missing_cache_directory_is_reported_not_raised(self):
        dlg = self.dialog()
        dlg._refresh_cache_usage()  # 空目录也不该抛
        self.assertTrue(dlg.cache_usage_label.text())

    def test_limit_is_persisted_on_save(self):
        dlg = self.dialog()
        dlg.disk_cache_spin.setValue(2048)
        dlg._save()
        self.assertEqual(self.settings.get("disk_cache_mb"), 2048)


if __name__ == "__main__":
    unittest.main()
