from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from teams_voice_translator.defense.pipeline import (
    STATE_IDLE,
    DefenseEngine,
    EngineCallbacks,
)
from teams_voice_translator.defense.settings import DefenseSettings


class CallbackCapture:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.errors: list[str] = []
        self.statuses: list[str] = []
        self.states: list[str] = []
        self.tts_started: list[tuple[str, str]] = []

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            on_state=lambda state: self._add(self.states, state),
            on_status=lambda message: self._add(self.statuses, message),
            on_error=lambda message: self._add(self.errors, message),
            on_tts_sentence_started=lambda text, zh: self._add(self.tts_started, (text, zh)),
        )

    def _add(self, target: list, value) -> None:
        with self.lock:
            target.append(value)

    def wait_error(self, substring: str, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if any(substring in message for message in self.errors):
                    return True
            time.sleep(0.05)
        return False


class DefenseEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = DefenseSettings(base_dir=Path(self._tmp.name))
        self.capture = CallbackCapture()
        self.engine = DefenseEngine(self.settings, self.capture.callbacks())
        self.engine.start()

    def tearDown(self) -> None:
        self.engine.shutdown()

    def test_start_defense_without_credentials_reports_error(self) -> None:
        self.engine.start_defense()
        self.assertTrue(self.capture.wait_error("Workspace"), "缺少 Workspace 时应回报错误")
        self.assertNotEqual(self.engine.state, "standby")

    def test_send_text_without_voice_reports_error(self) -> None:
        self.settings.shared.update({"workspace_id": "llm-demo"})
        self.engine.send_text("你好", translate=False)
        self.assertTrue(self.capture.wait_error("voice_id"))

    def test_segment_registry(self) -> None:
        seg_id = self.engine._register_segment("me", "第一句")
        self.engine._store_segment(seg_id, "First sentence.", latency_ms=1800)
        records = self.engine.segments()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].translation, "First sentence.")
        self.assertEqual(records[0].latency_ms, 1800)
        self.engine.retranslate(999)  # unknown id: no-op
        self.engine.clear_memory()
        self.assertEqual(self.engine.segments(), [])
        self.assertEqual(self.engine.memory.snapshot_turns(), [])

    def test_state_starts_idle(self) -> None:
        self.assertEqual(self.engine.state, STATE_IDLE)

    def test_pcm_cache_populated_from_tts_callbacks(self) -> None:
        self.engine._on_first_audio("Hello cache.")
        self.engine._on_tts_audio(b"")
        self.engine._on_utterance_done("Hello cache.", 2)
        cached = self.engine._pcm_cache.get("Hello cache.")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.pcm, b"")

    def test_replay_english_prefers_local_cache(self) -> None:
        class FakePlayer:
            def __init__(self) -> None:
                self.written: list[bytes] = []

            def write(self, pcm: bytes) -> None:
                self.written.append(pcm)

            def close(self) -> None:
                return None

        self.engine._store_pcm("Cached sentence.", b"pcm-bytes", zh="缓存句子。")
        fake_player = FakePlayer()
        self.engine.player = fake_player
        self.engine.replay_english("Cached sentence.")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not fake_player.written:
            time.sleep(0.05)
        self.assertEqual(fake_player.written, [b"pcm-bytes"], "命中缓存时不应请求服务器")
        with self.capture.lock:
            started = [text for text, _zh in self.capture.tts_started]
        self.assertIn("Cached sentence.", started)

    def test_pcm_cache_evicts_oldest(self) -> None:
        self.engine._cache_limit = 2
        self.engine._store_pcm("one", b"1")
        self.engine._store_pcm("two", b"2")
        self.engine._store_pcm("three", b"3")
        self.assertNotIn("one", self.engine._pcm_cache)
        self.assertIn("three", self.engine._pcm_cache)

    def test_clear_memory_clears_pcm_cache(self) -> None:
        self.engine._store_pcm("gone", b"x")
        self.engine.clear_memory()
        self.assertEqual(self.engine._pcm_cache, {})

    def test_save_archive_writes_files_without_recording(self) -> None:
        import tempfile as _tempfile
        from pathlib import Path as _Path

        with _tempfile.TemporaryDirectory() as tmp:
            folder = _Path(tmp) / "答辩_test"
            (folder / "字幕").mkdir(parents=True)
            self.engine.archive_folder = folder
            seg_id = self.engine._register_segment("me", "会议内容测试。")
            self.engine._store_segment(seg_id, "The meeting content test.", latency_ms=1500)
            self.engine.save_archive_now()
            record = folder / "会议全记录.md"
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not record.exists():
                time.sleep(0.05)
            self.assertTrue(record.exists())
            self.assertIn("会议内容测试。", record.read_text(encoding="utf-8"))

    def test_reload_context_picks_up_settings(self) -> None:
        self.settings.set("defense_brief", "新的论文摘要")
        self.settings.set_glossary([{"source": "反演", "target": "inversion"}])
        self.engine.reload_context()
        self.assertIn("新的论文摘要", self.engine.memory.brief)
        self.assertEqual(self.engine.memory.snapshot_glossary()[0]["target"], "inversion")


if __name__ == "__main__":
    unittest.main()
