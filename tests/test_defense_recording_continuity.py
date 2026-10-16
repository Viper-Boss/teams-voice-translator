"""Recording continuity: exercise real WAV writers with simulated audio devices."""
from __future__ import annotations

import tempfile
import time
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from teams_voice_translator.audio import DualTrackRecorder, PcmWaveSink, WavRecorder
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings


PCM = b"\x14\x00" * 1600


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return False


class InputStream:
    def __init__(self, *, callback, **kwargs):
        self.callback = callback
        self.closed = False

    def start(self):
        self.callback(PCM, len(PCM) // 2, None, None)

    def stop(self):
        pass

    def close(self):
        self.closed = True


class SystemCapture:
    def start(self, device, callback, **kwargs):
        self.callback = callback
        callback(PCM)

    def stop(self):
        pass


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.settings = DefenseSettings(base_dir=self.base / "settings")
        self.settings.set("output_directory", str(self.base / "recordings"))
        self.settings.shared.update({"input_device": 1, "loopback_device": 42})
        self.errors = []
        self.engine = DefenseEngine(self.settings, EngineCallbacks(on_error=self.errors.append))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("teams_voice_translator.audio.sd.RawInputStream", InputStream))
        self.stack.enter_context(patch("teams_voice_translator.audio.SystemAudioCapture", SystemCapture))
        for method in ("_build_clients", "_ensure_voice", "_start_committee_listening", "_start_ambience_bed"):
            self.stack.enter_context(patch.object(self.engine, method))
        self.stack.enter_context(patch.object(self.engine, "_submit", side_effect=lambda action, _desc: action()))
        self.addCleanup(self.engine.shutdown)

    def start_recording(self):
        self.engine._do_start_defense()
        self.assertIsNotNone(self.engine.recorder)
        self.assertTrue(wait_for(lambda: self.engine.recorder.system_sink.frames_written == 1600))
        return self.engine.recorder

    def assert_frames(self, path, count):
        with wave.open(str(path), "rb") as source:
            self.assertEqual(source.getnframes(), count)
            self.assertEqual(source.readframes(count), PCM * (count // 1600))

    def test_stop_then_restart_in_existing_archive_records_another_part(self):
        first = self.start_recording()
        self.engine._do_stop_defense()
        saved = first.paths["system"].read_bytes()
        folder = self.engine.archive_folder
        second = self.start_recording()
        self.engine._do_stop_defense()
        self.assertEqual(self.engine.archive_folder, folder)
        self.assertNotEqual(first.paths["system"], second.paths["system"])
        self.assertEqual(first.paths["system"].read_bytes(), saved)
        self.assert_frames(first.paths["system"], 1600)
        self.assert_frames(second.paths["system"], 1600)
        self.assertEqual(len(list((folder / "录音").glob("*.wav"))), 6)

    def test_rapid_restarts_at_identical_clock_time_do_not_overwrite(self):
        paths = []
        with patch("teams_voice_translator.audio.time.strftime", return_value="20261016_142934"):
            for _ in range(3):
                recorder = self.start_recording()
                self.engine._do_stop_defense()
                paths.append(recorder.paths["system"])
        self.assertEqual(len(set(paths)), 3)
        for path in paths:
            self.assert_frames(path, 1600)

    def test_stopping_live_speech_does_not_stop_teacher_recording(self):
        recorder = self.start_recording()
        self.engine.interrupt_speech()
        self.assertIs(self.engine.recorder, recorder)
        self.assertTrue(recorder.running.is_set())
        recorder.system_capture.callback(PCM)
        self.engine._do_stop_defense()
        self.assert_frames(recorder.paths["system"], 3200)

    def test_changing_voice_or_model_keeps_teacher_recording_through_initialization(self):
        recorder = self.start_recording()

        def while_reinitializing_voice():
            self.assertIs(self.engine.recorder, recorder)
            self.assertTrue(recorder.running.is_set())
            recorder.system_capture.callback(PCM)

        self.engine._ensure_voice.side_effect = while_reinitializing_voice
        for model in ("plus", "flash", "plus"):
            self.settings.set("tts_model", model)
            self.engine.apply_settings()
        self.engine._do_stop_defense()
        self.assert_frames(recorder.paths["system"], 6400)
        self.assertEqual(len(list(self.engine.archive_folder.joinpath("录音").glob("teacher*.wav"))), 1)
        self.assertFalse(self.errors)

    def test_device_change_keeps_old_file_and_records_new_device(self):
        old = self.start_recording()
        self.settings.shared.update({"loopback_device": 43})
        self.engine.apply_settings()
        new = self.engine.recorder
        self.assertIsNot(old, new)
        self.assertFalse(old.running.is_set())
        self.assertEqual(self.engine._recording_devices, (1, 43))
        self.engine._do_stop_defense()
        self.assert_frames(old.paths["system"], 1600)
        self.assert_frames(new.paths["system"], 1600)

    def test_checkbox_stops_immediately_and_resumes_to_new_part(self):
        old = self.start_recording()
        self.engine.set_auto_record(False)
        self.assertIsNone(self.engine.recorder)
        self.assertFalse(old.running.is_set())
        self.engine.set_auto_record(True)
        new = self.engine.recorder
        self.assertIsNotNone(new)
        self.engine._do_stop_defense()
        self.assert_frames(old.paths["system"], 1600)
        self.assert_frames(new.paths["system"], 1600)

    def test_disabled_auto_record_does_not_restart_on_settings_change(self):
        old = self.start_recording()
        self.settings.set("auto_record", False)
        self.engine.apply_settings()
        self.assertIsNone(self.engine.recorder)
        self.assertFalse(old.running.is_set())

    def test_voice_initialization_failure_does_not_kill_existing_teacher_recording(self):
        recorder = self.start_recording()
        self.engine._ensure_voice.side_effect = RuntimeError("模拟网络故障")
        with self.assertRaisesRegex(RuntimeError, "模拟网络故障"):
            self.engine.apply_settings()
        self.assertIs(self.engine.recorder, recorder)
        recorder.system_capture.callback(PCM)
        self.engine._do_stop_defense()
        self.assert_frames(recorder.paths["system"], 3200)

    def test_missing_loopback_is_not_reported_as_teacher_recording(self):
        self.settings.shared.update({"loopback_device": None})
        self.engine._do_start_defense()
        self.assertIsNotNone(self.engine.mic_recorder)
        self.assertEqual(self.engine.recording_status()[0], "error")
        self.assertIn("未录老师", self.engine.recording_status()[1])
        self.assertTrue(self.errors)

    def test_start_failure_can_retry_despite_existing_folder(self):
        with patch.object(DualTrackRecorder, "start", side_effect=RuntimeError("设备忙")):
            with self.assertRaisesRegex(RuntimeError, "设备忙"):
                self.engine._start_recording()
        folder = self.engine.archive_folder
        self.assertEqual(self.engine.recording_status()[0], "error")
        self.engine._start_recording()
        self.assertEqual(self.engine.archive_folder, folder)
        self.assertTrue(self.engine.recorder.running.is_set())
        self.assertEqual(self.engine._recording_error, "")

    def test_status_counts_written_samples_and_warns_on_stall_or_writer_failure(self):
        recorder = self.start_recording()
        recorder.system_capture.callback(PCM * 20)
        self.assertTrue(wait_for(lambda: recorder.system_sink.frames_written == 33600))
        state, text = self.engine.recording_status()
        self.assertEqual(state, "recording")
        self.assertIn("00:02", text)
        recorder.system_sink.last_audio_at = time.monotonic() - 20
        self.assertEqual(self.engine.recording_status()[0], "waiting")
        recorder.system_sink.error = "模拟磁盘已满"
        self.assertEqual(self.engine.recording_status(), ("error", "模拟磁盘已满"))


class WriterSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "teacher.wav"

    def test_open_file_already_has_playable_header(self):
        sink = PcmWaveSink()
        sink.start(self.path, 16000)
        self.addCleanup(sink.stop)
        sink.write(PCM)
        def readable():
            try:
                with wave.open(str(self.path), "rb") as source:
                    return source.getnframes() == 1600 and source.readframes(1600) == PCM
            except (EOFError, wave.Error):
                return False
        self.assertTrue(wait_for(readable))
        self.assertTrue(sink.running.is_set())

    def test_existing_recording_cannot_be_overwritten(self):
        self.path.write_bytes(b"keep the original audio")
        sink = PcmWaveSink()
        with self.assertRaises(FileExistsError):
            sink.start(self.path, 16000)
        self.assertEqual(self.path.read_bytes(), b"keep the original audio")

    def test_stop_drains_all_pending_samples(self):
        sink = PcmWaveSink()
        sink.start(self.path, 16000)
        for _ in range(200):
            sink.write(PCM)
        sink.stop()
        with wave.open(str(self.path), "rb") as source:
            self.assertEqual(source.getnframes(), 320000)
            self.assertEqual(source.readframes(320000), PCM * 200)

    def test_device_stop_failure_still_closes_and_finalizes_microphone_file(self):
        recorder = WavRecorder()
        with patch("teams_voice_translator.audio.sd.RawInputStream", InputStream):
            recorder.start(1, self.path, 16000)
        stream = recorder.stream
        with patch.object(stream, "stop", side_effect=RuntimeError("驱动停止失败")):
            with self.assertRaisesRegex(RuntimeError, "驱动停止失败"):
                recorder.stop()
        self.assertTrue(stream.closed)
        self.assertIsNone(recorder.thread)
        with wave.open(str(self.path), "rb") as source:
            self.assertEqual(source.getnframes(), 1600)

    def test_writer_error_is_observable_and_reported_by_stop(self):
        sink = PcmWaveSink()
        sink.start(self.path, 16000)
        with patch.object(wave.Wave_write, "writeframesraw", side_effect=OSError("磁盘已满")):
            sink.write(PCM)
            self.assertTrue(wait_for(lambda: bool(sink.error)))
        with self.assertRaisesRegex(RuntimeError, "磁盘已满"):
            sink.stop()
        self.assertFalse(sink.running.is_set())

    def test_file_close_failure_cannot_leave_a_false_running_state(self):
        raw = self.path.open("xb")
        wrapped = Mock(wraps=raw)

        def close_failure():
            raw.close()
            raise OSError("保存设备已断开")

        wrapped.close.side_effect = close_failure
        sink = PcmWaveSink()
        with patch.object(Path, "open", return_value=wrapped):
            sink.start(self.path, 16000)
        sink.write(PCM)
        with self.assertRaisesRegex(RuntimeError, "保存设备已断开"):
            sink.stop()
        self.assertFalse(sink.running.is_set())


class RecordingUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_recording_status_and_checkbox_follow_engine(self):
        from teams_voice_translator.defense import ui
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            settings = DefenseSettings(base_dir=Path(temp))
            stack.enter_context(patch.object(ui, "apply_default_loopback", return_value=False))
            stack.enter_context(patch.object(ui.DefenseEngine, "start"))
            stack.enter_context(patch.object(ui.DefenseWindow, "_auto_prerender"))
            window = ui.DefenseWindow(settings)
            try:
                with patch.object(window.engine, "recording_status", return_value=("recording", "老师录音中 · 01:23")):
                    window._refresh_recording_status()
                    self.assertIn("01:23", window.recording_status_label.text())
                    self.assertIn("#2fd19b", window.recording_status_label.styleSheet())
                with patch.object(window.engine, "recording_status", return_value=("error", "录音写入失败：磁盘已满")):
                    window._refresh_recording_status()
                    self.assertIn("磁盘已满", window.recording_status_label.text())
                    self.assertIn("#ff6262", window.recording_status_label.styleSheet())
                with patch.object(window.engine, "_submit", side_effect=lambda action, _desc: action()):
                    window.auto_record_check.setChecked(False)
                    self.assertFalse(settings.get("auto_record"))
                    window.auto_record_check.setChecked(True)
                    self.assertTrue(settings.get("auto_record"))
            finally:
                window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
