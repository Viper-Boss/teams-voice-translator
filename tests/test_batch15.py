"""第十五批：讲稿台换气/光标/保存/换音色、播放不再卡住开始答辩和 F5、同一段绝不读两遍、
点播后答案锁定、中英自动互改、朗读按钮三色、字幕跟随暂停、声卡锁定与默认监听设备、设置分页、
现场 flash + 提前渲染 plus 分开设置。

全部离线：合成、声卡、大模型都是假的。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from teams_voice_translator.defense.settings import DefenseSettings


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(predicate, rounds=300):
    app = _app()
    for _ in range(rounds):
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))


# ---------------------------------------------------------------- 引擎
class EnginePlaybackTests(_Base):
    def setUp(self):
        super().setUp()
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks

        self.controls = []
        self.engine = DefenseEngine(self.settings, EngineCallbacks(on_speech_control=self.controls.append))
        self.addCleanup(self.engine.shutdown)
        self.player = MagicMock()
        self.player.write.side_effect = lambda data: time.sleep(0.02)
        self.engine.player = self.player

    def _play(self, seconds=1.0, tag="t"):
        from teams_voice_translator.defense.pipeline import CachedUtterance

        pcm = b"\x01\x00" * int(24000 * seconds)
        self.engine._play_cached(CachedUtterance(pcm=pcm), "text", tag=tag)

    def test_整段播放时不长时间占着播放锁_开始答辩和F5不会卡住(self):
        self._play(2.0)
        time.sleep(0.1)
        started = time.monotonic()
        got = self.engine._player_lock.acquire(timeout=1.0)
        waited = time.monotonic() - started
        self.assertTrue(got)
        self.engine._player_lock.release()
        self.assertLess(waited, 0.3, "以前要等整段播完（2 秒）才拿得到锁")
        self.engine.interrupt_speech()

    def test_真实播放进度_暂停时不涨(self):
        self._play(1.0, tag="seg1")
        time.sleep(0.15)
        state = self.engine.cached_playback_state()
        self.assertEqual(state["tag"], "seg1")
        self.assertTrue(state["active"])
        self.assertGreater(state["played"], 0)
        self.engine.pause_speech()
        time.sleep(0.1)
        frozen = self.engine.cached_playback_state()["played"]
        time.sleep(0.2)
        self.assertAlmostEqual(self.engine.cached_playback_state()["played"], frozen, delta=0.05)
        self.assertIn("pause", self.controls)
        self.engine.resume_speech()
        self.assertTrue(_wait(lambda: not self.engine.cached_playback_state()["active"], rounds=400))
        self.assertAlmostEqual(self.engine.cached_playback_state()["played"], 1.0, delta=0.05)

    def test_Esc通知字幕熄灭_内部换段不熄(self):
        self._play(0.2)
        self.assertNotIn("stop", self.controls, "内部换播下一段不应熄灭字幕")
        self.engine.interrupt_speech()
        self.assertIn("stop", self.controls)

    def test_同一段连点只读一遍_重播可以强制(self):
        submitted = []
        self.engine._submit = lambda action, label: submitted.append(label)
        self.engine.answer_ready = lambda text: True
        self.engine.speak_prepared("The answer.", label="预案")
        self.engine.speak_prepared("The answer.", label="预案")
        self.engine.speak_prepared("The answer.", label="预案")
        self.assertEqual(len(submitted), 1, "连点三次只能提交一次")
        self.engine.speak_prepared("The answer.", label="重播", force=True)
        self.assertEqual(len(submitted), 2)
        self.engine.speak_prepared("A different answer.", label="预案")
        self.assertEqual(len(submitted), 3)

    def test_提前渲染用单独的音色_留空跟现场一致(self):
        self.settings.update({"tts_model": "qwen-audio-3.1-tts-flash", "tts_voice_id": "v-flash"})
        values = self.engine.script_settings("en")
        self.assertEqual((values["tts_model"], values["voice"]), ("qwen-audio-3.1-tts-flash", "v-flash"))
        self.settings.update({"prerender_tts_model": "qwen-audio-3.0-tts-plus", "prerender_voice_id": "v-plus"})
        values = self.engine.script_settings("en")
        self.assertEqual((values["tts_model"], values["voice"]), ("qwen-audio-3.0-tts-plus", "v-plus"))
        self.assertEqual(self.engine.answer_settings()["voice"], "v-plus", "问答库预渲染也用提前渲染音色")


class LoopbackChoiceTests(unittest.TestCase):
    def test_没保存过时避开ToDesk这类虚拟设备(self):
        from teams_voice_translator.audio import choose_committee_loopback

        devices = [SimpleNamespace(index=38, name="扬声器 (ToDesk Virtual Audio) [Loopback]"),
                   SimpleNamespace(index=8, name="扬声器 (Realtek) [Loopback]")]
        self.assertEqual(choose_committee_loopback(devices).index, 8)
        self.assertEqual(choose_committee_loopback(devices, 38).index, 38, "用户自己选的照样尊重")


# ---------------------------------------------------------------- 字幕悬浮窗
class OverlayTests(_Base):
    def setUp(self):
        super().setUp()
        from teams_voice_translator.defense.overlay import SubtitleOverlay

        _app()
        self.overlay = SubtitleOverlay(self.settings)
        self.addCleanup(self.overlay.stop)

    def test_暂停时高亮停住_继续后接着走(self):
        self.overlay.show_sentence("中", "one two three four five six", 3.0)
        self.overlay.pause()
        t0 = self.overlay._t0
        time.sleep(0.2)
        self.overlay._tick()
        self.overlay.resume()
        self.assertGreater(self.overlay._t0, t0 + 0.15, "起点往后挪了暂停的时长")

    def test_时长比较忽略空白_没时长先估一个(self):
        self.overlay.show_sentence("", "Hello  world, this is a test.")
        self.assertTrue(self.overlay._estimated)
        self.assertGreater(self.overlay._duration, 0)
        self.overlay.set_duration("Hello world, this is a test.", 4.2)
        self.assertEqual(self.overlay._duration, 4.2)
        self.assertFalse(self.overlay._estimated)


# ---------------------------------------------------------------- 讲稿台
class ScriptDeskTests(_Base):
    def setUp(self):
        super().setUp()
        _app()
        self.engine = MagicMock()
        self.store: dict[str, bytes] = {}
        self.engine.script_pcm.side_effect = lambda text, lang: self.store.get(
            f"{self.settings.get('prerender_voice_id')}|{text}")

    def panel(self):
        from teams_voice_translator.defense.ui import ScriptDeskPanel

        widget = ScriptDeskPanel(self.settings, self.engine)
        self.addCleanup(widget.shutdown)
        return widget

    def test_讲稿自动保存_下次打开还在_渲染过的直接就绪(self):
        self.store["|One."] = b"\x00\x00" * 24000
        panel = self.panel()
        panel.source_edit.setPlainText("One. Two.")
        panel._split_clicked()
        again = self.panel()
        self.assertEqual(again.source_edit.toPlainText(), "One. Two.")
        self.assertEqual(len(again.document), 2)
        self.assertTrue(again.document.lines[0].rendered)
        self.assertAlmostEqual(again.document.lines[0].seconds, 1.0)
        self.assertFalse(again.document.lines[1].rendered)

    def test_换渲染音色_状态按新音色重查_锁住才不能换(self):
        self.settings.update({"voice_library": [
            {"voice_id": "v-plus", "target_model": "qwen-audio-3.0-tts-plus", "name": "plus"},
            {"voice_id": "v-flash", "target_model": "qwen-audio-3.1-tts-flash", "name": "flash"}]})
        self.store["|One."] = b"\x00\x00" * 100
        panel = self.panel()
        panel.source_edit.setPlainText("One.")
        panel._split_clicked()
        self.assertTrue(panel.document.lines[0].rendered)
        self.assertFalse(panel.voice_combo.isEnabled(), "默认锁定")
        index = next(i for i in range(panel.voice_combo.count()) if panel.voice_combo.itemData(i)[1] == "v-plus")
        panel._on_voice_chosen(index)
        self.assertEqual(self.settings.get("prerender_voice_id"), "", "锁着时不换")
        panel.voice_lock_button.setChecked(False)
        panel._on_voice_chosen(index)
        self.assertEqual(self.settings.get("prerender_voice_id"), "v-plus")
        self.assertFalse(panel.document.lines[0].rendered, "新音色还没渲染：不能再说「已渲染」")
        self.assertTrue(panel.voice_lock_button.isChecked(), "换完自动锁回")
        self.assertTrue(panel.voice_combo.itemText(index).startswith("★"), "plus 推荐项高亮")

    def test_暂停_继续_停止通知主窗口(self):
        self.store["|One."] = b"\x00\x00" * 24000
        events = []
        panel = self.panel()
        panel.on_playback_changed = events.append
        panel.source_edit.setPlainText("One.")
        panel._split_clicked()
        panel.paragraph_table.selectRow(0)
        panel._play_paragraph()
        self.assertTrue(panel.is_playing())
        panel.pause()
        self.engine.pause_speech.assert_called_once()
        self.assertFalse(panel.is_playing())
        panel.resume()
        self.engine.resume_speech.assert_called_once()
        panel._stop_playback()
        self.assertEqual(events, [True, False, True, False])

    def test_中文稿翻英文带上口吻设置(self):
        self.settings.set("translation_style", "thoughtful")
        panel = self.panel()
        with mock.patch("teams_voice_translator.defense.ui.ContextTranslator"), \
                mock.patch("teams_voice_translator.defense.ui.calibrate", return_value="EN") as cal:
            run = panel._default_translate_runner()
            run("中文", "")
        self.assertIn("边想边讲", cal.call_args.kwargs["context"])


# ---------------------------------------------------------------- 助答面板
class QaPanelTests(_Base):
    def setUp(self):
        super().setUp()
        from teams_voice_translator.defense.ui import QaPanel

        _app()
        self.engine = MagicMock()
        self.engine.last_voice_write = 0.0
        self.engine.is_speaking.return_value = False
        self.engine.speech_paused = False
        self.engine.speak_prepared.return_value = True
        self.panel = QaPanel(self.settings, self.engine)
        self.addCleanup(self.panel.shutdown)
        self.panel.answer_en.setText("This is my answer.")

    def test_朗读按钮_黄色准备_红色在读_读完变回并显示已读(self):
        panel = self.panel
        panel._play_answer()
        self.assertEqual(panel._play_state, "pending")
        self.assertIn("#f5c542", panel.play_button.styleSheet())
        self.engine.last_voice_write = time.monotonic()
        self.engine.is_speaking.return_value = True
        panel._poll_play_state()
        self.assertEqual(panel._play_state, "playing")
        self.assertIn("#e5484d", panel.play_button.styleSheet())
        self.engine.is_speaking.return_value = False
        panel._poll_play_state()
        self.assertEqual(panel._play_state, "idle")
        self.assertIn("已读过", panel.play_button.text())

    def test_连点_读过再点都不会读第二遍_重播可以(self):
        panel = self.panel
        panel._play_answer()
        panel._play_answer()
        panel._set_play_state("idle")
        panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_count, 1)
        panel.replay_last()
        self.assertEqual(self.engine.speak_prepared.call_count, 2)
        self.assertTrue(self.engine.speak_prepared.call_args.kwargs["force"])

    def test_中途按停止_不算读过_可以再读(self):
        panel = self.panel
        panel._play_answer()
        panel.on_external_speech_control("stop")
        self.assertEqual(panel._play_state, "idle")
        panel._play_answer()
        self.assertEqual(self.engine.speak_prepared.call_count, 2)

    def test_点了朗读后迟到的新回答被丢弃_屏幕和读的一致(self):
        panel = self.panel
        panel._play_answer()
        panel._on_qa_answer("A totally new answer.", "新的")
        self.assertEqual(panel.answer_en.text(), "This is my answer.")
        self.assertIn("丢弃", panel.status_label.text())
        panel.on_committee_done("Next question?", "下一个问题？")
        self.assertFalse(panel._answer_locked, "新问题来了解锁")

    def test_改中文停手后英文自动跟上_手动按英文改中文(self):
        panel = self.panel
        calls = []

        def factory():
            def run(direction, edited, other, context):
                calls.append(direction)
                return "Adjusted English." if direction == "zh2en" else "调整后的中文"
            return run

        panel.calibrator_factory = factory
        panel.answer_zh.setText("我改过的中文")
        panel._on_zh_edited(user=True)
        self.assertTrue(panel._zh_sync_timer.isActive())
        panel._zh_sync_timer.stop()
        panel._auto_zh_to_en()
        self.assertTrue(_wait(lambda: panel.answer_en.text() == "Adjusted English."))
        panel.answer_en.setText("Edited English.")
        panel._calibrate_chinese()
        self.assertTrue(_wait(lambda: panel.answer_zh.text() == "调整后的中文"))
        self.assertEqual(calls, ["zh2en", "en2zh"])


# ---------------------------------------------------------------- 主窗口与设置
class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teams_voice_translator.defense.ui import DefenseWindow

        _app()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.settings = DefenseSettings(base_dir=Path(cls._tmp.name))
        cls.window = DefenseWindow(cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls._tmp.cleanup()

    def test_讲稿在播时F9保留持续模式_并先暂停讲稿(self):
        window = self.window
        window.continuous_check.setChecked(True)
        desk = window.script_panel
        desk._playing = [object()]
        desk._paused_at = 0.0
        with mock.patch.object(desk, "pause") as pause, \
                mock.patch.object(window.engine, "set_listening") as listen, \
                mock.patch.object(window.engine, "stop_direct"):
            window._input_press("translate", "hotkey")
            pause.assert_called_once()
            listen.assert_called_with(True)
            window._input_release("translate", "hotkey")
            listen.assert_called_with(True)
            window._input_press("translate", "hotkey")
            listen.assert_called_with(False)
        desk._playing = []
        window.continuous_check.setChecked(False)

    def test_字幕悬浮窗跟着暂停和停止(self):
        overlay = self.window.overlay
        with mock.patch.object(overlay, "pause") as pause, mock.patch.object(overlay, "stop") as stop:
            self.window._on_speech_control("pause")
            self.window._on_speech_control("stop")
        pause.assert_called_once()
        stop.assert_called_once()


class SettingsDialogTests(_Base):
    def test_分页_声卡默认收起_开关展开(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        _app()
        dialog = SettingsDialog(None, self.settings)
        names = [dialog.settings_tabs.tabText(i) for i in range(dialog.settings_tabs.count())]
        self.assertEqual(names[:4], ["声音", "识别与翻译", "快捷键", "声卡"])
        self.assertIn("术语表", names)
        self.assertFalse(dialog.device_box.isHidden(), "四个声卡选择一直显示，筛掉的是下拉里没用的条目")
        self.assertTrue(dialog.device_box.isAncestorOf(dialog.mic_combo))
        self.assertTrue(dialog.device_box.isAncestorOf(dialog.cable_combo))
        self.assertTrue(dialog.device_box.isAncestorOf(dialog.loopback_combo))
        self.assertTrue(dialog.device_box.isAncestorOf(dialog.monitor_combo))
        self.assertFalse(dialog.device_box.isAncestorOf(dialog.monitor_check))
        dialog.show_devices_check.setChecked(True)
        self.assertFalse(dialog.device_box.isHidden())
        self.assertTrue(self.settings.get("show_all_audio_devices"))

    def test_提前渲染音色可选并保存(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        _app()
        self.settings.update({"voice_library": [
            {"voice_id": "v-plus", "target_model": "qwen-audio-3.0-tts-plus", "name": "plus"}]})
        dialog = SettingsDialog(None, self.settings)
        # 先选提前渲染的模型，下面的音色只列和它匹配的
        dialog.prerender_model_combo.setCurrentIndex(dialog.prerender_model_combo.findData("qwen-audio-3.0-tts-plus"))
        combo = dialog.prerender_voice_combo
        index = next(i for i in range(combo.count()) if combo.itemData(i) and combo.itemData(i)[1] == "v-plus")
        combo.setCurrentIndex(index)
        with mock.patch("teams_voice_translator.defense.ui.QMessageBox"):
            dialog._save()
        self.assertEqual(self.settings.get("prerender_voice_id"), "v-plus")
        self.assertEqual(self.settings.get("prerender_tts_model"), "qwen-audio-3.0-tts-plus")

    def test_默认监听设备改成8号_只改一次_之后尊重手动选择(self):
        from teams_voice_translator.defense import ui

        devices = [SimpleNamespace(index=38, name="扬声器 (ToDesk Virtual Audio) [Loopback]"),
                   SimpleNamespace(index=8, name="耳机 (EDIFIER G4) [Loopback]")]
        self.settings.shared.update({"loopback_device": 38, "loopback_device_name": devices[0].name})
        with mock.patch.object(ui, "_safe_list_loopback_devices", return_value=devices):
            self.assertTrue(ui.apply_default_loopback(self.settings))
            self.assertEqual(self.settings.shared.get("loopback_device"), 8)
            self.settings.shared.update({"loopback_device": 38})
            self.assertFalse(ui.apply_default_loopback(self.settings), "只迁移一次")
        self.assertEqual(self.settings.shared.get("loopback_device"), 38)


class HotkeyDefaultTests(unittest.TestCase):
    """原声键默认 F8：F5 是 PowerPoint 放映键，答辩时会打乱 PPT。"""

    def _load(self, saved: dict) -> DefenseSettings:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "defense_settings.json").write_text(json.dumps(saved), encoding="utf-8")
        return DefenseSettings(base_dir=Path(tmp.name))

    def test_新装默认F8(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(DefenseSettings(base_dir=Path(tmp)).get("direct_hotkey"), "f8")

    def test_旧配置里的F5自动换成F8_只换一次(self):
        settings = self._load({"direct_hotkey": "f5"})
        self.assertEqual(settings.get("direct_hotkey"), "f8")
        settings.set("direct_hotkey", "f5")   # 迁移后用户自己又选回 F5：尊重
        again = DefenseSettings(base_dir=settings.base_dir)
        self.assertEqual(again.get("direct_hotkey"), "f5")

    def test_F8被占用就换F7_自己选的其他键不动(self):
        self.assertEqual(self._load({"direct_hotkey": "f5", "translate_hotkey": "f8"}).get("direct_hotkey"), "f7")
        self.assertEqual(self._load({"direct_hotkey": "f2"}).get("direct_hotkey"), "f2")

    def test_界面按钮显示新键(self):
        from teams_voice_translator.defense.ui import DefenseWindow

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            window = DefenseWindow(DefenseSettings(base_dir=Path(tmp)))
            try:
                self.assertIn("F8", window.direct_button.text())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
