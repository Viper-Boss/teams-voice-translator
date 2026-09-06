from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from teams_voice_translator.defense.settings import DefenseSettings  # noqa: E402
from teams_voice_translator.defense.ui import (  # noqa: E402
    CloneVoiceDialog,
    DefenseWindow,
    QaPanel,
    RehearsalDialog,
    SelfCheckDialog,
    SettingsDialog,
)


class DefenseUiTest(unittest.TestCase):
    app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.settings = DefenseSettings(base_dir=Path(cls._tmp.name))
        cls.settings.set_glossary([{"source": "有限元", "target": "finite element method"}])
        cls.window = DefenseWindow(cls.settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.window.close()
        cls._tmp.cleanup()

    def setUp(self) -> None:
        window = self.window
        window.engine.clear_memory()
        window.timeline.setRowCount(0)
        window._timeline_rows.clear()
        window.my_history.clear()
        window.committee_history.clear()
        window.latency_label.setText("")
        window._recent_spoken.clear()
        window._rebuild_recent_buttons()

    def test_window_title(self) -> None:
        self.assertIn("答辩模式", self.window.windowTitle())

    def test_wheel_does_not_change_combo_or_spin(self) -> None:
        from PySide6.QtCore import QPointF, QPoint, Qt
        from PySide6.QtGui import QWheelEvent
        from teams_voice_translator.defense.ui import NoWheelComboBox, NoWheelSpinBox

        combo = NoWheelComboBox()
        combo.addItems(["a", "b", "c"])
        combo.setCurrentIndex(1)
        event = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
        )
        combo.wheelEvent(event)
        self.assertEqual(combo.currentIndex(), 1, "滚轮不应改变下拉框选中值")
        self.assertFalse(event.isAccepted(), "滚轮事件应被忽略而非消费")

        spin = NoWheelSpinBox()
        spin.setRange(0, 100)
        spin.setValue(14)
        event2 = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
        )
        spin.wheelEvent(event2)
        self.assertEqual(spin.value(), 14, "滚轮不应改变数字框的值")

    def test_settings_uses_wheel_safe_widgets(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        from teams_voice_translator.defense.ui import NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox

        self.assertIsInstance(dialog.llm_combo, NoWheelComboBox)
        self.assertIsInstance(dialog.tts_model_combo, NoWheelComboBox)
        self.assertIsInstance(dialog.loopback_combo, NoWheelComboBox)
        self.assertIsInstance(dialog.silence_spin, NoWheelSpinBox)
        self.assertIsInstance(dialog.rate_spin, NoWheelDoubleSpinBox)
        self.assertIsInstance(dialog.pitch_spin, NoWheelDoubleSpinBox)
        self.assertIsInstance(dialog.volume_spin, NoWheelSpinBox)
        self.assertIsInstance(self.window.qa_panel.model_combo, NoWheelComboBox)

    def test_state_pill_updates(self) -> None:
        self.window._on_state_changed("listening")
        self.assertEqual(self.window.state_pill.text(), "F9 翻译中")
        self.window._on_state_changed("standby")
        self.assertEqual(self.window.state_pill.text(), "待命")

    def test_state_syncs_buttons_green(self) -> None:
        window = self.window
        window._on_state_changed("listening")
        self.assertTrue(window.translate_button.isChecked())
        # 持续监听开关只反映用户偏好，不随播放状态被强制勾选（防呆修复）
        self.assertFalse(window.continuous_check.isChecked())
        self.assertFalse(window.direct_button.isChecked())
        window._on_state_changed("direct")
        self.assertTrue(window.direct_button.isChecked())
        self.assertFalse(window.translate_button.isChecked())
        window._on_state_changed("standby")
        self.assertFalse(window.direct_button.isChecked())
        self.assertFalse(window.translate_button.isChecked())

    def test_control_buttons_exist(self) -> None:
        window = self.window
        self.assertTrue(window.direct_button.isCheckable())
        self.assertTrue(window.translate_button.isCheckable())
        self.assertTrue(hasattr(window, "continuous_check"))
        self.assertTrue(hasattr(window, "overlay_check"))
        self.assertIsNotNone(window.overlay)

    def test_recent_bar_keeps_last_five_with_replay(self) -> None:
        window = self.window
        calls: list[str] = []
        window.engine.replay_english = lambda text: calls.append(text)  # type: ignore[method-assign]
        for index in range(6):
            window._on_my_done(index, f"Sentence number {index}.", 0)
        visible = [b for b in window.recent_buttons if b.isVisibleTo(window)]
        self.assertEqual(len(visible), 5)
        self.assertIn("Sentence number 5", visible[0].text())
        self.assertIn("Sentence number 1", visible[-1].text())
        visible[-1].click()
        self.assertEqual(calls, ["Sentence number 1."])

    def test_replay_does_not_reorder_recent(self) -> None:
        window = self.window
        for index in range(3):
            window._on_my_done(index, f"Line {index}.", 0)
        # 缓存重播会触发 tts_sentence_started，但不应改变最近列表顺序
        window._on_tts_sentence_started("Line 0.", "第零句")
        self.assertEqual(window._recent_spoken[0], "Line 2.")
        self.assertEqual(window._recent_spoken, ["Line 2.", "Line 1.", "Line 0."])

    def test_timeline_newest_on_top_with_play_button(self) -> None:
        window = self.window
        window._on_my_done(1, "First sentence.", 900)
        window._on_my_done(2, "Second sentence.", 800)
        # 最新在最上面
        self.assertIn("Second", window.timeline.item(0, 3).text())
        self.assertIn("First", window.timeline.item(1, 3).text())
        # 我的行有重播按钮
        widget = window.timeline.cellWidget(0, 5)
        self.assertIsNotNone(widget, "我方行应有重播按钮")

    def test_tts_sentence_events_feed_overlay(self) -> None:
        window = self.window
        window._on_tts_sentence_started("Hello defense world.", "大家好。")
        window._on_tts_sentence_duration("Hello defense world.", 3.2)
        self.assertIn("Hello", window.overlay.en_label.text())
        self.assertIn("大家好", window.overlay.zh_label.text())
        self.assertAlmostEqual(window.overlay._duration, 3.2)
        window.overlay.stop()

    def test_hotkeys_persisted_and_reload(self) -> None:
        from teams_voice_translator.defense.hotkeys_poll import VK_MAP, PollingHotkeys

        window = self.window
        window.settings.set("direct_hotkey", "f6")
        window.settings.set("translate_hotkey", "f10")
        window.start_hotkeys()
        try:
            self.assertIsInstance(window.hotkeys, PollingHotkeys)
            self.assertEqual(window.hotkeys.keys["direct"], "f6")
            self.assertEqual(window.hotkeys.keys["translate"], "f10")
            self.assertEqual(VK_MAP["f6"], 0x75)
        finally:
            window.settings.set("direct_hotkey", "f5")
            window.settings.set("translate_hotkey", "f9")
            window.start_hotkeys()

    def test_proxy_settings_round_trip(self) -> None:
        from teams_voice_translator.aliyun import BailianClient

        self.settings.shared.update({"proxy_mode": "direct", "http_proxy": "127.0.0.1:7890"})
        self.assertEqual(self.settings.proxy_mode, "direct")
        self.assertEqual(self.settings.ws_proxy, "", "直连模式下 WS 不走代理")
        client = BailianClient("sk-test", "ws", proxy=self.settings.ws_proxy, proxy_mode=self.settings.proxy_mode)
        self.assertEqual(client.proxies, {"http": None, "https": None})
        self.settings.shared.update({"proxy_mode": "manual"})
        self.assertEqual(self.settings.ws_proxy, "127.0.0.1:7890")
        client = BailianClient("sk-test", "ws", proxy=self.settings.ws_proxy, proxy_mode="manual")
        self.assertEqual(client.proxies, {"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"})
        self.settings.shared.update({"proxy_mode": "direct"})

    def test_my_done_adds_timeline_row_and_latency(self) -> None:
        window = self.window
        window._on_my_segment(1, "本文提出了新方法。")
        window._on_my_delta(1, "This paper proposes")
        window._on_my_done(1, "This paper proposes a new method.", 1800)
        self.assertEqual(window.timeline.rowCount(), 1)
        self.assertIn("1.8", window.timeline.item(0, 4).text())
        self.assertEqual(window.timeline.item(0, 3).text(), "This paper proposes a new method.")
        self.assertIn("延迟", window.latency_label.text())

    def test_committee_done_adds_row(self) -> None:
        window = self.window
        window._on_committee_done(2, "What is the novelty?", "创新点是什么？")
        row = window.timeline.rowCount() - 1
        self.assertEqual(window.timeline.item(row, 1).text(), "评委")
        self.assertEqual(window.timeline.item(row, 3).text(), "创新点是什么？")
        self.assertIn("创新点", window.committee_history.toPlainText())

    def test_same_segment_updates_row(self) -> None:
        window = self.window
        window._on_my_done(1, "First take.", 900)
        before = window.timeline.rowCount()
        window._on_my_done(1, "Improved translation.", 700)
        self.assertEqual(window.timeline.rowCount(), before)
        self.assertEqual(window.timeline.item(0, 3).text(), "Improved translation.")

    def test_manual_text_callback(self) -> None:
        window = self.window
        before = window.timeline.rowCount()
        window._on_my_done(50, "Typed sentence.", 0)
        self.assertEqual(window.timeline.rowCount(), before + 1)

    def test_settings_dialog_round_trip(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        self.assertTrue(hasattr(dialog, "voice_combo"), "voice_id 应为下拉框")
        self.assertFalse(dialog.voice_combo.isEditable())
        dialog._refresh_voice_combo(keep="voice-ui-test")
        dialog.silence_spin.setValue(400)
        dialog._save()
        reloaded = DefenseSettings(base_dir=Path(self._tmp.name))
        self.assertEqual(reloaded.get("tts_voice_id"), "voice-ui-test")
        self.assertEqual(reloaded.get("vad_silence_ms"), 400)

    def test_tts_expression_controls_round_trip(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        dialog.rate_spin.setValue(0.9)
        dialog.pitch_spin.setValue(0.85)
        dialog.volume_spin.setValue(70)
        dialog.instruction_edit.setText("calm, confident academic English")
        dialog._save()
        reloaded = DefenseSettings(base_dir=Path(self._tmp.name))
        self.assertAlmostEqual(reloaded.get("tts_rate"), 0.9)
        self.assertAlmostEqual(reloaded.get("tts_pitch"), 0.85)
        self.assertEqual(reloaded.get("tts_volume"), 70)
        self.assertEqual(reloaded.get("tts_instruction"), "calm, confident academic English")

    def test_engine_tts_settings_use_expression_values(self) -> None:
        self.settings.update(
            {
                "tts_voice_id": "voice-x",
                "tts_rate": 0.9,
                "tts_pitch": 0.85,
                "tts_instruction": "calm and confident",
            }
        )
        self.settings.shared.update({"workspace_id": "llm-demo"})
        engine_settings = self.window.engine._tts_settings()
        self.assertEqual(engine_settings["tts_rate"], 0.9)
        self.assertEqual(engine_settings["tts_pitch"], 0.85)
        self.assertEqual(engine_settings["tts_instruction"], "calm and confident")

    def test_dialogs_construct(self) -> None:
        self.assertIsInstance(SelfCheckDialog(self.window, self.settings), SelfCheckDialog)
        self.assertIsInstance(RehearsalDialog(self.window, self.settings), RehearsalDialog)

    def test_clear_memory(self) -> None:
        window = self.window
        window.engine.clear_memory()
        window.timeline.setRowCount(0)
        window._timeline_rows.clear()
        self.assertEqual(window.timeline.rowCount(), 0)

    def test_qa_panel_updates_question_and_answer(self) -> None:
        panel = self.window.qa_panel
        panel._on_qa_status("正在生成…")
        self.assertEqual(panel.status_label.text(), "正在生成…")
        panel._on_qa_answer("We add Tikhonov regularization.", "我们加入 Tikhonov 正则化。")
        self.assertEqual(panel.answer_en.text(), "We add Tikhonov regularization.")
        self.assertIn("正则化", panel.answer_zh.text())
        self.panel = panel

    def test_committee_done_feeds_qa_question(self) -> None:
        panel = self.window.qa_panel
        panel.auto_check.setChecked(False)
        self.window._on_committee_done(9, "What is the main limitation?", "主要局限是什么？")
        self.assertEqual(panel.question_label.text(), "What is the main limitation?")
        self.assertEqual(panel._answer_en, "")

    def test_qa_model_persisted(self) -> None:
        panel = self.window.qa_panel
        self.assertFalse(panel.model_combo.isEditable())
        index = panel.model_combo.findData("qwen-max")
        self.assertGreaterEqual(index, 0, "模型清单里应有 qwen-max")
        panel.model_combo.setCurrentIndex(index)
        reloaded = DefenseSettings(base_dir=Path(self._tmp.name))
        self.assertEqual(reloaded.get("qa_model"), "qwen-max")
        panel.model_combo.setCurrentIndex(panel.model_combo.findData("qwen-plus"))

    def test_settings_model_combos_non_editable_with_blurbs(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        for combo in (dialog.tts_model_combo, dialog.llm_combo, dialog.asr_combo):
            self.assertFalse(combo.isEditable())
            self.assertIn("·", combo.currentText(), "每个模型项都应带一句简介")
        self.assertEqual(dialog.tts_model_combo.currentData(), "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(dialog.llm_combo.currentData(), "qwen-plus")
        self.assertGreaterEqual(dialog.llm_combo.count(), 5, "翻译模型列表应足够丰富")

    def test_clone_voice_dialog_constructs(self) -> None:
        dialog = CloneVoiceDialog(self.window, self.settings, "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(dialog.passage_edit.toPlainText()[:5], "各位老师好")
        self.assertFalse(dialog.clone_button.isEnabled())
        dialog._set_sample(Path("nonexistent.wav"), 15.0)
        self.assertTrue(dialog.clone_button.isEnabled())

    def test_clone_dialog_channel_is_automatic(self) -> None:
        from teams_voice_translator.defense.ui import clone_channel_for_model

        self.assertEqual(clone_channel_for_model("qwen3-tts-vc-realtime-2026-01-15"), "qwen3")
        self.assertEqual(clone_channel_for_model("cosyvoice-v3.5-plus"), "legacy")
        self.assertEqual(clone_channel_for_model("qwen-audio-3.0-tts-plus"), "legacy")

        # 防呆：下拉里只有唯一正确通道，不可能选错
        cosy_dialog = CloneVoiceDialog(self.window, self.settings, "cosyvoice-v3.5-plus")
        self.assertEqual(cosy_dialog.channel, "legacy")
        self.assertEqual(cosy_dialog.channel_combo.count(), 1)
        self.assertEqual(str(cosy_dialog.channel_combo.currentData()), "legacy")
        self.assertTrue(cosy_dialog.oss_group.isVisibleTo(cosy_dialog))
        self.assertFalse(cosy_dialog.use_transcript.isEnabled())
        cosy_dialog.close()

        qwen_dialog = CloneVoiceDialog(self.window, self.settings, "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(qwen_dialog.channel_combo.count(), 1)
        self.assertEqual(str(qwen_dialog.channel_combo.currentData()), "qwen3")
        self.assertFalse(qwen_dialog.oss_group.isVisibleTo(qwen_dialog))
        self.assertTrue(qwen_dialog.use_transcript.isEnabled())
        qwen_dialog.close()

    def test_cable_output_locked_to_cable_input(self) -> None:
        from types import SimpleNamespace

        dialog = SettingsDialog(self.window, self.settings)
        outputs = [
            SimpleNamespace(index=3, name="CU34G2XP (HD Audio)"),
            SimpleNamespace(index=7, name="CABLE Input (VB-Audio Virtual Cable)"),
            SimpleNamespace(index=9, name=" Speakers (Realtek)"),
        ]
        combo = dialog.cable_combo
        result = dialog._apply_cable_lock(combo, outputs)
        self.assertEqual(result, 7)
        self.assertFalse(combo.isEnabled(), "CABLE Input 应被锁定不可改")
        self.assertEqual(combo.currentData(), 7)
        self.assertIn("CABLE Input", combo.currentText())
        self.assertIn("已锁定", dialog.cable_note.text())
        # 解锁按钮：解锁后可手动选择，锁定后恢复
        dialog._toggle_cable_lock()
        self.assertTrue(combo.isEnabled())
        self.assertEqual(dialog.cable_lock_button.text(), "🔓 已解锁")
        combo.setCurrentIndex(combo.findData(9))
        dialog._toggle_cable_lock()
        self.assertFalse(combo.isEnabled())
        self.assertEqual(combo.currentData(), 7, "重新锁定应回到 CABLE Input")
        # 找不到 CABLE Input 时提示并禁用锁定按钮
        result2 = dialog._apply_cable_lock(combo, outputs[:1])
        self.assertIsNone(result2)
        self.assertTrue(combo.isEnabled())
        self.assertFalse(dialog.cable_lock_button.isEnabled())
        self.assertIn("未检测到", dialog.cable_note.text())

    def test_loopback_row_locks_to_active_device(self) -> None:
        from types import SimpleNamespace

        loopbacks = [
            SimpleNamespace(index=30, name="CU34G2XP [Loopback]"),
            SimpleNamespace(index=31, name="Speakers (HECATE G4) [Loopback]"),
        ]
        dialog = SettingsDialog(
            self.window, self.settings, active_loopback_name="Speakers (HECATE G4)"
        )
        dialog.loopback_combo.clear()
        for lb in loopbacks:
            dialog.loopback_combo.addItem(f"[{lb.index}] {lb.name}", lb.index)
        dialog._apply_loopback_lock_state()
        self.assertFalse(dialog.loopback_combo.isEnabled(), "默认锁定实际监听设备")
        self.assertEqual(dialog.loopback_combo.currentData(), 31)
        dialog._toggle_loopback_lock()
        self.assertTrue(dialog.loopback_combo.isEnabled())
        self.assertEqual(dialog.loopback_lock_button.text(), "🔓 已解锁")

    def test_committee_loopback_hides_program_output_cable(self) -> None:
        from types import SimpleNamespace
        loopbacks = [
            SimpleNamespace(index=28, name="CABLE In 16ch (VB-Audio Virtual Cable) [Loopback]"),
            SimpleNamespace(index=29, name="Speakers (HECATE G4) [Loopback]"),
        ]
        self.settings.shared.update({"loopback_device": 28})
        with patch("teams_voice_translator.defense.ui._safe_list_audio_devices", return_value=([], [])), \
             patch("teams_voice_translator.defense.ui._safe_list_loopback_devices", return_value=loopbacks):
            dialog = SettingsDialog(self.window, self.settings)
        self.addCleanup(dialog.close)
        labels = [dialog.loopback_combo.itemText(i) for i in range(dialog.loopback_combo.count())]
        self.assertFalse(any("CABLE" in label for label in labels))
        self.assertEqual(dialog.loopback_combo.currentData(), 29)

    def test_loopback_combo_highlights_active_device(self) -> None:
        from types import SimpleNamespace
        from teams_voice_translator.defense.ui import _device_combo

        combo = __import__("teams_voice_translator.defense.ui", fromlist=["NoWheelComboBox"]).NoWheelComboBox()
        items = [
            SimpleNamespace(index=3, name="Speakers (Realtek)"),
            SimpleNamespace(index=8, name="Speakers (HECATE G4 TE GAMING HEADSET)"),
        ]
        _device_combo(combo, items, None, allow_default="系统默认输出", direction="回环", highlight_name="HECATE")
        self.assertEqual(combo.currentData(), 8, "应自动选中实际监听的设备")
        fg = combo.itemData(2, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ForegroundRole)
        self.assertIsNotNone(fg, "监听中的设备应黄色高亮")

    def test_settings_voice_model_hint(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        # 未选择音色 → 橙色提示
        dialog.voice_combo.clear()
        dialog._update_voice_hint()
        self.assertIn("尚未填写", dialog.voice_hint.text())
        # 已知绑定的音色且家族匹配 → 绿色
        self.settings.set(
            "voice_library",
            [{"voice_id": "voice-x", "target_model": "qwen3-tts-vc-realtime-2026-01-15",
              "name": "x", "channel": "qwen3", "created_at": "t"}],
        )
        dialog._refresh_voice_combo(keep="voice-x")
        dialog._update_voice_hint()
        self.assertIn("匹配", dialog.voice_hint.text())
        # 家族不匹配 → 红色警告
        self.settings.set(
            "voice_library",
            self.settings.get("voice_library")
            + [{"voice_id": "voice-b", "target_model": "cosyvoice-v3.5-plus",
                "name": "b", "channel": "legacy", "created_at": "t"}],
        )
        dialog._refresh_voice_combo(keep="voice-b")
        dialog._update_voice_hint()
        self.assertIn("不是同一绑定模型", dialog.voice_hint.text())

    def test_voice_library_round_trip_and_switch(self) -> None:
        entries = [
            {"voice_id": "voice-a", "target_model": "qwen3-tts-vc-realtime-2026-01-15",
             "name": "Qwen3 版", "channel": "qwen3", "created_at": "2026-09-05 20:00"},
            {"voice_id": "voice-b", "target_model": "cosyvoice-v3.5-plus",
             "name": "CosyVoice 版", "channel": "legacy", "created_at": "2026-09-05 20:10"},
        ]
        self.settings.set("voice_library", entries)
        reloaded = DefenseSettings(base_dir=Path(self._tmp.name))
        self.assertEqual(len(reloaded.get("voice_library")), 2)

        dialog = CloneVoiceDialog(self.window, self.settings, "qwen3-tts-vc-realtime-2026-01-15")
        # 之前测试保存过的当前音色也会被自动收录，因此是 2+1=3 行
        self.assertEqual(dialog.library_table.rowCount(), 3)
        dialog.library_table.selectRow(0)
        dialog._use_selected_voice()
        self.assertEqual(dialog.created_voice_id, "voice-a")

    def test_current_voice_auto_shown_in_library(self) -> None:
        self.settings.set("tts_voice_id", "voice-current")
        self.settings.set("tts_model", "qwen3-tts-vc-realtime-2026-01-15")
        dialog = CloneVoiceDialog(self.window, self.settings, "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(dialog.library_table.rowCount(), 1)
        self.assertIn("当前", dialog.library_table.item(0, 0).text())
        self.assertEqual(dialog.library_table.item(0, 1).text(), "voice-current")

    def test_voice_library_mismatch_detected(self) -> None:
        dialog = CloneVoiceDialog(self.window, self.settings, "qwen3-tts-vc-realtime-2026-01-15")
        self.assertEqual(dialog._voice_family("cosyvoice-v3.5-plus"), "cosyvoice")
        self.assertEqual(dialog._voice_family("qwen3-tts-vc-2026-01-22"), "qwen3")
        dialog._remember_voice("voice-new")
        self.assertTrue(any(entry["voice_id"] == "voice-new" for entry in dialog._library_entries()))

    def test_settings_dialog_has_clone_button(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        self.assertTrue(hasattr(dialog, "_open_clone_voice"))

    def test_connection_test_button_exists_and_guards_missing_credentials(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        self.assertTrue(hasattr(dialog, "_run_connection_test"))
        dialog.workspace_edit.setText("")
        dialog.api_key_edit.setText("")
        self.settings.shared.update({"workspace_id": ""})
        dialog._run_connection_test()
        self.assertIn("请先填写", dialog.test_result_label.text())
        self.assertTrue(dialog.test_button.isEnabled())

    def test_connection_test_appends_lines(self) -> None:
        dialog = SettingsDialog(self.window, self.settings)
        # 只测信号到标签的纯 UI 链路，不触发网络（_run_connection_test 会真调 API）。
        dialog._on_test_line("<span>✅ 假结果</span>")
        dialog._on_test_line("<span>❌ 假失败</span>")
        self.assertIn("假结果", dialog.test_result_label.text())
        self.assertIn("假失败", dialog.test_result_label.text())


if __name__ == "__main__":
    unittest.main()
