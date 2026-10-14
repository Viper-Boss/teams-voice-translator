"""第 19 批：设置里现场 / 提前渲染两套「模型 + 音色」分框分色；音色库管理（改名、删除）。"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

LIBRARY = [
    {"voice_id": "v-plus-old", "target_model": "qwen-audio-3.0-tts-plus", "name": "旧 plus", "created_at": "2026-09-01"},
    {"voice_id": "v-plus-new", "target_model": "qwen-audio-3.0-tts-plus", "name": "新 plus", "created_at": "2026-09-20"},
    {"voice_id": "v-flash", "target_model": "qwen-audio-3.1-tts-flash", "name": "flash", "created_at": "2026-09-10"},
    {"voice_id": "v-qwen3", "target_model": "qwen3-tts-vc-2026-01-22", "name": "qwen3", "created_at": "2026-09-11"},
]


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(predicate, seconds=5.0):
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _Base(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.settings.update({"voice_library": [dict(e) for e in LIBRARY],
                              "tts_model": "qwen-audio-3.1-tts-flash", "tts_voice_id": "v-flash",
                              "prerender_tts_model": "qwen-audio-3.0-tts-plus", "prerender_voice_id": "v-plus-old"})


class VoiceFramesTests(_Base):
    def test_两套模型音色分框_都在声音页(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        live, pre = dialog._voice_frames
        self.assertTrue(live.isAncestorOf(dialog.voice_combo))
        self.assertTrue(live.isAncestorOf(dialog.tts_model_combo))
        self.assertTrue(pre.isAncestorOf(dialog.prerender_model_combo))
        self.assertTrue(pre.isAncestorOf(dialog.prerender_voice_combo))
        self.assertNotEqual(live.styleSheet(), pre.styleSheet(), "两个框颜色不同")
        page = dialog.settings_tabs.widget(0)
        self.assertTrue(page.isAncestorOf(live) and page.isAncestorOf(pre))

    def test_提前渲染音色只列匹配的_换模型自动换音色_保存成对(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        combo = dialog.prerender_voice_combo
        self.assertEqual(combo.currentData(), ("qwen-audio-3.0-tts-plus", "v-plus-old"))
        model = combo.model()
        self.assertFalse(model.item(next(i for i in range(combo.count())
                                         if combo.itemData(i) and combo.itemData(i)[1] == "v-flash")).isEnabled())
        dialog.prerender_model_combo.setCurrentIndex(dialog.prerender_model_combo.findData("qwen-audio-3.1-tts-flash"))
        self.assertEqual(combo.currentData(), ("qwen-audio-3.1-tts-flash", "v-flash"))
        dialog.prerender_model_combo.setCurrentIndex(dialog.prerender_model_combo.findData("cosyvoice-v3.5-plus"))
        self.assertIn("❌", dialog.prerender_hint.text(), "没有匹配的音色要明确提示")
        with patch("teams_voice_translator.defense.ui.QMessageBox"):
            dialog._save()
        self.assertEqual((self.settings.get("prerender_tts_model"), self.settings.get("prerender_voice_id")), ("", ""),
                         "没有能用的音色时退回「和现场相同」，免得渲染失败")
        dialog = SettingsDialog(None, self.settings)
        self.assertFalse(dialog.prerender_voice_combo.isEnabled())


class VoiceLibraryTests(_Base):
    def _dialog(self):
        from teams_voice_translator.defense.ui import VoiceLibraryDialog

        dialog = VoiceLibraryDialog(None, self.settings)
        dialog._confirm = lambda text: True
        return dialog

    def _select(self, dialog, voice_id):
        for row in range(dialog.table.rowCount()):
            if dialog.table.item(row, 0).data(0x0100) == voice_id:
                dialog.table.selectRow(row)
                return
        raise AssertionError(voice_id)

    def test_只从本机删_正在用的自动换成同模型最新的(self):
        dialog = self._dialog()
        self._select(dialog, "v-plus-old")
        dialog.delete_selected(remote=False)
        ids = [e["voice_id"] for e in self.settings.get("voice_library")]
        self.assertNotIn("v-plus-old", ids)
        self.assertEqual(self.settings.get("prerender_voice_id"), "v-plus-new")
        self._select(dialog, "v-flash")
        dialog.delete_selected(remote=False)
        self.assertEqual(self.settings.get("tts_voice_id"), "", "同模型没有别的音色就清空，不留指向已删音色的设置")
        self.assertTrue(dialog.changed)

    def test_连阿里云一起删_成功才从本机删_失败保留(self):
        dialog = self._dialog()
        client = MagicMock()
        client.delete_voice.side_effect = [None, RuntimeError("Forbidden")]
        dialog.client_factory = lambda: client
        self._select(dialog, "v-qwen3")
        dialog.delete_selected(remote=True)
        self.assertTrue(_wait(lambda: "v-qwen3" not in [e["voice_id"] for e in self.settings.get("voice_library")]))
        client.delete_voice.assert_called_with("v-qwen3", "qwen3-tts-vc-2026-01-22")
        self._select(dialog, "v-plus-new")
        dialog.delete_selected(remote=True)
        self.assertTrue(_wait(lambda: "没删成" in dialog.status_label.text()))
        self.assertIn("v-plus-new", [e["voice_id"] for e in self.settings.get("voice_library")])

    def test_不确认就不删_改名和设为现场(self):
        dialog = self._dialog()
        dialog._confirm = lambda text: False
        self._select(dialog, "v-flash")
        dialog.delete_selected(remote=False)
        self.assertEqual(len(self.settings.get("voice_library")), 4)
        dialog.table.item(0, 0).setText("我的旧声音")
        self.assertEqual(self.settings.get("voice_library")[0]["name"], "我的旧声音")
        self._select(dialog, "v-plus-new")
        dialog._assign("live")
        self.assertEqual((self.settings.get("tts_voice_id"), self.settings.get("tts_model")),
                         ("v-plus-new", "qwen-audio-3.0-tts-plus"))

    def test_设置页打开音色库删掉后下拉跟着变(self):
        from teams_voice_translator.defense import ui

        dialog = ui.SettingsDialog(None, self.settings)

        class FakeLibrary:
            def __init__(self, parent, settings):
                self.settings = settings
                self.changed = True

            def exec(self):
                library = [e for e in self.settings.get("voice_library") if e["voice_id"] != "v-flash"]
                self.settings.update({"voice_library": library, "tts_voice_id": ""})

        with patch.object(ui, "VoiceLibraryDialog", FakeLibrary):
            dialog._open_voice_library()
        self.assertEqual(dialog.voice_combo.findData("v-flash"), -1)


class DeleteVoiceApiTests(unittest.TestCase):
    def _client(self):
        from teams_voice_translator.aliyun import BailianClient

        return BailianClient("key", "ws")

    def test_按模型家族选删除接口_云上已不存在算成功(self):
        client = self._client()
        ok = MagicMock(ok=True, status_code=200)
        with mock.patch("teams_voice_translator.aliyun.requests.post", return_value=ok) as post:
            client.delete_voice("v1", "qwen3-tts-vc-2026-01-22")
            client.delete_voice("v2", "cosyvoice-v3.5-plus")
        first, second = post.call_args_list[0].kwargs["json"], post.call_args_list[1].kwargs["json"]
        self.assertEqual(first, {"model": "qwen-voice-enrollment", "input": {"action": "delete", "voice": "v1"}})
        self.assertEqual(second, {"model": "voice-enrollment", "input": {"action": "delete_voice", "voice_id": "v2"}})
        missing = MagicMock(ok=False, status_code=404)
        with mock.patch("teams_voice_translator.aliyun.requests.post", return_value=missing), \
                mock.patch("teams_voice_translator.aliyun._extract_api_error", return_value="voice not exist"):
            client.delete_voice("v3", "cosyvoice-v3.5-plus")
        from teams_voice_translator.aliyun import ApiError
        with self.assertRaises(ApiError):
            client.delete_voice("v4", "MiniMax/speech-2.8-hd")


if __name__ == "__main__":
    unittest.main()
