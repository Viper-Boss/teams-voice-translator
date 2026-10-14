"""第 20 批：问答库匹配严格度 + 第二通道（谁先出字用谁，只说一次）。"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.aliyun import ApiError  # noqa: E402
from teams_voice_translator.defense.channels import (  # noqa: E402
    BackupChannel, backup_channel, backup_endpoint, channel_summary, primary_endpoint,
)
from teams_voice_translator.defense.translator import ContextTranslator  # noqa: E402

PRIMARY = primary_endpoint("ws1")
SECOND = backup_endpoint("beijing-public")


def _translator(mode="race", delay=0.3):
    backup = BackupChannel(endpoint=SECOND, api_key="k2", label="北京公共入口", mode=mode, delay=delay)
    return ContextTranslator(api_key="k1", workspace_id="ws1", model="m", backup=backup)


def _fake(behaviour, calls):
    """behaviour[endpoint] = [(delay, piece or Exception), ...]；按顺序吐片段，可中途抛错。"""

    def stream_once(endpoint, api_key, payload, on_piece, cancel_event, *, on_reasoning=None, deadline_seconds=None):
        calls.append((endpoint, api_key, payload.get("model")))
        parts = []
        for delay, item in behaviour[endpoint]:
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    raise ApiError("已取消")
                time.sleep(0.005)
            if cancel_event is not None and cancel_event.is_set():
                raise ApiError("已取消")
            if isinstance(item, Exception):
                raise item
            parts.append(item)
            on_piece(item, "".join(parts))
        return "".join(parts)

    return stream_once


class RaceTests(unittest.TestCase):
    def _run(self, translator, behaviour):
        calls, got, threads = [], [], []
        translator._stream_once = _fake(behaviour, calls)

        def on_piece(piece, so_far):
            got.append(piece)
            threads.append(threading.current_thread())

        result = translator._post_stream({"model": "m", "messages": []}, on_piece, None)
        return result, got, calls, threads

    def test_谁先出字用谁_另一个的字一个都不用_回调在调用方线程(self):
        result, got, calls, threads = self._run(_translator(), {
            PRIMARY: [(0.4, "Slow "), (0.0, "primary.")],
            SECOND: [(0.05, "Fast "), (0.05, "second.")],
        })
        self.assertEqual(result, "Fast second.")
        self.assertEqual(got, ["Fast ", "second."], "只说一次：主通道的字一个都没进来")
        self.assertTrue(all(t is threading.current_thread() for t in threads))
        self.assertEqual({c[0] for c in calls}, {PRIMARY, SECOND})

    def test_赢家出过字后中途断了_报错_不换通道续说(self):
        translator = _translator()
        with self.assertRaises(ApiError):
            self._run(translator, {
                PRIMARY: [(0.02, "The model "), (0.05, ApiError("连接中断"))],
                SECOND: [(0.2, "The model "), (0.0, "works.")],
            })

    def test_还没出字就失败的出局_用另一个(self):
        result, got, _calls, _t = self._run(_translator(), {
            PRIMARY: [(0.02, ApiError("限流"))],
            SECOND: [(0.1, "OK.")],
        })
        self.assertEqual((result, got), ("OK.", ["OK."]))

    def test_两个都失败报主通道的错(self):
        with self.assertRaises(ApiError) as ctx:
            self._run(_translator(), {
                PRIMARY: [(0.02, ApiError("主通道挂了"))],
                SECOND: [(0.05, ApiError("第二通道也挂了"))],
            })
        self.assertIn("主通道挂了", str(ctx.exception))

    def test_稍后模式_主通道及时出字就不发第二通道(self):
        result, _got, calls, _t = self._run(_translator(mode="delayed", delay=0.3), {
            PRIMARY: [(0.05, "Quick.")],
            SECOND: [(0.0, "never")],
        })
        self.assertEqual(result, "Quick.")
        self.assertEqual([c[0] for c in calls], [PRIMARY])

    def test_稍后模式_主通道卡住就发第二通道并用它(self):
        result, got, calls, _t = self._run(_translator(mode="delayed", delay=0.1), {
            PRIMARY: [(1.5, "late")],
            SECOND: [(0.05, "Backup.")],
        })
        self.assertEqual((result, got), ("Backup.", ["Backup."]))
        self.assertEqual([c[0] for c in calls], [PRIMARY, SECOND])

    def test_第二通道可以用别的模型_没配第二通道时和原来一样(self):
        backup = BackupChannel(endpoint=SECOND, api_key="k2", label="x", model="other-model")
        translator = ContextTranslator(api_key="k1", workspace_id="ws1", model="m", backup=backup)
        _r, _g, calls, _t = self._run(translator, {PRIMARY: [(0.2, "a")], SECOND: [(0.01, "b")]})
        self.assertIn((SECOND, "k2", "other-model"), calls)
        plain = ContextTranslator(api_key="k1", workspace_id="ws1", model="m")
        _r, got, calls, _t = self._run(plain, {PRIMARY: [(0.0, "only")]})
        self.assertEqual((got, [c[0] for c in calls]), (["only"], [PRIMARY]))

    def test_取消时两个通道都停(self):
        translator = _translator()
        calls = []
        translator._stream_once = _fake({PRIMARY: [(2.0, "x")], SECOND: [(2.0, "y")]}, calls)
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        with self.assertRaises(ApiError):
            translator._post_stream({"model": "m", "messages": []}, lambda *_a: None, cancel)


class ChannelSettingsTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def test_没开或缺Key就只有主通道_北京公共入口用主Key(self):
        with patch.object(type(self.settings), "get_api_key", return_value="main-key"):
            self.assertIsNone(backup_channel(self.settings))
            self.settings.set("backup_channel_enabled", True)
            channel = backup_channel(self.settings)
            self.assertEqual((channel.endpoint, channel.api_key),
                             ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "main-key"))
            self.assertIn("北京", channel_summary(self.settings))
            self.settings.set("backup_route", "cn-hongkong")
            with patch("teams_voice_translator.defense.channels.get_backup_api_key", return_value=""):
                self.assertIsNone(backup_channel(self.settings), "别的地域没有 Key 就不启用")
                self.assertIn("缺 API Key", channel_summary(self.settings))
            with patch("teams_voice_translator.defense.channels.get_backup_api_key", return_value="hk-key"):
                self.settings.set("backup_workspace_id", "wsHK")
                channel = backup_channel(self.settings)
                self.assertEqual(channel.endpoint,
                                 "https://wsHK.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(backup_endpoint("ap-southeast-1"),
                         "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")

    def test_引擎和助答都带上第二通道(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks

        self.settings.set("backup_channel_enabled", True)
        self.settings.shared.set_api_key = lambda *_a: None
        with patch.object(type(self.settings), "get_api_key", return_value="main-key"), \
                patch.object(type(self.settings), "workspace_id", new="ws1", create=True):
            engine = DefenseEngine(self.settings, EngineCallbacks())
            self.addCleanup(engine.shutdown)
            engine._build_clients()
            self.assertIsNotNone(engine.translator.backup)


class StrictnessTests(unittest.TestCase):
    def test_严格适中宽松门槛(self):
        from teams_voice_translator.defense.qa_bank import QaEntry, match_question, strictness_thresholds

        entries = [QaEntry("A", "Why did you choose MFL?", "Because.")]
        chat = lambda _m: '{"id": "A", "confidence": 0.7}'   # noqa: E731
        strict = strictness_thresholds("strict")
        loose = strictness_thresholds("loose")
        self.assertIsNone(match_question(entries, "Why MFL?", chat, min_confidence=strict[0],
                                         lexical_strong=strict[1]).entry)
        self.assertIsNotNone(match_question(entries, "Why MFL?", chat, min_confidence=loose[0],
                                            lexical_strong=loose[1]).entry)
        self.assertEqual(strictness_thresholds("medium"), (0.6, 0.55))
        self.assertEqual(strictness_thresholds("???"), (0.6, 0.55))


class UiTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.settings import DefenseSettings

        QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))

    def test_助答面板严格度默认适中带星_改了会保存(self):
        from teams_voice_translator.defense.ui import QaPanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QaPanel(self.settings, engine)
        self.addCleanup(panel.shutdown)
        combo = panel.strictness_combo
        self.assertEqual(combo.currentData(), "medium")
        self.assertTrue(combo.currentText().startswith("★"))
        combo.setCurrentIndex(combo.findData("loose"))
        self.assertEqual(self.settings.get("qa_match_strictness"), "loose")

    def test_设置页双通道在账号与网络页_显示两个通道的地区_保存(self):
        from teams_voice_translator.defense import ui

        dialog = ui.SettingsDialog(None, self.settings)
        tabs = dialog.settings_tabs
        page = tabs.widget([tabs.tabText(i) for i in range(tabs.count())].index("账号与网络"))
        self.assertTrue(page.isAncestorOf(dialog.backup_check))
        self.assertIn("第一通道：华北2（北京）", dialog.channel_summary_label.text())
        self.assertIn("未开启", dialog.channel_summary_label.text())
        dialog.backup_check.setChecked(True)
        dialog.backup_route_combo.setCurrentIndex(dialog.backup_route_combo.findData("ap-southeast-1"))
        self.assertIn("第二通道：新加坡", dialog.channel_summary_label.text())
        self.assertTrue(dialog.backup_key_edit.isEnabled())
        dialog.backup_key_edit.setText("sg-key")
        dialog.backup_model_edit.setText("qwen-flash")
        with patch("teams_voice_translator.defense.ui.QMessageBox"), \
                patch.object(ui, "set_backup_api_key") as saver:
            dialog._save()
        saver.assert_called_once_with("sg-key")
        self.assertEqual((self.settings.get("backup_channel_enabled"), self.settings.get("backup_route"),
                          self.settings.get("backup_model")), (True, "ap-southeast-1", "qwen-flash"))

    def test_测试两个通道显示各自耗时(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        dialog.workspace_edit.setText("ws1")
        dialog.api_key_edit.setText("k1")
        dialog.backup_check.setChecked(True)
        seen = []

        def runner(endpoint, key, model):
            seen.append(endpoint)
            if "dashscope" in endpoint:
                raise ApiError("403 无权限")

        dialog.channel_test_runner = runner
        dialog._test_channels()
        deadline = time.monotonic() + 5
        while "第二通道" not in dialog.channel_test_label.text() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        text = dialog.channel_test_label.text()
        self.assertIn("第一通道 ✅", text)
        self.assertIn("第二通道 ❌ 403", text)


if __name__ == "__main__":
    unittest.main()
