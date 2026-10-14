"""第 17 批：术语表独立窗口 + 自动生成、声卡只列有用的、更多英文口吻、音色按模型分组、
背景环境场景 + 音量上限、助答按钮不截字、快捷用语自己录音 + 按问题推荐、余额 / 用量 / 消费记录。"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
    QApplication.processEvents()
    return predicate()


class _Base(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.settings import DefenseSettings

        _app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.settings = DefenseSettings(base_dir=self.base)


# ------------------------------------------------------------------ 声卡
class DeviceFilterTests(unittest.TestCase):
    def _devices(self):
        from teams_voice_translator.audio import AudioDevice

        rows = [
            (0, "Microsoft 声音映射器 - Input", "MME", 2, 0),
            (1, "麦克风 (5- EDIFIER G4)", "MME", 2, 0),
            (2, "麦克风 (ToDesk Virtual Audio)", "MME", 2, 0),
            (3, "麦克风 (USB_MIC)", "MME", 1, 0),
            (4, "CABLE Output (VB-Audio Virtual", "MME", 2, 0),
            (5, "立体声混音 (Realtek USB Audio)", "MME", 2, 0),
            (13, "主声音捕获驱动程序", "Windows DirectSound", 2, 0),
            (14, "麦克风 (5- EDIFIER G4)", "Windows DirectSound", 2, 0),
            (36, "麦克风 (5- EDIFIER G4)", "Windows WASAPI", 2, 0),
            (42, "麦克风 ()", "Windows WDM-KS", 2, 0),
            (7, "扬声器 (Realtek USB Audio)", "MME", 0, 2),
            (8, "CABLE Input (VB-Audio Virtual C", "MME", 0, 2),
            (9, "扬声器 (ToDesk Virtual Audio)", "MME", 0, 2),
            (10, "耳机 (EDIFIER G4)", "MME", 0, 2),
            (20, "CABLE Input (VB-Audio Virtual Cable)", "Windows DirectSound", 0, 2),
        ]
        return [AudioDevice(i, n, h, a, b) for i, n, h, a, b in rows]

    def test_麦克风只留真麦克风_去重去虚拟(self):
        from teams_voice_translator.audio import useful_devices

        inputs = [d for d in self._devices() if d.inputs]
        names = [d.name for d in useful_devices(inputs, kind="mic")]
        self.assertEqual(names, ["麦克风 (5- EDIFIER G4)", "麦克风 (USB_MIC)"])

    def test_程序输出只列CABLE_试听去掉虚拟(self):
        from teams_voice_translator.audio import useful_devices

        outputs = [d for d in self._devices() if d.outputs]
        self.assertEqual([d.index for d in useful_devices(outputs, kind="cable")], [8])
        self.assertEqual([d.name for d in useful_devices(outputs, kind="monitor")],
                         ["扬声器 (Realtek USB Audio)", "耳机 (EDIFIER G4)"])

    def test_已选的即使被筛掉也保留(self):
        from teams_voice_translator.audio import useful_devices

        inputs = [d for d in self._devices() if d.inputs]
        kept = useful_devices(inputs, kind="mic", keep_index=36)
        self.assertEqual(kept[0].index, 36)


class DeviceSettingsUiTests(_Base):
    def test_声卡选择一直显示_勾显示全部才全列(self):
        from teams_voice_translator.defense import ui

        devices = DeviceFilterTests()._devices()
        inputs = [d for d in devices if d.inputs]
        outputs = [d for d in devices if d.outputs]
        with patch.object(ui, "_safe_list_audio_devices", return_value=(inputs, outputs)), \
                patch.object(ui, "_safe_list_loopback_devices", return_value=[]):
            dialog = ui.SettingsDialog(None, self.settings)
        self.assertFalse(dialog.device_box.isHidden())
        self.assertEqual(dialog.mic_combo.count(), 1 + 2)       # 系统默认 + 两个真麦克风
        dialog.show_devices_check.setChecked(True)
        self.assertEqual(dialog.mic_combo.count(), 1 + len(inputs))
        self.assertTrue(self.settings.get("show_all_audio_devices"))


# ------------------------------------------------------------------ 英文口吻
class TranslationStyleTests(_Base):
    def test_新增口吻都进了现场翻译提示_未知值回到默认(self):
        from teams_voice_translator.defense.memory import STYLE_RULES, TRANSLATION_STYLES, MeetingMemory

        self.assertGreaterEqual(len(TRANSLATION_STYLES), 8)
        for style_id, _label, rule, _hint in TRANSLATION_STYLES:
            memory = MeetingMemory(translation_style=style_id)
            self.assertIn(rule, memory.system_prompt("zh2en"))
        self.assertIn(STYLE_RULES["accurate"], MeetingMemory(translation_style="nope").system_prompt("zh2en"))

    def test_设置页口吻下拉和提示范围(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        combo = dialog.translation_style_combo
        self.assertGreaterEqual(combo.count(), 8)
        self.assertIn("讲稿台", combo.toolTip())
        combo.setCurrentIndex(combo.findData("simple"))
        with patch("teams_voice_translator.defense.ui.QMessageBox"):
            dialog._save()
        self.assertEqual(self.settings.get("translation_style"), "simple")


# ------------------------------------------------------------------ 音色按模型分组
class VoiceGroupingTests(_Base):
    def test_匹配的在上面可选_不匹配的灰掉(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        self.settings.update({
            "tts_model": "qwen-audio-3.0-tts-plus", "tts_voice_id": "v-plus",
            "voice_library": [
                {"voice_id": "v-plus", "target_model": "qwen-audio-3.0-tts-plus", "name": "P"},
                {"voice_id": "v-cosy", "target_model": "cosyvoice-v3.5-plus", "name": "C"},
            ]})
        dialog = SettingsDialog(None, self.settings)
        combo = dialog.voice_combo
        model = combo.model()
        plus = combo.findData("v-plus")
        cosy = combo.findData("v-cosy")
        self.assertLess(plus, cosy)
        self.assertTrue(model.item(plus).isEnabled())
        self.assertFalse(model.item(cosy).isEnabled())
        # 换成 cosyvoice 模型：分组反过来
        dialog.tts_model_combo.setCurrentIndex(dialog.tts_model_combo.findData("cosyvoice-v3.5-plus"))
        self.assertEqual(combo.currentData(), "v-cosy")
        self.assertTrue(model.item(combo.findData("v-cosy")).isEnabled())
        self.assertFalse(model.item(combo.findData("v-plus")).isEnabled())


# ------------------------------------------------------------------ 背景环境
class AmbienceTests(_Base):
    def test_所有场景音量拉满也不超过上限(self):
        from teams_voice_translator.defense.ambience import (AMBIENCE_PRESETS, MAX_LEVEL_DB, PEAK_CAP_DB,
                                                             RoomTone, effective_level_db)

        cap = 32767 * 10 ** (PEAK_CAP_DB / 20)
        self.assertGreaterEqual(len(AMBIENCE_PRESETS), 12)
        for mode in AMBIENCE_PRESETS:
            self.assertLessEqual(effective_level_db(mode, 100), MAX_LEVEL_DB)
            tone = RoomTone(mode, volume=100, seed=3)
            samples = array("h", tone.bed(24000 * 3) + tone.tail())
            if samples:
                self.assertLessEqual(max(abs(v) for v in samples), cap + 1, mode)

    def test_机场高铁有提示音事件_音量越大电平越高(self):
        from teams_voice_translator.defense.ambience import AMBIENCE_PRESETS, effective_level_db

        self.assertIn("chime", AMBIENCE_PRESETS["airport"].events)
        self.assertIn("chime", AMBIENCE_PRESETS["rail"].events)
        self.assertLess(effective_level_db("subtle", 0), effective_level_db("subtle", 50))
        self.assertLess(effective_level_db("subtle", 50), effective_level_db("subtle", 100))

    def test_设置页音量滑块保存_引擎按音量重建(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        dialog.ambience_combo.setCurrentIndex(dialog.ambience_combo.findData("office"))
        dialog.ambience_volume_slider.setValue(100)
        self.assertIn("上限", dialog.ambience_volume_label.text())
        with patch("teams_voice_translator.defense.ui.QMessageBox"):
            dialog._save()
        self.assertEqual((self.settings.get("ambience_mode"), self.settings.get("ambience_volume")), ("office", 100))
        engine = DefenseEngine(self.settings, EngineCallbacks())
        self.addCleanup(engine.shutdown)
        self.assertEqual(engine._current_room_tone().volume, 100)


# ------------------------------------------------------------------ 术语表
class GlossaryGenTests(unittest.TestCase):
    def test_切块和容错解析(self):
        from teams_voice_translator.defense.glossary_gen import parse_terms, split_for_terms

        chunks = split_for_terms("段落一\n" + "字" * 30000 + "\n段落三", size=12000)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 12000 for c in chunks))
        messy = '好的：```json\n[{"source": "漏磁检测", "target": "MFL inspection"}, {"source": "", "target": "x"},'
        messy += ' {"source": "RMSE", "target": "RMSE"},]```'
        self.assertEqual(parse_terms(messy), [{"source": "漏磁检测", "target": "MFL inspection"},
                                              {"source": "RMSE", "target": "RMSE"}])

    def test_合并_新增冲突相同_按出现次数排序(self):
        from teams_voice_translator.defense.glossary_gen import generate_terms

        existing = [{"source": "知识蒸馏", "target": "Knowledge Distillation (KD)"},
                    {"source": "漏磁检测", "target": "Magnetic Flux Leakage (MFL) inspection"}]
        replies = iter([
            json.dumps([{"source": "小波去噪", "target": "Wavelet denoising"},
                        {"source": "知识蒸馏", "target": "Knowledge Distillation (KD)"}]),
            json.dumps([{"source": "小波去噪", "target": "Wavelet denoising"},
                        {"source": "漏磁检测", "target": "MFL detection"},
                        {"source": "结构化剪枝", "target": "Structured pruning"}]),
        ])
        seen = []

        def post(payload):
            seen.append(payload["messages"][1]["content"])
            return next(replies)

        text = ("甲" * 11000) + "\n" + ("乙" * 11000)
        result = generate_terms([("论文", text)], existing, model="qwen3.8-flash", post=post)
        self.assertEqual(result.chunks, 2)
        self.assertEqual([c.source for c in result.new], ["小波去噪", "结构化剪枝"])
        self.assertEqual(result.new[0].hits, 2)
        self.assertEqual([c.source for c in result.conflicts], ["漏磁检测"])
        self.assertEqual([c.source for c in result.same], ["知识蒸馏"])
        self.assertIn("小波去噪 → Wavelet denoising", seen[1], "后面的块带上前面找到的写法")

    def test_术语很多时翻译提示只带前60条和用到的(self):
        from teams_voice_translator.defense.memory import MeetingMemory

        terms = [{"source": f"术语{i}", "target": f"Term {i}"} for i in range(400)]
        memory = MeetingMemory(glossary=terms)
        block = memory.glossary_block("这里说到了术语321")
        lines = block.splitlines()
        self.assertLessEqual(len(lines), MeetingMemory.GLOSSARY_MAX)
        self.assertIn("术语0 → Term 0", block)
        self.assertIn("术语321 → Term 321", block)
        self.assertNotIn("术语200 → Term 200", block)
        small = MeetingMemory(glossary=terms[:20])
        self.assertEqual(len(small.glossary_block().splitlines()), 20)


class GlossaryDialogTests(_Base):
    def test_独立窗口_编辑排序保存(self):
        from teams_voice_translator.defense.ui import GlossaryDialog

        self.settings.set_glossary([{"source": "甲", "target": "A"}, {"source": "乙", "target": "B"},
                                    {"source": "丙", "target": "C"}])
        saved = []
        dialog = GlossaryDialog(None, self.settings, on_saved=lambda: saved.append(1))
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(len(dialog.terms()), 3)
        dialog.table.selectRow(2)
        dialog._to_top()
        self.assertEqual(dialog.terms()[0]["source"], "丙")
        self.assertTrue(dialog.save_button.text().endswith("*"))
        dialog.save()
        self.assertEqual(self.settings.glossary_terms()[0]["source"], "丙")
        self.assertEqual(saved, [1])
        self.assertIn("🔥", dialog.table.verticalHeaderItem(0).text())

    def test_从论文自动生成_核对后加入(self):
        from teams_voice_translator.defense.thesis_context import ThesisStore
        from teams_voice_translator.defense.ui import GlossaryDialog

        ThesisStore(self.base).save(source_path="t.pdf", chunks=[], page_count=1,
                                    full_text="本文研究漏磁检测与小波去噪。" * 10)
        self.settings.set_glossary([{"source": "漏磁检测", "target": "MFL inspection"}])
        dialog = GlossaryDialog(None, self.settings)
        self.addCleanup(dialog.deleteLater)
        self.assertTrue(dialog.use_thesis_check.isChecked())
        dialog.post_factory = lambda: (lambda payload: json.dumps([
            {"source": "小波去噪", "target": "Wavelet denoising"},
            {"source": "漏磁检测", "target": "MFL testing"}]))

        class FakeReview:
            def __init__(self, result):
                self.result = result

            def exec(self):
                from PySide6.QtWidgets import QDialog
                return QDialog.Accepted

            def chosen(self):
                return [c.as_term() for c in self.result.new]   # 只要新增的，不改已有写法

        dialog._review_dialog = FakeReview
        dialog.generate()
        self.assertTrue(_wait(lambda: dialog.generate_button.isEnabled()))
        terms = dialog.terms()
        self.assertEqual(terms, [{"source": "漏磁检测", "target": "MFL inspection"},
                                 {"source": "小波去噪", "target": "Wavelet denoising"}])
        self.assertTrue(dialog._dirty, "加入后要点保存才生效")

    def test_设置页不再有术语表格_保存设置不会冲掉术语(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        self.settings.set_glossary([{"source": "甲", "target": "A"}])
        dialog = SettingsDialog(None, self.settings)
        self.assertFalse(hasattr(dialog, "terms_table"))
        with patch("teams_voice_translator.defense.ui.QMessageBox"):
            dialog._save()
        self.assertEqual(self.settings.glossary_terms(), [{"source": "甲", "target": "A"}])


# ------------------------------------------------------------------ 助答按钮不截字
class QaButtonsTests(_Base):
    def test_按钮最小宽度够放下文字(self):
        from PySide6.QtGui import QFontMetrics
        from teams_voice_translator.defense.ui import QaPanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QaPanel(self.settings, engine)
        self.addCleanup(panel.shutdown)
        for button in (panel.bank_button, panel.restore_button, panel.regen_button, panel.save_bank_button):
            need = QFontMetrics(button.font()).horizontalAdvance(button.text())
            self.assertGreaterEqual(button.minimumWidth(), need, button.text())
        panel.change_font_size(6)
        need = QFontMetrics(panel.regen_button.font()).horizontalAdvance(panel.regen_button.text())
        self.assertGreaterEqual(panel.regen_button.minimumWidth(), need)


# ------------------------------------------------------------------ 快捷用语：自己录音 + 推荐
class PhraseRecordingTests(_Base):
    def test_录音剪掉首尾静音_小声放大(self):
        from teams_voice_translator.defense.quick_phrases import clean_recording

        quiet = array("h", [0] * 24000 + [3000, -3000] * 6000 + [0] * 24000).tobytes()
        cleaned = array("h", clean_recording(quiet))
        self.assertLess(len(cleaned), 12000 + 2 * int(24000 * 0.15) + 1000)
        self.assertGreater(max(abs(v) for v in cleaned), 3000)

    def test_录过音的点了播录音_不走合成(self):
        from teams_voice_translator.defense.ui import QuickPhrasePanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QuickPhrasePanel(self.settings, engine)
        phrase = panel.phrases[0]
        panel.recordings.save(phrase.id, b"\x01\x00" * 4800)
        panel.rebuild()
        self.assertTrue(panel._buttons[phrase.id].text().startswith("🎙"))
        self.assertNotIn(phrase.en, panel.texts(), "录过的不用再预渲染")
        panel.speak(phrase)
        engine.play_script_segment.assert_called_once()
        engine.speak_prepared.assert_not_called()
        panel.delete_recording(phrase)
        panel.speak(phrase)
        engine.speak_prepared.assert_called_once()

    def test_录音窗口_录好试听保存(self):
        from teams_voice_translator.defense.ui import PhraseRecordDialog, QuickPhrasePanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QuickPhrasePanel(self.settings, engine)
        phrase = panel.phrases[1]
        dialog = PhraseRecordDialog(panel, phrase)

        class FakeCapture:
            def __init__(self, callback):
                self.callback = callback

            def start(self):
                self.callback(array("h", [0] * 2400 + [4000, -4000] * 12000 + [0] * 2400).tobytes())

            def stop(self):
                pass

        written = []

        class FakePlayer:
            def write(self, data):
                written.append(len(data))

            def close(self):
                pass

        dialog.capture_factory = FakeCapture
        dialog.player_factory = FakePlayer
        dialog.toggle_record()
        dialog.toggle_record()
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.preview()
        self.assertTrue(_wait(lambda: sum(written) == len(dialog._pcm)))
        dialog.save()
        self.assertTrue(panel.recordings.has(phrase.id))

    def test_按评委问题推荐几条(self):
        from teams_voice_translator.defense.quick_phrases import build_suggest_messages, parse_suggestions
        from teams_voice_translator.defense.ui import QuickPhrasePanel

        engine = MagicMock()
        engine.answer_ready.return_value = False
        panel = QuickPhrasePanel(self.settings, engine)
        messages = build_suggest_messages("How exactly did you implement the pruning?", panel.phrases)
        self.assertIn("h1", messages[1]["content"])
        self.assertEqual(parse_suggestions('["h1", "t3", "zz", "h1", "c1"]', panel.phrases), ["h1", "t3", "c1"])
        asked = []
        panel.suggest_chat_factory = lambda: (lambda m: asked.append(m) or '["h1", "t1"]')
        panel.suggest_for("How exactly did you implement the pruning?")
        self.assertTrue(_wait(lambda: len(panel._suggest_buttons) == 2))
        self.assertFalse(panel.suggest_box.isHidden())
        panel.suggest_check.setChecked(False)
        panel.suggest_for("Another question?")
        self.assertEqual(len(asked), 1, "关掉推荐就不再调用模型")


# ------------------------------------------------------------------ 用量 / 余额
class UsageTests(_Base):
    def setUp(self):
        super().setUp()
        from teams_voice_translator.defense.usage import USAGE

        USAGE.reset()
        self.addCleanup(USAGE.reset)

    def test_阿里云签名和官方示例一致(self):
        from teams_voice_translator.defense.usage import sign_rpc

        params = {
            "AccessKeyId": "testid", "Action": "DescribeRegions", "Format": "XML",
            "SignatureMethod": "HMAC-SHA1", "SignatureNonce": "3ee8c1b8-83d3-44af-a94f-4e0ad82fd6cf",
            "SignatureVersion": "1.0", "Timestamp": "2016-02-23T12:46:24Z", "Version": "2014-05-26",
        }
        self.assertEqual(sign_rpc(params, "testsecret"), "OLeaidS1JvxuMvnyHOwuJ+uX5qY=")

    def test_查余额和账单_错误提示权限(self):
        from teams_voice_translator.defense.usage import query_balance, query_month_bill

        class Resp:
            def __init__(self, data, ok=True):
                self._data, self.ok, self.status_code = data, ok, 200 if ok else 400

            def json(self):
                return self._data

        calls = []

        def get(url, params, timeout, proxies):
            calls.append(params)
            if params["Action"] == "QueryAccountBalance":
                return Resp({"Code": "200", "Success": True, "Data": {"AvailableAmount": "1,234.56", "Currency": "CNY"}})
            return Resp({"Code": "Success", "Success": True, "Data": {"Items": {"Item": [
                {"ProductName": "大模型服务平台百炼", "PretaxAmount": 3.2, "Currency": "CNY"},
                {"ProductName": "大模型服务平台百炼", "PretaxAmount": 1.0, "Currency": "CNY"},
                {"ProductName": "对象存储 OSS", "PretaxAmount": 0.1, "Currency": "CNY"}]}}})

        self.assertEqual(query_balance("id", "sk", get=get), (1234.56, "CNY"))
        rows = query_month_bill("id", "sk", "2026-09", get=get)
        self.assertEqual(rows[0], {"product": "大模型服务平台百炼", "amount": 4.2, "currency": "CNY"})
        self.assertEqual(calls[1]["BillingCycle"], "2026-09")
        self.assertIn("Signature", calls[0])

        def forbidden(url, params, timeout, proxies):
            return Resp({"Code": "Forbidden.RAM", "Message": "User not authorized"}, ok=False)

        with self.assertRaises(RuntimeError) as ctx:
            query_balance("id", "sk", get=forbidden)
        self.assertIn("AliyunBSSReadOnlyAccess", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            query_balance("", "", get=get)

    def test_翻译流式返回的token被记下(self):
        from teams_voice_translator.defense.translator import ContextTranslator
        from teams_voice_translator.defense.usage import USAGE
        from tests.test_defense_translator import FakeStreamResponse

        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}),
                 "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 7}}),
                 "data: [DONE]"]
        translator = ContextTranslator(api_key="k", workspace_id="w", model="qwen3.8-flash")
        with mock.patch("teams_voice_translator.defense.translator.requests.post",
                        return_value=FakeStreamResponse(lines)) as poster:
            self.assertEqual(translator.complete({"messages": [{"role": "user", "content": "hi"}]}), "Hello")
        self.assertEqual(poster.call_args.kwargs["json"]["stream_options"], {"include_usage": True})
        snap = USAGE.snapshot()
        self.assertEqual((snap["input_tokens"], snap["output_tokens"]), (120, 7))

    def test_不认stream_options时自动去掉重试(self):
        from teams_voice_translator.defense import translator as module
        from teams_voice_translator.defense.translator import ContextTranslator
        from tests.test_defense_translator import FakeStreamResponse

        bad = FakeStreamResponse([], status_code=400,
                                 payload=json.dumps({"message": "unknown parameter stream_options"}))
        good = FakeStreamResponse(["data: " + json.dumps({"choices": [{"delta": {"content": "OK"},
                                                                       "finish_reason": "stop"}]}),
                                   "data: [DONE]"])
        translator = ContextTranslator(api_key="k", workspace_id="w", model="m")
        self.addCleanup(lambda: module._USAGE_OPTION.update(enabled=True))
        with mock.patch("teams_voice_translator.defense.translator._extract_api_error",
                        return_value="400 unknown parameter stream_options"), \
                mock.patch("teams_voice_translator.defense.translator.requests.post",
                           side_effect=[bad, good]) as poster:
            self.assertEqual(translator.complete({"messages": []}), "OK")
        self.assertNotIn("stream_options", poster.call_args_list[1].kwargs["json"])

    def test_合成和识别计数_按场次存档(self):
        from teams_voice_translator.defense.usage import USAGE, load_history, save_session

        USAGE.add_tts("qwen-audio-3.0-tts-plus", 120)
        USAGE.add_asr("qwen-audio-3.1-asr-flash-streaming", 90.0)
        USAGE.add_llm("qwen3.8-flash", 1000, 200)
        USAGE.set_balance(100.0)
        USAGE.set_balance(99.4)
        record = save_session(self.base)
        self.assertAlmostEqual(record["spent"], 0.6, places=4)
        history = load_history(self.base)
        self.assertEqual(history[0]["tokens"], 1200)
        self.assertEqual(history[0]["tts_chars"], 120)

    def test_顶栏显示余额和本次用量_消费记录窗口(self):
        from teams_voice_translator.defense.ui import UsageBar
        from teams_voice_translator.defense.usage import USAGE

        bar = UsageBar(self.settings)
        self.addCleanup(bar.deleteLater)
        USAGE.add_llm("qwen3.8-flash", 15000, 3200)
        bar.balance_query = lambda: (88.5, "CNY")
        bar.query_balance()
        self.assertTrue(_wait(lambda: "88.50" in bar.balance_label.text()))
        bar.refresh()
        self.assertIn("18.2k tokens", bar.session_label.text())
        bar.open_history()
        dialog = bar._dialog
        self.addCleanup(dialog.hide)
        self.assertIn("余额 ¥88.50", dialog.session_label.text())
        self.assertEqual(dialog.session_table.rowCount(), 1)
        dialog.bill_query = lambda cycle: [{"product": "百炼", "amount": 12.3, "currency": "CNY"}]
        dialog.query_bill()
        self.assertTrue(_wait(lambda: dialog.bill_table.rowCount() == 1))
        self.assertIn("12.30", dialog.bill_status.text())


if __name__ == "__main__":
    unittest.main()


class AmbiencePreviewTests(_Base):
    def test_试听有反馈_能停止_叠一句人声_场景事件前几秒就有(self):
        from teams_voice_translator.defense.ambience import make_room_tone, RecordedRoomTone
        from teams_voice_translator.defense.tts_cache import TtsDiskCache
        from teams_voice_translator.defense.ui import SettingsDialog

        tone = make_room_tone("office", volume=50)
        self.assertIsInstance(tone, RecordedRoomTone)
        self.assertTrue(any(array("h", tone.demo(24000 * 3))))
        TtsDiskCache(self.base).put("a" * 64, array("h", [6000, -6000] * 24000).tobytes(), pinned=True)
        dialog = SettingsDialog(None, self.settings)
        dialog.ambience_combo.setCurrentIndex(dialog.ambience_combo.findData("office"))
        pcm, with_voice = dialog._build_ambience_preview("office", 50)
        self.assertTrue(with_voice)
        written = []
        gate = threading.Event()

        class SlowPlayer:
            def write(self, data):
                written.append(len(data))
                gate.wait(0.01)

            def close(self):
                pass

        dialog.ambience_player_factory = SlowPlayer
        dialog._preview_ambience()
        self.assertEqual(dialog.ambience_preview_button.text(), "■ 停止试听")
        self.assertIn("你渲染过的一句话", dialog.ambience_preview_label.text())
        self.assertTrue(_wait(lambda: len(written) > 3))
        dialog._preview_ambience()          # 再点一次 = 停止
        self.assertTrue(_wait(lambda: dialog.ambience_preview_button.text().startswith("▶")))
        self.assertLess(sum(written), len(pcm))

    def test_放不出来时显示原因(self):
        from teams_voice_translator.defense.ui import SettingsDialog

        dialog = SettingsDialog(None, self.settings)
        dialog.ambience_combo.setCurrentIndex(dialog.ambience_combo.findData("room"))

        def broken():
            raise RuntimeError("Invalid device")

        dialog.ambience_player_factory = broken
        dialog._preview_ambience()
        self.assertTrue(_wait(lambda: "没放出来" in dialog.ambience_preview_label.text()))
        self.assertIn("Invalid device", dialog.ambience_preview_label.text())
