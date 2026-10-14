"""千问同传接入：配置报文、事件拼段、引擎切换。全部离线，不连服务器。"""
from __future__ import annotations

import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from teams_voice_translator.defense.livetranslate import (
    LiveTranslateRealtime,
    LiveTranslateState,
    build_session_update,
    glossary_phrases,
)


class SessionUpdateTests(unittest.TestCase):
    def test_38_uses_output_modalities_text_only(self):
        session = build_session_update("qwen3.8-livetranslate-flash-realtime",
                                       phrases={"漏磁": "magnetic flux leakage"})["session"]
        self.assertEqual(session["output_modalities"], ["text"])
        self.assertNotIn("modalities", session)
        self.assertEqual(session["translation"]["language"], "en")
        self.assertEqual(session["translation"]["corpus"]["phrases"], {"漏磁": "magnetic flux leakage"})
        self.assertEqual(session["input_audio_transcription"]["language"], "zh")

    def test_35_uses_modalities(self):
        session = build_session_update("qwen3.5-livetranslate-flash-realtime")["session"]
        self.assertEqual(session["modalities"], ["text"])
        self.assertNotIn("corpus", session["translation"], "没有术语就不发 corpus")

    def test_minimal_drops_optional_fields(self):
        session = build_session_update("qwen3.8-livetranslate-flash-realtime",
                                       phrases={"a": "b"}, minimal=True)["session"]
        self.assertEqual(set(session), {"output_modalities", "translation"})
        self.assertEqual(session["translation"], {"language": "en"})

    def test_glossary_to_phrases_skips_blank(self):
        phrases = glossary_phrases([{"source": "剪枝", "target": "pruning"}, {"source": "", "target": "x"}])
        self.assertEqual(phrases, {"剪枝": "pruning"})


class StateTests(unittest.TestCase):
    def test_38_events_make_one_segment(self):
        state = LiveTranslateState()
        self.assertEqual(state.on_event({"type": "session.updated"})[0], "ready")
        state.on_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "我用了"})
        action, preview = state.on_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "小波去噪"})
        self.assertEqual((action, preview), ("preview", "我用了小波去噪"))
        state.on_event({"type": "response.text.delta", "delta": "I used "})
        self.assertEqual(state.on_event({"type": "response.text.delta", "delta": "wavelet denoising."}),
                         ("translation_delta", "I used wavelet denoising."))
        action, data = state.on_event({"type": "response.text.done", "text": "I used wavelet denoising."})
        self.assertEqual(action, "segment")
        self.assertEqual(data, ("我用了小波去噪", "I used wavelet denoising."))
        # 下一段从空开始
        self.assertEqual(state.source_preview(), "")
        self.assertEqual(state.translation, "")

    def test_completed_transcript_replaces_partial(self):
        state = LiveTranslateState()
        state.on_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "草稿"})
        state.on_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "定稿。"})
        self.assertEqual(state.source_preview(), "定稿。")

    def test_35_text_and_stash(self):
        state = LiveTranslateState()
        self.assertEqual(state.on_event({"type": "response.text.text", "text": "Hello", "stash": " wor"}),
                         ("translation_delta", "Hello wor"))
        self.assertEqual(state.on_event({"type": "response.audio_transcript.done", "transcript": "Hello world."}),
                         ("segment", ("", "Hello world.")))

    def test_response_done_without_text_done_still_closes(self):
        state = LiveTranslateState()
        state.on_event({"type": "response.text.delta", "delta": "Partial answer"})
        self.assertEqual(state.on_event({"type": "response.done"}), ("segment", ("", "Partial answer")))
        self.assertEqual(state.on_event({"type": "response.done"}), ("", None))

    def test_empty_translation_is_not_a_segment(self):
        state = LiveTranslateState()
        self.assertEqual(state.on_event({"type": "response.text.done", "text": "  "}), ("", None))

    def test_error_message(self):
        action, text = LiveTranslateState().on_event({"type": "error", "error": {"message": "bad param"}})
        self.assertEqual((action, text), ("error", "bad param"))


class ClientTests(unittest.TestCase):
    def make(self, **kw):
        self.segments, self.errors, self.statuses = [], [], []
        return LiveTranslateRealtime(
            api_key="sk-test", workspace_id="ws-demo",
            on_preview=lambda text: None,
            on_translation=lambda s, e: self.segments.append((s, e)),
            on_status=self.statuses.append, on_error=self.errors.append, **kw)

    def test_url_and_first_session_update(self):
        client = self.make(phrases={"剪枝": "pruning"})
        self.assertIn("model=qwen3.8-livetranslate-flash-realtime", client.url())
        ws = MagicMock()
        client._on_open(ws)
        import json
        sent = json.loads(ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "session.update")
        self.assertIn("corpus", sent["session"]["translation"])

    def test_config_rejected_retries_minimal_once(self):
        import json
        client = self.make(phrases={"剪枝": "pruning"})
        ws = MagicMock()
        client._on_message(ws, json.dumps({"type": "error", "error": {"message": "unknown field corpus"}}))
        retry = json.loads(ws.send.call_args[0][0])
        self.assertNotIn("corpus", retry["session"]["translation"])
        self.assertEqual(self.errors, [], "第一次被拒不算失败，先重试")
        client._on_message(ws, json.dumps({"type": "error", "error": {"message": "still bad"}}))
        self.assertEqual(self.errors, ["still bad"])
        self.assertTrue(client.ready.is_set())
        from teams_voice_translator.aliyun import ApiError
        with self.assertRaises(ApiError):
            client.wait_ready(timeout=0.1)

    def test_segment_reaches_callback(self):
        import json
        client = self.make()
        ws = MagicMock()
        client._on_message(ws, json.dumps({"type": "session.updated"}))
        client._on_message(ws, json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "谢谢老师。"}))
        client._on_message(ws, json.dumps({"type": "response.text.done", "text": "Thank you, professor."}))
        self.assertEqual(self.segments, [("谢谢老师。", "Thank you, professor.")])

    def test_requires_credentials(self):
        from teams_voice_translator.aliyun import ApiError
        with self.assertRaises(ApiError):
            LiveTranslateRealtime(api_key="", workspace_id="ws", on_preview=print,
                                  on_translation=print, on_status=print, on_error=print)


class EngineTests(unittest.TestCase):
    def setUp(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        from teams_voice_translator.defense.settings import DefenseSettings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(base_dir=Path(self.tmp.name))
        self.segments, self.done = [], []
        self.engine = DefenseEngine(self.settings, EngineCallbacks(
            on_my_segment=lambda seg, text: self.segments.append(text),
            on_my_done=lambda seg, text, latency: self.done.append(text)))
        self.addCleanup(self.engine.shutdown)

    def test_live_translation_skips_translator_and_speaks(self):
        self.engine.translator = MagicMock()
        self.engine.tts = MagicMock()
        self.engine.tts.speak_batch.return_value = True
        self.engine._on_my_live_translation("我用了小波。", "I used wavelets.")
        job = self.engine._my_queue.get_nowait()
        self.assertEqual(job.preset_english, "I used wavelets.")
        self.engine._translate_my_job(job)
        self.engine.translator.translate.assert_not_called()
        self.engine.tts.speak_batch.assert_called_once()
        self.assertEqual(self.segments, ["我用了小波。"])
        self.assertEqual(self.done, ["I used wavelets."])
        self.assertIn("I used wavelets.", self.engine.memory.render_history())

    def test_empty_live_translation_ignored(self):
        self.engine._on_my_live_translation("嗯", "  ")
        with self.assertRaises(queue.Empty):
            self.engine._my_queue.get_nowait()

    def test_engine_setting_routes_to_livetranslate(self):
        self.settings.set("my_translate_engine", "livetranslate")
        self.engine._open_mic_for = MagicMock()
        with patch("teams_voice_translator.defense.livetranslate.LiveTranslateRealtime") as cls:
            self.engine._start_live_translate("sk-test", "ws-demo")
        kwargs = cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen3.8-livetranslate-flash-realtime")
        self.assertEqual((kwargs["source_language"], kwargs["target_language"]), ("zh", "en"))
        self.engine._open_mic_for.assert_called_once()

    def test_default_engine_is_llm(self):
        self.assertEqual(self.settings.get("my_translate_engine"), "llm")
        self.assertEqual(self.settings.get("llm_model"), "qwen3.8-flash")


class SettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_engine_combo_round_trip(self):
        from teams_voice_translator.defense.settings import DefenseSettings
        from teams_voice_translator.defense.ui import SettingsDialog

        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(base_dir=Path(tmp))
            dialog = SettingsDialog(None, settings)
            self.assertEqual(dialog._engine_values(), {"my_translate_engine": "llm"})
            index = dialog.engine_combo.findData("livetranslate:qwen3.8-livetranslate-flash-realtime")
            self.assertGreaterEqual(index, 0)
            dialog.engine_combo.setCurrentIndex(index)
            self.assertEqual(dialog._engine_values(), {
                "my_translate_engine": "livetranslate",
                "livetranslate_model": "qwen3.8-livetranslate-flash-realtime",
            })
            settings.update(dialog._engine_values())
            reopened = SettingsDialog(None, settings)
            self.assertEqual(reopened.engine_combo.currentData(),
                             "livetranslate:qwen3.8-livetranslate-flash-realtime")


if __name__ == "__main__":
    unittest.main()


class PromptTests(unittest.TestCase):
    def test_translation_prompts_ask_to_fix_recognition_errors(self):
        from teams_voice_translator.defense.memory import MeetingMemory

        memory = MeetingMemory()
        self.assertIn("语音识别", memory.system_prompt("zh2en"))
        self.assertIn("口头语", memory.system_prompt("zh2en"))
        self.assertIn("语音识别", memory.system_prompt("en2zh"))
        self.assertTrue(memory.system_prompt("zh2en").rstrip().endswith("不要引号、标注或任何解释。"))
