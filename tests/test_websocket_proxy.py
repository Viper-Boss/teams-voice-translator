import unittest
from unittest.mock import Mock, patch

from teams_voice_translator.aliyun import (
    FunASRRealtime,
    QwenRealtimeASR,
    _websocket_proxy_options,
    create_realtime_asr,
)
from teams_voice_translator.live_translate import QwenLiveTranslate


class WebSocketProxyTests(unittest.TestCase):
    def test_proxy_options_empty_when_no_proxy(self):
        self.assertEqual(_websocket_proxy_options(""), {})
        self.assertEqual(_websocket_proxy_options("   "), {})

    def test_proxy_options_parse_host_port(self):
        options = _websocket_proxy_options("127.0.0.1:7890")
        self.assertEqual(options["http_proxy_host"], "127.0.0.1")
        self.assertEqual(options["http_proxy_port"], 7890)
        self.assertEqual(options["proxy_type"], "http")
        self.assertNotIn("http_proxy_auth", options)

    def test_proxy_options_parse_url_with_auth(self):
        options = _websocket_proxy_options("http://user:pass@proxy.example.com:8080")
        self.assertEqual(options["http_proxy_host"], "proxy.example.com")
        self.assertEqual(options["http_proxy_port"], 8080)
        self.assertEqual(options["proxy_type"], "http")
        self.assertEqual(options["http_proxy_auth"], ("user", "pass"))

    def _assert_run_forever_with_proxy(self, client, mock_app):
        run_calls = [
            call for call in dir(mock_app.return_value) if call == "run_forever"
        ]
        # The thread starts run_forever; inspect the last call on the instance.
        run_forever = mock_app.return_value.run_forever
        # Wait briefly for the thread to call run_forever.
        import time
        for _ in range(50):
            if run_forever.called:
                break
            time.sleep(0.01)
        self.assertTrue(
            run_forever.called,
            f"{client.__class__.__name__} did not call run_forever",
        )
        call_kwargs = run_forever.call_args.kwargs
        self.assertEqual(call_kwargs["http_proxy_host"], "127.0.0.1")
        self.assertEqual(call_kwargs["http_proxy_port"], 7890)

    def test_qwen_realtime_asr_passes_proxy_to_run_forever(self):
        asr = QwenRealtimeASR(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3-asr-flash-realtime",
            language="zh",
            vad_threshold=0.0,
            vad_silence_ms=500,
            on_preview=lambda _t, _e: None,
            on_status=lambda _s: None,
            on_error=lambda _e: None,
            proxy="127.0.0.1:7890",
        )
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp") as mock_app:
            asr.start()
            self._assert_run_forever_with_proxy(asr, mock_app)
        asr.close()

    def test_fun_asr_realtime_passes_proxy_to_run_forever(self):
        asr = FunASRRealtime(
            api_key="sk-test",
            workspace_id="llm-test",
            model="fun-asr-realtime",
            language="zh",
            vad_threshold=0.0,
            vad_silence_ms=500,
            on_preview=lambda _t, _e: None,
            on_status=lambda _s: None,
            on_error=lambda _e: None,
            proxy="127.0.0.1:7890",
        )
        with patch("teams_voice_translator.aliyun.websocket.WebSocketApp") as mock_app:
            asr.start()
            self._assert_run_forever_with_proxy(asr, mock_app)
        asr.close()

    def test_live_translate_passes_proxy_to_run_forever(self):
        session = QwenLiveTranslate(
            api_key="sk-test",
            workspace_id="llm-test",
            model="qwen3.5-livetranslate-flash-realtime",
            proxy="127.0.0.1:7890",
        )
        with patch("teams_voice_translator.live_translate.websocket.WebSocketApp") as mock_app:
            session.start()
            self._assert_run_forever_with_proxy(session, mock_app)
        session.close()

    def test_create_realtime_asr_factory_passes_proxy(self):
        values = {
            "asr_model": "qwen3-asr-flash-realtime",
            "vad_threshold": 0.0,
            "vad_silence_ms": 500,
            "translation_terms": "",
        }
        asr = create_realtime_asr(
            api_key="sk-test",
            workspace_id="llm-test",
            values=values,
            language="zh",
            on_preview=lambda _t, _e: None,
            on_status=lambda _s: None,
            on_error=lambda _e: None,
            proxy="127.0.0.1:7890",
        )
        self.assertEqual(asr.proxy, "127.0.0.1:7890")

        values["asr_model"] = "fun-asr-realtime"
        asr2 = create_realtime_asr(
            api_key="sk-test",
            workspace_id="llm-test",
            values=values,
            language="zh",
            on_preview=lambda _t, _e: None,
            on_status=lambda _s: None,
            on_error=lambda _e: None,
            proxy="127.0.0.1:7890",
        )
        self.assertEqual(asr2.proxy, "127.0.0.1:7890")


if __name__ == "__main__":
    unittest.main()
