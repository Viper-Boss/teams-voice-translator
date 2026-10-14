"""余额刷新生命周期和显示语义；不使用真实凭据或云端接口。"""
from __future__ import annotations

import threading
from unittest import mock

from test_batch17 import _Base, _wait


class BalanceRefreshTests(_Base):
    def setUp(self):
        super().setUp()
        from teams_voice_translator.defense.ui import UsageBar
        from teams_voice_translator.defense.usage import USAGE

        USAGE.reset()
        self.addCleanup(USAGE.reset)
        self.credentials = mock.patch.object(UsageBar, "_credentials", return_value=("", ""))
        self.credentials.start()
        self.addCleanup(self.credentials.stop)
        self.bar = UsageBar(self.settings)
        self.bar._balance_timer.stop()
        self.addCleanup(self.bar.deleteLater)

    def dialog(self):
        self.bar.open_history()
        result = self.bar._dialog
        self.addCleanup(result.close)
        return result

    def test_success_updates_dialog_and_timestamp_without_reopen(self):
        dialog = self.dialog()
        self.bar.balance_query = lambda: (29.62, "CNY")
        dialog.refresh_button.click()
        self.assertTrue(_wait(lambda: not self.bar._querying))
        self.assertIn("29.62", dialog.session_label.text())
        self.assertIn("余额查询已完成", dialog.key_hint.text())
        self.assertNotIn("尚无成功查询", dialog.key_hint.text())
        self.assertNotIn("正在查询", dialog.key_hint.text())
        self.assertTrue(dialog.refresh_button.isEnabled())
        self.assertEqual(self.bar.BALANCE_EVERY_MS, 60000)

    def test_failure_survives_periodic_refresh_preserving_last_balance(self):
        from teams_voice_translator.defense.usage import USAGE

        USAGE.set_balance(29.62)
        dialog = self.dialog()
        self.bar.balance_query = mock.Mock(side_effect=RuntimeError("Forbidden.RAM"))
        self.bar.query_balance()
        self.assertTrue(_wait(lambda: not self.bar._querying))
        self.bar.refresh()
        self.assertIn("刷新失败", self.bar.balance_label.text())
        self.assertIn("29.62", self.bar.balance_label.text())
        self.assertIn("Forbidden.RAM", dialog.key_hint.text())
        self.assertTrue(dialog.save_key_button.isEnabled())

    def test_timeout_recovers_and_ignores_old_response_after_retry(self):
        from teams_voice_translator.defense.usage import USAGE

        gate, old_finished = threading.Event(), threading.Event()
        self.addCleanup(gate.set)

        def blocked():
            gate.wait(5)
            old_finished.set()
            return 999, "CNY"

        self.bar.balance_query = blocked
        self.bar.query_balance()
        old_id = self.bar._query_id
        self.bar._balance_timeout()
        self.assertFalse(self.bar._querying)
        self.assertIn("超时", self.bar.last_error)
        self.assertTrue(self.bar.refresh_button.isEnabled())
        self.bar.balance_query = lambda: (29.0, "CNY")
        self.bar.query_balance()
        self.assertTrue(_wait(lambda: not self.bar._querying))
        gate.set()
        self.assertTrue(_wait(old_finished.is_set))
        self.bar._on_balance(old_id, (999, "CNY"), "")
        self.assertEqual(USAGE.snapshot()["balance_now"], 29.0)
        self.assertEqual(self.bar.last_error, "")

    def test_timeout_timer_actually_fires(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.bar.QUERY_TIMEOUT_MS = 30
        self.bar.balance_query = lambda: (gate.wait(5) or 10, "CNY")
        self.bar.query_balance()
        self.assertTrue(_wait(lambda: not self.bar._querying))
        self.assertIn("超时", self.bar.last_error)

    def test_busy_query_does_not_duplicate_requests(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.bar.balance_query = mock.Mock(side_effect=lambda: (gate.wait(5) or 10, "CNY"))
        self.bar.query_balance()
        self.bar.query_balance()
        self.assertTrue(_wait(lambda: self.bar.balance_query.call_count == 1))
        gate.set()
        self.assertTrue(_wait(lambda: not self.bar._querying))

    def test_missing_credentials_finishes_with_reason(self):
        dialog = self.dialog()
        dialog._save_keys()
        self.assertFalse(self.bar._querying)
        self.assertIn("AccessKey", dialog.key_hint.text())
        self.assertIn("失败", dialog.key_hint.text())
        self.assertNotIn("正在查", dialog.key_hint.text())

    def test_zero_change_is_not_reported_as_zero_expense(self):
        from teams_voice_translator.defense.usage import USAGE

        USAGE.add_tts("qwen-audio-3.0-tts-plus", 118000)
        USAGE.set_balance(29.62)
        self.bar.refresh()
        self.assertNotIn("本次约", self.bar.balance_label.text())
        self.assertIn("暂无变化", self.bar.balance_label.text())
        dialog = self.dialog()
        self.assertIn("118.0k", dialog.session_label.text())
        self.assertIn("不代表没有费用", dialog.key_hint.text())

    def test_balance_drop_and_topup_display_direction(self):
        from teams_voice_translator.defense.usage import USAGE

        USAGE.set_balance(30)
        USAGE.set_balance(29)
        self.bar.refresh()
        self.assertIn("余额减少 ¥1.00", self.bar.balance_label.text())
        USAGE.set_balance(40)
        self.bar.refresh()
        self.assertIn("余额增加 ¥10.00", self.bar.balance_label.text())

    def test_invalid_balance_does_not_overwrite_good_value(self):
        from teams_voice_translator.defense.usage import USAGE

        USAGE.set_balance(29.62)
        self.bar.balance_query = lambda: (float("nan"), "CNY")
        self.bar.query_balance()
        self.assertTrue(_wait(lambda: not self.bar._querying))
        self.assertIn("格式无效", self.bar.last_error)
        self.assertEqual(USAGE.snapshot()["balance_now"], 29.62)

    def test_live_usage_refresh_preserves_table_selection_if_unchanged(self):
        from teams_voice_translator.defense.usage import USAGE

        USAGE.add_tts("m", 10)
        dialog = self.dialog()
        first = dialog.session_table.item(0, 0)
        self.bar.refresh()
        self.assertIs(first, dialog.session_table.item(0, 0))
        USAGE.add_tts("m", 20)
        self.bar.refresh()
        self.assertEqual(dialog.session_table.item(0, 3).text(), "30 字")

    def test_malformed_api_response_is_error_not_zero_balance(self):
        from teams_voice_translator.defense.usage import query_balance

        for payload in ({}, {"Data": {}}, {"Data": {"AvailableAmount": "NaN"}},
                        {"Data": {"AvailableAmount": "bad"}}):
            response = mock.Mock(ok=True)
            response.json.return_value = payload
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                query_balance("id", "secret", get=mock.Mock(return_value=response))

    def test_first_success_time_is_preserved_and_currency_change_rebaselines(self):
        from teams_voice_translator.defense.usage import UsageTracker

        tracker = UsageTracker()
        with mock.patch("teams_voice_translator.defense.usage.time.time", return_value=100):
            tracker.set_balance(30)
        with mock.patch("teams_voice_translator.defense.usage.time.time", return_value=200):
            tracker.set_balance(29)
        self.assertEqual(tracker.snapshot()["balance_started_at"], 100)
        self.assertEqual(tracker.snapshot()["balance_updated_at"], 200)
        tracker.set_balance(10, "USD")
        self.assertEqual(tracker.snapshot()["spent"], 0)
