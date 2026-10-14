"""批量复刻：一条录音 → 多个音色。

核心不变量：
  1. 除了要对比的那一项，所有任务的条件逐字相同（这是做受控 A/B 的前提）；
  2. 任何被跳过的组合都必须有明确理由，不能静默吞掉；
  3. 同一 (时长, 通道) 只转码一次、只上传一次 OSS；
  4. 单个任务失败不中断整批。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from teams_voice_translator.defense.clone_batch import (
    MAX_CLONE_SECONDS,
    CloneTask,
    build_tasks,
    clone_channel_for_model,
    describe_plan,
    group_tasks,
    minimum_sample_seconds,
    needs_oss,
)
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.ui import CloneVoiceDialog

QWEN_AUDIO = "qwen-audio-3.0-tts-plus"
COSY = "cosyvoice-v3.5-plus"
COSY_OLD = "cosyvoice-v3-plus"          # 不支持预处理 → 下限 5 秒
QWEN3_VC = "qwen3-tts-vc-2026-01-22"    # Base64 直传，无需 OSS


class ChannelTests(unittest.TestCase):
    def test_only_qwen3_vc_uses_direct_upload(self):
        self.assertEqual(clone_channel_for_model(QWEN3_VC), "qwen3")
        self.assertEqual(clone_channel_for_model("qwen3-tts-vc-realtime-2026-01-15"), "qwen3")
        self.assertEqual(clone_channel_for_model(QWEN_AUDIO), "legacy")
        self.assertEqual(clone_channel_for_model(COSY), "legacy")

    def test_preprocess_models_accept_shorter_samples(self):
        self.assertEqual(minimum_sample_seconds(QWEN_AUDIO), 3.0)
        self.assertEqual(minimum_sample_seconds(COSY_OLD), 5.0)


class BuildTaskTests(unittest.TestCase):
    def test_cross_product_of_models_and_durations(self):
        tasks, skipped = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=26.7)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(skipped, [])
        self.assertEqual(
            {(t.model, t.seconds) for t in tasks},
            {(QWEN_AUDIO, 18.0), (QWEN_AUDIO, 26.0), (COSY, 18.0), (COSY, 26.0)},
        )

    def test_durations_are_clamped_to_the_actual_sample(self):
        """样音只有 12 秒时，「18 秒」和「全长」都只能是 12 秒。"""
        tasks, _ = build_tasks([QWEN_AUDIO], [18, MAX_CLONE_SECONDS], sample_seconds=12.0)
        self.assertEqual([t.seconds for t in tasks], [12.0])

    def test_collapsed_durations_are_deduplicated(self):
        """夹完之后重复的组合不该真的提交两次，也不该产生两个几乎一样的音色。"""
        tasks, _ = build_tasks([QWEN_AUDIO], [18, 18.0, MAX_CLONE_SECONDS], sample_seconds=18.0)
        self.assertEqual(len(tasks), 1)

    def test_service_cap_is_enforced_even_for_long_samples(self):
        tasks, _ = build_tasks([QWEN_AUDIO], [60], sample_seconds=120.0)
        self.assertEqual(tasks[0].seconds, MAX_CLONE_SECONDS)

    def test_duration_below_model_floor_is_skipped_with_a_reason(self):
        tasks, skipped = build_tasks([COSY_OLD], [4], sample_seconds=30.0)
        self.assertEqual(tasks, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("5", skipped[0].reason)

    def test_same_duration_can_suit_one_model_and_not_another(self):
        tasks, skipped = build_tasks([QWEN_AUDIO, COSY_OLD], [4], sample_seconds=30.0)
        self.assertEqual([t.model for t in tasks], [QWEN_AUDIO])
        self.assertEqual([s.model for s in skipped], [COSY_OLD])

    def test_model_without_clone_support_is_skipped(self):
        tasks, skipped = build_tasks(["made-up-model"], [18], sample_seconds=30.0)
        self.assertEqual(tasks, [])
        self.assertIn("不支持声音复刻", skipped[0].reason)

    def test_preprocess_is_gated_per_model_not_per_dialog(self):
        """批量里混入不支持预处理的模型，不能让整批带上这个字段而报错。"""
        tasks, _ = build_tasks([QWEN_AUDIO, COSY_OLD], [18], sample_seconds=30.0, preprocess=True)
        by_model = {t.model: t for t in tasks}
        self.assertTrue(by_model[QWEN_AUDIO].preprocess)
        self.assertFalse(by_model[COSY_OLD].preprocess)

    def test_volume_normalization_only_applies_to_legacy_channel(self):
        tasks, _ = build_tasks(
            [COSY, QWEN3_VC], [18], sample_seconds=30.0, volume_normalization=True
        )
        by_model = {t.model: t for t in tasks}
        self.assertTrue(by_model[COSY].volume_normalization)
        self.assertFalse(by_model[QWEN3_VC].volume_normalization)

    def test_language_is_carried_to_every_task(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18], sample_seconds=30.0, language="zh")
        self.assertTrue(all(t.language == "zh" for t in tasks))

    def test_blank_model_names_are_ignored(self):
        tasks, skipped = build_tasks([QWEN_AUDIO, "", "   "], [18], sample_seconds=30.0)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(skipped, [])

    def test_zero_length_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            build_tasks([QWEN_AUDIO], [18], sample_seconds=0)


class GroupingTests(unittest.TestCase):
    def test_same_duration_and_channel_share_one_upload(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=30.0)
        groups = group_tasks(tasks)
        # 4 个任务，2 个时长 × 1 个通道 = 2 组 → 只转码/上传 2 次而不是 4 次
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(g[2]) == 2 for g in groups))

    def test_channels_never_share_a_group(self):
        tasks, _ = build_tasks([QWEN_AUDIO, QWEN3_VC], [18], sample_seconds=30.0)
        groups = group_tasks(tasks)
        self.assertEqual(len(groups), 2)
        self.assertEqual({g[1] for g in groups}, {"legacy", "qwen3"})

    def test_group_order_follows_first_appearance(self):
        tasks = [
            CloneTask(model=QWEN_AUDIO, seconds=26.0),
            CloneTask(model=COSY, seconds=18.0),
            CloneTask(model=COSY, seconds=26.0),
        ]
        self.assertEqual([g[0] for g in group_tasks(tasks)], [26.0, 18.0])

    def test_empty_input_groups_to_nothing(self):
        self.assertEqual(group_tasks([]), [])


class PlanSummaryTests(unittest.TestCase):
    def test_oss_is_required_only_when_a_legacy_task_exists(self):
        legacy, _ = build_tasks([QWEN_AUDIO], [18], sample_seconds=30.0)
        direct, _ = build_tasks([QWEN3_VC], [18], sample_seconds=30.0)
        self.assertTrue(needs_oss(legacy))
        self.assertFalse(needs_oss(direct))
        self.assertFalse(needs_oss([]))

    def test_summary_counts_voices_and_uploads(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=30.0)
        text = describe_plan(tasks)
        self.assertIn("4 个音色", text)
        self.assertIn("2 次 OSS", text)

    def test_summary_says_when_oss_is_not_needed(self):
        tasks, _ = build_tasks([QWEN3_VC], [18], sample_seconds=30.0)
        self.assertIn("无需 OSS", describe_plan(tasks))

    def test_summary_reports_skipped_combinations(self):
        tasks, skipped = build_tasks([QWEN_AUDIO, COSY_OLD], [4], sample_seconds=30.0)
        self.assertIn("1 个组合被跳过", describe_plan(tasks, skipped))

    def test_empty_plan_is_stated_plainly(self):
        self.assertIn("不会创建任何音色", describe_plan([]))


class BatchPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        for name in ("_maybe_auto_probe_oss", "_maybe_auto_probe_oss_for_batch"):
            probe = patch.object(CloneVoiceDialog, name)
            probe.start()
            self.addCleanup(probe.stop)

    def dialog(self, model=QWEN_AUDIO):
        dialog = CloneVoiceDialog(None, self.settings, model)
        self.addCleanup(dialog.close)
        return dialog

    def test_batch_is_off_by_default_and_hides_its_button(self):
        dialog = self.dialog()
        self.assertFalse(dialog.batch_group.isChecked())
        self.assertFalse(dialog.batch_button.isVisible())

    def test_current_model_starts_checked(self):
        dialog = self.dialog()
        self.assertEqual(dialog._selected_batch_models(), [QWEN_AUDIO])

    def test_only_clone_capable_models_are_offered(self):
        dialog = self.dialog()
        listed = [
            str(dialog.batch_model_list.item(row).data(Qt.UserRole))
            for row in range(dialog.batch_model_list.count())
        ]
        self.assertIn(QWEN_AUDIO, listed)
        self.assertEqual(len(listed), len(set(listed)))

    def test_recommended_button_picks_one_model_per_family(self):
        dialog = self.dialog()
        dialog._check_recommended_models()
        self.assertEqual(sorted(dialog._selected_batch_models()), sorted([COSY, QWEN_AUDIO]))

    def test_clear_button_unchecks_everything(self):
        dialog = self.dialog()
        dialog._set_all_batch_models(False)
        self.assertEqual(dialog._selected_batch_models(), [])

    def test_plan_is_empty_until_a_sample_exists(self):
        dialog = self.dialog()
        dialog.batch_group.setChecked(True)
        self.assertFalse(dialog.batch_button.isEnabled())
        self.assertIn("先录音", dialog.batch_plan_label.text())

    def test_plan_lists_every_task_once_a_sample_is_ready(self):
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 26.7)
        dialog.batch_group.setChecked(True)
        dialog._check_recommended_models()
        tasks, _ = dialog._current_batch_tasks()
        # 2 模型 × (18s + 全长 26.7→26.7) = 4
        self.assertEqual(len(tasks), 4)
        self.assertTrue(dialog.batch_button.isEnabled())
        self.assertIn("4 个音色", dialog.batch_plan_label.text())

    def test_short_sample_collapses_both_durations_into_one_task(self):
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 15.0)
        dialog.batch_group.setChecked(True)
        tasks, _ = dialog._current_batch_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].seconds, 15.0)

    def test_all_direct_upload_selection_hides_the_oss_form(self):
        """全 qwen3 直传时不该拦着人去配 OSS。"""
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 26.7)
        dialog.batch_group.setChecked(True)
        dialog._set_all_batch_models(False)
        for row in range(dialog.batch_model_list.count()):
            item = dialog.batch_model_list.item(row)
            if str(item.data(Qt.UserRole)) == QWEN3_VC:
                item.setCheckState(Qt.Checked)
        self.assertFalse(dialog.oss_group.isVisible())
        self.assertIn("无需 OSS", dialog.channel_note.text())

    def test_custom_duration_follows_the_spin_box(self):
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 26.7)
        dialog.batch_group.setChecked(True)
        dialog.batch_duration_recommended.setChecked(False)
        dialog.batch_duration_full.setChecked(False)
        dialog.batch_duration_custom.setChecked(True)
        dialog.sample_length_spin.setValue(12.5)
        tasks, _ = dialog._current_batch_tasks()
        self.assertEqual([t.seconds for t in tasks], [12.5])

    def test_no_duration_selected_disables_the_button(self):
        dialog = self.dialog()
        dialog._set_sample(Path("test.wav"), 26.7)
        dialog.batch_group.setChecked(True)
        dialog.batch_duration_recommended.setChecked(False)
        dialog.batch_duration_full.setChecked(False)
        self.assertFalse(dialog.batch_button.isEnabled())

    def test_results_are_recorded_into_the_voice_library(self):
        dialog = self.dialog()
        dialog._on_batch_result(
            {"model": COSY, "seconds": 18.0, "voice_id": "voice-abc", "language": "en", "error": ""}
        )
        entries = dialog._library_entries()
        saved = next(e for e in entries if e.get("voice_id") == "voice-abc")
        self.assertEqual(saved["target_model"], COSY)
        self.assertEqual(saved["sample_seconds"], 18.0)
        self.assertIn("18s", saved["name"])

    def test_failed_result_shows_the_reason_and_saves_nothing(self):
        dialog = self.dialog()
        before = len(dialog._library_entries())
        dialog._on_batch_result({"model": COSY, "seconds": 18.0, "error": "配额不足"})
        self.assertEqual(len(dialog._library_entries()), before)
        self.assertIn("配额不足", dialog.batch_result_table.item(0, 2).text())

    def test_matching_model_result_fills_the_voice_id_box(self):
        dialog = self.dialog(QWEN_AUDIO)
        dialog._on_batch_result(
            {"model": COSY, "seconds": 18.0, "voice_id": "other", "error": ""}
        )
        self.assertEqual(dialog.created_voice_id, "")
        dialog._on_batch_result(
            {"model": QWEN_AUDIO, "seconds": 18.0, "voice_id": "mine", "error": ""}
        )
        self.assertEqual(dialog.created_voice_id, "mine")

    def test_partial_failure_is_reported_as_partial(self):
        dialog = self.dialog()
        dialog._on_batch_done(3, 4)
        self.assertIn("成功 3 个，失败 1 个", dialog.result_label.text())

    def test_total_failure_is_not_dressed_up_as_success(self):
        dialog = self.dialog()
        dialog._on_batch_done(0, 2)
        self.assertIn("没有创建任何音色", dialog.result_label.text())

    def test_closing_the_dialog_cancels_the_batch(self):
        """线程还会 emit 信号，而信号连到的是即将销毁的控件。"""
        dialog = CloneVoiceDialog(None, self.settings, QWEN_AUDIO)
        self.assertFalse(dialog._batch_cancel.is_set())
        dialog.close()
        self.assertTrue(dialog._batch_cancel.is_set())

    def test_cancel_button_sets_the_token_without_killing_the_thread(self):
        dialog = self.dialog()
        dialog._cancel_batch()
        self.assertTrue(dialog._batch_cancel.is_set())
        self.assertIn("停止", dialog.result_label.text())


class _FakeUploaded:
    def __init__(self, key):
        self.key = key
        self.signed_url = f"https://oss.example/{key}"


class _FakeUploader:
    """记录每次上传/删除，用来证明同组只传一次、且一定会清理。"""

    instances: list = []

    def __init__(self, **values):
        self.values = values
        self.uploads: list = []
        self.deletes: list = []
        _FakeUploader.instances.append(self)

    def upload(self, path):
        uploaded = _FakeUploaded(f"temporary/{len(self.uploads)}")
        self.uploads.append(path)
        return uploaded

    def delete(self, key):
        self.deletes.append(key)


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.clone_calls: list = []
        self.fail_models: set = set()

    def clone_voice(self, **kwargs):
        self.clone_calls.append(kwargs)
        if kwargs["target_model"] in self.fail_models:
            raise RuntimeError("配额不足")
        return f"voice-{kwargs['target_model']}-{kwargs['max_seconds']:g}"

    def wait_for_voice_ready(self, voice_id, **kwargs):
        return True

    def clone_qwen_voice(self, **kwargs):
        self.clone_calls.append(kwargs)
        return f"voice-{kwargs['target_model']}", ""


class BatchWorkerTests(unittest.TestCase):
    """直接驱动 worker，不起 Qt 事件循环：信号用假对象接住。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = DefenseSettings(Path(self.tmp.name))
        _FakeUploader.instances = []
        self.normalize_calls: list = []

        import contextlib

        @contextlib.contextmanager
        def fake_normalize(source, *, max_seconds, min_seconds):
            self.normalize_calls.append(max_seconds)
            wav = Path(self.tmp.name) / f"sample-{max_seconds:g}.wav"
            wav.write_bytes(b"RIFFfake")
            yield wav

        for target, replacement in (
            ("normalized_voice_sample", fake_normalize),
            ("inspect_voice_sample", lambda path: "检查通过"),
            ("OssTemporaryUploader", _FakeUploader),
        ):
            patcher = patch(f"teams_voice_translator.defense.ui.{target}", replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = _FakeClient()
        client_patch = patch(
            "teams_voice_translator.defense.ui.BailianClient", return_value=self.client
        )
        client_patch.start()
        self.addCleanup(client_patch.stop)

        probe = patch.object(CloneVoiceDialog, "_maybe_auto_probe_oss")
        probe.start()
        self.addCleanup(probe.stop)

    def run_batch(self, tasks, cancel=None):
        import threading

        dialog = CloneVoiceDialog(None, self.settings, QWEN_AUDIO)
        self.addCleanup(dialog.close)
        results: list = []
        done: list = []
        dialog.batch_result.disconnect()
        dialog.batch_done.disconnect()
        dialog.batch_result.connect(results.append, Qt.DirectConnection)
        dialog.batch_done.connect(lambda ok, total: done.append((ok, total)), Qt.DirectConnection)
        dialog._batch_clone_worker(
            Path("source.m4a"), "", tuple(tasks),
            {"region": "cn-beijing", "bucket": "b", "access_key_id": "k", "access_key_secret": "s"},
            cancel or threading.Event(),
        )
        return results, done

    def test_one_transcode_and_one_upload_per_duration(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=30.0)
        results, done = self.run_batch(tasks)
        self.assertEqual(done, [(4, 4)])
        # 4 个音色，但只转码 2 次、只上传 2 次 —— 这就是批量相对手点的实际收益。
        self.assertEqual(sorted(self.normalize_calls), [18.0, 26.0])
        uploads = sum(len(u.uploads) for u in _FakeUploader.instances)
        self.assertEqual(uploads, 2)
        self.assertEqual(len(self.client.clone_calls), 4)
        self.assertTrue(all(r["voice_id"] for r in results))

    def test_temporary_oss_object_is_always_deleted(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18], sample_seconds=30.0)
        self.run_batch(tasks)
        deletes = sum(len(u.deletes) for u in _FakeUploader.instances)
        self.assertEqual(deletes, 1)

    def test_one_failing_model_does_not_abort_the_batch(self):
        """用户等了几分钟，不该因为第三个模型报错就丢掉前两个成功的音色。"""
        self.client.fail_models = {COSY}
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18], sample_seconds=30.0)
        results, done = self.run_batch(tasks)
        self.assertEqual(done, [(1, 2)])
        failed = [r for r in results if r["error"]]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["model"], COSY)
        self.assertIn("配额不足", failed[0]["error"])
        # 失败之后临时对象照样清理
        self.assertEqual(sum(len(u.deletes) for u in _FakeUploader.instances), 1)

    def test_sample_preparation_failure_reports_every_task_in_that_group(self):
        """整组样音没准备好时，不能只报一条，否则表里会凭空少几行。"""
        import contextlib

        @contextlib.contextmanager
        def boom(source, *, max_seconds, min_seconds):
            raise RuntimeError("转码失败")
            yield  # pragma: no cover

        with patch("teams_voice_translator.defense.ui.normalized_voice_sample", boom):
            tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18], sample_seconds=30.0)
            results, done = self.run_batch(tasks)
        self.assertEqual(done, [(0, 2)])
        self.assertEqual(len(results), 2)
        self.assertTrue(all("转码失败" in r["error"] for r in results))

    def test_cancellation_stops_before_the_next_task(self):
        import threading

        cancel = threading.Event()
        cancel.set()
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=30.0)
        results, done = self.run_batch(tasks, cancel=cancel)
        self.assertEqual(results, [])
        self.assertEqual(done, [(0, 0)])
        self.assertEqual(self.client.clone_calls, [])

    def test_direct_upload_tasks_never_touch_oss(self):
        tasks, _ = build_tasks([QWEN3_VC], [18], sample_seconds=30.0)
        results, done = self.run_batch(tasks)
        self.assertEqual(done, [(1, 1)])
        self.assertEqual(_FakeUploader.instances, [])
        self.assertIn("audio_data", self.client.clone_calls[0])

    def test_each_task_sends_its_own_model_and_duration(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18, 26], sample_seconds=30.0)
        self.run_batch(tasks)
        sent = {(c["target_model"], c["max_seconds"]) for c in self.client.clone_calls}
        self.assertEqual(
            sent,
            {(QWEN_AUDIO, 18.0), (QWEN_AUDIO, 26.0), (COSY, 18.0), (COSY, 26.0)},
        )

    def test_language_reaches_every_clone_call(self):
        tasks, _ = build_tasks([QWEN_AUDIO, COSY], [18], sample_seconds=30.0, language="zh")
        self.run_batch(tasks)
        self.assertTrue(all(c["language"] == "zh" for c in self.client.clone_calls))


if __name__ == "__main__":
    unittest.main()
