"""Whole-script duration and playback navigation regressions; no cloud/device calls."""
import unittest
from unittest.mock import patch

import test_script_desk_ui as fixtures


class ScriptTimingTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ScriptDeskPanelTests.setUpClass.__func__)
    setUp = fixtures.ScriptDeskPanelTests.setUp
    panel = fixtures.ScriptDeskPanelTests.panel
    _engine_at = fixtures.ScriptDeskPanelTests._engine_at

    def test_sentence_audio_counts_only_real_gaps(self):
        panel = self.panel("One. Two.\nThree.", ready=True)
        self.assertAlmostEqual(panel._timeline_total, 1.5 + .35 + .8)
        panel.gap_combo.setCurrentIndex(panel.gap_combo.findData(0))
        panel.paragraph_gap_combo.setCurrentIndex(panel.paragraph_gap_combo.findData(300))
        self.assertAlmostEqual(panel._timeline_total, 1.8)
        self.assertIn("总时长 00:02", panel.timing_label.text())

    def test_group_audio_does_not_add_sentence_gaps_twice(self):
        panel = self.panel("One. Two.\nThree.", ready=True)
        pcm = b"\x00\x00" * 24000
        def audio(paragraph):
            return {line.index: ((paragraph.index, 0), pcm, False) for line in paragraph.lines}
        with patch.object(panel, "_paragraph_audio", side_effect=audio):
            panel.refresh_render_state()
            self.assertAlmostEqual(panel._timeline_total, 3.8)
            panel.paragraph_table.selectRow(0)
            panel._play_from_here()
            self.assertAlmostEqual(panel._play_total, 2.0)

    def test_missing_audio_never_claims_complete_total(self):
        panel = self.panel("One. Two.")
        self.assertFalse(panel._timeline_complete)
        self.assertIn("总时长待定", panel.timing_label.text())
        self.assertIn("剩余 --:--", panel.timing_label.text())

    def test_chain_progress_includes_paragraph_gap_and_finishes_at_zero(self):
        panel = self.panel("One.\nTwo.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, .5, active=False)
        panel._advance_cursor()
        self.assertAlmostEqual(panel._timeline_position, .5)
        self._engine_at(panel, .4)  # inside the second paragraph's leading gap
        panel._advance_cursor()
        self.assertAlmostEqual(panel._timeline_position, .9)
        self._engine_at(panel, 1.3, active=False)
        panel._advance_cursor()
        self.assertAlmostEqual(panel._timeline_position, panel._timeline_total)
        self.assertIn("剩余 00:00", panel.timing_label.text())

    def test_starting_in_middle_uses_full_script_position(self):
        panel = self.panel("One.\nTwo.\nThree.", ready=True)
        panel.paragraph_table.selectRow(1)
        panel._play_from_here()
        self.assertAlmostEqual(panel._timeline_position, 1.3)
        self._engine_at(panel, .2)
        panel._advance_cursor()
        self.assertAlmostEqual(panel._timeline_position, 1.5)
        self.assertAlmostEqual(panel._timeline_total - panel._timeline_position, 1.6)

    def test_pause_does_not_use_wall_clock_and_stop_retains_position(self):
        panel = self.panel("One. Two.", ready=True)
        panel._play_paragraph()
        self._engine_at(panel, .4)
        panel.pause()
        before = panel.timing_label.text()
        with patch("teams_voice_translator.defense.ui.time.monotonic", return_value=999999):
            panel._advance_cursor()
            self.assertEqual(panel.timing_label.text(), before)
        panel.resume()
        self._engine_at(panel, .9)
        panel._stop_playback()
        self.assertAlmostEqual(panel._timeline_position, .9)
        self.assertTrue(panel.gap_combo.isEnabled())

    def test_clear_and_replace_reset_position(self):
        panel = self.panel("One. Two.", ready=True)
        panel._play_paragraph()
        self._engine_at(panel, .9)
        panel._advance_cursor()
        panel.source_edit.setPlainText("Replacement.")
        panel._split_clicked()
        self.assertEqual(panel._timeline_position, 0)
        panel._clear_script()
        self.assertEqual(panel._timeline_total, 0)
        self.assertIn("剩余 00:00", panel.timing_label.text())

    def test_timer_does_not_read_audio_again(self):
        panel = self.panel("One. Two.", ready=True)
        panel._play_paragraph()
        self._engine_at(panel, .2)
        with patch.object(panel, "_paragraph_audio", side_effect=AssertionError("audio read on UI timer")):
            panel._advance_cursor()

    def test_double_click_uses_clicked_row_even_if_selection_changed(self):
        panel = self.panel("One.\nTwo.\nThree.", ready=True)
        panel.paragraph_table.selectRow(0)
        panel._play_from_here()
        self._engine_at(panel, .2)
        panel._advance_cursor()
        clicked = panel.paragraph_table.model().index(2, 1)
        panel.paragraph_table.doubleClicked.emit(clicked)
        self.assertEqual(self.engine.play_script_segment.call_args.kwargs["text"], "Three.")
        self.assertEqual(panel._playing[0].paragraph, 2)
        self.assertEqual(panel.paragraph_table.currentRow(), 2)
        # The previous buffer may report one final tick before the new one starts.
        panel._advance_cursor()
        self.assertEqual(panel._playing[0].paragraph, 2)
        self.assertEqual(self.engine.play_script_segment.call_count, 2)
        self._engine_at(panel, .1)
        panel._advance_cursor()
        self.assertAlmostEqual(panel._timeline_position, 2.7)

    def test_browsing_another_paragraph_is_not_stolen_by_playback_timer(self):
        panel = self.panel("One. Two.\nThree.", ready=True)
        panel._play_paragraph()
        panel.paragraph_table.selectRow(1)
        self._engine_at(panel, .2)
        panel._advance_cursor()
        self.assertEqual(panel.paragraph_table.currentRow(), 1)
        self._engine_at(panel, 1.0)
        panel._advance_cursor()
        self.assertEqual(panel.paragraph_table.currentRow(), 1)

    def test_single_sentence_double_click_uses_event_row(self):
        panel = self.panel("One. Two.", ready=True)
        panel._play_paragraph()
        panel.line_table.selectRow(0)
        panel.line_table.doubleClicked.emit(panel.line_table.model().index(1, 1))
        self.assertEqual(self.engine.play_script_segment.call_args.kwargs["text"], "Two.")
        self.assertAlmostEqual(panel._timeline_position, .85)

    def test_long_duration_formats_hours(self):
        panel = self.panel()
        self.assertEqual(panel._format_script_time(3723), "1:02:03")


if __name__ == "__main__":
    unittest.main()
