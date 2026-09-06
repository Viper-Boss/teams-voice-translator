import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

# Ensure local package import works when running tests directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teams_voice_translator.ui import MainWindow  # noqa: E402
from teams_voice_translator.voicestudio import VoiceStudioClient  # noqa: E402
from teams_voice_translator.voicestudio_manager import (  # noqa: E402
    VoiceStudioInstallation,
    VoiceStudioRelease,
    VoiceStudioRuntime,
)


class TestMeetingUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        os.environ["APPDATA"] = self.tmp
        # Geometry tests do not need a real global pynput listener. Mocking it
        # also avoids a native Windows teardown race after the test suite.
        with patch.object(MainWindow, "start_hotkeys"):
            self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_long_text_controls_exist(self) -> None:
        self.assertTrue(hasattr(self.window, "long_text_input"))
        self.assertTrue(hasattr(self.window, "long_text_progress"))
        self.assertTrue(hasattr(self.window, "long_text_play"))
        self.assertTrue(hasattr(self.window, "long_text_pause"))
        self.assertTrue(hasattr(self.window, "long_text_stop"))
        self.assertTrue(hasattr(self.window, "long_text_status"))
        self.assertTrue(hasattr(self.window, "long_form_dialog"))
        self.assertEqual(self.window.long_text_play.text(), "▶ 双语整段朗读")

    def test_long_text_initial_state(self) -> None:
        self.assertTrue(self.window.long_text_play.isEnabled())
        self.assertFalse(self.window.long_text_pause.isEnabled())
        self.assertFalse(self.window.long_text_stop.isEnabled())
        self.assertEqual(self.window.long_text_progress.maximum(), 0)

    def test_split_long_text_by_lines(self) -> None:
        text = "第一句。\n第二句比较长需要被切分 " * 30
        chunks = MainWindow.split_long_text(text, max_len=80)
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 80)

    def test_history_replay_button(self) -> None:
        self.window._append_history_row("12:00", "我", "你好", "Hello", "1.2s")
        self.assertEqual(self.window.history.rowCount(), 1)
        widget = self.window.history.cellWidget(0, 5)
        self.assertIsNotNone(widget)
        self.assertEqual(widget.text(), "▶ 重播")
        self.assertGreaterEqual(widget.minimumWidth(), 68)

    def test_history_table_shows_five_rows(self) -> None:
        self.assertEqual(self.window.history_visible_rows, 5)
        row_height = self.window.history.verticalHeader().defaultSectionSize()
        self.assertGreaterEqual(
            self.window.history.minimumHeight(),
            32 + 5 * row_height,
        )

    def test_voice_controls_have_separate_rows(self) -> None:
        self.assertGreaterEqual(self.window.voice_group.minimumHeight(), 160)
        self.assertEqual(self.window.voice_group.layout().count(), 2)
        self.assertGreaterEqual(self.window.direct_button.height(), 64)
        self.assertGreaterEqual(self.window.translate_button.height(), 64)
        self.assertGreaterEqual(self.window.play_button.minimumHeight(), 32)
        self.assertGreaterEqual(self.window.teacher_button.minimumHeight(), 32)

    def test_compact_workspace_and_local_theme_controls(self) -> None:
        self.assertLessEqual(self.window.width(), 1460)
        self.assertEqual(self.window.layout_quick.count(), 4)
        self.assertEqual(self.window.layout_quick.itemData(0), "crystal")
        self.assertEqual(self.window.ui_layout.currentData(), "crystal")
        self.assertEqual(self.window.theme_quick.count(), 4)
        self.assertEqual(self.window.theme_quick.itemData(0), "shizuku")
        self.assertEqual(self.window.theme.currentData(), "shizuku")
        self.assertEqual(self.window.tts_instruction.maxLength(), 100)
        self.assertTrue(hasattr(self.window, "artwork_panel"))

    def test_four_layouts_and_four_palettes_are_independent(self) -> None:
        combinations = []
        for layout_name in ("crystal", "signal", "studio", "fluent"):
            for palette_name in ("shizuku", "dark", "warm", "light"):
                self.window._select_data(self.window.layout_quick, layout_name)
                self.window._select_data(self.window.theme_quick, palette_name)
                self.window.apply_layout()
                self.window.apply_theme()
                combinations.append((layout_name, palette_name))
                self.assertEqual(self.window.layout_quick.currentData(), layout_name)
                self.assertEqual(self.window.theme_quick.currentData(), palette_name)
        self.assertEqual(len(set(combinations)), 16)
        # The artwork panel is owned by the backdrop selector, not the layout,
        # so it survives all 16 combinations. Use isVisibleTo(): plain
        # isVisible() is always False here because the test window is never
        # shown, which made an earlier assertFalse() pass vacuously.
        self.assertTrue(self.window.artwork_panel.isVisibleTo(self.window))
        self.assertIn("font-size: 19px", self.window.styleSheet())

    def test_backdrop_is_a_third_independent_axis(self) -> None:
        self.assertEqual(self.window.backdrop_quick.count(), 8)
        self.assertEqual(self.window.backdrop_quick.itemData(0), "auto")
        self.assertEqual(self.window.backdrop_quick.itemData(7), "none")
        self.assertEqual(self.window.backdrop.currentData(), "auto")

        # Switching layout or palette must never overwrite the backdrop choice.
        for layout_name in ("crystal", "signal", "studio", "fluent"):
            for palette_name in ("shizuku", "dark", "warm", "light"):
                self.window._select_data(self.window.backdrop_quick, "sakura")
                self.window._select_data(self.window.layout_quick, layout_name)
                self.window._select_data(self.window.theme_quick, palette_name)
                self.window.apply_layout()
                self.window.apply_theme()
                self.assertEqual(self.window.backdrop_quick.currentData(), "sakura")
                self.assertEqual(self.window.backdrop.currentData(), "sakura")
                self.assertTrue(self.window.artwork_panel.isVisibleTo(self.window))

    def test_backdrop_auto_follows_palette(self) -> None:
        from teams_voice_translator.backdrops import backdrop_path
        expected = {
            "shizuku": "水晶雫 · 蓝白水晶",
            "light": "水晶雫 · 蓝白水晶",
            "dark": "星空 · 银河深蓝",
            "warm": "暖阳 · 金色黄昏",
        }
        self.window._select_data(self.window.backdrop_quick, "auto")
        for palette_name, credit in expected.items():
            self.window._select_data(self.window.theme_quick, palette_name)
            self.window.apply_theme()
            path = backdrop_path({"shizuku": "shizuku", "light": "shizuku",
                                  "dark": "stellar", "warm": "amber"}[palette_name])
            if path is not None and path.exists():
                self.assertEqual(self.window.art_credit.text().splitlines()[0], credit)
            else:
                self.assertIn("背景图缺失", self.window.art_credit.text())

    def test_backdrop_none_hides_panel_in_every_layout(self) -> None:
        self.window._select_data(self.window.backdrop_quick, "none")
        for layout_name in ("crystal", "signal", "studio", "fluent"):
            self.window._select_data(self.window.layout_quick, layout_name)
            self.window.apply_layout()
            self.window.apply_theme()
            self.assertFalse(self.window.artwork_panel.isVisibleTo(self.window))

    def test_backdrop_choice_round_trips_through_settings(self) -> None:
        self.window._select_data(self.window.backdrop_quick, "violet")
        values = self.window.current_settings()
        self.assertEqual(values["backdrop"], "violet")
        self.assertEqual(self.window.backdrop.currentData(), "violet")

    def test_artwork_panel_width_adapts_to_layout(self) -> None:
        widths = {}
        for layout_name in ("crystal", "signal", "studio", "fluent"):
            self.window._select_data(self.window.layout_quick, layout_name)
            self.window.apply_layout()
            widths[layout_name] = self.window.artwork_panel.width()
        # Crystal keeps the roomy original panel; the compact layouts get a
        # narrower strip so their content area is not squeezed.
        self.assertGreater(widths["crystal"], widths["fluent"])
        self.assertGreater(widths["studio"], widths["signal"])

    def test_voicestudio_has_four_distinct_reference_layouts(self) -> None:
        grid = self.window.voicestudio_dashboard_grid

        def position(widget):
            index = grid.indexOf(widget)
            self.assertGreaterEqual(index, 0)
            return grid.getItemPosition(index)

        self.window._select_data(self.window.layout_quick, "crystal")
        self.window.apply_layout()
        self.assertEqual(position(self.window.voicestudio_nav_card)[1], 0)
        self.assertEqual(position(self.window.voicestudio_library_card), (3, 1, 1, 2))
        self.assertEqual(grid.indexOf(self.window.voicestudio_preview_card), -1)
        self.assertTrue(self.window.voicestudio_status.isHidden())
        self.assertTrue(self.window.voicestudio_progress_label.isHidden())

        self.window._select_data(self.window.layout_quick, "fluent")
        self.window.apply_layout()
        self.assertEqual(position(self.window.voicestudio_library_card)[1], 2)
        self.assertEqual(position(self.window.voicestudio_versions_card)[:2], (0, 0))
        self.assertEqual(grid.indexOf(self.window.voicestudio_nav_card), -1)

        self.window._select_data(self.window.layout_quick, "signal")
        self.window.apply_layout()
        self.assertEqual(position(self.window.voicestudio_versions_card)[:2], (1, 1))
        self.assertEqual(position(self.window.voicestudio_preview_card)[:2], (3, 2))
        self.assertEqual(grid.indexOf(self.window.voicestudio_nav_card), -1)

        self.window._select_data(self.window.layout_quick, "studio")
        self.window.apply_layout()
        self.assertEqual(grid.indexOf(self.window.voicestudio_nav_card), -1)
        self.assertEqual(position(self.window.voicestudio_connection_card)[:2], (0, 0))
        self.assertEqual(position(self.window.voicestudio_library_card)[:2], (2, 0))

    def test_whole_window_shell_changes_with_reference_layout(self) -> None:
        shell = self.window.main_shell_grid

        self.window._select_data(self.window.layout_quick, "crystal")
        self.window.apply_layout()
        self.assertEqual(shell.indexOf(self.window.main_navigation), -1)
        self.assertFalse(self.window.tabs.tabBar().isHidden())
        self.assertFalse(self.window.header_title_container.isHidden())

        for layout_name in ("fluent", "signal"):
            self.window._select_data(self.window.layout_quick, layout_name)
            self.window.apply_layout()
            self.assertGreaterEqual(shell.indexOf(self.window.main_navigation), 0)
            self.assertTrue(self.window.tabs.tabBar().isHidden())
            self.assertTrue(self.window.header_title_container.isHidden())

        self.window._select_data(self.window.layout_quick, "studio")
        self.window.apply_layout()
        self.assertEqual(shell.indexOf(self.window.main_navigation), -1)
        self.assertFalse(self.window.tabs.tabBar().isHidden())
        self.assertFalse(self.window.app_footer.isHidden())

    def test_crystal_header_keeps_title_clear_of_hidden_quick_controls(self) -> None:
        self.window._select_data(self.window.layout_quick, "crystal")
        self.window.apply_layout()
        self.window.resize(1180, 800)
        self.window.show()
        self.app.processEvents()

        self.assertTrue(self.window.layout_quick.isHidden())
        self.assertTrue(self.window.theme_quick.isHidden())
        self.assertTrue(self.window.backdrop_quick.isHidden())
        self.assertGreaterEqual(self.window.header_title_container.width(), 470)
        self.assertGreaterEqual(self.window.header_title_container.geometry().left(), 0)
        self.assertLessEqual(
            self.window.header_title_container.geometry().right(),
            self.window.header_frame.contentsRect().right(),
        )

    def test_voicestudio_tiles_use_one_outline_icon_family(self) -> None:
        self.assertEqual(len(self.window.voicestudio_tile_icons), 8)
        self.assertTrue(
            all(name.startswith("ph.") for name, _color in self.window.voicestudio_tile_icons)
        )
        self.assertTrue(all(not button.icon().isNull() for button in self.window.voicestudio_lifecycle_buttons))

    def test_version_fact_labels_are_transparent(self) -> None:
        self.assertIn(
            "QFrame#versionFactCard QLabel { background: transparent;",
            self.window.styleSheet(),
        )
        labels = (
            self.window.voicestudio_installed_title,
            self.window.voicestudio_installed_version,
            self.window.voicestudio_installed_meta,
            self.window.voicestudio_api_title,
            self.window.voicestudio_api_version,
            self.window.voicestudio_api_meta,
            self.window.voicestudio_latest_title,
            self.window.voicestudio_latest_version,
            self.window.voicestudio_latest_meta,
        )
        self.assertTrue(all(not label.autoFillBackground() for label in labels))

    def test_github_release_card_shows_full_sha256_without_contains_wording(self) -> None:
        checksum = "ab" * 32
        snapshot = VoiceStudioRuntime(
            processes=(),
            api_ok=False,
            installation=VoiceStudioInstallation(),
            latest_release=VoiceStudioRelease(
                version="0.5.1",
                tag="v0.5.1",
                asset_name="VoiceStudio_0.5.1_x64_en-US.msi",
                download_url="https://example.invalid/VoiceStudio.msi",
                expected_sha256=checksum.upper(),
            ),
        )

        self.window._apply_voicestudio_runtime(snapshot)

        self.assertEqual(self.window.voicestudio_latest_version.text(), "v0.5.1")
        self.assertEqual(
            self.window.voicestudio_latest_meta.text(),
            f"SHA-256: {checksum}",
        )
        self.assertNotIn("含", self.window.voicestudio_latest_version.text())
        self.assertTrue(
            self.window.voicestudio_latest_meta.textInteractionFlags()
            & Qt.TextSelectableByMouse
        )

    def test_settings_reflow_avoids_sidebar_horizontal_clipping(self) -> None:
        grid = self.window.settings_grid

        def position(widget):
            index = grid.indexOf(widget)
            self.assertGreaterEqual(index, 0)
            return grid.getItemPosition(index)

        self.assertEqual(
            self.window.settings_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )
        for layout_name in ("fluent", "signal"):
            self.window._select_data(self.window.layout_quick, layout_name)
            self.window.apply_layout()
            self.assertEqual(position(self.window.settings_primary_groups[0])[:2], (0, 0))
            self.assertEqual(position(self.window.settings_primary_groups[1])[:2], (1, 0))
            self.assertEqual(position(self.window.advanced_panel)[1], 0)

        for layout_name in ("crystal", "studio"):
            self.window._select_data(self.window.layout_quick, layout_name)
            self.window.apply_layout()
            self.assertEqual(position(self.window.settings_primary_groups[0])[:2], (0, 0))
            self.assertEqual(position(self.window.settings_primary_groups[1])[:2], (0, 1))
            self.assertEqual(position(self.window.advanced_panel)[2:], (1, 2))

    def test_engine_scope_button_switches_local_and_cloud_tts(self) -> None:
        self.window._select_data(self.window.tts_provider, "aliyun")
        self.window.update_tts_model_ui()
        self.assertIn("云端引擎", self.window.engine_scope_button.text())
        with patch.object(self.window, "refresh_voicestudio_runtime"):
            self.window.toggle_tts_provider()
        self.assertEqual(self.window.tts_provider.currentData(), "voicestudio")
        self.assertIn("本地引擎", self.window.engine_scope_button.text())
        self.assertEqual(self.window.settings.get("tts_provider"), "voicestudio")
        self.window.toggle_tts_provider()
        self.assertEqual(self.window.tts_provider.currentData(), "aliyun")

    def test_long_text_controls_delegate_to_reader_dialog(self) -> None:
        reader = Mock()
        self.window.long_form_dialog = reader
        self.window.pause_or_resume_long_text()
        self.window.stop_long_text_speech()
        reader.toggle_pause.assert_called_once_with()
        reader.stop.assert_called_once_with()

    def test_new_long_form_settings_are_available(self) -> None:
        self.assertEqual(self.window.long_form_model.currentText(), "qwen-plus")
        self.window.tts_pronunciations.setPlainText('{"OpenAI":"Open A I"}')
        values = self.window.current_settings()
        self.assertEqual(values["long_form_model"], "qwen-plus")
        self.assertIn("Open A I", values["tts_pronunciations"])

    def test_voicestudio_workspace_and_tts_routing(self) -> None:
        self.assertEqual(self.window.tabs.tabText(1), "本地声音工作台")
        self.assertFalse(self.window.tabs.tabIcon(1).isNull())
        self.window._select_data(self.window.tts_provider, "voicestudio")
        self.window.voicestudio_model.setCurrentText("voxcpm2")
        self.window.voicestudio_voice.addItem("My voice", "profile-1")
        self.window._select_data(self.window.voicestudio_voice, "profile-1")
        values = self.window.current_settings()
        self.assertEqual(values["tts_provider"], "voicestudio")
        self.assertEqual(values["voicestudio_model"], "voxcpm2")
        self.assertEqual(values["voicestudio_voice"], "profile-1")
        self.assertIsInstance(self.window._make_tts_client(values), VoiceStudioClient)
        self.assertFalse(self.window.tts_model.isEnabled())
        self.assertTrue(self.window.tts_rate.isEnabled())

    def test_voicestudio_lifecycle_management_controls_exist(self) -> None:
        self.assertEqual(self.window.voicestudio_inner_tabs.count(), 2)
        self.assertEqual(self.window.voicestudio_inner_tabs.tabText(0), "运行管理")
        self.assertTrue(hasattr(self.window, "voicestudio_install_button"))
        self.assertTrue(hasattr(self.window, "voicestudio_import_button"))
        self.assertTrue(hasattr(self.window, "voicestudio_download_button"))
        self.assertTrue(hasattr(self.window, "voicestudio_upgrade_button"))
        self.assertTrue(hasattr(self.window, "voicestudio_cancel_task_button"))
        self.assertTrue(hasattr(self.window, "voicestudio_process_table"))
        self.assertEqual(self.window.voicestudio_process_table.columnCount(), 7)
        self.assertGreaterEqual(self.window.voicestudio_choose_install_dir_button.minimumWidth(), 40)
        self.assertGreaterEqual(self.window.voicestudio_choose_exe_button.minimumWidth(), 40)
        self.assertGreaterEqual(self.window.voicestudio_refresh_button.minimumWidth(), 40)
        self.assertTrue(self.window.voicestudio_manager_scroll.widgetResizable())
        self.assertGreaterEqual(self.window.voicestudio_install_dir.minimumHeight(), 38)
        self.assertGreaterEqual(self.window.voicestudio_install_button.minimumHeight(), 38)
        self.assertGreaterEqual(self.window.voicestudio_backend_switch.minimumHeight(), 38)
        self.assertTrue(hasattr(self.window, "voicestudio_preview_rate"))
        self.assertTrue(hasattr(self.window, "voicestudio_preview_volume"))
        self.window.voicestudio_install_dir.setText(r"D:\Apps\VoiceStudio")
        self.assertEqual(
            self.window.current_settings()["voicestudio_install_dir"],
            r"D:\Apps\VoiceStudio",
        )

    def test_voicestudio_progress_replaces_tiles_and_can_cancel(self) -> None:
        self.window._voicestudio_task_running = True
        self.window._voicestudio_task_cancellable = True
        self.window._voicestudio_task_cancel_event = threading.Event()
        self.window._on_voicestudio_task_progress("正在下载 18.5 MB / 162.2 MB", 11)

        self.assertFalse(self.window.voicestudio_task_bar.isHidden())
        self.assertFalse(self.window.voicestudio_cancel_task_button.isHidden())
        self.assertTrue(
            all(button.isHidden() for button in self.window.voicestudio_lifecycle_buttons)
        )
        self.assertEqual(self.window.voicestudio_progress.value(), 11)

        self.window.cancel_voicestudio_task()
        self.assertTrue(self.window._voicestudio_task_cancel_event.is_set())
        self.assertFalse(self.window.voicestudio_cancel_task_button.isEnabled())

        with patch.object(self.window, "_schedule_voicestudio_refreshes"):
            self.window._on_voicestudio_task_finished(
                False,
                "VoiceStudio 操作已取消。",
                {"cancelled": True},
            )
        self.assertEqual(self.window.voicestudio_progress_label.text(), "操作已取消")
        self.window._hide_voicestudio_task_progress()
        self.assertTrue(self.window.voicestudio_task_bar.isHidden())
        self.assertTrue(
            all(not button.isHidden() for button in self.window.voicestudio_lifecycle_buttons)
        )

    def test_voicestudio_refreshes_are_staggered_after_install(self) -> None:
        with patch("teams_voice_translator.ui.QTimer.singleShot") as single_shot:
            self.window._schedule_voicestudio_refreshes(include_catalog=True)
        delays = [call.args[0] for call in single_shot.call_args_list]
        self.assertEqual(delays[:4], [0, 1500, 4000, 8000])
        self.assertEqual(delays[4:], [2200, 5500, 9000])

    def test_close_waits_for_managed_voicestudio_processes(self) -> None:
        event = Mock()
        self.window.voicestudio_manager.has_managed_processes = Mock(return_value=True)
        self.window._begin_voicestudio_shutdown = Mock()
        self.window.closeEvent(event)
        event.ignore.assert_called_once_with()
        self.window._begin_voicestudio_shutdown.assert_called_once_with()
        self.window.voicestudio_manager.has_managed_processes.return_value = False


if __name__ == "__main__":
    unittest.main()
