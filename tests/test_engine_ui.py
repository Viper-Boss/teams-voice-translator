import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from teams_voice_translator.ui import MainWindow


class EngineUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["APPDATA"] = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_engine_radios_drive_hidden_combo_and_grey_out(self):
        w = self.window
        self.assertFalse(w.translation_engine.isVisible())
        w.engine_classic_radio.setChecked(True)
        self.assertEqual(w.translation_engine.currentData(), "classic")
        self.assertFalse(w.live_options.isEnabled())
        self.assertFalse(w.live_voice.isEnabled())

        w.engine_live_radio.setChecked(True)
        self.assertEqual(w.translation_engine.currentData(), "live")
        self.assertTrue(w.live_options.isEnabled())

    def test_combo_selection_updates_radios(self):
        w = self.window
        index = w.translation_engine.findData("classic")
        w.translation_engine.setCurrentIndex(index)
        self.assertTrue(w.engine_classic_radio.isChecked())
        self.assertFalse(w.engine_live_radio.isChecked())
        self.assertFalse(w.live_options.isEnabled())

    def test_fixed_voice_requires_live_engine(self):
        w = self.window
        w.engine_live_radio.setChecked(True)
        index = w.live_voice_clone_mode.findData("fixed")
        w.live_voice_clone_mode.setCurrentIndex(index)
        self.assertTrue(w.live_voice.isEnabled())
        w.engine_classic_radio.setChecked(True)
        self.assertFalse(w.live_voice.isEnabled())


if __name__ == "__main__":
    unittest.main()
