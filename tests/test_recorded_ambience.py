import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from array import array
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
from teams_voice_translator.defense.ambience import (BACKGROUND_CHOICES, RecordedRoomTone,
    RoomTone, make_room_tone, PEAK_CAP_DB, background_mode)


class RecordedAmbienceTests(unittest.TestCase):
    def test_every_selectable_background_uses_bundled_recording_without_noise_generator(self):
        signals=[]
        for mode,label in BACKGROUND_CHOICES:
            if mode=='off':
                continue
            with patch.object(RoomTone,'_noise',side_effect=AssertionError('synthetic generator used')):
                tone=make_room_tone(mode,volume=100)
                self.assertIsInstance(tone,RecordedRoomTone)
                pcm=tone.bed(24000)
                self.assertTrue(any(array('h',pcm)))
                self.assertLessEqual(max(abs(x) for x in array('h',pcm)),32767*10**(PEAK_CAP_DB/20)+1)
                signals.append(pcm)
        self.assertEqual(len(set(signals)),4)

    def test_recording_position_continues_across_blocks_and_loops(self):
        tone=make_room_tone('rain')
        count=len(tone._recorded_samples)+100
        whole=make_room_tone('rain').bed(count)
        split=tone.bed(1000)+tone.bed(count-1000)
        self.assertEqual(split,whole)
        self.assertEqual(make_room_tone('off').mix(b'\x01\x00'),b'\x01\x00')

    def test_unavailable_scenes_are_not_presented_as_recordings(self):
        modes=dict(BACKGROUND_CHOICES)
        self.assertNotIn('airport',modes)
        self.assertNotIn('rail',modes)
        self.assertEqual(background_mode('airport'),'off')

    def test_main_controls_save_immediately_without_restarting_recognition(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense import ui
        from teams_voice_translator.defense.settings import DefenseSettings
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            settings=DefenseSettings(Path(tmp))
            with patch.object(ui,'apply_default_loopback',return_value=False), \
                    patch.object(ui.DefenseEngine,'start'), \
                    patch.object(ui.DefenseWindow,'_auto_prerender'), \
                    patch.object(ui.DefenseEngine,'refresh_ambience') as refresh, \
                    patch.object(ui.DefenseEngine,'apply_settings') as restart:
                window=ui.DefenseWindow(settings)
                window.background_combo.setCurrentIndex(window.background_combo.findData('rain'))
                self.assertEqual(settings.get('ambience_mode'),'rain')
                self.assertTrue(window.background_toggle.isChecked())
                window.background_toggle.click()
                self.assertEqual(settings.get('ambience_mode'),'off')
                window.background_toggle.click()
                self.assertEqual(settings.get('ambience_mode'),'rain')
                refresh.assert_called()
                restart.assert_not_called()
                window.close()
                app.processEvents()


if __name__=='__main__':
    unittest.main()
