import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from array import array
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from teams_voice_translator.audio import AudioDevice, useful_devices, resolve_monitor_device, resolve_program_output
from teams_voice_translator.defense.device_test import play_device_test


class OutputTest(unittest.TestCase):
    def test_uses_selected_endpoint_native_rate_stereo_and_bounded_volume(self):
        stream = MagicMock()
        stream.__enter__.return_value = stream
        with patch('teams_voice_translator.defense.device_test.sd.query_devices', return_value={'max_output_channels':2,'default_samplerate':48000}), \
                patch('teams_voice_translator.defense.device_test.sd.RawOutputStream', return_value=stream) as create:
            self.assertTrue(play_device_test(44, threading.Event()))
        create.assert_called_once_with(device=44,samplerate=48000,channels=2,dtype='int16')
        pcm = b''.join(c.args[0] for c in stream.write.call_args_list)
        samples = array('h', pcm)
        self.assertEqual(len(samples), 48000 * 2 * 1.2)
        self.assertTrue(any(samples))
        self.assertLessEqual(max(abs(x) for x in samples), 3933)
        self.assertEqual(samples[::2], samples[1::2])

    def test_error_does_not_fall_back_to_other_device(self):
        with patch('teams_voice_translator.defense.device_test.sd.query_devices', return_value={'max_output_channels':2,'default_samplerate':48000}), \
                patch('teams_voice_translator.defense.device_test.sd.RawOutputStream', side_effect=RuntimeError('driver error')) as create:
            with self.assertRaisesRegex(RuntimeError,'driver error'):
                play_device_test(44, threading.Event())
            self.assertEqual(create.call_count,1)
            self.assertEqual(create.call_args.kwargs['device'],44)

    def test_stop_aborts_selected_stream_and_releases_it(self):
        stop = threading.Event()
        stream = MagicMock()
        stream.__enter__.return_value = stream
        stream.write.side_effect = lambda _: stop.set()
        with patch('teams_voice_translator.defense.device_test.sd.query_devices', return_value={'max_output_channels':2,'default_samplerate':48000}), \
                patch('teams_voice_translator.defense.device_test.sd.RawOutputStream',return_value=stream):
            self.assertFalse(play_device_test(44,stop))
        stream.abort.assert_called_once()
        stream.__exit__.assert_called_once()

    def test_compact_keeps_unique_headphones_and_selected_driver_without_duplicates(self):
        items = [AudioDevice(6,'小爱音箱','MME',0,2),AudioDevice(19,'小爱音箱','Windows DirectSound',0,2),
                 AudioDevice(26,'小爱音箱','Windows WASAPI',0,2),AudioDevice(44,'耳机 (Realtek USB Audio)','Windows WDM-KS',0,2)]
        self.assertEqual([d.index for d in useful_devices(items,kind='monitor')],[6,44])
        self.assertEqual([d.index for d in useful_devices(items,kind='monitor',keep_index=26)],[26,44])

    def test_identity_follows_renumbering_but_does_not_use_reused_index(self):
        class Settings:
            def get(self,key,default=None):
                return {'monitor_output_device':6,'monitor_output_device_name':'小爱音箱','monitor_output_host_api':'MME'}.get(key,default)
        rows = [AudioDevice(6,'Realtek 耳机','MME',0,2),AudioDevice(8,'小爱音箱','MME',0,2)]
        with patch('teams_voice_translator.audio.list_audio_devices',return_value=([],rows)):
            self.assertEqual(resolve_monitor_device(Settings()),8)
        with patch('teams_voice_translator.audio.list_audio_devices',return_value=([],rows[:1])):
            with self.assertRaisesRegex(RuntimeError,'断开'):
                resolve_monitor_device(Settings())

    def test_meeting_output_follows_cable_name_instead_of_reused_physical_index(self):
        class Settings:
            shared={'teams_output_device':7}
            def get(self,key,default=None):
                return {'teams_output_device_name':'CABLE In 16ch'}.get(key,default)
        rows=[AudioDevice(7,'Realtek 扬声器','MME',0,2),AudioDevice(9,'CABLE In 16ch','MME',0,2)]
        with patch('teams_voice_translator.audio.list_audio_devices',return_value=([],rows)):
            self.assertEqual(resolve_program_output(Settings()),9)

    def test_ui_test_uses_unsaved_selection_and_stop_cancels_without_saving(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense import ui
        from teams_voice_translator.defense.settings import DefenseSettings
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp:
            settings = DefenseSettings(Path(temp))
            settings.set('monitor_output_device',6)
            devices = [AudioDevice(6,'小爱音箱','MME',0,2),AudioDevice(44,'耳机 (Realtek USB Audio)','Windows WDM-KS',0,2)]
            with patch.object(ui,'_safe_list_audio_devices',return_value=([],devices)),patch.object(ui,'_safe_list_loopback_devices',return_value=[]):
                dialog = ui.SettingsDialog(None,settings)
            dialog.monitor_combo.setCurrentIndex(dialog.monitor_combo.findData(44))
            entered = threading.Event()
            seen = []
            def play(device,stop):
                seen.append(device)
                entered.set()
                stop.wait(2)
                return False
            dialog.device_test_player = play
            dialog.device_test_button.click()
            self.assertTrue(entered.wait(2))
            dialog.device_test_button.click()
            self.assertTrue(dialog._device_test_stop.is_set())
            deadline=time.monotonic()+2
            while dialog._device_testing and time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertFalse(dialog._device_testing)
            self.assertEqual(seen,[44])
            self.assertEqual(settings.get('monitor_output_device'),6)
            dialog.reject()

    def test_direct_bridge_uses_both_meeting_and_monitor_outputs(self):
        from teams_voice_translator.audio import DirectAudioBridge
        player = MagicMock()
        with patch('teams_voice_translator.audio.MultiOutputPlayer',return_value=player) as create, \
                patch('teams_voice_translator.audio.MicrophoneCapture'):
            bridge = DirectAudioBridge()
            bridge.start(1,7,48000,monitor_devices=[44])
            bridge.stop()
        create.assert_called_once_with([7,44],48000)

    def test_stereo_only_headphones_retry_same_device_and_convert_pcm(self):
        from teams_voice_translator.audio import MultiOutputPlayer, sd
        stream=MagicMock()
        with patch('teams_voice_translator.audio.sd.RawOutputStream',side_effect=[sd.PortAudioError('Invalid channels',-9998),stream]) as create, \
                patch('teams_voice_translator.audio.sd.query_devices',return_value={'default_samplerate':48000,'max_output_channels':2}):
            with MultiOutputPlayer([44],24000) as player:
                player.write(array('h',[1000,-1000]*240).tobytes())
        self.assertEqual([c.kwargs['device'] for c in create.call_args_list],[44,44])
        self.assertEqual(create.call_args.kwargs['channels'],2)
        values=array('h',stream.write.call_args.args[0])
        self.assertEqual(values[::2],values[1::2])
        self.assertGreater(len(values),960)

    def test_transient_settings_replace_denial_retries_without_losing_data(self):
        from teams_voice_translator.defense.settings import DefenseSettings
        original = Path.replace
        attempts = []
        def replace(path,target):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError('temporary scanner lock')
            return original(path,target)
        with tempfile.TemporaryDirectory() as temp:
            settings = DefenseSettings(Path(temp))
            with patch.object(Path,'replace',replace):
                settings.set('monitor_output_device',44)
            self.assertEqual(DefenseSettings(Path(temp)).get('monitor_output_device'),44)
            self.assertEqual(len(attempts),2)


if __name__ == '__main__':
    unittest.main()
