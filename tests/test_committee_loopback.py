import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from teams_voice_translator.audio import (
    LoopbackDevice, choose_committee_loopback, is_program_output_loopback,
)
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
from teams_voice_translator.defense.settings import DefenseSettings


def device(index, name):
    return LoopbackDevice(index, name, 2, 48000)


class CommitteeLoopbackTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            device(28, 'CABLE In 16ch (VB-Audio Virtual Cable) [Loopback]'),
            device(29, 'Speakers (HECATE G4) [Loopback]'),
            device(32, 'CABLE Input (VB-Audio Virtual Cable) [Loopback]'),
        ]

    def test_stale_index_never_resolves_to_program_output(self):
        selected = choose_committee_loopback(self.devices, 28, '')
        self.assertEqual(selected.index, 29)
        self.assertTrue(is_program_output_loopback(self.devices[0].name))

    def test_saved_name_survives_device_index_change(self):
        selected = choose_committee_loopback(self.devices, 999, 'Speakers (HECATE G4) [Loopback]')
        self.assertEqual(selected.index, 29)

    def test_engine_repairs_index_and_persists_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = DefenseSettings(Path(tmp))
            settings.shared.update({'loopback_device': 28})
            settings.get_api_key = lambda: 'key'
            statuses, names = [], []
            engine = DefenseEngine(settings, EngineCallbacks(
                on_status=statuses.append, on_committee_device=names.append,
            ))
            asr = Mock()
            capture = Mock()
            capture.start.return_value = self.devices[1]
            with patch('teams_voice_translator.defense.pipeline.list_loopback_devices', return_value=self.devices), \
                 patch('teams_voice_translator.defense.pipeline.CommitteeASRProcess', return_value=asr), \
                 patch('teams_voice_translator.defense.pipeline.SystemAudioCapture', return_value=capture):
                engine._start_committee_listening(28)
            capture.start.assert_called_once_with(29, engine._on_loopback_audio, target_rate=16000)
            self.assertEqual(settings.shared.get('loopback_device'), 29)
            self.assertIn('HECATE', settings.shared.get('loopback_device_name'))
            self.assertTrue(any('编号已变化' in status for status in statuses))
            engine.shutdown()

    def test_committee_is_received_while_direct_or_translated_output_is_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DefenseEngine(DefenseSettings(Path(tmp)), EngineCallbacks())
            engine.asr_committee = Mock()
            engine._direct_requested = True
            engine._on_loopback_audio(b'committee during F8 start')
            engine._direct_requested = False
            engine._direct_active = True
            engine._on_loopback_audio(b'committee during F8')
            engine._direct_active = False
            engine._suppress_until = time.monotonic() + .5
            engine._on_loopback_audio(b'committee during TTS')
            engine._suppress_until = 0
            engine._on_loopback_audio(b'committee')
            self.assertEqual([call.args[0] for call in engine.asr_committee.send_audio.call_args_list],
                             [b'committee during F8 start', b'committee during F8',
                              b'committee during TTS', b'committee'])
            engine.shutdown()


if __name__ == '__main__': unittest.main()
