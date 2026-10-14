"""Offline channel lifecycle tests; never connect to an audio device or cloud."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from teams_voice_translator.aliyun import ApiError
from teams_voice_translator.defense.asr_standby import StandbyASR, create_defense_asr
from teams_voice_translator.defense.tts_session import TtsBackupSession, _Cancelled, create_tts_session
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.speech_policy import speech_settings


PRIMARY_PCM = b'\x01\x00' * 30
BACKUP_PCM = b'\x02\x00' * 30


class SpeechRaceTests(unittest.TestCase):
    """现场发声第二通道：只交出一份音频。单个通道的合成换成假的（_channel_call）。"""

    def session(self, behaviour, **settings):
        """behaviour[index] = [(等多久, 一包音频 或 异常), ...]"""
        session = TtsBackupSession(None, {'tts_model':'cosyvoice-v3.5-plus', 'voice':'v1',
            'speech_mode':'natural', 'backup_tts_enabled':True, 'backup_tts_mode':'race',
            'backup_tts_model':'cosyvoice-v3.5-flash', 'backup_tts_voice_id':'v2', **settings},
            on_audio=lambda _: None)
        self.calls, self.flags, self.texts = [], {}, {}

        def fake(index, channel_settings, text, emit, cancel_event):
            self.calls.append(index); self.flags[index] = cancel_event; self.texts[index] = text
            self.channel_settings = getattr(self, 'channel_settings', {}); self.channel_settings[index] = channel_settings
            for delay, item in behaviour[index]:
                if cancel_event.wait(delay):
                    raise ApiError('cancelled')
                if isinstance(item, Exception):
                    raise item
                emit(item)
        session._channel_call = fake
        return session

    def test_first_complete_audio_wins_only_once_and_loser_cancelled(self):
        output = []
        session = self.session({0: [(.5, PRIMARY_PCM)], 1: [(.02, BACKUP_PCM[:20]), (.02, BACKUP_PCM[20:])]},
                               backup_tts_model='', backup_tts_voice_id='')
        session._stream_call('text', output.append, threading.Event())
        self.assertEqual(output, [BACKUP_PCM], '只交出一份，而且是完整的一整句')
        self.assertTrue(all(flag.is_set() for flag in self.flags.values()))
        self.assertFalse(session.took_backup('text'), '同模型同音色：可以照常写缓存')

    def test_other_voice_backup_yields_to_primary_within_grace(self):
        output = []
        session = self.session({0: [(.12, PRIMARY_PCM)], 1: [(.02, BACKUP_PCM)]})
        session._stream_call('text', output.append, threading.Event())
        self.assertEqual(output, [PRIMARY_PCM], '换了音色的第二通道先好了也礼让主通道一小会儿')
        output.clear()
        session = self.session({0: [(1.5, PRIMARY_PCM)], 1: [(.02, BACKUP_PCM)]})
        session._stream_call('text', output.append, threading.Event())
        self.assertEqual(output, [BACKUP_PCM])
        self.assertTrue(session.took_backup('text'), '另一个音色合成的：不写进主模型的磁盘缓存')
        self.assertFalse(session.took_backup('text'))

    def test_primary_failure_uses_backup(self):
        output = []
        self.session({0: [(.0, ApiError('empty'))], 1: [(.0, BACKUP_PCM)]},
                     backup_tts_mode='delayed')._stream_call('text', output.append, threading.Event())
        self.assertEqual(output, [BACKUP_PCM])

    def test_delayed_healthy_primary_never_launches_backup_even_when_slow_to_finish(self):
        """「稍后」按有没有音频出来判断：主通道一直在出音频，整句要 2.6 秒才合成完，也不启动第二通道。"""
        output = []
        chunks = [(.2, PRIMARY_PCM[i:i + 4]) for i in range(0, 52, 4)]      # 13 包 × 0.2 秒 = 2.6 秒
        session = self.session({0: chunks, 1: [(.0, BACKUP_PCM)]}, backup_tts_mode='delayed')
        session._stream_call('text', output.append, threading.Event())
        self.assertEqual(self.calls, [0])
        self.assertEqual(output, [PRIMARY_PCM[:52]])

    def test_delayed_silent_primary_launches_backup(self):
        output = []
        session = self.session({0: [(30, PRIMARY_PCM)], 1: [(.0, BACKUP_PCM)]}, backup_tts_mode='delayed')
        with patch('teams_voice_translator.defense.tts_session.TTS_BACKUP_DELAY', .1):
            session._stream_call('text', output.append, threading.Event())
        self.assertEqual((self.calls, output), ([0, 1], [BACKUP_PCM]))
        self.assertTrue(self.flags[0].is_set())

    def test_streaming_mode_first_packet_decides_and_never_switches(self):
        output = []
        session = self.session({0: [(.02, PRIMARY_PCM[:10]), (.1, ApiError('cut'))], 1: [(.3, BACKUP_PCM)]},
                               speech_mode='fast', backup_tts_model='', backup_tts_voice_id='')
        with self.assertRaisesRegex(ApiError, 'cut'):
            session._stream_call('text', output.append, threading.Event())
        self.assertEqual(output, [PRIMARY_PCM[:10]], '前半句已经播出去了：断了也不换第二通道重说')

    def test_both_fail_no_audio(self):
        output = []
        with self.assertRaisesRegex(ApiError, '两路合成都失败'):
            self.session({0: [(.0, ApiError('bad'))], 1: [(.0, ApiError('bad'))]})._stream_call(
                'text', output.append, threading.Event())
        self.assertEqual(output, [])

    def test_incomplete_pcm_never_wins(self):
        output = []
        self.session({0: [(.0, b'\x01\x00\x01')], 1: [(.05, BACKUP_PCM)]})._stream_call(
            'text', output.append, threading.Event())
        self.assertEqual(output, [BACKUP_PCM])

    def test_cancellation_no_audio(self):
        event = threading.Event(); event.set()
        with self.assertRaises(_Cancelled):
            self.session({0: [(.0, PRIMARY_PCM)], 1: [(.0, BACKUP_PCM)]})._stream_call('text', self.fail, event)

    def test_different_model_requires_its_voice(self):
        statuses, output = [], []
        session = self.session({0: [(.0, PRIMARY_PCM)], 1: [(.0, BACKUP_PCM)]}, backup_tts_voice_id='')
        session.on_status = statuses.append
        session._stream_call('text', output.append, threading.Event())
        session._stream_call('text', output.append, threading.Event())
        self.assertEqual(self.calls, [0, 0])
        self.assertEqual(self.channel_settings[0]['voice'], 'v1')
        self.assertEqual(len(statuses), 1, '只提示一次，不每句都提示')
        self.assertIn('绑定', statuses[0])
        self.assertEqual(output, [PRIMARY_PCM, PRIMARY_PCM])

    def test_same_model_may_reuse_voice_and_no_recursive_channel(self):
        session = self.session({0: [(.2, PRIMARY_PCM)], 1: [(.0, BACKUP_PCM)]},
                               backup_tts_model='', backup_tts_voice_id='')
        session._stream_call('text', lambda _: None, threading.Event())
        self.assertTrue(all(v['voice'] == 'v1' and not v['backup_tts_enabled']
                            for v in self.channel_settings.values()))

    def test_backup_without_ssml_reads_plain_text(self):
        ssml = '<speak>Hello.<break time="250ms"/>World.</speak>'
        session = self.session({0: [(.3, PRIMARY_PCM)], 1: [(.0, BACKUP_PCM)]},
                               tts_enable_ssml=True, backup_tts_model='qwen3-tts-vc-2026-01-22')
        session._stream_call(ssml, lambda _: None, threading.Event())
        self.assertEqual(self.texts[0], ssml)
        self.assertEqual(self.texts[1], 'Hello. World.')


class AsrFailoverTests(unittest.TestCase):
    def setup_session(self):
        self.sessions = []
        self.previews, self.segments, self.status, self.errors = [], [], [], []
        def factory(**kwargs):
            session = MagicMock(); session.callbacks = kwargs
            self.sessions.append(session)
            return session
        return StandbyASR(backup_model='backup', factory=factory, values={'asr_model':'primary'},
            on_preview=lambda *v: self.previews.append(v), on_segment=lambda *v: self.segments.append(v),
            on_status=self.status.append, on_error=self.errors.append)

    def test_standby_connects_only_on_fault_and_keeps_models_separate(self):
        session = self.setup_session(); session.start(); session.wait_ready()
        self.assertEqual(len(self.sessions), 1)
        self.sessions[0].callbacks['on_error']('failed')
        self.assertTrue(session._switch_done.wait(2))
        self.assertEqual(self.sessions[1].callbacks['values']['asr_model'], 'backup')
        self.assertTrue(self.sessions[0].close.called)
        session.close()

    def test_late_primary_callbacks_are_dropped(self):
        session = self.setup_session()
        self.sessions[0].callbacks['on_segment']('first', '')
        self.sessions[0].callbacks['on_error']('failed')
        self.assertTrue(session._switch_done.wait(2))
        self.sessions[0].callbacks['on_segment']('late', '')
        self.sessions[1].callbacks['on_segment']('next', '')
        self.assertEqual(self.segments, [('first',''), ('next','')])
        session.close()
        self.sessions[1].callbacks['on_preview']('after close', '')
        self.assertEqual(self.previews, [])

    def test_audio_not_replayed_to_backup(self):
        session = self.setup_session(); session.send_audio(b'old')
        self.sessions[0].callbacks['on_error']('failed')
        self.assertTrue(session._switch_done.wait(2))
        session.send_audio(b'new')
        self.sessions[1].send_audio.assert_called_once_with(b'new')
        session.close()

    def test_backup_failure_reported_without_third_channel(self):
        session = self.setup_session(); self.sessions[0].callbacks['on_error']('failed')
        self.assertTrue(session._switch_done.wait(2))
        self.sessions[1].callbacks['on_error']('backup failed')
        self.assertEqual(self.errors, ['backup failed'])
        self.assertEqual(len(self.sessions), 2)
        session.close()

    def test_backup_errors_do_not_flood(self):
        session = self.setup_session(); self.sessions[0].callbacks['on_error']('failed')
        self.assertTrue(session._switch_done.wait(2))
        self.sessions[1].send_audio.side_effect = ApiError('实时识别尚未连接')
        for _ in range(50):                      # 5 秒的麦克风音频
            session.send_audio(b'\0\0' * 1600)
        self.assertEqual(len(self.errors), 1, '备用通道也断了：只报一次，不刷屏')
        self.assertEqual(self.sessions[1].send_audio.call_count, 50, '继续喂音频，会自己重连的模型才连得回来')
        self.sessions[1].send_audio.side_effect = None
        session.send_audio(b'\0\0' * 1600)
        self.sessions[1].send_audio.side_effect = ApiError('又断了')
        session.send_audio(b'\0\0' * 1600)
        self.assertEqual(len(self.errors), 2, '恢复之后再出错要重新报')
        session.close()

    def test_stalled_primary_switches_only_with_voice_and_no_results(self):
        loud = (b'\x00\x20' * 1600)             # 0.1 秒、有声音
        quiet = b'\0\0' * 1600
        session = self.setup_session(); session.start(); session.wait_ready()
        self.assertEqual(session.current_model, 'primary')
        for _ in range(200):                     # 20 秒静音：不算卡住
            session.send_audio(quiet)
        for _ in range(50):                      # 5 秒人声 + 一直有预览：不算卡住
            session.send_audio(loud)
        self.sessions[0].callbacks['on_preview']('正在说', '')
        for _ in range(50):
            session.send_audio(loud)
        self.assertEqual(len(self.sessions), 1)
        for _ in range(20):                      # 累计 7 秒人声、没有任何结果 → 切
            session.send_audio(loud)
        self.assertTrue(session._switch_done.wait(2))
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual(session.current_model, 'backup')
        self.assertTrue(any('没有任何识别结果' in m or '备用' in m for m in self.status))
        session.close()

    def test_closed_primary_cannot_start_backup(self):
        session = self.setup_session(); session.close()
        self.sessions[0].callbacks['on_error']('failed')
        self.assertEqual(len(self.sessions), 1)

    def test_fun_asr_disconnect_status_triggers_backup_without_blocking_audio(self):
        session = self.setup_session()
        self.sessions[0].callbacks['on_status']('实时识别连接已断开；收到下一段音频时会自动重连')
        self.assertTrue(session._switch_done.wait(2))
        self.assertEqual(len(self.sessions), 2)
        session.close()

    def test_close_during_backup_connection_discards_late_results(self):
        session = self.setup_session()
        ready = threading.Event()
        original = session._factory
        def factory(**kwargs):
            result = original(**kwargs)
            result.wait_ready.side_effect = lambda *args: ready.wait(1)
            return result
        session._factory = factory
        self.sessions[0].callbacks['on_error']('failed')
        deadline = time.monotonic()+1
        while len(self.sessions)<2 and time.monotonic()<deadline: time.sleep(.005)
        session.close(); ready.set()
        self.assertTrue(session._switch_done.wait(2))
        self.sessions[1].callbacks['on_segment']('late', '')
        self.assertEqual(self.segments, [])
        self.assertTrue(self.sessions[1].close.called)


class IntegrationTests(unittest.TestCase):
    def test_backup_tts_never_touches_cache_keys_or_prerender(self):
        """合成第二通道只管现场发声：开了它，已经渲染好的讲稿 / 预案缓存照样命中，提前渲染也不走第二通道。"""
        from teams_voice_translator.defense.tts_cache import cache_key
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        with tempfile.TemporaryDirectory() as folder:
            settings = DefenseSettings(base_dir=Path(folder))
            settings.update({'tts_model': 'qwen-audio-3.1-tts-flash', 'tts_voice_id': 'v-flash',
                             'prerender_tts_model': 'qwen-audio-3.0-tts-plus', 'prerender_voice_id': 'v-plus'})
            engine = DefenseEngine(settings, EngineCallbacks())
            self.addCleanup(engine.shutdown)
            before = (cache_key('Hello.', engine.script_settings('en')), cache_key('Hello.', engine._tts_settings()))
            settings.update({'backup_tts_enabled': True, 'backup_tts_model': 'qwen-audio-3.0-tts-plus',
                             'backup_tts_voice_id': 'v-plus', 'backup_tts_mode': 'race'})
            after = (cache_key('Hello.', engine.script_settings('en')), cache_key('Hello.', engine._tts_settings()))
            self.assertEqual(before, after)
            self.assertNotIsInstance(create_tts_session(None, engine.script_settings('en'), on_audio=lambda _: None),
                                     TtsBackupSession)
            live = engine._live_tts_settings()
            self.assertEqual((live['backup_tts_model'], live['backup_tts_voice_id'], live['backup_tts_mode']),
                             ('qwen-audio-3.0-tts-plus', 'v-plus', 'race'))
            self.assertIsInstance(create_tts_session(None, live, on_audio=lambda _: None), TtsBackupSession)

    def test_other_voice_backup_audio_is_not_written_to_disk_cache(self):
        from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks
        with tempfile.TemporaryDirectory() as folder:
            settings = DefenseSettings(base_dir=Path(folder))
            settings.update({'tts_model': 'qwen-audio-3.1-tts-flash', 'tts_voice_id': 'v-flash'})
            engine = DefenseEngine(settings, EngineCallbacks())
            self.addCleanup(engine.shutdown)
            engine.disk_cache = MagicMock()
            for text, other in (('From backup.', True), ('From primary.', False)):
                engine.tts = MagicMock(); engine.tts.took_backup.return_value = other
                engine._live_utterance = {'text': text, 'zh': '', 'buf': bytearray(b'\x01\x00' * 10)}
                engine._on_utterance_done(text, 20)
                self.assertIn(text, engine._pcm_cache, '本场重播照用')
            self.assertEqual(engine.disk_cache.put.call_count, 1)

    def test_default_off_and_independent_audio_switches_saved(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import SettingsDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            settings = DefenseSettings(base_dir=Path(folder))
            dialog = SettingsDialog(None, settings)
            self.assertFalse(dialog.backup_asr_check.isChecked())
            self.assertFalse(dialog.backup_tts_check.isChecked())
            dialog.backup_asr_check.setChecked(True)
            dialog.backup_tts_check.setChecked(True)
            self.assertFalse(dialog.backup_check.isChecked())
            self.assertTrue(dialog.backup_asr_combo.isEnabled())
            self.assertTrue(dialog.backup_tts_voice_combo.isEnabled())
            with patch('teams_voice_translator.defense.ui.QMessageBox'):
                dialog._save()
            settings.load()
            self.assertTrue(settings.get('backup_asr_enabled'))
            self.assertTrue(settings.get('backup_tts_enabled'))
            self.assertFalse(settings.get('backup_channel_enabled'))
            self.assertEqual(settings.get('backup_tts_mode'), 'delayed')
            self.assertTrue(dialog.backup_tts_mode_combo.isEnabled())
            self.assertFalse(dialog.backup_mode_combo.isEnabled(), '大模型的模式和合成的模式分开')
            self.assertNotIn('backup_tts_enabled', speech_settings(settings))
            dialog.close()

    def test_llm_backup_can_change_only_the_model(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense import ui
        from teams_voice_translator.defense.channels import backup_channel, primary_endpoint
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            settings = DefenseSettings(base_dir=Path(folder))
            dialog = ui.SettingsDialog(None, settings)
            dialog.workspace_edit.setText('ws1')
            dialog.backup_check.setChecked(True)
            dialog.backup_route_combo.setCurrentIndex(dialog.backup_route_combo.findData('primary-model'))
            self.assertFalse(dialog.backup_key_edit.isEnabled(), '只换模型用同一个 Key')
            combo = dialog.backup_model_edit
            combo.setCurrentIndex(combo.findData('qwen3.7-flash'))
            self.assertEqual(combo.text(), 'qwen3.7-flash')
            self.assertIn('qwen3.7-flash', dialog.channel_summary_label.text())
            combo.setEditText('my-custom-model')
            self.assertEqual(combo.text(), 'my-custom-model')
            combo.setCurrentIndex(combo.findData('qwen3.7-flash'))
            with patch('teams_voice_translator.defense.ui.QMessageBox'):
                dialog._save()
            self.assertEqual((settings.get('backup_route'), settings.get('backup_model')),
                             ('primary-model', 'qwen3.7-flash'))
            with patch.object(type(settings), 'get_api_key', return_value='main-key'), \
                    patch.object(type(settings), 'workspace_id', new='ws1', create=True):
                channel = backup_channel(settings)
            self.assertEqual((channel.endpoint, channel.api_key, channel.model),
                             (primary_endpoint('ws1'), 'main-key', 'qwen3.7-flash'))
            dialog.close()

    def test_partial_failed_audio_is_rejected(self):
        from teams_voice_translator.defense.tts_session import synthesize_once
        def factory(client, settings, **kwargs):
            session = MagicMock()
            def speak(text):
                kwargs['on_audio'](b'\0\0')
                kwargs['on_error']('connection cut')
                kwargs['on_utterance_failed'](text)
                return True
            session.speak.side_effect = speak
            return session
        with patch('teams_voice_translator.defense.tts_session.create_tts_session', side_effect=factory):
            with self.assertRaisesRegex(ApiError, 'connection cut'):
                synthesize_once(None, {}, 'hello')


class LlmBoundaryTests(unittest.TestCase):
    def translator(self, **kwargs):
        from teams_voice_translator.defense.translator import ContextTranslator
        from teams_voice_translator.defense.channels import BackupChannel
        return ContextTranslator(api_key='fake', workspace_id='fake', model='qwen-plus',
            backup=BackupChannel(endpoint='backup', api_key='fake', label='test', **kwargs))

    def test_whitespace_does_not_win(self):
        translator = self.translator()
        def fake(endpoint, key, payload, on_piece, cancel_event, **kwargs):
            if endpoint != 'backup':
                on_piece(' ', ' ')
                cancel_event.wait(.2)
                raise ApiError('cancelled')
            time.sleep(.02)
            on_piece('actual', 'actual')
            return 'actual'
        translator._stream_once = fake
        got = []
        self.assertEqual(translator._post_stream({'model':'qwen-plus'}, lambda p,s:got.append(p), None), 'actual')
        self.assertEqual(got, ['actual'])

    def test_empty_primary_launches_delayed_backup_immediately(self):
        translator = self.translator(mode='delayed', delay=10)
        translator._stream_once = lambda endpoint,*args,**kwargs: 'backup' if endpoint == 'backup' else ''
        began = time.monotonic()
        self.assertEqual(translator._post_stream({'model':'qwen-plus'}, None, None), 'backup')
        self.assertLess(time.monotonic()-began, .5)

    def test_callback_failure_cancels_both(self):
        translator = self.translator()
        flags = []
        def fake(endpoint, key, payload, on_piece, cancel_event, **kwargs):
            flags.append(cancel_event)
            on_piece('text', 'text')
            return 'text'
        translator._stream_once = fake
        def broken(*args): raise RuntimeError('callback failed')
        with self.assertRaisesRegex(RuntimeError, 'callback failed'):
            translator._post_stream({'model':'qwen-plus'}, broken, None)
        self.assertTrue(all(flag.is_set() for flag in flags))

    def test_coordinator_timeout_even_when_workers_stall(self):
        translator = self.translator()
        translator._stream_once = lambda *args,**kwargs: (time.sleep(.2) or 'late')
        with self.assertRaisesRegex(ApiError, '超时'):
            translator._post_stream({'model':'qwen-plus'}, None, None, deadline_seconds=.03)

    def test_model_change_rebuilds_thinking_fields(self):
        translator = self.translator(model='qwen-plus')
        payloads = []
        def fake(endpoint, key, payload, on_piece, cancel_event, **kwargs):
            payloads.append((endpoint,payload))
            if endpoint != 'backup':
                cancel_event.wait(.1)
                raise ApiError('primary failed')
            return 'answer'
        translator._stream_once = fake
        translator._post_stream({'model':'qwen3.8-flash', 'enable_thinking':False, 'thinking_budget':100}, None, None)
        backup = next(p for endpoint,p in payloads if endpoint == 'backup')
        self.assertNotIn('thinking_budget', backup)

    def test_official_region_urls_and_missing_workspace(self):
        from teams_voice_translator.defense.channels import backup_endpoint
        self.assertIn('cn-hongkong.dashscope.aliyuncs.com', backup_endpoint('cn-hongkong'))
        self.assertIn('dashscope-us.aliyuncs.com', backup_endpoint('us-east-1'))
        with self.assertRaises(ValueError): backup_endpoint('ap-northeast-1')
        self.assertIn('ws.ap-northeast-1.maas.aliyuncs.com', backup_endpoint('ap-northeast-1','ws'))
