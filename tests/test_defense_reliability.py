from __future__ import annotations
import base64
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from teams_voice_translator.aliyun import ApiError, BailianClient
from teams_voice_translator.defense.tts_session import TtsHttpSession, TtsSession
from teams_voice_translator.defense.pipeline import DefenseEngine, EngineCallbacks, _Job
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.speech_policy import speech_settings, speech_chunks, NATURAL_INSTRUCTION
from teams_voice_translator.defense.translator import ContextTranslator, SentenceSplitter
from teams_voice_translator.defense.memory import MeetingMemory
from teams_voice_translator.audio import MultiOutputPlayer
from test_defense_tts_session import FakeWs, good_plan
from test_defense_translator import FakeStreamResponse, sse_lines

class SpeechReliabilityTests(unittest.TestCase):
    def session(self, function, *, natural=False):
        client=Mock()
        client._stream_qwen3_tts_http.side_effect=function
        received=[];errors=[]
        session=TtsHttpSession(client,{'tts_model':'qwen3-tts-vc-2026-01-22','voice':'test',
             'speech_mode':'natural' if natural else 'streaming'},on_audio=received.append,on_error=errors.append)
        self.addCleanup(session.close)
        return session,received,errors,client

    def test_empty_audio_is_failure(self):
        s,a,e,c=self.session(lambda *args:None)
        s.start();s.speak('empty');self.assertTrue(s.wait_until_idle(2))
        self.assertEqual(len(e),1);self.assertFalse(a)

    def test_partial_stream_is_never_replayed(self):
        def stream(text,settings,on_audio,cancel):
            on_audio(b'12');raise ApiError('connection lost')
        s,a,e,c=self.session(stream)
        s.start();s.speak('partial');self.assertTrue(s.wait_until_idle(2))
        self.assertEqual(c._stream_qwen3_tts_http.call_count,1)
        self.assertEqual(a,[b'12']);self.assertEqual(len(e),1)

    def test_natural_mode_retries_without_delivering_failed_audio(self):
        calls=[]
        def stream(text,settings,on_audio,cancel):
            calls.append(text);on_audio(b'12')
            if len(calls)==1:raise ApiError('connection lost')
            on_audio(b'34')
        s,a,e,c=self.session(stream,natural=True)
        s.start();s.speak('whole');self.assertTrue(s.wait_until_idle(2))
        self.assertEqual(a,[b'1234']);self.assertFalse(e);self.assertEqual(len(calls),2)

    def test_natural_buffer_waits_for_complete_audio(self):
        started=threading.Event();release=threading.Event()
        self.addCleanup(release.set)
        def stream(text,settings,on_audio,cancel):
            on_audio(b'12');started.set();release.wait(2);on_audio(b'34')
        s,a,e,c=self.session(stream,natural=True)
        s.start();s.speak('whole');self.assertTrue(started.wait(1))
        self.assertFalse(a);self.assertFalse(s.wait_until_idle(.02))
        release.set();self.assertTrue(s.wait_until_idle(2));self.assertEqual(a,[b'1234'])

    def test_interrupt_signals_inflight_http_and_next_sentence_survives(self):
        started=threading.Event();calls=[]
        def stream(text,settings,on_audio,cancel):
            calls.append(text)
            if text=='old':
                started.set();self.assertTrue(cancel.wait(2))
                raise ApiError('cancelled read')
            on_audio(b'34')
        s,a,e,c=self.session(stream)
        s.start();s.speak('old');self.assertTrue(started.wait(1));s.speak('queued')
        s.interrupt();s.speak('new');self.assertTrue(s.wait_until_idle(3))
        self.assertEqual(calls,['old','new']);self.assertEqual(a,[b'34']);self.assertFalse(e)

    def test_close_rejects_new_work_and_preserves_live_worker(self):
        started=threading.Event();release=threading.Event()
        def stream(*args):started.set();release.wait(2)
        s,a,e,c=self.session(stream);s.start();s.speak('old');self.assertTrue(started.wait(1))
        thread=s._thread;s.close(timeout=.01)
        try:
            self.assertIs(s._thread,thread);self.assertTrue(thread.is_alive());self.assertFalse(s.speak('new'))
            s.start();self.assertIs(s._thread,thread)
        finally:release.set();thread.join(2)
        self.assertFalse(a)

    def test_initial_websocket_failure_can_recover(self):
        received=[];errors=[]
        with patch('teams_voice_translator.defense.tts_session.websocket.create_connection',
                   side_effect=[OSError('offline'),FakeWs(good_plan)]):
            s=TtsSession(BailianClient('test','workspace'),{'tts_model':'qwen3-tts-vc-realtime-2026-01-15','voice':'test'},on_audio=received.append,on_error=errors.append)
            s.start();s.speak('recover');self.assertTrue(s.wait_until_idle(3));s.close()
        self.assertTrue(received);self.assertFalse(errors)

    def test_later_sentence_recovers_after_two_connection_failures(self):
        received=[];errors=[]
        with patch('teams_voice_translator.defense.tts_session.websocket.create_connection',
                   side_effect=[OSError('offline'),OSError('offline'),FakeWs(good_plan)]):
            s=TtsSession(BailianClient('test','workspace'),{'tts_model':'qwen3-tts-vc-realtime-2026-01-15','voice':'test'},on_audio=received.append,on_error=errors.append)
            s.start();s.speak('fail');self.assertTrue(s.wait_until_idle(3))
            s.speak('recover');self.assertTrue(s.wait_until_idle(3));s.close()
        self.assertEqual(len(errors),1);self.assertTrue(received)

    def test_slow_audio_callback_keeps_session_busy(self):
        started=threading.Event();release=threading.Event()
        s,a,e,c=self.session(lambda t,v,cb,token:cb(b'12'))
        def play(pcm):started.set();release.wait(2)
        s.on_audio=play;s.start();s.speak('hello');self.assertTrue(started.wait(1))
        try:self.assertFalse(s.wait_until_idle(.01))
        finally:release.set()
        self.assertTrue(s.wait_until_idle(2))

    def test_realtime_eof_fails_immediately(self):
        s=TtsSession(BailianClient('test','ws'),{'voice':'v'},on_audio=lambda b:None)
        s._ws=Mock();s._ws.recv.return_value=''
        with self.assertRaisesRegex(ApiError,'断开'):s._receive()
        s.close()

    def test_failed_output_stream_is_never_written_again_and_is_closed(self):
        good=Mock();dead=Mock();dead.write.side_effect=RuntimeError('device gone')
        player=MultiOutputPlayer([],24000);player.streams=[dead,good]
        with self.assertRaises(RuntimeError):player.write(b'12')
        player.write(b'34');player.close();player.close()
        self.assertEqual(dead.write.call_count,1);dead.close.assert_called_once()
        good.close.assert_called_once()
        with self.assertRaises(RuntimeError):player.write(b'56')

    def test_start_failure_closes_partially_opened_stream(self):
        stream=Mock();stream.start.side_effect=RuntimeError('cannot start')
        with patch('teams_voice_translator.audio.sd.RawOutputStream',return_value=stream):
            with self.assertRaises(RuntimeError):MultiOutputPlayer([1],24000).__enter__()
        stream.close.assert_called_once()

class EngineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.settings=DefenseSettings(Path(self.tmp.name));self.errors=[]
        self.engine=DefenseEngine(self.settings,EngineCallbacks(on_error=self.errors.append))
        self.addCleanup(self.engine.shutdown)

    def test_natural_translates_whole_answer_before_speaking(self):
        spoken=[]
        def translate(memory,direction,text,on_delta,cancel_event):
            on_delta('Thank you. ','Thank you. ');self.assertFalse(spoken)
            on_delta('Let me explain.','Thank you. Let me explain.');self.assertFalse(spoken)
            return 'Thank you. Let me explain.'
        self.engine.translator=Mock();self.engine.translator.translate.side_effect=translate
        self.engine.tts=Mock();self.engine.tts.speak_batch.side_effect=lambda texts:spoken.extend(texts) or True
        self.engine._translate_my_job(_Job(1,'谢谢，我解释一下',time.monotonic()))
        self.assertEqual(spoken,['Thank you. Let me explain.'])
        self.assertEqual(len(self.engine.memory.snapshot_turns()),1)

    def test_cancel_during_translation_cannot_enqueue_new_speech(self):
        job=_Job(1,'old',time.monotonic());self.engine._active_job=job
        def translate(memory,direction,text,on_delta,cancel_event):
            self.engine.interrupt_speech();return 'late answer'
        self.engine.translator=Mock();self.engine.translator.translate.side_effect=translate
        self.engine.tts=Mock();self.engine._translate_my_job(job)
        self.engine.tts.speak_batch.assert_not_called();self.assertFalse(self.engine.memory.snapshot_turns())

    def test_interrupt_cancels_active_and_queued_translation(self):
        first=_Job(1,'one',time.monotonic());queued=_Job(2,'two',time.monotonic())
        self.engine._active_job=first;self.engine._my_queue.put(queued)
        self.engine.interrupt_speech()
        self.assertTrue(first.cancel.is_set());self.assertTrue(queued.cancel.is_set());self.assertTrue(self.engine._my_queue.empty())

    def test_repeated_start_does_not_duplicate_workers(self):
        self.engine.start();original=list(self.engine._workers);self.engine.start()
        self.assertEqual(self.engine._workers,original)

    def test_unexpected_translation_exception_does_not_kill_worker(self):
        done=threading.Event()
        def process(job):
            if job.seg_id==1:raise ValueError('bad response')
            done.set()
        self.engine.translator=Mock();self.engine._translate_my_job=process
        self.engine.start();self.engine._my_queue.put(_Job(1,'bad',time.monotonic()))
        self.engine._my_queue.put(_Job(2,'good',time.monotonic()))
        self.assertTrue(done.wait(2));self.assertTrue(self.errors)

    def test_stop_direct_does_not_restart_microphone(self):
        self.engine._direct_active=True;self.engine._resume_listening_after_direct=True
        self.engine.bridge=Mock();self.engine._do_toggle_listening=Mock()
        self.engine._do_stop_defense()
        self.engine._do_toggle_listening.assert_not_called();self.assertEqual(self.engine.state,'idle')

    def test_failed_player_initialization_leaves_no_player(self):
        player=Mock();player.__enter__=Mock(side_effect=RuntimeError('device missing'))
        with patch('teams_voice_translator.defense.pipeline.MultiOutputPlayer',return_value=player):
            with self.assertRaises(RuntimeError):self.engine._ensure_player()
        self.assertIsNone(self.engine.player)

    def test_voice_metadata_survives_translation_finishing_before_audio(self):
        job=_Job(1,'中文原句',time.monotonic());self.engine._speech_jobs['English.']=job
        self.engine._active_job=None;self.engine._on_first_audio('English.')
        self.assertEqual(self.engine._live_utterance['zh'],'中文原句')
        self.assertIsNotNone(job.first_audio_at)

    def test_speech_policy_uses_supported_model_only_for_automatic_instruction(self):
        self.settings.update({'tts_model':'cosyvoice-v3.5-plus','tts_instruction':''})
        self.assertEqual(speech_settings(self.settings)['tts_instruction'],NATURAL_INSTRUCTION)
        self.settings.set('tts_model','qwen3-tts-vc-2026-01-22')
        self.assertEqual(speech_settings(self.settings)['tts_instruction'],'')
        self.assertLessEqual(len(NATURAL_INSTRUCTION),100)

    def test_long_text_preserves_words_and_bounds_requests(self):
        text=' '.join(['We measured a depth of 3.14 mm in Fig. 2.']*40)
        chunks=speech_chunks(text)
        self.assertEqual(' '.join(chunks),text);self.assertTrue(all(len(c)<=450 for c in chunks))

class TranslationRegressionTests(unittest.TestCase):
    def translate(self,lines):
        with patch('teams_voice_translator.defense.translator.requests.post',return_value=FakeStreamResponse(lines)):
            return ContextTranslator(api_key='test',workspace_id='test').translate(MeetingMemory(),'zh2en','你好')

    def test_truncated_network_response_is_not_a_complete_translation(self):
        with self.assertRaisesRegex(ApiError,'不完整'):self.translate(sse_lines(['partial'])[:-1])

    def test_token_limit_response_is_not_spoken_as_complete(self):
        lines=sse_lines(['partial'])[:-1]+['data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'length'}]})]
        with self.assertRaisesRegex(ApiError,'截断'):self.translate(lines)

    def test_api_error_envelope_is_reported(self):
        with self.assertRaisesRegex(ApiError,'错误'):self.translate(['data: '+json.dumps({'error':{'message':'unavailable'}})])

    def test_abbreviations_are_not_split_before_number(self):
        splitter=SentenceSplitter()
        result=splitter.feed('As shown in Fig. 2, the depth is 3.14 mm. Next result.')+splitter.flush()
        self.assertEqual(result,['As shown in Fig. 2, the depth is 3.14 mm.','Next result.'])

    def test_legacy_response_closes_when_playback_throws(self):
        response=Mock(ok=True)
        response.iter_lines.return_value=['data: '+json.dumps({'output':{'audio':{'data':base64.b64encode(b'12').decode()}}})]
        settings={'tts_model':'cosyvoice-v3.5-plus','voice':'v','tts_sample_rate':24000,'tts_volume':55,'tts_rate':1.,'tts_pitch':1.,'tts_seed':0,'tts_language_hint':'en'}
        with patch('teams_voice_translator.aliyun.requests.post',return_value=response):
            with self.assertRaises(RuntimeError):
                BailianClient('test','test')._stream_legacy_tts('text',settings,Mock(side_effect=RuntimeError('device failed')),threading.Event())
        response.close.assert_called_once()


class FinalReliabilityTests(unittest.TestCase):
    def test_esc_cancels_speech_commands_waiting_in_command_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=DefenseEngine(DefenseSettings(Path(tmp)),EngineCallbacks())
            pending=[]
            engine._submit=lambda action,description:pending.append(action)
            engine._ensure_voice=Mock()
            engine.tts=Mock()
            engine.send_text('old text',translate=False)
            engine.replay_english('old replay')
            engine.speak_as_me('old answer')
            engine.interrupt_speech()
            for action in pending:action()
            engine._ensure_voice.assert_not_called()
            engine.tts.speak.assert_not_called()
            engine.shutdown()

    def test_stop_failure_still_releases_remaining_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=DefenseEngine(DefenseSettings(Path(tmp)),EngineCallbacks())
            mic=Mock();mic.stop.side_effect=RuntimeError('unplugged')
            tts=Mock();player=Mock()
            engine.mic_capture=mic;engine.tts=tts;engine.player=player
            engine._listening=True;engine._session_active=True
            engine._do_stop_defense()
            tts.close.assert_called_once();player.close.assert_called_once()
            self.assertIsNone(engine.player);self.assertIsNone(engine.mic_capture)
            self.assertFalse(engine._listening);self.assertEqual(engine.state,'idle')
            engine.shutdown()

    def test_natural_caption_duration_is_known_before_playback(self):
        events=[];client=Mock()
        client._stream_qwen3_tts_http.side_effect=lambda t,s,cb,c:cb(b'1234')
        session=TtsHttpSession(client,{'voice':'test','speech_mode':'natural'},
            on_first_audio=lambda text:events.append('start'),
            on_audio_ready=lambda text,n:events.append(('duration',n)),
            on_audio=lambda pcm:events.append('audio'))
        session.start();session.speak('test');self.assertTrue(session.wait_until_idle(2));session.close()
        self.assertEqual(events,['start',('duration',4),'audio'])

    def test_voice_probe_reports_synthesis_failure(self):
        from teams_voice_translator.defense.ui import voice_probe
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            settings=DefenseSettings(Path(tmp));player=MagicMock();client=Mock()
            client._stream_legacy_tts.side_effect=ApiError('bad voice')
            with patch('teams_voice_translator.defense.ui.BailianClient',return_value=client), patch('teams_voice_translator.audio.MultiOutputPlayer',return_value=player):
                with self.assertRaisesRegex(ApiError,'bad voice'):
                    voice_probe(settings,'workspace','key','voice','cosyvoice-v3.5-plus',timeout=3)
            # Failed synthesis must not open a sound device at all.
            player.__enter__.assert_not_called()

    def test_voice_probe_uses_selected_volume_and_default_monitor(self):
        from teams_voice_translator.defense.ui import voice_probe
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            settings=DefenseSettings(Path(tmp));settings.set('tts_volume',72)
            player=MagicMock();client=Mock()
            client._stream_legacy_tts.side_effect=lambda t,s,cb,c:cb(b'1234')
            with patch('teams_voice_translator.defense.ui.BailianClient',return_value=client), patch('teams_voice_translator.audio.MultiOutputPlayer',return_value=player) as factory:
                voice_probe(settings,'workspace','key','voice','cosyvoice-v3.5-plus',timeout=3)
            factory.assert_called_once_with([None],24000)
            self.assertEqual(client._stream_legacy_tts.call_args.args[1]['tts_volume'],72)

    def test_settings_concurrent_writes_remain_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=DefenseSettings(Path(tmp));errors=[]
            def update(value):
                try:
                    for _ in range(15):settings.set('tts_volume',value)
                except Exception as exc:errors.append(exc)
            threads=[threading.Thread(target=update,args=(n,)) for n in (50,60,70)]
            for t in threads:t.start()
            for t in threads:t.join()
            self.assertFalse(errors)
            self.assertIn(DefenseSettings(Path(tmp)).get('tts_volume'),(50,60,70))

    def test_silent_sample_rejected_before_upload(self):
        import wave
        from teams_voice_translator.voice_sample import inspect_voice_sample,VoiceSampleError
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.wav'
            with wave.open(str(path),'wb') as handle:
                handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(24000)
                handle.writeframes(bytes(24000*2))
            with self.assertRaises(VoiceSampleError):inspect_voice_sample(path)

    def test_qwen_snapshots_must_not_share_a_voice_binding(self):
        from PySide6.QtWidgets import QApplication
        from teams_voice_translator.defense.ui import SettingsDialog
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            settings=DefenseSettings(Path(tmp))
            settings.update({'tts_voice_id':'voice','tts_model':'qwen3-tts-vc-realtime-2026-01-15',
               'voice_library':[{'voice_id':'voice','target_model':'qwen3-tts-vc-2026-01-22'}]})
            dialog=SettingsDialog(None,settings)
            with patch('teams_voice_translator.defense.ui.QMessageBox.warning') as warning:
                dialog._save()
            warning.assert_called_once();self.assertNotEqual(dialog.result(),1)
            dialog.close()

if __name__=='__main__':unittest.main()
