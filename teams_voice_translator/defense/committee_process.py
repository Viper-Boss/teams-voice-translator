"""Independent, recoverable committee ASR process; no microphone/TTS state is shared."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time


def _recognize(config, audio, events, stop, factory=None):
    from .asr_standby import create_defense_asr

    def emit(kind, *args):
        try:
            events.put((kind, args), timeout=.5)
        except queue.Full:
            # Preview loss is acceptable; the parent consumes finals independently.
            if kind != 'preview':
                raise RuntimeError('评委识别事件队列已满')

    attempt = 0
    while not stop.is_set():
        session = None
        fault = threading.Event()
        active = threading.Event()
        active.set()

        def guarded(kind):
            def callback(*args, active=active):
                if active.is_set() and not stop.is_set():
                    emit(kind, *args)
            return callback

        def error(message, active=active, fault=fault):
            if active.is_set() and not stop.is_set():
                emit('status', f'识别连接异常，正在自动重连：{message}')
                fault.set()

        def status(message, active=active, fault=fault):
            if active.is_set() and not stop.is_set():
                emit('status', message)
                if str(message).startswith(('实时识别连接异常：', '实时识别连接已断开；')):
                    fault.set()

        try:
            kwargs = dict(config['kwargs'])
            kwargs.update(on_preview=guarded('preview'), on_segment=guarded('segment'),
                          on_status=status, on_error=error)
            create = factory or (lambda **kw: create_defense_asr(config['backup'], **kw))
            session = create(**kwargs)
            session.start()
            session.wait_ready()
            if fault.is_set():
                raise RuntimeError('评委识别连接未就绪')
            emit('ready', os.getpid())
            emit('status', '识别已连接（独立进程）' if not attempt else '识别已自动重连（独立进程）')
            attempt = 0
            while not stop.is_set() and not fault.is_set():
                try:
                    pcm = audio.get(timeout=.1)
                except queue.Empty:
                    continue
                session.send_audio(pcm)
                model = getattr(session, 'current_model', '')
                if not isinstance(model, str) or not model:
                    model = str(config['kwargs']['values'].get('asr_model', ''))
                emit('usage', model, len(pcm) / 32000)
        except Exception as exc:
            if not stop.is_set():
                emit('status', f'识别暂时中断，将自动重连：{exc}')
        finally:
            active.clear()  # Discard callbacks from the old connection before closing it.
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
        # Never replay consumed audio or feed seconds of stale speech after a reconnect.
        while True:
            try:
                audio.get_nowait()
            except queue.Empty:
                break
        attempt += 1
        stop.wait(min(5.0, .5 * (2 ** min(attempt - 1, 4))))


def _offline_probe(config, audio, events, stop):
    """Packaged multiprocessing smoke test. Never opens a device or network."""
    events.put(('ready', (os.getpid(),)))
    while not stop.is_set():
        try:
            pcm = audio.get(timeout=.1)
        except queue.Empty:
            continue
        events.put(('segment', (f'offline:{len(pcm)}', 'neutral')))


class CommitteeASRProcess:
    def __init__(self, *, backup=None, worker_target=None, **kwargs):
        self._callbacks = {key: kwargs.pop(key, None) for key in
                           ('on_preview', 'on_segment', 'on_status', 'on_error', 'on_usage')}
        self._config = {'kwargs': kwargs, 'backup': dict(backup or {})}
        self._target = worker_target or _recognize
        self._ctx = mp.get_context('spawn')
        self._audio = self._ctx.Queue(maxsize=32)
        self._events = self._ctx.Queue(maxsize=128)
        self._stop = self._ctx.Event()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._process = None
        self._reader = None
        self.pid = None

    def start(self):
        self._process = self._ctx.Process(target=self._target,
                                         args=(self._config, self._audio, self._events, self._stop),
                                         name='defense-committee-asr', daemon=True)
        self._process.start()
        self._reader = threading.Thread(target=self._read_events, name='committee-events', daemon=True)
        self._reader.start()

    def _read_events(self):
        while not self._closed.is_set():
            try:
                kind, args = self._events.get(timeout=.2)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    self._dispatch('on_error', '独立识别进程已退出，请重新开始答辩')
                    return
                continue
            except (EOFError, OSError, ValueError):
                return
            if self._closed.is_set():
                return
            if kind == 'ready':
                self.pid = args[0]
                self._ready.set()
            else:
                self._dispatch('on_' + kind, *args)

    def _dispatch(self, key, *args):
        callback = self._callbacks.get(key)
        if callback and not self._closed.is_set():
            try:
                callback(*args)
            except Exception:
                pass  # A UI callback cannot kill the independent recognition channel.

    def wait_ready(self, timeout=20.0):
        if not self._ready.wait(timeout) or self._closed.is_set():
            raise RuntimeError('评委识别独立进程连接超时，请检查识别模型、账号与网络')

    def send_audio(self, pcm):
        if self._closed.is_set():
            return
        try:
            self._audio.put_nowait(pcm)
        except queue.Full:
            try:
                self._audio.get_nowait()
                self._audio.put_nowait(pcm)
            except (queue.Empty, queue.Full):
                pass

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self._stop.set()
        process = self._process
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=.5)
        for channel in (self._audio, self._events):
            channel.cancel_join_thread()
            channel.close()


def offline_process_smoke(output):
    import json
    from pathlib import Path
    received = []
    listener = CommitteeASRProcess(worker_target=_offline_probe, on_segment=lambda *args: received.append(args))
    try:
        listener.start()
        listener.wait_ready(15)
        listener.send_audio(b'\x01\x00' * 160)
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            time.sleep(.02)
        result = {'parent_pid': os.getpid(), 'child_pid': listener.pid,
                  'separate_process': listener.pid != os.getpid(),
                  'received': received, 'network_calls': False}
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'committee-process.json').write_text(json.dumps(result), encoding='utf-8')
        return 0 if received and result['separate_process'] else 1
    finally:
        listener.close()
