"""Fault-triggered ASR model failover. Never replay already consumed audio.

主通道报错 / 断线 / 卡住（一直有人声却迟迟没有任何识别结果）时一次性切到备用模型，本次聆听不再切回。
切换期间的音频直接丢掉、不补发，所以同一句话不会被识别两次。
"""
from __future__ import annotations

import threading
import time
from array import array

STALL_VOICED_SECONDS = 6.0     # 累计这么久的人声、主通道却没有任何预览 / 整句 → 判定卡住
VOICED_LEVEL = 500             # 16 位 PCM 平均绝对值，约 -36 dBFS；低于它当静音，不计时
REPORT_INTERVAL = 30.0         # 备用通道也出错时，同一类错误最多这么久报一次，不刷屏


def _voiced(pcm: bytes) -> bool:
    try:
        samples = array("h")
        samples.frombytes(pcm[:len(pcm) // 2 * 2])
        if not samples:
            return False
        step = max(1, len(samples) // 400)          # 抽样就够了，别在采集线程里做重活
        picked = samples[::step]
        return sum(abs(v) for v in picked) / len(picked) >= VOICED_LEVEL
    except Exception:  # noqa: BLE001
        return False

from ..aliyun import ApiError, create_realtime_asr


class StandbyASR:
    def __init__(self, *, backup_model, factory=create_realtime_asr, **kwargs):
        self._factory, self._kwargs = factory, kwargs
        self._backup_model = backup_model
        self._lock = threading.RLock()
        self._closed = False
        self._switching = False
        self._finishing = False
        self._index = 0
        self._sessions = []
        self._current = self._make(0)
        self._switch_done = threading.Event()
        self._switch_done.set()
        self._voiced_seconds = 0.0          # 上次有识别结果以来累计的人声时长（只看主通道）
        self._last_report = 0.0

    def _make(self, index):
        kwargs = dict(self._kwargs)
        kwargs['values'] = dict(kwargs['values'])
        if index:
            kwargs['values']['asr_model'] = self._backup_model
        for name in ('on_preview', 'on_segment', 'on_status'):
            callback = kwargs.get(name)
            if callback:
                def guarded(*args, callback=callback, activity=name != 'on_status'):
                    with self._lock:
                        if not self._closed and self._index == index:
                            if activity:
                                self._voiced_seconds = 0.0   # 有预览 / 整句 = 主通道还活着
                            callback(*args)
                kwargs[name] = guarded
        status = kwargs.get('on_status')
        if status:
            def on_status(message):
                status(message)
                if str(message).startswith(('实时识别连接异常：', '实时识别连接已断开；')):
                    self._error(index, message)
            kwargs['on_status'] = on_status
        kwargs['on_error'] = lambda message: self._error(index, message)
        session = self._factory(**kwargs)
        self._sessions.append(session)
        return session

    def _error(self, index, message):
        with self._lock:
            if self._closed or self._finishing or index != self._index or self._switching:
                return
            if index == 1:
                # 备用通道也出错：没有第三个通道了，照常报错——但同样的错不要每 0.1 秒报一次
                now = time.monotonic()
                if self._last_report and now - self._last_report < REPORT_INTERVAL:
                    return
                self._last_report = now
                self._kwargs['on_error'](message)
                return
            self._switching = True
            self._index = 1  # Suppress late callbacks from the failed primary.
            self._switch_done.clear()
        self._kwargs['on_status']('识别主通道异常，正在连接备用模型；请重说尚未识别完成的句子')
        threading.Thread(target=self._switch, name='asr-failover', daemon=True).start()

    def _switch(self):
        try:
            self._sessions[0].close()
            with self._lock:
                if self._closed:
                    return
                session = self._make(1)
                self._current = session
                session.start()
            session.wait_ready()
            with self._lock:
                closed = self._closed
            if closed:
                session.close()
            else:
                self._kwargs['on_status']('已切换备用识别模型；请重说刚才未完成的句子')
        except Exception as exc:
            if not self._closed:
                self._kwargs['on_error'](f'备用识别连接失败：{exc}')
        finally:
            with self._lock:
                self._switching = False
            self._switch_done.set()

    def start(self):
        try:
            with self._lock:
                if self._closed:
                    raise ApiError('已取消')
                self._current.start()
        except Exception as exc:
            self._error(0, str(exc))

    def wait_ready(self, timeout=12.0):
        with self._lock:
            session = self._current
        try:
            session.wait_ready(timeout)
        except Exception as exc:
            self._error(0, str(exc))
        if not self._switch_done.wait(timeout):
            raise ApiError('备用识别连接超时')
        with self._lock:
            if self._closed:
                raise ApiError('已取消')
            session = self._current
        session.wait_ready(timeout)

    @property
    def current_model(self):
        """现在真正在用（在计费）的识别模型。"""
        with self._lock:
            return self._backup_model if self._index else str(self._kwargs['values'].get('asr_model', ''))

    def send_audio(self, pcm):
        with self._lock:
            if self._closed or self._switching:
                return
            session, index = self._current, self._index
            stalled = False
            if index == 0 and _voiced(pcm):
                self._voiced_seconds += len(pcm) / 32000
                stalled = self._voiced_seconds >= STALL_VOICED_SECONDS
        if stalled:
            self._error(0, f'主通道 {STALL_VOICED_SECONDS:.0f} 秒有人声却没有任何识别结果')
            return
        try:
            opened = getattr(session, 'opened', None)
            # 备用通道（FunASR 一类会自己重连的）要继续喂音频才连得回来，所以只对主通道做这个预检
            if index == 0 and opened is not None and not opened.is_set():
                raise ApiError('实时识别连接已断开')
            session.send_audio(pcm)
            if index == 1:
                self._last_report = 0.0      # 备用通道恢复了：下次再出错要重新报
        except Exception as exc:
            self._error(index, str(exc))

    def finish(self, timeout=15.0):
        with self._lock:
            unavailable = self._closed or self._switching
            self._finishing = True
            session = self._current
        if unavailable:
            self.close()
            return ''
        try:
            return session.finish(timeout)
        finally:
            self.close()

    def close(self):
        with self._lock:
            self._closed = True
            sessions = tuple(self._sessions)
        for session in sessions:
            session.close()


def create_defense_asr(settings, **kwargs):
    if not settings.get('backup_asr_enabled', False):
        return create_realtime_asr(**kwargs)
    model = str(settings.get('backup_asr_model', '') or '').strip()
    if not model:
        raise ApiError('开启备用识别后，请选择备用识别模型')
    return StandbyASR(backup_model=model, **kwargs)
