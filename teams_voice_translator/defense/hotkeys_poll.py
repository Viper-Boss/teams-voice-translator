"""GetAsyncKeyState 轮询式全局热键（替代 pynput 低级钩子）。

pynput 的 WH_KEYBOARD_LL 钩子线程在 Windows 上偶发 access violation 直接闪退
（崩溃日志已捕获一次）。轮询方案没有钩子线程：40ms 读一次按键状态 + 边沿检测，
全局有效、与窗口焦点无关，按住/松开语义与钩子版一致。
"""
from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable

VK_MAP: dict[str, int] = {
    "esc": 0x1B,
    **{f"f{i}": 0x70 + (i - 1) for i in range(1, 13)},
}


class PollingHotkeys:
    def __init__(
        self,
        *,
        direct_key: str,
        translate_key: str,
        cancel_key: str,
        direct_press: Callable[[], None],
        direct_release: Callable[[], None],
        translate_press: Callable[[], None],
        translate_release: Callable[[], None],
        cancel: Callable[[], None],
        interval_ms: int = 40,
        state_reader: Callable[[int], bool] | None = None,
    ) -> None:
        self.keys = {
            "direct": direct_key.lower(),
            "translate": translate_key.lower(),
            "cancel": cancel_key.lower(),
        }
        self.direct_press = direct_press
        self.direct_release = direct_release
        self.translate_press = translate_press
        self.translate_release = translate_release
        self.cancel = cancel
        self.interval = max(0.02, interval_ms / 1000)
        self._reader = state_reader or self._read_key
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev: dict[str, bool] = {}

    @staticmethod
    def _read_key(vk: int) -> bool:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

    def start(self) -> None:
        self.stop()
        if sys_platform_windows() is False:
            raise RuntimeError("轮询热键仅支持 Windows")
        self._stop_event.clear()
        self._prev = {name: False for name in ("direct", "translate", "cancel")}
        self._thread = threading.Thread(target=self._loop, name="defense-hotkeys", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            for name, press, release in (
                ("direct", self.direct_press, self.direct_release),
                ("translate", self.translate_press, self.translate_release),
            ):
                down = self._read(self.keys[name])
                previous = self._prev[name]
                if down and not previous:
                    press()
                elif not down and previous:
                    release()
                self._prev[name] = down
            if self._read(self.keys["cancel"]):
                if not self._prev.get("cancel"):
                    self._prev["cancel"] = True
                    self.cancel()
            else:
                self._prev["cancel"] = False
            time.sleep(self.interval)

    def _read(self, name: str) -> bool:
        vk = VK_MAP.get(name)
        return bool(vk is not None and self._reader(vk))


def sys_platform_windows() -> bool:
    import sys

    return sys.platform == "win32"
