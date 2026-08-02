from __future__ import annotations

from collections.abc import Callable

from pynput import keyboard


KEY_MAP = {
    "esc": keyboard.Key.esc,
    **{f"f{i}": getattr(keyboard.Key, f"f{i}") for i in range(1, 13)},
}


class HoldHotkeys:
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
    ) -> None:
        self.direct_key = KEY_MAP[direct_key]
        self.translate_key = KEY_MAP[translate_key]
        self.cancel_key = KEY_MAP[cancel_key]
        self.direct_press = direct_press
        self.direct_release = direct_release
        self.translate_press = translate_press
        self.translate_release = translate_release
        self.cancel = cancel
        self.listener: keyboard.Listener | None = None
        self.held: set[object] = set()

    def start(self) -> None:
        self.stop()

        def on_press(key) -> None:
            if key in self.held:
                return
            self.held.add(key)
            if key == self.direct_key:
                self.direct_press()
            elif key == self.translate_key:
                self.translate_press()
            elif key == self.cancel_key:
                self.cancel()

        def on_release(key) -> None:
            self.held.discard(key)
            if key == self.direct_key:
                self.direct_release()
            elif key == self.translate_key:
                self.translate_release()

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self.held.clear()
