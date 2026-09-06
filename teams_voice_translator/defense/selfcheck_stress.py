"""Packaged regression: real Qt event loop and TTS worker; no credentials/network/devices."""
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QCoreApplication, QEvent, QThread, Slot
from PySide6.QtWidgets import QApplication

from .settings import DefenseSettings
from .ui import SelfCheckDialog


def run_stress():
    app = QApplication.instance() or QApplication([])
    counters = {"runs": 0, "gui_updates": 0, "thread_violations": 0, "pcm_blocks": 0}
    class CheckedDialog(SelfCheckDialog):
        @Slot(int, bool, str)
        def _finish_row(self, *args):
            counters["gui_updates"] += 1
            if QThread.currentThread() != self.thread():
                counters["thread_violations"] += 1
            super()._finish_row(*args)

    class Player:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def write(self, pcm):
            counters["pcm_blocks"] += 1
            time.sleep(.001)
        def close(self): pass

    def stream(text, values, on_audio, token):
        on_audio(bytes(1920))
        on_audio(bytes(1920))

    client = Mock()
    client._stream_legacy_tts.side_effect = stream
    output = Path("test-artifacts/selfcheck-stress")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, \
         patch("requests.post", side_effect=AssertionError("Network forbidden in stress test")), \
         patch("requests.get", side_effect=AssertionError("Network forbidden in stress test")), \
         patch("teams_voice_translator.defense.ui._safe_list_audio_devices", return_value=([], [])), \
         patch("teams_voice_translator.defense.ui._safe_list_loopback_devices", return_value=[]), \
         patch("teams_voice_translator.defense.ui.BailianClient", return_value=client), \
         patch("teams_voice_translator.audio.MultiOutputPlayer", Player), \
         patch("teams_voice_translator.defense.ui.ContextTranslator") as translator:
        translator.return_value.translate.return_value = "Hello."
        settings = DefenseSettings(Path(tmp))
        settings.shared.update({"workspace_id": "offline-test"})
        settings.update({"tts_voice_id": "offline-test", "tts_model": "cosyvoice-v3.5-plus"})
        settings.get_api_key = lambda: "offline-test"
        for _ in range(50):
            dialog = CheckedDialog(None, settings)
            dialog.show()
            dialog._run()
            deadline = time.monotonic() + 5
            while not dialog.run_button.isEnabled() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.002)
            if not dialog.run_button.isEnabled():
                raise RuntimeError("Self-check completion timed out")
            if "已在本机" not in dialog.table.item(7, 1).text():
                raise RuntimeError("Self-check TTS result did not complete")
            dialog._thread.join(1)
            counters["runs"] += 1
            dialog.close()
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
    counters["tts_threads_remaining"] = sum(t.name == "defense-tts" for t in threading.enumerate())
    counters["passed"] = (counters["runs"] == 50 and counters["gui_updates"] == 400
                           and counters["thread_violations"] == 0 and counters["tts_threads_remaining"] == 0)
    (output / "result.json").write_text(json.dumps(counters, indent=2), encoding="utf-8")
    return 0 if counters["passed"] else 1
