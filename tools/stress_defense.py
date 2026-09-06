"""Offline queue soak: no devices, credentials, or cloud API calls."""
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teams_voice_translator.defense.tts_session import TtsHttpSession


class LocalClient:
    def _stream_qwen3_tts_http(self, text, settings, on_audio, cancel):
        for _ in range(8):
            if cancel.is_set():
                return
            on_audio(b"\x01\x00" * 120)


def main():
    completed = []
    errors = []
    baseline = threading.active_count()
    for mode in ("natural", "streaming"):
        session = TtsHttpSession(LocalClient(), {"voice": "offline", "speech_mode": mode},
                                on_audio=lambda pcm: None,
                                on_utterance_done=lambda text, count: completed.append(text),
                                on_error=errors.append)
        session.start()
        for batch in range(16):
            for index in range(16):
                assert session.speak(f"{mode}-{batch}-{index}")
            assert session.wait_until_idle(5)
        session.close()
        assert not session._thread.is_alive()
    assert not errors, errors
    assert len(completed) == len(set(completed)) == 512
    assert threading.active_count() <= baseline
    print("PASS: 512 queued utterances, both modes, no duplicates, no residual TTS worker threads.")


if __name__ == "__main__":
    main()
