"""Local driver lifecycle check. Discards every sample; no files with audio or API calls."""
import faulthandler
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teams_voice_translator.audio import MicrophoneCapture
from teams_voice_translator.defense.settings import DefenseSettings

out = Path('test-artifacts/microphone-lifecycle')
out.mkdir(parents=True, exist_ok=True)
with (out / 'native-errors.log').open('w') as log:
    faulthandler.enable(log)
    device = DefenseSettings().shared.get('input_device')
    stats = {'cycles': 0, 'bytes_discarded': 0, 'reader_threads_remaining': 0}
    def discard(pcm): stats['bytes_discarded'] += len(pcm)
    for _ in range(20):
        capture = MicrophoneCapture(device, discard, block_ms=20)
        try:
            capture.start()
            time.sleep(.06)
        finally:
            capture.stop()
        if capture._thread.is_alive():
            raise RuntimeError('Microphone reader did not stop')
        if capture._error:
            raise RuntimeError(str(capture._error))
        stats['cycles'] += 1
    stats['passed'] = stats['cycles'] == 20 and stats['bytes_discarded'] > 0
    (out / 'result.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats))
    faulthandler.disable()
