"""Local output check. Open exactly the selected endpoint, with its native format."""
from array import array
import math
import threading
import sounddevice as sd


def play_device_test(device, stop: threading.Event) -> bool:
    if stop.is_set():
        return False
    info = sd.query_devices(device, 'output')
    channels = min(2, int(info['max_output_channels']))
    if channels < 1:
        raise ValueError('所选设备没有播放通道')
    rate = int(info['default_samplerate'])
    if not 8000 <= rate <= 384000:
        raise ValueError('设备返回的采样率无效')
    # Two gentle notes with fades. No cloud calls, default-device fallback or global sd.stop.
    with sd.RawOutputStream(device=device, samplerate=rate, channels=channels,
                            dtype='int16') as stream:
        total = int(rate * 1.2)
        for start in range(0, total, max(1, int(rate * .04))):
            if stop.is_set():
                stream.abort()
                return False
            samples = array('h')
            for frame in range(start, min(total, start + int(rate * .04))):
                t = frame / rate
                local = t if t < .45 else t - .65
                active = 0 <= local < .4
                envelope = max(0, min(1, local / .02, (.4 - local) / .03)) if active else 0
                value = int(32767 * .12 * envelope * math.sin(2 * math.pi * (660 if t < .45 else 880) * t))
                samples.extend([value] * channels)
            stream.write(samples.tobytes())
    return not stop.is_set()
