"""Two explicitly authorized short synthesis calls; audio remains in RAM."""
from __future__ import annotations

import argparse
import audioop
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sounddevice as sd
from teams_voice_translator.aliyun import BailianClient
from teams_voice_translator.audio import MultiOutputPlayer
from teams_voice_translator.defense.settings import DefenseSettings
from teams_voice_translator.defense.speech_policy import speech_settings, NATURAL_INSTRUCTION, INSTRUCTION_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--authorized-two-calls", action="store_true")
    args = parser.parse_args()
    settings = DefenseSettings()
    payload = speech_settings(settings)
    device = settings.get("monitor_output_device")
    info = sd.query_devices(device, kind="output")
    name = str(info["name"])
    print(json.dumps({"model": payload["tts_model"], "voice_present": bool(payload["voice"]),
                      "monitor_device": device, "monitor_name": name}, ensure_ascii=False), flush=True)
    if args.inspect or not args.authorized_two_calls:
        return 0
    if any(word in name.lower() for word in ("cable", "voicemeeter", "virtual")):
        raise RuntimeError("试听设备是虚拟会议输出，请先选择物理耳机或扬声器")
    if not payload["voice"] or not settings.workspace_id:
        raise RuntimeError("当前音色或 Workspace 未配置")
    client = BailianClient(settings.get_api_key(), settings.workspace_id,
                           timeout=60, proxy=settings.ws_proxy, proxy_mode=settings.proxy_mode)
    text = "Thank you for the question. Let me explain the main result of my research."
    payload.update(tts_rate=1.0, tts_pitch=1.0)
    report = {"model": payload["tts_model"], "text": text, "audio_saved": False, "results": []}
    with MultiOutputPlayer([device], 24000) as player:
        for label in ("A", "B"):
            values = dict(payload)
            values["tts_instruction"] = NATURAL_INSTRUCTION if label == "B" and payload["tts_model"] in INSTRUCTION_MODELS else ""
            pcm = bytearray()
            started = time.monotonic()
            first = None
            def receive(chunk):
                nonlocal first
                if first is None:
                    first = time.monotonic()
                pcm.extend(chunk)
                if len(pcm) > 8 * 1024 * 1024:
                    raise RuntimeError("短句响应异常过长，已停止")
            print(f"SYNTHESIZING_{label}", flush=True)
            # Intentionally no automatic retry: this run is limited to two calls.
            client.stream_tts(text, values, receive, threading.Event())
            if not pcm or len(pcm) % 2:
                raise RuntimeError(f"{label} 未返回有效 PCM16 音频")
            completed = time.monotonic()
            stats = {"label": label, "duration_seconds": round(len(pcm) / 48000, 3),
                     "first_chunk_seconds": round(first - started, 3),
                     "synthesis_seconds": round(completed - started, 3),
                     "peak_pcm16": audioop.max(pcm, 2), "rms_pcm16": audioop.rms(pcm, 2),
                     "instruction_enabled": bool(values["tts_instruction"])}
            print(f"PLAYING_{label}", flush=True)
            for offset in range(0, len(pcm), 1920):
                player.write(pcm[offset:offset + 1920])
            report["results"].append(stats)
            print(json.dumps(stats), flush=True)
            del pcm
            if label == "A":
                time.sleep(2)
    folder = Path(__file__).resolve().parents[1] / "test-artifacts" / "voice-ab"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
