"""Local rehearsal playback; synthesis must finish before entering this loop."""
from __future__ import annotations


def play_pcm(pcm, player, stop, pause, on_progress, sample_rate=24000):
    """Play short blocks, retaining the exact byte offset across a pause.

    Progress is based on PCM frames, not wall time, so synthesis and pauses
    cannot advance the reading highlight. Device buffers may add a small lag.
    """
    block_bytes = max(2, int(sample_rate * .04) * 2)
    for offset in range(0, len(pcm), block_bytes):
        while pause.is_set() and not stop.is_set():
            stop.wait(.01)
        if stop.is_set():
            return False
        part = pcm[offset:offset + block_bytes]
        player.write(part)
        on_progress((offset + len(part)) / len(pcm))
    return not stop.is_set()
