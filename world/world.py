"""Gate 1 world: turns the sender's wing motor spikes into a sound schedule — the only bridge between the two flies. (Claude Code, 2026-09-17)
usage: python world.py <seed> <in_dir> <mode: live|mute|replay:<path>> <out_path>
Rule (CHOSEN, gate1_protocol.md): song_on[step] = 1 if any pulse-song wing MN (hg1, ps1, i1, i2) spiked in the preceding 5 ms.
mute:   song_on is all zero whatever the sender did (lesion of the channel).
replay: song_on is copied from another schedule file (the scramble: a song with no singer).
The world reads spikes of motor neurons only. It never reads a brain.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

seed = int(sys.argv[1]); in_dir = Path(sys.argv[2]); mode = sys.argv[3]; out = Path(sys.argv[4])
WINDOW_MS = 5.0
z = np.load(in_dir / f"sender_{seed}.npz"); raster = z["raster"]; pulse = z["pulse"]; info = json.loads(str(z["info"]))
steps = raster.shape[0]; win = int(round(WINDOW_MS / info["dt_ms"]))
if mode == "live":
    hits = raster[:, pulse].any(1).astype(np.int32)
    song = np.zeros(steps, dtype=np.int8)
    for t in range(steps):
        if hits[max(0, t - win + 1): t + 1].any(): song[t] = 1
elif mode == "mute":
    song = np.zeros(steps, dtype=np.int8)
elif mode.startswith("replay:"):
    song = np.load(mode.split(":", 1)[1])["song"].astype(np.int8)
else:
    raise SystemExit(f"unknown mode {mode}")
np.savez_compressed(out, song=song, meta=json.dumps({"seed": seed, "mode": mode, "window_ms": WINDOW_MS, "rule": "any pulse-song wing MN spike in preceding 5 ms",
                                                    "song_on_fraction": round(float(song.mean()), 4), "song_on_ms": round(float(song.sum() * info["dt_ms"]), 1),
                                                    "sender_driven": info["driven"], "sender_pulse_song_mn_hz": info["pulse_song_mn_hz"]}))
print(json.dumps({"seed": seed, "mode": mode, "song_on_ms": round(float(song.sum() * info["dt_ms"]), 1), "sender_driven": info["driven"]}), flush=True)
