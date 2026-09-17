"""Gate 1 receiver process: a fresh fly whose ear (JO-A) is driven by the world's sound schedule; AMMC is read. (Claude Code, 2026-09-17)
usage: python receiver.py <seeds,comma> <schedule_dir> <condition> <out_dir>
Reads <schedule_dir>/song_<seed>_<condition>.npz (one per seed). Receiver RNG seed = seed + 1; JO-A hit draws are made on EVERY
step (used only while song_on == 1) so two conditions with the same seed share the same random stream — that is what makes the
identity checks R2/R3 exact. The receiver never reads the sender's files.
Writes <out_dir>/receiver_<seed>_<condition>.npz: full spike-count vector (n,), rates, provenance.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import stack

seeds = [int(s) for s in sys.argv[1].split(",")]; sched_dir = Path(sys.argv[2]); cond = sys.argv[3]; out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
SONG_HZ = 120.0                                                  # CHOSEN (gate1_protocol.md; Grade 0b row 5)
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
joa = np.flatnonzero(np.array([t.startswith("JO-A") for t in types]))
ammc = np.flatnonzero(np.array([t.startswith("AMMC") for t in types])); gf = gb.where("DNp01"); vpoen = gb.where("vpoEN")
songs = np.stack([np.load(sched_dir / f"song_{s}_{cond}.npz")["song"] for s in seeds], axis=1).astype(bool)      # (steps, B)
B = len(seeds); prob = min(1.0, SONG_HZ * gb.dt / 1000.0)
rngs = [np.random.default_rng(int(s) + 1) for s in seeds]
def ear(step):
    draws = np.stack([rng.random(len(joa)) < prob for rng in rngs], axis=1)                  # drawn every step, every fly
    if step >= songs.shape[0] or not songs[step].any(): return None
    return joa, draws & songs[step][None, :]
r = gb.trial([], seeds, pre_ms=stack.PRE_MS, pulse_ms=stack.PULSE_MS, tail_ms=stack.TAIL_MS, external=ear)
assert r["steps"] == songs.shape[0], (r["steps"], songs.shape)
for b, seed in enumerate(seeds):
    info = {"seed": seed, "receiver_rng_seed": seed + 1, "condition": cond, "song_hz": SONG_HZ, "joa_cells": int(len(joa)), "song_on_ms": round(float(songs[:, b].sum() * gb.dt), 1),
            "ammc_hz": round(float(stack.rates(r, ammc)[b]), 4), "ammc_cells": int(len(ammc)), "gf_hz": round(float(stack.rates(r, gf)[b]), 4),
            "vpoen_hz": round(float(stack.rates(r, vpoen)[b]), 4), "joa_hz": round(float(stack.rates(r, joa)[b]), 3),
            "brain_hz": round(float(r["hz"][b]), 4), "provenance": stack.provenance()}
    np.savez_compressed(out / f"receiver_{seed}_{cond}.npz", counts=r["counts"][:, b], info=json.dumps(info))
    print(json.dumps({k: v for k, v in info.items() if k != "provenance"}), flush=True)
print(f"receiver done {cond} {len(seeds)} seeds in {time.perf_counter()-t0:.0f}s on {stack.DEVICE}", flush=True)
