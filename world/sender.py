"""Gate 1 sender process: drive pIP10, record the wing motor neurons, write them to the world directory. (Claude Code, 2026-09-17)
usage: python sender.py <seeds,comma> <driven:1|0> <out_dir>
Writes <out_dir>/sender_<seed>.npz: wing MN spike raster (steps x 32 bool), pulse-song MN index, rates, provenance.
The sender never sees the receiver. It only writes here.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import stack

seeds = [int(s) for s in sys.argv[1].split(",")]; driven = sys.argv[2] == "1"; out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
PIP10_HZ = 100.0                                                 # CHOSEN (gate1_protocol.md)
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
pip10 = gb.where("pIP10")
wing = np.flatnonzero(np.array([bool(re.search(r"^(hg[1-4]|b[1-3]|i[12]|iii[13]|ps1|tp[12]) MN$|^MNwm", t)) for t in types]))
pulse = np.flatnonzero(np.isin(types[wing], ["hg1 MN", "ps1 MN", "i1 MN", "i2 MN"]))          # positions inside `wing`
song_pat = {n: gb.where(n) for n in ("dPR1", "TN1a_g", "TN1a_h", "TN1a_i", "vPR9_a", "vPR9_b", "vPR9_c")}
stims = [(pip10, PIP10_HZ)] if driven else [(pip10, 0.0)]
r = gb.trial(stims, seeds, pre_ms=stack.PRE_MS, pulse_ms=stack.PULSE_MS, tail_ms=stack.TAIL_MS, record=wing)
for b, seed in enumerate(seeds):
    rec = r["rec"][:, :, b]
    info = {"seed": seed, "driven": driven, "pip10_hz": PIP10_HZ if driven else 0.0, "pip10_cells": int(len(pip10)),
            "wing_types": [str(t) for t in types[wing]], "pulse_song_positions": pulse.tolist(),
            "wing_mn_hz_all": round(float(stack.rates(r, wing)[b]), 3), "pulse_song_mn_hz": round(float(rec[:, pulse].sum() / len(pulse) / r["secs"]), 3),
            "song_pattern_hz": {n: round(float(stack.rates(r, idx)[b]), 3) for n, idx in song_pat.items() if len(idx)},
            "brain_hz": round(float(r["hz"][b]), 4), "steps": int(r["steps"]), "dt_ms": gb.dt, "provenance": stack.provenance()}
    np.savez_compressed(out / f"sender_{seed}.npz", raster=rec, pulse=pulse, info=json.dumps(info))
    print(json.dumps({k: v for k, v in info.items() if k not in ("wing_types", "pulse_song_positions", "provenance")}), flush=True)
print(f"sender done {len(seeds)} seeds in {time.perf_counter()-t0:.0f}s on {stack.DEVICE}", flush=True)
