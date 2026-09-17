"""Gate 3 world: 1-D physics from the sender's MDN rate to cVA concentration at the receiver. (Claude Code, 2026-09-17; gate3_protocol.md)
usage: python gate3_world.py <seed> <sender_dir> <out_path>
d = d_start + k * MDN_Hz * bout ; c = 1 / (1 + (d/d0)^2) ; receiver ORN_DA1 rate = RMAX * c.   Reads MDN only.
Writes fields for LIVE (pre, post) and MUTE (post clamped to pre).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
seed = int(sys.argv[1]); sd = Path(sys.argv[2]); out = Path(sys.argv[3])
D_START, K, BOUT, D0, RMAX = 20.0, 0.5, 2.0, 40.0, 60.0                        # CHOSEN (gate3_protocol.md)
s = json.loads((sd / f"{seed}_sender.json").read_text())
def field(mdn_hz):
    d = D_START + K * float(mdn_hz) * BOUT; c = 1.0 / (1.0 + (d / D0) ** 2); return {"d_mm": round(d, 2), "c": round(c, 4), "orn_hz": round(RMAX * c, 3)}
w = {"seed": seed, "physics": {"d_start_mm": D_START, "k_mm_s_per_hz": K, "bout_s": BOUT, "d0_mm": D0, "rmax_hz": RMAX}, "LIVE": {}, "MUTE": {}}
for X in ("A", "C"):
    pre = [field(x) for x in s["pre"][X]["MDN"]]; post = [field(x) for x in s["post"][X]["MDN"]]
    w["LIVE"][X] = {"pre": pre, "post": post}; w["MUTE"][X] = {"pre": pre, "post": pre}
    print(json.dumps({"seed": seed, "X": X, "c_pre": round(float(np.mean([f["c"] for f in pre])), 4), "c_post": round(float(np.mean([f["c"] for f in post])), 4)}), flush=True)
out.write_text(json.dumps(w, indent=1), encoding="utf-8")
