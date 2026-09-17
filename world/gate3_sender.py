"""Gate 3 sender process: learns odour A (test-9 rule), reports MDN per trial pre and post; the world file is written by gate3_world.py.
usage: python gate3_sender.py <seed> <out_dir>      (Claude Code, 2026-09-17; gate3_protocol.md)
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
import stack
from flysim_v3 import FlyBrainV3
from mushroom import MushroomBody
import calibration as C

seed = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
REPS, EPOCHS, LR = int(__import__("os").environ.get("GATE_REPS", "24")), 12, float(__import__("os").environ.get("GATE_LR", "0.2")); A_TYPE, C_TYPE = "ORN_DA2", "ORN_VM5d"; HZ = 40.0
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
fb = FlyBrainV3(stack.GRAPH); fb.wdata[:] = fb.wdata          # fresh graph weights (this process runs one fly)
assert np.allclose(gb.wdata_csr[gb.csc_perm], fb.wdata)
A, Cc = gb.where(A_TYPE), gb.where(C_TYPE); kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
READ = {"MDN": gb.where("MDN"), "PAM5": gb.where("MBON09", "MBON01", "MBON05", "MBON03", "MBON06"), "DNa13": gb.where("DNa13"), "DNa03": gb.where("DNa03")}
def sfor(phase, x, rep): return int.from_bytes(hashlib.sha256(f"gate3:{seed}:{phase}:{x}:{rep}".encode()).digest()[:4], "little")
def cold():
    res = {}
    for X, cells in (("A", A), ("C", Cc)):
        r = gb.trial([(cells, HZ)], [sfor("cold", X, k) for k in range(REPS)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0)
        res[X] = {k: [round(float(x), 3) for x in stack.rates(r, idx)] for k, idx in READ.items()}
        res[X]["kc_n"] = [int(x) for x in (r["counts"][kc] > 0).sum(0)]; res[X]["brain_hz"] = [round(float(x), 3) for x in r["hz"]]
    return res
store = out / f"{seed}_mb.npz"; store.unlink(missing_ok=True)
mb = MushroomBody(fb, lr=LR, calibration=C.CHOSEN, sides=stack.RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
pre = cold(); teach = []
for epoch in range(EPOCHS):
    r = gb.trial([(A, HZ)], [sfor("teach", "A", epoch)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0)
    fired = kc[r["counts"][kc, 0] > 0]
    mb.forget_trace(); mb.observe(fired); hit = int(mb.dopamine(+1, 1.0)); mb.apply(); gb.sync_weights(fb.wdata)
    teach.append({"epoch": epoch, "kc_active": int(len(fired)), "synapses_hit": hit})
post = cold(); store.unlink(missing_ok=True)
info = {"seed": seed, "pre": pre, "post": post, "teach": teach, "mb": mb.stats(), "reps": REPS, "epochs": EPOCHS, "lr": LR, "A": A_TYPE, "C": C_TYPE, "hz": HZ,
        "mdn_mean": {X: [round(float(np.mean(pre[X]["MDN"])), 2), round(float(np.mean(post[X]["MDN"])), 2)] for X in ("A", "C")},
        "pam5_mean": {X: [round(float(np.mean(pre[X]["PAM5"])), 2), round(float(np.mean(post[X]["PAM5"])), 2)] for X in ("A", "C")},
        "provenance": stack.provenance(), "wall_s": round(time.perf_counter() - t0)}
(out / f"{seed}_sender.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
print(json.dumps({"seed": seed, "mdn": info["mdn_mean"], "pam5": info["pam5_mean"], "wall_s": info["wall_s"]}), flush=True)
