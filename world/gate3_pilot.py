"""Gate 3 PILOT (notebook, not the test): does a mushroom-body lesson reach the sender's song? (Claude Code, 2026-09-17 12:45 EDT)
Sender on the frozen fly-v10 stack, test-9 learning (A + reward, 12 epochs, lr 0.06, A-only). Before and after, on IDENTICAL
sensory realisations: odour X (A or C) at 40 Hz together with a courtship context - pIP10 at CTX Hz (0 / 30 / 60) - and we read the
wing motor neurons (song), the song-pattern neurons (dPR1, TN1a), pIP10's own downstream, and the 5 PAM-side MBONs (to confirm the
lesson landed). Pilot seeds only (30101, 30202); the test seeds stay untouched. Nothing here is a pass line.
"""
from __future__ import annotations
import hashlib, json, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
import stack
from flysim_v3 import FlyBrainV3
from mushroom import MushroomBody
import calibration as C

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "30101,30202").split(",")]
CTX = [0.0, 30.0, 60.0]; REPS = 8; EPOCHS = 12; LR = 0.06; A_TYPE, C_TYPE = "ORN_DA2", "ORN_VM5d"
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
fb = FlyBrainV3(stack.GRAPH); W_ORIG = fb.wdata.copy()
assert np.allclose(gb.wdata_csr[gb.csc_perm], fb.wdata)
A, Cc = gb.where(A_TYPE), gb.where(C_TYPE); pip10 = gb.where("pIP10")
wing = np.flatnonzero(np.array([bool(re.search(r"^(hg[1-4]|b[1-3]|i[12]|iii[13]|ps1|tp[12]) MN$|^MNwm", t)) for t in types]))
pulse = wing[np.isin(types[wing], ["hg1 MN", "ps1 MN", "i1 MN", "i2 MN"])]
kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
READ = {"pulseMN": pulse, "wingMN": wing, "dPR1": gb.where("dPR1"), "TN1a": np.flatnonzero(np.array([t.startswith("TN1a") for t in types])),
        "vPR9": np.flatnonzero(np.array([t.startswith("vPR9") for t in types])), "PAM5": gb.where("MBON09", "MBON01", "MBON05", "MBON03", "MBON06"),
        "DNa13+DNa03+MDN": gb.where("DNa13", "DNa03", "MDN"), "pC1": np.flatnonzero(np.array([t.startswith("pC1") for t in types]))}
def sfor(base, phase, x, ctx, rep): return int.from_bytes(hashlib.sha256(f"gate3:{base}:{phase}:{x}:{ctx}:{rep}".encode()).digest()[:4], "little")
def song_ms(rec):        # world rule: any pulse-song MN spike in the preceding 5 ms
    hits = rec.any(1); win = 25; on = np.zeros(len(hits), bool)
    for t in range(len(hits)):
        if hits[max(0, t - win + 1): t + 1].any(): on[t] = True
    return float(on.sum() * gb.dt)
def measure(seed, X, ctx):
    cells = A if X == "A" else Cc
    seeds = [sfor(seed, "cold", X, ctx, r) for r in range(REPS)]
    r = gb.trial([(cells, 40.0), (pip10, ctx)], seeds, pre_ms=stack.PRE_MS, pulse_ms=stack.PULSE_MS, tail_ms=stack.TAIL_MS, record=pulse)
    out = {k: round(float(stack.rates(r, idx).mean()), 3) for k, idx in READ.items()}
    out["song_ms"] = round(float(np.mean([song_ms(r["rec"][:, :, b]) for b in range(REPS)])), 1); out["brain_hz"] = round(float(r["hz"].mean()), 3)
    return out
rows = []
for seed in SEEDS:
    fb.wdata[:] = W_ORIG; gb.sync_weights(fb.wdata)
    store = HERE / "results" / f"gate3_pilot_{seed}.npz"; store.unlink(missing_ok=True)
    mb = MushroomBody(fb, lr=LR, calibration=C.CHOSEN, sides=stack.RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    pre = {(X, ctx): measure(seed, X, ctx) for X in ("A", "C") for ctx in CTX}
    for epoch in range(EPOCHS):
        r = gb.trial([(A, 40.0)], [sfor(seed, "teach", "A", 0, epoch)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0)
        fired_kc = kc[r["counts"][kc, 0] > 0]
        mb.forget_trace(); mb.observe(fired_kc); hit = int(mb.dopamine(+1, 1.0)); mb.apply(); gb.sync_weights(fb.wdata)
    post = {(X, ctx): measure(seed, X, ctx) for X in ("A", "C") for ctx in CTX}
    for (X, ctx) in pre:
        row = {"seed": seed, "X": X, "ctx_hz": ctx, "pre": pre[(X, ctx)], "post": post[(X, ctx)],
               "rel": {k: (round(1 - post[(X, ctx)][k] / pre[(X, ctx)][k], 3) if pre[(X, ctx)][k] else None) for k in pre[(X, ctx)]}}
        rows.append(row)
        print(f"seed {seed} {X} ctx {ctx:>4}: " + "  ".join(f"{k} {pre[(X, ctx)][k]}->{post[(X, ctx)][k]}" for k in ("PAM5", "pulseMN", "wingMN", "song_ms", "dPR1", "TN1a", "pC1", "DNa13+DNa03+MDN", "brain_hz")), flush=True)
    store.unlink(missing_ok=True)
json.dump(rows, open(HERE / "results" / "gate3_pilot.json", "w"), indent=1)
print(f"pilot done {time.perf_counter()-t0:.0f}s")
