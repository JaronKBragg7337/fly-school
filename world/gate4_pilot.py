"""Gate 4 PILOT (notebook, not the test): can two flies close a loop through a cVA world, and does B's reply reach A? (Claude Code, 2026-09-17)
Bout 1: A smells odour A (learned) [+ B's cVA]; B smells A's cVA. Both move by their own MDN through the same kinematic rule.
Bouts 2..N: odour off; each fly smells only the other's cVA (pilot: with odour on, A's cVA channel is suppressed to 0 at faint cVA).
World: d_k = d_{k-1} + K * (MDN_A + MDN_B) * BOUT   (both back away from each other); ORN_DA1 rate at each nose = RMAX * c(d), c = 1/(1+(d/D0)^2).
RMAX and D0 CHOSEN so both noses sit in the graded band measured in the pilot (MDN graded for ORN_DA1 0-4 Hz).
Conditions: LIVE (B's MDN as produced), YOKED (B's MDN per bout replayed from the matched realisation of the UNTRAINED-A LIVE run),
MUTE (A's cVA input frozen at its bout-1 value). Measured: A's MDN in bouts 2..N; paired LIVE-YOKED and LIVE-MUTE over realisations.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
import stack
from flysim_v3 import FlyBrainV3
from mushroom import MushroomBody
import calibration as C

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "60101,60202").split(",")]
REPS = int(os.environ.get("G4_REPS", 24)); BOUTS = int(os.environ.get("G4_BOUTS", 3)); LR, EPOCHS = 0.2, 12
D_START, K, BOUT, D0, RMAX = float(os.environ.get("G4_D0START", 40.0)), float(os.environ.get("G4_K", 0.5)), 2.0, float(os.environ.get("G4_D0", 40.0)), float(os.environ.get("G4_RMAX", 5.0))
CHANNEL = os.environ.get("G4_CHANNEL", "cva"); G_AIR = float(os.environ.get("G4_GAIR", 1.0)); PULSE = float(os.environ.get("G4_PULSE_MS", 300.0)); ODOUR_ALL = bool(os.environ.get("G4_ODOUR_ALL"))
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
fb = FlyBrainV3(stack.GRAPH); W_ORIG = fb.wdata.copy()
A_cells, orn, mdn = gb.where("ORN_DA2"), gb.where("ORN_DA1"), gb.where("MDN"); kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
vpn = gb.where("DA1_vPN")
import re as _re
wing = np.flatnonzero(np.array([bool(_re.search(r"^(hg[1-4]|b[1-3]|i[12]|iii[13]|ps1|tp[12]) MN$|^MNwm", x)) for x in types]))
jocef = np.concatenate([np.flatnonzero(np.array([x.startswith(p) for x in types])) for p in ("JO-C", "JO-E", "JO-F")])
def c_of(d): return 1.0 / (1.0 + (d / D0) ** 2)
def sfor(seed, who, phase, bout, rep): return int.from_bytes(hashlib.sha256(f"gate4:{seed}:{who}:{phase}:{bout}:{rep}".encode()).digest()[:4], "little")
def fly_trial(seed, who, phase, bout, orn_rates, odour, ear_cells=None):
    ear_cells = orn if ear_cells is None else ear_cells
    """One bout for one fly across REPS realisations: per-realisation cVA rate (external), optional odour (Poisson stim)."""
    seeds = [sfor(seed, who, phase, bout, r) for r in range(REPS)]
    probs = np.array([min(1.0, hz * gb.dt / 1000.0) for hz in orn_rates]); rngs = [np.random.default_rng(s + 7) for s in seeds]
    on0, on1 = int(round(20.0 / gb.dt)), int(round((20.0 + PULSE) / gb.dt))
    def ear(step):
        u = np.stack([rng.random(len(ear_cells)) for rng in rngs], axis=1)
        if not (on0 <= step < on1): return None
        return ear_cells, u < probs[None, :]
    r = gb.trial([(A_cells, 40.0)] if odour else [], seeds, pre_ms=20.0, pulse_ms=PULSE, tail_ms=40.0, external=ear)
    return stack.rates(r, mdn), stack.rates(r, vpn if who == "B" else wing) if CHANNEL == "cva" else stack.rates(r, wing), r["hz"]
def run_loop(seed, phase, cond, yoke_B=None):
    d = np.full(REPS, D_START); log = {"d": [d.copy()], "A_mdn": [], "B_mdn": [], "A_vpn": [], "B_vpn": []}
    a_rate_frozen = None; B_wing_prev = np.zeros(REPS)
    for k in range(1, BOUTS + 1):
        rate = RMAX * c_of(d)
        if CHANNEL == "air":
            # B's reply from the PREVIOUS bout's wings reaches A's antenna now (air), scaled by distance; A keeps smelling the symbol
            air = G_AIR * (yoke_B[k - 2] if (cond == "YOKED" and k >= 2) else B_wing_prev) * c_of(d)
            a_rate = air if (cond != "MUTE" or a_rate_frozen is None) else a_rate_frozen
            if cond == "MUTE" and a_rate_frozen is None: a_rate_frozen = air.copy()
            A_m, A_v, A_hz = fly_trial(seed, "A", phase, k, a_rate, odour=(ODOUR_ALL or k == 1), ear_cells=jocef)
            B_m, B_w, B_hz = fly_trial(seed, "B", phase, k, rate, odour=False)
            B_wing_prev = B_w; B_v = B_w
        else:
            a_rate = rate if (cond != "MUTE" or a_rate_frozen is None) else a_rate_frozen
            if cond == "MUTE" and a_rate_frozen is None: a_rate_frozen = rate.copy()
            A_m, A_v, A_hz = fly_trial(seed, "A", phase, k, a_rate, odour=(k == 1))
            B_m, B_v, B_hz = fly_trial(seed, "B", phase, k, rate, odour=False)
            if cond == "YOKED": B_m = yoke_B[k - 1]
        d = d + K * (A_m + B_m) * BOUT
        log["d"].append(d.copy()); log["A_mdn"].append(A_m); log["B_mdn"].append(B_m); log["A_vpn"].append(A_v); log["B_vpn"].append(B_v)
        assert max(A_hz.max(), B_hz.max()) < 10
    return log
out = []
for seed in SEEDS:
    fb.wdata[:] = W_ORIG; gb.sync_weights(fb.wdata)
    store = HERE / "results" / f"gate4_pilot_{seed}.npz"; store.unlink(missing_ok=True)
    mb = MushroomBody(fb, lr=LR, calibration=C.CHOSEN, sides=stack.RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    pre_live = run_loop(seed, "pre", "LIVE")
    for epoch in range(EPOCHS):
        r = gb.trial([(A_cells, 40.0)], [sfor(seed, "teach", "A", 0, epoch)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0)
        fired = kc[r["counts"][kc, 0] > 0]; mb.forget_trace(); mb.observe(fired); mb.dopamine(+1, 1.0); mb.apply(); gb.sync_weights(fb.wdata)
    post_live = run_loop(seed, "post", "LIVE")
    post_yoked = run_loop(seed, "post", "YOKED", yoke_B=pre_live["B_vpn"] if CHANNEL == "air" else pre_live["B_mdn"])
    post_mute = run_loop(seed, "post", "MUTE")
    store.unlink(missing_ok=True)
    def z(a, b): dd = np.asarray(a) - np.asarray(b); return float(dd.mean() / max(1e-9, dd.std(ddof=1) / np.sqrt(len(dd))))
    row = {"seed": seed, "bout1_A_mdn_pre_post": [round(float(pre_live["A_mdn"][0].mean()), 2), round(float(post_live["A_mdn"][0].mean()), 2)],
           "d_after_bout1_pre_post": [round(float(pre_live["d"][1].mean()), 1), round(float(post_live["d"][1].mean()), 1)],
           "B_cVA_orn_hz_bout2_pre_post": [round(float((RMAX * c_of(pre_live["d"][1])).mean()), 2), round(float((RMAX * c_of(post_live["d"][1])).mean()), 2)],
           "B_mdn_bout2_pre_post": [round(float(pre_live["B_mdn"][1].mean()), 2), round(float(post_live["B_mdn"][1].mean()), 2)], "B_mdn_bout2_z_post_vs_pre": round(z(post_live["B_mdn"][1], pre_live["B_mdn"][1]), 2),
           "A_vpn_bout2_live": round(float(post_live["A_vpn"][1].mean()), 2), "A_mdn_bout2_LIVE_YOKED_MUTE": [round(float(x["A_mdn"][1].mean()), 2) for x in (post_live, post_yoked, post_mute)],
           "A_mdn_bout2_z_live_vs_yoked": round(z(post_live["A_mdn"][1], post_yoked["A_mdn"][1]), 2), "A_mdn_bout2_z_live_vs_mute": round(z(post_live["A_mdn"][1], post_mute["A_mdn"][1]), 2),
           "A_mdn_bout3_LIVE_YOKED_MUTE": [round(float(x["A_mdn"][2].mean()), 2) for x in (post_live, post_yoked, post_mute)] if BOUTS >= 3 else None,
           "A_mdn_bout3_z_live_vs_yoked": round(z(post_live["A_mdn"][2], post_yoked["A_mdn"][2]), 2) if BOUTS >= 3 else None,
           "A_mdn_bout3_z_live_vs_mute": round(z(post_live["A_mdn"][2], post_mute["A_mdn"][2]), 2) if BOUTS >= 3 else None,
           "d_final_pre_post_LIVE": [round(float(pre_live["d"][-1].mean()), 1), round(float(post_live["d"][-1].mean()), 1)],
           "d_bout2_LIVE_vs_YOKED_mm": round(float((post_live["d"][2] - post_yoked["d"][2]).mean()), 2) if BOUTS >= 2 else None}
    out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(HERE / "results" / "gate4_pilot.json", "w"), indent=1); print(f"pilot {time.perf_counter()-t0:.0f}s")
