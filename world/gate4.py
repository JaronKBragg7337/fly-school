"""Gate 4 — the round trip. Runs gate4_protocol.md exactly. (Claude Code, 2026-09-17 19:40 EDT)
usage: python gate4.py <tag> [seeds,comma]     (env GATE4_REPS default 96, GATE4_BOUTS 3)
Two separate brain objects (A trained, B frozen) on the frozen fly-v10 stack; only the world crosses between them.
"""
from __future__ import annotations
import hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
import stack
from flysim_v3 import FlyBrainV3
from mushroom import MushroomBody
import calibration as C

TAG = sys.argv[1] if len(sys.argv) > 1 else "gate4"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "27101,27202,27303,27404,27505,27606,27707,27808").split(",")]
REPS = int(os.environ.get("GATE4_REPS", 96)); BOUTS = int(os.environ.get("GATE4_BOUTS", 3)); LR, EPOCHS = 0.2, 12
D_START, K, BOUT, D0, RMAX_CVA, G_AIR, PULSE = 30.0, 0.25, 2.0, 60.0, 5.0, 0.6, 1000.0           # CHOSEN (gate4_protocol.md)
RES = HERE / "results" / TAG; RES.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
gA = stack.load(); gB = stack.load(); types = gA.types                                             # two separate brains
fb = FlyBrainV3(stack.GRAPH); W_ORIG = fb.wdata.copy()
A_od, C_od, orn, mdn = gA.where("ORN_DA2"), gA.where("ORN_VM5d"), gA.where("ORN_DA1"), gA.where("MDN")
kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
wing = np.flatnonzero(np.array([bool(re.search(r"^(hg[1-4]|b[1-3]|i[12]|iii[13]|ps1|tp[12]) MN$|^MNwm", x)) for x in types]))
jocef = np.concatenate([np.flatnonzero(np.array([x.startswith(p) for x in types])) for p in ("JO-C", "JO-E", "JO-F")])
leak_cells = np.concatenate([A_od, C_od])
def c_of(d): return 1.0 / (1.0 + (d / D0) ** 2)
def sfor(seed, who, phase, bout, rep): return int.from_bytes(hashlib.sha256(f"gate4:{seed}:{who}:{phase}:{bout}:{rep}".encode()).digest()[:4], "little")
def fly_trial(g, seed, who, phase, bout, ear_cells, ear_rates, odour_cells):
    seeds = [sfor(seed, who, phase, bout, r) for r in range(REPS)]
    probs = np.array([min(1.0, hz * g.dt / 1000.0) for hz in ear_rates]); rngs = [np.random.default_rng(s + 7) for s in seeds]
    on0, on1 = int(round(20.0 / g.dt)), int(round((20.0 + PULSE) / g.dt))
    def ear(step):
        u = np.stack([rng.random(len(ear_cells)) for rng in rngs], axis=1)
        if not (on0 <= step < on1): return None
        return ear_cells, u < probs[None, :]
    r = g.trial([(odour_cells, 40.0)] if odour_cells is not None else [], seeds, pre_ms=20.0, pulse_ms=PULSE, tail_ms=40.0, external=ear)
    return {"mdn": stack.rates(r, mdn), "wing": stack.rates(r, wing), "hz": r["hz"], "leak": int(r["counts"][leak_cells].sum())}
def run_loop(seed, phase, odour_cells, cond, yoke_wing=None):
    d = np.full(REPS, D_START); log = {"d": [d.copy()], "A_mdn": [], "B_mdn": [], "B_wing": [], "A_air": [], "B_orn": [], "max_hz": 0.0, "B_leak": 0}
    B_wing_prev = np.zeros(REPS); a_frozen = None
    for k in range(1, BOUTS + 1):
        b_rate = RMAX_CVA * c_of(d)
        src = yoke_wing[k - 2] if (cond == "YOKED" and k >= 2) else B_wing_prev
        air = G_AIR * src * c_of(d)
        if cond == "MUTE":
            if a_frozen is None: a_frozen = air.copy()
            air = a_frozen
        A = fly_trial(gA, seed, "A", phase, k, jocef, air, odour_cells)
        B = fly_trial(gB, seed, "B", phase, k, orn, b_rate, None)
        B_wing_prev = B["wing"]
        d = d + K * (A["mdn"] + B["mdn"]) * BOUT
        log["d"].append(d.copy()); log["A_mdn"].append(A["mdn"]); log["B_mdn"].append(B["mdn"]); log["B_wing"].append(B["wing"]); log["A_air"].append(air.copy()); log["B_orn"].append(b_rate.copy())
        log["max_hz"] = max(log["max_hz"], float(A["hz"].max()), float(B["hz"].max())); log["B_leak"] += B["leak"]
    return log
def z(a, b): dd = np.asarray(a) - np.asarray(b); return float(dd.mean() / max(1e-9, dd.std(ddof=1) / np.sqrt(len(dd))))
rows = []; pooled_A = []; pooled_C = []
for seed in SEEDS:
    fb.wdata[:] = W_ORIG; gA.sync_weights(fb.wdata); gB.sync_weights(W_ORIG)
    store = RES / f"{seed}_mb.npz"; store.unlink(missing_ok=True)
    mb = MushroomBody(fb, lr=LR, calibration=C.CHOSEN, sides=stack.RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    preA = run_loop(seed, "preA", A_od, "LIVE"); preC = run_loop(seed, "preC", C_od, "LIVE")
    for epoch in range(EPOCHS):
        r = gA.trial([(A_od, 40.0)], [sfor(seed, "teach", "A", 0, epoch)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0)
        fired = kc[r["counts"][kc, 0] > 0]; mb.forget_trace(); mb.observe(fired); mb.dopamine(+1, 1.0); mb.apply(); gA.sync_weights(fb.wdata)
    b_weights_frozen = bool(np.allclose(gB.wdata_csr[gB.csc_perm], W_ORIG))
    postA = run_loop(seed, "postA", A_od, "LIVE"); yokA = run_loop(seed, "postA", A_od, "YOKED", yoke_wing=preA["B_wing"]); mutA = run_loop(seed, "postA", A_od, "MUTE")
    postC = run_loop(seed, "postC", C_od, "LIVE"); yokC = run_loop(seed, "postC", C_od, "YOKED", yoke_wing=preC["B_wing"])
    store.unlink(missing_ok=True)
    dA = postA["A_mdn"][1] - yokA["A_mdn"][1]; dC = postC["A_mdn"][1] - yokC["A_mdn"][1]; pooled_A.append(dA); pooled_C.append(dC)
    row = {"seed": seed, "L1_d_bout1_pre_post": [round(float(preA["d"][1].mean()), 2), round(float(postA["d"][1].mean()), 2)], "L1": bool(postA["d"][1].mean() < preA["d"][1].mean()),
           "A_mdn_bout1_pre_post": [round(float(preA["A_mdn"][0].mean()), 2), round(float(postA["A_mdn"][0].mean()), 2)],
           "B_orn_bout2_pre_post": [round(float(preA["B_orn"][1].mean()), 3), round(float(postA["B_orn"][1].mean()), 3)], "B_wing_bout2_pre_post": [round(float(preA["B_wing"][1].mean()), 2), round(float(postA["B_wing"][1].mean()), 2)],
           "A_air_bout2_LIVE_YOKED": [round(float(postA["A_air"][1].mean()), 2), round(float(yokA["A_air"][1].mean()), 2)],
           "A_mdn_bout2_LIVE_YOKED_MUTE": [round(float(postA["A_mdn"][1].mean()), 2), round(float(yokA["A_mdn"][1].mean()), 2), round(float(mutA["A_mdn"][1].mean()), 2)],
           "L2_z_live_vs_mute": round(z(postA["A_mdn"][1], mutA["A_mdn"][1]), 2), "L3_mean_delta": round(float(dA.mean()), 3), "L3_z_fly": round(z(postA["A_mdn"][1], yokA["A_mdn"][1]), 2),
           "bout3_A_mdn_LIVE_YOKED_MUTE": [round(float(postA["A_mdn"][2].mean()), 2), round(float(yokA["A_mdn"][2].mean()), 2), round(float(mutA["A_mdn"][2].mean()), 2)] if BOUTS >= 3 else None,
           "bout3_z_live_vs_yoked": round(z(postA["A_mdn"][2], yokA["A_mdn"][2]), 2) if BOUTS >= 3 else None,
           "C_d_bout1_pre_post": [round(float(preC["d"][1].mean()), 2), round(float(postC["d"][1].mean()), 2)], "C_A_mdn_bout2_LIVE_YOKED": [round(float(postC["A_mdn"][1].mean()), 2), round(float(yokC["A_mdn"][1].mean()), 2)],
           "L4_mean_delta_C": round(float(dC.mean()), 3), "L4_z_fly": round(z(postC["A_mdn"][1], yokC["A_mdn"][1]), 2),
           "L5_B_leak_spikes": int(sum(x["B_leak"] for x in (preA, preC, postA, yokA, mutA, postC, yokC))), "L5_B_weights_frozen": b_weights_frozen,
           "L5_max_hz": round(max(x["max_hz"] for x in (preA, preC, postA, yokA, mutA, postC, yokC)), 3), "mb": mb.stats(), "t": round(time.perf_counter() - t0)}
    row["L2"] = bool(row["L2_z_live_vs_mute"] <= -3); row["L3_sign"] = bool(dA.mean() < 0); row["L5"] = bool(row["L5_B_leak_spikes"] == 0 and b_weights_frozen and row["L5_max_hz"] < 10)
    rows.append(row); print(json.dumps(row), flush=True)
    np.savez_compressed(RES / f"{seed}_loops.npz", **{f"{n}_{k}": np.array(x[k]) for n, x in (("preA", preA), ("postA", postA), ("yokA", yokA), ("mutA", mutA), ("preC", preC), ("postC", postC), ("yokC", yokC)) for k in ("d", "A_mdn", "B_mdn", "B_wing", "A_air", "B_orn")})
pA = np.concatenate(pooled_A); pC = np.concatenate(pooled_C)
zA = float(pA.mean() / (pA.std(ddof=1) / np.sqrt(len(pA)))); zC = float(pC.mean() / (pC.std(ddof=1) / np.sqrt(len(pC))))
n = len(SEEDS); cnt = lambda k: sum(r[k] for r in rows)
crit = {"L1_d_closer_ge_7of8": cnt("L1") >= 7, "L2_hears_B_z_le_m3_ge_7of8": cnt("L2") >= 7, "L3_pooled_z_le_m3_and_sign_ge_6of8": zA <= -3 and cnt("L3_sign") >= 6,
        "L4_control_pooled_abs_z_lt_2": abs(zC) < 2, "L5_no_leak_frozen_B_no_ignition_all": cnt("L5") == n}
verdict = "PASS" if all(crit.values()) else "FAIL"
summary = {"tag": TAG, "protocol": "gate4_protocol.md (predeclared 2026-09-17 19:35 EDT)", "seeds": SEEDS, "reps": REPS, "bouts": BOUTS, "pooled_z_A": round(zA, 3), "pooled_mean_delta_A_hz": round(float(pA.mean()), 3),
           "pooled_z_C": round(zC, 3), "pooled_mean_delta_C_hz": round(float(pC.mean()), 3), "counts": {k: cnt(k) for k in ("L1", "L2", "L3_sign", "L5")}, "criteria": crit, "verdict": verdict,
           "rows": rows, "wall_s": round(time.perf_counter() - t0), "provenance": stack.provenance()}
(RES / "gate4_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("verdict", "criteria", "pooled_z_A", "pooled_mean_delta_A_hz", "pooled_z_C", "counts", "wall_s")}), flush=True)
