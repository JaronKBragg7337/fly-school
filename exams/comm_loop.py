"""The communication loop test - runs comm_protocol.md exactly. (Claude Code, 2026-09-17 03:25 EDT)
external symbol (odour) -> ORN sensory neurons -> fly-v3 network + KC->MBON learning -> MBON channels -> MDN (endogenous
descending neuron) -> decoded reply (AVOID / NO-AVOID). No model or classifier inside the loop; the decoder is a threshold on
the fly's own pre-training MDN response. Writes results/comm_loop_<tag>.json with every trial.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
from flysim_v3 import FlyBrainV3 as FlyBrain
from mushroom import MushroomBody
import calibration as C

GRAPH = Path(os.environ.get("COMM_GRAPH", r"C:/Users/lilli/Fly-Lab/versions/fly-v3/graph_v3_s0.35.npz"))
TAG = os.environ.get("COMM_TAG", "test")
SEEDS = tuple(int(s) for s in os.environ.get("COMM_SEEDS", "101,202,303,404,505,606,707,808").split(","))
A_TYPE, B_TYPE, C_TYPE = os.environ.get("COMM_A", "ORN_DL3"), os.environ.get("COMM_B", "ORN_DM1"), os.environ.get("COMM_C", "ORN_VA2")
A_HZ, B_HZ = float(os.environ.get("COMM_A_HZ", 40.0)), float(os.environ.get("COMM_B_HZ", 200.0))
TEACH_EPOCHS, REPS = int(os.environ.get("COMM_EPOCHS", 12)), int(os.environ.get("COMM_REPS", 6))
LR = float(os.environ.get("COMM_LR", 0.06))               # variant pupil: learning rate (CHOSEN, declared); 0.06 = control
ANS_TYPES = tuple(os.environ.get("COMM_ANSWER", "MDN").split(","))
PULSE_MS, PRE_MS, TAIL_MS = 300.0, 20.0, 40.0
R_TYPES = ("MBON09",); P_TYPES = ("MBON25", "MBON25-like", "MBON34"); ANSWER = "MDN"

fb = FlyBrain(GRAPH); p = fb.p
THRESH = np.load(os.environ["COMM_THRESH"]).astype(np.float32) if os.environ.get("COMM_THRESH") else np.full(fb.n, p.v_thresh, dtype=np.float32)   # fly-v4 per-KC thresholds
types = fb.types.astype(str)
T = lambda *names: np.flatnonzero(np.isin(types, names))
STIM = {"A": (T(A_TYPE), A_HZ), "B": (T(B_TYPE), B_HZ), "C": (T(C_TYPE), None)}
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
MBON_GAIN = float(os.environ.get("COMM_MBON_GAIN", 1.0))       # variant pupil: MBON output gain (CHOSEN, declared); 1.0 = control
if MBON_GAIN != 1.0:
    gpn[np.array([x.startswith("MBON") for x in types])] *= np.float32(MBON_GAIN)
CALIB = json.load(open(os.environ["COMM_CALIB"])) if os.environ.get("COMM_CALIB") else None      # fly-v5 calib.json
if CALIB:
    is_pn = np.array([("PN" in x and not x.startswith("MBON")) for x in types]); gpn[is_pn] *= np.float32(CALIB["pn"])
    gpn[types == "APL"] *= np.float32(CALIB["apl"]); gpn[np.array([x.startswith("KC") for x in types])] *= np.float32(CALIB["kc"])
    gpn[np.array([x.startswith("MBON") for x in types])] *= np.float32(CALIB["mbon"])
    THRESH = THRESH.copy(); THRESH[np.array([x.startswith("KC") for x in types])] += np.float32(CALIB["kc_thresh_shift_mv"])
READ = {"R": T(*R_TYPES), "P": T(*P_TYPES), "MDN": T(*ANS_TYPES), "DNa02": T("DNa02"), "DNa01": T("DNa01")}
for _x in os.environ.get("COMM_READ_EXTRA", "").split(","):
    if _x: READ[_x] = T(_x)
print(f"graph {GRAPH.name}; A {A_TYPE} {len(STIM['A'][0])} cells; B {B_TYPE} {len(STIM['B'][0])}; C {C_TYPE} {len(STIM['C'][0])}; "
      f"R {len(READ['R'])} cells, P {len(READ['P'])}, MDN {len(READ['MDN'])}", flush=True)


def seed_for(base, phase, stim, rep):
    return int.from_bytes(hashlib.sha256(f"comm:{base}:{phase}:{stim}:{rep}".encode()).digest()[:4], "little")


def trial(mb, stim, seed, hz_override=None):
    cells, hz = STIM[stim]; hz = hz_override or hz
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed); prob = min(1.0, hz * p.dt / 1000.0)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    kc_mask = np.zeros(fb.n, bool); kc_mask[mb.kc] = True
    kc_seen = np.zeros(fb.n, bool); per = np.zeros(fb.n, dtype=np.int32); total = 0
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            hit = cells[rng.random(len(cells)) < prob]
            if len(hit): v[hit] = THRESH[hit] + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= THRESH) & (refr <= 0))
        if len(fired):
            total += len(fired); per[fired] += 1; refr[fired] = fb.refr_steps; v[fired] = p.v_reset; kc_seen[fired[kc_mask[fired]]] = True
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = steps * p.dt / 1000.0
    out = {k: float(per[idx].sum() / max(1, len(idx)) / secs) for k, idx in READ.items()}
    out.update({"kc": np.flatnonzero(kc_seen), "kc_n": int(kc_seen.sum()), "hz": total / secs / fb.n}); return out


def cold(mb, seed, phase, c_hz):
    out = {}
    for s in ("A", "B", "C"):
        rs = [trial(mb, s, seed_for(seed, phase, s, r), c_hz if s == "C" else None) for r in range(REPS)]
        out[s] = {k: [round(r[k], 3) for r in rs] for k in READ}
        out[s]["kc_n"] = [r["kc_n"] for r in rs]; out[s]["hz"] = [round(r["hz"], 2) for r in rs]; out[s]["_kc"] = [set(r["kc"].tolist()) for r in rs]
    j = lambda a, b: len(a & b) / max(1, len(a | b))
    out["_jaccard"] = {"AA": round(float(np.mean([j(out["A"]["_kc"][0], out["A"]["_kc"][k]) for k in range(1, REPS)])), 3),
                       "AB": round(float(np.mean([j(a, b) for a in out["A"]["_kc"] for b in out["B"]["_kc"]])), 3),
                       "AC": round(float(np.mean([j(a, b) for a in out["A"]["_kc"] for b in out["C"]["_kc"]])), 3),
                       "BC": round(float(np.mean([j(a, b) for a in out["B"]["_kc"] for b in out["C"]["_kc"]])), 3)}
    for s in ("A", "B", "C"): out[s].pop("_kc")
    return out


t0 = time.perf_counter(); results = []
# C's rate: equalise KC count to A (probe with a fresh MB, not part of the test)
_mb = MushroomBody(fb, calibration=C.CHOSEN, sides=RUNTIME / "build" / "mb_sides.json", store=LAB / "comm_probe.npz", clock=lambda: 0.0)
ka = trial(_mb, "A", 11)["kc_n"]; c_hz = 40.0
for _ in range(5):
    kc_ = trial(_mb, "C", 12, c_hz)["kc_n"]
    if kc_ and abs(kc_ - ka) / max(1, ka) <= 0.2: break
    c_hz = float(np.clip(c_hz * ((ka / max(1, kc_)) ** 0.7 if kc_ else 1.5), 5, 250))
(LAB / "comm_probe.npz").unlink(missing_ok=True)
print(f"C rate equalised: A -> {ka} KCs; C @{c_hz:.1f} Hz -> {kc_} KCs", flush=True)

READ["PAM"] = np.asarray(_mb.reward_side); READ["PPL"] = np.asarray(_mb.punish_side)
for seed in SEEDS:
    store = LAB / f"comm_seed_{seed}.npz"; store.unlink(missing_ok=True)
    mb = MushroomBody(fb, lr=LR, calibration=C.CHOSEN, sides=RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    pre = cold(mb, seed, "pre", c_hz)
    teach = []
    for epoch in range(TEACH_EPOCHS):
        for s, val in (("A", +1), ("B", -1)):
            r = trial(mb, s, seed_for(seed, "teach", s, epoch))
            mb.forget_trace(); mb.observe(r["kc"]); hit = int(mb.dopamine(val, 1.0)); mb.apply()
            teach.append({"epoch": epoch, "stim": s, "dopamine": val, "synapses_hit": hit})
    mb.save()
    post = cold(mb, seed, "post", c_hz)
    # decode replies from the fly's own untrained MDN response to the same odour
    dec = {}
    for s in ("A", "B", "C"):
        mu, sd = float(np.mean(pre[s]["MDN"])), float(np.std(pre[s]["MDN"]))
        if os.environ.get("COMM_DECODE") == "valence2":          # test 3: two answer channels, each against its own pre baseline
            muP, sdP = float(np.mean(pre[s]["PAM"])), float(np.std(pre[s]["PAM"])) or 1e-6
            muL, sdL = float(np.mean(pre[s]["PPL"])), float(np.std(pre[s]["PPL"])) or 1e-6
            replies = []
            for xP, xL in zip(post[s]["PAM"], post[s]["PPL"]):
                zP, zL = (xP - muP) / sdP, (xL - muL) / sdL
                replies.append("APPROACH" if (zP < -1 and zP <= zL) else ("AVOID" if zL < -1 else "NONE"))
            dec[s] = {"replies": replies, "approach_rate": round(sum(r == "APPROACH" for r in replies) / REPS, 3), "avoid_rate": round(sum(r == "AVOID" for r in replies) / REPS, 3),
                      "pam_pre_post": [round(muP, 2), round(float(np.mean(post[s]["PAM"])), 2)], "ppl_pre_post": [round(muL, 2), round(float(np.mean(post[s]["PPL"])), 2)]}
        elif os.environ.get("COMM_DECODE") == "drop1sd":          # test 2: answer channel falls below its own pre baseline
            thr = mu - 1.0 * sd; replies = ["LEARNED" if x < thr else "NONE" for x in post[s]["MDN"]]
            dec[s] = {"threshold": round(thr, 3), "replies": replies, "avoid_rate": round(sum(r == "LEARNED" for r in replies) / REPS, 3)}
        else:
            thr = mu + 2 * sd; replies = ["AVOID" if x > thr else "NO-AVOID" for x in post[s]["MDN"]]
            dec[s] = {"threshold": round(thr, 3), "replies": replies, "avoid_rate": round(sum(r == "AVOID" for r in replies) / REPS, 3)}
    R_drop = float(np.mean(post["A"]["R"])) < float(np.mean(pre["A"]["R"]))
    L_drop = float(np.mean(post["A"]["MDN"])) < float(np.mean(pre["A"]["MDN"]))
    P_drop = float(np.mean(post["B"]["P"])) < float(np.mean(pre["B"]["P"]))
    row = {"seed": seed, "pre": pre, "post": post, "teach": teach, "decode": dec, "R_A_pre_post": [round(float(np.mean(pre["A"]["R"])), 2), round(float(np.mean(post["A"]["R"])), 2)],
           "P_B_pre_post": [round(float(np.mean(pre["B"]["P"])), 2), round(float(np.mean(post["B"]["P"])), 2)], "R_drop": R_drop, "P_drop": P_drop, "L_drop": L_drop, "L_A_pre_post": [round(float(np.mean(pre["A"]["MDN"])), 2), round(float(np.mean(post["A"]["MDN"])), 2)],
           "MDN_mean": {s: [round(float(np.mean(pre[s]["MDN"])), 2), round(float(np.mean(post[s]["MDN"])), 2)] for s in ("A", "B", "C")}, "mb": mb.stats()}
    results.append(row)
    if os.environ.get("COMM_DECODE") == "valence2":
        print(f"seed {seed}: A approach {dec['A']['approach_rate']} avoid {dec['A']['avoid_rate']} | B approach {dec['B']['approach_rate']} avoid {dec['B']['avoid_rate']} | C approach {dec['C']['approach_rate']} avoid {dec['C']['avoid_rate']} | "
              f"PAM A {dec['A']['pam_pre_post']} B {dec['B']['pam_pre_post']} C {dec['C']['pam_pre_post']} | PPL A {dec['A']['ppl_pre_post']} B {dec['B']['ppl_pre_post']} C {dec['C']['ppl_pre_post']} | MDN {row['MDN_mean']} {time.perf_counter()-t0:.0f}s", flush=True)
    else:
        print(f"seed {seed}: AVOID rate A {dec['A']['avoid_rate']} B {dec['B']['avoid_rate']} C {dec['C']['avoid_rate']} | MDN A {row['MDN_mean']['A']} B {row['MDN_mean']['B']} C {row['MDN_mean']['C']} | "
              f"R(A) {row['R_A_pre_post']} {'drop' if R_drop else 'no'}  P(B) {row['P_B_pre_post']} {'drop' if P_drop else 'no'}  {time.perf_counter()-t0:.0f}s", flush=True)
    if not os.environ.get("COMM_KEEP_STORE"):
        store.unlink(missing_ok=True)

pool = {s: round(float(np.mean([r["decode"][s]["avoid_rate"] for r in results])), 3) for s in ("A", "B", "C")}
seeds_B_gt_A = sum(r["decode"]["B"]["avoid_rate"] > r["decode"]["A"]["avoid_rate"] for r in results)
nR, nP, nL = sum(r["R_drop"] for r in results), sum(r["P_drop"] for r in results), sum(r["L_drop"] for r in results)
if os.environ.get("COMM_DECODE") == "valence2":
    pool = {s: {"approach": round(float(np.mean([r["decode"][s]["approach_rate"] for r in results])), 3), "avoid": round(float(np.mean([r["decode"][s]["avoid_rate"] for r in results])), 3)} for s in ("A", "B", "C")}
    nA = sum(r["decode"]["A"]["approach_rate"] > r["decode"]["C"]["approach_rate"] for r in results)
    nB = sum(r["decode"]["B"]["avoid_rate"] > r["decode"]["C"]["avoid_rate"] for r in results)
    crit = {"1_A_approach_ge_0.50": pool["A"]["approach"] >= 0.50, "2_B_avoid_ge_0.50": pool["B"]["avoid"] >= 0.50,
            "3_C_any_le_0.25": (pool["C"]["approach"] + pool["C"]["avoid"]) <= 0.25, "4_seeds_A_gt_C_and_B_gt_C_ge_6of8": nA >= 6 and nB >= 6}
    seeds_B_gt_A = {"A_gt_C": nA, "B_gt_C": nB}
elif os.environ.get("COMM_DECODE") == "drop1sd":
    seeds_A_gt_C = sum(r["decode"]["A"]["avoid_rate"] > r["decode"]["C"]["avoid_rate"] for r in results)
    crit = {"1_A_learned_ge_0.50": pool["A"] >= 0.50, "2_C_and_B_le_0.25": pool["C"] <= 0.25 and pool["B"] <= 0.25,
            "3_seeds_A_gt_C_ge_6of8": seeds_A_gt_C >= 6, "4_L_A_drop_ge_6of8": nL >= 6}
else:
    crit = {"1_B_avoid_ge_0.60": pool["B"] >= 0.60, "2_A_and_C_avoid_le_0.30": pool["A"] <= 0.30 and pool["C"] <= 0.30,
            "3_seeds_B_gt_A_ge_6of8": seeds_B_gt_A >= 6, "4_R_drop_ge5_and_P_drop_ge5": nR >= 5 and nP >= 5}
out = {"protocol": "comm_protocol.md (predeclared 2026-09-17 03:20 EDT)", "graph": str(GRAPH), "seeds": SEEDS, "c_hz": c_hz,
       "mbon_gain": MBON_GAIN, "calib": CALIB, "thresh": os.environ.get("COMM_THRESH", "scalar -45"), "lr": LR, "epochs": TEACH_EPOCHS, "reps": REPS, "A": A_TYPE, "B": B_TYPE, "C": C_TYPE, "a_hz": A_HZ, "b_hz": B_HZ, "answer": ANS_TYPES, "pooled_avoid_rate": pool, "seeds_B_gt_A": seeds_B_gt_A, "R_drop_seeds": nR, "P_drop_seeds": nP, "L_drop_seeds": nL, "decode": os.environ.get("COMM_DECODE", "avoid2sd"), "criteria": crit, "PASS": all(crit.values()),
       "elapsed_s": round(time.perf_counter() - t0, 1), "seeds_detail": results}
(LAB / "results" / f"comm_loop_{TAG}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "seeds_detail"}, indent=1))
