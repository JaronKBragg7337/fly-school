"""Mushroom-body calibration search — the real next step (Claude Code, 2026-09-17 04:05 EDT). Runs detached, N workers.

Objective (predeclared; FLY-BRAIN.md 'THE COMMUNICATION LOOP'): on fly-v3, find per-type gains (PN, APL, KC, MBON) such that
  - pairwise KC overlap across an odour panel is low (median Jaccard <= 0.15),
  - same-odour reliability is high (Jaccard >= 0.6),
  - 3-10% of KCs active per odour, brain < 8 Hz/cell,
then (stage 2, top candidates) a learned, odour-specific MBON drop: paired odour's PAM-side MBON output falls >= 30%,
control odour's falls <= 10%. Random search; every candidate's numbers are written; nothing is tuned on the test seeds.
Usage: python mb_calib_search.py --worker i --of n --rounds 1 --per-round 48   (writes results/mb_calib/cand_*.json)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
from flysim_v3 import FlyBrainV3
from mushroom import MushroomBody
import calibration as C

ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--of", type=int, default=1)
ap.add_argument("--per-round", type=int, default=48); ap.add_argument("--round", type=int, default=0); ap.add_argument("--stage2", action="store_true")
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=0.06); ap.add_argument("--only", type=str, default="")
args = ap.parse_args()
OUT = LAB / "results" / "mb_calib"; OUT.mkdir(parents=True, exist_ok=True)
GRAPH = Path(r"C:/Users/lilli/Fly-Lab/versions/fly-v3/graph_v3_s0.35.npz")
PANEL = ["ORN_DA2", "ORN_DL3", "ORN_DL4", "ORN_DM6", "ORN_VA1d", "ORN_VL2a", "ORN_VM4", "ORN_VM5d"]
HZ, PULSE_MS, PRE_MS, TAIL_MS = 40.0, 300.0, 20.0, 40.0
# search space (log-uniform), CHOSEN: gains multiply the CHOSEN calibration pn05_apl10_kc03
SPACE = {"pn": (0.3, 3.0), "apl": (0.5, 8.0), "kc": (0.3, 3.0), "kc_thresh_shift_mv": (0.0, 8.0), "mbon": (0.5, 3.0)}

fb = FlyBrainV3(GRAPH); p = fb.p
types = fb.types.astype(str)
is_pn = np.array([("PN" in x and not x.startswith("MBON")) for x in types]); is_apl = types == "APL"; is_kc = np.array([x.startswith("KC") for x in types]); is_mbon = np.array([x.startswith("MBON") for x in types])
base_gpn = C.gains_for(fb, C.CHOSEN)[fb.type_code].astype(np.float32)
kc = np.flatnonzero(is_kc)
cells = {g: fb.where(type_re=rf"^{g}$") for g in PANEL}


def candidate(i):
    rng = np.random.default_rng(1000 * args.round + i)
    c = {}
    for k, (lo, hi) in SPACE.items():
        c[k] = float(lo + (hi - lo) * rng.random()) if k == "kc_thresh_shift_mv" else float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * rng.random()))
    return c


def make(c):
    gpn = base_gpn.copy(); gpn[is_pn] *= c["pn"]; gpn[is_apl] *= c["apl"]; gpn[is_kc] *= c["kc"]; gpn[is_mbon] *= c["mbon"]
    thresh = np.full(fb.n, p.v_thresh, dtype=np.float32); thresh[is_kc] += c["kc_thresh_shift_mv"]
    return gpn.astype(np.float32), thresh


def trial(gpn, thresh, cell_idx, seed, mb=None):
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed); prob = min(1.0, HZ * p.dt / 1000.0)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata; ever = np.zeros(fb.n, bool); per = np.zeros(fb.n, np.int32); total = 0
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            hit = cell_idx[rng.random(len(cell_idx)) < prob]
            if len(hit): v[hit] = thresh[hit] + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= thresh) & (refr <= 0))
        if len(fired):
            total += len(fired); ever[fired] = True; per[fired] += 1; refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = steps * p.dt / 1000.0
    return {"kc": ever[kc], "hz": total / secs / fb.n, "per": per}


def jac(a, b): return float((a & b).sum() / max(1, (a | b).sum()))


def stage1(c):
    gpn, thresh = make(c); sets = {}; hz = []
    for g in PANEL:
        r0 = trial(gpn, thresh, cells[g], 11); r1 = trial(gpn, thresh, cells[g], 12); sets[g] = (r0["kc"], r1["kc"]); hz += [r0["hz"], r1["hz"]]
    rel = float(np.mean([jac(a, b) for a, b in sets.values()]))
    pairs = [jac(sets[a][0], sets[b][0]) for i, a in enumerate(PANEL) for b in PANEL[i + 1:]]
    frac = float(np.mean([s[0].mean() for s in sets.values()]))
    m = {"reliability": round(rel, 3), "median_overlap": round(float(np.median(pairs)), 3), "kc_frac": round(frac, 4), "brain_hz": round(float(np.mean(hz)), 2)}
    ok = m["median_overlap"] <= 0.15 and m["reliability"] >= 0.6 and 0.03 <= m["kc_frac"] <= 0.10 and m["brain_hz"] < 8
    # score: distance to targets (lower is better)
    m["score"] = round(max(0, m["median_overlap"] - 0.15) * 5 + max(0, 0.6 - m["reliability"]) * 3 + (0 if 0.03 <= frac <= 0.10 else abs(frac - 0.065) * 20) + max(0, m["brain_hz"] - 8) * 0.2, 4)
    m["stage1_pass"] = bool(ok); return m


def stage2(c, A="ORN_DA2", Ctrl="ORN_VM5d"):
    """learned, odour-specific MBON drop: 12 pairings A+reward; PAM-side MBON output on A vs control before/after."""
    gpn, thresh = make(c)
    mb = MushroomBody(fb, lr=args.lr, calibration=C.CHOSEN, sides=RUNTIME / "build" / "mb_sides.json", store=OUT / f"s2_{args.worker}.npz", clock=lambda: 0.0)
    pam = np.asarray(mb.reward_side)
    def out_rate(g, seed):
        r = trial(gpn, thresh, cells[g], seed, mb); return r["per"][pam].sum() / max(1, len(pam)) / ((PRE_MS + PULSE_MS + TAIL_MS) / 1000.0), r
    preA = [out_rate(A, 200 + k)[0] for k in range(4)]; preC = [out_rate(Ctrl, 300 + k)[0] for k in range(4)]
    for e in range(args.epochs):
        _, r = out_rate(A, 400 + e); mb.forget_trace(); mb.observe(kc[r["kc"]]); mb.dopamine(+1, 1.0); mb.apply()
    postA = [out_rate(A, 500 + k)[0] for k in range(4)]; postC = [out_rate(Ctrl, 600 + k)[0] for k in range(4)]
    (OUT / f"s2_{args.worker}.npz").unlink(missing_ok=True)
    dA = 1 - np.mean(postA) / max(1e-6, np.mean(preA)); dC = 1 - np.mean(postC) / max(1e-6, np.mean(preC))
    return {"pam_A_pre": round(float(np.mean(preA)), 2), "pam_A_post": round(float(np.mean(postA)), 2), "pam_C_pre": round(float(np.mean(preC)), 2), "pam_C_post": round(float(np.mean(postC)), 2),
            "drop_A": round(float(dA), 3), "drop_C": round(float(dC), 3), "stage2_pass": bool(dA >= 0.30 and dC <= 0.10), "depressed": mb.stats()["depressed"]}


t0 = time.perf_counter()
if not args.stage2:
    for i in range(args.worker, args.per_round, args.of):
        c = candidate(i); m = stage1(c)
        json.dump({"round": args.round, "cand": i, "params": c, "stage1": m, "t": round(time.perf_counter() - t0)}, open(OUT / f"r{args.round}_c{i:03d}.json", "w"), indent=1)
        print(f"r{args.round} c{i:03d} {json.dumps(c)} -> {json.dumps(m)}  {time.perf_counter()-t0:.0f}s", flush=True)
else:
    files = sorted(OUT.glob("r*_c*.json")); rows = [json.load(open(f)) for f in files]
    if args.only:
        rows = [r for r in rows if r["cand"] in [int(x) for x in args.only.split(",")]]
    else:
        rows = [r for r in rows if "stage2" not in r]
    rows.sort(key=lambda r: r["stage1"]["score"])
    for r in rows[args.worker::args.of][:6]:
        m2 = stage2(r["params"]); key = "stage2" if (args.epochs == 12 and args.lr == 0.06) else f"stage2_e{args.epochs}_lr{args.lr}"; r[key] = m2
        json.dump(r, open(OUT / f"r{r['round']}_c{r['cand']:03d}.json", "w"), indent=1)
        print(f"stage2 r{r['round']} c{r['cand']:03d} score {r['stage1']['score']} -> {json.dumps(m2)}  {time.perf_counter()-t0:.0f}s", flush=True)
