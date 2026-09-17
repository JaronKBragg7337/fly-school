"""fly-v4 = fly-v3 + per-Kenyon-cell homeostatic thresholds. (Claude Code, 2026-09-17 03:55 EDT)

WHY: at one input rate, most odour pairs share 40-70% of their active KCs (results/glom_scan_v3.json) - sparse but not
odour-specific, so every lesson lands on the same synapses. Real KCs regulate their own excitability so that each responds
to a small fraction of odours (sparse, decorrelated coding; Turner 2008, Honegger 2011). CHOSEN rule: iterate over an
odour panel; a KC that fires for > HI of the panel gets its threshold raised by UP mV; one that fires for none gets it
lowered by DOWN mV; bounded. Nothing else changes. MEASURED: per-KC response fractions, KC fraction per odour, pairwise
Jaccard. Writes versions/fly-v4/kc_thresh.npy (per-neuron threshold vector; non-KC cells keep -45 mV) + manifest.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime"); sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3")
from flysim_v3 import FlyBrainV3
import calibration as C
OUT = Path(r"C:/Users/lilli/Fly-Lab/versions/fly-v4"); OUT.mkdir(exist_ok=True)
GRAPH = Path(r"C:/Users/lilli/Fly-Lab/versions/fly-v3/graph_v3_s0.35.npz")
PANEL = ["ORN_DA1", "ORN_DA2", "ORN_DL1", "ORN_DL3", "ORN_DL4", "ORN_DL5", "ORN_DM2", "ORN_DM3", "ORN_DM6", "ORN_VA1d", "ORN_VA1v", "ORN_VA6", "ORN_VL2a", "ORN_VL2p", "ORN_VM2", "ORN_VM3", "ORN_VM4", "ORN_VM5d"]
HZ, PULSE_MS, PRE_MS, TAIL_MS = 40.0, 300.0, 20.0, 40.0
HI, UP, DOWN, ITERS = 0.20, 2.0, 1.5, 14          # CHOSEN (second pass: stronger on the silent side)
TH_MIN, TH_MAX = -50.0, -25.0                   # mV bounds (rest -52, default threshold -45)

fb = FlyBrainV3(GRAPH); p = fb.p
types = fb.types.astype(str); kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
thresh = np.full(fb.n, p.v_thresh, dtype=np.float32)


def trial(cells, seed):
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed); prob = min(1.0, HZ * p.dt / 1000.0)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata; ever = np.zeros(fb.n, bool); total = 0
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            hit = cells[rng.random(len(cells)) < prob]
            if len(hit): v[hit] = thresh[hit] + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= thresh) & (refr <= 0))
        if len(fired):
            total += len(fired); ever[fired] = True; refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    return ever[kc], total / (steps * p.dt / 1000.0) / fb.n


def jac(a, b): return float((a & b).sum() / max(1, (a | b).sum()))


t0 = time.perf_counter(); log = []
cells = {g: fb.where(type_re=rf"^{g}$") for g in PANEL}
for it in range(ITERS + 1):
    resp = np.zeros((len(PANEL), len(kc)), bool); hz = []
    for i, g in enumerate(PANEL):
        r, h = trial(cells[g], 500 + it * 100 + i); resp[i] = r; hz.append(h)
    frac_per_kc = resp.mean(0); frac_per_odour = resp.mean(1)
    pairs = [jac(resp[i], resp[j]) for i in range(len(PANEL)) for j in range(i + 1, len(PANEL))]
    row = {"iter": it, "kc_active_per_odour_pct": round(float(frac_per_odour.mean()) * 100, 2), "median_pair_jaccard": round(float(np.median(pairs)), 3),
           "mean_pair_jaccard": round(float(np.mean(pairs)), 3), "kcs_silent_pct": round(float((frac_per_kc == 0).mean()) * 100, 1),
           "kcs_generalist_pct": round(float((frac_per_kc > HI).mean()) * 100, 1), "brain_hz": round(float(np.mean(hz)), 2),
           "thresh_kc_mean": round(float(thresh[kc].mean()), 2), "t": round(time.perf_counter() - t0)}
    log.append(row); print(json.dumps(row), flush=True)
    if it == ITERS: break
    up = kc[frac_per_kc > HI]; down = kc[frac_per_kc == 0]
    thresh[up] = np.minimum(TH_MAX, thresh[up] + UP); thresh[down] = np.maximum(TH_MIN, thresh[down] - DOWN)

np.save(OUT / "kc_thresh.npy", thresh)
(OUT / "MANIFEST.md").write_text(f"""# fly-v4 — fly-v3 + per-KC homeostatic thresholds (2026-09-17, Claude Code)
Graph: fly-v3 graph_v3_s0.35.npz (unchanged). Cell: refractory 3.8 ms (v3). NEW: kc_thresh.npy — per-neuron spike threshold;
non-KC cells -45 mV; each KC's threshold set by the CHOSEN homeostatic rule below on an {len(PANEL)}-odour panel @{HZ} Hz.
Rule: iterate {ITERS}x; KC firing for > {HI*100:.0f}% of the panel: threshold +{UP} mV; KC firing for none: -{DOWN} mV; bounds [{TH_MIN}, {TH_MAX}] mV.
Why: sparse but non-specific KC code in v1-v3 (pair overlap 0.4-0.7). Biology: KC intrinsic excitability is regulated (sparse,
decorrelated odour coding; Turner 2008, Honegger 2011). Measured trajectory:
{chr(10).join(json.dumps(r) for r in log)}
Use: load kc_thresh.npy and use v >= thresh[i] in the kernel (comm_loop.py COMM_THRESH).
""", encoding="utf-8")
json.dump(log, open(OUT / "calibration_log.json", "w"), indent=1)
print("saved", OUT / "kc_thresh.npy")
