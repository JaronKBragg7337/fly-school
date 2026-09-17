"""Readout scan: can ANY descending neuron (or DN type) say `-.`? (Claude Code, 2026-09-17 00:00 EDT)

The Grade 1 exams read only DNa01. Rows 1-7 say DNa01 can sometimes produce `.-` and never `-.`. This scan plays the two
held-out rhythms cold (fresh fly-v1, no dopamine, protocol constants identical to morse_grade1.py) and records every
descending neuron's spikes in 10 ms bins. Then it decodes each single DN cell and each DN type (cells summed) with the
same decoder. Output: which cells/types produce `.-` and `-.` exactly, how often, plus the primitives `.` and `-` so we
know which readouts can say anything at all. CHOSEN: same timings/split as Grade 1; seeds 73, 7337, 101, 202, 303.
MEASURED: everything else. Writes results/dn_readout_scan.json.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); sys.path.insert(0, str(LAB))
import morse_grade1 as G
from flysim import FlyBrain
import calibration as C

PATTERNS = (".", "-", ".-", "-.")
REPS = int(os.environ.get("SCAN_REPS", 6))
SEEDS = tuple(int(s) for s in os.environ.get("SCAN_SEEDS", "73,7337,101,202,303").split(","))

fb = FlyBrain(G.RUNTIME / "build" / "graph.npz")
dn = fb.where(superclass="descending")
if not len(dn):
    dn = np.flatnonzero(np.array([t.startswith("DN") for t in fb.types]))
jo = fb.where(type_re=r"^JO-A")
gains = C.gains_for(fb, C.CHOSEN)
gpn = (np.ones(fb.n, dtype=np.float32) if gains is None else gains[fb.type_code].astype(np.float32))
p = fb.p
print(f"DNs {len(dn)} types {len(set(fb.types[dn]))} JO-A {len(jo)}", flush=True)


def trial(pattern, seed):
    sound, pre_steps = G.make_timeline(pattern, p.dt)
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    bin_steps = G.ms_steps(G.BIN_MS, p.dt)
    nb = int(np.ceil(len(sound) / bin_steps))
    counts = np.zeros((len(dn), nb), dtype=np.int32)
    dn_pos = -np.ones(fb.n, dtype=np.int64); dn_pos[dn] = np.arange(len(dn))
    prob = min(1.0, G.JO_HZ * p.dt / 1000.0)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    for step, on in enumerate(sound):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on:
            hit = jo[rng.random(len(jo)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            fd = fired[dn_pos[fired] >= 0]
            if len(fd):
                counts[dn_pos[fd], min(step // bin_steps, nb - 1)] += 1
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    return counts, int(round(pre_steps / bin_steps))


t0 = time.perf_counter()
types = fb.types[dn]
utypes, tidx = np.unique(types, return_inverse=True)
cell_hits = {pat: np.zeros(len(dn), dtype=np.int32) for pat in PATTERNS}
type_hits = {pat: np.zeros(len(utypes), dtype=np.int32) for pat in PATTERNS}
cell_any = np.zeros(len(dn), dtype=np.int32)
n_trials = 0
for seed in SEEDS:
    for pat in PATTERNS:
        for rep in range(REPS):
            counts, pre_bins = trial(pat, G.seed_for(seed, "scan", pat, rep))
            n_trials += 1
            # per cell
            for i in range(len(dn)):
                if counts[i].sum() == 0: continue
                cell_any[i] += 1
                dec, _, _ = G.decode(counts[i], pre_bins)
                if dec == pat: cell_hits[pat][i] += 1
            # per type (cells summed)
            tc = np.zeros((len(utypes), counts.shape[1]), dtype=np.int32)
            np.add.at(tc, tidx, counts)
            for k in range(len(utypes)):
                if tc[k].sum() == 0: continue
                dec, _, _ = G.decode(tc[k], pre_bins)
                if dec == pat: type_hits[pat][k] += 1
    print(f"seed {seed} done {time.perf_counter()-t0:.0f}s", flush=True)

per_pat = len(SEEDS) * REPS
def top(hits, names, k=25):
    order = np.argsort(-hits)[:k]
    return [{"name": str(names[i]), "exact": int(hits[i]), "of": per_pat} for i in order if hits[i] > 0]
out = {
    "chosen": {"patterns": PATTERNS, "reps": REPS, "seeds": SEEDS, "protocol": "morse_grade1 constants; fresh fly-v1; no dopamine"},
    "measured": {
        "dn_cells": int(len(dn)), "dn_types": int(len(utypes)), "trials": n_trials, "trials_per_pattern": per_pat,
        "cells_that_ever_spiked": int((cell_any > 0).sum()),
        "types_saying_-.": top(type_hits["-."], utypes), "types_saying_.-": top(type_hits[".-"], utypes),
        "types_saying_.": top(type_hits["."], utypes, 15), "types_saying_-": top(type_hits["-"], utypes, 15),
        "cells_saying_-.": top(cell_hits["-."], [f"{types[i]}#{int(fb.bodies[dn[i]])}" for i in range(len(dn))]),
        "cells_saying_.-": top(cell_hits[".-"], [f"{types[i]}#{int(fb.bodies[dn[i]])}" for i in range(len(dn))]),
        "n_types_any_-.": int((type_hits["-."] > 0).sum()), "n_types_any_.-": int((type_hits[".-"] > 0).sum()),
        "n_cells_any_-.": int((cell_hits["-."] > 0).sum()), "n_cells_any_.-": int((cell_hits[".-"] > 0).sum()),
        "dna01_type": {pat: int(type_hits[pat][list(utypes).index("DNa01")]) if "DNa01" in utypes else None for pat in PATTERNS},
        "elapsed_s": round(time.perf_counter() - t0, 1),
    },
}
(LAB / "results" / "dn_readout_scan.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
np.savez_compressed(LAB / "results" / "dn_readout_scan.npz", cell_names=np.array([f"{types[i]}#{int(fb.bodies[dn[i]])}" for i in range(len(dn))]),
                    cell_idx=dn, type_names=utypes, cell_any=cell_any, per_pat=per_pat,
                    **{f"cell_{pat}": cell_hits[pat] for pat in PATTERNS}, **{f"type_{pat}": type_hits[pat] for pat in PATTERNS})
print(json.dumps({k: v for k, v in out["measured"].items() if k.startswith("n_") or k in ("dna01_type", "elapsed_s", "cells_that_ever_spiked")}, indent=1))
print("TOP -. types:", out["measured"]["types_saying_-."][:8])
print("TOP -. cells:", out["measured"]["cells_saying_-."][:8])
