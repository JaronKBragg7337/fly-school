"""Global gain sweep: where does the brain stop igniting, and do dot and dash become different stimuli? (Claude, 01:40 EDT)

Finding behind rows 1-8, the valence exam, the memory saturation and the body loop: under the stock calibration any sustained
input ignites the whole brain (~33 Hz/cell), ~65% of Kenyon cells fire for any sound, and dot vs dash KC sets overlap 0.97.
Sweep: scale EVERY synaptic weight by s (CHOSEN grid). For each s, play `.` and `-` (3 reps each, seed 73) and MEASURE:
brain Hz/cell, KC fraction active, Jaccard(dot, dash) vs Jaccard(dot, dot), and whether the four readouts + DNa01 spike at
all. The target regime, stated before running: KC active fraction 5-15% (Turner 2008, Honegger 2011), dot/dash Jaccard
clearly below same-sound Jaccard, DNs still responding. Writes results/gain_sweep.json.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); sys.path.insert(0, str(LAB))
import morse_grade1 as G
from flysim import FlyBrain
from mushroom import MushroomBody
import calibration as C

SCALES = tuple(float(x) for x in os.environ.get("SWEEP", "1.0,0.7,0.5,0.35,0.25,0.15").split(","))
SEED = int(os.environ.get("SWEEP_SEED", 73))
fb = FlyBrain(G.RUNTIME / "build" / "graph.npz")
mb = MushroomBody(fb, calibration=C.CHOSEN, sides=G.RUNTIME / "build" / "mb_sides.json", store=LAB / "sweep_tmp.npz", clock=lambda: 0.0)
base_w = fb.wdata.copy()
jo = fb.where(type_re=r"^JO-A"); gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
p = fb.p
kc = np.zeros(fb.n, bool); kc[mb.kc] = True
READ = {"DNa01": fb.where(type_re=r"^DNa01$"), "DNp17": fb.where(type_re=r"^DNp17$"), "DNge131": fb.where(type_re=r"^DNge131$"),
        "MBON": mb.mbon, "DN_all": np.flatnonzero(fb.superclass.astype(str) == "descending_neuron")}


def trial(pattern, seed):
    sound, pre_steps = G.make_timeline(pattern, p.dt)
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    prob = min(1.0, G.JO_HZ * p.dt / 1000.0); indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    ever = np.zeros(fb.n, bool); total = 0; per = np.zeros(fb.n, np.int32)
    for step, on in enumerate(sound):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on:
            hit = jo[rng.random(len(jo)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            total += len(fired); ever[fired] = True; per[fired] += 1
            refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = len(sound) * p.dt / 1000.0
    return {"hz": total / secs / fb.n, "kc_set": set(np.flatnonzero(ever & kc).tolist()), "kc_frac": float((ever & kc).sum() / kc.sum()),
            "reads": {k: int(per[idx].sum()) for k, idx in READ.items()}, "fired_frac": float(ever.mean())}


def jac(a, b): return len(a & b) / max(1, len(a | b))


t0 = time.perf_counter(); rows = []
for s in SCALES:
    fb.wdata[:] = base_w * np.float32(s)
    dots = [trial(".", G.seed_for(SEED, "sweep", ".", r)) for r in range(3)]
    dashes = [trial("-", G.seed_for(SEED, "sweep", "-", r)) for r in range(3)]
    row = {"scale": s, "hz_per_cell": round(float(np.mean([t["hz"] for t in dots + dashes])), 2),
           "fired_frac": round(float(np.mean([t["fired_frac"] for t in dots + dashes])), 3),
           "kc_frac_active": round(float(np.mean([t["kc_frac"] for t in dots + dashes])), 3),
           "jaccard_dot_dot": round(float(np.mean([jac(dots[0]["kc_set"], dots[r]["kc_set"]) for r in (1, 2)])), 3),
           "jaccard_dot_dash": round(float(np.mean([jac(a["kc_set"], b["kc_set"]) for a in dots for b in dashes])), 3),
           "reads_dot": {k: int(np.mean([t["reads"][k] for t in dots])) for k in READ},
           "reads_dash": {k: int(np.mean([t["reads"][k] for t in dashes])) for k in READ}}
    rows.append(row)
    print(json.dumps(row), f"{time.perf_counter()-t0:.0f}s", flush=True)
fb.wdata[:] = base_w
out = {"chosen": {"scales": SCALES, "seed": SEED, "reps": 3, "target": "KC active 5-15%; jaccard(dot,dash) << jaccard(dot,dot); DNs respond"},
       "measured": rows}
(LAB / "results" / (os.environ.get("SWEEP_OUT") or "gain_sweep.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
(LAB / "sweep_tmp.npz").unlink(missing_ok=True)
