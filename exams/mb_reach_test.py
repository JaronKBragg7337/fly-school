"""Does the mushroom body reach the descending neurons? (Claude Code, 2026-09-17 00:55 EDT)

Attempt 8 showed training changes nothing on readouts that can speak. Competing mechanism: the KC->MBON weights - the only
weights that learn - may not influence DN output under this stimulus at all. Test: same seeds, same sound, three brains:
  fresh   : all KC->MBON gains 1.0
  floor   : all KC->MBON gains at the floor (0.25) - the most any lesson could ever do
  silent  : KC->MBON gains 0 (MB output cut entirely; not reachable by the rule, a bound)
Measure per DN cell: spike counts per trial; report how many DN cells change at all, and how the four readouts' decodes
change. If floor == fresh on the readouts, the exam cannot be passed by learning in this circuit. MEASURED.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); sys.path.insert(0, str(LAB))
import morse_grade1 as G
from flysim import FlyBrain
from mushroom import MushroomBody
import calibration as C

fb = FlyBrain(G.RUNTIME / "build" / "graph.npz")
mb = MushroomBody(fb, calibration=C.CHOSEN, sides=G.RUNTIME / "build" / "mb_sides.json", store=LAB / "reach_tmp.npz", clock=lambda: 0.0)
dn = fb.where(superclass="descending")
jo = fb.where(type_re=r"^JO-A")
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
p = fb.p
READOUTS = {"DNa01": fb.where(type_re=r"^DNa01$"), "DNp17": fb.where(type_re=r"^DNp17$"),
            "DNge131": fb.where(type_re=r"^DNge131$"), "DNp30": fb.where(type_re=r"^DNp30$")}
PATTERNS = (".", "-", ".-", "-.")
SEEDS = (73, 7337, 101)
REPS = 3


def trial(pattern, seed):
    sound, pre_steps = G.make_timeline(pattern, p.dt)
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    bin_steps = G.ms_steps(G.BIN_MS, p.dt); nb = int(np.ceil(len(sound) / bin_steps))
    counts = np.zeros((fb.n, nb), dtype=np.int16)
    prob = min(1.0, G.JO_HZ * p.dt / 1000.0)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    mbon_spikes = 0; kc_spikes = 0
    mbon = mb.mbon; is_mbon = np.zeros(fb.n, bool); is_mbon[mbon] = True
    is_kc = np.zeros(fb.n, bool); is_kc[mb.kc] = True
    for step, on in enumerate(sound):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on:
            hit = jo[rng.random(len(jo)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            counts[fired, min(step // bin_steps, nb - 1)] += 1
            mbon_spikes += int(is_mbon[fired].sum()); kc_spikes += int(is_kc[fired].sum())
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    return counts, int(round(pre_steps / bin_steps)), kc_spikes, mbon_spikes


t0 = time.perf_counter()
out = {"chosen": {"seeds": SEEDS, "reps": REPS, "patterns": PATTERNS, "conditions": ["fresh", "floor", "silent"]}, "measured": {}}
per_cond = {}
for cond, g in (("fresh", 1.0), ("floor", float(mb.floor)), ("silent", 0.0)):
    mb.gain[:] = g; mb.apply()
    rows = []
    for seed in SEEDS:
        for pat in PATTERNS:
            for rep in range(REPS):
                counts, pre_bins, kcs, mbs = trial(pat, G.seed_for(seed, "reach", pat, rep))
                dec = {name: G.decode(counts[idx].sum(0), pre_bins)[0] for name, idx in READOUTS.items()}
                rows.append({"seed": seed, "pattern": pat, "rep": rep, "kc_spikes": kcs, "mbon_spikes": mbs,
                             "dn_total": int(counts[dn].sum()), "dn_counts": counts[dn].sum(1).astype(int).tolist(), "decode": dec})
    per_cond[cond] = rows
    print(f"{cond}: mean KC spikes {np.mean([r['kc_spikes'] for r in rows]):.0f}  MBON spikes {np.mean([r['mbon_spikes'] for r in rows]):.0f}  "
          f"DN spikes {np.mean([r['dn_total'] for r in rows]):.0f}  {time.perf_counter()-t0:.0f}s", flush=True)

fresh, floor, silent = per_cond["fresh"], per_cond["floor"], per_cond["silent"]
def diff(a, b):
    A = np.array([r["dn_counts"] for r in a]); B = np.array([r["dn_counts"] for r in b])
    d = np.abs(A - B).sum(0)
    return {"dn_cells_changed": int((d > 0).sum()), "of": int(A.shape[1]), "total_abs_diff": int(d.sum()),
            "total_spikes_fresh": int(A.sum()), "decodes_changed": int(sum(x["decode"] != y["decode"] for x, y in zip(a, b))),
            "of_trials": len(a),
            "per_readout_changed": {k: int(sum(x["decode"][k] != y["decode"][k] for x, y in zip(a, b))) for k in READOUTS}}
out["measured"] = {"fresh_vs_floor": diff(fresh, floor), "fresh_vs_silent": diff(fresh, silent),
                   "mean_mbon_spikes": {c: float(np.mean([r["mbon_spikes"] for r in per_cond[c]])) for c in per_cond},
                   "mean_kc_spikes": {c: float(np.mean([r["kc_spikes"] for r in per_cond[c]])) for c in per_cond},
                   "exact_by_cond": {c: {k: int(sum(r["decode"][k] == r["pattern"] for r in per_cond[c])) for k in READOUTS} for c in per_cond},
                   "trials_per_cond": len(fresh), "elapsed_s": round(time.perf_counter() - t0, 1)}
(LAB / "results" / "mb_reach_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["measured"], indent=1))
