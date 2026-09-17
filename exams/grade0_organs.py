"""Grade 0 — does each organ do its job? (Claude Code, 2026-09-17 02:15 EDT; Jaron: "have we tested eyes being eyes…")

No learning. Each row: a sense population is driven for 300 ms; a named output population from the fly literature is
read; pass = the output rises above its silent baseline while the whole brain does not ignite. Run on fly-v1 and fly-v2.
Answer key (types as named in MaleCNS v1.0; papers in the ChatGPT brief, to be checked against this file):
  mouth/hands : tarsal taste GRNs (claw_tpGRN)          -> proboscis motor neuron MN9 (Shiu 2024: sugar -> proboscis extension)
  feet (hind) : T3 leg bristle neurons                   -> MDN, backward walking (Bidaye 2014; Mineplix screen)
  feet (front): T1 leg bristle neurons                   -> DNg07/DNg08, front-leg grooming (Seeds 2014; Hampel 2015)
  eyes        : LPLC2 + LC4 loom-sensitive VPNs          -> DNp01 giant fibre, DNp02/DNp04 takeoff (von Reyn 2014; Ache 2019)
  ears (wind) : JO-C/JO-E Johnston's organ               -> MDN (wind-evoked backing; Mineplix screen)
  ears (song) : JO-A                                     -> pIP10 / DNp13 (courtship pathway; weak expectation)
  nose        : ORN_DL3                                  -> Kenyon cells sparse (5-15%), DNa02 steering (Mineplix screen)
CHOSEN: 300 ms pulse, drive rate per row (DRIVE_HZ), 3 reps, seed 73; pass rule stated above. MEASURED: everything else.
Writes results/grade0_<graph>.json.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain
import calibration as C

GRAPH = Path(os.environ.get("G0_GRAPH", str(RUNTIME / "build" / "graph.npz")))
TAG = os.environ.get("G0_TAG", "v1"); REPS = int(os.environ.get("G0_REPS", 3)); SEED = 73
PULSE_MS, PRE_MS, TAIL_MS = 300.0, 50.0, 50.0
fb = FlyBrain(GRAPH); p = fb.p
ann = pd.read_feather(RUNTIME / "data" / "body-annotations.feather")
b2i = fb.body_to_i
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
types = fb.types.astype(str)


def by_type(*names):
    return np.flatnonzero(np.isin(types, names))


def bristles(neuromere):
    nerve = {"T1": "ProLN", "T2": "MesoLN", "T3": "MetaLN"}[neuromere]
    s = ann[(ann["superclass"].astype(str).str.contains("sensory", case=False)) & (ann["subclass"].astype(str).str.contains("bristle"))
            & (ann["entryNerve"].astype(str) == nerve)]
    return np.array([b2i[int(b)] for b in s["bodyId"] if int(b) in b2i], dtype=np.int64)


kc = np.flatnonzero(np.array([t.startswith("KC") for t in types]))
ORGANS = [
    {"organ": "mouth/hands: tarsal taste -> proboscis", "drive": by_type("claw_tpGRN"), "hz": 60, "read": {"MN9": by_type("MN9"), "MN1": by_type("MN1")}},
    {"organ": "feet (hind): T3 bristles -> MDN backing", "drive": bristles("T3"), "hz": 20, "read": {"MDN": by_type("MDN"), "DNa01": by_type("DNa01")}},
    {"organ": "feet (front): T1 bristles -> grooming DNs", "drive": bristles("T1"), "hz": 20, "read": {"DNg07": by_type("DNg07"), "DNg08": by_type("DNg08")}},
    {"organ": "eyes: loom (LPLC2+LC4) -> giant fibre / takeoff", "drive": by_type("LPLC2", "LC4"), "hz": 60, "read": {"DNp01": by_type("DNp01"), "DNp02+04": by_type("DNp02", "DNp04")}},
    {"organ": "ears (wind): JO-C/E -> MDN", "drive": np.flatnonzero(np.array([t.startswith("JO-C") or t.startswith("JO-E") for t in types])), "hz": 60, "read": {"MDN": by_type("MDN")}},
    {"organ": "ears (song): JO-A -> courtship pIP10/DNp13", "drive": np.flatnonzero(np.array([t.startswith("JO-A") for t in types])), "hz": 120, "read": {"pIP10": by_type("pIP10"), "DNp13": by_type("DNp13")}},
    {"organ": "nose: ORN_DL3 -> KC sparse + steering", "drive": by_type("ORN_DL3"), "hz": 40, "read": {"KC": kc, "DNa02": by_type("DNa02")}},
]


def run(drive, hz, seed, reads):
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed); prob = min(1.0, hz * p.dt / 1000.0)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    per = np.zeros(fb.n, dtype=np.int32); total = 0; ever = np.zeros(fb.n, bool)
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1 and len(drive):
            hit = drive[rng.random(len(drive)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            total += len(fired); per[fired] += 1; ever[fired] = True; refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = steps * p.dt / 1000.0
    out = {name: (float(per[idx].sum() / max(1, len(idx)) / secs) if len(idx) else None) for name, idx in reads.items()}
    out["_brain_hz"] = total / secs / fb.n
    out["_kc_frac"] = float(ever[kc].mean())
    return out


t0 = time.perf_counter(); rows = []
baseline = run(np.array([], dtype=np.int64), 0, 1, {name: idx for o in ORGANS for name, idx in o["read"].items()})
for o in ORGANS:
    res = [run(o["drive"], o["hz"], int.from_bytes(hashlib.sha256(f"{SEED}:{o['organ']}:{r}".encode()).digest()[:4], "little"), o["read"]) for r in range(REPS)]
    reads = {}
    for name in o["read"]:
        vals = [r[name] for r in res if r[name] is not None]
        reads[name] = {"hz": round(float(np.mean(vals)), 2) if vals else None, "baseline_hz": round(baseline[name] or 0.0, 2), "cells": int(len(o["read"][name]))}
    brain = float(np.mean([r["_brain_hz"] for r in res])); kcf = float(np.mean([r["_kc_frac"] for r in res]))
    responded = {n: (v["hz"] is not None and v["hz"] >= max(1.0, 2 * v["baseline_hz"])) for n, v in reads.items()}
    ignited = brain > 10.0
    verdict = "PASS" if any(responded.values()) and not ignited else ("IGNITES" if ignited else "SILENT")
    row = {"organ": o["organ"], "drive_cells": int(len(o["drive"])), "drive_hz": o["hz"], "reads": reads, "brain_hz_per_cell": round(brain, 2),
           "kc_frac_active": round(kcf, 3), "responded": responded, "verdict": verdict}
    rows.append(row)
    print(f"[{TAG}] {o['organ']:48s} drive {len(o['drive']):5d}@{o['hz']:3d}Hz  brain {brain:5.2f} Hz/cell  KC {kcf*100:4.1f}%  "
          + "  ".join(f"{n} {v['hz']}/{v['baseline_hz']}" for n, v in reads.items()) + f"  -> {verdict}  {time.perf_counter()-t0:.0f}s", flush=True)
out = {"chosen": {"graph": str(GRAPH), "pulse_ms": PULSE_MS, "reps": REPS, "seed": SEED, "pass_rule": "any named readout >= max(1 Hz, 2x baseline) and brain < 10 Hz/cell"},
       "measured": rows}
(LAB / "results" / f"grade0_{TAG}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
