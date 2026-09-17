"""Grade 0b — the organ check on the CORRECTED answer key (ChatGPT chat's literature table, 2026-09-17 02:30 EDT; Shiu 2024,
Ache 2019, Zhou 2015, Turner 2008, the 2026 MaleCNS gustatory cell-typing). Replaces grade0_organs.py's rows 2, 5, 6.
Rows (input -> expected; pass = expected output >= max(1 Hz, 2x baseline), brain < 10 Hz/cell; sign tests as stated):
  1 sugar   LB3b+LB3c labellar sweet GRNs @100 Hz          -> MN9 up (also MN6, MN8)                       [Shiu 2024]
  2 water   LB3a @100 Hz                                    -> MN9 up                                        [Shiu 2024]
  3 bitter  LB1a-e @100 Hz together with sugar              -> MN9 LOWER than sugar alone                    [Shiu 2024]
  4 loom    LPLC2 alone / LC4 alone / both @60 Hz           -> DNp01 giant fibre up; DNp02 (LC4), DNp04/06 (LPLC2) [Ache 2019]
  5 sound   JO-A @120 Hz                                    -> DNp01 up (type-1 JO-A synapse onto GF)        [ChatGPT key]
  6 song    JO-B @120 Hz                                    -> AMMC second-order cells up (aPN1/AMMC-B1 not typed here: WEAK)
  7 odour   ORN_DM1 @40 Hz                                  -> DM1_lPN up; KC 5-10% active                   [Turner 2008]
  8 steer   ALL left-side ORNs @40 Hz (lateralised odour)   -> DNa02 left > DNa02 right                      [Rayshubskiy 2020]
  9 touch   JO-C/E/F @60 Hz (antennal deflection)           -> MDN (informational only, per key: do not fail on it)
 10 groom   JO-C/E/F @60 Hz                                 -> DNg11 / DNg07 / DNg08 (aBN/aDN not typed here: WEAK)
Not in this release under the key's names: aBN1/2, aDN1/2, aPN1/AMMC-B1, vPN1, pC1, TLA, APN2/3 -> marked NOT TESTABLE.
Writes results/grade0b_<tag>.json. CHOSEN: rates above, 300 ms pulse, 3 reps, seed 73.
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

GRAPH = Path(os.environ.get("G0_GRAPH", str(RUNTIME / "build" / "graph.npz"))); TAG = os.environ.get("G0_TAG", "v1")
REPS = int(os.environ.get("G0_REPS", 3)); SEED = 73; PULSE_MS, PRE_MS, TAIL_MS = 300.0, 50.0, 50.0
fb = FlyBrain(GRAPH); p = fb.p
ann = pd.read_feather(RUNTIME / "data" / "body-annotations.feather"); b2i = fb.body_to_i
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
types = fb.types.astype(str)
side = np.array([""] * fb.n, dtype=object)
for bid, sd in zip(ann["bodyId"].astype(int), ann["somaSide"].astype(str)):
    if bid in b2i and sd in ("L", "R"): side[b2i[bid]] = sd
T = lambda *names: np.flatnonzero(np.isin(types, names))
P = lambda prefix: np.flatnonzero(np.array([t.startswith(prefix) for t in types]))
kc = P("KC"); orn = P("ORN_")
dna02 = T("DNa02"); dna02L = dna02[side[dna02] == "L"]; dna02R = dna02[side[dna02] == "R"]
jocef = np.concatenate([P("JO-C"), P("JO-E"), P("JO-F")])
ROWS = [
    {"id": 1, "organ": "sugar LB3b+LB3c -> MN9", "drives": [(T("LB3b", "LB3c"), 100)], "read": {"MN9": T("MN9"), "MN6": T("MN6"), "MN8": T("MN8")}},
    {"id": 2, "organ": "water LB3a -> MN9", "drives": [(T("LB3a"), 100)], "read": {"MN9": T("MN9")}},
    {"id": 3, "organ": "bitter LB1a-e + sugar -> MN9 lower than sugar", "drives": [(T("LB3b", "LB3c"), 100), (T("LB1a", "LB1b", "LB1c", "LB1d", "LB1e"), 100)], "read": {"MN9": T("MN9")}, "compare_to": 1, "expect": "lower"},
    {"id": 4, "organ": "loom LPLC2 alone -> GF/takeoff DNs", "drives": [(T("LPLC2"), 60)], "read": {"DNp01": T("DNp01"), "DNp02": T("DNp02"), "DNp04": T("DNp04"), "DNp06": T("DNp06")}},
    {"id": 4.1, "organ": "loom LC4 alone -> GF/takeoff DNs", "drives": [(T("LC4"), 60)], "read": {"DNp01": T("DNp01"), "DNp02": T("DNp02"), "DNp04": T("DNp04"), "DNp06": T("DNp06")}},
    {"id": 4.2, "organ": "loom LPLC2+LC4 -> GF/takeoff DNs", "drives": [(T("LPLC2", "LC4"), 60)], "read": {"DNp01": T("DNp01"), "DNp02": T("DNp02"), "DNp04": T("DNp04"), "DNp06": T("DNp06")}},
    {"id": 5, "organ": "sound JO-A -> giant fibre DNp01", "drives": [(P("JO-A"), 120)], "read": {"DNp01": T("DNp01")}},
    {"id": 6, "organ": "song JO-B -> AMMC second-order (weak: aPN1 not typed)", "drives": [(P("JO-B"), 120)], "read": {"AMMC": P("AMMC")}, "weak": True},
    {"id": 7, "organ": "odour ORN_DM1 -> DM1_lPN, KC 5-10%", "drives": [(T("ORN_DM1"), 40)], "read": {"DM1_lPN": T("DM1_lPN"), "KC": kc}, "kc_target": (0.03, 0.15)},
    {"id": 8, "organ": "steer: left ORNs -> DNa02 L > R", "drives": [(orn[side[orn] == "L"], 40)], "read": {"DNa02_L": dna02L, "DNa02_R": dna02R}, "lateral": ("DNa02_L", "DNa02_R")},
    {"id": 9, "organ": "antennal deflection JO-C/E/F -> MDN (informational)", "drives": [(jocef, 60)], "read": {"MDN": T("MDN")}, "info": True},
    {"id": 10, "organ": "antennal JO-C/E/F -> grooming DNs (weak: aBN/aDN not typed)", "drives": [(jocef, 60)], "read": {"DNg11": T("DNg11"), "DNg07": T("DNg07"), "DNg08": T("DNg08")}, "weak": True},
]


def run(drives, seed, reads):
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    per = np.zeros(fb.n, dtype=np.int32); total = 0; ever = np.zeros(fb.n, bool)
    dl = [(np.asarray(c, dtype=np.int64), min(1.0, hz * p.dt / 1000.0)) for c, hz in drives if len(c)]
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            for cells, prob in dl:
                hit = cells[rng.random(len(cells)) < prob]
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
    out = {name: (float(per[idx].sum() / len(idx) / secs) if len(idx) else None) for name, idx in reads.items()}
    out["_brain_hz"] = total / secs / fb.n; out["_kc_frac"] = float(ever[kc].mean()); return out


t0 = time.perf_counter(); rows = []; by_id = {}
all_reads = {n: idx for r in ROWS for n, idx in r["read"].items()}
base = run([], 1, all_reads)
for r in ROWS:
    res = [run(r["drives"], int.from_bytes(hashlib.sha256(f"{SEED}:{r['id']}:{k}".encode()).digest()[:4], "little"), r["read"]) for k in range(REPS)]
    reads = {n: {"hz": (round(float(np.mean([x[n] for x in res])), 2) if res[0][n] is not None else None), "baseline_hz": round(base[n] or 0.0, 2), "cells": int(len(r["read"][n]))} for n in r["read"]}
    brain = float(np.mean([x["_brain_hz"] for x in res])); kcf = float(np.mean([x["_kc_frac"] for x in res]))
    ignited = brain > 10.0
    note = ""
    if "compare_to" in r:
        ref = by_id[r["compare_to"]]["reads"]["MN9"]["hz"]; mine = reads["MN9"]["hz"]
        ok = mine is not None and ref is not None and mine < ref; note = f"MN9 {mine} vs sugar-alone {ref}"
    elif "lateral" in r:
        a, b = r["lateral"]; ok = reads[a]["hz"] is not None and reads[b]["hz"] is not None and reads[a]["hz"] > reads[b]["hz"]; note = f"{a} {reads[a]['hz']} vs {b} {reads[b]['hz']}"
    else:
        ok = any(v["hz"] is not None and v["hz"] >= max(1.0, 2 * v["baseline_hz"]) for v in reads.values())
    if "kc_target" in r:
        lo, hi = r["kc_target"]; ok = ok and lo <= kcf <= hi; note = f"KC active {kcf*100:.1f}% (target {lo*100:.0f}-{hi*100:.0f}%)"
    if any(len(c) == 0 for c, _ in r["drives"]) or all(v["cells"] == 0 for v in reads.values()):
        verdict = "NOT TESTABLE"
    elif ignited:
        verdict = "IGNITES"
    elif r.get("info"):
        verdict = "INFO " + ("responds" if ok else "quiet")
    else:
        verdict = ("PASS" if ok else "FAIL") + (" (weak key)" if r.get("weak") else "")
    row = {"id": r["id"], "organ": r["organ"], "drive_cells": [int(len(c)) for c, _ in r["drives"]], "drive_hz": [hz for _, hz in r["drives"]], "reads": reads,
           "brain_hz_per_cell": round(brain, 2), "kc_frac_active": round(kcf, 3), "note": note, "verdict": verdict}
    rows.append(row); by_id[r["id"]] = row
    print(f"[{TAG}] {r['id']:>4} {r['organ']:58s} drive {row['drive_cells']}@{row['drive_hz']}  brain {brain:5.2f}  KC {kcf*100:4.1f}%  "
          + "  ".join(f"{n} {v['hz']}" for n, v in reads.items()) + f"  {note}  -> {verdict}  {time.perf_counter()-t0:.0f}s", flush=True)
out = {"chosen": {"graph": str(GRAPH), "pulse_ms": PULSE_MS, "reps": REPS, "seed": SEED, "answer_key": "ChatGPT chat literature table 2026-09-17 (inbox brief); Shiu 2024 numbers as soft targets"},
       "measured": rows, "not_testable_in_this_release": ["aBN1/2", "aDN1/2", "aPN1/AMMC-B1", "vPN1", "pC1", "TLA", "APN2/3"]}
(LAB / "results" / f"grade0b_{TAG}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
