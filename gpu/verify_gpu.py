"""Verify the GPU kernel against the CPU kernel on the frozen fly-v10 stack (graph_v9 + calib). (Claude Code, 2026-09-17)
Same seeds, same Poisson hits (drawn identically), same maths. Reports, per Grade 0b organ row: per-neuron spike-count agreement,
read-set rates on both kernels, whole-brain rate; then the speed of 1 fly vs a batch of 8 on the GPU vs 1 fly on the CPU.
Writes results/gpu_verify.json. The GPU kernel is trusted only if every organ verdict is unchanged and rates agree within noise.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v3"); sys.path.insert(0, str(LAB / "gpu"))
from flysim_v3 import FlyBrainV3 as FlyBrain
import calibration as C
from flysim_gpu import GPUBrain, rates

GRAPH = Path(os.environ.get("VG_GRAPH", r"C:/Users/lilli/Fly-Lab/versions/fly-v9/graph_v9.npz"))
CALIB = os.environ.get("VG_CALIB", r"C:/Users/lilli/Fly-Lab/versions/fly-v10/calib.json")
PULSE_MS, PRE_MS, TAIL_MS = 300.0, 50.0, 50.0

fb = FlyBrain(GRAPH); p = fb.p; types = fb.types.astype(str)
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
THRESH = np.full(fb.n, p.v_thresh, dtype=np.float32)
_c = json.load(open(CALIB))
gpn[np.array([("PN" in x and not x.startswith("MBON")) for x in types])] *= np.float32(_c["pn"]); gpn[types == "APL"] *= np.float32(_c["apl"])
gpn[np.array([x.startswith("KC") for x in types])] *= np.float32(_c["kc"]); gpn[np.array([x.startswith("MBON") for x in types])] *= np.float32(_c["mbon"])
THRESH[np.array([x.startswith("KC") for x in types])] += np.float32(_c["kc_thresh_shift_mv"])
T = lambda *names: np.flatnonzero(np.isin(types, names))
P = lambda prefix: np.flatnonzero(np.array([t.startswith(prefix) for t in types]))
kc = P("KC")
ROWS = [("sugar LB3b+LB3c", [(T("LB3b", "LB3c"), 100)], {"MN9": T("MN9")}),
        ("loom LPLC2+LC4", [(T("LPLC2", "LC4"), 60)], {"DNp01": T("DNp01"), "DNp02": T("DNp02")}),
        ("sound JO-A", [(P("JO-A"), 120)], {"DNp01": T("DNp01")}),
        ("odour ORN_DM1", [(T("ORN_DM1"), 40)], {"DM1_lPN": T("DM1_lPN"), "KC": kc}),
        ("odour ORN_DA2 (test-9 A)", [(T("ORN_DA2"), 40)], {"KC": kc, "MBON09": T("MBON09"), "PAM5": T("MBON09", "MBON01", "MBON05", "MBON03", "MBON06")})]


def cpu_run(drives, seed):
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    per = np.zeros(fb.n, dtype=np.int32)
    dl = [(np.asarray(c, dtype=np.int64), min(1.0, hz * p.dt / 1000.0)) for c, hz in drives if len(c)]
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            for cells, prob in dl:
                hit = cells[rng.random(len(cells)) < prob]
                if len(hit): v[hit] = THRESH[hit] + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= THRESH) & (refr <= 0))
        if len(fired):
            per[fired] += 1; refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    return per, steps * p.dt / 1000.0


gb = GPUBrain(GRAPH); gb.set_gain(gpn); gb.set_thresh(THRESH)
assert np.allclose(gb.wdata_csr[gb.csc_perm], fb.wdata), "CSC->CSR permutation does not reproduce fb.wdata"
print(f"graph {GRAPH.name}  n {fb.n}  nnz {gb.nnz}  device {gb.dev}", flush=True)
out = {"graph": str(GRAPH), "calib": CALIB, "rows": []}
for name, drives, reads in ROWS:
    seed = int.from_bytes(hashlib.sha256(f"verify:{name}".encode()).digest()[:4], "little")
    if len(drives) > 1: raise SystemExit("verify rows use one drive so the rng order is identical")
    t0 = time.perf_counter(); per_cpu, secs = cpu_run(drives, seed); t_cpu = time.perf_counter() - t0
    t0 = time.perf_counter(); g = gb.trial([(drives[0][0], drives[0][1])], [seed], pre_ms=PRE_MS, pulse_ms=PULSE_MS, tail_ms=TAIL_MS); t_gpu = time.perf_counter() - t0
    per_gpu = g["counts"][:, 0]
    same = int((per_cpu == per_gpu).sum()); diff = int((per_cpu != per_gpu).sum())
    row = {"row": name, "seed": seed, "brain_hz_cpu": round(float(per_cpu.sum() / secs / fb.n), 4), "brain_hz_gpu": round(float(g["hz"][0]), 4),
           "neurons_identical": same, "neurons_differ": diff, "spikes_cpu": int(per_cpu.sum()), "spikes_gpu": int(per_gpu.sum()),
           "reads": {k: [round(float(per_cpu[idx].sum() / max(1, len(idx)) / secs), 3), round(float(rates(g, idx)[0]), 3)] for k, idx in reads.items()},
           "kc_frac": [round(float((per_cpu[kc] > 0).mean()), 4), round(float((per_gpu[kc] > 0).mean()), 4)],
           "t_cpu_s": round(t_cpu, 1), "t_gpu_s": round(t_gpu, 1)}
    out["rows"].append(row); print(json.dumps(row), flush=True)
# speed: batch of 8 flies on the GPU
drives = ROWS[4][1]; seeds = [1000 + i for i in range(8)]
t0 = time.perf_counter(); g8 = gb.trial([(drives[0][0], drives[0][1])], seeds, pre_ms=PRE_MS, pulse_ms=PULSE_MS, tail_ms=TAIL_MS); t8 = time.perf_counter() - t0
out["batch8"] = {"t_gpu_s": round(t8, 1), "per_fly_s": round(t8 / 8, 2), "brain_hz": [round(float(x), 3) for x in g8["hz"]], "kc_frac": [round(float((g8["counts"][kc, b] > 0).mean()), 4) for b in range(8)]}
print(json.dumps(out["batch8"]), flush=True)
(LAB / "results" / "gpu_verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
