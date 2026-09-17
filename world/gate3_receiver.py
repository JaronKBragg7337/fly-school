"""Gate 3 receiver process: an untrained fly whose cVA neurons (ORN_DA1) are driven at the rate the world file says. (Claude Code, 2026-09-17)
usage: python gate3_receiver.py <seed> <world_path> <out_path>
For each condition (LIVE, MUTE), odour X (A, C) and phase (pre, post): 24 realisations, ORN_DA1 Poisson at the world's per-trial rate.
Receiver rng seed = sha256("gate3r:<seed>:<X>:<rep>") - the SAME for pre and post and for LIVE and MUTE, drawn every step, so the
comparisons are paired and the MUTE identity check is exact. Never reads the sender. Reports ORN_DA2 / ORN_VM5d spike counts (must be 0).
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import stack
seed = int(sys.argv[1]); w = json.loads(Path(sys.argv[2]).read_text()); out = Path(sys.argv[3])
t0 = time.perf_counter(); gb = stack.load(); types = gb.types
orn = gb.where("ORN_DA1"); vpn = gb.where("DA1_vPN"); lpn = gb.where("DA1_lPN"); pc1 = np.flatnonzero(np.array([t.startswith("pC1") for t in types]))
leak = gb.where("ORN_DA2", "ORN_VM5d")
import os; REPS = int(os.environ.get("GATE_REPS", "24"))
def rseed(X, rep): return int.from_bytes(hashlib.sha256(f"gate3r:{seed}:{X}:{rep}".encode()).digest()[:4], "little")
def block(rates_hz, X):
    probs = np.array([min(1.0, hz * gb.dt / 1000.0) for hz in rates_hz]); rngs = [np.random.default_rng(rseed(X, k)) for k in range(REPS)]
    on0, on1 = int(round(20.0 / gb.dt)), int(round(320.0 / gb.dt))
    def ear(step):
        u = np.stack([rng.random(len(orn)) for rng in rngs], axis=1)          # drawn every step, every fly (pairing)
        if not (on0 <= step < on1): return None
        return orn, u < probs[None, :]
    r = gb.trial([], [rseed(X, k) for k in range(REPS)], pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0, external=ear)
    return {"vPN": [round(float(x), 3) for x in stack.rates(r, vpn)], "lPN": [round(float(x), 3) for x in stack.rates(r, lpn)], "pC1": [round(float(x), 3) for x in stack.rates(r, pc1)],
            "orn_hz_in": [round(float(h), 3) for h in rates_hz], "leak_spikes": int(r["counts"][leak].sum()), "brain_hz": [round(float(x), 3) for x in r["hz"]],
            "_counts": r["counts"]}
res = {"seed": seed, "receiver_rng": "sha256(gate3r:seed:X:rep)", "conds": {}}
ident = {}
for cond in ("LIVE", "MUTE"):
    res["conds"][cond] = {}
    for X in ("A", "C"):
        b = {ph: block([f["orn_hz"] for f in w[cond][X][ph]], X) for ph in ("pre", "post")}
        ident[(cond, X)] = bool(np.array_equal(b["pre"]["_counts"], b["post"]["_counts"]))
        spike_diff = int(np.abs(b["pre"]["_counts"].astype(np.int64) - b["post"]["_counts"].astype(np.int64)).sum())
        readouts_same = all(b["pre"][k] == b["post"][k] for k in ("vPN", "lPN", "pC1", "brain_hz"))
        trials_identical = int((b["pre"]["_counts"] == b["post"]["_counts"]).all(0).sum())
        mean_close = all(abs(np.mean(b["post"][k]) - np.mean(b["pre"][k])) < 0.01 * max(1e-9, np.mean(b["pre"][k])) for k in ("vPN", "lPN"))
        res["conds"][cond][X] = {ph: {k: v for k, v in b[ph].items() if k != "_counts"} for ph in b}
        d = np.array(b["post"]["vPN"]) - np.array(b["pre"]["vPN"]); z = d.mean() / max(1e-9, d.std(ddof=1) / np.sqrt(REPS))
        res["conds"][cond][X]["vPN_mean_pre_post"] = [round(float(np.mean(b["pre"]["vPN"])), 3), round(float(np.mean(b["post"]["vPN"])), 3)]
        res["conds"][cond][X]["vPN_paired_z"] = round(float(z), 3); res["conds"][cond][X]["identical_pre_post"] = ident[(cond, X)]
        res["conds"][cond][X]["spike_diff_pre_post"] = spike_diff; res["conds"][cond][X]["readouts_identical_pre_post"] = bool(readouts_same)
        res["conds"][cond][X]["trials_identical_pre_post"] = trials_identical; res["conds"][cond][X]["means_within_1pct"] = bool(mean_close)
        print(json.dumps({"seed": seed, "cond": cond, "X": X, "vPN": res["conds"][cond][X]["vPN_mean_pre_post"], "z": res["conds"][cond][X]["vPN_paired_z"], "identical": ident[(cond, X)],
                          "leak": [b["pre"]["leak_spikes"], b["post"]["leak_spikes"]]}), flush=True)
res["provenance"] = stack.provenance(); res["wall_s"] = round(time.perf_counter() - t0)
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
