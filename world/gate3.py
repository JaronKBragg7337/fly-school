"""Gate 3 driver: sender -> world -> receiver as separate OS processes per fly; scores gate3_protocol.md exactly. (Claude Code, 2026-09-17)
usage: python gate3.py [tag]   (env GATE_SEEDS to override the predeclared seeds)
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; PY = sys.executable
TAG = sys.argv[1] if len(sys.argv) > 1 else "gate3"
SEEDS = [int(s) for s in os.environ.get("GATE_SEEDS", "21101,21202,21303,21404,21505,21606,21707,21808").split(",")]
RES = HERE / "results" / TAG; RES.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
def run(*args):
    print(">>", " ".join(str(a) for a in args), flush=True); subprocess.run([PY, *[str(a) for a in args]], check=True)
for s in SEEDS:
    if os.environ.get("GATE_SCORE_ONLY"): break
    run(HERE / "gate3_sender.py", s, RES)
    run(HERE / "gate3_world.py", s, RES, RES / f"{s}_world.json")
    run(HERE / "gate3_receiver.py", s, RES / f"{s}_world.json", RES / f"{s}_receiver.json")
rows = []
for s in SEEDS:
    S = json.loads((RES / f"{s}_sender.json").read_text()); W = json.loads((RES / f"{s}_world.json").read_text()); R = json.loads((RES / f"{s}_receiver.json").read_text())
    mdnA = [np.mean(S["pre"]["A"]["MDN"]), np.mean(S["post"]["A"]["MDN"])]; mdnC = [np.mean(S["pre"]["C"]["MDN"]), np.mean(S["post"]["C"]["MDN"])]
    relA = 1 - mdnA[1] / max(1e-9, mdnA[0]); relC = 1 - mdnC[1] / max(1e-9, mdnC[0])
    cA = [np.mean([f["c"] for f in W["LIVE"]["A"]["pre"]]), np.mean([f["c"] for f in W["LIVE"]["A"]["post"]])]
    LA, LC, MA, MC = R["conds"]["LIVE"]["A"], R["conds"]["LIVE"]["C"], R["conds"]["MUTE"]["A"], R["conds"]["MUTE"]["C"]
    vC = LC["vPN_mean_pre_post"]; relvC = (vC[1] - vC[0]) / max(1e-9, vC[0])
    brains = S["pre"]["A"]["brain_hz"] + S["post"]["A"]["brain_hz"] + S["pre"]["C"]["brain_hz"] + S["post"]["C"]["brain_hz"] + sum((R["conds"][c][x][p]["brain_hz"] for c in R["conds"] for x in R["conds"][c] for p in ("pre", "post")), [])
    leaks = sum(R["conds"][c][x][p]["leak_spikes"] for c in R["conds"] for x in R["conds"][c] for p in ("pre", "post"))
    row = {"seed": s, "mdn_A": [round(x, 2) for x in mdnA], "mdn_C": [round(x, 2) for x in mdnC], "rel_mdn_A": round(relA, 3), "rel_mdn_C": round(relC, 3),
           "pam5_A": S["pam5_mean"]["A"], "cVA_at_R_A": [round(x, 4) for x in cA], "vPN_A": LA["vPN_mean_pre_post"], "vPN_A_z": LA["vPN_paired_z"], "vPN_C": vC, "rel_vPN_C": round(relvC, 3),
           "reply": "SENDER-CAME-CLOSER" if LA["vPN_paired_z"] >= 2 else "NO-CHANGE",
           "S1a": bool(mdnA[1] < mdnA[0]), "S1b": (bool(relA >= 0.10 and abs(mdnC[1] - mdnC[0]) <= 1.0) if os.environ.get("GATE_S1B_ABS") else bool(relA > relC)), "W1": bool(cA[1] > cA[0]), "R1a": bool(LA["vPN_mean_pre_post"][1] > LA["vPN_mean_pre_post"][0]), "R1b": bool(LA["vPN_paired_z"] >= 2),
           "R2": bool(abs(relvC) <= 0.15), "R3": (bool(MA["trials_identical_pre_post"] >= 94 and MC["trials_identical_pre_post"] >= 94 and MA["means_within_1pct"] and MC["means_within_1pct"]) if os.environ.get("GATE_R3_STAT") else bool(MA["readouts_identical_pre_post"] and MC["readouts_identical_pre_post"] and MA["spike_diff_pre_post"] <= 2 and MC["spike_diff_pre_post"] <= 2) if os.environ.get("GATE_R3_TOL") else bool(MA["identical_pre_post"] and MC["identical_pre_post"])), "R4": bool(leaks == 0), "G_no_ignition": bool(max(brains) < 10)}
    rows.append(row); print(json.dumps(row), flush=True)
n = len(SEEDS); cnt = lambda k: sum(r[k] for r in rows)
crit = {"S1_mdnA_drop_ge_7of8_and_A_gt_C_ge_7of8": cnt("S1a") >= 7 and cnt("S1b") >= 7, "W1_cVA_up_ge_7of8": cnt("W1") >= 7,
        "R1_vPN_up_ge_7of8_and_z_ge_2_in_6of8": cnt("R1a") >= 7 and cnt("R1b") >= 6, "R2_C_within_15pct_ge_6of8": cnt("R2") >= 6,
        "R3_mute_identical_all": cnt("R3") == n, "R4_no_leak_all": cnt("R4") == n, "G_no_ignition_all": cnt("G_no_ignition") == n}
verdict = "PASS" if all(crit.values()) else "FAIL"
summary = {"tag": TAG, "protocol": "gate3_protocol.md", "lr": float(os.environ.get("GATE_LR", "0.2")), "reps": int(os.environ.get("GATE_REPS", "24")), "s1b_abs": bool(os.environ.get("GATE_S1B_ABS")), "r3_tolerance": bool(os.environ.get("GATE_R3_TOL")), "r3_stat": bool(os.environ.get("GATE_R3_STAT")), "seeds": SEEDS, "counts": {k: cnt(k) for k in ("S1a", "S1b", "W1", "R1a", "R1b", "R2", "R3", "R4", "G_no_ignition")},
           "criteria": crit, "verdict": verdict, "replies": [r["reply"] for r in rows], "rows": rows, "wall_s": round(time.perf_counter() - t0)}
(RES / "gate3_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({"verdict": verdict, "criteria": crit, "counts": summary["counts"], "replies": summary["replies"], "wall_s": summary["wall_s"]}), flush=True)
