"""Gate 1 driver: runs sender, world and receiver as separate OS processes and scores gate1_protocol.md exactly. (Claude Code, 2026-09-17)
usage: python gate1.py [tag]   (env GATE_DEVICE=cuda|cpu, GATE_SEEDS=comma list to override the predeclared seeds)
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; PY = sys.executable
TAG = sys.argv[1] if len(sys.argv) > 1 else "gate1"
SEEDS = [int(s) for s in os.environ.get("GATE_SEEDS", "20101,20202,20303,20404,20505,20606,20707,20808").split(",")]
RES = HERE / "results" / TAG; S1, S0, SCH, RCV = RES / "sender_driven", RES / "sender_silent", RES / "schedules", RES / "receiver"
for d in (S1, S0, SCH, RCV): d.mkdir(parents=True, exist_ok=True)
seeds_arg = ",".join(map(str, SEEDS)); t0 = time.perf_counter()


def run(*args):
    print(">>", " ".join(str(a) for a in args[1:]), flush=True)
    subprocess.run([PY, *[str(a) for a in args]], check=True)


# 1. the two senders (driven, silent) - each its own process, sees nothing but its own output directory
run(HERE / "sender.py", seeds_arg, "1", S1)
run(HERE / "sender.py", seeds_arg, "0", S0)
# 2. the world, per seed and condition
for s in SEEDS:
    run(HERE / "world.py", s, S1, "live", SCH / f"song_{s}_LIVE.npz")
    run(HERE / "world.py", s, S1, "mute", SCH / f"song_{s}_MUTE.npz")
    run(HERE / "world.py", s, S0, "live", SCH / f"song_{s}_SILENT.npz")
    run(HERE / "world.py", s, S0, f"replay:{SCH / f'song_{s}_LIVE.npz'}", SCH / f"song_{s}_SCRAMBLE.npz")
    run(HERE / "world.py", s, S0, "mute", SCH / f"song_{s}_ZERO.npz")
# 3. the receiver - one process per condition, batched over seeds; it reads schedules only
for cond in ("LIVE", "MUTE", "SILENT", "SCRAMBLE", "ZERO"):
    run(HERE / "receiver.py", seeds_arg, SCH, cond, RCV)

# 4. score - exactly gate1_protocol.md
def sinfo(d, s): return json.loads(str(np.load(d / f"sender_{s}.npz")["info"]))
def rinfo(s, c): z = np.load(RCV / f"receiver_{s}_{c}.npz"); return json.loads(str(z["info"])), z["counts"]
rows = []
for s in SEEDS:
    sd, ss = sinfo(S1, s), sinfo(S0, s)
    R = {c: rinfo(s, c) for c in ("LIVE", "MUTE", "SILENT", "SCRAMBLE", "ZERO")}
    row = {"seed": s, "sender_pulse_mn_hz": [sd["pulse_song_mn_hz"], ss["pulse_song_mn_hz"]], "sender_wing_all_hz": [sd["wing_mn_hz_all"], ss["wing_mn_hz_all"]],
           "sender_song_pattern_hz": sd["song_pattern_hz"], "sender_brain_hz": [sd["brain_hz"], ss["brain_hz"]],
           "song_on_ms": {c: R[c][0]["song_on_ms"] for c in R}, "ammc_hz": {c: R[c][0]["ammc_hz"] for c in R}, "gf_hz": {c: R[c][0]["gf_hz"] for c in R},
           "receiver_brain_hz": {c: R[c][0]["brain_hz"] for c in R},
           "S1_sings": sd["pulse_song_mn_hz"] >= max(1.0, 2 * ss["pulse_song_mn_hz"]),
           "S2_no_ignition": all(x < 10 for x in (sd["brain_hz"], ss["brain_hz"], *[R[c][0]["brain_hz"] for c in R])),
           "R1a_live_gt_mute": R["LIVE"][0]["ammc_hz"] > R["MUTE"][0]["ammc_hz"],
           "R1b_live_ge_2x": R["LIVE"][0]["ammc_hz"] >= max(1.0, 2 * R["MUTE"][0]["ammc_hz"]),
           "R2_mute_identical_to_zero": bool(np.array_equal(R["MUTE"][1], R["ZERO"][1])),
           "R3_scramble_identical_to_live": bool(np.array_equal(R["SCRAMBLE"][1], R["LIVE"][1])),
           "R4_silent_le_1p1_mute": R["SILENT"][0]["ammc_hz"] <= 1.1 * R["MUTE"][0]["ammc_hz"] + 1e-9}
    rows.append(row); print(json.dumps(row), flush=True)
n = len(SEEDS); cnt = lambda k: sum(r[k] for r in rows)
crit = {"S1_sings_ge_7of8": cnt("S1_sings") >= 7, "S2_no_ignition_all": cnt("S2_no_ignition") == n,
        "R1_live_gt_mute_ge_7of8_and_2x_ge_6of8": cnt("R1a_live_gt_mute") >= 7 and cnt("R1b_live_ge_2x") >= 6,
        "R2_mute_identical_all": cnt("R2_mute_identical_to_zero") == n, "R3_scramble_identical_all": cnt("R3_scramble_identical_to_live") == n,
        "R4_silent_quiet_ge_7of8": cnt("R4_silent_le_1p1_mute") >= 7}
verdict = "PASS" if all(crit.values()) else "FAIL"
summary = {"tag": TAG, "protocol": "gate1_protocol.md (predeclared 2026-09-17 12:25 EDT)", "seeds": SEEDS, "device": os.environ.get("GATE_DEVICE", "cuda"),
           "counts": {k: cnt(k) for k in ("S1_sings", "S2_no_ignition", "R1a_live_gt_mute", "R1b_live_ge_2x", "R2_mute_identical_to_zero", "R3_scramble_identical_to_live", "R4_silent_le_1p1_mute")},
           "criteria": crit, "verdict": verdict, "rows": rows, "wall_s": round(time.perf_counter() - t0)}
(RES / "gate1_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({"verdict": verdict, "criteria": crit, "counts": summary["counts"], "wall_s": summary["wall_s"]}), flush=True)
