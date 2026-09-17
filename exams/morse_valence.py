"""Grade 1 (valence): the exam the mushroom body can actually take. (Claude Code, 2026-09-17 01:05 EDT)

WHY: rows 1-8 + mb_reach_test.py showed Morse *production* from descending neurons is untouched by mushroom-body learning
(cutting MB output entirely leaves every readout's accuracy unchanged). The MB is wired to change VALENCE - approach vs
avoid MBON output - which is what the trader reads (calibration.readout / valence). So the school's Grade 1 becomes a
choice exam, the way real fly labs test learning (Tully & Quinn: odour + shock -> avoid the odour):

  teach : sound A (`.` rhythm) paired with reward dopamine; sound B (`-` rhythm) paired with punishment.  TEACH_EPOCHS x 2.
  exam  : cold, no dopamine - valence (approach Hz - avoid Hz) for A, B, and the never-heard `.-` and `-.`.
  pass line (stated before running): after training, valence(A) - valence(B) > 0 in most seeds and larger than before
          training; generalization question: does `.-` read closer to A or B?  Zeros are zeros.

Protocol constants identical to morse_grade1.py (JO-A carrier 120 Hz, dot 20 / dash 60 / gap 40 ms; membrane reset per
trial; MB weights persist within a seed, fresh per seed; frozen clock). Readout: MBON approach/avoid populations summed
over the whole trial, in Hz. CHOSEN: 12 teach epochs; 6 cold reps per sound; seeds 73, 7337, 101, 202, 303.
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

TEACH_EPOCHS = int(os.environ.get("VAL_EPOCHS", 12)); REPS = int(os.environ.get("VAL_REPS", 6))
SEEDS = tuple(int(s) for s in os.environ.get("VAL_SEEDS", "73,7337,101,202,303").split(","))
A, B = ".", "-"
SOUNDS = (A, B, ".-", "-.")

fb = FlyBrain(G.RUNTIME / "build" / "graph.npz")
jo = fb.where(type_re=r"^JO-A")
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
p = fb.p


def trial(mb, pattern, seed):
    sound, pre_steps = G.make_timeline(pattern, p.dt)
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    prob = min(1.0, G.JO_HZ * p.dt / 1000.0)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    kc_mask = np.zeros(fb.n, bool); kc_mask[mb.kc] = True
    app = np.zeros(fb.n, bool); app[mb.punish_side] = True      # approach MBONs (PPL1 side)
    avd = np.zeros(fb.n, bool); avd[mb.reward_side] = True      # avoid MBONs (PAM side)
    kc_seen = np.zeros(fb.n, bool); n_app = n_avd = 0
    for step, on in enumerate(sound):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on:
            hit = jo[rng.random(len(jo)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            kc_seen[fired[kc_mask[fired]]] = True
            n_app += int(app[fired].sum()); n_avd += int(avd[fired].sum())
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = len(sound) * p.dt / 1000.0
    app_hz = n_app / max(1, app.sum()) / secs; avd_hz = n_avd / max(1, avd.sum()) / secs
    return {"valence": app_hz - avd_hz, "approach_hz": app_hz, "avoid_hz": avd_hz, "kc": np.flatnonzero(kc_seen)}


def cold(mb, seed, phase):
    out = {}
    for s in SOUNDS:
        vals = [trial(mb, s, G.seed_for(seed, phase, s, r))["valence"] for r in range(REPS)]
        out[s] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "values": [round(x, 3) for x in vals]}
    return out


t0 = time.perf_counter(); seeds_out = []
for seed in SEEDS:
    store = LAB / f"morse_valence_seed_{seed}.npz"
    if store.exists(): store.unlink()
    mb = MushroomBody(fb, calibration=C.CHOSEN, sides=G.RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    pre = cold(mb, seed, "pre")
    teach = []
    for epoch in range(TEACH_EPOCHS):
        for s, val in ((A, +1), (B, -1)):
            r = trial(mb, s, G.seed_for(seed, "teach", s, epoch))
            mb.forget_trace(); mb.observe(r["kc"]); hit = int(mb.dopamine(val, 1.0)); mb.apply()
            teach.append({"epoch": epoch, "sound": s, "dopamine": val, "synapses_hit": hit, "valence_during": round(r["valence"], 3)})
    post = cold(mb, seed, "post")
    sep_pre = pre[A]["mean"] - pre[B]["mean"]; sep_post = post[A]["mean"] - post[B]["mean"]
    seeds_out.append({"seed": seed, "pre": pre, "post": post, "teach": teach, "mb": mb.stats(),
                      "separation_pre": round(sep_pre, 3), "separation_post": round(sep_post, 3), "learned": bool(sep_post > sep_pre and sep_post > 0),
                      "generalization_post": {".-": round(post[".-"]["mean"], 3), "-.": round(post["-."]["mean"], 3)}})
    print(f"seed {seed}: valence A/B pre {pre[A]['mean']:+.2f}/{pre[B]['mean']:+.2f} -> post {post[A]['mean']:+.2f}/{post[B]['mean']:+.2f}  "
          f"sep {sep_pre:+.2f} -> {sep_post:+.2f}  .-={post['.-']['mean']:+.2f} -.={post['-.']['mean']:+.2f}  mb gain {mb.stats()['mean_gain']}  {time.perf_counter()-t0:.0f}s", flush=True)
    store.unlink(missing_ok=True)

n_learned = sum(s["learned"] for s in seeds_out)
out = {"experiment": "Grade 1 (valence): A=. rewarded, B=- punished; cold valence readout; generalization to .- and -.",
       "chosen": {"teach_epochs": TEACH_EPOCHS, "reps": REPS, "seeds": SEEDS, "A": A, "B": B,
                  "readout": "MBON approach (PPL1-side) Hz minus avoid (PAM-side) Hz over the trial",
                  "pass_line": "sep_post > sep_pre and sep_post > 0 in a majority of seeds"},
       "measured": {"seeds_learned": n_learned, "of": len(SEEDS),
                    "mean_sep_pre": round(float(np.mean([s["separation_pre"] for s in seeds_out])), 3),
                    "mean_sep_post": round(float(np.mean([s["separation_post"] for s in seeds_out])), 3),
                    "pass": bool(n_learned > len(SEEDS) / 2), "elapsed_s": round(time.perf_counter() - t0, 1)},
       "seeds": seeds_out}
(LAB / "results" / (os.environ.get("VAL_OUT") or "morse_valence.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["measured"], indent=1))
