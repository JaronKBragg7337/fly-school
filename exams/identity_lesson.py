"""The identity lesson - the exam the mushroom body is built for. (Claude Code, 2026-09-17 02:20 EDT)

Two odours, the fly's native mushroom-body input (ORN -> PN -> KC): A = one glomerulus's receptor neurons, B = another.
teach: A + reward dopamine, B + punishment, TEACH_EPOCHS x 2.   exam: cold (no dopamine) valence = approach-MBON Hz minus
avoid-MBON Hz, for A, B, and the never-paired mixture A+B.   Pass line, stated before running: after training, valence(A)
minus valence(B) is larger than before training and positive, in a majority of seeds. KC identity check: Jaccard(A,B) must
be well below Jaccard(A,A) - if it is not, the lesson is unanswerable (as Morse was) and we say so.
Brain: fly-v2 by default (IDENT_GRAPH). Odour pulse 300 ms at ODOUR_HZ on every receptor neuron of the glomerulus (CHOSEN).
Constants CHOSEN; everything measured. Writes results/identity_lesson.json. Store persists per seed only (fresh fly per seed).
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain
from mushroom import MushroomBody
import calibration as C

GRAPH = Path(os.environ.get("IDENT_GRAPH", r"C:/Users/lilli/Fly-Lab/versions/fly-v2/graph_v2.npz"))
A_TYPE, B_TYPE = os.environ.get("IDENT_A", "ORN_DL3"), os.environ.get("IDENT_B", "ORN_VL2a")
ODOUR_HZ = float(os.environ.get("IDENT_HZ", 80.0)); PULSE_MS, PRE_MS, TAIL_MS = 300.0, 20.0, 40.0
TEACH_EPOCHS = int(os.environ.get("IDENT_EPOCHS", 12)); REPS = int(os.environ.get("IDENT_REPS", 6))
SEEDS = tuple(int(s) for s in os.environ.get("IDENT_SEEDS", "73,7337,101,202,303").split(","))

fb = FlyBrain(GRAPH); p = fb.p
A = fb.where(type_re=rf"^{A_TYPE}$"); B = fb.where(type_re=rf"^{B_TYPE}$")
gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
STIM = {"A": A, "B": B, "AB": np.concatenate([A, B])}
print(f"graph {GRAPH.name}  A={A_TYPE} {len(A)} cells  B={B_TYPE} {len(B)} cells", flush=True)


def seed_for(base, phase, stim, rep):
    return int.from_bytes(hashlib.sha256(f"{base}:{phase}:{stim}:{rep}".encode()).digest()[:4], "little")


def trial(mb, stim, seed):
    cells = STIM[stim]
    steps = int(round((PRE_MS + PULSE_MS + TAIL_MS) / p.dt)); on0 = int(round(PRE_MS / p.dt)); on1 = on0 + int(round(PULSE_MS / p.dt))
    rng = np.random.default_rng(seed); prob = min(1.0, ODOUR_HZ * p.dt / 1000.0)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    kc_mask = np.zeros(fb.n, bool); kc_mask[mb.kc] = True
    app = np.zeros(fb.n, bool); app[mb.punish_side] = True; avd = np.zeros(fb.n, bool); avd[mb.reward_side] = True
    kc_seen = np.zeros(fb.n, bool); n_app = n_avd = total = 0
    for step in range(steps):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if on0 <= step < on1:
            hit = cells[rng.random(len(cells)) < prob]
            if len(hit): v[hit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            total += len(fired); refr[fired] = fb.refr_steps; v[fired] = p.v_reset
            kc_seen[fired[kc_mask[fired]]] = True; n_app += int(app[fired].sum()); n_avd += int(avd[fired].sum())
            starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt); g = off + np.arange(tot)
                v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
        refr -= 1
    secs = steps * p.dt / 1000.0
    return {"valence": n_app / max(1, app.sum()) / secs - n_avd / max(1, avd.sum()) / secs, "kc": np.flatnonzero(kc_seen),
            "approach_hz": n_app / max(1, app.sum()) / secs, "avoid_hz": n_avd / max(1, avd.sum()) / secs,
            "kc_n": int(kc_seen.sum()), "hz": total / secs / fb.n}


def cold(mb, seed, phase):
    out = {}
    for s in ("A", "B", "AB"):
        rs = [trial(mb, s, seed_for(seed, phase, s, r)) for r in range(REPS)]
        out[s] = {"mean": float(np.mean([r["valence"] for r in rs])), "sd": float(np.std([r["valence"] for r in rs])),
                  "values": [round(r["valence"], 3) for r in rs], "kc_n": int(np.mean([r["kc_n"] for r in rs])), "hz": round(float(np.mean([r["hz"] for r in rs])), 2),
                  "approach_hz": round(float(np.mean([r["approach_hz"] for r in rs])), 2), "avoid_hz": round(float(np.mean([r["avoid_hz"] for r in rs])), 2)}
        out[s]["_kc"] = [set(r["kc"].tolist()) for r in rs]
    return out


def jac(a, b): return len(a & b) / max(1, len(a | b))


t0 = time.perf_counter(); seeds_out = []
for seed in SEEDS:
    store = LAB / f"identity_seed_{seed}.npz"; store.unlink(missing_ok=True)
    mb = MushroomBody(fb, calibration=C.CHOSEN, sides=RUNTIME / "build" / "mb_sides.json", store=store, clock=lambda: 0.0)
    pre = cold(mb, seed, "pre")
    ident = {"jaccard_AA": round(float(np.mean([jac(pre["A"]["_kc"][0], pre["A"]["_kc"][r]) for r in range(1, REPS)])), 3),
             "jaccard_BB": round(float(np.mean([jac(pre["B"]["_kc"][0], pre["B"]["_kc"][r]) for r in range(1, REPS)])), 3),
             "jaccard_AB": round(float(np.mean([jac(a, b) for a in pre["A"]["_kc"] for b in pre["B"]["_kc"]])), 3),
             "kc_active_A": pre["A"]["kc_n"], "kc_active_B": pre["B"]["kc_n"], "kc_total": int(len(mb.kc))}
    teach = []
    for epoch in range(TEACH_EPOCHS):
        for s, val in (("A", +1), ("B", -1)):
            r = trial(mb, s, seed_for(seed, "teach", s, epoch))
            mb.forget_trace(); mb.observe(r["kc"]); hit = int(mb.dopamine(val, 1.0)); mb.apply()
            teach.append({"epoch": epoch, "stim": s, "dopamine": val, "synapses_hit": hit, "valence_during": round(r["valence"], 3)})
    post = cold(mb, seed, "post")
    for d in (pre, post):
        for s in d: d[s].pop("_kc", None)
    sep_pre, sep_post = pre["A"]["mean"] - pre["B"]["mean"], post["A"]["mean"] - post["B"]["mean"]
    learned = bool(sep_post > sep_pre and sep_post > 0)
    seeds_out.append({"seed": seed, "identity": ident, "pre": pre, "post": post, "teach": teach, "mb": mb.stats(),
                      "separation_pre": round(sep_pre, 3), "separation_post": round(sep_post, 3), "learned": learned,
                      "mixture_post": round(post["AB"]["mean"], 3)})
    print(f"seed {seed}: KC A/B active {ident['kc_active_A']}/{ident['kc_active_B']} of {ident['kc_total']}  J(AA) {ident['jaccard_AA']} J(BB) {ident['jaccard_BB']} J(AB) {ident['jaccard_AB']} | "
          f"valence A/B pre {pre['A']['mean']:+.2f}/{pre['B']['mean']:+.2f} -> post {post['A']['mean']:+.2f}/{post['B']['mean']:+.2f}  sep {sep_pre:+.2f} -> {sep_post:+.2f}  "
          f"AB post {post['AB']['mean']:+.2f}  {'LEARNED' if learned else 'no'}  gain {mb.stats()['mean_gain']}  {time.perf_counter()-t0:.0f}s", flush=True)
    store.unlink(missing_ok=True)

n = sum(s["learned"] for s in seeds_out)
out = {"experiment": "identity lesson: odour A rewarded, odour B punished; cold valence; mixture generalization",
       "chosen": {"graph": str(GRAPH), "A": A_TYPE, "B": B_TYPE, "odour_hz": ODOUR_HZ, "pulse_ms": PULSE_MS, "teach_epochs": TEACH_EPOCHS, "reps": REPS, "seeds": SEEDS,
                  "pass_line": "sep_post > sep_pre and sep_post > 0 in a majority of seeds; J(AB) well below J(AA)"},
       "measured": {"seeds_learned": n, "of": len(SEEDS), "pass": bool(n > len(SEEDS) / 2),
                    "mean_sep_pre": round(float(np.mean([s["separation_pre"] for s in seeds_out])), 3),
                    "mean_sep_post": round(float(np.mean([s["separation_post"] for s in seeds_out])), 3),
                    "mean_jaccard_AB": round(float(np.mean([s["identity"]["jaccard_AB"] for s in seeds_out])), 3),
                    "mean_jaccard_AA": round(float(np.mean([s["identity"]["jaccard_AA"] for s in seeds_out])), 3),
                    "elapsed_s": round(time.perf_counter() - t0, 1)},
       "seeds": seeds_out}
(LAB / "results" / (os.environ.get("IDENT_OUT") or "identity_lesson.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["measured"], indent=1))
