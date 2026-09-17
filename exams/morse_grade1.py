from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

LAB = Path(r"C:\Users\lilli\Fly-Lab-2")
RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
RESULTS = LAB / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain
if os.environ.get("MORSE_BRAIN") == "axon":   # fly-v1-axon fork (2026-09-17); default path untouched
    sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v1-axon")
    from flysim_axon import FlyBrainAxon as FlyBrain   # noqa: F811
    _AXON_GRAPH = Path(r"C:/Users/lilli/Fly-Lab/versions/fly-v1-axon/graph_axon.npz")
else:
    _AXON_GRAPH = None
from mushroom import MushroomBody
import calibration as C

# CHOSEN constants for School Grade 1 attempt 1.
DOT_MS = float(os.environ.get("MORSE_DOT_MS", 20.0))     # chosen; env override for the duration-swap test 2026-09-16
DASH_MS = float(os.environ.get("MORSE_DASH_MS", 60.0))
GAP_MS = float(os.environ.get("MORSE_GAP_MS", 40.0))
PRE_MS = 20.0
TAIL_MS = 40.0
JO_HZ = 120.0
BIN_MS = 10.0
DN_SPIKES_PER_BIN = 1
DOT_DASH_SPLIT_MS = 40.0
TRAIN_EPOCHS = 24
TEST_REPS = 6
BASE_SEEDS = (73, 7337)
TRAIN = (".", "-")
HELDOUT = (".-", "-.")
def ms_steps(ms, dt):
    return max(1, int(round(ms / dt)))


def make_timeline(pattern, dt):
    pre = ms_steps(PRE_MS, dt)
    tail = ms_steps(TAIL_MS, dt)
    gap = ms_steps(GAP_MS, dt)
    parts = []
    for i, ch in enumerate(pattern):
        dur = ms_steps(DOT_MS if ch == "." else DASH_MS, dt)
        parts.append((dur, True))
        if i + 1 < len(pattern):
            parts.append((gap, False))
    total = pre + sum(n for n, _ in parts) + tail
    sound = np.zeros(total, dtype=bool)
    p = pre
    for n, on in parts:
        if on:
            sound[p:p+n] = True
        p += n
    return sound, pre


def seed_for(base, phase, pattern, rep):
    b = hashlib.sha256(f"{base}:{phase}:{pattern}:{rep}".encode()).digest()[:4]
    return int.from_bytes(b, "little")


def decode(counts, pre_bins):
    counts = np.asarray(counts, dtype=int)
    active = counts[pre_bins:] >= DN_SPIKES_PER_BIN
    runs, n = [], len(active)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j]:
            j += 1
        runs.append((i, j))
        i = j
    split_bins = max(1, int(round(DOT_DASH_SPLIT_MS / BIN_MS)))
    pattern = "".join("." if (b-a) < split_bins else "-" for a, b in runs)
    return pattern, active.astype(int).tolist(), [[int(a), int(b)] for a, b in runs]


def run_trial(fb, mb, pattern, seed):
    p = fb.p
    sound, pre_steps = make_timeline(pattern, p.dt)
    jo = fb.where(type_re=r"^JO-A")
    dn = fb.where(type_re=r"^DNa01$")
    if not len(jo) or not len(dn):
        raise RuntimeError(f"missing JO-A or DNa01: jo={len(jo)} dn={len(dn)}")
    gains = C.gains_for(fb, C.CHOSEN)
    gain_per_neuron = (np.ones(fb.n, dtype=np.float32) if gains is None
                       else gains[fb.type_code].astype(np.float32))
    rng = np.random.default_rng(seed)
    v = np.full(fb.n, p.v_rest, dtype=np.float32)
    refr = np.zeros(fb.n, dtype=np.int32)
    bin_steps = ms_steps(BIN_MS, p.dt)
    counts = np.zeros(int(np.ceil(len(sound) / bin_steps)), dtype=np.int32)
    kc_mask = np.zeros(fb.n, dtype=bool); kc_mask[mb.kc] = True
    dn_mask = np.zeros(fb.n, dtype=bool); dn_mask[dn] = True
    kc_seen = np.zeros(fb.n, dtype=bool)
    prob = min(1.0, JO_HZ * p.dt / 1000.0)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    # fly-v1-axon (2026-09-17): axonal edges gate the emitter's output instead of adding to the target's voltage.
    # Only active when the brain object carries an `axonal` flag; otherwise identical to v1.
    axonal = getattr(fb, "axonal", None)
    use_gate = axonal is not None and bool(np.any(axonal))
    gate_in = np.zeros(fb.n, dtype=np.float32)
    gate_decay = float(getattr(fb, "gate_decay", 1.0))
    # tonic leg-sensory drive ("standing"): MORSE_LEG_HZ Hz on every leg sensory neuron, CHOSEN; 0 = off (v1 protocol)
    leg_hz = float(os.environ.get("MORSE_LEG_HZ", 0))
    leg = np.load(r"C:/Users/lilli/Fly-Lab/versions/fly-v1-axon/leg_sensory_idx.npy") if leg_hz > 0 else np.array([], dtype=np.int64)
    leg_prob = min(1.0, leg_hz * p.dt / 1000.0)
    for step, sound_on in enumerate(sound):
        v = p.v_rest + (v - p.v_rest) * fb.decay
        if use_gate:
            gate_in *= gate_decay
        if sound_on:
            hit = jo[rng.random(len(jo)) < prob]
            if len(hit):
                v[hit] = p.v_thresh + 1.0
        if len(leg):
            lhit = leg[rng.random(len(leg)) < leg_prob]
            if len(lhit):
                v[lhit] = p.v_thresh + 1.0
        v[refr > 0] = p.v_reset
        fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
        if len(fired):
            refr[fired] = fb.refr_steps
            v[fired] = p.v_reset
            fk = fired[kc_mask[fired]]
            if len(fk):
                kc_seen[fk] = True
            counts[min(step // bin_steps, len(counts)-1)] += int(dn_mask[fired].sum())
            starts = indptr[fired]
            cnt = indptr[fired + 1] - starts
            tot = int(cnt.sum())
            if tot:
                off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                g = off + np.arange(tot)
                tgt = indices[g]
                if use_gate:
                    gate = np.clip(1.0 + 0.05 * gate_in[fired], 0.0, 3.0).astype(np.float32)   # CHOSEN, same as flysim_axon
                    val = wdata[g] * np.repeat(gain_per_neuron[fired] * gate, cnt)
                    ax = axonal[g]
                    if ax.any():
                        gate_in += np.bincount(tgt[ax], weights=val[ax], minlength=fb.n).astype(np.float32)
                        val = val[~ax]; tgt = tgt[~ax]
                else:
                    val = wdata[g] * np.repeat(gain_per_neuron[fired], cnt)
                v += np.bincount(tgt, weights=val, minlength=fb.n).astype(np.float32)
        refr -= 1
    pre_bins = int(round(pre_steps / bin_steps))
    decoded, active, runs = decode(counts, pre_bins)
    return {
        "target": pattern, "decoded": decoded,
        "correct": decoded == pattern,
        "dn_counts": counts.tolist(), "dn_active": active, "dn_runs": runs,
        "kc": np.flatnonzero(kc_seen).astype(np.int64),
        "mean_mv_final": float(v.mean()),
    }
def public_trial(r):
    return {k: v for k, v in r.items() if k != "kc"}


def cold_test(fb, mb, base_seed):
    rows = []
    for pattern in HELDOUT:
        for rep in range(TEST_REPS):
            r = run_trial(fb, mb, pattern, seed_for(base_seed, "cold", pattern, rep))
            row = public_trial(r)
            row["rep"] = rep
            rows.append(row)
    return rows


def run_seed(base_seed):
    fb = FlyBrain(_AXON_GRAPH or RUNTIME / "build" / "graph.npz")
    store = LAB / f"morse_grade1_seed_{base_seed}.npz"
    if store.exists():
        store.unlink()
    mb = MushroomBody(fb, calibration=C.CHOSEN,
                      sides=RUNTIME / "build" / "mb_sides.json",
                      store=store, clock=lambda: 0.0)
    pre = cold_test(fb, mb, base_seed)
    train_rows = []
    for epoch in range(TRAIN_EPOCHS):
        for symbol in TRAIN:
            r = run_trial(fb, mb, symbol, seed_for(base_seed, "train", symbol, epoch))
            mb.forget_trace(); mb.observe(r["kc"])
            valence = +1 if r["correct"] else -1
            hit = mb.dopamine(valence, 1.0)
            mb.apply()
            train_rows.append({"epoch": epoch, "symbol": symbol,
                               "decoded": r["decoded"], "correct": bool(r["correct"]),
                               "dopamine": valence, "synapses_hit": int(hit),
                               "dn_counts": r["dn_counts"]})
    post = cold_test(fb, mb, base_seed)
    by_symbol = {}
    for pattern in HELDOUT:
        a = [r for r in pre if r["target"] == pattern]
        b = [r for r in post if r["target"] == pattern]
        by_symbol[pattern] = {
            "pre_correct": int(sum(r["correct"] for r in a)),
            "post_correct": int(sum(r["correct"] for r in b)),
            "trials": len(b),
            "pre_decoded": [r["decoded"] for r in a],
            "post_decoded": [r["decoded"] for r in b],
        }
    novel_exact = [p for p, s in by_symbol.items() if s["post_correct"] > 0]
    return {
        "seed": base_seed,
        "pre_test": pre,
        "training": train_rows,
        "post_test": post,
        "heldout_summary": by_symbol,
        "grade1_pass_literal": bool(novel_exact),
        "novel_exact_symbols": novel_exact,
        "mb": mb.stats(),
    }


def chosen_block():
    return {
        "dot_ms": DOT_MS, "dash_ms": DASH_MS, "gap_ms": GAP_MS,
        "pre_ms": PRE_MS, "tail_ms": TAIL_MS,
        "input_population": "all type names matching ^JO-A",
        "carrier_hz": JO_HZ, "output_population": "DNa01 (2 cells)",
        "output_bin_ms": BIN_MS, "readout_threshold_spikes_per_bin": DN_SPIKES_PER_BIN,
        "dot_dash_split_ms": DOT_DASH_SPLIT_MS,
        "leg_tonic_hz": float(os.environ.get("MORSE_LEG_HZ", 0)), "brain": os.environ.get("MORSE_BRAIN", "v1"),
        "reward_mapping": "exact primitive decode => +1; anything else => -1",
        "dopamine_amount": 1.0, "training_symbols": list(TRAIN),
        "heldout_symbols": list(HELDOUT), "training_epochs": TRAIN_EPOCHS,
        "cold_test_reps_per_heldout": TEST_REPS, "base_seeds": [int(s) for s in os.environ["MORSE_SEEDS"].split(",")] if os.environ.get("MORSE_SEEDS") else list(BASE_SEEDS),
        "calibration": C.CHOSEN, "trial_membrane_state": "reset each trial; MB weights persist",
        "runtime_modified": False,
    }


def main(smoke=False):
    t0 = time.perf_counter()
    if smoke:
        fb = FlyBrain(_AXON_GRAPH or RUNTIME / "build" / "graph.npz")
        store = LAB / "morse_grade1_smoke.npz"
        if store.exists():
            store.unlink()
        mb = MushroomBody(fb, calibration=C.CHOSEN,
                          sides=RUNTIME / "build" / "mb_sides.json",
                          store=store, clock=lambda: 0.0)
        rows = []
        for symbol in TRAIN:
            r = run_trial(fb, mb, symbol, seed_for(73, "smoke", symbol, 0))
            rows.append(public_trial(r))
        out = {
            "mode": "STARTUP ONLY — not a learning verdict",
            "chosen": chosen_block(),
            "measured": {"brain_n": fb.n, "jo_a_cells": len(fb.where(type_re=r"^JO-A")),
                         "dna01_cells": len(fb.where(type_re=r"^DNa01$")), "trials": rows},
        }
        path = RESULTS / "morse_grade1_smoke.json"
    else:
        seeds = []
        bases = tuple(int(s) for s in os.environ["MORSE_SEEDS"].split(",")) if os.environ.get("MORSE_SEEDS") else BASE_SEEDS
        for base in bases:
            print(f"seed {base} start", flush=True)
            seeds.append(run_seed(base))
            print(f"seed {base} done: {seeds[-1]['heldout_summary']}", flush=True)
        literal = any(s["grade1_pass_literal"] for s in seeds)
        out = {
            "experiment": "School Grade 1 attempt 1 — timing primitives to novel paired rhythm",
            "chosen": chosen_block(),
            "measured": {
                "brain_n": 165122,
                "jo_a_cells": 50,
                "dna01_cells": 2,
                "elapsed_wall_s": time.perf_counter() - t0,
                "grade1_pass_literal": literal,
                "pass_definition": "at least one exact DNa01-decoded held-out symbol after training",
            },
            "seeds": seeds,
        }
        path = RESULTS / (os.environ.get("MORSE_OUT") or "morse_grade1.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(path), "elapsed_s": round(time.perf_counter()-t0, 1),
                      "pass": out.get("measured", {}).get("grade1_pass_literal")}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(args.smoke)