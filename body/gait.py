"""Stepping pattern generator - PORTED from Mineplix/fly-brain (lulzx, MIT), src/sim/motor.js + public/body/gait.json.
Two-harmonic joint curves fitted to 100 FlySuite real-fly walking trajectories (tripod, 10 Hz, duty 0.68), refined for
stability on flybody. The brain's descending neurons command it (forward drive, steering); the generator moves the legs.
This is BORROWED walking, declared as such - no connectome-only model walks (their docs, ours). Claude Code, 2026-09-17.
DN roles and readout constants are theirs (Cande 2018, Bidaye 2014/2020, Rayshubskiy 2020): see DN_ROLES / READOUT.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DN_ROLES = {
    "forward": {"DNg100": 1, "DNg97": 1, "DNp09": 1, "DNa05": .2, "DNa07": .2, "DNp26": .2, "DNg25": .2, "DNa01": .1, "DNa02": .1},
    "backward": {"MDN": 1},
    "turn": {"DNa02": 1.0, "DNa01": 0.6, "DNp09": 0.5},
}
READOUT = {"fwdThreshold": 4.0, "fwdScale": 12.0, "turnScale": 25.0, "turnTau": 60.0, "turnAdaptTau": 3000.0, "backMax": 0.33}
LEGS, SIDES = ("T1", "T2", "T3"), ("left", "right")
PHASE = {"T1_left": 0, "T2_right": 0, "T3_left": 0, "T1_right": math.pi, "T2_left": math.pi, "T3_right": math.pi}


class Gait:
    def __init__(self, fb, side=None, path=HERE / "gait.mineplix.json"):
        """side: array of 'L'/'R'/'' per neuron (from body-annotations somaSide), for the steering populations"""
        self.g = json.load(open(path, encoding="utf-8"))
        types = fb.types.astype(str)
        side = np.asarray(side, dtype=object) if side is not None else None
        def pop(roles, s=None):
            out = []
            for t, w in roles.items():
                idx = np.flatnonzero(types == t)
                if s is not None and side is not None:
                    idx = idx[side[idx] == s]
                out += [(int(i), float(w)) for i in idx]
            return out
        self.fwd, self.back = pop(DN_ROLES["forward"]), pop(DN_ROLES["backward"])
        self.turnL, self.turnR = pop(DN_ROLES["turn"], "L"), pop(DN_ROLES["turn"], "R")
        self.rate = np.zeros(fb.n, dtype=np.float32); self.phase = 0.0; self.turnF = 0.0; self.turnBase = 0.0
        self.cmd = {"v": 0.0, "turn": 0.0, "drive": 0.0}

    def read_brain(self, spike_counts_window, dt_ms, tau=40.0):
        """low-pass filtered firing rate (Hz) from this window's spike counts"""
        k = dt_ms / tau
        self.rate += k * (spike_counts_window * (1000.0 / dt_ms) - self.rate)

    def _wmean(self, pairs):
        if not pairs: return 0.0
        w = sum(x for _, x in pairs); return float(sum(self.rate[i] * x for i, x in pairs) / w)

    def controls(self, dt_ms, act_index, ctrl):
        R, g = READOUT, self.g
        fwd, back = self._wmean(self.fwd), self._wmean(self.back)
        turn = self._wmean(self.turnL) - self._wmean(self.turnR)
        net = fwd - 2 * back
        sat = lambda x: 1 - math.exp(-x / R["fwdScale"])
        v = sat(net - R["fwdThreshold"]) if net > R["fwdThreshold"] else (-R["backMax"] * sat(back - R["fwdThreshold"]) if back > R["fwdThreshold"] else 0.0)
        self.turnF += dt_ms / R["turnTau"] * (turn - self.turnF)
        self.turnBase += dt_ms / R["turnAdaptTau"] * (self.turnF - self.turnBase)
        t = max(-0.6, min(0.6, (self.turnF - self.turnBase) / R["turnScale"]))
        self.cmd = {"v": v, "turn": t, "drive": fwd, "back": back}
        amp = min(1.0, abs(v) * 1.5); freq = g["freq"] * (0.5 + 0.5 * min(1.0, abs(v)))
        if amp > 0.05:
            self.phase += math.copysign(1, v) * 2 * math.pi * freq * dt_ms / 1000.0
        for leg in LEGS:
            for sd in SIDES:
                key = f"{leg}_{sd}"; phi = self.phase + PHASE[key]
                steer = 1 + t * (-1 if sd == "left" else 1)
                for j in g["joints"]:
                    off, a1, p1, a2, p2 = g["params"][leg][j]
                    q = a1 * math.cos(phi + p1) + a2 * math.cos(2 * phi + p2)
                    if j == "coxa": q *= steer
                    name = f"{j}_{key}"
                    if name in act_index: ctrl[act_index[name]] = amp * (off + q)
                stance = ((phi + g["adhPhase"]) % (2 * math.pi)) < 2 * math.pi * g["duty"]
                name = f"adhere_claw_{key}"
                if name in act_index: ctrl[act_index[name]] = (1.0 if stance else 0.0) if amp > 0.05 else 0.8
        return self.cmd
