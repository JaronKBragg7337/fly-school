"""The frozen fly-v10 stack, loaded the same way for the sender and the receiver. (Claude Code, 2026-09-17)
graph_v9 + fly-v10 calib.json (per-type gains + KC threshold shift) on gpu/flysim_gpu.GPUBrain. Device from GATE_DEVICE (cuda|cpu).
Records the SHA-256 of the graph and calib so every output can be checked against versions/comm-loop-1/MANIFEST.md.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
LAB = Path(r"C:\Users\lilli\Fly-Lab-2"); RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, str(LAB / "gpu"))
import calibration as C
from flysim_gpu import GPUBrain, rates   # noqa: F401

GRAPH = Path(os.environ.get("GATE_GRAPH", r"C:/Users/lilli/Fly-Lab/versions/fly-v9/graph_v9.npz"))
CALIB = Path(os.environ.get("GATE_CALIB", r"C:/Users/lilli/Fly-Lab/versions/fly-v10/calib.json"))
DEVICE = os.environ.get("GATE_DEVICE", "cuda")
PRE_MS, PULSE_MS, TAIL_MS = 50.0, 300.0, 50.0


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load():
    gb = GPUBrain(GRAPH, device=DEVICE); types = gb.types
    # per-type CHOSEN gains (calibration.py) then the fly-v10 calib on top - identical to comm_loop.py / grade0b_organs.py
    class _FB:  # calibration.gains_for wants .type_names / .types / .superclass
        pass
    fb = _FB(); fb.type_names = gb.type_names; fb.types = types; fb.superclass = gb.superclass; fb.n = gb.n; fb.type_code = gb.type_code; fb.n_types = len(gb.type_names)
    gains = C.gains_for(fb, C.CHOSEN); gpn = gains[gb.type_code].astype(np.float32)
    thresh = np.full(gb.n, -45.0, dtype=np.float32)
    c = json.load(open(CALIB))
    gpn[np.array([("PN" in x and not x.startswith("MBON")) for x in types])] *= np.float32(c["pn"]); gpn[types == "APL"] *= np.float32(c["apl"])
    gpn[np.array([x.startswith("KC") for x in types])] *= np.float32(c["kc"]); gpn[np.array([x.startswith("MBON") for x in types])] *= np.float32(c["mbon"])
    thresh[np.array([x.startswith("KC") for x in types])] += np.float32(c["kc_thresh_shift_mv"])
    gb.set_gain(gpn); gb.set_thresh(thresh)
    return gb


def provenance():
    return {"graph": str(GRAPH), "graph_sha256_16": sha(GRAPH), "calib": str(CALIB), "calib_sha256_16": sha(CALIB), "device": DEVICE,
            "trial_ms": [PRE_MS, PULSE_MS, TAIL_MS]}
