"""GPU kernel for the MaleCNS LIF fly - same maths as flysim.py / comm_loop.py trial(), batched over flies. (Claude Code, 2026-09-17 12:20 EDT)

WHY: the MSI's RTX 4060 sat at 0% while 16 CPU processes shared one fly each. One SpMM per step drives B flies at once.
WHAT IS IDENTICAL to the CPU kernel (comm_loop.py trial): dt 0.2 ms; v_rest -52, v_reset -52, threshold -45 (per-neuron vector);
tau_m 20 ms (decay per step); refractory 3.8 ms (fly-v3) = ceil(3.8/0.2) steps; Poisson injection sets v = thresh + 1 mV on the
hit cells during the pulse window, drawn with numpy default_rng(seed) in the SAME order as the CPU kernel; a spike adds
sign * weight * gain[pre] to every post cell in the same step; refractory cells are clamped to reset before the threshold test.
WHAT DIFFERS: float32 summation order inside cuSPARSE (a target receiving many spikes in one step may differ in the last bits);
that can flip a borderline threshold crossing, so runs are statistically equivalent, not bit-identical. Measured in verify_gpu.py.
Graph: the stored graph.npz is CSR with rows = POST, cols = PRE (flysim converts to CSC per PRE); here it is used as stored:
input = W @ S where S[pre, fly] = gain[pre] if pre fired. Learned weights (mushroom.py writes into fb.wdata, CSC order) are
synced with sync_weights(fb.wdata) through the CSC->CSR permutation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")


class GPUBrain:
    def __init__(self, graph_path, refractory_ms=3.8, dt=0.2, tau_m=20.0, v_rest=-52.0, v_reset=-52.0, v_thresh=-45.0, device="cuda"):
        z = np.load(graph_path, allow_pickle=True)
        self.n = int(z["shape"][0]); self.dev = torch.device(device)
        self.types = z["types"].astype(str); self.superclass = z["superclass"].astype(str) if "superclass" in z else None
        self.type_names, self.type_code = np.unique(self.types, return_inverse=True)
        csr = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))   # rows POST, cols PRE
        csr.sort_indices()
        perm_csr = sp.csr_matrix((np.arange(csr.nnz, dtype=np.int64), csr.indices, csr.indptr), shape=csr.shape)
        self.csc_perm = perm_csr.tocsc().data          # csc.data[i] == csr.data[csc_perm[i]]
        self.csr_shape = csr.shape; self.nnz = csr.nnz
        self.crow = torch.as_tensor(csr.indptr.astype(np.int64), device=self.dev)
        self.col = torch.as_tensor(csr.indices.astype(np.int64), device=self.dev)
        self.wdata_csr = csr.data.copy()
        self.W = torch.sparse_csr_tensor(self.crow, self.col, torch.as_tensor(csr.data, device=self.dev), size=csr.shape)
        self.dt, self.tau_m = dt, tau_m
        self.v_rest, self.v_reset = float(v_rest), float(v_reset)
        self.thresh = torch.full((self.n,), float(v_thresh), dtype=torch.float32, device=self.dev)
        self.decay = float(np.exp(-dt / tau_m)); self.refr_steps = int(np.ceil(refractory_ms / dt))
        self.gain = torch.ones(self.n, dtype=torch.float32, device=self.dev)

    # ---- configuration -------------------------------------------------------------------------------------------------
    def set_gain(self, gpn):
        self.gain = torch.as_tensor(np.asarray(gpn, dtype=np.float32), device=self.dev)

    def set_thresh(self, thresh):
        self.thresh = torch.as_tensor(np.asarray(thresh, dtype=np.float32), device=self.dev)

    def sync_weights(self, wdata_csc):
        """Push weights held in flysim's CSC order (fb.wdata, which mushroom.py edits) onto the GPU."""
        w = np.empty(self.nnz, dtype=np.float32); w[self.csc_perm] = np.asarray(wdata_csc, dtype=np.float32)
        self.wdata_csr = w
        self.W = torch.sparse_csr_tensor(self.crow, self.col, torch.as_tensor(w, device=self.dev), size=self.csr_shape)

    def where(self, *names):
        return np.flatnonzero(np.isin(self.types, names))

    # ---- the trial ------------------------------------------------------------------------------------------------------
    def trial(self, stims, seeds, pre_ms=20.0, pulse_ms=300.0, tail_ms=40.0, record=None, external=None):
        """Run B = len(seeds) independent flies in one pass.
        stims: list of (cells, hz) - the same Poisson drive for every fly (each fly draws its own hits from its own seed).
        external: optional callable(step) -> (cells, mask[len(cells), B] bool) or None, for world-driven input (Gate 1's ear);
                  applied like the Poisson hits (v = thresh + 1) so the world and the odour use one mechanism.
        record: optional index array; returns spikes[step, cell, fly] bool for those cells (the sender's wing motor neurons).
        Returns dict: counts (n, B) int32 spike counts, hz (B,) whole-brain rate, rec (steps, len(record), B) or None.
        """
        B = len(seeds); dev = self.dev
        steps = int(round((pre_ms + pulse_ms + tail_ms) / self.dt)); on0 = int(round(pre_ms / self.dt)); on1 = on0 + int(round(pulse_ms / self.dt))
        # Poisson hits drawn on the CPU in the CPU kernel's order: one rng per fly, one random(len(cells)) per step.
        # one rng per fly; per step the stims draw in order (stim 0, stim 1, ...) exactly as the CPU kernel does
        masks = [np.zeros((on1 - on0, len(cells), B), dtype=bool) for cells, _ in stims]
        probs = [min(1.0, hz * self.dt / 1000.0) for _, hz in stims]
        for b, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            for k in range(on1 - on0):
                for i, (cells, _) in enumerate(stims):
                    masks[i][k, :, b] = rng.random(len(cells)) < probs[i]
        hits = [(torch.as_tensor(np.asarray(cells), dtype=torch.int64, device=dev), torch.as_tensor(m, device=dev)) for (cells, _), m in zip(stims, masks)]
        v = torch.full((self.n, B), self.v_rest, dtype=torch.float32, device=dev)
        refr = torch.zeros((self.n, B), dtype=torch.int32, device=dev)
        counts = torch.zeros((self.n, B), dtype=torch.int32, device=dev)
        th = self.thresh[:, None]; gain = self.gain[:, None]
        reset = torch.full_like(v, self.v_reset); refr_full = torch.full_like(refr, self.refr_steps)
        rec_idx = torch.as_tensor(np.asarray(record), dtype=torch.int64, device=dev) if record is not None else None
        rec = torch.zeros((steps, len(record), B), dtype=torch.bool, device=dev) if record is not None else None
        for step in range(steps):
            v = self.v_rest + (v - self.v_rest) * self.decay
            if on0 <= step < on1:
                for cells, m in hits:
                    mk = m[step - on0]
                    v[cells] = torch.where(mk, th[cells] + 1.0, v[cells])
            if external is not None:
                ext = external(step)
                if ext is not None:
                    cells, mk = ext
                    cells = torch.as_tensor(np.asarray(cells), dtype=torch.int64, device=dev); mk = torch.as_tensor(np.asarray(mk), device=dev)
                    v[cells] = torch.where(mk, th[cells] + 1.0, v[cells])
            v = torch.where(refr > 0, reset, v)
            fired = (v >= th) & (refr <= 0)
            if bool(fired.any()):
                counts += fired
                refr = torch.where(fired, refr_full, refr)
                v = torch.where(fired, reset, v)
                if rec is not None: rec[step] = fired[rec_idx]
                S = fired.to(torch.float32) * gain
                v = v + torch.sparse.mm(self.W, S)
            refr -= 1
        secs = steps * self.dt / 1000.0
        c = counts.cpu().numpy()
        return {"counts": c, "hz": c.sum(0) / secs / self.n, "secs": secs, "steps": steps, "rec": rec.cpu().numpy() if rec is not None else None}


def rates(out, idx):
    """Mean firing rate (Hz) of the cells in idx, per fly."""
    return out["counts"][np.asarray(idx)].sum(0) / max(1, len(idx)) / out["secs"]
