"""fly-v1-axon: FlyBrain with two synapse classes. FORK of runtime/flysim.py (2026-09-17 00:50, Claude Code).

Somatic edges: unchanged - a presynaptic spike adds its weight to the target's voltage.
Axonal edges (per-edge `axonal` flag from graph_axon.npz; today only inputs onto descending neurons): the spike never
touches the target's voltage. It sets a per-target OUTPUT GATE instead: the target's emitted weights are scaled by
gate[i] for GATE_TAU_MS afterwards. Excitatory axonal input raises the gate, inhibitory lowers it (down to 0 = veto).
Biology: Ceballos et al., iScience 2026 - axo-axonic synapses on DNs "veto, amplify, or synchronize" spikes.

CHOSEN (v1, stated so they can be swapped): GATE_TAU_MS = 5.0; gate = 1 + A * (summed axonal input in mV) with
A = 0.05 per mV, clipped to [0, 3]; so +20 mV of excitatory axonal input doubles emitted weight, -20 mV silences it.
MEASURED: everything the run records. With no axonal edges (axonal all False) this class is behaviourally identical
to FlyBrain: the gate stays 1.0 and the code path reduces to the original.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain  # noqa: E402

GATE_TAU_MS = 5.0
GATE_A = 0.05
GATE_MAX = 3.0


class FlyBrainAxon(FlyBrain):
    def __init__(self, graph_path):
        super().__init__(graph_path)
        z = np.load(graph_path, allow_pickle=True)
        if "axonal" not in z.files:
            raise ValueError("graph has no per-edge 'axonal' flag - use graph_axon.npz")
        # the base class stores W as CSC (columns = presynaptic); rebuild the same CSC from the CSR the graph ships,
        # carrying the axonal flag through the identical permutation so it lines up with self.wdata.
        import scipy.sparse as sp
        n = int(z["shape"][0])
        csr = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
        flag = sp.csr_matrix((z["axonal"].astype(np.float32) + 1.0, z["indices"], z["indptr"]), shape=(n, n))
        csc_flag = flag.tocsc()
        # after tocsc the ordering is deterministic and shared with self.W.tocsc(); values 1.0 => somatic, 2.0 => axonal
        self.axonal = (csc_flag.data >= 1.5)
        assert len(self.axonal) == len(self.wdata), "axonal flag did not align with weights"
        self.gate_decay = float(np.exp(-self.p.dt / GATE_TAU_MS))
        self.n_axonal_edges = int(self.axonal.sum())

    def run(self, drive, steps, gains=None, record=None, seed=0, spike_log=False):
        p = self.p
        rng = np.random.default_rng(seed)
        n = self.n
        v = np.full(n, p.v_rest, dtype=np.float32)
        refr = np.zeros(n, dtype=np.int32)
        gain_per_neuron = (gains[self.type_code] if gains is not None else np.ones(n, dtype=np.float32)).astype(np.float32)
        # external drive - same shape as the base class
        idx_parts, p_parts = [], []
        for ii, rr in (drive or {}).items():
            ii = np.asarray(ii, dtype=np.int64)
            rr = np.broadcast_to(np.asarray(rr, dtype=np.float32), ii.shape) if np.ndim(rr) == 0 else np.asarray(rr, dtype=np.float32)
            idx_parts.append(ii); p_parts.append(np.clip(rr * p.dt / 1000.0, 0.0, 1.0))
        ext_idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=np.int64)
        ext_p = np.concatenate(p_parts).astype(np.float32) if p_parts else np.array([], dtype=np.float32)

        record = record or {}
        counts = {k: np.zeros(len(v_), dtype=np.int64) for k, v_ in record.items()}
        total_spikes = 0
        ever = np.zeros(n, dtype=bool)
        log = [] if spike_log else None
        indptr, indices, wdata, axonal = self.indptr, self.indices, self.wdata, self.axonal
        thresh, rest, reset, decay = p.v_thresh, p.v_rest, p.v_reset, self.decay
        gate_in = np.zeros(n, dtype=np.float32)      # summed axonal input (mV), decays with GATE_TAU_MS
        gd = self.gate_decay

        for _ in range(steps):
            v = rest + (v - rest) * decay
            gate_in *= gd
            if len(ext_idx):
                hit = ext_idx[rng.random(len(ext_idx)) < ext_p]
                if len(hit):
                    v[hit] = thresh + 1.0
            v[refr > 0] = reset
            fired = np.flatnonzero((v >= thresh) & (refr <= 0))
            if spike_log:
                log.append(fired.astype(np.int32))
            if len(fired):
                total_spikes += len(fired)
                ever[fired] = True
                refr[fired] = self.refr_steps
                v[fired] = reset
                starts = indptr[fired]
                cnt = indptr[fired + 1] - starts
                tot = int(cnt.sum())
                if tot:
                    off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                    g = off + np.arange(tot)
                    tgt = indices[g]
                    # the emitting cell's output is scaled by ITS gate (what its axon has received lately)
                    gate = np.clip(1.0 + GATE_A * gate_in[fired], 0.0, GATE_MAX).astype(np.float32)
                    val = wdata[g] * np.repeat(gain_per_neuron[fired] * gate, cnt)
                    ax = axonal[g]
                    if ax.any():
                        # axonal edges: no voltage change on the target; accumulate into the target's gate input
                        gate_in += np.bincount(tgt[ax], weights=val[ax], minlength=n).astype(np.float32)
                        val = val[~ax]; tgt = tgt[~ax]
                    v += np.bincount(tgt, weights=val, minlength=n).astype(np.float32)
                for name, sel in record.items():
                    counts[name] += np.isin(sel, fired)
            refr -= 1

        secs = steps * p.dt / 1000.0
        out = {k: c / secs for k, c in counts.items()}
        out["_total_hz"] = total_spikes / secs / n
        out["_spikes_per_sec"] = total_spikes / secs
        out["_fired"] = np.flatnonzero(ever)
        out["_mean_mv"] = float(v.mean())
        if spike_log:
            out["_spikes"] = log
        return out
