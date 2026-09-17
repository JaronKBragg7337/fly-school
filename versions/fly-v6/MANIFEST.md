# fly-v6 — fly-v5 with the APL gain moved to the APL->KC edges only (2026-09-17 09:20 EDT, Claude Code)
Why: in a trained fly-v5, the rewarded odour's KC->PAM-MBON synapses were depressed to gain 0.26, yet KCs supplied only 9% of the
reward-side MBONs' excitation during the odour and APL/DPM inhibition onto those MBONs summed to -1.0e6 mV: v5's per-neuron APL x6
crushed the MBONs. Real APL is GABAergic mainly onto KCs. Here APL/DPM->KC edges are x6.016 in the graph; all other APL/DPM
outputs x1; PN x1.265, KC x1.790, MBON x0.856, KC threshold +1.80 mV unchanged (calib.json).
graph_v6.npz sha256(16): c8504de5219ef8d2   base: fly-v3 graph_v3_s0.35.npz
