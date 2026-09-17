# fly-v4 — fly-v3 + per-KC homeostatic thresholds (2026-09-17, Claude Code)
Graph: fly-v3 graph_v3_s0.35.npz (unchanged). Cell: refractory 3.8 ms (v3). NEW: kc_thresh.npy — per-neuron spike threshold;
non-KC cells -45 mV; each KC's threshold set by the CHOSEN homeostatic rule below on an 18-odour panel @40.0 Hz.
Rule: iterate 14x; KC firing for > 20% of the panel: threshold +2.0 mV; KC firing for none: -1.5 mV; bounds [-50.0, -25.0] mV.
Why: sparse but non-specific KC code in v1-v3 (pair overlap 0.4-0.7). Biology: KC intrinsic excitability is regulated (sparse,
decorrelated odour coding; Turner 2008, Honegger 2011). Measured trajectory:
{"iter": 0, "kc_active_per_odour_pct": 5.03, "median_pair_jaccard": 0.333, "mean_pair_jaccard": 0.369, "kcs_silent_pct": 85.2, "kcs_generalist_pct": 9.4, "brain_hz": 5.08, "thresh_kc_mean": -45.0, "t": 27}
{"iter": 1, "kc_active_per_odour_pct": 9.34, "median_pair_jaccard": 0.434, "mean_pair_jaccard": 0.416, "kcs_silent_pct": 71.3, "kcs_generalist_pct": 13.1, "brain_hz": 5.22, "thresh_kc_mean": -46.09, "t": 52}
{"iter": 2, "kc_active_per_odour_pct": 16.04, "median_pair_jaccard": 0.357, "mean_pair_jaccard": 0.351, "kcs_silent_pct": 55.6, "kcs_generalist_pct": 26.3, "brain_hz": 5.36, "thresh_kc_mean": -46.9, "t": 78}
{"iter": 3, "kc_active_per_odour_pct": 16.31, "median_pair_jaccard": 0.433, "mean_pair_jaccard": 0.391, "kcs_silent_pct": 54.6, "kcs_generalist_pct": 26.0, "brain_hz": 5.01, "thresh_kc_mean": -47.21, "t": 104}
{"iter": 4, "kc_active_per_odour_pct": 18.83, "median_pair_jaccard": 0.38, "mean_pair_jaccard": 0.377, "kcs_silent_pct": 41.8, "kcs_generalist_pct": 31.1, "brain_hz": 5.16, "thresh_kc_mean": -47.34, "t": 129}
{"iter": 5, "kc_active_per_odour_pct": 12.86, "median_pair_jaccard": 0.373, "mean_pair_jaccard": 0.363, "kcs_silent_pct": 58.3, "kcs_generalist_pct": 21.2, "brain_hz": 5.39, "thresh_kc_mean": -47.18, "t": 155}
{"iter": 6, "kc_active_per_odour_pct": 8.65, "median_pair_jaccard": 0.042, "mean_pair_jaccard": 0.204, "kcs_silent_pct": 58.2, "kcs_generalist_pct": 13.0, "brain_hz": 5.12, "thresh_kc_mean": -47.41, "t": 180}
{"iter": 7, "kc_active_per_odour_pct": 33.56, "median_pair_jaccard": 0.58, "mean_pair_jaccard": 0.454, "kcs_silent_pct": 26.8, "kcs_generalist_pct": 50.1, "brain_hz": 5.41, "thresh_kc_mean": -47.77, "t": 205}
{"iter": 8, "kc_active_per_odour_pct": 9.03, "median_pair_jaccard": 0.262, "mean_pair_jaccard": 0.308, "kcs_silent_pct": 64.3, "kcs_generalist_pct": 14.2, "brain_hz": 5.15, "thresh_kc_mean": -47.02, "t": 230}
{"iter": 9, "kc_active_per_odour_pct": 25.88, "median_pair_jaccard": 0.61, "mean_pair_jaccard": 0.609, "kcs_silent_pct": 45.1, "kcs_generalist_pct": 35.0, "brain_hz": 5.42, "thresh_kc_mean": -47.53, "t": 255}
{"iter": 10, "kc_active_per_odour_pct": 13.27, "median_pair_jaccard": 0.341, "mean_pair_jaccard": 0.371, "kcs_silent_pct": 56.2, "kcs_generalist_pct": 21.3, "brain_hz": 5.16, "thresh_kc_mean": -47.28, "t": 280}
{"iter": 11, "kc_active_per_odour_pct": 16.77, "median_pair_jaccard": 0.37, "mean_pair_jaccard": 0.348, "kcs_silent_pct": 50.0, "kcs_generalist_pct": 28.6, "brain_hz": 5.18, "thresh_kc_mean": -47.48, "t": 305}
{"iter": 12, "kc_active_per_odour_pct": 19.18, "median_pair_jaccard": 0.296, "mean_pair_jaccard": 0.295, "kcs_silent_pct": 47.1, "kcs_generalist_pct": 32.1, "brain_hz": 5.2, "thresh_kc_mean": -47.43, "t": 328}
{"iter": 13, "kc_active_per_odour_pct": 13.45, "median_pair_jaccard": 0.259, "mean_pair_jaccard": 0.344, "kcs_silent_pct": 59.0, "kcs_generalist_pct": 23.0, "brain_hz": 5.23, "thresh_kc_mean": -47.29, "t": 353}
{"iter": 14, "kc_active_per_odour_pct": 17.07, "median_pair_jaccard": 0.051, "mean_pair_jaccard": 0.219, "kcs_silent_pct": 42.5, "kcs_generalist_pct": 36.2, "brain_hz": 5.42, "thresh_kc_mean": -47.52, "t": 380}
Use: load kc_thresh.npy and use v >= thresh[i] in the kernel (comm_loop.py COMM_THRESH).
