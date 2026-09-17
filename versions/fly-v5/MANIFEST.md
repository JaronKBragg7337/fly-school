# fly-v5 — fly-v3 + mushroom-body calibration c007 (frozen 2026-09-17 04:45 EDT, Claude Code)
Graph: fly-v3 graph_v3_s0.35.npz (unchanged; sha ed3b1f330b665ad3). Cell: refractory 3.8 ms (v3). NEW (calib.json): per-type gain
multipliers on the CHOSEN calibration pn05_apl10_kc03 — PN x1.265, APL x6.016, KC x1.790, MBON x0.856, and
KC spike threshold +1.80 mV. Found by mb_calib_search.py round 0, candidate 7 (96 random candidates; results/mb_calib/).
Stage 1 (odour panel of 8 @40 Hz): same-odour reliability 0.69, median pair overlap 0.06, 4.6% KCs active, brain 4.9 Hz/cell.
Stage 2 (A=DA2 rewarded 12x, C=VM5d never paired; PAM-side MBON output): lr 0.06: A -21%, C -2%; 24 pairings: A -27%, C -2%;
lr 0.2 (declared variant): A -31%, C +9% -> passes the predeclared stage-2 bar (>=30% / <=10%).
Why: v1-v3 KC codes shared 0.4-0.7 across odours; v4 (per-KC thresholds) was unreliable. APL is the sparsening inhibitor (biology).
{
 "round": 0,
 "cand": 7,
 "params": {
  "pn": 1.2653676328049124,
  "apl": 6.016211158078346,
  "kc": 1.789810060699586,
  "kc_thresh_shift_mv": 1.8016575199247349,
  "mbon": 0.8561399722476636
 },
 "stage1": {
  "reliability": 0.693,
  "median_overlap": 0.062,
  "kc_frac": 0.0458,
  "brain_hz": 4.92,
  "score": 0.0,
  "stage1_pass": true
 },
 "t": 45,
 "stage2": {
  "pam_A_pre": 6.93,
  "pam_A_post": 5.49,
  "pam_C_pre": 17.77,
  "pam_C_post": 18.06,
  "drop_A": 0.208,
  "drop_C": -0.016,
  "stage2_pass": false,
  "depressed": 2438
 },
 "stage2_e24_lr0.06": {
  "pam_A_pre": 6.93,
  "pam_A_post": 5.08,
  "pam_C_pre": 17.77,
  "pam_C_post": 17.38,
  "drop_A": 0.267,
  "drop_C": 0.022,
  "stage2_pass": false,
  "depressed": 3073
 }
}
