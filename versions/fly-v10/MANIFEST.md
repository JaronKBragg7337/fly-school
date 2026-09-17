# fly-v10 — fly-v9 graph + mushroom-body calibration r2_c025 (frozen 2026-09-17 ~10:40 EDT, Claude Code)
Graph: fly-v9 graph_v9.npz (fly-v7 + antennal-lobe lLN1/lLN2 local neurons made inhibitory). Cell: refractory 3.8 ms.
calib.json (per-type gains on pn05_apl10_kc03; applied by comm_loop.py / grade0b_organs.py): PN x2.962, APL x1.442 (per-neuron, on top of
the APL->KC x6.016 baked into the graph), KC x2.015, MBON x2.873, KC threshold +6.70 mV.
Search: round 2, 96 random candidates on v9 (results/mb_calib/r2_*.json); stage 1 (odour panel): reliability 0.717, median pair overlap
0.128, KC active 6.0%, brain 3.43 Hz/cell. Stage 2 (3 flies, weights restored per fly, 8 cold reps, lr 0.2,
12 pairings, A=DA2 rewarded vs C=VM5d never paired, PAM-side MBON output): drop_A 0.438 (0.503, 0.418, 0.392), drop_C 0.076 (0.093, 0.032, 0.103), z<-2 in 3/3.
Other stage-2 passes on v9: c060 (0.48/0.03), c056 (0.40/0.07), c043 (0.32/0.06). CHOSEN: c025 for control <= 0.10 in every fly.
Bugs fixed before this search (Grok chat review, 2026-09-17): shared fb.wdata across pupils in one process (comm_loop.py, mb_calib_search.py).
