# fly-v7 — fly-v6 + KC->MBON edge gain x24 (frozen 2026-09-17 09:55 EDT, Claude Code)
Graph: graph_v7_km24_am1.0.npz = fly-v6 graph with every KC->MBON edge x24 (29,169 edges); APL/DPM->MBON x1 (71 edges);
sha256(16) 516710dc67019ff2. Cell: refractory 3.8 ms (v3). Gains (calib.json, from v5/v6): PN x1.265, KC x1.790, MBON x0.856,
KC threshold +1.80 mV; APL x6.016 lives on the APL/DPM->KC edges (v6).
WHY: on v5/v6 a trained fly's rewarded-odour KC drive onto reward-side MBONs fell 60% (4,398 -> 1,741 mV) while those MBONs'
spikes did not change (92 -> 94): KC input was a minor term against APL inhibition (-33k mV) and non-KC excitation, and the
classic reward-side MBONs (01-07) never fired. In the fly, KC->MBON synapses are the MBONs' dominant odour drive (82% of
input weight here, statically). Sweep of KC->MBON x4/8/16/24 (results/mb_calib/v7_*.log), 3 flies x 6 reps, lr 0.2, 12 pairings:
  x4  drop_A 0.12 (0.15, 0.18, 0.04)   x8  0.18 (0.25, 0.16, 0.11)   x16 0.23 (0.35, 0.30, 0.05)   x24 0.21 (0.25, 0.26, 0.13), z<-2 in 3/3
  control odour drop within +-0.03 throughout; KC code reliability 0.66-0.68, overlap 0.02, 4.5% active at every setting.
CHOSEN: x24 (the only setting with every fly below z = -2). Nothing else changed. Grade 0b to be re-run on this graph.
