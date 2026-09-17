# comm-loop-1 — the frozen communication-loop configuration (2026-09-17 11:10 EDT, Claude Code)

Result to reproduce: TEST 9 (comm_protocol.md) PASS 8/8 + 8/8 on fly-v10; Grade 0b before and after identical (results/).
Fly: graph fly-v9 (fly-v7 = v3 graph + APL x6.016 on APL->KC edges + KC->MBON x24; + lLN antennal-lobe local neurons inhibitory), calib.json (fly-v10 r2_c025 gains), refractory 3.8 ms (flysim_v3.ParamsV3).
Learning rule: code/mushroom.py (dopamine-gated KC->MBON depression, trace x0.55, floor 0.25), lr 0.2 (COMM_LR), sides = mb_sides.json (PAM/PPL1 by DAN input, verified vs Aso 2014).
Symbols: A = ORN_DA2 @40 Hz Poisson, 300 ms pulse (20 ms pre, 40 ms tail), paired with reward (+1) x12; C = ORN_VM5d @40 Hz never paired; B = ORN_DL3 @40 Hz never trained (second control).
Cold test: 24 realisations per odour, seeds sha256('comm:{seed}:cold:{odour}:{rep}') replayed before and after training; dopamine off; membrane reset per trial; weights restored to the graph before each fly.
Answer population: MBON09+MBON01+MBON05+MBON03+MBON06 (all cells of those types). Decode per fly: paired z over the 24 matched pairs, reply APPROACH if z<-2 and drop>=30%; NONE if |drop|<=10%.
Pass: APPROACH on A in >=6/8 flies and NONE on C in >=6/8. Seeds test 9: 7373..8080. Reproduction seeds: 8181..8888 (+ motor criterion, see protocol).
Run: powershell -File run_test9.ps1 -Seeds 8181,8282,... -Tag repro   (from a fresh shell; requires the fly-brain venv, graph_v9.npz, calib.json, mb_sides.json in place)

SHA-256 (16):
  8eceb2264cff5a38  graph
  74115f226e854a1b  calib.json
  572e75291b732799  mb_sides.json
  2abf791d36fa957c  code/calibration.py
  7569536266ba3292  code/comm_loop.py
  5c670598856f557b  code/comm_protocol.md
  f977e29b16135bd6  code/flysim.py
  89531151fc2b254d  code/flysim_v3.py
  fee4a771be670721  code/grade0b_organs.py
  02da9a55133fa7c4  code/mb_calib_search.py
  d3147ae6188a77b4  code/mushroom.py

## Added 2026-09-17 12:40 EDT — test 10 / 10b (PASS, MB+motor), Grade 0b post-10, GPU kernel (verified), Gate 1 (PASS)
torch 2.6.0+cu124, RTX 4060 Laptop GPU. SHA-256 (first 16):
- code/gpu/flysim_gpu.py  a957dfa603622ad0
- code/gpu/gpu_verify.json  0da5d333c8cbdfc4
- code/gpu/verify_gpu.py  c2be01052abfc3da
- code/world/gate1.py  8d7cf17fb4aa4c3b
- code/world/gate1_protocol.md  4671e306093a6ca0
- code/world/receiver.py  65d301ccd2778d24
- code/world/sender.py  9966b840d0522644
- code/world/stack.py  14b378efaa77a0d7
- code/world/world.py  ed192c49ffbe51dc
- results/gate1/gate1_summary.json  664d615e32bff879
- results/test10/comm_loop_test10_pooled.json  092ee8520c0ce2b0
- results/test10/comm_loop_test10_s9191.json  c84455b03b24e59b
- results/test10/comm_loop_test10_s9292.json  39a63556a687b97a
- results/test10/comm_loop_test10_s9393.json  139cd787f28516da
- results/test10/comm_loop_test10_s9494.json  6778cb28ca7690a4
- results/test10/comm_loop_test10_s9595.json  77dc8d4a305b5f58
- results/test10/comm_loop_test10_s9696.json  823d554474a66e98
- results/test10/comm_loop_test10_s9797.json  7a599fdfe8ee5ae9
- results/test10/comm_loop_test10_s9898.json  d1d48d2988215ef7
- results/test10/comm_loop_test10b_s10101.json  0d3564b18f21d4b0
- results/test10/comm_loop_test10b_s10202.json  8131274361446406
- results/test10/comm_loop_test10b_s10303.json  a34cf964ab7ee6f7
- results/test10/comm_loop_test10b_s10404.json  20182c6b636fdaeb
- results/test10/comm_loop_test10b_s10505.json  8e380148accafaac
- results/test10/comm_loop_test10b_s10606.json  dd1ae2762cedbacd
- results/test10/comm_loop_test10b_s10707.json  30f2651a5c417eb3
- results/test10/comm_loop_test10b_s10808.json  ffc3ce0348fc261d
- results/test10/grade0b_v10calib_post10.json  216b8318117c08ab
