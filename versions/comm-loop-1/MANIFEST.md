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

## Added 2026-09-17 15:30 EDT — Gate 3 (a–f): two flies through a world; 3e PASS, 3f PASS (reproduction); Grade 0b identical after. SHA-256 (first 16):
- code/gpu/flysim_gpu.py  e24892007d4ded6d
- code/world/gate1.py  8d7cf17fb4aa4c3b
- code/world/gate1_protocol.md  4671e306093a6ca0
- code/world/gate3.py  60c119d23ad02b2f
- code/world/gate3_pilot.py  c1829f322d8056b1
- code/world/gate3_protocol.md  144dcdc58a8d2d40
- code/world/gate3_receiver.py  ab2e8b05df9fa46f
- code/world/gate3_sender.py  24db858f6f5f76cf
- code/world/gate3_world.py  7bc5d7114e630455
- code/world/receiver.py  65d301ccd2778d24
- code/world/sender.py  9966b840d0522644
- code/world/stack.py  14b378efaa77a0d7
- code/world/world.py  ed192c49ffbe51dc
- results/gate3/gate3/21101_receiver.json  e73e8b12c05c31fc
- results/gate3/gate3/21101_sender.json  0f898f0a09524021
- results/gate3/gate3/21101_world.json  275c5fcf5ce5a22e
- results/gate3/gate3/21202_receiver.json  1eef97ce7f3e262f
- results/gate3/gate3/21202_sender.json  82ab8de8c627e8ed
- results/gate3/gate3/21202_world.json  807b0f65d51c2dba
- results/gate3/gate3/21303_receiver.json  18cc53b21aa63dfb
- results/gate3/gate3/21303_sender.json  173abfd7d3c1e236
- results/gate3/gate3/21303_world.json  8ddb7ca9b4aca817
- results/gate3/gate3/21404_receiver.json  55ecb477eee09492
- results/gate3/gate3/21404_sender.json  62a3ca76dbccb31f
- results/gate3/gate3/21404_world.json  28cd6b512b0589a2
- results/gate3/gate3/21505_receiver.json  d0d2cf1b56d0bdf2
- results/gate3/gate3/21505_sender.json  371079bc1b766222
- results/gate3/gate3/21505_world.json  39082c4bbb935186
- results/gate3/gate3/21606_receiver.json  2264fe146abdd660
- results/gate3/gate3/21606_sender.json  531555e17c2006c2
- results/gate3/gate3/21606_world.json  ed04cbe5461f9739
- results/gate3/gate3/21707_receiver.json  3923296fea6ca609
- results/gate3/gate3/21707_sender.json  b60fc2bcbd40747f
- results/gate3/gate3/21707_world.json  76969a5c781f3da6
- results/gate3/gate3/21808_receiver.json  5cef3d6a262be344
- results/gate3/gate3/21808_sender.json  7789271de2ec8432
- results/gate3/gate3/21808_world.json  2014854dbc5ccd76
- results/gate3/gate3/gate3_summary.json  940f831787faa62f
- results/gate3/gate3b/22101_receiver.json  a5f90c71b77debd0
- results/gate3/gate3b/22101_sender.json  7788912133759916
- results/gate3/gate3b/22101_world.json  a5f0f3d3f879c9d0
- results/gate3/gate3b/22202_receiver.json  00800dca35c243d4
- results/gate3/gate3b/22202_sender.json  e3335252152a66a6
- results/gate3/gate3b/22202_world.json  0c214e7724a17dc0
- results/gate3/gate3b/22303_receiver.json  1c96e4bc1cca33b1
- results/gate3/gate3b/22303_sender.json  5d9683a8c8e353c3
- results/gate3/gate3b/22303_world.json  6c142ec1a9ec29d7
- results/gate3/gate3b/22404_receiver.json  8afffd4800abe001
- results/gate3/gate3b/22404_sender.json  176f927c9dfade5b
- results/gate3/gate3b/22404_world.json  0bab6d2ce26a170c
- results/gate3/gate3b/22505_receiver.json  6b3aa8a9984c9d5e
- results/gate3/gate3b/22505_sender.json  a1f35f6a0a5376f4
- results/gate3/gate3b/22505_world.json  5df2390993f7579a
- results/gate3/gate3b/22606_receiver.json  c6b2dcb21e19c038
- results/gate3/gate3b/22606_sender.json  a8075d7b5ddfedcc
- results/gate3/gate3b/22606_world.json  c545956431d23675
- results/gate3/gate3b/22707_receiver.json  8867cb1b64f1c9d3
- results/gate3/gate3b/22707_sender.json  9110593a398ecf51
- results/gate3/gate3b/22707_world.json  9a39fe6dfb8c0d4e
- results/gate3/gate3b/22808_receiver.json  67a61f6b0226ea7c
- results/gate3/gate3b/22808_sender.json  79e7de5730934f6b
- results/gate3/gate3b/22808_world.json  1a967aa007dcc3f4
- results/gate3/gate3b/gate3_summary.json  f20a14098abff14f
- results/gate3/gate3c/23101_receiver.json  8bfbe2550aeca3af
- results/gate3/gate3c/23101_sender.json  884b3ce9c1e1c0f9
- results/gate3/gate3c/23101_world.json  e9e1f061e73cf5fe
- results/gate3/gate3c/23202_receiver.json  bae111a2b5220a6f
- results/gate3/gate3c/23202_sender.json  cc0a5e383a81bf7f
- results/gate3/gate3c/23202_world.json  59d313f66f6df0e5
- results/gate3/gate3c/23303_receiver.json  9360250476fa7a20
- results/gate3/gate3c/23303_sender.json  67931ca4cd2ec007
- results/gate3/gate3c/23303_world.json  936bde32c923b16a
- results/gate3/gate3c/23404_receiver.json  b3d79635a309518e
- results/gate3/gate3c/23404_sender.json  6f8fe8b94c149378
- results/gate3/gate3c/23404_world.json  ec93636711f7ccb0
- results/gate3/gate3c/23505_receiver.json  61e3db8137d7ddae
- results/gate3/gate3c/23505_sender.json  f3969e02c0692da5
- results/gate3/gate3c/23505_world.json  2609cf9d7d15280d
- results/gate3/gate3c/23606_receiver.json  ce4b9579f041764b
- results/gate3/gate3c/23606_sender.json  3c15e87cf629b850
- results/gate3/gate3c/23606_world.json  81f3697e80526b44
- results/gate3/gate3c/23707_receiver.json  978dbbfc199f4ad1
- results/gate3/gate3c/23707_sender.json  9831ef35ce48cf61
- results/gate3/gate3c/23707_world.json  9dac41691442c7cf
- results/gate3/gate3c/23808_receiver.json  4b7e1502c9d827ae
- results/gate3/gate3c/23808_sender.json  a3c3244844ab6a0d
- results/gate3/gate3c/23808_world.json  ef9555f7c1f82b74
- results/gate3/gate3c/determinism.txt  379e907c0b9b43f7
- results/gate3/gate3c/gate3_summary.json  c6364bb66548fb82
- results/gate3/gate3d/24101_receiver.json  bcb5bab7aba51302
- results/gate3/gate3d/24101_sender.json  e4978fc76aa639ec
- results/gate3/gate3d/24101_world.json  310055ba05594260
- results/gate3/gate3d/24202_receiver.json  a504afb09dd625fa
- results/gate3/gate3d/24202_sender.json  0b11415c07fbf88e
- results/gate3/gate3d/24202_world.json  107071a1a4ab68ca
- results/gate3/gate3d/24303_receiver.json  4efcad93e82c1cd5
- results/gate3/gate3d/24303_sender.json  41bad467410c3523
- results/gate3/gate3d/24303_world.json  95c01b3a45d22291
- results/gate3/gate3d/24404_receiver.json  4ff4d4e2c4afaa17
- results/gate3/gate3d/24404_sender.json  5bf2116b109898d8
- results/gate3/gate3d/24404_world.json  c9bcd515dadd3b51
- results/gate3/gate3d/24505_receiver.json  657e3273f05047f9
- results/gate3/gate3d/24505_sender.json  5ead71d4a451ea70
- results/gate3/gate3d/24505_world.json  881a80fd54fccf98
- results/gate3/gate3d/24606_receiver.json  86ad648ad317ab2f
- results/gate3/gate3d/24606_sender.json  9b49882d7bcfa6e6
- results/gate3/gate3d/24606_world.json  95467b0b9f36ee22
- results/gate3/gate3d/24707_receiver.json  ab218284e5ada048
- results/gate3/gate3d/24707_sender.json  c93a41314e1091b9
- results/gate3/gate3d/24707_world.json  d90b5953ebd6f2bc
- results/gate3/gate3d/24808_receiver.json  b5c66dd30fc1deab
- results/gate3/gate3d/24808_sender.json  45e6dd95798032a9
- results/gate3/gate3d/24808_world.json  1b75969aa1a18c0e
- results/gate3/gate3d/gate3_summary.json  77fb5b5d79d6beae
- results/gate3/gate3e/25101_receiver.json  a0c3b47f29f5b436
- results/gate3/gate3e/25101_sender.json  16b1cfdbced4325a
- results/gate3/gate3e/25101_world.json  9d37697512b2230d
- results/gate3/gate3e/25202_receiver.json  e780bd5ffc49c4cf
- results/gate3/gate3e/25202_sender.json  e1bbdac35aa8d0c0
- results/gate3/gate3e/25202_world.json  e192306c6bf43125
- results/gate3/gate3e/25303_receiver.json  0bd863bffb0f49df
- results/gate3/gate3e/25303_sender.json  2e09bf3940e75091
- results/gate3/gate3e/25303_world.json  6fc24d93cceb56a3
- results/gate3/gate3e/25404_receiver.json  72f73bdadef78f88
- results/gate3/gate3e/25404_sender.json  dd8eecbdc85263a3
- results/gate3/gate3e/25404_world.json  12041aa0ab22a27a
- results/gate3/gate3e/25505_receiver.json  1bb4468c480da40d
- results/gate3/gate3e/25505_sender.json  e04bd274690551ab
- results/gate3/gate3e/25505_world.json  b5bdd18b4cf45ce2
- results/gate3/gate3e/25606_receiver.json  c0edfff859d54f09
- results/gate3/gate3e/25606_sender.json  971177a2f901db92
- results/gate3/gate3e/25606_world.json  bd95b173ae703eb9
- results/gate3/gate3e/25707_receiver.json  1c95487f7229ab23
- results/gate3/gate3e/25707_sender.json  305d8bb11339e7d0
- results/gate3/gate3e/25707_world.json  0a5c4be588a2fbf9
- results/gate3/gate3e/25808_receiver.json  1e9a346b81c0d570
- results/gate3/gate3e/25808_sender.json  dbbd4d63b30c8370
- results/gate3/gate3e/25808_world.json  c5c2253bd7a141ff
- results/gate3/gate3e/gate3_summary.json  8ec02ddb17065ca6
- results/gate3/gate3f/26101_receiver.json  e5a56cecdf53a559
- results/gate3/gate3f/26101_sender.json  23b60d81502009ca
- results/gate3/gate3f/26101_world.json  821815c48ee3deaa
- results/gate3/gate3f/26202_receiver.json  ae30135d713016ea
- results/gate3/gate3f/26202_sender.json  c8d6fe80e4ea470a
- results/gate3/gate3f/26202_world.json  1c86d5c517e10ba0
- results/gate3/gate3f/26303_receiver.json  e241be99c7bb30ad
- results/gate3/gate3f/26303_sender.json  b0aa69759550ed6c
- results/gate3/gate3f/26303_world.json  50b7b8bc3464e3c5
- results/gate3/gate3f/26404_receiver.json  2a329198c592d91e
- results/gate3/gate3f/26404_sender.json  c3e54c97156149f7
- results/gate3/gate3f/26404_world.json  6b8b6013db4928e5
- results/gate3/gate3f/26505_receiver.json  6287af097a6de687
- results/gate3/gate3f/26505_sender.json  b3354756ab38575a
- results/gate3/gate3f/26505_world.json  b9c6481e6a361693
- results/gate3/gate3f/26606_receiver.json  73f599ac71c72618
- results/gate3/gate3f/26606_sender.json  966020ca6a142484
- results/gate3/gate3f/26606_world.json  93df58c1659e3c0f
- results/gate3/gate3f/26707_receiver.json  fa505da0d076e28d
- results/gate3/gate3f/26707_sender.json  8a42cf1bea0115e5
- results/gate3/gate3f/26707_world.json  ca93c1b1dc72c0a9
- results/gate3/gate3f/26808_receiver.json  2fad9794f9f5508b
- results/gate3/gate3f/26808_sender.json  d97c6edeeb89211d
- results/gate3/gate3f/26808_world.json  080d56476cc0b203
- results/gate3/gate3f/gate3_summary.json  bc1f0d91704422e2
- results/gate3/grade0b_v10calib_post_gate3.json  216b8318117c08ab
