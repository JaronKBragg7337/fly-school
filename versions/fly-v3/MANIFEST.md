# fly-v3 — lulzx's recipe at the graph level (frozen 2026-09-17 03:20 EDT, Claude Code)

Forked from fly-v1's graph.npz. Changes (from Grok chat's table of lulzx/fly-brain docs/07-calibration.md):
  1. connections with fewer than 6 synapses dropped (|w|/0.275 mV < 6): 10,228,000 -> 4,912,241 edges (with 2.)
  2. sensory neurons receive no central input (rows whose POST is superclass *sensory* zeroed)
  3. every remaining weight x 0.35 (CHOSEN by Grade 0b sweep 0.35/0.5/0.7/1.0; 0.5 passes water but ignites on odour)
  4. refractory 3.8 ms (flysim_v3.ParamsV3) instead of 2.2
Not ported: conductance-based synapses, size-scaled PSPs, KC threshold +10.9 mV, inhibition gain 0.61, lamina bias.
Grade 0b on v3@0.35 (results/grade0b_v3_s0.35.json): sugar->MN9 36 Hz PASS, bitter veto 36->8 PASS, loom x3 PASS with
LC4->DNp02 / LPLC2->DNp06 specificity, JO-A->GF 4.2 Hz PASS (v2 failed), JO-B->AMMC PASS, grooming DNs PASS, MDN responds;
water FAIL, DM1 KC 2.6% FAIL (target 5-10%), lateral steer not testable. 8 pass / 2 fail. v2 was 7 / 3. v1 0 / 11 ignite.
SHA-256 (16): graph_v3_s0.35.npz ed3b1f330b665ad3 · s0.5 c9490aac3382e539 · s0.7 17d4f2a148ea7a0b · s1.0 8405cd872d6ec294
Use: from flysim_v3 import FlyBrainV3; FlyBrainV3(r"C:/Users/lilli/Fly-Lab/versions/fly-v3/graph_v3_s0.35.npz")
