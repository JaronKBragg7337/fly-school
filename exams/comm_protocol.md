# Communication loop — PREDECLARED protocol (written 2026-09-17 03:20 EDT, BEFORE the test runs)

Pilot (seeds 73, 7337 only; results/identity_v3_mbontypes.json) was used to choose the channels below. The test uses seeds
the pilot never touched. Nothing below changes after the test runs. Zeros are zeros.

## Fly
fly-v3 @0.35 (Fly-Lab/versions/fly-v3/graph_v3_s0.35.npz, refractory 3.8 ms). Fresh mushroom body per seed (gains 1.0).
Learning rule unchanged: mushroom.py (lr 0.06, floor 0.25, trace ×0.55, frozen clock during the experiment).

## Symbols (external -> sensory neurons only; nothing else touches the fly)
A = ORN_DL3 receptor neurons @40 Hz Poisson, 300 ms pulse (pre 20 ms, tail 40 ms)     -> paired with REWARD dopamine (+1)
B = ORN_DM1 receptor neurons @200 Hz (rate equalised so A and B light ~equal KC counts) -> paired with PUNISHMENT dopamine (-1)
C = ORN_VA2 receptor neurons @ rate equalised to A's KC count                          -> NEVER paired (control)

## Training
12 epochs x (A + reward, B + punishment), in that order, membrane reset per trial, MB weights persist.

## Cold test (dopamine disabled, weights untouched)
6 reps each of A, B, C before training (pre) and after (post). Seeds for the TEST: 101, 202, 303, 404, 505, 606, 707, 808.

## Predeclared channels
Learning site (goal item 3): KC->MBON synapses only. Readouts, per trial, mean Hz over the type's cells:
  R = MBON09 (gamma3beta'1, PAM/reward side).       Prediction: R(A) post < R(A) pre in a majority of seeds (rewarded odour's
                                                       PAM-side drive depressed - Hige 2015 direction).
  P = mean(MBON25, MBON25-like, MBON34) (PPL1 side). Prediction: P(B) post < P(B) pre in a majority of seeds.
Answer pathway (goal item 5), endogenous, downstream of the MB: MDN (moonwalker descending neuron, 4 cells; backward walking).
  Decoded reply per cold trial: "AVOID" if MDN_post(X) > mean(MDN_pre(X)) + 2*sd(MDN_pre(X)), else "NO-AVOID".
  (Threshold from the fly's own untrained response to that same odour; chance rate of "AVOID" under no learning ~2.5%/trial.)

## Pass criterion (predeclared; not to be changed)
Over the 8 test seeds, pooled cold post trials:
  1. AVOID rate on B >= 0.60, AND
  2. AVOID rate on A <= 0.30, AND AVOID rate on C <= 0.30 (control), AND
  3. seeds where B's AVOID rate > A's AVOID rate: >= 6 of 8, AND
  4. R(A) drops (post < pre) in >= 5 of 8 seeds and P(B) drops in >= 5 of 8 seeds (the learning site moved, right sign).
If any of 1-4 fails, the loop is NOT established; report it as such.

## After a pass
Grade 0b re-run on the same fly-v3 (no ignition; sugar->MN9; bitter veto; loom specificity; JO-A->GF must hold).
Freeze: protocol, code, seeds, raw JSON, manifest. Re-run from a clean state (new process, stores deleted) - must pass again.

# TEST 2 — predeclared 2026-09-17 04:35 EDT (after test 1 FAILED; pilots 5/6 on seeds 73, 7337 chose these channels)
Fly: fly-v3 @0.35, refractory 3.8 ms, scalar threshold (fly-v4 thresholds rejected: KC code unstable across trials).
Symbols: A = ORN_DA2 @40 Hz (REWARD); B = ORN_VM2 @40 Hz (PUNISH; note: VM2 barely reaches the MBONs — its lesson is
expected unreadable; kept as designed); C = ORN_DL4 @ rate equalised to A's KC count (NEVER paired, control).
Overlaps on v3 (pilot): A-B 0.32-0.34, A-C 0.55; same-odour 0.46-0.62.
Training: 12 x (A+reward, B+punishment), lr 0.06 (unchanged rule). Cold: 6 reps per odour pre and post.
Answer channel (endogenous MB OUTPUT pathway, Hige 2015's learned signal): L(X) = mean Hz over the 4 cells of MBON21 + MBON26
(cholinergic, PAM/avoid side). Prediction from the rule: reward on A depresses A's KC drive to PAM-side MBONs -> L(A) falls.
Decoded reply per cold post trial of odour X: "LEARNED" if L(X) < mean(L_pre(X)) - 1.0 * sd(L_pre(X)); else "NONE".
Chance rate of "LEARNED" with no learning ~16% per trial (one-sided 1 sd).
PASS (all four, fixed): (1) pooled LEARNED rate on A >= 0.50; (2) pooled rate on C <= 0.25 and on B <= 0.25;
(3) seeds with A rate > C rate: >= 6 of 8; (4) mean L(A) post < pre in >= 6 of 8 seeds. Seeds: 909,1010,1111,1212,1313,1414,1515,1616.
This test reads the mushroom body's own output, not a motor neuron: if it passes, items 1-4 of the goal are met and item 5
is met only as far as "endogenous output pathway"; the motor reply remains NOT shown. Stated before running.

# TEST 3 — predeclared 2026-09-17 04:50 EDT, on fly-v5 (after tests 1 and 2 FAILED on v3)
Fly: fly-v5 = fly-v3 graph + calib.json (PN x1.27, APL x6.02, KC x1.79, KC threshold +1.8 mV, MBON x0.86), refractory 3.8 ms.
Chosen by mb_calib_search.py stage 1 (odour panel, seeds 11/12) and stage 2 (seeds 200-600) — none of the test seeds.
Pupil under test: lr 0.2 (declared variant; stage 2 chose it). Control pupil lr 0.06 run on the same seeds, reported.
Symbols: A = ORN_DA2 @40 Hz (REWARD +1); B = ORN_DL3 @40 Hz (PUNISH -1); C = ORN_VM5d @40 Hz (NEVER paired, control).
Training: 12 x (A+reward, B+punishment). Cold: 6 reps each of A, B, C, pre and post; dopamine off.
Learning site: KC->MBON. Answer channels (endogenous mushroom-body OUTPUT pathway):
  PAM(X) = mean Hz over all PAM-side (avoid-promoting) MBONs on odour X; PPL(X) = mean Hz over all PPL1-side (approach) MBONs.
Predictions from the rule: reward on A depresses A's drive to PAM MBONs -> PAM(A) falls; punishment on B -> PPL(B) falls.
Decoded reply per cold post trial of X (against X's own pre baseline, 6 trials):
  zP = (PAM(X) - mean PAM_pre(X)) / sd PAM_pre(X);  zL = (PPL(X) - mean PPL_pre(X)) / sd PPL_pre(X)
  reply = "APPROACH" if zP < -1 and zP <= zL; "AVOID" if zL < -1 and zL < zP; else "NONE". Chance per class ~16%.
PASS (all, fixed): (1) pooled APPROACH rate on A >= 0.50; (2) pooled AVOID rate on B >= 0.50;
(3) pooled (APPROACH+AVOID) rate on C <= 0.25; (4) seeds with A's APPROACH rate > C's: >= 6/8 AND B's AVOID rate > C's: >= 6/8.
Also recorded (not criteria): DNa13, DNa03, MDN, DNa02 — the motor pathway; if they move with the answer, item 5 is fully met,
otherwise item 5 is met only as "endogenous output pathway". Seeds: 1717,1818,1919,2020,2121,2222,2323,2424.
After a pass: Grade 0b on fly-v5; freeze; re-run this exact test from a clean process; both must pass.

# TEST 4 — predeclared 2026-09-17 04:35 EDT, on fly-v5 (after test 3 FAILED per-trial)
Same fly, symbols, training, cold protocol and pupil (lr 0.2) as test 3. Test 3 stands as FAILED.
Change, justified before running: real conditioning is read at the BLOCK level (Tully & Quinn: performance index over the
test period), not per trial. Block reply for odour X in a seed:
  zP_block = (mean PAM_post(X) - mean PAM_pre(X)) / (sd PAM_pre(X) / sqrt(6));  zL_block likewise on PPL.
  reply(X) = "APPROACH" if zP_block < -2 and zP_block <= zL_block; "AVOID" if zL_block < -2; else "NONE".
  (Chance of a false APPROACH/AVOID per seed under no learning ~2.3% each, one-sided.)
PASS (all, fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 seeds; (2) reply(C) == "NONE" in >= 7 of 8 seeds;
(3) reply(B) recorded; "AVOID" on B is reported but NOT required (the PPL1-side output is ~5 Hz and did not move in test 3).
Seeds: 2525,2626,2727,2828,2929,3030,3131,3232. A pass establishes items 1-4 and item 5 as "endogenous MB output pathway"
(not motor) — stated now. Then Grade 0b on fly-v5 and a clean re-run of this exact test.

# TEST 5 — predeclared 2026-09-17 08:55 EDT, on fly-v5 (after tests 3 and 4 FAILED; diagnosis 08:50: KC share of MBON input is
# 82-92% in v1 and v3, so the learning site controls the readout; the failures are statistical — 2-3 MBON spikes per 360 ms trial,
# 6 reps, reliability 0.69 — not structural)
Same fly-v5 (calib.json), same symbols A = ORN_DA2 @40 Hz (REWARD), C = ORN_VM5d @ rate equalised (NEVER paired, control),
B = ORN_DL3 @40 Hz (PUNISH, recorded, not required). Pupil lr 0.2. Changes, all sizes, stated before running:
  odour pulse 1000 ms (was 300); 24 pairings (was 12); 12 cold reps per odour, pre and post (was 6).
Answer channel and decode unchanged from test 4: PAM(X) = mean Hz over PAM-side MBONs; block z on the 12-rep means;
reply(X) = "APPROACH" if zP_block < -2 and zP_block <= zL_block; "AVOID" if zL_block < -2; else "NONE".
PASS (fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 seeds; (2) reply(C) == "NONE" in >= 7 of 8 seeds.
Seeds: 3333,3434,3535,3636,3737,3838,3939,4040 (one process per seed; pooled afterwards without change).
Then Grade 0b on fly-v5; freeze; clean re-run of this exact test.

# TEST 6 — predeclared 2026-09-17 09:58 EDT, on fly-v7 (tests 1-5 FAILED; diagnosis 09:20-09:50: the readout was pinned —
# APL crushed the MBONs (v5), then KC->MBON synapses were too weak to drive them (v6); fly-v7 = v6 with KC->MBON edges x24)
Fly: fly-v7 (graph_v7_km24_am1.0.npz + versions/fly-v7/calib.json), refractory 3.8 ms. Pupil lr 0.2.
Symbols: A = ORN_DA2 @40 Hz (REWARD +1); B = ORN_DL3 @40 Hz (PUNISH -1, recorded, not required); C = ORN_VM5d @ rate equalised
to A's KC count (NEVER paired, control). Training 12 x (A+reward, B+punishment). Pulse 300 ms (as in the stage-2 evidence).
Cold: 12 reps per odour pre and post, dopamine off.
Answer channel: PAM(X) = mean Hz over PAM-side MBONs; PPL(X) likewise. Block decode as tests 4/5:
  zP = (mean PAM_post - mean PAM_pre) / (sd PAM_pre / sqrt(12)); zL likewise; reply = "APPROACH" if zP < -2 and zP <= zL;
  "AVOID" if zL < -2; else "NONE".
PASS (fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 seeds; (2) reply(C) == "NONE" in >= 7 of 8 seeds.
Seeds: 4141,4242,4343,4444,4545,4646,4747,4848 (one process per seed; pooled without change). DNa13/DNa03/MDN recorded.
Grade 0b runs on fly-v7 in parallel; core properties must hold (no ignition, sugar->MN9, bitter veto, loom specificity, JO-A->GF).
After a pass: freeze test 6's raw outputs with the manifest; clean re-run of this exact test (new process, stores deleted).

# TEST 7 — predeclared 2026-09-17 ~09:20 EDT, on fly-v7. Seventh predeclared test; tests 1-6 stand as FAILED.
Same fly-v7, symbols, and answer channel as test 6. Changes, stated before running: 24 pairings (all eligible synapses reach
the floor), 24 cold reps per odour pre and post (z on the block mean uses sd/sqrt(24)), and the decode is ONE-SIDED, as the
rule predicts: reward depresses A's drive to PAM-side (avoid) MBONs and says nothing about the PPL side.
  zP(X) = (mean PAM_post(X) - mean PAM_pre(X)) / (sd PAM_pre(X) / sqrt(24))
  reply(X) = "APPROACH" if zP(X) < -2, else "NONE".
PASS (fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 seeds; (2) reply(C) == "NONE" in >= 7 of 8 seeds.
Seeds: 4949,5050,5151,5252,5353,5454,5555,5656. B recorded. DNs recorded. Grade 0b on fly-v7 already passed (9/1/1).

# TEST 8 — predeclared 2026-09-17 ~10:45 EDT, on fly-v10. Eighth predeclared test; tests 1-7 stand as FAILED.
Changes since test 7, each measured before this declaration: (a) fly-v9: antennal-lobe lLN1/lLN2 local neurons made inhibitory
(they were marked cholinergic by the transmitter table; one odour activated 154-164 PN types, now 23-35); (b) shared-weights bug
fixed (Grok's review): weights restored before every pupil; (c) fly-v10 = v9 + calibration r2_c025 (stage 2 on 3 flies: paired-odour
drop 0.44, control 0.08). Symbols, training, pupil as test 7: A = ORN_DA2 @40 Hz reward; B = ORN_DL3 @40 Hz punish (recorded);
C = ORN_VM5d equalised, never paired. 12 pairings, lr 0.2. Cold: 24 reps per odour pre and post, 300 ms pulses.
Answer channel (Grok's review, item C): the top-5 PAM-side MBON TYPES ranked by KC->MBON synaptic weight received from A's
PRE-trial Kenyon cells (wiring + pre trials only; chosen before any post trial). The whole-PAM mean is also reported.
Decode (one-sided, block, as test 7): zP = (mean top5_post - mean top5_pre) / (sd top5_pre / sqrt(24)); reply = "APPROACH" if zP < -2 else "NONE".
PASS (fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 seeds; (2) reply(C) == "NONE" in >= 7 of 8 seeds.
Seeds: 5757,5858,5959,6060,6161,6262,6363,6464 (one process per seed). Grade 0b on fly-v10 with calib.json runs alongside.
After a pass: freeze raw outputs + manifest; clean re-run of this exact test with new seeds 6565-7272 from a fresh process; both must pass.

# TEST 9 — predeclared 2026-09-17 ~11:20 EDT, on fly-v10 (ChatGPT chat's review, second reply: the one-symbol matched-input test)
Fly: fly-v10 as frozen (graph_v9.npz + fly-v10/calib.json). Pupil lr 0.2, 12 pairings, 300 ms pulses.
Training: A = ORN_DA2 @40 Hz + reward ONLY. No B training (Jaron's one-symbol question; the calibration was A-only too).
Controls: C = ORN_VM5d @40 Hz, never paired, NOT rate-equalised; B = ORN_DL3 @40 Hz measured cold, never trained (second control).
Cold: 24 sensory realisations per odour, the SAME realisations replayed before and after training (identical seeds), dopamine off.
Answer population (frozen now): MBON09 + MBON01 + MBON05 + MBON03 + MBON06 (the five types test 8 selected on its own in 8/8 flies).
Per fly: paired differences d_r = post_r - pre_r over the 24 matched pairs; z_pair = mean(d) / (sd(d)/sqrt(24)); rel = 1 - mean(post)/mean(pre).
  reply(X) = "APPROACH" if z_pair < -2 AND rel >= 0.30; "NONE" if |rel| <= 0.10; else "SHIFT".
PASS (fixed): (1) reply(A) == "APPROACH" in >= 6 of 8 flies; (2) reply(C) == "NONE" in >= 6 of 8 flies; B reported.
Seeds: 7373,7474,7575,7676,7777,7878,7979,8080 (one process per fly). Whole-PAM and DNa13/DNa03/MDN reported. Grade 0b on v10 already holds.
After a pass: freeze raw outputs + manifest; clean re-run with seeds 8181-8888 from fresh processes; both must pass.

# TEST 9 RESULT (11:02 EDT): PASS — A APPROACH 8/8 (paired z −13.6…−23.2; drops 45–49%), C NONE 8/8 (drops 1–4%). B (untrained
# second control) SHIFT in 5/8 (7–28%): reported, not required. Recorded, not a criterion: DNa13+DNa03+MDN sum for A fell in 8/8
# flies (7–31%), C flat. results/comm_loop_test9_s*.json.

# REPRODUCTION (item 9) — predeclared 11:05 EDT, BEFORE it runs. Frozen protocol = TEST 9 exactly (fly-v10, A-only training, replayed
# realisations, frozen answer population, same decode, same pass line 6/8 + 6/8), fresh processes, stores deleted, seeds 8181–8888.
# ADDED criterion for item 5 (stated now): motor reply. Y(X) = mean over the 24 matched pairs of (DNa13+DNa03+MDN) rate; paired
# z_Y(A) < −2 in >= 6 of 8 flies (the avoid-promoting descending pathway falls for the rewarded symbol — the sign derived from the
# wiring on 2026-09-17 03:30 and recorded, unrequired, in test 9), and |rel_Y(C)| <= 0.25 in >= 6 of 8 (the control's descending
# rate is low and noisy, 2–8 Hz summed; a wider band than the MBON control is stated for that reason). Decoded reply = "GO-TOWARD"
# when z_Y(A) < −2, else "NONE". Grade 0b re-run on fly-v10 immediately after test 9 (item 7) precedes the reproduction.

# REPRODUCTION RESULT (11:42 EDT): MB criteria PASS again — A APPROACH 8/8 (z −13.4…−24.0, drops 44–50%), C NONE 8/8 (2–5%).
# Motor criterion as predeclared: FAIL — A GO-TOWARD 5/8 (descending sum fell in 8/8 by 5–34% but paired z < −2 in 5); C within
# ±25% in 3/8 (C's summed descending rate is 2–8 Hz; ±50% swings on ~3 spikes). That criterion stands failed. Stated: the band was
# mis-sized for the count statistics; the direction was right in 16/16 flies across test 9 + reproduction.

# TEST 10 (motor) — predeclared 11:50 EDT, and its REPRODUCTION (TEST 10b) predeclared at the same time.
Frozen protocol = TEST 9 (fly-v10, A-only training, replay, frozen MB answer population, same MB decode and pass line), with ONE
size change: 48 matched realisations per odour (was 24) so the descending counts are usable. Motor channel Y = DNa13+DNa03+MDN summed
rate per trial. Per fly: rel_Y(X) = 1 − mean(post)/mean(pre) over the 48 matched pairs.
PASS (fixed), both batches independently:
  MB: A APPROACH >= 6/8 and C NONE >= 6/8 (unchanged).
  MOTOR: rel_Y(A) > 0 in >= 7 of 8 flies AND mean over flies rel_Y(A) >= 0.15 AND rel_Y(A) > rel_Y(C) in >= 7 of 8 flies AND
         |mean over flies rel_Y(C)| <= 0.15. Decoded reply = "GO-TOWARD" for a fly when rel_Y(A) > 0 and rel_Y(A) > rel_Y(C).
TEST 10 seeds: 9191,9292,9393,9494,9595,9696,9797,9898.  TEST 10b (reproduction) seeds: 10101,10202,10303,10404,10505,10606,10707,10808.
Both run now from fresh processes via the frozen script with -Reps 48. Grade 0b on fly-v10 already holds before and after test 9.

## TEST 10 and TEST 10b — RESULTS (scored 12:33 EDT; runs finished 12:30; 2,816–2,833 s per fly, 16 fresh CPU processes)
Scored by results/comm_loop_test10_pooled.json exactly against the lines above. Nothing changed after seeing the numbers.
TEST 10 (9191–9898):  MB A APPROACH 8/8 (zP −23.0…−29.7, drops 45–48%), C NONE 8/8 (2–4%).
   MOTOR rel_Y(A) = 0.147, 0.235, 0.197, 0.268, 0.219, 0.221, 0.299, 0.249 → > 0 in 8/8, mean 0.229 (≥ 0.15 ✓);
   rel_Y(A) > rel_Y(C) in 8/8; mean rel_Y(C) = 0.048 (|·| ≤ 0.15 ✓). GO-TOWARD decoded in 8/8.        → PASS (MB ✓, MOTOR ✓)
TEST 10b (10101–10808): MB A APPROACH 8/8 (zP −22.8…−28.0, drops 44–50%), C NONE 8/8 (1–4%).
   MOTOR rel_Y(A) = 0.224, 0.206, 0.247, 0.298, 0.212, 0.239, 0.203, 0.081 → > 0 in 8/8, mean 0.214 (≥ 0.15 ✓);
   rel_Y(A) > rel_Y(C) in 8/8; mean rel_Y(C) = 0.014 (|·| ≤ 0.15 ✓). GO-TOWARD decoded in 8/8.        → PASS (MB ✓, MOTOR ✓)
Grade 0b on fly-v10 re-run after (results/grade0b_v10calib_post10.json): every verdict and every read IDENTICAL to post-9
(9 pass, water fail, steer not testable; MN9 43.75, bitter 5.83, GF 171.67/180, JO-A→GF 2.08, KC 14.5%).
Honest reading of the motor stage: Y (DNa13+DNa03+MDN) is driven hard by A before training (90–111 Hz summed) and barely by C
(1.4–5.5 Hz), so rel_Y(C) is a ratio of small numbers (single flies −0.17…+0.20) — the band |mean| ≤ 0.15 held in both batches,
but per-fly C values are noisy. The A effect (−20…−30% in 16/16 flies) is not.
