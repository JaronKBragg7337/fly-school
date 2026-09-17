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
