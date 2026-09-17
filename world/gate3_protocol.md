# Gate 3 — state-gated emission: a sender's LEARNED state changes what a receiver smells, through a world
Predeclared 2026-09-17 12:58 EDT by Claude Code (Grok chat's Gate 3, 2026-09-17). Written BEFORE any test-seed run. Nothing below
changes after the numbers are seen. Pilot (world/gate3_pilot.py, seeds 30101/30202, notebook only) showed: the mushroom-body lesson
does NOT reach the song circuit with a consistent sign (pulse-song wing MNs −5…+15%, mixed), and odours alone drive the wing motor
neurons at 50–80 Hz — so song is not usable as the emission channel on this stack. The lesson DOES reach the moonwalker descending
neurons MDN (test 10/10b: A −15…−36% in 16/16 flies). MDN drives backward walking (Bidaye 2014). A male carries cVA on his cuticle
and needs no command to emit it; a nearby male is smelled through Or67d → DA1 (Kurtovic 2007). So the channel is: the sender's
learned state changes how far it backs away from the odour source, and the receiver sitting at the source smells more or less cVA.

## The claim being tested
Sender S (trained on odour A + reward, never on C) and receiver R (untrained, never smells A or C) are separate OS processes on the
frozen fly-v10 stack. The ONLY shared thing is a world file holding cVA concentration at R, computed by world physics from S's MDN
spikes. After S learns, R's cVA neurons and their projection neurons respond differently to the world — and only through the world.

## CHOSEN (declared once)
- Stack fly-v10, GPU kernel (verified bit-identical). Trial 20/300/40 ms (comm-loop timing). Learning = test 9 exactly: A (ORN_DA2 @40 Hz)
  + reward dopamine, 12 epochs, lr 0.06, floor 0.25, A-only; weights restored from the graph before every fly.
- Sender readout to the world: MDN (4 cells) mean rate over the 300 ms trial. 24 replayed realisations per odour (A, C = ORN_VM5d @40 Hz,
  no equaliser), identical pre and post (paired), realisation seeds sha256("gate3:<seed>:cold:<X>:<rep>").
- World physics (1-D, declared; a kinematic stand-in like gait.py): S starts d_start = 20 mm from R; backward speed = k · MDN_Hz with
  k = 0.5 mm/s per Hz; bout = 2 s → d = 20 + MDN_Hz mm. cVA at R: c(d) = 1 / (1 + (d / d0)²), d0 = 40 mm. R's ORN_DA1 (204 cells) Poisson
  rate = 60 Hz · c. The world reads MDN spikes only. It never reads a mushroom body.
- Receiver: fresh untrained fly-v10, seed = sender seed + 1; ORN_DA1 hit draws made on every step (paired across conditions);
  readout DA1_vPN (2 cells) and DA1_lPN (13 cells) mean rate; pC1 reported.
- Seeds: 21101, 21202, 21303, 21404, 21505, 21606, 21707, 21808 (never used).

## Conditions per fly
LIVE:  R hears the world computed from S's actual MDN, pre and post training (24 paired realisations per odour).
MUTE:  the world is clamped to S's PRE field for the post block (channel lesion) — R's post must equal R's pre exactly.
LEAK:  R's ORN_DA2 and ORN_VM5d spike counts must be zero in every trial — R never smells the odours.

## PASS (all)
- S1  MDN(A) post < pre (mean over the 24 pairs) in ≥ 7 of 8 flies; rel_MDN(A) > rel_MDN(C) in ≥ 7 of 8.
- W1  cVA at R for odour A: post > pre (mean over pairs) in ≥ 7 of 8 flies.
- R1  DA1_vPN rate at R for odour A: post > pre in ≥ 7 of 8 flies AND paired z = mean(Δ)/(sd(Δ)/√24) ≥ +2 in ≥ 6 of 8.
- R2  DA1_vPN for odour C: |rel change| ≤ 0.15 in ≥ 6 of 8 flies.
- R3  MUTE: R's post spike-count vectors identical to R's pre in 8 of 8 (the only route is the world).
- R4  LEAK: zero ORN_DA2 / ORN_VM5d spikes in R, all trials, 8 of 8.
- G   no ignition (< 10 Hz/cell) anywhere; Grade 0b on fly-v10 re-run after, core properties identical.
Decoded reply at R (predeclared): "SENDER-CAME-CLOSER" when R1's paired z ≥ +2, else "NO-CHANGE".

## FAIL readings
S1 fails → the lesson did not reach MDN in these flies (contradicts test 10; report). W1/R1 fail with S1 passing → the world or the
nose is too coarse at these constants (report; do not retune inside the test). R3/R4 fail → harness leak; fix, redeclare, new seeds.

## Outputs
world/results/gate3/<seed>_sender.json (per-trial MDN, PAM5, fields), <seed>_receiver.json, gate3_summary.json.

## GATE 3 RESULT (13:00 EDT, run 12:41–12:53, 8 flies, 705 s) — FAIL at S1.
MDN(A) pre→post: 49.1→45.6, 45.5→44.7, 40.4→45.5, 43.6→42.3, 44.7→44.9, 46.1→42.4, 47.4→41.8, (21808 see summary) — drop in 5/8,
mean rel 0.04; PAM5 dropped 20% in every fly (the lesson landed, weakly). cVA at R and DA1_vPN followed MDN (nothing moved). MUTE
identical 8/8, no leak 8/8, no ignition. Verdict FAIL, recorded as such. results/gate3/gate3_summary.json.
WHY (found after, stated plainly): this protocol said "learning = test 9 exactly ... lr 0.06". That was a transcription error by
Claude: the frozen test-9 script (versions/comm-loop-1/run_test9.ps1) sets COMM_LR=0.2, and test 9/10/10b all ran at lr 0.2 (their
JSONs record "lr": 0.2). Gate 3 therefore trained the sender at one-third of the frozen rate, which is why PAM5 fell 20% instead of
45% and MDN barely moved. Gate 3 stays a FAIL on its own declaration. Gate 3b below is the same test with the frozen learning rate,
on new seeds, declared before running.

# GATE 3b — Gate 3 with the FROZEN learning rate (lr 0.2, as test 9/10/10b), predeclared 13:02 EDT
Everything identical to Gate 3 above except lr = 0.2. Seeds: 22101, 22202, 22303, 22404, 22505, 22606, 22707, 22808 (never used).
PASS lines unchanged (S1, W1, R1, R2, R3, R4, G). If it fails, it fails.

## GATE 3b RESULT (13:15 EDT, 8 flies, 628 s) — FAIL on two lines; every fly in the predicted direction.
With the frozen lr 0.2 the lesson landed as in test 9 (PAM5 −45…−48% in 8/8). MDN(A) fell in 8/8 (rel 0.10–0.37); cVA at R rose in 8/8
(0.28–0.38 → 0.37–0.42); DA1_vPN at R rose in 8/8 (98–124 → 120–131 Hz); odour C flat in 8/8 (|rel| < 0.01); MUTE identical 8/8; no
leak; no ignition. Failed lines: S1b "rel_MDN(A) > rel_MDN(C)" 5/8 — C's MDN is 0.5–2.5 Hz, so its ratio is noise (a badly designed
line, the same weakness noted in test 10's C band); R1b "paired z ≥ 2 in ≥ 6/8" 5/8 (z = 3.2, 1.2, 3.4, 1.6, 3.0, 0.5, 2.2, 2.6) —
under-powered at 24 pairs for the two flies whose MDN started low (35 Hz). Verdict FAIL, recorded. results/gate3b/gate3_summary.json.

# GATE 3c — properly powered, C line in absolute Hz — predeclared 13:18 EDT, before running
Same stack, same learning (lr 0.2), same world physics, same receiver. Changes, declared now: 96 replayed realisations per odour
(was 24); S1b becomes "MDN(A) rel drop ≥ 0.10 in ≥ 7/8 AND |MDN(C) post − pre| ≤ 1.0 Hz in ≥ 7/8" (absolute, because C's MDN sits
at 0.5–2.5 Hz); R1b unchanged in form (paired z ≥ 2 in ≥ 6/8, now over 96 pairs). All other lines unchanged.
Seeds: 23101, 23202, 23303, 23404, 23505, 23606, 23707, 23808. If it passes, GATE 3d = the same protocol on 24101…24808 must also pass
before anything is called established. Gate 3 and 3b stay on the record as FAIL.

## GATE 3c RESULT (13:37 EDT, 8 flies, 1,945 s) — FAIL by the letter (R3 7/8); every scientific line 8/8.
S1: MDN(A) −26…−35% in 8/8, MDN(C) within 1 Hz in 8/8. W1: cVA at R up in 8/8 (0.28–0.32 → 0.36–0.42). R1: DA1_vPN up in 8/8
(99–110 → 122–132 Hz), paired z = 5.9, 4.8, 5.5, 4.3, 6.2, 4.6, 4.9, 4.6 (all ≥ 2). R2: C flat, |rel| ≤ 0.004 in 8/8. R4 no leak 8/8.
G no ignition 8/8. Reply SENDER-CAME-CLOSER 8/8. R3 (MUTE spike vectors bit-identical pre vs post): 7/8 — fly 23505, odour C, one
entry differs. Diagnosed (results/gate3c/determinism.txt): the GPU sparse matmul is not bit-deterministic at 96 columns — the same
block on identical input differs by 0–1 spike between repeats, with every per-trial readout identical. A harness limit, not a leak.
Verdict FAIL as declared; stays on the record.

# GATE 3d — reproduction of 3c with R3 stated at the kernel's measured precision — predeclared 13:42 EDT
Identical to Gate 3c in every CHOSEN value and every line except R3, which becomes: "MUTE: per-trial vPN, lPN, pC1 and brain_hz
identical pre vs post in 8/8 flies, AND the full spike-count vectors differ by ≤ 2 spikes in total per block" (the measured
non-determinism is 0–1 spike per block). Seeds: 24101, 24202, 24303, 24404, 24505, 24606, 24707, 24808.
GATE 3e — if 3d passes, the same protocol again on 25101…25808 must also pass (item 9: reproduce from a clean start).

## GATE 3d RESULT (14:13 EDT, 8 flies, 1,950 s) — FAIL by the letter (R3 7/8) again; every scientific line 8/8 again.
S1 8/8 (MDN(A) −18…−37%, C within 1 Hz), W1 8/8, R1 8/8 (vPN 105–113 → 122–133 Hz; z = 4.4, 4.3, 3.2, 5.6, 4.0, 5.4, 4.1, 4.8),
R2 8/8 (|rel| ≤ 0.006), R4 8/8, G 8/8, reply SENDER-CAME-CLOSER 8/8. R3: fly 24303, odour A, ONE trial of 96 diverged — a single
borderline spike flipped early in that trial and the rest of the trial cascaded (13,904 spike-count differences, vPN 76.4 vs 77.8 Hz
in that one trial; 95/96 trials bit-identical). Re-running the same block three times at batch 8 and at batch 96 gave identical
results every time — the flip is rare (about one trial in ~800 across 3c+3d) and not tied to batch size. The "≤ 2 spikes" tolerance
was naive: the network is chaotic within a trial, so one flip is never two spikes. Verdict FAIL as declared; stays on the record.

# GATE 3e — reproduction with R3 stated at the kernel's measured behaviour — predeclared 14:20 EDT
Identical to 3c/3d in every CHOSEN value and every line except R3: "MUTE: in each of the 4 blocks (A/C × pre/post-vs-pre) at least
94 of 96 trials bit-identical, AND block-mean vPN and lPN differ by < 1% pre vs post". Seeds: 25101, 25202, 25303, 25404, 25505,
25606, 25707, 25808. This is the third statement of R3; the first two stay on the record as fails. If 3e passes it is ONE pass of this
protocol; a further clean reproduction (3f, 26101…26808) is required before "established".

## GATE 3e RESULT (14:50 EDT, 8 flies, 1,953 s) — PASS on every line, 8/8 each.
MDN(A) −18…−33% (43–49 → 30–35 Hz), C within 1 Hz; cVA at R 0.29–0.33 → 0.38–0.41; DA1_vPN 99–111 → 124–132 Hz, paired z = 4.6,
2.9, 4.2, 4.5, 4.9, 6.0, 4.9, 3.6; C |rel| ≤ 0.006; MUTE ≥ 94/96 identical and means within 1% in every block; no leak; no ignition.
Reply SENDER-CAME-CLOSER 8/8. results/gate3e/gate3_summary.json.  → GATE 3f (26101…26808) now, same protocol, as declared.

## GATE 3f RESULT (15:24 EDT, 8 flies, 1,958 s) — PASS on every line, 8/8 each. Reproduced.
MDN(A) −17…−32%; C within 1 Hz; cVA at R 0.29–0.32 → 0.38–0.40; DA1_vPN 102–110 → 124–129 Hz, z = 3.8, 4.6, 4.4, 4.8, 3.8, 5.5,
4.7, 4.5; C |rel| ≤ 0.005; MUTE clean; no leak; no ignition; SENDER-CAME-CLOSER 8/8. results/gate3f/gate3_summary.json.
Grade 0b on fly-v10 re-run after (results/grade0b_v10calib_post_gate3.json): every verdict and read identical to post-9/post-10.
Frozen: versions/comm-loop-1/code/world (protocol + code), results/gate3{,b,c,d,e,f}.
