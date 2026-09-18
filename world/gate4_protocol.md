# Gate 4 — the round trip: A learns, B replies, A's next move depends on the reply
Predeclared 2026-09-17 19:35 EDT by Claude Code, from Jaron's bar ("something can be sent back and go back-and-forth") and ChatGPT chat's
tightening (no scripted rule in the return path; LIVE vs YOKED vs MUTE; reproduce). Written BEFORE any test-seed run.
Pilots (world/gate4_pilot.py, seeds 60101–60606, notebook only): (1) cVA→MDN in the receiver saturates above ~4 Hz of ORN_DA1 drive,
and while A smells the odour its own cVA channel is silent at faint cVA — so the pure cVA loop had no measurable return path;
(2) cVA → B's wing motor neurons is clean and graded (0→12→25→40→54→66 Hz over 0–16 Hz ORN_DA1, 32 cells), and antennal deflection
(JO-C/E/F) → A's MDN is graded and works with the odour on (46→41→35→17→14 Hz over 0–40 Hz) — Grade 0b row 9 measured that link;
(3) with both ends in their graded bands the round trip showed at z −2.4 and −0.8 (48 pairs, predicted sign) — small, consistent.

## The claim being tested
A (trained: odour A + reward) and B (untrained, frozen weights) are two separate brain objects on the frozen fly-v10 stack. They share
nothing but the world: a distance d, A's cuticular cVA at B's nose, and the air B's wings push at A's antenna. B never receives odour
spikes. Bout by bout, both move by their own MDN. Pass = A's move in bout 2 differs between a B that is actually replying to A (LIVE)
and a B whose wing output is replayed from an untrained pairing (YOKED), in the predeclared direction, and the effect is absent when A
was given the never-paired odour C instead.

## CHOSEN (declared once; operating points from the pilots, not from the test)
- Stack fly-v10; GPU kernel (bit-identical to CPU; ~1 flip per 800 trials, see gate3). Bouts of 1,000 ms drive (20 ms pre, 40 ms tail).
- Learning = frozen test-9 rule on A only: odour A (ORN_DA2 @40 Hz) + reward, 12 epochs, lr 0.2. B's weights are never touched
  (asserted equal to the graph after A's training).
- World (declared 1-D physics): d starts at 30 mm; after each bout d += K·(MDN_A + MDN_B)·BOUT with K = 0.25 mm/s/Hz, BOUT = 2 s
  (both back away). c(d) = 1/(1 + (d/60 mm)²). B's ORN_DA1 rate = 5 Hz · c(d) (A's cuticular cVA; passive, no command).
  A's JO-C/E/F rate = 0.6 · wingMN_B(previous bout, Hz) · c(d) (air pushed by B's wings; reaches A one bout later).
  The world reads MDN and wing-motor-neuron spikes only. Nothing else crosses.
- A smells its odour (A, or C in the control loop) in every bout; B never smells an odour.
- 3 bouts. 96 paired realisations per loop (realisation seed sha256("gate4:<seed>:<fly>:<phase>:<bout>:<rep>"), identical across
  conditions so LIVE/YOKED/MUTE are paired).
- Conditions, post-training, odour A: LIVE (B's wings as produced); YOKED (B's wing rate fed to A's antenna replayed, bout by bout,
  from the matched realisation of the PRE-training LIVE loop — statistically the same reply, not caused by the trained A);
  MUTE (A's antennal input frozen at its bout-1 value, i.e. zero — B's reply never reaches A).
- Control loop: the same fly given odour C (ORN_VM5d @40 Hz, never paired) — PRE LIVE, POST LIVE, POST YOKED.
- Readout: A's MDN (4 cells) mean rate in bout 2 (bout 3 reported). Seeds: 27101, 27202, 27303, 27404, 27505, 27606, 27707, 27808.

## PASS (all)
- L1  learning enters the world: d after bout 1, odour A, post < pre (mean over 96) in ≥ 7 of 8 flies.
- L2  A hears B: A's MDN bout 2, LIVE < MUTE, paired z ≤ −3 in ≥ 7 of 8 flies.
- L3  the reply is B's reply to A (the round trip): pooled over 8 flies × 96 pairs, Δ = MDN_A(LIVE) − MDN_A(YOKED) at bout 2:
      pooled paired z ≤ −3 (predeclared sign: negative — a trained A ends closer, B smells more, B's wings beat harder, more air at A,
      A backs away less) AND per-fly mean Δ < 0 in ≥ 6 of 8.
- L4  no learning, no round trip: the odour-C loop, same statistic, pooled |z| < 2.
- L5  no leak: B's ORN_DA2/ORN_VM5d spike counts zero in every trial; B's weights equal the frozen graph after A's training; no ignition
      (< 10 Hz/cell) in any trial.
- G   Grade 0b on fly-v10 after: core properties identical.
Reproduction (Gate 4b): the same protocol on 28101…28808 must also pass before anything is called established.
Decoded, for humans only: LIVE vs YOKED at bout 2 → "A HEARD B'S REPLY" when L3 holds; the flies themselves decode nothing.

## FAIL readings
L1 fails → learning did not reach the world (contradicts Gate 3; report). L2 fails → A cannot hear wing-air on this stack (contradicts
the pilot; report). L3 fails with L1, L2 passing → the reply exists but does not depend on A's state at this world gain: the measured
gap for the modified-brain fork. L4 fails → the statistic is picking up something other than learning; report, do not reinterpret.

## GATE 4 RESULT (2026-09-18 00:57 EDT; 8 pairs, 96 pairs each, 5.5 h) — FAIL on L3. Everything else passed.
L1 8/8: learning enters the world (pair 11–14 mm closer after A learns). L2 8/8: A hears B — A's MDN 13–16 Hz with B's wing-air
arriving vs 31–38 Hz with it cut (z −8.9 to −11.8). L4: control odour pooled z +0.40 (flat). L5 8/8: zero odour spikes in B,
B's weights identical to the graph after A's training, no ignition.
L3 — does B's reply depend on what A learned? Pooled Δ (LIVE − YOKED, A's MDN, bout 2) = −0.65 Hz, pooled z = −1.34 (bar ≤ −3);
sign negative in 6/8 (meets 6/8). Per pair: −0.9, −0.5, +0.4, −2.0, −2.5, −0.2, −1.4, +1.7 Hz. Bout 3 no better.
Reading: the reply arrives and it is loud (a 20 Hz swing in A's walking neurons), but how loud depends on what A learned by only
~5% — the pilot's 2–3 Hz was an overestimate from n = 48. Verdict FAIL as declared. No reproduction. No re-declaration.
This is the measured gap, at a named link: A's learned state changes distance by ~12 mm; through this world's physics that is
~0.4 Hz at B's nose; B's wings track it, but A's MDN response to the air saturates near its floor (13–16 Hz) whatever B does.
The native fly carries a message one way strongly (Gate 3) and hears a reply strongly (L2) — but the reply is the same reply
whether or not A learned. For the round trip to carry the lesson, either the world's gain must be far higher than physics allows
here, or the fly needs what it does not have: a state that persists between bouts and makes the reply mean something (Q-020).
