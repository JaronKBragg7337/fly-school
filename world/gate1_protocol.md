# Gate 1 — "ears being ears": a sender fly's song reaches a receiver fly through a world, and nothing else
Predeclared 2026-09-17 12:25 EDT by Claude Code, from Grok chat's four-gate plan (2026-09-17, read by Jaron on X) and Jaron's
organ-first rule. Written BEFORE any run. Nothing below changes after the numbers are seen. If it fails, it fails.

## The claim being tested
Two separate operating-system processes, each running its own copy of the frozen fly-v10 brain (graph_v9 + fly-v10 calib.json,
SHA-256 in versions/comm-loop-1/MANIFEST.md). They share ONE thing: a world buffer on disk (a song on/off schedule). No weights,
no spike trains, no shared Python objects.
- Sender S: its song command neurons pIP10 (2 cells) are driven; its wing motor neurons are recorded.
- World: turns S's wing motor spikes into a sound schedule (the only bridge).
- Receiver R: its ear (JO-A, 50 cells) is driven by the schedule; its auditory second-order cells (AMMC, 208 cells) are read.
No learning anywhere. No mushroom body in the score. No decoder.

## CHOSEN (declared once; justified from the connectome and literature, not from the exam)
- Stack: fly-v10 (frozen). Kernel: gpu/flysim_gpu.py (verified against the CPU kernel in results/gpu_verify.json before use) or the
  CPU kernel with identical maths — device is recorded in the output.
- Trial: 50 ms pre, 300 ms drive, 50 ms tail (Grade 0b timing) = 2000 steps of 0.2 ms.
- Sender drive: pIP10 at 100 Hz Poisson (the Grade 0b "strong sensory" rate; pIP10 is the descending song command neuron,
  von Philipsborn 2011; its main posts in this graph are dPR1, TN1a, vPR9 — the song pattern generators, Lillvis 2024).
- Sender readout: pulse-song wing motor neurons hg1, ps1, i1, i2 (Lillvis 2024) — 8 cells; all 32 wing MNs reported too.
- World rule: song_on(step) = 1 if any pulse-song wing MN spike fell in the preceding 5 ms (25 steps), else 0. Binary. No fitting.
- Receiver drive: JO-A at 120 Hz Poisson while song_on == 1 (the Grade 0b "sound" rate, row 5), 0 Hz otherwise.
- Receiver readout: mean rate of all AMMC-typed cells (208); DNp01 (giant fibre, 2 cells) reported too.
- Seeds: 20101, 20202, 20303, 20404, 20505, 20606, 20707, 20808 (never used before). Receiver uses seed + 1.

## The four conditions per seed
1. LIVE      — S driven; world computes the schedule from S's wing MNs; R hears the schedule.
2. MUTE      — S driven; world forced to all-zero; R hears nothing.                            (lesion of the channel)
3. SILENT    — S not driven; world computes the schedule from S's (silent) wing MNs; R hears whatever comes out.
4. SCRAMBLE  — S not driven; world replays LIVE's schedule; R hears the song with no singer.  (proves R listens to the world)

## PASS (all of these)
- S1  sender sings: pulse-song wing MN rate under pIP10 drive >= max(1 Hz, 2 x its SILENT rate) in >= 7 of 8 seeds.
- S2  no ignition: sender and receiver whole-brain rate < 10 Hz/cell in every condition, every seed.
- R1  ear works: AMMC rate LIVE > MUTE in >= 7 of 8 seeds, and LIVE >= max(1 Hz, 2 x MUTE) in >= 6 of 8.
- R2  the lesion kills it: MUTE receiver spike counts are IDENTICAL to a receiver given an all-zero world (same seed) — the receiver
      has no other way to know the sender existed. (Architecture check; if this fails, something leaks.)
- R3  the world is the bridge: SCRAMBLE receiver spike counts are IDENTICAL to LIVE receiver counts (same seed, same schedule) —
      the sender's brain never touches the receiver except through the schedule.
- R4  silence is silence: SILENT AMMC rate <= 1.1 x MUTE AMMC rate in >= 7 of 8 (a sender that is not singing should not be heard).

## FAIL readings (kept apart, as Grok asked)
- S1 fails: "the LIF male cannot sing under this stack" — stop; report; do not add a hidden song generator.
- S1 passes, R1 fails: "the ear does not carry song to AMMC under this gain" — an organ finding for Grade 0b.
- R2 or R3 fails: the harness leaks; fix the harness, re-declare, re-run with new seeds.

## Outputs
world/results/gate1_<seed>_<condition>.json per run; world/results/gate1_summary.json with every criterion and PASS/FAIL.
