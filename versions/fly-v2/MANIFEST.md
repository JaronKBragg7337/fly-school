# fly-v2 — the gain regime (frozen 2026-09-17 02:20 EDT, Claude Code)

Forked from fly-v1. ONE change: every synaptic weight × 0.35 (graph_v2.npz = runtime/build/graph.npz with `data` scaled;
indices/indptr/bodies/sign/types/superclass identical). Everything else (LIF constants, calibration pn05_apl10_kc03,
mushroom.py rule, exams) unchanged.

WHY: under fly-v1 any sustained input ignites the whole brain (25-35 Hz/cell, 65-78% of Kenyon cells active for any
sound), which made rows 1-12 unanswerable (FLY-BRAIN.md "The night's chain"). Sweep (Fly-Lab-2/gain_sweep.py) on seeds
73, 7337, 101:
  scale 0.5 : 6.6-12.4 Hz/cell, KC 38-39%          (still ignites)
  scale 0.4 : 3.5-4.2 Hz/cell,  KC 12%             (edge)
  scale 0.35: 2.0-2.9 Hz/cell,  KC 7-11%, all-DN spikes dot/dash 1009/1702, 1207/2924, 1395/2639; MBON 50-81 / 75-181
  scale 0.3 : 1.6-1.8 Hz/cell,  KC 2-4%            (too quiet)
CHOSEN: 0.35 — KC in the measured fly range (5-10%, Turner 2008 / Honegger 2011), output alive and duration-sensitive.
KNOWN: dot vs dash do not become different KC identities at any scale (same tone) — the mushroom body codes which input,
not how long; the identity exam is the one to give it. Nothing about 0.35 is biology; it is the number that stops the seizure.

SHA-256 (first 16): graph_v2.npz 6477bbb1eedd15bb   source graph.npz 5a6e6fbfa5f2d384
Use: FlyBrain(r"C:/Users/lilli/Fly-Lab/versions/fly-v2/graph_v2.npz")  — MORSE_GRAPH / SCHOOL_GRAPH / BODY_GRAPH env below.
