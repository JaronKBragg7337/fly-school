# Fly School

A simulated male fruit-fly brain (MaleCNS v1.0, 165,122 neurons run as leaky integrate-and-fire cells) taught Morse
timing with its own learning rule, on school hours, with exams. Live: **https://www.heartbeatobservatory.com/school/**

This repository is written so that a person — or an AI handed this page — can rebuild the whole thing and get the same
numbers. Every constant is marked **CHOSEN** (ours, swappable) or **MEASURED** (from the data or the run). If something
below is not enough to rebuild from, that is a bug in this README; open an issue.

## Credits, in the order the work happened

- **The brain:** Male CNS connectome v1.0, HHMI Janelia FlyEM + Cambridge Connectomics + Google Research, CC-BY 4.0.
  Dataset released 2026-06-08; paper 2026-09-03. Files from `storage.googleapis.com/flyem-male-cns/v1.0/`.
- **The runtime** (`runtime/`): cloned from **fruitflydev/flycoinrh** (MIT), which follows Shiu et al. 2024 (Nature)
  for the LIF model and sign convention, and includes the mushroom-body learning circuit (`mushroom.py`,
  `calibration.py`, `mb_sides.py`). We changed nothing in these five files for the school; they are copied here so
  the exact code that ran is in one place. LICENSE and NOTICE are theirs.
- **The exam** (`exams/morse_grade1.py`): written by ChatGPT (Fly-Lab-2, 2026-09-14) at Jaron Bragg's design;
  env overrides and the axon-gate/leg-drive options added by Claude Code (2026-09-16/17).
- **The axo-axonic fork** (`axon/`): Claude Code, 2026-09-17, from Ceballos et al., iScience 2026-04-22 (PMC13126034).
- **The school** (`school/school.py`): Claude Code, 2026-09-17, at Jaron's rules (persistent fly, school hours,
  shifts, life source).
- **The site pages:** github.com/JaronKBragg7337/heartbeat-observatory, `/school/` and `/live-systems/flies/school/`.
- **Jaron Bragg** — the design, every rule, every "that's not honest, fix it." Codex, Grok and a local Qwen worked on
  neighbouring parts of the same project; each is credited in the project ledger where it worked.

## 1. Rebuild the brain (once, ~10 minutes, ~1.1 GB download)

```
pip install -r runtime/requirements.txt
# fetch the three flat-connectome tables (CC-BY, no account) into runtime/data/:
#   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights.feather
#   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters.feather
#   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations.feather
python runtime/build_graph.py      # -> runtime/build/graph.npz   (165,122 neurons, 10,228,000 signed edges)
python runtime/mb_sides.py         # -> runtime/build/mb_sides.json (which dopamine cluster innervates each MBON)
```

The LIF cell (`runtime/flysim.py`, MEASURED in Shiu et al., CHOSEN here by inheritance): v_rest −52 mV, v_thresh −45,
v_reset −52, τ_m 20 ms, refractory 2.2 ms, dt 0.2 ms, 0.275 mV per synaptic contact, ACh excitatory, GABA/Glu
inhibitory, monoamines zero fast weight. Calibration `pn05_apl10_kc03` (CHOSEN by the runtime's author; see
`calibration.py`) scales PN/APL/KC types so the mushroom body is sparse.

## 2. The learning rule (the only weights that change)

`runtime/mushroom.py`: Kenyon-cell → MBON synapses only (MEASURED site, 44,042 synapses on this graph). Direction:
**depression** under dopamine (Hige 2015, Cohn 2015), never potentiation. Which MBONs get reward vs punishment
dopamine: counted from PAM vs PPL1 synapses onto each MBON type (`mb_sides.json`). CHOSEN constants: learning rate
0.06, floor 0.25, eligibility trace ×0.55 per `observe()` call (~6 observations stay eligible), forgetting half-life
6 h wall-clock (Tully & Quinn 1985, "mostly gone within a day").

## 3. Grade 1 — the Morse exam (`exams/morse_grade1.py`)

CHOSEN protocol. Input: the 50 neurons whose type name starts with `JO-A` (annotated auditory Johnston's organ cells;
the real organ has ~480–720 — we drive the 50 that are typed). Carrier 120 Hz Poisson while sound is on. Dot 20 ms,
dash 60 ms, gap 40 ms, pre 20 ms, tail 40 ms. Output: the two `DNa01` descending neurons, spikes counted in 10 ms
bins; a bin with ≥1 spike is "on"; runs of on-bins <40 ms = dot, ≥40 ms = dash. Membrane reset every trial; MB
weights persist. **Teach:** 24 epochs × (`.`, `-`); exact decode → dopamine +1 (reward side), else −1 (punish side).
**Exam:** `.-` and `-.` — never taught as pairs — six cold trials each, before and after, no dopamine. Seeds are
SHA-256 of (base seed, phase, pattern, rep), so every trial is reproducible.

```
set MORSE_SEEDS=73,7337          # or any list
python exams/morse_grade1.py     # -> results/morse_grade1.json (every trial, every DNa01 bin)
```
Options: `MORSE_GAP_MS`, `MORSE_DOT_MS`, `MORSE_DASH_MS`; `MORSE_BRAIN=axon` (fork below); `MORSE_LEG_HZ=5` (tonic
drive on 3,899 leg sensory neurons). Paths at the top of the file point at one machine; set `LAB` and `RUNTIME`.

### Report card (MEASURED; the JSON for every row is in `results/`)

| row | fly | change | `.-` pre→post | `-.` pre→post | reading |
|---|---|---|---|---|---|
| 1 | fly-v1 (seeds 73, 7337) | — | 0/12 → 2/12 | 0 → 0 | one seed produced it; the other did not |
| 2 | fly-v1 (101, 202, 303) | — | 1/18 → 1/18 | 0 → 0 | not reproduced; `.-` occurs untrained |
| 3 | fly-v1 (10 seeds) | — | 0/60 → 4/60 | 0 → 0 | pooled 15 seeds: 1/90 → 7/90 (~1 in 13), p≈0.03 two-sided |
| 4 | fly-v1, gap 100 ms | echo test | 4/30 → 3/30 | 0 → 0 | `.-` appears **untrained** at the wide gap: a DNa01 output bias, not a lesson |
| 5 | fly-v1-axon | INVALID | — | — | the gate was loaded but never wired into the exam kernel (identical to v1 per seed was the tell) |
| 6 | fly-v1 + 5 Hz leg drive | control | 0/30 → 0/30 | 0 → 0 | flat leg noise erases the exam |
| 7 | fly-v1-axon + 5 Hz legs | gate wired in | 2/30 → 1/30 | 0 → 0 | the axo-axonic gate is real and does not produce `-.` |

Verdict so far: **not a pass.** `-.` has never appeared in 252 cold trials. The asymmetry is upstream of DNa01's axon:
its brain-side inputs during a dash, and/or the single-compartment LIF cell (which cannot ramp down).

## 4. The axo-axonic fork (`axon/`)

Ceballos et al. 2026 charted inputs onto the *axons* of descending neurons that "veto, amplify, or synchronize" spikes.
`dn_axon_inputs.py` streams the 124-M-row per-synapse table (`syn-partners-traced.feather`, 2.97 GB) and classifies
each synapse onto a DN as **axonal** if the postsynaptic ROI is in the VNC (CHOSEN token list in the file) — MEASURED:
4.59 M DN inputs, 290,602 axonal (6.3%); DNa01 1,135 (4.4%). Sanity check that reproduces the paper's validated example:
DNp01's top axonal presynaptic type is AN08B098 (399 synapses). `flysim_axon.py`: axonal edges do not touch the target's
voltage; they set an output **gate** on the target, `gate = clip(1 + 0.05·mV, 0, 3)`, τ 5 ms (CHOSEN). With no axonal
flags the class is spike-identical to the base runtime (verified).

## 5. The school (`school/school.py`) — one fly, persistent, on hours

Run it and leave it running (on Windows we register it as a scheduled task; any supervisor works):

```
set SCHOOL_FLY=school-1             # pupil name; its memory file is mb_<name>.npz and is never deleted
set SCHOOL_HOURS=9-12,13-16         # CHOSEN, local time; lessons only in these hours
set SCHOOL_NIGHT=22-7               # CHOSEN; sleeping (one silent trial / 5 min)
set SCHOOL_INHERIT=                 # optional: start from another pupil's memory file (same graph only)
python school/school.py
```
A **lesson** = teach 12 × (`.`, `-`) with dopamine ±1 → exam (`.-`, `-.`) × 6 cold → rest (2 silent trials, so "what the
fly does on its own" is measured — in this runtime: nothing, 0 spikes). Lesson 0 = a baseline exam. The **morning
exam** is the first exam of each school day, before any teaching: with a 6 h half-life and ~17 h without lessons,
~14% of the previous day's depression survives, so it measures what stuck (spaced vs massed training). **Free time:**
a silent trial per minute and a meal every 3 h — a clock event (fed_at, hunger), deliberately *not* dopamine, because
sugar with nothing recent in the eligibility trace changes no weight and counting it would lie. **Year:** if the
process finds the last sign of life ≥3 h old (the machine was off), it archives the report card and starts year n+1;
the memory carries with the forgetting applied. Publishing: one `fly_live` row per pupil (overwritten every trial) and
one `fly_school` row per trial (`supabase/2026-09-17-fly_school.sql`); local mirrors `report-card.json`, `*.state.json`.

**First MEASURED thing the persistent fly showed (2026-09-17, first hour):** under continuous ±1 dopamine the memory
saturates in one lesson — 41,096 of 44,042 KC→MBON synapses depressed after 24 dopamine events, mean gain 1.0 → 0.54,
sitting at ~0.28 (floor 0.25) an hour later. Everything depressed = no contrast. That is the CHOSEN rule (lr 0.06,
floor 0.25, −1 on every miss), not the connectome. Planned competing mechanisms, each as a new pupil beside the control:
smaller lr / higher floor; dopamine only on *change*; longer rest. `VERSIONS.md` is the freeze-and-fork ledger.

## 6. What is chosen vs measured — the short list

CHOSEN: LIF constants (inherited), calibration, Morse timings and the 40 ms split, the reward mapping, learning rate /
floor / trace / half-life, school hours, meal interval, the gate formula, the VNC-ROI rule for "axonal".
MEASURED: the wiring, the sign of every synapse, which neurons are JO-A / DNa01 / KC / MBON, the PAM/PPL1 side of each
MBON, every spike, every decode, every score, the axonal-input counts.

## 7. Rebuild check

You have rebuilt it when `python exams/morse_grade1.py` with `MORSE_SEEDS=73` reproduces row 1's seed-73 decodes in
`results/morse_grade1.json` (`.-` pre-test `['..', '.', '.', '.', '.', '.']`, post-test `['.-', '.', '.', '.', '.', '.-']`). Float summation order can differ
between BLAS builds; if it does, the pooled counts in row 3 are the check instead.

---
Code in `school/`, `exams/`, `axon/`: MIT (Jaron Bragg, 2026). `runtime/`: MIT, © fruitflydev. Connectome: CC-BY 4.0,
© HHMI Janelia FlyEM et al. — keep the attribution; it is the whole reason any of this is real.

## 8. What one night on the persistent fly found (2026-09-17, rows 8–14; every file in `results/`, scripts in `exams/`)

1. **The graded neuron could not speak.** `exams/dn_readout_scan.py`: DNa01 says a dash 0/30 times untrained. 117 DN
   types / 275 cells can say all four rhythms untrained. (`results/dn_readout_scan.json`)
2. **On readouts that can speak, training moves nothing.** Four readouts × 5 seeds: `.-` 29→26/120, `-.` 7→3/120.
   (`results/morse_grade1_readout_*.json`)
3. **The mushroom body does not reach a production exam.** `exams/mb_reach_test.py`: cutting MB output entirely leaves
   every readout's accuracy unchanged (36 trials × 3 conditions). (`results/mb_reach_test.json`)
4. **Its own exam — valence — fails too,** because dot and dash are the same Kenyon cells (Jaccard 0.97; ~65% of KCs fire
   for any sound). `exams/morse_valence.py`. (`results/morse_valence.json`)
5. **Root cause: whole-brain ignition** under the stock gain (25–35 Hz/cell for any sustained input).
6. **Fix, measured:** `exams/gain_sweep.py` on 3 seeds — every weight × 0.35 → 2–2.9 Hz/cell, KC 7–11% active (the real
   fly range), descending neurons alive and duration-sensitive. **This is `fly-v2`** (`versions/fly-v2/MANIFEST.md`; build
   it by scaling `graph.npz["data"]` by 0.35). (`results/gain_sweep*.json`)
7. **The body.** `body/body_loop.py` (flybody + MuJoCo 3.13, own venv): 328 leg motor neurons → 59 actuators by muscle
   name; 3,915 leg sensory cells ← touch / joint speed; same 0.2 ms step. On v1 the fly seizes at any touch; on v2 it
   stands (4–6 Hz/cell, five or six legs down, ball still). (`results/body_loop_*.json`)

Rebuild check for this section: `SWEEP=0.35 SWEEP_SEED=73 python exams/gain_sweep.py` → `hz_per_cell` ≈ 2.0, `kc_frac_active` ≈ 0.07.

## 9. Walking — borrowed, declared (2026-09-17 02:10)

No connectome-only model walks; every public embodied fly brain (FlyGM, Fly.exe, Mineplix/fly-brain, NeuroMechFly v2) puts a
stepping generator between descending-neuron rates and the legs, and says so. We do the same: `body/gait.py` is a port of
**Mineplix/fly-brain's** generator (lulzx, MIT) — `gait.mineplix.json`, two-harmonic joint curves fitted to 100 FlySuite
real-fly walks, tripod, 10 Hz — commanded by our brain's own DN rates (`BODY_MODE=descending`). Measured on fly-v2:
silent, the fly steps intermittently at the walking threshold; a 120 Hz tone makes MDN fire and it backs up and turns
(`results/body_loop_v2_gait_s0.json`, `_s120.json`). Mineplix also reports 7.4% active Kenyon cells in their fitted regime;
our 0.35× gives 7.3%. Their raw-muscle mode "cannot stand"; ours stands on v2 (2 s, on a ball — small, checkable).

## 10. Grade 0 — does each organ do its job? (2026-09-17 02:45)

Before any lesson: sense in, the literature's named output read, no learning (`exams/grade0_organs.py`). Pass = output
≥ max(1 Hz, 2× baseline) and brain < 10 Hz/cell. **fly-v1: 7 of 7 organs ignite the whole brain. fly-v2: 7 of 7 pass** —
taste → proboscis MN9 (20 Hz); loom → giant fibre (253 Hz); wind → MDN (11.5 Hz); song → pIP10/DNp13; odour → 6.3% Kenyon
cells + steering DNa02; hind-leg touch → DNa01/MDN; front-leg touch → grooming DNs (weak, 1–2 Hz). `results/grade0_v1.json`,
`grade0_v2.json`. Identity lesson on v2 (`exams/identity_lesson.py`): two odours are distinct (KC overlap 0.10–0.21) but
the population valence readout did not learn a preference in 12 pairings (0/5 seeds) — next: compartment-specific readout.
