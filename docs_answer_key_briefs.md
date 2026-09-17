# Two briefs for tonight (2026-09-17 ~02:20 EDT). Paste each into a fresh chat. Bring the answers back to Claude.

---

## Brief for ChatGPT chat — the answer key for Grade 0 (organ check), from the fly literature

I am running a simulated male fruit-fly brain (MaleCNS v1.0 connectome, 165,122 neurons, leaky integrate-and-fire, dt 0.2 ms;
runtime after Shiu et al. 2024). Before any learning experiment I am checking that each sense organ drives the reflex the
literature says it should — "eyes being eyes, feet being feet." I need the ANSWER KEY, with citations, in a table. For each
row give: the input cell types (as named in MaleCNS / FlyWire / MANC if you know them), the expected downstream neuron(s)
that should fire, the paper(s) that established it, and what a clean positive looks like (which cell, roughly how strong,
how fast). Rows:

1. Sugar taste (tarsal and labellar gustatory receptor neurons) → proboscis extension motor neuron(s). (Shiu et al. 2024
   Nature reproduced this from the connectome; which MN did they read — MN9? — and which GRN types did they drive?)
2. Bitter taste → suppression of the same motor neuron.
3. Hind-leg touch (T3 leg bristle mechanosensory neurons) → MDN (moonwalker descending neuron, backward walking). Bidaye 2014.
4. Front-leg / head touch → grooming descending neurons (DNg07, DNg08, DNg12?). Seeds 2014, Hampel 2015, Guo 2022.
5. Looming (LPLC2, LC4 visual projection neurons) → giant fibre (DNp01) and non-GF takeoff DNs (DNp02, DNp04). von Reyn 2014,
   Ache 2019, Klapoetke 2017.
6. Wind on the antennae (Johnston's organ JO-C / JO-E) → MDN or other backing/stopping DNs. Which JO subgroups, which DN?
7. Courtship song (JO-A, ~pulse song) → which auditory / courtship neurons (aPN1? pIP10? DNp13?) — what is the real expected
   downstream of JO-A?
8. Odour (one glomerulus's ORNs, e.g. DL3 or DM1) → projection neuron → Kenyon cells, expected sparse code 5–15% (Turner 2008,
   Honegger 2011) → which MBON compartments, and which descending/steering neurons for odour tracking?

Also answer: (a) in Shiu et al. 2024, what stimulus rates (Hz) did they use on sensory neurons, and what firing rates did they
report downstream — so I can compare mine; (b) does the MaleCNS annotation use "claw_tpGRN" / "SNch10" for taste neurons —
what are those; (c) any reflex I've missed that is cheap to test on wiring alone.
Give numbers and names, not prose. Mark anything you are unsure of.

---

## Brief for Grok chat — what the public embodied-fly projects measured for the same reflexes

Three public projects run the whole fruit-fly connectome as a spiking network inside a MuJoCo body: Mineplix/fly-brain
(github.com/Mineplix/fly-brain, MIT; docs in docs/guide and docs/textbook), Ibtisam-Mohammad/Fly.exe (GPL-2, GeNN on GPU),
and FlyGM (arXiv 2602.17997). I need, in a table, for each project:

1. Their "sensory screen": which reflexes they tested on wiring alone (sugar→proboscis, bitter, looming→giant fibre, touch→
   backing/grooming, wind, odour→steering), and the numbers they report (firing rates, pass/fail, thresholds).
2. How they prevent whole-brain ignition: global weight scale, fitted parameters (Mineplix says nine global parameters
   fitted over twenty generations — which nine, what values?), inhibition terms, noise, conductance-based vs current-based
   LIF, mV per synapse.
3. Their reported Kenyon-cell sparsity (Mineplix says 7.4% active) and how it was measured (which stimulus, which window).
4. What they say about walking from the raw nerve cord (Mineplix: "the fly cannot stand in that mode"), exact quote + where.
5. Anything on learning/plasticity — do any of them change weights at all?
6. Timing: their brain dt, body dt, coupling interval, wall-clock speed, hardware.
Quote exact file paths / doc pages / numbers. Mark anything you could not find. No summaries of what the projects "are" —
only what they measured.

---

Why both: Claude is running our Grade 0 (organ check) on fly-v1 and fly-v2 right now; ChatGPT's table is the answer key
we grade against, Grok's table is the neighbours' numbers we lay ours next to. Bring both back and Claude merges them into
the Grade 0 report card and the public page.
