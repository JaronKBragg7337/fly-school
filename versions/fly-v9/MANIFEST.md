# fly-v9 — fly-v7 with the antennal-lobe lLN1/lLN2 local neurons made inhibitory (2026-09-17 ~10:00 EDT, Claude Code)
Why: one glomerulus's ORNs activated 154-164 PN types (nearly the whole antennal lobe) with the same top list for every odour,
so no odour code could reach the Kenyon cells. The non-ORN excitation onto uniglomerular PNs came almost entirely from lLN1_bc,
lLN2T_*, lLN2X* — antennal-lobe local neurons the automated transmitter table marks acetylcholine. In the fly the lLN1/lLN2
panglomerular local neurons are GABAergic (lateral inhibition; Chou et al. 2010; only a small eLN population is cholinergic).
Change: every excitatory output edge of types matching ^lLN (171 cells, 15719 edges) has its sign flipped, magnitude kept.
Everything else = fly-v7 (graph_v7_km24_am1.0.npz + calib.json). graph_v9.npz sha256(16): 8eceb2264cff5a38
Note: this is a literature-based correction of a transmitter prediction, not a fitted parameter. To be checked by the reviewers.
