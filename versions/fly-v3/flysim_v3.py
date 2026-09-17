"""fly-v3 = lulzx/fly-brain's anti-ignition recipe applied at the graph level (2026-09-17 03:05 EDT, Claude Code).
Graph (graph_v3_s*.npz): connections with fewer than 6 synapses dropped; sensory neurons receive no central input; global
scale s (CHOSEN by Grade 0b). Cell: refractory 3.8 ms (theirs) instead of 2.2. Not ported: conductance-based synapses,
size-scaled PSPs, per-cell KC threshold (+10.9 mV) — the KC gain in calibration pn05_apl10_kc03 stands in. Their nine fitted
globals are in FLY-BRAIN.md 'GRADE 0b'. Use: FlyBrainV3(graph_path)."""
import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
from flysim import FlyBrain, Params

class ParamsV3(Params):
    refractory = 3.8

class FlyBrainV3(FlyBrain):
    def __init__(self, graph_path):
        super().__init__(graph_path, p=ParamsV3())
