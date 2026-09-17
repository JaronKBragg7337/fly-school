"""fly-v1-body v0: the MaleCNS brain on the flybody (MuJoCo) body, closed loop, no task. (Claude Code, 2026-09-17 01:10 EDT)

Brain -> body: the connectome's LEG MOTOR NEURONS (374 cells, annotated by muscle, leg T1-T3, side) drive flybody's leg
  actuators. A motor neuron spike adds to its muscle's activation; flexors pull one way, extensors the other. The mapping
  muscle -> actuator is anatomy (CHOSEN where flybody lumps joints); the sign convention of each actuator is CHOSEN and
  stated - if a leg bends the wrong way, that is the first thing to measure and flip.
Body -> brain: flybody's per-leg touch (tarsus contact) and joint velocities drive the connectome's LEG SENSORY NEURONS
  (3,915 cells by entry nerve ProLN/MesoLN/MetaLN = T1/T2/T3, side by soma side): tactile/bristle cells at TOUCH_HZ while the
  leg touches; chordotonal/hair-plate/campaniform cells at rate proportional to joint speed (CHOSEN coarse encoding).
Clock: brain dt 0.2 ms == MuJoCo physics dt 0.2 ms. Control every 2 ms (10 brain steps), as flybody does.
No task, no reward, no script: we run N seconds and MEASURE what the body does (touch pattern, height, ball velocity,
motor neuron spikes). Runtime: fly-v1 kernel (no gate). Writes results/body_loop.json and a spike/actuation trace.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain          # noqa: E402
import calibration as C              # noqa: E402
import pandas as pd                  # noqa: E402
from flybody.fly_envs import walk_on_ball  # noqa: E402

SECONDS = float(os.environ.get("BODY_SECONDS", 3.0))
TOUCH_HZ = float(os.environ.get("BODY_TOUCH_HZ", 40.0))       # CHOSEN: tactile cells while the tarsus touches
PROP_HZ_PER_RAD_S = float(os.environ.get("BODY_PROP_GAIN", 5.0))  # CHOSEN: proprioceptive Hz per rad/s of joint speed
MOTOR_GAIN = float(os.environ.get("BODY_MOTOR_GAIN", 0.05))    # CHOSEN: actuation per motor-neuron spike in the 2 ms window
MOTOR_DECAY = float(os.environ.get("BODY_MOTOR_DECAY", 0.8))   # CHOSEN: activation memory per control step (muscle is slow)
SEED = int(os.environ.get("BODY_SEED", 1))
MODE = os.environ.get("BODY_MODE", "connectome")          # 'connectome' = every leg muscle from its motor neurons (raw wiring);
                                                          # 'descending' = DN rates command a declared gait (ported from Mineplix, MIT)
SOUND_HZ = float(os.environ.get("BODY_SOUND_HZ", 0))      # optional: tone on the 50 JO-A ear cells for the whole run
SOUND_FROM_S = float(os.environ.get("BODY_SOUND_FROM_S", 0.5))

# muscle -> (actuator stem, sign). Anatomy: Tr = trochanter (coxa-trochanter joint; flybody's 'femur' hinge), Fe = femur
# reductor (rotates femur; 'femur_twist'), Ti = tibia, Ta = tarsus, ltm = long tendon (claw/tarsus grip), sternal
# rotators = coxa twist, pleural promotor/remotor = coxa swing, adductor/abductor = coxa abduct, tergotrochanter = jump.
MUSCLE_MAP = {
    "Tergopleural/Pleural promotor MN": ("coxa", +1), "Pleural remotor/abductor MN": ("coxa", -1),
    "Sternal anterior rotator MN": ("coxa_twist", +1), "Sternal posterior rotator MN": ("coxa_twist", -1),
    "Sternal adductor MN": ("coxa_abduct", -1),
    "Tr flexor MN": ("femur", +1), "Acc. tr flexor MN": ("femur", +1), "Tr extensor MN": ("femur", -1),
    "Sternotrochanter MN": ("femur", -1), "Tergotr. MN": ("femur", -1),
    "Fe reductor MN": ("femur_twist", -1),
    "Ti flexor MN": ("tibia", +1), "Acc. ti flexor MN": ("tibia", +1), "Ti extensor MN": ("tibia", -1),
    "Ta depressor MN": ("tarsus", +1), "Ta levator MN": ("tarsus", -1),
    "ltm MN": ("adhere_claw", +1), "ltm1-tibia MN": ("adhere_claw", +1), "ltm2-femur MN": ("adhere_claw", +1),
}
LEG_NERVE = {"ProLN": "T1", "MesoLN": "T2", "MetaLN": "T3"}


def main():
    t0 = time.perf_counter()
    fb = FlyBrain(Path(os.environ["BODY_GRAPH"]) if os.environ.get("BODY_GRAPH") else RUNTIME / "build" / "graph.npz")
    ann = pd.read_feather(RUNTIME / "data" / "body-annotations.feather")
    b2i = fb.body_to_i
    soma_side = np.array([""] * fb.n, dtype=object)
    for bid, sd in zip(ann["bodyId"].astype(int), ann["somaSide"].astype(str)):
        if bid in b2i and sd in ("L", "R"): soma_side[b2i[bid]] = sd
    jo = fb.where(type_re=r"^JO-A")
    env = walk_on_ball(); ts = env.reset()
    m = env.physics.model
    act_names = [m.actuator(i).name.replace("walker/", "") for i in range(m.nu)]
    act_index = {n: i for i, n in enumerate(act_names)}
    # ---- motor neurons -> actuators
    mn = ann[ann["superclass"].astype(str).str.contains("motor", case=False)]
    mn_idx, mn_act, mn_sign = [], [], []
    unmapped = {}
    for _, r in mn.iterrows():
        t = str(r["type"]); leg = str(r["somaNeuromere"]); side = {"L": "left", "R": "right"}.get(str(r["somaSide"]))
        if t not in MUSCLE_MAP or leg not in ("T1", "T2", "T3") or side is None or int(r["bodyId"]) not in b2i:
            unmapped[t] = unmapped.get(t, 0) + 1; continue
        stem, sign = MUSCLE_MAP[t]
        name = f"{stem}_{leg}_{side}"
        if name not in act_index:
            unmapped[t] = unmapped.get(t, 0) + 1; continue
        mn_idx.append(b2i[int(r["bodyId"])]); mn_act.append(act_index[name]); mn_sign.append(sign)
    mn_idx = np.array(mn_idx, dtype=np.int64); mn_act = np.array(mn_act); mn_sign = np.array(mn_sign, dtype=np.float32)
    # ---- leg sensory neurons per leg
    sn = ann[ann["superclass"].astype(str).str.contains("sensory", case=False)]
    sn = sn[sn["entryNerve"].astype(str).isin(LEG_NERVE)]
    legs = [f"{seg}_{side}" for seg in ("T1", "T2", "T3") for side in ("left", "right")]
    touch_cells = {l: [] for l in legs}; prop_cells = {l: [] for l in legs}
    for _, r in sn.iterrows():
        bid = int(r["bodyId"])
        if bid not in b2i: continue
        side = {"L": "left", "R": "right"}.get(str(r["somaSide"]) if str(r["somaSide"]) in ("L", "R") else str(r["rootSide"]))
        if side is None: continue
        leg = f"{LEG_NERVE[str(r['entryNerve'])]}_{side}"
        cls = str(r["class"]); sub = str(r["subclass"])
        if "proprio" in cls or sub in ("chordotonal organ", "hair plate", "campaniform sensilla"):
            prop_cells[leg].append(b2i[bid])
        else:
            touch_cells[leg].append(b2i[bid])
    touch_cells = {k: np.array(v, dtype=np.int64) for k, v in touch_cells.items()}
    prop_cells = {k: np.array(v, dtype=np.int64) for k, v in prop_cells.items()}
    # flybody touch sensor order: T1L, T1R, T2L, T2R, T3L, T3R (walker/touch, 6) - CHOSEN assumption, verified below by geometry
    touch_order = legs
    # joint velocity per leg: mean |qvel| over that leg's joints
    jnames = [m.joint(i).name.replace("walker/", "") for i in range(m.njnt)]
    leg_joints = {l: [i for i, n in enumerate(jnames) if n.endswith(f"_{l}")] for l in legs}
    print(f"motor neurons mapped {len(mn_idx)} (unmapped by type: {unmapped}); leg sensory touch "
          f"{ {k: len(v) for k, v in touch_cells.items()} } prop { {k: len(v) for k, v in prop_cells.items()} }", flush=True)

    gait = None
    if MODE == "descending":
        sys.path.insert(0, str(HERE)); from gait import Gait
        gait = Gait(fb, side=soma_side)
        print(f"gait: forward DNs {len(gait.fwd)}, backward {len(gait.back)}, turnL {len(gait.turnL)}, turnR {len(gait.turnR)}", flush=True)
    # ---- brain state
    p = fb.p; gains = C.gains_for(fb, C.CHOSEN); gpn = gains[fb.type_code].astype(np.float32)
    rng = np.random.default_rng(SEED)
    v = np.full(fb.n, p.v_rest, dtype=np.float32); refr = np.zeros(fb.n, dtype=np.int32)
    indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
    ctrl_dt = env.control_timestep(); steps_per_ctrl = int(round(ctrl_dt / (p.dt / 1000.0)))
    n_ctrl = int(SECONDS / ctrl_dt)
    activation = np.zeros(m.nu, dtype=np.float32)
    drive_rate = np.zeros(fb.n, dtype=np.float32)      # Hz per neuron from the body
    trace = []; mn_spikes_total = 0; total_spikes = 0
    for k in range(n_ctrl):
        # body -> brain rates for this control window
        obs = ts.observation
        touch = np.asarray(obs["walker/touch"]).ravel(); qvel = env.physics.data.qvel
        drive_rate[:] = 0.0
        for li, leg in enumerate(touch_order):
            if li < len(touch) and touch[li] > 0:
                drive_rate[touch_cells[leg]] = TOUCH_HZ
            js = leg_joints[leg]
            if js:
                dof = [m.jnt_dofadr[j] for j in js]
                speed = float(np.mean(np.abs(qvel[dof])))
                drive_rate[prop_cells[leg]] = min(200.0, PROP_HZ_PER_RAD_S * speed)
        if SOUND_HZ > 0 and k * ctrl_dt >= SOUND_FROM_S:
            drive_rate[jo] = SOUND_HZ
        prob = np.clip(drive_rate * p.dt / 1000.0, 0, 1); drive_idx = np.flatnonzero(prob > 0); drive_p = prob[drive_idx]
        # brain steps
        mn_count = np.zeros(m.nu, dtype=np.float32)
        win_count = np.zeros(fb.n, dtype=np.float32) if gait is not None else None
        for _ in range(steps_per_ctrl):
            v = p.v_rest + (v - p.v_rest) * fb.decay
            if len(drive_idx):
                hit = drive_idx[rng.random(len(drive_idx)) < drive_p]
                if len(hit): v[hit] = p.v_thresh + 1.0
            v[refr > 0] = p.v_reset
            fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
            if len(fired):
                total_spikes += len(fired)
                refr[fired] = fb.refr_steps; v[fired] = p.v_reset
                if win_count is not None: win_count[fired] += 1
                f_mn = np.isin(mn_idx, fired)
                if f_mn.any():
                    np.add.at(mn_count, mn_act[f_mn], mn_sign[f_mn]); mn_spikes_total += int(f_mn.sum())
                starts = indptr[fired]; cnt = indptr[fired + 1] - starts; tot = int(cnt.sum())
                if tot:
                    off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                    g = off + np.arange(tot)
                    v += np.bincount(indices[g], weights=wdata[g] * np.repeat(gpn[fired], cnt), minlength=fb.n).astype(np.float32)
            refr -= 1
        # brain -> body
        if gait is not None:
            gait.read_brain(win_count, ctrl_dt * 1000.0)
            activation[:] = 0.0; cmd = gait.controls(ctrl_dt * 1000.0, act_index, activation)
        else:
            activation = np.clip(activation * MOTOR_DECAY + MOTOR_GAIN * mn_count, -1.0, 1.0); cmd = None
        ts = env.step(activation)
        if k % 25 == 0:
            xpos = env.physics.named.data.xpos
            trace.append({"t": round(k * ctrl_dt, 3), "touch": [int(x > 0) for x in np.asarray(ts.observation["walker/touch"]).ravel()],
                          "ball_qvel": [round(float(x), 3) for x in np.asarray(ts.observation.get("walker/ball_qvel", [])).ravel()[:3]],
                          "act_abs_mean": round(float(np.abs(activation).mean()), 4), "mn_spikes_window": int(np.abs(mn_count).sum()),
                          "cmd": {k_: round(float(v_), 3) for k_, v_ in cmd.items()} if cmd else None,
                          "brain_hz_per_cell": round(total_spikes / max(1e-9, (k + 1) * ctrl_dt) / fb.n, 2)})
        if ts.last():
            ts = env.reset()
    secs = n_ctrl * ctrl_dt
    out = {"chosen": {"seconds": SECONDS, "touch_hz": TOUCH_HZ, "prop_gain": PROP_HZ_PER_RAD_S, "motor_gain": MOTOR_GAIN,
                      "motor_decay": MOTOR_DECAY, "muscle_map": MUSCLE_MAP, "seed": SEED, "env": "flybody walk_on_ball", "graph": os.environ.get("BODY_GRAPH", "runtime/build/graph.npz"), "mode": MODE, "sound_hz": SOUND_HZ,
                      "gait": "Mineplix/fly-brain gait.json (MIT) via gait.py" if MODE == "descending" else None},
           "measured": {"motor_neurons_mapped": int(len(mn_idx)), "unmapped_by_type": unmapped,
                        "touch_cells": {k: int(len(v)) for k, v in touch_cells.items()}, "prop_cells": {k: int(len(v)) for k, v in prop_cells.items()},
                        "brain_spikes_per_sec_per_cell": round(total_spikes / secs / fb.n, 3), "motor_neuron_spikes": mn_spikes_total,
                        "final_touch": [int(x > 0) for x in np.asarray(ts.observation["walker/touch"]).ravel()],
                        "elapsed_s": round(time.perf_counter() - t0, 1), "sim_s": secs},
           "trace": trace}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / (os.environ.get("BODY_OUT") or "body_loop.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["measured"], indent=1)); print("trace tail:", trace[-3:])


if __name__ == "__main__":
    main()
