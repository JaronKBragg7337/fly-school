"""Fly School - the 24/7 classroom. One fly (school-1), one mushroom body that is never reset, taught around the clock.

WHY (Jaron, 2026-09-17 ~02:20): "we can't expect to be produced what we want if it's not something that constantly
24/7 ... every time we started it again maybe that's what's wrong is that it's a new fly that's why I can't remember
anything." He is right: every Grade 1 attempt (rows 1-6) deleted the store, ran ~10 minutes on a frozen clock, and
threw the fly away. This process keeps ONE fly alive: its KC->MBON gains persist in mb_school-1.npz, its clock is the
wall clock (so forgetting is real: half-life 6 h, Tully & Quinn), and it takes lesson after lesson until stopped.

PROTOCOL (unchanged from Grade 1 so the exams stay comparable - versions/README.md rule 2):
  teach  : TEACH_EPOCHS x ('.', '-') on 50 JO-A cells at 120 Hz; dopamine +1 exact / -1 else; membrane reset per trial
  exam   : EXAM_REPS x ('.-', '-.') cold - no dopamine, weights untouched
  rest   : REST_TRIALS trials with no sound at all - what the fly does when nobody is teaching it (measured, not assumed)
Lesson 0 is an exam only (the baseline before any teaching). Lesson n = teach, exam, rest.

LIFE RULES (Jaron, 2026-09-17 ~22:40 EDT - "the MSI is not just a school, it's the life source"):
  * Nobody is taught 24/7. SCHOOL HOURS (CHOSEN): 09:00-12:00 and 13:00-16:00 America/New_York, every day. Lessons run
    only then. The first exam of each school day is the MORNING EXAM: what survived the night (memory half-life 6 h,
    so ~17 h without lessons leaves ~14% - spaced vs massed training, the question fly labs actually ask).
  * FREE TIME (waking hours outside school): the fly is alive and shown alive - one silent trial a minute (measured, not
    assumed) - and it EATS: a meal every MEAL_EVERY_H hours (CHOSEN 3 h). A meal is a clock event today (fed_at, hunger),
    not a dopamine event: sugar with nothing recent in the eligibility trace would change no weight, and the counter
    would lie. When the body-state layer exists, hunger becomes tonic drive. Said so on the page.
  * NIGHT (22:00-07:00): sleeping - one silent trial every five minutes. Flies sleep at night; forgetting runs on.
  * The MSI is the life source. If it is off, the fly is off; its memory keeps fading by the wall clock while it is
    off (load() applies the missed forgetting). A day off = ~6% of the memory left. That is a real death of everything
    learned, and the reason for the cloud goal. Until then a LIFELINE copy of the memory file goes to Supabase storage
    after every exam, so the fly can be revived on another machine with whatever it still had.
  * Work is a new student. A pupil in school stays in school; a working fly (the trader) is enrolled separately.

SHIFTS (Jaron, 2026-09-17 ~23:20 EDT): a fly is in class at every hour of the day, for him (he sleeps in increments) and
  for anyone in any time zone. Pupils have different school hours (SCHOOL_HOURS / SCHOOL_NIGHT env, local time), so
  while one sleeps another is in class. A new pupil may START FROM an older pupil's memory (SCHOOL_INHERIT=school-1):
  "only in quotes a new fly" - it continues where that one was; the older one keeps its own life. The classroom has a
  CAPACITY (this machine); when the machine dies the year ends: on the next start after a long gap, the report card is
  archived as that year's and the year counter goes up. Transparent, on the page.
Brain: fly-v1 (graph.npz, calibration CHOSEN, no leg drive, no axon gate). Output: DNa01 (2 cells), 10 ms bins.

WHAT IT PUBLISHES (the classroom page reads these; the service key never leaves this machine):
  fly_live  row fly='school-1'  - overwritten every trial: phase, what was played, what came out, atlas neurons that fired
  fly_school row per trial      - the permanent log; exams are the report card
  Fly-Lab-2/school/report-card.json + AI-Shared/state/fly-brain/live.school-1.json - local mirrors
Author: Claude Code (Opus 5), 2026-09-17. Constants marked CHOSEN are ours; everything else is measured.
"""
from __future__ import annotations
import hashlib, json, os, shutil, sys, time, traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RUNTIME = Path(r"C:\Users\lilli\AI-Shared\projects\fly-brain\runtime")
STATE = Path(r"C:\Users\lilli\AI-Shared\state\fly-brain")
sys.path.insert(0, str(RUNTIME))
from flysim import FlyBrain            # noqa: E402
from mushroom import MushroomBody      # noqa: E402
import calibration as C                # noqa: E402
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Enrolment (Jaron 2026-09-17: a new fly is ANOTHER student, never a replacement). One scheduled task per student:
#   SCHOOL_FLY=school-2 SCHOOL_VERSION=fly-v1-axon SCHOOL_BRAIN=axon  ->  its own store, state, card, log, page /school/school-2/
FLY = os.environ.get("SCHOOL_FLY", "school-1")
VERSION = os.environ.get("SCHOOL_VERSION", "fly-v1")
BRAIN = os.environ.get("SCHOOL_BRAIN", "")          # "" = fly-v1 graph.npz; "axon" = fly-v1-axon (gate wired below)
BASE_SEED = 1 if FLY == "school-1" else int.from_bytes(hashlib.sha256(FLY.encode()).digest()[:2], "little")   # CHOSEN; per pupil so shifts diverge; school-1 keeps 1
# Grade 1 constants (identical to morse_grade1.py)
DOT_MS, DASH_MS, GAP_MS, PRE_MS, TAIL_MS = 20.0, 60.0, 40.0, 20.0, 40.0
JO_HZ, BIN_MS, DN_SPIKES_PER_BIN, DOT_DASH_SPLIT_MS = 120.0, 10.0, 1, 40.0
TRAIN, HELDOUT = (".", "-"), (".-", "-.")
TEACH_EPOCHS = int(os.environ.get("SCHOOL_TEACH_EPOCHS", 12))   # CHOSEN: half a Grade-1 attempt per lesson (~3 min)
EXAM_REPS = 6                                                   # same as Grade 1 cold test
REST_TRIALS = int(os.environ.get("SCHOOL_REST_TRIALS", 2))      # CHOSEN: two silent trials, measured not assumed
TZ = "America/New_York"
def _hours(spec, default):
    try:
        return tuple(tuple(int(x) for x in part.split("-")) for part in spec.split(",")) if spec else default
    except Exception:
        return default
SCHOOL_HOURS = _hours(os.environ.get("SCHOOL_HOURS", ""), ((9, 12), (13, 16)))     # CHOSEN per pupil (local, [start,end))
NIGHT = _hours(os.environ.get("SCHOOL_NIGHT", ""), ((22, 7),))[0]                 # CHOSEN per pupil; may wrap midnight
INHERIT = os.environ.get("SCHOOL_INHERIT", "")      # older pupil whose memory this one starts from (same graph only)
NEW_YEAR_GAP_H = 3.0                                # CHOSEN: a gap this long since the last sign of life ends the year
MEAL_EVERY_H = float(os.environ.get("SCHOOL_MEAL_EVERY_H", 3.0))   # CHOSEN
FREE_TICK_S, SLEEP_TICK_S = 60, 300         # CHOSEN: how often the fly is shown alive when not in school
LIFELINE_BUCKET = "fly-lifeline"
STORE = HERE / f"mb_{FLY}.npz"
STATE_FILE = HERE / f"{FLY}.state.json"
CARD = HERE / ("report-card.json" if FLY == "school-1" else f"report-card.{FLY}.json")
LOG = HERE / ("school.log" if FLY == "school-1" else f"school.{FLY}.log")
if BRAIN == "axon":
    sys.path.insert(0, r"C:/Users/lilli/Fly-Lab/versions/fly-v1-axon")
    from flysim_axon import FlyBrainAxon as FlyBrain   # noqa: F811
    GRAPH = Path(r"C:/Users/lilli/Fly-Lab/versions/fly-v1-axon/graph_axon.npz")
else:
    GRAPH = RUNTIME / "build" / "graph.npz"


def now():
    return datetime.now(timezone.utc)


def log(msg):
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def secrets():
    out = {}
    p = Path.home() / ".secrets" / "keys.env"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def sb(env, path, body, prefer="return=minimal"):
    import urllib.request
    url = f"{env['SUPABASE_URL'].rstrip('/')}/rest/v1/{path}"
    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    req = urllib.request.Request(url, data=json.dumps(body, separators=(",", ":")).encode(), method="POST",
                                 headers={"apikey": key, "Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json", "Prefer": prefer})
    urllib.request.urlopen(req, timeout=20).read()


def local_now():
    return datetime.now(ZoneInfo(TZ))


def period(t=None):
    """'school' | 'free' | 'sleep' for a local time (CHOSEN hours above)."""
    t = t or local_now()
    h = t.hour + t.minute / 60.0
    a, b = NIGHT
    if (h >= a or h < b) if a > b else (a <= h < b):
        return "sleep"
    if any(a <= h < b for a, b in SCHOOL_HOURS):
        return "school"
    return "free"


def next_school_start(t=None):
    t = t or local_now()
    for d in range(0, 3):
        day = (t + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        for a, _ in SCHOOL_HOURS:
            cand = day.replace(hour=a)
            if cand > t:
                return cand
    return t


def lifeline(env, path, name):
    """Copy a file to Supabase storage (bucket fly-lifeline). Returns True when stored. Never blocks the fly."""
    import urllib.request
    try:
        key = env["SUPABASE_SERVICE_ROLE_KEY"]
        url = f"{env['SUPABASE_URL'].rstrip('/')}/storage/v1/object/{LIFELINE_BUCKET}/{name}"
        data = open(path, "rb").read()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}",
                                              "Content-Type": "application/octet-stream", "x-upsert": "true"})
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:
        log(f"lifeline failed ({name}): {str(e)[:120]}")
        return False


# ---- the protocol (copied from morse_grade1.py so the school and the exams agree byte for byte) ----
def ms_steps(ms, dt):
    return max(1, int(round(ms / dt)))


def make_timeline(pattern, dt):
    pre, tail, gap = ms_steps(PRE_MS, dt), ms_steps(TAIL_MS, dt), ms_steps(GAP_MS, dt)
    parts = []
    for i, ch in enumerate(pattern):
        parts.append((ms_steps(DOT_MS if ch == "." else DASH_MS, dt), True))
        if i + 1 < len(pattern):
            parts.append((gap, False))
    total = pre + sum(n for n, _ in parts) + tail
    sound = np.zeros(total, dtype=bool)
    p = pre
    for n, on in parts:
        if on:
            sound[p:p + n] = True
        p += n
    return sound, pre


def seed_for(lesson, phase, pattern, rep):
    b = hashlib.sha256(f"{BASE_SEED}:L{lesson}:{phase}:{pattern}:{rep}".encode()).digest()[:4]
    return int.from_bytes(b, "little")


def decode(counts, pre_bins):
    counts = np.asarray(counts, dtype=int)
    active = counts[pre_bins:] >= DN_SPIKES_PER_BIN
    runs, n, i = [], len(active), 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j]:
            j += 1
        runs.append((i, j))
        i = j
    split_bins = max(1, int(round(DOT_DASH_SPLIT_MS / BIN_MS)))
    return "".join("." if (b - a) < split_bins else "-" for a, b in runs), active.astype(int).tolist()


class School:
    def __init__(self):
        self.fb = FlyBrain(GRAPH)
        self.inherited = None
        if INHERIT and not STORE.exists():
            src = HERE / f"mb_{INHERIT}.npz"
            if src.exists():
                shutil.copy(src, STORE); self.inherited = INHERIT
        self.mb = MushroomBody(self.fb, calibration=C.CHOSEN, sides=RUNTIME / "build" / "mb_sides.json",
                               store=STORE, clock=time.time)          # wall clock: forgetting is real
        self.jo = self.fb.where(type_re=r"^JO-A")
        self.dn = self.fb.where(type_re=r"^DNa01$")
        gains = C.gains_for(self.fb, C.CHOSEN)
        self.gain_per_neuron = (np.ones(self.fb.n, dtype=np.float32) if gains is None
                                else gains[self.fb.type_code].astype(np.float32))
        self.kc_mask = np.zeros(self.fb.n, dtype=bool); self.kc_mask[self.mb.kc] = True
        self.dn_mask = np.zeros(self.fb.n, dtype=bool); self.dn_mask[self.dn] = True
        self.env = secrets()
        try:
            idx = json.load(open(STATE / "atlas.json", encoding="utf-8"))["index"]
            self.atlas = {int(n): i for i, n in enumerate(idx)}
        except Exception:
            self.atlas = {}
        self.state = {"lesson": 0, "born": now().isoformat(), "trials": 0, "fed_at": None, "meals": 0, "school_day": None}
        if STATE_FILE.exists():
            try:
                self.state.update(json.load(open(STATE_FILE, encoding="utf-8")))
            except Exception:
                pass
        self.card = json.load(open(CARD, encoding="utf-8")) if CARD.exists() else {"fly": FLY, "version": VERSION, "exams": []}
        self.started = time.time()
        if self.inherited:
            self.state["inherited_from"] = self.inherited
            self.state["born"] = now().isoformat()
        # the school year: a long gap since the last sign of life (machine off) archives the card and starts a new year
        self.state.setdefault("year", 1)
        last = self.state.get("last_alive")
        if last and (time.time() - datetime.fromisoformat(last).timestamp()) / 3600 >= NEW_YEAR_GAP_H:
            arch = CARD.with_name(CARD.stem + f".year{self.state['year']}.json")
            try:
                json.dump(self.card, open(arch, "w", encoding="utf-8"), indent=1)
            except OSError:
                pass
            self.state["year"] += 1
            self.state["year_started"] = now().isoformat()
            self.card = {"fly": FLY, "version": VERSION, "exams": [], "year": self.state["year"]}
            log(f"NEW YEAR {self.state['year']}: the machine was off {((time.time() - datetime.fromisoformat(last).timestamp()) / 3600):.1f} h; "
                f"last year's card archived to {arch.name}; memory carried with the forgetting applied")
        self.state.setdefault("year_started", self.state.get("born"))
        log(f"school open: brain n={self.fb.n} JO-A={len(self.jo)} DNa01={len(self.dn)} store_loaded={self.mb.loaded} "
            f"lesson={self.state['lesson']} mb={self.mb.stats()}")

    # one trial: the sound plays (or nothing plays), the whole brain runs, DNa01 is read in 10 ms bins
    def trial(self, pattern, seed):
        fb, p = self.fb, self.fb.p
        sound, pre_steps = make_timeline(pattern or "-", p.dt)      # rest trials: same length as a dash trial, sound off
        if not pattern:
            sound[:] = False
        rng = np.random.default_rng(seed)
        v = np.full(fb.n, p.v_rest, dtype=np.float32)
        refr = np.zeros(fb.n, dtype=np.int32)
        bin_steps = ms_steps(BIN_MS, p.dt)
        counts = np.zeros(int(np.ceil(len(sound) / bin_steps)), dtype=np.int32)
        ever = np.zeros(fb.n, dtype=bool)
        prob = min(1.0, JO_HZ * p.dt / 1000.0)
        indptr, indices, wdata = fb.indptr, fb.indices, fb.wdata
        axonal = getattr(fb, "axonal", None)
        use_gate = axonal is not None and bool(np.any(axonal))
        gate_in = np.zeros(fb.n, dtype=np.float32)
        gate_decay = float(getattr(fb, "gate_decay", 1.0))
        total = 0
        for step, sound_on in enumerate(sound):
            v = p.v_rest + (v - p.v_rest) * fb.decay
            if use_gate:
                gate_in *= gate_decay
            if sound_on:
                hit = self.jo[rng.random(len(self.jo)) < prob]
                if len(hit):
                    v[hit] = p.v_thresh + 1.0
            v[refr > 0] = p.v_reset
            fired = np.flatnonzero((v >= p.v_thresh) & (refr <= 0))
            if len(fired):
                total += len(fired)
                ever[fired] = True
                refr[fired] = fb.refr_steps
                v[fired] = p.v_reset
                counts[min(step // bin_steps, len(counts) - 1)] += int(self.dn_mask[fired].sum())
                starts = indptr[fired]
                cnt = indptr[fired + 1] - starts
                tot = int(cnt.sum())
                if tot:
                    off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                    g = off + np.arange(tot)
                    tgt = indices[g]
                    if use_gate:
                        gate = np.clip(1.0 + 0.05 * gate_in[fired], 0.0, 3.0).astype(np.float32)   # CHOSEN, = flysim_axon
                        val = wdata[g] * np.repeat(self.gain_per_neuron[fired] * gate, cnt)
                        ax = axonal[g]
                        if ax.any():
                            gate_in += np.bincount(tgt[ax], weights=val[ax], minlength=fb.n).astype(np.float32)
                            val = val[~ax]; tgt = tgt[~ax]
                    else:
                        val = wdata[g] * np.repeat(self.gain_per_neuron[fired], cnt)
                    v += np.bincount(tgt, weights=val, minlength=fb.n).astype(np.float32)
            refr -= 1
        pre_bins = int(round(pre_steps / bin_steps))
        decoded, active = decode(counts, pre_bins)
        secs = len(sound) * p.dt / 1000.0
        return {"target": pattern, "decoded": decoded, "correct": bool(pattern) and decoded == pattern,
                "dn_counts": counts.tolist(), "dn_active": active, "pre_bins": pre_bins,
                "kc": np.flatnonzero(ever & self.kc_mask), "fired": np.flatnonzero(ever),
                "spikes_per_sec": total / secs, "kc_fired": int((ever & self.kc_mask).sum()),
                "dn_spikes": int(counts.sum()), "sim_ms": len(sound) * p.dt}

    def publish(self, phase, lesson, epoch, r, dopamine=None, hit=None, note=None):
        at = now().isoformat()
        mb = self.mb.stats()
        self.state["trials"] += 1
        last_exam = self.card["exams"][-1] if self.card["exams"] else None
        state = {"phase": phase, "lesson": lesson, "epoch": epoch, "symbol": r["target"], "decoded": r["decoded"],
                 "correct": r["correct"], "dopamine": dopamine, "synapses_hit": hit,
                 "dn_counts": r["dn_counts"], "dn_active": r["dn_active"], "pre_bins": r["pre_bins"], "bin_ms": BIN_MS,
                 "spikes_per_sec": round(r["spikes_per_sec"], 1), "kc": r["kc_fired"], "dn_spikes": r["dn_spikes"],
                 "sim_ms": r["sim_ms"], "fired_total": int(len(r["fired"])),
                 "mb": mb, "version": VERSION, "born": self.state["born"], "trials": self.state["trials"],
                 "uptime_s": int(time.time() - self.started), "last_exam": last_exam, "brain": BRAIN or "v1",
                 "teach_epochs": TEACH_EPOCHS, "exam_reps": EXAM_REPS, "rest_trials": REST_TRIALS, "note": note,
                 "period": period(), "tz": TZ, "school_hours": SCHOOL_HOURS, "night": NIGHT,
                 "next_school": next_school_start().isoformat(), "fed_at": self.state.get("fed_at"),
                 "meals": self.state.get("meals", 0), "meal_every_h": MEAL_EVERY_H,
                 "hungry_h": round((time.time() - datetime.fromisoformat(self.state["fed_at"]).timestamp()) / 3600, 2) if self.state.get("fed_at") else None,
                 "lifeline_at": self.state.get("lifeline_at"), "school_day": self.state.get("school_day"),
                 "year": self.state.get("year", 1), "year_started": self.state.get("year_started"),
                 "inherited_from": self.state.get("inherited_from"), "capacity": int(os.environ.get("SCHOOL_CAPACITY", 3))}
        self.state["last_alive"] = at
        fired = [self.atlas[int(n)] for n in r["fired"] if int(n) in self.atlas]
        if len(fired) > 3000:
            fired = [int(x) for x in np.random.default_rng(0).choice(fired, 3000, replace=False)]
        state["fired_shown"] = len(fired)
        try:
            STATE.mkdir(parents=True, exist_ok=True)
            json.dump({"at": at, "mode": "school", "state": state, "fired": fired},
                      open(STATE / f"live.{FLY}.json", "w", encoding="utf-8"), separators=(",", ":"))
        except OSError:
            pass
        try:
            sb(self.env, "fly_live?on_conflict=fly", {"fly": FLY, "at": at, "mode": "school", "state": state, "fired": fired},
               prefer="return=minimal,resolution=merge-duplicates")
        except Exception as e:
            log(f"fly_live push failed: {e}")
        try:
            sb(self.env, "fly_school", {"at": at, "fly": FLY, "version": VERSION, "lesson": lesson, "phase": phase,
                                        "epoch": epoch, "symbol": r["target"], "decoded": r["decoded"],
                                        "correct": r["correct"] if r["target"] else None, "dopamine": dopamine,
                                        "synapses_hit": hit, "dn_counts": r["dn_counts"] if phase != "teach" else None,   # teach strips live in fly_live only (row size)
                                        "mb": mb, "note": note})
        except Exception as e:
            log(f"fly_school insert failed: {e}")
        try:
            json.dump(self.state, open(STATE_FILE, "w", encoding="utf-8"))
        except OSError:
            pass

    def exam(self, lesson, morning=False):
        self.mb.forget()                      # time that passed is forgetting, before we look
        rows = []
        for pattern in HELDOUT:
            for rep in range(EXAM_REPS):
                r = self.trial(pattern, seed_for(lesson, "exam", pattern, rep))
                rows.append(r)
                self.publish("exam", lesson, rep, r)
        summary = {p: {"correct": sum(r["correct"] for r in rows if r["target"] == p), "trials": EXAM_REPS,
                       "decoded": [r["decoded"] for r in rows if r["target"] == p]} for p in HELDOUT}
        entry = {"lesson": lesson, "at": now().isoformat(), "age_h": round((time.time() - self.born_ts()) / 3600, 2),
                 "morning": bool(morning), "heldout": summary, "mb": self.mb.stats(), "rewards_total": self.mb.events["reward"],
                 "punishments_total": self.mb.events["punish"]}
        self.card["exams"].append(entry)
        json.dump(self.card, open(CARD, "w", encoding="utf-8"), indent=1)
        log(f"exam L{lesson}{' MORNING' if morning else ''}: " + "  ".join(f"{p} {s['correct']}/{s['trials']} {s['decoded']}" for p, s in summary.items()))
        self.mb.save()
        ok = (lifeline(self.env, STORE, f"{FLY}/mb.npz") and lifeline(self.env, CARD, f"{FLY}/report-card.json")
              and lifeline(self.env, STATE_FILE, f"{FLY}/state.json"))
        if ok:
            self.state["lifeline_at"] = now().isoformat()

    def born_ts(self):
        try:
            return datetime.fromisoformat(self.state["born"]).timestamp()
        except Exception:
            return self.started

    def teach(self, lesson):
        ok = 0
        for epoch in range(TEACH_EPOCHS):
            for symbol in TRAIN:
                self.mb.forget()
                r = self.trial(symbol, seed_for(lesson, "teach", symbol, epoch))
                self.mb.forget_trace(); self.mb.observe(r["kc"])
                valence = +1 if r["correct"] else -1
                hit = int(self.mb.dopamine(valence, 1.0))
                self.mb.apply(); self.mb.save()
                ok += r["correct"]
                self.publish("teach", lesson, epoch, r, dopamine=valence, hit=hit)
        log(f"teach L{lesson}: {ok}/{TEACH_EPOCHS * len(TRAIN)} primitives exact; mb={self.mb.stats()}")

    def rest(self, lesson):
        for rep in range(REST_TRIALS):
            self.mb.forget()
            r = self.trial("", seed_for(lesson, "rest", "", rep))
            self.publish("rest", lesson, rep, r, note="no sound; whatever fires here is the fly on its own")
        log(f"rest L{lesson}: spikes/s {r['spikes_per_sec']:.1f}, DNa01 {r['dn_spikes']}")

    def alive(self, what, tick_s):
        """Free time / sleep: one silent trial so 'alive' is measured, then wait. Meals during free time."""
        self.mb.forget()
        fed = self.state.get("fed_at")
        hungry_h = (time.time() - datetime.fromisoformat(fed).timestamp()) / 3600 if fed else 1e9
        if what == "free" and hungry_h >= MEAL_EVERY_H:
            self.state["fed_at"] = now().isoformat(); self.state["meals"] = self.state.get("meals", 0) + 1
            r = self.trial("", seed_for(self.state["lesson"], "meal", "", self.state["meals"]))
            self.publish("meal", self.state["lesson"], self.state["meals"], r,
                         note="a meal: a clock event today, not dopamine (nothing recent to pair it with); becomes body state later")
            log(f"meal #{self.state['meals']}")
        else:
            r = self.trial("", seed_for(self.state["lesson"], what, "", int(time.time() // tick_s)))
            self.publish(what, self.state["lesson"], None, r,
                         note="free time: no lesson, the fly on its own" if what == "free" else "night: sleeping; forgetting runs on")
        time.sleep(tick_s)

    def run(self):
        if self.state["lesson"] == 0 and not self.card["exams"]:
            self.exam(0)                       # baseline: what the fly can do before anyone teaches it
        while True:
            try:
                what = period()
                if what == "sleep":
                    self.alive("sleep", SLEEP_TICK_S); continue
                if what == "free":
                    self.alive("free", FREE_TICK_S); continue
                today = local_now().date().isoformat()
                if self.state.get("school_day") != today:
                    self.state["school_day"] = today
                    self.exam(self.state["lesson"], morning=True)     # what survived the night, before any teaching
                    continue
                lesson = self.state["lesson"] + 1
                self.state["lesson"] = lesson
                self.teach(lesson)
                self.exam(lesson)
                self.rest(lesson)
            except Exception:
                log("lesson failed:\n" + traceback.format_exc())
                time.sleep(30)


if __name__ == "__main__":
    School().run()
