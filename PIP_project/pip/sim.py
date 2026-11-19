"""Lightweight virtual home (numpy only, ~minute-resolution, thousands of days/hour on a laptop).
Household: adult (PIP's user), child, elder, cat. PIP sits in the living room.
Provides: sensor events, noisy voice/face cues, injected anomalies (ground truth), and the
*simulated human* feedback (comfort reward, welcomed/annoyed) used to train PIP."""
import numpy as np
from .spec import *

ROOMS = ["bedroom", "living", "kitchen", "hall"]
MEALS = {"breakfast": 450, "lunch": 750, "dinner": 1140}
ANOMALIES = ["skip_breakfast_elder", "door_left_open", "child_up_at_night", "elder_inactive"]
# hidden "what looks right" tables used by the simulated human
_T = {  # cont target (x,y,z,roll,pitch,yaw,antL,antR,body), preferred sounds, preferred speech
 "calm":    ([0, 0, 0, 0, 0, 0, .1, .1, 0],            [0, 3], [0]),
 "happy":   ([0, 0, .5, 0, .4, 0, .8, .8, .3],         [1, 4], [1, 5]),
 "curious": ([0, 0, .2, .6, .2, .5, .7, .2, .4],       [4, 1], [4]),
 "sleepy":  ([0, 0, -.6, 0, -.6, 0, -.7, -.7, 0],      [0, 3], [0]),
 "worried": ([0, 0, -.2, -.3, -.1, 0, -.2, .2, 0],     [2], [2, 3]),
 "lonely":  ([0, 0, -.3, 0, -.3, .4, -.4, -.4, .4],    [2, 3], [1]),
}

class HomeSim:
    def __init__(s, seed=0, anomaly_rate=0.12):
        s.r = np.random.default_rng(seed); s.t = 0; s.anomaly_rate = anomaly_rate
        s.style = 1.0                      # hidden user preference: how expressive PIP should be
        s.uv, s.ua = .2, 0.0               # adult affect
        s.teach = np.r_[0., 0., 1, np.zeros(NM - 1)]  # teacher (appraisal) state
        s.last_event = {"door": -99, "meal": -99, "noise": -99, "visitor": -99}
        s.last_interact = 0; s._new_day()
    # ---------------- world ----------------
    def _new_day(s):
        r = s.r; s.day = s.t // 1440
        s.anom = {a for a in ANOMALIES if r.random() < s.anomaly_rate}
        s.truth = []                                           # [(kind, minute)] injected this day
        who = ["adult", "child", "elder"]
        s.meal_t = {(w, m): MEALS[m] + r.normal(0, 12) + {"adult": 0, "child": 10, "elder": -20}[w] * (m == "breakfast")
                    for w in who for m in MEALS}
        s.wake = {"adult": 400 + r.normal(0, 10), "child": 420 + r.normal(0, 10), "elder": 370 + r.normal(0, 15)}
        s.bed = {"adult": 1400 + r.normal(0, 20), "child": 1230 + r.normal(0, 15), "elder": 1290 + r.normal(0, 20)}
        s.leave = 500 + r.normal(0, 10); s.back = 1050 + r.normal(0, 20)
        s.loc = {"adult": "bedroom", "child": "bedroom", "elder": "bedroom", "cat": "living"}
        s.door_open = False; s.door_open_t = 0; s.door_close_at = -1; s.meal_seen = set(); s.in_kitchen = {w: 0 for w in who}
        s.inactive = (r.integers(840, 960), 120) if "elder_inactive" in s.anom else None
        s.night_walk = r.integers(120, 240) if "child_up_at_night" in s.anom else None
        s.alert_level = 0.0
    def _away(s, w, m):
        if w == "adult": return s.leave <= m < s.back
        if w == "child": return 480 <= m < 900 and s.day % 7 < 5
        return False
    def _where(s, w, m):
        if m < s.wake[w] or m >= s.bed[w]:
            if w == "child" and s.night_walk and s.night_walk <= m < s.night_walk + 30: return "kitchen"
            return "bedroom"
        if s._away(w, m): return "away"
        for meal in MEALS:
            if s.meal_t[(w, meal)] <= m < s.meal_t[(w, meal)] + 20:
                if w == "elder" and meal == "breakfast" and "skip_breakfast_elder" in s.anom: return "bedroom"
                return "kitchen"
        if s.inactive and s.inactive[0] <= m < sum(s.inactive) and w == "elder": return s.loc["elder"]
        if s.r.random() < .03 or s.loc[w] in ("away",): return s.r.choice(["living", "living", "bedroom", "hall"])
        return s.loc[w]
    def step(s):
        """Advance one minute -> observation dict + list of sensor events."""
        if s.t and s.t % 1440 == 0: s._new_day()
        m = s.t % 1440; ev = []; r = s.r
        for w in ["adult", "child", "elder"]:
            new = s._where(w, m); old = s.loc[w]
            if new == "away" and old != "away":
                ev.append(("door_open", w, "leave"))
                s.door_open, s.door_open_t = True, m
                s.door_close_at = m + (40 if "door_left_open" in s.anom and w == "adult" else 1)
                if "door_left_open" in s.anom and w == "adult": s.truth.append(("door_left_open", m))
            if old == "away" and new != "away":
                ev.append(("door_open", w, "enter")); s.door_open, s.door_open_t = True, m; s.door_close_at = m + 1
            if new == "kitchen":
                s.in_kitchen[w] += 1
                for meal in MEALS:
                    if s.meal_t[(w, meal)] <= m < s.meal_t[(w, meal)] + 20 and s.in_kitchen[w] >= 5 and (w, meal) not in s.meal_seen:
                        s.meal_seen.add((w, meal)); ev.append(("meal", w, meal))
            else: s.in_kitchen[w] = 0
            s.loc[w] = new
            silent = s.inactive and s.inactive[0] <= m < sum(s.inactive) and w == "elder"
            if new not in ("away",) and not (m < s.wake[w] or m >= s.bed[w]) and not silent and r.random() < .35:
                ev.append(("motion", w, new))
            if w == "child" and s.night_walk and s.night_walk <= m < s.night_walk + 30 and r.random() < .5:
                ev.append(("motion", w, "kitchen"))
        if s.door_open and m >= s.door_close_at: ev.append(("door_close", "adult", "")); s.door_open = False
        # pet
        act = .8 if (m < 420 or 1080 < m < 1200) else .3
        if r.random() < .05 * act + .01: s.loc["cat"] = r.choice(["living", "kitchen", "bedroom"])
        if "elder_inactive" in s.anom and s.inactive and m == s.inactive[0]: s.truth.append(("elder_inactive", m))
        if "skip_breakfast_elder" in s.anom and m == 600: s.truth.append(("skip_breakfast_elder", m))
        if "child_up_at_night" in s.anom and s.night_walk and m == s.night_walk: s.truth.append(("child_up_at_night", m))
        noise = float(r.random() < .004); visitor = float(r.random() < .002 and s.loc["adult"] != "away")
        for k, v in (("door", any(e[0].startswith("door") for e in ev)), ("meal", any(e[0] == "meal" for e in ev)),
                     ("noise", noise), ("visitor", visitor)):
            if v: s.last_event[k] = s.t
        # adult affect: slow AR(1) toward a time/event dependent mean
        mu = .1 + .25 * np.sin((m - 600) / 1440 * 2 * np.pi) - .5 * noise + .2 * (s.t - s.last_event["meal"] < 30)
        s.uv += .04 * (mu - s.uv) + r.normal(0, .04); s.ua += .05 * ((.3 if 420 < m < 1300 else -.5) + .6 * noise + .4 * visitor - s.ua) + r.normal(0, .05)
        s.uv, s.ua = np.clip(s.uv, -1, 1), np.clip(s.ua, -1, 1)
        present = s.loc["adult"] == "living"; sees = lambda k: s.t - s.last_event[k]
        speaking = present and r.random() < .3
        obs = dict(t=s.t, minute=m, events=ev, present=float(present),
                   voice=(np.clip(s.uv + r.normal(0, .15), -1, 1), np.clip(s.ua + r.normal(0, .15), -1, 1)) if speaking else None,
                   face=(np.clip(s.uv + r.normal(0, .1), -1, 1), np.clip(s.ua + r.normal(0, .1), -1, 1)) if present else None,
                   pet=float(s.loc["cat"] == "living") * act, door=float(sees("door") < 5), meal=float(sees("meal") < 10),
                   noise=float(sees("noise") < 3), visitor=float(sees("visitor") < 5),
                   kids_home=float(s.loc["child"] != "away"), idle_h=(s.t - s.last_interact) / 60)
        s.t += 1; s.obs = obs; return obs
    # ------------- simulated human (training signal) -------------
    def teacher(s, f):
        """Sim-only appraisal: the 'right' emotional state for PIP given the situation."""
        uv = np.nanmean([f[0], f[2]]) if (f[6] or f[0] or f[2]) else 0.0; ua = np.nanmean([f[1], f[3]]) if (f[6] or f[1] or f[3]) else 0.0
        v = .55 * uv + .25 * f[13] - .3 * f[11] + .15 * f[10] + .1 * f[8]
        a = .5 * ua + .5 * max(f[11], f[12], f[9]) + .3 * f[8] + .4 * f[15] - .2
        night = f[4] < -.5 or f[5] > .8                       # t_sin<0 around late evening / night
        sc = np.array([.4, 1.2 * v + .2, 1.4 * max(f[9], f[12], f[8]) + .1 * a, 1.0 * night * (a < .1) + .6 * night - .3 * a,
                       1.5 * max(f[15], f[11] * .8) - .5 * v, 1.2 * (f[7] > .6) * (1 - f[6])])
        tv, ta = np.clip(v, -1, 1), np.clip(a, -1, 1)
        p = np.eye(NM)[int(np.argmax(sc))]
        s.teach = .8 * s.teach + .2 * np.r_[tv, ta, p]        # emotions persist
        s.teach[2:] = s.teach[2:] / s.teach[2:].sum(); return s.teach.copy()
    def feedback(s, f, cont, snd, sp, teach):
        """Return (reward, quiet_violation). The human rewards expression that reads as the right mood."""
        mood = MOODS[int(np.argmax(teach[2:]))]; tgt, ps, pp = _T[mood]
        tgt = np.array(tgt) * s.style * (.6 + .4 * (teach[1] + 1) / 1.0 * .5 + .2)
        quiet = (f[4] < -.5 and f[6] > .5 and teach[1] < 0)
        r = -2.0 * float(np.mean((np.clip(cont, -1, 1) - tgt) ** 2)) + .5 * (snd in ps) + .5 * (sp in pp)
        if quiet and (snd != 0 or sp != 0): r -= 1.0
        s.uv = float(np.clip(s.uv + .03 * r, -1, 1)); s.last_interact = s.t if (sp or snd) and s.loc["adult"] == "living" else s.last_interact
        return float(r)
    def welcome(s, f, pa):
        """P(user welcomes a proactive action pa) — hidden; sampled to give a label."""
        if pa == 0: return None
        busy = f[1] > .5; night = f[4] < -.5; present = f[6] > .5
        p = {1: .8 * present * (1 - .6 * night) * (1 - .5 * busy) + .05, 2: .85 - .5 * night * (f[15] < .5) - .3 * busy,
             3: .6 * present * (1 - .7 * busy) * (1 - .7 * night) + .1}[pa]
        return float(s.r.random() < np.clip(p, .02, .98))
