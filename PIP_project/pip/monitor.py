"""Routine learner + anomaly alerts. Statistics are learned per household from sensor events (no hand-set schedules);
thresholds adapt as routines drift (exponentially-decayed mean/variance)."""
import numpy as np
from dataclasses import dataclass

@dataclass
class Alert:
    kind: str; who: str; minute: int; msg: str; severity: int = 1

class Stat:
    def __init__(s, decay=.95): s.n, s.m, s.v, s.k = 0, 0., 0., decay
    def add(s, x):
        s.n += 1; d = x - s.m; a = max(1 / s.n, 1 - s.k); s.m += a * d; s.v = (1 - a) * (s.v + a * d * d)
    @property
    def sd(s): return float(np.sqrt(s.v))

class RoutineMonitor:
    def __init__(s, warmup_days=7):
        s.first = {}; s.doordur = Stat(); s.hourly = {}; s.gap = {}; s.warm = warmup_days
        s.flagged = set(); s.iv = {}; s.reset_day(0); s.days = 0; s.away = set(); s.last_motion = {}; s.door_since = None; s.fired = set()
    def reset_day(s, day):
        s.day = day; s.flagged = set(); s.iv = {}; s.seen = {}; s.fired = {k for k in getattr(s, "fired", set()) if k[0] != "day"}; s.day_motion_hours = {}
    def _end_day(s):
        for k, t in s.seen.items(): s.first.setdefault(k, Stat()).add(t)
        for (who, h) in s.day_motion_hours:
            if (who, h) not in s.flagged: s.hourly[(who, h)] = s.hourly.get((who, h), 0) + 1
        s.days += 1
    def observe(s, ev, minute):
        kind, who, x = ev; h = minute // 60
        if kind == "meal": s.seen[(who, x)] = minute
        elif kind == "door_open":
            s.door_since = minute; s.last_motion[who] = minute
            if x == "leave": s.away.add(who); s.iv[who] = [minute, 1e9]
            else:
                s.away.discard(who)
                if who in s.iv: s.iv[who][1] = minute
        elif kind == "door_close":
            if s.door_since is not None:
                d = minute - s.door_since                       # don't let anomalies (door left open) poison the baseline
                if s.doordur.n < 3 or d <= max(s.doordur.m + 4 * s.doordur.sd, 15): s.doordur.add(d)
            s.door_since = None
        elif kind == "motion":
            s.day_motion_hours[(who, h)] = 1; g = s.last_motion.get(who)
            if g is not None and who not in s.away: s.gap.setdefault((who, h), Stat()).add(minute - g)
            s.last_motion[who] = minute
            return s._unusual_hour(who, h, minute)
        return []
    def _unusual_hour(s, who, h, minute):
        if s.days < 2 * s.warm: return []
        if s.hourly.get((who, h), 0) / s.days < .02 and ("day", "hr", who, h) not in s.fired:
            s.fired.add(("day", "hr", who, h)); s.flagged.add((who, h)); return [Alert("unusual_activity", who, minute, f"{who} is active at {h:02d}:00, which is rare", 2)]
        return []
    def tick(s, minute, day):
        """Call once per minute with the raw event stream already passed to observe(). Returns new alerts."""
        out = []
        if day != s.day: s._end_day(); s.reset_day(day)
        if s.days < s.warm: return out
        for (who, meal), st in s.first.items():
            key = ("day", "meal", who, meal)
            iv = s.iv.get(who); excused = iv and iv[0] < st.m + 30 and iv[1] > st.m - 10      # was out during the usual meal time
            if st.n >= s.warm and not excused and (who, meal) not in s.seen and minute > st.m + max(3 * st.sd, 30) and key not in s.fired and who not in s.away:
                s.fired.add(key); out.append(Alert("missed_meal", who, minute, f"{who} hasn't had {meal} (usually ~{int(st.m//60):02d}:{int(st.m%60):02d})", 3))
        if s.door_since is not None and s.doordur.n >= 3 and ("day", "door", s.door_since) not in s.fired:
            if minute - s.door_since > max(s.doordur.m + 4 * s.doordur.sd, 15):
                s.fired.add(("day", "door", s.door_since)); out.append(Alert("door_open", "home", minute, f"Front door open {minute - s.door_since} min", 3))
        for who, last in s.last_motion.items():
            st = s.gap.get((who, minute // 60)); key = ("day", "inact", who, last)
            rate = lambda h: s.hourly.get((who, h % 24), 0) / max(s.days, 1)
            if who in s.away or st is None or st.n < 30 or min(rate(minute // 60), rate(last // 60)) < .6 or key in s.fired: continue
            if minute - last > max(st.m + 8 * st.sd, 90):
                s.fired.add(key); out.append(Alert("inactivity", who, minute, f"No movement from {who} for {minute - last} min", 3))
        return out
