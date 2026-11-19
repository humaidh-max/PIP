"""Idle-time background learner. Replays the private log, fine-tunes SHADOW copies of the nets in short
time-sliced bursts (bounded duty cycle), validates on a held-out slice, and only then atomically swaps weights
into the live brain. Interrupts instantly on user activity; never blocks the interaction loop."""
import threading, time, os, numpy as np
from .brain import emo_update, pol_update, pref_update

class IdleGate:
    def __init__(s, idle_secs=30.0, max_load=0.7):
        s.last = time.monotonic() - idle_secs; s.idle_secs, s.max_load = idle_secs, max_load; s.forced = None
    def touch(s): s.last = time.monotonic()                 # call on any interaction / wake word / motion
    def set_busy(s, busy): s.forced = busy                  # simulator / schedule override
    def idle(s):
        if s.forced is not None: return not s.forced
        try: load = os.getloadavg()[0] / (os.cpu_count() or 1)
        except (AttributeError, OSError): load = 0.0
        return time.monotonic() - s.last > s.idle_secs and load < s.max_load

class BackgroundLearner:
    def __init__(s, brain, log, gate, batch=96, steps=40, lr=4e-4, duty=0.15, min_log=512, seed=0):
        s.b, s.log, s.g = brain, log, gate; s.batch, s.steps, s.lr, s.duty, s.min_log = batch, steps, lr, duty, min_log
        s.rng = np.random.default_rng(seed); s.stop = threading.Event(); s.th = None
        s.stats = dict(rounds=0, accepted=0, rejected=0, aborted=0, steps=0)
    def _eval(s, nets, H):
        return emo_update(nets["emo"], H, False) + pol_update(nets["pol"], nets["val"], H, False) + pref_update(nets["pref"], H, False)
    def run_round(s):
        """One bounded fine-tune round. Returns 'accepted'|'rejected'|'aborted'|'skipped'."""
        if len(s.log) < s.min_log: return "skipped"
        with s.b.lock: sh = {k: n.clone() for k, n in s.b.nets.items()}
        H = s.log.sample(256, recency=.7, rng=s.rng)                       # held-out probe (recent-heavy)
        base = s._eval(s.b.nets, H)
        for _ in range(s.steps):
            if not s.g.idle(): s.stats["aborted"] += 1; return "aborted"    # user is back: drop the round, no swap
            t0 = time.perf_counter(); B = s.log.sample(s.batch, recency=.5, rng=s.rng)
            emo_update(sh["emo"], B, lr=s.lr); pol_update(sh["pol"], sh["val"], B, lr=s.lr); pref_update(sh["pref"], B, lr=s.lr)
            s.stats["steps"] += 1; time.sleep((time.perf_counter() - t0) * (1 / s.duty - 1))  # cap CPU duty cycle
        s.stats["rounds"] += 1
        if s._eval(sh, H) <= base * 1.01:                                   # regression guard
            with s.b.lock:
                for k, n in s.b.nets.items(): n.load_from(sh[k])
            s.stats["accepted"] += 1; return "accepted"
        s.stats["rejected"] += 1; return "rejected"
    def _loop(s):
        while not s.stop.is_set():
            if s.g.idle(): s.run_round()
            s.stop.wait(0.05 if s.g.idle() else 0.5)
    def start(s): s.th = threading.Thread(target=_loop_wrap(s), daemon=True); s.th.start(); return s
    def close(s): s.stop.set(); s.th and s.th.join(2)
def _loop_wrap(s):
    try: os.nice(10)
    except Exception: pass
    return s._loop
