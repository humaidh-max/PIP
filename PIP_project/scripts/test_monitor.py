import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pip.sim import HomeSim
from pip.monitor import RoutineMonitor
sim, mon = HomeSim(3, anomaly_rate=0.0), RoutineMonitor()
for _ in range(1440 * 10):                       # 10 clean days: learn routines
    o = sim.step(); [mon.observe(e, o["minute"]) for e in o["events"]]; mon.tick(o["minute"], o["t"] // 1440)
sim.anomaly_rate = 0.15; hits, alerts, truth = 0, [], []
for _ in range(1440 * 120):
    o = sim.step(); day = o["t"] // 1440
    for e in o["events"]: alerts += [(day, a) for a in mon.observe(e, o["minute"])]
    alerts += [(day, a) for a in mon.tick(o["minute"], day)]
    if o["minute"] == 1439: truth += [(day, k) for k, _ in sim.truth]
M = {"skip_breakfast_elder": "missed_meal", "door_left_open": "door_open", "elder_inactive": "inactivity", "child_up_at_night": "unusual_activity"}
tp = sum(any(a.kind == M[k] and d == dd for dd, a in alerts) for d, k in truth)
fp = [a for d, a in alerts if not any(M[k] == a.kind and d == dd for dd, k in truth)]
print(f"injected {len(truth)}  detected {tp}  false alerts {len(fp)}"); [print(" FP:", a.kind, a.who, a.msg) for a in fp[:6]]
for k in M: print(k, sum(1 for _, kk in truth if kk == k), sum(1 for d, kk in truth if kk == k and any(a.kind == M[k] and dd == d for dd, a in alerts)))
