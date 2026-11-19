"""End-to-end: a week in the virtual home. PIP feels/acts, monitors routines, takes voice commands, learns while idle."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, os
from pip.sim import HomeSim
from pip.agent import PipAgent
from pip.monitor import RoutineMonitor
from pip.assistant import Assistant, SmartHome, Device
from pip.learner import BackgroundLearner, IdleGate
from pip.spec import PROACTIVE
sim, agent, mon = HomeSim(5, 0.0), PipAgent(), RoutineMonitor()
if os.path.exists("pip_brain.npz"): agent.brain.load("pip_brain.npz"); print("loaded pretrained brain")
home = SmartHome([Device("living room light", "light", on=False), Device("front door", "lock", locked=True), Device("thermostat", "climate", target=21)])
asst = Assistant(home); gate = IdleGate(); learner = BackgroundLearner(agent.brain, agent.log, gate, min_log=300).start()
for cmd in ["turn on the living room light", "set the thermostat to 23", "lock the front door", "set a timer for 10 minutes", "tell me a joke"]: print("voice:", cmd, "->", asst.handle(cmd))
alerts = []; hold = -99
for d in range(14):
    if d == 10: sim.anomaly_rate = 1.0
    for _ in range(1440):
        o = sim.step(); day = o["t"] // 1440; busy = o["present"] > 0 and o["voice"] is not None; gate.set_busy(busy)
        for e in o["events"]: alerts += [(day, o["minute"], a) for a in mon.observe(e, o["minute"])]
        new = [(day, o["minute"], a) for a in mon.tick(o["minute"], day)]; alerts += new
        hold = o['t'] if new else hold; a = agent.tick(o, alert=float(o['t'] - hold < 25)); f = a["_f"]; teach = sim.teacher(f)
        agent.feedback(sim.feedback(f, a["cont"], a["_snd"], a["_sp"], teach), teach, sim.welcome(f, PROACTIVE.index(a["proactive"])))
        if new: print(f"day {day} {o['minute']//60:02d}:{o['minute']%60:02d} ALERT {new[0][2].kind}: {new[0][2].msg} | PIP mood={a['mood']} sound={a['sound']} says='{a['speech']}'")
learner.close(); print("idle learner:", learner.stats, "| log size:", len(agent.log))
