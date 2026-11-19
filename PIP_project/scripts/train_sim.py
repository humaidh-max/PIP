"""Pretrain PIP in the virtual home, then show background-learner adaptation to a changed user."""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pip.sim import HomeSim
from pip.agent import PipAgent
from pip.brain import emo_update, pol_update, pref_update
from pip.learner import BackgroundLearner, IdleGate

def run_day(sim, agent, learn=0, explore=True, rng=None):
    R, E = [], []; pulses = rng.integers(0, 1400, 2) if rng is not None else [300, 900]
    for _ in range(1440):
        obs = sim.step(); al = float(any(0 <= obs['minute'] - p < 25 for p in pulses)); a = agent.tick(obs, alert=al, explore=explore); f = a["_f"]
        teach = sim.teach_for = sim.teacher(f); r = sim.feedback(f, a["cont"], a["_snd"], a["_sp"], teach)
        pa = PROACTIVE_IDX[a["proactive"]]; y = sim.welcome(f, pa)
        agent.feedback(r, teach, y); R.append(r); E.append(abs(a["valence"] - teach[0]))
        if learn and len(agent.log) > 512 and sim.t % 8 == 0:
            for _ in range(learn):
                B = agent.log.sample(128, .3, rng); emo_update(agent.brain.emo, B); pol_update(agent.brain.pol, agent.brain.val, B); pref_update(agent.brain.pref, B)
    return float(np.mean(R)), float(np.mean(E))
from pip.spec import PROACTIVE; PROACTIVE_IDX = {p: i for i, p in enumerate(PROACTIVE)}

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(0); sim = HomeSim(1); agent = PipAgent(); t0 = time.time()
    print("untrained eval reward: %.3f" % run_day(HomeSim(9), agent, 0, False)[0])
    for d in range(days):
        r, e = run_day(sim, agent, learn=1, rng=rng)
        if d % 5 == 4: print(f"day {d+1:3d}  train reward {r:6.3f}  emotion |valence err| {e:.3f}  ({time.time()-t0:.0f}s)")
    r, e = run_day(HomeSim(9), agent, 0, False); print(f"trained eval reward: {r:.3f}   valence err {e:.3f}")
    agent.brain.save("pip_brain.npz"); agent.log.save("pip_log.npz")
    # --- household change: user now prefers calmer, smaller expressions. Only the idle learner adapts PIP. ---
    sim.style = 0.4; r0, _ = run_day(sim, agent, 0, False); print(f"\nuser style changed -> reward drops to {r0:.3f}")
    gate = IdleGate(); gate.set_busy(False); L = BackgroundLearner(agent.brain, agent.log, gate, duty=0.5)
    for k in range(6):
        run_day(sim, agent, 0, True)                              # PIP lives normally (logging, no inline training)
        for _ in range(5): L.run_round()                          # idle window: replay-based fine-tune
        r1, _ = run_day(sim, agent, 0, False); print(f"after idle-learning night {k+1}: reward {r1:.3f}  {L.stats}")
