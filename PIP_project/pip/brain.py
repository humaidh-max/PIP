"""PIP's neural brain: emotion net (recurrent via fed-back state), expression policy + value net, and
proactivity-preference net. All numpy MLPs; update functions are pure so the idle learner can train shadow copies."""
import threading, numpy as np
from .nn import MLP, softmax
from .spec import *

def featurize(obs, last_reward=0.0, alert=0.0):
    m = obs["minute"] / 1440 * 2 * np.pi; v, f = obs["voice"], obs["face"]
    return np.array([v[0] if v else 0, v[1] if v else 0, f[0] if f else 0, f[1] if f else 0, np.sin(m), np.cos(m),
                     obs["present"], min(obs["idle_h"] / 12, 1), obs["pet"], obs["door"], obs["meal"], obs["noise"],
                     obs["visitor"], np.clip(last_reward, -1, 1), obs["kids_home"], alert], np.float32)

INIT_STATE = np.r_[0., 0., 1., np.zeros(NM - 1)].astype(np.float32)

def emo_update(net, B, train=True, lr=None):
    m = ~np.isnan(B["emo_t"][:, 0])
    if m.sum() < 4: return 0.0
    x = np.c_[B["f"][m], B["s_prev"][m]]; t = B["emo_t"][m]; o = net.forward(x, cache=train)
    v, a, p = np.tanh(o[:, :2]), None, softmax(o[:, 2:]); n = len(x)
    loss = float(((v - t[:, :2]) ** 2).mean() * 2 - (t[:, 2:] * np.log(p + 1e-8)).sum(1).mean() * .3)
    if train:
        d = np.c_[2 * (v - t[:, :2]) * (1 - v ** 2) / n, .3 * (p - t[:, 2:]) / n]; net.step(net.backward(d), lr)
    return loss

def pol_update(pol, val, B, train=True, lr=None, beta=0.25, sigma=0.2):
    """Advantage-weighted regression (off-policy => safe to run on replayed logs)."""
    x = np.c_[B["f"], B["s_new"]]; r = B["r"][:, None]; n = len(x)
    V = val.forward(x, cache=train); A = (r - V)[:, 0]
    w = np.clip(np.exp(A / beta), 0, 10); w = (w / (w.mean() + 1e-8))[:, None]
    o = pol.forward(x, cache=train); mu = o[:, :NC]; ps = softmax(o[:, NC:NC + 5]); pp = softmax(o[:, NC + 5:])
    ac, asn, asp = B["a_c"], B["a_s"].astype(int), B["a_sp"].astype(int); i = np.arange(n)
    loss = float((w * (mu - ac) ** 2).mean() + (w[:, 0] * -np.log(ps[i, asn] + 1e-8)).mean() * .3 +
                 (w[:, 0] * -np.log(pp[i, asp] + 1e-8)).mean() * .3 + ((r - V) ** 2).mean())
    if train:
        os_, op_ = ps.copy(), pp.copy(); os_[i, asn] -= 1; op_[i, asp] -= 1
        pol.step(pol.backward(np.c_[2 * w * (mu - ac) / n, .3 * w * os_ / n, .3 * w * op_ / n]), lr)
        val.step(val.backward(2 * (V - r) / n), lr)
    return loss

def pref_update(net, B, train=True, lr=None):
    m = ~np.isnan(B["y"]); 
    if m.sum() < 4: return 0.0
    x = np.c_[B["f"][m], np.eye(len(PROACTIVE))[B["pa"][m].astype(int)]]; y = B["y"][m]
    z = net.forward(x, cache=train)[:, 0]; p = 1 / (1 + np.exp(-z)); n = len(x)
    loss = float(-(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)).mean())
    if train: net.step(net.backward(((p - y) / n)[:, None]), lr)
    return loss

class PipBrain:
    def __init__(s, seed=0):
        s.emo = MLP([NF + NS, 48, 48, NS], seed, 2e-3); s.pol = MLP([NF + NS, 64, 64, NA], seed + 1, 1e-3)
        s.val = MLP([NF + NS, 32, 1], seed + 2, 2e-3); s.pref = MLP([NF + len(PROACTIVE), 24, 1], seed + 3, 3e-3)
        s.state = INIT_STATE.copy(); s.lock = threading.RLock(); s.rng = np.random.default_rng(seed); s.sigma = .12
    nets = property(lambda s: dict(emo=s.emo, pol=s.pol, val=s.val, pref=s.pref))
    def feel(s, f):
        """Update internal emotional state (valence, arousal, mood distribution)."""
        with s.lock:
            prev = s.state.copy(); o = s.emo.forward(np.r_[f, prev][None])[0]
            s.state = np.r_[np.tanh(o[:2]), softmax(o[2:])].astype(np.float32)
        return prev, s.state
    def act(s, f, explore=True):
        with s.lock:
            o = s.pol.forward(np.r_[f, s.state][None])[0]
            mu = o[:NC]; c = np.clip(mu + (s.rng.normal(0, s.sigma, NC) if explore else 0), -1, 1)
            ps, pp = softmax(o[NC:NC + 5]), softmax(o[NC + 5:])
            sn = int(s.rng.choice(5, p=ps)) if explore else int(ps.argmax()); sp = int(s.rng.choice(6, p=pp)) if explore else int(pp.argmax())
        return c, sn, sp
    def proactive(s, f, explore=True, thresh=0.5):
        with s.lock:
            z = s.pref.forward(np.c_[np.tile(f, (3, 1)), np.eye(len(PROACTIVE))[1:]])[:, 0]
        p = 1 / (1 + np.exp(-z))
        if explore and s.rng.random() < .1: return int(s.rng.integers(1, 4))
        k = int(p.argmax()); return k + 1 if p[k] > thresh else 0
    def mood(s): return MOODS[int(np.argmax(s.state[2:]))]
    def save(s, path):
        np.savez(path, **{f"{k}/{kk}": v for k, n in s.nets.items() for kk, v in n.state().items()})
    def load(s, path):
        d = np.load(path)
        for k, n in s.nets.items(): n.set_state({kk: d[f"{k}/{kk}"] for kk in n.state()})
