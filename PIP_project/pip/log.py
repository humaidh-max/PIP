"""Private on-device experience log (ring buffer, persisted to a local .npz; nothing leaves the device)."""
import numpy as np, threading
from .spec import *
FIELDS = dict(f=NF, s_prev=NS, s_new=NS, emo_t=NS, a_c=NC, a_s=1, a_sp=1, r=1, pa=1, y=1)

class ExperienceLog:
    def __init__(s, cap=50000):
        s.cap, s.n, s.i = cap, 0, 0; s.d = {k: np.full((cap, w), np.nan, np.float32) for k, w in FIELDS.items()}
        s.lock = threading.Lock()
    def __len__(s): return s.n
    def add(s, **kw):
        with s.lock:
            for k, w in FIELDS.items(): s.d[k][s.i] = np.nan if k not in kw else np.asarray(kw[k], np.float32).reshape(w)
            s.i = (s.i + 1) % s.cap; s.n = min(s.n + 1, s.cap)
    def idx_recent(s, k):  # most recent k indices
        return (s.i - 1 - np.arange(min(k, s.n))) % s.cap
    def sample(s, n, recency=0.5, rng=np.random):
        """Mix of uniform replay (stability) and recent replay (adaptation to a changing household)."""
        with s.lock:
            nr = int(n * recency); rec = s.idx_recent(max(n * 20, 1))
            idx = np.r_[rng.choice(rec, nr) if nr else [], rng.integers(0, s.n, n - nr)].astype(int) if s.n < s.cap else \
                  np.r_[rng.choice(rec, nr) if nr else [], (s.i + rng.integers(0, s.n, n - nr)) % s.cap].astype(int)
            B = {k: v[idx].copy() for k, v in s.d.items()}
        B["a_s"], B["a_sp"], B["pa"] = B["a_s"][:, 0], B["a_sp"][:, 0], np.nan_to_num(B["pa"][:, 0]); B["r"], B["y"] = B["r"][:, 0], B["y"][:, 0]
        B["emo_t"] = B["emo_t"]; return B
    def save(s, path):
        with s.lock: np.savez_compressed(path, n=s.n, i=s.i, **s.d)
    def load(s, path):
        z = np.load(path); s.n, s.i = int(z["n"]), int(z["i"])
        for k in s.d: s.d[k][...] = z[k]
