"""Tiny numpy MLP with manual backprop + Adam. No frameworks: runs anywhere."""
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

class MLP:
    def __init__(s, sizes, seed=0, lr=2e-3):
        r = np.random.default_rng(seed)
        s.W = [r.normal(0, np.sqrt(2 / (i + o)), (i, o)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
        s.b = [np.zeros(o, np.float32) for o in sizes[1:]]
        s.lr, s.t = lr, 0
        s.m = [np.zeros_like(p) for p in s.W + s.b]; s.v = [np.zeros_like(p) for p in s.W + s.b]
    def forward(s, x, cache=False):
        a = [np.asarray(x, np.float32)]
        for i, (W, b) in enumerate(zip(s.W, s.b)):
            z = a[-1] @ W + b
            a.append(np.tanh(z) if i < len(s.W) - 1 else z)
        if cache: s._a = a
        return a[-1]
    def backward(s, d):
        gW, gb = [], []
        for i in reversed(range(len(s.W))):
            gW.insert(0, s._a[i].T @ d); gb.insert(0, d.sum(0))
            if i: d = (d @ s.W[i].T) * (1 - s._a[i] ** 2)
        return gW + gb
    def step(s, grads, lr=None, clip=1.0):
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads)); k = min(1.0, clip / (gn + 1e-8))
        s.t += 1; lr = lr or s.lr; P = s.W + s.b
        for i, (p, g) in enumerate(zip(P, grads)):
            g = g * k; s.m[i] = .9 * s.m[i] + .1 * g; s.v[i] = .999 * s.v[i] + .001 * g * g
            p -= lr * (s.m[i] / (1 - .9 ** s.t)) / (np.sqrt(s.v[i] / (1 - .999 ** s.t)) + 1e-8)
    def clone(s):
        n = MLP.__new__(MLP); n.lr, n.t = s.lr, s.t
        n.W = [w.copy() for w in s.W]; n.b = [b.copy() for b in s.b]
        n.m = [m.copy() for m in s.m]; n.v = [v.copy() for v in s.v]; return n
    def load_from(s, o):
        for a, b in zip(s.W + s.b, o.W + o.b): a[...] = b
    def state(s): return {f"W{i}": w for i, w in enumerate(s.W)} | {f"b{i}": b for i, b in enumerate(s.b)}
    def set_state(s, d):
        for i in range(len(s.W)): s.W[i][...] = d[f"W{i}"]; s.b[i][...] = d[f"b{i}"]
