"""Stewart-platform neck (6 rotary servos + rods), body yaw, antennas, servo dynamics.
Geometry comes from params.json, the same file the CAD generator reads."""
import json, pathlib, numpy as np
from .spec import LIMITS

PARAMS = json.loads((pathlib.Path(__file__).parent / "params.json").read_text())

def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

class StewartNeck:
    def __init__(s, p=PARAMS):
        s.p = p; d = np.deg2rad
        cb, cp_ = d([90, 210, 330]), d([30, 150, 270])
        sg = np.array([-1, 1] * 3)
        s.thb = np.repeat(cb, 2) + sg * d(p["base_pair_gap_deg"]) / 2
        g = d(p["platform_pair_gap_deg"]) / 2          # each base joint links to the nearest platform joint
        s.thp = np.array([d(30) + g, d(150) - g, d(150) + g, d(270) - g, d(270) + g, d(30) - g])
        s.beta = s.thb + sg * np.pi / 2                       # horn direction (tangential, mirror-symmetric)
        s.B = np.c_[p["base_radius"] * np.cos(s.thb), p["base_radius"] * np.sin(s.thb), np.zeros(6)]
        s.P = np.c_[p["platform_radius"] * np.cos(s.thp), p["platform_radius"] * np.sin(s.thp), np.zeros(6)]
        tip = s.B[:, :2] + p["horn_mm"] * np.c_[np.cos(s.beta), np.sin(s.beta)]
        dd = np.linalg.norm(s.P[:, :2] - tip, axis=1)
        s.h0 = float(np.sqrt(p["rod_mm"] ** 2 - dd ** 2).mean())   # home height (alpha = 0)
    def ik(s, pose):
        """pose = (x,y,z mm, roll,pitch,yaw deg) -> 6 servo angles (deg) or None if unreachable."""
        x, y, z, r, pi, ya = pose; R = rot(*np.deg2rad([r, pi, ya]))
        q = (R @ s.P.T).T + np.array([x, y, s.h0 + z]); l = q - s.B
        a, rod = s.p["horn_mm"], s.p["rod_mm"]
        L = (l ** 2).sum(1) - (rod ** 2 - a ** 2)
        M = 2 * a * l[:, 2]; N = 2 * a * (l[:, 0] * np.cos(s.beta) + l[:, 1] * np.sin(s.beta))
        ratio = L / np.hypot(M, N)
        if np.any(np.abs(ratio) > 1): return None
        return np.rad2deg(np.arcsin(ratio) - np.arctan2(N, M))
    def feasible(s, pose):
        ang = s.ik(pose); return ang is not None and np.all(np.abs(ang) <= s.p["servo_limit_deg"])
    def project(s, pose):
        """Scale a pose toward home until reachable (workspace projection)."""
        pose = np.asarray(pose, float)
        if s.feasible(pose): return pose, False
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if s.feasible(pose * mid) else (lo, mid)
        return pose * lo, True

class PipBody:
    """Slew-limited, lagged actuator model. Command: 9 normalised DOF in [-1,1]."""
    def __init__(s, tau=0.12):
        s.neck = StewartNeck(); s.x = np.zeros(9); s.tau = tau; s.servos = np.zeros(6); s.saturated = False
        s.target = np.zeros(9)
    def command(s, a): s.target = np.clip(np.asarray(a, float), -1, 1) * LIMITS
    def step(s, dt=1 / 50):
        s.x += (1 - np.exp(-dt / s.tau)) * (s.target - s.x)
        head, s.saturated = s.neck.project(s.x[:6]); s.x[:6] = head
        s.servos = s.neck.ik(head)
        return s.x.copy(), s.servos.copy()
    def settle(s, secs=1.0):
        for _ in range(int(secs * 50)): out = s.step()
        return out
