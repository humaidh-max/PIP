"""Parametric PIP CAD (CadQuery). Reads ../pip/params.json -- the SAME file the kinematics/simulator use,
so sim and hardware cannot drift apart. Outputs STEP (import into OnShape) + hardpoints.csv.
Run:  pip install cadquery && python cad/pip_cad.py"""
import sys, json, pathlib, numpy as np, cadquery as cq
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from pip.kinematics import StewartNeck, PARAMS as P
out = root / "cad" / "out"; out.mkdir(exist_ok=True); N = StewartNeck()
deg = np.rad2deg; T = P["plate_thickness"]; Z0 = P["body_height"] + P["turntable_thickness"]   # stewart base plane

def base_shell():      # outer body: speaker bay (front, +Y) + cable slot (rear)
    b = cq.Workplane("XY").circle(P["body_radius"]).circle(P["body_radius"] - 3).extrude(P["body_height"])
    b = b.union(cq.Workplane("XY").circle(P["body_radius"]).extrude(3))                       # floor
    cz = P["body_height"] / 2
    ring = cq.Workplane("XZ").workplane(offset=-(P["body_radius"] + 1)).center(0, cz).polarArray(P["speaker_d"] / 3, 0, 360, 6).circle(1.6).extrude(8)
    slot = cq.Workplane("XZ").workplane(offset=P["body_radius"] - 6).center(0, 14).rect(14, 8).extrude(-8)
    return b.cut(ring).cut(slot)

def turntable():       # body-yaw stage: bearing seat + servo pocket + central cable bore
    t = cq.Workplane("XY").circle(P["body_radius"] - 4).extrude(P["turntable_thickness"]).faces(">Z").workplane().circle(8).cutThruAll()
    t = t.faces(">Z").workplane().pushPoints([(P["body_radius"] - 14, 0)]).rect(P["servo_l"], P["servo_w"]).cutBlind(-3)
    return t.faces(">Z").workplane().polarArray(P["body_radius"] - 12, 0, 360, 6).hole(2.5)

def stewart_base():    # 6 servo pockets placed AT the kinematic base joints
    w = cq.Workplane("XY").circle(P["base_radius"] + 14).extrude(T).faces(">Z").workplane().circle(9).cutThruAll()
    for i in range(6):
        x, y = N.B[i, 0], N.B[i, 1]
        pocket = cq.Workplane("XY").rect(P["servo_l"], P["servo_w"]).extrude(T + 1).rotate((0, 0, 0), (0, 0, 1), deg(N.beta[i])).translate((x * 1.0, y * 1.0, -.5))
        w = w.cut(pocket)
    return w

def horn():            # servo horn, length = horn_mm between shaft and ball-stud
    return cq.Workplane("XY").slot2D(P["horn_mm"] + 8, 7, 0).extrude(2.5).translate((P["horn_mm"] / 2, 0, 0)).faces(">Z").workplane().pushPoints([(0, 0)]).hole(3).faces(">Z").workplane().pushPoints([(P["horn_mm"], 0)]).hole(1.6)

def rod():
    return cq.Workplane("XY").circle(1.2).extrude(P["rod_mm"])

def top_plate():       # head platform: ball-stud holes at platform joints, camera, 2-mic array, antenna bosses
    w = cq.Workplane("XY").circle(P["platform_radius"] + 8).extrude(T)
    w = w.faces(">Z").workplane().pushPoints([(N.P[i, 0], N.P[i, 1]) for i in range(6)]).hole(2.4)
    cam = cq.Workplane("XY").center(0, P["platform_radius"] - 6).rect(P["camera_w"], P["camera_h"]).extrude(T + 1).translate((0, 0, -.5))
    mics = cq.Workplane("XY").pushPoints([(-P["mic_spacing"] / 2, P["platform_radius"] - 14), (P["mic_spacing"] / 2, P["platform_radius"] - 14)]).circle(P["mic_hole_d"] / 2).extrude(T + 1).translate((0, 0, -.5))
    boss = cq.Workplane("XY").pushPoints([(-P["antenna_spacing"] / 2, -6), (P["antenna_spacing"] / 2, -6)]).circle(5).extrude(T + 10)
    return w.cut(cam).cut(mics).union(boss)

def antenna():
    return cq.Workplane("XY").circle(1.0).extrude(P["antenna_len"]).union(cq.Workplane("XY").sphere(2.2).translate((0, 0, P["antenna_len"])))

parts = {"base_shell": base_shell(), "turntable": turntable(), "stewart_base": stewart_base(), "horn": horn(), "rod": rod(),
         "top_plate": top_plate(), "antenna": antenna()}
for n, p in parts.items(): cq.exporters.export(p, str(out / f"{n}.step"))
# ---- assembly at home pose (alpha = 0) ----
asm = cq.Assembly(name="PIP")
asm.add(parts["base_shell"], name="base_shell", color=cq.Color(.9, .9, .9))
asm.add(parts["turntable"], name="turntable", loc=cq.Location((0, 0, P["body_height"])), color=cq.Color(.3, .3, .3))
asm.add(parts["stewart_base"], name="stewart_base", loc=cq.Location((0, 0, Z0)), color=cq.Color(.2, .2, .2))
rows = ["part,i,x,y,z"]
for i in range(6):
    tip = N.B[i, :2] + P["horn_mm"] * np.array([np.cos(N.beta[i]), np.sin(N.beta[i])])
    asm.add(parts["horn"], name=f"horn_{i}", loc=cq.Location(cq.Vector(*N.B[i, :2], Z0 + T + 1), cq.Vector(0, 0, 1), deg(N.beta[i])))
    a, b = np.array([*tip, Z0 + T + 1]), np.array([*N.P[i, :2], Z0 + N.h0 + T])
    v = b - a; ang = float(np.degrees(np.arccos(v[2] / np.linalg.norm(v)))); ax = np.cross([0, 0, 1], v)
    asm.add(parts["rod"], name=f"rod_{i}", loc=cq.Location(cq.Vector(*a), cq.Vector(*(ax if np.linalg.norm(ax) > 1e-9 else [1, 0, 0])), ang))
    rows += [f"base_joint,{i},{N.B[i,0]:.3f},{N.B[i,1]:.3f},0", f"horn_tip,{i},{tip[0]:.3f},{tip[1]:.3f},0", f"platform_joint,{i},{N.P[i,0]:.3f},{N.P[i,1]:.3f},{N.h0:.3f}"]
asm.add(parts["top_plate"], name="top_plate", loc=cq.Location((0, 0, Z0 + N.h0 + T)), color=cq.Color(.95, .95, .95))
for sgn, nm in ((-1, "antenna_L"), (1, "antenna_R")):
    asm.add(parts["antenna"], name=nm, loc=cq.Location((sgn * P["antenna_spacing"] / 2, -6, Z0 + N.h0 + 2 * T + 10)))
asm.save(str(out / "PIP_assembly.step"))
(out / "hardpoints.csv").write_text("\n".join(rows)); print("home head height h0 = %.2f mm" % N.h0, "| exported:", ", ".join(sorted(p.name for p in out.iterdir())))
