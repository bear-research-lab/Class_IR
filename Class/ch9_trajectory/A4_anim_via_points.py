"""
A4_anim_via_points.py  --  A 2R arm moves its gripper through four timed via points
(the data of MR Figs. 9.7-9.8, shifted into the arm's workspace).
  left : via velocities as in the book       (vias 2, 3: (1, 0) and (0, -1))
  right: heuristic via velocities             (average of the neighbouring segment velocities)
Both pass through every via at exactly the requested time (T1..T4 = 0, 1, 2, 3 s).
The joint angles come from inverse kinematics of the gripper position.

The gripper's arrow shows its acceleration: watch it jump when the gripper passes a via.

Try this:
  1. Change VEL_A and watch the loops change.  2. Change the via TIMES.
  3. Set GIF = "A4_via_points.gif" to save an animation.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from anim_utils import Arm2DArtist, draw_base_2d, run

TIMES = [0, 1, 2, 3]
POS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)
VEL_A = np.array([[0, 0], [1, 0], [0, -1], [0, 0]], float)
OFFSET, SCALE = np.array([0.6, 0.3]), 0.9          # place the via pattern in the workspace
FPS = 25
SLOWMO = 1                  # playback SLOWMO times slower than real time (e.g. 3)
GIF = None

arm = tu.Planar2R(1.0, 1.0)
P = OFFSET + SCALE * POS
cases = {"(a) given via velocities": SCALE * VEL_A,
         "(b) heuristic via velocities": tu.heuristic_via_velocities(TIMES, P)}
t = np.arange(0, TIMES[-1] + 1e-9, 1 / (FPS * SLOWMO))

def sample(V):
    tt, x, xd, xdd = tu.cubic_via(TIMES, P, V, n_per_seg=400)
    xi = np.column_stack([np.interp(t, tt, x[:, i]) for i in range(2)])
    ai = np.column_stack([np.interp(t, tt, xdd[:, i]) for i in range(2)])
    th = np.array([arm.ik(p) for p in xi]); th[:, 0] = np.unwrap(th[:, 0])
    return xi, ai, th, x

fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
data, arts, arrows = [], [], []
for ax, (name, V) in zip(axs, cases.items()):
    xi, ai, th, xfull = sample(V); data.append((xi, ai, th))
    ax.plot(xfull[:, 0], xfull[:, 1], ":", color="0.6")
    ax.plot(P[:, 0], P[:, 1], "ks", ms=8)
    for k, (p, Tk) in enumerate(zip(P, TIMES)): ax.text(p[0] + 0.06, p[1] + 0.06, f"via {k+1}\nt = {Tk} s", fontsize=9)
    for p, v in zip(P, V):
        if np.linalg.norm(v) > 0: ax.annotate("", xy=p + 0.25 * v, xytext=p, arrowprops=dict(arrowstyle="->", ls="--", color="0.4"))
    ax.set_xlim(-0.6, 2.2); ax.set_ylim(-0.4, 1.8); ax.set_aspect("equal"); ax.grid(alpha=0.2); draw_base_2d(ax)
    ax.set_title(name, fontsize=13)
    arts.append(Arm2DArtist(ax, arm, "tab:blue" if "given" in name else "tab:orange"))
    arrows.append(ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=2)))
title = fig.suptitle("")

def update(f):
    k = min(f, len(t) - 1); out = []
    for art, arr, (xi, ai, th) in zip(arts, arrows, data):
        if k == 0: art.reset()
        out += art.update(th[k]); p = xi[k]
        arr.xy = tuple(p + 0.08 * ai[k]); arr.set_position(tuple(p))
    title.set_text(f"Cubic via-point interpolation   t = {t[k]:.2f} s   (red arrow = gripper acceleration)")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.94))
run(fig, update, len(t) + FPS, 1000 / FPS, GIF, FPS)
