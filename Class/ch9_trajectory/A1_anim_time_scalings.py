"""
A1_anim_time_scalings.py  --  Watch four robots race along the SAME joint-space straight line
with four different time scalings (cubic, quintic, trapezoidal, S-curve) and the same T.

All four arms trace exactly the same path and arrive at the same time.
The dots behind each gripper are dropped every DT_DOT seconds ("stroboscope"):
  close dots = slow, far-apart dots = fast. Compare how each robot starts and stops.
The bottom plot shows sdot(t) with a moving time cursor.

SLOWMO slows down the PLAYBACK only (the robot's own times and speeds are unchanged):
SLOWMO = 4 means one second of robot motion takes four seconds on screen.

Try this:
  1. Change T. Everything just scales in time.
  2. Change VT (top speed * T) for the trapezoid / S-curve.
  3. Set GIF = "A1_time_scalings.gif" to save an animation for your slides.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from anim_utils import Arm2DArtist, draw_base_2d, run

T = 2.0
VT, R = 1.25, 0.5                     # trapezoid/S-curve top speed and S-curve jerk fraction
DT_DOT = 0.1                          # stroboscope dot spacing [s]
FPS = 25
SLOWMO = 4                            # playback is SLOWMO times slower than real time (1 = real time)
GIF = None                            # e.g. "A1_time_scalings.gif"

arm = tu.Planar2R(1.0, 1.0)
deg = np.pi / 180
th_a, th_b = np.array([-30, 110]) * deg, np.array([150, 40]) * deg

t = np.arange(0, T + 1e-9, 1 / (FPS * SLOWMO))     # robot time of each animation frame
v, a = tu.trapezoid_from_T(T, VT)
prof = {"Cubic": tu.cubic_time_scaling(T, t), "Quintic": tu.quintic_time_scaling(T, t),
        "Trapezoidal": tu.trapezoidal_time_scaling(v, a, t)[:4], "S-curve": tu.scurve_time_scaling(T, t, VT, R)}
colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1])
artists, dots = [], []
for k, (name, c) in enumerate(zip(prof, colors)):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(-2.0, 1.5); ax.set_ylim(-0.3, 2.0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    path = arm.fk(tu.joint_line(th_a, th_b, np.linspace(0, 1, 200)))
    ax.plot(path[:, 0], path[:, 1], ":", color="0.6")
    draw_base_2d(ax); ax.set_title(name, color=c, fontsize=14, weight="bold")
    artists.append(Arm2DArtist(ax, arm, color=c, trail=False))
    dots.append(ax.plot([], [], "o", color=c, ms=5, alpha=0.7)[0])
axv = fig.add_subplot(gs[1, :])
for (name, (s, sd, _, _)), c in zip(prof.items(), colors):
    tt = np.linspace(0, T, 600)
    sd_fine = {"Cubic": tu.cubic_time_scaling, "Quintic": tu.quintic_time_scaling}.get(name)
    if sd_fine: y = sd_fine(T, tt)[1]
    elif name == "Trapezoidal": y = tu.trapezoidal_time_scaling(v, a, tt)[1]
    else: y = tu.scurve_time_scaling(T, tt, VT, R)[1]
    axv.plot(tt, y, color=c, lw=2, label=name)
cursor = axv.axvline(0, color="k", lw=1.5)
markers = [axv.plot([], [], "o", color=c, ms=8)[0] for c in colors]
axv.set_xlabel("t [s]"); axv.set_ylabel("speed along the path  sdot"); axv.legend(ncol=4, loc="upper right"); axv.grid(alpha=0.3)
title = fig.suptitle("")
every = max(1, int(round(DT_DOT * FPS * SLOWMO)))
frames = [0] * (FPS // 2) + list(range(len(t))) + [len(t) - 1] * FPS    # pause 0.5 s at the start, 1 s at the end

def update(f):
    k = min(f, len(t) - 1)
    out = []
    for (name, (s, sd, _, _)), art, dl, mk in zip(prof.items(), artists, dots, markers):
        th = tu.joint_line(th_a, th_b, [s[k]])[0]
        out += art.update(th)
        idx = np.arange(0, k + 1, every); P = arm.fk(tu.joint_line(th_a, th_b, s[idx]))
        dl.set_data(P[:, 0], P[:, 1]); mk.set_data([t[k]], [sd[k]])
    cursor.set_xdata([t[k], t[k]]); slow = f"   (playback {SLOWMO}x slower)" if SLOWMO != 1 else ""
    title.set_text(f"Same path, same T = {T} s, different s(t)      t = {t[k]:.2f} s{slow}")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.94))
run(fig, update, frames, 1000 / FPS, GIF, FPS)
