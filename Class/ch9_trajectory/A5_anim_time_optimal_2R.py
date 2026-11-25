"""
A5_anim_time_optimal_2R.py  --  Two identical 2R arms (vertical plane, gravity pulls down)
move along the SAME joint-space straight line under the SAME torque limits:
  left : time-optimal (bang-bang) time scaling
  right: the fastest cubic time scaling that respects the torque limits
The bars show each joint torque as a fraction of its limit. A bar turns red when the
actuator is saturated (|tau| = limit). Notice:
  * The time-optimal arm keeps at least one actuator saturated the whole time.
  * It finishes in about half the time, then waits.

Try this:
  1. Change TAU_MAX (e.g. [30, 11] or [30, 10]) and compare the two times.
  2. Set GIF = "A5_time_optimal.gif" to save an animation.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from time_optimal_2r import PathProblem
from anim_utils import Arm2DArtist, run

deg = np.pi / 180
TH_START, TH_END = np.array([-90, 0]) * deg, np.array([90, 150]) * deg
TAU_MAX = np.array([30.0, 12.0])
FPS = 25
SLOWMO = 1                  # playback SLOWMO times slower than real time (e.g. 3)
GIF = None

arm = tu.Planar2R(1.0, 1.0, 1.0, 1.0)
prob = PathProblem(arm, TH_START, TH_END, TAU_MAX)
t_opt, S, sd_opt, sdd_opt, s_switch = prob.time_optimal()
T_opt = t_opt[-1]; T_cub = prob.fastest_cubic()
print(f"time-optimal T = {T_opt:.2f} s (switch at s* = {s_switch:.2f}),  fastest cubic T = {T_cub:.2f} s")

T_show = max(T_opt, T_cub); t = np.arange(0, T_show + 1e-9, 1 / (FPS * SLOWMO))
def opt_state(tk):
    tk = min(tk, T_opt); return [np.interp(tk, t_opt, x) for x in (S, sd_opt, sdd_opt)]
def cub_state(tk):
    tk = min(tk, T_cub); s, sd, sdd, _ = tu.cubic_time_scaling(T_cub, tk)
    if tk >= T_cub: sd, sdd = 0.0, 0.0
    return [float(s), float(sd), float(sdd)]

fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1])
arms, bars, labels = [], [], []
for k, (name, c) in enumerate([("Time-optimal (bang-bang)", "k"), ("Fastest feasible cubic", "tab:green")]):
    ax = fig.add_subplot(gs[0, k]); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.plot(*arm.fk(tu.joint_line(TH_START, TH_END, np.linspace(0, 1, 200))).T, ":", color="0.6")
    ax.add_patch(plt.Rectangle((-0.15, -0.1), 0.3, 0.1, color="0.25")); ax.annotate("g", xy=(1.9, -1.9), xytext=(1.9, -1.3), arrowprops=dict(arrowstyle="->"))
    ax.set_title(name, fontsize=13)
    arms.append(Arm2DArtist(ax, arm, c if c != "k" else "0.2"))
    labels.append(ax.text(-2.1, 2.05, "", fontsize=12, va="top"))
    axb = fig.add_subplot(gs[1, k]); axb.set_xlim(-1.05, 1.05); axb.set_yticks([0, 1]); axb.set_yticklabels(["joint 1", "joint 2"])
    axb.axvline(-1, color="r", ls=":"); axb.axvline(1, color="r", ls=":"); axb.axvline(0, color="k", lw=0.8)
    axb.set_xlabel("torque / limit"); bars.append(axb.barh([0, 1], [0, 0], color="tab:blue"))
title = fig.suptitle("")

def update(f):
    tk = t[min(f, len(t) - 1)]; out = []
    for art, bar, lab, (state, T) in zip(arms, bars, labels, ((opt_state(tk), T_opt), (cub_state(tk), T_cub))):
        s, sd, sdd = state
        if f == 0: art.reset()
        out += art.update(prob.theta(s))
        tau = prob.torque(s, sd, sdd); frac = tau / TAU_MAX
        for b, fr in zip(bar, frac):
            b.set_width(fr); b.set_color("red" if abs(fr) > 0.995 else "tab:blue")
        lab.set_text(f"T = {T:.2f} s" + ("   done" if tk >= T else ""))
    title.set_text(f"Same path, same torque limits  |tau| <= {TAU_MAX.tolist()} N m      t = {tk:.2f} s")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.94))
run(fig, update, len(t) + FPS, 1000 / FPS, GIF, FPS)
