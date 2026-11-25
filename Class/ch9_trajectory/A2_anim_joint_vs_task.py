"""
A2_anim_joint_vs_task.py  --  Same start and goal, two "straight lines":
   blue   = straight line in JOINT space (theta1, theta2 change linearly)
   orange = straight line in TASK space (the gripper moves on a straight line; joints from IK)
Both use the same cubic time scaling and the same T.

CASE = "A": both endpoints are reachable, but the task-space line needs theta2 > 150 deg
            in the middle (joint limit). The arm turns RED while it violates the limit.
CASE = "B": the task-space line passes close to the base, where this arm is singular.
            Watch the orange arm whip around: joint speeds become enormous.
The right panel shows both motions in joint space (the joint-limit box is dashed).

Try this:
  1. Run both cases.  2. In case B, change Y_OFFSET (distance of the line from the base).
  3. Set GIF = "A2_case_B.gif" to save an animation.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from anim_utils import Arm2DArtist, draw_base_2d, run

CASE = "B"
T, FPS = 3.0, 25
Y_OFFSET = 0.05
SLOWMO = 1                  # playback SLOWMO times slower than real time (e.g. 3)
GIF = None

arm = tu.Planar2R(1.0, 1.0)
deg = np.pi / 180
TH2_MAX = 150 * deg

if CASE == "A":
    th_a, th_b = np.array([30, 130]) * deg, np.array([150, 130]) * deg
    p_a, p_b = arm.fk(th_a), arm.fk(th_b)
else:
    p_a, p_b = np.array([1.2, Y_OFFSET]), np.array([-1.2, Y_OFFSET])
    th_a, th_b = arm.ik(p_a), arm.ik(p_b)
    th_b[0] = th_a[0] + np.angle(np.exp(1j * (th_b[0] - th_a[0])))

t = np.arange(0, T + 1e-9, 1 / (FPS * SLOWMO))
s = tu.cubic_time_scaling(T, t)[0]
th_joint = tu.joint_line(th_a, th_b, s)
th_task = np.array([arm.ik(p_a + si * (p_b - p_a)) for si in s]); th_task[:, 0] = np.unwrap(th_task[:, 0])
# joint speeds on a fine time grid (the animation frames are too coarse to differentiate)
tf = np.linspace(0, T, 6001); sf = tu.cubic_time_scaling(T, tf)[0]
thf_task = np.array([arm.ik(p_a + si * (p_b - p_a)) for si in sf]); thf_task[:, 0] = np.unwrap(thf_task[:, 0])
sp_joint = np.interp(t, tf, np.abs(np.gradient(tu.joint_line(th_a, th_b, sf), tf, axis=0)).max(1)) / deg
sp_task = np.interp(t, tf, np.abs(np.gradient(thf_task, tf, axis=0)).max(1)) / deg
print(f"peak joint speed: joint-space line {sp_joint.max():.0f} deg/s, task-space line {np.abs(np.gradient(thf_task, tf, axis=0)).max() / deg:.0f} deg/s")
LIMITS = (CASE == "A")          # the joint-limit box is part of case A only

fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
P = np.vstack([arm.fk(th_joint), arm.fk(th_task), arm.elbow(th_joint), arm.elbow(th_task), [[0, 0]]])
lo, hi = P.min(0) - 0.35, P.max(0) + 0.35
for ax, ttl in zip(axs[:2], ["Straight line in JOINT space", "Straight line in TASK space"]):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1] + 0.3); ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_title(ttl, fontsize=13)
    draw_base_2d(ax); ax.plot(*np.array([p_a, p_b]).T, "k*", ms=12)
full = np.linspace(0, 1, 200)
axs[0].plot(*arm.fk(tu.joint_line(th_a, th_b, full)).T, ":", color="tab:blue")
axs[1].plot(*np.array([p_a, p_b]).T, ":", color="tab:orange")
A_j = Arm2DArtist(axs[0], arm, "tab:blue"); A_t = Arm2DArtist(axs[1], arm, "tab:orange")
txt_j = axs[0].text(0.03, 0.97, "", fontsize=11, va="top", transform=axs[0].transAxes); txt_t = axs[1].text(0.03, 0.97, "", fontsize=11, va="top", transform=axs[1].transAxes)
ax = axs[2]
if LIMITS:
    ax.add_patch(plt.Rectangle((0, 0), 180, 150, fill=False, ls="--", color="g")); ax.text(2, 152, "joint limits", color="g")
ax.plot(*(th_joint / deg).T, color="tab:blue", lw=1.5); ax.plot(*(th_task / deg).T, color="tab:orange", lw=1.5)
dj, = ax.plot([], [], "o", color="tab:blue", ms=10); dt_, = ax.plot([], [], "o", color="tab:orange", ms=10)
ax.set_xlabel("theta1 [deg]"); ax.set_ylabel("theta2 [deg]"); ax.set_title("Joint space", fontsize=13); ax.grid(alpha=0.3)
allth = np.vstack([th_joint, th_task]) / deg
ax.set_xlim(allth[:, 0].min() - 15, allth[:, 0].max() + 15); ax.set_ylim(allth[:, 1].min() - 15, allth[:, 1].max() + 15)
if LIMITS: ax.set_xlim(-10, 190); ax.set_ylim(-10, 185)
title = fig.suptitle("")

def update(f):
    k = min(f, len(t) - 1)
    if k == 0: A_j.reset(); A_t.reset()
    out = A_j.update(th_joint[k]) + A_t.update(th_task[k])
    bad = LIMITS and th_task[k, 1] > TH2_MAX
    A_t.line.set_color("red" if bad else "tab:orange")
    txt_j.set_text(f"max joint speed {sp_joint[k]:6.0f} deg/s")
    txt_t.set_text(f"max joint speed {sp_task[k]:6.0f} deg/s" + ("\nVIOLATES theta2 <= 150 deg" if bad else ""))
    txt_t.set_color("red" if bad else "k")
    dj.set_data([th_joint[k, 0] / deg], [th_joint[k, 1] / deg]); dt_.set_data([th_task[k, 0] / deg], [th_task[k, 1] / deg])
    title.set_text(f"Case {CASE}:  cubic time scaling, T = {T} s,  t = {t[k]:.2f} s")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.94))
run(fig, update, len(t) + FPS, 1000 / FPS, GIF, FPS)
