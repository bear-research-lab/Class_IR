"""
03_joint_vs_task_line.py  --  "Straight line" in joint space vs in task space (planar 2R arm).

Case A (joint limits): both endpoints are reachable, but the task-space line needs
         theta2 > 150 deg in the middle -> violates the joint limit.
         The joint-space line stays inside the (convex) box of joint limits.
Case B (singularity): the task-space line passes close to the base, where this arm
         (L1 = L2) is singular. With the same cubic time scaling, the joint speeds
         of the task-space line blow up, while the joint-space line stays tame.

Try this:
  1. Move the Case B line farther from the base (change Y_OFFSET). What happens to the peak joint speed?
  2. Set ANIMATE = True to watch the arm follow both lines.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traj_utils as tu

arm = tu.Planar2R(L1=1.0, L2=1.0)
deg = np.pi / 180
TH2_MAX = 150 * deg          # joint limits: 0 <= th1 <= 180 deg, 0 <= th2 <= 150 deg
T = 2.0                      # motion time for Case B
Y_OFFSET = 0.05              # Case B: how far the task-space line passes from the base
ANIMATE = False
SAVE = False

def task_line_in_joint_space(p_start, p_end, s):
    """IK of every point of a straight task-space line (elbow-down branch, unwrapped theta1)."""
    th = np.array([arm.ik(p_start + si*(p_end - p_start)) for si in s])
    th[:, 0] = np.unwrap(th[:, 0])
    return th

s = np.linspace(0, 1, 400)
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))

# ------------------- Case A: joint limits
thA0, thA1 = np.array([30, 130])*deg, np.array([150, 130])*deg
jA = tu.joint_line(thA0, thA1, s)
tA = task_line_in_joint_space(arm.fk(thA0), arm.fk(thA1), s)
ax = axs[0, 0]
ax.add_patch(plt.Rectangle((0, 0), 180, 150, fill=False, ls="--", color="g", label="joint limits"))
ax.plot(*(jA/deg).T, "b", lw=2, label="straight in joint space")
ax.plot(*(tA/deg).T, "orange", lw=2, ls="--", label="straight in task space")
bad = tA[:, 1] > TH2_MAX
ax.plot(*(tA[bad]/deg).T, "r", lw=4, label="violates theta2 <= 150 deg")
ax.set_xlabel("theta1 [deg]"); ax.set_ylabel("theta2 [deg]"); ax.set_title("Case A: joint space"); ax.legend(fontsize=8)
ax = axs[0, 1]
ax.plot(*arm.fk(jA).T, "b", lw=2); ax.plot(*arm.fk(tA).T, "orange", lw=2, ls="--")
for th in (thA0, thA1):
    e, p = arm.elbow(th), arm.fk(th); ax.plot([0, e[0], p[0]], [0, e[1], p[1]], "k-o", alpha=0.5)
ax.set_aspect("equal"); ax.set_title("Case A: task space"); ax.grid(alpha=0.3)
axs[0, 2].axis("off")
axs[0, 2].text(0, 0.5, f"Case A\n\nmax theta2 on task line = {tA[:, 1].max()/deg:.1f} deg (limit 150)\n"
               f"max theta2 on joint line = {jA[:, 1].max()/deg:.1f} deg\n\n"
               "Joint limits form a box (convex), so the\njoint-space line is always feasible.\n"
               "The reachable task space is not convex.", fontsize=11, va="center")

# ------------------- Case B: near the singularity at the base
pB0, pB1 = np.array([1.2, Y_OFFSET]), np.array([-1.2, Y_OFFSET])
thB0, thB1 = arm.ik(pB0), arm.ik(pB1)
thB1[0] = thB0[0] + np.angle(np.exp(1j*(thB1[0] - thB0[0])))       # take the short way round
t = np.linspace(0, T, 2000)
sc, sdc, _, _ = tu.cubic_time_scaling(T, t)
jB = tu.joint_line(thB0, thB1, sc)
tB = task_line_in_joint_space(pB0, pB1, sc)
jBd = np.gradient(jB, t, axis=0); tBd = np.gradient(tB, t, axis=0)
ax = axs[1, 0]
ax.plot(*arm.fk(jB).T, "b", lw=2, label="straight in joint space")
ax.plot(*arm.fk(tB).T, "orange", lw=2, ls="--", label="straight in task space")
ax.plot(0, 0, "ks", ms=8); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f"Case B: task space (line passes {Y_OFFSET} m from the base)")
ax = axs[1, 1]
for i, ls in ((0, "-"), (1, ":")):
    ax.plot(t, jBd[:, i]/deg, "b", ls=ls, label=f"joint line, joint {i+1}")
    ax.plot(t, tBd[:, i]/deg, "orange", ls=ls, label=f"task line, joint {i+1}")
ax.set_xlabel("t [s]"); ax.set_ylabel("joint speed [deg/s]"); ax.set_title("Case B: joint speeds, same cubic s(t)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
axs[1, 2].axis("off")
axs[1, 2].text(0, 0.5, f"Case B\n\npeak joint speed, joint-space line: {np.abs(jBd).max()/deg:7.1f} deg/s\n"
               f"peak joint speed, task-space line: {np.abs(tBd).max()/deg:7.1f} deg/s\n\n"
               "Near a singularity, the task-space line needs\nhuge joint speeds for almost any time scaling.", fontsize=11, va="center")
print(f"Case A: max theta2 on task-space line = {tA[:, 1].max()/deg:.1f} deg (limit 150 deg)")
print(f"Case B: peak joint speed  joint-line {np.abs(jBd).max()/deg:.1f} deg/s,  task-line {np.abs(tBd).max()/deg:.1f} deg/s")
plt.tight_layout()
if SAVE: plt.savefig("03_joint_vs_task_line.png", dpi=150)

if ANIMATE:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_aspect("equal"); ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-0.6, 2.2); ax2.grid(alpha=0.3)
    ax2.plot(*arm.fk(jB).T, "b", lw=1, alpha=0.4); ax2.plot(*arm.fk(tB).T, "orange", lw=1, alpha=0.4, ls="--")
    lj, = ax2.plot([], [], "b-o", lw=3, label="joint-space line"); lt, = ax2.plot([], [], "-o", color="orange", lw=3, label="task-space line")
    ax2.legend(loc="upper right"); title = ax2.set_title("")
    def upd(k):
        for line, th in ((lj, jB[k]), (lt, tB[k])):
            e, p = arm.elbow(th), arm.fk(th); line.set_data([0, e[0], p[0]], [0, e[1], p[1]])
        title.set_text(f"Case B,  t = {t[k]:.2f} s"); return lj, lt, title
    anim = FuncAnimation(fig2, upd, frames=range(0, len(t), 10), interval=20, blit=False)
plt.show()
