"""
A3_anim_screw_vs_decoupled_UR5.py  --  A UR5 (6R) moves its gripper between the same two poses
in three different ways, all with the same quintic time scaling:

   left   : straight line in JOINT space  theta(s) = theta_start + s (theta_end - theta_start)
   middle : SCREW path      X(s) = X_start exp(log(X_start^-1 X_end) s)
   right  : DECOUPLED path  p(s) on a straight line, R(s) = R_start exp(log(R_start^T R_end) s)

For the screw and decoupled paths, the joint angles come from numerical IK (Newton-Raphson, Ch. 6),
warm-started from the previous solution so the motion is smooth.

What to look for:
  * The gripper ORIENTATION changes in exactly the same way on the screw and decoupled paths.
  * Only the gripper's POSITION path differs: an arc (screw) vs a straight line (decoupled).
  * The joint-space line gives yet another gripper path and orientation history.
  * The dotted black line is the straight line between the start and end positions.
  * Purple line = INSTANTANEOUS SCREW AXIS of the gripper's motion (from the spatial twist V_s).
      screw path    : the axis never moves (one single screw motion)
      decoupled path: same direction and same pitch h as the screw path, but the axis SLIDES sideways
      joint line    : the axis moves, tilts slightly, and its pitch changes
    (Pitch h = omega.v / |omega|^2: how far the gripper advances along the axis per radian of rotation.
     The end pose is 0.2 m higher than the start, so h is not zero.)
  * Bottom row: the six joint angles. Only the joint-space line moves the joints "in a straight line"
    (every joint follows the same quintic s(t)); the other two need IK and curved joint motions.

Try this:
  1. ROT_DEG = 0: the screw and decoupled paths become identical. Why?
  2. ROT_DEG = 170: the screw path bulges even more.
  3. Make P_END the same height as P_START: the pitch becomes zero (a planar motion is a pure rotation).
  4. Set GIF = "A3_screw_vs_decoupled.gif" to save an animation for your slides.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from robots import UR5
from anim_utils import UR5Artist, setup_3d, run

ROT_DEG = 150                                       # rotation of the gripper about the vertical axis
P_START, P_END = np.array([0.4, -0.3, 0.25]), np.array([0.4, 0.3, 0.45])
T, FPS = 3.0, 20
SLOWMO = 1                  # playback SLOWMO times slower than real time (e.g. 3)
GIF = None

robot = UR5()
theta_ready = np.array([0.0, -1.4, 1.6, -1.8, -1.57, 0.0])
R0 = robot.fk(theta_ready)[:3, :3]                  # gripper pointing down
X_start = tu.RpToTrans(R0, P_START)
X_end = tu.RpToTrans(tu.rot([0, 0, 1], np.radians(ROT_DEG)) @ R0, P_END)

th_start, ok1 = robot.ik(X_start, theta_ready)
th_end, ok2 = robot.ik(X_end, th_start)
assert ok1 and ok2, "IK failed for the start or end pose"

t = np.arange(0, T + 1e-9, 1 / (FPS * SLOWMO))
s = tu.quintic_time_scaling(T, t)[0]
th_joint = tu.joint_line(th_start, th_end, s)
th_screw, ok_s = robot.ik_along(tu.screw_path(X_start, X_end, s), th_start)
th_dec, ok_d = robot.ik_along(tu.decoupled_path(X_start, X_end, s), th_start)
print("IK converged along the whole path:  screw", ok_s, "  decoupled", ok_d)
for name, th in [("joint line", th_joint), ("screw", th_screw), ("decoupled", th_dec)]:
    P = np.array([robot.fk(q)[:3, 3] for q in th])
    print(f"{name:<11}: gripper path length {np.linalg.norm(np.diff(P, axis=0), axis=1).sum():.3f} m"
          f"   (straight-line distance {np.linalg.norm(P_END - P_START):.3f} m)")

# ---------------- instantaneous screw axis of the gripper motion
# spatial twist per unit s:  [V_s] ds = log( X(s+ds) X(s)^-1 );  V_s = (omega, v)
# axis direction omega/|omega|, closest point to the origin q = omega x v / |omega|^2, pitch h = omega.v / |omega|^2
pose_fns = [lambda si: robot.fk(th_start + si * (th_end - th_start)),
            lambda si: tu.screw_path(X_start, X_end, [si])[0],
            lambda si: tu.decoupled_path(X_start, X_end, [si])[0]]

def screw_axis(pose, si, eps=1e-5):
    a, b = (si, si + eps) if si + eps <= 1 else (si - eps, si)
    V = np.concatenate([tu.unskew((m := tu.log6(pose(b) @ tu.trans_inv(pose(a))))[:3, :3]), m[:3, 3]]) / eps
    w, v = V[:3], V[3:]; nw = np.linalg.norm(w)
    if nw < 1e-9: return None, None, None          # pure translation: axis at infinity
    return w / nw, np.cross(w, v) / nw**2, w @ v / nw**2

axes_data = [[screw_axis(fn, si) for si in s] for fn in pose_fns]
C_MID = 0.5 * (P_START + P_END)

fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
artists, axis_lines, axis_txt, cursors, jdots = [], [], [], [], []
JCOL = ["tab:red", "tab:olive", "tab:cyan", "tab:brown", "tab:pink", "tab:gray"]
for k, (name, color) in enumerate([("Straight line in joint space", "tab:green"), ("Screw path", "tab:blue"), ("Decoupled path", "tab:orange")]):
    ax = fig.add_subplot(gs[0, k], projection="3d")
    setup_3d(ax, center=(0.25, 0.0, 0.28), r=0.42, elev=30, azim=-75)
    ax.plot(*np.array([P_START, P_END]).T, "k:", lw=1.5)
    ax.plot([-0.08, 0.08, 0.08, -0.08, -0.08], [-0.08, -0.08, 0.08, 0.08, -0.08], [0] * 5, color="0.3")   # base plate
    ax.set_title(name, color=color, fontsize=13, weight="bold")
    artists.append(UR5Artist(ax, robot, trail_color=color))
    axis_lines.append(ax.plot([], [], [], "-", color="purple", lw=2.5, label="instantaneous screw axis")[0])
    axis_txt.append(ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, color="purple"))
    if k == 0: ax.legend(loc="upper left", fontsize=9)
paths = [th_joint, th_screw, th_dec]
jmin, jmax = min(p.min() for p in paths) * 180 / np.pi, max(p.max() for p in paths) * 180 / np.pi
for k, th in enumerate(paths):
    axj = fig.add_subplot(gs[1, k])
    for i in range(6): axj.plot(t, np.degrees(th[:, i]), color=JCOL[i], lw=1.8, label=f"theta{i+1}")
    cursors.append(axj.axvline(0, color="k", lw=1))
    jdots.append([axj.plot([], [], "o", color=JCOL[i], ms=6)[0] for i in range(6)])
    axj.set_ylim(jmin - 15, jmax + 15); axj.set_xlabel("t [s]"); axj.grid(alpha=0.3)
    if k == 0: axj.set_ylabel("joint angle [deg]"); axj.legend(ncol=3, fontsize=8, loc="upper left")
title = fig.suptitle("")

def update(f):
    k = min(f, len(t) - 1)
    if k == 0:
        for a in artists: a.hist = []
    out = []
    for j, (a, th) in enumerate(zip(artists, paths)):
        out += a.update(th[k])
        w, q, h = axes_data[j][k]
        if w is None:
            axis_lines[j].set_data([], []); axis_lines[j].set_3d_properties([]); axis_txt[j].set_text("pure translation")
        else:
            c = q + w * ((C_MID - q) @ w)                     # point on the axis nearest the middle of the motion
            seg = np.array([c - 0.3 * w, c + 0.3 * w])
            axis_lines[j].set_data(seg[:, 0], seg[:, 1]); axis_lines[j].set_3d_properties(seg[:, 2])
            axis_txt[j].set_text(f"pitch h = {h:+.3f} m/rad\naxis direction = ({w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f})")
        cursors[j].set_xdata([t[k], t[k]])
        for i, d in enumerate(jdots[j]): d.set_data([t[k]], [np.degrees(th[k, i])])
    slow = f"   (playback {SLOWMO}x slower)" if SLOWMO != 1 else ""
    title.set_text(f"Same start and end pose, same quintic time scaling (T = {T} s),  t = {t[k]:.2f} s{slow}")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.95))
run(fig, update, len(t) + FPS, 1000 / FPS, GIF, FPS)
