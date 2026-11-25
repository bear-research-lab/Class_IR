"""
04_screw_vs_decoupled.py  --  Two "straight lines" between the same two end-effector poses in SE(3).

  Screw path (Eq. 9.6):     X(s) = X_start exp( log(X_start^-1 X_end) s )
                            -> one constant screw axis; the origin moves on a helix-like curve
  Decoupled path (9.7-9.8): p(s) = p_start + s (p_end - p_start)
                            R(s) = R_start exp( log(R_start^T R_end) s )
                            -> the origin moves on a straight line

Things to notice (printed and plotted):
  * Both paths start at X_start and end at X_end.
  * The ROTATION part is identical on both paths. Only the origin path differs.
  * The bigger the rotation, the more the screw path's origin bulges away from the line.
    With zero rotation, the two paths are identical (pure translation).
  * (Optional) If the modern_robotics package is installed, the results are checked against
    mr.ScrewTrajectory and mr.CartesianTrajectory.

Try this:
  1. Set ROT_ANGLE_DEG = 0. Are the two paths different?  (pure translation -> identical)
  2. Increase ROT_ANGLE_DEG to 170. How does the screw path change?
  3. Set ANIMATE = True to see both frames move with a quintic time scaling.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traj_utils as tu

ROT_AXIS = [0.2, 0.3, 1.0]      # rotation axis between start and end orientation (in the start frame)
ROT_ANGLE_DEG = 120
P_START, P_END = np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.8])
T, N = 3.0, 60                  # time and number of samples for the trajectory
ANIMATE = False
SAVE = False

X_start = tu.RpToTrans(np.eye(3), P_START)
X_end = tu.RpToTrans(tu.rot(ROT_AXIS, np.radians(ROT_ANGLE_DEG)), P_END)

t = np.linspace(0, T, N)
s, *_ = tu.quintic_time_scaling(T, t)          # same time scaling for both paths
X_screw = tu.screw_path(X_start, X_end, s)
X_dec = tu.decoupled_path(X_start, X_end, s)

p_screw = np.array([X[:3, 3] for X in X_screw]); p_dec = np.array([X[:3, 3] for X in X_dec])
rot_diff = max(np.linalg.norm(A[:3, :3] - B[:3, :3]) for A, B in zip(X_screw, X_dec))
line_dir = (P_END - P_START) / np.linalg.norm(P_END - P_START)
def dist_to_line(p): d = p - P_START; return np.linalg.norm(d - np.outer(d @ line_dir, line_dir), axis=1)
path_len = lambda P: np.linalg.norm(np.diff(P, axis=0), axis=1).sum()

S = tu.log6(tu.trans_inv(X_start) @ X_end)      # [S]theta in the {start} frame
w_theta = tu.unskew(S[:3, :3]); v_theta = S[:3, 3]
print("Twist that takes X_start to X_end in unit time (in {start} frame):")
print("  omega*theta =", np.round(w_theta, 3), "  v*theta =", np.round(v_theta, 3))
print(f"Endpoints reached:  screw {np.allclose(X_screw[-1], X_end)}   decoupled {np.allclose(X_dec[-1], X_end)}")
print(f"Max difference between the rotation parts: {rot_diff:.2e}   (identical!)")
print(f"Origin path length:  screw {path_len(p_screw):.3f}   decoupled {path_len(p_dec):.3f}   (straight-line distance {np.linalg.norm(P_END-P_START):.3f})")
print(f"Max distance of the origin from the straight line:  screw {dist_to_line(p_screw).max():.3f}   decoupled {dist_to_line(p_dec).max():.1e}")

try:
    import modern_robotics as mr
    a = mr.ScrewTrajectory(X_start, X_end, T, N, 5); b = mr.CartesianTrajectory(X_start, X_end, T, N, 5)
    print("Check vs modern_robotics:  ScrewTrajectory max err %.1e,  CartesianTrajectory max err %.1e" %
          (max(np.abs(np.array(x) - y).max() for x, y in zip(a, X_screw)), max(np.abs(np.array(x) - y).max() for x, y in zip(b, X_dec))))
except ImportError:
    print("(modern_robotics not installed: skipping the library check;  pip install modern_robotics)")

# ------------------------------------------------------------ plots
def draw_frame(ax, X, L=0.3, lw=1.5, alpha=1.0):
    o = X[:3, 3]
    return [ax.plot(*np.vstack([o, o + L*X[:3, i]]).T, color=c, lw=lw, alpha=alpha)[0] for i, c in enumerate(["r", "g", "b"])]

fig = plt.figure(figsize=(14, 5))
for k, (name, Xs, P, col) in enumerate([("Screw path", X_screw, p_screw, "tab:blue"), ("Decoupled path", X_dec, p_dec, "tab:orange")]):
    ax = fig.add_subplot(1, 3, k+1, projection="3d")
    ax.plot(*np.vstack([P_START, P_END]).T, "k:", lw=1, label="straight line")
    ax.plot(*P.T, color=col, lw=2.5, label="origin path")
    for X in Xs[::6] + [Xs[-1]]: draw_frame(ax, X)
    ax.set_title(name); ax.legend(fontsize=8, loc="upper left")
    allp = np.vstack([p_screw, p_dec]); c = allp.mean(0); r = (allp.max(0) - allp.min(0)).max()/2 + 0.4
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax = fig.add_subplot(1, 3, 3)
ax.plot(t, dist_to_line(p_screw), color="tab:blue", lw=2, label="screw")
ax.plot(t, dist_to_line(p_dec), color="tab:orange", lw=2, label="decoupled")
ax.set_xlabel("t [s]"); ax.set_ylabel("distance of origin from the line"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f"Rotation {ROT_ANGLE_DEG} deg: same rotation, different origin path")
plt.tight_layout()
if SAVE: plt.savefig("04_screw_vs_decoupled.png", dpi=150)

if ANIMATE:
    fig2 = plt.figure(figsize=(7, 6)); ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot(*p_screw.T, color="tab:blue", alpha=0.4, label="screw"); ax2.plot(*p_dec.T, color="tab:orange", alpha=0.4, label="decoupled")
    allp = np.vstack([p_screw, p_dec]); c = allp.mean(0); r = (allp.max(0) - allp.min(0)).max()/2 + 0.4
    ax2.set_xlim(c[0]-r, c[0]+r); ax2.set_ylim(c[1]-r, c[1]+r); ax2.set_zlim(c[2]-r, c[2]+r); ax2.legend()
    lines = draw_frame(ax2, X_screw[0], lw=3) + draw_frame(ax2, X_dec[0], lw=3)
    def upd(k):
        for j, X in enumerate((X_screw[k], X_dec[k])):
            o = X[:3, 3]
            for i in range(3):
                seg = np.vstack([o, o + 0.4*X[:3, i]]); ln = lines[3*j + i]
                ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])
        ax2.set_title(f"t = {t[k]:.2f} s"); return lines
    anim = FuncAnimation(fig2, upd, frames=len(t), interval=60)
plt.show()
