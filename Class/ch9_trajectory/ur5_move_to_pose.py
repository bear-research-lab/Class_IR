"""
Animate the UR5 moving to a target end-effector pose, using analytic IK.
Needs ur5_ik_poe.py in the same folder, numpy and matplotlib (pillow to save a GIF).

    python ur5_move_to_pose.py                         # default target
    python ur5_move_to_pose.py x y z roll pitch yaw    # target: position [m], RPY [deg]
    e.g.  python ur5_move_to_pose.py 0.15 0.5 0.15 180 0 60   (tool pointing down)

Left : MoveJ-like  - solve IK once at the goal, pick the solution closest to the
                      current joints, interpolate in JOINT space.
Right: MoveL-like  - interpolate the end-effector pose in CARTESIAN space
                      (straight line + rotation about one axis) and solve IK at
                      every step, picking the solution closest to the previous step.
Same start, same goal, different paths: watch the end-effector traces.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ur5_ik_poe import S, exp6, fk, ik, W1, W2, L1, L2, H1, H2

# ---------------------------------------------------------------- geometry --
P0 = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
               [L1 + L2, W1, H1], [L1 + L2, W1, H1 - H2], [L1 + L2, W1 + W2, H1 - H2]])


def skeleton(q):
    """Positions of base, joints 2-6 and tool flange (point k moved by joints 1..k)."""
    pts, T = [P0[0]], np.eye(4)
    for k in range(6):
        T = T @ exp6(S[k], q[k])
        pts.append((T @ np.append(P0[k + 1], 1.0))[:3])
    return np.array(pts)


def rpy(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll)   (Ch. 2, roll-pitch-yaw)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def log_rot(R):
    """Matrix logarithm of a rotation (Ch. 3): returns (omega_hat, theta)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([0, 0, 1.0]), 0.0
    if np.pi - th < 1e-6:                          # theta = pi special case
        i = np.argmax(np.diag(R))
        w = (R[:, i] + np.eye(3)[i]) / np.sqrt(2 * (1 + R[i, i]))
        return w, th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w, th


def rot(w, th):
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def smooth(t):
    """Cubic time scaling: zero velocity at start and end."""
    return 3 * t**2 - 2 * t**3


def below_floor(q, tol=1e-3):
    return bool((skeleton(q)[1:, 2] < -tol).any())


def closest(sols, q_ref, avoid_floor=True):
    """IK solution closest to q_ref (angles unwrapped around q_ref)."""
    cand = [q_ref + (np.asarray(s) - q_ref + np.pi) % (2 * np.pi) - np.pi for s in sols]
    if avoid_floor:
        ok = [c for c in cand if not below_floor(c)]
        cand = ok or cand
    return min(cand, key=lambda c: np.linalg.norm(c - q_ref)) if cand else None


# ------------------------------------------------------------ trajectories --
def pose(x, y, z, roll, pitch, yaw):
    """Pose from position [m] and roll-pitch-yaw [deg]."""
    X = np.eye(4)
    X[:3, :3] = rpy(*np.radians([roll, pitch, yaw]))
    X[:3, 3] = [x, y, z]
    return X


def elbow_up_solution(X):
    """Pick a natural start configuration: above the floor, elbow as high as possible."""
    sols = [s for s in ik(X) if not below_floor(s)]
    if not sols:
        raise ValueError("Start pose is not reachable above the floor.")
    return max(sols, key=lambda q: skeleton(q)[2, 2])


def move_j(q0, X_goal, n=80):
    q_goal = closest(ik(X_goal), q0)
    if q_goal is None:
        raise ValueError("Target pose is not reachable.")
    return np.array([q0 + smooth(t) * (q_goal - q0) for t in np.linspace(0, 1, n)])


def move_l(q0, X_goal, n=80, max_jump=np.radians(30)):
    X0 = fk(q0)
    p0, pg = X0[:3, 3], X_goal[:3, 3]
    w, th = log_rot(X0[:3, :3].T @ X_goal[:3, :3])   # R0^T Rg = Rot(w, th), body frame
    qs, q = [q0], q0
    for t in np.linspace(0, 1, n)[1:]:
        s = smooth(t)
        X = np.eye(4)
        X[:3, :3] = X0[:3, :3] @ rot(w, s * th)        # body-frame rotation: post-multiply
        X[:3, 3] = p0 + s * (pg - p0)                  # straight line
        q_new = closest(ik(X), q, avoid_floor=False)
        if q_new is None or np.abs(q_new - q).max() > max_jump:
            print(f"MoveL stopped at s = {s:.2f}: unreachable or near a singularity.")
            break
        qs.append(q_new)
        q = q_new
    return np.array(qs)


# --------------------------------------------------------------- animation --
def draw_frame(ax, T, length=0.1, lw=3):
    lines = []
    for i, c in enumerate(("tab:red", "tab:green", "tab:blue")):
        o, e = T[:3, 3], T[:3, 3] + length * T[:3, i]
        lines += ax.plot(*zip(o, e), color=c, lw=lw)
    return lines


def update_frame(lines, T, length=0.1):
    for i, ln in enumerate(lines):
        o, e = T[:3, 3], T[:3, 3] + length * T[:3, i]
        ln.set_data_3d(*zip(o, e))


def animate(paths, titles, X_goal, gif=None):
    allpts = np.vstack([skeleton(q) for path in paths for q in path[::5]] + [X_goal[:3, 3][None]])
    lo, hi = allpts.min(0) - 0.1, allpts.max(0) + 0.1
    c, h = (lo + hi) / 2, (hi - lo).max() / 2

    fig = plt.figure(figsize=(8 * len(paths), 7.5))
    artists = []
    for k, (path, title) in enumerate(zip(paths, titles)):
        ax = fig.add_subplot(1, len(paths), k + 1, projection="3d")
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=28, azim=35)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        xx, yy = np.meshgrid([c[0] - h, c[0] + h], [c[1] - h, c[1] + h])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="0.6", alpha=0.15)
        ax.set_title(title, fontsize=12)

        draw_frame(ax, X_goal, 0.12, lw=2)                    # target (thin)
        ax.scatter(*X_goal[:3, 3], s=60, color="k", depthshade=False)
        start = skeleton(path[0])
        ax.plot(*start.T, color="0.7", lw=4, alpha=0.6)        # start pose ghost
        robot, = ax.plot([], [], [], "-o", color="tab:blue", lw=7, ms=6,
                         mfc="white", mec="k", solid_capstyle="round")
        trace, = ax.plot([], [], [], color="tab:orange", lw=2.5)
        ee = draw_frame(ax, fk(path[0]), 0.1, lw=4)            # moving end-effector
        ax.scatter(*fk(path[0])[:3, 3], s=40, color="tab:orange", depthshade=False)
        txt = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, fontsize=10, family="monospace")
        artists.append((path, robot, trace, ee, txt))

    nmax = max(len(p) for p in paths)

    def step(i):
        for path, robot, trace, ee, txt in artists:
            j = min(i, len(path) - 1)
            pts = skeleton(path[j])
            robot.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            tr = np.array([fk(q)[:3, 3] for q in path[:j + 1]])
            trace.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
            update_frame(ee, fk(path[j]))
            txt.set_text("q [deg] = " + " ".join(f"{a:5.0f}" for a in np.degrees(path[j])))
        return []

    anim = FuncAnimation(fig, step, frames=nmax + 15, interval=40)  # +15: hold at the end
    fig.tight_layout()
    if gif:
        anim.save(gif, writer="pillow", fps=25)
        print("saved", gif)
    return anim


if __name__ == "__main__":
    # Start: tool pointing down (RPY = 180, 0, 0) above the table
    X_start = pose(0.45, 0.15, 0.35, 180, 0, 0)
    q0 = elbow_up_solution(X_start)

    # Target: from the command line, or a default pick-up pose
    if len(sys.argv) == 7:
        X_goal = pose(*map(float, sys.argv[1:]))
    else:
        X_goal = pose(0.15, 0.50, 0.15, 180, 0, 60)

    print("start q [deg]:", np.round(np.degrees(q0), 1))
    print("goal  position:", X_goal[:3, 3])
    pj, pl = move_j(q0, X_goal), move_l(q0, X_goal)
    for name, path in (("MoveJ", pj), ("MoveL", pl)):
        err = np.linalg.norm(fk(path[-1]) - X_goal)
        print(f"{name}: final q = {np.round(np.degrees(path[-1]), 1)} deg, pose error = {err:.1e}")

    anim = animate([pj, pl], ["MoveJ-like: interpolate joints", "MoveL-like: interpolate the pose"],
                   X_goal, gif="ur5_move_to_pose.gif")
    plt.show()
