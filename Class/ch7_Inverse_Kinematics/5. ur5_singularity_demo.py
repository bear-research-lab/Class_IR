"""
Why singularities are dangerous: UR5 wrist singularity demo (analytic IK, PoE model).
Needs ur5_ik_poe.py in the same folder, numpy and matplotlib (pillow for the GIF).

    python ur5_singularity_demo.py

Experiment
  The end-effector moves along a straight line at a constant 0.1 m/s with a FIXED
  orientation (MoveL). The tool axis is tilted by alpha from the horizontal, and the
  path crosses the configuration where the joint-6 axis lines up with the joint-2/3/4
  axes (sin(theta5) = 0, the wrist singularity). Smaller alpha = passes closer.

  Figure 1: joint speeds, |sin theta5|, smallest singular value of the Jacobian, and
            what happens if the joint speeds are limited (the path is no longer followed)
  Figure 2: animation of the closest pass (wrist flips almost instantly)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ur5_ik_poe import S, M, exp6, fk, ik, W1, W2, L1, L2, H1, H2

V_EE = 0.10                 # end-effector speed [m/s]
QDOT_MAX = np.radians(180)  # example joint speed limit [rad/s] (check your robot's datasheet)
ALPHAS = [10.0, 3.0, 0.5]   # tool tilt from horizontal [deg]; 0 = exactly through the singularity

P0 = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
               [L1 + L2, W1, H1], [L1 + L2, W1, H1 - H2], [L1 + L2, W1 + W2, H1 - H2]])


# ------------------------------------------------------------------ helpers --
def skeleton(q):
    pts, T = [P0[0]], np.eye(4)
    for k in range(6):
        T = T @ exp6(S[k], q[k])
        pts.append((T @ np.append(P0[k + 1], 1.0))[:3])
    return np.array(pts)


def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    P = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    return np.block([[R, np.zeros((3, 3))], [P @ R, R]])


def space_jacobian(q):
    """J_s = [S1, Ad_{e^[S1]q1} S2, ...]  (preview of the next chapter)."""
    J, T = np.zeros((6, 6)), np.eye(4)
    for i in range(6):
        J[:, i] = adjoint(T) @ S[i]
        T = T @ exp6(S[i], q[i])
    return J


def joint_axes(q):
    """Point and direction of joint 4 and joint 6 axes in the current configuration."""
    out, T = {}, np.eye(4)
    Q = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
                  [L1 + L2, W1, 0], [L1 + L2, 0, H1 - H2]], float)
    for i in range(6):
        if i in (3, 5):
            out[i] = ((T @ np.append(Q[i], 1))[:3], T[:3, :3] @ S[i][:3])
        T = T @ exp6(S[i], q[i])
    return out


def unwrap_to(q, ref):
    return ref + (np.asarray(q) - ref + np.pi) % (2 * np.pi) - np.pi


def target_pose(alpha_deg, y):
    a = np.radians(alpha_deg)
    xb = np.array([1.0, 0, 0])
    zb = np.array([0, np.cos(a), -np.sin(a)])         # tool axis, tilted by alpha
    X = np.eye(4)
    X[:3, :3] = np.column_stack([xb, np.cross(zb, xb), zb])
    X[:3, 3] = [0.45, y, 0.30]
    return X


def track(Xs, q=None):
    """IK at every step, always taking the solution closest to the previous one."""
    Q = []
    for X in Xs:
        sols = ik(X)
        q = (max(sols, key=lambda s: skeleton(s)[2, 2]) if q is None
             else min((unwrap_to(s, q) for s in sols), key=lambda c: np.linalg.norm(c - q)))
        Q.append(q)
    return np.array(Q)


def singular_y(alpha_deg):
    """y where the straight line passes closest to the wrist singularity (coarse search)."""
    ys = np.linspace(-0.3, 0.6, 181)
    Q = track([target_pose(alpha_deg, y) for y in ys])
    return ys[np.argmin(np.abs(np.sin(Q[:, 4])))]


def run(alpha_deg, half=0.25, dt=0.01):
    """MoveL along y through the singular region; IK at every step (closest solution)."""
    y_mid = singular_y(alpha_deg)                     # centre the path on the singular point
    T_total = 2 * half / V_EE
    ts = np.arange(0, T_total + dt / 2, dt)
    ys = y_mid - half + V_EE * ts
    Xs = [target_pose(alpha_deg, y) for y in ys]
    Q = track(Xs)
    dQ = np.gradient(Q, ts, axis=0)
    smin = np.array([np.linalg.svd(space_jacobian(qq), compute_uv=False)[-1] for qq in Q])

    # Same path, but the joints are not allowed to move faster than QDOT_MAX
    qc, err = Q[0].copy(), []
    for k in range(len(ts)):
        step = np.clip(Q[k] - qc, -QDOT_MAX * dt, QDOT_MAX * dt)
        qc = qc + step
        err.append(np.linalg.norm(fk(qc)[:3, 3] - Xs[k][:3, 3]))
    return dict(t=ts, Q=Q, dQ=dQ, smin=smin, err=np.array(err), X=Xs)


# ----------------------------------------------------------------- figure 1 --
def summary_figure(runs):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["tab:green", "tab:orange", "tab:red"]
    for (a, r), c in zip(runs.items(), colors):
        lab = f"α = {a:g}°"
        sp = np.degrees(np.abs(r["dQ"])).max(1)
        axs[0, 0].semilogy(r["t"], sp, color=c, lw=2, label=f"{lab}: peak {sp.max():.0f}°/s")
        axs[0, 1].plot(r["t"], np.abs(np.sin(r["Q"][:, 4])), color=c, lw=2, label=lab)
        axs[1, 0].semilogy(r["t"], r["smin"], color=c, lw=2, label=lab)
        axs[1, 1].plot(r["t"], 1000 * r["err"], color=c, lw=2, label=f"{lab}: max {1000*r['err'].max():.0f} mm")
    axs[0, 0].axhline(np.degrees(QDOT_MAX), color="k", ls="--", lw=1.5, label="joint speed limit (example)")
    axs[0, 0].set(title="Fastest joint speed  max|θ̇ᵢ|  [deg/s]", xlabel="time [s]")
    axs[0, 1].set(title="|sin θ5|  (0 = wrist singularity)", xlabel="time [s]", ylim=(0, 1.05))
    axs[1, 0].set(title="Smallest singular value of the Jacobian  σ_min(J)", xlabel="time [s]")
    axs[1, 1].set(title="Path error if joint speeds are limited  [mm]", xlabel="time [s]")
    for ax in axs.flat:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"UR5 MoveL at a constant {V_EE} m/s near the wrist singularity: "
                 "the end-effector is slow, but the joints are not", fontsize=14)
    fig.tight_layout()
    fig.savefig("ur5_singularity_summary.png", dpi=150)
    return fig


# ----------------------------------------------------------------- figure 2 --
def animation(r, alpha_deg, every=4, gif="ur5_singularity.gif"):
    t, Q, dQ = r["t"], r["Q"], np.degrees(np.abs(r["dQ"]))
    idx = np.arange(0, len(t), every)
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    pts_all = np.vstack([skeleton(q) for q in Q[::20]])
    lo, hi = pts_all.min(0) - 0.12, pts_all.max(0) + 0.12
    c, h = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=30)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    P = np.array([X[:3, 3] for X in r["X"]])
    ax.plot(*P.T, color="0.7", ls="--", lw=1.5)                    # commanded straight line
    robot, = ax.plot([], [], [], "-o", color="tab:blue", lw=7, ms=6, mfc="white", mec="k")
    ax4, = ax.plot([], [], [], color="tab:purple", lw=2, ls="--", label="joint 4 axis")
    ax6, = ax.plot([], [], [], color="tab:red", lw=2, ls="--", label="joint 6 axis")
    trace, = ax.plot([], [], [], color="tab:orange", lw=3, label="end-effector")
    ax.legend(loc="upper left", fontsize=9)
    title = ax.set_title("")

    names = ["θ1", "θ2", "θ3", "θ4", "θ5", "θ6"]
    cols = plt.cm.tab10(np.arange(6))
    lines = [ax2.semilogy([], [], color=cols[i], lw=2, label=names[i])[0] for i in range(6)]
    ax2.axhline(np.degrees(QDOT_MAX), color="k", ls="--", lw=1.5, label="limit (example)")
    ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(0.1, max(10 * dQ.max(), 1e3))
    ax2.set(xlabel="time [s]", ylabel="|joint speed| [deg/s]",
            title=f"Joint speeds, end-effector speed fixed at {V_EE} m/s")
    ax2.grid(alpha=0.3); ax2.legend(ncol=4, fontsize=9, loc="upper left")

    def step(k):
        i = idx[min(k, len(idx) - 1)]
        q = Q[i]
        p = skeleton(q)
        robot.set_data_3d(*p.T)
        axes = joint_axes(q)
        for ln, j in ((ax4, 3), (ax6, 5)):
            o, d = axes[j]
            e1, e2 = o - 0.25 * d, o + 0.25 * d
            ln.set_data_3d(*np.array([e1, e2]).T)
        tr = np.array([fk(qq)[:3, 3] for qq in Q[:i + 1:every]])
        trace.set_data_3d(*tr.T)
        for j, ln in enumerate(lines):
            ln.set_data(t[:i + 1], np.maximum(dQ[:i + 1, j], 1e-3))
        s5 = abs(np.sin(q[4]))
        warn = dQ[i].max() > np.degrees(QDOT_MAX)
        title.set_text(f"α = {alpha_deg}°   t = {t[i]:.2f} s   |sin θ5| = {s5:.3f}"
                       + ("   ⚠ joint speed limit exceeded" if warn else ""))
        title.set_color("tab:red" if warn else "k")
        return []

    anim = FuncAnimation(fig, step, frames=len(idx) + 12, interval=40)
    fig.tight_layout()
    if gif:
        anim.save(gif, writer="pillow", fps=25)
        print("saved", gif)
    return anim


if __name__ == "__main__":
    runs = {a: run(a) for a in ALPHAS}
    for a, r in runs.items():
        sp = np.degrees(np.abs(r["dQ"])).max(0)
        print(f"alpha = {a:4.1f} deg | min|sin th5| = {np.abs(np.sin(r['Q'][:, 4])).min():.4f} | "
              f"peak joint speeds [deg/s] = {np.round(sp).astype(int)} | "
              f"path error with speed limit = {1000 * r['err'].max():.0f} mm")
    summary_figure(runs)
    anim = animation(runs[ALPHAS[-1]], ALPHAS[-1])
    plt.show()
