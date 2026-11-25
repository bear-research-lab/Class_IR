"""
Interactive analytic IK for the UR5: move sliders for x, y, z, roll, pitch, yaw
and watch the robot follow the target end-effector pose.
Needs ur5_ik_poe.py in the same folder, numpy and matplotlib.

    python ur5_ik_interactive.py
    (Jupyter: run "%matplotlib widget" first, requires the ipympl package)

Controls
  sliders      : target position [m] and roll-pitch-yaw [deg], R = Rz(yaw) Ry(pitch) Rx(roll)
  "Solution"   : auto = IK solution closest to the current joints (smooth tracking),
                 1..8 = always the same branch (shoulder / elbow / wrist)
  "Show all 8" : draw every IK solution faintly
  "Wrist axes" : dashed purple = joint-4 axis, dashed red = joint-6 axis
  "Reset"      : back to the start pose

Singularity experiment: choose Solution 6 (or 8), set x = 0.45, z = 0.30,
roll = -89, pitch = 0, yaw = 0, and drag y slowly from 0.15 to -0.2 m (it crosses y = -0.03). The two dashed wrist axes line up
(|sin θ5| -> 0, σ_min(J) -> 0) and the wrist joints jump: a few mm of end-effector
motion asks for tens of degrees of joint motion.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from ur5_ik_poe import S, exp6, fk, ik, W1, W2, L1, L2, H1, H2

P0 = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
               [L1 + L2, W1, H1], [L1 + L2, W1, H1 - H2], [L1 + L2, W1 + W2, H1 - H2]])
START = dict(x=0.45, y=0.15, z=0.35, roll=180.0, pitch=0.0, yaw=0.0)


def skeleton(q):
    pts, T = [P0[0]], np.eye(4)
    for k in range(6):
        T = T @ exp6(S[k], q[k])
        pts.append((T @ np.append(P0[k + 1], 1.0))[:3])
    return np.array(pts)


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def pose(x, y, z, roll, pitch, yaw):
    X = np.eye(4)
    X[:3, :3] = rpy(*np.radians([roll, pitch, yaw]))
    X[:3, 3] = [x, y, z]
    return X


def below_floor(q, tol=1e-3):
    return bool((skeleton(q)[1:, 2] < -tol).any())


def space_jacobian(q):
    """J_s = [S1, Ad_{e^[S1]q1} S2, ...] (next chapter); its smallest singular value -> 0 at a singularity."""
    J, T = np.zeros((6, 6)), np.eye(4)
    for i in range(6):
        R, p = T[:3, :3], T[:3, 3]
        P = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        J[:, i] = np.block([[R, np.zeros((3, 3))], [P @ R, R]]) @ S[i]
        T = T @ exp6(S[i], q[i])
    return J


def wrist_axes(q):
    """Point and direction of the joint-4 and joint-6 axes (they align at the wrist singularity)."""
    Q = [None, None, None, [L1 + L2, 0, H1], None, [L1 + L2, 0, H1 - H2]]
    out, T = {}, np.eye(4)
    for i in range(6):
        if i in (3, 5):
            out[i] = ((T @ np.append(Q[i], 1.0))[:3], T[:3, :3] @ S[i][:3])
        T = T @ exp6(S[i], q[i])
    return out


def unwrap_to(q, q_ref):
    return q_ref + (np.asarray(q) - q_ref + np.pi) % (2 * np.pi) - np.pi


class App:
    def __init__(self):
        self.fig = plt.figure(figsize=(13, 8.5))
        self.ax = self.fig.add_axes([0.0, 0.28, 0.66, 0.66], projection="3d")

        # --- sliders ---
        spec = [("x", -0.9, 0.9, "m"), ("y", -0.9, 0.9, "m"), ("z", -0.2, 1.0, "m"),
                ("roll", -180, 180, "deg"), ("pitch", -90, 90, "deg"), ("yaw", -180, 180, "deg")]
        self.sl = {}
        for i, (name, lo, hi, unit) in enumerate(spec):
            ax_s = self.fig.add_axes([0.10, 0.23 - 0.037 * i, 0.50, 0.025])
            s = Slider(ax_s, f"{name} [{unit}]", lo, hi, valinit=START[name],
                       valstep=0.005 if unit == "m" else 1.0)
            s.on_changed(lambda _v: self.update())
            self.sl[name] = s

        # --- solution selector, options, reset ---
        ax_r = self.fig.add_axes([0.70, 0.52, 0.12, 0.40])
        ax_r.set_title("Solution", fontsize=11)
        self.radio = RadioButtons(ax_r, ["auto"] + [str(i) for i in range(1, 9)], active=0)
        self.radio.on_clicked(lambda _l: self.update())
        ax_c = self.fig.add_axes([0.84, 0.82, 0.15, 0.10])
        self.check = CheckButtons(ax_c, ["Show all 8", "Avoid floor", "Wrist axes"], [False, True, True])
        self.check.on_clicked(lambda _l: self.update())
        ax_b = self.fig.add_axes([0.84, 0.72, 0.10, 0.05])
        self.btn = Button(ax_b, "Reset")
        self.btn.on_clicked(self.reset)
        self.info = self.fig.text(0.70, 0.45, "", fontsize=10, family="monospace", va="top")

        self.q = None                      # current joints (for "auto" tracking)
        self.p_last = None                 # last target position (for the sensitivity readout)
        self.update()

    def reset(self, _event=None):
        self.q, self.p_last = None, None
        for k, s in self.sl.items():
            s.eventson = False; s.set_val(START[k]); s.eventson = True
        self.update()

    # ------------------------------------------------------------ drawing --
    def setup(self):
        ax = self.ax
        ax.cla()
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(-0.3, 1.0)
        ax.set_box_aspect((1.8, 1.8, 1.3))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        xx, yy = np.meshgrid([-0.9, 0.9], [-0.9, 0.9])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="0.6", alpha=0.15)
        # reach sphere outline (horizontal circle at shoulder height)
        t = np.linspace(0, 2 * np.pi, 100); R = L1 + L2 + W2
        ax.plot(R * np.cos(t), R * np.sin(t), H1 + 0 * t, color="0.75", lw=1, ls="--")

    def frame(self, T, length, lw, ls="-"):
        for i, c in enumerate(("tab:red", "tab:green", "tab:blue")):
            o, e = T[:3, 3], T[:3, 3] + length * T[:3, i]
            self.ax.plot(*zip(o, e), color=c, lw=lw, ls=ls)

    def robot(self, q, color, lw, alpha=1.0, joints=True):
        p = skeleton(q)
        self.ax.plot(*p.T, "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round")
        if joints:
            self.ax.scatter(*p[1:6].T, s=35, color="white", edgecolors="k", depthshade=False)

    # ------------------------------------------------------------- update --
    def update(self):
        v = {k: s.val for k, s in self.sl.items()}
        X = pose(**v)
        show_all, avoid_floor, show_axes = self.check.get_status()
        q_old = None if self.q is None else self.q.copy()
        sols = ik(X)
        choice = self.radio.value_selected

        self.setup()
        self.frame(X, 0.15, 2, ls="--")                       # target frame
        if show_all:
            for s in sols:
                self.robot(s, "0.55", 3, alpha=0.35, joints=False)

        msg, color, q_new, idx = "", "k", None, None
        if not sols:
            msg, color = "UNREACHABLE: no IK solution", "tab:red"
        else:
            cand = list(enumerate(sols))
            if avoid_floor:
                cand = [(i, s) for i, s in cand if not below_floor(s)]
            if choice == "auto":
                if not cand:
                    msg, color = "every solution goes below the floor", "tab:red"
                else:
                    if self.q is None:          # first pose: natural elbow-up configuration
                        idx, q_new = max(cand, key=lambda c: skeleton(c[1])[2, 2])
                    else:                       # then: closest to the current joints
                        idx, q_new = min(((i, unwrap_to(s, self.q)) for i, s in cand),
                                         key=lambda c: np.linalg.norm(c[1] - self.q))
            else:
                idx = int(choice) - 1
                if idx >= len(sols):
                    msg, color, idx = f"only {len(sols)} solutions here", "tab:red", None
                else:
                    q_new = sols[idx]
                    if avoid_floor and below_floor(q_new):
                        msg, color = "this solution goes below the floor", "tab:orange"

        if q_new is not None:
            if self.q is not None and choice == "auto" and np.abs(q_new - self.q).max() > np.radians(45):
                msg, color = "large joint jump (near a singularity?)", "tab:orange"
            self.q = q_new
        if self.q is not None:
            self.robot(self.q, "tab:blue" if q_new is not None else "0.4", 7)
            self.frame(fk(self.q), 0.1, 4)                     # actual end-effector
            if show_axes:
                for j, col in ((3, "tab:purple"), (5, "tab:red")):
                    o, d = wrist_axes(self.q)[j]
                    self.ax.plot(*np.array([o - 0.3 * d, o + 0.3 * d]).T, color=col, lw=1.8, ls="--")

        # sensitivity of the last slider move: how much did the joints move per mm?
        sens = ""
        if q_old is not None and q_new is not None and self.p_last is not None:
            dp = 1000 * np.linalg.norm(X[:3, 3] - self.p_last)
            dq = np.degrees(np.abs((q_new - q_old + np.pi) % (2 * np.pi) - np.pi)).max()
            if dp > 1e-6:
                sens = f"last move    : {dp:.0f} mm -> {dq:.0f} deg ({dq / dp:.1f} deg/mm)"
        self.p_last = X[:3, 3].copy()

        err = np.linalg.norm(fk(self.q) - X) if (self.q is not None and q_new is not None) else np.nan
        lines = [f"IK solutions : {len(sols)}",
                 f"using        : #{idx + 1}" if idx is not None else "using        : -",
                 f"pose error   : {err:.1e}", "", "joint angles [deg]"]
        if self.q is not None:
            lines += [f"  θ{i + 1} = {a:8.1f}" for i, a in enumerate(np.degrees(self.q))]
        if self.q is not None:
            smin = np.linalg.svd(space_jacobian(self.q), compute_uv=False)[-1]
            lines += ["", f"|sin θ5|     : {abs(np.sin(self.q[4])):.3f}", f"σ_min(J)     : {smin:.4f}"]
        if sens:
            lines += [sens]
        self.info.set_text("\n".join(lines))
        self.ax.set_title(msg or "UR5 analytic IK: dashed = target, solid = robot end-effector",
                          color=color, fontsize=12)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = App()
    plt.show()
