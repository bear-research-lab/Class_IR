"""
PUMA-type arm (spherical wrist, d6 = 0): three views that explain the position IK
Introduction to Robotics - Pieper's condition, Case 1

  (1) 3D view of the robot configuration
  (2) top view (from +z onto the xy-plane)          -> theta1 = atan2(p_wy, p_wx)
  (3) view of the arm plane (from the joint-2 axis) -> theta2, theta3 by the law of cosines

With d6 = 0 the end-effector sits exactly at the wrist center p_w, so theta4-theta6 only
change the orientation (the frame drawn in the 3D view), never the position.

Model (PoE, zero configuration = arm stretched along +x):
  joint 1 : about +z through the origin
  joints 2, 3 : about -y through (0, 0, H) and (L1, 0, H)   (positive angles lift the arm)
  joints 4, 5, 6 : about z, y, x through the wrist center (L1 + L2, 0, H)

    python puma_ik_views.py          (numpy, matplotlib; move the sliders)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

H, L1, L2 = 0.40, 0.45, 0.40


# ------------------------------------------------------------------ PoE model --
def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp6(s, th):
    w, v = s[:3], s[3:]
    K = skew(w)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    G = np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = G @ v
    return T


PW0 = np.array([L1 + L2, 0, H])
W = np.array([[0, 0, 1], [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
Q = np.array([[0, 0, 0], [0, 0, H], [L1, 0, H], PW0, PW0, PW0])
S = np.hstack([W, -np.cross(W, Q)])
M = np.eye(4); M[:3, 3] = PW0            # d6 = 0: end-effector frame at the wrist center


def joints_3d(q):
    """Base, shoulder, elbow and wrist center positions, plus the end-effector frame."""
    T, pts = np.eye(4), [np.zeros(3), np.array([0, 0, H])]
    for k in range(6):
        T = T @ exp6(S[k], q[k])
        if k == 1:
            pts.append((T @ np.append(Q[2], 1))[:3])      # elbow: moved by joints 1, 2
    X = T @ M
    pts.append(X[:3, 3])                                  # wrist center = end-effector
    return np.array(pts), X


def position_ik(pw):
    """Closed form for theta1, theta2, theta3 (Pieper Case 1, zero offsets): 4 solutions."""
    out = []
    rxy = np.hypot(pw[0], pw[1])
    for th1, rho in ((np.arctan2(pw[1], pw[0]), rxy), (np.arctan2(pw[1], pw[0]) + np.pi, -rxy)):
        z = pw[2] - H
        D = (rho**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        if abs(D) > 1:
            continue
        for sgn in (1, -1):
            th3 = np.arctan2(sgn * np.sqrt(1 - D**2), D)
            th2 = np.arctan2(z, rho) - np.arctan2(L2 * np.sin(th3), L1 + L2 * np.cos(th3))
            out.append((th1, th2, th3, rho, z, D))
    return out


def wrapd(a):
    return (np.degrees(a) + 180) % 360 - 180


# ------------------------------------------------------------------------- app --
class App:
    def __init__(self):
        self.fig = plt.figure(figsize=(18, 9.5))
        self.ax3 = self.fig.add_axes([0.00, 0.30, 0.34, 0.65], projection="3d")
        self.axt = self.fig.add_axes([0.37, 0.33, 0.28, 0.60])
        self.axp = self.fig.add_axes([0.70, 0.33, 0.28, 0.60])
        self.info = self.fig.text(0.63, 0.25, "", family="monospace", fontsize=10, va="top")
        init = [35, 30, -70, 20, 40, -30]
        self.sl = []
        for i, v in enumerate(init):
            ax = self.fig.add_axes([0.06, 0.22 - 0.035 * i, 0.50, 0.022])
            s = Slider(ax, f"θ{i + 1} [deg]", -180, 180, valinit=v, valstep=1)
            s.on_changed(lambda _v: self.draw())
            self.sl.append(s)
        self.draw()

    def q(self):
        return np.radians([s.val for s in self.sl])

    # ---------------------------------------------------------------- (1) 3D --
    def draw_3d(self, pts, X, th1):
        ax = self.ax3; ax.cla()
        R = L1 + L2
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, H + R)
        ax.set_box_aspect((2 * R, 2 * R, H + R))
        g = np.linspace(-R, R, 7)
        for v in g:
            ax.plot([v, v], [-R, R], [0, 0], color="0.88", lw=0.8)
            ax.plot([-R, R], [v, v], [0, 0], color="0.88", lw=0.8)
        # the arm plane (contains the z-axis, rotated by theta1)
        u = np.array([np.cos(th1), np.sin(th1), 0])
        corners = np.array([-0.3 * R * u, R * u, R * u + [0, 0, H + R], -0.3 * R * u + [0, 0, H + R]])
        ax.add_collection3d(Poly3DCollection([corners], color="tab:orange", alpha=0.12))
        ax.plot(*pts.T, "-o", color="tab:blue", lw=7, ms=8, mfc="white", mec="k")
        ax.plot([0, 0], [0, 0], [0, H + R], ":", color="0.5")                       # joint-1 axis
        for i, c in enumerate(("tab:red", "tab:green", "tab:blue")):
            ax.plot(*zip(X[:3, 3], X[:3, 3] + 0.15 * X[:3, i]), color=c, lw=3)
        pw = X[:3, 3]
        ax.plot([pw[0], pw[0]], [pw[1], pw[1]], [0, pw[2]], "--", color="tab:red", lw=1)
        ax.text(*pw + [0, 0, 0.06], "p_w", color="tab:red")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("(1) Robot configuration\n(orange: arm plane; θ4–θ6 only turn the frame)", fontsize=11)

    # ----------------------------------------------------------- (2) top view --
    def draw_top(self, pts, sol):
        ax = self.axt; ax.cla()
        R = L1 + L2
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ls=":", color="0.7"))
        ax.plot(pts[:, 0], pts[:, 1], "-o", color="tab:blue", lw=6, ms=8, mfc="white", mec="k", zorder=3)
        pw = pts[-1]
        rxy = np.hypot(pw[0], pw[1]); t1 = np.arctan2(pw[1], pw[0])
        ax.plot([0, pw[0]], [0, pw[1]], "--", color="tab:red", lw=1.5)
        ax.plot([0, -pw[0]], [0, -pw[1]], ":", color="0.5", lw=2, label="second solution θ1 + π (reach backwards)")
        ax.plot(*pw[:2], "o", color="tab:red", ms=10, zorder=4)
        right = pw[0] > 0
        ax.annotate(f"(p_wx, p_wy) = ({pw[0]:.2f}, {pw[1]:.2f})", pw[:2], textcoords="offset points",
                    xytext=(-10 if right else 10, 10), ha="right" if right else "left", color="tab:red", fontsize=10)
        ax.axhline(0, color="0.6", lw=1); ax.axvline(0, color="0.6", lw=1)
        ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=min(0, np.degrees(t1)), theta2=max(0, np.degrees(t1)), color="k", lw=2))
        ax.text(0.30 * np.cos(t1 / 2), 0.30 * np.sin(t1 / 2), "θ1", fontsize=14)
        ax.text(0.5 * pw[0] - 0.05 * np.sin(t1), 0.5 * pw[1] + 0.05 * np.cos(t1), f"r = {rxy:.3f}", color="tab:red", fontsize=10,
                rotation=np.degrees(t1) if abs(np.degrees(t1)) <= 90 else np.degrees(t1) - 180, ha="center")
        ax.set_xlim(-R - 0.1, R + 0.1); ax.set_ylim(-R - 0.1, R + 0.1); ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(f"(2) Top view (from +z)\nθ1 = atan2(p_wy, p_wx) = {np.degrees(t1):.1f}°", fontsize=11)

    # ---------------------------------------------------------- (3) arm plane --
    def draw_plane(self, q):
        ax = self.axp; ax.cla()
        th2, th3 = q[1], q[2]
        sh = np.array([0, H]); el = sh + L1 * np.array([np.cos(th2), np.sin(th2)])
        wr = el + L2 * np.array([np.cos(th2 + th3), np.sin(th2 + th3)])
        rho, z = wr[0], wr[1] - H
        # the other elbow solution for the same wrist point (dashed)
        D = (rho**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        t3b = -th3; t2b = np.arctan2(z, rho) - np.arctan2(L2 * np.sin(t3b), L1 + L2 * np.cos(t3b))
        elb = sh + L1 * np.array([np.cos(t2b), np.sin(t2b)])
        ax.plot([sh[0], elb[0], wr[0]], [sh[1], elb[1], wr[1]], "--o", color="tab:orange", lw=3, ms=6,
                mfc="white", label="other elbow solution (−θ3)")
        ax.plot([0, 0, el[0], wr[0]], [0, H, el[1], wr[1]], "-o", color="tab:blue", lw=6, ms=8, mfc="white", mec="k", zorder=3)
        ax.plot([sh[0], wr[0]], [sh[1], wr[1]], ":", color="tab:red", lw=2)          # shoulder -> wrist
        ax.axhline(H, color="0.75", lw=1, ls="--")
        # angle arcs
        a2 = np.degrees(th2)
        ax.add_patch(Arc(sh, 0.24, 0.24, theta1=min(0, a2), theta2=max(0, a2), color="k", lw=2))
        ax.text(sh[0] + 0.15 * np.cos(th2 / 2), sh[1] + 0.15 * np.sin(th2 / 2), "θ2", fontsize=14)
        ext = el + 0.2 * np.array([np.cos(th2), np.sin(th2)])
        ax.plot([el[0], ext[0]], [el[1], ext[1]], ":", color="0.4")
        a_from, a_to = np.degrees(th2), np.degrees(th2 + th3)
        ax.add_patch(Arc(el, 0.24, 0.24, theta1=min(a_from, a_to), theta2=max(a_from, a_to), color="k", lw=2))
        am = th2 + th3 / 2
        ax.text(el[0] + 0.16 * np.cos(am), el[1] + 0.16 * np.sin(am), "θ3", fontsize=14)
        ax.plot(*wr, "o", color="tab:red", ms=10, zorder=4)
        ax.annotate(f"(ρ, z) = ({rho:.2f}, {z:.2f})", wr, textcoords="offset points", xytext=(-10 if rho > 0 else 10, -20),
                    ha="right" if rho > 0 else "left", color="tab:red", fontsize=10)
        ax.text(0.5 * (sh[0] + el[0]), 0.5 * (sh[1] + el[1]) + 0.04, "L1", color="tab:blue", fontsize=12)
        ax.text(0.5 * (el[0] + wr[0]), 0.5 * (el[1] + wr[1]) + 0.04, "L2", color="tab:blue", fontsize=12)
        R = L1 + L2
        ax.set_xlim(-R - 0.1, R + 0.1); ax.set_ylim(-0.1, H + R + 0.1); ax.set_aspect("equal")
        ax.set_xlabel("ρ: distance along the arm plane [m]"); ax.set_ylabel("z [m]"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(f"(3) Arm plane, seen along the joint-2 axis\nθ2, θ3 from the law of cosines (D = {D:.3f})", fontsize=11)

    def draw(self):
        q = self.q()
        pts, X = joints_3d(q)
        sols = position_ik(X[:3, 3])
        self.draw_3d(pts, X, q[0])
        self.draw_top(pts, sols)
        self.draw_plane(q)
        lines = ["Formulas (p_w = end-effector position, d6 = 0):",
                 "  θ1 = atan2(p_wy, p_wx)   or  + 180°",
                 "  ρ = ±√(p_wx² + p_wy²),   z = p_wz − H",
                 "  D = (ρ² + z² − L1² − L2²) / (2 L1 L2)",
                 "  θ3 = atan2(±√(1 − D²), D)",
                 "  θ2 = atan2(z, ρ) − atan2(L2 sin θ3, L1 + L2 cos θ3)", "",
                 "Position IK from p_w only (4 solutions):", "   θ1      θ2      θ3"]
        for th1, th2, th3, rho, z, D in sols:
            mark = "  <- slider values" if np.allclose(wrapd(np.array([th1, th2, th3])), wrapd(q[:3]), atol=1e-6) else ""
            lines.append(f"{wrapd(th1):7.1f} {wrapd(th2):7.1f} {wrapd(th3):7.1f}{mark}")
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # sanity check: the planar formulas of view (3) match the PoE forward kinematics
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 6)
        pts, X = joints_3d(q)
        rho = L1 * np.cos(q[1]) + L2 * np.cos(q[1] + q[2])
        pw = np.array([rho * np.cos(q[0]), rho * np.sin(q[0]), H + L1 * np.sin(q[1]) + L2 * np.sin(q[1] + q[2])])
        assert np.allclose(pw, X[:3, 3])
    app = App()
    plt.show()
