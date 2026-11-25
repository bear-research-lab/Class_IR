"""
anim_utils.py  --  small drawing helpers shared by the animation scripts (A1-A5).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Arm2DArtist:
    """Draws a planar 2R arm (base, two links, end-effector) and an optional end-effector trail."""
    def __init__(self, ax, arm, color="tab:blue", lw=6, trail=True, label=None, alpha=1.0, trail_style="-"):
        self.arm = arm
        self.line, = ax.plot([], [], "-", color=color, lw=lw, solid_capstyle="round", alpha=alpha, label=label, zorder=3)
        self.joints, = ax.plot([], [], "o", color="white", mec=color, mew=2, ms=7, alpha=alpha, zorder=4)
        self.trail, = ax.plot([], [], trail_style, color=color, lw=1.5, ms=4, alpha=0.6 * alpha, zorder=2) if trail else (None,)
        self.hist = []

    def update(self, th):
        e, p = self.arm.elbow(th), self.arm.fk(th)
        self.line.set_data([0, e[0], p[0]], [0, e[1], p[1]])
        self.joints.set_data([0, e[0]], [0, e[1]])
        if self.trail is not None:
            self.hist.append(p); h = np.array(self.hist); self.trail.set_data(h[:, 0], h[:, 1])
        return [self.line, self.joints] + ([self.trail] if self.trail is not None else [])

    def reset(self):
        self.hist = []


def draw_base_2d(ax, size=0.12):
    ax.add_patch(plt.Rectangle((-size, -size), 2 * size, size, color="0.25", zorder=1))


class UR5Artist:
    """Draws the UR5 skeleton, a simple two-finger gripper and the end-effector frame in 3D."""
    def __init__(self, ax, robot, color="0.35", trail_color="tab:orange"):
        self.robot = robot
        self.links, = ax.plot([], [], [], "-", color=color, lw=6, solid_capstyle="round")
        self.joints, = ax.plot([], [], [], "o", color="0.15", ms=6)
        self.frame = [ax.plot([], [], [], "-", color=c, lw=2.5)[0] for c in ("r", "g", "b")]
        self.grip = [ax.plot([], [], [], "-", color="k", lw=3)[0] for _ in range(3)]
        self.trail, = ax.plot([], [], [], "-", color=trail_color, lw=2.5)
        self.hist = []

    @staticmethod
    def _set(line, P):
        P = np.asarray(P); line.set_data(P[:, 0], P[:, 1]); line.set_3d_properties(P[:, 2])

    def update(self, theta):
        pts = self.robot.skeleton(theta); T = self.robot.fk(theta); R, p = T[:3, :3], T[:3, 3]
        self._set(self.links, pts); self._set(self.joints, pts[:-1])
        for i, ln in enumerate(self.frame):
            self._set(ln, [p, p + 0.12 * R[:, i]])
        # gripper: palm along the frame's x-axis, fingers along its z-axis
        a, z = 0.05 * R[:, 0], R[:, 2]
        self._set(self.grip[0], [p - a, p + a])
        self._set(self.grip[1], [p - a, p - a + 0.06 * z])
        self._set(self.grip[2], [p + a, p + a + 0.06 * z])
        self.hist.append(p); self._set(self.trail, self.hist)
        return [self.links, self.joints, self.trail] + self.frame + self.grip


def setup_3d(ax, center=(0.25, 0.0, 0.3), r=0.55, elev=25, azim=-60):
    ax.set_xlim(center[0] - r, center[0] + r); ax.set_ylim(center[1] - r, center[1] + r); ax.set_zlim(0, 2 * r * 0.8)
    ax.set_box_aspect((1, 1, 0.8)); ax.view_init(elev, azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def run(fig, update, n_frames, interval_ms, gif_name=None, fps=25, dpi=70):
    """Show the animation, or save it as a GIF when gif_name is given.
    n_frames: number of frames, or a list of frame indices passed to update()."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False, repeat=True)
    if gif_name:
        anim.save(gif_name, writer=PillowWriter(fps=fps), dpi=dpi)
        print("saved", gif_name)
    else:
        plt.show()
    return anim
