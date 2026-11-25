"""
UR5 analytic IK - visual step-by-step walkthrough
=================================================

Companion to IR Ch.07 (Case 2: three parallel axes) and ur5_ik_explained.py
(keep both files in the same folder).

    uv run ur5_ik_visual.py                       # interactive window
    uv run ur5_ik_visual.py --target 30 -60 80 -110 50 20   # target = FK of these angles [deg]
    uv run ur5_ik_visual.py --save steps          # write one PNG per step into ./steps
    uv run ur5_ik_visual.py --save figs --slides  # figures only (no text), for lecture slides

Controls
    Next / Prev buttons (or right / left arrow keys)  : move through the steps
    Shoulder / Wrist / Elbow buttons (or keys 1, 2, 3): switch IK branch
    theta1..theta6 sliders + Zero pose button, step 0 : move the robot by hand / back to theta = 0
    Arm plane button (or key p), step 1-1 only         : show / hide the arm plane in the wrist close-up
    theta5 slider, step 3-1 only                       : turn joint 5 from the zero pose

Steps
    0  Zero pose: read omega_i, q_i -> screw axes S_i, and M
    1-1 p5, d4, r : what is known before solving any angle        (three panels)
    1-2 theta1 : right triangle O-Q-p5 in the top view
    2  theta5 : p . z1 = d4 + d6 cos(theta5)                      (arm plane seen edge-on)
    3-1 theta5 = angle(z1, z6) : zero pose, turn joint 5           (slider)
    3-2 theta6 : z1 seen from the tool frame                     (arm plane + two moves of the wrist)
    4  theta2, theta3, theta4 : planar 3R arm                     (arm plane, face-on)
    5  All 8 solutions                                            (solution tree)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Arc, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ur5_ik_explained import (S, M, q, omega, d1, d4, d5, d6, L1, L2, fk, ik_ur5,
                              joint_positions, step1_theta1, step2_theta5,
                              step3_theta6, step4_planar, z1_of, wrap)

BLUE, ORANGE, RED, GRAY, DARK, GREEN = "#156082", "#D9622B", "#C00000", "#A0A4A2", "#1C1F1D", "#2E7D32"
D4C = "#8E24AA"   # color used for the offset d4 in every view
STEPS = ["0", "1-1", "1-2", "2", "3-1", "3-2", "4", "5"]
STEP_TITLES = [
    "Step 0  Zero pose: where the screw axes come from",
    "Step 1-1  What we know first: p5, d4 and r",
    "Step 1-2  theta1 from p5, d4 and r",
    "Step 2  theta5 from the offset along z1",
    "Step 3-1  theta5 is the angle between z1 and z6",
    "Step 3-2  theta6 from z1 seen in the tool frame",
    "Step 4  theta2, theta3, theta4: the hidden planar arm",
    "Step 5  All solutions for the same pose",
]
deg = np.degrees


# ---------------------------------------------------------------------------
# IK along one branch, keeping every intermediate value for plotting
# ---------------------------------------------------------------------------
def solve_branch(T, b_sh, b_wr, b_el):
    r = {"T": T, "p": T[:3, 3], "R": T[:3, :3], "ok": False}
    r["p5"] = r["p"] - d6 * T[:3, 2]
    r["t1s"] = step1_theta1(T)
    if not r["t1s"]:
        r["msg"] = "No solution: the wrist point is closer than d4 to the base axis."
        return r
    r["t1"] = t1 = r["t1s"][b_sh]
    r["z1"] = z1_of(t1)
    r["t5s"] = step2_theta5(T, t1)
    if not r["t5s"]:
        r["msg"] = "This shoulder branch cannot reach the pose (|cos theta5| > 1)."
        return r
    r["t5"] = t5 = r["t5s"][b_wr]
    r["t6"] = t6 = step3_theta6(T, t1, t5)
    pl = step4_planar(T, t1, t5, t6)
    if not pl:
        r["msg"] = "Out of reach for the planar arm (|cos theta3| > 1)."
        return r
    r["planar"] = pl
    t2, t3, t4 = pl[b_el]
    r["theta"] = np.array([t1, t2, t3, t4, t5, t6])
    r["ok"] = True
    return r


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_robot(ax, theta, color=BLUE, lw=4, alpha=1.0, joints=True):
    P = joint_positions(theta)
    ax.plot([0, 0], [0, 0], [-0.12, 0], color=GRAY, lw=10, alpha=alpha, solid_capstyle="butt")
    ax.plot(*P.T, "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    if joints:
        ax.scatter(*P[:6].T, s=28, color="white", edgecolor=color, lw=1.5, alpha=alpha, depthshade=False)
    return P


def draw_frame(ax, T, length=0.1, colors=(RED, GREEN, BLUE), label=None):
    o = T[:3, 3]
    for k in range(3):
        ax.quiver(*o, *(length * T[:3, k]), color=colors[k], lw=1.8, arrow_length_ratio=0.2)
    if label:
        ax.text(*(o + np.array([0, 0, 0.05])), label, color=DARK, fontsize=9)


def arm_plane(ax, t1, color=BLUE, alpha=0.10, pts=None, origin=True, pad=0.08):
    h = np.array([np.cos(t1), np.sin(t1), 0.0])        # in-plane horizontal direction (perp. to z1)
    if pts is None:
        s, z = np.linspace(-0.95, 0.95, 2), np.linspace(-0.25, 0.95, 2)
    else:                                               # only as large as the robot needs
        P = np.vstack([np.atleast_2d(p) for p in pts] + ([np.zeros((1, 3))] if origin else []))
        sp, zp = P @ h, P[:, 2]
        s = np.array([sp.min() - pad, sp.max() + pad]); z = np.array([zp.min() - pad, zp.max() + pad])
    Sg, Zg = np.meshgrid(s, z)
    ax.plot_surface(Sg * h[0], Sg * h[1], Zg, color=color, alpha=alpha, shade=False)


def d4_segment(ax, p5, t1, chosen=True):
    """Perpendicular from the arm plane (normal z1) to p5: its length is d4."""
    z1 = z1_of(t1)
    foot = p5 - (p5 @ z1) * z1
    ax.plot(*np.c_[foot, p5], color=D4C if chosen else GRAY, lw=4 if chosen else 1.5,
            alpha=1.0 if chosen else 0.6, solid_capstyle="butt")
    ax.scatter(*foot, color=D4C if chosen else GRAY, s=25 if chosen else 10)
    if chosen:
        ax.text(*((foot + p5) / 2 + np.array([0, 0, 0.04])), "d4", color=D4C, fontsize=12, fontweight="bold")
    return foot


def setup3d(ax):
    ax.cla()
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.3, 0.9)
    ax.set_box_aspect((1, 1, 0.75))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.tick_params(labelsize=7)


def zoom3d(ax, pts, pad=0.12, base=True):
    """Equal-aspect limits that fit the given points (keeps the robot large in the view)."""
    extra = [np.array([[0, 0, -0.12], [0, 0, 0.1]])] if base else []
    P = np.vstack([np.atleast_2d(p) for p in pts] + extra)
    lo, hi = P.min(0), P.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half); ax.set_box_aspect((1, 1, 1))


def ghost(ax, r):
    """Faint final robot of the chosen branch, for context in steps 1-3."""
    if r["ok"]:
        return draw_robot(ax, r["theta"], GRAY, 2.5, 0.35, joints=False)
    return np.zeros((1, 3))


def setup2d(ax, title):
    ax.cla()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, color=DARK, loc="left")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)


def angle_arc(ax, center, r, a0, a1, color, label, fs=10):
    a0d, a1d = deg(a0), deg(a1)
    lo, hi = min(a0d, a1d), max(a0d, a1d)
    ax.add_patch(Arc(center, 2 * r, 2 * r, theta1=lo, theta2=hi, color=color, lw=1.5))
    am = np.radians((lo + hi) / 2)
    ax.text(center[0] + 1.25 * r * np.cos(am), center[1] + 1.25 * r * np.sin(am), label,
            color=color, fontsize=fs, ha="center", va="center")


def bracket(ax, p0, p1, label, color=DARK, off=(0, 0)):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
    m = (np.array(p0) + np.array(p1)) / 2 + np.array(off)
    ax.text(*m, label, color=color, fontsize=10, ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


# ---------------------------------------------------------------------------
# One function per step: (3D view, 2D diagram, explanation text)
# ---------------------------------------------------------------------------
def step0(ax3, ax2, r, th_deg=None):
    th0 = np.zeros(6)
    th = np.zeros(6) if th_deg is None else np.radians(th_deg)
    moved = np.abs(th).max() > 1e-9
    # zero pose (gray) with the screw axes read in it; faint once the sliders move the robot
    P = draw_robot(ax3, th0, color=GRAY, lw=5, alpha=0.35 if moved else 1.0)
    for i in range(6):
        c = BLUE if i in (1, 2, 3) else DARK
        ax3.quiver(*(q[i] - 0.06 * omega[i]), *(0.16 * omega[i]), color=c, lw=2, arrow_length_ratio=0.25,
                   alpha=0.4 if moved else 1.0)
        ax3.text(*(q[i] + np.array([0, 0.04, 0.06])), f"q{i+1}", color=c, fontsize=9, alpha=0.5 if moved else 1.0)
    draw_frame(ax3, M, 0.08, label=None if moved else "M")
    pts = [P]
    if moved:   # the robot at the slider angles: T(theta) = e^[S1]t1 ... e^[S6]t6 M
        Pm = draw_robot(ax3, th, BLUE, 5)
        draw_frame(ax3, fk(th), 0.1, label="T(theta)")
        pts.append(Pm)
    ax3.view_init(22, -60); zoom3d(ax3, pts, 0.05)

    setup2d(ax2, "Side view at theta = 0: joints 2, 3, 4 share z = d1")
    xs, zs = P[:, 0], P[:, 2]
    ax2.plot([0, 0], [-0.06, 0], color=GRAY, lw=10)
    ax2.plot(xs, zs, color=GRAY, lw=6, solid_capstyle="round")
    ax2.axhline(d1, color=BLUE, ls="--", lw=1)
    for i in (1, 2, 3):
        ax2.plot(q[i][0], q[i][2], "o", ms=11, mfc="white", mec=BLUE, mew=2)
        ax2.text(q[i][0], q[i][2] + 0.035, f"q{i+1}", color=BLUE, ha="center")
    bracket(ax2, (0, -0.03), (-L1, -0.03), "L1 = |a2|", off=(0, -0.025))
    bracket(ax2, (-L1, -0.03), (-L1 - L2, -0.03), "L2 = |a3|", off=(0, -0.025))
    bracket(ax2, (0.05, 0), (0.05, d1), "d1", off=(0.04, 0))
    ax2.set_xlim(-0.95, 0.2); ax2.set_ylim(-0.12, 0.17)
    rows = "\n".join(f"S{i+1} = ({', '.join(f'{v:6.3f}' for v in S[i])})" for i in range(6))
    # the screw-axis table goes in the free space above the side view (bottom text area is for the sliders)
    ax2.text(0.53, 0.9, "Screw axes S_i = (omega_i, v_i), read at theta = 0:\n" + rows, transform=ax2.figure.transFigure,
             fontsize=9.5, family="monospace", va="top", color=DARK)
    head = ("At theta = 0 the arm is stretched along -x_s.\n"
            "Read omega_i and q_i in THIS pose, then v_i = -omega_i x q_i.\n"
            "Axes 2, 3, 4 (blue) are parallel: they all point along -y_s.\n")
    if moved:
        p = fk(th)[:3, 3]
        head = (f"T(theta) = e^[S1]t1 ... e^[S6]t6 M,  tool p = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})\n"
                "The screw axes S_i are still read at the zero pose (gray).\n"
                "Axes 2, 3, 4 stay parallel in every pose.\n")
    return head.rstrip("\n")


def wrist_part(r):
    """Joint points of the final pose from joint 4 on, plus a short stub toward joint 3 (for close-ups)."""
    P = joint_positions(r["theta"])
    stub = P[3] + 0.25 * (P[2] - P[3])
    return np.vstack([stub, P[3:]])


def step1_1(axA, ax3, ax2, r, plane=False):
    """Three panels: (1) p5 from p, (2) the fixed offset d4, (3) the horizontal distance r."""
    p, p5, z6 = r["p"], r["p5"], r["T"][:3, 2]
    rr, psi = np.hypot(*p5[:2]), np.arctan2(p5[1], p5[0])

    # (1) close-up of the wrist: step back d6 from p along z6
    axA.set_title("(1) p5 = p - d6 z6", fontsize=11, color=DARK, loc="left")
    W = wrist_part(r) if r["ok"] else np.vstack([p, p5])
    if r["ok"]:
        axA.plot(*W.T, "-", color=BLUE, lw=5, solid_capstyle="round")
        axA.scatter(*W[1:4].T, s=28, color="white", edgecolor=BLUE, lw=1.5, depthshade=False)
        for k, name in zip((1, 2, 3), ("4", "5", "6")):
            axA.text(*(W[k] + [0, 0, 0.02]), name, color=DARK, fontsize=8)
    if plane and "t1" in r:
        arm_plane(axA, r["t1"], BLUE, 0.15, pts=[W, p, p5], origin=False, pad=0.02)
    draw_frame(axA, r["T"], 0.04)
    axA.text(*(p + 0.05 * z6), "z6", color=BLUE, fontsize=10)
    axA.plot(*np.c_[p, p5], "--", color=ORANGE, lw=2.5)
    axA.text(*((p + p5) / 2 + [0, 0, 0.015]), "d6", color=ORANGE, fontsize=11, fontweight="bold")
    axA.scatter(*p, color=DARK, s=35, depthshade=False)
    axA.text(*(p + [0, 0, -0.03]), "p", color=DARK, fontsize=11, fontweight="bold")
    axA.scatter(*p5, color=ORANGE, s=50, depthshade=False)
    axA.text(*(p5 + [0, 0, -0.03]), "p5", color=ORANGE, fontsize=11)
    azim = deg(np.arctan2(z6[1], z6[0])) + 90 if np.hypot(*z6[:2]) > 1e-6 else -60
    axA.view_init(20, azim); zoom3d(axA, [W, p, p5], 0.04, base=False)

    if "t1" not in r:   # r < d4: no arm plane can be d4 away from p5
        ax3.text2D(0.5, 0.5, r.get("msg", ""), transform=ax3.transAxes, ha="center", color=RED, wrap=True)
        ax2.cla(); ax2.axis("off")
        return r.get("msg", "")

    # (2) the arm plane and the fixed offset d4
    ax3.set_title("(2) d4: p5 is always d4 off the arm plane", fontsize=11, color=DARK, loc="left")
    P = draw_robot(ax3, r["theta"], BLUE, 4) if r["ok"] else np.vstack([p, p5])
    arm_plane(ax3, r["t1"], BLUE, 0.15, pts=[P, p5])
    d4_segment(ax3, p5, r["t1"])
    ax3.scatter(*p5, color=ORANGE, s=50, depthshade=False)
    ax3.text(*(p5 + [0, 0, -0.07]), "p5", color=ORANGE)
    # r: horizontal distance from the base axis to p5 (drawn at p5's height), same r as in panel (3)
    ax3.plot([0, 0], [0, 0], [0, p5[2]], ":", color=DARK, lw=1)
    ax3.plot([0, p5[0]], [0, p5[1]], [p5[2], p5[2]], color=DARK, lw=1.5)
    ax3.text(*(np.array([p5[0], p5[1], 0]) * 0.5 + [0, 0, p5[2] + 0.03]), "r", color=DARK, fontsize=11)
    ax3.view_init(20, deg(r["t1"]) + 180 + 30); zoom3d(ax3, [P, p5])

    # (3) top view: r is the horizontal distance from the base axis to p5
    setup2d(ax2, "(3) Top view: r = |p5_xy|, the distance\nfrom the base axis (z_s) to p5")
    ax2.axhline(0, color=GRAY, lw=0.8); ax2.axvline(0, color=GRAY, lw=0.8)
    if r["ok"]:
        ax2.plot(*P[:, :2].T, color=BLUE, lw=3, alpha=0.35, solid_capstyle="round")
    ax2.plot(0, 0, "ko", ms=8); ax2.text(0.02, 0.03, "base axis", fontsize=8)
    ax2.plot([0, p5[0]], [0, p5[1]], color=DARK, lw=2)
    ax2.text(*(p5[:2] * 0.5 + [0.03, -0.05]), f"r = {rr:.3f}", fontsize=11, color=DARK)
    ax2.plot(*p5[:2], "o", color=ORANGE, ms=9); ax2.text(p5[0], p5[1] - 0.06, "p5", color=ORANGE, ha="center", va="top")
    angle_arc(ax2, (0, 0), 0.12, 0, psi, DARK, "psi")
    # d4: seen from above the arm plane is a line through the base; p5 sits d4 off it
    h = np.array([np.cos(r["t1"]), np.sin(r["t1"])])
    ax2.plot(*np.c_[-1.5 * h, 1.5 * h], "--", color=BLUE, lw=1.2)
    sh = p5[:2] @ h                                     # where p5 sits along the arm plane (seen from above)
    ang = deg(np.arctan2(h[1], h[0]))
    ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang          # keep the label upright
    ax2.text(*(0.3 * sh * h), "arm plane", color=BLUE, fontsize=8, rotation=ang, rotation_mode="anchor",
             ha="center", va="bottom")
    foot = sh * h
    ax2.plot(*np.c_[p5[:2], foot], color=D4C, lw=3)
    lab = (p5[:2] + foot) / 2 + 0.09 * np.sign(sh) * h
    ax2.text(*lab, f"d4 = {d4:.3f}", color=D4C, fontsize=10,
             fontweight="bold", ha="center", va="center")   # beyond the segment, away from the base and r
    pts2 = np.vstack([[0, 0], p5[:2], lab - [0.08, 0], lab + [0.08, 0]] + ([P[:, :2]] if r["ok"] else []))
    lo, hi = pts2.min(0), pts2.max(0); c, hw = (lo + hi) / 2, (hi - lo).max() / 2 + 0.15
    ax2.set_xlim(c[0] - hw, c[0] + hw); ax2.set_ylim(c[1] - hw, c[1] + hw)
    return (f"(1) p5 = p - d6 z6 = ({p5[0]:.3f}, {p5[1]:.3f}, {p5[2]:.3f}): step back d6 = {d6:.3f} from the tool point p "
            "along the tool z-axis\n    (p and z6 are both read from the target T, so p5 is known before any angle).\n"
            f"(2) d4 = {d4:.3f} is a fixed sideways offset: joints 2-4 move the arm only inside the arm plane, "
            "so p5 always stays d4 off it.\n"
            f"(3) r = dist((0, 0, p5z), p5) = sqrt(p5x^2 + p5y^2) = {rr:.3f}: horizontal distance from the base axis to p5.")


def step1_2(ax3, ax2, r):
    """theta1 from the right triangle O - Q - p5 in the top view (|OQ| = d4 along z1, |Op5| = r)."""
    if "t1" not in r:
        ax2.cla(); ax2.axis("off")
        return r.get("msg", "")
    p, p5 = r["p"], r["p5"]
    rr, psi = np.hypot(*p5[:2]), np.arctan2(p5[1], p5[0])
    alpha = np.arccos(d4 / rr)

    G = draw_robot(ax3, r["theta"], BLUE, 2.5, 0.8, joints=False) if r["ok"] else np.zeros((1, 3))
    for t1 in r["t1s"]:
        chosen = t1 == r["t1"]
        arm_plane(ax3, t1, BLUE if chosen else GRAY, 0.12 if chosen else 0.05, pts=[G, p, p5])
        d4_segment(ax3, p5, t1, chosen=chosen)
    ax3.scatter(*p5, color=ORANGE, s=50); ax3.text(*(p5 + [0, 0, -0.07]), "p5", color=ORANGE)
    ax3.plot([0, 0], [0, 0], [0, p5[2]], ":", color=DARK, lw=1)
    ax3.plot([0, p5[0]], [0, p5[1]], [p5[2], p5[2]], color=DARK, lw=1.5)
    ax3.text(*(np.array([p5[0], p5[1], 0]) * 0.5 + [0, 0, p5[2] + 0.03]), "r", color=DARK, fontsize=11)
    ax3.view_init(60, -60); zoom3d(ax3, [G, p, p5])

    setup2d(ax2, "Top view: right triangle O-Q-p5\n(hypotenuse r, side d4 along z1)")
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.plot(0, 0, "ko", ms=8); ax2.text(0.02, 0.03, "O", fontsize=10)
    ax2.plot(*p5[:2], "o", color=ORANGE, ms=9); ax2.text(p5[0] - 0.06, p5[1] - 0.03, "p5", color=ORANGE)
    ax2.plot([0, p5[0]], [0, p5[1]], color=DARK, lw=1.5)
    ax2.text(*(p5[:2] * 0.5 + [0.03, -0.04]), "r", fontsize=11, color=DARK)
    for t1 in r["t1s"]:
        chosen = t1 == r["t1"]
        h = np.array([np.cos(t1), np.sin(t1)])           # arm plane, seen from above
        n = np.array([np.sin(t1), -np.cos(t1)])          # z1 (normal of the arm plane)
        Q = d4 * n
        c = BLUE if chosen else GRAY
        ax2.plot(*np.c_[-1.5 * h, 1.5 * h], color=c, lw=2.5 if chosen else 1.2, ls="-" if chosen else "--",
                 label=f"arm plane, theta1 = {deg(t1):.1f} deg" + (" (chosen)" if chosen else " (other shoulder)"))
        if not chosen:
            continue
        ax2.plot(*np.c_[[0, 0], Q], color=D4C, lw=3)                      # O -> Q: length d4 along z1
        ax2.text(*(Q / 2 + [0.02, 0.0]), "d4", color=D4C, fontsize=11, fontweight="bold")
        ax2.annotate("", xy=Q + 0.07 * n, xytext=Q, arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
        ax2.text(*(Q + 0.07 * n + [0.015, 0]), "z1", color=GREEN, fontsize=10)
        ax2.plot(*np.c_[Q, p5[:2]], ":", color=BLUE, lw=1.5)               # Q -> p5 runs parallel to the plane
        ax2.text(*(Q - 0.035 * h + [0, -0.01]), "Q", fontsize=10, color=DARK, ha="right")
        sgn = np.sign(h @ (p5[:2] - Q)); m = 0.03                           # right-angle mark at Q
        ax2.plot(*np.c_[Q + m * sgn * h, Q + m * sgn * h + m * n, Q + m * n], color=DARK, lw=0.8)
        angle_arc(ax2, (0, 0), 0.16, psi, t1 - np.pi / 2, D4C, "acos(d4/r)", fs=9)
        angle_arc(ax2, (0, 0), 0.26, 0, psi, DARK, "psi", fs=9)
        angle_arc(ax2, (0, 0), 0.20, 0, t1, BLUE, "theta1", fs=10)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    xs = [0, p5[0]]; ys = [0, p5[1]]
    cx, cy, hw = np.mean(xs), np.mean(ys), max(np.ptp(xs), np.ptp(ys)) / 2 + 0.3
    ax2.set_xlim(cx - hw, cx + hw); ax2.set_ylim(cy - hw, cy + hw)
    t1a, t1b = r["t1s"]
    return ("Seen from above, the arm plane is a line through the base at angle theta1; z1 is perpendicular to it.\n"
            "p5 is d4 off that line, so O-Q-p5 is a right triangle with hypotenuse r and side d4 along z1:\n"
            f"   angle(p5, z1) = acos(d4/r) = acos({d4:.3f}/{rr:.3f}) = {deg(alpha):.1f} deg,"
            f"   psi = atan2(p5y, p5x) = {deg(psi):.1f} deg\n"
            f"theta1 = psi + pi/2 +/- acos(d4/r) = {deg(t1a):.1f} or {deg(t1b):.1f} deg   "
            "(z1 on either side of p5: shoulder left / right; no solution if r < d4)")


def step2(ax3, ax2, r):
    p, p5, z1, z6 = r["p"], r["p5"], r["z1"], r["T"][:3, 2]
    G = ghost(ax3, r)
    foot0 = p5 - (p5 @ z1) * z1
    ext = p5 + ((p - p5) @ z1) * z1                      # the tool adds d6 cos(theta5) along z1
    footp = p - (p @ z1) * z1                            # p projected onto the arm plane
    box = [p, p5, foot0, ext, footp]
    arm_plane(ax3, r["t1"], pts=box)
    # p . z1: how far the tool point p is from the arm plane, measured along z1
    ax3.plot(*np.c_[footp, p], color=DARK, lw=2)
    ax3.scatter(*np.c_[footp, p], color=DARK, s=18, depthshade=False)
    ax3.text(*((footp + p) / 2 + np.array([0, 0, 0.015])), f"p . z1 = {p @ z1:.3f}", color=DARK, fontsize=10)
    ax3.text(*(p + np.array([0, 0, 0.015])), "p", color=DARK, fontsize=11, fontweight="bold")
    draw_frame(ax3, r["T"], 0.05)
    up = np.array([0, 0, -0.035])
    ax3.quiver(*(foot0 + up), *(0.07 * z1), color=GREEN, lw=2.5, arrow_length_ratio=0.3)
    ax3.text(*(foot0 + up + 0.04 * z1 + np.array([0, 0, -0.03])), "z1", color=GREEN, fontsize=11)
    ax3.plot(*np.c_[p5, p], color=ORANGE, lw=3)
    ax3.scatter(*p5, color=ORANGE, s=40); ax3.text(*(p5 + np.array([0, 0, 0.02])), "p5", color=ORANGE)
    d4_segment(ax3, p5, r["t1"])
    ax3.plot(*np.c_[p5, ext], color=ORANGE, lw=2, ls="--")
    ax3.text(*(ext + np.array([0, 0, -0.04])), "d6 cos(theta5)", color=ORANGE, fontsize=10)
    # close-up of the wrist so that d4 is easy to see (the faint robot gives context)
    ax3.view_init(15, deg(r["t1"]) + 25); zoom3d(ax3, box, 0.08, base=False)

    setup2d(ax2, "Arm plane seen edge-on (horizontal axis = distance along z1)")
    e1 = z1
    perp = z6 - (z6 @ e1) * e1
    e2 = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-9 else np.array([0, 0, 1.0])
    P5 = np.array([d4, 0.0])
    tool = P5 + d6 * np.array([z6 @ e1, z6 @ e2])
    ax2.axvline(0, color=BLUE, lw=4); ax2.text(0.004, 0.11, "arm plane", color=BLUE, rotation=90, fontsize=9)
    ax2.annotate("", xy=tool, xytext=P5, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
    ax2.plot(*P5, "o", color=ORANGE, ms=8); ax2.text(P5[0] - 0.012, P5[1] - 0.012, "p5", color=ORANGE, ha="right")
    ax2.text(*(tool + [0.005, 0.008]), "p (tool)", color=ORANGE)
    mirror = P5 + d6 * np.array([z6 @ e1, -(z6 @ e2)])
    ax2.annotate("", xy=mirror, xytext=P5, arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2, ls="--"))
    ax2.text(*(mirror + [0.005, -0.012]), "-theta5 (flipped)", color=GRAY, fontsize=8)
    ax2.plot([P5[0], P5[0] + 0.1], [0, 0], color=GREEN, lw=1, ls=":")
    angle_arc(ax2, P5, 0.035, 0, np.arctan2(tool[1] - P5[1], tool[0] - P5[0]), RED, "theta5")
    bracket(ax2, (0, -0.07), (d4, -0.07), "d4", color=D4C, off=(0, -0.012))
    ax2.plot([0, d4], [0, 0], color=D4C, lw=3)
    bracket(ax2, (d4, -0.07), (tool[0], -0.07), "d6 cos theta5", off=(0, -0.012))
    lo = min(0, tool[0], mirror[0]) - 0.03
    ax2.set_xlim(lo, max(0.25, tool[0] + 0.06)); ax2.set_ylim(-0.11, 0.13)
    return ("Joints 2, 3, 4 rotate about z1, so they cannot change how far the tool is along z1. Only theta5 can:\n"
            f"p . z1 = d4 + d6 cos(theta5)  ->  {p @ z1:.4f} = {d4:.4f} + {d6:.4f} cos(theta5)\n"
            f"theta5 = +/- {deg(abs(r['t5'])):.1f} deg   (wrist up / flipped).  Chosen: {deg(r['t5']):.1f} deg")


TOOL_COLORS = (RED, "#9CCC65", BLUE)   # step 3: light-green y6 so it is not confused with z1
VARIANTS = [(0, 0, 0), (25, -35, 15), (-30, 25, 35), (15, 30, -50)]   # offsets [deg] added to theta2, 3, 4


def step3_1(ax3, ax2, r, t5_deg=None):
    """Zero pose with only theta5 turned: the angle between z1 and z6 is exactly theta5."""
    t5 = np.radians(t5_deg) if t5_deg is not None else r.get("t5", 0.0)
    th = np.array([0, 0, 0, 0, t5, 0.0])
    P = draw_robot(ax3, th, BLUE, 5)
    T = fk(th); o = T[:3, 3]; z6 = T[:3, 2]
    z1 = omega[1] / np.linalg.norm(omega[1])                 # axis of joints 2, 3, 4 (theta1 = 0)
    w5 = omega[4] / np.linalg.norm(omega[4])                 # joint 5 axis (theta1..4 = 0, so unmoved)
    L = 0.25
    # z1 at joint 4 (its own axis) and a copy at the tool, z6 at the tool, joint 5 axis at joint 5
    ax3.quiver(*P[3], *(L * z1), color=GREEN, lw=3, arrow_length_ratio=0.2)
    ax3.text(*(P[3] + 1.15 * L * z1 + [0, 0, 0.02]), "z1", color=GREEN, fontsize=12, fontweight="bold")
    ax3.plot(*np.c_[o, o + L * z1], "--", color=GREEN, lw=1.5)
    ax3.quiver(*o, *(L * z6), color=BLUE, lw=3, arrow_length_ratio=0.2)
    ax3.text(*(o + 1.15 * L * z6 + [0, 0, 0.02]), "z6", color=BLUE, fontsize=12, fontweight="bold")
    ax3.quiver(*P[4], *(0.18 * w5), color=ORANGE, lw=2.5, arrow_length_ratio=0.25)
    ax3.text(*(P[4] + 0.2 * w5 + [0.02, 0, 0]), "joint 5 axis", color=ORANGE, fontsize=9)
    ang = np.arccos(np.clip(z1 @ z6, -1, 1))
    if ang > 1e-3:                                           # arc from z1 to z6 at the tool
        u = z6 - (z6 @ z1) * z1; u /= np.linalg.norm(u)
        ss = np.linspace(0, ang, 30)
        arc = o + 0.6 * L * (np.outer(np.cos(ss), z1) + np.outer(np.sin(ss), u))
        ax3.plot(*arc.T, color=RED, lw=2); ax3.text(*(arc[15] + [0, 0, 0.03]), "theta5", color=RED, fontsize=10)
    ax3.view_init(22, -60); zoom3d(ax3, [P, o + L * z1, o + L * z6], 0.05)

    # top view = looking down the joint 5 axis (-z_s): z1 and z6 both lie flat, so the true angle shows
    setup2d(ax2, "Top view of the wrist = looking down the joint 5 axis:\nz1 and z6 lie flat, the angle between them is theta5")
    ax2.plot(*P[:, :2].T, color=BLUE, lw=4, solid_capstyle="round", alpha=0.5)
    ax2.plot(0, 0, "ko", ms=7); ax2.text(0.02, 0.02, "base", fontsize=8)
    o2, l2 = o[:2], 0.15
    ax2.plot(*P[3:5, :2].T, "o", ms=8, mfc="white", mec=BLUE, mew=2)
    ax2.text(*(P[3, :2] + [0.012, 0.012]), "joint 4", fontsize=8); ax2.text(*(P[4, :2] + [0.012, 0.012]), "joint 5", fontsize=8)
    ax2.annotate("", o2 + l2 * z1[:2], o2, arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3))
    ax2.text(*(o2 + 1.15 * l2 * z1[:2] + [-0.03, 0]), "z1", color=GREEN, fontsize=12, fontweight="bold", ha="right")
    ax2.annotate("", o2 + l2 * z6[:2], o2, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3))
    ax2.text(*(o2 + 1.15 * l2 * z6[:2] + [0.03, 0]), "z6", color=BLUE, fontsize=12, fontweight="bold", ha="left")
    a1, a6 = np.arctan2(z1[1], z1[0]), np.arctan2(z6[1], z6[0])
    if ang > 1e-3:
        d = (a6 - a1 + np.pi) % (2 * np.pi) - np.pi
        angle_arc(ax2, tuple(o2), 0.07, a1, a1 + d, RED, f"theta5 = {deg(t5):.0f} deg", fs=10)
    pts = np.vstack([P[3:, :2], o2 + 1.3 * l2 * np.array([z1[:2], z6[:2]])])      # zoom on the wrist
    lo, hi = pts.min(0), pts.max(0); c, hw = (lo + hi) / 2, max((hi - lo).max() / 2, 0.15) + 0.06
    ax2.set_xlim(c[0] - hw, c[0] + hw); ax2.set_ylim(c[1] - hw, c[1] + hw)
    tgt = f"  (the target pose needs theta5 = {deg(r['t5']):.1f} deg)" if "t5" in r else ""
    return ("At theta = 0 (Step 0): joints 2, 3, 4 (z1) and joint 6 (z6) all point along -y_s, so z1 // z6 when theta5 = 0.\n"
            "Joint 5's axis (-z_s) is perpendicular to both, so turning it swings z6 away from z1 by exactly theta5.\n"
            "Joints 1-4 turn z1 and z6 together and joint 6 spins about z6 itself: none of them change this angle.\n"
            f"angle(z1, z6) = theta5:   z1 . z6 = cos theta5 = {z1 @ z6:.3f}   (theta5 = {deg(t5):.1f} deg){tgt}")


def step3_2(ax3, axA, axB, r):
    """Left: z1 is the arm-plane normal (theta2-4 cannot move it). Right (3D): z1 in the tool frame."""
    th, z1 = r["theta"], r["z1"]
    # the target pose and its arm plane: joints 2-4 move the arm only inside this plane
    P = draw_robot(ax3, th, BLUE, 4)
    arm_plane(ax3, th[0], BLUE, 0.15, pts=[P])
    # z1 = normal of the arm plane (= axis of joints 2, 3, 4), drawn at joint 4
    # (the link joint 4 -> joint 5 runs along +z1 itself, so the arrow is drawn on the other side of joint 4)
    j4 = P[3]
    ax3.plot(*np.c_[j4 - 0.3 * z1, j4 + 0.2 * z1], "--", color=GREEN, lw=1.2)          # joint-4 axis
    ax3.quiver(*(j4 - 0.3 * z1), *(0.25 * z1), color=GREEN, lw=3, arrow_length_ratio=0.25)
    ax3.text(*(j4 - 0.34 * z1 + [0, 0, 0.03]), "z1 (axis of joints 2, 3, 4)", color=GREEN, fontsize=10, fontweight="bold")
    ax3.text(*(j4 + [0, 0, -0.06]), "joint 4", color=DARK, fontsize=8)
    # tool frame and its x6-y6 plane (the plane the right panel projects z1 onto)
    T = r["T"]; o, x6, y6 = T[:3, 3], T[:3, 0], T[:3, 1]
    s = 0.09
    sq = np.array([o + a * x6 + b * y6 for a, b in [(-s, -s), (s, -s), (s, s), (-s, s), (-s, -s)]])
    ax3.plot_trisurf(*sq[:4].T, color=ORANGE, alpha=0.25, shade=False)
    ax3.plot(*sq.T, color=ORANGE, lw=1)
    draw_frame(ax3, T, 0.1, colors=TOOL_COLORS)
    ax3.text(*(o + 0.12 * x6), "x6", color=RED, fontsize=9); ax3.text(*(o + 0.12 * y6), "y6", color="#7CB342", fontsize=9)
    ax3.text(*(o + 0.12 * T[:3, 2]), "z6", color=BLUE, fontsize=9)
    ax3.text2D(0.02, 0.95, "the arm only moves inside the arm plane, so its normal z1 never changes;\n"
               "orange: the tool's x6-y6 plane (panels a, b)",
               transform=ax3.transAxes, fontsize=8, color=DARK, va="top")
    ax3.view_init(20, deg(th[0]) - 60); zoom3d(ax3, [P, sq, j4 - 0.3 * z1])

    # right: z1 seen with joint 6's axis z6 pointing up, in two moves
    t5, t6 = th[4], th[5]
    v = np.array([np.sin(t5), 0.0, np.cos(t5)])     # z1 after joint 5; joint 6 does not move it
    tool_view(axA, v, 0.0, "(a) theta6 = 0: joint 5 tilts z1 by theta5\n     toward x6, so its shadow lies along x6")
    tool_view(axB, v, t6, "(b) joint 6 turns x6, y6 by theta6 about z6;\n     z1 and its shadow do not move")
    return ("Left: joints 2-4 only move the arm inside the arm plane, so its normal z1 is fixed by theta1 alone.\n"
            "(a) theta6 = 0: joint 5 tilts z1 away from z6 by theta5, toward x6 -> height cos theta5, shadow sin theta5 along x6.\n"
            f"(b) joint 6 turns only the tool: x6, y6 rotate by theta6 = {deg(t6):.1f} deg, z1 stays. Measured from the new x6,\n"
            "    the shadow is at -theta6:  x6.z1 = s5 cos theta6,  y6.z1 = -s5 sin theta6,  z6.z1 = cos theta5\n"
            f"z1 (from theta1), x6, y6 (from the target) and theta5 give  theta6 = atan2(-y6.z1 / s5, x6.z1 / s5) = {deg(t6):.1f} deg"
            "   (undefined if sin theta5 = 0)")


def tool_view(ax, v, t6, title):
    """z1 (fixed vector v) with z6 up; x6, y6 turned by t6 about z6. Gray dashed: x6, y6 at theta6 = 0."""
    ax.set_title(title, fontsize=10, color=DARK, loc="left")
    x6 = np.array([np.cos(t6), np.sin(t6), 0.0]); y6 = np.array([-np.sin(t6), np.cos(t6), 0.0])
    if abs(t6) > 1e-9:
        for e in ((1, 0, 0), (0, 1, 0)):
            ax.plot(*np.c_[np.zeros(3), e], "--", color=GRAY, lw=1.2)
    for e, c, name in [(x6, RED, "x6"), (y6, "#7CB342", "y6"), (np.array([0, 0, 1.0]), BLUE, "z6")]:
        ax.quiver(0, 0, 0, *e, color=c, lw=2, arrow_length_ratio=0.08)
        ax.text(*(1.1 * e), name, color=c, fontsize=11)
    a = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(a), np.sin(a), 0 * a, ":", color=GRAY, lw=1)
    ax.quiver(0, 0, 0, *v, color=GREEN, lw=3, arrow_length_ratio=0.08)
    ax.text(*(1.1 * v + [0, 0, 0.05]), "z1", color=GREEN, fontsize=13, fontweight="bold")
    foot = np.array([v[0], v[1], 0.0])
    ax.plot(*np.c_[np.zeros(3), foot], color=DARK, lw=2.5)                    # shadow, length |sin theta5|
    ax.plot(*np.c_[foot, v], "--", color=DARK, lw=1.2)                        # height, cos theta5
    if abs(t6) < 1e-9:
        ax.text(*(foot / 2 + [0, -0.1, -0.12]), "sin theta5", color=DARK, fontsize=9, ha="center")
        ax.text(*(foot + [0.05, 0, v[2] / 2]), "cos theta5", color=DARK, fontsize=9)
        tt = np.linspace(0, np.arctan2(v[0], v[2]), 30)                      # theta5 arc from z6 to z1
        arc = 0.35 * np.c_[np.sin(tt), 0 * tt, np.cos(tt)]
        ax.plot(*arc.T, color=RED, lw=1.5); ax.text(*(arc[15] * 1.3), "theta5", color=RED, fontsize=10)
    else:
        pp = np.linspace(0, t6, 30)                                          # theta6: old x6 -> new x6
        ax.plot(0.6 * np.cos(pp), 0.6 * np.sin(pp), 0 * pp, color=ORANGE, lw=1.8)
        ax.text(0.7 * np.cos(t6 / 2), 0.7 * np.sin(t6 / 2), 0, "theta6", color=ORANGE, fontsize=10)
        ax.text(*(foot * 0.5 + [0, -0.12, -0.08]), "shadow: -theta6 from the new x6", color=DARK, fontsize=8, ha="center")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-0.3, 1.1); ax.set_box_aspect((1, 1, 0.7))
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel(""); ax.tick_params(labelsize=6)
    ax.view_init(28, -100)


def step4(ax3, ax2, r):
    th = r["theta"]
    G = draw_robot(ax3, th, BLUE, 5)
    arm_plane(ax3, th[0], pts=[G])
    for k, (t2, t3, t4) in enumerate(r["planar"]):
        th2 = np.array([th[0], t2, t3, t4, th[4], th[5]])
        if not np.allclose(th2, th):
            draw_robot(ax3, th2, GRAY, 2, 0.6, joints=False)
    draw_frame(ax3, r["T"], 0.1)
    ax3.view_init(10, deg(th[0]) - 90); zoom3d(ax3, [G])

    setup2d(ax2, "Arm plane, face-on: a planar arm with links L1, L2")
    def pts(t2, t3, t4):
        R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        j3 = R(t2) @ np.array([-L1, 0])
        j4 = j3 + R(t2 + t3) @ np.array([-L2, 0])
        j6 = j4 + R(t2 + t3 + t4) @ np.array([0, -d5])
        return np.array([[0, 0], j3, j4, j6])
    for (t2, t3, t4) in r["planar"]:
        P = pts(t2, t3, t4)
        chosen = np.isclose(t3, th[2])
        ax2.plot(*P.T, color=BLUE if chosen else GRAY, lw=4 if chosen else 1.5,
                 linestyle="-" if chosen else "--", marker="o", mfc="white", ms=8)
        if chosen:
            for k, (name, o) in enumerate(zip(["joint 2", "joint 3", "joint 4", "joint 6"],
                                              [(0.03, -0.05), (0.03, 0.04), (-0.03, 0.05), (0.03, -0.03)])):
                ax2.text(*(P[k] + o), name, fontsize=9, color=DARK, ha="left" if o[0] > 0 else "right")
            angle_arc(ax2, P[0], 0.1, np.pi, np.pi + th[1], RED, "theta2")
            a_prev = np.pi + th[1]
            angle_arc(ax2, P[1], 0.08, a_prev, a_prev + th[2], RED, "theta3")
    u = pts(*r["planar"][0])[2]
    ax2.plot([0, u[0]], [0, u[1]], ":", color=DARK)
    ax2.text(*(u / 2 + [0, -0.05]), f"|u| = {np.linalg.norm(u):.3f}", fontsize=9)
    allp = np.vstack([pts(*x) for x in r["planar"]])
    lo, hi = allp.min(0), allp.max(0); c, hw = (lo + hi) / 2, (hi - lo).max() / 2 + 0.15
    ax2.set_xlim(c[0] - hw, c[0] + hw); ax2.set_ylim(c[1] - hw, c[1] + hw)
    t2, t3, t4 = th[1:4]
    return ("Peel the known joints off both ends of the PoE chain:\n"
            "   e^[S2]t2 e^[S3]t3 e^[S4]t4 = e^-[S1]t1 T M^-1 e^-[S6]t6 e^-[S5]t5   (all known)\n"
            f"Joint 4 relative to joint 2 gives |u| -> law of cosines -> theta3 = +/- {deg(abs(t3)):.1f} deg (elbow up/down)\n"
            f"then theta2 = {deg(t2):.1f}, theta4 = phi - theta2 - theta3 = {deg(t4):.1f} deg")


def step5(ax3, ax2, r, sols, branch):
    G = [draw_robot(ax3, s, GRAY, 1.5, 0.5, joints=False) for s in sols]
    if r["ok"]:
        draw_robot(ax3, r["theta"], BLUE, 5)
    draw_frame(ax3, r["T"], 0.1)
    ax3.view_init(25, -60); zoom3d(ax3, G)

    ax2.cla(); ax2.axis("off")
    ax2.set_title("Solution tree: 2 x 2 x 1 x 2 = 8", fontsize=11, color=DARK, loc="left")
    ax2.set_xlim(0, 4.3); ax2.set_ylim(-0.5, 9.2)
    levels = ["theta1\n(shoulder)", "theta5\n(wrist)", "theta6", "theta3\n(elbow)"]
    for k, l in enumerate(levels):
        ax2.text(k + 0.6, 7.9, l, ha="center", va="center", fontsize=9, color=DARK)
    leaf = 0
    for i1 in range(2):
        y1 = 5.5 - 4 * i1
        for i5 in range(2):
            y5 = y1 + 1 - 2 * i5
            for i3 in range(2):
                y3 = y5 + 0.5 - i3
                on = (i1, i5, i3) == tuple(branch)
                c, lw = (BLUE, 3) if on else (GRAY, 1)
                ax2.plot([0.6, 1.6, 2.6, 3.6], [y1, y5, y5, y3], "-o", color=c, lw=lw, ms=5, zorder=3 if on else 1)
                ax2.text(3.75, y3, f"#{leaf+1}", va="center", fontsize=9, color=c, fontweight="bold" if on else None)
                leaf += 1
    return ("Every target pose has up to 8 joint solutions: shoulder (2) x wrist (2) x elbow (2).\n"
            "Use the Shoulder / Wrist / Elbow buttons (keys 1, 2, 3) to walk through them.\n"
            "A real controller picks one by joint limits or by closeness to the current joint angles.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class Walkthrough:
    def __init__(self, T, interactive=True, compact=False):
        self.T, self.step, self.branch, self.compact = T, 0, [0, 0, 0], compact
        self.plane = False   # step 1-1: show the arm plane in the wrist close-up
        self.sols = ik_ur5(T)
        # axes rectangles: two panels (3D + 2D) for most steps, three for step 1-1
        if compact:   # figures only, for lecture slides
            self.fig = plt.figure(figsize=(13, 5.2))
            self.lay2 = dict(ax3=[0.0, 0.0, 0.46, 1.0], ax2=[0.52, 0.07, 0.46, 0.85])
            self.lay31 = dict(ax3=[0.0, 0.0, 0.36, 1.0], axA=[0.375, 0.0, 0.31, 0.95], axB=[0.685, 0.0, 0.31, 0.95])
            self.lay3 = dict(axR=[0.02, 0.0, 0.31, 0.95], ax3=[0.32, 0.0, 0.33, 1.0], ax2=[0.68, 0.07, 0.31, 0.85])
        else:
            self.fig = plt.figure(figsize=(14, 8))
            self.lay2 = dict(ax3=[0.02, 0.27, 0.46, 0.66], ax2=[0.53, 0.30, 0.44, 0.60])
            self.lay31 = dict(ax3=[0.0, 0.27, 0.36, 0.66], axA=[0.375, 0.27, 0.3, 0.64], axB=[0.685, 0.27, 0.3, 0.64])
            self.lay3 = dict(axR=[0.02, 0.27, 0.31, 0.64], ax3=[0.345, 0.27, 0.32, 0.66], ax2=[0.68, 0.30, 0.31, 0.60])
        self.ax3 = self.fig.add_axes(self.lay2["ax3"], projection="3d")
        self.ax2 = self.fig.add_axes(self.lay2["ax2"])
        self.axR = self.fig.add_axes(self.lay3["axR"], projection="3d")
        self.axS = self.fig.add_axes(self.lay31["axB"], projection="3d"); self.axS.set_visible(False)
        self.txt = self.fig.text(0.03, 0.20, "", fontsize=10, family="monospace", va="top", wrap=True)
        self.head = self.fig.text(0.03, 0.96, "", fontsize=15, fontweight="bold", color=DARK)
        self.fig._walkthrough = self   # the window keeps this object (and its buttons) alive
        self.plane_btn = None
        self.t5_slider = None   # step 3-1: theta5 [deg] on the zero pose
        self.joint_sliders, self.zero_btn = [], None   # step 0: theta1..theta6 [deg], back to zero
        if interactive:
            # our keys must not also trigger matplotlib's toolbar shortcuts ('p' = pan mode, arrows = view history)
            for name in ("keymap.pan", "keymap.back", "keymap.forward"):
                plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ("p", "left", "right")]
            self.plane_btn = Button(self.fig.add_axes([0.41, 0.02, 0.09, 0.05]), "Arm plane")
            self.plane_btn.on_clicked(self.toggle_plane)
            self.t5_slider = Slider(self.fig.add_axes([0.62, 0.215, 0.3, 0.03]), "rotate theta5 [deg]", -180, 180, valinit=0)
            self.t5_slider.on_changed(lambda val: self.draw())
            # step 0: one slider per joint, and a button back to the zero pose
            self.joint_sliders = []
            for i in range(6):
                sl = Slider(self.fig.add_axes([0.62, 0.255 - 0.03 * i, 0.3, 0.02]), f"theta{i+1} [deg]",
                            -180, 180, valinit=0)
                sl.on_changed(lambda val: self.draw()); self.joint_sliders.append(sl)
            self.zero_btn = Button(self.fig.add_axes([0.41, 0.02, 0.09, 0.05]), "Zero pose")
            self.zero_btn.on_clicked(self.zero_pose)
            self.btns = []
            for x, label, cb in [(0.53, "< Prev", self.prev), (0.62, "Next >", self.next),
                                 (0.74, "Shoulder", lambda e: self.flip(0)),
                                 (0.83, "Wrist", lambda e: self.flip(1)),
                                 (0.92, "Elbow", lambda e: self.flip(2))]:
                b = Button(self.fig.add_axes([x, 0.02, 0.075, 0.05]), label)
                b.on_clicked(cb); self.btns.append(b)
            self.fig.canvas.mpl_connect("key_press_event", self.key)
        self.draw()

    def prev(self, e=None): self.step = max(0, self.step - 1); self.draw()
    def next(self, e=None): self.step = min(len(STEPS) - 1, self.step + 1); self.draw()
    def flip(self, k): self.branch[k] ^= 1; self.draw()
    def toggle_plane(self, e=None): self.plane = not self.plane; self.draw()

    def zero_pose(self, e=None):
        for sl in self.joint_sliders:          # reset silently, then redraw once
            sl.eventson = False; sl.set_val(0); sl.eventson = True
        self.draw()

    def key(self, e):
        if e.key == "right": self.next()
        elif e.key == "left": self.prev()
        elif e.key in ("1", "2", "3"): self.flip(int(e.key) - 1)
        elif e.key == "p": self.toggle_plane()

    def draw(self):
        r = solve_branch(self.T, *self.branch)
        setup3d(self.ax3)
        s, key = self.step, STEPS[self.step]
        three = key == "1-1"
        lay = self.lay3 if three else self.lay2
        self.ax3.set_position(lay["ax3"]); self.ax2.set_position(lay["ax2"])
        tool3d = key == "3-2" and r["ok"]          # step 3-2: two 3D panels (axR, axS) instead of ax2
        self.axR.set_visible(three or tool3d); self.axS.set_visible(tool3d); self.ax2.set_visible(not tool3d)
        self.axR.set_position(self.lay31["axA"] if tool3d else self.lay3["axR"])
        if tool3d:
            self.ax3.set_position(self.lay31["ax3"])
        # step-specific widgets: hidden widgets must also be inactive, or they still catch clicks
        for wdg, on in [(self.plane_btn, three), (self.t5_slider, key == "3-1"), (self.zero_btn, key == "0")] + \
                       [(sl, key == "0") for sl in self.joint_sliders]:
            if wdg is not None:
                wdg.ax.set_visible(on); wdg.set_active(on)
        if key not in ("0", "1-1", "1-2", "3-1") and not r["ok"]:
            self.ax2.cla(); self.ax2.axis("off")
            msg = r.get("msg", "No solution on this branch.") + "\nTry another branch."
        elif key == "0":
            msg = step0(self.ax3, self.ax2, r, [sl.val for sl in self.joint_sliders] if self.joint_sliders else None)
        elif key == "1-1":
            setup3d(self.axR); msg = step1_1(self.axR, self.ax3, self.ax2, r, self.plane)
        elif key == "1-2": msg = step1_2(self.ax3, self.ax2, r)
        elif key == "2": msg = step2(self.ax3, self.ax2, r)
        elif key == "3-1":
            msg = step3_1(self.ax3, self.ax2, r, None if self.t5_slider is None else self.t5_slider.val)
        elif key == "3-2":
            setup3d(self.axR); setup3d(self.axS); msg = step3_2(self.ax3, self.axR, self.axS, r)
        elif key == "4": msg = step4(self.ax3, self.ax2, r)
        else: msg = step5(self.ax3, self.ax2, r, self.sols, self.branch)
        names = (["A", "B"][self.branch[0]], ["+", "-"][self.branch[1]], ["up", "down"][self.branch[2]])
        tag = f"   [branch: shoulder {names[0]}, wrist {names[1]}, elbow {names[2]}]"
        if r["ok"]:
            tag += "   theta = (" + ", ".join(f"{deg(a):.1f}" for a in r["theta"]) + ") deg"
        self.head.set_text("" if self.compact else STEP_TITLES[s])
        self.txt.set_text("" if self.compact else msg + ("" if key == "0" else "\n" + tag))   # step 0: no IK branch yet
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=6, default=[30, -60, 80, -110, 50, 20],
                    help="joint angles [deg] whose FK pose is used as the IK target")
    ap.add_argument("--save", metavar="DIR", help="save one PNG per step instead of opening a window")
    ap.add_argument("--slides", action="store_true", help="with --save: figures only (no header/text), for slides")
    args = ap.parse_args()
    T = fk(np.radians(args.target))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        w = Walkthrough(T, interactive=False, compact=args.slides)
        for s, key in enumerate(STEPS):
            w.step = s; w.draw()
            w.fig.savefig(os.path.join(args.save, f"ur5_ik_step{key}.png"), dpi=150)
        print(f"Saved {len(STEPS)} images to {args.save}/")
    else:
        w = Walkthrough(T)   # keep a reference: matplotlib holds button callbacks only weakly
        plt.show()


if __name__ == "__main__":
    main()
