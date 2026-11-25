"""
A6_anim_three_methods_scenarios.py  --  When does each "straight line" win or lose?

A UR5 moves its gripper between two poses with the same quintic time scaling, in three ways:
   joint-space line   theta(s) = theta_start + s (theta_end - theta_start)
   screw path         X(s) = X_start exp(log(X_start^-1 X_end) s)
   decoupled path     p(s) straight,  R(s) = R_start exp(log(R_start^T R_end) s)

Pick a SCENARIO:
  1 "corridor"     : the gripper must stay near the straight line (a box sits just beside it).
                     joint-space line swings far off the line and enters the box; decoupled stays on it.
  2 "singularity"  : the straight line passes close to the robot's base, near a singular configuration.
                     decoupled path needs about 4x the joint speed; joint-space line stays gentle.
  3 "big rotation" : the gripper turns 170 deg. The screw path's origin bulges ~0.27 m off the line
                     and hits a box placed there; the decoupled path stays on the line.
  4 "joint limits" : both end configurations are inside the joint limits (range of motion), but the
                     decoupled path needs the elbow to fold to 158 deg and the shoulder to lean back to
                     -130 deg on the way -> it EXCEEDS the limits. The joint-space line can never do this:
                     the box of joint limits is convex, so a straight line between two points inside it
                     stays inside. (The limits here are example software limits, e.g. to avoid
                     self-collision; the real UR5 joints can turn +-360 deg.)

The gripper turns RED while it is inside a box (only the gripper point is checked, not the links).
In scenario 4 the whole arm turns RED while a joint is outside its limits.
Bottom row: joint speeds |theta_dot_i| (scenarios 1-3) or the limited joint angles (scenario 4).

Summary
  joint-space line : + no IK needed, smooth and gentle joint motion, no trouble near singularities
                     - gripper path is not controlled: it can swing far from the line
  screw path       : + one constant screw motion; "natural" in SE(3)
                     - origin path bulges with large rotations; IK needed; can pass near singularities
  decoupled path   : + gripper origin exactly on the straight line (corridors, insertion, wiping)
                     - IK needed; joint speeds blow up near singularities; can leave the joint limits

Try this: change the scenario, T, or the poses in SCENARIOS and watch the numbers in the panels.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu
from robots import UR5
from anim_utils import UR5Artist, setup_3d, run

SCENARIO = 1
T, FPS = 3.0, 20
SLOWMO = 1                  # playback SLOWMO times slower than real time (e.g. 3)
GIF = None

# start position, end position, rotation of the gripper about the vertical axis [deg], boxes (min corner, max corner)
SCENARIOS = {
    1: dict(name="corridor", p0=[0.2, 0.35, 0.12], p1=[0.6, -0.35, 0.12], rot=45,
            boxes=[([0.50, 0.04, 0.0], [0.62, 0.20, 0.25])]),
    2: dict(name="singularity", p0=[0.45, 0.12, 0.3], p1=[-0.45, 0.12, 0.3], rot=120, boxes=[]),
    3: dict(name="big rotation", p0=[0.35, -0.3, 0.3], p1=[0.35, 0.3, 0.3], rot=170,
            boxes=[([0.52, -0.12, 0.15], [0.70, 0.12, 0.45])]),
    # limits: {joint index (0-based): (min_deg, max_deg)}; None = no limit on that side
    4: dict(name="joint limits", p0=[0.45, 0.2, 0.15], p1=[-0.45, 0.2, 0.15], rot=60, boxes=[],
            limits={1: (-120, None), 2: (None, 150)}),
}
sc = SCENARIOS[SCENARIO]
robot = UR5()
theta_ready = np.array([0.0, -1.4, 1.6, -1.8, -1.57, 0.0])
R0 = robot.fk(theta_ready)[:3, :3]                                   # gripper pointing down
P0, P1 = np.array(sc["p0"], float), np.array(sc["p1"], float)
X_start = tu.RpToTrans(R0, P0)
X_end = tu.RpToTrans(tu.rot([0, 0, 1], np.radians(sc["rot"])) @ R0, P1)

# ---------------- joint trajectories on a fine grid in s
s_f = np.linspace(0, 1, 1500)
th0, ok = robot.ik(X_start, theta_ready); assert ok, "IK failed at the start pose"
th_dec, ok_d = robot.ik_along(tu.decoupled_path(X_start, X_end, s_f), th0)
th_scr, ok_s = robot.ik_along(tu.screw_path(X_start, X_end, s_f), th0)
th1 = th_dec[-1]                                   # same final joint configuration for the joint-space line
th_jnt = tu.joint_line(th0, th1, s_f)
methods = [("Joint-space line", "tab:green", th_jnt, True), ("Screw path", "tab:blue", th_scr, ok_s), ("Decoupled path", "tab:orange", th_dec, ok_d)]

# ---------------- time scaling and metrics
t = np.arange(0, T + 1e-9, 1 / (FPS * SLOWMO))
s_t, sd_t, _, _ = tu.quintic_time_scaling(T, t)
def in_box(p): return any(np.all(p >= np.array(lo)) and np.all(p <= np.array(hi)) for lo, hi in sc["boxes"])
LIMITS = sc.get("limits", {})
def rom_violations(th):
    """List of (joint number, angle, limit) for joints outside their limits (angles in degrees)."""
    out = []
    for i, (lo, hi) in LIMITS.items():
        a = np.degrees(th[i])
        if lo is not None and a < lo: out.append((i + 1, a, lo))
        if hi is not None and a > hi: out.append((i + 1, a, hi))
    return out
u = (P1 - P0) / np.linalg.norm(P1 - P0)
data = []
print(f"Scenario {SCENARIO} ({sc['name']}), T = {T} s, quintic time scaling")
print(f"{'method':<18}{'IK ok':>6}{'path [m]':>10}{'max off line [m]':>18}{'peak joint speed':>18}{'hits box':>10}{'outside joint limits':>22}")
for name, color, th, ok in methods:
    P = np.array([robot.fk(q)[:3, 3] for q in th])
    off = np.linalg.norm((P - P0) - np.outer((P - P0) @ u, u), axis=1)
    dth_ds = np.gradient(th, s_f, axis=0)
    th_t = np.column_stack([np.interp(s_t, s_f, th[:, i]) for i in range(6)])
    speed_t = np.abs(np.column_stack([np.interp(s_t, s_f, dth_ds[:, i]) for i in range(6)]) * sd_t[:, None])
    peak = np.degrees(np.abs(dth_ds).max() * 1.875 / T)          # quintic: max sdot = 15/(8T)
    hit = any(in_box(p) for p in P)
    worst = {}                                         # worst violation per joint
    for q in th:
        for j, a, lim in rom_violations(q):
            if j not in worst or abs(a - lim) > abs(worst[j][0] - lim): worst[j] = (a, lim)
    rom_txt = ", ".join(f"J{j} {a:.0f}° (limit {lim:.0f}°)" for j, (a, lim) in sorted(worst.items())) or "no"
    data.append(dict(th_t=th_t, speed=np.degrees(speed_t), L=np.linalg.norm(np.diff(P, axis=0), axis=1).sum(),
                     off=off.max(), peak=peak, hit=hit, ok=ok, rom=rom_txt))
    print(f"{name:<18}{str(ok):>6}{data[-1]['L']:>10.3f}{off.max():>18.3f}{peak:>14.0f} deg/s{('YES' if hit else 'no'):>10}   {rom_txt}")

# ---------------- figure
def draw_box(ax, lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    for a, b in [((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
                 ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
                 ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1))]:
        ax.plot(*np.array([a, b]).T, color="crimson", lw=1.5, alpha=0.8)

fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
artists, cursors = [], []
vmax = max(d["speed"].max() for d in data if d["ok"]) * 1.1
for k, ((name, color, th, ok), d) in enumerate(zip(methods, data)):
    ax = fig.add_subplot(gs[0, k], projection="3d")
    setup_3d(ax, center=(0.1, 0.0, 0.28), r=0.55, elev=32, azim=-70)
    ax.plot(*np.array([P0, P1]).T, "k:", lw=1.5)
    ax.plot([-0.08, 0.08, 0.08, -0.08, -0.08], [-0.08, -0.08, 0.08, 0.08, -0.08], [0] * 5, color="0.3")
    for lo, hi in sc["boxes"]: draw_box(ax, lo, hi)
    ax.set_title(name, color=color, fontsize=13, weight="bold")
    bad = d["hit"] or not d["ok"] or (LIMITS and d["rom"] != "no")
    ax.text2D(0.02, 0.0, f"gripper path {d['L']:.2f} m,  max off line {d['off']:.2f} m\n"
              f"peak joint speed {d['peak']:.0f} deg/s,  hits box: {'YES' if d['hit'] else 'no'}"
              + (f"\noutside joint limits: {d['rom']}" if LIMITS else "")
              + ("" if d["ok"] else "\nIK FAILED"), transform=ax.transAxes, fontsize=10,
              color="crimson" if bad else "k")
    artists.append(UR5Artist(ax, robot, trail_color=color))
    axs = fig.add_subplot(gs[1, k])
    if LIMITS:                                    # scenario 4: plot the limited joints and their limits
        allang = np.vstack([np.degrees(dd["th_t"][:, list(LIMITS)]) for dd in data])
        lo_y, hi_y = allang.min() - 15, allang.max() + 15
        for (i, (lo, hi)), c in zip(LIMITS.items(), ["tab:orange", "tab:green", "tab:purple"]):
            axs.plot(t, np.degrees(d["th_t"][:, i]), color=c, lw=2, label=f"joint {i+1}")
            if lo is not None: axs.axhline(lo, color=c, ls="--", lw=1.2); axs.axhspan(lo_y, lo, color=c, alpha=0.12)
            if hi is not None: axs.axhline(hi, color=c, ls="--", lw=1.2); axs.axhspan(hi, hi_y, color=c, alpha=0.12)
        axs.set_ylim(lo_y, hi_y)
        if k == 0: axs.set_ylabel("joint angle [deg]\n(shaded = outside limit)"); axs.legend(fontsize=8, loc="center left")
    else:
        for i in range(6): axs.plot(t, d["speed"][:, i], lw=1.6, label=f"joint {i+1}")
        axs.set_ylim(0, vmax)
        if k == 0: axs.set_ylabel("|joint speed| [deg/s]"); axs.legend(ncol=3, fontsize=8, loc="upper left")
    cursors.append(axs.axvline(0, color="k", lw=1))
    axs.set_xlabel("t [s]"); axs.grid(alpha=0.3)
title = fig.suptitle("")

def update(f):
    k = min(f, len(t) - 1); out = []
    for a, d, cur in zip(artists, data, cursors):
        if k == 0: a.hist = []
        out += a.update(d["th_t"][k])
        p = robot.fk(d["th_t"][k])[:3, 3]
        for g in a.grip: g.set_color("crimson" if in_box(p) else "k")
        if LIMITS: a.links.set_color("crimson" if rom_violations(d["th_t"][k]) else "0.35")
        cur.set_xdata([t[k], t[k]])
    slow = f"   (playback {SLOWMO}x slower)" if SLOWMO != 1 else ""
    title.set_text(f"Scenario {SCENARIO}: {sc['name']}   (same start/end pose, quintic, T = {T} s)   t = {t[k]:.2f} s{slow}")
    return out

plt.tight_layout(rect=(0, 0, 1, 0.95))
run(fig, update, len(t) + FPS, 1000 / FPS, GIF, FPS)
