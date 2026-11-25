"""
05_via_points.py  --  Piecewise-cubic interpolation through timed via points (MR Sec. 9.3).

Data from MR Figs. 9.7-9.8: vias (0,0) -> (0,1) -> (1,1) -> (1,0) at T = 0, 1, 2, 3.
  (a) via velocities given as in the book: (0,0), (1,0), (0,-1), (0,0)
  (b) via velocities from a simple heuristic: average of the neighbouring segment velocities

Things to notice:
  * Both pass exactly through the vias at the requested times.
  * The path SHAPE depends strongly on the via velocities.
  * Velocity is continuous at the vias, but acceleration JUMPS there (printed below).

Try this:
  1. Change the via velocities in VEL_A and see how the loops change.
  2. Move via 3 or change the via times. Does the heuristic still give a nice path?
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu

TIMES = [0, 1, 2, 3]
POS = [[0, 0], [0, 1], [1, 1], [1, 0]]
VEL_A = [[0, 0], [1, 0], [0, -1], [0, 0]]
VEL_B = tu.heuristic_via_velocities(TIMES, POS)
SAVE = False

print("heuristic via velocities:", VEL_B.tolist())
res = {"(a) given velocities": tu.cubic_via(TIMES, POS, VEL_A), "(b) heuristic velocities": tu.cubic_via(TIMES, POS, VEL_B)}

for name, V in (("(a) given velocities", VEL_A), ("(b) heuristic velocities", VEL_B)):
    print(name)
    segs = tu.cubic_via_coeffs(TIMES, POS, V)
    for k in range(1, len(TIMES) - 1):
        a0, a1, a2, a3, dT = segs[k-1]; acc_in = 2*a2 + 6*a3*dT     # end of the segment before the via
        acc_out = 2*segs[k][2]                                      # start of the segment after the via
        print(f"   via {k+1} at t={TIMES[k]}: acceleration jumps from {np.round(acc_in, 2)} to {np.round(acc_out, 2)}")

fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
P = np.array(POS)
for row, (name, (t, x, xd, xdd)) in enumerate(res.items()):
    V = np.array(VEL_A if row == 0 else VEL_B)
    ax = axs[row, 0]
    ax.plot(x[:, 0], x[:, 1], lw=2.5, color="tab:blue" if row == 0 else "tab:orange")
    ax.plot(P[:, 0], P[:, 1], "ko")
    for p, v in zip(P, V):
        if np.linalg.norm(v) > 0: ax.annotate("", xy=p + 0.3*v, xytext=p, arrowprops=dict(arrowstyle="->", ls="--"))
    ax.set_aspect("equal"); ax.set_title(name + ": path"); ax.grid(alpha=0.3)
    for col, (y, lab) in enumerate([(xd, "velocity"), (xdd, "acceleration")], start=1):
        ax = axs[row, col]
        ax.plot(t, y[:, 0], label="x"); ax.plot(t, y[:, 1], label="y")
        for Tk in TIMES: ax.axvline(Tk, color="gray", lw=0.6)
        ax.set_title(f"{name}: {lab}"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel("t [s]")
plt.tight_layout()
if SAVE: plt.savefig("05_via_points.png", dpi=150)
plt.show()
