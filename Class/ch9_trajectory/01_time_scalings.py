"""
01_time_scalings.py  --  Same path, same total time T, four different s(t).

What to look at:
  * s(t):    all four go from 0 to 1 in time T. They look almost the same!
  * sdot:    peak speeds differ (cubic 1.5/T, quintic 1.875/T, trapezoid/S-curve = v).
  * sddot:   cubic jumps at t = 0 and t = T; trapezoid jumps 4 times;
             quintic and S-curve are continuous.
  * jerk:    cubic and trapezoid have infinite jerk at the jumps (shown as spikes);
             quintic is finite; S-curve is piecewise constant.

Try this:
  1. Change T. How do the peak values scale with T?  (speed ~ 1/T, acceleration ~ 1/T^2)
  2. Change VT (top speed * T, 1 < VT <= 2). What happens to the trapezoid when VT -> 1? When VT = 2?
  3. Change R (S-curve jerk fraction, 0 < R <= 1). Compare R = 0.1 with R = 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu

T = 2.0      # total motion time [s]
VT = 1.25    # trapezoid / S-curve top speed v = VT / T  (need 1 < VT <= 2)
R = 0.5      # S-curve: fraction of the speed-up time spent ramping acceleration
SAVE = False

t = np.linspace(0, T, 2001)
v, a = tu.trapezoid_from_T(T, VT)
profiles = {
    "Cubic": tu.cubic_time_scaling(T, t),
    "Quintic": tu.quintic_time_scaling(T, t),
    "Trapezoidal": tu.trapezoidal_time_scaling(v, a, t)[:4],
    "S-curve": tu.scurve_time_scaling(T, t, VT, R),
}
colors = {"Cubic": "tab:blue", "Quintic": "tab:green", "Trapezoidal": "tab:orange", "S-curve": "tab:purple"}

# ---- numbers to compare with the formulas from the lecture
print(f"T = {T} s")
print(f"{'profile':<12} {'s(T)':>7} {'max sdot':>10} {'max |sddot|':>12} {'max |jerk|':>11}")
for name, (s, sd, sdd, sddd) in profiles.items():
    jerk = "infinite" if name in ("Cubic", "Trapezoidal") else f"{np.abs(sddd).max():11.3f}"
    print(f"{name:<12} {s[-1]:7.3f} {sd.max():10.4f} {np.abs(sdd).max():12.4f} {jerk:>11}")
print(f"formulas: cubic 3/(2T) = {1.5/T:.4f}, quintic 15/(8T) = {15/(8*T):.4f}, trapezoid v = {v:.4f}")
print(f"          cubic |sddot| = 6/T^2 = {6/T**2:.4f}, quintic 10/(sqrt(3) T^2) = {10/(np.sqrt(3)*T**2):.4f}, trapezoid a = {a:.4f}")

# ---- plots: rows = s, sdot, sddot, jerk ; one curve per profile
fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
labels = [r"$s$", r"$\dot s$", r"$\ddot s$", r"jerk $\dddot s$"]
for name, prof in profiles.items():
    for k in range(4):
        y = prof[k]
        if k == 2 and name == "Trapezoidal":
            axs[k].step(t, y, where="post", color=colors[name], label=name)
        else:
            axs[k].plot(t, y, color=colors[name], label=name)
# mark the infinite-jerk instants
for x in (0, T):
    axs[3].annotate("", xy=(x, 15/T**3), xytext=(x, 0), arrowprops=dict(arrowstyle="->", color=colors["Cubic"]))
ta = v / a
for x, sgn in ((0, 1), (ta, -1), (T - ta, -1), (T, 1)):
    axs[3].annotate("", xy=(x, sgn*12/T**3), xytext=(x, 0), arrowprops=dict(arrowstyle="->", color=colors["Trapezoidal"], ls="--"))
axs[3].text(0.02*T, -14/T**3, "arrows = infinite jerk (impulses)", fontsize=9)
for ax, lb in zip(axs, labels):
    ax.set_ylabel(lb, fontsize=13); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
axs[0].legend(ncol=4, fontsize=9, loc="upper left")
axs[-1].set_xlabel("t [s]")
fig.suptitle(f"Four time scalings with the same T = {T} s")
plt.tight_layout()
if SAVE: plt.savefig("01_time_scalings.png", dpi=150)
plt.show()
