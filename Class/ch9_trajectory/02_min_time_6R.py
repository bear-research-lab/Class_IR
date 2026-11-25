"""
02_min_time_6R.py  --  A 6-joint robot moves on a joint-space straight line.
Given joint speed and acceleration limits, what is the SHORTEST motion time T
for each time scaling?  Which joint limits the motion?

Joint-space straight line:  theta(t) = theta_start + s(t) * dtheta
  => theta_dot_i = sdot * dtheta_i ,  theta_ddot_i = sddot * dtheta_i
So every joint uses the same s(t): all joints start and stop together.

Minimum T (per joint i, take the max over joints):
  cubic     : T >= 1.5  |dtheta_i| / vlim_i   and  T >= sqrt(6        |dtheta_i| / alim_i)
  quintic   : T >= 1.875|dtheta_i| / vlim_i   and  T >= sqrt(10/sqrt3 |dtheta_i| / alim_i)
  trapezoid : v = min_i vlim_i/|dtheta_i| , a = min_i alim_i/|dtheta_i|
              T = (a + v^2)/(v a)  if v^2/a <= 1,  else triangular T = 2/sqrt(a)

The numbers below are example values, not the specs of a particular robot.

Try this:
  1. Make one joint move much farther. Does it become the limiting joint?
  2. Make the acceleration limits very large. Which formula now decides T?
  3. Make the move very short (small dtheta). When does the trapezoid become triangular?
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu

deg = np.pi / 180
theta_start = np.array([0, -30, 60, 0, 45, 0]) * deg
theta_end = np.array([90, 20, 10, -60, -30, 120]) * deg
vlim = np.array([150, 150, 180, 220, 220, 220]) * deg      # [rad/s]   example values
alim = np.array([300, 300, 400, 600, 600, 600]) * deg      # [rad/s^2] example values
SAVE = False

d = theta_end - theta_start; ad = np.abs(d) + 1e-12

T_cubic_v, T_cubic_a = 1.5*ad/vlim, np.sqrt(6*ad/alim)
T_quin_v, T_quin_a = 1.875*ad/vlim, np.sqrt(10/np.sqrt(3)*ad/alim)
T_cubic = max(T_cubic_v.max(), T_cubic_a.max())
T_quin = max(T_quin_v.max(), T_quin_a.max())
v = (vlim/ad).min(); a = (alim/ad).min()
if v**2/a <= 1:
    T_trap = (a + v**2)/(v*a); trap_kind = "3-phase trapezoid"
else:
    T_trap = 2/np.sqrt(a); trap_kind = "triangular (v never reached)"

def which(Tv, Ta):
    i = np.argmax(np.maximum(Tv, Ta)); kind = "speed" if Tv[i] >= Ta[i] else "acceleration"
    return f"joint {i+1} ({kind})"
print("Minimum motion times")
print(f"  cubic    : T = {T_cubic:.3f} s   limited by {which(T_cubic_v, T_cubic_a)}")
print(f"  quintic  : T = {T_quin:.3f} s   limited by {which(T_quin_v, T_quin_a)}")
print(f"  trapezoid: T = {T_trap:.3f} s   ({trap_kind}; v = {v:.3f}, a = {a:.3f} in units of s)")
print("  -> the trapezoid is the fastest straight-line motion under constant v, a limits (MR Sec. 9.2.2.2)")
print("  -> cubic vs quintic: the quintic has a higher peak speed (1.875/T vs 1.5/T) but a slightly LOWER")
print("     peak acceleration (5.77/T^2 vs 6/T^2), so when acceleration limits dominate it can even be faster.")

fig, axs = plt.subplots(3, 3, figsize=(13, 8), sharey="row")
for col, (name, T) in enumerate([("Cubic", T_cubic), ("Quintic", T_quin), ("Trapezoidal", T_trap)]):
    t = np.linspace(0, T, 1500)
    if name == "Cubic": s, sd, sdd, _ = tu.cubic_time_scaling(T, t)
    elif name == "Quintic": s, sd, sdd, _ = tu.quintic_time_scaling(T, t)
    elif v**2/a <= 1: s, sd, sdd, _, _ = tu.trapezoidal_time_scaling(v, a, t)
    else: s, sd, sdd, _, _ = tu.triangular_time_scaling(a, t)
    th = theta_start + s[:, None]*d; thd = sd[:, None]*d; thdd = sdd[:, None]*d
    for i in range(6):
        axs[0, col].plot(t, th[:, i]/deg, label=f"joint {i+1}")
        axs[1, col].plot(t, np.abs(thd[:, i])/vlim[i]*100)
        axs[2, col].plot(t, np.abs(thdd[:, i])/alim[i]*100)
    axs[0, col].set_title(f"{name}:  T = {T:.2f} s")
    for r in (1, 2): axs[r, col].axhline(100, color="r", ls="--", lw=1)
    axs[2, col].set_xlabel("t [s]")
axs[0, 0].set_ylabel("joint angle [deg]"); axs[1, 0].set_ylabel("|speed| / limit [%]"); axs[2, 0].set_ylabel("|accel| / limit [%]")
axs[0, 0].legend(fontsize=8, ncol=2)
for ax in axs.flat: ax.grid(alpha=0.3)
fig.suptitle("Fastest joint-space straight line for each time scaling (red = 100% of the limit)")
plt.tight_layout()
if SAVE: plt.savefig("02_min_time_6R.png", dpi=150)
plt.show()
