"""
06_time_optimal_2R.py  --  Time-optimal time scaling along a given path (MR Sec. 9.4).

Robot: planar 2R arm in a VERTICAL plane (gravity), point masses at the link ends.
Path:  joint-space straight line theta(s) = theta_start + s * dtheta  (so theta'' = 0).

Steps (same as the lecture):
  1. Dynamics along the path:  m(s) sddot + c(s) sdot^2 + g(s) = tau
        m(s) = M(theta) dtheta ,  c(s) = c(theta, dtheta) ,  g(s) = g(theta)
  2. Torque limits -> acceleration limits  L(s, sdot) <= sddot <= U(s, sdot)    (Eq. 9.36)
  3. Velocity limit curve: the largest sdot where L <= U.
  4. Bang-bang: integrate sddot = U forward from (0,0), sddot = L backward from (1,0);
     switch where they cross.
     (We integrate z = sdot^2 in s:  dz/ds = 2 sddot. This avoids dividing by sdot = 0.)
  5. Compare with the fastest CUBIC time scaling that respects the same torque limits.

This script handles the single-switch case. If the forward curve hits the velocity limit
curve first, it says so: then the full algorithm with several switches (MR Sec. 9.4.2) is needed.

Try this:
  1. Lower TAU_MAX[1] from 12 to 11 and then 10. The time-optimal T grows a little, but the fastest
     cubic slows down a lot (3.5 s -> 4.8 s -> 11.9 s). Why? (Hint: look at the torque plots near t = 0.)
  2. Lower TAU_MAX[1] to 9: the arm can no longer hold itself against gravity on this path.
  3. Set MOTOR_SPEED_DROP = True: available torque drops linearly with joint speed (DC motor).
  4. Change the path (THETA_START, THETA_END). Where is the switch now?
"""
import numpy as np
import matplotlib.pyplot as plt
import traj_utils as tu

arm = tu.Planar2R(L1=1.0, L2=1.0, m1=1.0, m2=1.0)
deg = np.pi / 180
THETA_START = np.array([-90, 0]) * deg
THETA_END = np.array([90, 150]) * deg
TAU_MAX = np.array([30.0, 12.0])            # |tau_i| <= TAU_MAX_i  [N m]
MOTOR_SPEED_DROP = False                    # if True: tau_max_i = TAU_MAX_i (1 - |theta_dot_i| / W_FREE_i)
W_FREE = np.array([6.0, 8.0])               # no-load joint speed [rad/s] for the speed-drop model
N = 4000                                    # number of steps in s
SAVE = False

d = THETA_END - THETA_START

def path_dynamics(s):
    th = THETA_START + s * d
    return arm.M(th) @ d, arm.c(th, d), arm.grav(th)          # m(s), c(s), g(s)

def tau_limits(sd):
    tmax = TAU_MAX.copy()
    if MOTOR_SPEED_DROP:
        tmax = TAU_MAX * np.clip(1 - np.abs(d * sd) / W_FREE, 0, None)
    return -tmax, tmax

def L_U(s, sd):
    """Eq. (9.36): each actuator gives an interval for sddot; intersect them."""
    m, c, g = path_dynamics(s); tmin, tmax = tau_limits(sd)
    Ls, Us = [], []
    for i in range(2):
        lo = (tmin[i] - c[i]*sd**2 - g[i]) / m[i]
        hi = (tmax[i] - c[i]*sd**2 - g[i]) / m[i]
        if m[i] > 0: Ls.append(lo); Us.append(hi)
        elif m[i] < 0: Ls.append(hi); Us.append(lo)
        else: raise RuntimeError("zero-inertia point (m_i = 0): see MR Sec. 9.4.4")
    return max(Ls), min(Us)

def velocity_limit(s, hi=50.0):
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        L, U = L_U(s, mid)
        lo, hi = (mid, hi) if L <= U else (lo, mid)
    return lo

# ---------------- 1) bang-bang in the phase plane (z = sdot^2)
S = np.linspace(0, 1, N + 1); ds = S[1] - S[0]
z_fwd = np.full(N + 1, np.nan); z_fwd[0] = 0.0; hit_fwd = None
for k in range(N):
    L, U = L_U(S[k], np.sqrt(z_fwd[k]))
    if L > U: hit_fwd = k; break
    z_fwd[k+1] = z_fwd[k] + 2*U*ds
z_bwd = np.full(N + 1, np.nan); z_bwd[N] = 0.0; hit_bwd = None
for k in range(N, 0, -1):
    L, U = L_U(S[k], np.sqrt(z_bwd[k]))
    if L > U: hit_bwd = k; break
    z_bwd[k-1] = max(z_bwd[k] - 2*L*ds, 0.0)

cross = np.where(z_fwd >= z_bwd)[0]
VLC = np.array([velocity_limit(s) for s in S[::40]])
static_ok = all(L_U(s, 0.0)[0] <= 0 <= L_U(s, 0.0)[1] for s in S[::20])
if not static_ok:
    print("The robot cannot hold some configurations on this path against gravity (sddot = 0 not in [L, U] at sdot = 0).")
    print("The 'static posture maintenance' assumption of MR Sec. 9.4.4 is violated. Increase TAU_MAX.")
    k_sw = None
elif len(cross) == 0:
    where = f"at s = {S[hit_fwd]:.3f} " if hit_fwd is not None else ""
    print(f"The forward curve hits the velocity limit curve {where}before meeting the backward curve.")
    print("A single switch is not enough: the full algorithm with several switches (MR Sec. 9.4.2) is needed.")
    k_sw = None
else:
    k_sw = cross[0]
    z_opt = np.where(np.arange(N + 1) < k_sw, z_fwd, z_bwd)
    sd_opt = np.sqrt(np.maximum(z_opt, 0))
    dt = 2*ds / (sd_opt[:-1] + sd_opt[1:])                    # time to cross each ds
    t_opt = np.concatenate([[0], np.cumsum(dt)]); T_opt = t_opt[-1]
    sdd_opt = np.array([L_U(s, v)[1] if k < k_sw else L_U(s, v)[0] for k, (s, v) in enumerate(zip(S, sd_opt))])
    tau_opt = np.array([m*a + c*v**2 + g for (m, c, g), a, v in ((path_dynamics(s), a, v) for s, a, v in zip(S, sdd_opt, sd_opt))])
    print(f"Time-optimal: T = {T_opt:.3f} s, switch from max accel to max decel at s* = {S[k_sw]:.3f}")

# ---------------- 2) fastest cubic time scaling that respects the same torque limits
def cubic_ok(T):
    t = np.linspace(0, T, 600); s, sd, sdd, _ = tu.cubic_time_scaling(T, t)
    for si, v, a in zip(s, sd, sdd):
        m, c, g = path_dynamics(si); tmin, tmax = tau_limits(v); tau = m*a + c*v**2 + g
        if np.any(tau > tmax + 1e-9) or np.any(tau < tmin - 1e-9): return False
    return True
lo, hi = 0.1, 20.0
if not cubic_ok(hi):
    raise SystemExit("No feasible cubic time scaling with T <= 20 s for these torque limits.")
for _ in range(40):
    mid = (lo + hi)/2; lo, hi = (lo, mid) if cubic_ok(mid) else (mid, hi)
T_cub = hi
t_c = np.linspace(0, T_cub, 600); s_c, sd_c, sdd_c, _ = tu.cubic_time_scaling(T_cub, t_c)
tau_c = np.array([m*a + c*v**2 + g for (m, c, g), a, v in ((path_dynamics(s), a, v) for s, a, v in zip(s_c, sdd_c, sd_c))])
print(f"Fastest feasible cubic: T = {T_cub:.3f} s")

# ---------------- plots
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axs[0]
ax.fill_between(S[::40], VLC, VLC.max()*1.3, color="lightgray", label="inadmissible (L > U)")
ax.plot(S[::40], VLC, "k", lw=1.5)
ax.plot(S, z_fwd**0.5, color="tab:orange", ls="--", lw=1.2, label="forward, sddot = U")
ax.plot(S, z_bwd**0.5, color="tab:blue", ls="--", lw=1.2, label="backward, sddot = L")
if k_sw is not None:
    ax.plot(S, sd_opt, color="k", lw=2.5, label=f"time-optimal (T = {T_opt:.2f} s)")
    ax.plot(S[k_sw], sd_opt[k_sw], "ro", ms=8)
ax.plot(s_c, sd_c, color="tab:green", lw=2, label=f"fastest cubic (T = {T_cub:.2f} s)")
for (s0, v0) in [(0.15, 0.3), (0.5, 0.3), (0.85, 0.3), (0.5, 0.8)]:      # motion cones
    L, U = L_U(s0, v0)
    if L <= U:
        for a in (L, U):
            vec = np.array([v0, a]); vec = vec/np.linalg.norm(vec)*0.07
            ax.annotate("", xy=(s0 + vec[0], v0 + vec[1]), xytext=(s0, v0), arrowprops=dict(arrowstyle="->", color="purple"))
ax.set_xlabel("s"); ax.set_ylabel("sdot"); ax.set_xlim(0, 1); ax.set_ylim(0, VLC.max()*1.05 if VLC.max() < 4 else 4)
ax.set_title("Phase plane (purple arrows: motion cones)"); ax.legend(fontsize=8, loc="upper right")
for i in range(2):
    ax = axs[i+1]
    if k_sw is not None: ax.plot(t_opt, tau_opt[:, i], "k", lw=2.5, label="time-optimal")
    ax.plot(t_c, tau_c[:, i], color="tab:green", lw=2, label="fastest cubic")
    if not MOTOR_SPEED_DROP:
        for sgn in (1, -1): ax.axhline(sgn*TAU_MAX[i], color="r", ls=":", label="limit" if sgn > 0 else None)
    ax.set_xlabel("t [s]"); ax.set_ylabel(f"tau_{i+1} [N m]"); ax.set_title(f"Joint {i+1} torque"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
if SAVE: plt.savefig("06_time_optimal_2R.png", dpi=150)
plt.show()
