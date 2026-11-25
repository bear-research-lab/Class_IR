"""
traj_utils.py  --  helper functions for Chapter 9 (Trajectory Generation) demos.

Everything here uses only NumPy, so students can read every line.
Notation follows Modern Robotics (Lynch & Park), Chapter 9:
    path            theta(s),  s in [0, 1]
    time scaling    s(t),      t in [0, T]
    trajectory      theta(s(t))
"""
import numpy as np

# =====================================================================
# 1. Time scalings  s(t)  (all return s, sdot, sddot, sdddot (jerk))
# =====================================================================
def cubic_time_scaling(T, t):
    """Cubic s(t) = 3(t/T)^2 - 2(t/T)^3 : rest-to-rest, Eq. (9.11)-(9.13)."""
    t = np.asarray(t, float); tau = t / T
    s = 3*tau**2 - 2*tau**3
    sd = (6*tau - 6*tau**2) / T
    sdd = (6 - 12*tau) / T**2
    sddd = np.full_like(t, -12 / T**3)
    return s, sd, sdd, sddd


def quintic_time_scaling(T, t):
    """Quintic s(t) = 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5 : also zero accel at both ends."""
    t = np.asarray(t, float); tau = t / T
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    sd = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    sdd = (60*tau - 180*tau**2 + 120*tau**3) / T**2
    sddd = (60 - 360*tau + 360*tau**2) / T**3
    return s, sd, sdd, sddd


def trapezoidal_time_scaling(v, a, t):
    """Trapezoidal profile with coasting speed v and acceleration a (Eqs. 9.16-9.24).
    Requires v^2/a <= 1 (otherwise v is never reached: use triangular_time_scaling).
    Returns s, sd, sdd, sddd and the total time T = (a + v^2)/(v a)."""
    if v**2 / a > 1:
        raise ValueError("v^2/a > 1: the profile is triangular (bang-bang), v is never reached")
    t = np.asarray(t, float)
    T = (a + v**2) / (v * a); ta = v / a
    s = np.where(t <= ta, 0.5*a*t**2,
        np.where(t <= T - ta, v*t - v**2/(2*a),
                 (2*a*v*T - 2*v**2 - a**2*(t - T)**2) / (2*a)))
    sd = np.where(t <= ta, a*t, np.where(t <= T - ta, v, a*(T - t)))
    sdd = np.where(t < ta, a, np.where(t <= T - ta, 0.0, -a))
    sddd = np.zeros_like(t)              # infinite at the 4 switching instants (not representable)
    return s, sd, sdd, sddd, T


def trapezoid_from_T(T, vT=1.25):
    """Pick a trapezoid that finishes in time T.  Choose v with 1 < vT <= 2, then a = v^2/(vT - 1)."""
    v = vT / T
    a = v**2 / (v*T - 1)
    return v, a


def triangular_time_scaling(a, t):
    """Bang-bang (v^2/a > 1 case): accelerate at a for T/2, decelerate for T/2.  T = 2/sqrt(a)."""
    t = np.asarray(t, float); T = 2/np.sqrt(a)
    s = np.where(t <= T/2, 0.5*a*t**2, 1 - 0.5*a*(T - t)**2)
    sd = np.where(t <= T/2, a*t, a*(T - t))
    sdd = np.where(t < T/2, a, -a)
    return s, sd, sdd, np.zeros_like(t), T


def scurve_time_scaling(T, t, vT=1.25, r=0.5):
    """7-phase S-curve with the SAME total time, top speed and acceleration time as
    trapezoid_from_T(T, vT).  r in (0, 1] is the fraction of each half of the
    acceleration phase spent ramping the acceleration (r = 1: no constant-accel phase).
    Each phase has constant jerk, so we propagate the state exactly."""
    v, a_trap = trapezoid_from_T(T, vT)
    t_acc = v / a_trap                      # duration of the speed-up (same as the trapezoid)
    tj = r * t_acc / 2                      # jerk phase duration
    ta2 = t_acc - 2*tj                      # constant-acceleration duration
    amax = v / (t_acc - tj)                 # peak acceleration needed to reach v in t_acc
    J = amax / tj
    tv = T - 2*t_acc                        # coasting time (same as trapezoid)
    phases = [(tj, J), (ta2, 0), (tj, -J), (tv, 0), (tj, -J), (ta2, 0), (tj, J)]
    t = np.asarray(t, float)
    out = np.zeros((4, t.size))
    for k, tk in enumerate(t):
        x = np.zeros(3); t0 = 0.0          # state (s, sd, sdd) at the start of the phase
        for i, (dur, j) in enumerate(phases):
            if tk <= t0 + dur or i == len(phases) - 1:
                h = min(tk - t0, dur)
                out[:, k] = [x[0] + x[1]*h + x[2]*h**2/2 + j*h**3/6, x[1] + x[2]*h + j*h**2/2, x[2] + j*h, j]
                break
            h = dur
            x = np.array([x[0] + x[1]*h + x[2]*h**2/2 + j*h**3/6, x[1] + x[2]*h + j*h**2/2, x[2] + j*h])
            t0 += dur
    return out[0], out[1], out[2], out[3]


# =====================================================================
# 2. Rigid-body tools (Chapter 3), needed for "straight lines" in SE(3)
# =====================================================================
def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def unskew(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def exp3(so3mat):
    """Matrix exponential of [w]theta (Rodrigues' formula)."""
    w_theta = unskew(so3mat); theta = np.linalg.norm(w_theta)
    if theta < 1e-12:
        return np.eye(3)
    W = so3mat / theta
    return np.eye(3) + np.sin(theta)*W + (1 - np.cos(theta))*W @ W


def log3(R):
    """Matrix logarithm of a rotation matrix, returns [w]theta."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    if c >= 1 - 1e-12:
        return np.zeros((3, 3))
    if c <= -1 + 1e-12:                    # theta = pi
        if 1 + R[2, 2] > 1e-9:
            w = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]) / np.sqrt(2*(1 + R[2, 2]))
        elif 1 + R[1, 1] > 1e-9:
            w = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]) / np.sqrt(2*(1 + R[1, 1]))
        else:
            w = np.array([1 + R[0, 0], R[1, 0], R[2, 0]]) / np.sqrt(2*(1 + R[0, 0]))
        return skew(np.pi * w)
    theta = np.arccos(c)
    return theta / (2*np.sin(theta)) * (R - R.T)


def exp6(se3mat):
    """Matrix exponential of an se(3) matrix [S]theta."""
    W = se3mat[:3, :3]; v = se3mat[:3, 3]
    theta = np.linalg.norm(unskew(W))
    T = np.eye(4)
    if theta < 1e-12:
        T[:3, 3] = v
        return T
    Wn = W / theta
    G = np.eye(3)*theta + (1 - np.cos(theta))*Wn + (theta - np.sin(theta))*Wn @ Wn
    T[:3, :3] = exp3(W); T[:3, 3] = G @ (v / theta)
    return T


def log6(T):
    """Matrix logarithm of a transformation matrix, returns [S]theta (4x4)."""
    R, p = T[:3, :3], T[:3, 3]
    W = log3(R); theta = np.linalg.norm(unskew(W))
    out = np.zeros((4, 4))
    if theta < 1e-12:
        out[:3, 3] = p
        return out
    Wn = W / theta
    Ginv = np.eye(3)/theta - Wn/2 + (1/theta - 0.5/np.tan(theta/2)) * Wn @ Wn
    out[:3, :3] = W; out[:3, 3] = Ginv @ p * theta
    return out


def trans_inv(T):
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ p
    return Ti


def RpToTrans(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def rot(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return exp3(skew(axis) * angle)


# =====================================================================
# 3. Paths
# =====================================================================
def joint_line(theta_start, theta_end, s):
    """theta(s) = theta_start + s (theta_end - theta_start), Eq. (9.3). Returns (len(s), n)."""
    s = np.atleast_1d(s)[:, None]
    return theta_start + s * (theta_end - theta_start)


def screw_path(X_start, X_end, s):
    """X(s) = X_start exp(log(X_start^-1 X_end) s), Eq. (9.6): constant screw axis."""
    S = log6(trans_inv(X_start) @ X_end)
    return [X_start @ exp6(S * si) for si in np.atleast_1d(s)]


def decoupled_path(X_start, X_end, s):
    """p(s) straight line, R(s) = R_start exp(log(R_start^T R_end) s), Eqs. (9.7)-(9.8)."""
    Rs, ps = X_start[:3, :3], X_start[:3, 3]; Re, pe = X_end[:3, :3], X_end[:3, 3]
    W = log3(Rs.T @ Re)
    return [RpToTrans(Rs @ exp3(W * si), ps + si*(pe - ps)) for si in np.atleast_1d(s)]


# =====================================================================
# 4. Via points: piecewise cubic, Eqs. (9.25)-(9.29)
# =====================================================================
def cubic_via_coeffs(times, pos, vel):
    """Coefficients (a0, a1, a2, a3, dT) of each cubic segment, Eqs. (9.26)-(9.29)."""
    times = np.asarray(times, float); pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    segs = []
    for j in range(len(times) - 1):
        dT = times[j+1] - times[j]
        b0, b1, v0, v1 = pos[j], pos[j+1], vel[j], vel[j+1]
        a2 = (3*b1 - 3*b0 - 2*v0*dT - v1*dT) / dT**2
        a3 = (2*b0 + (v0 + v1)*dT - 2*b1) / dT**3
        segs.append((b0, v0, a2, a3, dT))
    return segs


def cubic_via(times, pos, vel, n_per_seg=100):
    """times (k,), pos (k, d), vel (k, d). Returns t, x, xd, xdd sampled along all segments
    (each segment includes both of its end points, so jumps in acceleration show up as vertical lines)."""
    T_all, X, Xd, Xdd = [], [], [], []
    for (a0, a1, a2, a3, dT), T0 in zip(cubic_via_coeffs(times, pos, vel), times):
        for dt in np.linspace(0, dT, n_per_seg):
            T_all.append(T0 + dt)
            X.append(a0 + a1*dt + a2*dt**2 + a3*dt**3)
            Xd.append(a1 + 2*a2*dt + 3*a3*dt**2)
            Xdd.append(2*a2 + 6*a3*dt)
    return np.array(T_all), np.array(X), np.array(Xd), np.array(Xdd)


def heuristic_via_velocities(times, pos):
    """Via velocity = average of the velocities of the two neighbouring segments; zero at the ends."""
    times = np.asarray(times, float); pos = np.asarray(pos, float)
    vel = np.zeros_like(pos)
    for i in range(1, len(times) - 1):
        v_in = (pos[i] - pos[i-1]) / (times[i] - times[i-1])
        v_out = (pos[i+1] - pos[i]) / (times[i+1] - times[i])
        vel[i] = 0.5 * (v_in + v_out)
    return vel


# =====================================================================
# 5. Planar 2R arm (kinematics + dynamics with point masses at link ends)
# =====================================================================
class Planar2R:
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.L1, self.L2, self.m1, self.m2, self.g = L1, L2, m1, m2, g

    def fk(self, th):
        th = np.asarray(th, float)
        x = self.L1*np.cos(th[..., 0]) + self.L2*np.cos(th[..., 0] + th[..., 1])
        y = self.L1*np.sin(th[..., 0]) + self.L2*np.sin(th[..., 0] + th[..., 1])
        return np.stack([x, y], axis=-1)

    def elbow(self, th):
        th = np.asarray(th, float)
        return np.stack([self.L1*np.cos(th[..., 0]), self.L1*np.sin(th[..., 0])], axis=-1)

    def ik(self, p, elbow_up=False):
        """Closed-form IK; elbow_up=False gives theta2 >= 0."""
        x, y = p
        c2 = (x**2 + y**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2)
        if abs(c2) > 1 + 1e-9:
            raise ValueError(f"point {p} is out of reach")
        t2 = np.arccos(np.clip(c2, -1, 1)) * (-1 if elbow_up else 1)
        t1 = np.arctan2(y, x) - np.arctan2(self.L2*np.sin(t2), self.L1 + self.L2*np.cos(t2))
        return np.array([t1, t2])

    # --- dynamics  M(th) thdd + c(th, thd) + g(th) = tau  (vertical plane, gravity along -y)
    def M(self, th):
        L1, L2, m1, m2 = self.L1, self.L2, self.m1, self.m2; c2 = np.cos(th[1])
        return np.array([[m1*L1**2 + m2*(L1**2 + 2*L1*L2*c2 + L2**2), m2*(L1*L2*c2 + L2**2)],
                         [m2*(L1*L2*c2 + L2**2), m2*L2**2]])

    def c(self, th, thd):
        s2 = np.sin(th[1]); k = self.m2*self.L1*self.L2*s2
        return np.array([-k*(2*thd[0]*thd[1] + thd[1]**2), k*thd[0]**2])

    def grav(self, th):
        L1, L2, m1, m2, g = self.L1, self.L2, self.m1, self.m2, self.g
        return np.array([(m1 + m2)*L1*g*np.cos(th[0]) + m2*g*L2*np.cos(th[0] + th[1]),
                         m2*g*L2*np.cos(th[0] + th[1])])
