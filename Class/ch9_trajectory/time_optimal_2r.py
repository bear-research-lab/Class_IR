"""
time_optimal_2r.py  --  the computation of 06_time_optimal_2R.py packed into functions,
so that A5_anim_time_optimal_2R.py can reuse it.  See 06 for the step-by-step explanation.
"""
import numpy as np
import traj_utils as tu


class PathProblem:
    def __init__(self, arm, theta_start, theta_end, tau_max):
        self.arm, self.th0, self.d, self.tau_max = arm, np.asarray(theta_start, float), np.asarray(theta_end, float) - np.asarray(theta_start, float), np.asarray(tau_max, float)

    def theta(self, s):
        return self.th0 + s * self.d

    def dyn(self, s):                     # m(s), c(s), g(s) for a joint-space straight line
        th = self.theta(s)
        return self.arm.M(th) @ self.d, self.arm.c(th, self.d), self.arm.grav(th)

    def L_U(self, s, sd):
        m, c, g = self.dyn(s); Ls, Us = [], []
        for i in range(len(m)):
            lo = (-self.tau_max[i] - c[i] * sd**2 - g[i]) / m[i]; hi = (self.tau_max[i] - c[i] * sd**2 - g[i]) / m[i]
            (Ls.append(lo), Us.append(hi)) if m[i] > 0 else (Ls.append(hi), Us.append(lo))
        return max(Ls), min(Us)

    def torque(self, s, sd, sdd):
        m, c, g = self.dyn(s); return m * sdd + c * sd**2 + g

    def time_optimal(self, N=4000):
        """Single-switch bang-bang. Returns t, s, sd, sdd arrays (in time order) and the switch s*."""
        S = np.linspace(0, 1, N + 1); ds = S[1]
        zf = np.full(N + 1, np.nan); zf[0] = 0
        for k in range(N):
            L, U = self.L_U(S[k], np.sqrt(zf[k]))
            if L > U: break
            zf[k + 1] = zf[k] + 2 * U * ds
        zb = np.full(N + 1, np.nan); zb[N] = 0
        for k in range(N, 0, -1):
            L, U = self.L_U(S[k], np.sqrt(zb[k]))
            if L > U: break
            zb[k - 1] = max(zb[k] - 2 * L * ds, 0)
        cross = np.where(zf >= zb)[0]
        if len(cross) == 0:
            raise RuntimeError("no single-switch solution (velocity limit curve reached): see MR Sec. 9.4.2")
        ks = cross[0]; z = np.where(np.arange(N + 1) < ks, zf, zb); sd = np.sqrt(np.maximum(z, 0))
        tt = np.concatenate([[0], np.cumsum(2 * ds / (sd[:-1] + sd[1:]))])
        sdd = np.array([self.L_U(si, v)[1] if k < ks else self.L_U(si, v)[0] for k, (si, v) in enumerate(zip(S, sd))])
        return tt, S, sd, sdd, S[ks]

    def fastest_cubic(self):
        def ok(T):
            t = np.linspace(0, T, 600); s, sd, sdd, _ = tu.cubic_time_scaling(T, t)
            return all(np.all(np.abs(self.torque(a, b, c)) <= self.tau_max + 1e-9) for a, b, c in zip(s, sd, sdd))
        lo, hi = 0.1, 20.0
        if not ok(hi): raise RuntimeError("no feasible cubic with T <= 20 s")
        for _ in range(40):
            mid = (lo + hi) / 2; lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi
