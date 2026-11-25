"""
robots.py  --  robot models used by the animation scripts.

UR5 6R arm in product-of-exponentials form (MR Example 4.5), with
  * forward kinematics  T(theta) = e^[S1]th1 ... e^[S6]th6 M
  * a "skeleton" (points along the links) for drawing the robot
  * space Jacobian (Ch. 5) and Newton-Raphson numerical IK (Ch. 6)
"""
import numpy as np
import traj_utils as tu

W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095


def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6)); A[:3, :3] = R; A[3:, 3:] = R; A[3:, :3] = tu.skew(p) @ R
    return A


def vec_to_se3(V):
    out = np.zeros((4, 4)); out[:3, :3] = tu.skew(V[:3]); out[:3, 3] = V[3:]
    return out


def se3_to_vec(m):
    return np.concatenate([tu.unskew(m[:3, :3]), m[:3, 3]])


class UR5:
    def __init__(self):
        self.M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
        self.Slist = np.array([[0, 0, 1, 0, 0, 0],
                               [0, 1, 0, -H1, 0, 0],
                               [0, 1, 0, -H1, 0, L1],
                               [0, 1, 0, -H1, 0, L1 + L2],
                               [0, 0, -1, -W1, L1 + L2, 0],
                               [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T      # 6 x 6, column i = S_i
        # points on the robot at the zero configuration, for drawing.
        # point k rides on the link after joint k (moves with joints 1..k)
        self.home_pts = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
                                  [L1 + L2, W1, H1], [L1 + L2, W1, H1 - H2], [L1 + L2, W1 + W2, H1 - H2]], float)
        self.moves_with = [0, 1, 2, 3, 4, 5, 6]

    def _exps(self, theta):
        """Cumulative products e^[S1]th1 ... e^[Sk]thk for k = 0..6."""
        out = [np.eye(4)]
        for i in range(6):
            out.append(out[-1] @ tu.exp6(vec_to_se3(self.Slist[:, i] * theta[i])))
        return out

    def fk(self, theta):
        return self._exps(theta)[-1] @ self.M

    def skeleton(self, theta):
        """3D points along the robot (for plotting the links)."""
        E = self._exps(theta)
        return np.array([(E[k] @ np.append(p, 1))[:3] for k, p in zip(self.moves_with, self.home_pts)])

    def jacobian_space(self, theta):
        E = self._exps(theta)
        return np.column_stack([adjoint(E[i]) @ self.Slist[:, i] for i in range(6)])

    def ik(self, T_desired, theta0, eomg=1e-6, ev=1e-6, max_iter=50):
        """Newton-Raphson IK in the space frame (MR Sec. 6.2.2). Returns (theta, success)."""
        theta = np.array(theta0, float)
        for _ in range(max_iter):
            Tsb = self.fk(theta)
            Vb = se3_to_vec(tu.log6(tu.trans_inv(Tsb) @ T_desired))
            Vs = adjoint(Tsb) @ Vb
            if np.linalg.norm(Vs[:3]) < eomg and np.linalg.norm(Vs[3:]) < ev:
                return theta, True
            theta = theta + np.linalg.pinv(self.jacobian_space(theta)) @ Vs
        return theta, False

    def ik_along(self, X_list, theta0):
        """IK for a sequence of poses, warm-starting each from the previous solution (smooth tracking)."""
        th = np.array(theta0, float); out = []; ok_all = True
        for X in X_list:
            th, ok = self.ik(X, th); ok_all &= ok; out.append(th.copy())
        return np.array(out), ok_all
