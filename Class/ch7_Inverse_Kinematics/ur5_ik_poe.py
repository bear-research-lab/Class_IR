"""
Analytic inverse kinematics of the UR5 in product-of-exponentials (PoE) form
(Introduction to Robotics, Ch. 6 - case study)

Model: space-form PoE with the zero configuration of Lynch & Park's UR5 example
(arm stretched along +x_s, joints 2-4 about +y_s), with UR5 link dimensions:
    T(theta) = exp([S1]th1) exp([S2]th2) ... exp([S6]th6) M

Idea: apply both sides of  exp(...)...exp(...) = X M^-1  to points that some
joints cannot move, so those joints drop out of the equation.
Order: theta1 -> theta5 -> theta6 -> (theta2, theta3, theta4). Up to 8 solutions.
"""
import numpy as np

# UR5 dimensions [m]
W1, W2 = 0.10915, 0.0823      # offsets along y_s (shoulder, tool)
L1, L2 = 0.425, 0.39225       # upper arm, forearm
H1, H2 = 0.089159, 0.09465    # base height, wrist height

# Zero configuration M and screw axes S_i = (omega_i, v_i), v_i = -omega_i x q_i
M = np.array([[-1, 0, 0, L1 + L2],
              [ 0, 0, 1, W1 + W2],
              [ 0, 1, 0, H1 - H2],
              [ 0, 0, 0, 1.0]])
W = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 1, 0]], float)
Q = np.array([[0, 0, 0], [0, 0, H1], [L1, 0, H1], [L1 + L2, 0, H1],
              [L1 + L2, W1, 0], [L1 + L2, 0, H1 - H2]], float)
S = np.hstack([W, -np.cross(W, Q)])            # 6 x 6, row i = S_i


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp6(s, th):
    """exp([S] th) for a revolute screw axis (|omega| = 1), Ch. 5 Eq. (1.2)."""
    w, v = s[:3], s[3:]
    K = skew(w)
    R = np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K@K
    G = np.eye(3)*th + (1 - np.cos(th))*K + (th - np.sin(th))*K@K
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = G @ v
    return T


def fk(q):
    """Space-form PoE forward kinematics."""
    T = np.eye(4)
    for s, th in zip(S, q):
        T = T @ exp6(s, th)
    return T @ M


def wrap(x):
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def ik(X, eps=1e-9):
    R, p = X[:3, :3], X[:3, 3]
    sols = []

    # --- Step 1: theta1 ------------------------------------------------------
    # q5 = p - W2 * z_b lies on the axes of joints 5 and 6, so they cannot
    # move it. Joints 2-4 rotate about axes parallel to n = Rz(th1) y_s, so
    # they keep its n-component:  n(th1) . q5 = W1,  n = (-s1, c1, 0)
    q5 = p - W2 * R[:, 2]
    r, psi = np.hypot(q5[0], q5[1]), np.arctan2(q5[1], q5[0])
    if r < W1:
        return sols
    for th1 in (psi - np.arcsin(W1 / r), psi - np.pi + np.arcsin(W1 / r)):
        n = np.array([-np.sin(th1), np.cos(th1), 0.0])

        # --- Step 2: theta5 --------------------------------------------------
        # tool axis z_b tilts away from n by theta5:  n . z_b = cos(theta5)
        c5 = n @ R[:, 2]
        if abs(c5) > 1 + eps:
            continue
        c5 = np.clip(c5, -1, 1)
        for th5 in (np.arccos(c5), -np.arccos(c5)):
            s5 = np.sin(th5)

            # --- Step 3: theta6 ----------------------------------------------
            # n seen from the body frame: R^T n = (s5 c6, -s5 s6, c5)
            v = R.T @ n
            th6 = 0.0 if abs(s5) < eps else np.arctan2(-v[1] / s5, v[0] / s5)

            # --- Step 4: theta2, theta3, theta4 (planar 3R in the xz-plane) --
            # exp([S2]th2) exp([S3]th3) exp([S4]th4) = g1   (known)
            g1 = (exp6(S[0], -th1) @ X @ np.linalg.inv(M)
                  @ exp6(S[5], -th6) @ exp6(S[4], -th5))
            # q4 is on the joint-4 axis:  g1 q4 = exp([S2]th2) exp([S3]th3) q4
            q4 = np.array([L1 + L2, 0, H1, 1.0])
            w = g1 @ q4
            x, y = w[0], H1 - w[2]                    # planar coords (CCW = +th)
            phi = np.arctan2(g1[0, 2], g1[0, 0])      # rot(g1) = Ry(th2+th3+th4)
            D = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
            if abs(D) > 1 + eps:
                continue
            D = np.clip(D, -1, 1)
            for sgn in (1, -1):
                th3 = np.arctan2(sgn*np.sqrt(1 - D*D), D)
                th2 = np.arctan2(y, x) - np.arctan2(L2*np.sin(th3), L1 + L2*np.cos(th3))
                th4 = phi - th2 - th3
                sols.append(wrap([th1, th2, th3, th4, th5, th6]))
    return sols


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q_true = np.radians([30, -60, 90, -45, 60, 20])
    X = fk(q_true)
    print("Target X =\n", X)
    sols = ik(X)
    print(f"\n{len(sols)} solutions (deg)            FK error")
    for i, q in enumerate(sols):
        print(f"{i+1}: {np.degrees(q)}   {np.linalg.norm(fk(q) - X):.1e}")

    rng = np.random.default_rng(0)
    worst, found = 0.0, 0
    for _ in range(1000):
        q = rng.uniform(-np.pi, np.pi, 6)
        X = fk(q)
        Sq = ik(X)
        worst = max([worst] + [np.linalg.norm(fk(s) - X) for s in Sq])
        found += any(np.linalg.norm(wrap(s - q)) < 1e-6 for s in Sq)
    print(f"\nRandom test: worst FK error = {worst:.1e}, "
          f"original q recovered in {found}/1000 cases")
