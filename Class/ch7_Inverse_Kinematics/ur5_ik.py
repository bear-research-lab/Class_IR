"""
Analytic inverse kinematics of the Universal Robots UR5
(Introduction to Robotics, Ch. 6 - case study)

Convention: standard (classic) DH, as published by Universal Robots
    T_{i-1,i} = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

Solution order:  theta1 -> theta5 -> theta6 -> (theta2, theta3, theta4)
Up to 8 solutions: shoulder (2) x wrist (2) x elbow (2).
"""
import numpy as np

# UR5 DH parameters [m, rad]
D     = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
A     = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])


def dh(theta, d, a, alpha):
    """One standard-DH link transform Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0.0,    sa,     ca,    d],
                     [0.0,   0.0,    0.0,  1.0]])


def fk(q):
    """Forward kinematics: joint angles (6,) -> T06 (4x4)."""
    T = np.eye(4)
    for i in range(6):
        T = T @ dh(q[i], D[i], A[i], ALPHA[i])
    return T


def wrap(x):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def ik(T06, eps=1e-9):
    """Analytic IK. Returns a list of joint vectors (each shape (6,))."""
    d1, d4, d5, d6 = D[0], D[3], D[4], D[5]
    a2, a3 = A[1], A[2]
    R, p = T06[:3, :3], T06[:3, 3]
    sols = []

    # --- Step 1: theta1 (shoulder left/right) -------------------------------
    # Wrist point p5 = p - d6*z6 must lie in the arm plane, which is at
    # distance d4 from the base axis:  s1*p5x - c1*p5y = d4
    p5 = p - d6 * R[:, 2]
    r = np.hypot(p5[0], p5[1])
    if r < abs(d4):
        return sols                                  # unreachable
    psi = np.arctan2(p5[1], p5[0])
    for th1 in (psi + np.arcsin(d4 / r), psi + np.pi - np.arcsin(d4 / r)):
        s1, c1 = np.sin(th1), np.cos(th1)

        # --- Step 2: theta5 (wrist up/down) ---------------------------------
        #   s1*px - c1*py = d4 + d6*cos(theta5)
        c5 = (s1*p[0] - c1*p[1] - d4) / d6
        if abs(c5) > 1 + eps:
            continue
        c5 = np.clip(c5, -1.0, 1.0)
        for th5 in (np.arccos(c5), -np.arccos(c5)):
            s5 = np.sin(th5)

            # --- Step 3: theta6 ---------------------------------------------
            # z1 = (s1, -c1, 0) seen from frame 6 is (s5*c6, -s5*s6, c5)
            v = R.T @ np.array([s1, -c1, 0.0])
            if abs(s5) < eps:                        # wrist singularity
                th6 = 0.0                            # theta6 is free: pick 0
            else:
                th6 = np.arctan2(-v[1] / s5, v[0] / s5)

            # --- Step 4: theta2, theta3, theta4 = planar 3R problem ---------
            T01 = dh(th1, d1, 0.0, ALPHA[0])
            T45 = dh(th5, d5, 0.0, ALPHA[4])
            T56 = dh(th6, d6, 0.0, ALPHA[5])
            T14 = np.linalg.inv(T01) @ T06 @ np.linalg.inv(T45 @ T56)
            x, y = T14[0, 3], T14[1, 3]              # 2R target in frame {1}
            phi = np.arctan2(T14[1, 0], T14[0, 0])   # theta2+theta3+theta4

            Dc = (x*x + y*y - a2*a2 - a3*a3) / (2*a2*a3)
            if abs(Dc) > 1 + eps:
                continue
            Dc = np.clip(Dc, -1.0, 1.0)
            for sgn in (1.0, -1.0):                  # elbow up/down
                th3 = np.arctan2(sgn*np.sqrt(1 - Dc*Dc), Dc)
                k1, k2 = a2 + a3*np.cos(th3), a3*np.sin(th3)
                th2 = np.arctan2(y, x) - np.arctan2(k2, k1)
                th4 = phi - th2 - th3
                sols.append(wrap([th1, th2, th3, th4, th5, th6]))
    return sols


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # 1) One example: pick joint angles, compute the pose, solve IK back
    q_true = np.radians([20, -70, 100, -120, -80, 30])
    T = fk(q_true)
    print("Target pose T06 =\n", T)

    sols = ik(T)
    print(f"\n{len(sols)} solutions (deg)            FK error")
    for i, q in enumerate(sols):
        err = np.linalg.norm(fk(q) - T)
        print(f"{i+1}: {np.degrees(q)}   {err:.1e}")

    # 2) Random test: every solution must reproduce the pose,
    #    and the original joint vector must be among the solutions
    rng = np.random.default_rng(0)
    worst, found = 0.0, 0
    for _ in range(1000):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = fk(q)
        S = ik(T)
        worst = max([worst] + [np.linalg.norm(fk(s) - T) for s in S])
        found += any(np.linalg.norm(wrap(s - q)) < 1e-6 for s in S)
    print(f"\nRandom test: worst FK error = {worst:.1e}, "
          f"original q recovered in {found}/1000 cases")
