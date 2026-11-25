"""
UR5 analytic inverse kinematics, step by step (PoE version)
============================================================

Companion code for IR Ch.07 Inverse Kinematics (Case 2: three parallel axes).

    uv run ur5_ik_explained.py              # explain one example, then list all 8 solutions
    uv run ur5_ik_explained.py --plot       # also draw the 8 configurations
    uv run ur5_ik_explained.py --test 1000  # random round-trip test: FK(IK(T)) == T ?

What the script does
--------------------
0. Build the UR5 in PoE form from its ZERO POSE (all theta_i = 0):
   read omega_i and q_i in that pose, then v_i = -omega_i x q_i, and find M.
1. theta_1 : the wrist point p5 = p - d6*z6 stays d4 away from the arm plane.
2. theta_5 : the tool point p satisfies  p . z1 = d4 + d6 cos(theta_5).
3. theta_6 : z1 seen from the tool frame depends only on theta_5 and theta_6.
4. theta_2, theta_3, theta_4 : peel the known joints off both ends of the PoE chain,
       e^[S2]t2 e^[S3]t3 e^[S4]t4 = e^-[S1]t1  T  M^-1  e^-[S6]t6  e^-[S5]t5
   and solve the planar 3R arm that is left (law of cosines).

Key idea: joints 2, 3, 4 rotate about z1, so they can change neither
(a) how far a point is along z1, nor (b) the direction z1 itself.
That is why theta_1, theta_5, theta_6 can be found first.

Requirements: numpy (and matplotlib for --plot).
"""

import argparse
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
# UR5 dimensions [m]  (names match the lecture slides)
# ---------------------------------------------------------------------------
d1 = 0.089159   # shoulder height
L1 = 0.425      # upper arm   |a2|
L2 = 0.39225    # forearm     |a3|
d4 = 0.10915    # offset of the wrist from the arm plane
d5 = 0.09465    # wrist height (joint 5 -> joint 6)
d6 = 0.0823     # tool length

# ---------------------------------------------------------------------------
# Small Lie-group helpers (same as the Modern Robotics library)
# ---------------------------------------------------------------------------
def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def exp6(S, theta):
    """e^{[S] theta} for a revolute screw axis S = (omega, v), |omega| = 1."""
    w, v = S[:3], S[3:]
    W = skew(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Step 0: PoE model, read from the ZERO POSE
# ---------------------------------------------------------------------------
# At theta = 0 the arm is stretched horizontally along -x_s at height d1.
# Joints 2, 3, 4 point along -y_s, so their q_i share z = d1 and can use y = 0.
omega = np.array([[0, 0, 1],     # joint 1: vertical
                  [0, -1, 0],    # joint 2: shoulder
                  [0, -1, 0],    # joint 3: elbow
                  [0, -1, 0],    # joint 4: wrist 1
                  [0, 0, -1],    # joint 5: wrist 2 (points down)
                  [0, -1, 0]],   # joint 6: wrist 3
                 dtype=float)
q = np.array([[0, 0, 0],
              [0, 0, d1],
              [-L1, 0, d1],
              [-L1 - L2, 0, d1],
              [-L1 - L2, -d4, d1],
              [-L1 - L2, -d4, d1 - d5]])
S = np.array([np.r_[w, -np.cross(w, qi)] for w, qi in zip(omega, q)])  # v_i = -omega_i x q_i

M = np.array([[1, 0, 0, -L1 - L2],          # x_b =  x_s
              [0, 0, -1, -d4 - d6],         # y_b =  z_s,  z_b = -y_s
              [0, 1, 0, d1 - d5],
              [0, 0, 0, 1]], dtype=float)


def fk(theta):
    T = np.eye(4)
    for Si, ti in zip(S, theta):
        T = T @ exp6(Si, ti)
    return T @ M


def joint_positions(theta):
    """Where each joint (and the tool) is for the given angles, for plotting."""
    pts, T = [], np.eye(4)
    for i in range(6):
        pts.append((T @ np.r_[q[i], 1])[:3])  # joint i is moved by joints 1..i-1 only
        T = T @ exp6(S[i], theta[i])
    pts.append(fk(theta)[:3, 3])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Steps 1-4: analytic IK
# ---------------------------------------------------------------------------
def say(verbose, *msg):
    if verbose:
        print(*msg)


def step1_theta1(T, verbose=False):
    p, z6 = T[:3, 3], T[:3, 2]
    p5 = p - d6 * z6                         # back off from the tool along z6
    r = np.hypot(p5[0], p5[1])
    psi = np.arctan2(p5[1], p5[0])
    say(verbose, "\n[Step 1] theta1 from the wrist point p5")
    say(verbose, f"  p5 = p - d6*z6 = {p5}")
    say(verbose, f"  top view: r = |p5_xy| = {r:.4f},  psi = atan2(p5y, p5x) = {np.degrees(psi):.2f} deg")
    if r < d4:
        say(verbose, "  r < d4: the wrist point is too close to the base axis -> no solution")
        return []
    sols = [wrap(psi + np.pi / 2 + s * np.arccos(d4 / r)) for s in (+1, -1)]
    say(verbose, f"  p5 . z1 = d4  ->  theta1 = psi + pi/2 +/- acos(d4/r) = "
                 f"{np.degrees(sols[0]):.2f}, {np.degrees(sols[1]):.2f} deg  (shoulder left / right)")
    return sols


def z1_of(t1):
    """Common direction of axes 2, 3, 4 after joint 1 turns by t1."""
    return np.array([np.sin(t1), -np.cos(t1), 0.0])


def step2_theta5(T, t1, verbose=False):
    p, z1 = T[:3, 3], z1_of(t1)
    c5 = (p @ z1 - d4) / d6
    say(verbose, f"\n[Step 2] theta5 for theta1 = {np.degrees(t1):.2f} deg")
    say(verbose, f"  z1 = {z1},  p . z1 = {p @ z1:.4f}")
    say(verbose, f"  p . z1 = d4 + d6 cos(theta5)  ->  cos(theta5) = {c5:.4f}")
    if abs(c5) > 1 + 1e-9:
        say(verbose, "  |cos| > 1: this theta1 branch cannot reach the pose")
        return []
    a = np.arccos(np.clip(c5, -1, 1))
    say(verbose, f"  theta5 = +/- {np.degrees(a):.2f} deg  (wrist up / flipped)")
    return [a, -a]


def step3_theta6(T, t1, t5, verbose=False, eps=1e-6):
    R, z1 = T[:3, :3], z1_of(t1)
    s5 = np.sin(t5)
    say(verbose, f"\n[Step 3] theta6 for theta5 = {np.degrees(t5):.2f} deg")
    if abs(s5) < eps:
        say(verbose, "  sin(theta5) = 0: wrist singularity, theta6 is free -> pick theta6 = 0")
        return 0.0
    # z1 seen from the tool frame: R^T z1 = (s5 c6, -s5 s6, c5)
    x6z1, y6z1 = R[:, 0] @ z1, R[:, 1] @ z1
    t6 = np.arctan2(-y6z1 / s5, x6z1 / s5)
    say(verbose, f"  x6 . z1 = {x6z1:.4f},  y6 . z1 = {y6z1:.4f}")
    say(verbose, f"  theta6 = atan2(-y6.z1 / s5, x6.z1 / s5) = {np.degrees(t6):.2f} deg")
    return t6


def step4_planar(T, t1, t5, t6, verbose=False):
    # Peel the known joints off both ends of the chain (PoE, no DH frames needed):
    #   e^[S2]t2 e^[S3]t3 e^[S4]t4 = e^-[S1]t1 T M^-1 e^-[S6]t6 e^-[S5]t5
    T234 = inv(exp6(S[0], t1)) @ T @ inv(M) @ inv(exp6(S[5], t6)) @ inv(exp6(S[4], t5))
    # (1) Rotation part: a rotation about -y by phi = t2 + t3 + t4
    phi = np.arctan2(T234[2, 0], T234[0, 0])
    # (2) Joint 4 is not moved by its own rotation, so T234 * q4 = where joints 2, 3 put joint 4
    p4 = (T234 @ np.r_[q[3], 1])[:3]
    u = np.array([p4[0] - q[1][0], p4[2] - q[1][2]])   # planar (x, z) vector: joint 2 -> joint 4
    say(verbose, f"\n[Step 4] planar 3R arm (theta1, theta5, theta6 = "
                 f"{np.degrees(t1):.1f}, {np.degrees(t5):.1f}, {np.degrees(t6):.1f} deg)")
    say(verbose, f"  phi = theta2 + theta3 + theta4 = {np.degrees(phi):.2f} deg")
    say(verbose, f"  joint 4 relative to joint 2 in the arm plane: u = {u}, |u| = {np.linalg.norm(u):.4f}")
    c3 = (u @ u - L1**2 - L2**2) / (2 * L1 * L2)          # law of cosines
    say(verbose, f"  law of cosines: cos(theta3) = (|u|^2 - L1^2 - L2^2) / (2 L1 L2) = {c3:.4f}")
    if abs(c3) > 1 + 1e-9:
        say(verbose, "  |cos| > 1: out of reach for this branch")
        return []
    sols = []
    for t3 in (np.arccos(np.clip(c3, -1, 1)), -np.arccos(np.clip(c3, -1, 1))):
        # at zero pose the arm points along -x: w = vector joint 2 -> joint 4 after rotating joint 3 only
        w = np.array([-L1 - L2 * np.cos(t3), -L2 * np.sin(t3)])
        t2 = np.arctan2(u[1], u[0]) - np.arctan2(w[1], w[0])
        t4 = phi - t2 - t3
        sols.append((wrap(t2), wrap(t3), wrap(t4)))
        say(verbose, f"  theta3 = {np.degrees(t3):7.2f} -> theta2 = {np.degrees(wrap(t2)):7.2f}, "
                     f"theta4 = phi - theta2 - theta3 = {np.degrees(wrap(t4)):7.2f} deg")
    return sols


def ik_ur5(T, verbose=False):
    """All analytic IK solutions (up to 8) for target pose T (4x4)."""
    sols = []
    for t1 in step1_theta1(T, verbose):
        for t5 in step2_theta5(T, t1, verbose):
            t6 = step3_theta6(T, t1, t5, verbose)
            for t2, t3, t4 in step4_planar(T, t1, t5, t6, verbose):
                sols.append(np.array([t1, t2, t3, t4, t5, t6]))
    return sols


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def explain_model():
    print("=" * 72)
    print("Step 0. PoE model of the UR5, read from the zero pose (all theta = 0)")
    print("=" * 72)
    print(" i   omega_i          q_i                      v_i = -omega_i x q_i")
    for i in range(6):
        print(f" {i+1}  {omega[i]}  {q[i]}  {S[i][3:]}")
    print("M =\n", M)
    print("Note: q2, q3, q4 share z = d1 because the arm is horizontal at theta = 0.")


def plot_solutions(sols, T):
    import matplotlib.pyplot as plt
    n = len(sols)
    fig = plt.figure(figsize=(4 * min(n, 4), 4 * ((n + 3) // 4)))
    for k, th in enumerate(sols):
        ax = fig.add_subplot((n + 3) // 4, min(n, 4), k + 1, projection="3d")
        P = joint_positions(th)
        ax.plot([0, 0], [0, 0], [-0.1, 0], color="gray", lw=8)
        ax.plot(*P.T, "-o", color="#156082", lw=3, ms=4)
        ax.scatter(*T[:3, 3], color="#D9622B", s=60)
        ax.quiver(*T[:3, 3], *(0.1 * T[:3, 2]), color="#D9622B")
        ax.set_title(f"#{k+1}: " + ", ".join(f"{np.degrees(a):.0f}" for a in th), fontsize=8)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.4, 1.0)
        ax.set_box_aspect((1, 1, 0.9)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("All UR5 IK solutions for the same target pose (orange: tool position and z-axis)")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="draw all solutions")
    ap.add_argument("--test", type=int, default=0, help="number of random round-trip tests")
    args = ap.parse_args()

    explain_model()

    theta_true = np.radians([30, -60, 80, -110, 50, 20])
    T = fk(theta_true)
    print("\nTarget pose T = FK(theta_true), theta_true [deg] =", np.degrees(theta_true))
    print(T)

    print("\n" + "=" * 72 + "\nSolving IK step by step\n" + "=" * 72)
    sols = ik_ur5(T, verbose=True)

    print("\n" + "=" * 72 + f"\n{len(sols)} solutions  (check: ||FK(theta) - T||)\n" + "=" * 72)
    for k, th in enumerate(sols):
        err = np.linalg.norm(fk(th) - T)
        same = "  <- theta_true" if np.allclose(wrap(th - theta_true), 0, atol=1e-6) else ""
        print(f" #{k+1}: " + " ".join(f"{np.degrees(a):8.2f}" for a in th) + f"   err = {err:.1e}{same}")

    if args.test:
        rng = np.random.default_rng(0)
        worst, misses = 0.0, 0
        for _ in range(args.test):
            th = rng.uniform(-np.pi, np.pi, 6)
            Tt = fk(th)
            s = ik_ur5(Tt)
            if not s:
                misses += 1
                continue
            worst = max(worst, max(np.linalg.norm(fk(x) - Tt) for x in s))
            if not any(np.allclose(wrap(x - th), 0, atol=1e-5) for x in s):
                misses += 1
        print(f"\nRandom test: {args.test} poses, worst FK error {worst:.1e}, "
              f"original angles not recovered in {misses} cases")

    if args.plot:
        plot_solutions(sols, T)


if __name__ == "__main__":
    main()
