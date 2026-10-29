"""Part 1: forward kinematics. Test forward_kinematics() in student_kinematics.py.

Without a file, this runs the graded test: your forward_kinematics() is compared with
the simulator at random joint angles, for the flange itself, for the basic hand's
grasp site and for random tool frames.

With a joint file (see trajectories/), the robot moves through the joint angles in the
file. The viewer draws the grasp-site frame from YOUR forward kinematics (long axes,
x red / y green / z blue) and its trail; if your FK is right it stays on the hand.
At every waypoint the script prints your FK next to the simulator's pose.

Examples:
  uv run part1_fk.py                                          # graded test
  uv run part1_fk.py trajectories/joints_example.txt          # play a joint file in the viewer
  uv run part1_fk.py trajectories/joints_example.txt --headless
  uv run part1_fk.py trajectories/joints_example.txt --out my_poses.txt   # save your FK poses (a Part 2 input)
"""

import argparse
from pathlib import Path

import numpy as np

from rb5_sim.controller import RB5, ViewerClosed
from rb5_sim.kinematics import (DEFAULT_KINEMATICS, forbidden_imports, load_kinematics, pose_matrix,
                                read_joint_file, write_pose_file)
from rb5_sim.playback import FrameMarkers, open_robot, wait_for_close

PASS_POS = 1e-6   # m
PASS_ROT = 1e-6   # rad


def compare(robot: RB5, q: np.ndarray, tool: np.ndarray) -> tuple[float, float]:
    """Position [m] and rotation [rad] error of your FK against the simulator."""
    T = np.asarray(robot.kin.forward_kinematics(np.array(q, dtype=float), tool.copy()), dtype=float)
    if T.shape != (4, 4):
        raise SystemExit(f"forward_kinematics must return a 4x4 matrix, got shape {T.shape}")
    true = pose_matrix(*robot.sim_fk(q, site="tcp")) @ tool
    cos = np.clip((np.trace(T[:3, :3] @ true[:3, :3].T) - 1) / 2, -1, 1)
    return float(np.linalg.norm(T[:3, 3] - true[:3, 3])), float(np.arccos(cos))


def random_tool(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = rng.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return pose_matrix(rng.uniform(-0.2, 0.2, 3), np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)


def run_test(robot: RB5, n: int, seed: int, verbose: bool) -> bool:
    rng = np.random.default_rng(seed)
    tools = {"flange (tool = identity)": lambda: np.eye(4),
             "basic hand grasp site": lambda: robot.tool,
             "random tool frames": lambda: random_tool(rng)}
    all_passed = True
    print(f"Random joint angles in [-180, 180] deg, {n} per case (seed {seed}); "
          f"pass: < {PASS_POS * 1e3:g} mm and < {np.degrees(PASS_ROT):.1e} deg\n")
    for name, tool in tools.items():
        passed, worst_p, worst_r = 0, 0.0, 0.0
        for i in range(n):
            q = rng.uniform(-np.pi, np.pi, 6)
            dp, dr = compare(robot, q, tool())
            ok = dp < PASS_POS and dr < PASS_ROT
            passed += ok
            worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
            if verbose or (not ok and i - passed < 3):  # every case, or the first 3 failures
                print(f"  [{'PASS' if ok else 'FAIL'}] q = {np.round(np.degrees(q), 1)} deg: "
                      f"{dp * 1e3:.4f} mm, {np.degrees(dr):.4f} deg")
        all_passed &= passed == n
        print(f"{name:<26} {passed:>3}/{n} passed  (worst {worst_p * 1e3:.2e} mm, {np.degrees(worst_r):.2e} deg)")
    return all_passed


def play(robot: RB5, viewer, path: Path, out: Path | None) -> None:
    waypoints = read_joint_file(path)
    if viewer is not None:
        robot.sim.callbacks.append(FrameMarkers(viewer, robot))
    robot.reset(waypoints[0].value)
    print(f"{len(waypoints)} waypoints from {path}\n")
    print(f"{'T [s]':>6}  {'your FK x, y, z [mm]':>26}  {'simulator x, y, z [mm]':>26}  {'error':>17}")
    rows, prev_t = [], waypoints[0].t
    for wp in waypoints:
        if wp is not waypoints[0]:
            robot.move_joints(wp.value, duration=wp.t - prev_t, settle=0.0)
            prev_t = wp.t
        pos, rot = robot.fk(wp.value)
        true_pos, true_rot = robot.sim_fk(wp.value)
        dr = np.degrees(np.arccos(np.clip((np.trace(rot @ true_rot.T) - 1) / 2, -1, 1)))
        fmt = lambda p: ", ".join(f"{v * 1e3:7.1f}" for v in p)  # noqa: E731
        print(f"{wp.t:6.2f}  {fmt(pos):>26}  {fmt(true_pos):>26}  "
              f"{np.linalg.norm(pos - true_pos) * 1e3:6.2f} mm {dr:5.2f} deg")
        rows.append((wp.t, pose_matrix(pos, rot)))
    if out:
        write_pose_file(out, rows, f"Grasp-site poses from your forward_kinematics() for {path.name}.\n"
                                   "Input for Part 2: uv run part2_ik.py " + str(out))
        print(f"\nYour FK poses saved to {out} (try: uv run part2_ik.py {out})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, nargs="?", help="joint file to play (omit for the graded test)")
    parser.add_argument("--out", type=Path, help="with a file: save your FK poses as a pose file")
    parser.add_argument("--headless", action="store_true", help="with a file: no viewer")
    parser.add_argument("--playback", type=float, default=1.0, help="viewer playback speed, e.g. 0.5")
    parser.add_argument("-n", type=int, default=100, help="test: random joint angles per case (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="test: random seed (default 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="test: print every case")
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    bad = forbidden_imports(args.kinematics)
    if bad:
        raise SystemExit(f"{args.kinematics} imports {', '.join(bad)}: only NumPy is allowed (see PART1_FK.md)")
    kin = load_kinematics(args.kinematics)
    try:
        with open_robot(kin, viewer=bool(args.file) and not args.headless, playback=args.playback) as (robot, v):
            if args.file is None:
                ok = run_test(robot, args.n, args.seed, args.verbose)
                print(f"\nPart 1 {'PASSED' if ok else 'FAILED'}")
                return
            play(robot, v, args.file, args.out)
            wait_for_close(robot, v)
    except NotImplementedError as e:
        raise SystemExit(f"Not implemented yet: {e}")
    except ViewerClosed:
        print("Viewer closed.")


if __name__ == "__main__":
    main()
