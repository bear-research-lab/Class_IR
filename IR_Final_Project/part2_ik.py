"""Part 2: inverse kinematics. Test inverse_kinematics() in student_kinematics.py.

Without a file, this runs the graded test:
  Test 1 (random): targets made from random joint angles near the home pose; every solve
          starts from the home pose.
  Test 2 (task):   the grasp-site waypoints of the pipetting task for every layout,
          solved one after another, each starting from the previous solution.
A solve passes when the true pose error (checked with the simulator, not your FK) is
below 1e-4 and every joint is inside its limits. Add --task to then run the pipetting
task with the basic hand: every robot move uses your IK.

With a pose file (see trajectories/), the hand moves through the poses in the file in
straight lines, solving your IK at every simulation step with only 10 iterations. The
viewer draws the target frames (short axes) and your FK frame (long axes).

Examples:
  uv run part2_ik.py                                         # graded test
  uv run part2_ik.py --task                                  # ... then the pipetting task in the viewer
  uv run part2_ik.py --task --headless                       # ... without the viewer
  uv run part2_ik.py trajectories/poses_example.txt          # play a pose file in the viewer
  uv run part2_ik.py trajectories/poses_example.txt --joint-space   # one IK solve per waypoint
  uv run part2_ik.py trajectories/poses_example.txt --out my_joints.txt   # save the joint angles (a Part 1 input)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from part1_fk import PASS_POS, PASS_ROT, compare
from pipetting import run_trial
from rb5_sim.controller import HAND_FORWARD, HOME_Q, RB5, ViewerClosed, mat_to_quat, pose_error
from rb5_sim.env import HOLDER_SEAT_Z, HolderError
from rb5_sim.kinematics import DEFAULT_KINEMATICS, forbidden_imports, load_kinematics, read_pose_file, write_joint_file
from rb5_sim.layout_md import load_layouts
from rb5_sim.playback import FrameMarkers, open_robot, wait_for_close
from rb5_sim.task import APPROACH_DISTANCE, ASPIRATE_TIP_Z, DISPENSE_TIP_Z, TRANSIT_TIP_Z, tip_at

PASS_ERROR = 1e-4      # same threshold the controller uses for joint-space moves
RANDOM_SPREAD = 0.5    # rad, random targets come from HOME_Q +- this per joint


def check(robot: RB5, q_init, pos, rot) -> dict:
    """Solve one target and measure the result with the simulator, not with your FK."""
    start = time.perf_counter()
    q, _ = robot.ik(pos, rot, q_init)
    ms = (time.perf_counter() - start) * 1e3
    cur_pos, cur_rot = robot.sim_fk(q)
    err = pose_error(np.asarray(pos), mat_to_quat(rot), cur_pos, cur_rot)
    in_limits = bool(np.all(q >= robot.q_lo - 1e-9) and np.all(q <= robot.q_hi + 1e-9))
    return {"q": q, "pos_mm": np.linalg.norm(err[:3]) * 1e3, "rot_deg": np.degrees(np.linalg.norm(err[3:])),
            "passed": np.linalg.norm(err) < PASS_ERROR and in_limits, "in_limits": in_limits, "ms": ms}


def show(name: str, r: dict, verbose: bool) -> None:
    if verbose or not r["passed"]:
        limits = "" if r["in_limits"] else ", OUTSIDE JOINT LIMITS"
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {name}: {r['pos_mm']:.4f} mm, "
              f"{r['rot_deg']:.4f} deg, {r['ms']:.1f} ms{limits}")


def task_waypoints(layout) -> list[tuple[str, np.ndarray]]:
    holder_tip = np.array([*layout.holder, HOLDER_SEAT_Z])
    approach = np.array([APPROACH_DISTANCE, 0.0, 0.0])
    return [
        ("holder approach", tip_at(None, holder_tip) - approach),
        ("holder grasp", tip_at(None, holder_tip)),
        ("holder lift", tip_at(None, [*layout.holder, TRANSIT_TIP_Z])),
        ("water above", tip_at(None, [*layout.water, TRANSIT_TIP_Z])),
        ("water aspirate", tip_at(None, [*layout.water, ASPIRATE_TIP_Z])),
        ("beaker above", tip_at(None, [*layout.beaker, TRANSIT_TIP_Z])),
        ("beaker dispense", tip_at(None, [*layout.beaker, DISPENSE_TIP_Z])),
    ]


def summarize(name: str, results: list[dict]) -> None:
    passed = sum(r["passed"] for r in results)
    pos = max(r["pos_mm"] for r in results)
    rot = max(r["rot_deg"] for r in results)
    ms = np.mean([r["ms"] for r in results])
    print(f"{name}: {passed}/{len(results)} passed  (worst {pos:.4f} mm, {rot:.4f} deg; mean {ms:.1f} ms)")


def run_test(robot: RB5, layouts: dict, n: int, seed: int, verbose: bool) -> bool:
    rng = np.random.default_rng(seed)
    fk_ok = all(dp < PASS_POS and dr < PASS_ROT
                for dp, dr in (compare(robot, rng.uniform(-np.pi, np.pi, 6), robot.tool) for _ in range(20)))
    if not fk_ok:
        print("Warning: your forward_kinematics() doesn't match the simulator; fix Part 1 first "
              "(uv run part1_fk.py).\n")

    print(f"Test 1: {n} random targets (seed {seed})")
    random_results = []
    for i in range(n):
        q_true = np.clip(HOME_Q + rng.uniform(-RANDOM_SPREAD, RANDOM_SPREAD, 6), robot.q_lo, robot.q_hi)
        pos, rot = robot.sim_fk(q_true)
        r = check(robot, HOME_Q, pos, rot)
        show(f"target {i}", r, verbose)
        random_results.append(r)

    print(f"\nTest 2: pipetting waypoints for {len(layouts)} layouts")
    task_results = []
    for layout_name, layout in layouts.items():
        q = HOME_Q
        for name, pos in task_waypoints(layout):
            r = check(robot, q, pos, HAND_FORWARD)
            show(f"{layout_name} / {name}", r, verbose)
            task_results.append(r)
            q = r["q"]

    print()
    summarize("Random targets", random_results)
    summarize("Task waypoints", task_results)
    return all(r["passed"] for r in random_results + task_results)


def play(robot: RB5, viewer, path: Path, joint_space: bool, out: Path | None) -> None:
    waypoints = read_pose_file(path)
    targets = [(wp.value[:3, 3], wp.value[:3, :3]) for wp in waypoints]
    if viewer is not None:
        segments = [(a[0], b[0]) for a, b in zip(targets, targets[1:])]
        robot.sim.callbacks.append(FrameMarkers(viewer, robot, frames=targets, segments=segments))

    q0, err = robot.ik(*targets[0], HOME_Q, max_iters=1000)
    if err > PASS_ERROR:
        raise SystemExit(f"IK failed for the first pose in {path} (line {waypoints[0].line}): error {err:.2e}")
    robot.reset(q0)
    print(f"{len(waypoints)} waypoints from {path}, {'joint-space' if joint_space else 'straight-line'} moves\n")
    print(f"{'T [s]':>6}  {'joint angles [deg]':>44}  {'reached error (simulator)':>26}")
    rows, prev_t = [], waypoints[0].t
    for wp, (pos, rot) in zip(waypoints, targets):
        if wp is not waypoints[0]:
            try:
                robot.move_to(pos, rot, duration=wp.t - prev_t, settle=0.0, joint_space=joint_space)
            except RuntimeError as e:
                raise SystemExit(f"Line {wp.line}: {e}")
            prev_t = wp.t
        true_pos, true_rot = robot.sim_fk(robot.q_cmd)
        err = pose_error(pos, mat_to_quat(rot), true_pos, true_rot)
        print(f"{wp.t:6.2f}  {' '.join(f'{a:7.1f}' for a in np.degrees(robot.q_cmd)):>44}  "
              f"{np.linalg.norm(err[:3]) * 1e3:9.3f} mm {np.degrees(np.linalg.norm(err[3:])):7.3f} deg")
        rows.append((wp.t, robot.q_cmd.copy()))
    if out:
        write_joint_file(out, rows, f"Joint angles from your inverse_kinematics() for {path.name}.\n"
                                    "Input for Part 1: uv run part1_fk.py " + str(out))
        print(f"\nJoint angles saved to {out} (try: uv run part1_fk.py {out})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, nargs="?", help="pose file to play (omit for the graded test)")
    parser.add_argument("--joint-space", action="store_true",
                        help="with a file: one IK solve per waypoint and joint interpolation, not straight lines")
    parser.add_argument("--out", type=Path, help="with a file: save the joint angles at each waypoint")
    parser.add_argument("--task", action="store_true", help="test: then run the pipetting task with the basic hand")
    parser.add_argument("--headless", action="store_true", help="no viewer (with a file or --task)")
    parser.add_argument("--playback", type=float, default=1.0, help="viewer playback speed, e.g. 0.5")
    parser.add_argument("-n", type=int, default=50, help="test: number of random targets (default 50)")
    parser.add_argument("--seed", type=int, default=0, help="test: random seed (default 0)")
    parser.add_argument("--layouts", type=Path, default=Path("layouts.md"), help="Markdown file with layouts")
    parser.add_argument("-v", "--verbose", action="store_true", help="test: print every solve")
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    bad = forbidden_imports(args.kinematics)
    if bad:
        raise SystemExit(f"{args.kinematics} imports {', '.join(bad)}: only NumPy is allowed (see PART2_IK.md)")
    kin = load_kinematics(args.kinematics)
    layouts = load_layouts(args.layouts)
    try:
        if args.file is not None:
            with open_robot(kin, viewer=not args.headless, layouts=args.layouts, playback=args.playback) as (robot, v):
                play(robot, v, args.file, args.joint_space, args.out)
                wait_for_close(robot, v)
            return

        with open_robot(kin, viewer=False, layouts=args.layouts) as (robot, _):
            ok = run_test(robot, layouts, args.n, args.seed, args.verbose)
        print(f"\nPart 2 tests {'PASSED' if ok else 'FAILED'}")
        if args.task:
            name, layout = next(iter(layouts.items()))
            print(f"\nPipetting task with your IK and the basic hand (layout '{name}', no holder error)")
            try:
                report, _ = run_trial(layout, HolderError(), viewer=not args.headless, verbose=False,
                                      playback=args.playback, kinematics=kin)
            except RuntimeError as e:
                raise SystemExit(f"Task stopped: {e}")
            if args.headless:
                print(report)
            print(f"Task {'passed' if report.passed else 'FAILED'}")
    except NotImplementedError as e:
        raise SystemExit(f"Not implemented yet: {e}")
    except ViewerClosed:
        print("Viewer closed.")


if __name__ == "__main__":
    main()
