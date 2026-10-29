"""RB5 pipetting task in MuJoCo.

Object positions come from a Markdown file (default: layouts.md); see
rb5_sim/layout_md.py for the format.

Examples:
  uv run pipetting.py                              # viewer, first layout in layouts.md
  uv run pipetting.py --layout "Practice 2"        # pick a layout by name (or by index: --layout 1)
  uv run pipetting.py --layouts final.md           # positions from another file
  uv run pipetting.py --no-error --plot forces.png # nominal holder pose, save a force plot
  uv run pipetting.py --headless --trials 20       # batch validation, no window
  uv run pipetting.py --graph overlay              # live graphs inside the viewer instead of a window
  uv run pipetting.py --playback 0.5               # watch in slow motion (same physics)
  uv run pipetting.py --hand hands/team1/hand.xml  # test another hand design
  uv run pipetting.py --hand hands/team1/hand.xml --seed 3 --headless --video part3.mp4   # record a video

The robot moves with your kinematics from student_kinematics.py, so this needs Parts 1 and 2.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from rb5_sim.controller import RB5, Sim, ViewerClosed
from rb5_sim.env import DEFAULT_HAND, HolderError, build_model
from rb5_sim.forces import ForceMonitor
from rb5_sim.kinematics import DEFAULT_KINEMATICS, load_kinematics
from rb5_sim.layout_md import load_layouts
from rb5_sim.task import Probe, run_pipetting
from rb5_sim.video import VideoRecorder
from rb5_sim.viewer_plots import GraphWindow, ViewerOverlay


def _record(model, video: Path | None):
    return VideoRecorder(model, video) if video else nullcontext()


def run_trial(layout, error, viewer: bool, verbose: bool, graph: str = "window", hand=DEFAULT_HAND,
              playback: float = 1.0, motion_scale: float = 1.0, kinematics=None, video: Path | None = None):
    """video: also record the run to this MP4 file."""
    model = build_model(layout, error, hand=hand)
    data = mujoco.MjData(model)
    forces = ForceMonitor(model)
    log = print if verbose else (lambda *_: None)

    if not viewer:
        sim = Sim(model, data)
        sim.callbacks.append(forces)
        robot = RB5(sim, time_scale=motion_scale, kinematics=kinematics)
        robot.reset()
        with _record(model, video) as rec:
            if rec:
                sim.callbacks.append(rec)
            return run_pipetting(robot, layout, Probe(model, data, forces), log), forces

    graph_window = GraphWindow(model, forces) if graph == "window" else None
    with mujoco.viewer.launch_passive(model, data) as v:
        sim = Sim(model, data, viewer=v, playback=playback)
        sim.callbacks.append(forces)
        sim.callbacks.append(ViewerOverlay(v, model, forces, plots=graph == "overlay"))
        if graph_window:
            sim.callbacks.append(graph_window)
        robot = RB5(sim, time_scale=motion_scale, kinematics=kinematics)
        robot.reset()
        v.sync()
        with _record(model, video) as rec:
            if rec:
                sim.callbacks.append(rec)
            report = run_pipetting(robot, layout, Probe(model, data, forces), log)
        forces.phase = "finished"
        print(report)
        print(forces.summary())
        print("Close the viewer window to exit.")
        while v.is_running():
            sim.step()
    if graph_window:
        print("Close the graph window to exit.")
        graph_window.wait_closed()
    return report, forces


def pick_layout(layouts: dict, key: str):
    if key in layouts:
        return key, layouts[key]
    if key.lstrip("-").isdigit() and -len(layouts) <= int(key) < len(layouts):
        name = list(layouts)[int(key)]
        return name, layouts[name]
    raise SystemExit(f"Unknown layout '{key}'. Available: {', '.join(layouts)}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND, help="hand MJCF file (default: baseline hand)")
    parser.add_argument("--playback", type=float, default=1.0,
                        help="viewer playback speed, e.g. 0.5 = half speed (does not change the physics)")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="make every robot move this many times slower, e.g. 2 (changes the run; grade with 1)")
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS,
                        help="file with forward_kinematics/inverse_kinematics (default: student_kinematics.py)")


def kinematics_from_args(args):
    """Load the kinematics module, or stop with a clear message if it isn't ready."""
    kin = load_kinematics(args.kinematics)
    try:
        kin.inverse_kinematics(np.eye(4), np.zeros(6), np.eye(4), max_iters=1)
    except NotImplementedError as e:
        raise SystemExit(f"{e}\nThe robot moves with your kinematics: finish Parts 1 and 2 first "
                         f"(uv run part1_fk.py, uv run part2_ik.py).")
    return kin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", type=Path, default=Path("layouts.md"), help="Markdown file with layouts")
    parser.add_argument("--layout", default="0", help="layout name or index in the file (default: first)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the holder error")
    parser.add_argument("--no-error", action="store_true", help="put the holder exactly at its nominal pose")
    parser.add_argument("--headless", action="store_true", help="no viewer")
    parser.add_argument("--trials", type=int, default=1, help="number of random trials (headless)")
    parser.add_argument("--plot", type=Path, help="save a force plot of the (last) trial, e.g. forces.png")
    parser.add_argument("--csv", type=Path, help="save the force log of the (last) trial as CSV")
    parser.add_argument("--video", type=Path, help="record the run to this MP4 file (one trial only)")
    parser.add_argument("--graph", choices=("window", "overlay", "none"), default="window",
                        help="live force/z graphs: own window (default), inside the viewer, or none")
    add_common_args(parser)
    args = parser.parse_args()
    if args.video and args.trials != 1:
        parser.error("--video records a single trial; drop --trials")

    kinematics = kinematics_from_args(args)
    name, layout = pick_layout(load_layouts(args.layouts), args.layout)
    print(f"Layout '{name}' from {args.layouts}: {layout}")
    print(f"Hand: {args.hand}")
    rng = np.random.default_rng(args.seed)
    passed = 0
    for i in range(args.trials):
        error = HolderError() if args.no_error else HolderError.sample(rng)
        print(f"Trial {i + 1}/{args.trials}: holder error {error}")
        try:
            report, forces = run_trial(layout, error, viewer=not args.headless, verbose=args.trials == 1,
                                       graph=args.graph, hand=args.hand, playback=args.playback,
                                       motion_scale=args.motion_scale, kinematics=kinematics,
                                       video=args.video)
        except ViewerClosed:
            print("Viewer closed.")
            return
        if args.headless:
            print(report)
            if args.trials == 1:
                print(forces.summary())
        passed += report.passed
    print(f"\n{passed}/{args.trials} trials passed")

    if args.csv:
        forces.save_csv(args.csv)
        print(f"Force log saved to {args.csv}")
    if args.plot:
        forces.plot(args.plot)
        print(f"Force plot saved to {args.plot}")


if __name__ == "__main__":
    main()
