"""Evaluate the hand on several pipetting experiments with random pipette thickness.

Each experiment draws a random pipette (shaft and bulb diameter), a random holder
pose error, and a layout from the Markdown file, then runs the full task and
scores it with the reward in rb5_sim/reward.py. The hand's score is the mean
reward over all experiments.

Examples:
  uv run evaluate_hand.py                          # 5 experiments, random seed
  uv run evaluate_hand.py --seed 0                 # reproducible (compare hand designs with the same seed)
  uv run evaluate_hand.py -n 10 --plot eval.png    # 10 experiments + force traces plot
  uv run evaluate_hand.py --layout "Practice 2"    # all experiments on one layout
  uv run evaluate_hand.py -n 10 --visualize 1      # watch experiment 1, run the other 9 headless
  uv run evaluate_hand.py --visualize 1 --playback 0.5   # ... in slow motion
  uv run evaluate_hand.py --hand hands/team1/hand.xml --seed 0   # score another hand design

The robot moves with your kinematics from student_kinematics.py (Parts 1 and 2).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import mujoco.viewer

from rb5_sim.controller import RB5, Sim, ViewerClosed
from pipetting import add_common_args, kinematics_from_args
from rb5_sim.env import HolderError, PipetteSpec, build_model
from rb5_sim.forces import ForceMonitor
from rb5_sim.layout_md import load_layouts
from rb5_sim.reward import FORCE_TOLERANCE, TARGET_FORCE, score_experiment
from rb5_sim.task import Probe, run_pipetting
from rb5_sim.viewer_plots import GraphWindow, ViewerOverlay


def run_experiment(layout, error: HolderError, pipette: PipetteSpec, opts):
    model = build_model(layout, error, pipette, hand=opts.hand)
    data = mujoco.MjData(model)
    forces = ForceMonitor(model)
    sim = Sim(model, data)
    sim.callbacks.append(forces)
    robot = RB5(sim, time_scale=opts.motion_scale, kinematics=opts.kin)
    robot.reset()
    report = run_pipetting(robot, layout, Probe(model, data, forces), log=lambda *_: None)
    return report, forces, score_experiment(report, forces)


def run_experiment_visual(layout, error: HolderError, pipette: PipetteSpec, opts, title: str):
    """Same as run_experiment, but in the viewer (real time). Returns None if the
    viewer was closed before the task finished."""
    graph = opts.graph
    model = build_model(layout, error, pipette, hand=opts.hand)
    data = mujoco.MjData(model)
    forces = ForceMonitor(model)
    graph_window = GraphWindow(model, forces) if graph == "window" else None
    print(f"   [viewer] {title}")
    with mujoco.viewer.launch_passive(model, data) as v:
        sim = Sim(model, data, viewer=v, playback=opts.playback)
        sim.callbacks.append(forces)
        sim.callbacks.append(ViewerOverlay(v, model, forces, plots=graph == "overlay"))
        if graph_window:
            sim.callbacks.append(graph_window)
        robot = RB5(sim, time_scale=opts.motion_scale, kinematics=opts.kin)
        robot.reset()
        v.sync()
        try:
            report = run_pipetting(robot, layout, Probe(model, data, forces), log=lambda *_: None)
        except ViewerClosed:
            return None
        forces.phase = "finished"
        print("   [viewer] task finished - close the viewer window to continue")
        while v.is_running():
            sim.step()
    return report, forces, score_experiment(report, forces)


def plot_traces(results, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, event in zip(axes, ("aspirate hold", "dispense hold")):
        for i, (pipette, _, forces, score) in enumerate(results):
            ev = np.array(forces.events)
            idx = np.flatnonzero(ev == event)
            if not len(idx):
                continue
            t = np.array(forces.time)[idx]
            f = np.array(forces.series["bulb_force"])[idx]
            ax.plot(t - t[0], f, lw=1.6,
                    label=f"#{i + 1}: bulb {pipette.bulb_radius * 2e3:.1f} mm, R={score.reward:.2f}")
        ax.axhspan(TARGET_FORCE - FORCE_TOLERANCE, TARGET_FORCE + FORCE_TOLERANCE, color="tab:green", alpha=0.12)
        ax.axhline(TARGET_FORCE, color="tab:green", ls="--", lw=1)
        ax.set_title(event)
        ax.set_xlabel("time in hold [s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("bulb squeeze force per side [N]")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Bulb force while held squeezed (target {TARGET_FORCE} N, band = ±{FORCE_TOLERANCE} N)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--experiments", type=int, default=5, help="number of experiments")
    parser.add_argument("--seed", type=int, default=None, help="random seed (same seed = same pipettes and errors)")
    parser.add_argument("--layouts", type=Path, default=Path("layouts.md"), help="Markdown file with layouts")
    parser.add_argument("--layout", default=None, help="use only this layout (name or index); default: cycle")
    parser.add_argument("--no-error", action="store_true", help="no holder pose error")
    parser.add_argument("--plot", type=Path, help="save the bulb force traces of all experiments")
    parser.add_argument("--visualize", type=int, default=0, metavar="K",
                        help="show the first K experiments in the viewer (real time); the rest run headless")
    parser.add_argument("--graph", choices=("window", "overlay", "none"), default="window",
                        help="live graphs for visualized experiments (default: separate window)")
    add_common_args(parser)
    args = parser.parse_args()
    args.kin = kinematics_from_args(args)

    layouts = load_layouts(args.layouts)
    names = list(layouts)
    if args.layout is not None:
        key = names[int(args.layout)] if args.layout.lstrip("-").isdigit() else args.layout
        if key not in layouts:
            raise SystemExit(f"Unknown layout '{args.layout}'. Available: {', '.join(names)}")
        names = [key]

    rng = np.random.default_rng(args.seed)
    print(f"Hand: {args.hand}   kinematics: {args.kinematics}   seed: {args.seed}")
    ungraded = f" (--motion-scale {args.motion_scale:g}: not a graded score)" if args.motion_scale != 1 else ""
    if ungraded:
        print(f"WARNING: every robot move is {args.motion_scale:g}x slower than in grading{ungraded}")
    print(f"Target bulb force {TARGET_FORCE} N per side, tolerance {FORCE_TOLERANCE} N; "
          f"shaft dia {PipetteSpec.SHAFT_RANGE[0] * 2e3:.0f}-{PipetteSpec.SHAFT_RANGE[1] * 2e3:.0f} mm, "
          f"bulb dia {PipetteSpec.BULB_RANGE[0] * 2e3:.0f}-{PipetteSpec.BULB_RANGE[1] * 2e3:.0f} mm\n")
    header = (f"{'#':>2}  {'layout':<12} {'shaft':>5} {'bulb':>5}  {'task':<5} {'F mean':>6} {'F std':>5} "
              f"{'RMS err':>7} {'comp':>5} {'insert':>6}  {'reward':>6}\n"
              f"{'':>2}  {'':<12} {'[mm]':>5} {'[mm]':>5}  {'':<5} {'[N]':>6} {'[N]':>5} {'[N]':>7} {'[mm]':>5} "
              f"{'[N]':>6}")
    print(header)
    print("-" * len(header.splitlines()[0]))

    results = []
    for i in range(args.experiments):
        name = names[i % len(names)]
        pipette = PipetteSpec.sample(rng)
        error = HolderError() if args.no_error else HolderError.sample(rng)
        result = None
        if i < args.visualize:
            result = run_experiment_visual(layouts[name], error, pipette, args,
                                           title=f"experiment {i + 1}: {name}, {pipette}")
            if result is None:
                print("   [viewer] closed early - rerunning this experiment without the viewer")
        if result is None:
            result = run_experiment(layouts[name], error, pipette, args)
        report, forces, s = result
        results.append((pipette, report, forces, s))
        flat = " flat!" if s.bulb_bottomed_out else ""
        print(f"{i + 1:>2}  {name[:12]:<12} {pipette.shaft_radius * 2e3:5.1f} {pipette.bulb_radius * 2e3:5.1f}  "
              f"{'ok' if s.task_passed else 'FAIL':<5} {s.mean_force:6.2f} {s.force_std:5.2f} {s.force_rms_error:7.2f} "
              f"{s.max_compression * 1e3:5.1f} {s.insert_force:6.1f}  {s.reward:6.3f}{flat}")
        if not s.task_passed:
            for c in report.checks:
                if not c.passed and c.kind == "task":
                    print(f"      failed: {c.name}: {c.detail}")

    rewards = np.array([r[3].reward for r in results])
    means = np.array([r[3].mean_force for r in results])
    print("-" * len(header.splitlines()[0]))
    print(f"Hand score (mean reward): {rewards.mean():.3f}   min {rewards.min():.3f}   "
          f"task success {sum(r[3].task_passed for r in results)}/{len(results)}{ungraded}")
    if np.isfinite(means).any():
        print(f"Mean squeeze force across experiments: {np.nanmean(means):.2f} N, "
              f"spread {np.nanmax(means) - np.nanmin(means):.2f} N (max - min)")

    if args.plot:
        plot_traces(results, args.plot)
        print(f"Force traces saved to {args.plot}")
    if args.visualize and args.graph == "window":
        print("Close any open graph windows to exit.")


if __name__ == "__main__":
    main()
