# Chapter 9 (Trajectory Generation): Python demos

Six short plotting scripts (01–06) and five robot animations (A1–A5) that show how the choices in Chapter 9 of *Modern Robotics* change a robot's motion.
All of them use `traj_utils.py`, which implements every formula from the chapter with NumPy only, so you can read each line.

## Setup

```bash
pip install numpy matplotlib
pip install modern_robotics   # optional: 04 checks its results against the official library
```

Run any script from this folder, e.g. `python 01_time_scalings.py`.
Each script prints key numbers and opens a figure. Set `SAVE = True` near the top to also save a PNG.
Scripts 03 and 04 have `ANIMATE = True/False` for an animation.

## The scripts

| Script | MR section | What it shows |
|---|---|---|
| `01_time_scalings.py` | 9.2.2 | Cubic, quintic, trapezoidal and S-curve s(t) with the same T: s, ṡ, s̈ and jerk side by side |
| `02_min_time_6R.py` | 9.2.2 | A 6-joint robot on a joint-space straight line: shortest T for each time scaling under joint speed/acceleration limits, and which joint limits it |
| `03_joint_vs_task_line.py` | 9.2.1 | 2R arm: a task-space straight line can violate joint limits (case A) or need huge joint speeds near a singularity (case B) |
| `04_screw_vs_decoupled.py` | 9.2.1 | Screw path vs decoupled path in SE(3): same rotation, different origin path |
| `05_via_points.py` | 9.3 | Piecewise-cubic via-point interpolation (data from MR Figs. 9.7–9.8); via velocities change the shape; acceleration jumps at vias |
| `06_time_optimal_2R.py` | 9.4 | Vertical 2R arm with torque limits: velocity limit curve, bang-bang time scaling, comparison with the fastest feasible cubic |

## Robot animations (A1–A5)

These show how the robot actually moves. Each opens an animation window; set `GIF = "name.gif"` near the top to save it instead (handy for slides). Ready-made GIFs are in the `gifs/` folder.

**Joint-space line vs screw vs decoupled, in one line each:** the joint-space line needs no IK and is gentle on the joints but does not control the gripper path; the screw path is one constant screw motion but its origin bulges with large rotations; the decoupled path keeps the gripper exactly on the straight line but needs IK, suffers near singularities, and can leave the joint limits between two feasible end configurations. A6 shows each of these.

Each animation has a `SLOWMO` setting near the top that slows down the playback only (the robot's times and speeds stay the same). A1 uses `SLOWMO = 4` (one second of motion takes four seconds on screen); the others default to real time (`SLOWMO = 1`).

| Script | What you see |
|---|---|
| `A1_anim_time_scalings.py` | Four 2R arms race along the same path with cubic, quintic, trapezoidal and S-curve time scaling. Dots dropped every 0.1 s show where each arm is fast or slow; a cursor moves along the ṡ(t) plot |
| `A2_anim_joint_vs_task.py` | Joint-space line vs task-space line, side by side, with both motions in joint space. `CASE = "A"`: the arm turns red while it violates a joint limit. `CASE = "B"`: the task-space arm whips around near the singularity (live joint speed) |
| `A3_anim_screw_vs_decoupled_UR5.py` | A UR5 (6R, PoE model of MR Example 4.5) moves its gripper between the same two poses three ways: joint-space line, screw path, decoupled path. Same orientation change on the last two; only the gripper's position path differs. Also shows the instantaneous screw axis (fixed for the screw path, sliding for the decoupled path) with its pitch, and the six joint angles vs time. Joint angles from Newton–Raphson IK |
| `A6_anim_three_methods_scenarios.py` | Joint-space line vs screw vs decoupled on a UR5 in three scenarios that each expose a strength or weakness: `SCENARIO = 1` corridor (joint-space line swings 0.19 m off the line into a box), `2` singularity (decoupled path needs 394 deg/s vs 94 deg/s for the joint-space line), `3` big rotation (screw path bulges 0.27 m and hits a box), `4` joint limits (both end configurations are within example joint limits, but the decoupled path needs the elbow at 158° > 150° and the shoulder at −130° < −120° on the way; the joint-space line cannot leave the limits because the box of limits is convex). Prints a comparison table and shows joint speeds |
| `A4_anim_via_points.py` | A 2R arm's gripper passes through four timed via points with given vs heuristic via velocities; a red arrow shows the gripper acceleration jumping at the vias |
| `A5_anim_time_optimal_2R.py` | Vertical 2R arm: time-optimal vs fastest cubic on the same path and torque limits, with live torque bars (red = saturated) |

Extra modules used by the animations: `robots.py` (UR5 forward kinematics, Jacobian, numerical IK), `anim_utils.py` (drawing helpers), `time_optimal_2r.py` (the computation of script 06 as functions).

## What `traj_utils.py` contains

* Time scalings: `cubic_time_scaling`, `quintic_time_scaling`, `trapezoidal_time_scaling`, `triangular_time_scaling`, `scurve_time_scaling`
* SE(3) tools from Chapter 3: `exp3`, `log3`, `exp6`, `log6`, `trans_inv`, `rot`, `RpToTrans`
* Paths: `joint_line`, `screw_path`, `decoupled_path`
* Via points: `cubic_via_coeffs`, `cubic_via`, `heuristic_via_velocities`
* `Planar2R`: forward/inverse kinematics and dynamics (M, c, g) of a planar 2R arm with point masses

`screw_path` and `decoupled_path` give the same results as `mr.ScrewTrajectory` and `mr.CartesianTrajectory` (script 04 checks this when the library is installed).

## Suggested in-class questions

1. (01) Which profile has the highest peak speed for the same T? Which ones have infinite jerk, and where?
2. (02) Under acceleration limits, why can the quintic be slightly *faster* than the cubic?
3. (03) Move the case B line away from the base (`Y_OFFSET`). How does the peak joint speed change?
4. (04) Set the rotation to 0°. Are the screw and decoupled paths different? Why?
5. (05) Change the via velocities. Which choice gives the smoothest path?
6. (06) Lower the joint 2 torque limit from 12 to 10 N·m. Why does the cubic slow down so much more than the time-optimal motion?

Each script's header lists more "Try this" exercises.

## Notes

* Script 06 implements the single-switch (bang-bang) case. If the forward curve hits the velocity limit curve first, it says so; the full multi-switch algorithm is in MR Sec. 9.4.2.
* Script 06 also checks the "static posture maintenance" assumption (MR Sec. 9.4.4): the arm must be able to hold every configuration on the path against gravity.
* Joint limits and robot parameters are example values, not the specifications of any particular robot.
* Animations: when running inside Jupyter, use an interactive backend (e.g. `%matplotlib widget`) to see them.
