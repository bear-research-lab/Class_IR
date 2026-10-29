# Part 2: Inverse Kinematics

*[한국어](PART2_IK.ko.md)*

The pipetting program tells the robot where the hand should be, not the joint angles.
Inverse kinematics (IK) turns a pose of the hand's `grasp` site into the 6 joint angles.
In this part you write the IK, using your FK from Part 1, and finish by running the
pipetting task with the basic hand.

## 1. What to implement

Fill in `inverse_kinematics()` in **`student_kinematics.py`**:

```python
def inverse_kinematics(target, q_init, tool=np.eye(4), max_iters=200, tol=1e-6) -> (q, err)
```

| Argument | Meaning |
|---|---|
| `target` | (4, 4) target pose of the tool frame in the base frame |
| `q_init` | (6,) starting joint angles [rad]. Start here and stay on the same IK branch |
| `tool` | (4, 4) tool frame in the flange frame, as in Part 1 |
| `max_iters`, `tol` | stop after `max_iters` iterations, or when the error norm is below `tol` |
| returns `q` | (6,) joint angles inside `[Q_LO, Q_HI]` |
| returns `err` | norm of the 6D pose error at `q` |

The file gives you `pose_error(target, current)`: the 6D error [Δp; rotation vector] from
the current pose to the target. You write the rest, including a function for the 6×6
Jacobian of the tool frame (geometric, from the joint axes and positions of your FK, or
numerical by finite differences). A good starting point is damped least squares:

    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e

repeated until the error is small. Limit the size of each step and keep `q` inside the joint
limits. NumPy only, as in Part 1.

## 2. Pose files

A pose file gives the grasp point and orientation over time:

```
T:0  x:400 y:0    z:350
T:2  x:450 y:-100 z:350
T:4  x:450 y:-100 z:300 alpha:30 beta:90 gamma:180
```

- `T`: time [s], as in Part 1.
- `x`, `y`, `z`: grasp point in **mm** in the robot base frame.
- `alpha`, `beta`, `gamma` (optional): orientation as ZYZ Euler angles in **degrees**,
  R = Rz(α) Ry(β) Rz(γ). Without them the line uses (0, 90, 180): the task orientation,
  hand pointing forward (+x) with the pipette axis vertical.

`trajectories/poses_example.txt` is an example. `part1_fk.py --out` writes this format, and
`part2_ik.py --out` writes a joint file, so you can check one part against the other.

## 3. Test it

```
uv run part2_ik.py                                          # graded test
uv run part2_ik.py --task                                   # ... then the pipetting task (viewer)
uv run part2_ik.py --task --headless                        # ... without the viewer
uv run part2_ik.py trajectories/poses_example.txt           # play a pose file in the viewer
uv run part2_ik.py trajectories/poses_example.txt --joint-space
uv run part2_ik.py trajectories/poses_example.txt --out my_joints.txt
```

- **Test 1 (random):** 50 targets made from random joint angles near the home pose. Every
  solve starts from the home pose.
- **Test 2 (task):** the grasp-site waypoints of the pipetting task for every layout in
  `layouts.md`, solved in order, each starting from the previous solution.
- A solve passes when the 6D error is below 1e-4 (0.1 mm / 0.006°) and every joint is inside
  its limits. The script measures the error with the simulator, not with your FK or the
  `err` you return.
- **`--task`:** the full pipetting task with the **basic hand** and no holder error. Every
  robot move uses your IK. Straight-line moves call it at every simulation step (every 2 ms)
  with **only 10 iterations**, so it must converge fast from a nearby start, not only
  eventually. The basic hand passes every check here when your IK is right.
- **Playing a file:** the hand moves through the poses in straight lines (or with
  `--joint-space`, one IK solve per waypoint and joint interpolation). The viewer shows the
  targets (short axes) and your FK frame (long axes). The script prints the joint angles and
  the reached error at every waypoint.

## 4. Submit

- `student_kinematics.py`
- The output of `uv run part2_ik.py --task --headless`
- Your own pose file with at least 5 waypoints, and the output of playing it
- In the report: your Jacobian and update rule, the values of λ, step limit and tolerance,
  and what happens when you change them (for example near a singular pose, or with too few
  iterations)
