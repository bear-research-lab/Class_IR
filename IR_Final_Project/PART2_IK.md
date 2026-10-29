# Part 2: Inverse Kinematics

*[한국어](PART2_IK.ko.md)*

The pipetting program tells the robot where the hand should be, not the joint angles.
Inverse kinematics (IK) turns a pose of the hand's `grasp` site into the 6 joint angles.
In this part you derive the Jacobian (and get it from MuJoCo too), write the IK with your FK
from Part 1, and finish by running the pipetting task with the basic hand.

## 1. What to implement

### 1.1 Your own Jacobian

Fill in `jacobian()` in **`student_kinematics.py`**:

```python
def jacobian(q, tool=np.eye(4)) -> J   # 6x6 geometric Jacobian of the tool frame
```

Rows 0-2 are the linear velocity of the tool frame origin, rows 3-5 the angular velocity,
both in the base frame; column i belongs to joint i. Derive it from the joint axes and joint
positions of your FK (geometric Jacobian), or compute it by finite differences of
`forward_kinematics()` (for the angular rows, use `rotation_vector()` of the change in
orientation). NumPy only.

### 1.2 Jacobian with MuJoCo

Fill in `mujoco_jacobian(model, data, q, site="grasp")` in **`student_mujoco.py`**: the same
6×6 Jacobian, from `mujoco.mj_jacSite`. Run `mj_kinematics` and then `mj_comPos` before it,
and keep only the 6 arm columns (`model.joint(name).dofadr`) of the full Jacobian. The test
compares both Jacobians with the simulator, so you can use either one to find mistakes in
the other.

### 1.3 Inverse kinematics

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
the current pose to the target. Use it with your `jacobian()` from 1.1 (the robot moves
with your NumPy code, not with MuJoCo). A good starting point is damped least squares:

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

- **Jacobian:** `jacobian()` and `mujoco_jacobian()` are compared with the simulator's
  Jacobian of the grasp site at 20 random joint angles. Each passes when every entry is within
  1e-4. A failure prints the largest error in each joint's column, so you see which joint is
  wrong.
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
  It also has to be fast: a full task makes tens of thousands of IK calls. A finite-difference
  Jacobian calls your FK 6 or more times per iteration, so the run can get several times
  slower than with a geometric Jacobian.
- **Playing a file:** the hand moves through the poses in straight lines (or with
  `--joint-space`, one IK solve per waypoint and joint interpolation). The viewer shows the
  targets (short axes) and your FK frame (long axes). The script prints the joint angles and
  the reached error at every waypoint.

## 4. Final presentation

There is nothing to hand in. Each team presents, and for Part 2 shows:

- **A video** of your own pose file (at least 5 waypoints, including some orientation
  changes), recorded with `uv run part2_ik.py my_poses.txt --headless --video part2.mp4`.
  Reference: [what it should look like](https://youtu.be/VN9uyTWR938?t=10) (that video plays the
  example file; show your own).
- **A video** of the pipetting task with the basic hand and your IK, recorded with
  `uv run part2_ik.py --task --headless --video task.mp4`.
- The result of the graded test (`uv run part2_ik.py`), including both Jacobians.
- How you derived your Jacobian, and how you got it from MuJoCo.
- Your Jacobian and update rule, the values of λ, step limit and tolerance, and what happens
  when you change them (for example near a singular pose, or with too few iterations).
