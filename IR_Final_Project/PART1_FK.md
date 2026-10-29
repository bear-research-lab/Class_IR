# Part 1: Forward Kinematics

*[한국어](PART1_FK.ko.md)*

The RB5 is commanded with 6 joint angles. Forward kinematics (FK) answers: for these joint
angles, where is the hand, and which way does it point? In this part you write the FK of the
RB5 and check it against the simulator by playing joint-angle files.

## 1. What to implement

Fill in `forward_kinematics()` in **`student_kinematics.py`**:

```python
def forward_kinematics(q, tool=np.eye(4)) -> T   # 4x4 pose of the tool frame in the base frame
```

| Argument | Meaning |
|---|---|
| `q` | (6,) joint angles [rad]: base, shoulder, elbow, wrist1, wrist2, wrist3 |
| `tool` | (4, 4) pose of a tool frame in the flange frame. Identity = the flange itself. The robot passes the hand's `grasp` site here (for the basic hand: 90 mm along the flange z axis) |
| returns `T` | (4, 4) homogeneous transform T_base_tool = T_base_flange(q) · tool |

Use NumPy only. Don't import `mujoco` or `rb5_sim` in this file (the scripts check this).

## 2. The robot

The geometry comes from `assets/rb5_850e/rb5_850e.xml`. Each link frame is the parent frame
moved by an offset, then rotated by its joint about the joint axis (in the new frame).
All link frames are parallel when q = 0.

| Joint | Offset from the previous frame [m] | Axis |
|---|---|---|
| 1 base | (0, 0, 0.1692) | z |
| 2 shoulder | (0, 0, 0) | y |
| 3 elbow | (0, 0, 0.425) | y |
| 4 wrist1 | (0, 0, 0.392) | y |
| 5 wrist2 | (0, −0.1107, 0.1107) | z |
| 6 wrist3 | (0, 0, 0) | y |
| flange | (0, −0.0967, 0), then rotated +90° about x | - |

The flange frame has z pointing out of the flange. At q = 0 the arm points straight up.

You can chain these transforms directly or derive a Denavit–Hartenberg table from them;
either is fine. Explain your choice in the report. Some values to check against:

| q [deg] | flange position [m] | flange rotation |
|---|---|---|
| (0, 0, 0, 0, 0, 0) | (0, −0.2074, 1.0969) | [[1, 0, 0], [0, 0, −1], [0, 1, 0]] |
| (90, 0, 90, 0, 0, 0) | (0.2074, 0.5027, 0.5942) | [[0, 0, 1], [0, 1, 0], [−1, 0, 0]] |

## 3. Joint files

A joint file is a time series of joint angles, one waypoint per line, as `key:value` pairs:

```
# comment
T:0  theta1:31.27 theta2:-28.19 theta3:130.43 theta4:77.76 theta5:-58.73 theta6:90
T:2  theta1:0     theta2:-28.19 theta3:130.43 theta4:77.76 theta5:-58.73 theta6:90
```

- `T`: the time [s] at which the robot reaches the line; it must increase. The first line
  is where the robot starts.
- `theta1` … `theta6`: joint angles in **degrees**.

The robot moves smoothly between the lines. `trajectories/joints_example.txt` is an example.

## 4. Test it

```
uv run part1_fk.py                                          # graded test
uv run part1_fk.py trajectories/joints_example.txt          # play a joint file in the viewer
uv run part1_fk.py trajectories/joints_example.txt --out my_poses.txt   # save your FK poses
```

- **Graded test** (no file): your FK is compared with the simulator at 100 random joint
  angles each for the flange, the basic hand's grasp site, and random tool frames. Each case
  passes when the position is within 0.001 mm and the rotation within 1e-6 rad.
- **Playing a file**: the viewer draws the grasp-site frame from **your** FK (long axes:
  x red, y green, z blue) and the trail it leaves. If your FK is right, the frame sits on the
  hand between the fingers. At every waypoint the script prints your FK position next to the
  simulator's and the error.
- `--out` saves your FK result as a pose file: the input format of Part 2.

## 5. Submit

- `student_kinematics.py`
- The output of `uv run part1_fk.py`
- Your own joint file with at least 5 waypoints, and the output of playing it with `--out`
- In the report: your kinematic model (transform chain or DH table) with a figure of the
  frames
