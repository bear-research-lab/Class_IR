# RB5 Pipetting Project

*[한국어](README.ko.md)*

A MuJoCo simulation of a Rainbow Robotics RB5 cobot doing a pipetting task. The project
has three parts; each one builds on the one before:

| Part | You write | Handout | Test |
|---|---|---|---|
| 1. Forward kinematics | `forward_kinematics()` in `student_kinematics.py` | [PART1_FK.md](PART1_FK.md) | `uv run part1_fk.py` |
| 2. Inverse kinematics | `inverse_kinematics()` in `student_kinematics.py` | [PART2_IK.md](PART2_IK.md) | `uv run part2_ik.py --task` |
| 3. Pipetting hand | `hands/<your-team>/hand.xml` | [PART3_HAND.md](PART3_HAND.md) | `uv run evaluate_hand.py` |

The robot moves with your kinematics: every move calls your `forward_kinematics()` and
`inverse_kinematics()`. So the pipetting task (Part 3) runs only once Parts 1 and 2 work.

## 1. Install

You only need **uv**, which installs the right Python and all packages for you.

- **Windows** (PowerShell):
  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux**:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Open a new terminal, go to this folder, and run:

```
uv sync
```

The first time takes a minute or two (it downloads Python 3.12, MuJoCo, NumPy and matplotlib).

## 2. Run it

Parts 1 and 2 (see the handouts):

```
uv run part1_fk.py trajectories/joints_example.txt   # play joint angles; your FK is drawn on the hand
uv run part1_fk.py                                   # Part 1 test
uv run part2_ik.py trajectories/poses_example.txt    # move the hand through poses with your IK
uv run part2_ik.py --task                            # Part 2 test, then the task with the basic hand
```

Part 3, once your IK works:

```
uv run pipetting.py --no-error     # the basic hand, holder exactly where the robot expects it
uv run pipetting.py --seed 3       # a holder error where the basic hand drops the pipette
uv run pipetting.py --seed 6       # ... and one where it jams the pipette when putting it back
uv run evaluate_hand.py --seed 0   # score the hand on 5 random pipettes
```

Without `--seed` the holder error is random every run. Small errors sometimes work even
with the basic hand; larger ones don't. `pipetting.py` always uses the standard pipette
(8 mm shaft, 18 mm bulb), while `evaluate_hand.py` also randomizes the pipette size, so the
same `--seed` number gives different experiments in the two scripts.

`pipetting.py` opens a 3D viewer and a second window with live graphs of the bulb force and
the pipette tip height. When the task is done, close both windows to exit.

> **macOS:** the MuJoCo viewer must be started with `mjpython`:
> `uv run mjpython pipetting.py`. Commands that don't open the viewer (`--headless`,
> `evaluate_hand.py` without `--visualize`) work with plain `uv run`.

**Viewer controls:** left-drag to rotate, right-drag to pan, scroll to zoom. Double-click a
part to select it.

## 3. Work on your hand

```
hands/basic/hand.xml        the basic hand: copy it, don't edit it
hands/<your-team>/hand.xml  your design
```

Test your hand with `--hand`:

```
uv run pipetting.py --hand hands/<your-team>/hand.xml --playback 0.5
uv run evaluate_hand.py --hand hands/<your-team>/hand.xml --seed 0
```

## 4. Commands

| Command | What it does |
|---|---|
| `uv run part1_fk.py` | Part 1 test: your FK against the simulator |
| `uv run part1_fk.py FILE` | Play a joint file; `--out poses.txt` saves your FK poses, `--headless` without viewer |
| `uv run part2_ik.py` | Part 2 test: your IK on random and task targets |
| `... --task` | ... then the pipetting task with the basic hand and your IK |
| `uv run part2_ik.py FILE` | Play a pose file; `--joint-space`, `--out joints.txt`, `--headless` |
| `uv run pipetting.py` | Run the task once in the viewer |
| `... --no-error` | Put the holder exactly at its nominal pose |
| `... --seed N` | Reproducible holder error (same N = same error) |
| `... --playback 0.5` | Slow motion in the viewer (same physics) |
| `... --headless` | No window; prints the checks and a force summary |
| `... --headless --plot run.png --csv run.csv` | Save a force plot and the force log |
| `... --layout "Practice 2"` | Use another layout from `layouts.md` |
| `... --graph overlay` / `none` | Graphs inside the viewer, or no graphs |
| `uv run evaluate_hand.py --seed 0` | Score the hand: 5 experiments with random pipette sizes and holder errors |
| `... -n 10 --visualize 1` | 10 experiments, watch the first one |
| `... --plot forces.png` | Save the bulb force of every experiment |

Every command accepts `--hand path/to/hand.xml`. Run any script with `--help` for all options.

## 5. Files

| Path | Contents |
|---|---|
| `PART1_FK.md`, `PART2_IK.md`, `PART3_HAND.md` | The three parts: what to do, rules, scoring, what to submit |
| `student_kinematics.py` | Your FK and IK (Parts 1 and 2) |
| `trajectories/` | Example joint and pose files |
| `hands/basic/` | The basic hand to start from (Part 3) |
| `layouts.md` | Object positions (pipette holder, water box, dispense box) |
| `part1_fk.py`, `part2_ik.py` | Run and test Parts 1 and 2 |
| `pipetting.py`, `evaluate_hand.py` | Run the task and score the hand (Part 3) |
| `rb5_sim/` | Simulator code: scene, robot controller, task, force monitor, reward. Read it, don't change it |
| `assets/` | RB5 robot model and the task scene |

`layouts.md` holds the positions as Markdown tables (x and y in mm, in the robot base frame).
At the final presentation you will get new positions in the same format:
`uv run evaluate_hand.py --layouts final.md`.

## 6. Troubleshooting

- **`uv` is not found:** open a new terminal after installing it.
- **No window appears** (remote desktop, WSL, SSH): add `--headless`.
- **The graph window doesn't open:** use `--graph overlay` to draw the graphs in the viewer.
- **`... is missing: site 'grasp' ...`:** your hand file is missing part of the required
  interface; see section 4 of `PART3_HAND.md`.
- **`... implement inverse_kinematics() ...`:** the task needs your IK; finish Parts 1 and 2.

## Credits

- Robot model: Rainbow Robotics RB5-850e, converted from the URDF and meshes in
  [RainbowRobotics/rbpodo_ros2](https://github.com/RainbowRobotics/rbpodo_ros2)
  (Apache License 2.0, see `assets/rb5_850e/LICENSE`).
- Physics: [MuJoCo](https://mujoco.org).
