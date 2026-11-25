# Chapter 7 (Inverse Kinematics): Python demos

Five numbered demos (1–5) and three helper modules for the inverse kinematics lecture.
The demos go from numerical IK on a 1-DOF example, through the analytic IK of a PUMA-type arm and the UR5, to what happens at a wrist singularity.
Everything uses NumPy and Matplotlib only, so you can read every line.

## Setup

```bash
pip install numpy matplotlib pillow   # pillow only for saving GIFs
```

or, with `uv`, run `uv sync` once in the parent `Class/` folder.

Run each script from this folder. The demo file names contain spaces, so put them in quotes:

```bash
python "1. Newton_raphson.py"
uv run "3. ur5_ik_visual.py"
```

## The demos

| Script | What it shows |
|---|---|
| `1. Newton_raphson.py` | Newton–Raphson iteration for a 1-DOF IK problem x_d = f(θ) (MR Fig. 6.7), one step at a time (press a key or click to take the next step). Choose a preset with `PROBLEM=...`: `gauss` (the Fig. 6.7 curve), `easy` (θ², converges in ~3 steps), `zigzag`/`hard` (atan θ, overshoots back and forth) and `singular` (cos θ at full extension: df/dθ → 0, so convergence drops from quadratic to linear). Set `SAVE_GIF = True` to save `newton_raphson.gif` |
| `2. Puma_style_IK.py` | PUMA-type arm with a spherical wrist (Pieper's condition, case 1). Three linked views with sliders: the 3D configuration, the top view (θ1 = atan2(p_wy, p_wx)) and the arm plane (θ2, θ3 from the law of cosines) |
| `3. ur5_ik_visual.py` | Step-by-step visual walkthrough of the UR5 analytic IK (case 2: three parallel axes): zero pose and screw axes → θ1 (top view) → θ5 → θ6 → θ2, θ3, θ4 (planar 3R arm) → all 8 solutions as a tree. Use Next/Prev (or ←/→) to move through the steps and Shoulder/Wrist/Elbow (or 1/2/3) to switch branches |
| `4. ur5_ik_interactive.py` | Sliders for x, y, z, roll, pitch, yaw; the UR5 follows the target pose. Choose one of the 8 solutions or `auto` (closest to the current joints), and optionally draw all 8. The docstring describes a singularity experiment where the wrist joints jump |
| `5. ur5_singularity_demo.py` | The end-effector moves along a straight line at 0.1 m/s with a fixed orientation (MoveL) past the wrist singularity (sin θ5 = 0) at three tilt angles. Plots joint speeds, \|sin θ5\| and σ_min(J), shows how far the path is missed when joint speeds are limited, and animates the closest pass. Saves `ur5_singularity_summary.png` and `ur5_singularity.gif` |

### Options for `3. ur5_ik_visual.py`

```bash
uv run "3. ur5_ik_visual.py"                                  # interactive window
uv run "3. ur5_ik_visual.py" --target 30 -60 80 -110 50 20    # target = FK of these angles [deg]
uv run "3. ur5_ik_visual.py" --save steps                     # one PNG per step into ./steps
uv run "3. ur5_ik_visual.py" --save figs --slides             # figures only (no text), for slides
```

## Helper modules (UR5 analytic IK)

All three find θ1 → θ5 → θ6 → (θ2, θ3, θ4) and return up to 8 solutions (shoulder × wrist × elbow).
Running one on its own solves an example pose and runs a 1000-case random round-trip test (FK(IK(T)) = T).

| Module | Model | Used by |
|---|---|---|
| `ur5_ik_explained.py` | PoE built from the zero pose, with each step as its own function and printed explanations. `--plot` draws the 8 configurations; `--test N` runs N random round trips | `3. ur5_ik_visual.py` |
| `ur5_ik_poe.py` | Compact PoE version (Lynch & Park's UR5 zero configuration) | `4.` and `5.` |
| `ur5_ik.py` | Standard DH parameters, as published by Universal Robots | — (for comparison with the DH convention) |

Keep the helper modules in this folder: the demos import them from here.

## Ready-made outputs

`ur5_singularity.gif` and `ur5_singularity_summary.png` are the outputs of `5. ur5_singularity_demo.py`, ready for slides.
