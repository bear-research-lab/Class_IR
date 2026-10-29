# Part 3: Pipetting Hand

*[한국어](PART3_HAND.ko.md)*

Design a hand for the RB5 that picks up a pipette from its holder, aspirates from the
water box, dispenses into the dispense box, and puts the pipette back in its holder.
The robot is told only the **nominal**
positions. The hand itself has to absorb the holder pose error (±5 mm in x and y, ±2° tilt)
and the variation in pipette thickness, and it has to squeeze the bulb with a **consistent force**.

You change only the hand. The robot program, the task sequence and the scoring are fixed
and the same for everyone. The robot moves with **your kinematics** from Parts 1 and 2
(`student_kinematics.py`), so finish those first: until then the scripts stop and say so.

## 1. Setup

```
uv sync                                   # install dependencies (first time only)
uv run pipetting.py                       # watch the basic hand try the task
uv run pipetting.py --no-error            # ... with the holder exactly where the robot expects it
```

Make your own copy of the basic hand (`hands/basic/hand.xml`) and work on that copy:

```
hands/<your-team>/hand.xml                # copy of hands/basic/hand.xml
hands/<your-team>/meshes/...              # optional CAD meshes (STL/OBJ)
```

## 2. What the robot does (fixed)

For every experiment the robot runs the same program. It uses only the nominal positions
and never looks at where the pipette really is.

| Step | Robot action | Time given to your hand |
|---|---|---|
| 1 | Approach the holder horizontally, close the gripper, lift 150 mm | gripper close: 0.8 s |
| 2 | Move above the water box, **squeeze**, dip the tip to 25 mm below the surface, **release** | squeeze/release: 0.6 s each |
| 3 | Move above the dispense box and lower the tip into it | - |
| 4 | **Squeeze**, hold 0.5 s, **release**, lift | squeeze/release: 0.6 s each |
| 5 | Move above the **nominal** holder, lower the pipette straight down into the socket (slowly for the last 15 mm), **open**, back out horizontally | gripper open: 0.6 s |

The robot moves the hand's `grasp` site to where the pipette's grasp point **should** be,
which is 110 mm above the tip on the nominal pipette axis.

## 3. The pipette

The pipette is vertical in its holder, and its frame origin is at the tip.

| Part | Position along the axis (from the tip) | Size |
|---|---|---|
| Shaft | 0 - 160 mm | **diameter 6 - 10 mm** (random per experiment) |
| Grasp point (where the robot expects your fingers) | 110 mm | - |
| Bulb center | 180 mm (**70 mm above the grasp point**) | **diameter 16 - 22 mm** (random per experiment), about 26 mm long |

The holder socket is square, fits the shaft with 1 mm clearance, is 40 mm deep, and has only a
1 mm × 45° chamfer at the top. The holder itself is placed with up to ±5 mm (x, y) and 2° tilt
of error, and it is in the same (wrong) place when you return the pipette, so the return is a
peg-in-hole insertion that your hand has to make work. Friction between the glass shaft and
the holder is 0.3.

In the simulation the pipette cannot spin about its own axis (a real bulb is round, so this
spin does not matter). The bulb's four walls stay aligned with the pipette frame, so squeeze
it along the hand's y or z axis, not diagonally.

The bulb is compliant: four rubber walls on springs of **600 N/m**, which flatten after **6 mm**
of compression. The squeeze force per side is 600 N/m × compression.

## 4. Interface your hand must provide

Your `hand.xml` is attached to the RB5 flange. It must contain:

1. **Exactly one root body** in `<worldbody>` (the hand), whose frame is the flange:
   - **z**: out of the flange (the approach direction; the hand moves toward the pipette along +z)
   - **x**: along the pipette axis, pointing from the tip toward the bulb (up during the task)
   - **y**: completes the right-handed frame (horizontal during the task)
2. **A site named `grasp`** on a body rigidly fixed to the hand (not on a moving finger). The
   robot places this site at the nominal grasp point with the site's axes aligned to the hand
   axes above. In the basic hand it is at `pos="0 0 0.09"`, 90 mm in front of the flange.
3. **An actuator named `gripper`** that grasps the shaft, and **an actuator named `squeezer`**
   that squeezes the bulb. Any actuator type is allowed (`position`, `motor`, `general`, ...).
4. **Four numerics** giving the `ctrl` value the robot sends for each command. Their meaning
   depends on your actuators: a travel in meters for a position actuator, a force in N for a
   motor, and so on.

```xml
<custom>
  <numeric name="gripper_open"    data="0"/>
  <numeric name="gripper_close"   data="0.03"/>
  <numeric name="squeeze_release" data="0"/>
  <numeric name="squeeze_press"   data="0.0275"/>
</custom>
```

If something is missing, the scripts stop and list it.

## 5. Rules

- Edit only files in `hands/<your-team>/`. Don't change `rb5_sim/`, the scripts, or `assets/`.
- **Two actuators only**: `gripper` and `squeezer`. Everything else must be passive: springs
  (`stiffness`), dampers, joint limits, equality constraints, tendons and contacts.
- **Force limits**: `gripper` at most **20 N**, `squeezer` at most **15 N** (set `forcerange`).
  This is the force at the joint: a `gear` or a tendon `coef` multiplies it.
- **Hand mass** at most **1 kg** in total.
- **Only the hand**: `hand.xml` may refer only to its own parts (no welds, contact pairs or
  excludes with the pipette or the scene).
- **Buildable**: every passive joint, spring and constraint must correspond to a real
  mechanism (for example a linear guide plus a compression spring, or a flexure), and your
  3D CAD design must show it. Explain each one in your presentation.

The scripts check the actuators, force limits, mass and "only the hand", and stop with a
message if a rule is broken. Scores count only with the default `--motion-scale 1`.

## 6. How you are scored

```
uv run evaluate_hand.py --hand hands/<your-team>/hand.xml --seed 0
```

This runs 5 experiments. Each one draws a random pipette (shaft and bulb diameter), a random
holder error, and a layout from `layouts.md`. Each experiment gets a reward from 0 to 1:

```
reward = gate × exp(-(RMS force error / 0.5 N)²)
```

- **gate** is 1 only if every task check passes: pipette picked up, tip under water when the
  bulb is released, tip inside the dispense box, bulb released, no collision (no force above
  10 N from the environment on the pipette between pick-up and return), and the pipette
  **returned**: seated within 5 mm of the socket floor, with at most **15 N** of insertion
  force. Otherwise the reward is 0.
- **RMS force error** is measured against a **2.5 N per side** target, over every moment the
  bulb is held squeezed (during the dip and the dispense hold). Both the wrong level and a
  wobbly force cost points.

| Off target by | 0 N | 0.25 N | 0.5 N | 1 N |
|---|---|---|---|---|
| Reward | 1.00 | 0.78 | 0.37 | 0.02 |

**Your score is the mean reward** over seeds 0-4 (`-n 5` each, 25 experiments), run with the
default settings and shown in your presentation (section 8). Design for the whole range of
pipettes and holder errors, not for one seed.

The basic hand is a plain parallel gripper with flat fingers and a squeezer that always moves
to the same position. It passes every check when the holder is exactly where the robot
expects it (`--no-error`), but it scores **0** on `--seed 0`: with the holder error it jams
the pipette when putting it back, sometimes drops or knocks it over, and squeezes thick and
thin bulbs with very different forces. Run it, watch where it fails, and start from there.

## 7. Useful commands

```
# watch your hand in slow motion, with live bulb force and tip-height graphs
uv run pipetting.py --hand hands/<team>/hand.xml --playback 0.5

# score 10 random pipettes, watching the first one in the viewer
uv run evaluate_hand.py --hand hands/<team>/hand.xml --seed 0 -n 10 --visualize 1

# robustness: several seeds, and save the bulb force traces
uv run evaluate_hand.py --hand hands/<team>/hand.xml --seed 1 -n 10 --plot forces.png

# one run without the viewer, with a full force plot and CSV log
uv run pipetting.py --hand hands/<team>/hand.xml --headless --plot run.png --csv run.csv
```

When a check fails, the printout says which one and why (for example tip position error in mm).
The force summary shows grip, bulb and environment forces for each task phase.

## 8. Final presentation

There is nothing to hand in. Each team presents, and for Part 3 shows the following. The
basic hand doing the task (no holder error) is in the
[reference video](https://youtu.be/VN9uyTWR938?t=38).

- **Videos** of your hand doing the task with a holder error, including at least one seed
  where the basic hand fails, recorded with
  `uv run pipetting.py --hand hands/<your-team>/hand.xml --seed 3 --headless --video part3.mp4`.
- Your design: how it absorbs the holder error (when grasping and when returning), how it
  handles different shaft diameters, and how it keeps the squeeze force consistent.
- The real mechanism behind each simulated joint, spring and constraint.
- Your **3D CAD design** of the hand (covered in a later lecture): its parts, which part
  implements each passive joint and spring, and how the dimensions in `hand.xml` match the CAD.
- Results: the `evaluate_hand.py` table for seeds 0-4 (`-n 5` each), your mean score, and one
  force-trace plot (`--plot`).
- What still fails or scores low, and why.
